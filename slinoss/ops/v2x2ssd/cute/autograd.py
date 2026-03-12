"""Autograd wrapper for the CuTe ``v2x2ssd`` operator."""

from __future__ import annotations

from typing import cast

import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment import (
    chunk_increment_bwd_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import chunk_scan_bwd_cute
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import state_passing_bwd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    chunk_increment_cute,
    chunk_scan_cute,
    state_passing_cute,
)


def _as_dtype_contiguous(tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype and tensor.is_contiguous():
        return tensor
    return tensor.to(dtype=dtype).contiguous()


class _V2x2SSDCuTeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        U: torch.Tensor,
        M: torch.Tensor,
        K: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_states: torch.Tensor | None,
        B_prev: torch.Tensor | None,
        U_prev: torch.Tensor | None,
        chunk_size: int,
        compute_dtype: torch.dtype | None,
        output_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = compute_dtype
        ctx.has_initial_states = initial_states is not None
        ctx.has_prev = B_prev is not None

        U_d = U.detach()
        M_d = M.detach()
        K_d = K.detach()
        B_d = B.detach()
        C_d = C.detach()
        initial_states_d = (
            _as_dtype_contiguous(initial_states.detach(), dtype=torch.float32)
            if initial_states is not None
            else None
        )
        B_prev_d = B_prev.detach() if B_prev is not None else None
        U_prev_d = U_prev.detach() if U_prev is not None else None

        B_last = B_d[:, :, -1, :].to(dtype=output_dtype or U.dtype).contiguous()
        U_last = U_d[:, :, -1, :].to(dtype=output_dtype or U.dtype).contiguous()

        inc, m_chunk = chunk_increment_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            chunk_size=ctx.chunk_size,
            B_prev0=B_prev_d,
            U_prev0=U_prev_d,
            compute_dtype=compute_dtype,
        )
        chunk_starts, final_state = state_passing_cute(
            inc,
            m_chunk,
            initial_states=initial_states_d,
        )
        Y = chunk_scan_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            chunk_starts,
            B_prev=B_prev_d,
            U_prev=U_prev_d,
            output_dtype=output_dtype or U.dtype,
            chunk_size=ctx.chunk_size,
            compute_dtype=compute_dtype,
        )

        saved: list[torch.Tensor] = [
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            m_chunk,
            chunk_starts,
        ]
        if initial_states_d is not None:
            saved.append(initial_states_d)
        if B_prev_d is not None:
            saved.append(B_prev_d)
            saved.append(U_prev_d)  # type: ignore[arg-type]
        ctx.save_for_backward(*saved)
        return (
            Y,
            final_state.to(dtype=output_dtype or U.dtype).contiguous(),
            B_last,
            U_last,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        dY: torch.Tensor | None,
        d_final_state: torch.Tensor | None,
        dB_last: torch.Tensor | None,
        dU_last: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        idx = 0
        U = saved[idx]
        idx += 1
        M = saved[idx]
        idx += 1
        K = saved[idx]
        idx += 1
        B = saved[idx]
        idx += 1
        C = saved[idx]
        idx += 1
        m_chunk = saved[idx]
        idx += 1
        chunk_starts = saved[idx]
        idx += 1

        initial_states = None
        if ctx.has_initial_states:
            initial_states = saved[idx]
            idx += 1

        B_prev = None
        U_prev = None
        if ctx.has_prev:
            B_prev = saved[idx]
            U_prev = saved[idx + 1]

        batch_size, n_heads, _T, P = map(int, U.shape)
        D = int(B.shape[-1])
        rdtype = torch.float32

        if dY is not None:
            dY = dY if dY.is_contiguous() else dY.contiguous()
            (
                dU_scan,
                dM_scan,
                dK_scan,
                dB_scan,
                dC_scan,
                d_chunk_starts,
                dB_prev_scan,
                dU_prev_scan,
            ) = chunk_scan_bwd_cute(
                U,
                M,
                K,
                B,
                C,
                chunk_starts,
                dY,
                chunk_size=ctx.chunk_size,
                B_prev=B_prev,
                U_prev=U_prev,
                compute_dtype=ctx.compute_dtype,
            )
        else:
            dU_scan = torch.zeros_like(U, dtype=rdtype)
            dM_scan = torch.zeros_like(M, dtype=rdtype)
            dK_scan = torch.zeros_like(K, dtype=rdtype)
            dB_scan = torch.zeros_like(B, dtype=rdtype)
            dC_scan = torch.zeros_like(C, dtype=rdtype)
            d_chunk_starts = torch.zeros_like(chunk_starts, dtype=rdtype)
            dB_prev_scan = (
                None if B_prev is None else torch.zeros_like(B_prev, dtype=rdtype)
            )
            dU_prev_scan = (
                None if U_prev is None else torch.zeros_like(U_prev, dtype=rdtype)
            )

        if dY is not None or d_final_state is not None:
            if d_final_state is None:
                d_final_state = torch.zeros(
                    (batch_size, n_heads, P, D),
                    device=U.device,
                    dtype=torch.float32,
                )
            d_inc, d_m_chunk, d_initial_raw = state_passing_bwd_cute(
                chunk_starts,
                m_chunk,
                d_chunk_starts=(
                    d_chunk_starts
                    if d_chunk_starts.is_contiguous()
                    else d_chunk_starts.contiguous()
                ),
                d_final=_as_dtype_contiguous(d_final_state, dtype=torch.float32),
            )
            dU_inc, dM_inc, dK_inc, dB_inc, dB_prev_inc, dU_prev_inc = (
                chunk_increment_bwd_cute(
                    U,
                    M,
                    K,
                    B,
                    d_inc=d_inc,
                    d_m_chunk=d_m_chunk,
                    chunk_size=ctx.chunk_size,
                    B_prev=B_prev,
                    U_prev=U_prev,
                    compute_dtype=ctx.compute_dtype,
                )
            )
            d_initial = d_initial_raw if initial_states is not None else None
        else:
            dU_inc = torch.zeros_like(U, dtype=rdtype)
            dM_inc = torch.zeros_like(M, dtype=rdtype)
            dK_inc = torch.zeros_like(K, dtype=rdtype)
            dB_inc = torch.zeros_like(B, dtype=rdtype)
            dB_prev_inc = (
                None if B_prev is None else torch.zeros_like(B_prev, dtype=rdtype)
            )
            dU_prev_inc = (
                None if U_prev is None else torch.zeros_like(U_prev, dtype=rdtype)
            )
            d_initial = (
                None
                if initial_states is None
                else torch.zeros_like(initial_states, dtype=rdtype)
            )

        dU_scan.add_(dU_inc)
        dM_scan.add_(dM_inc)
        dK_scan.add_(dK_inc)
        dB_scan.add_(dB_inc)
        dC_total = dC_scan

        if dB_last is not None:
            dB_scan[:, :, -1, :] += dB_last.to(dtype=rdtype)
        if dU_last is not None:
            dU_scan[:, :, -1, :] += dU_last.to(dtype=rdtype)

        dB_prev_total = None
        if B_prev is not None and dB_prev_inc is not None and dB_prev_scan is not None:
            dB_prev_scan.add_(dB_prev_inc)
            dB_prev_total = dB_prev_scan

        dU_prev_total = None
        if U_prev is not None and dU_prev_inc is not None and dU_prev_scan is not None:
            dU_prev_scan.add_(dU_prev_inc)
            dU_prev_total = dU_prev_scan

        return (
            dU_scan.to(dtype=U.dtype),
            dM_scan.to(dtype=M.dtype),
            dK_scan.to(dtype=K.dtype),
            dB_scan.to(dtype=B.dtype),
            dC_total.to(dtype=C.dtype),
            None
            if initial_states is None or d_initial is None
            else d_initial.to(dtype=initial_states.dtype),
            None
            if B_prev is None or dB_prev_total is None
            else dB_prev_total.to(dtype=B_prev.dtype),
            None
            if U_prev is None or dU_prev_total is None
            else dU_prev_total.to(dtype=U_prev.dtype),
            None,
            None,
            None,
        )


def v2x2ssd_cute_autograd(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _V2x2SSDCuTeFn.apply(
            U,
            M,
            K,
            B,
            C,
            initial_states,
            B_prev,
            U_prev,
            int(chunk_size),
            compute_dtype,
            output_dtype,
        ),
    )


__all__ = ["v2x2ssd_cute_autograd"]
