"""Autograd wrapper for the CuTe ``v2x2ssd`` operator."""

from __future__ import annotations

from typing import cast

import torch

from .common import _materialize_boundary_tensor
from slinoss.ops.v2x2ssd.cute.kernels.bwd import v2x2ssd_bwd_stateful_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    _prepare_m_operand,
    _prepare_time_operand,
    _tc_input_dtype,
    _v2x2ssd_fwd_cute_prevalidated,
)


def _clone_if_capturing(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return x.clone()
    return x


def _prepare_backward_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_chunks = (int(U.shape[2]) + int(chunk_size) - 1) // int(chunk_size)
    padded_time = n_chunks * int(chunk_size)
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    return (
        _prepare_time_operand(U, padded_time=padded_time, dtype=tc_dtype),
        _prepare_m_operand(M, padded_time=padded_time),
        _prepare_time_operand(K, padded_time=padded_time, dtype=torch.float32),
        _prepare_time_operand(B, padded_time=padded_time, dtype=tc_dtype),
        _prepare_time_operand(C, padded_time=padded_time, dtype=tc_dtype),
    )


class _V2x2SSDCuTeTrainingFn(torch.autograd.Function):
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
        return_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = compute_dtype
        ctx.output_dtype = output_dtype or U.dtype
        ctx.return_state = bool(return_state)
        ctx.has_initial_states = initial_states is not None
        ctx.has_prev_state = B_prev is not None
        ctx.initial_state_dtype = (
            None if initial_states is None else cast(torch.dtype, initial_states.dtype)
        )

        U_d = U.detach()
        M_d = M.detach()
        K_d = K.detach()
        B_d = B.detach()
        C_d = C.detach()
        initial_states_d = None if initial_states is None else initial_states.detach()
        B_prev_d = None if B_prev is None else B_prev.detach()
        U_prev_d = None if U_prev is None else U_prev.detach()

        if ctx.return_state:
            Y, final_state, m_chunk, chunk_starts = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                _v2x2ssd_fwd_cute_prevalidated(
                    U_d,
                    M_d,
                    K_d,
                    B_d,
                    C_d,
                    chunk_size=ctx.chunk_size,
                    initial_states=initial_states_d,
                    B_prev=B_prev_d,
                    U_prev=U_prev_d,
                    compute_dtype=compute_dtype,
                    output_dtype=ctx.output_dtype,
                    return_final_state=True,
                ),
            )
            final_state_out = cast(
                torch.Tensor,
                _materialize_boundary_tensor(
                    final_state,
                    dtype=ctx.output_dtype,
                ),
            )
            B_last = cast(
                torch.Tensor,
                _materialize_boundary_tensor(
                    B_d[:, :, -1, :],
                    dtype=ctx.output_dtype,
                ),
            )
            U_last = cast(
                torch.Tensor,
                _materialize_boundary_tensor(
                    U_d[:, :, -1, :],
                    dtype=ctx.output_dtype,
                ),
            )

            saved_tensors = [U_d, M_d, K_d, B_d, C_d, m_chunk, chunk_starts]
            if ctx.has_prev_state:
                assert B_prev_d is not None
                assert U_prev_d is not None
                saved_tensors.extend([B_prev_d, U_prev_d])
            ctx.save_for_backward(*saved_tensors)
            return Y, final_state_out, B_last, U_last
        else:
            Y, m_chunk, chunk_starts = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                _v2x2ssd_fwd_cute_prevalidated(
                    U_d,
                    M_d,
                    K_d,
                    B_d,
                    C_d,
                    chunk_size=ctx.chunk_size,
                    initial_states=initial_states_d,
                    B_prev=B_prev_d,
                    U_prev=U_prev_d,
                    compute_dtype=compute_dtype,
                    output_dtype=ctx.output_dtype,
                ),
            )
            saved_tensors = [U_d, M_d, K_d, B_d, C_d, m_chunk, chunk_starts]
            if ctx.has_prev_state:
                assert B_prev_d is not None
                assert U_prev_d is not None
                saved_tensors.extend([B_prev_d, U_prev_d])
            ctx.save_for_backward(*saved_tensors)
            return Y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        U, M, K, B, C, m_chunk, chunk_starts = saved[:7]
        if ctx.has_prev_state:
            B_prev = cast(torch.Tensor, saved[7])
            U_prev = cast(torch.Tensor, saved[8])
        else:
            B_prev = None
            U_prev = None

        if ctx.return_state:
            dY, d_final_state, dB_last, dU_last = grad_outputs
        else:
            (dY,) = grad_outputs
            d_final_state = None
            dB_last = None
            dU_last = None

        dU_total: torch.Tensor | None = None
        dM_total: torch.Tensor | None = None
        dK_total: torch.Tensor | None = None
        dB_total: torch.Tensor | None = None
        dC_total: torch.Tensor | None = None
        d_initial_total: torch.Tensor | None = None
        dB_prev_total: torch.Tensor | None = None
        dU_prev_total: torch.Tensor | None = None

        needs_kernel = dY is not None or d_final_state is not None
        if needs_kernel:
            dY_contig = (
                dY
                if dY is not None and dY.is_contiguous()
                else (None if dY is None else dY.contiguous())
            )
            if dY_contig is None:
                dY_contig = torch.zeros(
                    tuple(U.shape), device=U.device, dtype=ctx.output_dtype
                )

            d_final_contig = (
                None
                if d_final_state is None
                else (
                    d_final_state
                    if d_final_state.is_contiguous()
                    else d_final_state.contiguous()
                )
            )
            prepared_inputs = _prepare_backward_inputs(
                U,
                M,
                K,
                B,
                C,
                chunk_size=ctx.chunk_size,
                compute_dtype=ctx.compute_dtype,
            )

            (
                dU_total,
                dM_total,
                dK_total,
                dB_total,
                dC_total,
                d_initial_total,
                dB_prev_total,
                dU_prev_total,
            ) = cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                v2x2ssd_bwd_stateful_cute(
                    U,
                    M,
                    K,
                    B,
                    C,
                    m_chunk,
                    chunk_starts,
                    dY_contig,
                    chunk_size=ctx.chunk_size,
                    initial_state_dtype=ctx.initial_state_dtype,
                    B_prev=B_prev,
                    U_prev=U_prev,
                    d_final_state=d_final_contig,
                    compute_dtype=ctx.compute_dtype,
                    prepared_inputs=prepared_inputs,
                ),
            )
            dM_total = _clone_if_capturing(dM_total)
        if dB_last is not None:
            if dB_total is None:
                dB_total = torch.zeros_like(B)
            dB_total[:, :, -1, :].add_(dB_last.to(dtype=B.dtype))
        if dU_last is not None:
            if dU_total is None:
                dU_total = torch.zeros_like(U)
            dU_total[:, :, -1, :].add_(dU_last.to(dtype=U.dtype))

        if ctx.return_state:
            d_initial_total = _materialize_boundary_tensor(d_initial_total)
            dB_prev_total = _materialize_boundary_tensor(dB_prev_total)
            dU_prev_total = _materialize_boundary_tensor(dU_prev_total)

        return (
            dU_total,
            dM_total,
            dK_total,
            dB_total,
            dC_total,
            d_initial_total if ctx.has_initial_states else None,
            dB_prev_total if ctx.has_prev_state else None,
            dU_prev_total if ctx.has_prev_state else None,
            None,
            None,
            None,
            None,
        )


def v2x2ssd_cute_training_autograd(
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
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return cast(
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _V2x2SSDCuTeTrainingFn.apply(
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
            bool(return_state),
        ),
    )


__all__ = ["v2x2ssd_cute_training_autograd"]
