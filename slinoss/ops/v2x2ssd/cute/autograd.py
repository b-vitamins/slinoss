"""Training-only autograd wrapper for the CuTe ``v2x2ssd`` operator."""

from __future__ import annotations

from typing import cast

import torch

from slinoss.perf import record_region
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


_ZERO_FINAL_GRAD_CACHE: dict[tuple, torch.Tensor] = {}


def _get_zero_final_grad(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_FINAL_GRAD_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(
            (batch_size, heads, P, D), device=device, dtype=torch.float32
        )
        _ZERO_FINAL_GRAD_CACHE[key] = cached
    return cached


class _V2x2SSDCuTeTrainingFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        U: torch.Tensor,
        M: torch.Tensor,
        K: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        compute_dtype: torch.dtype | None,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = compute_dtype
        ctx.output_dtype = output_dtype

        U_d = U.detach()
        M_d = M.detach()
        K_d = K.detach()
        B_d = B.detach()
        C_d = C.detach()

        inc, m_chunk = chunk_increment_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            chunk_size=ctx.chunk_size,
            compute_dtype=compute_dtype,
        )
        chunk_starts = cast(
            torch.Tensor,
            state_passing_cute(
                inc,
                m_chunk,
                initial_states=None,
                return_final_state=False,
            ),
        )
        Y = chunk_scan_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            chunk_starts,
            chunk_size=ctx.chunk_size,
            output_dtype=output_dtype or U.dtype,
            compute_dtype=compute_dtype,
        )

        ctx.save_for_backward(U_d, M_d, K_d, B_d, C_d, m_chunk, chunk_starts)
        return Y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        dY: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        with record_region("backward.v2x2ssd.custom_op_total"):
            U, M, K, B, C, m_chunk, chunk_starts = ctx.saved_tensors

            if dY is None:
                with record_region("backward.v2x2ssd.autograd_wrapper"):
                    return (
                        torch.zeros_like(U),
                        torch.zeros_like(M),
                        torch.zeros_like(K),
                        torch.zeros_like(B),
                        torch.zeros_like(C),
                        None,
                        None,
                        None,
                    )

            with record_region("backward.v2x2ssd.autograd_wrapper"):
                dY_contig = dY if dY.is_contiguous() else dY.contiguous()

            (
                dU_scan,
                dM_scan,
                dK_scan,
                dB_scan,
                dC_scan,
                d_chunk_starts,
            ) = cast(
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                chunk_scan_bwd_cute(
                    U,
                    M,
                    K,
                    B,
                    C,
                    chunk_starts,
                    dY_contig,
                    chunk_size=ctx.chunk_size,
                    compute_dtype=ctx.compute_dtype,
                    return_prev_grads=False,
                ),
            )

            zero_final = _get_zero_final_grad(
                device=U.device,
                batch_size=U.shape[0],
                heads=U.shape[1],
                P=U.shape[-1],
                D=B.shape[-1],
            )
            d_inc, d_m_chunk = cast(
                tuple[torch.Tensor, torch.Tensor],
                state_passing_bwd_cute(
                    chunk_starts,
                    m_chunk,
                    d_chunk_starts=d_chunk_starts,
                    d_final=zero_final,
                    return_d_initial=False,
                ),
            )

            dU_inc, dM_inc, dK_inc, dB_inc = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                chunk_increment_bwd_cute(
                    U,
                    M,
                    K,
                    B,
                    d_inc=d_inc,
                    d_m_chunk=d_m_chunk,
                    chunk_size=ctx.chunk_size,
                    compute_dtype=ctx.compute_dtype,
                    return_prev_grads=False,
                ),
            )

            with record_region("backward.v2x2ssd.autograd_wrapper"):
                dU_scan.add_(dU_inc)
                dM_scan.add_(dM_inc)
                dK_scan.add_(dK_inc)
                dB_scan.add_(dB_inc)

            return (
                dU_scan,
                dM_scan,
                dK_scan,
                dB_scan,
                dC_scan,
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
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _V2x2SSDCuTeTrainingFn.apply(
            U,
            M,
            K,
            B,
            C,
            int(chunk_size),
            compute_dtype,
            output_dtype,
        ),
    )


__all__ = ["v2x2ssd_cute_training_autograd"]
