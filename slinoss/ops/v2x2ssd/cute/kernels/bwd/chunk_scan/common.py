"""Common helpers for CuTe ``v2x2ssd`` chunk-scan backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)


@dataclass(frozen=True)
class ChunkScanBwdPackedContext:
    batch_size: int
    n_heads: int
    T: int
    T_pad: int
    n_chunks: int
    BHC: int
    L: int
    P: int
    D: int
    M_raw: torch.Tensor
    K_raw: torch.Tensor
    B_raw: torch.Tensor
    B_head: torch.Tensor
    Q: torch.Tensor
    Kprev: torch.Tensor
    Vprev: torch.Tensor
    Kcurr: torch.Tensor
    Vcurr: torch.Tensor
    logprefix_half: torch.Tensor
    Z0: torch.Tensor


def prepare_chunk_scan_bwd_packed_context(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
) -> ChunkScanBwdPackedContext:
    """Prepare the canonical packed backward context for ``chunk_scan``."""
    (
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix_half,
        Z0_raw,
        U_head,
        B_head,
        batch_size,
        n_heads,
        T,
        T_pad,
        _odtype,
    ) = _prepare_chunk_scan_small_operands(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
    )
    Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0 = _pack_chunk_scan_inner_inputs(
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix_half,
        Z0_raw,
        U_head,
        B_head,
    )

    BHC, L, _, D = map(int, Q.shape)
    P = int(Vprev.shape[-1])
    n_chunks = int(T_pad // L)
    return ChunkScanBwdPackedContext(
        batch_size=int(batch_size),
        n_heads=int(n_heads),
        T=int(T),
        T_pad=int(T_pad),
        n_chunks=n_chunks,
        BHC=BHC,
        L=L,
        P=P,
        D=D,
        M_raw=M_raw,
        K_raw=K_raw,
        B_raw=B_raw,
        B_head=B_head,
        Q=Q,
        Kprev=Kprev,
        Vprev=Vprev,
        Kcurr=Kcurr,
        Vcurr=Vcurr,
        logprefix_half=logprefix_half,
        Z0=Z0,
    )


def prepare_chunk_scan_bwd_dout(
    d_out: torch.Tensor,
    *,
    ctx: ChunkScanBwdPackedContext,
    tc_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare padded, flattened, and reversed ``d_out`` views."""
    if d_out.device.type != "cuda":
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if not d_out.is_contiguous():
        raise ValueError(f"d_out must be contiguous; got strides {d_out.stride()}.")
    if d_out.shape != (ctx.batch_size, ctx.n_heads, ctx.T, ctx.P):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P) = "
            f"{(ctx.batch_size, ctx.n_heads, ctx.T, ctx.P)}. Got {tuple(d_out.shape)}."
        )

    d_out_padded = d_out
    if ctx.T_pad != ctx.T:
        d_out_padded = torch.cat(
            [
                d_out,
                torch.zeros(
                    (ctx.batch_size, ctx.n_heads, ctx.T_pad - ctx.T, ctx.P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_flat = d_out_padded.reshape(ctx.BHC, ctx.L, ctx.P).to(torch.float32).contiguous()
    d_out_rev = torch.flip(
        d_out_padded.reshape(ctx.BHC, ctx.L, 1, ctx.P).to(dtype=tc_dtype),
        dims=[1],
    ).contiguous()
    return d_out_padded.contiguous(), d_out_flat, d_out_rev


__all__ = [
    "ChunkScanBwdPackedContext",
    "prepare_chunk_scan_bwd_packed_context",
    "prepare_chunk_scan_bwd_dout",
]
