"""Canonical backward entrypoint for the CuTe ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

import torch
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)

from .db import (
    chunk_scan_bwd_db_exact_cute,
)
from .dc import chunk_scan_bwd_dc_exact_cute
from .du import _chunk_scan_bwd_du_prepared_cute, prepare_chunk_scan_bwd_du_operands
from .dz0 import chunk_scan_bwd_dz0_packed_cute
from .param import chunk_scan_bwd_param_packed_cute


def _pack_chunk_scan_bwd_operands(
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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
        _batch_size,
        _n_heads,
        _T,
        _T_pad,
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
    return M_raw, K_raw, B_raw, B_head, Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0
def chunk_scan_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the chunk-scan backward stage on the canonical public contract."""
    if d_out.device.type != "cuda":
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if not d_out.is_contiguous():
        raise ValueError(f"d_out must be contiguous; got strides {d_out.stride()}.")

    batch_size, n_heads, T, _P = map(int, U.shape)
    M_raw, K_raw, B_raw, B_head, Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0 = (
        _pack_chunk_scan_bwd_operands(
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
        )
    )

    BHC, L, _, D = map(int, Q.shape)
    P = int(Vprev.shape[-1])
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    d_out_padded = d_out
    if T_pad != T:
        d_out_padded = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, T_pad - T, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )

    d_out_flat = d_out_padded.reshape(BHC, L, P).to(torch.float32)
    dZ0 = chunk_scan_bwd_dz0_packed_cute(
        Q.contiguous(),
        logprefix_half.contiguous(),
        d_out_flat.contiguous(),
    )
    N = D // 2
    d_chunk_starts = (
        torch.view_as_real(
            torch.conj(
                torch.view_as_complex(dZ0.reshape(BHC, P, N, 2).contiguous())
            ).resolve_conj()
        )
        .reshape(batch_size, n_heads, n_chunks, P, D)
        .to(dtype=torch.float32)
        .contiguous()
    )

    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = (
        prepare_chunk_scan_bwd_du_operands(
            Q.contiguous(),
            Kprev.contiguous(),
            Kcurr.contiguous(),
            logprefix_half.contiguous(),
        )
    )
    d_out_rev = torch.flip(
        d_out_padded.reshape(BHC, L, 1, P).to(dtype=Q_rev.dtype), dims=[1]
    ).contiguous()
    dU, dU_prev = _chunk_scan_bwd_du_prepared_cute(
        Q_rev,
        Kprev_rev,
        Kcurr_rev,
        neg_logprefix_half_rev,
        d_out_rev,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )

    dM, dQ, dK_prev_packed, dK_curr_packed, phase_real = (
        chunk_scan_bwd_param_packed_cute(
            Q,
            Kprev,
            Vprev,
            Kcurr,
            Vcurr,
            logprefix_half,
            Z0,
            M_raw,
            d_out,
            batch_size=batch_size,
            n_heads=n_heads,
            T=T,
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )
    dC = chunk_scan_bwd_dc_exact_cute(
        dQ,
        phase_real,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dB, dB_prev, dK = chunk_scan_bwd_db_exact_cute(
        dK_prev_packed.contiguous(),
        dK_curr_packed.contiguous(),
        phase_real,
        K_raw.to(dtype=torch.float32).contiguous(),
        B_raw.to(dtype=torch.float32).contiguous(),
        B_head.to(dtype=torch.float32).contiguous(),
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    return dU, dM, dK, dB, dC, d_chunk_starts, dB_prev, dU_prev


__all__ = ["chunk_scan_bwd_cute"]
