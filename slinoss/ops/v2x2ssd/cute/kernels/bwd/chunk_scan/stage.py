"""Canonical backward entrypoint for the CuTe ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    batched_sgemm_fp32_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)

from .db import (
    _chunk_scan_bwd_dk_prepared_cute,
    chunk_scan_bwd_db_exact_cute,
    prepare_chunk_scan_bwd_db_operands,
)
from .dc import chunk_scan_bwd_dc_exact_cute, chunk_scan_bwd_dc_packed_cute
from .dlogprefix import chunk_scan_bwd_dlogprefix_exact_cute
from .du import _chunk_scan_bwd_du_prepared_cute, prepare_chunk_scan_bwd_du_operands
from .dz0 import chunk_scan_bwd_dz0_packed_cute
from .param import _chunk_scan_bwd_param_from_intermediates


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


def _packed_causal_scales(logprefix_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    L = int(logprefix_half.shape[1])
    t_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(1)
    s_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    lp = logprefix_half.to(torch.float32)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)
    return scale, row_scale


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

    if T_pad != T:
        d_out = torch.cat(
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

    d_out_flat = d_out.reshape(BHC, L, P).to(torch.float32)
    Qf = Q.squeeze(2).to(torch.float32)
    Kprevf = Kprev.squeeze(2).to(torch.float32)
    Kcurrf = Kcurr.squeeze(2).to(torch.float32)
    Vprevf = Vprev.squeeze(2).to(torch.float32)
    Vcurrf = Vcurr.squeeze(2).to(torch.float32)
    Z0f = Z0.squeeze(2).to(torch.float32)

    scale, row_scale = _packed_causal_scales(logprefix_half)
    score_prev = batched_sgemm_fp32_cute(Qf, Kprevf.transpose(1, 2))
    score_curr = batched_sgemm_fp32_cute(Qf, Kcurrf.transpose(1, 2))
    dSprev = batched_sgemm_fp32_cute(d_out_flat, Vprevf.transpose(1, 2))
    dScurr = batched_sgemm_fp32_cute(d_out_flat, Vcurrf.transpose(1, 2))

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
        d_out.reshape(BHC, L, 1, P).to(dtype=Q_rev.dtype), dims=[1]
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

    y_off = batched_sgemm_fp32_cute(Qf, Z0f.transpose(1, 2)) * row_scale
    d_logprefix_half = chunk_scan_bwd_dlogprefix_exact_cute(
        score_prev,
        score_curr,
        dSprev,
        dScurr,
        logprefix_half.contiguous(),
        y_off.contiguous(),
        d_out_flat,
    )

    Q_rev_db, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev_db, phase_real = (
        prepare_chunk_scan_bwd_db_operands(
            Q.contiguous(),
            Vprev.contiguous(),
            Vcurr.contiguous(),
            logprefix_half.contiguous(),
            M_raw.contiguous(),
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )
    phase = torch.view_as_complex(phase_real.contiguous())
    z0_q = Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    dQ = chunk_scan_bwd_dc_packed_cute(
        Vprev.contiguous(),
        Kprev.contiguous(),
        Vcurr.contiguous(),
        Kcurr.contiguous(),
        logprefix_half.contiguous(),
        z0_q,
        d_out,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dC = chunk_scan_bwd_dc_exact_cute(
        dQ,
        phase_real,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dK_prev_packed, dK_curr_packed = _chunk_scan_bwd_dk_prepared_cute(
        Q_rev_db,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev_db,
        d_out_rev,
        batch_size=batch_size,
        n_heads=n_heads,
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
    dM = _chunk_scan_bwd_param_from_intermediates(
        Qf,
        Kprevf,
        Kcurrf,
        phase,
        M_raw,
        dQ,
        dK_prev_packed,
        dK_curr_packed,
        d_logprefix_half,
        batch_size=batch_size,
        n_heads=n_heads,
    )[:, :, :T, :].contiguous()
    return dU, dM, dK, dB, dC, d_chunk_starts, dB_prev, dU_prev


__all__ = ["chunk_scan_bwd_cute"]
