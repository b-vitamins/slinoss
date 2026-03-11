"""Exact packed helpers for chunk-scan validation and benchmarking."""

from __future__ import annotations

from typing import Any, cast

import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.db import (
    _chunk_scan_bwd_dk_prepared_cute,
    chunk_scan_bwd_db_exact_cute,
    prepare_chunk_scan_bwd_db_operands,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.dc import (
    chunk_scan_bwd_dc_exact_cute,
    chunk_scan_bwd_dc_packed_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.du import (
    _chunk_scan_bwd_du_prepared_cute,
    prepare_chunk_scan_bwd_du_operands,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.dz0 import (
    chunk_scan_bwd_dz0_packed_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.param_scan import (
    chunk_scan_bwd_dlogprefix_exact_cute,
    _chunk_scan_bwd_param_from_intermediates,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.common import (
    _scalar_grad_from_vec,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    batched_sgemm_fp32_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_chunk_scan,
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)


def run_chunk_scan_forward_and_pack(
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
    output_dtype: torch.dtype | None,
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
        batch_size,
        n_heads,
        T,
        T_pad,
        odtype,
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
        output_dtype=output_dtype,
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

    out = torch.empty(
        (Q.shape[0], Q.shape[1], 1, Vprev.shape[-1]),
        device=U.device,
        dtype=odtype,
    )
    compiled = cast(
        Any,
        _get_compiled_chunk_scan(
            Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0, out
        ),
    )
    compiled(
        from_dlpack(Q, assumed_align=16),
        from_dlpack(Kprev, assumed_align=16),
        from_dlpack(Vprev, assumed_align=16),
        from_dlpack(Kcurr, assumed_align=16),
        from_dlpack(Vcurr, assumed_align=16),
        from_dlpack(logprefix_half, assumed_align=16),
        from_dlpack(Z0, assumed_align=16),
        from_dlpack(out, assumed_align=16),
    )

    L = int(chunk_size)
    n_chunks = int(T_pad // L)
    Y = out.squeeze(2).reshape(batch_size, n_heads, n_chunks, L, out.shape[-1])
    Y = Y.reshape(batch_size, n_heads, T_pad, out.shape[-1])[:, :, :T].contiguous()
    return Y, M_raw, K_raw, B_raw, B_head, Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0


def _packed_phase_prefix(M_raw: torch.Tensor) -> torch.Tensor:
    m_c = torch.view_as_complex(M_raw.contiguous())
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.cumprod(m_c / mag, dim=1)


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


def _scatter_value_grads(
    dVprev: torch.Tensor,
    dVcurr: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    BHC, L, P = map(int, dVprev.shape)
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dU_blk = dVcurr.reshape(batch_size, n_heads, n_chunks, L, P).clone()
    dVprev_view = dVprev.reshape(batch_size, n_heads, n_chunks, L, P)
    if L > 1:
        dU_blk[..., :-1, :] += dVprev_view[..., 1:, :]

    d_head = dVprev_view[..., 0, :]
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += d_head[:, :, 1:, :]

    dU_prev = d_head[:, :, 0, :].contiguous()
    dU = dU_blk.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous()
    return dU, dU_prev


def _scatter_key_grads(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BHC, L, D = map(int, dK_prev_packed.shape)
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    N = D // 2

    phase_c = phase.unsqueeze(-1)
    dK_prev_c = torch.view_as_complex(
        dK_prev_packed.reshape(BHC, L, N, 2).contiguous()
    )
    dK_curr_c = torch.view_as_complex(
        dK_curr_packed.reshape(BHC, L, N, 2).contiguous()
    )
    d_beta_prev = phase_c * torch.conj(dK_prev_c)
    d_beta_curr = phase_c * torch.conj(dK_curr_c)

    b_curr = torch.view_as_complex(B_raw.reshape(BHC, L, N, 2).contiguous())
    b_head_c = torch.view_as_complex(B_head.reshape(BHC, N, 2).contiguous())
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if L > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]

    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous())
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous())

    dB_curr_c = torch.conj(k_curr_c).unsqueeze(-1) * d_beta_curr
    dB_prev_seq_c = torch.conj(k_prev_c).unsqueeze(-1) * d_beta_prev
    dK_prev_tap = _scalar_grad_from_vec(b_prev_seq, d_beta_prev)
    dK_curr_tap = _scalar_grad_from_vec(b_curr, d_beta_curr)

    dB_blk = dB_curr_c.reshape(batch_size, n_heads, n_chunks, L, N).clone()
    dB_prev_view = dB_prev_seq_c.reshape(batch_size, n_heads, n_chunks, L, N)
    if L > 1:
        dB_blk[..., :-1, :] += dB_prev_view[..., 1:, :]

    d_head = dB_prev_view[..., 0, :]
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_head[:, :, 1:, :]

    dB_prev = (
        torch.view_as_real(d_head[:, :, 0, :].contiguous())
        .reshape(batch_size, n_heads, D)
        .to(dtype=torch.float32)
        .contiguous()
    )
    dB = (
        torch.view_as_real(dB_blk)
        .reshape(batch_size, n_heads, T_pad, D)
        .to(dtype=torch.float32)[:, :, :T, :]
        .contiguous()
    )
    dK_prev_real = torch.view_as_real(
        dK_prev_tap.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK_curr_real = torch.view_as_real(
        dK_curr_tap.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK = (
        torch.stack((dK_prev_real, dK_curr_real), dim=4)
        .reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :]
        .contiguous()
    )
    return dB, dB_prev, dK


def chunk_scan_bwd_exact_packed(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    M_raw: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    BHC, L, _, D = map(int, Q.shape)
    P = int(Vprev.shape[-1])
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    N = D // 2

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


__all__ = ["chunk_scan_bwd_exact_packed", "run_chunk_scan_forward_and_pack"]
