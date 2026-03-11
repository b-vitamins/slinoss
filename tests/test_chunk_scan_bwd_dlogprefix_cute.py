from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.param_scan import (
    chunk_scan_bwd_dlogprefix_exact_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.param_scan import _dlogprefix_half_packed
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment, state_passing


def _make_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    B = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    B_prev = (
        torch.randn((batch, heads, 2 * N), device=device, dtype=torch.float32) * 0.1
    )
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, B_prev, U_prev


def _packed_causal_scales(
    logprefix_half: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lp = logprefix_half.to(torch.float32)
    L = int(lp.shape[1])
    t_idx = torch.arange(L, device=lp.device).unsqueeze(1)
    s_idx = torch.arange(L, device=lp.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)
    return scale, row_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_dlogprefix_exact_cute_matches_exact_reference() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )

    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )

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
        _batch,
        _heads,
        _T,
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
        compute_dtype=torch.float32,
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

    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    if T_pad != T:
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch, heads, T_pad - T, P),
                    device=device,
                    dtype=torch.float32,
                ),
            ],
            dim=2,
        )
    d_out_flat = d_out.reshape(Q.shape[0], chunk_size, P).contiguous()

    Qf = Q.squeeze(2).to(torch.float32)
    Kprevf = Kprev.squeeze(2).to(torch.float32)
    Kcurrf = Kcurr.squeeze(2).to(torch.float32)
    Vprevf = Vprev.squeeze(2).to(torch.float32)
    Vcurrf = Vcurr.squeeze(2).to(torch.float32)
    Z0f = Z0.squeeze(2).to(torch.float32)

    scale, row_scale = _packed_causal_scales(logprefix_half)
    score_prev = torch.bmm(Qf, Kprevf.transpose(1, 2))
    score_curr = torch.bmm(Qf, Kcurrf.transpose(1, 2))
    dSprev = torch.bmm(d_out_flat, Vprevf.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, Vcurrf.transpose(1, 2))
    y_off = (torch.bmm(Qf, Z0f.transpose(1, 2)) * row_scale).contiguous()

    dlog_ref = _dlogprefix_half_packed(
        score_prev,
        score_curr,
        dSprev,
        dScurr,
        y_off,
        scale,
        d_out_flat,
    )
    dlog_cute = chunk_scan_bwd_dlogprefix_exact_cute(
        score_prev.contiguous(),
        score_curr.contiguous(),
        dSprev.contiguous(),
        dScurr.contiguous(),
        logprefix_half.contiguous(),
        y_off,
        d_out_flat,
    )

    torch.testing.assert_close(dlog_cute, dlog_ref, atol=1e-5, rtol=0.0)
