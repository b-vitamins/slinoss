from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (
    chunk_scan_bwd_dc_cute,
    chunk_scan_bwd_dc_exact_cute,
    chunk_scan_bwd_dc_packed_cute,
    prepare_chunk_scan_bwd_dc_operands,
)
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


def _quantized_packed_dq_reference(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    phase: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Reference ``dC`` for the packed tensor-core chunk-scan contract."""
    q = Q.squeeze(2).to(torch.float32).detach().requires_grad_(True)
    kp = Kprev.squeeze(2).to(torch.float32)
    kc = Kcurr.squeeze(2).to(torch.float32)
    vp = Vprev.squeeze(2).to(torch.float32)
    vc = Vcurr.squeeze(2).to(torch.float32)
    z0 = Z0.squeeze(2).to(torch.float32)
    lp = logprefix_half.to(torch.float32)

    L = int(q.shape[1])
    t_idx = torch.arange(L, device=q.device).unsqueeze(1)
    s_idx = torch.arange(L, device=q.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(lp.unsqueeze(-1) - lp.unsqueeze(1)).masked_fill(~causal, 0.0)
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)

    y = torch.bmm(q, z0.transpose(1, 2)) * row_scale
    y = y + torch.bmm(
        (torch.bmm(q, kp.transpose(1, 2)) * scale).to(Q.dtype).to(torch.float32), vp
    )
    y = y + torch.bmm(
        (torch.bmm(q, kc.transpose(1, 2)) * scale).to(Q.dtype).to(torch.float32), vc
    )

    (dq_ref,) = torch.autograd.grad((y * d_out_flat).sum(), (q,), retain_graph=False)
    return dq_ref


def _scatter_packed_dq_to_public(dq_ref: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    pr = phase[..., 0].unsqueeze(-1)
    pi = phase[..., 1].unsqueeze(-1)
    dqr = dq_ref[..., 0::2]
    dqi = dq_ref[..., 1::2]
    dcr = dqr * pr + dqi * pi
    dci = dqr * pi - dqi * pr
    return torch.stack((dcr, dci), dim=-1).reshape(*dq_ref.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_dc_cute_matches_quantized_packed_reference() -> None:
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

    phase, half_logprefix_half, Z0_q = prepare_chunk_scan_bwd_dc_operands(
        M_raw,
        logprefix_half,
        Z0,
    )

    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    if T_pad != T:
        pad = T_pad - T
        d_out_pad = torch.cat(
            [
                d_out,
                torch.zeros((batch, heads, pad, P), device=device, dtype=torch.float32),
            ],
            dim=2,
        )
    else:
        d_out_pad = d_out
    d_out_flat = d_out_pad.reshape(Q.shape[0], chunk_size, P)

    dQ_ref = _quantized_packed_dq_reference(
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        Z0,
        phase,
        d_out_flat,
    )
    dC_ref = _scatter_packed_dq_to_public(dQ_ref, phase)
    dC_ref = dC_ref.reshape(batch, heads, T_pad, Q.shape[-1])[:, :, :T, :].contiguous()

    dC_cute = chunk_scan_bwd_dc_cute(
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        logprefix_half,
        half_logprefix_half,
        Z0_q,
        phase,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    torch.testing.assert_close(dC_cute, dC_ref, atol=1e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_dc_packed_cute_matches_quantized_packed_dq_reference() -> None:
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

    phase, _half_logprefix_half, Z0_q = prepare_chunk_scan_bwd_dc_operands(
        M_raw,
        logprefix_half,
        Z0,
    )

    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    if T_pad != T:
        pad = T_pad - T
        d_out_pad = torch.cat(
            [
                d_out,
                torch.zeros((batch, heads, pad, P), device=device, dtype=torch.float32),
            ],
            dim=2,
        )
    else:
        d_out_pad = d_out
    d_out_flat = d_out_pad.reshape(Q.shape[0], chunk_size, P)

    dQ_ref = _quantized_packed_dq_reference(
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        Z0,
        phase,
        d_out_flat,
    )
    dQ_cute = chunk_scan_bwd_dc_packed_cute(
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        logprefix_half,
        Z0_q,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    torch.testing.assert_close(dQ_cute, dQ_ref, atol=1e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_dc_exact_cute_matches_exact_scatter() -> None:
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
        d_out_pad = torch.cat(
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
    else:
        d_out_pad = d_out

    q = Q.squeeze(2).to(torch.float32)
    kp = Kprev.squeeze(2).to(torch.float32)
    kc = Kcurr.squeeze(2).to(torch.float32)
    vp = Vprev.squeeze(2).to(torch.float32)
    vc = Vcurr.squeeze(2).to(torch.float32)
    z0 = Z0.squeeze(2).to(torch.float32)
    lp = logprefix_half.to(torch.float32)
    d_out_flat = d_out_pad.reshape(Q.shape[0], chunk_size, P)

    L = int(q.shape[1])
    t_idx = torch.arange(L, device=q.device).unsqueeze(1)
    s_idx = torch.arange(L, device=q.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)
    dSprev = torch.bmm(d_out_flat, vp.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, vc.transpose(1, 2))
    dQ = (
        torch.bmm(d_out_flat * row_scale, z0)
        + torch.bmm(dSprev * scale, kp)
        + torch.bmm(dScurr * scale, kc)
    ).contiguous()

    phase = prepare_chunk_scan_bwd_dc_operands(M_raw, logprefix_half, Z0)[0]
    pr = phase[..., 0].unsqueeze(-1)
    pi = phase[..., 1].unsqueeze(-1)
    dqr = dQ[..., 0::2]
    dqi = dQ[..., 1::2]
    dC_ref = torch.stack((dqr * pr + dqi * pi, dqr * pi - dqi * pr), dim=-1)
    dC_ref = dC_ref.reshape(batch, heads, T_pad, dQ.shape[-1])[:, :, :T, :].contiguous()

    dC_cute = chunk_scan_bwd_dc_exact_cute(
        dQ,
        phase.to(torch.float32).contiguous(),
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    torch.testing.assert_close(dC_cute, dC_ref, atol=1e-6, rtol=0.0)
