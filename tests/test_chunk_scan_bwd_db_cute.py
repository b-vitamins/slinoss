from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.db import (
    chunk_scan_bwd_db_cute,
    chunk_scan_bwd_db_exact_cute,
    chunk_scan_bwd_dk_packed_cute,
    prepare_chunk_scan_bwd_db_operands,
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


def _scalar_grad_from_vec(base: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return torch.complex(
        (grad.real * base.real + grad.imag * base.imag).sum(dim=-1),
        (-grad.real * base.imag + grad.imag * base.real).sum(dim=-1),
    )


def _quantized_diag_key_grads(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference packed ``dKprev/dKcurr`` for the tensor-core chunk-scan contract."""
    q = Q.squeeze(2).to(torch.float32)
    kp = Kprev.squeeze(2).to(torch.float32).detach().requires_grad_(True)
    kc = Kcurr.squeeze(2).to(torch.float32).detach().requires_grad_(True)
    vp = Vprev.squeeze(2).to(torch.float32)
    vc = Vcurr.squeeze(2).to(torch.float32)
    lp = logprefix_half.to(torch.float32)

    L = int(q.shape[1])
    t_idx = torch.arange(L, device=q.device).unsqueeze(1)
    s_idx = torch.arange(L, device=q.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(lp.unsqueeze(-1) - lp.unsqueeze(1)).masked_fill(~causal, 0.0)

    scores_prev = torch.bmm(q, kp.transpose(1, 2)) * scale
    scores_curr = torch.bmm(q, kc.transpose(1, 2)) * scale
    y = torch.bmm(scores_prev.to(Q.dtype).to(torch.float32), vp)
    y = y + torch.bmm(scores_curr.to(Q.dtype).to(torch.float32), vc)

    loss = (y * d_out_flat).sum()
    dK_prev_ref, dK_curr_ref = torch.autograd.grad(loss, (kp, kc), retain_graph=False)
    return dK_prev_ref, dK_curr_ref


def _scatter_key_grads_to_public(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch: int,
    heads: int,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BHC, L, D = map(int, dK_prev_packed.shape)
    del L
    N = D // 2
    n_chunks = BHC // (batch * heads)
    T_pad = n_chunks * int(chunk_size)

    phase_c = torch.view_as_complex(phase.contiguous()).unsqueeze(-1)
    dK_prev_c = torch.view_as_complex(
        dK_prev_packed.reshape(BHC, chunk_size, N, 2).contiguous()
    )
    dK_curr_c = torch.view_as_complex(
        dK_curr_packed.reshape(BHC, chunk_size, N, 2).contiguous()
    )
    d_beta_prev = phase_c * torch.conj(dK_prev_c)
    d_beta_curr = phase_c * torch.conj(dK_curr_c)

    b_curr = torch.view_as_complex(B_raw.reshape(BHC, chunk_size, N, 2).contiguous())
    b_head_c = torch.view_as_complex(B_head.reshape(BHC, N, 2).contiguous())
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if chunk_size > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]

    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous())
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous())

    dB_curr_c = torch.conj(k_curr_c).unsqueeze(-1) * d_beta_curr
    dB_prev_seq_c = torch.conj(k_prev_c).unsqueeze(-1) * d_beta_prev
    dK_prev_tap_c = _scalar_grad_from_vec(b_prev_seq, d_beta_prev)
    dK_curr_tap_c = _scalar_grad_from_vec(b_curr, d_beta_curr)

    dB_blk = dB_curr_c.reshape(batch, heads, n_chunks, chunk_size, N).clone()
    dB_prev_view = dB_prev_seq_c.reshape(batch, heads, n_chunks, chunk_size, N)
    if chunk_size > 1:
        dB_blk[:, :, :, :-1, :] += dB_prev_view[:, :, :, 1:, :]

    d_head_c = dB_prev_view[:, :, :, 0, :]
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_head_c[:, :, 1:, :]

    dB_prev0_c = d_head_c[:, :, 0, :].contiguous()
    dB = (
        torch.view_as_real(dB_blk)
        .reshape(batch, heads, T_pad, D)
        .to(dtype=torch.float32)[:, :, :T, :]
        .contiguous()
    )
    dB_prev = (
        torch.view_as_real(dB_prev0_c)
        .reshape(batch, heads, D)
        .to(dtype=torch.float32)
        .contiguous()
    )

    dK_prev_real = torch.view_as_real(
        dK_prev_tap_c.reshape(batch, heads, n_chunks, chunk_size)
    ).to(dtype=torch.float32)
    dK_curr_real = torch.view_as_real(
        dK_curr_tap_c.reshape(batch, heads, n_chunks, chunk_size)
    ).to(dtype=torch.float32)
    dK = (
        torch.stack((dK_prev_real, dK_curr_real), dim=4)
        .reshape(batch, heads, T_pad, 2, 2)[:, :, :T, :, :]
        .contiguous()
    )
    return dB, dB_prev, dK


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_db_cute_matches_quantized_packed_reference() -> None:
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
    Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, _Z0 = _pack_chunk_scan_inner_inputs(
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

    Q_rev, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev, phase = (
        prepare_chunk_scan_bwd_db_operands(
            Q,
            Vprev,
            Vcurr,
            logprefix_half,
            M_raw,
        )
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

    dK_prev_ref, dK_curr_ref = _quantized_diag_key_grads(
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        d_out_flat,
    )
    dB_ref, dB_prev_ref, dK_ref = _scatter_key_grads_to_public(
        dK_prev_ref,
        dK_curr_ref,
        phase,
        K_raw,
        B_raw,
        B_head,
        batch=batch,
        heads=heads,
        T=T,
        chunk_size=chunk_size,
    )

    dK_prev_cute, dK_curr_cute = chunk_scan_bwd_dk_packed_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )
    dB_cute, dB_prev_cute, dK_cute = chunk_scan_bwd_db_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        phase,
        K_raw,
        B_raw,
        B_head,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    torch.testing.assert_close(dK_prev_cute, dK_prev_ref, atol=3e-2, rtol=0.0)
    torch.testing.assert_close(dK_curr_cute, dK_curr_ref, atol=3e-2, rtol=0.0)
    torch.testing.assert_close(dB_cute, dB_ref, atol=3e-2, rtol=0.0)
    torch.testing.assert_close(dB_prev_cute, dB_prev_ref, atol=3e-2, rtol=0.0)
    torch.testing.assert_close(dK_cute, dK_ref, atol=3e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_db_exact_cute_matches_exact_scatter() -> None:
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
    Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, _Z0 = _pack_chunk_scan_inner_inputs(
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
    d_out_flat = d_out_pad.reshape(Q.shape[0], chunk_size, P)

    q = Q.squeeze(2).to(torch.float32)
    vp = Vprev.squeeze(2).to(torch.float32)
    vc = Vcurr.squeeze(2).to(torch.float32)
    lp = logprefix_half.to(torch.float32)
    L = int(q.shape[1])
    t_idx = torch.arange(L, device=q.device).unsqueeze(1)
    s_idx = torch.arange(L, device=q.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    dSprev = torch.bmm(d_out_flat, vp.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, vc.transpose(1, 2))
    dK_prev_packed = torch.bmm((dSprev * scale).transpose(1, 2), q).contiguous()
    dK_curr_packed = torch.bmm((dScurr * scale).transpose(1, 2), q).contiguous()

    phase = torch.view_as_real(
        torch.view_as_complex(
            prepare_chunk_scan_bwd_db_operands(
                Q,
                Vprev,
                Vcurr,
                logprefix_half,
                M_raw,
            )[4].contiguous()
        )
    ).to(torch.float32)

    dB_ref, dB_prev_ref, dK_ref = _scatter_key_grads_to_public(
        dK_prev_packed,
        dK_curr_packed,
        phase,
        K_raw,
        B_raw,
        B_head,
        batch=batch,
        heads=heads,
        T=T,
        chunk_size=chunk_size,
    )

    dB_cute, dB_prev_cute, dK_cute = chunk_scan_bwd_db_exact_cute(
        dK_prev_packed,
        dK_curr_packed,
        phase.to(torch.float32).contiguous(),
        K_raw.to(torch.float32).contiguous(),
        B_raw.to(torch.float32).contiguous(),
        B_head.to(torch.float32).contiguous(),
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    torch.testing.assert_close(dB_cute, dB_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(dB_prev_cute, dB_prev_ref, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(dK_cute, dK_ref, atol=1e-6, rtol=0.0)
