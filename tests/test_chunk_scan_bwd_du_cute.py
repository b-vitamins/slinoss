from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (
    chunk_scan_bwd_du_cute,
    prepare_chunk_scan_bwd_du_operands,
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


def _quantized_diag_value_grads(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``dVprev/dVcurr`` for the packed tensor-core contract."""
    BHC, L, _, _ = map(int, Q.shape)

    q = Q.squeeze(2).to(torch.float32)
    kp = Kprev.squeeze(2).to(torch.float32)
    kc = Kcurr.squeeze(2).to(torch.float32)
    vp = Vprev.squeeze(2).to(torch.float32).detach().requires_grad_(True)
    vc = Vcurr.squeeze(2).to(torch.float32).detach().requires_grad_(True)
    lp = logprefix_half.to(torch.float32)

    t_idx = torch.arange(L, device=Q.device).unsqueeze(1)
    s_idx = torch.arange(L, device=Q.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    scale = torch.exp(lp.unsqueeze(-1) - lp.unsqueeze(1)).masked_fill(~causal, 0.0)

    scores_prev = torch.bmm(q, kp.transpose(1, 2)) * scale
    scores_curr = torch.bmm(q, kc.transpose(1, 2)) * scale
    y = torch.bmm(scores_prev.to(Q.dtype).to(torch.float32), vp)
    y = y + torch.bmm(scores_curr.to(Q.dtype).to(torch.float32), vc)

    loss = (y * d_out_flat).sum()
    dV_prev_ref, dV_curr_ref = torch.autograd.grad(loss, (vp, vc), retain_graph=False)
    return dV_prev_ref, dV_curr_ref


def _scatter_value_grads_to_u(
    dV_prev: torch.Tensor,
    dV_curr: torch.Tensor,
    *,
    batch: int,
    heads: int,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    BHC, L, P = map(int, dV_curr.shape)
    del L
    n_chunks = BHC // (batch * heads)
    T_pad = n_chunks * int(chunk_size)

    dU_blk = dV_curr.reshape(batch, heads, n_chunks, chunk_size, P).clone()
    dV_prev_blk = dV_prev.reshape(batch, heads, n_chunks, chunk_size, P)
    dU_blk[:, :, :, :-1, :] += dV_prev_blk[:, :, :, 1:, :]
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += dV_prev_blk[:, :, 1:, 0, :]

    dU = dU_blk.reshape(batch, heads, T_pad, P)[:, :, :T, :].contiguous()
    dU_prev = dV_prev_blk[:, :, 0, 0, :].contiguous()
    return dU, dU_prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_du_cute_matches_quantized_packed_reference() -> None:
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

    dV_prev_ref, dV_curr_ref = _quantized_diag_value_grads(
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        d_out_flat,
    )
    dU_ref, dU_prev_ref = _scatter_value_grads_to_u(
        dV_prev_ref,
        dV_curr_ref,
        batch=batch,
        heads=heads,
        T=T,
        chunk_size=chunk_size,
    )

    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = (
        prepare_chunk_scan_bwd_du_operands(Q, Kprev, Kcurr, logprefix_half)
    )
    dU_cute, dU_prev_cute = chunk_scan_bwd_du_cute(
        Q_rev,
        Kprev_rev,
        Kcurr_rev,
        neg_logprefix_half_rev,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    # The DU tensor-core path is intentionally approximate: it keeps the dense
    # causal contractions on tensor cores with fp32 accumulation, but the score
    # block itself is quantized back to the transport dtype before the value
    # MMA. That is the same principled low-precision contract we accept in the
    # other non-exact backward slices.
    torch.testing.assert_close(dU_cute, dU_ref, atol=1e-1, rtol=0.0)
    torch.testing.assert_close(dU_prev_cute, dU_prev_ref, atol=3e-2, rtol=0.0)
