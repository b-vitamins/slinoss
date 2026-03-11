from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (
    chunk_scan_bwd_cute,
    compile_chunk_scan_bwd_kernels,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.exact import (
    chunk_scan_bwd_exact_packed,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_stage_entrypoint_matches_packed_helper() -> None:
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
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    got = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
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
    want = chunk_scan_bwd_exact_packed(
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        Z0,
        M_raw,
        K_raw,
        B_raw,
        B_head,
        d_out,
        batch_size=batch,
        n_heads=heads,
        T=T,
    )

    atol_by_slot = (
        1e-1,   # dU
        2.5e-1, # dM
        3e-2,   # dK
        3e-2,   # dB
        1e-3,   # dC
        5e-4,   # d_chunk_starts
        3e-2,   # dB_prev
        3e-2,   # dU_prev
    )
    for got_tensor, want_tensor, atol in zip(got, want, atol_by_slot, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_compile_entrypoint_matches_public_stage() -> None:
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
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    got_public = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    compiled = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        return_launchers=True,
    )
    got_compiled = compiled[:8]
    launch_sequential = compiled[8]
    launch_sequential()

    atol_by_slot = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    for got_tensor, want_tensor, atol in zip(got_compiled, got_public, atol_by_slot, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)
