from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _compiled_key,
    chunk_scan_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.state_passing import state_passing_cute
from slinoss.ops.v2x2ssd.reference import chunk_increment, chunk_scan, state_passing


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_scan_inputs(
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
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, initial_states, B_prev, U_prev


def test_chunk_scan_compiled_key_distinguishes_full_operand_contract() -> None:
    Q = torch.empty((4, 32, 1, 16), dtype=torch.float16)
    Kprev = torch.empty((4, 32, 1, 16), dtype=torch.float16)
    Kcurr = torch.empty((4, 32, 1, 16), dtype=torch.float16)
    logprefix = torch.empty((4, 32), dtype=torch.float32)
    Z0 = torch.empty((4, 16, 1, 16), dtype=torch.float16)
    out = torch.empty((4, 32, 1, 16), dtype=torch.float32)

    Vprev_a = torch.empty((4, 32, 1, 16), dtype=torch.float16)
    Vprev_b = torch.empty((4, 32, 1, 32), dtype=torch.float16)
    Vcurr_a = torch.empty((4, 32, 1, 16), dtype=torch.float16)
    Vcurr_b = torch.empty((4, 32, 1, 32), dtype=torch.float16)

    key_a = _compiled_key(
        Q,
        Kprev,
        Vprev_a,
        Kcurr,
        Vcurr_a,
        logprefix,
        Z0,
        out,
        device_index=0,
    )
    key_b = _compiled_key(
        Q,
        Kprev,
        Vprev_b,
        Kcurr,
        Vcurr_b,
        logprefix,
        Z0,
        out,
        device_index=0,
    )

    assert key_a != key_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_passing_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, chunks, N, P = 2, 2, 16, 8, 16
    D = 2 * N
    inc = torch.randn((batch, heads, chunks, P, D), device="cuda", dtype=torch.float32)
    radius = 0.6 + 0.35 * torch.rand((batch, heads, chunks), device="cuda")
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, chunks), device="cuda"
    ) - math.pi
    m_chunk = (
        torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    )
    initial_states = torch.randn(
        (batch, heads, P, D), device="cuda", dtype=torch.float32
    )

    starts_ref, final_ref = state_passing(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    starts_cute, final_cute = state_passing_cute(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    assert starts_cute.dtype == torch.float16
    assert final_cute.dtype == torch.float16
    torch.testing.assert_close(starts_cute.float(), starts_ref, atol=4e-3, rtol=0.0)
    torch.testing.assert_close(final_cute.float(), final_ref, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    y_ref = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    y_cute = chunk_scan_cute(
        U,
        M,
        K,
        B,
        C,
        starts_ref.to(dtype=torch.float16),
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_forward() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    y_ref, final_ref, b_last_ref, u_last_ref = v2x2ssd(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    y_cute, final_cute, b_last_cute, u_last_cute = v2x2ssd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)
    torch.testing.assert_close(final_cute, final_ref, atol=2e-3, rtol=0.0)
    torch.testing.assert_close(b_last_cute, b_last_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(u_last_cute, u_last_ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_rejects_autograd_tracked_inputs() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    U.requires_grad_(True)

    with pytest.raises(ValueError, match="does not support autograd-tracked tensors"):
        v2x2ssd_cute(
            U,
            M,
            K,
            B,
            C,
            chunk_size=32,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
        )
