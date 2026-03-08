from __future__ import annotations

import math

import torch

from slinoss.layers import (
    ReferenceScanBackend,
    SLinOSSDiscretizer,
    ScanInputs,
    ScanState,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)
from slinoss.ops.v2x2ssd import v2x2ssd


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def test_build_transition_from_polar_matches_complex_scalar() -> None:
    r = torch.tensor([[0.7, 0.9], [0.95, 1.0]], dtype=torch.float32)
    theta = torch.tensor(
        [[0.0, math.pi / 3.0], [-math.pi / 4.0, math.pi]], dtype=torch.float32
    )

    actual = build_transition_from_polar(r, theta)
    expected = torch.view_as_real(torch.polar(r, principal_angle(theta))).contiguous()

    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_foh_taps_match_midpoint_rule_at_identity() -> None:
    dt = torch.tensor([[0.1, 0.25], [0.5, 1.0]], dtype=torch.float32)
    r = torch.ones_like(dt)
    theta = torch.zeros_like(dt)

    k_prev, k_curr = foh_taps_from_polar(dt, r, theta, eps=1e-8)
    half_dt = 0.5 * dt

    assert torch.allclose(k_prev[..., 0], half_dt, atol=1e-7, rtol=0.0)
    assert torch.allclose(k_curr[..., 0], half_dt, atol=1e-7, rtol=0.0)
    assert torch.equal(k_prev[..., 1], torch.zeros_like(half_dt))
    assert torch.equal(k_curr[..., 1], torch.zeros_like(half_dt))


def test_discretizer_outputs_are_bounded_and_finite() -> None:
    torch.manual_seed(0)
    disc = SLinOSSDiscretizer(
        n_heads=3,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
    )
    params = torch.randn((2, 7, 3, disc.param_dim), dtype=torch.float32)

    out = disc(params)
    m = torch.view_as_complex(out.M)
    r = torch.abs(m)

    assert out.M.shape == (2, 3, 7, 2)
    assert out.K.shape == (2, 3, 7, 2, 2)
    assert out.dt.shape == (2, 3, 7)
    assert out.r.shape == (2, 3, 7)
    assert out.theta.shape == (2, 3, 7)
    assert torch.isfinite(out.M).all()
    assert torch.isfinite(out.K).all()
    assert torch.isfinite(out.dt).all()
    assert torch.isfinite(out.r).all()
    assert torch.isfinite(out.theta).all()
    assert bool((out.dt >= disc.dt_min).all())
    assert bool((out.dt <= disc.dt_max).all())
    assert bool((out.r >= disc.r_min).all())
    assert bool((out.r <= disc.r_max).all())
    assert torch.allclose(r, out.r, atol=1e-6, rtol=1e-6)


def test_reference_scan_backend_matches_v2x2ssd() -> None:
    torch.manual_seed(1)
    device = torch.device("cpu")
    batch, heads, T, N, P = 2, 2, 9, 3, 4
    chunk_size = 4

    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    K_complex = (
        torch.randn((batch, heads, T, 2), device=device)
        + 1j * torch.randn((batch, heads, T, 2), device=device)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()
    U = torch.randn((batch, heads, T, P), device=device)
    B = torch.randn((batch, heads, T, 2 * N), device=device) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device) * 0.1

    z0 = (
        torch.randn((batch, heads, P, N), device=device)
        + 1j * torch.randn((batch, heads, P, N), device=device)
    ) * 0.1
    initial_state = _pack_complex_pairs(z0, real_dtype=torch.float32)
    b_prev = (
        torch.randn((batch, heads, N), device=device)
        + 1j * torch.randn((batch, heads, N), device=device)
    ) * 0.1
    prev_state = ScanState(
        state=initial_state,
        b_prev=_pack_complex_pairs(b_prev, real_dtype=torch.float32),
        u_prev=torch.randn((batch, heads, P), device=device),
    )

    inputs = ScanInputs(U=U, M=M, K=K, B=B, C=C)
    backend = ReferenceScanBackend(compute_dtype=torch.float64)
    y_backend, next_state = backend(inputs, chunk_size=chunk_size, state=prev_state)
    y_ref, final_state, b_last, u_last = v2x2ssd(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        initial_states=prev_state.state,
        B_prev=prev_state.b_prev,
        U_prev=prev_state.u_prev,
        compute_dtype=torch.float64,
        output_dtype=U.dtype,
    )

    assert torch.allclose(y_backend, y_ref, atol=1e-10, rtol=0.0)
    assert next_state.state is not None
    assert next_state.b_prev is not None
    assert next_state.u_prev is not None
    assert torch.equal(next_state.state, final_state)
    assert torch.equal(next_state.b_prev, b_last)
    assert torch.equal(next_state.u_prev, u_last)
