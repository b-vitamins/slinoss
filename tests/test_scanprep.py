from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import pytest

from slinoss.layers import (
    AutoScanPrepBackend,
    CuteScanPrepBackend,
    ReferenceScanPrepBackend,
    ReferenceScanBackend,
    SLinOSSScanPrep,
    ScanInputs,
    ScanState,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)
from slinoss.ops.scanprep.cute.common import make_ptr_arg
from slinoss.ops.v2x2ssd import v2x2ssd


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_noncontiguous_clone(x: torch.Tensor) -> torch.Tensor:
    padded = torch.empty((*x.shape, 2), device=x.device, dtype=x.dtype)
    padded[..., 0].copy_(x)
    padded[..., 1].zero_()
    out = padded[..., 0]
    assert tuple(out.shape) == tuple(x.shape)
    assert not out.is_contiguous()
    return out


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_ptr_cache_keeps_same_base_views_distinct() -> None:
    pytest.importorskip("cutlass")

    x = torch.empty((1, 1, 1, 2, 2), device="cuda", dtype=torch.float32)
    base_ptr, _ = make_ptr_arg(x)
    slice0_ptr, _ = make_ptr_arg(x[:, :, :, 0, :])
    slice1_ptr, _ = make_ptr_arg(x[:, :, :, 1, :])

    assert base_ptr is not slice0_ptr
    assert base_ptr is not slice1_ptr


def test_scanprep_coefficients_are_bounded_and_finite() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(
        n_heads=3,
        d_state=4,
        d_head=2,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
    )
    params = torch.randn((2, 7, 3, prep.param_dim), dtype=torch.float32)

    out = prep.coefficients(params)
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
    assert bool((out.dt >= prep.dt_min).all())
    assert bool((out.dt <= prep.dt_max).all())
    assert bool((out.r >= prep.r_min).all())
    assert bool((out.r <= prep.r_max).all())
    assert torch.allclose(r, out.r, atol=1e-6, rtol=1e-6)


def test_cute_scanprep_backend_requires_cuda() -> None:
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
    )
    value = torch.randn((2, 5, 8), dtype=torch.float32)
    params = torch.randn((2, 5, 2 * prep.param_dim), dtype=torch.float32)
    bc = torch.randn((2, 5, 2, 4, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="CuTe scanprep requires CUDA tensors"):
        prep(value, params, bc)


def test_auto_scanprep_backend_uses_reference_on_cpu() -> None:
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=AutoScanPrepBackend(),
    )
    assert isinstance(prep.backend, AutoScanPrepBackend)

    value = torch.randn((2, 5, 8), dtype=torch.float32)
    params = torch.randn((2, 5, 2 * prep.param_dim), dtype=torch.float32)
    bc = torch.randn((2, 5, 2, 4, 3), dtype=torch.float32)

    got = prep(value, params, bc)

    prep_ref = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=ReferenceScanPrepBackend(),
    )
    prep_ref.load_state_dict(prep.state_dict())
    expect = prep_ref(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.equal(got.M, expect.M)
    assert torch.equal(got.K, expect.K)
    assert torch.equal(got.B, expect.B)
    assert torch.equal(got.C, expect.C)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_scanprep_backend_uses_cute_on_cuda_fp32() -> None:
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=AutoScanPrepBackend(),
        device="cuda",
    )
    prep_cute = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    )
    prep_cute.load_state_dict(prep.state_dict())

    value = torch.randn((2, 5, 8), device="cuda", dtype=torch.float32)
    params = torch.randn((2, 5, 2 * prep.param_dim), device="cuda", dtype=torch.float32)
    bc = torch.randn((2, 5, 2, 4, 3), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        got = prep(value, params, bc)
        expect = prep_cute(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.allclose(got.B, expect.B, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.C, expect.C, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.M, expect.M, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.K, expect.K, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_auto_scanprep_backend_uses_cute_on_cuda_training_dtypes(
    dtype: torch.dtype,
) -> None:
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=AutoScanPrepBackend(),
        device="cuda",
    ).to(dtype=dtype)
    prep_cute = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    ).to(dtype=dtype)
    prep_cute.load_state_dict(prep.state_dict())

    value = torch.randn((2, 5, 8), device="cuda", dtype=dtype)
    params = torch.randn((2, 5, 2 * prep.param_dim), device="cuda", dtype=dtype)
    bc = torch.randn((2, 5, 2, 4, 3), device="cuda", dtype=dtype)

    with torch.no_grad():
        got = prep(value, params, bc)
        expect = prep_cute(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.allclose(got.B, expect.B, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.C, expect.C, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.M, expect.M, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.K, expect.K, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cute_scanprep_backend_matches_reference_forward() -> None:
    torch.manual_seed(0)
    ref = SLinOSSScanPrep(
        n_heads=3,
        d_state=5,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
        device="cuda",
    )
    cute = SLinOSSScanPrep(
        n_heads=3,
        d_state=5,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
        device="cuda",
        backend=CuteScanPrepBackend(),
    )
    cute.load_state_dict(ref.state_dict())

    value = torch.randn((2, 7, 12), device="cuda", dtype=torch.float32)
    params = torch.randn((2, 7, 3 * ref.param_dim), device="cuda", dtype=torch.float32)
    bc = torch.randn((2, 7, 3, 4, 5), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        got = cute(value, params, bc)
        expect = ref(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.allclose(got.B, expect.B, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.C, expect.C, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.M, expect.M, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.K, expect.K, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cute_scanprep_backend_matches_reference_forward_with_noncontiguous_scales() -> (
    None
):
    torch.manual_seed(0)
    ref = SLinOSSScanPrep(
        n_heads=3,
        d_state=5,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
        device="cuda",
    )
    cute = SLinOSSScanPrep(
        n_heads=3,
        d_state=5,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        theta_bound=math.pi / 2.0,
        k_max=0.25,
        device="cuda",
        backend=CuteScanPrepBackend(),
    )
    cute.load_state_dict(ref.state_dict())
    assert ref.b_scale is not None and ref.c_scale is not None
    cute.b_scale = nn.Parameter(_make_noncontiguous_clone(ref.b_scale.detach()))
    cute.c_scale = nn.Parameter(_make_noncontiguous_clone(ref.c_scale.detach()))
    assert cute.b_scale is not None and cute.c_scale is not None
    assert not cast(torch.Tensor, cute.b_scale).is_contiguous()
    assert not cast(torch.Tensor, cute.c_scale).is_contiguous()

    value = torch.randn((2, 7, 12), device="cuda", dtype=torch.float32)
    params = torch.randn((2, 7, 3 * ref.param_dim), device="cuda", dtype=torch.float32)
    bc = torch.randn((2, 7, 3, 4, 5), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        got = cute(value, params, bc)
        expect = ref(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.allclose(got.B, expect.B, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.C, expect.C, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.M, expect.M, atol=1e-6, rtol=1e-6)
    assert torch.allclose(got.K, expect.K, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cute_scanprep_backend_matches_reference_gradients() -> None:
    torch.manual_seed(1)
    ref = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        device="cuda",
    )
    cute = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    )
    cute.load_state_dict(ref.state_dict())

    value_ref = torch.randn(
        (2, 5, 8), device="cuda", dtype=torch.float32, requires_grad=True
    )
    params_ref = torch.randn(
        (2, 5, 2 * ref.param_dim),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bc_ref = torch.randn(
        (2, 5, 2, 4, 3), device="cuda", dtype=torch.float32, requires_grad=True
    )
    value_cute = value_ref.detach().clone().requires_grad_(True)
    params_cute = params_ref.detach().clone().requires_grad_(True)
    bc_cute = bc_ref.detach().clone().requires_grad_(True)

    out_ref = ref(value_ref, params_ref, bc_ref)
    out_cute = cute(value_cute, params_cute, bc_cute)

    g_u = torch.randn_like(out_ref.U)
    g_m = torch.randn_like(out_ref.M)
    g_k = torch.randn_like(out_ref.K)
    g_b = torch.randn_like(out_ref.B)
    g_c = torch.randn_like(out_ref.C)
    loss_ref = (
        (out_ref.U * g_u).sum()
        + (out_ref.M * g_m).sum()
        + (out_ref.K * g_k).sum()
        + (out_ref.B * g_b).sum()
        + (out_ref.C * g_c).sum()
    )
    loss_cute = (
        (out_cute.U * g_u).sum()
        + (out_cute.M * g_m).sum()
        + (out_cute.K * g_k).sum()
        + (out_cute.B * g_b).sum()
        + (out_cute.C * g_c).sum()
    )
    loss_ref.backward()
    loss_cute.backward()
    grad_atol = 5e-5
    grad_rtol = 5e-5

    assert value_ref.grad is not None and value_cute.grad is not None
    assert params_ref.grad is not None and params_cute.grad is not None
    assert bc_ref.grad is not None and bc_cute.grad is not None
    assert torch.allclose(
        value_cute.grad, value_ref.grad, atol=grad_atol, rtol=grad_rtol
    )
    assert torch.allclose(
        params_cute.grad, params_ref.grad, atol=grad_atol, rtol=grad_rtol
    )
    assert torch.allclose(bc_cute.grad, bc_ref.grad, atol=grad_atol, rtol=grad_rtol)

    names = (
        "dt_bias",
        "gamma_bias",
        "omega_bias",
        "mix_r_bias",
        "mix_theta_bias",
        "mix_k_prev_bias",
        "mix_k_curr_bias",
        "b_scale",
        "c_scale",
    )
    for name in names:
        ref_grad = getattr(ref, name).grad
        cute_grad = getattr(cute, name).grad
        assert ref_grad is not None
        assert cute_grad is not None
        assert torch.allclose(
            cast(torch.Tensor, cute_grad),
            cast(torch.Tensor, ref_grad),
            atol=grad_atol,
            rtol=grad_rtol,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cute_scanprep_backend_matches_reference_gradients_with_noncontiguous_scales() -> (
    None
):
    torch.manual_seed(1)
    ref = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        device="cuda",
    )
    cute = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    )
    cute.load_state_dict(ref.state_dict())
    assert ref.b_scale is not None and ref.c_scale is not None
    cute.b_scale = nn.Parameter(_make_noncontiguous_clone(ref.b_scale.detach()))
    cute.c_scale = nn.Parameter(_make_noncontiguous_clone(ref.c_scale.detach()))
    assert cute.b_scale is not None and cute.c_scale is not None
    assert not cast(torch.Tensor, cute.b_scale).is_contiguous()
    assert not cast(torch.Tensor, cute.c_scale).is_contiguous()

    value_ref = torch.randn(
        (2, 5, 8), device="cuda", dtype=torch.float32, requires_grad=True
    )
    params_ref = torch.randn(
        (2, 5, 2 * ref.param_dim),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    bc_ref = torch.randn(
        (2, 5, 2, 4, 3), device="cuda", dtype=torch.float32, requires_grad=True
    )
    value_cute = value_ref.detach().clone().requires_grad_(True)
    params_cute = params_ref.detach().clone().requires_grad_(True)
    bc_cute = bc_ref.detach().clone().requires_grad_(True)

    out_ref = ref(value_ref, params_ref, bc_ref)
    out_cute = cute(value_cute, params_cute, bc_cute)

    g_u = torch.randn_like(out_ref.U)
    g_m = torch.randn_like(out_ref.M)
    g_k = torch.randn_like(out_ref.K)
    g_b = torch.randn_like(out_ref.B)
    g_c = torch.randn_like(out_ref.C)
    loss_ref = (
        (out_ref.U * g_u).sum()
        + (out_ref.M * g_m).sum()
        + (out_ref.K * g_k).sum()
        + (out_ref.B * g_b).sum()
        + (out_ref.C * g_c).sum()
    )
    loss_cute = (
        (out_cute.U * g_u).sum()
        + (out_cute.M * g_m).sum()
        + (out_cute.K * g_k).sum()
        + (out_cute.B * g_b).sum()
        + (out_cute.C * g_c).sum()
    )
    loss_ref.backward()
    loss_cute.backward()
    grad_atol = 5e-5
    grad_rtol = 5e-5

    assert value_ref.grad is not None and value_cute.grad is not None
    assert params_ref.grad is not None and params_cute.grad is not None
    assert bc_ref.grad is not None and bc_cute.grad is not None
    assert torch.allclose(
        value_cute.grad, value_ref.grad, atol=grad_atol, rtol=grad_rtol
    )
    assert torch.allclose(
        params_cute.grad, params_ref.grad, atol=grad_atol, rtol=grad_rtol
    )
    assert torch.allclose(bc_cute.grad, bc_ref.grad, atol=grad_atol, rtol=grad_rtol)

    names = (
        "dt_bias",
        "gamma_bias",
        "omega_bias",
        "mix_r_bias",
        "mix_theta_bias",
        "mix_k_prev_bias",
        "mix_k_curr_bias",
        "b_scale",
        "c_scale",
    )
    for name in names:
        ref_grad = getattr(ref, name).grad
        cute_grad = getattr(cute, name).grad
        assert ref_grad is not None
        assert cute_grad is not None
        assert torch.allclose(
            cast(torch.Tensor, cute_grad),
            cast(torch.Tensor, ref_grad),
            atol=grad_atol,
            rtol=grad_rtol,
        )


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
    y_backend, next_state = cast(
        tuple[torch.Tensor, ScanState],
        backend(
            inputs,
            chunk_size=chunk_size,
            state=prev_state,
            return_state=True,
        ),
    )
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
