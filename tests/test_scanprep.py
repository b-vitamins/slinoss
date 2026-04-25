from __future__ import annotations

import math
from typing import Any, cast

import cutlass.cute as cute
import pytest
import torch

import slinoss.ops._cute_common as cute_common_mod
from slinoss.layers import (
    AutoScanPrepBackend,
    CuteScanPrepBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
    SLinOSSScanPrep,
    ScanInputs,
    ScanState,
)
from slinoss.ops.scanprep import (
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)
import slinoss.ops.scanprep.cute.kernels as scanprep_bwd_mod
import slinoss.ops.scanprep.cute.kernels as scanprep_fwd_mod
from slinoss.ops.scanprep.cute.kernels import (
    compile_scanprep_bwd_cute,
    compile_scanprep_fwd_cute,
    scanprep_bwd_cute,
    scanprep_fwd_cute,
)
from slinoss.ops.scanprep.cute.common import assumed_align
from slinoss.ops.scanprep.parameterization import PHASE_LIMIT
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


def _scanprep_runtime_kwargs(
    prep: SLinOSSScanPrep,
    *,
    detach: bool = False,
) -> dict[str, Any]:
    def _maybe_detach(x: torch.Tensor) -> torch.Tensor:
        return x.detach() if detach else x

    return {
        "n_heads": prep.n_heads,
        "bc_groups": prep.bc_groups,
        "d_state": prep.d_state,
        "d_head": prep.d_head,
        "dt_min": prep.dt_min,
        "dt_max": prep.dt_max,
        "theta_init_min": prep.theta_init_min,
        "theta_init_max": prep.theta_init_max,
        "theta_mod_scale": prep.theta_mod_scale,
        "alpha_min": prep.alpha_min,
        "alpha_max": prep.alpha_max,
        "r_min": prep.r_min,
        "r_max": prep.r_max,
        "eps": prep.eps,
        "dt_bias": _maybe_detach(prep.dt_bias),
        "alpha_bias": _maybe_detach(prep.alpha_bias),
        "theta_mod_bias": _maybe_detach(prep.theta_mod_bias),
        "theta_bias": _maybe_detach(prep.theta_bias),
        "theta_sign": _maybe_detach(cast(torch.Tensor, prep.theta_sign)),
    }


def _scanprep_bwd_runtime_kwargs(
    prep: SLinOSSScanPrep,
    params: torch.Tensor,
    *,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    detach: bool = False,
) -> dict[str, Any]:
    kwargs = _scanprep_runtime_kwargs(prep, detach=detach)
    kwargs.update(
        {
            "params": params.detach() if detach else params,
            "value_dtype": value_dtype,
            "params_dtype": params_dtype,
        }
    )
    return kwargs


def _canonical_bc(prep: SLinOSSScanPrep, bc_amp: torch.Tensor) -> torch.Tensor:
    return prep._parameterize_scan_bc_rows(bc_amp)


def _make_scanprep_cuda_fixture(
    *,
    dtype: torch.dtype = torch.float16,
    bc_groups: int | None = None,
) -> tuple[SLinOSSScanPrep, torch.Tensor, torch.Tensor, torch.Tensor]:
    prep = SLinOSSScanPrep(
        n_heads=2,
        bc_groups=bc_groups,
        d_state=3,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        device="cuda",
        backend=CuteScanPrepBackend(),
    ).to(dtype=dtype)
    value = torch.randn((2, 5, 8), device="cuda", dtype=dtype)
    params = torch.randn((2, 5, 2 * prep.param_dim), device="cuda", dtype=dtype)
    bc_amp = torch.randn(
        (2, 5, prep.bc_groups, prep.bc_param_rows, prep.d_state),
        device="cuda",
        dtype=dtype,
    )
    return prep, value, params, bc_amp


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


@pytest.mark.parametrize("r_min", (0.8, 0.7, 0.6))
def test_scanprep_low_r_min_init_stays_finite(r_min: float) -> None:
    prep = SLinOSSScanPrep(
        n_heads=32,
        bc_groups=1,
        d_state=64,
        d_head=64,
        dt_min=3e-2,
        dt_max=1e-1,
        dt_init_floor=3e-2,
        alpha_min=0.0,
        alpha_max=20.0,
        theta_init_min=0.2,
        theta_init_max=1.0,
        r_min=r_min,
        r_max=1.0,
        device="cpu",
    )
    assert torch.isfinite(prep.alpha_bias).all()

    zeros = torch.zeros((1, 1, prep.n_heads, prep.param_dim), dtype=torch.float32)
    coeffs = prep.coefficients(zeros)

    assert torch.isfinite(coeffs.dt).all()
    assert torch.isfinite(coeffs.r).all()
    assert torch.isfinite(coeffs.theta).all()
    assert torch.isfinite(coeffs.M).all()
    assert torch.isfinite(coeffs.K).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_scanprep_assumed_align_uses_actual_view_alignment(dtype: torch.dtype) -> None:
    pytest.importorskip("cutlass")

    base = torch.empty((257,), device="cuda", dtype=dtype)
    view = base[1:]
    assert view.is_contiguous()
    assert view.data_ptr() % 4 == 2

    assert assumed_align(view) == view.element_size()


def test_scanprep_fake_tensor_arg_prefers_compact_for_row_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.empty((2, 3, 4), dtype=torch.float16)
    calls: list[str] = []

    def fake_compact(*args, **kwargs):
        calls.append("compact")
        return ("compact", args, kwargs)

    def fake_tensor(*args, **kwargs):
        calls.append("tensor")
        return ("tensor", args, kwargs)

    monkeypatch.setattr(
        cute_common_mod.cute.runtime, "make_fake_compact_tensor", fake_compact
    )
    monkeypatch.setattr(cute_common_mod.cute.runtime, "make_fake_tensor", fake_tensor)

    result = cute_common_mod.make_fake_tensor_arg(tensor)

    assert calls == ["compact"]
    assert result[0] == "compact"


def test_scanprep_fake_tensor_arg_falls_back_for_noncompact_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.empty((2, 3, 4), dtype=torch.float16)
    calls: list[str] = []

    def fake_compact(*args, **kwargs):
        calls.append("compact")
        return ("compact", args, kwargs)

    def fake_tensor(*args, **kwargs):
        calls.append("tensor")
        return ("tensor", args, kwargs)

    monkeypatch.setattr(
        cute_common_mod.cute.runtime, "make_fake_compact_tensor", fake_compact
    )
    monkeypatch.setattr(cute_common_mod.cute.runtime, "make_fake_tensor", fake_tensor)

    result = cute_common_mod.make_fake_tensor_arg(
        tensor,
        shape=(2, 3, 4),
        stride=(12, 1, 3),
    )

    assert calls == ["tensor"]
    assert result[0] == "tensor"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_fwd_rejects_cold_cache_during_capture(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    scanprep_fwd_mod._SCANPREP_FWD_CACHE.clear()

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    with pytest.raises(RuntimeError, match="forward .*cold during CUDA graph capture"):
        with torch.no_grad():
            scanprep_fwd_cute(
                value,
                params,
                bc,
                **_scanprep_runtime_kwargs(prep, detach=True),
            )

    assert compile_calls == []
    assert scanprep_fwd_mod._SCANPREP_FWD_CACHE == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_fwd_cached_path_stays_capture_safe(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    scanprep_fwd_mod._SCANPREP_FWD_CACHE.clear()

    with torch.no_grad():
        scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    with torch.no_grad():
        scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    assert compile_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_fwd_compile_enables_tvm_ffi(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    scanprep_fwd_mod._SCANPREP_FWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    with torch.no_grad():
        scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    assert compile_options == ["--enable-tvm-ffi"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_scanprep_fwd_cute_enables_tvm_ffi(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    scanprep_fwd_mod._SCANPREP_FWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    compiled = compile_scanprep_fwd_cute(
        value,
        params,
        bc,
        **_scanprep_runtime_kwargs(prep, detach=True),
    )

    assert callable(compiled)
    assert compile_options == ["--enable-tvm-ffi"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_fwd_reuses_compiled_executor_across_batch_time_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    value_alt = torch.randn((3, 7, 8), device="cuda", dtype=value.dtype)
    params_alt = torch.randn(
        (3, 7, prep.n_heads * prep.param_dim),
        device="cuda",
        dtype=params.dtype,
    )
    bc_alt = _canonical_bc(
        prep,
        torch.randn(
            (3, 7, prep.n_heads, prep.bc_param_rows, prep.d_state),
            device="cuda",
            dtype=bc_amp.dtype,
        ),
    )

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    scanprep_fwd_mod._SCANPREP_FWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    with torch.no_grad():
        scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )
        scanprep_fwd_cute(
            value_alt,
            params_alt,
            bc_alt,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    assert compile_calls == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_rejects_cold_cache_during_capture(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )
    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )
    scanprep_bwd_mod._SCANPREP_BWD_CACHE.clear()

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    with pytest.raises(RuntimeError, match="backward .*cold during CUDA graph capture"):
        scanprep_bwd_cute(
            bc=bc,
            dU=torch.randn_like(U),
            dM=torch.randn_like(M),
            dK=torch.randn_like(K),
            dB=torch.randn_like(B),
            dC=torch.randn_like(C),
            **bwd_runtime_kwargs,
        )

    assert compile_calls == []
    assert scanprep_bwd_mod._SCANPREP_BWD_CACHE == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_cached_path_stays_capture_safe(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )
    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )
        scanprep_bwd_cute(
            bc=bc,
            dU=torch.randn_like(U),
            dM=torch.randn_like(M),
            dK=torch.randn_like(K),
            dB=torch.randn_like(B),
            dC=torch.randn_like(C),
            **bwd_runtime_kwargs,
        )

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    scanprep_bwd_cute(
        bc=bc,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        **bwd_runtime_kwargs,
    )

    assert compile_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_compile_enables_tvm_ffi(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )
    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    scanprep_bwd_mod._SCANPREP_BWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    scanprep_bwd_cute(
        bc=bc,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        **bwd_runtime_kwargs,
    )

    assert compile_options == ["--enable-tvm-ffi"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compile_scanprep_bwd_cute_enables_tvm_ffi(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )
    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    scanprep_bwd_mod._SCANPREP_BWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    compiled = compile_scanprep_bwd_cute(
        bc=bc,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        **bwd_runtime_kwargs,
    )

    assert callable(compiled)
    assert compile_options == ["--enable-tvm-ffi"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_reuses_compiled_executor_across_batch_time_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )
    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    value_alt = torch.randn((3, 7, 8), device="cuda", dtype=value.dtype)
    params_alt = torch.randn(
        (3, 7, prep.n_heads * prep.param_dim),
        device="cuda",
        dtype=params.dtype,
    )
    bc_alt = _canonical_bc(
        prep,
        torch.randn(
            (3, 7, prep.n_heads, prep.bc_param_rows, prep.d_state),
            device="cuda",
            dtype=bc_amp.dtype,
        ),
    )
    bwd_alt_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params_alt,
        value_dtype=value_alt.dtype,
        params_dtype=params_alt.dtype,
        detach=True,
    )
    with torch.no_grad():
        U_alt, M_alt, K_alt, B_alt, C_alt = scanprep_fwd_cute(
            value_alt,
            params_alt,
            bc_alt,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    scanprep_bwd_mod._SCANPREP_BWD_CACHE.clear()
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    scanprep_bwd_cute(
        bc=bc,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        **bwd_runtime_kwargs,
    )
    scanprep_bwd_cute(
        bc=bc_alt,
        dU=torch.randn_like(U_alt),
        dM=torch.randn_like(M_alt),
        dK=torch.randn_like(K_alt),
        dB=torch.randn_like(B_alt),
        dC=torch.randn_like(C_alt),
        **bwd_alt_runtime_kwargs,
    )

    assert compile_calls == [True]


def test_scanprep_coefficients_are_bounded_and_finite() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(n_heads=2, d_state=3, d_head=4)
    params = torch.randn((2, 9, prep.n_heads * prep.param_dim), dtype=torch.float32)

    out = prep.coefficients(params.view(2, 9, prep.n_heads, prep.param_dim))
    radius = torch.linalg.vector_norm(out.M, dim=-1)

    assert torch.isfinite(out.M).all()
    assert torch.isfinite(out.K).all()
    assert torch.isfinite(out.dt).all()
    assert torch.isfinite(out.r).all()
    assert torch.isfinite(out.theta).all()
    assert bool((out.dt >= prep.dt_min).all())
    assert bool((out.dt <= prep.dt_max).all())
    assert bool((out.r >= prep.r_min).all())
    assert bool((out.r <= prep.r_max).all())
    assert torch.allclose(radius, out.r, atol=1e-6, rtol=1e-6)


def test_scanprep_dt_is_token_conditioned_while_radius_and_phase_stay_fixed() -> None:
    prep = SLinOSSScanPrep(n_heads=1, d_state=2, d_head=4)
    params = torch.zeros((1, 2, prep.n_heads, prep.param_dim), dtype=torch.float32)
    params[0, 0, 0, 0] = -5.0
    params[0, 1, 0, 0] = 5.0

    out = prep.coefficients(params)

    dt0 = out.dt[0, 0, 0]
    dt1 = out.dt[0, 0, 1]
    r0 = out.r[0, 0, 0]
    r1 = out.r[0, 0, 1]
    theta0 = out.theta[0, 0, 0]
    theta1 = out.theta[0, 0, 1]

    assert not torch.isclose(dt0, dt1)
    torch.testing.assert_close(r0, r1, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(theta0, theta1, atol=1e-6, rtol=1e-6)


def test_scanprep_direct_polar_bc_pairs_follow_token_phase_rows() -> None:
    prep = SLinOSSScanPrep(n_heads=1, bc_groups=1, d_state=2, d_head=4)
    bc = torch.zeros((1, 1, prep.bc_groups, prep.bc_param_rows, prep.d_state))
    bc[..., 0, :] = 1.0
    bc[..., 1, :] = torch.tensor([0.75, -0.25])
    bc[..., 2, :] = 1.0
    bc[..., 3, :] = torch.tensor([-0.5, 0.5])

    b_pairs, c_pairs = prep._parameterize_scan_bc_pairs(bc)

    for lane, phase_logit in enumerate((0.75, -0.25)):
        b_direction = b_pairs[0, 0, 0, lane] / b_pairs[0, 0, 0, lane].norm()
        b_expected_phase = PHASE_LIMIT * math.tanh(phase_logit)
        b_expected = torch.tensor(
            [math.cos(b_expected_phase), math.sin(b_expected_phase)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(b_direction, b_expected, atol=1e-5, rtol=1e-5)

    for lane, phase_logit in enumerate((-0.5, 0.5)):
        c_direction = c_pairs[0, 0, 0, lane] / c_pairs[0, 0, 0, lane].norm()
        c_expected_phase = PHASE_LIMIT * math.tanh(phase_logit)
        c_expected = torch.tensor(
            [math.cos(c_expected_phase), math.sin(c_expected_phase)],
            dtype=torch.float32,
        )
        torch.testing.assert_close(c_direction, c_expected, atol=1e-5, rtol=1e-5)


def test_scanprep_parameterized_bc_pairs_match_row_packing() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(n_heads=2, d_state=3, d_head=4)
    bc = torch.randn((2, 5, prep.bc_groups, prep.bc_param_rows, prep.d_state))

    B_pairs, C_pairs = prep._parameterize_scan_bc_pairs(bc)
    bc_rows = prep._parameterize_scan_bc_rows(bc)

    torch.testing.assert_close(bc_rows[..., 0, :], B_pairs[..., 0])
    torch.testing.assert_close(bc_rows[..., 1, :], B_pairs[..., 1])
    torch.testing.assert_close(bc_rows[..., 2, :], C_pairs[..., 0])
    torch.testing.assert_close(bc_rows[..., 3, :], C_pairs[..., 1])


def test_scanprep_parameterized_bc_pairs_have_unit_complex_rms() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(n_heads=2, d_state=5, d_head=4)
    bc = torch.randn((2, 5, prep.bc_groups, prep.bc_param_rows, prep.d_state)) * 20.0

    B_pairs, C_pairs = prep._parameterize_scan_bc_pairs(bc)

    B_rms = B_pairs.square().sum(dim=-1, dtype=torch.float32).mean(dim=-1).sqrt()
    C_rms = C_pairs.square().sum(dim=-1, dtype=torch.float32).mean(dim=-1).sqrt()

    torch.testing.assert_close(B_rms, torch.ones_like(B_rms), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(C_rms, torch.ones_like(C_rms), atol=1e-5, rtol=1e-5)


def test_scanprep_reference_pack_produces_contiguous_bc() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=ReferenceScanPrepBackend(),
    )
    value = torch.randn((2, 5, prep.d_inner), dtype=torch.float32)
    params = torch.randn((2, 5, prep.n_heads * prep.param_dim), dtype=torch.float32)
    bc = torch.randn((2, 5, prep.bc_groups, prep.bc_param_rows, prep.d_state))

    out = prep(value, params, bc)

    assert out.B.is_contiguous()
    assert out.C.is_contiguous()


def test_scanprep_grouped_reference_contract_and_gradients() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(
        n_heads=4,
        bc_groups=2,
        d_state=3,
        d_head=5,
        backend=ReferenceScanPrepBackend(),
    )
    value = torch.randn((2, 6, prep.d_inner), dtype=torch.float32, requires_grad=True)
    params = torch.randn(
        (2, 6, prep.n_heads * prep.param_dim),
        dtype=torch.float32,
        requires_grad=True,
    )
    bc = torch.randn(
        (2, 6, prep.bc_groups, prep.bc_param_rows, prep.d_state),
        dtype=torch.float32,
        requires_grad=True,
    )

    out = prep(value, params, bc)

    assert tuple(out.U.shape) == (2, prep.n_heads, 6, prep.d_head)
    assert tuple(out.M.shape) == (2, prep.n_heads, 6, 2)
    assert tuple(out.K.shape) == (2, prep.n_heads, 6, 2, 2)
    assert tuple(out.B.shape) == (2, prep.bc_groups, 6, 2 * prep.d_state)
    assert tuple(out.C.shape) == (2, prep.bc_groups, 6, 2 * prep.d_state)

    loss = out.U.sum() + out.M.sum() + out.K.sum() + out.B.sum() + out.C.sum()
    loss.backward()

    assert value.grad is not None and tuple(value.grad.shape) == tuple(value.shape)
    assert params.grad is not None and tuple(params.grad.shape) == tuple(params.shape)
    assert bc.grad is not None and tuple(bc.grad.shape) == tuple(bc.shape)


def test_cute_scanprep_backend_requires_cuda() -> None:
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
    )
    value = torch.randn((2, 5, 8), dtype=torch.float32)
    params = torch.randn((2, 5, 2 * prep.param_dim), dtype=torch.float32)
    bc = torch.randn((2, 5, 2, prep.bc_param_rows, 3), dtype=torch.float32)

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
    bc = torch.randn((2, 5, 2, prep.bc_param_rows, 3), dtype=torch.float32)

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
    bc = torch.randn(
        (2, 5, 2, prep.bc_param_rows, 3), device="cuda", dtype=torch.float32
    )

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
    bc = torch.randn((2, 5, 2, prep.bc_param_rows, 3), device="cuda", dtype=dtype)

    with torch.no_grad():
        got = prep(value, params, bc)
        expect = prep_cute(value, params, bc)

    assert torch.equal(got.U, expect.U)
    assert torch.allclose(got.B, expect.B, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.C, expect.C, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.M, expect.M, atol=5e-3, rtol=5e-3)
    assert torch.allclose(got.K, expect.K, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_scanprep_matches_bc_outputs_to_u_dtype(dtype: torch.dtype) -> None:
    pytest.importorskip("cutlass")

    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    ).to(dtype=dtype)
    value = torch.randn((2, 5, 8), device="cuda", dtype=dtype)
    params = torch.randn((2, 5, 2 * prep.param_dim), device="cuda", dtype=dtype)
    bc = torch.randn(
        (2, 5, prep.n_heads, prep.bc_param_rows, prep.d_state),
        device="cuda",
        dtype=torch.float32,
    )

    with torch.no_grad():
        out = prep(value, params, bc)

    assert out.U.dtype == dtype
    assert out.B.dtype == dtype
    assert out.C.dtype == dtype


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
        device="cuda",
    )
    cute_prep = SLinOSSScanPrep(
        n_heads=3,
        d_state=5,
        d_head=4,
        dt_min=1e-3,
        dt_max=1e-1,
        r_min=0.8,
        r_max=0.98,
        device="cuda",
        backend=CuteScanPrepBackend(),
    )
    cute_prep.load_state_dict(ref.state_dict())

    value = torch.randn((2, 7, 12), device="cuda", dtype=torch.float32)
    params = torch.randn((2, 7, 3 * ref.param_dim), device="cuda", dtype=torch.float32)
    bc = torch.randn(
        (2, 7, 3, ref.bc_param_rows, 5), device="cuda", dtype=torch.float32
    )

    with torch.no_grad():
        got = cute_prep(value, params, bc)
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
        backend=ReferenceScanPrepBackend(),
        device="cuda",
    )
    cute_prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    )
    cute_prep.load_state_dict(ref.state_dict())

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
        (2, 5, 2, ref.bc_param_rows, 3),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    value_cute = value_ref.detach().clone().requires_grad_(True)
    params_cute = params_ref.detach().clone().requires_grad_(True)
    bc_cute = bc_ref.detach().clone().requires_grad_(True)

    out_ref = ref(value_ref, params_ref, bc_ref)
    out_cute = cute_prep(value_cute, params_cute, bc_cute)

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
        "alpha_bias",
        "theta_mod_bias",
        "theta_bias",
    )
    for name in names:
        ref_grad = getattr(ref, name).grad
        cute_grad = getattr(cute_prep, name).grad
        assert ref_grad is not None
        assert cute_grad is not None
        assert torch.allclose(
            cast(torch.Tensor, cute_grad),
            cast(torch.Tensor, ref_grad),
            atol=grad_atol,
            rtol=grad_rtol,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 required",
)
def test_cute_scanprep_backward_bf16_bc_pack_store_cast_regression() -> None:
    torch.manual_seed(1)
    prep = SLinOSSScanPrep(
        n_heads=2,
        d_state=3,
        d_head=4,
        backend=CuteScanPrepBackend(),
        device="cuda",
    )

    value = torch.randn(
        (2, 5, 8),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    params = torch.randn(
        (2, 5, 2 * prep.param_dim),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    bc = torch.randn(
        (2, 5, prep.bc_groups, prep.bc_param_rows, prep.d_state),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    out = prep(value, params, bc)
    loss = (
        out.U.to(dtype=torch.float32).square().mean()
        + out.M.to(dtype=torch.float32).square().mean()
        + out.K.to(dtype=torch.float32).square().mean()
        + out.B.to(dtype=torch.float32).square().mean()
        + out.C.to(dtype=torch.float32).square().mean()
    )
    loss.backward()

    assert value.grad is not None
    assert params.grad is not None
    assert bc.grad is not None
    assert value.grad.dtype == torch.bfloat16
    assert params.grad.dtype == torch.bfloat16
    assert bc.grad.dtype == torch.bfloat16
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(params.grad).all()
    assert torch.isfinite(bc.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 required",
)
def test_scanprep_bwd_normalizes_optional_grad_dtypes_to_contract() -> None:
    torch.manual_seed(7)
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture(dtype=torch.bfloat16)
    bc = _canonical_bc(prep, bc_amp)
    runtime_kwargs = _scanprep_runtime_kwargs(prep, detach=True)
    bwd_runtime_kwargs = _scanprep_bwd_runtime_kwargs(
        prep,
        params,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        detach=True,
    )

    with torch.no_grad():
        U, M, K, B, C = scanprep_fwd_cute(value, params, bc, **runtime_kwargs)

    # Deliberately pass mismatched optional-grad dtypes to ensure the runtime
    # argument boundary normalizes to the expected public-contract dtypes.
    dU = torch.randn_like(U, dtype=torch.float32)
    dM = torch.randn_like(M, dtype=torch.bfloat16)
    dK = torch.randn_like(K, dtype=torch.bfloat16)
    dB = torch.randn_like(B, dtype=torch.float32)
    dC = torch.randn_like(C, dtype=torch.float32)

    (
        value_grad,
        params_grad,
        bc_grad,
        _dt_bias_grad,
        _alpha_bias_grad,
        _theta_mod_bias_grad,
        _theta_bias_grad,
    ) = scanprep_bwd_cute(
        bc=bc,
        dU=dU,
        dM=dM,
        dK=dK,
        dB=dB,
        dC=dC,
        **bwd_runtime_kwargs,
    )

    assert value_grad.dtype == value.dtype
    assert params_grad.dtype == params.dtype
    assert bc_grad.dtype == bc.dtype


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
