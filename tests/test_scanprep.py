from __future__ import annotations

import math
import warnings
from typing import Any, cast

import cutlass.cute as cute
import pytest
import torch

from slinoss.layers import (
    AutoScanPrepBackend,
    CuteScanPrepBackend,
    ReferenceScanBackend,
    ReferenceScanPrepBackend,
    SLinOSSScanPrep,
    ScanInputs,
    ScanState,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
)
import slinoss.ops.scanprep.cute.bwd as scanprep_bwd_mod
import slinoss.ops.scanprep.cute.fwd as scanprep_fwd_mod
from slinoss.ops.scanprep.cute.bwd import scanprep_bwd
from slinoss.ops.scanprep.cute.common import assumed_align
from slinoss.ops.scanprep.cute.fwd import scanprep_fwd_cute, scanprep_fwd_cute_with_aux
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
        "d_state": prep.d_state,
        "d_head": prep.d_head,
        "dt_min": prep.dt_min,
        "dt_max": prep.dt_max,
        "omega_min": prep.omega_min,
        "zeta_max": prep.zeta_max,
        "r_min": prep.r_min,
        "r_max": prep.r_max,
        "eps": prep.eps,
        "dt_bias": _maybe_detach(prep.dt_bias),
        "zeta_bias": _maybe_detach(prep.zeta_bias),
        "omega_mod_bias": _maybe_detach(prep.omega_mod_bias),
        "omega_natural_bias": _maybe_detach(prep.omega_natural_bias),
        "mix_r_bias": _maybe_detach(prep.mix_r_bias),
        "omega_sign": _maybe_detach(cast(torch.Tensor, prep.omega_sign)),
    }


def _canonical_bc(prep: SLinOSSScanPrep, bc_amp: torch.Tensor) -> torch.Tensor:
    return prep._parameterize_scan_bc_rows(bc_amp)


def _make_scanprep_cuda_fixture(
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[SLinOSSScanPrep, torch.Tensor, torch.Tensor, torch.Tensor]:
    prep = SLinOSSScanPrep(
        n_heads=2,
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
        (2, 5, prep.n_heads, prep.bc_param_rows, prep.d_state),
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
        scanprep_fwd_mod.cute.runtime, "make_fake_compact_tensor", fake_compact
    )
    monkeypatch.setattr(scanprep_fwd_mod.cute.runtime, "make_fake_tensor", fake_tensor)

    result = scanprep_fwd_mod.make_fake_tensor_arg(tensor)

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
        scanprep_bwd_mod.cute.runtime, "make_fake_compact_tensor", fake_compact
    )
    monkeypatch.setattr(scanprep_bwd_mod.cute.runtime, "make_fake_tensor", fake_tensor)

    result = scanprep_bwd_mod.make_fake_tensor_arg(
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
    scanprep_fwd_mod._SCANPREP_DUMMY_COEFF_AUX_CACHE.clear()

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
    assert scanprep_fwd_mod._SCANPREP_DUMMY_COEFF_AUX_CACHE == {}


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
    with torch.no_grad():
        U, M, K, B, C, coeff_aux = scanprep_fwd_cute_with_aux(
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
        scanprep_bwd(
            bc=bc,
            coeff_aux=coeff_aux,
            dU=torch.randn_like(U),
            dM=torch.randn_like(M),
            dK=torch.randn_like(K),
            dB=torch.randn_like(B),
            dC=torch.randn_like(C),
            n_heads=prep.n_heads,
            d_head=prep.d_head,
            d_state=prep.d_state,
            value_dtype=value.dtype,
            params_dtype=params.dtype,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            omega_min=prep.omega_min,
            zeta_max=prep.zeta_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            dt_bias=prep.dt_bias.detach(),
            omega_natural_bias=prep.omega_natural_bias.detach(),
            omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
        )

    assert compile_calls == []
    assert scanprep_bwd_mod._SCANPREP_BWD_CACHE == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_cached_path_stays_capture_safe(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    with torch.no_grad():
        U, M, K, B, C, coeff_aux = scanprep_fwd_cute_with_aux(
            value,
            params,
            bc,
            **_scanprep_runtime_kwargs(prep, detach=True),
        )
        scanprep_bwd(
            bc=bc,
            coeff_aux=coeff_aux,
            dU=torch.randn_like(U),
            dM=torch.randn_like(M),
            dK=torch.randn_like(K),
            dB=torch.randn_like(B),
            dC=torch.randn_like(C),
            n_heads=prep.n_heads,
            d_head=prep.d_head,
            d_state=prep.d_state,
            value_dtype=value.dtype,
            params_dtype=params.dtype,
            dt_min=prep.dt_min,
            dt_max=prep.dt_max,
            omega_min=prep.omega_min,
            zeta_max=prep.zeta_max,
            r_min=prep.r_min,
            r_max=prep.r_max,
            eps=prep.eps,
            dt_bias=prep.dt_bias.detach(),
            omega_natural_bias=prep.omega_natural_bias.detach(),
            omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
        )

    compile_calls: list[bool] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_calls.append(True)
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(cute, "compile", wrapped_compile)

    scanprep_bwd(
        bc=bc,
        coeff_aux=coeff_aux,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        n_heads=prep.n_heads,
        d_head=prep.d_head,
        d_state=prep.d_state,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        dt_min=prep.dt_min,
        dt_max=prep.dt_max,
        omega_min=prep.omega_min,
        zeta_max=prep.zeta_max,
        r_min=prep.r_min,
        r_max=prep.r_max,
        eps=prep.eps,
        dt_bias=prep.dt_bias.detach(),
        omega_natural_bias=prep.omega_natural_bias.detach(),
        omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
    )

    assert compile_calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_compile_enables_tvm_ffi(monkeypatch) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    with torch.no_grad():
        U, M, K, B, C, coeff_aux = scanprep_fwd_cute_with_aux(
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

    scanprep_bwd(
        bc=bc,
        coeff_aux=coeff_aux,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        n_heads=prep.n_heads,
        d_head=prep.d_head,
        d_state=prep.d_state,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        dt_min=prep.dt_min,
        dt_max=prep.dt_max,
        omega_min=prep.omega_min,
        zeta_max=prep.zeta_max,
        r_min=prep.r_min,
        r_max=prep.r_max,
        eps=prep.eps,
        dt_bias=prep.dt_bias.detach(),
        omega_natural_bias=prep.omega_natural_bias.detach(),
        omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
    )

    assert compile_options == ["--enable-tvm-ffi"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scanprep_bwd_reuses_compiled_executor_across_batch_time_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    prep, value, params, bc_amp = _make_scanprep_cuda_fixture()
    bc = _canonical_bc(prep, bc_amp)
    with torch.no_grad():
        U, M, K, B, C, coeff_aux = scanprep_fwd_cute_with_aux(
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
    with torch.no_grad():
        U_alt, M_alt, K_alt, B_alt, C_alt, coeff_aux_alt = scanprep_fwd_cute_with_aux(
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

    scanprep_bwd(
        bc=bc,
        coeff_aux=coeff_aux,
        dU=torch.randn_like(U),
        dM=torch.randn_like(M),
        dK=torch.randn_like(K),
        dB=torch.randn_like(B),
        dC=torch.randn_like(C),
        n_heads=prep.n_heads,
        d_head=prep.d_head,
        d_state=prep.d_state,
        value_dtype=value.dtype,
        params_dtype=params.dtype,
        dt_min=prep.dt_min,
        dt_max=prep.dt_max,
        omega_min=prep.omega_min,
        zeta_max=prep.zeta_max,
        r_min=prep.r_min,
        r_max=prep.r_max,
        eps=prep.eps,
        dt_bias=prep.dt_bias.detach(),
        omega_natural_bias=prep.omega_natural_bias.detach(),
        omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
    )
    scanprep_bwd(
        bc=bc_alt,
        coeff_aux=coeff_aux_alt,
        dU=torch.randn_like(U_alt),
        dM=torch.randn_like(M_alt),
        dK=torch.randn_like(K_alt),
        dB=torch.randn_like(B_alt),
        dC=torch.randn_like(C_alt),
        n_heads=prep.n_heads,
        d_head=prep.d_head,
        d_state=prep.d_state,
        value_dtype=value_alt.dtype,
        params_dtype=params_alt.dtype,
        dt_min=prep.dt_min,
        dt_max=prep.dt_max,
        omega_min=prep.omega_min,
        zeta_max=prep.zeta_max,
        r_min=prep.r_min,
        r_max=prep.r_max,
        eps=prep.eps,
        dt_bias=prep.dt_bias.detach(),
        omega_natural_bias=prep.omega_natural_bias.detach(),
        omega_sign=cast(torch.Tensor, prep.omega_sign).detach(),
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


def test_scanprep_real_dtype_cast_preserves_complex_bc_base() -> None:
    prep = SLinOSSScanPrep(n_heads=2, d_state=3, d_head=4)
    before = prep.bc_complex_base.detach().clone()

    with warnings.catch_warnings(record=True) as caught:
        prep = prep.to(dtype=torch.float16)

    assert caught == []
    assert prep.bc_complex_base.is_complex()
    assert prep.bc_complex_base.dtype == torch.complex64
    torch.testing.assert_close(prep.bc_complex_base, before)


def test_scanprep_parameterized_bc_pairs_match_row_packing() -> None:
    torch.manual_seed(0)
    prep = SLinOSSScanPrep(n_heads=2, d_state=3, d_head=4)
    bc = torch.randn((2, 5, prep.n_heads, prep.bc_param_rows, prep.d_state))

    B_pairs, C_pairs = prep._parameterize_scan_bc_pairs(bc)
    bc_rows = prep._parameterize_scan_bc_rows(bc)

    torch.testing.assert_close(bc_rows[..., 0, :], B_pairs[..., 0])
    torch.testing.assert_close(bc_rows[..., 1, :], B_pairs[..., 1])
    torch.testing.assert_close(bc_rows[..., 2, :], C_pairs[..., 0])
    torch.testing.assert_close(bc_rows[..., 3, :], C_pairs[..., 1])


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
        "zeta_bias",
        "omega_mod_bias",
        "omega_natural_bias",
        "mix_r_bias",
        "bc_complex_base",
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
