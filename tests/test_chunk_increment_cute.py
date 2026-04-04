from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

import slinoss.ops.v2x2ssd.cute.kernels.fwd as v2x2ssd_fwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    chunk_increment_cute,
    compile_chunk_increment_cute,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment as reference_chunk_increment


@dataclass(frozen=True)
class Inputs:
    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    B_prev: torch.Tensor | None
    U_prev: torch.Tensor | None


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_inputs(
    *,
    b: int,
    h: int,
    T: int,
    N: int,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming: bool,
) -> Inputs:
    radius = 0.6 + 0.35 * torch.rand((b, h, T), device=device, dtype=torch.float32)
    angle = (2.0 * math.pi) * torch.rand(
        (b, h, T), device=device, dtype=torch.float32
    ) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(dtype=dtype).contiguous()

    K_complex = (
        torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(dtype=dtype).contiguous()

    U = torch.randn((b, h, T, P), device=device, dtype=dtype)
    B = torch.randn((b, h, T, 2 * N), device=device, dtype=dtype) * 0.1

    B_prev = None
    U_prev = None
    if streaming:
        b_prev = (
            torch.randn((b, h, N), device=device, dtype=torch.float32)
            + 1j * torch.randn((b, h, N), device=device, dtype=torch.float32)
        ) * 0.1
        B_prev = _pack_complex_pairs(b_prev, real_dtype=dtype)
        U_prev = torch.randn((b, h, P), device=device, dtype=dtype)

    return Inputs(U=U, M=M, K=K, B=B, B_prev=B_prev, U_prev=U_prev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=True,
    )

    inc_ref, m_ref = reference_chunk_increment(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        T=inputs.U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    inc_cute, m_cute = chunk_increment_cute(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        chunk_size=32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(inc_cute, inc_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(m_cute, m_ref, atol=2e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_cute_matches_reference_stage_mixer_regime() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=4,
        T=49,
        N=64,
        P=64,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=False,
    )

    inc_ref, m_ref = reference_chunk_increment(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=None,
        U_prev=None,
        T=inputs.U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    inc_cute, m_cute = chunk_increment_cute(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        chunk_size=32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(inc_cute, inc_ref, atol=2e-4, rtol=0.0)
    torch.testing.assert_close(m_cute, m_ref, atol=2e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_wrapper_reuses_cached_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=True,
    )

    compile_calls = 0
    real_compile = v2x2ssd_fwd_mod.cute.compile

    def _counting_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(v2x2ssd_fwd_mod.cute, "compile", _counting_compile)
    v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()
    try:
        inc_chunk_a, m_chunk_a = chunk_increment_cute(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        inc_chunk_b, m_chunk_b = chunk_increment_cute(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        assert compile_calls == 1
    finally:
        v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()

    assert inc_chunk_a.shape == inc_chunk_b.shape == (2, 2, 2, 16, 16)
    assert m_chunk_a.shape == m_chunk_b.shape == (2, 2, 2, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_chunk_increment_cute_avoids_runtime_launch_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=True,
    )

    def _unexpected_make_chunk_increment_runtime_artifacts(*args, **kwargs):
        raise AssertionError(
            "compile_chunk_increment_cute should not build runtime launch artifacts"
        )

    monkeypatch.setattr(
        v2x2ssd_fwd_mod,
        "_make_chunk_increment_runtime_artifacts",
        _unexpected_make_chunk_increment_runtime_artifacts,
    )
    v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()
    try:
        compiled = compile_chunk_increment_cute(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        assert compiled is not None
    finally:
        v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_wrapper_reuses_cache_for_same_padded_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs_a = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=device,
        streaming=True,
    )
    inputs_b = _make_inputs(
        b=2,
        h=2,
        T=60,
        N=8,
        P=16,
        dtype=torch.float32,
        device=device,
        streaming=True,
    )

    compile_calls = 0
    real_compile = v2x2ssd_fwd_mod.cute.compile

    def _counting_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(v2x2ssd_fwd_mod.cute, "compile", _counting_compile)
    v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()
    try:
        chunk_increment_cute(
            inputs_a.U,
            inputs_a.M,
            inputs_a.K,
            inputs_a.B,
            B_prev=inputs_a.B_prev,
            U_prev=inputs_a.U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        chunk_increment_cute(
            inputs_b.U,
            inputs_b.M,
            inputs_b.K,
            inputs_b.B,
            B_prev=inputs_b.B_prev,
            U_prev=inputs_b.U_prev,
            chunk_size=32,
            compute_dtype=torch.float32,
        )
        assert compile_calls == 1
    finally:
        v2x2ssd_fwd_mod._CHUNK_INCREMENT_CACHE.clear()
