from __future__ import annotations

from collections.abc import Callable
from typing import cast

import math

import pytest
import torch
import cutlass.cute as cute

import slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment as chunk_increment_bwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment import (
    chunk_increment_bwd_cute,
    compile_chunk_increment_bwd_kernels,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment as reference_chunk_increment


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
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

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, B_prev, U_prev


def _public_from_chunked(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    return x.reshape(B, H, C * L, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


def _public_from_param_scan(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    return x.reshape(B, H, C * L, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    dK = torch.stack((dKprev, dKcurr), dim=4)
    B, H, C, L, _, F = map(int, dK.shape)
    return (
        dK.reshape(B, H, C * L, 2, F)[:, :, :T, :, :]
        .to(dtype=torch.float32)
        .contiguous()
    )


def _fold_chunk_boundary_carries(x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    if int(x.shape[2]) > 1:
        x[:, :, :-1, -1, :].add_(x_prev[:, :, 1:, :])
    return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_cute_matches_autograd() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    U.requires_grad_(True)
    M.requires_grad_(True)
    K.requires_grad_(True)
    B.requires_grad_(True)
    B_prev.requires_grad_(True)
    U_prev.requires_grad_(True)

    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)
    loss = (inc * d_inc).sum() + (m_chunk * d_m_chunk).sum()

    dU_ref, dM_ref, dK_ref, dB_ref, dB_prev_ref, dU_prev_ref = torch.autograd.grad(
        loss,
        (U, M, K, B, B_prev, U_prev),
    )

    dU_cute, dM_cute, dK_cute, dB_cute, dB_prev_cute, dU_prev_cute = (
        chunk_increment_bwd_cute(
            U.detach(),
            M.detach(),
            K.detach(),
            B.detach(),
            d_inc=d_inc.detach(),
            d_m_chunk=d_m_chunk.detach(),
            chunk_size=32,
            B_prev=B_prev.detach(),
            U_prev=U_prev.detach(),
            compute_dtype=torch.float32,
        )
    )

    # The stage-native backward follows the same tensor-core contract as the
    # corresponding v3 kernels: fp16 transport, fp32 accumulation, and exact
    # public reassembly. The right correctness bar here is principled
    # low-precision agreement, not bitwise parity with the fp32 reference.
    rtol_h, atol_h = 2e-3, 5e-3

    torch.testing.assert_close(dU_cute, dU_ref, atol=atol_h, rtol=rtol_h)
    torch.testing.assert_close(dM_cute, dM_ref, atol=atol_h, rtol=rtol_h)
    torch.testing.assert_close(dK_cute, dK_ref, atol=atol_h, rtol=rtol_h)
    torch.testing.assert_close(dB_cute, dB_ref, atol=atol_h, rtol=rtol_h)
    torch.testing.assert_close(dB_prev_cute, dB_prev_ref, atol=atol_h, rtol=rtol_h)
    torch.testing.assert_close(dU_prev_cute, dU_prev_ref, atol=atol_h, rtol=rtol_h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_compile_entrypoint_matches_public_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)

    got_public = chunk_increment_bwd_cute(
        U.detach(),
        M.detach(),
        K.detach(),
        B.detach(),
        d_inc=d_inc.detach(),
        d_m_chunk=d_m_chunk.detach(),
        chunk_size=32,
        B_prev=B_prev.detach(),
        U_prev=U_prev.detach(),
        compute_dtype=torch.float32,
    )

    compiled = cast(
        tuple[
            object,
            object,
            object,
            object,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Callable[[], None],
        ],
        compile_chunk_increment_bwd_kernels(
            U.detach(),
            M.detach(),
            K.detach(),
            B.detach(),
            d_inc=d_inc.detach(),
            d_m_chunk=d_m_chunk.detach(),
            chunk_size=32,
            B_prev=B_prev.detach(),
            U_prev=U_prev.detach(),
            compute_dtype=torch.float32,
            return_launchers=True,
        ),
    )
    (
        _compiled_db,
        _compiled_du,
        _compiled_boundary,
        _compiled_param,
        dB,
        dU,
        dB_prev,
        dU_prev,
        _dMsum_part,
        _dMp0,
        dM,
        dKprev,
        dKcurr,
        launch,
    ) = compiled
    launch()

    got_compiled = (
        _public_from_chunked(_fold_chunk_boundary_carries(dU, dU_prev), T=U.shape[2]),
        _public_from_param_scan(dM, T=U.shape[2]),
        _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
        _public_from_chunked(_fold_chunk_boundary_carries(dB, dB_prev), T=U.shape[2]),
        dB_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
        dU_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
    )

    for got_tensor, want_tensor in zip(got_compiled, got_public, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_compile_entrypoint_reuses_cached_executors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)

    chunk_increment_bwd_mod._COMPILED_CACHE.clear()
    compile_chunk_increment_bwd_kernels(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=32,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    def _unexpected_compile(*args, **kwargs):
        raise AssertionError("unexpected recompilation on cache hit")

    monkeypatch.setattr(chunk_increment_bwd_mod.cute, "compile", _unexpected_compile)
    compile_chunk_increment_bwd_kernels(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=32,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_compile_entrypoint_enables_tvm_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)

    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", wrapped_compile)
    chunk_increment_bwd_mod._COMPILED_CACHE.clear()
    try:
        compile_chunk_increment_bwd_kernels(
            U,
            M,
            K,
            B,
            d_inc=d_inc,
            d_m_chunk=d_m_chunk,
            chunk_size=32,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )
    finally:
        chunk_increment_bwd_mod._COMPILED_CACHE.clear()

    assert compile_options == ["--enable-tvm-ffi"] * 5
