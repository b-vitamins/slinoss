from __future__ import annotations

from typing import cast
import math

import pytest
import torch

import slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan as chunk_scan_bwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (
    chunk_scan_bwd_cute,
    compile_chunk_scan_bwd_kernels,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment, state_passing


def _public_from_chunked(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    return x.reshape(B, H, C * L, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


def _fold_chunk_boundary_carries(x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    if int(out.shape[2]) > 1:
        out[:, :, :-1, -1, :] = out[:, :, :-1, -1, :] + x_prev[:, :, 1:, :]
    return out


def _public_from_param_scan(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, S, L, F = map(int, x.shape)
    assert S == 1
    return (
        x[:, :, :, 0, :, :]
        .reshape(B, H, C * L, F)[:, :, :T, :]
        .to(dtype=torch.float32)
        .contiguous()
    )


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    dKprev_public = _public_from_param_scan(dKprev, T=T)
    dKcurr_public = _public_from_param_scan(dKcurr, T=T)
    return torch.stack((dKprev_public, dKcurr_public), dim=3).contiguous()


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
    dZ0 = compiled[5]
    dU = compiled[6]
    dB = compiled[7]
    dU_prev = compiled[8]
    dB_prev = compiled[9]
    dC = compiled[11]
    dM = compiled[13]
    dKprev = compiled[14]
    dKcurr = compiled[15]
    launch_sequential = compiled[-2]
    launch_sequential()

    dU_public = _fold_chunk_boundary_carries(dU, dU_prev)
    dB_public = _fold_chunk_boundary_carries(dB, dB_prev)

    got_compiled = (
        _public_from_chunked(dU_public, T=T),
        _public_from_param_scan(dM, T=T),
        _public_dk_from_parts(dKprev, dKcurr, T=T),
        _public_from_chunked(dB_public, T=T),
        _public_from_chunked(dC, T=T),
        dZ0.to(dtype=torch.float32).contiguous(),
        dB_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
        dU_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
    )

    atol_by_slot = (
        0.0,
        5e-7,
        2e-7,
        2e-7,
        0.0,
        0.0,
        2e-7,
        0.0,
    )
    for got_tensor, want_tensor, atol in zip(
        got_compiled, got_public, atol_by_slot, strict=True
    ):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_overlapped_matches_sequential() -> None:
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

    seq_bundle = compile_chunk_scan_bwd_kernels(
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
    ov_bundle = compile_chunk_scan_bwd_kernels(
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

    seq_launch = seq_bundle[-2]
    ov_launch = ov_bundle[-1]
    seq_launch()
    ov_launch()

    def public_from_bundle(
        bundle: tuple[object, ...],
    ) -> tuple[torch.Tensor, ...]:
        dZ0 = cast(torch.Tensor, bundle[5])
        dU = cast(torch.Tensor, bundle[6])
        dB = cast(torch.Tensor, bundle[7])
        dU_prev = cast(torch.Tensor, bundle[8])
        dB_prev = cast(torch.Tensor, bundle[9])
        dC = cast(torch.Tensor, bundle[11])
        dM = cast(torch.Tensor, bundle[13])
        dKprev = cast(torch.Tensor, bundle[14])
        dKcurr = cast(torch.Tensor, bundle[15])

        dU_public = _fold_chunk_boundary_carries(dU, dU_prev)
        dB_public = _fold_chunk_boundary_carries(dB, dB_prev)
        return (
            _public_from_chunked(dU_public, T=T),
            _public_from_param_scan(dM, T=T),
            _public_dk_from_parts(dKprev, dKcurr, T=T),
            _public_from_chunked(dB_public, T=T),
            _public_from_chunked(dC, T=T),
            dZ0.to(dtype=torch.float32).contiguous(),
            dB_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
            dU_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
        )

    seq_public = public_from_bundle(seq_bundle)
    ov_public = public_from_bundle(ov_bundle)
    atol_by_slot = (
        0.0,
        5e-7,
        2e-7,
        2e-7,
        0.0,
        0.0,
        2e-7,
        0.0,
    )
    for got_tensor, want_tensor, atol in zip(
        ov_public, seq_public, atol_by_slot, strict=True
    ):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_compile_entrypoint_reuses_cached_executors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    chunk_scan_bwd_mod._COMPILED_CACHE.clear()
    compile_chunk_scan_bwd_kernels(
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

    def _unexpected_compile(*args, **kwargs):
        raise AssertionError("unexpected recompilation on cache hit")

    monkeypatch.setattr(chunk_scan_bwd_mod.cute, "compile", _unexpected_compile)
    compile_chunk_scan_bwd_kernels(
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
    torch.cuda.synchronize()
