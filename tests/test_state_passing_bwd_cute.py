from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import pytest
import torch
import cutlass.cute as cute

import slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing as state_passing_bwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (
    compile_state_passing_bwd_kernel,
    state_passing_bwd_cute,
)


def _as_complex_pairs(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Expected even trailing dimension. Got {tuple(x.shape)}.")
    return torch.view_as_complex(
        x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2).contiguous()
    )


def _pack_complex_pairs(z: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(z).reshape(*z.shape[:-1], z.shape[-1] * 2).contiguous()


def _state_passing_autograd(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inc_c = _as_complex_pairs(inc)
    m_c = torch.view_as_complex(m_chunk.contiguous())
    z = _as_complex_pairs(initial_states)

    starts: list[torch.Tensor] = []
    for c in range(int(inc.shape[2])):
        starts.append(z)
        z = m_c[:, :, c].unsqueeze(-1).unsqueeze(-1) * z + inc_c[:, :, c]

    chunk_starts = _pack_complex_pairs(torch.stack(starts, dim=2))
    final_state = _pack_complex_pairs(z)
    return chunk_starts, final_state


def _state_passing_backward_reference(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    initial_states: torch.Tensor,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inc_ref = inc.detach().clone().requires_grad_(True)
    m_ref = m_chunk.detach().clone().requires_grad_(True)
    initial_ref = initial_states.detach().clone().requires_grad_(True)
    chunk_starts, final_state = _state_passing_autograd(inc_ref, m_ref, initial_ref)
    loss = (chunk_starts * d_chunk_starts).sum() + (final_state * d_final).sum()
    d_inc, d_m_chunk, d_initial = torch.autograd.grad(
        loss,
        (inc_ref, m_ref, initial_ref),
        retain_graph=False,
    )
    return (
        d_inc.detach().to(dtype=torch.float32).contiguous(),
        d_m_chunk.detach().to(dtype=torch.float32).contiguous(),
        d_initial.detach().to(dtype=torch.float32).contiguous(),
    )


def _make_inputs(
    *,
    batch: int,
    heads: int,
    chunks: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inc = torch.randn(
        (batch, heads, chunks, P, 2 * N), device=device, dtype=torch.float32
    )
    radius = 0.6 + 0.35 * torch.rand((batch, heads, chunks), device=device)
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, chunks), device=device
    ) - math.pi
    m_chunk = (
        torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    )
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )
    return inc, m_chunk, initial_states


def _assert_state_passing_bwd_close(
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    want: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    atol_by_idx = (1e-6, 3e-4, 1e-6)
    for got_tensor, want_tensor, atol in zip(got, want, atol_by_idx, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_kernel_matches_wrapper_and_reference() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    chunk_starts_f32 = chunk_starts.detach().to(dtype=torch.float32).contiguous()
    d_chunk_starts = torch.randn_like(chunk_starts_f32)
    d_final = torch.randn_like(final_state, dtype=torch.float32)
    want = _state_passing_backward_reference(
        inc,
        m_chunk,
        initial,
        d_chunk_starts,
        d_final,
    )

    got_public = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        state_passing_bwd_cute(
            chunk_starts_f32,
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
        ),
    )

    compiled = cast(
        tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None]],
        compile_state_passing_bwd_kernel(
            chunk_starts_f32,
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launcher=True,
        ),
    )
    (_compiled, d_inc, d_m_chunk, d_initial, launch) = compiled
    launch()

    got_compiled = (
        d_inc.to(dtype=torch.float32).contiguous(),
        d_m_chunk.to(dtype=torch.float32).contiguous(),
        d_initial.to(dtype=torch.float32).contiguous(),
    )

    for got_tensor, want_tensor in zip(got_compiled, got_public, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)
    _assert_state_passing_bwd_close(got_public, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_kernel_relaunch_is_idempotent() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    chunk_starts_f32 = chunk_starts.detach().to(dtype=torch.float32).contiguous()
    d_chunk_starts = torch.randn_like(chunk_starts_f32)
    d_final = torch.randn_like(final_state, dtype=torch.float32)
    want = _state_passing_backward_reference(
        inc,
        m_chunk,
        initial,
        d_chunk_starts,
        d_final,
    )

    compiled = cast(
        tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, Callable[[], None]],
        compile_state_passing_bwd_kernel(
            chunk_starts_f32,
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launcher=True,
        ),
    )
    (_compiled, d_inc, d_m_chunk, d_initial, launch) = compiled

    launch()
    first = (
        d_inc.clone(),
        d_m_chunk.clone(),
        d_initial.clone(),
    )
    launch()
    second = (
        d_inc.to(dtype=torch.float32).contiguous(),
        d_m_chunk.to(dtype=torch.float32).contiguous(),
        d_initial.to(dtype=torch.float32).contiguous(),
    )

    for got_tensor, want_tensor in zip(second, first, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)
    _assert_state_passing_bwd_close(second, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_reuses_cached_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    chunk_starts_f32 = chunk_starts.detach().to(dtype=torch.float32).contiguous()
    d_chunk_starts = torch.randn_like(chunk_starts_f32)
    d_final = torch.randn_like(final_state, dtype=torch.float32)

    state_passing_bwd_mod._COMPILED_CACHE.clear()
    compile_state_passing_bwd_kernel(
        chunk_starts_f32,
        m_chunk.detach(),
        d_chunk_starts=d_chunk_starts.detach(),
        d_final=d_final.detach(),
    )

    def _unexpected_compile(*args, **kwargs):
        raise AssertionError("unexpected recompilation on cache hit")

    monkeypatch.setattr(state_passing_bwd_mod.cute, "compile", _unexpected_compile)
    compile_state_passing_bwd_kernel(
        chunk_starts_f32,
        m_chunk.detach(),
        d_chunk_starts=d_chunk_starts.detach(),
        d_final=d_final.detach(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_enables_tvm_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    inc, m_chunk, initial = _make_inputs(
        batch=2,
        heads=2,
        chunks=12,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    chunk_starts, final_state = _state_passing_autograd(inc, m_chunk, initial)
    chunk_starts_f32 = chunk_starts.detach().to(dtype=torch.float32).contiguous()
    d_chunk_starts = torch.randn_like(chunk_starts_f32)
    d_final = torch.randn_like(final_state, dtype=torch.float32)

    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", wrapped_compile)
    state_passing_bwd_mod._COMPILED_CACHE.clear()
    try:
        compile_state_passing_bwd_kernel(
            chunk_starts_f32,
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
        )
    finally:
        state_passing_bwd_mod._COMPILED_CACHE.clear()

    assert compile_options == ["--enable-tvm-ffi"]
    torch.cuda.synchronize()
