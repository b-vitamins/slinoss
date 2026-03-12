from __future__ import annotations

from collections.abc import Callable
from typing import cast

import math

import pytest
import torch

import slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing as state_passing_bwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import (
    compile_state_passing_bwd_kernels,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_kernels_matches_wrapper() -> None:
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

    got_public = state_passing_bwd_cute(
        chunk_starts_f32,
        m_chunk.detach(),
        d_chunk_starts=d_chunk_starts.detach(),
        d_final=d_final.detach(),
    )

    compiled = cast(
        tuple[
            object,
            object,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Callable[[], None],
            Callable[[], None],
        ],
        compile_state_passing_bwd_kernels(
            chunk_starts_f32,
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launchers=True,
        ),
    )
    (
        _compiled_state,
        _compiled_m,
        d_inc,
        d_m_chunk,
        d_initial,
        launch_sequential,
        _launch_overlapped,
    ) = compiled
    launch_sequential()

    got_compiled = (
        d_inc.to(dtype=torch.float32).contiguous(),
        d_m_chunk.to(dtype=torch.float32).contiguous(),
        d_initial.to(dtype=torch.float32).contiguous(),
    )

    for got_tensor, want_tensor in zip(got_compiled, got_public, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_overlapped_matches_sequential() -> None:
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

    seq_bundle = cast(
        tuple[
            object,
            object,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Callable[[], None],
            Callable[[], None],
        ],
        compile_state_passing_bwd_kernels(
            chunk_starts_f32.clone(),
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launchers=True,
        ),
    )
    ov_bundle = cast(
        tuple[
            object,
            object,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Callable[[], None],
            Callable[[], None],
        ],
        compile_state_passing_bwd_kernels(
            chunk_starts_f32.clone(),
            m_chunk.detach(),
            d_chunk_starts=d_chunk_starts.detach(),
            d_final=d_final.detach(),
            return_launchers=True,
        ),
    )

    seq_launch = seq_bundle[-2]
    ov_launch = ov_bundle[-1]
    seq_launch()
    ov_launch()

    seq_public = tuple(t.to(dtype=torch.float32).contiguous() for t in seq_bundle[2:5])
    ov_public = tuple(t.to(dtype=torch.float32).contiguous() for t in ov_bundle[2:5])
    for got_tensor, want_tensor in zip(ov_public, seq_public, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compile_state_passing_bwd_reuses_cached_executors(
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
    compile_state_passing_bwd_kernels(
        chunk_starts_f32,
        m_chunk.detach(),
        d_chunk_starts=d_chunk_starts.detach(),
        d_final=d_final.detach(),
    )

    def _unexpected_compile(*args, **kwargs):
        raise AssertionError("unexpected recompilation on cache hit")

    monkeypatch.setattr(state_passing_bwd_mod.cute, "compile", _unexpected_compile)
    compile_state_passing_bwd_kernels(
        chunk_starts_f32,
        m_chunk.detach(),
        d_chunk_starts=d_chunk_starts.detach(),
        d_final=d_final.detach(),
    )
    torch.cuda.synchronize()
