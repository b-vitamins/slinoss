"""CuTe backward kernels for the ``v2x2ssd`` state-passing stage."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from slinoss.perf import note_cache_event, record_region

from .common import _TileConfig, _choose_copy_bits_for_linear_tiles
from .m import StatePassingBwdMAmpere
from .state import StatePassingBwdStateAmpere


_COMPILED_CACHE: dict[tuple, tuple[object, object]] = {}


def _compiled_key(
    *,
    device_index: int,
    chunk_starts_shape: tuple[int, ...],
    m_chunk_shape: tuple[int, ...],
    d_chunk_starts_shape: tuple[int, ...],
    d_final_shape: tuple[int, ...],
    num_threads: int,
    pairs_per_thread: int,
    copy_bits_state: int,
    copy_bits_out: int,
) -> tuple:
    return (
        "state_passing_bwd",
        device_index,
        chunk_starts_shape,
        m_chunk_shape,
        d_chunk_starts_shape,
        d_final_shape,
        int(num_threads),
        int(pairs_per_thread),
        int(copy_bits_state),
        int(copy_bits_out),
    )


def compile_state_passing_bwd_kernels(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int = 128,
    pairs_per_thread: int = 8,
    return_launchers: bool = False,
    enable_overlapped_launcher: bool = True,
) -> tuple:
    """Compile the standalone state-passing backward kernels and allocate outputs."""
    del enable_overlapped_launcher

    if chunk_starts.ndim != 5:
        raise ValueError("chunk_starts must be (B,H,C,P,D).")
    if d_chunk_starts.shape != chunk_starts.shape:
        raise ValueError("d_chunk_starts must match chunk_starts.")
    if m_chunk.ndim != 4 or m_chunk.shape[-1] != 2:
        raise ValueError("m_chunk must be (B,H,C,2).")
    if chunk_starts.dtype != torch.float32:
        raise TypeError("chunk_starts must be float32.")
    if d_chunk_starts.dtype != torch.float32 or d_final.dtype != torch.float32:
        raise TypeError("Upstream grads must be float32.")
    if m_chunk.dtype != torch.float32:
        raise TypeError("m_chunk must be float32.")
    if chunk_starts.device.type != "cuda":
        raise ValueError("CUDA required.")

    B, H, C, P, D = map(int, chunk_starts.shape)
    if tuple(m_chunk.shape[:3]) != (B, H, C):
        raise ValueError("m_chunk leading dims must match chunk_starts.")
    if tuple(d_final.shape) != (B, H, P, D):
        raise ValueError("d_final must be (B,H,P,D).")

    cfg = _TileConfig(
        num_threads=int(num_threads),
        pairs_per_thread=int(pairs_per_thread),
    )
    if cfg.num_threads <= 0:
        raise ValueError("num_threads must be positive.")
    if cfg.num_threads % 32 != 0:
        raise ValueError("num_threads must be a multiple of 32.")
    if cfg.pairs_per_thread <= 0:
        raise ValueError("pairs_per_thread must be positive.")

    S = P * D

    d_inc = torch.empty(
        (B, H, C, P, D), device=chunk_starts.device, dtype=torch.float32
    )
    d_initial = torch.empty(
        (B, H, P, D), device=chunk_starts.device, dtype=torch.float32
    )
    d_m_chunk = torch.empty(
        (B, H, C, 2), device=chunk_starts.device, dtype=torch.float32
    )

    copy_bits_state = _choose_copy_bits_for_linear_tiles(
        d_chunk_starts,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )

    cache_key = _compiled_key(
        device_index=(
            chunk_starts.device.index if chunk_starts.device.index is not None else -1
        ),
        chunk_starts_shape=tuple(chunk_starts.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        d_chunk_starts_shape=tuple(d_chunk_starts.shape),
        d_final_shape=tuple(d_final.shape),
        num_threads=cfg.num_threads,
        pairs_per_thread=cfg.pairs_per_thread,
        copy_bits_state=copy_bits_state,
        copy_bits_out=copy_bits_out,
    )

    align_state = max(d_chunk_starts.element_size(), copy_bits_state // 8)
    align_out = max(d_inc.element_size(), copy_bits_out // 8)

    mDStarts = from_dlpack(d_chunk_starts.contiguous(), assumed_align=align_state)
    mDFinal = from_dlpack(d_final.contiguous(), assumed_align=align_state)
    mM = from_dlpack(m_chunk.contiguous(), assumed_align=16)
    mDInc = from_dlpack(d_inc, assumed_align=align_out)
    mDInit = from_dlpack(d_initial, assumed_align=align_out)

    mStarts = from_dlpack(chunk_starts.contiguous(), assumed_align=align_state)
    mDM = from_dlpack(d_m_chunk, assumed_align=16)

    cached = _COMPILED_CACHE.get(cache_key)
    if cached is None:
        note_cache_event("cache.v2x2ssd.backward.state_passing", hit=False)
        k_state = StatePassingBwdStateAmpere(
            cfg,
            copy_bits_in=copy_bits_state,
            copy_bits_out=copy_bits_out,
        )
        k_m = StatePassingBwdMAmpere(cfg, copy_bits_in=copy_bits_state)
        compiled_state = cute.compile(k_state, mDStarts, mDFinal, mM, mDInc, mDInit)
        compiled_m = cute.compile(k_m, mStarts, mDInc, mDM)
        cached = (compiled_state, compiled_m)
        _COMPILED_CACHE[cache_key] = cached
    else:
        note_cache_event("cache.v2x2ssd.backward.state_passing", hit=True)
        compiled_state, compiled_m = cached

    def launch_sequential() -> None:
        with record_region("backward.v2x2ssd.state_passing.state"):
            compiled_state(mDStarts, mDFinal, mM, mDInc, mDInit)
        with record_region("backward.v2x2ssd.state_passing.m"):
            compiled_m(mStarts, mDInc, mDM)

    def launch_overlapped() -> None:
        launch_sequential()

    base = (compiled_state, compiled_m, d_inc, d_m_chunk, d_initial)
    if return_launchers:
        return (*base, launch_sequential, launch_overlapped)
    return base


def state_passing_bwd_cute(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int = 128,
    pairs_per_thread: int = 8,
    return_d_initial: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Thin public wrapper over the compiled state-passing backward kernel bundle."""
    (
        _compiled_state,
        _compiled_m,
        d_inc,
        d_m_chunk,
        d_initial,
        launch_sequential,
        _launch_overlapped,
    ) = compile_state_passing_bwd_kernels(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        num_threads=num_threads,
        pairs_per_thread=pairs_per_thread,
        return_launchers=True,
    )
    with record_region("backward.v2x2ssd.state_passing.total"):
        launch_sequential()
    if not return_d_initial:
        return (
            d_inc.to(dtype=torch.float32).contiguous(),
            d_m_chunk.to(dtype=torch.float32).contiguous(),
        )
    return (
        d_inc.to(dtype=torch.float32).contiguous(),
        d_m_chunk.to(dtype=torch.float32).contiguous(),
        d_initial.to(dtype=torch.float32).contiguous(),
    )


__all__ = [
    "compile_state_passing_bwd_kernels",
    "state_passing_bwd_cute",
]
