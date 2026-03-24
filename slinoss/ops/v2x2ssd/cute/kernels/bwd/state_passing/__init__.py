"""CuTe backward kernels for the ``v2x2ssd`` state-passing stage."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from .common import (
    _TileConfig,
    _assumed_align,
    _choose_copy_bits_for_linear_tiles,
    _torch_to_cutlass_dtype,
)
from .m import StatePassingBwdMAmpere
from .state import StatePassingBwdStateAmpere


_COMPILED_CACHE: dict[tuple, tuple[object, object]] = {}
_PTR_ARG_CACHE: dict[tuple[object, ...], tuple[object, int]] = {}
_PTR_ARG_CACHE_LIMIT = 32768


def _make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def _make_tensor_spec(
    shape: tuple[int, ...],
    *,
    stride: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = _make_row_major_stride(shape)
    else:
        stride = tuple(int(step) for step in stride)
    return shape, stride


def _make_tensor_from_spec(
    ptr: cute.Pointer,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
):
    shape, stride = spec
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


def _make_ptr_arg(t: torch.Tensor) -> tuple[object, int]:
    device_index = (
        int(t.device.index)
        if t.device.type == "cuda" and t.device.index is not None
        else -1
    )
    key = (t.device.type, device_index, int(t.data_ptr()), t.dtype)
    cached = _PTR_ARG_CACHE.get(key)
    if cached is not None:
        return cached

    align = _assumed_align(t)
    cached = (
        make_ptr(
            _torch_to_cutlass_dtype(t.dtype),
            t.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=align,
        ),
        align,
    )
    if len(_PTR_ARG_CACHE) >= _PTR_ARG_CACHE_LIMIT:
        _PTR_ARG_CACHE.clear()
    _PTR_ARG_CACHE[key] = cached
    return cached


def _make_ptr_args(
    *tensors: torch.Tensor,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    ptrs: list[object] = []
    alignments: list[int] = []
    for tensor in tensors:
        ptr, align = _make_ptr_arg(tensor)
        ptrs.append(ptr)
        alignments.append(align)
    return tuple(ptrs), tuple(alignments)


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
    alignments: tuple[int, ...],
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
        alignments,
    )


def _make_state_passing_state_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    B, H, C, P, D = spec
    num_threads, pairs_per_thread, copy_bits_state, copy_bits_out = cfg

    d_starts_spec = _make_tensor_spec((B, H, C, P, D))
    d_final_spec = _make_tensor_spec((B, H, P, D))
    m_spec = _make_tensor_spec((B, H, C, 2))
    d_inc_spec = _make_tensor_spec((B, H, C, P, D))
    d_initial_spec = _make_tensor_spec((B, H, P, D))

    @cute.jit
    def _state_host_wrapper(
        DStarts_ptr: cute.Pointer,
        DFinal_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        DInc_ptr: cute.Pointer,
        DInit_ptr: cute.Pointer,
    ):
        mDStarts = _make_tensor_from_spec(DStarts_ptr, d_starts_spec)
        mDFinal = _make_tensor_from_spec(DFinal_ptr, d_final_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mDInit = _make_tensor_from_spec(DInit_ptr, d_initial_spec)

        kernel = StatePassingBwdStateAmpere(
            _TileConfig(
                num_threads=num_threads,
                pairs_per_thread=pairs_per_thread,
            ),
            copy_bits_in=copy_bits_state,
            copy_bits_out=copy_bits_out,
        )
        kernel(mDStarts, mDFinal, mM, mDInc, mDInit)

    return _state_host_wrapper


def _make_state_passing_m_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    B, H, C, P, D = spec
    num_threads, pairs_per_thread, copy_bits_state = cfg

    starts_spec = _make_tensor_spec((B, H, C, P, D))
    d_inc_spec = _make_tensor_spec((B, H, C, P, D))
    d_m_spec = _make_tensor_spec((B, H, C, 2))

    @cute.jit
    def _m_host_wrapper(
        Starts_ptr: cute.Pointer,
        DInc_ptr: cute.Pointer,
        DM_ptr: cute.Pointer,
    ):
        mStarts = _make_tensor_from_spec(Starts_ptr, starts_spec)
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_m_spec)

        kernel = StatePassingBwdMAmpere(
            _TileConfig(
                num_threads=num_threads,
                pairs_per_thread=pairs_per_thread,
            ),
            copy_bits_in=copy_bits_state,
        )
        kernel(mStarts, mDInc, mDM)

    return _m_host_wrapper


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

    starts_c = chunk_starts.contiguous()
    d_starts_c = d_chunk_starts.contiguous()
    d_final_c = d_final.contiguous()
    m_c = m_chunk.contiguous()

    state_args, state_alignments = _make_ptr_args(
        d_starts_c,
        d_final_c,
        m_c,
        d_inc,
        d_initial,
    )
    m_args, m_alignments = _make_ptr_args(starts_c, d_inc, d_m_chunk)
    alignments = state_alignments + m_alignments

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
        alignments=alignments,
    )

    cached = _COMPILED_CACHE.get(cache_key)
    if cached is None:
        state_wrapper = _make_state_passing_state_host_wrapper(
            spec=(B, H, C, P, D),
            cfg=(
                cfg.num_threads,
                cfg.pairs_per_thread,
                copy_bits_state,
                copy_bits_out,
            ),
        )
        m_wrapper = _make_state_passing_m_host_wrapper(
            spec=(B, H, C, P, D),
            cfg=(
                cfg.num_threads,
                cfg.pairs_per_thread,
                copy_bits_state,
            ),
        )
        compiled_state = cute.compile(state_wrapper, *state_args)
        compiled_m = cute.compile(m_wrapper, *m_args)
        cached = (compiled_state, compiled_m)
        _COMPILED_CACHE[cache_key] = cached
    else:
        compiled_state, compiled_m = cached

    def launch_sequential() -> None:
        compiled_state(*state_args)
        compiled_m(*m_args)

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
