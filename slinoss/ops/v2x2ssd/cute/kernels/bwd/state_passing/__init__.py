"""CuTe backward kernel for the fused ``v2x2ssd`` state-passing stage."""

from __future__ import annotations

import torch
import cutlass.cute as cute

from ...fwd.common import _make_fake_tensor_arg
from .common import (
    _TileConfig,
    _assumed_align,
    _choose_copy_bits_for_linear_tiles,
)
from .state import StatePassingBwdAmpere


_COMPILED_CACHE: dict[tuple, object] = {}


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    stream = None
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        ident = id(tensor)
        if ident in seen:
            continue
        if stream is None:
            stream = torch.cuda.current_stream(device=tensor.device)
        tensor.record_stream(stream)
        seen.add(ident)


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
    tensor: cute.Tensor,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
):
    shape, stride = spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _compiled_key(
    *,
    device_index: int,
    chunk_starts_shape: tuple[int, ...],
    m_chunk_shape: tuple[int, ...],
    d_chunk_starts_shape: tuple[int, ...],
    d_final_shape: tuple[int, ...],
    num_threads: int,
    pairs_per_thread: int,
    copy_bits_starts: int,
    copy_bits_dstarts: int,
    copy_bits_dinc: int,
    copy_bits_initial: int,
    copy_bits_final: int,
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
        int(copy_bits_starts),
        int(copy_bits_dstarts),
        int(copy_bits_dinc),
        int(copy_bits_initial),
        int(copy_bits_final),
        alignments,
    )


def _make_state_passing_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    B, H, C, P, D = spec
    (
        num_threads,
        pairs_per_thread,
        copy_bits_starts,
        copy_bits_dstarts,
        copy_bits_dinc,
        copy_bits_initial,
        copy_bits_final,
    ) = cfg

    starts_spec = _make_tensor_spec((B, H, C, P, D))
    d_starts_spec = _make_tensor_spec((B, H, C, P, D))
    d_final_spec = _make_tensor_spec((B, H, P, D))
    m_spec = _make_tensor_spec((B, H, C, 2))
    d_inc_spec = _make_tensor_spec((B, H, C, P, D))
    d_m_spec = _make_tensor_spec((B, H, C, 2))
    d_initial_spec = _make_tensor_spec((B, H, P, D))

    @cute.jit
    def _state_host_wrapper(
        Starts_ptr: cute.Tensor,
        DStarts_ptr: cute.Tensor,
        DFinal_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        DInc_ptr: cute.Tensor,
        DM_ptr: cute.Tensor,
        DInit_ptr: cute.Tensor,
    ):
        mStarts = _make_tensor_from_spec(Starts_ptr, starts_spec)
        mDStarts = _make_tensor_from_spec(DStarts_ptr, d_starts_spec)
        mDFinal = _make_tensor_from_spec(DFinal_ptr, d_final_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_m_spec)
        mDInit = _make_tensor_from_spec(DInit_ptr, d_initial_spec)

        kernel = StatePassingBwdAmpere(
            _TileConfig(
                num_threads=num_threads,
                pairs_per_thread=pairs_per_thread,
            ),
            copy_bits_starts=copy_bits_starts,
            copy_bits_dstarts=copy_bits_dstarts,
            copy_bits_dinc=copy_bits_dinc,
            copy_bits_initial=copy_bits_initial,
            copy_bits_final=copy_bits_final,
        )
        kernel(mStarts, mDStarts, mDFinal, mM, mDInc, mDM, mDInit)

    return _state_host_wrapper


def compile_state_passing_bwd_kernel(
    chunk_starts: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    d_chunk_starts: torch.Tensor,
    d_final: torch.Tensor,
    num_threads: int = 128,
    pairs_per_thread: int = 8,
    return_launcher: bool = False,
) -> tuple:
    """Compile the fused state-passing backward kernel and allocate outputs."""

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
    d_m_chunk = torch.zeros(
        (B, H, C, 2), device=chunk_starts.device, dtype=torch.float32
    )

    starts_c = chunk_starts.contiguous()
    d_starts_c = d_chunk_starts.contiguous()
    d_final_c = d_final.contiguous()
    m_c = m_chunk.contiguous()

    copy_bits_starts = _choose_copy_bits_for_linear_tiles(
        starts_c,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_dstarts = _choose_copy_bits_for_linear_tiles(
        d_starts_c,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_final = _choose_copy_bits_for_linear_tiles(
        d_final_c,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_dinc = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )
    copy_bits_initial = _choose_copy_bits_for_linear_tiles(
        d_initial,
        tile_stride_elems=S,
        elems_per_thread=cfg.elems_per_thread,
    )

    runtime_args = (
        starts_c,
        d_starts_c,
        d_final_c,
        m_c,
        d_inc,
        d_m_chunk,
        d_initial,
    )
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
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
        copy_bits_starts=copy_bits_starts,
        copy_bits_dstarts=copy_bits_dstarts,
        copy_bits_dinc=copy_bits_dinc,
        copy_bits_initial=copy_bits_initial,
        copy_bits_final=copy_bits_final,
        alignments=alignments,
    )

    compiled = _COMPILED_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_state_passing_host_wrapper(
            spec=(B, H, C, P, D),
            cfg=(
                cfg.num_threads,
                cfg.pairs_per_thread,
                copy_bits_starts,
                copy_bits_dstarts,
                copy_bits_dinc,
                copy_bits_initial,
                copy_bits_final,
            ),
        )
        compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(runtime_args, alignments, strict=True)
        )
        compiled = cute.compile(
            host_wrapper,
            *compile_args,
            options="--enable-tvm-ffi",
        )
        _COMPILED_CACHE[cache_key] = compiled

    def launch() -> None:
        d_m_chunk.zero_()
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*runtime_args)

    base = (compiled, d_inc, d_m_chunk, d_initial)
    if return_launcher:
        return (*base, launch)
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
    """Thin public wrapper over the fused state-passing backward kernel."""

    _compiled, d_inc, d_m_chunk, d_initial, launch = compile_state_passing_bwd_kernel(
        chunk_starts,
        m_chunk,
        d_chunk_starts=d_chunk_starts,
        d_final=d_final,
        num_threads=num_threads,
        pairs_per_thread=pairs_per_thread,
        return_launcher=True,
    )
    launch()
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
    "compile_state_passing_bwd_kernel",
    "state_passing_bwd_cute",
]
