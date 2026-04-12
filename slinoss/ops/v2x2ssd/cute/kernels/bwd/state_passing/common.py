"""Shared Python-side helpers for backward ``v2x2ssd`` state passing.

This module stays intentionally small. It carries only the pieces shared by the
backward wrapper and kernel:

- tensor-spec construction and static view materialization
- TVM FFI runtime and compile argument builders
- copy-width selection for linear state tiles
- assumed-alignment inference for fake/runtime tensor views
- tile configuration metadata
"""

from dataclasses import dataclass

import cutlass.cute as cute
import torch

from slinoss._cute_runtime import TensorSpec, make_runtime_tensor_spec_view

from ...fwd.common import _compile_env_stream_placeholder, _make_fake_tensor_spec_arg


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    """Extend raw-pointer tensor lifetimes through the current CUDA stream."""
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
) -> TensorSpec:
    resolved_shape = tuple(int(dim) for dim in shape)
    resolved_stride = (
        _make_row_major_stride(resolved_shape)
        if stride is None
        else tuple(int(step) for step in stride)
    )
    return resolved_shape, resolved_stride


def _make_static_tensor_spec_view(
    tensor: cute.Tensor,
    tensor_spec: TensorSpec,
) -> cute.Tensor:
    shape, stride = tensor_spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_runtime_tensor_views_from_specs(
    *tensor_specs: tuple[torch.Tensor, TensorSpec],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        make_runtime_tensor_spec_view(tensor, spec) for tensor, spec in tensor_specs
    )


def _make_tvm_ffi_runtime_and_compile_args_from_specs(
    *tensor_specs: tuple[torch.Tensor, torch.dtype, TensorSpec],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[object, ...]]:
    runtime_args = _make_runtime_tensor_views_from_specs(
        *((tensor, spec) for tensor, _dtype, spec in tensor_specs)
    )
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
    compile_args = tuple(
        _make_fake_tensor_spec_arg(
            dtype=dtype,
            shape=spec[0],
            stride=spec[1],
            align=align,
        )
        for (_tensor, dtype, spec), align in zip(tensor_specs, alignments, strict=True)
    ) + (_compile_env_stream_placeholder(),)
    return runtime_args, alignments, compile_args


def _elem_bits(dt: torch.dtype) -> int:
    if dt == torch.float32:
        return 32
    if dt in (torch.float16, torch.bfloat16):
        return 16
    raise TypeError(f"Unsupported dtype: {dt}")


def _choose_copy_bits_for_linear_tiles(
    t: torch.Tensor,
    tile_stride_elems: int,
    *,
    elems_per_thread: int,
    candidates_bits: tuple[int, ...] = (128, 64, 32),
) -> int:
    """Pick the widest CopyUniversalOp width safe for all tile starts."""
    eb = _elem_bits(t.dtype)
    elem_bytes = t.element_size()
    stride_bytes = tile_stride_elems * elem_bytes

    best = eb
    for bits in candidates_bits:
        if bits < eb:
            continue
        if bits % eb != 0:
            continue
        vec_elems = bits // eb
        if elems_per_thread % vec_elems != 0:
            continue
        align = bits // 8
        if (t.data_ptr() % align) == 0 and (stride_bytes % align) == 0:
            best = bits
            break
    return best


def _assumed_align(
    t: torch.Tensor,
    candidates_bytes: tuple[int, ...] = (16, 8, 4),
) -> int:
    """Return the widest safe assumed alignment for a tensor view."""
    elem_align = max(1, t.element_size())
    ptr = int(t.data_ptr())
    for align in candidates_bytes:
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


@dataclass(frozen=True)
class _TileConfig:
    num_threads: int = 128
    pairs_per_thread: int = 8

    @property
    def elems_per_thread(self) -> int:
        return 2 * int(self.pairs_per_thread)

    @property
    def tile(self) -> int:
        return int(self.num_threads) * self.elems_per_thread
