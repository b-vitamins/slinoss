"""Shared CuTe helper surface for operator-specific kernels."""

from __future__ import annotations

from typing import Any, cast

import torch

import cutlass
import cutlass.cute as cute

_cute_make_layout = cast(Any, getattr(cute, "make_layout"))
_cute_size = cast(Any, getattr(cute, "size"))
_cutlass_select = cast(Any, getattr(cutlass, "select_"))

FLOAT16_FINITE_MAX = 65504.0
BFLOAT16_FINITE_MAX = 3.3895313892515355e38
FLOAT32_FINITE_MAX = 3.4028234663852886e38


def _make_layout(*args, **kwargs):
    return _cute_make_layout(*args, **kwargs)


def _size(*args, **kwargs):
    return _cute_size(*args, **kwargs)


def _launchable(kernel_call):
    return cast(Any, kernel_call)


def _llvm_ptr(value):
    return cast(Any, value).llvm_ptr


def _device_cache_key(device: torch.device) -> int:
    return 0 if device.index is None else int(device.index)


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported CuTe dtype: {dtype}.")


def assumed_align(
    tensor: torch.Tensor,
    candidates_bytes: tuple[int, ...] = (16, 8, 4),
) -> int:
    elem_align = max(1, tensor.element_size())
    ptr = int(tensor.data_ptr())
    for align in candidates_bytes:
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


def contiguous_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def tensor_compile_signature(tensor: torch.Tensor) -> tuple[torch.dtype, int]:
    return tensor.dtype, assumed_align(tensor)


def make_fake_tensor_spec_arg(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    align: int,
    dynamic_stride: bool = False,
):
    fake_shape = tuple(int(dim) for dim in shape)
    fake_stride = tuple(int(step) for step in stride)
    assumed = int(align)
    if not dynamic_stride and fake_stride == make_row_major_stride(fake_shape):
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        return cute.runtime.make_fake_compact_tensor(
            torch_to_cutlass_dtype(dtype),
            dynamic_shape,
            stride_order=tuple(reversed(range(len(fake_shape)))),
            assumed_align=assumed,
        )
    if dynamic_stride:
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        dynamic_fake_stride = tuple(
            0 if step == 0 else cute.sym_int32() for step in fake_stride
        )
        return cute.runtime.make_fake_tensor(
            torch_to_cutlass_dtype(dtype),
            dynamic_shape,
            stride=dynamic_fake_stride,
            assumed_align=assumed,
        )
    return cute.runtime.make_fake_tensor(
        torch_to_cutlass_dtype(dtype),
        fake_shape,
        stride=fake_stride,
        assumed_align=assumed,
    )


def make_fake_tensor_arg(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    align: int | None = None,
    dynamic_stride: bool = False,
):
    return make_fake_tensor_spec_arg(
        dtype=tensor.dtype,
        shape=tuple(int(dim) for dim in (shape if shape is not None else tensor.shape)),
        stride=tuple(
            int(step) for step in (stride if stride is not None else tensor.stride())
        ),
        align=int(align if align is not None else assumed_align(tensor)),
        dynamic_stride=dynamic_stride,
    )


def _runtime_alignments(runtime_args: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    return tuple(tensor_compile_signature(tensor)[1] for tensor in runtime_args)


def _runtime_signature_key(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[object, ...]:
    return tuple(
        component
        for tensor in runtime_args
        for component in tensor_compile_signature(tensor)
    )


def _make_compile_args(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    alignments: tuple[int, ...],
) -> tuple[object, ...]:
    return tuple(
        make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, alignments, strict=True)
    )


def clamp_finite_for_dtype(x, dtype):
    x = cutlass.Float32(x)
    pos_inf = cutlass.Float32(float("inf"))
    neg_inf = cutlass.Float32(float("-inf"))
    max_abs = cutlass.Float32(FLOAT32_FINITE_MAX)
    if dtype == cutlass.Float16:
        max_abs = cutlass.Float32(FLOAT16_FINITE_MAX)
    elif dtype == cutlass.BFloat16:
        max_abs = cutlass.Float32(BFLOAT16_FINITE_MAX)
    clamped = _cutlass_select(x > max_abs, max_abs, x)
    clamped = _cutlass_select(clamped < -max_abs, -max_abs, clamped)
    clamped = _cutlass_select(x == pos_inf, x, clamped)
    clamped = _cutlass_select(x == neg_inf, x, clamped)
    clamped = _cutlass_select(x != x, x, clamped)
    return clamped


def safe_cast_to_dtype(x, dtype):
    x = clamp_finite_for_dtype(x, dtype)
    if dtype == cutlass.Float16:
        return cutlass.Float16(x)
    if dtype == cutlass.BFloat16:
        return cutlass.BFloat16(x)
    return cutlass.Float32(x)


def _compile_env_stream_placeholder():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


__all__ = [
    "BFLOAT16_FINITE_MAX",
    "FLOAT16_FINITE_MAX",
    "FLOAT32_FINITE_MAX",
    "_compile_env_stream_placeholder",
    "_device_cache_key",
    "_is_cuda_graph_capturing",
    "_launchable",
    "_llvm_ptr",
    "_make_compile_args",
    "_make_layout",
    "_runtime_alignments",
    "_runtime_signature_key",
    "_size",
    "assumed_align",
    "clamp_finite_for_dtype",
    "contiguous_tensor",
    "make_fake_tensor_arg",
    "make_fake_tensor_spec_arg",
    "make_row_major_stride",
    "safe_cast_to_dtype",
    "tensor_compile_signature",
    "torch_to_cutlass_dtype",
]
