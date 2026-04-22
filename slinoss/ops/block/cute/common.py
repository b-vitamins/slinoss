"""Shared CuTe helpers for the block FFN rowwise training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

ActivationKind = Literal["gelu", "swiglu"]

_cute_make_layout = cast(Any, getattr(cute, "make_layout"))
_cute_size = cast(Any, getattr(cute, "size"))
_cutlass_select = cast(Any, getattr(cutlass, "select_"))

FLOAT16_FINITE_MAX = 65504.0
BFLOAT16_FINITE_MAX = 3.3895313892515355e38
FLOAT32_FINITE_MAX = 3.4028234663852886e38
_GELU_TANH_ALPHA = 0.7978845608028654
_GELU_TANH_BETA = 0.044715


@dataclass(frozen=True)
class FfnNormInputInfo:
    batch_size: int
    time_steps: int
    hidden_dim: int
    device_index: int


@dataclass(frozen=True)
class FfnActivationInputInfo:
    batch_size: int
    time_steps: int
    hidden_dim: int
    kind: ActivationKind
    device_index: int

    @property
    def projected_dim(self) -> int:
        if self.kind == "swiglu":
            return int(2 * self.hidden_dim)
        return int(self.hidden_dim)


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


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    if not torch.cuda.is_available():
        return
    for tensor in tensors:
        if tensor is not None and tensor.device.type == "cuda":
            stream = torch.cuda.current_stream(device=tensor.device)
            tensor.record_stream(stream)


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_cold_capture_error(label: str) -> None:
    raise RuntimeError(
        f"CuTe block FFN {label} launcher cache is cold during CUDA graph capture. "
        "Warm the same FFN spec once outside capture before graph capture."
    )


def make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def torch_to_cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported CuTe block FFN dtype: {dtype}.")


def assumed_align(tensor: torch.Tensor) -> int:
    elem_align = max(1, tensor.element_size())
    ptr = int(tensor.data_ptr())
    for align in (16, 8, 4):
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


def contiguous_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def tensor_compile_signature(tensor: torch.Tensor) -> tuple[torch.dtype, int]:
    return tensor.dtype, assumed_align(tensor)


def make_fake_tensor_arg(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    align: int | None = None,
    dynamic_stride: bool = False,
):
    fake_shape = tuple(
        int(dim) for dim in (shape if shape is not None else tensor.shape)
    )
    fake_stride = tuple(
        int(step) for step in (stride if stride is not None else tensor.stride())
    )
    assumed = int(align if align is not None else assumed_align(tensor))
    if not dynamic_stride and fake_stride == make_row_major_stride(fake_shape):
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        return cute.runtime.make_fake_compact_tensor(
            torch_to_cutlass_dtype(tensor.dtype),
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
            torch_to_cutlass_dtype(tensor.dtype),
            dynamic_shape,
            stride=dynamic_fake_stride,
            assumed_align=assumed,
        )
    return cute.runtime.make_fake_tensor(
        torch_to_cutlass_dtype(tensor.dtype),
        fake_shape,
        stride=fake_stride,
        assumed_align=assumed,
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


def _clamp_finite_for_dtype(x, dtype):
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
    x = _clamp_finite_for_dtype(x, dtype)
    if dtype == cutlass.Float16:
        return cutlass.Float16(x)
    if dtype == cutlass.BFloat16:
        return cutlass.BFloat16(x)
    return cutlass.Float32(x)


def sigmoid(x):
    one = cutlass.Float32(1.0)
    x_f = cutlass.Float32(x)
    return cutlass.Float32(one / (one + cute_math.exp(-x_f)))


def silu(x):
    x_f = cutlass.Float32(x)
    return cutlass.Float32(x_f * sigmoid(x_f))


def silu_grad(x):
    x_f = cutlass.Float32(x)
    sig = sigmoid(x_f)
    return cutlass.Float32(
        sig * (cutlass.Float32(1.0) + x_f * (cutlass.Float32(1.0) - sig))
    )


def gelu_tanh(x):
    x_f = cutlass.Float32(x)
    x_sq = x_f * x_f
    x_cu = x_sq * x_f
    inner = cutlass.Float32(_GELU_TANH_ALPHA) * (
        x_f + cutlass.Float32(_GELU_TANH_BETA) * x_cu
    )
    return cutlass.Float32(
        cutlass.Float32(0.5) * x_f * (cutlass.Float32(1.0) + cute_math.tanh(inner))
    )


def gelu_tanh_grad(x):
    x_f = cutlass.Float32(x)
    x_sq = x_f * x_f
    x_cu = x_sq * x_f
    inner = cutlass.Float32(_GELU_TANH_ALPHA) * (
        x_f + cutlass.Float32(_GELU_TANH_BETA) * x_cu
    )
    tanh_inner = cute_math.tanh(inner)
    sech_sq = cutlass.Float32(1.0) - tanh_inner * tanh_inner
    inner_grad = cutlass.Float32(_GELU_TANH_ALPHA) * (
        cutlass.Float32(1.0) + cutlass.Float32(3.0 * _GELU_TANH_BETA) * x_sq
    )
    return cutlass.Float32(
        cutlass.Float32(0.5) * (cutlass.Float32(1.0) + tanh_inner)
        + cutlass.Float32(0.5) * x_f * sech_sq * inner_grad
    )


def warp_reduce_sum(val):
    total = cutlass.Float32(val)
    total = cutlass.Float32(
        total
        + cute.arch.shuffle_sync_bfly(total, offset=16, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=8, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=4, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=2, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=1, mask=-1, mask_and_clamp=31)
    )
    return total


def validate_norm_operands(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> FfnNormInputInfo:
    if residual.ndim != 3:
        raise ValueError(
            f"residual must be 3-D (batch, time, hidden); got {tuple(residual.shape)}."
        )
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != int(residual.shape[-1]):
        raise ValueError(
            "norm_weight must be 1-D with length matching residual hidden dim; got "
            f"{tuple(norm_weight.shape)} for residual {tuple(residual.shape)}."
        )
    if norm_weight.device != residual.device:
        raise ValueError("residual and norm_weight must be on the same device.")
    if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported residual dtype {residual.dtype}.")
    if norm_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported norm_weight dtype {norm_weight.dtype}.")
    batch_size, time_steps, hidden_dim = map(int, residual.shape)
    return FfnNormInputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        device_index=_device_cache_key(residual.device),
    )


def validate_activation_operands(
    projected: torch.Tensor,
    d_hidden: torch.Tensor | None,
    *,
    kind: ActivationKind,
) -> FfnActivationInputInfo:
    if projected.ndim != 3:
        raise ValueError(
            "projected must be 3-D (batch, time, hidden); got "
            f"{tuple(projected.shape)}."
        )
    batch_size, time_steps, projected_dim = map(int, projected.shape)
    if kind == "swiglu":
        if projected_dim % 2 != 0:
            raise ValueError(
                f"swiglu projected hidden dim must be even; got {projected_dim}."
            )
        hidden_dim = projected_dim // 2
    else:
        hidden_dim = projected_dim
    if d_hidden is not None and tuple(map(int, d_hidden.shape)) != (
        batch_size,
        time_steps,
        hidden_dim,
    ):
        raise ValueError(
            "d_hidden must match the activation output shape; got "
            f"{tuple(d_hidden.shape)} expected {(batch_size, time_steps, hidden_dim)}."
        )
    if d_hidden is not None and d_hidden.device != projected.device:
        raise ValueError("projected and d_hidden must be on the same device.")
    if projected.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"Unsupported projected dtype {projected.dtype}.")
    if d_hidden is not None and d_hidden.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise TypeError(f"Unsupported d_hidden dtype {d_hidden.dtype}.")
    return FfnActivationInputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        hidden_dim=hidden_dim,
        kind=kind,
        device_index=_device_cache_key(projected.device),
    )


__all__ = [
    "ActivationKind",
    "FfnActivationInputInfo",
    "FfnNormInputInfo",
    "_device_cache_key",
    "_is_cuda_graph_capturing",
    "_launchable",
    "_llvm_ptr",
    "_make_compile_args",
    "_make_layout",
    "_raise_cold_capture_error",
    "_record_tensors_on_current_stream",
    "_runtime_alignments",
    "_runtime_signature_key",
    "_size",
    "assumed_align",
    "contiguous_tensor",
    "gelu_tanh",
    "gelu_tanh_grad",
    "make_fake_tensor_arg",
    "safe_cast_to_dtype",
    "silu",
    "silu_grad",
    "torch_to_cutlass_dtype",
    "validate_activation_operands",
    "validate_norm_operands",
    "warp_reduce_sum",
]
