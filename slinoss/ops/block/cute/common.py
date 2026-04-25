"""Shared CuTe helpers for the block FFN rowwise training path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from slinoss._cute_runtime import (
    launch_tvm_ffi_on_current_stream,
)
from slinoss.ops._cute_common import (
    _device_cache_key,
    _is_cuda_graph_capturing,
    _launchable,
    _llvm_ptr,
    _make_compile_args,
    _make_layout,
    _runtime_alignments,
    _runtime_signature_key,
    _size,
    assumed_align,
    contiguous_tensor,
    make_fake_tensor_arg,
    safe_cast_to_dtype,
    torch_to_cutlass_dtype,
)

ActivationKind = Literal["gelu", "swiglu"]
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


def _raise_cold_capture_error(label: str) -> None:
    raise RuntimeError(
        f"CuTe block FFN {label} launcher cache is cold during CUDA graph capture. "
        "Warm the same FFN spec once outside capture before graph capture."
    )


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
    "_runtime_alignments",
    "_runtime_signature_key",
    "_size",
    "assumed_align",
    "contiguous_tensor",
    "gelu_tanh",
    "gelu_tanh_grad",
    "launch_tvm_ffi_on_current_stream",
    "make_fake_tensor_arg",
    "safe_cast_to_dtype",
    "silu",
    "silu_grad",
    "torch_to_cutlass_dtype",
    "validate_activation_operands",
    "validate_norm_operands",
    "warp_reduce_sum",
]
