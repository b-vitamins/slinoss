"""Shared CuTe helpers for the fused mixer-tail rowwise backend."""

from __future__ import annotations

from dataclasses import dataclass

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
    tensor_compile_signature,
    torch_to_cutlass_dtype,
)

_DUMMY_SKIP_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_DUMMY_D_SKIP_CACHE: dict[torch.device, torch.Tensor] = {}


@dataclass(frozen=True)
class TailRowwiseInputInfo:
    batch_size: int
    time_steps: int
    heads: int
    d_head: int
    has_skip: bool
    device_index: int

    @property
    def hidden_dim(self) -> int:
        return int(self.heads * self.d_head)


def _raise_cold_capture_error(direction: str) -> None:
    raise RuntimeError(
        f"CuTe mixer tail rowwise {direction} launcher cache is cold during CUDA graph capture. "
        "Warm the same tail spec once outside capture before graph capture."
    )


def dummy_skip_input(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    cached = _DUMMY_SKIP_CACHE.get(key)
    if cached is None:
        cached = torch.empty((1,), device=device, dtype=dtype)
        _DUMMY_SKIP_CACHE[key] = cached
    return cached


def dummy_d_skip(device: torch.device) -> torch.Tensor:
    cached = _DUMMY_D_SKIP_CACHE.get(device)
    if cached is None:
        cached = torch.empty((1,), device=device, dtype=torch.float32)
        _DUMMY_D_SKIP_CACHE[device] = cached
    return cached


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


def rowwise_input_info(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    has_skip: bool,
) -> TailRowwiseInputInfo:
    if scan_output.ndim != 4:
        raise ValueError(
            "scan_output must be 4-D; got "
            f"{scan_output.ndim}-D with shape {tuple(map(int, scan_output.shape))}."
        )
    batch_size, heads, time_steps, d_head = map(int, scan_output.shape)
    expected_gate = (batch_size, time_steps, heads * d_head)
    if gate.ndim != 3 or tuple(map(int, gate.shape)) != expected_gate:
        raise ValueError(
            f"gate must be {expected_gate}; got {tuple(map(int, gate.shape))}."
        )
    return TailRowwiseInputInfo(
        batch_size=batch_size,
        time_steps=time_steps,
        heads=heads,
        d_head=d_head,
        has_skip=bool(has_skip),
        device_index=_device_cache_key(scan_output.device),
    )


def validate_common_operands(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
    d_normed: torch.Tensor | None = None,
) -> TailRowwiseInputInfo:
    if scan_output.device.type != "cuda" or gate.device.type != "cuda":
        raise ValueError("CuTe mixer tail rowwise backend requires CUDA tensors.")
    if scan_output.device != gate.device:
        raise ValueError("scan_output and gate must be on the same device.")
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if scan_output.dtype not in supported_dtypes or gate.dtype not in supported_dtypes:
        raise NotImplementedError(
            "CuTe mixer tail rowwise backend supports float16, bfloat16, and float32 scan/gate tensors."
        )
    if out_norm_weight.device != scan_output.device:
        raise ValueError("out_norm_weight must be on the same device as scan_output.")
    if out_norm_weight.dtype not in supported_dtypes:
        raise NotImplementedError(
            "CuTe mixer tail rowwise backend supports float16, bfloat16, and float32 norm weights."
        )
    has_skip = skip_input is not None or d_skip is not None
    if (skip_input is None) != (d_skip is None):
        raise ValueError(
            "skip_input and d_skip must either both be provided or both be None."
        )
    info = rowwise_input_info(scan_output, gate, has_skip=has_skip)
    if out_norm_weight.ndim != 1 or int(out_norm_weight.numel()) != info.hidden_dim:
        raise ValueError(
            "out_norm_weight must be 1-D with length "
            f"{info.hidden_dim}; got {tuple(map(int, out_norm_weight.shape))}."
        )
    if has_skip:
        assert skip_input is not None
        assert d_skip is not None
        expected_skip = (
            info.batch_size,
            info.heads,
            info.time_steps,
            info.d_head,
        )
        if tuple(map(int, skip_input.shape)) != expected_skip:
            raise ValueError(
                f"skip_input must be {expected_skip}; got {tuple(map(int, skip_input.shape))}."
            )
        if skip_input.device != scan_output.device:
            raise ValueError("skip_input must be on the same device as scan_output.")
        if d_skip.device != scan_output.device:
            raise ValueError("d_skip must be on the same device as scan_output.")
        if d_skip.ndim != 1 or int(d_skip.numel()) != info.heads:
            raise ValueError(
                f"d_skip must be 1-D with length {info.heads}; got {tuple(map(int, d_skip.shape))}."
            )
        if skip_input.dtype not in supported_dtypes:
            raise NotImplementedError(
                "CuTe mixer tail rowwise backend supports float16, bfloat16, and float32 skip_input tensors."
            )
        if d_skip.dtype not in supported_dtypes:
            raise TypeError(
                "CuTe mixer tail rowwise backend supports float16, bfloat16, and float32 d_skip tensors. "
                f"Got {d_skip.dtype}."
            )
    if d_normed is not None:
        expected_d_normed = (info.batch_size, info.time_steps, info.hidden_dim)
        if tuple(map(int, d_normed.shape)) != expected_d_normed:
            raise ValueError(
                f"d_normed must be {expected_d_normed}; got {tuple(map(int, d_normed.shape))}."
            )
        if d_normed.device != scan_output.device:
            raise ValueError("d_normed must be on the same device as scan_output.")
        if d_normed.dtype not in supported_dtypes:
            raise NotImplementedError(
                "CuTe mixer tail rowwise backend supports float16, bfloat16, and float32 d_normed tensors."
            )
    return info


__all__ = [
    "TailRowwiseInputInfo",
    "_device_cache_key",
    "dummy_d_skip",
    "dummy_skip_input",
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
    "launch_tvm_ffi_on_current_stream",
    "make_fake_tensor_arg",
    "safe_cast_to_dtype",
    "sigmoid",
    "silu",
    "silu_grad",
    "tensor_compile_signature",
    "torch_to_cutlass_dtype",
    "validate_common_operands",
    "warp_reduce_sum",
]
