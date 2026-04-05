"""Shared helpers for the CuTe ``v2x2ssd`` package surface."""

from __future__ import annotations

import torch


def _tc_input_dtype(
    input_dtype: torch.dtype,
    compute_dtype: torch.dtype | None,
) -> torch.dtype:
    """Resolve the tensor-core operand dtype independently of accumulation dtype.

    The CuTe ``v2x2ssd`` kernels accumulate in fp32, but the TC operand staging
    dtype should follow the outer activation contract rather than blindly mirror
    ``compute_dtype``. This keeps bf16 inputs in bf16 end-to-end while preserving
    the existing fp16 TC fallback for true float32 activations, which is more
    stable for segmented training than a float32 -> bf16 fallback.
    """
    if compute_dtype not in (None, torch.float32):
        raise TypeError(f"Unsupported compute dtype: {compute_dtype}")
    if input_dtype in (torch.float16, torch.bfloat16):
        return input_dtype
    if input_dtype == torch.float32:
        return torch.float16
    raise TypeError(f"Unsupported input dtype: {input_dtype}")


def _materialize_boundary_tensor(
    x: torch.Tensor | None,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """Detach public boundary outputs from transient CuTe workspaces."""
    if x is None:
        return None
    y = x if dtype is None or x.dtype == dtype else x.to(dtype=dtype)
    return y.contiguous().clone()


__all__ = ["_materialize_boundary_tensor", "_tc_input_dtype"]
