"""Shared helpers for the CuTe ``v2x2ssd`` package surface."""

from __future__ import annotations

import torch


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


__all__ = ["_materialize_boundary_tensor"]
