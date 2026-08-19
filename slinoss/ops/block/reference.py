"""Block norm and activation. Pure-PyTorch reference.

Three maps, each the authority for one fused kernel:

- :func:`rmsnorm_ref`, an RMS norm over the trailing axis.
- :func:`rmsnorm_residual_ref`, the same norm fused with the residual add. A
  pre-norm stack adds the previous branch output and then normalizes, so the two
  operations are always adjacent and touch the same bytes. Fusing them halves the
  traffic and gives the residual back in the accumulation dtype, so a long stack
  does not accumulate the residual in bfloat16.
- :func:`swiglu_ref`, the FFN activation ``silu(gate) * up``.

Precision. Reductions and the residual accumulate in float32, or float64 when any
operand is float64. Normed output carries the input dtype; the residual is
returned wide, because narrowing it is what the fusion exists to avoid.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import silu

from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = ["NormResidual", "rmsnorm_ref", "rmsnorm_residual_ref", "swiglu_ref"]


def _check_norm(x: Tensor, weight: Tensor, eps: float) -> None:
    """Validate the operands shared by both norm entry points.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError(
            f"weight must be ({int(x.shape[-1])},), got {tuple(weight.shape)}"
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    check_supported(x, "x")
    check_supported(weight, "weight")


def rmsnorm_ref(x: Tensor, weight: Tensor, *, eps: float) -> Tensor:
    """RMS norm over the trailing axis.

    Args:
        x: Shape ``(..., D)``.
        weight: Shape ``(D,)``.
        eps: Added to the mean square. The summand is non-negative, so this is
            the only thing standing between a zero row and a division by zero.

    Returns:
        Shape ``(..., D)``, dtype of ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    dtype = pinned_dtype(x, weight)
    with autocast_disabled(x.device.type):
        wide = x.to(dtype)
        scale = torch.rsqrt(wide.square().mean(-1, keepdim=True) + eps)
        return (wide * scale * weight.to(dtype)).to(x.dtype)


class NormResidual(NamedTuple):
    """Result of the fused residual add and norm.

    Attributes:
        normed: Normed output, shape ``(..., D)``, dtype of ``x``.
        residual: ``x + residual``, shape ``(..., D)``, in the accumulation
            dtype. Feeds the next block's fused add so the residual stream is
            never narrowed.
    """

    normed: Tensor
    residual: Tensor


def rmsnorm_residual_ref(
    x: Tensor,
    residual: Tensor | None,
    weight: Tensor,
    *,
    eps: float,
) -> NormResidual:
    """Add the residual, then RMS norm over the trailing axis.

    Args:
        x: Branch output, shape ``(..., D)``.
        residual: Incoming residual stream, same shape, or None for the first
            block of a stack.
        weight: Shape ``(D,)``.
        eps: Added to the mean square.

    Returns:
        A :class:`NormResidual`.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _check_norm(x, weight, eps)
    if residual is not None:
        if tuple(residual.shape) != tuple(x.shape):
            raise ValueError(
                f"residual must be {tuple(x.shape)}, got {tuple(residual.shape)}"
            )
        check_supported(residual, "residual")

    operands = (x, weight) if residual is None else (x, residual, weight)
    dtype = pinned_dtype(*operands)
    with autocast_disabled(x.device.type):
        total = x.to(dtype) if residual is None else x.to(dtype) + residual.to(dtype)
        scale = torch.rsqrt(total.square().mean(-1, keepdim=True) + eps)
        normed = (total * scale * weight.to(dtype)).to(x.dtype)
        return NormResidual(normed=normed, residual=total)


def swiglu_ref(gate: Tensor, up: Tensor) -> Tensor:
    """``silu(gate) * up``.

    Args:
        gate: Shape ``(..., D)``.
        up: Same shape.

    Returns:
        Same shape, dtype of ``up``.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError(f"up must be {tuple(gate.shape)}, got {tuple(up.shape)}")
    check_supported(gate, "gate")
    check_supported(up, "up")
    dtype = pinned_dtype(gate, up)
    with autocast_disabled(gate.device.type):
        return (silu(gate.to(dtype)) * up.to(dtype)).to(up.dtype)
