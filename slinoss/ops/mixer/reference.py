"""Fused mixer tail. Pure-PyTorch reference.

Everything between the scan output and the output projection:

    x = y + d_skip * u
    x = x * silu(gate)
    out = x * rsqrt(mean(x^2) + eps) * weight

The skip term is the direct path from the scan input to the scan output, so a
head can pass information through without going around the state. The gate is
applied before the norm, so the norm sees the gated magnitude and the scale it
divides by is the one the next projection actually reads.

The reduction runs over ``P``, the rows of one head, and never crosses the head
axis. That keeps the whole tail rowwise: one ``(b, h, t)`` triple is one
independent problem of length ``P``, which is what lets the fused kernel read
each element once. A reduction over ``d_inner`` would couple every head at every
token and force either a second pass or a cross-head barrier. Parameter count is
unchanged: ``weight`` is ``(H,P)``, which is ``d_inner`` scalars.

Precision. The sum of squares accumulates in float32, or float64 when any operand
is float64 so a float64 call is an oracle end to end. The output carries the dtype
of ``y``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import silu

from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = ["MixerTailGrads", "mixer_tail_bwd_ref", "mixer_tail_ref"]


def mixer_tail_ref(
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Apply the skip, the gate, and the per-head RMS norm.

    Args:
        y: Scan output, shape ``(B,H,T,P)``.
        u: Scan input, shape ``(B,H,T,P)``. Source of the skip term.
        gate: Gate, shape ``(B,H,T,P)``.
        d_skip: Per-row skip scale, shape ``(H,P)``.
        weight: Per-row norm scale, shape ``(H,P)``.
        eps: Added to the mean square before the reciprocal square root. The
            reduction is over ``P`` and the summand is non-negative, so ``eps``
            is the only thing standing between a row of exact zeros and a
            division by zero.

    Returns:
        Shape ``(B,H,T,P)``, dtype of ``y``.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    if y.ndim != 4:
        raise ValueError(f"y must be (B,H,T,P), got {tuple(y.shape)}")
    shape = tuple(int(d) for d in y.shape)
    for name, tensor in (("u", u), ("gate", gate)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")
    rows = (shape[1], shape[3])
    for name, tensor in (("d_skip", d_skip), ("weight", weight)):
        if tuple(tensor.shape) != rows:
            raise ValueError(f"{name} must be {rows}, got {tuple(tensor.shape)}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    for name, tensor in (
        ("y", y),
        ("u", u),
        ("gate", gate),
        ("d_skip", d_skip),
        ("weight", weight),
    ):
        check_supported(tensor, name)

    dtype = pinned_dtype(y, u, gate, d_skip, weight)
    with autocast_disabled(y.device.type):
        # (H,P) broadcasts against (B,H,T,P) once the token axis is inserted.
        x = y.to(dtype) + d_skip.to(dtype)[:, None, :] * u.to(dtype)
        x = x * silu(gate.to(dtype))
        scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return (x * scale * weight.to(dtype)[:, None, :]).to(y.dtype)


class MixerTailGrads(NamedTuple):
    """Gradients of the fused tail.

    Attributes:
        dy: ``(B,H,T,P)``, dtype of ``y``.
        du: ``(B,H,T,P)``, dtype of ``u``.
        dgate: ``(B,H,T,P)``, dtype of ``gate``.
        dd_skip: ``(H,P)``, dtype of ``d_skip``.
        dweight: ``(H,P)``, dtype of ``weight``.
    """

    dy: Tensor
    du: Tensor
    dgate: Tensor
    dd_skip: Tensor
    dweight: Tensor


def mixer_tail_bwd_ref(
    dout: Tensor,
    y: Tensor,
    u: Tensor,
    gate: Tensor,
    d_skip: Tensor,
    weight: Tensor,
    /,
    *,
    eps: float,
) -> MixerTailGrads:
    """Pullback of :func:`mixer_tail_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an algebra
    error passes silently. In float64 this is the gradient authority the kernel is
    measured against.

    Args:
        dout: Cotangent of the output, shape ``(B,H,T,P)``.
        y: The forward's scan output, shape ``(B,H,T,P)``.
        u: The forward's scan input, shape ``(B,H,T,P)``.
        gate: The forward's gate, shape ``(B,H,T,P)``.
        d_skip: The forward's skip scale, shape ``(H,P)``.
        weight: The forward's norm scale, shape ``(H,P)``.
        eps: The forward's epsilon.

    Returns:
        A :class:`MixerTailGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    if tuple(dout.shape) != tuple(y.shape):
        raise ValueError(f"dout must be {tuple(y.shape)}, got {tuple(dout.shape)}")

    leaves = tuple(
        tensor.detach().requires_grad_(True) for tensor in (y, u, gate, d_skip, weight)
    )
    with torch.enable_grad():
        out = mixer_tail_ref(*leaves, eps=eps)
    return MixerTailGrads(*torch.autograd.grad(out, leaves, dout))
