"""Fused mixer tail. Pure-PyTorch reference.

Everything between the scan output and the output projection:

    x = y + d_skip * u
    x = x * silu(gate)
    out = x * rsqrt(mean(x^2) + eps) * weight

The skip term is the direct path from the scan input to the scan output, so a
head can pass information through without going around the state. The gate is
applied before the norm, so the norm sees the gated magnitude and the scale it
divides by is the one the next projection actually reads.

``d_skip`` is ``(H,)``: one scalar per head, which is the width Mamba2's ``D``
carries. The skip is a gain on the whole head's direct path, and a per-row gain
would be ``d_head`` times the parameters for a term the row's own ``B`` and ``C``
already shape. ``weight`` stays ``(H,P)``, because a norm gain is per row by
definition.

The reduction runs over ``P``, the rows of one head, and never crosses the head
axis. That keeps the whole tail rowwise: one ``(b, h, t)`` triple is one
independent problem of length ``P``, which is what lets the fused kernel read
each element once. A reduction over ``d_inner`` would couple every head at every
token and force either a second pass or a cross-head barrier.

Layout. ``y`` and ``u`` arrive head-major, ``(B,H,T,P)``, which is what the scan
and the convolution write. ``gate`` and the output are token-major, ``(B,T,H*P)``,
which is what the two projections around the tail read and write; head ``h``
occupies columns ``h*P`` through ``(h+1)*P``, so one head at one token is a
contiguous run of ``P`` either way. The tail is therefore the only place the two
orders meet, and it converts between them as part of a pass it already makes. A
standalone transpose would be a second pass over the largest tensor in the block.

Precision. The sum of squares accumulates in float32, or float64 when any operand
is float64 so a float64 call is an oracle end to end. The output carries the dtype
of ``y``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import silu

from slinoss._guard import check_pitched
from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = [
    "MixerTailGrads",
    "as_head_major",
    "as_token_major",
    "check_dgate_dest",
    "mixer_tail_bwd_ref",
    "mixer_tail_ref",
    "tail_shape",
]


def as_head_major(t: Tensor, heads: int) -> Tensor:
    """``(B,T,H*P) -> (B,H,T,P)``.

    A view whatever ``t``'s pitch is, because the split of the trailing axis needs
    only its unit stride.

    Args:
        t: Token-major tensor, ``(B,T,H*P)``.
        heads: ``H``.

    Returns:
        The head-major view.
    """
    return t.unflatten(-1, (heads, -1)).permute(0, 2, 1, 3)


def as_token_major(t: Tensor) -> Tensor:
    """``(B,H,T,P) -> (B,T,H*P)``.

    The flatten crosses a permuted pair, so this copies. The kernel does it in
    store addresses instead.

    Args:
        t: Head-major tensor, ``(B,H,T,P)``.

    Returns:
        The token-major tensor.
    """
    return t.permute(0, 2, 1, 3).flatten(-2, -1)


def tail_shape(
    y: Tensor, u: Tensor, gate: Tensor, d_skip: Tensor, weight: Tensor
) -> tuple[int, int, int, int]:
    """The ``(B,H,T,P)`` the five operands agree on.

    Args:
        y: Scan output, ``(B,H,T,P)``.
        u: Scan input, ``(B,H,T,P)``.
        gate: Gate, ``(B,T,H*P)``.
        d_skip: Skip scale, ``(H,)``.
        weight: Norm scale, ``(H,P)``.

    Returns:
        ``(B, H, T, P)``.

    Raises:
        ValueError: On a rank or shape mismatch.
    """
    if y.ndim != 4:
        raise ValueError(f"y must be (B,H,T,P), got {tuple(y.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in y.shape)
    if tuple(u.shape) != (bsz, heads, seqlen, rows):
        raise ValueError(
            f"u must be {(bsz, heads, seqlen, rows)}, got {tuple(u.shape)}"
        )
    flat = (bsz, seqlen, heads * rows)
    if tuple(gate.shape) != flat:
        raise ValueError(f"gate must be {flat}, got {tuple(gate.shape)}")
    if tuple(d_skip.shape) != (heads,):
        raise ValueError(f"d_skip must be {(heads,)}, got {tuple(d_skip.shape)}")
    if tuple(weight.shape) != (heads, rows):
        raise ValueError(f"weight must be {(heads, rows)}, got {tuple(weight.shape)}")
    return bsz, heads, seqlen, rows


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
        gate: Gate, shape ``(B,T,H*P)``, token-major.
        d_skip: Per-head skip scale, shape ``(H,)``.
        weight: Per-row norm scale, shape ``(H,P)``.
        eps: Added to the mean square before the reciprocal square root. The
            reduction is over ``P`` and the summand is non-negative, so ``eps``
            is the only thing standing between a row of exact zeros and a
            division by zero.

    Returns:
        Shape ``(B,T,H*P)``, token-major, dtype of ``y``.

    Raises:
        ValueError: On a rank or shape mismatch, or a non-positive ``eps``.
        TypeError: On an unsupported dtype.
    """
    _, heads, _, _ = tail_shape(y, u, gate, d_skip, weight)
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
        # (H,) broadcasts against (B,H,T,P) once the token and row axes are inserted,
        # and (H,P) once the token axis is.
        x = y.to(dtype) + d_skip.to(dtype)[:, None, None] * u.to(dtype)
        x = x * silu(as_head_major(gate.to(dtype), heads))
        scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        out = x * scale * weight.to(dtype)[:, None, :]
        return as_token_major(out).to(y.dtype)


class MixerTailGrads(NamedTuple):
    """Gradients of the fused tail.

    Attributes:
        dy: ``(B,H,T,P)``, dtype of ``y``.
        du: ``(B,H,T,P)``, dtype of ``u``.
        dgate: ``(B,T,H*P)``, dtype of ``gate``. The destination the caller
            supplied, when it supplied one, and not a copy of it.
        dd_skip: ``(H,)``, dtype of ``d_skip``.
        dweight: ``(H,P)``, dtype of ``weight``.
    """

    dy: Tensor
    du: Tensor
    dgate: Tensor
    dd_skip: Tensor
    dweight: Tensor


def check_dgate_dest(dgate: Tensor, gate: Tensor) -> None:
    """Hold a caller-supplied ``dgate`` destination to the allocation it replaces.

    The mixer's backward allocates one ``dproj`` and hands each consumer the band its
    gradient belongs in, so a destination is one column band of a wider buffer:
    pitched rather than contiguous, and held to
    :func:`slinoss._guard.check_pitched`. Shape and dtype come first, so a
    wrong-shaped destination reports its shape rather than an alignment complaint.

    Both backends call this, so the two refuse a destination under one wording.

    Args:
        dgate: The destination. Carries the shape, dtype, and device of ``gate``.
        gate: The forward's gate, ``(B,T,H*P)``.

    Raises:
        ValueError: On a shape or device mismatch, or on a band whose offset or
            pitch is not a multiple of :data:`slinoss._guard.SECTOR_BYTES`.
        TypeError: On a dtype other than ``gate``'s.
    """
    want = tuple(gate.shape)
    if tuple(dgate.shape) != want:
        raise ValueError(f"dgate must be {want}, got {tuple(dgate.shape)}")
    if dgate.dtype is not gate.dtype:
        raise TypeError(
            f"dgate is {dgate.dtype} and gate is {gate.dtype}; a destination "
            f"carries the dtype of the gradient it holds"
        )
    if dgate.device != gate.device:
        raise ValueError(f"dgate must be on {gate.device}, got {dgate.device}")
    # The band rule is a device rule: a row that starts mid-sector fetches a sector
    # it discards. This path also runs on CPU, where there is no sector to waste.
    if dgate.device.type == "cuda":
        check_pitched(((dgate, "dgate"),))


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
    dgate: Tensor | None = None,
) -> MixerTailGrads:
    """Pullback of :func:`mixer_tail_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an algebra
    error passes silently. In float64 this is the gradient authority the kernel is
    measured against.

    Args:
        dout: Cotangent of the output, shape ``(B,T,H*P)``.
        y: The forward's scan output, shape ``(B,H,T,P)``.
        u: The forward's scan input, shape ``(B,H,T,P)``.
        gate: The forward's gate, shape ``(B,T,H*P)``.
        d_skip: The forward's skip scale, shape ``(H,)``.
        weight: The forward's norm scale, shape ``(H,P)``.
        eps: The forward's epsilon.
        dgate: Destination for the gate gradient, ``(B,T,H*P)``, carrying the shape,
            dtype, and device the allocation would have had. Written in full, never
            accumulated into and never zeroed first, and returned in the result as
            this same object. ``None`` allocates one.

    Returns:
        A :class:`MixerTailGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, a non-positive ``eps``, or a
            ``dgate`` destination that is not a legal band.
        TypeError: On an unsupported dtype.
    """
    bsz, heads, seqlen, rows = tail_shape(y, u, gate, d_skip, weight)
    want = (bsz, seqlen, heads * rows)
    if tuple(dout.shape) != want:
        raise ValueError(f"dout must be {want}, got {tuple(dout.shape)}")
    if dgate is not None:
        check_dgate_dest(dgate, gate)

    leaves = tuple(
        tensor.detach().requires_grad_(True) for tensor in (y, u, gate, d_skip, weight)
    )
    with torch.enable_grad():
        out = mixer_tail_ref(*leaves, eps=eps)
    grads = torch.autograd.grad(out, leaves, dout)
    if dgate is not None:
        dgate.copy_(grads[2])
        return MixerTailGrads(grads[0], grads[1], dgate, grads[3], grads[4])
    return MixerTailGrads(*grads)
