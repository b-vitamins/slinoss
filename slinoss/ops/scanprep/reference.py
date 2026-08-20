"""The scan's parameter frontier. Pure-PyTorch reference.

Takes the token-major parameter slice of one projection output and emits the two
operands the scan cannot read off that projection as it lies: the packed
transition ``trans`` and the packed taps ``K``. ``B`` and ``C`` are pitched bands
of the same projection and reach the scan's kernels unchanged, so they are not
operands here.

``params`` is a slice of a single ``(B,T,W)`` projection output, so it is not
contiguous: the trailing axis has unit stride and the row stride is the full
projection width. Nothing here repacks it.

The numerical invariants the kernels rely on hold by construction, so no kernel
needs a clamp, an epsilon, or a validity pass:

- ``ls = -softplus(x) <= 0``, so every chunk-local log-scale prefix is monotone
  non-increasing and every decay factor lies in ``(0,1]``. Overflow is
  unreachable and underflow is graceful. The bound is non-strict only because
  ``softplus`` underflows to zero for very negative ``x``.
- ``|w| = w_max * |x| / sqrt(1 + |x|^2) <= w_max < pi``, so the quaternion
  exponential is a single branchless polynomial over the whole reachable domain.
  The map is analytic in ``x``: ``1 + |x|^2 >= 1``, so the rsqrt has no
  singularity and needs no guard. The bound is non-strict only because the ratio
  rounds to one once ``|x|`` exceeds the reciprocal of the machine epsilon.

Taps are unconstrained. In the polynomial chart ``K(v) = kr*v + g*(w.v)*w +
h*(w x v)`` there is no well-definedness condition to enforce; the axis-angle
normal form's constraint at ``w = 0`` is structural here.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import softplus

from slinoss._precision import (
    autocast_disabled,
    check_pinned,
    check_supported,
    pinned_dtype,
)

__all__ = [
    "LS_COLUMN",
    "PARAM_COLS",
    "ROTVEC_COLUMNS",
    "TAP_COLUMNS",
    "ScanGrads",
    "ScanParams",
    "bounded_logscale",
    "bounded_rotvec",
    "check_cotangents",
    "check_operands",
    "pack_params",
    "scanprep_bwd_ref",
    "scanprep_ref",
]

PARAM_COLS = 10
"""Projection columns one head spends on the transition and the two taps.

``(w_x, w_y, w_z, ls, kr0, g0, h0, kr1, g1, h1)``: three for the rotation vector,
one for the log-scale, three per tap. Not a shape multiple, so it does not live in
:mod:`slinoss.config`; it is this operator's own column count.
"""

ROTVEC_COLUMNS = slice(0, 3)
"""Columns of one head's parameter row holding the unconstrained rotation vector."""

LS_COLUMN = 3
"""Column of one head's parameter row holding the unconstrained log-scale."""

TAP_COLUMNS = slice(4, PARAM_COLS)
"""Columns of one head's parameter row holding both unconstrained taps."""


def bounded_rotvec(raw: Tensor, w_max: float) -> Tensor:
    """Map an unconstrained vector into the ball of radius ``w_max``.

    ``w = w_max * raw / sqrt(1 + |raw|^2)``. Analytic everywhere and monotone in
    ``|raw|``.

    Args:
        raw: Unconstrained vectors, shape ``(...,3)``.
        w_max: Radius bound. Must lie in ``(0, pi)``.

    Returns:
        Rotation vectors with ``|w| <= w_max``, shape ``(...,3)``.

    Raises:
        ValueError: If ``w_max`` is outside ``(0, pi)``.
    """
    if not 0.0 < w_max < math.pi:
        raise ValueError(f"w_max must lie in (0, pi), got {w_max}")
    return raw * (w_max * torch.rsqrt(1.0 + (raw * raw).sum(-1, keepdim=True)))


def bounded_logscale(raw: Tensor) -> Tensor:
    """Map an unconstrained scalar to a non-positive log-scale.

    ``ls = -softplus(raw)``, so ``ls <= 0`` for every finite input and the decay
    per step is in ``(0,1]``.

    Args:
        raw: Unconstrained scalars, any shape.

    Returns:
        Log-scales, same shape.
    """
    return -softplus(raw)


def pack_params(w_raw: Tensor, ls_raw: Tensor, tap_raw: Tensor) -> Tensor:
    """Lay head-major raw parameters out in the projection's column order.

    The mixer's projection emits this layout directly. A caller holding the three
    head-major tensors separately -- a test fixture or a benchmark -- packs them
    here rather than restating the column order.

    Args:
        w_raw: Unconstrained rotation vectors, ``(B,H,T,3)``.
        ls_raw: Unconstrained log-scales, ``(B,H,T)``.
        tap_raw: Unconstrained taps ``(kr, g, h)``, ``(B,H,T,2,3)``.

    Returns:
        ``(B,T,H*PARAM_COLS)``, contiguous, in the dtype of ``w_raw``.

    Raises:
        ValueError: On a rank or shape mismatch.
    """
    if w_raw.ndim != 4 or w_raw.shape[-1] != 3:
        raise ValueError(f"w_raw must be (B,H,T,3), got {tuple(w_raw.shape)}")
    lead = tuple(int(d) for d in w_raw.shape[:3])
    if tuple(ls_raw.shape) != lead:
        raise ValueError(f"ls_raw must be {lead}, got {tuple(ls_raw.shape)}")
    if tuple(tap_raw.shape) != (*lead, 2, 3):
        raise ValueError(f"tap_raw must be {(*lead, 2, 3)}, got {tuple(tap_raw.shape)}")
    bsz, heads, seqlen = lead
    row = torch.cat(
        [
            w_raw.permute(0, 2, 1, 3),
            ls_raw.permute(0, 2, 1)[..., None],
            tap_raw.permute(0, 2, 1, 3, 4).reshape(bsz, seqlen, heads, 6),
        ],
        dim=-1,
    )
    return row.reshape(bsz, seqlen, heads * PARAM_COLS).contiguous()


class ScanParams(NamedTuple):
    """The per-token operands the scan cannot read off the projection as it lies.

    Attributes:
        trans: ``(w_x, w_y, w_z, ls)``, shape ``(B,H,T,4)``, pinned dtype.
        K: Per tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned dtype. Tap
            index 0 is previous and 1 is current. Lane 3 is a hard zero, present
            for float4 alignment.
    """

    trans: Tensor
    K: Tensor


class ScanGrads(NamedTuple):
    """Gradients of :func:`scanprep_ref`.

    Attributes:
        dparams: ``(B,T,H*PARAM_COLS)``, contiguous, dtype of ``params``.
        dparam_bias: ``(H,PARAM_COLS)``, dtype of ``param_bias``.
    """

    dparams: Tensor
    dparam_bias: Tensor


def check_operands(params: Tensor, param_bias: Tensor, heads: int) -> None:
    """Validate the operand set.

    The shape, stride, and dtype contract every backend shares. The kernel host
    path calls this rather than restating it, so the two cannot disagree; the CuTe
    path adds only what is its own, namely device residency, base alignment, and
    the narrower kernel dtype set.

    Args:
        params: ``(B,T,H*PARAM_COLS)``.
        param_bias: ``(H,PARAM_COLS)``.
        heads: ``H``.

    Raises:
        ValueError: On a non-positive ``heads``, a rank or shape mismatch, or a
            trailing axis whose stride is not one.
        TypeError: On an unsupported dtype, or on a low-precision ``param_bias``.
    """
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    want = heads * PARAM_COLS
    if params.ndim != 3 or params.shape[-1] != want:
        raise ValueError(
            f"params must be (B,T,{want}) at heads={heads}, got {tuple(params.shape)}"
        )
    if tuple(param_bias.shape) != (heads, PARAM_COLS):
        raise ValueError(
            f"param_bias must be {(heads, PARAM_COLS)}, got {tuple(param_bias.shape)}"
        )
    # Row stride is the projection width, so only the trailing axis is pinned.
    if params.stride(-1) != 1:
        raise ValueError(
            f"params must have unit stride on its trailing axis, "
            f"got {params.stride(-1)}"
        )
    check_supported(params, "params")
    check_pinned(param_bias, "param_bias")


def scanprep_ref(
    params: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps and pack the result.

    Args:
        params: Projection slice, ``(B,T,H*PARAM_COLS)``, activation dtype.
            Trailing stride one; the row stride is the projection width. Per head,
            in order ``(w_x, w_y, w_z, ls, kr0, g0, h0, kr1, g1, h1)``.
        param_bias: ``(H,PARAM_COLS)``, float32, added to every token's row
            before the maps.
        heads: ``H``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`ScanParams`, both fields in the pinned dtype (I4) and
        contiguous.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, or a
            ``w_max`` outside ``(0, pi)``.
        TypeError: On an unsupported dtype, or on a low-precision ``param_bias``.
    """
    check_operands(params, param_bias, heads)
    dtype = pinned_dtype(params, param_bias)
    with autocast_disabled(params.device.type):
        # (B,T,H,PARAM_COLS) -> (B,H,T,PARAM_COLS). unflatten of a unit-stride
        # trailing axis is a view, so the strided operand is read where it lies.
        rows = params.unflatten(-1, (heads, PARAM_COLS)).to(dtype)
        rows = (rows + param_bias.to(dtype)).permute(0, 2, 1, 3)
        w = bounded_rotvec(rows[..., ROTVEC_COLUMNS], w_max)
        ls = bounded_logscale(rows[..., LS_COLUMN])
        tap = rows[..., TAP_COLUMNS].unflatten(-1, (2, 3))
        trans = torch.cat([w, ls[..., None]], dim=-1).contiguous()
        packed = torch.cat([tap, torch.zeros_like(tap[..., :1])], dim=-1).contiguous()

    return ScanParams(trans=trans, K=packed)


def scanprep_bwd_ref(
    dtrans: Tensor,
    dK: Tensor,
    params: Tensor,
    param_bias: Tensor,
    /,
    *,
    heads: int,
    w_max: float,
) -> ScanGrads:
    """Pullback of :func:`scanprep_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently. In float64 this is the gradient authority the
    kernel is measured against.

    Args:
        dtrans: Cotangent of ``trans``, ``(B,H,T,4)``.
        dK: Cotangent of ``K``, ``(B,H,T,2,4)``. Lane 3 is the cotangent of a
            constant and is discarded.
        params: The forward's projection slice, ``(B,T,H*PARAM_COLS)``.
        param_bias: The forward's bias, ``(H,PARAM_COLS)``. The maps' Jacobians
            are evaluated at ``params + param_bias``, so the bias is saved too.
        heads: ``H``.
        w_max: The forward's norm bound.

    Returns:
        A :class:`ScanGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, or a ``w_max`` outside
            ``(0, pi)``.
        TypeError: On an unsupported dtype.
    """
    check_cotangents(dtrans, dK, params, param_bias, heads)
    pl = params.detach().requires_grad_(True)
    cl = param_bias.detach().requires_grad_(True)
    with torch.enable_grad():
        out = scanprep_ref(pl, cl, heads=heads, w_max=w_max)
    dparams, dparam_bias = torch.autograd.grad(
        (out.trans, out.K), (pl, cl), (dtrans, dK)
    )
    return ScanGrads(dparams=dparams.contiguous(), dparam_bias=dparam_bias)


def check_cotangents(
    dtrans: Tensor,
    dK: Tensor,
    params: Tensor,
    param_bias: Tensor,
    heads: int,
) -> tuple[int, int]:
    """Validate the backward's operand set.

    Shared by both backends for the same reason as :func:`check_operands`.

    Args:
        dtrans: Cotangent of ``trans``.
        dK: Cotangent of ``K``.
        params: The forward's projection slice.
        param_bias: The forward's bias.
        heads: ``H``.

    Returns:
        ``(B, T)``.

    Raises:
        ValueError: On a rank or shape mismatch.
        TypeError: On an unsupported dtype.
    """
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    want = heads * PARAM_COLS
    if params.ndim != 3 or params.shape[-1] != want:
        raise ValueError(
            f"params must be (B,T,{want}) at heads={heads}, got {tuple(params.shape)}"
        )
    check_supported(params, "params")
    check_pinned(param_bias, "param_bias")
    if tuple(param_bias.shape) != (heads, PARAM_COLS):
        raise ValueError(
            f"param_bias must be {(heads, PARAM_COLS)}, got {tuple(param_bias.shape)}"
        )
    bsz, seqlen = int(params.shape[0]), int(params.shape[1])
    if tuple(dtrans.shape) != (bsz, heads, seqlen, 4):
        raise ValueError(
            f"dtrans must be {(bsz, heads, seqlen, 4)}, got {tuple(dtrans.shape)}"
        )
    if tuple(dK.shape) != (bsz, heads, seqlen, 2, 4):
        raise ValueError(
            f"dK must be {(bsz, heads, seqlen, 2, 4)}, got {tuple(dK.shape)}"
        )
    return bsz, seqlen
