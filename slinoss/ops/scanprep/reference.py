"""The scan's parameter frontier. Pure-PyTorch reference.

Takes the token-major slices of one projection output and emits every per-token
operand the scan reads except ``U``: the packed transition ``trans``, the packed
taps ``K``, and the head-major ``B`` and ``C``.

Both operands are slices of a single ``(B,T,W)`` projection output, so neither is
contiguous: the trailing axis has unit stride and the row stride is the full
projection width. Nothing here repacks them.

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
from slinoss.config import STATE_MULTIPLE

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
    """Every per-token operand the scan reads except ``U``.

    Attributes:
        trans: ``(w_x, w_y, w_z, ls)``, shape ``(B,H,T,4)``, pinned dtype.
        K: Per tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned dtype. Tap
            index 0 is previous and 1 is current. Lane 3 is a hard zero, present
            for float4 alignment.
        B: ``(B,G,T,3N)``, activation dtype, contiguous.
        C: ``(B,G,T,3N)``, activation dtype, contiguous.
    """

    trans: Tensor
    K: Tensor
    B: Tensor
    C: Tensor


class ScanGrads(NamedTuple):
    """Gradients of :func:`scanprep_ref`.

    Attributes:
        dparams: ``(B,T,H*PARAM_COLS)``, contiguous, dtype of ``params``.
        dbc: ``(B,T,2*G*3N)``, contiguous, dtype of ``bc``.
        dparam_bias: ``(H,PARAM_COLS)``, dtype of ``param_bias``.
    """

    dparams: Tensor
    dbc: Tensor
    dparam_bias: Tensor


def _groups(bc: Tensor, state_dim: int, heads: int) -> int:
    """``G``, read off ``bc`` rather than taken on trust.

    A caller that passed ``G`` could claim one grouping and hand over another, so
    the grouping is a property of the operand.

    Args:
        bc: The concatenated ``B``/``C`` operand, ``(B,T,2*G*3N)``.
        state_dim: ``3N``.
        heads: ``H``.

    Returns:
        ``G``.

    Raises:
        ValueError: If the trailing width is not ``2*G*3N`` for a positive ``G``
            dividing ``heads``.
    """
    width = int(bc.shape[-1])
    pair = 2 * state_dim
    if width % pair != 0 or width // pair < 1:
        raise ValueError(
            f"bc must be (B,T,2*G*{state_dim}) for a positive G, "
            f"got trailing width {width}"
        )
    groups = width // pair
    if heads % groups != 0:
        raise ValueError(
            f"G {groups}, read off bc, does not divide heads {heads}; "
            f"a group holds a whole number of heads"
        )
    return groups


def check_operands(
    params: Tensor,
    bc: Tensor,
    param_bias: Tensor,
    heads: int,
    state_dim: int,
) -> int:
    """Validate the operand set and return ``G``.

    The shape, stride, and grouping contract every backend shares. The kernel host
    path calls this rather than restating it, so the two cannot disagree; the CuTe
    path adds only what is its own, namely device residency, base alignment, and
    the narrower kernel dtype set.

    Args:
        params: ``(B,T,H*PARAM_COLS)``.
        bc: ``(B,T,2*G*3N)``.
        param_bias: ``(H,PARAM_COLS)``.
        heads: ``H``.
        state_dim: ``3N``.

    Returns:
        ``G``.

    Raises:
        ValueError: On a non-positive ``heads``, a ``state_dim`` that is not a
            positive multiple of :data:`slinoss.config.STATE_MULTIPLE`, a rank or
            shape mismatch, or a trailing axis whose stride is not one.
        TypeError: On an unsupported dtype, on two activation dtypes, or on a
            low-precision ``param_bias``.
    """
    if heads < 1:
        raise ValueError(f"heads must be positive, got {heads}")
    if state_dim < STATE_MULTIPLE or state_dim % STATE_MULTIPLE != 0:
        raise ValueError(
            f"state_dim is 3N with N a multiple of {STATE_MULTIPLE // 3}, so it "
            f"must be a positive multiple of {STATE_MULTIPLE}; got {state_dim}"
        )
    want = heads * PARAM_COLS
    if params.ndim != 3 or params.shape[-1] != want:
        raise ValueError(
            f"params must be (B,T,{want}) at heads={heads}, got {tuple(params.shape)}"
        )
    lead = (int(params.shape[0]), int(params.shape[1]))
    if bc.ndim != 3 or tuple(bc.shape[:2]) != lead:
        raise ValueError(
            f"bc must be ({lead[0]},{lead[1]},2*G*3N), got {tuple(bc.shape)}"
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
    if bc.stride(-1) != 1:
        raise ValueError(
            f"bc must have unit stride on its trailing axis, got {bc.stride(-1)}"
        )
    check_supported(params, "params")
    check_supported(bc, "bc")
    check_pinned(param_bias, "param_bias")
    if bc.dtype is not params.dtype:
        raise TypeError(
            f"bc is {bc.dtype} and params is {params.dtype}; "
            "both are slices of one projection, so one activation dtype per call"
        )
    return _groups(bc, state_dim, heads)


def scanprep_ref(
    params: Tensor,
    bc: Tensor,
    param_bias: Tensor,
    *,
    heads: int,
    state_dim: int,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps, pack, and permute ``bc`` head-major.

    Args:
        params: Projection slice, ``(B,T,H*PARAM_COLS)``, activation dtype.
            Trailing stride one; the row stride is the projection width. Per head,
            in order ``(w_x, w_y, w_z, ls, kr0, g0, h0, kr1, g1, h1)``.
        bc: Projection slice, ``(B,T,2*G*3N)``, same dtype. The first ``G*3N``
            columns are ``B`` and the second ``G*3N`` are ``C``; within each half
            group ``g`` occupies columns ``g*3N`` to ``(g+1)*3N``.
        param_bias: ``(H,PARAM_COLS)``, float32, added to every token's row
            before the maps.
        heads: ``H``.
        state_dim: ``3N``. Positive multiple of
            :data:`slinoss.config.STATE_MULTIPLE`. ``G`` is
            ``bc.shape[-1] // (2 * state_dim)``, read off the operand.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`ScanParams`. ``trans`` and ``K`` are in the pinned dtype (I4);
        ``B`` and ``C`` keep the activation dtype. All four are contiguous.

    Raises:
        ValueError: On a shape mismatch, a trailing stride other than one, a ``G``
            that does not divide ``heads``, or a ``w_max`` outside ``(0, pi)``.
        TypeError: On an unsupported dtype, on two activation dtypes, or on a
            low-precision ``param_bias``.
    """
    groups = check_operands(params, bc, param_bias, heads, state_dim)
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

    # (B,T,2,G,3N) -> (B,2,G,T,3N). A permute, so its pullback is a permute and
    # needs nothing saved.
    split = bc.unflatten(-1, (2, groups, state_dim)).permute(0, 2, 3, 1, 4)
    return ScanParams(
        trans=trans,
        K=packed,
        B=split[:, 0].contiguous(),
        C=split[:, 1].contiguous(),
    )


def scanprep_bwd_ref(
    dtrans: Tensor,
    dK: Tensor,
    dB: Tensor,
    dC: Tensor,
    params: Tensor,
    param_bias: Tensor,
    /,
    *,
    heads: int,
    state_dim: int,
    w_max: float,
) -> ScanGrads:
    """Pullback of :func:`scanprep_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently. In float64 this is the gradient authority the
    kernel is measured against.

    ``bc`` is not a parameter: the permute is linear, so its pullback depends on
    nothing but ``dB`` and ``dC``. The leaf differentiated here is a zero of the
    right shape and dtype, which the linearity makes exact.

    Args:
        dtrans: Cotangent of ``trans``, ``(B,H,T,4)``.
        dK: Cotangent of ``K``, ``(B,H,T,2,4)``. Lane 3 is the cotangent of a
            constant and is discarded.
        dB: Cotangent of ``B``, ``(B,G,T,3N)``.
        dC: Cotangent of ``C``, ``(B,G,T,3N)``.
        params: The forward's projection slice, ``(B,T,H*PARAM_COLS)``.
        param_bias: The forward's bias, ``(H,PARAM_COLS)``. The maps' Jacobians
            are evaluated at ``params + param_bias``, so the bias is saved too.
        heads: ``H``.
        state_dim: ``3N``.
        w_max: The forward's norm bound.

    Returns:
        A :class:`ScanGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, or a ``w_max`` outside
            ``(0, pi)``.
        TypeError: On an unsupported dtype.
    """
    bsz, seqlen, groups = check_cotangents(
        dtrans, dK, dB, dC, params, param_bias, heads, state_dim
    )
    pl = params.detach().requires_grad_(True)
    bl = torch.zeros(
        bsz,
        seqlen,
        2 * groups * state_dim,
        dtype=params.dtype,
        device=params.device,
        requires_grad=True,
    )
    cl = param_bias.detach().requires_grad_(True)
    with torch.enable_grad():
        out = scanprep_ref(pl, bl, cl, heads=heads, state_dim=state_dim, w_max=w_max)
    dparams, dbc, dparam_bias = torch.autograd.grad(
        (out.trans, out.K, out.B, out.C), (pl, bl, cl), (dtrans, dK, dB, dC)
    )
    return ScanGrads(
        dparams=dparams.contiguous(),
        dbc=dbc.contiguous(),
        dparam_bias=dparam_bias,
    )


def check_cotangents(
    dtrans: Tensor,
    dK: Tensor,
    dB: Tensor,
    dC: Tensor,
    params: Tensor,
    param_bias: Tensor,
    heads: int,
    state_dim: int,
) -> tuple[int, int, int]:
    """Validate the backward's operand set.

    Shared by both backends for the same reason as :func:`check_operands`. ``G`` is
    read off ``dB`` rather than passed.

    Args:
        dtrans: Cotangent of ``trans``.
        dK: Cotangent of ``K``.
        dB: Cotangent of ``B``.
        dC: Cotangent of ``C``.
        params: The forward's projection slice.
        param_bias: The forward's bias.
        heads: ``H``.
        state_dim: ``3N``.

    Returns:
        ``(B, T, G)``.

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
    if dB.ndim != 4 or tuple(dB.shape[2:]) != (seqlen, state_dim):
        raise ValueError(
            f"dB must be ({bsz},G,{seqlen},{state_dim}), got {tuple(dB.shape)}"
        )
    if tuple(dC.shape) != tuple(dB.shape):
        raise ValueError(f"dC must be {tuple(dB.shape)}, got {tuple(dC.shape)}")
    groups = int(dB.shape[1])
    if int(dB.shape[0]) != bsz or groups < 1 or heads % groups != 0:
        raise ValueError(
            f"dB must be ({bsz},G,{seqlen},{state_dim}) with G dividing "
            f"heads {heads}, got {tuple(dB.shape)}"
        )
    return bsz, seqlen, groups
