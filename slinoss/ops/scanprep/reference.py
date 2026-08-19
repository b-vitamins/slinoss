"""Bounded parameter maps. Pure-PyTorch reference.

Turns unconstrained projections into the packed `trans` and `K` the scan reads.
The numerical invariants the kernels rely on hold by construction here, so no
kernel needs a clamp, an epsilon, or a validity pass:

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

from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = [
    "ScanGrads",
    "ScanParams",
    "bounded_logscale",
    "bounded_rotvec",
    "scanprep_bwd_ref",
    "scanprep_ref",
]


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


class ScanParams(NamedTuple):
    """Packed scan parameters.

    Attributes:
        trans: ``(w_x, w_y, w_z, ls)``, shape ``(B,H,T,4)``, pinned dtype.
        K: Per tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned dtype. Tap
            index 0 is previous and 1 is current. Lane 3 is a hard zero, present
            for float4 alignment.
    """

    trans: Tensor
    K: Tensor


def scanprep_ref(
    w_raw: Tensor,
    ls_raw: Tensor,
    tap_raw: Tensor,
    *,
    w_max: float,
) -> ScanParams:
    """Apply the bounded maps and pack.

    Args:
        w_raw: Unconstrained rotation vectors, shape ``(B,H,T,3)``.
        ls_raw: Unconstrained log-scales, shape ``(B,H,T)``.
        tap_raw: Unconstrained taps ``(kr, g, h)``, shape ``(B,H,T,2,3)``.
        w_max: Rotation-vector norm bound, in ``(0, pi)``.

    Returns:
        A :class:`ScanParams` in the pinned dtype. Low-precision inputs are
        promoted: the transition never carries a low-precision dtype.

    Raises:
        ValueError: On a shape mismatch or a ``w_max`` outside ``(0, pi)``.
        TypeError: On an unsupported dtype.
    """
    if w_raw.ndim != 4 or w_raw.shape[-1] != 3:
        raise ValueError(f"w_raw must be (B,H,T,3), got {tuple(w_raw.shape)}")
    lead = tuple(int(d) for d in w_raw.shape[:3])
    if tuple(ls_raw.shape) != lead:
        raise ValueError(f"ls_raw must be {lead}, got {tuple(ls_raw.shape)}")
    if tuple(tap_raw.shape) != (*lead, 2, 3):
        raise ValueError(f"tap_raw must be {(*lead, 2, 3)}, got {tuple(tap_raw.shape)}")
    check_supported(w_raw, "w_raw")
    check_supported(ls_raw, "ls_raw")
    check_supported(tap_raw, "tap_raw")

    dtype = pinned_dtype(w_raw, ls_raw, tap_raw)
    with autocast_disabled(w_raw.device.type):
        w = bounded_rotvec(w_raw.to(dtype), w_max)
        ls = bounded_logscale(ls_raw.to(dtype))
        tap = tap_raw.to(dtype)
        return ScanParams(
            trans=torch.cat([w, ls[..., None]], dim=-1).contiguous(),
            K=torch.cat([tap, torch.zeros_like(tap[..., :1])], dim=-1).contiguous(),
        )


class ScanGrads(NamedTuple):
    """Gradients of the bounded maps.

    Attributes:
        dw_raw: ``(B,H,T,3)``, dtype of ``w_raw``.
        dls_raw: ``(B,H,T)``, dtype of ``ls_raw``.
        dtap_raw: ``(B,H,T,2,3)``, dtype of ``w_raw``.
    """

    dw_raw: Tensor
    dls_raw: Tensor
    dtap_raw: Tensor


def scanprep_bwd_ref(
    dtrans: Tensor,
    dK: Tensor,
    w_raw: Tensor,
    ls_raw: Tensor,
    /,
    *,
    w_max: float,
) -> ScanGrads:
    """Pullback of :func:`scanprep_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently. In float64 this is the gradient authority the
    kernel is measured against.

    ``tap_raw`` is not a parameter: the tap map is the identity, so its pullback
    depends on nothing but ``dK``. The leaf differentiated here is a zero of the
    right shape and dtype, which the linearity makes exact.

    Args:
        dtrans: Cotangent of ``trans``, shape ``(B,H,T,4)``.
        dK: Cotangent of ``K``, shape ``(B,H,T,2,4)``. Lane 3 is the cotangent of
            a constant and is discarded.
        w_raw: The forward's rotation vectors, shape ``(B,H,T,3)``.
        ls_raw: The forward's log-scales, shape ``(B,H,T)``.
        w_max: The forward's norm bound.

    Returns:
        A :class:`ScanGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, or a ``w_max`` outside
            ``(0, pi)``.
        TypeError: On an unsupported dtype.
    """
    if w_raw.ndim != 4 or w_raw.shape[-1] != 3:
        raise ValueError(f"w_raw must be (B,H,T,3), got {tuple(w_raw.shape)}")
    lead = tuple(int(d) for d in w_raw.shape[:3])
    if tuple(dtrans.shape) != (*lead, 4):
        raise ValueError(f"dtrans must be {(*lead, 4)}, got {tuple(dtrans.shape)}")
    if tuple(dK.shape) != (*lead, 2, 4):
        raise ValueError(f"dK must be {(*lead, 2, 4)}, got {tuple(dK.shape)}")

    wl = w_raw.detach().requires_grad_(True)
    ll = ls_raw.detach().requires_grad_(True)
    tl = torch.zeros(
        *lead, 2, 3, dtype=w_raw.dtype, device=w_raw.device, requires_grad=True
    )
    with torch.enable_grad():
        out = scanprep_ref(wl, ll, tl, w_max=w_max)
    dw_raw, dls_raw, dtap_raw = torch.autograd.grad(
        (out.trans, out.K), (wl, ll, tl), (dtrans, dK)
    )
    return ScanGrads(dw_raw=dw_raw, dls_raw=dls_raw, dtap_raw=dtap_raw)
