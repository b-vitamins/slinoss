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

__all__ = ["ScanParams", "bounded_logscale", "bounded_rotvec", "scanprep_ref"]


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
