"""Bounded parameter maps: the device-side implementation, and the only one.

    w  = w_max * raw * rsqrt(1 + |raw|^2)
    ls = -softplus(raw)
    K  = tap, unchanged

Both maps and both pullbacks live here as plain Python functions over
:data:`slinoss._cute.Scalar`, so a call from inside a ``@cute.kernel`` is inlined
at trace time. The kernels in :mod:`slinoss.ops.scanprep.cute.frontier` are the
only callers. A second copy of either map would diverge from this one, and the
divergence is a correctness bug.

Nothing here emits dynamic control flow. ``rsqrt`` acts on ``1 + |raw|^2 >= 1``
and ``softplus`` is evaluated through an identity whose exponential argument is
never positive, so I1 and I2 are produced without a clamp, an epsilon, or a
validity pass, and no function here contributes divergence.

Every quantity is float32 (I4), whatever width the raw parameters were stored at.
"""

from typing import Any

import cutlass
import cutlass.cute as cute

from slinoss._cute import LOG2_E, Scalar, f32, select

__all__ = [
    "log_scale",
    "log_scale_grad",
    "rotvec",
    "rotvec_grad",
]


def _softplus_parts(raw: Scalar) -> tuple[Any, Scalar]:
    """``(raw > 0, exp(-|raw|))``.

    The exponent is non-positive at every input, so the value lies in ``(0, 1]``
    and no input magnitude overflows. The absolute value is a select, not a
    branch, so the predicate costs one predicated move and no divergence.
    """
    positive = raw > cutlass.Float32(0.0)
    return positive, f32(cute.exp2(select(positive, -raw, raw) * LOG2_E))


def log_scale(raw: Scalar) -> Scalar:
    """``-softplus(raw) <= 0`` (I1).

    Evaluated as ``min(-raw, 0) - log1p(exp(-|raw|))``. Both terms are bounded by
    ``|raw|`` for every finite input, so neither the exponential nor the sum can
    overflow, and both halves are selects on one predicate.

    ``log1p`` is formed as ``log(1 + e)``. That addition drops the part of ``e``
    below float32 epsilon, which is an absolute error of at most ``2^-24`` on a
    quantity whose magnitude is ``|raw|``, and it drops nothing at all wherever
    ``e`` is normal against one.

    Args:
        raw: Unconstrained log-scale, float32.

    Returns:
        The log-scale, non-positive.
    """
    positive, small = _softplus_parts(raw)
    return select(positive, -raw, cutlass.Float32(0.0)) - f32(cute.log(small + 1.0))


def log_scale_grad(raw: Scalar) -> Scalar:
    """``d(-softplus)/draw = -sigmoid(raw)``.

    ``sigmoid(raw)`` is ``1 / (1 + e)`` where ``raw > 0`` and ``e / (1 + e)``
    elsewhere, with ``e = exp(-|raw|)`` in ``(0, 1]``: one select, and no
    intermediate exceeds one, so no input magnitude overflows.

    Args:
        raw: Unconstrained log-scale, float32.

    Returns:
        The derivative of :func:`log_scale`.
    """
    positive, small = _softplus_parts(raw)
    return -select(positive, cutlass.Float32(1.0), small) / (small + 1.0)


def rotvec(
    rx: Scalar, ry: Scalar, rz: Scalar, w_max: Scalar
) -> tuple[Scalar, Scalar, Scalar]:
    """Map an unconstrained vector into the closed ball of radius ``w_max`` (I2).

    ``1 + |raw|^2 >= 1``, so the rsqrt is regular over the whole domain and needs
    no guard. An overflowing ``|raw|^2`` gives ``rsqrt(inf) == 0``, which collapses
    the result to the centre of the ball: finite, and still inside it.

    Args:
        rx: First component of the unconstrained vector, float32.
        ry: Second component.
        rz: Third component.
        w_max: Radius bound. Checked against ``(0, pi)`` on the host.

    Returns:
        ``(w_x, w_y, w_z)`` with ``|w| <= w_max``.
    """
    scale = w_max * f32(cute.rsqrt(rx * rx + ry * ry + rz * rz + 1.0))
    return rx * scale, ry * scale, rz * scale


def rotvec_grad(
    rx: Scalar,
    ry: Scalar,
    rz: Scalar,
    gx: Scalar,
    gy: Scalar,
    gz: Scalar,
    w_max: Scalar,
) -> tuple[Scalar, Scalar, Scalar]:
    """Pullback of :func:`rotvec`, evaluated at the raw vector.

    The map is a radial rescaling, so its Jacobian is the scale times a rank-one
    correction along ``raw``; ``inv * inv`` is ``1 / (1 + |raw|^2)``. The raw
    vector is the argument rather than ``w``: the pullback is a function of
    ``raw``, and recovering ``raw`` from ``w`` would invert a saturating map.

    Args:
        rx: First component of the unconstrained vector, float32.
        ry: Second component.
        rz: Third component.
        gx: First component of the cotangent of ``w``.
        gy: Second component.
        gz: Third component.
        w_max: The bound the forward used.

    Returns:
        The cotangent of the unconstrained vector.
    """
    inv = f32(cute.rsqrt(rx * rx + ry * ry + rz * rz + 1.0))
    scale = w_max * inv
    pull = inv * inv * (gx * rx + gy * ry + gz * rz)
    return scale * (gx - pull * rx), scale * (gy - pull * ry), scale * (gz - pull * rz)
