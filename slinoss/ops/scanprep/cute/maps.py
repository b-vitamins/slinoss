"""Bounded parameter maps: the device-side implementation, and the only one.

    raw = bias + band
    w   = w_max * raw * rsqrt(1 + |raw|^2/4)
    ls  = -LS_MAX_MAG * sigmoid(band + bias)
    K   = the first-order-hold moments of exp(2*ls*I + [w]_x)

Every map and every pullback lives here as plain Python functions over
:data:`slinoss._cute.Scalar`, so a call from inside a ``@cute.kernel`` is inlined
at trace time. The kernels in :mod:`slinoss.ops.scanprep.cute.frontier` are the
only callers. A second copy of any map would diverge from this one, and the
divergence is a correctness bug.

Nothing here emits dynamic control flow. ``rsqrt`` acts on ``1 + |raw|^2/4 >= 1``
and ``sigmoid`` is evaluated through an identity whose exponential argument is
never positive, so I1 and I2 are produced without a clamp, an epsilon, or a
validity pass. The tap series and the
tap recurrence are both evaluated and one is selected, so the removable
singularity at the origin costs a predicated move rather than a branch: no
function here contributes divergence.

Every quantity is float32 (I4), whatever width the raw parameters were stored at.
"""

import math

import cutlass
import cutlass.cute as cute

from slinoss._cute import LOG2_E, Scalar, f32, select
from slinoss.ops.scanprep.reference import (
    FOH_TAYLOR_RADIUS_SQ,
    FP32_FOH_TERMS,
    LS_MAX_MAG,
    T2_FLOOR,
    foh_coeffs,
)
from slinoss.ops.so3ssd.cute.common import COS_HALF, SINC_HALF

__all__ = [
    "foh_taps",
    "foh_taps_grad",
    "log_scale",
    "log_scale_grad",
    "rotvec",
    "rotvec_grad",
]

_FOH_SERIES = tuple(foh_coeffs(order, FP32_FOH_TERMS) for order in (1, 2, 3))


def _folded_exp(raw: Scalar, gain: float) -> Scalar:
    """``exp(-gain*|raw|)``, ``gain > 0``.

    The exponent is non-positive at every input, so the value lies in ``(0, 1]``
    and no input magnitude overflows. The absolute value is a select, not a
    branch, so the fold costs one predicated move and no divergence.
    """
    positive = raw > cutlass.Float32(0.0)
    return f32(cute.exp2(select(positive, -raw, raw) * (gain * LOG2_E)))


def _sigmoid(raw: Scalar) -> Scalar:
    """``sigmoid(raw)``, in ``(0, 1)``.

    ``1 / (1 + e)`` where ``raw > 0`` and ``e / (1 + e)`` elsewhere, with
    ``e = exp(-|raw|)`` in ``(0, 1]``: one select, and no intermediate exceeds one,
    so no input magnitude overflows.
    """
    small = _folded_exp(raw, 1.0)
    return select(raw > cutlass.Float32(0.0), cutlass.Float32(1.0), small) / (
        small + 1.0
    )


def log_scale(raw: Scalar) -> Scalar:
    """``-LS_MAX_MAG * sigmoid(raw)``, in ``[-LS_MAX_MAG, 0]`` (I1).

    Args:
        raw: Unconstrained log-scale, float32.

    Returns:
        The log-scale, non-positive.
    """
    return -LS_MAX_MAG * _sigmoid(raw)


def log_scale_grad(raw: Scalar) -> Scalar:
    """``d(-LS_MAX_MAG*sigmoid)/draw = LS_MAX_MAG*s*(s - 1)`` at ``s = sigmoid(raw)``.

    Args:
        raw: Unconstrained log-scale, float32.

    Returns:
        The derivative of :func:`log_scale`.
    """
    s = _sigmoid(raw)
    return LS_MAX_MAG * s * (s - 1.0)


def rotvec(
    rx: Scalar, ry: Scalar, rz: Scalar, w_max: Scalar
) -> tuple[Scalar, Scalar, Scalar]:
    """Map an unconstrained vector into the closed ball of radius ``2*w_max`` (I2).

    ``1 + |raw|^2/4 >= 1``, so the rsqrt is regular over the whole domain and
    needs no guard. The scale at the origin is unchanged, and ``|w| = w_max`` is
    reached at finite raw radius.

    Args:
        rx: First component of the unconstrained vector, float32.
        ry: Second component.
        rz: Third component.
        w_max: Radius bound. Checked against ``(0, pi)`` on the host.

    Returns:
        ``(w_x, w_y, w_z)`` with ``|w| <= 2*w_max``.
    """
    scale = w_max * f32(cute.rsqrt(0.25 * (rx * rx + ry * ry + rz * rz) + 1.0))
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
    correction along ``raw``; ``0.25*inv*inv`` is
    ``0.25 / (1 + |raw|^2/4)``. The raw
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
    inv = f32(cute.rsqrt(0.25 * (rx * rx + ry * ry + rz * rz) + 1.0))
    scale = w_max * inv
    pull = 0.25 * inv * inv * (gx * rx + gy * ry + gz * rz)
    return scale * (gx - pull * rx), scale * (gy - pull * ry), scale * (gz - pull * rz)


def _horner_r(s: Scalar, coeffs: tuple[float, ...]) -> Scalar:
    out = cutlass.Float32(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * s + coeff
    return out


def _horner_c(p: Scalar, t: Scalar, coeffs: tuple[float, ...]) -> tuple[Scalar, Scalar]:
    re = cutlass.Float32(coeffs[-1])
    im = cutlass.Float32(0.0)
    for coeff in reversed(coeffs[:-1]):
        re, im = re * p + coeff - im * t, re * t + im * p
    return re, im


def _phi(p: Scalar, t: Scalar, orders: int) -> tuple[tuple[Scalar, Scalar], ...]:
    """``phi_1 .. phi_orders`` at ``p + i*t``, each as an ``(re, im)`` pair.

    ``phi_k(x) = sum_n x^n / (n + k)!``, by the recurrence ``phi_{k+1} = (phi_k -
    1/k!)/x`` from ``phi_0 = exp(x)``. The recurrence divides, so the series is
    summed instead inside :data:`slinoss.ops.scanprep.reference.FOH_TAYLOR_RADIUS_SQ`
    and both are evaluated unconditionally. The recurrence's value there is a
    division by a norm that reaches zero, so it is not merely inaccurate but
    infinite; a select discards it, which no arithmetic downstream of the select
    can observe.

    ``exp(p + i*t)`` needs ``cos t`` and ``sin(t)/t``, which come from the
    transition's own half-angle series at the double angle: ``cos t = 1 - 2
    sin^2(t/2)`` and ``sin(t)/t = sinc(t/2) cos(t/2)``. Evaluating the series at
    the half angle keeps its argument inside the truncation
    :data:`slinoss.ops.so3ssd.cute.common.FP32_SERIES_TERMS` was sized for.

    Args:
        p: Real part, non-positive by I1.
        t: Imaginary part, in ``[0, 2*w_max]`` by I2.
        orders: How many ``phi`` to return, at least one.

    Returns:
        ``orders`` pairs, ``phi_1`` first.
    """
    s = t * t
    norm = p * p + s
    inv = cutlass.Float32(1.0) / norm
    scale = f32(cute.exp2(p * LOG2_E))
    half_sinc = _horner_r(s, SINC_HALF)
    re = scale * (cutlass.Float32(1.0) - 0.5 * s * half_sinc * half_sinc)
    im = scale * t * half_sinc * _horner_r(s, COS_HALF)
    small = norm < FOH_TAYLOR_RADIUS_SQ
    out: list[tuple[Scalar, Scalar]] = []
    for order in range(1, orders + 1):
        re = re - 1.0 / float(math.factorial(order - 1))
        re, im = (re * p + im * t) * inv, (im * p - re * t) * inv
        series = _horner_c(p, t, _FOH_SERIES[order - 1])
        out.append((select(small, series[0], re), select(small, series[1], im)))
    return tuple(out)


def _chart(
    per: tuple[Scalar, Scalar], par: Scalar, inv_t2: Scalar, inv_t: Scalar
) -> tuple[Scalar, Scalar, Scalar]:
    """``(kr, g, h)`` from a tap's three eigenvalues."""
    return per[0], (par - per[0]) * inv_t2, per[1] * inv_t


def _radial(wx: Scalar, wy: Scalar, wz: Scalar) -> tuple[Scalar, Scalar, Scalar]:
    """``(|w|^2, 1/|w|^2, 1/|w|)``, the norm floored at ``T2_FLOOR``."""
    t2 = wx * wx + wy * wy + wz * wz
    t2 = select(t2 > T2_FLOOR, t2, cutlass.Float32(T2_FLOOR))
    inv_t = f32(cute.rsqrt(t2))
    return t2, inv_t * inv_t, inv_t


def foh_taps(
    wx: Scalar, wy: Scalar, wz: Scalar, ls: Scalar
) -> tuple[tuple[Scalar, Scalar, Scalar], tuple[Scalar, Scalar, Scalar]]:
    """First-order-hold taps of the step ``exp(2*ls*I + [w]_x)``, on the tap chart.

    The generator is a scalar plus a skew part, hence normal, with eigenvalue ``p =
    2*ls`` along ``w`` and ``p + i*|w|`` across it. The two moments are
    ``phi_1 - phi_2`` and ``phi_2`` at those eigenvalues; see
    :func:`slinoss.ops.scanprep.reference.foh_taps`, which is the authority this
    matches.

    Args:
        wx: First component of the rotation vector, float32.
        wy: Second component.
        wz: Third component.
        ls: The log-scale, non-positive by I1.

    Returns:
        ``(kr, g, h)`` for the previous tap and for the current tap.
    """
    t2, inv_t2, inv_t = _radial(wx, wy, wz)
    axial = 2.0 * ls
    plane = _phi(axial, t2 * inv_t, 2)
    real = _phi(axial, cutlass.Float32(0.0), 2)
    return (
        _chart(
            (plane[0][0] - plane[1][0], plane[0][1] - plane[1][1]),
            real[0][0] - real[1][0],
            inv_t2,
            inv_t,
        ),
        _chart(plane[1], real[1][0], inv_t2, inv_t),
    )


def foh_taps_grad(
    wx: Scalar,
    wy: Scalar,
    wz: Scalar,
    ls: Scalar,
    dprev: tuple[Scalar, Scalar, Scalar],
    dcurr: tuple[Scalar, Scalar, Scalar],
) -> tuple[Scalar, Scalar, Scalar, Scalar]:
    """Pullback of :func:`foh_taps`, evaluated at the mapped transition.

    ``phi_1' = phi_1 - phi_2`` and ``phi_2' = phi_2 - 2 phi_3``, so the derivative
    of both taps comes from one more order of the same recurrence and needs no
    second series. The chart's own two derivatives are the ones that divide: an
    entry carrying ``1/|w|^2`` contributes ``-g/|w|^2`` to the cotangent of
    ``|w|^2``, and one carrying ``1/|w|`` contributes ``-h/|w|``.

    Args:
        wx: First component of the rotation vector, float32.
        wy: Second component.
        wz: Third component.
        ls: The log-scale.
        dprev: Cotangent of the previous tap's ``(kr, g, h)``.
        dcurr: Cotangent of the current tap's.

    Returns:
        Cotangent of ``(w_x, w_y, w_z, ls)``, the taps' contribution alone.
    """
    t2, inv_t2, inv_t = _radial(wx, wy, wz)
    axial = 2.0 * ls
    plane = _phi(axial, t2 * inv_t, 3)
    real = _phi(axial, cutlass.Float32(0.0), 3)
    # Per tap: the value across the plane, its derivative there, and both along
    # the axis. The previous tap is phi_1 - phi_2 and the current one is phi_2.
    taps = (
        (
            (plane[0][0] - plane[1][0], plane[0][1] - plane[1][1]),
            (
                plane[0][0] - 2.0 * plane[1][0] + 2.0 * plane[2][0],
                plane[0][1] - 2.0 * plane[1][1] + 2.0 * plane[2][1],
            ),
            real[0][0] - real[1][0],
            real[0][0] - 2.0 * real[1][0] + 2.0 * real[2][0],
            dprev,
        ),
        (
            plane[1],
            (plane[1][0] - 2.0 * plane[2][0], plane[1][1] - 2.0 * plane[2][1]),
            real[1][0],
            real[1][0] - 2.0 * real[2][0],
            dcurr,
        ),
    )
    dp = cutlass.Float32(0.0)
    dt2 = cutlass.Float32(0.0)
    for per, deriv, par, axial_deriv, cot in taps:
        chart = _chart(per, par, inv_t2, inv_t)
        dpar = cot[1] * inv_t2
        dper_re = cot[0] - dpar
        dper_im = cot[2] * inv_t
        dp = dp + dper_re * deriv[0] + dper_im * deriv[1] + dpar * axial_deriv
        dt2 = dt2 - (cot[1] * chart[1] + 0.5 * cot[2] * chart[2]) * inv_t2
        dt2 = dt2 + 0.5 * (dper_im * deriv[0] - dper_re * deriv[1]) * inv_t
    radial = 2.0 * dt2
    return radial * wx, radial * wy, radial * wz, 2.0 * dp
