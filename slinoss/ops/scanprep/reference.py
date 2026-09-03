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

- ``-LS_MAX_MAG <= ls = -LS_MAX_MAG*sigmoid(x) <= 0``, so every chunk-local
  log-scale prefix is monotone non-increasing and every decay factor lies in
  ``(0,1]``. Overflow is unreachable and underflow is graceful. Bounded below as
  well, which no kernel reads: see :data:`LS_MAX_MAG`.
- ``|w| = w_max * |x| / sqrt(1 + |x|^2/4) <= 2*w_max < 2*pi``, so the
  quaternion exponential is a single branchless polynomial over the whole
  reachable domain. The map is analytic in ``x``: ``1 + |x|^2/4 >= 1``, so the
  rsqrt has no singularity and needs no guard. A half turn lies at a finite raw
  radius instead of at the chart's asymptote.

The rotation-vector row a token presents to that second map is the raw sum of its
projection columns and the head's bias. The outer radial map is the one bound:
the token projection retains an unconstrained chart and a unit pullback at its
zero initialization. See :func:`anchored_rotvec`.

Taps are not parameters. They are the first-order-hold moments of the transition
the two maps above define, so they carry no columns and no initialization:
:func:`foh_taps` computes them from ``(w, ls)`` in closed form.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from slinoss._guard import check_pitched
from slinoss._precision import (
    autocast_disabled,
    check_pinned,
    check_supported,
    pinned_dtype,
)
from slinoss.config import ROTATION_CHART_SCALE_MAX

__all__ = [
    "DRIVE_CEIL_SQ",
    "DRIVE_FLOOR_SQ",
    "FOH_TAYLOR_RADIUS_SQ",
    "FP32_FOH_TERMS",
    "FP64_FOH_TERMS",
    "LS_COLUMN",
    "LS_MAX_MAG",
    "PARAM_COLS",
    "ROTVEC_COLUMNS",
    "T2_FLOOR",
    "ScanGrads",
    "ScanParams",
    "anchored_rotvec",
    "bounded_logscale",
    "bounded_rotvec",
    "check_cotangents",
    "check_dparams_out",
    "check_operands",
    "foh_coeffs",
    "foh_taps",
    "pack_params",
    "scanprep_bwd_ref",
    "scanprep_ref",
]

PARAM_COLS = 4
"""Projection columns one head spends on the transition.

``(w_x, w_y, w_z, ls)``: three for the rotation vector, one for the log-scale. The
taps are not among them; they are that transition's own forcing moments, computed
by :func:`foh_taps`. Not a shape multiple, so it does not live in
:mod:`slinoss.config`; it is this operator's own column count.
"""

ROTVEC_COLUMNS = slice(0, 3)
"""Columns of one head's parameter row holding the unconstrained rotation vector."""

LS_COLUMN = 3
"""Column of one head's parameter row holding the unconstrained log-scale."""

LS_MAX_MAG = 0.25
"""Bound on ``|ls|``: the shortest amplitude lifetime the parameterization admits.

The per-token amplitude factor is ``exp(2*ls)``, so a lifetime of ``h`` tokens is
``ls = -0.5/h`` and this bound is a lifetime of ``0.5/LS_MAX_MAG``, two tokens. It
is the decay's half of the rotation chart's sampled-timescale limit: a lifetime
under two tokens is a decay a sampled sequence cannot resolve, exactly as a period
under two tokens is a rotation it cannot resolve. The upper end is unchanged, so
a token can still ask for no decay at all.

Two-sided, so no token annihilates a row. A one-sided map lets one outlier token
multiply the whole carried state by zero, which deletes the gradient of everything
before it; clearing 99 percent of a row instead takes ``ln(100)/(2*LS_MAX_MAG)``,
nine tokens, and a delimiter has that many.
"""

DRIVE_FLOOR_SQ = 1.0e-12
"""Legacy drive-radius floor, retained for source compatibility but no longer used."""

DRIVE_CEIL_SQ = float(torch.finfo(torch.float32).max)
"""Legacy drive-radius ceiling, retained for source compatibility but no longer used."""

# ---------------------------------------------------------------------------
# First-order-hold taps
#
# The step's homogeneous generator is L = 2*ls*I + [w]_x, so the step is
# exp(L) and the forcing of an input held linearly between its two token values
# is the pair of moments
#
#   K_prev = int_0^1 r exp(L r) dr,   K_curr = int_0^1 (1 - r) exp(L r) dr,
#
# which are the entire functions phi_k(x) = sum_n x^n / (n + k)! at L:
#
#   K_prev = phi_1(L) - phi_2(L),     K_curr = phi_2(L).
#
# L is a scalar plus a skew part, hence normal, with eigenvalue p = 2*ls along w
# and z = p + i*|w| on the plane across it. A function of L is therefore its
# values at those eigenvalues, and the tap chart names exactly those: k_par =
# f(p), k_re = Re f(z), k_im = Im f(z). Nothing is fitted and nothing is
# approximated; phi_1 and phi_2 are evaluated where the operator needs them.
#
# The two phi are computed by the recurrence phi_{k+1} = (phi_k - 1/k!)/x from
# phi_0 = exp(x), which is one complex division each. That division cancels for
# small |x|, so under the radius below the series is summed instead. The device
# path sums the same series at a shorter truncation, so it takes its
# coefficients from here rather than deriving them again.
# ---------------------------------------------------------------------------

FOH_TAYLOR_RADIUS_SQ = 1.0
"""``|z|^2`` below which the tap series is summed rather than divided.

The recurrence forms ``phi_{k+1}`` from a difference of order ``|x|`` and divides
it by ``x``, so it loses relative accuracy like ``eps/|x|``; the series loses
``|x|^terms/(terms+k)!``. At unit radius the truncations below put the series
error under both float32 and float64 rounding, and outside it the recurrence has
already recovered them, so one radius covers both dtypes.
"""

FP64_FOH_TERMS = 20
"""Series terms for the reference. ``1/21! = 2e-20`` relative at unit radius."""

FP32_FOH_TERMS = 12
"""Series terms for the device path. ``1/13! = 2e-10`` relative at unit radius."""

T2_FLOOR = 1.0e-30
"""Floor on ``|w|^2`` in the chart's two divisions.

``g`` and ``h`` are the axial and skew coordinates of the tap against the
unnormalized ``w``, so they carry ``1/|w|^2`` and ``1/|w|`` and are ill
conditioned as ``|w| -> 0`` while the operator they encode is not: ``g`` reaches
it multiplied by ``w w^T`` and ``h`` by ``[w]_x``. The floor is normal in float32,
so both quotients stay finite, and at ``|w| = 0`` both numerators are exactly
zero, which is the operator's own limit there. The chart entries themselves hold
absolute rather than relative accuracy in that corner.
"""


def bounded_rotvec(raw: Tensor, w_max: float) -> Tensor:
    """Map an unconstrained vector into the ball of radius ``2*w_max``.

    ``w = w_max * raw / sqrt(1 + |raw|^2/4)``. Analytic everywhere and monotone
    in ``|raw|``. The factor of four leaves the derivative at the origin equal to
    ``w_max`` while putting ``|w| = w_max`` at finite raw radius. With the default
    ``w_max`` this makes every canonical SO(3) rotation, including a half turn,
    an interior point of the chart.

    Args:
        raw: Unconstrained vectors, shape ``(...,3)``.
        w_max: Half the asymptotic radius. Its float32 value must lie below pi.

    Returns:
        Rotation vectors with ``|w| <= 2*w_max``, shape ``(...,3)``.

    Raises:
        ValueError: If ``w_max`` is outside the float32-safe interval.
    """
    if not 0.0 < w_max <= ROTATION_CHART_SCALE_MAX:
        raise ValueError(
            f"w_max must lie in (0, pi) and round below pi in float32, got {w_max}"
        )
    return raw * (w_max * torch.rsqrt(1.0 + 0.25 * (raw * raw).sum(-1, keepdim=True)))


def bounded_logscale(raw: Tensor) -> Tensor:
    """Map an unconstrained scalar to a log-scale in ``[-LS_MAX_MAG, 0]``.

    ``ls = -LS_MAX_MAG * sigmoid(raw)``, so ``ls <= 0`` for every finite input and
    the decay per step is in ``(0,1]``. Bounded below as well as above, which is
    what :data:`LS_MAX_MAG` is for.

    Args:
        raw: Unconstrained scalars, any shape.

    Returns:
        Log-scales, same shape.
    """
    return -LS_MAX_MAG * torch.sigmoid(raw)


def anchored_rotvec(band: Tensor, bias: Tensor) -> Tensor:
    """The rotation-vector row a token presents to :func:`bounded_rotvec`.

    ``raw = bias + band``. The bias is the initialized operating point and the band
    is an unconstrained token displacement. Bounding the displacement here would
    duplicate :func:`bounded_rotvec`'s job, restrict the rotations a token can
    reach, and make a zero-initialized projection's gradient proportional to the
    head's initialized frequency. That last coupling suppresses the slowest head's
    token gradient by thousands relative to the fastest one.

    Args:
        band: Unconstrained token rows, ``(...,3)``.
        bias: Per-head rows, broadcasting against ``band``, ``(...,3)``.

    Returns:
        Unconstrained rotation vectors, ``(...,3)``.
    """
    return bias + band


def foh_coeffs(order: int, terms: int) -> tuple[float, ...]:
    """Coefficients of ``phi_order`` in ascending powers of its argument.

    Term ``n`` is ``1/(n + order)!``.

    Args:
        order: ``1`` or ``2``.
        terms: How many terms to return.

    Returns:
        Coefficients in ascending powers.
    """
    return tuple(1.0 / math.factorial(n + order) for n in range(terms))


def _horner_c(z: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    out = torch.full_like(z, coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * z + coeff
    return out


def _phi_pair(z: Tensor, terms: int) -> tuple[Tensor, Tensor]:
    """``phi_1`` and ``phi_2`` of a complex argument.

    Args:
        z: Complex arguments, any shape.
        terms: Series terms under :data:`FOH_TAYLOR_RADIUS_SQ`.

    Returns:
        ``(phi_1(z), phi_2(z))``, both complex, the shape of ``z``.
    """
    small = z.real.square() + z.imag.square() < FOH_TAYLOR_RADIUS_SQ
    # The recurrence divides, so it runs at one where it would divide by zero.
    # Its value there is discarded; a NaN would reach the gradient through the
    # select, so it is kept out of the primal rather than masked afterwards.
    safe = torch.where(small, torch.ones_like(z), z)
    phi1 = (torch.exp(z) - 1.0) / safe
    phi2 = (phi1 - 1.0) / safe
    return (
        torch.where(small, _horner_c(z, foh_coeffs(1, terms)), phi1),
        torch.where(small, _horner_c(z, foh_coeffs(2, terms)), phi2),
    )


def foh_taps(w: Tensor, ls: Tensor, terms: int = FP64_FOH_TERMS) -> Tensor:
    """First-order-hold taps of the step ``exp(2*ls*I + [w]_x)``, on the tap chart.

    Exact: the two moments of the module header, evaluated at the generator's own
    three eigenvalues and read off the chart
    :func:`slinoss.ops.so3ssd.tap_matrix` applies.

    Args:
        w: Rotation vectors, shape ``(...,3)``.
        ls: Log-scales, shape ``(...)``.
        terms: Series terms under :data:`FOH_TAYLOR_RADIUS_SQ`.

    Returns:
        ``(kr, g, h)`` per tap, shape ``(...,2,3)``, tap 0 previous and 1 current,
        in the dtype of ``w``.
    """
    t2 = (w * w).sum(-1).clamp_min(T2_FLOOR)
    par_arg = 2.0 * ls
    phi1, phi2 = _phi_pair(torch.complex(par_arg, t2.sqrt()), terms)
    axial1, axial2 = _phi_pair(torch.complex(par_arg, torch.zeros_like(par_arg)), terms)
    # Tap 0 is the previous token's coefficient, tap 1 the current token's.
    per = torch.stack([phi1 - phi2, phi2], dim=-1)
    par = torch.stack([axial1 - axial2, axial2], dim=-1).real
    kr = per.real
    return torch.stack(
        [kr, (par - kr) / t2[..., None], per.imag * torch.rsqrt(t2)[..., None]], dim=-1
    )


def pack_params(w_raw: Tensor, ls_raw: Tensor) -> Tensor:
    """Lay head-major raw parameters out in the projection's column order.

    The mixer's projection emits this layout directly. A caller holding the two
    head-major tensors separately -- a test fixture or a benchmark -- packs them
    here rather than restating the column order.

    Args:
        w_raw: Unconstrained rotation vectors, ``(B,H,T,3)``.
        ls_raw: Unconstrained log-scales, ``(B,H,T)``.

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
    bsz, heads, seqlen = lead
    row = torch.cat(
        [w_raw.permute(0, 2, 1, 3), ls_raw.permute(0, 2, 1)[..., None]], dim=-1
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
        dparams: ``(B,T,H*PARAM_COLS)``, dtype of ``params``. Contiguous when the
            backward allocated it, and the caller's own tensor -- a pitched band of a
            wider gradient buffer -- when the caller supplied one.
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
    """Apply the bounded maps, derive the taps, and pack the result.

    Args:
        params: Projection slice, ``(B,T,H*PARAM_COLS)``, activation dtype.
            Trailing stride one; the row stride is the projection width. Per head,
            in order ``(w_x, w_y, w_z, ls)``.
        param_bias: ``(H,PARAM_COLS)``, float32. The head's operating point, added
            to the token row per :func:`anchored_rotvec`.
        heads: ``H``.
        w_max: Rotation-vector chart scale, in ``(0, pi)``.

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
        rows = rows.permute(0, 2, 1, 3)
        bias = param_bias.to(dtype)[:, None, :]
        w = bounded_rotvec(
            anchored_rotvec(rows[..., ROTVEC_COLUMNS], bias[..., ROTVEC_COLUMNS]),
            w_max,
        )
        ls = bounded_logscale(rows[..., LS_COLUMN] + bias[..., LS_COLUMN])
        tap = foh_taps(w, ls)
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
    dparams: Tensor | None = None,
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
        param_bias: The forward's bias, ``(H,PARAM_COLS)``. The maps' Jacobians are
            evaluated at the row the bias and the band form together, so the bias is
            saved too.
        heads: ``H``.
        w_max: The forward's chart scale.
        dparams: Destination for the parameter gradient, or ``None`` to allocate
            one. See :func:`check_dparams_out` for the contract.

    Returns:
        A :class:`ScanGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, a ``w_max`` outside ``(0, pi)``, or
            a destination off the pitched-layout contract.
        TypeError: On an unsupported dtype, or a destination whose dtype is not that
            of ``params``.
    """
    check_cotangents(dtrans, dK, params, param_bias, heads)
    if dparams is not None:
        check_dparams_out(dparams, params, heads)
    pl = params.detach().requires_grad_(True)
    cl = param_bias.detach().requires_grad_(True)
    with torch.enable_grad():
        out = scanprep_ref(pl, cl, heads=heads, w_max=w_max)
    grad, dparam_bias = torch.autograd.grad((out.trans, out.K), (pl, cl), (dtrans, dK))
    if dparams is None:
        dparams = grad.contiguous()
    else:
        # Copy, not accumulate: the destination is a band of a buffer whose other
        # columns belong to other operators, and no phase zeroed this band.
        dparams.copy_(grad)
    return ScanGrads(dparams=dparams, dparam_bias=dparam_bias)


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


def check_dparams_out(dparams: Tensor, params: Tensor, heads: int) -> None:
    """Validate a caller-supplied destination for ``dparams``.

    The mixer's backward allocates one ``(B,T,W)`` gradient buffer for the whole
    fused projection and hands every operator the band it owns, so the destination is
    pitched rather than contiguous: :func:`slinoss._guard.check_pitched` is the rule
    it is held to, and the parameter band is a strict one, so the pitch owes the
    sector rather than the vector width. Shared by both backends, for the same reason
    as :func:`check_operands`.

    The destination is written in full and never accumulated into, so nothing zeroes
    it first.

    Args:
        dparams: The destination. Must carry the shape, dtype, and device the
            allocated gradient would have had.
        params: The forward's projection slice.
        heads: ``H``.

    Raises:
        ValueError: On a shape or device mismatch, or a layout off the pitched
            contract.
        TypeError: On a dtype other than that of ``params``.
    """
    # Shape and dtype before layout: a misshaped destination reports its shape rather
    # than an alignment its offset also violates.
    want = (int(params.shape[0]), int(params.shape[1]), heads * PARAM_COLS)
    if tuple(dparams.shape) != want:
        raise ValueError(f"dparams must be {want}, got {tuple(dparams.shape)}")
    if dparams.dtype is not params.dtype:
        raise TypeError(
            f"dparams must be {params.dtype} like params, got {dparams.dtype}"
        )
    if dparams.device != params.device:
        raise ValueError(
            f"dparams must be on {params.device} like params, got {dparams.device}"
        )
    check_pitched(((dparams, "dparams"),))
