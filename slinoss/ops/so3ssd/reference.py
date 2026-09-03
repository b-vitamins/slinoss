"""Pure-PyTorch reference for the SO(3) scan. The mathematical authority.

Two implementations of the same operator:

- :func:`so3ssm` steps the recurrence token by token. It defines correctness.
- :func:`so3ssd_ref` evaluates the chunked factorization. It defines the shape
  the CUDA kernels implement.

Both are float64-capable and agree to float64 rounding. A kernel that disagrees
with these is wrong until proven otherwise in float64.

Recurrence, per token ``t``, with ``w_t = trans[...,t,:3]`` and
``ls_t = trans[...,t,3]``:

    q_t = quat_exp(w_t)
    z_t = exp(2*ls_t) * R(q_t) z_{t-1}
        + outer(u_{t-1}, Kprev_t(b_{t-1}))
        + outer(u_t,     Kcurr_t(b_t))
    y_t = <c_t, z_t>

``R(q)`` acts identically on each of the ``N`` 3-vectors of a state row and on
each of the ``P`` rows, so the transition is four numbers per token.

Chunk factorization, per chunk of length ``L``, with prefixes
``Q_t = q_t (*) ... (*) q_0`` and ``lp_t = sum_{j<=t} ls_j``. Reindexing the
previous-tap forcing onto its own token gives both taps the same decay and the
same causal mask:

    bn_r = R(Q_r)^T Kcurr_r b_r        weighted by u_r
    bp_r = R(Q_r)^T Kprev_r b_{r-1}    weighted by u_{r-1}
    crot_t = R(Q_t)^T c_t
    dmask(t,r) = exp(2*(lp_t - lp_r)) * [r <= t]

    y_diag = (<crot,bn> * dmask) @ u + (<crot,bp> * dmask) @ ushift
    y_off_t = exp(2*lp_t) * <crot_t, zstart>
    inc = R(Q_{L-1}) [ (u*wgt)^T bn + (ushift*wgt)^T bp ],
          wgt_r = exp(2*(lp_{L-1} - lp_r)) <= 1

Four dense real GEMMs and a separable scalar mask. No complex arithmetic and no
interleaved storage.

:func:`chunked_forward_fused` collapses the two taps to one. Reindexing the
now-tap of token ``s-1`` onto slot ``s`` factors the common decay out, so a single
table column ``Afuse_s = ap_s + exp(2*ls_s) * an_{s-1}`` carries both taps against
one operand ``b_{s-1}``, and what the reindex leaves over is two rank-one residues
rather than a GEMM. Same operator, four GEMMs per chunk instead of seven.

Value invariants -- ``ls <= 0`` and ``|w| < 2*pi`` -- are the parameterization's
job, not this module's. Nothing here scans a tensor to validate a numerical
range and nothing here clamps.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import pad as _pad

from slinoss._guard import Named, check_pitched
from slinoss._precision import (
    autocast_disabled,
    check_pinned,
    check_supported,
    pinned_dtype,
)
from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE

__all__ = [
    "ChunkedForward",
    "FusedForward",
    "SO3SSDResult",
    "ScanPrologue",
    "TapGrads",
    "TransformTable",
    "as_lanes",
    "check_grad_band",
    "chunk_pad",
    "chunked_forward",
    "chunked_forward_fused",
    "deriv_coeffs",
    "from_heads",
    "pad_time",
    "quat_conj",
    "quat_exp",
    "quat_exp_vjp",
    "quat_mul",
    "quat_prefix_scan",
    "quat_prefix_scan_vjp",
    "rot_matrix",
    "rot_matrix_vjp",
    "series_coeffs",
    "skew",
    "so3ssd_ref",
    "so3ssm",
    "tap_matrix",
    "tap_matrix_vjp",
    "to_heads",
    "transform_table",
]

# ---------------------------------------------------------------------------
# Quaternion exponential
#
# The scalar part of quat_exp is even in |w| and the vector part is an odd
# multiple of w, so both halves are entire functions of s = |w|^2:
#
#   cos(sqrt(s)/2)             = sum_k (-1)^k (s/4)^k / (2k)!
#   sin(sqrt(s)/2)/(sqrt(s)/2) = sum_k (-1)^k (s/4)^k / (2k+1)!
#
# Evaluating in s rather than |w| removes the sqrt, whose derivative is
# undefined at w = 0. The reachable domain is the closed ball of radius 2*pi,
# where s/4 <= pi^2 and the 14-term truncation remains exact to float64 rounding.
# Sizing for the closed ball rather than for one configured scale also absorbs the
# last-ulp rounding of the parameter map.
# tests/test_quat.py measures that against the transcendental form.
# ---------------------------------------------------------------------------

_SERIES_TERMS = 14


def series_coeffs(offset: int, terms: int = _SERIES_TERMS) -> tuple[float, ...]:
    """Coefficients of one half-angle series in ``s = |w|^2``.

    Term ``k`` is ``(-1)^k / (4^k (2k + offset)!)``. ``offset = 0`` gives
    ``cos(|w|/2)``, ``offset = 1`` gives ``sinc(|w|/2)``.

    The device path evaluates the same series at a shorter truncation, so it
    takes its coefficients from here rather than deriving them again.

    Args:
        offset: ``0`` for the scalar part, ``1`` for the vector part.
        terms: How many terms to return. Defaults to the float64 truncation.

    Returns:
        Coefficients in ascending powers of ``s``.
    """
    return tuple(
        (-1.0) ** k / (4.0**k * math.factorial(2 * k + offset)) for k in range(terms)
    )


def deriv_coeffs(coeffs: tuple[float, ...]) -> tuple[float, ...]:
    """Coefficients of ``d/ds`` of a series in ``s``.

    Term ``k`` becomes ``k`` times term ``k``, and the constant term drops, so the
    result is one shorter than its input.

    The device path differentiates the same series at a shorter truncation, so it
    takes these coefficients from here rather than deriving them again.

    Args:
        coeffs: Coefficients in ascending powers of ``s``.

    Returns:
        Coefficients of the derivative, in ascending powers of ``s``.
    """
    return tuple(k * coeffs[k] for k in range(1, len(coeffs)))


_COS_HALF: tuple[float, ...] = series_coeffs(0)
_SINC_HALF: tuple[float, ...] = series_coeffs(1)
_COS_HALF_D: tuple[float, ...] = deriv_coeffs(_COS_HALF)
_SINC_HALF_D: tuple[float, ...] = deriv_coeffs(_SINC_HALF)


def _horner(s: Tensor, coeffs: tuple[float, ...]) -> Tensor:
    out = torch.full_like(s, coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * s + coeff
    return out


def quat_exp(w: Tensor) -> Tensor:
    """Unit quaternion of a rotation vector.

    Args:
        w: Rotation vectors, shape ``(...,3)``. Axis times angle.

    Returns:
        Scalar-first unit quaternions ``(w,x,y,z)``, shape ``(...,4)``.
    """
    s = (w * w).sum(-1, keepdim=True)
    return torch.cat([_horner(s, _COS_HALF), 0.5 * _horner(s, _SINC_HALF) * w], dim=-1)


def quat_exp_vjp(dq: Tensor, w: Tensor) -> Tensor:
    """Adjoint of :func:`quat_exp`.

    Differentiating in ``s = |w|^2`` keeps the chain rule polynomial too: the
    inner derivative is ``2*w``, so nothing here divides by ``|w|``.

    Args:
        dq: Cotangent of the quaternion, shape ``(...,4)``.
        w: Rotation vectors the primal was evaluated at, shape ``(...,3)``.

    Returns:
        Cotangent of ``w``, shape ``(...,3)``.
    """
    s = (w * w).sum(-1, keepdim=True)
    dq_s, dq_v = dq[..., :1], dq[..., 1:]
    ds = dq_s * _horner(s, _COS_HALF_D) + 0.5 * _horner(s, _SINC_HALF_D) * (
        dq_v * w
    ).sum(-1, keepdim=True)
    return 0.5 * _horner(s, _SINC_HALF) * dq_v + 2.0 * w * ds


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product ``a (*) b``, so ``R(a (*) b) == R(a) R(b)``.

    Args:
        a: Scalar-first quaternions, shape ``(...,4)``.
        b: Scalar-first quaternions, shape ``(...,4)``.

    Returns:
        The product, shape ``(...,4)``.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: Tensor) -> Tensor:
    """Conjugate, which inverts the rotation of a unit quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_prefix_scan(q: Tensor) -> Tensor:
    """Inclusive non-commutative prefix product along the second-to-last axis.

    Hillis-Steele over ``ceil(log2(L))`` rounds, which is the shape of the warp
    scan the kernels use. Renormalized once at the end: rotation error enters the
    rotation matrix squared, so unit-norm drift is not tolerated.

    Args:
        q: Unit quaternions, shape ``(...,L,4)``.

    Returns:
        ``Q`` with ``Q_t = q_t (*) ... (*) q_0``, shape ``(...,L,4)``.
    """
    out = q
    length = q.shape[-2]
    step = 1
    while step < length:
        out = torch.cat(
            [out[..., :step, :], quat_mul(out[..., step:, :], out[..., :-step, :])],
            dim=-2,
        )
        step *= 2
    return out / out.norm(dim=-1, keepdim=True)


def quat_prefix_scan_vjp(dQ: Tensor, qprefix: Tensor) -> Tensor:
    """Adjoint of :func:`quat_prefix_scan`, in closed form.

    ``Q_l = q_l (*) Q_{l-1}`` and right multiplication is its own adjoint under
    conjugation, so

        dq_l = Q_l (*) S_l (*) conj(Q_{l-1}),
        S_l  = sum_{m >= l} conj(Q_m) (*) p_m.

    ``S`` is a reverse cumulative sum of four numbers per token, not a
    non-commutative scan, so the adjoint costs one suffix sum and two products.
    ``p_m`` is ``dQ_m`` with its radial component removed, which is the adjoint of
    the renormalization the forward applies once per chunk; the prefix is unit to
    rounding, so the divide by its norm is the identity.

    Args:
        dQ: Cotangent of the prefix, shape ``(...,L,4)``.
        qprefix: The prefix the primal returned, shape ``(...,L,4)``. The
            per-token quaternions are not needed: the closed form recovers each
            from a neighbouring pair of prefixes.

    Returns:
        Cotangent of the per-token quaternions, shape ``(...,L,4)``.
    """
    proj = dQ - (dQ * qprefix).sum(-1, keepdim=True) * qprefix
    suffix = quat_mul(quat_conj(qprefix), proj).flip(-2).cumsum(-2).flip(-2)
    ident = _pad(torch.ones_like(qprefix[..., :1, :1]), (0, 3))
    shifted = torch.cat([ident, qprefix[..., :-1, :]], dim=-2)
    return quat_mul(quat_mul(qprefix, suffix), quat_conj(shifted))


# ---------------------------------------------------------------------------
# 3x3 transforms
# ---------------------------------------------------------------------------


def _as_matrix(flat: Tensor) -> Tensor:
    return flat.unflatten(-1, (3, 3))


def skew(w: Tensor) -> Tensor:
    """Cross-product matrix, so ``skew(w) @ v == cross(w, v)``.

    Args:
        w: Vectors, shape ``(...,3)``.

    Returns:
        Antisymmetric matrices, shape ``(...,3,3)``.
    """
    wx, wy, wz = w.unbind(-1)
    zero = torch.zeros_like(wx)
    return _as_matrix(
        torch.stack([zero, -wz, wy, wz, zero, -wx, -wy, wx, zero], dim=-1)
    )


def rot_matrix(q: Tensor) -> Tensor:
    """Rotation matrix of a unit quaternion, so ``R(q) v == q v conj(q)``.

    Args:
        q: Scalar-first unit quaternions, shape ``(...,4)``.

    Returns:
        Rotation matrices, shape ``(...,3,3)``.
    """
    qw, qx, qy, qz = q.unbind(-1)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return _as_matrix(
        torch.stack(
            [
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
            ],
            dim=-1,
        )
    )


def _sym_asym(dM: Tensor) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor, Tensor]:
    """Off-diagonal symmetric parts, the axial vector, and the diagonal.

    Every 3x3 adjoint here contracts ``dM`` against a matrix that is either
    symmetric or antisymmetric, so both halves are formed once.

    Args:
        dM: Cotangent matrices, shape ``(...,3,3)``.

    Returns:
        ``((s01, s02, s12), axial, diag)`` where ``s_ij = dM_ij + dM_ji``,
        ``axial`` is the vector ``v`` with ``<dM, skew(v)> = <axial, v>``, shape
        ``(...,3)``, and ``diag`` holds the three diagonal entries, shape
        ``(...,3)``.
    """
    d00, d01, d02, d10, d11, d12, d20, d21, d22 = dM.flatten(-2, -1).unbind(-1)
    axial = torch.stack([d21 - d12, d02 - d20, d10 - d01], dim=-1)
    diag = torch.stack([d00, d11, d22], dim=-1)
    return (d01 + d10, d02 + d20, d12 + d21), axial, diag


def rot_matrix_vjp(dR: Tensor, q: Tensor) -> Tensor:
    """Adjoint of :func:`rot_matrix`.

    Args:
        dR: Cotangent of the rotation matrix, shape ``(...,3,3)``.
        q: Scalar-first unit quaternions the primal was evaluated at, shape
            ``(...,4)``.

    Returns:
        Cotangent of ``q``, shape ``(...,4)``. Includes the radial component;
        callers that renormalized must project it out.
    """
    qw, qx, qy, qz = q.unbind(-1)
    (s01, s02, s12), axial, diag = _sym_asym(dR)
    a0, a1, a2 = axial.unbind(-1)
    d00, d11, d22 = diag.unbind(-1)
    return torch.stack(
        [
            2.0 * (qx * a0 + qy * a1 + qz * a2),
            2.0 * (qy * s01 + qz * s02 + qw * a0) - 4.0 * qx * (d11 + d22),
            2.0 * (qx * s01 + qz * s12 + qw * a1) - 4.0 * qy * (d00 + d22),
            2.0 * (qx * s02 + qy * s12 + qw * a2) - 4.0 * qz * (d00 + d11),
        ],
        dim=-1,
    )


def tap_matrix(tap: Tensor, w: Tensor) -> Tensor:
    """Tap operator as an explicit matrix.

    ``K(v) = kr*v + g*(w.v)*w + h*(w x v)``, i.e.
    ``K = kr*I + g*w w^T + h*skew(w)``. Polynomial in ``w`` and therefore
    analytic at ``w = 0``; the axis-angle normal form is not, and needs an rsqrt,
    a clamp, and a whole-tensor validity pass that this chart makes structural.
    The charts are related by ``k_re = kr``, ``k_par = kr + g*|w|^2``,
    ``k_im = h*|w|``.

    Args:
        tap: ``(kr, g, h)``, shape ``(...,3)``.
        w: Rotation vectors, shape ``(...,3)``.

    Returns:
        Tap matrices, shape ``(...,3,3)``.
    """
    kr, par, imag = (component[..., None, None] for component in tap.unbind(-1))
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return kr * eye + par * (w[..., :, None] * w[..., None, :]) + imag * skew(w)


class TapGrads(NamedTuple):
    """Adjoints of :func:`tap_matrix`.

    Attributes:
        tap: Cotangent of ``(kr, g, h)``, shape ``(...,3)``.
        w: Cotangent of the rotation vector, shape ``(...,3)``.
    """

    tap: Tensor
    w: Tensor


def tap_matrix_vjp(dK: Tensor, tap: Tensor, w: Tensor) -> TapGrads:
    """Adjoint of :func:`tap_matrix`.

    The three basis matrices are the identity, ``w w^T``, and ``skew(w)``, so each
    tap cotangent is one Frobenius inner product: the trace, the quadratic form,
    and the axial contraction.

    Args:
        dK: Cotangent of the tap matrix, shape ``(...,3,3)``.
        tap: ``(kr, g, h)`` the primal was evaluated at, shape ``(...,3)``.
        w: Rotation vectors the primal was evaluated at, shape ``(...,3)``.

    Returns:
        A :class:`TapGrads`.
    """
    (s01, s02, s12), axial, diag = _sym_asym(dK)
    wx, wy, wz = w.unbind(-1)
    _, par, imag = tap.unbind(-1)
    d00, d11, d22 = diag.unbind(-1)
    symw = torch.stack(
        [
            2.0 * d00 * wx + s01 * wy + s02 * wz,
            s01 * wx + 2.0 * d11 * wy + s12 * wz,
            s02 * wx + s12 * wy + 2.0 * d22 * wz,
        ],
        dim=-1,
    )
    dtap = torch.stack(
        [
            diag.sum(-1),
            0.5 * (w * symw).sum(-1),
            (w * axial).sum(-1),
        ],
        dim=-1,
    )
    return TapGrads(tap=dtap, w=par[..., None] * symw + imag[..., None] * axial)


class TransformTable(NamedTuple):
    """Per-token 3x3 matrices in the chunk-local frame.

    One quaternion exponential, one prefix product, and one 3x3 composition per
    token replace a per-lane transform of a ``3N``-sized vector. Every vector
    transform downstream is a 9-FMA matvec against a broadcast operand.

    Attributes:
        ac: ``R(Q_t)^T``, applied to ``c_t``. Shape ``(...,3,3)``.
        ap: ``R(Q_t)^T Kprev_t``, applied to ``b_{t-1}``. Shape ``(...,3,3)``.
        an: ``R(Q_t)^T Kcurr_t``, applied to ``b_t``. Shape ``(...,3,3)``.
        rot: ``R(Q_t)``, the inverse change of basis. Shape ``(...,3,3)``.
    """

    ac: Tensor
    ap: Tensor
    an: Tensor
    rot: Tensor


def transform_table(w: Tensor, tap: Tensor, qprefix: Tensor) -> TransformTable:
    """Compose the per-token table of :class:`TransformTable`.

    Args:
        w: Rotation vectors, shape ``(...,3)``.
        tap: Tap parameters, shape ``(...,2,3)``, index 0 previous and 1 current.
        qprefix: Inclusive quaternion prefix ``Q_t``, shape ``(...,4)``.

    Returns:
        The table.
    """
    rot = rot_matrix(qprefix)
    ac = rot.transpose(-1, -2)
    return TransformTable(
        ac=ac,
        ap=ac @ tap_matrix(tap[..., 0, :], w),
        an=ac @ tap_matrix(tap[..., 1, :], w),
        rot=rot,
    )


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class ScanPrologue(NamedTuple):
    """The chunk-boundary quantities the backward reads rather than recomputes.

    Every one is an output of the forward's first two stages. A backend that
    returns it lets the autograd boundary hold it and the backward skip those two
    stages; a backend that returns ``None`` in its place leaves the backward to
    rematerialize them. Nothing in either direction writes these buffers after
    the forward's inter-chunk recurrence has left them, so holding them cannot
    perturb the forward's own outputs.

    Attributes:
        zstart: Chunk-start state, shape ``(B,H,C,P,3N)``, contiguous, over ``C``
            chunks. The dtype is the producing backend's: float32 where the
            recurrence stores it wide, the activation dtype where it narrows on the
            store.
        cquat: Unit chunk rotation, shape ``(B,H,C,4)``, float32, contiguous.
        cscale: Chunk decay ``exp(2*lp_{L-1})``, shape ``(B,H,C)``, float32,
            contiguous.
    """

    zstart: Tensor
    cquat: Tensor
    cscale: Tensor


class SO3SSDResult(NamedTuple):
    """Return type of both reference implementations.

    Every tensor field is contiguous. A ragged tail leaves the chunked path
    holding a time slice of a padded buffer, and that must not reach a caller.

    Attributes:
        y: Output, shape ``(B,H,T,P)``, dtype of ``U``.
        state: State after the last token, shape ``(B,H,P,3N)``, pinned dtype.
        b_last: ``b`` at the last token, shape ``(B,G,3N)``, contiguous. Feeds
            ``b_prev`` of the next call in a streaming split.
        u_last: ``u`` at the last token, shape ``(B,H,P)``, contiguous. Feeds
            ``u_prev``.
        prologue: The chunk-boundary quantities of :class:`ScanPrologue`, for a
            backward that reads them, or ``None`` from a backend whose backward
            rematerializes them. It is not an output of the operator: the public
            callable returns ``None`` here, and only the autograd boundary reads
            it.

    ``b_last`` and ``u_last`` are contiguous because they are fed straight back
    in and the operator repacks nothing. A time slice of ``B`` is strided over
    groups, so it is copied here rather than at every call site.
    """

    y: Tensor
    state: Tensor
    b_last: Tensor
    u_last: Tensor
    prologue: ScanPrologue | None = None


class _Shapes(NamedTuple):
    bsz: int
    heads: int
    groups: int
    seqlen: int
    rows: int
    state_dim: int
    lanes: int


def to_heads(t: Tensor, heads: int) -> Tensor:
    """Broadcast a grouped tensor onto heads: ``(B,G,...) -> (B,H,...)``.

    Head ``h`` reads group ``h // (H // G)``, so the group axis expands to
    ``(G, H // G)`` and flattens in that order. Identity when ``G == H``.

    Autograd through this is the group reduction: the pullback sums the cotangent
    over the ``H // G`` heads of each group. That is why the reference broadcasts
    rather than indexing per head, and why the reference gradient of a grouped
    ``B`` or ``C`` needs no hand-written cross-head sum.

    Args:
        t: Grouped tensor, shape ``(B,G,...)``.
        heads: ``H``. Must be a multiple of ``G``.

    Returns:
        Shape ``(B,H,...)``.

    Raises:
        ValueError: If ``G`` does not divide ``heads``.
    """
    groups = int(t.shape[1])
    if heads % groups != 0:
        raise ValueError(f"G={groups} does not divide H={heads}")
    if groups == heads:
        return t
    per = heads // groups
    return (
        t.unsqueeze(2).expand(int(t.shape[0]), groups, per, *t.shape[2:]).flatten(1, 2)
    )


def from_heads(t: Tensor, groups: int) -> Tensor:
    """Adjoint of :func:`to_heads`: ``(B,H,...) -> (B,G,...)`` by summation.

    A grouped input is read by ``H // G`` heads, so its cotangent is the sum of
    what each of those heads contributed. Identity when ``G == H``.

    Args:
        t: Per-head cotangent, shape ``(B,H,...)``.
        groups: ``G``. Must divide ``H``.

    Returns:
        Shape ``(B,G,...)``.

    Raises:
        ValueError: If ``groups`` does not divide ``H``.
    """
    heads = int(t.shape[1])
    if groups < 1 or heads % groups != 0:
        raise ValueError(f"G={groups} does not divide H={heads}")
    if groups == heads:
        return t
    return t.unflatten(1, (groups, heads // groups)).sum(2)


def as_lanes(t: Tensor) -> Tensor:
    """Split the trailing ``3N`` into ``N`` 3-vectors: ``(...,3N) -> (...,N,3)``.

    Lane-major: element ``3n+i`` is component ``i`` of 3-vector ``n``.

    Args:
        t: Tensor whose last axis is ``3N``.

    Returns:
        A view with shape ``(...,N,3)``.
    """
    return t.unflatten(-1, (-1, 3))


def pad_time(t: Tensor, length: int) -> Tensor:
    """Zero-pad the token axis up to a multiple of ``L``, leaving it flat.

    Args:
        t: Time-major tensor, shape ``(B,H,T,...)``.
        length: Chunk length ``L``.

    Returns:
        Shape ``(B,H,ceil(T/L)*L,...)``. ``t`` itself when ``L`` divides ``T``.
    """
    tail = (-t.shape[2]) % length
    if tail:
        t = _pad(t, (0, 0) * (t.ndim - 3) + (0, tail))
    return t


def chunk_pad(t: Tensor, length: int) -> Tensor:
    """``(B,H,T,...) -> (B,H,ceil(T/L),L,...)``, zero-padding a ragged tail.

    Under the two-tap factorization of :func:`chunked_forward` a zero-padded token
    is an exact no-op: ``w = 0`` and ``ls = 0`` give the identity transition while
    a zero tap kills the forcing. The one-tap factorization of
    :func:`chunked_forward_fused` reindexes each now-tap onto the following slot,
    so there a pad slot carries the last real token's forcing and the shifted
    operands are built by :func:`pad_time` before the shift rather than padded
    after it.

    Args:
        t: Time-major tensor, shape ``(B,H,T,...)``.
        length: Chunk length ``L``.

    Returns:
        The chunked tensor, shape ``(B,H,ceil(T/L),L,...)``.
    """
    return pad_time(t, length).unflatten(2, (-1, length))


def _check_inputs(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    z0: Tensor | None,
    b_prev: Tensor | None,
    u_prev: Tensor | None,
) -> _Shapes:
    if U.ndim != 4:
        raise ValueError(f"U must be (B,H,T,P), got shape {tuple(U.shape)}")
    if B.ndim != 4:
        raise ValueError(f"B must be (B,G,T,3N), got shape {tuple(B.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in U.shape)
    groups = int(B.shape[1])
    state_dim = int(B.shape[-1])
    if seqlen < 1:
        raise ValueError("T must be at least 1")
    if groups < 1 or heads % groups != 0:
        raise ValueError(
            f"B and C carry G groups with G dividing H; got G={groups}, H={heads}"
        )

    named: list[tuple[str, Tensor, tuple[int, ...]]] = [
        ("trans", trans, (bsz, heads, seqlen, 4)),
        ("K", K, (bsz, heads, seqlen, 2, 4)),
        ("B", B, (bsz, groups, seqlen, state_dim)),
        ("C", C, (bsz, groups, seqlen, state_dim)),
    ]
    if z0 is not None:
        named.append(("z", z0, (bsz, heads, rows, state_dim)))
    if b_prev is not None:
        named.append(("b_prev", b_prev, (bsz, groups, state_dim)))
    if u_prev is not None:
        named.append(("u_prev", u_prev, (bsz, heads, rows)))

    for name, tensor, shape in named:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )

    if state_dim % 3 != 0 or (state_dim // 3) % LANE_MULTIPLE != 0:
        raise ValueError(
            f"3N must be 3 times a multiple of {LANE_MULTIPLE}, got 3N={state_dim}"
        )
    if rows % HEAD_MULTIPLE != 0:
        raise ValueError(f"P must be a multiple of {HEAD_MULTIPLE}, got P={rows}")
    if (b_prev is None) != (u_prev is None):
        raise ValueError("b_prev and u_prev are passed together or not at all")

    # ``B`` and ``C`` arrive as column bands of the mixer's fused projection: their
    # token stride is the projection width, not ``3N``. Demanding contiguity of them
    # would demand a copy of that projection. Every other operand owns its buffer, so
    # the layout rule splits here while the device rule does not. Layout is checked
    # after the shapes above: a wrong-shaped operand reports its shape, not a pitch.
    bands: Named = ((B, "B"), (C, "C"))
    banded = {name for _, name in bands}
    for name, tensor in [("U", U), *((n, t) for n, t, _ in named)]:
        if name not in banded and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; no repacking is done")
        if tensor.device != U.device:
            raise ValueError(
                f"{name} is on {tensor.device}, U is on {U.device}; one device only"
            )
    # A contiguous band meets the pitched rule at a pitch equal to its row width, so
    # only a strided one is handed to it. That rule's alignment clause is a device
    # rule, and this reference is the CPU oracle as well.
    check_pitched(tuple(one for one in bands if not one[0].is_contiguous()))
    check_supported(U, "U")
    check_supported(B, "B")
    check_supported(C, "C")
    check_pinned(trans, "trans")
    check_pinned(K, "K")
    if z0 is not None:
        check_pinned(z0, "z")
    if b_prev is not None:
        check_supported(b_prev, "b_prev")
    if u_prev is not None:
        check_supported(u_prev, "u_prev")

    return _Shapes(bsz, heads, groups, seqlen, rows, state_dim, state_dim // 3)


def check_grad_band(t: Tensor, operand: Tensor, name: str) -> None:
    """Hold a caller-supplied gradient buffer to the operand whose gradient it holds.

    Three buffers cross the backward's boundary: the ``dB`` and ``dC`` destinations,
    and the ``dU_init`` accumulate seed. Each carries the shape, dtype, and device of
    one forward operand, and each may arrive as a column band of a wider tensor,
    because the mixer's backward allocates one buffer for its fused projection's
    gradient and hands every operator the band it owns. The layout rule is therefore
    :func:`slinoss._guard.check_pitched` rather than contiguity: a row pitch above the
    row width is legal, and contiguity is the case where the two agree.

    Order is shape, then dtype, then device, then layout, so a buffer of the wrong
    extent reports its extent rather than an alignment its offset also violates.

    Args:
        t: The caller's buffer.
        operand: The forward operand it belongs to: ``B`` for ``dB``, ``C`` for
            ``dC``, ``U`` for ``dU_init``.
        name: What to report ``t`` under.

    Raises:
        ValueError: On a shape or device mismatch, or on a band with a strided
            trailing axis, overlapping rows, or an offset or pitch off the boundary
            :func:`slinoss._guard.check_pitched` holds it to.
        TypeError: On a dtype other than ``operand``'s.
    """
    want = tuple(operand.shape)
    if tuple(t.shape) != want:
        raise ValueError(f"{name} must have shape {want}, got {tuple(t.shape)}")
    if t.dtype is not operand.dtype:
        raise TypeError(f"{name} must be {operand.dtype}, got {t.dtype}")
    if t.device != operand.device:
        raise ValueError(f"{name} must be on {operand.device}, got {t.device}")
    # The pitched rule's alignment clause is a device rule -- a band row that starts
    # mid-sector fetches a sector nobody reads -- and this path is the CPU oracle as
    # well, so a host buffer is held to its shape and dtype alone.
    if t.device.type == "cuda":
        check_pitched(((t, name),))


def _promote(t: Tensor, dtype: torch.dtype) -> Tensor:
    return t if t.dtype is dtype else t.to(dtype)


def so3ssm(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> SO3SSDResult:
    """Sequential reference. Defines correctness.

    Steps the recurrence one token at a time in the pinned dtype. Cost is linear
    in ``T`` with a Python-level loop, so this is a test oracle, not a path.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned. Tap index 0
            is previous and 1 is current; lane 3 is ignored.
        B: Input vectors, shape ``(B,G,T,3N)``. Grouped: head ``h`` reads group
            ``h // (H // G)``.
        C: Output vectors, shape ``(B,G,T,3N)``. Grouped like ``B``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.

    Returns:
        A :class:`SO3SSDResult`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    shapes = _check_inputs(U, trans, K, B, C, z0, b_prev, u_prev)
    dtype = pinned_dtype(U, trans, K, B, C)
    device = U.device

    with autocast_disabled(device.type):
        w = _promote(trans[..., :3], dtype)
        scale = torch.exp(2.0 * _promote(trans[..., 3], dtype))
        rot = rot_matrix(quat_exp(w))
        kprev = tap_matrix(_promote(K[..., 0, :3], dtype), w)
        kcurr = tap_matrix(_promote(K[..., 1, :3], dtype), w)
        u = _promote(U, dtype)
        # B and C are grouped, everything else is per head. Broadcast once here so
        # the recurrence below is written per head throughout.
        blane = as_lanes(to_heads(_promote(B, dtype), shapes.heads))
        clane = as_lanes(to_heads(_promote(C, dtype), shapes.heads))

        state = (
            torch.zeros(
                shapes.bsz,
                shapes.heads,
                shapes.rows,
                shapes.lanes,
                3,
                dtype=dtype,
                device=device,
            )
            if z0 is None
            else as_lanes(_promote(z0, dtype))
        )
        bp = (
            torch.zeros(
                shapes.bsz, shapes.heads, shapes.lanes, 3, dtype=dtype, device=device
            )
            if b_prev is None
            else as_lanes(to_heads(_promote(b_prev, dtype), shapes.heads))
        )
        up = (
            torch.zeros(
                shapes.bsz, shapes.heads, shapes.rows, dtype=dtype, device=device
            )
            if u_prev is None
            else _promote(u_prev, dtype)
        )

        outputs = []
        for t in range(shapes.seqlen):
            state = scale[:, :, t, None, None, None] * torch.einsum(
                "bhij,bhpnj->bhpni", rot[:, :, t], state
            )
            vprev = torch.einsum("bhij,bhnj->bhni", kprev[:, :, t], bp)
            vcurr = torch.einsum("bhij,bhnj->bhni", kcurr[:, :, t], blane[:, :, t])
            state = (
                state
                + up[..., None, None] * vprev[..., None, :, :]
                + u[:, :, t][..., None, None] * vcurr[..., None, :, :]
            )
            outputs.append(torch.einsum("bhni,bhpni->bhp", clane[:, :, t], state))
            bp = blane[:, :, t]
            up = u[:, :, t]

        y = torch.stack(outputs, dim=2)

    return SO3SSDResult(
        y=y.to(U.dtype).contiguous(),
        state=state.flatten(-2, -1).contiguous(),
        b_last=B[:, :, -1].contiguous(),
        u_last=U[:, :, -1].contiguous(),
    )


class ChunkedForward(NamedTuple):
    """Every chunk-local intermediate of the chunked factorization.

    Produced by :func:`chunked_forward`. The backward rematerializes this from the
    saved inputs, so it is the single definition of what the two passes share.
    Nothing here crosses a kernel boundary on the fast path.

    Time is chunked: an axis pair ``(C,L)`` replaces ``T``, with the ragged tail
    zero-padded. ``d`` denotes the flattened ``3N``.

    Attributes:
        length: Chunk length ``L``.
        seqlen: Unpadded ``T``, for slicing the tail off the output.
        w: Rotation vectors, ``(B,H,C,L,3)``.
        lprefix: Chunk-local log-scale prefix ``lp``, ``(B,H,C,L)``.
        tap: Tap parameters, ``(B,H,C,L,2,3)``.
        u: ``U`` chunked, ``(B,H,C,L,P)``.
        ushift: ``u_{t-1}`` chunked, ``(B,H,C,L,P)``.
        b: ``B`` chunked, ``(B,H,C,L,3N)``.
        bshift: ``b_{t-1}`` chunked, ``(B,H,C,L,3N)``.
        c: ``C`` chunked, ``(B,H,C,L,3N)``.
        quat: Per-token quaternion ``q_t``, ``(B,H,C,L,4)``.
        qprefix: Chunk-local quaternion prefix ``Q_t``, ``(B,H,C,L,4)``.
        table: The per-token 3x3 transforms.
        crot: ``R(Q_t)^T c_t``, ``(B,H,C,L,N,3)``.
        bnow: ``R(Q_t)^T Kcurr_t b_t``, ``(B,H,C,L,N,3)``.
        bprv: ``R(Q_t)^T Kprev_t b_{t-1}``, ``(B,H,C,L,N,3)``.
        score_now: ``<crot_t, bnow_r>``, ``(B,H,C,L,L)``. Unmasked.
        score_prv: ``<crot_t, bprv_r>``, ``(B,H,C,L,L)``. Unmasked.
        dmask: ``exp(2*(lp_t - lp_r)) * [r <= t]``, ``(B,H,C,L,L)``.
        wgt: ``exp(2*(lp_{L-1} - lp_r))``, ``(B,H,C,L,1)``.
        inc_local: Chunk increment before the frame change, ``(B,H,C,P,3N)``.
        zstart: State entering each chunk, ``(B,H,C,P,N,3)``.
        state: State after the last token, ``(B,H,P,N,3)``.
        y: Output in the pinned dtype, tail already sliced, ``(B,H,T,P)``.
        y_off: The ``zstart`` half of ``y``, still chunked, ``(B,H,C,L,P)``.
    """

    length: int
    seqlen: int
    w: Tensor
    lprefix: Tensor
    tap: Tensor
    u: Tensor
    ushift: Tensor
    b: Tensor
    bshift: Tensor
    c: Tensor
    quat: Tensor
    qprefix: Tensor
    table: TransformTable
    crot: Tensor
    bnow: Tensor
    bprv: Tensor
    score_now: Tensor
    score_prv: Tensor
    dmask: Tensor
    wgt: Tensor
    inc_local: Tensor
    zstart: Tensor
    state: Tensor
    y: Tensor
    y_off: Tensor


def chunked_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> ChunkedForward:
    """Evaluate the chunked factorization and keep every intermediate.

    Vectorized over ``T``. The only Python loop is over chunks, which is the
    inter-chunk recurrence the ``state_passing`` kernel owns.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned.
        B: Input vectors, shape ``(B,G,T,3N)``. Grouped: head ``h`` reads group
            ``h // (H // G)``.
        C: Output vectors, shape ``(B,G,T,3N)``. Grouped like ``B``.
        chunk_size: Chunk length ``L``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.

    Returns:
        A :class:`ChunkedForward`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, or a
            non-positive ``chunk_size``.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    shapes = _check_inputs(U, trans, K, B, C, z0, b_prev, u_prev)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    dtype = pinned_dtype(U, trans, K, B, C)
    device = U.device
    length = chunk_size

    with autocast_disabled(device.type):
        u = _promote(U, dtype)
        # B and C are grouped, everything else is per head. Broadcast once here so
        # the factorization below is written per head throughout.
        bvec = to_heads(_promote(B, dtype), shapes.heads)
        cvec = to_heads(_promote(C, dtype), shapes.heads)
        bhead = (
            torch.zeros(
                shapes.bsz,
                shapes.heads,
                1,
                shapes.state_dim,
                dtype=dtype,
                device=device,
            )
            if b_prev is None
            else to_heads(_promote(b_prev, dtype), shapes.heads)[:, :, None]
        )
        uhead = (
            torch.zeros(
                shapes.bsz, shapes.heads, 1, shapes.rows, dtype=dtype, device=device
            )
            if u_prev is None
            else _promote(u_prev, dtype)[:, :, None]
        )
        bshift = torch.cat([bhead, bvec[:, :, :-1]], dim=2)
        ushift = torch.cat([uhead, u[:, :, :-1]], dim=2)

        w = chunk_pad(_promote(trans[..., :3], dtype), length)
        lprefix = torch.cumsum(
            chunk_pad(_promote(trans[..., 3:], dtype), length)[..., 0], dim=-1
        )
        tap = chunk_pad(_promote(K[..., :3].flatten(-2, -1), dtype), length).unflatten(
            -1, (2, 3)
        )
        u_c = chunk_pad(u, length)
        ushift_c = chunk_pad(ushift, length)
        b_c = chunk_pad(bvec, length)
        bshift_c = chunk_pad(bshift, length)
        c_c = chunk_pad(cvec, length)
        n_chunks = int(w.shape[2])

        quat = quat_exp(w)
        qprefix = quat_prefix_scan(quat)
        table = transform_table(w, tap, qprefix)
        crot = torch.einsum("bhclij,bhclnj->bhclni", table.ac, as_lanes(c_c))
        bnow = torch.einsum("bhclij,bhclnj->bhclni", table.an, as_lanes(b_c))
        bprv = torch.einsum("bhclij,bhclnj->bhclni", table.ap, as_lanes(bshift_c))
        crot_f = crot.flatten(-2, -1)
        bnow_f = bnow.flatten(-2, -1)
        bprv_f = bprv.flatten(-2, -1)

        # I3: one exponential of a log difference. Masking before the exponential
        # keeps the strictly-upper triangle at exactly zero with no infinity.
        causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        dmask = torch.exp(
            (2.0 * (lprefix[..., :, None] - lprefix[..., None, :])).masked_fill(
                ~causal, -float("inf")
            )
        )
        score_now = crot_f @ bnow_f.transpose(-1, -2)
        score_prv = crot_f @ bprv_f.transpose(-1, -2)
        y_diag = (score_now * dmask) @ u_c + (score_prv * dmask) @ ushift_c

        # I6: fold the increment weight into u, size P, not into brot, size 3N.
        wgt = torch.exp(2.0 * (lprefix[..., -1:] - lprefix))[..., None]
        inc_local = torch.einsum(
            "bhclp,bhcld->bhcpd", u_c * wgt, bnow_f
        ) + torch.einsum("bhclp,bhcld->bhcpd", ushift_c * wgt, bprv_f)
        chunk_rot = table.rot[..., -1, :, :]
        inc = torch.einsum("bhcij,bhcpnj->bhcpni", chunk_rot, as_lanes(inc_local))
        chunk_scale = torch.exp(2.0 * lprefix[..., -1])

        state = (
            torch.zeros(
                shapes.bsz,
                shapes.heads,
                shapes.rows,
                shapes.lanes,
                3,
                dtype=dtype,
                device=device,
            )
            if z0 is None
            else as_lanes(_promote(z0, dtype))
        )
        starts = []
        for c in range(n_chunks):
            starts.append(state)
            state = (
                chunk_scale[:, :, c, None, None, None]
                * torch.einsum("bhij,bhpnj->bhpni", chunk_rot[:, :, c], state)
                + inc[:, :, c]
            )
        zstart = torch.stack(starts, dim=2)

        y_off = torch.exp(2.0 * lprefix)[..., None] * torch.einsum(
            "bhclni,bhcpni->bhclp", crot, zstart
        )
        y = (y_diag + y_off).flatten(2, 3)[:, :, : shapes.seqlen]

    return ChunkedForward(
        length=length,
        seqlen=shapes.seqlen,
        w=w,
        lprefix=lprefix,
        tap=tap,
        u=u_c,
        ushift=ushift_c,
        b=b_c,
        bshift=bshift_c,
        c=c_c,
        quat=quat,
        qprefix=qprefix,
        table=table,
        crot=crot,
        bnow=bnow,
        bprv=bprv,
        score_now=score_now,
        score_prv=score_prv,
        dmask=dmask,
        wgt=wgt,
        inc_local=inc_local,
        zstart=zstart,
        state=state,
        y=y,
        y_off=y_off,
    )


class FusedForward(NamedTuple):
    """Every chunk-local intermediate of the one-tap factorization.

    Produced by :func:`chunked_forward_fused`. Time is chunked: an axis pair
    ``(C,L)`` replaces ``T``. ``d`` denotes the flattened ``3N``.

    Attributes:
        length: Chunk length ``L``.
        seqlen: Unpadded ``T``, for slicing the tail off the output.
        w: Rotation vectors, ``(B,H,C,L,3)``.
        lprefix: Chunk-local log-scale prefix ``lp``, ``(B,H,C,L)``.
        step: Per-token decay ``exp(2*ls_s)``, ``(B,H,C,L)``. The factor the fused
            column carries.
        tap: Tap parameters, ``(B,H,C,L,2,3)``.
        u: ``U`` chunked, ``(B,H,C,L,P)``. Zero past the ragged tail.
        ushift: ``u_{t-1}`` over the padded sequence, ``(B,H,C,L,P)``. Slot ``n`` of
            a ragged tail chunk holds ``u_{T-1}``, not zero.
        b: ``B`` chunked, ``(B,H,C,L,3N)``. Zero past the ragged tail.
        bshift: ``b_{t-1}`` over the padded sequence, ``(B,H,C,L,3N)``. Slot ``n``
            of a ragged tail chunk holds ``b_{T-1}``, not zero.
        c: ``C`` chunked, ``(B,H,C,L,3N)``.
        qprefix: Chunk-local quaternion prefix ``Q_t``, ``(B,H,C,L,4)``.
        table: The per-token 3x3 transforms of the two-tap form.
        afuse: ``ap_s + exp(2*ls_s) * an_{s-1}``, ``(B,H,C,L,3,3)``. Replaces
            ``table.ap`` in the table a kernel reads; column ``s = 0`` is ``ap_0``.
        crot: ``R(Q_t)^T c_t``, ``(B,H,C,L,N,3)``.
        bnow: ``R(Q_t)^T Kcurr_t b_t``, ``(B,H,C,L,N,3)``. Only the two residues
            read it, at one slot each.
        bfuse: ``Afuse_s b_{s-1}``, ``(B,H,C,L,N,3)``. The single score operand.
        dnow: ``<crot_t, bnow_t>``, ``(B,H,C,L)``. The diagonal residue, equal to
            ``<c_t, Kcurr_t b_t>``: both frames cancel.
        score: ``<crot_t, bfuse_s>``, ``(B,H,C,L,L)``. Unmasked.
        dmask: ``exp(2*(lp_t - lp_s)) * [s <= t]``, ``(B,H,C,L,L)``. Unchanged from
            the two-tap form.
        wgt: ``exp(2*(lp_{L-1} - lp_s))``, ``(B,H,C,L,1)``.
        inc_local: Chunk increment before the frame change, ``(B,H,C,P,3N)``.
        zstart: State entering each chunk, ``(B,H,C,P,N,3)``.
        state: State after the last token, ``(B,H,P,N,3)``.
        y: Output in the pinned dtype, tail already sliced, ``(B,H,T,P)``.
        y_off: The ``zstart`` half of ``y``, still chunked, ``(B,H,C,L,P)``.
    """

    length: int
    seqlen: int
    w: Tensor
    lprefix: Tensor
    step: Tensor
    tap: Tensor
    u: Tensor
    ushift: Tensor
    b: Tensor
    bshift: Tensor
    c: Tensor
    qprefix: Tensor
    table: TransformTable
    afuse: Tensor
    crot: Tensor
    bnow: Tensor
    bfuse: Tensor
    dnow: Tensor
    score: Tensor
    dmask: Tensor
    wgt: Tensor
    inc_local: Tensor
    zstart: Tensor
    state: Tensor
    y: Tensor
    y_off: Tensor


def chunked_forward_fused(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> FusedForward:
    """Evaluate the one-tap factorization and keep every intermediate.

    Same operator as :func:`chunked_forward`, four dense GEMMs per chunk instead of
    seven. Reindexing the now-tap of token ``s-1`` onto slot ``s`` gives both taps
    the same operand ``b_{s-1}``, the same weight ``u_{s-1}``, and the same causal
    mask, so one column of the table carries both:

        Afuse_s  = ap_s + exp(2*ls_s) * an_{s-1},        Afuse_0 = ap_0
        y_diag_t = <crot_t, bnow_t> u_t
                   + sum_{s<=t} dmask[t,s] <crot_t, Afuse_s b_{s-1}> u_{s-1}
        inc      = sum_s wgt_s * u_{s-1} (x) (Afuse_s b_{s-1})
                   + u_{L-1} (x) bnow_{L-1}

    The two residues are what the reindex leaves over: row ``t = s-1`` is outside
    the mask, and slot ``L`` is outside the chunk. Neither is a GEMM.

    ``Afuse_0`` takes no ``an`` term. The previous chunk's ``an_{L-1}`` lives in the
    previous chunk's frame and its contribution already arrives through ``zstart``,
    so injecting it is wrong, not redundant.

    The shifted operands are built from the zero-padded sequence rather than
    zero-padded after the shift. At ``T mod L == n > 0`` the reindex puts the last
    real token's now-tap in slot ``n``, and padding the shift would put a zero
    there. That term reaches ``state`` only, because pad columns enter rows
    ``t >= n`` alone and the tail slice discards those, so ``y`` cannot see the
    difference.

    A zero-padded token is therefore not a no-op here. ``u`` and ``b`` are zero past
    the tail, ``ushift`` and ``bshift`` are not, and a pad token's table row is the
    one an identity rotation and zero taps produce.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned.
        B: Input vectors, shape ``(B,G,T,3N)``. Grouped: head ``h`` reads group
            ``h // (H // G)``.
        C: Output vectors, shape ``(B,G,T,3N)``. Grouped like ``B``.
        chunk_size: Chunk length ``L``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.

    Returns:
        A :class:`FusedForward`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, or a
            non-positive ``chunk_size``.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    shapes = _check_inputs(U, trans, K, B, C, z0, b_prev, u_prev)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    dtype = pinned_dtype(U, trans, K, B, C)
    device = U.device
    length = chunk_size

    with autocast_disabled(device.type):
        u = _promote(U, dtype)
        bvec = to_heads(_promote(B, dtype), shapes.heads)
        cvec = to_heads(_promote(C, dtype), shapes.heads)
        bhead = (
            torch.zeros(
                shapes.bsz,
                shapes.heads,
                1,
                shapes.state_dim,
                dtype=dtype,
                device=device,
            )
            if b_prev is None
            else to_heads(_promote(b_prev, dtype), shapes.heads)[:, :, None]
        )
        uhead = (
            torch.zeros(
                shapes.bsz, shapes.heads, 1, shapes.rows, dtype=dtype, device=device
            )
            if u_prev is None
            else _promote(u_prev, dtype)[:, :, None]
        )
        # Shift the padded sequence. Padding the shift puts zero in slot n of the
        # tail chunk, where the reindex puts the last real token's now-tap.
        bshift = torch.cat([bhead, pad_time(bvec, length)[:, :, :-1]], dim=2)
        ushift = torch.cat([uhead, pad_time(u, length)[:, :, :-1]], dim=2)

        w = chunk_pad(_promote(trans[..., :3], dtype), length)
        ls = chunk_pad(_promote(trans[..., 3:], dtype), length)[..., 0]
        lprefix = torch.cumsum(ls, dim=-1)
        tap = chunk_pad(_promote(K[..., :3].flatten(-2, -1), dtype), length).unflatten(
            -1, (2, 3)
        )
        u_c = chunk_pad(u, length)
        ushift_c = chunk_pad(ushift, length)
        b_c = chunk_pad(bvec, length)
        bshift_c = chunk_pad(bshift, length)
        c_c = chunk_pad(cvec, length)
        n_chunks = int(w.shape[2])

        quat = quat_exp(w)
        qprefix = quat_prefix_scan(quat)
        table = transform_table(w, tap, qprefix)
        step = torch.exp(2.0 * ls)
        # I3 holds: ls <= 0 puts both this factor and dmask in (0,1], so the fused
        # column is a product of two decays and never underflow times overflow.
        an_shift = torch.cat(
            [torch.zeros_like(table.an[..., :1, :, :]), table.an[..., :-1, :, :]],
            dim=-3,
        )
        afuse = table.ap + step[..., None, None] * an_shift

        crot = torch.einsum("bhclij,bhclnj->bhclni", table.ac, as_lanes(c_c))
        bnow = torch.einsum("bhclij,bhclnj->bhclni", table.an, as_lanes(b_c))
        bfuse = torch.einsum("bhclij,bhclnj->bhclni", afuse, as_lanes(bshift_c))
        crot_f = crot.flatten(-2, -1)
        bnow_f = bnow.flatten(-2, -1)
        bfuse_f = bfuse.flatten(-2, -1)

        causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        dmask = torch.exp(
            (2.0 * (lprefix[..., :, None] - lprefix[..., None, :])).masked_fill(
                ~causal, -float("inf")
            )
        )
        score = crot_f @ bfuse_f.transpose(-1, -2)
        # Rotation-free: R(Q_t) is orthogonal, so this is <c_t, Kcurr_t b_t> and the
        # residue needs neither the quaternion prefix nor the chunk-local frame.
        dnow = (crot_f * bnow_f).sum(-1)
        y_diag = (score * dmask) @ ushift_c + dnow[..., None] * u_c

        wgt = torch.exp(2.0 * (lprefix[..., -1:] - lprefix))[..., None]
        inc_local = (
            torch.einsum("bhclp,bhcld->bhcpd", ushift_c * wgt, bfuse_f)
            + u_c[..., -1, :, None] * bnow_f[..., -1, None, :]
        )
        chunk_rot = table.rot[..., -1, :, :]
        inc = torch.einsum("bhcij,bhcpnj->bhcpni", chunk_rot, as_lanes(inc_local))
        chunk_scale = torch.exp(2.0 * lprefix[..., -1])

        state = (
            torch.zeros(
                shapes.bsz,
                shapes.heads,
                shapes.rows,
                shapes.lanes,
                3,
                dtype=dtype,
                device=device,
            )
            if z0 is None
            else as_lanes(_promote(z0, dtype))
        )
        starts = []
        for c in range(n_chunks):
            starts.append(state)
            state = (
                chunk_scale[:, :, c, None, None, None]
                * torch.einsum("bhij,bhpnj->bhpni", chunk_rot[:, :, c], state)
                + inc[:, :, c]
            )
        zstart = torch.stack(starts, dim=2)

        y_off = torch.exp(2.0 * lprefix)[..., None] * torch.einsum(
            "bhclni,bhcpni->bhclp", crot, zstart
        )
        y = (y_diag + y_off).flatten(2, 3)[:, :, : shapes.seqlen]

    return FusedForward(
        length=length,
        seqlen=shapes.seqlen,
        w=w,
        lprefix=lprefix,
        step=step,
        tap=tap,
        u=u_c,
        ushift=ushift_c,
        b=b_c,
        bshift=bshift_c,
        c=c_c,
        qprefix=qprefix,
        table=table,
        afuse=afuse,
        crot=crot,
        bnow=bnow,
        bfuse=bfuse,
        dnow=dnow,
        score=score,
        dmask=dmask,
        wgt=wgt,
        inc_local=inc_local,
        zstart=zstart,
        state=state,
        y=y,
        y_off=y_off,
    )


def so3ssd_ref(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    chunk_size: int,
    *,
    z0: Tensor | None = None,
    b_prev: Tensor | None = None,
    u_prev: Tensor | None = None,
) -> SO3SSDResult:
    """Chunked reference. Defines the shape the CUDA kernels implement.

    A thin projection of :func:`chunked_forward` onto the operator's return
    contract. A ragged tail is zero-padded, and a zero-padded token is an exact
    no-op.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned.
        B: Input vectors, shape ``(B,G,T,3N)``. Grouped: head ``h`` reads group
            ``h // (H // G)``.
        C: Output vectors, shape ``(B,G,T,3N)``. Grouped like ``B``.
        chunk_size: Chunk length ``L``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,G,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.

    Returns:
        A :class:`SO3SSDResult`.

    Raises:
        ValueError: On a shape, contiguity, device, or pairing violation, or a
            non-positive ``chunk_size``.
        TypeError: On an unsupported dtype, or a low-precision pinned tensor.
    """
    fw = chunked_forward(
        U, trans, K, B, C, chunk_size, z0=z0, b_prev=b_prev, u_prev=u_prev
    )
    return SO3SSDResult(
        y=fw.y.to(U.dtype).contiguous(),
        state=fw.state.flatten(-2, -1).contiguous(),
        b_last=B[:, :, -1].contiguous(),
        u_last=U[:, :, -1].contiguous(),
    )
