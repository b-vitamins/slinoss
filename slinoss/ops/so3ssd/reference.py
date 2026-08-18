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

Value invariants -- ``ls <= 0`` and ``|w| < pi`` -- are the parameterization's
job, not this module's. Nothing here scans a tensor to validate a numerical
range and nothing here clamps.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import pad as _pad

from slinoss._precision import (
    autocast_disabled,
    check_pinned,
    check_supported,
    pinned_dtype,
)
from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE

__all__ = [
    "SO3SSDResult",
    "TransformTable",
    "quat_conj",
    "quat_exp",
    "quat_mul",
    "quat_prefix_scan",
    "rot_matrix",
    "skew",
    "so3ssd_ref",
    "so3ssm",
    "tap_matrix",
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
# undefined at w = 0. The reachable domain is the closed ball of radius pi, where
# s/4 <= 2.4675 and the k = 12 term falls below 1e-19 relative, so 14 terms are
# exact to float64 rounding over all of it. Sizing for the closed ball rather
# than for w_max also absorbs the last-ulp rounding of the parameter map.
# tests/test_quat.py measures that against the transcendental form.
# ---------------------------------------------------------------------------

_SERIES_TERMS = 14


def _series_coeffs(offset: int) -> tuple[float, ...]:
    return tuple(
        (-1.0) ** k / (4.0**k * math.factorial(2 * k + offset))
        for k in range(_SERIES_TERMS)
    )


_COS_HALF: tuple[float, ...] = _series_coeffs(0)
_SINC_HALF: tuple[float, ...] = _series_coeffs(1)


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


class SO3SSDResult(NamedTuple):
    """Return type of both reference implementations.

    Every field is contiguous. A ragged tail leaves the chunked path holding a
    time slice of a padded buffer, and that must not reach a caller.

    Attributes:
        y: Output, shape ``(B,H,T,P)``, dtype of ``U``.
        state: State after the last token, shape ``(B,H,P,3N)``, pinned dtype.
        b_last: ``b`` at the last token, shape ``(B,H,3N)``, contiguous. Feeds
            ``b_prev`` of the next call in a streaming split.
        u_last: ``u`` at the last token, shape ``(B,H,P)``, contiguous. Feeds
            ``u_prev``.

    ``b_last`` and ``u_last`` are contiguous because they are fed straight back
    in and the operator repacks nothing. A time slice of ``B`` is strided over
    heads, so it is copied here rather than at every call site.
    """

    y: Tensor
    state: Tensor
    b_last: Tensor
    u_last: Tensor


class _Shapes(NamedTuple):
    bsz: int
    heads: int
    seqlen: int
    rows: int
    state_dim: int
    lanes: int


def _lanes(t: Tensor) -> Tensor:
    """``(...,3N) -> (...,N,3)``.

    Lane-major: element ``3n+i`` is component ``i`` of 3-vector ``n``.
    """
    return t.unflatten(-1, (-1, 3))


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
        raise ValueError(f"B must be (B,H,T,3N), got shape {tuple(B.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in U.shape)
    state_dim = int(B.shape[-1])
    if seqlen < 1:
        raise ValueError("T must be at least 1")

    named: list[tuple[str, Tensor, tuple[int, ...]]] = [
        ("trans", trans, (bsz, heads, seqlen, 4)),
        ("K", K, (bsz, heads, seqlen, 2, 4)),
        ("B", B, (bsz, heads, seqlen, state_dim)),
        ("C", C, (bsz, heads, seqlen, state_dim)),
    ]
    if z0 is not None:
        named.append(("z", z0, (bsz, heads, rows, state_dim)))
    if b_prev is not None:
        named.append(("b_prev", b_prev, (bsz, heads, state_dim)))
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

    for name, tensor in [("U", U), *((n, t) for n, t, _ in named)]:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous; no repacking is done")
        if tensor.device != U.device:
            raise ValueError(
                f"{name} is on {tensor.device}, U is on {U.device}; one device only"
            )
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

    return _Shapes(bsz, heads, seqlen, rows, state_dim, state_dim // 3)


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
        B: Input vectors, shape ``(B,H,T,3N)``.
        C: Output vectors, shape ``(B,H,T,3N)``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,H,3N)``.
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
        blane = _lanes(_promote(B, dtype))
        clane = _lanes(_promote(C, dtype))

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
            else _lanes(_promote(z0, dtype))
        )
        bp = (
            torch.zeros(
                shapes.bsz, shapes.heads, shapes.lanes, 3, dtype=dtype, device=device
            )
            if b_prev is None
            else _lanes(_promote(b_prev, dtype))
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

    Vectorized over ``T``. The only Python loop is over chunks, which is the
    inter-chunk recurrence the ``state_passing`` kernel owns. A ragged tail is
    zero-padded, and a zero-padded token is an exact no-op: ``w = 0`` and
    ``ls = 0`` give the identity transition while a zero tap kills the forcing.

    Args:
        U: Input weights, shape ``(B,H,T,P)``.
        trans: ``(w_x, w_y, w_z, ls)`` per token, shape ``(B,H,T,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, shape ``(B,H,T,2,4)``, pinned.
        B: Input vectors, shape ``(B,H,T,3N)``.
        C: Output vectors, shape ``(B,H,T,3N)``.
        chunk_size: Chunk length ``L``.
        z0: Initial state, shape ``(B,H,P,3N)``, pinned. Zero if omitted.
        b_prev: ``b_{-1}`` for a streaming split, shape ``(B,H,3N)``.
        u_prev: ``u_{-1}`` for a streaming split, shape ``(B,H,P)``.

    Returns:
        A :class:`SO3SSDResult`.

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
    tail = (-shapes.seqlen) % length
    n_chunks = (shapes.seqlen + tail) // length

    with autocast_disabled(device.type):
        u = _promote(U, dtype)
        bvec = _promote(B, dtype)
        cvec = _promote(C, dtype)
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
            else _promote(b_prev, dtype)[:, :, None]
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

        def chunked(t: Tensor) -> Tensor:
            if tail:
                t = _pad(t, (0, 0, 0, tail))
            return t.unflatten(2, (n_chunks, length))

        w = chunked(_promote(trans[..., :3], dtype))
        lprefix = torch.cumsum(chunked(_promote(trans[..., 3:], dtype))[..., 0], dim=-1)
        tap = chunked(_promote(K[..., :3].flatten(-2, -1), dtype)).unflatten(-1, (2, 3))
        u_c = chunked(u)
        ushift_c = chunked(ushift)
        b_c = chunked(bvec)
        bshift_c = chunked(bshift)
        c_c = chunked(cvec)

        table = transform_table(w, tap, quat_prefix_scan(quat_exp(w)))
        crot = torch.einsum("bhclij,bhclnj->bhclni", table.ac, _lanes(c_c))
        bnow = torch.einsum("bhclij,bhclnj->bhclni", table.an, _lanes(b_c))
        bprv = torch.einsum("bhclij,bhclnj->bhclni", table.ap, _lanes(bshift_c))
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
        y_diag = ((crot_f @ bnow_f.transpose(-1, -2)) * dmask) @ u_c + (
            (crot_f @ bprv_f.transpose(-1, -2)) * dmask
        ) @ ushift_c

        # I6: fold the increment weight into u, size P, not into brot, size 3N.
        wgt = torch.exp(2.0 * (lprefix[..., -1:] - lprefix))[..., None]
        inc_local = torch.einsum(
            "bhclp,bhcld->bhcpd", u_c * wgt, bnow_f
        ) + torch.einsum("bhclp,bhcld->bhcpd", ushift_c * wgt, bprv_f)
        chunk_rot = table.rot[..., -1, :, :]
        inc = torch.einsum("bhcij,bhcpnj->bhcpni", chunk_rot, _lanes(inc_local))
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
            else _lanes(_promote(z0, dtype))
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

    return SO3SSDResult(
        y=y.to(U.dtype).contiguous(),
        state=state.flatten(-2, -1).contiguous(),
        b_last=B[:, :, -1].contiguous(),
        u_last=U[:, :, -1].contiguous(),
    )
