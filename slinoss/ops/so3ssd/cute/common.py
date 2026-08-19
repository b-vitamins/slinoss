"""Device-side transition math and shared-memory tiles for the SO(3) scan.

One implementation each of the quaternion exponential, composition, conjugation,
normalization, the homogeneous rotation matrix, the tap chart, and the 3x3
composition. Every kernel in this package calls these. Duplicated device math
diverges, and the divergence is a correctness bug.

DSL shims that no operator owns -- the scalar retype, the select, the shuffle, the
dtype map, the DLPack wrapper, the tile and its budget -- are in
:mod:`slinoss._cute`.

Everything here is a plain Python function over ``cutlass.Float32`` scalars and
tuples of them, so a call from inside a ``@cute.kernel`` is inlined at trace time
and the loops over tuple entries unroll in the Python interpreter. Nothing here
emits dynamic control flow: I1 (``ls <= 0``) and I2 (``|w| <= w_max < pi``) make
every branch unreachable. No function here contributes divergence, so whatever a
caller's active-thread ratio is, this math does not lower it.

Precision. Every quantity here is float32 (I4). ``U``, ``B``, ``C``, ``Y``, the
score matrix, and GEMM operands are the only tensors allowed to be narrower, and
none of them appear in this module.

Shared-memory tiles. Per-chunk staging is component-major: consecutive tokens
are consecutive addresses within one component. The staging and table builds are
one thread per token, so those accesses are unit stride across the warp. The
prefix scan is the exception: it gives lane ``l`` a block of ``ceil(L/32)``
consecutive tokens, so its reads are strided by that block size. Measured with
``l1tex__data_bank_conflicts_pipe_lsu_mem_shared``, both counters are zero at
every legal chunk size, because the block is at most four words and the compiler
vectorizes it into one wide load. That is why ``MAX_CHUNK`` is 128: at 256 the
block is eight words, wider than any vector load, and the counters go nonzero.
The 3x3 table is token-major with the nine entries innermost; a nine-word stride
is coprime with the 32 banks, so the build stores conflict-free, and every read
of it during application is a broadcast.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from slinoss._cute import Scalar, Tile, f32
from slinoss.ops.so3ssd.reference import series_coeffs

__all__ = [
    "COS_HALF",
    "FP32_SERIES_TERMS",
    "SINC_HALF",
    "TABLE_AC",
    "TABLE_AN",
    "TABLE_AP",
    "THREADS",
    "WARPS",
    "mat3_matvec",
    "mat3_mul",
    "mat3_transpose",
    "quat_conj",
    "quat_exp",
    "quat_mul",
    "quat_normalize",
    "rot_hom",
    "scalar_tile",
    "table_tile",
    "tap_matrix",
    "tap_tile",
    "trans_tile",
    "vec_tile",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The float64 reference truncates the half-angle series at 14 terms. Over the
# reachable domain s = |w|^2 <= pi^2 the term at k = 9 is 5e-13 relative to a
# scalar part of order one, so 10 terms is exact to float32 rounding with three
# terms of margin. Both truncations come from one generator in the reference, so
# the device series cannot drift from the authority it is checked against.
FP32_SERIES_TERMS: int = 10

COS_HALF: tuple[float, ...] = series_coeffs(0, FP32_SERIES_TERMS)
SINC_HALF: tuple[float, ...] = series_coeffs(1, FP32_SERIES_TERMS)

WARPS: int = 4
"""Warps per block, everywhere in the tree.

Set by the tiled MMA: four warps partition the M mode of every scan GEMM, so the
block width is fixed by the operator's arithmetic rather than chosen per kernel.
Kernels with no GEMM use the same width so one launch geometry covers the tree.
"""

THREADS: int = WARPS * 32
"""Threads per block. ``P*N`` is a multiple of ``HEAD_MULTIPLE * LANE_MULTIPLE``,
which is a multiple of this, so a lane-per-thread launch is exact: no tail tile,
no bounds predicate, no padding path."""


def trans_tile(chunk: int) -> Tile:
    """Staging tile for ``trans``: ``(4, L)``, component-major.

    Component ``0..2`` is the rotation vector, ``3`` is the log scale.
    """
    return Tile((4, chunk), (chunk, 1))


def tap_tile(chunk: int) -> Tile:
    """Staging tile for ``K``: ``(8, L)``, component-major.

    Component ``4*tap + j`` is ``(kr, g, h, 0)[j]`` of tap ``tap``. Lane 3 of
    each tap is the hard zero the float4 alignment of the global tensor carries.
    """
    return Tile((8, chunk), (chunk, 1))


def vec_tile(chunk: int, width: int) -> Tile:
    """Per-token vector tile: ``(width, L)``, component-major."""
    return Tile((width, chunk), (chunk, 1))


def scalar_tile(chunk: int) -> Tile:
    """One float32 per token: ``(L,)``, dense."""
    return Tile((chunk,), (1,))


TABLE_AP: int = 0
"""``Ap = R(Q_t)^T Kprev_t``, applied to ``b_{t-1}``."""

TABLE_AN: int = 1
"""``An = R(Q_t)^T Kcurr_t``, applied to ``b_t``."""

TABLE_AC: int = 2
"""``Ac = R(Q_t)^T``, applied to ``c_t``."""


def table_tile(chunk: int, mats: int = 3) -> Tile:
    """3x3 transform table: ``(mats, L, 9)``, nine entries innermost.

    Slot order is :data:`TABLE_AP`, :data:`TABLE_AN`, :data:`TABLE_AC`. Entry
    ``3*r + c`` of a slot is row ``r`` column ``c``.

    The two tap matrices come first so a kernel that forces but does not read out
    -- the chunk increment -- takes a prefix of the table and the slot indices
    stay the same constants in both kernels.

    Args:
        chunk: ``L``.
        mats: Slots to allocate, 2 or 3. Two omits ``Ac``.
    """
    if mats not in (2, 3):
        raise ValueError(f"table needs 2 or 3 matrices, got {mats}")
    return Tile((mats, chunk, 9), (9 * chunk, 9, 1))


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

Quat = tuple[Scalar, Scalar, Scalar, Scalar]
Vec3 = tuple[Scalar, Scalar, Scalar]
Mat3 = tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]


def _horner(s: Scalar, coeffs: tuple[float, ...]) -> Scalar:
    out = cutlass.Float32(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * s + coeff
    return out


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def quat_exp(w: Vec3) -> Quat:
    """Unit quaternion of a rotation vector, scalar-first.

    Two minimax-free polynomials in ``s = |w|^2``, branchless over the whole
    reachable domain by I2. Ten terms, exact to float32 rounding.

    Args:
        w: Rotation vector ``(wx, wy, wz)``.

    Returns:
        ``(qw, qx, qy, qz)``.
    """
    s = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    half_sinc = 0.5 * _horner(s, SINC_HALF)
    return (
        _horner(s, COS_HALF),
        half_sinc * w[0],
        half_sinc * w[1],
        half_sinc * w[2],
    )


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product ``a (*) b``, so ``R(a (*) b) == R(a) R(b)``."""
    return (
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def quat_conj(q: Quat) -> Quat:
    """Conjugate, which inverts the rotation of a unit quaternion."""
    return (q[0], -q[1], -q[2], -q[3])


def quat_normalize(q: Quat) -> Quat:
    """Project a quaternion back to unit norm.

    Applied once per chunk, after the prefix scan (I5). Rotation error enters the
    rotation matrix squared, so drift is projected out at every chunk boundary
    rather than allowed to compound across chunks.

    ``cute.rsqrt`` lowers to ``rsqrt.approx.f32``: on sm_86 it is bitwise
    identical with and without fastmath, and its measured relative error over
    ``[0.1, 4)`` is 1.24e-07. That error is the residual norm drift, and it does
    not grow with the chunk size, which is the property the projection buys.

    Args:
        q: A quaternion of near-unit norm.

    Returns:
        ``q / |q|``, to the accuracy of the approximate reciprocal square root.
    """
    inv = f32(cute.rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]))
    return (q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv)


def rot_hom(q: Quat) -> Mat3:
    """Rotation matrix of a quaternion, homogeneous of degree two.

    ``R(q) = (qw^2 - v.v) I + 2 v v^T + 2 qw skew(v)`` with ``v = (qx,qy,qz)``.
    For a unit quaternion this is the usual rotation matrix.

    The homogeneous form is used rather than the unit-norm one because it costs
    nothing: the diagonal carries ``qw^2 - v.v`` either way, and hardcoding it to
    one would need the norm to be exact, which after a prefix product it is not.

    Args:
        q: A quaternion, not necessarily unit.

    Returns:
        Row-major 3x3, entry ``3*r + c``.
    """
    xx = q[1] * q[1]
    yy = q[2] * q[2]
    zz = q[3] * q[3]
    diag = q[0] * q[0] - xx - yy - zz
    xy = q[1] * q[2]
    xz = q[1] * q[3]
    yz = q[2] * q[3]
    wx = q[0] * q[1]
    wy = q[0] * q[2]
    wz = q[0] * q[3]
    return (
        diag + 2.0 * xx,
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        diag + 2.0 * yy,
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        diag + 2.0 * zz,
    )


# ---------------------------------------------------------------------------
# Tap chart and 3x3 algebra
# ---------------------------------------------------------------------------


def tap_matrix(tap: Vec3, w: Vec3) -> Mat3:
    """Matrix of the tap ``K(v) = kr v + g (w.v) w + h (w x v)``.

    Polynomial in ``w``, analytic at the origin. The axis normal form is
    singular there; this chart makes well-definedness structural, so no clamp
    and no whole-tensor validity check exist anywhere on the path.

    Args:
        tap: ``(kr, g, h)``.
        w: The rotation vector of the same token.

    Returns:
        Row-major 3x3, entry ``3*r + c``.
    """
    kr, g, h = tap
    gx = g * w[0]
    gy = g * w[1]
    gz = g * w[2]
    hx = h * w[0]
    hy = h * w[1]
    hz = h * w[2]
    return (
        kr + gx * w[0],
        gx * w[1] - hz,
        gx * w[2] + hy,
        gy * w[0] + hz,
        kr + gy * w[1],
        gy * w[2] - hx,
        gz * w[0] - hy,
        gz * w[1] + hx,
        kr + gz * w[2],
    )


def mat3_mul(a: Mat3, b: Mat3) -> Mat3:
    """Row-major 3x3 product ``a @ b``."""
    out = []
    for r in range(3):
        for c in range(3):
            out.append(
                a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c]
            )
    return (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8])


def mat3_transpose(a: Mat3) -> Mat3:
    """Row-major 3x3 transpose."""
    return (a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8])


def mat3_matvec(a: Mat3, v: Vec3) -> Vec3:
    """Row-major 3x3 matrix times 3-vector. Nine FMAs."""
    return (
        a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
        a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
        a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
    )
