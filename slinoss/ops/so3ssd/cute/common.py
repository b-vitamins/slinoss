"""Device-side transition math and layout policy for the SO(3) scan kernels.

One implementation each of the quaternion exponential, composition, conjugation,
normalization, the homogeneous rotation matrix, the tap chart, and the 3x3
composition. Every kernel in this package calls these. Duplicated device math
diverges, and the divergence is a correctness bug.

Everything here is a plain Python function over ``cutlass.Float32`` scalars and
tuples of them, so a call from inside a ``@cute.kernel`` is inlined at trace time
and the loops over tuple entries unroll in the Python interpreter. Nothing here
emits dynamic control flow: I1 (``ls <= 0``) and I2 (``|w| <= w_max < pi``) make
every branch unreachable, which is what holds average active threads per warp at
32.00.

Precision. Every quantity here is float32 (I4). ``U``, ``B``, ``C``, ``Y``, the
score matrix, and GEMM operands are the only tensors allowed to be narrower, and
none of them appear in this module.

Decay. ``exp(2*x)`` is evaluated as ``exp2(x * TWO_LOG2_E)`` on the hardware
``ex2.approx.f32`` path. ``x`` is always a log-prefix difference, never a sum of
two separately exponentiated terms (I3).

Shared-memory layouts. Per-chunk staging is component-major: consecutive tokens
are consecutive addresses within one component. One thread owns one token, so
every access is unit stride across the warp and no bank conflict is reachable.
The 3x3 table is token-major with the nine entries innermost; a nine-word stride
is coprime with the 32 banks, so the build stores conflict-free, and every read
of it during application is a broadcast.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import get_smem_capacity_in_bytes

from slinoss.ops.so3ssd.reference import series_coeffs

__all__ = [
    "COS_HALF",
    "FP32_SERIES_TERMS",
    "LOG2_E",
    "SINC_HALF",
    "TWO_LOG2_E",
    "assert_smem_fits",
    "cute_dtype",
    "decay",
    "dev_tensor",
    "mat3_matvec",
    "mat3_mul",
    "mat3_transpose",
    "quat_conj",
    "quat_exp",
    "quat_mul",
    "quat_normalize",
    "rot_hom",
    "select",
    "shuffle_up",
    "smem_capacity",
    "table_layout",
    "tap_layout",
    "tap_matrix",
    "trans_layout",
    "vec_layout",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# exp(x) == exp2(x * LOG2_E), exp(2*x) == exp2(x * TWO_LOG2_E). One multiply
# ahead of one ex2.approx.f32.
LOG2_E: float = math.log2(math.e)
TWO_LOG2_E: float = 2.0 * LOG2_E

# The float64 reference truncates the half-angle series at 14 terms. Over the
# reachable domain s = |w|^2 <= pi^2 the term at k = 9 is 5e-13 relative to a
# scalar part of order one, so 10 terms is exact to float32 rounding with three
# terms of margin. Both truncations come from one generator in the reference, so
# the device series cannot drift from the authority it is checked against.
FP32_SERIES_TERMS: int = 10

COS_HALF: tuple[float, ...] = series_coeffs(0, FP32_SERIES_TERMS)
SINC_HALF: tuple[float, ...] = series_coeffs(1, FP32_SERIES_TERMS)

_TORCH_TO_CUTE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def cute_dtype(dtype: torch.dtype) -> type:
    """Map a torch dtype to the CuTe numeric type.

    Args:
        dtype: One of bfloat16, float16, float32.

    Returns:
        The corresponding ``cutlass`` numeric type.

    Raises:
        TypeError: If the dtype has no kernel path.
    """
    try:
        return _TORCH_TO_CUTE[dtype]
    except KeyError:
        raise TypeError(f"no so3ssd kernel path for {dtype}") from None


def dev_tensor(tensor: torch.Tensor) -> cute.Tensor:
    """Wrap a contiguous torch tensor for a kernel launch.

    Only the trailing mode is declared contiguous; every other stride stays
    dynamic, so one compiled kernel serves every batch, head, and chunk count.
    All tensor contracts here are time-major and contiguous, so the 16-byte
    alignment claim holds for any allocation torch returns.

    Args:
        tensor: A contiguous CUDA tensor.

    Returns:
        The CuTe view of it.
    """
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


# ---------------------------------------------------------------------------
# Shared-memory capacity
# ---------------------------------------------------------------------------


def smem_capacity() -> int:
    """Opt-in shared-memory capacity per block, in bytes.

    Queried from the DSL's own architecture, so no architecture string appears
    here or in any caller. The 48 KiB default is not the budget: the DSL attaches
    the dynamic-shared-memory opt-in attribute to every kernel it generates.

    Returns:
        Capacity in bytes.
    """
    return get_smem_capacity_in_bytes()


def assert_smem_fits(name: str, nbytes: int) -> int:
    """Check one kernel's shared-memory budget against the queried capacity.

    Args:
        name: Kernel name, for the message.
        nbytes: Bytes the kernel's layouts add up to.

    Returns:
        ``nbytes``, so a caller can use this inline.

    Raises:
        ValueError: If the budget exceeds capacity. There is no slop constant:
            either the layouts fit or the layouts change.
    """
    capacity = smem_capacity()
    if nbytes > capacity:
        raise ValueError(
            f"{name} needs {nbytes} B of shared memory, capacity is {capacity} B"
        )
    return nbytes


# ---------------------------------------------------------------------------
# Per-chunk shared-memory layouts
# ---------------------------------------------------------------------------


def trans_layout(chunk: int) -> cute.Layout:
    """Staging layout for ``trans``: ``(4, L)``, component-major.

    Component ``0..2`` is the rotation vector, ``3`` is the log scale.
    """
    return cute.make_layout((4, chunk), stride=(chunk, 1))


def tap_layout(chunk: int) -> cute.Layout:
    """Staging layout for ``K``: ``(8, L)``, component-major.

    Component ``4*tap + j`` is ``(kr, g, h, 0)[j]`` of tap ``tap``. Lane 3 of
    each tap is the hard zero the float4 alignment of the global tensor carries.
    """
    return cute.make_layout((8, chunk), stride=(chunk, 1))


def vec_layout(chunk: int, width: int) -> cute.Layout:
    """Per-token vector layout: ``(width, L)``, component-major."""
    return cute.make_layout((width, chunk), stride=(chunk, 1))


def table_layout(chunk: int) -> cute.Layout:
    """3x3 transform table: ``(3, L, 9)``, nine entries innermost.

    Matrix ``0`` is ``Ac = R(Q_t)^T``, ``1`` is ``Ap = R(Q_t)^T Kprev_t``, ``2``
    is ``An = R(Q_t)^T Kcurr_t``. Entry ``3*r + c`` is row ``r`` column ``c``.
    """
    return cute.make_layout((3, chunk, 9), stride=(9 * chunk, 9, 1))


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

Scalar = cutlass.Float32
Quat = tuple[Scalar, Scalar, Scalar, Scalar]
Vec3 = tuple[Scalar, Scalar, Scalar]
Mat3 = tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]


def shuffle_up(value: Scalar, offset: int) -> Scalar:
    """``shfl.sync.up`` across a full warp: lane ``l`` reads lane ``l - offset``.

    Lanes below ``offset`` keep their own value, which is the identity the scans
    here rely on.

    The clamp field of the shuffle's packed operand is a lower bound on the
    source lane for the ``up`` direction, so a full-warp up-shuffle needs zero
    there. The DSL default is ``31``, which makes every lane read itself and
    turns the scan into a doubling.

    Args:
        value: The value to shift.
        offset: Lane distance. Compile-time in every caller here.

    Returns:
        Lane ``l - offset``'s value, or the lane's own below ``offset``.
    """
    return cute.arch.shuffle_sync_up(value, offset, mask_and_clamp=0)


def select(cond: cutlass.Boolean, if_true: Scalar, if_false: Scalar) -> Scalar:
    """Branchless float32 select.

    ``cute.where`` is a tensor operation and rejects two scalar operands, so the
    scalar form goes through the DSL's conditional expression and is retyped.
    Lowers to one ``arith.select``, which is one predicated move: no divergence,
    so average active threads per warp is unaffected.

    Args:
        cond: A dynamic predicate. A compile-time predicate belongs in an
            ``if cutlass.const_expr(...)``, not here.
        if_true: Value taken where the predicate holds.
        if_false: Value taken elsewhere.

    Returns:
        The selected value.
    """
    return cutlass.Float32(cutlass.select_(cond, if_true, if_false))


def _horner(s: Scalar, coeffs: tuple[float, ...]) -> Scalar:
    out = cutlass.Float32(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        out = out * s + coeff
    return out


def decay(log_diff: Scalar) -> Scalar:
    """``exp(2 * log_diff)``.

    Args:
        log_diff: A difference of log-scale prefixes, never a bare prefix pair
            (I3). Non-positive wherever the segment decay is formed, so the
            result lies in ``(0, 1]`` and overflow is unreachable (I1).

    Returns:
        The decay factor.
    """
    return cute.exp2(log_diff * TWO_LOG2_E)


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
    rotation matrix squared, so the reciprocal square root is taken at full
    float32 accuracy rather than on the approximate path.

    Args:
        q: A quaternion of near-unit norm.

    Returns:
        ``q / |q|``.
    """
    scale = cute.rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] * scale, q[1] * scale, q[2] * scale, q[3] * scale)


def rot_hom(q: Quat) -> Mat3:
    """Rotation matrix of a quaternion, homogeneous of degree two.

    ``R(q) = (qw^2 - v.v) I + 2 v v^T + 2 qw skew(v)`` with ``v = (qx,qy,qz)``.
    For a unit quaternion this is the usual rotation matrix. Degree-two
    homogeneity makes it also the whole transition of a scaled quaternion:
    ``rot_hom(exp(lp) * Q) == exp(2*lp) * R(Q)`` exactly, which is how a chunk
    transition travels as four floats instead of four floats and a scale.

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
