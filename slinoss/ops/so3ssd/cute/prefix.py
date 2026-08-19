"""Chunk-local prefixes: the log-scale scan and the quaternion prefix product.

Two quantities, one per token of a chunk:

- ``lp_t = sum_{j<=t} ls_j``, a commutative float32 scan.
- ``Q_t = q_t (*) ... (*) q_0``, a non-commutative quaternion prefix product,
  renormalized once after the scan (I5).

Both are shared by all ``N`` lanes and all ``P`` rows of the chunk, so one warp
computes them and the block reads them from shared memory. They never cross a
kernel boundary and never touch global memory: every kernel that needs them,
forward or backward, recomputes them from ``trans``.

Structure. Lane ``l`` of warp 0 owns ``seg = ceil(L/32)`` consecutive tokens and
runs them serially, then the lane totals are combined by a shuffle scan of
``ceil(log2(min(32, L)))`` rounds and folded back as an exclusive lane offset.
Sequential depth is ``L/32 + log2(32)`` rather than ``L``.

The lane predicate is a select, not a branch. The tail predicate exists only when
``L`` is not a multiple of the warp width, and it is resolved at compile time.
The one dynamic branch is warp-uniform, so average active threads per warp stays
at 32.00.
"""

import cutlass
import cutlass.cute as cute

from slinoss.ops.so3ssd.cute.common import (
    LOG2_E,
    Quat,
    Scalar,
    quat_exp,
    quat_mul,
    quat_normalize,
    select,
    shuffle_up,
)

__all__ = ["chunk_prefixes", "quat_prefix_endpoint"]


def _scan_offsets(active: int) -> tuple[int, ...]:
    """Shuffle distances of a Hillis-Steele scan over ``active`` lanes."""
    offsets: list[int] = []
    reach = 1
    while reach < active:
        offsets.append(reach)
        reach *= 2
    return tuple(offsets)


def _identity_quat() -> Quat:
    return (
        cutlass.Float32(1.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
    )


def _inclusive_add(value: Scalar, lane: cutlass.Int32, active: int) -> Scalar:
    """Inclusive add-scan of one float across the first ``active`` lanes."""
    for offset in _scan_offsets(active):
        shifted = shuffle_up(value, offset)
        value = select(lane >= offset, value + shifted, value)
    return value


def _inclusive_quat(quat: Quat, lane: cutlass.Int32, active: int) -> Quat:
    """Inclusive non-commutative quaternion scan across the first ``active`` lanes.

    ``out_l = q_l (*) ... (*) q_0``. The shuffled value is always the right
    operand, which is why this cannot reuse the add-scan.
    """
    for offset in _scan_offsets(active):
        shifted = tuple(shuffle_up(c, offset) for c in quat)
        combined = quat_mul(quat, shifted)
        take = lane >= offset
        quat = (
            select(take, combined[0], quat[0]),
            select(take, combined[1], quat[1]),
            select(take, combined[2], quat[2]),
            select(take, combined[3], quat[3]),
        )
    return quat


def _exclusive_add(total: Scalar, lane: cutlass.Int32) -> Scalar:
    """Shift an inclusive lane add-scan down one lane, identity into lane 0."""
    shifted = shuffle_up(total, 1)
    return select(lane >= 1, shifted, cutlass.Float32(0.0))


def _exclusive_quat(total: Quat, lane: cutlass.Int32) -> Quat:
    """Shift an inclusive lane quaternion scan down one lane, identity into lane 0."""
    shifted = tuple(shuffle_up(c, 1) for c in total)
    keep = lane >= 1
    return (
        select(keep, shifted[0], cutlass.Float32(1.0)),
        select(keep, shifted[1], cutlass.Float32(0.0)),
        select(keep, shifted[2], cutlass.Float32(0.0)),
        select(keep, shifted[3], cutlass.Float32(0.0)),
    )


@cute.jit
def chunk_prefixes(
    strans: cute.Tensor,
    slp: cute.Tensor,
    squat: cute.Tensor,
    tid: cutlass.Int32,
    chunk: cutlass.Constexpr,
) -> None:
    """Fill the two chunk-local prefixes in shared memory.

    Entered by the whole block; only warp 0 does work. The block barrier is the
    caller's, because only the caller knows what else it staged.

    Args:
        strans: ``(4, L)`` float32, component-major. Rows ``0..2`` are the
            rotation vector, row ``3`` is the log scale.
        slp: ``(L,)`` float32, written with the inclusive log-scale scan.
        squat: ``(4, L)`` float32, component-major, written with the inclusive
            quaternion prefix product, renormalized once.
        tid: Thread index within the block.
        chunk: ``L``. Compile-time.

    Invariants:
        ``ls <= 0`` (I1), so ``slp`` is monotone non-increasing and every decay
        formed from its differences lies in ``(0, 1]``.
    """
    seg = (chunk + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
    active = min(cute.arch.WARP_SIZE, chunk)
    exact = chunk % cute.arch.WARP_SIZE == 0

    if tid < cute.arch.WARP_SIZE:
        lane = tid
        base = lane * seg

        # Serial pass over the lane's own tokens. The running quaternion products
        # are kept so the second pass does not redo the exponentials.
        run_lp = cutlass.Float32(0.0)
        run_q = _identity_quat()
        local_lp: list[Scalar] = []
        local_q: list[Quat] = []
        for j in cutlass.range_constexpr(seg):
            idx = base + j
            if cutlass.const_expr(exact):
                wvec = (strans[0, idx], strans[1, idx], strans[2, idx])
                ls = strans[3, idx]
            else:
                # Clamp rather than branch: the read stays in bounds and the
                # value is then replaced by the scan identity.
                pos = cutlass.min(idx, chunk - 1)
                inside = idx < chunk
                zero = cutlass.Float32(0.0)
                wvec = (
                    select(inside, strans[0, pos], zero),
                    select(inside, strans[1, pos], zero),
                    select(inside, strans[2, pos], zero),
                )
                ls = select(inside, strans[3, pos], zero)
            run_lp = run_lp + ls
            run_q = quat_mul(quat_exp(wvec), run_q)
            local_lp.append(run_lp)
            local_q.append(run_q)

        # Combine the lane totals, then fold the exclusive offset back in.
        off_lp = _exclusive_add(_inclusive_add(run_lp, lane, active), lane)
        off_q = _exclusive_quat(_inclusive_quat(run_q, lane, active), lane)

        for j in cutlass.range_constexpr(seg):
            idx = base + j
            total_lp = local_lp[j] + off_lp
            total_q = quat_normalize(quat_mul(local_q[j], off_q))
            if cutlass.const_expr(exact):
                slp[idx] = total_lp
                squat[0, idx] = total_q[0]
                squat[1, idx] = total_q[1]
                squat[2, idx] = total_q[2]
                squat[3, idx] = total_q[3]
            else:
                if idx < chunk:
                    slp[idx] = total_lp
                    squat[0, idx] = total_q[0]
                    squat[1, idx] = total_q[1]
                    squat[2, idx] = total_q[2]
                    squat[3, idx] = total_q[3]


def quat_prefix_endpoint(squat: cute.Tensor, slp: cute.Tensor, chunk: int) -> Quat:
    """The chunk transition, packed as one scaled quaternion.

    ``exp(lp_{L-1}) * Q_{L-1}``. :func:`slinoss.ops.so3ssd.cute.common.rot_hom`
    is homogeneous of degree two, so its matrix for this argument is exactly
    ``exp(2*lp_{L-1}) * R(Q_{L-1})``: the chunk-level scale rides inside the four
    floats instead of travelling beside them. ``lp_{L-1} <= 0`` by I1, so the
    factor lies in ``(0, 1]``, underflow degrades to zero, and no overflow is
    ever multiplied by an underflow (I3).

    Args:
        squat: ``(4, L)`` float32 quaternion prefix.
        slp: ``(L,)`` float32 log-scale prefix.
        chunk: ``L``. Compile-time.

    Returns:
        ``(qw, qx, qy, qz)`` scaled by ``exp(lp_{L-1})``.
    """
    last = chunk - 1
    scale = cute.exp2(slp[last] * LOG2_E)
    return (
        squat[0, last] * scale,
        squat[1, last] * scale,
        squat[2, last] * scale,
        squat[3, last] * scale,
    )
