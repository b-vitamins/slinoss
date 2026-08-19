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
Sequential depth is ``seg + ceil(log2(min(32, L)))`` rather than ``L``.

Reverse scans. The backward needs the mirror of each: :func:`chunk_suffix` for the
log-scale cotangent and :func:`quat_suffix_vjp` for the adjoint of the quaternion
prefix product. Same segmentation, same depth, with the shuffle running down and
the lane predicate bounding the source lane from above instead of from below. The
forward scans are not reusable in reverse: an up-shuffle scan carries values one
way only.

The lane predicate is a select, not a branch. One dynamic branch remains, the
store predicate, and it exists only when ``L`` is not a multiple of the warp
width; below the warp width it is not warp-uniform, because the idle lanes are
exactly the ones the predicate excludes. Measured on the probe kernel,
``smsp__thread_inst_executed_per_inst_executed.ratio`` is 22.10 at ``L = 16``,
30.66 at 32, and 31.19 to 31.54 from 64 up. Only warp 0 runs this at all, so at
the default block width the cost is bounded by a quarter of one warp's issue
slots over the scan.
"""

import cutlass
import cutlass.cute as cute

from slinoss._cute import Scalar, decay, select, shuffle_down, shuffle_up
from slinoss.ops.so3ssd.cute.common import (
    Quat,
    quat_add,
    quat_conj,
    quat_exp,
    quat_mul,
    quat_normalize,
)

__all__ = ["chunk_endpoint", "chunk_prefixes", "chunk_suffix", "quat_suffix_vjp"]


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


def _zero_quat() -> Quat:
    """Identity of the quaternion sum the adjoint accumulates, not of the product."""
    return (
        cutlass.Float32(0.0),
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


def _suffix_add(value: Scalar, lane: cutlass.Int32, active: int) -> Scalar:
    """Inclusive reverse add-scan of one float across the first ``active`` lanes.

    ``out_l = sum_{m >= l} value_m``. The mirror of :func:`_inclusive_add`: the
    shuffle runs down, so the predicate bounds the source lane from above. A lane
    that falls off the top of a down-shuffle keeps its own value, so the predicate
    is what supplies the identity, not the shuffle.
    """
    for offset in _scan_offsets(active):
        shifted = shuffle_down(value, offset)
        value = select(lane + offset < active, value + shifted, value)
    return value


def _suffix_add_quat(value: Quat, lane: cutlass.Int32, active: int) -> Quat:
    """Inclusive reverse add-scan of four floats. Componentwise: a sum of
    quaternions is a quaternion, so this needs none of :func:`_inclusive_quat`'s
    ordering."""
    return (
        _suffix_add(value[0], lane, active),
        _suffix_add(value[1], lane, active),
        _suffix_add(value[2], lane, active),
        _suffix_add(value[3], lane, active),
    )


def _exclusive_add(total: Scalar, lane: cutlass.Int32) -> Scalar:
    """Shift an inclusive lane add-scan down one lane, identity into lane 0."""
    shifted = shuffle_up(total, 1)
    return select(lane >= 1, shifted, cutlass.Float32(0.0))


def _exclusive_suffix_add(total: Scalar, lane: cutlass.Int32, active: int) -> Scalar:
    """Drop a lane's own total from an inclusive lane reverse add-scan: lane ``l``
    takes lane ``l + 1``'s. ``active`` is needed where :func:`_exclusive_add` needs
    nothing, because the identity goes to lane ``active - 1`` rather than to a
    fixed lane."""
    shifted = shuffle_down(total, 1)
    return select(lane + 1 < active, shifted, cutlass.Float32(0.0))


def _exclusive_suffix_add_quat(total: Quat, lane: cutlass.Int32, active: int) -> Quat:
    """Componentwise :func:`_exclusive_suffix_add`."""
    return (
        _exclusive_suffix_add(total[0], lane, active),
        _exclusive_suffix_add(total[1], lane, active),
        _exclusive_suffix_add(total[2], lane, active),
        _exclusive_suffix_add(total[3], lane, active),
    )


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


@cute.jit
def chunk_suffix(
    sval: cute.Tensor,
    sout: cute.Tensor,
    tid: cutlass.Int32,
    chunk: cutlass.Constexpr,
) -> None:
    """Reverse inclusive add-scan within a chunk: ``sout[t] = sum_{t' >= t} sval[t']``.

    The adjoint of the log-scale prefix is exactly this scan, because that prefix
    is a commutative sum: every token's log scale enters every later prefix once.

    Entered by the whole block; only warp 0 does work. The block barrier is the
    caller's, because only the caller knows what else it staged.

    Args:
        sval: ``(L,)`` float32 shared tile, read.
        sout: ``(L,)`` float32 shared tile, written.
        tid: Thread index within the block.
        chunk: ``L``. Compile-time.
    """
    seg = (chunk + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
    active = min(cute.arch.WARP_SIZE, chunk)
    exact = chunk % cute.arch.WARP_SIZE == 0

    if tid < cute.arch.WARP_SIZE:
        lane = tid
        base = lane * seg

        # Serial pass over the lane's own tokens, high index first, so ``local`` is
        # indexed by depth rather than by token: entry ``d`` is token
        # ``base + seg - 1 - d``. The out-of-range slots of a ragged chunk sit at
        # the high end of the segment, which is where this pass starts.
        run = cutlass.Float32(0.0)
        local: list[Scalar] = []
        for d in cutlass.range_constexpr(seg):
            idx = base + seg - 1 - d
            if cutlass.const_expr(exact):
                run = run + sval[idx]
            else:
                # Clamp rather than branch: the read stays in bounds and the
                # value is then replaced by the scan identity.
                pos = cutlass.min(idx, chunk - 1)
                run = run + select(idx < chunk, sval[pos], cutlass.Float32(0.0))
            local.append(run)

        # Combine the lane totals, then fold the exclusive offset back in.
        off = _exclusive_suffix_add(_suffix_add(run, lane, active), lane, active)

        for j in cutlass.range_constexpr(seg):
            idx = base + j
            total = local[seg - 1 - j] + off
            if cutlass.const_expr(exact):
                sout[idx] = total
            else:
                if idx < chunk:
                    sout[idx] = total


@cute.jit
def quat_suffix_vjp(
    squat: cute.Tensor,
    sdrot: cute.Tensor,
    sdquat: cute.Tensor,
    tid: cutlass.Int32,
    chunk: cutlass.Constexpr,
) -> None:
    """Adjoint of the quaternion prefix product :func:`chunk_prefixes` computes.

    ``Q_l = q_l (*) Q_{l-1}`` and right multiplication is its own adjoint under
    conjugation, so with ``Q_{-1}`` the identity::

        p_m  = dQ_m - <dQ_m, Q_m> Q_m
        S_l  = sum_{m >= l} conj(Q_m) (*) p_m
        dq_l = Q_l (*) S_l (*) conj(Q_{l-1})

    ``S`` is a reverse sum of four floats per token rather than a non-commutative
    scan, so the adjoint costs one suffix scan and two products, and the scan is
    :func:`_suffix_add_quat`.

    The projection is the adjoint of the once-per-chunk renormalization (I5), and
    it is not optional: ``dQ`` arrives with its radial component intact, so
    dropping the projection differentiates a non-unit quaternion the forward never
    returned. The prefix is unit to rounding, so the norm divide the projection
    accompanies is the identity and does not appear.

    Entered by the whole block; only warp 0 does work. The block barrier is the
    caller's, because only the caller knows what else it staged. ``sdquat`` must
    not alias ``squat``: the store pass reads ``Q_{l-1}``, which at a lane's first
    token belongs to the previous lane's segment.

    Args:
        squat: ``(4, L)`` float32 quaternion prefix, component-major and already
            renormalized, as :func:`chunk_prefixes` writes it.
        sdrot: ``(4, L)`` float32 cotangent of that prefix, component-major.
        sdquat: ``(4, L)`` float32, written with the cotangent of the per-step
            quaternions, component-major.
        tid: Thread index within the block.
        chunk: ``L``. Compile-time.
    """
    seg = (chunk + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
    active = min(cute.arch.WARP_SIZE, chunk)
    exact = chunk % cute.arch.WARP_SIZE == 0
    last = chunk - 1

    if tid < cute.arch.WARP_SIZE:
        lane = tid
        base = lane * seg

        # Serial pass over the lane's own tokens, high index first, so both lists
        # are indexed by depth rather than by token: entry ``d`` is token
        # ``base + seg - 1 - d``. The out-of-range slots of a ragged chunk sit at
        # the high end of the segment, which is where this pass starts. The prefix
        # is kept so the store pass does not reread it.
        run = _zero_quat()
        local_q: list[Quat] = []
        local_s: list[Quat] = []
        for d in cutlass.range_constexpr(seg):
            idx = base + seg - 1 - d
            if cutlass.const_expr(exact):
                pos = idx
                drot = (sdrot[0, pos], sdrot[1, pos], sdrot[2, pos], sdrot[3, pos])
            else:
                # Clamp rather than branch. A zero cotangent projects to zero
                # whatever the clamped prefix holds, which is the sum's identity,
                # so only the cotangent needs the select.
                pos = cutlass.min(idx, last)
                inside = idx < chunk
                zero = cutlass.Float32(0.0)
                drot = (
                    select(inside, sdrot[0, pos], zero),
                    select(inside, sdrot[1, pos], zero),
                    select(inside, sdrot[2, pos], zero),
                    select(inside, sdrot[3, pos], zero),
                )
            quat = (squat[0, pos], squat[1, pos], squat[2, pos], squat[3, pos])
            radial = (
                drot[0] * quat[0]
                + drot[1] * quat[1]
                + drot[2] * quat[2]
                + drot[3] * quat[3]
            )
            proj = (
                drot[0] - radial * quat[0],
                drot[1] - radial * quat[1],
                drot[2] - radial * quat[2],
                drot[3] - radial * quat[3],
            )
            local_q.append(quat)
            run = quat_add(run, quat_mul(quat_conj(quat), proj))
            local_s.append(run)

        # Combine the lane totals, then fold the exclusive offset back in.
        off = _exclusive_suffix_add_quat(
            _suffix_add_quat(run, lane, active), lane, active
        )

        for j in cutlass.range_constexpr(seg):
            idx = base + j
            depth = seg - 1 - j
            suffix = quat_add(local_s[depth], off)
            # Q_{l-1}, the identity at the chunk's first token. Both clamps keep
            # the read in bounds; the low one at token zero, the high one at a pad
            # slot whose store is skipped anyway.
            pos = cutlass.max(cutlass.min(idx - 1, last), 0)
            keep = idx >= 1
            prev = (
                select(keep, squat[0, pos], cutlass.Float32(1.0)),
                select(keep, squat[1, pos], cutlass.Float32(0.0)),
                select(keep, squat[2, pos], cutlass.Float32(0.0)),
                select(keep, squat[3, pos], cutlass.Float32(0.0)),
            )
            dquat = quat_mul(quat_mul(local_q[depth], suffix), quat_conj(prev))
            if cutlass.const_expr(exact):
                for c in cutlass.range_constexpr(4):
                    sdquat[c, idx] = dquat[c]
            else:
                if idx < chunk:
                    for c in cutlass.range_constexpr(4):
                        sdquat[c, idx] = dquat[c]


def chunk_endpoint(
    squat: cute.Tensor, slp: cute.Tensor, chunk: int
) -> tuple[Quat, Scalar]:
    """The chunk transition as a unit rotation and a separate scale.

    ``(Q_{L-1}, exp(2*lp_{L-1}))``. The two travel separately rather than as one
    scaled quaternion: the rotation stays unit, so the renormalization of I5 is
    what the consumer receives, and the scale keeps its own exponent instead of
    riding at half range inside four floats that then square. A scale small
    enough to underflow the packed form is a transition that is already zero to
    float32, but recovering the rotation from the packed form would need an
    ``rsqrt`` of that zero and a guard the invariants forbid.

    This is also the reference's own factorization of the chunk step, so the
    kernel and the authority agree term by term rather than up to an identity.

    ``lp_{L-1} <= 0`` by I1, so the scale lies in ``(0, 1]``, underflow degrades
    to zero, and no overflow is ever multiplied by an underflow (I3).

    Args:
        squat: ``(4, L)`` float32 quaternion prefix, already renormalized.
        slp: ``(L,)`` float32 log-scale prefix.
        chunk: ``L``. Compile-time.

    Returns:
        ``((qw, qx, qy, qz), exp(2*lp_{L-1}))``.
    """
    last = chunk - 1
    return (
        (squat[0, last], squat[1, last], squat[2, last], squat[3, last]),
        decay(slp[last]),
    )
