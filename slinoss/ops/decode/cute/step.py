"""One-token step of the SO(3) scan. CuTe DSL forward.

The map is :func:`slinoss.ops.decode.reference.decode_ref`'s, term for term:

    state_n <- scale * R(exp(w)) state_n + up * Kprev b_prev_n + u * Kcurr b_n
    y       <- sum_n <c_n, state_n>

over the ``N = 3N/3`` lanes of one ``(b,h,p)`` state row, with ``Kprev`` and
``Kcurr`` the two first-order-hold taps of the same token. Nothing is derived here.
Every transition quantity comes from :mod:`slinoss.ops.so3ssd.cute.common`, which is
the tree's only device-side implementation of the quaternion exponential, the
homogeneous rotation matrix, the tap chart, and the 3x3 matrix-vector product.

The chunked factorization is not mirrored. Its change of basis into the chunk-local
frame, and the ``Afuse`` column that fuses a token's previous tap into its
predecessor's current one, both belong to a factorization over ``T > 1`` tokens and
have no ``T = 1`` form: at one token there is no chunk to be local to and no
predecessor inside the chunk. Importing either into this path computes a different
operator.

Parallel decomposition. One state row is one independent problem: the transition is
lane-local and the readout reduces over the row's own ``3N`` elements and nothing
else. ``row_group`` threads share a row, thread ``t`` of the group owning lanes
``t, t + row_group, ...`` at :func:`lanes_per_thread` of them. The group is
:data:`cute.arch.WARP_SIZE` threads where ``N`` is a multiple of it and half that
otherwise: ``N`` is a multiple of :data:`slinoss.config.LANE_MULTIPLE`, which is 16,
so ``N = 16`` is legal and reachable -- ``d_state = 48`` is the smallest legal state
width and ``tests/test_decode_op.py`` runs it -- and a warp per row would leave half
the warp with no lane to own. Half a warp instead of a masked tail: a predicated
store cannot be replaced by a clamped one here, because a clamped store address puts
every masked thread on the last lane of the row, writing the same three addresses
with different values.

Owning a lane is not addressing one. A 3-vector is three consecutive float32, so a
thread that reads its own lane's three components reads at a 12-byte stride across
the group: the request spans ``12 * row_group`` bytes to use four of every twelve,
and every sector it touches is charged whole. Measured on sm_86 at ``3N = 240``,
``row_group = 16``: 10.71 sectors per store request against the four the bytes need,
67,092,480 store sectors over five launches, 3.02 times the DRAM traffic crossing
L1, and 52.77% of the fitted copy floor with ``long_scoreboard`` at 88.6% and
``issue_active`` at 9.0%. The stall is not bandwidth. It is sector capacity in
flight, three quarters of it spent on bytes no thread asked for.

So ``gssm`` is addressed by component plane, not by lane. The group's step of the
walk covers ``3 * row_group`` consecutive floats; three accesses at
``run + k * row_group + slot`` for ``k`` in 0, 1, 2 cover exactly those floats, each
one ``row_group`` consecutive float32 across the group and therefore sector-exact.
What arrives is one component of three different lanes rather than three components
of one, and the exchange back to one lane per thread is a permutation inside the
group. Three is invertible modulo a power of two -- 11 at both 16 and 32 -- so the
permutation factors into three shuffles per direction: on the way in each lane
selects which of its three planes to present and every destination reads one lane,
on the way out each lane reads one lane per component and selects which arrival
belongs to which plane. Six ``shfl.sync`` and twelve selects per step of the walk,
against 48 sectors per step deleted; the port they issue on runs at 14% of its
issue rate here, and the sectors are what the kernel is short of. Every index in
the exchange is a function of ``slot`` alone and is hoisted out of the walk.

What that reads, same card and same session as the figures above: 3.65 sectors per
store request and 2.76 per load request, 22,855,680 store sectors, 2,124.30 us
against 3,977.94, 669.75 GB/s against 357.64, and 98.94% of the fitted floor with
``long_scoreboard`` down to 63.9% and ``issue_active`` up to 20.9%. DRAM bytes did
not move: 1.0009 times compulsory before and after, so the traffic is the same
traffic and only its sector cost changed. 64 registers before and after, no local
memory in either. Warp instructions rose 23.9%, which is the exchange, and the
kernel is no longer where instructions are what it is short of. The store ratio is
3.65 and not 4.00 because the two per-row broadcasts, ``gy`` and ``guprev``, are
one-sector requests that the average is taken over.

``gb``, ``gc`` and ``gbprev`` keep lane addressing. They are indexed ``(b,g,n)`` with
no row axis, so every thread of the warp reads the same ``row_group`` addresses and
the two halves coalesce onto one span; the same transpose applies to them and buys
18 of the remaining sectors per step against nine more shuffles. Not taken: at
98.94% of the floor there is 1.06% left to buy and the arm costs instructions on a
kernel already at 92.1% of the memory speed of light.

Those nine lines are where every remaining sector excess is. A kernel-wide sectors
per request is not an instrument for locating a sector defect -- it is an average
over requests of different widths and dilutes the offender -- so the account is per
tensor, off the NCU source page under ``CUTE_DSL_LINEINFO=1``, which resolves
``memory_l2_theoretical_sectors_global`` and its ``_ideal`` onto the line that owns
them. At ``B = 128``, ``3N = 240``: the three ``gssm`` plane loads and the ``gssm``
store are at 2.000 sectors per tag request against an ideal of 2.000, zero excess,
and every scalar operand line is at 1.000 = ideal, so the transpose's own accesses
are sector-exact rather than merely cheaper. The nine ``gb``/``gc``/``gbprev`` lines
read 2.000 against an ideal of 1.333, which is 3,317,760 sectors, 16.73% of the
launch. At ``3N = 288`` and a 32-lane group the same nine read 3.000 against 1.000,
43.55%, and ``gssm`` stays at ideal. It does not bind: those three tensors are
L2-resident by the redundancy argument below, so the excess is tag work and not
DRAM bytes, and the kernel holds 98.40% of the floor at that width.

``THREADS // row_group`` rows per block and ``grid = (P / rows_per_block, H, B)``.
``P`` is a multiple of :data:`slinoss.config.HEAD_MULTIPLE`, which is 16, and
``rows_per_block`` is 4 or 8, so the launch is exact: no tail tile, no bounds
predicate, no padding path.

No shared memory and no barrier. The readout is a butterfly over the row group,
``log2(row_group)`` shuffles, which leaves the row's total in every lane of the
group with no separate broadcast; the group is an aligned power-of-two run inside one
warp, so flipping a lane bit below ``row_group`` never leaves it. The transpose's
shuffles are confined the same way, by the segment field of the packed operand
rather than by a mask: ``segmask = 32 - row_group`` makes the source lane
``(self & segmask) | (offset & ~segmask)``, which is an index that wraps inside the
group, and the clamp cannot fire because ``maxLane`` is the group's top lane.

The packed operand is ``((32 - row_group) << 8) | 31``, and the two legal row groups
give two different constants: ``0x101f``, which preserves the half-warp selector bit
of ``self``, and ``0x001f``, which is a whole-warp index with no segment at all. They
are two shuffle configurations, not one configuration parameterized by a variable,
and a wrap that is wrong at one of them still produces finite plausible numbers at
the other. Both are covered: the index-set test is parameterized over both widths and
asserts the constant and the permutation it implies, and the state-width sweep runs
``N`` at 16 through 128 through the kernel against the float64 oracle, which is both
widths at one and at three or more lanes per thread.

Two launches, not one. ``ssm``, ``u_prev`` and ``y`` are partitioned by ``(b,h,p)``,
so the threads that write a row's three buffers are the only threads that read them,
and the write is ordered against the read by the butterfly that sits between them:
``shfl.sync`` names every lane of the warp, so no lane reaches the store of
``u_prev`` before every lane has passed the load. ``b_prev`` is not partitioned that
way. It is indexed ``(b,g)``, so every one of the ``(H/G) * P / rows_per_block``
blocks whose head maps to group ``g`` reads the same row and one of them would have
to overwrite it, and a grid has no barrier to order that read against that write.
The overwrite is therefore a second launch, :func:`decode_carry_kernel`, ordered
against the first by the stream. Its alternatives were an aten ``copy_`` (a kernel in
the step's glue class for the same traffic), a staging buffer (a third pass over the
band), and an arrival counter (a hot-path zero fill the step does not have).

Redundant reads, not redundant traffic. ``rot``, the two tap matrices and ``scale``
depend on ``(b,h)`` alone, and ``b_n``, ``b_prev_n`` and ``c_n`` on ``(b,g,n)``
alone, so every row group in a block reads them again. They are L2-resident by the
time the second group asks -- one block covers 4 or 8 consecutive ``p`` at one
``(b,h)`` -- so the redundancy costs L1 tag work and LSU instructions, not DRAM
bytes. Staging them would need shared memory and therefore a barrier, and the
barrier is the thing this kernel does not have.

Loads and transforms share the lane loop, which the rule against that permits: the
trip count is a :class:`cutlass.Constexpr`, so the loop is unrolled at trace time and
its loads are a hoistable group rather than one global latency per step. The bound is
``N / row_group``, at most 8 over the legal state widths.

Precision. Every quantity is float32 (I4). ``U``, ``B``, ``C``, ``y`` and the two
activation carries are read and written at their own width and widened on load;
``ssm`` is the call's pinned dtype, which is float32 for every dtype this path
accepts. The three scalars are folded into the vector rather than into the matrix --
``R (scale * s)`` rather than ``(scale * R) s`` -- which is 9 multiplies per lane
against 27 per thread and is the cheaper of the two at the one and two lanes per
thread the common state widths give.

Traffic, per call, with ``s_a`` the activation element size and ``s_z`` the state's:

    read   U        B*H*P*s_a          write  y         B*H*P*s_a
    read   trans    16*B*H             write  ssm       B*H*P*3N*s_z
    read   K        32*B*H             write  b_prev    B*G*3N*s_a
    read   B        B*G*3N*s_a         write  u_prev    B*H*P*s_a
    read   C        B*G*3N*s_a
    read   ssm      B*H*P*3N*s_z

At ``B 1, H 16, P 64, 3N 96, G 1`` with bfloat16 activations and a float32 state that
is 793,920 B, of which the two state passes are 786,432 B: 99.06%. 12 flop per state
element against those 8 bytes is 1.5 flop/byte against a machine balance of 163 on
sm_86, so the arithmetic is under 1% of the roofline and the class is ``DRAM-bound``:
at least 85% of measured achievable bandwidth at the kernel's own footprint, per
``docs/kernels.md``. Held across the batch ladder at ``H 18, P 64, 3N 240, G 1``, one
fit per session: 100.80, 99.76, 99.29, 98.72% of the floor at ``B`` 16, 32, 64, 128.
``B = 8`` reads 102.79 and is not a pass -- its measured traffic is 0.967 times
compulsory, so 3.3% of it was served out of L2 and the ratio is inflated by exactly
what the floor is not charged for; ``B = 16``, at 1.00002, is the first rung whose
traffic is entirely compulsory. The class does not depend on the group width or on
the activation dtype: 99.60 and 98.40% at ``3N = 288``, where the group is a full
warp, and 98.17% at ``B = 128`` with float32 activations, which is the path
``decode`` dispatches at float32 and the shape at which the state is pinned float32
either way.

Every term above is linear in ``B``, so the footprint is ``793,920 * B`` bytes at
that shape and crosses the 6,291,456 B L2 of this part at ``B = 8``. Below it there is
no verdict and the kernel is named unjudged, which includes ``B = 1``, the shape a
decode step actually runs at. The judged reading is at ``B = 8`` and above; see
``docs/kernels.md`` on why a sub-L2 traffic figure is not a lower bound on the work a
launch did.

The block floor is the other thing the decode shape does not clear. ``grid`` covers
``B*H*P / rows_per_block`` blocks, which at ``B 1, H 16, P 64`` and a 32-thread row
group is 256, above twice the 84 multiprocessors of this part; at 8 rows per block --
any ``N`` that is 16 modulo 32 -- it is 128, under it. The parallelism is exactly
``B*H*P`` independent rows and no more, so a shape with too few rows to fill the
device is a statement about the shape.
"""

# No ``from __future__ import annotations`` in this module, and none in any other
# CuTe module of the tree. The DSL reads ``cutlass.Constexpr`` off the annotation
# object itself, so PEP 563 turns every compile-time parameter into a runtime one:
# the launch reports a dynamic block size and ``range_constexpr`` refuses its own
# trip count. Silent until trace time.
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Stream,
    decay,
    jit_launch,
    narrow,
    select,
    shuffle_xor,
    widen,
)
from slinoss._guard import check_dtypes
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.decode.reference import TOKENS, check_operands
from slinoss.ops.so3ssd.cute.common import (
    THREADS,
    mat3_matvec,
    quat_exp,
    rot_hom,
    tap_matrix,
)

__all__ = [
    "decode_carry",
    "decode_carry_kernel",
    "decode_forward",
    "decode_fwd",
    "decode_fwd_kernel",
    "lane_exchange",
    "lanes_per_thread",
    "row_group",
    "rows_per_block",
]


class LaneExchange(NamedTuple):
    """The component transpose's index set for one thread, hoisted out of the walk.

    ``group`` threads hold ``3 * group`` consecutive float32 of a state row in two
    layouts. Plane layout is what a sector-exact access produces: thread ``t`` holds
    the element at offset ``k * group + t`` in plane ``k``. Lane layout is what the
    recurrence needs: thread ``t`` holds offsets ``3t``, ``3t + 1``, ``3t + 2``, the
    three components of one 3-vector. Each layout is a bijection between ``group``
    threads times three registers and ``3 * group`` offsets, so the change of layout
    is a permutation of the group, and every field below is a function of the
    thread's own ``slot`` and of nothing in the walk.

    Both directions are three shuffles because ``3`` is a unit modulo ``group``.
    Inbound the destination map ``t -> (3t + c) mod group`` is a bijection for each
    fixed component ``c``, so one shuffle per component suffices and the source lane
    chooses which plane it presents. Outbound the plane map ``t -> (k*group + t)
    div 3`` is three-to-one, so the same factorization is taken on the component
    index instead -- the source map ``s -> (3s + c) mod group`` is a bijection -- and
    the destination chooses which arrival belongs to which plane.

    Attributes:
        segment: The ``mask_and_clamp`` operand of every shuffle in the exchange:
            ``segmask = 32 - group`` in bits 12:8 and the warp's top lane index in
            bits 4:0. Confines the index to the thread's own group and puts
            ``maxLane`` at the group's top lane, so the clamp never fires.
        inbound_lane: Per component ``c``, the lane this thread reads, which is
            ``(3*slot + c) mod group``.
        inbound_plane: Per component ``c``, the plane this thread presents to
            whichever lane reads it in shuffle ``c``.
        outbound_lane: Per component ``c``, the lane this thread reads, which is
            ``inv3 * (slot - c) mod group``. Also the thread whose plane index the
            inbound direction is derived from, so the two share one register.
        outbound_pick: Per plane ``k``, which of the three outbound arrivals is the
            element at offset ``k*group + slot``, which is ``(slot + k*group) mod 3``.
    """

    segment: int
    inbound_lane: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    inbound_plane: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    outbound_lane: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
    outbound_pick: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]


def lane_exchange(slot: cutlass.Int32, group: cutlass.Constexpr) -> LaneExchange:
    """Build the component transpose's index set for one thread.

    Args:
        slot: This thread's index inside its row group, in ``[0, group)``.
        group: Threads per state row, 16 or 32. A power of two, so ``3`` is a unit
            modulo it and every modulus below is a mask.

    Returns:
        The index set. Eight dynamic integers, all invariant in the lane walk.
    """
    warp = cute.arch.WARP_SIZE
    inv3 = pow(3, -1, group)
    shift = group.bit_length() - 1
    mask = group - 1
    # ``slot + group - c`` rather than ``slot - c``: a mask is a modulus only on a
    # non-negative value, and the shifted argument is congruent because ``group``
    # is one period.
    outbound = tuple(((slot + group - c) * inv3) & mask for c in range(3))
    return LaneExchange(
        segment=((warp - group) << 8) | (warp - 1),
        inbound_lane=tuple((3 * slot + c) & mask for c in range(3)),
        inbound_plane=tuple((3 * outbound[c] + c) >> shift for c in range(3)),
        outbound_lane=outbound,
        outbound_pick=tuple((slot + k * group) % 3 for k in range(3)),
    )


def _pick(index: cutlass.Int32, first: object, second: object, third: object) -> object:
    """One of three float32 registers by a dynamic index in ``[0, 3)``.

    Two ``arith.select``, no branch. The index is loop-invariant and the values are
    not, so this is a register move per call and not an address computation.

    Args:
        index: 0, 1 or 2. Values outside that range select ``third``.
        first: Value at index 0.
        second: Value at index 1.
        third: Value at index 2.

    Returns:
        The selected value.
    """
    return select(index == 0, first, select(index == 1, second, third))


def _gather(value: object, lane: cutlass.Int32, segment: int) -> object:
    """``shfl.sync.idx`` inside one row group: read ``lane``'s ``value``.

    Args:
        value: The float32 this thread offers.
        lane: The lane to read, taken modulo the group width by the segment field.
        segment: :attr:`LaneExchange.segment`.

    Returns:
        The value ``lane`` offered.
    """
    return cute.arch.shuffle_sync(value, lane, mask_and_clamp=segment)


def row_group(lanes: int) -> int:
    """Threads that share one state row.

    A whole warp where ``N`` is a multiple of the warp width, half a warp otherwise.
    ``N`` is a multiple of :data:`slinoss.config.LANE_MULTIPLE`, which is half a
    warp, so the half-warp group divides every legal ``N`` exactly and there is no
    masked tail at any state width.

    Args:
        lanes: ``N``, a positive multiple of 16.

    Returns:
        16 or 32.
    """
    warp = cute.arch.WARP_SIZE
    return warp if lanes % warp == 0 else warp // 2


def lanes_per_thread(lanes: int) -> int:
    """3-vectors one thread owns. ``N // row_group(lanes)``, exact.

    Args:
        lanes: ``N``, a positive multiple of 16.

    Returns:
        The trip count of the lane walk, at least one.
    """
    return lanes // row_group(lanes)


def rows_per_block(lanes: int) -> int:
    """State rows one block owns. ``THREADS // row_group(lanes)``, 4 or 8.

    Both divide :data:`slinoss.config.HEAD_MULTIPLE`, so the row axis of the grid is
    exact at every legal ``P``.

    Args:
        lanes: ``N``, a positive multiple of 16.

    Returns:
        Rows per block.
    """
    return THREADS // row_group(lanes)


@cute.kernel
def decode_fwd_kernel(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gk: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    gssm: cute.Tensor,
    gbprev: cute.Tensor,
    guprev: cute.Tensor,
    gy: cute.Tensor,
    heads_per_group: cutlass.Int32,
    threads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Advance one state row one token and read it out.

    Args:
        gu: ``(B,H,TOKENS,P)`` activations. The element type is read off the operand
            and is the width every activation of the call carries.
        gtrans: ``(B,H,TOKENS,4)`` float32, ``(w_x, w_y, w_z, ls)``.
        gk: ``(B,H,TOKENS,2,4)`` float32, ``(kr, g, h, 0)`` per tap. Tap 0 is
            previous, 1 is current; lane 3 is never read.
        gb: ``(B,G,TOKENS,3N)`` activations. Contiguous or one pitched band.
        gc: ``(B,G,TOKENS,3N)`` activations. Like ``gb``.
        gssm: ``(B,H,P,3N)`` float32. Read, then overwritten with the state after
            this token.
        gbprev: ``(B,G,3N)`` activations. Read only. The overwrite is
            :func:`decode_carry_kernel`.
        guprev: ``(B,H,P)`` activations. Read, then overwritten with ``U``.
        gy: ``(B,H,TOKENS,P)`` activations, written.
        heads_per_group: ``H // G``. Dynamic, so one variant covers every grouping.
        threads: Block width. Compile-time.
        group: Threads per state row, 16 or 32. Compile-time.
        lanes: 3-vectors one thread owns. Compile-time.

    Invariants:
        ``grid.x * (threads // group) == P`` exactly and ``group * lanes == N``
        exactly, so no thread is out of range and no lane is unowned. The reduction
        order is fixed by the launch geometry alone: ascending lane within a thread,
        then the butterfly. No atomics and no shared memory, so one shape reproduces
        bit for bit. ``|R| == 1`` to float32 and ``scale`` lies in ``(0, 1]`` by I1,
        so the homogeneous rotation cannot grow the state.

        Thread ``slot`` owns lane ``slot + step * group`` of the walk, which is what
        :class:`LaneExchange` restores after each plane-layout access, so the
        reduction sums the same lanes in the same order that lane addressing did and
        ``gb``, ``gc`` and ``gbprev`` keep the lane-indexed ``base``. The transpose is
        therefore bit-neutral: it moves which thread issues an access, not which
        value any expression is evaluated on.

        Each of the three plane accesses reads and writes one address per thread,
        ``run + k * group + slot``, and reads it before it writes it in that thread's
        own program order. No address is read by one thread and written by another,
        so the exchange needs no memory ordering beyond that; ``shfl.sync`` names
        every lane of the warp and orders the instructions, not the traffic. The two
        row groups of a warp sit on different ``pidx`` and their memory is disjoint,
        and the segment field keeps each group's shuffles inside itself.

        The store to ``guprev`` overwrites an address every lane of the row group
        loaded. It is ordered against those loads by the butterfly between them,
        which is a ``shfl.sync`` over the whole warp: no lane reaches the store
        before every lane has passed the load. ``gy`` and ``guprev`` are stored by
        every lane of the group rather than by lane zero. The butterfly leaves
        bitwise-identical values in all of them -- float32 addition is commutative,
        so a lane's partner sums the same pair in the other order -- so which lane
        wins the write cannot change the result, and predicating would add a
        divergent branch for no effect.
    """
    tid, _, _ = cute.arch.thread_idx()
    tile, hidx, bidx = cute.arch.block_idx()
    pidx = tile * (threads // group) + tid // group
    slot = tid % group
    gidx = hidx // heads_per_group

    act = gu.element_type
    pin = gtrans.element_type
    zdt = gssm.element_type

    w = (
        widen(gtrans[bidx, hidx, 0, 0], pin),
        widen(gtrans[bidx, hidx, 0, 1], pin),
        widen(gtrans[bidx, hidx, 0, 2], pin),
    )
    scale = decay(widen(gtrans[bidx, hidx, 0, 3], pin))
    rot = rot_hom(quat_exp(w))
    kprev = tap_matrix(
        (
            widen(gk[bidx, hidx, 0, 0, 0], pin),
            widen(gk[bidx, hidx, 0, 0, 1], pin),
            widen(gk[bidx, hidx, 0, 0, 2], pin),
        ),
        w,
    )
    kcurr = tap_matrix(
        (
            widen(gk[bidx, hidx, 0, 1, 0], pin),
            widen(gk[bidx, hidx, 0, 1, 1], pin),
            widen(gk[bidx, hidx, 0, 1, 2], pin),
        ),
        w,
    )
    u = widen(gu[bidx, hidx, 0, pidx], act)
    up = widen(guprev[bidx, hidx, pidx], act)

    swap = lane_exchange(slot, group)

    total = cutlass.Float32(0.0)
    for step in cutlass.range_constexpr(lanes):
        base = 3 * (slot + step * group)
        # ``run`` is the group's whole step of the walk: ``3 * group`` consecutive
        # float32, covered by three accesses of ``group`` consecutive float32 each.
        # ``run`` is a trace-time constant, so the three differ by an immediate.
        run = 3 * step * group
        plane = (
            widen(gssm[bidx, hidx, pidx, run + slot], zdt),
            widen(gssm[bidx, hidx, pidx, run + group + slot], zdt),
            widen(gssm[bidx, hidx, pidx, run + 2 * group + slot], zdt),
        )
        held = tuple(
            _gather(
                _pick(swap.inbound_plane[c], plane[0], plane[1], plane[2]),
                swap.inbound_lane[c],
                swap.segment,
            )
            for c in range(3)
        )
        rotated = mat3_matvec(
            rot,
            (held[0] * scale, held[1] * scale, held[2] * scale),
        )
        # A zero carry annihilates its term exactly: the tap matrix is applied to a
        # vector already scaled by the carry weight, so a zero weight or a zero
        # vector makes every product a zero and the sum an exact zero.
        forced = mat3_matvec(
            kprev,
            (
                widen(gbprev[bidx, gidx, base], act) * up,
                widen(gbprev[bidx, gidx, base + 1], act) * up,
                widen(gbprev[bidx, gidx, base + 2], act) * up,
            ),
        )
        driven = mat3_matvec(
            kcurr,
            (
                widen(gb[bidx, gidx, 0, base], act) * u,
                widen(gb[bidx, gidx, 0, base + 1], act) * u,
                widen(gb[bidx, gidx, 0, base + 2], act) * u,
            ),
        )
        sx = rotated[0] + forced[0] + driven[0]
        sy = rotated[1] + forced[1] + driven[1]
        sz = rotated[2] + forced[2] + driven[2]
        # Back to plane layout for the store. The three arrivals belong to the three
        # planes in an order that depends on ``slot``: an arrival gathered for
        # component ``c`` sits at an offset congruent to ``slot`` modulo ``group``,
        # and which of ``slot``, ``slot + group``, ``slot + 2*group`` it is, is fixed
        # by its residue modulo three.
        sent = (
            _gather(sx, swap.outbound_lane[0], swap.segment),
            _gather(sy, swap.outbound_lane[1], swap.segment),
            _gather(sz, swap.outbound_lane[2], swap.segment),
        )
        for k in cutlass.range_constexpr(3):
            gssm[bidx, hidx, pidx, run + k * group + slot] = narrow(
                _pick(swap.outbound_pick[k], sent[0], sent[1], sent[2]), zdt
            )
        # The readout contracts against the state after the forcing, over all 3N
        # elements of the row and not per 3-vector.
        total = total + (
            widen(gc[bidx, gidx, 0, base], act) * sx
            + widen(gc[bidx, gidx, 0, base + 1], act) * sy
            + widen(gc[bidx, gidx, 0, base + 2], act) * sz
        )

    # Plain Python loop over a compile-time bound: unrolled during the trace. Every
    # mask is below ``group``, so an aligned group of that width is closed under the
    # exchange and no lane reads outside its own row.
    reach = 1
    while reach < group:
        total = total + shuffle_xor(total, reach)
        reach *= 2
    gy[bidx, hidx, 0, pidx] = narrow(total, act)
    guprev[bidx, hidx, pidx] = narrow(u, act)


@cute.jit
def decode_fwd(
    gu: cute.Tensor,
    gtrans: cute.Tensor,
    gk: cute.Tensor,
    gb: cute.Tensor,
    gc: cute.Tensor,
    gssm: cute.Tensor,
    gbprev: cute.Tensor,
    guprev: cute.Tensor,
    gy: cute.Tensor,
    heads_per_group: cutlass.Int32,
    tiles: cutlass.Int32,
    heads: cutlass.Int32,
    bsz: cutlass.Int32,
    stream: Stream,
    threads: cutlass.Constexpr,
    group: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Launch :func:`decode_fwd_kernel` over ``(P / rows_per_block, H, B)``.

    Only ``threads``, ``group`` and ``lanes`` are compile-time, so one variant covers
    every batch, head, group and row count at one state width.
    """
    decode_fwd_kernel(
        gu,
        gtrans,
        gk,
        gb,
        gc,
        gssm,
        gbprev,
        guprev,
        gy,
        heads_per_group,
        threads,
        group,
        lanes,
    ).launch(grid=(tiles, heads, bsz), block=(threads, 1, 1), stream=stream)


@cute.kernel
def decode_carry_kernel(
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    width: cutlass.Int32,
    threads: cutlass.Constexpr,
) -> None:
    """Overwrite ``b_prev`` with this token's ``B``.

    A launch of its own because the row it writes is read by every block of
    :func:`decode_fwd_kernel` whose head maps to the same group, and a grid has no
    barrier to order that read against this write. See the module docstring.

    Args:
        gb: ``(B,G,TOKENS,3N)`` activations. Contiguous or one pitched band.
        gbprev: ``(B,G,3N)`` activations, contiguous, overwritten.
        width: ``3N``. Dynamic.
        threads: Block width. Compile-time.

    Invariants:
        One block per ``(b,g)``, so the walk over ``3N`` is a stride loop and no
        element is written twice. Element for element, so no dtype conversion.
    """
    tid, _, _ = cute.arch.thread_idx()
    _, gidx, bidx = cute.arch.block_idx()
    for index in cutlass.range(tid, width, threads):
        gbprev[bidx, gidx, index] = gb[bidx, gidx, 0, index]


@cute.jit
def decode_carry(
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    width: cutlass.Int32,
    groups: cutlass.Int32,
    bsz: cutlass.Int32,
    stream: Stream,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`decode_carry_kernel` over ``(1, G, B)``."""
    decode_carry_kernel(gb, gbprev, width, threads).launch(
        grid=(1, groups, bsz), block=(threads, 1, 1), stream=stream
    )


def decode_forward(
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    /,
    *,
    ssm: Tensor,
    b_prev: Tensor,
    u_prev: Tensor,
) -> Tensor:
    """Step the recurrence one token and advance the state in place.

    Args:
        U: Input weights, ``(B,H,TOKENS,P)``, activation dtype.
        trans: ``(w_x, w_y, w_z, ls)``, ``(B,H,TOKENS,4)``, pinned.
        K: Per-tap ``(kr, g, h, 0)``, ``(B,H,TOKENS,2,4)``, pinned.
        B: Input vectors, ``(B,G,TOKENS,3N)``, activation dtype. Contiguous or one
            pitched band of a wider tensor.
        C: Output vectors, ``(B,G,TOKENS,3N)``, activation dtype. Like ``B``.
        ssm: Recurrent state, ``(B,H,P,3N)``, float32, contiguous. Read, then
            overwritten with the state after this token.
        b_prev: ``b`` at the previous token, ``(B,G,3N)``, dtype of ``B``,
            contiguous. Read, then overwritten with ``B[:, :, 0]``.
        u_prev: ``u`` at the previous token, ``(B,H,P)``, dtype of ``U``,
            contiguous. Read, then overwritten with ``U[:, :, 0]``.

    Returns:
        ``y``, shape ``(B,H,TOKENS,P)``, dtype of ``U``, contiguous.

    Raises:
        ValueError: On a rank, token-extent, shape, shape-multiple, contiguity,
            pitch, device, state-dtype, or storage-sharing violation.
        TypeError: On an operand dtype with no kernel path, or on activations that do
            not share one dtype. The reference path takes both.
    """
    shapes = check_operands(U, trans, K, B, C, ssm, b_prev, u_prev)
    # One activation dtype for the whole call, which the shared contract does not
    # demand: it pairs each carry with its own operand and leaves ``U`` and ``B``
    # free to disagree. A kernel widens on load from one element type per tensor, so
    # a disagreement here would be two widening types in one body for no case that
    # produces one.
    check_dtypes(
        (
            (U, "U"),
            (B, "B"),
            (C, "C"),
            (b_prev, "b_prev"),
            (u_prev, "u_prev"),
        ),
        KERNEL_DTYPES,
        "kernel dtypes",
    )
    group = row_group(shapes.lanes)
    y = torch.empty(
        shapes.bsz,
        shapes.heads,
        TOKENS,
        shapes.rows,
        dtype=U.dtype,
        device=U.device,
    )
    jit_launch(
        decode_fwd,
        (
            U,
            trans,
            K,
            B,
            C,
            ssm,
            b_prev,
            u_prev,
            y,
            shapes.heads // shapes.groups,
            shapes.rows // rows_per_block(shapes.lanes),
            shapes.heads,
            shapes.bsz,
        ),
        (THREADS, group, shapes.lanes // group),
    )
    # After every read of it, and in its own launch: see the module docstring.
    jit_launch(
        decode_carry,
        (B, b_prev, shapes.state_dim, shapes.groups, shapes.bsz),
        (THREADS,),
    )
    return y
