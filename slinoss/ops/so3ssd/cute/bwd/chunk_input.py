"""``dU``, the input carry, and the log-scale and chunk-transition cotangents.

Everything the backward owes the forcing input and the closing transition of a
chunk, in one block per ``(chunk, batch, head)``. The reference terms are ``du``,
``dushift``, ``dexpo``, ``duw``, ``dupw``, ``dexpw``, ``dchunk_rot`` and
``dchunk_scale``.

Five contractions, all of them dense real GEMMs off the one atom, all indexed by
the source token first:

    D_tap(r,d)   = sum_p  u_tap(r,p) dinc_local(p,d)
    duw_tap(r,p) = sum_d  b_tap(r,d) dinc_local(p,d)
    score(r,t)   = sum_d  b_tap(r,d) crot(t,d)
    dm(r,t)      = sum_p  u_tap(r,p) dy(t,p)
    ddiag(r,p)   = sum_t  Smasked(r,t) dy(t,p)

with ``u_tap`` and ``b_tap`` the current tap at ``t`` and the previous tap at
``t-1``, ``Smasked = score * dmask``, and ``dinc_local = Ac_{L-1} dinc`` the
increment cotangent carried back into the chunk-local frame. ``ddiag`` and ``duw``
accumulate into one pair of fragments, so the current tap's pair is ``du`` and the
previous tap's is ``dushift``.

The score is built transposed relative to the forward's. That is what makes the
masked score the left operand of the diagonal GEMM with no shared-memory round
trip: its N mode is the target token and the diagonal GEMM contracts over the
target token, so :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` rereads the fragment
in place. It removes a score tile, one ``ldmatrix`` and one barrier per slice, and
it is what lets the two ``B`` passes below fuse. ``dm`` is transposed with it,
because the two are multiplied elementwise.

The cost of the transpose is the direction of the ``dexpo`` reduction. Summed over
the source token it is now a reduction over the accumulator's M mode, whose rows
are disjoint across warps, so it goes through one scratch row per warp and is
summed over warps at the store. Over the N mode it would have been two shuffles
and no scratch.

One pass over ``B`` per tap. Every contraction that reads a tap's forcing vector
runs while that tap is staged, so ``B`` is read twice rather than four times: 19.0
MB against 38.0 MB at ``standard``, on a kernel whose floor is its traffic. The
price is that the increment cotangent, the readout and the output cotangent are
live at once. The block is 42,832 B at ``standard`` and 48,752 B at every ``3N``
once :func:`lblock` slices the lane extent, and the carveout holds either two deep;
both read off the device as ``launch__shared_mem_per_block_dynamic`` on sm_86.

``dU`` is ``du(t) + dushift(t+1)``, the second term one row behind, which is why
``dushift`` goes through a float32 tile before the store. The chunk's last valid
token gets ``du`` alone: its ``dushift`` partner lives in the next chunk and
:func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds it there.
``carry_u`` is row 0 of ``dushift``, the cotangent that crosses into the previous
chunk.

The log-scale term is
``dlogp(t) = 2 * (sum_r dexpo(r,t) - <u(t),du(t)> - <ushift(t),dushift(t)>)``
plus ``2 * sum_r dexpw(r)`` at the chunk's last slot, which is what the reference's
``_scatter_last`` writes and holds whether or not that slot carries a token. The
sum of ``dexpo`` over the target token is never formed: it equals the two inner
products against the finished fragments, so it costs one lane reduction rather than
a second pass over the score.

The chunk-transition pair factors through the same frame change. With
``X = chunk_scale * zstart + inc_local``,

    dchunk_rot = R(Q_{L-1}) sum_{p,n} outer(dinc_local(p,n), X(p,n))

because ``dinc = R dinc_local`` and ``R`` comes out of the sum, so the increment
never has to be rematerialized: its half of the outer product is
``sum_r wgt(r) sum_n D_tap(r,3n+i) b_tap(r,3n+j)``, read off the ``D`` fragment in
its epilogue. ``dchunk_scale`` is ``sum <dinc_local, zstart>`` for the same reason.
Both are eleven float32 block reductions and ten stores per block.

Shared memory is one resident set and one phase arena. Resident: the log-scale
prefix, the increment weight, the per-warp log-scale scratch, the reduction
scratch, the three-slot transform table, and the shifted ``U`` tile that serves
both taps. The arena holds the forcing tile, restaged per tap, then the increment
cotangent, the readout and the output cotangent, staged once; the ``trans``, ``K``
and quaternion tiles of the prologue and the float32 shift tile of the epilogue
alias the last three, neither being live when the other is.

DRAM-bound. Analytic traffic at ``standard`` is ``dy 9.44 + U 9.58 + trans 1.57 +
K 3.15 + B 19.02 + C 9.44 + dinc 7.08 + zstart 7.08 + dU 9.44 + carry_u 0.29 +
dlogp 0.39 + dchunk_rot 0.06 + dchunk_scale 0.01 = 76.54 MB`` against ``1536 *
3.54 MFLOP = 5.44 GFLOP``, so 71.1 flop/byte against a ridge point of 164: memory
bound by a factor of over two. ``dinc`` and ``zstart`` are two of the three reads
that are not activations and the only two at the operand width rather than float32:
each is a state a recurrence stored, and this kernel and ``chunk_vector_bwd`` are
its only readers. A supplied ``du_init`` adds one read of 9.44 MB at that shape,
against the 28.32 MB a caller-side add of the same tensor would cost.

Every measurement in the rest of this docstring was taken with those two at float32.
Narrowing them is -95.9 us on this launch at the acceptance shape, 861.9 to 766.0,
with the measured read falling 484.33 MB to 257.79 MB: 226.54 MB against the 141.56
the two planes account for, because a float32 plane at this kernel's lane blocking
also costs sectors the narrow one does not. Local load and store traffic is zero on
both sides, so none of it is spill.

The class is met at ``3N = 240 H = 18`` and at ``wide``, and not at the other three
shapes. Measured on an RTX A6000, sm_86, clocks not locked, 2026-08-21, one launch per
profile, against a copy time law fitted in the same process at the same clocks: fixed
cost 4.17 to 5.01 us, asymptote 683.0 to 684.6 GB/s, worst residual 2.13%. Durations
are ``gpu__time_duration.sum``, and the device probe names the part by UUID. Every
profile in the table below ran with foreign processes resident on the device in both
brackets, so every duration is stamped rather than quoted. The per-launch counters are
what the verdicts rest on: they are deterministic in the launch, and the sector counts
repeat exactly across profiles where the wall does not.

    shape         blocks  us/launch      MB  GB/s  class  dominant stall
    standard        1536      235.1  101.46   432  65.1%  long_sb    27.6%
    ragged          1536      231.1  100.19   434  65.6%  long_sb    26.8%
    wide            1536      449.7  259.27   577  85.3%  long_sb    38.8%
    long            1536     1653.6  181.19   110  16.3%  no_instr   52.9%
    3N=240 H=18     2304      871.3  506.24   581  85.6%  long_sb    20.6%

``class`` is the fitted floor over the duration, against a bar of 85%. It falls when a
change removes bytes, because the floor falls with them; the duration is the figure to
read across the rows. The last row is the acceptance shape,
``B=4 H=18 T=2048 P=64 3N=240 L=64`` at one group: 85.6% of the floor,
``dram__throughput`` 80.6%, 28.21% issue, 254 registers, 48,752 B of shared memory,
16.7% theoretical occupancy against 16.4% achieved, 0.1236 shared bank conflicts per
wavefront, and SOL 56.3% sm against 80.9% memory.

Six changes took that shape from 3528.2 us and 1700.54 MB, the before figure taken
with the device to itself in both brackets. Traffic after each, at that shape:

- The two forcing-cotangent GEMMs accumulate into the output fragment. The increment
  weight is a function of the source token alone, so weighting the finished sum over
  lane blocks is the same sum as weighting each block's part, and the accumulator the
  weighted part used to land in is gone. 1700.54 MB to 978.67 MB.
- The rotation cotangent's increment half accumulates in the thread's own rotated
  basis. Its matrix row is the accumulator column modulo three, which is dynamic in
  the component basis and trace-time in the rotated one, because the column offsets
  are trace-time and the thread's base column is common to the fragment. One nine-slot
  roll before the block reduction undoes it. 978.67 MB to 862.73 MB.
- The head is the fastest grid mode, so the blocks that share a group's forcing and
  readout rows are consecutive and co-resident. At one group that turns eighteen reads
  of the same chunk of ``B`` and ``C`` into one read and seventeen L2 hits. 862.73 MB
  to 618.11 MB.
- The lane loop is rolled, ``cutlass.range(ltiles, unroll=1)``. Unrolled five deep the
  kernel was instruction-fetch starved, ``no_instruction`` 48.3% at a 9.21% issue rate;
  rolled it issues at 24.31% and the dominant stall is ``long_scoreboard`` 29.2% with
  ``mio_throttle`` 15.2%, which is one lane block's global staging latency exposed once
  per trip. It costs traffic, 618.11 MB to 631.20 MB, and it is the step that moved the
  issue rate. A rolled loop that stages and transforms what it loads is what
  ``docs/kernels.md`` warns against, so that rule holds only while the unrolled body
  still fits the instruction cache.
- The increment staging pass indexes its row affinely in the step rather than through a
  division of the flat index. 631.20 MB to 574.89 MB, and 1058.4 us to 964.4 us. It is
  the only one of the five that shortened ``long`` as well, 1891.7 us to 1762.3 us,
  and the only one that cost a shape: ``wide`` went 505.6 us to 524.4 us on a different
  specialization, ``G != H``, whose spill loads rose 2,064,384 to 2,752,512 while every
  other shape's fell.
- The increment's outer-product GEMM and its ``mrotp`` epilogue run below the forcing
  GEMM instead of above it. Issued first, that accumulator is live across every other
  GEMM in the body; issued last it dies inside its own statement group, and the
  allocation that spilled at the acceptance shape stops spilling there. 584.10 MB to
  506.24 MB, and 962.5 us to 871.3 us, both brackets in one session. The pre-arm
  bracket reads 9.21 MB above the 574.89 MB the fifth change recorded, which is the
  294,912 extra spill sectors -- eight words per thread -- that this toolchain
  allocates and the earlier one did not. ``wide`` fell with it, 507.2 us to 449.7 us
  as its spill went 103.02 MB to 66.06 MB. ``standard`` and ``ragged`` did not move,
  and neither did their spill. ``long`` rose 0.8% on an unchanged 4,030,464 sectors
  that land in L1 less often, 1,134,600 hits to 905,528.

Registers sit at the 255 architectural cap at every shape and the kernel spills at
every shape: 761,856 sectors per launch each way at ``standard`` and ``ragged``,
2,752,512 and 958,464 at ``wide``, 1,966,080 each way at ``long``, 2,985,984 and
995,328 at ``3N = 240 H = 18``. L1 absorbs 3.4% of the spill loads and none of the
stores at the acceptance shape, so the spill is device traffic: 127.40 MB of the
574.89 MB moved there. The carveout is why. Two resident blocks of 48,752 B leave
under 30 KB of the 128 KB unified cache for data, against their own streaming global
staging, so a local line does not survive to its reload.

That paragraph is the state before the sixth change, and the acceptance shape has
left it: 254 registers, no local sector either way, 506.24 MB. The spill survives
everywhere else, per launch each way at the same date as the table above: 540,672 at
``standard`` and ``ragged``, 1,376,256 and 688,128 at ``wide``, 2,015,232 at ``long``.
So the cap is not what the spill tracks. A live range is: the increment accumulator
is the only one in the lane-loop body that dies inside the body, and where the body's
order lets it die the allocation fits.

What the spill holds is loop-invariant address arithmetic, not the accumulators. Read
off the SASS at the acceptance shape before the sixth change, the frame was 160 B and
both rolled lane loops carried no store at all: every ``STL`` was in the prologue or
between the taps, and the ``LDL`` cluster sat at the top of each loop body. At 36,864
sectors per launch per word per thread that made 167 words loaded and 40 stored, of
which 14 were reloaded on each of the ten trips: the eight per-step element offsets of
the increment staging pass and three 64-bit base pointers. Eliminating the eight left
81 loaded and 27 stored, about five words per trip. The three pointers survive slicing
the two state tensors to the chunk's plane, so what remained was not addressed by
removing coordinates, and what removed it was the live range and not the addresses.

Block width is a bound at some shapes and not at others. At four warps occupancy is
16.7% theoretical against 16.3
to 16.5% achieved at ``L = 64``, with ``launch__occupancy_limit_registers`` and
``launch__occupancy_limit_shared_mem`` both two; ``long`` gets one block, 8.3% against
8.3%, its shared-memory limit reading one, because a 128-row M tile puts its arena at
79,952 B and the lane block is the only lever :func:`lblock` has. Eight warps widen the
tiling in N, not in M, so the chunk's row count is untouched and a thread's accumulator
halves. What that costs is the diagonal GEMM's rereadable left operand, which the wide
form stages through shared memory instead.

``long`` is the one shape whose bound is still instruction fetch: ``no_instruction``
52.9% at a 7.96% issue rate and 15.0% of ``dram__throughput``, with one lane tile and
four target-token slices unrolled over a 128-row M tile. The lane loop's remedy applies
to the slice loop unchanged, and is not taken here because the score bank is a list of
fragments a trace-time index addresses, which a dynamic trip count cannot. ``standard``
and ``ragged`` are latency-bound instead, ``long_scoreboard`` 27.6% and 26.8% at issue
rates of 27.42% and 27.48%, with one lane tile and 34.60 MB of their 101.46 and 100.19
MB in the spill.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Stream,
    Tile,
    assert_smem_fits,
    block_reduce_add,
    cute_dtype,
    decay,
    jit_launch,
    narrow,
    select,
    shuffle_xor,
    smem_bytes,
    smem_capacity,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    THREADS,
    WARPS,
    mat3_matvec,
    mat3_mul,
    mat3_transpose,
    scalar_tile,
    table_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_extents,
    check_layout,
    check_operands,
    check_pinned,
    check_pitched,
    check_rows,
    check_shapes,
    check_stored,
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_TILE_K,
    SMEM_SEGMENT,
    fp32_tile,
    make_mma,
    mma_acc,
    mma_areg,
    mma_coords,
    mma_gemm,
    mma_gemm_areg,
    mma_offsets,
    mma_rows,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    PREFETCH,
    build_table,
    stage_chunk,
    stage_rotated,
    stage_shifted,
)
from slinoss.ops.so3ssd.reference import check_grad_band

__all__ = [
    "LANE_MULTIPLE",
    "REDUCTIONS",
    "RESIDENT_MAX",
    "RESIDENT_MIN",
    "TBLOCK_MAX",
    "Arena",
    "ChunkInputBwd",
    "arena",
    "chunk_input_backward",
    "chunk_input_bwd",
    "chunk_input_bwd_kernel",
    "cotangent_tile",
    "forced_tile",
    "input_smem_bytes",
    "input_tile",
    "lblock",
    "local_tile",
    "reduce_tile",
    "shift_tile",
    "tblock",
    "warp_tile",
]

TBLOCK_MAX: int = 32
"""Target-token columns of the score, of ``dm`` and of the narrowed score at once.

Three fragments of ``(mma_rows(L), TBLOCK_MAX)``, two float32 and one narrowed: at
``standard`` 16, 16 and 8 registers per thread against two 24-register output
fragments. It bounds the score itself only while the lane extent is unsliced. Sliced,
the lane extent is a K mode the score accumulates over, so every slice is live until
the last lane block and the score is the whole ``(mma_rows(L), L)`` matrix whatever
this is -- 32 registers at ``L = 64``. A multiple of 16, which
:func:`slinoss.ops.so3ssd.cute.mma.mma_areg` requires of the N extent it rereads as
K."""

LANE_MULTIPLE: int = 48
"""Divisor every lane block is a multiple of.

16 divides it because the lane extent is the N mode of the increment's second
product and the K mode of two other contractions, and 3 divides it because the frame
change and the rotation cotangent both work on whole lane triples. ``3N`` is a
multiple of both for the same reasons, so this always divides ``3N``."""

RESIDENT_MIN: int = 2
"""Blocks per SM :func:`lblock` sizes the lane block for.

Two, because one block of 128 threads reaches 8.3% of the device's warp slots and no
amount of latency hiding recovers a kernel that is DRAM-latency-bound at that
occupancy. The lane block is the only lever on the total: every other tile is fixed
by ``L`` and ``P``."""

REDUCTIONS: int = 11
"""Float32 block reductions the epilogue pays: nine for the chunk-rotation
cotangent, one for the chunk-scale cotangent, one for the increment weight's
log-scale term. One scratch row each, so no ordering between them and no barrier of
the caller's own."""

RESIDENT_MAX: int = 3
"""Blocks per SM the launch asks for, before the shared-memory budget lowers it.

The budget lowers it to two at ``L = 64`` and to one at ``L = 128``, and two is also
what the register file allows: at four warps the kernel sits at the 255-register
architectural cap and spills, and at eight it lands at 128 registers over twice the
threads, which is the same 32,768 registers a block.
``launch__occupancy_limit_registers`` reads two on sm_86 at every shape and both widths
profiled, so a third block is unreachable whatever this asks for. The
cap is not derived from shared memory alone because a shape whose arena is small
enough for three blocks still would not get them."""


def tblock(chunk: int) -> int:
    """Target-token slice width, ``min(L, TBLOCK_MAX)``.

    Args:
        chunk: ``L``.
    """
    return min(chunk, TBLOCK_MAX)


def lblock(
    chunk: int, rows: int, dim: int, itemsize: int = 2, *, warps: int = WARPS
) -> int:
    """Lane extent held in shared memory at once.

    The three tiles that carry the lane dimension are the rotated forcing vectors,
    the increment cotangent and the rotated readout. Together they are the whole
    lane-dependent part of the block, and at ``3N = 240`` they are 95,232 B of a
    101,376 B carveout, so the block does not fit at all. Slicing them bounds the
    footprint by the lane block instead of by ``3N``: at ``L = 64``, ``P = 64`` the
    block is 48,752 B at every ``3N``, against 104,048 B at ``3N = 192`` and
    122,480 B at ``3N = 240`` unsliced.

    What the slicing costs is the contraction structure, not traffic. The lane extent
    is a K mode for the forcing cotangent and the score, so both accumulate across
    blocks; the score's accumulator therefore stays live over the whole lane loop.
    The tap loop is outside the lane loop, which is what holds that to one score
    rather than two, and the price is that the increment cotangent and the readout are
    staged once per tap when there is more than one lane block.

    What slicing buys is shared bytes and the occupancy they decide: 48,752 B at
    ``wide``, which reads 16.5% achieved occupancy and an
    ``launch__occupancy_limit_shared_mem`` of two, against 67,184 B unsliced, which one
    128-thread block per SM holds to 8.3%; and at ``3N = 240`` a kernel that launches at
    all against 122,480 B that does not. The pairwise durations this docstring used to
    carry, sliced against unsliced at each shape, were taken through a device probe that
    named the part by torch ordinal rather than by driver index, so which of them ran
    uncontended is not established and none is quoted. They are not retaken: the sliced
    form is the only one that reaches the acceptance shape.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        warps: Warps per block, which the two per-warp scratch tiles carry.

    Returns:
        The widest divisor of ``3N`` that is a multiple of :data:`LANE_MULTIPLE` and
        whose block fits :data:`RESIDENT_MIN` times in the device's carveout, or the
        widest that fits once when none does, or :data:`LANE_MULTIPLE` when even that
        does not fit. The last case is what
        :func:`slinoss.ops.so3ssd.cute.guard.assert_smem_fits` reports on.
    """
    legal = [blk for blk in range(dim, 0, -LANE_MULTIPLE) if dim % blk == 0]
    capacity = smem_capacity()
    for budget in (capacity // RESIDENT_MIN, capacity):
        for blk in legal:
            if (
                input_smem_bytes(chunk, rows, dim, itemsize, lblk=blk, warps=warps)
                <= budget
            ):
                return blk
    return legal[-1]


def input_tile(chunk: int, rows: int) -> Tile:
    """Shifted ``U`` tile, ``(mma_rows(L) + 1, pitch)``.

    Row ``j`` holds token ``t0 + j - 1``, so the previous tap reads rows
    ``0..mma_rows(L)-1`` and the current tap the same rows one further on. Both are
    M modes of a GEMM, hence the rounded row count, and the extra row is what the
    current tap's rounded extent needs.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk) + 1, rows)


def forced_tile(chunk: int, lblk: int) -> Tile:
    """Rotated forcing or readout tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        lblk: Lane extent, from :func:`lblock`.
    """
    return operand_tile(mma_rows(chunk), lblk)


def local_tile(rows: int, lblk: int) -> Tile:
    """Increment cotangent tile in the chunk-local frame, ``(P, pitch)``.

    ``P`` is an N mode of one GEMM and a K mode of the other, never an M mode, so
    the row count is ``P`` itself. Both uses need ``P`` to be a multiple of
    :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, which
    :func:`slinoss.ops.so3ssd.cute.guard.check_rows` enforces.

    Args:
        rows: ``P``.
        lblk: Lane extent, from :func:`lblock`.
    """
    return operand_tile(rows, lblk)


def cotangent_tile(chunk: int, rows: int) -> Tile:
    """Output cotangent tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk), rows)


def shift_tile(chunk: int, rows: int) -> Tile:
    """Float32 tile the shifted ``dushift`` passes through, ``(mma_rows(L), pitch)``.

    Float32 because it is not an operand. It is a gradient on its way to ``dU``, and
    the reference rounds that once, at the store; a second rounding here would
    double the error on the shifted half of every row.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return fp32_tile(mma_rows(chunk), rows)


def warp_tile(chunk: int, warps: int = WARPS) -> Tile:
    """Per-warp log-scale scratch, ``(warps, pitch)``.

    One row per warp because the ``dexpo`` reduction is over the accumulator's M
    mode, whose rows are split across warps: a single row would be four warps
    reading and writing one address.

    Args:
        chunk: ``L``.
        warps: Warps per block. :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`
            doubles the rows, which is the only shared cost of the wide form.
    """
    return Tile((warps, chunk), (smem_pitch(chunk, 4), 1))


def reduce_tile(warps: int = WARPS) -> Tile:
    """Block-reduction scratch, ``(REDUCTIONS, warps)``.

    One word per warp per reduced value, which is what lets the epilogue reduce
    eleven float32 with no barrier of its own between them.

    Args:
        warps: Warps per block.
    """
    return Tile((REDUCTIONS, warps), (warps, 1))


class Arena(NamedTuple):
    """Float32-word offsets of the phase-shared tiles inside the one arena.

    The tiles below overlap in address and not in time. The forcing tile, the
    increment cotangent, the readout and the output cotangent are live together
    through the lane loop and are laid out end to end; the prologue's staging tiles
    and the epilogue's shift tile alias the last three. The first three hold one lane
    block, not ``3N``, which is what bounds the whole allocation.

    Attributes:
        forced: The rotated forcing tile, restaged once per lane block per tap.
        local: The increment cotangent in the chunk-local frame, one lane block.
        readout: The rotated readout, one lane block.
        cotangent: The output cotangent.
        shift: The float32 shift tile. Epilogue only.
        trans: ``trans`` staging. Prologue only.
        tap: ``K`` staging. Prologue only.
        quat: Quaternion prefix staging. Prologue only.
        score: The wide form's staged score, below the lane loop. Equal to
            :attr:`local` where the increment cotangent is restaged every tap, and
            its own words where it is not.
        words: Float32 words the arena spans.
    """

    forced: int
    local: int
    readout: int
    cotangent: int
    shift: int
    trans: int
    tap: int
    quat: int
    score: int
    words: int


def _words(tile: Tile, itemsize: int) -> int:
    """Float32 words a tile of ``itemsize``-byte elements spans.

    Exact at every legal shape: an operand pitch is an odd multiple of eight
    elements and a float32 pitch an odd multiple of four, so both spans are a whole
    number of float32 words and every offset below is 16-byte aligned.

    Args:
        tile: The tile.
        itemsize: Bytes per element.
    """
    return itemsize * tile.words // 4


def arena(
    chunk: int,
    rows: int,
    dim: int,
    itemsize: int = 2,
    *,
    lblk: int | None = None,
    warps: int = WARPS,
) -> Arena:
    """Lay the phase-shared tiles out in one allocation.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        lblk: Lane extent of the three lane-dependent tiles. Defaults to
            :func:`lblock`, which passes it explicitly to ask what a candidate would
            cost.
        warps: Warps per block. Above :data:`slinoss.ops.so3ssd.cute.mma.WARPS` the
            diagonal GEMM needs its left operand in shared memory.
    """
    if lblk is None:
        lblk = lblock(chunk, rows, dim, itemsize)
    forced = _words(forced_tile(chunk, lblk), itemsize)
    local = _words(local_tile(rows, lblk), itemsize)
    readout = forced
    cotangent = _words(cotangent_tile(chunk, rows), itemsize)
    tail = max(
        local + readout + cotangent,
        _words(shift_tile(chunk, rows), 4),
        16 * chunk,
    )
    # The wide form's diagonal GEMM stages its left operand below the lane loop. At
    # more than one lane block the increment cotangent is restaged every tap, so the
    # staging aliases it and costs nothing; at one block it is staged on the first tap
    # and read on the second, so the staging takes its own words.
    staged = (
        _words(operand_tile(mma_rows(chunk), tblock(chunk)), itemsize)
        if warps > WARPS
        else 0
    )
    if staged and dim // lblk == 1:
        score = forced + tail
        tail = tail + staged
    else:
        score = forced
        tail = max(tail, staged)
    return Arena(
        forced=0,
        local=forced,
        readout=forced + local,
        cotangent=forced + local + readout,
        shift=forced,
        trans=forced,
        tap=forced + 4 * chunk,
        quat=forced + 12 * chunk,
        score=score,
        words=forced + tail,
    )


def input_smem_bytes(
    chunk: int,
    rows: int,
    dim: int,
    itemsize: int = 2,
    *,
    lblk: int | None = None,
    warps: int = WARPS,
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_input_bwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        lblk: Lane extent of the three lane-dependent tiles. Defaults to
            :func:`lblock`, which passes it explicitly to ask what a candidate would
            cost.
        warps: Warps per block. Only the two per-warp scratch tiles carry it.
    """
    words = arena(chunk, rows, dim, itemsize, lblk=lblk, warps=warps).words
    return smem_bytes(
        [
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (warp_tile(chunk, warps), 4),
            (reduce_tile(warps), 4),
            (table_tile(chunk, 3), 4),
            (input_tile(chunk, rows), itemsize),
            (Tile((words,), (1,)), 4),
        ]
    )


def _tile_at(base: cute.Tensor, words: int, tile: Tile, elem: object) -> cute.Tensor:
    """One arena tile, at a float32-word offset and possibly a narrower element.

    Undecorated, so the branch is taken during the trace and no recast reaches the
    IR for a float32 view.

    Args:
        base: The float32 arena tensor.
        words: Float32-word offset, from :func:`arena`.
        tile: Layout to build at that offset.
        elem: Element type of the view.
    """
    ptr = base.iterator + words
    if elem is not cutlass.Float32:
        ptr = cute.recast_ptr(ptr, dtype=elem)
    return cute.make_tensor(ptr, tile.layout())


def _lane_slice(tensor: cute.Tensor, lbase: int) -> cute.Tensor:
    """One lane block of a ``(...,3N)`` source, as a move of the lane origin.

    The stagers address the lane mode by a pair index bounded by the lane count they
    are handed, never by the source's own extent, so moving the origin is all a lane
    block needs. Undecorated, so the offset folds into the trace.

    Args:
        tensor: Global source with unit stride on its lane mode.
        lbase: First lane element of the block, a multiple of :data:`LANE_MULTIPLE`.
            Every alignment the stagers restate on the iterator survives it, because
            48 elements of either operand dtype is a whole number of 16-byte segments.
    """
    return cute.make_tensor(tensor.iterator + lbase, tensor.layout)


def _sum_over_n(value: cutlass.Float32) -> cutlass.Float32:
    """Sum one accumulator element over the four lanes that share its row.

    The atom gives the four lanes of an aligned quad the same accumulator row and
    disjoint columns, so two butterfly rounds leave that row's partial column sum in
    all four. Rows are disjoint across quads and across warps, so the read-modify-
    write that follows is by one thread per row and needs no barrier.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 1)
    return value + shuffle_xor(value, 2)


def _sum_over_m(value: cutlass.Float32) -> cutlass.Float32:
    """Sum one accumulator element over the eight lanes that share its column.

    The atom gives lanes ``l``, ``l^4``, ``l^8`` and ``l^16`` the same accumulator
    column and disjoint rows, so three butterfly rounds leave that column's
    within-warp row sum in all eight. Columns are shared across warps, so the caller
    reduces over warps as well, through one scratch row each.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 4)
    value = value + shuffle_xor(value, 8)
    return value + shuffle_xor(value, 16)


@cute.kernel
def chunk_input_bwd_kernel(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gduinit: cute.Tensor,
    gdu: cute.Tensor,
    gcarry: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    seqlen: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    lblk: cutlass.Constexpr,
    tblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    has_seed: cutlass.Constexpr,
) -> None:
    """Differentiate one chunk's forcing input and closing transition.

    One block per ``(chunk, batch, head)``.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gu: ``(B,H,T,P)`` operand-dtype forcing input.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 ``(kr, g, h, 0)`` per tap.
        gb: ``(B,G,T,3N)`` operand-dtype forcing vectors.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gdinc: ``(B,H,C,P,3N)`` operand-dtype increment cotangent, global frame.
        gz: ``(B,H,C,P,3N)`` operand-dtype chunk-start state.
        gduinit: ``(B,H,T,P)`` operand-dtype addend for ``dU``. Read only when
            ``has_seed``.
        gdu: ``(B,H,T,P)`` operand-dtype, written with ``dU`` except at the chunk's
            last valid token, which gets the diagonal term alone.
        gcarry: ``(B,H,C,P)`` float32, written with row 0 of ``dushift``.
        gdlp: ``(B,H,C,L)`` float32, written with the diagonal and increment half of
            the log-scale-prefix cotangent, every slot including the padded ones.
        gdrot: ``(B,H,C,3,3)`` float32, written with the closing rotation cotangent.
        gdscale: ``(B,H,C)`` float32, written with the closing scale cotangent.
        seqlen: ``T``. Dynamic.
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        lblk: Lane block, from :func:`lblock`. Compile-time.
        tblk: Target-token slice, from :func:`tblock`. Compile-time.
        per_group: ``H // G``, heads sharing one ``b`` and ``c``. Compile-time.
        has_prev: Whether the streaming carry-in pair was supplied. Compile-time.
        has_seed: Whether ``gduinit`` is an addend rather than a stand-in.
            Compile-time.

    Invariants:
        ``chunk``, ``dim``, ``rows``, ``lblk`` and ``tblk`` are multiples of the
        atom's extents, so no contraction mode is padded; only ``M``, a token count,
        is rounded, and its rows are zero-filled by the stagers. ``lblk`` divides
        ``dim`` and is a multiple of 3, so a lane block holds whole lane triples and
        the frame change never straddles one. The prefixes, the
        table, the increment weight and every reduction are float32 (I4). Both
        decays come from one exponential of a log difference (I3), the mask lands on
        the float32 accumulator before the one narrowing (I6), and the quaternion
        prefix is renormalized once inside :func:`chunk_prefixes` (I5).
    """
    tid, _, _ = cute.arch.thread_idx()
    hidx, bidx, cidx = cute.arch.block_idx()

    # Only gb and gc are grouped; everything else this block reads is per head. The
    # branch is trace-time, so the ungrouped shape emits no divide at all.
    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    lanes = lblk // 3
    ltiles = dim // lblk
    mpad = mma_rows(chunk)
    last = chunk - 1
    slices = chunk // tblk
    elem = gdy.element_type
    out = gdu.element_type
    # The two chunk-plane states are read by the epilogue below rather than by a
    # stager, so this pass widens them itself.
    dinc_elem = gdinc.element_type
    state_elem = gz.element_type
    zero = cutlass.Float32(0.0)

    ldu = smem_pitch(rows)
    ldv = smem_pitch(lblk)
    nwarps = threads // 32
    one_group = nwarps == WARPS
    where = arena(chunk, rows, dim, lblk=lblk, warps=nwarps)

    smem = cutlass.utils.SmemAllocator()
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(cutlass.Float32, warp_tile(chunk, nwarps).layout(), 16)
    sred = smem.allocate_tensor(cutlass.Float32, reduce_tile(nwarps).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 3).layout(), 16)
    su = smem.allocate_tensor(elem, input_tile(chunk, rows).layout(), SMEM_SEGMENT)
    pool = smem.allocate_tensor(
        cutlass.Float32, Tile((where.words,), (1,)).layout(), SMEM_SEGMENT
    )

    sbu = _tile_at(pool, where.forced, forced_tile(chunk, lblk), elem)
    sdinc = _tile_at(pool, where.local, local_tile(rows, lblk), elem)
    sc = _tile_at(pool, where.readout, forced_tile(chunk, lblk), elem)
    sdy = _tile_at(pool, where.cotangent, cotangent_tile(chunk, rows), elem)
    # The wide form only, where the score fragment cannot be reread as an A operand.
    # The view drops the pitch's padding, so the K mode is the stride-1 mode of an
    # ``(M,K)`` operand.
    sscore = _tile_at(pool, where.score, operand_tile(mpad, tblk), elem)
    vscore = cute.make_tensor(
        sscore.iterator, cute.make_layout((mpad, tblk), stride=(smem_pitch(tblk), 1))
    )
    sshift = _tile_at(pool, where.shift, shift_tile(chunk, rows), cutlass.Float32)
    strans = _tile_at(pool, where.trans, trans_tile(chunk), cutlass.Float32)
    stap = _tile_at(pool, where.tap, tap_tile(chunk), cutlass.Float32)
    squat = _tile_at(pool, where.quat, trans_tile(chunk), cutlass.Float32)

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)
    warp = tid // 32

    stage_chunk(
        gtrans[bidx, hidx, None, None],
        gtap[bidx, hidx, None, None, None],
        strans,
        stap,
        t0,
        valid,
        tid,
        threads,
        chunk,
    )
    for step in cutlass.range_constexpr(-(-(nwarps * chunk) // threads)):
        i = tid + step * threads
        if i < nwarps * chunk:
            sdlp[i // chunk, i - (i // chunk) * chunk] = zero

    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr(-(-chunk // threads)):
        token = tid + step * threads
        if token < chunk:
            # I3: one exponential of a log difference. The padded tokens carry the
            # identity transition, so the prefix flattens past valid and the last
            # slot's weight is the last token's weight.
            swgt[token] = decay(slp[last] - slp[token])
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
    cute.arch.sync_threads()

    # The closing transition, read once per block. Ac is R(Q)^T, so its transpose is
    # the rotation the chunk-transition cotangent is expressed in. Held in registers
    # across the whole kernel rather than reread from the table where it is used: the
    # reread measured 8,220,672 spill load sectors against 6,156,288 and 667.14 MB
    # against 631.20 at ``3N = 240 H = 18``, so nine fewer live registers cost more
    # spill elsewhere than they saved.
    # A plain range: the DSL preprocessor rewrites `range_constexpr` in a `for`
    # statement only, so inside a comprehension it reaches the runtime stub and
    # raises. Both unroll at trace time; only the statement form needs the marker.
    aclast = tuple(stable[TABLE_AC, last, i] for i in range(9))
    cscale = decay(slp[last])

    # The two tiles with no lane extent, staged once. Both passes issue their global
    # loads before either consumes one, so the reads overlap rather than serializing.
    stage_shifted(
        gu, guprev, su, bidx, hidx, t0, 0, valid, tid, threads, mpad, rows, has_prev
    )
    stage_shifted(
        gdy, gdy, sdy, bidx, hidx, t0, 1, valid, tid, threads, mpad - 1, rows, False
    )

    du = mma_acc(tiled_mma, tid, (mpad, rows))
    dushift = mma_acc(tiled_mma, tid, (mpad, rows))
    wcrd = mma_coords(tiled_mma, tid, (mpad, rows))
    # The lane extent is the score's K mode. Sliced, a slice's score is complete only
    # after the last lane block, so every slice is live at once and the whole
    # ``(mma_rows(L), L)`` score sits in registers: 32 per thread at ``standard``.
    # Unsliced it is complete where it is produced and one accumulator serves every
    # slice, which is 16. The banked form is taken only when it is needed.
    banked = ltiles > 1
    score = [
        mma_acc(tiled_mma, tid, (mpad, tblk)) for _ in range(slices if banked else 1)
    ]
    scrd = mma_coords(tiled_mma, tid, (mpad, tblk))
    dcrd = mma_coords(tiled_mma, tid, (mpad, lblk))
    # The rotation cotangent, in two sets. ``mrot`` is in the component basis, written
    # by the staging pass, whose component index is the loop's and static. ``mrotp`` is
    # in this thread's basis, rotated by ``lphase``, and is written by the increment
    # epilogue, whose component index is a residue of the accumulator column: dynamic
    # in the component basis, static in the rotated one, since the column offsets are
    # trace-time and the thread's base column is common to the fragment. Undone once
    # before the block reduction.
    doff = mma_offsets(tiled_mma, (mpad, lblk))
    dres = tuple((off[1] - doff[0][1]) % 3 for off in doff)
    lphase = dcrd[0][1] % 3
    # Fragments, not values: the lane loop is dynamic, and a fragment written across it
    # needs no loop-carried argument. Every index into both is trace-time, so they
    # promote to registers.
    mrot = cute.make_fragment((9,), cutlass.Float32)
    mrotp = cute.make_fragment((9,), cutlass.Float32)
    mrot.fill(0.0)
    mrotp.fill(0.0)
    dscale = zero
    dexpw = zero

    # Every operand is sliced along its contraction mode, one atom step per slice, and
    # the GEMM is called once per slice. A whole-K fragment is ``N*K/32`` elements per
    # thread, 48 registers of operand against a 24-register accumulator at ``P = 64``,
    # and it is live across the whole contraction; a slice's is 12 and dies at the next
    # one. The atom consumes K in :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_K` steps
    # either way, so the split changes the live range and not the instruction count, the
    # ldmatrix count or the accumulation order.
    # A plain range: a comprehension is not a `for` statement, so `range_constexpr`
    # would reach the runtime stub. Views, so this is layout and no storage.
    ksteps = rows // MMA_TILE_K
    lsteps = lblk // MMA_TILE_K
    vlocal_k = [
        cute.make_tensor(
            sdinc.iterator + k * MMA_TILE_K * ldv,
            cute.make_layout((lblk, MMA_TILE_K), stride=(1, ldv)),
        )
        for k in range(ksteps)
    ]
    vlocal_n = [
        cute.make_tensor(
            sdinc.iterator + k * MMA_TILE_K,
            cute.make_layout((rows, MMA_TILE_K), stride=(ldv, 1)),
        )
        for k in range(lsteps)
    ]
    vforced = [
        cute.make_tensor(
            sbu.iterator + k * MMA_TILE_K,
            cute.make_layout((mpad, MMA_TILE_K), stride=(ldv, 1)),
        )
        for k in range(lsteps)
    ]
    vreadout = [
        [
            cute.make_tensor(
                sc.iterator + s * tblk * ldv + k * MMA_TILE_K,
                cute.make_layout((tblk, MMA_TILE_K), stride=(ldv, 1)),
            )
            for k in range(lsteps)
        ]
        for s in range(slices)
    ]

    # The chunk's plane of both stored state tensors, sliced once, as every other
    # stager here takes its argument. The five-dimensional form measured three 64-bit
    # local slots reloaded at the top of every lane-block trip, but slicing removed
    # neither: the counters were sector-identical, 6,156,288 loads either way at
    # ``3N = 240 H = 18``. The leading coordinates were already folded. Kept for the one
    # address expression, not for a saving.
    pdinc = gdinc[bidx, hidx, cidx, None, None]
    pz = gz[bidx, hidx, cidx, None, None]

    # The two taps differ by the table slot, by which token the forcing vector comes
    # from, and by which row of the shifted tile pairs with an output row. The tap loop
    # is outside the lane loop because the score's K mode is the lane extent: one score
    # per slice serves both taps this way, two would be needed the other way round. The
    # price is that the increment cotangent and the readout are restaged per tap
    # whenever there is more than one lane block, and at one block they are not.
    for tap in cutlass.range_constexpr(2):
        vu = [
            cute.make_tensor(
                su.iterator + tap * ldu + k * MMA_TILE_K,
                cute.make_layout((mpad, MMA_TILE_K), stride=(ldu, 1)),
            )
            for k in range(ksteps)
        ]
        target = dushift if tap == 0 else du
        restage = tap == 0 or ltiles > 1
        if cutlass.const_expr(banked):
            for s in cutlass.range_constexpr(slices):
                score[s].fill(0.0)

        for lt in cutlass.range(ltiles, unroll=1):
            l0 = lt * lblk
            cute.arch.sync_threads()
            stage_rotated(
                _lane_slice(gb, l0),
                _lane_slice(gbprev, l0),
                sbu,
                stable,
                swgt,
                bidx,
                gidx,
                t0,
                0,
                valid,
                tid,
                TABLE_AP if tap == 0 else TABLE_AN,
                1 - tap,
                threads,
                mpad,
                lanes,
                has_prev,
                False,
            )
            if cutlass.const_expr(restage):
                stage_rotated(
                    _lane_slice(gc, l0),
                    _lane_slice(gc, l0),
                    sc,
                    stable,
                    swgt,
                    bidx,
                    gidx,
                    t0,
                    0,
                    valid,
                    tid,
                    TABLE_AC,
                    0,
                    threads,
                    mpad,
                    lanes,
                    False,
                    False,
                )

                # The increment cotangent, into the chunk-local frame and into the two
                # products it feeds. One matrix for the whole chunk, so its nine
                # entries are a broadcast read and the pass is one 3-vector per thread
                # per step: six coalesced reads widened to float32, nine FMA for the
                # frame change, twelve for the products. Both reads are of a state a
                # recurrence stored at the operand width, so the widening is free of
                # accuracy and the arithmetic below is float32 (I4). The two products
                # are over the whole lane extent and are taken on the tap that stages
                # the block first, so the chunk-start state is read once.
                total = rows * lanes
                steps = -(-total // threads)
                exact = total % threads == 0
                # A block is a whole number of lanes wide, so a step advances the flat
                # index by a whole number of rows: the lane a thread reads is the same
                # at every step and the row moves by a trace-time constant. Derived
                # from the flat index instead, each step carries its own element offset
                # and the eight of them are loop-invariant, outlive the whole lane loop
                # and are spilled: measured eight local words reloaded at the top of
                # every trip. Affine, they share one base and one row stride.
                affine = exact and threads % lanes == 0
                prow = tid // lanes
                pcol = tid - prow * lanes
                for group in cutlass.range_constexpr(-(-steps // PREFETCH)):
                    count = min(PREFETCH, steps - group * PREFETCH)
                    held = []
                    for step in cutlass.range_constexpr(count):
                        if cutlass.const_expr(affine):
                            p = prow + (group * PREFETCH + step) * (threads // lanes)
                            n = pcol
                        else:
                            i = tid + (group * PREFETCH + step) * threads
                            if cutlass.const_expr(not exact):
                                i = cutlass.min(i, total - 1)
                            p = i // lanes
                            n = i - p * lanes
                        d0 = l0 + 3 * n
                        got = (
                            widen(pdinc[p, d0], dinc_elem),
                            widen(pdinc[p, d0 + 1], dinc_elem),
                            widen(pdinc[p, d0 + 2], dinc_elem),
                        )
                        if cutlass.const_expr(tap == 0):
                            held.append(
                                (
                                    p,
                                    n,
                                    got,
                                    (
                                        widen(pz[p, d0], state_elem),
                                        widen(pz[p, d0 + 1], state_elem),
                                        widen(pz[p, d0 + 2], state_elem),
                                    ),
                                )
                            )
                        else:
                            held.append((p, n, got, None))

                    for step in cutlass.range_constexpr(count):
                        p, n, got, state = held[step]
                        local = mat3_matvec(aclast, got)
                        # A stride-3 store, so the eight threads of a phase touch
                        # three segments each. The kernel measures 0.0900 shared bank
                        # conflicts per wavefront at ``standard`` with
                        # ``mio_throttle`` at 3.7% against ``long_scoreboard`` at
                        # 27.4%, so this is not what bounds it and the staging order
                        # stays as `table.py` writes every other rotated tile.
                        for j in cutlass.range_constexpr(3):
                            sdinc[p, 3 * n + j] = narrow(local[j], elem)
                        if cutlass.const_expr(tap == 0):
                            if cutlass.const_expr(not exact):
                                # A clamped step repeats the last element, so its store
                                # repeats a correct value and only the reductions need
                                # the zero. Zeroing the state zeroes both of them.
                                live = tid + (group * PREFETCH + step) * threads < total
                                state = tuple(
                                    select(live, state[j], zero) for j in range(3)
                                )
                            # The closing scale rides the state rather than the finished
                            # sum, because the rotation cotangent's other half comes
                            # from the forcing product below and is not scaled.
                            scaled = tuple(cscale * state[j] for j in range(3))
                            for j in cutlass.range_constexpr(3):
                                dscale = dscale + local[j] * state[j]
                                for i in cutlass.range_constexpr(3):
                                    mrot[3 * i + j] = (
                                        mrot[3 * i + j] + local[i] * scaled[j]
                                    )
            cute.arch.sync_threads()

            # sum_d b_tap(r,d) dinc_local(p,d), the increment's contribution to the
            # forcing cotangent, accumulated unweighted into the output fragment. No
            # accumulator of its own: the fragment carries nothing else until the
            # diagonal GEMM below the lane loop, and the weight is a function of the
            # source token alone, so weighting the finished sum over lane blocks is the
            # same sum as weighting each block's part. The epilogue that applies it then
            # runs once per tap rather than once per lane block.
            for k in cutlass.range_constexpr(lsteps):
                mma_gemm(tiled_mma, tid, target, vforced[k], vlocal_n[k], True, True)

            # sum_p u_tap(r,p) dinc_local(p,d), the other half of the increment's outer
            # product. The element's matrix row is the accumulator column modulo three,
            # so it lands in the rotated basis at the static ``dres`` slot and costs
            # three multiply-adds rather than three selects and nine. The column is
            # block-local and the block starts on a lane triple, so the residue is the
            # same one the whole lane extent would give.
            #
            # Below the forcing GEMM, not above it. This accumulator is the only one in
            # the body that dies inside the body, and issued first it was live across
            # the forcing and score GEMMs as well: that allocation spilled 3,133,440
            # local load and 1,142,784 local store sectors per launch at the acceptance
            # shape, 136.84 MB of the 584.10 MB moved, which this order removes
            # outright. Nothing between the two writes shared memory and no barrier
            # separates them, so the order is free; both read ``sdinc``, which the
            # staging pass above finished before the barrier.
            dloc = mma_acc(tiled_mma, tid, (mpad, lblk))
            for k in cutlass.range_constexpr(ksteps):
                mma_gemm(tiled_mma, tid, dloc, vu[k], vlocal_k[k], True, False)
            for i in cutlass.range_constexpr(cute.size(dloc)):
                m, d = dcrd[i]
                base = d - d % 3
                weighted = dloc[i] * swgt[cutlass.min(m, last)]
                for j in cutlass.range_constexpr(3):
                    forced = widen(sbu[m, base + j], elem)
                    slot = 3 * dres[i] + j
                    mrotp[slot] = mrotp[slot] + weighted * forced

            if cutlass.const_expr(banked):
                for s in cutlass.range_constexpr(slices):
                    for k in cutlass.range_constexpr(lsteps):
                        mma_gemm(
                            tiled_mma,
                            tid,
                            score[s],
                            vforced[k],
                            vreadout[s][k],
                            True,
                            True,
                        )

        # The increment weight, on the finished sum over lane blocks, and the exponent's
        # forcing term with it. The fragment holds only the increment product here, so
        # this is the whole of what the weight applies to.
        for i in cutlass.range_constexpr(cute.size(target)):
            m, p = wcrd[i]
            weighted = target[i] * swgt[cutlass.min(m, last)]
            dexpw = dexpw + weighted * widen(su[m + tap, p], elem)
            target[i] = weighted

        for s in cutlass.range_constexpr(slices):
            tbase = s * tblk
            acc = score[s] if banked else score[0]
            if cutlass.const_expr(not banked):
                # One lane block, so the forcing tile and the readout still hold it
                # and the slice's score is taken here rather than banked.
                acc.fill(0.0)
                for k in cutlass.range_constexpr(lsteps):
                    mma_gemm(
                        tiled_mma, tid, acc, vforced[k], vreadout[s][k], True, True
                    )
            vb_dy = [
                cute.make_tensor(
                    sdy.iterator + tbase * ldu + k * MMA_TILE_K,
                    cute.make_layout((tblk, MMA_TILE_K), stride=(ldu, 1)),
                )
                for k in range(ksteps)
            ]
            vdiag = cute.make_tensor(
                sdy.iterator + tbase * ldu,
                cute.make_layout((rows, tblk), stride=(1, ldu)),
            )
            # Allocated here, not before the lane loop: neither is live inside it, and
            # the lane loop is where the pressure peaks. The narrowed score is the A
            # operand of the diagonal GEMM, so its view is built with it; the retile is
            # a layout and costs nothing per slice.
            dmacc = mma_acc(tiled_mma, tid, (mpad, tblk))
            sfrag = cute.make_fragment_like(dmacc, elem)
            fa_score = mma_areg(sfrag) if one_group else sfrag
            for k in cutlass.range_constexpr(ksteps):
                mma_gemm(tiled_mma, tid, dmacc, vu[k], vb_dy[k], True, True)
            for i in cutlass.range_constexpr(cute.size(dmacc)):
                m, n = scrd[i]
                token = tbase + n
                # I6: the mask lands on the float32 accumulator, then one narrowing
                # into the operand. I3: one exponential of a log difference. The clamp
                # only feeds rows the M mode was rounded up by, whose operands the
                # stagers zeroed.
                masked = acc[i] * decay(slp[token] - slp[cutlass.min(m, last)])
                masked = select(token >= m, masked, zero)
                sfrag[i] = narrow(masked, elem)
                # The exponent's cotangent, summed over the source token, which is
                # this accumulator's M mode. Its other sum, over the target token, is
                # the pair of inner products the epilogue takes against the finished
                # fragments, so the score is never revisited.
                column = _sum_over_m(masked * dmacc[i])
                if tid % 32 < 4:
                    sdlp[warp, token] = sdlp[warp, token] + column
            if cutlass.const_expr(one_group):
                mma_gemm_areg(tiled_mma, tid, target, fa_score, vdiag, False)
            else:
                # At two N groups a thread's consecutive N steps are two atoms apart,
                # so the fragment cannot be reread as a K-contiguous A operand. The
                # score goes through shared memory instead. The first barrier covers
                # the lane loop's last read of the aliased tile and the previous
                # slice's ldmatrix; the second covers this slice's store.
                cute.arch.sync_threads()
                cute.autovec_copy(sfrag, tiled_mma.get_slice(tid).partition_C(vscore))
                cute.arch.sync_threads()
                mma_gemm(tiled_mma, tid, target, vscore, vdiag, True, False)

    # Both fragments are final, so the log-scale sum over the target token, the carry
    # and the shift all read them in place.
    for i in cutlass.range_constexpr(cute.size(du)):
        m, p = wcrd[i]
        held = _sum_over_n(
            du[i] * widen(su[m + 1, p], elem) + dushift[i] * widen(su[m, p], elem)
        )
        if tid % 4 == 0 and m < chunk:
            sdlp[warp, m] = sdlp[warp, m] - held
        if m == 0:
            gcarry[bidx, hidx, cidx, p] = dushift[i]

    # Out of this thread's rotated basis, where slot ``a`` holds component
    # ``(lphase + a) % 3``. Eighteen selects once, against three per accumulator
    # element per lane block per tap in the component basis.
    for k in cutlass.range_constexpr(3):
        for j in cutlass.range_constexpr(3):
            rolled = select(
                lphase == cutlass.Int32(1),
                mrotp[3 * ((k + 2) % 3) + j],
                select(
                    lphase == cutlass.Int32(2),
                    mrotp[3 * ((k + 1) % 3) + j],
                    mrotp[3 * k + j],
                ),
            )
            mrot[3 * k + j] = mrot[3 * k + j] + rolled

    total_expw = block_reduce_add(dexpw, sred[0, None], tid, threads)
    total_scale = block_reduce_add(dscale, sred[1, None], tid, threads)
    mfull = tuple(
        block_reduce_add(mrot[i], sred[2 + i, None], tid, threads) for i in range(9)
    )
    drot = mat3_mul(mat3_transpose(aclast), mfull)
    if tid == 0:
        gdscale[bidx, hidx, cidx] = total_scale
        for i in cutlass.range_constexpr(3):
            for j in cutlass.range_constexpr(3):
                gdrot[bidx, hidx, cidx, i, j] = drot[3 * i + j]

    for step in cutlass.range_constexpr(-(-chunk // threads)):
        token = tid + step * threads
        if token < chunk:
            summed = zero
            for w in cutlass.range_constexpr(nwarps):
                summed = summed + sdlp[w, token]
            # The scatter lands on the chunk's last slot whether or not it carries a
            # token, because the increment weight differentiates the padded prefix
            # too.
            gdlp[bidx, hidx, cidx, token] = 2.0 * (
                summed + select(token == last, total_expw, zero)
            )

    for i in cutlass.range_constexpr(cute.size(dushift)):
        m, p = wcrd[i]
        sshift[m, p] = dushift[i]
    cute.arch.sync_threads()

    for i in cutlass.range_constexpr(cute.size(du)):
        m, p = wcrd[i]
        # The row above, clamped and then corrected, never predicated. The last valid
        # token's partner is the next chunk's first row and belongs to the boundary
        # kernel, so it reads zero here.
        above = sshift[cutlass.min(m + 1, mpad - 1), p]
        held = du[i] + select(m + 1 < valid, above, zero)
        if m < valid:
            # The seed joins the float32 sum ahead of the one narrowing, so a seeded
            # dU carries no rounding a bare one does not, and it costs one read
            # rather than the read, read and write a caller-side add would. Inside
            # the predicate because a padded row has no token to seed. Its element
            # type is dU's: the host holds the seed to U, and U is the stand-in when
            # no seed was given.
            if cutlass.const_expr(has_seed):
                stored = held + widen(gduinit[bidx, hidx, t0 + m, p], out)
            else:
                stored = held
            gdu[bidx, hidx, t0 + m, p] = narrow(stored, out)


@cute.jit
def chunk_input_bwd(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gduinit: cute.Tensor,
    gdu: cute.Tensor,
    gcarry: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    lblk: cutlass.Constexpr,
    tblk: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    has_seed: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_input_bwd_kernel`.

    ``P``, ``3N``, the lane block, the slice width and ``H // G`` are compile-time
    because the accumulator partitions and the arena offsets are. Batch, head, chunk
    count and sequence length are dynamic.
    """
    chunk_input_bwd_kernel(
        gdy,
        gu,
        guprev,
        gtrans,
        gtap,
        gb,
        gbprev,
        gc,
        gdinc,
        gz,
        gduinit,
        gdu,
        gcarry,
        gdlp,
        gdrot,
        gdscale,
        seqlen,
        make_mma(dtype, threads // 32),
        threads,
        chunk,
        rows,
        dim,
        lblk,
        tblk,
        per_group,
        has_prev,
        has_seed,
    ).launch(
        # The head is the fastest grid mode, so the blocks that share a group's forcing
        # and readout rows are consecutive and co-resident. At ``n_groups = 1`` that
        # turns eighteen reads of the same chunk of B and C into one read and seventeen
        # L2 hits; the per-head reads are the same either way.
        grid=(heads, bsz, chunks),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


class ChunkInputBwd(NamedTuple):
    """What one launch of the chunk-input backward produces.

    Attributes:
        dU: ``(B,H,T,P)`` cotangent of the forcing input, in the activation dtype.
            The chunk-boundary rows carry the diagonal term alone;
            :func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds the
            shifted term there.
        carry_u: ``(B,H,C,P)`` float32 cotangent that each chunk's first token sends
            to the token before it. Index 0 is the streaming feedback.
        dlogp: ``(B,H,C,L)`` float32 diagonal and increment half of the
            log-scale-prefix cotangent, the reference's ``dlogp_scan``.
        dchunk_rot: ``(B,H,C,3,3)`` float32 cotangent of each chunk's closing
            rotation, row-major.
        dchunk_scale: ``(B,H,C)`` float32 cotangent of each chunk's closing scale.
    """

    dU: Tensor
    carry_u: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor


def chunk_input_backward(
    dy: Tensor,
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    dinc: Tensor,
    zstart: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
    du_init: Tensor | None = None,
    threads: int = THREADS,
) -> ChunkInputBwd:
    """Differentiate the forcing input and the closing transition of every chunk.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` passes zeros: unlike the chunk-start cotangent, the
            increment terms survive and this kernel still has work.
        U: ``(B,H,T,P)`` forcing input, the dtype of ``dy``, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. ``(kr, g, h, 0)`` per tap.
        B: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. ``G`` divides ``H``; head
            ``h`` reads group ``h // (H // G)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched.
        dinc: ``(B,H,C,P,3N)`` increment cotangent in the global frame, the dtype
            of ``dy``, contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`.
        zstart: ``(B,H,C,P,3N)`` chunk-start state, the dtype of ``dy``,
            contiguous, from the forward's inter-chunk recurrence. Read, never
            written.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, or None.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, or None.
        du_init: ``(B,H,T,P)`` addend for ``dU``, shaped and typed like ``U``,
            pitched, or None. Read only. The epilogue adds it to the float32 sum
            before the one narrowing, so a caller with a gradient already bound for
            ``dU`` pays one read rather than a pass of its own.
        threads: Block width, a multiple of 128 at most
            ``32 * slinoss.ops.so3ssd.cute.mma.WARPS_WIDE``. The wide form halves
            every per-thread accumulator and adds two per-warp scratch rows.

    Returns:
        :class:`ChunkInputBwd`.

    Raises:
        ValueError: On a layout, rank, shape or extent violation, on a shared-memory
            budget the device cannot hold, on half a streaming pair, on a
            float32-pinned operand that is not float32, or on a stored state that is
            not at the activation dtype.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (U, "U"), (B, "B"), (C, "C"))
    pinned: Named = ((trans, "trans"), (K, "K"))
    stored: Named = ((dinc, "dinc"), (zstart, "zstart"))
    check_layout(((dy, "dy"), (U, "U"), *pinned, *stored))
    check_pitched(((B, "B"), (C, "C")))
    dtype = check_operands(activations)
    check_pinned(pinned)
    check_stored(stored, dtype)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        U, trans, K, (B, "B"), (C, "C")
    )
    if tuple(dy.shape) != tuple(U.shape):
        raise ValueError(f"dy must be {tuple(U.shape)}, got {tuple(dy.shape)}")
    check_rows(rows)
    check_extents(chunk_size, dim, tblock(chunk_size))
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))
    if du_init is not None:
        check_grad_band(du_init, U, "du_init")

    chunks = -(-seqlen // chunk_size)
    state = (bsz, heads, chunks, rows, dim)
    for tensor, name in ((dinc, "dinc"), (zstart, "zstart")):
        if tuple(tensor.shape) != state:
            raise ValueError(f"{name} must be {state}, got {tuple(tensor.shape)}")

    nwarps = threads // 32
    lblk = lblock(chunk_size, rows, dim, dy.element_size(), warps=nwarps)
    budget = assert_smem_fits(
        f"chunk_input_bwd[L{chunk_size}/P{rows}/3N{dim}/lane{lblk}/W{nwarps}]",
        input_smem_bytes(
            chunk_size, rows, dim, dy.element_size(), lblk=lblk, warps=nwarps
        ),
    )

    device = dy.device
    dU = torch.empty(bsz, heads, seqlen, rows, dtype=dtype, device=device)
    carry_u = torch.empty(bsz, heads, chunks, rows, dtype=torch.float32, device=device)
    dlogp = torch.empty(
        bsz, heads, chunks, chunk_size, dtype=torch.float32, device=device
    )
    dchunk_rot = torch.empty(
        bsz, heads, chunks, 3, 3, dtype=torch.float32, device=device
    )
    dchunk_scale = torch.empty(bsz, heads, chunks, dtype=torch.float32, device=device)
    jit_launch(
        chunk_input_bwd,
        (
            dy,
            U,
            U if u_prev is None else u_prev,
            trans,
            K,
            B,
            B if b_prev is None else b_prev,
            C,
            dinc,
            zstart,
            U if du_init is None else du_init,
            dU,
            carry_u,
            dlogp,
            dchunk_rot,
            dchunk_scale,
            seqlen,
            chunks,
            bsz,
            heads,
        ),
        (
            cute_dtype(dtype),
            threads,
            chunk_size,
            rows,
            dim,
            lblk,
            tblock(chunk_size),
            heads // groups,
            has_prev,
            du_init is not None,
            min(RESIDENT_MAX, max(1, smem_capacity() // budget)),
        ),
    )
    return ChunkInputBwd(
        dU=dU,
        carry_u=carry_u,
        dlogp=dlogp,
        dchunk_rot=dchunk_rot,
        dchunk_scale=dchunk_scale,
    )
