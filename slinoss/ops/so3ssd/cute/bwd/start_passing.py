"""Chunk-start cotangent and reverse state recurrence in one launch.

Unfused, the two halves talk through a whole ``(B,H,C,P,3N)`` float32 buffer:
:func:`slinoss.ops.so3ssd.cute.bwd.chunk_start.chunk_start_backward` writes
``dzstart`` and :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`
reads it back and overwrites it with ``dinc``. That round trip is 283 MB of the
pair's 455 MB at the model geometry, and it is the only thing either kernel has
left to give: the recurrence measured 99.6% of its own DRAM floor.

Fused, ``dzstart`` never reaches DRAM. One block holds a chunk's tile in shared
memory, the recurrence consumes it there, and only ``dinc`` and ``dz0`` are
written:

    dgram(L,P)      = dy * exp(2*lp)
    dzstart_c(P,3N) = dgram_c^T crot_c
    acc_C = dstate,  dinc_c = acc_{c+1},
    acc_c = a_c R(Q_c)^T acc_{c+1} + dzstart_c,  dz0 = acc_0

Parallelism. The recurrence is serial in the chunk, so the chunk mode leaves the
grid and becomes a reverse loop inside the block, exactly the arm
``chunk_start_backward(serial=True)`` prices. What replaces it is the lane band:
the state's ``3N`` columns split into ``SPLIT``-wide bands, one block per band,
because each 3-vector's recurrence is independent of every other. ``dy`` and
``trans`` are then read once per band rather than once, and ``C`` is still read
once because each band reads its own columns of it.

Traffic at ``B=4 H=18 T=2048 P=64 3N=240 L=64``, ``G=1``, five bands, against the
455 MB the unfused pair moves:

    dy       5*18.87           =  94.35 MB read
    trans    5*2.36            =  11.79 MB read
    C        4*2048*240*2      =   3.93 MB read
    cquat    5*4*18*32*4*4     =   0.23 MB read
    cscale   5*4*18*32*4       =   0.06 MB read
    dinc     4*18*32*64*240*2  =  70.78 MB written
    dz0      4*18*64*240*4     =   4.42 MB written
                                 185.56 MB

The ``trans`` line is the rescanning arm's. The shipped arm reads the prefixes
instead and reads no transitions at all, so it moves 173.77 MB and 14.75 MB of
prefix records in their place. See the prefix source below.

``dinc`` is written at the operand width, not float32. Its two consumers narrow it
to that width on the way into shared memory whatever it arrives at, so the store
here is the same rounding two launches earlier; ``dz0`` is a gradient of ``z0`` and
stays float32.

The five bands of one head are dispatched 18 blocks apart and the whole grid is
360 blocks against 168 resident, so the bands sharing a head's ``dy`` are
co-resident and L2 can serve four of their five reads. The counted figure above
charges none of that.

Registers are what sets ``SPLIT``. The GEMM accumulator is ``mpad*SPLIT/threads``
float32 and the recurrence carries ``rows*SPLIT/threads`` more, so the pair grows
linearly in the band width: at ``P=64`` and 128 threads, 48 columns is 24 and 24.
The whole state at once is 120 and 120, which cannot fit 255 registers, and a
spilled recurrence accumulator touched once per chunk would move exactly the bytes
this kernel exists to delete. The band is not a tuning knob for anything else.

Shared memory. The two GEMM operand tiles and the tile the recurrence reads share
one region, because no chunk has both live at once: the contraction reads the
operands and then writes its result, and the next chunk restages the operands only
after the recurrence has read that result. At the model geometry that makes the
recurrence's tile free and the block's footprint the unfused GEMM's 21,760 B, which
is four resident blocks of the 101,376 B carveout against the two that 34,048 B
allowed. Occupancy is what this kernel is short of, so the overlay is the point of
it and not a saving.

Barriers. Three per chunk, and the overlay is why two of them exist. One inside
:func:`start_chunk` between the contraction and the store, or a warp writes its
accumulator over an operand another warp is still reading. One after the store,
before the recurrence reads the tile. One after the recurrence, before the next
chunk's staging writes over what it just read -- the barriers inside the staging
come after its writes and cannot stand in for this one.

Every measurement in the rest of this docstring was taken with a float32 ``dinc``
store, which is 70.78 MB a launch more than the shipped one writes. The arm ranking
and the stall reasons are what they are read for; the absolute byte and time figures
are that width's. At the acceptance shape the narrowing takes the counted write from
146.93 MB to 76.72 MB and the launch from 410.3 us to 350.4, -59.9 us for -70.21 MB.

Measured on sm_86, bfloat16, no final-state cotangent, at the geometry above, both
arms in one process under one floor fit, every figure twice:

    arm    us/call       DRAM MB/call
    pair   741.0  741.7  453.9  454.0
    fused  474.9  474.9  176.5  176.8

267 us and 277 MB a call, one call a layer. The fused kernel reaches 55.2% of the
262 us floor its own DRAM traffic implies, where the recurrence it absorbed reached
99.3% of its; ranking the arms by that percentage inverts them, and
:data:`slinoss.perf.declared.DECLARED` carries the reading. The request stream is
the 256 MB the count above comes to at that width, not the 176 MB crossing the bus,
and the gap is the band re-reads L2 serves. What is left is 51.1% of the bus at 26.3% issue, 20.0% achieved
occupancy of 25.0% theoretical, 152 registers, and no local memory.

Block width. Occupancy is what is left, and the width buys it without buying bytes:
``mma_atoms`` pins the M mode, so warps past the first four go to the tile's N mode
at atom granularity and both accumulators halve at unchanged shared bytes. The GEMM
accumulator goes from ``mpad*span/128`` to ``mpad*span/256`` and the recurrence's
from ``rows*lanes/128`` to half that, which takes the kernel from 152 registers to
80 and lets the same 21,760 B block hold three resident blocks of eight warps rather
than three of four. Measured on sm_86 at the geometry above, one call, medians of
three event runs and one NCU capture of three launches each:

    warps  us/launch  MB/launch  GB/s  of 85%  regs  occ theo/ach  issue  barrier
    4          473.7     176.51  372.6   55.4%   152  25.0%/19.2%  26.3%    12.8%
    8          408.7     176.52  431.9   64.3%    80  50.0%/39.0%  32.5%    22.3%

65 us a call for a parameter, traffic identical to two decimal places and nothing
spilled at either width. The percentage rises with the time here, unlike the
fusion's, because the width moves no bytes. What it does move is the barrier stall,
12.8% to 22.3%: three barriers a chunk in this kernel and four more inside
:func:`start_chunk`, and the chunk prefix behind one of them is warp 0's work, so a
wider block waits wider.

Staging. Everything from here down is at the shipped ``dinc`` width.
:func:`slinoss.ops.so3ssd.cute.table.stage_weighted` carried one element an access
with the decay inside it: at ``L=64 P=64`` and 256 threads, 16 global loads, 16
shared stores, 16 prefix reads and 16 exponentials a thread a chunk.
:func:`slinoss.ops.so3ssd.cute.table.stage_raw` and
:func:`slinoss.ops.so3ssd.cute.table.weight_rows` split it at one 16-byte segment an
access and cost 2 of each. Measured on sm_86, eight warps, resident 3, one NCU
capture of one launch:

    counter                  before        after    delta
    sm__inst_executed    78,966,720   51,471,360   -34.8%
    pipe_lsu             13,262,400    9,391,680   -29.2%
    op_global_ld          2,615,040    1,324,800   -49.3%
    op_shared_ld          4,377,600    3,087,360   -29.5%
    op_shared_st          3,306,240    2,016,000   -39.0%
    registers                    80           70
    barrier stall             23.9%        34.9%
    long_scoreboard           15.5%        11.8%

42 accesses a thread a chunk go, 4 more than the two halves account for: the
toolchain forwards the raw store into the scale pass's read and drops both, so the
split pays no round trip. Paired against the fused pass in one process with the
order swapped, 16 pairs a trial and three trials: -20.5%, -19.9% and -19.8% of the
call, 48 of 48 pairs negative. The pass had been priced on its LSU port term and its
exponentials, and what dominated it was the per-element narrow, select and index
arithmetic.

The barrier stall was not a lever while the scan stayed. Nothing behind the first
barrier of :func:`start_chunk` depends on the chunk prefix scan -- the ``dy`` load
needs the token index and the valid count and nothing else, and the scan is warp 0's
alone, so seven warps of eight wait on it. Both forms of issuing that load ahead of
the scan were measured, both bitwise clean, neither faster. Staging unweighted through
shared and scaling in place after the scan restores the store and the load the
forwarding had dropped, +368,640 LSU and +1,584,390 shared wavefronts, and costs +2.9% to
+4.3% of the call over three paired trials. Holding the loaded fragment in registers
across the scan instead moves no memory-pipe count at all and costs 2 registers, 70
to 72; it does not resolve, at paired medians -0.9%, +0.5% and -0.3% with 11, 6 and
9 of 16 pairs agreeing in sign. It does move the stall it was aimed at, barrier
34.9% to 28.1%, and long_scoreboard rises 11.8% to 15.2% in exchange. The block's
critical path is the serial scan, so a warp that reaches the barrier earlier only
waits there longer.

Prefix source. So the scan goes instead of the wait.
:func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_prefix_bwd_kernel` writes both
prefixes to global once per ``(batch, head, chunk)``, and
:func:`slinoss.ops.so3ssd.cute.bwd.chunk_start.start_loaded` reads ``5 * L`` words of
them in the staging pass rather than rescanning: one barrier where the scan needed
two, and no warp-serial region between them. ``scanned`` keeps the rescan
reachable and the two are asserted bitwise equal, the global tensors holding the
output of the same :func:`slinoss.ops.so3ssd.cute.prefix.chunk_prefixes` code
including its renormalization. The producer is not new work: it already runs ahead of
``chunk_vector_bwd`` at the same grid and block, so
:func:`slinoss.ops.so3ssd.cute.backward.so3ssd_bwd_cute` hoists that one launch
rather than adding a second.

The transitions go with the scan. ``strans`` is the scan's operand and nothing
else's: :func:`slinoss.ops.so3ssd.cute.table.build_table` at ``mats == 1`` composes
``Ac`` from the quaternion prefix alone, and neither :func:`start_chunk` nor the chunk
loop below reads that tile. So the read arm stages ``4 * L`` fewer float32 a chunk a
block than the rescan rather than ``5 * L`` more, and at the acceptance geometry it
reads no transitions at all.

Measured on sm_86, the read against the rescan, 1000 order-swapped pairs a shape,
medians, launch-order position estimated from the two parities and removed, coverage
95.37%, two reps a row. ``null`` is the widest of a read-against-read and a
rescan-against-rescan control run in the same session at the same shape:

    shape        P   3N  bands  blocks   rep0 us   rep1 us  delta pct     null
    tiny        16   48      1       1    -2.048     +0.512  -15.4/+0.7   0.000
    standard    48   48      1      48   -10.752    -10.184  -11.3       +0.512
    ragged      48   48      1      48    -9.776    -10.240  -11.0       -0.512
    long        48   48      1      24   -28.672    -30.208  -12.6       -0.512
    wide        64   96      2      96   -11.264    -12.288   -9.0       -0.512
    acceptance  64  240      5     360   -25.088    -26.112   -8.5       +0.512
    P80 probe   80   48      1      48   -11.832    -10.752   -9.3       -0.512

Every row but ``tiny`` resolves in both reps, agrees within 1.536 us across them, and
no null control leaves one timer granule. ``tiny`` is one block and 13 us, both
readings are within two granules of zero, and they disagree in sign: the arm is not
measurable at that shape and is reported as such. Utilization from co-tenants ran
0-100% through the session, which is why every row carries its own two controls run
back to back with it rather than a quiet-device claim.

Staging them anyway is what made the arm regress at one band, and the reading that
regression got is worth recording because it was wrong twice. It read +2.096 us at
``standard`` and +7.9 at the ``P`` 80 probe with the staging left in, which no
computed predicate separates from the wins: the sign followed neither the band count
(a two-band probe regressed) nor the machine fill (a 96-block probe regressed) nor
the exact fill of the M mode (a ``P`` 32 probe won at ``mma_rows`` 64, the same tile
``P`` 48 pads into), and the residency step it correlates with cannot be the
mechanism at 48 blocks over 84 multiprocessors, where achieved
``sm__warps_active.avg.pct_of_peak`` is 16.66% in both arms and no residency limit
binds. Ordered by ``P`` the sign alternates, which is a lookup and not a rule.

The residency step is real but secondary. Forcing the read arm back to three blocks
with a dead shared allocation, at unchanged geometry, registers and instruction count,
splits the acceptance win of the form that still staged the transitions: the read
still beat the rescan by 16.896 us [-16.90, -16.90] at three blocks against three,
and the fourth block was the remaining 4.888 us. Those sum to 21.784 against a
21.720 us total, 0.064 us apart, and the null control -- the rescan arm with and
without the same dead allocation -- read 0.512 us, one timer granule. So 78% of that
shape's win was the deleted serial region and 22% the block, and a fourth block on
this kernel is a gain, not the loss a residency sweep of the rescanning form reported.
That 22% is not what the shipped arm rests on: widening
:func:`slinoss.ops.so3ssd.cute.table.stage_rotated`'s run put both arms at 64
registers and four blocks at ``acceptance``, so the register difference is gone there,
and the paired win is a granule larger than it was with the step present.

The workspace read is at its floor. Its footprint is ``5 * L`` float32 a chunk, 1.97 MB
a launch at ``P`` 48 and 2.95 MB at acceptance. Counted by difference against the
rescan, the read is 3 requests and 40 sectors a chunk a block at ``L=64`` and 5 and 80
at ``L=128``, which is 1,280 B and 2,560 B of sectors for 1,280 B and 2,560 B of
payload: over-fetch is zero and no rearrangement can lower it. What it replaces was
not at its floor. :func:`slinoss.ops.so3ssd.cute.table.stage_trans` reads a
token-major ``(T,4)`` row one component at a time and costs 128 sectors a chunk a
block for 1,024 B of payload, a fourfold over-fetch, so the two do not trade byte for
byte: global load sectors fall 135,168 a launch at ``standard``, 1.479 M to 1.344 M,
where the payload counts predict a rise of 12,288.

What the arm buys is the barrier, not the bus. At ``standard``, per launch, averaged
over two NCU launches: barrier stall 16.43 M to 10.61 M warp-cycles, -35.4%;
long_scoreboard 12.61 M to 12.83 M, +1.8%; ``smsp__warps_active`` 57.48 M to 51.73 M;
``issue_active`` 24.38% to 26.49%; ``sm__cycles_active`` 85,160 to 76,800, which tracks
the paired -11.3%. Requests fall 132,480 to 124,800, ``48 * 32 * 5`` exactly. Registers
68 to 67, shared bytes 21,760 either way, zero local spill sectors, and achieved
``sm__warps_active.avg.pct_of_peak`` 16.66% in both arms, so no occupancy term moves at
that shape. DRAM read rises, 20.94 MB to 21.31 MB: the records are not reused at one
band and the transitions were L2-served, so the bus gets slightly worse while the
launch gets 11% shorter. An arm on this kernel is not priced by DRAM bytes.

Nor is it occupancy, and that was checked where it could have been. At acceptance both
arms report 64 registers, ``launch__occupancy_limit_registers`` 4,
``sm__maximum_warps_per_active_cycle_pct`` 66.67 and 21,760 shared bytes, with achieved
``sm__warps_active.avg.pct_of_peak`` 54.44% against 53.85%; the arm is -25.088 and
-26.112 us there anyway. Its sectors and requests are ``360 * 32 * 88`` and
``360 * 32 * 5`` to the counter's last digit, 11.935 M to 10.921 M and 1.048 M to
990,720, shared store wavefronts fall 5.057 M to 4.957 M, and DRAM read rises again,
42.98 MB to 44.36 MB. So one mechanism carries every shape: the sectors the staging was
fetching and the barrier the block reached them through.

Two forms of covering the read's latency are refused, both measured on the form that
still staged the transitions. Serving the record from cache -- the same code with a
zero chunk stride, so every chunk reads one resident record and the outputs are wrong
by construction -- was worth 8.2 us at ``standard`` and 15.1 at the ``P`` 80 probe,
which is what put the staging beside it under suspicion. Fetching the record a chunk
ahead into two register fragments, so the whole of :func:`start_chunk` stands over the
load, moved nothing at one band, +2.560 us at ``standard`` against +2.056, and cost
the arm its largest wins: the fragments are 8 registers, the read arm at ``wide`` and
``acceptance`` sat at exactly 64 before the staging widen, the four-block bar at eight
warps, and the 8 took
``acceptance`` from -23.040 to +3.584. The cost was the sector count, which a prefetch
reorders and does not lower. The occupancy figures in the row-band table below are the
rescanning form's.

Row band. 360 blocks against 168 resident is 1.43 waves at three blocks an SM, and
the 0.715 quantization efficiency of that wave count accounts for the whole of the
39.0%-against-50.0% occupancy gap. Splitting ``P`` into bands as well as ``3N``
doubles the grid and closes it, and the kernel is slower anyway:
:func:`slinoss.ops.so3ssd.cute.mma.mma_rows` rounds the M extent back up to
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_M`, so half of every contraction is
padding, and ``trans``, ``C``, ``cquat`` and ``cscale`` are requested once per row
band. Measured on sm_86 at the geometry above, eight warps, one NCU capture of three
launches, event medians of three runs:

    rows  blocks  us/launch  MB/launch  occ theo/ach  issue  tensor  l1tex  smem st
    64       360      410.1     176.52  50.0%/39.0%  32.4%    7.5%  44.8%  5.21 M
    32       720      454.5     199.03  66.7%/58.4%  48.3%   13.8%  66.4%  7.40 M
    16      1440      738.2          -            -      -       -      -       -

The 64-row line recaptures the width table's eight-warp line and lands 0.3% off it.
Occupancy and issue both rose; the shared pipe and the bus took more than they gave.
The class percentage rose with them, 64.0% to 64.9%, on a kernel 11% slower -- an arm
:data:`slinoss.perf.declared.DECLARED` cannot be ranked by. The M mode has no atom
narrower than 64 rows, so the band has no cheaper form and is not a lever.

``warps`` is a parameter and the default is
:data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`, the width the table above measures
fastest. The narrow width stays reachable because the driver prices both under one
floor fit and the two are asserted bitwise equal.
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
    cute_dtype,
    jit_launch,
    narrow,
    smem_bytes,
    smem_capacity,
)
from slinoss.ops.so3ssd.cute.bwd.chunk_start import (
    gram_tile,
    rotated_tile,
    start_chunk,
    start_loaded,
    start_scanned,
)
from slinoss.ops.so3ssd.cute.bwd.chunk_vector import PREFIX_WARPS, chunk_prefix_bwd
from slinoss.ops.so3ssd.cute.bwd.state_passing import StatePassingBwd
from slinoss.ops.so3ssd.cute.common import (
    mat3_matvec,
    mat3_transpose,
    rot_hom,
    scalar_tile,
    table_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.guard import (
    Named,
    check_extents,
    check_layout,
    check_operands,
    check_pinned,
    check_pitched,
    check_shapes,
)
from slinoss.ops.so3ssd.cute.mma import (
    SMEM_SEGMENT,
    WARPS_WIDE,
    make_mma,
    mma_acc,
    mma_atoms,
    mma_rows,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.table import TABLE_PITCH, stage_pad

__all__ = [
    "RESIDENT_MAX",
    "SPLIT",
    "ChunkPrefixes",
    "chunk_prefix_backward",
    "fold_smem_bytes",
    "start_passing_backward",
    "start_passing_bwd",
    "start_passing_bwd_kernel",
    "state_tile",
]

RESIDENT_MAX: int = 3
"""Blocks per SM the launch bound asks for, before the shared-memory budget cuts it.

The chunk loop is serial and every iteration barriers three times, so a block spends
most of its time waiting: measured on sm_86 at the model geometry with the tiles
allocated separately, 42.1% of the issue slots stalled on ``long_scoreboard`` at
36.4% of the bus and 20.8% issue, which is a kernel bounded by having eight warps per
SM and not by the pipe. Residency is the only thing that covers it, and residency is
what overlaying the recurrence's tile on the operands' bytes buys: 21,760 B a block
at the model geometry, four blocks of it against the 101,376 B carveout.

Four is what the tiles admit; three is what the register file prefers, because the
bound caps a thread at ``65536 / (blocks * threads)`` registers and this kernel holds
a GEMM accumulator and a recurrence accumulator at once. Measured on sm_86 at the
model geometry, one call of the fused kernel:

    blocks/SM   us    regs   occupancy theo/achieved
    2          607.6   196   16.7% / 15.8%
    3          473.6   152   25.0% / 19.5%
    4          514.2   128   33.3% / 29.6%

Nothing spills at any of the three, so the cost of asking for four is the scheduling
the 24 registers buy and not local memory.

Three is also what the wide block prefers, and there the bound is the whole of it.
sm_86 allocates registers in granules of 256 a warp out of a 65,536-register file,
so the cap at eight warps is that quotient floored to a granule: 80 registers a
thread at three blocks, 128 at two, 64 at four. The staged split takes the kernel
from 80 to 70, which admits a fragment held across the chunk prefix scan and still
does not admit four blocks. Measured on sm_86 at the model geometry, medians of three
event runs, one call:

    blocks/SM   us
    1          492.5
    2          492.5
    3          411.6
    4          439.3

The four durations are at the float32 ``dinc`` width.

One and two tie because both admit the register count the kernel wants and both fit
two blocks of eight warps per SM; three is the first bound that fits a third.
"""

SPLIT: int = 48
"""Columns of ``3N`` one block contracts and carries.

A multiple of 3, so a band holds whole 3-vectors, and of
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`, so it is an N extent the atom
covers. 48 is the smallest width satisfying both, and every legal ``3N`` is a
multiple of it, so one width divides every shape. The module docstring carries why
a wider band does not fit the register file.
"""


class ChunkPrefixes(NamedTuple):
    """The two chunk-local transition prefixes, in global memory.

    Attributes:
        lp: ``(B,H,C,L)`` float32 inclusive log-scale scan.
        q: ``(B,H,C,4,L)`` float32 inclusive quaternion prefix product,
            component-major and renormalized once (I5).
    """

    lp: Tensor
    q: Tensor


def chunk_prefix_backward(trans: Tensor, chunk_size: int, groups: int) -> ChunkPrefixes:
    """Scan every chunk's transition prefixes to global memory.

    One pass over ``trans`` for the launches that read the prefixes instead of
    rescanning them: :func:`start_passing_bwd_kernel` and
    :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_bwd_kernel`. Both
    read the same two tensors, so neither can disagree with the other about a
    prefix.

    Args:
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        chunk_size: ``L``. A multiple of 16.
        groups: ``G``, the grid's slowest mode. Divides ``H``.

    Returns:
        A :class:`ChunkPrefixes`. ``5 * L`` float32 a chunk, 2.95 MB at the
        acceptance shape.

    Raises:
        ValueError: If ``groups`` does not divide ``H``.
    """
    bsz, heads, seqlen, _ = trans.shape
    if groups < 1 or heads % groups:
        raise ValueError(f"groups must divide H={heads}, got {groups}")
    chunks = -(-seqlen // chunk_size)
    device = trans.device
    lp = torch.empty(bsz, heads, chunks, chunk_size, dtype=torch.float32, device=device)
    q = torch.empty(
        bsz, heads, chunks, 4, chunk_size, dtype=torch.float32, device=device
    )
    # The kernel's head index is ``(gidx * splits + sidx) * fold + hstep``, so the
    # covered set is ``H`` exactly when ``splits * fold == H // G``. Pinning the fold
    # to one head a block gives that at every shape and does not inherit
    # ``chunk_vector_bwd``'s sharding, where a missing head would be silent.
    jit_launch(
        chunk_prefix_bwd,
        (trans, lp, q, seqlen, chunks, bsz, groups),
        (PREFIX_WARPS, chunk_size, 1, heads // groups),
    )
    return ChunkPrefixes(lp=lp, q=q)


def state_tile(rows: int, span: int) -> Tile:
    """Chunk-start cotangent tile, ``(mpad, span)`` float32.

    Contiguous and row-major, which is what :func:`mma_store` requires of any
    destination, and pitched to the band width so the flat index of a 3-vector is
    the same arithmetic in shared memory and in ``dinc``.

    The rounded M extent is allocated even where the store predicates the added
    rows away, because the vectorized path writes all of them.

    Args:
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
    """
    return Tile((mma_rows(rows), span), (span, 1))


def arena_words(chunk: int, rows: int, span: int, itemsize: int = 2) -> tuple[int, int]:
    """Float32-word extent of the overlaid region, and the offset inside it.

    The two GEMM operand tiles and the tile the recurrence reads are never live at
    the same time: the contraction reads the operands and then writes its result,
    and the next chunk restages the operands only after the recurrence has read
    that result. So one region holds both, and the recurrence's tile costs nothing
    at the shape the kernel is declared against -- which is what leaves room for the
    residency :data:`RESIDENT_MAX` asks for.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        ``(words, gram_words)``: the region's float32-word extent, and the offset at
        which the rotated readout tile starts. Both are whole words because every
        tile's row pitch is a multiple of :data:`SMEM_SEGMENT`.
    """
    gram = smem_bytes([(gram_tile(chunk, rows), itemsize)])
    rotated = smem_bytes([(rotated_tile(chunk, span), itemsize)])
    state = smem_bytes([(state_tile(rows, span), 4)])
    return max(gram + rotated, state) // 4, gram // 4


def fold_smem_bytes(chunk: int, rows: int, span: int, itemsize: int = 2) -> int:
    """Shared memory the kernel allocates, in bytes.

    The four small float32 tiles of the chunk-start GEMM, then the one region
    :func:`arena_words` describes.

    Args:
        chunk: ``L``.
        rows: ``P``.
        span: Band width, :data:`SPLIT`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    words, _ = arena_words(chunk, rows, span, itemsize)
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, 1, TABLE_PITCH), 4),
            (Tile((words,), (1,)), 4),
        ]
    )


@cute.kernel
def start_passing_bwd_kernel(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    gc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdinc: cute.Tensor,
    gdz0: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
    scanned: cutlass.Constexpr,
) -> None:
    """Contract each chunk's readout cotangent and run the reverse recurrence.

    One block per ``(band, batch, head)``, walking the chunks in reverse.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gslp: ``(B,H,C,L)`` float32 inclusive log-scale scan, from
            :func:`chunk_prefix_backward`. Read only when not ``scanned``.
        gsquat: ``(B,H,C,4,L)`` float32 inclusive quaternion prefix product,
            component-major, from the same. Read only when not ``scanned``.
        gc: ``(B,G,T,3N)`` operand-dtype output vectors.
        gcquat: ``(B,H,C,4)`` float32 unit chunk rotations.
        gcscale: ``(B,H,C)`` float32 chunk decays, ``exp(2*lp_{L-1})``.
        gdstate: ``(B,H,3*P*N)`` float32 cotangent of the final state. Read only
            when ``has_dstate``; the zero-seed variant is handed ``gdz0`` here so
            the signature has one form.
        gdinc: ``(B,H,C,3*P*N)``, written with the cotangent of each chunk's
            increment in the global frame. Write only, so it carries the operand
            dtype its consumers read it at and the store narrows to it.
        gdz0: ``(B,H,3*P*N)`` float32, written with the cotangent of the initial
            state.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic.
        tiled_mma: From :func:`make_mma`.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        span: Band width, :data:`SPLIT`. Compile-time.
        per_group: ``H // G``, heads sharing one ``c``. Compile-time.
        has_dstate: Whether a final-state cotangent is supplied. Compile-time.
        scanned: Whether to rescan the chunk prefixes from ``gtrans`` in warp 0
            rather than read ``gslp`` and ``gsquat``. Compile-time. The rescan is
            the arm the read is priced against; the module docstring carries what
            it measured.

    Invariants:
        ``span`` divides ``dim``, 3 divides ``span``, and ``threads`` divides
        ``rows * span / 3``, so the band is a whole number of 3-vectors and every
        thread owns the same count of them. ``|R(Q_c)| == 1`` and ``a_c`` lies in
        ``(0, 1]`` by I1, so the reverse recurrence cannot grow.
    """
    tid, _, _ = cute.arch.thread_idx()
    # Head is the fastest grid mode, as in the unfused GEMM: the ``H // G`` blocks
    # reading one group's readout band are co-resident. The band mode is next, so
    # the bands sharing a head's ``dy`` are close enough behind it to hit L2.
    hidx, sidx, bidx = cute.arch.block_idx()

    gidx = hidx
    if cutlass.const_expr(per_group != 1):
        gidx = hidx // per_group

    mpad = mma_rows(rows)
    lanes = span // 3
    vecs = rows * lanes // threads
    dinc_elem = gdinc.element_type
    lda = smem_pitch(mpad)
    ldb = smem_pitch(span)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(
        cutlass.Float32, table_tile(chunk, 1, TABLE_PITCH).layout(), SMEM_SEGMENT
    )

    # One region, three views: the two GEMM operands, and the contraction's result
    # over the same bytes. The pitches make every offset a whole float32 word and a
    # whole 16-byte segment, which is the alignment both stagers and ``mma_store``
    # restate on the pointer they are handed.
    words, gwords = arena_words(chunk, rows, span, gdy.element_type.width // 8)
    arena = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((words,), stride=(1,)), SMEM_SEGMENT
    )
    sdy = cute.make_tensor(
        cute.recast_ptr(arena.iterator, dtype=gdy.element_type),
        gram_tile(chunk, rows).layout(),
    )
    scrot = cute.make_tensor(
        cute.recast_ptr(arena.iterator + gwords, dtype=gc.element_type),
        rotated_tile(chunk, span).layout(),
    )
    sdz = cute.make_tensor(arena.iterator, state_tile(rows, span).layout())

    # The band's origin is a pointer offset: ``3N`` is the last mode at unit stride,
    # so the staging pass indexes the band's own columns and never learns that the
    # state is wider.
    band = cute.make_tensor(gc.iterator + sidx * span, gc.layout)

    acc = mma_acc(tiled_mma, tid, (mpad, span))
    state = cute.make_fragment((3 * vecs,), cutlass.Float32)

    # One 3-vector's coordinates in the tile and in the ``(P,3N)`` plane. The band
    # index is dynamic and the vector index is not, so these are hoisted out of the
    # chunk loop rather than recomputed inside it.
    tile_row = []
    tile_col = []
    plane = []
    for k in cutlass.range_constexpr(vecs):
        # A name reused inside the chunk loop below at another structure is read as
        # a loop-carried variable whose type changes, which the DSL refuses.
        owned = tid + k * threads
        row = owned // lanes
        lane = owned - row * lanes
        tile_row.append(row)
        tile_col.append(3 * lane)
        plane.append(row * dim + sidx * span + 3 * lane)

    for k in cutlass.range_constexpr(vecs):
        for j in cutlass.range_constexpr(3):
            if cutlass.const_expr(has_dstate):
                state[3 * k + j] = gdstate[bidx, hidx, plane[k] + j]
            else:
                state[3 * k + j] = cutlass.Float32(0.0)

    for step in cutlass.range(chunks):
        cidx = chunks - 1 - step
        # Restaged every chunk, not once: the columns at or past each tile's data
        # width are read as operands and never restaged by the stagers, and the
        # previous chunk's result was written over them.
        stage_pad(scrot, tid, threads, chunk, span, ldb)
        stage_pad(sdy, tid, threads, chunk, rows, lda)
        if cutlass.const_expr(scanned):
            start_scanned(
                gtrans,
                strans,
                slp,
                squat,
                seqlen,
                cidx,
                bidx,
                hidx,
                tid,
                threads,
                chunk,
            )
        else:
            start_loaded(
                gslp, gsquat, slp, squat, cidx, bidx, hidx, tid, threads, chunk
            )
        start_chunk(
            gdy,
            band,
            sdz,
            strans,
            slp,
            squat,
            stable,
            scrot,
            sdy,
            acc,
            tiled_mma,
            seqlen,
            cidx,
            bidx,
            hidx,
            gidx,
            tid,
            threads,
            chunk,
            rows,
            span,
            True,
        )
        cute.arch.sync_threads()

        quat = (
            gcquat[bidx, hidx, cidx, 0],
            gcquat[bidx, hidx, cidx, 1],
            gcquat[bidx, hidx, cidx, 2],
            gcquat[bidx, hidx, cidx, 3],
        )
        decayed = gcscale[bidx, hidx, cidx]
        # The transpose is a reindexing of a tuple at trace time, and the matrix is
        # one per chunk for every 3-vector the thread carries.
        mat = mat3_transpose(rot_hom(quat))
        for k in cutlass.range_constexpr(vecs):
            carried = (state[3 * k], state[3 * k + 1], state[3 * k + 2])
            # The increment cotangent is the accumulator before this chunk's
            # readout term enters it, so the store precedes the update.
            for j in cutlass.range_constexpr(3):
                gdinc[bidx, hidx, cidx, plane[k] + j] = narrow(carried[j], dinc_elem)
            # One rotation, then the scale, then the readout cotangent: the forward
            # scales the state alone and rotates the sum, and transposing that
            # order puts the scale outside the rotation.
            turned = mat3_matvec(mat, carried)
            for j in cutlass.range_constexpr(3):
                state[3 * k + j] = (
                    decayed * turned[j] + sdz[tile_row[k], tile_col[k] + j]
                )
        # The next chunk's staging writes over the tile just read, the two being one
        # region, so the reads have to finish first. The barriers inside the staging
        # come after its writes and cannot stand in for this one.
        cute.arch.sync_threads()

    for k in cutlass.range_constexpr(vecs):
        for j in cutlass.range_constexpr(3):
            gdz0[bidx, hidx, plane[k] + j] = state[3 * k + j]


@cute.jit
def start_passing_bwd(
    gdy: cute.Tensor,
    gtrans: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    gc: cute.Tensor,
    gcquat: cute.Tensor,
    gcscale: cute.Tensor,
    gdstate: cute.Tensor,
    gdinc: cute.Tensor,
    gdz0: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bands: cutlass.Int32,
    bsz: cutlass.Int32,
    heads: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    per_group: cutlass.Constexpr,
    has_dstate: cutlass.Constexpr,
    scanned: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`start_passing_bwd_kernel`.

    The grid is ``(H, 3N/span, B)``, head-fastest, so the ordering argument the
    unfused GEMM makes about ``C`` survives the fusion.

    ``resident`` is the launch bound, computed from the tiles by the host entry
    rather than chosen here; see :data:`RESIDENT_MAX`.

    The block width is the tiling's warp count and nothing else, so it arrives as
    ``warps`` and the thread count is derived from it: two parameters would let the
    launch geometry and the accumulator partition disagree.
    """
    threads = warps * 32
    start_passing_bwd_kernel(
        gdy,
        gtrans,
        gslp,
        gsquat,
        gc,
        gcquat,
        gcscale,
        gdstate,
        gdinc,
        gdz0,
        seqlen,
        chunks,
        make_mma(dtype, warps),
        threads,
        chunk,
        rows,
        dim,
        span,
        per_group,
        has_dstate,
        scanned,
    ).launch(
        grid=(heads, bands, bsz),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


def start_passing_backward(
    dy: Tensor,
    trans: Tensor,
    C: Tensor,
    cquat: Tensor,
    cscale: Tensor,
    chunk_size: int,
    dstate: Tensor | None = None,
    *,
    prefixes: ChunkPrefixes | None = None,
    span: int = SPLIT,
    warps: int = WARPS_WIDE,
    scanned: bool = False,
    resident: int | None = None,
) -> StatePassingBwd:
    """Form every chunk's start-state cotangent and pass it back through the scan.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` has no chunk-start cotangent to form and runs the
            recurrence alone rather than this kernel against zeros.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. ``G`` divides ``H``; head
            ``h`` reads group ``h // (H // G)``.
        cquat: ``(B,H,C,4)`` float32, contiguous. Unit chunk rotations
            ``Q_{L-1}``, scalar-first.
        cscale: ``(B,H,C)`` float32, contiguous. Chunk decays ``exp(2*lp_{L-1})``.
        chunk_size: ``L``. A multiple of 16.
        dstate: ``(B,H,P,3N)`` float32, contiguous. Zero seed if omitted.
        prefixes: The two chunk-local prefixes from
            :func:`chunk_prefix_backward`, or None to run that pass here. A caller
            whose next launch reads them too supplies them, so the pass runs once
            for both. Ignored under ``scanned``, which reads neither.
        span: Band width. :data:`SPLIT` is the only value the register budget
            admits at every legal ``P``; it is an argument so a driver can price a
            wider one.
        warps: Warps per block, a multiple of
            :data:`slinoss.ops.so3ssd.cute.common.WARPS` at most
            :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`. Warps past the first
            four go to the tile's N mode, which halves both accumulators at
            unchanged shared bytes and unchanged traffic; see the module docstring
            for what that is worth.
        scanned: Whether warp 0 rescans the chunk prefixes from ``trans`` instead
            of the block reading them from ``prefixes``. Same result bitwise, and
            it exists to price the read against the rescan under one floor fit.
            Not a tuning knob.
        resident: Blocks per SM the launch bound asks for. Defaults to
            :data:`RESIDENT_MAX` capped by the shared-memory budget; it is an
            argument for the same reason ``span`` is, since the cap it puts on the
            register file is what decides whether the residency is reached.

    Returns:
        A :class:`slinoss.ops.so3ssd.cute.bwd.state_passing.StatePassingBwd`. Both
        fields are fresh buffers: nothing is written in place, because the
        chunk-start cotangent this fuses away never exists in memory.

    Raises:
        ValueError: On a layout, rank, shape, or extent violation, on a supplied
            prefix that is not the shape the scan writes, on a band width the
            launch cannot cover exactly, or on a ``warps`` that is not a legal
            block width.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (C, "C"))
    pinned: Named = ((trans, "trans"), (cquat, "cquat"), (cscale, "cscale"))
    if dstate is not None:
        pinned = (*pinned, (dstate, "dstate"))
    check_layout(((dy, "dy"), *pinned))
    check_pitched(((C, "C"),))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        dy, trans, None, (C, "C"), label="dy"
    )
    check_extents(chunk_size, dim, chunk_size)

    chunks = -(-seqlen // chunk_size)
    if tuple(cquat.shape) != (bsz, heads, chunks, 4):
        raise ValueError(
            f"cquat must be {(bsz, heads, chunks, 4)}, got {tuple(cquat.shape)}"
        )
    if tuple(cscale.shape) != (bsz, heads, chunks):
        raise ValueError(
            f"cscale must be {(bsz, heads, chunks)}, got {tuple(cscale.shape)}"
        )
    # Raises on an illegal width, so the block geometry is checked here rather than
    # inside the trace.
    mma_atoms(warps)
    threads = warps * 32
    if span % 3 != 0 or dim % span != 0 or (rows * span // 3) % threads != 0:
        raise ValueError(
            f"span must divide 3N={dim}, be a multiple of 3, and give a whole "
            f"number of {threads}-thread tiles of P*span/3, got span={span} P={rows}"
        )
    budget = fold_smem_bytes(chunk_size, rows, span, dy.element_size())
    assert_smem_fits(
        f"start_passing_bwd[L{chunk_size}/P{rows}/span{span}/W{warps}]", budget
    )
    asked = RESIDENT_MAX if resident is None else resident
    blocks = min(asked, max(1, smem_capacity() // budget))

    # dinc leaves this launch and is read by two GEMM kernels that narrow it to the
    # operand width on the way into shared memory, so it is stored at that width
    # here: the same rounding one launch earlier and half the traffic. Under float16
    # the store also flushes what float32 held as a subnormal: at the acceptance
    # shape the smallest nonzero element is 2.537e-09 and 9.325e-07 of them are
    # below float16's smallest subnormal, so they reach the consumer as zero.
    dinc = torch.empty(bsz, heads, chunks, rows, dim, dtype=dtype, device=dy.device)
    dz0 = torch.empty(bsz, heads, rows, dim, dtype=torch.float32, device=dy.device)
    if dstate is None:
        seed = dz0
    elif tuple(dstate.shape) != (bsz, heads, rows, dim):
        raise ValueError(
            f"dstate must be {(bsz, heads, rows, dim)}, got {tuple(dstate.shape)}"
        )
    else:
        seed = dstate

    # A rescanning launch reads neither tensor, so it takes the two it is handed
    # whatever they hold: the kernel's own compile-time branch drops the loads, and
    # allocating a workspace nothing reads would be 2.95 MB at the acceptance shape.
    if scanned:
        prefix_lp, prefix_q = trans, trans
    else:
        held = (
            chunk_prefix_backward(trans, chunk_size, groups)
            if prefixes is None
            else prefixes
        )
        want = (
            (held.lp, "prefixes.lp", (bsz, heads, chunks, chunk_size)),
            (held.q, "prefixes.q", (bsz, heads, chunks, 4, chunk_size)),
        )
        for tensor, name, shape in want:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")
        check_layout(((held.lp, "prefixes.lp"), (held.q, "prefixes.q")))
        check_pinned(((held.lp, "prefixes.lp"), (held.q, "prefixes.q")))
        prefix_lp, prefix_q = held.lp, held.q

    jit_launch(
        start_passing_bwd,
        (
            dy,
            trans,
            prefix_lp,
            prefix_q,
            C,
            cquat,
            cscale,
            seed.view(bsz, heads, rows * dim),
            dinc.view(bsz, heads, chunks, rows * dim),
            dz0.view(bsz, heads, rows * dim),
            seqlen,
            chunks,
            dim // span,
            bsz,
            heads,
        ),
        (
            cute_dtype(dtype),
            warps,
            chunk_size,
            rows,
            dim,
            span,
            heads // groups,
            dstate is not None,
            scanned,
            blocks,
        ),
    )
    return StatePassingBwd(dinc=dinc, dz0=dz0)
