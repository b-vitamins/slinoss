"""``dB``, ``dC``, ``dtrans``, ``dK`` and the forcing-vector carry.

Everything the backward owes the two rowwise vectors and the per-token transition
parameters, in one block per ``(chunk, batch, group)``. The reference terms are
``dcrot``, ``dbnow``, ``dbprv``, ``dac``, ``dan``, ``dap``, ``dtap``, ``dw`` and
``dls``.

Six contractions, all dense real GEMMs off the one atom:

    dm_tap(t,r)  = sum_p  dy(t,p) u_tap(r,p)
    dmT_tap(r,t) = sum_p  u_tap(r,p) dy(t,p)
    dcrot(t,d)   = sum_p  dy(t,p) zstart(p,d)  + sum_r Smasked(t,r) brot_tap(r,d)
    dbnow(r,d)   = wgt(r) sum_p u_tap(r,p) dlocal(p,d)
                                              + sum_t SmaskedT(r,t) crot(t,d)

with ``u_tap`` the current tap at ``r`` and the previous tap at ``r-1``,
``Smasked = dm * dmask``, ``dlocal = R(Q_{L-1})^T dinc`` the increment cotangent
carried back into the chunk-local frame, and ``wgt(r) = exp(2*(lp_{L-1} - lp_r))``.
The offset term opens the readout accumulator and the increment term opens the
forcing accumulator, because both carry a factor that depends on the accumulator's
M mode alone: ``exp(2*lp_t)`` on the first and ``wgt(r)`` on the second, and a
factor applied to a finished sum of two terms would reach the wrong one.

``dm`` is built once and transposed through shared memory. The readout consumer
contracts over the source token and the forcing consumer over the target token, so
each wants the other's N mode as its K mode, and
:func:`slinoss.ops.so3ssd.cute.mma.mma_areg` rereads a fragment in place only along
its own N. The readout consumer takes the fragment; the forcing consumer reads the
same values back through a transposed ``ldmatrix`` from an operand tile that aliases
the tap transfer region. One ``L*L*P`` GEMM per tap instead of two, one float32
accumulator and one operand fragment fewer in the register frame, two barriers more,
and no bytes either way.

``B`` is read once. The two rotated forcing tiles are built from the raw tile in
shared memory by :func:`_rotate_rows`, which is the transform
:func:`slinoss.ops.so3ssd.cute.table.stage_rotated` applies, sourced from shared
memory instead of from global: the raw tile is needed anyway for the tap
cotangent, and the rotation is nine FMA against a second and third pass over
``B``. Bit-identical to the staged form, which widens the same stored value.

``C`` is read once, rotated on the way in, and the raw readout vector is never
read. ``ac`` is a rotation, so ``c = ac^T crot``, and every term that would read
``c`` collapses onto the rotated tile::

    dac = sum_n outer(dcrot_n, c_n) + dap Kprev^T + dan Kcurr^T
        = [sum_n outer(dcrot_n, crot_n) + outer(dbprv_n, bprv_n)
                                       + outer(dbnow_n, bnow_n)] ac

because ``Kcurr b = ac^T bnow`` and ``Kprev bshift = ac^T bprv`` by the same
identity. One 3x3 product per token replaces a second pass over ``C`` and two
tiles. The raw forcing vector survives the collapse only in the tap cotangent,
``ac^T dan = sum_n outer(ac^T dbnow_n, b_n)``, whose second factor no rotation of
``b`` can supply.

The log-scale offset term is never a second pass. With
``gram(t,p) = <crot_t, zstart_p>``,

    dlp_off(t) = 2 exp(2 lp_t) sum_p dy(t,p) gram(t,p)
               = 2 exp(2 lp_t) <crot_t, dcrot_unscaled_t>

so it is one lane reduction against the offset GEMM's own accumulator, taken
before the exponential scales it.

The tap and rotation cotangents are lane reductions over ``N`` of an outer
product, so they are taken in the epilogue that walks the finished vector tile
three columns per thread. ``tap_matrix_vjp`` runs there too: its input is complete
the moment that tap's reduction is, which keeps the scratch at nine floats per
token rather than twenty-seven.

``dB``, ``dC`` and the carry are sums over the heads sharing a group, and the fold
``H // G`` is cut into :func:`vector_splits` shards, one block to a shard. The fold is
one at ``standard`` and eighteen at the default configuration, and the depth is the
fold: one head to a block.

A head a block walks past the first is a rolled iteration between an accumulator's
allocation and its uses, which is what defeats register promotion. What that costs
does not follow the trip count, so the two costs of the sum do not trade: at
``L 64 P 64 3N 240 G 1`` and the shipped width a call takes 4,301.8 us at depth one,
holds between 4,033.5 and 4,262.9 from depth two to depth nine as the partials
accumulate on top of a spill that has not moved, and falls to 3,493.9 at the full
depth, where the loop's trip count is one and the spill is gone with it. One counter
pass at each end of the sweep: 255 registers, 144.51 MB of local loads and 8.52 MB of
local stores at depth one, against 242 registers and no local traffic at all at the
full depth. The local traffic is a property of the loop existing.

A shard owns a partial and not an output. At depth one the block writes the three
outputs itself, after the last head, in shared memory throughout. Above one it writes
rows that :func:`vector_reduce` sums in a second launch, which is what
:mod:`slinoss.ops.so3ssd.cute.bwd.boundary` does for partials of ``dB``. There are no
atomics either way, and each partial carries its own output's width: the closure
accumulates in float32 whatever it reads, so a partial wider than the output buys a
rounding the output does not keep, and the depth changes the summation order and the
rounding count together. That is the launch's largest single traffic item and it is
what the width is measured against, below.

The state width is tiled. Every tile of either set that spans ``3N`` spans
:data:`LANE_BLOCK` lanes of it, so the live set is bounded by a lane tile and one
launch path serves every ``d_state``. The tile is a grid axis, not a loop: ``3N`` is
the one extent the budget does not bound, so a loop over it multiplies the code by a
count that grows with the model while a grid axis multiplies the blocks. At the
default configuration that is 640 blocks where the loop form launched 128, on an
84-multiprocessor part.

What crosses a lane tile is the transition chart: ``dtrans`` and ``dK`` are sums over
lanes. A loop accumulated them in the row one block owned; separate blocks cannot,
so each tile writes its own float32 slot row and :func:`close_slots` sums the slots
in a second launch. At one tile there are no slots and the store is the output
itself, which is the whole of the difference between the two modes.

Shared memory is one resident set and one phase arena. Resident: ``trans``, ``K``,
the two chunk-local prefixes, the three-slot transform table, the nine-float
per-token scratch, the log-scale and quaternion cotangents, and the rotated
readout. The arena holds the two float32 sums that outlive a head, one region per
tap holding either the float32 forcing gradient or the narrowed score, and five
operand tiles: the output cotangent, one tile that carries the chunk-start state and
then the increment cotangent, the raw and rotated forcing tiles and the ``U`` tile.
The float32 readout gradient of the epilogue aliases those five, none being live
when it is.

The source-token block is :func:`vblock`, one M tile of the atom where the budget
allows it and half of one where it does not. Below one M tile every warp still
carries rows of every GEMM, because the transposed contractions round their M mode
up to the tile.

The budget bounds ``L``, ``P`` and the fold. It does not bound ``3N``, and this is
still the widest live set in the tree. ``L 16`` and ``L 32`` fit at every ``P``,
every fold and every ``3N``.
``L 64`` fits to ``P 64`` at every fold, in 93,904 B at fold one and 95,952 B above
it at the shipped width and 256 B less at one warp group, whether ``3N`` is 48 or
240. The table's segment-aligned pitch costs ``36 * L`` B of each figure, 2,304 B at
``L 64``, and it does not scale with the fold. ``L 64`` at ``P 128`` and ``L 128`` at
every ``P``
are refused: the smallest live set at ``L 128`` is 125,136 B, above the capacity of
every device the DSL reports. :func:`slinoss._cute.assert_smem_fits` refuses the
rest rather than any path here degrading.

The largest ``L`` this layout admits is 64, at one resident block, at ``P 64`` and at
``P 48`` alike. Two resident blocks of 128 threads need 50,688 B of the 101,376 B
carveout, and no legal shape at ``P 64`` reaches it: the smallest arena there is
54,224 B, at ``L 16`` and fold one. Splitting the fold across blocks removes the one
accumulator that exists only above fold one, 13,312 B of the 95,696, and leaves
82,384 B, so it buys parallelism and not occupancy. Four extents scale with ``L`` --
the two float32 fold sums, the output tile and the aliased float32 readout -- which
is why 128 is refused and why grid-izing the lane tile does not unpin it.

DRAM-bound. Analytic traffic at ``standard``, operand by operand, with ``U`` and
``B`` at the ``L + 1`` rows per chunk their shifted span reads::

    reads   dy 9.44 + U 9.58 + B 9.58 + C 9.44 + trans 1.57 + K 3.15
          + dinc 14.16 + zstart 14.16 + dlogp 0.39 + dchunk_rot 0.06
          + dchunk_scale 0.01                                        = 71.53 MB
    writes  dB 9.44 + dC 9.44 + dtrans 1.57 + dK 3.15 + carry_b 0.29 = 23.89 MB

95.42 MB against ``1536 * 4.03 MFLOP = 6.19 GFLOP``, so 64.9 flop/byte against a
ridge point of 164: memory bound by a factor of 2.5. That table is the ``span 64``
form. A shape whose budget forces ``span 32`` doubles the ``U`` term, since the
``U`` tile is one atom M tile whatever the block, and raises the intensity to 108
flop/byte at the default configuration, still under the ridge.

It is also the one-lane-tile form. Every operand carrying ``3N`` is read once
whatever the tile count, since each tile reads its own columns; the operands with no
state extent are read once per tile. At the scale above that is ``dy``, ``U``,
``trans``, ``K``, ``dlogp`` and the two chunk cotangents, 24.20 MB. ``standard`` is
one tile, so the table stands there as written.

At the default configuration, ``B 4 H 18 T 2048 P 64 3N 240 L 64`` chunk 64 with one
group, there are five tiles and the split is what the tile count costs::

    per tile x5   dy 18.87 + U 38.34 + trans 2.36 + K 4.72 + dlogp 0.59
                + dchunk_rot 0.08 + dchunk_scale 0.01     = 64.97 -> 324.86 MB
    once          B 3.99 + C 3.93 + dinc 141.56 + zstart 141.56 r,
                  dB 3.93 + dC 3.93 + carry_b 0.12 w      =           298.99 MB
    slot rows     dtrans 2.36 + dK 4.72 written and read back per tile,
                  then 7.08 written out once               =           77.85 MB

701.75 MB, and the slot rows are 14.16 MB more than the loop form's
read-modify-write of the same two outputs, 2.1%. ``U`` dominates the per-tile term
because its tile is one atom M tile whatever the ``span``, so a ``span 32`` shape
reads it twice. ``dinc`` and ``zstart`` are float32 ``(B, H, C, P, 3N)`` and together
are 40% of the total. The three write terms are the depth-one form. At the shipped
depth they are :func:`partial_bytes` instead, 143.77 MB, which is the largest single
item in the launch and the whole of what the closure reads.

That total is an upper bound and the launch does not pay it. 837.5 MB analytic at the
shipped depth against 515.81 MB of DRAM counted, 62% over. The 321.7 MB of daylight is
the size of the per-tile re-read term, 259.89 MB, and the lane tile is the innermost
axis of ``x``, so the five tiles of a token block run back to back and L2 serves the
repeats. Score this kernel against its counted traffic, never against the table.

Measured, the bar is missed, and the distance is latency and not traffic. Every
counter below is from one profile of this kernel on an RTX A6000, ``sm_86``, 84
multiprocessors, one profiled launch per counter pass at the shipped depth and six for
the extent table below, clocks unlocked because
locking is denied on this fleet. A floor is ``c + bytes / B`` on a fit taken in the
same process, ``c`` about 4.3 us and ``B`` about 685 GB/s at a worst residual of
0.40%, and ``bytes`` is the analytic traffic above unless stated otherwise. A
duration is stamped with the compute-apps query taken before and after it; where
that query named another process the duration is a bound, not a rate.

At the default configuration -- 11,520 blocks of 256 threads, five lane tiles, the
fold of 18 cut into eighteen shards, one head to a block -- the main kernel moves
515.74 MB of DRAM per launch in 1,851.8 us on device time at 1.7969 GHz, median of
five steady-state launches in each of two A/B blocks, 3,285,344 active cycles at a
0.59% spread.
:func:`vector_reduce` closes the head sum in 222.0 us at 152.77 MB and 102.7% of its
own floor; the two lane-slot reductions add 44.2 and 23.0 us. Three of the four
launches are at their bandwidth; the main kernel is the one that misses the 85% the
class asks, and it is issue-bound rather than short of bandwidth.

Read a microsecond figure here against the clock it was taken at. The part boosts
between 1.4 and 1.9 GHz with contention from other processes on the device, and the
same kernel has measured 2,649.0 us on a contended part and 2,103.5 us on an idle
one under the same counter passes. Cycles are the invariant; a duration is not.
Stamp the clock from ``gpc__cycles_elapsed.avg.per_second`` beside any duration.

That percentage is not a traffic problem, and at the shipped width it is not
instruction supply either. 93,904 B and 154 registers a thread each admit one
256-thread block per multiprocessor, ``launch__occupancy_limit_shared_mem`` and
``launch__occupancy_limit_registers`` both reading 1: 16.7% theoretical occupancy,
16.6% achieved, two warps a scheduler. ``mio_throttle`` leads the stalls, issue-active
is 30.7%, and DRAM runs at 26.6% of peak against tensor at 13.9%. What the launch
spends its cycles on is the LSU issue port, which carries 66.80 M warp instructions
and occupies 48.4% of the port's peak issue rate: read ``l1tex__throughput`` as that
occupancy and not as a bandwidth, since it measures equal to
``sm__inst_executed_pipe_lsu`` against its own peak digit for digit. The padding rule
is held centrally. Local traffic is zero at every one of the three launches.

Where the bytes are, measured rather than counted. The launch's DRAM splits 347.69 MB
read and 319.73 MB write with float32 vector partials, against a write side the
partials, ``dtrans`` slots and ``dK`` slots account for to within 1.00 MB and a read
side ``dinc`` and ``zstart`` alone account for 283.12 MB of. The remaining 64.57 MB of
reads stands against 370 MB of requests for ``dy``, ``U``, ``trans``, ``K``, ``dlogp``,
``B`` and ``C``, whose distinct footprint is 53.27 MB: the L2 read hit rate is 59.39%
and it serves the lane-tile and group re-reads at 1.21x compulsory. So neither the
five-fold re-read of the per-token operands nor the eighteen-fold re-read of the
group's two vectors is a device-traffic item, and only the head-sum partials and the
two float32 state buffers are.

Narrowing the two vector partials from float32 to the activation width is what the
measured split says to do, and what it pays depends on the width it is measured at.
At four warps, three runs each on one device in one session with the narrowed runs
under strictly more foreign compute-apps contention, the main launch goes from
668.40 MB and 7,601.9 us to 515.36 MB and 5,202.4 us, 22.9% of the traffic for 31.6%
of the time, and the closure from 294.82 MB and 433.2 us to 153.05 MB and 323.6 us.
The call falls 7,575.6 us to 5,584.9 us and the workspace 285.33 MB to 143.77 MB. The
time is not the bytes: ``no_instruction`` falls 54.3% to 32.3%, issue-active rises
12.13% to 18.68%, and every speed-of-light rises with it, memory 30.6% to 44.9%. At
14.6% of its own floor the launch was never paying for those bytes at the bus; the
store width was gating the front end.

At eight warps the same 141.56 MB comes off the main launch for 1% of its time,
3,306.7 us and 656.85 MB against 3,274.1 and 515.74, which is inside the spread of
three runs, and the call falls 3,811.3 us to 3,559.4 entirely inside the closure. The
front end it was gating is no longer the constraint at that width, so the bytes are
worth their bytes and no more. The mechanism of the four-warp figure is not
established here. What is established is that the counter that moved there is an
issue counter, so a traffic argument does not predict this launch's time in either
direction, and the remaining 283.12 MB of float32 ``dinc`` and ``zstart`` cannot be
priced from its bytes either. Narrowing those two would be DRAM-only whatever it
paid: :func:`slinoss.ops.so3ssd.cute.table.stage_state` and
:func:`slinoss.ops.so3ssd.cute.table.stage_matrix` narrow both to the operand width
on the way into the one state tile, so their global width reaches no shared byte and
no operand.

The closure paid for it in class until it took the request back. ``vector_reduce`` read
float32 at 680.5 GB/s and 100.4% of floor; one narrowed element a thread makes a warp's
32 columns 64 B and two sectors rather than 128 B and four at an unchanged request
rate, and it fell to 472.9 GB/s and 70.4% of floor with ``long_scoreboard`` at 92.7%.
:func:`partial_pack` restores the request at :data:`PARTIAL_REQUEST_BYTES` a thread,
which is two bfloat16 columns and one float32 column, and the pass returns to 689.5
GB/s and 102.8% of floor in 221.6 us. That is 102.0 us off the narrowed pass and 211.2
us off the float32 pass it replaces, at 56 registers against 96, no local traffic, and
the same bits: the shard order is the launch geometry's and the vector width does not
enter it.

The depth is what took it there, and not by cutting traffic. At depth one the same
kernel is 11,376.9 us over 640 blocks, moves 775.81 MB, and spills 536.74 MB of local
load and 344.06 MB of local store a launch, 94.9% and 99.9% of that past L1. Depth one
carries no workspace at all, 819.1 MB of operator DRAM against 1,005.3 MB at the full
depth when the vector partials were float32, and still lost by 34.0% of the call: what
the depth buys is 18x the blocks and no spill, so the traffic sum is not what it is
chosen against. At the narrowed partial the full depth wins on both, 711.5 MB.

The allocator stops at the 255-register cap at either depth. The fold is what spills,
and only the fold. Holding ``P 64``, ``3N 240`` and the code fixed and moving one
extent, with the fold folded in the block::

    L    fold   smem     regs   local MB   DRAM MB   GB/s    us/launch
    16      1   53,648    205       0.00   1,507.72  214.8      7,018.1
    32      1   64,848    255       0.00     935.91  200.8      4,661.1
    64      1   91,344    255       0.00     654.49  114.6      5,711.7
    32     18   71,504    255     398.46     693.58  137.1      5,060.0
    64     18   93,392    255     922.42     985.73   82.3     11,982.6

The three fold-one rows carry a float32 vector partial, so their DRAM column is
141.56 MB high for the shipped width; the local column, which is what the table is
for, does not depend on it. At fold one there is no local traffic at any ``L``, at 255
registers, with the lane tile in the grid and the ``L`` extents at their largest. At
fold 18 there is, and it grows with ``L``. The register count is at the cap either way, so the cap is not the
signal; what crosses a rolled fold iteration is. Unrolling the fold at trace time
does not fix it and makes it worse, 1,290.4 MB, because the live ranges of eighteen
inlined bodies overlap. Only taking the fold out of the block removes it, and it does:
zero local sectors at the shipped depth.

The lane tile is not implicated. At fold 18 and one lane tile the local traffic is
276.96 MB, and the tile count moves neither the register count nor the footprint.

Two resident blocks need 50,688 B and no legal shape at ``P 64`` reaches it: the
smallest arena there is 53,648 B, at ``L 16`` and fold one. Taking the fold out of the
block drops the summed accumulator that exists only above fold one, but the spaces
alias and only 2,048 B of the 93,392 come back, so it does not reach it either. What
it does reach is zero spill, worth 34.0% of the call. Beyond that, occupancy is worth
what the one shape that fits shows: ``L 16`` at ``P 48`` is 47,728 B, two resident
blocks, 16.5% achieved against 8.3%, and 71.5% of memory speed-of-light against 45.5%
at ``P 64`` where the same ``L`` holds one block.

The other lever is the block width, and it is a parameter: ``warps`` selects the atom
tiling and the thread count together, and eight is the default. The M mode of the tiling
is pinned and the extra warps go to ``N`` at atom granularity, so the tile, the pitches
and the staging passes are the width's invariants. Measured at both widths on the shape
above, medians of three runs, at both vector-partial widths. Every row was taken in one
session on a contended part at a lower clock than the shipped figure above, so the rows
compare to each other and not to it::

    warps  partial  us/launch  MB/launch  GB/s  of 85%  regs  arena   occ theo/ach
        4      f32    7,029.2     667.01  95.0   13.9%   255  91,344   8.3% / 8.3%
        8      f32    3,306.7     656.85 198.6   29.2%   242  91,600  16.7% / 16.6%
        4     bf16    5,221.0     517.82  99.2   14.6%   255  91,344   8.3% / 8.3%
        8     bf16    3,274.1     515.74 157.5   23.2%   242  91,600  16.7% / 16.6%

Registers fall to 242, so 256 threads hold 61,952 of the 65,536 a multiprocessor has
and the second warp per scheduler is available. Across the float32 pair,
``no_instruction`` goes 52.8% to 1.0% and issue-active 12.5% to 28.8%: instruction
supply was the whole of that gap. What takes its place is the shared-memory pipe, and
the shipped width's counters for it are in the measured record above. Local traffic
stays zero at either width, conflicts 0.1511 per wavefront against 0.1612,
instructions issued rise 1.1% for the readout term's reread, and the arena grows by
the ``4 * L`` bytes a warp group past the first that :func:`offset_tile` takes -- one
resident block at either width. Sixteen warps is refused by the register file before
any tiling question: 512 threads admit 128 registers a thread.

The same arithmetic refuses a second resident block at eight, and it is the register
file that refuses it rather than the arena. Two 256-thread blocks need 128 registers a
thread and 50,688 B of the 101,376 B carveout; three, which is what a 50% occupancy
bar asks for, need 85 and 33,792 B. At 242 registers ``launch__occupancy_limit_registers``
is 1 block, so no arena reaches a second block. Nor could the arena reach 50,688 B by
aliasing: the source-token loop holds every tile live at once except ``sdrot``,
``sdquat`` and ``sdls``, which are written and read inside the closing chart, so
lifetime aliasing has 2,304 B in it and its floor is 89,296 B against a 40,912 B gap.
The forcing sum outlives the head, the score and the tap gradient already share one
region, and ``dy``, the increment cotangent, the raw and rotated forcing tiles and
``U`` are each read by a GEMM of every tap. Halving the source-token block buys
11,264 B of that gap at a second pass over ``U``, and the tile cannot narrow below 48
columns. Occupancy at this width is a register problem.

An arena under 50,688 B would also raise ``min_blocks_per_mp`` to two by the
expression above, and that half of the plan is measured: the request alone takes the
allocator to exactly 128 registers a thread and it spills, 11.80 MB of local load and
11.80 MB of local store a launch with 368,556 of the 368,640 load sectors and all of
the store sectors missing L1, at 3,413.2 us against 3,274.1 and 519.91 MB against
515.74. Occupancy does not move, the arena still admitting one block. So the register
ceiling a second block imposes is not free at this shape, and the arena and the
register file would have to clear together for either to pay.

The width also reaches a spill the depth cannot. At ``B 4 H 12 T 2048 L 64 P 48 3N 48
G 12`` the fold is already one, and four warps still move 0.79 MB each of local load
and store a launch, 75.4% and 100% of that past L1. Eight warps take the register
count from 255 to 208 and the local traffic to zero, 959.1 us to 463.2 and 15.3% to
31.3% of the bar. The declared class follows the traffic; the figures above are what
the kernel reaches against it.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    Scalar,
    Stream,
    Tile,
    assert_smem_fits,
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
from slinoss._reduce import reduce_partials
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AN,
    TABLE_AP,
    THREADS,
    Mat3,
    Vec3,
    mat3_add,
    mat3_matvec,
    mat3_mul,
    mat3_outer,
    mat3_transpose,
    quat_exp_vjp,
    rot_hom_vjp,
    scalar_tile,
    tap_matrix_vjp,
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
    check_stream,
)
from slinoss.ops.so3ssd.cute.mma import (
    MMA_INST,
    MMA_TILE_M,
    SMEM_SEGMENT,
    WARPS_WIDE,
    fp32_tile,
    make_mma,
    mma_acc,
    mma_areg,
    mma_atoms,
    mma_coords,
    mma_gemm,
    mma_gemm_areg,
    mma_groups,
    mma_offsets,
    mma_rows,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes, chunk_suffix, quat_suffix_vjp
from slinoss.ops.so3ssd.cute.table import (
    LANE_PAIR,
    build_table,
    paired,
    stage_chunk,
    stage_matrix,
    stage_rotated,
    stage_shifted,
    stage_state,
    store_pair,
)
from slinoss.ops.so3ssd.reference import check_grad_band

__all__ = [
    "LANE_BLOCK",
    "LANE_GROUP",
    "PARTIAL_REQUEST_BYTES",
    "RESIDENT_MAX",
    "ROW_WORDS",
    "TABLE_PITCH",
    "TABLE_QUAD",
    "Arena",
    "ChunkVectorBwd",
    "Slots",
    "arena",
    "chunk_vector_backward",
    "chunk_vector_bwd",
    "chunk_vector_bwd_kernel",
    "close_slots",
    "forced_tile",
    "gradient_tile",
    "lane_block",
    "offset_tile",
    "open_slots",
    "out_tile",
    "partial_bytes",
    "partial_pack",
    "quad_table_tile",
    "readout_tile",
    "row_tile",
    "score_tile",
    "shifted_tile",
    "state_tile",
    "vblock",
    "vector_reduce",
    "vector_reduce_kernel",
    "vector_smem_bytes",
    "vector_splits",
]

LANE_BLOCK: int = 16
"""Lanes of one lane tile: the block the resident set is bounded by, not ``N``.

Every tile that spans ``3N`` is cut to ``3 * LANE_BLOCK`` columns and the kernel
loops over the cut, so the footprint is flat in ``3N`` and one launch path serves
every state width. 16 lanes is the smallest legal tile: 48 columns is a multiple of
the atom's N mode and of the 3-vector, and no smaller multiple of 3 is a multiple
of 16. It is also the largest tile that divides every legal ``3N``, since ``3N`` is
a multiple of 48 and 96 does not divide 240.

The cut is along a mode the six contractions carry as N or M, never as K, so no
partial sum crosses a tile. What crosses is the transition chart: ``dtrans`` and
``dK`` are sums over lanes, so a tile past the first accumulates into them."""

LANE_GROUP: int = 4
"""Threads that cooperate on one token in a rowwise epilogue.

One thread holds one 3-vector, which is what the rowwise transforms and the outer
products need and what an accumulator fragment cannot give: the atom hands a thread
two adjacent columns, and a 3-vector straddles that pair. The group must divide the
lanes of a lane tile, 16 at every shape that tiles, and stay inside a warp.

The group is priced by its butterfly. A rowwise epilogue reduces nine floats over
the group, so a pass issues ``9 * log2(group)`` shuffles per tuple, and a run of
``span`` tokens takes ``span * group / threads`` passes: the shuffle count grows as
``group * log2(group)`` while the reduced work is flat, every thread holding
``lanes * span / threads`` 3-vectors at any group. On GA10x a shuffle moves no bytes
and still costs a full LSU warp instruction, the same issue slot a shared load takes,
so the butterfly is priced like traffic it does not generate. Everything else a pass
carries -- the two table reads, the lane-zero tail, its four ``dtap`` words -- falls
with the pass count as well.

Measured on an A6000 at the acceptance shape, per warp: a group of 16 issues 751.5
shuffles and 1,802.9 LSU instructions, a group of 8 issues 301.5 and 1,202.9, and a
group of 4 issues 121.5 and 941.9. That is 73.9%, 64.0% and 54.6% of the L1TEX issue
port, and 29.4% off the kernel's cycles between 16 and 4. Registers hold at 242, 242
and 241 with no local traffic, so the shorter butterfly costs no spill.

Below 4 a pass covers more tokens than ``L 64`` has and threads idle, so 4 is the
floor at 256 threads. It already idles three quarters of the block at ``L 16``, where
one pass of 64 tokens covers four chunks; that shape pays four passes where a group
of 16 paid one, and is not what the group is set for.

Nine words do not fit one to a lane below a group of nine, so the scratch row is
stored in ``ceil(ROW_WORDS / LANE_GROUP)`` rounds of ``LANE_GROUP`` words, against
a pass count that falls by the same factor the group does."""

ROW_WORDS: int = 9
"""Float32 scratch per token: the 3x3 rotation cotangent, summed over ``N``.

The tap cotangents do not appear because ``tap_matrix_vjp`` runs inside the
epilogue that reduces them, so only the rotation's own sum outlives a phase.

The pitch is this count itself, so a token's nine words are consecutive. The
rowwise epilogues accumulate them one word a lane, :data:`LANE_GROUP` words a round,
so a warp's read-modify-write spans ``32 / LANE_GROUP`` tokens and
``9 * 32 / LANE_GROUP`` consecutive words. The pitch separates the banks of one
round only while that span stays under 32 words: at a group of 16 a warp is two
tokens and 18 words on 18 distinct banks, and below that the span passes 32 and the
pitch has to be checked round by round.

At a group of eight a warp is four tokens and 36 words. The full round takes words
0 to 7 of each, 32 addresses, and ``9 * 3 + 5 == 32``, so word ``k`` of the first
token and word ``k + 5`` of the fourth share a bank: three pairs collide, ``k`` being
0, 1 or 2, one pair on each of three banks. That round is two-way and takes two
wavefronts; the second round is word 8 alone on four lanes and is conflict-free.
The same check at a group of four gives eight tokens, 72 words, and ``9 * 7 + 1 ==
64``: three pairs again in each of the two full rounds, two-way, with the third
round eight lanes wide and clean.

No pitch clears both accesses. The post-loop reader takes a whole row per thread,
where nine and 32 being coprime is what keeps 32 consecutive tokens on 32 distinct
banks; the only pitches that make a group of four one-way, 12 and 20, share a factor
with 32 and put that reader at four-way for a third more scratch. The pitch stays
nine and the two-way stands, which is the cheaper side. A two-way round costs one
extra wavefront, and two-way rounds times passes times the three epilogue calls comes
to six a warp on the store and six on the load at either group: 552,960 stores at the
acceptance shape, against a measured 553,737 rise in store conflicts from a group of
16 to a group of 8, where nothing else on that access changed. What bounds this kernel
is LSU warp instructions, which a conflict does not add to and the round count does.

Every round holds at every ``L``: the chunk length sets how many lanes the
``token < chunk`` guard leaves active, and masking lanes off removes accesses
rather than colliding them."""

RESIDENT_MAX: int = 2
"""Blocks per SM the launch asks for, before the shared-memory budget lowers it.

The budget lowers it to one at every standard size. Asking for two costs nothing
where it cannot be had and takes it at the small shapes where the arena is half as
wide."""

TABLE_QUAD: int = SMEM_SEGMENT // 4
"""Float32 words in one 16-byte shared-memory segment."""

TABLE_PITCH: int = 3 * TABLE_QUAD
"""Float32 pitch of one transform-table entry: nine words padded to three segments.

The pitch :func:`slinoss.ops.so3ssd.cute.common.table_tile` gives the entry is nine,
which makes the token stride 36 bytes and leaves only every fourth entry 16-byte
aligned, so no entry can be read at vector width. At twelve the stride is 48 bytes,
every entry is aligned, and three 16-byte loads cover a row exactly: nine scalar
loads become three, which is the deletion :func:`_mat_at` exists for.

The three padding words are never written and never read for their value.

The cost is a third more table bytes, ``3 * L * 12`` against ``3 * L * 9``, 2,304 B
at ``L 64``. Residency does not move for it: registers and shared memory each pin
one block per SM on their own at every standard size, and the padded budget stays
inside the queried carveout at every shape
:func:`slinoss.ops.so3ssd.cute.guard.check_extents` admits."""


def lane_block(dim: int) -> int:
    """Columns of one lane tile, ``min(3N, 3 * LANE_BLOCK)``.

    Divides ``3N`` at every legal shape:
    :func:`slinoss.ops.so3ssd.cute.guard.check_extents` holds ``3N`` to a multiple
    of 3 and of 16, which are coprime, so ``3N`` is a multiple of 48.

    Args:
        dim: ``3N``.
    """
    return min(dim, 3 * LANE_BLOCK)


def row_tile(chunk: int) -> Tile:
    """Per-token float32 scratch, ``(L, ROW_WORDS)``."""
    return Tile((chunk, ROW_WORDS), (ROW_WORDS, 1))


def quad_table_tile(chunk: int, mats: int = 3) -> Tile:
    """Transform table at the segment-aligned pitch, ``(mats, L, TABLE_PITCH)``.

    The table :func:`slinoss.ops.so3ssd.cute.common.table_tile` describes, with the
    innermost extent padded from nine to :data:`TABLE_PITCH` so an entry is a whole
    number of 16-byte segments and :func:`_mat_at` can read it at vector width. Slot
    and entry order are unchanged, so every producer and consumer that indexes
    ``[slot, token, entry]`` reaches the same value at either pitch.

    Args:
        chunk: ``L``.
        mats: Slots to allocate, 1, 2 or 3.

    Raises:
        ValueError: If ``mats`` is not 1, 2 or 3.
    """
    if mats not in (1, 2, 3):
        raise ValueError(f"table needs 1, 2 or 3 matrices, got {mats}")
    return Tile((mats, chunk, TABLE_PITCH), (TABLE_PITCH * chunk, TABLE_PITCH, 1))


def offset_tile(chunk: int, warp_groups: int = 1) -> Tile:
    """Log-scale cotangent under accumulation, ``(warp_groups, L)`` float32.

    The offset term's reduction over the state width ends in one read-modify-write
    per accumulator row, by the leader of the quad that owns the row. A tiling that
    splits the tile's N mode gives a row one such leader per warp group, in
    different warps, with no barrier between them and no shuffle able to join them,
    so each group gets a row of its own. Row 0 is the sum, and the rows above it are
    folded into it once, before the reverse scan reads it.

    Row-major with ``L`` innermost, so a pass over the tokens of one row is unit
    stride and row 0 alone is a dense ``(L,)`` tile at the same address.

    Args:
        chunk: ``L``.
        warp_groups: Warp groups the tiling splits the N mode into,
            :func:`slinoss.ops.so3ssd.cute.mma.mma_groups`.
    """
    return Tile((warp_groups, chunk), (chunk, 1))


def readout_tile(chunk: int, dim: int) -> Tile:
    """Rotated readout tile, ``(mma_rows(L), pitch)``.

    An M mode of the offset and the readout GEMM and a K mode of the forcing GEMM,
    hence the rounded row count.

    Args:
        chunk: ``L``.
        dim: ``3N``.
    """
    return operand_tile(mma_rows(chunk), dim)


def forced_tile(span: int, dim: int) -> Tile:
    """Rotated forcing tile, ``(span, pitch)``.

    Only ever an N mode, of the readout GEMM, so the row count is the block
    itself.

    Args:
        span: Tokens of the source-token block.
        dim: ``3N``.
    """
    return operand_tile(span, dim)


def shifted_tile(span: int, width: int) -> Tile:
    """Shifted staging tile, ``(span + 1, pitch)``.

    Row ``j`` holds token ``nbase + j - 1``, so the previous tap reads rows
    ``0..span-1`` and the current tap the same rows one further on.

    Args:
        span: Tokens of the run.
        width: ``P`` or ``3N``.
    """
    return operand_tile(span + 1, width)


def score_tile(chunk: int, span: int) -> Tile:
    """Narrowed score tile, ``(mma_rows(L), pitch of mma_rows(span))``.

    The masked score, target token by source token, on its way from the readout
    orientation to the forcing one. Both modes are rounded up: the target token is
    the M mode of the GEMM that writes it and the K mode of the GEMM that reads it,
    and the source token is the read's M mode, which the transposed
    ``ldmatrix`` needs a whole atom tile of whatever the source-token block is.

    Args:
        chunk: ``L``.
        span: Tokens of the source-token block.
    """
    return operand_tile(mma_rows(chunk), mma_rows(span))


def state_tile(rows: int, dim: int) -> Tile:
    """Chunk-start state or increment cotangent, ``(P, pitch)``.

    ``P`` is a K mode of both GEMMs that read these, never an M mode.

    Args:
        rows: ``P``.
        dim: ``3N``.
    """
    return operand_tile(rows, dim)


def out_tile(chunk: int, rows: int) -> Tile:
    """Output cotangent tile, ``(mma_rows(L), pitch)``.

    Args:
        chunk: ``L``.
        rows: ``P``.
    """
    return operand_tile(mma_rows(chunk), rows)


def gradient_tile(rows: int, dim: int) -> Tile:
    """Float32 tile a vector gradient passes through, ``(rows, pitch)``.

    Float32 because it is not an operand. It is a gradient on its way to ``dB``,
    ``dC`` and the transition parameters, and the reference rounds that once, at
    the store; a second rounding here would double the error on every term it
    feeds, including the two float32 outputs.

    Args:
        rows: Rows to allocate.
        dim: ``3N``.
    """
    return fp32_tile(rows, dim)


class Arena(NamedTuple):
    """Float32-word offsets of the phase-shared tiles inside the one arena.

    The tiles below overlap in address and not in time. The three float32 tiles
    come first and alias nothing: two of them are live across the whole fold and
    the third carries one tap. The five operand tiles follow, and the readout
    gradient of the epilogue aliases all five, none being live when it is.

    ``state`` holds the chunk-start state through the offset contraction and the
    increment cotangent for the rest of a head's pass. One tile rather than two:
    the two have the same extents, neither is read while the other is being
    written, and the barrier that separates them is the one the source-token loop
    needs anyway.

    ``summed`` spans no words at ``fold == 1``, where the readout gradient goes
    straight to global and nothing reads the tile; its offset then aliases
    ``out``.

    No further pair is disjoint. Every region below except ``readout`` is live across
    the source-token loop, so the arena's extent is that loop's live set, and the
    module docstring carries what the remaining 2,304 B of the resident set would buy.

    Attributes:
        forcing: The float32 forcing gradient, summed over taps, blocks and the
            fold. Row ``t + 1`` is token ``t`` and row 0 is the row that crosses
            the chunk boundary.
        tapped: The float32 forcing gradient of one tap, the GEMM's own output, and
            the narrowed score of that same tap, which the forcing GEMM has finished
            reading before the gradient it writes exists. The region is the wider of
            the two.
        summed: The float32 readout gradient summed over the fold.
        out: The output cotangent, ``dy``.
        state: The chunk-start state, then the increment cotangent in the
            chunk-local frame.
        raw: The raw forcing tile, restaged once per source-token block.
        forced: The rotated forcing tile, rebuilt once per tap.
        input: The shifted ``U`` tile.
        readout: The float32 readout gradient of one head. Epilogue only.
        words: Float32 words the arena spans.
    """

    forcing: int
    tapped: int
    summed: int
    out: int
    state: int
    raw: int
    forced: int
    input: int
    readout: int
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
    chunk: int, rows: int, dim: int, fold: int, span: int, itemsize: int = 2
) -> Arena:
    """Lay the phase-shared tiles out in one allocation.

    Every tile that spans the state width spans :func:`lane_block` of it, so this is
    flat in ``3N`` above the first lane tile.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        span: Source-token block, from :func:`vblock`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    tile = lane_block(dim)
    forcing = _words(gradient_tile(chunk + 1, tile), 4)
    tapped = max(
        _words(gradient_tile(span, tile), 4),
        _words(score_tile(chunk, span), itemsize),
    )
    summed = _words(gradient_tile(chunk, tile), 4) if fold > 1 else 0
    out = _words(out_tile(chunk, rows), itemsize)
    state = _words(state_tile(rows, tile), itemsize)
    raw = _words(shifted_tile(span, tile), itemsize)
    forced = _words(forced_tile(span, tile), itemsize)
    inp = _words(shifted_tile(mma_rows(span), rows), itemsize)
    base = forcing + tapped + summed
    return Arena(
        forcing=0,
        tapped=forcing,
        summed=forcing + tapped,
        out=base,
        state=base + out,
        raw=base + out + state,
        forced=base + out + state + raw,
        input=base + out + state + raw + forced,
        readout=base,
        words=base
        + max(
            out + state + raw + forced + inp,
            _words(gradient_tile(mma_rows(chunk), tile), 4),
        ),
    )


def vector_smem_bytes(
    chunk: int,
    rows: int,
    dim: int,
    fold: int,
    span: int,
    itemsize: int = 2,
    warp_groups: int = 1,
) -> int:
    """Shared memory the kernel allocates, in bytes.

    The same tiles :func:`chunk_vector_bwd_kernel` allocates, in the same order.
    Computed from the layouts, so there is one description of the budget.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        span: Source-token block, from :func:`vblock`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
        warp_groups: Warp groups of the tiling, which is the only extent the block
            width moves: :func:`offset_tile` takes a row per group and every other
            tile is flat in the width. ``4 * L`` bytes per group past the first.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (offset_tile(chunk, warp_groups), 4),
            (scalar_tile(chunk), 4),
            (quad_table_tile(chunk, 3), 4),
            (row_tile(chunk), 4),
            (readout_tile(chunk, lane_block(dim)), itemsize),
            (
                Tile(
                    (arena(chunk, rows, dim, fold, span, itemsize).words,),
                    (1,),
                ),
                4,
            ),
        ]
    )


def vblock(chunk: int, rows: int, dim: int, fold: int, itemsize: int = 2) -> int:
    """Source-token block: one M tile of the atom, or half of one to fit.

    ``min(L, MMA_TILE_M)`` is the block every mode of every GEMM is happiest at,
    and it is taken wherever the budget holds it. Where it does not, the block
    halves once, which is the only other candidate: four of the nine regions the
    arena holds scale with the block, and only two of those four keep scaling below
    half an M tile, since the other two round the block back up to one. At
    ``L 64/P 48/3N 48`` and fold one the first halving buys 11,264 B and a second
    would buy 3,584 B for a fourth pass over ``U``. Both candidates divide ``L`` and
    are multiples of the atom's K and N modes, which is what
    :func:`slinoss.ops.so3ssd.cute.guard.check_extents` and
    :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` require.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``.
        itemsize: Bytes per operand element.

    Returns:
        The block. A shape that fits at neither candidate is refused by
        :func:`slinoss._cute.assert_smem_fits`, not run at a third.
    """
    span = min(chunk, MMA_TILE_M)
    floor = min(chunk, MMA_TILE_M // 2)
    if span > floor:
        budget = vector_smem_bytes(chunk, rows, dim, fold, span, itemsize)
        if budget > smem_capacity():
            span = floor
    return span


def vector_splits(fold: int, splits: int | None = None) -> int:
    """Partial depth of the head sum: shards the fold ``H // G`` is cut into.

    The whole fold by default, one head to a block. The two costs of the sum look
    like they trade -- a block that walks more heads spills, a sum cut into more
    shards writes more partials -- but they do not, because the spill does not
    follow the trip count. One rolled iteration is enough to sink the accumulators
    to local memory, and every head then pays that traffic whatever the loop's
    extent, so every depth between the two ends carries the full spill and its own
    partials as well. Measured at ``B 4 H 18 T 2048 L 64 P 64 3N 240 G 1`` and the
    shipped width, event medians of three, clocks unlocked, MB the workspace:

        depth      1       2       3       6       9      18
        us     4,301.8 4,165.6 4,033.5 4,160.0 4,262.9 3,493.9
        MB        0.00   15.97   23.96   47.92   71.88  143.77

    So the depth is the fold unless a caller has a reason of its own, and the
    workspace that costs is linear in it. The full depth is the only one that clears
    the spill, and it wins by 807.9 us over the depth that carries no workspace at
    all; the four between the ends span 5.7% and none of them reaches either end.
    The ordering is set by the spill and not by the workspace, so narrowing the
    partial cannot reorder it, and the depth that deletes the workspace is the one
    that also divides the grid by the fold.

    Args:
        fold: ``H // G``, the heads sharing a group.
        splits: The depth. ``None`` takes the fold.

    Returns:
        The depth. One where the fold is one, and never above the fold.

    Raises:
        ValueError: If ``splits`` is not a positive divisor of the fold. A depth that
            does not divide it would leave a block walking a ragged head count and
            the reduction reading rows no producer wrote.
    """
    if splits is None:
        return fold
    if splits < 1 or fold % splits:
        raise ValueError(f"splits must be a positive divisor of the fold {fold}")
    return splits


def partial_bytes(
    bsz: int,
    groups: int,
    seqlen: int,
    chunks: int,
    dim: int,
    splits: int,
    itemsize: int,
) -> int:
    """Device workspace the head-sum partials occupy, in bytes.

    Args:
        bsz: ``B``.
        groups: ``G``.
        seqlen: ``T``.
        chunks: ``C``.
        dim: ``3N``.
        splits: Partial depth, from :func:`vector_splits`.
        itemsize: Activation dtype width, which the two vector partials carry.

    Returns:
        Bytes, zero at depth one where the three outputs are written directly. The
        depth multiplies the row axis of ``dB``, ``dC`` and the carry. The two
        vectors follow their output's width; the carry is a float32 output and its
        partial follows it.
    """
    if splits == 1:
        return 0
    rows = bsz * groups * splits * dim
    return itemsize * rows * 2 * seqlen + 4 * rows * chunks


PARTIAL_REQUEST_BYTES: int = 4
"""Bytes one thread loads from one partial per shard in :func:`vector_reduce_kernel`.

The closure is request-rate bound and not sector bound. One element per thread puts
32 elements in a warp's request, which is a whole 128-byte line at float32 and half
of one at bfloat16 for the same instruction count, and the narrowed partial cost the
kernel its class that way: 680.5 GB/s at float32 against 472.9 GB/s at bfloat16 for
the same loop. Four bytes per thread holds the request at a full line whatever the
partial's width."""


def partial_pack(itemsize: int, dim: int) -> int:
    """Elements one thread loads per partial per shard, at least one.

    Args:
        itemsize: Partial dtype width in bytes.
        dim: ``3N``, the mode the vector is cut from.

    Returns:
        ``PARTIAL_REQUEST_BYTES // itemsize`` where that divides ``3N``, one otherwise.
        Divisibility is the whole condition: the width times the itemsize is the
        request itself, so a ``3N`` the width divides puts every row of the partial at
        a multiple of the access as well. ``3N`` is a multiple of 48 and the partial is
        two or four bytes wide, so the fallback is for a width no dtype table carries
        rather than for a shape the operator runs.
    """
    pack = max(1, PARTIAL_REQUEST_BYTES // itemsize)
    return 1 if dim % pack else pack


class Slots(NamedTuple):
    """One output and the slot rows the grid writes it through.

    Two of this kernel's axes are grid axes over the terms of a sum: the lane tile
    over ``dtrans`` and ``dK``, which are sums over lanes, and the head shard over
    ``dB``, ``dC`` and the carry, which are sums over the heads of a group. The blocks
    holding the terms are concurrent and none can see the others' partials. Each such
    output gains a slot axis immediately outside its row axis: a block writes row
    ``slot * rows + local``, and one second launch sums the ``slots`` copies of a row
    into the output's own.

    At ``slots == 1`` there is no buffer and ``dest`` is the output. The row index is
    the same expression either way, ``slot`` being zero, so the kernel body does not
    know which mode it is in and neither mode carries a branch.

    Outside the row axis rather than inside it, because the reduction's slab count is
    what its grid puts on the y axis and that axis stops at 65,535. Inside, one slab
    is one row: at ``B 4 H 18 T 2048`` that is 147,456 slabs of four columns and the
    launch is refused. Outside, a slab is a head: 72 slabs of ``T*4`` columns, the
    grid over columns carries the parallelism, and the block count rises rather than
    the slab count.

    Which second launch closes a buffer follows its output's layout, not its axis.
    ``dtrans`` and ``dK`` are contiguous, so :func:`close_slots` views them as
    ``(S, R, W)`` and :func:`slinoss._reduce.reduce_partials` sums the rows.
    ``dB`` and ``dC`` are bands that may be pitched in their last mode, where no such
    view exists, so :func:`vector_reduce` sums them by mode instead; it takes the
    carry with them, the three sharing one launch.

    Attributes:
        dest: What the kernel writes. The output at one slot, a partial buffer at the
            output's own dtype above one.
        slots: Copies of each row, one per lane tile or one per head shard.
        out: The output the sum closes onto, or None at one slot.
        slabs: ``S`` of the reduction, the modes before the slot axis.
        width: ``W`` of the reduction, the row axis and everything after it.
    """

    dest: Tensor
    slots: int
    out: Tensor | None
    slabs: int
    width: int


def open_slots(out: Tensor, slots: int, axis: int = -2) -> Slots:
    """Allocate the slot rows one output needs, or none.

    Args:
        out: The output. Its shape is read, not its layout: the buffer this allocates
            is contiguous whether the output is.
        slots: Copies of each row, the lane tile count or the partial depth.
        axis: Row axis of ``out``, the mode the slot count multiplies. ``-2`` where
            one trailing mode follows it, ``-3`` where two do.

    Returns:
        :class:`Slots`. ``torch.empty``, never zeroed: the kernel writes every
        element of every slot row it allocates, one block per row.
    """
    shape = [int(extent) for extent in out.shape]
    axis %= len(shape)
    slabs, width = 1, 1
    for extent in shape[:axis]:
        slabs *= extent
    for extent in shape[axis:]:
        width *= extent
    if slots == 1:
        return Slots(dest=out, slots=1, out=None, slabs=slabs, width=width)
    # The output's own width, not float32. A partial is one term of a sum the closure
    # rounds again on its own store, so a wider partial buys a rounding the output
    # cannot keep, and at the widest configured state the two vector buffers are
    # 141.56 MB of the launch's write traffic and all of the closure's read traffic.
    held = torch.empty(
        *shape[:axis],
        shape[axis] * slots,
        *shape[axis + 1 :],
        dtype=out.dtype,
        device=out.device,
    )
    return Slots(dest=held, slots=slots, out=out, slabs=slabs, width=width)


def close_slots(held: Slots) -> Tensor:
    """Sum a slot buffer onto its output, in one launch, or return the output.

    The output must be contiguous, the reduction being a row view of it. A pitched
    destination goes through :func:`vector_reduce`.

    Args:
        held: From :func:`open_slots`.

    Returns:
        The output.
    """
    if held.out is None:
        return held.dest
    reduce_partials(
        held.dest.view(held.slabs, held.slots, held.width),
        out=held.out.view(held.slabs, held.width),
    )
    return held.out


def _lane_view(src: cute.Tensor, joff: cutlass.Int32) -> cute.Tensor:
    """The same global tensor with its lane origin moved to a lane tile.

    Every tensor this is applied to carries ``3N`` as its last mode at unit stride,
    so a column offset is a pointer offset and the layout is the one the tensor
    arrived with. The staging helpers and the stores then index the lane tile's own
    columns, and none of them learns that the state is wider.

    Undecorated, so the offset folds into the addresses the trace already builds.

    Args:
        src: Global tensor whose last mode is ``3N``.
        joff: First column of the lane tile.
    """
    return cute.make_tensor(src.iterator + joff, src.layout)


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


def _sum_over_n(value: Scalar) -> Scalar:
    """Sum one accumulator element over the four lanes that share its row.

    The atom gives the four lanes of an aligned quad the same accumulator row and
    disjoint columns, so two butterfly rounds leave that row's partial column sum in
    all four. The atom's C layout is per warp, so the quad is the same four lanes at
    every block width.

    Rows are disjoint across quads and across the warps of one warp group, so within
    a group the read-modify-write that follows is by one thread per row and needs no
    barrier. Across warp groups a row is shared: the M mode is not partitioned by the
    N atoms, so a tiling with more than one group gives every row one quad leader per
    group. Each writes its own row of :func:`offset_tile` and the rows are summed
    once, which is a reduction order the widths do not share.

    Args:
        value: The lane's contribution.
    """
    value = value + shuffle_xor(value, 1)
    return value + shuffle_xor(value, 2)


def _sum_over_lanes(vals: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
    """Sum a tuple of floats over the :data:`LANE_GROUP` lanes of one token.

    Undecorated: the round count is compile-time and the loop is unrolled during
    the trace.

    Args:
        vals: One value per component, this lane's partial.

    Returns:
        The group's sum, in every lane of the group.
    """
    out = vals
    reach = 1
    while reach < LANE_GROUP:
        out = tuple(v + shuffle_xor(v, reach) for v in out)
        reach *= 2
    return out


def _spread(vals: tuple[Scalar, ...], lane: cutlass.Int32) -> Scalar:
    """Component ``lane`` of a tuple every lane of the group holds identically.

    A select chain rather than an indexed fragment: a dynamic index into a fragment
    puts the fragment in local memory. Undecorated, so the chain length is resolved
    during the trace.

    Args:
        vals: One value per component, the same in every lane of the group, as
            :func:`_sum_over_lanes` leaves it. A caller with more components than the
            group is wide passes one compile-time slice a round, so the chain is never
            longer than the group and a one-component round costs no select at all.
        lane: Lane within the group. A lane past the last component takes component
            zero, so the caller must predicate it off.

    Returns:
        ``vals[lane]``.
    """
    held = vals[0]
    for k in range(1, len(vals)):
        held = select(lane == k, vals[k], held)
    return held


def _mat_at(stable: cute.Tensor, slot: int, token: cutlass.Int32) -> Mat3:
    """One transform-table entry as a 3x3, row-major.

    Three 16-byte shared loads, not nine scalar ones. :data:`TABLE_PITCH` pads the
    entry to three whole segments, so the row divides and the entry is aligned; the
    claim is restated on the sliced iterator for the reason
    :func:`slinoss.ops.so3ssd.cute.table._paired` gives, a tile arriving as a
    parameter reporting one element whatever its allocation asked for.

    Undecorated, so the slice and the retile are trace-time algebra and every
    fragment index is compile-time, which is what keeps the fragment in registers.
    A plain ``range``, for the reason the closing loop's comprehension gives: the
    preprocessor rewrites ``range_constexpr`` only inside a decorated body, so here
    it would reach the runtime stub and raise.

    Conflict-free at every map the callers use. A load at vector width is serviced in
    four phases of eight threads, so the unit is the segment and the modulus is 8
    rather than 32: eight threads on consecutive tokens take segment ``3 * token + q
    mod 8``, a bijection, and threads sharing a token share an address and broadcast.

    Args:
        stable: ``(mats, L, TABLE_PITCH)`` float32 table from
            :func:`quad_table_tile`, 16-byte aligned.
        slot: Table slot. Compile-time.
        token: Chunk-local token, already bounded by ``L``.

    Returns:
        Entries 0 through 8. The padding words ride the third load and are dropped.
    """
    entry = stable[slot, token, None]
    quads = cute.zipped_divide(
        cute.make_tensor(entry.iterator.align(SMEM_SEGMENT), entry.layout),
        (TABLE_QUAD,),
    )
    frag = cute.make_fragment((TABLE_PITCH // TABLE_QUAD, TABLE_QUAD), cutlass.Float32)
    for quad in range(TABLE_PITCH // TABLE_QUAD):
        cute.autovec_copy(quads[(None, quad)], frag[(quad, None)])
    return (
        frag[0, 0],
        frag[0, 1],
        frag[0, 2],
        frag[0, 3],
        frag[1, 0],
        frag[1, 1],
        frag[1, 2],
        frag[1, 3],
        frag[2, 0],
    )


def _row_pairs(threads: int, span: int, pairs: int) -> int:
    """Threads one row of a rotation pass takes.

    The largest divisor of ``pairs`` reached by doubling that keeps the whole block
    inside one row pass, so a thread's pairs tile the row and the block covers
    ``threads // per_row`` rows a step. It lands on ``threads // span`` exactly
    wherever that quotient is such a divisor, which is every shape :func:`vblock`
    returns against a block of 128 or 256 threads except the shortest span, where the
    block is wider than the row count and the ragged arm runs.

    Undecorated, and taking plain ints rather than tensors, so the search runs during
    the trace and no loop reaches the IR. A ``while`` inside a ``cute.jit`` body is
    rewritten into a dynamic loop and its result stops being compile-time.

    Args:
        threads: Block width.
        span: Rows to fill.
        pairs: Lane pairs of one row.
    """
    per_row = 1
    while pairs % (per_row * 2) == 0 and per_row * 2 * span <= threads:
        per_row *= 2
    return per_row


def _mat_of(mats: tuple[Mat3, ...], step: int) -> Mat3:
    """One matrix of a per-step tuple.

    A ``range_constexpr`` variable carries no static type, so indexing a tuple with
    it directly widens to the slice result. This pins the index to an int and the
    result to one matrix.

    Args:
        mats: One matrix per step of a row pass.
        step: Step of the pass. Compile-time.
    """
    return mats[step]


def _vec_at(src: cute.Tensor, row: cutlass.Int32, col: cutlass.Int32) -> Vec3:
    """One lane's 3-vector of a shared tile, widened to float32.

    Args:
        src: Shared tile.
        row: Row.
        col: First of the three columns.
    """
    elem = src.element_type
    return (
        widen(src[row, col], elem),
        widen(src[row, col + 1], elem),
        widen(src[row, col + 2], elem),
    )


@cute.jit
def _fill_zero(
    dst: cute.Tensor, total: cutlass.Constexpr, tid: cutlass.Int32, threads: int
) -> None:
    """Zero a dense shared tile, padding included.

    Args:
        dst: Shared float32 tile whose storage is dense.
        total: Elements the tile spans, padding included. Compile-time.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
    """
    flat = cute.make_tensor(dst.iterator, cute.make_layout((total,), stride=(1,)))
    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if cutlass.const_expr(total % threads == 0):
            flat[i] = 0.0
        else:
            if i < total:
                flat[i] = 0.0


@cute.jit
def _rotate_rows(
    src: cute.Tensor,
    dst: cute.Tensor,
    stable: cute.Tensor,
    tid: cutlass.Int32,
    nbase: cutlass.Int32,
    slot: cutlass.Constexpr,
    shift: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Transform a shifted shared tile by one table slot into another.

    ``dst[r] = A_slot[nbase + r] src[r + shift]``, which is what
    :func:`slinoss.ops.so3ssd.cute.table.stage_rotated` writes from global. Reading
    the raw tile the tap cotangent needs anyway costs nine FMA a lane and saves a
    pass over ``B`` per tap.

    A step carries :data:`slinoss.ops.so3ssd.cute.table.LANE_PAIR` adjacent lanes,
    the unit :func:`slinoss.ops.so3ssd.cute.table.stage_rotated` already pairs from
    global. The pair's six components are one contiguous twelve-byte run at either
    width, so the read and the write are three paired accesses each rather than six
    scalars.

    A row is held by ``per_row`` threads and a thread holds one row for the whole
    call, covering ``pairs / per_row`` of its lane pairs, so the row's nine table
    words are read once a row rather than once a step and are applied to every pair
    the thread holds.

    Rows of ``src`` past the chunk's valid tokens are already zero, so the rows an
    M extent was rounded up by stay zero and no consumer needs a predicate.

    Args:
        src: Operand-dtype shifted tile, row ``j`` holding token ``nbase + j - 1``.
        dst: Operand-dtype tile of at least ``span`` rows, written.
        stable: ``(mats, L, TABLE_PITCH)`` float32 transform table.
        tid: Thread index within the block.
        nbase: First chunk-local token of the run.
        slot: Table slot. Compile-time.
        shift: Row offset into ``src``, which is the tap index: the previous tap
            takes the row before the token and the current tap the token's own.
            Compile-time.
        threads: Block width. Compile-time.
        span: Rows of ``dst`` to fill. Compile-time.
        lanes: ``N``. Compile-time.

    Invariants:
        ``lanes`` is even and both tiles are pitched by :func:`smem_pitch`, which is
        what the pair rests on. The tail predicate is reachable, at the one legal
        shape whose span is shorter than the rows one pass covers.
    """
    raw = src.element_type
    pairs = lanes // LANE_PAIR
    # One thread to a row, walking that row's pairs, so the row's table entry is read
    # once and held in registers across them. A flat pass over the pairs gave a thread
    # a different row at each step and reread the nine words there. The entry is what
    # the step count multiplies; the copies and the stores are one per pair either way.
    #
    # Every legal shape but the shortest span puts :func:`_row_pairs` on
    # ``threads // span`` exactly, so the row loop has a trip count of one and no
    # thread idles. The ragged arm is what that span takes.
    per_row = _row_pairs(threads, span, pairs)
    rows_per_pass = threads // per_row
    exact = span % rows_per_pass == 0
    col0 = 3 * (tid % per_row)

    words = paired(dst)
    source = paired(src)
    frag = cute.make_fragment((1, LANE_PAIR), dst.element_type)
    held = cute.make_fragment((3, LANE_PAIR), raw)

    for step in cutlass.range_constexpr(-(-span // rows_per_pass)):
        r = tid // per_row + step * rows_per_pass
        # Clamped rather than branched: a row past the run reads real data whose every
        # use below is predicated away.
        rs = r
        if cutlass.const_expr(not exact):
            rs = cutlass.min(r, span - 1)
        mat = _mat_at(stable, slot, nbase + rs)
        for rep in cutlass.range_constexpr(pairs // per_row):
            col = col0 + 3 * rep * per_row
            for k in cutlass.range_constexpr(3):
                cute.autovec_copy(
                    source[(None, (rs + shift, col + k))], held[(k, None)]
                )
            got = tuple(
                widen(held[j // LANE_PAIR, j % LANE_PAIR], raw)
                for j in range(3 * LANE_PAIR)
            )
            out = mat3_matvec(mat, (got[0], got[1], got[2])) + mat3_matvec(
                mat, (got[3], got[4], got[5])
            )
            if cutlass.const_expr(exact):
                store_pair(words, frag, rs, col, out)
            else:
                if r < span:
                    store_pair(words, frag, rs, col, out)


@cute.jit
def _tap_epilogue(
    gdtap: cute.Tensor,
    sdb: cute.Tensor,
    sbrot: cute.Tensor,
    sb: cute.Tensor,
    ssum: cute.Tensor,
    stable: cute.Tensor,
    acrow: tuple[Mat3, ...],
    stap: cute.Tensor,
    strans: cute.Tensor,
    srow: cute.Tensor,
    sdw: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    nbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    jbase: cutlass.Int32,
    tap: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Turn one tap's finished forcing gradient into ``dB``, ``dK`` and two sums.

    Per token and lane, with ``atap`` the tap's table slot and ``ac`` the readout
    slot::

        dbs      = atap^T dbnow                 into the forcing sum
        rotation += outer(dbnow, brot)          into the nine-float scratch
        tap, w    = tap_matrix_vjp(sum_n outer(ac^T dbnow, b), tap, w)

    The rotation term is the collapsed form: ``outer(dbnow, ac^T bnow) ac`` with
    the trailing ``ac`` deferred to the one place that applies it, so no raw
    readout vector is read. The tap term does not collapse, which is the only
    reason the raw forcing tile is staged.

    ``brot`` is recomputed from ``atap`` and the raw vector, both already in
    registers here, rather than read back out of the tile :func:`_rotate_rows`
    wrote. The tile itself stays, because the GEMM takes it as an operand.

    The forcing sum is indexed by token rather than by row of the run: row ``t + 1``
    is token ``t``, so the current tap lands one row above the previous tap's and
    the previous tap of the chunk's first token lands on row 0, which is the carry.

    Args:
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer, written at
            this tap.
        sdb: ``(span, pitch)`` float32 forcing gradient, the GEMM's output.
        sbrot: ``(span, pitch)`` operand-dtype rotated forcing tile. Taken for its
            element type, not read: the rotated vector is recomputed.
        sb: ``(span + 1, pitch)`` operand-dtype raw forcing tile.
        ssum: ``(L + 1, pitch)`` float32 forcing sum, accumulated.
        stable: ``(mats, L, TABLE_PITCH)`` float32 transform table.
        acrow: One transposed readout entry per step of the row pass, for the row
            this thread holds at that step. Read by the caller because the row does
            not depend on the tap, so the pair of taps reads it once.
        stap: ``(8, L)`` float32 tap parameters, component-major.
        strans: ``(4, L)`` float32 ``(w, ls)``, component-major.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated.
        sdw: ``(4, L)`` float32 rotation-vector cotangent, accumulated.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        nbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        jbase: First row of this block's slot, ``jstep * T``. Zero at one lane tile,
            where the destination is ``dK`` itself.
        tap: 0 for the previous tap and 1 for the current, the order ``K`` packs
            them in. Compile-time.
        threads: Block width. Compile-time.
        span: Tokens of the run. Compile-time.
        lanes: Lanes of the lane tile. Compile-time.
    """
    slot = TABLE_AP if cutlass.const_expr(tap == 0) else TABLE_AN
    # The previous tap's gradient belongs to the token before its own, which is one
    # row of the shifted tiles back, so the row offset is the tap index itself.
    shift = tap
    per_pass = threads // LANE_GROUP
    exact = span % per_pass == 0
    lane = tid % LANE_GROUP
    zero = cutlass.Float32(0.0)

    for step in cutlass.range_constexpr(-(-span // per_pass)):
        r = tid // LANE_GROUP + step * per_pass
        # Clamped rather than branched: a row past the run reads real data whose
        # every use below is predicated away.
        rs = cutlass.min(r, span - 1)
        token = nbase + rs
        inside = r < span
        gsum = tuple(zero for _ in range(9))
        msum = tuple(zero for _ in range(9))
        act = _mat_of(acrow, step)
        atap = _mat_at(stable, slot, token)
        atapt = mat3_transpose(atap)
        rotated = sbrot.element_type
        for rep in cutlass.range_constexpr(lanes // LANE_GROUP):
            col = 3 * (lane + rep * LANE_GROUP)
            dvec = (sdb[rs, col], sdb[rs, col + 1], sdb[rs, col + 2])
            bvec = _vec_at(sb, rs + shift, col)
            # What _rotate_rows stored, recomputed from the table entry and the raw
            # vector this thread already holds: nine FMA in place of a pass over the
            # rotated tile. The round trip through the operand dtype is what makes
            # this the stored bits rather than a value near them.
            rot = mat3_matvec(atap, bvec)
            brot = (
                widen(narrow(rot[0], rotated), rotated),
                widen(narrow(rot[1], rotated), rotated),
                widen(narrow(rot[2], rotated), rotated),
            )
            gsum = mat3_add(gsum, mat3_outer(dvec, brot))
            msum = mat3_add(msum, mat3_outer(mat3_matvec(act, dvec), bvec))
            out = mat3_matvec(atapt, dvec)
            if cutlass.const_expr(exact):
                for j in cutlass.range_constexpr(3):
                    ssum[token + shift, col + j] += out[j]
            else:
                if inside:
                    for j in cutlass.range_constexpr(3):
                        ssum[token + shift, col + j] += out[j]
        gsum = _sum_over_lanes(gsum)
        msum = _sum_over_lanes(msum)
        keep = lane == 0
        if cutlass.const_expr(not exact):
            keep = keep & inside
        # One word a lane, a group of words a round. The butterfly leaves the whole
        # nine in every lane of the group, so the read-modify-write costs one access
        # on as many lanes as a round is wide instead of nine on one, and the words a
        # round touches are consecutive.
        for word in cutlass.range_constexpr(-(-ROW_WORDS // LANE_GROUP)):
            base = word * LANE_GROUP
            rows = lane < ROW_WORDS - base
            if cutlass.const_expr(not exact):
                rows = rows & inside
            held = _spread(gsum[base : base + LANE_GROUP], lane)
            if rows:
                srow[token, base + lane] += held
        if keep:
            dtap, dw = tap_matrix_vjp(
                msum,
                (
                    stap[4 * tap, token],
                    stap[4 * tap + 1, token],
                    stap[4 * tap + 2, token],
                ),
                (strans[0, token], strans[1, token], strans[2, token]),
            )
            for j in cutlass.range_constexpr(3):
                sdw[j, token] += dw[j]
            if token < valid:
                krow = jbase + t0 + token
                for j in cutlass.range_constexpr(3):
                    gdtap[bidx, hidx, krow, tap, j] = dtap[j]
                # Lane 3 of K is a hard zero in the forward, so it is one here.
                gdtap[bidx, hidx, krow, tap, 3] = zero


@cute.jit
def _readout_epilogue(
    gdc: cute.Tensor,
    sdc: cute.Tensor,
    ssum: cute.Tensor,
    scrot: cute.Tensor,
    stable: cute.Tensor,
    srow: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    sbase: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    fold: cutlass.Constexpr,
) -> None:
    """Turn one head's finished readout gradient into ``dC`` and a rotation sum.

    Per token and lane, ``dc = ac^T dcrot`` and ``rotation += outer(dcrot,
    crot)``, the same collapsed form the tap epilogue accumulates into.

    One head writes its shard's ``dC`` row in the destination's own dtype. A fold
    above one accumulates in float32 instead, because a shard's ``dC`` is a sum over
    the heads it walks and the reference rounds the head sum once.

    Args:
        gdc: ``(B,G,splits*T,3N)`` ``dC`` or its shard buffer, written when ``fold``
            is one.
        sdc: ``(mma_rows(L), pitch)`` float32 readout gradient.
        ssum: ``(L, pitch)`` float32 readout sum over the fold, accumulated when
            ``fold`` is above one and untouched otherwise.
        scrot: ``(mma_rows(L), pitch)`` operand-dtype rotated readout.
        stable: ``(mats, L, TABLE_PITCH)`` float32 transform table.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated.
        bidx: Batch index.
        gidx: Group index.
        sbase: First row of this block's shard, ``shard * T``. Zero at one shard,
            where the destination is the output.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        lanes: ``N``. Compile-time.
        fold: Heads one block walks. Compile-time.
    """
    out = gdc.element_type
    per_pass = threads // LANE_GROUP
    exact = chunk % per_pass == 0
    lane = tid % LANE_GROUP
    zero = cutlass.Float32(0.0)

    for step in cutlass.range_constexpr(-(-chunk // per_pass)):
        token = tid // LANE_GROUP + step * per_pass
        ts = cutlass.min(token, chunk - 1)
        inside = token < chunk
        gsum = tuple(zero for _ in range(9))
        act = mat3_transpose(_mat_at(stable, TABLE_AC, ts))
        for rep in cutlass.range_constexpr(lanes // LANE_GROUP):
            col = 3 * (lane + rep * LANE_GROUP)
            dvec = (sdc[ts, col], sdc[ts, col + 1], sdc[ts, col + 2])
            gsum = mat3_add(gsum, mat3_outer(dvec, _vec_at(scrot, ts, col)))
            dc = mat3_matvec(act, dvec)
            keep = ts < valid
            if cutlass.const_expr(not exact):
                keep = keep & inside
            if keep:
                for j in cutlass.range_constexpr(3):
                    if cutlass.const_expr(fold == 1):
                        gdc[bidx, gidx, sbase + t0 + ts, col + j] = narrow(dc[j], out)
                    else:
                        ssum[ts, col + j] += dc[j]
        gsum = _sum_over_lanes(gsum)
        # One word a lane, a group of words a round, as in :func:`_tap_epilogue`.
        for word in cutlass.range_constexpr(-(-ROW_WORDS // LANE_GROUP)):
            base = word * LANE_GROUP
            rows = lane < ROW_WORDS - base
            if cutlass.const_expr(not exact):
                rows = rows & inside
            held = _spread(gsum[base : base + LANE_GROUP], lane)
            if rows:
                srow[ts, base + lane] += held


@cute.kernel
def chunk_vector_bwd_kernel(
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
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    tiled_mma: cute.TiledMma,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
) -> None:
    """Differentiate one chunk's rowwise vectors and transition parameters.

    One block per ``(chunk, head shard, lane tile, batch, group)``, walking the
    ``fold`` heads of its shard. Everything a head owns alone is rebuilt per head; the
    two vector sums and the carry outlive the fold and belong to the block's own shard
    and lane tile.

    Two outputs a block does not own alone, and they part company on which axis takes
    them. ``dtrans`` and ``dK`` are sums over lanes, so a lane tile past the first
    writes its own slot row; ``dB``, ``dC`` and the carry are sums over the heads of a
    group, so a shard past the first writes its own. Both row offsets are zero at one
    copy, where the destination is the output itself.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gu: ``(B,H,T,P)`` operand-dtype forcing input.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 ``(kr, g, h, 0)`` per tap.
        gb: ``(B,G,T,3N)`` operand-dtype forcing vectors.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``.
        gc: ``(B,G,T,3N)`` operand-dtype readout vectors.
        gdinc: ``(B,H,C,P,3N)`` float32 increment cotangent, global frame.
        gz: ``(B,H,C,P,3N)`` float32 chunk-start state.
        gdlp: ``(B,H,C,L)`` float32 diagonal and increment half of the log-scale
            cotangent, from the chunk-input stage.
        gdrot: ``(B,H,C,3,3)`` float32 closing-rotation cotangent, row-major, from
            the chunk-input stage.
        gdscale: ``(B,H,C)`` float32 closing-scale cotangent, from the chunk-input
            stage.
        gdb: ``(B,G,splits*T,3N)`` ``dB`` or its shard buffer, written, under the
            output's own dtype either way. At one shard it is the output; above one it
            is the partial :func:`vector_reduce` sums.
        gdc: ``(B,G,splits*T,3N)`` ``dC`` or its shard buffer, under the contract of
            ``gdb``.
        gcarry: ``(B,G,splits*C,3N)`` float32 carry or its shard buffer, written with
            the forcing gradient of the token before the chunk's first.
        gdtrans: ``(B,H,tiles*T,4)`` float32 ``dtrans`` or its slot buffer, written.
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer, written.
        seqlen: ``T``. Dynamic.
        chunks: ``C``. Dynamic. The row extent the carry's shard axis multiplies.
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`. Its warp count
            is the block's: warps past the first four go to the tile's N mode, so
            every M extent, every pitch and every staging pass is unchanged and the
            two places the split shows are the readout term's left operand and the
            offset term's owner count.
        threads: Block width, ``32 *`` the tiling's warp count. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        span: Source-token block, :func:`vblock`. Compile-time.
        fold: Heads of one shard, ``H // G // splits``. Compile-time.
        splits: Partial depth of the head sum, from :func:`vector_splits`.
            Compile-time, the grid's first-mode divides being compile-time.
        has_prev: Whether the streaming carry-in was supplied. Compile-time.

    Invariants:
        ``chunk`` is a multiple of ``span`` and of :data:`MMA_TILE_K`, ``dim`` and
        ``rows`` are multiples of :data:`MMA_TILE_N`, ``N`` is a multiple of
        :data:`LANE_GROUP`, and ``dim`` is a multiple of ``3 * LANE_BLOCK``, which
        follows from being a multiple of 3 and of 16. ``L`` and the source-token
        block are the padded modes:
        M is rounded up in shared memory, the rounded rows are zeroed by the
        staging predicate or masked by the score, and every store is predicated.
        ``fold * splits`` divides ``H``, and one head reaches exactly one block, so no
        row of any output is written twice.
    """
    tid, _, _ = cute.arch.thread_idx()
    xidx, bidx, gidx = cute.arch.block_idx()

    tile = lane_block(dim)
    tiles = dim // tile
    lanes = tile // 3
    # Read off the tiling, not off ``threads``: the tiling is what decides how many
    # threads own one accumulator row, and a second derivation of it could disagree.
    wgroups = mma_groups(tiled_mma)
    mpad = mma_rows(chunk)
    spad = mma_rows(span)
    blocks = chunk // span
    last = chunk - 1
    ldv = smem_pitch(tile)
    ldu = smem_pitch(rows)
    lds = smem_pitch(spad)
    ldf = smem_pitch(tile, 4)
    zero = cutlass.Float32(0.0)
    elem = gb.element_type
    out = gdb.element_type

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdrot = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdquat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdw = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(
        cutlass.Float32, offset_tile(chunk, wgroups).layout(), 16
    )
    sdls = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(
        cutlass.Float32, quad_table_tile(chunk, 3).layout(), SMEM_SEGMENT
    )
    srow = smem.allocate_tensor(cutlass.Float32, row_tile(chunk).layout(), 16)
    scrot = smem.allocate_tensor(elem, readout_tile(chunk, tile).layout(), SMEM_SEGMENT)
    space = arena(chunk, rows, dim, fold, span, elem.width // 8)
    base = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((space.words,), stride=(1,)), 16
    )
    sdy = _tile_at(base, space.out, out_tile(chunk, rows), elem)
    sstate = _tile_at(base, space.state, state_tile(rows, tile), elem)
    sb = _tile_at(base, space.raw, shifted_tile(span, tile), elem)
    sbrot = _tile_at(base, space.forced, forced_tile(span, tile), elem)
    su = _tile_at(base, space.input, shifted_tile(spad, rows), elem)
    sdb = _tile_at(base, space.tapped, gradient_tile(span, tile), cutlass.Float32)
    sscore = _tile_at(base, space.tapped, score_tile(chunk, span), elem)
    sdc = _tile_at(base, space.readout, gradient_tile(mpad, tile), cutlass.Float32)
    sumb = _tile_at(
        base, space.forcing, gradient_tile(chunk + 1, tile), cutlass.Float32
    )
    sumc = _tile_at(base, space.summed, gradient_tile(chunk, tile), cutlass.Float32)
    # Row 0 of the offset accumulator, which is where its rows are summed and what the
    # reverse scan reads. The scan takes a dense ``(L,)`` tile and must not learn that
    # the tiling gave the term more than one owner.
    vdlp = cute.make_tensor(sdlp.iterator, cute.make_layout((chunk,), stride=(1,)))

    # Every view a GEMM reads is a layout over a tile that never moves, so all of
    # them are built once and none is per head, per block, per tap or per lane tile.
    vdy = cute.make_tensor(
        sdy.iterator, cute.make_layout((mpad, rows), stride=(ldu, 1))
    )
    vstate = cute.make_tensor(
        sstate.iterator, cute.make_layout((tile, rows), stride=(1, ldv))
    )
    vbrot = cute.make_tensor(
        sbrot.iterator, cute.make_layout((tile, span), stride=(1, ldv))
    )
    vcrot = cute.make_tensor(
        scrot.iterator, cute.make_layout((tile, mpad), stride=(1, ldv))
    )
    # The score in the orientation the forcing GEMM contracts: source token as M,
    # target token as K, so the stride-1 mode is M and the load transposes.
    vscore = cute.make_tensor(
        sscore.iterator, cute.make_layout((spad, mpad), stride=(1, lds))
    )
    # The same tile as the readout GEMM's left operand, target token as M and source
    # token as K. The K extent is the block rather than the block rounded up: the pad
    # columns of the store below are never written, and a K mode reads them into the
    # sum where an M mode only reaches accumulator rows the store drops. Used at a
    # tiling that splits the N mode, where the register reread is not available.
    vscorem = cute.make_tensor(
        sscore.iterator, cute.make_layout((mpad, span), stride=(lds, 1))
    )

    dcacc = mma_acc(tiled_mma, tid, (mpad, tile))
    ccrd = mma_coords(tiled_mma, tid, (mpad, tile))
    dmacc = mma_acc(tiled_mma, tid, (mpad, span))
    mcrd = mma_coords(tiled_mma, tid, (mpad, span))
    dbacc = mma_acc(tiled_mma, tid, (spad, tile))
    bcrd = mma_coords(tiled_mma, tid, (spad, tile))
    # The narrowed score is the A operand of the GEMM that rereads it in place.
    # Fragment and view are built once: the retile is a layout, so nothing here is
    # per-tap work.
    #
    # These accumulators reach registers only when both loops below have a trip count
    # of one. Every accumulator allocation is hoisted to the kernel entry, and a
    # rolled loop between the allocation and its uses defeats register promotion, so
    # each fragment access becomes a local load and a local store. Measured at
    # ``P = 64``, ``L = 64``, span 64, fold one, at 255 registers and 91,344 B of
    # shared memory in both runs: one lane tile moves no local traffic at all, and
    # five lane tiles move 1,892.16 MB per call. The declaration site is not the
    # lever, and moving all four inside both loops leaves the counters unchanged to
    # the sector.
    mfrag = cute.make_fragment_like(dmacc, elem)
    # Built at both widths and read at one. The reread is contiguous in K only where
    # the tiling keeps the tile's N mode whole; past that the wide arm below takes the
    # same score out of ``sscore``, which the transpose stores anyway.
    fa_m = mma_areg(mfrag)

    # Row of :func:`offset_tile` this thread's read-modify-write of the offset term
    # belongs to. Element 0 of the accumulator sits in the tiling's first N tile, so
    # its column names the warp group directly. Taken from the coordinates rather than
    # from ``tid`` so the thread layout of the tiling is not restated here.
    wgroup = 0 if cutlass.const_expr(wgroups == 1) else ccrd[0][1] // MMA_INST[1]

    # The offset accumulator's element indices grouped by row. The atom's C layout
    # gives a thread several columns of one row, and the offset term's destination is
    # per row, so one group is one read-modify-write of :func:`offset_tile` and one
    # exponential rather than one of each per element. Grouped through the offsets
    # because they are trace-time constants where a coordinate is not; the row itself
    # comes from the coordinate of the group's first element, the base being one value
    # for the whole fragment.
    coffsets = mma_offsets(tiled_mma, (mpad, tile))
    crows = tuple(
        tuple(i for i, (m, _) in enumerate(coffsets) if m == row)
        for row in sorted({m for m, _ in coffsets})
    )

    # The lane tile and the head shard are grid axes, not loops. Every accumulator
    # above then sits between its hoisted allocation and its uses with a loop of the
    # shard's heads and nothing else in between, which is what register promotion
    # needs; the divisors are compile-time, so each decode is a multiply and a
    # subtract rather than an integer division.
    #
    # Chunk outermost, then shard, then lane tile: the tiles of one head share every
    # operand with no state extent, which is the traffic the tile count multiplies, and
    # the shards of one chunk share the ``B`` and ``C`` rows. Both sets are dispatched
    # together in this order.
    cidx = xidx // (splits * tiles)
    within = xidx - cidx * (splits * tiles)
    sidx = within // tiles
    jstep = within - sidx * tiles

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    joff = jstep * tile
    # Slot row base of the two outputs that are sums over lanes. The slot axis sits
    # immediately outside the token axis, so this is the whole of the row offset and
    # it is zero at one tile, where the destination is the output itself.
    jbase = jstep * seqlen
    # Shard row base of the three outputs that are sums over heads, on the same
    # convention: the shard axis sits immediately outside the row axis, and the row
    # axis of the carry is the chunk rather than the token.
    sbase = sidx * seqlen
    cbase = sidx * chunks
    first = jstep == 0
    gbj = _lane_view(gb, joff)
    gbprevj = _lane_view(gbprev, joff)
    gcj = _lane_view(gc, joff)
    gdincj = _lane_view(gdinc, joff)
    gzj = _lane_view(gz, joff)
    gdbj = _lane_view(gdb, joff)
    gdcj = _lane_view(gdc, joff)
    gcarryj = _lane_view(gcarry, joff)
    # The chunk-input stage's half of the log-scale cotangent is the head's, not
    # the lane tile's, so a later tile must not add it a second time.
    wlp = cutlass.Float32(1.0)
    if cutlass.const_expr(tiles > 1):
        wlp = select(first, cutlass.Float32(1.0), zero)

    cute.arch.sync_threads()
    _fill_zero(sumb, (chunk + 1) * ldf, tid, threads)
    if cutlass.const_expr(fold > 1):
        _fill_zero(sumc, chunk * ldf, tid, threads)

    # The heads of one shard, rolled. Unrolling it at trace time does not promote the
    # accumulators: at fold 18 local traffic rose from 1,135.3 MB to 1,290.4 MB a
    # launch, and a call went from 12,260.9 us to 13,596.7 at fold 2 and 11,530.8 to
    # 12,297.2 at fold 3. The depth is what cuts the spill, not the loop form. At one
    # head, which is the default depth, the unrolled form has no body to duplicate and
    # drops the loop instead, worth 562 us a call; taking that would mean the body
    # written twice, since the loop form cannot be selected at trace time from one
    # statement, and the body is four hundred lines.
    for hstep in cutlass.range(fold, unroll=1):
        hidx = (gidx * splits + sidx) * fold + hstep
        cute.arch.sync_threads()
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
        _fill_zero(srow, chunk * ROW_WORDS, tid, threads)
        _fill_zero(sdlp, wgroups * chunk, tid, threads)
        _fill_zero(sdw, 4 * chunk, tid, threads)
        cute.arch.sync_threads()
        chunk_prefixes(strans, slp, squat, tid, chunk)
        cute.arch.sync_threads()
        build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
        cute.arch.sync_threads()

        # The closing transition, read once per head. Ac is R(Q)^T, so it is the
        # frame change the increment cotangent needs. Its own two cotangents are
        # read where the chart closes instead of here: nine live floats across the
        # source-token loop is nine the accumulators do not get.
        aclast = _mat_at(stable, TABLE_AC, last)
        lplast = slp[last]

        # Three staging passes back to back, so their global loads overlap rather
        # than serializing. The readout basis is the M mode of two GEMMs and the K
        # mode of a third, so it is staged once; ``slp`` is passed as its scale tile
        # and left unread, the per-token exponential belonging to the offset term
        # alone.
        stage_rotated(
            gcj,
            gcj,
            scrot,
            stable,
            slp,
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
        stage_shifted(
            gdy,
            gdy,
            sdy,
            bidx,
            hidx,
            t0,
            1,
            valid,
            tid,
            threads,
            mpad - 1,
            rows,
            False,
        )
        stage_state(gzj[bidx, hidx, cidx, None, None], sstate, tid, threads, rows, tile)
        cute.arch.sync_threads()

        # The offset term, and the log-scale cotangent it carries. The scale is
        # per target token, so it rides the accumulator's M mode and is applied
        # after the reduction that needs the unscaled value.
        dcacc.fill(0.0)
        mma_gemm(tiled_mma, tid, dcacc, vdy, vstate, True, False)
        # One store per row, not one per element. The row's contributions are summed
        # in a register in fragment order and the tile is zeroed before the head, so
        # the sum the row receives is the one the per-element form left there:
        # ``0 + p0`` is ``p0``, and every later addition is in the same order and the
        # same association.
        for group in crows:
            m, _ = ccrd[group[0]]
            expl = decay(slp[cutlass.min(m, last)])
            term = zero
            for i in group:
                _, d = ccrd[i]
                held = _sum_over_n(dcacc[i] * widen(scrot[m, d], elem))
                term = term + 2.0 * expl * held
                dcacc[i] = dcacc[i] * expl
            if tid % 4 == 0 and m < chunk:
                sdlp[wgroup, m] = sdlp[wgroup, m] + term

        cute.arch.sync_threads()
        stage_matrix(
            gdincj,
            sstate,
            sstate,
            aclast,
            bidx,
            hidx,
            cidx,
            tid,
            threads,
            rows,
            lanes,
            False,
        )
        cute.arch.sync_threads()

        for nstep in cutlass.range_constexpr(blocks):
            nbase = nstep * span
            stage_shifted(
                gbj,
                gbprevj,
                sb,
                bidx,
                gidx,
                t0,
                nbase,
                valid,
                tid,
                threads,
                span,
                tile,
                has_prev,
            )
            stage_shifted(
                gu,
                guprev,
                su,
                bidx,
                hidx,
                t0,
                nbase,
                valid,
                tid,
                threads,
                spad,
                rows,
                has_prev,
            )
            # The readout entry the tap epilogue applies is indexed by the row a
            # thread holds, and a thread holds the same row at both taps, so the pair
            # reads it once here instead of once a tap. Transposed at the read because
            # that is the only form used. The table is published by the barrier above
            # the source-token loop and nothing writes it after, so this needs no
            # barrier of its own.
            taprows = threads // LANE_GROUP
            acrow = tuple(
                mat3_transpose(
                    _mat_at(
                        stable,
                        TABLE_AC,
                        nbase + cutlass.min(tid // LANE_GROUP + s * taprows, span - 1),
                    )
                )
                for s in range(-(-span // taprows))
            )
            for tap in cutlass.range_constexpr(2):
                cute.arch.sync_threads()
                _rotate_rows(
                    sb,
                    sbrot,
                    stable,
                    tid,
                    nbase,
                    TABLE_AP if tap == 0 else TABLE_AN,
                    tap,
                    threads,
                    span,
                    lanes,
                )
                cute.arch.sync_threads()

                # Two views of the one staged run, one row of pitch apart: the
                # current tap reads token nbase+r, the previous one nbase+r-1.
                vun = cute.make_tensor(
                    su.iterator + tap * ldu,
                    cute.make_layout((span, rows), stride=(ldu, 1)),
                )
                vum = cute.make_tensor(
                    su.iterator + tap * ldu,
                    cute.make_layout((spad, rows), stride=(ldu, 1)),
                )

                # The score, target token by source token, into the readout
                # accumulator. I6: the mask lands on the float32 accumulator, then
                # one narrowing into the operand. I3: one exponential of a log
                # difference.
                dmacc.fill(0.0)
                mma_gemm(tiled_mma, tid, dmacc, vdy, vun, True, True)
                for i in cutlass.range_constexpr(cute.size(dmacc)):
                    m, n = mcrd[i]
                    src = nbase + n
                    masked = dmacc[i] * decay(slp[cutlass.min(m, last)] - slp[src])
                    mfrag[i] = narrow(select(src <= m, masked, zero), elem)
                if cutlass.const_expr(wgroups == 1):
                    mma_gemm_areg(tiled_mma, tid, dcacc, fa_m, vbrot, False)

                # The same score, transposed, for the forcing accumulator, which
                # contracts over the target token the readout consumer holds as N.
                # A fragment cannot be reread across its M mode, so the transpose
                # is a store and a transposed ``ldmatrix``: the pair of adjacent
                # columns a thread holds lands in one bank word, and the eight rows
                # of a warp's quads stride four banks apart, so the store is
                # conflict-free by the pitch. The pad rows the source-token M mode
                # was rounded up by are never written and reach only the
                # accumulator rows the store below drops.
                for i in cutlass.range_constexpr(cute.size(dmacc)):
                    m, n = mcrd[i]
                    sscore[m, n] = mfrag[i]

                # The increment term opens the forcing accumulator, because its
                # weight is per source token and the score term's is not. It runs
                # between the score's store and the barrier that publishes it.
                dbacc.fill(0.0)
                mma_gemm(tiled_mma, tid, dbacc, vum, vstate, True, False)
                for i in cutlass.range_constexpr(cute.size(dbacc)):
                    r, _ = bcrd[i]
                    src = nbase + cutlass.min(r, span - 1)
                    dbacc[i] = dbacc[i] * decay(lplast - slp[src])
                cute.arch.sync_threads()
                mma_gemm(tiled_mma, tid, dbacc, vscore, vcrot, False, False)
                # The readout term of a split N mode, which cannot take the score out
                # of the fragment that produced it. Its left operand is the tile the
                # transpose above already published, so it needs no barrier of its own
                # and it costs one more pass over that tile. It sits here rather than
                # before the store because the store is what publishes the operand;
                # the accumulator is untouched in between, so the K order the readout
                # sums in is the order the register form sums in.
                if cutlass.const_expr(wgroups != 1):
                    mma_gemm(tiled_mma, tid, dcacc, vscorem, vbrot, True, False)
                # The gradient overwrites the score it was built from, so the read
                # has to be complete in every thread before the store begins.
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(cute.size(dbacc)):
                    r, d = bcrd[i]
                    if r < span:
                        sdb[r, d] = dbacc[i]

                cute.arch.sync_threads()
                _tap_epilogue(
                    gdtap,
                    sdb,
                    sbrot,
                    sb,
                    sumb,
                    stable,
                    acrow,
                    stap,
                    strans,
                    srow,
                    sdw,
                    bidx,
                    hidx,
                    t0,
                    nbase,
                    valid,
                    tid,
                    jbase,
                    tap,
                    threads,
                    span,
                    lanes,
                )
            cute.arch.sync_threads()

        # The readout accumulator is final. It goes to shared memory because its
        # three columns per token are held by two threads, and the transform and
        # the outer product below need all three in one.
        for i in cutlass.range_constexpr(cute.size(dcacc)):
            m, d = ccrd[i]
            sdc[m, d] = dcacc[i]
        cute.arch.sync_threads()
        _readout_epilogue(
            gdcj,
            sdc,
            sumc,
            scrot,
            stable,
            srow,
            bidx,
            gidx,
            sbase,
            t0,
            valid,
            tid,
            threads,
            chunk,
            lanes,
            fold,
        )
        cute.arch.sync_threads()

        # The rotation cotangent is complete, so the transition chart closes: one
        # 3x3 product per token, the chunk-transition cotangent on the last token,
        # then the two reverse scans the chunk-local prefixes owe.
        #
        # The transition's own two cotangents are read here rather than before the
        # source-token loop, where they would sit live in eleven registers across
        # the widest phase of the kernel. Both are the head's, so a tiled state
        # width takes them on its first lane tile and adds zero on the rest.
        dclose = tuple(gdrot[bidx, hidx, cidx, i // 3, i % 3] for i in range(9))
        dclast = gdscale[bidx, hidx, cidx]
        cscale = decay(slp[last])
        if cutlass.const_expr(tiles > 1):
            dclose = tuple(select(first, v, zero) for v in dclose)
            dclast = select(first, dclast, zero)
        for step in cutlass.range_constexpr(-(-chunk // threads)):
            token = tid + step * threads
            if token < chunk:
                gsum = tuple(srow[token, k] for k in range(ROW_WORDS))
                dac = mat3_mul(gsum, _mat_at(stable, TABLE_AC, token))
                closing = token == last
                drot = tuple(
                    select(
                        closing,
                        dac[3 * (k % 3) + k // 3] + dclose[k],
                        dac[3 * (k % 3) + k // 3],
                    )
                    for k in range(9)
                )
                dquat = rot_hom_vjp(
                    drot,
                    (
                        squat[0, token],
                        squat[1, token],
                        squat[2, token],
                        squat[3, token],
                    ),
                )
                for j in cutlass.range_constexpr(4):
                    sdrot[j, token] = dquat[j]
                # One thread per token, so the rows the tiling gave the offset term
                # fold in here rather than needing a pass and a barrier of their own.
                offset = vdlp[token]
                for g in cutlass.range_constexpr(wgroups - 1):
                    offset = offset + sdlp[g + 1, token]
                vdlp[token] = (
                    offset
                    + wlp * gdlp[bidx, hidx, cidx, token]
                    + select(closing, 2.0 * cscale * dclast, zero)
                )
        cute.arch.sync_threads()
        chunk_suffix(vdlp, sdls, tid, chunk)
        quat_suffix_vjp(squat, sdrot, sdquat, tid, chunk)
        cute.arch.sync_threads()

        for step in cutlass.range_constexpr(-(-chunk // threads)):
            token = tid + step * threads
            if token < chunk:
                dexp = quat_exp_vjp(
                    (
                        sdquat[0, token],
                        sdquat[1, token],
                        sdquat[2, token],
                        sdquat[3, token],
                    ),
                    (strans[0, token], strans[1, token], strans[2, token]),
                )
                if token < valid:
                    # Both halves are sums over lanes, so a tiled state width writes
                    # one slot row per tile and :func:`close_slots` sums them. The
                    # expression is the token's own row at one tile.
                    trow = jbase + t0 + token
                    for j in cutlass.range_constexpr(3):
                        gdtrans[bidx, hidx, trow, j] = sdw[j, token] + dexp[j]
                    gdtrans[bidx, hidx, trow, 3] = sdls[token]

    cute.arch.sync_threads()

    # The shard's two sums for this lane tile, rounded once: the narrowing is here at
    # one shard and at the reduction above one, never twice. Row t+1 of the forcing sum
    # is token t and row 0 is the row the boundary kernel owns.
    total = chunk * tile
    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if i < total:
            t = i // tile
            c = i - t * tile
            if t < valid:
                gdbj[bidx, gidx, sbase + t0 + t, c] = narrow(sumb[t + 1, c], out)
                if cutlass.const_expr(fold > 1):
                    gdcj[bidx, gidx, sbase + t0 + t, c] = narrow(sumc[t, c], out)
    for step in cutlass.range_constexpr(-(-tile // threads)):
        c = tid + step * threads
        if c < tile:
            gcarryj[bidx, gidx, cbase + cidx, c] = sumb[0, c]


@cute.jit
def chunk_vector_bwd(
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
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    groups: cutlass.Int32,
    stream: Stream,
    dtype: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    span: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    resident: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_vector_bwd_kernel`.

    ``P``, ``3N``, the source-token block, the fold and the partial depth are
    compile-time because the accumulator partitions, the arena offsets and the grid's
    first-mode divides are. Batch, group, chunk count and sequence length are dynamic.

    The x extent carries the chunk, the head shard and the lane tile, chunk outermost
    and lane tile innermost. Both inner axes are grid axes rather than loops: the lane
    tile so that no rolled loop sits between an accumulator's allocation and its uses,
    the shard so that the loop which remains there is short. The block count rises by
    their product.

    The block width is the tiling's warp count and nothing else, so it is passed as
    ``warps`` and the thread count is derived from it: two parameters would let the
    launch geometry and the accumulator partition disagree.
    """
    threads = warps * 32
    chunk_vector_bwd_kernel(
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
        gdlp,
        gdrot,
        gdscale,
        gdb,
        gdc,
        gcarry,
        gdtrans,
        gdtap,
        seqlen,
        chunks,
        make_mma(dtype, warps),
        threads,
        chunk,
        rows,
        dim,
        span,
        fold,
        splits,
        has_prev,
    ).launch(
        grid=(chunks * splits * (dim // lane_block(dim)), bsz, groups),
        block=(threads, 1, 1),
        min_blocks_per_mp=resident,
        stream=stream,
    )


@cute.kernel
def vector_reduce_kernel(
    gpb: cute.Tensor,
    gpc: cute.Tensor,
    gpcarry: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    threads: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    pack: cutlass.Constexpr,
) -> None:
    """Sum the head-shard partials of ``dB``, ``dC`` and the carry.

    One block per ``(token, batch, group)``, walking ``3N`` at ``pack`` elements a
    thread; the blocks whose token indexes a chunk take that chunk's carry row too.
    There are never fewer token blocks than chunks, ``L`` being at least 16.

    Indexed by mode rather than by a flat row view, because ``dB`` and ``dC`` are
    bands that may be pitched in their last mode. The partials are contiguous and the
    shard stride is the row extent, so each of a thread's ``splits`` loads is
    coalesced across the warp; the vector is cut from the contiguous mode of the
    partial alone, which is why the reads take it and the stores do not.

    The pass is at its own ceiling -- 152.77 MB in 221.6 us, 689.4 GB/s, 102.6% of the
    copy floor of those bytes, part of the buffer being served from L2 -- so its cost
    is the buffer's and not the kernel's, and the only way to spend less is not to
    round trip. Closing with a float32 atomic instead does not: at this destination
    extent and this fold, an accumulating scatter costs 151.6 us more than a
    non-accumulating one of the same bytes where the shards land adjacent and 193.5
    where they land blocked, which is the layout the partials carry, so the two vector
    outputs are 303.2 to 387.0 us of atomic tax against the 221.6 us it would delete,
    before the shadow buffer such a close needs. It needs one: the destinations carry
    the activation width, and eighteen read-modify-writes at eight mantissa bits is not
    the float32 sum the invariant below promises, so the atomic would land in a float32
    shadow to be filled and then narrowed, 15.7 and 23.6 MB more. ``--atomic-probe`` on
    the profile driver is that measurement.

    The form that is not refused is a close inside the producer. The shard axis sits
    inside the chunk axis of the launch grid, so the eighteen shards of one chunk are
    ninety consecutive blocks holding 1.1 MB of partials, which L2 holds; the last
    shard of a token block could sum them from there for an arrival counter and a
    fence, deleting this launch and most of its DRAM read. That epilogue is in the
    producer's device body, not here.

    Args:
        gpb: ``(B,G,splits*T,3N)`` ``dB`` partials at the activation width.
        gpc: ``(B,G,splits*T,3N)`` ``dC`` partials at the activation width.
        gpcarry: ``(B,G,splits*C,3N)`` float32 carry partials.
        gdb: ``(B,G,T,3N)`` ``dB`` at the activation width, written.
        gdc: ``(B,G,T,3N)`` ``dC`` at the activation width, written.
        gcarry: ``(B,G,C,3N)`` float32 carry, written.
        seqlen: ``T``, the row extent of the two token partials. Dynamic.
        chunks: ``C``, the row extent of the carry partial. Dynamic.
        threads: Block width. Compile-time.
        dim: ``3N``. Compile-time.
        splits: Partial depth, at least two. Compile-time, so a column's loads issue
            together.
        pack: Elements one thread takes per load, from :func:`partial_pack`. Divides
            ``dim``. Compile-time, the vector width being a static property of the
            slice.

    Invariants:
        The producer writes every element of every partial row exactly once, so this
        needs no fill and no atomic. The accumulator is float32 whatever the partial's
        width, so the sum itself is exact to float32 and the roundings on the path are
        the producer's store and this one. Reduction order is ascending shard, fixed
        by the launch geometry and independent of ``pack``, so a rerun at one shape
        reproduces the result bit for bit and a change of vector width does not move
        a bit of it.
    """
    tid, _, _ = cute.arch.thread_idx()
    token, bidx, gidx = cute.arch.block_idx()
    out = gdb.element_type
    part = gpb.element_type
    vectors: cutlass.Constexpr = dim // pack

    # Allocated outside the column loop and indexed only by a trace-time constant,
    # which is what keeps all four of them in registers.
    fragb = cute.make_fragment((pack,), part)
    fragc = cute.make_fragment((pack,), part)
    accb = cute.make_fragment((pack,), cutlass.Float32)
    accc = cute.make_fragment((pack,), cutlass.Float32)

    for group in cutlass.range(tid, vectors, threads):
        for j in cutlass.range_constexpr(pack):
            accb[j] = cutlass.Float32(0.0)
            accc[j] = cutlass.Float32(0.0)
        row = token
        for _ in cutlass.range_constexpr(splits):
            vecb = cute.zipped_divide(gpb[bidx, gidx, row, None], (pack,))
            vecc = cute.zipped_divide(gpc[bidx, gidx, row, None], (pack,))
            cute.autovec_copy(vecb[(None, group)], fragb)
            cute.autovec_copy(vecc[(None, group)], fragc)
            for j in cutlass.range_constexpr(pack):
                accb[j] = accb[j] + widen(fragb[j], part)
                accc[j] = accc[j] + widen(fragc[j], part)
            row = row + seqlen
        for j in cutlass.range_constexpr(pack):
            col = group * pack + j
            gdb[bidx, gidx, token, col] = narrow(accb[j], out)
            gdc[bidx, gidx, token, col] = narrow(accc[j], out)

    if token < chunks:
        for col in cutlass.range(tid, dim, threads):
            held = cutlass.Float32(0.0)
            row = token
            for _ in cutlass.range_constexpr(splits):
                held = held + gpcarry[bidx, gidx, row, col]
                row = row + chunks
            gcarry[bidx, gidx, token, col] = held


@cute.jit
def vector_reduce(
    gpb: cute.Tensor,
    gpc: cute.Tensor,
    gpcarry: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    groups: cutlass.Int32,
    stream: Stream,
    threads: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    splits: cutlass.Constexpr,
    pack: cutlass.Constexpr,
) -> None:
    """Launch :func:`vector_reduce_kernel`, one block per token per batch and group.

    No activation dtype is taken: the kernel reads the destination's element type off
    the operand, and :func:`slinoss._cute.jit_launch` puts that type in the executor
    key itself. ``pack`` is passed rather than derived, the partial's width being a
    host fact.
    """
    vector_reduce_kernel(
        gpb, gpc, gpcarry, gdb, gdc, gcarry, seqlen, chunks, threads, dim, splits, pack
    ).launch(grid=(seqlen, bsz, groups), block=(threads, 1, 1), stream=stream)


class ChunkVectorBwd(NamedTuple):
    """What one launch of the chunk-vector backward produces.

    Attributes:
        dB: ``(B,G,T,3N)`` cotangent of the forcing vectors, in the activation
            dtype, summed over the heads of each group. The chunk-boundary rows
            carry the current tap alone;
            :func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds the
            previous tap's row there.
        dC: ``(B,G,T,3N)`` cotangent of the readout vectors, in the activation
            dtype, summed over the heads of each group.
        carry_b: ``(B,G,C,3N)`` float32 cotangent that each chunk's first token
            sends to the token before it. Index 0 is the streaming feedback.
        dtrans: ``(B,H,T,4)`` float32 cotangent of ``(w_x, w_y, w_z, ls)``.
        dK: ``(B,H,T,2,4)`` float32 cotangent of the two taps. Lane 3 is zero.
    """

    dB: Tensor
    dC: Tensor
    carry_b: Tensor
    dtrans: Tensor
    dK: Tensor


def chunk_vector_backward(
    dy: Tensor,
    U: Tensor,
    trans: Tensor,
    K: Tensor,
    B: Tensor,
    C: Tensor,
    dinc: Tensor,
    zstart: Tensor,
    dlogp: Tensor,
    dchunk_rot: Tensor,
    dchunk_scale: Tensor,
    chunk_size: int,
    *,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
    dB: Tensor | None = None,
    dC: Tensor | None = None,
    splits: int | None = None,
    warps: int = WARPS_WIDE,
) -> ChunkVectorBwd:
    """Differentiate the rowwise vectors and the transition parameters.

    The three cotangents this takes from the chunk-input stage are consumed, never
    recomputed: ``dlogp`` is that stage's half of the log-scale cotangent, and the
    closing rotation and scale are one contraction over the chunk-start state that
    stage already ran.

    Two workspaces, both allocated here and freed on return, and both holding one
    partial row per block of a sum whose terms separate blocks cannot share. Above one
    lane tile the transition chart is a sum over lanes: ``(B, H, tiles * T, 4)`` and
    ``(B, H, tiles * T, 2, 4)``, float32 with their outputs. Above partial depth one
    the two vectors and the carry are sums over heads: ``(B, G, splits * T, 3N)``
    twice at the activation width and ``(B, G, splits * C, 3N)`` at float32, which is
    :func:`partial_bytes`. Each partial carries its own output's width, that output's
    store being one more rounding on the same path. At one copy of either there is no
    buffer and the kernel stores the output directly.

    Args:
        dy: ``(B,H,T,P)`` cotangent of ``y``, one of
            :data:`slinoss.ops.so3ssd.cute.guard.OPERAND_DTYPES`, contiguous. A
            caller with no ``dy`` passes zeros: the increment terms survive.
        U: ``(B,H,T,P)`` forcing input, the dtype of ``dy``, contiguous.
        trans: ``(B,H,T,4)`` float32, contiguous. ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)`` float32, contiguous. ``(kr, g, h, 0)`` per tap.
        B: ``(B,G,T,3N)``, the dtype of ``dy``, pitched. ``G`` divides ``H``; head
            ``h`` reads group ``h // (H // G)``.
        C: ``(B,G,T,3N)``, the dtype of ``dy``, pitched.
        dinc: ``(B,H,C,P,3N)`` float32 increment cotangent in the global frame,
            contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`.
        zstart: ``(B,H,C,P,3N)`` float32 chunk-start state, contiguous, held from the
            forward, or rebuilt when the boundary did not cross.
        dlogp: ``(B,H,C,L)`` float32, contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_input.chunk_input_backward`.
        dchunk_rot: ``(B,H,C,3,3)`` float32, contiguous, from the same.
        dchunk_scale: ``(B,H,C)`` float32, contiguous, from the same.
        chunk_size: ``L``. A multiple of 16.
        u_prev: ``(B,H,P)`` streaming ``u_{-1}``, or None.
        b_prev: ``(B,G,3N)`` streaming ``b_{-1}``, or None.
        dB: Destination for the ``B`` cotangent, with the shape, dtype and device of
            ``B`` and possibly pitched. Every row is written by the kernel's own
            indexed stores, so it is never accumulated into and never zeroed first,
            and it is returned as this same object. ``None`` allocates. See
            :func:`slinoss.ops.so3ssd.reference.check_grad_band`.
        dC: Destination for the ``C`` cotangent, under the contract of ``dB``.
        splits: Partial depth of the head sum, a divisor of ``H // G``. ``None`` takes
            :func:`vector_splits`'s default. One walks every head of a group in one
            block and writes the three summed outputs directly; above one the heads
            are shared out over that many blocks and a second launch closes the
            partials. The returned tensors are the full sums either way.
        warps: Warps per block of the main kernel, a multiple of
            :data:`slinoss.ops.so3ssd.cute.common.WARPS` at most
            :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`, which is the default.
            Warps past the first four go to the atom tiling's N mode, so the tile,
            every M extent and every pitch are the width's invariants and the
            footprint grows by ``4 * L`` bytes per warp group past the first. The
            source-token block is chosen at the default width, so a shape that fits at
            four warps and not at eight is refused rather than run at a narrower block.
            Every output but the log-scale column of ``dtrans`` is bitwise the same at
            both widths; that column sums a row's state width group by group, which
            measured 7.8e-08 of the reference magnitude against the column's own
            1.4e-03 residual.

    Returns:
        :class:`ChunkVectorBwd`.

    Raises:
        ValueError: On a layout, rank, shape or extent violation, on a destination
            that is not the band of its operand, on a shared-memory budget the
            device cannot hold, on half a streaming pair, on a ``splits`` that
            does not divide the fold, or on a ``warps`` that is not a legal block
            width.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (U, "U"), (B, "B"), (C, "C"))
    pinned: Named = (
        (trans, "trans"),
        (K, "K"),
        (dinc, "dinc"),
        (zstart, "zstart"),
        (dlogp, "dlogp"),
        (dchunk_rot, "dchunk_rot"),
        (dchunk_scale, "dchunk_scale"),
    )
    check_layout(((dy, "dy"), (U, "U"), *pinned))
    check_pitched(((B, "B"), (C, "C")))
    dtype = check_operands(activations)
    check_pinned(pinned)
    bsz, heads, groups, seqlen, rows, dim = check_shapes(
        U, trans, K, (B, "B"), (C, "C")
    )
    if tuple(dy.shape) != tuple(U.shape):
        raise ValueError(f"dy must be {tuple(U.shape)}, got {tuple(dy.shape)}")
    check_rows(rows)
    shards = vector_splits(heads // groups, splits)
    fold = heads // groups // shards
    span = vblock(chunk_size, rows, dim, fold, dy.element_size())
    check_extents(chunk_size, dim, span)
    has_prev = check_stream(u_prev, b_prev, (bsz, heads, groups, rows, dim))

    chunks = -(-seqlen // chunk_size)
    state = (bsz, heads, chunks, rows, dim)
    for tensor, name in ((dinc, "dinc"), (zstart, "zstart")):
        if tuple(tensor.shape) != state:
            raise ValueError(f"{name} must be {state}, got {tuple(tensor.shape)}")
    closing = (
        (dlogp, "dlogp", (bsz, heads, chunks, chunk_size)),
        (dchunk_rot, "dchunk_rot", (bsz, heads, chunks, 3, 3)),
        (dchunk_scale, "dchunk_scale", (bsz, heads, chunks)),
    )
    for tensor, name, shape in closing:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")

    # The N atoms of the tiling, which is the warp-group count the offset term takes a
    # row of. Raises on an illegal width, so the block geometry is checked here rather
    # than inside the trace.
    wgroups = mma_atoms(warps)[1]
    budget = assert_smem_fits(
        f"chunk_vector_bwd[L{chunk_size}/P{rows}/3N{dim}"
        f"/fold{fold}/S{shards}/W{warps}]",
        vector_smem_bytes(
            chunk_size, rows, dim, fold, span, dy.element_size(), wgroups
        ),
    )

    # After the operand guards, so a destination is measured against an operand that
    # has already been held to its own shape and layout.
    if dB is not None:
        check_grad_band(dB, B, "dB")
    if dC is not None:
        check_grad_band(dC, C, "dC")

    device = dy.device
    if dB is None:
        dB = torch.empty(bsz, groups, seqlen, dim, dtype=dtype, device=device)
    if dC is None:
        dC = torch.empty(bsz, groups, seqlen, dim, dtype=dtype, device=device)
    carry_b = torch.empty(bsz, groups, chunks, dim, dtype=torch.float32, device=device)
    dtrans = torch.empty(bsz, heads, seqlen, 4, dtype=torch.float32, device=device)
    dK = torch.empty(bsz, heads, seqlen, 2, 4, dtype=torch.float32, device=device)
    # The two outputs that are sums over lanes, and the three that are sums over the
    # heads of a group. Nothing else needs a partial row: every other output spans the
    # state width and belongs to one head, so one block owns it.
    tiles = dim // lane_block(dim)
    held = (open_slots(dtrans, tiles), open_slots(dK, tiles, axis=-3))
    shared = tuple(open_slots(out, shards) for out in (dB, dC, carry_b))
    jit_launch(
        chunk_vector_bwd,
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
            dlogp,
            dchunk_rot,
            dchunk_scale,
            shared[0].dest,
            shared[1].dest,
            shared[2].dest,
            held[0].dest,
            held[1].dest,
            seqlen,
            chunks,
            bsz,
            groups,
        ),
        (
            cute_dtype(dtype),
            warps,
            chunk_size,
            rows,
            dim,
            span,
            fold,
            shards,
            has_prev,
            min(RESIDENT_MAX, max(1, smem_capacity() // budget)),
        ),
    )
    for slots in held:
        close_slots(slots)
    if shards > 1:
        jit_launch(
            vector_reduce,
            (
                shared[0].dest,
                shared[1].dest,
                shared[2].dest,
                dB,
                dC,
                carry_b,
                seqlen,
                chunks,
                bsz,
                groups,
            ),
            (THREADS, dim, shards, partial_pack(dB.element_size(), dim)),
        )
    return ChunkVectorBwd(dB=dB, dC=dC, carry_b=carry_b, dtrans=dtrans, dK=dK)
