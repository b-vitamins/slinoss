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
``H // G`` may be cut into :func:`vector_splits` shards, one block to a shard. The fold
is one at ``standard`` and eighteen at the default configuration, and the depth is one:
a block walks the whole fold and writes the three outputs itself.

A depth above one costs a launch, and the launch is what it loses on. A block that
walks the fold holds the readout gradient's sum across the loop in the register
fragment :func:`_fold_frag`, twelve float32 a thread and no shared memory at all, and
it reads and writes the same operands a block walking one head does.

Depth one is worth -41.5 us plus or minus 1.5 at ``L 64 P 64 3N 240 G 1`` and the
shipped width, on a paired wall against the full depth over 3,000 pairs. The interval
is [-41.472, -41.472] and excludes zero, replicated across independent runs; the two
null controls bounding the position bias read +1.536 and -1.024.
``sm__cycles_active`` agrees in sign at -8.67%, 231,083,593 against 211,057,167 over
six launches a side. The effect needs an idle device to be visible at all: the same
wrapper reads 1,589 to 1,617 us idle and 4,065 to 4,075 us beside a foreign job, so
41 us is 2.6% of the one and 1.0% of the other. An earlier attempt to price the depth
by differencing per-kernel duration sums across separate profiler processes reported
-62.0 us and is withdrawn; that method credits the deleted launch in full and charges
the loop nothing.

The gain is the second launch ceasing to exist and the traffic that went with it, not
the loop. DRAM over both kernels falls 552.97 MB to 296.92 MB, -46%: the full depth
pays 197.88 MB read and 201.21 MB written here plus 143.78 MB read and 10.10 MB
written in :func:`vector_reduce`, and depth one pays 230.72 MB read and 66.20 MB
written with no second kernel at all. Instructions move the other way, +4.12% for the
loop, 273,740,544 against 285,024,640, and that fee is already inside the -41.5.

The register form is separately worth nothing in time. Holding the fold sum in
registers rather than shared, at the full depth on both sides, measures [0.000, 0.000]
over 3,000 pairs with every output bitwise, replicated. Its whole value is the
13,312 B it frees, which is what makes depth one reachable at a source-token block of
64.

Registers are 238 a thread at depth one against 142 at the full depth, 17 under the
cap, with zero local load and zero local store sectors at both. The arena is 99,760 B
at either depth and both occupancy limits read one block.

Depth one changes the summation order, so the two summed outputs move at the default
configuration: 806,644 of 1,966,080 elements of ``dB`` at 1.600e+01 maximum absolute
difference and 807,928 of ``dC`` at 8.000e+00, both inside the declared bounds, with
``carry_b``, ``dtrans`` and ``dK`` bitwise. At fold one every output is bitwise, so
only the default configuration is reached: it is the one shape whose fold exceeds one.

What used to refuse the shallow depth was neither the fee nor the bytes. The readout
gradient's float32 fold accumulator was 13,312 B of shared memory, allocated only above
fold one, and that is the amount that put a source-token block of 64 over the carveout:
every depth below the fold took :func:`vblock` to 32, which is a second pass over ``U``
and costs more than the launch the depth deletes. The accumulator is registers now, the
arena is flat in the fold, and the block stays at 64 at every depth.

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
lanes. A loop accumulated them in the row one block owned; separate blocks cannot. Each
tile writes its own row and both sums close inside the launch, under one arrival
counter: every tile increments it and the tile that reads the last value closes both.
``dtrans`` publishes :data:`PART_WORDS` words a token, every map between the rotation
cotangent and it being linear so the sum may cross them, and the closing tile runs the
maps once, which pays the two warp-serial scans and the exponential's adjoint on one
tile of ``tiles``. ``dK`` leaves its epilogue in registers into a float32 slot row that
is past every map already, so closing it is an add of ``tiles`` rows in slot order and
no map at all. Neither sum reaches a second launch. At one tile there are no slots, no
counter and no published words, and the store is the output itself.

The block's whole width waits on the increment. On the acceptance shape it is the
largest single barrier site in the kernel: the shared read that broadcasts the
returned count carried 9,365 of 33,198 ``barrier`` samples, 4.36% of all samples and
817 cycles a block, and it is fully exposed because one block is resident and all
eight warps wait at the same barrier. A device fence ahead of the increment cost
30.3% of that. The fence lowered to the ``MEMBAR.ALL.GPU``, ``ERRBAR`` and
``CCTL.IVALL`` triple the ``acq_rel`` increment lowers to anyway, back to back with
only address arithmetic between the two triples, so deleting it removed 34,560 warp
instructions a launch, took the site to 6,527 samples, and measured -22.016 us at
acceptance and -3.328 at ``wide`` bitwise clean, interval excluding zero over 600
pairs against a null control that did not. A sample share converted to time at 1.10x
here. At one tile there is no increment and the shapes measured exactly zero.

Shared memory is one resident set and one phase arena. Resident: ``trans``, ``K``,
the two chunk-local prefixes, the three-slot transform table, the nine-float
per-token scratch, the log-scale and quaternion cotangents, and the rotated
readout. The arena holds the float32 forcing sum that outlives a head, one region per
tap holding either the float32 forcing gradient or the narrowed score, and five
operand tiles: the output cotangent, one tile that carries the chunk-start state and
then the increment cotangent, the raw and rotated forcing tiles and the ``U`` tile.
The float32 readout gradient of the epilogue aliases those five, none being live
when it is. The readout gradient's sum over the fold has no region: it is the register
fragment :func:`_fold_frag` holds.

The source-token block is :func:`vblock`, one M tile of the atom where the budget
allows it and half of one where it does not. Below one M tile every warp still
carries rows of every GEMM, because the transposed contractions round their M mode
up to the tile.

The budget bounds ``L``, ``P`` and the fold. It does not bound ``3N``, and this is
still the widest live set in the tree. ``L 16`` and ``L 32`` fit at every ``P``,
every fold and every ``3N``.
``L 64`` fits to ``P 64`` at every fold, in 98,736 B at the shipped width and 256 B
less at one warp group, whether ``3N`` is 48 or 240 and whatever the fold. The fold was
an axis of that figure until the readout gradient's fold sum moved to registers: a
shared region for it is 13,312 B, which took ``L 64`` at ``P 64`` and any fold above one
to 112,048 B, over the carveout, and :func:`vblock` then halved the source-token block
at every depth below the fold.
The table's segment-aligned pitch costs ``36 * L`` B of each figure, 2,304 B at
``L 64``, and it does not scale with the fold either. ``L 64`` at ``P 128`` and ``L 128`` at
every ``P``
are refused: the smallest live set at ``L 128`` is 127,920 B, above the capacity of
every device the DSL reports. :func:`slinoss._cute.assert_smem_fits` refuses the
rest rather than any path here degrading. The one-tap form's ``dls_step`` slot raised
every figure in this paragraph by 1,488 B at ``L 64``, 848 B at ``L 32`` and 528 B at
``L 16``, and the ``t-1`` deposit buffers :func:`_tap_epilogue` writes raised them by
a further ``52 * L`` B, 3,328 B at ``L 64``. Neither moved :func:`vblock`'s choice or
a residency at any shape: every legal shape stays legal and every refused one stays
refused.

The largest ``L`` this layout admits is 64, at one resident block, at ``P 64`` and at
``P 48`` alike. Two resident blocks of 128 threads need 50,176 B each, which is
:func:`slinoss._cute.smem_budget` at two and not half the 101,376 B capacity: the
capacity has one driver reservation subtracted from it and two blocks pay two. No
legal shape at ``P 64`` reaches that bar: the smallest arena there is 55,600 B, at
``L 16``. Splitting the fold across blocks buys parallelism and not occupancy, and no
longer buys a byte either: the accumulator that existed only above fold one is a
register fragment. Three extents scale with ``L`` -- the float32 forcing sum, the
output tile and the aliased float32 readout -- which is why 128 is refused and why
grid-izing the lane tile does not unpin it.

DRAM-bound. Analytic traffic at ``standard``, operand by operand, with ``U`` and
``B`` at the ``L + 1`` rows per chunk their shifted span reads::

    reads   dy 9.44 + U 9.58 + B 9.58 + C 9.44 + trans 1.57 + K 3.15
          + dinc 7.08 + zstart 7.08 + dlogp 0.39 + dchunk_rot 0.06
          + dchunk_scale 0.01                                        = 57.37 MB
    writes  dB 9.44 + dC 9.44 + dtrans 1.57 + dK 3.15 + carry_b 0.29 = 23.89 MB

81.26 MB against ``1536 * 4.03 MFLOP = 6.19 GFLOP``, so 76.2 flop/byte against a
ridge point of 164: memory bound by a factor of 2.2. That table is the ``span 64``
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
    once          B 3.99 + C 3.93 + dinc 70.78 + zstart 70.78 r,
                  dB 3.93 + dC 3.93 + carry_b 0.12 w      =           157.43 MB
    chart words   dtrans 4.72 published and read back per tile,
                  then 2.36 written out once              =           49.55 MB
    slot rows     dK 4.72 written and read back per tile,
                  then 4.72 written out once              =           51.90 MB

583.79 MB. The chart words are 23.60 MB more than the four-word slot rows they
replaced, all of it the read-back, and the read-back is the term the table overstates:
the tiles of one chunk are consecutive blocks in ``x``, so they publish and are summed
inside one L2 residency. Measured at the acceptance shape, the producer's
``dram__bytes_read.sum`` moves +0.011 MB for 23.6 MB of published words, so L2 serves
the whole of the read-back. ``U`` dominates the per-tile term
because its tile is one atom M tile whatever the ``span``, so a ``span 32`` shape
reads it twice. ``dinc`` and ``zstart`` are ``(B, H, C, P, 3N)`` at the operand width
and together are 25% of the total. The three write terms are the shipped depth's, which
is one. A depth above one replaces them with :func:`partial_bytes`, 143.77 MB at the
full depth, the largest single item in that launch and the whole of what its closure
reads.

Every measurement in the rest of this docstring was taken with ``dinc`` and ``zstart``
at float32, which is 141.56 MB a launch more than the shipped kernel reads and puts
the full-depth analytic total at 837.5 MB rather than 696.0 MB. Narrowing them
takes the counted read from 336.60 MB to 195.01 MB, the full 141.60, and the launch
from 1,855.1 us to 1,837.4: -17.7 us for 17% of the traffic, which is the ratio a
launch this far off its byte floor converts at.

That total is an upper bound and the launch does not pay it. 837.5 MB analytic at the
full depth against 515.81 MB of DRAM counted, 62% over. The 321.7 MB of daylight is
the size of the per-tile re-read term, 259.89 MB, and the lane tile is the innermost
axis of ``x``, so the five tiles of a token block run back to back and L2 serves the
repeats. Score this kernel against its counted traffic, never against the table.

What the one-tap form is worth here, as a paired delta of the two launches in a single
process with the launch order swapped every iteration, 240 pairs, 2026-08-22, ``dinc``
and ``zstart`` at bfloat16:

    shape       delta us  interval          null   position
    standard      -14.848 [-15.872,-14.848] 0.000    0.512
    wide           -9.216 [ -9.216, -9.216]    --    1.024
    acceptance    -44.544 [-46.592,-43.520] 0.256    3.584

The counters say the saving is not where the GEMM count put it. At the acceptance shape
``sm__inst_executed_pipe_tensor`` falls 44.83%, 10,690,560 to 5,898,240, and
``smsp__sass_thread_inst_executed_op_fmul_pred_on`` falls 11.24%, 801,527,588 to
711,417,338, but ``smsp__sass_thread_inst_executed_op_ffma_pred_on`` rises 21.00%,
1,445,363,712 to 1,748,961,792. The fusion deletes a tap of the score GEMM, which is
tensor-pipe work, and pays for it with ``dls_step``, which is a per-token lane reduction
over the fused column and so is FFMA-pipe work. The launch is shorter because this
kernel is nowhere near tensor-bound, not because it does less arithmetic: the two
floating-point counters together rise 9.50%. Registers fall 145 to 132 and local traffic
is zero on both sides. A prediction for this kernel derived by counting GEMM MACs
predicts the wrong pipe and gets the sign wrong.

Measured, the bar is missed, and the distance is latency and not traffic. Every
counter below is from one profile of this kernel on an RTX A6000, ``sm_86``, 84
multiprocessors, one profiled launch per counter pass, clocks unlocked because
locking is denied on this fleet. A floor is ``c + bytes / B`` on a fit taken in the
same process, ``c`` about 4.3 us and ``B`` about 685 GB/s at a worst residual of
0.40%, and ``bytes`` is the analytic traffic above unless stated otherwise. A
duration is stamped with the compute-apps query taken before and after it; where
that query named another process the duration is a bound, not a rate.

Every counter in the rest of this section was taken at the full depth -- 11,520 blocks
of 256 threads, five lane tiles, the fold of 18 cut into eighteen shards, one head to a
block -- which is not the shipped depth and is where this kernel's traffic and issue
record was built. There the main kernel moves
515.74 MB of DRAM per launch in 1,851.8 us on device time at 1.7969 GHz, median of
five steady-state launches in each of two A/B blocks, 3,285,344 active cycles at a
0.59% spread.
:func:`vector_reduce` closes the head sum in 222.0 us at 152.77 MB and 102.7% of its
own floor; the two lane-slot reductions add 44.2 and 23.0 us. Three of the four
launches are at their bandwidth; the main kernel is the one that misses the 85% the
class asks, and it is issue-bound rather than short of bandwidth.

At the shipped depth the same shape is 640 blocks and one launch fewer, and the main
kernel moves 296.92 MB across the pair at 238 registers, 99,760 B of arena and a
source-token block of 64. Its class figure falls with the bytes and not with the time:
22.8% of memory speed-of-light against 38.1%. The paragraphs below read as the record of
what this kernel's cycles go on, which the depth does not move -- issue-active 36.0%
against 37.3%, one resident block either way -- and not as the shipped launch's own
byte or duration figures.

Read a microsecond figure here against the clock it was taken at. The part boosts
between 1.4 and 1.9 GHz with contention from other processes on the device, and the
same kernel has measured 2,649.0 us on a contended part and 2,103.5 us on an idle
one under the same counter passes. Cycles are the invariant; a duration is not.
Stamp the clock from ``gpc__cycles_elapsed.avg.per_second`` beside any duration.

That percentage is not a traffic problem, and at the shipped width it is not
instruction supply either. 95,408 B and 132 registers a thread each admit one
256-thread block per multiprocessor, ``launch__occupancy_limit_shared_mem`` and
``launch__occupancy_limit_registers`` both reading 1: 16.7% theoretical occupancy,
16.5% achieved, two warps a scheduler. After the tap fusion ``barrier`` leads the
stalls at 17.3% against ``mio_throttle`` at 15.3%, issue-active is 36.4%, and DRAM
runs at 36.5% of peak against tensor at 10.8%. What the launch
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
two state buffers are. Both have since been narrowed to the operand width.

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
direction, and the 141.56 MB the narrowed ``dinc`` and ``zstart`` take off the read
side cannot be priced from its bytes either. That narrowing is DRAM-only:
:func:`slinoss.ops.so3ssd.cute.table.stage_state` and
:func:`slinoss.ops.so3ssd.cute.table.stage_matrix` already narrowed both to the
operand width on the way into the one state tile, so the global width reaches no
shared byte and no operand, and no extent in the tables below follows it.

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

The in-block fold spills nothing, at any ``L``, and this is where the record of it
being the whole reason for the depth used to sit. That record read 536.74 MB of local
load and 344.06 MB of local store a launch at depth one over 640 blocks, against zero
at the full depth, and a table of it growing with ``L``. It no longer reproduces. The
fold of 18 walked in the block, one profiled launch a counter pass at ``P 64 3N 240``
and the shipped width::

    L    smem    regs   local sectors   DRAM MB   GB/s   us/launch   blocks
    16   55,664   210         0            689.10  140.0    4,923.2    2,560
    32   68,656   220         0            404.92  158.4    2,556.3    1,280
    64   98,736   238         0            264.45  165.3    1,600.1      640

``l1tex__t_sectors_pipe_lsu_mem_local_op_ld`` and ``_st``, hit and miss summed: zero at
every row, zero at the full depth, and zero at the depths of 2, 3, 6 and 9 between them.
The register count rises with ``L`` and stays under the 255 cap. Two changes retired the
spill: the one-tap fusion took the count 145 to 132 at fold one, and the readout
gradient's fold sum left a 13,312 B shared region for the twelve-word float32 fragment
:func:`_fold_frag`. So the depth is chosen on the pass and the loop alone, which is the
sweep in the head of this docstring.

What survives of that table is its one occupancy datum, and it is about ``P`` and not
about the fold. Two resident blocks are out of reach at ``P 64`` for the reason the
budget paragraph above gives. ``L 16`` at ``P 48`` is the one legal shape that does
fit two, 49,680 B at 128 threads against the 50,176 B bar, and it reads 16.5% achieved
occupancy against 8.3% and 71.5% of memory speed-of-light against 45.5% at ``P 64``,
where the same ``L`` holds one block.

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

The register column above predates the merge and is stale, and so were the 236, 149 and
120 figures that stood here for the same quantity. Measured on the current tree, one
ptxas probe a row, with zero local load and store sectors in every row: no configuration
of this kernel spills. ``occ_R`` and ``occ_S`` are
``launch__occupancy_limit_registers`` and ``launch__occupancy_limit_shared_mem``, in
blocks::

    shape        warps  regs  smem dyn   occ_R  occ_S
    tiny             4   191    80,720      2      1
    tiny             8   110    80,976      2      1
    standard         4   213    92,560      2      1
    standard         8   126    92,816      2      1
    ragged           4   213    92,560      2      1
    ragged           8   126    92,816      2      1
    wide             4   228    98,480      2      1
    wide             8   133    98,736      1      1
    acceptance       4   232    98,480      2      1
    acceptance       8   140    98,736      1      1
    long           4/8  unreachable: L 128 needs 142,224 B against 101,376

Eight is the shipped width, so **140** is the shipped count and 232 is a four-warp
probe. ``occ_S`` is 1 in all ten rows while ``occ_R`` admits two in eight of them, so
shared capacity is the limiter at every reachable shape and the register file ties it
only at the two widest. Register headroom is therefore headroom and not slack: nothing
buys occupancy with it, and what a held operand costs is priced at the one place that
spends it.

The rows above were taken at the full head-sum depth, one head to a block. The shipped
depth is one, and the fold loop it runs instead spends registers: **238** a thread at
the acceptance shape, so ``occ_R`` is 1 there too and shared capacity is the limiter in
every row without exception. Zero local sectors at either depth.

Registers fall to 242, so 256 threads hold 61,952 of the 65,536 a multiprocessor has
and the second warp per scheduler is available. Across the float32 pair,
``no_instruction`` goes 52.8% to 1.0% and issue-active 12.5% to 28.8%: instruction
supply was the whole of that gap. What takes its place is the shared-memory pipe, and
the shipped width's counters for it are in the measured record above. Local traffic
stays zero at either width, conflicts 0.1511 per wavefront against 0.1612,
instructions issued rise 1.1% for the readout term's reread, and the arena grows by
the ``4 * L`` bytes a warp group past the first that :func:`offset_tile` takes -- one
resident block at either width.

Sixteen warps is refused, but not by the register file, and the earlier claim here that
512 threads admit only 128 registers a thread was the wrong reason. Registers fall with
the width because a warp group splits the N mode of one tile, so every accumulator is
``M * N / threads`` elements and halves: the measured 236 to 149 across one doubling
projects 94 to 106 at sixteen, inside the 128 bar, and the arena at four groups is
94,416 B inside the 101,376 B carveout. What refuses it is number theory. Four groups
need ``MMA_TILE_N`` 32, and ``3N`` at 240 factors as ``2^4 * 3 * 5``, whose divisors
include no multiple of 32. ``M`` is pinned at the span, which is ``L``, and ``K`` is
refused by :func:`slinoss.ops.so3ssd.cute.mma.mma_atoms`, which replicates the
accumulator across it. Adding groups and widening the tile's warp count are the same
parameter, not two.

A second resident block at eight is refused by shared capacity, not by the register
file. The claim that stood here, that ``launch__occupancy_limit_registers`` is 1 at 242
registers so no arena reaches a second block, is false: that counter reads 2 in eight of
the ten rows above and shared reads 1 in all ten. Its companion record, that requesting
``min_blocks_per_mp = 2`` takes the allocator to exactly 128 registers a thread and
spills 11.80 MB each way, does not reproduce either. The request now yields 120
registers, zero local sectors and 2.7% more instructions.

The residency class is closed, and closed on arithmetic rather than on a mechanism. The
prize is real and it is the largest the operator has: isolated on this kernel's own body
by dead allocation at fixed geometry with instruction and register counts held identical,
residency 1 to 2 is **-34.8%** of cycles at ``P 16`` and -33.4% at ``P 32``, against an
inert 512 B control that moves cycles 0.009%, replicating inside 0.11%. That is -560 to
-600 us at the acceptance shape.

It is unreachable. The residency-2 bar is **50,176 B**, measured to the 128 B granule:
50,160 B reads two blocks and 50,288 B reads one. The five GEMM operands and the
transform table are all live across one barrier interval, and they pass the bar with no
staging buffer allocated at all::

    sdy     out       9,216   A of the offset GEMM, read by the diagonal
    su      input     9,360   A of the increment GEMM, read by the diagonal
    sstate  state     7,168   B of the offset and increment GEMMs
    sbrot   forced    7,168   B of the readout GEMM
    sscore  tapped   13,312   staged record, A of the forcing and readout GEMMs
    stable            9,216   read by :func:`_rotate_rows` in the same iteration
                     55,440   B against a 50,176 B bar

That floor is split-invariant, so no phase split reaches it, and separating any two of
those GEMMs writes an operand to global at 566 MB and about 944 us against a 600 us
prize. Aliasing does not reach it either: the source-token loop holds every tile live at
once except ``sdrot``, ``sdquat`` and ``sdls``, which are written and read inside the
closing chart, so lifetime aliasing has 2,304 B in it. ``L`` is dead as a lever in both
directions, :func:`mma_rows` pinning M to ``MMA_TILE_M`` at every ``L``: ``L 16`` triples
the instruction count, 923,019,264 against 311,030,784, and ``L 128`` does not fit.
Registers cannot buy the bytes: the cap at residency 2 is 128 and forcing it yields 120,
so 8 spare registers are 8,192 B against the 48,560 B by which the shipped arena passes
the bar, 5.9x underwater.

Occupancy at this width is a shared-capacity problem and it has no lever.

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
from cutlass._mlir.dialects import llvm
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
    smem_residency,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AFUSE,
    TABLE_AN,
    THREADS,
    Mat3,
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
    check_stored,
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
    mma_atoms,
    mma_coords,
    mma_gemm,
    mma_groups,
    mma_matrices,
    mma_offsets,
    mma_rows,
    operand_tile,
    smem_pitch,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes, chunk_suffix, quat_suffix_vjp
from slinoss.ops.so3ssd.cute.table import (
    LANE_PAIR,
    apply_matrix,
    apply_rotated,
    build_table,
    matrix_frag,
    paired,
    read_matrix,
    read_rotated,
    rotated_frags,
    stage_chunk,
    stage_score,
    stage_shifted,
    stage_state,
    stage_trans,
    store_pair,
)
from slinoss.ops.so3ssd.reference import check_grad_band

__all__ = [
    "LANE_BLOCK",
    "LANE_GROUP",
    "PARTIAL_REQUEST_BYTES",
    "PART_WORDS",
    "PREFIX_WARPS",
    "RESIDENT_MAX",
    "ROW_WORDS",
    "TABLE_PITCH",
    "TABLE_QUAD",
    "Arena",
    "ChunkVectorBwd",
    "Slots",
    "arena",
    "chunk_prefix_bwd",
    "chunk_prefix_bwd_kernel",
    "chunk_vector_backward",
    "chunk_vector_bwd",
    "chunk_vector_bwd_kernel",
    "forced_tile",
    "gradient_tile",
    "lane_block",
    "offset_tile",
    "open_slots",
    "out_tile",
    "partial_bytes",
    "partial_pack",
    "prefix_smem_bytes",
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
``dK`` are sums over lanes, so every tile contributes a term to them."""

LANE_GROUP: int = 4
"""Threads that cooperate on one token in a rowwise epilogue.

One thread holds one 3-vector, which is what the rowwise transforms and the outer
products need and what an accumulator fragment cannot give: the atom hands a thread
two adjacent columns, and a 3-vector straddles that pair. The group must divide the
lanes of a lane tile, 16 at every shape that tiles, and stay inside a warp.

A thread takes an adjacent RUN of ``N / LANE_GROUP`` lanes, not every
``LANE_GROUP``-th lane. Three components times four adjacent lanes is twelve
adjacent columns, so a thread's whole share of a row is one vector access per
16-byte segment instead of three scalars a lane: twelve accesses fall to three at
each of six sites in the two epilogues. The strided map cannot be widened at any
layout of the trailing axis, component-major included, because a strided thread
holds three columns and then a gap whichever axis is contiguous.

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

DIAG_SPLIT: int = 2
"""Threads a destination in the two rowwise contractions of the source-token pass.

The diagonal cotangent and the increment's closing rank-one are each ``rows`` deep
with a serial float accumulation, and each has one destination a token or a lane, so
one thread a destination left the whole chain on ``span`` or ``tile`` of ``threads``
while the rest of the block stood at the pass's barrier.

Two rather than the full :data:`LANE_GROUP`. The loop's prologue and its closing
butterfly are paid once a warp whatever the depth, so a wider split buys chain depth
at a fixed instruction cost that rises with the warps it wakes, and past two the fee
outruns the latency it deletes.

A thread takes an adjacent SEGMENT of the contracted mode, never a stride of the
split, for the reason :data:`LANE_GROUP` gives: the mode is adjacent in shared memory
and is read eight elements to an ``LDS.128``, so a stride of the split costs four
times the shared loads and loses outright."""

DIAG_CHAINS: int = 2
"""Independent accumulator chains a thread holds in those two contractions.

Float addition does not reassociate, so one chain issues one FFMA every latency and
leaves the pipe idle between. A chain costs one FADD to close and nothing to carry,
where a wider thread split costs a warp's prologue. ``DIAG_SPLIT * DIAG_CHAINS`` must
divide ``rows``, which follows from ``rows`` being a multiple of
:data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_N`."""

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

PART_WORDS: int = 9
"""Float32 words a lane tile publishes per token when the state width is tiled.

Four for the quaternion prefix cotangent, one for the log-scale offset term, four for
the transition parameters' own term, which is one word per component of ``strans``.
What crosses the sum over lane tiles at the point the chart closes, and the point is
chosen for this count: the four maps that remain -- both suffix scans, the
exponential's adjoint and the tap sum -- are all linear, so the sum may cross them,
and every map already run is a contraction, so running one more before the sum widens
what crosses it. Ten words would cross before ``mat3_mul``, thirteen before the
readout epilogue's own sum; five cross after ``quat_exp_vjp``, which is what the slot
form published and cost the two warp-serial scans on every tile.

The ninth word is row 3 of ``sdw``, the fused column's ``dls_step``. It rides the
``strans`` cotangent record because that is what it is the fourth component of, and it
cannot join the log-scale offset term ahead of the sum: the offset term is
reverse-scanned at the closure and ``dls_step`` is not, so the two reach ``dtrans``
through different maps.

``dK`` is not here and does not need to be. Its slot row is past every map, so its sum
over lanes is an add and the closing tile reads the row back rather than a published
word. Routing it through this record would publish eight more words a token on the
token-innermost pitch the coalesced read-back needs, which is eight scalar stores where
the row is two ``STG.E.128``."""

PREFIX_WARPS: int = 4
"""Warps per block of :func:`chunk_prefix_bwd_kernel`.

Only the first warp scans; the rest cover the staging and the store, each at most
``4 * L`` float32 wide. Four is the narrowest width that fills the warp slots:
shared memory is 2,304 B a block and registers are few, so the resident block count
is capped at sixteen by the hardware and only four warps a block reach the
forty-eight slots :data:`slinoss.perf.ceiling.MIN_OCCUPANCY_PCT` is read against.
The idle warps cost issue slots the kernel does not use, its bound being the
warp-serial scan and the transport."""

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

    The masked score, target token by source token, staged from the record
    :func:`slinoss.ops.so3ssd.cute.bwd.chunk_input.chunk_input_bwd` publishes. Both
    modes are rounded up because each is the M mode of one of the two GEMMs that
    read the tile, and the transposed ``ldmatrix`` needs a whole atom tile of
    whatever the source-token block is.

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

    The accumulator store into it is two-way and no pitch clears it. A float32 pair
    goes out eight bytes wide, so a phase is sixteen threads over four accumulator
    rows and the bank pair is ``pitch // 2 * r + d // 2`` modulo sixteen: at 52 words
    a row that is ``{0,10,4,14}`` against a column span of four, and two banks take
    two rows each. Freedom needs the pitch congruent to eight or 24 words modulo 32,
    an even count of 16-byte segments, and the run reads of :func:`_pass_row` need an
    odd one. The staged score escapes because its store is one 16-byte segment a
    thread. Measured at ``standard``: 294,912 excess store wavefronts a launch from the
    forcing gradient and 147,456 from the readout, which at one shared cycle a
    wavefront over 84 multiprocessors bounds the defect at 2.9 us of 215.

    Args:
        rows: Rows to allocate.
        dim: ``3N``.
    """
    return fp32_tile(rows, dim)


class Arena(NamedTuple):
    """Float32-word offsets of the phase-shared tiles inside the one arena.

    The tiles below overlap in address and not in time. The two float32 tiles come
    first and alias nothing: one is live across the whole fold and the other carries
    one tap. The five operand tiles follow, and the readout gradient of the epilogue
    aliases all five, none being live when it is.

    ``state`` holds the chunk-start state through the offset contraction and the
    increment cotangent for the rest of a head's pass. One tile rather than two:
    the two have the same extents, neither is read while the other is being
    written, and the barrier that separates them is the one the source-token loop
    needs anyway.

    The readout gradient's sum over the fold has no region here. It is the register
    fragment :func:`_fold_frag` holds, which is legal because the epilogue's cover of
    the tile is invariant across fold iterations: a thread owns the same run of the
    same token at every head. A shared region for it is 13,312 B, and that is exactly
    what put the source-token block of 64 over the carveout at every depth below the
    fold -- the demotion to 32 that follows costs 1.8 times the launch the depth
    deletes.

    No further pair is disjoint. Every region below except ``readout`` is live across
    the source-token loop, so the arena's extent is that loop's live set, and the
    module docstring carries what the remaining 2,304 B of the resident set would buy.

    Attributes:
        forcing: The float32 forcing gradient, summed over taps, blocks and the
            fold. Row ``t + 1`` is token ``t`` and row 0 is the row that crosses
            the chunk boundary.
        tapped: The float32 forcing gradient of one tap, the GEMM's own output, and
            the staged score of that same tap. Both GEMMs finish reading the score
            before the gradient that overwrites it exists, and the epilogue finishes
            reading the gradient before the next block's score is staged over it.
            The region is the wider of the two.
        out: The output cotangent, ``dy``.
        state: The chunk-start state, then the increment cotangent in the
            chunk-local frame.
        raw: The raw forcing tile, restaged once per source-token block.
        forced: The rotated forcing tile, rebuilt once per tap.
        input: The shifted ``U`` tile.
        readout: The float32 readout gradient of one head. Epilogue only.
        words: Float32 words the arena spans.
        raw_held: Whether ``readout`` stops below ``raw``, which is what decides
            whether the raw forcing tile survives a head's epilogue and so may be
            staged once for the block rather than once a head. True wherever ``dy``
            and the state span at least the readout gradient, which is every shape
            whose ``P`` reaches the lane tile.
    """

    forcing: int
    tapped: int
    out: int
    state: int
    raw: int
    forced: int
    input: int
    readout: int
    words: int
    raw_held: bool


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
    flat in ``3N`` above the first lane tile, and flat in the fold as well: the
    readout gradient's fold sum is the register fragment :func:`_fold_frag` holds, so
    no region here is conditional on the depth.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        fold: Heads one block walks, ``H // G``. Read by no term below.
        span: Source-token block, from :func:`vblock`.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.
    """
    tile = lane_block(dim)
    forcing = _words(gradient_tile(chunk + 1, tile), 4)
    tapped = max(
        _words(gradient_tile(span, tile), 4),
        _words(score_tile(chunk, span), itemsize),
    )
    out = _words(out_tile(chunk, rows), itemsize)
    state = _words(state_tile(rows, tile), itemsize)
    raw = _words(shifted_tile(span, tile), itemsize)
    forced = _words(forced_tile(span, tile), itemsize)
    inp = _words(shifted_tile(mma_rows(span), rows), itemsize)
    read = _words(gradient_tile(mma_rows(chunk), tile), 4)
    base = forcing + tapped
    return Arena(
        forcing=0,
        tapped=forcing,
        out=base,
        state=base + out,
        raw=base + out + state,
        forced=base + out + state + raw,
        input=base + out + state + raw + forced,
        readout=base,
        words=base + max(out + state + raw + forced + inp, read),
        raw_held=read <= out + state,
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
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (offset_tile(chunk, warp_groups), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (gradient_tile(1, lane_block(dim)), 4),
            (quad_table_tile(chunk, 3), 4),
            (row_tile(chunk), 4),
            (row_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (readout_tile(chunk, lane_block(dim)), itemsize),
            (
                Tile(
                    (arena(chunk, rows, dim, fold, span, itemsize).words,),
                    (1,),
                ),
                4,
            ),
            (Tile((TABLE_QUAD,), (1,)), 4),
        ]
    )


def prefix_smem_bytes(chunk: int) -> int:
    """Shared memory :func:`chunk_prefix_bwd_kernel` allocates, in bytes.

    The same tiles that kernel allocates, in the same order, so the budget has one
    description here rather than one at the launch and one in the kernel body. Flat in
    every extent but ``L``: the scan depends on the head and the chunk alone, so no
    tile spans the state width, the fold or the shard.

    Args:
        chunk: ``L``.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (trans_tile(chunk), 4),
        ]
    )


def vblock(
    chunk: int,
    rows: int,
    dim: int,
    fold: int,
    itemsize: int = 2,
    warp_groups: int = 1,
) -> int:
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
        warp_groups: Warp groups of the tiling, as
            :func:`vector_smem_bytes` takes it. The default is not the width the
            operator ships: a caller that launches must pass the width's own group
            count, or the budget this reads is short by ``4 * L`` bytes a group and
            the block it returns is one the launch cannot allocate.

    Returns:
        The block. A shape that fits at neither candidate is refused by
        :func:`slinoss._cute.assert_smem_fits`, not run at a third.
    """
    span = min(chunk, MMA_TILE_M)
    floor = min(chunk, MMA_TILE_M // 2)
    if span > floor:
        budget = vector_smem_bytes(chunk, rows, dim, fold, span, itemsize, warp_groups)
        if budget > smem_capacity():
            span = floor
    return span


def vector_splits(fold: int, splits: int | None = None) -> int:
    """Partial depth of the head sum: shards the fold ``H // G`` is cut into.

    One by default: a block walks the whole fold and writes the three summed outputs
    itself, so there is no workspace and no closing launch. A depth above one buys
    nothing that pays for the launch it costs. Counter sums at
    ``B 4 H 18 T 2048 L 64 P 64 3N 240 G 1`` and the shipped width, one profiled launch
    a counter pass, clocks unlocked, microseconds a launch and MB the workspace:

        depth        1       2       3       6       9      18
        producer 1,583.9 1,707.8 1,653.4 1,632.5 1,563.0 1,423.0
        closure       --    37.5    48.8    84.4   118.0   222.2
        slots      43.5    44.4    44.3    44.4    43.7    44.2
        total    1,627.4 1,789.7 1,746.5 1,761.3 1,724.7 1,689.4
        MB        0.00   15.97   23.96   47.92   71.88  143.77

    Both ends twice, the four between them once. Depth one wins by 62.0 us over the
    full depth, -45.9 and -78.1 across the two pairs, and every depth between the two
    loses to both: the closure is linear in the depth and the producer is not monotone
    in it. The four middle producers sit inside a 9% band whose own pass spread reaches
    14%, so this instrument separates the ends and does not order the middle.

    Nothing here follows a spill: local traffic is zero at every depth. The producer
    pays 160.9 us for the loop at depth one and takes 130.3 MB off its own DRAM, worth
    -31 to -44 us at the rate a deleted byte converts at on this tree, so what orders
    the depths is the closure's launch and not the traffic. A caller with a reason of
    its own can still ask for a depth: the workspace is linear in it, and
    :func:`vector_reduce_kernel` is reachable no other way.

    Args:
        fold: ``H // G``, the heads sharing a group.
        splits: The depth. ``None`` takes one.

    Returns:
        The depth. One by default, and never above the fold.

    Raises:
        ValueError: If ``splits`` is not a positive divisor of the fold. A depth that
            does not divide it would leave a block walking a ragged head count and
            the reduction reading rows no producer wrote.
    """
    if splits is None:
        return 1
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
    over ``dK``, which is a sum over lanes, and the head shard over ``dB``, ``dC`` and
    the carry, which are sums over the heads of a group. The blocks holding the terms
    are concurrent and none can see the others' partials. Each such output gains a slot
    axis immediately outside its row axis: a block writes row ``slot * rows + local``,
    and the ``slots`` copies of a row are summed into the output's own.

    Which sum needs a second launch follows its grid axis, not its width. The lane
    tiles of one ``(chunk, shard)`` are consecutive ``x`` indices, so ``dK``'s slots are
    summed inside the producing launch under the arrival counter ``dtrans`` already
    carries: the tile that arrives last reads the ``slots`` rows of its own
    ``(batch, head, chunk)`` and adds them, which needs no map, no shared memory and no
    zeroed accumulator. The head shards are not adjacent on any axis and ``dB`` and
    ``dC`` are the launch's largest write, so those three keep :func:`vector_reduce`.

    ``dtrans`` needs no slot buffer at all, its remaining maps being linear, so its
    partials cross those maps as :data:`PART_WORDS` published words instead of a row.
    That is available only there: a slot row has already had every map run on it, so
    the only thing left to do to it is add it.

    At ``slots == 1`` there is no buffer and ``dest`` is the output. The row index is
    the same expression either way, ``slot`` being zero, so the kernel body does not
    know which mode it is in and neither mode carries a branch.

    Outside the row axis rather than inside it, so that a slot is one contiguous run of
    the rows a block owns. ``dK``'s closure reads a whole ``(batch, head, chunk)``
    record of a slot as ``L`` tokens of eight adjacent float32, which is the form a
    warp covers in one 128-byte access; inside the row axis the same record would be
    ``tiles``-strided and every closing load would take its own sector.

    Attributes:
        dest: What the kernel writes. The output at one slot, a partial buffer at the
            output's own dtype above one.
        slots: Copies of each row, one per lane tile or one per head shard.
    """

    dest: Tensor
    slots: int


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
    if slots == 1:
        return Slots(dest=out, slots=1)
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
    return Slots(dest=held, slots=slots)


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
    """Sum one accumulator row's partial over the four lanes that share the row.

    The atom gives the four lanes of an aligned quad the same accumulator row and
    disjoint columns, so two butterfly rounds leave that row's partial column sum in
    all four. The atom's C layout is per warp, so the quad is the same four lanes at
    every block width.

    Entered once per accumulator row, not once per element: the caller sums the
    columns one lane holds of a row before crossing the quad, which is the same terms
    over one butterfly instead of one a column.

    Rows are disjoint across quads and across the warps of one warp group, so within
    a group the read-modify-write that follows is by one thread per row and needs no
    barrier. Across warp groups a row is shared: the M mode is not partitioned by the
    N atoms, so a tiling with more than one group gives every row one quad leader per
    group. Each writes its own row of :func:`offset_tile` and the rows are summed
    once, which is a reduction order the widths do not share.

    Args:
        value: The lane's partial over the columns it holds of one row.
    """
    value = value + shuffle_xor(value, 1)
    return value + shuffle_xor(value, 2)


def _sum_over_split(value: Scalar, width: int) -> Scalar:
    """Sum one float over the ``width`` adjacent lanes that share its destination.

    A butterfly over an aligned group, which is what :func:`_sum_over_n` runs at a
    quad. Undecorated: the round count is compile-time and the loop is unrolled
    during the trace.

    Args:
        value: The lane's partial over the segment it holds.
        width: Lanes a group. A power of two, at most the warp.

    Returns:
        The group's sum, in every lane of the group.
    """
    out = value
    reach = 1
    while reach < width:
        out = out + shuffle_xor(out, reach)
        reach *= 2
    return out


def _sum_over_lanes(vals: tuple[Scalar, ...]) -> tuple[Scalar, ...]:
    """Sum a tuple of floats over the :data:`LANE_GROUP` lanes of one token.

    Args:
        vals: One value per component, this lane's partial.

    Returns:
        The group's sum, in every lane of the group.
    """
    return tuple(_sum_over_split(v, LANE_GROUP) for v in vals)


def _mat_sub(a: Mat3, b: Mat3) -> Mat3:
    """Row-major ``a - b``, entry by entry. Nine FADD.

    :func:`slinoss.ops.so3ssd.cute.common.mat3_add` with the second operand negated
    would be nine more instructions, the negation not being foldable into an operand
    on this pipe.

    Args:
        a: The minuend.
        b: The subtrahend.
    """
    return (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
        a[6] - b[6],
        a[7] - b[7],
        a[8] - b[8],
    )


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


def _pass_row(group: cutlass.Int32) -> cutlass.Int32:
    """Row a thread group takes on a rowwise pass, permuted for the vector width.

    A run access is 16 bytes on the float32 tiles, so a phase is eight threads,
    which is two thread groups. The run's segment index is
    ``segments * row + 3 * lane + k``; ``3 * lane`` modulo eight is ``{0,3,6,1}``
    and two rows one apart differ by ``segments`` modulo eight, which
    :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch` makes odd, so consecutive rows
    put the two groups of a phase on the same banks. Rows four apart differ by
    ``4 * segments`` modulo eight, which is four for every odd segment count, and
    ``{0,3,6,1} + 4`` is that set's complement, so the phase is conflict-free at
    every pitch the pitch function can return.

    A bijection on each aligned block of eight groups, and ``per_pass`` is a
    multiple of eight at both block widths, so a warp covers the same row set at
    the same step. Every other access of a pass is indexed by the row alone -- the
    table entry, the nine-word scratch, the tap and transition rows, the ``dK``
    store -- so each keeps its address set and its conflict profile.

    Args:
        group: ``tid // LANE_GROUP + step * per_pass``.
    """
    low = group % 8
    return group - low + low // 2 + 4 * (low % 2)


def _run_vec(width: int, itemsize: int) -> int:
    """Elements one access covers over a run of ``width`` adjacent columns.

    The largest power of two that divides the run and fits a 16-byte segment. A run
    is ``3 * N / LANE_GROUP`` elements, twelve at every legal shape, so this is four
    at either element width and a run is three accesses.

    Args:
        width: Elements of the run.
        itemsize: Bytes per element.
    """
    vec = 1
    while 2 * vec * itemsize <= SMEM_SEGMENT and width % (2 * vec) == 0:
        vec *= 2
    return vec


def _runs(tile: cute.Tensor, vec: int) -> cute.Tensor:
    """View a row-major tile in units of ``vec`` adjacent elements.

    :func:`slinoss.ops.so3ssd.cute.table.paired` at a width the staging pass has no
    use for: a rowwise epilogue thread owns a whole run of ``N / LANE_GROUP`` lanes
    rather than a pair, so the run is six accesses fewer at four elements than at
    two. Widening that function instead is an edit to the file the staging pass
    owns.

    Args:
        tile: ``(rows, pitch)`` shared tile, unit stride on the columns and a pitch
            from :func:`slinoss.ops.so3ssd.cute.mma.smem_pitch`.
        vec: Elements an access covers, from :func:`_run_vec`.

    Returns:
        The retiled view. Element ``(None, (r, k))`` is elements ``vec * k`` through
        ``vec * k + vec - 1`` of row ``r``, statically shaped, which is what lets
        :func:`cutlass.cute.autovec_copy` pick one access rather than ``vec``.

    Invariants:
        The pitch is a whole number of 16-byte segments, so a row starts on a
        16-byte boundary, and a run offset is ``width * lane`` elements with
        ``vec`` dividing ``width``. The claim is restated on the iterator because a
        tile arriving as a parameter reports one element whatever its allocation
        asked for, and ``autovec_copy`` caps the access at the claim.
    """
    base = tile.iterator.align(vec * (tile.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, tile.layout), (1, vec))


def _run_at(row: cute.Tensor, vec: int) -> cute.Tensor:
    """View one contiguous global row in units of ``vec`` adjacent elements.

    Args:
        row: ``(3N,)`` unit-stride view of one row of a global tensor.
        vec: Elements an access covers, from :func:`_run_vec`.

    Invariants:
        The lane origin and the row stride are both multiples of 16 bytes -- ``3N``
        is a multiple of 48 and a pitched band's stride lands on a segment -- and a
        run offset is a multiple of ``vec``.
    """
    base = row.iterator.align(vec * (row.element_type.width // 8))
    return cute.zipped_divide(cute.make_tensor(base, row.layout), (vec,))


def _run_vals(
    frag: cute.Tensor, vec: int, width: int, base: int = 0
) -> tuple[Scalar, ...]:
    """A run fragment's elements in column order, widened to float32.

    Args:
        frag: ``(width // vec, vec)`` fragment filled by :func:`_read_run`, or a taller
            one holding several runs.
        vec: Elements an access covers.
        width: Elements of the run.
        base: First access row of the run. Nonzero only where one fragment holds the
            runs of several steps.
    """
    elem = frag.element_type
    return tuple(widen(frag[base + i // vec, i % vec], elem) for i in range(width))


@cute.jit
def _read_run(
    words: cute.Tensor,
    frag: cute.Tensor,
    row: cutlass.Int32,
    lane: cutlass.Int32,
    accs: cutlass.Constexpr,
) -> None:
    """Read a thread's whole column run of one row.

    Args:
        words: A tile from :func:`_runs`.
        frag: ``(accs, vec)`` fragment of the tile's element type.
        row: Row of the tile.
        lane: Lane group of the thread. Its run starts at access ``accs * lane``.
        accs: Accesses the run takes. Compile-time.
    """
    for k in cutlass.range_constexpr(accs):
        cute.autovec_copy(words[(None, (row, accs * lane + k))], frag[(k, None)])


@cute.jit
def _write_run(
    row: cute.Tensor,
    frag: cute.Tensor,
    lane: cutlass.Int32,
    accs: cutlass.Constexpr,
    vals: tuple[Scalar, ...],
) -> None:
    """Write a run of float32 values to one global row, narrowed here.

    Args:
        row: A row from :func:`_run_at`.
        frag: ``(accs, vec)`` fragment of the row's element type.
        lane: Lane group of the thread.
        accs: Accesses the run takes. Compile-time.
        vals: The run's values in column order, float32.
    """
    elem = frag.element_type
    vec = len(vals) // accs
    for k in cutlass.range_constexpr(accs):
        for j in cutlass.range_constexpr(vec):
            frag[k, j] = narrow(vals[vec * k + j], elem)
        cute.autovec_copy(frag[(k, None)], row[(None, accs * lane + k)])


@cute.jit
def _add_run(
    words: cute.Tensor,
    frag: cute.Tensor,
    row: cutlass.Int32,
    lane: cutlass.Int32,
    accs: cutlass.Constexpr,
    vals: tuple[Scalar, ...],
) -> None:
    """Add a run of float32 values into a float32 row, read-modify-write.

    One vector read and one vector write an access, in place of one scalar pair an
    element. A ``(row, column)`` of either destination is reached by one thread of
    the block between two barriers, so batching the run reorders nothing against
    another thread.

    Args:
        words: A float32 tile from :func:`_runs`.
        frag: ``(accs, vec)`` float32 fragment.
        row: Row of the tile.
        lane: Lane group of the thread.
        accs: Accesses the run takes. Compile-time.
        vals: The run's addends in column order.
    """
    vec = len(vals) // accs
    _read_run(words, frag, row, lane, accs)
    for k in cutlass.range_constexpr(accs):
        for j in cutlass.range_constexpr(vec):
            frag[k, j] = frag[k, j] + vals[vec * k + j]
        cute.autovec_copy(frag[(k, None)], words[(None, (row, accs * lane + k))])


def _fold_frag(threads: int, chunk: int, lanes: int, fold: int) -> cute.Tensor:
    """The readout gradient's sum over the fold, in registers.

    :func:`_readout_epilogue` covers the ``(L, tile)`` gradient by giving thread
    ``tid`` of pass ``step`` the run of ``3 * lanes / LANE_GROUP`` adjacent columns at
    lane ``tid % LANE_GROUP`` of token ``_pass_row(tid // LANE_GROUP + step *
    per_pass)``. Both coordinates depend on ``tid`` and ``step`` alone, so the cover is
    the same disjoint cover at every head of the shard and a thread may hold its own
    runs across the fold instead of reading them back out of shared memory. That is
    the whole of why the sum needs no region in :func:`arena`, and the region it needed
    is 13,312 B -- the exact amount that put a source-token block of 64 over the
    carveout at every depth below the fold.

    Twelve float32 a thread at every legal shape and one pass, twenty-four at the
    narrow block width where a pass covers half the chunk.

    Args:
        threads: Block width.
        chunk: ``L``.
        lanes: ``N`` of one lane tile.
        fold: Heads one block walks. One holds no sum: the epilogue stores each head's
            run straight to the output, so the fragment is a single dead access there
            rather than a live run.

    Returns:
        The ``(passes * accs, vec)`` fragment, indexed ``[step * accs + k, j]`` by the
        pass, the access within the run and the element within the access.
    """
    width = 3 * (lanes // LANE_GROUP)
    vec = _run_vec(width, 4)
    accs = width // vec
    passes = -(-chunk // (threads // LANE_GROUP))
    return cute.make_fragment((passes * accs if fold > 1 else 1, vec), cutlass.Float32)


@cute.jit
def _store_run(
    row: cute.Tensor,
    words: cute.Tensor,
    ofrag: cute.Tensor,
    sfrag: cute.Tensor,
    src: cutlass.Int32,
    run: cutlass.Int32,
    accs: cutlass.Constexpr,
    vec: cutlass.Constexpr,
) -> None:
    """Narrow one run of a float32 shared row into one run of a global row.

    A float32 access covers ``vec`` elements and an output access covers
    ``accs * vec``, the two widths differing because the segment holds twice as many
    narrowed elements as float32 ones. The narrowing is elementwise on the same
    float32 values in the same order as a scalar store, so the result is unchanged.

    Args:
        row: A row from :func:`_run_at`, in units of ``accs * vec`` elements.
        words: A float32 tile from :func:`_runs`, in units of ``vec`` elements.
        ofrag: ``(accs * vec,)`` fragment of the row's element type.
        sfrag: ``(accs, vec)`` float32 fragment.
        src: Row of ``words``.
        run: Run of ``row``.
        accs: Float32 accesses one output access covers. Compile-time.
        vec: Elements a float32 access covers. Compile-time.
    """
    elem = ofrag.element_type
    for k in cutlass.range_constexpr(accs):
        cute.autovec_copy(words[(None, (src, accs * run + k))], sfrag[(k, None)])
    for k in cutlass.range_constexpr(accs):
        for j in cutlass.range_constexpr(vec):
            ofrag[vec * k + j] = narrow(sfrag[k, j], elem)
    cute.autovec_copy(ofrag, row[(None, run)])


def _load_cg(ptr: cute.Pointer) -> cutlass.Float32:
    """One float32 from L2, bypassing L1.

    ``ld.global.cg``. There is no wrapper for it in the DSL, so it is inline asm:
    ``toint`` gives the address the ``l`` constraint takes, which is how the vendor's
    own distributed primitives pass a pointer to asm. Marked as having side effects so
    the load is not hoisted out of the branch that established that the data exists.

    Args:
        ptr: Global float32 address.

    Returns:
        The value.

    Invariants:
        The publishing store is another block's and L1 is not coherent across
        multiprocessors, so a plain load may read a line the store never touched. The
        acquire fence orders the reader's own accesses; nothing in the memory model
        makes it invalidate that line. Assembles to ``LDG.E.STRONG.GPU`` on sm_86,
        where the fence in turn assembles to ``MEMBAR.ALL.GPU`` and ``CCTL.IVALL``:
        the qualifier is what the model guarantees, the invalidate is what one ptxas
        chose.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [ptr.toint().ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _sum_slots(
    part: cute.Tensor, word: int, token: cutlass.Int32, tiles: int
) -> cutlass.Float32:
    """Sum one published chart word over the lane tiles, lowest slot first.

    Args:
        part: ``(tiles * PART_WORDS, L)`` float32 view of one ``(batch, head, chunk)``
            of the chart partials, slot outside word.
        word: Word of :data:`PART_WORDS`. Compile-time.
        token: Token within the chunk.
        tiles: Lane tiles. Compile-time.

    Returns:
        The sum.

    Invariants:
        Ascending slot index, so the sum is the same term order whichever tile
        arrives last. The tile's own slot is read back rather than taken from shared
        memory for the same reason: one order, not one order per arriver.
    """
    total = cutlass.Float32(0.0)
    for slot in range(tiles):
        total = total + _load_cg(
            part.iterator + part.layout((slot * PART_WORDS + word, token))
        )
    return total


@cute.jit
def _fill_zero(
    dst: cute.Tensor, total: cutlass.Constexpr, tid: cutlass.Int32, threads: int
) -> None:
    """Zero a dense shared tile, padding included.

    One :data:`TABLE_QUAD`-word segment a thread a step. A thread's four words are
    adjacent and a step's threads cover ``TABLE_QUAD * threads`` adjacent words, so
    the step count falls by four and the wavefront count is unchanged: a warp covered
    128 B in one access and now covers 512 B in four. The widest fill at
    ``L 64/P 64/3N 240`` goes fourteen steps to four.

    Args:
        dst: Shared float32 tile whose storage is dense and segment-aligned.
        total: Elements the tile spans, padding included. Compile-time.
        tid: Thread index within the block.
        threads: Block width. Compile-time.

    Invariants:
        The claim is restated on the iterator because a tile arriving as a parameter
        reports one element whatever its allocation asked for, and ``autovec_copy``
        caps the access at the claim. ``total`` is a multiple of :data:`TABLE_QUAD`
        at every legal shape -- a float32 pitch is an odd multiple of four words, and
        :func:`slinoss.ops.so3ssd.cute.guard.check_extents` holds ``L`` to a multiple
        of 16 -- so the scalar tail below is dead code and traces away.
    """
    quads = total // TABLE_QUAD
    zero = cute.make_fragment((TABLE_QUAD,), cutlass.Float32)
    zero.fill(0.0)
    words = cute.make_tensor(
        dst.iterator.align(SMEM_SEGMENT),
        cute.make_layout((quads, TABLE_QUAD), stride=(TABLE_QUAD, 1)),
    )
    for step in cutlass.range_constexpr(-(-quads // threads)):
        q = tid + step * threads
        if cutlass.const_expr(quads % threads == 0):
            cute.autovec_copy(zero, words[(q, None)])
        else:
            if q < quads:
                cute.autovec_copy(zero, words[(q, None)])
    if cutlass.const_expr(total % TABLE_QUAD):
        flat = cute.make_tensor(dst.iterator, cute.make_layout((total,), stride=(1,)))
        if tid < total - quads * TABLE_QUAD:
            flat[quads * TABLE_QUAD + tid] = 0.0


@cute.jit
def _copy_words(
    src: cute.Tensor,
    dst: cute.Tensor,
    total: cutlass.Constexpr,
    tid: cutlass.Int32,
    threads: int,
) -> None:
    """Copy a dense float32 run between two spaces, one segment a thread a step.

    Either end may be global or shared. Both runs are dense and segment-aligned, so
    the copy is one access a thread a step and the direction is the caller's.

    Args:
        src: Dense float32 tile or row of at least ``total`` elements.
        dst: Dense float32 tile or row of at least ``total`` elements.
        total: Elements to copy. Compile-time, and a multiple of
            :data:`TABLE_QUAD` because
            :func:`slinoss.ops.so3ssd.cute.guard.check_extents` holds ``L`` to a
            multiple of 16.
        tid: Thread index within the block.
        threads: Block width. Compile-time.

    Invariants:
        The alignment claim is restated on both iterators, because a tile arriving
        as a parameter reports one element whatever its allocation asked for and
        ``autovec_copy`` caps the access at the claim. A global run's origin is a
        whole number of ``L``-element or ``4L``-element records, and ``L`` is a
        multiple of 16, so a record starts on a segment.
    """
    quads = total // TABLE_QUAD
    unit = cute.make_layout((quads, TABLE_QUAD), stride=(TABLE_QUAD, 1))
    from_words = cute.make_tensor(src.iterator.align(SMEM_SEGMENT), unit)
    to_words = cute.make_tensor(dst.iterator.align(SMEM_SEGMENT), unit)
    for step in cutlass.range_constexpr(-(-quads // threads)):
        q = tid + step * threads
        if cutlass.const_expr(quads % threads == 0):
            cute.autovec_copy(from_words[(q, None)], to_words[(q, None)])
        else:
            if q < quads:
                cute.autovec_copy(from_words[(q, None)], to_words[(q, None)])


@cute.jit
def _hold_b(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    vb: cute.Tensor,
    b_k_major: cutlass.Constexpr,
) -> cute.Tensor:
    """Load one right operand's fragment, for reuse across several products.

    :func:`slinoss.ops.so3ssd.cute.mma.mma_gemm` loads both operands on every call,
    so a tile several products read pays one ``ldmatrix`` set per product. The
    barrier between the products is what stops the compiler from merging the loads:
    it fences shared memory, so the second load cannot be proved redundant. Splitting
    the load from the product moves the proof to the caller, which is where the
    invariant that licenses it lives.

    Args:
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`.
        tid: Thread index within the block.
        vb: Shared-memory view of shape ``(N,K)``.
        b_k_major: Whether ``vb``'s K mode is the stride-1 mode.

    Returns:
        Register-backed B fragment for :func:`_gemm_bheld`, ``N * K / 32`` elements
        of ``vb``'s dtype a thread.

    Invariants:
        A held fragment is only the tile's value while no thread writes the tile. A
        write between the load and a use is a stale operand and no barrier catches
        it, so the caller must own that range.
    """
    thr = tiled_mma.get_slice(tid)
    fb = tiled_mma.make_fragment_B(thr.partition_B(vb))
    atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not b_k_major, mma_matrices(tiled_mma)),
        vb.element_type,
    )
    copy = cute.make_tiled_copy_B(atom, tiled_mma)
    slc = copy.get_slice(tid)
    cute.copy(copy, slc.partition_S(vb), slc.retile(fb))
    return fb


@cute.jit
def _gemm_bheld(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    acc: cute.Tensor,
    va: cute.Tensor,
    a_k_major: cutlass.Constexpr,
    fb: cute.Tensor,
) -> None:
    """Accumulate ``va @ vb^T`` into ``acc`` with B already in registers.

    The mirror of :func:`slinoss.ops.so3ssd.cute.mma.mma_gemm_areg`, which holds the
    left operand instead. No group restriction applies: the B fragment is the
    tiling's own partition of a shared tile, not a reread of a C fragment, so a
    split N mode partitions it rather than scattering it.

    Args:
        tiled_mma: From :func:`slinoss.ops.so3ssd.cute.mma.make_mma`. The same one
            that produced ``fb``.
        tid: Thread index within the block.
        acc: From :func:`slinoss.ops.so3ssd.cute.mma.mma_acc`. Updated in place.
        va: Shared-memory view of shape ``(M,K)``.
        a_k_major: Whether ``va``'s K mode is the stride-1 mode.
        fb: From :func:`_hold_b`, over the same K extent ``va`` carries.
    """
    thr = tiled_mma.get_slice(tid)
    fa = tiled_mma.make_fragment_A(thr.partition_A(va))
    atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not a_k_major, 4), va.element_type
    )
    copy = cute.make_tiled_copy_A(atom, tiled_mma)
    slc = copy.get_slice(tid)
    cute.copy(copy, slc.partition_S(va), slc.retile(fa))
    cute.gemm(tiled_mma, acc, fa, fb, acc)


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
def _stage_run(
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gscore: cute.Tensor,
    sb: cute.Tensor,
    su: cute.Tensor,
    sscore: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    nbase: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    tile: cutlass.Constexpr,
    span: cutlass.Constexpr,
    spad: cutlass.Constexpr,
    mpad: cutlass.Constexpr,
    has_prev: cutlass.Constexpr,
    forcing: cutlass.Constexpr,
) -> None:
    """Issue one source-token block's staged tiles, asynchronously, no wait.

    Stated once because it has two call sites: hoisted for the first block, whose
    write-after-read fence is a barrier the head already carries, and in the block
    loop for every later one, whose fence is the barrier that closed the previous
    iteration. Neither reads the transform table, so neither has to follow the build.

    The passes sit together so their global loads overlap: each one's issues cover the
    next one's runs, and the caller's single ``cp_async_wait_group`` retires them all.

    Args:
        gb: ``(B,G,T,tile)`` lane view of the forcing operand.
        gbprev: ``(B,G,tile)`` streaming predecessor of ``gb``.
        gu: ``(B,H,T,P)`` input operand.
        guprev: ``(B,H,P)`` streaming predecessor of ``gu``.
        gscore: ``(L,L)`` masked score record of this head and chunk.
        sb: Operand-dtype tile of ``span + 1`` rows, written when ``forcing``.
        su: Operand-dtype tile of ``spad + 1`` rows, written.
        sscore: Operand-dtype ``(L, span)`` tile, written.
        bidx: Batch index.
        gidx: Group index, which is what the lane view of ``gb`` is sliced on.
        hidx: Head index.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        nbase: First chunk-local token of the block. Compile-time, so the shifted
            passes fold their row offsets as they did at the call site this replaces.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        tile: Lane-tile width. Compile-time.
        span: Tokens of the block. Compile-time.
        spad: ``span`` rounded to the atom's M mode. Compile-time.
        mpad: ``chunk`` rounded to the atom's M mode. Compile-time.
        has_prev: Whether a streaming predecessor exists. Compile-time.
        forcing: Include the forcing pass. Compile-time. False where the caller staged
            that tile once for the block instead, which it may do only when the block
            loop has one iteration, ``nbase`` being the pass's only per-iteration term.
    """
    if cutlass.const_expr(forcing):
        stage_shifted(
            gb,
            gbprev,
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
            True,
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
        True,
    )
    stage_score(
        gscore,
        sscore,
        nbase,
        tid,
        threads,
        chunk,
        span,
        mpad - chunk,
        True,
    )


@cute.jit
def _tap_epilogue(
    gdtap: cute.Tensor,
    sdb: cute.Tensor,
    sb: cute.Tensor,
    ssum: cute.Tensor,
    stable: cute.Tensor,
    stap: cute.Tensor,
    strans: cute.Tensor,
    srow: cute.Tensor,
    srow2: cute.Tensor,
    sdw: cute.Tensor,
    sdw2: cute.Tensor,
    sdk: cute.Tensor,
    bidx: cutlass.Int32,
    hidx: cutlass.Int32,
    t0: cutlass.Int32,
    nbase: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    jbase: cutlass.Int32,
    threads: cutlass.Constexpr,
    span: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Turn the fused forcing gradient into ``dB``, both taps of ``dK`` and four sums.

    The fused column's transform is ``Afuse(t) = Ap(t) + e_t An(t-1)`` with
    ``e_t = exp(2 ls_t)``, and its forcing input is token ``t-1``'s. So one pass owns
    the whole of ``dAfuse(t) = sum_n outer(dbfuse_n(t), b_n(t-1))``, written ``D``
    below, and deposits at two tokens::

        dbs         = Afuse(t)^T dbfuse            into the forcing sum, row t
        srow[t]    += D Ap(t)^T
        srow2[t-1]  = e_t D An(t-1)^T
        dKprev(t)   = tap_matrix_vjp(Ac(t)^T D, ...)
        dKcurr(t-1)+= e_t tap_matrix_vjp(Ac(t-1)^T D, ...)
        dls_step(t) = 2 e_t <An(t-1), D>

    Every term but the forcing sum is a 3x3 product against ``D``, so the lane
    reduction runs once a token. The slot form reduced a rotation and a tap matrix
    per tap, four butterflies where this runs one, and recomputed a rotated vector
    per lane where this recomputes none.

    ``Ap(t)`` is not a table slot under fusion: slot :data:`TABLE_AFUSE` holds
    ``Afuse``, so ``D Ap(t)^T`` is formed as ``D Afuse(t)^T - e_t D An(t-1)^T``, and
    ``D Afuse(t)^T`` reuses the transpose the forcing sum already needs.

    The rotation terms are the collapsed form: the trailing ``Ac`` is deferred to the
    closure, so no raw readout vector is read. The tap terms do not collapse, which
    is the only reason the raw forcing tile is staged.

    ``t-1`` is a destination two threads of one pass reach, so the deposits split into
    a group indexed by the row's own token and a group indexed by the row before it.
    The second group writes its own buffers, ``srow2`` and ``sdw2``, so the two groups
    share no address and the pass needs no barrier between them. Two barriers a step
    go: the one between the halves and the one closing the step. The chunk's first
    token is predicated out of the second group rather than left to ``e_0 = 0``: the
    clamp would put two threads on row 0 at once.

    The second group's deposits are stores, not read-modify-writes. Each destination
    row is written by the thread group holding token ``r + 1``, ``_pass_row`` is a
    bijection on each aligned block of eight groups so the group range of a pass is
    permuted and not duplicated, and the runs of successive blocks are disjoint, so
    row ``r`` of ``srow2`` and ``sdw2`` is reached exactly once a head. Six shared
    loads and six adds a warp go with the read half.

    ``sdk`` needs no split: the second group is its only writer here.

    The buffers are summed into ``srow`` and ``sdw`` by :func:`_readout_epilogue`,
    which already stands behind a barrier of its own, so the sum is a deletion and not
    a move. It is folded ahead of that pass's own deposit, which keeps the term order
    the single buffer had and the result bit-identical.

    Measured on an A6000 at the acceptance shape, where ``steps`` is one and the
    barrier at the end of a step never lowers: -22.016 and -20.992 us over 400 pairs
    each, bitwise on all five outputs, against a +0.512 us null control interleaved in
    the same run. The launch executes 820,224 more instructions than the single-buffer
    form and 311,040 more ``STS``, so the win is latency and not instruction count.

    Its own barrier stall priced it at -4.7 us, 4.6x under: that barrier carried 1.89%
    of the launch's barrier stall, rank 17 of 18, 0.26% of its PC samples. The stall
    partition says why. ``mio_throttle`` fell 27,093 -> 21,436 samples and
    ``short_scoreboard`` 6,765 -> 4,337, together -3.87% of the launch's samples, while
    ``barrier`` itself ROSE 29,039 -> 32,036. A fence between two shared-heavy halves
    is paid by the instructions on both sides of it, which queue against a full MIO
    pipe in two serialized bursts instead of one interleaved stream; the fence's own
    stall is a lower bound on its price. Barrier samples rising on a launch that got
    shorter is the same effect from the other side: a warp that no longer waits here
    waits at the next barrier instead.

    The barrier counter is not a barrier instrument on this kernel at all, and the
    residency isolation above shows it from a third direction: making the second block
    resident raises ``barrier`` 37%, ``mio_throttle`` 53% and ``math_pipe_throttle`` 40%
    while elapsed cycles fall a third, because twice the warps are resident to be
    stalled and issue-active cycles are conserved. Price a barrier arm on cycles or
    duration against a control, never on the counter's own share.

    Nothing else moved. Registers 138 -> 140, local sectors zero on both, achieved
    occupancy 16.50% -> 16.48%, global store sectors and store requests bit-identical,
    DRAM read and write within 0.06%. ``smsp__issue_active`` 37.47% -> 38.19% is the
    whole win: the same warps issue denser.

    The forcing sum is indexed by token rather than by row of the run: row ``t + 1``
    is token ``t``, so the fused column's ``dB`` at token ``t-1`` lands on row ``t``
    and the chunk's first token lands on row 0, which is the carry.

    Args:
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer. Only the
            previous tap's rows are written here; the current tap's accumulate in
            ``sdk``.
        sdb: ``(span, pitch)`` float32 forcing gradient, the GEMM's output.
        sb: ``(span + 1, pitch)`` operand-dtype raw forcing tile. Row ``r`` is token
            ``nbase + r - 1``.
        ssum: ``(L + 1, pitch)`` float32 forcing sum, accumulated.
        stable: ``(mats, L, TABLE_PITCH)`` float32 transform table, fused.
        stap: ``(8, L)`` float32 tap parameters, component-major.
        strans: ``(4, L)`` float32 ``(w, ls)``, component-major.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated at the row's
            own token.
        srow2: ``(L, ROW_WORDS)`` float32 rotation scratch, stored at the row before
            the pass's own token. Row ``L-1`` is never reached and is zeroed by the
            caller.
        sdw: ``(4, L)`` float32 cotangent of ``strans``, accumulated at the row's own
            token. Row 3 is the fused column's own ``dls``, which is the cotangent of
            the component ``strans`` holds there.
        sdw2: ``(4, L)`` float32 cotangent of ``strans``, stored at the row before the
            pass's own token. Rows 0 to 2 only; row 3 has no ``t-1`` contributor.
        sdk: ``(4, L)`` float32 current-tap cotangent, accumulated. Not stored
            through: that tap has a contributor from token ``t+1``'s pass and one
            from the readout epilogue, so no single pass holds the whole of it.
        bidx: Batch index.
        hidx: Head index.
        t0: First token of the chunk.
        nbase: First chunk-local token of the run.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        jbase: First row of this block's slot, ``jstep * T``. Zero at one lane tile,
            where the destination is ``dK`` itself.
        threads: Block width. Compile-time.
        span: Tokens of the run. Compile-time.
        lanes: Lanes of the lane tile. Compile-time.
    """
    per_pass = threads // LANE_GROUP
    steps = -(-span // per_pass)
    exact = span % per_pass == 0
    lane = tid % LANE_GROUP
    zero = cutlass.Float32(0.0)

    # A thread owns a run of adjacent lanes, so its three-vectors are one adjacent
    # column run and each of the three tiles it touches is read at vector width. The
    # strided map this replaced gave a thread every fourth lane, which is three
    # scalar accesses a lane at either layout of the trailing axis.
    width = 3 * (lanes // LANE_GROUP)
    fvec = _run_vec(width, 4)
    ovec = _run_vec(width, sb.element_type.width // 8)
    faccs = width // fvec
    oaccs = width // ovec
    grad = _runs(sdb, fvec)
    total = _runs(ssum, fvec)
    raws = _runs(sb, ovec)
    dfrag = cute.make_fragment((faccs, fvec), sdb.element_type)
    bfrag = cute.make_fragment((oaccs, ovec), sb.element_type)
    sfrag = cute.make_fragment((faccs, fvec), ssum.element_type)
    kfrag = cute.make_fragment((1, 4), gdtap.element_type)

    for step in cutlass.range_constexpr(steps):
        r = _pass_row(tid // LANE_GROUP + step * per_pass)
        # Clamped rather than branched: a row past the run reads real data whose
        # every use below is predicated away.
        rs = cutlass.min(r, span - 1)
        token = nbase + rs
        prev = cutlass.max(token - 1, 0)
        inside = r < span
        afuset = mat3_transpose(_mat_at(stable, TABLE_AFUSE, token))
        dmat = tuple(zero for _ in range(9))
        _read_run(grad, dfrag, rs, lane, faccs)
        _read_run(raws, bfrag, rs, lane, oaccs)
        dvals = _run_vals(dfrag, fvec, width)
        bvals = _run_vals(bfrag, ovec, width)
        outs: list[Scalar] = []
        for rep in cutlass.range_constexpr(width // 3):
            dvec = (dvals[3 * rep], dvals[3 * rep + 1], dvals[3 * rep + 2])
            bvec = (bvals[3 * rep], bvals[3 * rep + 1], bvals[3 * rep + 2])
            dmat = mat3_add(dmat, mat3_outer(dvec, bvec))
            outs.extend(mat3_matvec(afuset, dvec))
        if cutlass.const_expr(exact):
            _add_run(total, sfrag, token, lane, faccs, tuple(outs))
        else:
            if inside:
                _add_run(total, sfrag, token, lane, faccs, tuple(outs))
        dmat = _sum_over_lanes(dmat)

        # ``e_t`` of the chunk's first token is zero, matching what
        # :func:`slinoss.ops.so3ssd.cute.table.build_table` composed the slot from.
        estep = select(token > 0, decay(strans[3, token]), zero)
        anprev = _mat_at(stable, TABLE_AN, prev)
        # ``e_t`` rides the nine words in rather than the products out. Every deposit
        # below the barrier is a shared read-modify-write whose only other operand
        # would be a multiply by it, and a lone multiply feeding an add is
        # contractible: the backend contracts it at four warps and not at eight, which
        # made this kernel's rotation cotangent depend on the launch width. Scaled
        # here, each product has several consumers and no add can absorb one.
        dstepmat = tuple(estep * d for d in dmat)
        gprev = mat3_mul(dstepmat, mat3_transpose(anprev))
        gnow = _mat_sub(mat3_mul(dmat, afuset), gprev)
        mnow = mat3_mul(mat3_transpose(_mat_at(stable, TABLE_AC, token)), dmat)
        mprev = mat3_mul(mat3_transpose(_mat_at(stable, TABLE_AC, prev)), dstepmat)
        # ``decay`` is ``exp(2 ls)``, so the derivative of the fused slot in ``ls_t``
        # is twice the term ``e_t`` multiplies, contracted against its own cotangent.
        # The two is exact, so folding it into the deposit changes no bit.
        dstep = zero
        for e in cutlass.range_constexpr(9):
            dstep = dstep + anprev[e] * dstepmat[e]

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
            held = _spread(gnow[base : base + LANE_GROUP], lane)
            if rows:
                srow[token, base + lane] += held
        if keep:
            dtap, dw = tap_matrix_vjp(
                mnow,
                (stap[0, token], stap[1, token], stap[2, token]),
                (strans[0, token], strans[1, token], strans[2, token]),
            )
            for j in cutlass.range_constexpr(3):
                sdw[j, token] += dw[j]
            sdw[3, token] += 2.0 * dstep
            if token < valid:
                krow = jbase + t0 + token
                # One access a row, on :func:`chunk_vector_bwd_kernel`'s account. Lane 3
                # of K is a hard zero in the forward, so it is one here and the run is
                # the whole row.
                _write_run(
                    _run_at(gdtap[bidx, hidx, krow, 0, None], 4),
                    kfrag,
                    0,
                    1,
                    (dtap[0], dtap[1], dtap[2], zero),
                )

        # Everything above is indexed by the row's own token, everything below by the
        # row before it, and a pass holds adjacent tokens on adjacent thread groups. The
        # two groups write disjoint buffers, so no barrier separates them and none
        # closes the step.
        back = token > 0
        if cutlass.const_expr(not exact):
            back = back & inside
        for word in cutlass.range_constexpr(-(-ROW_WORDS // LANE_GROUP)):
            base = word * LANE_GROUP
            rows = back & (lane < ROW_WORDS - base)
            held = _spread(gprev[base : base + LANE_GROUP], lane)
            if rows:
                srow2[prev, base + lane] = held
        if keep & back:
            # Its cotangent carries ``e_t`` already, so nothing is scaled on the way
            # out: every deposit here is a bare store or a bare add.
            dtapp, dwp = tap_matrix_vjp(
                mprev,
                (stap[4, prev], stap[5, prev], stap[6, prev]),
                (strans[0, prev], strans[1, prev], strans[2, prev]),
            )
            for j in cutlass.range_constexpr(3):
                sdk[j, prev] += dtapp[j]
                sdw2[j, prev] = dwp[j]


def _readout_pass(
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    itemsize: cutlass.Constexpr,
) -> tuple[int, int, int, int]:
    """The stride-loop geometry of :func:`_readout_epilogue`.

    Args:
        threads: Block width.
        chunk: ``L``.
        lanes: ``N``.
        itemsize: Bytes of the operand dtype.

    Returns:
        The tokens one pass covers, the steps a thread takes, the run width of the
        operand-dtype forcing read, and the runs a row holds.
    """
    per_pass = threads // LANE_GROUP
    width = 3 * (lanes // LANE_GROUP)
    ovec = _run_vec(width, itemsize)
    return per_pass, -(-chunk // per_pass), ovec, width // ovec


def readout_bfrag(
    gb: cute.Tensor,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> cute.Tensor:
    """The register file :func:`read_readout_b` fills.

    Args:
        gb: ``(B,G,T,tile)`` lane view of the forcing vectors, for its element type.
        threads: Block width.
        chunk: ``L``.
        lanes: ``N``.

    Returns:
        ``(steps * oaccs, ovec)``, indexed ``[step * oaccs + k, j]``.
    """
    size = gb.element_type.width // 8
    _, steps, ovec, oaccs = _readout_pass(threads, chunk, lanes, size)
    return cute.make_fragment((steps * oaccs, ovec), gb.element_type)


@cute.jit
def read_readout_b(
    gb: cute.Tensor,
    bfrag: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
) -> None:
    """Issue the forcing read :func:`_readout_epilogue` consumes.

    Split out because the read is head-invariant while the pass around it is not: the
    address is the batch, the group, the chunk and the thread, so one issue serves every
    head the block walks.

    Args:
        gb: ``(B,G,T,tile)`` lane view of the operand-dtype forcing vectors.
        bfrag: The fragment from :func:`readout_bfrag`.
        bidx: Batch index.
        gidx: Group index.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        lanes: ``N``. Compile-time.
    """
    size = gb.element_type.width // 8
    per_pass, steps, ovec, oaccs = _readout_pass(threads, chunk, lanes, size)
    lane = tid % LANE_GROUP
    for step in cutlass.range_constexpr(steps):
        ts = cutlass.min(_pass_row(tid // LANE_GROUP + step * per_pass), chunk - 1)
        # A pad token's row is clamped into range: its diagonal scalar is zero and its
        # closing three-vector is zero, so every use of the value is zero-valued.
        brow = _run_at(gb[bidx, gidx, t0 + cutlass.min(ts, valid - 1), None], ovec)
        for k in cutlass.range_constexpr(oaccs):
            cute.autovec_copy(
                brow[(None, oaccs * lane + k)], bfrag[(step * oaccs + k, None)]
            )


@cute.jit
def _readout_epilogue(
    gdc: cute.Tensor,
    bfrag: cute.Tensor,
    sdc: cute.Tensor,
    csum: cute.Tensor,
    bsum: cute.Tensor,
    scrot: cute.Tensor,
    stable: cute.Tensor,
    stap: cute.Tensor,
    strans: cute.Tensor,
    srow: cute.Tensor,
    srow2: cute.Tensor,
    sdw: cute.Tensor,
    sdw2: cute.Tensor,
    sdk: cute.Tensor,
    sdnow: cute.Tensor,
    sdures: cute.Tensor,
    bidx: cutlass.Int32,
    gidx: cutlass.Int32,
    sbase: cutlass.Int32,
    t0: cutlass.Int32,
    valid: cutlass.Int32,
    tid: cutlass.Int32,
    hstep: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    lanes: cutlass.Constexpr,
    fold: cutlass.Constexpr,
) -> None:
    """Turn one head's readout gradient and the two fused residues into their outputs.

    Per token and lane, ``dc = ac^T dcrot`` and ``rotation += outer(dcrot, crot)``,
    the same collapsed form the tap epilogue accumulates into.

    The one-tap column carries a source token one behind its own, so two terms of the
    map have no home in it and close here instead. Both are rank one per token and
    neither is a GEMM. With ``dnow(t) = sum_p dy(t,p) u(t,p)`` the diagonal's scalar,
    ``dures`` the increment's closing three-vector at ``L-1``, ``bnow_n = An(t) b_n``
    and ``dbnow_n = dnow crot_n + [t = L-1] dures_n``::

        dcrot_n    += dnow bnow_n
        Dr          = sum_n outer(dbnow_n, b_n)
        rotation   += Dr An(t)^T
        dB(t)      += An(t)^T dbnow_n
        dKcurr(t)  += tap_matrix_vjp(Ac(t)^T Dr, ...)

    The first line rides the existing readout cotangent, so the diagonal's ``dC`` and
    rotation halves cost no term of their own.

    ``b`` is read from global, by :func:`read_readout_b` above the head loop. The staged
    forcing tile holds one source-token run and this pass walks the chunk, so at ``L``
    above the run it is the wrong rows; accumulating the term in the run's own pass
    instead needs an ``L`` by lane-tile float32 accumulator, which is 13,312 B against a
    7,456 B gap. The read is 6,144 B per chunk and lane tile, on a launch whose limiter
    is shared capacity, so the accumulator is the expensive half of that trade and the
    read is the cheap one.

    One head writes its shard's ``dC`` row in the destination's own dtype. A fold
    above one accumulates in float32 instead, because a shard's ``dC`` is a sum over
    the heads it walks and the reference rounds the head sum once. The accumulator is
    :func:`_fold_frag`, held in registers across the fold, and the last head of the
    shard narrows it and stores it from here: the run this pass owns is the run it
    owned at every earlier head, so no other thread and no other pass can reach it and
    the store needs neither a barrier nor a pass of its own.

    The lane sum closes after the two 3x3 products and the tap adjoint, not before.
    Every term of :func:`tap_matrix_vjp` is degree one in its first argument, and the
    rest of what it and the products read -- ``stap`` rows 4 to 6, ``strans`` rows 0 to
    2, and the two transform matrices -- is indexed by ``ts`` alone, hence uniform
    across a butterfly group and commuting with the sum. ``keep`` is ``ts``-derived and
    uniform for the same reason, so a shuffle under it cannot diverge. Six outputs cost
    12 SHFL a warp where nine inputs cost 18: 30 a warp against 36. Measured at
    acceptance, SHFL 5,425,920 -> 4,872,960, exactly the -552,960 predicted, with
    ``smsp__inst_executed_pipe_lsu`` and MIO down the same count and global store
    sectors, shared wavefronts, 140 registers and 16.48% occupancy all unmoved.
    ``sm__inst_executed`` fell 1,198,080, 2.17x the prediction, because
    :func:`_sum_over_lanes` spends one FADD with every shuffle: three fewer reduced
    components delete 12 instructions a warp, not 6. Paired against the two-butterfly
    form at acceptance on a contended device, four rounds of 400 pairs read -8.704,
    -8.440, -12.288 and -8.960 us, every interval excluding zero, against same-run null
    controls of +0.752, 0, 0 and +3.584 us, none of them negative.

    The reassociation is not bitwise and what moves is bounded. ``dB``, ``dC`` and
    ``carry_b`` are bitwise: none reads ``drs``. ``dtrans[..., 3]`` is bitwise too,
    because ``dw`` reaches only rows 0 to 2 of ``sdw`` and row 3 rides ``sdls``, and so
    is the ``t-1`` tap ``dK[..., 0, :]``, which this pass does not write. ``dtrans[...,
    0:3]`` and ``dK[..., 1, 0:3]`` move, at 5.3e-07 of the field's own magnitude at
    worst over the acceptance, wide and standard shapes. Under ``--tolerance-report``
    the 30-row table is unchanged to four figures in every row but one,
    ``2x2x128/L32/P16/N16/float16`` ``dtrans`` at 1.536e-04 -> 1.537e-04 against a
    3.0e-04 bound. No bound was widened and no accumulation narrowed.

    Args:
        gdc: ``(B,G,splits*T,3N)`` ``dC`` or its shard buffer, written by the head that
            completes the sum.
        bfrag: The forcing runs :func:`read_readout_b` filled, one per step of this
            pass. Head-invariant, hence read once for the block.
        sdc: ``(mma_rows(L), pitch)`` float32 readout gradient.
        csum: The float32 readout sum over the fold, from :func:`_fold_frag`.
            Accumulated when ``fold`` is above one and untouched otherwise.
        bsum: ``(L + 1, pitch)`` float32 forcing sum, accumulated at row ``t + 1``.
        scrot: ``(mma_rows(L), pitch)`` operand-dtype rotated readout.
        stable: ``(mats, L, TABLE_PITCH)`` float32 transform table, fused.
        stap: ``(8, L)`` float32 tap parameters, component-major.
        strans: ``(4, L)`` float32 ``(w, ls)``, component-major.
        srow: ``(L, ROW_WORDS)`` float32 rotation scratch, accumulated. This pass folds
            ``srow2`` into it and adds its own term, in that order.
        srow2: ``(L, ROW_WORDS)`` float32 rotation scratch, the tap epilogue's ``t-1``
            half. Read here and not written.
        sdw: ``(4, L)`` float32 rotation-vector cotangent, accumulated. This pass folds
            rows 0 to 2 of ``sdw2`` into it and adds its own term, in that order.
        sdw2: ``(4, L)`` float32 rotation-vector cotangent, the tap epilogue's ``t-1``
            half. Rows 0 to 2 are read here; row 3 is unused.
        sdk: ``(4, L)`` float32 current-tap cotangent, accumulated.
        sdnow: ``(L,)`` float32 diagonal cotangent, one scalar a token.
        sdures: ``(1, tile)`` float32 increment's closing rank-one at token ``L-1``.
        bidx: Batch index.
        gidx: Group index.
        sbase: First row of this block's shard, ``shard * T``. Zero at one shard,
            where the destination is the output.
        t0: First token of the chunk.
        valid: Tokens of the chunk that exist.
        tid: Thread index within the block.
        hstep: Head within the shard. The last one stores the fold sum.
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

    # One adjacent column run a thread, as in :func:`_tap_epilogue`. The destination
    # run is adjacent in ``dC`` too, so the store goes out at vector width whether it
    # lands in global memory or in the fold's accumulator.
    width = 3 * (lanes // LANE_GROUP)
    fvec = _run_vec(width, 4)
    ovec = _run_vec(width, scrot.element_type.width // 8)
    gvec = _run_vec(width, out.width // 8)
    faccs = width // fvec
    oaccs = width // ovec
    gaccs = width // gvec
    grad = _runs(sdc, fvec)
    rots = _runs(scrot, ovec)
    bwords = _runs(bsum, fvec)
    dfrag = cute.make_fragment((faccs, fvec), sdc.element_type)
    cfrag = cute.make_fragment((oaccs, ovec), scrot.element_type)
    ofrag = cute.make_fragment((gaccs, gvec), out)
    bsfrag = cute.make_fragment((faccs, fvec), bsum.element_type)
    closes = hstep == fold - 1

    # One row, so the read is hoisted: every token of the pass reads the same three
    # vectors and only ``L-1`` uses them.
    ufrag = cute.make_fragment((faccs, fvec), sdures.element_type)
    _read_run(_runs(sdures, fvec), ufrag, 0, lane, faccs)
    uvals = _run_vals(ufrag, fvec, width)

    for step in cutlass.range_constexpr(-(-chunk // per_pass)):
        token = _pass_row(tid // LANE_GROUP + step * per_pass)
        ts = cutlass.min(token, chunk - 1)
        inside = token < chunk
        gsum = tuple(zero for _ in range(9))
        drs = tuple(zero for _ in range(9))
        an = _mat_at(stable, TABLE_AN, ts)
        ant = mat3_transpose(an)
        act = mat3_transpose(_mat_at(stable, TABLE_AC, ts))
        dnow = sdnow[ts]
        closing = ts == chunk - 1
        _read_run(grad, dfrag, ts, lane, faccs)
        _read_run(rots, cfrag, ts, lane, oaccs)
        dvals = _run_vals(dfrag, fvec, width)
        cvals = _run_vals(cfrag, ovec, width)
        bvals = _run_vals(bfrag, ovec, width, step * oaccs)
        dcs: list[Scalar] = []
        dbs: list[Scalar] = []
        for rep in cutlass.range_constexpr(width // 3):
            dvec = (dvals[3 * rep], dvals[3 * rep + 1], dvals[3 * rep + 2])
            crot = (cvals[3 * rep], cvals[3 * rep + 1], cvals[3 * rep + 2])
            bvec = (bvals[3 * rep], bvals[3 * rep + 1], bvals[3 * rep + 2])
            bnow = mat3_matvec(an, bvec)
            dbnow = tuple(
                dnow * crot[j] + select(closing, uvals[3 * rep + j], zero)
                for j in range(3)
            )
            dvec = (
                dvec[0] + dnow * bnow[0],
                dvec[1] + dnow * bnow[1],
                dvec[2] + dnow * bnow[2],
            )
            drs = mat3_add(drs, mat3_outer(dbnow, bvec))
            gsum = mat3_add(gsum, mat3_outer(dvec, crot))
            dcs.extend(mat3_matvec(act, dvec))
            dbs.extend(mat3_matvec(ant, dbnow))
        keep = ts < valid
        if cutlass.const_expr(not exact):
            keep = keep & inside
        if cutlass.const_expr(fold == 1):
            if keep:
                _write_run(
                    _run_at(gdc[bidx, gidx, sbase + t0 + ts, None], gvec),
                    ofrag,
                    lane,
                    gaccs,
                    tuple(dcs),
                )
        else:
            # The addends arrive in head order at every access, which is the order the
            # shared accumulator this replaced summed in, so no bit of ``dC`` moves.
            if keep:
                for k in cutlass.range_constexpr(faccs):
                    for j in cutlass.range_constexpr(fvec):
                        held = csum[step * faccs + k, j]
                        csum[step * faccs + k, j] = held + dcs[fvec * k + j]
            if keep & closes:
                _write_run(
                    _run_at(gdc[bidx, gidx, sbase + t0 + ts, None], gvec),
                    ofrag,
                    lane,
                    gaccs,
                    tuple(
                        csum[step * faccs + i // fvec, i % fvec] for i in range(width)
                    ),
                )
        if cutlass.const_expr(exact):
            _add_run(bwords, bsfrag, ts + 1, lane, faccs, tuple(dbs))
        else:
            if inside:
                _add_run(bwords, bsfrag, ts + 1, lane, faccs, tuple(dbs))
        # Both 3x3 products and the tap adjoint are linear in ``drs``, and every other
        # operand they take is indexed by ``ts`` alone and so is uniform across the
        # group. They therefore run on this lane's partial, and one butterfly reduces
        # their six outputs rather than its nine inputs: 30 SHFL a warp against 36.
        # The arithmetic count does not move. The block sat under ``lane == 0``, which
        # every warp enters, so it already issued warp-wide.
        dtap, dw = tap_matrix_vjp(
            mat3_mul(act, drs),
            (stap[4, ts], stap[5, ts], stap[6, ts]),
            (strans[0, ts], strans[1, ts], strans[2, ts]),
        )
        gsum = _sum_over_lanes(mat3_add(gsum, mat3_mul(drs, ant)))
        taps = _sum_over_lanes(dtap + dw)
        if keep & (lane == 0):
            for j in cutlass.range_constexpr(3):
                sdk[j, ts] += taps[j]
                # The tap epilogue's ``t-1`` half is folded in ahead of this pass's own
                # term, which is the order the single buffer summed in, so no bit moves
                # in the fold. Rows past ``valid`` miss it and reach no output: every
                # read of ``sdw`` past this point is under ``token < valid``.
                sdw[j, ts] = sdw[j, ts] + sdw2[j, ts] + taps[3 + j]
        # One word a lane, a group of words a round, as in :func:`_tap_epilogue`.
        for word in cutlass.range_constexpr(-(-ROW_WORDS // LANE_GROUP)):
            base = word * LANE_GROUP
            rows = lane < ROW_WORDS - base
            if cutlass.const_expr(not exact):
                rows = rows & inside
            held = _spread(gsum[base : base + LANE_GROUP], lane)
            if rows:
                srow[ts, base + lane] = (
                    srow[ts, base + lane] + srow2[ts, base + lane] + held
                )


@cute.kernel
def chunk_prefix_bwd_kernel(
    gtrans: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    seqlen: cutlass.Int32,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    splits: cutlass.Constexpr,
) -> None:
    """Scan one chunk's transition prefixes to global memory.

    One block per ``(chunk, head shard, batch, group)``, walking the ``fold`` heads
    of its shard. :func:`chunk_vector_bwd_kernel` reads the result instead of
    rescanning it, which is what this launch exists for: the scan depends on the
    head and the chunk and not on the lane tile, so the fused form ran it
    ``3N / lane_block(3N)`` times for one answer.

    The scan is a warp-serial pass over ``L / 32`` tokens a lane followed by two
    shuffle reductions, so nothing in the block hides it. Moving it here does not
    delete instructions; it deletes the repeats and the barrier that fenced them.

    Args:
        gtrans: ``(B,H,T,4)`` float32 transition parameters.
        gslp: ``(B,H,C,L)`` float32, written with the inclusive log-scale scan.
        gsquat: ``(B,H,C,4,L)`` float32, written with the inclusive quaternion
            prefix product, component-major.
        seqlen: ``T``. Dynamic.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        fold: Heads of a shard this block walks. Compile-time.
        splits: Shards the heads of a group are cut into. Compile-time, and the
            same value :func:`chunk_vector_bwd_kernel` decodes with, so the two
            launches agree on which head a shard holds.
    """
    tid, _, _ = cute.arch.thread_idx()
    xidx, bidx, gidx = cute.arch.block_idx()

    cidx = xidx // splits
    sidx = xidx - cidx * splits
    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), seqlen - t0)

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(
        cutlass.Float32, trans_tile(chunk).layout(), SMEM_SEGMENT
    )
    slp = smem.allocate_tensor(
        cutlass.Float32, scalar_tile(chunk).layout(), SMEM_SEGMENT
    )
    squat = smem.allocate_tensor(
        cutlass.Float32, trans_tile(chunk).layout(), SMEM_SEGMENT
    )

    for hstep in cutlass.range(fold, unroll=1):
        hidx = (gidx * splits + sidx) * fold + hstep
        cute.arch.sync_threads()
        stage_trans(
            gtrans[bidx, hidx, None, None], strans, t0, valid, tid, threads, chunk
        )
        cute.arch.sync_threads()
        chunk_prefixes(strans, slp, squat, tid, chunk)
        cute.arch.sync_threads()
        # Staged through shared rather than stored from the scan: the scan holds its
        # result in one warp, so a direct store would be one scalar access a token a
        # component from 32 lanes instead of one segment a thread from the block.
        _copy_words(slp, gslp[bidx, hidx, cidx, None], chunk, tid, threads)
        _copy_words(
            squat, gsquat[bidx, hidx, cidx, None, None], 4 * chunk, tid, threads
        )


@cute.jit
def chunk_prefix_bwd(
    gtrans: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    seqlen: cutlass.Int32,
    chunks: cutlass.Int32,
    bsz: cutlass.Int32,
    groups: cutlass.Int32,
    stream: Stream,
    warps: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    fold: cutlass.Constexpr,
    splits: cutlass.Constexpr,
) -> None:
    """Launch :func:`chunk_prefix_bwd_kernel`.

    The grid is :func:`chunk_vector_bwd`'s with the lane-tile factor removed, so it
    is ``dim // lane_block(dim)`` times smaller and every block of the main launch
    finds its own prefix already written.

    The budget is checked here rather than at the caller: this launch is the one place
    ``chunk`` reaches these tiles, and an unchecked launch fails at the driver instead
    of at the guard. 2,304 B at ``L 64``, flat in every other extent, so no shape the
    tree admits can reach the capacity.
    """
    threads = warps * 32
    assert_smem_fits(f"chunk_prefix_bwd[L{chunk}]", prefix_smem_bytes(chunk))
    chunk_prefix_bwd_kernel(
        gtrans, gslp, gsquat, seqlen, threads, chunk, fold, splits
    ).launch(
        grid=(chunks * splits, bsz, groups),
        block=(threads, 1, 1),
        stream=stream,
    )


@cute.kernel
def chunk_vector_bwd_kernel(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdscore: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    gdk: cute.Tensor,
    gpart: cute.Tensor,
    gcount: cute.Tensor,
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

    Two axes carry outputs a block does not own alone. ``dB``, ``dC`` and the carry are
    sums over the heads of a group, so a shard past the first writes its own slot row,
    the offset being zero at one shard where the destination is the output itself.
    ``dtrans`` and ``dK`` are sums over lanes and neither reaches a second launch. ``dK``
    takes a slot row on the same convention and the tile that arrives last adds the
    ``tiles`` rows of its own ``(batch, head, chunk)``, the row being past every map
    already. ``dtrans`` takes no row: every map between the rotation cotangent and it is
    linear, so each tile publishes :data:`PART_WORDS` words a token to ``gpart`` and
    the same tile sums them and runs the maps once.

    Both chart rows leave in one access. ``dtrans`` is four float32 a token and ``dK``
    is four a tap, and both destinations are allocated whole, so a row's offset is a
    compile-time multiple of 16 B and the run is one ``STG.E.128``. Three sites: this
    kernel's ``dK`` slot-one and ``dtrans`` stores, and :func:`_tap_epilogue`'s
    slot-zero store. Component-wise they cost one L1 sector request an element a lane,
    four where the row needs one, and the lane stride is 32 B for ``dK`` and 16 B for
    ``dtrans`` so no two lanes share a sector either.

    Measured on an A6000 at the acceptance shape, bitwise on all five outputs, base
    against arm: global store sectors 15,939,072 -> 11,294,208, -29.14%; store
    requests 1,112,832 -> 753,408, -32.30%; LSU warp instructions -359,424. L2 traffic
    does not move, read -0.53% and write +0.001%, and DRAM read and write stay within
    0.05%: the L1 sectors were never bytes. A four-byte store covering four of a
    sector's thirty-two does not make L2 read that sector -- L2 has byte enables -- so
    what the run buys is L1 sector requests and nothing downstream of them. Registers
    140, local sectors zero, occupancy 16.48% all unchanged.

    It buys no time worth claiming. Six paired runs on a contended device, three of
    them excluding zero and all six negative in position, put it at -2.048 to -3.048
    us on a 1,820 us launch, 0.1%, against a null band of the same width on identical
    code. So a store class is not priced by its sectors here: -29% of them converted at
    0.44 us a million, and the 359,424 deleted LSU warp instructions converted at 5.7
    us a million against the tree's 14.31. A global store is fire-and-forget -- no warp
    waits on the sector -- so deleting sectors on a launch that is latency-bound at 38%
    issue returns the issue slots of the deleted instructions and nothing else. Kept
    for the counters and for costing nothing, not for the microseconds.

    Args:
        gdy: ``(B,H,T,P)`` operand-dtype cotangent of ``y``.
        gu: ``(B,H,T,P)`` operand-dtype forcing input.
        guprev: ``(B,H,P)`` streaming ``u_{-1}``. Read only when ``has_prev``.
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 ``(kr, g, h, 0)`` per tap.
        gslp: ``(B,H,C,L)`` float32 inclusive log-scale scan, from
            :func:`chunk_prefix_bwd_kernel`.
        gsquat: ``(B,H,C,4,L)`` float32 inclusive quaternion prefix product, from
            the same, component-major.
        gb: ``(B,G,T,3N)`` operand-dtype forcing vectors.
        gbprev: ``(B,G,3N)`` streaming ``b_{-1}``. Read only when ``has_prev``.
        gc: ``(B,G,T,3N)`` operand-dtype readout vectors.
        gdinc: ``(B,H,C,P,3N)`` operand-dtype increment cotangent, global frame.
        gz: ``(B,H,C,P,3N)`` operand-dtype chunk-start state.
        gdlp: ``(B,H,C,L)`` float32 diagonal and increment half of the log-scale
            cotangent, from the chunk-input stage.
        gdrot: ``(B,H,C,3,3)`` float32 closing-rotation cotangent, row-major, from
            the chunk-input stage.
        gdscale: ``(B,H,C)`` float32 closing-scale cotangent, from the chunk-input
            stage.
        gdscore: ``(B,H,C,L,L)`` operand-dtype masked score, target token by source
            token, from the chunk-input stage. Read, never written. That stage holds
            both products of the pair in one fragment and applies this mask and this
            factor to the other one, so the record costs it four instructions an
            element and costs this kernel one staging pass a source block instead of
            a GEMM, an exponential and a mask per lane tile.
        gdb: ``(B,G,splits*T,3N)`` ``dB`` or its shard buffer, written, under the
            output's own dtype either way. At one shard it is the output; above one it
            is the partial :func:`vector_reduce` sums.
        gdc: ``(B,G,splits*T,3N)`` ``dC`` or its shard buffer, under the contract of
            ``gdb``.
        gcarry: ``(B,G,splits*C,3N)`` float32 carry or its shard buffer, written with
            the forcing gradient of the token before the chunk's first.
        gdtrans: ``(B,H,T,4)`` float32 ``dtrans``, written by the lane tile that
            closes the chart. Never a slot buffer: the chart's sum over lane tiles is
            closed here.
        gdtap: ``(B,H,tiles*T,2,4)`` float32 ``dK`` or its slot buffer, written by
            every lane tile and read back by the one that arrives last.
        gdk: ``(B,H,T,2,4)`` float32 ``dK``, written by the lane tile that closes the
            slots. ``gdtap`` itself at one tile, where the store above is the output's
            and this pass traces away.
        gpart: ``(B,H,C,tiles*PART_WORDS,L)`` float32 chart partials, written by every
            lane tile and read back by the one that arrives last. A rank-matched
            placeholder at one tile, where nothing crosses.
        gcount: ``(B,H,C)`` int32 arrival counters, zero at entry. A placeholder at
            one tile.
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

        Which lane tile closes the chart depends on arrival order and is not
        reproducible. What it sums is: :func:`_sum_slots` reads the slots of a word in
        ascending slot index and the affine terms enter after the sum, so the term
        order is one order for every arriver and a rerun at one shape reproduces
        ``dtrans`` bit for bit. ``dK``'s close reads ``gdtap``'s slots in the same
        ascending order, into a float32 accumulator opened at zero, so it holds too.
        ``gcount`` must be zero at entry.
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
    # The current tap of dK, accumulated rather than stored: under fusion that tap has
    # two contributors and one of them is token t+1's pass, so no single pass holds the
    # whole of it. The previous tap keeps its direct store, having one contributor.
    sdk = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(
        cutlass.Float32, offset_tile(chunk, wgroups).layout(), 16
    )
    sdls = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    # The diagonal cotangent, one scalar a token. Under fusion the diagonal is no
    # longer the s == t entry of a second tap's score, so it is contracted here and
    # read by the readout epilogue.
    sdnow = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    # The increment's closing rank-one, one three-vector a lane. Only token L-1 has
    # one, so the tile is a single row.
    sdures = smem.allocate_tensor(
        cutlass.Float32, gradient_tile(1, lane_block(dim)).layout(), 16
    )
    stable = smem.allocate_tensor(
        cutlass.Float32, quad_table_tile(chunk, 3).layout(), SMEM_SEGMENT
    )
    srow = smem.allocate_tensor(cutlass.Float32, row_tile(chunk).layout(), 16)
    # The tap epilogue's ``t-1`` deposits, split off ``srow`` and rows 0 to 2 of
    # ``sdw`` so that pass carries no barrier between its two halves. Written once a
    # row per head, not accumulated: see :func:`_tap_epilogue`. Row 3 of the second
    # is unused and costs 256 B against keeping the index expressions of ``sdw``.
    srow2 = smem.allocate_tensor(cutlass.Float32, row_tile(chunk).layout(), 16)
    sdw2 = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    scrot = smem.allocate_tensor(elem, readout_tile(chunk, tile).layout(), SMEM_SEGMENT)
    space = arena(chunk, rows, dim, fold, span, elem.width // 8)
    # One source-token block, and a readout gradient that stops below the raw
    # forcing tile: together they are what lets that tile be staged once for the
    # block instead of once a head.
    rawheld = blocks == 1 and space.raw_held
    base = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((space.words,), stride=(1,)), 16
    )
    # The arrival counter's returned value, broadcast to the block. Last so that no
    # segment-aligned tile moves, and four words rather than one so the budget counts
    # it exactly under 16-byte alignment.
    sflag = smem.allocate_tensor(
        cutlass.Int32, cute.make_layout((TABLE_QUAD,), stride=(1,)), 16
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
    # token as K. The K extent is the block rather than the block rounded up: the
    # staging never writes the pad columns, and a K mode reads them into the sum where
    # an M mode only reaches accumulator rows the store drops.
    vscorem = cute.make_tensor(
        sscore.iterator, cute.make_layout((mpad, span), stride=(lds, 1))
    )

    # These accumulators reach registers only when both loops below have a trip count
    # of one. Every accumulator allocation is hoisted to the kernel entry, and a
    # rolled loop between the allocation and its uses defeats register promotion, so
    # each fragment access becomes a local load and a local store. Measured at
    # ``P = 64``, ``L = 64``, span 64, fold one, at 255 registers and 91,344 B of
    # shared memory in both runs: one lane tile moves no local traffic at all, and
    # five lane tiles move 1,892.16 MB per call. The declaration site is not the
    # lever, and moving them inside both loops leaves the counters unchanged to the
    # sector.
    dcacc = mma_acc(tiled_mma, tid, (mpad, tile))
    ccrd = mma_coords(tiled_mma, tid, (mpad, tile))
    dbacc = mma_acc(tiled_mma, tid, (spad, tile))
    bcrd = mma_coords(tiled_mma, tid, (spad, tile))

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
    # Word base of this tile's chart slot. The slot axis sits outside the word axis so
    # a slot's words are a contiguous ``PART_WORDS * L`` record.
    prow = jstep * PART_WORDS
    gbj = _lane_view(gb, joff)
    gbprevj = _lane_view(gbprev, joff)
    gcj = _lane_view(gc, joff)
    gdincj = _lane_view(gdinc, joff)
    gzj = _lane_view(gz, joff)
    gdbj = _lane_view(gdb, joff)
    gdcj = _lane_view(gdc, joff)
    gcarryj = _lane_view(gcarry, joff)

    # No barrier ahead of the zero fill. Everything above this point is an allocation
    # or an index, so the fill is the block's first shared access and there is nothing
    # for a barrier to fence; the fill's own reader is several barriers further on.
    _fill_zero(sumb, (chunk + 1) * ldf, tid, threads)

    # The readout gradient's sum over the fold, one run a pass a thread. Zeroed here
    # and stored by the epilogue's last head, so nothing between the two touches
    # shared memory for it and the 13,312 B a shared region took is not spent.
    csum = _fold_frag(threads, chunk, lanes, fold)
    if cutlass.const_expr(fold > 1):
        csum.fill(0.0)

    # The closing state's read, held in registers across one barrier of a head's body.
    # Allocated here and not at the read because the head loop below is a dynamic loop,
    # where a fragment is an allocation an iteration.
    dincl = matrix_frag(gdincj, threads, rows, lanes)

    # The readout basis's global read, hoisted out of the head loop. Its addressing is
    # the batch, the group, the chunk and the thread, none of which the loop below
    # varies, so the shipped form issued the same three loads once a head; only the
    # transform is per head, because only the transform reads the table. The fragment is
    # live across the whole body, six registers against 41 of headroom.
    crotl, crotp = rotated_frags(gcj, threads, mpad, lanes, False, 0)
    read_rotated(
        gcj,
        gcj,
        crotl,
        crotp,
        bidx,
        gidx,
        t0,
        0,
        valid,
        tid,
        0,
        threads,
        mpad,
        lanes,
        False,
    )

    # The readout epilogue's forcing read, on the same grounds: its address is the
    # batch, the group, the chunk and the thread, so the pass read the same rows once a
    # head. It walks the chunk rather than a source-token run, which is why the term
    # cannot ride the staged tile instead.
    bfrag = readout_bfrag(gbj, threads, chunk, lanes)
    read_readout_b(gbj, bfrag, bidx, gidx, t0, valid, tid, threads, chunk, lanes)

    # The forcing tile's whole staging pass, hoisted out of the head loop. It is the
    # longest pass the kernel stages, ``(span + 1) * tile`` elements, and its address is
    # the batch, the group, the chunk, the thread and the block base; the head loop
    # varies none of them, so the shipped form restaged the same rows once a head. This
    # call is the tile's only writer, so no head below fences against it and the first
    # head's wait and barrier publish it for all of them.
    #
    # Two conditions, both from the arena. The block base has to be the pass's only
    # per-iteration term, which is one source-token block; and the epilogue's readout
    # gradient, which aliases the operand tiles from the same base, has to stop below
    # this one, which is ``raw_held``. Where either fails the pass stays inside the loop
    # with :func:`_stage_run` and the tile is restaged, because there the epilogue of
    # the head before overwrote it.
    if cutlass.const_expr(rawheld):
        stage_shifted(
            gbj,
            gbprevj,
            sb,
            bidx,
            gidx,
            t0,
            0,
            valid,
            tid,
            threads,
            span,
            tile,
            has_prev,
            True,
        )

    # The heads of one shard, rolled. Unrolling it at trace time was refused on a spill
    # -- local traffic 1,135.3 MB to 1,290.4 MB a launch at fold 18, a call 12,260.9 us
    # to 13,596.7 at fold 2 and 11,530.8 to 12,297.2 at fold 3 -- and that spill no
    # longer reproduces at any depth, so what refuses it now is the source: the body is
    # four hundred lines and the loop form cannot be selected at trace time from one
    # statement, so an unrolled arm means it written twice. At one head in the block,
    # which is what ``standard`` runs and what the full depth gives, the unrolled form
    # has no body to duplicate and drops the loop instead, worth 562 us a call.
    for hstep in cutlass.range(fold, unroll=1):
        hidx = (gidx * splits + sidx) * fold + hstep
        # The per-token tail's only global load, issued a whole head iteration above
        # its consumer. That pass admits ``chunk`` of ``threads`` threads and holds no
        # other long-latency operand, so left in place the round trip is uncovered and
        # is most of what the pass costs; here every stage below stands in front of it.
        # An out-of-range step reads ``last`` and discards it, which keeps the access
        # in bounds without a predicate.
        glp = tuple(
            gdlp[bidx, hidx, cidx, cutlass.min(tid + s * threads, last)]
            for s in range(-(-chunk // threads))
        )
        # Loop-carried only. This is the write-after-read fence between the previous
        # head's reads of the per-head tiles and this head's staging of them, so a
        # shard of one head has nothing for it to order: the zero fill above targets
        # the forcing sum, which no tile staged below overlaps, and the barrier under
        # the staging publishes it.
        if cutlass.const_expr(fold > 1):
            cute.arch.sync_threads()
        # Issued here and waited three barriers down. Neither pass reads the transform
        # table, so neither has to follow the build, and both go global to shared with
        # no register in between: the barrier above is the write-after-read fence their
        # destinations need, nothing between here and the wait reads either tile, and
        # the build plus the two barriers under it are the cover their runs get.
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
            True,
        )
        stage_state(
            gzj[bidx, hidx, cidx, None, None], sstate, tid, threads, rows, tile, True
        )
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
        # Read, not rescanned. The scan is warp-serial and depends on the head and
        # the chunk alone, so the fused form ran it once a lane tile for one answer
        # and fenced it with a barrier of its own.
        # :func:`chunk_prefix_bwd_kernel` runs it once and this loads 5 * L words
        # alongside the staging loads already in flight, which closes the barrier.
        _copy_words(gslp[bidx, hidx, cidx, None], slp, chunk, tid, threads)
        _copy_words(
            gsquat[bidx, hidx, cidx, None, None], squat, 4 * chunk, tid, threads
        )
        _fill_zero(srow, chunk * ROW_WORDS, tid, threads)
        # Not a fill. Every other row of the split pair is stored through exactly once
        # a head, so only the last token's -- which no ``t-1`` deposit reaches -- needs
        # a value, and one thread writes the twelve words. A full fill of the two would
        # be 144 predicated segment stores a block, 0.56 a warp, on a launch whose
        # class is warp instructions.
        if tid == 0:
            for k in cutlass.range_constexpr(ROW_WORDS):
                srow2[chunk - 1, k] = zero
            for j in cutlass.range_constexpr(3):
                sdw2[j, chunk - 1] = zero
        _fill_zero(sdlp, wgroups * chunk, tid, threads)
        _fill_zero(sdw, 4 * chunk, tid, threads)
        _fill_zero(sdk, 4 * chunk, tid, threads)
        cute.arch.sync_threads()
        # At the pitch the tile was allocated at, not the signature's default. The
        # table is :func:`quad_table_tile`, so a slot's row is a whole segment and
        # the build writes three of them where the default wrote nine scalars; the
        # default also leaves the padding word of each row unwritten, and
        # :func:`_mat_at` reads the whole segment.
        build_table(
            strans, stap, squat, stable, tid, threads, chunk, 3, True, TABLE_PITCH
        )
        cute.arch.sync_threads()

        # The closing transition, read once per head. Ac is R(Q)^T, so it is the
        # frame change the increment cotangent needs. Its own two cotangents are
        # read where the chart closes instead of here: nine live floats across the
        # source-token loop is nine the accumulators do not get.
        aclast = _mat_at(stable, TABLE_AC, last)
        lplast = slp[last]

        # The readout basis is the M mode of two GEMMs and the K mode of a third, so it
        # is staged once; ``slp`` is passed as its scale tile and left unread, the
        # per-token exponential belonging to the offset term alone. The transform reads
        # the table, so this half cannot precede the build; the read it consumes has no
        # such order to keep and sits above the head loop.
        apply_rotated(
            crotl,
            crotp,
            scrot,
            stable,
            slp,
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
        # One wait for all three groups. The two hoisted passes have had the table build
        # and its barriers under them, so their runs are in flight or retired by here;
        # this one has only its own issues, and the wait is what makes every run this
        # thread's.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # The first source-token block's per-head tiles, issued here and waited two
        # barriers down. They were issued at the top of the block loop, where their runs
        # had only their own issues to cover them: the wait there was 8.62% of the
        # kernel's samples for 1.55% of its instructions, the widest gap between work
        # and wall any region of this kernel holds. What covers them here is the offset
        # term's GEMM and butterfly and the closing transition's staging, and the
        # barrier above is the write-after-read fence their destinations need.
        # ``sscore``'s destination aliases the forcing gradient, whose last read is the
        # previous head's epilogue; that barrier is the one under the head's own
        # staging. The forcing tile is not among them, being staged once for the block
        # above the loop.
        if cutlass.const_expr(blocks == 1):
            _stage_run(
                gbj,
                gbprevj,
                gu,
                guprev,
                gdscore[bidx, hidx, cidx, None, None],
                sb,
                su,
                sscore,
                bidx,
                gidx,
                hidx,
                t0,
                valid,
                tid,
                0,
                threads,
                chunk,
                rows,
                tile,
                span,
                spad,
                mpad,
                has_prev,
                not rawheld,
            )

        # The closing state's global read, issued a GEMM and a barrier above the
        # transform that consumes it. The pass is six 32-bit loads a thread and its own
        # three-access step is all that stood in front of the round trip: at the
        # acceptance shape the first consumer, the widen of the first load, carried
        # 6,342 of the kernel's 163,462 long-scoreboard samples on one instruction. The
        # reads are global and land in registers, so the barrier below orders nothing
        # they touch.
        read_matrix(gdincj, dincl, bidx, hidx, cidx, tid, threads, rows, lanes)

        # The offset term, and the log-scale cotangent it carries. The scale is
        # per target token, so it rides the accumulator's M mode and is applied
        # after the reduction that needs the unscaled value.
        dcacc.fill(0.0)
        mma_gemm(tiled_mma, tid, dcacc, vdy, vstate, True, False)
        # One butterfly and one store per row, not one per element. The quad sum is
        # linear, so summing a row's columns in a register first and crossing the quad
        # once is the same terms: the group is one accumulator row, and
        # :func:`_sum_over_n` is entered ``len(crows)`` times a head where the
        # per-element form entered it once an element. At ``(mpad, tile)`` over eight
        # warps the fragment is two rows of six columns, so 4 ``SHFL`` a warp against
        # 24. The scale is uniform over the row and leaves with the sum for the same
        # reason.
        #
        # Measured at ``acceptance``: ``SHFL`` 7,257,600 -> 5,414,400 launch-wide
        # (-20 a warp, to the instruction), ``FADD`` -21 a warp, MIO and LSU each
        # -2,396,160, 31.2 to 32.3 us shorter over three paired runs of 240 to 480
        # order-swapped pairs. ``SHF`` does not move: the reach is an immediate here,
        # so the quad butterfly carries no ALU mask. Only the reduction order of
        # ``dtrans[..., 3]`` changes, by 9.2e-9 relative, and no tolerance moved.
        for group in crows:
            m, _ = ccrd[group[0]]
            expl = decay(slp[cutlass.min(m, last)])
            part = zero
            for i in group:
                _, d = ccrd[i]
                part = part + dcacc[i] * widen(scrot[m, d], elem)
                dcacc[i] = dcacc[i] * expl
            term = 2.0 * expl * _sum_over_n(part)
            if tid % 4 == 0 and m < chunk:
                sdlp[wgroup, m] = sdlp[wgroup, m] + term

        cute.arch.sync_threads()
        apply_matrix(dincl, sstate, sstate, aclast, tid, threads, rows, lanes, False)
        cute.arch.sync_threads()

        # The forcing product's right operand, loaded once a head. The rotated readout
        # tile is written once above this point and read by that product at every tap
        # of every source-token block, and nothing in the loop below writes it. Held
        # rather than reloaded: the barriers inside the loop fence shared memory, so
        # the compiler cannot prove the second load of an unwritten tile redundant, and
        # the invariant that it is redundant is not expressible to it. Bit-identical by
        # construction, the held fragment being the bits the reload would have
        # returned. The tile carries the lane extent, so a tiling of the lane block
        # cannot hoist the load further out than this.
        #
        # The increment cotangent is the other operand of this shape and is *not* held,
        # measured rather than reasoned. Both hold the same 1,013,760 LSU warp
        # instructions off the launch, and the difference is what the allocator charges
        # for the live range: 145 registers a thread for this one against 180 for that
        # one and 190 for the pair, on a 120-register base, at the acceptance shape and
        # eight warps. Only this one keeps ``sm__inst_executed.sum`` falling with the
        # LSU count, -658,944 against +1,294,848 for the increment tile and +2,953,728
        # for the pair. The pressure at the use converts a deleted shared load into
        # more than two moves. A held operand is not free, and the price is the range
        # and not the bytes.
        fb_crot = _hold_b(tiled_mma, tid, vcrot, False)

        for nstep in cutlass.range_constexpr(blocks):
            nbase = nstep * span
            # Only the blocks that have a predecessor to fence against. The first one is
            # hoisted above the offset term, where there is cover for its runs; the
            # fence a later one needs is the barrier that closed the previous iteration,
            # which is here, so a later one cannot be hoisted with it. The masked score
            # lands in a tile that aliases the forcing gradient, whose last read is that
            # same epilogue.
            if cutlass.const_expr(blocks > 1):
                _stage_run(
                    gbj,
                    gbprevj,
                    gu,
                    guprev,
                    gdscore[bidx, hidx, cidx, None, None],
                    sb,
                    su,
                    sscore,
                    bidx,
                    gidx,
                    hidx,
                    t0,
                    valid,
                    tid,
                    nbase,
                    threads,
                    chunk,
                    rows,
                    tile,
                    span,
                    spad,
                    mpad,
                    has_prev,
                    True,
                )
            # The staged tiles are published here. One forcing column, so the rotation
            # and the three products below run once a block where the two-tap form ran
            # each of them twice. The wait retires the group for this thread before the
            # barrier retires it for the block, and is a no-op where every pass took the
            # register form, no group having been committed.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # The diagonal cotangent, one scalar a token. In the two-tap form the
            # diagonal was the ``s == t`` entry of the current tap's score, where the
            # mask is one and the weight is one; the fused column's source token is
            # always the row before its own, so the entry has no home there and the
            # contraction runs here. Row ``r + 1`` of the shifted forcing input is
            # token ``nbase + r``, and a pad token's row is zero at both ends.
            # Predicated, not clamped. The clamp idiom the staging passes use wastes
            # at most one partial pass, but here the destination is one token and the
            # block has ``threads``, so at ``span 64`` on 256 threads six warps of
            # eight ran a ``rows``-deep contraction of the clamped row and stored
            # nothing. Testing the destination first deletes that, bitwise: under the
            # predicate the clamp is the identity.
            #
            # :data:`DIAG_SPLIT` threads a token and :data:`DIAG_CHAINS` chains in
            # each, not one thread and one chain: the depth falls by their product and
            # the block's idle warps take a quarter of the row each. Both constants
            # carry the arithmetic that set them.
            seg = rows // (DIAG_SPLIT * DIAG_CHAINS)
            for step in cutlass.range_constexpr(-(-span * DIAG_SPLIT // threads)):
                i = tid + step * threads
                r = i // DIAG_SPLIT
                c = i - r * DIAG_SPLIT
                if r < span:
                    parts = [zero] * DIAG_CHAINS
                    for k in cutlass.range_constexpr(DIAG_CHAINS):
                        q0 = (c * DIAG_CHAINS + k) * seg
                        for p in cutlass.range_constexpr(seg):
                            parts[k] = parts[k] + widen(
                                sdy[nbase + r, q0 + p], elem
                            ) * widen(su[r + 1, q0 + p], elem)
                    dnow = parts[0]
                    for k in cutlass.range_constexpr(DIAG_CHAINS - 1):
                        dnow = dnow + parts[k + 1]
                    # Ahead of the store's predicate, not under it: the butterfly's
                    # partner is the lane the predicate drops, and a shuffle from an
                    # inactive lane has no value. The whole group shares ``r``, so the
                    # predicate above it is uniform across the group and every partner
                    # is live here.
                    dnow = _sum_over_split(dnow, DIAG_SPLIT)
                    if c == 0:
                        sdnow[nbase + r] = dnow

            # The increment's closing rank-one, token L-1's alone. In the two-tap form
            # it was the last row of the current tap's increment product, whose weight
            # is one there. Under fusion a ragged chunk carries it in the padded slot
            # ``valid``, where ``Afuse`` is the previous token's current tap and the
            # weight is again one, and a full chunk has no such slot, so this covers
            # ``L - 1`` for both: at a ragged length that row of ``U`` is zero and the
            # term is the padded slot's.
            if cutlass.const_expr(nstep == blocks - 1):
                # Predicated for the same reason the diagonal above is: the
                # destination is one lane of ``tile`` and the block has ``threads``.
                # Split and chained for the same reason too, one lane of ``tile`` on
                # 256 threads leaving even fewer warps awake than one token does. The
                # state operand's contracted mode is its pitched one, so only the
                # forcing row's reads are adjacent; the segment map costs nothing there
                # and is what the row needs.
                for step in cutlass.range_constexpr(-(-tile * DIAG_SPLIT // threads)):
                    i = tid + step * threads
                    d = i // DIAG_SPLIT
                    c = i - d * DIAG_SPLIT
                    if d < tile:
                        parts = [zero] * DIAG_CHAINS
                        for k in cutlass.range_constexpr(DIAG_CHAINS):
                            q0 = (c * DIAG_CHAINS + k) * seg
                            for p in cutlass.range_constexpr(seg):
                                parts[k] = parts[k] + widen(
                                    sstate[q0 + p, d], elem
                                ) * widen(su[span, q0 + p], elem)
                        ures = parts[0]
                        for k in cutlass.range_constexpr(DIAG_CHAINS - 1):
                            ures = ures + parts[k + 1]
                        ures = _sum_over_split(ures, DIAG_SPLIT)
                        if c == 0:
                            sdures[0, d] = ures

            _rotate_rows(
                sb,
                sbrot,
                stable,
                tid,
                nbase,
                TABLE_AFUSE,
                0,
                threads,
                span,
                lanes,
            )
            cute.arch.sync_threads()

            # One view of the staged run: the fused column's forcing input is the row
            # before its own token, which is row r.
            vum = cute.make_tensor(
                su.iterator,
                cute.make_layout((spad, rows), stride=(ldu, 1)),
            )

            # The increment term opens the forcing accumulator, because its
            # weight is per source token and the score term's is not.
            dbacc.fill(0.0)
            mma_gemm(tiled_mma, tid, dbacc, vum, vstate, True, False)
            for i in cutlass.range_constexpr(cute.size(dbacc)):
                r, _ = bcrd[i]
                src = nbase + cutlass.min(r, span - 1)
                dbacc[i] = dbacc[i] * decay(lplast - slp[src])
            # No barrier here. The one this pass had published the score's transpose
            # store; the staged record is published by the barrier at the top of the
            # iteration and nothing between them writes a tile either GEMM below
            # reads.
            _gemm_bheld(tiled_mma, tid, dbacc, vscore, False, fb_crot)
            # Both terms take the score out of shared memory, at either warp-group
            # count: the fragment the register form reread is not produced here any
            # more. One more pass over the tile the staging published, and the K order
            # is the tile's own either way.
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
                sb,
                sumb,
                stable,
                stap,
                strans,
                srow,
                srow2,
                sdw,
                sdw2,
                sdk,
                bidx,
                hidx,
                t0,
                nbase,
                valid,
                tid,
                jbase,
                threads,
                span,
                lanes,
            )
            cute.arch.sync_threads()

        # The readout accumulator is final. It goes to shared memory because its
        # three columns per token are held by two threads, and the transform and
        # the outer product below need all three in one.
        # One store per element and not one per adjacent-column pair: ptxas already
        # merges the pair. Measured on the SASS source page at the acceptance shape,
        # base against a paired-store arm, both at 102 PCs of 32-bit ``STS``, 28 of
        # ``STS.64``, 17 of ``STS.128``, zero of any 16-bit form, and identical to the
        # wavefront. The arm moved no counter at all.
        for i in cutlass.range_constexpr(cute.size(dcacc)):
            m, d = ccrd[i]
            sdc[m, d] = dcacc[i]
        cute.arch.sync_threads()
        _readout_epilogue(
            gdcj,
            bfrag,
            sdc,
            csum,
            sumb,
            scrot,
            stable,
            stap,
            strans,
            srow,
            srow2,
            sdw,
            sdw2,
            sdk,
            sdnow,
            sdures,
            bidx,
            gidx,
            sbase,
            t0,
            valid,
            tid,
            hstep,
            threads,
            chunk,
            lanes,
            fold,
        )
        cute.arch.sync_threads()

        # This head and chunk's chart slots, every tile's. A layout, so the slice is
        # free and the placeholder at one tile is indexed in range.
        vpart = gpart[bidx, hidx, cidx, None, None]
        # The two chart rows' run fragments, hoisted out of the passes below.
        tapfrag = cute.make_fragment((1, 4), gdtap.element_type)
        trfrag = cute.make_fragment((1, 4), gdtrans.element_type)

        # The rotation cotangent is complete, so the transition chart closes: one
        # 3x3 product per token and the prefix adjoint it feeds.
        #
        # Everything from the product to ``dtrans`` is linear in that cotangent, so the
        # sum over lane tiles commutes with all of it. A tiled state width runs the
        # product and the rotation's own adjoint here, publishes the :data:`PART_WORDS`
        # words that remain, and the tile that arrives last runs the rest once on the
        # sum: two warp-serial scans, the exponential's adjoint and the store are paid
        # by one tile of ``tiles`` instead of every one, and the slot closure launch
        # goes with them. Nine words cross the sum where four crossed it before, which
        # is the price of moving the maps rather than the sum.
        #
        # The current tap's ``dK`` leaves shared memory here rather than from an
        # epilogue: under fusion its cotangent has a contributor from the next token's
        # forcing pass and one from the readout pass, so no single pass holds it whole.
        # It keeps the slot route the epilogue store used, so the lane sum is
        # unchanged.
        for step in cutlass.range_constexpr(-(-chunk // threads)):
            token = tid + step * threads
            if token < chunk:
                if token < valid:
                    krow = jbase + t0 + token
                    # Lane 3 of K is a hard zero in the forward, so it is one here.
                    _write_run(
                        _run_at(gdtap[bidx, hidx, krow, 1, None], 4),
                        tapfrag,
                        0,
                        1,
                        (sdk[0, token], sdk[1, token], sdk[2, token], zero),
                    )
                gsum = tuple(srow[token, k] for k in range(ROW_WORDS))
                dac = mat3_mul(gsum, _mat_at(stable, TABLE_AC, token))
                dquat = rot_hom_vjp(
                    tuple(dac[3 * (k % 3) + k // 3] for k in range(9)),
                    (
                        squat[0, token],
                        squat[1, token],
                        squat[2, token],
                        squat[3, token],
                    ),
                )
                # One thread per token, so the rows the tiling gave the offset term
                # fold in here rather than needing a pass and a barrier of their own.
                offset = vdlp[token]
                for g in cutlass.range_constexpr(wgroups - 1):
                    offset = offset + sdlp[g + 1, token]
                if cutlass.const_expr(tiles > 1):
                    for j in cutlass.range_constexpr(4):
                        vpart[prow + j, token] = dquat[j]
                    vpart[prow + 4, token] = offset
                    for j in cutlass.range_constexpr(4):
                        vpart[prow + 5 + j, token] = sdw[j, token]
                else:
                    for j in cutlass.range_constexpr(4):
                        sdrot[j, token] = dquat[j]
                    vdlp[token] = offset
        cute.arch.sync_threads()

        # The barrier above orders the block's publishing stores at block scope, which
        # puts them in the increment's happens-before set; the increment's own
        # ``acq_rel`` at gpu scope carries them to device scope. So a tile that reads
        # ``tiles - 1`` reads every slot's words, with no separate fence: a fence here
        # lowers to the same MEMBAR.ALL.GPU, ERRBAR and CCTL.IVALL the atomic already
        # lowers to, with no memory instruction between the two triples, so it is the
        # same fence twice. The acquire half needs no cache reasoning either, since
        # :func:`_sum_slots` reads the slots with ``ld.global.cg``, past L1.
        # One thread increments and the second barrier broadcasts what it read: the
        # branch below is block-uniform, which is what makes the barriers inside it
        # legal.
        run = True
        if cutlass.const_expr(tiles > 1):
            if tid == 0:
                sflag[0] = cute.arch.atomic_add(
                    gcount.iterator + gcount.layout((bidx, hidx, cidx)),
                    cutlass.Int32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
            cute.arch.sync_threads()
            run = sflag[0] == tiles - 1
        if run:
            # ``dK``'s slot rows close here and not in a launch of their own. They are
            # past every map, so this is a float32 add of ``tiles`` rows in slot order --
            # the order the reduction it replaces summed in, and the same order for
            # every arriver, so it is bitwise what that launch produced. It needs no
            # shared memory, no map and no zeroed accumulator, and the barrier, the
            # fence and the arrival branch above are the ones ``dtrans`` already pays.
            #
            # A record is ``L`` tokens of eight adjacent float32 and the block owns two
            # words a thread, so a warp covers 128 B of a slot in one access. The two
            # stores are the reduction's own.
            if cutlass.const_expr(tiles > 1):
                vslot = gdtap[bidx, hidx, None, None, None]
                kwords = chunk * 8
                for step in cutlass.range_constexpr(-(-kwords // threads)):
                    i = tid + step * threads
                    # One guard, not two: past ``kwords`` the token index runs past
                    # ``chunk`` and therefore past ``valid``, which is at most it.
                    if i // 8 < valid:
                        krow = t0 + i // 8
                        ksum = zero
                        for slot in range(tiles):
                            ksum = ksum + _load_cg(
                                vslot.iterator
                                + vslot.layout(
                                    (slot * seqlen + krow, (i % 8) // 4, i % 4)
                                )
                            )
                        gdk[bidx, hidx, krow, (i % 8) // 4, i % 4] = ksum
            # The transition's own two cotangents and the chunk-input stage's half of
            # the log-scale cotangent are affine in the sum, so they enter after it and
            # are read by this tile alone. The rotation term passes through the same
            # adjoint the tiles' terms did, which is what lets it be added after:
            # ``rot_hom_vjp(dac + dclose, q) == rot_hom_vjp(dac, q)
            # + rot_hom_vjp(dclose, q)``.
            dclose = tuple(gdrot[bidx, hidx, cidx, i // 3, i % 3] for i in range(9))
            dclast = gdscale[bidx, hidx, cidx]
            cscale = decay(slp[last])
            for step in cutlass.range_constexpr(-(-chunk // threads)):
                token = tid + step * threads
                if token < chunk:
                    closing = token == last
                    dq = rot_hom_vjp(
                        dclose,
                        (
                            squat[0, token],
                            squat[1, token],
                            squat[2, token],
                            squat[3, token],
                        ),
                    )
                    # A fresh name: a tuple bound inside a dynamic ``if`` under a name
                    # the kernel already holds a scalar under is a structure change to
                    # the DSL, and the trace fails.
                    if cutlass.const_expr(tiles > 1):
                        psum = tuple(
                            _sum_slots(vpart, word, token, tiles)
                            for word in range(PART_WORDS)
                        )
                    else:
                        psum = (*(sdrot[j, token] for j in range(4)), vdlp[token])
                    for j in cutlass.range_constexpr(4):
                        sdrot[j, token] = psum[j] + select(closing, dq[j], zero)
                    vdlp[token] = (
                        psum[4]
                        + glp[step]
                        + select(closing, 2.0 * cscale * dclast, zero)
                    )
                    if cutlass.const_expr(tiles > 1):
                        for j in cutlass.range_constexpr(4):
                            sdw[j, token] = psum[5 + j]
            cute.arch.sync_threads()
            # Two independent warp-serial scans, one warp each rather than warp 0
            # twice. ``tid ^ 32`` is the whole change: it maps warp 1 onto lanes 0-31
            # and every other warp above the guard, since flipping bit 5 of a thread
            # past 63 leaves a higher bit set. The shuffles inside are warp-relative,
            # so neither scan's arithmetic or order moves.
            chunk_suffix(vdlp, sdls, tid, chunk)
            quat_suffix_vjp(squat, sdrot, sdquat, tid ^ 32, chunk)
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
                        trow = t0 + token
                        # The fused column's own log-scale term is not reverse-scanned:
                        # ``e_t`` multiplies one slot of one token, where the offset
                        # term rides the prefix every later token carries.
                        _write_run(
                            _run_at(gdtrans[bidx, hidx, trow, None], 4),
                            trfrag,
                            0,
                            1,
                            (
                                sdw[0, token] + dexp[0],
                                sdw[1, token] + dexp[1],
                                sdw[2, token] + dexp[2],
                                sdls[token] + sdw[3, token],
                            ),
                        )

    # No barrier between the fold and the two sums below. The last write to either sum
    # is a tap epilogue's and a readout epilogue's, and every path from there to here
    # crosses the barriers that close the chart, so a barrier at this point orders a
    # pair the ones inside the fold have already ordered.
    #
    # The shard's forcing sum for this lane tile, rounded once: the narrowing is here at
    # one shard and at the reduction above one, never twice. Row t+1 of the forcing sum
    # is token t and row 0 is the row the boundary kernel owns. The readout sum has no
    # pass here at any fold: at one head the epilogue stores it and above one the
    # epilogue's last head does, from the register fragment that carries it.
    #
    # One access per run of adjacent lanes and not one per lane. The scalar form
    # indexed a four-mode global tensor once an element, so every element paid the
    # whole coordinate: three stride products, the widen and the carry. A run pays
    # the coordinate once and covers a 16-byte segment, which is the form
    # :func:`_readout_epilogue` already stores through.
    fvec = _run_vec(tile, 4)
    ovec = _run_vec(tile, out.width // 8)
    oaccs = ovec // fvec
    bwords = _runs(sumb, fvec)
    ofrag = cute.make_fragment((ovec,), out)
    sfrag = cute.make_fragment((oaccs, fvec), cutlass.Float32)
    runs = tile // ovec
    total = chunk * runs
    for step in cutlass.range_constexpr(-(-total // threads)):
        i = tid + step * threads
        if i < total:
            t = i // runs
            r = i - t * runs
            if t < valid:
                _store_run(
                    _run_at(gdbj[bidx, gidx, sbase + t0 + t, None], ovec),
                    bwords,
                    ofrag,
                    sfrag,
                    t + 1,
                    r,
                    oaccs,
                    fvec,
                )
    # The carry is float32 at both ends, so its run is the float32 one and it needs
    # no narrowing fragment.
    kruns = tile // fvec
    kfrag = cute.make_fragment((fvec,), cutlass.Float32)
    for step in cutlass.range_constexpr(-(-kruns // threads)):
        r = tid + step * threads
        if r < kruns:
            cute.autovec_copy(bwords[(None, (0, r))], kfrag)
            cute.autovec_copy(
                kfrag,
                _run_at(gcarryj[bidx, gidx, cbase + cidx, None], fvec)[(None, r)],
            )


@cute.jit
def chunk_vector_bwd(
    gdy: cute.Tensor,
    gu: cute.Tensor,
    guprev: cute.Tensor,
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gslp: cute.Tensor,
    gsquat: cute.Tensor,
    gb: cute.Tensor,
    gbprev: cute.Tensor,
    gc: cute.Tensor,
    gdinc: cute.Tensor,
    gz: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    gdscale: cute.Tensor,
    gdscore: cute.Tensor,
    gdb: cute.Tensor,
    gdc: cute.Tensor,
    gcarry: cute.Tensor,
    gdtrans: cute.Tensor,
    gdtap: cute.Tensor,
    gdk: cute.Tensor,
    gpart: cute.Tensor,
    gcount: cute.Tensor,
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
        gslp,
        gsquat,
        gb,
        gbprev,
        gc,
        gdinc,
        gz,
        gdlp,
        gdrot,
        gdscale,
        gdscore,
        gdb,
        gdc,
        gcarry,
        gdtrans,
        gdtap,
        gdk,
        gpart,
        gcount,
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
    producer's device body, not here. It is what closes the transition chart, where the
    counter costs two barriers and the sum reads back eight words a token; here it
    would close a destination at the activation width, which is the shadow buffer
    above, so the two are not the same arm.

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
    dscore: Tensor,
    u_prev: Tensor | None = None,
    b_prev: Tensor | None = None,
    dB: Tensor | None = None,
    dC: Tensor | None = None,
    splits: int | None = None,
    warps: int = WARPS_WIDE,
    prefix_lp: Tensor | None = None,
    prefix_q: Tensor | None = None,
    arrived: Tensor | None = None,
) -> ChunkVectorBwd:
    """Differentiate the rowwise vectors and the transition parameters.

    What this takes from the chunk-input stage is consumed, never recomputed:
    ``dlogp`` is that stage's half of the log-scale cotangent, the closing rotation
    and scale are one contraction over the chunk-start state that stage already ran,
    and ``dscore`` is the masked score it forms anyway.

    Two workspaces, both allocated here and freed on return, and both holding one
    partial row per block of a sum whose terms separate blocks cannot share. Above one
    lane tile ``dK`` is a sum over lanes: ``(B, H, tiles * T, 2, 4)`` float32 with its
    output, closed inside the main launch by the tile that arrives last. Above partial
    depth one the two vectors and the carry are sums over heads:
    ``(B, G, splits * T, 3N)`` twice at the activation width and
    ``(B, G, splits * C, 3N)`` at float32, which is :func:`partial_bytes`, closed by
    :func:`vector_reduce`. Each partial carries its own output's width, that output's
    store being one more rounding on the same path. At one copy of either there is no
    buffer and the kernel stores the output directly.

    A third workspace, unconditional: ``(B,H,C,L)`` and ``(B,H,C,4,L)`` float32 hold
    the two chunk-local transition prefixes, which :func:`chunk_prefix_bwd` scans once
    a ``(batch, head, chunk)`` before the main launch reads them. The main grid carries
    a lane-tile axis the scan does not depend on, so the fused form ran the scan
    ``3N / lane_block(3N)`` times for one answer.

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
        dinc: ``(B,H,C,P,3N)`` increment cotangent in the global frame, the dtype of
            ``dy``, contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward`.
        zstart: ``(B,H,C,P,3N)`` chunk-start state, the dtype of ``dy``, contiguous,
            held from the forward, or rebuilt when the boundary did not cross.
        dlogp: ``(B,H,C,L)`` float32, contiguous, from
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_input.chunk_input_backward`.
        dchunk_rot: ``(B,H,C,3,3)`` float32, contiguous, from the same.
        dchunk_scale: ``(B,H,C)`` float32, contiguous, from the same.
        chunk_size: ``L``. A multiple of 16.
        dscore: ``(B,H,C,L,L)`` masked score, target token by source token, the dtype
            of ``dy``, contiguous, from the same. Read, never written. That stage
            forms this product to mask and scale the other half of its own pair, so
            the record costs it four instructions an element; forming it here costs a
            GEMM, an exponential an element and a transpose store per lane tile, for
            a value no lane tile depends on.
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
        prefix_lp: ``(B,H,C,L)`` float32 inclusive log-scale scan, from
            :func:`slinoss.ops.so3ssd.cute.bwd.start_passing.chunk_prefix_backward`,
            or None to run that pass here. Supplied by a caller whose earlier launch
            reads it too, so the scan runs once for both.
        prefix_q: ``(B,H,C,4,L)`` float32 inclusive quaternion prefix product,
            component-major, under the contract of ``prefix_lp``. Both or neither.
        arrived: ``(B,H,C)`` int32 zeros, contiguous, the arrival counter the lane
            sums close on, or None to allocate and zero it here. A caller whose
            earlier launch already wrote the zeros hands them over and the fill costs
            no launch; the tensor is read and incremented, never read after, so what
            it holds on return is arrival order.

    Returns:
        :class:`ChunkVectorBwd`.

    Raises:
        ValueError: On a layout, rank, shape or extent violation, on a destination
            that is not the band of its operand, on a shared-memory budget the
            device cannot hold, on half a streaming pair, on a ``splits`` that
            does not divide the fold, on a ``warps`` that is not a legal block
            width, on a float32-pinned operand that is not float32, or on a stored
            state that is not at the activation dtype.
        TypeError: On an activation dtype with no tensor-core path.
    """
    activations: Named = ((dy, "dy"), (U, "U"), (B, "B"), (C, "C"))
    pinned: Named = (
        (trans, "trans"),
        (K, "K"),
        (dlogp, "dlogp"),
        (dchunk_rot, "dchunk_rot"),
        (dchunk_scale, "dchunk_scale"),
    )
    stored: Named = ((dinc, "dinc"), (zstart, "zstart"), (dscore, "dscore"))
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
    shards = vector_splits(heads // groups, splits)
    fold = heads // groups // shards
    # The N atoms of the tiling, which is the warp-group count the offset term takes a
    # row of. Raises on an illegal width, so the block geometry is checked here rather
    # than inside the trace. Ahead of the span, which prices the same rows.
    wgroups = mma_atoms(warps)[1]
    span = vblock(chunk_size, rows, dim, fold, dy.element_size(), wgroups)
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
    record = (bsz, heads, chunks, chunk_size, chunk_size)
    if tuple(dscore.shape) != record:
        raise ValueError(f"dscore must be {record}, got {tuple(dscore.shape)}")

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
    # The one output that is a sum over lanes, and the three that are sums over the
    # heads of a group. Nothing else needs a partial row: every other output spans the
    # state width and belongs to one head, so one block owns it. ``dK``'s rows are read
    # back and closed by the tile that arrives last, so the buffer is freed on return
    # and no launch of its own reads it.
    tiles = dim // lane_block(dim)
    held = open_slots(dK, tiles, axis=-3)
    shared = tuple(open_slots(out, shards) for out in (dB, dC, carry_b))
    # The chart partials the lane tiles publish and the arrival counter that closes both
    # lane sums: :data:`PART_WORDS` words a token a tile, 23.59 MB at the acceptance
    # shape, freed on return with the prefixes. ``dtrans`` takes no slot buffer at all.
    #
    # The counter is zeroed rather than counted by generation: a reset bug in a monotone
    # counter is invisible, where a fill is not. Its own launch is 9.2 kB and 1.86 us, so
    # a caller whose earlier launch holds one block per element writes the zeros there
    # and supplies it. At one tile both are placeholders of the same rank and dtype, so
    # the launch key is the same and the leading modes still index in range.
    chart = torch.empty(
        bsz,
        heads,
        chunks,
        tiles * PART_WORDS if tiles > 1 else 1,
        chunk_size if tiles > 1 else 1,
        dtype=torch.float32,
        device=device,
    )
    if arrived is None:
        arrived = torch.zeros(bsz, heads, chunks, dtype=torch.int32, device=device)
    elif (
        tuple(arrived.shape) != (bsz, heads, chunks)
        or arrived.dtype is not torch.int32
        or not arrived.is_contiguous()
        or arrived.device != device
    ):
        raise ValueError(
            f"arrived must be a contiguous int32 {(bsz, heads, chunks)} on {device}, "
            f"got {tuple(arrived.shape)} {arrived.dtype} on {arrived.device}"
        )
    # The two chunk-local transition prefixes, scanned once per (batch, head, chunk)
    # rather than once per lane tile. A caller whose earlier launch reads them hands
    # them over and the scan runs once for both; otherwise they are allocated here and
    # freed on return with the partials, 5 * L float32 a chunk, 2.95 MB at the
    # acceptance shape.
    if prefix_lp is None or prefix_q is None:
        prefix_lp = torch.empty(
            bsz, heads, chunks, chunk_size, dtype=torch.float32, device=device
        )
        prefix_q = torch.empty(
            bsz, heads, chunks, 4, chunk_size, dtype=torch.float32, device=device
        )
        jit_launch(
            chunk_prefix_bwd,
            (trans, prefix_lp, prefix_q, seqlen, chunks, bsz, groups),
            (PREFIX_WARPS, chunk_size, fold, shards),
        )
    else:
        supplied: Named = ((prefix_lp, "prefix_lp"), (prefix_q, "prefix_q"))
        check_layout(supplied)
        check_pinned(supplied)
        for tensor, name, shape in (
            (prefix_lp, "prefix_lp", (bsz, heads, chunks, chunk_size)),
            (prefix_q, "prefix_q", (bsz, heads, chunks, 4, chunk_size)),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be {shape}, got {tuple(tensor.shape)}")
    jit_launch(
        chunk_vector_bwd,
        (
            dy,
            U,
            U if u_prev is None else u_prev,
            trans,
            K,
            prefix_lp,
            prefix_q,
            B,
            B if b_prev is None else b_prev,
            C,
            dinc,
            zstart,
            dlogp,
            dchunk_rot,
            dchunk_scale,
            dscore,
            shared[0].dest,
            shared[1].dest,
            shared[2].dest,
            dtrans,
            held.dest,
            dK,
            chart,
            arrived,
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
            min(RESIDENT_MAX, smem_residency(budget)),
        ),
    )
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
