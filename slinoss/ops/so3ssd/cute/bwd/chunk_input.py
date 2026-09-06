"""``dU``, the input carry, and the log-scale and chunk-transition cotangents.

Everything the backward owes the forcing input and the closing transition of a
chunk, in one block per ``(chunk, batch, head)``. The reference terms are ``du``,
``dushift``, ``dexpo``, ``duw``, ``dupw``, ``dexpw``, ``dchunk_rot`` and
``dchunk_scale``.

Five contractions, all of them dense real GEMMs off the one atom, all indexed by
the source token first:

    D(r,d)     = sum_p  ushift(r,p)  dinc_local(p,d)
    duw(r,p)   = sum_d  bfuse(r,d)   dinc_local(p,d)
    score(r,t) = sum_d  bfuse(r,d)   crot(t,d)
    dm(r,t)    = sum_p  ushift(r,p)  dy(t,p)
    ddiag(r,p) = sum_t  Smasked(r,t) dy(t,p)

with ``ushift(r) = u_{r-1}``, ``bfuse(r) = Afuse_r b_{r-1}`` the one-tap column
``Afuse_r = Ap_r + exp(2 ls_r) An_{r-1}``, ``Smasked = score * dmask``, and
``dinc_local = Ac_{L-1} dinc`` the increment cotangent carried back into the
chunk-local frame. One tap, so five GEMMs and not ten, and ``ddiag`` and ``duw``
accumulate into the one fragment ``dushift``.

What the second tap carried is two residues, neither a GEMM:

    dnow(t)  = sum_d crot(t,d) bnow(t,d)
    dures(p) = sum_d dinc_local(p,d) bnow(L-1,d)

with ``bnow(t) = An_t b_t``. ``dnow`` is the diagonal's ``s == t`` term, whose mask
is ``dmask(t,t) = 1``; ``dures`` closes the increment, whose weight is
``wgt(L-1) = 1``. Each contracts the whole lane extent for one token or one row, so
each is a lane reduction accumulated across lane blocks rather than a GEMM. Neither
carries a log scale, which is why ``dlogp`` below has one inner product and not two.

The two forms that keep the two-tap ``dlogp`` -- hold ``score_now`` as well, or build
``dbfuse`` here -- both need ``bnow`` staged as a full ``L x 3N`` GEMM operand and
both restore the second ``B`` pass: 0.642 and 0.783 of the two-tap MAC count against
0.500 for this one. The relocated mass belongs to ``chunk_vector_bwd``'s ``dls_step``
instead, and the two kernels' forms are therefore coupled.

The score is built transposed relative to the forward's. That is what makes the
masked score the left operand of the diagonal GEMM with no shared-memory round
trip: its N mode is the target token and the diagonal GEMM contracts over the
target token, so :func:`slinoss.ops.so3ssd.cute.mma.mma_areg` rereads the fragment
in place. It removes a score tile, one ``ldmatrix`` and one barrier per slice.
``dm`` is transposed with it, because the two are multiplied elementwise.

The cost of the transpose is the direction of the ``dexpo`` reduction. Summed over
the source token it is now a reduction over the accumulator's M mode, whose rows
are disjoint across warps, so it goes through one scratch row per warp and is
summed over warps at the store. Over the N mode it would have been two shuffles
and no scratch.

Two passes over ``B``. Every contraction that reads the fused column runs while that
column is staged, which is one pass; the ``dnow`` residue reads the same lane block
back one token offset, which is the second. 19.0 MB either way at ``standard``, on a
kernel whose floor is its traffic. What fusion does remove is the second pass over
the increment cotangent and over ``C``, which the two-tap form restaged per tap
wherever the lane extent was sliced. The price of the fused column is that the
increment cotangent, the readout and the output cotangent are live at once. The block
is 43,280 B at ``standard`` and 49,264 B at every ``3N`` once :func:`lblock` slices
the lane extent, and the carveout holds either two deep; both read off the device as
``launch__shared_mem_per_block_dynamic`` on sm_86.

``dU`` is ``ddiag(t) + dushift(t+1)``, the second term one row behind, which is why
``dushift`` goes through a float32 tile before the store. The last slot has no row
above it and takes ``dures`` instead. On a ragged chunk the partner of the last valid
token is the first padded row, which the fused column fills with ``bnow`` of that
token, so the bound on the shift is the chunk and not ``valid``.
:func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` still owns the term
that crosses into the next chunk. ``carry_u`` is row 0 of ``dushift``, the cotangent
that crosses into the previous chunk, and ``Afuse_0 = Ap_0`` makes it the same
expression the two-tap form gave.

The log-scale term is
``dlogp(t) = 2 * (sum_r dexpo(r,t) - <ushift(t),dushift(t)>)``
plus ``2 * sum_r dexpw(r)`` at the chunk's last slot, which is what the reference's
``_scatter_last`` writes and holds whether or not that slot carries a token. One
inner product, not two: the current tap's own terms are the two residues and neither
depends on a log scale. The sum of ``dexpo`` over the target token is never formed:
it equals that inner product against the finished fragment, so it costs one lane
reduction rather than a second pass over the score.

The chunk-transition pair factors through the same frame change. With
``X = chunk_scale * zstart + inc_local``,

    dchunk_rot = R(Q_{L-1}) sum_{p,n} outer(dinc_local(p,n), X(p,n))

because ``dinc = R dinc_local`` and ``R`` comes out of the sum, so the increment
never has to be rematerialized: its half of the outer product is
``sum_r wgt(r) sum_n D(r,3n+i) bfuse(r,3n+j)``, read off the ``D`` fragment in
its epilogue. ``dchunk_scale`` is ``sum <dinc_local, zstart>`` for the same reason.
Both are eleven float32 block reductions and ten stores per block.

Shared memory is one resident set and one phase arena. Resident: the log-scale
prefix, the increment weight, the two residues, the per-warp log-scale scratch, the
reduction scratch, the three-slot transform table, and the shifted ``U`` tile. The
two residues are resident and not in the arena because each is written inside the
lane loop and read in the epilogue, where the arena holds the shift tile. The arena
holds the fused column, the increment cotangent, the readout and the output
cotangent, all four staged once per lane block; the ``trans``, ``K`` and quaternion
tiles of the prologue and the float32 shift tile of the epilogue alias the last
three, neither being live when the other is. The shift tile overlaps the output
cotangent by 768 float32 words at the acceptance shape, so the diagonal residue is
taken into a fragment before the shift is written rather than at the store.

DRAM-bound. Analytic traffic at ``standard`` is ``dy 9.44 + U 9.58 + trans 1.57 +
K 3.15 + B 19.02 + C 9.44 + dinc 7.08 + zstart 7.08 + dU 9.44 + carry_u 0.29 +
dlogp 0.39 + dchunk_rot 0.06 + dchunk_scale 0.01 = 76.54 MB`` against ``1536 *
1.77 MFLOP = 2.72 GFLOP``, so 35.5 flop/byte against a ridge point of 164: memory
bound by a factor of over four. The GEMM count is the fused one, exactly half the
two-tap form's ``3.54 MFLOP`` per block, and the traffic is unchanged at one lane
block, so fusion moved this shape further from its ridge, not nearer.
``dinc`` and ``zstart`` are two of the three reads
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

That table is the two-tap form. It was taken before the fusion and is not retaken:
the shapes are unchanged, but a single absolute duration on this host is a stamp of
its own launch and not a rate, so the fused form is priced by paired delta instead.

``class`` is the fitted floor over the duration, against a bar of 85%. It falls when a
change removes bytes, because the floor falls with them; the duration is the figure to
read across the rows. The last row is the acceptance shape,
``B=4 H=18 T=2048 P=64 3N=240 L=64`` at one group: 85.6% of the floor,
``dram__throughput`` 80.6%, 28.21% issue, 254 registers, 48,752 B of shared memory,
16.7% theoretical occupancy against 16.4% achieved, 0.1236 shared bank conflicts per
wavefront, and SOL 56.3% sm against 80.9% memory.

What the fusion is worth, measured as a paired delta of the one-tap launch against the
two-tap one in a single process with the launch order swapped every iteration, 240
pairs, 2026-08-22, ``dinc`` and ``zstart`` at bfloat16:

    shape       delta us  interval          null   position  smem B   local ld/st
    standard      -28.160 [-29.184,-27.136] -0.512  -16.896  49,664   0 / 0
    wide          -45.064 [-46.088,-45.048]  0.000   11.256  74,080   0 / 0
    acceptance    -86.024 [-87.032,-86.008]  0.512    9.224  49,264   3,723,264 / 921,600

``null`` is the same machinery run with the two-tap launch on both sides, which is what
establishes the channel: its band is 0.5 to 1.5 us against signals of 28 to 86, so the
durations above are read rather than inferred from counters. ``position`` is the
launch-order cost the pairing removes from every pair; at ``standard`` it is 60% of the
delta, which is why an unpaired bracket at that shape cannot resolve this arm.

``sm__inst_executed_pipe_tensor`` at the acceptance shape is 7,815,168 in the two-tap
form and 3,907,584 in the one-tap one, exactly half, which is the five-GEMM count above
counted off the device. ``smsp__sass_thread_inst_executed_op_ffma_pred_on`` falls 14.70%
with it, 832,555,008 to 710,166,528: less than half, because the two residues are FFMA
where the tap they replace was tensor.

The two log-scale reductions are entered once per destination word, not once per
accumulator element. The exponent's cotangent is summed over the source token into
``sdlp[warp, token]`` and the shift inner product over the target token into
``sdlp[warp, m]``, so the fragment elements that share a destination sum in a register
and cross the lanes once: at the acceptance shape 16 score elements enter as 8 columns
and 32 output elements as 2 rows, at ``standard`` 8 as 4 and 12 as 2. Both sums are
linear, so the collapse is exact in the reals and a reassociation in float32.

Counted off the source page at the acceptance shape, per launch: ``SHFL`` 2,142,720 to
1,147,392, ``LDS`` 9,556,992 to 9,133,056, ``STS`` 3,117,312 to 2,693,376, ``FADD``
3,674,880 to 2,255,616, and the launch 109,368,576 to 105,200,640, -3.81%. ``FFMA``
rises 589,824 and ``FMUL`` falls by the same, the register pre-sum folding a multiply
into an accumulate, so the FMA pipe count does not move. Static ``SHFL`` sites fall 215
to 107 and static ``FADD`` sites 363 to 209, which is the per-warp count: 108 shuffles
and 154 adds. At ``standard`` three of those counts close exactly against the fragment
arithmetic, ``SHFL`` -540,672, ``STS`` -221,184 and ``FADD`` -761,856, and the launch
falls 6.27%. Registers, shared bytes, both occupancy limits and achieved occupancy hold
at every shape. Spill moves in both directions and neither decides a delta: unchanged at
2,469,888 sectors loaded and 884,736 stored at acceptance, up 20% at ``standard`` and
``ragged``, down 89.5% at ``standard`` in the narrow form, zero at ``tiny`` and ``wide``
on both sides. The 3,723,264 and 921,600 below is the fusion arm's own bracket; this
toolchain reads the smaller pair on both sides of this one.

Paired against the per-element form with the launch order swapped every iteration, 1000
pairs, 2026-08-22, every run stamped with foreign processes on the device:

    shape       delta us  interval            null   position  base us   arm us
    tiny         -0.072   [-0.072,-0.072]            -0.072      12.288   12.288
    standard    -16.384   [-16.384,-16.384]  +2.024   0.000     174.080  157.696
    ragged      -16.384   [-16.384,-16.384]          +1.024     174.080  157.696
    wide        -32.256   [-32.256,-32.256]          -1.536     324.608  292.864
    acceptance  -31.232   [-31.232,-31.232]  -2.048  -6.656     622.592  589.824
    acceptance  -33.280   [-33.280,-33.280]          -6.656     647.168  611.328

The two acceptance rows are one arm measured twice, 6% apart. ``tiny`` is inside the
null band and reads as no change. Predicted -18 us at acceptance from 8.82 us per million
MIO instructions, the rate the lane-run arm on this kernel converted at, and delivered
1.8x that. Two terms were missing. The prediction counted the shuffles, the adds and the
``sdlp`` round trip and not the address arithmetic and the branch each deleted entry
carried, so the deletion is 1.47x what it priced. And the chain deleted is dependent,
``SHFL`` into ``FADD``: ``FADD``'s ``short_scoreboard`` share falls 55.45% to 41.12% and
its sample share 7.45% to 4.81%, which a flat rate over an instruction class cannot
carry.

``dlogp`` is the only output that moves and it is not bitwise, the reassociation being
inside its two sums: worst absolute disagreement against the per-element form 2.808e-3
at acceptance over 125,713 of 147,456 words. Nothing moves that a bound reads. Of the 48
rows of a ``--tolerance-report`` run against the float64 oracle, 47 are identical to the
per-element form's; ``dlogp``'s worst relative error is 2.784e-3 in bfloat16 and 2.774e-4
in float16 against bounds of 5e-3 and 5e-4, to four figures what the table at
:data:`BOUNDS` in the test file records; the row that moves is ``wide.dlogp``, 8.743e-8
to 9.772e-8 against 1e-6. ``dU``, ``carry_u``, ``dchunk_rot`` and ``dchunk_scale`` are
bit-identical at every shape. This kernel is not bit-stable across its own block widths
and was not before: at ``standard`` the per-element form disagrees with itself at 128
against 256 threads on ``dU`` by 1.0 over 266 words, on ``dchunk_rot`` by 2.441e-4 and on
``dchunk_scale`` by 3.052e-5, and the arm reproduces every one of those counts and
maxima.

The two residue reductions are the same defect at a size that does not pay.
:func:`_sum_over_lanes` is entered once per lane block where its destination needs the
whole lane extent, five blocks at the acceptance shape against one destination, so
hoisting the partial sum out of the lane loop would delete 10 ``SHFL``, 8 ``LDS`` and 8
``STS`` per warp, 239,616 instructions, about 1.8 us at this arm's own conversion and
inside the 2 us null band. It also puts ``chunk // trows`` and ``rows // trows`` float32
live across a dynamic loop on a launch already at the 255-register cap. Refused on both.

The two-tap form's own local traffic is 344,064 sectors each way at ``standard`` and
983,040 loaded against 491,520 stored at ``wide``; the fused form has none at either,
and 9.50% and 33.36% less device traffic. At the acceptance shape the direction
reverses: the two-tap form spills nothing there and the fused form spills 3,723,264
sectors loaded and 921,600 stored, 148.63 MB, while device traffic rises 0.23%, from
281.13 MB to 281.77 MB. So the spill is absorbed by L1 and L2 and does not reach DRAM,
and the launch is 11.3% shorter with it. Registers are at the 255 cap on both sides at
``wide`` and at acceptance and at 128 on both sides at ``standard``, so the spill
tracks the live range and not the cap, as below.

Both of those last two claims are wrong. The frame at the acceptance shape is 80 B, 20
dwords of which 19 are loaded, across 39 sites, 19 ``LDL`` and 20 ``STL``. The cubin has
one backward branch, the lane loop, ``0x6da0`` to ``0xd470``, 1,646 instructions over
five trips. Twelve slots, ``+0x00`` to ``+0x2c``, are stored once in the prologue and
reloaded once per trip; seven, ``+0x30`` to ``+0x48``, are stored in the prologue and
loaded once after the loop exits; ``+0x08`` alone is zeroed in the prologue and round
trips once per trip. That is 67 ``LDL`` and 24 ``STL`` a thread, and at four sectors a
word over 9,216 warps it closes on the counters exactly, 2,469,888 loaded and 884,736
stored. What the slots hold is address arithmetic: the shared store base of
:func:`stage_rotated` and four 64-bit global element-offset bases for the lane-sliced
reads. The per-trip advance is not among them, it is in the uniform datapath,
``UIADD3 UR10, UR10, 0x60``. The whole spill is 838,656 warp-instructions, 0.797% of
``sm__inst_executed``.

The live set is not what puts it there. ``min_blocks_per_mp`` emits ``nvvm.minctasm``,
and dropping that one directive, at the same 255 registers, the same 49,264 B and the
same residency two, takes the frame to 16 B, the local sectors to 737,280 each way, and
device traffic from 211.56 MB read and 43.98 MB written to 195.16 MB and 26.59 MB. So
33.79 MB of the launch's 255.54 MB is spill that does reach DRAM, 13.2% of it, and
deleting it is worth 11.776 us of 596.99 us, 1.973%, over 600 order-swapped pairs
against a 1.536 us null, bitwise on all five outputs. Instructions rise 221,184 doing
it, so the win is traffic and not issue. ``standard`` agrees: 294,912 sectors each way
gone, 3.072 us of 154.62 us, 1.987%.

The frame is a ptxas artifact one register wide. Re-assembling the dumped PTX,
``.minnctapersm`` absent gives 16 B and present at either 1 or 2 gives 80 B, all three
at 255 registers; ``.minnctapersm 2`` with ``.maxnreg 254`` gives 16 B at 254, which
still holds two blocks of 128 threads inside 65,536. So the cure is one register of
headroom under the architectural cap and it is orthogonal to the residency ask.
``nvvm.maxnreg`` is in the DSL's NVVM backend and not in its ``LaunchConfig``, so at
4.4.2 the only reachable lever is dropping the ask, and that lever is refused. At
``wide`` the ask is the license that lets ptxas take 255 registers; without it ptxas
takes 105 and rematerialises 393,216 instructions, 18.944 us of 272.38 us, 6.955%.
Nothing the trace can compute separates the two: ``wide`` and acceptance carry the same
256-register budget and the same 255 allocated, and spill 0 B and 80 B. Reopen this when
``LaunchConfig`` carries a register cap, not before.

The frame is 0 B without that lever. It grew to 88 B, 22 slots, and the seventh change
below deletes all of them by rolling one loop. What the twelve reloaded slots hold is
still address arithmetic, and the live set is still what puts them there, but the phase
that owns the peak is the post-barrier tail and not the staging pass: read off the
disassembly one live range to a definition, the body count climbs from 162 at the loop
top to 384 at ``0xd190``, and the growth is entirely in loads, 39 to 138, and in float
accumulators, 20 to 108, while integers stay flat at 137 to 148. 106 of those 108 floats
are ``dushift`` 32, the two score slices 32, ``mrot`` and ``mrotp`` 18 and ``dloc`` 24,
so the float term is the operator's shape. The loads are not: the ``dnow`` residue's lane
loop alone hands ptxas 24 independent global loads and 24 shared ones with nothing
between them, and at the cap it hoists them and pays for them out of the frame.

Two things the sweep says that the live-set bound does not. Rolling the closing residue's
lane loop instead gives 16 B, and rolling both gives 16 B: the allocator is not monotone
in the pressure cut, so one compile of ``cuobjdump -res-usage`` per variant is the
instrument and the bound is only the routing. And two arms that the bound and a
predecessor's reading both nominated do nothing at all -- a smaller prefetch group in the
staging pass, and promoting ``pres`` out of a fragment allocated inside the dynamic lane
loop -- because neither touches a register at ``0xd190``.

Seven changes took that shape from 3528.2 us and 1700.54 MB, the before figure taken
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
- The ``dnow`` residue's lane loop is rolled, ``unroll=1``. The frame goes 88 B to 0 B
  and the local sectors 2,617,344 loaded and 995,328 stored to none, at the same 255
  registers and the same 49,264 B of shared. ``sm__inst_executed`` rises 5.10%,
  107,965,440 to 113,472,000, which is the rolled trip's own address arithmetic and
  branch at 16 warp-instructions a trip against a predicted 8; ``pipe_lsu`` falls 4.34%,
  ``mio_throttle`` 14.7% to 7.0% and the issue rate 33.05% to 35.21%. ``dram__bytes``
  falls 16.0%, 278.67 MB to 234.11 MB. No other shape moves: the frame stays 16 B at
  ``standard`` and ``long`` in bf16 and 8 B in fp16, 0 B at ``tiny`` and ``wide``, and
  ``wide`` gains 30 registers of headroom, 255 to 225.

The seventh change is the only one whose time is below this tree's instruments. Three
NCU brackets read -13.5, -8.1 and -9.5 us on a 565 us launch, agreeing in sign and
disagreeing 42% in magnitude, and the order-swapped event pair returns -1.024 to
-2.560 us against a null of -2.560 to -3.112 us at 1200 pairs, so the null is wider than
the result and the paired clock does not resolve it. The device was contended for every
bracket. What is measured is the counters, and the byte credit prices at the tree's
0.334 us/MB: 44.56 MB is 14.9 us, which brackets the NCU readings. The 1.17 us/MB this
docstring's own sixth change implies over-predicts by 8 to 14 times, because a local
sector is an L1TEX request and not a DRAM byte -- 109.2 MB left L1 and 64.6 MB of it was
L2-resident -- and because that rate came from an arm that moved instructions as well as
bytes. The mechanism is in the stall partition: ``mio_throttle`` falls 7.7 points and
``long_scoreboard`` rises 4.4, so the freed queue slots go to exposed global latency
rather than to issue.

Registers sit at the 255 architectural cap at every shape and the kernel spills at
every shape: 761,856 sectors per launch each way at ``standard`` and ``ragged``,
2,752,512 and 958,464 at ``wide``, 1,966,080 each way at ``long``, 2,985,984 and
995,328 at ``3N = 240 H = 18``. L1 absorbs 3.4% of the spill loads and none of the
stores at the acceptance shape, so the spill is device traffic: 127.40 MB of the
574.89 MB moved there. The carveout is why. Two resident blocks of 48,752 B leave
under 30 KB of the 128 KB unified cache for data, against their own streaming global
staging, so a local line does not survive to its reload.

That paragraph is the state before the sixth change, and in the two-tap form the
acceptance shape had left it: 254 registers, no local sector either way, 506.24 MB.
The spill survived everywhere else, per launch each way at the same date as the table
above: 540,672 at ``standard`` and ``ragged``, 1,376,256 and 688,128 at ``wide``,
2,015,232 at ``long``. So the cap is not what the spill tracks. A live range is: the
increment accumulator is the only one in the lane-loop body that dies inside the body,
and where the body's order lets it die the allocation fits. The one-tap form moved the
spill again in both directions at once, and by the same rule: it holds the fused column
where the two-tap form held one transition at a time, which lengthens the live ranges of
the acceptance shape's five lane blocks and shortens them at ``standard`` and ``wide``,
where the whole extent is one block.

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

Block width is a bound at some shapes and not at others, and :func:`input_threads`
picks between the two forms. At four warps occupancy is 16.7% theoretical against 16.3
to 16.5% achieved at ``L = 64``, with ``launch__occupancy_limit_registers`` and
``launch__occupancy_limit_shared_mem`` both two; ``long`` gets one block, 8.3% against
8.3%, its shared-memory limit reading one, because a 128-row M tile puts its arena at
80,656 B and the lane block is the only lever :func:`lblock` has. Eight warps widen the
tiling in N, not in M, so the chunk's row count is untouched and a thread's accumulator
halves. What that costs is the diagonal GEMM's rereadable left operand, and where it
pays is stated at :func:`input_threads`.

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
    smem_residency,
    widen,
)
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AFUSE,
    TABLE_AN,
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
    THREADS_WIDE,
    WARPS_WIDE,
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
    "LANE_THREADS",
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
    "lane_threads",
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

LANE_THREADS: int = 16
"""Widest run :func:`lane_threads` may cut the block into for the lane reductions.

Sixteen and not the lane count itself. The two reductions -- the diagonal residue's
per-token inner product and the increment residue's per-row one -- sum over the lane
extent, and a butterfly needs the cooperating threads to be a power-of-two-aligned run
within one warp. The lane count is a multiple of :data:`LANE_MULTIPLE` thirds, so it is
a multiple of 16 and every power of two to 16 divides it; the run width is therefore
bounded by the row mapping and not by the lane extent."""

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


def _lane_block(
    chunk: int,
    rows: int,
    dim: int,
    itemsize: int,
    warps: int,
    capacity: int | None = None,
) -> tuple[int, int]:
    """Lane block the carveout admits, and the bytes it costs.

    Split from :func:`lblock` so a shape no block fits has a cost to report. The byte
    query is total over shapes and answers what a refused shape would need;
    :func:`lblock` is the caller that refuses it.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element.
        warps: Warps per block.

    Returns:
        The widest candidate held :data:`RESIDENT_MIN` deep, else the widest held once,
        else the narrowest candidate paired with a cost above the carveout.
    """
    costs = [
        (blk, input_smem_bytes(chunk, rows, dim, itemsize, lblk=blk, warps=warps))
        for blk in range(dim, 0, -LANE_MULTIPLE)
        if dim % blk == 0
    ]
    for candidate in costs:
        if smem_residency(candidate[1], capacity=capacity) >= RESIDENT_MIN:
            return candidate
    available = smem_capacity() if capacity is None else capacity
    for candidate in costs:
        if candidate[1] <= available:
            return candidate
    return costs[-1]


def lblock(
    chunk: int,
    rows: int,
    dim: int,
    itemsize: int = 2,
    *,
    warps: int = WARPS,
    capacity: int | None = None,
) -> int:
    """Lane extent held in shared memory at once.

    The three tiles that carry the lane dimension are the rotated forcing vectors,
    the increment cotangent and the rotated readout. Together they are the whole
    lane-dependent part of the block, and at ``3N = 240`` they are 95,232 B of a
    101,376 B carveout, so the block does not fit at all. Slicing them bounds the
    footprint by the lane block instead of by ``3N``: at ``L = 64``, ``P = 64`` the
    block is 49,264 B at every ``3N``, against 104,560 B at ``3N = 192`` and
    122,992 B at ``3N = 240`` unsliced.

    What the slicing costs is the contraction structure, not traffic. The lane extent
    is a K mode for the forcing cotangent and the score, so both accumulate across
    blocks; the score's accumulator therefore stays live over the whole lane loop. One
    tap, so that is one score and not two, and what a second lane block costs is a
    second staging of the increment cotangent and the readout rather than a second
    score.

    What slicing buys is shared bytes and the occupancy they decide: 49,264 B at
    ``wide`` in the narrow form, which reads 16.4% achieved occupancy and an
    ``launch__occupancy_limit_shared_mem`` of two, against 67,696 B unsliced, which one
    128-thread block per SM holds to 8.3%; and at ``3N = 240`` a kernel that launches at
    all against 122,992 B that does not. The pairwise durations this docstring used to
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
        capacity: Explicit carveout for offline layout modelling. The current
            device is queried when omitted.

    Returns:
        The widest divisor of ``3N`` that is a multiple of :data:`LANE_MULTIPLE` and
        whose block the carveout holds :data:`RESIDENT_MIN` deep, or the widest that
        fits once when none is held that deep. Residency is asked of
        :func:`slinoss._cute.smem_residency` rather than of
        ``capacity // RESIDENT_MIN``, which reads one block too high in the 512 B below
        the cliff because every resident block pays
        :data:`slinoss._cute.SMEM_RESERVED` and the capacity has one of them subtracted
        already. Both passes compare a block against a budget it has been shown to
        meet, so what is returned is a block that fits.

    Raises:
        ValueError: If no candidate fits at all. A width cannot carry the refusal: the
            caller cannot tell a block that fits from one that overflows by its value,
            and every caller of this function goes on to launch at what it returns.
    """
    blk, nbytes = _lane_block(chunk, rows, dim, itemsize, warps, capacity)
    available = smem_capacity() if capacity is None else capacity
    if nbytes > available:
        raise ValueError(
            f"chunk_input_bwd has no lane block that fits at L={chunk} P={rows} "
            f"3N={dim}: the narrowest, {blk}, needs {nbytes} B and the carveout is "
            f"{available} B"
        )
    return blk


def input_threads(chunk: int, rows: int, dim: int, itemsize: int = 2) -> int:
    """Block width for a shape, narrow or wide.

    The wide form doubles the warps and halves every per-thread accumulator. It also
    costs the diagonal GEMM its rereadable left operand: at two N groups a thread's
    consecutive N steps are two atoms apart, so the masked score goes through shared
    memory instead of staying in registers. That is about a fifth more work on the LSU
    issue port, against twice the warp slots. The lane extent decides which side wins.

    One lane block, and the wide form pays. The score accumulator is built once, its
    staging aliases a tile that is already dead, and four warps leave the issue port
    far from saturated. Against the narrow form at locked clocks on sm_86, in kernel
    cycles: 0.409 at ``L = 128, P = 48, 3N = 48``, where the narrow form gets one
    block per SM and runs its port at 18%; 0.882 at ``L = 64, P = 48, 3N = 48``;
    0.921 at that shape with a ragged tail.

    More than one lane block, and it refuses. The score is banked across blocks, the
    staging is restaged with the increment cotangent, and the added port work outruns
    the occupancy: 1.137 at ``3N = 96`` and 1.048 at ``3N = 240``, both taken in the
    two-tap form. The second is the shape the DRAM-bound class is declared against,
    where the narrow form spills nothing at all and the wide form spills 139 MB.

    ``3N = 96`` has since crossed to the wide form and the 1.137 does not apply to it,
    because it is no longer a two-lane-block shape. The one-tap form's two residues put
    512 B into the resident set, which takes the eight-warp arena at a 48-block from
    50,016 B to 50,528 B against the 50,176 B a second resident block admits, so
    :func:`_lane_block`'s first pass finds no candidate held two deep and the second
    returns the whole extent. That makes ``lblk == dim`` and this function takes the
    wide form. Occupancy is unchanged by the trade, eight warps once against four warps
    twice, 16.58% achieved against 16.39%, and what the single block buys is the second
    staging pass: the launch spills 983,040 sectors loaded and 491,520 stored in the
    two-tap narrow form and none in the one-tap wide one, moves 33.36% less device
    traffic, and is 45.064 us shorter over 1536 blocks, interval [-46.088, -45.048] at
    240 pairs against a null band of 1.024 us. ``3N = 240`` is unaffected: the wide
    form's whole extent needs 129,376 B against a 101,376 B carveout, so the width
    stays narrow and :func:`lblock` returns a 48-block. 122,992 B is the narrow form's
    figure at that extent and does not decide this width.

    Three shapes for, two against, and the lane count separates them with nothing left
    over. What the lane count does not decide is whether the wide arena fits at all: at
    ``L = 128, P = 64`` the wide form's 103,264 B overflows a 101,376 B carveout that
    the narrow form's 90,736 B clears. Fitting is checked here rather than left to the
    host guard, which would refuse the launch instead of narrowing it, and it is checked
    at the whole lane extent ahead of :func:`lblock` rather than at the block that
    function returns: the wide form is taken only at one lane block, so a whole extent
    the wide arena cannot hold refuses the width whatever a narrower block would cost,
    and asking first is also what keeps :func:`lblock`'s overflow error off this path.

    Args:
        chunk: ``L``.
        rows: ``P``.
        dim: ``3N``.
        itemsize: Bytes per operand element. Always 2 for the tensor-core atom.

    Returns:
        :data:`slinoss.ops.so3ssd.cute.mma.THREADS_WIDE` when the wide form's own lane
        block holds ``3N`` whole and its arena fits the carveout,
        :data:`slinoss.ops.so3ssd.cute.common.THREADS` otherwise. The block is asked of
        the wide form rather than the narrow one because the wide arena is the larger of
        the two and so decides its own lane extent; where it holds ``3N`` whole the
        narrow one does too.
    """
    bytes_wide = input_smem_bytes(
        chunk, rows, dim, itemsize, lblk=dim, warps=WARPS_WIDE
    )
    if bytes_wide > smem_capacity():
        return THREADS
    lblk = lblock(chunk, rows, dim, itemsize, warps=WARPS_WIDE)
    return THREADS_WIDE if lblk == dim else THREADS


def lane_threads(chunk: int, rows: int, threads: int) -> int:
    """Threads that cooperate on one row of the two residue lane reductions.

    The two residues -- the diagonal's per-token inner product and the increment's
    per-row one -- each reduce the lane extent for one row. Cutting the block into runs
    of ``run`` threads puts ``threads // run`` rows in flight and needs ``log2(run)``
    butterfly rounds per row, over a row loop of ``chunk * run // threads`` steps. Total
    rounds are the product, ``run log2(run) chunk / threads``, so the narrowest run that
    maps rows exactly is the cheapest: at ``acceptance`` a run of 2 costs one round where
    16 costs 32. The lane work per thread is the residue's row count times
    ``lanes / threads`` at every run, the two factors of ``run`` cancelling, so what the
    cut moves is the butterfly count and nothing else.

    The only constraint is the row mapping. ``threads // run`` rows must divide both
    ``L`` and ``P``, or a step walks off the tile and the mapping needs a predicate the
    reduction cannot carry. The lane extent never constrains it: a lane block is a
    multiple of :data:`LANE_MULTIPLE`, so the lanes are a multiple of 16 and every power
    of two to :data:`LANE_THREADS` divides them.

    Measured at ``acceptance``, where this returns 2 against a hardcoded 16. The two
    butterflies are 320 ``SHFL`` per warp over the launch at a run of 16 and 10 at a run
    of 2, and the counter falls by exactly that 310: 4,999,680 to 2,142,720 warp
    instructions. ``short_scoreboard``, which the ``FADD`` dependent on each round sits
    in, halves with it, 18.7% to 9.4%. The launch is 83.7 to 94.2 us shorter over 2304
    blocks, 11.15% to 12.30%, over three paired runs of 240 order-swapped pairs each
    against a null band of 2.048 us; the run-to-run spread is the device's, every run
    having been taken beside a foreign process, and the ratio is quoted as a range for
    that reason. What pays is the MIO class and not the register file: registers stay
    pinned at 255 and spill rises 12%, while shared loads fall 25.7% and the whole MIO
    class 27.3%.

    Args:
        chunk: ``L``.
        rows: ``P``.
        threads: Block width.

    Returns:
        The narrowest power of two whose row mapping divides both extents, at most
        :data:`LANE_THREADS`. Falls back to :data:`LANE_THREADS`, which every legal shape
        admits, when no narrower run maps.
    """
    run = 1
    while run <= LANE_THREADS:
        held = threads // run
        if chunk % held == 0 and rows % held == 0:
            return run
        run *= 2
    return LANE_THREADS


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

    One word per warp per reduced value, so no reduction reuses another's slot and
    the epilogue adds no barrier of its own between them. It does not follow that
    the launch pays one:
    :func:`slinoss._cute.block_reduce_add` carries its own ``sync_threads``, so the
    eleven calls below pay eleven, 26 ``BAR`` per warp at the acceptance shape of
    which eleven are theirs. Collapsing them needs the helper split into a staging
    half and a summing half, which is an edit to that helper and not to this tile.

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
        forced: The fused forcing column, restaged once per lane block.
        local: The increment cotangent in the chunk-local frame, one lane block.
        readout: The rotated readout, one lane block.
        cotangent: The output cotangent.
        shift: The float32 shift tile. Epilogue only.
        trans: ``trans`` staging. Prologue only.
        tap: ``K`` staging. Prologue only.
        quat: Quaternion prefix staging. Prologue only.
        score: The wide form's staged score, below the lane loop. Equal to
            :attr:`local` where more than one lane block restages the increment
            cotangent, and its own words where one block holds it live throughout.
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
        lblk: Lane extent of the three lane-dependent tiles. Defaults to the block
            :func:`_lane_block` admits, which passes it explicitly to ask what a
            candidate would cost. Taken from there rather than from :func:`lblock` so a
            shape no block fits reports its floor cost instead of raising: the launch
            guard is :func:`lblock`, and a legality map wants the number.
        warps: Warps per block. Above :data:`slinoss.ops.so3ssd.cute.mma.WARPS` the
            diagonal GEMM needs its left operand in shared memory.
    """
    if lblk is None:
        lblk = _lane_block(chunk, rows, dim, itemsize, WARPS)[0]
    forced = _words(forced_tile(chunk, lblk), itemsize)
    local = _words(local_tile(rows, lblk), itemsize)
    readout = forced
    cotangent = _words(cotangent_tile(chunk, rows), itemsize)
    tail = max(
        local + readout + cotangent,
        _words(shift_tile(chunk, rows), 4),
        16 * chunk,
    )
    # The wide form's diagonal GEMM stages its left operand below the lane loop. At more
    # than one lane block the increment cotangent is restaged per block and is dead by
    # then, so the staging aliases it and costs nothing; at one block it is staged once
    # and read to the end of the loop, so the staging takes its own words.
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
        lblk: Lane extent of the three lane-dependent tiles. Defaults to the block
            :func:`_lane_block` admits. Total over shapes: where no block fits, the
            narrowest one's cost is reported and the refusal is :func:`lblock`'s.
        warps: Warps per block. Only the two per-warp scratch tiles carry it.
    """
    words = arena(chunk, rows, dim, itemsize, lblk=lblk, warps=warps).words
    return smem_bytes(
        [
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (scalar_tile(chunk), 4),
            (Tile((rows,), (1,)), 4),
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


def _sum_over_lanes(value: cutlass.Float32, run: int) -> cutlass.Float32:
    """Sum one row's contribution over the ``run`` threads sharing it.

    ``log2(run)`` butterfly rounds over a ``run``-aligned run of one warp, so the total
    lands in all of them and the accumulation that follows is by one. Not
    :func:`_sum_over_n`: those threads are the atom's quad, these are the run the two
    residue reductions map a row onto, which no accumulator layout decides. A run of one
    is the identity: that thread already holds the whole lane extent.

    Args:
        value: The thread's contribution.
        run: Cooperating threads. A power of two, at most 32, compile-time.
    """
    mask = 1
    while mask < run:
        value = value + shuffle_xor(value, mask)
        mask *= 2
    return value


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
    gdscore: cute.Tensor,
    gcount: cute.Tensor,
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
        gdscore: ``(B,H,C,L,L)`` operand-dtype, written with the masked score of the
            other product this fragment carries, target token by source token. Not a
            cotangent: it is the record
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_bwd` reads
            in place of forming it once per lane tile. The mask, the factor and both
            operands are already here, so the publish is one multiply, one select,
            one narrow and one store an element.
        gcount: ``(B,H,C)`` int32, written with zero. Not read here: it is the arrival
            counter :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_bwd`
            closes its lane sums on, and this grid holds one block per element of it, so
            the zero rides an already-uniform branch instead of its own launch.
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
    # stager, so this pass widens them itself. ``b`` too: the two residues of the
    # one-tap form need the current tap's forcing vector, which no tile carries.
    dinc_elem = gdinc.element_type
    state_elem = gz.element_type
    b_elem = gb.element_type
    zero = cutlass.Float32(0.0)

    ldu = smem_pitch(rows)
    ldv = smem_pitch(lblk)
    nwarps = threads // 32
    one_group = nwarps == WARPS
    where = arena(chunk, rows, dim, lblk=lblk, warps=nwarps)

    smem = cutlass.utils.SmemAllocator()
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    swgt = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    # The two residues of the one-tap form, both reductions over the whole lane extent
    # and so both accumulated across lane blocks. Resident rather than in the arena:
    # each is written inside the lane loop and read in the epilogue, where the arena
    # holds the shift tile.
    sdnow = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdures = smem.allocate_tensor(cutlass.Float32, Tile((rows,), (1,)).layout(), 16)
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
            sdnow[token] = zero
    for step in cutlass.range_constexpr(-(-rows // threads)):
        row = tid + step * threads
        if row < rows:
            sdures[row] = zero
    # The one-tap column: Afuse_t = Ap_t + exp(2*ls_t) An_{t-1}, applied to b_{t-1}. An
    # keeps its own slot, which the two residues below read per token.
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3, True)
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

    # One output fragment, not two. The one-tap form's diagonal and increment both act
    # on the shifted forcing vector, and the current tap's cotangent is the two residues
    # below: a per-token scalar and a per-row one, neither of them a GEMM.
    dushift = mma_acc(tiled_mma, tid, (mpad, rows))
    wcrd = mma_coords(tiled_mma, tid, (mpad, rows))
    # The lane extent is the score's K mode. Sliced, a slice's score is complete only
    # after the last lane block, so every slice is live at once and the whole
    # ``(mma_rows(L), L)`` score sits in registers: 32 per thread at ``standard``.
    # Unsliced it is complete where it is produced and one accumulator serves every
    # slice, which is 16. The banked form is taken only when it is needed.
    banked = ltiles > 1
    # One row past the tokens that exist, capped at the chunk. Slot ``valid`` of the
    # fused column is ``An_{valid-1}``, so staged against this bound the row holds
    # ``bnow_{valid-1}``: the increment's rank-one residue for a ragged chunk, and the
    # shifted partner of ``dU`` at its last valid token. At a full chunk it is the chunk
    # and the staging is unchanged.
    vplus = cutlass.min(valid + 1, cutlass.Int32(chunk))
    # The two residue reductions. ``lthreads`` threads to a row, so ``trows`` rows or
    # tokens are in flight per step and each thread walks ``lanes // lthreads`` lanes.
    # The lane work per thread is the product and is fixed; what the cut moves is the
    # butterfly count, which is the row loop's trip count times ``log2(lthreads)``.
    lthreads = lane_threads(chunk, rows, threads)
    trows = threads // lthreads
    noff = tid % lthreads
    nrow = tid // lthreads
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
    # Accumulator elements grouped by the word the two log-scale reductions write.
    # ``scols`` groups the score fragment by target token, whose destination is
    # ``sdlp[warp, token]``; ``wrows`` groups the output fragment by source token, whose
    # destination is ``sdlp[warp, m]``. A thread's coordinate is its own base plus a
    # trace-time offset and the base is one value per fragment, so elements sharing an
    # offset share the destination and the grouping is static. Both reductions are
    # linear, so the elements of a group sum in a register and the butterfly is entered
    # once per destination instead of once per element: at ``acceptance`` 8 entries
    # against 16 for the score and 2 against 32 for the output.
    soff = mma_offsets(tiled_mma, (mpad, tblk))
    scols = tuple(
        tuple(i for i in range(len(soff)) if soff[i][1] == col)
        for col in dict.fromkeys(off[1] for off in soff)
    )
    woff = mma_offsets(tiled_mma, (mpad, rows))
    wrows = tuple(
        tuple(i for i in range(len(woff)) if woff[i][0] == row)
        for row in dict.fromkeys(off[0] for off in woff)
    )
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

    # One tap. The fused column carries the previous tap and the shifted current one at
    # once, so the forcing vector always comes from ``t-1`` and the table slot is always
    # :data:`slinoss.ops.so3ssd.cute.common.TABLE_AFUSE`. There is no tap loop, so the
    # increment cotangent and the readout are staged once per lane block at every lane
    # extent, and the chunk-start state is read once whatever ``ltiles`` is.
    #
    # What the second tap used to carry is two residues, both below the GEMMs: the
    # diagonal's ``s == t`` term ``<crot_t, bnow_t> u_t`` and the increment's closing
    # rank-one term ``u_{L-1} (x) bnow_{L-1}``. Neither is a GEMM.
    vu = [
        cute.make_tensor(
            su.iterator + k * MMA_TILE_K,
            cute.make_layout((mpad, MMA_TILE_K), stride=(ldu, 1)),
        )
        for k in range(ksteps)
    ]

    for lt in cutlass.range(ltiles, unroll=1):
        l0 = lt * lblk
        cute.arch.sync_threads()
        # ``vplus`` and not ``valid``: one row past the tokens that exist, whose fused
        # matrix is the last real token's ``An``, so the row is ``bnow_{valid-1}``. On a
        # ragged chunk that row is the increment's closing term and the shifted partner
        # of the last valid token's ``dU``; the unconditional residue below carries the
        # zero it leaves at slot ``L-1``.
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
            vplus,
            tid,
            TABLE_AFUSE,
            1,
            threads,
            mpad,
            lanes,
            has_prev,
            False,
        )
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
        # accuracy and the arithmetic below is float32 (I4).
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
                held.append(
                    (
                        p,
                        n,
                        (
                            widen(pdinc[p, d0], dinc_elem),
                            widen(pdinc[p, d0 + 1], dinc_elem),
                            widen(pdinc[p, d0 + 2], dinc_elem),
                        ),
                        (
                            widen(pz[p, d0], state_elem),
                            widen(pz[p, d0 + 1], state_elem),
                            widen(pz[p, d0 + 2], state_elem),
                        ),
                    )
                )

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
                if cutlass.const_expr(not exact):
                    # A clamped step repeats the last element, so its store
                    # repeats a correct value and only the reductions need
                    # the zero. Zeroing the state zeroes both of them.
                    live = tid + (group * PREFETCH + step) * threads < total
                    state = tuple(select(live, state[j], zero) for j in range(3))
                # The closing scale rides the state rather than the finished
                # sum, because the rotation cotangent's other half comes
                # from the forcing product below and is not scaled.
                scaled = tuple(cscale * state[j] for j in range(3))
                for j in cutlass.range_constexpr(3):
                    dscale = dscale + local[j] * state[j]
                    for i in cutlass.range_constexpr(3):
                        mrot[3 * i + j] = mrot[3 * i + j] + local[i] * scaled[j]
        cute.arch.sync_threads()

        # sum_d bfuse(r,d) dinc_local(p,d), the increment's contribution to the
        # forcing cotangent, accumulated unweighted into the output fragment. No
        # accumulator of its own: the fragment carries nothing else until the
        # diagonal GEMM below the lane loop, and the weight is a function of the
        # source token alone, so weighting the finished sum over lane blocks is the
        # same sum as weighting each block's part. The epilogue that applies it then
        # runs once per launch rather than once per lane block.
        for k in cutlass.range_constexpr(lsteps):
            mma_gemm(tiled_mma, tid, dushift, vforced[k], vlocal_n[k], True, True)

        # sum_p ushift(r,p) dinc_local(p,d), the other half of the increment's outer
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

        # The diagonal's ``s == t`` residue, ``dnow_t = <crot_t, An_t b_t>``, one scalar
        # per token. Not a GEMM: every other term of row ``t`` comes from the fused
        # column, and this one contracts the lane extent against a single token's
        # current-tap forcing vector. :func:`lane_threads` threads to a token,
        # ``log2`` of that many butterfly rounds, one accumulation into the resident tile
        # per lane block, so the sum over the whole lane extent is the sum of the blocks'.
        #
        # The token loop encloses the lane loop so the nine table entries are read once
        # per token, which is the second thing a narrow run buys: at one token in flight
        # per thread the table is read once per lane block, not once per token step.
        # ``An`` is the zero matrix at a padded token, which is what bounds the clamped
        # read of ``b``: the row it repeats contributes nothing.
        for ts in cutlass.range_constexpr(chunk // trows):
            token = nrow + ts * trows
            anow = tuple(stable[TABLE_AN, token, e] for e in range(9))
            tsafe = cutlass.min(token, valid - 1)
            diag = zero
            # Rolled, and it is the whole spill. Unrolled this loop hands ptxas 24
            # independent global loads and 24 shared ones with nothing between them,
            # and at the 255-register cap it hoists them all: the frame was 88 B of
            # loop-invariant address arithmetic, twelve words reloaded at the top of
            # every trip. Rolled it is 0 B at the acceptance shape, and the trip's
            # own address arithmetic is the fee. Rolling the closing residue's lane
            # loop as well is worse, 16 B, so it is one loop and not both.
            for q in cutlass.range(lanes // lthreads, unroll=1):
                n = noff + q * lthreads
                d0 = l0 + 3 * n
                bnow = mat3_matvec(
                    anow,
                    (
                        widen(gb[bidx, gidx, t0 + tsafe, d0], b_elem),
                        widen(gb[bidx, gidx, t0 + tsafe, d0 + 1], b_elem),
                        widen(gb[bidx, gidx, t0 + tsafe, d0 + 2], b_elem),
                    ),
                )
                for j in cutlass.range_constexpr(3):
                    diag = diag + widen(sc[token, 3 * n + j], elem) * bnow[j]
            diag = _sum_over_lanes(diag, lthreads)
            if noff == 0:
                sdnow[token] = sdnow[token] + diag

        # The increment's closing residue, ``u_{L-1} (x) An_{L-1} b_{L-1}``, whose weight
        # is one by construction, so it is added below the weight epilogue rather than
        # inside it. Two contractions: the forcing cotangent's last row, one scalar per
        # output row, and the rotation cotangent's share, which the residue owes in the
        # same form the forcing GEMM's epilogue does. ``An_{L-1}`` is the zero matrix on a
        # ragged chunk, where slot ``L-1`` carries no token, so the residue vanishes there
        # and the clamped read needs no predicate.
        #
        # The lane loop is outermost so the matrix-vector runs once per lane instead of
        # once per lane and row, and the nine rotation terms are taken against the row sum
        # rather than per row. ``su[L]`` is the last token's forcing vector.
        anlast = tuple(stable[TABLE_AN, last, e] for e in range(9))
        tlast = cutlass.min(cutlass.Int32(last), valid - 1)
        pres = cute.make_fragment((rows // trows,), cutlass.Float32)
        pres.fill(0.0)
        for q in cutlass.range_constexpr(lanes // lthreads):
            n = noff + q * lthreads
            d0 = l0 + 3 * n
            bnow = mat3_matvec(
                anlast,
                (
                    widen(gb[bidx, gidx, t0 + tlast, d0], b_elem),
                    widen(gb[bidx, gidx, t0 + tlast, d0 + 1], b_elem),
                    widen(gb[bidx, gidx, t0 + tlast, d0 + 2], b_elem),
                ),
            )
            wsum = [zero, zero, zero]
            for ps in cutlass.range_constexpr(rows // trows):
                p = nrow + ps * trows
                ulast = widen(su[chunk, p], elem)
                for i in cutlass.range_constexpr(3):
                    loc = widen(sdinc[p, 3 * n + i], elem)
                    pres[ps] = pres[ps] + loc * bnow[i]
                    wsum[i] = wsum[i] + loc * ulast
            for i in cutlass.range_constexpr(3):
                for j in cutlass.range_constexpr(3):
                    mrot[3 * i + j] = mrot[3 * i + j] + wsum[i] * bnow[j]
        for ps in cutlass.range_constexpr(rows // trows):
            summed = _sum_over_lanes(pres[ps], lthreads)
            if noff == 0:
                row = nrow + ps * trows
                sdures[row] = sdures[row] + summed

    # The increment weight, on the finished sum over lane blocks, and the exponent's
    # forcing term with it. The fragment holds only the increment product here, so
    # this is the whole of what the weight applies to.
    for i in cutlass.range_constexpr(cute.size(dushift)):
        m, p = wcrd[i]
        weighted = dushift[i] * swgt[cutlass.min(m, last)]
        dexpw = dexpw + weighted * widen(su[m, p], elem)
        dushift[i] = weighted

    for s in cutlass.range_constexpr(slices):
        tbase = s * tblk
        acc = score[s] if banked else score[0]
        if cutlass.const_expr(not banked):
            # One lane block, so the forcing tile and the readout still hold it
            # and the slice's score is taken here rather than banked.
            acc.fill(0.0)
            for k in cutlass.range_constexpr(lsteps):
                mma_gemm(tiled_mma, tid, acc, vforced[k], vreadout[s][k], True, True)
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
        # One address for the record's whole slice, not one an element. Both bases are
        # this thread's own and common to the fragment, so ``soff`` addresses every
        # element of it at trace time and the store carries an immediate. Unread at a
        # shape whose M mode is rounded up past the record's own extent, where the pad
        # rows have to be dropped and the store takes the dynamic source token under a
        # predicate instead.
        grec = cute.make_tensor(
            gdscore[bidx, hidx, cidx, None, None].iterator
            + (tbase + scrd[0][1] - soff[0][1]) * chunk
            + (scrd[0][0] - soff[0][0]),
            cute.make_layout((mpad, tblk), stride=(1, chunk)),
        )
        for k in cutlass.range_constexpr(ksteps):
            mma_gemm(tiled_mma, tid, dmacc, vu[k], vb_dy[k], True, True)
        # Column-major over the fragment, one group of ``scols`` at a time. The
        # exponent's cotangent is summed over the source token, which is this
        # accumulator's M mode, and its destination is one word per target token: the
        # rows a thread holds at one column sum in a register and cross the lanes once.
        # Its other sum, over the target token, is the pair of inner products the
        # epilogue takes against the finished fragments, so the score is never
        # revisited.
        for g in cutlass.range_constexpr(len(scols)):
            group = scols[g]
            _, n = scrd[group[0]]
            token = tbase + n
            summed = zero
            for e in cutlass.range_constexpr(len(group)):
                i = group[e]
                m, _ = scrd[i]
                # I6: the mask lands on the float32 accumulator, then one narrowing
                # into the operand. I3: one exponential of a log difference. The clamp
                # only feeds rows the M mode was rounded up by, whose operands the
                # stagers zeroed.
                factor = decay(slp[token] - slp[cutlass.min(m, last)])
                causal = token >= m
                masked = select(causal, acc[i] * factor, zero)
                sfrag[i] = narrow(masked, elem)
                summed = summed + masked * dmacc[i]
                # The same mask and the same factor on the other product of this
                # fragment pair, which is the record ``chunk_vector_bwd`` reads
                # instead of forming it once a lane tile. One multiply, one select,
                # one narrow and one store an element; no second exponential and no
                # reread of either operand.
                published = narrow(select(causal, dmacc[i] * factor, zero), elem)
                if cutlass.const_expr(mpad == chunk):
                    grec[soff[i][0], soff[i][1]] = published
                else:
                    # The record carries the source tokens that exist. The rows the M
                    # mode was rounded up by reach no column of it, and the consumer
                    # zeroes the pad rows of its own tile.
                    if m < chunk:
                        gdscore[bidx, hidx, cidx, token, m] = published
            column = _sum_over_m(summed)
            if tid % 32 < 4:
                sdlp[warp, token] = sdlp[warp, token] + column
        if cutlass.const_expr(one_group):
            mma_gemm_areg(tiled_mma, tid, dushift, fa_score, vdiag, False)
        else:
            # At two N groups a thread's consecutive N steps are two atoms apart,
            # so the fragment cannot be reread as a K-contiguous A operand. The
            # score goes through shared memory instead. The first barrier covers
            # the lane loop's last read of the aliased tile and the previous
            # slice's ldmatrix; the second covers this slice's store.
            cute.arch.sync_threads()
            cute.autovec_copy(sfrag, tiled_mma.get_slice(tid).partition_C(vscore))
            cute.arch.sync_threads()
            mma_gemm(tiled_mma, tid, dushift, vscore, vdiag, True, False)

    # The fragment is final, so the log-scale sum over the source token, the carry and
    # the shift all read it in place. One term, not two: the diagonal residue's factor is
    # ``D(t,t) = 1`` and the increment residue's weight is ``wgt_{L-1} = 1``, so neither
    # carries a log scale and the current tap contributes nothing here.
    # Row-major over the fragment, one group of ``wrows`` at a time. The inner product
    # is over the output's N mode and its destination is one word per source token, so
    # the columns a thread holds in one row sum in a register and cross the quad once.
    for g in cutlass.range_constexpr(len(wrows)):
        group = wrows[g]
        m, _ = wcrd[group[0]]
        rowsum = zero
        for e in cutlass.range_constexpr(len(group)):
            _, p = wcrd[group[e]]
            rowsum = rowsum + dushift[group[e]] * widen(su[m, p], elem)
        held = _sum_over_n(rowsum)
        if tid % 4 == 0 and m < chunk:
            sdlp[warp, m] = sdlp[warp, m] - held
    for i in cutlass.range_constexpr(cute.size(dushift)):
        m, p = wcrd[i]
        if m == 0:
            gcarry[bidx, hidx, cidx, p] = dushift[i]

    # Out of this thread's rotated basis, where slot ``a`` holds component
    # ``(lphase + a) % 3``. Eighteen selects once, against three per accumulator
    # element per lane block in the component basis.
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
        # The next launch's arrival counter, zeroed on the branch that was already
        # uniform and already storing. This grid holds one block per element of it, so
        # the fill needs no launch of its own; the launch boundary between here and the
        # reader is the ordering.
        gcount[bidx, hidx, cidx] = cutlass.Int32(0)

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

    # The diagonal residue, into a fragment of the output's shape and not into the store
    # below, because the shift tile aliases the output cotangent: at the acceptance shape
    # the two overlap by 768 float32 words, so ``sdy`` is gone once ``sshift`` is written.
    # A fragment and not an arena move, which would cost 5,120 B at ``P = 128``. Live only
    # here, where the lane loop's pressure is over.
    ddiag = cute.make_fragment_like(dushift)
    for i in cutlass.range_constexpr(cute.size(ddiag)):
        m, p = wcrd[i]
        ddiag[i] = sdnow[cutlass.min(m, last)] * widen(sdy[m, p], elem)
    # The two tiles alias and have different pitches and element widths, so a thread's
    # own store does not cover the element another thread has still to read. The last
    # barrier above this is inside the block reductions, which run before the read.
    cute.arch.sync_threads()

    for i in cutlass.range_constexpr(cute.size(dushift)):
        m, p = wcrd[i]
        sshift[m, p] = dushift[i]
    cute.arch.sync_threads()

    for i in cutlass.range_constexpr(cute.size(ddiag)):
        m, p = wcrd[i]
        # The row above, clamped and then replaced, never predicated. Inside the chunk the
        # partner is the next row of the shifted fragment, which on a ragged chunk is the
        # first padded row and carries the last valid token's current tap: the bound is the
        # chunk and not ``valid``, because slot ``valid`` of the fused column is
        # ``An_{valid-1}``. At the last slot there is no row above and the increment's
        # closing residue takes its place. ``m + 1 < chunk`` and ``m == L-1`` are exact
        # complements inside the stored range ``m < valid <= L``, so one select delivers
        # both.
        above = sshift[cutlass.min(m + 1, mpad - 1), p]
        held = ddiag[i] + select(m + 1 < chunk, above, sdures[p])
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
    gdscore: cute.Tensor,
    gcount: cute.Tensor,
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
        gdscore,
        gcount,
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
        dscore: ``(B,H,C,L,L)`` masked score in the activation dtype, target token by
            source token, not a cotangent. The record
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_backward`
            takes in place of forming the same product once per lane tile. Both
            products of the pair are in this kernel's fragment and the mask and the
            factor are already applied to the other one, so the record leaves here
            for one multiply, one select, one narrow and one store an element.
        arrived: ``(B,H,C)`` int32 zeros, not a cotangent. The arrival counter
            :func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_backward`
            closes its lane sums on, filled here because this grid is one block per
            element of it and the fill would otherwise be a launch.
    """

    dU: Tensor
    carry_u: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor
    dscore: Tensor
    arrived: Tensor


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
    threads: int | None = None,
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
            every per-thread accumulator and adds two per-warp scratch rows. None
            takes :func:`input_threads`, the measured choice for the shape; an
            explicit width overrides it and is what an A/B against that choice
            passes.

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

    if threads is None:
        threads = input_threads(chunk_size, rows, dim, dy.element_size())
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
    # The masked score, 18.87 MB at the acceptance shape. Held by the caller across
    # the vector stage and freed there: the alternative is the vector stage forming
    # it once per lane tile, which is five times at that shape for one answer.
    dscore = torch.empty(
        bsz, heads, chunks, chunk_size, chunk_size, dtype=dtype, device=device
    )
    arrived = torch.empty(bsz, heads, chunks, dtype=torch.int32, device=device)
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
            dscore,
            arrived,
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
            min(RESIDENT_MAX, smem_residency(budget)),
        ),
    )
    return ChunkInputBwd(
        dU=dU,
        carry_u=carry_u,
        dlogp=dlogp,
        dchunk_rot=dchunk_rot,
        dchunk_scale=dchunk_scale,
        dscore=dscore,
        arrived=arrived,
    )
