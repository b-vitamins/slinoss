# Kernel notes

The chunked scan factorizes so that, after one rowwise change of basis into the
chunk-local frame, every contraction is a dense real GEMM. Every rule below
exists to preserve that, or follows from a counter that was read off the
hardware.

## The GEMM forms

Four shapes cover the operator, forward and backward:

    increment  D(P,3N) = U(L,P)^T @ Bn(L,3N)     M=P,  N=3N, K=L
    score      D(L,L)  = Cr(L,3N) @ Bn(L,3N)^T   M=L,  N=L,  K=3N
    diagonal   D(L,P) += S(L,L) @ U(L,P)         M=L,  N=P,  K=L
    offset     D(L,P)  = Cr(L,3N) @ Z(P,3N)^T    M=L,  N=P,  K=3N

The backward reaches the same four with modes widened or narrowed and the
k-major flags flipped per operand. Flipping a compile-time flag is free; storing
a transposed score tile is not.

## Composition and staging

- Compose the rotation and the tap into explicit 3x3 matrices once per token, in
  shared memory. Every vector transform is then a 9-FMA 3x3 matvec whose matrix
  operand is a broadcast shared-memory read, bank-conflict-free by
  construction. The rotation and the tap are never applied as separate per-lane
  passes.
- That rule was held under review on the backward for two profiles, on the
  argument that a broadcast read is conflict-free and still costs a full LSU
  instruction, so the matvec form buys freedom from conflicts with instruction
  count. The review is closed and the rule survives it. Measured on sm_86 at the
  acceptance shape, the table reads the rule prescribes are 12,648,960 `LDS.32`,
  137.25 per warp, 40.5% of `LDS` and 14.9% of the LSU term, which prices at
  214.6 us of a 2,652.5 us launch. Deleting the table and everything that reads
  it is worth 8.1%, so the form is not what makes this kernel slow. Two further
  measurements say the same thing from the other side: 76.9% of the kernel's
  shared loads are already classified broadcast or conflict-free, so these reads
  are as cheap per instruction as shared memory gets and no access-pattern change
  reaches them; and the tensor pipe the alternative would move work onto sits at
  13.86%, against an LSU class that is 22.56% of all instructions issued. A form
  that puts the 3x3 apply on the tensor pipe has to stage its operands in fewer
  than 137.25 instructions per warp to win anything at all, and it cannot win
  more than 8.1% however cheap it is. The rule stands on the measurement rather
  than on the assertion it was written with.
- Chunk-local prefixes are recomputed inside every kernel that needs them,
  forward and backward. They never cross a kernel boundary and never touch
  global memory. They are shared across all `N` lanes and all `P` rows, so one
  warp computes them per chunk and broadcasts.
- No standalone kernel whose only job is a rowwise transform of a `3N`-sized
  tensor. The 3x3 matvec has arithmetic intensity near 1.5 flop/byte, so it is
  memory bound and must fuse into its producer or consumer. The rotated `B` and
  `C` never reach global memory.
- No staging copies to satisfy a kernel's layout preference.
- A pitched band is read at its own pitch, so the pitch sets sector alignment. A
  pitch that is not a multiple of the 32-byte sector starts every other row
  mid-sector: a third more read sectors at a 96-byte row, and about 8% more time
  on a DRAM-bound kernel. The achieved fraction cannot see that loss, since it
  divides measured bytes by measured time and both rise together, so the width of
  the tensor the band is cut from carries the requirement.
- cuBLAS reads a GEMM's operand alignment off its extents, so an extent that is
  not a whole number of operand-load elements costs the kernel its wide load and
  half its MMA K-extent, not a predicated edge. At bf16 that unit is 8. A width
  that appears on a gating mode of several stages takes all of them down together:
  the head's output width is the forward's `N`, the input gradient's `K` and the
  weight gradient's `M`. Census, one Ampere part, bf16, clocks unlocked, foreign
  residents throughout so the durations are stamped and the comparison rests on the
  ratios: at 8192 tokens, `d_model` 576 and a width of 50257 all three ran on
  `cutlass_75_tensorop_bf16_s1688gemm_bf16_128x256_*_align1` at 47.8-48.4%,
  58.2-59.5% and 64.0-65.4% of an in-process bf16 ceiling of 112.1-114.4 TFLOPS.
  Widening to 50264 put the forward and the input gradient on
  `s16816gemm ... align8`/`ldg8` and the weight gradient on `s1688gemm ... ldg8`,
  which reach the ceiling inside its own run-to-run range: 1.84x, 1.83x and 1.65x,
  10.1 ms of the class's 59.3 ms. Widths that are also whole tiles, 50304 and
  50432, moved none of the three further, so the config rule is one operand load
  and not one output tile.
- A warp's global access is charged per request, not per sector. Lanes covering
  consecutive elements of one row fill the request; lanes strided by the warp
  width leave the last segment carrying `width mod 32` lanes and that fraction of
  the request's bytes. At a 48-element row the strided map averaged 48 bytes a
  request against 64, and packing the columns cut the time of a kernel whose DRAM
  traffic did not move. L1 sector count rose in the same change, so sectors are
  not the quantity to minimize. Sectors per request is, read against the
  instruction's own bytes per lane. Bytes over sectors is not a coalescing
  measure at all: NCU's global byte counters are sector-granular by
  construction, so that quotient is exactly 32 for every kernel and can never
  fire.
- No `torch.zeros` or `aten::fill_` on a hot path. Accumulators initialize
  inside kernels.
- No gradient tensor doubles as scratch. No tensor whose name and contents
  disagree.

## Occupancy and shared memory

- Shared memory is not the scarce resource. Registers and occupancy are. Spend
  shared memory to buy them: query the device capacity and opt into the
  carveout, since the 48 KiB default is not the budget. Never hardcode an
  architecture string.
- It buys them only up to half the carveout. On a 101,376 B carveout that is
  50,688 B; past it the second resident block is gone, so the spending stops
  buying occupancy and starts costing it. Count the tapped region and the operand
  tiles, not the accumulators alone.
- Past that point the grid is the remaining lever on latency hiding, and on a kernel
  pinned by both registers and shared memory it is not a lever on occupancy.
  `chunk_vector_bwd` is the standing example: at `P = 64` no legal chunk size reaches
  two blocks per multiprocessor. Putting the loop extents in the grid took it from
  128 blocks to 11,520 and from 1.52 waves to 137 per multiprocessor, which was worth
  taking, and occupancy did not move off 16.6% because two limiters hold it there at
  once. Put the loop extents in the grid, and price the result as issue rate rather
  than as occupancy.
- A footprint that grows with a configuration knob is a ceiling on that knob.
  Tile the knob, or the supported range is whatever the arena happens to hold.
  The backward holds whole `3N` extents, which caps `d_state` well below what
  `SLinOSSConfig` accepts.
- The shared-memory budget is computed from the layouts and asserted against the
  queried capacity by a test. No guard or slop constants.
- Bank conflicts are a bug, not a tradeoff. Conflict freedom comes from the
  pitch: every shared tile is allocated at an odd number of 16-byte units, a
  power-of-two pitch being the worst case. `smem_pitch` in `ops/so3ssd/cute/mma.py`
  is the one implementation; a hand-picked pitch is a defect. The pitch depends
  on the element size, so a float32 tile takes `fp32_tile`, not the operand
  pitch.
- No kernel launches fewer blocks than twice the SM count unless it is provably
  serial, documented as such, and measured under 2% of step time.

## The LSU port

A kernel far under its DRAM bar is not moving bytes slowly. It is issuing memory
instructions, and the class bar says nothing about that.

- The unit of cost is the LSU warp-instruction, not the byte and not the shared
  wavefront. On GA10x the port issues 4 threads per cycle per scheduler, which is
  half a warp-instruction per multiprocessor per cycle, so **every** LSU
  warp-instruction costs two multiprocessor cycles: the same for an 8-bit access as
  for a 128-bit one, and the same however few lanes are active. At 84
  multiprocessors and 1.78 GHz that is 13.43 ps each, 74.4 k per microsecond. The
  rate is clock-dependent; recompute it as `2 / (SMs * clock)` rather than quoting
  74.4 k.
- The identity that establishes it, on `chunk_vector_bwd` at two clocks:

      sm__inst_executed_pipe_lsu / (SMs * SM_active_cycles * 0.5) == L1/TEX throughput
      166,152,960 / (84 * 5,421,733 * 0.5) = 72.97%   reported 72.97%
      166,152,960 / (84 * 5,456,401 * 0.5) = 72.51%   reported 72.50%
      110,856,960 / (84 * 4,122,470 * 0.5) = 64.03%   reported 64.03%
       86,803,200 / (84 * 3,782,303 * 0.5) = 54.64%   reported 54.64%

  The reported throughput is the instruction count and nothing else, so a kernel
  near that ceiling is issue-bound on the port and no byte, wavefront or occupancy
  figure will say so.
- `l1tex__throughput` is not a bandwidth, and neither is `sm__throughput`. Across
  five arms the L1/TEX figure equals `sm__inst_executed_pipe_lsu` against its own
  peak digit for digit (55.05 / 55.21 / 55.11 / 54.93 / 54.94), which is why the
  identity above closes at all: the two are the same quantity. `sm__throughput` is
  that same LSU figure re-based on elapsed instead of active cycles,
  `55.05 * 0.98670 = 54.32` against a reported 54.33. So neither metric reports
  bytes, and a kernel cannot be judged bandwidth-bound from either one.
- Every pipe has its own issue width, and the width is read off
  `.avg.peak_sustained` rather than derived. Measured on this part, in
  warp-instructions per multiprocessor per cycle, with every identity
  `inst / (SMs * cycles_active * width)` closing against the reported percentage to
  0.005 points:

      pipe                                       width    ps per warp-inst
      LSU, XU, ADU, TEX, IPA                       0.5              16.934
      ALU, UNIFORM, IMAD.WIDE, IMAD.HI               2               4.234
      FMA (FFMA, FMUL, FADD, plain IMAD), CBU        4               2.117
      HMMA                                    4 SM-cycles          33.868

  The picoseconds are at 1.406 GHz over 84 multiprocessors and are clock-dependent;
  the widths are not. **One memory instruction costs four integer instructions or
  eight float instructions, and one `HMMA` costs two memory instructions.** That
  ratio, not the instruction count, is what ranks two candidate forms.
- Divide by processing blocks, not by multiprocessors, or the answer is out by four.
  The ALU is 16 INT32 units per scheduler, so an integer warp-instruction does cost
  two cycles -- confirmed to the last digit, `sm__pipe_alu_cycles_active.avg`
  predicted 2,926,628.57 and measured 2,926,628.57 -- but it spends them in one of
  four parallel blocks, which is a width of 2.0 per multiprocessor and not 0.5. The
  same reasoning applied per multiprocessor priced the integer term at 82.6% of a
  launch whose busiest pipe was 55.05%. `sm__throughput` refuted it before any arm
  was dispatched against it. An arithmetic derivation over a datasheet is the
  weakest evidence class there is; check it against a counter before acting.
- The issue-view and cycles-view counters disagree where an instruction issues at
  reduced rate, and the cycles view is the one to price with.
  `sm__pipe_fma_cycles_active` runs 8.9% above what the FMA instruction count
  predicts at full rate; the excess is exactly the 12,810,240 `IMAD.WIDE.U32`,
  `IMAD.HI.U32` and `IMAD.HI` instructions issuing at half rate, and pricing those
  correctly moves FMA from 11.61% to 12.65%. `HMMA` occupies 16 unit-cycles across
  4 units, so its cycles view is exactly twice its issue view: 13.85%, not 6.94%.
- Price every pipe at once and the limiter cannot hide. On `chunk_vector_bwd`: LSU
  55.05%, ALU 19.91%, tensor 13.85%, FMA 12.65%, XU 3.73%, ADU 1.61%, CBU 0.30%,
  uniform 0.03%, FP64 zero. The eight budgets sum to 107.1% of the active wall, so
  the machine is accounted for and there is no unmeasured unit left for a limiter to
  sit in. The LSU port alone exceeds the sum of every other pipe on the chip, by a
  factor of 2.76 over the next one. Replays are nil: `sm__inst_issued` exceeds
  `sm__inst_executed` by 0.007%.
- An instruction class being large does not make it a limiter, and this is where the
  census alone misleads. Integer work is the largest class in `chunk_vector_bwd` by
  count, 1,333.75 warp-instructions per warp against LSU's 921.88, and it prices at
  520 us against LSU's 1,439 because it issues at four times the width. So no
  integer-reduction arm can be rank 1 while the port is where it is. Rank by pipe
  occupancy, then by instruction count within the limiting pipe.
- Deleting work off a pipe that is not the limiter is unpriced, not free money. The
  measured conversion of freed cycles to time, 83-93%, was established on the
  limiting pipe only. A candidate that moves work from one idle pipe to another idle
  pipe -- packing two `F2F` on the XU into one `F2FP` on the ALU, worth 33 us by port
  arithmetic -- has no measured conversion behind it and does not reduce the port
  term at all. State such a candidate as unpriced rather than banking its
  microseconds.
- A freed port cycle is not a freed nanosecond. Across three group widths of
  `chunk_vector_bwd`, 93.7% of the freed LSU cycles converted to time at one step
  and 83.3% at the next, so the port term is an upper bound on what deleting an
  instruction buys and the fraction falls as the term shrinks. Predict with it,
  then measure.
- Warp shuffles issue on the port. `SHFL` moves no bytes, produces no wavefronts and
  cannot conflict, and it costs full price. The butterfly in `_sum_over_lanes` was
  69,258,240 warp-instructions on `chunk_vector_bwd`, 30.4% of the kernel and the
  single largest item in it, and every wavefront-denominated estimate priced it at
  zero. Narrowing the group the butterfly spans from 16 lanes to 4 cut it to
  11,197,440 and the kernel by 29% of its cycles, at one register less and no spill.
  Rank an arm by the instructions it deletes.
- The ranked cost is the instruction census, one row per opcode. On
  `chunk_vector_bwd`, before and after that cut:

      opcode    16 lanes       4 lanes     4 lanes + paired stores
      SHFL    69,258,240    11,197,440    11,197,440
      LDS     49,271,040    32,129,280    31,207,680
      STS     20,989,440    19,054,080    18,132,480
      LDSM    14,008,320    14,008,320    14,008,320
      LDG      6,842,880     6,842,880     6,842,880
      STG      5,276,160     3,064,320     3,064,320
      S2R              -             -       368,640
      total  166,152,960    86,803,200    84,960,000
      port         73.0%         54.6%        55.06%

  The three reduction tails beside it are DRAM-bound and issue almost nothing.
- A census is complete only when it closes against the counter with nothing left
  over. The first two columns above left 506,880 instructions, 5.5 per warp,
  unnamed, and an unnamed row is a place a wrong conclusion hides. It resolved into
  `S2R` at 4.0 per warp, which is a register move off the special-register file and
  not a memory access at all, plus a 1.5 per warp shortfall of the `LDG` counter
  against the `SASS` line count. Name the row or state that it is unnamed. Two
  narrower disagreements are open and stated rather than folded away: the
  `op_shared_st` counter under-counts `SASS` `STS` by 3.0 per warp, and two harnesses
  that agree on a store delta to the instruction disagree on the absolute by 99 per
  warp, so a delta from one report is not comparable to an absolute from another.
- A census goes stale the moment an arm lands, and so does everything derived from
  it. The group cut halved the total and moved the largest row from `SHFL` to `LDS`.
  It also moved the dominant stall from `mio_throttle` to `wait`, and then the store
  pairing moved it back, so a stall ordering read once is not a property of the
  kernel. Every rank, share and width histogram taken at the old width was void in
  one commit. Re-profile before choosing the next arm.
- Rank by source line, not only by opcode, because an opcode row spans call sites
  that different arms reach. Setting `CUTE_DSL_LINEINFO=1` in the profiling
  environment is sufficient and costs nothing measurable: instruction counts and
  register allocation come out bit-identical and the duration moves 0.004%. Two
  defects bound what it can say. The line numbers are sound and the file identity is
  not, because one `.file` is emitted for the whole module, so intersect the reported
  lines with the module's own location set. And an op is attributed to the innermost
  frame that emitted it, so a helper called from many sites reports as the helper and
  cannot be split among its callers. Locate a shared-memory region by its immediates
  in `SASS` rather than by inference, then attribute every access to it by base
  register and check that the parts sum to the whole.
- Shared wavefronts are a second-order term, not the currency. They set how long a
  conflicted access occupies the pipe, so a conflict is still a bug, but removing
  wavefronts without removing instructions does not move an issue-bound kernel:
  3.3 M excess wavefronts out of 128 M, from an `STS.64` pitch defect, price at
  roughly nothing while the port binds. The earlier reading of this kernel derived a
  marginal 102.4 k wavefronts per microsecond and a 0.41 payback ratio from two
  landed arms. Both arms removed instructions as well as wavefronts, and the
  instruction count alone accounts for their microseconds. The ratio was a wrong
  denominator, not a property of the hardware.
- A width change is therefore worth more than the wavefront law allows, not less.
  Folding four scalar `LDS` into one `LDS.128` leaves the wavefront count invariant
  and deletes three instructions of four. Two adjacent bfloat16 accesses pack into
  one 32-bit access for a straight halving. Price the lever off the width histogram
  of the tree in hand, never off a share carried forward. On `chunk_vector_bwd`:

      width    instructions    per warp
       16b       20,643,840       224.0
       32b       35,619,840       386.5
       64b        3,214,080      34.875
      128b       14,192,640       154.0

  Perfect 2:1 packing of the 16-bit class deletes 10,321,920 instructions, 245,760
  port cycles, 175 us, 6.6% of the launch. The same lever was advertised at 835 us
  from a share taken one arm earlier, which was 4.8x optimistic. The histogram sums
  to the `SASS` memory-instruction total exactly, so it is a partition and not an
  estimate.
- 16-byte access also changes the conflict question, and the scalar answer does
  not carry over. An `LDS.128` is serviced in four phases of eight threads, each
  phase covering the full 128-byte width, so the unit is the 16-byte segment and
  the modulus is 8 rather than 32. A 48-byte row is 4-way conflicting at scalar
  width, `gcd(12, 32) = 4`, and conflict-free at vector width, `segment = 3t mod
  8` being a bijection on eight threads. `48 = 3*16` divides exactly, so three
  `float4` cover the row with no remainder. The tile base must be 16-byte aligned
  for it.
- That argument is about the row, so check what the access is actually shaped like
  before spending on it. A 3-vector is 12 bytes at float32 and 6 at bfloat16 and no
  16-byte form of it exists, whatever the tile's base: the widest the tree found is
  a lane pair. The nine-float table entry is different -- consecutive and divisible
  -- but `table_tile` pitches it at 9, so the token stride is 36 bytes and only every
  fourth entry is 16-byte aligned. Pad the innermost extent to 12 and the stride is
  48, every entry is aligned, and three `float4` replace nine scalar reads. The
  padding is what makes the vector form legal, and it is not free: it is a third more
  table bytes, against a shared budget the residency already binds.
- Bound an arm against every floor at once, or the binding one stays hidden. At
  `L 64 P 64 3N 240` the six contractions of `chunk_vector_bwd` count 34.12 G flop,
  40% of the backward's 85.46 G. That puts its tensor floor at 304.7 us and its 70%
  bar at 435.3 us; its 515.84 MB of counted traffic puts its DRAM floor at 889.8 us;
  its instruction census puts the port term at 3,956,022 SM-cycles before the group
  cut and 2,022,857 after it, which is 2,232 us and 1,166 us at 1.773 GHz but 2,818
  and 1,441 at the 1.404 GHz the profiler actually clocked. Quote the cycles. The
  port bound by a factor of 2.5 over the next floor and still
  binds above the target, so the arithmetic is not the obstacle and neither is the
  traffic. The port term is not a floor in the way the other two are: it is the
  current instruction count priced, and it falls with every instruction deleted --
  which is also why it cannot be quoted from a previous arm.
- Read the dominant stall to name the queue, then attribute the stall by opcode
  before prescribing for it. `mio_throttle` at 28.16% of warp-active cycles said the
  MIO queue was saturated; PC sampling said 55.69% of those samples sat on `SHFL` and
  25.55% on `LDS`. A saturated queue whose top occupant is a shuffle is not short of
  requests in flight, so adding independent loads per warp -- the standing
  prescription for a latency-bound warp -- makes the top stall worse. The same
  correction applies to the latency term: `short_scoreboard` at 9.84% is held by
  `FADD` at 25,259 of about 41 k samples, which are the butterfly's own dependent
  adds, not a load. Stall percentages name a queue; only attribution names an
  instruction.
- Attribution also decides whether a stall is worth an arm at all, and the shape of
  its distribution is the answer. After the group cut `mio_throttle` at 18.18% and
  `wait` at 17.65% were within noise of each other, and they are not the same kind of
  object. `mio_throttle` concentrates: 98.31% of its samples sit on the LSU class,
  `LDS` 39.63%, `SHFL` 21.66% at 7.3x its instruction share, `STS` 19.72%, and single
  `SASS` lines carry 2-3% each. `wait` is flat: its hottest line is 0.35%, the top 25
  lines carry under 8%, and it spreads over roughly 4,900 lines with only `HMMA` at
  3.0x its share and the alignment `NOP`s beside the MMA sequences at 12.6x standing
  out. So `wait` is tensor result latency plus a tax spread over everything, and
  there is no chain in it to cut. `stall_barrier` is the opposite extreme, four
  predicated `BRA` lines carrying 43.7% of its samples. Rank a stall by how
  concentrated its attribution is, not by its percentage.
- Occupancy is not the answer when two limiters pin it independently. On
  `chunk_vector_bwd` `launch__occupancy_limit_registers` and `_shared_mem` are each
  exactly 1, so cutting registers alone cannot buy a second block while the arena is
  91.6 KB, and cutting the arena alone cannot while the kernel is at 242 registers.
  Achieved occupancy is 16.58-16.63% against a theoretical 16.67%, which is 99.5%:
  there is no imbalance and no tail to recover, and the launch runs 137 waves per
  multiprocessor, so wave quantization is not in play either.
- `ldmatrix` reads per flop are set by the warp tiling, not by the instruction.
  The A operand is broadcast across the N warp groups and B across the M groups,
  so a `(warps_m, warps_n)` tiling of an `M*N*K` tile reads `warps_n*M*K +
  warps_m*N*K` and the ratio is `warps_n/N + warps_m/M`. Widening a block in N
  buys issue rate and pays that ratio: four warps on a `(64, 16)` tile read
  0.1250, eight warps read 0.1875, and eight warps on `(64, 32)` read 0.1250
  again. Widen the tile with the block or the width costs 1.5x in operand reads.

## DSL rules that come from measurement

- Unrolling fixes the loop counter and cannot fix a tensor stride. Every stride
  arrives as a runtime kernel parameter in the constant bank, so a fully unrolled
  body still recomputes each address rather than folding it into an immediate.
  `chunk_vector_bwd` is unrolled to the point that every `SASS` site executes
  exactly once per warp, and it still issues 1,334 integer instructions per warp.
  Of those, 282 per warp exist only because a global address is 64 bits wide with a
  runtime stride -- `IMAD.WIDE.U32`, `IADD3.X`, `LEA.HI.X` -- and 64.4% of all
  integer work sits in four staging and epilogue regions, two of which contain no
  floating-point instruction at all, against 6.3% in the two matvec bodies. Address
  arithmetic is a property of the parameter, so reduce it by touching fewer
  addresses, not by unrolling harder.
- Clamping unconditionally is right, and it is not free. The rule below says never
  predicate a load; the compiler honours it by materialising every clamp and every
  in-bounds mask as a value select paid on all lanes. On `chunk_vector_bwd` that is
  382 warp-instructions per warp of `ISETP`, `SEL` and `IMNMX`, 27.5% of the integer
  work, with only 3 per warp of all integer work under a predicate. Correct, and
  worth knowing before attributing that count to indexing.
- A cross-lane reduction costs more than its shuffle. `SHF.L.U32` appears exactly
  once per `SHFL.BFLY` -- 10,506,240 of each -- because the lane mask is computed in
  the ALU. A butterfly step is 16.934 ps on the port plus 4.234 ps on the ALU, not
  16.934.
- No global load inside a divergent branch, and none inside a `cutlass.range`
  loop that also transforms what it loads: neither can be unrolled or hoisted,
  so both serialize on one global latency per step. Split load from transform --
  a `range_constexpr` group of `PREFETCH` loads, then the group's math. Clamp the
  index unconditionally and correct with `select`; never predicate a load. The
  rule holds while the unrolled body fits the instruction cache. Past that the
  serialization is the cheaper cost: `chunk_input_bwd`'s lane loop both loads and
  transforms, and rolling it moved the dominant stall from `no_instruction` at
  48.3% and a 9.21% issue rate to `long_scoreboard` at 29.2% and 24.31%, halving
  the launch. Measure both arms once the body is large.
- The DSL emits no phi node for a dynamic `if`, so a value produced inside a
  dynamic branch cannot be read after it. A trace-time `const_expr` branch is
  plain Python and has no such limit.
- `cutlass.range_constexpr` is rewritten only as the iterable of a `for`
  statement. In a comprehension or a generator expression it reaches the runtime
  stub and raises. Use a plain `range` there; both unroll at trace time.
- A fragment reaches registers only if every loop between its allocation and its
  uses has a trip count of one. Allocations are hoisted to kernel entry, and a
  rolled loop in between defeats promotion: each access becomes a local load and
  a local store. Measured on `chunk_vector_bwd` at 255 registers and 91,344 B of
  shared memory in both runs, one lane tile moves no local traffic and five move
  1,892.16 MB per launch. The declaration site is not the lever, a bounded partial
  unroll halves the traffic rather than removing it, and a kernel with two rolled
  loops has to reach trip count one in both at once. Read the local load and store
  sector counters at more than one `3N`; registers per thread alone hides this.
- Trip count one is where that rule bites, not the whole account of a spill.
  `chunk_scan_fwd` holds both accumulators across a rolled loop of trip count two
  at the standard shape and moves no local traffic, and its spill is non-monotonic
  in `3N`: 144 spills, 192 is clean, 240 spills, every spilling geometry sitting
  at 255 registers. That is ptxas resolving an over-subscribed live set at the
  architectural ceiling, not a footprint law. Rolling `chunk_input_bwd`'s lane
  loop cut its local loads from 19,574,784 sectors to 6,156,288 without clearing
  them, because the residual is loop-invariant state rematerialized once per trip
  rather than accumulator thrash. Name which of the two a spill is before picking
  a fix, and sweep the geometry: one shape cannot tell them apart.
- Compile once. Every launch goes through the executor cache in
  `slinoss/_cute.py`; a `@cute.jit` function called directly retraces on every
  call.
- CuTe owns non-GEMM rowwise math. GEMMs stay on cuBLAS or CUTLASS unless a
  fused variant is measured to win.

## One implementation each

- The quaternion exponential, composition, conjugation, the tap chart, the 3x3
  composition, and both prefixes have exactly one device-side implementation.
  Duplicated math diverges, and the divergence is a correctness bug.
- Parameter gradients are nine float32 accumulators per token per tap, reduced
  over the lane dimension, fused into the `dB` and `dC` epilogues. There is no
  dedicated parameter-gradient kernel and there will not be one: the two
  predecessor implementations shipped one at 32 threads per CTA and 4.75%
  achieved occupancy.
- Any lane-indexed reduction over 3-vectors has an explicit stride loop and is
  tested at two distinct `N`. A kernel correct at one shape is not correct.
- One entry point per kernel. A variant reachable from the benchmark and not from
  the public path is a defect; the benchmarked path is the shipped path.

## Roofline class

Every kernel is declared `DRAM-bound`, `TENSOR-bound`, or `SERIAL-tiny` and held
to that class:

- `DRAM-bound`: at least 85% of measured achievable bandwidth.
- `TENSOR-bound`: at least 70% of achievable tensor-core throughput.
- `SERIAL-tiny`: under 2% of step time.

A kernel that is none of the three is a defect, not a result. The declaration
lives in the module docstring with the analytic byte count it rests on, and a
`SERIAL-tiny` claim is a measurement, not an assertion. Redeclaring a class to
make a number pass is the same defect as loosening a tolerance.

The bandwidth a `DRAM-bound` kernel is held to is measured at the kernel's own
footprint, not at the largest one the device can run. A copy carries a fixed cost,
so its rate rises with its size, and a rate measured at 512 MiB is not a
denominator for a kernel moving 10 MB. The probe sweeps the copy over footprints
spanning thirty-twofold, each above L2, and fits `t = c + bytes/B`; the floor at
the kernel's traffic comes from the fit, with the fixed term charged once per
launch. The fit's residual is reported beside the verdict, because the
extrapolation to a small footprint is only worth that residual.

Below L2 there is no verdict. Measured traffic under the cache size is not a lower
bound on the work a launch did, so the same kernel scores anywhere depending on
what L2 already held, and the fit is extrapolated there as well. The kernel is
named as unjudged rather than passed or failed. This is the reading at the smallest
shape, where DRAM reads are zero.

A percentage far under the bar names the shortfall and not its cause. Nothing in
the class model is a statement about instruction issue, so a kernel at 13.6% of its
bandwidth is reporting that bytes are not what it is spending its time on. Read the
LSU instruction census beside the percentage; see "The LSU port" above.

The three classes do not cover the machine. `chunk_vector_bwd` sat at 23-24% of DRAM
speed of light and 13.86% of the tensor pipe while spending 73.0% of the LSU issue
port, so it was issue-bound and the model has no name for that. The group cut took
the port term to 55.06% against 26.61% of DRAM, so the kernel is now no single
thing, which the model has no name for either. Its instruction classes read `INT`
33.87%, `FP32` 30.34%, `LSU` 22.56%, `MOVE` 5.10%, `TENSOR` 2.83%: the rule set's own
bottleneck rule calls compute and memory well balanced, which is the tooling
independently reporting that no throughput unit is the limiter. The declaration in
`slinoss/perf/declared.py` stays as it is and the kernel stays failing until either
the port term comes down or the model grows a fourth class with its own measured
bar. Declaring the shortfall away is not available.

The percentage also cannot rank two ways of computing the same thing, because an
arm that deletes traffic takes bytes out of its own numerator. `start_passing_bwd`
scored 55.2% moving 176 MB in 474.9 us against the 78.7% and 99.3% of the two
kernels it replaces, which together moved 454 MB in 741.7 us. A row band on the
same kernel scored 64.9% against 64.0% while running 11% slower. Rank arms by
microseconds and use the class to ask whether one kernel is done.

A register spill fails a `DRAM-bound` or `TENSOR-bound` kernel outright, whatever
the percentage says. Both classes hold a counted quantity against a duration, and
a spill moves both: at three blocks per SM `chunk_scan_fwd_kernel` scored 2.7
points higher than the same body at two blocks per SM while running 10% slower,
114.8 us against 104.4 at the standard shape on sm_86.
Local-memory sectors are read in their own NCU pass and are required, not
optional -- a pass that was never run must not read as clean.
