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
- Past that point the only remaining lever on latency hiding is the grid.
  `chunk_vector_bwd` is the standing example: at `P = 64` no legal chunk size
  reaches two blocks per multiprocessor, so at 128 blocks over 84
  multiprocessors it runs 1.52 waves at 8.33% occupancy and no traffic fix
  touches that. Put the loop extents in the grid instead.
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

## DSL rules that come from measurement

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

A register spill fails a `DRAM-bound` or `TENSOR-bound` kernel outright, whatever
the percentage says. Both classes hold a counted quantity against a duration, and
a spill moves both: at three blocks per SM `chunk_scan_fwd_kernel` scored 2.7
points higher than the same body at two blocks per SM while running 10% slower,
114.8 us against 104.4 at the standard shape on sm_86.
Local-memory sectors are read in their own NCU pass and are required, not
optional -- a pass that was never run must not read as clean.
