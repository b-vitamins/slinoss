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
- A warp's global access is charged per request, not per sector. Lanes covering
  consecutive elements of one row fill the request; lanes strided by the warp
  width leave the last segment carrying `width mod 32` lanes and that fraction of
  the request's bytes. At a 48-element row the strided map averaged 48 bytes a
  request against 64, and packing the columns cut 15% off a kernel whose DRAM
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
- It buys them only up to half the carveout. Past that the second resident block
  is gone, so the spending stops buying occupancy and starts costing it, and a
  live set that also drives the allocator to the 255-register cap pays twice.
  `chunk_vector_bwd` is the standing example: 85,424 B, one block per
  multiprocessor, 11.8 MB of spill per launch, 13.5% of its DRAM time floor.
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
  index unconditionally and correct with `select`; never predicate a load.
- The DSL emits no phi node for a dynamic `if`, so a value produced inside a
  dynamic branch cannot be read after it. A trace-time `const_expr` branch is
  plain Python and has no such limit.
- `cutlass.range_constexpr` is rewritten only as the iterable of a `for`
  statement. In a comprehension or a generator expression it reaches the runtime
  stub and raises. Use a plain `range` there; both unroll at trace time.
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
points higher than the same body at two blocks per SM while running 5.8% slower.
Local-memory sectors are read in their own NCU pass and are required, not
optional -- a pass that was never run must not read as clean.
