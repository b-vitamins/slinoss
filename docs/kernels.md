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
- No `torch.zeros` or `aten::fill_` on a hot path. Accumulators initialize
  inside kernels.
- No gradient tensor doubles as scratch. No tensor whose name and contents
  disagree.

## Occupancy and shared memory

- Shared memory is not the scarce resource. Registers and occupancy are. Spend
  shared memory to buy them: query the device capacity and opt into the
  carveout, since the 48 KiB default is not the budget. Never hardcode an
  architecture string.
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
