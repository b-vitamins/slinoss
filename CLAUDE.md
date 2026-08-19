# CLAUDE.md

Project instructions for `slinoss`. Read before changing anything.

## What this is

`slinoss` is an oscillatory state-space sequence mixer. The core operator is a
chunked scan whose per-step homogeneous dynamics are an SO(3) rotation plus an
isotropic scale, applied by quaternion conjugation, with two-tap first-order-hold
forcing. The public mixer is `SLinOSSMixer`.

Every operator lives in `slinoss/ops/<name>/` and has the same five parts:

- `reference.py` -- pure PyTorch. The mathematical authority. Correctness is
  defined here and nowhere else. Holds the reference forward, the reference
  backward, and the gradient named types.
- `cute/` -- CuTe DSL CUDA kernels and their host wrappers. Fast path. Must match
  the reference within a declared tolerance at every supported shape. Nothing
  else: no autograd function, no public differentiable callable, no named type.
- `backends.py` -- the forward and backward `Protocol`s, the `Registry`, and the
  registrations.
- `interface.py` -- the one `torch.autograd.Function` and the one public callable.
- `__init__.py` -- the re-export surface.

A kernel is never the specification. If a kernel and the reference disagree, the
kernel is wrong until proven otherwise in float64.

## Environment

Guix only. Never `pip`, `apt`, `npm -g`, or `cargo install`. Every dependency
belongs in `manifest.scm`.

```
guix shell -m manifest.scm -- python3 -m pytest -xvs
guix shell -m manifest.scm -- ruff format . && ruff check .
guix shell -m manifest.scm -- pyright
```

Use `python3`, never `python`.

Guix package name mappings: `requests` ->
`python-requests`, `sklearn` -> `python-scikit-learn`, `yaml` ->
`python-pyyaml`.

## Gate before every commit

```
ruff format . && ruff check . && pytest -xvs
```

All three must be clean. `pyright` must also be clean. Never commit unless
asked. Never amend another author's commit. Never commit `.env`, `secrets/`, or
credentials. Conventional commit format. Do not add `Co-Authored-By` trailers or
any agent attribution to commit messages.

## Layout

```
slinoss/
  mixer.py        SLinOSSMixer
  blocks.py       SLinOSSBlock
  stack.py        SLinOSSStack
  config.py       SLinOSSConfig, and the shape multiples every operator asserts
  state.py        inference state containers
  decode.py       single-token decode path
  graph.py        CUDA-graph capture and replay
  _precision.py   float32-pinning policy, dtype sets
  _registry.py    Backend and Registry, shared by every operator
  _cute.py        the one device-side helper set and the executor cache
  _guard.py       host-side layout, device, and dtype checks
  ops/
    so3ssd/       the scan operator
      cute/
        common.py   MMA constants, tile arithmetic, shared-memory pitches
        table.py    the 3x3 transform table and every staging helper
        prefix.py   both chunk-local prefixes
        mma.py      the four GEMM forms
        guard.py    shared-memory budget against the queried capacity
        forward.py  the forward host path
        fwd/        chunk_increment, state_passing, chunk_scan
        bwd/        the backward kernels
    scanprep/     parameter maps: rotation vector, log-scale, taps
    mixer/        fused rowwise mixer tail
    block/        fused norm and activation kernels
    conv/         causal conv1d; the fast path is the C++ extension, not CuTe
  _C/             causal conv1d extension bindings
  perf/           budget taxonomy, region timers, memory forensics, workload table
csrc/             causal conv1d CUDA and C++
tests/
scripts/{bench,perf,aot}/
examples/
assets/           plots only, 300 DPI minimum, vector where possible
```

`slinoss/_cute.py`, `_guard.py`, `_registry.py`, and every shared module under
`ops/so3ssd/cute/` are single-implementation modules. A helper that belongs in one
of them is added there, never copied into a kernel module.

## Tensor contracts

Time-major, contiguous, no exceptions. Backends do not transpose or repack
inputs to suit a kernel; a kernel that wants a different layout is rewritten.

| tensor  | shape         | dtype           |
|---------|---------------|-----------------|
| `U`     | `(B,H,T,P)`   | bf16/fp16/fp32  |
| `trans` | `(B,H,T,4)`   | fp32            |
| `K`     | `(B,H,T,2,4)` | fp32            |
| `B`     | `(B,H,T,3N)`  | bf16/fp16/fp32  |
| `C`     | `(B,H,T,3N)`  | bf16/fp16/fp32  |
| `z`     | `(B,H,P,3N)`  | fp32            |
| `Y`     | `(B,H,T,P)`   | bf16/fp16/fp32  |

`trans` packs `(w_x, w_y, w_z, ls)`. `K` packs per tap `(kr, g, h, 0)` with tap
index `0` = previous and `1` = current; lane 3 is a hard zero, present for
float4 alignment.

The trailing `3N` is `N` independent 3-vectors in lane-major order: element
`3n+i` is component `i` of 3-vector `n`. One quaternion acts on every lane
identically.

`N` must be a multiple of 16, so `3N` is a multiple of 48 and therefore of 16.
This makes every contraction MMA-k friendly with no padding. Do not add a
padding path; fix the shape constraint instead.

The multiples live in `config.py` and nowhere else: `LANE_MULTIPLE = 16`,
`STATE_MULTIPLE = 48`, `HEAD_MULTIPLE = 16`, `MIN_CHUNK = 16`, `MAX_CHUNK = 128`.
`MAX_CHUNK` is a shared-memory bound, not a taste: the score tile is `L*L` and the
staged operands scale with `L`, so `L = 128` puts `chunk_scan` at roughly 56 KB and
one block per SM. `L` is a lever at long `T`, where the grid absorbs the occupancy
loss; it is not a lever at the standard shape.

One exception to the no-padding rule, and only one. In a GEMM the `N` and `K` modes
must be multiples of 16 and are, by the constraints above; the `M` mode is free,
because `P` is a row count and not a contraction extent. `M` is rounded up to
`MMA_TILE_M`, the pad rows are zero-filled, and the store is predicated on the real
row count. That is padding inside one tile's register fragment, never a padded
tensor and never a staging copy.

## Numerical invariants

These are guaranteed by parameterization, then asserted by tests. Do not add a
clamp, an epsilon, or a branch to work around one. If a kernel needs a guard,
the parameterization is wrong.

1. `ls <= 0`. Therefore the chunk-local log-scale prefix is monotone
   non-increasing and every decay factor lies in `(0,1]`. Overflow is
   unreachable; underflow is graceful and correct.
2. `|w| <= w_max < pi`. Therefore the quaternion exponential is a single
   branchless minimax polynomial accurate to float32 epsilon over the whole
   reachable domain. Average active threads per warp stays at 32.00.
3. Never factor a segment decay as `exp(2*lp_t) * exp(-2*lp_s)`. Always form
   `exp(2*(lp_t - lp_s))` from the log difference. Underflow times overflow is
   how NaN gets in.
4. `trans`, `K`, the per-step quaternions, both chunk-local prefixes, and the
   3x3 transform table are float32 everywhere, including under autocast. Only
   `U`, `B`, `C`, `Y`, the score matrix, and GEMM operands are low precision.
5. Quaternion prefix products are renormalized once per chunk after the scan.
   Rotation error enters the rotation matrix squared; unit-norm drift is not
   tolerated.
6. The score decay mask is applied to the float32 accumulator after the GEMM,
   never folded into a bfloat16 operand.

## Tap parameterization

The tap acts on each 3-vector as

```
K(v) = kr * v + g * (w . v) w + h * (w x v)
```

This is a polynomial in `w`, analytic at `w = 0`. Do not reintroduce the axis
normal form `k_par v_par + k_re v_perp + k_im (a x v)` with `a = w/|w|`: it is
singular at the origin, costs an `rsqrt` and a clamp, and forces a whole-tensor
validity check. The polynomial chart makes the well-definedness condition
structural. The two are related by `k_re = kr`, `k_par = kr + g*|w|^2`,
`k_im = h*|w|`.

## Kernel engineering rules

The chunked scan factorizes so that, after one rowwise change of basis into the
chunk-local frame, every contraction is a dense real GEMM. Preserve that.

- Compose the rotation and the tap into explicit 3x3 matrices once per token,
  in shared memory. Every vector transform is then a 9-FMA 3x3 matvec whose
  matrix operand is a broadcast shared-memory read, which is bank-conflict-free
  by construction. Do not apply the rotation and the tap as separate per-lane
  passes.
- Chunk-local prefixes are recomputed inside every kernel that needs them,
  forward and backward. They never cross a kernel boundary and never touch
  global memory. They are shared across all `N` lanes and all `P` rows, so they
  are computed once per chunk by one warp and broadcast.
- No standalone kernel whose only job is a rowwise transform of a `3N`-sized
  tensor. The 3x3 matvec has arithmetic intensity near 1.5 flop/byte; it is
  memory bound and must be fused into its producer or consumer. The rotated
  `B` and `C` never reach global memory.
- No kernel launches fewer blocks than twice the SM count unless it is provably
  serial, documented as such, and measured under 2% of step time.
- No `torch.zeros` or `aten::fill_` on a hot path. Accumulators initialize
  inside kernels.
- No staging copies to satisfy a kernel's layout preference.
- Bank conflicts are a bug, not a tradeoff. Conflict freedom comes from the pitch,
  not from a swizzle functor: the pinned DSL cannot compose a swizzled layout with
  the MMA fragment loads these kernels need, so every shared tile is allocated at
  an odd number of 16-byte units. `smem_pitch` in `common.py` is the one
  implementation; do not hand-pick a pitch. Measured ways for a bf16 tile: 48 -> 2,
  56 -> 1, 64 -> 8, 72 -> 1, 80 -> 2. A power-of-two pitch is the worst case, which
  is why the rounding is to an odd multiple and not to the next power of two.
- Never load from global inside a divergent branch, and never inside a
  `cutlass.range` loop that also transforms what it loads. `cutlass.range` lowers
  to IR that ptxas cannot unroll, and a load under a branch cannot be hoisted above
  it, so both shapes serialize the loop on one full global latency per step. Split
  load from transform: issue a `range_constexpr` group of `PREFETCH` loads into
  trace-time lists, then transform the group. Clamp the index unconditionally and
  fix the result with `select`; never predicate the load. This is also forced by the
  DSL, which emits no phi node for a dynamic `if`, so a value produced inside a
  branch cannot be read after it.
- Compile once. A `@cute.jit` function called directly retraces on every call.
  Every launch goes through the executor cache in `slinoss/_cute.py`.
- Parameter gradients are nine float32 accumulators per token per tap, reduced
  over the lane dimension. They fuse into the `dB` and `dC` kernel epilogues.
  There is no dedicated parameter-gradient kernel and there will not be one.
- Shared memory is not the scarce resource. Registers and occupancy are. Spend
  shared memory to buy them. Query the device capacity and opt into the carveout;
  the 48 KiB default is not the budget. Never hardcode an architecture string.
- The shared-memory budget is computed from the layouts and asserted against the
  queried capacity by a test. No guard or slop constants.
- Quaternion exponential, composition, conjugation, the tap chart, the 3x3
  composition, and both prefixes have exactly one device-side implementation.
  Duplicated math diverges, and the divergence is a correctness bug.
- Any lane-indexed reduction over 3-vectors has an explicit stride loop and is
  tested at two distinct `N`. A kernel correct at one shape is not correct.
- No gradient tensor doubles as scratch. No tensor whose name and contents
  disagree.
- One entry point per kernel. A variant reachable from the benchmark and not from
  the public path is a defect; the benchmarked path is the shipped path.
- CuTe owns non-GEMM rowwise math. GEMMs stay on cuBLAS or CUTLASS unless a
  fused variant is measured to win.

## Dispatch

One entry point per operator, not one per implementation. An operator whose fast
path is reachable only by importing a kernel module has two entry points, and the
second one is untested by definition.

- `interface.py` holds the operator's single `torch.autograd.Function` and its
  single public callable. The callable takes `backend: str | None`, resolves, then
  applies. Nothing else constructs an autograd function for that operator.
- `backends.py` holds the forward and backward `Protocol`s, the `Registry`, and the
  registrations: reference at priority 0 over `("cpu", "cuda")` with
  `SUPPORTED_DTYPES`; CuTe at priority 10 over `("cuda",)` with `KERNEL_DTYPES`,
  behind a `torch.cuda.is_available()` check and then an `ImportError` guard. A tree
  with no CUDA and no DSL must still import and resolve to the reference.
- The gradient named type lives in `reference.py`. Defining it in the kernel module
  forces the reference backend to import a kernel to name its own return type, which
  inverts the dependency and breaks the CPU-only tree.
- Resolution is on device type and activation dtype. Shape is not a resolution axis.
- No `torch.amp.custom_fwd`. It casts every input to the autocast dtype, which is
  the opposite of I4. The backend decides the promotion.
- The registry's resolution rules are tested once, against a fixture the test owns.
  They are not re-tested per operator.

## Optimization workflow

One change at a time, always measured.

1. Baseline: run the bench and the NCU report for the target kernel at the
   standard sizes. Save both.
2. Hypothesis: name the bottleneck from the counters. Propose one focused
   change.
3. Implement exactly that change. Do not bundle.
4. Validate: run the kernel's parity tests.
5. Re-measure with the identical bench and NCU commands.
6. Keep only if it improved without regressing correctness. Otherwise revert.
7. Record the delta.

Use `scripts/bench/` and `scripts/perf/`. Never write an ad-hoc timing script
outside them; extend them instead.

Rank candidates from the stall decomposition, never from a byte count alone. A byte
count says where the floor is; it does not say what the kernel is waiting on. A
kernel at half its bandwidth ceiling with `long_scoreboard` near 100% is latency
bound, and every traffic-cut hypothesis aimed at it is mis-ranked. Reading a
bottleneck off the shape of an indexing expression instead of off the layout and the
counters has already produced one retracted finding.

Every kernel is declared `DRAM-bound`, `TENSOR-bound`, or `SERIAL-tiny` and
held to that class: at least 85% of measured achievable bandwidth, or at least
70% of achievable tensor-core throughput, or under 2% of step time. A kernel
that is none of the three is a defect, not a result. The declaration lives in the
module docstring with the analytic byte count it rests on, and a `SERIAL-tiny`
claim needs a committed step-time measurement, not an assertion.

## Measurement honesty

Each rule below exists because it has been violated before.

- Every duration and rate field carries its unit in its name: `duration_us`,
  never a `duration_ms` field holding microseconds.
- A bandwidth derived from an analytic byte count is named `model_gbs` and is
  never printed adjacent to a measured figure. Do not report a number above
  hardware peak.
- Trust `dram_pct` over reconstructed read and write rates.
- Lock clocks, or mark the result unlocked in the report.
- Cross-check CUDA-event, NSYS, and NCU totals. Disagreement beyond 5% means
  the report refuses to emit rather than picking a favourite.
- A budget bucket that reads exactly zero is a broken label, not a free
  operation. A test asserts every declared bucket is nonzero on the fused path.

## Test policy

100% coverage on public APIs. Coverage is the floor, not the goal: every test
justifies its own existence by naming the thing that breaks without it. An
indiscriminate parametrize product is a defect, not thoroughness.

- Sweep an axis; do not cross it. A parametrize axis that does not interact with
  another axis is swept once: full sweep on the interacting axis, one representative
  case per independent axis, and the non-interaction stated in the docstring. Dtype
  crossed with shape crossed with backend is a product that tests one thing many
  times and costs the suite its ability to be run.
- A rule shared by N operators is tested once, against a fixture the test owns.
- A pure performance change that alters no contract needs no new test. Its evidence
  is the measurement, and the existing parity tests are what protect it.
- Write the failing test before fixing a bug. Always.
- Correctness ground truth is float64 autograd through the reference, not a
  hand-derived VJP. A hand-derived reference shares its derivation with the
  kernel, so a derivation error passes silently.
- `gradcheck` in float64 on every gradient. No quantity is exempt.
- The forward and the backward must be connected in at least one test: compute
  the output with the fast path, then backpropagate through it, and compare
  against the reference end to end. Testing a backward against a surrogate
  forward hides any disagreement between the surrogate and the real kernel.
- Never derive an intermediate from `randn` when the real pipeline can produce
  it. A fabricated chunk-start state does not test chunk composition.
- Shapes are swept, not fixed: sequence length not a multiple of the chunk,
  single chunk, three or more chunks, `B = 1`, `H = 1`, smallest legal `N`,
  smallest legal `P`, and the streaming split.
- Tolerances must be tight enough to fail. A tolerance loose enough to admit
  any output is not a test. Justify every tolerance above `1e-2` in a comment.
  Never write a tolerance, or a measured error in a comment beside one, that has
  not been run and read off. A fabricated bound is worse than a loose one, because
  it reads as evidence.
- Never relax a gate. Do not xfail, skip, or loosen an existing test to make a
  change pass. A failing existing test is the change's bug.
- Every `raise` in a public path has a test that triggers it.
- Missing CUDA or missing CuTe skips cleanly at module level. It never errors
  at collection.

## Style

- Technical, terse, imperative. No `we`, `you`, or `our`. No marketing
  language. ASCII only.
- Prose is minimal: README, docs, comments, and docstrings state the fact and
  stop. No intensifiers, no adjective piles, no narrative, no sentence that only
  restates the previous one. A comment explains the non-obvious constraint, not
  the code.
- Docstrings state shapes, dtypes, and invariants. Google style.
- Public API returns named types, never long positional tuples. A function
  returning eighteen positional values is an API defect.
- Every package directory has an `__init__.py`. A namespace package that works
  in-tree and breaks in a wheel is a packaging bug.
- Do not describe performance in a comment unless a committed measurement backs
  the claim.
