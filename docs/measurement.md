# Measuring

## The loop

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

Use `scripts/bench/` and `scripts/perf/`. An ad-hoc timing script outside them is
a measurement nobody can reproduce; extend them instead.

Rank candidates from the stall decomposition, never from a byte count alone. A
byte count gives the floor; it does not say what the kernel waits on. A kernel at
half its bandwidth ceiling with `long_scoreboard` dominant is latency bound, and
a traffic cut will not move it. Read the bottleneck off the layout and the
counters, never off the shape of an indexing expression.

## Honesty

Each rule below exists because it has been violated before.

- Every duration and rate field carries its unit in its name: `duration_us`,
  never a `duration_ms` field holding microseconds.
- A bandwidth derived from an analytic byte count is named `model_gbs` and is
  never printed adjacent to a measured figure. A number above hardware peak is
  never reported.
- Trust `dram_pct` over reconstructed read and write rates.
- Lock clocks, or mark the result unlocked in the report.
- Cross-check CUDA-event, NSYS, and NCU totals. Disagreement beyond 5% means the
  report refuses to emit rather than picking a favourite.
- A contended device produces a number, not a measurement. Stamp it and do not
  quote it.
- A contention stamp names the physical part it probed, by UUID. `nvidia-smi`
  numbers devices the way the driver does and `CUDA_VISIBLE_DEVICES` renumbers
  only torch's ordinals, so a stamp that names an ordinal is a claim about
  whichever device the driver numbered the same, and it reads clean because the
  probe succeeded.
- Every quoted figure is measured twice.
- A budget bucket that reads exactly zero is a broken label, not a free
  operation. A test asserts every declared bucket is nonzero on the fused path.
- A performance claim in a comment needs a committed measurement behind it.
- A kernel's declared class needs a benchmarked operator that launches it. A
  class no driver reaches is a claim with no gate behind it, and it reads as
  verified because the audit judges only what a capture contained.
- A ceiling is measured in the same process, on the same device, at the same
  clocks as the kernel it is a denominator for, and at the kernel's own
  footprint. A ceiling carried over from another run drifts against the number
  it divides.
- A fitted denominator is reported with its residual. A fit quoted without one is
  an extrapolation presented as a measurement.
- A percentage of a ceiling is an efficiency, not a speed. Two configurations of
  one kernel are ranked by duration; the percentage says only how much of the
  bus each reached.
