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

## The denominator

A DRAM-bound kernel is scored against the time its own measured DRAM traffic
implies: `dram_time_floor` fits `c + bytes / B` from a copy sweep, and the verdict
is that floor over the measured duration, against `CLASS_FLOOR_PCT`. The
denominator is therefore the kernel's own bytes, and a kernel that moves less
traffic is measured against a smaller floor.

The case that exposed it, measured on sm_86 at the model geometry. The pair the
backward used to launch moved 453.9 MB in 741.0 us and scored 78.7% and 99.3%. The
fused kernel that replaced it moves 176.5 MB in 474.9 us and scores 55.2%. The
fusion is 36.0% faster and moves 61.1% less, and it reads red. The mechanism is
arithmetic: deleting a round trip removes bytes from the numerator and from the
floor together, so the percentage holds only when time falls in the same
proportion as traffic, and here traffic fell further than time.

The floor is not the thing that is wrong. Derived from those two rows, the pair
achieved 612.6 GB/s and the fused kernel achieves 371.7 GB/s, about half the bus.
That is what 55.2% says, and it is true: the fused kernel is no longer limited by
DRAM. What became wrong is the class. A percentage of a bus ceiling is an
efficiency and never a ranking, so two arms that move different traffic are
compared by duration at fixed work, and the kernel that fails its DRAM class by
that margin is answered from the stall decomposition, not by moving the bar.

An L2-aware floor would not rescue the percentage either. The request stream is
counted at L1TEX: the same kernel requests 256 MB and reads 176.5 MB from DRAM, so
31% of its demand never left the chip. The two-level form is
`max(dram_bytes / B_dram, requested_bytes / B_l2)`, one term per level, and `B_l2`
is measurable with the machinery already present -- the same copy fit run at
footprints below the `l2_bytes` the floor record already carries. With L2 read
bandwidth on this part between 1.5 and 2 times DRAM, which is a model and not a
measurement here, the request term is the smaller one and the max stays the DRAM
term, so the verdict stays near 55%. The L2-served fraction is a diagnostic that
says where the missing bandwidth went, not a larger denominator to divide by.

`slinoss/perf/traffic.py` reports the request stream, the DRAM stream, and the
ratio, per kernel. Nothing reads those columns for a verdict, and no floor moved:
changing a floor is the one edit that turns a failure into a pass, so it does not
happen as a side effect of adding a column.

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
- An audit that judged nothing fails. The class floor, the spill rule, the
  occupancy rule and the block floor are each a statement about a kernel the
  capture held, so a capture holding no kernel clears every one of them and the
  run exits zero having measured nothing. A conv audit did that: the compiled
  extension had not been built in that environment, the operator resolved to its
  reference, and thirteen torch kernels were reported as unjudged. Every rule
  held. `slinoss/perf/coverage.py` names the declared kernels each `(op, mode)`
  arm launches, and the run exits nonzero when the audit judged fewer than that.
  The count is of verdicts, not of captured kernels: a kernel the capture held
  and the audit could not judge is not covered.
- A kernel legitimately absent at a shape says so in the table. `Conditional`
  carries the shape property that makes the launch happen, and is judged when the
  capture holds it. A kernel no benchmarked arm launches at any shape is
  `Targeted`, names the driver that does launch it, and is a line in every report
  so the excuse is read every time the audit runs. Both hatches excuse a kernel
  from `unreachable` and from nothing else. An absence with no entry is a defect.
- A reference dispatch ends the run before any profiler starts. Each registry the
  operator selects through is asked what it resolves for the profiled device and
  dtype, and a `reference` answer is fatal. The signal is the registration guard
  every operator already has -- `_C.is_available()` for the conv, a CUDA check
  plus a DSL import for the rest -- so the check is one rule over six operators
  rather than a patch for the one that failed.
- A profiler that is not installed is an environment defect, named as one, before
  the workload is allocated. `ncu` and `nsys` are probed on `PATH` and then in the
  CUDA bin directories, and a miss raises with every path tried in the message.
  Never a broad `except` around the profiler: a skipped profiler is an audit with
  nothing to judge, which is the rule above with a clean exit status.
- A report stamps the tree it measured. A remote directory that accumulates files,
  or a `PYTHONPATH` naming a second checkout, measures code nobody edited and
  reads clean, which is the vacuous pass one layer below the coverage rule. The
  stamp is the resolved package directory, the driver's repository root, whether
  one contains the other, and the compiled extension's path and mtime. It is
  reported and not judged: calling a tree wrong needs a declared expected tree,
  and nothing here records one. A rule would need that declaration -- a revision
  or a content hash written by whatever publishes the tree, compared on load.
- A ceiling is measured in the same process, on the same device, at the same
  clocks as the kernel it is a denominator for, and at the kernel's own
  footprint. A ceiling carried over from another run drifts against the number
  it divides.
- A fitted denominator is reported with its residual. A fit quoted without one is
  an extrapolation presented as a measurement.
- A cost model carries how it was obtained, and a derived figure never shares a
  voice with a measured one. Four strengths, decreasing: an identity that closes
  against a reported metric, a counter read directly, a ratio fitted to
  observations, arithmetic over a datasheet. `chunk_vector_bwd` was priced at 42 k
  shared wavefronts per microsecond, a ratio fitted to three launches, and that
  figure entered this repo's rules in the same declarative voice as a counter. The
  limiter was LSU instruction issue, which closes as an identity to two decimal
  places at two clocks. Nobody can audit a number whose strength the number does
  not state.
- A change that beats its own prediction falsifies the model, and the microseconds
  are not the evidence to keep. Three consecutive arms on `chunk_vector_bwd`
  delivered about three times what the wavefront model priced them at. Each was
  banked and the overshoot was read as diminishing returns in the model's favour.
  All three had deleted LSU instructions, which that model did not count. An
  unexplained win is a defect report against the cost model, and it outranks the
  win.
- A resource absent from the metric list is unmeasured, not free. `--metrics`
  collects what somebody named, so a pipe nobody named reads as zero cost, and the
  gap then gets filled with arithmetic over the pipes that were named. Derive a
  floor only for a resource the capture counted, and name the ones it did not.
- A percentage of a ceiling is an efficiency, not a speed. Two configurations of
  one kernel are ranked by duration; the percentage says only how much of the
  bus each reached.
- A quotient of two clocks carries the offset between them. The per-iteration
  event sum over the host wall bracketing the loop is bounded by that wall in
  exact arithmetic, but the sum comes off the GPU timer and the wall off the host
  timer, so once device work fills the wall the quotient sits on the crystal
  offset: measured on an A6000, 4 to 13 ppm above unity over regions of 3.5 to
  10.7 s, and 78 to 155 ppm below it under half a second where host overhead
  dominates instead. A bound of exactly one rejects correct measurements; a bound
  that is a real check is ppm-scale, not percent-scale.
- A pass that ran and failed something reaches the machine-readable output, not
  only the prose. The spill pass fed the class audit and the audit overturned a
  verdict on it while the JSON carried neither the sectors nor the fact, so a
  harvest that read the file rather than the log read clean. This is the
  never-run-pass rule one layer out: an unserialized result is as invisible as an
  uncollected one.
