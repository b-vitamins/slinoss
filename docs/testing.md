# Testing

100% coverage on public APIs. Coverage is the floor, not the goal: every test
names the thing that breaks without it. An indiscriminate parametrize product is
a defect.

## What a test is for

- Sweep an axis; do not cross it. An axis that does not interact with another is
  swept once: full sweep on the interacting axis, one representative case per
  independent axis, non-interaction stated in the docstring.
- A rule shared by several operators is tested once, against a fixture the test
  owns.
- A pure performance change that alters no contract needs no new test. The
  existing parity tests protect it and the measurement is its evidence.
- Every `raise` in a public path has a test that triggers it.

## Ground truth

- Write the failing test before fixing a bug. Always.
- Correctness ground truth is float64 autograd through the reference, not a
  hand-derived VJP. A hand-derived reference shares its derivation with the
  kernel, so a derivation error passes silently.
- `gradcheck` in float64 on every gradient. No quantity is exempt.
- The forward and the backward must be connected in at least one test: compute
  the output with the fast path, backpropagate through it, and compare against
  the reference end to end. Testing a backward against a surrogate forward hides
  any disagreement between the surrogate and the real kernel.
- Never derive an intermediate from `randn` when the real pipeline can produce
  it. A fabricated chunk-start state does not test chunk composition.
- Compare bitwise wherever the kernel does one add or one copy. A tolerance
  there would hide a wrong index.

## Shapes

Swept, not fixed: sequence length not a multiple of the chunk, single chunk,
three or more chunks, `B = 1`, `H = 1`, smallest legal `N`, smallest legal `P`,
the grouped and ungrouped `B`/`C` cases, and the streaming split.

## Tolerances and gates

- Tolerances must be tight enough to fail. A tolerance loose enough to admit any
  output is not a test. Justify every tolerance above `1e-2` in a comment, and
  read every tolerance and every measured error quoted beside one off an actual
  run.
- Never relax a gate. Do not xfail, skip, or loosen an existing test to make a
  change pass. A failing existing test is the change's bug.
- Missing CUDA or missing CuTe skips cleanly at module level. It never errors at
  collection.
