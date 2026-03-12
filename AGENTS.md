# AGENTS.md

## Environment

This repo uses a Guix-managed development environment.

- Use `./scripts/guix-run <command> ...` for dependency-sensitive work.
- Prefer `./scripts/guix-run python3 ...` over bare `python` or `python3`.
- Prefer `./scripts/guix-run ruff ...`, `./scripts/guix-run pyright`, and
  `./scripts/guix-run pytest ...` when verifying behavior that depends on the
  repo's declared toolchain.
- `manifest.scm` is the source of truth for CuTe/CUTLASS, PyTorch, lint, and
  profiling dependencies.
- `pyproject.toml` and `requirements*.txt` are the pip-facing install surface.
  Keep them aligned with the repo's actual runtime/dev needs, but do not treat
  them as a replacement for the Guix environment.

## Commit Conventions

Use lightweight Conventional Commits for all new commits:

- `feat:` new capabilities or APIs
- `fix:` correctness fixes
- `refactor:` structure-preserving code changes
- `perf:` measurable performance work
- `test:` test-only changes
- `docs:` documentation-only changes
- `chore:` repo maintenance that does not affect behavior

Keep commit subjects short and specific. If a layout decision, kernel contract, or
benchmark change is important, make that visible in the subject line.

## Changelog Policy

Do not maintain a hand-written `CHANGELOG.md` during active development.

Rely on:

- disciplined commit messages
- milestone tags for important checkpoints

Add a real `CHANGELOG.md` only when one of these happens:

- the repo is prepared for a public release
- the paper artifact is being frozen
- the kernel/benchmark stack reaches a release candidate state

## CuTe Workflow Notes

- Do not add CuTe availability fallbacks or placeholder benchmark behavior until
  the Guix environment and launch path are in place.
- Once CuTe kernel work begins, keep the op contract fixed and make the package
  structure match the staged forward decomposition:
  `chunk_increment -> state_passing -> chunk_scan`.
- For benchmark and profiling scripts, prefer repo-local `scripts/` helpers over
  ad hoc commands once those helpers exist.

## Performance Workflow

Model-level throughput is the primary metric. Kernel-level wins matter only
insofar as they improve the real training/inference path through `layers/` and
`ops/`, especially the mixer.

- Treat `scripts/perf/bench_nextchar.py` as the primary workload bench.
- Treat `scripts/perf/profile_nextchar.py` as the primary workload profiler.
- Treat `scripts/perf/bench_v2x2ssd.py` and `scripts/perf/profile_v2x2ssd.py`
  as stage/kernel drills to use only after the workload harness identifies the
  current hot bucket.
- Treat `scripts/perf/compare_perf.py` as the default before/after reporting
  tool for workload runs.

### Required Perf Loop

When doing performance work, use this loop:

1. Run a workload bench and save JSON:
   `./scripts/guix-run python3 scripts/perf/bench_nextchar.py --backend both --suite training --json-out /tmp/run.json`
2. Compare against the previous run:
   `./scripts/guix-run python3 scripts/perf/compare_perf.py before.json after.json --backend cute --case default`
3. Identify the largest actionable bucket from the workload tree.
4. Only then drill into stage/kernel benches or traces for that bucket.
5. Re-run the workload bench and compare again before claiming a win.

Do not optimize from isolated kernel numbers alone if the workload tree says the
top offender lives elsewhere.

### Budget Naming

Perf labels should stay compact and stable:

- top level:
  - `step.*`
  - `forward.*`
  - `backward.*`
- scan backend:
  - `forward.v2x2ssd.*`
  - `backward.v2x2ssd.*`
- non-scan model buckets:
  - `forward.mixer.*`, `backward.mixer.*`
  - `forward.embed.*`, `backward.embed.*`
  - `forward.norms.*`, `backward.norms.*`
  - `forward.ffn.*`, `backward.ffn.*`
  - `forward.residual.*`, `backward.residual.*`
  - `forward.head.*`, `backward.head.*`
- aggregate non-scan split:
  - `forward.other.total`
  - `forward.other.unattributed`
  - `backward.other.total`
  - `backward.other.unattributed`

Do not reintroduce verbose taxonomies like `everything_else.mixer_non_scan.*`.

### Harness Discipline

- Keep the user-facing example in `examples/` minimal. Perf-only instrumentation
  belongs under `scripts/perf/`.
- Emit JSON from benches/profilers and keep payload schemas stable.
- If `*.other.unattributed` grows materially, fix instrumentation before doing
  more optimization work.
- Cache hit/miss behavior for compiled CuTe launchers is part of the perf
  surface; regressions there count as perf bugs.
- Include last-batch or reduced-batch cases when using the `training` suite so
  cache/dispatch behavior is exercised under realistic shape variation.
