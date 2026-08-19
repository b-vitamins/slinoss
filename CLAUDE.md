# CLAUDE.md

`slinoss` is an oscillatory state-space sequence mixer. The core operator is a
chunked scan whose per-step homogeneous dynamics are an SO(3) rotation plus an
isotropic scale, applied by quaternion conjugation, with two-tap
first-order-hold forcing. The public mixer is `SLinOSSMixer`.

## Read before changing

- `docs/operator.md` -- the map, the tap chart, the numerical invariants, the
  tensor and shape contracts.
- `docs/kernels.md` -- the GEMM forms and the kernel engineering rules, including
  the roofline class every kernel is held to.
- `docs/structure.md` -- the five parts every operator has, dispatch, the tree.
- `docs/measurement.md` -- the optimization loop and the reporting rules.
- `docs/testing.md` -- what a test is for, ground truth, tolerances.

Those five hold the contracts. This file holds only the environment and the gate.

## Environment

Guix only. Never `pip`, `apt`, `npm -g`, or `cargo install`. Every dependency
belongs in `manifest.scm`.

```
guix shell -m manifest.scm -- python3 -m pytest -xvs
guix shell -m manifest.scm -- ruff format . && ruff check .
guix shell -m manifest.scm -- pyright
```

Use `python3`, never `python`.

Guix package name mappings: `requests` -> `python-requests`, `sklearn` ->
`python-scikit-learn`, `yaml` -> `python-pyyaml`.

## Gate before every commit

```
ruff format . && ruff check . && pytest -xvs
```

All three must be clean, and `pyright` too. Never commit unless asked. Never
amend another author's commit. Never commit `.env`, `secrets/`, or credentials.
Conventional commit format. Do not add `Co-Authored-By` trailers or any agent
attribution to commit messages.

## Style

- Technical, terse, imperative. No `we`, `you`, or `our`. No marketing language.
  ASCII only.
- Prose states the fact and stops. No intensifiers, no adjective piles, no
  narrative, no sentence that only restates the previous one. A comment explains
  the non-obvious constraint, not the code.
- Docstrings state shapes, dtypes, and invariants. Google style.
- Public API returns named types, never long positional tuples.
