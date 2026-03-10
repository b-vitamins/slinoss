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
