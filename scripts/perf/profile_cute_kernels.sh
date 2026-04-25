#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/scripts/guix-run" python3 "$ROOT/scripts/perf/profile_cute_kernels.py" "$@"
