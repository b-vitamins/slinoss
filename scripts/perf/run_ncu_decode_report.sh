#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec ncu --set full "$ROOT/scripts/guix-run" python3 \
  "$ROOT/scripts/perf/profile_decode.py" "$@"
