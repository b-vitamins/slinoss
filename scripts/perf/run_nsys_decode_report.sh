#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec nsys profile --trace=cuda,nvtx,osrt --sample=none \
  "$ROOT/scripts/guix-run" python3 \
  "$ROOT/scripts/perf/profile_decode.py" "$@"
