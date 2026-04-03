from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slinoss._cute_runtime import ensure_cute_runtime_env  # noqa: E402


ensure_cute_runtime_env()
