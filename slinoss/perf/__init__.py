"""Runtime performance instrumentation for SLinOSS."""

from .budget import build_tree, derive_nextchar_budget
from .compare import compare_budget_trees, flatten_tree_stats, rank_budget_deltas
from .runtime import (
    PerfRecorder,
    attach_module_timer,
    call_region,
    current_step,
    note_cache_event,
    record_region,
)
from .schema import validate_nextchar_bench_payload, validate_nextchar_profile_payload

__all__ = [
    "build_tree",
    "derive_nextchar_budget",
    "compare_budget_trees",
    "flatten_tree_stats",
    "rank_budget_deltas",
    "validate_nextchar_bench_payload",
    "validate_nextchar_profile_payload",
    "PerfRecorder",
    "attach_module_timer",
    "call_region",
    "current_step",
    "note_cache_event",
    "record_region",
]
