"""Runtime performance instrumentation for SLinOSS."""

from .budget import build_tree, derive_training_budget
from .compare import compare_budget_trees, flatten_tree_stats, rank_budget_deltas
from .memory import (
    EagerMemoryForensics,
    allocator_snapshot_metadata,
    current_memory_stats,
    peak_memory_stats,
    reset_peak_memory_stats,
)
from .runtime import (
    PerfRecorder,
    attach_module_timer,
    call_region,
    current_step,
    note_cache_event,
    record_region,
)
from .schema import (
    validate_decode_bench_payload,
    validate_decode_profile_payload,
    validate_training_bench_payload,
    validate_training_memory_payload,
    validate_training_profile_payload,
)

__all__ = [
    "build_tree",
    "derive_training_budget",
    "compare_budget_trees",
    "flatten_tree_stats",
    "rank_budget_deltas",
    "EagerMemoryForensics",
    "allocator_snapshot_metadata",
    "current_memory_stats",
    "peak_memory_stats",
    "reset_peak_memory_stats",
    "validate_decode_bench_payload",
    "validate_decode_profile_payload",
    "validate_training_bench_payload",
    "validate_training_memory_payload",
    "validate_training_profile_payload",
    "PerfRecorder",
    "attach_module_timer",
    "call_region",
    "current_step",
    "note_cache_event",
    "record_region",
]
