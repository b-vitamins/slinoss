"""Autotuning helpers for the CuTe ``v2x2ssd`` forward stack."""

from __future__ import annotations

from typing import Any

from .bench import benchmark_cuda_callable
from .db import (
    load_cute_tuning_db,
    lookup_tuning_record,
    save_cute_tuning_db,
    store_tuning_record,
    tuning_db_path,
)
from .hardware import current_hardware_fingerprint
from .types import (
    ChunkIncrementConfig,
    ChunkIncrementProblemKey,
    ChunkScanConfig,
    ChunkScanProblemKey,
    ForwardConfigBundle,
    ForwardProblemKey,
    HardwareFingerprint,
    StatePassingConfig,
    StatePassingProblemKey,
    TuneResult,
)

__all__ = (
    "ChunkIncrementConfig",
    "ChunkIncrementProblemKey",
    "ChunkScanConfig",
    "ChunkScanProblemKey",
    "ForwardConfigBundle",
    "ForwardProblemKey",
    "HardwareFingerprint",
    "StatePassingConfig",
    "StatePassingProblemKey",
    "TuneResult",
    "benchmark_cuda_callable",
    "current_hardware_fingerprint",
    "load_cute_tuning_db",
    "lookup_tuning_record",
    "save_cute_tuning_db",
    "store_tuning_record",
    "tuning_db_path",
    "autotune_enabled",
    "autotune_force_retune",
    "autotune_iterations",
    "autotune_mode",
    "autotune_warmup_iterations",
    "chunk_increment_candidate_configs",
    "chunk_increment_problem_key",
    "chunk_scan_candidate_configs",
    "chunk_scan_problem_key",
    "forward_bundle_candidates",
    "forward_problem_key",
    "state_passing_candidate_configs",
    "state_passing_problem_key",
)


def autotune_mode() -> str:
    from . import fwd

    return fwd.autotune_mode()


def autotune_enabled() -> bool:
    from . import fwd

    return fwd.autotune_enabled()


def autotune_force_retune() -> bool:
    from . import fwd

    return fwd.autotune_force_retune()


def autotune_warmup_iterations() -> int:
    from . import fwd

    return fwd.autotune_warmup_iterations()


def autotune_iterations() -> int:
    from . import fwd

    return fwd.autotune_iterations()


def chunk_increment_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[ChunkIncrementConfig, ...]:
    from . import fwd

    return fwd.chunk_increment_candidate_configs(*args, **kwargs)


def state_passing_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[StatePassingConfig, ...]:
    from . import fwd

    return fwd.state_passing_candidate_configs(*args, **kwargs)


def chunk_scan_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[ChunkScanConfig, ...]:
    from . import fwd

    return fwd.chunk_scan_candidate_configs(*args, **kwargs)


def forward_bundle_candidates(
    *args: Any, **kwargs: Any
) -> tuple[ForwardConfigBundle, ...]:
    from . import fwd

    return fwd.forward_bundle_candidates(*args, **kwargs)


def chunk_increment_problem_key(*args: Any, **kwargs: Any) -> ChunkIncrementProblemKey:
    from . import fwd

    return fwd.chunk_increment_problem_key(*args, **kwargs)


def state_passing_problem_key(*args: Any, **kwargs: Any) -> StatePassingProblemKey:
    from . import fwd

    return fwd.state_passing_problem_key(*args, **kwargs)


def chunk_scan_problem_key(*args: Any, **kwargs: Any) -> ChunkScanProblemKey:
    from . import fwd

    return fwd.chunk_scan_problem_key(*args, **kwargs)


def forward_problem_key(*args: Any, **kwargs: Any) -> ForwardProblemKey:
    from . import fwd

    return fwd.forward_problem_key(*args, **kwargs)
