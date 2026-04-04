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
    from .fwd import autotune_mode as _autotune_mode

    return _autotune_mode()


def autotune_enabled() -> bool:
    from .fwd import autotune_enabled as _autotune_enabled

    return _autotune_enabled()


def autotune_force_retune() -> bool:
    from .fwd import autotune_force_retune as _autotune_force_retune

    return _autotune_force_retune()


def autotune_warmup_iterations() -> int:
    from .fwd import autotune_warmup_iterations as _autotune_warmup_iterations

    return _autotune_warmup_iterations()


def autotune_iterations() -> int:
    from .fwd import autotune_iterations as _autotune_iterations

    return _autotune_iterations()


def chunk_increment_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[ChunkIncrementConfig, ...]:
    from .fwd import (
        chunk_increment_candidate_configs as _chunk_increment_candidate_configs,
    )

    return _chunk_increment_candidate_configs(*args, **kwargs)


def state_passing_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[StatePassingConfig, ...]:
    from .fwd import state_passing_candidate_configs as _state_passing_candidate_configs

    return _state_passing_candidate_configs(*args, **kwargs)


def chunk_scan_candidate_configs(
    *args: Any, **kwargs: Any
) -> tuple[ChunkScanConfig, ...]:
    from .fwd import chunk_scan_candidate_configs as _chunk_scan_candidate_configs

    return _chunk_scan_candidate_configs(*args, **kwargs)


def forward_bundle_candidates(
    *args: Any, **kwargs: Any
) -> tuple[ForwardConfigBundle, ...]:
    from .fwd import forward_bundle_candidates as _forward_bundle_candidates

    return _forward_bundle_candidates(*args, **kwargs)


def chunk_increment_problem_key(*args: Any, **kwargs: Any) -> ChunkIncrementProblemKey:
    from .fwd import chunk_increment_problem_key as _chunk_increment_problem_key

    return _chunk_increment_problem_key(*args, **kwargs)


def state_passing_problem_key(*args: Any, **kwargs: Any) -> StatePassingProblemKey:
    from .fwd import state_passing_problem_key as _state_passing_problem_key

    return _state_passing_problem_key(*args, **kwargs)


def chunk_scan_problem_key(*args: Any, **kwargs: Any) -> ChunkScanProblemKey:
    from .fwd import chunk_scan_problem_key as _chunk_scan_problem_key

    return _chunk_scan_problem_key(*args, **kwargs)


def forward_problem_key(*args: Any, **kwargs: Any) -> ForwardProblemKey:
    from .fwd import forward_problem_key as _forward_problem_key

    return _forward_problem_key(*args, **kwargs)
