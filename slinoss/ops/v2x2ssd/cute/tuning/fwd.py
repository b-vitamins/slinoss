"""Forward-kernel autotuning config spaces and environment controls."""

from __future__ import annotations

import os
from typing import Iterable

import cutlass
import torch

from ..kernels.fwd.chunk_increment import ChunkIncrementFwdAmpere
from ..kernels.fwd.chunk_scan import ChunkScanFwdAmpere
from .types import (
    ChunkIncrementConfig,
    ChunkIncrementProblemKey,
    ChunkScanConfig,
    ChunkScanProblemKey,
    ForwardConfigBundle,
    ForwardProblemKey,
    StatePassingConfig,
    StatePassingProblemKey,
)


def autotune_mode() -> str:
    """Return the configured autotune mode."""

    mode = os.environ.get("SLINOSS_CUTE_AUTOTUNE", "1").strip().lower()
    if mode in {"0", "false", "off", "disable", "disabled"}:
        return "0"
    if mode in {"force", "retune"}:
        return "force"
    return "1"


def autotune_enabled() -> bool:
    return autotune_mode() != "0"


def autotune_force_retune() -> bool:
    return autotune_mode() == "force"


def autotune_warmup_iterations() -> int:
    return max(1, int(os.environ.get("SLINOSS_CUTE_AUTOTUNE_WARMUP", "5")))


def autotune_iterations() -> int:
    return max(1, int(os.environ.get("SLINOSS_CUTE_AUTOTUNE_ITERS", "15")))


def _chunk_increment_candidate_space(
    *,
    P: int,
    D: int,
    chunk_size: int,
) -> Iterable[ChunkIncrementConfig]:
    for bM in (32, 64):
        if bM > int(P) or int(P) % int(bM) != 0:
            continue
        for bN in (64, 96, 128):
            for bK in (32, 64):
                if int(chunk_size) % int(bK) != 0:
                    continue
                if bK == int(chunk_size) and bN == 64:
                    num_stage_values = (1,)
                else:
                    num_stage_values = (2, 3)
                for num_stages in num_stage_values:
                    try:
                        ChunkIncrementFwdAmpere(
                            cutlass.Float16,
                            chunk_size=int(chunk_size),
                            cta_tiler=(int(bM), int(bN), int(bK)),
                            num_stages=int(num_stages),
                        )
                    except Exception:
                        continue
                    yield ChunkIncrementConfig(
                        cta_tiler=(int(bM), int(bN), int(bK)),
                        num_stages=int(num_stages),
                    )


def chunk_increment_candidate_configs(
    *,
    P: int,
    D: int,
    chunk_size: int,
) -> tuple[ChunkIncrementConfig, ...]:
    """Return the curated chunk-increment config family for this problem."""

    seen: set[tuple[object, ...]] = set()
    configs: list[ChunkIncrementConfig] = []
    for config in _chunk_increment_candidate_space(P=P, D=D, chunk_size=chunk_size):
        if config.cache_key in seen:
            continue
        seen.add(config.cache_key)
        configs.append(config)
    return tuple(configs)


_STATE_PASSING_BASE_CANDIDATES: tuple[StatePassingConfig, ...] = (
    StatePassingConfig(64, 8),
    StatePassingConfig(64, 16),
    StatePassingConfig(128, 4),
    StatePassingConfig(128, 8),
    StatePassingConfig(128, 16),
    StatePassingConfig(256, 4),
    StatePassingConfig(256, 8),
)


def state_passing_candidate_configs(
    *,
    P: int,
    D: int,
    has_init: bool,
) -> tuple[StatePassingConfig, ...]:
    """Return the curated state-passing config family for this problem."""
    from ..kernels.fwd import _make_state_passing_cfg

    state_elem_count = int(P) * int(D)
    configs: list[StatePassingConfig] = []
    for config in _STATE_PASSING_BASE_CANDIDATES:
        tile_elems = int(config.num_threads) * (2 * int(config.vecs_per_thread))
        if tile_elems > max(state_elem_count, 512) * 2:
            continue
        _ = _make_state_passing_cfg(
            num_threads=config.num_threads,
            vecs_per_thread=config.vecs_per_thread,
            copy_bits_in=128,
            copy_bits_out=128,
            copy_bits_state_in=32,
            copy_bits_state_out=32,
            has_init=has_init,
        )
        configs.append(config)
    return tuple(configs)


def chunk_scan_candidate_configs(
    *,
    P: int,
    D: int,
    chunk_size: int,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int | None,
) -> tuple[ChunkScanConfig, ...]:
    """Return the curated chunk-scan config family for this problem."""
    from ..kernels.fwd import _iter_chunk_scan_n_block_candidates

    cutlass_tc_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[tc_dtype]
    cutlass_output_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[output_dtype]

    family_candidates = tuple(
        family
        for family in ChunkScanFwdAmpere._SUPPORTED_TILE_FAMILIES
        if int(family[0]) <= int(chunk_size)
    )
    n_candidate_seed_values = [min(128, int(chunk_size)), 64, 32, 16]
    preferred_n_values = tuple(
        dict.fromkeys(
            value for value in n_candidate_seed_values if value <= int(chunk_size)
        )
    )

    configs: list[ChunkScanConfig] = []
    seen: set[tuple[object, ...]] = set()
    for m_block_size, num_threads in family_candidates:
        for requested_n in preferred_n_values:
            for n_block_size in _iter_chunk_scan_n_block_candidates(
                int(chunk_size), int(requested_n)
            ):
                config = ChunkScanConfig(
                    m_block_size=int(m_block_size),
                    n_block_size=int(n_block_size),
                    num_threads=int(num_threads),
                )
                if config.cache_key in seen:
                    continue
                kernel = ChunkScanFwdAmpere(
                    D=int(D),
                    P=int(P),
                    L=int(chunk_size),
                    m_block_size=config.m_block_size,
                    n_block_size=config.n_block_size,
                    num_threads=config.num_threads,
                )
                if device_index is not None and not kernel.can_implement(
                    cutlass_tc_dtype,
                    cutlass_output_dtype,
                    device_index=device_index,
                ):
                    continue
                seen.add(config.cache_key)
                configs.append(config)
    return tuple(configs)


def forward_bundle_candidates(
    *,
    P: int,
    D: int,
    chunk_size: int,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int | None,
    has_init: bool,
    chunk_increment_limit: int | None = None,
    state_passing_limit: int | None = None,
    chunk_scan_limit: int | None = None,
) -> tuple[ForwardConfigBundle, ...]:
    """Return candidate forward bundles from the curated stage spaces."""

    chunk_increment_candidates = chunk_increment_candidate_configs(
        P=P,
        D=D,
        chunk_size=chunk_size,
    )
    state_passing_candidates = state_passing_candidate_configs(
        P=P,
        D=D,
        has_init=has_init,
    )
    chunk_scan_candidates = chunk_scan_candidate_configs(
        P=P,
        D=D,
        chunk_size=chunk_size,
        tc_dtype=tc_dtype,
        output_dtype=output_dtype,
        device_index=device_index,
    )
    if chunk_increment_limit is not None:
        chunk_increment_candidates = chunk_increment_candidates[
            : int(chunk_increment_limit)
        ]
    if state_passing_limit is not None:
        state_passing_candidates = state_passing_candidates[: int(state_passing_limit)]
    if chunk_scan_limit is not None:
        chunk_scan_candidates = chunk_scan_candidates[: int(chunk_scan_limit)]

    bundles: list[ForwardConfigBundle] = []
    for chunk_increment_config in chunk_increment_candidates:
        for state_passing_config in state_passing_candidates:
            for chunk_scan_config in chunk_scan_candidates:
                bundles.append(
                    ForwardConfigBundle(
                        chunk_increment=chunk_increment_config,
                        state_passing=state_passing_config,
                        chunk_scan=chunk_scan_config,
                    )
                )
    return tuple(bundles)


def chunk_increment_problem_key(
    *,
    tc_dtype: torch.dtype,
    P: int,
    D: int,
    chunk_size: int,
    has_prev: bool,
) -> ChunkIncrementProblemKey:
    return ChunkIncrementProblemKey(
        tc_dtype_name=str(tc_dtype).replace("torch.", ""),
        P=int(P),
        D=int(D),
        chunk_size=int(chunk_size),
        has_prev=bool(has_prev),
    )


def state_passing_problem_key(
    *,
    P: int,
    D: int,
    has_init: bool,
) -> StatePassingProblemKey:
    return StatePassingProblemKey(P=int(P), D=int(D), has_init=bool(has_init))


def chunk_scan_problem_key(
    *,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    P: int,
    D: int,
    chunk_size: int,
    has_prev: bool,
) -> ChunkScanProblemKey:
    return ChunkScanProblemKey(
        tc_dtype_name=str(tc_dtype).replace("torch.", ""),
        output_dtype_name=str(output_dtype).replace("torch.", ""),
        P=int(P),
        D=int(D),
        chunk_size=int(chunk_size),
        has_prev=bool(has_prev),
    )


def _n_chunks_bucket(n_chunks: int) -> int:
    bucket = 1
    while bucket < int(n_chunks):
        bucket <<= 1
    return bucket


def forward_problem_key(
    *,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    P: int,
    D: int,
    chunk_size: int,
    has_prev: bool,
    has_init: bool,
    n_chunks: int,
) -> ForwardProblemKey:
    return ForwardProblemKey(
        tc_dtype_name=str(tc_dtype).replace("torch.", ""),
        output_dtype_name=str(output_dtype).replace("torch.", ""),
        P=int(P),
        D=int(D),
        chunk_size=int(chunk_size),
        has_prev=bool(has_prev),
        has_init=bool(has_init),
        n_chunks_bucket=_n_chunks_bucket(int(n_chunks)),
    )


__all__ = [
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
]
