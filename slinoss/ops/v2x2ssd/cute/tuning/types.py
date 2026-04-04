"""Typed autotuning metadata for the CuTe ``v2x2ssd`` forward stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast
from typing import Any


@dataclass(frozen=True)
class HardwareFingerprint:
    """Stable hardware/toolchain fingerprint used for tuning records."""

    arch_tag: str
    device_name: str
    sm_major: int
    sm_minor: int
    multiprocessor_count: int
    shared_memory_per_block_optin: int
    total_memory_bytes: int
    cuda_runtime_version: str
    torch_cuda_version: str
    cutlass_version: str

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.arch_tag,
            self.device_name,
            self.sm_major,
            self.sm_minor,
            self.multiprocessor_count,
            self.shared_memory_per_block_optin,
            self.total_memory_bytes,
            self.cuda_runtime_version,
            self.torch_cuda_version,
            self.cutlass_version,
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkIncrementConfig:
    """Launch-shaping configuration for the chunk-increment forward kernel."""

    cta_tiler: tuple[int, int, int]
    num_stages: int

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (*self.cta_tiler, self.num_stages)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ChunkIncrementConfig:
        return cls(
            cta_tiler=cast(
                tuple[int, int, int],
                tuple(int(v) for v in record["cta_tiler"]),
            ),
            num_stages=int(record["num_stages"]),
        )


@dataclass(frozen=True)
class StatePassingConfig:
    """Launch-shaping configuration for the state-passing forward kernel."""

    num_threads: int
    vecs_per_thread: int

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (self.num_threads, self.vecs_per_thread)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> StatePassingConfig:
        return cls(
            num_threads=int(record["num_threads"]),
            vecs_per_thread=int(record["vecs_per_thread"]),
        )


@dataclass(frozen=True)
class ChunkScanConfig:
    """Launch-shaping configuration for the chunk-scan forward kernel."""

    m_block_size: int
    n_block_size: int
    num_threads: int

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (self.m_block_size, self.n_block_size, self.num_threads)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ChunkScanConfig:
        return cls(
            m_block_size=int(record["m_block_size"]),
            n_block_size=int(record["n_block_size"]),
            num_threads=int(record["num_threads"]),
        )


@dataclass(frozen=True)
class ForwardConfigBundle:
    """Combined launch policy for the staged ``v2x2ssd`` forward path."""

    chunk_increment: ChunkIncrementConfig
    state_passing: StatePassingConfig
    chunk_scan: ChunkScanConfig

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.chunk_increment.cache_key,
            self.state_passing.cache_key,
            self.chunk_scan.cache_key,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_increment": self.chunk_increment.to_record(),
            "state_passing": self.state_passing.to_record(),
            "chunk_scan": self.chunk_scan.to_record(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ForwardConfigBundle:
        return cls(
            chunk_increment=ChunkIncrementConfig.from_record(record["chunk_increment"]),
            state_passing=StatePassingConfig.from_record(record["state_passing"]),
            chunk_scan=ChunkScanConfig.from_record(record["chunk_scan"]),
        )


@dataclass(frozen=True)
class ChunkIncrementProblemKey:
    """Stable tuning key for chunk-increment configs."""

    tc_dtype_name: str
    P: int
    D: int
    chunk_size: int
    has_prev: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StatePassingProblemKey:
    """Stable tuning key for state-passing configs."""

    P: int
    D: int
    has_init: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkScanProblemKey:
    """Stable tuning key for chunk-scan configs."""

    tc_dtype_name: str
    output_dtype_name: str
    P: int
    D: int
    chunk_size: int
    has_prev: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForwardProblemKey:
    """Stable tuning key for the combined forward host wrapper."""

    tc_dtype_name: str
    output_dtype_name: str
    P: int
    D: int
    chunk_size: int
    has_prev: bool
    has_init: bool
    n_chunks_bucket: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuneResult:
    """Benchmark result for one candidate configuration."""

    latency_ms: float
    candidate_count: int
    config_record: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
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
]
