"""CuTe forward kernels for the ``v2x2ssd`` staged pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import cutlass
import torch
from cuda.bindings import driver as cuda
import cutlass.cute as cute
from typing import cast

from slinoss._cute_runtime import make_runtime_tensor_spec_view
from slinoss.ops.v2x2ssd.cute.tuning.bench import benchmark_cuda_callable
from slinoss.ops.v2x2ssd.cute.tuning.db import lookup_tuning_record, store_tuning_record
from slinoss.ops.v2x2ssd.cute.tuning.fwd import (
    autotune_enabled,
    autotune_force_retune,
    autotune_iterations,
    autotune_warmup_iterations,
    chunk_increment_candidate_configs,
    chunk_increment_problem_key,
    chunk_scan_candidate_configs,
    chunk_scan_problem_key,
    forward_bundle_candidates,
    forward_problem_key,
    state_passing_candidate_configs,
    state_passing_problem_key,
)
from slinoss.ops.v2x2ssd.cute.tuning.hardware import current_hardware_fingerprint
from slinoss.ops.v2x2ssd.cute.tuning.types import (
    ChunkIncrementConfig,
    ChunkScanConfig,
    ForwardConfigBundle,
    StatePassingConfig,
)
from slinoss.perf import note_cache_event

from .chunk_increment import ChunkIncrementFwdAmpere
from .chunk_scan import ChunkScanFwdAmpere
from .common import (
    _assumed_align,
    _choose_copy_bits_for_linear_tiles,
    _compile_env_stream_placeholder,
    _ensure_min_alignment,
    _guard_prev_time_base,
    _make_fake_tensor_arg,
    _make_fake_tensor_spec_arg,
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
    _torch_to_cutlass_dtype,
)
from .state_passing import StatePassingFwdAmpere


_CHUNK_INCREMENT_CACHE: dict[tuple, object] = {}
_STATE_PASSING_CACHE: dict[tuple, object] = {}
_CHUNK_SCAN_CACHE: dict[tuple, object] = {}
_FWD_HOST_CACHE: dict[tuple, object] = {}
_FWD_WORKSPACE_CACHE: dict[tuple, torch.Tensor] = {}
_FWD_NO_GRAD_INTERMEDIATE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_INITIAL_STATE_CACHE: dict[tuple, torch.Tensor] = {}
_DUMMY_FINAL_STATE_CACHE: dict[tuple, torch.Tensor] = {}
_FWD_WORKSPACE_CACHE_LIMIT = 4
_FWD_NO_GRAD_INTERMEDIATE_CACHE_LIMIT = 4
_ZERO_PREV_CACHE_LIMIT = 8
_ZERO_INITIAL_STATE_CACHE_LIMIT = 8
_DUMMY_FINAL_STATE_CACHE_LIMIT = 8


@dataclass(frozen=True)
class ForwardOutputs:
    output: torch.Tensor
    final_state: torch.Tensor | None
    chunk_multiplier: torch.Tensor
    chunk_starts: torch.Tensor


@dataclass(frozen=True)
class ForwardRuntimeArtifacts:
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, ...]
    config_bundle: ForwardConfigBundle
    outputs: ForwardOutputs


@dataclass(frozen=True)
class ForwardCompileArtifacts:
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, ...]
    config_bundle: ForwardConfigBundle
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]


@dataclass(frozen=True)
class ForwardInputInfo:
    batch_size: int
    heads: int
    time_steps: int
    P: int
    D: int
    chunk_size: int
    n_chunks: int
    padded_time: int
    tc_dtype: torch.dtype
    device_index: int
    resolved_m_block: int
    resolved_n_block: int
    resolved_scan_num_threads: int


@dataclass(frozen=True)
class ChunkIncrementOutputs:
    increment_chunk: torch.Tensor
    chunk_multiplier_storage: torch.Tensor


@dataclass(frozen=True)
class ChunkIncrementRuntimeArtifacts:
    problem_shape: tuple[int, ...]
    config: ChunkIncrementConfig
    compile_args: tuple[object, ...]
    runtime_args: tuple[torch.Tensor, ...]
    cache_key: tuple
    outputs: ChunkIncrementOutputs


@dataclass(frozen=True)
class ChunkIncrementCompileArtifacts:
    problem_shape: tuple[int, ...]
    config: ChunkIncrementConfig
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class PreparedChunkIncrementLaunch:
    compiled: object
    runtime_args: tuple[torch.Tensor, ...]
    outputs: ChunkIncrementOutputs


@dataclass(frozen=True)
class StatePassingOutputs:
    chunk_starts: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class StatePassingRuntimeArtifacts:
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, int, int, int, int, int, bool]
    config: StatePassingConfig
    compile_args: tuple[object, ...]
    runtime_args: tuple[torch.Tensor, ...]
    cache_key: tuple
    outputs: StatePassingOutputs


@dataclass(frozen=True)
class StatePassingCompileArtifacts:
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, int, int, int, int, int, bool]
    config: StatePassingConfig
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class PreparedStatePassingLaunch:
    compiled: object
    runtime_args: tuple[torch.Tensor, ...]
    outputs: StatePassingOutputs


@dataclass(frozen=True)
class ChunkScanOutputs:
    output_chunk: torch.Tensor
    output_view: torch.Tensor


@dataclass(frozen=True)
class ChunkScanRuntimeArtifacts:
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, int, int]
    config: ChunkScanConfig
    compile_args: tuple[object, ...]
    runtime_args: tuple[torch.Tensor, ...]
    cache_key: tuple
    outputs: ChunkScanOutputs


@dataclass(frozen=True)
class ChunkScanCompileArtifacts:
    problem_shape: tuple[int, ...]
    launch_cfg: tuple[int, int, int]
    config: ChunkScanConfig
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    cache_key: tuple


@dataclass(frozen=True)
class PreparedChunkScanLaunch:
    compiled: object
    runtime_args: tuple[torch.Tensor, ...]
    outputs: ChunkScanOutputs


def _cache_set(cache: dict, key: tuple, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
    """Extend raw-pointer tensor lifetimes through the current CUDA stream."""
    stream = None
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or tensor.device.type != "cuda":
            continue
        ident = id(tensor)
        if ident in seen:
            continue
        if stream is None:
            stream = torch.cuda.current_stream(device=tensor.device)
        tensor.record_stream(stream)
        seen.add(ident)


def _get_zero_prev_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        dtype,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_PREV_CACHE.get(key)
    if cached is None:
        note_cache_event("cute.v2x2ssd.fwd.zero_prev", hit=False)
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
    else:
        note_cache_event("cute.v2x2ssd.fwd.zero_prev", hit=True)
    return cached


def _get_zero_initial_state(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _ZERO_INITIAL_STATE_CACHE.get(key)
    if cached is None:
        note_cache_event("cute.v2x2ssd.fwd.zero_initial_state", hit=False)
        cached = torch.zeros(
            (batch_size, heads, P, D),
            device=device,
            dtype=torch.float32,
        )
        _cache_set(
            _ZERO_INITIAL_STATE_CACHE,
            key,
            cached,
            limit=_ZERO_INITIAL_STATE_CACHE_LIMIT,
        )
    else:
        note_cache_event("cute.v2x2ssd.fwd.zero_initial_state", hit=True)
    return cached


def _get_dummy_final_state(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    P: int,
    D: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(P),
        int(D),
    )
    cached = _DUMMY_FINAL_STATE_CACHE.get(key)
    if cached is None:
        note_cache_event("cute.v2x2ssd.fwd.dummy_final_state", hit=False)
        cached = torch.empty(
            (batch_size, heads, P, D),
            device=device,
            dtype=torch.float32,
        )
        _cache_set(
            _DUMMY_FINAL_STATE_CACHE,
            key,
            cached,
            limit=_DUMMY_FINAL_STATE_CACHE_LIMIT,
        )
    else:
        note_cache_event("cute.v2x2ssd.fwd.dummy_final_state", hit=True)
    return cached


def _get_fwd_workspace(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    n_chunks: int,
    P: int,
    D: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(n_chunks),
        int(P),
        int(D),
    )
    cached = _FWD_WORKSPACE_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.v2x2ssd.fwd.workspace", hit=True)
        return cached
    note_cache_event("cute.v2x2ssd.fwd.workspace", hit=False)
    cached = torch.empty(
        (batch_size * heads * n_chunks, P, D), device=device, dtype=torch.float32
    )
    _cache_set(_FWD_WORKSPACE_CACHE, key, cached, limit=_FWD_WORKSPACE_CACHE_LIMIT)
    return cached


def _get_no_grad_intermediates(
    *,
    device: torch.device,
    batch_size: int,
    heads: int,
    n_chunks: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch_size),
        int(heads),
        int(n_chunks),
        int(P),
        int(D),
    )
    cached = _FWD_NO_GRAD_INTERMEDIATE_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.v2x2ssd.fwd.no_grad_intermediates", hit=True)
        return cached
    note_cache_event("cute.v2x2ssd.fwd.no_grad_intermediates", hit=False)
    cached = (
        torch.empty(
            (batch_size * heads * n_chunks, 2), device=device, dtype=torch.float32
        ),
        torch.empty(
            (batch_size, heads, n_chunks, P, D), device=device, dtype=torch.float32
        ),
    )
    _cache_set(
        _FWD_NO_GRAD_INTERMEDIATE_CACHE,
        key,
        cached,
        limit=_FWD_NO_GRAD_INTERMEDIATE_CACHE_LIMIT,
    )
    return cached


def _resolve_chunk_scan_n_block_size(chunk_size: int, requested: int) -> int:
    if requested <= 0:
        raise ValueError("n_block_size must be positive.")
    if requested % 16 != 0:
        raise ValueError("n_block_size must be a multiple of 16.")
    if chunk_size % requested == 0:
        return requested
    limit = min(chunk_size, requested)
    candidate = limit - (limit % 16)
    while candidate >= 16:
        if chunk_size % candidate == 0:
            return candidate
        candidate -= 16
    raise ValueError(
        f"No valid n_block_size multiple of 16 divides chunk_size={chunk_size} "
        f"for requested n_block_size={requested}."
    )


def _iter_chunk_scan_n_block_candidates(chunk_size: int, requested: int):
    candidate = _resolve_chunk_scan_n_block_size(chunk_size, requested)
    while candidate >= 16:
        if chunk_size % candidate == 0:
            yield candidate
        candidate -= 16


def _chunk_scan_supported_tile_families(
    chunk_size: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (m_block_size, num_threads)
        for m_block_size, num_threads in ChunkScanFwdAmpere._SUPPORTED_TILE_FAMILIES
        if m_block_size <= chunk_size
    )


def _chunk_scan_device_label(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def _resolve_chunk_scan_launch_cfg(
    *,
    D: int,
    P: int,
    chunk_size: int,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    requested_m_block_size: int | None,
    requested_n_block_size: int,
    requested_num_threads: int,
) -> tuple[int, int, int]:
    supported_families = _chunk_scan_supported_tile_families(chunk_size)
    if not supported_families:
        raise ValueError(
            f"chunk_size={chunk_size} is too small for the CuTe chunk_scan tile families; "
            "supported chunk sizes must be at least 16."
        )

    requested_num_threads = int(requested_num_threads)
    if requested_m_block_size is not None:
        resolved_m_block_size = int(requested_m_block_size)
        family = next(
            (
                (m_block_size, num_threads)
                for m_block_size, num_threads in supported_families
                if m_block_size == resolved_m_block_size
            ),
            None,
        )
        if family is None:
            supported_m = ", ".join(
                str(m_block_size) for m_block_size, _ in supported_families
            )
            raise ValueError(
                f"Unsupported chunk_scan m_block_size={resolved_m_block_size} for chunk_size={chunk_size}. "
                f"Supported tile families use m_block_size in {{{supported_m}}}."
            )

        expected_num_threads = int(family[1])
        if requested_num_threads not in (128, expected_num_threads):
            raise ValueError(
                f"chunk_scan m_block_size={resolved_m_block_size} requires "
                f"num_threads={expected_num_threads}; got {requested_num_threads}."
            )
        candidate_families = (family,)
    else:
        if requested_num_threads == 128:
            candidate_families = supported_families
        else:
            candidate_families = tuple(
                family
                for family in supported_families
                if int(family[1]) == requested_num_threads
            )
            if not candidate_families:
                supported_threads = ", ".join(
                    str(num_threads) for _, num_threads in supported_families
                )
                raise ValueError(
                    f"Unsupported chunk_scan num_threads={requested_num_threads} for "
                    f"chunk_size={chunk_size}. Supported tile families use num_threads in "
                    f"{{{supported_threads}}}."
                )

    cutlass_tc_dtype = _torch_to_cutlass_dtype(tc_dtype)
    cutlass_out_dtype = _torch_to_cutlass_dtype(output_dtype)
    attempts: list[str] = []
    for m_block_size, num_threads in candidate_families:
        for n_block_size in _iter_chunk_scan_n_block_candidates(
            chunk_size, requested_n_block_size
        ):
            kernel = ChunkScanFwdAmpere(
                D=D,
                P=P,
                L=chunk_size,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                num_threads=num_threads,
            )
            info = kernel.support_info(
                cutlass_tc_dtype,
                cutlass_out_dtype,
                device_index=device_index,
            )
            if info.supported:
                return int(m_block_size), int(n_block_size), int(num_threads)
            attempts.append(
                f"(m={m_block_size}, n={n_block_size}, threads={num_threads}) "
                f"needs {info.required_smem_bytes}B > {info.smem_capacity_bytes}B"
            )

    device_label = _chunk_scan_device_label(device_index)
    attempt_summary = "; ".join(attempts[:4])
    if len(attempts) > 4:
        attempt_summary += f"; ... {len(attempts) - 4} more"
    raise ValueError(
        f"No supported chunk_scan tile family fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}). Tried: {attempt_summary}"
    )


def _make_row_major_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    stride = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = running
        running *= int(shape[i])
    return tuple(stride)


def _make_tensor_spec(
    shape: tuple[int, ...],
    *,
    stride: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = _make_row_major_stride(shape)
    else:
        stride = tuple(int(step) for step in stride)
    return shape, stride


def _make_static_tensor_spec_view(
    tensor: cute.Tensor,
    tensor_spec: tuple[tuple[int, ...], tuple[int, ...]],
) -> cute.Tensor:
    shape, stride = tensor_spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_runtime_tensor_views_from_specs(
    *tensor_specs: tuple[torch.Tensor, tuple[tuple[int, ...], tuple[int, ...]]],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        make_runtime_tensor_spec_view(tensor, spec) for tensor, spec in tensor_specs
    )


def _make_tvm_ffi_runtime_and_compile_args_from_specs(
    *tensor_specs: tuple[
        torch.Tensor,
        torch.dtype,
        tuple[tuple[int, ...], tuple[int, ...]],
    ],
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], tuple[object, ...]]:
    runtime_args = _make_runtime_tensor_views_from_specs(
        *((tensor, spec) for tensor, _dtype, spec in tensor_specs)
    )
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
    _, compile_args = _make_tvm_ffi_compile_args_from_specs(
        *(
            (dtype, spec, align)
            for (_tensor, dtype, spec), align in zip(
                tensor_specs, alignments, strict=True
            )
        )
    )
    return runtime_args, alignments, compile_args


def _make_tvm_ffi_compile_args(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    alignments: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[object, ...]]:
    resolved_alignments = (
        tuple(_assumed_align(tensor) for tensor in runtime_args)
        if alignments is None
        else tuple(int(align) for align in alignments)
    )
    compile_args = tuple(
        _make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, resolved_alignments, strict=True)
    ) + (_compile_env_stream_placeholder(),)
    return resolved_alignments, compile_args


def _make_tvm_ffi_compile_args_from_specs(
    *tensor_specs: tuple[torch.dtype, tuple[tuple[int, ...], tuple[int, ...]], int],
) -> tuple[tuple[int, ...], tuple[object, ...]]:
    alignments = tuple(int(align) for _, _, align in tensor_specs)
    compile_args = tuple(
        _make_fake_tensor_spec_arg(
            dtype=dtype,
            shape=spec[0],
            stride=spec[1],
            align=align,
        )
        for dtype, spec, align in tensor_specs
    ) + (_compile_env_stream_placeholder(),)
    return alignments, compile_args


def _chunk_increment_tensor_specs(
    problem_shape: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]:
    batch_size, heads, padded_time, P, D, n_chunks, _chunk_size = problem_shape
    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks
    return (
        _make_tensor_spec(
            (P, padded_time, batch_head_count),
            stride=(1, P, padded_time * P),
        ),
        _make_tensor_spec(
            (D, padded_time, batch_head_count),
            stride=(1, D, padded_time * D),
        ),
        _make_tensor_spec(
            (2, padded_time, batch_head_count),
            stride=(1, 2, padded_time * 2),
        ),
        _make_tensor_spec(
            (2, padded_time, batch_head_count),
            stride=(1, 4, padded_time * 4),
        ),
        _make_tensor_spec((P, batch_head_count), stride=(1, P)),
        _make_tensor_spec((D, batch_head_count), stride=(1, D)),
        _make_tensor_spec((P, D, batch_head_chunk_count), stride=(D, 1, P * D)),
        _make_tensor_spec((2, batch_head_chunk_count), stride=(1, 2)),
    )


def _resolve_chunk_increment_problem_shape(
    *,
    U: torch.Tensor,
    B: torch.Tensor,
    padded_time: int,
    chunk_size: int,
) -> tuple[int, ...]:
    batch_size, heads, _time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    resolved_chunk_size = int(chunk_size)
    n_chunks = int(padded_time) // resolved_chunk_size
    return (
        batch_size,
        heads,
        int(padded_time),
        P,
        D,
        n_chunks,
        resolved_chunk_size,
    )


def _make_chunk_increment_runtime_artifacts(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    U_prev_state: torch.Tensor,
    B_prev_state: torch.Tensor,
    inc_chunk: torch.Tensor,
    chunk_multiplier_storage: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    has_prev: bool,
    config: ChunkIncrementConfig | None = None,
) -> ChunkIncrementRuntimeArtifacts:
    problem_shape = _resolve_chunk_increment_problem_shape(
        U=U,
        B=B,
        padded_time=int(U_tc.shape[2]),
        chunk_size=chunk_size,
    )
    resolved_config = (
        _resolve_default_chunk_increment_config(
            D=int(B.shape[-1]),
            chunk_size=int(chunk_size),
        )
        if config is None
        else _normalize_chunk_increment_config(
            chunk_size=int(chunk_size),
            config=config,
        )
    )
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        u_prev_spec,
        b_prev_spec,
        inc_spec,
        m_chunk_spec,
    ) = _chunk_increment_tensor_specs(problem_shape)
    runtime_args, alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (U_tc, tc_dtype, u_spec),
            (B_tc, tc_dtype, b_spec),
            (M_f, torch.float32, m_spec),
            (K_f, torch.float32, k_spec),
            (U_prev_state, tc_dtype, u_prev_spec),
            (B_prev_state, tc_dtype, b_prev_spec),
            (inc_chunk, torch.float32, inc_spec),
            (chunk_multiplier_storage, torch.float32, m_chunk_spec),
        )
    )
    cache_key = _chunk_increment_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        problem_shape=problem_shape,
        has_prev=has_prev,
        config=resolved_config,
        alignments=alignments,
    )
    return ChunkIncrementRuntimeArtifacts(
        problem_shape=problem_shape,
        config=resolved_config,
        compile_args=compile_args,
        runtime_args=runtime_args,
        cache_key=cache_key,
        outputs=ChunkIncrementOutputs(
            increment_chunk=inc_chunk,
            chunk_multiplier_storage=chunk_multiplier_storage,
        ),
    )


def _make_chunk_increment_compile_artifacts(
    U: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    has_prev: bool,
    config: ChunkIncrementConfig | None = None,
) -> ChunkIncrementCompileArtifacts:
    batch_size, heads, time_steps, P = map(int, U.shape)
    resolved_chunk_size = int(chunk_size)
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size
    problem_shape = _resolve_chunk_increment_problem_shape(
        U=U,
        B=B,
        padded_time=padded_time,
        chunk_size=resolved_chunk_size,
    )
    resolved_config = (
        _resolve_default_chunk_increment_config(
            D=int(B.shape[-1]),
            chunk_size=resolved_chunk_size,
        )
        if config is None
        else _normalize_chunk_increment_config(
            chunk_size=resolved_chunk_size,
            config=config,
        )
    )
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        u_prev_spec,
        b_prev_spec,
        increment_spec,
        chunk_multiplier_spec,
    ) = _chunk_increment_tensor_specs(problem_shape)
    tc_align = _compile_min_align_for_dtype(tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    alignments, compile_args = _make_tvm_ffi_compile_args_from_specs(
        (tc_dtype, u_spec, tc_align),
        (tc_dtype, b_spec, tc_align),
        (torch.float32, m_spec, fp32_align),
        (torch.float32, k_spec, fp32_align),
        (tc_dtype, u_prev_spec, tc_align),
        (tc_dtype, b_prev_spec, tc_align),
        (torch.float32, increment_spec, fp32_align),
        (torch.float32, chunk_multiplier_spec, fp32_align),
    )
    cache_key = _chunk_increment_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        problem_shape=problem_shape,
        has_prev=has_prev,
        config=resolved_config,
        alignments=alignments,
    )
    return ChunkIncrementCompileArtifacts(
        problem_shape=problem_shape,
        config=resolved_config,
        compile_args=compile_args,
        alignments=alignments,
        cache_key=cache_key,
    )


def _state_passing_tensor_specs(
    problem_shape: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]:
    B, H, C, P, D = problem_shape
    return (
        _make_tensor_spec((B, H, C, P, D)),
        _make_tensor_spec((B, H, C, 2)),
        _make_tensor_spec((B, H, C, P, D)),
        _make_tensor_spec((B, H, P, D)),
    )


def _chunk_scan_tensor_specs(
    problem_shape: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]:
    batch_size, heads, _padded_time, P, D, n_chunks, chunk_size = problem_shape
    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks
    return (
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 1, P)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 1, D)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 2)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 2, 2)),
        _make_tensor_spec((batch_head_chunk_count, P, 1, D)),
        _make_tensor_spec((batch_head_count, P)),
        _make_tensor_spec((batch_head_count, D)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 1, P)),
    )


def _compile_min_align_for_dtype(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return 16
    return max(4, torch.empty((), dtype=dtype).element_size())


def _choose_copy_bits_for_linear_tiles_from_properties(
    *,
    dtype: torch.dtype,
    assumed_align: int,
    tile_stride_elems: int,
    elems_per_thread: int,
    candidates_bits: tuple[int, ...] = (128, 64, 32),
) -> int:
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    elem_bits = elem_bytes * 8
    stride_bytes = int(tile_stride_elems) * elem_bytes
    best = elem_bits
    for bits in candidates_bits:
        if bits < elem_bits or bits % elem_bits != 0:
            continue
        vec_elems = bits // elem_bits
        if int(elems_per_thread) % vec_elems != 0:
            continue
        align = bits // 8
        if int(assumed_align) >= align and stride_bytes % align == 0:
            best = bits
            break
    return best


def _prepare_time_operand(
    tensor: torch.Tensor,
    *,
    padded_time: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        tensor.dtype == dtype
        and int(tensor.shape[2]) == padded_time
        and tensor.is_contiguous()
    ):
        tensor = tensor
    else:
        tensor = _pad_zero_time(tensor, T_pad=padded_time, dtype=dtype)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = _ensure_min_alignment(tensor, min_align=16)
    return tensor


def _prepare_m_operand(M: torch.Tensor, *, padded_time: int) -> torch.Tensor:
    if (
        int(M.shape[2]) == padded_time
        and M.dtype == torch.float32
        and M.is_contiguous()
    ):
        return M
    return _pad_m_identity(M, T_pad=padded_time)


def _chunk_increment_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    problem_shape: tuple[int, ...],
    has_prev: bool,
    config: ChunkIncrementConfig,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "chunk_increment_fwd",
        device_index,
        tc_dtype,
        tuple(int(dim) for dim in problem_shape),
        has_prev,
        config.cache_key,
        alignments,
    )


def _state_passing_key(
    *,
    device_index: int,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, int, int, int, int, int, bool],
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "state_passing_fwd",
        device_index,
        tuple(int(dim) for dim in problem_shape),
        tuple(bool(dim) if isinstance(dim, bool) else int(dim) for dim in launch_cfg),
        alignments,
    )


def _make_state_passing_cfg(
    *,
    num_threads: int,
    vecs_per_thread: int,
    copy_bits_in: int,
    copy_bits_out: int,
    copy_bits_state_in: int,
    copy_bits_state_out: int,
    has_init: bool,
) -> tuple[int, int, int, int, int, int, bool]:
    return (
        int(num_threads),
        int(vecs_per_thread),
        int(copy_bits_in),
        int(copy_bits_out),
        int(copy_bits_state_in),
        int(copy_bits_state_out),
        bool(has_init),
    )


def _make_state_passing_runtime_artifacts(
    *,
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    chunk_starts: torch.Tensor,
    final_state: torch.Tensor,
    initial_state_arg: torch.Tensor,
    has_init: bool,
    num_threads: int,
    vecs_per_thread: int,
) -> StatePassingRuntimeArtifacts:
    B, H, C, P, D = map(int, increment.shape)
    problem_shape = (B, H, C, P, D)
    (
        increment_spec,
        chunk_multiplier_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = _state_passing_tensor_specs(problem_shape)

    state_elem_count = P * D
    elems_per_thread = 2 * int(vecs_per_thread)
    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        increment,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        chunk_starts,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state_in = 32
    copy_bits_state_out = 32
    launch_cfg = _make_state_passing_cfg(
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=has_init,
    )
    config = StatePassingConfig(
        num_threads=int(num_threads),
        vecs_per_thread=int(vecs_per_thread),
    )

    runtime_args, alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (increment, torch.float32, increment_spec),
            (chunk_multiplier, torch.float32, chunk_multiplier_spec),
            (chunk_starts, torch.float32, chunk_starts_spec),
            (final_state, torch.float32, final_state_spec),
            (initial_state_arg, torch.float32, final_state_spec),
        )
    )
    cache_key = _state_passing_key(
        device_index=(
            increment.device.index if increment.device.index is not None else -1
        ),
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        alignments=alignments,
    )
    return StatePassingRuntimeArtifacts(
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config=config,
        compile_args=compile_args,
        runtime_args=runtime_args,
        cache_key=cache_key,
        outputs=StatePassingOutputs(
            chunk_starts=chunk_starts,
            final_state=final_state,
        ),
    )


def _make_state_passing_compile_artifacts(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    has_init: bool,
    num_threads: int,
    vecs_per_thread: int,
) -> StatePassingCompileArtifacts:
    B, H, C, P, D = map(int, increment.shape)
    problem_shape = (B, H, C, P, D)
    (
        increment_spec,
        chunk_multiplier_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = _state_passing_tensor_specs(problem_shape)
    state_elem_count = P * D
    elems_per_thread = 2 * int(vecs_per_thread)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    copy_bits_in = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=fp32_align,
        tile_stride_elems=state_elem_count,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state_in = 32
    copy_bits_state_out = 32
    launch_cfg = _make_state_passing_cfg(
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=has_init,
    )
    config = StatePassingConfig(
        num_threads=int(num_threads),
        vecs_per_thread=int(vecs_per_thread),
    )
    alignments, compile_args = _make_tvm_ffi_compile_args_from_specs(
        (torch.float32, increment_spec, fp32_align),
        (torch.float32, chunk_multiplier_spec, fp32_align),
        (torch.float32, chunk_starts_spec, fp32_align),
        (torch.float32, final_state_spec, fp32_align),
        (torch.float32, final_state_spec, fp32_align),
    )
    cache_key = _state_passing_key(
        device_index=(
            increment.device.index if increment.device.index is not None else -1
        ),
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        alignments=alignments,
    )
    return StatePassingCompileArtifacts(
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config=config,
        compile_args=compile_args,
        alignments=alignments,
        cache_key=cache_key,
    )


def _chunk_scan_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    out_dtype: torch.dtype,
    problem_shape: tuple[int, ...],
    m_block_size: int,
    n_block_size: int,
    num_threads: int,
    has_prev: bool,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "chunk_scan_fwd",
        device_index,
        tc_dtype,
        out_dtype,
        tuple(int(dim) for dim in problem_shape),
        int(m_block_size),
        int(n_block_size),
        int(num_threads),
        has_prev,
        alignments,
    )


def _fwd_host_cache_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    out_dtype: torch.dtype,
    problem_shape: tuple[int, ...],
    config_bundle: ForwardConfigBundle,
    launch_cfg: tuple[int, ...],
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "v2x2ssd_fwd",
        device_index,
        tc_dtype,
        out_dtype,
        tuple(int(dim) for dim in problem_shape),
        config_bundle.cache_key,
        tuple(bool(dim) if isinstance(dim, bool) else int(dim) for dim in launch_cfg),
        alignments,
    )


def _resolve_chunk_increment_cta_tiler(*, D: int) -> tuple[int, int, int]:
    # The 96-wide N tile is efficient when it covers the full state width, but
    # mixed full+tail tiling can perturb the current epilogue path on realistic
    # D=2N mixer shapes. Pick a tail-safe family instead of changing semantics.
    if D <= 96 or D % 96 == 0:
        return (64, 96, 32)
    return (64, 64, 32)


def _normalize_chunk_increment_config(
    *,
    chunk_size: int,
    config: ChunkIncrementConfig,
) -> ChunkIncrementConfig:
    normalized_kernel = ChunkIncrementFwdAmpere(
        cutlass.Float16,
        chunk_size=int(chunk_size),
        cta_tiler=tuple(int(v) for v in config.cta_tiler),
        num_stages=int(config.num_stages),
    )
    return ChunkIncrementConfig(
        cta_tiler=tuple(int(v) for v in normalized_kernel.cta_tiler),
        num_stages=int(normalized_kernel.num_stages),
    )


def _resolve_default_chunk_increment_config(
    *,
    D: int,
    chunk_size: int,
) -> ChunkIncrementConfig:
    return _normalize_chunk_increment_config(
        chunk_size=chunk_size,
        config=ChunkIncrementConfig(
            cta_tiler=_resolve_chunk_increment_cta_tiler(D=D),
            num_stages=3,
        ),
    )


def _validate_chunk_increment_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None,
    B_prev: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if (U_prev is None) ^ (B_prev is None):
        raise ValueError("U_prev and B_prev must be passed together (or both omitted).")
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if U.dtype != B.dtype:
        raise ValueError("U and B must share dtype.")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")

    batch_size, heads, time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (batch_size, heads, time_steps, D):
        raise ValueError("B must be (B,H,T,D) matching U.")
    if M.shape != (batch_size, heads, time_steps, 2):
        raise ValueError(f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}.")
    if K.shape != (batch_size, heads, time_steps, 2, 2):
        raise ValueError(
            f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
        )
    if D % 2 != 0:
        raise ValueError("B last dim must be divisible by 2.")
    if U_prev is not None and (
        U_prev.shape != (batch_size, heads, P) or B_prev.shape != (batch_size, heads, D)
    ):
        raise ValueError("U_prev/B_prev must be (B,H,P)/(B,H,D).")
    return batch_size, heads, time_steps, P, D


def _validate_state_passing_inputs(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if increment.ndim != 5:
        raise ValueError(
            f"increment must be (B,H,C,P,D). Got {tuple(increment.shape)}."
        )
    if chunk_multiplier.ndim != 4 or chunk_multiplier.shape[-1] != 2:
        raise ValueError(
            f"chunk_multiplier must be (B,H,C,2). Got {tuple(chunk_multiplier.shape)}."
        )
    if increment.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if increment.dtype != torch.float32 or chunk_multiplier.dtype != torch.float32:
        raise TypeError(
            "increment and chunk_multiplier must be float32 at the stage boundary."
        )

    B, H, C, P, D = map(int, increment.shape)
    if D % 2 != 0:
        raise ValueError("increment last dim must be divisible by 2.")
    if tuple(chunk_multiplier.shape[:3]) != (B, H, C):
        raise ValueError("chunk_multiplier leading dims must match increment.")
    if initial_states is not None:
        if tuple(initial_states.shape) != (B, H, P, D):
            raise ValueError("initial_states must be (B,H,P,D) and match increment.")
        if initial_states.dtype != torch.float32:
            raise TypeError("initial_states must be float32.")
    return B, H, C, P, D


def _validate_chunk_scan_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    chunk_size: int,
    output_dtype: torch.dtype,
) -> tuple[int, int, int, int, int, int]:
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if U.dtype != B.dtype or U.dtype != C.dtype:
        raise ValueError("U/B/C must share dtype.")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B/C must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")
    if chunk_starts.dtype != torch.float32:
        raise TypeError("chunk_starts must be float32.")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("output_dtype must be float16/bfloat16/float32.")

    batch_size, heads, time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (batch_size, heads, time_steps, D) or C.shape != (
        batch_size,
        heads,
        time_steps,
        D,
    ):
        raise ValueError("B/C must be (B,H,T,D) matching U.")
    if M.shape != (batch_size, heads, time_steps, 2):
        raise ValueError(f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}.")
    if K.shape != (batch_size, heads, time_steps, 2, 2):
        raise ValueError(
            f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
        )
    if D % 2 != 0:
        raise ValueError("B/C last dim must be divisible by 2.")

    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    if tuple(chunk_starts.shape) != (batch_size, heads, n_chunks, P, D):
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D) ={(batch_size, heads, n_chunks, P, D)}."
        )
    if B_prev is not None and (
        B_prev.shape != (batch_size, heads, D) or U_prev.shape != (batch_size, heads, P)
    ):
        raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
    return batch_size, heads, time_steps, P, D, n_chunks


def _make_chunk_scan_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, ...],
):
    batch_size, heads, padded_time, P, D, n_chunks, chunk_size = problem_shape
    m_block_size, n_block_size, num_threads = launch_cfg

    @cute.jit
    def _chunk_scan_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        Z0_t: cute.Tensor,
        U_prev_t: cute.Tensor,
        B_prev_t: cute.Tensor,
        Out_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        u_spec, b_spec, m_spec, k_spec, z0_spec, u_prev_spec, b_prev_spec, out_spec = (
            _chunk_scan_tensor_specs(
                (batch_size, heads, padded_time, P, D, n_chunks, chunk_size)
            )
        )
        mU = _make_static_tensor_spec_view(U_t, u_spec)
        mB = _make_static_tensor_spec_view(B_t, b_spec)
        mC = _make_static_tensor_spec_view(C_t, b_spec)
        mM = _make_static_tensor_spec_view(M_t, m_spec)
        mK = _make_static_tensor_spec_view(K_t, k_spec)
        mZ0 = _make_static_tensor_spec_view(Z0_t, z0_spec)
        mU_prev0 = _make_static_tensor_spec_view(U_prev_t, u_prev_spec)
        mB_prev0 = _make_static_tensor_spec_view(B_prev_t, b_prev_spec)
        mOut = _make_static_tensor_spec_view(Out_t, out_spec)

        chunk_scan = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=chunk_size,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
        )
        chunk_scan._launch_main_kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mZ0,
            mU_prev0,
            mB_prev0,
            mOut,
            stream=stream,
        )

    return _chunk_scan_host_wrapper


def _make_chunk_scan_runtime_artifacts(
    *,
    U_scan: torch.Tensor,
    B_scan: torch.Tensor,
    C_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    chunk_starts: torch.Tensor,
    U_prev_state: torch.Tensor,
    B_prev_state: torch.Tensor,
    output_chunk: torch.Tensor,
    batch_size: int,
    heads: int,
    padded_time: int,
    P: int,
    D: int,
    n_chunks: int,
    chunk_size: int,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    resolved_m_block_size: int,
    resolved_n_block_size: int,
    resolved_num_threads: int,
    has_prev: bool,
) -> ChunkScanRuntimeArtifacts:
    problem_shape = (batch_size, heads, padded_time, P, D, n_chunks, chunk_size)
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        z0_spec,
        u_prev_spec,
        b_prev_spec,
        output_spec,
    ) = _chunk_scan_tensor_specs(problem_shape)
    runtime_args, alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (U_scan, tc_dtype, u_spec),
            (B_scan, tc_dtype, b_spec),
            (C_tc, tc_dtype, b_spec),
            (M_f, torch.float32, m_spec),
            (K_f, torch.float32, k_spec),
            (chunk_starts, torch.float32, z0_spec),
            (U_prev_state, tc_dtype, u_prev_spec),
            (B_prev_state, tc_dtype, b_prev_spec),
            (output_chunk, output_dtype, output_spec),
        )
    )
    cache_key = _chunk_scan_key(
        device_index=device_index,
        tc_dtype=tc_dtype,
        out_dtype=output_dtype,
        problem_shape=problem_shape,
        m_block_size=resolved_m_block_size,
        n_block_size=resolved_n_block_size,
        num_threads=resolved_num_threads,
        has_prev=has_prev,
        alignments=alignments,
    )
    return ChunkScanRuntimeArtifacts(
        problem_shape=problem_shape,
        launch_cfg=(
            resolved_m_block_size,
            resolved_n_block_size,
            resolved_num_threads,
        ),
        config=ChunkScanConfig(
            m_block_size=resolved_m_block_size,
            n_block_size=resolved_n_block_size,
            num_threads=resolved_num_threads,
        ),
        compile_args=compile_args,
        runtime_args=runtime_args,
        cache_key=cache_key,
        outputs=ChunkScanOutputs(
            output_chunk=output_chunk,
            output_view=output_chunk.reshape(
                batch_size, heads, n_chunks, chunk_size, 1, P
            ).reshape(batch_size, heads, padded_time, P),
        ),
    )


def _make_chunk_scan_compile_artifacts(
    U: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    device_index: int,
    m_block_size: int | None,
    n_block_size: int,
    num_threads: int,
    has_prev: bool,
) -> ChunkScanCompileArtifacts:
    batch_size, heads, time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    resolved_chunk_size = int(chunk_size)
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    resolved_m_block_size, resolved_n_block_size, resolved_num_threads = (
        _resolve_chunk_scan_launch_cfg(
            D=D,
            P=P,
            chunk_size=resolved_chunk_size,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
            requested_m_block_size=m_block_size,
            requested_n_block_size=int(n_block_size),
            requested_num_threads=int(num_threads),
        )
    )
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        resolved_chunk_size,
    )
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        z0_spec,
        u_prev_spec,
        b_prev_spec,
        output_spec,
    ) = _chunk_scan_tensor_specs(problem_shape)
    tc_align = _compile_min_align_for_dtype(tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    output_align = _compile_min_align_for_dtype(output_dtype)
    alignments, compile_args = _make_tvm_ffi_compile_args_from_specs(
        (tc_dtype, u_spec, tc_align),
        (tc_dtype, b_spec, tc_align),
        (tc_dtype, b_spec, tc_align),
        (torch.float32, m_spec, fp32_align),
        (torch.float32, k_spec, fp32_align),
        (torch.float32, z0_spec, fp32_align),
        (tc_dtype, u_prev_spec, tc_align),
        (tc_dtype, b_prev_spec, tc_align),
        (output_dtype, output_spec, output_align),
    )
    cache_key = _chunk_scan_key(
        device_index=device_index,
        tc_dtype=tc_dtype,
        out_dtype=output_dtype,
        problem_shape=problem_shape,
        m_block_size=resolved_m_block_size,
        n_block_size=resolved_n_block_size,
        num_threads=resolved_num_threads,
        has_prev=has_prev,
        alignments=alignments,
    )
    return ChunkScanCompileArtifacts(
        problem_shape=problem_shape,
        launch_cfg=(
            resolved_m_block_size,
            resolved_n_block_size,
            resolved_num_threads,
        ),
        config=ChunkScanConfig(
            m_block_size=resolved_m_block_size,
            n_block_size=resolved_n_block_size,
            num_threads=resolved_num_threads,
        ),
        compile_args=compile_args,
        alignments=alignments,
        cache_key=cache_key,
    )


def _make_chunk_increment_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    config: ChunkIncrementConfig,
):
    batch_size, heads, padded_time, P, D, n_chunks, chunk_size = problem_shape

    @cute.jit
    def _chunk_increment_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        U_prev_t: cute.Tensor,
        B_prev_t: cute.Tensor,
        increment_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        (
            u_spec,
            b_spec,
            m_spec,
            k_spec,
            u_prev_spec,
            b_prev_spec,
            inc_spec,
            m_chunk_spec,
        ) = _chunk_increment_tensor_specs(
            (batch_size, heads, padded_time, P, D, n_chunks, chunk_size)
        )
        mU = _make_static_tensor_spec_view(U_t, u_spec)
        mB = _make_static_tensor_spec_view(B_t, b_spec)
        mM = _make_static_tensor_spec_view(M_t, m_spec)
        mKprev = _make_static_tensor_spec_view(K_t, k_spec)
        mKcurr = cute.make_tensor(mKprev.iterator + 2, mKprev.layout)
        mU_prev = _make_static_tensor_spec_view(U_prev_t, u_prev_spec)
        mB_prev = _make_static_tensor_spec_view(B_prev_t, b_prev_spec)
        mIncrement = _make_static_tensor_spec_view(increment_t, inc_spec)
        mChunkMultiplier = _make_static_tensor_spec_view(
            chunk_multiplier_t, m_chunk_spec
        )

        chunk_increment = ChunkIncrementFwdAmpere(
            mU.element_type,
            chunk_size=chunk_size,
            cta_tiler=config.cta_tiler,
            num_stages=config.num_stages,
        )
        chunk_increment._launch_kernel(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mU_prev,
            mB_prev,
            mIncrement,
            mChunkMultiplier,
            stream=stream,
        )

    return _chunk_increment_host_wrapper


def _make_state_passing_host_wrapper(
    *, problem_shape: tuple[int, ...], launch_cfg: tuple[int, ...]
):
    increment_spec, chunk_multiplier_spec, chunk_starts_spec, final_state_spec = (
        _state_passing_tensor_specs(problem_shape)
    )
    (
        num_threads,
        vecs_per_thread,
        copy_bits_in,
        copy_bits_out,
        copy_bits_state_in,
        copy_bits_state_out,
        has_init,
    ) = launch_cfg
    state_passing_kernel = StatePassingFwdAmpere(
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=has_init,
    )

    @cute.jit
    def _state_passing_host_wrapper(
        increment_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        chunk_starts_t: cute.Tensor,
        final_state_t: cute.Tensor,
        initial_state_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        increment_view = _make_static_tensor_spec_view(increment_t, increment_spec)
        chunk_multiplier_view = _make_static_tensor_spec_view(
            chunk_multiplier_t, chunk_multiplier_spec
        )
        chunk_starts_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_spec
        )
        final_state_view = _make_static_tensor_spec_view(
            final_state_t, final_state_spec
        )
        initial_state_view = _make_static_tensor_spec_view(
            initial_state_t, final_state_spec
        )

        state_passing_kernel.call_on_stream(
            increment_view,
            chunk_multiplier_view,
            chunk_starts_view,
            final_state_view,
            initial_state_view,
            stream,
        )

    return _state_passing_host_wrapper


def _make_v2x2ssd_fwd_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    config_bundle: ForwardConfigBundle,
    launch_cfg: tuple[int, ...],
):
    batch_size, heads, padded_time, P, D, n_chunks, chunk_size = problem_shape
    (
        m_block_size,
        n_block_size,
        scan_num_threads,
        state_num_threads,
        state_vecs_per_thread,
        state_copy_bits_in,
        state_copy_bits_out,
        state_copy_bits_state_in,
        state_copy_bits_state_out,
        has_init,
    ) = launch_cfg
    chunk_increment_spec = _chunk_increment_tensor_specs(problem_shape)
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
    )
    chunk_scan_spec = _chunk_scan_tensor_specs(problem_shape)
    state_passing_kernel = StatePassingFwdAmpere(
        num_threads=state_num_threads,
        vecs_per_thread=state_vecs_per_thread,
        copy_bits_in=state_copy_bits_in,
        copy_bits_out=state_copy_bits_out,
        copy_bits_state_in=state_copy_bits_state_in,
        copy_bits_state_out=state_copy_bits_state_out,
        has_init=has_init,
    )

    @cute.jit
    def _v2x2ssd_fwd_host_wrapper(
        U_increment_t: cute.Tensor,
        B_increment_t: cute.Tensor,
        M_increment_t: cute.Tensor,
        K_increment_t: cute.Tensor,
        U_prev_increment_t: cute.Tensor,
        B_prev_increment_t: cute.Tensor,
        increment_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        increment_state_t: cute.Tensor,
        chunk_multiplier_state_t: cute.Tensor,
        chunk_starts_t: cute.Tensor,
        final_state_t: cute.Tensor,
        initial_state_t: cute.Tensor,
        U_scan_t: cute.Tensor,
        B_scan_t: cute.Tensor,
        C_scan_t: cute.Tensor,
        M_scan_t: cute.Tensor,
        K_scan_t: cute.Tensor,
        Z0_scan_t: cute.Tensor,
        U_prev_scan_t: cute.Tensor,
        B_prev_scan_t: cute.Tensor,
        output_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        (
            u_increment_spec,
            b_increment_spec,
            m_increment_spec,
            k_increment_spec,
            u_prev_increment_spec,
            b_prev_increment_spec,
            increment_spec,
            chunk_multiplier_spec,
        ) = chunk_increment_spec
        (
            increment_state_spec,
            chunk_multiplier_state_spec,
            chunk_starts_spec,
            final_state_spec,
        ) = state_passing_spec
        (
            u_scan_spec,
            b_scan_spec,
            m_scan_spec,
            k_scan_spec,
            z0_scan_spec,
            u_prev_scan_spec,
            b_prev_scan_spec,
            output_spec,
        ) = chunk_scan_spec

        U_increment_view = _make_static_tensor_spec_view(
            U_increment_t, u_increment_spec
        )
        B_increment_view = _make_static_tensor_spec_view(
            B_increment_t, b_increment_spec
        )
        M_increment_view = _make_static_tensor_spec_view(
            M_increment_t, m_increment_spec
        )
        K_previous_view = _make_static_tensor_spec_view(K_increment_t, k_increment_spec)
        K_current_view = cute.make_tensor(
            K_previous_view.iterator + 2, K_previous_view.layout
        )
        U_prev_increment_view = _make_static_tensor_spec_view(
            U_prev_increment_t, u_prev_increment_spec
        )
        B_prev_increment_view = _make_static_tensor_spec_view(
            B_prev_increment_t, b_prev_increment_spec
        )
        increment_view = _make_static_tensor_spec_view(increment_t, increment_spec)
        chunk_multiplier_view = _make_static_tensor_spec_view(
            chunk_multiplier_t, chunk_multiplier_spec
        )

        chunk_increment_kernel = ChunkIncrementFwdAmpere(
            U_increment_view.element_type,
            chunk_size=chunk_size,
            cta_tiler=config_bundle.chunk_increment.cta_tiler,
            num_stages=config_bundle.chunk_increment.num_stages,
        )
        chunk_increment_kernel._launch_kernel(
            U_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_view,
            K_current_view,
            U_prev_increment_view,
            B_prev_increment_view,
            increment_view,
            chunk_multiplier_view,
            stream=stream,
        )

        increment_state_view = _make_static_tensor_spec_view(
            increment_state_t, increment_state_spec
        )
        chunk_multiplier_state_view = _make_static_tensor_spec_view(
            chunk_multiplier_state_t, chunk_multiplier_state_spec
        )
        chunk_starts_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_spec
        )
        final_state_view = _make_static_tensor_spec_view(
            final_state_t, final_state_spec
        )
        initial_state_view = _make_static_tensor_spec_view(
            initial_state_t, final_state_spec
        )
        state_passing_kernel.call_on_stream(
            increment_state_view,
            chunk_multiplier_state_view,
            chunk_starts_view,
            final_state_view,
            initial_state_view,
            stream,
        )

        U_scan_view = _make_static_tensor_spec_view(U_scan_t, u_scan_spec)
        B_scan_view = _make_static_tensor_spec_view(B_scan_t, b_scan_spec)
        C_scan_view = _make_static_tensor_spec_view(C_scan_t, b_scan_spec)
        M_scan_view = _make_static_tensor_spec_view(M_scan_t, m_scan_spec)
        K_scan_view = _make_static_tensor_spec_view(K_scan_t, k_scan_spec)
        Z0_scan_view = _make_static_tensor_spec_view(Z0_scan_t, z0_scan_spec)
        U_prev_scan_view = _make_static_tensor_spec_view(
            U_prev_scan_t, u_prev_scan_spec
        )
        B_prev_scan_view = _make_static_tensor_spec_view(
            B_prev_scan_t, b_prev_scan_spec
        )
        output_view = _make_static_tensor_spec_view(output_t, output_spec)

        chunk_scan_kernel = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=chunk_size,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=scan_num_threads,
        )
        chunk_scan_kernel._launch_main_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            Z0_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            output_view,
            stream=stream,
        )

    return _v2x2ssd_fwd_host_wrapper


def _get_compiled_state_passing_kernel(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, int, int, int, int, int, bool],
    compile_args: tuple[object, ...],
    cache_key: tuple,
):
    compiled = _STATE_PASSING_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_state_passing_host_wrapper(
            problem_shape=problem_shape,
            launch_cfg=launch_cfg,
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _STATE_PASSING_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_chunk_increment_kernel(
    *,
    problem_shape: tuple[int, ...],
    config: ChunkIncrementConfig,
    compile_args: tuple[object, ...],
    cache_key: tuple,
):
    compiled = _CHUNK_INCREMENT_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_chunk_increment_host_wrapper(
            problem_shape=problem_shape,
            config=config,
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_INCREMENT_CACHE[cache_key] = compiled
    return compiled


def _get_compiled_chunk_scan_kernel(
    *,
    problem_shape: tuple[int, ...],
    launch_cfg: tuple[int, int, int],
    compile_args: tuple[object, ...],
    cache_key: tuple,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    D: int,
    P: int,
    chunk_size: int,
):
    compiled = _CHUNK_SCAN_CACHE.get(cache_key)
    if compiled is None:
        in_cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
        out_cutlass_dtype = _torch_to_cutlass_dtype(output_dtype)
        resolved_m_block_size, resolved_n_block_size, resolved_num_threads = launch_cfg
        kernel = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=chunk_size,
            m_block_size=resolved_m_block_size,
            n_block_size=resolved_n_block_size,
            num_threads=resolved_num_threads,
        )
        if not kernel.can_implement(
            in_cutlass_dtype,
            out_cutlass_dtype,
            device_index=device_index,
        ):
            raise ValueError("Resolved chunk_scan configuration is not supported.")
        host_wrapper = _make_chunk_scan_host_wrapper(
            problem_shape=problem_shape,
            launch_cfg=launch_cfg,
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_SCAN_CACHE[cache_key] = compiled
    return compiled


def _reset_chunk_increment_outputs(outputs: ChunkIncrementOutputs) -> None:
    outputs.increment_chunk.zero_()
    outputs.chunk_multiplier_storage.zero_()


def _reset_state_passing_outputs(outputs: StatePassingOutputs) -> None:
    outputs.chunk_starts.zero_()
    outputs.final_state.zero_()


def _reset_chunk_scan_outputs(outputs: ChunkScanOutputs) -> None:
    outputs.output_chunk.zero_()


def _matching_packaged_chunk_increment_specs(
    *,
    device_index: int,
    problem_shape: tuple[int, ...],
    tc_dtype: torch.dtype,
) -> tuple[object, ...]:
    from slinoss.ops.v2x2ssd.cute.aot import list_packaged_chunk_increment_aot_specs

    hardware = current_hardware_fingerprint(device_index=device_index)
    return tuple(
        spec
        for spec in list_packaged_chunk_increment_aot_specs(arch_tag=hardware.arch_tag)
        if spec.P == int(problem_shape[3])
        and spec.D == int(problem_shape[4])
        and spec.chunk_size == int(problem_shape[6])
        and spec.tc_dtype_name == str(tc_dtype).replace("torch.", "")
    )


def _matching_packaged_state_passing_specs(
    *,
    device_index: int,
    problem_shape: tuple[int, ...],
    has_init: bool,
) -> tuple[object, ...]:
    from slinoss.ops.v2x2ssd.cute.aot import list_packaged_state_passing_aot_specs

    hardware = current_hardware_fingerprint(device_index=device_index)
    return tuple(
        spec
        for spec in list_packaged_state_passing_aot_specs(arch_tag=hardware.arch_tag)
        if spec.P == int(problem_shape[3])
        and spec.D == int(problem_shape[4])
        and spec.has_init == bool(has_init)
    )


def _matching_packaged_chunk_scan_specs(
    *,
    device_index: int,
    problem_shape: tuple[int, ...],
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> tuple[object, ...]:
    from slinoss.ops.v2x2ssd.cute.aot import list_packaged_chunk_scan_aot_specs

    hardware = current_hardware_fingerprint(device_index=device_index)
    matching_specs = []
    for spec in list_packaged_chunk_scan_aot_specs(arch_tag=hardware.arch_tag):
        if (
            spec.P != int(problem_shape[3])
            or spec.D != int(problem_shape[4])
            or spec.chunk_size != int(problem_shape[6])
            or spec.tc_dtype_name != str(tc_dtype).replace("torch.", "")
            or spec.output_dtype_name != str(output_dtype).replace("torch.", "")
        ):
            continue
        kernel = ChunkScanFwdAmpere(
            D=spec.D,
            P=spec.P,
            L=spec.chunk_size,
            m_block_size=spec.config.m_block_size,
            n_block_size=spec.config.n_block_size,
            num_threads=spec.config.num_threads,
        )
        if not kernel.can_implement(
            _torch_to_cutlass_dtype(tc_dtype),
            _torch_to_cutlass_dtype(output_dtype),
            device_index=device_index,
        ):
            continue
        matching_specs.append(spec)
    return tuple(matching_specs)


def tune_chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> ChunkIncrementConfig:
    _validate_chunk_increment_inputs(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
    )
    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    problem_shape = _make_chunk_increment_compile_artifacts(
        U,
        B,
        chunk_size=resolved_chunk_size,
        compute_dtype=compute_dtype,
        has_prev=U_prev is not None,
    ).problem_shape
    hardware = current_hardware_fingerprint(device_index=device_index)
    problem_key = chunk_increment_problem_key(
        tc_dtype=tc_dtype,
        P=int(problem_shape[3]),
        D=int(problem_shape[4]),
        chunk_size=int(problem_shape[6]),
        has_prev=U_prev is not None,
    )
    if not autotune_force_retune():
        cached_record = lookup_tuning_record(
            scope="chunk_increment",
            hardware=hardware,
            problem_key=problem_key.to_record(),
        )
        if cached_record is not None and isinstance(cached_record.get("config"), dict):
            return ChunkIncrementConfig.from_record(
                cast(dict[str, object], cached_record["config"])
            )

    from slinoss.ops.v2x2ssd.cute.aot import try_load_packaged_chunk_increment_function

    candidate_specs = _matching_packaged_chunk_increment_specs(
        device_index=device_index,
        problem_shape=problem_shape,
        tc_dtype=tc_dtype,
    )
    best_config: ChunkIncrementConfig | None = None
    best_latency_ms = float("inf")
    if candidate_specs:
        batch_size, heads, time_steps, P, D = _validate_chunk_increment_inputs(
            U,
            M,
            K,
            B,
            U_prev=U_prev,
            B_prev=B_prev,
        )
        n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
        padded_time = n_chunks * resolved_chunk_size
        U_tc = _pad_zero_time(U, T_pad=padded_time, dtype=tc_dtype)
        B_tc = _pad_zero_time(B, T_pad=padded_time, dtype=tc_dtype)
        M_f = _pad_m_identity(M, T_pad=padded_time)
        K_f = _pad_zero_time(K, T_pad=padded_time, dtype=torch.float32)
        U_tc = _ensure_min_alignment(U_tc, min_align=16)
        B_tc = _ensure_min_alignment(B_tc, min_align=16)
        if U_prev is None:
            U_prev_state, B_prev_state = _get_zero_prev_tensors(
                device=U.device,
                dtype=tc_dtype,
                batch_size=batch_size,
                heads=heads,
                P=P,
                D=D,
            )
        else:
            U_prev_state = _ensure_min_alignment(
                U_prev.to(dtype=tc_dtype).contiguous(),
                min_align=16,
            )
            B_prev_state = _ensure_min_alignment(
                B_prev.to(dtype=tc_dtype).contiguous(),
                min_align=16,
            )
        batch_head_chunk_count = batch_size * heads * n_chunks
        for spec in candidate_specs:
            packaged = try_load_packaged_chunk_increment_function(spec)
            if packaged is None:
                continue
            outputs = ChunkIncrementOutputs(
                increment_chunk=torch.zeros(
                    (batch_head_chunk_count, P, D),
                    device=U.device,
                    dtype=torch.float32,
                ),
                chunk_multiplier_storage=torch.zeros(
                    (batch_head_chunk_count, 2),
                    device=U.device,
                    dtype=torch.float32,
                ),
            )
            runtime_artifacts = _make_chunk_increment_runtime_artifacts(
                U=U,
                M=M,
                K=K,
                B=B,
                U_tc=U_tc,
                B_tc=B_tc,
                M_f=M_f,
                K_f=K_f,
                U_prev_state=U_prev_state,
                B_prev_state=B_prev_state,
                inc_chunk=outputs.increment_chunk,
                chunk_multiplier_storage=outputs.chunk_multiplier_storage,
                chunk_size=resolved_chunk_size,
                compute_dtype=compute_dtype,
                has_prev=U_prev is not None,
                config=spec.config,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: packaged(*runtime_artifacts.runtime_args),
                device=U.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_chunk_increment_outputs(
                    runtime_artifacts.outputs
                ),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = spec.config
    else:
        for config in chunk_increment_candidate_configs(
            P=int(problem_shape[3]),
            D=int(problem_shape[4]),
            chunk_size=resolved_chunk_size,
        ):
            prepared = _make_chunk_increment_prepared_launch(
                U,
                M,
                K,
                B,
                U_prev=U_prev,
                B_prev=B_prev,
                chunk_size=resolved_chunk_size,
                compute_dtype=compute_dtype,
                cta_tiler=config.cta_tiler,
                num_stages=config.num_stages,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: prepared.compiled(*prepared.runtime_args),
                device=U.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_chunk_increment_outputs(
                    prepared.outputs
                ),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = config
    if best_config is None:
        return _resolve_default_chunk_increment_config(
            D=int(problem_shape[4]),
            chunk_size=int(problem_shape[6]),
        )
    store_tuning_record(
        scope="chunk_increment",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=best_config.to_record(),
        metadata={
            "latency_ms": best_latency_ms,
            "candidate_count": len(candidate_specs)
            if candidate_specs
            else len(
                chunk_increment_candidate_configs(
                    P=int(problem_shape[3]),
                    D=int(problem_shape[4]),
                    chunk_size=resolved_chunk_size,
                )
            ),
            "source": "packaged_aot" if candidate_specs else "jit",
        },
    )
    return best_config


def tune_state_passing_cute(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
) -> StatePassingConfig:
    B, H, C, P, D = _validate_state_passing_inputs(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
    )
    device_index = (
        int(increment.device.index)
        if increment.device.index is not None
        else torch.cuda.current_device()
    )
    hardware = current_hardware_fingerprint(device_index=device_index)
    problem_key = state_passing_problem_key(
        P=P, D=D, has_init=initial_states is not None
    )
    if not autotune_force_retune():
        cached_record = lookup_tuning_record(
            scope="state_passing",
            hardware=hardware,
            problem_key=problem_key.to_record(),
        )
        if cached_record is not None and isinstance(cached_record.get("config"), dict):
            return StatePassingConfig.from_record(
                cast(dict[str, object], cached_record["config"])
            )

    from slinoss.ops.v2x2ssd.cute.aot import try_load_packaged_state_passing_function

    problem_shape = (B, H, C, P, D)
    candidate_specs = _matching_packaged_state_passing_specs(
        device_index=device_index,
        problem_shape=problem_shape,
        has_init=initial_states is not None,
    )
    best_config: StatePassingConfig | None = None
    best_latency_ms = float("inf")
    if candidate_specs:
        increment_contig = increment.contiguous()
        chunk_multiplier_contig = chunk_multiplier.contiguous()
        initial_state_arg = (
            increment_contig if initial_states is None else initial_states.contiguous()
        )
        for spec in candidate_specs:
            packaged = try_load_packaged_state_passing_function(spec)
            if packaged is None:
                continue
            outputs = StatePassingOutputs(
                chunk_starts=torch.empty(
                    (B, H, C, P, D),
                    device=increment.device,
                    dtype=torch.float32,
                ),
                final_state=torch.empty(
                    (B, H, P, D),
                    device=increment.device,
                    dtype=torch.float32,
                ),
            )
            outputs.final_state.zero_()
            runtime_artifacts = _make_state_passing_runtime_artifacts(
                increment=increment_contig,
                chunk_multiplier=chunk_multiplier_contig,
                chunk_starts=outputs.chunk_starts,
                final_state=outputs.final_state,
                initial_state_arg=initial_state_arg,
                has_init=initial_states is not None,
                num_threads=spec.config.num_threads,
                vecs_per_thread=spec.config.vecs_per_thread,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: packaged(*runtime_artifacts.runtime_args),
                device=increment.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_state_passing_outputs(
                    runtime_artifacts.outputs
                ),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = spec.config
    else:
        for config in state_passing_candidate_configs(
            P=P, D=D, has_init=initial_states is not None
        ):
            prepared = _make_state_passing_prepared_launch(
                increment,
                chunk_multiplier,
                initial_states=initial_states,
                num_threads=config.num_threads,
                vecs_per_thread=config.vecs_per_thread,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: prepared.compiled(*prepared.runtime_args),
                device=increment.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_state_passing_outputs(prepared.outputs),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = config
    assert best_config is not None
    store_tuning_record(
        scope="state_passing",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=best_config.to_record(),
        metadata={
            "latency_ms": best_latency_ms,
            "candidate_count": len(candidate_specs)
            if candidate_specs
            else len(
                state_passing_candidate_configs(
                    P=P, D=D, has_init=initial_states is not None
                )
            ),
            "source": "packaged_aot" if candidate_specs else "jit",
        },
    )
    return best_config


def tune_chunk_scan_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_size: int = 64,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> ChunkScanConfig:
    batch_size, heads, time_steps, P, D, n_chunks = _validate_chunk_scan_inputs(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )
    resolved_chunk_size = int(chunk_size)
    padded_time = n_chunks * resolved_chunk_size
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        resolved_chunk_size,
    )
    hardware = current_hardware_fingerprint(device_index=device_index)
    problem_key = chunk_scan_problem_key(
        tc_dtype=tc_dtype,
        output_dtype=output_dtype,
        P=P,
        D=D,
        chunk_size=resolved_chunk_size,
        has_prev=B_prev is not None,
    )
    if not autotune_force_retune():
        cached_record = lookup_tuning_record(
            scope="chunk_scan",
            hardware=hardware,
            problem_key=problem_key.to_record(),
        )
        if cached_record is not None and isinstance(cached_record.get("config"), dict):
            return ChunkScanConfig.from_record(
                cast(dict[str, object], cached_record["config"])
            )

    from slinoss.ops.v2x2ssd.cute.aot import try_load_packaged_chunk_scan_function

    candidate_specs = _matching_packaged_chunk_scan_specs(
        device_index=device_index,
        problem_shape=problem_shape,
        tc_dtype=tc_dtype,
        output_dtype=output_dtype,
    )
    best_config: ChunkScanConfig | None = None
    best_latency_ms = float("inf")
    if candidate_specs:
        U_tc = _pad_zero_time(U, T_pad=padded_time, dtype=tc_dtype)
        B_tc = _pad_zero_time(B, T_pad=padded_time, dtype=tc_dtype)
        C_tc = _pad_zero_time(C, T_pad=padded_time, dtype=tc_dtype)
        M_f = _pad_m_identity(M, T_pad=padded_time)
        K_f = _pad_zero_time(K, T_pad=padded_time, dtype=torch.float32)
        U_tc = _ensure_min_alignment(U_tc, min_align=16)
        B_tc = _ensure_min_alignment(B_tc, min_align=16)
        C_tc = _ensure_min_alignment(C_tc, min_align=16)
        U_scan = _guard_prev_time_base(U_tc, min_align=16)
        B_scan = _guard_prev_time_base(B_tc, min_align=16)
        if B_prev is None:
            U_prev_state, B_prev_state = _get_zero_prev_tensors(
                device=U.device,
                dtype=tc_dtype,
                batch_size=batch_size,
                heads=heads,
                P=P,
                D=D,
            )
        else:
            B_prev_state = _ensure_min_alignment(
                B_prev.to(dtype=tc_dtype).contiguous(),
                min_align=16,
            )
            U_prev_state = _ensure_min_alignment(
                U_prev.to(dtype=tc_dtype).contiguous(),
                min_align=16,
            )
        chunk_starts_c = _ensure_min_alignment(chunk_starts.contiguous(), min_align=16)
        batch_head_chunk_count = batch_size * heads * n_chunks
        for spec in candidate_specs:
            packaged = try_load_packaged_chunk_scan_function(spec)
            if packaged is None:
                continue
            outputs = ChunkScanOutputs(
                output_chunk=torch.empty(
                    (batch_head_chunk_count, resolved_chunk_size, 1, P),
                    device=U.device,
                    dtype=output_dtype,
                ),
                output_view=torch.empty(0, device=U.device, dtype=output_dtype),
            )
            runtime_artifacts = _make_chunk_scan_runtime_artifacts(
                U_scan=U_scan,
                B_scan=B_scan,
                C_tc=C_tc,
                M_f=M_f,
                K_f=K_f,
                chunk_starts=chunk_starts_c,
                U_prev_state=U_prev_state,
                B_prev_state=B_prev_state,
                output_chunk=outputs.output_chunk,
                batch_size=batch_size,
                heads=heads,
                padded_time=padded_time,
                P=P,
                D=D,
                n_chunks=n_chunks,
                chunk_size=resolved_chunk_size,
                tc_dtype=tc_dtype,
                output_dtype=output_dtype,
                device_index=device_index,
                resolved_m_block_size=spec.config.m_block_size,
                resolved_n_block_size=spec.config.n_block_size,
                resolved_num_threads=spec.config.num_threads,
                has_prev=B_prev is not None,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: packaged(*runtime_artifacts.runtime_args),
                device=U.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_chunk_scan_outputs(
                    runtime_artifacts.outputs
                ),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = spec.config
    else:
        for config in chunk_scan_candidate_configs(
            P=P,
            D=D,
            chunk_size=resolved_chunk_size,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
        ):
            prepared = _make_chunk_scan_prepared_launch(
                U,
                M,
                K,
                B,
                C,
                chunk_starts,
                B_prev=B_prev,
                U_prev=U_prev,
                chunk_size=resolved_chunk_size,
                m_block_size=config.m_block_size,
                n_block_size=config.n_block_size,
                num_threads=config.num_threads,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
            latency_ms = benchmark_cuda_callable(
                lambda: prepared.compiled(*prepared.runtime_args),
                device=U.device,
                warmup_iterations=autotune_warmup_iterations(),
                timed_iterations=autotune_iterations(),
                before_iteration=lambda: _reset_chunk_scan_outputs(prepared.outputs),
            )
            if latency_ms < best_latency_ms:
                best_latency_ms = latency_ms
                best_config = config
    assert best_config is not None
    store_tuning_record(
        scope="chunk_scan",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=best_config.to_record(),
        metadata={
            "latency_ms": best_latency_ms,
            "candidate_count": len(candidate_specs)
            if candidate_specs
            else len(
                chunk_scan_candidate_configs(
                    P=P,
                    D=D,
                    chunk_size=resolved_chunk_size,
                    tc_dtype=tc_dtype,
                    output_dtype=output_dtype,
                    device_index=device_index,
                )
            ),
            "source": "packaged_aot" if candidate_specs else "jit",
        },
    )
    return best_config


def tune_v2x2ssd_fwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
) -> ForwardConfigBundle:
    resolved = _resolve_forward_autotune_bundle(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
    )
    if resolved is not None:
        return resolved

    input_info = _make_forward_input_info(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=None,
        n_block_size=64,
        scan_num_threads=128,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        validate_runtime_contract=True,
    )
    hardware = current_hardware_fingerprint(device_index=input_info.device_index)
    problem_key = forward_problem_key(
        tc_dtype=input_info.tc_dtype,
        output_dtype=output_dtype,
        P=input_info.P,
        D=input_info.D,
        chunk_size=input_info.chunk_size,
        has_prev=B_prev is not None,
        has_init=initial_states is not None,
        n_chunks=input_info.n_chunks,
    )
    if not autotune_force_retune():
        cached_record = lookup_tuning_record(
            scope="forward",
            hardware=hardware,
            problem_key=problem_key.to_record(),
        )
        if cached_record is not None and isinstance(cached_record.get("config"), dict):
            return ForwardConfigBundle.from_record(
                cast(dict[str, object], cached_record["config"])
            )
    best_bundle: ForwardConfigBundle | None = None
    best_latency_ms = float("inf")
    for config_bundle in forward_bundle_candidates(
        P=input_info.P,
        D=input_info.D,
        chunk_size=input_info.chunk_size,
        tc_dtype=input_info.tc_dtype,
        output_dtype=output_dtype,
        device_index=input_info.device_index,
        has_init=initial_states is not None,
    ):
        runtime_artifacts = _make_forward_runtime_artifacts(
            U,
            M,
            K,
            B,
            C,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            m_block_size=config_bundle.chunk_scan.m_block_size,
            n_block_size=config_bundle.chunk_scan.n_block_size,
            scan_num_threads=config_bundle.chunk_scan.num_threads,
            state_num_threads=config_bundle.state_passing.num_threads,
            state_vecs_per_thread=config_bundle.state_passing.vecs_per_thread,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            return_final_state=True,
            return_intermediates=True,
            prepared_inputs=None,
            validate_runtime_contract=True,
            config_bundle=config_bundle,
        )
        compiled = _get_compiled_v2x2ssd_fwd_kernel(
            U=U,
            M=M,
            K=K,
            B=B,
            C=C,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            compile_artifacts=_make_forward_compile_artifacts_from_runtime_artifacts(
                runtime_artifacts
            ),
        )
        latency_ms = benchmark_cuda_callable(
            lambda: compiled(*runtime_artifacts.runtime_args),
            device=U.device,
            warmup_iterations=autotune_warmup_iterations(),
            timed_iterations=autotune_iterations(),
            before_iteration=lambda: _reset_forward_runtime_outputs(runtime_artifacts),
        )
        if latency_ms < best_latency_ms:
            best_latency_ms = latency_ms
            best_bundle = config_bundle
    assert best_bundle is not None
    store_tuning_record(
        scope="forward",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=best_bundle.to_record(),
        metadata={
            "latency_ms": best_latency_ms,
            "candidate_count": len(
                forward_bundle_candidates(
                    P=input_info.P,
                    D=input_info.D,
                    chunk_size=input_info.chunk_size,
                    tc_dtype=input_info.tc_dtype,
                    output_dtype=output_dtype,
                    device_index=input_info.device_index,
                    has_init=initial_states is not None,
                )
            ),
            "source": "jit",
        },
    )
    return best_bundle


def compile_chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    cta_tiler: tuple[int, int, int] | None = None,
    num_stages: int | None = None,
) -> object:
    """Compile the chunk-increment host wrapper with TVM FFI enabled."""
    _validate_chunk_increment_inputs(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
    )
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    compile_artifacts = _make_chunk_increment_compile_artifacts(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        has_prev=U_prev is not None,
        config=(
            None
            if cta_tiler is None and num_stages is None
            else ChunkIncrementConfig(
                cta_tiler=(
                    _resolve_chunk_increment_cta_tiler(D=int(B.shape[-1]))
                    if cta_tiler is None
                    else tuple(int(v) for v in cta_tiler)
                ),
                num_stages=3 if num_stages is None else int(num_stages),
            )
        ),
    )
    return _get_compiled_chunk_increment_kernel(
        problem_shape=compile_artifacts.problem_shape,
        config=compile_artifacts.config,
        compile_args=compile_artifacts.compile_args,
        cache_key=compile_artifacts.cache_key,
    )


def _make_chunk_increment_prepared_launch(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    cta_tiler: tuple[int, int, int] | None = None,
    num_stages: int | None = None,
) -> PreparedChunkIncrementLaunch:
    """Prepare runtime arguments and outputs for the forward chunk-increment kernel."""
    batch_size, heads, time_steps, P, D = _validate_chunk_increment_inputs(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
    )

    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    U_tc = _pad_zero_time(U, T_pad=padded_time, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=padded_time, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=padded_time)
    K_f = _pad_zero_time(K, T_pad=padded_time, dtype=torch.float32)
    U_tc = _ensure_min_alignment(U_tc, min_align=16)
    B_tc = _ensure_min_alignment(B_tc, min_align=16)

    if U_prev is None:
        U_prev_state, B_prev_state = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
    else:
        U_prev_state = U_prev.to(dtype=tc_dtype).contiguous()
        B_prev_state = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev_state = _ensure_min_alignment(U_prev_state, min_align=16)
        B_prev_state = _ensure_min_alignment(B_prev_state, min_align=16)

    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks

    inc_chunk = torch.zeros(
        (batch_head_chunk_count, P, D), device=U.device, dtype=torch.float32
    )
    chunk_multiplier_storage = torch.zeros(
        (batch_head_chunk_count, 2), device=U.device, dtype=torch.float32
    )

    runtime_artifacts = _make_chunk_increment_runtime_artifacts(
        U=U,
        M=M,
        K=K,
        B=B,
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev_state=U_prev_state,
        B_prev_state=B_prev_state,
        inc_chunk=inc_chunk,
        chunk_multiplier_storage=chunk_multiplier_storage,
        chunk_size=resolved_chunk_size,
        compute_dtype=compute_dtype,
        has_prev=U_prev is not None,
        config=(
            None
            if cta_tiler is None and num_stages is None
            else ChunkIncrementConfig(
                cta_tiler=(
                    _resolve_chunk_increment_cta_tiler(D=D)
                    if cta_tiler is None
                    else tuple(int(v) for v in cta_tiler)
                ),
                num_stages=3 if num_stages is None else int(num_stages),
            )
        ),
    )
    compiled = _get_compiled_chunk_increment_kernel(
        problem_shape=runtime_artifacts.problem_shape,
        config=runtime_artifacts.config,
        compile_args=runtime_artifacts.compile_args,
        cache_key=runtime_artifacts.cache_key,
    )
    return PreparedChunkIncrementLaunch(
        compiled=compiled,
        runtime_args=runtime_artifacts.runtime_args,
        outputs=runtime_artifacts.outputs,
    )


def chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    cta_tiler: tuple[int, int, int] | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin public wrapper over the compiled forward chunk-increment kernel."""
    prepared = _make_chunk_increment_prepared_launch(
        U,
        M,
        K,
        B,
        U_prev=U_prev,
        B_prev=B_prev,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        cta_tiler=cta_tiler,
        num_stages=num_stages,
    )
    prepared.compiled(*prepared.runtime_args)
    _record_tensors_on_current_stream(*prepared.runtime_args)
    inc_chunk = prepared.outputs.increment_chunk
    chunk_multiplier_storage = prepared.outputs.chunk_multiplier_storage
    batch_head_count = int(U.shape[0]) * int(U.shape[1])
    n_chunks = inc_chunk.shape[0] // batch_head_count
    return (
        inc_chunk.reshape(U.shape[0], U.shape[1], n_chunks, U.shape[-1], B.shape[-1]),
        chunk_multiplier_storage.reshape(U.shape[0], U.shape[1], n_chunks, 2),
    )


def compile_state_passing_cute(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
) -> object:
    """Compile the state-passing host wrapper with TVM FFI enabled."""
    _validate_state_passing_inputs(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
    )
    compile_artifacts = _make_state_passing_compile_artifacts(
        increment,
        chunk_multiplier,
        has_init=initial_states is not None,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    return _get_compiled_state_passing_kernel(
        problem_shape=compile_artifacts.problem_shape,
        launch_cfg=compile_artifacts.launch_cfg,
        compile_args=compile_artifacts.compile_args,
        cache_key=compile_artifacts.cache_key,
    )


def _make_state_passing_prepared_launch(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
) -> PreparedStatePassingLaunch:
    """Prepare runtime arguments and outputs for the forward state-passing kernel."""
    B, H, C, P, D = _validate_state_passing_inputs(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
    )

    chunk_starts = torch.empty(
        (B, H, C, P, D), device=increment.device, dtype=torch.float32
    )
    final_state = torch.empty(
        (B, H, P, D), device=increment.device, dtype=torch.float32
    )
    # The state-passing kernel defines the live state domain but may leave
    # untouched lanes outside that domain on issue-9-style shapes. The public
    # final-state output must not depend on stale allocator contents.
    final_state.zero_()

    increment_contig = increment.contiguous()
    chunk_multiplier_contig = chunk_multiplier.contiguous()
    initial_state_contig = (
        initial_states.contiguous() if initial_states is not None else None
    )

    if initial_state_contig is None:
        initial_state_arg = increment_contig
        has_init = False
    else:
        initial_state_arg = initial_state_contig
        has_init = True

    runtime_artifacts = _make_state_passing_runtime_artifacts(
        increment=increment_contig,
        chunk_multiplier=chunk_multiplier_contig,
        chunk_starts=chunk_starts,
        final_state=final_state,
        initial_state_arg=initial_state_arg,
        has_init=has_init,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    compiled = _get_compiled_state_passing_kernel(
        problem_shape=runtime_artifacts.problem_shape,
        launch_cfg=runtime_artifacts.launch_cfg,
        compile_args=runtime_artifacts.compile_args,
        cache_key=runtime_artifacts.cache_key,
    )
    return PreparedStatePassingLaunch(
        compiled=compiled,
        runtime_args=runtime_artifacts.runtime_args,
        outputs=runtime_artifacts.outputs,
    )


def state_passing_cute(
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
    return_final_state: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Thin public wrapper over the compiled forward state-passing kernel."""
    prepared = _make_state_passing_prepared_launch(
        increment,
        chunk_multiplier,
        initial_states=initial_states,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    prepared.compiled(*prepared.runtime_args)
    _record_tensors_on_current_stream(*prepared.runtime_args)
    if not return_final_state:
        return prepared.outputs.chunk_starts
    return prepared.outputs.chunk_starts, prepared.outputs.final_state


def compile_chunk_scan_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_size: int = 64,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    num_threads: int = 128,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> object:
    """Compile the chunk-scan host wrapper with TVM FFI enabled."""
    batch_size, heads, _time_steps, P, D, _n_chunks = _validate_chunk_scan_inputs(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    compile_artifacts = _make_chunk_scan_compile_artifacts(
        U,
        B,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        device_index=device_index,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        has_prev=B_prev is not None,
    )
    return _get_compiled_chunk_scan_kernel(
        problem_shape=compile_artifacts.problem_shape,
        launch_cfg=compile_artifacts.launch_cfg,
        compile_args=compile_artifacts.compile_args,
        cache_key=compile_artifacts.cache_key,
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        output_dtype=output_dtype,
        device_index=device_index,
        D=D,
        P=P,
        chunk_size=int(chunk_size),
    )


def _make_chunk_scan_prepared_launch(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_size: int = 64,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    num_threads: int = 128,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> PreparedChunkScanLaunch:
    """Prepare runtime arguments and outputs for the end-to-end forward chunk-scan kernel."""
    batch_size, heads, time_steps, P, D, n_chunks = _validate_chunk_scan_inputs(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=output_dtype,
    )

    resolved_chunk_size = int(chunk_size)
    padded_time = n_chunks * resolved_chunk_size
    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    resolved_m_block_size, resolved_n_block_size, resolved_num_threads = (
        _resolve_chunk_scan_launch_cfg(
            D=D,
            P=P,
            chunk_size=resolved_chunk_size,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
            requested_m_block_size=m_block_size,
            requested_n_block_size=int(n_block_size),
            requested_num_threads=int(num_threads),
        )
    )

    U_tc = _pad_zero_time(U, T_pad=padded_time, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=padded_time, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=padded_time, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=padded_time)
    K_f = _pad_zero_time(K, T_pad=padded_time, dtype=torch.float32)
    U_tc = _ensure_min_alignment(U_tc, min_align=16)
    B_tc = _ensure_min_alignment(B_tc, min_align=16)
    C_tc = _ensure_min_alignment(C_tc, min_align=16)
    U_scan = _guard_prev_time_base(U_tc, min_align=16)
    B_scan = _guard_prev_time_base(B_tc, min_align=16)

    if B_prev is None:
        U_prev_state, B_prev_state = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
    else:
        B_prev_state = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev_state = U_prev.to(dtype=tc_dtype).contiguous()
        B_prev_state = _ensure_min_alignment(B_prev_state, min_align=16)
        U_prev_state = _ensure_min_alignment(U_prev_state, min_align=16)

    chunk_starts_c = chunk_starts.contiguous()
    chunk_starts_c = _ensure_min_alignment(chunk_starts_c, min_align=16)
    out_chunk = torch.empty(
        (batch_head_chunk_count, resolved_chunk_size, 1, P),
        device=U.device,
        dtype=output_dtype,
    )
    runtime_artifacts = _make_chunk_scan_runtime_artifacts(
        U_scan=U_scan,
        B_scan=B_scan,
        C_tc=C_tc,
        M_f=M_f,
        K_f=K_f,
        chunk_starts=chunk_starts_c,
        U_prev_state=U_prev_state,
        B_prev_state=B_prev_state,
        output_chunk=out_chunk,
        batch_size=batch_size,
        heads=heads,
        padded_time=padded_time,
        P=P,
        D=D,
        n_chunks=n_chunks,
        chunk_size=resolved_chunk_size,
        tc_dtype=tc_dtype,
        output_dtype=output_dtype,
        device_index=device_index,
        resolved_m_block_size=resolved_m_block_size,
        resolved_n_block_size=resolved_n_block_size,
        resolved_num_threads=resolved_num_threads,
        has_prev=B_prev is not None,
    )

    compiled = _get_compiled_chunk_scan_kernel(
        problem_shape=runtime_artifacts.problem_shape,
        launch_cfg=runtime_artifacts.launch_cfg,
        compile_args=runtime_artifacts.compile_args,
        cache_key=runtime_artifacts.cache_key,
        tc_dtype=tc_dtype,
        output_dtype=output_dtype,
        device_index=device_index,
        D=D,
        P=P,
        chunk_size=resolved_chunk_size,
    )

    return PreparedChunkScanLaunch(
        compiled=compiled,
        runtime_args=runtime_artifacts.runtime_args,
        outputs=ChunkScanOutputs(
            output_chunk=runtime_artifacts.outputs.output_chunk,
            output_view=runtime_artifacts.outputs.output_view[:, :, :time_steps, :],
        ),
    )


def chunk_scan_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    chunk_size: int = 64,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    num_threads: int = 128,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Thin public wrapper over the compiled forward chunk-scan kernel."""
    prepared = _make_chunk_scan_prepared_launch(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )
    prepared.compiled(*prepared.runtime_args)
    _record_tensors_on_current_stream(*prepared.runtime_args)
    return prepared.outputs.output_view


def _make_forward_input_info(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    m_block_size: int | None,
    n_block_size: int,
    scan_num_threads: int,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    validate_runtime_contract: bool,
) -> ForwardInputInfo:
    if validate_runtime_contract:
        if U.device.type != "cuda":
            raise ValueError("CUDA tensor required.")
        if (B_prev is None) ^ (U_prev is None):
            raise ValueError(
                "B_prev and U_prev must be passed together (or both omitted)."
            )
        if U.dtype != B.dtype or U.dtype != C.dtype:
            raise ValueError("U/B/C must share dtype.")
        if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("U/B/C must be float16/bfloat16/float32.")
        if M.dtype != torch.float32 or K.dtype != torch.float32:
            raise TypeError("M and K must be float32.")
        if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("output_dtype must be float16/bfloat16/float32.")

    batch_size, heads, time_steps, P = map(int, U.shape)
    D = int(B.shape[-1])
    if validate_runtime_contract:
        if B.shape != (batch_size, heads, time_steps, D) or C.shape != (
            batch_size,
            heads,
            time_steps,
            D,
        ):
            raise ValueError("B/C must be (B,H,T,D) matching U.")
        if M.shape != (batch_size, heads, time_steps, 2):
            raise ValueError(
                f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}."
            )
        if K.shape != (batch_size, heads, time_steps, 2, 2):
            raise ValueError(
                f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
            )
        if D % 2 != 0:
            raise ValueError("B/C last dim must be divisible by 2.")
        if initial_states is not None:
            if not torch.is_floating_point(initial_states):
                raise TypeError("initial_states must be floating-point.")
            if tuple(initial_states.shape) != (batch_size, heads, P, D):
                raise ValueError(
                    "initial_states must be (B,H,P,D) matching scan inputs."
                )
            if initial_states.device != U.device:
                raise ValueError("initial_states must be on the same device as U.")
        if B_prev is not None:
            if not torch.is_floating_point(B_prev) or not torch.is_floating_point(
                U_prev
            ):
                raise TypeError("B_prev and U_prev must be floating-point.")
            if tuple(B_prev.shape) != (batch_size, heads, D) or tuple(U_prev.shape) != (
                batch_size,
                heads,
                P,
            ):
                raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
            if B_prev.device != U.device or U_prev.device != U.device:
                raise ValueError("B_prev and U_prev must be on the same device as U.")

    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size
    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
    )
    resolved_m_block, resolved_n_block, resolved_scan_num_threads = (
        _resolve_chunk_scan_launch_cfg(
            D=D,
            P=P,
            chunk_size=resolved_chunk_size,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
            requested_m_block_size=m_block_size,
            requested_n_block_size=int(n_block_size),
            requested_num_threads=int(scan_num_threads),
        )
    )
    return ForwardInputInfo(
        batch_size=batch_size,
        heads=heads,
        time_steps=time_steps,
        P=P,
        D=D,
        chunk_size=resolved_chunk_size,
        n_chunks=n_chunks,
        padded_time=padded_time,
        tc_dtype=tc_dtype,
        device_index=device_index,
        resolved_m_block=resolved_m_block,
        resolved_n_block=resolved_n_block,
        resolved_scan_num_threads=resolved_scan_num_threads,
    )


def _make_forward_runtime_artifacts(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    m_block_size: int | None,
    n_block_size: int,
    scan_num_threads: int,
    state_num_threads: int,
    state_vecs_per_thread: int,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    return_final_state: bool = False,
    return_intermediates: bool = True,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    validate_runtime_contract: bool = True,
    config_bundle: ForwardConfigBundle | None = None,
) -> ForwardRuntimeArtifacts:
    requested_scan_config = (
        config_bundle.chunk_scan if config_bundle is not None else None
    )
    input_info = _make_forward_input_info(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=(
            requested_scan_config.m_block_size
            if requested_scan_config is not None
            else m_block_size
        ),
        n_block_size=(
            requested_scan_config.n_block_size
            if requested_scan_config is not None
            else n_block_size
        ),
        scan_num_threads=(
            requested_scan_config.num_threads
            if requested_scan_config is not None
            else scan_num_threads
        ),
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        validate_runtime_contract=validate_runtime_contract,
    )
    batch_size = input_info.batch_size
    heads = input_info.heads
    time_steps = input_info.time_steps
    P = input_info.P
    D = input_info.D
    resolved_chunk_size = input_info.chunk_size
    n_chunks = input_info.n_chunks
    padded_time = input_info.padded_time
    tc_dtype = input_info.tc_dtype
    resolved_m_block = input_info.resolved_m_block
    resolved_n_block = input_info.resolved_n_block
    resolved_scan_num_threads = input_info.resolved_scan_num_threads
    resolved_chunk_increment_config = (
        _resolve_default_chunk_increment_config(
            D=D,
            chunk_size=resolved_chunk_size,
        )
        if config_bundle is None
        else _normalize_chunk_increment_config(
            chunk_size=resolved_chunk_size,
            config=config_bundle.chunk_increment,
        )
    )
    resolved_state_config = (
        StatePassingConfig(
            num_threads=int(state_num_threads),
            vecs_per_thread=int(state_vecs_per_thread),
        )
        if config_bundle is None
        else config_bundle.state_passing
    )
    resolved_scan_config = ChunkScanConfig(
        m_block_size=resolved_m_block,
        n_block_size=resolved_n_block,
        num_threads=resolved_scan_num_threads,
    )

    if prepared_inputs is None:
        U_tc = _prepare_time_operand(U, padded_time=padded_time, dtype=tc_dtype)
        M_f = _prepare_m_operand(M, padded_time=padded_time)
        K_f = _prepare_time_operand(K, padded_time=padded_time, dtype=torch.float32)
        B_tc = _prepare_time_operand(B, padded_time=padded_time, dtype=tc_dtype)
        C_tc = _prepare_time_operand(C, padded_time=padded_time, dtype=tc_dtype)
    else:
        U_tc, M_f, K_f, B_tc, C_tc = prepared_inputs
        expected_u_shape = (batch_size, heads, padded_time, P)
        expected_b_shape = (batch_size, heads, padded_time, D)
        expected_m_shape = (batch_size, heads, padded_time, 2)
        expected_k_shape = (batch_size, heads, padded_time, 2, 2)
        if (
            tuple(U_tc.shape) != expected_u_shape
            or U_tc.dtype != tc_dtype
            or not U_tc.is_contiguous()
        ):
            raise ValueError("prepared U_tc must match padded scan input layout.")
        if (
            tuple(B_tc.shape) != expected_b_shape
            or B_tc.dtype != tc_dtype
            or not B_tc.is_contiguous()
        ):
            raise ValueError("prepared B_tc must match padded scan input layout.")
        if (
            tuple(C_tc.shape) != expected_b_shape
            or C_tc.dtype != tc_dtype
            or not C_tc.is_contiguous()
        ):
            raise ValueError("prepared C_tc must match padded scan input layout.")
        if (
            tuple(M_f.shape) != expected_m_shape
            or M_f.dtype != torch.float32
            or not M_f.is_contiguous()
        ):
            raise ValueError("prepared M_f must match padded scan parameter layout.")
        if (
            tuple(K_f.shape) != expected_k_shape
            or K_f.dtype != torch.float32
            or not K_f.is_contiguous()
        ):
            raise ValueError("prepared K_f must match padded scan parameter layout.")

    if B_prev is None:
        U_prev_state, B_prev_state = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
    else:
        U_prev_state = U_prev.to(dtype=tc_dtype).contiguous()
        B_prev_state = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev_state = _ensure_min_alignment(U_prev_state, min_align=16)
        B_prev_state = _ensure_min_alignment(B_prev_state, min_align=16)

    inc_chunk = _get_fwd_workspace(
        device=U.device,
        batch_size=batch_size,
        heads=heads,
        n_chunks=n_chunks,
        P=P,
        D=D,
    )
    inc_chunk.zero_()
    increment = inc_chunk.reshape(batch_size, heads, n_chunks, P, D)
    if initial_states is None:
        initial_state0 = _get_zero_initial_state(
            device=U.device,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
        has_init = False
    else:
        initial_state0 = initial_states.to(dtype=torch.float32).contiguous()
        has_init = True
    final_state_workspace = None
    if return_final_state:
        final_state_workspace = torch.empty(
            (batch_size, heads, P, D), device=U.device, dtype=torch.float32
        )
        final_state_workspace.zero_()
        final_state = final_state_workspace
    else:
        final_state = _get_dummy_final_state(
            device=U.device,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
    if return_intermediates:
        chunk_multiplier_storage = torch.empty(
            (batch_size * heads * n_chunks, 2),
            device=U.device,
            dtype=torch.float32,
        )
        chunk_starts = torch.empty(
            (batch_size, heads, n_chunks, P, D),
            device=U.device,
            dtype=torch.float32,
        )
    else:
        chunk_multiplier_storage, chunk_starts = _get_no_grad_intermediates(
            device=U.device,
            batch_size=batch_size,
            heads=heads,
            n_chunks=n_chunks,
            P=P,
            D=D,
        )
    chunk_starts = _ensure_min_alignment(chunk_starts, min_align=16)
    chunk_multiplier_storage.zero_()
    chunk_multiplier = chunk_multiplier_storage.reshape(batch_size, heads, n_chunks, 2)
    out_chunk = torch.empty(
        (batch_size * heads * n_chunks, resolved_chunk_size, 1, P),
        device=U.device,
        dtype=output_dtype,
    )
    out_pad = out_chunk.reshape(
        batch_size,
        heads,
        n_chunks,
        resolved_chunk_size,
        1,
        P,
    ).reshape(batch_size, heads, padded_time, P)

    state_elems_per_thread = 2 * int(resolved_state_config.vecs_per_thread)
    state_tile_stride_elems = P * D
    state_copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc_chunk,
        tile_stride_elems=state_tile_stride_elems,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles(
        chunk_starts,
        tile_stride_elems=state_tile_stride_elems,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_state_in = 32
    state_copy_bits_state_out = 32

    U_scan = _guard_prev_time_base(U_tc, min_align=16)
    B_scan = _guard_prev_time_base(B_tc, min_align=16)
    chunk_increment_spec = _chunk_increment_tensor_specs(
        (batch_size, heads, padded_time, P, D, n_chunks, resolved_chunk_size)
    )
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
    )
    chunk_scan_spec = _chunk_scan_tensor_specs(
        (batch_size, heads, padded_time, P, D, n_chunks, resolved_chunk_size)
    )
    (
        u_increment_spec,
        b_increment_spec,
        m_increment_spec,
        k_increment_spec,
        u_prev_increment_spec,
        b_prev_increment_spec,
        increment_spec,
        chunk_multiplier_spec,
    ) = chunk_increment_spec
    (
        increment_state_spec,
        chunk_multiplier_state_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = state_passing_spec
    (
        u_scan_spec,
        b_scan_spec,
        m_scan_spec,
        k_scan_spec,
        z0_scan_spec,
        u_prev_scan_spec,
        b_prev_scan_spec,
        output_spec,
    ) = chunk_scan_spec
    runtime_args = (
        make_runtime_tensor_spec_view(U_tc, u_increment_spec),
        make_runtime_tensor_spec_view(B_tc, b_increment_spec),
        make_runtime_tensor_spec_view(M_f, m_increment_spec),
        make_runtime_tensor_spec_view(K_f, k_increment_spec),
        make_runtime_tensor_spec_view(U_prev_state, u_prev_increment_spec),
        make_runtime_tensor_spec_view(B_prev_state, b_prev_increment_spec),
        make_runtime_tensor_spec_view(inc_chunk, increment_spec),
        make_runtime_tensor_spec_view(chunk_multiplier_storage, chunk_multiplier_spec),
        make_runtime_tensor_spec_view(increment, increment_state_spec),
        make_runtime_tensor_spec_view(chunk_multiplier, chunk_multiplier_state_spec),
        make_runtime_tensor_spec_view(chunk_starts, chunk_starts_spec),
        make_runtime_tensor_spec_view(final_state, final_state_spec),
        make_runtime_tensor_spec_view(initial_state0, final_state_spec),
        make_runtime_tensor_spec_view(U_scan, u_scan_spec),
        make_runtime_tensor_spec_view(B_scan, b_scan_spec),
        make_runtime_tensor_spec_view(C_tc, b_scan_spec),
        make_runtime_tensor_spec_view(M_f, m_scan_spec),
        make_runtime_tensor_spec_view(K_f, k_scan_spec),
        make_runtime_tensor_spec_view(chunk_starts, z0_scan_spec),
        make_runtime_tensor_spec_view(U_prev_state, u_prev_scan_spec),
        make_runtime_tensor_spec_view(B_prev_state, b_prev_scan_spec),
        make_runtime_tensor_spec_view(out_chunk, output_spec),
    )
    alignments = tuple(_assumed_align(tensor) for tensor in runtime_args)
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        resolved_chunk_size,
    )
    launch_cfg = (
        resolved_m_block,
        resolved_n_block,
        resolved_scan_num_threads,
        int(resolved_state_config.num_threads),
        int(resolved_state_config.vecs_per_thread),
        int(state_copy_bits_in),
        int(state_copy_bits_out),
        int(state_copy_bits_state_in),
        int(state_copy_bits_state_out),
        has_init,
    )
    return ForwardRuntimeArtifacts(
        runtime_args=runtime_args,
        alignments=alignments,
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config_bundle=ForwardConfigBundle(
            chunk_increment=resolved_chunk_increment_config,
            state_passing=resolved_state_config,
            chunk_scan=resolved_scan_config,
        ),
        outputs=ForwardOutputs(
            output=out_pad[:, :, :time_steps, :],
            final_state=final_state_workspace,
            chunk_multiplier=chunk_multiplier,
            chunk_starts=chunk_starts,
        ),
    )


def _make_forward_compile_artifacts(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    m_block_size: int | None,
    n_block_size: int,
    scan_num_threads: int,
    state_num_threads: int,
    state_vecs_per_thread: int,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    config_bundle: ForwardConfigBundle | None = None,
) -> ForwardCompileArtifacts:
    requested_scan_config = (
        config_bundle.chunk_scan if config_bundle is not None else None
    )
    input_info = _make_forward_input_info(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=(
            requested_scan_config.m_block_size
            if requested_scan_config is not None
            else m_block_size
        ),
        n_block_size=(
            requested_scan_config.n_block_size
            if requested_scan_config is not None
            else n_block_size
        ),
        scan_num_threads=(
            requested_scan_config.num_threads
            if requested_scan_config is not None
            else scan_num_threads
        ),
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        validate_runtime_contract=True,
    )
    batch_size = input_info.batch_size
    heads = input_info.heads
    P = input_info.P
    D = input_info.D
    resolved_chunk_size = input_info.chunk_size
    n_chunks = input_info.n_chunks
    padded_time = input_info.padded_time
    tc_dtype = input_info.tc_dtype
    resolved_chunk_increment_config = (
        _resolve_default_chunk_increment_config(
            D=D,
            chunk_size=resolved_chunk_size,
        )
        if config_bundle is None
        else _normalize_chunk_increment_config(
            chunk_size=resolved_chunk_size,
            config=config_bundle.chunk_increment,
        )
    )
    resolved_state_config = (
        StatePassingConfig(
            num_threads=int(state_num_threads),
            vecs_per_thread=int(state_vecs_per_thread),
        )
        if config_bundle is None
        else config_bundle.state_passing
    )
    resolved_scan_config = ChunkScanConfig(
        m_block_size=input_info.resolved_m_block,
        n_block_size=input_info.resolved_n_block,
        num_threads=input_info.resolved_scan_num_threads,
    )
    state_tile_stride_elems = P * D
    state_elems_per_thread = 2 * int(resolved_state_config.vecs_per_thread)
    state_assumed_align = _compile_min_align_for_dtype(torch.float32)
    state_copy_bits_in = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=state_assumed_align,
        tile_stride_elems=state_tile_stride_elems,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles_from_properties(
        dtype=torch.float32,
        assumed_align=state_assumed_align,
        tile_stride_elems=state_tile_stride_elems,
        elems_per_thread=state_elems_per_thread,
    )
    state_copy_bits_state_in = 32
    state_copy_bits_state_out = 32
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        resolved_chunk_size,
    )
    launch_cfg = (
        input_info.resolved_m_block,
        input_info.resolved_n_block,
        input_info.resolved_scan_num_threads,
        int(resolved_state_config.num_threads),
        int(resolved_state_config.vecs_per_thread),
        int(state_copy_bits_in),
        int(state_copy_bits_out),
        int(state_copy_bits_state_in),
        int(state_copy_bits_state_out),
        initial_states is not None,
    )

    chunk_increment_spec = _chunk_increment_tensor_specs(problem_shape)
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
    )
    chunk_scan_spec = _chunk_scan_tensor_specs(problem_shape)
    (
        u_increment_spec,
        b_increment_spec,
        m_increment_spec,
        k_increment_spec,
        u_prev_increment_spec,
        b_prev_increment_spec,
        increment_spec,
        chunk_multiplier_spec,
    ) = chunk_increment_spec
    (
        increment_state_spec,
        chunk_multiplier_state_spec,
        chunk_starts_spec,
        final_state_spec,
    ) = state_passing_spec
    (
        u_scan_spec,
        b_scan_spec,
        m_scan_spec,
        k_scan_spec,
        z0_scan_spec,
        u_prev_scan_spec,
        b_prev_scan_spec,
        output_spec,
    ) = chunk_scan_spec
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    tc_align = _compile_min_align_for_dtype(tc_dtype)
    out_align = _compile_min_align_for_dtype(output_dtype)
    alignments, compile_args = _make_tvm_ffi_compile_args_from_specs(
        (tc_dtype, u_increment_spec, tc_align),
        (tc_dtype, b_increment_spec, tc_align),
        (torch.float32, m_increment_spec, fp32_align),
        (torch.float32, k_increment_spec, fp32_align),
        (tc_dtype, u_prev_increment_spec, tc_align),
        (tc_dtype, b_prev_increment_spec, tc_align),
        (torch.float32, increment_spec, fp32_align),
        (torch.float32, chunk_multiplier_spec, fp32_align),
        (torch.float32, increment_state_spec, fp32_align),
        (torch.float32, chunk_multiplier_state_spec, fp32_align),
        (torch.float32, chunk_starts_spec, fp32_align),
        (torch.float32, final_state_spec, fp32_align),
        (torch.float32, final_state_spec, fp32_align),
        (tc_dtype, u_scan_spec, tc_align),
        (tc_dtype, b_scan_spec, tc_align),
        (tc_dtype, b_scan_spec, tc_align),
        (torch.float32, m_scan_spec, fp32_align),
        (torch.float32, k_scan_spec, fp32_align),
        (torch.float32, z0_scan_spec, fp32_align),
        (tc_dtype, u_prev_scan_spec, tc_align),
        (tc_dtype, b_prev_scan_spec, tc_align),
        (output_dtype, output_spec, out_align),
    )
    return ForwardCompileArtifacts(
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        config_bundle=ForwardConfigBundle(
            chunk_increment=resolved_chunk_increment_config,
            state_passing=resolved_state_config,
            chunk_scan=resolved_scan_config,
        ),
        compile_args=compile_args,
        alignments=alignments,
    )


def _make_forward_compile_artifacts_from_runtime_artifacts(
    runtime_artifacts: ForwardRuntimeArtifacts,
) -> ForwardCompileArtifacts:
    _, compile_args = _make_tvm_ffi_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return ForwardCompileArtifacts(
        problem_shape=runtime_artifacts.problem_shape,
        launch_cfg=runtime_artifacts.launch_cfg,
        config_bundle=runtime_artifacts.config_bundle,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
    )


def _get_compiled_v2x2ssd_fwd_kernel(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    compile_artifacts: ForwardCompileArtifacts,
):
    from slinoss.ops.v2x2ssd.cute.aot import (
        ForwardAOTSpec,
        try_load_packaged_v2x2ssd_fwd_function,
    )

    cache_key = _fwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        out_dtype=output_dtype,
        problem_shape=compile_artifacts.problem_shape,
        config_bundle=compile_artifacts.config_bundle,
        launch_cfg=compile_artifacts.launch_cfg,
        alignments=compile_artifacts.alignments,
    )
    compiled = _FWD_HOST_CACHE.get(cache_key)
    if compiled is not None:
        note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=True)
        return compiled

    forward_aot_spec = ForwardAOTSpec(
        arch_tag="any",
        P=int(compile_artifacts.problem_shape[3]),
        D=int(compile_artifacts.problem_shape[4]),
        chunk_size=int(compile_artifacts.problem_shape[6]),
        tc_dtype_name={
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[_tc_input_dtype(U.dtype, compute_dtype)],
        output_dtype_name={
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[output_dtype],
        config_bundle=compile_artifacts.config_bundle,
        has_init=bool(compile_artifacts.launch_cfg[-1]),
    )
    packaged = try_load_packaged_v2x2ssd_fwd_function(forward_aot_spec)
    if packaged is not None:
        note_cache_event("cute.v2x2ssd.fwd.host_aot", hit=True)
        _FWD_HOST_CACHE[cache_key] = packaged
        return packaged

    note_cache_event("cute.v2x2ssd.fwd.host_aot", hit=False)
    note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=False)
    host_wrapper = _make_v2x2ssd_fwd_host_wrapper(
        problem_shape=compile_artifacts.problem_shape,
        config_bundle=compile_artifacts.config_bundle,
        launch_cfg=compile_artifacts.launch_cfg,
    )
    compiled = cute.compile(
        host_wrapper, *compile_artifacts.compile_args, options="--enable-tvm-ffi"
    )
    _FWD_HOST_CACHE[cache_key] = compiled
    return compiled


def _matching_packaged_forward_aot_specs(
    *,
    device_index: int,
    problem_shape: tuple[int, ...],
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    has_init: bool,
) -> tuple[object, ...]:
    from slinoss.ops.v2x2ssd.cute.aot import list_packaged_forward_aot_specs

    hardware = current_hardware_fingerprint(device_index=device_index)
    P = int(problem_shape[3])
    D = int(problem_shape[4])
    chunk_size = int(problem_shape[6])
    tc_dtype_name = str(tc_dtype).replace("torch.", "")
    output_dtype_name = str(output_dtype).replace("torch.", "")
    return tuple(
        spec
        for spec in list_packaged_forward_aot_specs(arch_tag=hardware.arch_tag)
        if spec.P == P
        and spec.D == D
        and spec.chunk_size == chunk_size
        and spec.tc_dtype_name == tc_dtype_name
        and spec.output_dtype_name == output_dtype_name
        and spec.has_init == has_init
        and ChunkScanFwdAmpere(
            D=spec.D,
            P=spec.P,
            L=spec.chunk_size,
            m_block_size=spec.config_bundle.chunk_scan.m_block_size,
            n_block_size=spec.config_bundle.chunk_scan.n_block_size,
            num_threads=spec.config_bundle.chunk_scan.num_threads,
        ).can_implement(
            _torch_to_cutlass_dtype(tc_dtype),
            _torch_to_cutlass_dtype(output_dtype),
            device_index=device_index,
        )
    )


def _reset_forward_runtime_outputs(runtime_artifacts: ForwardRuntimeArtifacts) -> None:
    runtime_artifacts.outputs.output.zero_()
    runtime_artifacts.outputs.chunk_multiplier.zero_()
    runtime_artifacts.outputs.chunk_starts.zero_()
    if runtime_artifacts.outputs.final_state is not None:
        runtime_artifacts.outputs.final_state.zero_()


def _benchmark_packaged_forward_aot_spec(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    spec,
) -> float | None:
    from slinoss.ops.v2x2ssd.cute.aot import try_load_packaged_v2x2ssd_fwd_function

    packaged = try_load_packaged_v2x2ssd_fwd_function(spec)
    if packaged is None:
        return None
    runtime_artifacts = _make_forward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=spec.config_bundle.chunk_scan.m_block_size,
        n_block_size=spec.config_bundle.chunk_scan.n_block_size,
        scan_num_threads=spec.config_bundle.chunk_scan.num_threads,
        state_num_threads=spec.config_bundle.state_passing.num_threads,
        state_vecs_per_thread=spec.config_bundle.state_passing.vecs_per_thread,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=True,
        return_intermediates=True,
        prepared_inputs=None,
        validate_runtime_contract=True,
        config_bundle=spec.config_bundle,
    )
    return benchmark_cuda_callable(
        lambda: packaged(*runtime_artifacts.runtime_args),
        device=U.device,
        warmup_iterations=autotune_warmup_iterations(),
        timed_iterations=autotune_iterations(),
        before_iteration=lambda: _reset_forward_runtime_outputs(runtime_artifacts),
    )


def _resolve_forward_autotune_bundle(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> ForwardConfigBundle | None:
    if not autotune_enabled():
        return None

    input_info = _make_forward_input_info(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=None,
        n_block_size=64,
        scan_num_threads=128,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        validate_runtime_contract=True,
    )
    hardware = current_hardware_fingerprint(device_index=input_info.device_index)
    problem_key = forward_problem_key(
        tc_dtype=input_info.tc_dtype,
        output_dtype=output_dtype,
        P=input_info.P,
        D=input_info.D,
        chunk_size=input_info.chunk_size,
        has_prev=B_prev is not None,
        has_init=initial_states is not None,
        n_chunks=input_info.n_chunks,
    )
    if not autotune_force_retune():
        cached_record = lookup_tuning_record(
            scope="forward",
            hardware=hardware,
            problem_key=problem_key.to_record(),
        )
        if cached_record is not None and isinstance(cached_record.get("config"), dict):
            return ForwardConfigBundle.from_record(
                cast(dict[str, object], cached_record["config"])
            )

    candidate_specs = _matching_packaged_forward_aot_specs(
        device_index=input_info.device_index,
        problem_shape=(
            input_info.batch_size,
            input_info.heads,
            input_info.padded_time,
            input_info.P,
            input_info.D,
            input_info.n_chunks,
            input_info.chunk_size,
        ),
        tc_dtype=input_info.tc_dtype,
        output_dtype=output_dtype,
        has_init=initial_states is not None,
    )
    if not candidate_specs:
        return None

    best_spec = None
    best_latency_ms = float("inf")
    for spec in candidate_specs:
        latency_ms = _benchmark_packaged_forward_aot_spec(
            U=U,
            M=M,
            K=K,
            B=B,
            C=C,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            spec=spec,
        )
        if latency_ms is None or latency_ms >= best_latency_ms:
            continue
        best_latency_ms = latency_ms
        best_spec = spec
    if best_spec is None:
        return None

    store_tuning_record(
        scope="forward",
        hardware=hardware,
        problem_key=problem_key.to_record(),
        config_record=best_spec.config_bundle.to_record(),
        metadata={
            "latency_ms": best_latency_ms,
            "candidate_count": len(candidate_specs),
            "source": "packaged_aot",
        },
    )
    return best_spec.config_bundle


def _run_v2x2ssd_fwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    m_block_size: int | None,
    n_block_size: int,
    scan_num_threads: int,
    state_num_threads: int,
    state_vecs_per_thread: int,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    return_final_state: bool,
    return_intermediates: bool,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None,
    validate_runtime_contract: bool,
    config_bundle: ForwardConfigBundle | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    resolved_config_bundle = config_bundle
    explicit_launch_override = (
        config_bundle is not None
        or m_block_size is not None
        or int(n_block_size) != 64
        or int(scan_num_threads) != 128
        or int(state_num_threads) != 128
        or int(state_vecs_per_thread) != 8
    )
    if not explicit_launch_override:
        resolved_config_bundle = _resolve_forward_autotune_bundle(
            U=U,
            M=M,
            K=K,
            B=B,
            C=C,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )
    runtime_artifacts = _make_forward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=return_final_state,
        return_intermediates=return_intermediates,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=validate_runtime_contract,
        config_bundle=resolved_config_bundle,
    )
    compile_artifacts = _make_forward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    compiled = _get_compiled_v2x2ssd_fwd_kernel(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        compile_artifacts=compile_artifacts,
    )
    compiled(*runtime_artifacts.runtime_args)
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
    output = runtime_artifacts.outputs.output
    final_state = runtime_artifacts.outputs.final_state
    chunk_multiplier = runtime_artifacts.outputs.chunk_multiplier
    chunk_starts = runtime_artifacts.outputs.chunk_starts
    if not return_final_state:
        return output, chunk_multiplier, chunk_starts
    assert final_state is not None
    return output, final_state, chunk_multiplier, chunk_starts


def compile_v2x2ssd_fwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    scan_num_threads: int = 128,
    state_num_threads: int = 128,
    state_vecs_per_thread: int = 8,
    config_bundle: ForwardConfigBundle | None = None,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
) -> object:
    resolved_config_bundle = config_bundle
    explicit_launch_override = (
        config_bundle is not None
        or m_block_size is not None
        or int(n_block_size) != 64
        or int(scan_num_threads) != 128
        or int(state_num_threads) != 128
        or int(state_vecs_per_thread) != 8
    )
    if not explicit_launch_override:
        resolved_config_bundle = _resolve_forward_autotune_bundle(
            U=U,
            M=M,
            K=K,
            B=B,
            C=C,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )
    compile_artifacts = _make_forward_compile_artifacts(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        config_bundle=resolved_config_bundle,
    )
    return _get_compiled_v2x2ssd_fwd_kernel(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        compile_artifacts=compile_artifacts,
    )


def v2x2ssd_fwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    scan_num_threads: int = 128,
    state_num_threads: int = 128,
    state_vecs_per_thread: int = 8,
    config_bundle: ForwardConfigBundle | None = None,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    return_final_state: bool = False,
    return_intermediates: bool = True,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    return _run_v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        config_bundle=config_bundle,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=return_final_state,
        return_intermediates=return_intermediates,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=True,
    )


def _v2x2ssd_fwd_cute_prevalidated(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype = torch.float32,
    m_block_size: int | None = None,
    n_block_size: int = 64,
    scan_num_threads: int = 128,
    state_num_threads: int = 128,
    state_vecs_per_thread: int = 8,
    config_bundle: ForwardConfigBundle | None = None,
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    return_final_state: bool = False,
    return_intermediates: bool = True,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    return _run_v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        scan_num_threads=scan_num_threads,
        state_num_threads=state_num_threads,
        state_vecs_per_thread=state_vecs_per_thread,
        config_bundle=config_bundle,
        initial_states=initial_states,
        B_prev=B_prev,
        U_prev=U_prev,
        return_final_state=return_final_state,
        return_intermediates=return_intermediates,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=False,
    )


__all__ = [
    "chunk_increment_cute",
    "chunk_scan_cute",
    "compile_chunk_increment_cute",
    "compile_chunk_scan_cute",
    "compile_state_passing_cute",
    "compile_v2x2ssd_fwd_cute",
    "state_passing_cute",
    "tune_chunk_increment_cute",
    "tune_chunk_scan_cute",
    "tune_state_passing_cute",
    "tune_v2x2ssd_fwd_cute",
    "v2x2ssd_fwd_cute",
]
