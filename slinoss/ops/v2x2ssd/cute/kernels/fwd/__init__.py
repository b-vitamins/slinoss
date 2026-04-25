"""CuTe forward kernels for the ``v2x2ssd`` operator."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from cuda.bindings import driver as cuda
import cutlass.cute as cute
from typing import cast

from slinoss._cute_runtime import (
    launch_tvm_ffi_on_current_stream,
    make_runtime_tensor_spec_view,
    prepare_cached_tensors_on_current_stream,
)
from slinoss.ops._cute_common import (
    _compile_env_stream_placeholder,
    _is_cuda_graph_capturing,
    assumed_align,
    make_fake_tensor_arg,
    make_fake_tensor_spec_arg,
    make_row_major_stride,
    torch_to_cutlass_dtype,
)
from slinoss.ops.v2x2ssd.cute.tuning.bench import benchmark_cuda_callable
from slinoss.ops.v2x2ssd.cute.tuning.db import lookup_tuning_record, store_tuning_record
from slinoss.ops.v2x2ssd.cute.tuning.fwd import (
    autotune_enabled,
    autotune_force_retune,
    autotune_iterations,
    autotune_warmup_iterations,
    forward_problem_key,
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
    _choose_copy_bits_for_linear_tiles,
    _ensure_min_alignment,
    _guard_prev_time_base,
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
)
from .state_passing import StatePassingFwdAmpere


_FWD_HOST_CACHE: dict[tuple, object] = {}
_BOUNDARY_METADATA_CACHE: dict[tuple, object] = {}
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


def _raise_cold_capture_error(resource: str) -> None:
    raise RuntimeError(
        f"CuTe v2x2ssd forward {resource} is cold during CUDA graph capture. "
        "Warm the same forward spec once outside capture before graph capture."
    )


def _prepare_cached_forward_tensors(*tensors: torch.Tensor | None) -> None:
    prepare_cached_tensors_on_current_stream(*tensors)


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
    bc_groups: int
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
class BoundaryMetadataRuntimeArtifacts:
    problem_shape: tuple[int, ...]
    config: ChunkIncrementConfig
    launch_cfg: tuple[int, int, int, int, int, int, bool]
    compile_args: tuple[object, ...]
    runtime_args: tuple[torch.Tensor, ...]
    cache_key: tuple


def _cache_set(cache: dict, key: tuple, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _resolve_heads_per_group(*, heads: int, bc_groups: int, name: str) -> int:
    if bc_groups <= 0:
        raise ValueError(f"{name} must have a positive group dimension.")
    if heads % bc_groups != 0:
        raise ValueError(
            f"{name} group dimension must divide heads. Got heads={heads}, groups={bc_groups}."
        )
    return heads // bc_groups


def _get_zero_prev_tensors(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    heads: int,
    bc_groups: int,
    P: int,
    D: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        dtype,
        int(batch_size),
        int(heads),
        int(bc_groups),
        int(P),
        int(D),
    )
    cached = _ZERO_PREV_CACHE.get(key)
    if cached is None:
        if _is_cuda_graph_capturing(device):
            _raise_cold_capture_error("zero-prev cache")
        note_cache_event("cute.v2x2ssd.fwd.zero_prev", hit=False)
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, bc_groups, D), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
    else:
        note_cache_event("cute.v2x2ssd.fwd.zero_prev", hit=True)
    _prepare_cached_forward_tensors(*cached)
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
        if _is_cuda_graph_capturing(device):
            _raise_cold_capture_error("zero-initial-state cache")
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
    _prepare_cached_forward_tensors(cached)
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
        if _is_cuda_graph_capturing(device):
            _raise_cold_capture_error("dummy-final-state cache")
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
    _prepare_cached_forward_tensors(cached)
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
        _prepare_cached_forward_tensors(cached)
        return cached
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("workspace cache")
    note_cache_event("cute.v2x2ssd.fwd.workspace", hit=False)
    cached = torch.empty(
        (batch_size * heads * n_chunks, P, D), device=device, dtype=torch.float32
    )
    _cache_set(_FWD_WORKSPACE_CACHE, key, cached, limit=_FWD_WORKSPACE_CACHE_LIMIT)
    _prepare_cached_forward_tensors(cached)
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
        _prepare_cached_forward_tensors(*cached)
        return cached
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("no-grad-intermediate cache")
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
    _prepare_cached_forward_tensors(*cached)
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

    cutlass_tc_dtype = torch_to_cutlass_dtype(tc_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(output_dtype)
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


def _make_tensor_spec(
    shape: tuple[int, ...],
    *,
    stride: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = make_row_major_stride(shape)
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
    alignments = tuple(assumed_align(tensor) for tensor in runtime_args)
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
        tuple(assumed_align(tensor) for tensor in runtime_args)
        if alignments is None
        else tuple(int(align) for align in alignments)
    )
    compile_args = tuple(
        make_fake_tensor_arg(tensor, align=align)
        for tensor, align in zip(runtime_args, resolved_alignments, strict=True)
    ) + (_compile_env_stream_placeholder(),)
    return resolved_alignments, compile_args


def _make_tvm_ffi_compile_args_from_specs(
    *tensor_specs: tuple[torch.dtype, tuple[tuple[int, ...], tuple[int, ...]], int],
) -> tuple[tuple[int, ...], tuple[object, ...]]:
    alignments = tuple(int(align) for _, _, align in tensor_specs)
    compile_args = tuple(
        make_fake_tensor_spec_arg(
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
    batch_size, heads, padded_time, P, D, n_chunks, _chunk_size, bc_groups = (
        problem_shape
    )
    batch_head_count = batch_size * heads
    batch_group_count = batch_size * bc_groups
    batch_head_chunk_count = batch_head_count * n_chunks
    return (
        _make_tensor_spec(
            (P, padded_time, batch_head_count),
            stride=(1, P, padded_time * P),
        ),
        _make_tensor_spec(
            (D, padded_time, batch_group_count),
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
        _make_tensor_spec((D, batch_group_count), stride=(1, D)),
        _make_tensor_spec((P, D, batch_head_chunk_count), stride=(D, 1, P * D)),
        _make_tensor_spec((2, batch_head_chunk_count), stride=(1, 2)),
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
    batch_size, heads, _padded_time, P, D, n_chunks, chunk_size, bc_groups = (
        problem_shape
    )
    batch_head_count = batch_size * heads
    batch_group_count = batch_size * bc_groups
    batch_head_chunk_count = batch_head_count * n_chunks
    batch_group_chunk_count = batch_group_count * n_chunks
    return (
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 1, P)),
        _make_tensor_spec((batch_group_chunk_count, chunk_size, 1, D)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 2)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size, 2, 2)),
        _make_tensor_spec((batch_head_chunk_count, P, 1, D)),
        _make_tensor_spec((batch_head_count, P)),
        _make_tensor_spec((batch_group_count, D)),
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


def _boundary_metadata_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    problem_shape: tuple[int, ...],
    has_prev: bool,
    config: ChunkIncrementConfig,
    launch_cfg: tuple[int, int, int, int, int, int, bool],
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "boundary_metadata_fwd",
        device_index,
        tc_dtype,
        tuple(int(dim) for dim in problem_shape),
        has_prev,
        config.cache_key,
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


def _make_boundary_metadata_runtime_artifacts(
    *,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    U_prev_state: torch.Tensor,
    B_prev_state: torch.Tensor,
    inc_chunk: torch.Tensor,
    chunk_multiplier_storage: torch.Tensor,
    increment: torch.Tensor,
    chunk_multiplier: torch.Tensor,
    chunk_starts: torch.Tensor,
    final_state: torch.Tensor,
    initial_state_arg: torch.Tensor,
    chunk_size: int,
    tc_dtype: torch.dtype,
    has_init: bool,
    has_prev: bool,
    config_bundle: ForwardConfigBundle,
) -> BoundaryMetadataRuntimeArtifacts:
    batch_size, heads, n_chunks, P, D = map(int, increment.shape)
    bc_groups = int(B_tc.shape[1])
    padded_time = int(U_tc.shape[2])
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        int(chunk_size),
        bc_groups,
    )
    resolved_increment_config = _normalize_chunk_increment_config(
        tc_dtype=tc_dtype,
        chunk_size=int(chunk_size),
        config=config_bundle.chunk_increment,
    )
    chunk_increment_spec = _chunk_increment_tensor_specs(problem_shape)
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
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

    state_elem_count = P * D
    elems_per_thread = 2 * int(config_bundle.state_passing.vecs_per_thread)
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
        num_threads=config_bundle.state_passing.num_threads,
        vecs_per_thread=config_bundle.state_passing.vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        has_init=has_init,
    )
    runtime_args, alignments, compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            (U_tc, tc_dtype, u_increment_spec),
            (B_tc, tc_dtype, b_increment_spec),
            (M_f, torch.float32, m_increment_spec),
            (K_f, torch.float32, k_increment_spec),
            (U_prev_state, tc_dtype, u_prev_increment_spec),
            (B_prev_state, tc_dtype, b_prev_increment_spec),
            (inc_chunk, torch.float32, increment_spec),
            (chunk_multiplier_storage, torch.float32, chunk_multiplier_spec),
            (increment, torch.float32, increment_state_spec),
            (chunk_multiplier, torch.float32, chunk_multiplier_state_spec),
            (chunk_starts, torch.float32, chunk_starts_spec),
            (final_state, torch.float32, final_state_spec),
            (initial_state_arg, torch.float32, final_state_spec),
        )
    )
    cache_key = _boundary_metadata_key(
        device_index=increment.device.index
        if increment.device.index is not None
        else -1,
        tc_dtype=tc_dtype,
        problem_shape=problem_shape,
        has_prev=has_prev,
        config=resolved_increment_config,
        launch_cfg=launch_cfg,
        alignments=alignments,
    )
    return BoundaryMetadataRuntimeArtifacts(
        problem_shape=problem_shape,
        config=resolved_increment_config,
        launch_cfg=launch_cfg,
        compile_args=compile_args,
        runtime_args=runtime_args,
        cache_key=cache_key,
    )


def _recompute_boundary_metadata_prevalidated(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    config_bundle: ForwardConfigBundle,
    initial_states: torch.Tensor | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    increment_workspace: torch.Tensor,
    chunk_multiplier_workspace: torch.Tensor,
    chunk_starts_workspace: torch.Tensor,
    final_state_workspace: torch.Tensor,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    if prepared_inputs is None:
        U_tc = _prepare_time_operand(U, padded_time=padded_time, dtype=tc_dtype)
        M_f = _prepare_m_operand(M, padded_time=padded_time)
        K_f = _prepare_time_operand(K, padded_time=padded_time, dtype=torch.float32)
        B_tc = _prepare_time_operand(B, padded_time=padded_time, dtype=tc_dtype)
    else:
        U_tc, M_f, K_f, B_tc, _ = prepared_inputs
        expected_u_shape = (batch_size, heads, padded_time, P)
        expected_b_shape = (batch_size, int(B.shape[1]), padded_time, D)
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

    if increment_workspace.shape == (batch_size * heads * n_chunks, P, D):
        inc_chunk = increment_workspace
    elif increment_workspace.shape == (batch_size, heads, n_chunks, P, D):
        inc_chunk = increment_workspace.reshape(batch_size * heads * n_chunks, P, D)
    else:
        raise ValueError(
            "increment_workspace must be flat (B*H*C,P,D) or public (B,H,C,P,D)."
        )
    if chunk_multiplier_workspace.shape == (batch_size * heads * n_chunks, 2):
        chunk_multiplier_storage = chunk_multiplier_workspace
    elif chunk_multiplier_workspace.shape == (batch_size, heads, n_chunks, 2):
        chunk_multiplier_storage = chunk_multiplier_workspace.reshape(
            batch_size * heads * n_chunks,
            2,
        )
    else:
        raise ValueError(
            "chunk_multiplier_workspace must be flat (B*H*C,2) or public (B,H,C,2)."
        )
    if tuple(chunk_starts_workspace.shape) != (batch_size, heads, n_chunks, P, D):
        raise ValueError(
            "chunk_starts_workspace must be (B,H,C,P,D) matching the chunked state domain."
        )
    if tuple(final_state_workspace.shape) != (batch_size, heads, P, D):
        raise ValueError(
            "final_state_workspace must be (B,H,P,D) matching the boundary-state domain."
        )

    if B_prev is None:
        U_prev_state, B_prev_state = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=batch_size,
            heads=heads,
            bc_groups=int(B.shape[1]),
            P=P,
            D=D,
        )
    else:
        assert U_prev is not None
        U_prev_state = _ensure_min_alignment(
            U_prev.to(dtype=tc_dtype).contiguous(),
            min_align=16,
        )
        B_prev_state = _ensure_min_alignment(
            B_prev.to(dtype=tc_dtype).contiguous(),
            min_align=16,
        )

    increment_workspace.zero_()
    chunk_multiplier_storage.zero_()
    final_state_workspace.zero_()

    increment = inc_chunk.reshape(batch_size, heads, n_chunks, P, D)
    chunk_multiplier = chunk_multiplier_storage.reshape(batch_size, heads, n_chunks, 2)
    if initial_states is None:
        initial_state_arg = _get_zero_initial_state(
            device=U.device,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
        has_init = False
    else:
        initial_state_arg = initial_states.to(dtype=torch.float32).contiguous()
        has_init = True

    runtime_artifacts = _make_boundary_metadata_runtime_artifacts(
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev_state=U_prev_state,
        B_prev_state=B_prev_state,
        inc_chunk=inc_chunk,
        chunk_multiplier_storage=chunk_multiplier_storage,
        increment=increment,
        chunk_multiplier=chunk_multiplier,
        chunk_starts=chunk_starts_workspace,
        final_state=final_state_workspace,
        initial_state_arg=initial_state_arg,
        chunk_size=resolved_chunk_size,
        tc_dtype=tc_dtype,
        has_init=has_init,
        has_prev=U_prev is not None,
        config_bundle=config_bundle,
    )
    compiled = _get_compiled_boundary_metadata_kernel(
        runtime_artifacts=runtime_artifacts,
    )
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return chunk_multiplier, chunk_starts_workspace


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
    # The 64-wide N family keeps the epilogue on the same accumulator-coordinate
    # path for full and tail D tiles, avoiding the bank-heavy shared f32 output
    # staging needed by the older 96-wide family.
    return (64, 64, 32)


def _normalize_chunk_increment_config(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    config: ChunkIncrementConfig,
) -> ChunkIncrementConfig:
    normalized_kernel = ChunkIncrementFwdAmpere(
        torch_to_cutlass_dtype(tc_dtype),
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
    tc_dtype: torch.dtype,
    D: int,
    chunk_size: int,
) -> ChunkIncrementConfig:
    return _normalize_chunk_increment_config(
        tc_dtype=tc_dtype,
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
    bc_groups = int(B.shape[1])
    _resolve_heads_per_group(heads=heads, bc_groups=bc_groups, name="B")
    if B.shape != (batch_size, bc_groups, time_steps, D):
        raise ValueError("B must be (B,G,T,D) matching U on batch/time/width.")
    if M.shape != (batch_size, heads, time_steps, 2):
        raise ValueError(f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}.")
    if K.shape != (batch_size, heads, time_steps, 2, 2):
        raise ValueError(
            f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
        )
    if D % 2 != 0:
        raise ValueError("B last dim must be divisible by 2.")
    if U_prev is not None and (
        U_prev.shape != (batch_size, heads, P)
        or B_prev.shape != (batch_size, bc_groups, D)
    ):
        raise ValueError("U_prev/B_prev must be (B,H,P)/(B,G,D).")
    return batch_size, heads, time_steps, P, D


def _make_boundary_metadata_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    config: ChunkIncrementConfig,
    launch_cfg: tuple[int, ...],
):
    batch_size, heads, padded_time, P, D, n_chunks, chunk_size, bc_groups = (
        problem_shape
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
    chunk_increment_spec = _chunk_increment_tensor_specs(problem_shape)
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
    )
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
    def _boundary_metadata_host_wrapper(
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
        ) = chunk_increment_spec
        (
            increment_state_spec,
            chunk_multiplier_state_spec,
            chunk_starts_spec,
            final_state_spec,
        ) = state_passing_spec

        mU = _make_static_tensor_spec_view(U_increment_t, u_spec)
        mB = _make_static_tensor_spec_view(B_increment_t, b_spec)
        mM = _make_static_tensor_spec_view(M_increment_t, m_spec)
        mKprev = _make_static_tensor_spec_view(K_increment_t, k_spec)
        mKcurr = cute.make_tensor(mKprev.iterator + 2, mKprev.layout)
        mU_prev = _make_static_tensor_spec_view(U_prev_increment_t, u_prev_spec)
        mB_prev = _make_static_tensor_spec_view(B_prev_increment_t, b_prev_spec)
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

        increment_view = _make_static_tensor_spec_view(
            increment_state_t, increment_state_spec
        )
        chunk_multiplier_view = _make_static_tensor_spec_view(
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
            increment_view,
            chunk_multiplier_view,
            chunk_starts_view,
            final_state_view,
            initial_state_view,
            stream,
        )

    return _boundary_metadata_host_wrapper


def _make_v2x2ssd_fwd_host_wrapper(
    *,
    problem_shape: tuple[int, ...],
    config_bundle: ForwardConfigBundle,
    launch_cfg: tuple[int, ...],
):
    batch_size, heads, padded_time, P, D, n_chunks, chunk_size, bc_groups = (
        problem_shape
    )
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


def _get_compiled_boundary_metadata_kernel(
    *,
    runtime_artifacts: BoundaryMetadataRuntimeArtifacts,
):
    compiled = _BOUNDARY_METADATA_CACHE.get(runtime_artifacts.cache_key)
    if compiled is None:
        device = runtime_artifacts.runtime_args[0].device
        if _is_cuda_graph_capturing(device):
            _raise_cold_capture_error("boundary-metadata launcher cache")
        host_wrapper = _make_boundary_metadata_host_wrapper(
            problem_shape=runtime_artifacts.problem_shape,
            config=runtime_artifacts.config,
            launch_cfg=runtime_artifacts.launch_cfg,
        )
        compiled = cute.compile(
            host_wrapper,
            *runtime_artifacts.compile_args,
            options="--enable-tvm-ffi",
        )
        _BOUNDARY_METADATA_CACHE[runtime_artifacts.cache_key] = compiled
    return compiled


def _packaged_bc_groups_identity_for_problem_shape(
    problem_shape: tuple[int, ...],
) -> int | None:
    if len(problem_shape) != 8:
        raise ValueError(
            "v2x2ssd forward packaged identity expects an 8D problem shape. "
            f"Got {problem_shape}."
        )
    heads = int(problem_shape[1])
    bc_groups = int(problem_shape[7])
    if bc_groups <= 0:
        raise ValueError("bc_groups must be positive.")
    if heads % bc_groups != 0:
        raise ValueError(
            "Grouped BC packaged identity requires bc_groups to divide heads. "
            f"Got heads={heads}, bc_groups={bc_groups}."
        )
    return None if bc_groups == heads else bc_groups


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
    bc_groups = int(B.shape[1])
    if validate_runtime_contract:
        _resolve_heads_per_group(heads=heads, bc_groups=bc_groups, name="B/C")
        if B.shape != (batch_size, bc_groups, time_steps, D) or C.shape != (
            batch_size,
            bc_groups,
            time_steps,
            D,
        ):
            raise ValueError("B/C must be grouped as (B,G,T,D) and match each other.")
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
            if tuple(B_prev.shape) != (batch_size, bc_groups, D) or tuple(
                U_prev.shape
            ) != (
                batch_size,
                heads,
                P,
            ):
                raise ValueError("B_prev/U_prev must be (B,G,D)/(B,H,P).")
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
        bc_groups=bc_groups,
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
    bc_groups = input_info.bc_groups
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
            tc_dtype=tc_dtype,
            D=D,
            chunk_size=resolved_chunk_size,
        )
        if config_bundle is None
        else _normalize_chunk_increment_config(
            tc_dtype=tc_dtype,
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
        expected_b_shape = (batch_size, bc_groups, padded_time, D)
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
            bc_groups=bc_groups,
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
        (batch_size, heads, padded_time, P, D, n_chunks, resolved_chunk_size, bc_groups)
    )
    state_passing_spec = _state_passing_tensor_specs(
        (batch_size, heads, n_chunks, P, D)
    )
    chunk_scan_spec = _chunk_scan_tensor_specs(
        (batch_size, heads, padded_time, P, D, n_chunks, resolved_chunk_size, bc_groups)
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
    alignments = tuple(assumed_align(tensor) for tensor in runtime_args)
    problem_shape = (
        batch_size,
        heads,
        padded_time,
        P,
        D,
        n_chunks,
        resolved_chunk_size,
        bc_groups,
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
    bc_groups = input_info.bc_groups
    P = input_info.P
    D = input_info.D
    resolved_chunk_size = input_info.chunk_size
    n_chunks = input_info.n_chunks
    padded_time = input_info.padded_time
    tc_dtype = input_info.tc_dtype
    resolved_chunk_increment_config = (
        _resolve_default_chunk_increment_config(
            tc_dtype=tc_dtype,
            D=D,
            chunk_size=resolved_chunk_size,
        )
        if config_bundle is None
        else _normalize_chunk_increment_config(
            tc_dtype=tc_dtype,
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
        bc_groups,
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


def _make_forward_aot_runtime_args(
    runtime_args: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return runtime_args


def _make_packaged_v2x2ssd_fwd_callable(packaged: object):
    def _packaged_v2x2ssd_fwd_callable(*runtime_args):
        return packaged(*_make_forward_aot_runtime_args(runtime_args))

    return _packaged_v2x2ssd_fwd_callable


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

    device_index = (
        int(U.device.index)
        if U.device.index is not None
        else torch.cuda.current_device()
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
    if _is_cuda_graph_capturing(torch.device("cuda", device_index)):
        _raise_cold_capture_error("host launcher cache")

    packaged = None
    forward_aot_spec = ForwardAOTSpec(
        arch_tag=current_hardware_fingerprint(device_index=device_index).arch_tag,
        P=int(compile_artifacts.problem_shape[3]),
        D=int(compile_artifacts.problem_shape[4]),
        chunk_size=int(compile_artifacts.problem_shape[6]),
        bc_groups=_packaged_bc_groups_identity_for_problem_shape(
            compile_artifacts.problem_shape
        ),
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
        wrapped_packaged = _make_packaged_v2x2ssd_fwd_callable(packaged)
        _FWD_HOST_CACHE[cache_key] = wrapped_packaged
        return wrapped_packaged

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
    if len(problem_shape) != 8:
        return ()
    from slinoss.ops.v2x2ssd.cute.aot import list_packaged_forward_aot_specs

    hardware = current_hardware_fingerprint(device_index=device_index)
    P = int(problem_shape[3])
    D = int(problem_shape[4])
    chunk_size = int(problem_shape[6])
    bc_groups_identity = _packaged_bc_groups_identity_for_problem_shape(problem_shape)
    tc_dtype_name = str(tc_dtype).replace("torch.", "")
    output_dtype_name = str(output_dtype).replace("torch.", "")
    return tuple(
        spec
        for spec in list_packaged_forward_aot_specs(arch_tag=hardware.arch_tag)
        if spec.P == P
        and spec.D == D
        and spec.chunk_size == chunk_size
        and spec.bc_groups == bc_groups_identity
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
            torch_to_cutlass_dtype(tc_dtype),
            torch_to_cutlass_dtype(output_dtype),
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
    try:
        return benchmark_cuda_callable(
            lambda: packaged(*runtime_artifacts.runtime_args),
            device=U.device,
            warmup_iterations=autotune_warmup_iterations(),
            timed_iterations=autotune_iterations(),
            before_iteration=lambda: _reset_forward_runtime_outputs(runtime_artifacts),
        )
    except Exception:
        return None


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
            input_info.bc_groups,
        ),
        tc_dtype=input_info.tc_dtype,
        output_dtype=output_dtype,
        has_init=initial_states is not None,
    )
    if not candidate_specs:
        return None

    best_spec = None
    best_latency_ms = float("inf")
    skipped_invalid_candidates = 0
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
        if latency_ms is None:
            skipped_invalid_candidates += 1
            continue
        if latency_ms >= best_latency_ms:
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
            "skipped_invalid_candidates": skipped_invalid_candidates,
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
    runtime_artifacts = _v2x2ssd_fwd_runtime_artifacts_prevalidated(
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
        config_bundle=config_bundle,
    )
    output = runtime_artifacts.outputs.output
    final_state = runtime_artifacts.outputs.final_state
    chunk_multiplier = runtime_artifacts.outputs.chunk_multiplier
    chunk_starts = runtime_artifacts.outputs.chunk_starts
    if not return_final_state:
        return output, chunk_multiplier, chunk_starts
    assert final_state is not None
    return output, final_state, chunk_multiplier, chunk_starts


def _v2x2ssd_fwd_runtime_artifacts_prevalidated(
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
) -> ForwardRuntimeArtifacts:
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
    launch_tvm_ffi_on_current_stream(compiled, *runtime_artifacts.runtime_args)
    return runtime_artifacts


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
    "compile_v2x2ssd_fwd_cute",
    "tune_v2x2ssd_fwd_cute",
    "v2x2ssd_fwd_cute",
]
