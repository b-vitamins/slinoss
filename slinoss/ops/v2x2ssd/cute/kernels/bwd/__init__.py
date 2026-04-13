"""CuTe backward host stack for the ``v2x2ssd`` staged pipeline."""

from dataclasses import dataclass

import torch
import cutlass.cute as cute

from slinoss._cute_runtime import TensorSpec, make_runtime_tensor_spec_view
from slinoss.perf import note_cache_event

from ..fwd.common import (
    _assumed_align,
    _ensure_min_alignment,
    _make_fake_tensor_arg,
    _make_fake_tensor_spec_arg,
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
    _torch_to_cutlass_dtype,
)
from .chunk_increment.boundary import ChunkIncrementBwdBoundaryAmpere
from .chunk_increment.db import ChunkIncrementBwdDBAmpere
from .chunk_increment.du import ChunkIncrementBwdDUAmpere
from .chunk_increment.param_scan import ChunkIncrementBwdParamScanAmpere
from .chunk_scan import (
    _fold_chunk_boundary_carries,
    _materialize_public_output,
    _public_from_chunked,
    _public_from_param_scan,
    _resolve_dz0_cta_tiler,
)
from .chunk_scan.db import ChunkScanBwdDBAmpere
from .chunk_scan.dcdr import ChunkScanBwdDCDRAmpere
from .chunk_scan.dlp import ChunkScanBwdDLPAmpere
from .chunk_scan.du import ChunkScanBwdDUAmpere
from .chunk_scan.dz0 import ChunkScanBwdDZ0Ampere
from .chunk_scan.param_scan import ChunkScanBwdParamScanAmpere
from .state_passing.common import _TileConfig
from .state_passing import StatePassingBwdAmpere


_BWD_HOST_CACHE: dict[tuple, object] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_FINAL_GRAD_CACHE: dict[tuple, torch.Tensor] = {}
_BWD_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}
_ZERO_PREV_CACHE_LIMIT = 8
_ZERO_FINAL_GRAD_CACHE_LIMIT = 8
_BWD_WORKSPACE_CACHE_LIMIT = 4


BackwardProblemShape = tuple[int, int, int, int, int, int, int, int, int]
BackwardLaunchConfig = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    tuple[int, int, int],
]


@dataclass(frozen=True)
class BackwardOutputs:
    d_initial_state: torch.Tensor
    d_chunk_multiplier: torch.Tensor
    dU_scan: torch.Tensor
    dU_prev_scan: torch.Tensor
    dB_scan: torch.Tensor
    dB_prev_scan: torch.Tensor
    dC_scan: torch.Tensor
    dM_scan: torch.Tensor
    dK_scan: torch.Tensor
    dU_increment: torch.Tensor
    dU_prev_increment: torch.Tensor
    dB_increment: torch.Tensor
    dB_prev_increment: torch.Tensor
    dM_increment: torch.Tensor
    dK_increment: torch.Tensor


@dataclass(frozen=True)
class BackwardRuntimeArtifacts:
    runtime_args: tuple[torch.Tensor, ...]
    alignments: tuple[int, ...]
    problem_shape: BackwardProblemShape
    launch_cfg: BackwardLaunchConfig
    device_index: int
    tc_dtype: torch.dtype
    outputs: BackwardOutputs


@dataclass(frozen=True)
class BackwardCompileArtifacts:
    problem_shape: BackwardProblemShape
    launch_cfg: BackwardLaunchConfig
    compile_args: tuple[object, ...]
    alignments: tuple[int, ...]
    device_index: int
    tc_dtype: torch.dtype
    cache_key: tuple


@dataclass(frozen=True)
class BackwardInputInfo:
    batch_size: int
    heads: int
    bc_groups: int
    time_steps: int
    P: int
    D: int
    chunk_size: int
    n_chunks: int
    padded_time: int
    heads_per_bc_group: int
    tc_dtype: torch.dtype
    device_index: int
    n_d_tiles: int
    dz0_cta_tiler: tuple[int, int, int]


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
        note_cache_event("cute.v2x2ssd.bwd.zero_prev", hit=False)
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, bc_groups, D), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
    else:
        note_cache_event("cute.v2x2ssd.bwd.zero_prev", hit=True)
    return cached


def _get_zero_final_grad(
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
    cached = _ZERO_FINAL_GRAD_CACHE.get(key)
    if cached is None:
        note_cache_event("cute.v2x2ssd.bwd.zero_final_grad", hit=False)
        cached = torch.zeros(
            (batch_size, heads, P, D), device=device, dtype=torch.float32
        )
        _cache_set(
            _ZERO_FINAL_GRAD_CACHE,
            key,
            cached,
            limit=_ZERO_FINAL_GRAD_CACHE_LIMIT,
        )
    else:
        note_cache_event("cute.v2x2ssd.bwd.zero_final_grad", hit=True)
    return cached


def _get_bwd_workspace(
    *,
    device: torch.device,
    tc_dtype: torch.dtype,
    batch_size: int,
    heads: int,
    bc_groups: int,
    n_chunks: int,
    chunk_size: int,
    P: int,
    D: int,
    n_d_tiles: int,
) -> tuple[torch.Tensor, ...]:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        tc_dtype,
        int(batch_size),
        int(heads),
        int(bc_groups),
        int(n_chunks),
        int(chunk_size),
        int(P),
        int(D),
        int(n_d_tiles),
    )
    cached = _BWD_WORKSPACE_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.v2x2ssd.bwd.workspace", hit=True)
        return cached

    note_cache_event("cute.v2x2ssd.bwd.workspace", hit=False)
    batch_head_count = int(batch_size) * int(heads)
    batch_head_chunk_count = batch_head_count * int(n_chunks)
    batch_group_count = int(batch_size) * int(bc_groups)
    resolved_chunk_size = int(chunk_size)

    cached = (
        torch.empty((batch_head_count, n_chunks, P), device=device, dtype=tc_dtype),
        torch.empty((batch_group_count, n_chunks, D), device=device, dtype=tc_dtype),
        torch.empty((batch_head_chunk_count, P, D), device=device, dtype=torch.float32),
        torch.empty((batch_size, heads, n_chunks, P, D), device=device, dtype=tc_dtype),
        torch.empty((batch_size, heads, P, D), device=device, dtype=torch.float32),
        torch.empty(
            (batch_size, heads, n_chunks, 2), device=device, dtype=torch.float32
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 1, P),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, P), device=device, dtype=tc_dtype),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 1, D),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, D), device=device, dtype=tc_dtype),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 1, D),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 1, P),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, P), device=device, dtype=tc_dtype),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 1, D),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, D), device=device, dtype=tc_dtype),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 4),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 2),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, 2),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (batch_head_chunk_count, 1, resolved_chunk_size, 2),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (2, batch_head_chunk_count, 1, resolved_chunk_size, 2),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, D),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, D), device=device, dtype=tc_dtype),
        torch.empty(
            (batch_head_chunk_count, resolved_chunk_size, P),
            device=device,
            dtype=tc_dtype,
        ),
        torch.empty((batch_head_chunk_count, P), device=device, dtype=tc_dtype),
        torch.empty(
            (2, resolved_chunk_size, n_d_tiles, batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty((2, batch_head_chunk_count), device=device, dtype=torch.float32),
        torch.empty(
            (2, resolved_chunk_size, batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
        torch.empty(
            (2, 2, resolved_chunk_size, batch_head_chunk_count),
            device=device,
            dtype=torch.float32,
        ),
    )
    _cache_set(_BWD_WORKSPACE_CACHE, key, cached, limit=_BWD_WORKSPACE_CACHE_LIMIT)
    return cached


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
) -> TensorSpec:
    shape = tuple(int(dim) for dim in shape)
    if stride is None:
        stride = _make_row_major_stride(shape)
    else:
        stride = tuple(int(step) for step in stride)
    return shape, stride


def _chunk_scan_bwd_tensor_specs(
    problem_shape: BackwardProblemShape,
) -> tuple[TensorSpec, ...]:
    (
        batch_size,
        heads,
        bc_groups,
        padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        _n_d_tiles,
    ) = problem_shape
    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks
    batch_group_count = batch_size * bc_groups
    batch_group_chunk_count = batch_group_count * n_chunks
    return (
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 1, P),
            stride=(chunk_size * P, P, P, 1),
        ),
        _make_tensor_spec(
            (batch_group_chunk_count, chunk_size, 1, D),
            stride=(chunk_size * D, D, D, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 2),
            stride=(chunk_size * 2, 2, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 2, 2),
            stride=(chunk_size * 4, 4, 2, 1),
        ),
        _make_tensor_spec((batch_head_chunk_count, P, D), stride=(P * D, D, 1)),
        _make_tensor_spec((batch_head_count, P), stride=(P, 1)),
        _make_tensor_spec((batch_group_count, D), stride=(D, 1)),
        _make_tensor_spec((batch_head_chunk_count, chunk_size), stride=(chunk_size, 1)),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 2),
            stride=(chunk_size * 2, 2, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 4),
            stride=(chunk_size * 4, 4, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, 1, chunk_size, 2),
            stride=(chunk_size * 2, chunk_size * 2, 2, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, 1, chunk_size), stride=(chunk_size, chunk_size, 1)
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, 1, chunk_size, 4),
            stride=(chunk_size * 4, chunk_size * 4, 4, 1),
        ),
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
        _make_tensor_spec((P, D, batch_head_chunk_count), stride=(D, 1, P * D)),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 1, P),
            stride=(chunk_size * P, P, P, 1),
        ),
        _make_tensor_spec(
            (batch_head_chunk_count, chunk_size, 1, D),
            stride=(chunk_size * D, D, D, 1),
        ),
        _make_tensor_spec((batch_head_chunk_count, P), stride=(P, 1)),
        _make_tensor_spec((batch_head_chunk_count, D), stride=(D, 1)),
    )


def _state_passing_bwd_tensor_specs(
    problem_shape: BackwardProblemShape,
) -> tuple[TensorSpec, TensorSpec, TensorSpec]:
    (
        batch_size,
        heads,
        _bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        _chunk_size,
        _n_d_tiles,
    ) = problem_shape
    return (
        _make_tensor_spec((batch_size, heads, n_chunks, P, D)),
        _make_tensor_spec((batch_size, heads, n_chunks, 2)),
        _make_tensor_spec((batch_size, heads, P, D)),
    )


def _chunk_increment_bwd_tensor_specs(
    problem_shape: BackwardProblemShape,
) -> tuple[TensorSpec, ...]:
    (
        batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        n_d_tiles,
    ) = problem_shape
    batch_head_count = batch_size * heads
    batch_head_chunk_count = batch_head_count * n_chunks
    batch_group_count = batch_size * bc_groups
    batch_group_chunk_count = batch_group_count * n_chunks
    return (
        _make_tensor_spec(
            (chunk_size, P, batch_head_chunk_count), stride=(P, 1, chunk_size * P)
        ),
        _make_tensor_spec(
            (P, chunk_size, batch_head_chunk_count), stride=(1, P, chunk_size * P)
        ),
        _make_tensor_spec(
            (chunk_size, D, batch_group_chunk_count), stride=(D, 1, chunk_size * D)
        ),
        _make_tensor_spec(
            (2, chunk_size, batch_head_chunk_count), stride=(1, 2, chunk_size * 2)
        ),
        _make_tensor_spec(
            (2, chunk_size, batch_head_chunk_count), stride=(1, 4, chunk_size * 4)
        ),
        _make_tensor_spec((P, D, batch_head_chunk_count), stride=(D, 1, P * D)),
        _make_tensor_spec((D, P, batch_head_chunk_count), stride=(1, D, P * D)),
        _make_tensor_spec((batch_head_chunk_count, P, D), stride=(P * D, D, 1)),
        _make_tensor_spec((P, batch_head_chunk_count), stride=(1, P)),
        _make_tensor_spec((D, batch_group_count), stride=(1, D)),
        _make_tensor_spec((2, chunk_size, n_d_tiles, batch_head_chunk_count)),
        _make_tensor_spec((2, batch_head_chunk_count)),
        _make_tensor_spec((2, batch_head_chunk_count), stride=(1, 2)),
        _make_tensor_spec((2, chunk_size, batch_head_chunk_count)),
        _make_tensor_spec(
            (chunk_size, D, batch_head_chunk_count), stride=(D, 1, chunk_size * D)
        ),
        _make_tensor_spec((D, batch_head_chunk_count), stride=(1, D)),
    )


def _public_from_packed_dk(
    x: torch.Tensor,
    *,
    time_steps: int,
) -> torch.Tensor:
    B, H, C, L, _, F = map(int, x.shape)
    return x.reshape(B, H, C * L, 2, F)[:, :, :time_steps, :, :].contiguous()


def _reduce_heads_to_bc_groups(
    x: torch.Tensor,
    *,
    bc_groups: int,
) -> torch.Tensor:
    if int(x.shape[1]) == int(bc_groups):
        return x
    batch_size, heads = map(int, x.shape[:2])
    if heads % int(bc_groups) != 0:
        raise ValueError(
            "Grouped BC reduction requires bc_groups to divide heads. "
            f"Got heads={heads}, bc_groups={bc_groups}."
        )
    heads_per_bc_group = heads // int(bc_groups)
    grouped_shape = (batch_size, int(bc_groups), heads_per_bc_group, *x.shape[2:])
    return x.reshape(grouped_shape).sum(dim=2, dtype=torch.float32)


def _make_static_tensor_spec_view(
    ptr: cute.Tensor,
    tensor_spec: TensorSpec,
) -> cute.Tensor:
    shape, stride = tensor_spec
    return cute.make_tensor(ptr.iterator, cute.make_layout(shape, stride=stride))


def _make_runtime_tensor_views_from_specs(
    *tensor_specs: tuple[torch.Tensor, TensorSpec],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        make_runtime_tensor_spec_view(tensor, spec) for tensor, spec in tensor_specs
    )


def _make_tvm_ffi_runtime_and_compile_args_from_specs(
    *tensor_specs: tuple[torch.Tensor, torch.dtype, TensorSpec],
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
    )
    return resolved_alignments, compile_args


def _make_tvm_ffi_compile_args_from_specs(
    *tensor_specs: tuple[torch.dtype, TensorSpec, int],
) -> tuple[tuple[int, ...], tuple[object, ...]]:
    alignments = tuple(int(align) for _dtype, _spec, align in tensor_specs)
    compile_args = tuple(
        _make_fake_tensor_spec_arg(
            dtype=dtype,
            shape=spec[0],
            stride=spec[1],
            align=align,
        )
        for dtype, spec, align in tensor_specs
    )
    return alignments, compile_args


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
        prepared = tensor
    else:
        prepared = _pad_zero_time(tensor, T_pad=padded_time, dtype=dtype)
    if prepared.dtype in (torch.float16, torch.bfloat16):
        prepared = _ensure_min_alignment(prepared, min_align=16)
    return prepared


def _prepare_m_operand(M: torch.Tensor, *, padded_time: int) -> torch.Tensor:
    if (
        int(M.shape[2]) == padded_time
        and M.dtype == torch.float32
        and M.is_contiguous()
    ):
        return M
    return _pad_m_identity(M, T_pad=padded_time)


def _bwd_host_cache_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    problem_shape: BackwardProblemShape,
    launch_cfg: BackwardLaunchConfig,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "v2x2ssd_bwd_host",
        int(device_index),
        tc_dtype,
        tuple(int(dim) for dim in problem_shape),
        tuple(
            tuple(int(value) for value in item)
            if isinstance(item, tuple)
            else int(item)
            for item in launch_cfg
        ),
        tuple(int(align) for align in alignments),
    )


def _make_backward_input_info(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dcdr: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    validate_runtime_contract: bool = True,
) -> BackwardInputInfo:
    if validate_runtime_contract:
        if U.device.type != "cuda":
            raise ValueError("CUDA tensor required.")
        if U.dtype != B.dtype or U.dtype != C.dtype:
            raise ValueError("U/B/C must share dtype.")
        if (B_prev is None) ^ (U_prev is None):
            raise ValueError(
                "B_prev and U_prev must be passed together (or both omitted)."
            )
        if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("U/B/C must be float16/bfloat16/float32.")
        if M.dtype != torch.float32 or K.dtype != torch.float32:
            raise TypeError("M and K must be float32.")
        if d_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("d_out must be float16/bfloat16/float32.")
        if m_chunk.dtype != torch.float32 or chunk_starts.dtype != torch.float32:
            raise TypeError("m_chunk and chunk_starts must be float32.")

    batch_size, heads, time_steps, P = map(int, U.shape)
    bc_groups = int(B.shape[1])
    D = int(B.shape[-1])
    if heads % bc_groups != 0:
        raise ValueError(
            "B/C grouped BC contract requires bc_groups to divide heads. "
            f"Got heads={heads}, bc_groups={bc_groups}."
        )
    heads_per_bc_group = heads // bc_groups
    if validate_runtime_contract:
        if B.shape != (batch_size, bc_groups, time_steps, D) or C.shape != (
            batch_size,
            bc_groups,
            time_steps,
            D,
        ):
            raise ValueError("B/C must be (B,G,T,D) matching U and grouped BC.")
        if M.shape != (batch_size, heads, time_steps, 2):
            raise ValueError(
                f"M must be (B,H,T,2)={(batch_size, heads, time_steps, 2)}."
            )
        if K.shape != (batch_size, heads, time_steps, 2, 2):
            raise ValueError(
                f"K must be (B,H,T,2,2)={(batch_size, heads, time_steps, 2, 2)}."
            )
        if d_out.shape != (batch_size, heads, time_steps, P):
            raise ValueError("d_out must be (B,H,T,P) matching U.")
        if B_prev is not None:
            if tuple(B_prev.shape) != (batch_size, bc_groups, D) or tuple(
                U_prev.shape
            ) != (
                batch_size,
                heads,
                P,
            ):
                raise ValueError("B_prev/U_prev must be (B,G,D)/(B,H,P).")
            if B_prev.device != U.device or U_prev.device != U.device:
                raise ValueError("B_prev/U_prev must be on the same device as U.")
        if d_final_state is not None:
            if tuple(d_final_state.shape) != (batch_size, heads, P, D):
                raise ValueError("d_final_state must be (B,H,P,D).")
            if d_final_state.device != U.device:
                raise ValueError("d_final_state must be on the same device as U.")
        if D % 2 != 0:
            raise ValueError("D must be divisible by 2.")

    resolved_chunk_size = int(chunk_size)
    if resolved_chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (time_steps + resolved_chunk_size - 1) // resolved_chunk_size
    padded_time = n_chunks * resolved_chunk_size
    if validate_runtime_contract:
        if tuple(m_chunk.shape) != (batch_size, heads, n_chunks, 2):
            raise ValueError(
                f"m_chunk must be (B,H,C,2)={(batch_size, heads, n_chunks, 2)}. "
                f"Got {tuple(m_chunk.shape)}."
            )
        if tuple(chunk_starts.shape) != (batch_size, heads, n_chunks, P, D):
            raise ValueError(
                "chunk_starts must be (B,H,C,P,D)="
                f"{(batch_size, heads, n_chunks, P, D)}. "
                f"Got {tuple(chunk_starts.shape)}."
            )
        for num_threads, label in (
            (scan_num_threads_du, "scan_num_threads_du"),
            (scan_num_threads_db, "scan_num_threads_db"),
            (scan_num_threads_dcdr, "scan_num_threads_dcdr"),
            (scan_num_threads_param, "scan_num_threads_param"),
        ):
            if int(num_threads) <= 0:
                raise ValueError(f"{label} must be positive.")

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    chunk_increment_db = ChunkIncrementBwdDBAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=resolved_chunk_size,
        D=D,
        P=P,
    )
    n_d_tiles = (D + chunk_increment_db.bN - 1) // chunk_increment_db.bN
    return BackwardInputInfo(
        batch_size=batch_size,
        heads=heads,
        bc_groups=bc_groups,
        time_steps=time_steps,
        P=P,
        D=D,
        chunk_size=resolved_chunk_size,
        n_chunks=n_chunks,
        padded_time=padded_time,
        heads_per_bc_group=heads_per_bc_group,
        tc_dtype=tc_dtype,
        device_index=(U.device.index if U.device.index is not None else -1),
        n_d_tiles=n_d_tiles,
        dz0_cta_tiler=_resolve_dz0_cta_tiler(D=D),
    )


def _normalize_backward_state_config(
    *,
    num_threads: int,
    pairs_per_thread: int,
) -> _TileConfig:
    config = _TileConfig(
        num_threads=int(num_threads),
        pairs_per_thread=int(pairs_per_thread),
    )
    if config.num_threads <= 0:
        raise ValueError("state_num_threads must be positive.")
    if config.num_threads % 32 != 0:
        raise ValueError("state_num_threads must be a multiple of 32.")
    if config.pairs_per_thread <= 0:
        raise ValueError("state_pairs_per_thread must be positive.")
    return config


def _make_backward_problem_shape(
    input_info: BackwardInputInfo,
) -> BackwardProblemShape:
    return (
        input_info.batch_size,
        input_info.heads,
        input_info.bc_groups,
        input_info.padded_time,
        input_info.P,
        input_info.D,
        input_info.n_chunks,
        input_info.chunk_size,
        input_info.n_d_tiles,
    )


def _make_backward_launch_cfg(
    *,
    input_info: BackwardInputInfo,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dcdr: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
) -> BackwardLaunchConfig:
    state_config = _normalize_backward_state_config(
        num_threads=state_num_threads,
        pairs_per_thread=state_pairs_per_thread,
    )
    state_assumed_align = _compile_min_align_for_dtype(torch.float32)
    state_tile_stride_elems = int(input_info.P) * int(input_info.D)
    return (
        int(scan_num_threads_du),
        int(scan_num_threads_db),
        int(scan_num_threads_dcdr),
        int(scan_num_threads_param),
        int(state_config.num_threads),
        int(state_config.pairs_per_thread),
        _choose_copy_bits_for_linear_tiles_from_properties(
            dtype=torch.float32,
            assumed_align=state_assumed_align,
            tile_stride_elems=state_tile_stride_elems,
            elems_per_thread=state_config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles_from_properties(
            dtype=torch.float32,
            assumed_align=state_assumed_align,
            tile_stride_elems=state_tile_stride_elems,
            elems_per_thread=state_config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles_from_properties(
            dtype=torch.float32,
            assumed_align=state_assumed_align,
            tile_stride_elems=state_tile_stride_elems,
            elems_per_thread=state_config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles_from_properties(
            dtype=torch.float32,
            assumed_align=state_assumed_align,
            tile_stride_elems=state_tile_stride_elems,
            elems_per_thread=state_config.elems_per_thread,
        ),
        _choose_copy_bits_for_linear_tiles_from_properties(
            dtype=torch.float32,
            assumed_align=state_assumed_align,
            tile_stride_elems=state_tile_stride_elems,
            elems_per_thread=state_config.elems_per_thread,
        ),
        input_info.dz0_cta_tiler,
    )


def _make_backward_runtime_arg_descriptors(
    problem_shape: BackwardProblemShape,
    *,
    tc_dtype: torch.dtype,
) -> tuple[tuple[torch.dtype, TensorSpec, int], ...]:
    batch_size, heads, bc_groups, padded_time, P, D, n_chunks, chunk_size, n_d_tiles = (
        problem_shape
    )
    batch_head_count = batch_size * heads
    batch_group_count = batch_size * bc_groups
    tc_align = _compile_min_align_for_dtype(tc_dtype)
    fp32_align = _compile_min_align_for_dtype(torch.float32)
    k_pair_stride = (heads * padded_time * 4, padded_time * 4, 4, 1)
    (
        U_scan_spec,
        BC_scan_spec,
        _M_scan_spec,
        _K_scan_spec,
        _chunk_starts_scan_spec,
        _U_prev_scan_spec,
        _B_prev_scan_spec,
        dlogp_scan_spec,
        dM_scan_scratch_spec,
        dR_scan_spec,
        d_param_scan_spec,
        _dlogp_param_spec,
        _dR_param_spec,
        _d_out_dz0_spec,
        _C_dz0_spec,
        _M_dz0_spec,
        d_chunk_starts_spec,
        dU_db_dummy_spec,
        dBC_scan_spec,
        dU_prev_dummy_spec,
        dB_prev_dummy_spec,
    ) = _chunk_scan_bwd_tensor_specs(problem_shape)
    chunk_starts_state_spec, chunk_multiplier_state_spec, final_state_spec = (
        _state_passing_bwd_tensor_specs(problem_shape)
    )
    (
        _U_increment_spec,
        dU_increment_spec,
        _B_increment_input_spec,
        _M_increment_spec,
        _K_increment_spec,
        _d_increment_spec,
        _d_increment_dp_spec,
        _d_increment_boundary_spec,
        U_prev_chunks_spec,
        _B_prev_chunks_input_spec,
        dM_sum_part_spec,
        dMp0_spec,
        _d_chunk_multiplier_increment_spec,
        d_param_increment_spec,
        dB_increment_spec,
        dB_prev_increment_spec,
    ) = _chunk_increment_bwd_tensor_specs(problem_shape)
    return (
        (tc_dtype, _make_tensor_spec((batch_size, heads, padded_time, P)), tc_align),
        (
            tc_dtype,
            _make_tensor_spec((batch_size, bc_groups, padded_time, D)),
            tc_align,
        ),
        (
            tc_dtype,
            _make_tensor_spec((batch_size, bc_groups, padded_time, D)),
            tc_align,
        ),
        (
            torch.float32,
            _make_tensor_spec((batch_size, heads, padded_time, 2)),
            fp32_align,
        ),
        (
            torch.float32,
            _make_tensor_spec((batch_size, heads, padded_time, 2, 2)),
            fp32_align,
        ),
        (
            torch.float32,
            _make_tensor_spec(
                (batch_size, heads, padded_time, 2),
                stride=k_pair_stride,
            ),
            fp32_align,
        ),
        (
            torch.float32,
            _make_tensor_spec(
                (batch_size, heads, padded_time, 2),
                stride=k_pair_stride,
            ),
            8,
        ),
        (tc_dtype, _make_tensor_spec((batch_size, heads, padded_time, P)), tc_align),
        (
            torch.float32,
            chunk_multiplier_state_spec,
            fp32_align,
        ),
        (
            torch.float32,
            chunk_starts_state_spec,
            fp32_align,
        ),
        (tc_dtype, _make_tensor_spec((batch_size, heads, P)), tc_align),
        (tc_dtype, _make_tensor_spec((batch_size, bc_groups, D)), tc_align),
        (torch.float32, final_state_spec, fp32_align),
        (
            tc_dtype,
            _make_tensor_spec((batch_head_count, n_chunks, P)),
            tc_align,
        ),
        (
            tc_dtype,
            _make_tensor_spec((batch_group_count, n_chunks, D)),
            tc_align,
        ),
        (
            torch.float32,
            d_chunk_starts_spec,
            fp32_align,
        ),
        (
            tc_dtype,
            chunk_starts_state_spec,
            tc_align,
        ),
        (torch.float32, final_state_spec, fp32_align),
        (
            torch.float32,
            chunk_multiplier_state_spec,
            fp32_align,
        ),
        (tc_dtype, U_scan_spec, tc_align),
        (tc_dtype, dU_prev_dummy_spec, tc_align),
        (tc_dtype, dBC_scan_spec, tc_align),
        (tc_dtype, dB_prev_dummy_spec, tc_align),
        (tc_dtype, dBC_scan_spec, tc_align),
        (tc_dtype, dU_db_dummy_spec, tc_align),
        (tc_dtype, dU_prev_dummy_spec, tc_align),
        (tc_dtype, dBC_scan_spec, tc_align),
        (tc_dtype, dB_prev_dummy_spec, tc_align),
        (torch.float32, dlogp_scan_spec, fp32_align),
        (torch.float32, dR_scan_spec, fp32_align),
        (torch.float32, dM_scan_scratch_spec, fp32_align),
        (torch.float32, dM_scan_scratch_spec, fp32_align),
        (torch.float32, d_param_scan_spec, fp32_align),
        (torch.float32, d_param_scan_spec, fp32_align),
        (torch.float32, d_param_scan_spec, fp32_align),
        (
            tc_dtype,
            dB_increment_spec,
            tc_align,
        ),
        (
            tc_dtype,
            dB_prev_increment_spec,
            tc_align,
        ),
        (
            tc_dtype,
            dU_increment_spec,
            tc_align,
        ),
        (
            tc_dtype,
            U_prev_chunks_spec,
            tc_align,
        ),
        (
            torch.float32,
            dM_sum_part_spec,
            fp32_align,
        ),
        (
            torch.float32,
            dMp0_spec,
            fp32_align,
        ),
        (
            torch.float32,
            d_param_increment_spec,
            fp32_align,
        ),
        (
            torch.float32,
            d_param_increment_spec,
            fp32_align,
        ),
        (
            torch.float32,
            d_param_increment_spec,
            fp32_align,
        ),
    )


def _make_backward_compile_artifacts(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dcdr: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
) -> BackwardCompileArtifacts:
    input_info = _make_backward_input_info(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        validate_runtime_contract=True,
    )
    problem_shape = _make_backward_problem_shape(input_info)
    launch_cfg = _make_backward_launch_cfg(
        input_info=input_info,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
    )
    arg_descriptors = _make_backward_runtime_arg_descriptors(
        problem_shape,
        tc_dtype=input_info.tc_dtype,
    )
    alignments, compile_args = _make_tvm_ffi_compile_args_from_specs(
        *((dtype, tensor_spec, align) for dtype, tensor_spec, align in arg_descriptors)
    )
    return BackwardCompileArtifacts(
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        compile_args=compile_args,
        alignments=alignments,
        device_index=input_info.device_index,
        tc_dtype=input_info.tc_dtype,
        cache_key=_bwd_host_cache_key(
            device_index=input_info.device_index,
            tc_dtype=input_info.tc_dtype,
            problem_shape=problem_shape,
            launch_cfg=launch_cfg,
            alignments=alignments,
        ),
    )


def _make_backward_compile_artifacts_from_runtime_artifacts(
    runtime_artifacts: BackwardRuntimeArtifacts,
) -> BackwardCompileArtifacts:
    _, compile_args = _make_tvm_ffi_compile_args(
        runtime_artifacts.runtime_args,
        alignments=runtime_artifacts.alignments,
    )
    return BackwardCompileArtifacts(
        problem_shape=runtime_artifacts.problem_shape,
        launch_cfg=runtime_artifacts.launch_cfg,
        compile_args=compile_args,
        alignments=runtime_artifacts.alignments,
        device_index=runtime_artifacts.device_index,
        tc_dtype=runtime_artifacts.tc_dtype,
        cache_key=_bwd_host_cache_key(
            device_index=runtime_artifacts.device_index,
            tc_dtype=runtime_artifacts.tc_dtype,
            problem_shape=runtime_artifacts.problem_shape,
            launch_cfg=runtime_artifacts.launch_cfg,
            alignments=runtime_artifacts.alignments,
        ),
    )


def _make_backward_runtime_artifacts(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dcdr: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    validate_runtime_contract: bool = True,
) -> BackwardRuntimeArtifacts:
    input_info = _make_backward_input_info(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        validate_runtime_contract=validate_runtime_contract,
    )
    problem_shape = _make_backward_problem_shape(input_info)
    launch_cfg = _make_backward_launch_cfg(
        input_info=input_info,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
    )
    batch_size = input_info.batch_size
    heads = input_info.heads
    bc_groups = input_info.bc_groups
    P = input_info.P
    D = input_info.D
    n_chunks = input_info.n_chunks
    padded_time = input_info.padded_time
    tc_dtype = input_info.tc_dtype
    chunk_size = input_info.chunk_size
    batch_head_count = batch_size * heads
    batch_group_count = batch_size * bc_groups

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

    d_out_tc = _prepare_time_operand(d_out, padded_time=padded_time, dtype=tc_dtype)
    chunk_multiplier = (
        m_chunk
        if m_chunk.dtype == torch.float32 and m_chunk.is_contiguous()
        else m_chunk.to(dtype=torch.float32).contiguous()
    )
    chunk_starts_f = (
        chunk_starts
        if chunk_starts.dtype == torch.float32 and chunk_starts.is_contiguous()
        else chunk_starts.to(dtype=torch.float32).contiguous()
    )
    chunk_multiplier = _ensure_min_alignment(chunk_multiplier, min_align=16)
    chunk_starts_f = _ensure_min_alignment(chunk_starts_f, min_align=16)

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
        U_prev_state = _ensure_min_alignment(
            U_prev.to(dtype=tc_dtype).contiguous(),
            min_align=16,
        )
        B_prev_state = _ensure_min_alignment(
            B_prev.to(dtype=tc_dtype).contiguous(),
            min_align=16,
        )

    if d_final_state is None:
        d_final_buffer = _get_zero_final_grad(
            device=U.device,
            batch_size=batch_size,
            heads=heads,
            P=P,
            D=D,
        )
    else:
        d_final_buffer = d_final_state.to(dtype=torch.float32).contiguous()
    d_final_buffer = _ensure_min_alignment(d_final_buffer, min_align=16)

    U_chunked = U_tc.reshape(batch_head_count, n_chunks, chunk_size, P)
    B_chunked = B_tc.reshape(batch_group_count, n_chunks, chunk_size, D)

    (
        U_prev_chunks,
        B_prev_chunks,
        d_chunk_starts,
        d_increment_state,
        d_initial_state,
        d_chunk_multiplier,
        dU_scan_storage,
        dU_prev_scan_storage,
        dB_scan_storage,
        dB_prev_scan_storage,
        dC_scan_storage,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dM_previous_scratch,
        dM_current_scratch,
        dM_scan_storage,
        dK_scan_storage,
        dB_increment_storage,
        dB_prev_increment_storage,
        dU_increment_storage,
        dU_prev_increment_storage,
        dM_sum_part,
        dMp0,
        dM_increment_storage,
        dK_increment_storage,
    ) = _get_bwd_workspace(
        device=U.device,
        tc_dtype=tc_dtype,
        batch_size=batch_size,
        heads=heads,
        bc_groups=bc_groups,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
        P=P,
        D=D,
        n_d_tiles=input_info.n_d_tiles,
    )

    dK_previous_scan = dK_scan_storage[0]
    dK_current_scan = dK_scan_storage[1]
    dK_previous_increment = dK_increment_storage[0]
    dK_current_increment = dK_increment_storage[1]

    # These workspaces are cached across calls. The fused backward path only
    # defines the valid scan domain, so cached accumulators must start from zero
    # rather than stale previous contents.
    d_chunk_multiplier.zero_()
    dM_scan_storage.zero_()
    dM_increment_storage.zero_()

    U_prev_chunks[:, 0, :] = U_prev_state.reshape(batch_head_count, P)
    B_prev_chunks[:, 0, :] = B_prev_state.reshape(batch_group_count, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunked[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunked[:, :-1, -1, :]

    K_previous_view = K_f[:, :, :, 0, :]
    K_current_view = K_f[:, :, :, 1, :]
    runtime_tensors = (
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        K_previous_view,
        K_current_view,
        d_out_tc,
        chunk_multiplier,
        chunk_starts_f,
        U_prev_state,
        B_prev_state,
        d_final_buffer,
        U_prev_chunks,
        B_prev_chunks,
        d_chunk_starts,
        d_increment_state,
        d_initial_state,
        d_chunk_multiplier,
        dU_scan_storage,
        dU_prev_scan_storage,
        dB_scan_storage,
        dB_prev_scan_storage,
        dC_scan_storage,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dM_previous_scratch,
        dM_current_scratch,
        dM_scan_storage,
        dK_previous_scan,
        dK_current_scan,
        dB_increment_storage,
        dB_prev_increment_storage,
        dU_increment_storage,
        dU_prev_increment_storage,
        dM_sum_part,
        dMp0,
        dM_increment_storage,
        dK_previous_increment,
        dK_current_increment,
    )
    arg_descriptors = _make_backward_runtime_arg_descriptors(
        problem_shape,
        tc_dtype=tc_dtype,
    )
    runtime_args, alignments, _compile_args = (
        _make_tvm_ffi_runtime_and_compile_args_from_specs(
            *(
                (tensor, dtype, tensor_spec)
                for tensor, (dtype, tensor_spec, _align) in zip(
                    runtime_tensors, arg_descriptors, strict=True
                )
            )
        )
    )

    dU_scan = dU_scan_storage.reshape(batch_size, heads, n_chunks, chunk_size, P)
    dU_prev_scan = dU_prev_scan_storage.reshape(batch_size, heads, n_chunks, P)
    dB_scan = dB_scan_storage.reshape(batch_size, heads, n_chunks, chunk_size, D)
    dB_prev_scan = dB_prev_scan_storage.reshape(batch_size, heads, n_chunks, D)
    dC_scan = dC_scan_storage.reshape(batch_size, heads, n_chunks, chunk_size, D)
    dM_scan = dM_scan_storage.reshape(batch_size, heads, n_chunks, 1, chunk_size, 2)
    dK_scan = (
        dK_scan_storage.permute(1, 2, 3, 0, 4)
        .reshape(batch_size, heads, n_chunks, 1, chunk_size, 2, 2)
        .reshape(batch_size, heads, n_chunks, chunk_size, 2, 2)
    )
    dU_increment = dU_increment_storage.reshape(
        batch_size, heads, n_chunks, chunk_size, P
    )
    dU_prev_increment = dU_prev_increment_storage.reshape(
        batch_size, heads, n_chunks, P
    )
    dB_increment = dB_increment_storage.reshape(
        batch_size, heads, n_chunks, chunk_size, D
    )
    dB_prev_increment = dB_prev_increment_storage.reshape(
        batch_size, heads, n_chunks, D
    )
    dM_increment = dM_increment_storage.permute(2, 1, 0).reshape(
        batch_size, heads, n_chunks, chunk_size, 2
    )
    dK_increment = dK_increment_storage.permute(3, 2, 0, 1).reshape(
        batch_size, heads, n_chunks, chunk_size, 2, 2
    )

    return BackwardRuntimeArtifacts(
        runtime_args=runtime_args,
        alignments=alignments,
        problem_shape=problem_shape,
        launch_cfg=launch_cfg,
        device_index=input_info.device_index,
        tc_dtype=tc_dtype,
        outputs=BackwardOutputs(
            d_initial_state=d_initial_state,
            d_chunk_multiplier=d_chunk_multiplier,
            dU_scan=dU_scan,
            dU_prev_scan=dU_prev_scan,
            dB_scan=dB_scan,
            dB_prev_scan=dB_prev_scan,
            dC_scan=dC_scan,
            dM_scan=dM_scan,
            dK_scan=dK_scan,
            dU_increment=dU_increment,
            dU_prev_increment=dU_prev_increment,
            dB_increment=dB_increment,
            dB_prev_increment=dB_prev_increment,
            dM_increment=dM_increment,
            dK_increment=dK_increment,
        ),
    )


def _make_v2x2ssd_bwd_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg: BackwardLaunchConfig,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        n_d_tiles,
    ) = problem_shape
    (
        scan_num_threads_du,
        scan_num_threads_db,
        scan_num_threads_dcdr,
        scan_num_threads_param,
        state_num_threads,
        state_pairs_per_thread,
        state_copy_bits_starts,
        state_copy_bits_dstarts,
        state_copy_bits_dinc,
        state_copy_bits_initial,
        state_copy_bits_final,
        dz0_cta_tiler,
    ) = launch_cfg
    chunk_scan_specs = _chunk_scan_bwd_tensor_specs(problem_shape)
    state_passing_specs = _state_passing_bwd_tensor_specs(problem_shape)
    chunk_increment_specs = _chunk_increment_bwd_tensor_specs(problem_shape)

    @cute.jit
    def _v2x2ssd_bwd_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        K_previous_t: cute.Tensor,
        K_current_t: cute.Tensor,
        d_out_t: cute.Tensor,
        chunk_multiplier_t: cute.Tensor,
        chunk_starts_t: cute.Tensor,
        U_prev_state_t: cute.Tensor,
        B_prev_state_t: cute.Tensor,
        d_final_state_t: cute.Tensor,
        U_prev_chunks_t: cute.Tensor,
        B_prev_chunks_t: cute.Tensor,
        d_chunk_starts_t: cute.Tensor,
        d_increment_t: cute.Tensor,
        d_initial_state_t: cute.Tensor,
        d_chunk_multiplier_t: cute.Tensor,
        dU_scan_t: cute.Tensor,
        dU_prev_scan_t: cute.Tensor,
        dB_scan_t: cute.Tensor,
        dB_prev_scan_t: cute.Tensor,
        dC_scan_t: cute.Tensor,
        dU_db_dummy_t: cute.Tensor,
        dU_prev_db_dummy_t: cute.Tensor,
        dB_du_dummy_t: cute.Tensor,
        dB_prev_du_dummy_t: cute.Tensor,
        dlogp_t: cute.Tensor,
        dR_t: cute.Tensor,
        dM_previous_scratch_t: cute.Tensor,
        dM_current_scratch_t: cute.Tensor,
        dM_scan_t: cute.Tensor,
        dK_previous_scan_t: cute.Tensor,
        dK_current_scan_t: cute.Tensor,
        dB_increment_t: cute.Tensor,
        dB_prev_increment_t: cute.Tensor,
        dU_increment_t: cute.Tensor,
        dU_prev_increment_t: cute.Tensor,
        dM_sum_part_t: cute.Tensor,
        dMp0_t: cute.Tensor,
        dM_increment_t: cute.Tensor,
        dK_previous_increment_t: cute.Tensor,
        dK_current_increment_t: cute.Tensor,
    ):
        (
            U_scan_spec,
            BC_scan_spec,
            M_scan_spec,
            K_scan_spec,
            chunk_starts_scan_spec,
            U_prev_scan_spec,
            B_prev_scan_spec,
            dlogp_scan_spec,
            dM_scan_scratch_spec,
            dR_scan_spec,
            d_param_scan_spec,
            dlogp_param_spec,
            dR_param_spec,
            d_out_dz0_spec,
            C_dz0_spec,
            M_dz0_spec,
            d_chunk_starts_scan_spec,
            dU_db_dummy_spec,
            dBC_scan_spec,
            dU_prev_dummy_spec,
            dB_prev_dummy_spec,
        ) = chunk_scan_specs
        (
            chunk_starts_state_spec,
            chunk_multiplier_state_spec,
            final_state_spec,
        ) = state_passing_specs
        (
            U_increment_spec,
            dU_increment_spec,
            B_increment_input_spec,
            M_increment_spec,
            K_increment_spec,
            d_increment_spec,
            d_increment_dp_spec,
            d_increment_boundary_spec,
            U_prev_chunks_spec,
            B_prev_chunks_input_spec,
            dM_sum_part_spec,
            dMp0_spec,
            d_chunk_multiplier_increment_spec,
            d_param_increment_spec,
            dB_increment_spec,
            dB_prev_increment_spec,
        ) = chunk_increment_specs

        U_scan_view = _make_static_tensor_spec_view(U_t, U_scan_spec)
        B_scan_view = _make_static_tensor_spec_view(B_t, BC_scan_spec)
        C_scan_view = _make_static_tensor_spec_view(C_t, BC_scan_spec)
        M_scan_view = _make_static_tensor_spec_view(M_t, M_scan_spec)
        K_scan_view = _make_static_tensor_spec_view(K_t, K_scan_spec)
        d_out_scan_view = _make_static_tensor_spec_view(d_out_t, U_scan_spec)
        chunk_starts_scan_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_scan_spec
        )
        U_prev_scan_view = _make_static_tensor_spec_view(
            U_prev_state_t, U_prev_scan_spec
        )
        B_prev_scan_view = _make_static_tensor_spec_view(
            B_prev_state_t, B_prev_scan_spec
        )
        dU_scan_view = _make_static_tensor_spec_view(dU_scan_t, U_scan_spec)
        dB_scan_view = _make_static_tensor_spec_view(dB_scan_t, dBC_scan_spec)
        dU_prev_scan_view = _make_static_tensor_spec_view(
            dU_prev_scan_t, dU_prev_dummy_spec
        )
        dB_prev_scan_view = _make_static_tensor_spec_view(
            dB_prev_scan_t, dB_prev_dummy_spec
        )
        dlogp_scan_view = _make_static_tensor_spec_view(dlogp_t, dlogp_scan_spec)
        dlogp_param_view = _make_static_tensor_spec_view(dlogp_t, dlogp_param_spec)
        dM_previous_scan_view = _make_static_tensor_spec_view(
            dM_previous_scratch_t, dM_scan_scratch_spec
        )
        dM_current_scan_view = _make_static_tensor_spec_view(
            dM_current_scratch_t, dM_scan_scratch_spec
        )
        dM_previous_param_view = _make_static_tensor_spec_view(
            dM_previous_scratch_t, d_param_scan_spec
        )
        dM_current_param_view = _make_static_tensor_spec_view(
            dM_current_scratch_t, d_param_scan_spec
        )
        dC_scan_view = _make_static_tensor_spec_view(dC_scan_t, dBC_scan_spec)
        dR_scan_view = _make_static_tensor_spec_view(dR_t, dR_scan_spec)
        dR_param_view = _make_static_tensor_spec_view(dR_t, dR_param_spec)
        dM_scan_view = _make_static_tensor_spec_view(dM_scan_t, d_param_scan_spec)
        dK_previous_scan_view = _make_static_tensor_spec_view(
            dK_previous_scan_t, d_param_scan_spec
        )
        dK_current_scan_view = _make_static_tensor_spec_view(
            dK_current_scan_t, d_param_scan_spec
        )

        d_out_dz0_view = _make_static_tensor_spec_view(d_out_t, d_out_dz0_spec)
        C_dz0_view = _make_static_tensor_spec_view(C_t, C_dz0_spec)
        M_dz0_view = _make_static_tensor_spec_view(M_t, M_dz0_spec)
        d_chunk_starts_scan_view = _make_static_tensor_spec_view(
            d_chunk_starts_t, d_chunk_starts_scan_spec
        )

        chunk_starts_state_view = _make_static_tensor_spec_view(
            chunk_starts_t, chunk_starts_state_spec
        )
        chunk_multiplier_state_view = _make_static_tensor_spec_view(
            chunk_multiplier_t, chunk_multiplier_state_spec
        )
        d_chunk_starts_state_view = _make_static_tensor_spec_view(
            d_chunk_starts_t, chunk_starts_state_spec
        )
        d_final_state_view = _make_static_tensor_spec_view(
            d_final_state_t, final_state_spec
        )
        d_increment_state_view = _make_static_tensor_spec_view(
            d_increment_t, chunk_starts_state_spec
        )
        d_initial_state_view = _make_static_tensor_spec_view(
            d_initial_state_t, final_state_spec
        )
        d_chunk_multiplier_state_view = _make_static_tensor_spec_view(
            d_chunk_multiplier_t, chunk_multiplier_state_spec
        )

        U_increment_view = _make_static_tensor_spec_view(U_t, U_increment_spec)
        B_increment_view = _make_static_tensor_spec_view(B_t, B_increment_input_spec)
        M_increment_view = _make_static_tensor_spec_view(M_t, M_increment_spec)
        K_previous_increment_view = _make_static_tensor_spec_view(
            K_previous_t, K_increment_spec
        )
        K_current_increment_view = _make_static_tensor_spec_view(
            K_current_t, K_increment_spec
        )
        d_increment_dp_view = _make_static_tensor_spec_view(
            d_increment_t, d_increment_dp_spec
        )
        d_increment_view = _make_static_tensor_spec_view(
            d_increment_t, d_increment_spec
        )
        d_increment_boundary_view = _make_static_tensor_spec_view(
            d_increment_t, d_increment_boundary_spec
        )
        B_prev_chunks_view = _make_static_tensor_spec_view(
            B_prev_chunks_t, B_prev_chunks_input_spec
        )
        U_prev_chunks_view = _make_static_tensor_spec_view(
            U_prev_chunks_t, U_prev_chunks_spec
        )
        dB_increment_view = _make_static_tensor_spec_view(
            dB_increment_t, dB_increment_spec
        )
        dU_increment_view = _make_static_tensor_spec_view(
            dU_increment_t, dU_increment_spec
        )
        dB_prev_increment_view = _make_static_tensor_spec_view(
            dB_prev_increment_t, dB_prev_increment_spec
        )
        dU_prev_increment_view = _make_static_tensor_spec_view(
            dU_prev_increment_t, U_prev_chunks_spec
        )
        dM_sum_part_view = _make_static_tensor_spec_view(
            dM_sum_part_t, dM_sum_part_spec
        )
        dMp0_view = _make_static_tensor_spec_view(dMp0_t, dMp0_spec)
        d_chunk_multiplier_increment_view = _make_static_tensor_spec_view(
            d_chunk_multiplier_t, d_chunk_multiplier_increment_spec
        )
        dM_increment_view = _make_static_tensor_spec_view(
            dM_increment_t, d_param_increment_spec
        )
        dK_previous_increment_view = _make_static_tensor_spec_view(
            dK_previous_increment_t, d_param_increment_spec
        )
        dK_current_increment_view = _make_static_tensor_spec_view(
            dK_current_increment_t, d_param_increment_spec
        )

        dU_db_dummy_view = _make_static_tensor_spec_view(
            dU_db_dummy_t, dU_db_dummy_spec
        )
        dU_prev_db_dummy_view = _make_static_tensor_spec_view(
            dU_prev_db_dummy_t, dU_prev_dummy_spec
        )
        dB_du_dummy_view = _make_static_tensor_spec_view(dB_du_dummy_t, dBC_scan_spec)
        dB_prev_du_dummy_view = _make_static_tensor_spec_view(
            dB_prev_du_dummy_t, dB_prev_dummy_spec
        )

        tc_dtype = U_scan_view.element_type

        chunk_scan_db_kernel = ChunkScanBwdDBAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_db,
        )
        chunk_scan_dcdr_kernel = ChunkScanBwdDCDRAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_dcdr,
        )
        chunk_scan_dlp_kernel = ChunkScanBwdDLPAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_dcdr,
        )
        chunk_scan_param_kernel = ChunkScanBwdParamScanAmpere(
            chunk_size=chunk_size,
            num_threads=scan_num_threads_param,
        )
        chunk_scan_du_kernel = ChunkScanBwdDUAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_du,
        )
        chunk_scan_dz0_kernel = ChunkScanBwdDZ0Ampere(
            tc_dtype,
            chunk_size=chunk_size,
            heads=heads,
            bc_groups=bc_groups,
            cta_tiler=dz0_cta_tiler,
        )

        state_config = _TileConfig(
            num_threads=state_num_threads,
            pairs_per_thread=state_pairs_per_thread,
        )
        state_passing_kernel = StatePassingBwdAmpere(
            state_config,
            copy_bits_starts=state_copy_bits_starts,
            copy_bits_dstarts=state_copy_bits_dstarts,
            copy_bits_dinc=state_copy_bits_dinc,
            copy_bits_initial=state_copy_bits_initial,
            copy_bits_final=state_copy_bits_final,
        )

        chunk_increment_db_kernel = ChunkIncrementBwdDBAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        chunk_increment_boundary_kernel = ChunkIncrementBwdBoundaryAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        chunk_increment_param_kernel = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            n_d_tiles=n_d_tiles,
        )
        chunk_increment_du_kernel = ChunkIncrementBwdDUAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )

        chunk_scan_db_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dU_db_dummy_view,
            dB_scan_view,
            dU_prev_db_dummy_view,
            dB_prev_scan_view,
            dlogp_scan_view,
            dM_previous_scan_view,
            dM_current_scan_view,
        )
        chunk_scan_dcdr_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            chunk_starts_scan_view,
            dC_scan_view,
            dlogp_scan_view,
            dR_scan_view,
        )
        chunk_scan_dlp_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dlogp_scan_view,
        )
        chunk_scan_param_kernel(
            M_scan_view,
            K_scan_view,
            dlogp_param_view,
            dM_previous_param_view,
            dM_current_param_view,
            dR_param_view,
            dM_scan_view,
            dK_previous_scan_view,
            dK_current_scan_view,
        )
        chunk_scan_du_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dU_scan_view,
            dB_du_dummy_view,
            dU_prev_scan_view,
            dB_prev_du_dummy_view,
            dlogp_scan_view,
            dM_previous_scan_view,
            dM_current_scan_view,
        )
        chunk_scan_dz0_kernel(
            d_out_dz0_view,
            C_dz0_view,
            M_dz0_view,
            d_chunk_starts_scan_view,
        )

        state_passing_kernel(
            chunk_starts_state_view,
            d_chunk_starts_state_view,
            d_final_state_view,
            chunk_multiplier_state_view,
            d_increment_state_view,
            d_chunk_multiplier_state_view,
            d_initial_state_view,
        )

        chunk_increment_db_kernel(
            U_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            d_increment_dp_view,
            dB_increment_view,
            dM_sum_part_view,
        )
        chunk_increment_boundary_kernel(
            d_increment_boundary_view,
            B_prev_chunks_view,
            U_prev_chunks_view,
            M_increment_view,
            K_previous_increment_view,
            dU_prev_increment_view,
            dB_prev_increment_view,
            dMp0_view,
        )
        chunk_increment_param_kernel(
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dM_sum_part_view,
            dMp0_view,
            d_chunk_multiplier_increment_view,
            dM_increment_view,
            dK_previous_increment_view,
            dK_current_increment_view,
        )
        chunk_increment_du_kernel(
            d_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dU_increment_view,
        )

    return _v2x2ssd_bwd_host_wrapper


def _make_v2x2ssd_bwd_aot_host_wrapper(
    *,
    problem_shape: BackwardProblemShape,
    launch_cfg: BackwardLaunchConfig,
):
    (
        _batch_size,
        heads,
        bc_groups,
        _padded_time,
        P,
        D,
        n_chunks,
        chunk_size,
        n_d_tiles,
    ) = problem_shape
    (
        scan_num_threads_du,
        scan_num_threads_db,
        scan_num_threads_dcdr,
        scan_num_threads_param,
        state_num_threads,
        state_pairs_per_thread,
        state_copy_bits_starts,
        state_copy_bits_dstarts,
        state_copy_bits_dinc,
        state_copy_bits_initial,
        state_copy_bits_final,
        dz0_cta_tiler,
    ) = launch_cfg

    @cute.jit
    def _v2x2ssd_bwd_aot_host_wrapper(
        U_scan_view: cute.Tensor,
        B_scan_view: cute.Tensor,
        C_scan_view: cute.Tensor,
        M_scan_view: cute.Tensor,
        K_scan_view: cute.Tensor,
        d_out_scan_view: cute.Tensor,
        chunk_starts_scan_view: cute.Tensor,
        U_prev_scan_view: cute.Tensor,
        B_prev_scan_view: cute.Tensor,
        dU_scan_view: cute.Tensor,
        dB_scan_view: cute.Tensor,
        dU_prev_scan_view: cute.Tensor,
        dB_prev_scan_view: cute.Tensor,
        dlogp_scan_view: cute.Tensor,
        dlogp_param_view: cute.Tensor,
        dM_previous_scan_view: cute.Tensor,
        dM_current_scan_view: cute.Tensor,
        dM_previous_param_view: cute.Tensor,
        dM_current_param_view: cute.Tensor,
        dC_scan_view: cute.Tensor,
        dR_scan_view: cute.Tensor,
        dR_param_view: cute.Tensor,
        dM_scan_view: cute.Tensor,
        dK_previous_scan_view: cute.Tensor,
        dK_current_scan_view: cute.Tensor,
        d_out_dz0_view: cute.Tensor,
        C_dz0_view: cute.Tensor,
        M_dz0_view: cute.Tensor,
        d_chunk_starts_scan_view: cute.Tensor,
        chunk_starts_state_view: cute.Tensor,
        chunk_multiplier_state_view: cute.Tensor,
        d_chunk_starts_state_view: cute.Tensor,
        d_final_state_view: cute.Tensor,
        d_increment_state_view: cute.Tensor,
        d_initial_state_view: cute.Tensor,
        d_chunk_multiplier_state_view: cute.Tensor,
        U_increment_view: cute.Tensor,
        B_increment_view: cute.Tensor,
        M_increment_view: cute.Tensor,
        K_previous_increment_view: cute.Tensor,
        K_current_increment_view: cute.Tensor,
        d_increment_dp_view: cute.Tensor,
        d_increment_view: cute.Tensor,
        d_increment_boundary_view: cute.Tensor,
        B_prev_chunks_view: cute.Tensor,
        U_prev_chunks_view: cute.Tensor,
        dB_increment_view: cute.Tensor,
        dU_increment_view: cute.Tensor,
        dB_prev_increment_view: cute.Tensor,
        dU_prev_increment_view: cute.Tensor,
        dM_sum_part_view: cute.Tensor,
        dMp0_view: cute.Tensor,
        d_chunk_multiplier_increment_view: cute.Tensor,
        dM_increment_view: cute.Tensor,
        dK_previous_increment_view: cute.Tensor,
        dK_current_increment_view: cute.Tensor,
        dU_db_dummy_view: cute.Tensor,
        dU_prev_db_dummy_view: cute.Tensor,
        dB_du_dummy_view: cute.Tensor,
        dB_prev_du_dummy_view: cute.Tensor,
    ):
        tc_dtype = U_scan_view.element_type

        chunk_scan_db_kernel = ChunkScanBwdDBAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_db,
        )
        chunk_scan_dcdr_kernel = ChunkScanBwdDCDRAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_dcdr,
        )
        chunk_scan_dlp_kernel = ChunkScanBwdDLPAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_dcdr,
        )
        chunk_scan_param_kernel = ChunkScanBwdParamScanAmpere(
            chunk_size=chunk_size,
            num_threads=scan_num_threads_param,
        )
        chunk_scan_du_kernel = ChunkScanBwdDUAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            num_threads=scan_num_threads_du,
        )
        chunk_scan_dz0_kernel = ChunkScanBwdDZ0Ampere(
            tc_dtype,
            chunk_size=chunk_size,
            heads=heads,
            bc_groups=bc_groups,
            cta_tiler=dz0_cta_tiler,
        )

        state_config = _TileConfig(
            num_threads=state_num_threads,
            pairs_per_thread=state_pairs_per_thread,
        )
        state_passing_kernel = StatePassingBwdAmpere(
            state_config,
            copy_bits_starts=state_copy_bits_starts,
            copy_bits_dstarts=state_copy_bits_dstarts,
            copy_bits_dinc=state_copy_bits_dinc,
            copy_bits_initial=state_copy_bits_initial,
            copy_bits_final=state_copy_bits_final,
        )

        chunk_increment_db_kernel = ChunkIncrementBwdDBAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        chunk_increment_boundary_kernel = ChunkIncrementBwdBoundaryAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )
        chunk_increment_param_kernel = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            n_d_tiles=n_d_tiles,
        )
        chunk_increment_du_kernel = ChunkIncrementBwdDUAmpere(
            tc_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
            heads=heads,
            bc_groups=bc_groups,
            n_chunks=n_chunks,
        )

        chunk_scan_db_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dU_db_dummy_view,
            dB_scan_view,
            dU_prev_db_dummy_view,
            dB_prev_scan_view,
            dlogp_scan_view,
            dM_previous_scan_view,
            dM_current_scan_view,
        )
        chunk_scan_dcdr_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            chunk_starts_scan_view,
            dC_scan_view,
            dlogp_scan_view,
            dR_scan_view,
        )
        chunk_scan_dlp_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dlogp_scan_view,
        )
        chunk_scan_param_kernel(
            M_scan_view,
            K_scan_view,
            dlogp_param_view,
            dM_previous_param_view,
            dM_current_param_view,
            dR_param_view,
            dM_scan_view,
            dK_previous_scan_view,
            dK_current_scan_view,
        )
        chunk_scan_du_kernel(
            U_scan_view,
            B_scan_view,
            C_scan_view,
            M_scan_view,
            K_scan_view,
            d_out_scan_view,
            U_prev_scan_view,
            B_prev_scan_view,
            dU_scan_view,
            dB_du_dummy_view,
            dU_prev_scan_view,
            dB_prev_du_dummy_view,
            dlogp_scan_view,
            dM_previous_scan_view,
            dM_current_scan_view,
        )
        chunk_scan_dz0_kernel(
            d_out_dz0_view,
            C_dz0_view,
            M_dz0_view,
            d_chunk_starts_scan_view,
        )

        state_passing_kernel(
            chunk_starts_state_view,
            d_chunk_starts_state_view,
            d_final_state_view,
            chunk_multiplier_state_view,
            d_increment_state_view,
            d_chunk_multiplier_state_view,
            d_initial_state_view,
        )

        chunk_increment_db_kernel(
            U_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            d_increment_dp_view,
            dB_increment_view,
            dM_sum_part_view,
        )
        chunk_increment_boundary_kernel(
            d_increment_boundary_view,
            B_prev_chunks_view,
            U_prev_chunks_view,
            M_increment_view,
            K_previous_increment_view,
            dU_prev_increment_view,
            dB_prev_increment_view,
            dMp0_view,
        )
        chunk_increment_param_kernel(
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dM_sum_part_view,
            dMp0_view,
            d_chunk_multiplier_increment_view,
            dM_increment_view,
            dK_previous_increment_view,
            dK_current_increment_view,
        )
        chunk_increment_du_kernel(
            d_increment_view,
            B_increment_view,
            M_increment_view,
            K_previous_increment_view,
            K_current_increment_view,
            dU_increment_view,
        )

    return _v2x2ssd_bwd_aot_host_wrapper


def _make_backward_aot_runtime_args(
    runtime_args: tuple[torch.Tensor, ...],
    *,
    problem_shape: BackwardProblemShape,
) -> tuple[torch.Tensor, ...]:
    (
        U_t,
        B_t,
        C_t,
        M_t,
        K_t,
        K_previous_t,
        K_current_t,
        d_out_t,
        chunk_multiplier_t,
        chunk_starts_t,
        U_prev_state_t,
        B_prev_state_t,
        d_final_state_t,
        U_prev_chunks_t,
        B_prev_chunks_t,
        d_chunk_starts_t,
        d_increment_t,
        d_initial_state_t,
        d_chunk_multiplier_t,
        dU_scan_t,
        dU_prev_scan_t,
        dB_scan_t,
        dB_prev_scan_t,
        dC_scan_t,
        dU_db_dummy_t,
        dU_prev_db_dummy_t,
        dB_du_dummy_t,
        dB_prev_du_dummy_t,
        dlogp_t,
        dR_t,
        dM_previous_scratch_t,
        dM_current_scratch_t,
        dM_scan_t,
        dK_previous_scan_t,
        dK_current_scan_t,
        dB_increment_t,
        dB_prev_increment_t,
        dU_increment_t,
        dU_prev_increment_t,
        dM_sum_part_t,
        dMp0_t,
        dM_increment_t,
        dK_previous_increment_t,
        dK_current_increment_t,
    ) = runtime_args
    chunk_scan_specs = _chunk_scan_bwd_tensor_specs(problem_shape)
    state_passing_specs = _state_passing_bwd_tensor_specs(problem_shape)
    chunk_increment_specs = _chunk_increment_bwd_tensor_specs(problem_shape)
    (
        U_scan_spec,
        BC_scan_spec,
        M_scan_spec,
        K_scan_spec,
        chunk_starts_scan_spec,
        U_prev_scan_spec,
        B_prev_scan_spec,
        dlogp_scan_spec,
        dM_scan_scratch_spec,
        dR_scan_spec,
        d_param_scan_spec,
        dlogp_param_spec,
        dR_param_spec,
        d_out_dz0_spec,
        C_dz0_spec,
        M_dz0_spec,
        d_chunk_starts_scan_spec,
        dU_db_dummy_spec,
        dBC_scan_spec,
        dU_prev_dummy_spec,
        dB_prev_dummy_spec,
    ) = chunk_scan_specs
    (
        chunk_starts_state_spec,
        chunk_multiplier_state_spec,
        final_state_spec,
    ) = state_passing_specs
    (
        U_increment_spec,
        dU_increment_spec,
        B_increment_input_spec,
        M_increment_spec,
        K_increment_spec,
        d_increment_spec,
        d_increment_dp_spec,
        d_increment_boundary_spec,
        U_prev_chunks_spec,
        B_prev_chunks_input_spec,
        dM_sum_part_spec,
        dMp0_spec,
        d_chunk_multiplier_increment_spec,
        d_param_increment_spec,
        dB_increment_spec,
        dB_prev_increment_spec,
    ) = chunk_increment_specs
    return (
        make_runtime_tensor_spec_view(U_t, U_scan_spec),
        make_runtime_tensor_spec_view(B_t, BC_scan_spec),
        make_runtime_tensor_spec_view(C_t, BC_scan_spec),
        make_runtime_tensor_spec_view(M_t, M_scan_spec),
        make_runtime_tensor_spec_view(K_t, K_scan_spec),
        make_runtime_tensor_spec_view(d_out_t, U_scan_spec),
        make_runtime_tensor_spec_view(chunk_starts_t, chunk_starts_scan_spec),
        make_runtime_tensor_spec_view(U_prev_state_t, U_prev_scan_spec),
        make_runtime_tensor_spec_view(B_prev_state_t, B_prev_scan_spec),
        make_runtime_tensor_spec_view(dU_scan_t, U_scan_spec),
        make_runtime_tensor_spec_view(dB_scan_t, dBC_scan_spec),
        make_runtime_tensor_spec_view(dU_prev_scan_t, dU_prev_dummy_spec),
        make_runtime_tensor_spec_view(dB_prev_scan_t, dB_prev_dummy_spec),
        make_runtime_tensor_spec_view(dlogp_t, dlogp_scan_spec),
        make_runtime_tensor_spec_view(dlogp_t, dlogp_param_spec),
        make_runtime_tensor_spec_view(dM_previous_scratch_t, dM_scan_scratch_spec),
        make_runtime_tensor_spec_view(dM_current_scratch_t, dM_scan_scratch_spec),
        make_runtime_tensor_spec_view(dM_previous_scratch_t, d_param_scan_spec),
        make_runtime_tensor_spec_view(dM_current_scratch_t, d_param_scan_spec),
        make_runtime_tensor_spec_view(dC_scan_t, dBC_scan_spec),
        make_runtime_tensor_spec_view(dR_t, dR_scan_spec),
        make_runtime_tensor_spec_view(dR_t, dR_param_spec),
        make_runtime_tensor_spec_view(dM_scan_t, d_param_scan_spec),
        make_runtime_tensor_spec_view(dK_previous_scan_t, d_param_scan_spec),
        make_runtime_tensor_spec_view(dK_current_scan_t, d_param_scan_spec),
        make_runtime_tensor_spec_view(d_out_t, d_out_dz0_spec),
        make_runtime_tensor_spec_view(C_t, C_dz0_spec),
        make_runtime_tensor_spec_view(M_t, M_dz0_spec),
        make_runtime_tensor_spec_view(d_chunk_starts_t, d_chunk_starts_scan_spec),
        make_runtime_tensor_spec_view(chunk_starts_t, chunk_starts_state_spec),
        make_runtime_tensor_spec_view(chunk_multiplier_t, chunk_multiplier_state_spec),
        make_runtime_tensor_spec_view(d_chunk_starts_t, chunk_starts_state_spec),
        make_runtime_tensor_spec_view(d_final_state_t, final_state_spec),
        make_runtime_tensor_spec_view(d_increment_t, chunk_starts_state_spec),
        make_runtime_tensor_spec_view(d_initial_state_t, final_state_spec),
        make_runtime_tensor_spec_view(
            d_chunk_multiplier_t, chunk_multiplier_state_spec
        ),
        make_runtime_tensor_spec_view(U_t, U_increment_spec),
        make_runtime_tensor_spec_view(B_t, B_increment_input_spec),
        make_runtime_tensor_spec_view(M_t, M_increment_spec),
        make_runtime_tensor_spec_view(K_previous_t, K_increment_spec),
        make_runtime_tensor_spec_view(K_current_t, K_increment_spec),
        make_runtime_tensor_spec_view(d_increment_t, d_increment_dp_spec),
        make_runtime_tensor_spec_view(d_increment_t, d_increment_spec),
        make_runtime_tensor_spec_view(d_increment_t, d_increment_boundary_spec),
        make_runtime_tensor_spec_view(B_prev_chunks_t, B_prev_chunks_input_spec),
        make_runtime_tensor_spec_view(U_prev_chunks_t, U_prev_chunks_spec),
        make_runtime_tensor_spec_view(dB_increment_t, dB_increment_spec),
        make_runtime_tensor_spec_view(dU_increment_t, dU_increment_spec),
        make_runtime_tensor_spec_view(dB_prev_increment_t, dB_prev_increment_spec),
        make_runtime_tensor_spec_view(dU_prev_increment_t, U_prev_chunks_spec),
        make_runtime_tensor_spec_view(dM_sum_part_t, dM_sum_part_spec),
        make_runtime_tensor_spec_view(dMp0_t, dMp0_spec),
        make_runtime_tensor_spec_view(
            d_chunk_multiplier_t, d_chunk_multiplier_increment_spec
        ),
        make_runtime_tensor_spec_view(dM_increment_t, d_param_increment_spec),
        make_runtime_tensor_spec_view(dK_previous_increment_t, d_param_increment_spec),
        make_runtime_tensor_spec_view(dK_current_increment_t, d_param_increment_spec),
        make_runtime_tensor_spec_view(dU_db_dummy_t, dU_db_dummy_spec),
        make_runtime_tensor_spec_view(dU_prev_db_dummy_t, dU_prev_dummy_spec),
        make_runtime_tensor_spec_view(dB_du_dummy_t, dBC_scan_spec),
        make_runtime_tensor_spec_view(dB_prev_du_dummy_t, dB_prev_dummy_spec),
    )


def _make_packaged_v2x2ssd_bwd_callable(
    packaged: object,
    *,
    problem_shape: BackwardProblemShape,
):
    def _packaged_v2x2ssd_bwd_callable(*runtime_args):
        return packaged(
            *_make_backward_aot_runtime_args(
                runtime_args,
                problem_shape=problem_shape,
            )
        )

    return _packaged_v2x2ssd_bwd_callable


def _get_compiled_v2x2ssd_bwd_kernel(
    compile_artifacts: BackwardCompileArtifacts,
):
    from slinoss.ops.v2x2ssd.cute.aot import (
        BackwardAOTSpec,
        ChunkScanBackwardConfig,
        StatePassingBackwardConfig,
        try_load_packaged_v2x2ssd_bwd_function,
    )
    from slinoss.ops.v2x2ssd.cute.tuning.hardware import current_hardware_fingerprint

    compiled = _BWD_HOST_CACHE.get(compile_artifacts.cache_key)
    if compiled is not None:
        note_cache_event("cute.v2x2ssd.bwd.host_compile", hit=True)
        return compiled

    device_index = (
        int(compile_artifacts.device_index)
        if int(compile_artifacts.device_index) >= 0
        else torch.cuda.current_device()
    )
    backward_aot_spec = BackwardAOTSpec(
        arch_tag=current_hardware_fingerprint(device_index=device_index).arch_tag,
        P=int(compile_artifacts.problem_shape[4]),
        D=int(compile_artifacts.problem_shape[5]),
        chunk_size=int(compile_artifacts.problem_shape[7]),
        bc_groups=(
            None
            if int(compile_artifacts.problem_shape[2])
            == int(compile_artifacts.problem_shape[1])
            else int(compile_artifacts.problem_shape[2])
        ),
        tc_dtype_name={
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }[compile_artifacts.tc_dtype],
        chunk_scan_config=ChunkScanBackwardConfig(
            num_threads_du=int(compile_artifacts.launch_cfg[0]),
            num_threads_db=int(compile_artifacts.launch_cfg[1]),
            num_threads_dcdr=int(compile_artifacts.launch_cfg[2]),
            num_threads_param=int(compile_artifacts.launch_cfg[3]),
        ),
        state_passing_config=StatePassingBackwardConfig(
            num_threads=int(compile_artifacts.launch_cfg[4]),
            pairs_per_thread=int(compile_artifacts.launch_cfg[5]),
        ),
    )
    packaged = try_load_packaged_v2x2ssd_bwd_function(backward_aot_spec)
    if packaged is not None:
        note_cache_event("cute.v2x2ssd.bwd.host_aot", hit=True)
        wrapped_packaged = _make_packaged_v2x2ssd_bwd_callable(
            packaged,
            problem_shape=compile_artifacts.problem_shape,
        )
        _BWD_HOST_CACHE[compile_artifacts.cache_key] = wrapped_packaged
        return wrapped_packaged

    note_cache_event("cute.v2x2ssd.bwd.host_aot", hit=False)
    note_cache_event("cute.v2x2ssd.bwd.host_compile", hit=False)
    host_wrapper = _make_v2x2ssd_bwd_host_wrapper(
        problem_shape=compile_artifacts.problem_shape,
        launch_cfg=compile_artifacts.launch_cfg,
    )
    compiled = cute.compile(
        host_wrapper,
        *compile_artifacts.compile_args,
        options="--enable-tvm-ffi",
    )
    _BWD_HOST_CACHE[compile_artifacts.cache_key] = compiled
    return compiled


def compile_v2x2ssd_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dcdr: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
) -> object:
    compile_artifacts = _make_backward_compile_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
    )
    return _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts)


def _materialize_backward_public_outputs(
    runtime_artifacts: BackwardRuntimeArtifacts,
    *,
    time_steps: int,
    U_dtype: torch.dtype,
    B_dtype: torch.dtype,
    C_dtype: torch.dtype,
    initial_state_dtype: torch.dtype | None,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    outputs = runtime_artifacts.outputs
    (
        _batch_size,
        _heads,
        bc_groups,
        _padded_time,
        _P,
        _D,
        _n_chunks,
        _chunk_size,
        _n_d_tiles,
    ) = runtime_artifacts.problem_shape
    outputs.dU_scan.add_(outputs.dU_increment)
    outputs.dU_prev_scan.add_(outputs.dU_prev_increment)
    outputs.dB_scan.add_(outputs.dB_increment)
    outputs.dB_prev_scan.add_(outputs.dB_prev_increment)
    outputs.dM_scan[:, :, :, 0, :, :].add_(outputs.dM_increment)
    outputs.dK_scan.add_(outputs.dK_increment)

    dU_public = _fold_chunk_boundary_carries(outputs.dU_scan, outputs.dU_prev_scan)
    dB_public = _fold_chunk_boundary_carries(outputs.dB_scan, outputs.dB_prev_scan)
    dB_grouped = _reduce_heads_to_bc_groups(dB_public, bc_groups=bc_groups)
    dC_grouped = _reduce_heads_to_bc_groups(outputs.dC_scan, bc_groups=bc_groups)
    dB_prev_grouped = _reduce_heads_to_bc_groups(
        outputs.dB_prev_scan,
        bc_groups=bc_groups,
    )

    return (
        _public_from_chunked(dU_public, T=time_steps, dtype=U_dtype),
        _public_from_param_scan(outputs.dM_scan, T=time_steps),
        _public_from_packed_dk(outputs.dK_scan, time_steps=time_steps),
        _public_from_chunked(dB_grouped, T=time_steps, dtype=B_dtype),
        _public_from_chunked(dC_grouped, T=time_steps, dtype=C_dtype),
        _materialize_public_output(
            outputs.d_initial_state,
            outputs.d_initial_state,
            dtype=initial_state_dtype or torch.float32,
        ),
        _materialize_public_output(
            dB_prev_grouped,
            dB_prev_grouped[:, :, 0, :],
            dtype=B_prev.dtype if B_prev is not None else B_dtype,
        ),
        _materialize_public_output(
            outputs.dU_prev_scan,
            outputs.dU_prev_scan[:, :, 0, :],
            dtype=U_prev.dtype if U_prev is not None else U_dtype,
        ),
    )


def _run_v2x2ssd_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dcdr: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
    initial_state_dtype: torch.dtype | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
    validate_runtime_contract: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    runtime_artifacts = _make_backward_runtime_artifacts(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=validate_runtime_contract,
    )
    compile_artifacts = _make_backward_compile_artifacts_from_runtime_artifacts(
        runtime_artifacts
    )
    compiled = _get_compiled_v2x2ssd_bwd_kernel(compile_artifacts)
    compiled(*runtime_artifacts.runtime_args)
    _record_tensors_on_current_stream(*runtime_artifacts.runtime_args)
    return _materialize_backward_public_outputs(
        runtime_artifacts,
        time_steps=U.shape[2],
        U_dtype=U.dtype,
        B_dtype=B.dtype,
        C_dtype=C.dtype,
        initial_state_dtype=initial_state_dtype,
        B_prev=B_prev,
        U_prev=U_prev,
    )


def _v2x2ssd_bwd_cute_prevalidated(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dcdr: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
    initial_state_dtype: torch.dtype | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _run_v2x2ssd_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        initial_state_dtype=initial_state_dtype,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=False,
    )


def v2x2ssd_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dcdr: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dU, dM, dK, dB, dC, _d_initial_state, _dB_prev, _dU_prev = _run_v2x2ssd_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=True,
    )
    return dU, dM, dK, dB, dC


def v2x2ssd_bwd_stateful_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    initial_state_dtype: torch.dtype | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    scan_num_threads_du: int = 128,
    scan_num_threads_db: int = 128,
    scan_num_threads_dcdr: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
    prepared_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return _run_v2x2ssd_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        m_chunk,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        scan_num_threads_du=scan_num_threads_du,
        scan_num_threads_db=scan_num_threads_db,
        scan_num_threads_dcdr=scan_num_threads_dcdr,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        initial_state_dtype=initial_state_dtype,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=prepared_inputs,
        validate_runtime_contract=True,
    )


__all__ = [
    "compile_v2x2ssd_bwd_cute",
    "v2x2ssd_bwd_cute",
    "v2x2ssd_bwd_stateful_cute",
]
