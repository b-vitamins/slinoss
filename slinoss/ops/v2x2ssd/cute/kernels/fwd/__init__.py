"""CuTe forward kernels for the ``v2x2ssd`` staged pipeline."""

from __future__ import annotations

import torch
from cuda.bindings import driver as cuda
import cutlass.cute as cute
from typing import Callable

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
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
    _torch_to_cutlass_dtype,
)
from .state_passing import StatePassingFwdAmpere


_CHUNK_INCREMENT_CACHE: dict[tuple, object] = {}
_STATE_PASSING_CACHE: dict[tuple, object] = {}
_CHUNK_SCAN_CACHE: dict[tuple, object] = {}
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


def _resolve_chunk_scan_n_block_size(L: int, requested: int) -> int:
    if requested <= 0:
        raise ValueError("n_block_size must be positive.")
    if requested % 16 != 0:
        raise ValueError("n_block_size must be a multiple of 16.")
    if L % requested == 0:
        return requested
    limit = min(L, requested)
    candidate = limit - (limit % 16)
    while candidate >= 16:
        if L % candidate == 0:
            return candidate
        candidate -= 16
    raise ValueError(
        f"No valid n_block_size multiple of 16 divides chunk_size={L} "
        f"for requested n_block_size={requested}."
    )


def _iter_chunk_scan_n_block_candidates(L: int, requested: int):
    candidate = _resolve_chunk_scan_n_block_size(L, requested)
    while candidate >= 16:
        if L % candidate == 0:
            yield candidate
        candidate -= 16


def _chunk_scan_supported_tile_families(L: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (m_block_size, num_threads)
        for m_block_size, num_threads in ChunkScanFwdAmpere._SUPPORTED_TILE_FAMILIES
        if m_block_size <= L
    )


def _chunk_scan_device_label(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def _resolve_chunk_scan_launch_cfg(
    *,
    D: int,
    P: int,
    L: int,
    tc_dtype: torch.dtype,
    output_dtype: torch.dtype,
    device_index: int,
    requested_m_block_size: int | None,
    requested_n_block_size: int,
    requested_num_threads: int,
) -> tuple[int, int, int]:
    supported_families = _chunk_scan_supported_tile_families(L)
    if not supported_families:
        raise ValueError(
            f"chunk_size={L} is too small for the CuTe chunk_scan tile families; "
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
                f"Unsupported chunk_scan m_block_size={resolved_m_block_size} for chunk_size={L}. "
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
                    f"chunk_size={L}. Supported tile families use num_threads in "
                    f"{{{supported_threads}}}."
                )

    cutlass_tc_dtype = _torch_to_cutlass_dtype(tc_dtype)
    cutlass_out_dtype = _torch_to_cutlass_dtype(output_dtype)
    attempts: list[str] = []
    for m_block_size, num_threads in candidate_families:
        for n_block_size in _iter_chunk_scan_n_block_candidates(
            L, requested_n_block_size
        ):
            kernel = ChunkScanFwdAmpere(
                D=D,
                P=P,
                L=L,
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
        f"(chunk_size={L}, D={D}, P={P}). Tried: {attempt_summary}"
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


def _make_static_tensor_view(
    tensor: cute.Tensor,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
) -> cute.Tensor:
    shape, stride = spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_runtime_tensor_view(
    tensor: torch.Tensor,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
) -> torch.Tensor:
    shape, stride = spec
    return torch.as_strided(tensor, size=shape, stride=stride)


def _chunk_increment_tensor_specs(
    spec: tuple[int, ...],
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
    Bsz, H, T_pad, P, D, n_chunks, _L = spec
    BH = Bsz * H
    BHC = BH * n_chunks
    return (
        _make_tensor_spec((P, T_pad, BH), stride=(1, P, T_pad * P)),
        _make_tensor_spec((D, T_pad, BH), stride=(1, D, T_pad * D)),
        _make_tensor_spec((2, T_pad, BH), stride=(1, 2, T_pad * 2)),
        _make_tensor_spec((2, T_pad, BH), stride=(1, 4, T_pad * 4)),
        _make_tensor_spec((P, BH), stride=(1, P)),
        _make_tensor_spec((D, BH), stride=(1, D)),
        _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D)),
        _make_tensor_spec((2, BHC), stride=(1, 2)),
    )


def _resolve_chunk_increment_problem_spec(
    *,
    U: torch.Tensor,
    B: torch.Tensor,
    U_tc: torch.Tensor,
    chunk_size: int,
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    Bsz, H, _T, P = map(int, U.shape)
    D = int(B.shape[-1])
    L = int(chunk_size)
    T_pad = int(U_tc.shape[2])
    n_chunks = T_pad // L
    problem_spec = (Bsz, H, T_pad, P, D, n_chunks, L)
    cta_tiler = _resolve_chunk_increment_cta_tiler(D=D)
    return problem_spec, cta_tiler


def _make_chunk_increment_launch_artifacts(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    U_prev0: torch.Tensor,
    B_prev0: torch.Tensor,
    inc_chunk: torch.Tensor,
    m_chunk_tile: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    has_prev: bool,
) -> tuple[
    tuple[int, ...],
    tuple[int, int, int],
    tuple[object, ...],
    tuple[object, ...],
    tuple[object, ...],
]:
    problem_spec, cta_tiler = _resolve_chunk_increment_problem_spec(
        U=U,
        B=B,
        U_tc=U_tc,
        chunk_size=chunk_size,
    )
    (
        u_spec,
        b_spec,
        m_spec,
        k_spec,
        u_prev_spec,
        b_prev_spec,
        inc_spec,
        m_chunk_spec,
    ) = _chunk_increment_tensor_specs(problem_spec)
    alignments = (
        _assumed_align(U_tc),
        _assumed_align(B_tc),
        _assumed_align(M_f),
        _assumed_align(K_f),
        _assumed_align(U_prev0),
        _assumed_align(B_prev0),
        _assumed_align(inc_chunk),
        _assumed_align(m_chunk_tile),
    )
    compile_args = (
        _make_fake_tensor_arg(
            U_tc, shape=u_spec[0], stride=u_spec[1], align=alignments[0]
        ),
        _make_fake_tensor_arg(
            B_tc, shape=b_spec[0], stride=b_spec[1], align=alignments[1]
        ),
        _make_fake_tensor_arg(
            M_f, shape=m_spec[0], stride=m_spec[1], align=alignments[2]
        ),
        _make_fake_tensor_arg(
            K_f, shape=k_spec[0], stride=k_spec[1], align=alignments[3]
        ),
        _make_fake_tensor_arg(
            U_prev0, shape=u_prev_spec[0], stride=u_prev_spec[1], align=alignments[4]
        ),
        _make_fake_tensor_arg(
            B_prev0, shape=b_prev_spec[0], stride=b_prev_spec[1], align=alignments[5]
        ),
        _make_fake_tensor_arg(
            inc_chunk, shape=inc_spec[0], stride=inc_spec[1], align=alignments[6]
        ),
        _make_fake_tensor_arg(
            m_chunk_tile,
            shape=m_chunk_spec[0],
            stride=m_chunk_spec[1],
            align=alignments[7],
        ),
        _compile_env_stream_placeholder(),
    )
    runtime_args = (
        _make_runtime_tensor_view(U_tc, u_spec),
        _make_runtime_tensor_view(B_tc, b_spec),
        _make_runtime_tensor_view(M_f, m_spec),
        _make_runtime_tensor_view(K_f, k_spec),
        _make_runtime_tensor_view(U_prev0, u_prev_spec),
        _make_runtime_tensor_view(B_prev0, b_prev_spec),
        _make_runtime_tensor_view(inc_chunk, inc_spec),
        _make_runtime_tensor_view(m_chunk_tile, m_chunk_spec),
    )
    cache_key = _chunk_increment_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        chunk_size=problem_spec[-1],
        has_prev=has_prev,
        cta_tiler=cta_tiler,
        alignments=alignments,
    )
    return problem_spec, cta_tiler, compile_args, runtime_args, cache_key


def _state_passing_tensor_specs(
    spec: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]:
    B, H, C, P, D = spec
    return (
        _make_tensor_spec((B, H, C, P, D)),
        _make_tensor_spec((B, H, C, 2)),
        _make_tensor_spec((B, H, C, P, D)),
        _make_tensor_spec((B, H, P, D)),
    )


def _chunk_scan_tensor_specs(
    spec: tuple[int, ...],
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
    Bsz, H, _T_pad, P, D, n_chunks, L = spec
    BH = Bsz * H
    BHC = BH * n_chunks
    return (
        _make_tensor_spec((BHC, L, 1, P)),
        _make_tensor_spec((BHC, L, 1, D)),
        _make_tensor_spec((BHC, L, 2)),
        _make_tensor_spec((BHC, L, 2, 2)),
        _make_tensor_spec((BHC, P, 1, D)),
        _make_tensor_spec((BH, P)),
        _make_tensor_spec((BH, D)),
        _make_tensor_spec((BHC, L, 1, P)),
    )


def _prepare_time_operand(
    tensor: torch.Tensor,
    *,
    T_pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        tensor.dtype == dtype
        and int(tensor.shape[2]) == T_pad
        and tensor.is_contiguous()
    ):
        tensor = tensor
    else:
        tensor = _pad_zero_time(tensor, T_pad=T_pad, dtype=dtype)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        tensor = _ensure_min_alignment(tensor, min_align=16)
    return tensor


def _prepare_m_operand(M: torch.Tensor, *, T_pad: int) -> torch.Tensor:
    if int(M.shape[2]) == T_pad and M.dtype == torch.float32 and M.is_contiguous():
        return M
    return _pad_m_identity(M, T_pad=T_pad)


def _chunk_increment_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    chunk_size: int,
    has_prev: bool,
    cta_tiler: tuple[int, int, int],
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "chunk_increment_fwd",
        device_index,
        tc_dtype,
        U_shape,
        M_shape,
        K_shape,
        B_shape,
        int(chunk_size),
        has_prev,
        tuple(int(dim) for dim in cta_tiler),
        alignments,
    )


def _state_passing_key(
    *,
    device_index: int,
    inc_shape: tuple[int, ...],
    m_chunk_shape: tuple[int, ...],
    initial_shape: tuple[int, ...] | None,
    num_threads: int,
    vecs_per_thread: int,
    copy_bits_in: int,
    copy_bits_out: int,
    copy_bits_state_in: int,
    copy_bits_state_out: int,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "state_passing_fwd",
        device_index,
        inc_shape,
        m_chunk_shape,
        initial_shape,
        int(num_threads),
        int(vecs_per_thread),
        int(copy_bits_in),
        int(copy_bits_out),
        int(copy_bits_state_in),
        int(copy_bits_state_out),
        alignments,
    )


def _chunk_scan_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    out_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    chunk_starts_shape: tuple[int, ...],
    chunk_size: int,
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
        U_shape,
        M_shape,
        K_shape,
        B_shape,
        C_shape,
        chunk_starts_shape,
        int(chunk_size),
        int(m_block_size),
        int(n_block_size),
        int(num_threads),
        has_prev,
        alignments,
    )


def _resolve_chunk_increment_cta_tiler(*, D: int) -> tuple[int, int, int]:
    # The 96-wide N tile is efficient when it covers the full state width, but
    # mixed full+tail tiling can perturb the current epilogue path on realistic
    # D=2N mixer shapes. Pick a tail-safe family instead of changing semantics.
    if D <= 96 or D % 96 == 0:
        return (64, 96, 32)
    return (64, 64, 32)


def _make_chunk_scan_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    Bsz, H, T_pad, P, D, n_chunks, L = spec
    m_block_size, n_block_size, num_threads = cfg

    @cute.jit
    def _chunk_scan_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        C_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        Z0_t: cute.Tensor,
        U_prev0_t: cute.Tensor,
        B_prev0_t: cute.Tensor,
        Out_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        u_spec, b_spec, m_spec, k_spec, z0_spec, u_prev_spec, b_prev_spec, out_spec = (
            _chunk_scan_tensor_specs((Bsz, H, T_pad, P, D, n_chunks, L))
        )
        mU = _make_static_tensor_view(U_t, u_spec)
        mB = _make_static_tensor_view(B_t, b_spec)
        mC = _make_static_tensor_view(C_t, b_spec)
        mM = _make_static_tensor_view(M_t, m_spec)
        mK = _make_static_tensor_view(K_t, k_spec)
        mZ0 = _make_static_tensor_view(Z0_t, z0_spec)
        mU_prev0 = _make_static_tensor_view(U_prev0_t, u_prev_spec)
        mB_prev0 = _make_static_tensor_view(B_prev0_t, b_prev_spec)
        mOut = _make_static_tensor_view(Out_t, out_spec)

        chunk_scan = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=L,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
        )
        chunk_scan.call_on_stream(
            mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut, stream
        )

    return _chunk_scan_host_wrapper


def _make_chunk_increment_host_wrapper(
    *,
    spec: tuple[int, ...],
    cta_tiler: tuple[int, int, int],
):
    Bsz, H, T_pad, P, D, n_chunks, L = spec

    @cute.jit
    def _chunk_increment_host_wrapper(
        U_t: cute.Tensor,
        B_t: cute.Tensor,
        M_t: cute.Tensor,
        K_t: cute.Tensor,
        U_prev0_t: cute.Tensor,
        B_prev0_t: cute.Tensor,
        Inc_t: cute.Tensor,
        Mchunk_t: cute.Tensor,
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
        ) = _chunk_increment_tensor_specs((Bsz, H, T_pad, P, D, n_chunks, L))
        mU = _make_static_tensor_view(U_t, u_spec)
        mB = _make_static_tensor_view(B_t, b_spec)
        mM = _make_static_tensor_view(M_t, m_spec)
        mKprev = _make_static_tensor_view(K_t, k_spec)
        mKcurr = cute.make_tensor(mKprev.iterator + 2, mKprev.layout)
        mU_prev0 = _make_static_tensor_view(U_prev0_t, u_prev_spec)
        mB_prev0 = _make_static_tensor_view(B_prev0_t, b_prev_spec)
        mInc = _make_static_tensor_view(Inc_t, inc_spec)
        mMchunk = _make_static_tensor_view(Mchunk_t, m_chunk_spec)

        chunk_increment = ChunkIncrementFwdAmpere(
            mU.element_type,
            chunk_size=L,
            cta_tiler=cta_tiler,
        )
        chunk_increment.call_on_stream(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mU_prev0,
            mB_prev0,
            mInc,
            mMchunk,
            stream,
        )

    return _chunk_increment_host_wrapper


def _make_state_passing_host_wrapper(*, spec: tuple[int, ...], cfg: tuple[int, ...]):
    B, H, C, P, D = spec
    (
        num_threads,
        vecs_per_thread,
        copy_bits_in,
        copy_bits_out,
        copy_bits_state_in,
        copy_bits_state_out,
        has_init,
    ) = cfg

    @cute.jit
    def _state_passing_host_wrapper(
        Inc_t: cute.Tensor,
        M_t: cute.Tensor,
        OutStarts_t: cute.Tensor,
        OutFinal_t: cute.Tensor,
        Init_t: cute.Tensor,
        stream: cuda.CUstream,
    ):
        inc_spec, m_spec, out_starts_spec, out_final_spec = _state_passing_tensor_specs(
            (B, H, C, P, D)
        )
        mInc = _make_static_tensor_view(Inc_t, inc_spec)
        mM = _make_static_tensor_view(M_t, m_spec)
        mOutStarts = _make_static_tensor_view(OutStarts_t, out_starts_spec)
        mOutFinal = _make_static_tensor_view(OutFinal_t, out_final_spec)
        mInit = _make_static_tensor_view(Init_t, out_final_spec)

        state_passing = StatePassingFwdAmpere(
            num_threads=num_threads,
            vecs_per_thread=vecs_per_thread,
            copy_bits_in=copy_bits_in,
            copy_bits_out=copy_bits_out,
            copy_bits_state_in=copy_bits_state_in,
            copy_bits_state_out=copy_bits_state_out,
            has_init=has_init,
        )
        state_passing.call_on_stream(mInc, mM, mOutStarts, mOutFinal, mInit, stream)

    return _state_passing_host_wrapper


def _compile_chunk_increment_kernel_impl(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev0: torch.Tensor | None = None,
    B_prev0: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor, Callable[[], None]]:
    """Compile the forward chunk-increment kernel, allocate outputs, and build a launcher."""
    if (U_prev0 is None) ^ (B_prev0 is None):
        raise ValueError(
            "U_prev0 and B_prev0 must be passed together (or both omitted)."
        )
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if U.dtype != B.dtype:
        raise ValueError("U and B must share dtype.")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D):
        raise ValueError("B must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if D % 2 != 0:
        raise ValueError("B last dim must be divisible by 2.")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    U_tc = _ensure_min_alignment(U_tc, min_align=16)
    B_tc = _ensure_min_alignment(B_tc, min_align=16)

    if U_prev0 is None:
        U_prev, B_prev = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=Bsz,
            heads=H,
            P=P,
            D=D,
        )
    else:
        if U_prev0.shape != (Bsz, H, P) or B_prev0.shape != (Bsz, H, D):
            raise ValueError("U_prev0/B_prev0 must be (B,H,P)/(B,H,D).")
        U_prev = U_prev0.to(dtype=tc_dtype).contiguous()
        B_prev = B_prev0.to(dtype=tc_dtype).contiguous()
        U_prev = _ensure_min_alignment(U_prev, min_align=16)
        B_prev = _ensure_min_alignment(B_prev, min_align=16)

    batch_head_count = Bsz * H
    batch_head_chunk_count = batch_head_count * n_chunks

    inc_chunk = torch.zeros(
        (batch_head_chunk_count, P, D), device=U.device, dtype=torch.float32
    )
    m_chunk_tile = torch.zeros(
        (batch_head_chunk_count, 2), device=U.device, dtype=torch.float32
    )

    (
        problem_spec,
        cta_tiler,
        compile_args,
        runtime_args,
        cache_key,
    ) = _make_chunk_increment_launch_artifacts(
        U=U,
        M=M,
        K=K,
        B=B,
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev0=U_prev,
        B_prev0=B_prev,
        inc_chunk=inc_chunk,
        m_chunk_tile=m_chunk_tile,
        chunk_size=L,
        compute_dtype=compute_dtype,
        has_prev=U_prev0 is not None,
    )
    runtime_tensors = (
        U_tc,
        B_tc,
        M_f,
        K_f,
        U_prev,
        B_prev,
        inc_chunk,
        m_chunk_tile,
    )

    compiled = _CHUNK_INCREMENT_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_chunk_increment_host_wrapper(
            spec=problem_spec,
            cta_tiler=cta_tiler,
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_INCREMENT_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*runtime_tensors)

    return compiled, inc_chunk, m_chunk_tile, launch


def compile_chunk_increment_kernel(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev0: torch.Tensor | None = None,
    B_prev0: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Compile the forward chunk-increment kernel and allocate outputs."""
    compiled, inc_chunk, m_chunk_tile, _launch = _compile_chunk_increment_kernel_impl(
        U,
        M,
        K,
        B,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
    )
    return compiled, inc_chunk, m_chunk_tile


def chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    U_prev0: torch.Tensor | None = None,
    B_prev0: torch.Tensor | None = None,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin public wrapper over the compiled forward chunk-increment kernel."""
    _compiled, inc_chunk, m_chunk_tile, launch = _compile_chunk_increment_kernel_impl(
        U,
        M,
        K,
        B,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
    )
    launch()
    batch_head_count = int(U.shape[0]) * int(U.shape[1])
    n_chunks = inc_chunk.shape[0] // batch_head_count
    return (
        inc_chunk.reshape(U.shape[0], U.shape[1], n_chunks, U.shape[-1], B.shape[-1]),
        m_chunk_tile.reshape(U.shape[0], U.shape[1], n_chunks, 2),
    )


def _compile_state_passing_kernel_impl(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
) -> tuple[object, torch.Tensor, torch.Tensor, Callable[[], None]]:
    """Compile the forward state-passing kernel, allocate outputs, and build a launcher."""
    if inc.ndim != 5:
        raise ValueError(f"inc must be (B,H,C,P,D). Got {tuple(inc.shape)}.")
    if m_chunk.ndim != 4 or m_chunk.shape[-1] != 2:
        raise ValueError(f"m_chunk must be (B,H,C,2). Got {tuple(m_chunk.shape)}.")
    if inc.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if inc.dtype != torch.float32 or m_chunk.dtype != torch.float32:
        raise TypeError("inc and m_chunk must be float32 at the stage boundary.")

    B, H, C, P, D = map(int, inc.shape)
    if D % 2 != 0:
        raise ValueError("inc last dim must be divisible by 2.")
    if tuple(m_chunk.shape[:3]) != (B, H, C):
        raise ValueError("m_chunk leading dims must match inc.")
    if initial_states is not None:
        if tuple(initial_states.shape) != (B, H, P, D):
            raise ValueError("initial_states must be (B,H,P,D) and match inc.")
        if initial_states.dtype != torch.float32:
            raise TypeError("initial_states must be float32.")

    out_starts = torch.empty((B, H, C, P, D), device=inc.device, dtype=torch.float32)
    out_final = torch.empty((B, H, P, D), device=inc.device, dtype=torch.float32)
    # The state-passing kernel defines the live state domain but may leave
    # untouched lanes outside that domain on issue-9-style shapes. The public
    # final-state output must not depend on stale allocator contents.
    out_final.zero_()
    inc_spec, m_spec, out_starts_spec, out_final_spec = _state_passing_tensor_specs(
        (B, H, C, P, D)
    )

    inc_c = inc.contiguous()
    m_c = m_chunk.contiguous()
    init_c = initial_states.contiguous() if initial_states is not None else None

    if init_c is None:
        init_arg = inc_c
        has_init = False
    else:
        init_arg = init_c
        has_init = True

    S = P * D
    elems_per_thread = 2 * vecs_per_thread
    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc_c,
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        out_starts,
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state_in = 32
    copy_bits_state_out = 32

    alignments = (
        _assumed_align(inc_c),
        _assumed_align(m_c),
        _assumed_align(out_starts),
        _assumed_align(out_final),
        _assumed_align(init_arg),
    )
    keepalive = (inc_c, m_c, out_starts, out_final, init_c)
    runtime_args = (
        _make_runtime_tensor_view(inc_c, inc_spec),
        _make_runtime_tensor_view(m_c, m_spec),
        _make_runtime_tensor_view(out_starts, out_starts_spec),
        _make_runtime_tensor_view(out_final, out_final_spec),
        _make_runtime_tensor_view(init_arg, out_final_spec),
    )
    compile_args = (
        _make_fake_tensor_arg(
            inc_c, shape=inc_spec[0], stride=inc_spec[1], align=alignments[0]
        ),
        _make_fake_tensor_arg(
            m_c, shape=m_spec[0], stride=m_spec[1], align=alignments[1]
        ),
        _make_fake_tensor_arg(
            out_starts,
            shape=out_starts_spec[0],
            stride=out_starts_spec[1],
            align=alignments[2],
        ),
        _make_fake_tensor_arg(
            out_final,
            shape=out_final_spec[0],
            stride=out_final_spec[1],
            align=alignments[3],
        ),
        _make_fake_tensor_arg(
            init_arg,
            shape=out_final_spec[0],
            stride=out_final_spec[1],
            align=alignments[4],
        ),
        _compile_env_stream_placeholder(),
    )
    cache_key = _state_passing_key(
        device_index=(inc.device.index if inc.device.index is not None else -1),
        inc_shape=tuple(inc.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        initial_shape=(None if initial_states is None else tuple(initial_states.shape)),
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        alignments=alignments,
    )

    compiled = _STATE_PASSING_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_state_passing_host_wrapper(
            spec=(B, H, C, P, D),
            cfg=(
                num_threads,
                vecs_per_thread,
                copy_bits_in,
                copy_bits_out,
                copy_bits_state_in,
                copy_bits_state_out,
                has_init,
            ),
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _STATE_PASSING_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*keepalive)

    return compiled, out_starts, out_final, launch


def compile_state_passing_kernel(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Compile the forward state-passing kernel and allocate outputs."""
    compiled, out_starts, out_final, _launch = _compile_state_passing_kernel_impl(
        inc,
        m_chunk,
        initial_states=initial_states,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    return compiled, out_starts, out_final


def state_passing_cute(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None,
    num_threads: int = 128,
    vecs_per_thread: int = 8,
    return_final_state: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Thin public wrapper over the compiled forward state-passing kernel."""
    _compiled, out_starts, out_final, launch = _compile_state_passing_kernel_impl(
        inc,
        m_chunk,
        initial_states=initial_states,
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
    )
    launch()
    if not return_final_state:
        return out_starts
    return out_starts, out_final


def _compile_chunk_scan_kernel_impl(
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
) -> tuple[object, torch.Tensor, torch.Tensor, Callable[[], None]]:
    """Compile the end-to-end forward chunk-scan kernel, allocate outputs, and build a launcher."""
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

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D) or C.shape != (Bsz, H, T, D):
        raise ValueError("B/C must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if D % 2 != 0:
        raise ValueError("B/C last dim must be divisible by 2.")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    BH = Bsz * H
    BHC = BH * n_chunks
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
            L=L,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
            requested_m_block_size=m_block_size,
            requested_n_block_size=int(n_block_size),
            requested_num_threads=int(num_threads),
        )
    )

    if tuple(chunk_starts.shape) != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D) ={(Bsz, H, n_chunks, P, D)}."
        )

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    U_tc = _ensure_min_alignment(U_tc, min_align=16)
    B_tc = _ensure_min_alignment(B_tc, min_align=16)
    C_tc = _ensure_min_alignment(C_tc, min_align=16)
    U_scan = _guard_prev_time_base(U_tc, min_align=16)
    B_scan = _guard_prev_time_base(B_tc, min_align=16)

    if B_prev is None:
        U_prev0, B_prev0 = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=Bsz,
            heads=H,
            P=P,
            D=D,
        )
    else:
        if B_prev.shape != (Bsz, H, D) or U_prev.shape != (Bsz, H, P):
            raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
        B_prev0 = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev0 = U_prev.to(dtype=tc_dtype).contiguous()
        B_prev0 = _ensure_min_alignment(B_prev0, min_align=16)
        U_prev0 = _ensure_min_alignment(U_prev0, min_align=16)

    chunk_starts_c = chunk_starts.contiguous()
    chunk_starts_c = _ensure_min_alignment(chunk_starts_c, min_align=16)
    u_spec, b_spec, m_spec, k_spec, z0_spec, u_prev_spec, b_prev_spec, out_spec = (
        _chunk_scan_tensor_specs((Bsz, H, T_pad, P, D, n_chunks, L))
    )

    out_chunk = torch.empty((BHC, L, 1, P), device=U.device, dtype=output_dtype)
    out_pad = out_chunk.reshape(Bsz, H, n_chunks, L, 1, P).reshape(Bsz, H, T_pad, P)
    out_view = out_pad[:, :, :T, :]

    alignments = (
        _assumed_align(U_scan),
        _assumed_align(B_scan),
        _assumed_align(C_tc),
        _assumed_align(M_f),
        _assumed_align(K_f),
        _assumed_align(chunk_starts_c),
        _assumed_align(U_prev0),
        _assumed_align(B_prev0),
        _assumed_align(out_chunk),
    )
    runtime_tensors = (
        U_scan,
        B_scan,
        C_tc,
        M_f,
        K_f,
        chunk_starts_c,
        U_prev0,
        B_prev0,
        out_chunk,
    )
    runtime_args = (
        _make_runtime_tensor_view(U_scan, u_spec),
        _make_runtime_tensor_view(B_scan, b_spec),
        _make_runtime_tensor_view(C_tc, b_spec),
        _make_runtime_tensor_view(M_f, m_spec),
        _make_runtime_tensor_view(K_f, k_spec),
        _make_runtime_tensor_view(chunk_starts_c, z0_spec),
        _make_runtime_tensor_view(U_prev0, u_prev_spec),
        _make_runtime_tensor_view(B_prev0, b_prev_spec),
        _make_runtime_tensor_view(out_chunk, out_spec),
    )
    compile_args = (
        _make_fake_tensor_arg(
            U_scan, shape=u_spec[0], stride=u_spec[1], align=alignments[0]
        ),
        _make_fake_tensor_arg(
            B_scan, shape=b_spec[0], stride=b_spec[1], align=alignments[1]
        ),
        _make_fake_tensor_arg(
            C_tc, shape=b_spec[0], stride=b_spec[1], align=alignments[2]
        ),
        _make_fake_tensor_arg(
            M_f, shape=m_spec[0], stride=m_spec[1], align=alignments[3]
        ),
        _make_fake_tensor_arg(
            K_f, shape=k_spec[0], stride=k_spec[1], align=alignments[4]
        ),
        _make_fake_tensor_arg(
            chunk_starts_c, shape=z0_spec[0], stride=z0_spec[1], align=alignments[5]
        ),
        _make_fake_tensor_arg(
            U_prev0, shape=u_prev_spec[0], stride=u_prev_spec[1], align=alignments[6]
        ),
        _make_fake_tensor_arg(
            B_prev0, shape=b_prev_spec[0], stride=b_prev_spec[1], align=alignments[7]
        ),
        _make_fake_tensor_arg(
            out_chunk, shape=out_spec[0], stride=out_spec[1], align=alignments[8]
        ),
        _compile_env_stream_placeholder(),
    )
    keepalive = runtime_tensors

    cache_key = _chunk_scan_key(
        device_index=device_index,
        tc_dtype=tc_dtype,
        out_dtype=output_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        chunk_size=L,
        m_block_size=resolved_m_block_size,
        n_block_size=resolved_n_block_size,
        num_threads=resolved_num_threads,
        has_prev=B_prev is not None,
        alignments=alignments,
    )

    compiled = _CHUNK_SCAN_CACHE.get(cache_key)
    if compiled is None:
        in_cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
        out_cutlass_dtype = _torch_to_cutlass_dtype(output_dtype)
        kernel = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=L,
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
            spec=(Bsz, H, T_pad, P, D, n_chunks, L),
            cfg=(
                resolved_m_block_size,
                resolved_n_block_size,
                resolved_num_threads,
            ),
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_SCAN_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*keepalive)

    return compiled, out_chunk, out_view, launch


def compile_chunk_scan_kernel(
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
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Compile the end-to-end forward chunk-scan kernel and allocate outputs."""
    compiled, out_chunk, out_view, _launch = _compile_chunk_scan_kernel_impl(
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
    return compiled, out_chunk, out_view


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
    _compiled, _out_chunk, out_view, launch = _compile_chunk_scan_kernel_impl(
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
    launch()
    return out_view


def _build_forward_args(
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
) -> tuple[
    list[object],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor],
]:
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.dtype != B.dtype or U.dtype != C.dtype:
        raise ValueError("U/B/C must share dtype.")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B/C must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("output_dtype must be float16/bfloat16/float32.")

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D) or C.shape != (Bsz, H, T, D):
        raise ValueError("B/C must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if D % 2 != 0:
        raise ValueError("B/C last dim must be divisible by 2.")
    if initial_states is not None:
        if not torch.is_floating_point(initial_states):
            raise TypeError("initial_states must be floating-point.")
        if tuple(initial_states.shape) != (Bsz, H, P, D):
            raise ValueError("initial_states must be (B,H,P,D) matching scan inputs.")
        if initial_states.device != U.device:
            raise ValueError("initial_states must be on the same device as U.")
    if B_prev is not None:
        if not torch.is_floating_point(B_prev) or not torch.is_floating_point(U_prev):
            raise TypeError("B_prev and U_prev must be floating-point.")
        if tuple(B_prev.shape) != (Bsz, H, D) or tuple(U_prev.shape) != (Bsz, H, P):
            raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
        if B_prev.device != U.device or U_prev.device != U.device:
            raise ValueError("B_prev and U_prev must be on the same device as U.")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L

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
            L=L,
            tc_dtype=tc_dtype,
            output_dtype=output_dtype,
            device_index=device_index,
            requested_m_block_size=m_block_size,
            requested_n_block_size=int(n_block_size),
            requested_num_threads=int(scan_num_threads),
        )
    )

    if prepared_inputs is None:
        U_tc = _prepare_time_operand(U, T_pad=T_pad, dtype=tc_dtype)
        M_f = _prepare_m_operand(M, T_pad=T_pad)
        K_f = _prepare_time_operand(K, T_pad=T_pad, dtype=torch.float32)
        B_tc = _prepare_time_operand(B, T_pad=T_pad, dtype=tc_dtype)
        C_tc = _prepare_time_operand(C, T_pad=T_pad, dtype=tc_dtype)
    else:
        U_tc, M_f, K_f, B_tc, C_tc = prepared_inputs
        expected_u_shape = (Bsz, H, T_pad, P)
        expected_b_shape = (Bsz, H, T_pad, D)
        expected_m_shape = (Bsz, H, T_pad, 2)
        expected_k_shape = (Bsz, H, T_pad, 2, 2)
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
        U_prev0, B_prev0 = _get_zero_prev_tensors(
            device=U.device,
            dtype=tc_dtype,
            batch_size=Bsz,
            heads=H,
            P=P,
            D=D,
        )
    else:
        U_prev0 = U_prev.to(dtype=tc_dtype).contiguous()
        B_prev0 = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev0 = _ensure_min_alignment(U_prev0, min_align=16)
        B_prev0 = _ensure_min_alignment(B_prev0, min_align=16)

    inc_chunk = _get_fwd_workspace(
        device=U.device,
        batch_size=Bsz,
        heads=H,
        n_chunks=n_chunks,
        P=P,
        D=D,
    )
    inc_chunk.zero_()
    inc = inc_chunk.reshape(Bsz, H, n_chunks, P, D)
    if initial_states is None:
        initial_state0 = _get_zero_initial_state(
            device=U.device,
            batch_size=Bsz,
            heads=H,
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
            (Bsz, H, P, D), device=U.device, dtype=torch.float32
        )
        final_state_workspace.zero_()
        final_state = final_state_workspace
    else:
        final_state = _get_dummy_final_state(
            device=U.device,
            batch_size=Bsz,
            heads=H,
            P=P,
            D=D,
        )
    if return_intermediates:
        m_chunk_tile = torch.empty(
            (Bsz * H * n_chunks, 2), device=U.device, dtype=torch.float32
        )
        chunk_starts = torch.empty(
            (Bsz, H, n_chunks, P, D), device=U.device, dtype=torch.float32
        )
    else:
        m_chunk_tile, chunk_starts = _get_no_grad_intermediates(
            device=U.device,
            batch_size=Bsz,
            heads=H,
            n_chunks=n_chunks,
            P=P,
            D=D,
        )
    chunk_starts = _ensure_min_alignment(chunk_starts, min_align=16)
    m_chunk_tile.zero_()
    m_chunk = m_chunk_tile.reshape(Bsz, H, n_chunks, 2)
    out_chunk = torch.empty(
        (Bsz * H * n_chunks, L, 1, P), device=U.device, dtype=output_dtype
    )
    out_pad = out_chunk.reshape(Bsz, H, n_chunks, L, 1, P).reshape(Bsz, H, T_pad, P)

    state_elems_per_thread = 2 * int(state_vecs_per_thread)
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

    alignments = tuple(
        _assumed_align(tensor)
        for tensor in (
            U_tc,
            B_tc,
            C_tc,
            M_f,
            K_f,
            U_prev0,
            B_prev0,
            inc_chunk,
            m_chunk_tile,
            chunk_starts,
            final_state,
            initial_state0,
            out_chunk,
            out_pad,
        )
    )
    dynamic_args: tuple[object, ...] = ()
    spec = (
        Bsz,
        H,
        T_pad,
        P,
        D,
        n_chunks,
        L,
    )
    cfg = (
        resolved_m_block,
        resolved_n_block,
        resolved_scan_num_threads,
        int(state_num_threads),
        int(state_vecs_per_thread),
        int(state_copy_bits_in),
        int(state_copy_bits_out),
        int(state_copy_bits_state_in),
        int(state_copy_bits_state_out),
        has_init,
    )
    return (
        dynamic_args,
        alignments,
        spec,
        cfg,
        (
            out_pad[:, :, :T, :],
            final_state_workspace,
            m_chunk,
            chunk_starts,
        ),
        (
            U_tc,
            B_tc,
            C_tc,
            M_f,
            K_f,
            U_prev0,
            B_prev0,
            inc_chunk,
            inc,
            m_chunk_tile,
            m_chunk,
            chunk_starts,
            final_state,
            initial_state0,
            out_chunk,
            out_pad,
        ),
    )


def _make_prepared_chunk_increment_launch(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    U_prev0: torch.Tensor,
    B_prev0: torch.Tensor,
    inc_chunk: torch.Tensor,
    m_chunk_tile: torch.Tensor,
    chunk_size: int,
    compute_dtype: torch.dtype | None,
    has_prev: bool,
) -> Callable[[], None]:
    (
        problem_spec,
        cta_tiler,
        compile_args,
        runtime_args,
        cache_key,
    ) = _make_chunk_increment_launch_artifacts(
        U=U,
        M=M,
        K=K,
        B=B,
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        inc_chunk=inc_chunk,
        m_chunk_tile=m_chunk_tile,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        has_prev=has_prev,
    )
    compiled = _CHUNK_INCREMENT_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_chunk_increment_host_wrapper(
            spec=problem_spec,
            cta_tiler=cta_tiler,
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_INCREMENT_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(
            U_tc, B_tc, M_f, K_f, U_prev0, B_prev0, inc_chunk, m_chunk_tile
        )

    return launch


def _make_prepared_state_passing_launch(
    *,
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    chunk_starts: torch.Tensor,
    final_state: torch.Tensor,
    initial_state0: torch.Tensor,
    has_init: bool,
    num_threads: int,
    vecs_per_thread: int,
) -> Callable[[], None]:
    B, H, C, P, D = map(int, inc.shape)
    init_arg = initial_state0 if has_init else inc
    inc_spec, m_spec, out_starts_spec, out_final_spec = _state_passing_tensor_specs(
        (B, H, C, P, D)
    )
    S = P * D
    elems_per_thread = 2 * vecs_per_thread
    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc,
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        chunk_starts,
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state_in = 32
    copy_bits_state_out = 32
    alignments = (
        _assumed_align(inc),
        _assumed_align(m_chunk),
        _assumed_align(chunk_starts),
        _assumed_align(final_state),
        _assumed_align(init_arg),
    )
    runtime_tensors = (inc, m_chunk, chunk_starts, final_state, init_arg)
    runtime_args = (
        _make_runtime_tensor_view(inc, inc_spec),
        _make_runtime_tensor_view(m_chunk, m_spec),
        _make_runtime_tensor_view(chunk_starts, out_starts_spec),
        _make_runtime_tensor_view(final_state, out_final_spec),
        _make_runtime_tensor_view(init_arg, out_final_spec),
    )
    compile_args = (
        _make_fake_tensor_arg(
            inc, shape=inc_spec[0], stride=inc_spec[1], align=alignments[0]
        ),
        _make_fake_tensor_arg(
            m_chunk, shape=m_spec[0], stride=m_spec[1], align=alignments[1]
        ),
        _make_fake_tensor_arg(
            chunk_starts,
            shape=out_starts_spec[0],
            stride=out_starts_spec[1],
            align=alignments[2],
        ),
        _make_fake_tensor_arg(
            final_state,
            shape=out_final_spec[0],
            stride=out_final_spec[1],
            align=alignments[3],
        ),
        _make_fake_tensor_arg(
            init_arg,
            shape=out_final_spec[0],
            stride=out_final_spec[1],
            align=alignments[4],
        ),
        _compile_env_stream_placeholder(),
    )
    cache_key = _state_passing_key(
        device_index=(inc.device.index if inc.device.index is not None else -1),
        inc_shape=tuple(inc.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        initial_shape=(tuple(initial_state0.shape) if has_init else None),
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        copy_bits_state_in=copy_bits_state_in,
        copy_bits_state_out=copy_bits_state_out,
        alignments=alignments,
    )
    compiled = _STATE_PASSING_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_state_passing_host_wrapper(
            spec=(B, H, C, P, D),
            cfg=(
                num_threads,
                vecs_per_thread,
                copy_bits_in,
                copy_bits_out,
                copy_bits_state_in,
                copy_bits_state_out,
                has_init,
            ),
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _STATE_PASSING_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*runtime_tensors)

    return launch


def _make_prepared_chunk_scan_launch(
    *,
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    C_tc: torch.Tensor,
    M_f: torch.Tensor,
    K_f: torch.Tensor,
    chunk_starts: torch.Tensor,
    U_prev0: torch.Tensor,
    B_prev0: torch.Tensor,
    out_chunk: torch.Tensor,
    chunk_size: int,
    m_block_size: int,
    n_block_size: int,
    num_threads: int,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype,
    has_prev: bool,
) -> Callable[[], None]:
    Bsz, H, _T, P = map(int, U.shape)
    D = int(B.shape[-1])
    L = int(chunk_size)
    T_pad = int(U_tc.shape[2])
    n_chunks = T_pad // L
    u_spec, b_spec, m_spec, k_spec, z0_spec, u_prev_spec, b_prev_spec, out_spec = (
        _chunk_scan_tensor_specs((Bsz, H, T_pad, P, D, n_chunks, L))
    )
    U_scan = _guard_prev_time_base(U_tc, min_align=16)
    B_scan = _guard_prev_time_base(B_tc, min_align=16)
    alignments = (
        _assumed_align(U_scan),
        _assumed_align(B_scan),
        _assumed_align(C_tc),
        _assumed_align(M_f),
        _assumed_align(K_f),
        _assumed_align(chunk_starts),
        _assumed_align(U_prev0),
        _assumed_align(B_prev0),
        _assumed_align(out_chunk),
    )
    runtime_tensors = (
        U_scan,
        B_scan,
        C_tc,
        M_f,
        K_f,
        chunk_starts,
        U_prev0,
        B_prev0,
        out_chunk,
    )
    runtime_args = (
        _make_runtime_tensor_view(U_scan, u_spec),
        _make_runtime_tensor_view(B_scan, b_spec),
        _make_runtime_tensor_view(C_tc, b_spec),
        _make_runtime_tensor_view(M_f, m_spec),
        _make_runtime_tensor_view(K_f, k_spec),
        _make_runtime_tensor_view(chunk_starts, z0_spec),
        _make_runtime_tensor_view(U_prev0, u_prev_spec),
        _make_runtime_tensor_view(B_prev0, b_prev_spec),
        _make_runtime_tensor_view(out_chunk, out_spec),
    )
    compile_args = (
        _make_fake_tensor_arg(
            U_scan, shape=u_spec[0], stride=u_spec[1], align=alignments[0]
        ),
        _make_fake_tensor_arg(
            B_scan, shape=b_spec[0], stride=b_spec[1], align=alignments[1]
        ),
        _make_fake_tensor_arg(
            C_tc, shape=b_spec[0], stride=b_spec[1], align=alignments[2]
        ),
        _make_fake_tensor_arg(
            M_f, shape=m_spec[0], stride=m_spec[1], align=alignments[3]
        ),
        _make_fake_tensor_arg(
            K_f, shape=k_spec[0], stride=k_spec[1], align=alignments[4]
        ),
        _make_fake_tensor_arg(
            chunk_starts, shape=z0_spec[0], stride=z0_spec[1], align=alignments[5]
        ),
        _make_fake_tensor_arg(
            U_prev0, shape=u_prev_spec[0], stride=u_prev_spec[1], align=alignments[6]
        ),
        _make_fake_tensor_arg(
            B_prev0, shape=b_prev_spec[0], stride=b_prev_spec[1], align=alignments[7]
        ),
        _make_fake_tensor_arg(
            out_chunk, shape=out_spec[0], stride=out_spec[1], align=alignments[8]
        ),
        _compile_env_stream_placeholder(),
    )
    runtime_args = (
        _make_runtime_tensor_view(U_scan, u_spec),
        _make_runtime_tensor_view(B_scan, b_spec),
        _make_runtime_tensor_view(C_tc, b_spec),
        _make_runtime_tensor_view(M_f, m_spec),
        _make_runtime_tensor_view(K_f, k_spec),
        _make_runtime_tensor_view(chunk_starts, z0_spec),
        _make_runtime_tensor_view(U_prev0, u_prev_spec),
        _make_runtime_tensor_view(B_prev0, b_prev_spec),
        _make_runtime_tensor_view(out_chunk, out_spec),
    )
    cache_key = _chunk_scan_key(
        device_index=(
            U.device.index
            if U.device.index is not None
            else torch.cuda.current_device()
        ),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        out_dtype=output_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        chunk_size=L,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
        has_prev=has_prev,
        alignments=alignments,
    )
    compiled = _CHUNK_SCAN_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_chunk_scan_host_wrapper(
            spec=(Bsz, H, T_pad, P, D, n_chunks, L),
            cfg=(m_block_size, n_block_size, num_threads),
        )
        compiled = cute.compile(host_wrapper, *compile_args, options="--enable-tvm-ffi")
        _CHUNK_SCAN_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*runtime_args)
        _record_tensors_on_current_stream(*runtime_tensors)

    return launch


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
    initial_states: torch.Tensor | None = None,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
) -> object:
    _dynamic_args, _alignments, _spec, cfg, _outputs, keepalive = _build_forward_args(
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
    )
    (
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        U_prev0,
        B_prev0,
        inc_chunk,
        inc,
        m_chunk_tile,
        m_chunk,
        chunk_starts,
        final_state,
        initial_state0,
        out_chunk,
        out_pad,
    ) = keepalive
    launch_inc = _make_prepared_chunk_increment_launch(
        U=U,
        M=M,
        K=K,
        B=B,
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        inc_chunk=inc_chunk,
        m_chunk_tile=m_chunk_tile,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        has_prev=B_prev is not None,
    )
    launch_state = _make_prepared_state_passing_launch(
        inc=inc,
        m_chunk=m_chunk,
        chunk_starts=chunk_starts,
        final_state=final_state,
        initial_state0=initial_state0,
        has_init=bool(cfg[9]),
        num_threads=int(cfg[3]),
        vecs_per_thread=int(cfg[4]),
    )
    launch_scan = _make_prepared_chunk_scan_launch(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        U_tc=U_tc,
        B_tc=B_tc,
        C_tc=C_tc,
        M_f=M_f,
        K_f=K_f,
        chunk_starts=chunk_starts,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        out_chunk=out_chunk,
        chunk_size=chunk_size,
        m_block_size=int(cfg[0]),
        n_block_size=int(cfg[1]),
        num_threads=int(cfg[2]),
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        has_prev=B_prev is not None,
    )

    def launch() -> None:
        launch_inc()
        launch_state()
        launch_scan()
        _record_tensors_on_current_stream(*keepalive)

    return launch


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
    _dynamic_args, _alignments, _spec, cfg, outputs, keepalive = _build_forward_args(
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
    )
    (
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        U_prev0,
        B_prev0,
        inc_chunk,
        inc,
        m_chunk_tile,
        m_chunk,
        chunk_starts,
        final_state,
        initial_state0,
        out_chunk,
        out_pad,
    ) = keepalive
    launch_inc = _make_prepared_chunk_increment_launch(
        U=U,
        M=M,
        K=K,
        B=B,
        U_tc=U_tc,
        B_tc=B_tc,
        M_f=M_f,
        K_f=K_f,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        inc_chunk=inc_chunk,
        m_chunk_tile=m_chunk_tile,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
        has_prev=B_prev is not None,
    )
    launch_state = _make_prepared_state_passing_launch(
        inc=inc,
        m_chunk=m_chunk,
        chunk_starts=chunk_starts,
        final_state=final_state,
        initial_state0=initial_state0,
        has_init=bool(cfg[9]),
        num_threads=int(cfg[3]),
        vecs_per_thread=int(cfg[4]),
    )
    launch_scan = _make_prepared_chunk_scan_launch(
        U=U,
        M=M,
        K=K,
        B=B,
        C=C,
        U_tc=U_tc,
        B_tc=B_tc,
        C_tc=C_tc,
        M_f=M_f,
        K_f=K_f,
        chunk_starts=chunk_starts,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        out_chunk=out_chunk,
        chunk_size=chunk_size,
        m_block_size=int(cfg[0]),
        n_block_size=int(cfg[1]),
        num_threads=int(cfg[2]),
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        has_prev=B_prev is not None,
    )
    launch_inc()
    launch_state()
    state_chunk_starts = chunk_starts
    state_final_state = final_state
    launch_scan()
    _record_tensors_on_current_stream(*keepalive)
    Y, final_state, m_chunk, chunk_starts = outputs
    if not return_final_state:
        return Y, m_chunk, chunk_starts
    return Y, state_final_state, m_chunk, state_chunk_starts


__all__ = [
    "chunk_increment_cute",
    "chunk_scan_cute",
    "compile_chunk_increment_kernel",
    "compile_chunk_scan_kernel",
    "compile_state_passing_kernel",
    "compile_v2x2ssd_fwd_cute",
    "state_passing_cute",
    "v2x2ssd_fwd_cute",
]
