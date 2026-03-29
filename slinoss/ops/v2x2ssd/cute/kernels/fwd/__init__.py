"""CuTe forward kernels for the ``v2x2ssd`` staged pipeline."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from typing import Callable

from slinoss.perf import note_cache_event

from .chunk_increment import ChunkIncrementFwdAmpere
from .chunk_scan import ChunkScanFwdAmpere
from .common import (
    _choose_copy_bits_for_linear_tiles,
    _make_ptr_args,
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


def _cache_set(cache: dict, key: tuple, value, *, limit: int) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= int(limit):
        cache.pop(next(iter(cache)), None)
    cache[key] = value


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
        (batch_size, heads, n_chunks, P, D), device=device, dtype=torch.float32
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
            (batch_size, heads, n_chunks, 2), device=device, dtype=torch.float32
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


def _make_tensor_from_spec(
    ptr: cute.Pointer,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
):
    shape, stride = spec
    return cute.make_tensor(ptr, cute.make_layout(shape, stride=stride))


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
        return tensor
    return _pad_zero_time(tensor, T_pad=T_pad, dtype=dtype)


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
    copy_bits_state: int,
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
        int(copy_bits_state),
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
    BH = Bsz * H
    BHC = BH * n_chunks

    u_spec = _make_tensor_spec((BHC, L, 1, P))
    b_spec = _make_tensor_spec((BHC, L, 1, D))
    m_spec = _make_tensor_spec((BHC, L, 2))
    k_spec = _make_tensor_spec((BHC, L, 2, 2))
    z0_spec = _make_tensor_spec((BHC, P, 1, D))
    u_prev_spec = _make_tensor_spec((BH, P))
    b_prev_spec = _make_tensor_spec((BH, D))
    out_spec = _make_tensor_spec((BHC, L, 1, P))

    @cute.jit
    def _chunk_scan_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        Z0_ptr: cute.Pointer,
        U_prev0_ptr: cute.Pointer,
        B_prev0_ptr: cute.Pointer,
        Out_ptr: cute.Pointer,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mZ0 = _make_tensor_from_spec(Z0_ptr, z0_spec)
        mU_prev0 = _make_tensor_from_spec(U_prev0_ptr, u_prev_spec)
        mB_prev0 = _make_tensor_from_spec(B_prev0_ptr, b_prev_spec)
        mOut = _make_tensor_from_spec(Out_ptr, out_spec)

        chunk_scan = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=L,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
        )
        chunk_scan(mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut)

    return _chunk_scan_host_wrapper


def _make_chunk_increment_host_wrapper(
    *,
    spec: tuple[int, ...],
    cta_tiler: tuple[int, int, int],
):
    Bsz, H, T_pad, P, D, n_chunks, L = spec
    BH = Bsz * H
    BHC = BH * n_chunks

    u_spec = _make_tensor_spec((P, T_pad, BH), stride=(1, P, T_pad * P))
    b_spec = _make_tensor_spec((D, T_pad, BH), stride=(1, D, T_pad * D))
    m_spec = _make_tensor_spec((2, T_pad, BH), stride=(1, 2, T_pad * 2))
    k_spec = _make_tensor_spec((2, T_pad, BH), stride=(1, 4, T_pad * 4))
    u_prev_spec = _make_tensor_spec((P, BH), stride=(1, P))
    b_prev_spec = _make_tensor_spec((D, BH), stride=(1, D))
    inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    m_chunk_spec = _make_tensor_spec((2, BHC), stride=(1, 2))

    @cute.jit
    def _chunk_increment_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        Kprev_ptr: cute.Pointer,
        Kcurr_ptr: cute.Pointer,
        U_prev0_ptr: cute.Pointer,
        B_prev0_ptr: cute.Pointer,
        Inc_ptr: cute.Pointer,
        Mchunk_ptr: cute.Pointer,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, k_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, k_spec)
        mU_prev0 = _make_tensor_from_spec(U_prev0_ptr, u_prev_spec)
        mB_prev0 = _make_tensor_from_spec(B_prev0_ptr, b_prev_spec)
        mInc = _make_tensor_from_spec(Inc_ptr, inc_spec)
        mMchunk = _make_tensor_from_spec(Mchunk_ptr, m_chunk_spec)

        chunk_increment = ChunkIncrementFwdAmpere(
            U_ptr.value_type,
            chunk_size=L,
            cta_tiler=cta_tiler,
        )
        chunk_increment(mU, mB, mM, mKprev, mKcurr, mU_prev0, mB_prev0, mInc, mMchunk)

    return _chunk_increment_host_wrapper


def _make_state_passing_host_wrapper(*, spec: tuple[int, ...], cfg: tuple[int, ...]):
    B, H, C, P, D = spec
    (
        num_threads,
        vecs_per_thread,
        copy_bits_in,
        copy_bits_out,
        copy_bits_state,
        has_init,
    ) = cfg

    inc_spec = _make_tensor_spec((B, H, C, P, D))
    m_spec = _make_tensor_spec((B, H, C, 2))
    out_starts_spec = _make_tensor_spec((B, H, C, P, D))
    out_final_spec = _make_tensor_spec((B, H, P, D))

    @cute.jit
    def _state_passing_host_wrapper(
        Inc_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        OutStarts_ptr: cute.Pointer,
        OutFinal_ptr: cute.Pointer,
        Init_ptr: cute.Pointer,
    ):
        mInc = _make_tensor_from_spec(Inc_ptr, inc_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mOutStarts = _make_tensor_from_spec(OutStarts_ptr, out_starts_spec)
        mOutFinal = _make_tensor_from_spec(OutFinal_ptr, out_final_spec)
        mInit = _make_tensor_from_spec(Init_ptr, out_final_spec)

        state_passing = StatePassingFwdAmpere(
            num_threads=num_threads,
            vecs_per_thread=vecs_per_thread,
            copy_bits_in=copy_bits_in,
            copy_bits_out=copy_bits_out,
            copy_bits_state=copy_bits_state,
            has_init=has_init,
        )
        state_passing(mInc, mM, mOutStarts, mOutFinal, mInit)

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

    BH = Bsz * H
    BHC = BH * n_chunks
    cta_tiler = _resolve_chunk_increment_cta_tiler(D=D)

    inc_chunk = torch.zeros((BHC, P, D), device=U.device, dtype=torch.float32)
    m_chunk_chunk = torch.zeros((BHC, 2), device=U.device, dtype=torch.float32)

    Kprev_view = K_f[:, :, :, 0, :]
    Kcurr_view = K_f[:, :, :, 1, :]

    dynamic_args, alignments = _make_ptr_args(
        U_tc,
        B_tc,
        M_f,
        Kprev_view,
        Kcurr_view,
        U_prev,
        B_prev,
        inc_chunk,
        m_chunk_chunk,
    )
    cache_key = _chunk_increment_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        chunk_size=L,
        has_prev=U_prev0 is not None,
        cta_tiler=cta_tiler,
        alignments=alignments,
    )

    compiled = _CHUNK_INCREMENT_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_chunk_increment_host_wrapper(
            spec=(Bsz, H, T_pad, P, D, n_chunks, L),
            cta_tiler=cta_tiler,
        )
        compiled = cute.compile(host_wrapper, *dynamic_args)
        _CHUNK_INCREMENT_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*dynamic_args)

    return compiled, inc_chunk, m_chunk_chunk, launch


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
    compiled, inc_chunk, m_chunk_chunk, _launch = _compile_chunk_increment_kernel_impl(
        U,
        M,
        K,
        B,
        U_prev0=U_prev0,
        B_prev0=B_prev0,
        chunk_size=chunk_size,
        compute_dtype=compute_dtype,
    )
    return compiled, inc_chunk, m_chunk_chunk


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
    _compiled, inc_chunk, m_chunk_chunk, launch = _compile_chunk_increment_kernel_impl(
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
    BH = int(U.shape[0]) * int(U.shape[1])
    n_chunks = inc_chunk.shape[0] // BH
    return (
        inc_chunk.reshape(U.shape[0], U.shape[1], n_chunks, U.shape[-1], B.shape[-1]),
        m_chunk_chunk.reshape(U.shape[0], U.shape[1], n_chunks, 2),
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

    S = P * D
    elems_per_thread = 2 * int(vecs_per_thread)
    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc.contiguous(),
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        out_starts,
        tile_stride_elems=S,
        elems_per_thread=elems_per_thread,
    )
    copy_bits_state = (
        _choose_copy_bits_for_linear_tiles(
            initial_states.contiguous(),
            tile_stride_elems=S,
            elems_per_thread=elems_per_thread,
        )
        if initial_states is not None
        else 32
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

    dynamic_args, alignments = _make_ptr_args(
        inc_c,
        m_c,
        out_starts,
        out_final,
        init_arg,
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
        copy_bits_state=copy_bits_state,
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
                copy_bits_state,
                has_init,
            ),
        )
        compiled = cute.compile(host_wrapper, *dynamic_args)
        _STATE_PASSING_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*dynamic_args)

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

    chunk_starts_c = chunk_starts.contiguous()

    out_chunk = torch.empty((BHC, L, 1, P), device=U.device, dtype=output_dtype)
    out_pad = out_chunk.reshape(Bsz, H, n_chunks, L, 1, P).reshape(Bsz, H, T_pad, P)
    out_view = out_pad[:, :, :T, :]

    dynamic_args, alignments = _make_ptr_args(
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        chunk_starts_c,
        U_prev0,
        B_prev0,
        out_chunk,
    )

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
        compiled = cute.compile(host_wrapper, *dynamic_args)
        _CHUNK_SCAN_CACHE[cache_key] = compiled

    def launch() -> None:
        compiled(*dynamic_args)

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


def _fwd_host_cache_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    out_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    chunk_size: int,
    m_block_size: int,
    n_block_size: int,
    scan_num_threads: int,
    state_num_threads: int,
    state_vecs_per_thread: int,
    state_copy_bits_in: int,
    state_copy_bits_out: int,
    state_copy_bits_state: int,
    has_init: bool,
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "v2x2ssd_fwd_host",
        device_index,
        tc_dtype,
        out_dtype,
        U_shape,
        M_shape,
        K_shape,
        B_shape,
        C_shape,
        int(chunk_size),
        int(m_block_size),
        int(n_block_size),
        int(scan_num_threads),
        int(state_num_threads),
        int(state_vecs_per_thread),
        int(state_copy_bits_in),
        int(state_copy_bits_out),
        int(state_copy_bits_state),
        bool(has_init),
        alignments,
    )


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

    inc = _get_fwd_workspace(
        device=U.device,
        batch_size=Bsz,
        heads=H,
        n_chunks=n_chunks,
        P=P,
        D=D,
    )
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
        m_chunk = torch.empty(
            (Bsz, H, n_chunks, 2), device=U.device, dtype=torch.float32
        )
        chunk_starts = torch.empty(
            (Bsz, H, n_chunks, P, D), device=U.device, dtype=torch.float32
        )
    else:
        m_chunk, chunk_starts = _get_no_grad_intermediates(
            device=U.device,
            batch_size=Bsz,
            heads=H,
            n_chunks=n_chunks,
            P=P,
            D=D,
        )
    out_pad = torch.empty((Bsz, H, T_pad, P), device=U.device, dtype=output_dtype)

    state_copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc,
        tile_stride_elems=P * D,
        elems_per_thread=2 * int(state_vecs_per_thread),
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles(
        chunk_starts,
        tile_stride_elems=P * D,
        elems_per_thread=2 * int(state_vecs_per_thread),
    )
    state_copy_bits_state = _choose_copy_bits_for_linear_tiles(
        final_state,
        tile_stride_elems=P * D,
        elems_per_thread=2 * int(state_vecs_per_thread),
    )

    dynamic_args, alignments = _make_ptr_args(
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        K_f[:, :, :, 0, :],
        K_f[:, :, :, 1, :],
        U_prev0,
        B_prev0,
        inc,
        m_chunk,
        chunk_starts,
        final_state,
        initial_state0,
        out_pad,
    )
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
        int(state_copy_bits_state),
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
    )


def _make_v2x2ssd_fwd_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    Bsz, H, T_pad, P, D, n_chunks, L = spec
    chunk_increment_cta_tiler = _resolve_chunk_increment_cta_tiler(D=D)
    (
        m_block_size,
        n_block_size,
        scan_num_threads,
        state_num_threads,
        state_vecs_per_thread,
        state_copy_bits_in,
        state_copy_bits_out,
        state_copy_bits_state,
        has_init,
    ) = cfg
    BH = Bsz * H
    BHC = BH * n_chunks

    u_inc_spec = _make_tensor_spec((P, T_pad, BH), stride=(1, P, T_pad * P))
    b_inc_spec = _make_tensor_spec((D, T_pad, BH), stride=(1, D, T_pad * D))
    m_inc_spec = _make_tensor_spec((2, T_pad, BH), stride=(1, 2, T_pad * 2))
    k_inc_spec = _make_tensor_spec((2, T_pad, BH), stride=(1, 4, T_pad * 4))
    u_prev_inc_spec = _make_tensor_spec((P, BH), stride=(1, P))
    b_prev_inc_spec = _make_tensor_spec((D, BH), stride=(1, D))
    inc_out_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    m_chunk_out_spec = _make_tensor_spec((2, BHC), stride=(1, 2))

    inc_state_spec = _make_tensor_spec((Bsz, H, n_chunks, P, D))
    m_chunk_state_spec = _make_tensor_spec((Bsz, H, n_chunks, 2))
    chunk_starts_spec = _make_tensor_spec((Bsz, H, n_chunks, P, D))
    final_state_spec = _make_tensor_spec((Bsz, H, P, D))

    u_scan_spec = _make_tensor_spec((BHC, L, 1, P))
    b_scan_spec = _make_tensor_spec((BHC, L, 1, D))
    c_scan_spec = _make_tensor_spec((BHC, L, 1, D))
    m_scan_spec = _make_tensor_spec((BHC, L, 2))
    k_scan_spec = _make_tensor_spec((BHC, L, 2, 2))
    z0_scan_spec = _make_tensor_spec((BHC, P, 1, D))
    u_prev_scan_spec = _make_tensor_spec((BH, P))
    b_prev_scan_spec = _make_tensor_spec((BH, D))
    out_scan_spec = _make_tensor_spec((BHC, L, 1, P))

    @cute.jit
    def _v2x2ssd_fwd_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        Kprev_ptr: cute.Pointer,
        Kcurr_ptr: cute.Pointer,
        U_prev_ptr: cute.Pointer,
        B_prev_ptr: cute.Pointer,
        inc_ptr: cute.Pointer,
        m_chunk_ptr: cute.Pointer,
        chunk_starts_ptr: cute.Pointer,
        final_state_ptr: cute.Pointer,
        initial_state_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
    ):
        mU_inc = _make_tensor_from_spec(U_ptr, u_inc_spec)
        mB_inc = _make_tensor_from_spec(B_ptr, b_inc_spec)
        mM_inc = _make_tensor_from_spec(M_ptr, m_inc_spec)
        mKprev_inc = _make_tensor_from_spec(Kprev_ptr, k_inc_spec)
        mKcurr_inc = _make_tensor_from_spec(Kcurr_ptr, k_inc_spec)
        mU_prev_inc = _make_tensor_from_spec(U_prev_ptr, u_prev_inc_spec)
        mB_prev_inc = _make_tensor_from_spec(B_prev_ptr, b_prev_inc_spec)
        mInc = _make_tensor_from_spec(inc_ptr, inc_out_spec)
        mMchunk = _make_tensor_from_spec(m_chunk_ptr, m_chunk_out_spec)

        inc_state_t = _make_tensor_from_spec(inc_ptr, inc_state_spec)
        m_chunk_state_t = _make_tensor_from_spec(m_chunk_ptr, m_chunk_state_spec)
        chunk_starts_state_t = _make_tensor_from_spec(
            chunk_starts_ptr, chunk_starts_spec
        )
        final_state_t = _make_tensor_from_spec(final_state_ptr, final_state_spec)
        initial_state_t = _make_tensor_from_spec(initial_state_ptr, final_state_spec)

        mU_scan = _make_tensor_from_spec(U_ptr, u_scan_spec)
        mB_scan = _make_tensor_from_spec(B_ptr, b_scan_spec)
        mC_scan = _make_tensor_from_spec(C_ptr, c_scan_spec)
        mM_scan = _make_tensor_from_spec(M_ptr, m_scan_spec)
        mK_scan = _make_tensor_from_spec(K_ptr, k_scan_spec)
        mZ0_scan = _make_tensor_from_spec(chunk_starts_ptr, z0_scan_spec)
        mU_prev_scan_t = _make_tensor_from_spec(U_prev_ptr, u_prev_scan_spec)
        mB_prev_scan_t = _make_tensor_from_spec(B_prev_ptr, b_prev_scan_spec)
        mOut = _make_tensor_from_spec(out_ptr, out_scan_spec)

        tc_dtype = U_ptr.value_type

        chunk_increment = ChunkIncrementFwdAmpere(
            tc_dtype,
            chunk_size=L,
            cta_tiler=chunk_increment_cta_tiler,
        )
        chunk_increment(
            mU_inc,
            mB_inc,
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mU_prev_inc,
            mB_prev_inc,
            mInc,
            mMchunk,
        )

        state_passing = StatePassingFwdAmpere(
            num_threads=state_num_threads,
            vecs_per_thread=state_vecs_per_thread,
            copy_bits_in=state_copy_bits_in,
            copy_bits_out=state_copy_bits_out,
            copy_bits_state=state_copy_bits_state,
            has_init=has_init,
        )
        # Keep the stateless dummy init separate from the final-state output.
        # Aliasing these makes the fused path effectively stateful across calls.
        state_passing(
            inc_state_t,
            m_chunk_state_t,
            chunk_starts_state_t,
            final_state_t,
            initial_state_t,
        )

        chunk_scan = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=L,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=scan_num_threads,
        )
        if cutlass.const_expr(not chunk_scan._tile_family_supported()):
            raise ValueError("chunk_scan tile family is inconsistent.")
        chunk_scan(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mZ0_scan,
            mU_prev_scan_t,
            mB_prev_scan_t,
            mOut,
        )

    return _v2x2ssd_fwd_host_wrapper


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
    dynamic_args, alignments, spec, cfg, _outputs = _build_forward_args(
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
    cache_key = _fwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        out_dtype=output_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        chunk_size=int(chunk_size),
        m_block_size=int(cfg[0]),
        n_block_size=int(cfg[1]),
        scan_num_threads=int(cfg[2]),
        state_num_threads=int(state_num_threads),
        state_vecs_per_thread=int(state_vecs_per_thread),
        state_copy_bits_in=int(cfg[5]),
        state_copy_bits_out=int(cfg[6]),
        state_copy_bits_state=int(cfg[7]),
        has_init=bool(cfg[8]),
        alignments=alignments,
    )
    cached = _FWD_HOST_CACHE.get(cache_key)
    if cached is not None:
        note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=True)
        return cached

    note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=False)
    host_wrapper = _make_v2x2ssd_fwd_host_wrapper(spec=spec, cfg=cfg)
    compiled = cute.compile(host_wrapper, *dynamic_args)
    _FWD_HOST_CACHE[cache_key] = compiled
    return compiled


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
    dynamic_args, alignments, spec, cfg, outputs = _build_forward_args(
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
    cache_key = _fwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        out_dtype=output_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        chunk_size=int(chunk_size),
        m_block_size=int(cfg[0]),
        n_block_size=int(cfg[1]),
        scan_num_threads=int(cfg[2]),
        state_num_threads=int(state_num_threads),
        state_vecs_per_thread=int(state_vecs_per_thread),
        state_copy_bits_in=int(cfg[5]),
        state_copy_bits_out=int(cfg[6]),
        state_copy_bits_state=int(cfg[7]),
        has_init=bool(cfg[8]),
        alignments=alignments,
    )
    compiled = _FWD_HOST_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=False)
        host_wrapper = _make_v2x2ssd_fwd_host_wrapper(spec=spec, cfg=cfg)
        compiled = cute.compile(host_wrapper, *dynamic_args)
        _FWD_HOST_CACHE[cache_key] = compiled
    else:
        note_cache_event("cute.v2x2ssd.fwd.host_compile", hit=True)
    compiled(*dynamic_args)
    Y, final_state, m_chunk, chunk_starts = outputs
    if not return_final_state:
        return Y, m_chunk, chunk_starts
    assert final_state is not None
    return Y, final_state, m_chunk, chunk_starts


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
