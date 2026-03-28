"""Thin CuTe host-JIT boundary for training-only ``v2x2ssd`` backward."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from ..fwd.common import (
    _assumed_align,
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
    _public_dk_from_parts,
    _public_from_chunked,
    _public_from_param_scan,
    _resolve_dz0_cta_tiler,
)
from .chunk_scan.db import ChunkScanBwdDBAmpere
from .chunk_scan.dc import ChunkScanBwdDCAmpere
from .chunk_scan.dlp import ChunkScanBwdDLPAmpere
from .chunk_scan.du import ChunkScanBwdDUAmpere
from .chunk_scan.dz0 import ChunkScanBwdDZ0Ampere
from .chunk_scan.param_scan import ChunkScanBwdParamScanAmpere
from .state_passing.common import _TileConfig, _choose_copy_bits_for_linear_tiles
from .state_passing.m import StatePassingBwdMAmpere
from .state_passing.state import StatePassingBwdStateAmpere


_BWD_HOST_CACHE: dict[tuple, object] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_FINAL_GRAD_CACHE: dict[tuple, torch.Tensor] = {}
_BWD_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}
_ZERO_PREV_CACHE_LIMIT = 8
_ZERO_FINAL_GRAD_CACHE_LIMIT = 8
_BWD_WORKSPACE_CACHE_LIMIT = 4
_CAST_D_INC_NUM_THREADS = 256


@cute.kernel
def _cast_f32_to_tc_kernel(gSrc: cute.Tensor, gDst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * int(_CAST_D_INC_NUM_THREADS) + tidx
    if idx < gSrc.shape[0]:
        gDst[idx] = gSrc[idx].to(gDst.element_type)


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
        cached = (
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
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
        cached = torch.zeros(
            (batch_size, heads, P, D), device=device, dtype=torch.float32
        )
        _cache_set(
            _ZERO_FINAL_GRAD_CACHE,
            key,
            cached,
            limit=_ZERO_FINAL_GRAD_CACHE_LIMIT,
        )
    return cached


def _get_bwd_workspace(
    *,
    device: torch.device,
    tc_dtype: torch.dtype,
    batch_size: int,
    heads: int,
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
        int(n_chunks),
        int(chunk_size),
        int(P),
        int(D),
        int(n_d_tiles),
    )
    cached = _BWD_WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached

    BH = int(batch_size) * int(heads)
    BHC = BH * int(n_chunks)
    L = int(chunk_size)

    cached = (
        torch.empty((BH, n_chunks, P), device=device, dtype=tc_dtype),
        torch.empty((BH, n_chunks, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, P, D), device=device, dtype=torch.float32),
        torch.empty(
            (batch_size, heads, n_chunks, P, D), device=device, dtype=torch.float32
        ),
        torch.empty((batch_size, heads, n_chunks, P, D), device=device, dtype=tc_dtype),
        torch.empty((batch_size, heads, P, D), device=device, dtype=torch.float32),
        torch.empty(
            (batch_size, heads, n_chunks, 2), device=device, dtype=torch.float32
        ),
        torch.empty((BHC, L, 1, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, 1, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 4), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, 1, L, 2), device=device, dtype=torch.float32),
        torch.empty((BHC, L, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, D), device=device, dtype=tc_dtype),
        torch.empty((BHC, L, P), device=device, dtype=tc_dtype),
        torch.empty((BHC, P), device=device, dtype=tc_dtype),
        torch.empty((2, L, n_d_tiles, BHC), device=device, dtype=torch.float32),
        torch.empty((2, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
        torch.empty((2, L, BHC), device=device, dtype=torch.float32),
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


def _make_ptr_arg(t: torch.Tensor) -> tuple[object, int]:
    align = _assumed_align(t)
    return (
        make_ptr(
            _torch_to_cutlass_dtype(t.dtype),
            t.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=align,
        ),
        align,
    )


def _make_ptr_args(
    *tensors: torch.Tensor,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    ptrs: list[object] = []
    alignments: list[int] = []
    for tensor in tensors:
        ptr, align = _make_ptr_arg(tensor)
        ptrs.append(ptr)
        alignments.append(align)
    return tuple(ptrs), tuple(alignments)


def _make_cast_f32_to_tc(*, total_elems: int, num_threads: int = 256):
    if int(num_threads) != _CAST_D_INC_NUM_THREADS:
        raise ValueError(
            f"cast kernel expects {_CAST_D_INC_NUM_THREADS} threads, got {num_threads}."
        )
    grid_x = (int(total_elems) + _CAST_D_INC_NUM_THREADS - 1) // _CAST_D_INC_NUM_THREADS
    return _cast_f32_to_tc_kernel, int(grid_x), _CAST_D_INC_NUM_THREADS


def _bwd_host_cache_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    m_chunk_shape: tuple[int, ...],
    chunk_starts_shape: tuple[int, ...],
    d_out_shape: tuple[int, ...],
    chunk_size: int,
    scan_num_threads_du: int,
    scan_num_threads_db: int,
    scan_num_threads_dc: int,
    scan_num_threads_param: int,
    state_num_threads: int,
    state_pairs_per_thread: int,
    state_copy_bits_state: int,
    state_copy_bits_out: int,
    state_copy_bits_final: int,
    n_d_tiles: int,
    dz0_cta_tiler: tuple[int, int, int],
    alignments: tuple[int, ...],
) -> tuple:
    return (
        "v2x2ssd_bwd_host",
        device_index,
        tc_dtype,
        U_shape,
        M_shape,
        K_shape,
        B_shape,
        C_shape,
        m_chunk_shape,
        chunk_starts_shape,
        d_out_shape,
        int(chunk_size),
        int(scan_num_threads_du),
        int(scan_num_threads_db),
        int(scan_num_threads_dc),
        int(scan_num_threads_param),
        int(state_num_threads),
        int(state_pairs_per_thread),
        int(state_copy_bits_state),
        int(state_copy_bits_out),
        int(state_copy_bits_final),
        int(n_d_tiles),
        dz0_cta_tiler,
        alignments,
    )


def _build_backward_args(
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
    scan_num_threads_dc: int,
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
) -> tuple[
    tuple[object, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[object, ...],
    tuple[torch.Tensor, ...],
]:
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")
    if U.dtype != B.dtype or U.dtype != C.dtype:
        raise ValueError("U/B/C must share dtype.")
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("U/B/C must be float16/bfloat16/float32.")
    if M.dtype != torch.float32 or K.dtype != torch.float32:
        raise TypeError("M and K must be float32.")
    if d_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("d_out must be float16/bfloat16/float32.")
    if m_chunk.dtype != torch.float32 or chunk_starts.dtype != torch.float32:
        raise TypeError("m_chunk and chunk_starts must be float32.")

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D) or C.shape != (Bsz, H, T, D):
        raise ValueError("B/C must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if d_out.shape != (Bsz, H, T, P):
        raise ValueError("d_out must be (B,H,T,P) matching U.")
    if B_prev is not None:
        if tuple(B_prev.shape) != (Bsz, H, D) or tuple(U_prev.shape) != (Bsz, H, P):
            raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
        if B_prev.device != U.device or U_prev.device != U.device:
            raise ValueError("B_prev/U_prev must be on the same device as U.")
    if d_final_state is not None:
        if tuple(d_final_state.shape) != (Bsz, H, P, D):
            raise ValueError("d_final_state must be (B,H,P,D).")
        if d_final_state.device != U.device:
            raise ValueError("d_final_state must be on the same device as U.")
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2.")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    if tuple(m_chunk.shape) != (Bsz, H, n_chunks, 2):
        raise ValueError(
            f"m_chunk must be (B,H,C,2)={(Bsz, H, n_chunks, 2)}. "
            f"Got {tuple(m_chunk.shape)}."
        )
    if tuple(chunk_starts.shape) != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D)={(Bsz, H, n_chunks, P, D)}. "
            f"Got {tuple(chunk_starts.shape)}."
        )

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    dz0_cta_tiler = _resolve_dz0_cta_tiler(D=D)

    if prepared_inputs is None:
        U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
        M_f = _pad_m_identity(M, T_pad=T_pad)
        K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
        B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
        C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
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
    d_out_tc = _pad_zero_time(d_out, T_pad=T_pad, dtype=tc_dtype)
    m_chunk_f = (
        m_chunk
        if m_chunk.dtype == torch.float32 and m_chunk.is_contiguous()
        else m_chunk.to(dtype=torch.float32).contiguous()
    )
    chunk_starts_f = (
        chunk_starts
        if chunk_starts.dtype == torch.float32 and chunk_starts.is_contiguous()
        else chunk_starts.to(dtype=torch.float32).contiguous()
    )

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
    if d_final_state is None:
        d_final = _get_zero_final_grad(
            device=U.device,
            batch_size=Bsz,
            heads=H,
            P=P,
            D=D,
        )
    else:
        d_final = d_final_state.to(dtype=torch.float32).contiguous()

    BH = Bsz * H
    U_chunks = U_tc.reshape(BH, n_chunks, L, P)
    B_chunks = B_tc.reshape(BH, n_chunks, L, D)

    cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
    inc_db_kernel = ChunkIncrementBwdDBAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    n_d_tiles = (D + inc_db_kernel.bN - 1) // inc_db_kernel.bN

    (
        U_prev_chunks,
        B_prev_chunks,
        dZ0,
        d_inc,
        d_inc_tc,
        d_initial,
        d_m_chunk,
        dU_scan,
        dU_prev_scan,
        dB_scan,
        dB_prev_scan,
        dC_scan,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dMp_scratch,
        dMc_scratch,
        dM_scan,
        dKprev_scan,
        dKcurr_scan,
        dB_inc,
        dB_prev_inc,
        dU_inc,
        dU_prev_inc,
        dMsum_part,
        dMp0,
        dM_inc,
        dKprev_inc,
        dKcurr_inc,
    ) = _get_bwd_workspace(
        device=U.device,
        tc_dtype=tc_dtype,
        batch_size=Bsz,
        heads=H,
        n_chunks=n_chunks,
        chunk_size=L,
        P=P,
        D=D,
        n_d_tiles=n_d_tiles,
    )

    U_prev_chunks[:, 0, :] = U_prev0.reshape(BH, P)
    B_prev_chunks[:, 0, :] = B_prev0.reshape(BH, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunks[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunks[:, :-1, -1, :]

    state_cfg = _TileConfig(
        num_threads=int(state_num_threads),
        pairs_per_thread=int(state_pairs_per_thread),
    )
    state_copy_bits_state = _choose_copy_bits_for_linear_tiles(
        dZ0.reshape(Bsz, H, n_chunks, P, D),
        tile_stride_elems=P * D,
        elems_per_thread=state_cfg.elems_per_thread,
    )
    state_copy_bits_final = _choose_copy_bits_for_linear_tiles(
        d_final,
        tile_stride_elems=P * D,
        elems_per_thread=state_cfg.elems_per_thread,
    )
    state_copy_bits_out = _choose_copy_bits_for_linear_tiles(
        d_inc,
        tile_stride_elems=P * D,
        elems_per_thread=state_cfg.elems_per_thread,
    )

    Kprev_view = K_f[:, :, :, 0, :]
    Kcurr_view = K_f[:, :, :, 1, :]

    ptr_tensors = [
        U_tc,
        B_tc,
        C_tc,
        M_f,
        K_f,
        Kprev_view,
        Kcurr_view,
        d_out_tc,
        m_chunk_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        d_final,
        U_prev_chunks,
        B_prev_chunks,
        dZ0,
        d_inc,
        d_inc_tc,
        d_initial,
        d_m_chunk,
        dU_scan,
        dU_prev_scan,
        dB_scan,
        dB_prev_scan,
        dC_scan,
        dU_db_dummy,
        dU_prev_db_dummy,
        dB_du_dummy,
        dB_prev_du_dummy,
        dlogp,
        dR,
        dMp_scratch,
        dMc_scratch,
        dM_scan,
        dKprev_scan,
        dKcurr_scan,
        dB_inc,
        dB_prev_inc,
        dU_inc,
        dU_prev_inc,
        dMsum_part,
        dMp0,
        dM_inc,
        dKprev_inc,
        dKcurr_inc,
    ]
    dynamic_args, alignments = _make_ptr_args(*ptr_tensors)

    spec = (
        Bsz,
        H,
        T_pad,
        P,
        D,
        n_chunks,
        L,
        n_d_tiles,
    )
    cfg = (
        int(scan_num_threads_du),
        int(scan_num_threads_db),
        int(scan_num_threads_dc),
        int(scan_num_threads_param),
        int(state_num_threads),
        int(state_pairs_per_thread),
        int(state_copy_bits_state),
        int(state_copy_bits_out),
        int(state_copy_bits_final),
        dz0_cta_tiler,
    )

    dU_scan_view = dU_scan.reshape(Bsz, H, n_chunks, L, P)
    dU_prev_scan_view = dU_prev_scan.reshape(Bsz, H, n_chunks, P)
    dB_scan_view = dB_scan.reshape(Bsz, H, n_chunks, L, D)
    dB_prev_scan_view = dB_prev_scan.reshape(Bsz, H, n_chunks, D)
    dC_scan_view = dC_scan.reshape(Bsz, H, n_chunks, L, D)
    dM_scan_view = dM_scan.reshape(Bsz, H, n_chunks, 1, L, 2)
    dKprev_scan_view = dKprev_scan.reshape(Bsz, H, n_chunks, 1, L, 2)
    dKcurr_scan_view = dKcurr_scan.reshape(Bsz, H, n_chunks, 1, L, 2)

    dU_inc_view = dU_inc.reshape(Bsz, H, n_chunks, L, P)
    dU_prev_inc_view = dU_prev_inc.reshape(Bsz, H, n_chunks, P)
    dB_inc_view = dB_inc.reshape(Bsz, H, n_chunks, L, D)
    dB_prev_inc_view = dB_prev_inc.reshape(Bsz, H, n_chunks, D)
    dM_inc_view = dM_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKprev_inc_view = dKprev_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKcurr_inc_view = dKcurr_inc.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)

    keepalive = (
        U_tc,
        B_tc,
        C_tc,
        d_out_tc,
        M_f,
        K_f,
        m_chunk_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        d_final,
        U_prev_chunks,
        B_prev_chunks,
        d_inc_tc,
    )
    return (
        dynamic_args,
        alignments,
        spec,
        cfg,
        (
            d_initial,
            dU_scan_view,
            dU_prev_scan_view,
            dB_scan_view,
            dB_prev_scan_view,
            dC_scan_view,
            dM_scan_view,
            dKprev_scan_view,
            dKcurr_scan_view,
            dU_inc_view,
            dU_prev_inc_view,
            dB_inc_view,
            dB_prev_inc_view,
            dM_inc_view,
            dKprev_inc_view,
            dKcurr_inc_view,
            *keepalive,
        ),
    )


def _make_v2x2ssd_bwd_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[object, ...],
):
    Bsz, H, T_pad, P, D, n_chunks, L, n_d_tiles = spec
    (
        scan_num_threads_du,
        scan_num_threads_db,
        scan_num_threads_dc,
        scan_num_threads_param,
        state_num_threads,
        state_pairs_per_thread,
        state_copy_bits_state,
        state_copy_bits_out,
        state_copy_bits_final,
        dz0_cta_tiler,
    ) = cfg
    BH = Bsz * H
    BHC = BH * n_chunks

    u_scan_spec = _make_tensor_spec((BHC, L, 1, P), stride=(L * P, P, P, 1))
    b_scan_spec = _make_tensor_spec((BHC, L, 1, D), stride=(L * D, D, D, 1))
    m_scan_spec = _make_tensor_spec((BHC, L, 2), stride=(L * 2, 2, 1))
    k_scan_spec = _make_tensor_spec((BHC, L, 2, 2), stride=(L * 4, 4, 2, 1))
    z0_scan_spec = _make_tensor_spec((BHC, P, D), stride=(P * D, D, 1))
    u_prev0_scan_spec = _make_tensor_spec((BH, P), stride=(P, 1))
    b_prev0_scan_spec = _make_tensor_spec((BH, D), stride=(D, 1))
    dlogp_scan_spec = _make_tensor_spec((BHC, L), stride=(L, 1))
    dm_scratch_scan_spec = _make_tensor_spec((BHC, L, 2), stride=(L * 2, 2, 1))
    dr_scan_spec = _make_tensor_spec((BHC, L, 4), stride=(L * 4, 4, 1))
    dparam_scan_spec = _make_tensor_spec((BHC, 1, L, 2), stride=(L * 2, L * 2, 2, 1))
    dlp_param_spec = _make_tensor_spec((BHC, 1, L), stride=(L, L, 1))
    dr_param_spec = _make_tensor_spec((BHC, 1, L, 4), stride=(L * 4, L * 4, 4, 1))
    flat_state_spec = _make_tensor_spec((BHC * P * D,))

    dz0_dout_spec = _make_tensor_spec((P, T_pad, BH), stride=(1, P, T_pad * P))
    dz0_c_spec = _make_tensor_spec((D, T_pad, BH), stride=(1, D, T_pad * D))
    dz0_m_spec = _make_tensor_spec((2, T_pad, BH), stride=(1, 2, T_pad * 2))
    dz0_out_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))

    state_spec = _make_tensor_spec((Bsz, H, n_chunks, P, D))
    m_chunk_state_spec = _make_tensor_spec((Bsz, H, n_chunks, 2))
    final_state_spec = _make_tensor_spec((Bsz, H, P, D))

    u_inc_spec = _make_tensor_spec((L, P, BHC), stride=(P, 1, L * P))
    du_inc_spec = _make_tensor_spec((P, L, BHC), stride=(1, P, L * P))
    b_inc_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_inc_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    k_inc_spec = _make_tensor_spec((2, L, BHC), stride=(1, 4, L * 4))
    d_inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    d_inc_dp_spec = _make_tensor_spec((D, P, BHC), stride=(1, D, P * D))
    d_inc_boundary_spec = _make_tensor_spec((BHC, P, D), stride=(P * D, D, 1))
    prev_u_chunks_spec = _make_tensor_spec((P, BHC), stride=(1, P))
    prev_b_chunks_spec = _make_tensor_spec((D, BHC), stride=(1, D))
    dmsum_part_spec = _make_tensor_spec((2, L, n_d_tiles, BHC))
    dmp0_spec = _make_tensor_spec((2, BHC))
    dmchunk_inc_spec = _make_tensor_spec((2, BHC), stride=(1, 2))
    dparam_inc_spec = _make_tensor_spec((2, L, BHC))
    d_dummy_u_spec = _make_tensor_spec((BHC, L, 1, P), stride=(L * P, P, P, 1))
    d_dummy_b_spec = _make_tensor_spec((BHC, L, 1, D), stride=(L * D, D, D, 1))
    d_dummy_u_prev_spec = _make_tensor_spec((BHC, P), stride=(P, 1))
    d_dummy_b_prev_spec = _make_tensor_spec((BHC, D), stride=(D, 1))

    @cute.jit
    def _v2x2ssd_bwd_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        Kprev_ptr: cute.Pointer,
        Kcurr_ptr: cute.Pointer,
        dOut_ptr: cute.Pointer,
        m_chunk_ptr: cute.Pointer,
        chunk_starts_ptr: cute.Pointer,
        U_prev0_ptr: cute.Pointer,
        B_prev0_ptr: cute.Pointer,
        d_final_ptr: cute.Pointer,
        U_prev_chunks_ptr: cute.Pointer,
        B_prev_chunks_ptr: cute.Pointer,
        dZ0_ptr: cute.Pointer,
        d_inc_ptr: cute.Pointer,
        d_inc_tc_ptr: cute.Pointer,
        d_initial_ptr: cute.Pointer,
        d_m_chunk_ptr: cute.Pointer,
        dU_scan_ptr: cute.Pointer,
        dU_prev_scan_ptr: cute.Pointer,
        dB_scan_ptr: cute.Pointer,
        dB_prev_scan_ptr: cute.Pointer,
        dC_scan_ptr: cute.Pointer,
        dU_db_dummy_ptr: cute.Pointer,
        dU_prev_db_dummy_ptr: cute.Pointer,
        dB_du_dummy_ptr: cute.Pointer,
        dB_prev_du_dummy_ptr: cute.Pointer,
        dlogp_ptr: cute.Pointer,
        dR_ptr: cute.Pointer,
        dMp_scratch_ptr: cute.Pointer,
        dMc_scratch_ptr: cute.Pointer,
        dM_scan_ptr: cute.Pointer,
        dKprev_scan_ptr: cute.Pointer,
        dKcurr_scan_ptr: cute.Pointer,
        dB_inc_ptr: cute.Pointer,
        dB_prev_inc_ptr: cute.Pointer,
        dU_inc_ptr: cute.Pointer,
        dU_prev_inc_ptr: cute.Pointer,
        dMsum_part_ptr: cute.Pointer,
        dMp0_ptr: cute.Pointer,
        dM_inc_ptr: cute.Pointer,
        dKprev_inc_ptr: cute.Pointer,
        dKcurr_inc_ptr: cute.Pointer,
    ):
        mU_scan = _make_tensor_from_spec(U_ptr, u_scan_spec)
        mB_scan = _make_tensor_from_spec(B_ptr, b_scan_spec)
        mC_scan = _make_tensor_from_spec(C_ptr, b_scan_spec)
        mM_scan = _make_tensor_from_spec(M_ptr, m_scan_spec)
        mK_scan = _make_tensor_from_spec(K_ptr, k_scan_spec)
        mDOut_scan = _make_tensor_from_spec(dOut_ptr, u_scan_spec)
        mZ0_scan = _make_tensor_from_spec(chunk_starts_ptr, z0_scan_spec)
        mU_prev0_scan = _make_tensor_from_spec(U_prev0_ptr, u_prev0_scan_spec)
        mB_prev0_scan = _make_tensor_from_spec(B_prev0_ptr, b_prev0_scan_spec)
        mDU_scan = _make_tensor_from_spec(dU_scan_ptr, u_scan_spec)
        mDB_scan = _make_tensor_from_spec(dB_scan_ptr, b_scan_spec)
        mDU_prev_scan = _make_tensor_from_spec(dU_prev_scan_ptr, d_dummy_u_prev_spec)
        mDB_prev_scan = _make_tensor_from_spec(dB_prev_scan_ptr, d_dummy_b_prev_spec)
        mDLogp = _make_tensor_from_spec(dlogp_ptr, dlogp_scan_spec)
        mDLogp_param = _make_tensor_from_spec(dlogp_ptr, dlp_param_spec)
        mDMprev_scan = _make_tensor_from_spec(dMp_scratch_ptr, dm_scratch_scan_spec)
        mDMcurr_scan = _make_tensor_from_spec(dMc_scratch_ptr, dm_scratch_scan_spec)
        mDMprev_param = _make_tensor_from_spec(dMp_scratch_ptr, dparam_scan_spec)
        mDMcurr_param = _make_tensor_from_spec(dMc_scratch_ptr, dparam_scan_spec)
        mDC_scan = _make_tensor_from_spec(dC_scan_ptr, b_scan_spec)
        mDR_scan = _make_tensor_from_spec(dR_ptr, dr_scan_spec)
        mDR_param = _make_tensor_from_spec(dR_ptr, dr_param_spec)
        mDM_scan = _make_tensor_from_spec(dM_scan_ptr, dparam_scan_spec)
        mDKprev_scan = _make_tensor_from_spec(dKprev_scan_ptr, dparam_scan_spec)
        mDKcurr_scan = _make_tensor_from_spec(dKcurr_scan_ptr, dparam_scan_spec)

        mDOut_dz0 = _make_tensor_from_spec(dOut_ptr, dz0_dout_spec)
        mC_dz0 = _make_tensor_from_spec(C_ptr, dz0_c_spec)
        mM_dz0 = _make_tensor_from_spec(M_ptr, dz0_m_spec)
        mDZ0 = _make_tensor_from_spec(dZ0_ptr, dz0_out_spec)

        mChunkStarts_state = _make_tensor_from_spec(chunk_starts_ptr, state_spec)
        mMchunk_state = _make_tensor_from_spec(m_chunk_ptr, m_chunk_state_spec)
        mDChunkStarts_state = _make_tensor_from_spec(dZ0_ptr, state_spec)
        mDFinal = _make_tensor_from_spec(d_final_ptr, final_state_spec)
        mDInc_state = _make_tensor_from_spec(d_inc_ptr, state_spec)
        mDInc_flat = _make_tensor_from_spec(d_inc_ptr, flat_state_spec)
        mDInc_tc_flat = _make_tensor_from_spec(d_inc_tc_ptr, flat_state_spec)
        mDInitial = _make_tensor_from_spec(d_initial_ptr, final_state_spec)
        mDMchunk_state = _make_tensor_from_spec(d_m_chunk_ptr, m_chunk_state_spec)

        mU_inc = _make_tensor_from_spec(U_ptr, u_inc_spec)
        mB_inc = _make_tensor_from_spec(B_ptr, b_inc_spec)
        mM_inc = _make_tensor_from_spec(M_ptr, m_inc_spec)
        mKprev_inc = _make_tensor_from_spec(Kprev_ptr, k_inc_spec)
        mKcurr_inc = _make_tensor_from_spec(Kcurr_ptr, k_inc_spec)
        mDInc_DP = _make_tensor_from_spec(d_inc_tc_ptr, d_inc_dp_spec)
        mDInc = _make_tensor_from_spec(d_inc_tc_ptr, d_inc_spec)
        mDInc_boundary = _make_tensor_from_spec(d_inc_tc_ptr, d_inc_boundary_spec)
        mBPrev_chunks = _make_tensor_from_spec(B_prev_chunks_ptr, prev_b_chunks_spec)
        mUPrev_chunks = _make_tensor_from_spec(U_prev_chunks_ptr, prev_u_chunks_spec)
        mDB_inc = _make_tensor_from_spec(dB_inc_ptr, b_inc_spec)
        mDU_inc = _make_tensor_from_spec(dU_inc_ptr, du_inc_spec)
        mDBPrev_inc = _make_tensor_from_spec(dB_prev_inc_ptr, prev_b_chunks_spec)
        mDUPrev_inc = _make_tensor_from_spec(dU_prev_inc_ptr, prev_u_chunks_spec)
        mDMsum_part = _make_tensor_from_spec(dMsum_part_ptr, dmsum_part_spec)
        mDMp0 = _make_tensor_from_spec(dMp0_ptr, dmp0_spec)
        mDMchunk_inc = _make_tensor_from_spec(d_m_chunk_ptr, dmchunk_inc_spec)
        mDM_inc = _make_tensor_from_spec(dM_inc_ptr, dparam_inc_spec)
        mDKprev_inc = _make_tensor_from_spec(dKprev_inc_ptr, dparam_inc_spec)
        mDKcurr_inc = _make_tensor_from_spec(dKcurr_inc_ptr, dparam_inc_spec)

        mDU_db_dummy = _make_tensor_from_spec(dU_db_dummy_ptr, d_dummy_u_spec)
        mDU_prev_db_dummy = _make_tensor_from_spec(
            dU_prev_db_dummy_ptr, d_dummy_u_prev_spec
        )
        mDB_du_dummy = _make_tensor_from_spec(dB_du_dummy_ptr, d_dummy_b_spec)
        mDB_prev_du_dummy = _make_tensor_from_spec(
            dB_prev_du_dummy_ptr, d_dummy_b_prev_spec
        )

        tc_dtype = U_ptr.value_type

        scan_db = ChunkScanBwdDBAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_db,
        )
        scan_dc = ChunkScanBwdDCAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_dc,
        )
        scan_dlp = ChunkScanBwdDLPAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_dc,
        )
        scan_param = ChunkScanBwdParamScanAmpere(
            chunk_size=L, num_threads=scan_num_threads_param
        )
        scan_du = ChunkScanBwdDUAmpere(
            tc_dtype,
            chunk_size=L,
            D=D,
            P=P,
            num_threads=scan_num_threads_du,
        )
        scan_dz0 = ChunkScanBwdDZ0Ampere(
            tc_dtype, chunk_size=L, cta_tiler=dz0_cta_tiler
        )
        cast_d_inc, cast_grid_x, cast_num_threads = _make_cast_f32_to_tc(
            total_elems=BHC * P * D
        )

        state_cfg = _TileConfig(
            num_threads=state_num_threads,
            pairs_per_thread=state_pairs_per_thread,
        )
        state_bwd = StatePassingBwdStateAmpere(
            state_cfg,
            copy_bits_in=state_copy_bits_state,
            copy_bits_out=state_copy_bits_out,
            copy_bits_final=state_copy_bits_final,
        )
        state_m_bwd = StatePassingBwdMAmpere(
            state_cfg, copy_bits_in=state_copy_bits_state
        )

        inc_db = ChunkIncrementBwdDBAmpere(tc_dtype, chunk_size=L, D=D, P=P)
        inc_boundary = ChunkIncrementBwdBoundaryAmpere(tc_dtype, chunk_size=L, D=D, P=P)
        inc_param = ChunkIncrementBwdParamScanAmpere(chunk_size=L, nDtiles=n_d_tiles)
        inc_du = ChunkIncrementBwdDUAmpere(tc_dtype, chunk_size=L, D=D, P=P)

        scan_db(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mDU_db_dummy,
            mDB_scan,
            mDU_prev_db_dummy,
            mDB_prev_scan,
            mDLogp,
            mDMprev_scan,
            mDMcurr_scan,
        )
        scan_dc(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mZ0_scan,
            mDC_scan,
            mDLogp,
            mDR_scan,
        )
        scan_dlp(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mDLogp,
        )
        scan_param(
            mM_scan,
            mK_scan,
            mDLogp_param,
            mDMprev_param,
            mDMcurr_param,
            mDR_param,
            mDM_scan,
            mDKprev_scan,
            mDKcurr_scan,
        )
        scan_du(
            mU_scan,
            mB_scan,
            mC_scan,
            mM_scan,
            mK_scan,
            mDOut_scan,
            mU_prev0_scan,
            mB_prev0_scan,
            mDU_scan,
            mDB_du_dummy,
            mDU_prev_scan,
            mDB_prev_du_dummy,
            mDLogp,
            mDMprev_scan,
            mDMcurr_scan,
        )
        scan_dz0(mDOut_dz0, mC_dz0, mM_dz0, mDZ0)

        state_bwd(
            mDChunkStarts_state,
            mDFinal,
            mMchunk_state,
            mDInc_state,
            mDInitial,
        )
        state_m_bwd(mChunkStarts_state, mDInc_state, mDMchunk_state)
        cast_d_inc(mDInc_flat, mDInc_tc_flat).launch(
            grid=[cast_grid_x, 1, 1],
            block=[cast_num_threads, 1, 1],
        )

        inc_db(
            mU_inc,
            mB_inc,
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDInc_DP,
            mDB_inc,
            mDMsum_part,
        )
        inc_boundary(
            mDInc_boundary,
            mBPrev_chunks,
            mUPrev_chunks,
            mM_inc,
            mKprev_inc,
            mDUPrev_inc,
            mDBPrev_inc,
            mDMp0,
        )
        inc_param(
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDMsum_part,
            mDMp0,
            mDMchunk_inc,
            mDM_inc,
            mDKprev_inc,
            mDKcurr_inc,
        )
        inc_du(
            mDInc,
            mB_inc,
            mM_inc,
            mKprev_inc,
            mKcurr_inc,
            mDU_inc,
        )

    return _v2x2ssd_bwd_host_wrapper


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
    scan_num_threads_dc: int = 128,
    scan_num_threads_param: int = 32,
    state_num_threads: int = 128,
    state_pairs_per_thread: int = 8,
) -> object:
    dynamic_args, alignments, spec, cfg, _outputs = _build_backward_args(
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
        scan_num_threads_dc=scan_num_threads_dc,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
    )
    cache_key = _bwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        d_out_shape=tuple(d_out.shape),
        chunk_size=int(chunk_size),
        scan_num_threads_du=int(scan_num_threads_du),
        scan_num_threads_db=int(scan_num_threads_db),
        scan_num_threads_dc=int(scan_num_threads_dc),
        scan_num_threads_param=int(scan_num_threads_param),
        state_num_threads=int(state_num_threads),
        state_pairs_per_thread=int(state_pairs_per_thread),
        state_copy_bits_state=int(cfg[6]),
        state_copy_bits_out=int(cfg[7]),
        state_copy_bits_final=int(cfg[8]),
        dz0_cta_tiler=cfg[9],
        n_d_tiles=int(spec[7]),
        alignments=alignments,
    )
    cached = _BWD_HOST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    host_wrapper = _make_v2x2ssd_bwd_host_wrapper(spec=spec, cfg=cfg)
    compiled = cute.compile(host_wrapper, *dynamic_args)
    _BWD_HOST_CACHE[cache_key] = compiled
    return compiled


def _v2x2ssd_bwd_cute_impl(
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
    scan_num_threads_dc: int = 128,
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
    dynamic_args, alignments, spec, cfg, outputs = _build_backward_args(
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
        scan_num_threads_dc=scan_num_threads_dc,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=prepared_inputs,
    )
    cache_key = _bwd_host_cache_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=_tc_input_dtype(U.dtype, compute_dtype),
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        d_out_shape=tuple(d_out.shape),
        chunk_size=int(chunk_size),
        scan_num_threads_du=int(scan_num_threads_du),
        scan_num_threads_db=int(scan_num_threads_db),
        scan_num_threads_dc=int(scan_num_threads_dc),
        scan_num_threads_param=int(scan_num_threads_param),
        state_num_threads=int(state_num_threads),
        state_pairs_per_thread=int(state_pairs_per_thread),
        state_copy_bits_state=int(cfg[6]),
        state_copy_bits_out=int(cfg[7]),
        state_copy_bits_final=int(cfg[8]),
        dz0_cta_tiler=cfg[9],
        n_d_tiles=int(spec[7]),
        alignments=alignments,
    )
    compiled = _BWD_HOST_CACHE.get(cache_key)
    if compiled is None:
        host_wrapper = _make_v2x2ssd_bwd_host_wrapper(spec=spec, cfg=cfg)
        compiled = cute.compile(host_wrapper, *dynamic_args)
        _BWD_HOST_CACHE[cache_key] = compiled

    compiled(*dynamic_args)

    (
        d_initial,
        dU_scan,
        dU_prev_scan,
        dB_scan,
        dB_prev_scan,
        dC_scan,
        dM_scan,
        dKprev_scan,
        dKcurr_scan,
        dU_inc,
        dU_prev_inc,
        dB_inc,
        dB_prev_inc,
        dM_inc,
        dKprev_inc,
        dKcurr_inc,
        *_keepalive,
    ) = outputs

    dU_scan.add_(dU_inc)
    dU_prev_scan.add_(dU_prev_inc)
    dB_scan.add_(dB_inc)
    dB_prev_scan.add_(dB_prev_inc)
    dM_scan[:, :, :, 0, :, :].add_(dM_inc)
    dKprev_scan[:, :, :, 0, :, :].add_(dKprev_inc)
    dKcurr_scan[:, :, :, 0, :, :].add_(dKcurr_inc)

    dU_public = _fold_chunk_boundary_carries(dU_scan, dU_prev_scan)
    dB_public = _fold_chunk_boundary_carries(dB_scan, dB_prev_scan)

    return (
        _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
        _public_from_param_scan(dM_scan, T=U.shape[2]),
        _public_dk_from_parts(dKprev_scan, dKcurr_scan, T=U.shape[2]),
        _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
        _public_from_chunked(dC_scan, T=U.shape[2], dtype=C.dtype),
        d_initial.to(dtype=initial_state_dtype or torch.float32).contiguous(),
        dB_prev_scan[:, :, 0, :]
        .to(dtype=B_prev.dtype if B_prev is not None else B.dtype)
        .contiguous(),
        dU_prev_scan[:, :, 0, :]
        .to(dtype=U_prev.dtype if U_prev is not None else U.dtype)
        .contiguous(),
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
    scan_num_threads_dc: int = 128,
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
    dU, dM, dK, dB, dC, _d_initial, _dB_prev, _dU_prev = _v2x2ssd_bwd_cute_impl(
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
        scan_num_threads_dc=scan_num_threads_dc,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        prepared_inputs=prepared_inputs,
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
    scan_num_threads_dc: int = 128,
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
    return _v2x2ssd_bwd_cute_impl(
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
        scan_num_threads_dc=scan_num_threads_dc,
        scan_num_threads_param=scan_num_threads_param,
        state_num_threads=state_num_threads,
        state_pairs_per_thread=state_pairs_per_thread,
        initial_state_dtype=initial_state_dtype,
        B_prev=B_prev,
        U_prev=U_prev,
        d_final_state=d_final_state,
        prepared_inputs=prepared_inputs,
    )


__all__ = [
    "compile_v2x2ssd_bwd_cute",
    "v2x2ssd_bwd_cute",
    "v2x2ssd_bwd_stateful_cute",
]
