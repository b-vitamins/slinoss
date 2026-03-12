"""CuTe backward kernels for the ``v2x2ssd`` chunk-increment stage."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from slinoss.perf import note_cache_event, record_region

from .boundary import ChunkIncrementBwdBoundaryAmpere
from .common import _torch_to_cutlass_dtype
from .db import ChunkIncrementBwdDBAmpere
from .du import ChunkIncrementBwdDUAmpere
from .param_scan import ChunkIncrementBwdParamScanAmpere


_COMPILED_CACHE: dict[tuple, tuple[object, object, object, object]] = {}
_OVERLAP_RESOURCES: dict[
    int,
    tuple[
        torch.cuda.Stream,
        torch.cuda.Stream,
        torch.cuda.Stream,
        torch.cuda.Event,
        torch.cuda.Event,
        torch.cuda.Event,
        torch.cuda.Event,
    ],
] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


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
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
        )
        _ZERO_PREV_CACHE[key] = cached
    return cached


def _tc_input_dtype(
    input_dtype: torch.dtype, compute_dtype: torch.dtype | None
) -> torch.dtype:
    dt = input_dtype if compute_dtype is None else compute_dtype
    if dt in (torch.float16, torch.bfloat16):
        return dt
    if dt == torch.float32:
        return torch.float16
    raise TypeError(f"Unsupported compute dtype: {dt}")


def _pad_zero_time(
    tensor: torch.Tensor,
    *,
    T_pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(dtype=dtype).contiguous()
    T = int(tensor.shape[2])
    if T == T_pad:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=dtype)
    return torch.cat((tensor, pad), dim=2).contiguous()


def _pad_m_identity(M: torch.Tensor, *, T_pad: int) -> torch.Tensor:
    M = M.to(dtype=torch.float32).contiguous()
    T = int(M.shape[2])
    if T == T_pad:
        return M
    pad_shape = list(M.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=M.device, dtype=torch.float32)
    pad[..., 0] = 1.0
    return torch.cat((M, pad), dim=2).contiguous()


def _public_from_chunked(
    x: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    out = x.reshape(B, H, C * L, F)[:, :, :T, :]
    target_dtype = out.dtype if dtype is None else dtype
    if out.dtype != target_dtype:
        out = out.to(dtype=target_dtype)
    return out.contiguous()


def _public_from_param_scan(
    x: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    out = x.reshape(B, H, C * L, F)[:, :, :T, :]
    target_dtype = out.dtype if dtype is None else dtype
    if out.dtype != target_dtype:
        out = out.to(dtype=target_dtype)
    return out.contiguous()


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    if dKprev.shape != dKcurr.shape:
        raise ValueError("dKprev and dKcurr must have identical shapes.")
    dK = torch.stack((dKprev, dKcurr), dim=4)
    B, H, C, L, _, F = map(int, dK.shape)
    return (
        dK.reshape(B, H, C * L, 2, F)[:, :, :T, :, :]
        .to(dtype=torch.float32)
        .contiguous()
    )


def _fold_chunk_boundary_carries(
    x: torch.Tensor,
    x_prev: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 5 or x_prev.ndim != 4:
        raise ValueError("Expected chunked main grads and per-chunk boundary carries.")
    if x.shape[:3] != x_prev.shape[:3] or x.shape[-1] != x_prev.shape[-1]:
        raise ValueError("Chunked grads and boundary carries must agree on (B,H,C,F).")
    n_chunks = int(x.shape[2])
    if n_chunks <= 1:
        return x
    x[:, :, :-1, -1, :].add_(x_prev[:, :, 1:, :])
    return x


def _compiled_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    d_inc_shape: tuple[int, ...],
    d_m_chunk_shape: tuple[int, ...],
    chunk_size: int,
    has_prev: bool,
) -> tuple:
    return (
        "chunk_increment_bwd",
        device_index,
        tc_dtype,
        U_shape,
        B_shape,
        M_shape,
        K_shape,
        d_inc_shape,
        d_m_chunk_shape,
        int(chunk_size),
        has_prev,
    )


def _get_overlap_resources(
    device: torch.device,
) -> tuple[
    torch.cuda.Stream,
    torch.cuda.Stream,
    torch.cuda.Stream,
    torch.cuda.Event,
    torch.cuda.Event,
    torch.cuda.Event,
    torch.cuda.Event,
]:
    device_index = device.index if device.index is not None else -1
    cached = _OVERLAP_RESOURCES.get(device_index)
    if cached is None:
        cached = (
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            torch.cuda.Stream(device=device),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
        )
        _OVERLAP_RESOURCES[device_index] = cached
    return cached


def compile_chunk_increment_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_launchers: bool = False,
    enable_overlapped_launcher: bool = True,
) -> tuple:
    """Compile the standalone chunk-increment backward kernels and allocate outputs."""
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")

    Bsz, H, T, P = map(int, U.shape)
    D = int(B.shape[-1])
    if B.shape != (Bsz, H, T, D):
        raise ValueError("B must be (B,H,T,D) matching U.")
    if M.shape != (Bsz, H, T, 2):
        raise ValueError(f"M must be (B,H,T,2)={(Bsz, H, T, 2)}.")
    if K.shape != (Bsz, H, T, 2, 2):
        raise ValueError(f"K must be (B,H,T,2,2)={(Bsz, H, T, 2, 2)}.")
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2 (flattened 2N).")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L

    if tuple(d_inc.shape) != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            f"d_inc must be (B,H,C,P,D)={(Bsz, H, n_chunks, P, D)}. "
            f"Got {tuple(d_inc.shape)}."
        )
    if tuple(d_m_chunk.shape) != (Bsz, H, n_chunks, 2):
        raise ValueError(
            f"d_m_chunk must be (B,H,C,2)={(Bsz, H, n_chunks, 2)}. "
            f"Got {tuple(d_m_chunk.shape)}."
        )

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
    cache_key = _compiled_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        U_shape=tuple(U.shape),
        B_shape=tuple(B.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        d_inc_shape=tuple(d_inc.shape),
        d_m_chunk_shape=tuple(d_m_chunk.shape),
        chunk_size=L,
        has_prev=B_prev is not None,
    )

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    d_inc_tc = d_inc.to(dtype=tc_dtype).contiguous()
    d_m_chunk_f = d_m_chunk.to(dtype=torch.float32).contiguous()

    if B_prev is None:
        B_prev0, U_prev0 = _get_zero_prev_tensors(
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

    BH = Bsz * H
    BHC = BH * n_chunks

    U_chunks = U_tc.reshape(BH, n_chunks, L, P)
    B_chunks = B_tc.reshape(BH, n_chunks, L, D)
    M_chunks = M_f.reshape(BH, n_chunks, L, 2)
    K_chunks = K_f.reshape(BH, n_chunks, L, 2, 2)

    U_blk = U_chunks.reshape(BHC, L, P).contiguous()
    B_blk = B_chunks.reshape(BHC, L, D).contiguous()
    M_blk = M_chunks.reshape(BHC, L, 2).contiguous()
    Kprev_blk = K_chunks[..., 0, :].reshape(BHC, L, 2).contiguous()
    Kcurr_blk = K_chunks[..., 1, :].reshape(BHC, L, 2).contiguous()

    U_prev_chunks = torch.empty((BH, n_chunks, P), device=U.device, dtype=tc_dtype)
    B_prev_chunks = torch.empty((BH, n_chunks, D), device=U.device, dtype=tc_dtype)
    U_prev_chunks[:, 0, :] = U_prev0.reshape(BH, P)
    B_prev_chunks[:, 0, :] = B_prev0.reshape(BH, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunks[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunks[:, :-1, -1, :]

    U_prev_flat = U_prev_chunks.reshape(BHC, P).contiguous()
    B_prev_flat = B_prev_chunks.reshape(BHC, D).contiguous()
    d_inc_flat = d_inc_tc.reshape(BHC, P, D).contiguous()

    mU = from_dlpack(U_blk.permute(1, 2, 0), assumed_align=16)
    mB = from_dlpack(B_blk.permute(1, 2, 0), assumed_align=16)
    mM = from_dlpack(M_blk.permute(2, 1, 0), assumed_align=16)
    mKprev = from_dlpack(Kprev_blk.permute(2, 1, 0), assumed_align=16)
    mKcurr = from_dlpack(Kcurr_blk.permute(2, 1, 0), assumed_align=16)

    k_db = ChunkIncrementBwdDBAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    k_du = ChunkIncrementBwdDUAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    k_boundary = ChunkIncrementBwdBoundaryAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    nDtiles = (D + k_db.bN - 1) // k_db.bN
    k_param = ChunkIncrementBwdParamScanAmpere(chunk_size=L, nDtiles=nDtiles)

    dB = torch.empty((BHC, L, D), device=U.device, dtype=tc_dtype)
    dU = torch.empty((BHC, L, P), device=U.device, dtype=tc_dtype)
    dB_prev = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dU_prev = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dMsum_part = torch.empty((2, L, nDtiles, BHC), device=U.device, dtype=torch.float32)
    dMp0 = torch.empty((2, BHC), device=U.device, dtype=torch.float32)
    dM_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)
    dKprev_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)
    dKcurr_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)

    mDInc = from_dlpack(d_inc_flat.permute(1, 2, 0), assumed_align=16)
    mDInc_DP = from_dlpack(d_inc_flat.permute(2, 1, 0), assumed_align=16)
    mDInc_boundary = from_dlpack(d_inc_flat, assumed_align=16)
    mBPrev = from_dlpack(B_prev_flat.transpose(0, 1), assumed_align=16)
    mUPrev = from_dlpack(U_prev_flat.transpose(0, 1), assumed_align=16)
    mDB = from_dlpack(dB.permute(1, 2, 0), assumed_align=16)
    mDU = from_dlpack(dU.permute(2, 1, 0), assumed_align=16)
    mDBPrev = from_dlpack(dB_prev.transpose(0, 1), assumed_align=16)
    mDUPrev = from_dlpack(dU_prev.transpose(0, 1), assumed_align=16)
    mDMsum_part = from_dlpack(dMsum_part, assumed_align=16)
    mDMp0 = from_dlpack(dMp0, assumed_align=16)
    mDMchunk = from_dlpack(
        d_m_chunk_f.reshape(BHC, 2).transpose(0, 1), assumed_align=16
    )
    mDM = from_dlpack(dM_out, assumed_align=16)
    mDKprev = from_dlpack(dKprev_out, assumed_align=16)
    mDKcurr = from_dlpack(dKcurr_out, assumed_align=16)

    cached = _COMPILED_CACHE.get(cache_key)
    if cached is None:
        note_cache_event("cache.v2x2ssd.backward.chunk_increment", hit=False)
        compiled_db = cute.compile(
            k_db, mU, mB, mM, mKprev, mKcurr, mDInc_DP, mDB, mDMsum_part
        )
        compiled_du = cute.compile(k_du, mDInc, mB, mM, mKprev, mKcurr, mDU)
        compiled_boundary = cute.compile(
            k_boundary,
            mDInc_boundary,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
        )
        compiled_param = cute.compile(
            k_param,
            mM,
            mKprev,
            mKcurr,
            mDMsum_part,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )
        cached = (compiled_db, compiled_du, compiled_boundary, compiled_param)
        _COMPILED_CACHE[cache_key] = cached
    else:
        note_cache_event("cache.v2x2ssd.backward.chunk_increment", hit=True)
        compiled_db, compiled_du, compiled_boundary, compiled_param = cached

    dB_view = dB.reshape(Bsz, H, n_chunks, L, D)
    dU_view = dU.reshape(Bsz, H, n_chunks, L, P)
    dB_prev_view = dB_prev.reshape(Bsz, H, n_chunks, D)
    dU_prev_view = dU_prev.reshape(Bsz, H, n_chunks, P)
    dMsum_part_view = dMsum_part.permute(3, 1, 2, 0).reshape(
        Bsz, H, n_chunks, L, nDtiles, 2
    )
    dMp0_view = dMp0.permute(1, 0).reshape(Bsz, H, n_chunks, 2)
    dM_view = dM_out.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKprev_view = dKprev_out.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)
    dKcurr_view = dKcurr_out.permute(2, 1, 0).reshape(Bsz, H, n_chunks, L, 2)

    def _launch_db() -> None:
        with record_region("backward.v2x2ssd.chunk_increment.db"):
            compiled_db(mU, mB, mM, mKprev, mKcurr, mDInc_DP, mDB, mDMsum_part)

    def _launch_du() -> None:
        with record_region("backward.v2x2ssd.chunk_increment.du"):
            compiled_du(mDInc, mB, mM, mKprev, mKcurr, mDU)

    def _launch_boundary() -> None:
        with record_region("backward.v2x2ssd.chunk_increment.boundary"):
            compiled_boundary(
                mDInc_boundary,
                mBPrev,
                mUPrev,
                mM,
                mKprev,
                mDUPrev,
                mDBPrev,
                mDMp0,
            )

    def _launch_param() -> None:
        with record_region("backward.v2x2ssd.chunk_increment.param_scan"):
            compiled_param(
                mM,
                mKprev,
                mKcurr,
                mDMsum_part,
                mDMp0,
                mDMchunk,
                mDM,
                mDKprev,
                mDKcurr,
            )

    def launch_sequential() -> None:
        _launch_db()
        _launch_du()
        _launch_boundary()
        _launch_param()

    stream_db = None
    stream_du = None
    stream_boundary = None
    ev_start = None
    ev_db_done = None
    ev_du_done = None
    ev_boundary_done = None

    if return_launchers and enable_overlapped_launcher:
        (
            stream_db,
            stream_du,
            stream_boundary,
            ev_start,
            ev_db_done,
            ev_du_done,
            ev_boundary_done,
        ) = _get_overlap_resources(U.device)

    def launch_overlapped() -> None:
        if stream_db is None or stream_du is None or stream_boundary is None:
            launch_sequential()
            return

        current = torch.cuda.current_stream(device=U.device)
        current.record_event(ev_start)

        stream_db.wait_event(ev_start)
        stream_du.wait_event(ev_start)
        stream_boundary.wait_event(ev_start)

        with torch.cuda.stream(stream_db):
            _launch_db()
            stream_db.record_event(ev_db_done)

        with torch.cuda.stream(stream_du):
            _launch_du()
            stream_du.record_event(ev_du_done)

        with torch.cuda.stream(stream_boundary):
            _launch_boundary()
            stream_boundary.record_event(ev_boundary_done)

        current.wait_event(ev_db_done)
        current.wait_event(ev_boundary_done)
        _launch_param()
        current.wait_event(ev_du_done)

    base = (
        compiled_db,
        compiled_du,
        compiled_boundary,
        compiled_param,
        dB_view,
        dU_view,
        dB_prev_view,
        dU_prev_view,
        dMsum_part_view,
        dMp0_view,
        dM_view,
        dKprev_view,
        dKcurr_view,
    )
    if return_launchers:
        return (*base, launch_sequential, launch_overlapped)
    return base


def chunk_increment_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_prev_grads: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Thin public wrapper over the compiled chunk-increment backward kernel bundle."""
    (
        _compiled_db,
        _compiled_du,
        _compiled_boundary,
        _compiled_param,
        dB,
        dU,
        dB_prev,
        dU_prev,
        _dMsum_part,
        _dMp0,
        dM,
        dKprev,
        dKcurr,
        _launch_sequential,
        launch_overlapped,
    ) = compile_chunk_increment_bwd_kernels(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
        return_launchers=True,
    )
    with record_region("backward.v2x2ssd.chunk_increment.total"):
        launch_overlapped()

    dU_public = _fold_chunk_boundary_carries(dU, dU_prev)
    dB_public = _fold_chunk_boundary_carries(dB, dB_prev)

    if not return_prev_grads:
        return (
            _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
            _public_from_param_scan(dM, T=U.shape[2]),
            _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
            _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
        )
    return (
        _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
        _public_from_param_scan(dM, T=U.shape[2]),
        _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
        _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
        dB_prev[:, :, 0, :].to(dtype=B.dtype).contiguous(),
        dU_prev[:, :, 0, :].to(dtype=U.dtype).contiguous(),
    )


__all__ = [
    "chunk_increment_bwd_cute",
    "compile_chunk_increment_bwd_kernels",
]
