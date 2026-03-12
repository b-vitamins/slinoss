"""CuTe backward kernels for the ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .db import ChunkScanBwdDBAmpere
from .dcdr import ChunkScanBwdDCDRAmpere
from .du import ChunkScanBwdDUAmpere
from .dz0 import ChunkScanBwdDZ0Ampere
from .param_scan import ChunkScanBwdParamScanAmpere


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


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


def _as_cute_compact(
    t: torch.Tensor,
    *,
    leading_dim: int,
    mode: int,
    stride_order: tuple[int, ...],
    divisibility: int,
):
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=mode,
            stride_order=stride_order,
            divisibility=divisibility,
        )
    )


def _public_from_chunked(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    return x.reshape(B, H, C * L, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


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


def _public_from_param_scan(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, S, L, F = map(int, x.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    return (
        x[:, :, :, 0, :, :]
        .reshape(B, H, C * L, F)[:, :, :T, :]
        .to(dtype=torch.float32)
        .contiguous()
    )


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    if dKprev.shape != dKcurr.shape:
        raise ValueError("dKprev and dKcurr must have identical shapes.")
    B, H, C, S, L, F = map(int, dKprev.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    dK = torch.stack((dKprev[:, :, :, 0, :, :], dKcurr[:, :, :, 0, :, :]), dim=4)
    return dK.reshape(B, H, C * L, 2, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


def compile_chunk_scan_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    num_threads_du: int = 128,
    num_threads_db: int = 128,
    num_threads_dc: int = 128,
    num_threads_param: int = 32,
    return_launchers: bool = False,
    enable_overlapped_launcher: bool = True,
) -> tuple:
    """Compile the standalone chunk-scan backward kernels and allocate outputs."""
    if (B_prev is None) ^ (U_prev is None):
        raise ValueError("B_prev and U_prev must be passed together (or both omitted).")
    if U.device.type != "cuda":
        raise ValueError("CUDA tensor required.")

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
    if D % 2 != 0:
        raise ValueError("D must be divisible by 2 (flattened 2N).")

    L = int(chunk_size)
    if L <= 0:
        raise ValueError("chunk_size must be positive.")
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    if chunk_starts.shape != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            "chunk_starts must be (B,H,C,P,D) "
            f"={(Bsz, H, n_chunks, P, D)}. Got {tuple(chunk_starts.shape)}."
        )
    if num_threads_dc != 128:
        raise ValueError("num_threads_dc must be 128 for the dC/dR kernel.")

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
    d_out_tc = _pad_zero_time(d_out, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    chunk_starts_f = chunk_starts.to(dtype=torch.float32).contiguous()

    if B_prev is None:
        B_prev0 = torch.zeros((Bsz, H, D), device=U.device, dtype=tc_dtype)
        U_prev0 = torch.zeros((Bsz, H, P), device=U.device, dtype=tc_dtype)
    else:
        if B_prev.shape != (Bsz, H, D) or U_prev.shape != (Bsz, H, P):
            raise ValueError("B_prev/U_prev must be (B,H,D)/(B,H,P).")
        B_prev0 = B_prev.to(dtype=tc_dtype).contiguous()
        U_prev0 = U_prev.to(dtype=tc_dtype).contiguous()

    BH = Bsz * H
    BHC = BH * n_chunks

    dOut2 = d_out_tc.reshape(BH, T_pad, P).permute(2, 1, 0)
    C2 = C_tc.reshape(BH, T_pad, D).permute(2, 1, 0)
    M2 = M_f.reshape(BH, T_pad, 2).permute(2, 1, 0)

    dZ0 = torch.empty((BHC, P, D), device=U.device, dtype=torch.float32)
    mDOut_dz0 = _as_cute_compact(
        dOut2,
        leading_dim=0,
        mode=0,
        stride_order=(2, 1, 0),
        divisibility=(128 // cutlass_dtype.width),
    )
    mC_dz0 = _as_cute_compact(
        C2,
        leading_dim=0,
        mode=0,
        stride_order=(2, 1, 0),
        divisibility=(128 // cutlass_dtype.width),
    )
    mM_dz0 = from_dlpack(M2, assumed_align=16)
    mDZ0 = (
        from_dlpack(dZ0.permute(1, 2, 0), assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=4)
    )

    k_dz0 = ChunkScanBwdDZ0Ampere(cutlass_dtype, chunk_size=L)
    compiled_dz0 = cute.compile(k_dz0, mDOut_dz0, mC_dz0, mM_dz0, mDZ0)
    dZ0_view = dZ0.reshape(Bsz, H, n_chunks, P, D)

    U_blk = U_tc.reshape(BH, n_chunks, L, P).reshape(BHC, L, 1, P).contiguous()
    B_blk = B_tc.reshape(BH, n_chunks, L, D).reshape(BHC, L, 1, D).contiguous()
    C_blk = C_tc.reshape(BH, n_chunks, L, D).reshape(BHC, L, 1, D).contiguous()
    M_blk = M_f.reshape(BH, n_chunks, L, 2).reshape(BHC, L, 2).contiguous()
    K_blk = K_f.reshape(BH, n_chunks, L, 2, 2).reshape(BHC, L, 2, 2).contiguous()
    dOut_blk = d_out_tc.reshape(BH, n_chunks, L, P).reshape(BHC, L, 1, P).contiguous()
    Z0_blk = chunk_starts_f.reshape(BH, n_chunks, P, D).reshape(BHC, P, D).contiguous()

    U_prev0_flat = U_prev0.reshape(BH, P).contiguous()
    B_prev0_flat = B_prev0.reshape(BH, D).contiguous()

    mU = from_dlpack(U_blk, assumed_align=16)
    mB = from_dlpack(B_blk, assumed_align=16)
    mC = from_dlpack(C_blk, assumed_align=16)
    mM = from_dlpack(M_blk, assumed_align=16)
    mK = from_dlpack(K_blk, assumed_align=16)
    mDOut = from_dlpack(dOut_blk, assumed_align=16)
    mU_prev0 = from_dlpack(U_prev0_flat, assumed_align=16)
    mB_prev0 = from_dlpack(B_prev0_flat, assumed_align=16)

    dU = torch.empty_like(U_blk)
    dU_prev = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dB_du_scratch = torch.empty_like(B_blk)
    dB_prev_du_scratch = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dlp_du_scratch = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dMp_du_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    dMc_du_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)

    mDU = from_dlpack(dU, assumed_align=16)
    mDU_prev = from_dlpack(dU_prev, assumed_align=16)
    mDB_du_scratch = from_dlpack(dB_du_scratch, assumed_align=16)
    mDB_prev_du_scratch = from_dlpack(dB_prev_du_scratch, assumed_align=16)
    mDLp_du_scratch = from_dlpack(dlp_du_scratch, assumed_align=16)
    mDMp_du_scratch = from_dlpack(dMp_du_scratch, assumed_align=16)
    mDMc_du_scratch = from_dlpack(dMc_du_scratch, assumed_align=16)

    k_du = ChunkScanBwdDUAmpere(
        cutlass_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_du,
    )
    compiled_du = cute.compile(
        k_du,
        mU,
        mB,
        mC,
        mM,
        mK,
        mDOut,
        mU_prev0,
        mB_prev0,
        mDU,
        mDB_du_scratch,
        mDU_prev,
        mDB_prev_du_scratch,
        mDLp_du_scratch,
        mDMp_du_scratch,
        mDMc_du_scratch,
    )

    dB = torch.empty_like(B_blk)
    dB_prev = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dU_db_scratch = torch.empty_like(U_blk)
    dU_prev_db_scratch = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dlp_db_scratch = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dMp_db_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    dMc_db_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)

    mDB = from_dlpack(dB, assumed_align=16)
    mDB_prev = from_dlpack(dB_prev, assumed_align=16)
    mDU_db_scratch = from_dlpack(dU_db_scratch, assumed_align=16)
    mDU_prev_db_scratch = from_dlpack(dU_prev_db_scratch, assumed_align=16)
    mDLp_db_scratch = from_dlpack(dlp_db_scratch, assumed_align=16)
    mDMp_db_scratch = from_dlpack(dMp_db_scratch, assumed_align=16)
    mDMc_db_scratch = from_dlpack(dMc_db_scratch, assumed_align=16)

    k_db = ChunkScanBwdDBAmpere(
        cutlass_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_db,
    )
    compiled_db = cute.compile(
        k_db,
        mU,
        mB,
        mC,
        mM,
        mK,
        mDOut,
        mU_prev0,
        mB_prev0,
        mDU_db_scratch,
        mDB,
        mDU_prev_db_scratch,
        mDB_prev,
        mDLp_db_scratch,
        mDMp_db_scratch,
        mDMc_db_scratch,
    )

    dU_view = dU.reshape(Bsz, H, n_chunks, L, P)
    dB_view = dB.reshape(Bsz, H, n_chunks, L, D)
    dU_prev_view = dU_prev.reshape(Bsz, H, n_chunks, P)
    dB_prev_view = dB_prev.reshape(Bsz, H, n_chunks, D)

    dlogp = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dC = torch.empty_like(C_blk)
    dR = torch.empty((BHC, L, 4), device=U.device, dtype=torch.float32)

    mZ0 = from_dlpack(Z0_blk, assumed_align=16)
    mDLogp = from_dlpack(dlogp, assumed_align=16)
    mDC = from_dlpack(dC, assumed_align=16)
    mDR = from_dlpack(dR, assumed_align=16)

    k_dc = ChunkScanBwdDCDRAmpere(
        cutlass_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_dc,
    )
    compiled_dc = cute.compile(
        k_dc,
        mU,
        mB,
        mC,
        mM,
        mK,
        mDOut,
        mU_prev0,
        mB_prev0,
        mZ0,
        mDLogp,
        mDC,
        mDR,
    )
    compiled_dc_fast = None
    mZ0_fast = None
    Z0_blk_fast_keepalive = None
    if return_launchers and enable_overlapped_launcher:
        Z0_blk_fast_keepalive = Z0_blk.to(dtype=tc_dtype)
        mZ0_fast = from_dlpack(Z0_blk_fast_keepalive, assumed_align=16)
        compiled_dc_fast = cute.compile(
            k_dc,
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mZ0_fast,
            mDLogp,
            mDC,
            mDR,
        )

    dlogp_view = dlogp.reshape(Bsz, H, n_chunks, L)
    dC_view = dC.reshape(Bsz, H, n_chunks, L, D)
    dR_view = dR.reshape(Bsz, H, n_chunks, L, 4)

    n_splits = 1
    dlp_blk = dlogp.unsqueeze(1).contiguous()
    dMp_blk = dMp_db_scratch.unsqueeze(1).contiguous()
    dMc_blk = dMc_db_scratch.unsqueeze(1).contiguous()
    dR_blk = dR.unsqueeze(1).contiguous()
    dM_out = torch.empty_like(dMp_blk)
    dkprev_out = torch.empty_like(dMp_blk)
    dkcurr_out = torch.empty_like(dMp_blk)

    mDLp = from_dlpack(dlp_blk, assumed_align=16)
    mDMprev = from_dlpack(dMp_blk, assumed_align=16)
    mDMcurr = from_dlpack(dMc_blk, assumed_align=16)
    mDR_param = from_dlpack(dR_blk, assumed_align=16)
    mDM = from_dlpack(dM_out, assumed_align=16)
    mDKprev = from_dlpack(dkprev_out, assumed_align=16)
    mDKcurr = from_dlpack(dkcurr_out, assumed_align=16)

    k_param = ChunkScanBwdParamScanAmpere(chunk_size=L, num_threads=num_threads_param)
    compiled_param = cute.compile(
        k_param,
        mM,
        mK,
        mDLp,
        mDMprev,
        mDMcurr,
        mDR_param,
        mDM,
        mDKprev,
        mDKcurr,
    )

    dM_view = dM_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)
    dkprev_view = dkprev_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)
    dkcurr_view = dkcurr_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)

    def _launch_dz0() -> None:
        compiled_dz0(mDOut_dz0, mC_dz0, mM_dz0, mDZ0)

    def _launch_du() -> None:
        compiled_du(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB_du_scratch,
            mDU_prev,
            mDB_prev_du_scratch,
            mDLp_du_scratch,
            mDMp_du_scratch,
            mDMc_du_scratch,
        )

    def _launch_db() -> None:
        compiled_db(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU_db_scratch,
            mDB,
            mDU_prev_db_scratch,
            mDB_prev,
            mDLp_db_scratch,
            mDMp_db_scratch,
            mDMc_db_scratch,
        )

    def _launch_dc() -> None:
        if compiled_dc_fast is not None:
            _ = Z0_blk_fast_keepalive
            compiled_dc_fast(
                mU,
                mB,
                mC,
                mM,
                mK,
                mDOut,
                mU_prev0,
                mB_prev0,
                mZ0_fast,
                mDLogp,
                mDC,
                mDR,
            )
        else:
            compiled_dc(
                mU,
                mB,
                mC,
                mM,
                mK,
                mDOut,
                mU_prev0,
                mB_prev0,
                mZ0,
                mDLogp,
                mDC,
                mDR,
            )

    def _launch_param() -> None:
        compiled_param(
            mM,
            mK,
            mDLp,
            mDMprev,
            mDMcurr,
            mDR_param,
            mDM,
            mDKprev,
            mDKcurr,
        )

    def launch_sequential() -> None:
        _launch_dz0()
        _launch_du()
        _launch_db()
        _launch_dc()
        _launch_param()

    stream_dz0 = None
    stream_du = None
    stream_db = None
    stream_dc = None
    ev_start = None
    ev_dz0_done = None
    ev_du_done = None
    ev_db_done = None
    ev_dc_done = None

    if return_launchers and enable_overlapped_launcher:
        stream_dz0 = torch.cuda.Stream(device=U.device)
        stream_du = torch.cuda.Stream(device=U.device)
        stream_db = torch.cuda.Stream(device=U.device)
        stream_dc = torch.cuda.Stream(device=U.device)
        ev_start = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_dz0_done = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_du_done = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_db_done = torch.cuda.Event(blocking=False, enable_timing=False)
        ev_dc_done = torch.cuda.Event(blocking=False, enable_timing=False)

    def launch_overlapped() -> None:
        if (
            stream_dz0 is None
            or stream_du is None
            or stream_db is None
            or stream_dc is None
        ):
            launch_sequential()
            return

        current = torch.cuda.current_stream(device=U.device)
        current.record_event(ev_start)

        stream_dz0.wait_event(ev_start)
        stream_du.wait_event(ev_start)
        stream_db.wait_event(ev_start)
        stream_dc.wait_event(ev_start)

        with torch.cuda.stream(stream_dz0):
            _launch_dz0()
            stream_dz0.record_event(ev_dz0_done)

        with torch.cuda.stream(stream_du):
            _launch_du()
            stream_du.record_event(ev_du_done)

        with torch.cuda.stream(stream_db):
            _launch_db()
            stream_db.record_event(ev_db_done)

        with torch.cuda.stream(stream_dc):
            _launch_dc()
            stream_dc.record_event(ev_dc_done)

        current.wait_event(ev_db_done)
        current.wait_event(ev_dc_done)
        _launch_param()

        current.wait_event(ev_dz0_done)
        current.wait_event(ev_du_done)

    base = (
        compiled_dz0,
        compiled_du,
        compiled_db,
        compiled_dc,
        compiled_param,
        dZ0_view,
        dU_view,
        dB_view,
        dU_prev_view,
        dB_prev_view,
        dlogp_view,
        dC_view,
        dR_view,
        dM_view,
        dkprev_view,
        dkcurr_view,
    )
    if return_launchers:
        return (*base, launch_sequential, launch_overlapped)
    return base


def chunk_scan_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
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
    """Thin public wrapper over the compiled chunk-scan backward kernel bundle."""
    (
        _compiled_dz0,
        _compiled_du,
        _compiled_db,
        _compiled_dc,
        _compiled_param,
        dZ0,
        dU,
        dB,
        dU_prev,
        dB_prev,
        _dlogp,
        dC,
        _dR,
        dM,
        dKprev,
        dKcurr,
        launch_sequential,
        _launch_overlapped,
    ) = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
        return_launchers=True,
        enable_overlapped_launcher=False,
    )
    launch_sequential()

    dU_public = _fold_chunk_boundary_carries(dU, dU_prev)
    dB_public = _fold_chunk_boundary_carries(dB, dB_prev)

    return (
        _public_from_chunked(dU_public, T=U.shape[2]),
        _public_from_param_scan(dM, T=U.shape[2]),
        _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
        _public_from_chunked(dB_public, T=U.shape[2]),
        _public_from_chunked(dC, T=U.shape[2]),
        dZ0.to(dtype=torch.float32).contiguous(),
        dB_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
        dU_prev[:, :, 0, :].to(dtype=torch.float32).contiguous(),
    )


__all__ = [
    "chunk_scan_bwd_cute",
    "compile_chunk_scan_bwd_kernels",
]
