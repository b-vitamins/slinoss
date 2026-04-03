"""CuTe backward kernels for the ``v2x2ssd`` chunk-increment stage."""

from __future__ import annotations

import torch
import cutlass.cute as cute

from ...fwd.common import _make_fake_tensor_arg
from .boundary import ChunkIncrementBwdBoundaryAmpere
from .common import _assumed_align, _torch_to_cutlass_dtype
from .db import ChunkIncrementBwdDBAmpere
from .du import ChunkIncrementBwdDUAmpere
from .param_scan import ChunkIncrementBwdParamScanAmpere


_COMPILED_CACHE: dict[tuple, tuple[object, object, object, object, object]] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_ZERO_PREV_CACHE_LIMIT = 8


def _record_tensors_on_current_stream(*tensors: torch.Tensor | None) -> None:
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
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
        )
        _cache_set(_ZERO_PREV_CACHE, key, cached, limit=_ZERO_PREV_CACHE_LIMIT)
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
    alignments: tuple[int, ...],
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
        alignments,
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
    tensor: cute.Tensor,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
):
    shape, stride = spec
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_db_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = spec
    chunk_size, n_d_tiles = cfg

    u_spec = _make_tensor_spec((L, P, BHC), stride=(P, 1, L * P))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_inc_dp_spec = _make_tensor_spec((D, P, BHC), stride=(1, D, P * D))
    d_b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )

    @cute.jit
    def _db_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DIncDP_ptr: cute.Tensor,
        DB_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDIncDP = _make_tensor_from_spec(DIncDP_ptr, d_inc_dp_spec)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)

        kernel = ChunkIncrementBwdDBAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)

    return _db_host_wrapper


def _make_du_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = spec
    (chunk_size,) = cfg

    d_inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_u_spec = _make_tensor_spec((P, L, BHC), stride=(1, P, L * P))

    @cute.jit
    def _du_host_wrapper(
        DInc_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
    ):
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)

        kernel = ChunkIncrementBwdDUAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mDInc, mB, mM, mKprev, mKcurr, mDU)

    return _du_host_wrapper


def _make_boundary_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = spec
    (chunk_size,) = cfg

    d_inc_boundary_spec = _make_tensor_spec((BHC, P, D), stride=(P * D, D, 1))
    prev_b_spec = _make_tensor_spec((D, BHC), stride=(1, D))
    prev_u_spec = _make_tensor_spec((P, BHC), stride=(1, P))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_mp0_spec = _make_tensor_spec((2, BHC), stride=(BHC, 1))

    @cute.jit
    def _boundary_host_wrapper(
        DInc_ptr: cute.Tensor,
        BPrev_ptr: cute.Tensor,
        UPrev_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DBPrev_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
    ):
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_boundary_spec)
        mBPrev = _make_tensor_from_spec(BPrev_ptr, prev_b_spec)
        mUPrev = _make_tensor_from_spec(UPrev_ptr, prev_u_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, prev_u_spec)
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, prev_b_spec)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, d_mp0_spec)

        kernel = ChunkIncrementBwdBoundaryAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        kernel(mDInc, mBPrev, mUPrev, mM, mKprev, mDUPrev, mDBPrev, mDMp0)

    return _boundary_host_wrapper


def _make_param_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
):
    L, BHC = spec
    (chunk_size, n_d_tiles) = cfg

    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )
    d_mp0_spec = _make_tensor_spec((2, BHC), stride=(BHC, 1))
    d_mchunk_spec = _make_tensor_spec((2, BHC), stride=(1, 2))
    d_param_spec = _make_tensor_spec((2, L, BHC), stride=(L * BHC, BHC, 1))

    @cute.jit
    def _param_host_wrapper(
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
        DMchunk_ptr: cute.Tensor,
        DM_ptr: cute.Tensor,
        DKprev_ptr: cute.Tensor,
        DKcurr_ptr: cute.Tensor,
    ):
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, d_mp0_spec)
        mDMchunk = _make_tensor_from_spec(DMchunk_ptr, d_mchunk_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_param_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, d_param_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, d_param_spec)

        kernel = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            nDtiles=n_d_tiles,
        )
        kernel(
            mM,
            mKprev,
            mKcurr,
            mDMsumPart,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
        )

    return _param_host_wrapper


def _make_stage_host_wrapper(
    *,
    spec: tuple[int, ...],
    cfg: tuple[int, ...],
    cutlass_dtype,
):
    L, P, D, BHC = spec
    chunk_size, n_d_tiles = cfg

    u_spec = _make_tensor_spec((L, P, BHC), stride=(P, 1, L * P))
    b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    m_spec = _make_tensor_spec((2, L, BHC), stride=(1, 2, L * 2))
    d_inc_spec = _make_tensor_spec((P, D, BHC), stride=(D, 1, P * D))
    d_inc_dp_spec = _make_tensor_spec((D, P, BHC), stride=(1, D, P * D))
    d_inc_boundary_spec = _make_tensor_spec((BHC, P, D), stride=(P * D, D, 1))
    prev_b_spec = _make_tensor_spec((D, BHC), stride=(1, D))
    prev_u_spec = _make_tensor_spec((P, BHC), stride=(1, P))
    d_b_spec = _make_tensor_spec((L, D, BHC), stride=(D, 1, L * D))
    d_u_spec = _make_tensor_spec((P, L, BHC), stride=(1, P, L * P))
    d_msum_part_spec = _make_tensor_spec(
        (2, L, n_d_tiles, BHC),
        stride=(L * n_d_tiles * BHC, n_d_tiles * BHC, BHC, 1),
    )
    d_mp0_spec = _make_tensor_spec((2, BHC), stride=(BHC, 1))
    d_mchunk_spec = _make_tensor_spec((2, BHC), stride=(1, 2))
    d_param_spec = _make_tensor_spec((2, L, BHC), stride=(L * BHC, BHC, 1))

    @cute.jit
    def _stage_host_wrapper(
        U_ptr: cute.Tensor,
        B_ptr: cute.Tensor,
        M_ptr: cute.Tensor,
        Kprev_ptr: cute.Tensor,
        Kcurr_ptr: cute.Tensor,
        DInc_ptr: cute.Tensor,
        BPrev_ptr: cute.Tensor,
        UPrev_ptr: cute.Tensor,
        DB_ptr: cute.Tensor,
        DU_ptr: cute.Tensor,
        DBPrev_ptr: cute.Tensor,
        DUPrev_ptr: cute.Tensor,
        DMsumPart_ptr: cute.Tensor,
        DMp0_ptr: cute.Tensor,
        DMchunk_ptr: cute.Tensor,
        DM_ptr: cute.Tensor,
        DKprev_ptr: cute.Tensor,
        DKcurr_ptr: cute.Tensor,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mKprev = _make_tensor_from_spec(Kprev_ptr, m_spec)
        mKcurr = _make_tensor_from_spec(Kcurr_ptr, m_spec)
        mDInc = _make_tensor_from_spec(DInc_ptr, d_inc_spec)
        mDIncDP = _make_tensor_from_spec(DInc_ptr, d_inc_dp_spec)
        mDIncBoundary = _make_tensor_from_spec(DInc_ptr, d_inc_boundary_spec)
        mBPrev = _make_tensor_from_spec(BPrev_ptr, prev_b_spec)
        mUPrev = _make_tensor_from_spec(UPrev_ptr, prev_u_spec)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, prev_b_spec)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, prev_u_spec)
        mDMsumPart = _make_tensor_from_spec(DMsumPart_ptr, d_msum_part_spec)
        mDMp0 = _make_tensor_from_spec(DMp0_ptr, d_mp0_spec)
        mDMchunk = _make_tensor_from_spec(DMchunk_ptr, d_mchunk_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_param_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, d_param_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, d_param_spec)

        k_db = ChunkIncrementBwdDBAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_du = ChunkIncrementBwdDUAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_boundary = ChunkIncrementBwdBoundaryAmpere(
            cutlass_dtype,
            chunk_size=chunk_size,
            D=D,
            P=P,
        )
        k_param = ChunkIncrementBwdParamScanAmpere(
            chunk_size=chunk_size,
            nDtiles=n_d_tiles,
        )

        k_db(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)
        k_du(mDInc, mB, mM, mKprev, mKcurr, mDU)
        k_boundary(mDIncBoundary, mBPrev, mUPrev, mM, mKprev, mDUPrev, mDBPrev, mDMp0)
        k_param(mM, mKprev, mKcurr, mDMsumPart, mDMp0, mDMchunk, mDM, mDKprev, mDKcurr)

    return _stage_host_wrapper


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
    K_chunks = K_f.reshape(BH, n_chunks, L, 2, 2)
    Kprev_blk = K_chunks[..., 0, :].reshape(BHC, L, 2).contiguous()
    Kcurr_blk = K_chunks[..., 1, :].reshape(BHC, L, 2).contiguous()

    U_prev_chunks = torch.empty((BH, n_chunks, P), device=U.device, dtype=tc_dtype)
    B_prev_chunks = torch.empty((BH, n_chunks, D), device=U.device, dtype=tc_dtype)
    U_prev_chunks[:, 0, :] = U_prev0.reshape(BH, P)
    B_prev_chunks[:, 0, :] = B_prev0.reshape(BH, D)
    if n_chunks > 1:
        U_prev_chunks[:, 1:, :] = U_chunks[:, :-1, -1, :]
        B_prev_chunks[:, 1:, :] = B_chunks[:, :-1, -1, :]

    k_db = ChunkIncrementBwdDBAmpere(cutlass_dtype, chunk_size=L, D=D, P=P)
    nDtiles = (D + k_db.bN - 1) // k_db.bN

    dB = torch.empty((BHC, L, D), device=U.device, dtype=tc_dtype)
    dU = torch.empty((BHC, L, P), device=U.device, dtype=tc_dtype)
    dB_prev = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dU_prev = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dMsum_part = torch.empty((2, L, nDtiles, BHC), device=U.device, dtype=torch.float32)
    dMp0 = torch.empty((2, BHC), device=U.device, dtype=torch.float32)
    dM_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)
    dKprev_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)
    dKcurr_out = torch.empty((2, L, BHC), device=U.device, dtype=torch.float32)

    db_runtime_args = (
        U_tc,
        B_tc,
        M_f,
        Kprev_blk,
        Kcurr_blk,
        d_inc_tc,
        dB,
        dMsum_part,
    )
    du_runtime_args = (
        d_inc_tc,
        B_tc,
        M_f,
        Kprev_blk,
        Kcurr_blk,
        dU,
    )
    boundary_runtime_args = (
        d_inc_tc,
        B_prev_chunks,
        U_prev_chunks,
        M_f,
        Kprev_blk,
        dU_prev,
        dB_prev,
        dMp0,
    )
    param_runtime_args = (
        M_f,
        Kprev_blk,
        Kcurr_blk,
        dMsum_part,
        dMp0,
        d_m_chunk_f,
        dM_out,
        dKprev_out,
        dKcurr_out,
    )
    stage_runtime_args = (
        U_tc,
        B_tc,
        M_f,
        Kprev_blk,
        Kcurr_blk,
        d_inc_tc,
        B_prev_chunks,
        U_prev_chunks,
        dB,
        dU,
        dB_prev,
        dU_prev,
        dMsum_part,
        dMp0,
        d_m_chunk_f,
        dM_out,
        dKprev_out,
        dKcurr_out,
    )
    db_alignments = tuple(_assumed_align(tensor) for tensor in db_runtime_args)
    du_alignments = tuple(_assumed_align(tensor) for tensor in du_runtime_args)
    boundary_alignments = tuple(
        _assumed_align(tensor) for tensor in boundary_runtime_args
    )
    param_alignments = tuple(_assumed_align(tensor) for tensor in param_runtime_args)
    stage_alignments = tuple(_assumed_align(tensor) for tensor in stage_runtime_args)
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
        alignments=stage_alignments,
    )

    cached = _COMPILED_CACHE.get(cache_key)
    if cached is None:
        db_wrapper = _make_db_host_wrapper(
            spec=(L, P, D, BHC),
            cfg=(L, nDtiles),
            cutlass_dtype=cutlass_dtype,
        )
        du_wrapper = _make_du_host_wrapper(
            spec=(L, P, D, BHC),
            cfg=(L,),
            cutlass_dtype=cutlass_dtype,
        )
        boundary_wrapper = _make_boundary_host_wrapper(
            spec=(L, P, D, BHC),
            cfg=(L,),
            cutlass_dtype=cutlass_dtype,
        )
        param_wrapper = _make_param_host_wrapper(
            spec=(L, BHC),
            cfg=(L, nDtiles),
        )
        stage_wrapper = _make_stage_host_wrapper(
            spec=(L, P, D, BHC),
            cfg=(L, nDtiles),
            cutlass_dtype=cutlass_dtype,
        )
        db_compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(db_runtime_args, db_alignments, strict=True)
        )
        du_compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(du_runtime_args, du_alignments, strict=True)
        )
        boundary_compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(
                boundary_runtime_args, boundary_alignments, strict=True
            )
        )
        param_compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(param_runtime_args, param_alignments, strict=True)
        )
        stage_compile_args = tuple(
            _make_fake_tensor_arg(tensor, align=align)
            for tensor, align in zip(stage_runtime_args, stage_alignments, strict=True)
        )
        compiled_db = cute.compile(
            db_wrapper,
            *db_compile_args,
            options="--enable-tvm-ffi",
        )
        compiled_du = cute.compile(
            du_wrapper,
            *du_compile_args,
            options="--enable-tvm-ffi",
        )
        compiled_boundary = cute.compile(
            boundary_wrapper,
            *boundary_compile_args,
            options="--enable-tvm-ffi",
        )
        compiled_param = cute.compile(
            param_wrapper,
            *param_compile_args,
            options="--enable-tvm-ffi",
        )
        compiled_stage = cute.compile(
            stage_wrapper,
            *stage_compile_args,
            options="--enable-tvm-ffi",
        )
        cached = (
            compiled_db,
            compiled_du,
            compiled_boundary,
            compiled_param,
            compiled_stage,
        )
        _COMPILED_CACHE[cache_key] = cached
    else:
        compiled_db, compiled_du, compiled_boundary, compiled_param, compiled_stage = (
            cached
        )

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

    def launch() -> None:
        compiled_stage(*stage_runtime_args)
        _record_tensors_on_current_stream(*stage_runtime_args)

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
        return (*base, launch)
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
        launch,
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
    launch()

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
