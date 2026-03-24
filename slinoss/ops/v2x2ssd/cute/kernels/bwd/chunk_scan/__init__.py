"""CuTe backward kernels for the ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from .db import ChunkScanBwdDBAmpere
from .dcdr import ChunkScanBwdDCDRAmpere
from .dlp import ChunkScanBwdDLPAmpere
from .du import ChunkScanBwdDUAmpere
from .dz0 import ChunkScanBwdDZ0Ampere
from .param_scan import ChunkScanBwdParamScanAmpere


_COMPILED_CACHE: dict[
    tuple, tuple[object, object, object, object, object, object | None]
] = {}
_OVERLAP_RESOURCES: dict[
    int,
    tuple[
        torch.cuda.Stream,
        torch.cuda.Stream,
        torch.cuda.Stream,
        torch.cuda.Stream,
        torch.cuda.Event,
        torch.cuda.Event,
        torch.cuda.Event,
        torch.cuda.Event,
        torch.cuda.Event,
    ],
] = {}
_ZERO_PREV_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


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


def _get_overlap_resources(
    device: torch.device,
) -> tuple[
    torch.cuda.Stream,
    torch.cuda.Stream,
    torch.cuda.Stream,
    torch.cuda.Stream,
    torch.cuda.Event,
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
            torch.cuda.Stream(device=device),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
            torch.cuda.Event(blocking=False, enable_timing=False),
        )
        _OVERLAP_RESOURCES[device_index] = cached
    return cached


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
        _ZERO_PREV_CACHE[key] = cached
    return cached


def _chunk_scan_device_label(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    return f"{props.name} (sm_{props.major}{props.minor})"


def _validate_dcdr_support(
    *,
    tc_dtype: torch.dtype,
    chunk_size: int,
    D: int,
    P: int,
    num_threads: int,
    device_index: int,
) -> None:
    kernel = ChunkScanBwdDCDRAmpere(
        _torch_to_cutlass_dtype(tc_dtype),
        chunk_size=chunk_size,
        D=D,
        P=P,
        num_threads=num_threads,
    )
    info = kernel.support_info(
        _torch_to_cutlass_dtype(tc_dtype),
        device_index=device_index,
    )
    if info.supported:
        return

    device_label = _chunk_scan_device_label(device_index)
    raise ValueError(
        f"No supported chunk_scan backward dcdr kernel fits {device_label} for "
        f"(chunk_size={chunk_size}, D={D}, P={P}, num_threads={num_threads}). "
        f"The current low-SMEM variant needs {info.required_smem_bytes}B > "
        f"{info.smem_capacity_bytes}B shared memory."
    )


def _assumed_align(
    t: torch.Tensor,
    candidates_bytes: tuple[int, ...] = (16, 8, 4),
) -> int:
    elem_align = max(1, t.element_size())
    ptr = int(t.data_ptr())
    for align in candidates_bytes:
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


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


def _make_tensor_spec_from_tensor(
    t: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _make_tensor_spec(
        tuple(map(int, t.shape)), stride=tuple(map(int, t.stride()))
    )


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


def _public_from_param_scan(
    x: torch.Tensor,
    *,
    T: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    B, H, C, S, L, F = map(int, x.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    out = x[:, :, :, 0, :, :].reshape(B, H, C * L, F)[:, :, :T, :]
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
    B, H, C, S, L, F = map(int, dKprev.shape)
    if S != 1:
        raise ValueError("Only n_splits=1 is supported by the public wrapper.")
    dK = torch.stack((dKprev[:, :, :, 0, :, :], dKcurr[:, :, :, 0, :, :]), dim=4)
    return (
        dK.reshape(B, H, C * L, 2, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()
    )


def _compiled_key(
    *,
    device_index: int,
    tc_dtype: torch.dtype,
    U_shape: tuple[int, ...],
    B_shape: tuple[int, ...],
    C_shape: tuple[int, ...],
    M_shape: tuple[int, ...],
    K_shape: tuple[int, ...],
    chunk_starts_shape: tuple[int, ...],
    d_out_shape: tuple[int, ...],
    chunk_size: int,
    has_prev: bool,
    alignments: tuple[int, ...],
    num_threads_du: int,
    num_threads_db: int,
    num_threads_dc: int,
    num_threads_param: int,
) -> tuple:
    return (
        "chunk_scan_bwd",
        device_index,
        tc_dtype,
        U_shape,
        B_shape,
        C_shape,
        M_shape,
        K_shape,
        chunk_starts_shape,
        d_out_shape,
        int(chunk_size),
        has_prev,
        alignments,
        int(num_threads_du),
        int(num_threads_db),
        int(num_threads_dc),
        int(num_threads_param),
    )


def _make_dz0_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    (chunk_size,) = cfg
    d_out_spec, c_spec, m_spec, dz0_spec = spec

    @cute.jit
    def _dz0_host_wrapper(
        DOut_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        DZ0_ptr: cute.Pointer,
    ):
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mDZ0 = _make_tensor_from_spec(DZ0_ptr, dz0_spec)

        kernel = ChunkScanBwdDZ0Ampere(DOut_ptr.value_type, chunk_size=chunk_size)
        kernel(mDOut, mC, mM, mDZ0)

    return _dz0_host_wrapper


def _make_du_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    chunk_size, D, P, num_threads = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        d_u_spec,
        d_b_scratch_spec,
        d_u_prev_spec,
        d_b_prev_scratch_spec,
        dlp_spec,
        dmp_spec,
        dmc_spec,
    ) = spec

    @cute.jit
    def _du_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        DOut_ptr: cute.Pointer,
        UPrev0_ptr: cute.Pointer,
        BPrev0_ptr: cute.Pointer,
        DU_ptr: cute.Pointer,
        DBScratch_ptr: cute.Pointer,
        DUPrev_ptr: cute.Pointer,
        DBPrevScratch_ptr: cute.Pointer,
        DLp_ptr: cute.Pointer,
        DMp_ptr: cute.Pointer,
        DMc_ptr: cute.Pointer,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mDU = _make_tensor_from_spec(DU_ptr, d_u_spec)
        mDBScratch = _make_tensor_from_spec(DBScratch_ptr, d_b_scratch_spec)
        mDUPrev = _make_tensor_from_spec(DUPrev_ptr, d_u_prev_spec)
        mDBPrevScratch = _make_tensor_from_spec(
            DBPrevScratch_ptr, d_b_prev_scratch_spec
        )
        mDLp = _make_tensor_from_spec(DLp_ptr, dlp_spec)
        mDMp = _make_tensor_from_spec(DMp_ptr, dmp_spec)
        mDMc = _make_tensor_from_spec(DMc_ptr, dmc_spec)

        kernel = ChunkScanBwdDUAmpere(
            U_ptr.value_type,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mDU,
            mDBScratch,
            mDUPrev,
            mDBPrevScratch,
            mDLp,
            mDMp,
            mDMc,
        )

    return _du_host_wrapper


def _make_db_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    chunk_size, D, P, num_threads = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        d_u_scratch_spec,
        d_b_spec,
        d_u_prev_scratch_spec,
        d_b_prev_spec,
        dlp_spec,
        dmp_spec,
        dmc_spec,
    ) = spec

    @cute.jit
    def _db_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        DOut_ptr: cute.Pointer,
        UPrev0_ptr: cute.Pointer,
        BPrev0_ptr: cute.Pointer,
        DUScratch_ptr: cute.Pointer,
        DB_ptr: cute.Pointer,
        DUPrevScratch_ptr: cute.Pointer,
        DBPrev_ptr: cute.Pointer,
        DLp_ptr: cute.Pointer,
        DMp_ptr: cute.Pointer,
        DMc_ptr: cute.Pointer,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mDUScratch = _make_tensor_from_spec(DUScratch_ptr, d_u_scratch_spec)
        mDB = _make_tensor_from_spec(DB_ptr, d_b_spec)
        mDUPrevScratch = _make_tensor_from_spec(
            DUPrevScratch_ptr, d_u_prev_scratch_spec
        )
        mDBPrev = _make_tensor_from_spec(DBPrev_ptr, d_b_prev_spec)
        mDLp = _make_tensor_from_spec(DLp_ptr, dlp_spec)
        mDMp = _make_tensor_from_spec(DMp_ptr, dmp_spec)
        mDMc = _make_tensor_from_spec(DMc_ptr, dmc_spec)

        kernel = ChunkScanBwdDBAmpere(
            U_ptr.value_type,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mDUScratch,
            mDB,
            mDUPrevScratch,
            mDBPrev,
            mDLp,
            mDMp,
            mDMc,
        )

    return _db_host_wrapper


def _make_dc_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    chunk_size, D, P, num_threads = cfg
    (
        u_spec,
        b_spec,
        c_spec,
        m_spec,
        k_spec,
        d_out_spec,
        u_prev0_spec,
        b_prev0_spec,
        z0_spec,
        dlp_spec,
        d_c_spec,
        d_r_spec,
    ) = spec

    @cute.jit
    def _dc_host_wrapper(
        U_ptr: cute.Pointer,
        B_ptr: cute.Pointer,
        C_ptr: cute.Pointer,
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        DOut_ptr: cute.Pointer,
        UPrev0_ptr: cute.Pointer,
        BPrev0_ptr: cute.Pointer,
        Z0_ptr: cute.Pointer,
        DLp_ptr: cute.Pointer,
        DC_ptr: cute.Pointer,
        DR_ptr: cute.Pointer,
    ):
        mU = _make_tensor_from_spec(U_ptr, u_spec)
        mB = _make_tensor_from_spec(B_ptr, b_spec)
        mC = _make_tensor_from_spec(C_ptr, c_spec)
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDOut = _make_tensor_from_spec(DOut_ptr, d_out_spec)
        mUPrev0 = _make_tensor_from_spec(UPrev0_ptr, u_prev0_spec)
        mBPrev0 = _make_tensor_from_spec(BPrev0_ptr, b_prev0_spec)
        mZ0 = _make_tensor_from_spec(Z0_ptr, z0_spec)
        mDLp = _make_tensor_from_spec(DLp_ptr, dlp_spec)
        mDC = _make_tensor_from_spec(DC_ptr, d_c_spec)
        mDR = _make_tensor_from_spec(DR_ptr, d_r_spec)

        kernel = ChunkScanBwdDLPAmpere(
            U_ptr.value_type,
            chunk_size=chunk_size,
            D=D,
            P=P,
            num_threads=num_threads,
        )
        kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mUPrev0,
            mBPrev0,
            mZ0,
            mDLp,
            mDC,
            mDR,
        )

    return _dc_host_wrapper


def _make_param_host_wrapper(
    *,
    spec: tuple[tuple[int, ...], ...],
    cfg: tuple[int, ...],
):
    chunk_size, num_threads = cfg
    (
        m_spec,
        k_spec,
        dlp_spec,
        dmp_spec,
        dmc_spec,
        d_r_spec,
        d_m_spec,
        d_kprev_spec,
        d_kcurr_spec,
    ) = spec

    @cute.jit
    def _param_host_wrapper(
        M_ptr: cute.Pointer,
        K_ptr: cute.Pointer,
        DLp_ptr: cute.Pointer,
        DMp_ptr: cute.Pointer,
        DMc_ptr: cute.Pointer,
        DR_ptr: cute.Pointer,
        DM_ptr: cute.Pointer,
        DKprev_ptr: cute.Pointer,
        DKcurr_ptr: cute.Pointer,
    ):
        mM = _make_tensor_from_spec(M_ptr, m_spec)
        mK = _make_tensor_from_spec(K_ptr, k_spec)
        mDLp = _make_tensor_from_spec(DLp_ptr, dlp_spec)
        mDMp = _make_tensor_from_spec(DMp_ptr, dmp_spec)
        mDMc = _make_tensor_from_spec(DMc_ptr, dmc_spec)
        mDR = _make_tensor_from_spec(DR_ptr, d_r_spec)
        mDM = _make_tensor_from_spec(DM_ptr, d_m_spec)
        mDKprev = _make_tensor_from_spec(DKprev_ptr, d_kprev_spec)
        mDKcurr = _make_tensor_from_spec(DKcurr_ptr, d_kcurr_spec)

        kernel = ChunkScanBwdParamScanAmpere(
            chunk_size=chunk_size,
            num_threads=num_threads,
        )
        kernel(mM, mK, mDLp, mDMp, mDMc, mDR, mDM, mDKprev, mDKcurr)

    return _param_host_wrapper


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
    device_index = (
        U.device.index if U.device.index is not None else torch.cuda.current_device()
    )
    _validate_dcdr_support(
        tc_dtype=tc_dtype,
        chunk_size=L,
        D=D,
        P=P,
        num_threads=num_threads_dc,
        device_index=int(device_index),
    )

    U_tc = _pad_zero_time(U, T_pad=T_pad, dtype=tc_dtype)
    B_tc = _pad_zero_time(B, T_pad=T_pad, dtype=tc_dtype)
    C_tc = _pad_zero_time(C, T_pad=T_pad, dtype=tc_dtype)
    d_out_tc = _pad_zero_time(d_out, T_pad=T_pad, dtype=tc_dtype)
    M_f = _pad_m_identity(M, T_pad=T_pad)
    K_f = _pad_zero_time(K, T_pad=T_pad, dtype=torch.float32)
    chunk_starts_f = chunk_starts.to(dtype=torch.float32).contiguous()

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

    BH = Bsz * H
    BHC = BH * n_chunks

    dOut2 = d_out_tc.reshape(BH, T_pad, P).permute(2, 1, 0)
    C2 = C_tc.reshape(BH, T_pad, D).permute(2, 1, 0)
    M2 = M_f.reshape(BH, T_pad, 2).permute(2, 1, 0)

    dZ0 = torch.empty((BHC, P, D), device=U.device, dtype=torch.float32)
    dZ0_perm = dZ0.permute(1, 2, 0)
    compiled_dz0 = None
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

    dU = torch.empty_like(U_blk)
    dU_prev = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dB_du_scratch = torch.empty_like(B_blk)
    dB_prev_du_scratch = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dlp_du_scratch = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dMp_du_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    dMc_du_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    compiled_du = None

    dB = torch.empty_like(B_blk)
    dB_prev = torch.empty((BHC, D), device=U.device, dtype=tc_dtype)
    dU_db_scratch = torch.empty_like(U_blk)
    dU_prev_db_scratch = torch.empty((BHC, P), device=U.device, dtype=tc_dtype)
    dlp_db_scratch = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dMp_db_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    dMc_db_scratch = torch.empty((BHC, L, 2), device=U.device, dtype=torch.float32)
    compiled_db = None

    dU_view = dU.reshape(Bsz, H, n_chunks, L, P)
    dB_view = dB.reshape(Bsz, H, n_chunks, L, D)
    dU_prev_view = dU_prev.reshape(Bsz, H, n_chunks, P)
    dB_prev_view = dB_prev.reshape(Bsz, H, n_chunks, D)

    dlogp = torch.empty((BHC, L), device=U.device, dtype=torch.float32)
    dC = torch.empty_like(C_blk)
    dR = torch.empty((BHC, L, 4), device=U.device, dtype=torch.float32)
    compiled_dc = None
    compiled_dc_fast = None

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
    compiled_param = None

    dM_view = dM_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)
    dkprev_view = dkprev_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)
    dkcurr_view = dkcurr_out.reshape(Bsz, H, n_chunks, n_splits, L, 2)

    dz0_args, dz0_alignments = _make_ptr_args(dOut2, C2, M2, dZ0_perm)
    du_args, du_alignments = _make_ptr_args(
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        dU,
        dB_du_scratch,
        dU_prev,
        dB_prev_du_scratch,
        dlp_du_scratch,
        dMp_du_scratch,
        dMc_du_scratch,
    )
    db_args, db_alignments = _make_ptr_args(
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        dU_db_scratch,
        dB,
        dU_prev_db_scratch,
        dB_prev,
        dlp_db_scratch,
        dMp_db_scratch,
        dMc_db_scratch,
    )
    dc_args, dc_alignments = _make_ptr_args(
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        U_prev0_flat,
        B_prev0_flat,
        Z0_blk,
        dlogp,
        dC,
        dR,
    )
    param_args, param_alignments = _make_ptr_args(
        M_blk,
        K_blk,
        dlp_blk,
        dMp_blk,
        dMc_blk,
        dR_blk,
        dM_out,
        dkprev_out,
        dkcurr_out,
    )
    alignments = (
        dz0_alignments
        + du_alignments
        + db_alignments
        + dc_alignments
        + param_alignments
    )
    keepalive = (
        U_tc,
        B_tc,
        C_tc,
        d_out_tc,
        M_f,
        K_f,
        chunk_starts_f,
        U_prev0,
        B_prev0,
        dOut2,
        C2,
        M2,
        dZ0_perm,
        U_blk,
        B_blk,
        C_blk,
        M_blk,
        K_blk,
        dOut_blk,
        Z0_blk,
        U_prev0_flat,
        B_prev0_flat,
        dlp_blk,
        dMp_blk,
        dMc_blk,
        dR_blk,
        dB_du_scratch,
        dB_prev_du_scratch,
        dlp_du_scratch,
        dMp_du_scratch,
        dMc_du_scratch,
        dU_db_scratch,
        dU_prev_db_scratch,
        dlp_db_scratch,
        dMp_db_scratch,
        dMc_db_scratch,
    )
    cache_key = _compiled_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        U_shape=tuple(U.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        d_out_shape=tuple(d_out.shape),
        chunk_size=L,
        has_prev=B_prev is not None,
        alignments=alignments,
        num_threads_du=num_threads_du,
        num_threads_db=num_threads_db,
        num_threads_dc=num_threads_dc,
        num_threads_param=num_threads_param,
    )

    use_compiled_cache = not return_launchers
    cached = _COMPILED_CACHE.get(cache_key) if use_compiled_cache else None
    if cached is None:
        dz0_wrapper = _make_dz0_host_wrapper(
            spec=(
                _make_tensor_spec_from_tensor(dOut2),
                _make_tensor_spec_from_tensor(C2),
                _make_tensor_spec_from_tensor(M2),
                _make_tensor_spec_from_tensor(dZ0_perm),
            ),
            cfg=(L,),
        )
        du_wrapper = _make_du_host_wrapper(
            spec=tuple(
                _make_tensor_spec_from_tensor(t)
                for t in (
                    U_blk,
                    B_blk,
                    C_blk,
                    M_blk,
                    K_blk,
                    dOut_blk,
                    U_prev0_flat,
                    B_prev0_flat,
                    dU,
                    dB_du_scratch,
                    dU_prev,
                    dB_prev_du_scratch,
                    dlp_du_scratch,
                    dMp_du_scratch,
                    dMc_du_scratch,
                )
            ),
            cfg=(L, D, P, num_threads_du),
        )
        db_wrapper = _make_db_host_wrapper(
            spec=tuple(
                _make_tensor_spec_from_tensor(t)
                for t in (
                    U_blk,
                    B_blk,
                    C_blk,
                    M_blk,
                    K_blk,
                    dOut_blk,
                    U_prev0_flat,
                    B_prev0_flat,
                    dU_db_scratch,
                    dB,
                    dU_prev_db_scratch,
                    dB_prev,
                    dlp_db_scratch,
                    dMp_db_scratch,
                    dMc_db_scratch,
                )
            ),
            cfg=(L, D, P, num_threads_db),
        )
        dc_wrapper = _make_dc_host_wrapper(
            spec=tuple(
                _make_tensor_spec_from_tensor(t)
                for t in (
                    U_blk,
                    B_blk,
                    C_blk,
                    M_blk,
                    K_blk,
                    dOut_blk,
                    U_prev0_flat,
                    B_prev0_flat,
                    Z0_blk,
                    dlogp,
                    dC,
                    dR,
                )
            ),
            cfg=(L, D, P, num_threads_dc),
        )
        param_wrapper = _make_param_host_wrapper(
            spec=tuple(
                _make_tensor_spec_from_tensor(t)
                for t in (
                    M_blk,
                    K_blk,
                    dlp_blk,
                    dMp_blk,
                    dMc_blk,
                    dR_blk,
                    dM_out,
                    dkprev_out,
                    dkcurr_out,
                )
            ),
            cfg=(L, num_threads_param),
        )
        compiled_dz0 = cute.compile(dz0_wrapper, *dz0_args)
        compiled_du = cute.compile(du_wrapper, *du_args)
        compiled_db = cute.compile(db_wrapper, *db_args)
        compiled_dc = cute.compile(dc_wrapper, *dc_args)
        compiled_param = cute.compile(param_wrapper, *param_args)
        cached = (
            compiled_dz0,
            compiled_du,
            compiled_db,
            compiled_dc,
            compiled_param,
            compiled_dc_fast,
        )
        if use_compiled_cache:
            _COMPILED_CACHE[cache_key] = cached
    else:
        (
            compiled_dz0,
            compiled_du,
            compiled_db,
            compiled_dc,
            compiled_param,
            compiled_dc_fast,
        ) = cached

    def _launch_dz0() -> None:
        _ = keepalive
        compiled_dz0(*dz0_args)

    def _launch_du() -> None:
        _ = keepalive
        compiled_du(*du_args)

    def _launch_db() -> None:
        _ = keepalive
        compiled_db(*db_args)

    def _launch_dc() -> None:
        _ = keepalive
        compiled_dc(*dc_args)

    def _launch_param() -> None:
        _ = keepalive
        compiled_param(*param_args)

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
        (
            stream_dz0,
            stream_du,
            stream_db,
            stream_dc,
            ev_start,
            ev_dz0_done,
            ev_du_done,
            ev_db_done,
            ev_dc_done,
        ) = _get_overlap_resources(U.device)

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
    return_prev_grads: bool = True,
) -> tuple[torch.Tensor, ...]:
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
        _launch_sequential,
        launch_overlapped,
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
    )
    launch_overlapped()

    dU_public = _fold_chunk_boundary_carries(dU, dU_prev)
    dB_public = _fold_chunk_boundary_carries(dB, dB_prev)

    if not return_prev_grads:
        return (
            _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
            _public_from_param_scan(dM, T=U.shape[2]),
            _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
            _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
            _public_from_chunked(dC, T=U.shape[2], dtype=C.dtype),
            dZ0.to(dtype=torch.float32).contiguous(),
        )

    return (
        _public_from_chunked(dU_public, T=U.shape[2], dtype=U.dtype),
        _public_from_param_scan(dM, T=U.shape[2]),
        _public_dk_from_parts(dKprev, dKcurr, T=U.shape[2]),
        _public_from_chunked(dB_public, T=U.shape[2], dtype=B.dtype),
        _public_from_chunked(dC, T=U.shape[2], dtype=C.dtype),
        dZ0.to(dtype=torch.float32).contiguous(),
        dB_prev[:, :, 0, :].to(dtype=B.dtype).contiguous(),
        dU_prev[:, :, 0, :].to(dtype=U.dtype).contiguous(),
    )


__all__ = [
    "chunk_scan_bwd_cute",
    "compile_chunk_scan_bwd_kernels",
]
