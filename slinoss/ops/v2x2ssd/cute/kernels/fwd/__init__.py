"""CuTe forward kernels for the ``v2x2ssd`` staged pipeline."""

from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from typing import Callable

from slinoss.perf import note_cache_event, record_region

from .chunk_increment import ChunkIncrementFwdAmpere
from .chunk_scan import ChunkScanFwdAmpere
from .common import (
    _assumed_align,
    _choose_copy_bits_for_linear_tiles,
    _pad_m_identity,
    _pad_zero_time,
    _tc_input_dtype,
    _torch_to_cutlass_dtype,
)
from .state_passing import StatePassingFwdAmpere


_CHUNK_INCREMENT_CACHE: dict[tuple, object] = {}
_STATE_PASSING_CACHE: dict[tuple, object] = {}
_CHUNK_SCAN_CACHE: dict[tuple, object] = {}
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
            torch.zeros((batch_size, heads, P), device=device, dtype=dtype),
            torch.zeros((batch_size, heads, D), device=device, dtype=dtype),
        )
        _ZERO_PREV_CACHE[key] = cached
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
    )


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
    cache_key = _chunk_increment_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        chunk_size=L,
        has_prev=U_prev0 is not None,
    )

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

    U2 = U_tc.reshape(BH, T_pad, P).permute(2, 1, 0)
    B2 = B_tc.reshape(BH, T_pad, D).permute(2, 1, 0)
    M2 = M_f.reshape(BH, T_pad, 2).permute(2, 1, 0)
    Kprev2 = K_f[:, :, :, 0, :].reshape(BH, T_pad, 2).permute(2, 1, 0)
    Kcurr2 = K_f[:, :, :, 1, :].reshape(BH, T_pad, 2).permute(2, 1, 0)
    U_prev2 = U_prev.reshape(BH, P).permute(1, 0)
    B_prev2 = B_prev.reshape(BH, D).permute(1, 0)

    inc_chunk = torch.empty((BHC, P, D), device=U.device, dtype=torch.float32)
    m_chunk_chunk = torch.empty((BHC, 2), device=U.device, dtype=torch.float32)

    mU = from_dlpack(U2, assumed_align=16)
    mB = from_dlpack(B2, assumed_align=16)
    mM = from_dlpack(M2, assumed_align=_assumed_align(M2))
    mKprev = from_dlpack(Kprev2, assumed_align=_assumed_align(Kprev2))
    mKcurr = from_dlpack(Kcurr2, assumed_align=_assumed_align(Kcurr2))
    mU_prev = from_dlpack(U_prev2, assumed_align=16)
    mB_prev = from_dlpack(B_prev2, assumed_align=16)
    mInc = from_dlpack(inc_chunk.permute(1, 2, 0), assumed_align=16)
    mMchunk = from_dlpack(m_chunk_chunk.permute(1, 0), assumed_align=16)

    compiled = _CHUNK_INCREMENT_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cache.v2x2ssd.forward.chunk_increment", hit=False)
        cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
        kernel = ChunkIncrementFwdAmpere(cutlass_dtype, chunk_size=L)
        compiled = cute.compile(
            kernel, mU, mB, mM, mKprev, mKcurr, mU_prev, mB_prev, mInc, mMchunk
        )
        _CHUNK_INCREMENT_CACHE[cache_key] = compiled
    else:
        note_cache_event("cache.v2x2ssd.forward.chunk_increment", hit=True)

    def launch() -> None:
        with record_region("forward.v2x2ssd.chunk_increment.total"):
            compiled(
                mU,
                mB,
                mM,
                mKprev,
                mKcurr,
                mU_prev,
                mB_prev,
                mInc,
                mMchunk,
            )

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

    cache_key = _state_passing_key(
        device_index=(inc.device.index if inc.device.index is not None else -1),
        inc_shape=tuple(inc.shape),
        m_chunk_shape=tuple(m_chunk.shape),
        initial_shape=(None if initial_states is None else tuple(initial_states.shape)),
        num_threads=num_threads,
        vecs_per_thread=vecs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
    )

    align_in = max(inc.element_size(), copy_bits_in // 8)
    align_out = max(out_starts.element_size(), copy_bits_out // 8)

    inc_c = inc.contiguous()
    m_c = m_chunk.contiguous()
    init_c = initial_states.contiguous() if initial_states is not None else None

    mInc = from_dlpack(inc_c, assumed_align=align_in)
    mM = from_dlpack(m_c, assumed_align=16)
    mOutStarts = from_dlpack(out_starts, assumed_align=align_out)
    mOutFinal = from_dlpack(out_final, assumed_align=align_out)

    if init_c is None:
        mInit = from_dlpack(inc_c, assumed_align=align_in)
        has_init = False
    else:
        mInit = from_dlpack(init_c, assumed_align=max(init_c.element_size(), 4))
        has_init = True

    compiled = _STATE_PASSING_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cache.v2x2ssd.forward.state_passing", hit=False)
        kernel = StatePassingFwdAmpere(
            num_threads=num_threads,
            vecs_per_thread=vecs_per_thread,
            copy_bits_in=copy_bits_in,
            copy_bits_out=copy_bits_out,
            has_init=has_init,
        )
        compiled = cute.compile(kernel, mInc, mM, mOutStarts, mOutFinal, mInit)
        _STATE_PASSING_CACHE[cache_key] = compiled
    else:
        note_cache_event("cache.v2x2ssd.forward.state_passing", hit=True)

    def launch() -> None:
        with record_region("forward.v2x2ssd.state_passing.total"):
            compiled(mInc, mM, mOutStarts, mOutFinal, mInit)

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
    if m_block_size is None:
        m_block_size = L
    n_block_size = _resolve_chunk_scan_n_block_size(L, int(n_block_size))
    n_chunks = (T + L - 1) // L
    T_pad = n_chunks * L
    BH = Bsz * H
    BHC = BH * n_chunks

    if tuple(chunk_starts.shape) != (Bsz, H, n_chunks, P, D):
        raise ValueError(
            f"chunk_starts must be (B,H,C,P,D) ={(Bsz, H, n_chunks, P, D)}."
        )

    tc_dtype = _tc_input_dtype(U.dtype, compute_dtype)
    cache_key = _chunk_scan_key(
        device_index=(U.device.index if U.device.index is not None else -1),
        tc_dtype=tc_dtype,
        out_dtype=output_dtype,
        U_shape=tuple(U.shape),
        M_shape=tuple(M.shape),
        K_shape=tuple(K.shape),
        B_shape=tuple(B.shape),
        C_shape=tuple(C.shape),
        chunk_starts_shape=tuple(chunk_starts.shape),
        chunk_size=L,
        m_block_size=int(m_block_size),
        n_block_size=int(n_block_size),
        num_threads=int(num_threads),
        has_prev=B_prev is not None,
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

    U_blk = U_tc.reshape(Bsz, H, n_chunks, L, P).reshape(BHC, L, 1, P).contiguous()
    B_blk = B_tc.reshape(Bsz, H, n_chunks, L, D).reshape(BHC, L, 1, D).contiguous()
    C_blk = C_tc.reshape(Bsz, H, n_chunks, L, D).reshape(BHC, L, 1, D).contiguous()
    M_blk = M_f.reshape(Bsz, H, n_chunks, L, 2).reshape(BHC, L, 2).contiguous()
    K_blk = K_f.reshape(Bsz, H, n_chunks, L, 2, 2).reshape(BHC, L, 2, 2).contiguous()
    Z0_blk = chunk_starts.reshape(BHC, P, 1, D).contiguous()
    U_prev0_flat = U_prev0.reshape(BH, P).contiguous()
    B_prev0_flat = B_prev0.reshape(BH, D).contiguous()

    out_chunk = torch.empty((BHC, L, 1, P), device=U.device, dtype=output_dtype)
    out_pad = out_chunk.reshape(Bsz, H, n_chunks, L, 1, P).reshape(Bsz, H, T_pad, P)
    out_view = out_pad[:, :, :T, :]

    mU = from_dlpack(U_blk, assumed_align=16)
    mB = from_dlpack(B_blk, assumed_align=16)
    mC = from_dlpack(C_blk, assumed_align=16)
    mM = from_dlpack(M_blk, assumed_align=16)
    mK = from_dlpack(K_blk, assumed_align=16)
    mZ0 = from_dlpack(Z0_blk, assumed_align=16)
    mU_prev0 = from_dlpack(U_prev0_flat, assumed_align=16)
    mB_prev0 = from_dlpack(B_prev0_flat, assumed_align=16)
    mOut = from_dlpack(out_chunk, assumed_align=16)

    compiled = _CHUNK_SCAN_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cache.v2x2ssd.forward.chunk_scan", hit=False)
        in_cutlass_dtype = _torch_to_cutlass_dtype(tc_dtype)
        out_cutlass_dtype = _torch_to_cutlass_dtype(output_dtype)
        kernel = ChunkScanFwdAmpere(
            D=D,
            P=P,
            L=L,
            m_block_size=int(m_block_size),
            n_block_size=int(n_block_size),
            num_threads=int(num_threads),
        )
        if not kernel.can_implement(in_cutlass_dtype, out_cutlass_dtype):
            raise ValueError(
                "Configuration exceeds SM80 shared-memory capacity or violates alignment."
            )
        compiled = cute.compile(
            kernel, mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut
        )
        _CHUNK_SCAN_CACHE[cache_key] = compiled
    else:
        note_cache_event("cache.v2x2ssd.forward.chunk_scan", hit=True)

    def launch() -> None:
        with record_region("forward.v2x2ssd.chunk_scan.total"):
            compiled(mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut)

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


__all__ = [
    "chunk_increment_cute",
    "chunk_scan_cute",
    "compile_chunk_increment_kernel",
    "compile_chunk_scan_kernel",
    "compile_state_passing_kernel",
    "state_passing_cute",
]
