"""Exact CuTe helpers for the ``chunk_increment`` backward tail.

Logical contract
----------------
- ``suffix`` / ``k_prev`` / ``k_curr`` / ``M``: ``(BHC, L, 2)``
- ``B`` / ``d_alpha``: ``(BHC, L, D)`` with interleaved complex pairs
- ``b_prev0`` / ``d_boundary``: ``(BHC, D)``
- ``d_m_chunk``: ``(BHC, 2)``

These kernels keep the public backward numerics exact while moving the hot
short-sequence reductions off Torch:

- per-row scalar reductions for ``dK_prev``, ``dK_curr``, and ``d_suffix``
- the reverse scalar scan from ``d_suffix`` / ``d_m_chunk`` into ``dM``
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_ReduceKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_ScanKey = tuple[int, tuple[int, int, int], tuple[int, int]]
_COMPILED_REDUCE: dict[_ReduceKey, object] = {}
_COMPILED_SCAN: dict[_ScanKey, object] = {}


class _ChunkIncrementBwdReduce:
    """Warp reduction for exact scalar tail contributions.

    Thread/value layout:
    - one warp owns one ``(bhc, row)`` item
    - lanes stride over the complex pair axis ``N = D / 2``
    - outputs are packed ``(re, im)`` complex scalars
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mSuffix: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mB: cute.Tensor,
        mBPrev0: cute.Tensor,
        mDAlpha: cute.Tensor,
        mDBoundary: cute.Tensor,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mDSuffix: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mSuffix.element_type
                == mKPrev.element_type
                == mKCurr.element_type
                == mB.element_type
                == mBPrev0.element_type
                == mDAlpha.element_type
                == mDBoundary.element_type
                == mDKPrev.element_type
                == mDKCurr.element_type
                == mDSuffix.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("chunk_increment tail reduce expects Float32 tensors.")
        if cutlass.const_expr(
            mSuffix.shape != mKPrev.shape
            or mSuffix.shape != mKCurr.shape
            or mSuffix.shape != mDKPrev.shape
            or mSuffix.shape != mDKCurr.shape
            or mSuffix.shape != mDSuffix.shape
        ):
            raise ValueError("suffix/k/dK/dSuffix tensors must share shape.")
        if cutlass.const_expr(mSuffix.shape[2] != 2):
            raise ValueError("suffix/k/dK/dSuffix tensors must be (BHC, L, 2).")
        if cutlass.const_expr(mB.shape[0] != mSuffix.shape[0] or mB.shape[1] != mSuffix.shape[1]):
            raise ValueError("B and d_alpha must agree on (BHC, L).")
        if cutlass.const_expr(mB.shape != mDAlpha.shape):
            raise ValueError("B and d_alpha must share shape.")
        if cutlass.const_expr(mBPrev0.shape != (mSuffix.shape[0], mB.shape[2])):
            raise ValueError("b_prev0 must be (BHC, D).")
        if cutlass.const_expr(mDBoundary.shape != (mSuffix.shape[0], mB.shape[2])):
            raise ValueError("d_boundary must be (BHC, D).")

        BHC = cute.size(mSuffix.shape[0])
        L = cute.size(mSuffix.shape[1])
        total_items = BHC * L
        warps_per_block = self.num_threads // 32
        self.kernel(
            mSuffix,
            mKPrev,
            mKCurr,
            mB,
            mBPrev0,
            mDAlpha,
            mDBoundary,
            mDKPrev,
            mDKCurr,
            mDSuffix,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mSuffix: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mB: cute.Tensor,
        mBPrev0: cute.Tensor,
        mDAlpha: cute.Tensor,
        mDBoundary: cute.Tensor,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mDSuffix: cute.Tensor,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        L = cute.size(mSuffix.shape[1])
        bhc = item_safe // L
        row = item_safe - bhc * L
        N = cute.size(mB.shape[2]) // 2

        sr = cutlass.Float32(mSuffix[bhc, row, 0])
        si = cutlass.Float32(mSuffix[bhc, row, 1])
        kcr = cutlass.Float32(mKCurr[bhc, row, 0])
        kci = cutlass.Float32(mKCurr[bhc, row, 1])
        kpr = cutlass.Float32(mKPrev[bhc, row, 0])
        kpi = cutlass.Float32(mKPrev[bhc, row, 1])

        dkpr_re = cutlass.Float32(0.0)
        dkpr_im = cutlass.Float32(0.0)
        dkcr_re = cutlass.Float32(0.0)
        dkcr_im = cutlass.Float32(0.0)
        dsuf_re = cutlass.Float32(0.0)
        dsuf_im = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2

            bcr = cutlass.Float32(mB[bhc, row, col + 0])
            bci = cutlass.Float32(mB[bhc, row, col + 1])
            dcr = cutlass.Float32(mDAlpha[bhc, row, col + 0])
            dci = cutlass.Float32(mDAlpha[bhc, row, col + 1])

            curr_base_k_re = sr * bcr - si * bci
            curr_base_k_im = sr * bci + si * bcr
            curr_base_s_re = kcr * bcr - kci * bci
            curr_base_s_im = kcr * bci + kci * bcr

            dkcr_re += dcr * curr_base_k_re + dci * curr_base_k_im
            dkcr_im += -dcr * curr_base_k_im + dci * curr_base_k_re
            dsuf_re += dcr * curr_base_s_re + dci * curr_base_s_im
            dsuf_im += -dcr * curr_base_s_im + dci * curr_base_s_re

            bpr = cutlass.Float32(0.0)
            bpi = cutlass.Float32(0.0)
            dpr = cutlass.Float32(0.0)
            dpi = cutlass.Float32(0.0)
            if row == cutlass.Int32(0):
                bpr = cutlass.Float32(mBPrev0[bhc, col + 0])
                bpi = cutlass.Float32(mBPrev0[bhc, col + 1])
                dpr = cutlass.Float32(mDBoundary[bhc, col + 0])
                dpi = cutlass.Float32(mDBoundary[bhc, col + 1])
            else:
                bpr = cutlass.Float32(mB[bhc, row - 1, col + 0])
                bpi = cutlass.Float32(mB[bhc, row - 1, col + 1])
                dpr = cutlass.Float32(mDAlpha[bhc, row - 1, col + 0])
                dpi = cutlass.Float32(mDAlpha[bhc, row - 1, col + 1])

            prev_base_k_re = sr * bpr - si * bpi
            prev_base_k_im = sr * bpi + si * bpr
            prev_base_s_re = kpr * bpr - kpi * bpi
            prev_base_s_im = kpr * bpi + kpi * bpr

            dkpr_re += dpr * prev_base_k_re + dpi * prev_base_k_im
            dkpr_im += -dpr * prev_base_k_im + dpi * prev_base_k_re
            dsuf_re += dpr * prev_base_s_re + dpi * prev_base_s_im
            dsuf_im += -dpr * prev_base_s_im + dpi * prev_base_s_re
            n += 32

        for offset in (16, 8, 4, 2, 1):
            dkpr_re += cute.arch.shuffle_sync_bfly(dkpr_re, offset=offset, mask=-1, mask_and_clamp=31)
            dkpr_im += cute.arch.shuffle_sync_bfly(dkpr_im, offset=offset, mask=-1, mask_and_clamp=31)
            dkcr_re += cute.arch.shuffle_sync_bfly(dkcr_re, offset=offset, mask=-1, mask_and_clamp=31)
            dkcr_im += cute.arch.shuffle_sync_bfly(dkcr_im, offset=offset, mask=-1, mask_and_clamp=31)
            dsuf_re += cute.arch.shuffle_sync_bfly(dsuf_re, offset=offset, mask=-1, mask_and_clamp=31)
            dsuf_im += cute.arch.shuffle_sync_bfly(dsuf_im, offset=offset, mask=-1, mask_and_clamp=31)

        if item_valid and lane == 0:
            mDKPrev[bhc, row, 0] = dkpr_re
            mDKPrev[bhc, row, 1] = dkpr_im
            mDKCurr[bhc, row, 0] = dkcr_re
            mDKCurr[bhc, row, 1] = dkcr_im
            mDSuffix[bhc, row, 0] = dsuf_re
            mDSuffix[bhc, row, 1] = dsuf_im


class _ChunkIncrementBwdMScan:
    """Short reverse scan from ``d_suffix`` and ``d_m_chunk`` into ``dM``."""

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mM: cute.Tensor,
        mSuffix: cute.Tensor,
        mDSuffix: cute.Tensor,
        mDMChunk: cute.Tensor,
        mDM: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mM.element_type
                == mSuffix.element_type
                == mDSuffix.element_type
                == mDMChunk.element_type
                == mDM.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("chunk_increment dM scan expects Float32 tensors.")
        if cutlass.const_expr(mM.shape != mSuffix.shape or mM.shape != mDSuffix.shape or mM.shape != mDM.shape):
            raise ValueError("M/suffix/d_suffix/dM tensors must share shape.")
        if cutlass.const_expr(mM.shape[2] != 2):
            raise ValueError("M/suffix/d_suffix/dM must be (BHC, L, 2).")
        if cutlass.const_expr(mDMChunk.shape != (mM.shape[0], 2)):
            raise ValueError("d_m_chunk must be (BHC, 2).")

        BHC = cute.size(mM.shape[0])
        warps_per_block = self.num_threads // 32
        self.kernel(mM, mSuffix, mDSuffix, mDMChunk, mDM, BHC).launch(
            grid=[cute.ceil_div(BHC, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mM: cute.Tensor,
        mSuffix: cute.Tensor,
        mDSuffix: cute.Tensor,
        mDMChunk: cute.Tensor,
        mDM: cute.Tensor,
        BHC: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        bhc = bidx * warps_per_block + warp
        if cute.elem_less(bhc, BHC) and lane == 0:
            L = cute.size(mM.shape[1])
            carry_re = cutlass.Float32(mDMChunk[bhc, 0])
            carry_im = cutlass.Float32(mDMChunk[bhc, 1])

            for t in cutlass.range(L, unroll=1):
                sr = cutlass.Float32(mSuffix[bhc, t, 0])
                si = cutlass.Float32(mSuffix[bhc, t, 1])
                mDM[bhc, t, 0] = carry_re * sr + carry_im * si
                mDM[bhc, t, 1] = -carry_re * si + carry_im * sr

                if t + 1 < L:
                    mr = cutlass.Float32(mM[bhc, t, 0])
                    mi = cutlass.Float32(mM[bhc, t, 1])
                    next_re = carry_re * mr + carry_im * mi
                    next_im = -carry_re * mi + carry_im * mr
                    carry_re = next_re + cutlass.Float32(mDSuffix[bhc, t, 0])
                    carry_im = next_im + cutlass.Float32(mDSuffix[bhc, t, 1])


def _get_compiled_reduce(
    suffix: torch.Tensor,
    b_prev0: torch.Tensor,
    B: torch.Tensor,
) -> object:
    device_index = 0 if suffix.device.index is None else int(suffix.device.index)
    key: _ReduceKey = (
        device_index,
        tuple(int(x) for x in suffix.shape),
        tuple(int(x) for x in b_prev0.shape),
        tuple(int(x) for x in B.shape),
    )
    compiled = _COMPILED_REDUCE.get(key)
    if compiled is not None:
        return compiled

    out_scalar = torch.empty_like(suffix)
    kernel = _ChunkIncrementBwdReduce()
    compiled = cute.compile(
        kernel,
        from_dlpack(suffix, assumed_align=suffix.element_size()),
        from_dlpack(suffix, assumed_align=suffix.element_size()),
        from_dlpack(suffix, assumed_align=suffix.element_size()),
        from_dlpack(B, assumed_align=B.element_size()),
        from_dlpack(b_prev0, assumed_align=b_prev0.element_size()),
        from_dlpack(B, assumed_align=B.element_size()),
        from_dlpack(b_prev0, assumed_align=b_prev0.element_size()),
        from_dlpack(out_scalar, assumed_align=out_scalar.element_size()),
        from_dlpack(out_scalar, assumed_align=out_scalar.element_size()),
        from_dlpack(out_scalar, assumed_align=out_scalar.element_size()),
    )
    _COMPILED_REDUCE[key] = compiled
    return compiled


def _get_compiled_scan(
    M: torch.Tensor,
    d_m_chunk: torch.Tensor,
) -> object:
    device_index = 0 if M.device.index is None else int(M.device.index)
    key: _ScanKey = (
        device_index,
        tuple(int(x) for x in M.shape),
        tuple(int(x) for x in d_m_chunk.shape),
    )
    compiled = _COMPILED_SCAN.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkIncrementBwdMScan()
    compiled = cute.compile(
        kernel,
        from_dlpack(M, assumed_align=M.element_size()),
        from_dlpack(M, assumed_align=M.element_size()),
        from_dlpack(M, assumed_align=M.element_size()),
        from_dlpack(d_m_chunk, assumed_align=d_m_chunk.element_size()),
        from_dlpack(M, assumed_align=M.element_size()),
    )
    _COMPILED_SCAN[key] = compiled
    return compiled


def chunk_increment_bwd_tail_exact_cute(
    suffix: torch.Tensor,
    k_prev: torch.Tensor,
    k_curr: torch.Tensor,
    B: torch.Tensor,
    b_prev0: torch.Tensor,
    d_alpha: torch.Tensor,
    d_boundary: torch.Tensor,
    M: torch.Tensor,
    d_m_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fp32 CuTe kernels for the scalar-heavy chunk_increment tail."""
    tensors = (
        ("suffix", suffix),
        ("k_prev", k_prev),
        ("k_curr", k_curr),
        ("B", B),
        ("b_prev0", b_prev0),
        ("d_alpha", d_alpha),
        ("d_boundary", d_boundary),
        ("M", M),
        ("d_m_chunk", d_m_chunk),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("chunk_increment tail CuTe helpers require CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError("chunk_increment tail CuTe helpers expect contiguous tensors.")
    if any(t.dtype != torch.float32 for _name, t in tensors):
        raise ValueError("chunk_increment tail CuTe helpers expect float32 tensors.")
    if suffix.shape != k_prev.shape or suffix.shape != k_curr.shape or suffix.shape != M.shape:
        raise ValueError("suffix, k_prev, k_curr, and M must share shape.")
    if suffix.shape[2] != 2:
        raise ValueError("suffix/k/M must be (BHC, L, 2).")
    if B.shape != d_alpha.shape:
        raise ValueError("B and d_alpha must share shape.")
    if B.shape[:2] != suffix.shape[:2]:
        raise ValueError("B and suffix must agree on (BHC, L).")
    if b_prev0.shape != (suffix.shape[0], B.shape[2]):
        raise ValueError("b_prev0 must be (BHC, D).")
    if d_boundary.shape != b_prev0.shape:
        raise ValueError("d_boundary must match b_prev0.")
    if d_m_chunk.shape != (suffix.shape[0], 2):
        raise ValueError("d_m_chunk must be (BHC, 2).")

    dK_prev = torch.empty_like(suffix)
    dK_curr = torch.empty_like(suffix)
    d_suffix = torch.empty_like(suffix)
    compiled_reduce = _get_compiled_reduce(suffix, b_prev0, B)
    compiled_reduce(
        from_dlpack(suffix, assumed_align=suffix.element_size()),
        from_dlpack(k_prev, assumed_align=k_prev.element_size()),
        from_dlpack(k_curr, assumed_align=k_curr.element_size()),
        from_dlpack(B, assumed_align=B.element_size()),
        from_dlpack(b_prev0, assumed_align=b_prev0.element_size()),
        from_dlpack(d_alpha, assumed_align=d_alpha.element_size()),
        from_dlpack(d_boundary, assumed_align=d_boundary.element_size()),
        from_dlpack(dK_prev, assumed_align=dK_prev.element_size()),
        from_dlpack(dK_curr, assumed_align=dK_curr.element_size()),
        from_dlpack(d_suffix, assumed_align=d_suffix.element_size()),
    )

    dM = torch.empty_like(M)
    compiled_scan = _get_compiled_scan(M, d_m_chunk)
    compiled_scan(
        from_dlpack(M, assumed_align=M.element_size()),
        from_dlpack(suffix, assumed_align=suffix.element_size()),
        from_dlpack(d_suffix, assumed_align=d_suffix.element_size()),
        from_dlpack(d_m_chunk, assumed_align=d_m_chunk.element_size()),
        from_dlpack(dM, assumed_align=dM.element_size()),
    )
    return dK_prev, dK_curr, d_suffix, dM


__all__ = ["chunk_increment_bwd_tail_exact_cute"]
