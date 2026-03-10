"""CuTe backward slice for ``chunk_scan`` gradients into ``B`` and ``K``.

Logical contract
----------------
This slice consumes cached forward-packed tensors plus the lightweight raw
metadata needed to map packed key gradients back to the public operator inputs:

- ``Q_rev``: ``flip(Q, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Vprev_rev``: ``flip(Vprev, dim=1)``, shape ``(BHC, L, 1, P)``
- ``Vcurr_rev``: ``flip(Vcurr, dim=1)``, shape ``(BHC, L, 1, P)``
- ``neg_logprefix_half_rev``: ``-flip(logprefix_half, dim=1)``, shape ``(BHC, L)``
- ``phase``: ``(BHC, L, 2)``, the unit-complex phase prefix from ``M_raw``
- ``K_raw``: ``(BHC, L, 2, 2)``, raw public taps in packed-complex form
- ``B_raw``: ``(BHC, L, D)``, raw public ``B`` rows in interleaved ``2N``
- ``B_head``: ``(BHC, D)``, per-chunk boundary ``B`` input used at ``t = 0``
- ``d_out``: ``(B, H, T, P)``

Why this contract
-----------------
The packed-real key gradient is another causal attention-like contraction after:

- reversing time,
- swapping the forward value vectors into the query role,
- using reversed ``d_out`` as the key vectors,
- keeping the reversed/negated logprefix metadata.

After the dense packed ``dKprev/dKcurr`` work, the remaining map back to the
public ``(B, B_prev, K)`` contract is a short explicit complex scatter. That
host-side algebra stays readable and avoids inventing another CuTe layout
family for a non-dominant reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.common import (
    _scalar_grad_from_vec,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_chunk_scan,
    _get_compiled_phase,
)
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _complex_dtype_from_real,
    _pack_complex_pairs,
)


@dataclass
class _ChunkScanBwdDBScratch:
    K_zero: torch.Tensor
    V_zero: torch.Tensor
    Z0_zero: torch.Tensor
    dKprev_rev: torch.Tensor
    dKcurr_rev: torch.Tensor


_ScratchKey = tuple[int, torch.dtype, int, int, int, int]
_SCRATCH_DB: dict[_ScratchKey, _ChunkScanBwdDBScratch] = {}
_CompiledScatterKey = tuple[int, tuple[int, int, int], tuple[int, int, int]]
_CompiledReduceKey = tuple[int, tuple[int, int, int], tuple[int, int], tuple[int, int, int, int]]
_COMPILED_DB_SCATTER: dict[_CompiledScatterKey, object] = {}
_COMPILED_DK_REDUCE: dict[_CompiledReduceKey, object] = {}


class _ChunkScanBwdDBExactScatter:
    """Exact fp32 scatter from packed ``dK`` intermediates into ``dB``/``dB_prev``.

    Logical shape:
    - ``dKprev/dKcurr``: ``(BHC, L, D)``, interleaved complex pairs in fp32
    - ``phase``: ``(BHC, L, 2)``
    - ``K_raw``: ``(BHC, L, 2, 2)``
    - output ``dB_pad``: ``(BH, T_pad, D)``
    - output ``dB_prev``: ``(BH, D)``

    Mapping:
    - one thread owns one complex pair for one ``(bhc, row)``
    - output ``dB[t]`` is the sum of the current-tap contribution at ``t`` and
      the next-row previous-tap contribution, including the chunk-boundary carry
      into the final row
    """

    def __init__(self, *, pair_tile: int, num_threads: int = 128) -> None:
        self.pair_tile = int(pair_tile)
        self.num_threads = int(num_threads)
        if self.pair_tile <= 0 or self.num_threads % self.pair_tile != 0:
            raise ValueError("num_threads must be divisible by pair_tile.")
        self.row_tile = self.num_threads // self.pair_tile

    @cute.jit
    def __call__(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mDBPad: cute.Tensor,
        mDBPrev: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDKPrev.element_type
                == mDKCurr.element_type
                == mPhase.element_type
                == mKRaw.element_type
                == mDBPad.element_type
                == mDBPrev.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact dB scatter expects Float32 tensors.")
        if cutlass.const_expr(mDKPrev.shape != mDKCurr.shape):
            raise ValueError("dKprev and dKcurr must share shape.")
        if cutlass.const_expr(mPhase.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mKRaw.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2, 2)):
            raise ValueError("K_raw must be (BHC, L, 2, 2).")

        BHC = cute.size(mDKPrev.shape[0])
        L = cute.size(mDKPrev.shape[1])
        pair_cols = cute.size(mDKPrev.shape[2]) // 2
        grid_x = cute.ceil_div(pair_cols, self.pair_tile)
        grid_y = cute.ceil_div(L, self.row_tile)
        self.kernel(mDKPrev, mDKCurr, mPhase, mKRaw, mDBPad, mDBPrev, n_chunks).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mDBPad: cute.Tensor,
        mDBPrev: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pair_tile_idx, row_tile_idx, bhc = cute.arch.block_idx()

        pair_local = tidx % self.pair_tile
        row_local = tidx // self.pair_tile
        row = row_tile_idx * self.row_tile + row_local
        pair_idx = pair_tile_idx * self.pair_tile + pair_local
        pair_cols = mDKPrev.shape[2] // 2

        if cute.elem_less(row, mDKPrev.shape[1]) and cute.elem_less(pair_idx, pair_cols):
            bh = bhc // n_chunks
            chunk = bhc - bh * n_chunks
            global_t = chunk * mDKPrev.shape[1] + row
            col = pair_idx * 2

            pr = cutlass.Float32(mPhase[bhc, row, 0])
            pi = cutlass.Float32(mPhase[bhc, row, 1])
            dkp_re = cutlass.Float32(mDKPrev[bhc, row, col + 0])
            dkp_im = cutlass.Float32(mDKPrev[bhc, row, col + 1])
            dkc_re = cutlass.Float32(mDKCurr[bhc, row, col + 0])
            dkc_im = cutlass.Float32(mDKCurr[bhc, row, col + 1])

            dbp_re = pr * dkp_re + pi * dkp_im
            dbp_im = pi * dkp_re - pr * dkp_im
            dbc_re = pr * dkc_re + pi * dkc_im
            dbc_im = pi * dkc_re - pr * dkc_im

            kcr = cutlass.Float32(mKRaw[bhc, row, 1, 0])
            kci = cutlass.Float32(mKRaw[bhc, row, 1, 1])
            out_re = kcr * dbc_re + kci * dbc_im
            out_im = kcr * dbc_im - kci * dbc_re

            next_row = row + 1
            if cute.elem_less(next_row, mDKPrev.shape[1]):
                npr = cutlass.Float32(mPhase[bhc, next_row, 0])
                npi = cutlass.Float32(mPhase[bhc, next_row, 1])
                ndkp_re = cutlass.Float32(mDKPrev[bhc, next_row, col + 0])
                ndkp_im = cutlass.Float32(mDKPrev[bhc, next_row, col + 1])
                ndbp_re = npr * ndkp_re + npi * ndkp_im
                ndbp_im = npi * ndkp_re - npr * ndkp_im
                nkpr = cutlass.Float32(mKRaw[bhc, next_row, 0, 0])
                nkpi = cutlass.Float32(mKRaw[bhc, next_row, 0, 1])
                out_re += nkpr * ndbp_re + nkpi * ndbp_im
                out_im += nkpr * ndbp_im - nkpi * ndbp_re
            else:
                next_chunk = chunk + 1
                if cute.elem_less(next_chunk, n_chunks):
                    next_bhc = bhc + 1
                    npr = cutlass.Float32(mPhase[next_bhc, 0, 0])
                    npi = cutlass.Float32(mPhase[next_bhc, 0, 1])
                    ndkp_re = cutlass.Float32(mDKPrev[next_bhc, 0, col + 0])
                    ndkp_im = cutlass.Float32(mDKPrev[next_bhc, 0, col + 1])
                    ndbp_re = npr * ndkp_re + npi * ndkp_im
                    ndbp_im = npi * ndkp_re - npr * ndkp_im
                    nkpr = cutlass.Float32(mKRaw[next_bhc, 0, 0, 0])
                    nkpi = cutlass.Float32(mKRaw[next_bhc, 0, 0, 1])
                    out_re += nkpr * ndbp_re + nkpi * ndbp_im
                    out_im += nkpr * ndbp_im - nkpi * ndbp_re

            mDBPad[bh, global_t, col + 0] = out_re
            mDBPad[bh, global_t, col + 1] = out_im

            if chunk == cutlass.Int32(0) and row == cutlass.Int32(0):
                kpr = cutlass.Float32(mKRaw[bhc, 0, 0, 0])
                kpi = cutlass.Float32(mKRaw[bhc, 0, 0, 1])
                mDBPrev[bh, col + 0] = kpr * dbp_re + kpi * dbp_im
                mDBPrev[bh, col + 1] = kpr * dbp_im - kpi * dbp_re


class _ChunkScanBwdDKExactReduce:
    """Warp reduction from exact packed intermediates into public tap gradients."""

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDKPad: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDKPrev.element_type
                == mDKCurr.element_type
                == mPhase.element_type
                == mBRaw.element_type
                == mBHead.element_type
                == mDKPad.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact dK reduction expects Float32 tensors.")
        if cutlass.const_expr(mDKPrev.shape != mDKCurr.shape or mDKPrev.shape != mBRaw.shape):
            raise ValueError("dKprev, dKcurr, and B_raw must share shape.")
        if cutlass.const_expr(mPhase.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mBHead.shape != (mDKPrev.shape[0], mDKPrev.shape[2])):
            raise ValueError("B_head must be (BHC, D).")
        if cutlass.const_expr(mDKPad.shape[2] != 4):
            raise ValueError("dK pad must store 4 packed tap scalars.")

        BHC = cute.size(mDKPrev.shape[0])
        L = cute.size(mDKPrev.shape[1])
        total_items = BHC * L
        warps_per_block = self.num_threads // 32
        self.kernel(mDKPrev, mDKCurr, mPhase, mBRaw, mBHead, mDKPad, n_chunks, total_items).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDKPad: cute.Tensor,
        n_chunks: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        L = cute.size(mDKPrev.shape[1])
        bhc = item_safe // L
        row = item_safe - bhc * L
        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        global_t = chunk * L + row
        N = cute.size(mDKPrev.shape[2]) // 2

        pr = cutlass.Float32(mPhase[bhc, row, 0])
        pi = cutlass.Float32(mPhase[bhc, row, 1])
        acc_prev_re = cutlass.Float32(0.0)
        acc_prev_im = cutlass.Float32(0.0)
        acc_curr_re = cutlass.Float32(0.0)
        acc_curr_im = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2
            dkp_re = cutlass.Float32(mDKPrev[bhc, row, col + 0])
            dkp_im = cutlass.Float32(mDKPrev[bhc, row, col + 1])
            dkc_re = cutlass.Float32(mDKCurr[bhc, row, col + 0])
            dkc_im = cutlass.Float32(mDKCurr[bhc, row, col + 1])
            dbp_re = pr * dkp_re + pi * dkp_im
            dbp_im = pi * dkp_re - pr * dkp_im
            dbc_re = pr * dkc_re + pi * dkc_im
            dbc_im = pi * dkc_re - pr * dkc_im

            bpr = cutlass.Float32(0.0)
            bpi = cutlass.Float32(0.0)
            if row == cutlass.Int32(0):
                bpr = cutlass.Float32(mBHead[bhc, col + 0])
                bpi = cutlass.Float32(mBHead[bhc, col + 1])
            else:
                bpr = cutlass.Float32(mBRaw[bhc, row - 1, col + 0])
                bpi = cutlass.Float32(mBRaw[bhc, row - 1, col + 1])
            bcr = cutlass.Float32(mBRaw[bhc, row, col + 0])
            bci = cutlass.Float32(mBRaw[bhc, row, col + 1])

            acc_prev_re += bpr * dbp_re + bpi * dbp_im
            acc_prev_im += bpr * dbp_im - bpi * dbp_re
            acc_curr_re += bcr * dbc_re + bci * dbc_im
            acc_curr_im += bcr * dbc_im - bci * dbc_re
            n += 32

        for offset in (16, 8, 4, 2, 1):
            acc_prev_re += cute.arch.shuffle_sync_bfly(acc_prev_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_prev_im += cute.arch.shuffle_sync_bfly(acc_prev_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_re += cute.arch.shuffle_sync_bfly(acc_curr_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_im += cute.arch.shuffle_sync_bfly(acc_curr_im, offset=offset, mask=-1, mask_and_clamp=31)

        if item_valid and lane == 0:
            mDKPad[bh, global_t, 0] = acc_prev_re
            mDKPad[bh, global_t, 1] = acc_prev_im
            mDKPad[bh, global_t, 2] = acc_curr_re
            mDKPad[bh, global_t, 3] = acc_curr_im


def _get_compiled_db_exact_scatter(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    dB_pad: torch.Tensor,
    dB_prev: torch.Tensor,
) -> object:
    device_index = 0 if dKprev.device.index is None else int(dKprev.device.index)
    key: _CompiledScatterKey = (
        device_index,
        tuple(int(x) for x in dKprev.shape),
        tuple(int(x) for x in dB_pad.shape),
    )
    compiled = _COMPILED_DB_SCATTER.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDBExactScatter(pair_tile=16)
    compiled = cute.compile(
        kernel,
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKcurr, assumed_align=dKcurr.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(dB_pad, assumed_align=dB_pad.element_size()),
        from_dlpack(dB_prev, assumed_align=dB_prev.element_size()),
        int(dB_pad.shape[1] // dKprev.shape[1]),
    )
    _COMPILED_DB_SCATTER[key] = compiled
    return compiled


def _get_compiled_dk_exact_reduce(
    dKprev: torch.Tensor,
    phase: torch.Tensor,
    B_head: torch.Tensor,
    dK_pad: torch.Tensor,
) -> object:
    device_index = 0 if dKprev.device.index is None else int(dKprev.device.index)
    key: _CompiledReduceKey = (
        device_index,
        tuple(int(x) for x in dKprev.shape),
        tuple(int(x) for x in B_head.shape),
        tuple(int(x) for x in dK_pad.shape),
    )
    compiled = _COMPILED_DK_REDUCE.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDKExactReduce()
    compiled = cute.compile(
        kernel,
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(dK_pad, assumed_align=dK_pad.element_size()),
        int(dK_pad.shape[1] // dKprev.shape[1]),
    )
    _COMPILED_DK_REDUCE[key] = compiled
    return compiled


def chunk_scan_bwd_db_exact_cute(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fp32 CuTe scatter from packed key grads into public ``dB/dB_prev/dK``."""
    tensors = (
        ("dK_prev_packed", dK_prev_packed),
        ("dK_curr_packed", dK_curr_packed),
        ("phase", phase),
        ("K_raw", K_raw),
        ("B_raw", B_raw),
        ("B_head", B_head),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("Exact CuTe dB scatter requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError("Exact CuTe dB scatter expects contiguous tensors.")
    if any(t.dtype != torch.float32 for _name, t in tensors):
        raise ValueError("Exact CuTe dB scatter expects float32 tensors.")
    if dK_prev_packed.shape != dK_curr_packed.shape or dK_prev_packed.shape != B_raw.shape:
        raise ValueError("dK_prev_packed, dK_curr_packed, and B_raw must share shape.")
    if phase.shape != (*dK_prev_packed.shape[:2], 2):
        raise ValueError("phase must be (BHC, L, 2).")
    if K_raw.shape != (*dK_prev_packed.shape[:2], 2, 2):
        raise ValueError("K_raw must be (BHC, L, 2, 2).")
    if B_head.shape != (dK_prev_packed.shape[0], dK_prev_packed.shape[2]):
        raise ValueError("B_head must be (BHC, D).")

    BHC, L, D = map(int, dK_prev_packed.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"dK leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dB_pad = torch.empty((BH, T_pad, D), device=dK_prev_packed.device, dtype=torch.float32)
    dB_prev = torch.empty((BH, D), device=dK_prev_packed.device, dtype=torch.float32)
    dK_pad = torch.empty((BH, T_pad, 4), device=dK_prev_packed.device, dtype=torch.float32)

    compiled_scatter = _get_compiled_db_exact_scatter(
        dK_prev_packed,
        dK_curr_packed,
        phase,
        K_raw,
        dB_pad,
        dB_prev,
    )
    compiled_reduce = _get_compiled_dk_exact_reduce(
        dK_prev_packed,
        phase,
        B_head,
        dK_pad,
    )

    compiled_scatter(
        from_dlpack(dK_prev_packed, assumed_align=dK_prev_packed.element_size()),
        from_dlpack(dK_curr_packed, assumed_align=dK_curr_packed.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(dB_pad, assumed_align=dB_pad.element_size()),
        from_dlpack(dB_prev, assumed_align=dB_prev.element_size()),
        n_chunks,
    )
    compiled_reduce(
        from_dlpack(dK_prev_packed, assumed_align=dK_prev_packed.element_size()),
        from_dlpack(dK_curr_packed, assumed_align=dK_curr_packed.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(B_raw, assumed_align=B_raw.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(dK_pad, assumed_align=dK_pad.element_size()),
        n_chunks,
    )
    dB = dB_pad.reshape(batch_size, n_heads, T_pad, D)[:, :, :T, :].contiguous()
    dB_prev_out = dB_prev.reshape(batch_size, n_heads, D).contiguous()
    dK = dK_pad.reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :].contiguous()
    return dB, dB_prev_out, dK


def prepare_chunk_scan_bwd_db_operands(
    Q: torch.Tensor,
    Vprev: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    M_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cached reverse-time operands plus phase metadata for the ``dB`` slice."""
    if Q.ndim != 4 or Vprev.ndim != 4 or Vcurr.ndim != 4:
        raise ValueError("Q/Vprev/Vcurr must be rank-4 tensors.")
    if Q.shape[:3] != Vprev.shape[:3] or Q.shape[:3] != Vcurr.shape[:3]:
        raise ValueError(
            "Q/Vprev/Vcurr must agree on the leading packed dims. Got "
            f"{tuple(Q.shape)}, {tuple(Vprev.shape)}, {tuple(Vcurr.shape)}."
        )
    if Q.shape[2] != 1 or Vprev.shape[2] != 1 or Vcurr.shape[2] != 1:
        raise ValueError("Packed Q/V tensors must be shaped as (BHC, L, 1, feat).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)} for Q shape {tuple(Q.shape)}."
        )
    if not (
        Q.is_contiguous()
        and Vprev.is_contiguous()
        and Vcurr.is_contiguous()
        and logprefix_half.is_contiguous()
        and M_raw.is_contiguous()
    ):
        raise ValueError(
            "Q, Vprev, Vcurr, logprefix_half, and M_raw must be contiguous cached "
            "forward tensors."
        )

    phase = torch.empty(
        (M_raw.shape[0], M_raw.shape[1], 2),
        device=M_raw.device,
        dtype=torch.float32,
    )
    compiled_phase = _get_compiled_phase(M_raw, phase)
    compiled_phase(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
    )
    return (
        torch.flip(Q, dims=[1]).contiguous(),
        torch.flip(Vprev, dims=[1]).contiguous(),
        torch.flip(Vcurr, dims=[1]).contiguous(),
        (-torch.flip(logprefix_half, dims=[1])).contiguous(),
        phase,
    )


def _get_db_scratch(
    *,
    vprev_rev: torch.Tensor,
    D: int,
) -> _ChunkScanBwdDBScratch:
    device_index = 0 if vprev_rev.device.index is None else int(vprev_rev.device.index)
    BHC, L, _, P = map(int, vprev_rev.shape)
    key: _ScratchKey = (
        device_index,
        vprev_rev.dtype,
        BHC,
        L,
        P,
        D,
    )
    scratch = _SCRATCH_DB.get(key)
    if scratch is not None:
        return scratch

    K_zero = torch.zeros_like(vprev_rev)
    V_zero = torch.zeros((BHC, L, 1, D), device=vprev_rev.device, dtype=vprev_rev.dtype)
    Z0_zero = torch.zeros(
        (BHC, D, 1, P), device=vprev_rev.device, dtype=vprev_rev.dtype
    )
    dKprev_rev = torch.empty(
        (BHC, L, 1, D), device=vprev_rev.device, dtype=torch.float32
    )
    dKcurr_rev = torch.empty_like(dKprev_rev)
    scratch = _ChunkScanBwdDBScratch(
        K_zero=K_zero,
        V_zero=V_zero,
        Z0_zero=Z0_zero,
        dKprev_rev=dKprev_rev,
        dKcurr_rev=dKcurr_rev,
    )
    _SCRATCH_DB[key] = scratch
    return scratch


def _chunk_scan_bwd_dk_packed_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute packed ``dKprev/dKcurr`` on the cached reverse-time contract."""
    BHC, L, _, D = map(int, Q_rev.shape)
    P = int(Vprev_rev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )

    scratch = _get_db_scratch(vprev_rev=Vprev_rev, D=D)
    if T_pad != T:
        pad = T_pad - T
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, pad, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_rev = torch.flip(
        d_out.reshape(BHC, L, 1, P).to(dtype=Vprev_rev.dtype), dims=[1]
    ).contiguous()
    # The packed key-gradient contraction needs the same inner-kernel prefix
    # renormalization as the packed ``dQ`` path: after reversing time, the
    # stable segment ratio is represented by half of the negated half-logprefix.
    half_neg_logprefix_half_rev = (0.5 * neg_logprefix_half_rev).contiguous()

    compiled_prev = _get_compiled_chunk_scan(
        Vprev_rev,
        d_out_rev,
        Q_rev,
        scratch.K_zero,
        scratch.V_zero,
        half_neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.dKprev_rev,
    )
    compiled_curr = _get_compiled_chunk_scan(
        Vcurr_rev,
        scratch.K_zero,
        scratch.V_zero,
        d_out_rev,
        Q_rev,
        half_neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.dKcurr_rev,
    )

    compiled_prev(
        from_dlpack(Vprev_rev, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(half_neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.dKprev_rev, assumed_align=16),
    )
    compiled_curr(
        from_dlpack(Vcurr_rev, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(half_neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.dKcurr_rev, assumed_align=16),
    )
    return (
        torch.flip(scratch.dKprev_rev.squeeze(2), dims=[1]).contiguous(),
        torch.flip(scratch.dKcurr_rev.squeeze(2), dims=[1]).contiguous(),
    )


def chunk_scan_bwd_db_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(dB, dB_prev, dK)`` for ``chunk_scan`` from cached forward packs."""
    tensors = (
        ("Q_rev", Q_rev),
        ("Vprev_rev", Vprev_rev),
        ("Vcurr_rev", Vcurr_rev),
        ("neg_logprefix_half_rev", neg_logprefix_half_rev),
        ("phase", phase),
        ("K_raw", K_raw),
        ("B_raw", B_raw),
        ("B_head", B_head),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q_rev.ndim != 4 or Vprev_rev.ndim != 4 or Vcurr_rev.ndim != 4:
        raise ValueError("Q_rev/Vprev_rev/Vcurr_rev must be rank-4 tensors.")
    if Q_rev.shape[:3] != Vprev_rev.shape[:3] or Q_rev.shape[:3] != Vcurr_rev.shape[:3]:
        raise ValueError(
            "Q_rev/Vprev_rev/Vcurr_rev must share leading packed dims. Got "
            f"{tuple(Q_rev.shape)}, {tuple(Vprev_rev.shape)}, {tuple(Vcurr_rev.shape)}."
        )
    if Q_rev.shape[2] != 1 or Vprev_rev.shape[2] != 1 or Vcurr_rev.shape[2] != 1:
        raise ValueError("Packed reverse-time tensors must be (BHC, L, 1, feat).")
    if neg_logprefix_half_rev.shape != Q_rev.shape[:2]:
        raise ValueError(
            "neg_logprefix_half_rev must be (BHC, L) matching Q_rev. Got "
            f"{tuple(neg_logprefix_half_rev.shape)}."
        )
    if phase.shape != (*Q_rev.shape[:2], 2):
        raise ValueError(
            f"phase must be (BHC, L, 2) matching Q_rev. Got {tuple(phase.shape)}."
        )
    if K_raw.shape != (*Q_rev.shape[:2], 2, 2):
        raise ValueError(
            "K_raw must be (BHC, L, 2, 2). Got "
            f"{tuple(K_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_raw.shape != (*Q_rev.shape[:2], Q_rev.shape[-1]):
        raise ValueError(
            "B_raw must be (BHC, L, D) matching Q_rev. Got "
            f"{tuple(B_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_head.shape != (Q_rev.shape[0], Q_rev.shape[-1]):
        raise ValueError(
            "B_head must be (BHC, D) matching Q_rev. Got "
            f"{tuple(B_head.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if (
        d_out.ndim != 4
        or d_out.shape[:2] != (batch_size, n_heads)
        or int(d_out.shape[2]) != T
    ):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P). Got "
            f"{tuple(d_out.shape)} for {(batch_size, n_heads, T)}."
        )

    BHC, L, _, D = map(int, Q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )

    cplx_dtype = _complex_dtype_from_real(torch.float32)
    phase_c = (
        torch.view_as_complex(phase.contiguous()).to(dtype=cplx_dtype).unsqueeze(-1)
    )
    dKprev_packed, dKcurr_packed = _chunk_scan_bwd_dk_packed_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        d_out,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dKprev_c = _as_complex_pairs(dKprev_packed, name="dKprev_packed").to(
        dtype=cplx_dtype
    )
    dKcurr_c = _as_complex_pairs(dKcurr_packed, name="dKcurr_packed").to(
        dtype=cplx_dtype
    )

    # ``Kprev/Kcurr`` are packed ``conj(beta) * phase``. For the underlying
    # real 2x2 map, the gradient back to ``beta`` is ``phase * conj(dK)``.
    d_beta_prev = phase_c * torch.conj(dKprev_c)
    d_beta_curr = phase_c * torch.conj(dKcurr_c)

    b_curr = _as_complex_pairs(B_raw, name="B_raw").to(dtype=cplx_dtype)
    b_head_c = (
        _as_complex_pairs(B_head.unsqueeze(1), name="B_head")
        .squeeze(1)
        .to(dtype=cplx_dtype)
    )
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if L > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]

    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous()).to(
        dtype=cplx_dtype
    )
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous()).to(
        dtype=cplx_dtype
    )

    dB_curr_c = torch.conj(k_curr_c).unsqueeze(-1) * d_beta_curr
    dB_prev_seq_c = torch.conj(k_prev_c).unsqueeze(-1) * d_beta_prev
    dK_prev_tap_c = _scalar_grad_from_vec(b_prev_seq, d_beta_prev)
    dK_curr_tap_c = _scalar_grad_from_vec(b_curr, d_beta_curr)

    N = D // 2
    dB_blk = dB_curr_c.reshape(batch_size, n_heads, n_chunks, L, N).clone()
    dB_prev_view = dB_prev_seq_c.reshape(batch_size, n_heads, n_chunks, L, N)
    if L > 1:
        dB_blk[:, :, :, :-1, :] += dB_prev_view[:, :, :, 1:, :]

    d_head_c = dB_prev_view[:, :, :, 0, :]
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_head_c[:, :, 1:, :]

    dB_prev0_c = d_head_c[:, :, 0, :].contiguous()
    dB = _pack_complex_pairs(
        dB_blk.reshape(batch_size, n_heads, T_pad, N),
        real_dtype=torch.float32,
    )[:, :, :T, :].contiguous()
    dB_prev = _pack_complex_pairs(dB_prev0_c, real_dtype=torch.float32)

    dK_prev_real = torch.view_as_real(
        dK_prev_tap_c.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK_curr_real = torch.view_as_real(
        dK_curr_tap_c.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK = (
        torch.stack((dK_prev_real, dK_curr_real), dim=4)
        .reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :]
        .contiguous()
    )
    return dB, dB_prev, dK


__all__ = [
    "prepare_chunk_scan_bwd_db_operands",
    "chunk_scan_bwd_db_cute",
    "chunk_scan_bwd_db_exact_cute",
]
