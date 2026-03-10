"""Exact CuTe kernel for ``chunk_scan`` cumulative logprefix gradients."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_CompiledKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_DLOGPREFIX: dict[_CompiledKey, object] = {}


class _ChunkScanBwdDLogprefixExact:
    """Warp-cooperative exact reduction for packed ``dlogprefix_half``.

    Logical shape:
    - ``score_prev``, ``score_curr``, ``dSprev``, ``dScurr``: ``(BHC, L, L)``
    - ``logprefix_half``: ``(BHC, L)``
    - ``y_off`` / ``d_out``: ``(BHC, L, P)``
    - output ``dlogprefix_half``: ``(BHC, L)``

    Thread layout:
    - one warp owns one ``(bhc, row)``
    - lanes stride across ``L`` for row/column contributions and across ``P``
      for the off-term reduction
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mScorePrev: cute.Tensor,
        mScoreCurr: cute.Tensor,
        mDSPrev: cute.Tensor,
        mDSCurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mYOff: cute.Tensor,
        mDOut: cute.Tensor,
        mDLogprefix: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mScorePrev.element_type
                == mScoreCurr.element_type
                == mDSPrev.element_type
                == mDSCurr.element_type
                == mLogprefix.element_type
                == mYOff.element_type
                == mDOut.element_type
                == mDLogprefix.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact dlogprefix kernel expects Float32 tensors.")
        if cutlass.const_expr(
            mScorePrev.shape != mScoreCurr.shape
            or mScorePrev.shape != mDSPrev.shape
            or mScorePrev.shape != mDSCurr.shape
        ):
            raise ValueError("score and dS tensors must share shape.")
        if cutlass.const_expr(mScorePrev.shape[0] != mLogprefix.shape[0] or mScorePrev.shape[1] != mLogprefix.shape[1]):
            raise ValueError("logprefix must be (BHC, L) matching score tensors.")
        if cutlass.const_expr(mYOff.shape != mDOut.shape):
            raise ValueError("y_off and d_out must share shape.")
        if cutlass.const_expr(mYOff.shape[0] != mLogprefix.shape[0] or mYOff.shape[1] != mLogprefix.shape[1]):
            raise ValueError("y_off/d_out must be (BHC, L, P) matching logprefix.")
        if cutlass.const_expr(mDLogprefix.shape != mLogprefix.shape):
            raise ValueError("dlogprefix output must match logprefix shape.")

        BHC = cute.size(mLogprefix.shape[0])
        L = cute.size(mLogprefix.shape[1])
        total_items = BHC * L
        warps_per_block = self.num_threads // 32
        self.kernel(
            mScorePrev,
            mScoreCurr,
            mDSPrev,
            mDSCurr,
            mLogprefix,
            mYOff,
            mDOut,
            mDLogprefix,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mScorePrev: cute.Tensor,
        mScoreCurr: cute.Tensor,
        mDSPrev: cute.Tensor,
        mDSCurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mYOff: cute.Tensor,
        mDOut: cute.Tensor,
        mDLogprefix: cute.Tensor,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        L = cute.size(mLogprefix.shape[1])
        bhc = item_safe // L
        row = item_safe - bhc * L
        lp_row = cutlass.Float32(mLogprefix[bhc, row])

        off = cutlass.Float32(0.0)
        P = cute.size(mYOff.shape[2])
        p = lane
        while p < P:
            off += cutlass.Float32(mDOut[bhc, row, p]) * cutlass.Float32(mYOff[bhc, row, p])
            p += 32

        row_prev = cutlass.Float32(0.0)
        row_curr = cutlass.Float32(0.0)
        col_prev = cutlass.Float32(0.0)
        col_curr = cutlass.Float32(0.0)

        j = lane
        while j < L:
            lp_j = cutlass.Float32(mLogprefix[bhc, j])
            if j <= row:
                s_row = cutlass.Float32(cute.math.exp(cutlass.Float32(2.0) * (lp_row - lp_j)))
                row_prev += (
                    cutlass.Float32(mDSPrev[bhc, row, j])
                    * cutlass.Float32(mScorePrev[bhc, row, j])
                    * s_row
                )
                row_curr += (
                    cutlass.Float32(mDSCurr[bhc, row, j])
                    * cutlass.Float32(mScoreCurr[bhc, row, j])
                    * s_row
                )
            if row <= j:
                s_col = cutlass.Float32(cute.math.exp(cutlass.Float32(2.0) * (lp_j - lp_row)))
                col_prev += (
                    cutlass.Float32(mDSPrev[bhc, j, row])
                    * cutlass.Float32(mScorePrev[bhc, j, row])
                    * s_col
                )
                col_curr += (
                    cutlass.Float32(mDSCurr[bhc, j, row])
                    * cutlass.Float32(mScoreCurr[bhc, j, row])
                    * s_col
                )
            j += 32

        for offset in (16, 8, 4, 2, 1):
            off += cute.arch.shuffle_sync_bfly(off, offset=offset, mask=-1, mask_and_clamp=31)
            row_prev += cute.arch.shuffle_sync_bfly(row_prev, offset=offset, mask=-1, mask_and_clamp=31)
            row_curr += cute.arch.shuffle_sync_bfly(row_curr, offset=offset, mask=-1, mask_and_clamp=31)
            col_prev += cute.arch.shuffle_sync_bfly(col_prev, offset=offset, mask=-1, mask_and_clamp=31)
            col_curr += cute.arch.shuffle_sync_bfly(col_curr, offset=offset, mask=-1, mask_and_clamp=31)

        if item_valid and lane == 0:
            mDLogprefix[bhc, row] = cutlass.Float32(2.0) * (
                off + row_prev - col_prev + row_curr - col_curr
            )


def chunk_scan_bwd_dlogprefix_exact_cute(
    score_prev: torch.Tensor,
    score_curr: torch.Tensor,
    dSprev: torch.Tensor,
    dScurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    y_off: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Exact fp32 CuTe reduction for ``dlogprefix_half``."""
    tensors = (
        ("score_prev", score_prev),
        ("score_curr", score_curr),
        ("dSprev", dSprev),
        ("dScurr", dScurr),
        ("logprefix_half", logprefix_half),
        ("y_off", y_off),
        ("d_out_flat", d_out_flat),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("Exact CuTe dlogprefix requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError("Exact CuTe dlogprefix expects contiguous tensors.")
    if any(t.dtype != torch.float32 for _name, t in tensors):
        raise ValueError("Exact CuTe dlogprefix expects float32 tensors.")
    if score_prev.shape != score_curr.shape or score_prev.shape != dSprev.shape or score_prev.shape != dScurr.shape:
        raise ValueError("score and dS tensors must share shape.")
    if score_prev.ndim != 3 or logprefix_half.shape != score_prev.shape[:2]:
        raise ValueError("score tensors must be (BHC, L, L) and logprefix must be (BHC, L).")
    if y_off.shape != d_out_flat.shape:
        raise ValueError("y_off and d_out_flat must share shape.")
    if y_off.shape[:2] != logprefix_half.shape:
        raise ValueError("y_off/d_out_flat must be (BHC, L, P) matching logprefix.")

    out = torch.empty_like(logprefix_half)
    device_index = 0 if score_prev.device.index is None else int(score_prev.device.index)
    key: _CompiledKey = (
        device_index,
        tuple(int(x) for x in score_prev.shape),
        tuple(int(x) for x in logprefix_half.shape),
        tuple(int(x) for x in y_off.shape),
    )
    compiled = _COMPILED_DLOGPREFIX.get(key)
    if compiled is None:
        kernel = _ChunkScanBwdDLogprefixExact()
        compiled = cute.compile(
            kernel,
            from_dlpack(score_prev, assumed_align=score_prev.element_size()),
            from_dlpack(score_curr, assumed_align=score_curr.element_size()),
            from_dlpack(dSprev, assumed_align=dSprev.element_size()),
            from_dlpack(dScurr, assumed_align=dScurr.element_size()),
            from_dlpack(logprefix_half, assumed_align=logprefix_half.element_size()),
            from_dlpack(y_off, assumed_align=y_off.element_size()),
            from_dlpack(d_out_flat, assumed_align=d_out_flat.element_size()),
            from_dlpack(out, assumed_align=out.element_size()),
        )
        _COMPILED_DLOGPREFIX[key] = compiled

    compiled(
        from_dlpack(score_prev, assumed_align=score_prev.element_size()),
        from_dlpack(score_curr, assumed_align=score_curr.element_size()),
        from_dlpack(dSprev, assumed_align=dSprev.element_size()),
        from_dlpack(dScurr, assumed_align=dScurr.element_size()),
        from_dlpack(logprefix_half, assumed_align=logprefix_half.element_size()),
        from_dlpack(y_off, assumed_align=y_off.element_size()),
        from_dlpack(d_out_flat, assumed_align=d_out_flat.element_size()),
        from_dlpack(out, assumed_align=out.element_size()),
    )
    return out


__all__ = ["chunk_scan_bwd_dlogprefix_exact_cute"]
