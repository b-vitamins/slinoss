"""Parameter scan-backward for ``chunk_scan`` gradients into ``M``."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .common import prepare_chunk_scan_bwd_dout, prepare_chunk_scan_bwd_packed_context
from .db import _chunk_scan_bwd_dk_prepared_cute, prepare_chunk_scan_bwd_db_operands
from .dc import chunk_scan_bwd_dc_packed_cute
from .du import prepare_chunk_scan_bwd_du_operands

_CompiledPhaseReduceKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_CompiledPhaseScanKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_CompiledDLogprefixKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_PHASE_REDUCE: dict[_CompiledPhaseReduceKey, object] = {}
_COMPILED_PHASE_SCAN: dict[_CompiledPhaseScanKey, object] = {}
_COMPILED_DLOGPREFIX: dict[_CompiledDLogprefixKey, object] = {}


class _ChunkScanBwdDLogprefixExact:
    """Warp-cooperative exact reduction for packed ``dlogprefix_half``."""

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
        if cutlass.const_expr(
            mScorePrev.shape[0] != mLogprefix.shape[0]
            or mScorePrev.shape[1] != mLogprefix.shape[1]
        ):
            raise ValueError("logprefix must be (BHC, L) matching score tensors.")
        if cutlass.const_expr(mYOff.shape != mDOut.shape):
            raise ValueError("y_off and d_out must share shape.")
        if cutlass.const_expr(
            mYOff.shape[0] != mLogprefix.shape[0]
            or mYOff.shape[1] != mLogprefix.shape[1]
        ):
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
                s_row = cutlass.Float32(
                    cute.math.exp(cutlass.Float32(2.0) * (lp_row - lp_j))
                )
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
                s_col = cutlass.Float32(
                    cute.math.exp(cutlass.Float32(2.0) * (lp_j - lp_row))
                )
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
    key: _CompiledDLogprefixKey = (
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


class _ChunkScanParamReduce:
    """Warp-cooperative packed reduction for ``d_phase`` and ``dlogprefix``.

    Logical shape:
    - inputs: ``Q/Kprev/Kcurr/dQ/dKprev/dKcurr`` are ``(BHC, L, D)``
    - ``phase``: ``(BHC, L, 2)``
    - outputs: ``d_phase`` is ``(BHC, L, 2)`` and ``d_logprefix_half`` is ``(BHC, L)``

    Layout / launch:
    - one warp owns one ``(bhc, t)`` item
    - lanes stride over the interleaved complex feature pairs ``N = D / 2``
    - grid: linear over ``BHC * L``

    This matches the saved packed layout: ``D`` is contiguous, so a warp sees
    coalesced loads while reducing both metadata gradients directly from the
    packed backward intermediates.
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDQ: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mQ.element_type
                == mKprev.element_type
                == mKcurr.element_type
            )
        ):
            raise TypeError("Q/K inputs must share dtype.")
        if cutlass.const_expr(
            not (
                mDQ.element_type
                == mDKprev.element_type
                == mDKcurr.element_type
                == mPhase.element_type
                == mDPhase.element_type
                == mDLogprefixHalf.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("param-reduce metadata tensors must be Float32.")
        if cutlass.const_expr(mQ.shape != mKprev.shape or mQ.shape != mKcurr.shape):
            raise ValueError("Q/K tensors must share the same shape.")
        if cutlass.const_expr(mQ.shape != mDQ.shape or mQ.shape != mDKprev.shape or mQ.shape != mDKcurr.shape):
            raise ValueError("Q and dQ/dK tensors must share the same shape.")
        if cutlass.const_expr(mPhase.shape[0] != mQ.shape[0] or mPhase.shape[1] != mQ.shape[1] or mPhase.shape[2] != 2):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase shape.")
        if cutlass.const_expr(mDLogprefixHalf.shape[0] != mQ.shape[0] or mDLogprefixHalf.shape[1] != mQ.shape[1]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mQ.shape[0])
        L = cute.size(mQ.shape[1])
        warps_per_block = self.num_threads // 32
        total_items = BHC * L
        self.kernel(
            mQ,
            mKprev,
            mKcurr,
            mDQ,
            mDKprev,
            mDKcurr,
            mPhase,
            mDPhase,
            mDLogprefixHalf,
            BHC,
            L,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDQ: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        BHC: cutlass.Int32,
        L: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        bhc = item_safe // L
        t = item_safe - bhc * L

        pr = cutlass.Float32(mPhase[bhc, t, 0])
        pi = cutlass.Float32(mPhase[bhc, t, 1])
        N = cute.size(mQ.shape[2]) // 2

        acc_re = cutlass.Float32(0.0)
        acc_im = cutlass.Float32(0.0)
        acc_q = cutlass.Float32(0.0)
        acc_kp = cutlass.Float32(0.0)
        acc_kc = cutlass.Float32(0.0)
        n = lane
        while n < N:
            col = n * 2
            qr = cutlass.Float32(mQ[bhc, t, col + 0])
            qi = cutlass.Float32(mQ[bhc, t, col + 1])
            kpr = cutlass.Float32(mKprev[bhc, t, col + 0])
            kpi = cutlass.Float32(mKprev[bhc, t, col + 1])
            kcr = cutlass.Float32(mKcurr[bhc, t, col + 0])
            kci = cutlass.Float32(mKcurr[bhc, t, col + 1])

            dqr = cutlass.Float32(mDQ[bhc, t, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, t, col + 1])
            dkpr = cutlass.Float32(mDKprev[bhc, t, col + 0])
            dkpi = cutlass.Float32(mDKprev[bhc, t, col + 1])
            dkcr = cutlass.Float32(mDKcurr[bhc, t, col + 0])
            dkci = cutlass.Float32(mDKcurr[bhc, t, col + 1])

            qbr = qr * pr + qi * pi
            qbi = qi * pr - qr * pi
            kpbr = kpr * pr + kpi * pi
            kpbi = kpi * pr - kpr * pi
            kcbr = kcr * pr + kci * pi
            kcbi = kci * pr - kcr * pi

            acc_re += dqr * qbr + dqi * qbi
            acc_im += -dqr * qbi + dqi * qbr
            acc_re += dkpr * kpbr + dkpi * kpbi
            acc_im += -dkpr * kpbi + dkpi * kpbr
            acc_re += dkcr * kcbr + dkci * kcbi
            acc_im += -dkcr * kcbi + dkci * kcbr
            acc_q += dqr * qr + dqi * qi
            acc_kp += dkpr * kpr + dkpi * kpi
            acc_kc += dkcr * kcr + dkci * kci
            n += 32

        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=16, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=16, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=8, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=8, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=4, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=4, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=2, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=2, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=1, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=1, mask=-1, mask_and_clamp=31
        )
        for offset in (16, 8, 4, 2, 1):
            acc_q += cute.arch.shuffle_sync_bfly(
                acc_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_kp += cute.arch.shuffle_sync_bfly(
                acc_kp, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_kc += cute.arch.shuffle_sync_bfly(
                acc_kc, offset=offset, mask=-1, mask_and_clamp=31
            )
        if item_valid and lane == 0:
            mDPhase[bhc, t, 0] = acc_re
            mDPhase[bhc, t, 1] = acc_im
            mDLogprefixHalf[bhc, t] = cutlass.Float32(2.0) * (
                acc_q - acc_kp - acc_kc
            )


class _ChunkScanParamPhaseScan:
    """Short reverse scan from ``(phase, d_phase, d_logprefix_half)`` to ``dM``.

    Logical shape:
    - ``M_raw`` / ``phase`` / ``d_phase`` / output ``dM``: ``(BHC, L, 2)``
    - ``d_logprefix_half``: ``(BHC, L)``

    Thread layout:
    - one warp owns one ``bhc`` sequence
    - lane 0 executes the scalar SO(2) reverse scan while other lanes stay idle

    The hot packed reduction already happened in ``_ChunkScanParamReduce``.
    This kernel removes the many tiny Torch launches in the remaining exact
    reverse scan without trying to invent a tensor-core shape for scalar work.
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mMRaw: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        mDM: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mMRaw.element_type
                == mPhase.element_type
                == mDPhase.element_type
                == mDLogprefixHalf.element_type
                == mDM.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("phase-scan expects Float32 tensors.")
        if cutlass.const_expr(mMRaw.shape != mPhase.shape or mMRaw.shape != mDPhase.shape or mMRaw.shape != mDM.shape):
            raise ValueError("M_raw, phase, d_phase, and dM must share the same shape.")
        if cutlass.const_expr(mMRaw.shape[2] != 2):
            raise ValueError("M_raw/phase/d_phase/dM must be (BHC, L, 2).")
        if cutlass.const_expr(mDLogprefixHalf.shape[0] != mMRaw.shape[0] or mDLogprefixHalf.shape[1] != mMRaw.shape[1]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mMRaw.shape[0])
        warps_per_block = self.num_threads // 32
        self.kernel(
            mMRaw,
            mPhase,
            mDPhase,
            mDLogprefixHalf,
            mDM,
            BHC,
        ).launch(
            grid=[cute.ceil_div(BHC, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mMRaw: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        mDM: cute.Tensor,
        BHC: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        bhc = bidx * warps_per_block + warp
        if cute.elem_less(bhc, BHC) and lane == 0:
            L = cute.size(mMRaw.shape[1])
            eps = cutlass.Float32(1.0e-20)

            carry_re = cutlass.Float32(0.0)
            carry_im = cutlass.Float32(0.0)
            dlogr_running = cutlass.Float32(0.0)

            for t_it in cutlass.range(L, unroll=1):
                t = (L - 1) - t_it

                dlogr_running += cutlass.Float32(mDLogprefixHalf[bhc, t])

                mr = cutlass.Float32(mMRaw[bhc, t, 0])
                mi = cutlass.Float32(mMRaw[bhc, t, 1])
                mag2 = mr * mr + mi * mi
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2 + eps))
                mag = mag2 * inv_mag
                ur = mr * inv_mag
                ui = mi * inv_mag

                dpr = cutlass.Float32(mDPhase[bhc, t, 0]) + carry_re
                dpi = cutlass.Float32(mDPhase[bhc, t, 1]) + carry_im

                ppr = cutlass.Float32(1.0)
                ppi = cutlass.Float32(0.0)
                if t == 0:
                    pass
                else:
                    ppr = cutlass.Float32(mPhase[bhc, t - 1, 0])
                    ppi = cutlass.Float32(mPhase[bhc, t - 1, 1])

                # Scalar complex gradient: d_unit = grad(total wrt p_prev * unit).
                d_unit_re = dpr * ppr + dpi * ppi
                d_unit_im = -dpr * ppi + dpi * ppr
                carry_re = dpr * ur + dpi * ui
                carry_im = -dpr * ui + dpi * ur

                dot = ur * d_unit_re + ui * d_unit_im
                dphase_m_re = (d_unit_re - ur * dot) / mag
                dphase_m_im = (d_unit_im - ui * dot) / mag

                scale = cutlass.Float32(0.5) * dlogr_running / (mag2 + eps)
                dmag_m_re = scale * mr
                dmag_m_im = scale * mi

                mDM[bhc, t, 0] = dphase_m_re + dmag_m_re
                mDM[bhc, t, 1] = dphase_m_im + dmag_m_im


def _get_compiled_phase_reduce(
    Q: torch.Tensor,
    dQ: torch.Tensor,
    phase: torch.Tensor,
    d_phase: torch.Tensor,
    d_logprefix_half: torch.Tensor,
) -> object:
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    key: _CompiledPhaseReduceKey = (
        device_index,
        Q.dtype,
        dQ.dtype,
        tuple(int(x) for x in Q.shape),
        tuple(int(x) for x in phase.shape),
        tuple(int(x) for x in d_logprefix_half.shape),
        tuple(int(x) for x in d_phase.shape),
    )
    compiled = _COMPILED_PHASE_REDUCE.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanParamReduce()
    compiled = cute.compile(
        kernel,
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
    )
    _COMPILED_PHASE_REDUCE[key] = compiled
    return compiled


def _get_compiled_phase_scan(
    M_raw: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    dM: torch.Tensor,
) -> object:
    device_index = 0 if M_raw.device.index is None else int(M_raw.device.index)
    key: _CompiledPhaseScanKey = (
        device_index,
        tuple(int(x) for x in M_raw.shape),
        tuple(int(x) for x in d_logprefix_half.shape),
        tuple(int(x) for x in dM.shape),
    )
    compiled = _COMPILED_PHASE_SCAN.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanParamPhaseScan()
    compiled = cute.compile(
        kernel,
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(d_logprefix_half, assumed_align=d_logprefix_half.element_size()),
        from_dlpack(dM, assumed_align=dM.element_size()),
    )
    _COMPILED_PHASE_SCAN[key] = compiled
    return compiled

def _dlogprefix_half_packed(
    score_prev: torch.Tensor,
    score_curr: torch.Tensor,
    dSprev: torch.Tensor,
    dScurr: torch.Tensor,
    y_off: torch.Tensor,
    scale: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Exact packed-contract gradient for cumulative ``logprefix_half``.

    For the diagonal terms, each ``lp[k]`` contributes with opposite signs to
    row ``k`` and column ``k`` of the stable segment-ratio matrix. Writing that
    contribution explicitly as row-sum minus column-sum avoids building an
    autograd graph for this short metadata path.
    """

    e_prev = dSprev * score_prev * scale
    e_curr = dScurr * score_curr * scale
    return (
        2.0 * (d_out_flat * y_off).sum(dim=-1)
        + 2.0 * (e_prev.sum(dim=2) - e_prev.sum(dim=1))
        + 2.0 * (e_curr.sum(dim=2) - e_curr.sum(dim=1))
    ).contiguous()


def _packed_phase_prefix(M_raw: torch.Tensor) -> torch.Tensor:
    """Build the unit-complex phase prefix from raw packed ``M``."""

    m_c = torch.view_as_complex(M_raw.contiguous())
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    unit = m_c / mag
    return torch.cumprod(unit, dim=1)


def _chunk_scan_bwd_param_from_intermediates(
    Q_packed: torch.Tensor,
    Kprev_packed: torch.Tensor,
    Kcurr_packed: torch.Tensor,
    phase_real: torch.Tensor,
    M_raw: torch.Tensor,
    dQ: torch.Tensor,
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
) -> torch.Tensor:
    """Map packed metadata intermediates onto public ``dM``."""
    BHC, L, _ = map(int, Q_packed.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Packed leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    d_phase = torch.empty_like(phase_real)
    compiled_reduce = _get_compiled_phase_reduce(
        Q_packed,
        dQ,
        phase_real,
        d_phase,
        d_logprefix_half,
    )
    compiled_reduce(
        from_dlpack(Q_packed, assumed_align=Q_packed.element_size()),
        from_dlpack(Kprev_packed, assumed_align=Kprev_packed.element_size()),
        from_dlpack(Kcurr_packed, assumed_align=Kcurr_packed.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKcurr, assumed_align=dKcurr.element_size()),
        from_dlpack(phase_real, assumed_align=phase_real.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
    )

    dM = torch.empty_like(M_raw)
    compiled_scan = _get_compiled_phase_scan(M_raw, d_logprefix_half, dM)
    compiled_scan(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase_real, assumed_align=phase_real.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(d_logprefix_half, assumed_align=d_logprefix_half.element_size()),
        from_dlpack(dM, assumed_align=dM.element_size()),
    )
    return dM.reshape(batch_size, n_heads, T_pad, 2).to(dtype=torch.float32).contiguous()


def chunk_scan_bwd_param_scan_packed_cute(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    M_raw: torch.Tensor,
    d_out_flat: torch.Tensor,
    dQ: torch.Tensor,
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    phase_real: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T_pad: int,
) -> torch.Tensor:
    """Consume packed ``dQ/dK`` intermediates and produce public ``dM``."""

    tensors = (
        ("Q", Q),
        ("Kprev", Kprev),
        ("Vprev", Vprev),
        ("Kcurr", Kcurr),
        ("Vcurr", Vcurr),
        ("logprefix_half", logprefix_half),
        ("Z0", Z0),
        ("M_raw", M_raw),
        ("d_out_flat", d_out_flat),
        ("dQ", dQ),
        ("dKprev", dKprev),
        ("dKcurr", dKcurr),
        ("phase_real", phase_real),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q.ndim != 4 or Kprev.ndim != 4 or Kcurr.ndim != 4:
        raise ValueError("Q/K tensors must be rank-4 packed tensors.")
    if Q.shape != Kprev.shape or Q.shape != Kcurr.shape:
        raise ValueError(
            "Q, Kprev, and Kcurr must share the same packed D contract. Got "
            f"{tuple(Q.shape)}, {tuple(Kprev.shape)}, {tuple(Kcurr.shape)}."
        )
    if Vprev.shape != Vcurr.shape or Vprev.ndim != 4 or Vprev.shape[2] != 1:
        raise ValueError("Vprev/Vcurr must be packed as (BHC, L, 1, P).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError("logprefix_half must be (BHC, L) matching Q.")
    if Z0.ndim != 4 or Z0.shape[0] != Q.shape[0] or Z0.shape[2] != 1:
        raise ValueError("Z0 must be packed as (BHC, P, 1, D).")
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)}."
        )
    if d_out_flat.shape != (Q.shape[0], Q.shape[1], Vprev.shape[-1]):
        raise ValueError("d_out_flat must be (BHC, L, P) matching packed tensors.")
    if dQ.shape != Q.squeeze(2).shape:
        raise ValueError("dQ must be packed as (BHC, L, D).")
    if dKprev.shape != dQ.shape or dKcurr.shape != dQ.shape:
        raise ValueError("dKprev and dKcurr must match dQ.")
    if phase_real.shape != (*Q.shape[:2], 2):
        raise ValueError("phase_real must be (BHC, L, 2).")

    BHC, L, _, _ = map(int, Q.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    if int(T_pad) != n_chunks * L:
        raise ValueError(
            f"T_pad={T_pad} is inconsistent with packed shape {(BHC, L)} and batch*heads={BH}."
        )
    del Vprev, Vcurr, logprefix_half, Z0, d_out_flat

    Q_packed = Q.squeeze(2).contiguous()
    Kprev_packed = Kprev.squeeze(2).contiguous()
    Kcurr_packed = Kcurr.squeeze(2).contiguous()
    d_logprefix_half = torch.empty(
        (BHC, L),
        device=Q.device,
        dtype=torch.float32,
    )

    dM = _chunk_scan_bwd_param_from_intermediates(
        Q_packed,
        Kprev_packed,
        Kcurr_packed,
        phase_real,
        M_raw,
        dQ,
        dKprev,
        dKcurr,
        d_logprefix_half,
        batch_size=batch_size,
        n_heads=n_heads,
    )
    return dM

 
def chunk_scan_bwd_param_scan_cute(
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
) -> torch.Tensor:
    """Compute canonical public-contract ``dM`` for ``chunk_scan``."""
    ctx = prepare_chunk_scan_bwd_packed_context(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    _d_out_padded, d_out_flat, d_out_rev = prepare_chunk_scan_bwd_dout(
        d_out,
        ctx=ctx,
        tc_dtype=ctx.Q.dtype,
    )
    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = prepare_chunk_scan_bwd_du_operands(
        ctx.Q.contiguous(),
        ctx.Kprev.contiguous(),
        ctx.Kcurr.contiguous(),
        ctx.logprefix_half.contiguous(),
    )
    Q_rev_db, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev_db, phase_real = (
        prepare_chunk_scan_bwd_db_operands(
            ctx.Q.contiguous(),
            ctx.Vprev.contiguous(),
            ctx.Vcurr.contiguous(),
            ctx.logprefix_half.contiguous(),
            ctx.M_raw.contiguous(),
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )
    dQ = chunk_scan_bwd_dc_packed_cute(
        ctx.Vprev.contiguous(),
        ctx.Kprev.contiguous(),
        ctx.Vcurr.contiguous(),
        ctx.Kcurr.contiguous(),
        ctx.logprefix_half.contiguous(),
        ctx.Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous(),
        d_out,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )
    dKprev, dKcurr = _chunk_scan_bwd_dk_prepared_cute(
        Q_rev_db,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev_db,
        d_out_rev,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
    )
    dM = chunk_scan_bwd_param_scan_packed_cute(
        ctx.Q,
        ctx.Kprev,
        ctx.Vprev,
        ctx.Kcurr,
        ctx.Vcurr,
        ctx.logprefix_half,
        ctx.Z0,
        ctx.M_raw,
        d_out_flat,
        dQ,
        dKprev,
        dKcurr,
        phase_real,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T_pad=ctx.T_pad,
    )
    return dM[:, :, : ctx.T, :].contiguous()


chunk_scan_bwd_param_packed_cute = chunk_scan_bwd_param_scan_packed_cute
chunk_scan_bwd_param_cute = chunk_scan_bwd_param_scan_cute


__all__ = [
    "chunk_scan_bwd_dlogprefix_exact_cute",
    "chunk_scan_bwd_param_scan_cute",
    "chunk_scan_bwd_param_scan_packed_cute",
    "chunk_scan_bwd_param_cute",
    "chunk_scan_bwd_param_packed_cute",
]
