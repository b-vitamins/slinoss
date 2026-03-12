from __future__ import annotations

import cutlass
import cutlass.cute as cute


class ChunkIncrementBwdParamScanAmpere:
    """Per-chunk scan-backward for ``(dM, dKprev, dKcurr)``.

    Consumes:
      - dMsum_part: (2, L, nDtiles, BHC) fp32
      - dMp0:       (2, BHC) fp32 (from boundary kernel)
      - d_m_chunk:  (2, BHC) fp32

    And produces:
      - dM:      (2, L, BHC) fp32
      - dKprev:  (2, L, BHC) fp32
      - dKcurr:  (2, L, BHC) fp32
    """

    def __init__(self, *, chunk_size: int, nDtiles: int):
        self.L = int(chunk_size)
        self.nDtiles = int(nDtiles)
        self.num_threads = 32
        if self.num_threads != 32:
            raise ValueError("This kernel assumes one warp per CTA.")

    @cute.jit
    def __call__(
        self,
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32
        mDMsum_part: cute.Tensor,  # (2, L, nDtiles, BHC) fp32
        mDMp0: cute.Tensor,  # (2, BHC) fp32
        mDMchunk: cute.Tensor,  # (2, BHC) fp32
        mDM: cute.Tensor,  # (2, L, BHC) fp32
        mDKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDKcurr: cute.Tensor,  # (2, L, BHC) fp32
    ):
        BHC = cute.size(mM.shape[2])
        grid_x = cute.ceil_div(BHC, self.num_threads)
        self.kernel(
            mM,
            mKprev,
            mKcurr,
            mDMsum_part,
            mDMp0,
            mDMchunk,
            mDM,
            mDKprev,
            mDKcurr,
            BHC,
        ).launch(
            grid=[cute.size(grid_x), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDMsum_part: cute.Tensor,
        mDMp0: cute.Tensor,
        mDMchunk: cute.Tensor,
        mDM: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        BHC: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        b0, _, _ = cute.arch.block_idx()
        bhc = b0 * self.num_threads + tidx
        bhc_valid = cute.elem_less(bhc, BHC)
        bhc = cutlass.min(bhc, BHC - cutlass.Int32(1))

        lane = tidx

        smem = cutlass.utils.SmemAllocator()
        sSuffix = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.L, 2, self.num_threads),
                stride=(2 * self.num_threads, self.num_threads, 1),
            ),
            4,
        )

        suf_r = cutlass.Float32(1.0)
        suf_i = cutlass.Float32(0.0)
        for t_it in cutlass.range(self.L, unroll=1):
            t = (self.L - 1) - t_it
            if bhc_valid:
                sSuffix[t, 0, lane] = suf_r
                sSuffix[t, 1, lane] = suf_i

            mr = cutlass.Float32(mM[0, t, bhc])
            mi = cutlass.Float32(mM[1, t, bhc])
            nr = suf_r * mr - suf_i * mi
            ni = suf_r * mi + suf_i * mr
            suf_r = nr
            suf_i = ni

        dmp_re = cutlass.Float32(mDMp0[0, bhc])
        dmp_im = cutlass.Float32(mDMp0[1, bhc])
        carry_re = cutlass.Float32(mDMchunk[0, bhc])
        carry_im = cutlass.Float32(mDMchunk[1, bhc])

        for t in cutlass.range(self.L, unroll=1):
            sr = cutlass.Float32(sSuffix[t, 0, lane])
            si = cutlass.Float32(sSuffix[t, 1, lane])

            dmc_re = cutlass.Float32(0.0)
            dmc_im = cutlass.Float32(0.0)
            for dt in cutlass.range(self.nDtiles, unroll=1):
                dmc_re = dmc_re + cutlass.Float32(mDMsum_part[0, t, dt, bhc])
                dmc_im = dmc_im + cutlass.Float32(mDMsum_part[1, t, dt, bhc])

            dkpr_re = sr * dmp_re + si * dmp_im
            dkpr_im = sr * dmp_im - si * dmp_re
            dkcr_re = sr * dmc_re + si * dmc_im
            dkcr_im = sr * dmc_im - si * dmc_re

            kpr = cutlass.Float32(mKprev[0, t, bhc])
            kpi = cutlass.Float32(mKprev[1, t, bhc])
            kcr = cutlass.Float32(mKcurr[0, t, bhc])
            kci = cutlass.Float32(mKcurr[1, t, bhc])

            dsuf_re = (kpr * dmp_re + kpi * dmp_im) + (kcr * dmc_re + kci * dmc_im)
            dsuf_im = (kpr * dmp_im - kpi * dmp_re) + (kcr * dmc_im - kci * dmc_re)

            dm_re = carry_re * sr + carry_im * si
            dm_im = carry_im * sr - carry_re * si

            if bhc_valid:
                mDKprev[0, t, bhc] = dkpr_re
                mDKprev[1, t, bhc] = dkpr_im
                mDKcurr[0, t, bhc] = dkcr_re
                mDKcurr[1, t, bhc] = dkcr_im
                mDM[0, t, bhc] = dm_re
                mDM[1, t, bhc] = dm_im

            if t + 1 < self.L:
                mr = cutlass.Float32(mM[0, t, bhc])
                mi = cutlass.Float32(mM[1, t, bhc])
                next_re = carry_re * mr + carry_im * mi
                next_im = carry_im * mr - carry_re * mi
                carry_re = next_re + dsuf_re
                carry_im = next_im + dsuf_im

            dmp_re = dmc_re
            dmp_im = dmc_im

        return
