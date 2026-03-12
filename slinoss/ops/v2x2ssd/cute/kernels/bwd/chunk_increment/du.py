from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import _default_async_copy_bits, _default_tc_k_tile, _next_pow2


class ChunkIncrementBwdDUAmpere:
    """Ampere tensor-core kernel for ``dU_main``.

    Computes, per chunk (BHC):
      alpha_sum[t] = suffix_after[t] * K_curr[t] * B[t]
                   + suffix_after[t + 1] * K_prev[t + 1] * B[t]
      dU^T         = d_inc * alpha_sum^T          => (P, L)

    This kernel does *not* handle the boundary rank-1 term (u0 ⊗ b0); that is
    produced by a separate boundary kernel in the full backward.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        cta_tiler: tuple[int, int, int] | None = None,  # (bM=P, bN=L, bK=D)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int | None = None,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        if cta_tiler is None:
            cta_tiler = (64, self.L, _default_tc_k_tile(self.D))
        self.cta_tiler = cta_tiler
        if num_stages is None:
            num_stages = min(3, (self.D // int(cta_tiler[2])) + 1)
        self.num_stages = int(num_stages)
        self.atom_layout_mnk = atom_layout_mnk

        self.bM, self.bN, self.bK = map(int, self.cta_tiler)
        if self.bN != self.L:
            raise ValueError("This kernel assumes bN == chunk_size (single tile in N).")
        if self.bK % 16 != 0:
            raise ValueError("bK must be multiple of 16 for tensor cores.")
        if self.bK % 2 != 0:
            raise ValueError("bK must be divisible by 2 for complex-pair lanes.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2")
        if self.D % self.bK != 0:
            raise ValueError("D must be divisible by bK for this kernel.")
        if (self.D // self.bK) < (self.num_stages - 1):
            raise ValueError(
                "D/bK must be >= (num_stages-1) for the cp.async pipeline."
            )

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        atomM, atomN, atomK = self.atom_layout_mnk
        self.num_threads = atomM * atomN * atomK * 32
        self.scan_threads = _next_pow2(self.L)
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )
        if atomK != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bM % (atomM * mmaM) != 0:
            raise ValueError("bM must be divisible by atomM*mmaM.")
        if self.bN % (atomN * mmaN * 2) != 0:
            raise ValueError("bN must be divisible by atomN*mmaN*2.")
        if self.bK % mmaK != 0:
            raise ValueError("bK must be divisible by mmaK.")

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        if major_mode_size >= 64:
            if major_mode_size % 64 == 0:
                major_mode_size = 64
            elif major_mode_size % 32 == 0:
                major_mode_size = 32
            else:
                major_mode_size = 64

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy_AB(
        self, atom_copy, dtype, major_mode, copy_bits, *, tile_m: int
    ):
        copy_elems = copy_bits // dtype.width
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            value_layout = cute.make_layout((1, copy_elems))

            best_tm = None
            best_tn = None
            for tm in range(1, self.num_threads + 1):
                if self.num_threads % tm != 0:
                    continue
                tn = self.num_threads // tm
                tile_k_seg = tn * copy_elems
                if (self.bK % tile_k_seg) != 0:
                    continue
                if best_tm is None or tile_k_seg > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn

            if best_tm is None:
                raise ValueError(
                    "Failed to find a legal row-major async-copy tiling."
                )

            thread_layout = cute.make_layout((best_tm, best_tn), stride=(best_tn, 1))
            return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

        shape_dim_1 = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = (int(tile_m) + int(copy_elems) - 1) // int(copy_elems)
            if shape_dim_0 > self.num_threads:
                raise ValueError("tile_m too large for vectorized col-major copy.")

            tm = None
            for cand in range(shape_dim_0, self.num_threads + 1):
                if self.num_threads % cand == 0:
                    tm = cand
                    break
            if tm is None:
                raise ValueError(
                    "Internal error: failed to find divisor for col-major copy."
                )
            thread_layout = cute.make_layout(
                (tm, self.num_threads // tm), stride=(1, tm)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit
    def __call__(
        self,
        mDInc: cute.Tensor,  # (P, D, BHC) fp16/bf16
        mB: cute.Tensor,  # (L, D, BHC) fp16/bf16
        mM: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mKprev: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mDU: cute.Tensor,  # (P, L, BHC) fp16/bf16
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mDInc)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mDU)

        a_copy_bits = _default_async_copy_bits(
            dtype_width=mDInc.element_type.width,
            major_mode=self.a_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        b_copy_bits = _default_async_copy_bits(
            dtype_width=mB.element_type.width,
            major_mode=self.b_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        sA_layout = self._make_smem_layout_AB(
            mDInc.element_type,
            self.a_major_mode,
            a_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            b_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))

        smem_size_AB = cute.size_in_bytes(
            mDInc.element_type, sA_layout
        ) + cute.size_in_bytes(mB.element_type, sB_layout)
        smem_size_C = cute.size_in_bytes(self.c_dtype, sC_layout)

        msum_bytes = self.L * 2 * 4
        mprev_bytes = self.L * 2 * 4
        scan_scratch_bytes = (32 + 32 + 32 + 32) * 4
        pad_bytes = max(0, int(smem_size_C) - int(smem_size_AB))
        guard_bytes = 2048 + ((pad_bytes + 3) // 4) * 4
        guard_elems = int(guard_bytes // 4)
        sGuard_layout = cute.make_layout((guard_elems,), stride=(1,))
        extra_bytes = guard_bytes + msum_bytes + mprev_bytes + scan_scratch_bytes + 256
        smem_needed = smem_size_AB + extra_bytes

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mDInc.element_type,
            num_bits_per_copy=a_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            mDInc.element_type,
            self.a_major_mode,
            a_copy_bits,
            tile_m=self.bM,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mB.element_type,
            num_bits_per_copy=b_copy_bits,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy_B,
            mB.element_type,
            self.b_major_mode,
            b_copy_bits,
            tile_m=self.bN,
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        grid_x = cute.ceil_div(self.P, self.bM)
        grid_y = cute.ceil_div(self.L, self.bN)
        grid_z = cute.size(mDInc.shape[2])

        self.kernel(
            mDInc,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDU,
            sA_layout,
            sB_layout,
            sC_layout,
            sGuard_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=(cute.size(grid_x), cute.size(grid_y), grid_z),
            block=[self.num_threads, 1, 1],
            smem=smem_needed,
        )

    @cute.kernel
    def kernel(
        self,
        mDInc: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDU: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        sGuard_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        tiler_coord = (bidx, bidy, None)

        gA = cute.local_tile(
            mDInc[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mDInc.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Float32), sC_layout
        )
        _guard = smem.allocate_tensor(cutlass.Float32, sGuard_layout, 16)
        warp_re_total = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4
        )
        warp_im_total = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4
        )
        warp_re_offset = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4
        )
        warp_im_offset = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), 4
        )
        s_Msum = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.L, 2), stride=(2, 1)), 4
        )
        s_Mprev = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.L, 2), stride=(2, 1)), 4
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)

        tAgA = thr_copy_A.partition_S(gA)
        tBgB = thr_copy_B.partition_S(gB)
        tAsA = thr_copy_A.partition_D(sA)
        tBsB = thr_copy_B.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(
            cute.local_tile(
                mDU[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )
        )

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mDInc.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mB.element_type,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
        thr_copy_ld_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ld_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy = thr_copy_ld_A.partition_S(sA)
        tCrA_copy = thr_copy_ld_A.retile(tCrA)
        tCsB_copy = thr_copy_ld_B.partition_S(sB)
        tCrB_copy = thr_copy_ld_B.retile(tCrB)

        mcA = cute.make_identity_tensor(mDInc.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mDInc.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        t = cutlass.Int32(self.L - 1) - tidx

        mr = cutlass.Float32(0.0)
        mi = cutlass.Float32(0.0)
        kpr = cutlass.Float32(0.0)
        kpi = cutlass.Float32(0.0)
        kcr = cutlass.Float32(0.0)
        kci = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            mr = cutlass.Float32(mM[0, t, bidz].to(cutlass.Float32))
            mi = cutlass.Float32(mM[1, t, bidz].to(cutlass.Float32))
            kpr = cutlass.Float32(mKprev[0, t, bidz].to(cutlass.Float32))
            kpi = cutlass.Float32(mKprev[1, t, bidz].to(cutlass.Float32))
            kcr = cutlass.Float32(mKcurr[0, t, bidz].to(cutlass.Float32))
            kci = cutlass.Float32(mKcurr[1, t, bidz].to(cutlass.Float32))

        scan_threads = cutlass.Int32(self.scan_threads)
        num_warps_scan = cutlass.Int32(self.scan_threads // 32)
        in_scan = tidx < scan_threads

        qr = cutlass.select_(in_scan, mr, cutlass.Float32(1.0))
        qi = cutlass.select_(in_scan, mi, cutlass.Float32(0.0))

        if warp < num_warps_scan:
            for offset in (1, 2, 4, 8, 16):
                orr = cute.arch.shuffle_sync_up(
                    qr, offset=offset, mask=-1, mask_and_clamp=0
                )
                oii = cute.arch.shuffle_sync_up(
                    qi, offset=offset, mask=-1, mask_and_clamp=0
                )
                pred = lane >= cutlass.Int32(offset)
                nr = orr * qr - oii * qi
                ni = orr * qi + oii * qr
                qr = cutlass.select_(pred, nr, qr)
                qi = cutlass.select_(pred, ni, qi)

            if lane == cutlass.Int32(31):
                warp_re_total[warp] = qr
                warp_im_total[warp] = qi

        cute.arch.sync_threads()

        if cutlass.const_expr(self.scan_threads > 32):
            if warp == cutlass.Int32(0):
                w = lane
                has_warp = w < num_warps_scan
                wr = cutlass.select_(has_warp, warp_re_total[w], cutlass.Float32(1.0))
                wi = cutlass.select_(has_warp, warp_im_total[w], cutlass.Float32(0.0))

                for offset in (1, 2, 4, 8, 16):
                    orr = cute.arch.shuffle_sync_up(
                        wr, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    oii = cute.arch.shuffle_sync_up(
                        wi, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    pred = lane >= cutlass.Int32(offset)
                    nr = orr * wr - oii * wi
                    ni = orr * wi + oii * wr
                    wr = cutlass.select_(pred, nr, wr)
                    wi = cutlass.select_(pred, ni, wi)

                off_r = cute.arch.shuffle_sync_up(
                    wr, offset=1, mask=-1, mask_and_clamp=0
                )
                off_i = cute.arch.shuffle_sync_up(
                    wi, offset=1, mask=-1, mask_and_clamp=0
                )
                is0 = lane == cutlass.Int32(0)
                off_r = cutlass.select_(is0, cutlass.Float32(1.0), off_r)
                off_i = cutlass.select_(is0, cutlass.Float32(0.0), off_i)

                if has_warp:
                    warp_re_offset[w] = off_r
                    warp_im_offset[w] = off_i

            cute.arch.sync_threads()

            if warp < num_warps_scan:
                off_r = warp_re_offset[warp]
                off_i = warp_im_offset[warp]
                nr = off_r * qr - off_i * qi
                ni = off_r * qi + off_i * qr
                qr, qi = nr, ni

        if warp < num_warps_scan and lane == cutlass.Int32(31):
            warp_re_total[warp] = qr
            warp_im_total[warp] = qi

        cute.arch.sync_threads()

        suf_r = cutlass.Float32(1.0)
        suf_i = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            r_prev = cute.arch.shuffle_sync_up(
                qr, offset=1, mask=-1, mask_and_clamp=0
            )
            i_prev = cute.arch.shuffle_sync_up(
                qi, offset=1, mask=-1, mask_and_clamp=0
            )
            if tidx == cutlass.Int32(0):
                suf_r = cutlass.Float32(1.0)
                suf_i = cutlass.Float32(0.0)
            else:
                if lane == cutlass.Int32(0):
                    suf_r = warp_re_total[warp - 1]
                    suf_i = warp_im_total[warp - 1]
                else:
                    suf_r = r_prev
                    suf_i = i_prev

            mp_r = suf_r * kpr - suf_i * kpi
            mp_i = suf_r * kpi + suf_i * kpr
            mc_r = suf_r * kcr - suf_i * kci
            mc_i = suf_r * kci + suf_i * kcr
            s_Mprev[t, 0] = mp_r
            s_Mprev[t, 1] = mp_i
            s_Msum[t, 0] = mc_r
            s_Msum[t, 1] = mc_i

        cute.arch.sync_threads()

        if tidx < cutlass.Int32(self.L):
            if t < cutlass.Int32(self.L - 1):
                s_Msum[t, 0] = s_Msum[t, 0] + s_Mprev[t + 1, 0]
                s_Msum[t, 1] = s_Msum[t, 1] + s_Mprev[t + 1, 1]

        cute.arch.sync_threads()

        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        k_tile_count = self.D // self.bK
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for kk in range(cute.size(tAsA, mode=[2])):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, kk, k_tile_index],
                tAsA[None, None, kk, 0],
                pred=tApA[None, None, kk],
            )
        for kk in range(cute.size(tBsB, mode=[2])):
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, kk, k_tile_index],
                tBsB[None, None, kk, 0],
                pred=tBpB[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(num_smem_stages - 1)
        num_k_block = cute.size(tCrA, mode=[2])

        nvec_full = self.bK // 2
        total_full = self.bN * nvec_full
        num_iters_full = (total_full + self.num_threads - 1) // self.num_threads

        for kt in range(k_tile_count):
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            for it in cutlass.range_constexpr(num_iters_full):
                idx = tidx + it * self.num_threads
                if cute.elem_less(idx, total_full):
                    tt = idx // nvec_full
                    v = idx - tt * nvec_full
                    d0 = v * 2

                    vx = sB[tt, d0 + 0, smem_pipe_read].to(cutlass.Float32)
                    vy = sB[tt, d0 + 1, smem_pipe_read].to(cutlass.Float32)
                    ar = s_Msum[tt, 0]
                    ai = s_Msum[tt, 1]

                    rx = ar * vx - ai * vy
                    ry = ar * vy + ai * vx

                    sB[tt, d0 + 0, smem_pipe_read] = rx.to(mB.element_type)
                    sB[tt, d0 + 1, smem_pipe_read] = ry.to(mB.element_type)

            cute.arch.sync_threads()

            next_tile = kt + (num_smem_stages - 1)
            if next_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, smem_pipe_write],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy[None, None, 0])
            for kb in cutlass.range(num_k_block, unroll_full=True):
                kb_next = (kb + 1) % num_k_block
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, kb_next],
                    tCrA_copy[None, None, kb_next],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, kb_next],
                    tCrB_copy[None, None, kb_next],
                )
                cute.gemm(
                    tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC
                )

            smem_pipe_write = smem_pipe_read
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == num_smem_stages:
                smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.autovec_copy(tCrC, tCsC)
        cute.arch.sync_threads()

        tile_m0 = bidx * self.bM
        tile_n0 = bidy * self.bN
        total_elems = self.bM * self.bN
        idx = tidx
        while cute.elem_less(idx, total_elems):
            m = idx // self.bN
            n = idx - (m * self.bN)
            g_m = tile_m0 + m
            g_n = tile_n0 + n
            if cute.elem_less(g_m, self.P) and cute.elem_less(g_n, self.L):
                mDU[g_m, g_n, bidz] = sC[m, n].to(mDU.element_type)
            idx = idx + self.num_threads

        return
