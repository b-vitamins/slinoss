from __future__ import annotations

import cutlass
import cutlass.cute as cute


class ChunkIncrementBwdBoundaryAmpere:
    """Boundary kernel for the per-chunk ``u_prev0 ⊗ b_prev_decay0`` term.

    For each chunk (BHC), ``chunk_increment`` contains the rank-1 contribution:

      inc += u_boundary ⊗ (Mp0 * b_prev_input)

    where:
      - ``u_boundary`` is ``U_prev0`` for the first chunk, else the previous
        chunk's last ``U``,
      - ``b_prev_input`` is ``B_prev0`` for the first chunk, else the previous
        chunk's last ``B``,
      - ``Mp0`` is the step-0 prev-tap + suffix transform scalar for the chunk.

    This kernel computes:
      - ``dU_boundary``   (P)   (same dtype as U/B)
      - ``dB_prev_input`` (D)   (same dtype as U/B)
      - ``dMp0``          (2)   (fp32 packed complex), to be consumed by the
        parameter scan-backward kernel.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        num_threads: int = 192,
    ):
        self.ab_dtype = dtype
        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.num_threads = int(num_threads)
        self.du_prev_threads = 64
        self.db_prev_threads = self.num_threads - self.du_prev_threads
        self.async_copy_bits = 128
        if self.L <= 64:
            self.scan_threads = 64
        elif self.L <= 128:
            self.scan_threads = 128
        else:
            self.scan_threads = 0
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads <= 64:
            raise ValueError("num_threads must be > 64.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.db_prev_threads <= 0:
            raise ValueError("num_threads must leave room for the dB_prev workers.")
        if self.scan_threads > self.num_threads:
            raise ValueError("num_threads must cover the configured suffix scan.")

    @cute.jit
    def __call__(
        self,
        mDInc: cute.Tensor,  # (BHC, P, D) fp16/bf16
        mBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mM: cute.Tensor,  # (2, L, BHC) fp32
        mKprev: cute.Tensor,  # (2, L, BHC) fp32
        mDUPrev: cute.Tensor,  # (P, BHC) fp16/bf16
        mDBPrev: cute.Tensor,  # (D, BHC) fp16/bf16
        mDMp0: cute.Tensor,  # (2, BHC) fp32
    ):
        grid_x = cute.size(mDInc.shape[0])

        self.kernel(
            mDInc,
            mBPrev,
            mUPrev,
            mM,
            mKprev,
            mDUPrev,
            mDBPrev,
            mDMp0,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDInc: cute.Tensor,  # (BHC, P, D)
        mBPrev: cute.Tensor,  # (D, BHC)
        mUPrev: cute.Tensor,  # (P, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mDUPrev: cute.Tensor,  # (P, BHC)
        mDBPrev: cute.Tensor,  # (D, BHC)
        mDMp0: cute.Tensor,  # (2, BHC)
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bhc, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        smem = cutlass.utils.SmemAllocator()
        async_copy_bits = self.async_copy_bits
        async_elems = async_copy_bits // mDInc.element_type.width
        fast_tiled = (
            self.P <= self.du_prev_threads
            and self.D >= 128
            and self.D % async_elems == 0
        )
        s_Mp0 = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((2,), stride=(1,)), 8
        )
        warp_re_total = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((4,), stride=(1,)), 4
        )
        warp_im_total = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((4,), stride=(1,)), 4
        )
        s_u_prev = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.P,), stride=(1,)), 4
        )
        p = tidx
        while cute.elem_less(p, self.P):
            s_u_prev[p] = mUPrev[p, bhc].to(cutlass.Float32)
            p = p + self.num_threads

        if cutlass.const_expr(fast_tiled):
            d_stage = 128
            d_smem_stride = d_stage + async_elems
            s_b_decay = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((d_stage,), stride=(1,)), 4
            )
            s_dinc = smem.allocate_tensor(
                mDInc.element_type,
                cute.make_layout((self.P, d_smem_stride), stride=(d_smem_stride, 1)),
                16,
            )
        else:
            d_stage = self.D
            d_smem_stride = (
                (self.D + async_elems - 1) // async_elems
            ) * async_elems + async_elems
            s_b_decay = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((self.D,), stride=(1,)), 4
            )
            s_dinc = smem.allocate_tensor(
                mDInc.element_type,
                cute.make_layout((self.P, d_smem_stride), stride=(d_smem_stride, 1)),
                16,
            )
        d_vec = self.D // async_elems
        mDInc_vec = cute.make_tensor(
            mDInc.iterator,
            cute.make_layout(
                (mDInc.shape[0], mDInc.shape[1], d_vec, async_elems),
                stride=(self.P * self.D, self.D, async_elems, 1),
            ),
        )
        s_dinc_vec = cute.make_tensor(
            s_dinc.iterator,
            cute.make_layout(
                (self.P, d_smem_stride // async_elems, async_elems),
                stride=(d_smem_stride, async_elems, 1),
            ),
        )
        atom_async_copy_dinc = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mDInc.element_type,
            num_bits_per_copy=async_copy_bits,
        )
        if cutlass.const_expr(not fast_tiled):
            p = tidx
            while cute.elem_less(p, self.P):
                dv = 0
                while cute.elem_less(dv, d_vec):
                    cute.copy(
                        atom_async_copy_dinc,
                        mDInc_vec[bhc, p, dv, None],
                        s_dinc_vec[p, dv, None],
                    )
                    dv = dv + 1
                d = d_vec * async_elems
                while cute.elem_less(d, self.D):
                    s_dinc[p, d] = mDInc[bhc, p, d]
                    d = d + 1
                p = p + self.num_threads
            cute.arch.cp_async_commit_group()

        if tidx == 0:
            for w in cutlass.range(4, unroll=1):
                warp_re_total[w] = cutlass.Float32(1.0)
                warp_im_total[w] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        if self.scan_threads != 0 and cute.elem_less(
            tidx, cutlass.Int32(self.scan_threads)
        ):
            t = (self.L - 1) - tidx
            qr = cutlass.Float32(1.0)
            qi = cutlass.Float32(0.0)
            if cute.elem_less(cutlass.Int32(0), t):
                qr = cutlass.Float32(mM[0, t, bhc])
                qi = cutlass.Float32(mM[1, t, bhc])

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

            if lane == cutlass.Int32(31) and cute.elem_less(
                warp, cutlass.Int32(self.scan_threads // 32)
            ):
                warp_re_total[warp] = qr
                warp_im_total[warp] = qi

        cute.arch.sync_threads()

        if tidx == 0:
            suf_r = cutlass.Float32(1.0)
            suf_i = cutlass.Float32(0.0)

            if self.scan_threads != 0:
                num_warps = self.scan_threads // 32
                for w in cutlass.range(num_warps, unroll=1):
                    orr = warp_re_total[w]
                    oii = warp_im_total[w]
                    nr = suf_r * orr - suf_i * oii
                    ni = suf_r * oii + suf_i * orr
                    suf_r = nr
                    suf_i = ni
            else:
                for t_it in cutlass.range(self.L - 1, unroll=1):
                    t = (self.L - 1) - t_it
                    mr = cutlass.Float32(mM[0, t, bhc])
                    mi = cutlass.Float32(mM[1, t, bhc])
                    nr = suf_r * mr - suf_i * mi
                    ni = suf_r * mi + suf_i * mr
                    suf_r = nr
                    suf_i = ni

            kpr0 = cutlass.Float32(mKprev[0, 0, bhc])
            kpi0 = cutlass.Float32(mKprev[1, 0, bhc])
            s_Mp0[0] = suf_r * kpr0 - suf_i * kpi0
            s_Mp0[1] = suf_r * kpi0 + suf_i * kpr0

        cute.arch.sync_threads()

        mp0r = s_Mp0[0]
        mp0i = s_Mp0[1]

        nvec = self.D // 2
        if cutlass.const_expr(fast_tiled):
            acc_du = cutlass.Float32(0.0)
            p_row = tidx
            db_m0 = cutlass.Float32(0.0)
            db_m1 = cutlass.Float32(0.0)
            d_stage_vec = d_stage // async_elems
            n_d_tiles = (self.D + d_stage - 1) // d_stage
            for d_tile in cutlass.range_constexpr(n_d_tiles):
                d_base = cutlass.Int32(d_tile * d_stage)
                d_tile_width = min(d_stage, self.D - d_tile * d_stage)
                d_tile_vec = d_tile_width // async_elems
                copy_task = tidx
                total_copy_tasks = cutlass.Int32(self.P * d_tile_vec)
                while cute.elem_less(copy_task, total_copy_tasks):
                    p = copy_task // cutlass.Int32(d_tile_vec)
                    dv = copy_task - p * cutlass.Int32(d_tile_vec)
                    cute.copy(
                        atom_async_copy_dinc,
                        mDInc_vec[bhc, p, d_tile * d_stage_vec + dv, None],
                        s_dinc_vec[p, dv, None],
                    )
                    copy_task = copy_task + self.num_threads
                cute.arch.cp_async_commit_group()

                v = tidx
                while cute.elem_less(v, cutlass.Int32(d_tile_width // 2)):
                    d0 = v * 2
                    bx = mBPrev[d_base + d0 + 0, bhc].to(cutlass.Float32)
                    by = mBPrev[d_base + d0 + 1, bhc].to(cutlass.Float32)
                    s_b_decay[d0 + 0] = mp0r * bx - mp0i * by
                    s_b_decay[d0 + 1] = mp0r * by + mp0i * bx
                    v = v + self.num_threads

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                if cute.elem_less(
                    tidx, cutlass.Int32(self.du_prev_threads)
                ) and cute.elem_less(p_row, self.P):
                    d = 0
                    while cute.elem_less(d, cutlass.Int32(d_tile_width)):
                        acc_du = (
                            acc_du + s_dinc[p_row, d].to(cutlass.Float32) * s_b_decay[d]
                        )
                        d = d + 1

                if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                    db_local = tidx - self.du_prev_threads
                    pair = db_local // 2
                    half = db_local - pair * 2
                    if cute.elem_less(pair, cutlass.Int32(d_tile_width // 2)):
                        d0 = pair * 2
                        g0 = cutlass.Float32(0.0)
                        g1 = cutlass.Float32(0.0)
                        p_base = half * 32
                        for p_off in cutlass.range(32, unroll=1):
                            p = cutlass.Int32(p_base + p_off)
                            if cute.elem_less(p, self.P):
                                up = s_u_prev[p]
                                g0 = g0 + up * s_dinc[p, d0 + 0].to(cutlass.Float32)
                                g1 = g1 + up * s_dinc[p, d0 + 1].to(cutlass.Float32)
                        g0 = g0 + cute.arch.shuffle_sync_bfly(
                            g0, offset=1, mask=-1, mask_and_clamp=31
                        )
                        g1 = g1 + cute.arch.shuffle_sync_bfly(
                            g1, offset=1, mask=-1, mask_and_clamp=31
                        )
                        if half == cutlass.Int32(0):
                            mDBPrev[d_base + d0 + 0, bhc] = (mp0r * g0 + mp0i * g1).to(
                                mDBPrev.element_type
                            )
                            mDBPrev[d_base + d0 + 1, bhc] = (mp0r * g1 - mp0i * g0).to(
                                mDBPrev.element_type
                            )

                            bx = mBPrev[d_base + d0 + 0, bhc].to(cutlass.Float32)
                            by = mBPrev[d_base + d0 + 1, bhc].to(cutlass.Float32)
                            db_m0 = db_m0 + (g0 * bx + g1 * by)
                            db_m1 = db_m1 + (bx * g1 - by * g0)

                cute.arch.sync_threads()

            if cute.elem_less(
                tidx, cutlass.Int32(self.du_prev_threads)
            ) and cute.elem_less(p_row, self.P):
                mDUPrev[p_row, bhc] = acc_du.to(mDUPrev.element_type)

            if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                for off in (16, 8, 4, 2, 1):
                    db_m0 = db_m0 + cute.arch.shuffle_sync_bfly(
                        db_m0, offset=off, mask=-1, mask_and_clamp=31
                    )
                    db_m1 = db_m1 + cute.arch.shuffle_sync_bfly(
                        db_m1, offset=off, mask=-1, mask_and_clamp=31
                    )
                if lane == cutlass.Int32(0):
                    warp_re_total[warp - cutlass.Int32(self.du_prev_threads // 32)] = (
                        db_m0
                    )
                    warp_im_total[warp - cutlass.Int32(self.du_prev_threads // 32)] = (
                        db_m1
                    )
        else:
            v = tidx
            while cute.elem_less(v, nvec):
                d0 = v * 2
                bx = mBPrev[d0 + 0, bhc].to(cutlass.Float32)
                by = mBPrev[d0 + 1, bhc].to(cutlass.Float32)
                s_b_decay[d0 + 0] = mp0r * bx - mp0i * by
                s_b_decay[d0 + 1] = mp0r * by + mp0i * bx
                v = v + self.num_threads

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            if cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                p = tidx
                while cute.elem_less(p, self.P):
                    acc = cutlass.Float32(0.0)
                    d = 0
                    while cute.elem_less(d, self.D):
                        acc = acc + s_dinc[p, d].to(cutlass.Float32) * s_b_decay[d]
                        d = d + 1
                    mDUPrev[p, bhc] = acc.to(mDUPrev.element_type)
                    p = p + self.du_prev_threads

            if not cute.elem_less(tidx, cutlass.Int32(self.du_prev_threads)):
                v = tidx - self.du_prev_threads
                v_stride = cutlass.Int32(self.db_prev_threads)
                m0 = cutlass.Float32(0.0)
                m1 = cutlass.Float32(0.0)
                while cute.elem_less(v, nvec):
                    d0 = v * 2
                    g0 = cutlass.Float32(0.0)
                    g1 = cutlass.Float32(0.0)
                    for p in cutlass.range(self.P, unroll=1):
                        up = s_u_prev[p]
                        g0 = g0 + up * s_dinc[p, d0 + 0].to(cutlass.Float32)
                        g1 = g1 + up * s_dinc[p, d0 + 1].to(cutlass.Float32)
                    mDBPrev[d0 + 0, bhc] = (mp0r * g0 + mp0i * g1).to(
                        mDBPrev.element_type
                    )
                    mDBPrev[d0 + 1, bhc] = (mp0r * g1 - mp0i * g0).to(
                        mDBPrev.element_type
                    )

                    bx = mBPrev[d0 + 0, bhc].to(cutlass.Float32)
                    by = mBPrev[d0 + 1, bhc].to(cutlass.Float32)
                    m0 = m0 + (g0 * bx + g1 * by)
                    m1 = m1 + (bx * g1 - by * g0)
                    v = v + v_stride

                for off in (16, 8, 4, 2, 1):
                    m0 = m0 + cute.arch.shuffle_sync_bfly(
                        m0, offset=off, mask=-1, mask_and_clamp=31
                    )
                    m1 = m1 + cute.arch.shuffle_sync_bfly(
                        m1, offset=off, mask=-1, mask_and_clamp=31
                    )
                if lane == cutlass.Int32(0):
                    warp_re_total[warp - cutlass.Int32(self.du_prev_threads // 32)] = m0
                    warp_im_total[warp - cutlass.Int32(self.du_prev_threads // 32)] = m1

        cute.arch.sync_threads()

        if tidx == 0:
            m0 = cutlass.Float32(0.0)
            m1 = cutlass.Float32(0.0)
            for w in cutlass.range(self.db_prev_threads // 32, unroll=1):
                m0 = m0 + warp_re_total[w]
                m1 = m1 + warp_im_total[w]
            mDMp0[0, bhc] = m0
            mDMp0[1, bhc] = m1
