from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .common import _TileConfig


class StatePassingBwdStateAmpere:
    """Backward kernel for (d_inc, d_initial) (no reductions)."""

    def __init__(
        self,
        cfg: _TileConfig,
        *,
        copy_bits_in: int,
        copy_bits_out: int,
    ):
        self.cfg = cfg
        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)

    @cute.jit
    def __call__(
        self,
        d_chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        d_final: cute.Tensor,  # (B,H,P,D) fp32
        m_chunk: cute.Tensor,  # (B,H,C,2) fp32
        d_inc: cute.Tensor,  # (B,H,C,P,D) fp32
        d_initial: cute.Tensor,  # (B,H,P,D) fp32
    ):
        B, H, C, P, D = d_inc.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bs = cute.make_layout((BH, S), stride=(S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))

        dstarts_flat = cute.make_tensor(d_chunk_starts.iterator, layout_bcs)
        dfinal_flat = cute.make_tensor(d_final.iterator, layout_bs)
        m_flat = cute.make_tensor(m_chunk.iterator, layout_bcm)
        dinc_flat = cute.make_tensor(d_inc.iterator, layout_bcs)
        dinitial_flat = cute.make_tensor(d_initial.iterator, layout_bs)

        tv_layout = cute.make_layout(
            (self.cfg.num_threads, self.cfg.elems_per_thread),
            stride=(self.cfg.elems_per_thread, 1),
        )
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.cfg.tile))

        grid_x = cute.ceil_div(S, self.cfg.tile)
        grid_y = BH

        self.kernel(
            dstarts_flat,
            dfinal_flat,
            m_flat,
            dinc_flat,
            dinitial_flat,
            cS,
            tv_layout,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        dstarts_flat: cute.Tensor,  # (BH, C, S)
        dfinal_flat: cute.Tensor,  # (BH, S)
        m_flat: cute.Tensor,  # (BH, C, 2)
        dinc_flat: cute.Tensor,  # (BH, C, S)
        dinitial_flat: cute.Tensor,  # (BH, S)
        cS: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tile_idx, bh, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()

        S = dinc_flat.shape[2]
        C = dinc_flat.shape[1]

        tile_start = cutlass.Int32(self.cfg.tile) * tile_idx
        residue = S - tile_start
        is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.cfg.tile))

        cta_coord = (None, tile_idx)
        ctaCrd = cS[cta_coord]
        tidCrd = cute.composition(ctaCrd, tv_layout)
        thrCrd = tidCrd[tidx, None]

        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        frgPred.fill(cutlass.Boolean(True))
        if is_partial_tile:
            for i in cutlass.range_constexpr(cute.size(frgPred)):
                frgPred[i] = cute.elem_less(thrCrd[i], S)

        copy_in_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dstarts_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_in_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dstarts_flat.element_type,
            num_bits_per_copy=dstarts_flat.element_type.width,
        )
        copy_out_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=self.copy_bits_out,
        )
        copy_out_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dinc_flat.element_type,
            num_bits_per_copy=dinc_flat.element_type.width,
        )
        copy_df_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dfinal_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_df_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dfinal_flat.element_type,
            num_bits_per_copy=dfinal_flat.element_type.width,
        )

        copy_m = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_flat.element_type,
            num_bits_per_copy=m_flat.element_type.width * 2,
        )

        tile_layout = cute.make_layout(cS.shape[0])

        gF = dfinal_flat[bh, None]
        tF = cute.zipped_divide(gF, tiler=tile_layout)
        ctaF = tF[cta_coord]
        tidF = cute.composition(ctaF, tv_layout)
        thrF = tidF[tidx, None]

        accG = cute.make_rmem_tensor(thrF.shape, cutlass.Float32)
        accG.fill(0.0)
        frgF = cute.make_rmem_tensor_like(thrF)
        frgF.fill(0)
        if is_partial_tile:
            cute.copy(copy_df_scalar, thrF, frgF, pred=frgPred)
        else:
            cute.copy(copy_df_vec, thrF, frgF)
        accG.store(frgF.load().to(cutlass.Float32))

        frgIn = cute.make_rmem_tensor_like(thrF)
        pairs_per_thread = cute.size(accG) // 2

        for c_it in cutlass.range(C, unroll=1):
            c = C - 1 - c_it

            gOut = dinc_flat[bh, c, None]
            tOut = cute.zipped_divide(gOut, tiler=tile_layout)
            ctaOut = tOut[cta_coord]
            tidOut = cute.composition(ctaOut, tv_layout)
            thrOut = tidOut[tidx, None]

            frgTmp = cute.make_rmem_tensor_like(thrOut)
            frgTmp.store(accG.load().to(dinc_flat.element_type))
            if is_partial_tile:
                cute.copy(copy_out_scalar, frgTmp, thrOut, pred=frgPred)
            else:
                cute.copy(copy_out_vec, frgTmp, thrOut)

            gIn = dstarts_flat[bh, c, None]
            tIn = cute.zipped_divide(gIn, tiler=tile_layout)
            ctaIn = tIn[cta_coord]
            tidIn = cute.composition(ctaIn, tv_layout)
            thrIn = tidIn[tidx, None]

            frgIn.fill(0)
            if is_partial_tile:
                cute.copy(copy_in_scalar, thrIn, frgIn, pred=frgPred)
            else:
                cute.copy(copy_in_vec, thrIn, frgIn)
            dstart_f32 = frgIn.load().to(cutlass.Float32)

            mr = cutlass.Float32(0.0)
            mi = cutlass.Float32(0.0)
            if lane == cutlass.Int32(0):
                gM = m_flat[bh, c, None]
                frgM = cute.make_rmem_tensor_like(gM)
                cute.copy(copy_m, gM, frgM)
                m = frgM.load().to(cutlass.Float32)
                mr = m[0]
                mi = m[1]
            for offset in (1, 2, 4, 8, 16):
                mr += cute.arch.shuffle_sync_bfly(
                    mr, offset=offset, mask=-1, mask_and_clamp=31
                )
                mi += cute.arch.shuffle_sync_bfly(
                    mi, offset=offset, mask=-1, mask_and_clamp=31
                )

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                gr = accG[base + 0]
                gi = accG[base + 1]

                rr = mr * gr + mi * gi
                ri = mr * gi - mi * gr

                accG[base + 0] = rr + dstart_f32[base + 0]
                accG[base + 1] = ri + dstart_f32[base + 1]

        gI = dinitial_flat[bh, None]
        tI = cute.zipped_divide(gI, tiler=tile_layout)
        ctaI = tI[cta_coord]
        tidI = cute.composition(ctaI, tv_layout)
        thrI = tidI[tidx, None]

        frgTmp = cute.make_rmem_tensor_like(thrI)
        frgTmp.store(accG.load().to(dinitial_flat.element_type))
        if is_partial_tile:
            cute.copy(copy_out_scalar, frgTmp, thrI, pred=frgPred)
        else:
            cute.copy(copy_out_vec, frgTmp, thrI)

        return
