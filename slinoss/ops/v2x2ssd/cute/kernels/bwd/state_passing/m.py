from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .common import _TileConfig


class StatePassingBwdMAmpere:
    """Backward kernel for d_m_chunk (reduction over S)."""

    def __init__(
        self,
        cfg: _TileConfig,
        *,
        copy_bits_in: int,
    ):
        self.cfg = cfg
        self.copy_bits_in = int(copy_bits_in)

    @cute.jit
    def __call__(
        self,
        chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        d_inc: cute.Tensor,  # (B,H,C,P,D) fp32
        m_chunk: cute.Tensor,  # (B,H,C,2) fp32
        d_m_chunk: cute.Tensor,  # (B,H,C,2) fp32
    ):
        B, H, C, P, D = chunk_starts.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))

        starts_flat = cute.make_tensor(chunk_starts.iterator, layout_bcs)
        dinc_flat = cute.make_tensor(d_inc.iterator, layout_bcs)
        m_flat = cute.make_tensor(m_chunk.iterator, layout_bcm)
        dm_flat = cute.make_tensor(d_m_chunk.iterator, layout_bcm)

        tv_layout = cute.make_layout(
            (self.cfg.num_threads, self.cfg.elems_per_thread),
            stride=(self.cfg.elems_per_thread, 1),
        )
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.cfg.tile))

        grid_x = C
        grid_y = BH
        self.kernel(
            starts_flat,
            dinc_flat,
            m_flat,
            dm_flat,
            cS,
            tv_layout,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        starts_flat: cute.Tensor,  # (BH, C, S)
        dinc_flat: cute.Tensor,  # (BH, C, S)
        m_flat: cute.Tensor,  # (BH, C, 2)
        dm_flat: cute.Tensor,  # (BH, C, 2)
        cS: cute.Tensor,
        tv_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, bh, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        S = starts_flat.shape[2]

        copy_m = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_flat.element_type,
            num_bits_per_copy=m_flat.element_type.width * 2,
        )
        gM = m_flat[bh, c, None]
        frgM = cute.make_rmem_tensor_like(gM)
        cute.copy(copy_m, gM, frgM)
        frgM.load().to(cutlass.Float32)

        copy_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            starts_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_scalar = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            starts_flat.element_type,
            num_bits_per_copy=starts_flat.element_type.width,
        )

        tile_layout = cute.make_layout(cS.shape[0])
        pairs_per_thread = self.cfg.elems_per_thread // 2

        dmr = cutlass.Float32(0.0)
        dmi = cutlass.Float32(0.0)

        n_tiles = (S + cutlass.Int32(self.cfg.tile) - 1) // cutlass.Int32(self.cfg.tile)

        for tile_idx in cutlass.range(n_tiles, unroll=1):
            tile_start = cutlass.Int32(self.cfg.tile) * tile_idx
            residue = S - tile_start
            is_partial = cute.elem_less(residue, cutlass.Int32(self.cfg.tile))

            cta_coord = (None, tile_idx)
            ctaCrd = cS[cta_coord]
            tidCrd = cute.composition(ctaCrd, tv_layout)
            thrCrd = tidCrd[tidx, None]
            frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
            frgPred.fill(cutlass.Boolean(True))
            if is_partial:
                for i in cutlass.range_constexpr(cute.size(frgPred)):
                    frgPred[i] = cute.elem_less(thrCrd[i], S)

            gZ = starts_flat[bh, c, None]
            gG = dinc_flat[bh, c, None]

            tZ = cute.zipped_divide(gZ, tiler=tile_layout)
            tG = cute.zipped_divide(gG, tiler=tile_layout)
            ctaZ = tZ[cta_coord]
            ctaG = tG[cta_coord]
            tidZ = cute.composition(ctaZ, tv_layout)
            tidG = cute.composition(ctaG, tv_layout)
            thrZ = tidZ[tidx, None]
            thrG = tidG[tidx, None]

            frgZ = cute.make_rmem_tensor_like(thrZ)
            frgG = cute.make_rmem_tensor_like(thrG)
            frgZ.fill(0)
            frgG.fill(0)
            if is_partial:
                cute.copy(copy_scalar, thrZ, frgZ, pred=frgPred)
                cute.copy(copy_scalar, thrG, frgG, pred=frgPred)
            else:
                cute.copy(copy_vec, thrZ, frgZ)
                cute.copy(copy_vec, thrG, frgG)

            z = frgZ.load().to(cutlass.Float32)
            g = frgG.load().to(cutlass.Float32)

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = z[base + 0]
                zi = z[base + 1]
                gr = g[base + 0]
                gi = g[base + 1]

                dmr += gr * zr + gi * zi
                dmi += gi * zr - gr * zi

        for offset in (16, 8, 4, 2, 1):
            dmr += cute.arch.shuffle_sync_bfly(
                dmr, offset=offset, mask=-1, mask_and_clamp=31
            )
            dmi += cute.arch.shuffle_sync_bfly(
                dmi, offset=offset, mask=-1, mask_and_clamp=31
            )

        smem = cutlass.utils.SmemAllocator()
        warp_dmr = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), byte_alignment=4
        )
        warp_dmi = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((32,), stride=(1,)), byte_alignment=4
        )

        if lane == cutlass.Int32(0):
            warp_dmr[warp] = dmr
            warp_dmi[warp] = dmi

        cute.arch.barrier()

        if warp == cutlass.Int32(0):
            num_warps = cutlass.Int32(self.cfg.num_threads // 32)
            w = lane
            vdmr = cutlass.select_(w < num_warps, warp_dmr[w], cutlass.Float32(0.0))
            vdmi = cutlass.select_(w < num_warps, warp_dmi[w], cutlass.Float32(0.0))

            for offset in (16, 8, 4, 2, 1):
                vdmr += cute.arch.shuffle_sync_bfly(
                    vdmr, offset=offset, mask=-1, mask_and_clamp=31
                )
                vdmi += cute.arch.shuffle_sync_bfly(
                    vdmi, offset=offset, mask=-1, mask_and_clamp=31
                )

            if lane == cutlass.Int32(0):
                out = dm_flat[bh, c, None]
                r = cute.make_rmem_tensor_like(out, dtype=cutlass.Float32)
                r[0] = vdmr
                r[1] = vdmi
                out.store(r.load())

        return
