from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .common import _TileConfig, _make_layout_bundle, _thread_tile_view


@dataclass(frozen=True)
class StatePassingCopyBundle:
    copy_starts_vec: object
    copy_starts_scalar: object
    copy_dstarts_vec: object
    copy_dstarts_scalar: object
    copy_out_vec: object
    copy_out_scalar: object
    copy_final_vec: object
    copy_final_scalar: object
    copy_m: object


class StatePassingBwdAmpere:
    """Fused backward kernel for d_inc, d_m_chunk, and d_initial."""

    def __init__(
        self,
        cfg: _TileConfig,
        *,
        copy_bits_starts: int,
        copy_bits_dstarts: int,
        copy_bits_out: int,
        copy_bits_final: int,
    ):
        self.cfg = cfg
        self.copy_bits_starts = int(copy_bits_starts)
        self.copy_bits_dstarts = int(copy_bits_dstarts)
        self.copy_bits_out = int(copy_bits_out)
        self.copy_bits_final = int(copy_bits_final)

    @staticmethod
    def _make_copy_atom(dtype: type[cutlass.Numeric], num_bits: int):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=int(num_bits),
        )

    def _make_copy_bundle(
        self,
        *,
        starts_dtype: type[cutlass.Numeric],
        dstarts_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        final_dtype: type[cutlass.Numeric],
        m_dtype: type[cutlass.Numeric],
    ) -> StatePassingCopyBundle:
        return StatePassingCopyBundle(
            copy_starts_vec=self._make_copy_atom(starts_dtype, self.copy_bits_starts),
            copy_starts_scalar=self._make_copy_atom(starts_dtype, starts_dtype.width),
            copy_dstarts_vec=self._make_copy_atom(
                dstarts_dtype, self.copy_bits_dstarts
            ),
            copy_dstarts_scalar=self._make_copy_atom(
                dstarts_dtype, dstarts_dtype.width
            ),
            copy_out_vec=self._make_copy_atom(out_dtype, self.copy_bits_out),
            copy_out_scalar=self._make_copy_atom(out_dtype, out_dtype.width),
            copy_final_vec=self._make_copy_atom(final_dtype, self.copy_bits_final),
            copy_final_scalar=self._make_copy_atom(final_dtype, final_dtype.width),
            copy_m=self._make_copy_atom(m_dtype, m_dtype.width * 2),
        )

    @cute.jit
    def __call__(
        self,
        chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        d_chunk_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        d_final: cute.Tensor,  # (B,H,P,D) fp32
        m_chunk: cute.Tensor,  # (B,H,C,2) fp32
        d_inc: cute.Tensor,  # (B,H,C,P,D) fp32
        d_m_chunk: cute.Tensor,  # (B,H,C,2) fp32
        d_initial: cute.Tensor,  # (B,H,P,D) fp32
    ):
        B, H, C, P, D = d_inc.shape
        BH = B * H
        S = P * D

        layouts = _make_layout_bundle(BH=BH, C=C, S=S, cfg=self.cfg)
        copies = self._make_copy_bundle(
            starts_dtype=chunk_starts.element_type,
            dstarts_dtype=d_chunk_starts.element_type,
            out_dtype=d_inc.element_type,
            final_dtype=d_final.element_type,
            m_dtype=m_chunk.element_type,
        )

        starts_flat = cute.make_tensor(chunk_starts.iterator, layouts.layout_bcs)
        dstarts_flat = cute.make_tensor(d_chunk_starts.iterator, layouts.layout_bcs)
        dfinal_flat = cute.make_tensor(d_final.iterator, layouts.layout_bs)
        m_flat = cute.make_tensor(m_chunk.iterator, layouts.layout_bcm)
        dinc_flat = cute.make_tensor(d_inc.iterator, layouts.layout_bcs)
        dm_flat = cute.make_tensor(d_m_chunk.iterator, layouts.layout_bcm)
        dinitial_flat = cute.make_tensor(d_initial.iterator, layouts.layout_bs)

        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=layouts.tile_layout)

        grid_x = cute.ceil_div(S, self.cfg.tile)
        grid_y = BH

        self.kernel(
            starts_flat,
            dstarts_flat,
            dfinal_flat,
            m_flat,
            dinc_flat,
            dm_flat,
            dinitial_flat,
            cS,
            layouts.tile_layout,
            layouts.tv_layout,
            copies.copy_starts_vec,
            copies.copy_starts_scalar,
            copies.copy_dstarts_vec,
            copies.copy_dstarts_scalar,
            copies.copy_out_vec,
            copies.copy_out_scalar,
            copies.copy_final_vec,
            copies.copy_final_scalar,
            copies.copy_m,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.cfg.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        starts_flat: cute.Tensor,  # (BH, C, S)
        dstarts_flat: cute.Tensor,  # (BH, C, S)
        dfinal_flat: cute.Tensor,  # (BH, S)
        m_flat: cute.Tensor,  # (BH, C, 2)
        dinc_flat: cute.Tensor,  # (BH, C, S)
        dm_flat: cute.Tensor,  # (BH, C, 2)
        dinitial_flat: cute.Tensor,  # (BH, S)
        cS: cute.Tensor,
        tile_layout: cute.Layout,
        tv_layout: cute.Layout,
        copy_starts_vec,
        copy_starts_scalar,
        copy_dstarts_vec,
        copy_dstarts_scalar,
        copy_out_vec,
        copy_out_scalar,
        copy_final_vec,
        copy_final_scalar,
        copy_m,
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

        gF = dfinal_flat[bh, None]
        thrF = _thread_tile_view(gF, tile_layout, cta_coord, tv_layout, tidx)

        accG = cute.make_rmem_tensor(thrF.shape, cutlass.Float32)
        accG.fill(0.0)
        frgF = cute.make_rmem_tensor_like(thrF)
        frgF.fill(0)
        if is_partial_tile:
            cute.copy(copy_final_scalar, thrF, frgF, pred=frgPred)
        else:
            cute.copy(copy_final_vec, thrF, frgF)
        accG.store(frgF.load().to(cutlass.Float32))

        frgStarts = cute.make_rmem_tensor_like(thrF)
        frgDStarts = cute.make_rmem_tensor_like(thrF)
        pairs_per_thread = cute.size(accG) // 2

        for c_it in cutlass.range(C, unroll=1):
            c = C - 1 - c_it

            gOut = dinc_flat[bh, c, None]
            thrOut = _thread_tile_view(gOut, tile_layout, cta_coord, tv_layout, tidx)

            frgTmp = cute.make_rmem_tensor_like(thrOut)
            frgTmp.store(accG.load().to(dinc_flat.element_type))
            if is_partial_tile:
                cute.copy(copy_out_scalar, frgTmp, thrOut, pred=frgPred)
            else:
                cute.copy(copy_out_vec, frgTmp, thrOut)

            gStarts = starts_flat[bh, c, None]
            thrStarts = _thread_tile_view(
                gStarts, tile_layout, cta_coord, tv_layout, tidx
            )
            frgStarts.fill(0)
            if is_partial_tile:
                cute.copy(copy_starts_scalar, thrStarts, frgStarts, pred=frgPred)
            else:
                cute.copy(copy_starts_vec, thrStarts, frgStarts)
            starts_f32 = frgStarts.load().to(cutlass.Float32)

            dmr = cutlass.Float32(0.0)
            dmi = cutlass.Float32(0.0)
            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = starts_f32[base + 0]
                zi = starts_f32[base + 1]
                gr = accG[base + 0]
                gi = accG[base + 1]
                dmr += gr * zr + gi * zi
                dmi += gi * zr - gr * zi
            for offset in (16, 8, 4, 2, 1):
                dmr += cute.arch.shuffle_sync_bfly(
                    dmr, offset=offset, mask=-1, mask_and_clamp=31
                )
                dmi += cute.arch.shuffle_sync_bfly(
                    dmi, offset=offset, mask=-1, mask_and_clamp=31
                )
            if lane == cutlass.Int32(0):
                gDM = dm_flat[bh, c, None]
                cute.arch.atomic_add((gDM.iterator + 0).llvm_ptr, dmr)
                cute.arch.atomic_add((gDM.iterator + 1).llvm_ptr, dmi)

            gIn = dstarts_flat[bh, c, None]
            thrIn = _thread_tile_view(gIn, tile_layout, cta_coord, tv_layout, tidx)

            frgDStarts.fill(0)
            if is_partial_tile:
                cute.copy(copy_dstarts_scalar, thrIn, frgDStarts, pred=frgPred)
            else:
                cute.copy(copy_dstarts_vec, thrIn, frgDStarts)
            dstart_f32 = frgDStarts.load().to(cutlass.Float32)

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
        thrI = _thread_tile_view(gI, tile_layout, cta_coord, tv_layout, tidx)

        frgTmp = cute.make_rmem_tensor_like(thrI)
        frgTmp.store(accG.load().to(dinitial_flat.element_type))
        if is_partial_tile:
            cute.copy(copy_out_scalar, frgTmp, thrI, pred=frgPred)
        else:
            cute.copy(copy_out_vec, frgTmp, thrI)

        return
