"""CuTe forward kernel for the v2x2ssd state-passing stage.

This mirrors the structure of ``v3x3ssd.cute.kernels.fwd.state_passing`` while
adapting the transport from scaled quaternions to packed complex scalars.

The algorithm is bandwidth-oriented:
  - sequential over chunks,
  - parallel over the flattened state axis ``S = P * D`` where ``D = 2N``.

Per chunk c:
  - write ``chunk_starts[..., c, :, :] = z`` (the pre-update state),
  - update ``z = m_chunk[..., c] * z + inc[..., c, :, :]``.

Outputs are float32, matching the reference path.
"""

from dataclasses import dataclass

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute


@dataclass(frozen=True)
class StatePassingLayoutBundle:
    layout_bcs: object
    layout_bcm: object
    layout_bs: object
    tile_layout: object
    tv_layout: object


@dataclass(frozen=True)
class StatePassingCopyBundle:
    copy_inc_vec: object
    copy_inc_scalar: object
    copy_out_vec: object
    copy_out_scalar: object
    copy_state_in_vec: object
    copy_state_in_scalar: object
    copy_state_out_vec: object
    copy_state_out_scalar: object
    copy_m: object


@dataclass(frozen=True)
class StatePassingKernelBundle:
    layouts: StatePassingLayoutBundle
    copies: StatePassingCopyBundle


class StatePassingFwdAmpere:
    """Ampere state-passing kernel (CopyUniversalOp + fp32 math)."""

    def __init__(
        self,
        *,
        num_threads: int = 128,
        vecs_per_thread: int = 8,
        copy_bits_in: int,
        copy_bits_out: int,
        copy_bits_state_in: int,
        copy_bits_state_out: int,
        has_init: bool,
    ):
        self.num_threads = int(num_threads)
        self.vecs_per_thread = int(vecs_per_thread)
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.vecs_per_thread <= 0:
            raise ValueError("vecs_per_thread must be positive.")

        self.elems_per_thread = 2 * self.vecs_per_thread
        self.tile = self.num_threads * self.elems_per_thread

        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)
        self.copy_bits_state_in = int(copy_bits_state_in)
        self.copy_bits_state_out = int(copy_bits_state_out)
        self.has_init = bool(has_init)

    def _make_layout_bundle(
        self, *, BH: int, C: int, S: int
    ) -> StatePassingLayoutBundle:
        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))
        layout_bs = cute.make_layout((BH, S), stride=(S, 1))
        tile_layout = cute.make_layout(self.tile)
        tv_layout = cute.make_layout(
            (self.num_threads, self.elems_per_thread),
            stride=(self.elems_per_thread, 1),
        )
        return StatePassingLayoutBundle(
            layout_bcs=layout_bcs,
            layout_bcm=layout_bcm,
            layout_bs=layout_bs,
            tile_layout=tile_layout,
            tv_layout=tv_layout,
        )

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
        inc_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        state_in_dtype: type[cutlass.Numeric],
        state_out_dtype: type[cutlass.Numeric],
        m_dtype: type[cutlass.Numeric],
    ) -> StatePassingCopyBundle:
        return StatePassingCopyBundle(
            copy_inc_vec=self._make_copy_atom(inc_dtype, self.copy_bits_in),
            copy_inc_scalar=self._make_copy_atom(inc_dtype, inc_dtype.width),
            copy_out_vec=self._make_copy_atom(out_dtype, self.copy_bits_out),
            copy_out_scalar=self._make_copy_atom(out_dtype, out_dtype.width),
            copy_state_in_vec=self._make_copy_atom(
                state_in_dtype, self.copy_bits_state_in
            ),
            copy_state_in_scalar=self._make_copy_atom(
                state_in_dtype, state_in_dtype.width
            ),
            copy_state_out_vec=self._make_copy_atom(
                state_out_dtype, self.copy_bits_state_out
            ),
            copy_state_out_scalar=self._make_copy_atom(
                state_out_dtype, state_out_dtype.width
            ),
            copy_m=self._make_copy_atom(m_dtype, m_dtype.width * 2),
        )

    def _make_kernel_bundle(
        self,
        *,
        BH: int,
        C: int,
        S: int,
        inc_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        state_in_dtype: type[cutlass.Numeric],
        state_out_dtype: type[cutlass.Numeric],
        m_dtype: type[cutlass.Numeric],
    ) -> StatePassingKernelBundle:
        return StatePassingKernelBundle(
            layouts=self._make_layout_bundle(BH=BH, C=C, S=S),
            copies=self._make_copy_bundle(
                inc_dtype=inc_dtype,
                out_dtype=out_dtype,
                state_in_dtype=state_in_dtype,
                state_out_dtype=state_out_dtype,
                m_dtype=m_dtype,
            ),
        )

    @cute.jit
    def _thread_tile_view(
        self,
        g_tensor: cute.Tensor,
        tile_layout: cute.Layout,
        cta_coord,
        tv_layout: cute.Layout,
        tidx: cutlass.Int32,
    ):
        t_tensor = cute.zipped_divide(g_tensor, tiler=tile_layout)
        cta_tensor = t_tensor[cta_coord]
        tid_tensor = cute.composition(cta_tensor, tv_layout)
        return tid_tensor[tidx, None]

    @cute.jit
    def __call__(
        self,
        inc: cute.Tensor,  # (B,H,C,P,D)
        m_chunk: cute.Tensor,  # (B,H,C,2)
        out_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        out_final: cute.Tensor,  # (B,H,P,D) fp32
        init_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored when has_init=False
    ):
        B, H, C, P, D = inc.shape
        BH = B * H
        S = P * D

        bundle = self._make_kernel_bundle(
            BH=BH,
            C=C,
            S=S,
            inc_dtype=inc.element_type,
            out_dtype=out_starts.element_type,
            state_in_dtype=init_or_dummy.element_type,
            state_out_dtype=out_final.element_type,
            m_dtype=m_chunk.element_type,
        )
        layouts = bundle.layouts
        copies = bundle.copies

        inc_flat = cute.make_tensor(inc.iterator, layouts.layout_bcs)
        m_flat = cute.make_tensor(m_chunk.iterator, layouts.layout_bcm)
        out_starts_flat = cute.make_tensor(out_starts.iterator, layouts.layout_bcs)
        out_final_flat = cute.make_tensor(out_final.iterator, layouts.layout_bs)
        init_flat = cute.make_tensor(init_or_dummy.iterator, layouts.layout_bs)
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=layouts.tile_layout)

        grid_x = cute.ceil_div(S, self.tile)
        grid_y = BH

        self.kernel(
            inc_flat,
            m_flat,
            out_starts_flat,
            out_final_flat,
            init_flat,
            cS,
            layouts.tile_layout,
            layouts.tv_layout,
            copies.copy_inc_vec,
            copies.copy_inc_scalar,
            copies.copy_out_vec,
            copies.copy_out_scalar,
            copies.copy_state_in_vec,
            copies.copy_state_in_scalar,
            copies.copy_state_out_vec,
            copies.copy_state_out_scalar,
            copies.copy_m,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.jit
    def call_on_stream(
        self,
        inc: cute.Tensor,  # (B,H,C,P,D)
        m_chunk: cute.Tensor,  # (B,H,C,2)
        out_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        out_final: cute.Tensor,  # (B,H,P,D) fp32
        init_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored when has_init=False
        stream: cuda.CUstream,
    ):
        B, H, C, P, D = inc.shape
        BH = B * H
        S = P * D

        bundle = self._make_kernel_bundle(
            BH=BH,
            C=C,
            S=S,
            inc_dtype=inc.element_type,
            out_dtype=out_starts.element_type,
            state_in_dtype=init_or_dummy.element_type,
            state_out_dtype=out_final.element_type,
            m_dtype=m_chunk.element_type,
        )
        layouts = bundle.layouts
        copies = bundle.copies

        inc_flat = cute.make_tensor(inc.iterator, layouts.layout_bcs)
        m_flat = cute.make_tensor(m_chunk.iterator, layouts.layout_bcm)
        out_starts_flat = cute.make_tensor(out_starts.iterator, layouts.layout_bcs)
        out_final_flat = cute.make_tensor(out_final.iterator, layouts.layout_bs)
        init_flat = cute.make_tensor(init_or_dummy.iterator, layouts.layout_bs)
        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=layouts.tile_layout)

        grid_x = cute.ceil_div(S, self.tile)
        grid_y = BH

        self.kernel(
            inc_flat,
            m_flat,
            out_starts_flat,
            out_final_flat,
            init_flat,
            cS,
            layouts.tile_layout,
            layouts.tv_layout,
            copies.copy_inc_vec,
            copies.copy_inc_scalar,
            copies.copy_out_vec,
            copies.copy_out_scalar,
            copies.copy_state_in_vec,
            copies.copy_state_in_scalar,
            copies.copy_state_out_vec,
            copies.copy_state_out_scalar,
            copies.copy_m,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        inc_flat: cute.Tensor,  # (BH, C, S)
        m_flat: cute.Tensor,  # (BH, C, 2)
        out_starts_flat: cute.Tensor,  # (BH, C, S) fp32
        out_final_flat: cute.Tensor,  # (BH, S) fp32
        init_flat: cute.Tensor,  # (BH, S)
        cS: cute.Tensor,  # (tile, ntiles)
        tile_layout: cute.Layout,
        tv_layout: cute.Layout,  # (tid, vid) -> linear coord in [0, tile)
        copy_inc_vec,
        copy_inc_scalar,
        copy_out_vec,
        copy_out_scalar,
        copy_state_in_vec,
        copy_state_in_scalar,
        copy_state_out_vec,
        copy_state_out_scalar,
        copy_m,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        bh = bidy
        tile_idx = bidx

        S = inc_flat.shape[2]
        C = inc_flat.shape[1]

        tile_start = cutlass.Int32(self.tile) * tile_idx
        residue = S - tile_start
        is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.tile))

        cta_coord = (None, tile_idx)
        ctaCrd = cS[cta_coord]
        thrCrd = cute.composition(ctaCrd, tv_layout)[tidx, None]
        frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
        frgPred.fill(cutlass.Boolean(True))
        if is_partial_tile:
            for i in cutlass.range_constexpr(cute.size(frgPred)):
                frgPred[i] = cute.elem_less(thrCrd[i], S)

        gInit = init_flat[bh, None]
        thrInit = self._thread_tile_view(gInit, tile_layout, cta_coord, tv_layout, tidx)

        accZ = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)
        accZ.fill(0.0)
        if cutlass.const_expr(self.has_init):
            frgInit = cute.make_rmem_tensor_like(thrInit)
            frgInit.fill(0)
            if is_partial_tile:
                cute.copy(copy_state_in_scalar, thrInit, frgInit, pred=frgPred)
            else:
                cute.copy(copy_state_in_vec, thrInit, frgInit)
            accZ.store(frgInit.load().to(cutlass.Float32))

        frgIn = cute.make_rmem_tensor_like(thrInit)
        frgOut = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)

        pairs_per_thread = cute.size(accZ) // 2

        for c in cutlass.range(C, unroll=1):
            gOut = out_starts_flat[bh, c, None]
            thrOut = self._thread_tile_view(
                gOut, tile_layout, cta_coord, tv_layout, tidx
            )

            frgOut.store(accZ.load())
            if is_partial_tile:
                cute.copy(copy_out_scalar, frgOut, thrOut, pred=frgPred)
            else:
                cute.copy(copy_out_vec, frgOut, thrOut)

            gInc = inc_flat[bh, c, None]
            thrInc = self._thread_tile_view(
                gInc, tile_layout, cta_coord, tv_layout, tidx
            )

            frgIn.fill(0)
            if is_partial_tile:
                cute.copy(copy_inc_scalar, thrInc, frgIn, pred=frgPred)
            else:
                cute.copy(copy_inc_vec, thrInc, frgIn)
            inc_f32 = frgIn.load().to(cutlass.Float32)

            gM = m_flat[bh, c, None]
            frgM = cute.make_rmem_tensor_like(gM)
            cute.copy(copy_m, gM, frgM)
            m = frgM.load().to(cutlass.Float32)
            mr, mi = m[0], m[1]

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = accZ[base + 0]
                zi = accZ[base + 1]

                rr = mr * zr - mi * zi
                ri = mr * zi + mi * zr

                accZ[base + 0] = rr + inc_f32[base + 0]
                accZ[base + 1] = ri + inc_f32[base + 1]

        gF = out_final_flat[bh, None]
        thrF = self._thread_tile_view(gF, tile_layout, cta_coord, tv_layout, tidx)

        frgOut.store(accZ.load())
        if is_partial_tile:
            cute.copy(copy_state_out_scalar, frgOut, thrF, pred=frgPred)
        else:
            cute.copy(copy_state_out_vec, frgOut, thrF)

        return
