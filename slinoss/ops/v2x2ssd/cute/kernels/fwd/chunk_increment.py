"""CuTe forward kernel for the v2x2ssd chunk-increment stage.

This file mirrors the structural shape of the tensor-core path in
``v3x3ssd.cute.kernels.fwd.chunk_increment`` while adapting the math to the
``v2`` scan:

- one monolithic tensor-core workhorse: ``ChunkIncrementFwdAmpere``
- reverse-time suffix scan happens inside the kernel
- the per-step transformed ``B`` stream is built in shared memory
- one GEMM produces the summed interior contribution
- the step-0 boundary rank-1 term is added in the epilogue

The ``v2`` stage differs from ``v3`` in the transport algebra:

- ``M[t]`` is a complex scalar transition
- ``Kprev[t]`` / ``Kcurr[t]`` are complex scalar taps
- ``B`` stores complex pairs packed along ``D = 2N``

The kernel contract intentionally follows the ``v3`` tensor-core class rather
than any external wrapper surface. Compile/wrapper glue belongs in
``fwd/__init__.py``.
"""

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


@dataclass(frozen=True)
class ChunkIncrementLayoutBundle:
    a_major_mode: object
    b_major_mode: object
    c_major_mode: object
    sA_layout: object
    sB_layout: object
    sC_layout: object


@dataclass(frozen=True)
class ChunkIncrementCopyBundle:
    tiled_copy_A: object
    tiled_copy_B: object
    tiled_copy_C: object


@dataclass(frozen=True)
class ChunkIncrementKernelBundle:
    layouts: ChunkIncrementLayoutBundle
    copies: ChunkIncrementCopyBundle
    tiled_mma: object
    SharedStorage: object
    smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.smem_bytes


class ChunkIncrementFwdAmpere:
    """Ampere tensor-core kernel for the ``v2`` chunk-increment stage."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        chunk_size: int,
        cta_tiler: tuple[int, int, int] = (64, 96, 32),  # (bM=P, bN=D, bK=time)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int = 3,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.cta_tiler = cta_tiler
        self.num_stages = int(num_stages)
        self.atom_layout_mnk = atom_layout_mnk

        self.bM, self.bN, self.bK = map(int, self.cta_tiler)
        if self.L % self.bK != 0:
            raise ValueError("chunk_size must be divisible by bK for this kernel")
        if self.bN % 2 != 0:
            raise ValueError("bN (D-tile) must be divisible by 2 because D = 2N")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.L == self.bK and self.bN == 64:
            # A single K tile has no steady-state pipeline to overlap, so one
            # shared-memory stage is sufficient and removes a dead A/B slab.
            self.num_stages = 1
        elif self.num_stages < 2:
            raise ValueError("num_stages must be >= 2 for multi-tile kernels")

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        atomM, atomN, atomK = self.atom_layout_mnk

        self.num_threads = atomM * atomN * atomK * 32
        self.scan_threads = 1 << (max(1, self.L) - 1).bit_length()
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )

        if self.bM % (atomM * mmaM) != 0:
            raise ValueError("bM must be divisible by MMA instruction shape")
        if self.bN % (atomN * mmaN * 2) != 0:
            raise ValueError("bN must be divisible by MMA instruction shape")
        if atomK != 1:
            raise ValueError("atom_layout_mnk K must be 1")
        if self.bK % mmaK != 0:
            raise ValueError("bK must be divisible by MMA instruction shape")

    def _alpha_layout(self):
        return cute.make_layout((self.L, 2), stride=(2, 1))

    def _u0_layout(self):
        return cute.make_layout((self.bM,))

    def _b0_layout(self):
        return cute.make_layout((self.bN,))

    def _warp_scan_layout(self):
        return cute.make_layout((max(1, self.scan_threads // 32), 2), stride=(2, 1))

    def _output_alias_guard_layout(self):
        return cute.make_layout((256,))

    def _tail_pad_layout(self):
        return cute.make_layout((64,))

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
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    def _make_gmem_tiled_copy_AB(
        self, atom_copy, dtype, major_mode, copy_bits, *, tile_m: int
    ):
        copy_elems = copy_bits // dtype.width
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

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            value_layout = cute.make_layout((1, copy_elems))
            best_tm = None
            best_tn = None
            for tm in range(1, self.num_threads + 1):
                if self.num_threads % tm != 0:
                    continue
                tn = self.num_threads // tm
                tile_m = tm
                tile_n = tn * copy_elems
                if (self.bM % tile_m) != 0 or (self.bN % tile_n) != 0:
                    continue
                if best_tm is None or tile_n > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn

            if best_tm is None:
                value_layout = cute.make_layout((1, 1))
                best_tm = self.num_threads
                best_tn = 1

            thread_layout = cute.make_layout((best_tm, best_tn), stride=(best_tn, 1))
            return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

        value_layout = cute.make_layout((copy_elems, 1))
        best_tm = None
        best_tn = None
        for tm in range(1, self.num_threads + 1):
            if self.num_threads % tm != 0:
                continue
            tn = self.num_threads // tm
            tile_m = tm * copy_elems
            tile_n = tn
            if (self.bM % tile_m) != 0 or (self.bN % tile_n) != 0:
                continue
            if best_tm is None or tile_m > (best_tm * copy_elems):
                best_tm = tm
                best_tn = tn

        if best_tm is None:
            value_layout = cute.make_layout((1, 1))
            best_tm = self.num_threads
            best_tn = 1

        thread_layout = cute.make_layout((best_tm, best_tn), stride=(best_tn, 1))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_layout_bundle(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mInc: cute.Tensor,
    ) -> ChunkIncrementLayoutBundle:
        a_major_mode = utils.LayoutEnum.from_tensor(mU)
        b_major_mode = utils.LayoutEnum.from_tensor(mB)
        c_major_mode = utils.LayoutEnum.from_tensor(mInc)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mU.element_type,
            a_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            b_major_mode,
            ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
        return ChunkIncrementLayoutBundle(
            a_major_mode=a_major_mode,
            b_major_mode=b_major_mode,
            c_major_mode=c_major_mode,
            sA_layout=sA_layout,
            sB_layout=sB_layout,
            sC_layout=sC_layout,
        )

    def _make_copy_bundle(
        self,
        layouts: ChunkIncrementLayoutBundle,
        in_dtype: type[cutlass.Numeric],
    ) -> ChunkIncrementCopyBundle:
        copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            in_dtype,
            layouts.a_major_mode,
            copy_bits,
            tile_m=self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            in_dtype,
            layouts.b_major_mode,
            copy_bits,
            tile_m=self.bN,
        )

        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy,
            self.c_dtype,
            layouts.c_major_mode,
            copy_bits,
        )
        return ChunkIncrementCopyBundle(
            tiled_copy_A=tiled_copy_A,
            tiled_copy_B=tiled_copy_B,
            tiled_copy_C=tiled_copy_C,
        )

    def _make_tiled_mma(self):
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        atoms_layout = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(op, atoms_layout, permutation_mnk=permutation_mnk)

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkIncrementLayoutBundle,
    ):
        alpha_layout = self._alpha_layout()
        u0_layout = self._u0_layout()
        b0_layout = self._b0_layout()
        warp_scan_layout = self._warp_scan_layout()
        output_alias_guard_layout = self._output_alias_guard_layout()
        tail_pad_layout = self._tail_pad_layout()

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sA": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sA_layout)], 16
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sB_layout)], 16
            ],
            # Preserve the historical aliasing envelope for the shared-memory
            # epilogue path used outside the 64-wide single-tile fast path.
            "output_alias_guard": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(output_alias_guard_layout)
                ],
                16,
            ],
            "alpha_prev": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 4
            ],
            "alpha_curr": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 4
            ],
            "u0": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(u0_layout)], 4
            ],
            "b0": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(b0_layout)], 4
            ],
            "warp_m_total": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)], 8
            ],
            "warp_m_offset": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)], 8
            ],
            "tail_pad": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tail_pad_layout)], 4
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mInc: cute.Tensor,
    ) -> ChunkIncrementKernelBundle:
        layouts = self._make_layout_bundle(mU, mB, mInc)
        copies = self._make_copy_bundle(layouts, mU.element_type)
        SharedStorage = self._make_shared_storage(mU.element_type, layouts)
        return ChunkIncrementKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=self._make_tiled_mma(),
            SharedStorage=SharedStorage,
            smem_bytes=int(SharedStorage.size_in_bytes()),
        )

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,  # (P, T_pad, BH)
        mB: cute.Tensor,  # (D, T_pad, BH)
        mM: cute.Tensor,  # (2, T_pad, BH)
        mKprev: cute.Tensor,  # (2, T_pad, BH)
        mKcurr: cute.Tensor,  # (2, T_pad, BH)
        mU_prev0: cute.Tensor,  # (P, BH)
        mB_prev0: cute.Tensor,  # (D, BH)
        mInc: cute.Tensor,  # (P, D, BHC) fp32
        mMchunk: cute.Tensor,  # (2, BHC) fp32
    ):
        bundle = self._make_kernel_bundle(mU, mB, mInc)
        grid_dim = cute.ceil_div(mInc.shape, (self.bM, self.bN, 1))
        grid_z = cute.size(mInc.shape[2])

        self.kernel(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mU_prev0,
            mB_prev0,
            mInc,
            mMchunk,
            bundle.layouts.a_major_mode,
            bundle.layouts.b_major_mode,
            bundle.layouts.sA_layout,
            bundle.layouts.sB_layout,
            bundle.layouts.sC_layout,
            bundle.copies.tiled_copy_A,
            bundle.copies.tiled_copy_B,
            bundle.copies.tiled_copy_C,
            bundle.tiled_mma,
            bundle.SharedStorage,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), grid_z),
            block=[self.num_threads, 1, 1],
            smem=bundle.smem_size,
        )

    @cute.jit
    def _make_copy_row_predicate(
        self,
        partitioned_dst: cute.Tensor,
        partitioned_coord: cute.Tensor,
        row_limit: int,
    ):
        pred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    partitioned_dst.shape[0][1],
                    cute.size(partitioned_dst, mode=[1]),
                    cute.size(partitioned_dst, mode=[2]),
                ),
                stride=(cute.size(partitioned_dst, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(pred.shape[0]):
            for row in cutlass.range_constexpr(pred.shape[1]):
                pred[rest_v, row, 0] = cute.elem_less(
                    partitioned_coord[(0, rest_v), row, 0, 0][0], row_limit
                )
        return pred

    @cute.kernel
    def kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mInc: cute.Tensor,
        mMchunk: cute.Tensor,
        a_major_mode: cutlass.Constexpr,
        b_major_mode: cutlass.Constexpr,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        BH = mU.shape[2]
        BHC = mInc.shape[2]
        n_chunks = BHC // BH

        bh = bidz // n_chunks
        chunk = bidz - bh * n_chunks
        chunk_start = chunk * self.L

        tiler_coord = (bidx, bidy, None)

        gC = cute.local_tile(
            mInc[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout
        )
        alpha_layout = self._alpha_layout()
        warp_scan_layout = self._warp_scan_layout()
        s_alpha_prev = storage.alpha_prev.get_tensor(alpha_layout)
        s_alpha_curr = storage.alpha_curr.get_tensor(alpha_layout)
        s_u0 = storage.u0.get_tensor(self._u0_layout())
        s_b0 = storage.b0.get_tensor(self._b0_layout())
        warp_m_total = storage.warp_m_total.get_tensor(warp_scan_layout)
        warp_m_offset = storage.warp_m_offset.get_tensor(warp_scan_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)

        tAsA = thr_copy_A.partition_D(sA)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mU.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
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

        mcU = cute.make_identity_tensor(mU.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        mcU_off = cute.domain_offset((0, chunk_start, 0), mcU)
        mcB_off = cute.domain_offset((0, chunk_start, 0), mcB)
        cU = cute.local_tile(
            mcU_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        tUcU = thr_copy_A.partition_S(cU)
        tBcB = thr_copy_B.partition_S(cB)

        tUpU = self._make_copy_row_predicate(tAsA, tUcU, mU.shape[0])
        tBpB = self._make_copy_row_predicate(tBsB, tBcB, mB.shape[0])

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        t = cutlass.Int32(self.L - 1) - tidx
        t_glob = chunk_start + t

        mr_step = cutlass.Float32(1.0)
        mi_step = cutlass.Float32(0.0)
        kp_re = cutlass.Float32(0.0)
        kp_im = cutlass.Float32(0.0)
        kc_re = cutlass.Float32(0.0)
        kc_im = cutlass.Float32(0.0)

        if tidx < cutlass.Int32(self.L):
            mr_step = cutlass.Float32(mM[0, t_glob, bh].to(cutlass.Float32))
            mi_step = cutlass.Float32(mM[1, t_glob, bh].to(cutlass.Float32))
            kp_re = cutlass.Float32(mKprev[0, t_glob, bh].to(cutlass.Float32))
            kp_im = cutlass.Float32(mKprev[1, t_glob, bh].to(cutlass.Float32))
            kc_re = cutlass.Float32(mKcurr[0, t_glob, bh].to(cutlass.Float32))
            kc_im = cutlass.Float32(mKcurr[1, t_glob, bh].to(cutlass.Float32))

        scan_threads = cutlass.Int32(self.scan_threads)
        num_warps_scan = cutlass.Int32(self.scan_threads // 32)
        in_scan = tidx < scan_threads

        mr = cutlass.select_(in_scan, mr_step, cutlass.Float32(1.0))
        mi = cutlass.select_(in_scan, mi_step, cutlass.Float32(0.0))

        if warp < num_warps_scan:
            for offset in (1, 2, 4, 8, 16):
                orr = cute.arch.shuffle_sync_up(
                    mr, offset=offset, mask=-1, mask_and_clamp=0
                )
                ori = cute.arch.shuffle_sync_up(
                    mi, offset=offset, mask=-1, mask_and_clamp=0
                )
                pred = lane >= cutlass.Int32(offset)
                nr = orr * mr - ori * mi
                ni = orr * mi + ori * mr
                mr = cutlass.select_(pred, nr, mr)
                mi = cutlass.select_(pred, ni, mi)

            if lane == cutlass.Int32(31):
                warp_m_total[warp, 0] = mr
                warp_m_total[warp, 1] = mi

        cute.arch.sync_threads()

        if cutlass.const_expr(self.scan_threads > 32):
            if warp == cutlass.Int32(0):
                w = lane
                has_warp = w < num_warps_scan

                wr = cutlass.select_(has_warp, warp_m_total[w, 0], cutlass.Float32(1.0))
                wi = cutlass.select_(has_warp, warp_m_total[w, 1], cutlass.Float32(0.0))

                for offset in (1, 2, 4, 8, 16):
                    orr = cute.arch.shuffle_sync_up(
                        wr, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    ori = cute.arch.shuffle_sync_up(
                        wi, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    pred = lane >= cutlass.Int32(offset)
                    nr = orr * wr - ori * wi
                    ni = orr * wi + ori * wr
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
                    warp_m_offset[w, 0] = off_r
                    warp_m_offset[w, 1] = off_i

            cute.arch.sync_threads()

            if warp < num_warps_scan:
                off_r = warp_m_offset[warp, 0]
                off_i = warp_m_offset[warp, 1]
                nr = off_r * mr - off_i * mi
                ni = off_r * mi + off_i * mr
                mr, mi = nr, ni

        if warp < num_warps_scan and lane == cutlass.Int32(31):
            warp_m_total[warp, 0] = mr
            warp_m_total[warp, 1] = mi

        cute.arch.sync_threads()

        suf_r = cutlass.Float32(1.0)
        suf_i = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            mr_prev = cute.arch.shuffle_sync_up(mr, offset=1, mask=-1, mask_and_clamp=0)
            mi_prev = cute.arch.shuffle_sync_up(mi, offset=1, mask=-1, mask_and_clamp=0)

            if tidx == cutlass.Int32(0):
                suf_r = cutlass.Float32(1.0)
                suf_i = cutlass.Float32(0.0)
            else:
                if lane == cutlass.Int32(0):
                    suf_r = warp_m_total[warp - 1, 0]
                    suf_i = warp_m_total[warp - 1, 1]
                else:
                    suf_r = mr_prev
                    suf_i = mi_prev

            ap_r = suf_r * kp_re - suf_i * kp_im
            ap_i = suf_r * kp_im + suf_i * kp_re
            ac_r = suf_r * kc_re - suf_i * kc_im
            ac_i = suf_r * kc_im + suf_i * kc_re

            s_alpha_prev[t, 0] = ap_r
            s_alpha_prev[t, 1] = ap_i
            s_alpha_curr[t, 0] = ac_r
            s_alpha_curr[t, 1] = ac_i

            if (bidx == 0) and (bidy == 0) and (t == cutlass.Int32(0)):
                mMchunk[0, bidz] = mr.to(cutlass.Float32)
                mMchunk[1, bidz] = mi.to(cutlass.Float32)

        cute.arch.sync_threads()

        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = self.L // self.bK

        mU_off = cute.domain_offset((0, chunk_start, 0), mU)
        mB_off = cute.domain_offset((0, chunk_start, 0), mB)

        gA = cute.local_tile(
            mU_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        tAgA = thr_copy_A.partition_S(gA)
        tBgB = thr_copy_B.partition_S(gB)

        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()
        k_tile_index = cutlass.Int32(0)

        for kk in cutlass.range_constexpr(tUpU.shape[2]):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, kk, k_tile_index],
                tAsA[None, None, kk, 0],
                pred=tUpU[None, None, kk],
            )
        for kk in cutlass.range_constexpr(tBpB.shape[2]):
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, kk, k_tile_index],
                tBsB[None, None, kk, 0],
                pred=tBpB[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile >= k_tile_count:
                tUpU.fill(0)
                tBpB.fill(0)
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tUpU,
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

        nvec = self.bN // 2
        total = self.bK * nvec
        num_iters = (total + self.num_threads - 1) // self.num_threads

        for kt in range(k_tile_count):
            if cutlass.const_expr(num_smem_stages == 1):
                cute.arch.cp_async_wait_group(0)
            else:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            k_tile_offset = kt * self.bK

            for it in cutlass.range_constexpr(num_iters):
                idx = tidx + (it * self.num_threads)
                if cute.elem_less(idx, total):
                    k = idx // nvec
                    v = idx - k * nvec
                    d0 = v * 2
                    step = k_tile_offset + k

                    br = sB[d0 + 0, k, smem_pipe_read].to(cutlass.Float32)
                    bi = sB[d0 + 1, k, smem_pipe_read].to(cutlass.Float32)

                    ar = s_alpha_curr[step, 0]
                    ai = s_alpha_curr[step, 1]
                    if step < cutlass.Int32(self.L - 1):
                        ar = ar + s_alpha_prev[step + 1, 0]
                        ai = ai + s_alpha_prev[step + 1, 1]

                    rr = ar * br - ai * bi
                    ri = ar * bi + ai * br

                    sB[d0 + 0, k, smem_pipe_read] = rr.to(mB.element_type)
                    sB[d0 + 1, k, smem_pipe_read] = ri.to(mB.element_type)

            cute.arch.sync_threads()

            if cutlass.const_expr(num_smem_stages > 1):
                next_tile = kt + (num_smem_stages - 1)
                if next_tile < k_tile_count:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, k_tile_index],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tUpU,
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

        tile_m0 = bidx * self.bM
        tile_n0 = bidy * self.bN

        m = tidx
        while cute.elem_less(m, self.bM):
            g_m = tile_m0 + m
            if cute.elem_less(g_m, mU.shape[0]):
                if chunk == 0:
                    s_u0[m] = cutlass.Float32(mU_prev0[g_m, bh].to(cutlass.Float32))
                else:
                    s_u0[m] = cutlass.Float32(
                        mU[g_m, chunk_start - 1, bh].to(cutlass.Float32)
                    )
            else:
                s_u0[m] = cutlass.Float32(0.0)
            m = m + self.num_threads

        a0r = s_alpha_prev[0, 0]
        a0i = s_alpha_prev[0, 1]

        v = tidx
        nvec = self.bN // 2
        while cute.elem_less(v, nvec):
            d0 = v * 2
            g_d0 = tile_n0 + d0

            br = cutlass.Float32(0.0)
            bi = cutlass.Float32(0.0)
            if cute.elem_less(g_d0 + 1, mB.shape[0]):
                if chunk == 0:
                    br = cutlass.Float32(mB_prev0[g_d0 + 0, bh].to(cutlass.Float32))
                    bi = cutlass.Float32(mB_prev0[g_d0 + 1, bh].to(cutlass.Float32))
                else:
                    br = cutlass.Float32(
                        mB[g_d0 + 0, chunk_start - 1, bh].to(cutlass.Float32)
                    )
                    bi = cutlass.Float32(
                        mB[g_d0 + 1, chunk_start - 1, bh].to(cutlass.Float32)
                    )

            rr = a0r * br - a0i * bi
            ri = a0r * bi + a0i * br

            s_b0[d0 + 0] = rr
            s_b0[d0 + 1] = ri
            v = v + self.num_threads

        cute.arch.sync_threads()

        ceilM, ceilN, _ = cute.ceil_div(mInc.shape, (self.bM, self.bN, 1))
        mcC = cute.make_identity_tensor(
            (cute.size(ceilM) * self.bM, cute.size(ceilN) * self.bN, 1)
        )
        cC = cute.local_tile(
            mcC[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        if cutlass.const_expr(num_smem_stages == 1):
            tCcC = thr_mma.partition_C(cC)
            tCcC_mn = self._make_acc_tensor_mn_view(tCcC)
            tCrC_mn = self._make_acc_tensor_mn_view(tCrC)

            for r in cutlass.range_constexpr(cute.size(tCrC_mn.shape[0])):
                for c in cutlass.range_constexpr(cute.size(tCrC_mn.shape[1])):
                    row_idx = cutlass.Int32(tCcC_mn[r, c][0])
                    col_idx = cutlass.Int32(tCcC_mn[r, c][1])
                    if cute.elem_less(row_idx, mInc.shape[0]) and cute.elem_less(
                        col_idx, mInc.shape[1]
                    ):
                        row_local = row_idx - tile_m0
                        col_local = col_idx - tile_n0
                        mInc[row_idx, col_idx, bidz] = (
                            cutlass.Float32(tCrC_mn[r, c])
                            + s_u0[row_local] * s_b0[col_local]
                        )
        else:
            tCcC = thr_copy_C.partition_S(cC)

            cute.autovec_copy(tCrC, tCsC)
            cute.arch.sync_threads()

            total_elems = self.bM * self.bN
            idx = tidx
            while cute.elem_less(idx, total_elems):
                m_idx = idx // self.bN
                n_idx = idx - (m_idx * self.bN)
                sC[m_idx, n_idx] = sC[m_idx, n_idx] + s_u0[m_idx] * s_b0[n_idx]
                idx = idx + self.num_threads

            cute.arch.sync_threads()

            tCrC_epilogue = cute.make_fragment_like(tCgC_epilogue)
            cute.copy(tiled_copy_C, tCsC_epilogue, tCrC_epilogue)

            tCpC = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tCgC_epilogue.shape[0][1],
                        cute.size(tCgC_epilogue, mode=[1]),
                        cute.size(tCgC_epilogue, mode=[2]),
                    ),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tCpC.shape[0]):
                for m in cutlass.range_constexpr(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(
                        tCcC[(0, rest_v), m, 0][0], mInc.shape[0]
                    )

            for rest_v in cutlass.range_constexpr(tCpC.shape[0]):
                for n in cutlass.range_constexpr(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mInc.shape[1]):
                        cute.copy(
                            tiled_copy_C,
                            tCrC_epilogue[None, None, n],
                            tCgC_epilogue[None, None, n],
                            pred=tCpC[None, None, n],
                        )

        return
