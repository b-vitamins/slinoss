"""CuTe backward ``dz0`` workhorse for the ``v2x2ssd`` chunk-scan stage.

This file is intentionally written in the same overall shape as the
``v3x3ssd`` ``chunk_scan`` ``dz0`` workhorse:

- one monolithic stage-native kernel class
- ``__call__`` only builds layouts, copies, MMA, shared memory, and launch
- the kernel maps ``bidz`` to ``(bh, chunk)``
- prefix metadata is reconstructed inside the CTA from raw stage inputs
- ``dOut`` and ``C`` are transformed tile-by-tile in shared memory
- one tensor-core GEMM produces public ``dZ0`` directly

The adaptation is only in the scan algebra:

- quaternion transport becomes unit-complex transport
- 3-vectors become interleaved complex pairs
- ``trans`` becomes raw packed complex transitions ``M``
- ``row_scale`` is the prefix magnitude, and the column transform is
  ``C * conj(phase_prefix)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import LOG2_E, TWO_LOG2_E, complex_mul, mul_conj_phase


@dataclass(frozen=True)
class ChunkScanBwdDZ0LayoutBundle:
    a_major_mode: object
    b_major_mode: object
    c_major_mode: object
    sA_layout: object
    sB_layout: object
    sC_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDZ0CopyBundle:
    tiled_copy_A: object
    tiled_copy_B: object
    tiled_copy_C: object
    tiled_copy_s2r_A: object
    tiled_copy_s2r_B: object


@dataclass(frozen=True)
class ChunkScanBwdDZ0KernelBundle:
    layouts: ChunkScanBwdDZ0LayoutBundle
    copies: ChunkScanBwdDZ0CopyBundle
    tiled_mma: object
    smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.smem_bytes


class ChunkScanBwdDZ0Ampere:
    """Ampere tensor-core kernel for ``dZ0`` (fp16/bf16 inputs, accum fp32)."""

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        cta_tiler: tuple[int, int, int] = (64, 96, 32),  # (bM=P, bN=D, bK=time)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int = 2,
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
            raise ValueError("chunk_size must be divisible by bK for this kernel.")
        k_tile_count = self.L // self.bK
        if (self.num_stages - 1) > k_tile_count:
            raise ValueError(
                "num_stages too large for chunk_size/bK (insufficient K tiles)."
            )
        if self.bN % 2 != 0:
            raise ValueError("bN (D-tile) must be divisible by 2 because D = 2N.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2.")

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        atomM, atomN, atomK = self.atom_layout_mnk

        self.num_threads = atomM * atomN * atomK * 32
        if self.L > self.num_threads:
            raise ValueError("chunk_size too large for this CTA thread count.")

        if self.bM % (atomM * mmaM) != 0:
            raise ValueError("bM must be divisible by MMA instruction shape.")
        if self.bN % (atomN * mmaN * 2) != 0:
            raise ValueError("bN must be divisible by MMA instruction shape.")
        if atomK != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bK % mmaK != 0:
            raise ValueError("bK must be divisible by MMA instruction shape.")

    def _row_layout(self):
        return cute.make_layout((self.num_threads,), stride=(1,))

    def _phase_layout(self):
        return cute.make_layout((self.num_threads, 2), stride=(2, 1))

    def _output_alias_guard_layout(
        self,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
    ):
        smem_size_AB = cute.cosize(sA_layout) * (in_a_dtype.width // 8) + cute.cosize(
            sB_layout
        ) * (in_b_dtype.width // 8)
        smem_size_C = cute.size_in_bytes(self.c_dtype, sC_layout)
        pad_bytes = max(0, int(smem_size_C) - int(smem_size_AB))
        return cute.make_layout((((pad_bytes + 3) // 4),), stride=(1,))

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

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size
        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            layout_atom_outer,
        )
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1))

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
                if (self.bM % tile_m) != 0:
                    continue
                if (self.bN % tile_n) != 0:
                    continue
                if best_tm is None or tile_n > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn
            if best_tm is None:
                shape_dim_1 = cute.size(self.bN) // copy_elems
                thread_layout = cute.make_layout(
                    (self.num_threads // shape_dim_1, shape_dim_1),
                    stride=(shape_dim_1, 1),
                )
            else:
                thread_layout = cute.make_layout(
                    (best_tm, best_tn), stride=(best_tn, 1)
                )
            return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

        value_layout = cute.make_layout((copy_elems, 1))
        shape_dim_0 = (int(self.bM) + int(copy_elems) - 1) // int(copy_elems)
        if shape_dim_0 > self.num_threads:
            raise ValueError("bM too large for vectorized col-major store.")
        tm = None
        for cand in range(shape_dim_0, self.num_threads + 1):
            if self.num_threads % cand == 0:
                tm = cand
                break
        if tm is None:
            raise ValueError(
                "Internal error: failed to find divisor for col-major store."
            )
        thread_layout = cute.make_layout((tm, self.num_threads // tm), stride=(1, tm))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_layout_bundle(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mDZ0: cute.Tensor,
    ) -> ChunkScanBwdDZ0LayoutBundle:
        a_major_mode = utils.LayoutEnum.from_tensor(mDOut)
        b_major_mode = utils.LayoutEnum.from_tensor(mC)
        c_major_mode = utils.LayoutEnum.from_tensor(mDZ0)
        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mDOut.element_type,
            a_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mC.element_type,
            b_major_mode,
            ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
        return ChunkScanBwdDZ0LayoutBundle(
            a_major_mode=a_major_mode,
            b_major_mode=b_major_mode,
            c_major_mode=c_major_mode,
            sA_layout=sA_layout,
            sB_layout=sB_layout,
            sC_layout=sC_layout,
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
        tC = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    def _make_copy_bundle(
        self,
        layouts: ChunkScanBwdDZ0LayoutBundle,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        tiled_mma: cute.TiledMma,
    ) -> ChunkScanBwdDZ0CopyBundle:
        ab_copy_bits = 128
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_a_dtype,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            in_a_dtype,
            layouts.a_major_mode,
            ab_copy_bits,
            tile_m=self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            in_b_dtype,
            layouts.b_major_mode,
            ab_copy_bits,
            tile_m=self.bN,
        )
        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, self.c_dtype, layouts.c_major_mode, c_copy_bits
        )
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            in_a_dtype,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            in_b_dtype,
        )
        return ChunkScanBwdDZ0CopyBundle(
            tiled_copy_A=tiled_copy_A,
            tiled_copy_B=tiled_copy_B,
            tiled_copy_C=tiled_copy_C,
            tiled_copy_s2r_A=cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma),
            tiled_copy_s2r_B=cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma),
        )

    def _make_shared_storage(
        self,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
    ):
        row_layout = self._row_layout()
        phase_layout = self._phase_layout()
        output_alias_guard_layout = self._output_alias_guard_layout(
            in_a_dtype, in_b_dtype, sA_layout, sB_layout, sC_layout
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "s_row": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ],
            "s_phase": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 8
            ],
            "s_logp": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ],
            "s_phase_scan": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 8
            ],
            "sA": cute.struct.Align[
                cute.struct.MemRange[in_a_dtype, cute.cosize(sA_layout)], 16
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[in_b_dtype, cute.cosize(sB_layout)], 16
            ],
            "output_alias_guard": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(output_alias_guard_layout)
                ],
                16,
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self,
        mDOut: cute.Tensor,
        mC: cute.Tensor,
        mDZ0: cute.Tensor,
    ) -> ChunkScanBwdDZ0KernelBundle:
        layouts = self._make_layout_bundle(mDOut, mC, mDZ0)
        tiled_mma = self._make_tiled_mma()
        copies = self._make_copy_bundle(
            layouts, mDOut.element_type, mC.element_type, tiled_mma
        )
        shared_storage = self._make_shared_storage(
            mDOut.element_type,
            mC.element_type,
            layouts.sA_layout,
            layouts.sB_layout,
            layouts.sC_layout,
        )
        return ChunkScanBwdDZ0KernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=tiled_mma,
            smem_bytes=int(shared_storage.size_in_bytes()),
        )

    @cute.jit(preprocess=True)
    def __call__(
        self,
        mDOut: cute.Tensor,  # (P, T, BH)   fp16/bf16
        mC: cute.Tensor,  # (D, T, BH)   fp16/bf16
        mM: cute.Tensor,  # (2, T, BH)   fp32
        mDZ0: cute.Tensor,  # (P, D, BHC)  fp32
    ):
        bundle = self._make_kernel_bundle(mDOut, mC, mDZ0)

        grid_dim = cute.ceil_div(mDZ0.shape, (self.bM, self.bN, 1))
        grid_z = cute.size(mDZ0.shape[2])

        self.kernel(
            mDOut,
            mC,
            mM,
            mDZ0,
            bundle.layouts.sA_layout,
            bundle.layouts.sB_layout,
            bundle.layouts.sC_layout,
            bundle.copies.tiled_copy_A,
            bundle.copies.tiled_copy_B,
            bundle.copies.tiled_copy_C,
            bundle.copies.tiled_copy_s2r_A,
            bundle.copies.tiled_copy_s2r_B,
            bundle.tiled_mma,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), grid_z),
            block=[self.num_threads, 1, 1],
            smem=bundle.smem_size,
        )

    @cute.kernel(preprocess=True)
    def kernel(
        self,
        mDOut: cute.Tensor,  # (P, T, BH)
        mC: cute.Tensor,  # (D, T, BH)
        mM: cute.Tensor,  # (2, T, BH)
        mDZ0: cute.Tensor,  # (P, D, BHC)
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_copy_s2r_A: cute.TiledCopy,
        tiled_copy_s2r_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        BH = mDOut.shape[2]
        BHC = mDZ0.shape[2]
        n_chunks = BHC // BH
        bh = bidz // n_chunks
        chunk = bidz - bh * n_chunks
        chunk_start = chunk * self.L

        gC = cute.local_tile(
            mDZ0[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None),
        )

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._make_shared_storage(
            mDOut.element_type, mC.element_type, sA_layout, sB_layout, sC_layout
        )
        storage = smem.allocate(SharedStorage)
        s_row = storage.s_row.get_tensor(self._row_layout())
        s_phase = storage.s_phase.get_tensor(self._phase_layout())
        s_logp = storage.s_logp.get_tensor(self._row_layout())
        s_phase_scan = storage.s_phase_scan.get_tensor(self._phase_layout())
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Float32), sC_layout
        )

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)

        tAsA = thr_copy_A.partition_D(sA)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_ep = thr_copy_C.partition_S(sC)
        tCgC_ep = thr_copy_C.partition_D(gC)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        thr_copy_ld_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ld_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy = thr_copy_ld_A.partition_S(sA)
        tCrA_copy = thr_copy_ld_A.retile(tCrA)
        tCsB_copy = thr_copy_ld_B.partition_S(sB)
        tCrB_copy = thr_copy_ld_B.retile(tCrB)

        mcA = cute.make_identity_tensor(mDOut.layout.shape)
        mcB = cute.make_identity_tensor(mC.layout.shape)
        mcA_off = cute.domain_offset((0, chunk_start, 0), mcA)
        mcB_off = cute.domain_offset((0, chunk_start, 0), mcB)

        cA = cute.local_tile(
            mcA_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
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
                    tAcA[(0, rest_v), m, 0, 0][0], mDOut.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mC.shape[0]
                )

        k_tile_count = self.L // self.bK
        mA_off = cute.domain_offset((0, chunk_start, 0), mDOut)
        mB_off = cute.domain_offset((0, chunk_start, 0), mC)

        gA = cute.local_tile(
            mA_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB_off[None, None, bh],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        tAgA = thr_copy_A.partition_S(gA)
        tBgB = thr_copy_B.partition_S(gB)

        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        t_raw = cutlass.Int32(chunk_start) + tidx
        t_max = cutlass.Int32(chunk_start + (self.L - 1))
        t_safe = cutlass.select_(t_raw > t_max, t_max, t_raw)

        mr = cutlass.Float32(mM[0, t_safe, bh].to(cutlass.Float32))
        mi = cutlass.Float32(mM[1, t_safe, bh].to(cutlass.Float32))

        pred_t = tidx < cutlass.Int32(self.L)
        mr = cutlass.Float32(cutlass.select_(pred_t, mr, cutlass.Float32(1.0)))
        mi = cutlass.Float32(cutlass.select_(pred_t, mi, cutlass.Float32(0.0)))

        eps = cutlass.Float32(1.0e-20)
        mag2 = cutlass.Float32(mr * mr + mi * mi + eps)
        inv_mag = cutlass.Float32(cute.math.rsqrt(mag2))
        ur = mr * inv_mag
        ui = mi * inv_mag
        logp = cutlass.Float32(
            cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.25 / LOG2_E)
        )

        for offset in (1, 2, 4, 8, 16):
            other_log = cute.arch.shuffle_sync_up(
                logp, offset=offset, mask=-1, mask_and_clamp=0
            )
            orr = cute.arch.shuffle_sync_up(
                ur, offset=offset, mask=-1, mask_and_clamp=0
            )
            oii = cute.arch.shuffle_sync_up(
                ui, offset=offset, mask=-1, mask_and_clamp=0
            )

            pred = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(pred, logp + other_log, logp)

            nr, ni = complex_mul(ur, ui, orr, oii)
            ur = cutlass.select_(pred, nr, ur)
            ui = cutlass.select_(pred, ni, ui)

        s_logp[tidx] = logp
        s_phase_scan[tidx, 0] = ur
        s_phase_scan[tidx, 1] = ui
        cute.arch.barrier()

        t0 = cutlass.Int32(31)
        t1 = cutlass.Int32(63)
        t2 = cutlass.Int32(95)
        log0 = cutlass.Float32(s_logp[t0])
        log1 = cutlass.Float32(0.0)
        log2 = cutlass.Float32(0.0)
        p0r = cutlass.Float32(s_phase_scan[t0, 0])
        p0i = cutlass.Float32(s_phase_scan[t0, 1])
        p1r = cutlass.Float32(1.0)
        p1i = cutlass.Float32(0.0)
        p2r = cutlass.Float32(1.0)
        p2i = cutlass.Float32(0.0)
        if cutlass.const_expr(self.num_threads > 32):
            log1 = cutlass.Float32(s_logp[t1])
            p1r = cutlass.Float32(s_phase_scan[t1, 0])
            p1i = cutlass.Float32(s_phase_scan[t1, 1])
        if cutlass.const_expr(self.num_threads > 64):
            log2 = cutlass.Float32(s_logp[t2])
            p2r = cutlass.Float32(s_phase_scan[t2, 0])
            p2i = cutlass.Float32(s_phase_scan[t2, 1])

        off1_log = log0
        off2_log = log0 + log1
        off3_log = log0 + log1 + log2

        off2r, off2i = complex_mul(p1r, p1i, p0r, p0i)
        off3r, off3i = complex_mul(p2r, p2i, off2r, off2i)

        pred_w1 = warp >= cutlass.Int32(1)
        pred_w2 = warp >= cutlass.Int32(2)
        pred_w3 = warp >= cutlass.Int32(3)

        off_log = cutlass.Float32(0.0)
        off_r = cutlass.Float32(1.0)
        off_i = cutlass.Float32(0.0)

        off_log = cutlass.select_(pred_w1, off1_log, off_log)
        off_r = cutlass.select_(pred_w1, p0r, off_r)
        off_i = cutlass.select_(pred_w1, p0i, off_i)

        off_log = cutlass.select_(pred_w2, off2_log, off_log)
        off_r = cutlass.select_(pred_w2, off2r, off_r)
        off_i = cutlass.select_(pred_w2, off2i, off_i)

        off_log = cutlass.select_(pred_w3, off3_log, off_log)
        off_r = cutlass.select_(pred_w3, off3r, off_r)
        off_i = cutlass.select_(pred_w3, off3i, off_i)

        logp = logp + off_log
        ur, ui = complex_mul(ur, ui, off_r, off_i)

        phase_n2 = cutlass.Float32(ur * ur + ui * ui)
        phase_inv = cutlass.Float32(cute.math.rsqrt(phase_n2 + eps))
        ur = ur * phase_inv
        ui = ui * phase_inv

        s_row[tidx] = cute.math.exp2(logp * cutlass.Float32(TWO_LOG2_E), fastmath=True)
        s_phase[tidx, 0] = ur
        s_phase[tidx, 1] = ui

        cute.arch.barrier()

        num_smem_stages = self.num_stages
        k_tile_index = cutlass.Int32(0)
        for st in range(num_smem_stages - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, st],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, st],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        num_k_block = cute.size(tCrA, mode=[2])

        for kt in range(k_tile_count):
            smem_pipe_read = kt % num_smem_stages
            smem_pipe_write = (kt + (num_smem_stages - 1)) % num_smem_stages
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            k_tile_offset = kt * self.bK
            total_a = self.bM * self.bK
            iters_a = (total_a + self.num_threads - 1) // self.num_threads
            for it in range(iters_a):
                idx = tidx + cutlass.Int32(it * self.num_threads)
                if idx < cutlass.Int32(total_a):
                    mm = idx // self.bK
                    kk = idx - mm * self.bK
                    t_global = k_tile_offset + kk
                    a = cutlass.Float32(sA[mm, kk, smem_pipe_read])
                    a = a * cutlass.Float32(s_row[t_global])
                    sA[mm, kk, smem_pipe_read] = a.to(mDOut.element_type)

            nvec = self.bN // 2
            total_b = self.bK * nvec
            iters_b = (total_b + self.num_threads - 1) // self.num_threads
            for it in range(iters_b):
                idx = tidx + cutlass.Int32(it * self.num_threads)
                if idx < cutlass.Int32(total_b):
                    kk = idx // nvec
                    vv = idx - kk * nvec
                    t_global = k_tile_offset + kk
                    d0 = vv * 2
                    xr = cutlass.Float32(sB[d0 + 0, kk, smem_pipe_read])
                    xi = cutlass.Float32(sB[d0 + 1, kk, smem_pipe_read])
                    pr = cutlass.Float32(s_phase[t_global, 0])
                    pi = cutlass.Float32(s_phase[t_global, 1])
                    rr, ri = mul_conj_phase(xr, xi, pr, pi)
                    sB[d0 + 0, kk, smem_pipe_read] = rr.to(mC.element_type)
                    sB[d0 + 1, kk, smem_pipe_read] = ri.to(mC.element_type)

            cute.arch.sync_threads()

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

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.autovec_copy(tCrC, tCsC)
        cute.arch.sync_threads()
        tCrC_ep = cute.make_rmem_tensor_like(tCsC_ep, self.c_dtype)
        cute.autovec_copy(tCsC_ep, tCrC_ep)

        ceilM, ceilN, _ = cute.ceil_div(mDZ0.shape, (self.bM, self.bN, 1))
        mcC = cute.make_identity_tensor(
            (cute.size(ceilM) * self.bM, cute.size(ceilN) * self.bN, 1)
        )
        cC = cute.local_tile(
            mcC[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None),
        )
        tCcC = thr_copy_C.partition_S(cC)
        tCpC = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tCgC_ep.shape[0][1],
                    cute.size(tCgC_ep, mode=[1]),
                    cute.size(tCgC_ep, mode=[2]),
                ),
                stride=(
                    cute.size(tCgC_ep, mode=[1]) * cute.size(tCgC_ep, mode=[2]),
                    cute.size(tCgC_ep, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tCpC.shape[0]):
            for n in range(tCpC.shape[2]):
                n_ok = cute.elem_less(tCcC[(0, rest_v), 0, n][1], mDZ0.shape[1])
                for m in range(tCpC.shape[1]):
                    m_ok = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mDZ0.shape[0])
                    tCpC[rest_v, m, n] = n_ok & m_ok
        for n in range(tCpC.shape[2]):
            cute.copy(
                tiled_copy_C,
                tCrC_ep[None, None, n],
                tCgC_ep[None, None, n],
                pred=tCpC[None, None, n],
            )


__all__ = ["ChunkScanBwdDZ0Ampere"]
