from __future__ import annotations

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import _default_async_copy_bits, _default_tc_k_tile, _next_pow2


@dataclass(frozen=True)
class ChunkIncrementBwdDBLayoutBundle:
    a_major_mode: object
    b_major_mode: object
    sA_layout: object
    sB_layout: object
    sC_layout: object


@dataclass(frozen=True)
class ChunkIncrementBwdDBCopyBundle:
    tiled_copy_A: object
    tiled_copy_B: object
    tiled_copy_s2r_A: object
    tiled_copy_s2r_B: object


@dataclass(frozen=True)
class ChunkIncrementBwdDBKernelBundle:
    layouts: ChunkIncrementBwdDBLayoutBundle
    copies: ChunkIncrementBwdDBCopyBundle
    tiled_mma: object
    compute_smem_bytes: int
    output_smem_bytes: int

    @property
    def smem_size(self) -> int:
        return max(self.compute_smem_bytes, self.output_smem_bytes)


class ChunkIncrementBwdDBAmpere:
    """Ampere tensor-core kernel for ``(dB_main, dM_sum partials)``.

    Computes, per chunk (BHC):
      dB_sum[t, d] = Σ_p U[t, p] * d_inc[p, d]
      dB[t]        = conj(M_sum[t]) * dB_sum[t]

    And writes partial reductions:
      dM_sum_part[t] = Σ_{n in tile} conj(B[t, n]) * dB_sum[t, n]

    The partials are packed as ``(real, imag)`` for each D tile.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        cta_tiler: tuple[int, int, int] | None = None,  # (bM=L, bN=D, bK=P)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int = 2,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        if cta_tiler is None:
            cta_tiler = (self.L, 64, _default_tc_k_tile(self.P))
        self.cta_tiler = cta_tiler
        self.num_stages = int(num_stages)
        self.atom_layout_mnk = atom_layout_mnk

        self.bM, self.bN, self.bK = map(int, self.cta_tiler)
        if self.bM != self.L:
            raise ValueError("This kernel assumes bM == chunk_size (single tile in M).")
        if self.bN % 2 != 0:
            raise ValueError("bN must be divisible by 2 (D=2N).")
        if (self.bN // 2) > 32:
            raise ValueError("bN/2 must fit within one warp for the fused epilogue.")
        if self.bK % 16 != 0:
            raise ValueError("bK must be multiple of 16 for tensor cores.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2")
        if self.P % self.bK != 0:
            raise ValueError("P must be divisible by bK for this kernel.")
        if (self.P // self.bK) < (self.num_stages - 1):
            raise ValueError(
                "P/bK must be >= (num_stages-1) for the cp.async pipeline."
            )

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        atomM, atomN, atomK = self.atom_layout_mnk
        self.num_threads = atomM * atomN * atomK * 32
        if atomK != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bM % (atomM * mmaM) != 0:
            raise ValueError("bM must be divisible by atomM*mmaM.")
        if self.bN % (atomN * mmaN * 2) != 0:
            raise ValueError("bN must be divisible by atomN*mmaN*2.")
        if self.bK % mmaK != 0:
            raise ValueError("bK must be divisible by mmaK.")

        self.scan_threads = _next_pow2(self.L)
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )

    @staticmethod
    def _align_up(offset: int, align: int) -> int:
        return ((offset + align - 1) // align) * align

    @classmethod
    def _struct_size_bytes(cls, fields: list[tuple[int, int]]) -> int:
        offset = 0
        max_align = 1
        for size, align in fields:
            offset = cls._align_up(offset, align)
            offset += size
            max_align = max(max_align, align)
        return cls._align_up(offset, max_align)

    def _alpha_layout(self):
        return cute.make_layout((self.L, 2), stride=(2, 1))

    def _warp_scan_layout(self):
        return cute.make_layout((32,), stride=(1,))

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
        guard_bytes = 2048 + ((pad_bytes + 3) // 4) * 4
        return cute.make_layout((guard_bytes // 4,), stride=(1,))

    def _tail_pad_layout(self):
        return cute.make_layout((64,), stride=(1,))

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

    def _make_layout_bundle(
        self,
        mU: cute.Tensor,
        mDInc_DP: cute.Tensor,
    ) -> ChunkIncrementBwdDBLayoutBundle:
        a_major_mode = utils.LayoutEnum.from_tensor(mU)
        b_major_mode = utils.LayoutEnum.from_tensor(mDInc_DP)
        a_copy_bits = _default_async_copy_bits(
            dtype_width=mU.element_type.width,
            major_mode=a_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        b_copy_bits = _default_async_copy_bits(
            dtype_width=mDInc_DP.element_type.width,
            major_mode=b_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        sA_layout = self._make_smem_layout_AB(
            mU.element_type,
            a_major_mode,
            a_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mDInc_DP.element_type,
            b_major_mode,
            b_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
        return ChunkIncrementBwdDBLayoutBundle(
            a_major_mode=a_major_mode,
            b_major_mode=b_major_mode,
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
        layouts: ChunkIncrementBwdDBLayoutBundle,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        tiled_mma: cute.TiledMma,
    ) -> ChunkIncrementBwdDBCopyBundle:
        a_copy_bits = _default_async_copy_bits(
            dtype_width=in_a_dtype.width,
            major_mode=layouts.a_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        b_copy_bits = _default_async_copy_bits(
            dtype_width=in_b_dtype.width,
            major_mode=layouts.b_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_a_dtype,
            num_bits_per_copy=a_copy_bits,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_b_dtype,
            num_bits_per_copy=b_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy_A,
            in_a_dtype,
            layouts.a_major_mode,
            a_copy_bits,
            tile_m=self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy_B,
            in_b_dtype,
            layouts.b_major_mode,
            b_copy_bits,
            tile_m=self.bN,
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
        return ChunkIncrementBwdDBCopyBundle(
            tiled_copy_A=tiled_copy_A,
            tiled_copy_B=tiled_copy_B,
            tiled_copy_s2r_A=cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma),
            tiled_copy_s2r_B=cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma),
        )

    def _shared_storage_fields(
        self,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
    ) -> list[tuple[int, int]]:
        alpha_layout = self._alpha_layout()
        warp_scan_layout = self._warp_scan_layout()
        output_alias_guard_layout = self._output_alias_guard_layout(
            in_a_dtype, in_b_dtype, sA_layout, sB_layout, sC_layout
        )
        return [
            (cute.cosize(sA_layout) * (in_a_dtype.width // 8), 16),
            (cute.cosize(sB_layout) * (in_b_dtype.width // 8), 16),
            (cute.cosize(output_alias_guard_layout) * 4, 16),
            (cute.cosize(warp_scan_layout) * 4, 4),
            (cute.cosize(warp_scan_layout) * 4, 4),
            (cute.cosize(warp_scan_layout) * 4, 4),
            (cute.cosize(warp_scan_layout) * 4, 4),
            (cute.cosize(alpha_layout) * 4, 4),
            (cute.cosize(alpha_layout) * 4, 4),
            (cute.cosize(self._tail_pad_layout()) * 4, 4),
        ]

    def _make_shared_storage(
        self,
        in_a_dtype: type[cutlass.Numeric],
        in_b_dtype: type[cutlass.Numeric],
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
    ):
        alpha_layout = self._alpha_layout()
        warp_scan_layout = self._warp_scan_layout()
        output_alias_guard_layout = self._output_alias_guard_layout(
            in_a_dtype, in_b_dtype, sA_layout, sB_layout, sC_layout
        )
        tail_pad_layout = self._tail_pad_layout()

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sA": cute.struct.Align[
                cute.struct.MemRange[in_a_dtype, cute.cosize(sA_layout)], 16
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[in_b_dtype, cute.cosize(sB_layout)], 16
            ],
            # Preserve the historical aliasing envelope for the sA -> sC epilogue reuse.
            "output_alias_guard": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(output_alias_guard_layout)
                ],
                16,
            ],
            "warp_re_total": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)],
                4,
            ],
            "warp_im_total": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)],
                4,
            ],
            "warp_re_offset": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)],
                4,
            ],
            "warp_im_offset": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(warp_scan_layout)],
                4,
            ],
            "s_Msum": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)],
                4,
            ],
            "s_Mprev": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)],
                4,
            ],
            "tail_pad": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tail_pad_layout)],
                4,
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self,
        mU: cute.Tensor,
        mDInc_DP: cute.Tensor,
    ) -> ChunkIncrementBwdDBKernelBundle:
        layouts = self._make_layout_bundle(mU, mDInc_DP)
        tiled_mma = self._make_tiled_mma()
        copies = self._make_copy_bundle(
            layouts, mU.element_type, mDInc_DP.element_type, tiled_mma
        )
        return ChunkIncrementBwdDBKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=tiled_mma,
            compute_smem_bytes=self._struct_size_bytes(
                self._shared_storage_fields(
                    mU.element_type,
                    mDInc_DP.element_type,
                    layouts.sA_layout,
                    layouts.sB_layout,
                    layouts.sC_layout,
                )
            ),
            output_smem_bytes=cute.size_in_bytes(self.c_dtype, layouts.sC_layout),
        )

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,  # (L, P, BHC) fp16/bf16
        mB: cute.Tensor,  # (L, D, BHC) fp16/bf16
        mM: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mKprev: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mKcurr: cute.Tensor,  # (2, L, BHC) fp32 packed complex
        mDInc_DP: cute.Tensor,  # (D, P, BHC) fp16/bf16
        mDB: cute.Tensor,  # (L, D, BHC) fp16/bf16
        mDMsum_part: cute.Tensor,  # (2, L, nDtiles, BHC) fp32
    ):
        bundle = self._make_kernel_bundle(mU, mDInc_DP)

        grid_x = cute.ceil_div(self.L, self.bM)
        grid_y = cute.ceil_div(self.D, self.bN)
        grid_z = cute.size(mU.shape[2])

        self.kernel(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDInc_DP,
            mDB,
            mDMsum_part,
            bundle.layouts.sA_layout,
            bundle.layouts.sB_layout,
            bundle.layouts.sC_layout,
            bundle.copies.tiled_copy_A,
            bundle.copies.tiled_copy_B,
            bundle.copies.tiled_copy_s2r_A,
            bundle.copies.tiled_copy_s2r_B,
            bundle.tiled_mma,
        ).launch(
            grid=(cute.size(grid_x), cute.size(grid_y), grid_z),
            block=[self.num_threads, 1, 1],
            smem=bundle.smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDInc_DP: cute.Tensor,
        mDB: cute.Tensor,
        mDMsum_part: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_s2r_A: cute.TiledCopy,
        tiled_copy_s2r_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        tile_n0 = bidy * self.bN

        tiler_coord = (cutlass.Int32(0), bidy, None)
        gA = cute.local_tile(
            mU[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mDInc_DP[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mDB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        smem = cutlass.utils.SmemAllocator()
        SharedStorage = self._make_shared_storage(
            mU.element_type, mDInc_DP.element_type, sA_layout, sB_layout, sC_layout
        )
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Float32), sC_layout
        )
        warp_scan_layout = self._warp_scan_layout()
        alpha_layout = self._alpha_layout()
        warp_re_total = storage.warp_re_total.get_tensor(warp_scan_layout)
        warp_im_total = storage.warp_im_total.get_tensor(warp_scan_layout)
        warp_re_offset = storage.warp_re_offset.get_tensor(warp_scan_layout)
        warp_im_offset = storage.warp_im_offset.get_tensor(warp_scan_layout)
        s_Msum = storage.s_Msum.get_tensor(alpha_layout)
        s_Mprev = storage.s_Mprev.get_tensor(alpha_layout)

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

        mcA = cute.make_identity_tensor(mU.layout.shape)
        mcB = cute.make_identity_tensor(mDInc_DP.layout.shape)
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
                    tAcA[(0, rest_v), m, 0, 0][0], mU.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mDInc_DP.shape[0]
                )

        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        k_tile_count = self.P // self.bK
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for kk in range(tAsA.shape[2]):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, kk, k_tile_index],
                tAsA[None, None, kk, 0],
                pred=tApA[None, None, kk],
            )
        for kk in range(tBsB.shape[2]):
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, kk, k_tile_index],
                tBsB[None, None, kk, 0],
                pred=tBpB[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

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
            r_prev = cute.arch.shuffle_sync_up(qr, offset=1, mask=-1, mask_and_clamp=0)
            i_prev = cute.arch.shuffle_sync_up(qi, offset=1, mask=-1, mask_and_clamp=0)
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

        for kt in range(k_tile_count):
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
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

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        num_warps = cutlass.Int32(self.num_threads // 32)
        nvec = cutlass.Int32(self.bN // 2)
        t = warp
        while cute.elem_less(t, cutlass.Int32(self.L)):
            d0 = lane * cutlass.Int32(2)
            g_d0 = tile_n0 + d0

            dx = cutlass.Float32(0.0)
            dy = cutlass.Float32(0.0)
            bx = cutlass.Float32(0.0)
            by = cutlass.Float32(0.0)
            if cute.elem_less(lane, nvec) and cute.elem_less(g_d0 + 1, self.D):
                dx = cutlass.Float32(sC[t, d0 + 0])
                dy = cutlass.Float32(sC[t, d0 + 1])
                bx = cutlass.Float32(mB[t, g_d0 + 0, bidz].to(cutlass.Float32))
                by = cutlass.Float32(mB[t, g_d0 + 1, bidz].to(cutlass.Float32))

                ar = s_Msum[t, 0]
                ai = s_Msum[t, 1]

                rx = ar * dx + ai * dy
                ry = ar * dy - ai * dx
                mDB[t, g_d0 + 0, bidz] = rx.to(mDB.element_type)
                mDB[t, g_d0 + 1, bidz] = ry.to(mDB.element_type)

            m0 = dx * bx + dy * by
            m1 = dy * bx - dx * by

            for off in (16, 8, 4, 2, 1):
                m0 = m0 + cute.arch.shuffle_sync_bfly(
                    m0, offset=off, mask=-1, mask_and_clamp=31
                )
                m1 = m1 + cute.arch.shuffle_sync_bfly(
                    m1, offset=off, mask=-1, mask_and_clamp=31
                )

            if lane == cutlass.Int32(0):
                mDMsum_part[0, t, bidy, bidz] = m0
                mDMsum_part[1, t, bidy, bidz] = m1

            t = t + num_warps

        return
