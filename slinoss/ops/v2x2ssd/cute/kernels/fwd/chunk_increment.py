"""CuTe forward kernel for the ``v2x2ssd`` chunk-increment stage.

``ChunkIncrementFwdAmpere`` computes one chunk-local increment tile and the
chunk-end transition summary used by the later forward stages. The kernel runs
a reverse-time suffix scan over the complex transition stream ``M``, applies
the two complex taps carried by ``Kprev``/``Kcurr`` to the packed-complex
``B`` stream, accumulates the interior contribution with one tensor-core GEMM,
and adds the chunk-start boundary rank-1 term in the epilogue.

Tensor contracts:

- ``U``: ``(P, T_pad, BH)`` fp16/bf16 value stream
- ``B``: ``(D, T_pad, BH)`` fp16/bf16 packed-complex key stream
- ``M``: ``(2, T_pad, BH)`` fp32 packed-complex transitions
- ``Kprev``: ``(2, T_pad, BH)`` fp32 packed-complex previous-pass taps
- ``Kcurr``: ``(2, T_pad, BH)`` fp32 packed-complex current-pass taps
- ``U_prev0``: ``(P, BH)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(D, BH)`` fp16/bf16 chunk-0 boundary key row
- ``Inc``: ``(P, D, BHC)`` fp32 output increment tiles
- ``Mchunk``: ``(2, BHC)`` fp32 chunk-end packed-complex summaries

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be
even and conceptually corresponds to ``2 * N``.
"""

import math
from dataclasses import dataclass

from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


@dataclass(frozen=True)
class ChunkIncrementLayoutBundle:
    u_major_mode: object
    b_major_mode: object
    increment_major_mode: object
    u_tile_layout: object
    b_tile_layout: object
    increment_tile_layout: object


@dataclass(frozen=True)
class ChunkIncrementCopyBundle:
    gmem_tiled_copy_u: object
    gmem_tiled_copy_b: object
    gmem_tiled_copy_increment: object


@dataclass(frozen=True)
class ChunkIncrementKernelBundle:
    layouts: ChunkIncrementLayoutBundle
    copies: ChunkIncrementCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int


class ChunkIncrementFwdAmpere:
    """Ampere tensor-core forward kernel for the ``v2x2ssd`` chunk-increment op."""

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
        mma_m, mma_n, mma_k = self.mma_inst_shape
        atom_m, atom_n, atom_k = self.atom_layout_mnk

        self.num_threads = atom_m * atom_n * atom_k * 32
        self.scan_threads = 1 << (max(1, self.L) - 1).bit_length()
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )

        if self.bM % (atom_m * mma_m) != 0:
            raise ValueError("bM must be divisible by MMA instruction shape")
        if self.bN % (atom_n * mma_n * 2) != 0:
            raise ValueError("bN must be divisible by MMA instruction shape")
        if atom_k != 1:
            raise ValueError("atom_layout_mnk K must be 1")
        if self.bK % mma_k != 0:
            raise ValueError("bK must be divisible by MMA instruction shape")

    def _suffix_coeff_layout(self):
        return cute.make_layout((self.L, 2), stride=(2, 1))

    def _boundary_value_smem_layout(self):
        return cute.make_layout((self.bM,))

    def _boundary_key_smem_layout(self):
        return cute.make_layout((self.bN,))

    def _warp_transition_layout(self):
        return cute.make_layout((max(1, self.scan_threads // 32), 2), stride=(2, 1))

    def _make_operand_smem_layout(self, dtype, major_mode, copy_bits, smem_tiler):
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

    def _make_operand_gmem_tiled_copy(
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

    def _make_increment_gmem_tiled_copy(self, atom_copy, dtype, major_mode, copy_bits):
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
        u_major_mode = utils.LayoutEnum.from_tensor(mU)
        b_major_mode = utils.LayoutEnum.from_tensor(mB)
        increment_major_mode = utils.LayoutEnum.from_tensor(mInc)

        ab_copy_bits = 128
        u_tile_layout = self._make_operand_smem_layout(
            mU.element_type,
            u_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        b_tile_layout = self._make_operand_smem_layout(
            mB.element_type,
            b_major_mode,
            ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        increment_tile_layout = cute.make_layout(
            (self.bM, self.bN), stride=(self.bN, 1)
        )
        return ChunkIncrementLayoutBundle(
            u_major_mode=u_major_mode,
            b_major_mode=b_major_mode,
            increment_major_mode=increment_major_mode,
            u_tile_layout=u_tile_layout,
            b_tile_layout=b_tile_layout,
            increment_tile_layout=increment_tile_layout,
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
        gmem_tiled_copy_u = self._make_operand_gmem_tiled_copy(
            atom_async_copy,
            in_dtype,
            layouts.u_major_mode,
            copy_bits,
            tile_m=self.bM,
        )
        gmem_tiled_copy_b = self._make_operand_gmem_tiled_copy(
            atom_async_copy,
            in_dtype,
            layouts.b_major_mode,
            copy_bits,
            tile_m=self.bN,
        )

        # The chunk-increment epilogue writes an f32 accumulator tile through a
        # row-major (P, D, BHC) view. Under TVM FFI the destination tile is not
        # universally 128-bit aligned, so the output copy must stay scalar-width.
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=32,
        )
        gmem_tiled_copy_increment = self._make_increment_gmem_tiled_copy(
            atom_sync_copy,
            self.c_dtype,
            layouts.increment_major_mode,
            32,
        )
        return ChunkIncrementCopyBundle(
            gmem_tiled_copy_u=gmem_tiled_copy_u,
            gmem_tiled_copy_b=gmem_tiled_copy_b,
            gmem_tiled_copy_increment=gmem_tiled_copy_increment,
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

    def _make_accumulator_mn_view(
        self, accumulator_fragment: cute.Tensor
    ) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(accumulator_fragment.layout.shape)
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
        acc_layout_mn = cute.composition(accumulator_fragment.layout, acc_layout_mn)
        return cute.make_tensor(accumulator_fragment.iterator, acc_layout_mn)

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkIncrementLayoutBundle,
    ):
        suffix_coeff_layout = self._suffix_coeff_layout()
        boundary_value_layout = self._boundary_value_smem_layout()
        boundary_key_layout = self._boundary_key_smem_layout()
        warp_transition_layout = self._warp_transition_layout()

        @cute.struct
        class SharedStorage:
            u_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.u_tile_layout)], 16
            ]
            b_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.b_tile_layout)], 16
            ]
            increment_tile: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.increment_tile_layout)
                ],
                16,
            ]
            suffix_coeff_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            suffix_coeff_curr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            boundary_value: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(boundary_value_layout)
                ],
                4,
            ]
            boundary_key: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(boundary_key_layout)],
                4,
            ]
            warp_transition_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                8,
            ]
            warp_transition_offset: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                8,
            ]

        return SharedStorage

    def _make_kernel_bundle(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mInc: cute.Tensor,
    ) -> ChunkIncrementKernelBundle:
        layouts = self._make_layout_bundle(mU, mB, mInc)
        copies = self._make_copy_bundle(layouts, mU.element_type)
        shared_storage_cls = self._make_shared_storage(mU.element_type, layouts)
        return ChunkIncrementKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=self._make_tiled_mma(),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    @cute.jit
    def _validate_operands(
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
    ):
        value_stream_dtype_ok = (
            mU.element_type
            == mB.element_type
            == mU_prev0.element_type
            == mB_prev0.element_type
        )
        output_dtype_ok = (
            mKprev.element_type
            == mKcurr.element_type
            == mInc.element_type
            == mMchunk.element_type
            == cutlass.Float32
        )
        time_dims_match = (
            mU.shape[1]
            == mB.shape[1]
            == mM.shape[1]
            == mKprev.shape[1]
            == mKcurr.shape[1]
        )
        batch_head_dims_match = (
            mU.shape[2]
            == mB.shape[2]
            == mM.shape[2]
            == mKprev.shape[2]
            == mKcurr.shape[2]
        )

        if cutlass.const_expr(not value_stream_dtype_ok):
            raise TypeError("U/B/U_prev0/B_prev0 must share element type.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/U_prev0/B_prev0 must be Float16 or BFloat16.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(not output_dtype_ok):
            raise TypeError("Kprev/Kcurr/Inc/Mchunk must be Float32.")
        if cutlass.const_expr(mB.shape[0] % 2 != 0):
            raise ValueError("B last dimension must be even because D stores pairs.")
        if cutlass.const_expr(not time_dims_match):
            raise ValueError("U/B/M/Kprev/Kcurr must share the padded time dimension.")
        if cutlass.const_expr(not batch_head_dims_match):
            raise ValueError("U/B/M/Kprev/Kcurr must share the batch-head dimension.")
        if cutlass.const_expr(
            not (mU_prev0.shape[1] == mB_prev0.shape[1] == mU.shape[2])
        ):
            raise ValueError("U_prev0/B_prev0 must match the batch-head dimension.")
        if cutlass.const_expr(mU_prev0.shape[0] != mU.shape[0]):
            raise ValueError("U_prev0 rows must match U rows.")
        if cutlass.const_expr(mB_prev0.shape[0] != mB.shape[0]):
            raise ValueError("B_prev0 rows must match B rows.")
        if cutlass.const_expr(
            mInc.shape[0] != mU.shape[0] or mInc.shape[1] != mB.shape[0]
        ):
            raise ValueError("Inc must have leading shape (P, D) matching U/B.")
        if cutlass.const_expr(mInc.shape[2] % mU.shape[2] != 0):
            raise ValueError("Inc batch-head-chunk dimension must be divisible by BH.")
        if cutlass.const_expr(
            not (mMchunk.shape[0] == 2 and mMchunk.shape[1] == mInc.shape[2])
        ):
            raise ValueError("Mchunk must have shape (2, BHC) matching Inc.")

    def _launch_kernel(
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
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mU, mB, mInc)
        grid_dim = cute.ceil_div(mInc.shape, (self.bM, self.bN, 1))
        launch_kwargs = {
            "grid": (
                cute.size(grid_dim[0]),
                cute.size(grid_dim[1]),
                cute.size(mInc.shape[2]),
            ),
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_bytes,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

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
            bundle.layouts.u_major_mode,
            bundle.layouts.b_major_mode,
            bundle.layouts.u_tile_layout,
            bundle.layouts.b_tile_layout,
            bundle.layouts.increment_tile_layout,
            bundle.copies.gmem_tiled_copy_u,
            bundle.copies.gmem_tiled_copy_b,
            bundle.copies.gmem_tiled_copy_increment,
            bundle.tiled_mma,
            bundle.shared_storage_cls,
        ).launch(**launch_kwargs)

    @cute.jit
    def _validate_and_launch(
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
        stream: cuda.CUstream | None = None,
    ):
        self._validate_operands(
            mU, mB, mM, mKprev, mKcurr, mU_prev0, mB_prev0, mInc, mMchunk
        )
        self._launch_kernel(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mU_prev0,
            mB_prev0,
            mInc,
            mMchunk,
            stream=stream,
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
        self._validate_and_launch(
            mU, mB, mM, mKprev, mKcurr, mU_prev0, mB_prev0, mInc, mMchunk
        )

    @cute.jit
    def call_on_stream(
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
        stream: cuda.CUstream,
    ):
        self._validate_and_launch(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mU_prev0,
            mB_prev0,
            mInc,
            mMchunk,
            stream=stream,
        )

    @cute.jit
    def _make_copy_tile_row_predicate(
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
        u_major_mode: cutlass.Constexpr,
        b_major_mode: cutlass.Constexpr,
        u_tile_layout: cute.ComposedLayout,
        b_tile_layout: cute.ComposedLayout,
        increment_tile_layout: cute.Layout,
        gmem_tiled_copy_u: cute.TiledCopy,
        gmem_tiled_copy_b: cute.TiledCopy,
        gmem_tiled_copy_increment: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        batch_head_count = mU.shape[2]
        batch_head_chunk_count = mInc.shape[2]
        chunk_count = batch_head_chunk_count // batch_head_count

        batch_head = bidz // chunk_count
        chunk_index = bidz - batch_head * chunk_count
        chunk_start = chunk_index * self.L

        tiler_coord = (bidx, bidy, None)

        # CTA tile setup.
        g_increment = cute.local_tile(
            mInc[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_u_tile = storage.u_tile.get_tensor(u_tile_layout)
        s_b_tile = storage.b_tile.get_tensor(b_tile_layout)
        s_increment_tile = storage.increment_tile.get_tensor(increment_tile_layout)
        suffix_coeff_layout = self._suffix_coeff_layout()
        warp_transition_layout = self._warp_transition_layout()
        s_suffix_coeff_prev = storage.suffix_coeff_prev.get_tensor(suffix_coeff_layout)
        s_suffix_coeff_curr = storage.suffix_coeff_curr.get_tensor(suffix_coeff_layout)
        s_boundary_value = storage.boundary_value.get_tensor(
            self._boundary_value_smem_layout()
        )
        s_boundary_key = storage.boundary_key.get_tensor(
            self._boundary_key_smem_layout()
        )
        s_warp_transition_total = storage.warp_transition_total.get_tensor(
            warp_transition_layout
        )
        s_warp_transition_offset = storage.warp_transition_offset.get_tensor(
            warp_transition_layout
        )

        copy_slice_u = gmem_tiled_copy_u.get_slice(tidx)
        copy_slice_b = gmem_tiled_copy_b.get_slice(tidx)
        copy_slice_increment = gmem_tiled_copy_increment.get_slice(tidx)

        t_u_copy_dst = copy_slice_u.partition_D(s_u_tile)
        t_b_copy_dst = copy_slice_b.partition_D(s_b_tile)
        t_increment_store_smem = copy_slice_increment.partition_S(s_increment_tile)
        t_increment_store_gmem = copy_slice_increment.partition_D(g_increment)

        mma_slice = tiled_mma.get_slice(tidx)
        t_u_mma_smem = mma_slice.partition_A(s_u_tile)
        t_b_mma_smem = mma_slice.partition_B(s_b_tile)
        t_increment_mma_smem = mma_slice.partition_C(s_increment_tile)
        t_increment_mma_gmem = mma_slice.partition_C(g_increment)

        r_u_mma = tiled_mma.make_fragment_A(t_u_mma_smem[None, None, None, 0])
        r_b_mma = tiled_mma.make_fragment_B(t_b_mma_smem[None, None, None, 0])
        r_increment_accum = tiled_mma.make_fragment_C(t_increment_mma_gmem)
        r_increment_accum.fill(0.0)

        ldmatrix_copy_atom_u = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                u_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mU.element_type,
        )
        ldmatrix_copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mB.element_type,
        )
        s2r_tiled_copy_u = cute.make_tiled_copy_A(ldmatrix_copy_atom_u, tiled_mma)
        s2r_tiled_copy_b = cute.make_tiled_copy_B(ldmatrix_copy_atom_b, tiled_mma)
        ldmatrix_slice_u = s2r_tiled_copy_u.get_slice(tidx)
        ldmatrix_slice_b = s2r_tiled_copy_b.get_slice(tidx)
        t_u_ldmatrix_smem = ldmatrix_slice_u.partition_S(s_u_tile)
        r_u_ldmatrix = ldmatrix_slice_u.retile(r_u_mma)
        t_b_ldmatrix_smem = ldmatrix_slice_b.partition_S(s_b_tile)
        r_b_ldmatrix = ldmatrix_slice_b.retile(r_b_mma)

        u_coord_identity = cute.make_identity_tensor(mU.layout.shape)
        b_coord_identity = cute.make_identity_tensor(mB.layout.shape)
        u_coord_chunk = cute.domain_offset((0, chunk_start, 0), u_coord_identity)
        b_coord_chunk = cute.domain_offset((0, chunk_start, 0), b_coord_identity)
        u_coord_tile = cute.local_tile(
            u_coord_chunk[None, None, batch_head],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        b_coord_tile = cute.local_tile(
            b_coord_chunk[None, None, batch_head],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        t_u_coord = copy_slice_u.partition_S(u_coord_tile)
        t_b_coord = copy_slice_b.partition_S(b_coord_tile)

        u_row_predicate = self._make_copy_tile_row_predicate(
            t_u_copy_dst, t_u_coord, mU.shape[0]
        )
        b_row_predicate = self._make_copy_tile_row_predicate(
            t_b_copy_dst, t_b_coord, mB.shape[0]
        )

        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        reverse_time_index = cutlass.Int32(self.L - 1) - tidx
        global_time_index = chunk_start + reverse_time_index

        transition_step_re = cutlass.Float32(1.0)
        transition_step_im = cutlass.Float32(0.0)
        prev_tap_re = cutlass.Float32(0.0)
        prev_tap_im = cutlass.Float32(0.0)
        curr_tap_re = cutlass.Float32(0.0)
        curr_tap_im = cutlass.Float32(0.0)

        # Reverse-time suffix scan over the packed-complex transition stream.
        if tidx < cutlass.Int32(self.L):
            transition_step_re = cutlass.Float32(
                mM[0, global_time_index, batch_head].to(cutlass.Float32)
            )
            transition_step_im = cutlass.Float32(
                mM[1, global_time_index, batch_head].to(cutlass.Float32)
            )
            prev_tap_re = cutlass.Float32(
                mKprev[0, global_time_index, batch_head].to(cutlass.Float32)
            )
            prev_tap_im = cutlass.Float32(
                mKprev[1, global_time_index, batch_head].to(cutlass.Float32)
            )
            curr_tap_re = cutlass.Float32(
                mKcurr[0, global_time_index, batch_head].to(cutlass.Float32)
            )
            curr_tap_im = cutlass.Float32(
                mKcurr[1, global_time_index, batch_head].to(cutlass.Float32)
            )

        scan_threads = cutlass.Int32(self.scan_threads)
        scan_warp_count = cutlass.Int32(self.scan_threads // 32)
        in_scan = tidx < scan_threads

        suffix_transition_re = cutlass.select_(
            in_scan, transition_step_re, cutlass.Float32(1.0)
        )
        suffix_transition_im = cutlass.select_(
            in_scan, transition_step_im, cutlass.Float32(0.0)
        )

        if warp_idx < scan_warp_count:
            for offset in (1, 2, 4, 8, 16):
                prior_re = cute.arch.shuffle_sync_up(
                    suffix_transition_re, offset=offset, mask=-1, mask_and_clamp=0
                )
                prior_im = cute.arch.shuffle_sync_up(
                    suffix_transition_im, offset=offset, mask=-1, mask_and_clamp=0
                )
                should_update = lane_idx >= cutlass.Int32(offset)
                next_re = (
                    prior_re * suffix_transition_re - prior_im * suffix_transition_im
                )
                next_im = (
                    prior_re * suffix_transition_im + prior_im * suffix_transition_re
                )
                suffix_transition_re = cutlass.select_(
                    should_update, next_re, suffix_transition_re
                )
                suffix_transition_im = cutlass.select_(
                    should_update, next_im, suffix_transition_im
                )

            if lane_idx == cutlass.Int32(31):
                s_warp_transition_total[warp_idx, 0] = suffix_transition_re
                s_warp_transition_total[warp_idx, 1] = suffix_transition_im

        cute.arch.sync_threads()

        if cutlass.const_expr(self.scan_threads > 32):
            if warp_idx == cutlass.Int32(0):
                warp_lane = lane_idx
                has_scan_warp = warp_lane < scan_warp_count

                warp_transition_re = cutlass.select_(
                    has_scan_warp,
                    s_warp_transition_total[warp_lane, 0],
                    cutlass.Float32(1.0),
                )
                warp_transition_im = cutlass.select_(
                    has_scan_warp,
                    s_warp_transition_total[warp_lane, 1],
                    cutlass.Float32(0.0),
                )

                for offset in (1, 2, 4, 8, 16):
                    prior_re = cute.arch.shuffle_sync_up(
                        warp_transition_re, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    prior_im = cute.arch.shuffle_sync_up(
                        warp_transition_im, offset=offset, mask=-1, mask_and_clamp=0
                    )
                    should_update = lane_idx >= cutlass.Int32(offset)
                    next_re = (
                        prior_re * warp_transition_re - prior_im * warp_transition_im
                    )
                    next_im = (
                        prior_re * warp_transition_im + prior_im * warp_transition_re
                    )
                    warp_transition_re = cutlass.select_(
                        should_update, next_re, warp_transition_re
                    )
                    warp_transition_im = cutlass.select_(
                        should_update, next_im, warp_transition_im
                    )

                warp_offset_re = cute.arch.shuffle_sync_up(
                    warp_transition_re, offset=1, mask=-1, mask_and_clamp=0
                )
                warp_offset_im = cute.arch.shuffle_sync_up(
                    warp_transition_im, offset=1, mask=-1, mask_and_clamp=0
                )
                is_first_lane = lane_idx == cutlass.Int32(0)
                warp_offset_re = cutlass.select_(
                    is_first_lane, cutlass.Float32(1.0), warp_offset_re
                )
                warp_offset_im = cutlass.select_(
                    is_first_lane, cutlass.Float32(0.0), warp_offset_im
                )

                if has_scan_warp:
                    s_warp_transition_offset[warp_lane, 0] = warp_offset_re
                    s_warp_transition_offset[warp_lane, 1] = warp_offset_im

            cute.arch.sync_threads()

            if warp_idx < scan_warp_count:
                warp_offset_re = s_warp_transition_offset[warp_idx, 0]
                warp_offset_im = s_warp_transition_offset[warp_idx, 1]
                next_re = (
                    warp_offset_re * suffix_transition_re
                    - warp_offset_im * suffix_transition_im
                )
                next_im = (
                    warp_offset_re * suffix_transition_im
                    + warp_offset_im * suffix_transition_re
                )
                suffix_transition_re, suffix_transition_im = next_re, next_im

        if warp_idx < scan_warp_count and lane_idx == cutlass.Int32(31):
            s_warp_transition_total[warp_idx, 0] = suffix_transition_re
            s_warp_transition_total[warp_idx, 1] = suffix_transition_im

        cute.arch.sync_threads()

        # Convert the inclusive suffix products into per-step tap coefficients.
        suffix_prefix_re = cutlass.Float32(1.0)
        suffix_prefix_im = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            prior_transition_re = cute.arch.shuffle_sync_up(
                suffix_transition_re, offset=1, mask=-1, mask_and_clamp=0
            )
            prior_transition_im = cute.arch.shuffle_sync_up(
                suffix_transition_im, offset=1, mask=-1, mask_and_clamp=0
            )

            if tidx == cutlass.Int32(0):
                suffix_prefix_re = cutlass.Float32(1.0)
                suffix_prefix_im = cutlass.Float32(0.0)
            else:
                if lane_idx == cutlass.Int32(0):
                    suffix_prefix_re = s_warp_transition_total[warp_idx - 1, 0]
                    suffix_prefix_im = s_warp_transition_total[warp_idx - 1, 1]
                else:
                    suffix_prefix_re = prior_transition_re
                    suffix_prefix_im = prior_transition_im

            prev_coeff_re = (
                suffix_prefix_re * prev_tap_re - suffix_prefix_im * prev_tap_im
            )
            prev_coeff_im = (
                suffix_prefix_re * prev_tap_im + suffix_prefix_im * prev_tap_re
            )
            curr_coeff_re = (
                suffix_prefix_re * curr_tap_re - suffix_prefix_im * curr_tap_im
            )
            curr_coeff_im = (
                suffix_prefix_re * curr_tap_im + suffix_prefix_im * curr_tap_re
            )

            s_suffix_coeff_prev[reverse_time_index, 0] = prev_coeff_re
            s_suffix_coeff_prev[reverse_time_index, 1] = prev_coeff_im
            s_suffix_coeff_curr[reverse_time_index, 0] = curr_coeff_re
            s_suffix_coeff_curr[reverse_time_index, 1] = curr_coeff_im

            if (bidx == 0) and (bidy == 0) and (reverse_time_index == cutlass.Int32(0)):
                mMchunk[0, bidz] = suffix_transition_re.to(cutlass.Float32)
                mMchunk[1, bidz] = suffix_transition_im.to(cutlass.Float32)

        cute.arch.sync_threads()

        # Pipelined tensor-core mainloop over the chunk-local time tiles.
        num_smem_stages = cute.size(t_u_copy_dst, mode=[3])
        k_tile_count = self.L // self.bK

        mU_off = cute.domain_offset((0, chunk_start, 0), mU)
        mB_off = cute.domain_offset((0, chunk_start, 0), mB)

        g_u = cute.local_tile(
            mU_off[None, None, batch_head],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        g_b = cute.local_tile(
            mB_off[None, None, batch_head],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        g_u = cute.make_tensor(g_u.iterator.align(16), g_u.layout)
        g_b = cute.make_tensor(g_b.iterator.align(16), g_b.layout)

        t_u_gmem = copy_slice_u.partition_S(g_u)
        t_b_gmem = copy_slice_b.partition_S(g_b)

        t_u_copy_dst.fill(0)
        t_b_copy_dst.fill(0)
        cute.arch.sync_threads()
        k_tile_index = cutlass.Int32(0)

        for kk in cutlass.range_constexpr(u_row_predicate.shape[2]):
            cute.copy(
                gmem_tiled_copy_u,
                t_u_gmem[None, None, kk, k_tile_index],
                t_u_copy_dst[None, None, kk, 0],
                pred=u_row_predicate[None, None, kk],
            )
        for kk in cutlass.range_constexpr(b_row_predicate.shape[2]):
            cute.copy(
                gmem_tiled_copy_b,
                t_b_gmem[None, None, kk, k_tile_index],
                t_b_copy_dst[None, None, kk, 0],
                pred=b_row_predicate[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile >= k_tile_count:
                u_row_predicate.fill(0)
                b_row_predicate.fill(0)
            cute.copy(
                gmem_tiled_copy_u,
                t_u_gmem[None, None, None, k_tile_index],
                t_u_copy_dst[None, None, None, k_tile],
                pred=u_row_predicate,
            )
            cute.copy(
                gmem_tiled_copy_b,
                t_b_gmem[None, None, None, k_tile_index],
                t_b_copy_dst[None, None, None, k_tile],
                pred=b_row_predicate,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(num_smem_stages - 1)
        mma_k_block_count = cute.size(r_u_mma, mode=[2])

        complex_pair_count = self.bN // 2
        stage_transform_count = self.bK * complex_pair_count
        stage_transform_iters = (
            stage_transform_count + self.num_threads - 1
        ) // self.num_threads

        for kt in range(k_tile_count):
            if cutlass.const_expr(num_smem_stages == 1):
                cute.arch.cp_async_wait_group(0)
            else:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            k_tile_offset = kt * self.bK

            for it in cutlass.range_constexpr(stage_transform_iters):
                idx = tidx + (it * self.num_threads)
                if cute.elem_less(idx, stage_transform_count):
                    k_step = idx // complex_pair_count
                    complex_pair = idx - k_step * complex_pair_count
                    d_pair_start = complex_pair * 2
                    time_step = k_tile_offset + k_step

                    b_re = s_b_tile[d_pair_start + 0, k_step, smem_pipe_read].to(
                        cutlass.Float32
                    )
                    b_im = s_b_tile[d_pair_start + 1, k_step, smem_pipe_read].to(
                        cutlass.Float32
                    )

                    coeff_re = s_suffix_coeff_curr[time_step, 0]
                    coeff_im = s_suffix_coeff_curr[time_step, 1]
                    if time_step < cutlass.Int32(self.L - 1):
                        coeff_re = coeff_re + s_suffix_coeff_prev[time_step + 1, 0]
                        coeff_im = coeff_im + s_suffix_coeff_prev[time_step + 1, 1]

                    rotated_re = coeff_re * b_re - coeff_im * b_im
                    rotated_im = coeff_re * b_im + coeff_im * b_re

                    s_b_tile[d_pair_start + 0, k_step, smem_pipe_read] = rotated_re.to(
                        mB.element_type
                    )
                    s_b_tile[d_pair_start + 1, k_step, smem_pipe_read] = rotated_im.to(
                        mB.element_type
                    )

            cute.arch.sync_threads()

            if cutlass.const_expr(num_smem_stages > 1):
                next_tile = kt + (num_smem_stages - 1)
                if next_tile < k_tile_count:
                    cute.copy(
                        gmem_tiled_copy_u,
                        t_u_gmem[None, None, None, k_tile_index],
                        t_u_copy_dst[None, None, None, smem_pipe_write],
                        pred=u_row_predicate,
                    )
                    cute.copy(
                        gmem_tiled_copy_b,
                        t_b_gmem[None, None, None, k_tile_index],
                        t_b_copy_dst[None, None, None, smem_pipe_write],
                        pred=b_row_predicate,
                    )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()

            t_u_ldmatrix_stage = t_u_ldmatrix_smem[None, None, None, smem_pipe_read]
            t_b_ldmatrix_stage = t_b_ldmatrix_smem[None, None, None, smem_pipe_read]

            cute.copy(
                s2r_tiled_copy_u,
                t_u_ldmatrix_stage[None, None, 0],
                r_u_ldmatrix[None, None, 0],
            )
            cute.copy(
                s2r_tiled_copy_b,
                t_b_ldmatrix_stage[None, None, 0],
                r_b_ldmatrix[None, None, 0],
            )
            for kb in cutlass.range(mma_k_block_count, unroll_full=True):
                kb_next = (kb + 1) % mma_k_block_count
                cute.copy(
                    s2r_tiled_copy_u,
                    t_u_ldmatrix_stage[None, None, kb_next],
                    r_u_ldmatrix[None, None, kb_next],
                )
                cute.copy(
                    s2r_tiled_copy_b,
                    t_b_ldmatrix_stage[None, None, kb_next],
                    r_b_ldmatrix[None, None, kb_next],
                )
                cute.gemm(
                    tiled_mma,
                    r_increment_accum,
                    r_u_mma[None, None, kb],
                    r_b_mma[None, None, kb],
                    r_increment_accum,
                )

            smem_pipe_write = smem_pipe_read
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == num_smem_stages:
                smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Boundary correction and epilogue store.
        tile_row_start = bidx * self.bM
        tile_col_start = bidy * self.bN

        row_index = tidx
        while cute.elem_less(row_index, self.bM):
            global_row = tile_row_start + row_index
            if cute.elem_less(global_row, mU.shape[0]):
                if chunk_index == 0:
                    s_boundary_value[row_index] = cutlass.Float32(
                        mU_prev0[global_row, batch_head].to(cutlass.Float32)
                    )
                else:
                    s_boundary_value[row_index] = cutlass.Float32(
                        mU[global_row, chunk_start - 1, batch_head].to(cutlass.Float32)
                    )
            else:
                s_boundary_value[row_index] = cutlass.Float32(0.0)
            row_index = row_index + self.num_threads

        boundary_coeff_re = s_suffix_coeff_prev[0, 0]
        boundary_coeff_im = s_suffix_coeff_prev[0, 1]

        complex_pair = tidx
        complex_pair_count = self.bN // 2
        while cute.elem_less(complex_pair, complex_pair_count):
            d_pair_start = complex_pair * 2
            global_d_pair_start = tile_col_start + d_pair_start

            boundary_key_re = cutlass.Float32(0.0)
            boundary_key_im = cutlass.Float32(0.0)
            if cute.elem_less(global_d_pair_start + 1, mB.shape[0]):
                if chunk_index == 0:
                    boundary_key_re = cutlass.Float32(
                        mB_prev0[global_d_pair_start + 0, batch_head].to(
                            cutlass.Float32
                        )
                    )
                    boundary_key_im = cutlass.Float32(
                        mB_prev0[global_d_pair_start + 1, batch_head].to(
                            cutlass.Float32
                        )
                    )
                else:
                    boundary_key_re = cutlass.Float32(
                        mB[global_d_pair_start + 0, chunk_start - 1, batch_head].to(
                            cutlass.Float32
                        )
                    )
                    boundary_key_im = cutlass.Float32(
                        mB[global_d_pair_start + 1, chunk_start - 1, batch_head].to(
                            cutlass.Float32
                        )
                    )

            rotated_boundary_re = (
                boundary_coeff_re * boundary_key_re
                - boundary_coeff_im * boundary_key_im
            )
            rotated_boundary_im = (
                boundary_coeff_re * boundary_key_im
                + boundary_coeff_im * boundary_key_re
            )

            s_boundary_key[d_pair_start + 0] = rotated_boundary_re
            s_boundary_key[d_pair_start + 1] = rotated_boundary_im
            complex_pair = complex_pair + self.num_threads

        cute.arch.sync_threads()

        ceil_tile_rows, ceil_tile_cols, _ = cute.ceil_div(
            mInc.shape, (self.bM, self.bN, 1)
        )
        increment_coord_identity = cute.make_identity_tensor(
            (
                cute.size(ceil_tile_rows) * self.bM,
                cute.size(ceil_tile_cols) * self.bN,
                1,
            )
        )
        increment_coord_tile = cute.local_tile(
            increment_coord_identity[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        if cutlass.const_expr(num_smem_stages == 1):
            # Direct-store epilogue for the single-stage specialization.
            t_increment_coord = mma_slice.partition_C(increment_coord_tile)
            t_increment_coord_mn = self._make_accumulator_mn_view(t_increment_coord)
            r_increment_accum_mn = self._make_accumulator_mn_view(r_increment_accum)

            for r in cutlass.range_constexpr(cute.size(r_increment_accum_mn.shape[0])):
                for c in cutlass.range_constexpr(
                    cute.size(r_increment_accum_mn.shape[1])
                ):
                    row_idx = cutlass.Int32(t_increment_coord_mn[r, c][0])
                    col_idx = cutlass.Int32(t_increment_coord_mn[r, c][1])
                    if cute.elem_less(row_idx, mInc.shape[0]) and cute.elem_less(
                        col_idx, mInc.shape[1]
                    ):
                        row_local = row_idx - tile_row_start
                        col_local = col_idx - tile_col_start
                        mInc[row_idx, col_idx, bidz] = (
                            cutlass.Float32(r_increment_accum_mn[r, c])
                            + s_boundary_value[row_local] * s_boundary_key[col_local]
                        )
        else:
            # Shared-memory staged epilogue for the multi-stage pipeline.
            t_increment_coord = copy_slice_increment.partition_S(increment_coord_tile)

            cute.autovec_copy(r_increment_accum, t_increment_mma_smem)
            cute.arch.sync_threads()

            total_elems = self.bM * self.bN
            idx = tidx
            while cute.elem_less(idx, total_elems):
                m_idx = idx // self.bN
                n_idx = idx - (m_idx * self.bN)
                s_increment_tile[m_idx, n_idx] = (
                    s_increment_tile[m_idx, n_idx]
                    + s_boundary_value[m_idx] * s_boundary_key[n_idx]
                )
                idx = idx + self.num_threads

            cute.arch.sync_threads()

            r_increment_store = cute.make_fragment_like(t_increment_store_smem)
            cute.autovec_copy(t_increment_store_smem, r_increment_store)

            increment_store_predicate = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        t_increment_store_gmem.shape[0][1],
                        cute.size(t_increment_store_gmem, mode=[1]),
                        cute.size(t_increment_store_gmem, mode=[2]),
                    ),
                    stride=(cute.size(t_increment_store_gmem, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(increment_store_predicate.shape[0]):
                for m in cutlass.range_constexpr(increment_store_predicate.shape[1]):
                    increment_store_predicate[rest_v, m, 0] = cute.elem_less(
                        t_increment_coord[(0, rest_v), m, 0][0], mInc.shape[0]
                    )

            for rest_v in cutlass.range_constexpr(increment_store_predicate.shape[0]):
                for n in cutlass.range_constexpr(increment_store_predicate.shape[2]):
                    if cute.elem_less(
                        t_increment_coord[(0, rest_v), 0, n][1], mInc.shape[1]
                    ):
                        cute.copy(
                            gmem_tiled_copy_increment,
                            r_increment_store[None, None, n],
                            t_increment_store_gmem[None, None, n],
                            pred=increment_store_predicate[None, None, n],
                        )
