"""CuTe backward kernel for the ``v2x2ssd`` chunk-increment ``dU`` stage.

``ChunkIncrementBwdDUAmpere`` computes the chunk-local interior ``dU`` tile
used by the staged chunk-increment backward. The kernel runs a reverse-time
suffix scan over the complex transition stream ``M``, converts the
``Kprev``/``Kcurr`` taps into per-step packed-complex suffix coefficients,
rotates the chunk-local ``B`` stream in shared memory, and accumulates the
interior gradient tile with one tensor-core GEMM.

Tensor contracts:

- ``DInc``: ``(P, D, BHC)`` fp16/bf16 chunk-local ``d_inc`` tiles
- ``B``: ``(L, D, BHC)`` fp16/bf16 packed-complex key stream
- ``M``: ``(2, L, BHC)`` fp32 packed-complex transitions
- ``Kprev``: ``(2, L, BHC)`` fp32 packed-complex previous-pass taps
- ``Kcurr``: ``(2, L, BHC)`` fp32 packed-complex current-pass taps
- ``DU``: ``(P, L, BHC)`` fp16/bf16 chunk-local output gradients

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be
even and conceptually corresponds to ``2 * N``.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import _default_async_copy_bits, _default_tc_k_tile, _next_pow2


@dataclass(frozen=True)
class ChunkIncrementBwdDULayoutBundle:
    d_inc_major_mode: object
    b_major_mode: object
    d_inc_tile_layout: object
    b_tile_layout: object


@dataclass(frozen=True)
class ChunkIncrementBwdDUCopyBundle:
    gmem_tiled_copy_d_inc: object
    gmem_tiled_copy_b: object


@dataclass(frozen=True)
class ChunkIncrementBwdDUKernelBundle:
    layouts: ChunkIncrementBwdDULayoutBundle
    copies: ChunkIncrementBwdDUCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkIncrementBwdDUSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkIncrementBwdDUAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dU`` stage."""

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkIncrementBwdDUSupportInfo]
    ] = {}

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        cta_tiler: tuple[int, int, int] | None = None,  # (bM=P, bN=L, bK)
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int | None = None,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32

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
        if (
            self.D == 256
            and self.bK == 32
            and self.bN == self.L
            and self.num_stages > 2
        ):
            # The training-hot DU shape has eight K tiles, so double buffering
            # is sufficient and recovers one staged A/B slab.
            self.num_stages = 2

        self.mma_inst_shape = (16, 8, 16)
        mma_m, mma_n, mma_k = self.mma_inst_shape
        atom_m, atom_n, atom_k = self.atom_layout_mnk

        self.num_threads = atom_m * atom_n * atom_k * 32
        self.scan_threads = _next_pow2(self.L)
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )

        if atom_k != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bM % (atom_m * mma_m) != 0:
            raise ValueError("bM must be divisible by atomM*mmaM.")
        if self.bN % (atom_n * mma_n * 2) != 0:
            raise ValueError("bN must be divisible by atomN*mmaN*2.")
        if self.bK % mma_k != 0:
            raise ValueError("bK must be divisible by mmaK.")

    def _suffix_coeff_layout(self):
        return cute.make_layout((self.L, 2), stride=(2, 1))

    def _warp_transition_layout(self):
        return cute.make_layout((max(1, self.scan_threads // 32), 2), stride=(2, 1))

    def _shared_tile_guard_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        # The historical DU kernel needed a non-trivial gap between the staged
        # operand slabs and the later bookkeeping buffers. Tightening that
        # envelope regresses correctness, so keep the guard explicit here
        # instead of leaving it as an unexplained legacy field.
        operand_bytes = (
            (self.bM * self.bK + self.bN * self.bK)
            * self.num_stages
            * (in_dtype.width // 8)
        )
        accumulator_tile_bytes = self.bM * self.bN * (self.acc_dtype.width // 8)
        pad_bytes = max(0, accumulator_tile_bytes - operand_bytes)
        return 2048 + self._align_up(pad_bytes, 4)

    def _shared_tile_guard_layout(self, in_dtype: type[cutlass.Numeric]):
        return cute.make_layout(
            (self._shared_tile_guard_bytes(in_dtype) // 4,), stride=(1,)
        )

    def _shared_tail_pad_layout(self):
        return cute.make_layout((64,), stride=(1,))

    def _smem_capacity_bytes(self, device_index: int | None = None) -> int:
        if torch.cuda.is_available():
            if device_index is None:
                device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(int(device_index))
            capacity = int(getattr(props, "shared_memory_per_block_optin", 0))
            if capacity > 0:
                return capacity
            cc = f"sm_{props.major}{props.minor}"
            return int(utils.get_smem_capacity_in_bytes(cc))
        return int(utils.get_smem_capacity_in_bytes("sm_80"))

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

    def _required_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        in_bytes = in_dtype.width // 8
        warp_count = max(1, self.scan_threads // 32)
        fields = [
            (self.bM * self.bK * self.num_stages * in_bytes, 16),
            (self.bN * self.bK * self.num_stages * in_bytes, 16),
            (self._shared_tile_guard_bytes(in_dtype), 16),
            (warp_count * 2 * 4, 8),
            (warp_count * 2 * 4, 8),
            (self.L * 2 * 4, 4),
            (self.L * 2 * 4, 4),
            (64 * 4, 4),
        ]
        return self._struct_size_bytes(fields)

    def support_info(
        self,
        dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkIncrementBwdDUSupportInfo:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkIncrementBwdDUSupportInfo(0, 1)

        if device_index is None:
            device_key = (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            )
        else:
            device_key = int(device_index)
        cache_key = (
            type(self),
            self.L,
            self.D,
            self.P,
            self.cta_tiler,
            self.num_stages,
            dtype,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkIncrementBwdDUSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._required_smem_bytes(dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(
        self,
        dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> bool:
        return self.support_info(dtype, device_index=device_index).supported

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
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            value_layout = cute.make_layout((1, copy_elems))

            best_tm = None
            best_tn = None
            for tm in range(1, self.num_threads + 1):
                if self.num_threads % tm != 0:
                    continue
                tn = self.num_threads // tm
                tile_k_segment = tn * copy_elems
                if (self.bK % tile_k_segment) != 0:
                    continue
                if best_tm is None or tile_k_segment > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn

            if best_tm is None:
                raise ValueError("Failed to find a legal row-major async-copy tiling.")

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

    def _make_layout_bundle(
        self,
        mDInc: cute.Tensor,
        mB: cute.Tensor,
    ) -> ChunkIncrementBwdDULayoutBundle:
        d_inc_major_mode = utils.LayoutEnum.from_tensor(mDInc)
        b_major_mode = utils.LayoutEnum.from_tensor(mB)

        d_inc_copy_bits = _default_async_copy_bits(
            dtype_width=mDInc.element_type.width,
            major_mode=d_inc_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        b_copy_bits = _default_async_copy_bits(
            dtype_width=mB.element_type.width,
            major_mode=b_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        d_inc_tile_layout = self._make_operand_smem_layout(
            mDInc.element_type,
            d_inc_major_mode,
            d_inc_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        b_tile_layout = self._make_operand_smem_layout(
            mB.element_type,
            b_major_mode,
            b_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        return ChunkIncrementBwdDULayoutBundle(
            d_inc_major_mode=d_inc_major_mode,
            b_major_mode=b_major_mode,
            d_inc_tile_layout=d_inc_tile_layout,
            b_tile_layout=b_tile_layout,
        )

    def _make_copy_bundle(
        self,
        layouts: ChunkIncrementBwdDULayoutBundle,
        in_dtype: type[cutlass.Numeric],
    ) -> ChunkIncrementBwdDUCopyBundle:
        d_inc_copy_bits = _default_async_copy_bits(
            dtype_width=in_dtype.width,
            major_mode=layouts.d_inc_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        b_copy_bits = _default_async_copy_bits(
            dtype_width=in_dtype.width,
            major_mode=layouts.b_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        atom_async_copy_d_inc = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=d_inc_copy_bits,
        )
        atom_async_copy_b = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=b_copy_bits,
        )
        return ChunkIncrementBwdDUCopyBundle(
            gmem_tiled_copy_d_inc=self._make_operand_gmem_tiled_copy(
                atom_async_copy_d_inc,
                in_dtype,
                layouts.d_inc_major_mode,
                d_inc_copy_bits,
                tile_m=self.bM,
            ),
            gmem_tiled_copy_b=self._make_operand_gmem_tiled_copy(
                atom_async_copy_b,
                in_dtype,
                layouts.b_major_mode,
                b_copy_bits,
                tile_m=self.bN,
            ),
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
        layouts: ChunkIncrementBwdDULayoutBundle,
    ):
        suffix_coeff_layout = self._suffix_coeff_layout()
        warp_transition_layout = self._warp_transition_layout()
        shared_tile_guard_layout = self._shared_tile_guard_layout(in_dtype)
        shared_tail_pad_layout = self._shared_tail_pad_layout()

        @cute.struct
        class SharedStorage:
            d_inc_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.d_inc_tile_layout)],
                16,
            ]
            b_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.b_tile_layout)],
                16,
            ]
            shared_tile_guard: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(shared_tile_guard_layout)
                ],
                16,
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
            suffix_coeff_sum: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            suffix_coeff_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            shared_tail_pad: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(shared_tail_pad_layout)
                ],
                4,
            ]

        return SharedStorage

    def _make_kernel_bundle(
        self,
        mDInc: cute.Tensor,
        mB: cute.Tensor,
    ) -> ChunkIncrementBwdDUKernelBundle:
        layouts = self._make_layout_bundle(mDInc, mB)
        copies = self._make_copy_bundle(layouts, mDInc.element_type)
        shared_storage_cls = self._make_shared_storage(mDInc.element_type, layouts)
        return ChunkIncrementBwdDUKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=self._make_tiled_mma(),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
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

    @cute.jit
    def _validate_operands(
        self,
        mDInc: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDU: cute.Tensor,
    ):
        input_dtype_ok = mDInc.element_type == mB.element_type == mDU.element_type
        packed_complex_dtype_ok = (
            mM.element_type
            == mKprev.element_type
            == mKcurr.element_type
            == cutlass.Float32
        )
        batch_head_chunk_dims_match = (
            mDInc.shape[2]
            == mB.shape[2]
            == mM.shape[2]
            == mKprev.shape[2]
            == mKcurr.shape[2]
            == mDU.shape[2]
        )

        if cutlass.const_expr(not input_dtype_ok):
            raise TypeError("DInc/B/DU must share element type.")
        if cutlass.const_expr(
            mDInc.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("DInc/B/DU must be Float16 or BFloat16.")
        if cutlass.const_expr(not packed_complex_dtype_ok):
            raise TypeError("M/Kprev/Kcurr must be Float32.")
        if cutlass.const_expr(mDInc.shape[0] != self.P or mDInc.shape[1] != self.D):
            raise ValueError("DInc must have shape (P, D, BHC) matching kernel config.")
        if cutlass.const_expr(mB.shape[0] != self.L or mB.shape[1] != self.D):
            raise ValueError("B must have shape (L, D, BHC) matching kernel config.")
        if cutlass.const_expr(
            not (
                mM.shape[0] == 2
                and mKprev.shape[0] == 2
                and mKcurr.shape[0] == 2
                and mM.shape[1] == self.L
                and mKprev.shape[1] == self.L
                and mKcurr.shape[1] == self.L
            )
        ):
            raise ValueError("M/Kprev/Kcurr must have shape (2, L, BHC).")
        if cutlass.const_expr(mDU.shape[0] != self.P or mDU.shape[1] != self.L):
            raise ValueError("DU must have shape (P, L, BHC) matching kernel config.")
        if cutlass.const_expr(mB.shape[1] % 2 != 0):
            raise ValueError("B D dimension must be even because D stores pairs.")
        if cutlass.const_expr(not batch_head_chunk_dims_match):
            raise ValueError("All operands must share the BHC dimension.")

    def _launch_kernel(
        self,
        mDInc: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDU: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mDInc, mB)
        grid_dim = cute.ceil_div(mDU.shape, (self.bM, self.bN, 1))
        launch_kwargs = {
            "grid": (
                cute.size(grid_dim[0]),
                cute.size(grid_dim[1]),
                cute.size(mDU.shape[2]),
            ),
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_bytes,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

        self.kernel(
            mDInc,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDU,
            bundle.layouts.d_inc_major_mode,
            bundle.layouts.b_major_mode,
            bundle.layouts.d_inc_tile_layout,
            bundle.layouts.b_tile_layout,
            bundle.copies.gmem_tiled_copy_d_inc,
            bundle.copies.gmem_tiled_copy_b,
            bundle.tiled_mma,
            bundle.shared_storage_cls,
        ).launch(**launch_kwargs)

    @cute.jit
    def _validate_and_launch(
        self,
        mDInc: cute.Tensor,  # (P, D, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDU: cute.Tensor,  # (P, L, BHC)
        stream: cuda.CUstream | None = None,
    ):
        self._validate_operands(mDInc, mB, mM, mKprev, mKcurr, mDU)
        self._launch_kernel(
            mDInc,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDU,
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        mDInc: cute.Tensor,  # (P, D, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDU: cute.Tensor,  # (P, L, BHC)
    ):
        self._validate_and_launch(mDInc, mB, mM, mKprev, mKcurr, mDU)

    @cute.jit
    def call_on_stream(
        self,
        mDInc: cute.Tensor,  # (P, D, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDU: cute.Tensor,  # (P, L, BHC)
        stream: cuda.CUstream,
    ):
        self._validate_and_launch(
            mDInc,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDU,
            stream=stream,
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
        d_inc_major_mode: cutlass.Constexpr,
        b_major_mode: cutlass.Constexpr,
        d_inc_tile_layout: cute.ComposedLayout,
        b_tile_layout: cute.ComposedLayout,
        gmem_tiled_copy_d_inc: cute.TiledCopy,
        gmem_tiled_copy_b: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        tiler_coord = (bidx, bidy, None)

        # CTA tile setup.
        g_d_inc = cute.local_tile(
            mDInc[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        g_b = cute.local_tile(
            mB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        g_d_inc = cute.make_tensor(g_d_inc.iterator.align(16), g_d_inc.layout)
        g_b = cute.make_tensor(g_b.iterator.align(16), g_b.layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_d_inc_tile = storage.d_inc_tile.get_tensor(d_inc_tile_layout)
        s_b_tile = storage.b_tile.get_tensor(b_tile_layout)
        suffix_coeff_layout = self._suffix_coeff_layout()
        warp_transition_layout = self._warp_transition_layout()
        s_suffix_coeff_prev = storage.suffix_coeff_prev.get_tensor(suffix_coeff_layout)
        s_suffix_coeff_sum = storage.suffix_coeff_sum.get_tensor(suffix_coeff_layout)
        s_warp_transition_total = storage.warp_transition_total.get_tensor(
            warp_transition_layout
        )
        s_warp_transition_offset = storage.warp_transition_offset.get_tensor(
            warp_transition_layout
        )

        copy_slice_d_inc = gmem_tiled_copy_d_inc.get_slice(tidx)
        copy_slice_b = gmem_tiled_copy_b.get_slice(tidx)

        t_d_inc_gmem = copy_slice_d_inc.partition_S(g_d_inc)
        t_b_gmem = copy_slice_b.partition_S(g_b)
        t_d_inc_copy_dst = copy_slice_d_inc.partition_D(s_d_inc_tile)
        t_b_copy_dst = copy_slice_b.partition_D(s_b_tile)

        mma_slice = tiled_mma.get_slice(tidx)
        t_d_inc_mma_smem = mma_slice.partition_A(s_d_inc_tile)
        t_b_mma_smem = mma_slice.partition_B(s_b_tile)
        t_du_mma_gmem = mma_slice.partition_C(
            cute.local_tile(
                mDU[None, None, bidz],
                tiler=self.cta_tiler,
                coord=tiler_coord,
                proj=(1, 1, None),
            )
        )

        r_d_inc_mma = tiled_mma.make_fragment_A(t_d_inc_mma_smem[None, None, None, 0])
        r_b_mma = tiled_mma.make_fragment_B(t_b_mma_smem[None, None, None, 0])
        r_du_accum = tiled_mma.make_fragment_C(t_du_mma_gmem)
        r_du_accum.fill(0.0)

        ldmatrix_copy_atom_d_inc = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                d_inc_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mDInc.element_type,
        )
        ldmatrix_copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mB.element_type,
        )
        s2r_tiled_copy_d_inc = cute.make_tiled_copy_A(
            ldmatrix_copy_atom_d_inc, tiled_mma
        )
        s2r_tiled_copy_b = cute.make_tiled_copy_B(ldmatrix_copy_atom_b, tiled_mma)
        ldmatrix_slice_d_inc = s2r_tiled_copy_d_inc.get_slice(tidx)
        ldmatrix_slice_b = s2r_tiled_copy_b.get_slice(tidx)
        t_d_inc_ldmatrix_smem = ldmatrix_slice_d_inc.partition_S(s_d_inc_tile)
        r_d_inc_ldmatrix = ldmatrix_slice_d_inc.retile(r_d_inc_mma)
        t_b_ldmatrix_smem = ldmatrix_slice_b.partition_S(s_b_tile)
        r_b_ldmatrix = ldmatrix_slice_b.retile(r_b_mma)

        d_inc_coord_identity = cute.make_identity_tensor(mDInc.layout.shape)
        b_coord_identity = cute.make_identity_tensor(mB.layout.shape)
        d_inc_coord_tile = cute.local_tile(
            d_inc_coord_identity[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        b_coord_tile = cute.local_tile(
            b_coord_identity[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        t_d_inc_coord = copy_slice_d_inc.partition_S(d_inc_coord_tile)
        t_b_coord = copy_slice_b.partition_S(b_coord_tile)

        d_inc_row_predicate = self._make_copy_tile_row_predicate(
            t_d_inc_copy_dst,
            t_d_inc_coord,
            mDInc.shape[0],
        )
        b_row_predicate = self._make_copy_tile_row_predicate(
            t_b_copy_dst,
            t_b_coord,
            mB.shape[0],
        )

        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        reverse_time_index = cutlass.Int32(self.L - 1) - tidx

        transition_step_re = cutlass.Float32(1.0)
        transition_step_im = cutlass.Float32(0.0)
        prev_tap_re = cutlass.Float32(0.0)
        prev_tap_im = cutlass.Float32(0.0)
        curr_tap_re = cutlass.Float32(0.0)
        curr_tap_im = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            transition_step_re = cutlass.Float32(
                mM[0, reverse_time_index, bidz].to(cutlass.Float32)
            )
            transition_step_im = cutlass.Float32(
                mM[1, reverse_time_index, bidz].to(cutlass.Float32)
            )
            prev_tap_re = cutlass.Float32(
                mKprev[0, reverse_time_index, bidz].to(cutlass.Float32)
            )
            prev_tap_im = cutlass.Float32(
                mKprev[1, reverse_time_index, bidz].to(cutlass.Float32)
            )
            curr_tap_re = cutlass.Float32(
                mKcurr[0, reverse_time_index, bidz].to(cutlass.Float32)
            )
            curr_tap_im = cutlass.Float32(
                mKcurr[1, reverse_time_index, bidz].to(cutlass.Float32)
            )

        # Reverse-time suffix scan over the packed-complex transition stream.
        scan_threads = cutlass.Int32(self.scan_threads)
        scan_warp_count = cutlass.Int32(max(1, self.scan_threads // 32))
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
            # A single-lane shared-memory inter-warp scan keeps the suffix
            # product algebra explicit while avoiding another layer of
            # warp-shuffle bookkeeping.
            if warp_idx == cutlass.Int32(0) and lane_idx == cutlass.Int32(0):
                running_transition_re = cutlass.Float32(1.0)
                running_transition_im = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(self.scan_threads // 32):
                    s_warp_transition_offset[w, 0] = running_transition_re
                    s_warp_transition_offset[w, 1] = running_transition_im

                    total_transition_re = s_warp_transition_total[w, 0]
                    total_transition_im = s_warp_transition_total[w, 1]
                    next_running_transition_re = (
                        running_transition_re * total_transition_re
                        - running_transition_im * total_transition_im
                    )
                    next_running_transition_im = (
                        running_transition_re * total_transition_im
                        + running_transition_im * total_transition_re
                    )
                    running_transition_re = next_running_transition_re
                    running_transition_im = next_running_transition_im

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

        # Convert inclusive suffix products into the per-time-step coefficients
        # used to rotate the staged B slab.
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
            s_suffix_coeff_sum[reverse_time_index, 0] = curr_coeff_re
            s_suffix_coeff_sum[reverse_time_index, 1] = curr_coeff_im

        cute.arch.sync_threads()

        # Each DU rotation uses the current-step tap plus the next-step previous
        # tap, so fold that dependency once up front instead of redoing it in
        # the tensor-core mainloop.
        if tidx < cutlass.Int32(self.L):
            if reverse_time_index < cutlass.Int32(self.L - 1):
                s_suffix_coeff_sum[reverse_time_index, 0] = (
                    s_suffix_coeff_sum[reverse_time_index, 0]
                    + s_suffix_coeff_prev[reverse_time_index + 1, 0]
                )
                s_suffix_coeff_sum[reverse_time_index, 1] = (
                    s_suffix_coeff_sum[reverse_time_index, 1]
                    + s_suffix_coeff_prev[reverse_time_index + 1, 1]
                )

        cute.arch.sync_threads()

        # Pipelined tensor-core mainloop over the packed-complex D tiles.
        t_d_inc_copy_dst.fill(0)
        t_b_copy_dst.fill(0)
        cute.arch.sync_threads()

        k_tile_count = self.D // self.bK
        num_smem_stages = cute.size(t_d_inc_copy_dst, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for kk in cutlass.range_constexpr(t_d_inc_copy_dst.shape[2]):
            cute.copy(
                gmem_tiled_copy_d_inc,
                t_d_inc_gmem[None, None, kk, k_tile_index],
                t_d_inc_copy_dst[None, None, kk, 0],
                pred=d_inc_row_predicate[None, None, kk],
            )
        for kk in cutlass.range_constexpr(t_b_copy_dst.shape[2]):
            cute.copy(
                gmem_tiled_copy_b,
                t_b_gmem[None, None, kk, k_tile_index],
                t_b_copy_dst[None, None, kk, 0],
                pred=b_row_predicate[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            cute.copy(
                gmem_tiled_copy_d_inc,
                t_d_inc_gmem[None, None, None, k_tile_index],
                t_d_inc_copy_dst[None, None, None, k_tile],
                pred=d_inc_row_predicate,
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
        mma_k_block_count = cute.size(r_d_inc_mma, mode=[2])

        complex_pair_count = self.bK // 2
        stage_transform_count = self.bN * complex_pair_count
        stage_transform_iters = (
            stage_transform_count + self.num_threads - 1
        ) // self.num_threads

        for kt in range(k_tile_count):
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            # Rotate the staged B slab by the precomputed suffix coefficients.
            for it in cutlass.range_constexpr(stage_transform_iters):
                idx = tidx + (it * self.num_threads)
                if cute.elem_less(idx, stage_transform_count):
                    time_step = idx // complex_pair_count
                    complex_pair = idx - time_step * complex_pair_count
                    d_pair_start = complex_pair * 2

                    b_re = s_b_tile[time_step, d_pair_start + 0, smem_pipe_read].to(
                        cutlass.Float32
                    )
                    b_im = s_b_tile[time_step, d_pair_start + 1, smem_pipe_read].to(
                        cutlass.Float32
                    )
                    coeff_re = s_suffix_coeff_sum[time_step, 0]
                    coeff_im = s_suffix_coeff_sum[time_step, 1]

                    rotated_re = coeff_re * b_re - coeff_im * b_im
                    rotated_im = coeff_re * b_im + coeff_im * b_re

                    s_b_tile[time_step, d_pair_start + 0, smem_pipe_read] = (
                        rotated_re.to(mB.element_type)
                    )
                    s_b_tile[time_step, d_pair_start + 1, smem_pipe_read] = (
                        rotated_im.to(mB.element_type)
                    )

            cute.arch.sync_threads()

            next_tile = kt + (num_smem_stages - 1)
            if next_tile < k_tile_count:
                cute.copy(
                    gmem_tiled_copy_d_inc,
                    t_d_inc_gmem[None, None, None, k_tile_index],
                    t_d_inc_copy_dst[None, None, None, smem_pipe_write],
                    pred=d_inc_row_predicate,
                )
                cute.copy(
                    gmem_tiled_copy_b,
                    t_b_gmem[None, None, None, k_tile_index],
                    t_b_copy_dst[None, None, None, smem_pipe_write],
                    pred=b_row_predicate,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            t_d_inc_ldmatrix_stage = t_d_inc_ldmatrix_smem[
                None, None, None, smem_pipe_read
            ]
            t_b_ldmatrix_stage = t_b_ldmatrix_smem[None, None, None, smem_pipe_read]
            cute.copy(
                s2r_tiled_copy_d_inc,
                t_d_inc_ldmatrix_stage[None, None, 0],
                r_d_inc_ldmatrix[None, None, 0],
            )
            cute.copy(
                s2r_tiled_copy_b,
                t_b_ldmatrix_stage[None, None, 0],
                r_b_ldmatrix[None, None, 0],
            )
            for kb in cutlass.range(mma_k_block_count, unroll_full=True):
                kb_next = (kb + 1) % mma_k_block_count
                cute.copy(
                    s2r_tiled_copy_d_inc,
                    t_d_inc_ldmatrix_stage[None, None, kb_next],
                    r_d_inc_ldmatrix[None, None, kb_next],
                )
                cute.copy(
                    s2r_tiled_copy_b,
                    t_b_ldmatrix_stage[None, None, kb_next],
                    r_b_ldmatrix[None, None, kb_next],
                )
                cute.gemm(
                    tiled_mma,
                    r_du_accum,
                    r_d_inc_mma[None, None, kb],
                    r_b_mma[None, None, kb],
                    r_du_accum,
                )

            smem_pipe_write = smem_pipe_read
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == num_smem_stages:
                smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Direct-store epilogue.
        ceil_tile_rows, ceil_tile_cols, _ = cute.ceil_div(
            mDU.shape, (self.bM, self.bN, 1)
        )
        du_coord_identity = cute.make_identity_tensor(
            (
                cute.size(ceil_tile_rows) * self.bM,
                cute.size(ceil_tile_cols) * self.bN,
                1,
            )
        )
        du_coord_tile = cute.local_tile(
            du_coord_identity[None, None, bidz],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        t_du_coord = mma_slice.partition_C(du_coord_tile)
        t_du_coord_mn = self._make_accumulator_mn_view(t_du_coord)
        r_du_accum_mn = self._make_accumulator_mn_view(r_du_accum)

        for r in cutlass.range_constexpr(cute.size(r_du_accum_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(r_du_accum_mn.shape[1])):
                row_idx = cutlass.Int32(t_du_coord_mn[r, c][0])
                col_idx = cutlass.Int32(t_du_coord_mn[r, c][1])
                if cute.elem_less(row_idx, mDU.shape[0]) and cute.elem_less(
                    col_idx, mDU.shape[1]
                ):
                    mDU[row_idx, col_idx, bidz] = cutlass.Float32(
                        r_du_accum_mn[r, c]
                    ).to(mDU.element_type)
