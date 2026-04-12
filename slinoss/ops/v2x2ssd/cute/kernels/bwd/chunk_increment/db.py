"""CuTe backward kernel for the ``v2x2ssd`` chunk-increment ``dB`` stage.

``ChunkIncrementBwdDBAmpere`` computes one chunk-local ``dB_main`` tile and the
per-``D``-tile partial reductions consumed by the later parameter scan stage.
The kernel runs a reverse-time suffix scan over ``M``, builds the two complex
coefficient streams induced by ``Kprev``/``Kcurr``, accumulates
``dB_sum = U @ dInc^T`` with one tensor-core GEMM, rotates that accumulator
through the packed-complex ``B`` stream to produce ``dB``, and reduces
``conj(B) * dB_sum`` across the current ``D`` tile into ``DMsumPart``.

Tensor contracts:

- ``U``: ``(L, P, BHC)`` fp16/bf16 chunk-local value rows
- ``B``: ``(L, D, BHC)`` fp16/bf16 packed-complex key rows
- ``M``: ``(2, L, BHC)`` fp32 packed-complex transitions
- ``Kprev``: ``(2, L, BHC)`` fp32 packed-complex previous-pass taps
- ``Kcurr``: ``(2, L, BHC)`` fp32 packed-complex current-pass taps
- ``DIncDP``: ``(D, P, BHC)`` fp16/bf16 transpose view of ``d_inc``
- ``DB``: ``(L, D, BHC)`` fp16/bf16 chunk-local ``dB`` output
- ``DMsumPart``: ``(2, L, n_d_tiles, BHC)`` fp32 packed-complex partials

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
class ChunkIncrementBwdDBLayoutBundle:
    u_major_mode: object
    d_inc_major_mode: object
    u_tile_layout: object
    d_inc_tile_layout: object
    db_tile_layout: object


@dataclass(frozen=True)
class ChunkIncrementBwdDBCopyBundle:
    gmem_tiled_copy_u: object
    gmem_tiled_copy_d_inc: object
    ldmatrix_tiled_copy_u: object
    ldmatrix_tiled_copy_d_inc: object


@dataclass(frozen=True)
class ChunkIncrementBwdDBKernelBundle:
    layouts: ChunkIncrementBwdDBLayoutBundle
    copies: ChunkIncrementBwdDBCopyBundle
    tiled_mma: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkIncrementBwdDBSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkIncrementBwdDBAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dB`` stage."""

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkIncrementBwdDBSupportInfo]
    ] = {}

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
            raise ValueError("bN must be divisible by 2 because D stores pairs.")
        if (self.bN // 2) > 32:
            raise ValueError("bN/2 must fit within one warp for the fused epilogue.")
        if self.bK % 16 != 0:
            raise ValueError("bK must be divisible by 16 for tensor cores.")
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2.")
        if self.P % self.bK != 0:
            raise ValueError("P must be divisible by bK for this kernel.")
        if (self.P // self.bK) < (self.num_stages - 1):
            raise ValueError("P/bK must be >= (num_stages - 1) for the cp.async pipe.")

        self.mma_inst_shape = (16, 8, 16)
        mma_m, mma_n, mma_k = self.mma_inst_shape
        atom_m, atom_n, atom_k = self.atom_layout_mnk
        self.num_threads = atom_m * atom_n * atom_k * 32
        if atom_k != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bM % (atom_m * mma_m) != 0:
            raise ValueError("bM must be divisible by atomM * mmaM.")
        if self.bN % (atom_n * mma_n * 2) != 0:
            raise ValueError("bN must be divisible by atomN * mmaN * 2.")
        if self.bK % mma_k != 0:
            raise ValueError("bK must be divisible by mmaK.")

        self.scan_threads = _next_pow2(self.L)
        if self.scan_threads > self.num_threads:
            raise ValueError(
                "chunk_size too large for scan_threads with this CTA thread count."
            )

    def _suffix_coeff_layout(self):
        return cute.make_layout((self.L, 2), stride=(2, 1))

    def _warp_transition_layout(self):
        return cute.make_layout((32,), stride=(1,))

    def _db_tile_layout(self):
        major_mode_size = self.bN
        if major_mode_size >= 64:
            if major_mode_size % 64 == 0:
                major_mode_size = 64
            elif major_mode_size % 32 == 0:
                major_mode_size = 32
            else:
                major_mode_size = 64

        swizzle_bits = int(math.log2(major_mode_size * self.c_dtype.width // 128))
        swizzle_bits = min(swizzle_bits, 3)
        layout_atom_outer = cute.make_layout(
            (8, major_mode_size), stride=(major_mode_size, 1)
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        return cute.tile_to_shape(layout_atom, (self.bM, self.bN), (0, 1))

    def _output_alias_guard_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        operand_bytes = (
            (self.bM * self.bK + self.bN * self.bK)
            * self.num_stages
            * (in_dtype.width // 8)
        )
        db_tile_bytes = self.bM * self.bN * (self.c_dtype.width // 8)
        pad_bytes = max(0, db_tile_bytes - operand_bytes)
        return 2048 + self._align_up(pad_bytes, 4)

    def _output_alias_guard_layout(self, in_dtype: type[cutlass.Numeric]):
        return cute.make_layout(
            (self._output_alias_guard_bytes(in_dtype) // 4,), stride=(1,)
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
        # Keep this accounting aligned with `_make_shared_storage`.
        fields = [
            (
                self.bM * self.bK * self.num_stages * (in_dtype.width // 8),
                16,
            ),
            (
                self.bN * self.bK * self.num_stages * (in_dtype.width // 8),
                16,
            ),
            (self._output_alias_guard_bytes(in_dtype), 16),
            (self.L * 2 * 4, 4),
            (self.L * 2 * 4, 4),
            (32 * 4, 4),
            (32 * 4, 4),
            (32 * 4, 4),
            (32 * 4, 4),
            (64 * 4, 4),
        ]
        return self._struct_size_bytes(fields)

    def support_info(
        self,
        dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkIncrementBwdDBSupportInfo:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkIncrementBwdDBSupportInfo(0, 1)

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

        info = ChunkIncrementBwdDBSupportInfo(
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
        mDIncDP: cute.Tensor,
    ) -> ChunkIncrementBwdDBLayoutBundle:
        u_major_mode = utils.LayoutEnum.from_tensor(mU)
        d_inc_major_mode = utils.LayoutEnum.from_tensor(mDIncDP)
        u_copy_bits = _default_async_copy_bits(
            dtype_width=mU.element_type.width,
            major_mode=u_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        d_inc_copy_bits = _default_async_copy_bits(
            dtype_width=mDIncDP.element_type.width,
            major_mode=d_inc_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        u_tile_layout = self._make_operand_smem_layout(
            mU.element_type,
            u_major_mode,
            u_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        d_inc_tile_layout = self._make_operand_smem_layout(
            mDIncDP.element_type,
            d_inc_major_mode,
            d_inc_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        return ChunkIncrementBwdDBLayoutBundle(
            u_major_mode=u_major_mode,
            d_inc_major_mode=d_inc_major_mode,
            u_tile_layout=u_tile_layout,
            d_inc_tile_layout=d_inc_tile_layout,
            db_tile_layout=self._db_tile_layout(),
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

    def _make_copy_bundle(
        self,
        layouts: ChunkIncrementBwdDBLayoutBundle,
        in_u_dtype: type[cutlass.Numeric],
        in_d_inc_dtype: type[cutlass.Numeric],
        tiled_mma: cute.TiledMma,
    ) -> ChunkIncrementBwdDBCopyBundle:
        u_copy_bits = _default_async_copy_bits(
            dtype_width=in_u_dtype.width,
            major_mode=layouts.u_major_mode,
            tile_m=self.bM,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        d_inc_copy_bits = _default_async_copy_bits(
            dtype_width=in_d_inc_dtype.width,
            major_mode=layouts.d_inc_major_mode,
            tile_m=self.bN,
            tile_k=self.bK,
            num_threads=self.num_threads,
        )
        atom_async_copy_u = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_u_dtype,
            num_bits_per_copy=u_copy_bits,
        )
        atom_async_copy_d_inc = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_d_inc_dtype,
            num_bits_per_copy=d_inc_copy_bits,
        )
        gmem_tiled_copy_u = self._make_operand_gmem_tiled_copy(
            atom_async_copy_u,
            in_u_dtype,
            layouts.u_major_mode,
            u_copy_bits,
            tile_m=self.bM,
        )
        gmem_tiled_copy_d_inc = self._make_operand_gmem_tiled_copy(
            atom_async_copy_d_inc,
            in_d_inc_dtype,
            layouts.d_inc_major_mode,
            d_inc_copy_bits,
            tile_m=self.bN,
        )
        ldmatrix_copy_atom_u = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.u_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            in_u_dtype,
        )
        ldmatrix_copy_atom_d_inc = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                layouts.d_inc_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            in_d_inc_dtype,
        )
        return ChunkIncrementBwdDBCopyBundle(
            gmem_tiled_copy_u=gmem_tiled_copy_u,
            gmem_tiled_copy_d_inc=gmem_tiled_copy_d_inc,
            ldmatrix_tiled_copy_u=cute.make_tiled_copy_A(
                ldmatrix_copy_atom_u, tiled_mma
            ),
            ldmatrix_tiled_copy_d_inc=cute.make_tiled_copy_B(
                ldmatrix_copy_atom_d_inc, tiled_mma
            ),
        )

    def _make_shared_storage(
        self,
        in_u_dtype: type[cutlass.Numeric],
        in_d_inc_dtype: type[cutlass.Numeric],
        u_tile_layout: cute.ComposedLayout,
        d_inc_tile_layout: cute.ComposedLayout,
    ):
        suffix_coeff_layout = self._suffix_coeff_layout()
        warp_transition_layout = self._warp_transition_layout()
        output_alias_guard_layout = self._output_alias_guard_layout(in_u_dtype)
        shared_tail_pad_layout = self._shared_tail_pad_layout()

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "u_tile": cute.struct.Align[
                cute.struct.MemRange[in_u_dtype, cute.cosize(u_tile_layout)], 16
            ],
            "d_inc_tile": cute.struct.Align[
                cute.struct.MemRange[in_d_inc_dtype, cute.cosize(d_inc_tile_layout)],
                16,
            ],
            # Preserve the historical accumulator aliasing envelope instead of
            # paying for a dedicated shared-memory db tile.
            "output_alias_guard": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(output_alias_guard_layout)
                ],
                16,
            ],
            "suffix_coeff_sum": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ],
            "suffix_coeff_prev": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ],
            "warp_transition_re_total": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ],
            "warp_transition_im_total": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ],
            "warp_transition_re_offset": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ],
            "warp_transition_im_offset": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ],
            "shared_tail_pad": cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(shared_tail_pad_layout)
                ],
                4,
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self,
        mU: cute.Tensor,
        mDIncDP: cute.Tensor,
    ) -> ChunkIncrementBwdDBKernelBundle:
        layouts = self._make_layout_bundle(mU, mDIncDP)
        tiled_mma = self._make_tiled_mma()
        copies = self._make_copy_bundle(
            layouts,
            mU.element_type,
            mDIncDP.element_type,
            tiled_mma,
        )
        shared_storage_cls = self._make_shared_storage(
            mU.element_type,
            mDIncDP.element_type,
            layouts.u_tile_layout,
            layouts.d_inc_tile_layout,
        )
        return ChunkIncrementBwdDBKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=tiled_mma,
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
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDIncDP: cute.Tensor,
        mDB: cute.Tensor,
        mDMsumPart: cute.Tensor,
    ):
        value_stream_dtype_ok = (
            mU.element_type
            == mB.element_type
            == mDIncDP.element_type
            == mDB.element_type
        )
        coeff_dtype_ok = (
            mM.element_type
            == mKprev.element_type
            == mKcurr.element_type
            == mDMsumPart.element_type
            == cutlass.Float32
        )
        chunk_dims_match = (
            mU.shape[0]
            == mB.shape[0]
            == mM.shape[1]
            == mKprev.shape[1]
            == mKcurr.shape[1]
            == mDB.shape[0]
        )
        batch_head_chunk_dims_match = (
            mU.shape[2]
            == mB.shape[2]
            == mM.shape[2]
            == mKprev.shape[2]
            == mKcurr.shape[2]
            == mDIncDP.shape[2]
            == mDB.shape[2]
            == mDMsumPart.shape[3]
        )

        if cutlass.const_expr(not value_stream_dtype_ok):
            raise TypeError("U/B/DIncDP/DB must share element type.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/DIncDP/DB must be Float16 or BFloat16.")
        if cutlass.const_expr(not coeff_dtype_ok):
            raise TypeError("M/Kprev/Kcurr/DMsumPart must be Float32.")
        if cutlass.const_expr(not chunk_dims_match):
            raise ValueError(
                "U/B/M/Kprev/Kcurr/DB must share the chunk time dimension."
            )
        if cutlass.const_expr(not batch_head_chunk_dims_match):
            raise ValueError("All operands must share the batch-head-chunk dimension.")
        if cutlass.const_expr(mB.shape[1] % 2 != 0):
            raise ValueError("B/DB D dimension must be even because D stores pairs.")
        if cutlass.const_expr(mDIncDP.shape[0] != mB.shape[1]):
            raise ValueError("DIncDP rows must match the D dimension of B.")
        if cutlass.const_expr(mDIncDP.shape[1] != mU.shape[1]):
            raise ValueError("DIncDP columns must match the P dimension of U.")
        if cutlass.const_expr(mDB.shape[1] != mB.shape[1]):
            raise ValueError("DB columns must match the D dimension of B.")
        if cutlass.const_expr(
            not (mM.shape[0] == mKprev.shape[0] == mKcurr.shape[0] == 2)
        ):
            raise ValueError(
                "M/Kprev/Kcurr must have leading packed-complex dimension 2."
            )
        if cutlass.const_expr(mDMsumPart.shape[0] != 2):
            raise ValueError("DMsumPart must have leading packed-complex dimension 2.")
        if cutlass.const_expr(mDMsumPart.shape[1] != mU.shape[0]):
            raise ValueError("DMsumPart time dimension must match U rows.")

    def _launch_kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDIncDP: cute.Tensor,
        mDB: cute.Tensor,
        mDMsumPart: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mU, mDIncDP)
        grid_dim = cute.ceil_div(mDB.shape, (self.bM, self.bN, 1))
        launch_kwargs = {
            "grid": (
                cute.size(grid_dim[0]),
                cute.size(grid_dim[1]),
                cute.size(mDB.shape[2]),
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
            mDIncDP,
            mDB,
            mDMsumPart,
            bundle.layouts.u_tile_layout,
            bundle.layouts.d_inc_tile_layout,
            bundle.layouts.db_tile_layout,
            bundle.copies.gmem_tiled_copy_u,
            bundle.copies.gmem_tiled_copy_d_inc,
            bundle.copies.ldmatrix_tiled_copy_u,
            bundle.copies.ldmatrix_tiled_copy_d_inc,
            bundle.tiled_mma,
        ).launch(**launch_kwargs)

    @cute.jit
    def _validate_and_launch(
        self,
        mU: cute.Tensor,  # (L, P, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDIncDP: cute.Tensor,  # (D, P, BHC)
        mDB: cute.Tensor,  # (L, D, BHC)
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC)
        stream: cuda.CUstream | None = None,
    ):
        self._validate_operands(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)
        self._launch_kernel(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDIncDP,
            mDB,
            mDMsumPart,
            stream=stream,
        )

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,  # (L, P, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDIncDP: cute.Tensor,  # (D, P, BHC)
        mDB: cute.Tensor,  # (L, D, BHC)
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC)
    ):
        self._validate_and_launch(mU, mB, mM, mKprev, mKcurr, mDIncDP, mDB, mDMsumPart)

    @cute.jit
    def call_on_stream(
        self,
        mU: cute.Tensor,  # (L, P, BHC)
        mB: cute.Tensor,  # (L, D, BHC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDIncDP: cute.Tensor,  # (D, P, BHC)
        mDB: cute.Tensor,  # (L, D, BHC)
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC)
        stream: cuda.CUstream,
    ):
        self._validate_and_launch(
            mU,
            mB,
            mM,
            mKprev,
            mKcurr,
            mDIncDP,
            mDB,
            mDMsumPart,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDIncDP: cute.Tensor,
        mDB: cute.Tensor,
        mDMsumPart: cute.Tensor,
        u_tile_layout: cute.ComposedLayout,
        d_inc_tile_layout: cute.ComposedLayout,
        db_tile_layout: cute.ComposedLayout,
        gmem_tiled_copy_u: cute.TiledCopy,
        gmem_tiled_copy_d_inc: cute.TiledCopy,
        ldmatrix_tiled_copy_u: cute.TiledCopy,
        ldmatrix_tiled_copy_d_inc: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, block_col_idx, batch_head_chunk_idx = cute.arch.block_idx()

        tile_col_start = block_col_idx * self.bN
        tiler_coord = (cutlass.Int32(0), block_col_idx, None)

        # CTA tile setup.
        g_u = cute.local_tile(
            mU[None, None, batch_head_chunk_idx],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        g_d_inc = cute.local_tile(
            mDIncDP[None, None, batch_head_chunk_idx],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        g_db = cute.local_tile(
            mDB[None, None, batch_head_chunk_idx],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        g_u = cute.make_tensor(g_u.iterator.align(16), g_u.layout)
        g_d_inc = cute.make_tensor(g_d_inc.iterator.align(16), g_d_inc.layout)

        smem = cutlass.utils.SmemAllocator()
        shared_storage_cls = self._make_shared_storage(
            mU.element_type,
            mDIncDP.element_type,
            u_tile_layout,
            d_inc_tile_layout,
        )
        storage = smem.allocate(shared_storage_cls)
        suffix_coeff_layout = self._suffix_coeff_layout()
        warp_transition_layout = self._warp_transition_layout()
        s_u_tile = storage.u_tile.get_tensor(u_tile_layout)
        s_d_inc_tile = storage.d_inc_tile.get_tensor(d_inc_tile_layout)
        s_db_tile = cute.make_tensor(
            cute.recast_ptr(s_u_tile.iterator, dtype=cutlass.Float32),
            db_tile_layout,
        )
        s_suffix_coeff_sum = storage.suffix_coeff_sum.get_tensor(suffix_coeff_layout)
        s_suffix_coeff_prev = storage.suffix_coeff_prev.get_tensor(suffix_coeff_layout)
        s_warp_transition_re_total = storage.warp_transition_re_total.get_tensor(
            warp_transition_layout
        )
        s_warp_transition_im_total = storage.warp_transition_im_total.get_tensor(
            warp_transition_layout
        )
        s_warp_transition_re_offset = storage.warp_transition_re_offset.get_tensor(
            warp_transition_layout
        )
        s_warp_transition_im_offset = storage.warp_transition_im_offset.get_tensor(
            warp_transition_layout
        )

        copy_slice_u = gmem_tiled_copy_u.get_slice(tidx)
        copy_slice_d_inc = gmem_tiled_copy_d_inc.get_slice(tidx)
        t_u_gmem = copy_slice_u.partition_S(g_u)
        t_d_inc_gmem = copy_slice_d_inc.partition_S(g_d_inc)
        t_u_copy_dst = copy_slice_u.partition_D(s_u_tile)
        t_d_inc_copy_dst = copy_slice_d_inc.partition_D(s_d_inc_tile)

        mma_slice = tiled_mma.get_slice(tidx)
        t_u_mma_smem = mma_slice.partition_A(s_u_tile)
        t_d_inc_mma_smem = mma_slice.partition_B(s_d_inc_tile)
        t_db_mma_smem = mma_slice.partition_C(s_db_tile)
        t_db_mma_gmem = mma_slice.partition_C(g_db)

        r_u_mma = tiled_mma.make_fragment_A(t_u_mma_smem[None, None, None, 0])
        r_d_inc_mma = tiled_mma.make_fragment_B(t_d_inc_mma_smem[None, None, None, 0])
        r_db_accum = tiled_mma.make_fragment_C(t_db_mma_gmem)
        r_db_accum.fill(0.0)

        ldmatrix_slice_u = ldmatrix_tiled_copy_u.get_slice(tidx)
        ldmatrix_slice_d_inc = ldmatrix_tiled_copy_d_inc.get_slice(tidx)
        t_u_ldmatrix_smem = ldmatrix_slice_u.partition_S(s_u_tile)
        r_u_ldmatrix = ldmatrix_slice_u.retile(r_u_mma)
        t_d_inc_ldmatrix_smem = ldmatrix_slice_d_inc.partition_S(s_d_inc_tile)
        r_d_inc_ldmatrix = ldmatrix_slice_d_inc.retile(r_d_inc_mma)

        u_coord_identity = cute.make_identity_tensor(mU.layout.shape)
        d_inc_coord_identity = cute.make_identity_tensor(mDIncDP.layout.shape)
        u_coord_tile = cute.local_tile(
            u_coord_identity[None, None, batch_head_chunk_idx],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        d_inc_coord_tile = cute.local_tile(
            d_inc_coord_identity[None, None, batch_head_chunk_idx],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        t_u_coord = copy_slice_u.partition_S(u_coord_tile)
        t_d_inc_coord = copy_slice_d_inc.partition_S(d_inc_coord_tile)

        u_row_predicate = self._make_copy_tile_row_predicate(
            t_u_copy_dst,
            t_u_coord,
            mU.shape[0],
        )
        d_inc_row_predicate = self._make_copy_tile_row_predicate(
            t_d_inc_copy_dst,
            t_d_inc_coord,
            mDIncDP.shape[0],
        )

        t_u_copy_dst.fill(0)
        t_d_inc_copy_dst.fill(0)
        cute.arch.sync_threads()

        k_tile_count = self.P // self.bK
        num_smem_stages = cute.size(t_u_copy_dst, mode=[3])
        k_tile_index = cutlass.Int32(0)

        for kk in cutlass.range_constexpr(u_row_predicate.shape[2]):
            cute.copy(
                gmem_tiled_copy_u,
                t_u_gmem[None, None, kk, k_tile_index],
                t_u_copy_dst[None, None, kk, 0],
                pred=u_row_predicate[None, None, kk],
            )
        for kk in cutlass.range_constexpr(d_inc_row_predicate.shape[2]):
            cute.copy(
                gmem_tiled_copy_d_inc,
                t_d_inc_gmem[None, None, kk, k_tile_index],
                t_d_inc_copy_dst[None, None, kk, 0],
                pred=d_inc_row_predicate[None, None, kk],
            )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        reverse_time_index = cutlass.Int32(self.L - 1) - tidx
        transition_step_re = cutlass.Float32(0.0)
        transition_step_im = cutlass.Float32(0.0)
        prev_tap_re = cutlass.Float32(0.0)
        prev_tap_im = cutlass.Float32(0.0)
        curr_tap_re = cutlass.Float32(0.0)
        curr_tap_im = cutlass.Float32(0.0)

        # Reverse-time suffix scan over the packed-complex transition stream.
        if tidx < cutlass.Int32(self.L):
            transition_step_re = cutlass.Float32(
                mM[0, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
            )
            transition_step_im = cutlass.Float32(
                mM[1, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
            )
            prev_tap_re = cutlass.Float32(
                mKprev[0, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
            )
            prev_tap_im = cutlass.Float32(
                mKprev[1, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
            )
            curr_tap_re = cutlass.Float32(
                mKcurr[0, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
            )
            curr_tap_im = cutlass.Float32(
                mKcurr[1, reverse_time_index, batch_head_chunk_idx].to(cutlass.Float32)
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
                s_warp_transition_re_total[warp_idx] = suffix_transition_re
                s_warp_transition_im_total[warp_idx] = suffix_transition_im

        cute.arch.sync_threads()

        if cutlass.const_expr(self.scan_threads > 32):
            if warp_idx == cutlass.Int32(0):
                warp_lane = lane_idx
                has_scan_warp = warp_lane < scan_warp_count
                warp_transition_re = cutlass.select_(
                    has_scan_warp,
                    s_warp_transition_re_total[warp_lane],
                    cutlass.Float32(1.0),
                )
                warp_transition_im = cutlass.select_(
                    has_scan_warp,
                    s_warp_transition_im_total[warp_lane],
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
                    s_warp_transition_re_offset[warp_lane] = warp_offset_re
                    s_warp_transition_im_offset[warp_lane] = warp_offset_im

            cute.arch.sync_threads()

            if warp_idx < scan_warp_count:
                warp_offset_re = s_warp_transition_re_offset[warp_idx]
                warp_offset_im = s_warp_transition_im_offset[warp_idx]
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
            s_warp_transition_re_total[warp_idx] = suffix_transition_re
            s_warp_transition_im_total[warp_idx] = suffix_transition_im

        cute.arch.sync_threads()

        # Convert the inclusive suffix products into the two coefficient streams
        # used by the fused dB / dMsum epilogue.
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
                    suffix_prefix_re = s_warp_transition_re_total[warp_idx - 1]
                    suffix_prefix_im = s_warp_transition_im_total[warp_idx - 1]
                else:
                    suffix_prefix_re = prior_transition_re
                    suffix_prefix_im = prior_transition_im

            prev_coeff_re = (
                suffix_prefix_re * prev_tap_re - suffix_prefix_im * prev_tap_im
            )
            prev_coeff_im = (
                suffix_prefix_re * prev_tap_im + suffix_prefix_im * prev_tap_re
            )
            sum_coeff_re = (
                suffix_prefix_re * curr_tap_re - suffix_prefix_im * curr_tap_im
            )
            sum_coeff_im = (
                suffix_prefix_re * curr_tap_im + suffix_prefix_im * curr_tap_re
            )
            s_suffix_coeff_prev[reverse_time_index, 0] = prev_coeff_re
            s_suffix_coeff_prev[reverse_time_index, 1] = prev_coeff_im
            s_suffix_coeff_sum[reverse_time_index, 0] = sum_coeff_re
            s_suffix_coeff_sum[reverse_time_index, 1] = sum_coeff_im

        cute.arch.sync_threads()

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

        # Pipelined tensor-core mainloop over the chunk-local P tiles.
        for k_tile in range(1, num_smem_stages - 1):
            cute.copy(
                gmem_tiled_copy_u,
                t_u_gmem[None, None, None, k_tile_index],
                t_u_copy_dst[None, None, None, k_tile],
                pred=u_row_predicate,
            )
            cute.copy(
                gmem_tiled_copy_d_inc,
                t_d_inc_gmem[None, None, None, k_tile_index],
                t_d_inc_copy_dst[None, None, None, k_tile],
                pred=d_inc_row_predicate,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(num_smem_stages - 1)
        mma_k_block_count = cute.size(r_u_mma, mode=[2])

        for kt in range(k_tile_count):
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()

            next_tile = kt + (num_smem_stages - 1)
            if next_tile < k_tile_count:
                cute.copy(
                    gmem_tiled_copy_u,
                    t_u_gmem[None, None, None, k_tile_index],
                    t_u_copy_dst[None, None, None, smem_pipe_write],
                    pred=u_row_predicate,
                )
                cute.copy(
                    gmem_tiled_copy_d_inc,
                    t_d_inc_gmem[None, None, None, k_tile_index],
                    t_d_inc_copy_dst[None, None, None, smem_pipe_write],
                    pred=d_inc_row_predicate,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            t_u_ldmatrix_stage = t_u_ldmatrix_smem[None, None, None, smem_pipe_read]
            t_d_inc_ldmatrix_stage = t_d_inc_ldmatrix_smem[
                None, None, None, smem_pipe_read
            ]
            cute.copy(
                ldmatrix_tiled_copy_u,
                t_u_ldmatrix_stage[None, None, 0],
                r_u_ldmatrix[None, None, 0],
            )
            cute.copy(
                ldmatrix_tiled_copy_d_inc,
                t_d_inc_ldmatrix_stage[None, None, 0],
                r_d_inc_ldmatrix[None, None, 0],
            )
            for kb in cutlass.range(mma_k_block_count, unroll_full=True):
                kb_next = (kb + 1) % mma_k_block_count
                cute.copy(
                    ldmatrix_tiled_copy_u,
                    t_u_ldmatrix_stage[None, None, kb_next],
                    r_u_ldmatrix[None, None, kb_next],
                )
                cute.copy(
                    ldmatrix_tiled_copy_d_inc,
                    t_d_inc_ldmatrix_stage[None, None, kb_next],
                    r_d_inc_ldmatrix[None, None, kb_next],
                )
                cute.gemm(
                    tiled_mma,
                    r_db_accum,
                    r_u_mma[None, None, kb],
                    r_d_inc_mma[None, None, kb],
                    r_db_accum,
                )

            smem_pipe_write = smem_pipe_read
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == num_smem_stages:
                smem_pipe_read = 0

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Fused dB store and dMsum partial reduction epilogue.
        cute.autovec_copy(r_db_accum, t_db_mma_smem)
        cute.arch.sync_threads()

        warp_count = cutlass.Int32(self.num_threads // 32)
        complex_pair_count = cutlass.Int32(self.bN // 2)
        time_index = warp_idx
        while cute.elem_less(time_index, cutlass.Int32(self.L)):
            d_pair_start = lane_idx * cutlass.Int32(2)
            global_d_pair_start = tile_col_start + d_pair_start

            d_b_sum_re = cutlass.Float32(0.0)
            d_b_sum_im = cutlass.Float32(0.0)
            b_re = cutlass.Float32(0.0)
            b_im = cutlass.Float32(0.0)
            if cute.elem_less(lane_idx, complex_pair_count) and cute.elem_less(
                global_d_pair_start + 1, self.D
            ):
                d_b_sum_re = cutlass.Float32(s_db_tile[time_index, d_pair_start + 0])
                d_b_sum_im = cutlass.Float32(s_db_tile[time_index, d_pair_start + 1])
                b_re = cutlass.Float32(
                    mB[time_index, global_d_pair_start + 0, batch_head_chunk_idx].to(
                        cutlass.Float32
                    )
                )
                b_im = cutlass.Float32(
                    mB[time_index, global_d_pair_start + 1, batch_head_chunk_idx].to(
                        cutlass.Float32
                    )
                )

                db_coeff_re = s_suffix_coeff_sum[time_index, 0]
                db_coeff_im = s_suffix_coeff_sum[time_index, 1]
                rotated_db_re = db_coeff_re * d_b_sum_re + db_coeff_im * d_b_sum_im
                rotated_db_im = db_coeff_re * d_b_sum_im - db_coeff_im * d_b_sum_re
                mDB[time_index, global_d_pair_start + 0, batch_head_chunk_idx] = (
                    rotated_db_re.to(mDB.element_type)
                )
                mDB[time_index, global_d_pair_start + 1, batch_head_chunk_idx] = (
                    rotated_db_im.to(mDB.element_type)
                )

            d_msum_partial_re = d_b_sum_re * b_re + d_b_sum_im * b_im
            d_msum_partial_im = d_b_sum_im * b_re - d_b_sum_re * b_im

            for offset in (16, 8, 4, 2, 1):
                d_msum_partial_re = d_msum_partial_re + cute.arch.shuffle_sync_bfly(
                    d_msum_partial_re, offset=offset, mask=-1, mask_and_clamp=31
                )
                d_msum_partial_im = d_msum_partial_im + cute.arch.shuffle_sync_bfly(
                    d_msum_partial_im, offset=offset, mask=-1, mask_and_clamp=31
                )

            if lane_idx == cutlass.Int32(0):
                mDMsumPart[0, time_index, block_col_idx, batch_head_chunk_idx] = (
                    d_msum_partial_re
                )
                mDMsumPart[1, time_index, block_col_idx, batch_head_chunk_idx] = (
                    d_msum_partial_im
                )

            time_index = time_index + warp_count
