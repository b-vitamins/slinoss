"""CuTe backward kernel for the ``v2x2ssd`` chunk-scan ``du`` slice.

``ChunkScanBwdDUAmpere`` is the live Ampere tensor-core implementation that
owns the backward value-gradient slice of the standalone ``chunk_scan`` stage.
It reconstructs the chunk-local prefix metadata from raw packed transitions
``M``, applies the two packed-complex tap passes from ``K``, accumulates the
two causal score/value contractions needed for ``dU`` / ``dU_prev``, and writes
the public value-gradient outputs directly.

Tensor contracts:

- ``U``: ``(BHC, L, 1, P)`` fp16/bf16 value input
- ``B``: ``(BGC, L, 1, D)`` fp16/bf16 key-side packed complex input
- ``C``: ``(BGC, L, 1, D)`` fp16/bf16 query-side packed complex input
- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps
- ``dOut``: ``(BHC, L, 1, P)`` fp16/bf16 output gradient
- ``U_prev0``: ``(BH, P)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(BG, D)`` fp16/bf16 chunk-0 boundary key row
- ``dU``: ``(BHC, L, 1, P)`` fp16/bf16 value gradients
- ``dU_prev``: ``(BHC, P)`` fp16/bf16 chunk-boundary value carry gradients

The remaining outputs (``dB``, ``dB_prev``, ``dlogprefix``, ``dMprev``,
``dMcurr``) are accepted only to preserve the staged backward ABI; this kernel
does not write them.

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be
even and conceptually corresponds to ``2 * N``.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import torch
from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    apply_complex_tap,
    clamp_nonpositive_prefix_log,
    complex_mul,
    conj_mul_phase,
    safe_cast_to_dtype,
)


@dataclass(frozen=True)
class ChunkScanBwdDULayoutBundle:
    query_layout: object
    grad_output_layout: object
    key_layout: object
    output_layout: object
    prev_output_layout: object
    score_layout: object
    phase_layout: object
    tap_prev_layout: object
    tap_curr_layout: object
    row_scale_layout: object
    inv_row_scale_layout: object
    carry_layout: object
    warp_log_layout: object
    warp_phase_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDUCopyBundle:
    gmem_tiled_copy_d: object
    gmem_tiled_copy_p: object
    gmem_tiled_store_p: object


@dataclass(frozen=True)
class ChunkScanBwdDUKernelBundle:
    layouts: ChunkScanBwdDULayoutBundle
    copies: ChunkScanBwdDUCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.smem_bytes


@dataclass(frozen=True)
class ChunkScanBwdDUSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkScanBwdDUAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``du`` slice.

    This kernel owns the chunk-local value-gradient path. It rebuilds the
    prefix metadata from raw packed ``M``, applies the two complex tap passes
    from ``K``, accumulates the causal score/value contractions for ``dU`` and
    ``dU_prev``, and writes the public value-gradient outputs directly.
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkScanBwdDUSupportInfo]
    ] = {}

    def __init__(
        self,
        dtype,
        *,
        chunk_size,
        D,
        P,
        num_threads=128,
        heads: int | None = None,
        bc_groups: int | None = None,
    ):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.num_threads = int(num_threads)
        self.has_group_geometry = heads is not None
        if self.has_group_geometry:
            self.heads = int(heads)
            self.bc_groups = self.heads if bc_groups is None else int(bc_groups)
            if self.heads <= 0 or self.bc_groups <= 0:
                raise ValueError("heads and bc_groups must be positive.")
            if self.heads % self.bc_groups != 0:
                raise ValueError("bc_groups must divide heads.")
            self.heads_per_bc_group = self.heads // self.bc_groups
        else:
            if bc_groups is not None:
                raise ValueError("bc_groups requires heads to be specified.")
            self.heads = 0
            self.bc_groups = 0
            self.heads_per_bc_group = 0

        self.kv_tile = 32
        self.p_tile = 32
        self.mma_inst_shape = (16, 8, 16)
        self.warp_layout_mnk = (2, 2, 1)
        self.atom_layout_mnk = self.warp_layout_mnk

        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.L % self.kv_tile != 0:
            raise ValueError("chunk_size must be a multiple of 32.")
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")

        expected_threads = 32 * self.warp_layout_mnk[0] * self.warp_layout_mnk[1]
        if self.num_threads != expected_threads:
            raise ValueError(
                f"num_threads must be {expected_threads} for the 32x32 tensorop tile."
            )

    @cute.jit
    def _batch_group(self, batch_head: int):
        if cutlass.const_expr(not self.has_group_geometry):
            return batch_head
        batch_idx = batch_head // cutlass.Int32(self.heads)
        head_idx = batch_head - batch_idx * cutlass.Int32(self.heads)
        group_idx = head_idx // cutlass.Int32(self.heads_per_bc_group)
        return batch_idx * cutlass.Int32(self.bc_groups) + group_idx

    @cute.jit
    def _batch_group_chunk(self, batch_head_chunk: int, n_chunks: int):
        if cutlass.const_expr(not self.has_group_geometry):
            return batch_head_chunk
        batch_head = batch_head_chunk // n_chunks
        chunk_index = batch_head_chunk - batch_head * n_chunks
        batch_group = self._batch_group(batch_head)
        return batch_group * n_chunks + chunk_index

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

    @property
    def num_warps(self) -> int:
        return self.num_threads // 32

    @property
    def num_kv_tiles(self) -> int:
        return self.L // self.kv_tile

    @property
    def num_p_tiles(self) -> int:
        return self.P_padded // self.p_tile

    # End-to-end specialization
    def _d_stage_size(self) -> int:
        """Width of each staged ``D`` slice used by the backward DU kernel."""
        d_padded = self.D_padded
        if d_padded <= 64:
            return d_padded
        return 64

    def _d_stage_count(self) -> int:
        d_stage_width = self._d_stage_size()
        return (self.D_padded + d_stage_width - 1) // d_stage_width

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

    def _compute_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        in_bytes = in_dtype.width // 8
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        d_stage_width = self._d_stage_size()
        p_padded = self.P_padded

        return self._struct_size_bytes(
            [
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * 4, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (self.L * 2 * 4, 16),
                (self.L * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (self.L * 4, 16),
                (self.L * 4, 16),
                (p_padded * 4, 16),
                (p_padded * 4, 16),
                (self.num_warps * 4, 16),
                (self.num_warps * 4, 16),
                (self.num_warps * 2 * 4, 16),
                (self.num_warps * 2 * 4, 16),
            ]
        )

    def support_info(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkScanBwdDUSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkScanBwdDUSupportInfo(0, 1)

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
            self.num_threads,
            in_dtype,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkScanBwdDUSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._compute_smem_bytes(in_dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> bool:
        return self.support_info(in_dtype, device_index=device_index).supported

    def _make_accumulator_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
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

    def _make_layout_bundle(self) -> ChunkScanBwdDULayoutBundle:
        kv_tile = self.kv_tile
        d_stage_width = self._d_stage_size()

        smem_k_block_size_d = 64 if d_stage_width % 64 == 0 else 32
        swizzle_bits_d = 3 if smem_k_block_size_d == 64 else 2
        d_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_d, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_d), stride=(smem_k_block_size_d, 1)),
        )
        query_layout = cute.tile_to_shape(
            d_layout_atom, (kv_tile, d_stage_width), (0, 1)
        )
        key_layout = cute.tile_to_shape(d_layout_atom, (kv_tile, d_stage_width), (0, 1))

        smem_k_block_size_p = self.p_tile
        swizzle_bits_p = 3 if smem_k_block_size_p == 64 else 2
        p_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_p, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_p), stride=(smem_k_block_size_p, 1)),
        )
        grad_output_layout = cute.tile_to_shape(
            p_layout_atom, (kv_tile, self.p_tile), (0, 1)
        )
        output_layout = cute.tile_to_shape(
            p_layout_atom, (kv_tile, self.p_tile), (0, 1)
        )
        prev_output_layout = cute.make_layout((kv_tile,), stride=(1,))

        smem_k_block_size_score = kv_tile
        swizzle_bits_score = 3 if smem_k_block_size_score == 64 else 2
        score_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_score, 3, 3),
            0,
            cute.make_layout(
                (8, smem_k_block_size_score), stride=(smem_k_block_size_score, 1)
            ),
        )
        score_layout = cute.tile_to_shape(score_layout_atom, (kv_tile, kv_tile), (0, 1))

        return ChunkScanBwdDULayoutBundle(
            query_layout=query_layout,
            grad_output_layout=grad_output_layout,
            key_layout=key_layout,
            output_layout=output_layout,
            prev_output_layout=prev_output_layout,
            score_layout=score_layout,
            phase_layout=cute.make_layout((self.L, 2), stride=(2, 1)),
            tap_prev_layout=cute.make_layout((self.L, 2), stride=(2, 1)),
            tap_curr_layout=cute.make_layout((self.kv_tile, 2), stride=(2, 1)),
            row_scale_layout=cute.make_layout((self.L,), stride=(1,)),
            inv_row_scale_layout=cute.make_layout((self.L,), stride=(1,)),
            carry_layout=cute.make_layout((self.P_padded,), stride=(1,)),
            warp_log_layout=cute.make_layout((self.num_warps,), stride=(1,)),
            warp_phase_layout=cute.make_layout((self.num_warps, 2), stride=(2, 1)),
        )

    def _make_copy_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDUCopyBundle:
        universal_copy_bits = 128
        elems_per_copy = universal_copy_bits // in_dtype.width
        d_stage_width = self._d_stage_size()

        smem_k_block_size_d = 64 if d_stage_width % 64 == 0 else 32
        t_d_shape_dim_1 = smem_k_block_size_d // elems_per_copy
        t_d_layout = cute.make_layout(
            (self.num_threads // t_d_shape_dim_1, t_d_shape_dim_1),
            stride=(t_d_shape_dim_1, 1),
        )
        t_p_shape_dim_1 = self.p_tile // elems_per_copy
        t_p_layout = cute.make_layout(
            (self.num_threads // t_p_shape_dim_1, t_p_shape_dim_1),
            stride=(t_p_shape_dim_1, 1),
        )
        value_layout = cute.make_layout((1, elems_per_copy))

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        return ChunkScanBwdDUCopyBundle(
            gmem_tiled_copy_d=cute.make_tiled_copy_tv(
                atom_async_copy, t_d_layout, value_layout
            ),
            gmem_tiled_copy_p=cute.make_tiled_copy_tv(
                atom_async_copy, t_p_layout, value_layout
            ),
            gmem_tiled_store_p=cute.make_tiled_copy_tv(
                atom_store, t_p_layout, value_layout
            ),
        )

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        op = cute.nvgpu.warp.MmaF16BF16Op(in_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanBwdDULayoutBundle,
    ):
        @cute.struct
        class SharedStorage:
            query_stage: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)],
                16,
            ]
            grad_output_stage: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.grad_output_layout)],
                16,
            ]
            key_stage: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.key_layout)],
                16,
            ]
            output_stage: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.output_layout)],
                16,
            ]
            prev_output_tile: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.prev_output_layout)
                ],
                16,
            ]
            score_block: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)],
                16,
            ]
            score_block_alt: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)],
                16,
            ]
            phase_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.phase_layout)
                ],
                16,
            ]
            tap_prev_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tap_prev_layout)
                ],
                16,
            ]
            tap_curr_tile: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tap_curr_layout)
                ],
                16,
            ]
            row_scale_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_scale_layout)
                ],
                16,
            ]
            inv_row_scale_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.inv_row_scale_layout)
                ],
                16,
            ]
            du_carry: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.carry_layout)
                ],
                16,
            ]
            du_carry_next: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.carry_layout)
                ],
                16,
            ]
            warp_log_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.warp_log_layout)
                ],
                16,
            ]
            warp_log_offset: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.warp_log_layout)
                ],
                16,
            ]
            warp_phase_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.warp_phase_layout)
                ],
                16,
            ]
            warp_phase_offset: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.warp_phase_layout)
                ],
                16,
            ]

        return SharedStorage

    def _make_kernel_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDUKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        return ChunkScanBwdDUKernelBundle(
            layouts=layouts,
            copies=self._make_copy_bundle(in_dtype),
            tiled_mma=self._make_tiled_mma(in_dtype),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    # Copy and predication helpers
    @cute.jit
    def _accumulate_from_staged_tiles(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        smem_tiled_copy_lhs: cute.TiledCopy,
        smem_tiled_copy_rhs: cute.TiledCopy,
        t_smem_lhs: cute.Tensor,
        t_reg_lhs_view: cute.Tensor,
        t_smem_rhs: cute.Tensor,
        t_reg_rhs_view: cute.Tensor,
        t_reg_lhs: cute.Tensor,
        t_reg_rhs: cute.Tensor,
    ):
        cute.copy(
            smem_tiled_copy_lhs,
            t_smem_lhs[None, None, 0],
            t_reg_lhs_view[None, None, 0],
        )
        cute.copy(
            smem_tiled_copy_rhs,
            t_smem_rhs[None, None, 0],
            t_reg_rhs_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(t_smem_lhs.shape[2])):
            k_next = (k + 1) % cute.size(t_smem_lhs.shape[2])
            cute.copy(
                smem_tiled_copy_lhs,
                t_smem_lhs[None, None, k_next],
                t_reg_lhs_view[None, None, k_next],
            )
            cute.copy(
                smem_tiled_copy_rhs,
                t_smem_rhs[None, None, k_next],
                t_reg_rhs_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc,
                t_reg_lhs[None, None, k],
                t_reg_rhs[None, None, k],
                acc,
            )
        cute.arch.barrier()

    @cute.jit
    def _make_copy_column_predicate(
        self,
        partitioned_tensor: cute.Tensor,
        partitioned_coord: cute.Tensor,
        col_limit: int,
    ):
        pred = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    partitioned_tensor.shape[0][1],
                    cute.size(partitioned_tensor, mode=[1]),
                    cute.size(partitioned_tensor, mode=[2]),
                ),
                stride=(cute.size(partitioned_tensor, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(pred.shape[0]):
            for rest_k in cutlass.range_constexpr(pred.shape[2]):
                pred[rest_v, 0, rest_k] = cute.elem_less(
                    partitioned_coord[(0, rest_v), 0, rest_k][3], col_limit
                )
        return pred

    @cute.jit
    def _refresh_d_stage_copy_predicate(
        self,
        copy_pred: cute.Tensor,
        partitioned_coord: cute.Tensor,
        *,
        d_col_base: int,
        col_limit: int,
    ):
        for rest_v in cutlass.range_constexpr(copy_pred.shape[0]):
            for rest_k in cutlass.range_constexpr(copy_pred.shape[2]):
                g_col = cutlass.Int32(
                    partitioned_coord[(0, rest_v), 0, rest_k][3]
                ) + cutlass.Int32(d_col_base)
                copy_pred[rest_v, 0, rest_k] = cute.elem_less(g_col, col_limit)

    @cute.jit
    def _copy_rows_with_zero_fill(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        t_copy_src: cute.Tensor,
        t_copy_dst: cute.Tensor,
        t_coord: cute.Tensor,
        copy_pred: cute.Tensor,
        row_limit: int,
    ):
        for idx in cutlass.range_constexpr(cute.size(t_copy_dst.shape[1])):
            if cute.elem_less(t_coord[0, idx, 0][1], row_limit):
                cute.copy(
                    gmem_tiled_copy,
                    t_copy_src[None, idx, None],
                    t_copy_dst[None, idx, None],
                    pred=copy_pred[None, idx, None],
                )
            else:
                t_copy_dst[None, idx, None].fill(0)

    @cute.jit
    def _copy_rows_if_valid(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        t_copy_src: cute.Tensor,
        t_copy_dst: cute.Tensor,
        t_coord: cute.Tensor,
        copy_pred: cute.Tensor,
        row_limit: int,
    ):
        for idx in cutlass.range_constexpr(cute.size(t_copy_dst.shape[1])):
            if cute.elem_less(t_coord[0, idx, 0][1], row_limit):
                cute.copy(
                    gmem_tiled_copy,
                    t_copy_src[None, idx, None],
                    t_copy_dst[None, idx, None],
                    pred=copy_pred[None, idx, None],
                )

    # Prefix helpers
    @cute.jit
    def _compute_phase_prefix_metadata(self, prefix_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        seq_idx = tidx
        logp = cutlass.Float32(0.0)
        phase_re = cutlass.Float32(1.0)
        phase_im = cutlass.Float32(0.0)

        if tidx < self.L:
            mr = cutlass.Float32(
                prefix_state.m_transition[prefix_state.batch_head_chunk, seq_idx, 0]
            )
            mi = cutlass.Float32(
                prefix_state.m_transition[prefix_state.batch_head_chunk, seq_idx, 1]
            )
            mag2 = mr * mr + mi * mi + cutlass.Float32(1.0e-20)
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2))
            phase_re = mr * inv_mag
            phase_im = mi * inv_mag
            logp = cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.25 / LOG2_E)

        for offset in (1, 2, 4, 8, 16):
            other_log = cute.arch.shuffle_sync_up(
                logp, offset=offset, mask=-1, mask_and_clamp=0
            )
            other_phase_re = cute.arch.shuffle_sync_up(
                phase_re, offset=offset, mask=-1, mask_and_clamp=0
            )
            other_phase_im = cute.arch.shuffle_sync_up(
                phase_im, offset=offset, mask=-1, mask_and_clamp=0
            )
            pred = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(pred, logp + other_log, logp)
            next_phase_re, next_phase_im = complex_mul(
                phase_re, phase_im, other_phase_re, other_phase_im
            )
            phase_re = cutlass.select_(pred, next_phase_re, phase_re)
            phase_im = cutlass.select_(pred, next_phase_im, phase_im)

        if lane == cutlass.Int32(31):
            prefix_state.warp_log_total[warp] = logp
            prefix_state.warp_phase_total[warp, 0] = phase_re
            prefix_state.warp_phase_total[warp, 1] = phase_im
        cute.arch.barrier()

        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            running_log = cutlass.Float32(0.0)
            running_phase_re = cutlass.Float32(1.0)
            running_phase_im = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(self.num_warps):
                prefix_state.warp_log_offset[w] = running_log
                prefix_state.warp_phase_offset[w, 0] = running_phase_re
                prefix_state.warp_phase_offset[w, 1] = running_phase_im

                total_log = prefix_state.warp_log_total[w]
                total_phase_re = prefix_state.warp_phase_total[w, 0]
                total_phase_im = prefix_state.warp_phase_total[w, 1]
                next_running_phase_re, next_running_phase_im = complex_mul(
                    running_phase_re,
                    running_phase_im,
                    total_phase_re,
                    total_phase_im,
                )
                running_log = running_log + total_log
                running_phase_re = next_running_phase_re
                running_phase_im = next_running_phase_im
        cute.arch.barrier()

        warp_log_offset = prefix_state.warp_log_offset[warp]
        warp_phase_re_offset = prefix_state.warp_phase_offset[warp, 0]
        warp_phase_im_offset = prefix_state.warp_phase_offset[warp, 1]
        logp = logp + warp_log_offset
        phase_re, phase_im = complex_mul(
            phase_re, phase_im, warp_phase_re_offset, warp_phase_im_offset
        )

        if tidx < self.L:
            stable_logp = clamp_nonpositive_prefix_log(logp)
            row_scale = cute.math.exp2(
                stable_logp * cutlass.Float32(TWO_LOG2_E), fastmath=False
            )
            prefix_state.s_row_scale[seq_idx] = row_scale
            prefix_state.s_inv_row_scale[seq_idx] = cutlass.Float32(1.0) / row_scale
            prefix_state.s_phase[seq_idx, 0] = phase_re
            prefix_state.s_phase[seq_idx, 1] = phase_im

            tap_prev = (
                prefix_state.m_tap[prefix_state.batch_head_chunk, seq_idx, 0, None]
                .load()
                .to(cutlass.Float32)
            )
            prefix_state.s_tap_prev[seq_idx, 0] = cutlass.Float32(tap_prev[0])
            prefix_state.s_tap_prev[seq_idx, 1] = cutlass.Float32(tap_prev[1])
        cute.arch.barrier()

    @cute.jit
    def _rotate_staged_query_tile_from_prefix(
        self,
        s_query: cute.Tensor,
        s_phase: cute.Tensor,
        *,
        m_tile_start: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        stage_complex = stage_width // 2
        total_pairs = self.kv_tile * stage_complex
        iters_pairs_stage = (total_pairs + self.num_threads - 1) // self.num_threads
        for it in range(iters_pairs_stage):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < cutlass.Int32(total_pairs):
                row_local = idx // stage_complex
                vec_idx = idx - row_local * stage_complex
                row_idx = cutlass.Int32(m_tile_start) + row_local
                if row_idx < cutlass.Int32(self.L):
                    d0 = vec_idx * 2
                    xr = cutlass.Float32(s_query[row_local, d0 + 0])
                    xi = cutlass.Float32(s_query[row_local, d0 + 1])
                    pr = cutlass.Float32(s_phase[row_idx, 0])
                    pi = cutlass.Float32(s_phase[row_idx, 1])
                    yr, yi = conj_mul_phase(xr, xi, pr, pi)
                    s_query[row_local, d0 + 0] = safe_cast_to_dtype(yr, out_dtype)
                    s_query[row_local, d0 + 1] = safe_cast_to_dtype(yi, out_dtype)
        cute.arch.barrier()

    @cute.jit
    def _load_current_tap_tile(
        self,
        m_tap: cute.Tensor,
        s_tap_curr: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile_start: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        if tidx < cutlass.Int32(self.kv_tile):
            key_idx = cutlass.Int32(n_tile_start) + tidx
            tap = m_tap[batch_head_chunk, key_idx, 1, None].load().to(cutlass.Float32)
            s_tap_curr[tidx, 0] = cutlass.Float32(tap[0])
            s_tap_curr[tidx, 1] = cutlass.Float32(tap[1])

    @cute.jit
    def _inject_previous_boundary_key_row(
        self,
        s_key: cute.Tensor,
        m_key: cute.Tensor,
        m_key_prev0: cute.Tensor,
        *,
        batch_group_chunk: int,
        batch_group: int,
        chunk_index: int,
        d_col_base: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        iters_stage_width = (stage_width + self.num_threads - 1) // self.num_threads
        for it in range(iters_stage_width):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < cutlass.Int32(stage_width):
                d_idx = cutlass.Int32(d_col_base) + idx
                value = out_dtype(0)
                if d_idx < cutlass.Int32(self.D):
                    if chunk_index == cutlass.Int32(0):
                        value = m_key_prev0[batch_group, d_idx]
                    else:
                        value = m_key[
                            batch_group_chunk - cutlass.Int32(1),
                            cutlass.Int32(self.L - 1),
                            0,
                            d_idx,
                        ]
                s_key[0, idx] = value
        cute.arch.barrier()

    @cute.jit
    def _apply_tap_phase_to_staged_keys(
        self,
        s_key: cute.Tensor,
        s_phase: cute.Tensor,
        s_tap: cute.Tensor,
        *,
        n_tile_start: int,
        stage_width: int,
        tap_is_full: cutlass.Constexpr,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        stage_complex = stage_width // 2
        total_pairs = self.kv_tile * stage_complex
        iters_pairs_stage = (total_pairs + self.num_threads - 1) // self.num_threads
        for it in range(iters_pairs_stage):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < cutlass.Int32(total_pairs):
                row_local = idx // stage_complex
                vec_idx = idx - row_local * stage_complex
                key_idx = cutlass.Int32(n_tile_start) + row_local
                if key_idx < cutlass.Int32(self.L):
                    d0 = vec_idx * 2
                    bx = cutlass.Float32(s_key[row_local, d0 + 0])
                    by = cutlass.Float32(s_key[row_local, d0 + 1])

                    kr = cutlass.Float32(0.0)
                    ki = cutlass.Float32(0.0)
                    if cutlass.const_expr(tap_is_full):
                        kr = cutlass.Float32(s_tap[key_idx, 0])
                        ki = cutlass.Float32(s_tap[key_idx, 1])
                    else:
                        kr = cutlass.Float32(s_tap[row_local, 0])
                        ki = cutlass.Float32(s_tap[row_local, 1])

                    tr, ti = apply_complex_tap(bx, by, kr, ki)
                    pr = cutlass.Float32(s_phase[key_idx, 0])
                    pi = cutlass.Float32(s_phase[key_idx, 1])
                    kx, ky = conj_mul_phase(tr, ti, pr, pi)
                    s_key[row_local, d0 + 0] = safe_cast_to_dtype(kx, out_dtype)
                    s_key[row_local, d0 + 1] = safe_cast_to_dtype(ky, out_dtype)
        cute.arch.barrier()

    # Score helpers
    @cute.jit
    def _make_score_coord_tile(
        self,
        coord_score: cute.Tensor,
        *,
        batch_group_chunk: int,
        m_tile: int,
        n_tile: int,
    ):
        return cute.local_tile(
            coord_score[batch_group_chunk, None, 0, None],
            (self.kv_tile, self.kv_tile),
            (m_tile, n_tile),
        )

    @cute.jit
    def _apply_score_scales_and_mask(
        self,
        prefix_state: SimpleNamespace,
        acc_score: cute.Tensor,
        t_score_coord_mn: cute.Tensor,
        s_score: cute.Tensor,
        *,
        n_tile_start: int,
        m_tile_start: int,
        out_dtype: type[cutlass.Numeric],
    ):
        acc_score_mn = self._make_accumulator_mn_view(acc_score)
        n_tile_start = cutlass.Int32(n_tile_start)
        m_tile_start = cutlass.Int32(m_tile_start)
        for r in cutlass.range_constexpr(cute.size(acc_score_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            row_local = row_idx - m_tile_start
            row_scale = cutlass.Float32(prefix_state.s_row_scale[row_idx])
            for c in cutlass.range_constexpr(cute.size(acc_score_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                col_local = col_idx - n_tile_start
                inv_row_scale = cutlass.Float32(prefix_state.s_inv_row_scale[col_idx])
                scaled = cutlass.Float32(0.0)
                if cute.elem_less(col_idx, row_idx + 1):
                    scaled = acc_score_mn[r, c] * (row_scale * inv_row_scale)
                s_score[col_local, row_local] = safe_cast_to_dtype(scaled, out_dtype)
        cute.arch.barrier()

    @cute.jit
    def _apply_score_scales_and_mask_shifted_prev(
        self,
        prefix_state: SimpleNamespace,
        acc_score: cute.Tensor,
        t_score_coord_mn: cute.Tensor,
        s_score: cute.Tensor,
        s_score_row0: cute.Tensor,
        *,
        n_tile_start: int,
        m_tile_start: int,
        out_dtype: type[cutlass.Numeric],
    ):
        acc_score_mn = self._make_accumulator_mn_view(acc_score)
        n_tile_start = cutlass.Int32(n_tile_start)
        m_tile_start = cutlass.Int32(m_tile_start)
        last_row_local = cutlass.Int32(self.kv_tile - 1)
        for r in cutlass.range_constexpr(cute.size(acc_score_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            row_local = row_idx - m_tile_start
            row_scale = cutlass.Float32(prefix_state.s_row_scale[row_idx])
            for c in cutlass.range_constexpr(cute.size(acc_score_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                col_local = col_idx - n_tile_start
                inv_row_scale = cutlass.Float32(prefix_state.s_inv_row_scale[col_idx])
                scaled = cutlass.Float32(0.0)
                if cute.elem_less(col_idx, row_idx + 1):
                    scaled = acc_score_mn[r, c] * (row_scale * inv_row_scale)
                if col_local == cutlass.Int32(0):
                    s_score_row0[row_local] = scaled
                    s_score[last_row_local, row_local] = out_dtype(0)
                else:
                    s_score[col_local - cutlass.Int32(1), row_local] = (
                        safe_cast_to_dtype(scaled, out_dtype)
                    )
        cute.arch.barrier()

    @cute.jit
    def _accumulate_score_block_pair_reuse_query(
        self,
        m_query: cute.Tensor,
        m_key: cute.Tensor,
        m_key_prev0: cute.Tensor,
        coord_query: cute.Tensor,
        coord_key: cute.Tensor,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_thr_copy_d: object,
        t_query_smem: cute.Tensor,
        t_key_smem: cute.Tensor,
        d_stage_copy_pred: cute.Tensor,
        s_query: cute.Tensor,
        s_key: cute.Tensor,
        s_tap_curr: cute.Tensor,
        prefix_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        acc_score_curr: cute.Tensor,
        acc_score_prev: cute.Tensor,
        *,
        batch_group_chunk: int,
        batch_group: int,
        chunk_index: int,
        n_tile: int,
        m_tile: int,
        n_tile_start: int,
        m_tile_start: int,
        out_dtype: cutlass.Constexpr,
    ):
        d_stage_width = self._d_stage_size()

        query_coord_stage0 = cute.local_tile(
            coord_query[batch_group_chunk, None, 0, None],
            (self.kv_tile, d_stage_width),
            (m_tile, 0),
        )
        t_query_coord_stage0 = gmem_thr_copy_d.partition_S(query_coord_stage0)

        key_coord_curr_stage0 = cute.local_tile(
            coord_key[batch_group_chunk, None, 0, None],
            (self.kv_tile, d_stage_width),
            (n_tile, 0),
        )
        t_key_coord_curr_stage0 = gmem_thr_copy_d.partition_S(key_coord_curr_stage0)
        key_coord_prev_stage0 = cute.domain_offset((-1, 0), key_coord_curr_stage0)
        t_key_coord_prev_stage0 = gmem_thr_copy_d.partition_S(key_coord_prev_stage0)

        for d_stage_idx in cutlass.range_constexpr(self._d_stage_count()):
            d_col_base = cutlass.Int32(d_stage_idx * d_stage_width)
            self._refresh_d_stage_copy_predicate(
                d_stage_copy_pred,
                t_query_coord_stage0,
                d_col_base=d_col_base,
                col_limit=m_query.layout.shape[3],
            )

            g_query_stage = cute.local_tile(
                m_query[batch_group_chunk, None, 0, None],
                (self.kv_tile, d_stage_width),
                (m_tile, d_stage_idx),
            )
            t_query_gmem = gmem_thr_copy_d.partition_S(g_query_stage)
            self._copy_rows_with_zero_fill(
                gmem_tiled_copy_d,
                t_query_gmem,
                t_query_smem,
                t_query_coord_stage0,
                d_stage_copy_pred,
                m_query.layout.shape[1],
            )

            g_key_curr_stage = cute.local_tile(
                m_key[batch_group_chunk, None, 0, None],
                (self.kv_tile, d_stage_width),
                (n_tile, d_stage_idx),
            )
            t_key_curr_gmem = gmem_thr_copy_d.partition_S(g_key_curr_stage)
            for idx in cutlass.range_constexpr(cute.size(t_key_smem.shape[1])):
                row_idx = cutlass.Int32(t_key_coord_curr_stage0[0, idx, 0][1])
                valid_row = cute.elem_less(row_idx, m_key.layout.shape[1])
                if valid_row:
                    cute.copy(
                        gmem_tiled_copy_d,
                        t_key_curr_gmem[None, idx, None],
                        t_key_smem[None, idx, None],
                        pred=d_stage_copy_pred[None, idx, None],
                    )
                else:
                    t_key_smem[None, idx, None].fill(0)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            self._apply_tap_phase_to_staged_keys(
                s_key,
                prefix_state.s_phase,
                s_tap_curr,
                n_tile_start=n_tile_start,
                stage_width=d_stage_width,
                tap_is_full=False,
                out_dtype=out_dtype,
            )

            self._rotate_staged_query_tile_from_prefix(
                s_query,
                prefix_state.s_phase,
                m_tile_start=m_tile_start,
                stage_width=d_stage_width,
                out_dtype=out_dtype,
            )

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_score_curr,
                mma_state.smem_tiled_copy_query,
                mma_state.smem_tiled_copy_key,
                mma_state.t_smem_query,
                mma_state.t_reg_query_view,
                mma_state.t_smem_key,
                mma_state.t_reg_key_view,
                mma_state.t_reg_query,
                mma_state.t_reg_key,
            )

            g_key_prev_stage = cute.local_tile(
                m_key[batch_group_chunk, None, 0, None],
                (self.kv_tile, d_stage_width),
                (n_tile, d_stage_idx),
            )
            g_key_prev_stage = cute.domain_offset((-1, 0), g_key_prev_stage)
            g_key_prev_stage = cute.make_tensor(
                g_key_prev_stage.iterator.align(16), g_key_prev_stage.layout
            )
            t_key_prev_gmem = gmem_thr_copy_d.partition_S(g_key_prev_stage)
            for idx in cutlass.range_constexpr(cute.size(t_key_smem.shape[1])):
                row_idx = cutlass.Int32(t_key_coord_prev_stage0[0, idx, 0][1])
                valid_row = cute.elem_less(
                    cutlass.Int32(-1), row_idx
                ) and cute.elem_less(row_idx, m_key.layout.shape[1])
                if valid_row:
                    cute.copy(
                        gmem_tiled_copy_d,
                        t_key_prev_gmem[None, idx, None],
                        t_key_smem[None, idx, None],
                        pred=d_stage_copy_pred[None, idx, None],
                    )
                else:
                    t_key_smem[None, idx, None].fill(0)

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            if n_tile_start == cutlass.Int32(0):
                self._inject_previous_boundary_key_row(
                    s_key,
                    m_key,
                    m_key_prev0,
                    batch_group_chunk=batch_group_chunk,
                    batch_group=batch_group,
                    chunk_index=chunk_index,
                    d_col_base=d_col_base,
                    stage_width=d_stage_width,
                    out_dtype=out_dtype,
                )

            self._apply_tap_phase_to_staged_keys(
                s_key,
                prefix_state.s_phase,
                prefix_state.s_tap_prev,
                n_tile_start=n_tile_start,
                stage_width=d_stage_width,
                tap_is_full=True,
                out_dtype=out_dtype,
            )

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_score_prev,
                mma_state.smem_tiled_copy_query,
                mma_state.smem_tiled_copy_key,
                mma_state.t_smem_query,
                mma_state.t_reg_query_view,
                mma_state.t_smem_key,
                mma_state.t_reg_key_view,
                mma_state.t_reg_query,
                mma_state.t_reg_key,
            )

    @cute.jit
    def _accumulate_output_from_score_pair(
        self,
        m_grad_output: cute.Tensor,
        coord_grad_output: cute.Tensor,
        coord_output: cute.Tensor,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_thr_copy_p: object,
        t_grad_output_smem: cute.Tensor,
        s_grad_output: cute.Tensor,
        mma_state: SimpleNamespace,
        s_score_alt: cute.Tensor,
        s_score_row0: cute.Tensor,
        s_du_carry_next: cute.Tensor,
        acc_curr_tiles,
        *,
        batch_head_chunk: int,
        m_tile: int,
        n_tile: int,
        n_tile_start: int,
    ):
        t_reg_score_alt = mma_state.thr_mma.make_fragment_A(
            mma_state.thr_mma.partition_A(s_score_alt)
        )
        t_smem_score_alt = mma_state.thr_copy_score.partition_S(s_score_alt)
        t_reg_score_alt_view = mma_state.thr_copy_score.retile(t_reg_score_alt)

        for p_tile_idx in cutlass.range_constexpr(self.num_p_tiles):
            g_grad_output = cute.local_tile(
                m_grad_output[batch_head_chunk, None, 0, None],
                (self.kv_tile, self.p_tile),
                (m_tile, p_tile_idx),
            )
            t_grad_output_gmem = gmem_thr_copy_p.partition_S(g_grad_output)
            grad_output_coord_tile = cute.local_tile(
                coord_grad_output[batch_head_chunk, None, 0, None],
                (self.kv_tile, self.p_tile),
                (m_tile, p_tile_idx),
            )
            t_grad_output_coord = gmem_thr_copy_p.partition_S(grad_output_coord_tile)

            if cutlass.const_expr(self.P == self.P_padded):
                cute.copy(
                    gmem_tiled_copy_p,
                    t_grad_output_gmem,
                    t_grad_output_smem,
                )
            else:
                grad_output_pred = self._make_copy_column_predicate(
                    t_grad_output_smem,
                    t_grad_output_coord,
                    m_grad_output.layout.shape[3],
                )
                self._copy_rows_with_zero_fill(
                    gmem_tiled_copy_p,
                    t_grad_output_gmem,
                    t_grad_output_smem,
                    t_grad_output_coord,
                    grad_output_pred,
                    m_grad_output.layout.shape[1],
                )

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_curr_tiles[p_tile_idx],
                mma_state.smem_tiled_copy_score,
                mma_state.smem_tiled_copy_grad_output_transposed,
                mma_state.t_smem_score,
                mma_state.t_reg_score_view,
                mma_state.t_smem_grad_output_transposed,
                mma_state.t_reg_grad_output_transposed_view,
                mma_state.t_reg_score,
                mma_state.t_reg_grad_output_transposed,
            )
            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_curr_tiles[p_tile_idx],
                mma_state.smem_tiled_copy_score,
                mma_state.smem_tiled_copy_grad_output_transposed,
                t_smem_score_alt,
                t_reg_score_alt_view,
                mma_state.t_smem_grad_output_transposed,
                mma_state.t_reg_grad_output_transposed_view,
                t_reg_score_alt,
                mma_state.t_reg_grad_output_transposed,
            )
            p_base = cutlass.Int32(p_tile_idx * self.p_tile)
            self._accumulate_output_carry_from_score_row(
                s_score_row0,
                s_grad_output,
                s_du_carry_next,
                p_base=p_base,
            )
            cute.arch.barrier()

    # Output helpers
    @cute.jit
    def _make_output_coord_tile(
        self,
        coord_output: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile: int,
        p_tile_idx: int,
    ):
        return cute.local_tile(
            coord_output[batch_head_chunk, None, 0, None],
            (self.kv_tile, self.p_tile),
            (n_tile, p_tile_idx),
        )

    @cute.jit
    def _initialize_output_carry(self, s_du_carry: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        iters_p_padded = (self.P_padded + self.num_threads - 1) // self.num_threads
        for it in range(iters_p_padded):
            p_idx = tidx + cutlass.Int32(it * self.num_threads)
            if p_idx < cutlass.Int32(self.P_padded):
                s_du_carry[p_idx] = cutlass.Float32(0.0)
        cute.arch.barrier()

    @cute.jit
    def _store_output_carry(
        self,
        s_du_carry: cute.Tensor,
        mDU_prev: cute.Tensor,
        *,
        batch_head_chunk: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        iters_p = (self.P + self.num_threads - 1) // self.num_threads
        for it in range(iters_p):
            p_idx = tidx + cutlass.Int32(it * self.num_threads)
            if p_idx < cutlass.Int32(self.P):
                mDU_prev[batch_head_chunk, p_idx] = safe_cast_to_dtype(
                    s_du_carry[p_idx], mDU_prev.element_type
                )
        cute.arch.barrier()

    @cute.jit
    def _stage_output_tile_from_accumulator(
        self,
        acc_output_mn: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        s_output: cute.Tensor,
        *,
        p_base: int,
        row_tile_start: int,
        out_dtype: type[cutlass.Numeric],
    ):
        p_base = cutlass.Int32(p_base)
        row_tile_start = cutlass.Int32(row_tile_start)
        for r in cutlass.range_constexpr(cute.size(acc_output_mn.shape[0])):
            row_idx = cutlass.Int32(t_output_coord_mn[r, 0][1])
            row_local = row_idx - row_tile_start
            for c in cutlass.range_constexpr(cute.size(acc_output_mn.shape[1])):
                col_idx = cutlass.Int32(t_output_coord_mn[0, c][3])
                col_local = col_idx - p_base
                if cute.elem_less(col_idx, cutlass.Int32(self.P)):
                    s_output[row_local, col_local] = safe_cast_to_dtype(
                        acc_output_mn[r, c], out_dtype
                    )

    @cute.jit
    def _accumulate_output_carry_from_score_row(
        self,
        s_score_row0: cute.Tensor,
        s_grad_output: cute.Tensor,
        s_du_carry_next: cute.Tensor,
        *,
        p_base: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        p_base = cutlass.Int32(p_base)
        if tidx < cutlass.Int32(self.p_tile):
            p_idx = p_base + tidx
            if p_idx < cutlass.Int32(self.P):
                carry = cutlass.Float32(0.0)
                for k in cutlass.range_constexpr(self.kv_tile):
                    carry = carry + cutlass.Float32(s_score_row0[k]) * cutlass.Float32(
                        s_grad_output[k, tidx].to(cutlass.Float32)
                    )
                s_du_carry_next[p_idx] = cutlass.Float32(s_du_carry_next[p_idx]) + carry

    @cute.jit
    def _accumulate_output_tile_from_boundary_carry(
        self,
        acc_output_mn: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        s_du_carry: cute.Tensor,
        *,
        row_tile_start: int,
    ):
        row_tile_start = cutlass.Int32(row_tile_start)
        last_row_local = cutlass.Int32(self.kv_tile - 1)
        for r in cutlass.range_constexpr(cute.size(acc_output_mn.shape[0])):
            row_idx = cutlass.Int32(t_output_coord_mn[r, 0][1])
            row_local = row_idx - row_tile_start
            next_row_idx = row_idx + cutlass.Int32(1)
            if row_local == last_row_local and next_row_idx < cutlass.Int32(self.L):
                for c in cutlass.range_constexpr(cute.size(acc_output_mn.shape[1])):
                    col_idx = cutlass.Int32(t_output_coord_mn[0, c][3])
                    if cute.elem_less(col_idx, cutlass.Int32(self.P)):
                        carry = cutlass.Float32(s_du_carry[col_idx])
                        acc_output_mn[r, c] = acc_output_mn[r, c] + carry.to(
                            self.acc_dtype
                        )

    @cute.jit
    def _commit_output_carry(
        self,
        s_du_carry: cute.Tensor,
        s_du_carry_next: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        iters_p_padded = (self.P_padded + self.num_threads - 1) // self.num_threads
        for it in range(iters_p_padded):
            p_idx = tidx + cutlass.Int32(it * self.num_threads)
            if p_idx < cutlass.Int32(self.P_padded):
                s_du_carry[p_idx] = cutlass.Float32(s_du_carry_next[p_idx])
        cute.arch.barrier()

    @cute.jit
    def _store_output_tile(
        self,
        m_output: cute.Tensor,
        coord_output_tile: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        s_output: cute.Tensor,
        gmem_tiled_store_p: cute.TiledCopy,
        gmem_thr_store_p: object,
        t_acc_smem_output: cute.Tensor,
        acc_output: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile: int,
        p_tile_idx: int,
        p_base: int,
        row_tile_start: int,
        out_dtype: cutlass.Constexpr,
    ):
        if cutlass.const_expr(self.P == self.P_padded):
            reg_output = cute.make_fragment_like(t_acc_smem_output, out_dtype)
            acc_output_vals = acc_output.load()
            for i in cutlass.range_constexpr(cute.size(reg_output)):
                reg_output[i] = safe_cast_to_dtype(acc_output_vals[i], out_dtype)
            cute.autovec_copy(reg_output, t_acc_smem_output)
        else:
            acc_output_mn = self._make_accumulator_mn_view(acc_output)
            self._stage_output_tile_from_accumulator(
                acc_output_mn,
                t_output_coord_mn,
                s_output,
                p_base=p_base,
                row_tile_start=row_tile_start,
                out_dtype=out_dtype,
            )
        cute.arch.barrier()

        g_output = cute.local_tile(
            m_output[batch_head_chunk, None, 0, None],
            (self.kv_tile, self.p_tile),
            (n_tile, p_tile_idx),
        )
        t_smem_output = gmem_thr_store_p.partition_S(s_output)
        t_gmem_output = gmem_thr_store_p.partition_D(g_output)
        if cutlass.const_expr(self.P == self.P_padded):
            cute.copy(gmem_tiled_store_p, t_smem_output, t_gmem_output)
            return

        reg_output = cute.make_rmem_tensor_like(t_gmem_output, out_dtype)
        cute.copy(gmem_tiled_store_p, t_smem_output, reg_output)
        t_output_coord_store = gmem_thr_store_p.partition_D(coord_output_tile)
        output_pred = self._make_copy_column_predicate(
            t_gmem_output,
            t_output_coord_store,
            m_output.layout.shape[3],
        )
        self._copy_rows_if_valid(
            gmem_tiled_store_p,
            reg_output,
            t_gmem_output,
            t_output_coord_store,
            output_pred,
            m_output.layout.shape[1],
        )

    @cute.jit
    def _store_output_tiles_and_update_carry(
        self,
        m_output: cute.Tensor,
        coord_output: cute.Tensor,
        s_output: cute.Tensor,
        s_du_carry: cute.Tensor,
        s_du_carry_next: cute.Tensor,
        gmem_tiled_store_p: cute.TiledCopy,
        mma_state: SimpleNamespace,
        acc_curr_tiles,
        *,
        batch_head_chunk: int,
        n_tile: int,
        n_tile_start: int,
        out_dtype: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gmem_thr_store_p = gmem_tiled_store_p.get_slice(tidx)

        for p_tile_idx in cutlass.range_constexpr(self.num_p_tiles):
            p_base = cutlass.Int32(p_tile_idx * self.p_tile)
            coord_output_tile = self._make_output_coord_tile(
                coord_output,
                batch_head_chunk=batch_head_chunk,
                n_tile=n_tile,
                p_tile_idx=p_tile_idx,
            )
            t_output_coord = mma_state.thr_mma.partition_C(coord_output_tile)
            t_output_coord_mn = self._make_accumulator_mn_view(t_output_coord)

            acc_curr_mn = self._make_accumulator_mn_view(acc_curr_tiles[p_tile_idx])
            self._accumulate_output_tile_from_boundary_carry(
                acc_curr_mn,
                t_output_coord_mn,
                s_du_carry,
                row_tile_start=n_tile_start,
            )

            self._store_output_tile(
                m_output,
                coord_output_tile,
                t_output_coord_mn,
                s_output,
                gmem_tiled_store_p,
                gmem_thr_store_p,
                mma_state.t_acc_smem_output,
                acc_curr_tiles[p_tile_idx],
                batch_head_chunk=batch_head_chunk,
                n_tile=n_tile,
                p_tile_idx=p_tile_idx,
                p_base=p_base,
                row_tile_start=n_tile_start,
                out_dtype=out_dtype,
            )
            if cutlass.const_expr(self.P_padded > self.p_tile):
                if cutlass.const_expr(p_tile_idx + 1 < self.num_p_tiles):
                    cute.arch.barrier()
        self._commit_output_carry(s_du_carry, s_du_carry_next)

    @cute.jit
    def _accumulate_output_tiles_for_pass_pair(
        self,
        m_query: cute.Tensor,
        m_key: cute.Tensor,
        m_tap: cute.Tensor,
        m_key_prev0: cute.Tensor,
        m_grad_output: cute.Tensor,
        coord_query: cute.Tensor,
        coord_key: cute.Tensor,
        coord_score: cute.Tensor,
        coord_grad_output: cute.Tensor,
        coord_output: cute.Tensor,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_thr_copy_d: object,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_thr_copy_p: object,
        t_query_smem: cute.Tensor,
        t_key_smem: cute.Tensor,
        t_grad_output_smem: cute.Tensor,
        s_grad_output: cute.Tensor,
        d_stage_copy_pred: cute.Tensor,
        s_query: cute.Tensor,
        s_key: cute.Tensor,
        s_score: cute.Tensor,
        s_score_alt: cute.Tensor,
        s_score_row0: cute.Tensor,
        s_tap_curr: cute.Tensor,
        prefix_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        acc_shape_score,
        acc_output_curr_tiles,
        s_du_carry_next: cute.Tensor,
        *,
        batch_head_chunk: int,
        batch_group_chunk: int,
        batch_group: int,
        chunk_index: int,
        n_tile: int,
        n_tile_start: int,
        m_tiles: int,
        out_dtype: cutlass.Constexpr,
    ):
        for mi in cutlass.range_constexpr(m_tiles):
            m_tile = n_tile + mi
            m_tile_start = m_tile * self.kv_tile
            self._load_current_tap_tile(
                m_tap,
                s_tap_curr,
                batch_head_chunk=batch_head_chunk,
                n_tile_start=n_tile_start,
            )
            cute.arch.barrier()

            score_coord_tile = self._make_score_coord_tile(
                coord_score,
                batch_group_chunk=batch_group_chunk,
                m_tile=m_tile,
                n_tile=n_tile,
            )
            t_score_coord = mma_state.thr_mma.partition_C(score_coord_tile)
            t_score_coord_mn = self._make_accumulator_mn_view(t_score_coord)

            acc_score_curr = cute.make_rmem_tensor(acc_shape_score, self.acc_dtype)
            acc_score_curr.fill(0.0)
            acc_score_prev = cute.make_rmem_tensor(acc_shape_score, self.acc_dtype)
            acc_score_prev.fill(0.0)
            self._accumulate_score_block_pair_reuse_query(
                m_query,
                m_key,
                m_key_prev0,
                coord_query,
                coord_key,
                gmem_tiled_copy_d,
                gmem_thr_copy_d,
                t_query_smem,
                t_key_smem,
                d_stage_copy_pred,
                s_query,
                s_key,
                s_tap_curr,
                prefix_state,
                mma_state,
                acc_score_curr,
                acc_score_prev,
                batch_group_chunk=batch_group_chunk,
                batch_group=batch_group,
                chunk_index=chunk_index,
                n_tile=n_tile,
                m_tile=m_tile,
                n_tile_start=n_tile_start,
                m_tile_start=m_tile_start,
                out_dtype=out_dtype,
            )
            self._apply_score_scales_and_mask(
                prefix_state,
                acc_score_curr,
                t_score_coord_mn,
                s_score,
                n_tile_start=n_tile_start,
                m_tile_start=m_tile_start,
                out_dtype=out_dtype,
            )
            self._apply_score_scales_and_mask_shifted_prev(
                prefix_state,
                acc_score_prev,
                t_score_coord_mn,
                s_score_alt,
                s_score_row0,
                n_tile_start=n_tile_start,
                m_tile_start=m_tile_start,
                out_dtype=out_dtype,
            )

            self._accumulate_output_from_score_pair(
                m_grad_output,
                coord_grad_output,
                coord_output,
                gmem_tiled_copy_p,
                gmem_thr_copy_p,
                t_grad_output_smem,
                s_grad_output,
                mma_state,
                s_score_alt,
                s_score_row0,
                s_du_carry_next,
                acc_output_curr_tiles,
                batch_head_chunk=batch_head_chunk,
                m_tile=m_tile,
                n_tile=n_tile,
                n_tile_start=n_tile_start,
            )

    # Kernel setup helpers
    def _make_prefix_state(
        self,
        *,
        batch_head_chunk: int,
        m_transition: cute.Tensor,
        m_tap: cute.Tensor,
        s_phase: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_row_scale: cute.Tensor,
        s_inv_row_scale: cute.Tensor,
        warp_log_total: cute.Tensor,
        warp_log_offset: cute.Tensor,
        warp_phase_total: cute.Tensor,
        warp_phase_offset: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_transition=m_transition,
            m_tap=m_tap,
            s_phase=s_phase,
            s_tap_prev=s_tap_prev,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            warp_log_total=warp_log_total,
            warp_log_offset=warp_log_offset,
            warp_phase_total=warp_phase_total,
            warp_phase_offset=warp_phase_offset,
        )

    def _make_coordinate_bundle(
        self,
        m_query: cute.Tensor,
        m_key: cute.Tensor,
        m_grad_output: cute.Tensor,
        m_output: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            query=cute.make_identity_tensor(m_query.layout.shape),
            key=cute.make_identity_tensor(m_key.layout.shape),
            grad_output=cute.make_identity_tensor(m_grad_output.layout.shape),
            score=cute.make_identity_tensor(
                (m_key.shape[0], self.L, m_key.shape[2], self.L)
            ),
            output=cute.make_identity_tensor(m_output.layout.shape),
        )

    def _make_d_stage_copy_predicate(
        self,
        coord_query: cute.Tensor,
        gmem_thr_copy_d: object,
        t_query_smem: cute.Tensor,
        *,
        batch_group_chunk: int,
        query_col_limit: int,
    ):
        query_coord_tile = cute.local_tile(
            coord_query[batch_group_chunk, None, 0, None],
            (self.kv_tile, self._d_stage_size()),
            (0, 0),
        )
        t_query_coord = gmem_thr_copy_d.partition_S(query_coord_tile)
        return self._make_copy_column_predicate(
            t_query_smem,
            t_query_coord,
            query_col_limit,
        )

    def _make_mma_state(
        self,
        *,
        tidx: int,
        tiled_mma: cute.TiledMma,
        in_dtype: type[cutlass.Numeric],
        s_query: cute.Tensor,
        s_key: cute.Tensor,
        s_score: cute.Tensor,
        s_grad_output: cute.Tensor,
        s_output: cute.Tensor,
    ) -> SimpleNamespace:
        thr_mma = tiled_mma.get_slice(tidx)
        t_reg_query = thr_mma.make_fragment_A(thr_mma.partition_A(s_query))
        t_reg_key = thr_mma.make_fragment_B(thr_mma.partition_B(s_key))
        t_reg_score = thr_mma.make_fragment_A(thr_mma.partition_A(s_score))
        s_grad_output_transposed = cute.composition(
            s_grad_output,
            cute.make_layout((self.p_tile, self.kv_tile), stride=(self.kv_tile, 1)),
        )
        t_reg_grad_output_transposed = thr_mma.make_fragment_B(
            thr_mma.partition_B(s_grad_output_transposed)
        )

        smem_copy_atom_query = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            in_dtype,
        )
        smem_copy_atom_key = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            in_dtype,
        )
        smem_copy_atom_grad_output = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            in_dtype,
        )
        smem_tiled_copy_query = cute.make_tiled_copy_A(smem_copy_atom_query, tiled_mma)
        smem_tiled_copy_key = cute.make_tiled_copy_B(smem_copy_atom_key, tiled_mma)
        smem_tiled_copy_score = cute.make_tiled_copy_A(smem_copy_atom_query, tiled_mma)
        smem_tiled_copy_grad_output_transposed = cute.make_tiled_copy_B(
            smem_copy_atom_grad_output, tiled_mma
        )

        smem_thr_copy_query = smem_tiled_copy_query.get_slice(tidx)
        smem_thr_copy_key = smem_tiled_copy_key.get_slice(tidx)
        smem_thr_copy_score = smem_tiled_copy_score.get_slice(tidx)
        smem_thr_copy_grad_output = smem_tiled_copy_grad_output_transposed.get_slice(
            tidx
        )

        return SimpleNamespace(
            tiled_mma=tiled_mma,
            thr_mma=thr_mma,
            smem_tiled_copy_query=smem_tiled_copy_query,
            smem_tiled_copy_key=smem_tiled_copy_key,
            smem_tiled_copy_score=smem_tiled_copy_score,
            smem_tiled_copy_grad_output_transposed=smem_tiled_copy_grad_output_transposed,
            thr_copy_score=smem_thr_copy_score,
            t_smem_query=smem_thr_copy_query.partition_S(s_query),
            t_reg_query_view=smem_thr_copy_query.retile(t_reg_query),
            t_smem_key=smem_thr_copy_key.partition_S(s_key),
            t_reg_key_view=smem_thr_copy_key.retile(t_reg_key),
            t_smem_score=smem_thr_copy_score.partition_S(s_score),
            t_reg_score_view=smem_thr_copy_score.retile(t_reg_score),
            t_smem_grad_output_transposed=smem_thr_copy_grad_output.partition_S(
                s_grad_output_transposed
            ),
            t_reg_grad_output_transposed_view=smem_thr_copy_grad_output.retile(
                t_reg_grad_output_transposed
            ),
            t_reg_query=t_reg_query,
            t_reg_key=t_reg_key,
            t_reg_score=t_reg_score,
            t_reg_grad_output_transposed=t_reg_grad_output_transposed,
            t_acc_smem_output=thr_mma.partition_C(s_output),
        )

    def _make_output_accumulator_tiles(self, acc_shape_output):
        acc_output_tiles = []
        for _ in range(self.num_p_tiles):
            acc_output = cute.make_rmem_tensor(acc_shape_output, self.acc_dtype)
            acc_output.fill(0.0)
            acc_output_tiles.append(acc_output)
        return acc_output_tiles

    @cute.jit
    def _validate_main_operands(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
    ):
        if cutlass.const_expr(
            not (
                mU.element_type
                == mB.element_type
                == mC.element_type
                == mDOut.element_type
                == mU_prev0.element_type
                == mB_prev0.element_type
                == mDU.element_type
                == mDB.element_type
                == mDU_prev.element_type
                == mDB_prev.element_type
            )
        ):
            raise TypeError(
                "U/B/C/dOut/prev tensors/DU scratch tensors must share dtype."
            )
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("Tensor-core path supports only Float16/BFloat16 inputs.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(mDLogPrefix.element_type != cutlass.Float32):
            raise TypeError("dlogprefix must be Float32.")
        if cutlass.const_expr(
            mDMprev.element_type != cutlass.Float32
            or mDMcurr.element_type != cutlass.Float32
        ):
            raise TypeError("dM buffers must be Float32.")

        if cutlass.const_expr(
            mU.shape[1] != self.L or mB.shape[1] != self.L or mC.shape[1] != self.L
        ):
            raise ValueError("U must be (BHC, L, 1, P) and B/C must be (BGC, L, 1, D).")
        if cutlass.const_expr(mU.shape[2] != 1 or mB.shape[2] != 1 or mC.shape[2] != 1):
            raise ValueError("U must be (BHC, L, 1, P) and B/C must be (BGC, L, 1, D).")
        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be (BHC, L, 2).")
        if cutlass.const_expr(
            mK.shape[1] != self.L or mK.shape[2] != 2 or mK.shape[3] != 2
        ):
            raise ValueError("K must be (BHC, L, 2, 2).")
        if cutlass.const_expr(mDOut.shape[1] != self.L or mDOut.shape[2] != 1):
            raise ValueError("dOut must be (BHC, L, 1, P).")

        if cutlass.const_expr(mU.layout.shape[3] != self.P):
            raise ValueError("U must be (BHC, L, 1, P).")
        if cutlass.const_expr(mDOut.layout.shape[3] != self.P):
            raise ValueError("dOut must be (BHC, L, 1, P).")
        if cutlass.const_expr(
            mB.layout.shape[3] != self.D or mC.layout.shape[3] != self.D
        ):
            raise ValueError("B/C must be (BGC, L, 1, D).")
        if cutlass.const_expr(
            mDU.shape[1] != self.L or mDU.shape[2] != 1 or mDU.layout.shape[3] != self.P
        ):
            raise ValueError("dU must be (BHC, L, 1, P).")
        if cutlass.const_expr(
            mDB.shape[1] != self.L or mDB.shape[2] != 1 or mDB.layout.shape[3] != self.D
        ):
            raise ValueError("dB must be (BGC, L, 1, D).")
        if cutlass.const_expr(mDU_prev.shape[1] != self.P):
            raise ValueError("dU_prev must be (BHC, P).")
        if cutlass.const_expr(mDB_prev.shape[1] != self.D):
            raise ValueError("dB_prev must be (BHC, D).")
        if cutlass.const_expr(mDLogPrefix.shape[1] != self.L):
            raise ValueError("dlogprefix must be (BHC, L).")
        if cutlass.const_expr(
            mDMprev.shape[1] != self.L
            or mDMprev.shape[2] != 2
            or mDMcurr.shape[1] != self.L
            or mDMcurr.shape[2] != 2
        ):
            raise ValueError("dM buffers must be (BHC, L, 2).")
        if cutlass.const_expr(mU_prev0.shape[1] != self.P):
            raise ValueError("U_prev0 must be (BH, P).")
        if cutlass.const_expr(mB_prev0.shape[1] != self.D):
            raise ValueError("B_prev0 must be (BG, D).")

    # Host launch
    def _launch_main_kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        kernel_bundle = self._make_kernel_bundle(mU.element_type)
        layouts = kernel_bundle.layouts
        copies = kernel_bundle.copies

        launch_kwargs = {
            "grid": (1, 1, cute.size(mB.shape[0])),
            "block": [self.num_threads, 1, 1],
            "smem": kernel_bundle.smem_size,
        }
        if stream is not None:
            launch_kwargs["stream"] = stream

        self.kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
            layouts.query_layout,
            layouts.grad_output_layout,
            layouts.key_layout,
            layouts.output_layout,
            layouts.prev_output_layout,
            layouts.score_layout,
            layouts.phase_layout,
            layouts.tap_prev_layout,
            layouts.tap_curr_layout,
            layouts.row_scale_layout,
            layouts.inv_row_scale_layout,
            layouts.carry_layout,
            layouts.warp_log_layout,
            layouts.warp_phase_layout,
            copies.gmem_tiled_copy_d,
            copies.gmem_tiled_copy_p,
            copies.gmem_tiled_store_p,
            kernel_bundle.tiled_mma,
            kernel_bundle.shared_storage_cls,
        ).launch(**launch_kwargs)

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
    ):
        self._validate_main_operands(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
        )
        self._launch_main_kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
        )

    @cute.jit
    def call_on_stream(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._validate_main_operands(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
        )
        self._launch_main_kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mDOut,
            mU_prev0,
            mB_prev0,
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
            stream=stream,
        )

    @cute.kernel(preprocess=True)
    def kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mDOut: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        query_layout: cute.ComposedLayout,
        grad_output_layout: cute.ComposedLayout,
        key_layout: cute.ComposedLayout,
        output_layout: cute.ComposedLayout,
        prev_output_layout: cute.Layout,
        score_layout: cute.ComposedLayout,
        phase_layout: cute.Layout,
        tap_prev_layout: cute.Layout,
        tap_curr_layout: cute.Layout,
        row_scale_layout: cute.Layout,
        inv_row_scale_layout: cute.Layout,
        carry_layout: cute.Layout,
        warp_log_layout: cute.Layout,
        warp_phase_layout: cute.Layout,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_tiled_store_p: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_head_chunk = cute.arch.block_idx()

        in_dtype = mU.element_type
        n_tiles = self.num_kv_tiles
        batch_head_count = mU_prev0.shape[0]
        batch_head_chunk_count = mU.shape[0]
        n_chunks = batch_head_chunk_count // batch_head_count
        batch_head = batch_head_chunk // n_chunks
        batch_group = self._batch_group(batch_head)
        batch_group_chunk = self._batch_group_chunk(batch_head_chunk, n_chunks)
        chunk_index = batch_head_chunk - batch_head * n_chunks

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_query = storage.query_stage.get_tensor(query_layout)
        s_grad_output = storage.grad_output_stage.get_tensor(grad_output_layout)
        s_key = storage.key_stage.get_tensor(key_layout)
        s_output = storage.output_stage.get_tensor(output_layout)
        s_score_row0 = storage.prev_output_tile.get_tensor(prev_output_layout)
        s_score = storage.score_block.get_tensor(score_layout)
        s_score_alt = storage.score_block_alt.get_tensor(score_layout)
        s_phase = storage.phase_full.get_tensor(phase_layout)
        s_tap_prev = storage.tap_prev_full.get_tensor(tap_prev_layout)
        s_tap_curr = storage.tap_curr_tile.get_tensor(tap_curr_layout)
        s_row_scale = storage.row_scale_full.get_tensor(row_scale_layout)
        s_inv_row_scale = storage.inv_row_scale_full.get_tensor(inv_row_scale_layout)
        s_du_carry = storage.du_carry.get_tensor(carry_layout)
        s_du_carry_next = storage.du_carry_next.get_tensor(carry_layout)
        warp_log_total = storage.warp_log_total.get_tensor(warp_log_layout)
        warp_log_offset = storage.warp_log_offset.get_tensor(warp_log_layout)
        warp_phase_total = storage.warp_phase_total.get_tensor(warp_phase_layout)
        warp_phase_offset = storage.warp_phase_offset.get_tensor(warp_phase_layout)

        prefix_state = self._make_prefix_state(
            batch_head_chunk=batch_head_chunk,
            m_transition=mM,
            m_tap=mK,
            s_phase=s_phase,
            s_tap_prev=s_tap_prev,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            warp_log_total=warp_log_total,
            warp_log_offset=warp_log_offset,
            warp_phase_total=warp_phase_total,
            warp_phase_offset=warp_phase_offset,
        )
        self._compute_phase_prefix_metadata(prefix_state)

        gmem_thr_copy_d = gmem_tiled_copy_d.get_slice(tidx)
        gmem_thr_copy_p = gmem_tiled_copy_p.get_slice(tidx)
        t_query_smem = gmem_thr_copy_d.partition_D(s_query)
        t_key_smem = gmem_thr_copy_d.partition_D(s_key)
        t_grad_output_smem = gmem_thr_copy_p.partition_D(s_grad_output)

        coord_bundle = self._make_coordinate_bundle(
            mC,
            mB,
            mDOut,
            mDU,
        )
        d_stage_copy_pred = self._make_d_stage_copy_predicate(
            coord_bundle.query,
            gmem_thr_copy_d,
            t_query_smem,
            batch_group_chunk=batch_group_chunk,
            query_col_limit=mC.layout.shape[3],
        )

        mma_state = self._make_mma_state(
            tidx=tidx,
            tiled_mma=tiled_mma,
            in_dtype=in_dtype,
            s_query=s_query,
            s_key=s_key,
            s_score=s_score,
            s_grad_output=s_grad_output,
            s_output=s_output,
        )

        acc_shape_score = mma_state.thr_mma.partition_shape_C(
            (self.kv_tile, self.kv_tile)
        )
        acc_shape_output = mma_state.thr_mma.partition_shape_C(
            (self.kv_tile, self.p_tile)
        )

        self._initialize_output_carry(s_du_carry)

        for n_tile_rev in cutlass.range_constexpr(n_tiles):
            n_tile = (n_tiles - 1) - n_tile_rev
            n_tile_start = n_tile * self.kv_tile
            m_tiles = n_tiles - n_tile

            acc_output_curr_tiles = self._make_output_accumulator_tiles(
                acc_shape_output
            )
            self._initialize_output_carry(s_du_carry_next)

            self._accumulate_output_tiles_for_pass_pair(
                mC,
                mB,
                mK,
                mB_prev0,
                mDOut,
                coord_bundle.query,
                coord_bundle.key,
                coord_bundle.score,
                coord_bundle.grad_output,
                coord_bundle.output,
                gmem_tiled_copy_d,
                gmem_thr_copy_d,
                gmem_tiled_copy_p,
                gmem_thr_copy_p,
                t_query_smem,
                t_key_smem,
                t_grad_output_smem,
                s_grad_output,
                d_stage_copy_pred,
                s_query,
                s_key,
                s_score,
                s_score_alt,
                s_score_row0,
                s_tap_curr,
                prefix_state,
                mma_state,
                acc_shape_score,
                acc_output_curr_tiles,
                s_du_carry_next,
                batch_head_chunk=batch_head_chunk,
                batch_group_chunk=batch_group_chunk,
                batch_group=batch_group,
                chunk_index=chunk_index,
                n_tile=n_tile,
                n_tile_start=n_tile_start,
                m_tiles=m_tiles,
                out_dtype=in_dtype,
            )

            self._store_output_tiles_and_update_carry(
                mDU,
                coord_bundle.output,
                s_output,
                s_du_carry,
                s_du_carry_next,
                gmem_tiled_store_p,
                mma_state,
                acc_output_curr_tiles,
                batch_head_chunk=batch_head_chunk,
                n_tile=n_tile,
                n_tile_start=n_tile_start,
                out_dtype=mDU.element_type,
            )

        self._store_output_carry(
            s_du_carry,
            mDU_prev,
            batch_head_chunk=batch_head_chunk,
        )

    # Kernel entrypoint


__all__ = ["ChunkScanBwdDUAmpere"]
