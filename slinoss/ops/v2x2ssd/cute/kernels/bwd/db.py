"""CuTe backward ``dB`` kernels for the ``v2x2ssd`` backward pipeline.

``BwdDBAmpere`` computes the main backward key-gradient contribution.
``BwdDBIncrementAccumulatorAmpere`` accumulates the increment-side key-gradient
contribution into the same public ``dB`` tile. Both kernels reconstruct the
needed prefix metadata from ``M``, sweep tiles in reverse causal order, apply
the packed-complex taps from ``K``, and write ``dB`` plus the cross-chunk
boundary carry ``dB_prev``.

Tensor contracts:

- ``U``: ``(BHC, L, 1, P)`` fp16/bf16 value input
- ``B``: ``(BHC, L, 1, D)`` fp16/bf16 key input with packed complex pairs
- ``C``: ``(BHC, L, 1, D)`` fp16/bf16 query input with packed complex pairs
- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for previous/current
  backward passes
- ``dOut``: ``(BHC, L, 1, P)`` fp16/bf16 upstream gradient
- ``U_prev0``: ``(BH, P)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(BH, D)`` fp16/bf16 chunk-0 boundary key row
- ``dB``: ``(BHC, L, 1, D)`` fp16/bf16 output key gradient
- ``dB_prev``: ``(BHC, D)`` fp16/bf16 boundary carry for the previous chunk

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be even
and conceptually corresponds to ``2 * N``.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import torch
from cuda.bindings import driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    _default_async_copy_bits,
    _default_tc_k_tile,
    _next_pow2,
    apply_complex_tap_adjoint,
    clamp_nonpositive_prefix_log,
    complex_mul,
    conj_mul_phase,
    safe_cast_to_dtype,
)


@dataclass(frozen=True)
class BwdDBLayoutBundle:
    query_layout: object
    key_layout: object
    grad_output_layout: object
    key_grad_layout: object
    value_layout: object
    score_layout: object


@dataclass(frozen=True)
class BwdDBCopyBundle:
    gmem_tiled_copy_d: object
    gmem_tiled_copy_p: object
    gmem_tiled_store_d: object


@dataclass(frozen=True)
class BwdDBKernelBundle:
    layouts: BwdDBLayoutBundle
    copies: BwdDBCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int


@dataclass(frozen=True)
class BwdDBSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class BwdDBAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dB`` contract.

    This kernel owns the scan-side key-gradient path. It rebuilds the prefix
    metadata from raw packed ``M``, forms the current and shifted score tiles
    from ``dOut @ U``, contracts those scores against the rotated query tiles,
    applies the two complex taps from ``K``, and writes ``dB`` plus the reverse
    chunk-boundary carry ``dB_prev``.
    """

    _SUPPORT_INFO_CACHE: ClassVar[dict[tuple[object, ...], BwdDBSupportInfo]] = {}

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
        if self.L % self.kv_tile != 0:
            raise ValueError("chunk_size must be a multiple of 32.")
        self.num_threads = int(num_threads)
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        self.warp_layout_mnk = (2, 2, 1)
        expected_threads = 32 * self.warp_layout_mnk[0] * self.warp_layout_mnk[1]
        if self.num_threads != expected_threads:
            raise ValueError(
                f"num_threads must be {expected_threads} for kv_tile={self.kv_tile} "
                f"(warp_layout_mnk={self.warp_layout_mnk})."
            )
        self.atom_layout_mnk = self.warp_layout_mnk
        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        self.mma_inst_shape = (16, 8, 16)

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
    def num_warps(self) -> int:
        return self.num_threads // 32

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

    def _d_stage_size(self) -> int:
        """Shared-memory D slice width used by the backward DB kernel."""
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

    def _smem_block_size_d(self) -> int:
        return 64 if self._d_stage_size() % 64 == 0 else 32

    @staticmethod
    def _swizzle_bits(smem_block_size: int) -> int:
        return 3 if smem_block_size == 64 else 2

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
        d_stage_width = self._d_stage_size()
        p_tile = 32
        d_padded = self.D_padded
        return self._struct_size_bytes(
            [
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (self.L * 2 * 4, 16),
                (self.L * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (d_padded * 4, 8),
                (d_stage_width * in_bytes, 16),
                (self.L * 4, 4),
                (self.L * 4, 4),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (self.num_warps * 4, 4),
                (self.num_warps * 4, 4),
                (self.num_warps * 2 * 4, 16),
                (self.num_warps * 2 * 4, 16),
            ]
        )

    def support_info(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> BwdDBSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return BwdDBSupportInfo(0, 1)

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

        info = BwdDBSupportInfo(
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

    def _make_layout_bundle(self) -> BwdDBLayoutBundle:
        kv_tile = self.kv_tile
        d_stage_width = self._d_stage_size()
        p_tile = 32

        smem_block_size_d = self._smem_block_size_d()
        swizzle_bits_d = self._swizzle_bits(smem_block_size_d)
        d_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_d, 3, 3),
            0,
            cute.make_layout((8, smem_block_size_d), stride=(smem_block_size_d, 1)),
        )
        query_layout = cute.tile_to_shape(
            d_layout_atom, (kv_tile, d_stage_width), (0, 1)
        )
        key_layout = cute.tile_to_shape(d_layout_atom, (kv_tile, d_stage_width), (0, 1))
        key_grad_layout = cute.tile_to_shape(
            d_layout_atom, (kv_tile, d_stage_width), (0, 1)
        )

        p_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, p_tile), stride=(p_tile, 1)),
        )
        grad_output_layout = cute.tile_to_shape(
            p_layout_atom, (kv_tile, p_tile), (0, 1)
        )
        value_layout = cute.tile_to_shape(p_layout_atom, (kv_tile, p_tile), (0, 1))

        score_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, kv_tile), stride=(kv_tile, 1)),
        )
        score_layout = cute.tile_to_shape(score_layout_atom, (kv_tile, kv_tile), (0, 1))

        return BwdDBLayoutBundle(
            query_layout=query_layout,
            key_layout=key_layout,
            grad_output_layout=grad_output_layout,
            key_grad_layout=key_grad_layout,
            value_layout=value_layout,
            score_layout=score_layout,
        )

    def _make_copy_bundle(self, in_dtype: type[cutlass.Numeric]) -> BwdDBCopyBundle:
        universal_copy_bits = 128
        elems_per_copy = universal_copy_bits // in_dtype.width
        p_tile = 32
        smem_block_size_d = self._smem_block_size_d()
        t_p_shape_dim_1 = p_tile // elems_per_copy
        t_p_layout = cute.make_layout(
            (self.num_threads // t_p_shape_dim_1, t_p_shape_dim_1),
            stride=(t_p_shape_dim_1, 1),
        )
        t_d_shape_dim_1 = smem_block_size_d // elems_per_copy
        t_d_layout = cute.make_layout(
            (self.num_threads // t_d_shape_dim_1, t_d_shape_dim_1),
            stride=(t_d_shape_dim_1, 1),
        )
        vector_layout = cute.make_layout((1, elems_per_copy))

        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        return BwdDBCopyBundle(
            gmem_tiled_copy_d=cute.make_tiled_copy_tv(
                atom_async_copy_in, t_d_layout, vector_layout
            ),
            gmem_tiled_copy_p=cute.make_tiled_copy_tv(
                atom_async_copy_in, t_p_layout, vector_layout
            ),
            gmem_tiled_store_d=cute.make_tiled_copy_tv(
                atom_universal_copy_out, t_d_layout, vector_layout
            ),
        )

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        op = cute.nvgpu.warp.MmaF16BF16Op(in_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        return cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: BwdDBLayoutBundle,
    ):
        phase_layout = cute.make_layout((self.L, 2), stride=(2, 1))
        tap_curr_layout = cute.make_layout((self.kv_tile, 2), stride=(2, 1))
        carry_layout = cute.make_layout((self.D_padded,), stride=(1,))
        row_layout = cute.make_layout((self.L,), stride=(1,))

        @cute.struct
        class SharedStorage:
            query_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)], 16
            ]
            key_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.key_layout)], 16
            ]
            grad_output_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.grad_output_layout)],
                16,
            ]
            key_grad_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.key_grad_layout)], 16
            ]
            value_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.value_layout)], 16
            ]
            score_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)], 16
            ]
            score_diag_curr: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)], 16
            ]
            phase_full: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 16
            ]
            tap_prev_full: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 16
            ]
            tap_curr_tile: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ]
            db_carry: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(carry_layout)], 8
            ]
            key_boundary: cute.struct.Align[
                cute.struct.MemRange[in_dtype, self._d_stage_size()], 16
            ]
            row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ]
            inv_row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ]
            dm_curr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ]
            dm_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ]
            warp_log_total: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 4
            ]
            warp_log_offset: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 4
            ]
            warp_phase_total: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
            ]
            warp_phase_offset: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
            ]

        return SharedStorage

    def _make_kernel_bundle(self, in_dtype: type[cutlass.Numeric]) -> BwdDBKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        return BwdDBKernelBundle(
            layouts=layouts,
            copies=self._make_copy_bundle(in_dtype),
            tiled_mma=self._make_tiled_mma(in_dtype),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

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

    @cute.jit
    def _copy_rows_in_range_with_zero_fill(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        t_copy_src: cute.Tensor,
        t_copy_dst: cute.Tensor,
        t_coord: cute.Tensor,
        copy_pred: cute.Tensor,
        *,
        row_lower_bound: int,
        row_upper_bound: int,
    ):
        row_lower_bound = cutlass.Int32(row_lower_bound)
        row_upper_bound = cutlass.Int32(row_upper_bound)
        for idx in cutlass.range_constexpr(cute.size(t_copy_dst.shape[1])):
            row_coord = cutlass.Int32(t_coord[0, idx, 0][1])
            if row_coord >= row_lower_bound and cute.elem_less(
                row_coord, row_upper_bound
            ):
                cute.copy(
                    gmem_tiled_copy,
                    t_copy_src[None, idx, None],
                    t_copy_dst[None, idx, None],
                    pred=copy_pred[None, idx, None],
                )
            else:
                t_copy_dst[None, idx, None].fill(0)

    @cute.jit
    def _stage_query_tile_from_gmem(
        self,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_thr_copy_d,
        m_query: cute.Tensor,
        coord_query: cute.Tensor,
        t_query_smem: cute.Tensor,
        *,
        batch_group_chunk: int,
        m_tile_index: int,
        d_stage_index: int,
        stage_width: int,
    ):
        g_query_stage = cute.local_tile(
            m_query[batch_group_chunk, None, 0, None],
            (self.kv_tile, stage_width),
            (m_tile_index, d_stage_index),
        )
        t_query_gmem = gmem_thr_copy_d.partition_S(g_query_stage)
        c_query_stage = cute.local_tile(
            coord_query[batch_group_chunk, None, 0, None],
            (self.kv_tile, stage_width),
            (m_tile_index, d_stage_index),
        )
        t_query_coord = gmem_thr_copy_d.partition_S(c_query_stage)
        t_query_pred = self._make_copy_column_predicate(
            t_query_smem, t_query_coord, m_query.layout.shape[3]
        )
        self._copy_rows_with_zero_fill(
            gmem_tiled_copy_d,
            t_query_gmem,
            t_query_smem,
            t_query_coord,
            t_query_pred,
            m_query.layout.shape[1],
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        pipeline.sync()

    @cute.jit
    def _stage_key_boundary_from_gmem(
        self,
        s_key_boundary: cute.Tensor,
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
        for it in cutlass.range_constexpr(
            (stage_width + self.num_threads - 1) // self.num_threads
        ):
            d_local = tidx + cutlass.Int32(it * self.num_threads)
            boundary_value = cutlass.Float32(0.0).to(out_dtype)
            if d_local < cutlass.Int32(stage_width):
                d = cutlass.Int32(d_col_base) + d_local
                if d < cutlass.Int32(self.D):
                    if chunk_index == cutlass.Int32(0):
                        boundary_value = m_key_prev0[batch_group, d]
                    else:
                        boundary_value = m_key[
                            batch_group_chunk - cutlass.Int32(1),
                            cutlass.Int32(self.L - 1),
                            0,
                            d,
                        ]
                s_key_boundary[d_local] = boundary_value
        pipeline.sync()

    @cute.jit
    def _stage_p_tile_from_gmem(
        self,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_thr_copy_p,
        source_tensor: cute.Tensor,
        coord_tensor: cute.Tensor,
        t_stage_smem: cute.Tensor,
        *,
        batch_head_chunk: int,
        row_tile_index: int,
        p_tile_index: int,
    ):
        p_tile_width = 32
        g_stage = cute.local_tile(
            source_tensor[batch_head_chunk, None, 0, None],
            (self.kv_tile, p_tile_width),
            (row_tile_index, p_tile_index),
        )
        t_stage_gmem = gmem_thr_copy_p.partition_S(g_stage)
        c_stage = cute.local_tile(
            coord_tensor[batch_head_chunk, None, 0, None],
            (self.kv_tile, p_tile_width),
            (row_tile_index, p_tile_index),
        )
        t_stage_coord = gmem_thr_copy_p.partition_S(c_stage)
        t_stage_pred = self._make_copy_column_predicate(
            t_stage_smem, t_stage_coord, source_tensor.layout.shape[3]
        )
        self._copy_rows_with_zero_fill(
            gmem_tiled_copy_p,
            t_stage_gmem,
            t_stage_smem,
            t_stage_coord,
            t_stage_pred,
            source_tensor.layout.shape[1],
        )

    @cute.jit
    def _inject_shifted_value_boundary_row(
        self,
        s_value: cute.Tensor,
        m_value: cute.Tensor,
        m_value_prev0: cute.Tensor,
        *,
        batch_head_chunk: int,
        batch_head: int,
        chunk_index: int,
        p_tile_start: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        p_tile_width = 32
        for it in cutlass.range_constexpr(
            (p_tile_width + self.num_threads - 1) // self.num_threads
        ):
            p_local = tidx + cutlass.Int32(it * self.num_threads)
            if p_local < cutlass.Int32(p_tile_width):
                p = cutlass.Int32(p_tile_start) + p_local
                boundary_value = cutlass.Float32(0.0).to(out_dtype)
                if p < cutlass.Int32(self.P):
                    if chunk_index == cutlass.Int32(0):
                        boundary_value = m_value_prev0[batch_head, p]
                    else:
                        boundary_value = m_value[
                            batch_head_chunk - cutlass.Int32(1),
                            cutlass.Int32(self.L - 1),
                            0,
                            p,
                        ]
                s_value[0, p_local] = boundary_value
        pipeline.sync()

    @cute.jit
    def _stage_shifted_value_p_tile(
        self,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_thr_copy_p,
        m_value: cute.Tensor,
        coord_value: cute.Tensor,
        m_value_prev0: cute.Tensor,
        t_value_smem: cute.Tensor,
        s_value: cute.Tensor,
        *,
        batch_head_chunk: int,
        batch_head: int,
        chunk_index: int,
        n_tile_index: int,
        p_tile_index: int,
        out_dtype: type[cutlass.Numeric],
    ):
        p_tile_width = 32
        g_shifted_value = cute.local_tile(
            m_value[batch_head_chunk, None, 0, None],
            (self.kv_tile, p_tile_width),
            (n_tile_index, p_tile_index),
        )
        g_shifted_value = cute.domain_offset((-1, 0), g_shifted_value)
        g_shifted_value = cute.make_tensor(
            g_shifted_value.iterator.align(16), g_shifted_value.layout
        )
        t_value_gmem = gmem_thr_copy_p.partition_S(g_shifted_value)
        c_shifted_value = cute.local_tile(
            coord_value[batch_head_chunk, None, 0, None],
            (self.kv_tile, p_tile_width),
            (n_tile_index, p_tile_index),
        )
        c_shifted_value = cute.domain_offset((-1, 0), c_shifted_value)
        t_value_coord = gmem_thr_copy_p.partition_S(c_shifted_value)
        t_value_pred = self._make_copy_column_predicate(
            t_value_smem, t_value_coord, m_value.layout.shape[3]
        )
        self._copy_rows_in_range_with_zero_fill(
            gmem_tiled_copy_p,
            t_value_gmem,
            t_value_smem,
            t_value_coord,
            t_value_pred,
            row_lower_bound=0,
            row_upper_bound=m_value.layout.shape[1],
        )
        if cutlass.const_expr(n_tile_index == 0):
            self._inject_shifted_value_boundary_row(
                s_value,
                m_value,
                m_value_prev0,
                batch_head_chunk=batch_head_chunk,
                batch_head=batch_head,
                chunk_index=chunk_index,
                p_tile_start=p_tile_index * p_tile_width,
                out_dtype=out_dtype,
            )

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
        barrier_after: cutlass.Constexpr,
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
        if cutlass.const_expr(barrier_after):
            pipeline.sync()

    @cute.jit
    def _compute_phase_prefix_metadata(self, prefix_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        seq_idx = tidx
        phase_re = cutlass.Float32(1.0)
        phase_im = cutlass.Float32(0.0)
        logp = cutlass.Float32(0.0)

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
        pipeline.sync()

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
                next_phase_re, next_phase_im = complex_mul(
                    running_phase_re,
                    running_phase_im,
                    total_phase_re,
                    total_phase_im,
                )
                running_log = running_log + total_log
                running_phase_re = next_phase_re
                running_phase_im = next_phase_im
        pipeline.sync()

        logp = logp + prefix_state.warp_log_offset[warp]
        phase_re, phase_im = complex_mul(
            phase_re,
            phase_im,
            prefix_state.warp_phase_offset[warp, 0],
            prefix_state.warp_phase_offset[warp, 1],
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
        pipeline.sync()

    @cute.jit
    def _initialize_prev_taps(
        self,
        mK: cute.Tensor,
        s_tap_prev: cute.Tensor,
        *,
        batch_head_chunk: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < self.L:
            s_tap_prev[tidx, 0] = cutlass.Float32(mK[batch_head_chunk, tidx, 0, 0])
            s_tap_prev[tidx, 1] = cutlass.Float32(mK[batch_head_chunk, tidx, 0, 1])
        pipeline.sync()

    @cute.jit
    def _initialize_curr_taps_and_dm_tiles(
        self,
        mK: cute.Tensor,
        s_tap_curr: cute.Tensor,
        s_dm_curr: cute.Tensor,
        s_dm_prev: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile_start: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < self.kv_tile:
            row = n_tile_start + tidx
            s_tap_curr[tidx, 0] = cutlass.Float32(mK[batch_head_chunk, row, 1, 0])
            s_tap_curr[tidx, 1] = cutlass.Float32(mK[batch_head_chunk, row, 1, 1])
            s_dm_curr[tidx, 0] = cutlass.Float32(0.0)
            s_dm_curr[tidx, 1] = cutlass.Float32(0.0)
            s_dm_prev[tidx, 0] = cutlass.Float32(0.0)
            s_dm_prev[tidx, 1] = cutlass.Float32(0.0)
        pipeline.sync()

    @cute.jit
    def _rotate_and_scale_staged_query_tile_from_prefix(
        self,
        s_query: cute.Tensor,
        s_phase: cute.Tensor,
        s_row_scale: cute.Tensor,
        *,
        m_tile_start: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        stage_complex = stage_width // 2
        total_pairs = self.kv_tile * stage_complex
        idx = tidx
        while cute.elem_less(idx, total_pairs):
            row_local = idx // stage_complex
            pair_local = idx - row_local * stage_complex
            row = cutlass.Int32(m_tile_start) + row_local
            if row < cutlass.Int32(self.L):
                d_local = pair_local * 2
                xr = cutlass.Float32(
                    s_query[row_local, d_local + 0].to(cutlass.Float32)
                )
                xi = cutlass.Float32(
                    s_query[row_local, d_local + 1].to(cutlass.Float32)
                )
                pr = cutlass.Float32(s_phase[row, 0])
                pi = cutlass.Float32(s_phase[row, 1])
                yr, yi = conj_mul_phase(xr, xi, pr, pi)
                scale = cutlass.Float32(s_row_scale[row])
                s_query[row_local, d_local + 0] = safe_cast_to_dtype(
                    yr * scale, out_dtype
                )
                s_query[row_local, d_local + 1] = safe_cast_to_dtype(
                    yi * scale, out_dtype
                )
            idx = idx + self.num_threads
        pipeline.sync()

    @cute.jit
    def _stage_score_operands_for_p_tile(
        self,
        score_pipeline: SimpleNamespace,
        *,
        m_tile_index: int,
        n_tile_index: int,
        p_tile_index: int,
        use_shifted_values: cutlass.Constexpr,
        use_stage1: cutlass.Constexpr,
    ):
        if cutlass.const_expr(use_stage1):
            t_dout_stage = score_pipeline.t_dout_stage1
            t_value_stage = score_pipeline.t_value_stage1
            s_value_stage = score_pipeline.s_value_stage1
        else:
            t_dout_stage = score_pipeline.t_dout_stage0
            t_value_stage = score_pipeline.t_value_stage0
            s_value_stage = score_pipeline.s_value_stage0

        self._stage_p_tile_from_gmem(
            score_pipeline.gmem_tiled_copy_p,
            score_pipeline.gmem_thr_copy_p,
            score_pipeline.m_dout,
            score_pipeline.coord_dout,
            t_dout_stage,
            batch_head_chunk=score_pipeline.batch_head_chunk,
            row_tile_index=m_tile_index,
            p_tile_index=p_tile_index,
        )
        if cutlass.const_expr(use_shifted_values):
            self._stage_shifted_value_p_tile(
                score_pipeline.gmem_tiled_copy_p,
                score_pipeline.gmem_thr_copy_p,
                score_pipeline.m_value,
                score_pipeline.coord_value,
                score_pipeline.m_value_prev0,
                t_value_stage,
                s_value_stage,
                batch_head_chunk=score_pipeline.batch_head_chunk,
                batch_head=score_pipeline.batch_head,
                chunk_index=score_pipeline.chunk_index,
                n_tile_index=n_tile_index,
                p_tile_index=p_tile_index,
                out_dtype=score_pipeline.m_value.element_type,
            )
        else:
            self._stage_p_tile_from_gmem(
                score_pipeline.gmem_tiled_copy_p,
                score_pipeline.gmem_thr_copy_p,
                score_pipeline.m_value,
                score_pipeline.coord_value,
                t_value_stage,
                batch_head_chunk=score_pipeline.batch_head_chunk,
                row_tile_index=n_tile_index,
                p_tile_index=p_tile_index,
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _accumulate_score_block_from_p_tiles(
        self,
        score_pipeline: SimpleNamespace,
        acc_score_block: cute.Tensor,
        *,
        m_tile_index: int,
        n_tile_index: int,
        use_shifted_values: cutlass.Constexpr,
    ):
        self._stage_score_operands_for_p_tile(
            score_pipeline,
            m_tile_index=m_tile_index,
            n_tile_index=n_tile_index,
            p_tile_index=0,
            use_shifted_values=use_shifted_values,
            use_stage1=False,
        )
        for p_tile_index in cutlass.range_constexpr(score_pipeline.num_p_tiles):
            cute.arch.cp_async_wait_group(0)
            pipeline.sync()

            if cutlass.const_expr(p_tile_index + 1 < score_pipeline.num_p_tiles):
                self._stage_score_operands_for_p_tile(
                    score_pipeline,
                    m_tile_index=m_tile_index,
                    n_tile_index=n_tile_index,
                    p_tile_index=p_tile_index + 1,
                    use_shifted_values=use_shifted_values,
                    use_stage1=((p_tile_index & 1) == 0),
                )

            if cutlass.const_expr((p_tile_index & 1) == 0):
                s_dout_stage = score_pipeline.s_dout_stage0
                s_value_stage = score_pipeline.s_value_stage0
            else:
                s_dout_stage = score_pipeline.s_dout_stage1
                s_value_stage = score_pipeline.s_value_stage1

            t_reg_dout = score_pipeline.thr_mma.make_fragment_A(
                score_pipeline.thr_mma.partition_A(s_dout_stage)
            )
            t_smem_dout = score_pipeline.thr_copy_lhs.partition_S(s_dout_stage)
            t_reg_dout_view = score_pipeline.thr_copy_lhs.retile(t_reg_dout)
            t_reg_value = score_pipeline.thr_mma.make_fragment_B(
                score_pipeline.thr_mma.partition_B(s_value_stage)
            )
            t_smem_value = score_pipeline.thr_copy_rhs.partition_S(s_value_stage)
            t_reg_value_view = score_pipeline.thr_copy_rhs.retile(t_reg_value)
            self._accumulate_from_staged_tiles(
                score_pipeline.tiled_mma,
                acc_score_block,
                score_pipeline.smem_tiled_copy_lhs,
                score_pipeline.smem_tiled_copy_rhs,
                t_smem_dout,
                t_reg_dout_view,
                t_smem_value,
                t_reg_value_view,
                t_reg_dout,
                t_reg_value,
                True,
            )

    @cute.jit
    def _accumulate_score_block_from_p_tiles_single_stage(
        self,
        score_pipeline: SimpleNamespace,
        acc_score_block: cute.Tensor,
        *,
        m_tile_index: int,
        n_tile_index: int,
        use_shifted_values: cutlass.Constexpr,
    ):
        for p_tile_index in cutlass.range_constexpr(score_pipeline.num_p_tiles):
            self._stage_score_operands_for_p_tile(
                score_pipeline,
                m_tile_index=m_tile_index,
                n_tile_index=n_tile_index,
                p_tile_index=p_tile_index,
                use_shifted_values=use_shifted_values,
                use_stage1=False,
            )
            cute.arch.cp_async_wait_group(0)
            pipeline.sync()

            t_reg_dout = score_pipeline.thr_mma.make_fragment_A(
                score_pipeline.thr_mma.partition_A(score_pipeline.s_dout_stage0)
            )
            t_smem_dout = score_pipeline.thr_copy_lhs.partition_S(
                score_pipeline.s_dout_stage0
            )
            t_reg_dout_view = score_pipeline.thr_copy_lhs.retile(t_reg_dout)
            t_reg_value = score_pipeline.thr_mma.make_fragment_B(
                score_pipeline.thr_mma.partition_B(score_pipeline.s_value_stage0)
            )
            t_smem_value = score_pipeline.thr_copy_rhs.partition_S(
                score_pipeline.s_value_stage0
            )
            t_reg_value_view = score_pipeline.thr_copy_rhs.retile(t_reg_value)
            self._accumulate_from_staged_tiles(
                score_pipeline.tiled_mma,
                acc_score_block,
                score_pipeline.smem_tiled_copy_lhs,
                score_pipeline.smem_tiled_copy_rhs,
                t_smem_dout,
                t_reg_dout_view,
                t_smem_value,
                t_reg_value_view,
                t_reg_dout,
                t_reg_value,
                True,
            )

    @cute.jit
    def _store_causal_score_block(
        self,
        acc_score_block: cute.Tensor,
        t_score_coord_mn: cute.Tensor,
        s_score: cute.Tensor,
        *,
        m_tile_start: int,
        n_tile_start: int,
        out_dtype: type[cutlass.Numeric],
    ):
        acc_score_block_mn = self._make_accumulator_mn_view(acc_score_block)
        for r in cutlass.range_constexpr(cute.size(acc_score_block_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            row_local = row_idx - cutlass.Int32(m_tile_start)
            for c in cutlass.range_constexpr(cute.size(acc_score_block_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                col_local = col_idx - cutlass.Int32(n_tile_start)
                score_value = cutlass.Float32(0.0)
                if cute.elem_less(col_idx, row_idx + 1):
                    score_value = acc_score_block_mn[r, c]
                s_score[col_local, row_local] = safe_cast_to_dtype(
                    score_value, out_dtype
                )
        pipeline.sync()

    @cute.jit
    def _accumulate_key_grad_from_staged_score_block(
        self,
        tiled_mma: cute.TiledMma,
        acc_dk: cute.Tensor,
        smem_tiled_copy_score: cute.TiledCopy,
        smem_tiled_copy_query_t: cute.TiledCopy,
        t_smem_score: cute.Tensor,
        t_reg_score_view: cute.Tensor,
        t_smem_query_t: cute.Tensor,
        t_reg_query_t_view: cute.Tensor,
        t_reg_score: cute.Tensor,
        t_reg_query_t: cute.Tensor,
        barrier_after: cutlass.Constexpr,
    ):
        self._accumulate_from_staged_tiles(
            tiled_mma,
            acc_dk,
            smem_tiled_copy_score,
            smem_tiled_copy_query_t,
            t_smem_score,
            t_reg_score_view,
            t_smem_query_t,
            t_reg_query_t_view,
            t_reg_score,
            t_reg_query_t,
            barrier_after,
        )

    @cute.jit
    def _convert_accumulator_to_staged_grad_fragment(
        self,
        acc_dk: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        s_inv_row_scale: cute.Tensor,
        s_phase: cute.Tensor,
        target_mn: cute.Tensor,
        *,
        out_dtype: type[cutlass.Numeric],
    ):
        acc_dk_mn = self._make_accumulator_mn_view(acc_dk)
        for r in cutlass.range_constexpr(cute.size(acc_dk_mn.shape[0])):
            row_idx = cutlass.Int32(t_output_coord_mn[r, 0][1])
            if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                inv_row_scale = cutlass.Float32(s_inv_row_scale[row_idx])
                for c in cutlass.range(cute.size(acc_dk_mn.shape[1])):
                    d = cutlass.Int32(t_output_coord_mn[0, c][3])
                    if d + cutlass.Int32(1) < cutlass.Int32(self.D) and (
                        d & cutlass.Int32(1)
                    ) == cutlass.Int32(0):
                        gx = acc_dk_mn[r, c] * inv_row_scale
                        gy = acc_dk_mn[r, c + 1] * inv_row_scale
                        pr = cutlass.Float32(s_phase[row_idx, 0])
                        pi = cutlass.Float32(s_phase[row_idx, 1])
                        br, bi = conj_mul_phase(gx, gy, pr, pi)
                        target_mn[r, c] = safe_cast_to_dtype(br, out_dtype)
                        target_mn[r, c + 1] = safe_cast_to_dtype(bi, out_dtype)

    @cute.jit
    def _accumulate_dm_and_apply_tap_adjoint(
        self,
        s_db_output: cute.Tensor,
        s_db_curr: cute.Tensor,
        s_db_prev: cute.Tensor,
        s_phase: cute.Tensor,
        s_key_tile: cute.Tensor,
        s_key_boundary: cute.Tensor,
        s_dm_curr: cute.Tensor,
        s_dm_prev: cute.Tensor,
        s_tap_curr: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_db_carry: cute.Tensor,
        *,
        n_tile_start: int,
        d_col_base: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        nvec_stage = stage_width // 2
        row_local = warp
        while row_local < cutlass.Int32(self.kv_tile):
            row = cutlass.Int32(n_tile_start) + row_local
            if row < cutlass.Int32(self.L):
                dmy_curr_re = cutlass.Float32(0.0)
                dmy_curr_im = cutlass.Float32(0.0)
                dmy_prev_re = cutlass.Float32(0.0)
                dmy_prev_im = cutlass.Float32(0.0)
                pr = cutlass.Float32(s_phase[row, 0])
                pi = cutlass.Float32(s_phase[row, 1])
                if lane < cutlass.Int32(nvec_stage):
                    d_local = lane * 2
                    d = cutlass.Int32(d_col_base) + d_local
                    if d + cutlass.Int32(1) < cutlass.Int32(self.D):
                        curr_re = cutlass.Float32(
                            s_db_curr[row_local, d_local + 0].to(cutlass.Float32)
                        )
                        curr_im = cutlass.Float32(
                            s_db_curr[row_local, d_local + 1].to(cutlass.Float32)
                        )
                        gx_curr, gy_curr = conj_mul_phase(curr_re, curr_im, pr, pi)
                        br_curr = cutlass.Float32(
                            s_key_tile[row_local, d_local + 0].to(cutlass.Float32)
                        )
                        bi_curr = cutlass.Float32(
                            s_key_tile[row_local, d_local + 1].to(cutlass.Float32)
                        )
                        dmy_curr_re = gx_curr * br_curr - gy_curr * bi_curr
                        dmy_curr_im = gx_curr * bi_curr + gy_curr * br_curr

                        prev_re = cutlass.Float32(
                            s_db_prev[row_local, d_local + 0].to(cutlass.Float32)
                        )
                        prev_im = cutlass.Float32(
                            s_db_prev[row_local, d_local + 1].to(cutlass.Float32)
                        )
                        gx_prev, gy_prev = conj_mul_phase(prev_re, prev_im, pr, pi)
                        br_prev = cutlass.Float32(0.0)
                        bi_prev = cutlass.Float32(0.0)
                        if row_local > cutlass.Int32(0):
                            br_prev = cutlass.Float32(
                                s_key_tile[
                                    row_local - cutlass.Int32(1), d_local + 0
                                ].to(cutlass.Float32)
                            )
                            bi_prev = cutlass.Float32(
                                s_key_tile[
                                    row_local - cutlass.Int32(1), d_local + 1
                                ].to(cutlass.Float32)
                            )
                        else:
                            br_prev = cutlass.Float32(
                                s_key_boundary[d_local + 0].to(cutlass.Float32)
                            )
                            bi_prev = cutlass.Float32(
                                s_key_boundary[d_local + 1].to(cutlass.Float32)
                            )
                        dmy_prev_re = gx_prev * br_prev - gy_prev * bi_prev
                        dmy_prev_im = gx_prev * bi_prev + gy_prev * br_prev

                        tap_curr_re = cutlass.Float32(s_tap_curr[row_local, 0])
                        tap_curr_im = cutlass.Float32(s_tap_curr[row_local, 1])
                        out_re, out_im = apply_complex_tap_adjoint(
                            curr_re, curr_im, tap_curr_re, tap_curr_im
                        )

                        carry_re = cutlass.Float32(0.0)
                        carry_im = cutlass.Float32(0.0)
                        if row + cutlass.Int32(1) < cutlass.Int32(self.L):
                            if row_local + cutlass.Int32(1) < cutlass.Int32(
                                self.kv_tile
                            ):
                                row_next = row + cutlass.Int32(1)
                                next_prev_re = cutlass.Float32(
                                    s_db_prev[
                                        row_local + cutlass.Int32(1), d_local + 0
                                    ].to(cutlass.Float32)
                                )
                                next_prev_im = cutlass.Float32(
                                    s_db_prev[
                                        row_local + cutlass.Int32(1), d_local + 1
                                    ].to(cutlass.Float32)
                                )
                                tap_prev_re = cutlass.Float32(s_tap_prev[row_next, 0])
                                tap_prev_im = cutlass.Float32(s_tap_prev[row_next, 1])
                                carry_re, carry_im = apply_complex_tap_adjoint(
                                    next_prev_re,
                                    next_prev_im,
                                    tap_prev_re,
                                    tap_prev_im,
                                )
                            else:
                                carry_re = cutlass.Float32(
                                    s_db_carry[d_col_base + d_local + 0].to(
                                        cutlass.Float32
                                    )
                                )
                                carry_im = cutlass.Float32(
                                    s_db_carry[d_col_base + d_local + 1].to(
                                        cutlass.Float32
                                    )
                                )

                        s_db_output[row_local, d_local + 0] = safe_cast_to_dtype(
                            out_re + carry_re, out_dtype
                        )
                        s_db_output[row_local, d_local + 1] = safe_cast_to_dtype(
                            out_im + carry_im, out_dtype
                        )

                for offset in (16, 8, 4, 2, 1):
                    dmy_curr_re = dmy_curr_re + cute.arch.shuffle_sync_bfly(
                        dmy_curr_re, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    dmy_curr_im = dmy_curr_im + cute.arch.shuffle_sync_bfly(
                        dmy_curr_im, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    dmy_prev_re = dmy_prev_re + cute.arch.shuffle_sync_bfly(
                        dmy_prev_re, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    dmy_prev_im = dmy_prev_im + cute.arch.shuffle_sync_bfly(
                        dmy_prev_im, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if lane == cutlass.Int32(0):
                    s_dm_curr[row_local, 0] = cutlass.Float32(
                        s_dm_curr[row_local, 0] + dmy_curr_re
                    )
                    s_dm_curr[row_local, 1] = cutlass.Float32(
                        s_dm_curr[row_local, 1] + dmy_curr_im
                    )
                    s_dm_prev[row_local, 0] = cutlass.Float32(
                        s_dm_prev[row_local, 0] + dmy_prev_re
                    )
                    s_dm_prev[row_local, 1] = cutlass.Float32(
                        s_dm_prev[row_local, 1] + dmy_prev_im
                    )
            row_local = row_local + cutlass.Int32(self.num_warps)
        pipeline.sync()

    @cute.jit
    def _clear_db_carry(self, s_db_carry: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        for it in cutlass.range_constexpr(
            (self.D_padded + self.num_threads - 1) // self.num_threads
        ):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D_padded):
                s_db_carry[d] = cutlass.Float32(0.0)
        pipeline.sync()

    @cute.jit
    def _update_db_carry_from_tile_head(
        self,
        s_db_prev: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_db_carry: cute.Tensor,
        *,
        n_tile_start: int,
        d_col_base: int,
        stage_width: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        for it in cutlass.range_constexpr(
            (stage_width + self.num_threads - 1) // self.num_threads
        ):
            d_local = tidx + cutlass.Int32(it * self.num_threads)
            if d_local + cutlass.Int32(1) < cutlass.Int32(stage_width) and (
                d_local & cutlass.Int32(1)
            ) == cutlass.Int32(0):
                d = cutlass.Int32(d_col_base) + d_local
                if d < cutlass.Int32(self.D_padded):
                    head_re = cutlass.Float32(
                        s_db_prev[0, d_local + 0].to(cutlass.Float32)
                    )
                    head_im = cutlass.Float32(
                        s_db_prev[0, d_local + 1].to(cutlass.Float32)
                    )
                    tap_re = cutlass.Float32(s_tap_prev[n_tile_start, 0])
                    tap_im = cutlass.Float32(s_tap_prev[n_tile_start, 1])
                    carry_re, carry_im = apply_complex_tap_adjoint(
                        head_re, head_im, tap_re, tap_im
                    )
                    s_db_carry[d_col_base + d_local + 0] = cutlass.Float32(carry_re)
                    s_db_carry[d_col_base + d_local + 1] = cutlass.Float32(carry_im)
        pipeline.sync()

    @cute.jit
    def _store_dm_tile(
        self,
        mDMcurr: cute.Tensor,
        mDMprev: cute.Tensor,
        s_dm_curr: cute.Tensor,
        s_dm_prev: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile_start: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < cutlass.Int32(self.kv_tile):
            row = cutlass.Int32(n_tile_start) + tidx
            if row < cutlass.Int32(self.L):
                mDMcurr[batch_head_chunk, row, 0] = s_dm_curr[tidx, 0]
                mDMcurr[batch_head_chunk, row, 1] = s_dm_curr[tidx, 1]
                mDMprev[batch_head_chunk, row, 0] = s_dm_prev[tidx, 0]
                mDMprev[batch_head_chunk, row, 1] = s_dm_prev[tidx, 1]
        pipeline.sync()

    @cute.jit
    def _store_db_prev_carry(
        self,
        mDB_prev: cute.Tensor,
        s_db_carry: cute.Tensor,
        *,
        batch_head_chunk: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        for it in cutlass.range_constexpr(
            (self.D + self.num_threads - 1) // self.num_threads
        ):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D):
                mDB_prev[batch_head_chunk, d] = safe_cast_to_dtype(
                    s_db_carry[d], mDB_prev.element_type
                )

    @cute.jit
    def _store_db_tile(
        self,
        gmem_tiled_store_d: cute.TiledCopy,
        mDB: cute.Tensor,
        coord_db: cute.Tensor,
        s_db_tile: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_tile_index: int,
        d_stage_index: int,
        stage_width: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        g_db_stage = cute.local_tile(
            mDB[batch_head_chunk, None, 0, None],
            (self.kv_tile, stage_width),
            (n_tile_index, d_stage_index),
        )
        gmem_thr_store_d = gmem_tiled_store_d.get_slice(tidx)
        t_db_smem = gmem_thr_store_d.partition_S(s_db_tile)
        t_db_gmem = gmem_thr_store_d.partition_D(g_db_stage)
        if cutlass.const_expr(self.D == self.D_padded):
            cute.copy(gmem_tiled_store_d, t_db_smem, t_db_gmem)
            return

        t_db_reg = cute.make_rmem_tensor_like(t_db_gmem, mDB.element_type)
        cute.copy(gmem_tiled_store_d, t_db_smem, t_db_reg)
        c_db_stage = cute.local_tile(
            coord_db[batch_head_chunk, None, 0, None],
            (self.kv_tile, stage_width),
            (n_tile_index, d_stage_index),
        )
        t_db_coord = gmem_thr_store_d.partition_D(c_db_stage)
        t_db_pred = self._make_copy_column_predicate(
            t_db_gmem, t_db_coord, mDB.layout.shape[3]
        )
        self._copy_rows_if_valid(
            gmem_tiled_store_d,
            t_db_reg,
            t_db_gmem,
            t_db_coord,
            t_db_pred,
            mDB.layout.shape[1],
        )

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
        mDB: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
    ):
        if cutlass.const_expr(
            mU.element_type != mB.element_type or mU.element_type != mC.element_type
        ):
            raise TypeError("U/B/C must share dtype.")
        if cutlass.const_expr(mDOut.element_type != mU.element_type):
            raise TypeError("dOut must share dtype with U/B/C.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("Tensor-core path supports only Float16/BFloat16 inputs.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(
            mDMprev.element_type != cutlass.Float32
            or mDMcurr.element_type != cutlass.Float32
        ):
            raise TypeError("dM buffers must be Float32.")
        if cutlass.const_expr(
            mU.shape[1] != self.L or mB.shape[1] != self.L or mC.shape[1] != self.L
        ):
            raise ValueError("U/B/C must have shape (BHC, L, 1, ...).")
        if cutlass.const_expr(mU.shape[2] != 1 or mB.shape[2] != 1 or mC.shape[2] != 1):
            raise ValueError("U/B/C must have a singleton dim2 (BHC, L, 1, ...).")
        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be (BHC, L, 2).")
        if cutlass.const_expr(
            mK.shape[1] != self.L or mK.shape[2] != 2 or mK.shape[3] != 2
        ):
            raise ValueError("K must be (BHC, L, 2, 2).")
        if cutlass.const_expr(mDOut.shape[1] != self.L or mDOut.shape[2] != 1):
            raise ValueError("dOut must be (BHC, L, 1, P).")
        if cutlass.const_expr(mDB.shape[1] != self.L or mDB.shape[2] != 1):
            raise ValueError("dB must be (BHC, L, 1, D).")
        if cutlass.const_expr(mDB_prev.shape[1] != self.D):
            raise ValueError("dB_prev must be (BHC, D).")
        if cutlass.const_expr(
            mDMprev.shape[1] != self.L
            or mDMprev.shape[2] != 2
            or mDMcurr.shape[1] != self.L
            or mDMcurr.shape[2] != 2
        ):
            raise ValueError("dM buffers must be (BHC, L, 2).")
        if cutlass.const_expr(self.P_padded % 32 != 0):
            raise ValueError("P must be padded to a multiple of 32.")

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
        mDB: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mU.element_type)
        layouts = bundle.layouts
        copies = bundle.copies
        grid_dim = (1, 1, cute.size(mU.shape[0]))
        launch_kwargs = {
            "grid": grid_dim,
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_bytes,
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
            mDB,
            mDB_prev,
            mDMprev,
            mDMcurr,
            layouts.query_layout,
            layouts.key_layout,
            layouts.grad_output_layout,
            layouts.key_grad_layout,
            layouts.value_layout,
            layouts.score_layout,
            copies.gmem_tiled_copy_d,
            copies.gmem_tiled_copy_p,
            copies.gmem_tiled_store_d,
            bundle.tiled_mma,
            bundle.shared_storage_cls,
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
        mDB: cute.Tensor,
        mDB_prev: cute.Tensor,
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
            mDB,
            mDB_prev,
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
            mDB,
            mDB_prev,
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
        mDB: cute.Tensor,
        mDB_prev: cute.Tensor,
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
            mDB,
            mDB_prev,
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
            mDB,
            mDB_prev,
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
        mDB: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        query_layout: cute.ComposedLayout,
        key_layout: cute.ComposedLayout,
        grad_output_layout: cute.ComposedLayout,
        key_grad_layout: cute.ComposedLayout,
        value_layout: cute.ComposedLayout,
        score_layout: cute.ComposedLayout,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_tiled_store_d: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_head_chunk = cute.arch.block_idx()
        d_padded = self.D_padded
        d_stage_width = self._d_stage_size()
        d_stage_count = self._d_stage_count()
        p_padded = self.P_padded
        p_tile = 32
        num_p_tiles = p_padded // p_tile
        num_batch_heads = mU_prev0.shape[0]
        num_batch_head_chunks = mU.shape[0]
        num_chunks = num_batch_head_chunks // num_batch_heads
        batch_head = batch_head_chunk // num_chunks
        batch_group = self._batch_group(batch_head)
        batch_group_chunk = self._batch_group_chunk(batch_head_chunk, num_chunks)
        chunk_index = batch_head_chunk - batch_head * num_chunks
        kv_tile = int(self.kv_tile)
        num_n_tiles = int(self.L // kv_tile)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)

        s_query_tile = storage.query_tile.get_tensor(query_layout)
        s_key_tile = storage.key_tile.get_tensor(key_layout)
        s_grad_output_stage0 = storage.grad_output_tile.get_tensor(grad_output_layout)
        s_key_grad_tile = storage.key_grad_tile.get_tensor(key_grad_layout)
        s_value_stage0 = storage.value_tile.get_tensor(value_layout)
        s_score_temp = storage.score_tile.get_tensor(score_layout)
        s_score_diag_curr = storage.score_diag_curr.get_tensor(score_layout)
        s_score_diag_prev = cute.make_tensor(
            s_grad_output_stage0.iterator.align(16), score_layout
        )
        s_score_offdiag_curr = s_score_temp
        s_score_offdiag_prev = cute.make_tensor(
            s_value_stage0.iterator.align(16), score_layout
        )
        s_grad_output_stage1 = cute.make_tensor(
            s_key_grad_tile.iterator.align(16), grad_output_layout
        )
        s_value_stage1 = cute.make_tensor(s_score_temp.iterator.align(16), value_layout)
        s_phase = storage.phase_full.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_prev = storage.tap_prev_full.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_curr = storage.tap_curr_tile.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )
        s_db_carry = storage.db_carry.get_tensor(
            cute.make_layout((d_padded,), stride=(1,))
        )
        s_key_boundary = storage.key_boundary.get_tensor(
            cute.make_layout((d_stage_width,), stride=(1,))
        )
        s_row_scale = storage.row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )
        s_inv_row_scale = storage.inv_row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )
        s_dm_curr = storage.dm_curr.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )
        s_dm_prev = storage.dm_prev.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )

        warp_log_total = storage.warp_log_total.get_tensor(
            cute.make_layout((self.num_warps,), stride=(1,))
        )
        warp_log_offset = storage.warp_log_offset.get_tensor(
            cute.make_layout((self.num_warps,), stride=(1,))
        )
        warp_phase_total = storage.warp_phase_total.get_tensor(
            cute.make_layout((self.num_warps, 2), stride=(2, 1))
        )
        warp_phase_offset = storage.warp_phase_offset.get_tensor(
            cute.make_layout((self.num_warps, 2), stride=(2, 1))
        )
        prefix_state = SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_transition=mM,
            s_phase=s_phase,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            warp_log_total=warp_log_total,
            warp_log_offset=warp_log_offset,
            warp_phase_total=warp_phase_total,
            warp_phase_offset=warp_phase_offset,
        )
        self._compute_phase_prefix_metadata(prefix_state)
        self._initialize_prev_taps(
            mK,
            s_tap_prev,
            batch_head_chunk=batch_head_chunk,
        )

        smem_copy_atom_lhs = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_rhs = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_query_t = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_lhs = cute.make_tiled_copy_A(smem_copy_atom_lhs, tiled_mma)
        smem_tiled_copy_rhs = cute.make_tiled_copy_B(smem_copy_atom_rhs, tiled_mma)
        smem_tiled_copy_query_t = cute.make_tiled_copy_B(
            smem_copy_atom_query_t, tiled_mma
        )
        thr_mma = tiled_mma.get_slice(tidx)
        thr_copy_lhs = smem_tiled_copy_lhs.get_slice(tidx)
        thr_copy_rhs = smem_tiled_copy_rhs.get_slice(tidx)
        thr_copy_query_t = smem_tiled_copy_query_t.get_slice(tidx)
        gmem_thr_copy_d = gmem_tiled_copy_d.get_slice(tidx)
        gmem_thr_copy_p = gmem_tiled_copy_p.get_slice(tidx)
        coord_score = cute.make_identity_tensor(
            (mU.shape[0], self.L, mU.shape[2], self.L)
        )
        coord_db = cute.make_identity_tensor(mDB.layout.shape)
        coord_key = cute.make_identity_tensor(mB.layout.shape)
        coord_value = cute.make_identity_tensor(mU.layout.shape)
        coord_query = cute.make_identity_tensor(mC.layout.shape)
        coord_dout = cute.make_identity_tensor(mDOut.layout.shape)
        coord_score_tile_base = coord_score[batch_head_chunk, None, 0, None]
        coord_db_tile_base = coord_db[batch_head_chunk, None, 0, None]
        t_query_smem = gmem_thr_copy_d.partition_D(s_query_tile)
        t_key_smem = gmem_thr_copy_d.partition_D(s_key_tile)
        t_grad_output_stage0_smem = gmem_thr_copy_p.partition_D(s_grad_output_stage0)
        t_grad_output_stage1_smem = gmem_thr_copy_p.partition_D(s_grad_output_stage1)
        t_value_stage0_smem = gmem_thr_copy_p.partition_D(s_value_stage0)
        t_value_stage1_smem = gmem_thr_copy_p.partition_D(s_value_stage1)

        t_score_temp = thr_mma.make_fragment_A(thr_mma.partition_A(s_score_temp))
        t_score_temp_smem = thr_copy_lhs.partition_S(s_score_temp)
        t_score_temp_view = thr_copy_lhs.retile(t_score_temp)
        t_score_diag_curr = thr_mma.make_fragment_A(
            thr_mma.partition_A(s_score_diag_curr)
        )
        t_score_diag_curr_smem = thr_copy_lhs.partition_S(s_score_diag_curr)
        t_score_diag_curr_view = thr_copy_lhs.retile(t_score_diag_curr)
        t_score_diag_prev = thr_mma.make_fragment_A(
            thr_mma.partition_A(s_score_diag_prev)
        )
        t_score_diag_prev_smem = thr_copy_lhs.partition_S(s_score_diag_prev)
        t_score_diag_prev_view = thr_copy_lhs.retile(t_score_diag_prev)
        t_score_offdiag_prev = thr_mma.make_fragment_A(
            thr_mma.partition_A(s_score_offdiag_prev)
        )
        t_score_offdiag_prev_smem = thr_copy_lhs.partition_S(s_score_offdiag_prev)
        t_score_offdiag_prev_view = thr_copy_lhs.retile(t_score_offdiag_prev)
        s_query_transposed_layout = cute.make_layout(
            (d_stage_width, kv_tile), stride=(kv_tile, 1)
        )
        s_query_transposed = cute.composition(s_query_tile, s_query_transposed_layout)
        t_query_transposed = thr_mma.make_fragment_B(
            thr_mma.partition_B(s_query_transposed)
        )
        t_query_transposed_smem = thr_copy_query_t.partition_S(s_query_transposed)
        t_query_transposed_view = thr_copy_query_t.retile(t_query_transposed)

        acc_shape_blk = thr_mma.partition_shape_C((kv_tile, kv_tile))
        acc_shape_tile_d = thr_mma.partition_shape_C((kv_tile, d_stage_width))
        score_pipeline = SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            batch_head=batch_head,
            chunk_index=chunk_index,
            num_p_tiles=num_p_tiles,
            tiled_mma=tiled_mma,
            smem_tiled_copy_lhs=smem_tiled_copy_lhs,
            smem_tiled_copy_rhs=smem_tiled_copy_rhs,
            thr_mma=thr_mma,
            thr_copy_lhs=thr_copy_lhs,
            thr_copy_rhs=thr_copy_rhs,
            gmem_tiled_copy_p=gmem_tiled_copy_p,
            gmem_thr_copy_p=gmem_thr_copy_p,
            m_dout=mDOut,
            m_value=mU,
            m_value_prev0=mU_prev0,
            coord_dout=coord_dout,
            coord_value=coord_value,
            t_dout_stage0=t_grad_output_stage0_smem,
            t_dout_stage1=t_grad_output_stage1_smem,
            t_value_stage0=t_value_stage0_smem,
            t_value_stage1=t_value_stage1_smem,
            s_dout_stage0=s_grad_output_stage0,
            s_dout_stage1=s_grad_output_stage1,
            s_value_stage0=s_value_stage0,
            s_value_stage1=s_value_stage1,
        )
        score_pipeline_no_prev_alias = SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            batch_head=batch_head,
            chunk_index=chunk_index,
            num_p_tiles=num_p_tiles,
            tiled_mma=tiled_mma,
            smem_tiled_copy_lhs=smem_tiled_copy_lhs,
            smem_tiled_copy_rhs=smem_tiled_copy_rhs,
            thr_mma=thr_mma,
            thr_copy_lhs=thr_copy_lhs,
            thr_copy_rhs=thr_copy_rhs,
            gmem_tiled_copy_p=gmem_tiled_copy_p,
            gmem_thr_copy_p=gmem_thr_copy_p,
            m_dout=mDOut,
            m_value=mU,
            m_value_prev0=mU_prev0,
            coord_dout=coord_dout,
            coord_value=coord_value,
            t_dout_stage0=t_grad_output_stage1_smem,
            t_dout_stage1=t_grad_output_stage1_smem,
            t_value_stage0=t_value_stage0_smem,
            t_value_stage1=t_value_stage0_smem,
            s_dout_stage0=s_grad_output_stage1,
            s_dout_stage1=s_grad_output_stage1,
            s_value_stage0=s_value_stage0,
            s_value_stage1=s_value_stage0,
        )
        self._clear_db_carry(s_db_carry)

        for n_tile_rev in cutlass.range_constexpr(num_n_tiles):
            n_tile_index = (num_n_tiles - 1) - n_tile_rev
            n_tile_start = n_tile_index * kv_tile
            num_m_tiles = num_n_tiles - n_tile_index

            cached_m_tile_index = n_tile_index
            cached_m_tile_start = n_tile_start
            cached_score_coord_tile = cute.local_tile(
                coord_score_tile_base,
                (kv_tile, kv_tile),
                (cached_m_tile_index, n_tile_index),
            )
            t_cached_score_coord = thr_mma.partition_C(cached_score_coord_tile)
            t_cached_score_coord_mn = self._make_accumulator_mn_view(
                t_cached_score_coord
            )

            acc_score_diag_curr = cute.make_rmem_tensor(acc_shape_blk, cutlass.Float32)
            acc_score_diag_curr.fill(0.0)
            self._accumulate_score_block_from_p_tiles(
                score_pipeline,
                acc_score_diag_curr,
                m_tile_index=cached_m_tile_index,
                n_tile_index=n_tile_index,
                use_shifted_values=False,
            )
            self._store_causal_score_block(
                acc_score_diag_curr,
                t_cached_score_coord_mn,
                s_score_diag_curr,
                m_tile_start=cached_m_tile_start,
                n_tile_start=n_tile_start,
                out_dtype=mU.element_type,
            )

            acc_score_diag_prev = cute.make_rmem_tensor(acc_shape_blk, cutlass.Float32)
            acc_score_diag_prev.fill(0.0)
            self._accumulate_score_block_from_p_tiles(
                score_pipeline,
                acc_score_diag_prev,
                m_tile_index=cached_m_tile_index,
                n_tile_index=n_tile_index,
                use_shifted_values=True,
            )
            self._store_causal_score_block(
                acc_score_diag_prev,
                t_cached_score_coord_mn,
                s_score_diag_prev,
                m_tile_start=cached_m_tile_start,
                n_tile_start=n_tile_start,
                out_dtype=mU.element_type,
            )

            if cutlass.const_expr(num_n_tiles == 2 and num_m_tiles > 1):
                offdiag_m_tile_index = n_tile_index + 1
                offdiag_m_tile_start = offdiag_m_tile_index * kv_tile
                offdiag_score_coord_tile = cute.local_tile(
                    coord_score_tile_base,
                    (kv_tile, kv_tile),
                    (offdiag_m_tile_index, n_tile_index),
                )
                t_offdiag_score_coord = thr_mma.partition_C(offdiag_score_coord_tile)
                t_offdiag_score_coord_mn = self._make_accumulator_mn_view(
                    t_offdiag_score_coord
                )

                acc_score_offdiag_curr = cute.make_rmem_tensor(
                    acc_shape_blk, cutlass.Float32
                )
                acc_score_offdiag_curr.fill(0.0)
                self._accumulate_score_block_from_p_tiles_single_stage(
                    score_pipeline_no_prev_alias,
                    acc_score_offdiag_curr,
                    m_tile_index=offdiag_m_tile_index,
                    n_tile_index=n_tile_index,
                    use_shifted_values=False,
                )
                self._store_causal_score_block(
                    acc_score_offdiag_curr,
                    t_offdiag_score_coord_mn,
                    s_score_offdiag_curr,
                    m_tile_start=offdiag_m_tile_start,
                    n_tile_start=n_tile_start,
                    out_dtype=mU.element_type,
                )

                acc_score_offdiag_prev = cute.make_rmem_tensor(
                    acc_shape_blk, cutlass.Float32
                )
                acc_score_offdiag_prev.fill(0.0)
                self._accumulate_score_block_from_p_tiles_single_stage(
                    score_pipeline_no_prev_alias,
                    acc_score_offdiag_prev,
                    m_tile_index=offdiag_m_tile_index,
                    n_tile_index=n_tile_index,
                    use_shifted_values=True,
                )
                self._store_causal_score_block(
                    acc_score_offdiag_prev,
                    t_offdiag_score_coord_mn,
                    s_score_offdiag_prev,
                    m_tile_start=offdiag_m_tile_start,
                    n_tile_start=n_tile_start,
                    out_dtype=mU.element_type,
                )

            self._initialize_curr_taps_and_dm_tiles(
                mK,
                s_tap_curr,
                s_dm_curr,
                s_dm_prev,
                batch_head_chunk=batch_head_chunk,
                n_tile_start=n_tile_start,
            )

            for d_stage_index in cutlass.range_constexpr(d_stage_count):
                d_col_base = cutlass.Int32(d_stage_index * d_stage_width)
                self._stage_query_tile_from_gmem(
                    gmem_tiled_copy_d,
                    gmem_thr_copy_d,
                    mB,
                    coord_key,
                    t_key_smem,
                    batch_group_chunk=batch_group_chunk,
                    m_tile_index=n_tile_index,
                    d_stage_index=d_stage_index,
                    stage_width=d_stage_width,
                )
                self._stage_key_boundary_from_gmem(
                    s_key_boundary,
                    mB,
                    mB_prev0,
                    batch_group_chunk=batch_group_chunk,
                    batch_group=batch_group,
                    chunk_index=chunk_index,
                    d_col_base=d_col_base,
                    stage_width=d_stage_width,
                    out_dtype=mU.element_type,
                )
                acc_dk_curr = cute.make_rmem_tensor(acc_shape_tile_d, cutlass.Float32)
                acc_dk_curr.fill(0.0)
                acc_dk_prev = cute.make_rmem_tensor(acc_shape_tile_d, cutlass.Float32)
                acc_dk_prev.fill(0.0)

                for m_tile_offset in cutlass.range_constexpr(num_m_tiles):
                    m_tile_index = n_tile_index + m_tile_offset
                    m_tile_start = m_tile_index * kv_tile
                    cache_diag_score_tiles = m_tile_offset == 0
                    cache_offdiag_score_tiles = num_n_tiles == 2 and m_tile_offset == 1
                    cache_score_tiles = (
                        cache_diag_score_tiles or cache_offdiag_score_tiles
                    )

                    self._stage_query_tile_from_gmem(
                        gmem_tiled_copy_d,
                        gmem_thr_copy_d,
                        mC,
                        coord_query,
                        t_query_smem,
                        batch_group_chunk=batch_group_chunk,
                        m_tile_index=m_tile_index,
                        d_stage_index=d_stage_index,
                        stage_width=d_stage_width,
                    )
                    self._rotate_and_scale_staged_query_tile_from_prefix(
                        s_query_tile,
                        s_phase,
                        s_row_scale,
                        m_tile_start=m_tile_start,
                        stage_width=d_stage_width,
                        out_dtype=mU.element_type,
                    )

                    coord_score_tile = cute.local_tile(
                        coord_score_tile_base,
                        (kv_tile, kv_tile),
                        (m_tile_index, n_tile_index),
                    )
                    t_score_coord = thr_mma.partition_C(coord_score_tile)
                    t_score_coord_mn = self._make_accumulator_mn_view(t_score_coord)

                    if cutlass.const_expr(not cache_score_tiles):
                        acc_score_curr = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_score_curr.fill(0.0)
                        self._accumulate_score_block_from_p_tiles_single_stage(
                            score_pipeline_no_prev_alias,
                            acc_score_curr,
                            m_tile_index=m_tile_index,
                            n_tile_index=n_tile_index,
                            use_shifted_values=False,
                        )
                        self._store_causal_score_block(
                            acc_score_curr,
                            t_score_coord_mn,
                            s_score_temp,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_tile_start,
                            out_dtype=mU.element_type,
                        )

                    if cutlass.const_expr(cache_diag_score_tiles):
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_curr,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_diag_curr_smem,
                            t_score_diag_curr_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_diag_curr,
                            t_query_transposed,
                            False,
                        )
                    elif cutlass.const_expr(cache_offdiag_score_tiles):
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_curr,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_temp_smem,
                            t_score_temp_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_temp,
                            t_query_transposed,
                            False,
                        )
                    else:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_curr,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_temp_smem,
                            t_score_temp_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_temp,
                            t_query_transposed,
                            True,
                        )

                    if cutlass.const_expr(not cache_score_tiles):
                        acc_score_prev = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_score_prev.fill(0.0)
                        self._accumulate_score_block_from_p_tiles_single_stage(
                            score_pipeline_no_prev_alias,
                            acc_score_prev,
                            m_tile_index=m_tile_index,
                            n_tile_index=n_tile_index,
                            use_shifted_values=True,
                        )
                        self._store_causal_score_block(
                            acc_score_prev,
                            t_score_coord_mn,
                            s_score_temp,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_tile_start,
                            out_dtype=mU.element_type,
                        )
                    if cutlass.const_expr(cache_diag_score_tiles):
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_prev,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_diag_prev_smem,
                            t_score_diag_prev_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_diag_prev,
                            t_query_transposed,
                            True,
                        )
                    elif cutlass.const_expr(cache_offdiag_score_tiles):
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_prev,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_offdiag_prev_smem,
                            t_score_offdiag_prev_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_offdiag_prev,
                            t_query_transposed,
                            True,
                        )
                    else:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_prev,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_temp_smem,
                            t_score_temp_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_temp,
                            t_query_transposed,
                            True,
                        )

                coord_db_stage = cute.local_tile(
                    coord_db_tile_base,
                    (kv_tile, d_stage_width),
                    (n_tile_index, d_stage_index),
                )
                t_db_stage_coord = thr_mma.partition_C(coord_db_stage)
                t_db_stage_coord_mn = self._make_accumulator_mn_view(t_db_stage_coord)

                t_db_prev_smem = thr_mma.partition_C(s_query_tile)
                t_db_prev_reg = cute.make_rmem_tensor_like(
                    t_db_prev_smem, mU.element_type
                )
                t_db_prev_reg.fill(0.0)
                t_db_prev_reg_mn = self._make_accumulator_mn_view(t_db_prev_reg)
                t_db_curr_smem = thr_mma.partition_C(s_key_grad_tile)
                t_db_curr_reg = cute.make_rmem_tensor_like(
                    t_db_curr_smem, mU.element_type
                )
                t_db_curr_reg.fill(0.0)
                t_db_curr_reg_mn = self._make_accumulator_mn_view(t_db_curr_reg)

                self._convert_accumulator_to_staged_grad_fragment(
                    acc_dk_curr,
                    t_db_stage_coord_mn,
                    s_inv_row_scale,
                    s_phase,
                    t_db_curr_reg_mn,
                    out_dtype=mU.element_type,
                )
                self._convert_accumulator_to_staged_grad_fragment(
                    acc_dk_prev,
                    t_db_stage_coord_mn,
                    s_inv_row_scale,
                    s_phase,
                    t_db_prev_reg_mn,
                    out_dtype=mU.element_type,
                )
                cute.autovec_copy(t_db_prev_reg, t_db_prev_smem)
                cute.autovec_copy(t_db_curr_reg, t_db_curr_smem)
                pipeline.sync()

                self._accumulate_dm_and_apply_tap_adjoint(
                    s_key_grad_tile,
                    s_key_grad_tile,
                    s_query_tile,
                    s_phase,
                    s_key_tile,
                    s_key_boundary,
                    s_dm_curr,
                    s_dm_prev,
                    s_tap_curr,
                    s_tap_prev,
                    s_db_carry,
                    n_tile_start=n_tile_start,
                    d_col_base=d_col_base,
                    stage_width=d_stage_width,
                    out_dtype=mU.element_type,
                )
                self._store_db_tile(
                    gmem_tiled_store_d,
                    mDB,
                    coord_db,
                    s_key_grad_tile,
                    batch_head_chunk=batch_head_chunk,
                    n_tile_index=n_tile_index,
                    d_stage_index=d_stage_index,
                    stage_width=d_stage_width,
                )
                self._update_db_carry_from_tile_head(
                    s_query_tile,
                    s_tap_prev,
                    s_db_carry,
                    n_tile_start=n_tile_start,
                    d_col_base=d_col_base,
                    stage_width=d_stage_width,
                )

            self._store_dm_tile(
                mDMcurr,
                mDMprev,
                s_dm_curr,
                s_dm_prev,
                batch_head_chunk=batch_head_chunk,
                n_tile_start=n_tile_start,
            )
        self._store_db_prev_carry(
            mDB_prev,
            s_db_carry,
            batch_head_chunk=batch_head_chunk,
        )


@dataclass(frozen=True)
class _BwdDBIncrementAccumulatorLayoutBundle:
    u_major_mode: object
    d_inc_major_mode: object
    u_tile_layout: object
    d_inc_tile_layout: object
    db_tile_layout: object


@dataclass(frozen=True)
class _BwdDBIncrementAccumulatorCopyBundle:
    gmem_tiled_copy_u: object
    gmem_tiled_copy_d_inc: object
    ldmatrix_tiled_copy_u: object
    ldmatrix_tiled_copy_d_inc: object


@dataclass(frozen=True)
class _BwdDBIncrementAccumulatorKernelBundle:
    layouts: _BwdDBIncrementAccumulatorLayoutBundle
    copies: _BwdDBIncrementAccumulatorCopyBundle
    tiled_mma: object
    smem_bytes: int


@dataclass(frozen=True)
class _BwdDBIncrementAccumulatorSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class _BwdDBIncrementAccumulatorBase:
    """Shared tensor-core plumbing for increment-side ``dB`` accumulation."""

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], _BwdDBIncrementAccumulatorSupportInfo]
    ] = {}

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        D: int,
        P: int,
        heads: int | None = None,
        bc_groups: int | None = None,
        n_chunks: int | None = None,
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
        self.has_group_geometry = heads is not None
        if self.has_group_geometry:
            self.heads = int(heads)
            self.bc_groups = self.heads if bc_groups is None else int(bc_groups)
            if n_chunks is None:
                raise ValueError(
                    "n_chunks must be specified when grouped BC is active."
                )
            self.n_chunks = int(n_chunks)
            if self.heads <= 0 or self.bc_groups <= 0:
                raise ValueError("heads and bc_groups must be positive.")
            if self.heads % self.bc_groups != 0:
                raise ValueError("bc_groups must divide heads.")
            if self.n_chunks <= 0:
                raise ValueError("n_chunks must be positive.")
            self.heads_per_bc_group = self.heads // self.bc_groups
        else:
            if bc_groups is not None:
                raise ValueError("bc_groups requires heads to be specified.")
            self.heads = 0
            self.bc_groups = 0
            self.n_chunks = 0
            self.heads_per_bc_group = 0
        if cta_tiler is None:
            cta_tiler = (self.L, 64, _default_tc_k_tile(self.P))
        self.cta_tiler = cta_tiler
        self.num_stages = int(num_stages)
        self.atom_layout_mnk = atom_layout_mnk

        self.bM, self.bN, self.bK = map(int, self.cta_tiler)
        if self.bM != self.L:
            raise ValueError("This kernel assumes bM == chunk_size (single tile in M).")
        if self.bN % 2 != 0:
            raise ValueError("bN must be divisible by 2; D stores pairs.")
        if (self.bN // 2) > 32:
            raise ValueError("bN/2 must fit within one warp for the DB epilogue.")
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

    @cute.jit
    def _batch_group_chunk_index(
        self,
        batch_head_chunk_idx: int,
    ):
        if cutlass.const_expr(not self.has_group_geometry):
            return batch_head_chunk_idx
        batch_head = batch_head_chunk_idx // cutlass.Int32(self.n_chunks)
        chunk_index = batch_head_chunk_idx - batch_head * cutlass.Int32(self.n_chunks)
        batch_idx = batch_head // cutlass.Int32(self.heads)
        head_idx = batch_head - batch_idx * cutlass.Int32(self.heads)
        group_idx = head_idx // cutlass.Int32(self.heads_per_bc_group)
        batch_group = batch_idx * cutlass.Int32(self.bc_groups) + group_idx
        return batch_group * cutlass.Int32(self.n_chunks) + chunk_index

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
        return max(4, self._align_up(pad_bytes, 4))

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
    ) -> _BwdDBIncrementAccumulatorSupportInfo:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return _BwdDBIncrementAccumulatorSupportInfo(0, 1)

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

        info = _BwdDBIncrementAccumulatorSupportInfo(
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
    ) -> _BwdDBIncrementAccumulatorLayoutBundle:
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
        return _BwdDBIncrementAccumulatorLayoutBundle(
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
        layouts: _BwdDBIncrementAccumulatorLayoutBundle,
        in_u_dtype: type[cutlass.Numeric],
        in_d_inc_dtype: type[cutlass.Numeric],
        tiled_mma: cute.TiledMma,
    ) -> _BwdDBIncrementAccumulatorCopyBundle:
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
        return _BwdDBIncrementAccumulatorCopyBundle(
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

        @cute.struct
        class SharedStorage:
            u_tile: cute.struct.Align[
                cute.struct.MemRange[in_u_dtype, cute.cosize(u_tile_layout)], 16
            ]
            d_inc_tile: cute.struct.Align[
                cute.struct.MemRange[in_d_inc_dtype, cute.cosize(d_inc_tile_layout)],
                16,
            ]
            # Guard storage covers the accumulator aliasing envelope.
            output_alias_guard: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(output_alias_guard_layout)
                ],
                16,
            ]
            suffix_coeff_sum: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            suffix_coeff_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(suffix_coeff_layout)],
                4,
            ]
            warp_transition_re_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ]
            warp_transition_im_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ]
            warp_transition_re_offset: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
                4,
            ]
            warp_transition_im_offset: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(warp_transition_layout)
                ],
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
        mU: cute.Tensor,
        mDIncDP: cute.Tensor,
    ) -> _BwdDBIncrementAccumulatorKernelBundle:
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
        return _BwdDBIncrementAccumulatorKernelBundle(
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


class BwdDBIncrementAccumulatorAmpere(_BwdDBIncrementAccumulatorBase):
    """Ampere increment accumulator for the backward ``dB`` accumulation path."""

    @cute.jit
    def _validate_operands(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mM: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDIncDP: cute.Tensor,
        mDBTotal: cute.Tensor,
        mDMsumPart: cute.Tensor,
    ):
        value_stream_dtype_ok = (
            mU.element_type
            == mB.element_type
            == mDIncDP.element_type
            == mDBTotal.element_type
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
            == mDBTotal.shape[1]
        )
        if cutlass.const_expr(not value_stream_dtype_ok):
            raise TypeError("U/B/DIncDP/DBTotal must share element type.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/DIncDP/DBTotal must be Float16 or BFloat16.")
        if cutlass.const_expr(not coeff_dtype_ok):
            raise TypeError("M/Kprev/Kcurr/DMsumPart must be Float32.")
        if cutlass.const_expr(not chunk_dims_match):
            raise ValueError(
                "U/B/M/Kprev/Kcurr/DBTotal must share the chunk time dimension."
            )
        if cutlass.const_expr(mB.shape[1] % 2 != 0):
            raise ValueError("B/DBTotal D dimension must be even.")
        if cutlass.const_expr(mDIncDP.shape[0] != mB.shape[1]):
            raise ValueError("DIncDP rows must match the D dimension of B.")
        if cutlass.const_expr(mDIncDP.shape[1] != mU.shape[1]):
            raise ValueError("DIncDP columns must match the P dimension of U.")
        if cutlass.const_expr(mDBTotal.shape[2] != 1):
            raise ValueError("DBTotal must have singleton scan dimension 2.")
        if cutlass.const_expr(mDBTotal.shape[3] != mB.shape[1]):
            raise ValueError("DBTotal D dimension must match B.")
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
        mDBTotal: cute.Tensor,
        mDMsumPart: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mU, mDIncDP)
        grid_dim = cute.ceil_div(
            (mDBTotal.shape[1], mDBTotal.shape[3], mDBTotal.shape[0]),
            (self.bM, self.bN, 1),
        )
        launch_kwargs = {
            "grid": (
                cute.size(grid_dim[0]),
                cute.size(grid_dim[1]),
                cute.size(grid_dim[2]),
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
            mDBTotal,
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
    def __call__(
        self,
        mU: cute.Tensor,  # (L, P, BHC)
        mB: cute.Tensor,  # (L, D, BGC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDIncDP: cute.Tensor,  # (D, P, BHC)
        mDBTotal: cute.Tensor,  # (BHC, L, 1, D)
        mDMsumPart: cute.Tensor,  # (2, L, n_d_tiles, BHC)
    ):
        self._validate_and_launch(
            mU, mB, mM, mKprev, mKcurr, mDIncDP, mDBTotal, mDMsumPart
        )

    @cute.jit
    def call_on_stream(
        self,
        mU: cute.Tensor,  # (L, P, BHC)
        mB: cute.Tensor,  # (L, D, BGC)
        mM: cute.Tensor,  # (2, L, BHC)
        mKprev: cute.Tensor,  # (2, L, BHC)
        mKcurr: cute.Tensor,  # (2, L, BHC)
        mDIncDP: cute.Tensor,  # (D, P, BHC)
        mDBTotal: cute.Tensor,  # (BHC, L, 1, D)
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
            mDBTotal,
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
        mDBTotal: cute.Tensor,
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
        batch_group_chunk_idx = self._batch_group_chunk_index(batch_head_chunk_idx)

        tile_col_start = block_col_idx * self.bN
        tiler_coord = (cutlass.Int32(0), block_col_idx, None)

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

        r_u_mma = tiled_mma.make_fragment_A(t_u_mma_smem[None, None, None, 0])
        r_d_inc_mma = tiled_mma.make_fragment_B(t_d_inc_mma_smem[None, None, None, 0])
        r_db_accum = tiled_mma.make_fragment_C(t_db_mma_smem)
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
        pipeline.sync()

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

        pipeline.sync()

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

            pipeline.sync()

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

        pipeline.sync()

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

        pipeline.sync()

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

        pipeline.sync()

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
            pipeline.sync()

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
        pipeline.sync()

        cute.autovec_copy(r_db_accum, t_db_mma_smem)
        pipeline.sync()

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
                    mB[time_index, global_d_pair_start + 0, batch_group_chunk_idx].to(
                        cutlass.Float32
                    )
                )
                b_im = cutlass.Float32(
                    mB[time_index, global_d_pair_start + 1, batch_group_chunk_idx].to(
                        cutlass.Float32
                    )
                )

                db_coeff_re = s_suffix_coeff_sum[time_index, 0]
                db_coeff_im = s_suffix_coeff_sum[time_index, 1]
                rotated_db_re = db_coeff_re * d_b_sum_re + db_coeff_im * d_b_sum_im
                rotated_db_im = db_coeff_re * d_b_sum_im - db_coeff_im * d_b_sum_re
                increment_re = cutlass.Float32(
                    rotated_db_re.to(mDBTotal.element_type).to(cutlass.Float32)
                )
                increment_im = cutlass.Float32(
                    rotated_db_im.to(mDBTotal.element_type).to(cutlass.Float32)
                )
                total_re = (
                    cutlass.Float32(
                        mDBTotal[
                            batch_head_chunk_idx,
                            time_index,
                            0,
                            global_d_pair_start + 0,
                        ].to(cutlass.Float32)
                    )
                    + increment_re
                )
                total_im = (
                    cutlass.Float32(
                        mDBTotal[
                            batch_head_chunk_idx,
                            time_index,
                            0,
                            global_d_pair_start + 1,
                        ].to(cutlass.Float32)
                    )
                    + increment_im
                )
                mDBTotal[
                    batch_head_chunk_idx,
                    time_index,
                    0,
                    global_d_pair_start + 0,
                ] = total_re.to(mDBTotal.element_type)
                mDBTotal[
                    batch_head_chunk_idx,
                    time_index,
                    0,
                    global_d_pair_start + 1,
                ] = total_im.to(mDBTotal.element_type)

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


__all__ = [
    "BwdDBAmpere",
    "BwdDBIncrementAccumulatorAmpere",
]
