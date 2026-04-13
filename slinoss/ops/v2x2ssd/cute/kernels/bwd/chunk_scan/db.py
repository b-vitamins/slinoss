"""CuTe backward ``db`` kernel for the ``v2x2ssd`` chunk-scan stage.

``ChunkScanBwdDBAmpere`` is the live Ampere tensor-core implementation for the
backward key-gradient slice of the chunk-scan stage.

It consumes the public stage inputs plus the stage-local upstream gradient, then
reconstructs prefix metadata from ``M``, sweeps tiles in reverse causal order,
forms the current and shifted score blocks from ``dY @ V``, contracts those
score blocks against scaled/rotated ``Q``, applies the two complex taps from
``K``, and writes ``dB`` plus the cross-chunk boundary carry ``dB_prev``.

Tensor contracts:

- ``U``: ``(BHC, L, 1, P)`` fp16/bf16 value input
- ``B``: ``(BHC, L, 1, D)`` fp16/bf16 key input with packed complex pairs
- ``C``: ``(BHC, L, 1, D)`` fp16/bf16 query input with packed complex pairs
- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for previous/current
  backward passes
- ``dOut``: ``(BHC, L, 1, P)`` fp16/bf16 upstream stage gradient
- ``U_prev0``: ``(BH, P)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(BH, D)`` fp16/bf16 chunk-0 boundary key row
- ``dB``: ``(BHC, L, 1, D)`` fp16/bf16 output key gradient
- ``dB_prev``: ``(BHC, D)`` fp16/bf16 boundary carry for the previous chunk

The trailing ``D`` dimension stores packed complex pairs, so ``D`` must be even
and conceptually corresponds to ``2 * N``.
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
    apply_complex_tap_adjoint,
    clamp_nonpositive_prefix_log,
    complex_mul,
    conj_mul_phase,
    safe_cast_to_dtype,
)


@dataclass(frozen=True)
class ChunkScanBwdDBLayoutBundle:
    query_layout: object
    grad_output_layout: object
    key_grad_layout: object
    value_layout: object
    score_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDBCopyBundle:
    gmem_tiled_copy_d: object
    gmem_tiled_copy_p: object
    gmem_tiled_store_d: object


@dataclass(frozen=True)
class ChunkScanBwdDBKernelBundle:
    layouts: ChunkScanBwdDBLayoutBundle
    copies: ChunkScanBwdDBCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int


@dataclass(frozen=True)
class ChunkScanBwdDBSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkScanBwdDBAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``db`` slice.

    This kernel owns the chunk-local key-gradient path. It rebuilds the prefix
    metadata from raw packed ``M``, forms the current and shifted score tiles
    from ``dOut @ U``, contracts those scores against the rotated query tiles,
    applies the two complex taps from ``K``, and writes ``dB`` plus the reverse
    chunk-boundary carry ``dB_prev``.
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkScanBwdDBSupportInfo]
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
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * d_stage_width * in_bytes, 16),
                (kv_tile * p_tile * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (kv_tile * kv_tile * in_bytes, 16),
                (self.L * 2 * 4, 16),
                (self.L * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (d_padded * 4, 8),
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
    ) -> ChunkScanBwdDBSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkScanBwdDBSupportInfo(0, 1)

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

        info = ChunkScanBwdDBSupportInfo(
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

    def _make_layout_bundle(self) -> ChunkScanBwdDBLayoutBundle:
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

        return ChunkScanBwdDBLayoutBundle(
            query_layout=query_layout,
            grad_output_layout=grad_output_layout,
            key_grad_layout=key_grad_layout,
            value_layout=value_layout,
            score_layout=score_layout,
        )

    def _make_copy_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDBCopyBundle:
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
        return ChunkScanBwdDBCopyBundle(
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
        layouts: ChunkScanBwdDBLayoutBundle,
    ):
        phase_layout = cute.make_layout((self.L, 2), stride=(2, 1))
        tap_curr_layout = cute.make_layout((self.kv_tile, 2), stride=(2, 1))
        carry_layout = cute.make_layout((self.D_padded,), stride=(1,))
        row_layout = cute.make_layout((self.L,), stride=(1,))

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "query_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)], 16
            ],
            "grad_output_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.grad_output_layout)],
                16,
            ],
            "key_grad_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.key_grad_layout)], 16
            ],
            "value_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.value_layout)], 16
            ],
            "score_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)], 16
            ],
            "score_cache": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.score_layout)], 16
            ],
            "phase_full": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 16
            ],
            "tap_prev_full": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)], 16
            ],
            "tap_curr_tile": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ],
            "db_carry": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(carry_layout)], 8
            ],
            "row_scale": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ],
            "inv_row_scale": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(row_layout)], 4
            ],
            "dm_curr": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ],
            "dm_prev": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tap_curr_layout)], 16
            ],
            "warp_log_total": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 4
            ],
            "warp_log_offset": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps], 4
            ],
            "warp_phase_total": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
            ],
            "warp_phase_offset": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
            ],
        }
        return cute.struct(SharedStorage)

    def _make_kernel_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDBKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        return ChunkScanBwdDBKernelBundle(
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
        cute.arch.barrier()

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
        cute.arch.barrier()

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
                next_phase_re, next_phase_im = complex_mul(
                    running_phase_re,
                    running_phase_im,
                    total_phase_re,
                    total_phase_im,
                )
                running_log = running_log + total_log
                running_phase_re = next_phase_re
                running_phase_im = next_phase_im
        cute.arch.barrier()

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
        cute.arch.barrier()

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
        cute.arch.barrier()

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
                s_query[row_local, d_local + 0] = safe_cast_to_dtype(yr, out_dtype)
                s_query[row_local, d_local + 1] = safe_cast_to_dtype(yi, out_dtype)
            idx = idx + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _scale_staged_query_tile(
        self,
        s_query: cute.Tensor,
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
                scale = cutlass.Float32(s_row_scale[row])
                xr = cutlass.Float32(
                    s_query[row_local, d_local + 0].to(cutlass.Float32)
                )
                xi = cutlass.Float32(
                    s_query[row_local, d_local + 1].to(cutlass.Float32)
                )
                s_query[row_local, d_local + 0] = safe_cast_to_dtype(
                    xr * scale, out_dtype
                )
                s_query[row_local, d_local + 1] = safe_cast_to_dtype(
                    xi * scale, out_dtype
                )
            idx = idx + self.num_threads
        cute.arch.barrier()

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
            cute.arch.barrier()

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
        cute.arch.barrier()

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
    def _accumulate_dm_for_tile_rows(
        self,
        s_db_curr: cute.Tensor,
        s_db_prev: cute.Tensor,
        s_phase: cute.Tensor,
        mB: cute.Tensor,
        mB_prev0: cute.Tensor,
        s_dm_curr: cute.Tensor,
        s_dm_prev: cute.Tensor,
        *,
        batch_group_chunk: int,
        batch_group: int,
        chunk_index: int,
        n_tile_start: int,
        d_col_base: int,
        stage_width: int,
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

                pair_local = lane
                while pair_local < cutlass.Int32(nvec_stage):
                    d_local = pair_local * 2
                    d = cutlass.Int32(d_col_base) + d_local
                    if d + cutlass.Int32(1) < cutlass.Int32(self.D):
                        bxr_curr = cutlass.Float32(
                            s_db_curr[row_local, d_local + 0].to(cutlass.Float32)
                        )
                        bxi_curr = cutlass.Float32(
                            s_db_curr[row_local, d_local + 1].to(cutlass.Float32)
                        )
                        gx_curr, gy_curr = conj_mul_phase(bxr_curr, bxi_curr, pr, pi)
                        br_curr = cutlass.Float32(
                            mB[batch_group_chunk, row, 0, d + 0].to(cutlass.Float32)
                        )
                        bi_curr = cutlass.Float32(
                            mB[batch_group_chunk, row, 0, d + 1].to(cutlass.Float32)
                        )
                        dmy_curr_re = (
                            dmy_curr_re + gx_curr * br_curr - gy_curr * bi_curr
                        )
                        dmy_curr_im = (
                            dmy_curr_im + gx_curr * bi_curr + gy_curr * br_curr
                        )

                        bxr_prev = cutlass.Float32(
                            s_db_prev[row_local, d_local + 0].to(cutlass.Float32)
                        )
                        bxi_prev = cutlass.Float32(
                            s_db_prev[row_local, d_local + 1].to(cutlass.Float32)
                        )
                        gx_prev, gy_prev = conj_mul_phase(bxr_prev, bxi_prev, pr, pi)
                        br_prev = cutlass.Float32(0.0)
                        bi_prev = cutlass.Float32(0.0)
                        if row > cutlass.Int32(0):
                            br_prev = cutlass.Float32(
                                mB[
                                    batch_group_chunk, row - cutlass.Int32(1), 0, d + 0
                                ].to(cutlass.Float32)
                            )
                            bi_prev = cutlass.Float32(
                                mB[
                                    batch_group_chunk, row - cutlass.Int32(1), 0, d + 1
                                ].to(cutlass.Float32)
                            )
                        elif chunk_index == cutlass.Int32(0):
                            br_prev = cutlass.Float32(
                                mB_prev0[batch_group, d + 0].to(cutlass.Float32)
                            )
                            bi_prev = cutlass.Float32(
                                mB_prev0[batch_group, d + 1].to(cutlass.Float32)
                            )
                        else:
                            br_prev = cutlass.Float32(
                                mB[
                                    batch_group_chunk - cutlass.Int32(1),
                                    cutlass.Int32(self.L - 1),
                                    0,
                                    d + 0,
                                ].to(cutlass.Float32)
                            )
                            bi_prev = cutlass.Float32(
                                mB[
                                    batch_group_chunk - cutlass.Int32(1),
                                    cutlass.Int32(self.L - 1),
                                    0,
                                    d + 1,
                                ].to(cutlass.Float32)
                            )
                        dmy_prev_re = (
                            dmy_prev_re + gx_prev * br_prev - gy_prev * bi_prev
                        )
                        dmy_prev_im = (
                            dmy_prev_im + gx_prev * bi_prev + gy_prev * br_prev
                        )
                    pair_local = pair_local + cutlass.Int32(32)

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
        cute.arch.barrier()

    @cute.jit
    def _apply_tap_adjoint_and_add_carry(
        self,
        s_db_output: cute.Tensor,
        s_db_curr: cute.Tensor,
        s_db_prev: cute.Tensor,
        s_tap_curr: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_db_carry: cute.Tensor,
        *,
        n_tile_start: int,
        d_col_base: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nvec_stage = stage_width // 2
        total_pairs = self.kv_tile * nvec_stage
        idx = tidx
        while cute.elem_less(idx, total_pairs):
            row_local = idx // nvec_stage
            pair_local = idx - row_local * nvec_stage
            row = cutlass.Int32(n_tile_start) + row_local
            if row < cutlass.Int32(self.L):
                d_local = pair_local * 2
                curr_re = cutlass.Float32(
                    s_db_curr[row_local, d_local + 0].to(cutlass.Float32)
                )
                curr_im = cutlass.Float32(
                    s_db_curr[row_local, d_local + 1].to(cutlass.Float32)
                )
                tap_curr_re = cutlass.Float32(s_tap_curr[row_local, 0])
                tap_curr_im = cutlass.Float32(s_tap_curr[row_local, 1])
                out_re, out_im = apply_complex_tap_adjoint(
                    curr_re, curr_im, tap_curr_re, tap_curr_im
                )

                carry_re = cutlass.Float32(0.0)
                carry_im = cutlass.Float32(0.0)
                if row + cutlass.Int32(1) < cutlass.Int32(self.L):
                    if row_local + cutlass.Int32(1) < cutlass.Int32(self.kv_tile):
                        row_next = row + cutlass.Int32(1)
                        prev_re = cutlass.Float32(
                            s_db_prev[row_local + cutlass.Int32(1), d_local + 0].to(
                                cutlass.Float32
                            )
                        )
                        prev_im = cutlass.Float32(
                            s_db_prev[row_local + cutlass.Int32(1), d_local + 1].to(
                                cutlass.Float32
                            )
                        )
                        tap_prev_re = cutlass.Float32(s_tap_prev[row_next, 0])
                        tap_prev_im = cutlass.Float32(s_tap_prev[row_next, 1])
                        carry_re, carry_im = apply_complex_tap_adjoint(
                            prev_re, prev_im, tap_prev_re, tap_prev_im
                        )
                    else:
                        carry_re = cutlass.Float32(
                            s_db_carry[d_col_base + d_local + 0].to(cutlass.Float32)
                        )
                        carry_im = cutlass.Float32(
                            s_db_carry[d_col_base + d_local + 1].to(cutlass.Float32)
                        )

                s_db_output[row_local, d_local + 0] = safe_cast_to_dtype(
                    out_re + carry_re, out_dtype
                )
                s_db_output[row_local, d_local + 1] = safe_cast_to_dtype(
                    out_im + carry_im, out_dtype
                )
            idx = idx + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _clear_db_carry(self, s_db_carry: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        for it in cutlass.range_constexpr(
            (self.D_padded + self.num_threads - 1) // self.num_threads
        ):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D_padded):
                s_db_carry[d] = cutlass.Float32(0.0)
        cute.arch.barrier()

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
        cute.arch.barrier()

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
        cute.arch.barrier()

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
        mDU: cute.Tensor,
        mDB: cute.Tensor,
        mDU_prev: cute.Tensor,
        mDB_prev: cute.Tensor,
        mDLogPrefix: cute.Tensor,
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
        if cutlass.const_expr(mDU.shape[1] != self.L or mDU.shape[2] != 1):
            raise ValueError("dU must be (BHC, L, 1, P).")
        if cutlass.const_expr(mDB.shape[1] != self.L or mDB.shape[2] != 1):
            raise ValueError("dB must be (BHC, L, 1, D).")
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
            mDU,
            mDB,
            mDU_prev,
            mDB_prev,
            mDLogPrefix,
            mDMprev,
            mDMcurr,
            layouts.query_layout,
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
        single_tile_cache = num_n_tiles == 1

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)

        s_query_tile = storage.query_tile.get_tensor(query_layout)
        s_grad_output_stage0 = storage.grad_output_tile.get_tensor(grad_output_layout)
        s_score_prev = cute.make_tensor(
            s_grad_output_stage0.iterator.align(16), score_layout
        )
        s_key_grad_tile = storage.key_grad_tile.get_tensor(key_grad_layout)
        s_value_stage0 = storage.value_tile.get_tensor(value_layout)
        s_score_block = storage.score_tile.get_tensor(score_layout)
        s_score_cache = storage.score_cache.get_tensor(score_layout)
        s_grad_output_stage1 = cute.make_tensor(
            s_key_grad_tile.iterator.align(16), grad_output_layout
        )
        s_value_stage1 = cute.make_tensor(
            s_score_block.iterator.align(16), value_layout
        )
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
        coord_value = cute.make_identity_tensor(mU.layout.shape)
        coord_query = cute.make_identity_tensor(mC.layout.shape)
        coord_dout = cute.make_identity_tensor(mDOut.layout.shape)
        coord_score_tile_base = coord_score[batch_head_chunk, None, 0, None]
        coord_db_tile_base = coord_db[batch_head_chunk, None, 0, None]
        t_query_smem = gmem_thr_copy_d.partition_D(s_query_tile)
        t_grad_output_stage0_smem = gmem_thr_copy_p.partition_D(s_grad_output_stage0)
        t_grad_output_stage1_smem = gmem_thr_copy_p.partition_D(s_grad_output_stage1)
        t_value_stage0_smem = gmem_thr_copy_p.partition_D(s_value_stage0)
        t_value_stage1_smem = gmem_thr_copy_p.partition_D(s_value_stage1)

        t_score_block = thr_mma.make_fragment_A(thr_mma.partition_A(s_score_block))
        t_score_block_smem = thr_copy_lhs.partition_S(s_score_block)
        t_score_block_view = thr_copy_lhs.retile(t_score_block)
        t_score_cache = thr_mma.make_fragment_A(thr_mma.partition_A(s_score_cache))
        t_score_cache_smem = thr_copy_lhs.partition_S(s_score_cache)
        t_score_cache_view = thr_copy_lhs.retile(t_score_cache)
        t_score_prev = thr_mma.make_fragment_A(thr_mma.partition_A(s_score_prev))
        t_score_prev_smem = thr_copy_lhs.partition_S(s_score_prev)
        t_score_prev_view = thr_copy_lhs.retile(t_score_prev)
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
        self._clear_db_carry(s_db_carry)

        for n_tile_rev in cutlass.range_constexpr(num_n_tiles):
            n_tile_index = (num_n_tiles - 1) - n_tile_rev
            n_tile_start = n_tile_index * kv_tile
            num_m_tiles = num_n_tiles - n_tile_index

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
                acc_dk_curr = cute.make_rmem_tensor(acc_shape_tile_d, cutlass.Float32)
                acc_dk_curr.fill(0.0)
                acc_dk_prev = cute.make_rmem_tensor(acc_shape_tile_d, cutlass.Float32)
                acc_dk_prev.fill(0.0)

                for m_tile_offset in cutlass.range_constexpr(num_m_tiles):
                    m_tile_index = n_tile_index + m_tile_offset
                    m_tile_start = m_tile_index * kv_tile

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
                    self._rotate_staged_query_tile_from_prefix(
                        s_query_tile,
                        s_phase,
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

                    if (not single_tile_cache) or d_stage_index == 0:
                        acc_score_curr = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_score_curr.fill(0.0)
                        self._accumulate_score_block_from_p_tiles(
                            score_pipeline,
                            acc_score_curr,
                            m_tile_index=m_tile_index,
                            n_tile_index=n_tile_index,
                            use_shifted_values=False,
                        )
                        self._store_causal_score_block(
                            acc_score_curr,
                            t_score_coord_mn,
                            s_score_cache if single_tile_cache else s_score_block,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_tile_start,
                            out_dtype=mU.element_type,
                        )

                    self._scale_staged_query_tile(
                        s_query_tile,
                        s_row_scale,
                        m_tile_start=m_tile_start,
                        stage_width=d_stage_width,
                        out_dtype=mU.element_type,
                    )
                    if single_tile_cache:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_curr,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_cache_smem,
                            t_score_cache_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_cache,
                            t_query_transposed,
                        )
                    else:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_curr,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_block_smem,
                            t_score_block_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_block,
                            t_query_transposed,
                        )

                    if (not single_tile_cache) or d_stage_index == 0:
                        acc_score_prev = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_score_prev.fill(0.0)
                        self._accumulate_score_block_from_p_tiles(
                            score_pipeline,
                            acc_score_prev,
                            m_tile_index=m_tile_index,
                            n_tile_index=n_tile_index,
                            use_shifted_values=True,
                        )
                        self._store_causal_score_block(
                            acc_score_prev,
                            t_score_coord_mn,
                            s_score_prev if single_tile_cache else s_score_block,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_tile_start,
                            out_dtype=mU.element_type,
                        )
                    if single_tile_cache:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_prev,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_prev_smem,
                            t_score_prev_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_prev,
                            t_query_transposed,
                        )
                    else:
                        self._accumulate_key_grad_from_staged_score_block(
                            tiled_mma,
                            acc_dk_prev,
                            smem_tiled_copy_lhs,
                            smem_tiled_copy_query_t,
                            t_score_block_smem,
                            t_score_block_view,
                            t_query_transposed_smem,
                            t_query_transposed_view,
                            t_score_block,
                            t_query_transposed,
                        )

                coord_db_stage = cute.local_tile(
                    coord_db_tile_base,
                    (kv_tile, d_stage_width),
                    (n_tile_index, d_stage_index),
                )
                t_db_stage_coord = thr_mma.partition_C(coord_db_stage)
                t_db_stage_coord_mn = self._make_accumulator_mn_view(t_db_stage_coord)

                t_db_prev_smem = thr_mma.partition_C(s_query_tile)
                t_db_prev_reg = cute.make_fragment_like(t_db_prev_smem, mU.element_type)
                t_db_prev_reg.fill(0.0)
                t_db_prev_reg_mn = self._make_accumulator_mn_view(t_db_prev_reg)
                t_db_curr_smem = thr_mma.partition_C(s_key_grad_tile)
                t_db_curr_reg = cute.make_fragment_like(t_db_curr_smem, mU.element_type)
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
                cute.arch.barrier()

                self._accumulate_dm_for_tile_rows(
                    s_key_grad_tile,
                    s_query_tile,
                    s_phase,
                    mB,
                    mB_prev0,
                    s_dm_curr,
                    s_dm_prev,
                    batch_group_chunk=batch_group_chunk,
                    batch_group=batch_group,
                    chunk_index=chunk_index,
                    n_tile_start=n_tile_start,
                    d_col_base=d_col_base,
                    stage_width=d_stage_width,
                )

                self._apply_tap_adjoint_and_add_carry(
                    s_key_grad_tile,
                    s_key_grad_tile,
                    s_query_tile,
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


__all__ = ["ChunkScanBwdDBAmpere"]
