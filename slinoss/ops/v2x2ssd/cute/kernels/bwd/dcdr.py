"""CuTe backward fused ``dC``/``dR`` kernel for ``v2x2ssd``.

``BwdDCDRAmpere`` reconstructs per-row magnitude and phase metadata from
``M``, accumulates causal score tiles from ``dOut`` and shifted ``U`` rows,
projects those scores into ``dC`` through the tapped key path, and folds the
packed ``dlogprefix`` reduction into the same traversal that writes ``dR``.

Tensor contracts:

- ``U``: ``(BHC, L, 1, P)`` fp16/bf16 value input
- ``B``: ``(BHC, L, 1, D)`` fp16/bf16 key-side packed complex input
- ``C``: ``(BHC, L, 1, D)`` fp16/bf16 query-side packed complex input
- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for the previous/current
  diagonal passes
- ``dOut``: ``(BHC, L, 1, P)`` fp16/bf16 output gradient
- ``U_prev0``: ``(BH, P)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(BH, D)`` fp16/bf16 chunk-0 boundary key row
- ``Z0``: ``(BHC, P, D)`` fp32 or fp16/bf16 packed complex initial state
- ``dC``: ``(BHC, L, 1, D)`` fp16/bf16 query gradient output
- ``dlogprefix``: ``(BHC, L)`` fp32 packed log-magnitude gradient
- ``dR``: ``(BHC, L, 4)`` fp32 packed rotation-row gradient

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
import cutlass.pipeline as pipeline
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
class BwdDCDRLayoutBundle:
    u_prev_layout: object
    b_prev_layout: object
    grad_output_layout: object
    value_layout: object
    key_layout: object
    query_layout: object
    score_layout: object
    z0_layout: object
    full_scale_layout: object
    full_inv_scale_layout: object
    full_phase_layout: object
    full_tap_layout: object
    tile_prefix_log_layout: object
    tile_prefix_phase_layout: object
    row_scale_layout: object
    row_inv_scale_layout: object
    column_phase_layout: object
    tap_layout: object
    row_dlogprefix_layout: object
    row_dr_layout: object
    dlogprefix_accum_layout: object
    column_accumulator_layout: object


@dataclass(frozen=True)
class BwdDCDRCopyBundle:
    gmem_tiled_copy_d_async: object
    gmem_tiled_copy_p: object
    gmem_tiled_store_d: object


@dataclass(frozen=True)
class BwdDCDRKernelBundle:
    layouts: BwdDCDRLayoutBundle
    copies: BwdDCDRCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.smem_bytes


@dataclass(frozen=True)
class BwdDCDRSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class BwdDCDRAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dcdr`` slice.

    This kernel owns the query-gradient and rotation-metadata path. It rebuilds
    row-wise magnitude and phase metadata from ``M``, accumulates the causal
    score tiles from ``dOut`` and shifted ``U`` rows, projects those scores
    through the tapped key path into ``dC``, and accumulates ``dlogprefix`` and
    ``dR`` without a separate log-prefix launch.
    """

    _SUPPORT_INFO_CACHE: ClassVar[dict[tuple[object, ...], BwdDCDRSupportInfo]] = {}

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

        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.L % self.kv_tile != 0:
            raise ValueError("chunk_size must be a multiple of 32.")
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads != 128:
            raise ValueError("num_threads must be 128 for the 32x32 tensorop tile.")

        self.mma_inst_shape = (16, 8, 16)
        self.warp_layout_mnk = (2, 2, 1)
        self.atom_layout_mnk = self.warp_layout_mnk

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

    def _smem_block_size_d(self) -> int:
        return 64 if self.D_padded % 64 == 0 else 32

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

    def _make_layout_bundle(self) -> BwdDCDRLayoutBundle:
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        n_tiles = self.num_kv_tiles

        smem_d_block_size = self._smem_block_size_d()
        swizzle_bits_d = self._swizzle_bits(smem_d_block_size)
        s_d_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_d, 3, 3),
            0,
            cute.make_layout((8, smem_d_block_size), stride=(smem_d_block_size, 1)),
        )
        key_layout = cute.tile_to_shape(
            s_d_layout_atom, (kv_tile, smem_d_block_size), (0, 1)
        )
        z0_layout = cute.tile_to_shape(
            s_d_layout_atom, (p_tile, smem_d_block_size), (0, 1)
        )

        smem_p_block_size = p_tile
        swizzle_bits_p = self._swizzle_bits(smem_p_block_size)
        s_p_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_p, 3, 3),
            0,
            cute.make_layout((8, smem_p_block_size), stride=(smem_p_block_size, 1)),
        )
        grad_output_layout = cute.tile_to_shape(
            s_p_layout_atom, (kv_tile, self.P_padded), (0, 1)
        )
        value_layout = cute.tile_to_shape(
            s_p_layout_atom, (kv_tile, 2 * self.P_padded), (0, 1)
        )

        smem_score_block_size = kv_tile
        swizzle_bits_score = self._swizzle_bits(smem_score_block_size)
        s_score_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_score, 3, 3),
            0,
            cute.make_layout(
                (8, smem_score_block_size), stride=(smem_score_block_size, 1)
            ),
        )
        score_layout = cute.tile_to_shape(
            s_score_layout_atom, (kv_tile, kv_tile), (0, 1)
        )

        return BwdDCDRLayoutBundle(
            u_prev_layout=cute.make_layout((self.P,), stride=(1,)),
            b_prev_layout=cute.make_layout((self.D,), stride=(1,)),
            grad_output_layout=grad_output_layout,
            value_layout=value_layout,
            key_layout=key_layout,
            query_layout=key_layout,
            score_layout=score_layout,
            z0_layout=z0_layout,
            full_scale_layout=cute.make_layout((self.L,), stride=(1,)),
            full_inv_scale_layout=cute.make_layout((self.L,), stride=(1,)),
            full_phase_layout=cute.make_layout((self.L, 2), stride=(2, 1)),
            full_tap_layout=cute.make_layout((self.L, 4), stride=(4, 1)),
            tile_prefix_log_layout=cute.make_layout((n_tiles,), stride=(1,)),
            tile_prefix_phase_layout=cute.make_layout((n_tiles, 2), stride=(2, 1)),
            row_scale_layout=cute.make_layout((kv_tile,), stride=(1,)),
            row_inv_scale_layout=cute.make_layout((kv_tile,), stride=(1,)),
            column_phase_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            tap_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            row_dlogprefix_layout=cute.make_layout((kv_tile,), stride=(1,)),
            row_dr_layout=cute.make_layout((kv_tile, 4), stride=(4, 1)),
            dlogprefix_accum_layout=cute.make_layout((self.L,), stride=(1,)),
            column_accumulator_layout=cute.make_layout(
                (self.num_warps, 4, 4), stride=(16, 4, 1)
            ),
        )

    def _compute_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        in_bytes = in_dtype.width // 8
        kv_tile = self.kv_tile
        d_block = self._smem_block_size_d()
        n_tiles = self.num_kv_tiles
        return self._struct_size_bytes(
            [
                (kv_tile * self.P_padded * in_bytes, 16),
                (kv_tile * (2 * self.P_padded) * in_bytes, 16),
                (kv_tile * d_block * in_bytes, 16),
                (kv_tile * d_block * in_bytes, 16),
                (self.P * in_bytes, 16),
                (self.D * in_bytes, 16),
                (self.L * 4, 4),
                (self.L * 4, 4),
                (self.L * 2 * 4, 16),
                (self.L * 4 * 4, 16),
                (n_tiles * 4, 4),
                (n_tiles * 2 * 4, 16),
                (n_tiles * 4, 4),
                (n_tiles * 2 * 4, 16),
                (kv_tile * 4, 4),
                (kv_tile * 4, 4),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 2 * 4, 16),
                (kv_tile * 4, 4),
                (kv_tile * 4 * 4, 16),
                (self.L * 4, 4),
                (self.num_warps * 4 * 4 * 4, 16),
            ]
        )

    def support_info(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> BwdDCDRSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return BwdDCDRSupportInfo(0, 1)

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

        info = BwdDCDRSupportInfo(
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            required_smem_bytes=self._compute_smem_bytes(in_dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def _make_copy_bundle(self, in_dtype: type[cutlass.Numeric]) -> BwdDCDRCopyBundle:
        universal_copy_bits = 128
        async_elems_in = universal_copy_bits // in_dtype.width
        smem_k_block_size_d = self._smem_block_size_d()
        t_d_shape_dim_1 = smem_k_block_size_d // async_elems_in
        t_d_layout = cute.make_layout(
            (self.num_threads // t_d_shape_dim_1, t_d_shape_dim_1),
            stride=(t_d_shape_dim_1, 1),
        )
        t_p_shape_dim_1 = self.p_tile // async_elems_in
        t_p_layout = cute.make_layout(
            (self.num_threads // t_p_shape_dim_1, t_p_shape_dim_1),
            stride=(t_p_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))

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
        return BwdDCDRCopyBundle(
            gmem_tiled_copy_d_async=cute.make_tiled_copy_tv(
                atom_async_copy_in, t_d_layout, v_in_layout
            ),
            gmem_tiled_copy_p=cute.make_tiled_copy_tv(
                atom_async_copy_in, t_p_layout, v_in_layout
            ),
            gmem_tiled_store_d=cute.make_tiled_copy_tv(
                atom_universal_copy_out, t_d_layout, v_in_layout
            ),
        )

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        op = cute.nvgpu.warp.MmaF16BF16Op(in_dtype, self.acc_dtype, (16, 8, 16))
        permutation_mnk = (
            self.atom_layout_mnk[0] * 16,
            self.atom_layout_mnk[1] * 8 * 2,
            self.atom_layout_mnk[2] * 16,
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

    def _make_kernel_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> BwdDCDRKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        return BwdDCDRKernelBundle(
            layouts=layouts,
            copies=self._make_copy_bundle(in_dtype),
            tiled_mma=self._make_tiled_mma(in_dtype),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: BwdDCDRLayoutBundle,
    ):
        @cute.struct
        class SharedStorage:
            dout_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.grad_output_layout)],
                16,
            ]
            value_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.value_layout)], 16
            ]
            key_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.key_layout)], 16
            ]
            query_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)], 16
            ]
            u_prev_row: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.u_prev_layout)], 16
            ]
            b_prev_row: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.b_prev_layout)], 16
            ]
            full_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_scale_layout)
                ],
                4,
            ]
            full_inv_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_inv_scale_layout)
                ],
                4,
            ]
            full_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_phase_layout)
                ],
                16,
            ]
            full_tap: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_tap_layout)
                ],
                16,
            ]
            tile_end_log: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ]
            tile_end_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ]
            tile_offset_log: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ]
            tile_offset_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ]
            row_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_scale_layout)
                ],
                4,
            ]
            row_inv_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_inv_scale_layout)
                ],
                4,
            ]
            phase_col: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.column_phase_layout)
                ],
                16,
            ]
            tap_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(layouts.tap_layout)],
                16,
            ]
            tap_curr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(layouts.tap_layout)],
                16,
            ]
            row_dlogprefix: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_dlogprefix_layout)
                ],
                4,
            ]
            row_dr: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_dr_layout)
                ],
                16,
            ]
            dlogprefix_accum: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.dlogprefix_accum_layout)
                ],
                4,
            ]
            column_accum: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.column_accumulator_layout)
                ],
                16,
            ]

        return SharedStorage

    def can_implement(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> bool:
        return self.support_info(in_dtype, device_index=device_index).supported

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
        pipeline.sync()

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
    def _copy_shifted_rows_with_zero_fill(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        t_copy_src: cute.Tensor,
        t_copy_dst: cute.Tensor,
        t_coord: cute.Tensor,
        copy_pred: cute.Tensor,
        row_limit: int,
    ):
        for idx in cutlass.range_constexpr(cute.size(t_copy_dst.shape[1])):
            row = cutlass.Int32(t_coord[0, idx, 0][1])
            if cute.elem_less(row, row_limit):
                if cute.elem_less(cutlass.Int32(-1), row):
                    cute.copy(
                        gmem_tiled_copy,
                        t_copy_src[None, idx, None],
                        t_copy_dst[None, idx, None],
                        pred=copy_pred[None, idx, None],
                    )
                else:
                    t_copy_dst[None, idx, None].fill(0)
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
    def _compute_phase_prefix_metadata(self, prefix_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        eps = cutlass.Float32(1.0e-20)
        one = cutlass.Float32(1.0)
        num_warps = self.num_warps
        n_tiles = self.num_kv_tiles

        for tile in cutlass.range_constexpr(n_tiles):
            if warp == cutlass.Int32(tile % num_warps) and lane < cutlass.Int32(
                self.kv_tile
            ):
                seq_idx = cutlass.Int32(tile * self.kv_tile) + lane
                mr = cutlass.Float32(
                    prefix_state.m_transition[prefix_state.batch_head_chunk, seq_idx, 0]
                )
                mi = cutlass.Float32(
                    prefix_state.m_transition[prefix_state.batch_head_chunk, seq_idx, 1]
                )
                mag2 = cutlass.Float32(mr * mr + mi * mi + eps)
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
                phase_re = mr * inv_mag
                phase_im = mi * inv_mag
                logp = cutlass.Float32(
                    cute.math.log2(mag2, fastmath=False)
                    * cutlass.Float32(0.25 / LOG2_E)
                )
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
                prefix_state.s_scale_full[seq_idx] = logp
                prefix_state.s_phase_full[seq_idx, 0] = phase_re
                prefix_state.s_phase_full[seq_idx, 1] = phase_im
                if lane == cutlass.Int32(self.kv_tile - 1):
                    prefix_state.s_tile_end_log[tile] = logp
                    prefix_state.s_tile_end_phase[tile, 0] = phase_re
                    prefix_state.s_tile_end_phase[tile, 1] = phase_im
        pipeline.sync()

        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            offset_log = cutlass.Float32(0.0)
            offset_phase_re = cutlass.Float32(1.0)
            offset_phase_im = cutlass.Float32(0.0)
            for tile in cutlass.range_constexpr(n_tiles):
                prefix_state.s_tile_off_log[tile] = offset_log
                prefix_state.s_tile_off_phase[tile, 0] = offset_phase_re
                prefix_state.s_tile_off_phase[tile, 1] = offset_phase_im
                tile_end_log = cutlass.Float32(prefix_state.s_tile_end_log[tile])
                tile_end_phase_re = cutlass.Float32(
                    prefix_state.s_tile_end_phase[tile, 0]
                )
                tile_end_phase_im = cutlass.Float32(
                    prefix_state.s_tile_end_phase[tile, 1]
                )
                offset_log = offset_log + tile_end_log
                offset_phase_re, offset_phase_im = complex_mul(
                    tile_end_phase_re,
                    tile_end_phase_im,
                    offset_phase_re,
                    offset_phase_im,
                )
        pipeline.sync()

        for tile in cutlass.range_constexpr(n_tiles):
            if warp == cutlass.Int32(tile % num_warps) and lane < cutlass.Int32(
                self.kv_tile
            ):
                seq_idx = cutlass.Int32(tile * self.kv_tile) + lane
                logp = cutlass.Float32(
                    prefix_state.s_scale_full[seq_idx]
                ) + cutlass.Float32(prefix_state.s_tile_off_log[tile])
                phase_re = cutlass.Float32(prefix_state.s_phase_full[seq_idx, 0])
                phase_im = cutlass.Float32(prefix_state.s_phase_full[seq_idx, 1])
                offset_phase_re = cutlass.Float32(
                    prefix_state.s_tile_off_phase[tile, 0]
                )
                offset_phase_im = cutlass.Float32(
                    prefix_state.s_tile_off_phase[tile, 1]
                )
                phase_re, phase_im = complex_mul(
                    phase_re, phase_im, offset_phase_re, offset_phase_im
                )
                mag2 = cutlass.Float32(phase_re * phase_re + phase_im * phase_im + eps)
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
                phase_re = phase_re * inv_mag
                phase_im = phase_im * inv_mag
                stable_logp = clamp_nonpositive_prefix_log(logp)
                scale = cute.math.exp2(
                    stable_logp * cutlass.Float32(TWO_LOG2_E), fastmath=True
                )
                prefix_state.s_scale_full[seq_idx] = scale
                prefix_state.s_inv_scale_full[seq_idx] = one / scale
                prefix_state.s_phase_full[seq_idx, 0] = phase_re
                prefix_state.s_phase_full[seq_idx, 1] = phase_im
        pipeline.sync()

    @cute.jit
    def _load_full_tap_metadata(self, tap_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        num_warps = self.num_warps
        n_tiles = self.num_kv_tiles

        for tile in cutlass.range_constexpr(n_tiles):
            if warp == cutlass.Int32(tile % num_warps) and lane < cutlass.Int32(
                self.kv_tile
            ):
                seq_idx = cutlass.Int32(tile * self.kv_tile) + lane
                tap_state.s_tap_full[seq_idx, 0] = cutlass.Float32(
                    tap_state.m_tap[tap_state.batch_head_chunk, seq_idx, 0, 0]
                )
                tap_state.s_tap_full[seq_idx, 1] = cutlass.Float32(
                    tap_state.m_tap[tap_state.batch_head_chunk, seq_idx, 0, 1]
                )
                tap_state.s_tap_full[seq_idx, 2] = cutlass.Float32(
                    tap_state.m_tap[tap_state.batch_head_chunk, seq_idx, 1, 0]
                )
                tap_state.s_tap_full[seq_idx, 3] = cutlass.Float32(
                    tap_state.m_tap[tap_state.batch_head_chunk, seq_idx, 1, 1]
                )
        pipeline.sync()

    @cute.jit
    def _load_previous_boundary_rows(self, boundary_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        prev_batch_group_chunk = cutlass.select_(
            boundary_state.chunk_index > cutlass.Int32(0),
            boundary_state.batch_group_chunk - cutlass.Int32(1),
            boundary_state.batch_group_chunk,
        )

        iters_p = (self.P + self.num_threads - 1) // self.num_threads
        for it in range(iters_p):
            p = tidx + cutlass.Int32(it * self.num_threads)
            if p < cutlass.Int32(self.P):
                u_prev0 = boundary_state.m_u_prev0[boundary_state.batch_head, p]
                u_prev_chunk = boundary_state.m_u[
                    boundary_state.batch_head_chunk
                    - cutlass.select_(
                        boundary_state.chunk_index > cutlass.Int32(0),
                        cutlass.Int32(1),
                        cutlass.Int32(0),
                    ),
                    self.L - 1,
                    0,
                    p,
                ]
                boundary_state.s_u_prev[p] = cutlass.select_(
                    boundary_state.chunk_index == cutlass.Int32(0),
                    u_prev0,
                    u_prev_chunk,
                )

        iters_d = (self.D + self.num_threads - 1) // self.num_threads
        for it in range(iters_d):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D):
                b_prev0 = boundary_state.m_b_prev0[boundary_state.batch_group, d]
                b_prev_chunk = boundary_state.m_b[
                    prev_batch_group_chunk, self.L - 1, 0, d
                ]
                boundary_state.s_b_prev[d] = cutlass.select_(
                    boundary_state.chunk_index == cutlass.Int32(0),
                    b_prev0,
                    b_prev_chunk,
                )
        pipeline.sync()

    @cute.jit
    def _initialize_row_outputs(self, row_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        row_tile_start = cutlass.Int32(row_state.m_tile * self.kv_tile)

        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.kv_tile):
            seq_idx = row_tile_start + lane
            row_state.s_row_scale[lane] = cutlass.Float32(
                row_state.s_scale_full[seq_idx]
            )
        if tidx < cutlass.Int32(self.kv_tile):
            row_state.s_row_dlogprefix[tidx] = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.kv_tile * 4):
            row = tidx // cutlass.Int32(4)
            col = tidx - row * cutlass.Int32(4)
            row_state.s_row_dr[row, col] = cutlass.Float32(0.0)
        pipeline.sync()

    @cute.jit
    def _initialize_dlogprefix_accumulator(self, logprefix_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        iters = (self.L + self.num_threads - 1) // self.num_threads
        for it in range(iters):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < cutlass.Int32(self.L):
                logprefix_state.s_dlogprefix_accum[idx] = cutlass.Float32(0.0)
        pipeline.sync()

    @cute.jit
    def _store_dlogprefix_accumulator(self, logprefix_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        iters = (self.L + self.num_threads - 1) // self.num_threads
        for it in range(iters):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < cutlass.Int32(self.L):
                logprefix_state.m_dlogprefix[logprefix_state.batch_head_chunk, idx] = (
                    logprefix_state.m_dlogprefix[logprefix_state.batch_head_chunk, idx]
                    + logprefix_state.s_dlogprefix_accum[idx]
                )
        pipeline.sync()

    @cute.jit
    def _load_dout_tiles(self, dout_state: SimpleNamespace):
        for p_tile_idx in cutlass.range_constexpr(dout_state.n_p_tiles):
            s_dout_tile = cute.local_tile(
                dout_state.s_dout,
                (self.kv_tile, self.p_tile),
                (0, p_tile_idx),
            )
            t_dout_smem = dout_state.gmem_thr_copy_p.partition_D(s_dout_tile)
            g_dout = cute.local_tile(
                dout_state.m_dout[dout_state.batch_head_chunk, None, 0, None],
                (self.kv_tile, self.p_tile),
                (dout_state.m_tile, p_tile_idx),
            )
            t_dout_gmem = dout_state.gmem_thr_copy_p.partition_S(g_dout)
            if cutlass.const_expr(self.P == dout_state.p_padded):
                cute.copy(dout_state.gmem_tiled_copy_p, t_dout_gmem, t_dout_smem)
            else:
                c_dout = cute.local_tile(
                    dout_state.coord_dout,
                    (self.kv_tile, self.p_tile),
                    (dout_state.m_tile, p_tile_idx),
                )
                t_dout_coord = dout_state.gmem_thr_copy_p.partition_S(c_dout)
                t_dout_pred = self._make_copy_column_predicate(
                    t_dout_smem, t_dout_coord, dout_state.m_dout.layout.shape[3]
                )
                self._copy_rows_with_zero_fill(
                    dout_state.gmem_tiled_copy_p,
                    t_dout_gmem,
                    t_dout_smem,
                    t_dout_coord,
                    t_dout_pred,
                    dout_state.m_dout.layout.shape[1],
                )
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        pipeline.sync()

    @cute.jit
    def _initialize_column_metadata(self, column_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.kv_tile):
            seq_idx = cutlass.Int32(column_state.n_tile * self.kv_tile) + lane
            column_state.s_inv_row_scale[lane] = cutlass.Float32(
                column_state.s_inv_scale_full[seq_idx]
            )
            column_state.s_phase_col[lane, 0] = cutlass.Float32(
                column_state.s_phase_full[seq_idx, 0]
            )
            column_state.s_phase_col[lane, 1] = cutlass.Float32(
                column_state.s_phase_full[seq_idx, 1]
            )
            column_state.s_tap_prev[lane, 0] = cutlass.Float32(
                column_state.s_tap_full[seq_idx, 0]
            )
            column_state.s_tap_prev[lane, 1] = cutlass.Float32(
                column_state.s_tap_full[seq_idx, 1]
            )
            column_state.s_tap_curr[lane, 0] = cutlass.Float32(
                column_state.s_tap_full[seq_idx, 2]
            )
            column_state.s_tap_curr[lane, 1] = cutlass.Float32(
                column_state.s_tap_full[seq_idx, 3]
            )
        pipeline.sync()

    @cute.jit
    def _stage_shifted_value_tile(self, value_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        total_value_tile = cutlass.Int32(self.kv_tile * self.p_tile)
        iters_value_tile = (total_value_tile + self.num_threads - 1) // self.num_threads
        out_dtype = value_state.m_u.element_type
        for it in range(iters_value_tile):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_value_tile:
                row_local = idx // cutlass.Int32(self.p_tile)
                p_local = idx - row_local * cutlass.Int32(self.p_tile)
                p = value_state.p_tile_start + p_local
                value = cutlass.Float32(0.0).to(out_dtype)
                if p < cutlass.Int32(self.P):
                    if value_state.pass_id == 1:
                        src_row = value_state.row_tile_start + row_local
                        if src_row < cutlass.Int32(self.L):
                            value = value_state.m_u[
                                value_state.batch_head_chunk, src_row, 0, p
                            ]
                    else:
                        src_row = (
                            value_state.row_tile_start + row_local - cutlass.Int32(1)
                        )
                        if src_row >= cutlass.Int32(0):
                            value = value_state.m_u[
                                value_state.batch_head_chunk, src_row, 0, p
                            ]
                        else:
                            value = value_state.s_u_prev[p]
                value_state.s_value_tile[row_local, p_local] = value

    @cute.jit
    def _apply_score_scales_and_mask(self, score_state: SimpleNamespace):
        out_dtype = score_state.output_scores_mn.element_type
        for r in cutlass.range_constexpr(cute.size(score_state.acc_scores_mn.shape[0])):
            row_idx = cutlass.Int32(score_state.coord_scores_mn[r, 0][1])
            row_local = row_idx - score_state.row_tile_start
            row_scale = cutlass.Float32(0.0)
            if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                row_scale = cutlass.Float32(score_state.s_row_scale[row_local])
            for c in cutlass.range_constexpr(
                cute.size(score_state.acc_scores_mn.shape[1])
            ):
                col_idx = cutlass.Int32(score_state.coord_scores_mn[0, c][3])
                col_local = col_idx - score_state.col_tile_start
                scaled_score = cutlass.Float32(0.0).to(out_dtype)
                if cute.elem_less(row_idx, cutlass.Int32(self.L)) and cute.elem_less(
                    col_idx, cutlass.Int32(self.L)
                ):
                    score_is_causal = (
                        not score_state.causal_diagonal_only
                    ) or cute.elem_less(col_idx, row_idx + cutlass.Int32(1))
                    if score_is_causal:
                        inv_scale = cutlass.Float32(
                            score_state.s_inv_row_scale[col_local]
                        )
                        scaled_score = safe_cast_to_dtype(
                            score_state.acc_scores_mn[r, c] * row_scale * inv_scale,
                            out_dtype,
                        )
                score_state.output_scores_mn[r, c] = scaled_score

    @cute.jit
    def _accumulate_scores_from_shifted_values(self, score_state: SimpleNamespace):
        for p_tile_idx in cutlass.range_constexpr(score_state.n_p_tiles):
            p_tile_start = cutlass.Int32(p_tile_idx * self.p_tile)
            s_dout_tile = cute.local_tile(
                score_state.s_dout,
                (self.kv_tile, self.p_tile),
                (0, p_tile_idx),
            )
            tSr_dout_tile = score_state.thr_mma.make_fragment_A(
                score_state.thr_mma.partition_A(s_dout_tile)
            )
            tSs_dout_tile = score_state.thr_copy_a.partition_S(s_dout_tile)
            tSr_dout_tile_view = score_state.thr_copy_a.retile(tSr_dout_tile)
            s_value_pass_tile = cute.local_tile(
                score_state.s_value_tile,
                (self.kv_tile, self.p_tile),
                (0, score_state.pass_id * score_state.n_p_tiles + p_tile_idx),
            )
            value_state = SimpleNamespace(
                pass_id=score_state.pass_id,
                batch_head_chunk=score_state.batch_head_chunk,
                row_tile_start=score_state.col_tile_start,
                p_tile_start=p_tile_start,
                m_u=score_state.m_u,
                s_u_prev=score_state.s_u_prev,
                s_value_tile=s_value_pass_tile,
            )
            self._stage_shifted_value_tile(value_state)
            pipeline.sync()
            tSr_value_tile = score_state.thr_mma.make_fragment_B(
                score_state.thr_mma.partition_B(s_value_pass_tile)
            )
            tSs_value_tile = score_state.thr_copy_b.partition_S(s_value_pass_tile)
            tSr_value_tile_view = score_state.thr_copy_b.retile(tSr_value_tile)
            self._accumulate_from_staged_tiles(
                score_state.tiled_mma,
                score_state.acc_scores,
                score_state.smem_tiled_copy_a,
                score_state.smem_tiled_copy_b,
                tSs_dout_tile,
                tSr_dout_tile_view,
                tSs_value_tile,
                tSr_value_tile_view,
                tSr_dout_tile,
                tSr_value_tile,
            )

    @cute.jit
    def _materialize_score_tile(self, score_state: SimpleNamespace):
        acc_scores_mn = self._make_accumulator_mn_view(score_state.acc_scores)
        tCs_score_tile = score_state.thr_mma.partition_C(score_state.s_score_tile)
        tCr_score_tile = cute.make_rmem_tensor_like(
            tCs_score_tile, score_state.s_score_tile.element_type
        )
        t_coord_scores_mn = self._make_accumulator_mn_view(score_state.t_coord_scores)
        tCr_score_tile_mn = self._make_accumulator_mn_view(tCr_score_tile)
        scaled_score_state = SimpleNamespace(
            acc_scores_mn=acc_scores_mn,
            coord_scores_mn=t_coord_scores_mn,
            row_tile_start=score_state.row_tile_start,
            col_tile_start=score_state.col_tile_start,
            s_row_scale=score_state.s_row_scale,
            s_inv_row_scale=score_state.s_inv_row_scale,
            output_scores_mn=tCr_score_tile_mn,
            causal_diagonal_only=score_state.causal_diagonal_only,
        )
        self._apply_score_scales_and_mask(scaled_score_state)
        cute.autovec_copy(tCr_score_tile, tCs_score_tile)
        pipeline.sync()

    @cute.jit
    def _accumulate_offterm_from_conjugated_z0(self, offterm_state: SimpleNamespace):
        for p_tile_idx in cutlass.range_constexpr(offterm_state.n_p_tiles):
            p_tile_start = cutlass.Int32(p_tile_idx * self.p_tile)
            s_dout_tile = cute.local_tile(
                offterm_state.s_dout,
                (self.kv_tile, self.p_tile),
                (0, p_tile_idx),
            )
            tSr_dout_tile = offterm_state.thr_mma.make_fragment_A(
                offterm_state.thr_mma.partition_A(s_dout_tile)
            )
            tSs_dout_tile = offterm_state.thr_copy_a.partition_S(s_dout_tile)
            tSr_dout_tile_view = offterm_state.thr_copy_a.retile(tSr_dout_tile)
            z0_state = SimpleNamespace(
                batch_head_chunk=offterm_state.batch_head_chunk,
                p_tile_start=p_tile_start,
                p_tile_idx=p_tile_idx,
                d_tile_idx=offterm_state.d_tile_idx,
                d_tile_start=offterm_state.d_tile_start,
                m_z0=offterm_state.m_z0,
                coord_z0=offterm_state.coord_z0,
                s_z0_tile=offterm_state.s_z0_tile,
                gmem_tiled_copy_d_async=offterm_state.gmem_tiled_copy_d_async,
                gmem_thr_copy_d_async=offterm_state.gmem_thr_copy_d_async,
            )
            self._load_conjugated_z0_stage(z0_state)
            self._accumulate_from_staged_tiles(
                offterm_state.tiled_mma,
                offterm_state.acc_offterm,
                offterm_state.smem_tiled_copy_a,
                offterm_state.smem_tiled_copy_bt,
                tSs_dout_tile,
                tSr_dout_tile_view,
                offterm_state.tSs_z0_tile,
                offterm_state.tSr_z0_tile_view,
                tSr_dout_tile,
                offterm_state.tSr_z0_tile,
            )

    @cute.jit
    def _materialize_dq_tile(self, dq_state: SimpleNamespace):
        acc_dq_total_mn = self._make_accumulator_mn_view(dq_state.acc_dq_total)
        acc_dq_offterm_mn = self._make_accumulator_mn_view(dq_state.acc_dq_offterm)
        t_coord_dq_tile_mn = self._make_accumulator_mn_view(dq_state.t_coord_dq_tile)
        for r in cutlass.range_constexpr(cute.size(acc_dq_total_mn.shape[0])):
            row_idx = cutlass.Int32(t_coord_dq_tile_mn[r, 0][1])
            row_local = row_idx - dq_state.row_tile_start
            row_scale = cutlass.Float32(0.0)
            if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                row_scale = cutlass.Float32(dq_state.s_row_scale[row_local])
            for c in cutlass.range(
                cute.size(acc_dq_total_mn.shape[1]), unroll_full=True
            ):
                offterm_scaled = acc_dq_offterm_mn[r, c] * row_scale
                acc_dq_total_mn[r, c] = acc_dq_total_mn[r, c] + offterm_scaled
                acc_dq_offterm_mn[r, c] = offterm_scaled

        tCs_dq_tile = dq_state.thr_mma.partition_C(dq_state.s_dq_tile)
        tCr_dq_tile = cute.make_rmem_tensor_like(
            tCs_dq_tile, dq_state.s_dq_tile.element_type
        )
        acc_dq_values = dq_state.acc_dq_total.load()
        for i in cutlass.range_constexpr(cute.size(tCr_dq_tile)):
            tCr_dq_tile[i] = safe_cast_to_dtype(
                acc_dq_values[i], dq_state.s_dq_tile.element_type
            )
        cute.autovec_copy(tCr_dq_tile, tCs_dq_tile)
        pipeline.sync()

    @cute.jit
    def _load_boundary_aware_key_tile(self, key_load_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        d_block = self._smem_block_size_d()
        out_dtype = key_load_state.s_key_tile.element_type
        if key_load_state.pass_id == 1:
            g_key = cute.local_tile(
                key_load_state.m_b[key_load_state.batch_group_chunk, None, 0, None],
                (self.kv_tile, d_block),
                (key_load_state.n_tile, key_load_state.d_tile),
            )
            t_key_gmem = key_load_state.gmem_thr_copy_d_async.partition_S(g_key)
            if cutlass.const_expr(self.D == self.D_padded):
                cute.copy(
                    key_load_state.gmem_tiled_copy_d_async,
                    t_key_gmem,
                    key_load_state.t_key_smem,
                )
            else:
                c_key = cute.local_tile(
                    key_load_state.coord_b,
                    (self.kv_tile, d_block),
                    (key_load_state.n_tile, key_load_state.d_tile),
                )
                t_key_coord = key_load_state.gmem_thr_copy_d_async.partition_S(c_key)
                t_key_pred = self._make_copy_column_predicate(
                    key_load_state.t_key_smem,
                    t_key_coord,
                    key_load_state.m_b.layout.shape[3],
                )
                self._copy_rows_with_zero_fill(
                    key_load_state.gmem_tiled_copy_d_async,
                    t_key_gmem,
                    key_load_state.t_key_smem,
                    t_key_coord,
                    t_key_pred,
                    key_load_state.m_b.layout.shape[1],
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        else:
            if key_load_state.use_shifted_global:
                g_key = cute.local_tile(
                    key_load_state.m_b[key_load_state.batch_group_chunk, None, 0, None],
                    (self.kv_tile, d_block),
                    (key_load_state.n_tile, key_load_state.d_tile),
                )
                g_key = cute.domain_offset((-1, 0), g_key)
                t_key_gmem = key_load_state.gmem_thr_copy_d_async.partition_S(g_key)
                c_key = cute.local_tile(
                    key_load_state.coord_b,
                    (self.kv_tile, d_block),
                    (key_load_state.n_tile, key_load_state.d_tile),
                )
                c_key = cute.domain_offset((-1, 0), c_key)
                t_key_coord = key_load_state.gmem_thr_copy_d_async.partition_S(c_key)
                t_key_pred = self._make_copy_column_predicate(
                    key_load_state.t_key_smem,
                    t_key_coord,
                    key_load_state.m_b.layout.shape[3],
                )
                self._copy_rows_with_zero_fill(
                    key_load_state.gmem_tiled_copy_d_async,
                    t_key_gmem,
                    key_load_state.t_key_smem,
                    t_key_coord,
                    t_key_pred,
                    key_load_state.m_b.layout.shape[1],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            else:
                g_key = cute.local_tile(
                    key_load_state.m_b[key_load_state.batch_group_chunk, None, 0, None],
                    (self.kv_tile, d_block),
                    (key_load_state.n_tile, key_load_state.d_tile),
                )
                g_key = cute.domain_offset((-1, 0), g_key)
                t_key_gmem = key_load_state.gmem_thr_copy_d_async.partition_S(g_key)
                c_key = cute.local_tile(
                    key_load_state.coord_b,
                    (self.kv_tile, d_block),
                    (key_load_state.n_tile, key_load_state.d_tile),
                )
                c_key = cute.domain_offset((-1, 0), c_key)
                t_key_coord = key_load_state.gmem_thr_copy_d_async.partition_S(c_key)
                t_key_pred = self._make_copy_column_predicate(
                    key_load_state.t_key_smem,
                    t_key_coord,
                    key_load_state.m_b.layout.shape[3],
                )
                self._copy_shifted_rows_with_zero_fill(
                    key_load_state.gmem_tiled_copy_d_async,
                    t_key_gmem,
                    key_load_state.t_key_smem,
                    t_key_coord,
                    t_key_pred,
                    key_load_state.m_b.layout.shape[1],
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

                iters_boundary_row = (
                    d_block + self.num_threads - 1
                ) // self.num_threads
                for it in range(iters_boundary_row):
                    idx = tidx + cutlass.Int32(it * self.num_threads)
                    if idx < cutlass.Int32(d_block):
                        d = key_load_state.d_tile_start + idx
                        value = cutlass.Float32(0.0).to(out_dtype)
                        if d < cutlass.Int32(self.D):
                            value = key_load_state.s_b_prev[d]
                        key_load_state.s_key_tile[0, idx] = value
        pipeline.sync()

    @cute.jit
    def _apply_tap_phase_to_staged_keys(self, key_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        d_block = self._smem_block_size_d()
        out_dtype = key_state.s_key_tile.element_type
        total_pairs_tile = cutlass.Int32(d_block * self.kv_tile // 2)
        iters_pairs_tile = (total_pairs_tile + self.num_threads - 1) // self.num_threads
        pairs_per_row = cutlass.Int32(d_block // 2)
        for it in range(iters_pairs_tile):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_pairs_tile:
                row_local = idx // pairs_per_row
                pair_idx = idx - row_local * pairs_per_row
                seq_idx = key_state.row_tile_start + row_local
                d0_local = pair_idx * cutlass.Int32(2)
                d0 = key_state.d_tile_start + d0_local
                out0 = cutlass.Float32(0.0).to(out_dtype)
                out1 = cutlass.Float32(0.0).to(out_dtype)
                if seq_idx < cutlass.Int32(self.L) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    bx = cutlass.Float32(
                        key_state.s_key_tile[row_local, d0_local + 0].to(
                            cutlass.Float32
                        )
                    )
                    by = cutlass.Float32(
                        key_state.s_key_tile[row_local, d0_local + 1].to(
                            cutlass.Float32
                        )
                    )
                    tap_re = cutlass.Float32(0.0)
                    tap_im = cutlass.Float32(0.0)
                    if key_state.pass_id == 1:
                        tap_re = cutlass.Float32(key_state.s_tap_curr[row_local, 0])
                        tap_im = cutlass.Float32(key_state.s_tap_curr[row_local, 1])
                    else:
                        tap_re = cutlass.Float32(key_state.s_tap_prev[row_local, 0])
                        tap_im = cutlass.Float32(key_state.s_tap_prev[row_local, 1])
                    tap_out_re, tap_out_im = apply_complex_tap(bx, by, tap_re, tap_im)
                    phase_re = cutlass.Float32(key_state.s_phase_col[row_local, 0])
                    phase_im = cutlass.Float32(key_state.s_phase_col[row_local, 1])
                    out_re, out_im = conj_mul_phase(
                        tap_out_re, tap_out_im, phase_re, phase_im
                    )
                    out0 = safe_cast_to_dtype(out_re, out_dtype)
                    out1 = safe_cast_to_dtype(out_im, out_dtype)
                key_state.s_key_tile[row_local, d0_local + 0] = out0
                key_state.s_key_tile[row_local, d0_local + 1] = out1
        pipeline.sync()

    @cute.jit
    def _load_conjugated_z0_stage(self, z0_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        d_block = self._smem_block_size_d()
        out_dtype = z0_state.s_z0_tile.element_type
        if cutlass.const_expr(z0_state.m_z0.element_type != cutlass.Float32):
            g_z0 = cute.local_tile(
                z0_state.m_z0[z0_state.batch_head_chunk, None, None],
                (self.p_tile, d_block),
                (z0_state.p_tile_idx, z0_state.d_tile_idx),
            )
            t_z0_smem = z0_state.gmem_thr_copy_d_async.partition_D(z0_state.s_z0_tile)
            t_z0_gmem = z0_state.gmem_thr_copy_d_async.partition_S(g_z0)
            if cutlass.const_expr(self.D == self.D_padded):
                cute.copy(z0_state.gmem_tiled_copy_d_async, t_z0_gmem, t_z0_smem)
            else:
                coord_z0_tile = cute.local_tile(
                    z0_state.coord_z0,
                    (self.p_tile, d_block),
                    (z0_state.p_tile_idx, z0_state.d_tile_idx),
                )
                t_z0_coord = z0_state.gmem_thr_copy_d_async.partition_S(coord_z0_tile)
                t_z0_pred = self._make_copy_column_predicate(
                    t_z0_smem, t_z0_coord, z0_state.m_z0.layout.shape[2]
                )
                self._copy_rows_with_zero_fill(
                    z0_state.gmem_tiled_copy_d_async,
                    t_z0_gmem,
                    t_z0_smem,
                    t_z0_coord,
                    t_z0_pred,
                    z0_state.m_z0.layout.shape[1],
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            pipeline.sync()

            total_z0_pairs_fast = cutlass.Int32(self.p_tile * d_block // 2)
            iters_z0_pairs_fast = (
                total_z0_pairs_fast + self.num_threads - 1
            ) // self.num_threads
            pairs_per_row_fast = cutlass.Int32(d_block // 2)
            for it in range(iters_z0_pairs_fast):
                idx = tidx + cutlass.Int32(it * self.num_threads)
                if idx < total_z0_pairs_fast:
                    p_local = idx // pairs_per_row_fast
                    pair_idx = idx - p_local * pairs_per_row_fast
                    d0_local = pair_idx * cutlass.Int32(2)
                    out1 = z0_state.s_z0_tile[p_local, d0_local + 1]
                    z0_state.s_z0_tile[p_local, d0_local + 1] = -out1
            pipeline.sync()
            return

        total_z0_pairs = cutlass.Int32(self.p_tile * d_block // 2)
        iters_z0_pairs = (total_z0_pairs + self.num_threads - 1) // self.num_threads
        pairs_per_row = cutlass.Int32(d_block // 2)
        for it in range(iters_z0_pairs):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_z0_pairs:
                p_local = idx // pairs_per_row
                pair_idx = idx - p_local * pairs_per_row
                d0_local = pair_idx * cutlass.Int32(2)
                d0 = z0_state.d_tile_start + d0_local
                p = z0_state.p_tile_start + p_local
                out0 = cutlass.Float32(0.0).to(out_dtype)
                out1 = cutlass.Float32(0.0).to(out_dtype)
                if p < cutlass.Int32(self.P) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    if cutlass.const_expr(
                        z0_state.m_z0.element_type == cutlass.Float32
                    ):
                        out0 = safe_cast_to_dtype(
                            z0_state.m_z0[z0_state.batch_head_chunk, p, d0 + 0],
                            out_dtype,
                        )
                        out1 = safe_cast_to_dtype(
                            -z0_state.m_z0[z0_state.batch_head_chunk, p, d0 + 1],
                            out_dtype,
                        )
                    else:
                        out0 = z0_state.m_z0[z0_state.batch_head_chunk, p, d0 + 0]
                        out1 = -z0_state.m_z0[z0_state.batch_head_chunk, p, d0 + 1]
                z0_state.s_z0_tile[p_local, d0_local + 0] = out0
                z0_state.s_z0_tile[p_local, d0_local + 1] = out1
        pipeline.sync()

    @cute.jit
    def _load_c_tile(self, c_state: SimpleNamespace):
        d_block = self._smem_block_size_d()
        g_c = cute.local_tile(
            c_state.m_c[c_state.batch_group_chunk, None, 0, None],
            (self.kv_tile, d_block),
            (c_state.m_tile, c_state.d_tile),
        )
        t_c_smem = c_state.gmem_thr_copy_d_async.partition_D(c_state.s_c_tile)
        t_c_gmem = c_state.gmem_thr_copy_d_async.partition_S(g_c)
        if cutlass.const_expr(self.D == self.D_padded):
            cute.copy(c_state.gmem_tiled_copy_d_async, t_c_gmem, t_c_smem)
        else:
            coord_c_tile = cute.local_tile(
                c_state.coord_c,
                (self.kv_tile, d_block),
                (c_state.m_tile, c_state.d_tile),
            )
            t_c_coord = c_state.gmem_thr_copy_d_async.partition_S(coord_c_tile)
            t_c_pred = self._make_copy_column_predicate(
                t_c_smem, t_c_coord, c_state.m_c.layout.shape[3]
            )
            self._copy_rows_with_zero_fill(
                c_state.gmem_tiled_copy_d_async,
                t_c_gmem,
                t_c_smem,
                t_c_coord,
                t_c_pred,
                c_state.m_c.layout.shape[1],
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        pipeline.sync()

    @cute.jit
    def _rotate_c_tile_in_place(self, c_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        d_block = self._smem_block_size_d()
        out_dtype = c_state.s_c_tile.element_type
        total_pairs = cutlass.Int32(self.kv_tile * d_block // 2)
        iters_pairs = (total_pairs + self.num_threads - 1) // self.num_threads
        pairs_per_row = cutlass.Int32(d_block // 2)
        for it in range(iters_pairs):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_pairs:
                row_local = idx // pairs_per_row
                pair_idx = idx - row_local * pairs_per_row
                seq_idx = cutlass.Int32(c_state.m_tile * self.kv_tile) + row_local
                d0_local = pair_idx * cutlass.Int32(2)
                d0 = c_state.d_tile_start + d0_local
                out0 = cutlass.Float32(0.0).to(out_dtype)
                out1 = cutlass.Float32(0.0).to(out_dtype)
                if cute.elem_less(seq_idx, cutlass.Int32(self.L)) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    c0 = cutlass.Float32(
                        c_state.s_c_tile[row_local, d0_local + 0].to(cutlass.Float32)
                    )
                    c1 = cutlass.Float32(
                        c_state.s_c_tile[row_local, d0_local + 1].to(cutlass.Float32)
                    )
                    phase_re = cutlass.Float32(c_state.s_phase_full[seq_idx, 0])
                    phase_im = cutlass.Float32(c_state.s_phase_full[seq_idx, 1])
                    c0, c1 = conj_mul_phase(c0, c1, phase_re, phase_im)
                    out0 = safe_cast_to_dtype(c0, out_dtype)
                    out1 = safe_cast_to_dtype(c1, out_dtype)
                c_state.s_c_tile[row_local, d0_local + 0] = out0
                c_state.s_c_tile[row_local, d0_local + 1] = out1
        pipeline.sync()

    @cute.jit
    def _accumulate_dlogprefix_from_score_partial(
        self,
        prefix_state: SimpleNamespace,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        n0 = cutlass.Int32(prefix_state.n_tile * self.kv_tile)
        c_score = cute.local_tile(
            prefix_state.coord_scores,
            (self.kv_tile, self.kv_tile),
            (prefix_state.m_tile, prefix_state.n_tile),
        )
        t_score_coord = prefix_state.thr_mma.partition_C(c_score)
        t_score_coord_mn = self._make_accumulator_mn_view(t_score_coord)
        t_scaled_dscore_smem = prefix_state.thr_mma.partition_C(
            prefix_state.s_score_tile
        )
        scaled_dscore = cute.make_rmem_tensor_like(
            t_scaled_dscore_smem, prefix_state.s_score_tile.element_type
        )
        cute.autovec_copy(t_scaled_dscore_smem, scaled_dscore)
        scaled_dscore_mn = self._make_accumulator_mn_view(scaled_dscore)
        acc_score_mn = self._make_accumulator_mn_view(prefix_state.acc_score)

        col_acc0 = cutlass.Float32(0.0)
        col_acc1 = cutlass.Float32(0.0)
        col_acc2 = cutlass.Float32(0.0)
        col_acc3 = cutlass.Float32(0.0)
        diag_tile = prefix_state.m_tile == prefix_state.n_tile
        for r in cutlass.range_constexpr(cute.size(acc_score_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            for c in cutlass.range_constexpr(cute.size(acc_score_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                if cute.elem_less(row_idx, cutlass.Int32(self.L)) and cute.elem_less(
                    col_idx, cutlass.Int32(self.L)
                ):
                    score_is_causal = (not diag_tile) or cute.elem_less(
                        col_idx, row_idx + cutlass.Int32(1)
                    )
                    if score_is_causal:
                        prod = (
                            cutlass.Float32(scaled_dscore_mn[r, c]) * acc_score_mn[r, c]
                        )
                        if c == 0:
                            col_acc0 = col_acc0 + prod
                        elif c == 1:
                            col_acc1 = col_acc1 + prod
                        elif c == 2:
                            col_acc2 = col_acc2 + prod
                        else:
                            col_acc3 = col_acc3 + prod

        for offset in (4, 8, 16):
            col_acc0 = col_acc0 + cute.arch.shuffle_sync_bfly(
                col_acc0, offset=offset, mask=-1, mask_and_clamp=31
            )
            col_acc1 = col_acc1 + cute.arch.shuffle_sync_bfly(
                col_acc1, offset=offset, mask=-1, mask_and_clamp=31
            )
            col_acc2 = col_acc2 + cute.arch.shuffle_sync_bfly(
                col_acc2, offset=offset, mask=-1, mask_and_clamp=31
            )
            col_acc3 = col_acc3 + cute.arch.shuffle_sync_bfly(
                col_acc3, offset=offset, mask=-1, mask_and_clamp=31
            )

        if lane < cutlass.Int32(4):
            prefix_state.s_col_accum[warp, lane, 0] = col_acc0
            prefix_state.s_col_accum[warp, lane, 1] = col_acc1
            prefix_state.s_col_accum[warp, lane, 2] = col_acc2
            prefix_state.s_col_accum[warp, lane, 3] = col_acc3
        pipeline.sync()

        if warp == cutlass.Int32(0) and lane < cutlass.Int32(8):
            src_lane = cutlass.select_(
                lane < cutlass.Int32(4), lane, lane - cutlass.Int32(4)
            )
            src_warp0 = cutlass.select_(
                lane < cutlass.Int32(4),
                cutlass.Int32(0),
                cutlass.Int32(2),
            )
            src_warp1 = src_warp0 + cutlass.Int32(1)
            base = cutlass.select_(
                lane < cutlass.Int32(4),
                cutlass.Int32(0),
                cutlass.Int32(8),
            ) + src_lane * cutlass.Int32(2)

            col_sum0 = cutlass.Float32(prefix_state.s_col_accum[src_warp0, src_lane, 0])
            col_sum0 = col_sum0 + cutlass.Float32(
                prefix_state.s_col_accum[src_warp1, src_lane, 0]
            )
            col_sum1 = cutlass.Float32(prefix_state.s_col_accum[src_warp0, src_lane, 1])
            col_sum1 = col_sum1 + cutlass.Float32(
                prefix_state.s_col_accum[src_warp1, src_lane, 1]
            )
            col_sum2 = cutlass.Float32(prefix_state.s_col_accum[src_warp0, src_lane, 2])
            col_sum2 = col_sum2 + cutlass.Float32(
                prefix_state.s_col_accum[src_warp1, src_lane, 2]
            )
            col_sum3 = cutlass.Float32(prefix_state.s_col_accum[src_warp0, src_lane, 3])
            col_sum3 = col_sum3 + cutlass.Float32(
                prefix_state.s_col_accum[src_warp1, src_lane, 3]
            )

            col0 = n0 + base
            col1 = col0 + cutlass.Int32(1)
            col2 = col0 + cutlass.Int32(16)
            col3 = col2 + cutlass.Int32(1)
            if col0 < cutlass.Int32(self.L):
                prefix_state.s_dlogprefix_accum[col0] = (
                    prefix_state.s_dlogprefix_accum[col0]
                    - cutlass.Float32(2.0) * col_sum0
                )
            if col1 < cutlass.Int32(self.L):
                prefix_state.s_dlogprefix_accum[col1] = (
                    prefix_state.s_dlogprefix_accum[col1]
                    - cutlass.Float32(2.0) * col_sum1
                )
            if col2 < cutlass.Int32(self.L):
                prefix_state.s_dlogprefix_accum[col2] = (
                    prefix_state.s_dlogprefix_accum[col2]
                    - cutlass.Float32(2.0) * col_sum2
                )
            if col3 < cutlass.Int32(self.L):
                prefix_state.s_dlogprefix_accum[col3] = (
                    prefix_state.s_dlogprefix_accum[col3]
                    - cutlass.Float32(2.0) * col_sum3
                )
        pipeline.sync()

    @cute.jit
    def _accumulate_row_outputs_from_dc(self, dc_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        d_block = self._smem_block_size_d()
        out_dtype = dc_state.s_dc_tile.element_type
        nvec = cutlass.Int32(d_block // 2)
        row_local = warp
        while cute.elem_less(row_local, cutlass.Int32(self.kv_tile)):
            seq_idx = dc_state.row_tile_start + row_local
            row_dlogprefix_sum = cutlass.Float32(0.0)
            dR00_sum = cutlass.Float32(0.0)
            dR01_sum = cutlass.Float32(0.0)
            dR10_sum = cutlass.Float32(0.0)
            dR11_sum = cutlass.Float32(0.0)
            pair_idx = lane
            while cute.elem_less(pair_idx, nvec):
                d0_local = pair_idx * cutlass.Int32(2)
                d0 = dc_state.d_tile_start + d0_local
                out0 = cutlass.Float32(0.0).to(out_dtype)
                out1 = cutlass.Float32(0.0).to(out_dtype)
                if cute.elem_less(seq_idx, cutlass.Int32(self.L)) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    dq0 = cutlass.Float32(
                        dc_state.s_dc_tile[row_local, d0_local + 0].to(cutlass.Float32)
                    )
                    dq1 = cutlass.Float32(
                        dc_state.s_dc_tile[row_local, d0_local + 1].to(cutlass.Float32)
                    )
                    phase_re = cutlass.Float32(dc_state.s_phase_full[seq_idx, 0])
                    phase_im = cutlass.Float32(dc_state.s_phase_full[seq_idx, 1])
                    dc0, dc1 = conj_mul_phase(dq0, dq1, phase_re, phase_im)
                    c_rot0 = cutlass.Float32(dc_state.s_c_tile[row_local, d0_local + 0])
                    c_rot1 = cutlass.Float32(dc_state.s_c_tile[row_local, d0_local + 1])
                    # s_c_tile holds conj(C) * phase for row-output reuse.
                    c0, c1 = conj_mul_phase(c_rot0, c_rot1, phase_re, phase_im)
                    row_dlogprefix_sum = (
                        row_dlogprefix_sum + dq0 * c_rot0 + dq1 * c_rot1
                    )
                    dR00_sum = dR00_sum + dq0 * c0
                    dR01_sum = dR01_sum + dq0 * c1
                    dR10_sum = dR10_sum + dq1 * c0
                    dR11_sum = dR11_sum + dq1 * c1
                    out0 = safe_cast_to_dtype(dc0, out_dtype)
                    out1 = safe_cast_to_dtype(dc1, out_dtype)
                dc_state.s_dc_tile[row_local, d0_local + 0] = out0
                dc_state.s_dc_tile[row_local, d0_local + 1] = out1
                pair_idx = pair_idx + cutlass.Int32(32)
            for offset in (16, 8, 4, 2, 1):
                row_dlogprefix_sum = row_dlogprefix_sum + cute.arch.shuffle_sync_bfly(
                    row_dlogprefix_sum, offset=offset, mask=-1, mask_and_clamp=31
                )
                dR00_sum = dR00_sum + cute.arch.shuffle_sync_bfly(
                    dR00_sum, offset=offset, mask=-1, mask_and_clamp=31
                )
                dR01_sum = dR01_sum + cute.arch.shuffle_sync_bfly(
                    dR01_sum, offset=offset, mask=-1, mask_and_clamp=31
                )
                dR10_sum = dR10_sum + cute.arch.shuffle_sync_bfly(
                    dR10_sum, offset=offset, mask=-1, mask_and_clamp=31
                )
                dR11_sum = dR11_sum + cute.arch.shuffle_sync_bfly(
                    dR11_sum, offset=offset, mask=-1, mask_and_clamp=31
                )
            if lane == cutlass.Int32(0):
                dc_state.s_row_dlogprefix[row_local] = (
                    dc_state.s_row_dlogprefix[row_local] + row_dlogprefix_sum
                )
                dc_state.s_row_dr[row_local, 0] = (
                    dc_state.s_row_dr[row_local, 0] + dR00_sum
                )
                dc_state.s_row_dr[row_local, 1] = (
                    dc_state.s_row_dr[row_local, 1] + dR01_sum
                )
                dc_state.s_row_dr[row_local, 2] = (
                    dc_state.s_row_dr[row_local, 2] + dR10_sum
                )
                dc_state.s_row_dr[row_local, 3] = (
                    dc_state.s_row_dr[row_local, 3] + dR11_sum
                )
            row_local = row_local + cutlass.Int32(self.num_warps)
        pipeline.sync()

    @cute.jit
    def _store_dc_tile(self, dc_store_state: SimpleNamespace):
        d_block = self._smem_block_size_d()
        out_dtype = dc_store_state.s_dc_tile.element_type
        g_dc = cute.local_tile(
            dc_store_state.m_dc[dc_store_state.batch_head_chunk, None, 0, None],
            (self.kv_tile, d_block),
            (dc_store_state.m_tile, dc_store_state.d_tile),
        )
        t_dc_smem = dc_store_state.gmem_thr_store_d.partition_S(
            dc_store_state.s_dc_tile
        )
        t_dc_gmem = dc_store_state.gmem_thr_store_d.partition_D(g_dc)
        if cutlass.const_expr(self.D == self.D_padded):
            cute.copy(dc_store_state.gmem_tiled_store_d, t_dc_smem, t_dc_gmem)
        else:
            t_dc_rmem = cute.make_rmem_tensor_like(t_dc_gmem, out_dtype)
            cute.copy(dc_store_state.gmem_tiled_store_d, t_dc_smem, t_dc_rmem)
            coord_dc_tile = cute.local_tile(
                dc_store_state.coord_dc,
                (self.kv_tile, d_block),
                (dc_store_state.m_tile, dc_store_state.d_tile),
            )
            t_dc_coord = dc_store_state.gmem_thr_store_d.partition_D(coord_dc_tile)
            t_dc_pred = self._make_copy_column_predicate(
                t_dc_gmem,
                t_dc_coord,
                dc_store_state.m_dc.layout.shape[3],
            )
            self._copy_rows_if_valid(
                dc_store_state.gmem_tiled_store_d,
                t_dc_rmem,
                t_dc_gmem,
                t_dc_coord,
                t_dc_pred,
                dc_store_state.m_dc.layout.shape[1],
            )

    @cute.jit
    def _store_row_outputs(self, row_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        row_tile_start = cutlass.Int32(row_state.m_tile * self.kv_tile)
        if tidx < cutlass.Int32(self.kv_tile):
            seq_idx = row_tile_start + tidx
            if cute.elem_less(seq_idx, cutlass.Int32(self.L)):
                row_state.s_dlogprefix_accum[seq_idx] = (
                    row_state.s_dlogprefix_accum[seq_idx]
                    + cutlass.Float32(2.0) * row_state.s_row_dlogprefix[tidx]
                )
                row_state.m_dr[row_state.batch_head_chunk, seq_idx, 0] = (
                    row_state.s_row_dr[tidx, 0]
                )
                row_state.m_dr[row_state.batch_head_chunk, seq_idx, 1] = (
                    row_state.s_row_dr[tidx, 1]
                )
                row_state.m_dr[row_state.batch_head_chunk, seq_idx, 2] = (
                    row_state.s_row_dr[tidx, 2]
                )
                row_state.m_dr[row_state.batch_head_chunk, seq_idx, 3] = (
                    row_state.s_row_dr[tidx, 3]
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
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDR: cute.Tensor,
    ):
        if cutlass.const_expr(
            mU.element_type != mB.element_type
            or mU.element_type != mC.element_type
            or mU.element_type != mDOut.element_type
        ):
            raise TypeError("U/B/C/dOut must share dtype.")
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/C/dOut must be Float16/BFloat16.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(
            mZ0.element_type != cutlass.Float32 and mZ0.element_type != mU.element_type
        ):
            raise TypeError("Z0 must be Float32 or match U/B dtype.")
        if cutlass.const_expr(mDC.element_type != mU.element_type):
            raise TypeError("dC must share dtype with U/B.")
        if cutlass.const_expr(mDLogPrefix.element_type != cutlass.Float32):
            raise TypeError("dlogprefix must be Float32.")
        if cutlass.const_expr(mDR.element_type != cutlass.Float32):
            raise TypeError("dR must be Float32.")

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
        if cutlass.const_expr(mZ0.shape[1] != self.P or mZ0.shape[2] != self.D):
            raise ValueError("Z0 must be (BHC, P, D).")
        if cutlass.const_expr(mDC.shape[1] != self.L or mDC.shape[2] != 1):
            raise ValueError("dC must be (BHC, L, 1, D).")
        if cutlass.const_expr(mDLogPrefix.shape[1] != self.L):
            raise ValueError("dlogprefix must be (BHC, L).")
        if cutlass.const_expr(mDR.shape[1] != self.L or mDR.shape[2] != 4):
            raise ValueError("dR must be (BHC, L, 4).")

        if cutlass.const_expr(self.P_padded % self.p_tile != 0):
            raise ValueError("P_padded must be a multiple of 32.")

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
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDR: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        kernel_bundle = self._make_kernel_bundle(mU.element_type)
        launch_kwargs = {
            "grid": (1, 1, cute.size(mU.shape[0])),
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
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
            kernel_bundle.layouts.u_prev_layout,
            kernel_bundle.layouts.b_prev_layout,
            kernel_bundle.layouts.grad_output_layout,
            kernel_bundle.layouts.value_layout,
            kernel_bundle.layouts.key_layout,
            kernel_bundle.layouts.query_layout,
            kernel_bundle.layouts.score_layout,
            kernel_bundle.layouts.z0_layout,
            kernel_bundle.layouts.full_scale_layout,
            kernel_bundle.layouts.full_inv_scale_layout,
            kernel_bundle.layouts.full_phase_layout,
            kernel_bundle.layouts.full_tap_layout,
            kernel_bundle.layouts.tile_prefix_log_layout,
            kernel_bundle.layouts.tile_prefix_phase_layout,
            kernel_bundle.layouts.row_scale_layout,
            kernel_bundle.layouts.row_inv_scale_layout,
            kernel_bundle.layouts.column_phase_layout,
            kernel_bundle.layouts.tap_layout,
            kernel_bundle.layouts.row_dlogprefix_layout,
            kernel_bundle.layouts.row_dr_layout,
            kernel_bundle.layouts.dlogprefix_accum_layout,
            kernel_bundle.layouts.column_accumulator_layout,
            kernel_bundle.copies.gmem_tiled_copy_d_async,
            kernel_bundle.copies.gmem_tiled_copy_p,
            kernel_bundle.copies.gmem_tiled_store_d,
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
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDR: cute.Tensor,
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
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
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
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
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
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDR: cute.Tensor,
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
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
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
            mZ0,
            mDC,
            mDLogPrefix,
            mDR,
            stream=stream,
        )

    @cute.kernel
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
        mZ0: cute.Tensor,
        mDC: cute.Tensor,
        mDLogPrefix: cute.Tensor,
        mDR: cute.Tensor,
        u_prev_layout: cute.Layout | cute.ComposedLayout,
        b_prev_layout: cute.Layout | cute.ComposedLayout,
        grad_output_layout: cute.Layout | cute.ComposedLayout,
        value_layout: cute.Layout | cute.ComposedLayout,
        key_layout: cute.Layout | cute.ComposedLayout,
        query_layout: cute.Layout | cute.ComposedLayout,
        score_layout: cute.Layout | cute.ComposedLayout,
        z0_layout: cute.Layout | cute.ComposedLayout,
        full_scale_layout: cute.Layout | cute.ComposedLayout,
        full_inv_scale_layout: cute.Layout | cute.ComposedLayout,
        full_phase_layout: cute.Layout | cute.ComposedLayout,
        full_tap_layout: cute.Layout | cute.ComposedLayout,
        tile_prefix_log_layout: cute.Layout | cute.ComposedLayout,
        tile_prefix_phase_layout: cute.Layout | cute.ComposedLayout,
        row_scale_layout: cute.Layout | cute.ComposedLayout,
        row_inv_scale_layout: cute.Layout | cute.ComposedLayout,
        column_phase_layout: cute.Layout | cute.ComposedLayout,
        tap_layout: cute.Layout | cute.ComposedLayout,
        row_dlogprefix_layout: cute.Layout | cute.ComposedLayout,
        row_dr_layout: cute.Layout | cute.ComposedLayout,
        dlogprefix_accum_layout: cute.Layout | cute.ComposedLayout,
        column_accumulator_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_d_async: cute.TiledCopy,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_tiled_store_d: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()

        batch_heads = mU_prev0.shape[0]
        batch_head_chunks = mU.shape[0]
        n_chunks = batch_head_chunks // batch_heads
        batch_head = bidz // n_chunks
        batch_group = self._batch_group(batch_head)
        batch_group_chunk = self._batch_group_chunk(bidz, n_chunks)
        chunk_index = bidz - batch_head * n_chunks

        d_padded = self.D_padded
        p_padded = self.P_padded
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        n_p_tiles = self.num_p_tiles
        n_tiles = self.num_kv_tiles

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_dout = storage.dout_tile.get_tensor(grad_output_layout)
        s_value_tile = storage.value_tile.get_tensor(value_layout)
        s_key_tile = storage.key_tile.get_tensor(key_layout)
        s_score_tile = cute.local_tile(s_value_tile, (kv_tile, kv_tile), (0, 0))
        s_c_stage = storage.query_tile.get_tensor(query_layout)
        # `Z0` shares the 32xd_padded key slab after the causal `n_tile` sweep.
        s_z0_tile = cute.make_tensor(s_key_tile.iterator, z0_layout)
        s_u_prev = storage.u_prev_row.get_tensor(u_prev_layout)
        s_b_prev = storage.b_prev_row.get_tensor(b_prev_layout)

        s_scale_full = storage.full_scale.get_tensor(full_scale_layout)
        s_inv_scale_full = storage.full_inv_scale.get_tensor(full_inv_scale_layout)
        s_phase_full = storage.full_phase.get_tensor(full_phase_layout)
        s_tap_full = storage.full_tap.get_tensor(full_tap_layout)
        s_tile_end_log = storage.tile_end_log.get_tensor(tile_prefix_log_layout)
        s_tile_end_phase = storage.tile_end_phase.get_tensor(tile_prefix_phase_layout)
        s_tile_off_log = storage.tile_offset_log.get_tensor(tile_prefix_log_layout)
        s_tile_off_phase = storage.tile_offset_phase.get_tensor(
            tile_prefix_phase_layout
        )

        s_row_scale = storage.row_scale.get_tensor(row_scale_layout)
        s_inv_row_scale = storage.row_inv_scale.get_tensor(row_inv_scale_layout)
        s_phase_col = storage.phase_col.get_tensor(column_phase_layout)
        s_tap_prev = storage.tap_prev.get_tensor(tap_layout)
        s_tap_curr = storage.tap_curr.get_tensor(tap_layout)
        s_row_dlogprefix = storage.row_dlogprefix.get_tensor(row_dlogprefix_layout)
        s_row_dr = storage.row_dr.get_tensor(row_dr_layout)
        s_dlogprefix_accum = storage.dlogprefix_accum.get_tensor(
            dlogprefix_accum_layout
        )
        s_col_accum = storage.column_accum.get_tensor(column_accumulator_layout)

        logprefix_init_state = SimpleNamespace(s_dlogprefix_accum=s_dlogprefix_accum)
        self._initialize_dlogprefix_accumulator(logprefix_init_state)

        prefix_state = SimpleNamespace(
            batch_head_chunk=bidz,
            m_transition=mM,
            s_scale_full=s_scale_full,
            s_inv_scale_full=s_inv_scale_full,
            s_phase_full=s_phase_full,
            s_tile_end_log=s_tile_end_log,
            s_tile_end_phase=s_tile_end_phase,
            s_tile_off_log=s_tile_off_log,
            s_tile_off_phase=s_tile_off_phase,
        )
        self._compute_phase_prefix_metadata(prefix_state)
        tap_state = SimpleNamespace(
            batch_head_chunk=bidz,
            m_tap=mK,
            s_tap_full=s_tap_full,
        )
        self._load_full_tap_metadata(tap_state)

        boundary_state = SimpleNamespace(
            batch_head=batch_head,
            batch_group=batch_group,
            batch_head_chunk=bidz,
            batch_group_chunk=batch_group_chunk,
            chunk_index=chunk_index,
            m_u=mU,
            m_b=mB,
            m_u_prev0=mU_prev0,
            m_b_prev0=mB_prev0,
            s_u_prev=s_u_prev,
            s_b_prev=s_b_prev,
        )
        self._load_previous_boundary_rows(boundary_state)

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_BT = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_tiled_copy_BT = cute.make_tiled_copy_B(smem_copy_atom_BT, tiled_mma)
        thr_mma = tiled_mma.get_slice(tidx)

        coord_scores = cute.make_identity_tensor(
            (mU.shape[0], self.L, mU.shape[2], self.L)
        )
        coord_b = cute.make_identity_tensor(mB.layout.shape)
        coord_c = cute.make_identity_tensor(mC.layout.shape)
        coord_z0 = cute.make_identity_tensor(mZ0.layout.shape)
        coord_dc = cute.make_identity_tensor(
            (mU.shape[0], self.L, mU.shape[2], d_padded)
        )
        coord_scores_tile = coord_scores[bidz, None, 0, None]
        coord_dc_tile = coord_dc[bidz, None, 0, None]
        coord_b_tile = coord_b[batch_group_chunk, None, 0, None]
        coord_c_tile = coord_c[batch_group_chunk, None, 0, None]
        coord_z0_tile = coord_z0[bidz, None, None]

        d_block = self._smem_block_size_d()
        n_d_tiles = d_padded // d_block

        acc_shape_blk = thr_mma.partition_shape_C((kv_tile, kv_tile))
        acc_shape_tileD_blk = thr_mma.partition_shape_C((kv_tile, d_block))

        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        gmem_thr_copy_D_async = gmem_tiled_copy_d_async.get_slice(tidx)
        gmem_thr_copy_P = gmem_tiled_copy_p.get_slice(tidx)

        coord_dout = cute.make_identity_tensor(mDOut.layout.shape)
        coord_dout_tile = coord_dout[bidz, None, 0, None]

        for m_tile in cutlass.range_constexpr(n_tiles):
            m0 = cutlass.Int32(m_tile * kv_tile)
            row_state = SimpleNamespace(
                m_tile=m_tile,
                s_scale_full=s_scale_full,
                s_row_scale=s_row_scale,
                s_row_dlogprefix=s_row_dlogprefix,
                s_row_dr=s_row_dr,
            )
            self._initialize_row_outputs(row_state)

            dout_state = SimpleNamespace(
                batch_head_chunk=bidz,
                m_tile=m_tile,
                n_p_tiles=n_p_tiles,
                p_padded=p_padded,
                gmem_tiled_copy_p=gmem_tiled_copy_p,
                gmem_thr_copy_p=gmem_thr_copy_P,
                s_dout=s_dout,
                m_dout=mDOut,
                coord_dout=coord_dout_tile,
            )
            self._load_dout_tiles(dout_state)

            for d_tile in range(n_d_tiles):
                d_base = cutlass.Int32(d_tile * d_block)

                s_key_block = s_key_tile
                sKt_blk_layout = cute.make_layout(
                    (d_block, kv_tile), stride=(kv_tile, 1)
                )
                sKt_blk = cute.composition(s_key_block, sKt_blk_layout)
                tKs_key_block = gmem_thr_copy_D_async.partition_D(s_key_block)
                s_z0_block = s_z0_tile
                s_z0t_blk_layout = cute.make_layout(
                    (d_block, p_tile), stride=(p_tile, 1)
                )
                s_z0t_blk = cute.composition(s_z0_block, s_z0t_blk_layout)

                tSrK_blk = thr_mma.make_fragment_B(thr_mma.partition_B(sKt_blk))
                tSsK_blk = thr_copy_BT.partition_S(sKt_blk)
                tSrK_blk_view = thr_copy_BT.retile(tSrK_blk)
                tSrK_score_blk = thr_mma.make_fragment_B(
                    thr_mma.partition_B(s_key_block)
                )
                tSsK_score_blk = thr_copy_B.partition_S(s_key_block)
                tSrK_score_blk_view = thr_copy_B.retile(tSrK_score_blk)
                tSrZ0_blk = thr_mma.make_fragment_B(thr_mma.partition_B(s_z0t_blk))
                tSsZ0_blk = thr_copy_BT.partition_S(s_z0t_blk)
                tSrZ0_blk_view = thr_copy_BT.retile(tSrZ0_blk)

                acc_dq_total = cute.make_rmem_tensor(
                    acc_shape_tileD_blk, cutlass.Float32
                )
                acc_dq_total.fill(0.0)
                s_c_block = s_c_stage
                tSrC_blk = thr_mma.make_fragment_A(thr_mma.partition_A(s_c_block))
                tSsC_blk = thr_copy_A.partition_S(s_c_block)
                tSrC_blk_view = thr_copy_A.retile(tSrC_blk)
                c_state = SimpleNamespace(
                    batch_group_chunk=batch_group_chunk,
                    m_tile=m_tile,
                    d_tile=d_tile,
                    d_tile_start=d_base,
                    d_block=d_block,
                    m_c=mC,
                    coord_c=coord_c_tile,
                    s_c_tile=s_c_block,
                    s_phase_full=s_phase_full,
                    gmem_tiled_copy_d_async=gmem_tiled_copy_d_async,
                    gmem_thr_copy_d_async=gmem_thr_copy_D_async,
                )
                self._load_c_tile(c_state)
                self._rotate_c_tile_in_place(c_state)

                for n_tile in cutlass.range_constexpr(m_tile + 1):
                    n0 = cutlass.Int32(n_tile * kv_tile)

                    column_state = SimpleNamespace(
                        n_tile=n_tile,
                        s_inv_scale_full=s_inv_scale_full,
                        s_phase_full=s_phase_full,
                        s_tap_full=s_tap_full,
                        s_inv_row_scale=s_inv_row_scale,
                        s_phase_col=s_phase_col,
                        s_tap_prev=s_tap_prev,
                        s_tap_curr=s_tap_curr,
                    )
                    self._initialize_column_metadata(column_state)

                    for pass_id in range(2):
                        key_load_state = SimpleNamespace(
                            pass_id=pass_id,
                            use_shifted_global=(pass_id == 0) and (n_tile > 0),
                            batch_group_chunk=batch_group_chunk,
                            n_tile=n_tile,
                            d_tile=d_tile,
                            row_tile_start=n0,
                            d_tile_start=d_base,
                            d_block=d_block,
                            m_b=mB,
                            coord_b=coord_b_tile,
                            gmem_tiled_copy_d_async=gmem_tiled_copy_d_async,
                            gmem_thr_copy_d_async=gmem_thr_copy_D_async,
                            t_key_smem=tKs_key_block,
                            s_key_tile=s_key_block,
                            s_b_prev=s_b_prev,
                        )
                        self._load_boundary_aware_key_tile(key_load_state)

                        key_state = SimpleNamespace(
                            pass_id=pass_id,
                            row_tile_start=n0,
                            d_tile_start=d_base,
                            d_block=d_block,
                            s_key_tile=s_key_block,
                            s_phase_col=s_phase_col,
                            s_tap_prev=s_tap_prev,
                            s_tap_curr=s_tap_curr,
                        )
                        self._apply_tap_phase_to_staged_keys(key_state)

                        coord_scores_block = cute.local_tile(
                            coord_scores_tile, (kv_tile, kv_tile), (m_tile, n_tile)
                        )
                        t_coord_scores_block = thr_mma.partition_C(coord_scores_block)
                        s_score_current = s_score_tile
                        acc_scores_block = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_scores_block.fill(0.0)
                        score_acc_state = SimpleNamespace(
                            pass_id=pass_id,
                            batch_head_chunk=bidz,
                            col_tile_start=n0,
                            n_p_tiles=n_p_tiles,
                            m_u=mU,
                            s_u_prev=s_u_prev,
                            s_dout=s_dout,
                            s_value_tile=s_value_tile,
                            acc_scores=acc_scores_block,
                            tiled_mma=tiled_mma,
                            thr_mma=thr_mma,
                            thr_copy_a=thr_copy_A,
                            thr_copy_b=thr_copy_B,
                            smem_tiled_copy_a=smem_tiled_copy_A,
                            smem_tiled_copy_b=smem_tiled_copy_B,
                        )
                        self._accumulate_scores_from_shifted_values(score_acc_state)
                        score_tile_state = SimpleNamespace(
                            acc_scores=acc_scores_block,
                            thr_mma=thr_mma,
                            s_score_tile=s_score_current,
                            t_coord_scores=t_coord_scores_block,
                            row_tile_start=m0,
                            col_tile_start=n0,
                            s_row_scale=s_row_scale,
                            s_inv_row_scale=s_inv_row_scale,
                            causal_diagonal_only=m_tile == n_tile,
                        )
                        self._materialize_score_tile(score_tile_state)

                        tSr_score_current = thr_mma.make_fragment_A(
                            thr_mma.partition_A(s_score_current)
                        )
                        tSs_score_current = thr_copy_A.partition_S(s_score_current)
                        tSr_score_current_view = thr_copy_A.retile(tSr_score_current)

                        self._accumulate_from_staged_tiles(
                            tiled_mma,
                            acc_dq_total,
                            smem_tiled_copy_A,
                            smem_tiled_copy_BT,
                            tSs_score_current,
                            tSr_score_current_view,
                            tSsK_blk,
                            tSrK_blk_view,
                            tSr_score_current,
                            tSrK_blk,
                        )
                        acc_score_partial = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_score_partial.fill(0.0)
                        self._accumulate_from_staged_tiles(
                            tiled_mma,
                            acc_score_partial,
                            smem_tiled_copy_A,
                            smem_tiled_copy_B,
                            tSsC_blk,
                            tSrC_blk_view,
                            tSsK_score_blk,
                            tSrK_score_blk_view,
                            tSrC_blk,
                            tSrK_score_blk,
                        )
                        prefix_update_state = SimpleNamespace(
                            batch_head_chunk=bidz,
                            m_tile=m_tile,
                            n_tile=n_tile,
                            coord_scores=coord_scores_tile,
                            s_score_tile=s_score_current,
                            s_dlogprefix_accum=s_dlogprefix_accum,
                            s_col_accum=s_col_accum,
                            acc_score=acc_score_partial,
                            thr_mma=thr_mma,
                        )
                        self._accumulate_dlogprefix_from_score_partial(
                            prefix_update_state
                        )

                acc_dq_offterm = cute.make_rmem_tensor(
                    acc_shape_tileD_blk, cutlass.Float32
                )
                acc_dq_offterm.fill(0.0)
                offterm_state = SimpleNamespace(
                    batch_head_chunk=bidz,
                    n_p_tiles=n_p_tiles,
                    d_tile_idx=d_tile,
                    d_tile_start=d_base,
                    m_z0=mZ0,
                    coord_z0=coord_z0_tile,
                    s_z0_tile=s_z0_block,
                    s_dout=s_dout,
                    acc_offterm=acc_dq_offterm,
                    tiled_mma=tiled_mma,
                    thr_mma=thr_mma,
                    thr_copy_a=thr_copy_A,
                    smem_tiled_copy_a=smem_tiled_copy_A,
                    smem_tiled_copy_bt=smem_tiled_copy_BT,
                    tSs_z0_tile=tSsZ0_blk,
                    tSr_z0_tile_view=tSrZ0_blk_view,
                    tSr_z0_tile=tSrZ0_blk,
                    gmem_tiled_copy_d_async=gmem_tiled_copy_d_async,
                    gmem_thr_copy_d_async=gmem_thr_copy_D_async,
                )
                self._accumulate_offterm_from_conjugated_z0(offterm_state)

                coord_dc_block = cute.local_tile(
                    coord_dc_tile, (kv_tile, d_block), (m_tile, d_tile)
                )
                t_coord_dc_block = thr_mma.partition_C(coord_dc_block)
                dq_tile_state = SimpleNamespace(
                    acc_dq_total=acc_dq_total,
                    acc_dq_offterm=acc_dq_offterm,
                    t_coord_dq_tile=t_coord_dc_block,
                    row_tile_start=m0,
                    s_row_scale=s_row_scale,
                    thr_mma=thr_mma,
                    s_dq_tile=s_key_block,
                )
                self._materialize_dq_tile(dq_tile_state)

                dc_state = SimpleNamespace(
                    row_tile_start=m0,
                    d_tile_start=d_base,
                    d_block=d_block,
                    s_dc_tile=s_key_block,
                    s_c_tile=s_c_block,
                    s_phase_full=s_phase_full,
                    s_row_dlogprefix=s_row_dlogprefix,
                    s_row_dr=s_row_dr,
                )
                self._accumulate_row_outputs_from_dc(dc_state)

                gmem_thr_store_d = gmem_tiled_store_d.get_slice(tidx)
                dc_store_state = SimpleNamespace(
                    batch_head_chunk=bidz,
                    m_tile=m_tile,
                    d_tile=d_tile,
                    d_block=d_block,
                    s_dc_tile=s_key_block,
                    coord_dc=coord_dc_tile,
                    m_dc=mDC,
                    gmem_tiled_store_d=gmem_tiled_store_d,
                    gmem_thr_store_d=gmem_thr_store_d,
                )
                self._store_dc_tile(dc_store_state)

            pipeline.sync()
            row_store_state = SimpleNamespace(
                batch_head_chunk=bidz,
                m_tile=m_tile,
                m_dr=mDR,
                s_dlogprefix_accum=s_dlogprefix_accum,
                s_row_dlogprefix=s_row_dlogprefix,
                s_row_dr=s_row_dr,
            )
            self._store_row_outputs(row_store_state)

        logprefix_store_state = SimpleNamespace(
            batch_head_chunk=bidz,
            m_dlogprefix=mDLogPrefix,
            s_dlogprefix_accum=s_dlogprefix_accum,
        )
        self._store_dlogprefix_accumulator(logprefix_store_state)


__all__ = ["BwdDCDRAmpere"]
