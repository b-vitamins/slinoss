"""CuTe backward kernel for the ``v2x2ssd`` chunk-scan ``dlogprefix`` slice.

``ChunkScanBwdDLPAmpere`` is the live Ampere tensor-core implementation used by
the backward path. It reconstructs prefix magnitude/phase metadata from ``M``,
stages causal ``dOut`` / ``U`` score tiles and rotated ``C`` / ``B`` tiles,
contracts them into score blocks, and accumulates the resulting column
reductions directly into ``dlogprefix``.

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
- ``dlogprefix``: ``(BHC, L)`` fp32 scalar metadata partials

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
    clamp_nonpositive_prefix_log,
    complex_mul,
    conj_mul_phase,
    safe_cast_to_dtype,
)


@dataclass(frozen=True)
class ChunkScanBwdDLPLayoutBundle:
    value_key_alias_layout: object
    u_prev_layout: object
    b_prev_layout: object
    dlogprefix_layout: object
    grad_output_layout: object
    value_layout: object
    query_layout: object
    key_layout: object
    column_accumulator_layout: object
    full_scale_layout: object
    full_inv_scale_layout: object
    full_phase_layout: object
    tile_prefix_log_layout: object
    tile_prefix_phase_layout: object
    row_scale_layout: object
    row_inv_scale_layout: object
    row_phase_layout: object
    column_phase_layout: object
    tap_layout: object


@dataclass(frozen=True)
class ChunkScanBwdDLPCopyBundle:
    gmem_tiled_copy_d: object
    gmem_tiled_copy_p: object


@dataclass(frozen=True)
class ChunkScanBwdDLPKernelBundle:
    layouts: ChunkScanBwdDLPLayoutBundle
    copies: ChunkScanBwdDLPCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    smem_bytes: int

    @property
    def smem_size(self) -> int:
        return self.smem_bytes


@dataclass(frozen=True)
class ChunkScanBwdDLPSupportInfo:
    smem_capacity_bytes: int
    required_smem_bytes: int

    @property
    def supported(self) -> bool:
        return self.required_smem_bytes <= self.smem_capacity_bytes


class ChunkScanBwdDLPAmpere:
    """Ampere tensor-core backward kernel for the ``v2x2ssd`` ``dlp`` slice.

    This kernel owns the scalar ``dlogprefix`` path. It rebuilds the prefix
    magnitude and phase metadata from raw packed ``M``, stages the causal
    ``dOut``/``U`` score tiles together with the rotated ``C``/``B`` tiles,
    contracts those tiles into score blocks, and reduces the resulting columns
    directly into ``dlogprefix``.
    """

    _SUPPORT_INFO_CACHE: ClassVar[
        dict[tuple[object, ...], ChunkScanBwdDLPSupportInfo]
    ] = {}

    def __init__(self, dtype, *, chunk_size, D, P, num_threads=128):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.num_threads = int(num_threads)
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

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

    @property
    def num_warps(self) -> int:
        """Number of resident warps in the fixed 128-thread CTA."""
        return self.num_threads // 32

    @property
    def num_kv_tiles(self) -> int:
        """Number of 32-row key/value tiles that cover the chunk."""
        return self.L // self.kv_tile

    def _metadata_length(self) -> int:
        """Length of the prefix scratch arrays after single-tile folding."""
        if self.L == self.kv_tile:
            return 1
        return self.L

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

    def _tail_pad_layout(self):
        return cute.make_layout((128,), stride=(1,))

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

    # Host-side support/build helpers
    def _make_layout_bundle(self) -> ChunkScanBwdDLPLayoutBundle:
        d_padded = self.D_padded
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        d_block = self._smem_block_size_d()
        n_tiles = self.num_kv_tiles
        meta_l = self._metadata_length()

        swizzle_bits_d = self._swizzle_bits(d_block)
        s_d_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_d, 3, 3),
            0,
            cute.make_layout((8, d_block), stride=(d_block, 1)),
        )
        query_layout = cute.tile_to_shape(s_d_layout_atom, (kv_tile, d_padded), (0, 1))
        key_layout = cute.tile_to_shape(s_d_layout_atom, (kv_tile, d_block), (0, 1))

        smem_k_block_size_p = p_tile
        swizzle_bits_p = self._swizzle_bits(smem_k_block_size_p)
        s_p_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_p, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_p), stride=(smem_k_block_size_p, 1)),
        )
        value_key_alias_layout = cute.tile_to_shape(
            s_p_layout_atom, (kv_tile, 2 * p_tile), (0, 1)
        )
        grad_output_layout = cute.tile_to_shape(
            s_p_layout_atom, (kv_tile, p_tile), (0, 1)
        )
        value_layout = cute.tile_to_shape(s_p_layout_atom, (kv_tile, p_tile), (0, 1))

        return ChunkScanBwdDLPLayoutBundle(
            value_key_alias_layout=value_key_alias_layout,
            u_prev_layout=cute.make_layout((self.P,), stride=(1,)),
            b_prev_layout=cute.make_layout((self.D,), stride=(1,)),
            dlogprefix_layout=cute.make_layout((self.L,), stride=(1,)),
            grad_output_layout=grad_output_layout,
            value_layout=value_layout,
            query_layout=query_layout,
            key_layout=key_layout,
            column_accumulator_layout=cute.make_layout(
                (self.num_warps, 4, 4), stride=(16, 4, 1)
            ),
            full_scale_layout=cute.make_layout((meta_l,), stride=(1,)),
            full_inv_scale_layout=cute.make_layout((meta_l,), stride=(1,)),
            full_phase_layout=cute.make_layout((meta_l, 2), stride=(2, 1)),
            tile_prefix_log_layout=cute.make_layout((n_tiles,), stride=(1,)),
            tile_prefix_phase_layout=cute.make_layout((n_tiles, 2), stride=(2, 1)),
            row_scale_layout=cute.make_layout((kv_tile,), stride=(1,)),
            row_inv_scale_layout=cute.make_layout((kv_tile,), stride=(1,)),
            row_phase_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            column_phase_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
            tap_layout=cute.make_layout((kv_tile, 2), stride=(2, 1)),
        )

    def _compute_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        """Estimate dynamic SMEM for support checks without constructing layouts."""
        in_bytes = in_dtype.width // 8
        kv_tile = self.kv_tile
        p_tile = self.p_tile
        d_block = self._smem_block_size_d()
        d_padded = self.D_padded
        n_tiles = self.num_kv_tiles
        meta_l = self._metadata_length()
        pvk_alias_bytes = max(
            2 * kv_tile * p_tile * in_bytes,
            kv_tile * d_block * in_bytes,
        )
        # Keep this accounting aligned with `_make_shared_storage`. It must stay
        # host-only because the CuTe layout builders need an MLIR context.
        smem_fields = [
            ("pvk_alias", pvk_alias_bytes, 16),
            ("query_tile", kv_tile * d_padded * in_bytes, 16),
            ("u_prev", self.P * in_bytes, 16),
            ("b_prev", self.D * in_bytes, 16),
            ("dlogprefix", self.L * 4, 4),
            ("full_scale", meta_l * 4, 4),
            ("full_inv_scale", meta_l * 4, 4),
            ("full_phase", meta_l * 2 * 4, 16),
            ("tile_end_log", n_tiles * 4, 4),
            ("tile_end_phase", n_tiles * 2 * 4, 16),
            ("tile_off_log", n_tiles * 4, 4),
            ("tile_off_phase", n_tiles * 2 * 4, 16),
            ("row_scale", kv_tile * 4, 4),
            ("row_inv_scale", kv_tile * 4, 4),
            ("row_phase", kv_tile * 2 * 4, 16),
            ("col_phase", kv_tile * 2 * 4, 16),
            ("tap_prev", kv_tile * 2 * 4, 16),
            ("tap_curr", kv_tile * 2 * 4, 16),
            ("col_accum", self.num_warps * 4 * 4 * 4, 16),
            ("tail_pad", 128 * 4, 16),
        ]
        return self._struct_size_bytes(
            [(field_bytes, align) for _, field_bytes, align in smem_fields]
        )

    def support_info(
        self,
        in_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkScanBwdDLPSupportInfo:
        if in_dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkScanBwdDLPSupportInfo(0, 1)

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

        info = ChunkScanBwdDLPSupportInfo(
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

    def _make_copy_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDLPCopyBundle:
        universal_copy_bits = 128
        elems_per_copy = universal_copy_bits // in_dtype.width
        smem_k_block_size_d = self._smem_block_size_d()
        td_shape_dim_1 = smem_k_block_size_d // elems_per_copy
        td_layout = cute.make_layout(
            (self.num_threads // td_shape_dim_1, td_shape_dim_1),
            stride=(td_shape_dim_1, 1),
        )
        tp_shape_dim_1 = self.p_tile // elems_per_copy
        tp_layout = cute.make_layout(
            (self.num_threads // tp_shape_dim_1, tp_shape_dim_1),
            stride=(tp_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, elems_per_copy))

        atom_universal_copy_in = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        return ChunkScanBwdDLPCopyBundle(
            gmem_tiled_copy_d=cute.make_tiled_copy_tv(
                atom_universal_copy_in, td_layout, v_in_layout
            ),
            gmem_tiled_copy_p=cute.make_tiled_copy_tv(
                atom_universal_copy_in, tp_layout, v_in_layout
            ),
        )

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            in_dtype,
            self.acc_dtype,
            (16, 8, 16),
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * 16,
            self.atom_layout_mnk[1] * 8 * 2,
            self.atom_layout_mnk[2] * 16,
        )
        tc_layout = cute.make_layout(self.atom_layout_mnk)
        return cute.make_tiled_mma(mma_op, tc_layout, permutation_mnk=permutation_mnk)

    def _make_kernel_bundle(
        self, in_dtype: type[cutlass.Numeric]
    ) -> ChunkScanBwdDLPKernelBundle:
        layouts = self._make_layout_bundle()
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        return ChunkScanBwdDLPKernelBundle(
            layouts=layouts,
            copies=self._make_copy_bundle(in_dtype),
            tiled_mma=self._make_tiled_mma(in_dtype),
            shared_storage_cls=shared_storage_cls,
            smem_bytes=int(shared_storage_cls.size_in_bytes()),
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanBwdDLPLayoutBundle,
    ):
        tail_pad_layout = self._tail_pad_layout()
        pvk_alias_size = max(
            cute.cosize(layouts.value_key_alias_layout),
            cute.cosize(layouts.key_layout),
        )

        @cute.struct
        class SharedStorage:
            # dY/V staging and K-score staging are lifetime-disjoint, so they
            # share one CTA-local slab.
            s_pvk_alias: cute.struct.Align[
                cute.struct.MemRange[in_dtype, pvk_alias_size],
                16,
            ]
            s_q_tile: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)], 16
            ]
            s_u_prev: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.u_prev_layout)], 16
            ]
            s_b_prev: cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.b_prev_layout)], 16
            ]
            s_dlp: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.dlogprefix_layout)
                ],
                4,
            ]
            s_scale_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_scale_layout)
                ],
                4,
            ]
            s_inv_scale_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_inv_scale_layout)
                ],
                4,
            ]
            s_phase_full: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.full_phase_layout)
                ],
                16,
            ]
            s_tile_end_log: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ]
            s_tile_end_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ]
            s_tile_off_log: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_log_layout)
                ],
                4,
            ]
            s_tile_off_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.tile_prefix_phase_layout)
                ],
                16,
            ]
            s_row_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_scale_layout)
                ],
                4,
            ]
            s_inv_row_scale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_inv_scale_layout)
                ],
                4,
            ]
            s_phase_row: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.row_phase_layout)
                ],
                16,
            ]
            s_phase_col: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.column_phase_layout)
                ],
                16,
            ]
            s_tap_prev: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(layouts.tap_layout)],
                16,
            ]
            s_tap_curr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(layouts.tap_layout)],
                16,
            ]
            s_col_accum: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(layouts.column_accumulator_layout)
                ],
                16,
            ]
            tail_pad: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(tail_pad_layout)],
                16,
            ]

        return SharedStorage

    # Device copy/MMA helpers
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

    # Prefix metadata helpers
    @cute.jit
    def _initialize_dlogprefix_scratch(
        self,
        mDLogPrefix: cute.Tensor,
        s_dlp: cute.Tensor,
        *,
        batch_head_chunk: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < cutlass.Int32(self.L):
            s_dlp[tidx] = cutlass.Float32(mDLogPrefix[batch_head_chunk, tidx])
        cute.arch.barrier()

    @cute.jit
    def _compute_single_tile_prefix_metadata(self, prefix_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        eps = cutlass.Float32(1.0e-20)
        one = cutlass.Float32(1.0)
        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.kv_tile):
            seq_idx = lane
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
                cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.25 / LOG2_E)
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

            stable_logp = clamp_nonpositive_prefix_log(logp)
            scale = cute.math.exp2(
                stable_logp * cutlass.Float32(TWO_LOG2_E), fastmath=True
            )
            prefix_state.s_row_scale[lane] = scale
            prefix_state.s_inv_row_scale[lane] = one / scale
            prefix_state.s_phase_row[lane, 0] = phase_re
            prefix_state.s_phase_row[lane, 1] = phase_im
        cute.arch.barrier()

    @cute.jit
    def _compute_multi_tile_prefix_metadata(self, prefix_state: SimpleNamespace):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        num_warps = self.num_warps
        eps = cutlass.Float32(1.0e-20)
        one = cutlass.Float32(1.0)
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
        cute.arch.barrier()

        if warp == cutlass.Int32(0) and lane == cutlass.Int32(0):
            off_log = cutlass.Float32(0.0)
            off_phase_re = cutlass.Float32(1.0)
            off_phase_im = cutlass.Float32(0.0)
            for tile in cutlass.range_constexpr(n_tiles):
                prefix_state.s_tile_off_log[tile] = off_log
                prefix_state.s_tile_off_phase[tile, 0] = off_phase_re
                prefix_state.s_tile_off_phase[tile, 1] = off_phase_im
                last_log = cutlass.Float32(prefix_state.s_tile_end_log[tile])
                last_phase_re = cutlass.Float32(prefix_state.s_tile_end_phase[tile, 0])
                last_phase_im = cutlass.Float32(prefix_state.s_tile_end_phase[tile, 1])
                off_log = off_log + last_log
                off_phase_re, off_phase_im = complex_mul(
                    last_phase_re, last_phase_im, off_phase_re, off_phase_im
                )
        cute.arch.barrier()

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
                off_phase_re = cutlass.Float32(prefix_state.s_tile_off_phase[tile, 0])
                off_phase_im = cutlass.Float32(prefix_state.s_tile_off_phase[tile, 1])
                phase_re, phase_im = complex_mul(
                    phase_re, phase_im, off_phase_re, off_phase_im
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
        cute.arch.barrier()

    @cute.jit
    def _compute_prefix_metadata(self, prefix_state: SimpleNamespace):
        if cutlass.const_expr(self.L == self.kv_tile):
            self._compute_single_tile_prefix_metadata(prefix_state)
        else:
            self._compute_multi_tile_prefix_metadata(prefix_state)

    @cute.jit
    def _load_row_prefix_tile(self, prefix_state: SimpleNamespace, *, m_tile: int):
        if cutlass.const_expr(self.L == self.kv_tile):
            return

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.kv_tile):
            seq_idx = cutlass.Int32(m_tile * self.kv_tile) + lane
            prefix_state.s_row_scale[lane] = cutlass.Float32(
                prefix_state.s_scale_full[seq_idx]
            )
            prefix_state.s_phase_row[lane, 0] = cutlass.Float32(
                prefix_state.s_phase_full[seq_idx, 0]
            )
            prefix_state.s_phase_row[lane, 1] = cutlass.Float32(
                prefix_state.s_phase_full[seq_idx, 1]
            )
        cute.arch.barrier()

    @cute.jit
    def _load_col_prefix_tile(self, prefix_state: SimpleNamespace, *, n_tile: int):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        n0 = cutlass.Int32(n_tile * self.kv_tile)

        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.kv_tile):
            seq_idx = n0 + lane
            inv_scale = cutlass.Float32(prefix_state.s_inv_row_scale[lane])
            phase_re = cutlass.Float32(prefix_state.s_phase_row[lane, 0])
            phase_im = cutlass.Float32(prefix_state.s_phase_row[lane, 1])
            if cutlass.const_expr(self.L != self.kv_tile):
                inv_scale = cutlass.Float32(prefix_state.s_inv_scale_full[seq_idx])
                phase_re = cutlass.Float32(prefix_state.s_phase_full[seq_idx, 0])
                phase_im = cutlass.Float32(prefix_state.s_phase_full[seq_idx, 1])

            prefix_state.s_inv_row_scale[lane] = inv_scale
            prefix_state.s_phase_col[lane, 0] = phase_re
            prefix_state.s_phase_col[lane, 1] = phase_im

            tap_prev_re = cutlass.Float32(
                prefix_state.m_tap[prefix_state.batch_head_chunk, seq_idx, 0, 0]
            )
            tap_prev_im = cutlass.Float32(
                prefix_state.m_tap[prefix_state.batch_head_chunk, seq_idx, 0, 1]
            )
            tap_curr_re = cutlass.Float32(
                prefix_state.m_tap[prefix_state.batch_head_chunk, seq_idx, 1, 0]
            )
            tap_curr_im = cutlass.Float32(
                prefix_state.m_tap[prefix_state.batch_head_chunk, seq_idx, 1, 1]
            )
            prefix_state.s_tap_prev[lane, 0] = (
                tap_prev_re * phase_re + tap_prev_im * phase_im
            )
            prefix_state.s_tap_prev[lane, 1] = (
                tap_prev_re * phase_im - tap_prev_im * phase_re
            )
            prefix_state.s_tap_curr[lane, 0] = (
                tap_curr_re * phase_re + tap_curr_im * phase_im
            )
            prefix_state.s_tap_curr[lane, 1] = (
                tap_curr_re * phase_im - tap_curr_im * phase_re
            )
        cute.arch.barrier()

    # Boundary staging helpers
    @cute.jit
    def _load_prev_boundary_rows(self, boundary_state: SimpleNamespace):
        tidx, _, _ = cute.arch.thread_idx()
        prev_batch_head_chunk = cutlass.select_(
            boundary_state.chunk_index > cutlass.Int32(0),
            boundary_state.batch_head_chunk - cutlass.Int32(1),
            boundary_state.batch_head_chunk,
        )

        iters_p = (self.P + self.num_threads - 1) // self.num_threads
        for it in range(iters_p):
            p = tidx + cutlass.Int32(it * self.num_threads)
            if p < cutlass.Int32(self.P):
                boundary_prev = boundary_state.m_u_prev0[boundary_state.batch_head, p]
                chunk_prev = boundary_state.m_u[prev_batch_head_chunk, self.L - 1, 0, p]
                boundary_state.s_u_prev[p] = cutlass.select_(
                    boundary_state.chunk_index == cutlass.Int32(0),
                    boundary_prev,
                    chunk_prev,
                )

        iters_d = (self.D + self.num_threads - 1) // self.num_threads
        for it in range(iters_d):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D):
                boundary_prev = boundary_state.m_b_prev0[boundary_state.batch_head, d]
                chunk_prev = boundary_state.m_b[prev_batch_head_chunk, self.L - 1, 0, d]
                boundary_state.s_b_prev[d] = cutlass.select_(
                    boundary_state.chunk_index == cutlass.Int32(0),
                    boundary_prev,
                    chunk_prev,
                )
        cute.arch.barrier()

    # Score accumulation helpers
    @cute.jit
    def _load_rotated_query_tile(
        self,
        query_state: SimpleNamespace,
        *,
        m_tile: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m0 = cutlass.Int32(m_tile * self.kv_tile)

        g_query_tile = cute.local_tile(
            query_state.m_query[query_state.batch_head_chunk, None, 0, None],
            (self.kv_tile, self.D_padded),
            (m_tile, 0),
        )
        t_query_gmem = query_state.gmem_thr_copy_d.partition_S(g_query_tile)
        if cutlass.const_expr(self.D == self.D_padded):
            cute.copy(
                query_state.gmem_tiled_copy_d,
                t_query_gmem,
                query_state.query_tile_dst,
            )
        else:
            c_query_tile = cute.local_tile(
                query_state.coord_query,
                (self.kv_tile, self.D_padded),
                (m_tile, 0),
            )
            t_query_coord = query_state.gmem_thr_copy_d.partition_S(c_query_tile)
            t_query_pred = self._make_copy_column_predicate(
                query_state.query_tile_dst,
                t_query_coord,
                query_state.m_query.layout.shape[3],
            )
            self._copy_rows_with_zero_fill(
                query_state.gmem_tiled_copy_d,
                t_query_gmem,
                query_state.query_tile_dst,
                t_query_coord,
                t_query_pred,
                query_state.m_query.layout.shape[1],
            )
        cute.arch.barrier()

        total_pairs = cutlass.Int32(self.kv_tile * (self.D // 2))
        iters_pairs = (total_pairs + self.num_threads - 1) // self.num_threads
        for it in range(iters_pairs):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_pairs:
                row_local = idx // cutlass.Int32(self.D // 2)
                pair_idx = idx - row_local * cutlass.Int32(self.D // 2)
                seq_idx = m0 + row_local
                d0 = pair_idx * cutlass.Int32(2)
                out_re = cutlass.Float32(0.0).to(query_state.m_query.element_type)
                out_im = cutlass.Float32(0.0).to(query_state.m_query.element_type)
                if seq_idx < cutlass.Int32(self.L) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    query_re = cutlass.Float32(
                        query_state.s_query_tile[row_local, d0 + 0].to(cutlass.Float32)
                    )
                    query_im = cutlass.Float32(
                        query_state.s_query_tile[row_local, d0 + 1].to(cutlass.Float32)
                    )
                    phase_re = cutlass.Float32(query_state.s_phase_row[row_local, 0])
                    phase_im = cutlass.Float32(query_state.s_phase_row[row_local, 1])
                    query_re, query_im = conj_mul_phase(
                        query_re, query_im, phase_re, phase_im
                    )
                    out_re = safe_cast_to_dtype(
                        query_re, query_state.m_query.element_type
                    )
                    out_im = safe_cast_to_dtype(
                        query_im, query_state.m_query.element_type
                    )
                query_state.s_query_tile[row_local, d0 + 0] = out_re
                query_state.s_query_tile[row_local, d0 + 1] = out_im
        cute.arch.barrier()

    @cute.jit
    def _accumulate_dscore_for_pass(
        self,
        dscore_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        acc_dscore: cute.Tensor,
        *,
        m_tile: int,
        n_tile: int,
        pass_id: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        n0 = cutlass.Int32(n_tile * self.kv_tile)
        total_v_tile = cutlass.Int32(self.kv_tile * self.p_tile)
        iters_v_tile = (total_v_tile + self.num_threads - 1) // self.num_threads

        for p_tile_idx in cutlass.range_constexpr(self.P_padded // self.p_tile):
            p0 = cutlass.Int32(p_tile_idx * self.p_tile)
            if cutlass.const_expr(self.P == self.P_padded):
                g_dout = cute.local_tile(
                    dscore_state.m_output_grad[
                        dscore_state.batch_head_chunk, None, 0, None
                    ],
                    (self.kv_tile, self.p_tile),
                    (m_tile, p_tile_idx),
                )
                t_dout_gmem = dscore_state.gmem_thr_copy_p.partition_S(g_dout)
                cute.copy(
                    dscore_state.gmem_tiled_copy_p,
                    t_dout_gmem,
                    dscore_state.dout_tile_dst,
                )
                if cutlass.const_expr(pass_id == 1):
                    g_value = cute.local_tile(
                        dscore_state.m_value[
                            dscore_state.batch_head_chunk, None, 0, None
                        ],
                        (self.kv_tile, self.p_tile),
                        (n_tile, p_tile_idx),
                    )
                    t_value_gmem = dscore_state.gmem_thr_copy_p.partition_S(g_value)
                    cute.copy(
                        dscore_state.gmem_tiled_copy_p,
                        t_value_gmem,
                        dscore_state.value_tile_dst,
                    )
                else:
                    for it in range(iters_v_tile):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < total_v_tile:
                            row_local = idx // cutlass.Int32(self.p_tile)
                            p_local = idx - row_local * cutlass.Int32(self.p_tile)
                            p = p0 + p_local
                            value = cutlass.Float32(0.0).to(
                                dscore_state.m_value.element_type
                            )
                            src_row = n0 + row_local - cutlass.Int32(1)
                            if src_row >= cutlass.Int32(0):
                                value = dscore_state.m_value[
                                    dscore_state.batch_head_chunk, src_row, 0, p
                                ]
                            else:
                                value = dscore_state.s_u_prev[p]
                            dscore_state.s_value_tile[row_local, p_local] = value
            else:
                for it in range(iters_v_tile):
                    idx = tidx + cutlass.Int32(it * self.num_threads)
                    if idx < total_v_tile:
                        row_local = idx // cutlass.Int32(self.p_tile)
                        p_local = idx - row_local * cutlass.Int32(self.p_tile)
                        row = cutlass.Int32(m_tile * self.kv_tile) + row_local
                        p = p0 + p_local
                        d_out_val = cutlass.Float32(0.0).to(
                            dscore_state.m_output_grad.element_type
                        )
                        value = cutlass.Float32(0.0).to(
                            dscore_state.m_value.element_type
                        )
                        if row < cutlass.Int32(self.L) and p < cutlass.Int32(self.P):
                            d_out_val = dscore_state.m_output_grad[
                                dscore_state.batch_head_chunk, row, 0, p
                            ]
                            if cutlass.const_expr(pass_id == 1):
                                src_row = n0 + row_local
                                if src_row < cutlass.Int32(self.L):
                                    value = dscore_state.m_value[
                                        dscore_state.batch_head_chunk, src_row, 0, p
                                    ]
                            else:
                                src_row = n0 + row_local - cutlass.Int32(1)
                                if src_row >= cutlass.Int32(0):
                                    value = dscore_state.m_value[
                                        dscore_state.batch_head_chunk, src_row, 0, p
                                    ]
                                else:
                                    value = dscore_state.s_u_prev[p]
                        dscore_state.s_dout_tile[row_local, p_local] = d_out_val
                        dscore_state.s_value_tile[row_local, p_local] = value
            cute.arch.barrier()

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_dscore,
                mma_state.smem_tiled_copy_a,
                mma_state.smem_tiled_copy_b,
                mma_state.smem_dout_frag,
                mma_state.reg_dout_frag_view,
                mma_state.smem_value_frag,
                mma_state.reg_value_frag_view,
                mma_state.reg_dout_frag,
                mma_state.reg_value_frag,
            )

    @cute.jit
    def _load_tapped_key_tile(
        self,
        key_state: SimpleNamespace,
        *,
        n_tile: int,
        d_tile: int,
        pass_id: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        d_block = self._smem_block_size_d()
        d_base = cutlass.Int32(d_tile * d_block)
        n0 = cutlass.Int32(n_tile * self.kv_tile)
        total_k_tile = cutlass.Int32(self.kv_tile * d_block)
        iters_k_tile = (total_k_tile + self.num_threads - 1) // self.num_threads
        total_pairs = cutlass.Int32(self.kv_tile * (d_block // 2))
        iters_pairs = (total_pairs + self.num_threads - 1) // self.num_threads
        pairs_per_row = cutlass.Int32(d_block // 2)

        if cutlass.const_expr(self.D == self.D_padded) and cutlass.const_expr(
            pass_id == 1
        ):
            g_key = cute.local_tile(
                key_state.m_key[key_state.batch_head_chunk, None, 0, None],
                (self.kv_tile, d_block),
                (n_tile, d_tile),
            )
            t_key_gmem = key_state.gmem_thr_copy_d.partition_S(g_key)
            cute.copy(
                key_state.gmem_tiled_copy_d,
                t_key_gmem,
                key_state.key_tile_dst,
            )
        else:
            for it in range(iters_k_tile):
                idx = tidx + cutlass.Int32(it * self.num_threads)
                if idx < total_k_tile:
                    row_local = idx // cutlass.Int32(d_block)
                    d_local = idx - row_local * cutlass.Int32(d_block)
                    d = d_base + d_local
                    value = cutlass.Float32(0.0).to(key_state.m_key.element_type)
                    if d < cutlass.Int32(self.D):
                        seq_idx = n0 + row_local
                        if cutlass.const_expr(pass_id == 1):
                            if seq_idx < cutlass.Int32(self.L):
                                value = key_state.m_key[
                                    key_state.batch_head_chunk, seq_idx, 0, d
                                ]
                        else:
                            src_idx = seq_idx - cutlass.Int32(1)
                            if src_idx >= cutlass.Int32(0):
                                value = key_state.m_key[
                                    key_state.batch_head_chunk, src_idx, 0, d
                                ]
                            else:
                                value = key_state.s_b_prev[d]
                    key_state.s_key_tile[row_local, d_local] = value
        cute.arch.barrier()

        for it in range(iters_pairs):
            idx = tidx + cutlass.Int32(it * self.num_threads)
            if idx < total_pairs:
                row_local = idx // pairs_per_row
                pair_idx = idx - row_local * pairs_per_row
                seq_idx = n0 + row_local
                d0_local = pair_idx * cutlass.Int32(2)
                d0 = d_base + d0_local
                out_re = cutlass.Float32(0.0).to(key_state.m_key.element_type)
                out_im = cutlass.Float32(0.0).to(key_state.m_key.element_type)
                if seq_idx < cutlass.Int32(self.L) and cute.elem_less(
                    d0 + cutlass.Int32(1), cutlass.Int32(self.D)
                ):
                    key_re = cutlass.Float32(
                        key_state.s_key_tile[row_local, d0_local + 0].to(
                            cutlass.Float32
                        )
                    )
                    key_im = cutlass.Float32(
                        key_state.s_key_tile[row_local, d0_local + 1].to(
                            cutlass.Float32
                        )
                    )
                    tap_re = cutlass.Float32(0.0)
                    tap_im = cutlass.Float32(0.0)
                    if cutlass.const_expr(pass_id == 1):
                        tap_re = cutlass.Float32(key_state.s_tap_curr[row_local, 0])
                        tap_im = cutlass.Float32(key_state.s_tap_curr[row_local, 1])
                    else:
                        tap_re = cutlass.Float32(key_state.s_tap_prev[row_local, 0])
                        tap_im = cutlass.Float32(key_state.s_tap_prev[row_local, 1])
                    out_re = safe_cast_to_dtype(
                        key_re * tap_re + key_im * tap_im,
                        key_state.m_key.element_type,
                    )
                    out_im = safe_cast_to_dtype(
                        key_re * tap_im - key_im * tap_re,
                        key_state.m_key.element_type,
                    )
                key_state.s_key_tile[row_local, d0_local + 0] = out_re
                key_state.s_key_tile[row_local, d0_local + 1] = out_im
        cute.arch.barrier()

    @cute.jit
    def _accumulate_score_for_pass(
        self,
        score_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        acc_score: cute.Tensor,
        *,
        n_tile: int,
        pass_id: cutlass.Constexpr,
    ):
        d_block = self._smem_block_size_d()
        n_d_tiles = self.D_padded // d_block

        for d_tile in cutlass.range_constexpr(n_d_tiles):
            s_query_block = cute.local_tile(
                score_state.s_query_tile,
                (self.kv_tile, d_block),
                (0, d_tile),
            )
            t_reg_query = mma_state.thr_mma.make_fragment_A(
                mma_state.thr_mma.partition_A(s_query_block)
            )
            t_smem_query = mma_state.smem_thr_copy_a.partition_S(s_query_block)
            t_reg_query_view = mma_state.smem_thr_copy_a.retile(t_reg_query)

            self._load_tapped_key_tile(
                score_state,
                n_tile=n_tile,
                d_tile=d_tile,
                pass_id=pass_id,
            )
            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                acc_score,
                mma_state.smem_tiled_copy_a,
                mma_state.smem_tiled_copy_b,
                t_smem_query,
                t_reg_query_view,
                mma_state.smem_key_frag,
                mma_state.reg_key_frag_view,
                t_reg_query,
                mma_state.reg_key_frag,
            )

    @cute.jit
    def _accumulate_dlogprefix_from_score_tiles(
        self,
        epilogue_state: SimpleNamespace,
        acc_dscore: cute.Tensor,
        acc_score: cute.Tensor,
        *,
        m_tile: int,
        n_tile: int,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        m0 = cutlass.Int32(m_tile * self.kv_tile)
        n0 = cutlass.Int32(n_tile * self.kv_tile)
        c_score = cute.local_tile(
            epilogue_state.coord_score,
            (self.kv_tile, self.kv_tile),
            (m_tile, n_tile),
        )
        t_score_coord = epilogue_state.thr_mma.partition_C(c_score)
        t_score_coord_mn = self._make_accumulator_mn_view(t_score_coord)
        acc_dscore_mn = self._make_accumulator_mn_view(acc_dscore)
        acc_score_mn = self._make_accumulator_mn_view(acc_score)

        col_acc0 = cutlass.Float32(0.0)
        col_acc1 = cutlass.Float32(0.0)
        col_acc2 = cutlass.Float32(0.0)
        col_acc3 = cutlass.Float32(0.0)
        diag_tile = m_tile == n_tile
        for r in cutlass.range_constexpr(cute.size(acc_score_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            row_scale = cutlass.Float32(0.0)
            if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                row_scale = cutlass.Float32(epilogue_state.s_row_scale[row_idx - m0])
            for c in cutlass.range_constexpr(cute.size(acc_score_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                col_local = col_idx - n0
                if cute.elem_less(row_idx, cutlass.Int32(self.L)) and cute.elem_less(
                    col_idx, cutlass.Int32(self.L)
                ):
                    if (not diag_tile) or cute.elem_less(
                        col_idx, row_idx + cutlass.Int32(1)
                    ):
                        inv_row_scale = cutlass.Float32(
                            epilogue_state.s_inv_row_scale[col_local]
                        )
                        scale = row_scale * inv_row_scale
                        prod = acc_dscore_mn[r, c] * (acc_score_mn[r, c] * scale)
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
            epilogue_state.s_col_accum[warp, lane, 0] = col_acc0
            epilogue_state.s_col_accum[warp, lane, 1] = col_acc1
            epilogue_state.s_col_accum[warp, lane, 2] = col_acc2
            epilogue_state.s_col_accum[warp, lane, 3] = col_acc3
        cute.arch.barrier()

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

            col_sum0 = cutlass.Float32(
                epilogue_state.s_col_accum[src_warp0, src_lane, 0]
            )
            col_sum0 = col_sum0 + cutlass.Float32(
                epilogue_state.s_col_accum[src_warp1, src_lane, 0]
            )
            col_sum1 = cutlass.Float32(
                epilogue_state.s_col_accum[src_warp0, src_lane, 1]
            )
            col_sum1 = col_sum1 + cutlass.Float32(
                epilogue_state.s_col_accum[src_warp1, src_lane, 1]
            )
            col_sum2 = cutlass.Float32(
                epilogue_state.s_col_accum[src_warp0, src_lane, 2]
            )
            col_sum2 = col_sum2 + cutlass.Float32(
                epilogue_state.s_col_accum[src_warp1, src_lane, 2]
            )
            col_sum3 = cutlass.Float32(
                epilogue_state.s_col_accum[src_warp0, src_lane, 3]
            )
            col_sum3 = col_sum3 + cutlass.Float32(
                epilogue_state.s_col_accum[src_warp1, src_lane, 3]
            )

            col0 = n0 + base
            col1 = col0 + cutlass.Int32(1)
            col2 = col0 + cutlass.Int32(16)
            col3 = col2 + cutlass.Int32(1)
            if col0 < cutlass.Int32(self.L):
                epilogue_state.s_dlp[col0] = (
                    epilogue_state.s_dlp[col0] - cutlass.Float32(2.0) * col_sum0
                )
            if col1 < cutlass.Int32(self.L):
                epilogue_state.s_dlp[col1] = (
                    epilogue_state.s_dlp[col1] - cutlass.Float32(2.0) * col_sum1
                )
            if col2 < cutlass.Int32(self.L):
                epilogue_state.s_dlp[col2] = (
                    epilogue_state.s_dlp[col2] - cutlass.Float32(2.0) * col_sum2
                )
            if col3 < cutlass.Int32(self.L):
                epilogue_state.s_dlp[col3] = (
                    epilogue_state.s_dlp[col3] - cutlass.Float32(2.0) * col_sum3
                )
        cute.arch.barrier()

    # Kernel orchestration helpers
    def _make_prefix_state(
        self,
        *,
        batch_head_chunk,
        m_transition: cute.Tensor,
        m_tap: cute.Tensor,
        s_scale_full: cute.Tensor,
        s_inv_scale_full: cute.Tensor,
        s_phase_full: cute.Tensor,
        s_tile_end_log: cute.Tensor,
        s_tile_end_phase: cute.Tensor,
        s_tile_off_log: cute.Tensor,
        s_tile_off_phase: cute.Tensor,
        s_row_scale: cute.Tensor,
        s_inv_row_scale: cute.Tensor,
        s_phase_row: cute.Tensor,
        s_phase_col: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_tap_curr: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_transition=m_transition,
            m_tap=m_tap,
            s_scale_full=s_scale_full,
            s_inv_scale_full=s_inv_scale_full,
            s_phase_full=s_phase_full,
            s_tile_end_log=s_tile_end_log,
            s_tile_end_phase=s_tile_end_phase,
            s_tile_off_log=s_tile_off_log,
            s_tile_off_phase=s_tile_off_phase,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            s_phase_row=s_phase_row,
            s_phase_col=s_phase_col,
            s_tap_prev=s_tap_prev,
            s_tap_curr=s_tap_curr,
        )

    def _make_boundary_state(
        self,
        *,
        batch_head_chunk,
        batch_head,
        chunk_index,
        m_u: cute.Tensor,
        m_b: cute.Tensor,
        m_u_prev0: cute.Tensor,
        m_b_prev0: cute.Tensor,
        s_u_prev: cute.Tensor,
        s_b_prev: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            batch_head=batch_head,
            chunk_index=chunk_index,
            m_u=m_u,
            m_b=m_b,
            m_u_prev0=m_u_prev0,
            m_b_prev0=m_b_prev0,
            s_u_prev=s_u_prev,
            s_b_prev=s_b_prev,
        )

    def _make_query_state(
        self,
        *,
        batch_head_chunk,
        m_query: cute.Tensor,
        coord_query: cute.Tensor,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_thr_copy_d,
        query_tile_dst: cute.Tensor,
        s_query_tile: cute.Tensor,
        s_phase_row: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_query=m_query,
            coord_query=coord_query,
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_thr_copy_d=gmem_thr_copy_d,
            query_tile_dst=query_tile_dst,
            s_query_tile=s_query_tile,
            s_phase_row=s_phase_row,
        )

    def _make_dscore_state(
        self,
        *,
        batch_head_chunk,
        m_output_grad: cute.Tensor,
        m_value: cute.Tensor,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_thr_copy_p,
        dout_tile_dst: cute.Tensor,
        value_tile_dst: cute.Tensor,
        s_dout_tile: cute.Tensor,
        s_value_tile: cute.Tensor,
        s_u_prev: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_output_grad=m_output_grad,
            m_value=m_value,
            gmem_tiled_copy_p=gmem_tiled_copy_p,
            gmem_thr_copy_p=gmem_thr_copy_p,
            dout_tile_dst=dout_tile_dst,
            value_tile_dst=value_tile_dst,
            s_dout_tile=s_dout_tile,
            s_value_tile=s_value_tile,
            s_u_prev=s_u_prev,
        )

    def _make_score_state(
        self,
        *,
        batch_head_chunk,
        m_key: cute.Tensor,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_thr_copy_d,
        key_tile_dst: cute.Tensor,
        s_key_tile: cute.Tensor,
        s_b_prev: cute.Tensor,
        s_tap_prev: cute.Tensor,
        s_tap_curr: cute.Tensor,
        s_query_tile: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_key=m_key,
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_thr_copy_d=gmem_thr_copy_d,
            key_tile_dst=key_tile_dst,
            s_key_tile=s_key_tile,
            s_b_prev=s_b_prev,
            s_tap_prev=s_tap_prev,
            s_tap_curr=s_tap_curr,
            s_query_tile=s_query_tile,
        )

    def _make_mma_state(
        self,
        *,
        tiled_mma: cute.TiledMma,
        thr_mma,
        smem_tiled_copy_a: cute.TiledCopy,
        smem_tiled_copy_b: cute.TiledCopy,
        smem_thr_copy_a,
        smem_dout_frag: cute.Tensor,
        reg_dout_frag_view: cute.Tensor,
        reg_dout_frag: cute.Tensor,
        smem_value_frag: cute.Tensor,
        reg_value_frag_view: cute.Tensor,
        reg_value_frag: cute.Tensor,
        smem_key_frag: cute.Tensor,
        reg_key_frag_view: cute.Tensor,
        reg_key_frag: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            tiled_mma=tiled_mma,
            thr_mma=thr_mma,
            smem_tiled_copy_a=smem_tiled_copy_a,
            smem_tiled_copy_b=smem_tiled_copy_b,
            smem_thr_copy_a=smem_thr_copy_a,
            smem_dout_frag=smem_dout_frag,
            reg_dout_frag_view=reg_dout_frag_view,
            reg_dout_frag=reg_dout_frag,
            smem_value_frag=smem_value_frag,
            reg_value_frag_view=reg_value_frag_view,
            reg_value_frag=reg_value_frag,
            smem_key_frag=smem_key_frag,
            reg_key_frag_view=reg_key_frag_view,
            reg_key_frag=reg_key_frag,
        )

    def _make_epilogue_state(
        self,
        *,
        coord_score: cute.Tensor,
        thr_mma,
        s_row_scale: cute.Tensor,
        s_inv_row_scale: cute.Tensor,
        s_col_accum: cute.Tensor,
        s_dlp: cute.Tensor,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            coord_score=coord_score,
            thr_mma=thr_mma,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            s_col_accum=s_col_accum,
            s_dlp=s_dlp,
        )

    @cute.jit
    def _run_causal_tile_mainloop(
        self,
        prefix_state: SimpleNamespace,
        query_state: SimpleNamespace,
        dscore_state: SimpleNamespace,
        score_state: SimpleNamespace,
        mma_state: SimpleNamespace,
        epilogue_state: SimpleNamespace,
        score_tile_acc_shape,
    ):
        for m_tile in cutlass.range_constexpr(self.num_kv_tiles):
            self._load_row_prefix_tile(prefix_state, m_tile=m_tile)
            self._load_rotated_query_tile(query_state, m_tile=m_tile)

            for n_tile in cutlass.range_constexpr(m_tile + 1):
                self._load_col_prefix_tile(prefix_state, n_tile=n_tile)

                for pass_id in cutlass.range_constexpr(2):
                    acc_dscore_tile = cute.make_rmem_tensor(
                        score_tile_acc_shape,
                        cutlass.Float32,
                    )
                    acc_dscore_tile.fill(0.0)
                    self._accumulate_dscore_for_pass(
                        dscore_state,
                        mma_state,
                        acc_dscore_tile,
                        m_tile=m_tile,
                        n_tile=n_tile,
                        pass_id=pass_id,
                    )

                    acc_score_tile = cute.make_rmem_tensor(
                        score_tile_acc_shape,
                        cutlass.Float32,
                    )
                    acc_score_tile.fill(0.0)
                    self._accumulate_score_for_pass(
                        score_state,
                        mma_state,
                        acc_score_tile,
                        n_tile=n_tile,
                        pass_id=pass_id,
                    )
                    self._accumulate_dlogprefix_from_score_tiles(
                        epilogue_state,
                        acc_dscore_tile,
                        acc_score_tile,
                        m_tile=m_tile,
                        n_tile=n_tile,
                    )

    # Host launch API
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
        mDLogPrefix: cute.Tensor,
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
        if cutlass.const_expr(mDLogPrefix.element_type != cutlass.Float32):
            raise TypeError("dlogprefix must be Float32.")

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
        if cutlass.const_expr(mDLogPrefix.shape[1] != self.L):
            raise ValueError("dlogprefix must be (BHC, L).")

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
        mDLogPrefix: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        bundle = self._make_kernel_bundle(mU.element_type)
        layouts = bundle.layouts
        copies = bundle.copies
        launch_kwargs = {
            "grid": (1, 1, cute.size(mU.shape[0])),
            "block": [self.num_threads, 1, 1],
            "smem": bundle.smem_size,
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
            mDLogPrefix,
            layouts.u_prev_layout,
            layouts.b_prev_layout,
            layouts.dlogprefix_layout,
            layouts.value_key_alias_layout,
            layouts.grad_output_layout,
            layouts.value_layout,
            layouts.query_layout,
            layouts.key_layout,
            layouts.column_accumulator_layout,
            layouts.full_scale_layout,
            layouts.full_inv_scale_layout,
            layouts.full_phase_layout,
            layouts.tile_prefix_log_layout,
            layouts.tile_prefix_phase_layout,
            layouts.row_scale_layout,
            layouts.row_inv_scale_layout,
            layouts.row_phase_layout,
            layouts.column_phase_layout,
            layouts.tap_layout,
            copies.gmem_tiled_copy_d,
            copies.gmem_tiled_copy_p,
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
        mDLogPrefix: cute.Tensor,
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
            mDLogPrefix,
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
            mDLogPrefix,
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
        mDLogPrefix: cute.Tensor,
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
            mDLogPrefix,
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
            mDLogPrefix,
            stream=stream,
        )

    # Kernel entrypoint
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
        mDLogPrefix: cute.Tensor,
        u_prev_layout: cute.Layout,
        b_prev_layout: cute.Layout,
        dlogprefix_layout: cute.Layout,
        value_key_alias_layout: cute.Layout | cute.ComposedLayout,
        grad_output_layout: cute.Layout | cute.ComposedLayout,
        value_layout: cute.Layout | cute.ComposedLayout,
        query_layout: cute.Layout | cute.ComposedLayout,
        key_layout: cute.Layout | cute.ComposedLayout,
        column_accumulator_layout: cute.Layout,
        full_scale_layout: cute.Layout,
        full_inv_scale_layout: cute.Layout,
        full_phase_layout: cute.Layout,
        tile_prefix_log_layout: cute.Layout,
        tile_prefix_phase_layout: cute.Layout,
        row_scale_layout: cute.Layout,
        row_inv_scale_layout: cute.Layout,
        row_phase_layout: cute.Layout,
        column_phase_layout: cute.Layout,
        tap_layout: cute.Layout,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_tiled_copy_p: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, batch_head_chunk = cute.arch.block_idx()

        batch_head_count = mU_prev0.shape[0]
        batch_head_chunk_count = mU.shape[0]
        n_chunks = batch_head_chunk_count // batch_head_count
        batch_head = batch_head_chunk // n_chunks
        chunk_index = batch_head_chunk - batch_head * n_chunks

        d_padded = self.D_padded
        kv_tile = self.kv_tile
        p_tile = self.p_tile

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_pvk_alias = storage.s_pvk_alias.get_tensor(value_key_alias_layout)
        s_dout_tile = cute.local_tile(s_pvk_alias, (kv_tile, p_tile), (0, 0))
        s_value_tile = cute.local_tile(s_pvk_alias, (kv_tile, p_tile), (0, 1))
        s_query_tile = storage.s_q_tile.get_tensor(query_layout)
        s_key_tile = cute.make_tensor(s_pvk_alias.iterator.align(16), key_layout)
        s_col_accum = storage.s_col_accum.get_tensor(column_accumulator_layout)
        s_u_prev = storage.s_u_prev.get_tensor(u_prev_layout)
        s_b_prev = storage.s_b_prev.get_tensor(b_prev_layout)
        s_dlp = storage.s_dlp.get_tensor(dlogprefix_layout)

        s_scale_full = storage.s_scale_full.get_tensor(full_scale_layout)
        s_inv_scale_full = storage.s_inv_scale_full.get_tensor(full_inv_scale_layout)
        s_phase_full = storage.s_phase_full.get_tensor(full_phase_layout)
        s_tile_end_log = storage.s_tile_end_log.get_tensor(tile_prefix_log_layout)
        s_tile_end_phase = storage.s_tile_end_phase.get_tensor(tile_prefix_phase_layout)
        s_tile_off_log = storage.s_tile_off_log.get_tensor(tile_prefix_log_layout)
        s_tile_off_phase = storage.s_tile_off_phase.get_tensor(tile_prefix_phase_layout)

        s_row_scale = storage.s_row_scale.get_tensor(row_scale_layout)
        s_inv_row_scale = storage.s_inv_row_scale.get_tensor(row_inv_scale_layout)
        s_phase_row = storage.s_phase_row.get_tensor(row_phase_layout)
        s_phase_col = storage.s_phase_col.get_tensor(column_phase_layout)
        s_tap_prev = storage.s_tap_prev.get_tensor(tap_layout)
        s_tap_curr = storage.s_tap_curr.get_tensor(tap_layout)
        self._initialize_dlogprefix_scratch(
            mDLogPrefix,
            s_dlp,
            batch_head_chunk=batch_head_chunk,
        )

        prefix_state = self._make_prefix_state(
            batch_head_chunk=batch_head_chunk,
            m_transition=mM,
            m_tap=mK,
            s_scale_full=s_scale_full,
            s_inv_scale_full=s_inv_scale_full,
            s_phase_full=s_phase_full,
            s_tile_end_log=s_tile_end_log,
            s_tile_end_phase=s_tile_end_phase,
            s_tile_off_log=s_tile_off_log,
            s_tile_off_phase=s_tile_off_phase,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            s_phase_row=s_phase_row,
            s_phase_col=s_phase_col,
            s_tap_prev=s_tap_prev,
            s_tap_curr=s_tap_curr,
        )
        self._compute_prefix_metadata(prefix_state)

        boundary_state = self._make_boundary_state(
            batch_head_chunk=batch_head_chunk,
            batch_head=batch_head,
            chunk_index=chunk_index,
            m_u=mU,
            m_b=mB,
            m_u_prev0=mU_prev0,
            m_b_prev0=mB_prev0,
            s_u_prev=s_u_prev,
            s_b_prev=s_b_prev,
        )
        self._load_prev_boundary_rows(boundary_state)

        smem_copy_atom_a = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_a = cute.make_tiled_copy_A(smem_copy_atom_a, tiled_mma)
        smem_tiled_copy_b = cute.make_tiled_copy_B(smem_copy_atom_b, tiled_mma)
        thr_mma = tiled_mma.get_slice(tidx)

        gmem_thr_copy_d = gmem_tiled_copy_d.get_slice(tidx)
        gmem_thr_copy_p = gmem_tiled_copy_p.get_slice(tidx)
        thr_copy_a = smem_tiled_copy_a.get_slice(tidx)
        thr_copy_b = smem_tiled_copy_b.get_slice(tidx)

        reg_dout_frag = thr_mma.make_fragment_A(thr_mma.partition_A(s_dout_tile))
        smem_dout_frag = thr_copy_a.partition_S(s_dout_tile)
        reg_dout_frag_view = thr_copy_a.retile(reg_dout_frag)
        reg_value_frag = thr_mma.make_fragment_B(thr_mma.partition_B(s_value_tile))
        smem_value_frag = thr_copy_b.partition_S(s_value_tile)
        reg_value_frag_view = thr_copy_b.retile(reg_value_frag)
        dout_tile_dst = gmem_thr_copy_p.partition_D(s_dout_tile)
        value_tile_dst = gmem_thr_copy_p.partition_D(s_value_tile)
        query_tile_dst = gmem_thr_copy_d.partition_D(s_query_tile)
        reg_key_frag = thr_mma.make_fragment_B(thr_mma.partition_B(s_key_tile))
        smem_key_frag = thr_copy_b.partition_S(s_key_tile)
        reg_key_frag_view = thr_copy_b.retile(reg_key_frag)
        key_tile_dst = gmem_thr_copy_d.partition_D(s_key_tile)
        score_coord = cute.make_identity_tensor(
            (mU.shape[0], self.L, mU.shape[2], self.L)
        )[batch_head_chunk, None, 0, None]
        query_coord = cute.make_identity_tensor(
            (mU.shape[0], self.L, mU.shape[2], d_padded)
        )[batch_head_chunk, None, 0, None]
        score_tile_acc_shape = thr_mma.partition_shape_C((kv_tile, kv_tile))

        query_state = self._make_query_state(
            batch_head_chunk=batch_head_chunk,
            m_query=mC,
            coord_query=query_coord,
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_thr_copy_d=gmem_thr_copy_d,
            query_tile_dst=query_tile_dst,
            s_query_tile=s_query_tile,
            s_phase_row=s_phase_row,
        )
        dscore_state = self._make_dscore_state(
            batch_head_chunk=batch_head_chunk,
            m_output_grad=mDOut,
            m_value=mU,
            gmem_tiled_copy_p=gmem_tiled_copy_p,
            gmem_thr_copy_p=gmem_thr_copy_p,
            dout_tile_dst=dout_tile_dst,
            value_tile_dst=value_tile_dst,
            s_dout_tile=s_dout_tile,
            s_value_tile=s_value_tile,
            s_u_prev=s_u_prev,
        )
        score_state = self._make_score_state(
            batch_head_chunk=batch_head_chunk,
            m_key=mB,
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_thr_copy_d=gmem_thr_copy_d,
            key_tile_dst=key_tile_dst,
            s_key_tile=s_key_tile,
            s_b_prev=s_b_prev,
            s_tap_prev=s_tap_prev,
            s_tap_curr=s_tap_curr,
            s_query_tile=s_query_tile,
        )
        mma_state = self._make_mma_state(
            tiled_mma=tiled_mma,
            thr_mma=thr_mma,
            smem_tiled_copy_a=smem_tiled_copy_a,
            smem_tiled_copy_b=smem_tiled_copy_b,
            smem_thr_copy_a=thr_copy_a,
            smem_dout_frag=smem_dout_frag,
            reg_dout_frag_view=reg_dout_frag_view,
            reg_dout_frag=reg_dout_frag,
            smem_value_frag=smem_value_frag,
            reg_value_frag_view=reg_value_frag_view,
            reg_value_frag=reg_value_frag,
            smem_key_frag=smem_key_frag,
            reg_key_frag_view=reg_key_frag_view,
            reg_key_frag=reg_key_frag,
        )
        epilogue_state = self._make_epilogue_state(
            coord_score=score_coord,
            thr_mma=thr_mma,
            s_row_scale=s_row_scale,
            s_inv_row_scale=s_inv_row_scale,
            s_col_accum=s_col_accum,
            s_dlp=s_dlp,
        )

        self._run_causal_tile_mainloop(
            prefix_state,
            query_state,
            dscore_state,
            score_state,
            mma_state,
            epilogue_state,
            score_tile_acc_shape,
        )

        cute.arch.barrier()
        if tidx < cutlass.Int32(self.L):
            mDLogPrefix[batch_head_chunk, tidx] = s_dlp[tidx]


__all__ = ["ChunkScanBwdDLPAmpere"]
