"""CuTe forward kernel for the ``v2x2ssd`` chunk-scan stage.

``ChunkScanFwdAmpere`` is the live Ampere tensor-core implementation used by
the forward path. It reconstructs prefix magnitude/phase metadata from ``M``,
accumulates the off-term from ``Z0``, runs the two diagonal scan passes using
the two complex taps in ``K``, and writes the stage output ``Out``.

Tensor contracts:

- ``U``: ``(BHC, L, 1, P)`` fp16/bf16 value input
- ``B``: ``(BHC, L, 1, D)`` fp16/bf16 key-side packed complex input
- ``C``: ``(BHC, L, 1, D)`` fp16/bf16 query-side packed complex input
- ``M``: ``(BHC, L, 2)`` fp32 packed complex transitions
- ``K``: ``(BHC, L, 2, 2)`` fp32 packed complex taps for the previous/current
  diagonal passes
- ``Z0``: ``(BHC, P, 1, D)`` fp32 packed complex initial state
- ``U_prev0``: ``(BH, P)`` fp16/bf16 chunk-0 boundary value row
- ``B_prev0``: ``(BH, D)`` fp16/bf16 chunk-0 boundary key row
- ``Out``: ``(BHC, L, 1, P)`` fp16/bf16/fp32 stage output

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

LOG2_E = 1.4426950408889634


@dataclass(frozen=True)
class ChunkScanSupportInfo:
    tile_family_ok: bool
    expected_m_block_size: int | None
    smem_capacity_bytes: int
    compute_smem_bytes: int
    output_smem_bytes: int

    @property
    def required_smem_bytes(self) -> int:
        return max(self.compute_smem_bytes, self.output_smem_bytes)

    @property
    def supported(self) -> bool:
        return (
            self.tile_family_ok and self.required_smem_bytes <= self.smem_capacity_bytes
        )


@dataclass(frozen=True)
class ChunkScanLayoutBundle:
    query_layout: object
    bz_alias_layout: object
    key_layout: object
    z_layout: object
    value_layout: object
    output_layout: object


@dataclass(frozen=True)
class ChunkScanCopyBundle:
    gmem_tiled_copy_d: object
    gmem_tiled_copy_p: object
    gmem_tiled_copy_output: object


@dataclass(frozen=True)
class ChunkScanKernelBundle:
    layouts: ChunkScanLayoutBundle
    copies: ChunkScanCopyBundle
    tiled_mma: object
    shared_storage_cls: object
    compute_smem_bytes: int
    output_smem_bytes: int

    @property
    def smem_size(self) -> int:
        return max(self.compute_smem_bytes, self.output_smem_bytes)


class ChunkScanFwdAmpere:
    """Ampere tensor-core forward kernel for the ``v2x2ssd`` chunk-scan op."""

    _SUPPORTED_TILE_FAMILIES: tuple[tuple[int, int], ...] = (
        (64, 128),
        (32, 64),
        (16, 32),
    )
    _SUPPORT_INFO_CACHE: ClassVar[dict[tuple[object, ...], ChunkScanSupportInfo]] = {}
    _EXPECTED_M_BLOCK_BY_THREADS = {
        threads: m_block for m_block, threads in _SUPPORTED_TILE_FAMILIES
    }

    def __init__(
        self,
        *,
        D: int,
        P: int,
        L: int,
        m_block_size: int = 128,
        n_block_size: int = 64,
        num_threads: int = 128,
    ):
        self.D = int(D)
        self.P = int(P)
        self.L = int(L)
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.num_threads = int(num_threads)

        if self.m_block_size % 16 != 0:
            raise ValueError("m_block_size must be a multiple of 16.")
        if self.n_block_size % 16 != 0:
            raise ValueError("n_block_size must be a multiple of 16.")
        if self.L % self.n_block_size != 0:
            raise ValueError("L must be divisible by n_block_size.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.D % 2 != 0:
            raise ValueError("D must be even for packed complex pairs.")

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

    @property
    def n_block_max(self) -> int:
        return (self.L + self.n_block_size - 1) // self.n_block_size

    @property
    def num_warps(self) -> int:
        return self.num_threads // 32

    def _expected_m_block_size(self) -> int | None:
        return self._EXPECTED_M_BLOCK_BY_THREADS.get(self.num_threads)

    def _tile_family_supported(self) -> bool:
        expected = self._expected_m_block_size()
        return (
            expected is not None
            and self.m_block_size == expected
            and self.m_block_size <= self.L
        )

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

    def _v_smem_elems(self) -> int:
        return int(self.n_block_size) * int(self.P_padded)

    def _v_alias_in_bz(self) -> bool:
        """Return whether the staged value tile fits inside the ``bz_alias`` slab."""
        return self._v_smem_elems() <= self._b_smem_elems()

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

    def _compute_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        """Estimate dynamic SMEM for support checks without constructing layouts."""
        in_bytes = in_dtype.width // 8
        # Keep this accounting aligned with `_make_shared_storage`. It must stay
        # host-only because the CuTe layout builders need an MLIR context.
        compute_fields = [
            ("q_tile", self._q_smem_elems() * in_bytes),
            ("bz_alias", self._b_smem_elems() * in_bytes),
        ]
        if not self._v_alias_in_bz():
            compute_fields.append(("v_tile", self._v_smem_elems() * in_bytes))
        compute_fields.extend(
            [
                ("query_scale", self.m_block_size * 4),
                ("key_scale", self.n_block_size * 4),
                ("log_prefix", self.L * 4),
                ("phase_re", self.L * 4),
                ("phase_im", self.L * 4),
                ("tap_phase_re", self.n_block_size * 4),
                ("tap_phase_im", self.n_block_size * 4),
                ("warp_log_total", self.num_warps * 4),
                ("warp_log_offset", self.num_warps * 4),
                ("warp_phase_total", self.num_warps * 2 * 4),
                ("warp_phase_offset", self.num_warps * 2 * 4),
            ]
        )
        return self._struct_size_bytes(
            [(field_bytes, 16) for _, field_bytes in compute_fields]
        )

    def _output_smem_bytes(self, out_dtype: type[cutlass.Numeric]) -> int:
        out_bytes = out_dtype.width // 8
        output_fields = [("out_tile", self.m_block_size * self.P_padded * out_bytes)]
        return self._struct_size_bytes(
            [(field_bytes, 16) for _, field_bytes in output_fields]
        )

    def _make_output_shared_storage(
        self,
        out_dtype: type[cutlass.Numeric],
        layouts: ChunkScanLayoutBundle,
    ):
        @cute.struct
        class SharedStorage:
            out_tile: cute.struct.Align[
                cute.struct.MemRange[out_dtype, cute.cosize(layouts.output_layout)],
                16,
            ]

        return SharedStorage

    def _make_tiled_mma(self, in_dtype: type[cutlass.Numeric]):
        return cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(in_dtype, cutlass.Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

    def _make_kernel_bundle(
        self,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
    ) -> ChunkScanKernelBundle:
        layouts = self._make_layout_bundle(out_dtype)
        copies = self._make_copy_bundle(in_dtype, out_dtype)
        shared_storage_cls = self._make_shared_storage(in_dtype, layouts)
        output_storage_cls = self._make_output_shared_storage(out_dtype, layouts)
        return ChunkScanKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=self._make_tiled_mma(in_dtype),
            shared_storage_cls=shared_storage_cls,
            compute_smem_bytes=int(shared_storage_cls.size_in_bytes()),
            output_smem_bytes=int(output_storage_cls.size_in_bytes()),
        )

    def support_info(
        self,
        dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> ChunkScanSupportInfo:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return ChunkScanSupportInfo(False, self._expected_m_block_size(), 0, 0, 0)
        if out_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return ChunkScanSupportInfo(False, self._expected_m_block_size(), 0, 0, 0)
        if self.D % 8 != 0 or self.P % 8 != 0:
            return ChunkScanSupportInfo(False, self._expected_m_block_size(), 0, 0, 0)

        if device_index is None:
            device_key = (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
            )
        else:
            device_key = int(device_index)
        cache_key = (
            type(self),
            self.D,
            self.P,
            self.L,
            self.m_block_size,
            self.n_block_size,
            self.num_threads,
            dtype,
            out_dtype,
            device_key,
        )
        cached = self._SUPPORT_INFO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        info = ChunkScanSupportInfo(
            tile_family_ok=self._tile_family_supported(),
            expected_m_block_size=self._expected_m_block_size(),
            smem_capacity_bytes=self._smem_capacity_bytes(device_key),
            compute_smem_bytes=self._compute_smem_bytes(dtype),
            output_smem_bytes=self._output_smem_bytes(out_dtype),
        )
        self._SUPPORT_INFO_CACHE[cache_key] = info
        return info

    def can_implement(
        self,
        dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> bool:
        return self.support_info(dtype, out_dtype, device_index=device_index).supported

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
        # All warps must finish consuming the staged shared tiles before the
        # caller can repopulate the same storage for the next slice.
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
    def _accumulate_output_from_scores(
        self,
        tiled_mma: cute.TiledMma,
        acc_score: cute.Tensor,
        acc_output: cute.Tensor,
        score_dtype: type[cutlass.Numeric],
        smem_tiled_copy_value: cute.TiledCopy,
        t_smem_value_transposed: cute.Tensor,
        t_reg_value_transposed_view: cute.Tensor,
        t_reg_value_transposed: cute.Tensor,
    ):
        r_score = cute.make_rmem_tensor_like(acc_score, score_dtype)
        r_score.store(acc_score.load().to(score_dtype))
        r_score_layout_divided = cute.logical_divide(r_score.layout, (None, None, 2))
        r_score_mma_view = cute.make_layout(
            (
                (r_score_layout_divided.shape[0], r_score_layout_divided.shape[2][0]),
                r_score_layout_divided.shape[1],
                r_score_layout_divided.shape[2][1],
            ),
            stride=(
                (r_score_layout_divided.stride[0], r_score_layout_divided.stride[2][0]),
                r_score_layout_divided.stride[1],
                r_score_layout_divided.stride[2][1],
            ),
        )
        t_reg_score = cute.make_tensor(r_score.iterator, r_score_mma_view)

        cute.copy(
            smem_tiled_copy_value,
            t_smem_value_transposed[None, None, 0],
            t_reg_value_transposed_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(t_reg_score.shape[2])):
            k_next = (k + 1) % cute.size(t_reg_score.shape[2])
            cute.copy(
                smem_tiled_copy_value,
                t_smem_value_transposed[None, None, k_next],
                t_reg_value_transposed_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc_output,
                t_reg_score[None, None, k],
                t_reg_value_transposed[None, None, k],
                acc_output,
            )
        # The V tile aliases the main shared staging buffer in the fast path,
        # so do not let later stages overwrite it until every warp is done.
        cute.arch.barrier()

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

    # End-to-end specialization
    def _d_stage_size(self) -> int:
        """Width of each staged ``D`` slice used by the end-to-end kernel."""
        d_padded = self.D_padded
        if d_padded <= 64:
            return d_padded
        return 64

    def _d_stage_count(self) -> int:
        d_stage_width = self._d_stage_size()
        return (self.D_padded + d_stage_width - 1) // d_stage_width

    def _q_smem_elems(self) -> int:
        return int(self.m_block_size) * int(self._d_stage_size())

    def _b_smem_elems(self) -> int:
        return int(max(self.P_padded, self.n_block_size)) * int(self._d_stage_size())

    def _make_layout_bundle(
        self, out_dtype: type[cutlass.Numeric]
    ) -> ChunkScanLayoutBundle:
        d_stage_width = self._d_stage_size()
        p_padded = self.P_padded
        m_block_size = self.m_block_size
        n_block_size = self.n_block_size

        smem_k_block_size_d = 64 if d_stage_width % 64 == 0 else 32
        swizzle_bits_d = 3 if smem_k_block_size_d == 64 else 2
        d_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_d, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_d), stride=(smem_k_block_size_d, 1)),
        )
        query_layout = cute.tile_to_shape(
            d_layout_atom, (m_block_size, d_stage_width), (0, 1)
        )
        bz_alias_layout = cute.tile_to_shape(
            d_layout_atom, (max(p_padded, n_block_size), d_stage_width), (0, 1)
        )
        key_layout = cute.tile_to_shape(
            d_layout_atom, (n_block_size, d_stage_width), (0, 1)
        )
        z_layout = cute.tile_to_shape(d_layout_atom, (p_padded, d_stage_width), (0, 1))

        smem_k_block_size_p = 64 if p_padded % 64 == 0 else 32
        swizzle_bits_p = 3 if smem_k_block_size_p == 64 else 2
        p_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_p, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_p), stride=(smem_k_block_size_p, 1)),
        )
        value_layout = cute.tile_to_shape(
            p_layout_atom, (n_block_size, p_padded), (0, 1)
        )
        output_layout = cute.tile_to_shape(
            p_layout_atom, (m_block_size, p_padded), (0, 1)
        )
        if out_dtype == cutlass.Float32:
            output_layout = cute.make_layout(
                (m_block_size, p_padded), stride=(p_padded, 1)
            )

        return ChunkScanLayoutBundle(
            query_layout=query_layout,
            bz_alias_layout=bz_alias_layout,
            key_layout=key_layout,
            z_layout=z_layout,
            value_layout=value_layout,
            output_layout=output_layout,
        )

    def _make_copy_bundle(
        self,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
    ) -> ChunkScanCopyBundle:
        universal_copy_bits = 128
        d_stage_width = self._d_stage_size()
        p_padded = self.P_padded
        async_elems_in = universal_copy_bits // in_dtype.width

        smem_k_block_size_d = 64 if d_stage_width % 64 == 0 else 32
        t_d_shape_dim_1 = smem_k_block_size_d // async_elems_in
        t_d_layout = cute.make_layout(
            (self.num_threads // t_d_shape_dim_1, t_d_shape_dim_1),
            stride=(t_d_shape_dim_1, 1),
        )
        smem_k_block_size_p = 64 if p_padded % 64 == 0 else 32
        t_p_shape_dim_1 = smem_k_block_size_p // async_elems_in
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
        gmem_tiled_copy_d = cute.make_tiled_copy_tv(
            atom_async_copy_in, t_d_layout, v_in_layout
        )
        gmem_tiled_copy_p = cute.make_tiled_copy_tv(
            atom_async_copy_in, t_p_layout, v_in_layout
        )

        store_elems = universal_copy_bits // out_dtype.width
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        v_out_layout = cute.make_layout((1, store_elems))
        gmem_tiled_copy_output = cute.make_tiled_copy_tv(
            atom_universal_copy_out, t_p_layout, v_out_layout
        )
        return ChunkScanCopyBundle(
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_tiled_copy_p=gmem_tiled_copy_p,
            gmem_tiled_copy_output=gmem_tiled_copy_output,
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanLayoutBundle,
    ):
        if self._v_alias_in_bz():

            @cute.struct
            class SharedStorage:
                q_tile: cute.struct.Align[
                    cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)],
                    16,
                ]
                bz_alias: cute.struct.Align[
                    cute.struct.MemRange[
                        in_dtype, cute.cosize(layouts.bz_alias_layout)
                    ],
                    16,
                ]
                query_scale: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.m_block_size], 16
                ]
                key_scale: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                log_prefix: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                phase_re: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                phase_im: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                tap_phase_re: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                tap_phase_im: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                warp_log_total: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
                ]
                warp_log_offset: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
                ]
                warp_phase_total: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
                ]
                warp_phase_offset: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
                ]

        else:

            @cute.struct
            class SharedStorage:
                q_tile: cute.struct.Align[
                    cute.struct.MemRange[in_dtype, cute.cosize(layouts.query_layout)],
                    16,
                ]
                bz_alias: cute.struct.Align[
                    cute.struct.MemRange[
                        in_dtype, cute.cosize(layouts.bz_alias_layout)
                    ],
                    16,
                ]
                v_tile: cute.struct.Align[
                    cute.struct.MemRange[in_dtype, cute.cosize(layouts.value_layout)],
                    16,
                ]
                query_scale: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.m_block_size], 16
                ]
                key_scale: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                log_prefix: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                phase_re: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                phase_im: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.L], 16
                ]
                tap_phase_re: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                tap_phase_im: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
                ]
                warp_log_total: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
                ]
                warp_log_offset: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
                ]
                warp_phase_total: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
                ]
                warp_phase_offset: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
                ]

        return SharedStorage

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
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
            phase_re = mr * inv_mag
            phase_im = mi * inv_mag
            logp = cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.5 / LOG2_E)

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
            next_phase_re = phase_re * other_phase_re - phase_im * other_phase_im
            next_phase_im = phase_re * other_phase_im + phase_im * other_phase_re
            phase_re = cutlass.select_(pred, next_phase_re, phase_re)
            phase_im = cutlass.select_(pred, next_phase_im, phase_im)

        if lane == cutlass.Int32(31):
            prefix_state.warp_log_total[warp] = logp
            prefix_state.warp_phase_total[warp, 0] = phase_re
            prefix_state.warp_phase_total[warp, 1] = phase_im
        cute.arch.barrier()

        # Per the Hopper diagnosis for Issue #9, the inter-warp shuffle-based
        # exclusive scan in this block can trigger the H100 illegal-address fault
        # surface. A single-lane shared-memory scan preserves the same prefix
        # algebra while sidestepping that Hopper-specific failure mode.
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
                next_running_phase_re = (
                    running_phase_re * total_phase_re
                    - running_phase_im * total_phase_im
                )
                next_running_phase_im = (
                    running_phase_re * total_phase_im
                    + running_phase_im * total_phase_re
                )
                running_log = running_log + total_log
                running_phase_re = next_running_phase_re
                running_phase_im = next_running_phase_im

        cute.arch.barrier()

        warp_log_offset = prefix_state.warp_log_offset[warp]
        warp_phase_re_offset = prefix_state.warp_phase_offset[warp, 0]
        warp_phase_im_offset = prefix_state.warp_phase_offset[warp, 1]
        logp = logp + warp_log_offset
        next_phase_re = (
            phase_re * warp_phase_re_offset - phase_im * warp_phase_im_offset
        )
        next_phase_im = (
            phase_re * warp_phase_im_offset + phase_im * warp_phase_re_offset
        )
        phase_re, phase_im = next_phase_re, next_phase_im

        if tidx < self.L:
            prefix_state.s_log_prefix[seq_idx] = logp
            prefix_state.s_phase_re[seq_idx] = phase_re
            prefix_state.s_phase_im[seq_idx] = phase_im

        cute.arch.barrier()

    @cute.jit
    def _initialize_query_scales_from_prefix(
        self,
        prefix_state: SimpleNamespace,
        s_query_scale: cute.Tensor,
        *,
        m_tile_start: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        q_row = m_tile_start + tidx
        if tidx < self.m_block_size:
            scale = cutlass.Float32(0.0)
            if cute.elem_less(q_row, self.L):
                scale = cute.math.exp2(
                    cutlass.Float32(prefix_state.s_log_prefix[q_row])
                    * cutlass.Float32(LOG2_E),
                    fastmath=True,
                )
            s_query_scale[tidx] = scale

    @cute.jit
    def _initialize_key_scales_from_prefix(
        self,
        s_log_prefix: cute.Tensor,
        s_key_scale: cute.Tensor,
        *,
        n_block: int,
        seqlen: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        key_idx = n_block * self.n_block_size + tidx
        if tidx < self.n_block_size:
            if cute.elem_less(key_idx, seqlen):
                log_prefix = cutlass.Float32(s_log_prefix[key_idx])
                s_key_scale[tidx] = cute.math.exp2(
                    -log_prefix * cutlass.Float32(LOG2_E), fastmath=True
                )
            else:
                s_key_scale[tidx] = 0.0
        cute.arch.barrier()

    # Off-term helpers
    @cute.jit
    def _rotate_staged_query_tile_from_prefix(
        self,
        s_query: cute.Tensor,
        s_phase_re: cute.Tensor,
        s_phase_im: cute.Tensor,
        *,
        m_tile_start: int,
        row_limit: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        stage_complex = stage_width // 2
        total_q_stage = self.m_block_size * stage_complex
        idx = tidx
        while cute.elem_less(idx, total_q_stage):
            rr = idx // stage_complex
            vv = idx - rr * stage_complex
            q_row = m_tile_start + rr
            if cute.elem_less(q_row, row_limit):
                re_col = vv * 2 + 0
                im_col = vv * 2 + 1
                cre = cutlass.Float32(s_query[rr, re_col])
                cim = cutlass.Float32(s_query[rr, im_col])
                phase_re = cutlass.Float32(s_phase_re[q_row])
                phase_im = cutlass.Float32(s_phase_im[q_row])
                qre = cre * phase_re + cim * phase_im
                qim = cre * phase_im - cim * phase_re
                s_query[rr, re_col] = qre.to(out_dtype)
                s_query[rr, im_col] = qim.to(out_dtype)
            idx = idx + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _load_conjugated_z0_stage(
        self,
        mZ0: cute.Tensor,
        s_z: cute.Tensor,
        *,
        batch_head_chunk: int,
        d_col_base: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        vec = 4
        vec_cols = stage_width // vec
        total_vec = self.P_padded * vec_cols
        idx = tidx
        while cute.elem_less(idx, total_vec):
            rr = idx // vec_cols
            cc0 = (idx - rr * vec_cols) * vec
            g_col = d_col_base + cc0

            f0 = cutlass.Float32(0.0)
            f1 = cutlass.Float32(0.0)
            f2 = cutlass.Float32(0.0)
            f3 = cutlass.Float32(0.0)
            if cute.elem_less(rr, mZ0.layout.shape[1]) and cute.elem_less(
                g_col + (vec - 1), mZ0.layout.shape[3]
            ):
                row = mZ0[batch_head_chunk, rr, 0, None]
                row = cute.domain_offset((g_col,), row)
                seg = cute.make_tensor(
                    row.iterator.align(16), cute.make_layout((vec,), stride=(1,))
                )
                vals = seg.load()
                f0 = cutlass.Float32(vals[0])
                f1 = cutlass.Float32(vals[1])
                f2 = cutlass.Float32(vals[2])
                f3 = cutlass.Float32(vals[3])

            s_z[rr, cc0 + 0] = f0.to(out_dtype)
            s_z[rr, cc0 + 1] = (-f1).to(out_dtype)
            s_z[rr, cc0 + 2] = f2.to(out_dtype)
            s_z[rr, cc0 + 3] = (-f3).to(out_dtype)
            idx = idx + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _initialize_tap_phase_from_prefix(
        self,
        mK: cute.Tensor,
        s_phase_re: cute.Tensor,
        s_phase_im: cute.Tensor,
        s_tap_phase_re: cute.Tensor,
        s_tap_phase_im: cute.Tensor,
        *,
        batch_head_chunk: int,
        n_block: int,
        tap_idx: int,
        row_limit: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        if tidx < self.n_block_size:
            key_idx = n_block * self.n_block_size + tidx
            tap_phase_re = cutlass.Float32(0.0)
            tap_phase_im = cutlass.Float32(0.0)
            if cute.elem_less(key_idx, row_limit):
                tap_re = cutlass.Float32(mK[batch_head_chunk, key_idx, tap_idx, 0])
                tap_im = cutlass.Float32(mK[batch_head_chunk, key_idx, tap_idx, 1])
                phase_re = cutlass.Float32(s_phase_re[key_idx])
                phase_im = cutlass.Float32(s_phase_im[key_idx])
                tap_phase_re = tap_re * phase_re + tap_im * phase_im
                tap_phase_im = tap_im * phase_re - tap_re * phase_im
            s_tap_phase_re[tidx] = tap_phase_re
            s_tap_phase_im[tidx] = tap_phase_im
        cute.arch.barrier()

    @cute.jit
    def _inject_boundary_key_row(
        self,
        s_key: cute.Tensor,
        mB_prev0: cute.Tensor,
        *,
        batch_head: int,
        d_col_base: int,
        stage_width: int,
        out_dtype: type[cutlass.Numeric],
        is_chunk0: cutlass.Boolean,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        ii = tidx
        while cute.elem_less(ii, stage_width // 2):
            re_col = ii * 2 + 0
            im_col = ii * 2 + 1
            g_col_re = d_col_base + re_col
            g_col_im = d_col_base + im_col
            bre0 = s_key[0, re_col]
            bim0 = s_key[0, im_col]
            breb = out_dtype(0)
            bimb = out_dtype(0)
            if cute.elem_less(g_col_re, mB_prev0.layout.shape[1]):
                breb = mB_prev0[batch_head, g_col_re]
            if cute.elem_less(g_col_im, mB_prev0.layout.shape[1]):
                bimb = mB_prev0[batch_head, g_col_im]
            s_key[0, re_col] = cutlass.select_(is_chunk0, breb, bre0)
            s_key[0, im_col] = cutlass.select_(is_chunk0, bimb, bim0)
            ii = ii + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _apply_tap_phase_to_staged_keys(
        self,
        s_key: cute.Tensor,
        s_tap_phase_re: cute.Tensor,
        s_tap_phase_im: cute.Tensor,
        *,
        n_block: int,
        stage_width: int,
        row_limit: int,
        out_dtype: type[cutlass.Numeric],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        total_k_stage = self.n_block_size * (stage_width // 2)
        ii = tidx
        while cute.elem_less(ii, total_k_stage):
            rr = ii // (stage_width // 2)
            vv = ii - rr * (stage_width // 2)
            key_idx = n_block * self.n_block_size + rr
            re_col = vv * 2 + 0
            im_col = vv * 2 + 1
            bre = cutlass.Float32(s_key[rr, re_col])
            bim = cutlass.Float32(s_key[rr, im_col])
            tap_phase_re = cutlass.Float32(0.0)
            tap_phase_im = cutlass.Float32(0.0)
            if cute.elem_less(key_idx, row_limit):
                tap_phase_re = cutlass.Float32(s_tap_phase_re[rr])
                tap_phase_im = cutlass.Float32(s_tap_phase_im[rr])

            kre = bre * tap_phase_re - bim * tap_phase_im
            kim = bre * tap_phase_im + bim * tap_phase_re
            s_key[rr, re_col] = kre.to(out_dtype)
            s_key[rr, im_col] = (-kim).to(out_dtype)
            ii = ii + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _inject_boundary_value_row(
        self,
        s_value: cute.Tensor,
        mU_prev0: cute.Tensor,
        *,
        batch_head: int,
        out_dtype: type[cutlass.Numeric],
        is_chunk0: cutlass.Boolean,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        ii = tidx
        while cute.elem_less(ii, self.P_padded):
            boundary = out_dtype(0)
            old = s_value[0, ii]
            if cute.elem_less(ii, self.P):
                boundary = mU_prev0[batch_head, ii]
            s_value[0, ii] = cutlass.select_(is_chunk0, boundary, old)
            ii = ii + self.num_threads
        cute.arch.barrier()

    @cute.jit
    def _accumulate_offterm_from_rotated_z0(
        self,
        offterm_state: SimpleNamespace,
        prefix_state: SimpleNamespace,
        mma_state: SimpleNamespace,
    ):
        d_stage_width = self._d_stage_size()

        for d_stage_idx in cutlass.range_constexpr(self._d_stage_count()):
            d_col_base = d_stage_idx * d_stage_width

            g_query_stage = cute.local_tile(
                offterm_state.m_query[offterm_state.batch_head_chunk, None, 0, None],
                (self.m_block_size, d_stage_width),
                (offterm_state.m_block, d_stage_idx),
            )
            c_query_stage = cute.local_tile(
                offterm_state.coord_query[
                    offterm_state.batch_head_chunk, None, 0, None
                ],
                (self.m_block_size, d_stage_width),
                (offterm_state.m_block, d_stage_idx),
            )
            t_query_gmem = offterm_state.gmem_thr_copy_d.partition_S(g_query_stage)
            t_query_coord = offterm_state.gmem_thr_copy_d.partition_S(c_query_stage)
            t_query_pred = self._make_copy_column_predicate(
                offterm_state.t_query_smem,
                t_query_coord,
                offterm_state.m_query.layout.shape[3],
            )
            self._copy_rows_with_zero_fill(
                offterm_state.gmem_tiled_copy_d,
                t_query_gmem,
                offterm_state.t_query_smem,
                t_query_coord,
                t_query_pred,
                offterm_state.m_query.layout.shape[1],
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            self._rotate_staged_query_tile_from_prefix(
                offterm_state.s_query,
                prefix_state.s_phase_re,
                prefix_state.s_phase_im,
                m_tile_start=offterm_state.m_tile_start,
                row_limit=self.L,
                stage_width=d_stage_width,
                out_dtype=offterm_state.m_query.element_type,
            )

            self._load_conjugated_z0_stage(
                offterm_state.m_z0,
                offterm_state.s_z,
                batch_head_chunk=offterm_state.batch_head_chunk,
                d_col_base=d_col_base,
                stage_width=d_stage_width,
                out_dtype=offterm_state.m_query.element_type,
            )

            self._accumulate_from_staged_tiles(
                mma_state.tiled_mma,
                mma_state.acc_output,
                mma_state.smem_tiled_copy_query,
                mma_state.smem_tiled_copy_key,
                mma_state.t_smem_query,
                mma_state.t_reg_query_view,
                mma_state.t_smem_z,
                mma_state.t_reg_z_view,
                mma_state.t_reg_query,
                mma_state.t_reg_z,
            )

    # Diag helpers
    @cute.jit
    def _apply_score_scales_and_mask(
        self,
        acc_score: cute.Tensor,
        t_score_coord_mn: cute.Tensor,
        s_query_scale: cute.Tensor,
        s_key_scale: cute.Tensor,
        *,
        m_tile_start: int,
        n_tile_start: int,
        seqlen: int,
        apply_mask: cutlass.Constexpr,
    ):
        acc_score_mn = self._make_accumulator_mn_view(acc_score)
        scale_buf = cute.make_rmem_tensor(acc_score_mn[0, None].layout, cutlass.Float32)

        for r in cutlass.range_constexpr(cute.size(acc_score_mn.shape[0])):
            row_idx = cutlass.Int32(t_score_coord_mn[r, 0][1])
            row_scale = cutlass.Float32(1.0)
            if cute.elem_less(row_idx, seqlen):
                row_scale = cutlass.Float32(s_query_scale[row_idx - m_tile_start])

            col_limit = seqlen
            if cutlass.const_expr(apply_mask):
                col_limit = cutlass.min(row_idx + 1, seqlen)
            for c in cutlass.range_constexpr(cute.size(acc_score_mn.shape[1])):
                col_idx = cutlass.Int32(t_score_coord_mn[0, c][3])
                if cute.elem_less(col_limit, col_idx + 1) or cute.elem_less(
                    seqlen, col_idx + 1
                ):
                    scale_buf[c] = 0.0
                else:
                    key_scale = cutlass.Float32(s_key_scale[col_idx - n_tile_start])
                    scale_buf[c] = row_scale * key_scale

            acc_row = acc_score_mn[r, None].load()
            acc_score_mn[r, None] = acc_row * scale_buf.load()

    # Epilogue helpers
    def _make_output_coord_tile(
        self, mOut: cute.Tensor, batch_head_chunk: int, m_block: int
    ):
        coord_output = cute.make_identity_tensor(mOut.layout.shape)
        return cute.local_tile(
            coord_output[batch_head_chunk, None, 0, None],
            (self.m_block_size, self.P_padded),
            (m_block, 0),
        )

    @cute.jit
    def _scale_output_with_query_scales(
        self,
        acc_output_mn: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        s_query_scale: cute.Tensor,
        *,
        m_tile_start: int,
        row_limit: int,
    ):
        for r in cutlass.range_constexpr(cute.size(acc_output_mn.shape[0])):
            row_idx = t_output_coord_mn[r, 0][1]
            if cute.elem_less(row_idx, row_limit):
                scale = s_query_scale[row_idx - m_tile_start]
                acc_output_mn[r, None] = acc_output_mn[r, None].load() * scale

    @cute.jit
    def _store_output_fp32(
        self,
        mOut: cute.Tensor,
        t_output_coord_mn: cute.Tensor,
        acc_output_mn: cute.Tensor,
    ):
        for r in cutlass.range_constexpr(cute.size(acc_output_mn.shape[0])):
            row_idx = t_output_coord_mn[r, 0][1]
            if cute.elem_less(row_idx, mOut.layout.shape[1]):
                for c in cutlass.range_constexpr(cute.size(acc_output_mn.shape[1])):
                    col_idx = t_output_coord_mn[0, c][3]
                    if cute.elem_less(col_idx, mOut.layout.shape[3]):
                        mOut[t_output_coord_mn[r, c][0], row_idx, 0, col_idx] = (
                            cutlass.Float32(acc_output_mn[r, c])
                        )

    @cute.jit
    def _store_output_staged(
        self,
        mOut: cute.Tensor,
        g_output: cute.Tensor,
        c_output: cute.Tensor,
        s_query: cute.Tensor,
        output_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_output: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        acc_output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        r_output = cute.make_rmem_tensor_like(acc_output, mOut.element_type)
        r_output.store(acc_output.load().to(mOut.element_type))

        s_output = cute.make_tensor(
            cute.recast_ptr(s_query.iterator, dtype=mOut.element_type), output_layout
        )
        smem_copy_atom_output = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mOut.element_type
        )
        smem_tiled_copy_output = cute.make_tiled_copy_C(
            smem_copy_atom_output, tiled_mma
        )
        smem_thr_copy_output = smem_tiled_copy_output.get_slice(tidx)
        t_acc_reg_output = smem_thr_copy_output.retile(r_output)
        t_acc_smem_output = smem_thr_copy_output.partition_D(s_output)
        cute.copy(smem_copy_atom_output, t_acc_reg_output, t_acc_smem_output)
        cute.arch.barrier()

        gmem_thr_copy_output = gmem_tiled_copy_output.get_slice(tidx)
        t_smem_output = gmem_thr_copy_output.partition_S(s_output)
        t_gmem_output = gmem_thr_copy_output.partition_D(g_output)
        t_reg_output = cute.make_rmem_tensor_like(t_gmem_output, mOut.element_type)
        cute.copy(gmem_tiled_copy_output, t_smem_output, t_reg_output)

        t_output_coord = gmem_thr_copy_output.partition_D(c_output)
        t_output_pred = self._make_copy_column_predicate(
            t_gmem_output, t_output_coord, mOut.layout.shape[3]
        )
        self._copy_rows_if_valid(
            gmem_tiled_copy_output,
            t_reg_output,
            t_gmem_output,
            t_output_coord,
            t_output_pred,
            mOut.layout.shape[1],
        )

    @cute.jit
    def _store_output(
        self,
        mOut: cute.Tensor,
        g_output: cute.Tensor,
        c_output: cute.Tensor,
        s_query: cute.Tensor,
        output_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_output: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        acc_output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thr_mma = tiled_mma.get_slice(tidx)
        t_output_coord = thr_mma.partition_C(c_output)
        t_output_coord_mn = self._make_accumulator_mn_view(t_output_coord)
        acc_output_mn = self._make_accumulator_mn_view(acc_output)

        if cutlass.const_expr(mOut.element_type == cutlass.Float32):
            self._store_output_fp32(mOut, t_output_coord_mn, acc_output_mn)
        else:
            self._store_output_staged(
                mOut,
                g_output,
                c_output,
                s_query,
                output_layout,
                gmem_tiled_copy_output,
                tiled_mma,
                acc_output,
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
        mZ0: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mOut: cute.Tensor,
    ):
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("U/B/C must be Float16/BFloat16 for the tensor-core path.")
        if cutlass.const_expr(
            not (
                mU.element_type
                == mB.element_type
                == mC.element_type
                == mU_prev0.element_type
                == mB_prev0.element_type
            )
        ):
            raise TypeError("U/B/C/U_prev0/B_prev0 must share element type.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mK.element_type != cutlass.Float32):
            raise TypeError("K must be Float32.")
        if cutlass.const_expr(mZ0.element_type != cutlass.Float32):
            raise TypeError("Z0 must be Float32.")
        if cutlass.const_expr(
            mOut.element_type
            not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError("Out must be Float16/BFloat16/Float32.")

    def _launch_main_kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mZ0: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mOut: cute.Tensor,
        *,
        stream: cuda.CUstream | None = None,
    ):
        in_dtype = mU.element_type
        out_dtype = mOut.element_type
        bundle = self._make_kernel_bundle(in_dtype, out_dtype)
        layouts = bundle.layouts
        copies = bundle.copies
        grid_dim = (
            cute.ceil_div(mU.shape[1], self.m_block_size),
            cute.size(mU.shape[0]),
            1,
        )

        launch_kwargs = {
            "grid": grid_dim,
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
            mZ0,
            mU_prev0,
            mB_prev0,
            mOut,
            layouts.query_layout,
            layouts.bz_alias_layout,
            layouts.key_layout,
            layouts.z_layout,
            layouts.value_layout,
            layouts.output_layout,
            copies.gmem_tiled_copy_d,
            copies.gmem_tiled_copy_p,
            copies.gmem_tiled_copy_output,
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
        mZ0: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mOut: cute.Tensor,
    ):
        self._validate_main_operands(mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut)
        self._launch_main_kernel(mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut)

    @cute.jit
    def call_on_stream(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mK: cute.Tensor,
        mZ0: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._validate_main_operands(mU, mB, mC, mM, mK, mZ0, mU_prev0, mB_prev0, mOut)
        self._launch_main_kernel(
            mU,
            mB,
            mC,
            mM,
            mK,
            mZ0,
            mU_prev0,
            mB_prev0,
            mOut,
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
        mZ0: cute.Tensor,
        mU_prev0: cute.Tensor,
        mB_prev0: cute.Tensor,
        mOut: cute.Tensor,
        query_layout: cute.Layout | cute.ComposedLayout,
        bz_alias_layout: cute.Layout | cute.ComposedLayout,
        key_layout: cute.Layout | cute.ComposedLayout,
        z_layout: cute.Layout | cute.ComposedLayout,
        value_layout: cute.Layout | cute.ComposedLayout,
        output_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_d: cute.TiledCopy,
        gmem_tiled_copy_p: cute.TiledCopy,
        gmem_tiled_copy_output: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        shared_storage_cls: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_head_chunk, _ = cute.arch.block_idx()

        in_dtype = mU.element_type
        p_padded = self.P_padded
        m_block_size = self.m_block_size
        n_block_size = self.n_block_size
        d_stage_width = self._d_stage_size()
        d_stage_count = self._d_stage_count()
        n_block_max = self.n_block_max
        seq_len = self.L
        m_tile_start = m_block * m_block_size

        batch_head_count = mU_prev0.shape[0]
        n_chunks = mU.shape[0] // batch_head_count
        batch_head = batch_head_chunk // n_chunks
        chunk_index = batch_head_chunk - batch_head * n_chunks
        is_chunk0 = chunk_index == 0

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(shared_storage_cls)
        s_query = storage.q_tile.get_tensor(query_layout)
        s_bz_alias = storage.bz_alias.get_tensor(bz_alias_layout)
        if cutlass.const_expr(self._v_alias_in_bz()):
            s_value = cute.make_tensor(s_bz_alias.iterator, value_layout)
        else:
            s_value = storage.v_tile.get_tensor(value_layout)
        s_key = cute.make_tensor(s_bz_alias.iterator, key_layout)
        s_z = cute.make_tensor(s_bz_alias.iterator, z_layout)

        s_query_scale = storage.query_scale.get_tensor(
            cute.make_layout((m_block_size,), stride=(1,))
        )
        s_key_scale = storage.key_scale.get_tensor(
            cute.make_layout((n_block_size,), stride=(1,))
        )
        s_log_prefix = storage.log_prefix.get_tensor(
            cute.make_layout((seq_len,), stride=(1,))
        )
        s_phase_re = storage.phase_re.get_tensor(
            cute.make_layout((seq_len,), stride=(1,))
        )
        s_phase_im = storage.phase_im.get_tensor(
            cute.make_layout((seq_len,), stride=(1,))
        )
        s_tap_phase_re = storage.tap_phase_re.get_tensor(
            cute.make_layout((n_block_size,), stride=(1,))
        )
        s_tap_phase_im = storage.tap_phase_im.get_tensor(
            cute.make_layout((n_block_size,), stride=(1,))
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

        s_value_transposed = cute.composition(
            s_value,
            cute.make_layout((p_padded, n_block_size), stride=(n_block_size, 1)),
        )
        prefix_state = SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_transition=mM,
            s_log_prefix=s_log_prefix,
            s_phase_re=s_phase_re,
            s_phase_im=s_phase_im,
            warp_log_total=warp_log_total,
            warp_log_offset=warp_log_offset,
            warp_phase_total=warp_phase_total,
            warp_phase_offset=warp_phase_offset,
        )
        self._compute_phase_prefix_metadata(prefix_state)

        g_output = cute.local_tile(
            mOut[batch_head_chunk, None, 0, None],
            (m_block_size, p_padded),
            (m_block, 0),
        )

        gmem_thr_copy_d = gmem_tiled_copy_d.get_slice(tidx)
        gmem_thr_copy_p = gmem_tiled_copy_p.get_slice(tidx)
        t_query_smem = gmem_thr_copy_d.partition_D(s_query)
        t_key_smem = gmem_thr_copy_d.partition_D(s_key)
        t_value_smem = gmem_thr_copy_p.partition_D(s_value)

        coord_query = cute.make_identity_tensor(mC.layout.shape)
        coord_key = cute.make_identity_tensor(mB.layout.shape)
        coord_value = cute.make_identity_tensor(mU.layout.shape)
        coord_score = cute.make_identity_tensor(
            (mU.shape[0], mU.shape[1], mU.shape[2], mB.shape[1])
        )

        coord_value_tile0 = cute.local_tile(
            coord_value[batch_head_chunk, None, 0, None],
            (n_block_size, p_padded),
            (0, 0),
        )
        t_value_coord0 = gmem_thr_copy_p.partition_S(coord_value_tile0)
        t_value_pred = self._make_copy_column_predicate(
            t_value_smem, t_value_coord0, mU.layout.shape[3]
        )

        self._initialize_query_scales_from_prefix(
            prefix_state,
            s_query_scale,
            m_tile_start=m_tile_start,
        )

        thr_mma = tiled_mma.get_slice(tidx)
        t_reg_query = thr_mma.make_fragment_A(thr_mma.partition_A(s_query))
        t_reg_key = thr_mma.make_fragment_B(thr_mma.partition_B(s_key))
        t_reg_z = thr_mma.make_fragment_B(thr_mma.partition_B(s_z))
        t_reg_value_transposed = thr_mma.make_fragment_B(
            thr_mma.partition_B(s_value_transposed)
        )

        acc_shape_output = thr_mma.partition_shape_C((m_block_size, p_padded))
        acc_output = cute.make_rmem_tensor(acc_shape_output, cutlass.Float32)
        acc_output.fill(0.0)
        acc_shape_score = thr_mma.partition_shape_C((m_block_size, n_block_size))
        acc_score_template = cute.make_rmem_tensor(acc_shape_score, cutlass.Float32)

        smem_copy_atom_query = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_key = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_value = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_query = cute.make_tiled_copy_A(smem_copy_atom_query, tiled_mma)
        smem_tiled_copy_key = cute.make_tiled_copy_B(smem_copy_atom_key, tiled_mma)
        smem_tiled_copy_value = cute.make_tiled_copy_B(smem_copy_atom_value, tiled_mma)

        smem_thr_copy_query = smem_tiled_copy_query.get_slice(tidx)
        smem_thr_copy_key = smem_tiled_copy_key.get_slice(tidx)
        smem_thr_copy_value = smem_tiled_copy_value.get_slice(tidx)

        t_smem_query = smem_thr_copy_query.partition_S(s_query)
        t_reg_query_view = smem_thr_copy_query.retile(t_reg_query)
        t_smem_key = smem_thr_copy_key.partition_S(s_key)
        t_reg_key_view = smem_thr_copy_key.retile(t_reg_key)
        t_smem_z = smem_thr_copy_key.partition_S(s_z)
        t_reg_z_view = smem_thr_copy_key.retile(t_reg_z)
        t_smem_value_transposed = smem_thr_copy_value.partition_S(s_value_transposed)
        t_reg_value_transposed_view = smem_thr_copy_value.retile(t_reg_value_transposed)

        offterm_state = SimpleNamespace(
            batch_head_chunk=batch_head_chunk,
            m_block=m_block,
            m_tile_start=m_tile_start,
            gmem_tiled_copy_d=gmem_tiled_copy_d,
            gmem_thr_copy_d=gmem_thr_copy_d,
            m_query=mC,
            m_z0=mZ0,
            coord_query=coord_query,
            t_query_smem=t_query_smem,
            s_query=s_query,
            s_z=s_z,
        )
        mma_state = SimpleNamespace(
            tiled_mma=tiled_mma,
            thr_mma=thr_mma,
            acc_output=acc_output,
            acc_score_template=acc_score_template,
            smem_tiled_copy_query=smem_tiled_copy_query,
            smem_tiled_copy_key=smem_tiled_copy_key,
            smem_tiled_copy_value=smem_tiled_copy_value,
            t_smem_query=t_smem_query,
            t_reg_query_view=t_reg_query_view,
            t_smem_key=t_smem_key,
            t_reg_key_view=t_reg_key_view,
            t_smem_z=t_smem_z,
            t_reg_z_view=t_reg_z_view,
            t_smem_value_transposed=t_smem_value_transposed,
            t_reg_value_transposed_view=t_reg_value_transposed_view,
            t_reg_query=t_reg_query,
            t_reg_key=t_reg_key,
            t_reg_z=t_reg_z,
            t_reg_value_transposed=t_reg_value_transposed,
        )
        self._accumulate_offterm_from_rotated_z0(
            offterm_state,
            prefix_state,
            mma_state,
        )

        c_output = self._make_output_coord_tile(mOut, batch_head_chunk, m_block)
        t_output_coord = thr_mma.partition_C(c_output)
        t_output_coord_mn = self._make_accumulator_mn_view(t_output_coord)
        acc_output_mn = self._make_accumulator_mn_view(acc_output)
        self._scale_output_with_query_scales(
            acc_output_mn,
            t_output_coord_mn,
            s_query_scale,
            m_tile_start=m_tile_start,
            row_limit=seq_len,
        )

        visible_n_block_max = cutlass.min(
            cute.ceil_div((m_block + 1) * m_block_size, n_block_size), n_block_max
        )
        full_visible_n_blocks = (m_block * m_block_size) // n_block_size

        for pass_id in cutlass.range_constexpr(2):
            use_prev = cutlass.const_expr(pass_id == 0)

            m_key_pass = mB
            m_value_pass = mU
            if cutlass.const_expr(use_prev):
                m_key_pass = cute.domain_offset((0, -1, 0, 0), mB)
                m_value_pass = cute.domain_offset((0, -1, 0, 0), mU)
            tap_idx = 0 if cutlass.const_expr(use_prev) else 1

            for n_block in range(visible_n_block_max):
                is_prev_boundary_block = cutlass.const_expr(use_prev) and n_block == 0
                acc_score = cute.make_rmem_tensor_like(
                    acc_score_template, cutlass.Float32
                )
                acc_score.fill(0.0)

                self._initialize_tap_phase_from_prefix(
                    mK,
                    s_phase_re,
                    s_phase_im,
                    s_tap_phase_re,
                    s_tap_phase_im,
                    batch_head_chunk=batch_head_chunk,
                    n_block=n_block,
                    tap_idx=tap_idx,
                    row_limit=seq_len,
                )

                for d_stage_idx in cutlass.range_constexpr(d_stage_count):
                    d_col_base = d_stage_idx * d_stage_width

                    g_query_stage = cute.local_tile(
                        mC[batch_head_chunk, None, 0, None],
                        (m_block_size, d_stage_width),
                        (m_block, d_stage_idx),
                    )
                    c_query_stage = cute.local_tile(
                        coord_query[batch_head_chunk, None, 0, None],
                        (m_block_size, d_stage_width),
                        (m_block, d_stage_idx),
                    )
                    t_query_gmem = gmem_thr_copy_d.partition_S(g_query_stage)
                    t_query_coord = gmem_thr_copy_d.partition_S(c_query_stage)
                    t_query_pred = self._make_copy_column_predicate(
                        t_query_smem, t_query_coord, mC.layout.shape[3]
                    )
                    self._copy_rows_with_zero_fill(
                        gmem_tiled_copy_d,
                        t_query_gmem,
                        t_query_smem,
                        t_query_coord,
                        t_query_pred,
                        mC.layout.shape[1],
                    )

                    g_key_stage = cute.local_tile(
                        m_key_pass[batch_head_chunk, None, 0, None],
                        (n_block_size, d_stage_width),
                        (n_block, d_stage_idx),
                    )
                    c_key_stage = cute.local_tile(
                        coord_key[batch_head_chunk, None, 0, None],
                        (n_block_size, d_stage_width),
                        (n_block, d_stage_idx),
                    )
                    t_key_gmem = gmem_thr_copy_d.partition_S(g_key_stage)
                    t_key_coord = gmem_thr_copy_d.partition_S(c_key_stage)
                    t_key_pred = self._make_copy_column_predicate(
                        t_key_smem, t_key_coord, mB.layout.shape[3]
                    )
                    for ni in cutlass.range_constexpr(cute.size(t_key_smem.shape[1])):
                        row = t_key_coord[0, ni, 0][1]
                        if cute.elem_less(row, mB.layout.shape[1]):
                            if (
                                is_prev_boundary_block
                                and is_chunk0
                                and cute.elem_less(row, 1)
                            ):
                                t_key_smem[None, ni, None].fill(0)
                            else:
                                cute.copy(
                                    gmem_tiled_copy_d,
                                    t_key_gmem[None, ni, None],
                                    t_key_smem[None, ni, None],
                                    pred=t_key_pred[None, ni, None],
                                )
                        else:
                            t_key_smem[None, ni, None].fill(0)

                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    self._rotate_staged_query_tile_from_prefix(
                        s_query,
                        s_phase_re,
                        s_phase_im,
                        m_tile_start=m_tile_start,
                        row_limit=seq_len,
                        stage_width=d_stage_width,
                        out_dtype=in_dtype,
                    )

                    if is_prev_boundary_block:
                        self._inject_boundary_key_row(
                            s_key,
                            mB_prev0,
                            batch_head=batch_head,
                            d_col_base=d_col_base,
                            stage_width=d_stage_width,
                            out_dtype=in_dtype,
                            is_chunk0=is_chunk0,
                        )

                    self._apply_tap_phase_to_staged_keys(
                        s_key,
                        s_tap_phase_re,
                        s_tap_phase_im,
                        n_block=n_block,
                        stage_width=d_stage_width,
                        row_limit=seq_len,
                        out_dtype=in_dtype,
                    )

                    self._accumulate_from_staged_tiles(
                        tiled_mma,
                        acc_score,
                        smem_tiled_copy_query,
                        smem_tiled_copy_key,
                        t_smem_query,
                        t_reg_query_view,
                        t_smem_key,
                        t_reg_key_view,
                        t_reg_query,
                        t_reg_key,
                    )

                self._initialize_key_scales_from_prefix(
                    s_log_prefix,
                    s_key_scale,
                    n_block=n_block,
                    seqlen=seq_len,
                )

                c_score = cute.local_tile(
                    coord_score[batch_head_chunk, None, 0, None],
                    (m_block_size, n_block_size),
                    (m_block, n_block),
                )
                t_score_coord = thr_mma.partition_C(c_score)
                t_score_coord_mn = self._make_accumulator_mn_view(t_score_coord)
                if cute.elem_less(n_block, full_visible_n_blocks):
                    self._apply_score_scales_and_mask(
                        acc_score,
                        t_score_coord_mn,
                        s_query_scale,
                        s_key_scale,
                        m_tile_start=cutlass.Int32(m_tile_start),
                        n_tile_start=n_block * n_block_size,
                        seqlen=cutlass.Int32(seq_len),
                        apply_mask=False,
                    )
                else:
                    self._apply_score_scales_and_mask(
                        acc_score,
                        t_score_coord_mn,
                        s_query_scale,
                        s_key_scale,
                        m_tile_start=cutlass.Int32(m_tile_start),
                        n_tile_start=n_block * n_block_size,
                        seqlen=cutlass.Int32(seq_len),
                        apply_mask=True,
                    )

                g_value = cute.local_tile(
                    m_value_pass[batch_head_chunk, None, 0, None],
                    (n_block_size, p_padded),
                    (n_block, 0),
                )
                t_value_gmem = gmem_thr_copy_p.partition_S(g_value)
                c_value = cute.local_tile(
                    coord_value[batch_head_chunk, None, 0, None],
                    (n_block_size, p_padded),
                    (n_block, 0),
                )
                t_value_coord = gmem_thr_copy_p.partition_S(c_value)
                for vi in cutlass.range_constexpr(cute.size(t_value_smem.shape[1])):
                    row = t_value_coord[0, vi, 0][1]
                    if cute.elem_less(row, mU.layout.shape[1]):
                        if (
                            is_prev_boundary_block
                            and is_chunk0
                            and cute.elem_less(row, 1)
                        ):
                            t_value_smem[None, vi, None].fill(0)
                        else:
                            cute.copy(
                                gmem_tiled_copy_p,
                                t_value_gmem[None, vi, None],
                                t_value_smem[None, vi, None],
                                pred=t_value_pred[None, vi, None],
                            )
                    else:
                        t_value_smem[None, vi, None].fill(0)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                if is_prev_boundary_block:
                    self._inject_boundary_value_row(
                        s_value,
                        mU_prev0,
                        batch_head=batch_head,
                        out_dtype=in_dtype,
                        is_chunk0=is_chunk0,
                    )

                self._accumulate_output_from_scores(
                    tiled_mma,
                    acc_score,
                    acc_output,
                    in_dtype,
                    smem_tiled_copy_value,
                    t_smem_value_transposed,
                    t_reg_value_transposed_view,
                    t_reg_value_transposed,
                )

        self._store_output(
            mOut,
            g_output,
            c_output,
            s_query,
            output_layout,
            gmem_tiled_copy_output,
            tiled_mma,
            acc_output,
        )
