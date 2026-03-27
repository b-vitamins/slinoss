"""CuTe forward kernels for the v2x2ssd chunk-scan stage.

This file mirrors the structural shape of
``v3x3ssd.cute.kernels.fwd.chunk_scan``:

- ``ChunkScanFwdInnerAmpere`` owns the FA2-like inner workhorse over already
  phase-rotated per-chunk operands.
- ``ChunkScanFwdAmpere`` owns the end-to-end stage kernel, reconstructing the
  per-step prefix metadata and complex tap application in-kernel before running
  the same off-term + two diag passes.

Unlike ``v3``, the ``v2`` scan uses complex scalars / complex taps instead of
quaternion transport and 3x3 tap matrices:

- ``M[t]`` is a complex scalar transition.
- ``Kprev[t]`` / ``Kcurr[t]`` are complex scalar taps.
- ``B[t]`` / ``C[t]`` / ``chunk_starts`` store complex pairs packed along the
  trailing ``D = 2N`` dimension.

The inner kernel expects already phase-rotated tensors:

- ``Q = conj(C) * phase_prefix``               packed complex, ``(BHC, L, 1, D)``
- ``Kprev = (Kprev tap * Bprev) * conj(phase_prefix)``
- ``Kcurr = (Kcurr tap * Bcurr) * conj(phase_prefix)``
- ``Vprev`` / ``Vcurr`` are real ``U`` streams, ``(BHC, L, 1, P)``
- ``logprefix`` is the fp32 cumulative log-magnitude prefix, ``(BHC, L)``
- ``Z0`` is the chunk-start state, packed complex, ``(BHC, P, 1, D)``
- ``Out`` is real ``(BHC, L, 1, P)``

The end-to-end kernel consumes raw stage inputs:

- ``U``      ``(BHC, L, 1, P)``  fp16/bf16
- ``B``      ``(BHC, L, 1, D)``  fp16/bf16
- ``C``      ``(BHC, L, 1, D)``  fp16/bf16
- ``M``      ``(BHC, L, 2)``     fp32 packed complex
- ``K``      ``(BHC, L, 2, 2)``  fp32 packed complex taps
- ``Z0``     ``(BHC, P, 1, D)``  fp32 packed complex states
- ``U_prev0`` ``(BH, P)``        fp16/bf16
- ``B_prev0`` ``(BH, D)``        fp16/bf16
"""

from dataclasses import dataclass
from typing import ClassVar, Sequence

import torch
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
    sQ_layout: object
    sB_layout: object
    sK_layout: object
    sZ_layout: object
    sV_layout: object
    sO_layout: object


@dataclass(frozen=True)
class ChunkScanCopyBundle:
    gmem_tiled_copy_D: object
    gmem_tiled_copy_P: object
    gmem_tiled_copy_O: object


@dataclass(frozen=True)
class ChunkScanKernelBundle:
    layouts: ChunkScanLayoutBundle
    copies: ChunkScanCopyBundle
    tiled_mma: object
    compute_smem_bytes: int
    output_smem_bytes: int

    @property
    def smem_size(self) -> int:
        return max(self.compute_smem_bytes, self.output_smem_bytes)


class ChunkScanFwdInnerAmpere:
    """Ampere tensor-core inner kernel for the ``v2`` chunk-scan stage."""

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
    def n_complex(self) -> int:
        return self.D // 2

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

    def _q_smem_elems(self) -> int:
        return int(self.m_block_size) * int(self.D_padded)

    def _b_smem_elems(self) -> int:
        return int(max(self.P_padded, self.n_block_size)) * int(self.D_padded)

    def _v_smem_elems(self) -> int:
        return int(self.n_block_size) * int(self.P_padded)

    def _v_alias_in_bz(self) -> bool:
        """Whether ``sV`` can alias ``bz_alias`` storage instead of separate smem."""
        return self._v_smem_elems() <= self._b_smem_elems()

    def _o_smem_elems(self) -> int:
        return int(self.m_block_size) * int(self.P_padded)

    @staticmethod
    def _align_up(offset: int, align: int) -> int:
        return ((offset + align - 1) // align) * align

    @classmethod
    def _struct_size_bytes(cls, fields: Sequence[tuple[int, int]]) -> int:
        offset = 0
        max_align = 1
        for size, align in fields:
            offset = cls._align_up(offset, align)
            offset += size
            max_align = max(max_align, align)
        return cls._align_up(offset, max_align)

    def _shared_storage_fields(
        self, in_dtype: type[cutlass.Numeric]
    ) -> list[tuple[int, int]]:
        in_bytes = in_dtype.width // 8
        fields = [
            (self._q_smem_elems() * in_bytes, 16),
            (self._b_smem_elems() * in_bytes, 16),
            (self.m_block_size * 4, 16),
            (self.n_block_size * 4, 16),
        ]
        if not self._v_alias_in_bz():
            fields.insert(2, (self._v_smem_elems() * in_bytes, 16))
        return fields

    def _compute_smem_bytes(self, in_dtype: type[cutlass.Numeric]) -> int:
        return self._struct_size_bytes(self._shared_storage_fields(in_dtype))

    def _output_smem_bytes(self, out_dtype: type[cutlass.Numeric]) -> int:
        out_bytes = out_dtype.width // 8
        return self._o_smem_elems() * out_bytes

    def _make_layout_bundle(
        self, out_dtype: type[cutlass.Numeric]
    ) -> ChunkScanLayoutBundle:
        Dp = self.D_padded
        Pp = self.P_padded
        m = self.m_block_size
        n = self.n_block_size

        smem_k_block_size_D = 64 if Dp % 64 == 0 else 32
        swizzle_bits_D = 3 if smem_k_block_size_D == 64 else 2
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sQ_layout = cute.tile_to_shape(sD_layout_atom, (m, Dp), (0, 1))
        sB_layout = cute.tile_to_shape(sD_layout_atom, (max(Pp, n), Dp), (0, 1))
        sK_layout = cute.tile_to_shape(sD_layout_atom, (n, Dp), (0, 1))
        sZ_layout = cute.tile_to_shape(sD_layout_atom, (Pp, Dp), (0, 1))

        smem_k_block_size_P = 64 if Pp % 64 == 0 else 32
        swizzle_bits_P = 3 if smem_k_block_size_P == 64 else 2
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sV_layout = cute.tile_to_shape(sP_layout_atom, (n, Pp), (0, 1))
        sO_layout = cute.tile_to_shape(sP_layout_atom, (m, Pp), (0, 1))
        if out_dtype == cutlass.Float32:
            sO_layout = cute.make_layout((m, Pp), stride=(Pp, 1))

        return ChunkScanLayoutBundle(
            sQ_layout=sQ_layout,
            sB_layout=sB_layout,
            sK_layout=sK_layout,
            sZ_layout=sZ_layout,
            sV_layout=sV_layout,
            sO_layout=sO_layout,
        )

    def _make_copy_bundle(
        self,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
    ) -> ChunkScanCopyBundle:
        universal_copy_bits = 128
        Dp = self.D_padded
        Pp = self.P_padded
        async_elems_in = universal_copy_bits // in_dtype.width

        smem_k_block_size_D = 64 if Dp % 64 == 0 else 32
        tD_shape_dim_1 = smem_k_block_size_D // async_elems_in
        tD_layout = cute.make_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        smem_k_block_size_P = 64 if Pp % 64 == 0 else 32
        tP_shape_dim_1 = smem_k_block_size_P // async_elems_in
        tP_layout = cute.make_layout(
            (self.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))

        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )
        gmem_tiled_copy_P = cute.make_tiled_copy_tv(
            atom_async_copy_in, tP_layout, v_in_layout
        )

        store_elems = universal_copy_bits // out_dtype.width
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        v_out_layout = cute.make_layout((1, store_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy_out, tP_layout, v_out_layout
        )
        return ChunkScanCopyBundle(
            gmem_tiled_copy_D=gmem_tiled_copy_D,
            gmem_tiled_copy_P=gmem_tiled_copy_P,
            gmem_tiled_copy_O=gmem_tiled_copy_O,
        )

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
        return ChunkScanKernelBundle(
            layouts=layouts,
            copies=copies,
            tiled_mma=self._make_tiled_mma(in_dtype),
            compute_smem_bytes=self._compute_smem_bytes(in_dtype),
            output_smem_bytes=self._output_smem_bytes(out_dtype),
        )

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanLayoutBundle,
    ):
        class SharedStorage:
            pass

        annotations = {
            "q_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sQ_layout)], 16
            ],
            "bz_alias": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sB_layout)], 16
            ],
        }
        if not self._v_alias_in_bz():
            annotations["v_tile"] = cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sV_layout)], 16
            ]
        annotations["lp_q"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.m_block_size], 16
        ]
        annotations["lp_k"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
        ]
        SharedStorage.__annotations__ = annotations
        return cute.struct(SharedStorage)

    @cute.jit
    def _accumulate_from_smem_mma(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        smem_tiled_copy_A: cute.TiledCopy,
        smem_tiled_copy_B: cute.TiledCopy,
        tSsA: cute.Tensor,
        tSrA_view: cute.Tensor,
        tSsB: cute.Tensor,
        tSrB_view: cute.Tensor,
        tSrA: cute.Tensor,
        tSrB: cute.Tensor,
    ):
        cute.copy(smem_tiled_copy_A, tSsA[None, None, 0], tSrA_view[None, None, 0])
        cute.copy(smem_tiled_copy_B, tSsB[None, None, 0], tSrB_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tSsA.shape[2])):
            k_next = (k + 1) % cute.size(tSsA.shape[2])
            cute.copy(
                smem_tiled_copy_A,
                tSsA[None, None, k_next],
                tSrA_view[None, None, k_next],
            )
            cute.copy(
                smem_tiled_copy_B,
                tSsB[None, None, k_next],
                tSrB_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc,
                tSrA[None, None, k],
                tSrB[None, None, k],
                acc,
            )

    @cute.jit
    def _make_mma_accumulator(self, layout_or_shape, dtype):
        return cute.make_rmem_tensor(layout_or_shape, dtype)

    @cute.jit
    def _make_mma_accumulator_like(self, src, dtype):
        return cute.make_rmem_tensor_like(src, dtype)

    @cute.jit
    def _make_copy_col_predicate(
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
    def _copy_gmem_rows_or_zero(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tSg: cute.Tensor,
        tSd: cute.Tensor,
        tSc: cute.Tensor,
        copy_pred: cute.Tensor,
        row_limit: int,
    ):
        for idx in cutlass.range_constexpr(cute.size(tSd.shape[1])):
            if cute.elem_less(tSc[0, idx, 0][1], row_limit):
                cute.copy(
                    gmem_tiled_copy,
                    tSg[None, idx, None],
                    tSd[None, idx, None],
                    pred=copy_pred[None, idx, None],
                )
            else:
                tSd[None, idx, None].fill(0)

    def _make_output_tile(self, mOut: cute.Tensor, bhc: int, m_block: int):
        mcO = cute.make_identity_tensor(mOut.layout.shape)
        return cute.local_tile(
            mcO[bhc, None, 0, None],
            (self.m_block_size, self.P_padded),
            (m_block, 0),
        )

    @cute.jit
    def _accumulate_scores_times_v(
        self,
        tiled_mma: cute.TiledMma,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        score_dtype: type[cutlass.Numeric],
        smem_tiled_copy_V: cute.TiledCopy,
        tOsVt: cute.Tensor,
        tOrVt_view: cute.Tensor,
        tOrVt: cute.Tensor,
    ):
        rP = self._make_mma_accumulator_like(acc_S, score_dtype)
        rP.store(acc_S.load().to(score_dtype))
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        cute.copy(smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_tiled_copy_V,
                tOsVt[None, None, k_next],
                tOrVt_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrS[None, None, k],
                tOrVt[None, None, k],
                acc_O,
            )

    @cute.jit
    def _store_output(
        self,
        mOut: cute.Tensor,
        gO: cute.Tensor,
        cO: cute.Tensor,
        sQ: cute.Tensor,
        sO_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        acc_O: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thr_mma = tiled_mma.get_slice(tidx)
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

        if cutlass.const_expr(mOut.element_type == cutlass.Float32):
            for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                row_idx = tOcO_mn[r, 0][1]
                if cute.elem_less(row_idx, mOut.layout.shape[1]):
                    for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                        col_idx = tOcO_mn[0, c][3]
                        if cute.elem_less(col_idx, mOut.layout.shape[3]):
                            mOut[tOcO_mn[r, c][0], row_idx, 0, col_idx] = (
                                cutlass.Float32(acc_O_mn[r, c])
                            )
        else:
            rO = cute.make_rmem_tensor_like(acc_O, mOut.element_type)
            rO.store(acc_O.load().to(mOut.element_type))

            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=mOut.element_type), sO_layout
            )
            smem_copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mOut.element_type
            )
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
            cute.arch.barrier()

            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_rmem_tensor_like(tOgO, mOut.element_type)
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            tOcOut = gmem_thr_copy_O.partition_D(cO)
            tOpOut = self._make_copy_col_predicate(tOgO, tOcOut, mOut.layout.shape[3])
            self._copy_gmem_rows_or_zero(
                gmem_tiled_copy_O,
                tOrO,
                tOgO,
                tOcOut,
                tOpOut,
                mOut.layout.shape[1],
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

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mVprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mVcurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mZ0: cute.Tensor,
        mOut: cute.Tensor,
    ):
        if cutlass.const_expr(
            not (
                mQ.element_type
                == mKprev.element_type
                == mVprev.element_type
                == mKcurr.element_type
                == mVcurr.element_type
                == mZ0.element_type
            )
        ):
            raise TypeError("Q/K/V/Z0 must share the same element type.")

        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Tensor-core path supports only Float16/BFloat16 inputs.")

        if cutlass.const_expr(mLogprefix.element_type != cutlass.Float32):
            raise TypeError("logprefix must be Float32.")

        if cutlass.const_expr(
            mOut.element_type
            not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError("Output dtype must be Float16/BFloat16/Float32.")

        if cutlass.const_expr(mQ.shape[2] != 1 or mKprev.shape[2] != 1):
            raise ValueError("Q/K must be shaped as (BHC, L, 1, D).")
        if cutlass.const_expr(mVprev.shape[2] != 1 or mVcurr.shape[2] != 1):
            raise ValueError("V must be shaped as (BHC, L, 1, P).")
        if cutlass.const_expr(mZ0.shape[2] != 1):
            raise ValueError("Z0 must be shaped as (BHC, P, 1, D).")
        if cutlass.const_expr(mOut.shape[2] != 1):
            raise ValueError("Out must be shaped as (BHC, L, 1, P).")

        in_dtype = mQ.element_type
        out_dtype = mOut.element_type
        bundle = self._make_kernel_bundle(in_dtype, out_dtype)
        layouts = bundle.layouts
        copies = bundle.copies
        SharedStorage = self._make_shared_storage(in_dtype, layouts)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self.m_block_size),
            cute.size(mQ.shape[0]),
            1,
        )

        self.kernel(
            mQ,
            mKprev,
            mVprev,
            mKcurr,
            mVcurr,
            mLogprefix,
            mZ0,
            mOut,
            layouts.sQ_layout,
            layouts.sB_layout,
            layouts.sK_layout,
            layouts.sZ_layout,
            layouts.sV_layout,
            layouts.sO_layout,
            copies.gmem_tiled_copy_D,
            copies.gmem_tiled_copy_P,
            copies.gmem_tiled_copy_O,
            bundle.tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=bundle.smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mVprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mVcurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mZ0: cute.Tensor,
        mOut: cute.Tensor,
        sQ_layout: cute.Layout | cute.ComposedLayout,
        sB_layout: cute.Layout | cute.ComposedLayout,
        sK_layout: cute.Layout | cute.ComposedLayout,
        sZ_layout: cute.Layout | cute.ComposedLayout,
        sV_layout: cute.Layout | cute.ComposedLayout,
        sO_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, bhc, _ = cute.arch.block_idx()

        Dp = self.D_padded
        Pp = self.P_padded
        m = self.m_block_size
        n = self.n_block_size
        n_block_max = self.n_block_max
        N = self.n_complex

        gQ = cute.local_tile(mQ[bhc, None, 0, None], (m, Dp), (m_block, 0))
        gKprev = cute.local_tile(mKprev[bhc, None, 0, None], (n, Dp), (None, 0))
        gKcurr = cute.local_tile(mKcurr[bhc, None, 0, None], (n, Dp), (None, 0))
        gVprev = cute.local_tile(mVprev[bhc, None, 0, None], (n, Pp), (None, 0))
        gVcurr = cute.local_tile(mVcurr[bhc, None, 0, None], (n, Pp), (None, 0))
        gZ0 = cute.local_tile(mZ0[bhc, None, 0, None], (Pp, Dp), (0, 0))
        gO = cute.local_tile(mOut[bhc, None, 0, None], (m, Pp), (m_block, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.q_tile.get_tensor(sQ_layout)
        sB = storage.bz_alias.get_tensor(sB_layout)
        if cutlass.const_expr(self._v_alias_in_bz()):
            sV = cute.make_tensor(sB.iterator, sV_layout)
        else:
            sV = storage.v_tile.get_tensor(sV_layout)
        sK = cute.make_tensor(sB.iterator, sK_layout)
        sZ = cute.make_tensor(sB.iterator, sZ_layout)
        sLpQ = storage.lp_q.get_tensor(cute.make_layout((m,), stride=(1,)))
        sLpK = storage.lp_k.get_tensor(cute.make_layout((n,), stride=(1,)))

        sVt = cute.composition(sV, cute.make_layout((Pp, n), stride=(n, 1)))

        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        tQgQ = gmem_thr_copy_D.partition_S(gQ)
        tQsQ = gmem_thr_copy_D.partition_D(sQ)

        tKgKprev = gmem_thr_copy_D.partition_S(gKprev)
        tKgKcurr = gmem_thr_copy_D.partition_S(gKcurr)
        tKsK = gmem_thr_copy_D.partition_D(sK)

        tZgZ = gmem_thr_copy_D.partition_S(gZ0)
        tZsZ = gmem_thr_copy_D.partition_D(sZ)

        gmem_thr_copy_P = gmem_tiled_copy_P.get_slice(tidx)
        tVgVprev = gmem_thr_copy_P.partition_S(gVprev)
        tVgVcurr = gmem_thr_copy_P.partition_S(gVcurr)
        tVsV = gmem_thr_copy_P.partition_D(sV)

        mcQ = cute.make_identity_tensor(mQ.layout.shape)
        cQ = cute.local_tile(mcQ[bhc, None, 0, None], (m, Dp), (m_block, 0))
        tQcQ = gmem_thr_copy_D.partition_S(cQ)
        tQpQ = self._make_copy_col_predicate(tQsQ, tQcQ, mQ.layout.shape[3])

        mcK = cute.make_identity_tensor(mKprev.layout.shape)
        cK0 = cute.local_tile(mcK[bhc, None, 0, None], (n, Dp), (0, 0))
        tKcK0 = gmem_thr_copy_D.partition_S(cK0)
        tKpK = self._make_copy_col_predicate(tKsK, tKcK0, mKprev.layout.shape[3])

        mcZ = cute.make_identity_tensor(mZ0.layout.shape)
        cZ = cute.local_tile(mcZ[bhc, None, 0, None], (Pp, Dp), (0, 0))
        tZcZ = gmem_thr_copy_D.partition_S(cZ)
        tZpZ = self._make_copy_col_predicate(tZsZ, tZcZ, mZ0.layout.shape[3])

        mcV = cute.make_identity_tensor(mVcurr.layout.shape)
        cV0 = cute.local_tile(mcV[bhc, None, 0, None], (n, Pp), (0, 0))
        tVcV0 = gmem_thr_copy_P.partition_S(cV0)
        tVpV = self._make_copy_col_predicate(tVsV, tVcV0, mVcurr.layout.shape[3])

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tSrZ = thr_mma.make_fragment_B(thr_mma.partition_B(sZ))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_shape_O = thr_mma.partition_shape_C((m, Pp))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)
        acc_shape_S = thr_mma.partition_shape_C((m, n))
        acc_template_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)

        smem_copy_atom_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mQ.element_type,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mQ.element_type,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mQ.element_type,
        )

        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_view = smem_thr_copy_K.retile(tSrK)
        tSsZ = smem_thr_copy_K.partition_S(sZ)
        tSrZ_view = smem_thr_copy_K.retile(tSrZ)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_view = smem_thr_copy_V.retile(tOrVt)

        q_row = m_block * m + tidx
        if tidx < m:
            scale = cutlass.Float32(0.0)
            if cute.elem_less(q_row, mLogprefix.shape[1]):
                lp = cutlass.Float32(mLogprefix[bhc, q_row])
                scale = cute.math.exp2(lp * cutlass.Float32(LOG2_E), fastmath=True)
            sLpQ[tidx] = scale

        self._copy_gmem_rows_or_zero(
            gmem_tiled_copy_D,
            tQgQ,
            tQsQ,
            tQcQ,
            tQpQ,
            mQ.layout.shape[1],
        )

        self._copy_gmem_rows_or_zero(
            gmem_tiled_copy_D,
            tZgZ,
            tZsZ,
            tZcZ,
            tZpZ,
            mZ0.layout.shape[1],
        )

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        idx = tidx
        total_z = self.P * N
        while cute.elem_less(idx, total_z):
            rr = idx // N
            vv = idx - rr * N
            imag_col = vv * 2 + 1
            sZ[rr, imag_col] = -cutlass.Float32(sZ[rr, imag_col]).to(mQ.element_type)
            idx = idx + self.num_threads
        cute.arch.barrier()

        self._accumulate_from_smem_mma(
            tiled_mma,
            acc_O,
            smem_tiled_copy_Q,
            smem_tiled_copy_K,
            tSsQ,
            tSrQ_view,
            tSsZ,
            tSrZ_view,
            tSrQ,
            tSrZ,
        )

        cO = self._make_output_tile(mOut, bhc, m_block)
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

        m_tile_start = m_block * m
        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
            row_idx = tOcO_mn[r, 0][1]
            if cute.elem_less(row_idx, mOut.layout.shape[1]):
                scale = sLpQ[row_idx - m_tile_start]
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

        mcS = cute.make_identity_tensor(
            (mQ.shape[0], mQ.shape[1], mQ.shape[2], mKprev.shape[1])
        )

        visible_n_block_max = cutlass.min(
            cute.ceil_div((m_block + 1) * m, n), n_block_max
        )
        full_visible_n_blocks = (m_block * m) // n
        for pass_id in cutlass.range_constexpr(2):
            use_prev = cutlass.const_expr(pass_id == 0)
            tKgK = tKgKprev if use_prev else tKgKcurr
            tVgV = tVgVprev if use_prev else tVgVcurr

            self._copy_gmem_rows_or_zero(
                gmem_tiled_copy_D,
                tKgK[None, None, None, 0],
                tKsK,
                tKcK0,
                tKpK,
                mKprev.layout.shape[1],
            )
            cute.arch.cp_async_commit_group()

            total_k = n * N
            for n_block in cutlass.range_constexpr(self.n_block_max):
                if cute.elem_less(n_block, visible_n_block_max):
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    idx = tidx
                    while cute.elem_less(idx, total_k):
                        rr = idx // N
                        vv = idx - rr * N
                        re_col = vv * 2 + 0
                        im_col = vv * 2 + 1
                        bre = cutlass.Float32(sK[rr, re_col])
                        bim = cutlass.Float32(sK[rr, im_col])
                        sK[rr, re_col] = bre.to(mQ.element_type)
                        sK[rr, im_col] = (-bim).to(mQ.element_type)
                        idx = idx + self.num_threads
                    cute.arch.barrier()

                    k_col = n_block * n + tidx
                    if tidx < n:
                        if cute.elem_less(k_col, mLogprefix.shape[1]):
                            lp_s = cutlass.Float32(mLogprefix[bhc, k_col])
                            sLpK[tidx] = cute.math.exp2(
                                -lp_s * cutlass.Float32(LOG2_E), fastmath=True
                            )
                        else:
                            sLpK[tidx] = 0.0

                    if cutlass.const_expr(not self._v_alias_in_bz()):
                        cV = cute.local_tile(
                            mcV[bhc, None, 0, None], (n, Pp), (n_block, 0)
                        )
                        tVcV = gmem_thr_copy_P.partition_S(cV)
                        self._copy_gmem_rows_or_zero(
                            gmem_tiled_copy_P,
                            tVgV[None, None, None, n_block],
                            tVsV,
                            tVcV,
                            tVpV,
                            mVcurr.layout.shape[1],
                        )
                        cute.arch.cp_async_commit_group()

                    acc_S = cute.make_rmem_tensor_like(acc_template_S, cutlass.Float32)
                    acc_S.fill(0.0)
                    cute.copy(
                        smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0]
                    )
                    cute.copy(
                        smem_tiled_copy_K, tSsK[None, None, 0], tSrK_view[None, None, 0]
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                        k_next = (k + 1) % cute.size(tSsQ.shape[2])
                        cute.copy(
                            smem_tiled_copy_Q,
                            tSsQ[None, None, k_next],
                            tSrQ_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_K,
                            tSsK[None, None, k_next],
                            tSrK_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tSrQ[None, None, k],
                            tSrK[None, None, k],
                            acc_S,
                        )

                    cS = cute.local_tile(
                        mcS[bhc, None, 0, None], (m, n), (m_block, n_block)
                    )
                    tScS = thr_mma.partition_C(cS)
                    tScS_mn = self._make_acc_tensor_mn_view(tScS)
                    if cute.elem_less(n_block, full_visible_n_blocks):
                        self._scale_and_mask_scores(
                            acc_S,
                            tScS_mn,
                            sLpQ,
                            sLpK,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_block * n,
                            seqlen=mQ.layout.shape[1],
                            apply_mask=False,
                        )
                    else:
                        self._scale_and_mask_scores(
                            acc_S,
                            tScS_mn,
                            sLpQ,
                            sLpK,
                            m_tile_start=m_tile_start,
                            n_tile_start=n_block * n,
                            seqlen=mQ.layout.shape[1],
                            apply_mask=True,
                        )

                    if cutlass.const_expr(self._v_alias_in_bz()):
                        cV = cute.local_tile(
                            mcV[bhc, None, 0, None], (n, Pp), (n_block, 0)
                        )
                        tVcV = gmem_thr_copy_P.partition_S(cV)
                        self._copy_gmem_rows_or_zero(
                            gmem_tiled_copy_P,
                            tVgV[None, None, None, n_block],
                            tVsV,
                            tVcV,
                            tVpV,
                            mVcurr.layout.shape[1],
                        )
                        cute.arch.cp_async_commit_group()

                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    self._accumulate_scores_times_v(
                        tiled_mma,
                        acc_S,
                        acc_O,
                        mQ.element_type,
                        smem_tiled_copy_V,
                        tOsVt,
                        tOrVt_view,
                        tOrVt,
                    )

                    if cute.elem_less(n_block + 1, visible_n_block_max):
                        cK_next = cute.local_tile(
                            mcK[bhc, None, 0, None], (n, Dp), (n_block + 1, 0)
                        )
                        tKcK_next = gmem_thr_copy_D.partition_S(cK_next)
                        self._copy_gmem_rows_or_zero(
                            gmem_tiled_copy_D,
                            tKgK[None, None, None, n_block + 1],
                            tKsK,
                            tKcK_next,
                            tKpK,
                            mKprev.layout.shape[1],
                        )
                        cute.arch.cp_async_commit_group()

        self._store_output(
            mOut,
            gO,
            cO,
            sQ,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma,
            acc_O,
        )

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
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

    @cute.jit
    def _scale_and_mask_scores(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        sLpQ: cute.Tensor,
        sLpK: cute.Tensor,
        *,
        m_tile_start: int,
        n_tile_start: int,
        seqlen: int,
        apply_mask: cutlass.Constexpr,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        scale_buf = cute.make_rmem_tensor(acc_S_mn[0, None].layout, cutlass.Float32)

        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            row_idx = cutlass.Int32(tScS_mn[r, 0][1])
            row_scale = cutlass.Float32(1.0)
            if cute.elem_less(row_idx, seqlen):
                row_scale = cutlass.Float32(sLpQ[row_idx - m_tile_start])

            col_limit = seqlen
            if cutlass.const_expr(apply_mask):
                col_limit = cutlass.min(row_idx + 1, seqlen)
            for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                col_idx = cutlass.Int32(tScS_mn[0, c][3])
                if cute.elem_less(col_limit, col_idx + 1) or cute.elem_less(
                    seqlen, col_idx + 1
                ):
                    scale_buf[c] = 0.0
                else:
                    key_scale = cutlass.Float32(sLpK[col_idx - n_tile_start])
                    scale_buf[c] = row_scale * key_scale

            acc_row = acc_S_mn[r, None].load()
            acc_S_mn[r, None] = acc_row * scale_buf.load()


class ChunkScanFwdAmpere(ChunkScanFwdInnerAmpere):
    """Ampere tensor-core end-to-end ``v2`` chunk-scan kernel."""

    def _d_stage_size(self) -> int:
        """Shared-memory D slice width used by the end-to-end kernel."""
        Dp = self.D_padded
        if Dp <= 96:
            return Dp
        return 96

    def _d_stage_count(self) -> int:
        Ds = self._d_stage_size()
        return (self.D_padded + Ds - 1) // Ds

    def _q_smem_elems(self) -> int:
        return int(self.m_block_size) * int(self._d_stage_size())

    def _b_smem_elems(self) -> int:
        return int(max(self.P_padded, self.n_block_size)) * int(self._d_stage_size())

    def _make_layout_bundle(
        self, out_dtype: type[cutlass.Numeric]
    ) -> ChunkScanLayoutBundle:
        Ds = self._d_stage_size()
        Pp = self.P_padded
        m = self.m_block_size
        n = self.n_block_size

        smem_k_block_size_D = 64 if Ds % 64 == 0 else 32
        swizzle_bits_D = 3 if smem_k_block_size_D == 64 else 2
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sQ_layout = cute.tile_to_shape(sD_layout_atom, (m, Ds), (0, 1))
        sB_layout = cute.tile_to_shape(sD_layout_atom, (max(Pp, n), Ds), (0, 1))
        sK_layout = cute.tile_to_shape(sD_layout_atom, (n, Ds), (0, 1))
        sZ_layout = cute.tile_to_shape(sD_layout_atom, (Pp, Ds), (0, 1))

        smem_k_block_size_P = 64 if Pp % 64 == 0 else 32
        swizzle_bits_P = 3 if smem_k_block_size_P == 64 else 2
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sV_layout = cute.tile_to_shape(sP_layout_atom, (n, Pp), (0, 1))
        sO_layout = cute.tile_to_shape(sP_layout_atom, (m, Pp), (0, 1))
        if out_dtype == cutlass.Float32:
            sO_layout = cute.make_layout((m, Pp), stride=(Pp, 1))

        return ChunkScanLayoutBundle(
            sQ_layout=sQ_layout,
            sB_layout=sB_layout,
            sK_layout=sK_layout,
            sZ_layout=sZ_layout,
            sV_layout=sV_layout,
            sO_layout=sO_layout,
        )

    def _make_copy_bundle(
        self,
        in_dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
    ) -> ChunkScanCopyBundle:
        universal_copy_bits = 128
        Ds = self._d_stage_size()
        Pp = self.P_padded
        async_elems_in = universal_copy_bits // in_dtype.width

        smem_k_block_size_D = 64 if Ds % 64 == 0 else 32
        tD_shape_dim_1 = smem_k_block_size_D // async_elems_in
        tD_layout = cute.make_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        smem_k_block_size_P = 64 if Pp % 64 == 0 else 32
        tP_shape_dim_1 = smem_k_block_size_P // async_elems_in
        tP_layout = cute.make_layout(
            (self.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))

        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )
        gmem_tiled_copy_P = cute.make_tiled_copy_tv(
            atom_async_copy_in, tP_layout, v_in_layout
        )

        store_elems = universal_copy_bits // out_dtype.width
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        v_out_layout = cute.make_layout((1, store_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy_out, tP_layout, v_out_layout
        )
        return ChunkScanCopyBundle(
            gmem_tiled_copy_D=gmem_tiled_copy_D,
            gmem_tiled_copy_P=gmem_tiled_copy_P,
            gmem_tiled_copy_O=gmem_tiled_copy_O,
        )

    def can_implement(
        self,
        dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        *,
        device_index: int | None = None,
    ) -> bool:
        return self.support_info(dtype, out_dtype, device_index=device_index).supported

    def _shared_storage_fields(
        self, in_dtype: type[cutlass.Numeric]
    ) -> list[tuple[int, int]]:
        fields = super()._shared_storage_fields(in_dtype)
        fields.extend(
            [
                (self.L * 4, 16),
                (self.L * 4, 16),
                (self.L * 4, 16),
                (self.num_warps * 4, 16),
                (self.num_warps * 4, 16),
                (self.num_warps * 2 * 4, 16),
                (self.num_warps * 2 * 4, 16),
            ]
        )
        return fields

    def _make_shared_storage(
        self,
        in_dtype: type[cutlass.Numeric],
        layouts: ChunkScanLayoutBundle,
    ):
        class SharedStorage:
            pass

        annotations = {
            "q_tile": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sQ_layout)], 16
            ],
            "bz_alias": cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sB_layout)], 16
            ],
        }
        if not self._v_alias_in_bz():
            annotations["v_tile"] = cute.struct.Align[
                cute.struct.MemRange[in_dtype, cute.cosize(layouts.sV_layout)], 16
            ]
        annotations["lp_q"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.m_block_size], 16
        ]
        annotations["lp_k"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.n_block_size], 16
        ]
        annotations["logpref"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.L], 16
        ]
        annotations["phase_re"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.L], 16
        ]
        annotations["phase_im"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.L], 16
        ]
        annotations["warp_log_total"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
        ]
        annotations["warp_log_offset"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.num_warps], 16
        ]
        annotations["warp_phase_total"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
        ]
        annotations["warp_phase_offset"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, self.num_warps * 2], 16
        ]
        SharedStorage.__annotations__ = annotations
        return cute.struct(SharedStorage)

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

        in_dtype = mU.element_type
        out_dtype = mOut.element_type
        bundle = self._make_kernel_bundle(in_dtype, out_dtype)
        layouts = bundle.layouts
        copies = bundle.copies
        SharedStorage = self._make_shared_storage(in_dtype, layouts)
        grid_dim = (
            cute.ceil_div(mU.shape[1], self.m_block_size),
            cute.size(mU.shape[0]),
            1,
        )
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
            layouts.sQ_layout,
            layouts.sB_layout,
            layouts.sK_layout,
            layouts.sZ_layout,
            layouts.sV_layout,
            layouts.sO_layout,
            copies.gmem_tiled_copy_D,
            copies.gmem_tiled_copy_P,
            copies.gmem_tiled_copy_O,
            bundle.tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=bundle.smem_size,
        )

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
        sQ_layout: cute.Layout | cute.ComposedLayout,
        sB_layout: cute.Layout | cute.ComposedLayout,
        sK_layout: cute.Layout | cute.ComposedLayout,
        sZ_layout: cute.Layout | cute.ComposedLayout,
        sV_layout: cute.Layout | cute.ComposedLayout,
        sO_layout: cute.Layout | cute.ComposedLayout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, bhc, _ = cute.arch.block_idx()

        Pp = self.P_padded
        m = self.m_block_size
        n = self.n_block_size
        n_block_max = self.n_block_max
        L = self.L

        BH = mU_prev0.shape[0]
        n_chunks = mU.shape[0] // BH
        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        is_chunk0 = chunk == 0

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.q_tile.get_tensor(sQ_layout)
        sB = storage.bz_alias.get_tensor(sB_layout)
        if cutlass.const_expr(self._v_alias_in_bz()):
            sV = cute.make_tensor(sB.iterator, sV_layout)
        else:
            sV = storage.v_tile.get_tensor(sV_layout)
        sK = cute.make_tensor(sB.iterator, sK_layout)
        sZ = cute.make_tensor(sB.iterator, sZ_layout)

        sLpQ = storage.lp_q.get_tensor(cute.make_layout((m,), stride=(1,)))
        sLpK = storage.lp_k.get_tensor(cute.make_layout((n,), stride=(1,)))
        sLogpref = storage.logpref.get_tensor(cute.make_layout((L,), stride=(1,)))
        sPhaseRe = storage.phase_re.get_tensor(cute.make_layout((L,), stride=(1,)))
        sPhaseIm = storage.phase_im.get_tensor(cute.make_layout((L,), stride=(1,)))
        warpLogTotal = storage.warp_log_total.get_tensor(
            cute.make_layout((self.num_warps,), stride=(1,))
        )
        warpLogOffset = storage.warp_log_offset.get_tensor(
            cute.make_layout((self.num_warps,), stride=(1,))
        )
        warpPhaseTotal = storage.warp_phase_total.get_tensor(
            cute.make_layout((self.num_warps, 2), stride=(2, 1))
        )
        warpPhaseOffset = storage.warp_phase_offset.get_tensor(
            cute.make_layout((self.num_warps, 2), stride=(2, 1))
        )

        sVt = cute.composition(sV, cute.make_layout((Pp, n), stride=(n, 1)))
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        tt = tidx
        logp = cutlass.Float32(0.0)
        pr = cutlass.Float32(1.0)
        pi = cutlass.Float32(0.0)

        if tidx < L:
            mr = cutlass.Float32(mM[bhc, tt, 0])
            mi = cutlass.Float32(mM[bhc, tt, 1])
            mag2 = mr * mr + mi * mi + cutlass.Float32(1.0e-20)
            inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
            pr = mr * inv_mag
            pi = mi * inv_mag
            logp = cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.5 / LOG2_E)

        for offset in (1, 2, 4, 8, 16):
            other_log = cute.arch.shuffle_sync_up(
                logp, offset=offset, mask=-1, mask_and_clamp=0
            )
            other_pr = cute.arch.shuffle_sync_up(
                pr, offset=offset, mask=-1, mask_and_clamp=0
            )
            other_pi = cute.arch.shuffle_sync_up(
                pi, offset=offset, mask=-1, mask_and_clamp=0
            )

            pred = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(pred, logp + other_log, logp)
            next_pr = pr * other_pr - pi * other_pi
            next_pi = pr * other_pi + pi * other_pr
            pr = cutlass.select_(pred, next_pr, pr)
            pi = cutlass.select_(pred, next_pi, pi)

        if lane == cutlass.Int32(31):
            warpLogTotal[warp] = logp
            warpPhaseTotal[warp, 0] = pr
            warpPhaseTotal[warp, 1] = pi
        cute.arch.barrier()

        if warp == cutlass.Int32(0):
            w = lane
            num_warps = cutlass.Int32(self.num_threads // 32)
            has_warp = w < num_warps

            wlog = cutlass.select_(has_warp, warpLogTotal[w], cutlass.Float32(0.0))
            wpr = cutlass.select_(has_warp, warpPhaseTotal[w, 0], cutlass.Float32(1.0))
            wpi = cutlass.select_(has_warp, warpPhaseTotal[w, 1], cutlass.Float32(0.0))

            for offset in (1, 2, 4, 8, 16):
                other_log = cute.arch.shuffle_sync_up(
                    wlog, offset=offset, mask=-1, mask_and_clamp=0
                )
                other_pr = cute.arch.shuffle_sync_up(
                    wpr, offset=offset, mask=-1, mask_and_clamp=0
                )
                other_pi = cute.arch.shuffle_sync_up(
                    wpi, offset=offset, mask=-1, mask_and_clamp=0
                )
                pred = lane >= cutlass.Int32(offset)
                wlog = cutlass.select_(pred, wlog + other_log, wlog)
                next_wpr = wpr * other_pr - wpi * other_pi
                next_wpi = wpr * other_pi + wpi * other_pr
                wpr = cutlass.select_(pred, next_wpr, wpr)
                wpi = cutlass.select_(pred, next_wpi, wpi)

            off_log = cute.arch.shuffle_sync_up(
                wlog, offset=1, mask=-1, mask_and_clamp=0
            )
            off_pr = cute.arch.shuffle_sync_up(wpr, offset=1, mask=-1, mask_and_clamp=0)
            off_pi = cute.arch.shuffle_sync_up(wpi, offset=1, mask=-1, mask_and_clamp=0)

            is_first_warp = lane == cutlass.Int32(0)
            off_log = cutlass.select_(is_first_warp, cutlass.Float32(0.0), off_log)
            off_pr = cutlass.select_(is_first_warp, cutlass.Float32(1.0), off_pr)
            off_pi = cutlass.select_(is_first_warp, cutlass.Float32(0.0), off_pi)

            if has_warp:
                warpLogOffset[w] = off_log
                warpPhaseOffset[w, 0] = off_pr
                warpPhaseOffset[w, 1] = off_pi

        cute.arch.barrier()

        off_log = warpLogOffset[warp]
        off_pr = warpPhaseOffset[warp, 0]
        off_pi = warpPhaseOffset[warp, 1]
        logp = logp + off_log
        next_pr = pr * off_pr - pi * off_pi
        next_pi = pr * off_pi + pi * off_pr
        pr, pi = next_pr, next_pi

        if tidx < L:
            sLogpref[tt] = logp
            sPhaseRe[tt] = pr
            sPhaseIm[tt] = pi

        cute.arch.barrier()

        Ds = self._d_stage_size()
        Ns = Ds // 2
        d_stage_count = self._d_stage_count()
        vec = 4

        gO = cute.local_tile(mOut[bhc, None, 0, None], (m, Pp), (m_block, 0))

        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        gmem_thr_copy_P = gmem_tiled_copy_P.get_slice(tidx)
        tQsQ = gmem_thr_copy_D.partition_D(sQ)
        tKsK = gmem_thr_copy_D.partition_D(sK)
        tVsV = gmem_thr_copy_P.partition_D(sV)

        mcQ = cute.make_identity_tensor(mC.layout.shape)
        mcK = cute.make_identity_tensor(mB.layout.shape)
        mcV = cute.make_identity_tensor(mU.layout.shape)
        mcS = cute.make_identity_tensor(
            (mU.shape[0], mU.shape[1], mU.shape[2], mB.shape[1])
        )

        cV0 = cute.local_tile(mcV[bhc, None, 0, None], (n, Pp), (0, 0))
        tVcV0 = gmem_thr_copy_P.partition_S(cV0)
        tVpV = self._make_copy_col_predicate(tVsV, tVcV0, mU.layout.shape[3])

        q_row = m_block * m + tidx
        if tidx < m:
            scale = cutlass.Float32(0.0)
            if cute.elem_less(q_row, L):
                scale = cute.math.exp2(
                    cutlass.Float32(sLogpref[q_row]) * cutlass.Float32(LOG2_E),
                    fastmath=True,
                )
            sLpQ[tidx] = scale

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tSrZ = thr_mma.make_fragment_B(thr_mma.partition_B(sZ))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_shape_O = thr_mma.partition_shape_C((m, Pp))
        acc_O = self._make_mma_accumulator(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)
        acc_shape_S = thr_mma.partition_shape_C((m, n))
        acc_template_S = self._make_mma_accumulator(acc_shape_S, cutlass.Float32)

        smem_copy_atom_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mU.element_type,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            mU.element_type,
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_view = smem_thr_copy_K.retile(tSrK)
        tSsZ = smem_thr_copy_K.partition_S(sZ)
        tSrZ_view = smem_thr_copy_K.retile(tSrZ)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_view = smem_thr_copy_V.retile(tOrVt)

        # Off-term accumulation (Q x conj(Z0)) across staged D slices.
        for d_stage_idx in cutlass.range_constexpr(d_stage_count):
            d_col_base = d_stage_idx * Ds

            gQ_stage = cute.local_tile(
                mC[bhc, None, 0, None], (m, Ds), (m_block, d_stage_idx)
            )
            cQ_stage = cute.local_tile(
                mcQ[bhc, None, 0, None], (m, Ds), (m_block, d_stage_idx)
            )
            tQgQ = gmem_thr_copy_D.partition_S(gQ_stage)
            tQcQ = gmem_thr_copy_D.partition_S(cQ_stage)
            tQpQ = self._make_copy_col_predicate(tQsQ, tQcQ, mC.layout.shape[3])
            self._copy_gmem_rows_or_zero(
                gmem_tiled_copy_D,
                tQgQ,
                tQsQ,
                tQcQ,
                tQpQ,
                mC.layout.shape[1],
            )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            total_q_stage = m * Ns
            idx = tidx
            while cute.elem_less(idx, total_q_stage):
                rr = idx // Ns
                vv = idx - rr * Ns
                t_idx = m_block * m + rr
                if cute.elem_less(t_idx, L):
                    re_col = vv * 2 + 0
                    im_col = vv * 2 + 1
                    cre = cutlass.Float32(sQ[rr, re_col])
                    cim = cutlass.Float32(sQ[rr, im_col])
                    phase_re = cutlass.Float32(sPhaseRe[t_idx])
                    phase_im = cutlass.Float32(sPhaseIm[t_idx])
                    qre = cre * phase_re + cim * phase_im
                    qim = cre * phase_im - cim * phase_re
                    sQ[rr, re_col] = qre.to(mU.element_type)
                    sQ[rr, im_col] = qim.to(mU.element_type)
                idx = idx + self.num_threads
            cute.arch.barrier()

            vec_cols = Ds // vec
            total_vec = Pp * vec_cols
            v = tidx
            while cute.elem_less(v, total_vec):
                rr = v // vec_cols
                cc0 = (v - rr * vec_cols) * vec
                g_col = d_col_base + cc0

                f0 = cutlass.Float32(0.0)
                f1 = cutlass.Float32(0.0)
                f2 = cutlass.Float32(0.0)
                f3 = cutlass.Float32(0.0)
                if cute.elem_less(rr, mZ0.layout.shape[1]) and cute.elem_less(
                    g_col + (vec - 1), mZ0.layout.shape[3]
                ):
                    row = mZ0[bhc, rr, 0, None]
                    row = cute.domain_offset((g_col,), row)
                    seg = cute.make_tensor(
                        row.iterator.align(16), cute.make_layout((vec,), stride=(1,))
                    )
                    r = seg.load()
                    f0 = cutlass.Float32(r[0])
                    f1 = cutlass.Float32(r[1])
                    f2 = cutlass.Float32(r[2])
                    f3 = cutlass.Float32(r[3])

                sZ[rr, cc0 + 0] = f0.to(mU.element_type)
                sZ[rr, cc0 + 1] = (-f1).to(mU.element_type)
                sZ[rr, cc0 + 2] = f2.to(mU.element_type)
                sZ[rr, cc0 + 3] = (-f3).to(mU.element_type)
                v = v + self.num_threads
            cute.arch.barrier()

            self._accumulate_from_smem_mma(
                tiled_mma,
                acc_O,
                smem_tiled_copy_Q,
                smem_tiled_copy_K,
                tSsQ,
                tSrQ_view,
                tSsZ,
                tSrZ_view,
                tSrQ,
                tSrZ,
            )

        cO = self._make_output_tile(mOut, bhc, m_block)
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        m_tile_start = m_block * m
        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
            row_idx = tOcO_mn[r, 0][1]
            if cute.elem_less(row_idx, L):
                scale = sLpQ[row_idx - m_tile_start]
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

        visible_n_block_max = cutlass.min(
            cute.ceil_div((m_block + 1) * m, n), n_block_max
        )
        full_visible_n_blocks = (m_block * m) // n

        for pass_id in cutlass.range_constexpr(2):
            use_prev = cutlass.const_expr(pass_id == 0)

            mB_source = mB
            mU_source = mU
            if cutlass.const_expr(use_prev):
                mB_source = cute.domain_offset((0, -1, 0, 0), mB)
                mU_source = cute.domain_offset((0, -1, 0, 0), mU)

            for n_block in cutlass.range_constexpr(self.n_block_max):
                if cute.elem_less(n_block, visible_n_block_max):
                    acc_S = self._make_mma_accumulator_like(
                        acc_template_S, cutlass.Float32
                    )
                    acc_S.fill(0.0)

                    for d_stage_idx in cutlass.range_constexpr(d_stage_count):
                        d_col_base = d_stage_idx * Ds

                        gQ_stage = cute.local_tile(
                            mC[bhc, None, 0, None], (m, Ds), (m_block, d_stage_idx)
                        )
                        cQ_stage = cute.local_tile(
                            mcQ[bhc, None, 0, None], (m, Ds), (m_block, d_stage_idx)
                        )
                        tQgQ = gmem_thr_copy_D.partition_S(gQ_stage)
                        tQcQ = gmem_thr_copy_D.partition_S(cQ_stage)
                        tQpQ = self._make_copy_col_predicate(
                            tQsQ, tQcQ, mC.layout.shape[3]
                        )
                        self._copy_gmem_rows_or_zero(
                            gmem_tiled_copy_D,
                            tQgQ,
                            tQsQ,
                            tQcQ,
                            tQpQ,
                            mC.layout.shape[1],
                        )

                        gK_stage = cute.local_tile(
                            mB_source[bhc, None, 0, None],
                            (n, Ds),
                            (n_block, d_stage_idx),
                        )
                        cK_stage = cute.local_tile(
                            mcK[bhc, None, 0, None],
                            (n, Ds),
                            (n_block, d_stage_idx),
                        )
                        tKgK = gmem_thr_copy_D.partition_S(gK_stage)
                        tKcK = gmem_thr_copy_D.partition_S(cK_stage)
                        tKpK = self._make_copy_col_predicate(
                            tKsK, tKcK, mB.layout.shape[3]
                        )
                        for ni in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                            row = tKcK[0, ni, 0][1]
                            if cute.elem_less(row, mB.layout.shape[1]):
                                if (
                                    cutlass.const_expr(use_prev)
                                    and n_block == 0
                                    and is_chunk0
                                    and cute.elem_less(row, 1)
                                ):
                                    tKsK[None, ni, None].fill(0)
                                else:
                                    cute.copy(
                                        gmem_tiled_copy_D,
                                        tKgK[None, ni, None],
                                        tKsK[None, ni, None],
                                        pred=tKpK[None, ni, None],
                                    )
                            else:
                                tKsK[None, ni, None].fill(0)

                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.barrier()

                        total_q_stage = m * Ns
                        idx = tidx
                        while cute.elem_less(idx, total_q_stage):
                            rr = idx // Ns
                            vv = idx - rr * Ns
                            t_idx = m_block * m + rr
                            if cute.elem_less(t_idx, L):
                                re_col = vv * 2 + 0
                                im_col = vv * 2 + 1
                                cre = cutlass.Float32(sQ[rr, re_col])
                                cim = cutlass.Float32(sQ[rr, im_col])
                                phase_re = cutlass.Float32(sPhaseRe[t_idx])
                                phase_im = cutlass.Float32(sPhaseIm[t_idx])
                                qre = cre * phase_re + cim * phase_im
                                qim = cre * phase_im - cim * phase_re
                                sQ[rr, re_col] = qre.to(mU.element_type)
                                sQ[rr, im_col] = qim.to(mU.element_type)
                            idx = idx + self.num_threads
                        cute.arch.barrier()

                        if cutlass.const_expr(use_prev) and n_block == 0:
                            ii = tidx
                            while cute.elem_less(ii, Ns):
                                re_col = ii * 2 + 0
                                im_col = ii * 2 + 1
                                g_col_re = d_col_base + re_col
                                g_col_im = d_col_base + im_col
                                bre0 = sK[0, re_col]
                                bim0 = sK[0, im_col]
                                breb = mU.element_type(0)
                                bimb = mU.element_type(0)
                                if cute.elem_less(g_col_re, mB_prev0.layout.shape[1]):
                                    breb = mB_prev0[bh, g_col_re]
                                if cute.elem_less(g_col_im, mB_prev0.layout.shape[1]):
                                    bimb = mB_prev0[bh, g_col_im]
                                sK[0, re_col] = cutlass.select_(is_chunk0, breb, bre0)
                                sK[0, im_col] = cutlass.select_(is_chunk0, bimb, bim0)
                                ii = ii + self.num_threads
                            cute.arch.barrier()

                        total_k_stage = n * Ns
                        ii = tidx
                        while cute.elem_less(ii, total_k_stage):
                            rr = ii // Ns
                            vv = ii - rr * Ns
                            key_idx = n_block * n + rr
                            re_col = vv * 2 + 0
                            im_col = vv * 2 + 1
                            bre = cutlass.Float32(sK[rr, re_col])
                            bim = cutlass.Float32(sK[rr, im_col])

                            tr = cutlass.Float32(0.0)
                            ti = cutlass.Float32(0.0)
                            if cute.elem_less(key_idx, L):
                                tap_idx = 0 if cutlass.const_expr(use_prev) else 1
                                tap_re = cutlass.Float32(mK[bhc, key_idx, tap_idx, 0])
                                tap_im = cutlass.Float32(mK[bhc, key_idx, tap_idx, 1])
                                phase_re = cutlass.Float32(sPhaseRe[key_idx])
                                phase_im = cutlass.Float32(sPhaseIm[key_idx])
                                tr = tap_re * phase_re + tap_im * phase_im
                                ti = tap_im * phase_re - tap_re * phase_im

                            kre = bre * tr - bim * ti
                            kim = bre * ti + bim * tr
                            sK[rr, re_col] = kre.to(mU.element_type)
                            sK[rr, im_col] = (-kim).to(mU.element_type)
                            ii = ii + self.num_threads
                        cute.arch.barrier()

                        self._accumulate_from_smem_mma(
                            tiled_mma,
                            acc_S,
                            smem_tiled_copy_Q,
                            smem_tiled_copy_K,
                            tSsQ,
                            tSrQ_view,
                            tSsK,
                            tSrK_view,
                            tSrQ,
                            tSrK,
                        )

                    k_col = n_block * n + tidx
                    if tidx < n:
                        if cute.elem_less(k_col, L):
                            lp_s = cutlass.Float32(sLogpref[k_col])
                            sLpK[tidx] = cute.math.exp2(
                                -lp_s * cutlass.Float32(LOG2_E), fastmath=True
                            )
                        else:
                            sLpK[tidx] = 0.0
                    cute.arch.barrier()

                    cS = cute.local_tile(
                        mcS[bhc, None, 0, None],
                        (m, n),
                        (m_block, n_block),
                    )
                    tScS = thr_mma.partition_C(cS)
                    tScS_mn = self._make_acc_tensor_mn_view(tScS)
                    if cute.elem_less(n_block, full_visible_n_blocks):
                        self._scale_and_mask_scores(
                            acc_S,
                            tScS_mn,
                            sLpQ,
                            sLpK,
                            m_tile_start=cutlass.Int32(m_tile_start),
                            n_tile_start=n_block * n,
                            seqlen=cutlass.Int32(L),
                            apply_mask=False,
                        )
                    else:
                        self._scale_and_mask_scores(
                            acc_S,
                            tScS_mn,
                            sLpQ,
                            sLpK,
                            m_tile_start=cutlass.Int32(m_tile_start),
                            n_tile_start=n_block * n,
                            seqlen=cutlass.Int32(L),
                            apply_mask=True,
                        )

                    gVtile = cute.local_tile(
                        mU_source[bhc, None, 0, None],
                        (n, Pp),
                        (n_block, 0),
                    )
                    tVgV = gmem_thr_copy_P.partition_S(gVtile)
                    cV_tile = cute.local_tile(
                        mcV[bhc, None, 0, None],
                        (n, Pp),
                        (n_block, 0),
                    )
                    tVcV = gmem_thr_copy_P.partition_S(cV_tile)
                    for vi in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                        row = tVcV[0, vi, 0][1]
                        if cute.elem_less(row, mU.layout.shape[1]):
                            if (
                                cutlass.const_expr(use_prev)
                                and n_block == 0
                                and is_chunk0
                                and cute.elem_less(row, 1)
                            ):
                                tVsV[None, vi, None].fill(0)
                            else:
                                cute.copy(
                                    gmem_tiled_copy_P,
                                    tVgV[None, vi, None],
                                    tVsV[None, vi, None],
                                    pred=tVpV[None, vi, None],
                                )
                        else:
                            tVsV[None, vi, None].fill(0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    if cutlass.const_expr(use_prev) and n_block == 0:
                        ii = tidx
                        while cute.elem_less(ii, Pp):
                            boundary = mU.element_type(0)
                            old = sV[0, ii]
                            if cute.elem_less(ii, self.P):
                                boundary = mU_prev0[bh, ii]
                            sV[0, ii] = cutlass.select_(is_chunk0, boundary, old)
                            ii = ii + self.num_threads
                        cute.arch.barrier()

                    rP = cute.make_rmem_tensor_like(acc_S, mU.element_type)
                    rP.store(acc_S.load().to(mU.element_type))
                    rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
                    rP_mma_view = cute.make_layout(
                        (
                            (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                            rP_layout_divided.shape[1],
                            rP_layout_divided.shape[2][1],
                        ),
                        stride=(
                            (
                                rP_layout_divided.stride[0],
                                rP_layout_divided.stride[2][0],
                            ),
                            rP_layout_divided.stride[1],
                            rP_layout_divided.stride[2][1],
                        ),
                    )
                    tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

                    cute.copy(
                        smem_tiled_copy_V,
                        tOsVt[None, None, 0],
                        tOrVt_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                        k_next = (k + 1) % cute.size(tOrS.shape[2])
                        cute.copy(
                            smem_tiled_copy_V,
                            tOsVt[None, None, k_next],
                            tOrVt_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_O,
                            tOrS[None, None, k],
                            tOrVt[None, None, k],
                            acc_O,
                        )

        if cutlass.const_expr(mOut.element_type == cutlass.Float32):
            for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
                row_idx = tOcO_mn[r, 0][1]
                if cute.elem_less(row_idx, mOut.layout.shape[1]):
                    for c in cutlass.range_constexpr(cute.size(acc_O_mn.shape[1])):
                        col_idx = tOcO_mn[0, c][3]
                        if cute.elem_less(col_idx, mOut.layout.shape[3]):
                            mOut[tOcO_mn[r, c][0], row_idx, 0, col_idx] = (
                                cutlass.Float32(acc_O_mn[r, c])
                            )
        else:
            rO = cute.make_rmem_tensor_like(acc_O, mOut.element_type)
            rO.store(acc_O.load().to(mOut.element_type))

            sO = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=mOut.element_type), sO_layout
            )
            smem_copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mOut.element_type
            )
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
            cute.arch.barrier()

            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_rmem_tensor_like(tOgO, mOut.element_type)
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            tOcOut = gmem_thr_copy_O.partition_D(cO)
            tOpOut = self._make_copy_col_predicate(tOgO, tOcOut, mOut.layout.shape[3])
            self._copy_gmem_rows_or_zero(
                gmem_tiled_copy_O,
                tOrO,
                tOgO,
                tOcOut,
                tOpOut,
                mOut.layout.shape[1],
            )
