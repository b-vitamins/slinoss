"""CuTe backward slice for ``chunk_scan`` gradients into ``C``.

There are two surfaces here:

- ``chunk_scan_bwd_dc_packed_cute`` is the hot-path kernel used when the caller
  already has the cached packed forward tensors. It computes packed ``dQ`` in
  fp32 from tensor-core-friendly transport.
- ``chunk_scan_bwd_dc_cute`` is the public wrapper that maps the packed result
  back to the public ``dC`` contract via the exact fp32 phase scatter.

Numerical contract
------------------
Like the v3 ``dcdr`` slice, the dense packed contractions use fp16/bf16
transport with fp32 accumulation. That is an intentional approximation for the
``dC`` slice; the final packed-to-public scatter remains exact fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_phase,
)


LOG2_E = 1.4426950408889634


@dataclass
class _ChunkScanBwdDCScratch:
    dQ: torch.Tensor


_ScratchKey = tuple[int, torch.dtype, int, int, int, int]
_SCRATCH_DC: dict[_ScratchKey, _ChunkScanBwdDCScratch] = {}
_CompiledScatterKey = tuple[int, tuple[int, int, int], tuple[int, int, int]]
_COMPILED_DC_SCATTER: dict[_CompiledScatterKey, object] = {}
_CompiledFusedKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_DC_FUSED: dict[_CompiledFusedKey, object] = {}
_CompiledMetaKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_DC_META: dict[_CompiledMetaKey, object] = {}
_CompiledRawKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
    tuple[int, int, int, int],
]
_COMPILED_DC_RAW: dict[_CompiledRawKey, object] = {}


@dataclass(frozen=True)
class _ChunkScanBwdDCRawConfig:
    D: int
    P: int
    L: int
    tile: int = 32
    num_threads: int = 128

    def __post_init__(self) -> None:
        if self.tile != 32:
            raise ValueError("The current DC raw kernel expects tile=32.")
        if self.L % self.tile != 0:
            raise ValueError("L must be divisible by 32.")
        if self.num_threads != 128:
            raise ValueError("The current DC raw kernel expects 128 threads.")

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32


class _ChunkScanBwdDCRawAmpereTc:
    """Ampere tensor-core workhorse for packed ``dQ``.

    Logical tensors
    ---------------
    - ``mQuery``: ``(BHC, L, 1, P)``, low-precision ``d_out`` rows
    - ``mKprev/mKcurr``: ``(BHC, L, 1, P)``, low-precision packed key rows
    - ``mVprev/mVcurr``: ``(BHC, L, 1, D)``, low-precision packed value rows
    - ``mLogprefix``: ``(BHC, L)``, fp32 ``logprefix_half`` metadata
    - ``mZ0``: ``(BHC, D, 1, P)``, low-precision packed off-term factors
    - ``mOut``: ``(BHC, L, 1, D)``, fp32 packed ``dQ`` rows

    Engineering contract
    --------------------
    - One CTA owns one ``(row_tile, d_tile, bhc)`` output tile.
    - The query tile is staged once and reused for:
      - the off-term GEMM ``Q @ Z0^T`` with a rowwise ``exp(2 * lp)``
      - the two diagonal causal passes against ``(Kprev,Vprev)`` and
        ``(Kcurr,Vcurr)``
    - The diagonal score block is kept on chip and spilled to shared only long
      enough to feed the second tensor-core MMA, following the same basic
      `dcdr`/FA2 pattern as ``v3`` but stripped to the SO(2) packed contract.
    """

    def __init__(self, dtype: type[cutlass.Numeric], cfg: _ChunkScanBwdDCRawConfig):
        self.cfg = cfg
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = (16, 8, 16)
        self.warp_layout_mnk = (2, 2, 1)

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
        return cute.make_tensor(
            acc.iterator,
            cute.composition(acc.layout, acc_layout_mn),
        )

    @cute.jit
    def __call__(
        self,
        mQuery: cute.Tensor,
        mKprev: cute.Tensor,
        mVprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mVcurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mZ0: cute.Tensor,
        mOut: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mQuery.element_type
                == mKprev.element_type
                == mVprev.element_type
                == mKcurr.element_type
                == mVcurr.element_type
                == mZ0.element_type
                == self.ab_dtype
            )
        ):
            raise TypeError("DC raw inputs must share the tensor-core transport dtype.")
        if cutlass.const_expr(
            not (
                self.ab_dtype == cutlass.Float16 or self.ab_dtype == cutlass.BFloat16
            )
        ):
            raise TypeError("DC raw kernel supports only Float16/BFloat16 inputs.")
        if cutlass.const_expr(
            mLogprefix.element_type != cutlass.Float32
            or mOut.element_type != cutlass.Float32
        ):
            raise TypeError("logprefix and output must be Float32.")
        if cutlass.const_expr(
            mQuery.shape[2] != 1
            or mKprev.shape[2] != 1
            or mVprev.shape[2] != 1
            or mKcurr.shape[2] != 1
            or mVcurr.shape[2] != 1
            or mZ0.shape[2] != 1
        ):
            raise ValueError("Packed DC raw tensors must have singleton dim2.")

        Pp = self.cfg.P_padded
        d_tile = self.cfg.tile
        n = self.cfg.tile
        m = self.cfg.tile

        smem_k_block_size_P = 64 if Pp % 64 == 0 else 32
        swizzle_bits_P = 3 if smem_k_block_size_P == 64 else 2
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sQ_layout = cute.tile_to_shape(sP_layout_atom, (m, Pp), (0, 1))
        sK_layout = cute.tile_to_shape(sP_layout_atom, (n, Pp), (0, 1))
        sZ0_layout = cute.tile_to_shape(sP_layout_atom, (d_tile, Pp), (0, 1))

        smem_k_block_size_D = 64 if self.cfg.D_padded % 64 == 0 else 32
        swizzle_bits_D = 3 if smem_k_block_size_D == 64 else 2
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sV_layout = cute.tile_to_shape(sD_layout_atom, (n, d_tile), (0, 1))

        sBlk_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, n), stride=(n, 1)),
        )
        sS_layout = cute.tile_to_shape(sBlk_layout_atom, (m, n), (0, 1))
        sRowLP_layout = cute.make_layout((m,), stride=(1,))
        sColLP_layout = cute.make_layout((n,), stride=(1,))

        universal_copy_bits = 128
        async_elems_in = universal_copy_bits // mQuery.element_type.width
        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mQuery.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        tP_shape_dim_1 = sP_layout_atom.outer.shape[1] // async_elems_in
        tP_layout = cute.make_layout(
            (self.cfg.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_elems_in
        tD_layout = cute.make_layout(
            (self.cfg.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
        gmem_tiled_copy_P = cute.make_tiled_copy_tv(
            atom_async_copy_in, tP_layout, v_in_layout
        )
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        perm = (
            self.warp_layout_mnk[0] * self.mma_inst_shape[0],
            self.warp_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.warp_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.warp_layout_mnk),
            permutation_mnk=perm,
        )

        smem_size = 0
        smem_size += cute.size_in_bytes(self.ab_dtype, sQ_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sK_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sZ0_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sV_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sS_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sRowLP_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sColLP_layout)
        smem_size += 512

        grid_x = cute.ceil_div(mOut.shape[3], d_tile)
        grid_y = cute.ceil_div(mOut.shape[1], m)
        grid_z = cute.size(mOut.shape[0])
        self.kernel(
            mQuery,
            mKprev,
            mVprev,
            mKcurr,
            mVcurr,
            mLogprefix,
            mZ0,
            mOut,
            sQ_layout,
            sK_layout,
            sZ0_layout,
            sV_layout,
            sS_layout,
            sRowLP_layout,
            sColLP_layout,
            gmem_tiled_copy_P,
            gmem_tiled_copy_D,
            tiled_mma,
        ).launch(
            grid=(grid_x, grid_y, grid_z),
            block=[self.cfg.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mQuery: cute.Tensor,
        mKprev: cute.Tensor,
        mVprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mVcurr: cute.Tensor,
        mLogprefix: cute.Tensor,
        mZ0: cute.Tensor,
        mOut: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sZ0_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sS_layout: cute.ComposedLayout,
        sRowLP_layout: cute.Layout,
        sColLP_layout: cute.Layout,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_copy_D: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        d_block, m_block, bhc = cute.arch.block_idx()
        d_tile = self.cfg.tile
        n = self.cfg.tile
        m = self.cfg.tile

        smem = utils.SmemAllocator()
        sQ = smem.allocate_tensor(self.ab_dtype, sQ_layout, 16)
        sK = smem.allocate_tensor(self.ab_dtype, sK_layout, 16)
        sZ0 = smem.allocate_tensor(self.ab_dtype, sZ0_layout, 16)
        sV = smem.allocate_tensor(self.ab_dtype, sV_layout, 16)
        sS = smem.allocate_tensor(self.ab_dtype, sS_layout, 16)
        s_row_lp = smem.allocate_tensor(cutlass.Float32, sRowLP_layout, 4)
        s_col_lp = smem.allocate_tensor(cutlass.Float32, sColLP_layout, 4)

        g_thr_P = gmem_tiled_copy_P.get_slice(tidx)
        g_thr_D = gmem_tiled_copy_D.get_slice(tidx)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tSrZ0 = thr_mma.make_fragment_B(thr_mma.partition_B(sZ0))
        sVt = cute.composition(sV, cute.make_layout((d_tile, n), stride=(n, 1)))
        tSrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_S = thr_mma.partition_shape_C((m, n))
        acc_shape_Q = thr_mma.partition_shape_C((m, d_tile))
        acc_Q = cute.make_rmem_tensor(acc_shape_Q, cutlass.Float32)
        acc_Q.fill(0.0)

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.ab_dtype,
        )
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.ab_dtype,
        )
        smem_copy_atom_BT = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.ab_dtype,
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_tiled_copy_BT = cute.make_tiled_copy_B(smem_copy_atom_BT, tiled_mma)
        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        tSsQ = thr_copy_A.partition_S(sQ)
        tSrQ_view = thr_copy_A.retile(tSrQ)
        tSrS = thr_mma.make_fragment_A(thr_mma.partition_A(sS))
        tSsS = thr_copy_A.partition_S(sS)
        tSrS_view = thr_copy_A.retile(tSrS)
        tSsK = thr_copy_B.partition_S(sK)
        tSrK_view = thr_copy_B.retile(tSrK)
        tSsZ0 = thr_copy_B.partition_S(sZ0)
        tSrZ0_view = thr_copy_B.retile(tSrZ0)
        tSsVt = thr_copy_BT.partition_S(sVt)
        tSrVt_view = thr_copy_BT.retile(tSrVt)

        gQ = cute.local_tile(mQuery[bhc, None, 0, None], (m, self.cfg.P_padded), (m_block, 0))
        tQg = g_thr_P.partition_S(gQ)
        tQs = g_thr_P.partition_D(sQ)
        mcQ = cute.make_identity_tensor(mQuery.layout.shape)
        cQ = cute.local_tile(mcQ[bhc, None, 0, None], (m, self.cfg.P_padded), (m_block, 0))
        tQc = g_thr_P.partition_S(cQ)
        tQp = cute.make_rmem_tensor(
            cute.make_layout(
                (tQs.shape[0][1], cute.size(tQs, mode=[1]), cute.size(tQs, mode=[2])),
                stride=(cute.size(tQs, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQp.shape[0]):
            for vi in cutlass.range_constexpr(tQp.shape[1]):
                for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                    coord = tQc[(0, rest_v), vi, rest_k]
                    tQp[rest_v, vi, rest_k] = cute.elem_less(
                        coord[1], mQuery.shape[1]
                    ) and cute.elem_less(coord[3], mQuery.shape[3])
        for vi in cutlass.range_constexpr(cute.size(tQs.shape[1])):
            cute.copy(
                gmem_tiled_copy_P,
                tQg[None, vi, None],
                tQs[None, vi, None],
                pred=tQp[None, vi, None],
            )
        cute.arch.cp_async_commit_group()

        if tidx < cutlass.Int32(m):
            row = m_block * m + tidx
            s_row_lp[tidx] = cutlass.select_(
                cute.elem_less(row, mLogprefix.shape[1]),
                cutlass.Float32(mLogprefix[bhc, row]),
                cutlass.Float32(0.0),
            )

        gZ0 = cute.local_tile(
            mZ0[bhc, None, 0, None], (d_tile, self.cfg.P_padded), (d_block, 0)
        )
        tZg = g_thr_P.partition_S(gZ0)
        tZs = g_thr_P.partition_D(sZ0)
        mcZ0 = cute.make_identity_tensor(mZ0.layout.shape)
        cZ0 = cute.local_tile(
            mcZ0[bhc, None, 0, None], (d_tile, self.cfg.P_padded), (d_block, 0)
        )
        tZc = g_thr_P.partition_S(cZ0)
        tZp = cute.make_rmem_tensor(
            cute.make_layout(
                (tZs.shape[0][1], cute.size(tZs, mode=[1]), cute.size(tZs, mode=[2])),
                stride=(cute.size(tZs, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tZp.shape[0]):
            for vi in cutlass.range_constexpr(tZp.shape[1]):
                for rest_k in cutlass.range_constexpr(tZp.shape[2]):
                    coord = tZc[(0, rest_v), vi, rest_k]
                    tZp[rest_v, vi, rest_k] = cute.elem_less(
                        coord[1], mZ0.shape[1]
                    ) and cute.elem_less(coord[3], mZ0.shape[3])
        for vi in cutlass.range_constexpr(cute.size(tZs.shape[1])):
            cute.copy(
                gmem_tiled_copy_P,
                tZg[None, vi, None],
                tZs[None, vi, None],
                pred=tZp[None, vi, None],
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        cute.copy(smem_tiled_copy_A, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            kk_next = (kk + 1) % cute.size(tSsQ.shape[2])
            cute.copy(
                smem_tiled_copy_A,
                tSsQ[None, None, kk_next],
                tSrQ_view[None, None, kk_next],
            )

        off_acc = cute.make_rmem_tensor(acc_shape_Q, cutlass.Float32)
        off_acc.fill(0.0)
        cute.copy(smem_tiled_copy_B, tSsZ0[None, None, 0], tSrZ0_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsZ0.shape[2])):
            kk_next = (kk + 1) % cute.size(tSsZ0.shape[2])
            cute.copy(
                smem_tiled_copy_B,
                tSsZ0[None, None, kk_next],
                tSrZ0_view[None, None, kk_next],
            )
            cute.gemm(
                tiled_mma,
                off_acc,
                tSrQ[None, None, kk],
                tSrZ0[None, None, kk],
                off_acc,
            )
        mcOff = cute.make_identity_tensor(
            (mQuery.shape[0], mQuery.shape[1], mQuery.shape[2], mOut.shape[3])
        )
        cOff = cute.local_tile(mcOff[bhc, None, 0, None], (m, d_tile), (m_block, d_block))
        tOcOff = thr_mma.partition_C(cOff)
        tOcOff_mn = self._make_acc_tensor_mn_view(tOcOff)
        off_acc_mn = self._make_acc_tensor_mn_view(off_acc)
        acc_Q_mn = self._make_acc_tensor_mn_view(acc_Q)
        for r in cutlass.range_constexpr(cute.size(off_acc_mn.shape[0])):
            row_idx = cutlass.Int32(tOcOff_mn[r, 0][1])
            row_scale = cute.math.exp2(
                cutlass.Float32(2.0)
                * cutlass.Float32(s_row_lp[row_idx - m_block * m])
                * cutlass.Float32(LOG2_E)
            )
            for c in cutlass.range_constexpr(cute.size(off_acc_mn.shape[1])):
                off_acc_mn[r, c] = off_acc_mn[r, c] * row_scale
                acc_Q_mn[r, c] = off_acc_mn[r, c]

        for n_block in range(0, m_block + 1):
            if tidx < cutlass.Int32(n):
                col = n_block * n + tidx
                s_col_lp[tidx] = cutlass.select_(
                    cute.elem_less(col, mLogprefix.shape[1]),
                    cutlass.Float32(mLogprefix[bhc, col]),
                    cutlass.Float32(0.0),
                )

            for gKsrc, gVsrc in ((mKprev, mVprev), (mKcurr, mVcurr)):
                gK = cute.local_tile(gKsrc[bhc, None, 0, None], (n, self.cfg.P_padded), (n_block, 0))
                tKg = g_thr_P.partition_S(gK)
                tKs = g_thr_P.partition_D(sK)
                mcK = cute.make_identity_tensor(gKsrc.layout.shape)
                cK = cute.local_tile(mcK[bhc, None, 0, None], (n, self.cfg.P_padded), (n_block, 0))
                tKc = g_thr_P.partition_S(cK)
                tKp = cute.make_rmem_tensor(
                    cute.make_layout(
                        (
                            tKs.shape[0][1],
                            cute.size(tKs, mode=[1]),
                            cute.size(tKs, mode=[2]),
                        ),
                        stride=(cute.size(tKs, mode=[2]), 0, 1),
                    ),
                    cutlass.Boolean,
                )
                for rest_v in cutlass.range_constexpr(tKp.shape[0]):
                    for vi in cutlass.range_constexpr(tKp.shape[1]):
                        for rest_k in cutlass.range_constexpr(tKp.shape[2]):
                            coord = tKc[(0, rest_v), vi, rest_k]
                            tKp[rest_v, vi, rest_k] = cute.elem_less(
                                coord[1], gKsrc.shape[1]
                            ) and cute.elem_less(coord[3], gKsrc.shape[3])
                for vi in cutlass.range_constexpr(cute.size(tKs.shape[1])):
                    cute.copy(
                        gmem_tiled_copy_P,
                        tKg[None, vi, None],
                        tKs[None, vi, None],
                        pred=tKp[None, vi, None],
                    )
                cute.arch.cp_async_commit_group()

                gV = cute.local_tile(gVsrc[bhc, None, 0, None], (n, d_tile), (n_block, d_block))
                tVg = g_thr_D.partition_S(gV)
                tVs = g_thr_D.partition_D(sV)
                mcV = cute.make_identity_tensor(gVsrc.layout.shape)
                cV = cute.local_tile(mcV[bhc, None, 0, None], (n, d_tile), (n_block, d_block))
                tVc = g_thr_D.partition_S(cV)
                tVp = cute.make_rmem_tensor(
                    cute.make_layout(
                        (
                            tVs.shape[0][1],
                            cute.size(tVs, mode=[1]),
                            cute.size(tVs, mode=[2]),
                        ),
                        stride=(cute.size(tVs, mode=[2]), 0, 1),
                    ),
                    cutlass.Boolean,
                )
                for rest_v in cutlass.range_constexpr(tVp.shape[0]):
                    for vi in cutlass.range_constexpr(tVp.shape[1]):
                        for rest_k in cutlass.range_constexpr(tVp.shape[2]):
                            coord = tVc[(0, rest_v), vi, rest_k]
                            tVp[rest_v, vi, rest_k] = cute.elem_less(
                                coord[1], gVsrc.shape[1]
                            ) and cute.elem_less(coord[3], gVsrc.shape[3])
                for vi in cutlass.range_constexpr(cute.size(tVs.shape[1])):
                    cute.copy(
                        gmem_tiled_copy_D,
                        tVg[None, vi, None],
                        tVs[None, vi, None],
                        pred=tVp[None, vi, None],
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_B, tSsK[None, None, 0], tSrK_view[None, None, 0])
                for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSsK.shape[2])
                    cute.copy(
                        smem_tiled_copy_B,
                        tSsK[None, None, kk_next],
                        tSrK_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tSrQ[None, None, kk],
                        tSrK[None, None, kk],
                        acc_S,
                    )

                mcS = cute.make_identity_tensor(
                    (mQuery.shape[0], mQuery.shape[1], mQuery.shape[2], gKsrc.shape[1])
                )
                cS = cute.local_tile(mcS[bhc, None, 0, None], (m, n), (m_block, n_block))
                tScS = thr_mma.partition_C(cS)
                tScS_mn = self._make_acc_tensor_mn_view(tScS)
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
                    row_idx = cutlass.Int32(tScS_mn[r, 0][1])
                    for c in cutlass.range_constexpr(cute.size(acc_S_mn.shape[1])):
                        col_idx = cutlass.Int32(tScS_mn[0, c][3])
                        if cute.elem_less(row_idx + 1, col_idx + 1) or cute.elem_less(
                            gKsrc.shape[1], col_idx + 1
                        ):
                            acc_S_mn[r, c] = cutlass.Float32(0.0)
                        else:
                            row_lp = cutlass.Float32(s_row_lp[row_idx - m_block * m])
                            col_lp = cutlass.Float32(s_col_lp[col_idx - n_block * n])
                            acc_S_mn[r, c] = acc_S_mn[r, c] * cute.math.exp2(
                                (row_lp - col_lp) * cutlass.Float32(LOG2_E)
                            )
                        s_row = row_idx - m_block * m
                        s_col = col_idx - n_block * n
                        if cute.elem_less(s_row, m) and cute.elem_less(s_col, n):
                            sS[s_row, s_col] = acc_S_mn[r, c].to(self.ab_dtype)

                cute.arch.barrier()
                cute.copy(smem_tiled_copy_A, tSsS[None, None, 0], tSrS_view[None, None, 0])
                cute.copy(
                    smem_tiled_copy_BT,
                    tSsVt[None, None, 0],
                    tSrVt_view[None, None, 0],
                )
                for kk in cutlass.range_constexpr(cute.size(tSrS.shape[2])):
                    kk_next = (kk + 1) % cute.size(tSrS.shape[2])
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsS[None, None, kk_next],
                        tSrS_view[None, None, kk_next],
                    )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsVt[None, None, kk_next],
                        tSrVt_view[None, None, kk_next],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc_Q,
                        tSrS[None, None, kk],
                        tSrVt[None, None, kk],
                        acc_Q,
                    )

        mcOut = cute.make_identity_tensor(mOut.layout.shape)
        cOut = cute.local_tile(
            mcOut[bhc, None, 0, None], (m, d_tile), (m_block, d_block)
        )
        tOcOut = thr_mma.partition_C(cOut)
        tOcOut_mn = self._make_acc_tensor_mn_view(tOcOut)
        for r in cutlass.range_constexpr(cute.size(acc_Q_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_Q_mn.shape[1])):
                row_idx = cutlass.Int32(tOcOut_mn[r, c][1])
                col_idx = cutlass.Int32(tOcOut_mn[r, c][3])
                if cute.elem_less(row_idx, mOut.shape[1]) and cute.elem_less(
                    col_idx, mOut.shape[3]
                ):
                    mOut[bhc, row_idx, 0, col_idx] = acc_Q_mn[r, c]


class _ChunkScanBwdDCScatter:
    """Exact float32 scatter from packed ``dQ`` into public ``dC``.

    Logical shape:
    - ``dQ``: ``(BHC, L, D)``, interleaved complex pairs in fp32
    - ``phase``: ``(BHC, L, 2)``, unit-complex prefix in fp32
    - output ``dC_pad``: ``(BH, T_pad, D)`` in fp32

    Major mode:
    - ``D`` is the contiguous hot axis.
    - each thread owns one complex pair for one row.

    Launch / mapping:
    - grid ``(pair_tiles, row_tiles, BHC)``
    - ``bhc = bh * n_chunks + chunk`` with ``global_t = chunk * L + row``
    - writes land directly in the public padded time layout, so trimming to ``T``
      is only a final cheap slice on the host.
    """

    def __init__(self, *, pair_tile: int, num_threads: int = 128) -> None:
        self.pair_tile = int(pair_tile)
        self.num_threads = int(num_threads)
        if self.pair_tile <= 0 or self.num_threads % self.pair_tile != 0:
            raise ValueError("num_threads must be divisible by pair_tile.")
        self.row_tile = self.num_threads // self.pair_tile

    @cute.jit
    def __call__(
        self,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDQ.element_type
                == mPhase.element_type
                == mDCPad.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("dC scatter expects Float32 tensors.")
        if cutlass.const_expr(mDQ.shape[0] != mPhase.shape[0] or mDQ.shape[1] != mPhase.shape[1]):
            raise ValueError("dQ and phase must agree on (BHC, L).")
        if cutlass.const_expr(mPhase.shape[2] != 2):
            raise ValueError("phase must be (BHC, L, 2).")

        BHC = cute.size(mDQ.shape[0])
        L = cute.size(mDQ.shape[1])
        pair_cols = cute.size(mDQ.shape[2]) // 2
        grid_x = cute.ceil_div(pair_cols, self.pair_tile)
        grid_y = cute.ceil_div(L, self.row_tile)
        self.kernel(mDQ, mPhase, mDCPad, n_chunks).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pair_tile_idx, row_tile_idx, bhc = cute.arch.block_idx()

        pair_local = tidx % self.pair_tile
        row_local = tidx // self.pair_tile
        row = row_tile_idx * self.row_tile + row_local
        pair_idx = pair_tile_idx * self.pair_tile + pair_local
        pair_cols = mDQ.shape[2] // 2

        if cute.elem_less(row, mDQ.shape[1]) and cute.elem_less(pair_idx, pair_cols):
            bh = bhc // n_chunks
            chunk = bhc - bh * n_chunks
            global_t = chunk * mDQ.shape[1] + row
            col = pair_idx * 2

            pr = cutlass.Float32(mPhase[bhc, row, 0])
            pi = cutlass.Float32(mPhase[bhc, row, 1])
            dqr = cutlass.Float32(mDQ[bhc, row, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, row, col + 1])

            mDCPad[bh, global_t, col + 0] = dqr * pr + dqi * pi
            mDCPad[bh, global_t, col + 1] = dqr * pi - dqi * pr


class _ChunkScanBwdDCFused:
    """Single exact fp32 pass for public ``dC`` and ``Q/dQ`` metadata partials."""

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDQ.element_type
                == mPhase.element_type
                == mDCPad.element_type
                == mDPhase.element_type
                == mDLogprefixHalf.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact fused dC kernel expects fp32 dQ/phase/outputs.")
        if cutlass.const_expr(mQ.shape != mDQ.shape):
            raise ValueError("Q and dQ must share shape.")
        if cutlass.const_expr(mPhase.shape != (mQ.shape[0], mQ.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase.")
        if cutlass.const_expr(mDLogprefixHalf.shape != mQ.shape[:2]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mQ.shape[0])
        L = cute.size(mQ.shape[1])
        warps_per_block = self.num_threads // 32
        total_items = BHC * L
        self.kernel(
            mQ,
            mDQ,
            mPhase,
            mDCPad,
            mDPhase,
            mDLogprefixHalf,
            n_chunks,
            L,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
        L: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        bhc = item_safe // L
        row = item_safe - bhc * L
        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        global_t = chunk * L + row
        N = cute.size(mQ.shape[2]) // 2

        pr = cutlass.Float32(mPhase[bhc, row, 0])
        pi = cutlass.Float32(mPhase[bhc, row, 1])
        acc_re = cutlass.Float32(0.0)
        acc_im = cutlass.Float32(0.0)
        acc_q = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2
            qr = cutlass.Float32(mQ[bhc, row, col + 0])
            qi = cutlass.Float32(mQ[bhc, row, col + 1])
            dqr = cutlass.Float32(mDQ[bhc, row, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, row, col + 1])

            qbr = qr * pr + qi * pi
            qbi = qi * pr - qr * pi
            acc_re += dqr * qbr + dqi * qbi
            acc_im += -dqr * qbi + dqi * qbr
            acc_q += dqr * qr + dqi * qi

            if item_valid:
                mDCPad[bh, global_t, col + 0] = dqr * pr + dqi * pi
                mDCPad[bh, global_t, col + 1] = dqr * pi - dqi * pr
            n += 32

        for offset in (16, 8, 4, 2, 1):
            acc_re += cute.arch.shuffle_sync_bfly(
                acc_re, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_im += cute.arch.shuffle_sync_bfly(
                acc_im, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_q += cute.arch.shuffle_sync_bfly(
                acc_q, offset=offset, mask=-1, mask_and_clamp=31
            )

        if item_valid and lane == 0:
            mDPhase[bhc, row, 0] = acc_re
            mDPhase[bhc, row, 1] = acc_im
            mDLogprefixHalf[bhc, row] = cutlass.Float32(2.0) * acc_q


class _ChunkScanBwdDQMetaReduce:
    """Warp reduction for the ``Q/dQ`` contribution to ``(d_phase, dlogprefix)``."""

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(mPhase.element_type != cutlass.Float32):
            raise TypeError("phase must be Float32.")
        if cutlass.const_expr(
            mDQ.element_type != mDPhase.element_type
            or mDQ.element_type != mDLogprefixHalf.element_type
            or mDQ.element_type != cutlass.Float32
        ):
            raise TypeError("dQ metadata outputs must be Float32.")
        if cutlass.const_expr(mQ.shape != mDQ.shape):
            raise ValueError("Q and dQ must share shape.")
        if cutlass.const_expr(mPhase.shape != (mQ.shape[0], mQ.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase.")
        if cutlass.const_expr(mDLogprefixHalf.shape != mQ.shape[:2]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mQ.shape[0])
        L = cute.size(mQ.shape[1])
        warps_per_block = self.num_threads // 32
        total_items = BHC * L
        self.kernel(
            mQ,
            mDQ,
            mPhase,
            mDPhase,
            mDLogprefixHalf,
            L,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        L: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        bhc = item_safe // L
        t = item_safe - bhc * L
        N = cute.size(mQ.shape[2]) // 2

        pr = cutlass.Float32(mPhase[bhc, t, 0])
        pi = cutlass.Float32(mPhase[bhc, t, 1])
        acc_re = cutlass.Float32(0.0)
        acc_im = cutlass.Float32(0.0)
        acc_q = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2
            qr = cutlass.Float32(mQ[bhc, t, col + 0])
            qi = cutlass.Float32(mQ[bhc, t, col + 1])
            dqr = cutlass.Float32(mDQ[bhc, t, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, t, col + 1])

            qbr = qr * pr + qi * pi
            qbi = qi * pr - qr * pi
            acc_re += dqr * qbr + dqi * qbi
            acc_im += -dqr * qbi + dqi * qbr
            acc_q += dqr * qr + dqi * qi
            n += 32

        for offset in (16, 8, 4, 2, 1):
            acc_re += cute.arch.shuffle_sync_bfly(
                acc_re, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_im += cute.arch.shuffle_sync_bfly(
                acc_im, offset=offset, mask=-1, mask_and_clamp=31
            )
            acc_q += cute.arch.shuffle_sync_bfly(
                acc_q, offset=offset, mask=-1, mask_and_clamp=31
            )

        if item_valid and lane == 0:
            mDPhase[bhc, t, 0] = acc_re
            mDPhase[bhc, t, 1] = acc_im
            mDLogprefixHalf[bhc, t] = cutlass.Float32(2.0) * acc_q


def _get_compiled_dc_scatter(
    dQ: torch.Tensor,
    phase: torch.Tensor,
    dC_pad: torch.Tensor,
) -> object:
    device_index = 0 if dQ.device.index is None else int(dQ.device.index)
    key: _CompiledScatterKey = (
        device_index,
        tuple(int(x) for x in dQ.shape),
        tuple(int(x) for x in dC_pad.shape),
    )
    compiled = _COMPILED_DC_SCATTER.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDCScatter(pair_tile=16)
    compiled = cute.compile(
        kernel,
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        int(dC_pad.shape[1] // dQ.shape[1]),
    )
    _COMPILED_DC_SCATTER[key] = compiled
    return compiled


def _get_compiled_dc_fused(
    Q: torch.Tensor,
    dQ: torch.Tensor,
    phase: torch.Tensor,
    dC_pad: torch.Tensor,
    d_phase: torch.Tensor,
    d_logprefix_half: torch.Tensor,
) -> object:
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    key: _CompiledFusedKey = (
        device_index,
        Q.dtype,
        tuple(int(x) for x in Q.shape),
        tuple(int(x) for x in dQ.shape),
        tuple(int(x) for x in phase.shape[:2]),
        tuple(int(x) for x in dC_pad.shape),
    )
    compiled = _COMPILED_DC_FUSED.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDCFused()
    compiled = cute.compile(
        kernel,
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
        int(dC_pad.shape[1] // dQ.shape[1]),
    )
    _COMPILED_DC_FUSED[key] = compiled
    return compiled


def _get_compiled_dc_meta(
    Q: torch.Tensor,
    dQ: torch.Tensor,
    phase: torch.Tensor,
    d_phase: torch.Tensor,
    d_logprefix_half: torch.Tensor,
) -> object:
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    key: _CompiledMetaKey = (
        device_index,
        Q.dtype,
        dQ.dtype,
        tuple(int(x) for x in Q.shape),
        tuple(int(x) for x in phase.shape[:2]),
        tuple(int(x) for x in d_phase.shape),
    )
    compiled = _COMPILED_DC_META.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDQMetaReduce()
    compiled = cute.compile(
        kernel,
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
    )
    _COMPILED_DC_META[key] = compiled
    return compiled


def chunk_scan_bwd_dc_exact_cute(
    dQ: torch.Tensor,
    phase: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Scatter exact packed ``dQ`` into public ``dC`` with an fp32 CuTe kernel."""
    if dQ.device.type != "cuda" or phase.device.type != "cuda":
        raise ValueError("Exact CuTe dC scatter requires CUDA tensors.")
    if not dQ.is_contiguous() or not phase.is_contiguous():
        raise ValueError("dQ and phase must be contiguous.")
    if dQ.dtype != torch.float32 or phase.dtype != torch.float32:
        raise ValueError("Exact CuTe dC scatter expects float32 tensors.")
    if dQ.ndim != 3 or phase.shape != (*dQ.shape[:2], 2):
        raise ValueError(
            f"dQ must be (BHC, L, D) and phase must be (BHC, L, 2). Got {tuple(dQ.shape)} and {tuple(phase.shape)}."
        )

    BHC, L, D = map(int, dQ.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"dQ leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dC_pad = torch.empty((BH, T_pad, D), device=dQ.device, dtype=torch.float32)
    compiled = _get_compiled_dc_scatter(dQ, phase, dC_pad)
    compiled(
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        n_chunks,
    )
    return (
        dC_pad.reshape(batch_size, n_heads, T_pad, D)[:, :, :T, :].contiguous()
    )


def chunk_scan_bwd_dc_exact_with_meta_cute(
    Q: torch.Tensor,
    dQ: torch.Tensor,
    phase: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single exact fp32 pass for public ``dC`` and ``Q/dQ`` metadata partials."""
    if Q.device.type != "cuda" or dQ.device.type != "cuda" or phase.device.type != "cuda":
        raise ValueError("Exact fused dC path requires CUDA tensors.")
    if not Q.is_contiguous() or not dQ.is_contiguous() or not phase.is_contiguous():
        raise ValueError("Q, dQ, and phase must be contiguous.")
    if Q.ndim != 3 or dQ.shape != Q.shape:
        raise ValueError("Q and dQ must be packed as (BHC, L, D) with matching shape.")
    if phase.shape != (*dQ.shape[:2], 2):
        raise ValueError("phase must be (BHC, L, 2) matching dQ.")
    if dQ.dtype != torch.float32 or phase.dtype != torch.float32:
        raise ValueError("Exact fused dC path expects fp32 dQ and phase.")

    BHC, L, D = map(int, dQ.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"dQ leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dC_pad = torch.empty((BH, T_pad, D), device=dQ.device, dtype=torch.float32)
    d_phase = torch.empty_like(phase)
    d_logprefix_half = torch.empty((BHC, L), device=dQ.device, dtype=torch.float32)
    compiled = _get_compiled_dc_fused(
        Q,
        dQ,
        phase,
        dC_pad,
        d_phase,
        d_logprefix_half,
    )
    compiled(
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
        n_chunks,
    )
    dC = dC_pad.reshape(batch_size, n_heads, T_pad, D)[:, :, :T, :].contiguous()
    return dC, d_phase, d_logprefix_half


def chunk_scan_bwd_dq_meta_cute(
    Q: torch.Tensor,
    dQ: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the packed ``Q/dQ`` metadata contribution for ``dM``."""
    if Q.device.type != "cuda" or dQ.device.type != "cuda" or phase.device.type != "cuda":
        raise ValueError("Q, dQ, and phase must be CUDA tensors.")
    if not Q.is_contiguous() or not dQ.is_contiguous() or not phase.is_contiguous():
        raise ValueError("Q, dQ, and phase must be contiguous.")
    if Q.ndim != 4 or Q.shape[2] != 1:
        raise ValueError(f"Q must be packed as (BHC, L, 1, D). Got {tuple(Q.shape)}.")
    if dQ.ndim != 3 or dQ.shape != Q.squeeze(2).shape:
        raise ValueError("dQ must be packed as (BHC, L, D) matching Q.")
    if phase.shape != (*dQ.shape[:2], 2):
        raise ValueError("phase must be (BHC, L, 2) matching dQ.")
    if dQ.dtype != torch.float32 or phase.dtype != torch.float32:
        raise ValueError("dQ metadata reduction expects fp32 dQ and phase.")

    q_packed = Q.squeeze(2).contiguous()
    d_phase = torch.empty_like(phase)
    d_logprefix_half = torch.empty(
        dQ.shape[:2],
        device=dQ.device,
        dtype=torch.float32,
    )
    compiled = _get_compiled_dc_meta(
        q_packed,
        dQ,
        phase,
        d_phase,
        d_logprefix_half,
    )
    compiled(
        from_dlpack(q_packed, assumed_align=q_packed.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
    )
    return d_phase, d_logprefix_half


def prepare_chunk_scan_bwd_dc_operands(
    M_raw: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cached metadata for the ``chunk_scan`` ``dC`` slice.

    Returns ``(phase, half_logprefix_half, Z0_q)``.
    """
    if M_raw.ndim != 3 or M_raw.shape[-1] != 2:
        raise ValueError(f"M_raw must be (BHC, L, 2). Got {tuple(M_raw.shape)}.")
    if logprefix_half.shape != M_raw.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching M_raw. Got "
            f"{tuple(logprefix_half.shape)} for M_raw shape {tuple(M_raw.shape)}."
        )
    if Z0.ndim != 4 or Z0.shape[0] != M_raw.shape[0] or Z0.shape[2] != 1:
        raise ValueError(
            "Z0 must be the packed forward tensor shaped as (BHC, P, 1, D). "
            f"Got {tuple(Z0.shape)}."
        )
    if not (
        M_raw.is_contiguous() and logprefix_half.is_contiguous() and Z0.is_contiguous()
    ):
        raise ValueError("M_raw, logprefix_half, and Z0 must be contiguous.")

    phase = torch.empty(
        (M_raw.shape[0], M_raw.shape[1], 2),
        device=M_raw.device,
        dtype=torch.float32,
    )
    compiled_phase = _get_compiled_phase(M_raw, phase)
    compiled_phase(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
    )

    half_logprefix_half = (0.5 * logprefix_half).contiguous()
    Z0_q = Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    return phase, half_logprefix_half, Z0_q


def _get_dc_scratch(
    *,
    Kprev: torch.Tensor,
    P: int,
) -> _ChunkScanBwdDCScratch:
    device_index = 0 if Kprev.device.index is None else int(Kprev.device.index)
    BHC, L, _, D = map(int, Kprev.shape)
    key: _ScratchKey = (
        device_index,
        Kprev.dtype,
        BHC,
        L,
        P,
        D,
    )
    scratch = _SCRATCH_DC.get(key)
    if scratch is not None:
        return scratch

    dQ = torch.empty((BHC, L, 1, D), device=Kprev.device, dtype=torch.float32)
    scratch = _ChunkScanBwdDCScratch(dQ=dQ)
    _SCRATCH_DC[key] = scratch
    return scratch


def _compiled_dc_raw_key(
    query: torch.Tensor,
    kprev: torch.Tensor,
    vprev: torch.Tensor,
    logprefix: torch.Tensor,
    out: torch.Tensor,
    *,
    device_index: int,
) -> _CompiledRawKey:
    return (
        device_index,
        query.dtype,
        tuple(int(x) for x in query.shape),
        tuple(int(x) for x in kprev.shape),
        tuple(int(x) for x in vprev.shape),
        tuple(int(x) for x in logprefix.shape),
        tuple(int(x) for x in out.shape),
    )


def _get_compiled_dc_raw(
    query: torch.Tensor,
    kprev: torch.Tensor,
    vprev: torch.Tensor,
    kcurr: torch.Tensor,
    vcurr: torch.Tensor,
    logprefix: torch.Tensor,
    z0_q: torch.Tensor,
    out: torch.Tensor,
) -> object:
    device_index = 0 if query.device.index is None else int(query.device.index)
    key = _compiled_dc_raw_key(
        query,
        kprev,
        vprev,
        logprefix,
        out,
        device_index=device_index,
    )
    compiled = _COMPILED_DC_RAW.get(key)
    if compiled is not None:
        return compiled

    _, L, _, D = map(int, vprev.shape)
    P = int(query.shape[-1])
    cfg = _ChunkScanBwdDCRawConfig(D=D, P=P, L=L)
    cutlass_dtype = (
        cutlass.Float16 if query.dtype == torch.float16 else cutlass.BFloat16
    )
    kernel = _ChunkScanBwdDCRawAmpereTc(cutlass_dtype, cfg)
    compiled = cute.compile(
        kernel,
        from_dlpack(query, assumed_align=16),
        from_dlpack(kprev, assumed_align=16),
        from_dlpack(vprev, assumed_align=16),
        from_dlpack(kcurr, assumed_align=16),
        from_dlpack(vcurr, assumed_align=16),
        from_dlpack(logprefix, assumed_align=logprefix.element_size()),
        from_dlpack(z0_q, assumed_align=16),
        from_dlpack(out, assumed_align=16),
    )
    _COMPILED_DC_RAW[key] = compiled
    return compiled


def chunk_scan_bwd_dc_packed_cute(
    Vprev: torch.Tensor,
    Kprev: torch.Tensor,
    Vcurr: torch.Tensor,
    Kcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0_q: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Compute packed ``dQ`` for ``chunk_scan`` from cached forward tensors."""
    tensors = (
        ("Vprev", Vprev),
        ("Kprev", Kprev),
        ("Vcurr", Vcurr),
        ("Kcurr", Kcurr),
        ("logprefix_half", logprefix_half),
        ("Z0_q", Z0_q),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Vprev.shape != Vcurr.shape:
        raise ValueError(
            f"Vprev and Vcurr must have the same shape. Got {tuple(Vprev.shape)} "
            f"and {tuple(Vcurr.shape)}."
        )
    if Kprev.shape != Kcurr.shape:
        raise ValueError(
            f"Kprev and Kcurr must have the same shape. Got {tuple(Kprev.shape)} "
            f"and {tuple(Kcurr.shape)}."
        )
    if Vprev.ndim != 4 or Kprev.ndim != 4 or Vprev.shape[2] != 1 or Kprev.shape[2] != 1:
        raise ValueError("Packed V/K tensors must be rank-4 with a singleton dim2.")
    if logprefix_half.shape != Kprev.shape[:2]:
        raise ValueError("logprefix_half must be (BHC, L) matching Kprev.")
    if Z0_q.ndim != 4 or Z0_q.shape[0] != Kprev.shape[0] or Z0_q.shape[2] != 1:
        raise ValueError(
            f"Z0_q must be shaped as (BHC, D, 1, P). Got {tuple(Z0_q.shape)}."
        )
    if d_out.ndim != 4 or d_out.shape[:2] != (batch_size, n_heads):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T_or_T_pad, P). Got "
            f"{tuple(d_out.shape)} for batch/heads {(batch_size, n_heads)}."
        )

    BHC, L, _, D = map(int, Kprev.shape)
    P = int(Vprev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Kprev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Kprev."
        )
    if int(d_out.shape[2]) not in (T, T_pad):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P) or the corresponding "
            f"padded length T_pad={T_pad}. Got {tuple(d_out.shape)}."
        )
    if Z0_q.shape != (BHC, D, 1, P):
        raise ValueError(f"Z0_q must be {(BHC, D, 1, P)}. Got {tuple(Z0_q.shape)}.")

    scratch = _get_dc_scratch(Kprev=Kprev, P=P)

    # This path is meant to consume forward saved tensors directly. Like the
    # other CuTe backward slices, it assumes sane finite saved state and avoids
    # whole-tensor finite scans in the hot path.
    if int(d_out.shape[2]) == T:
        pad = T_pad - T
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, pad, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_tc = d_out.reshape(BHC, L, 1, P).to(dtype=Kprev.dtype).contiguous()

    compiled_raw = _get_compiled_dc_raw(
        d_out_tc,
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        logprefix_half,
        Z0_q,
        scratch.dQ,
    )
    compiled_raw(
        from_dlpack(d_out_tc, assumed_align=16),
        from_dlpack(Vprev, assumed_align=16),
        from_dlpack(Kprev, assumed_align=16),
        from_dlpack(Vcurr, assumed_align=16),
        from_dlpack(Kcurr, assumed_align=16),
        from_dlpack(
            logprefix_half, assumed_align=logprefix_half.element_size()
        ),
        from_dlpack(Z0_q, assumed_align=16),
        from_dlpack(scratch.dQ, assumed_align=16),
    )

    return scratch.dQ.squeeze(2).contiguous()


def chunk_scan_bwd_dc_cute(
    Vprev: torch.Tensor,
    Kprev: torch.Tensor,
    Vcurr: torch.Tensor,
    Kcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    half_logprefix_half: torch.Tensor,
    Z0_q: torch.Tensor,
    phase: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Public ``dC`` wrapper over the packed tensor-core ``dQ`` kernel."""
    del half_logprefix_half
    if phase.device.type != "cuda" or not phase.is_contiguous():
        raise ValueError("phase must be a contiguous CUDA tensor.")
    if phase.shape != (*Kprev.shape[:2], 2):
        raise ValueError(
            "phase must be (BHC, L, 2) matching Kprev. Got "
            f"{tuple(phase.shape)} for Kprev shape {tuple(Kprev.shape)}."
        )

    dq = chunk_scan_bwd_dc_packed_cute(
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        logprefix_half,
        Z0_q,
        d_out,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    return chunk_scan_bwd_dc_exact_cute(
        dq,
        phase,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )


__all__ = [
    "prepare_chunk_scan_bwd_dc_operands",
    "chunk_scan_bwd_dc_packed_cute",
    "chunk_scan_bwd_dq_meta_cute",
    "chunk_scan_bwd_dc_cute",
    "chunk_scan_bwd_dc_exact_cute",
    "chunk_scan_bwd_dc_exact_with_meta_cute",
]
