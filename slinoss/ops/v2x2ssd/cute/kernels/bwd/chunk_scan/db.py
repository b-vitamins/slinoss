"""CuTe backward slice for ``chunk_scan`` gradients into ``B`` and ``K``.

Logical contract
----------------
This slice consumes cached forward-packed tensors plus the lightweight raw
metadata needed to map packed key gradients back to the public operator inputs:

- ``Q_rev``: ``flip(Q, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Vprev_rev``: ``flip(Vprev, dim=1)``, shape ``(BHC, L, 1, P)``
- ``Vcurr_rev``: ``flip(Vcurr, dim=1)``, shape ``(BHC, L, 1, P)``
- ``neg_logprefix_half_rev``: ``-flip(logprefix_half, dim=1)``, shape ``(BHC, L)``
- ``phase``: ``(BHC, L, 2)``, the unit-complex phase prefix from ``M_raw``
- ``K_raw``: ``(BHC, L, 2, 2)``, raw public taps in packed-complex form
- ``B_raw``: ``(BHC, L, D)``, raw public ``B`` rows in interleaved ``2N``
- ``B_head``: ``(BHC, D)``, per-chunk boundary ``B`` input used at ``t = 0``
- ``d_out``: ``(B, H, T, P)``

Why this contract
-----------------
The packed-real key gradient is another causal attention-like contraction after:

- reversing time,
- swapping the forward value vectors into the query role,
- using reversed ``d_out`` as the key vectors,
- keeping the reversed/negated logprefix metadata.

After the dense packed ``dKprev/dKcurr`` work, the remaining map back to the
public ``(B, B_prev, K)`` contract is a short explicit complex scatter. That
host-side algebra stays readable and avoids inventing another CuTe layout
family for a non-dominant reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    ChunkScanConfig,
    ChunkScanInnerAmpereTc,
    _get_compiled_phase,
    _torch_to_cutlass_dtype,
)
from slinoss.ops.v2x2ssd.reference import (
    _pack_complex_pairs,
)


@dataclass
class _ChunkScanBwdDBScratch:
    K_zero: torch.Tensor
    V_zero: torch.Tensor
    Z0_zero: torch.Tensor
    dKprev_rev: torch.Tensor
    dKcurr_rev: torch.Tensor

_ScratchKey = tuple[int, torch.dtype, int, int, int, int]
_SCRATCH_DB: dict[_ScratchKey, _ChunkScanBwdDBScratch] = {}
_CompiledRawKey = tuple[
    int,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
]
_CompiledPairRawKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int],
    tuple[int, int, int, int],
]
_CompiledFusedKey = tuple[
    int,
    bool,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
_CompiledReduceKey = tuple[
    int,
    bool,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
]
_COMPILED_DB_RAW: dict[_CompiledRawKey, object] = {}
_COMPILED_DB_RAW_PAIR: dict[_CompiledPairRawKey, object] = {}
_COMPILED_DB_EXACT_FUSED: dict[_CompiledFusedKey, object] = {}
_COMPILED_DK_REDUCE: dict[_CompiledReduceKey, object] = {}


LOG2_E = 1.4426950408889634


def _db_raw_config(
    query_dim: int,
    value_dim: int,
    L: int,
    *,
    dtype: torch.dtype,
) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]]
    if L <= 32:
        candidates = [
            (16, 16, 64),
            (32, 16, 64),
            (32, 32, 64),
            (64, 32, 64),
        ]
    else:
        candidates = [
            (64, 32, 64),
            (64, 64, 128),
            (64, 32, 128),
            (32, 32, 64),
            (128, 64, 128),
        ]

    cutlass_in = _torch_to_cutlass_dtype(dtype)
    cutlass_out = cutlass.Float32
    for m_block_size, n_block_size, num_threads in candidates:
        if L % n_block_size != 0:
            continue
        cfg = ChunkScanConfig(
            D=query_dim,
            P=value_dim,
            L=L,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
        )
        if ChunkScanInnerAmpereTc.can_implement(cutlass_in, cutlass_out, cfg):
            return m_block_size, n_block_size, num_threads

    return 128, L, 128


@dataclass(frozen=True)
class _ChunkScanBwdDBRawPairConfig:
    D: int
    P: int
    L: int
    tile: int = 32
    num_threads: int = 128

    def __post_init__(self) -> None:
        if self.tile != 32:
            raise ValueError("The current DB raw pair kernel expects tile=32.")
        if self.L % self.tile != 0:
            raise ValueError("L must be divisible by 32.")
        if self.num_threads != 128:
            raise ValueError("The current DB raw pair kernel expects 128 threads.")

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32


class _ChunkScanBwdDBRawPairAmpereTc:
    """Ampere tensor-core kernel for packed ``dKprev`` and ``dKcurr`` together."""

    def __init__(self, dtype: type[cutlass.Numeric], cfg: _ChunkScanBwdDBRawPairConfig):
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
        mQueryPrev: cute.Tensor,
        mQueryCurr: cute.Tensor,
        mKey: cute.Tensor,
        mValue: cute.Tensor,
        mLogprefix: cute.Tensor,
        mOutPrev: cute.Tensor,
        mOutCurr: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mQueryPrev.element_type
                == mQueryCurr.element_type
                == mKey.element_type
                == mValue.element_type
                == self.ab_dtype
            )
        ):
            raise TypeError("DB raw pair inputs must share the tensor-core transport dtype.")
        if cutlass.const_expr(
            not (
                self.ab_dtype == cutlass.Float16 or self.ab_dtype == cutlass.BFloat16
            )
        ):
            raise TypeError("DB raw pair kernel supports only Float16/BFloat16 inputs.")
        if cutlass.const_expr(
            mLogprefix.element_type != cutlass.Float32
            or mOutPrev.element_type != cutlass.Float32
            or mOutCurr.element_type != cutlass.Float32
        ):
            raise TypeError("logprefix and outputs must be Float32.")
        if cutlass.const_expr(
            mQueryPrev.shape != mQueryCurr.shape
            or mQueryPrev.shape[:3] != mKey.shape[:3]
            or mQueryPrev.shape[0] != mValue.shape[0]
            or mQueryPrev.shape[1] != mValue.shape[1]
            or mQueryPrev.shape[2] != mValue.shape[2]
            or mOutPrev.shape != mOutCurr.shape
        ):
            raise ValueError("DB raw pair tensors must share packed leading shapes.")
        if cutlass.const_expr(
            mQueryPrev.shape[2] != 1
            or mKey.shape[2] != 1
            or mValue.shape[2] != 1
            or mOutPrev.shape[2] != 1
            or mOutCurr.shape[2] != 1
        ):
            raise ValueError("Packed DB raw pair tensors must have singleton dim2.")

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
        async_elems_in = universal_copy_bits // mQueryPrev.element_type.width
        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mQueryPrev.element_type,
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
        smem_size += cute.size_in_bytes(self.ab_dtype, sQ_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sK_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sV_layout)
        smem_size += cute.size_in_bytes(self.ab_dtype, sS_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sRowLP_layout)
        smem_size += cute.size_in_bytes(cutlass.Float32, sColLP_layout)
        smem_size += 512

        grid_x = cute.ceil_div(mOutPrev.shape[3], d_tile)
        grid_y = cute.ceil_div(mOutPrev.shape[1], m)
        grid_z = cute.size(mOutPrev.shape[0])
        self.kernel(
            mQueryPrev,
            mQueryCurr,
            mKey,
            mValue,
            mLogprefix,
            mOutPrev,
            mOutCurr,
            sQ_layout,
            sK_layout,
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
        mQueryPrev: cute.Tensor,
        mQueryCurr: cute.Tensor,
        mKey: cute.Tensor,
        mValue: cute.Tensor,
        mLogprefix: cute.Tensor,
        mOutPrev: cute.Tensor,
        mOutCurr: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
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
        sQPrev = smem.allocate_tensor(self.ab_dtype, sQ_layout, 16)
        sQCurr = smem.allocate_tensor(self.ab_dtype, sQ_layout, 16)
        sK = smem.allocate_tensor(self.ab_dtype, sK_layout, 16)
        sV = smem.allocate_tensor(self.ab_dtype, sV_layout, 16)
        sS = smem.allocate_tensor(self.ab_dtype, sS_layout, 16)
        s_row_lp = smem.allocate_tensor(cutlass.Float32, sRowLP_layout, 4)
        s_col_lp = smem.allocate_tensor(cutlass.Float32, sColLP_layout, 4)

        g_thr_P = gmem_tiled_copy_P.get_slice(tidx)
        g_thr_D = gmem_tiled_copy_D.get_slice(tidx)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQPrev = thr_mma.make_fragment_A(thr_mma.partition_A(sQPrev))
        tSrQCurr = thr_mma.make_fragment_A(thr_mma.partition_A(sQCurr))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        sVt = cute.composition(sV, cute.make_layout((d_tile, n), stride=(n, 1)))
        tSrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_shape_S = thr_mma.partition_shape_C((m, n))
        acc_shape_D = thr_mma.partition_shape_C((m, d_tile))
        acc_prev = cute.make_rmem_tensor(acc_shape_D, cutlass.Float32)
        acc_prev.fill(0.0)
        acc_curr = cute.make_rmem_tensor(acc_shape_D, cutlass.Float32)
        acc_curr.fill(0.0)

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
        tSsQPrev = thr_copy_A.partition_S(sQPrev)
        tSrQPrev_view = thr_copy_A.retile(tSrQPrev)
        tSsQCurr = thr_copy_A.partition_S(sQCurr)
        tSrQCurr_view = thr_copy_A.retile(tSrQCurr)
        tSrS = thr_mma.make_fragment_A(thr_mma.partition_A(sS))
        tSsS = thr_copy_A.partition_S(sS)
        tSrS_view = thr_copy_A.retile(tSrS)
        tSsK = thr_copy_B.partition_S(sK)
        tSrK_view = thr_copy_B.retile(tSrK)
        tSsVt = thr_copy_BT.partition_S(sVt)
        tSrVt_view = thr_copy_BT.retile(tSrVt)

        def _copy_query(
            gsrc: cute.Tensor,
            sdst: cute.Tensor,
            tSsQ,
            tSrQ_view,
            row_block: cutlass.Int32,
        ) -> None:
            gQ = cute.local_tile(
                gsrc[bhc, None, 0, None], (m, self.cfg.P_padded), (row_block, 0)
            )
            tQg = g_thr_P.partition_S(gQ)
            tQs = g_thr_P.partition_D(sdst)
            mcQ = cute.make_identity_tensor(gsrc.layout.shape)
            cQ = cute.local_tile(
                mcQ[bhc, None, 0, None], (m, self.cfg.P_padded), (row_block, 0)
            )
            tQc = g_thr_P.partition_S(cQ)
            tQp = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tQs.shape[0][1],
                        cute.size(tQs, mode=[1]),
                        cute.size(tQs, mode=[2]),
                    ),
                    stride=(cute.size(tQs, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQp.shape[0]):
                for vi in cutlass.range_constexpr(tQp.shape[1]):
                    for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                        coord = tQc[(0, rest_v), vi, rest_k]
                        tQp[rest_v, vi, rest_k] = cute.elem_less(
                            coord[1], gsrc.shape[1]
                        ) and cute.elem_less(coord[3], gsrc.shape[3])
            for vi in cutlass.range_constexpr(cute.size(tQs.shape[1])):
                cute.copy(
                    gmem_tiled_copy_P,
                    tQg[None, vi, None],
                    tQs[None, vi, None],
                    pred=tQp[None, vi, None],
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

        _copy_query(
            mQueryPrev,
            sQPrev,
            tSsQPrev,
            tSrQPrev_view,
            cutlass.Int32(m_block),
        )
        _copy_query(
            mQueryCurr,
            sQCurr,
            tSsQCurr,
            tSrQCurr_view,
            cutlass.Int32(m_block),
        )

        if tidx < cutlass.Int32(m):
            row = m_block * m + tidx
            s_row_lp[tidx] = cutlass.select_(
                cute.elem_less(row, mLogprefix.shape[1]),
                cutlass.Float32(mLogprefix[bhc, row]),
                cutlass.Float32(0.0),
            )

        acc_prev_mn = self._make_acc_tensor_mn_view(acc_prev)
        acc_curr_mn = self._make_acc_tensor_mn_view(acc_curr)

        for n_block in range(0, m_block + 1):
            if tidx < cutlass.Int32(n):
                col = n_block * n + tidx
                s_col_lp[tidx] = cutlass.select_(
                    cute.elem_less(col, mLogprefix.shape[1]),
                    cutlass.Float32(mLogprefix[bhc, col]),
                    cutlass.Float32(0.0),
                )

            gK = cute.local_tile(mKey[bhc, None, 0, None], (n, self.cfg.P_padded), (n_block, 0))
            tKg = g_thr_P.partition_S(gK)
            tKs = g_thr_P.partition_D(sK)
            mcK = cute.make_identity_tensor(mKey.layout.shape)
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
                            coord[1], mKey.shape[1]
                        ) and cute.elem_less(coord[3], mKey.shape[3])
            for vi in cutlass.range_constexpr(cute.size(tKs.shape[1])):
                cute.copy(
                    gmem_tiled_copy_P,
                    tKg[None, vi, None],
                    tKs[None, vi, None],
                    pred=tKp[None, vi, None],
                )
            cute.arch.cp_async_commit_group()

            gV = cute.local_tile(mValue[bhc, None, 0, None], (n, d_tile), (n_block, d_block))
            tVg = g_thr_D.partition_S(gV)
            tVs = g_thr_D.partition_D(sV)
            mcV = cute.make_identity_tensor(mValue.layout.shape)
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
                            coord[1], mValue.shape[1]
                        ) and cute.elem_less(coord[3], mValue.shape[3])
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

            cute.copy(smem_tiled_copy_B, tSsK[None, None, 0], tSrK_view[None, None, 0])
            for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                kk_next = (kk + 1) % cute.size(tSsK.shape[2])
                cute.copy(
                    smem_tiled_copy_B,
                    tSsK[None, None, kk_next],
                    tSrK_view[None, None, kk_next],
                )

            for q_frag, acc_out in ((tSrQPrev, acc_prev), (tSrQCurr, acc_curr)):
                acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
                acc_S.fill(0.0)
                for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        q_frag[None, None, kk],
                        tSrK[None, None, kk],
                        acc_S,
                    )

                mcS = cute.make_identity_tensor(
                    (mQueryPrev.shape[0], mQueryPrev.shape[1], mQueryPrev.shape[2], mKey.shape[1])
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
                            mKey.shape[1], col_idx + 1
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
                        acc_out,
                        tSrS[None, None, kk],
                        tSrVt[None, None, kk],
                        acc_out,
                    )
                cute.arch.barrier()

        mcOut = cute.make_identity_tensor(mOutPrev.layout.shape)
        cOut = cute.local_tile(
            mcOut[bhc, None, 0, None], (m, d_tile), (m_block, d_block)
        )
        tOcOut = thr_mma.partition_C(cOut)
        tOcOut_mn = self._make_acc_tensor_mn_view(tOcOut)
        for r in cutlass.range_constexpr(cute.size(acc_prev_mn.shape[0])):
            for c in cutlass.range_constexpr(cute.size(acc_prev_mn.shape[1])):
                row_idx = cutlass.Int32(tOcOut_mn[r, c][1])
                col_idx = cutlass.Int32(tOcOut_mn[r, c][3])
                if cute.elem_less(row_idx, mOutPrev.shape[1]) and cute.elem_less(
                    col_idx, mOutPrev.shape[3]
                ):
                    mOutPrev[bhc, row_idx, 0, col_idx] = acc_prev_mn[r, c]
                    mOutCurr[bhc, row_idx, 0, col_idx] = acc_curr_mn[r, c]


def _get_compiled_db_raw(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix: torch.Tensor,
    Z0: torch.Tensor,
    out: torch.Tensor,
) -> object:
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    Dq = int(Q.shape[-1])
    Dv = int(Vprev.shape[-1])
    L = int(Q.shape[1])
    config = _db_raw_config(Dq, Dv, L, dtype=Q.dtype)
    key: _CompiledRawKey = (
        device_index,
        tuple(int(x) for x in Q.shape),
        tuple(int(x) for x in Kprev.shape),
        tuple(int(x) for x in Vprev.shape),
        tuple(int(x) for x in logprefix.shape),
        tuple(int(x) for x in Z0.shape),
        config,
    )
    compiled = _COMPILED_DB_RAW.get(key)
    if compiled is not None:
        return compiled

    m_block_size, n_block_size, num_threads = config
    cfg = ChunkScanConfig(
        D=Dq,
        P=Dv,
        L=L,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        num_threads=num_threads,
    )
    kernel = ChunkScanInnerAmpereTc(cfg)
    compiled = cute.compile(
        kernel,
        from_dlpack(Q, assumed_align=16),
        from_dlpack(Kprev, assumed_align=16),
        from_dlpack(Vprev, assumed_align=16),
        from_dlpack(Kcurr, assumed_align=16),
        from_dlpack(Vcurr, assumed_align=16),
        from_dlpack(logprefix, assumed_align=16),
        from_dlpack(Z0, assumed_align=16),
        from_dlpack(out, assumed_align=16),
    )
    _COMPILED_DB_RAW[key] = compiled
    return compiled


def _compiled_db_raw_pair_key(
    query_prev: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    logprefix: torch.Tensor,
    out: torch.Tensor,
    *,
    device_index: int,
) -> _CompiledPairRawKey:
    return (
        device_index,
        query_prev.dtype,
        tuple(int(x) for x in query_prev.shape),
        tuple(int(x) for x in key.shape),
        tuple(int(x) for x in logprefix.shape),
        tuple(int(x) for x in out.shape),
    )


def _get_compiled_db_raw_pair(
    query_prev: torch.Tensor,
    query_curr: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    logprefix: torch.Tensor,
    out_prev: torch.Tensor,
    out_curr: torch.Tensor,
) -> object:
    device_index = 0 if query_prev.device.index is None else int(query_prev.device.index)
    key_cache = _compiled_db_raw_pair_key(
        query_prev,
        key,
        value,
        logprefix,
        out_prev,
        device_index=device_index,
    )
    compiled = _COMPILED_DB_RAW_PAIR.get(key_cache)
    if compiled is not None:
        return compiled

    _, L, _, D = map(int, value.shape)
    P = int(query_prev.shape[-1])
    cfg = _ChunkScanBwdDBRawPairConfig(D=D, P=P, L=L)
    cutlass_dtype = (
        cutlass.Float16 if query_prev.dtype == torch.float16 else cutlass.BFloat16
    )
    kernel = _ChunkScanBwdDBRawPairAmpereTc(cutlass_dtype, cfg)
    compiled = cute.compile(
        kernel,
        from_dlpack(query_prev, assumed_align=16),
        from_dlpack(query_curr, assumed_align=16),
        from_dlpack(key, assumed_align=16),
        from_dlpack(value, assumed_align=16),
        from_dlpack(logprefix, assumed_align=logprefix.element_size()),
        from_dlpack(out_prev, assumed_align=16),
        from_dlpack(out_curr, assumed_align=16),
    )
    _COMPILED_DB_RAW_PAIR[key_cache] = compiled
    return compiled


class _ChunkScanBwdDKExactReduce:
    """Warp reduction from exact packed intermediates into public tap gradients."""

    def __init__(self, *, num_threads: int = 128, reverse_time: bool = False) -> None:
        self.num_threads = int(num_threads)
        self.reverse_time = bool(reverse_time)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDKPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDKPrev.element_type
                == mDKCurr.element_type
                == mDKPad.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact dK reduction expects fp32 dK, outputs, and raw tensors.")
        if cutlass.const_expr(mPhase.element_type != cutlass.Float32):
            raise TypeError("phase must be Float32.")
        if cutlass.const_expr(
            mBRaw.element_type != cutlass.Float32
            or mBHead.element_type != cutlass.Float32
        ):
            raise TypeError("B_raw and B_head must be Float32.")
        if cutlass.const_expr(mKPrev.element_type != mKCurr.element_type):
            raise TypeError("Kprev and Kcurr must share dtype.")
        if cutlass.const_expr(mDKPrev.shape != mDKCurr.shape or mDKPrev.shape != mBRaw.shape):
            raise ValueError("dKprev, dKcurr, and B_raw must share shape.")
        if cutlass.const_expr(mKPrev.shape != mDKPrev.shape or mKCurr.shape != mDKPrev.shape):
            raise ValueError("Kprev/Kcurr must match the packed dK shape.")
        if cutlass.const_expr(mPhase.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mBHead.shape != (mDKPrev.shape[0], mDKPrev.shape[2])):
            raise ValueError("B_head must be (BHC, D).")
        if cutlass.const_expr(mDKPad.shape[2] != 4):
            raise ValueError("dK pad must store 4 packed tap scalars.")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase.")
        if cutlass.const_expr(mDLogprefixHalf.shape != mDKPrev.shape[:2]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mDKPrev.shape[0])
        L = cute.size(mDKPrev.shape[1])
        total_items = BHC * L
        warps_per_block = self.num_threads // 32
        self.kernel(
            mDKPrev,
            mDKCurr,
            mKPrev,
            mKCurr,
            mPhase,
            mBRaw,
            mBHead,
            mDKPad,
            mDPhase,
            mDLogprefixHalf,
            n_chunks,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDKPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        L = cute.size(mDKPrev.shape[1])
        bhc = item_safe // L
        row = item_safe - bhc * L
        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        global_t = chunk * L + row
        src_row = (
            (L - cutlass.Int32(1)) - row
            if cutlass.const_expr(self.reverse_time)
            else row
        )
        N = cute.size(mDKPrev.shape[2]) // 2

        pr = cutlass.Float32(mPhase[bhc, row, 0])
        pi = cutlass.Float32(mPhase[bhc, row, 1])
        acc_prev_re = cutlass.Float32(0.0)
        acc_prev_im = cutlass.Float32(0.0)
        acc_curr_re = cutlass.Float32(0.0)
        acc_curr_im = cutlass.Float32(0.0)
        acc_phase_re = cutlass.Float32(0.0)
        acc_phase_im = cutlass.Float32(0.0)
        acc_kp = cutlass.Float32(0.0)
        acc_kc = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2
            dkp_re = cutlass.Float32(mDKPrev[bhc, src_row, col + 0])
            dkp_im = cutlass.Float32(mDKPrev[bhc, src_row, col + 1])
            dkc_re = cutlass.Float32(mDKCurr[bhc, src_row, col + 0])
            dkc_im = cutlass.Float32(mDKCurr[bhc, src_row, col + 1])
            kpr = cutlass.Float32(mKPrev[bhc, row, col + 0])
            kpi = cutlass.Float32(mKPrev[bhc, row, col + 1])
            kcr = cutlass.Float32(mKCurr[bhc, row, col + 0])
            kci = cutlass.Float32(mKCurr[bhc, row, col + 1])
            kpbr = kpr * pr + kpi * pi
            kpbi = kpi * pr - kpr * pi
            kcbr = kcr * pr + kci * pi
            kcbi = kci * pr - kcr * pi
            dbp_re = pr * dkp_re + pi * dkp_im
            dbp_im = pi * dkp_re - pr * dkp_im
            dbc_re = pr * dkc_re + pi * dkc_im
            dbc_im = pi * dkc_re - pr * dkc_im

            bpr = cutlass.Float32(0.0)
            bpi = cutlass.Float32(0.0)
            if row == cutlass.Int32(0):
                bpr = cutlass.Float32(mBHead[bhc, col + 0])
                bpi = cutlass.Float32(mBHead[bhc, col + 1])
            else:
                bpr = cutlass.Float32(mBRaw[bhc, row - 1, col + 0])
                bpi = cutlass.Float32(mBRaw[bhc, row - 1, col + 1])
            bcr = cutlass.Float32(mBRaw[bhc, row, col + 0])
            bci = cutlass.Float32(mBRaw[bhc, row, col + 1])

            acc_prev_re += bpr * dbp_re + bpi * dbp_im
            acc_prev_im += bpr * dbp_im - bpi * dbp_re
            acc_curr_re += bcr * dbc_re + bci * dbc_im
            acc_curr_im += bcr * dbc_im - bci * dbc_re
            acc_phase_re += dkp_re * kpbr + dkp_im * kpbi
            acc_phase_im += -dkp_re * kpbi + dkp_im * kpbr
            acc_phase_re += dkc_re * kcbr + dkc_im * kcbi
            acc_phase_im += -dkc_re * kcbi + dkc_im * kcbr
            acc_kp += dkp_re * kpr + dkp_im * kpi
            acc_kc += dkc_re * kcr + dkc_im * kci
            n += 32

        for offset in (16, 8, 4, 2, 1):
            acc_prev_re += cute.arch.shuffle_sync_bfly(acc_prev_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_prev_im += cute.arch.shuffle_sync_bfly(acc_prev_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_re += cute.arch.shuffle_sync_bfly(acc_curr_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_im += cute.arch.shuffle_sync_bfly(acc_curr_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_phase_re += cute.arch.shuffle_sync_bfly(acc_phase_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_phase_im += cute.arch.shuffle_sync_bfly(acc_phase_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_kp += cute.arch.shuffle_sync_bfly(acc_kp, offset=offset, mask=-1, mask_and_clamp=31)
            acc_kc += cute.arch.shuffle_sync_bfly(acc_kc, offset=offset, mask=-1, mask_and_clamp=31)

        if item_valid and lane == 0:
            mDKPad[bh, global_t, 0] = acc_prev_re
            mDKPad[bh, global_t, 1] = acc_prev_im
            mDKPad[bh, global_t, 2] = acc_curr_re
            mDKPad[bh, global_t, 3] = acc_curr_im
            mDPhase[bhc, row, 0] = acc_phase_re
            mDPhase[bhc, row, 1] = acc_phase_im
            mDLogprefixHalf[bhc, row] = -cutlass.Float32(2.0) * (acc_kp + acc_kc)


class _ChunkScanBwdDBExactFused:
    """Single exact fp32 pass for public ``dB/dB_prev/dK`` and metadata partials."""

    def __init__(self, *, num_threads: int = 128, reverse_time: bool = False) -> None:
        self.num_threads = int(num_threads)
        self.reverse_time = bool(reverse_time)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDBPad: cute.Tensor,
        mDBPrev: cute.Tensor,
        mDKPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDKPrev.element_type
                == mDKCurr.element_type
                == mPhase.element_type
                == mKRaw.element_type
                == mBRaw.element_type
                == mBHead.element_type
                == mDBPad.element_type
                == mDBPrev.element_type
                == mDKPad.element_type
                == mDPhase.element_type
                == mDLogprefixHalf.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("Exact fused db kernel expects Float32 exact tensors.")
        if cutlass.const_expr(mKPrev.element_type != mKCurr.element_type):
            raise TypeError("Kprev and Kcurr must share dtype.")
        if cutlass.const_expr(
            mDKPrev.shape != mDKCurr.shape
            or mDKPrev.shape != mKPrev.shape
            or mDKPrev.shape != mKCurr.shape
            or mDKPrev.shape != mBRaw.shape
        ):
            raise ValueError("dKprev/dKcurr/Kprev/Kcurr/B_raw must share shape.")
        if cutlass.const_expr(mPhase.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2)):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mKRaw.shape != (mDKPrev.shape[0], mDKPrev.shape[1], 2, 2)):
            raise ValueError("K_raw must be (BHC, L, 2, 2).")
        if cutlass.const_expr(mBHead.shape != (mDKPrev.shape[0], mDKPrev.shape[2])):
            raise ValueError("B_head must be (BHC, D).")
        if cutlass.const_expr(mDKPad.shape[2] != 4):
            raise ValueError("dK pad must store 4 packed tap scalars.")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase.")
        if cutlass.const_expr(mDLogprefixHalf.shape != mDKPrev.shape[:2]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        L = cute.size(mDKPrev.shape[1])
        warps_per_block = self.num_threads // 32
        self.kernel(
            mDKPrev,
            mDKCurr,
            mKPrev,
            mKCurr,
            mPhase,
            mKRaw,
            mBRaw,
            mBHead,
            mDBPad,
            mDBPrev,
            mDKPad,
            mDPhase,
            mDLogprefixHalf,
            n_chunks,
            L,
        ).launch(
            grid=[cute.ceil_div(L, warps_per_block), 1, cute.size(mDKPrev.shape[0])],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDKPrev: cute.Tensor,
        mDKCurr: cute.Tensor,
        mKPrev: cute.Tensor,
        mKCurr: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mBRaw: cute.Tensor,
        mBHead: cute.Tensor,
        mDBPad: cute.Tensor,
        mDBPrev: cute.Tensor,
        mDKPad: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        n_chunks: cutlass.Int32,
        L: cutlass.Int32,
    ) -> None:
        row_tile_idx, _, bhc = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        row = row_tile_idx * warps_per_block + warp
        row_valid = cute.elem_less(row, L)
        row_safe = cutlass.min(row, L - cutlass.Int32(1))

        bh = bhc // n_chunks
        chunk = bhc - bh * n_chunks
        global_t = chunk * L + row_safe
        src_row = (
            (L - cutlass.Int32(1)) - row_safe
            if cutlass.const_expr(self.reverse_time)
            else row_safe
        )
        N = cute.size(mDKPrev.shape[2]) // 2

        pr = cutlass.Float32(mPhase[bhc, row_safe, 0])
        pi = cutlass.Float32(mPhase[bhc, row_safe, 1])
        acc_prev_re = cutlass.Float32(0.0)
        acc_prev_im = cutlass.Float32(0.0)
        acc_curr_re = cutlass.Float32(0.0)
        acc_curr_im = cutlass.Float32(0.0)
        acc_phase_re = cutlass.Float32(0.0)
        acc_phase_im = cutlass.Float32(0.0)
        acc_kp = cutlass.Float32(0.0)
        acc_kc = cutlass.Float32(0.0)

        n = lane
        while n < N:
            col = n * 2
            dkp_re = cutlass.Float32(mDKPrev[bhc, src_row, col + 0])
            dkp_im = cutlass.Float32(mDKPrev[bhc, src_row, col + 1])
            dkc_re = cutlass.Float32(mDKCurr[bhc, src_row, col + 0])
            dkc_im = cutlass.Float32(mDKCurr[bhc, src_row, col + 1])

            dbp_re = pr * dkp_re + pi * dkp_im
            dbp_im = pi * dkp_re - pr * dkp_im
            dbc_re = pr * dkc_re + pi * dkc_im
            dbc_im = pi * dkc_re - pr * dkc_im

            kpr = cutlass.Float32(mKPrev[bhc, row_safe, col + 0])
            kpi = cutlass.Float32(mKPrev[bhc, row_safe, col + 1])
            kcr = cutlass.Float32(mKCurr[bhc, row_safe, col + 0])
            kci = cutlass.Float32(mKCurr[bhc, row_safe, col + 1])
            kpbr = kpr * pr + kpi * pi
            kpbi = kpi * pr - kpr * pi
            kcbr = kcr * pr + kci * pi
            kcbi = kci * pr - kcr * pi

            kcr_tap = cutlass.Float32(mKRaw[bhc, row_safe, 1, 0])
            kci_tap = cutlass.Float32(mKRaw[bhc, row_safe, 1, 1])
            out_re = kcr_tap * dbc_re + kci_tap * dbc_im
            out_im = kcr_tap * dbc_im - kci_tap * dbc_re

            next_row = row_safe + 1
            if cute.elem_less(next_row, L):
                next_src_row = (
                    (L - cutlass.Int32(1)) - next_row
                    if cutlass.const_expr(self.reverse_time)
                    else next_row
                )
                npr = cutlass.Float32(mPhase[bhc, next_row, 0])
                npi = cutlass.Float32(mPhase[bhc, next_row, 1])
                ndkp_re = cutlass.Float32(mDKPrev[bhc, next_src_row, col + 0])
                ndkp_im = cutlass.Float32(mDKPrev[bhc, next_src_row, col + 1])
                ndbp_re = npr * ndkp_re + npi * ndkp_im
                ndbp_im = npi * ndkp_re - npr * ndkp_im
                nkpr = cutlass.Float32(mKRaw[bhc, next_row, 0, 0])
                nkpi = cutlass.Float32(mKRaw[bhc, next_row, 0, 1])
                out_re += nkpr * ndbp_re + nkpi * ndbp_im
                out_im += nkpr * ndbp_im - nkpi * ndbp_re
            else:
                next_chunk = chunk + 1
                if cute.elem_less(next_chunk, n_chunks):
                    next_bhc = bhc + 1
                    next_src_row = (
                        L - cutlass.Int32(1)
                        if cutlass.const_expr(self.reverse_time)
                        else cutlass.Int32(0)
                    )
                    npr = cutlass.Float32(mPhase[next_bhc, 0, 0])
                    npi = cutlass.Float32(mPhase[next_bhc, 0, 1])
                    ndkp_re = cutlass.Float32(mDKPrev[next_bhc, next_src_row, col + 0])
                    ndkp_im = cutlass.Float32(mDKPrev[next_bhc, next_src_row, col + 1])
                    ndbp_re = npr * ndkp_re + npi * ndkp_im
                    ndbp_im = npi * ndkp_re - npr * ndkp_im
                    nkpr = cutlass.Float32(mKRaw[next_bhc, 0, 0, 0])
                    nkpi = cutlass.Float32(mKRaw[next_bhc, 0, 0, 1])
                    out_re += nkpr * ndbp_re + nkpi * ndbp_im
                    out_im += nkpr * ndbp_im - nkpi * ndbp_re

            if row_valid:
                mDBPad[bh, global_t, col + 0] = out_re
                mDBPad[bh, global_t, col + 1] = out_im

            if row_valid and chunk == cutlass.Int32(0) and row_safe == cutlass.Int32(0):
                kpr0 = cutlass.Float32(mKRaw[bhc, 0, 0, 0])
                kpi0 = cutlass.Float32(mKRaw[bhc, 0, 0, 1])
                mDBPrev[bh, col + 0] = kpr0 * dbp_re + kpi0 * dbp_im
                mDBPrev[bh, col + 1] = kpr0 * dbp_im - kpi0 * dbp_re

            bpr = cutlass.Float32(0.0)
            bpi = cutlass.Float32(0.0)
            if row_safe == cutlass.Int32(0):
                bpr = cutlass.Float32(mBHead[bhc, col + 0])
                bpi = cutlass.Float32(mBHead[bhc, col + 1])
            else:
                bpr = cutlass.Float32(mBRaw[bhc, row_safe - 1, col + 0])
                bpi = cutlass.Float32(mBRaw[bhc, row_safe - 1, col + 1])
            bcr = cutlass.Float32(mBRaw[bhc, row_safe, col + 0])
            bci = cutlass.Float32(mBRaw[bhc, row_safe, col + 1])
            acc_prev_re += bpr * dbp_re + bpi * dbp_im
            acc_prev_im += bpr * dbp_im - bpi * dbp_re
            acc_curr_re += bcr * dbc_re + bci * dbc_im
            acc_curr_im += bcr * dbc_im - bci * dbc_re
            acc_phase_re += dkp_re * kpbr + dkp_im * kpbi
            acc_phase_im += -dkp_re * kpbi + dkp_im * kpbr
            acc_phase_re += dkc_re * kcbr + dkc_im * kcbi
            acc_phase_im += -dkc_re * kcbi + dkc_im * kcbr
            acc_kp += dkp_re * kpr + dkp_im * kpi
            acc_kc += dkc_re * kcr + dkc_im * kci
            n += 32

        for offset in (16, 8, 4, 2, 1):
            acc_prev_re += cute.arch.shuffle_sync_bfly(acc_prev_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_prev_im += cute.arch.shuffle_sync_bfly(acc_prev_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_re += cute.arch.shuffle_sync_bfly(acc_curr_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_curr_im += cute.arch.shuffle_sync_bfly(acc_curr_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_phase_re += cute.arch.shuffle_sync_bfly(acc_phase_re, offset=offset, mask=-1, mask_and_clamp=31)
            acc_phase_im += cute.arch.shuffle_sync_bfly(acc_phase_im, offset=offset, mask=-1, mask_and_clamp=31)
            acc_kp += cute.arch.shuffle_sync_bfly(acc_kp, offset=offset, mask=-1, mask_and_clamp=31)
            acc_kc += cute.arch.shuffle_sync_bfly(acc_kc, offset=offset, mask=-1, mask_and_clamp=31)

        if row_valid and lane == 0:
            mDKPad[bh, global_t, 0] = acc_prev_re
            mDKPad[bh, global_t, 1] = acc_prev_im
            mDKPad[bh, global_t, 2] = acc_curr_re
            mDKPad[bh, global_t, 3] = acc_curr_im
            mDPhase[bhc, row_safe, 0] = acc_phase_re
            mDPhase[bhc, row_safe, 1] = acc_phase_im
            mDLogprefixHalf[bhc, row_safe] = -cutlass.Float32(2.0) * (acc_kp + acc_kc)


def _get_compiled_db_exact_fused(
    dKprev: torch.Tensor,
    Kprev: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    dB_pad: torch.Tensor,
    dB_prev: torch.Tensor,
    dK_pad: torch.Tensor,
    d_phase: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    *,
    reverse_time: bool = False,
) -> object:
    device_index = 0 if dKprev.device.index is None else int(dKprev.device.index)
    key: _CompiledFusedKey = (
        device_index,
        bool(reverse_time),
        Kprev.dtype,
        tuple(int(x) for x in dKprev.shape),
        tuple(int(x) for x in Kprev.shape),
        tuple(int(x) for x in dB_pad.shape),
    )
    compiled = _COMPILED_DB_EXACT_FUSED.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDBExactFused(reverse_time=reverse_time)
    compiled = cute.compile(
        kernel,
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(B_raw, assumed_align=B_raw.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(dB_pad, assumed_align=dB_pad.element_size()),
        from_dlpack(dB_prev, assumed_align=dB_prev.element_size()),
        from_dlpack(dK_pad, assumed_align=dK_pad.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
        int(dB_pad.shape[1] // dKprev.shape[1]),
    )
    _COMPILED_DB_EXACT_FUSED[key] = compiled
    return compiled
def _get_compiled_dk_exact_reduce(
    dKprev: torch.Tensor,
    Kprev: torch.Tensor,
    phase: torch.Tensor,
    B_head: torch.Tensor,
    dK_pad: torch.Tensor,
    d_phase: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    *,
    reverse_time: bool = False,
) -> object:
    device_index = 0 if dKprev.device.index is None else int(dKprev.device.index)
    key: _CompiledReduceKey = (
        device_index,
        bool(reverse_time),
        Kprev.dtype,
        tuple(int(x) for x in dKprev.shape),
        tuple(int(x) for x in Kprev.shape),
        tuple(int(x) for x in B_head.shape),
        tuple(int(x) for x in dK_pad.shape),
        tuple(int(x) for x in d_phase.shape),
    )
    compiled = _COMPILED_DK_REDUCE.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDKExactReduce(reverse_time=reverse_time)
    compiled = cute.compile(
        kernel,
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(dK_pad, assumed_align=dK_pad.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
        int(dK_pad.shape[1] // dKprev.shape[1]),
    )
    _COMPILED_DK_REDUCE[key] = compiled
    return compiled


def chunk_scan_bwd_db_exact_with_meta_cute(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    Kprev_packed: torch.Tensor,
    Kcurr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
    reverse_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fp32 CuTe scatter from packed key grads into public ``dB/dB_prev/dK`` and metadata partials."""
    tensors = (
        ("dK_prev_packed", dK_prev_packed),
        ("dK_curr_packed", dK_curr_packed),
        ("Kprev_packed", Kprev_packed),
        ("Kcurr_packed", Kcurr_packed),
        ("phase", phase),
        ("K_raw", K_raw),
        ("B_raw", B_raw),
        ("B_head", B_head),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("Exact CuTe dB scatter requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError("Exact CuTe dB scatter expects contiguous tensors.")
    if any(t.dtype != torch.float32 for _name, t in tensors if _name not in {"Kprev_packed", "Kcurr_packed"}):
        raise ValueError("Exact CuTe dB scatter expects fp32 exact tensors.")
    if dK_prev_packed.shape != dK_curr_packed.shape or dK_prev_packed.shape != B_raw.shape:
        raise ValueError("dK_prev_packed, dK_curr_packed, and B_raw must share shape.")
    if Kprev_packed.shape != dK_prev_packed.shape or Kcurr_packed.shape != dK_prev_packed.shape:
        raise ValueError("Kprev_packed/Kcurr_packed must match dK tensors.")
    if phase.shape != (*dK_prev_packed.shape[:2], 2):
        raise ValueError("phase must be (BHC, L, 2).")
    if K_raw.shape != (*dK_prev_packed.shape[:2], 2, 2):
        raise ValueError("K_raw must be (BHC, L, 2, 2).")
    if B_head.shape != (dK_prev_packed.shape[0], dK_prev_packed.shape[2]):
        raise ValueError("B_head must be (BHC, D).")

    BHC, L, D = map(int, dK_prev_packed.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"dK leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dB_pad = torch.empty((BH, T_pad, D), device=dK_prev_packed.device, dtype=torch.float32)
    dB_prev = torch.empty((BH, D), device=dK_prev_packed.device, dtype=torch.float32)
    dK_pad = torch.empty((BH, T_pad, 4), device=dK_prev_packed.device, dtype=torch.float32)
    d_phase = torch.empty_like(phase)
    d_logprefix_half = torch.empty((BHC, L), device=dK_prev_packed.device, dtype=torch.float32)

    compiled_fused = _get_compiled_db_exact_fused(
        dK_prev_packed,
        Kprev_packed,
        phase,
        K_raw,
        B_raw,
        B_head,
        dB_pad,
        dB_prev,
        dK_pad,
        d_phase,
        d_logprefix_half,
        reverse_time=reverse_time,
    )

    compiled_fused(
        from_dlpack(dK_prev_packed, assumed_align=dK_prev_packed.element_size()),
        from_dlpack(dK_curr_packed, assumed_align=dK_curr_packed.element_size()),
        from_dlpack(Kprev_packed, assumed_align=Kprev_packed.element_size()),
        from_dlpack(Kcurr_packed, assumed_align=Kcurr_packed.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(B_raw, assumed_align=B_raw.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(dB_pad, assumed_align=dB_pad.element_size()),
        from_dlpack(dB_prev, assumed_align=dB_prev.element_size()),
        from_dlpack(dK_pad, assumed_align=dK_pad.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(
            d_logprefix_half,
            assumed_align=d_logprefix_half.element_size(),
        ),
        n_chunks,
    )
    dB = dB_pad.reshape(batch_size, n_heads, T_pad, D)[:, :, :T, :].contiguous()
    dB_prev_out = dB_prev.reshape(batch_size, n_heads, D).contiguous()
    dK = dK_pad.reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :].contiguous()
    return dB, dB_prev_out, dK, d_phase, d_logprefix_half


def chunk_scan_bwd_db_exact_cute(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
    reverse_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fp32 CuTe scatter from packed key grads into public ``dB/dB_prev/dK``."""
    dB, dB_prev, dK, _d_phase, _d_logprefix_half = chunk_scan_bwd_db_exact_with_meta_cute(
        dK_prev_packed,
        dK_curr_packed,
        torch.zeros_like(dK_prev_packed),
        torch.zeros_like(dK_curr_packed),
        phase,
        K_raw,
        B_raw,
        B_head,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
        reverse_time=reverse_time,
    )
    return dB, dB_prev, dK


def prepare_chunk_scan_bwd_db_operands(
    Q: torch.Tensor,
    Vprev: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    M_raw: torch.Tensor,
    *,
    Q_rev: torch.Tensor | None = None,
    neg_logprefix_half_rev: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cached reverse-time operands plus phase metadata for the ``dB`` slice."""
    if Q.ndim != 4 or Vprev.ndim != 4 or Vcurr.ndim != 4:
        raise ValueError("Q/Vprev/Vcurr must be rank-4 tensors.")
    if Q.shape[:3] != Vprev.shape[:3] or Q.shape[:3] != Vcurr.shape[:3]:
        raise ValueError(
            "Q/Vprev/Vcurr must agree on the leading packed dims. Got "
            f"{tuple(Q.shape)}, {tuple(Vprev.shape)}, {tuple(Vcurr.shape)}."
        )
    if Q.shape[2] != 1 or Vprev.shape[2] != 1 or Vcurr.shape[2] != 1:
        raise ValueError("Packed Q/V tensors must be shaped as (BHC, L, 1, feat).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)} for Q shape {tuple(Q.shape)}."
        )
    if not (
        Q.is_contiguous()
        and Vprev.is_contiguous()
        and Vcurr.is_contiguous()
        and logprefix_half.is_contiguous()
        and M_raw.is_contiguous()
    ):
        raise ValueError(
            "Q, Vprev, Vcurr, logprefix_half, and M_raw must be contiguous cached "
            "forward tensors."
        )

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
    if Q_rev is None:
        Q_rev = torch.flip(Q, dims=[1]).contiguous()
    elif not Q_rev.is_contiguous() or Q_rev.shape != Q.shape:
        raise ValueError("Q_rev must be contiguous and match Q when provided.")
    if neg_logprefix_half_rev is None:
        neg_logprefix_half_rev = (-torch.flip(logprefix_half, dims=[1])).contiguous()
    elif (
        not neg_logprefix_half_rev.is_contiguous()
        or neg_logprefix_half_rev.shape != logprefix_half.shape
    ):
        raise ValueError(
            "neg_logprefix_half_rev must be contiguous and match logprefix_half "
            "when provided."
        )

    return (
        Q_rev,
        torch.flip(Vprev, dims=[1]).contiguous(),
        torch.flip(Vcurr, dims=[1]).contiguous(),
        neg_logprefix_half_rev,
        phase,
    )


def _get_db_scratch(
    *,
    vprev_rev: torch.Tensor,
    D: int,
) -> _ChunkScanBwdDBScratch:
    device_index = 0 if vprev_rev.device.index is None else int(vprev_rev.device.index)
    BHC, L, _, P = map(int, vprev_rev.shape)
    key: _ScratchKey = (
        device_index,
        vprev_rev.dtype,
        BHC,
        L,
        P,
        D,
    )
    scratch = _SCRATCH_DB.get(key)
    if scratch is not None:
        return scratch

    K_zero = torch.zeros_like(vprev_rev)
    V_zero = torch.zeros((BHC, L, 1, D), device=vprev_rev.device, dtype=vprev_rev.dtype)
    Z0_zero = torch.zeros(
        (BHC, D, 1, P), device=vprev_rev.device, dtype=vprev_rev.dtype
    )
    dKprev_rev = torch.empty(
        (BHC, L, 1, D), device=vprev_rev.device, dtype=torch.float32
    )
    dKcurr_rev = torch.empty_like(dKprev_rev)
    scratch = _ChunkScanBwdDBScratch(
        K_zero=K_zero,
        V_zero=V_zero,
        Z0_zero=Z0_zero,
        dKprev_rev=dKprev_rev,
        dKcurr_rev=dKcurr_rev,
    )
    _SCRATCH_DB[key] = scratch
    return scratch
def chunk_scan_bwd_dk_packed_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
    reverse_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute packed ``dKprev/dKcurr`` on the cached reverse-time contract."""
    BHC, L, _, D = map(int, Q_rev.shape)
    P = int(Vprev_rev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )
    if not d_out.is_contiguous():
        raise ValueError("d_out must be contiguous.")

    if T_pad != T:
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
    d_out_rev = torch.flip(
        d_out.reshape(BHC, L, 1, P).to(dtype=Vprev_rev.dtype), dims=[1]
    ).contiguous()
    return _chunk_scan_bwd_dk_prepared_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        d_out_rev,
        batch_size=batch_size,
        n_heads=n_heads,
        reverse_time=reverse_time,
    )


def _chunk_scan_bwd_dk_prepared_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out_rev: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    reverse_time: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute packed ``dKprev/dKcurr`` from already padded reverse-time ``d_out``."""
    if not d_out_rev.is_contiguous():
        raise ValueError("d_out_rev must be contiguous.")

    BHC, _, _, D = map(int, Q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )

    scratch = _get_db_scratch(vprev_rev=Vprev_rev, D=D)
    compiled_pair = _get_compiled_db_raw_pair(
        Vprev_rev,
        Vcurr_rev,
        d_out_rev,
        Q_rev,
        neg_logprefix_half_rev,
        scratch.dKprev_rev,
        scratch.dKcurr_rev,
    )
    compiled_pair(
        from_dlpack(Vprev_rev, assumed_align=16),
        from_dlpack(Vcurr_rev, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.dKprev_rev, assumed_align=16),
        from_dlpack(scratch.dKcurr_rev, assumed_align=16),
    )
    if reverse_time:
        return scratch.dKprev_rev.squeeze(2), scratch.dKcurr_rev.squeeze(2)
    return (
        torch.flip(scratch.dKprev_rev.squeeze(2), dims=[1]).contiguous(),
        torch.flip(scratch.dKcurr_rev.squeeze(2), dims=[1]).contiguous(),
    )


def _chunk_scan_bwd_db_prepared_packed_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    phase: torch.Tensor,
    Kprev_packed: torch.Tensor,
    Kcurr_packed: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out_rev: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the packed ``db`` path on prepared reverse-time operands."""
    if not d_out_rev.is_contiguous():
        raise ValueError("d_out_rev must be contiguous.")

    BHC, L, _, D = map(int, Q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    dK_prev_packed_rev, dK_curr_packed_rev = _chunk_scan_bwd_dk_prepared_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        d_out_rev,
        batch_size=batch_size,
        n_heads=n_heads,
        reverse_time=True,
    )
    return chunk_scan_bwd_db_exact_with_meta_cute(
        dK_prev_packed_rev,
        dK_curr_packed_rev,
        Kprev_packed,
        Kcurr_packed,
        phase,
        K_raw,
        B_raw,
        B_head,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T_pad,
        reverse_time=True,
    )


def chunk_scan_bwd_db_packed_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    phase: torch.Tensor,
    Kprev_packed: torch.Tensor,
    Kcurr_packed: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute public ``(dB, dB_prev, dK)`` and metadata from cached reverse-time packs."""
    BHC, L, _, _ = map(int, Q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )
    if not d_out.is_contiguous():
        raise ValueError("d_out must be contiguous.")

    if T_pad != T:
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, T_pad - T, d_out.shape[-1]),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_rev = torch.flip(
        d_out.reshape(BHC, L, 1, d_out.shape[-1]).to(dtype=Vprev_rev.dtype),
        dims=[1],
    ).contiguous()
    dB_pad, dB_prev, dK_pad, d_phase, d_logprefix_half = (
        _chunk_scan_bwd_db_prepared_packed_cute(
            Q_rev,
            Vprev_rev,
            Vcurr_rev,
            neg_logprefix_half_rev,
            phase,
            Kprev_packed,
            Kcurr_packed,
            K_raw,
            B_raw,
            B_head,
            d_out_rev,
            batch_size=batch_size,
            n_heads=n_heads,
        )
    )
    return (
        dB_pad[:, :, :T, :].contiguous(),
        dB_prev.contiguous(),
        dK_pad[:, :, :T, :, :].contiguous(),
        d_phase.contiguous(),
        d_logprefix_half.contiguous(),
    )


def chunk_scan_bwd_db_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(dB, dB_prev, dK)`` for ``chunk_scan`` from cached forward packs."""
    tensors = (
        ("Q_rev", Q_rev),
        ("Vprev_rev", Vprev_rev),
        ("Vcurr_rev", Vcurr_rev),
        ("neg_logprefix_half_rev", neg_logprefix_half_rev),
        ("phase", phase),
        ("K_raw", K_raw),
        ("B_raw", B_raw),
        ("B_head", B_head),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q_rev.ndim != 4 or Vprev_rev.ndim != 4 or Vcurr_rev.ndim != 4:
        raise ValueError("Q_rev/Vprev_rev/Vcurr_rev must be rank-4 tensors.")
    if Q_rev.shape[:3] != Vprev_rev.shape[:3] or Q_rev.shape[:3] != Vcurr_rev.shape[:3]:
        raise ValueError(
            "Q_rev/Vprev_rev/Vcurr_rev must share leading packed dims. Got "
            f"{tuple(Q_rev.shape)}, {tuple(Vprev_rev.shape)}, {tuple(Vcurr_rev.shape)}."
        )
    if Q_rev.shape[2] != 1 or Vprev_rev.shape[2] != 1 or Vcurr_rev.shape[2] != 1:
        raise ValueError("Packed reverse-time tensors must be (BHC, L, 1, feat).")
    if neg_logprefix_half_rev.shape != Q_rev.shape[:2]:
        raise ValueError(
            "neg_logprefix_half_rev must be (BHC, L) matching Q_rev. Got "
            f"{tuple(neg_logprefix_half_rev.shape)}."
        )
    if phase.shape != (*Q_rev.shape[:2], 2):
        raise ValueError(
            f"phase must be (BHC, L, 2) matching Q_rev. Got {tuple(phase.shape)}."
        )
    if K_raw.shape != (*Q_rev.shape[:2], 2, 2):
        raise ValueError(
            "K_raw must be (BHC, L, 2, 2). Got "
            f"{tuple(K_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_raw.shape != (*Q_rev.shape[:2], Q_rev.shape[-1]):
        raise ValueError(
            "B_raw must be (BHC, L, D) matching Q_rev. Got "
            f"{tuple(B_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_head.shape != (Q_rev.shape[0], Q_rev.shape[-1]):
        raise ValueError(
            "B_head must be (BHC, D) matching Q_rev. Got "
            f"{tuple(B_head.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if (
        d_out.ndim != 4
        or d_out.shape[:2] != (batch_size, n_heads)
        or int(d_out.shape[2]) != T
    ):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P). Got "
            f"{tuple(d_out.shape)} for {(batch_size, n_heads, T)}."
        )

    BHC, L, _, D = map(int, Q_rev.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )

    cplx_dtype = torch.complex64
    N = D // 2
    phase_c = (
        torch.view_as_complex(phase.contiguous()).to(dtype=cplx_dtype).unsqueeze(-1)
    )
    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous()).to(
        dtype=cplx_dtype
    )
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous()).to(
        dtype=cplx_dtype
    )
    b_curr = torch.view_as_complex(B_raw.reshape(BHC, L, N, 2).contiguous()).to(
        dtype=cplx_dtype
    )
    b_head_c = torch.view_as_complex(B_head.reshape(BHC, N, 2).contiguous()).to(
        dtype=cplx_dtype
    )
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if L > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]
    Kprev_packed = _pack_complex_pairs(
        torch.conj(k_prev_c.unsqueeze(-1) * b_prev_seq) * phase_c,
        real_dtype=Q_rev.dtype,
    ).contiguous()
    Kcurr_packed = _pack_complex_pairs(
        torch.conj(k_curr_c.unsqueeze(-1) * b_curr) * phase_c,
        real_dtype=Q_rev.dtype,
    ).contiguous()
    dB, dB_prev, dK, _d_phase, _d_logprefix_half = chunk_scan_bwd_db_packed_cute(
        Q_rev,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev,
        phase,
        Kprev_packed,
        Kcurr_packed,
        K_raw,
        B_raw,
        B_head,
        d_out,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    return dB, dB_prev, dK


__all__ = [
    "prepare_chunk_scan_bwd_db_operands",
    "chunk_scan_bwd_dk_packed_cute",
    "chunk_scan_bwd_db_packed_cute",
    "chunk_scan_bwd_db_cute",
    "chunk_scan_bwd_db_exact_cute",
    "chunk_scan_bwd_db_exact_with_meta_cute",
]
