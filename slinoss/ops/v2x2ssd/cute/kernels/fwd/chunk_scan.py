"""CuTe forward kernel for the ``v2x2ssd`` chunk-scan stage.

This stage preserves the reference contract but uses an Ampere tensor-core
inner kernel for the dense chunk-local work. The current path keeps the inner
kernel on-device and performs the SO(2)-specific packing on the host:

- ``Q[t]     = pack(conj(C[t]) * phase_prefix[t])``
- ``Kprev[s] = pack(conj(beta_prev[s]) * phase_prefix[s])``
- ``Kcurr[s] = pack(conj(beta_curr[s]) * phase_prefix[s])``
- ``Vprev[s] = U[s-1]`` with the explicit chunk boundary value at ``s=0``
- ``Vcurr[s] = U[s]``
- ``Z0       = pack(conj(chunk_starts))``

The inner kernel then evaluates the off-term and both diagonal causal passes
with the stable segment-ratio construction from the reference path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _chunked_transition_prefix_parts,
    _complex_dtype_from_real,
    _pad_time_full,
    _resolve_dtypes,
    _resolve_prev0,
    _to_complex_scalar,
    _to_complex_taps,
    _validate_chunk_scan_inputs,
)


LOG2_E = 1.4426950408889634
TWO_LOG2_E = 2.0 * LOG2_E

_CompiledKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int, int],
    tuple[int, int],
]
_COMPILED_CHUNK_SCAN: dict[_CompiledKey, object] = {}
_CompiledPackDKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int],
]
_CompiledPhaseKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int],
]
_CompiledPackPKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int],
]
_CompiledPackZ0Key = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int],
]
_CompiledFusedKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int, int],
    int,
]
_COMPILED_CHUNK_SCAN_FUSED: dict[_CompiledFusedKey, object] = {}
_COMPILED_PACK_D: dict[_CompiledPackDKey, object] = {}
_COMPILED_PACK_P: dict[_CompiledPackPKey, object] = {}
_COMPILED_PACK_Z0: dict[_CompiledPackZ0Key, object] = {}
_COMPILED_PHASE: dict[_CompiledPhaseKey, object] = {}


@dataclass(frozen=True)
class ChunkScanConfig:
    D: int
    P: int
    L: int
    m_block_size: int = 128
    n_block_size: int = 64
    num_threads: int = 128

    def __post_init__(self) -> None:
        if self.m_block_size % 16 != 0:
            raise ValueError("m_block_size must be a multiple of 16.")
        if self.n_block_size % 16 != 0:
            raise ValueError("n_block_size must be a multiple of 16.")
        if self.L % self.n_block_size != 0:
            raise ValueError("L must be divisible by n_block_size.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")

    @property
    def D_padded(self) -> int:
        return ((self.D + 31) // 32) * 32

    @property
    def P_padded(self) -> int:
        return ((self.P + 31) // 32) * 32

    @property
    def n_block_max(self) -> int:
        return (self.L + self.n_block_size - 1) // self.n_block_size


class ChunkScanInnerAmpereTc:
    """Ampere tensor-core kernel for the dense chunk-local scan work."""

    def __init__(self, cfg: ChunkScanConfig):
        self.cfg = cfg

    @staticmethod
    def can_implement(
        dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        cfg: ChunkScanConfig,
    ) -> bool:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if out_dtype not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32):
            return False
        if cfg.D % 8 != 0 or cfg.P % 8 != 0:
            return False

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        in_bytes = dtype.width // 8
        out_bytes = out_dtype.width // 8
        Dp = cfg.D_padded
        Pp = cfg.P_padded
        m = cfg.m_block_size
        n = cfg.n_block_size

        compute_smem = 0
        compute_smem += m * Dp * in_bytes
        compute_smem += max(Pp, n) * Dp * in_bytes
        compute_smem += n * Pp * in_bytes
        compute_smem += (m + n) * 4
        out_smem = (m * Pp) * out_bytes
        return max(compute_smem, out_smem) <= smem_capacity

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

        Dp = self.cfg.D_padded
        Pp = self.cfg.P_padded
        m = self.cfg.m_block_size
        n = self.cfg.n_block_size
        num_threads = self.cfg.num_threads

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

        universal_copy_bits = 128
        in_dtype = mQ.element_type
        out_dtype = mOut.element_type
        async_elems_in = universal_copy_bits // in_dtype.width

        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            in_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_elems_in
        tD_layout = cute.make_layout(
            (num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        tP_shape_dim_1 = sP_layout_atom.outer.shape[1] // async_elems_in
        tP_layout = cute.make_layout(
            (num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
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

        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(in_dtype, cutlass.Float32, (16, 8, 16)),
            (num_threads // 32, 1, 1),
            permutation_mnk=(num_threads // 32 * 16, 16, 16),
        )

        compute_smem = 0
        compute_smem += cute.size_in_bytes(in_dtype, sQ_layout)
        compute_smem += cute.size_in_bytes(in_dtype, sB_layout)
        compute_smem += cute.size_in_bytes(in_dtype, sV_layout)
        compute_smem += (m + n) * 4
        out_smem = cute.size_in_bytes(out_dtype, sO_layout)
        smem_size = max(compute_smem, out_smem)

        grid_dim = (
            cute.ceil_div(mQ.shape[1], m),
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
            sQ_layout,
            sB_layout,
            sK_layout,
            sZ_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_D,
            gmem_tiled_copy_P,
            gmem_tiled_copy_O,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=[num_threads, 1, 1],
            smem=smem_size,
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
        sQ_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sZ_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, bhc, _ = cute.arch.block_idx()

        Dp = self.cfg.D_padded
        Pp = self.cfg.P_padded
        m = self.cfg.m_block_size
        n = self.cfg.n_block_size
        n_block_max = self.cfg.n_block_max

        gQ = cute.local_tile(mQ[bhc, None, 0, None], (m, Dp), (m_block, 0))
        gKprev = cute.local_tile(mKprev[bhc, None, 0, None], (n, Dp), (None, 0))
        gKcurr = cute.local_tile(mKcurr[bhc, None, 0, None], (n, Dp), (None, 0))
        gVprev = cute.local_tile(mVprev[bhc, None, 0, None], (n, Pp), (None, 0))
        gVcurr = cute.local_tile(mVcurr[bhc, None, 0, None], (n, Pp), (None, 0))
        gZ0 = cute.local_tile(mZ0[bhc, None, 0, None], (Pp, Dp), (0, 0))
        gO = cute.local_tile(mOut[bhc, None, 0, None], (m, Pp), (m_block, 0))

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mQ.element_type, sQ_layout, 16)
        sB = smem.allocate_tensor(mQ.element_type, sB_layout, 16)
        sV = smem.allocate_tensor(mQ.element_type, sV_layout, 16)

        sK = cute.make_tensor(sB.iterator, sK_layout)
        sZ = cute.make_tensor(sB.iterator, sZ_layout)

        sLpQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((m,)), 4)
        sLpK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((n,)), 4)
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
        tQpQ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ.shape[0][1],
                    cute.size(tQsQ, mode=[1]),
                    cute.size(tQsQ, mode=[2]),
                ),
                stride=(cute.size(tQsQ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
            for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                    tQcQ[(0, rest_v), 0, rest_k][3], mQ.layout.shape[3]
                )

        mcK = cute.make_identity_tensor(mKprev.layout.shape)
        cK0 = cute.local_tile(mcK[bhc, None, 0, None], (n, Dp), (0, 0))
        tKcK0 = gmem_thr_copy_D.partition_S(cK0)
        tKpK = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tKsK.shape[0][1],
                    cute.size(tKsK, mode=[1]),
                    cute.size(tKsK, mode=[2]),
                ),
                stride=(cute.size(tKsK, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tKpK.shape[0]):
            for rest_k in cutlass.range_constexpr(tKpK.shape[2]):
                tKpK[rest_v, 0, rest_k] = cute.elem_less(
                    tKcK0[(0, rest_v), 0, rest_k][3], mKprev.layout.shape[3]
                )

        mcZ = cute.make_identity_tensor(mZ0.layout.shape)
        cZ = cute.local_tile(mcZ[bhc, None, 0, None], (Pp, Dp), (0, 0))
        tZcZ = gmem_thr_copy_D.partition_S(cZ)
        tZpZ = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tZsZ.shape[0][1],
                    cute.size(tZsZ, mode=[1]),
                    cute.size(tZsZ, mode=[2]),
                ),
                stride=(cute.size(tZsZ, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tZpZ.shape[0]):
            for rest_k in cutlass.range_constexpr(tZpZ.shape[2]):
                tZpZ[rest_v, 0, rest_k] = cute.elem_less(
                    tZcZ[(0, rest_v), 0, rest_k][3], mZ0.layout.shape[3]
                )

        mcV = cute.make_identity_tensor(mVcurr.layout.shape)
        cV0 = cute.local_tile(mcV[bhc, None, 0, None], (n, Pp), (0, 0))
        tVcV0 = gmem_thr_copy_P.partition_S(cV0)
        tVpV = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tVsV.shape[0][1],
                    cute.size(tVsV, mode=[1]),
                    cute.size(tVsV, mode=[2]),
                ),
                stride=(cute.size(tVsV, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tVpV.shape[0]):
            for rest_k in cutlass.range_constexpr(tVpV.shape[2]):
                tVpV[rest_v, 0, rest_k] = cute.elem_less(
                    tVcV0[(0, rest_v), 0, rest_k][3], mVcurr.layout.shape[3]
                )

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tSrZ = thr_mma.make_fragment_B(thr_mma.partition_B(sZ))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_shape_O = thr_mma.partition_shape_C((m, Pp))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)
        acc_shape_S = thr_mma.partition_shape_C((m, n))

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
            lp = cutlass.Float32(0.0)
            if cute.elem_less(q_row, mLogprefix.shape[1]):
                lp = cutlass.Float32(mLogprefix[bhc, q_row])
            sLpQ[tidx] = cute.math.exp2(lp * TWO_LOG2_E, fastmath=True)

        for mi in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if cute.elem_less(tQcQ[0, mi, 0][1], mQ.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_D,
                    tQgQ[None, mi, None],
                    tQsQ[None, mi, None],
                    pred=tQpQ[None, mi, None],
                )
            else:
                tQsQ[None, mi, None].fill(0)
        for zi in cutlass.range_constexpr(cute.size(tZsZ.shape[1])):
            if cute.elem_less(tZcZ[0, zi, 0][1], mZ0.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_D,
                    tZgZ[None, zi, None],
                    tZsZ[None, zi, None],
                    pred=tZpZ[None, zi, None],
                )
            else:
                tZsZ[None, zi, None].fill(0)
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        cute.copy(smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
        cute.copy(smem_tiled_copy_K, tSsZ[None, None, 0], tSrZ_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            k_next = (k + 1) % cute.size(tSsQ.shape[2])
            cute.copy(
                smem_tiled_copy_Q,
                tSsQ[None, None, k_next],
                tSrQ_view[None, None, k_next],
            )
            cute.copy(
                smem_tiled_copy_K,
                tSsZ[None, None, k_next],
                tSrZ_view[None, None, k_next],
            )
            cute.gemm(tiled_mma, acc_O, tSrQ[None, None, k], tSrZ[None, None, k], acc_O)

        mcO = cute.make_identity_tensor(mOut.layout.shape)
        cO = cute.local_tile(mcO[bhc, None, 0, None], (m, Pp), (m_block, 0))
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        m_tile_start = m_block * m
        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
            row_idx = tOcO_mn[r, 0][1]
            if cute.elem_less(row_idx, mOut.layout.shape[1]):
                s2 = sLpQ[row_idx - m_tile_start]
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * s2

        mcS = cute.make_identity_tensor(
            (mQ.shape[0], mQ.shape[1], mQ.shape[2], mKprev.shape[1])
        )
        for pass_id in cutlass.range_constexpr(2):
            use_prev = cutlass.const_expr(pass_id == 0)
            tKgK = tKgKprev if use_prev else tKgKcurr
            tVgV = tVgVprev if use_prev else tVgVcurr

            for ni in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKcK0[0, ni, 0][1], mKprev.layout.shape[1]):
                    cute.copy(
                        gmem_tiled_copy_D,
                        tKgK[None, ni, None, 0],
                        tKsK[None, ni, None],
                        pred=tKpK[None, ni, None],
                    )
                else:
                    tKsK[None, ni, None].fill(0)
            cute.arch.cp_async_commit_group()

            for n_block in cutlass.range_constexpr(n_block_max):
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                k_col = n_block * n + tidx
                if tidx < n:
                    if cute.elem_less(k_col, mLogprefix.shape[1]):
                        lp_s = cutlass.Float32(mLogprefix[bhc, k_col])
                        sLpK[tidx] = cute.math.exp2(-lp_s * TWO_LOG2_E, fastmath=True)
                    else:
                        sLpK[tidx] = 0.0

                cV = cute.local_tile(mcV[bhc, None, 0, None], (n, Pp), (n_block, 0))
                tVcV = gmem_thr_copy_P.partition_S(cV)
                for vi in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                    if cute.elem_less(tVcV[0, vi, 0][1], mVcurr.layout.shape[1]):
                        cute.copy(
                            gmem_tiled_copy_P,
                            tVgV[None, vi, None, n_block],
                            tVsV[None, vi, None],
                            pred=tVpV[None, vi, None],
                        )
                    else:
                        tVsV[None, vi, None].fill(0)
                cute.arch.cp_async_commit_group()

                acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
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
                self._scale_and_mask_scores(
                    acc_S,
                    tScS_mn,
                    sLpQ,
                    sLpK,
                    m_tile_start=m_tile_start,
                    n_tile_start=n_block * n,
                    seqlen=mQ.layout.shape[1],
                )

                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                if cutlass.const_expr(n_block + 1 < n_block_max):
                    cK_next = cute.local_tile(
                        mcK[bhc, None, 0, None], (n, Dp), (n_block + 1, 0)
                    )
                    tKcK_next = gmem_thr_copy_D.partition_S(cK_next)
                    for ni in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if cute.elem_less(
                            tKcK_next[0, ni, 0][1], mKprev.layout.shape[1]
                        ):
                            cute.copy(
                                gmem_tiled_copy_D,
                                tKgK[None, ni, None, n_block + 1],
                                tKsK[None, ni, None],
                                pred=tKpK[None, ni, None],
                            )
                        else:
                            tKsK[None, ni, None].fill(0)
                    cute.arch.cp_async_commit_group()

                rP = cute.make_rmem_tensor_like(acc_S, mQ.element_type)
                rP.store(acc_S.load().to(mQ.element_type))
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

                cute.copy(
                    smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0]
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

        mcOut = cute.make_identity_tensor(mOut.layout.shape)
        cOut = cute.local_tile(mcOut[bhc, None, 0, None], (m, Pp), (m_block, 0))
        tOcOut = gmem_thr_copy_O.partition_D(cOut)
        tOpOut = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpOut.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpOut.shape[2])):
                tOpOut[rest_v, 0, rest_n] = cute.elem_less(
                    tOcOut[(0, rest_v), 0, rest_n][3], mOut.layout.shape[3]
                )

        for rest_m in cutlass.range_constexpr(cute.size(tOpOut.shape[1])):
            if cute.elem_less(tOcOut[0, rest_m, 0][1], mOut.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpOut[None, rest_m, None],
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
        m_tile_start: cutlass.Int32,
        n_tile_start: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        scale_buf = cute.make_rmem_tensor(acc_S_mn[0, None].layout, cutlass.Float32)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            row_idx = cutlass.Int32(tScS_mn[r, 0][1])
            row_scale = cutlass.Float32(1.0)
            if cute.elem_less(row_idx, seqlen):
                row_scale = cutlass.Float32(sLpQ[row_idx - m_tile_start])

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


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


def _pack_complex_pairs_tc(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z.resolve_conj())
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _tc_dtype_for_inputs(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return dtype
    if dtype == torch.float32:
        return torch.float16
    raise TypeError(f"Unsupported dtype for chunk_scan_cute: {dtype}")


def _choose_pair_tile(D: int) -> int:
    # One thread owns one interleaved complex pair. Keep the pair axis contiguous
    # in x so each warp issues coalesced half-pair loads/stores.
    return 32 if D >= 64 else 16


def _choose_feat_tile(P: int) -> int:
    return 32 if P >= 32 else 16


def _prepare_chunk_scan_small_operands(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
    torch.dtype,
]:
    batch_size, n_heads, T, N, P, n_chunks = _validate_chunk_scan_inputs(
        U, M, K, B, C, chunk_starts, B_prev, U_prev, int(U.shape[2]), chunk_size
    )
    D = 2 * N

    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype, chunk_starts.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_scan path supports only float32 reference "
            f"compute. Got compute_dtype={rdtype}."
        )

    device = U.device

    U_f, M_f, K_f, B_f, C_f, T_pad, _ = _pad_time_full(
        U, M, K, B, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)
    BHC = batch_size * n_heads * n_chunks

    M_blk = M_f.reshape(batch_size, n_heads, n_chunks, L, 2)
    K_raw = K_f.reshape(BHC, L, 2, 2).contiguous()

    mr = M_blk[..., 0]
    mi = M_blk[..., 1]
    # Fast CuTe paths assume the SLinOSS operating region, where transitions are
    # strictly nonzero by construction. Clamp the squared magnitude away from
    # zero here to avoid a synchronizing device-side ``any()`` check in the hot
    # path while still keeping ``logprefix_half`` finite under tiny underflow.
    mag2 = (mr * mr + mi * mi).clamp_min(torch.finfo(rdtype).tiny)
    logprefix_half = (
        (0.25 * torch.cumsum(torch.log(mag2), dim=-1)).reshape(BHC, L).contiguous()
    )
    M_raw = M_blk.reshape(BHC, L, 2).contiguous()

    U_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P)
    B_blk = B_f.reshape(batch_size, n_heads, n_chunks, L, D)
    C_blk = C_f.reshape(batch_size, n_heads, n_chunks, L, D)
    Z0_raw = chunk_starts.reshape(BHC, P, D).contiguous()

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=_complex_dtype_from_real(rdtype),
    )

    U_head = torch.empty(
        (batch_size, n_heads, n_chunks, P), device=device, dtype=rdtype
    )
    U_head[:, :, 0] = u_prev0
    if n_chunks > 1:
        U_head[:, :, 1:] = U_blk[:, :, :-1, -1, :]

    B_head = torch.empty(
        (batch_size, n_heads, n_chunks, D), device=device, dtype=rdtype
    )
    B_head[:, :, 0] = torch.view_as_real(b_prev0).reshape(batch_size, n_heads, D)
    if n_chunks > 1:
        B_head[:, :, 1:] = B_blk[:, :, :-1, -1, :]

    # Keep raw scan inputs in fp32 here. The lightweight CuTe prep kernels cast
    # directly into the packed tensor-core inputs, which avoids an extra full
    # ``U/B/C`` materialization pass before the real packing work.
    U_raw = U_blk.reshape(BHC, L, P).contiguous()
    B_raw = B_blk.reshape(BHC, L, D).contiguous()
    C_raw = C_blk.reshape(BHC, L, D).contiguous()
    U_head = U_head.reshape(BHC, P).contiguous()
    B_head = B_head.reshape(BHC, D).contiguous()

    return (
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix_half,
        Z0_raw,
        U_head,
        B_head,
        batch_size,
        n_heads,
        T,
        T_pad,
        odtype,
    )


def _prepare_chunk_scan_operands(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
    torch.dtype,
]:
    batch_size, n_heads, T, N, P, n_chunks = _validate_chunk_scan_inputs(
        U, M, K, B, C, chunk_starts, B_prev, U_prev, int(U.shape[2]), chunk_size
    )
    D = 2 * N

    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype, chunk_starts.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_scan path supports only float32 reference "
            f"compute. Got compute_dtype={rdtype}."
        )

    tc_dtype = _tc_dtype_for_inputs(U.dtype)
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device

    U_f, M_f, K_f, B_f, C_f, T_pad, _ = _pad_time_full(
        U, M, K, B, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_f, name="C").to(dtype=cplx_dtype))
    z0 = _as_complex_pairs(chunk_starts.to(dtype=rdtype), name="chunk_starts").to(
        dtype=cplx_dtype
    )

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    b_prev_seq = torch.cat([b_prev0.unsqueeze(2), b_t[:, :, :-1]], dim=2)
    beta_prev = k_prev.unsqueeze(-1) * b_prev_seq
    beta_curr = k_curr.unsqueeze(-1) * b_t

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    u_curr_blk = U_f.reshape(batch_size, n_heads, n_chunks, L, P)
    u_prev_boundary = torch.empty(
        (batch_size, n_heads, n_chunks, P), device=device, dtype=rdtype
    )
    u_prev_boundary[:, :, 0] = u_prev0
    if n_chunks > 1:
        u_prev_boundary[:, :, 1:] = u_curr_blk[:, :, :-1, -1, :]

    c_blk = c_conj.reshape(batch_size, n_heads, n_chunks, L, N)
    bprev_blk = beta_prev.reshape(batch_size, n_heads, n_chunks, L, N)
    bcurr_blk = beta_curr.reshape(batch_size, n_heads, n_chunks, L, N)

    logprefix, phase_prefix, _ = _chunked_transition_prefix_parts(m_blk)
    BHC = batch_size * n_heads * n_chunks
    phase_flat = phase_prefix.reshape(BHC, L)

    q = c_blk.reshape(BHC, L, N) * phase_flat.unsqueeze(-1)
    kprev_rot = torch.conj(bprev_blk.reshape(BHC, L, N)) * phase_flat.unsqueeze(-1)
    kcurr_rot = torch.conj(bcurr_blk.reshape(BHC, L, N)) * phase_flat.unsqueeze(-1)

    v_curr = u_curr_blk.reshape(BHC, L, P)
    v_prev = torch.empty_like(v_curr)
    v_prev[:, 0, :] = u_prev_boundary.reshape(BHC, P)
    if L > 1:
        v_prev[:, 1:, :] = v_curr[:, :-1, :]

    z0_flat = torch.conj(z0.reshape(BHC, P, N))
    logprefix_half = (0.5 * logprefix.reshape(BHC, L)).contiguous()

    Q = _pack_complex_pairs_tc(q, real_dtype=tc_dtype).unsqueeze(2)
    Kprev = _pack_complex_pairs_tc(kprev_rot, real_dtype=tc_dtype).unsqueeze(2)
    Kcurr = _pack_complex_pairs_tc(kcurr_rot, real_dtype=tc_dtype).unsqueeze(2)
    Vprev = v_prev.to(dtype=tc_dtype).contiguous().unsqueeze(2)
    Vcurr = v_curr.to(dtype=tc_dtype).contiguous().unsqueeze(2)
    Z0 = _pack_complex_pairs_tc(z0_flat, real_dtype=tc_dtype).unsqueeze(2)

    return (
        Q,
        Kprev,
        Vprev,
        Kcurr,
        Vcurr,
        logprefix_half,
        Z0,
        batch_size,
        n_heads,
        T,
        T_pad,
        odtype,
    )


def _compiled_key(
    Q: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    device_index: int,
) -> _CompiledKey:
    return (
        device_index,
        Q.dtype,
        output_dtype,
        tuple(int(x) for x in Q.shape),
        (128, int(Q.shape[1])),
    )


def _get_compiled_chunk_scan(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix: torch.Tensor,
    Z0: torch.Tensor,
    out: torch.Tensor,
) -> object:
    if Q.device.type != "cuda":
        raise ValueError("CuTe chunk_scan requires CUDA tensors.")
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    key = _compiled_key(Q, output_dtype=out.dtype, device_index=device_index)
    compiled = _COMPILED_CHUNK_SCAN.get(key)
    if compiled is not None:
        return compiled

    BHC, L, _, D = map(int, Q.shape)
    P = int(Vprev.shape[-1])
    del BHC
    n_block_size = L
    cfg = ChunkScanConfig(
        D=D,
        P=P,
        L=L,
        m_block_size=128,
        n_block_size=n_block_size,
        num_threads=128,
    )
    cutlass_in_dtype = _torch_to_cutlass_dtype(Q.dtype)
    cutlass_out_dtype = _torch_to_cutlass_dtype(out.dtype)
    if not ChunkScanInnerAmpereTc.can_implement(
        cutlass_in_dtype, cutlass_out_dtype, cfg
    ):
        raise ValueError(
            "ChunkScanInnerAmpereTc cannot implement this shape/dtype on SM80."
        )

    mQ = from_dlpack(Q.contiguous(), assumed_align=16)
    mKprev = from_dlpack(Kprev.contiguous(), assumed_align=16)
    mVprev = from_dlpack(Vprev.contiguous(), assumed_align=16)
    mKcurr = from_dlpack(Kcurr.contiguous(), assumed_align=16)
    mVcurr = from_dlpack(Vcurr.contiguous(), assumed_align=16)
    mLogp = from_dlpack(logprefix.contiguous(), assumed_align=16)
    mZ0 = from_dlpack(Z0.contiguous(), assumed_align=16)
    mOut = from_dlpack(out.contiguous(), assumed_align=16)

    kernel = ChunkScanInnerAmpereTc(cfg)
    compiled = cute.compile(
        kernel, mQ, mKprev, mVprev, mKcurr, mVcurr, mLogp, mZ0, mOut
    )
    _COMPILED_CHUNK_SCAN[key] = compiled
    return compiled


class _ChunkScanPackD:
    """Packs raw ``C/B/K`` plus phase metadata into ``Q/Kprev/Kcurr``.

    Logical shapes:
    - ``C_raw, B_raw``: ``(BHC, L, D)``
    - ``B_head``: ``(BHC, D)``
    - ``phase``: ``(BHC, L, 2)``
    - ``K_raw``: ``(BHC, L, 2, 2)``
    - outputs: ``(BHC, L, 1, D)``

    Major mode:
    - ``D`` is the contiguous hot axis.
    - each thread owns one complex pair ``(re, im)`` for one time row.

    Thread/value layout:
    - CTA tile is ``(row_tile, pair_tile)``, flattened onto ``num_threads``.
    - pair index is the innermost coordinate for coalesced interleaved-pair
      loads and stores.

    Predication:
    - only row-tail and pair-tail guards are needed.
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
        mC: cute.Tensor,
        mB: cute.Tensor,
        mBHead: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
    ) -> None:
        BHC = cute.size(mC.shape[0])
        L = cute.size(mC.shape[1])
        pair_cols = cute.size(mC.shape[2]) // 2
        grid_x = cute.ceil_div(pair_cols, self.pair_tile)
        grid_y = cute.ceil_div(L, self.row_tile)
        self.kernel(
            mC,
            mB,
            mBHead,
            mPhase,
            mKRaw,
            mQ,
            mKprev,
            mKcurr,
        ).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mC: cute.Tensor,
        mB: cute.Tensor,
        mBHead: cute.Tensor,
        mPhase: cute.Tensor,
        mKRaw: cute.Tensor,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pair_tile_idx, row_tile_idx, bhc = cute.arch.block_idx()

        pair_local = tidx % self.pair_tile
        row_local = tidx // self.pair_tile
        row = row_tile_idx * self.row_tile + row_local
        pair_idx = pair_tile_idx * self.pair_tile + pair_local
        pair_cols = mC.shape[2] // 2

        if cute.elem_less(row, mC.shape[1]) and cute.elem_less(pair_idx, pair_cols):
            col = pair_idx * 2
            pr = cutlass.Float32(mPhase[bhc, row, 0])
            pi = cutlass.Float32(mPhase[bhc, row, 1])

            cr = cutlass.Float32(mC[bhc, row, col + 0])
            ci = cutlass.Float32(mC[bhc, row, col + 1])
            qr = cr * pr + ci * pi
            qi = cr * pi - ci * pr
            mQ[bhc, row, 0, col + 0] = qr.to(mQ.element_type)
            mQ[bhc, row, 0, col + 1] = qi.to(mQ.element_type)

            bcr = cutlass.Float32(mB[bhc, row, col + 0])
            bci = cutlass.Float32(mB[bhc, row, col + 1])
            kcr_raw = cutlass.Float32(mKRaw[bhc, row, 1, 0])
            kci_raw = cutlass.Float32(mKRaw[bhc, row, 1, 1])
            apr = pr * kcr_raw + pi * kci_raw
            api = pi * kcr_raw - pr * kci_raw
            kcr = apr * bcr + api * bci
            kci = api * bcr - apr * bci
            mKcurr[bhc, row, 0, col + 0] = kcr.to(mKcurr.element_type)
            mKcurr[bhc, row, 0, col + 1] = kci.to(mKcurr.element_type)

            bpr = cutlass.Float32(0.0)
            bpi = cutlass.Float32(0.0)
            if row == cutlass.Int32(0):
                bpr = cutlass.Float32(mBHead[bhc, col + 0])
                bpi = cutlass.Float32(mBHead[bhc, col + 1])
            else:
                bpr = cutlass.Float32(mB[bhc, row - 1, col + 0])
                bpi = cutlass.Float32(mB[bhc, row - 1, col + 1])
            kpr_raw = cutlass.Float32(mKRaw[bhc, row, 0, 0])
            kpi_raw = cutlass.Float32(mKRaw[bhc, row, 0, 1])
            ar = pr * kpr_raw + pi * kpi_raw
            ai = pi * kpr_raw - pr * kpi_raw
            kpr = ar * bpr + ai * bpi
            kpi = ai * bpr - ar * bpi
            mKprev[bhc, row, 0, col + 0] = kpr.to(mKprev.element_type)
            mKprev[bhc, row, 0, col + 1] = kpi.to(mKprev.element_type)


class _ChunkScanPackP:
    """Packs raw ``U`` plus boundary values into ``Vprev/Vcurr``."""

    def __init__(self, *, feat_tile: int, num_threads: int = 128) -> None:
        self.feat_tile = int(feat_tile)
        self.num_threads = int(num_threads)
        if self.feat_tile <= 0 or self.num_threads % self.feat_tile != 0:
            raise ValueError("num_threads must be divisible by feat_tile.")
        self.row_tile = self.num_threads // self.feat_tile

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,
        mUHead: cute.Tensor,
        mVprev: cute.Tensor,
        mVcurr: cute.Tensor,
    ) -> None:
        BHC = cute.size(mU.shape[0])
        L = cute.size(mU.shape[1])
        P = cute.size(mU.shape[2])
        grid_x = cute.ceil_div(P, self.feat_tile)
        grid_y = cute.ceil_div(L, self.row_tile)
        self.kernel(mU, mUHead, mVprev, mVcurr).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mU: cute.Tensor,
        mUHead: cute.Tensor,
        mVprev: cute.Tensor,
        mVcurr: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        feat_tile_idx, row_tile_idx, bhc = cute.arch.block_idx()

        feat_local = tidx % self.feat_tile
        row_local = tidx // self.feat_tile
        row = row_tile_idx * self.row_tile + row_local
        feat = feat_tile_idx * self.feat_tile + feat_local

        if cute.elem_less(row, mU.shape[1]) and cute.elem_less(feat, mU.shape[2]):
            curr = cutlass.Float32(mU[bhc, row, feat])
            mVcurr[bhc, row, 0, feat] = curr.to(mVcurr.element_type)
            prev = mUHead[bhc, feat]
            if row == cutlass.Int32(0):
                prev = cutlass.Float32(mUHead[bhc, feat])
            else:
                prev = cutlass.Float32(mU[bhc, row - 1, feat])
            mVprev[bhc, row, 0, feat] = prev.to(mVprev.element_type)


class _ChunkScanPackZ0:
    """Packs conjugated ``chunk_starts`` into the inner-kernel ``Z0`` layout."""

    def __init__(self, *, pair_tile: int, num_threads: int = 128) -> None:
        self.pair_tile = int(pair_tile)
        self.num_threads = int(num_threads)
        if self.pair_tile <= 0 or self.num_threads % self.pair_tile != 0:
            raise ValueError("num_threads must be divisible by pair_tile.")
        self.p_tile = self.num_threads // self.pair_tile

    @cute.jit
    def __call__(self, mZ0Raw: cute.Tensor, mZ0: cute.Tensor) -> None:
        BHC = cute.size(mZ0Raw.shape[0])
        P = cute.size(mZ0Raw.shape[1])
        pair_cols = cute.size(mZ0Raw.shape[2]) // 2
        grid_x = cute.ceil_div(pair_cols, self.pair_tile)
        grid_y = cute.ceil_div(P, self.p_tile)
        self.kernel(mZ0Raw, mZ0).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(self, mZ0Raw: cute.Tensor, mZ0: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pair_tile_idx, p_tile_idx, bhc = cute.arch.block_idx()

        pair_local = tidx % self.pair_tile
        p_local = tidx // self.pair_tile
        p_idx = p_tile_idx * self.p_tile + p_local
        pair_idx = pair_tile_idx * self.pair_tile + pair_local
        pair_cols = mZ0Raw.shape[2] // 2

        if cute.elem_less(p_idx, mZ0Raw.shape[1]) and cute.elem_less(
            pair_idx, pair_cols
        ):
            col = pair_idx * 2
            zr = cutlass.Float32(mZ0Raw[bhc, p_idx, col + 0])
            zi = -cutlass.Float32(mZ0Raw[bhc, p_idx, col + 1])
            mZ0[bhc, p_idx, 0, col + 0] = zr.to(mZ0.element_type)
            mZ0[bhc, p_idx, 0, col + 1] = zi.to(mZ0.element_type)


class _ChunkScanPhasePrefix:
    """Builds unit-complex phase prefixes from raw ``M`` pairs.

    Logical shape:
    - ``M_raw``: ``(BHC, L, 2)``
    - ``phase``: ``(BHC, L, 2)``

    Thread layout:
    - one linear thread owns one ``bhc`` sequence and scans its ``L`` steps
      sequentially. ``L`` is tiny, so this avoids the slow host-side complex
      ``cumprod`` without introducing extra synchronization.
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(self, mMRaw: cute.Tensor, mPhase: cute.Tensor) -> None:
        BHC = cute.size(mMRaw.shape[0])
        self.kernel(mMRaw, mPhase).launch(
            grid=[cute.ceil_div(BHC, self.num_threads), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(self, mMRaw: cute.Tensor, mPhase: cute.Tensor) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        bhc = bidx * self.num_threads + tidx
        if cute.elem_less(bhc, mMRaw.shape[0]):
            pr = cutlass.Float32(1.0)
            pi = cutlass.Float32(0.0)
            eps = cutlass.Float32(1.0e-20)
            for t in cutlass.range(mMRaw.shape[1], unroll=1):
                mr = cutlass.Float32(mMRaw[bhc, t, 0])
                mi = cutlass.Float32(mMRaw[bhc, t, 1])
                inv_mag = cutlass.Float32(cute.math.rsqrt(mr * mr + mi * mi + eps))
                ur = mr * inv_mag
                ui = mi * inv_mag
                tr = pr * ur - pi * ui
                ti = pr * ui + pi * ur
                mPhase[bhc, t, 0] = tr
                mPhase[bhc, t, 1] = ti
                pr = tr
                pi = ti


def _get_compiled_phase(M_raw: torch.Tensor, phase: torch.Tensor) -> object:
    device_index = 0 if M_raw.device.index is None else int(M_raw.device.index)
    key: _CompiledPhaseKey = (
        device_index,
        M_raw.dtype,
        tuple(int(x) for x in M_raw.shape),
    )
    compiled = _COMPILED_PHASE.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanPhasePrefix()
    compiled = cute.compile(
        kernel,
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
    )
    _COMPILED_PHASE[key] = compiled
    return compiled


def _get_compiled_pack_d(
    C_raw: torch.Tensor,
    B_raw: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_head: torch.Tensor,
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Kcurr: torch.Tensor,
) -> object:
    device_index = 0 if C_raw.device.index is None else int(C_raw.device.index)
    key: _CompiledPackDKey = (
        device_index,
        C_raw.dtype,
        Q.dtype,
        tuple(int(x) for x in C_raw.shape),
    )
    compiled = _COMPILED_PACK_D.get(key)
    if compiled is not None:
        return compiled

    pair_tile = _choose_pair_tile(int(C_raw.shape[-1]))
    kernel = _ChunkScanPackD(pair_tile=pair_tile)
    compiled = cute.compile(
        kernel,
        from_dlpack(C_raw, assumed_align=C_raw.element_size()),
        from_dlpack(B_raw, assumed_align=B_raw.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(Kcurr, assumed_align=Kcurr.element_size()),
    )
    _COMPILED_PACK_D[key] = compiled
    return compiled


def _get_compiled_pack_p(
    U_raw: torch.Tensor,
    U_head: torch.Tensor,
    Vprev: torch.Tensor,
    Vcurr: torch.Tensor,
) -> object:
    device_index = 0 if U_raw.device.index is None else int(U_raw.device.index)
    key: _CompiledPackPKey = (
        device_index,
        U_raw.dtype,
        Vprev.dtype,
        tuple(int(x) for x in U_raw.shape),
    )
    compiled = _COMPILED_PACK_P.get(key)
    if compiled is not None:
        return compiled

    feat_tile = _choose_feat_tile(int(U_raw.shape[-1]))
    kernel = _ChunkScanPackP(feat_tile=feat_tile)
    compiled = cute.compile(
        kernel,
        from_dlpack(U_raw, assumed_align=U_raw.element_size()),
        from_dlpack(U_head, assumed_align=U_head.element_size()),
        from_dlpack(Vprev, assumed_align=Vprev.element_size()),
        from_dlpack(Vcurr, assumed_align=Vcurr.element_size()),
    )
    _COMPILED_PACK_P[key] = compiled
    return compiled


def _get_compiled_pack_z0(Z0_raw: torch.Tensor, Z0: torch.Tensor) -> object:
    device_index = 0 if Z0_raw.device.index is None else int(Z0_raw.device.index)
    key: _CompiledPackZ0Key = (
        device_index,
        Z0_raw.dtype,
        Z0.dtype,
        tuple(int(x) for x in Z0_raw.shape),
    )
    compiled = _COMPILED_PACK_Z0.get(key)
    if compiled is not None:
        return compiled

    pair_tile = _choose_pair_tile(int(Z0_raw.shape[-1]))
    kernel = _ChunkScanPackZ0(pair_tile=pair_tile)
    compiled = cute.compile(
        kernel,
        from_dlpack(Z0_raw, assumed_align=Z0_raw.element_size()),
        from_dlpack(Z0, assumed_align=Z0.element_size()),
    )
    _COMPILED_PACK_Z0[key] = compiled
    return compiled


def _pack_chunk_scan_inner_inputs(
    U_raw: torch.Tensor,
    B_raw: torch.Tensor,
    C_raw: torch.Tensor,
    M_raw: torch.Tensor,
    K_raw: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0_raw: torch.Tensor,
    U_head: torch.Tensor,
    B_head: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    BHC, L, D = map(int, C_raw.shape)
    P = int(U_raw.shape[-1])
    tc_dtype = _tc_dtype_for_inputs(C_raw.dtype)

    phase = torch.empty((BHC, L, 2), device=M_raw.device, dtype=torch.float32)
    Q = torch.empty((BHC, L, 1, D), device=C_raw.device, dtype=tc_dtype)
    Kprev = torch.empty_like(Q)
    Kcurr = torch.empty_like(Q)
    Vprev = torch.empty((BHC, L, 1, P), device=U_raw.device, dtype=tc_dtype)
    Vcurr = torch.empty_like(Vprev)
    Z0 = torch.empty((BHC, P, 1, D), device=Z0_raw.device, dtype=tc_dtype)

    compiled_phase = _get_compiled_phase(M_raw, phase)
    compiled_d = _get_compiled_pack_d(
        C_raw, B_raw, phase, K_raw, B_head, Q, Kprev, Kcurr
    )
    compiled_p = _get_compiled_pack_p(U_raw, U_head, Vprev, Vcurr)
    compiled_z0 = _get_compiled_pack_z0(Z0_raw, Z0)

    compiled_phase(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
    )
    compiled_d(
        from_dlpack(C_raw, assumed_align=C_raw.element_size()),
        from_dlpack(B_raw, assumed_align=B_raw.element_size()),
        from_dlpack(B_head, assumed_align=B_head.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(K_raw, assumed_align=K_raw.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Kprev, assumed_align=Kprev.element_size()),
        from_dlpack(Kcurr, assumed_align=Kcurr.element_size()),
    )
    compiled_p(
        from_dlpack(U_raw, assumed_align=U_raw.element_size()),
        from_dlpack(U_head, assumed_align=U_head.element_size()),
        from_dlpack(Vprev, assumed_align=Vprev.element_size()),
        from_dlpack(Vcurr, assumed_align=Vcurr.element_size()),
    )
    compiled_z0(
        from_dlpack(Z0_raw, assumed_align=Z0_raw.element_size()),
        from_dlpack(Z0, assumed_align=Z0.element_size()),
    )
    return Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0


class ChunkScanFused32AmpereTc(ChunkScanInnerAmpereTc):
    """Specialized fused chunk-scan kernel for ``chunk_size == 32``."""

    @staticmethod
    def can_implement(
        dtype: type[cutlass.Numeric],
        out_dtype: type[cutlass.Numeric],
        cfg: ChunkScanConfig,
    ) -> bool:
        if cfg.L != 32:
            return False
        if not ChunkScanInnerAmpereTc.can_implement(dtype, out_dtype, cfg):
            return False

        smem_capacity = utils.get_smem_capacity_in_bytes("sm_80")
        in_bytes = dtype.width // 8
        out_bytes = out_dtype.width // 8
        Dp = cfg.D_padded
        Pp = cfg.P_padded
        m = cfg.m_block_size
        n = cfg.n_block_size

        compute_smem = 0
        compute_smem += m * Dp * in_bytes
        compute_smem += max(Pp, n) * Dp * in_bytes
        compute_smem += n * Pp * in_bytes
        compute_smem += (m + n) * 4
        compute_smem += 6 * cfg.L * 4
        out_smem = (m * Pp) * out_bytes
        return max(compute_smem, out_smem) <= smem_capacity

    @cute.jit
    def __call__(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mLoghalf: cute.Tensor,
        mPhase: cute.Tensor,
        mCoeffPrev: cute.Tensor,
        mCoeffCurr: cute.Tensor,
        mZ0: cute.Tensor,
        mUHead: cute.Tensor,
        mBHead: cute.Tensor,
        mOut: cute.Tensor,
    ):
        if cutlass.const_expr(
            mU.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("Fused chunk_scan expects Float16/BFloat16 raw U/B/C.")
        if cutlass.const_expr(
            not (
                mU.element_type
                == mB.element_type
                == mC.element_type
                == mUHead.element_type
                == mBHead.element_type
            )
        ):
            raise TypeError("U/B/C/UHead/BHead must share the same dtype.")
        if cutlass.const_expr(
            mLoghalf.element_type != cutlass.Float32
            or mPhase.element_type != cutlass.Float32
            or mCoeffPrev.element_type != cutlass.Float32
            or mCoeffCurr.element_type != cutlass.Float32
            or mZ0.element_type != cutlass.Float32
        ):
            raise TypeError("Fused chunk_scan metadata and Z0 must be Float32.")
        if cutlass.const_expr(
            mOut.element_type
            not in (cutlass.Float16, cutlass.BFloat16, cutlass.Float32)
        ):
            raise TypeError("Out must be Float16/BFloat16/Float32.")

        Dp = self.cfg.D_padded
        Pp = self.cfg.P_padded
        m = self.cfg.m_block_size
        n = self.cfg.n_block_size
        num_threads = self.cfg.num_threads

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

        universal_copy_bits = 128
        in_dtype = mU.element_type
        out_dtype = mOut.element_type
        store_elems = universal_copy_bits // out_dtype.width
        tP_shape_dim_1 = sP_layout_atom.outer.shape[1] // (
            universal_copy_bits // in_dtype.width
        )
        tP_layout = cute.make_layout(
            (num_threads // tP_shape_dim_1, tP_shape_dim_1), stride=(tP_shape_dim_1, 1)
        )
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy_out,
            tP_layout,
            cute.make_layout((1, store_elems)),
        )
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(in_dtype, cutlass.Float32, (16, 8, 16)),
            (num_threads // 32, 1, 1),
            permutation_mnk=(num_threads // 32 * 16, 16, 16),
        )

        in_bytes = in_dtype.width // 8
        out_bytes = out_dtype.width // 8
        compute_smem = (
            cute.cosize(sQ_layout) * in_bytes
            + cute.cosize(sB_layout) * in_bytes
            + cute.cosize(sV_layout) * in_bytes
            + (m + n) * 4
            + 6 * self.cfg.L * 4
        )
        out_smem = cute.cosize(sO_layout) * out_bytes
        smem_size = cutlass.max(compute_smem, out_smem)

        self.kernel(
            mU,
            mB,
            mC,
            mLoghalf,
            mPhase,
            mCoeffPrev,
            mCoeffCurr,
            mZ0,
            mUHead,
            mBHead,
            mOut,
            sQ_layout,
            sB_layout,
            sK_layout,
            sZ_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma,
        ).launch(
            grid=(1, cute.size(mU.shape[0]), 1),
            block=[num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mU: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mLoghalf: cute.Tensor,
        mPhase: cute.Tensor,
        mCoeffPrev: cute.Tensor,
        mCoeffCurr: cute.Tensor,
        mZ0: cute.Tensor,
        mUHead: cute.Tensor,
        mBHead: cute.Tensor,
        mOut: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sZ_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, bhc, _ = cute.arch.block_idx()

        D = self.cfg.D
        Dp = self.cfg.D_padded
        P = self.cfg.P
        Pp = self.cfg.P_padded
        m = self.cfg.m_block_size
        n = self.cfg.n_block_size
        L = self.cfg.L
        pair_cols = Dp // 2

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mU.element_type, sQ_layout, 16)
        sB = smem.allocate_tensor(mU.element_type, sB_layout, 16)
        sV = smem.allocate_tensor(mU.element_type, sV_layout, 16)
        sK = cute.make_tensor(sB.iterator, sK_layout)
        sZ = cute.make_tensor(sB.iterator, sZ_layout)
        sLpQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((m,)), 4)
        sLpK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((n,)), 4)
        sPr = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sPi = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sCpr = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sCpi = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sCcr = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sCci = smem.allocate_tensor(cutlass.Float32, cute.make_layout((L,)), 4)
        sVt = cute.composition(sV, cute.make_layout((Pp, n), stride=(n, 1)))

        if tidx < L:
            lp = cutlass.Float32(mLoghalf[bhc, tidx])
            sLpK[tidx] = cute.math.exp2(-lp * TWO_LOG2_E, fastmath=True)
            sPr[tidx] = cutlass.Float32(mPhase[bhc, tidx, 0])
            sPi[tidx] = cutlass.Float32(mPhase[bhc, tidx, 1])
            sCpr[tidx] = cutlass.Float32(mCoeffPrev[bhc, tidx, 0])
            sCpi[tidx] = cutlass.Float32(mCoeffPrev[bhc, tidx, 1])
            sCcr[tidx] = cutlass.Float32(mCoeffCurr[bhc, tidx, 0])
            sCci[tidx] = cutlass.Float32(mCoeffCurr[bhc, tidx, 1])
        if tidx < m:
            lp = cutlass.Float32(0.0)
            if tidx < L:
                lp = cutlass.Float32(mLoghalf[bhc, tidx])
            sLpQ[tidx] = cute.math.exp2(lp * TWO_LOG2_E, fastmath=True)
        cute.arch.barrier()

        idx = cutlass.Int32(tidx)
        total_q = cutlass.Int32(m * pair_cols)
        while cute.elem_less(idx, total_q):
            rr = idx // pair_cols
            vv = idx - rr * pair_cols
            col = vv * 2
            qr = cutlass.Float32(0.0)
            qi = cutlass.Float32(0.0)
            if cute.elem_less(rr, L) and cute.elem_less(col + 1, D):
                cr = cutlass.Float32(mC[bhc, rr, 0, col + 0])
                ci = cutlass.Float32(mC[bhc, rr, 0, col + 1])
                pr = cutlass.Float32(sPr[rr])
                pi = cutlass.Float32(sPi[rr])
                qr = cr * pr + ci * pi
                qi = cr * pi - ci * pr
            sQ[rr, col + 0] = qr.to(mU.element_type)
            sQ[rr, col + 1] = qi.to(mU.element_type)
            idx = idx + self.cfg.num_threads

        idx = cutlass.Int32(tidx)
        total_z = cutlass.Int32(Pp * pair_cols)
        while cute.elem_less(idx, total_z):
            rr = idx // pair_cols
            vv = idx - rr * pair_cols
            col = vv * 2
            zr = cutlass.Float32(0.0)
            zi = cutlass.Float32(0.0)
            if cute.elem_less(rr, P) and cute.elem_less(col + 1, D):
                zr = cutlass.Float32(mZ0[bhc, rr, 0, col + 0])
                zi = -cutlass.Float32(mZ0[bhc, rr, 0, col + 1])
            sZ[rr, col + 0] = zr.to(mU.element_type)
            sZ[rr, col + 1] = zi.to(mU.element_type)
            idx = idx + self.cfg.num_threads
        cute.arch.barrier()

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tSrZ = thr_mma.make_fragment_B(thr_mma.partition_B(sZ))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))

        acc_shape_O = thr_mma.partition_shape_C((m, Pp))
        acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
        acc_O.fill(0.0)
        acc_shape_S = thr_mma.partition_shape_C((m, n))

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

        cute.copy(smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
        cute.copy(smem_tiled_copy_K, tSsZ[None, None, 0], tSrZ_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            k_next = (kk + 1) % cute.size(tSsQ.shape[2])
            cute.copy(
                smem_tiled_copy_Q,
                tSsQ[None, None, k_next],
                tSrQ_view[None, None, k_next],
            )
            cute.copy(
                smem_tiled_copy_K,
                tSsZ[None, None, k_next],
                tSrZ_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma, acc_O, tSrQ[None, None, kk], tSrZ[None, None, kk], acc_O
            )

        mcO = cute.make_identity_tensor(mOut.layout.shape)
        cO = cute.local_tile(mcO[bhc, None, 0, None], (m, Pp), (0, 0))
        tOcO = thr_mma.partition_C(cO)
        tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(acc_O_mn.shape[0])):
            row_idx = tOcO_mn[r, 0][1]
            if cute.elem_less(row_idx, L):
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * cutlass.Float32(
                    sLpQ[row_idx]
                )

        mcS = cute.make_identity_tensor(
            (mU.shape[0], mU.shape[1], mU.shape[2], mB.shape[1])
        )
        idx_local = cutlass.Int32(tidx)
        total_k = cutlass.Int32(n * pair_cols)
        while cute.elem_less(idx_local, total_k):
            rr = idx_local // pair_cols
            vv = idx_local - rr * pair_cols
            col = vv * 2
            kr = cutlass.Float32(0.0)
            ki = cutlass.Float32(0.0)
            if cute.elem_less(rr, L) and cute.elem_less(col + 1, D):
                br = cutlass.select_(
                    rr == cutlass.Int32(0),
                    cutlass.Float32(mBHead[bhc, col + 0]),
                    cutlass.Float32(mB[bhc, rr - 1, 0, col + 0]),
                )
                bi = cutlass.select_(
                    rr == cutlass.Int32(0),
                    cutlass.Float32(mBHead[bhc, col + 1]),
                    cutlass.Float32(mB[bhc, rr - 1, 0, col + 1]),
                )
                ar = cutlass.Float32(sCpr[rr])
                ai = cutlass.Float32(sCpi[rr])
                kr = ar * br + ai * bi
                ki = ai * br - ar * bi
            sK[rr, col + 0] = kr.to(mU.element_type)
            sK[rr, col + 1] = ki.to(mU.element_type)
            idx_local = idx_local + self.cfg.num_threads

        idx_local = cutlass.Int32(tidx)
        total_v = cutlass.Int32(n * Pp)
        while cute.elem_less(idx_local, total_v):
            rr = idx_local // Pp
            cc = idx_local - rr * Pp
            val = cutlass.Float32(0.0)
            if cute.elem_less(rr, L) and cute.elem_less(cc, P):
                val = cutlass.select_(
                    rr == cutlass.Int32(0),
                    cutlass.Float32(mUHead[bhc, cc]),
                    cutlass.Float32(mU[bhc, rr - 1, 0, cc]),
                )
            sV[rr, cc] = val.to(mU.element_type)
            idx_local = idx_local + self.cfg.num_threads
        cute.arch.barrier()

        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        cute.copy(smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
        cute.copy(smem_tiled_copy_K, tSsK[None, None, 0], tSrK_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            k_next = (kk + 1) % cute.size(tSsQ.shape[2])
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
                tSrQ[None, None, kk],
                tSrK[None, None, kk],
                acc_S,
            )

        cS = cute.local_tile(mcS[bhc, None, 0, None], (m, n), (0, 0))
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)
        self._scale_and_mask_scores(
            acc_S,
            tScS_mn,
            sLpQ,
            sLpK,
            m_tile_start=cutlass.Int32(0),
            n_tile_start=cutlass.Int32(0),
            seqlen=cutlass.Int32(L),
        )

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
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        cute.copy(smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (kk + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_tiled_copy_V,
                tOsVt[None, None, k_next],
                tOrVt_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrS[None, None, kk],
                tOrVt[None, None, kk],
                acc_O,
            )

        idx_local = cutlass.Int32(tidx)
        while cute.elem_less(idx_local, total_k):
            rr = idx_local // pair_cols
            vv = idx_local - rr * pair_cols
            col = vv * 2
            kr = cutlass.Float32(0.0)
            ki = cutlass.Float32(0.0)
            if cute.elem_less(rr, L) and cute.elem_less(col + 1, D):
                br = cutlass.Float32(mB[bhc, rr, 0, col + 0])
                bi = cutlass.Float32(mB[bhc, rr, 0, col + 1])
                ar = cutlass.Float32(sCcr[rr])
                ai = cutlass.Float32(sCci[rr])
                kr = ar * br + ai * bi
                ki = ai * br - ar * bi
            sK[rr, col + 0] = kr.to(mU.element_type)
            sK[rr, col + 1] = ki.to(mU.element_type)
            idx_local = idx_local + self.cfg.num_threads

        idx_local = cutlass.Int32(tidx)
        while cute.elem_less(idx_local, total_v):
            rr = idx_local // Pp
            cc = idx_local - rr * Pp
            val = cutlass.Float32(0.0)
            if cute.elem_less(rr, L) and cute.elem_less(cc, P):
                val = cutlass.Float32(mU[bhc, rr, 0, cc])
            sV[rr, cc] = val.to(mU.element_type)
            idx_local = idx_local + self.cfg.num_threads
        cute.arch.barrier()

        acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
        acc_S.fill(0.0)
        cute.copy(smem_tiled_copy_Q, tSsQ[None, None, 0], tSrQ_view[None, None, 0])
        cute.copy(smem_tiled_copy_K, tSsK[None, None, 0], tSrK_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            k_next = (kk + 1) % cute.size(tSsQ.shape[2])
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
                tSrQ[None, None, kk],
                tSrK[None, None, kk],
                acc_S,
            )

        cS = cute.local_tile(mcS[bhc, None, 0, None], (m, n), (0, 0))
        tScS = thr_mma.partition_C(cS)
        tScS_mn = self._make_acc_tensor_mn_view(tScS)
        self._scale_and_mask_scores(
            acc_S,
            tScS_mn,
            sLpQ,
            sLpK,
            m_tile_start=cutlass.Int32(0),
            n_tile_start=cutlass.Int32(0),
            seqlen=cutlass.Int32(L),
        )

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
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
        cute.copy(smem_tiled_copy_V, tOsVt[None, None, 0], tOrVt_view[None, None, 0])
        for kk in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (kk + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_tiled_copy_V,
                tOsVt[None, None, k_next],
                tOrVt_view[None, None, k_next],
            )
            cute.gemm(
                tiled_mma,
                acc_O,
                tOrS[None, None, kk],
                tOrVt[None, None, kk],
                acc_O,
            )

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

        gO = cute.local_tile(mOut[bhc, None, 0, None], (m, Pp), (0, 0))
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_rmem_tensor_like(tOgO, mOut.element_type)
        cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

        mcOut = cute.make_identity_tensor(mOut.layout.shape)
        cOut = cute.local_tile(mcOut[bhc, None, 0, None], (m, Pp), (0, 0))
        tOcOut = gmem_thr_copy_O.partition_D(cOut)
        tOpOut = cute.make_rmem_tensor(
            cute.make_layout(
                (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                stride=(tOgO.shape[2], 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tOpOut.shape[0]):
            for rest_n in cutlass.range_constexpr(cute.size(tOpOut.shape[2])):
                tOpOut[rest_v, 0, rest_n] = cute.elem_less(
                    tOcOut[(0, rest_v), 0, rest_n][3], mOut.layout.shape[3]
                )
        for rest_m in cutlass.range_constexpr(cute.size(tOpOut.shape[1])):
            if cute.elem_less(tOcOut[0, rest_m, 0][1], mOut.layout.shape[1]):
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None],
                    tOgO[None, rest_m, None],
                    pred=tOpOut[None, rest_m, None],
                )


def _prepare_chunk_scan_fused_operands(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    torch.dtype,
]:
    batch_size, n_heads, T, N, P, n_chunks = _validate_chunk_scan_inputs(
        U, M, K, B, C, chunk_starts, B_prev, U_prev, int(U.shape[2]), chunk_size
    )
    if chunk_size != 32 or (T % 32) != 0:
        raise ValueError(
            "The fused chunk_scan path currently requires chunk_size=32 and T divisible by 32."
        )
    D = 2 * N

    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype, chunk_starts.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The fused CuTe chunk_scan path supports only float32 reference "
            f"compute. Got compute_dtype={rdtype}."
        )
    tc_dtype = _tc_dtype_for_inputs(U.dtype)
    cplx_dtype = _complex_dtype_from_real(rdtype)

    U_r = U if U.dtype == rdtype else U.to(dtype=rdtype)
    B_r = B if B.dtype == rdtype else B.to(dtype=rdtype)
    C_r = C if C.dtype == rdtype else C.to(dtype=rdtype)
    M_r = M if M.dtype == rdtype else M.to(dtype=rdtype)
    K_r = K if K.dtype == rdtype else K.to(dtype=rdtype)

    BHC = batch_size * n_heads * n_chunks
    L = 32
    U_tc = U_r.to(dtype=tc_dtype).reshape(BHC, L, 1, P).contiguous()
    B_tc = B_r.to(dtype=tc_dtype).reshape(BHC, L, 1, D).contiguous()
    C_tc = C_r.to(dtype=tc_dtype).reshape(BHC, L, 1, D).contiguous()

    m = (
        _to_complex_scalar(M_r, name="M")
        .to(dtype=cplx_dtype)
        .reshape(batch_size, n_heads, n_chunks, L)
    )
    mag = torch.abs(m)
    if bool((mag == 0).any()):
        raise ValueError(
            "M must be strictly nonzero in fused chunk_scan paths. Exact-zero "
            "transitions are outside the SLinOSS operating region."
        )
    phase_prefix = torch.cumprod(m / mag, dim=-1)
    loghalf = (0.5 * torch.cumsum(torch.log(mag), dim=-1)).reshape(BHC, L).contiguous()

    k = (
        _to_complex_taps(K_r, name="K")
        .to(dtype=cplx_dtype)
        .reshape(batch_size, n_heads, n_chunks, L, 2)
    )
    coeff_prev = (phase_prefix * torch.conj(k[..., 0])).reshape(BHC, L)
    coeff_curr = (phase_prefix * torch.conj(k[..., 1])).reshape(BHC, L)
    phase = torch.view_as_real(phase_prefix).reshape(BHC, L, 2).contiguous()
    coeff_prev = torch.view_as_real(coeff_prev).reshape(BHC, L, 2).contiguous()
    coeff_curr = torch.view_as_real(coeff_curr).reshape(BHC, L, 2).contiguous()

    z0 = chunk_starts.to(dtype=rdtype).reshape(BHC, P, 1, D).contiguous()

    U_head = torch.empty(
        (batch_size, n_heads, n_chunks, P), device=U.device, dtype=rdtype
    )
    B_head = torch.empty(
        (batch_size, n_heads, n_chunks, D), device=U.device, dtype=rdtype
    )
    if U_prev is None:
        U_head[:, :, 0] = 0.0
    else:
        U_head[:, :, 0] = U_prev.to(dtype=rdtype)
    if B_prev is None:
        B_head[:, :, 0] = 0.0
    else:
        B_head[:, :, 0] = B_prev.to(dtype=rdtype)
    if n_chunks > 1:
        U_blk = U_r.reshape(batch_size, n_heads, n_chunks, L, P)
        B_blk = B_r.reshape(batch_size, n_heads, n_chunks, L, D)
        U_head[:, :, 1:] = U_blk[:, :, :-1, -1, :]
        B_head[:, :, 1:] = B_blk[:, :, :-1, -1, :]
    U_head = U_head.reshape(BHC, P).to(dtype=tc_dtype).contiguous()
    B_head = B_head.reshape(BHC, D).to(dtype=tc_dtype).contiguous()

    return (
        U_tc,
        B_tc,
        C_tc,
        loghalf,
        phase,
        coeff_prev,
        coeff_curr,
        z0,
        U_head,
        B_head,
        batch_size,
        n_heads,
        T,
        odtype,
    )


def _get_compiled_chunk_scan_fused(
    U_tc: torch.Tensor,
    B_tc: torch.Tensor,
    C_tc: torch.Tensor,
    loghalf: torch.Tensor,
    phase: torch.Tensor,
    coeff_prev: torch.Tensor,
    coeff_curr: torch.Tensor,
    z0: torch.Tensor,
    U_head: torch.Tensor,
    B_head: torch.Tensor,
    out: torch.Tensor,
) -> object:
    device_index = 0 if U_tc.device.index is None else int(U_tc.device.index)
    key: _CompiledFusedKey = (
        device_index,
        U_tc.dtype,
        out.dtype,
        tuple(int(x) for x in U_tc.shape),
        int(out.shape[-1]),
    )
    compiled = _COMPILED_CHUNK_SCAN_FUSED.get(key)
    if compiled is not None:
        return compiled

    cfg = ChunkScanConfig(
        D=int(B_tc.shape[-1]),
        P=int(U_tc.shape[-1]),
        L=32,
        m_block_size=128,
        n_block_size=32,
        num_threads=128,
    )
    cutlass_in_dtype = _torch_to_cutlass_dtype(U_tc.dtype)
    cutlass_out_dtype = _torch_to_cutlass_dtype(out.dtype)
    if not ChunkScanFused32AmpereTc.can_implement(
        cutlass_in_dtype, cutlass_out_dtype, cfg
    ):
        raise ValueError(
            "ChunkScanFused32AmpereTc cannot implement this shape/dtype on SM80."
        )

    kernel = ChunkScanFused32AmpereTc(cfg)
    compiled = cute.compile(
        kernel,
        from_dlpack(U_tc.contiguous(), assumed_align=16),
        from_dlpack(B_tc.contiguous(), assumed_align=16),
        from_dlpack(C_tc.contiguous(), assumed_align=16),
        from_dlpack(loghalf.contiguous(), assumed_align=16),
        from_dlpack(phase.contiguous(), assumed_align=16),
        from_dlpack(coeff_prev.contiguous(), assumed_align=16),
        from_dlpack(coeff_curr.contiguous(), assumed_align=16),
        from_dlpack(z0.contiguous(), assumed_align=16),
        from_dlpack(U_head.contiguous(), assumed_align=16),
        from_dlpack(B_head.contiguous(), assumed_align=16),
        from_dlpack(out.contiguous(), assumed_align=16),
    )
    _COMPILED_CHUNK_SCAN_FUSED[key] = compiled
    return compiled


def chunk_scan_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes within-chunk outputs with the CuTe chunk-scan kernels."""
    (
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix,
        Z0_raw,
        U_head,
        B_head,
        batch_size,
        n_heads,
        T,
        T_pad,
        odtype,
    ) = _prepare_chunk_scan_small_operands(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )
    Q, Kprev, Vprev, Kcurr, Vcurr, logprefix, Z0 = _pack_chunk_scan_inner_inputs(
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix,
        Z0_raw,
        U_head,
        B_head,
    )

    out = torch.empty(
        (Q.shape[0], Q.shape[1], 1, Vprev.shape[-1]),
        device=U.device,
        dtype=odtype,
    )
    compiled = _get_compiled_chunk_scan(
        Q, Kprev, Vprev, Kcurr, Vcurr, logprefix, Z0, out
    )

    mQ = from_dlpack(Q, assumed_align=16)
    mKprev = from_dlpack(Kprev, assumed_align=16)
    mVprev = from_dlpack(Vprev, assumed_align=16)
    mKcurr = from_dlpack(Kcurr, assumed_align=16)
    mVcurr = from_dlpack(Vcurr, assumed_align=16)
    mLogp = from_dlpack(logprefix, assumed_align=16)
    mZ0 = from_dlpack(Z0, assumed_align=16)
    mOut = from_dlpack(out, assumed_align=16)
    compiled(mQ, mKprev, mVprev, mKcurr, mVcurr, mLogp, mZ0, mOut)

    L = int(chunk_size)
    n_chunks = int(T_pad // L)
    Y = out.squeeze(2).reshape(batch_size, n_heads, n_chunks, L, out.shape[-1])
    Y = Y.reshape(batch_size, n_heads, T_pad, out.shape[-1])
    return Y[:, :, :T].contiguous()


__all__ = ["chunk_scan_cute"]
