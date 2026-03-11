"""CuTe backward slice for ``chunk_scan`` gradients into ``chunk_starts``.

There are two surfaces here:

- ``chunk_scan_bwd_dz0_packed_cute`` is the real hot-path kernel used by
  autograd. It consumes the cached forward-packed ``Q`` and ``logprefix_half``
  tensors directly and computes ``dZ0`` with tensor cores and fp32
  accumulation.
- ``chunk_scan_bwd_dz0_cute`` is the legacy public wrapper over raw ``M/C``.
  It keeps the SO(2) prep in Torch, but delegates the dense contraction to the
  packed kernel instead of the old generic fp32 GEMM path.

Numerical contract
------------------
The packed kernel uses fp16/bf16 transport with fp32 accumulation. The only
per-time scale is ``row_scale[t] = exp(2 * logprefix_half[t])``. It is formed
in-kernel from the forward-cached logprefix metadata, so there is no reciprocal
factorization and no ``0 * inf`` hazard on this path.
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import _tc_dtype_for_inputs
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _chunked_transition_prefix_parts,
    _complex_dtype_from_real,
    _pack_complex_pairs,
    _pad_time_full,
    _resolve_dtypes,
    _to_complex_scalar,
)


_CompiledPackedKey = tuple[
    int,
    torch.dtype,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_PACKED_DZ0: dict[_CompiledPackedKey, object] = {}


def _dz0_packed_config(P: int, D: int, L: int) -> tuple[tuple[int, int, int], int]:
    bM = 64 if P >= 64 else 32
    if D >= 96 and D % 32 == 0:
        bN = 96
    else:
        bN = 64 if D >= 64 else 32
    if L % 64 == 0:
        bK = 64
    elif L % 32 == 0:
        bK = 32
    else:
        bK = 16
    return (bM, bN, bK), 2


class _ChunkScanBwdDZ0PackedAmpere:
    """Ampere tensor-core kernel for packed ``dZ0``.

    Logical tensors
    ---------------
    - ``mDOut``: ``(P, L, BHC)``, low-precision transport
    - ``mQ``: ``(D, L, BHC)``, low-precision packed off-term features
    - ``mLogprefix``: ``(L, BHC)``, fp32
    - ``mDZ0``: ``(P, D, BHC)``, fp32

    Layout / launch contract
    ------------------------
    - One CTA owns one ``(P_tile, D_tile, bhc)`` output tile.
    - ``K`` is the chunk-time axis ``L`` and is tiled by ``bK``.
    - Global loads use ``cp.async`` into swizzled shared tiles.
    - Inputs are moved to registers with ``ldmatrix`` and multiplied on tensor
      cores with fp32 accumulation.
    - ``row_scale[t] = exp(2 * logprefix_half[t])`` is computed once per time
      step and applied in shared before the MMA mainloop.

    Correctness invariants
    ----------------------
    - ``L`` must be divisible by ``bK``.
    - ``L`` must fit within the CTA thread count so the per-time scale scratch
      is one thread per time step.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        *,
        chunk_size: int,
        cta_tiler: tuple[int, int, int] = (64, 64, 16),
        atom_layout_mnk: tuple[int, int, int] = (2, 2, 1),
        num_stages: int = 2,
    ) -> None:
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.Float32

        self.L = int(chunk_size)
        self.cta_tiler = cta_tiler
        self.num_stages = int(num_stages)
        self.atom_layout_mnk = atom_layout_mnk

        self.bM, self.bN, self.bK = map(int, self.cta_tiler)
        if self.L % self.bK != 0:
            raise ValueError("chunk_size must be divisible by bK for this kernel.")
        k_tile_count = self.L // self.bK
        if (self.num_stages - 1) > k_tile_count:
            raise ValueError(
                "num_stages too large for chunk_size/bK (insufficient K tiles)."
            )
        if self.num_stages < 2:
            raise ValueError("num_stages must be >= 2.")

        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        atomM, atomN, atomK = self.atom_layout_mnk

        self.num_threads = atomM * atomN * atomK * 32
        if self.L > self.num_threads:
            raise ValueError("chunk_size too large for this CTA thread count.")

        if self.bM % (atomM * mmaM) != 0:
            raise ValueError("bM must be divisible by MMA instruction shape.")
        if self.bN % (atomN * mmaN * 2) != 0:
            raise ValueError("bN must be divisible by MMA instruction shape.")
        if atomK != 1:
            raise ValueError("atom_layout_mnk K must be 1.")
        if self.bK % mmaK != 0:
            raise ValueError("bK must be divisible by MMA instruction shape.")

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
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

    def _make_gmem_tiled_copy_AB(
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

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            value_layout = cute.make_layout((1, copy_elems))

            best_tm = None
            best_tn = None
            for tm in range(1, self.num_threads + 1):
                if self.num_threads % tm != 0:
                    continue
                tn = self.num_threads // tm
                tile_m = tm
                tile_n = tn * copy_elems
                if (self.bM % tile_m) != 0:
                    continue
                if (self.bN % tile_n) != 0:
                    continue
                if best_tm is None or tile_n > (best_tn * copy_elems):
                    best_tm = tm
                    best_tn = tn
            if best_tm is None:
                shape_dim_1 = cute.size(self.bN) // copy_elems
                thread_layout = cute.make_layout(
                    (self.num_threads // shape_dim_1, shape_dim_1),
                    stride=(shape_dim_1, 1),
                )
            else:
                thread_layout = cute.make_layout(
                    (best_tm, best_tn), stride=(best_tn, 1)
                )
            return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

        value_layout = cute.make_layout((copy_elems, 1))
        shape_dim_0 = (int(self.bM) + int(copy_elems) - 1) // int(copy_elems)
        if shape_dim_0 > self.num_threads:
            raise ValueError("bM too large for vectorized col-major store.")
        tm = None
        for cand in range(shape_dim_0, self.num_threads + 1):
            if self.num_threads % cand == 0:
                tm = cand
                break
        if tm is None:
            raise ValueError(
                "Internal error: failed to find divisor for col-major store."
            )
        thread_layout = cute.make_layout((tm, self.num_threads // tm), stride=(1, tm))
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    @cute.jit(preprocess=True)
    def __call__(
        self,
        mDOut: cute.Tensor,  # (P, L, BHC)   fp16/bf16
        mQ: cute.Tensor,  # (D, L, BHC)   fp16/bf16
        mLogprefix: cute.Tensor,  # (BHC, L)   fp32
        mDZ0: cute.Tensor,  # (P, D, BHC)  fp32
    ) -> None:
        self.a_major_mode = utils.LayoutEnum.from_tensor(mDOut)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mQ)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mDZ0)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mDOut.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mQ.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))

        smem_size_AB = cute.size_in_bytes(
            mDOut.element_type, sA_layout
        ) + cute.size_in_bytes(mQ.element_type, sB_layout)
        smem_size_C = cute.size_in_bytes(self.c_dtype, sC_layout)
        extra_bytes = self.L * 4  # row_scale[t]
        smem_needed = max(smem_size_AB, smem_size_C) + extra_bytes

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mDOut.element_type,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            mDOut.element_type,
            self.a_major_mode,
            ab_copy_bits,
            tile_m=self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy,
            mQ.element_type,
            self.b_major_mode,
            ab_copy_bits,
            tile_m=self.bN,
        )

        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=128,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, self.c_dtype, self.c_major_mode, 128
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        grid_dim = cute.ceil_div(mDZ0.shape, (self.bM, self.bN, 1))
        grid_z = cute.size(mDZ0.shape[2])

        self.kernel(
            mDOut,
            mQ,
            mLogprefix,
            mDZ0,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
        ).launch(
            grid=(cute.size(grid_dim[0]), cute.size(grid_dim[1]), grid_z),
            block=[self.num_threads, 1, 1],
            smem=smem_needed,
        )

    @cute.kernel(preprocess=True)
    def kernel(
        self,
        mDOut: cute.Tensor,
        mQ: cute.Tensor,
        mLogprefix: cute.Tensor,
        mDZ0: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        gC = cute.local_tile(
            mDZ0[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None),
        )

        smem = cutlass.utils.SmemAllocator()
        sRow = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.L,), stride=(1,)), 4
        )
        sA = smem.allocate_tensor(mDOut.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mQ.element_type, sB_layout, 16)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=cutlass.Float32), sC_layout
        )

        if tidx < cutlass.Int32(self.L):
            lp = cutlass.Float32(mLogprefix[bidz, tidx].to(cutlass.Float32))
            sRow[tidx] = cute.math.exp2(
                lp * cutlass.Float32(2.0 * math.log2(math.e)),
                fastmath=True,
            )
        cute.arch.barrier()

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)

        tAsA = thr_copy_A.partition_D(sA)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_ep = thr_copy_C.partition_S(sC)
        tCgC_ep = thr_copy_C.partition_D(gC)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mDOut.element_type,
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
            ),
            mQ.element_type,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ld_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ld_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy = thr_copy_ld_A.partition_S(sA)
        tCrA_copy = thr_copy_ld_A.retile(tCrA)
        tCsB_copy = thr_copy_ld_B.partition_S(sB)
        tCrB_copy = thr_copy_ld_B.retile(tCrB)

        mcA = cute.make_identity_tensor(mDOut.layout.shape)
        mcB = cute.make_identity_tensor(mQ.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1),
        )
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mDOut.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mQ.shape[0]
                )

        gA = cute.local_tile(
            mDOut[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mQ[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(None, 1, 1),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        tAgA = thr_copy_A.partition_S(gA)
        tBgB = thr_copy_B.partition_S(gB)

        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        k_tile_count = self.L // self.bK
        k_tile_index = cutlass.Int32(0)
        for st in range(self.num_stages - 1):
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, st],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, st],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        num_k_block = cute.size(tCrA, mode=[2])

        for kt in range(k_tile_count):
            smem_pipe_read = kt % self.num_stages
            smem_pipe_write = (kt + (self.num_stages - 1)) % self.num_stages
            cute.arch.cp_async_wait_group(self.num_stages - 2)
            cute.arch.sync_threads()

            k_tile_offset = kt * self.bK
            iters_a = (self.bM * self.bK) // self.num_threads
            for it in range(iters_a):
                idx = tidx + cutlass.Int32(it * self.num_threads)
                mm = idx // self.bK
                kk = idx - mm * self.bK
                t_global = k_tile_offset + kk
                a = cutlass.Float32(sA[mm, kk, smem_pipe_read])
                a = a * cutlass.Float32(sRow[t_global])
                sA[mm, kk, smem_pipe_read] = a.to(mDOut.element_type)

            cute.arch.sync_threads()

            tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy[None, None, 0])
            for kb in cutlass.range(num_k_block, unroll_full=True):
                kb_next = (kb + 1) % num_k_block
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, kb_next],
                    tCrA_copy[None, None, kb_next],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, kb_next],
                    tCrB_copy[None, None, kb_next],
                )
                cute.gemm(
                    tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC
                )

            next_tile = kt + (self.num_stages - 1)
            if next_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, smem_pipe_write],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, smem_pipe_write],
                    pred=tBpB,
                )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.autovec_copy(tCrC, tCsC)
        cute.arch.sync_threads()
        tCrC_ep = cute.make_rmem_tensor_like(tCsC_ep, self.c_dtype)
        cute.autovec_copy(tCsC_ep, tCrC_ep)

        ceilM, ceilN, _ = cute.ceil_div(mDZ0.shape, (self.bM, self.bN, 1))
        mcC = cute.make_identity_tensor(
            (cute.size(ceilM) * self.bM, cute.size(ceilN) * self.bN, 1)
        )
        cC = cute.local_tile(
            mcC[None, None, bidz],
            tiler=self.cta_tiler,
            coord=(bidx, bidy, None),
            proj=(1, 1, None),
        )
        tCcC = thr_copy_C.partition_S(cC)
        tCpC = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tCgC_ep.shape[0][1],
                    cute.size(tCgC_ep, mode=[1]),
                    cute.size(tCgC_ep, mode=[2]),
                ),
                stride=(
                    cute.size(tCgC_ep, mode=[1]) * cute.size(tCgC_ep, mode=[2]),
                    cute.size(tCgC_ep, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tCpC.shape[0]):
            for n in range(tCpC.shape[2]):
                n_ok = cute.elem_less(tCcC[(0, rest_v), 0, n][1], mDZ0.shape[1])
                for m in range(tCpC.shape[1]):
                    m_ok = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mDZ0.shape[0])
                    tCpC[rest_v, m, n] = n_ok & m_ok
        for n in range(tCpC.shape[2]):
            cute.copy(
                tiled_copy_C,
                tCrC_ep[None, None, n],
                tCgC_ep[None, None, n],
                pred=tCpC[None, None, n],
            )


def _get_compiled_chunk_scan_bwd_dz0_packed(
    mDOut: torch.Tensor,
    mQ: torch.Tensor,
    mLogprefix: torch.Tensor,
    mDZ0: torch.Tensor,
) -> object:
    device_index = 0 if mDOut.device.index is None else int(mDOut.device.index)
    key: _CompiledPackedKey = (
        device_index,
        mDOut.dtype,
        tuple(int(x) for x in mDOut.shape),
        tuple(int(x) for x in mQ.shape),
        tuple(int(x) for x in mLogprefix.shape),
        tuple(int(x) for x in mDZ0.shape),
    )
    compiled = _COMPILED_PACKED_DZ0.get(key)
    if compiled is not None:
        return compiled

    cta_tiler, num_stages = _dz0_packed_config(
        int(mDOut.shape[0]), int(mQ.shape[0]), int(mDOut.shape[1])
    )
    kernel = _ChunkScanBwdDZ0PackedAmpere(
        cutlass.BFloat16 if mDOut.dtype == torch.bfloat16 else cutlass.Float16,
        chunk_size=int(mDOut.shape[1]),
        cta_tiler=cta_tiler,
        num_stages=num_stages,
    )
    compiled = cute.compile(
        kernel,
        from_dlpack(mDOut, assumed_align=16),
        from_dlpack(mQ, assumed_align=16),
        from_dlpack(mLogprefix, assumed_align=mLogprefix.element_size()),
        from_dlpack(mDZ0, assumed_align=max(mDZ0.element_size(), 16)),
    )
    _COMPILED_PACKED_DZ0[key] = compiled
    return compiled


def chunk_scan_bwd_dz0_packed_cute(
    Q: torch.Tensor,
    logprefix_half: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Packed tensor-core ``dZ0`` for the hot backward contract.

    Inputs:
    - ``Q``: ``(BHC, L, 1, D)``, cached packed off-term features
    - ``logprefix_half``: ``(BHC, L)``, forward-cached half log-prefix
    - ``d_out_flat``: ``(BHC, L, P)``, upstream row gradients

    Output:
    - ``dZ0``: ``(BHC, P, D)`` in fp32, matching the packed-conjugated ``Z0``
      contract used by the forward CuTe inner kernel.
    """
    if Q.device.type != "cuda" or logprefix_half.device.type != "cuda" or d_out_flat.device.type != "cuda":
        raise ValueError("Packed CuTe dZ0 requires CUDA tensors.")
    if not (Q.is_contiguous() and logprefix_half.is_contiguous() and d_out_flat.is_contiguous()):
        raise ValueError("Packed CuTe dZ0 expects contiguous tensors.")
    if Q.ndim != 4 or Q.shape[2] != 1:
        raise ValueError(f"Q must be (BHC, L, 1, D). Got {tuple(Q.shape)}.")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if d_out_flat.ndim != 3 or d_out_flat.shape[:2] != Q.shape[:2]:
        raise ValueError(
            "d_out_flat must be (BHC, L, P) matching Q. Got "
            f"{tuple(d_out_flat.shape)} for Q shape {tuple(Q.shape)}."
        )

    BHC, L, _, D = map(int, Q.shape)
    P = int(d_out_flat.shape[-1])
    tc_dtype = _tc_dtype_for_inputs(Q.dtype)

    Q_tc = Q.squeeze(2)
    if Q_tc.dtype != tc_dtype:
        Q_tc = Q_tc.to(dtype=tc_dtype)
    d_out_tc = d_out_flat.to(dtype=tc_dtype)
    mDOut = d_out_tc.permute(2, 1, 0)
    mQ = Q_tc.permute(2, 1, 0)
    dZ0 = torch.empty((BHC, P, D), device=Q.device, dtype=torch.float32)
    mDZ0 = dZ0.permute(1, 2, 0)

    compiled = _get_compiled_chunk_scan_bwd_dz0_packed(
        mDOut, mQ, logprefix_half, mDZ0
    )
    compiled(
        from_dlpack(mDOut, assumed_align=16),
        from_dlpack(mQ, assumed_align=16),
        from_dlpack(
            logprefix_half, assumed_align=logprefix_half.element_size()
        ),
        from_dlpack(mDZ0, assumed_align=max(mDZ0.element_size(), 16)),
    )
    return dZ0


def chunk_scan_bwd_dz0_cute(
    M: torch.Tensor,
    C: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Legacy public ``d_chunk_starts`` wrapper over the packed TC kernel.

    This wrapper keeps the raw ``M/C`` SO(2) prep for the public test/debug
    surface. The dense work is still delegated to the packed TC kernel. The
    prepared ``Q`` already folds the prefix factors into the complex features,
    so the packed kernel runs with zero ``logprefix_half`` here.
    """
    if (
        M.device.type != "cuda"
        or C.device.type != "cuda"
        or d_out.device.type != "cuda"
    ):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")
    if M.ndim != 4 or M.shape[-1] != 2:
        raise ValueError(f"M must be (batch, heads, T, 2). Got {tuple(M.shape)}.")
    if C.ndim != 4 or d_out.ndim != 4:
        raise ValueError("C and d_out must be rank-4 tensors.")
    if M.shape[:3] != C.shape[:3] or M.shape[:3] != d_out.shape[:3]:
        raise ValueError("Leading (batch, heads, T) dims of M/C/d_out must match.")
    if C.shape[-1] % 2 != 0:
        raise ValueError(f"C must have an even 2N trailing dim. Got {tuple(C.shape)}.")
    if not (M.is_contiguous() and C.is_contiguous() and d_out.is_contiguous()):
        raise ValueError("M, C, and d_out must be contiguous.")

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[M.dtype, C.dtype, d_out.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_scan dZ0 path supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    batch_size, n_heads, T, P = map(int, d_out.shape)
    D = int(C.shape[-1])
    N = D // 2
    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = C.device

    dummy_u = torch.empty((batch_size, n_heads, T, P), device=device, dtype=rdtype)
    dummy_k = torch.empty((batch_size, n_heads, T, 2, 2), device=device, dtype=rdtype)
    dummy_b = torch.empty((batch_size, n_heads, T, D), device=device, dtype=rdtype)
    _, M_f, _, _, C_f, T_pad, n_chunks = _pad_time_full(
        dummy_u, M, dummy_k, dummy_b, C, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)
    BHC = batch_size * n_heads * n_chunks

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    c_conj = torch.conj(_as_complex_pairs(C_f, name="C").to(dtype=cplx_dtype))

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    c_blk = c_conj.reshape(batch_size, n_heads, n_chunks, L, N)
    _, _, prefix = _chunked_transition_prefix_parts(m_blk)

    q_off = torch.conj(c_blk * prefix.unsqueeze(-1)).resolve_conj()
    q_packed = _pack_complex_pairs(
        q_off.reshape(BHC, L, N),
        real_dtype=rdtype,
    ).unsqueeze(2)

    d_out_blk = d_out.to(dtype=rdtype)
    if T_pad != T:
        pad = T_pad - T
        d_out_blk = torch.cat(
            [
                d_out_blk,
                torch.zeros((batch_size, n_heads, pad, P), device=device, dtype=rdtype),
            ],
            dim=2,
        )
    d_out_blk = d_out_blk.reshape(BHC, L, P).contiguous()
    zero_logprefix = torch.zeros((BHC, L), device=device, dtype=torch.float32)
    d_chunk_starts = chunk_scan_bwd_dz0_packed_cute(
        q_packed.contiguous(),
        zero_logprefix,
        d_out_blk,
    )
    return d_chunk_starts.reshape(batch_size, n_heads, n_chunks, P, D).contiguous()


__all__ = ["chunk_scan_bwd_dz0_packed_cute", "chunk_scan_bwd_dz0_cute"]
