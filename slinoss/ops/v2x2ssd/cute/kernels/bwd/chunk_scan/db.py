"""CuTe backward ``db`` workhorse for the ``v2x2ssd`` chunk-scan stage.

This file is intentionally written in the same overall shape as the
``v3x3ssd`` ``chunk_scan`` ``db`` workhorse:

- one monolithic stage-native kernel class
- direct public-stage inputs and outputs
- in-kernel prefix reconstruction from raw stage inputs
- reverse chunk sweep over ``n_tile``
- two FA2-style tensor-core products per ``m_tile``:
  current-``V`` and shifted-``V``
- direct ownership of public ``dB`` / ``dB_prev``

The adaptation is only in the scan algebra:

- quaternion transport becomes unit-complex transport
- 3-vectors become interleaved complex pairs
- FOH vector taps become complex-scalar taps
- ``trans`` becomes raw packed ``M``
"""

import cutlass
import cutlass.cute as cute

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    apply_complex_tap_adjoint,
    complex_mul,
    conj_mul_phase,
)


class ChunkScanBwdDBAmpere:
    """Ampere tensor-core kernel for ``chunk_scan`` backward key grads.

    Computes, per chunk (BHC):
      - ``dB`` and ``dB_prev``
      - placeholder zeroed ``dU`` / ``dU_prev`` / ``dlogprefix`` / ``dM`` outputs

    Notes:
      - This kernel rebuilds the chunk-local prefix phase and magnitude from
        raw packed ``M`` inside the CTA.
      - The dense work follows the same shape as the ``v3`` workhorse:
        ``dS = dY @ V`` for both current and shifted values, then
        ``dK = dS @ scaled(Q)`` with tensor-core GEMMs.
      - Public ``dB`` rows are owned directly by this kernel. The contribution
        from ``Kprev[t + 1]`` is carried across reverse ``n_tile`` iterations.
    """

    def __init__(self, dtype, *, chunk_size, D, P, num_threads=128):
        self.ab_dtype = dtype
        self.acc_dtype = cutlass.Float32
        self.L = int(chunk_size)
        self.D = int(D)
        self.P = int(P)
        self.kv_tile = 32
        if self.L % self.kv_tile != 0:
            raise ValueError("chunk_size must be a multiple of 32.")
        self.num_threads = int(num_threads)
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        self.num_warps = self.num_threads // 32
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
        self.do_du = False
        self.do_db = True
        self.do_dlp = False
        self.db_enable_curr = True
        self.db_enable_prev = True
        self.db_enable_epilogue = True

    @property
    def D_padded(self) -> int:
        return (self.D + 31) // 32 * 32

    def _d_stage_size(self) -> int:
        """Shared-memory D slice width used by the backward DB kernel."""
        Dp = self.D_padded
        if Dp <= 64:
            return Dp
        return 64

    def _d_stage_count(self) -> int:
        Ds = self._d_stage_size()
        return (self.D_padded + Ds - 1) // Ds

    @property
    def P_padded(self) -> int:
        return (self.P + 31) // 32 * 32

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
        mDLogp: cute.Tensor,
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
        if cutlass.const_expr(mDLogp.element_type != cutlass.Float32):
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
        if cutlass.const_expr(mDLogp.shape[1] != self.L):
            raise ValueError("dlogprefix must be (BHC, L).")
        if cutlass.const_expr(
            mDMprev.shape[1] != self.L
            or mDMprev.shape[2] != 2
            or mDMcurr.shape[1] != self.L
            or mDMcurr.shape[2] != 2
        ):
            raise ValueError("dM buffers must be (BHC, L, 2).")

        Dp = self.D_padded
        Ds = self._d_stage_size()
        Pp = self.P_padded
        kv_tile = self.kv_tile
        p_tile = 32
        if cutlass.const_expr(Pp % p_tile != 0):
            raise ValueError("P must be padded to a multiple of 32.")

        smem_k_block_size_D = 64 if Ds % 64 == 0 else 32
        swizzle_bits_D = 3 if smem_k_block_size_D >= 32 else 2
        sD_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_D, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_D), stride=(smem_k_block_size_D, 1)),
        )
        sQ_layout = cute.tile_to_shape(sD_layout_atom, (kv_tile, Ds), (0, 1))
        sK_layout = cute.tile_to_shape(sD_layout_atom, (kv_tile, Ds), (0, 1))

        smem_k_block_size_P = p_tile
        swizzle_bits_P = 3
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sDY_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))
        sV_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))

        smem_k_block_size_blk = kv_tile
        swizzle_bits_blk = 3
        sBlk_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_blk, 3, 3),
            0,
            cute.make_layout(
                (8, smem_k_block_size_blk), stride=(smem_k_block_size_blk, 1)
            ),
        )
        sBlk_layout = cute.tile_to_shape(sBlk_layout_atom, (kv_tile, kv_tile), (0, 1))

        universal_copy_bits = 128
        async_elems_in = universal_copy_bits // mU.element_type.width
        atom_async_copy_in = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mU.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        tP_shape_dim_1 = sP_layout_atom.outer.shape[1] // async_elems_in
        tP_layout = cute.make_layout(
            (self.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_elems_in
        tD_layout = cute.make_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        gmem_tiled_copy_P = cute.make_tiled_copy_tv(
            atom_async_copy_in, tP_layout, v_in_layout
        )
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mU.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        v_out_layout = cute.make_layout((1, async_elems_in))
        gmem_tiled_store_D = cute.make_tiled_copy_tv(
            atom_universal_copy_out, tD_layout, v_out_layout
        )

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sQ_layout)], 16
            ]
            sDY: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sDY_layout)], 16
            ]
            sK_tile: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sK_layout)], 16
            ]
            sV_tile: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sV_layout)], 16
            ]
            sS_blk: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sBlk_layout)], 16
            ]
            sS_cache: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, cute.cosize(sBlk_layout)], 16
            ]
            s_phase: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(cute.make_layout((self.L, 2), stride=(2, 1))),
                ],
                16,
            ]
            s_tap_prev: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(cute.make_layout((self.L, 2), stride=(2, 1))),
                ],
                16,
            ]
            s_tap_curr: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(cute.make_layout((kv_tile, 2), stride=(2, 1))),
                ],
                16,
            ]
            sDB_carry: cute.struct.Align[cute.struct.MemRange[mU.element_type, Dp], 8]
            s_dlp: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.L], 4]
            s_row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L], 4
            ]
            s_inv_row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L], 4
            ]
            s_dM_curr: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(cute.make_layout((kv_tile, 2), stride=(2, 1))),
                ],
                16,
            ]
            s_dM_prev: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    cute.cosize(cute.make_layout((kv_tile, 2), stride=(2, 1))),
                ],
                16,
            ]

        grid_z = cute.size(mB.shape[0])
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
            mDLogp,
            mDMprev,
            mDMcurr,
            sQ_layout,
            sDY_layout,
            sK_layout,
            sV_layout,
            sBlk_layout,
            gmem_tiled_copy_D,
            gmem_tiled_copy_P,
            gmem_tiled_store_D,
            tiled_mma,
            SharedStorage,
        ).launch(grid=(1, 1, grid_z), block=[self.num_threads, 1, 1])

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
        mDLogp: cute.Tensor,
        mDMprev: cute.Tensor,
        mDMcurr: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sDY_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sBlk_layout: cute.ComposedLayout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_store_D: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        Dp = self.D_padded
        Ds = self._d_stage_size()
        d_stage_count = self._d_stage_count()
        Pp = self.P_padded
        p_tile = 32
        n_p_tiles = Pp // p_tile
        nvec_stage = Ds // 2
        BH = mU_prev0.shape[0]
        BHC = mB.shape[0]
        n_chunks = BHC // BH
        bh = bidz // n_chunks
        chunk = bidz - bh * n_chunks
        kv_tile = int(self.kv_tile)
        n_tiles = int(self.L // kv_tile)
        single_tile_cache = n_tiles == 1

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ0 = storage.sQ.get_tensor(sQ_layout)
        sDY0 = storage.sDY.get_tensor(sDY_layout)
        sS_prev = cute.make_tensor(sDY0.iterator.align(16), sBlk_layout)
        sK_tile = storage.sK_tile.get_tensor(sK_layout)
        sV_tile = storage.sV_tile.get_tensor(sV_layout)
        sS_blk = storage.sS_blk.get_tensor(sBlk_layout)
        sS_cache = storage.sS_cache.get_tensor(sBlk_layout)
        sDY1 = cute.make_tensor(sK_tile.iterator.align(16), sDY_layout)
        sV1 = cute.make_tensor(sS_blk.iterator.align(16), sV_layout)
        s_phase = storage.s_phase.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_prev = storage.s_tap_prev.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_curr = storage.s_tap_curr.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )
        sDB_carry = storage.sDB_carry.get_tensor(cute.make_layout((Dp,), stride=(1,)))
        s_dlp = storage.s_dlp.get_tensor(cute.make_layout((self.L,), stride=(1,)))
        s_row_scale = storage.s_row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )
        s_inv_row_scale = storage.s_inv_row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )
        s_dM_curr = storage.s_dM_curr.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )
        s_dM_prev = storage.s_dM_prev.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        mr = cutlass.Float32(1.0)
        mi = cutlass.Float32(0.0)
        if tidx < cutlass.Int32(self.L):
            mr = cutlass.Float32(mM[bidz, tidx, 0])
            mi = cutlass.Float32(mM[bidz, tidx, 1])
        pred_t = tidx < cutlass.Int32(self.L)
        mr = cutlass.Float32(cutlass.select_(pred_t, mr, cutlass.Float32(1.0)))
        mi = cutlass.Float32(cutlass.select_(pred_t, mi, cutlass.Float32(0.0)))
        eps = cutlass.Float32(1.0e-20)
        mag2 = cutlass.Float32(mr * mr + mi * mi + eps)
        inv_mag = cutlass.Float32(cute.math.rsqrt(mag2))
        ur = mr * inv_mag
        ui = mi * inv_mag
        logp = cutlass.Float32(
            cute.math.log2(mag2, fastmath=False) * cutlass.Float32(0.25 / LOG2_E)
        )

        for offset in (1, 2, 4, 8, 16):
            other_log = cute.arch.shuffle_sync_up(
                logp, offset=offset, mask=-1, mask_and_clamp=0
            )
            opr = cute.arch.shuffle_sync_up(
                ur, offset=offset, mask=-1, mask_and_clamp=0
            )
            opi = cute.arch.shuffle_sync_up(
                ui, offset=offset, mask=-1, mask_and_clamp=0
            )
            pred = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(pred, logp + other_log, logp)
            nr, ni = complex_mul(ur, ui, opr, opi)
            ur = cutlass.select_(pred, nr, ur)
            ui = cutlass.select_(pred, ni, ui)

        if lane == cutlass.Int32(31):
            s_dlp[warp] = logp
            s_phase[warp, 0] = ur
            s_phase[warp, 1] = ui
        cute.arch.barrier()

        if warp == cutlass.Int32(0) and lane < cutlass.Int32(self.num_warps):
            wid = lane
            log0 = cutlass.Float32(s_dlp[0])
            p0r = cutlass.Float32(s_phase[0, 0])
            p0i = cutlass.Float32(s_phase[0, 1])
            log1 = cutlass.Float32(0.0)
            p1r = cutlass.Float32(1.0)
            p1i = cutlass.Float32(0.0)
            if cutlass.const_expr(self.num_warps > 1):
                log1 = cutlass.Float32(s_dlp[1])
                p1r = cutlass.Float32(s_phase[1, 0])
                p1i = cutlass.Float32(s_phase[1, 1])
            log2 = cutlass.Float32(0.0)
            p2r = cutlass.Float32(1.0)
            p2i = cutlass.Float32(0.0)
            if cutlass.const_expr(self.num_warps > 2):
                log2 = cutlass.Float32(s_dlp[2])
                p2r = cutlass.Float32(s_phase[2, 0])
                p2i = cutlass.Float32(s_phase[2, 1])
            off_log = cutlass.Float32(0.0)
            off_r = cutlass.Float32(1.0)
            off_i = cutlass.Float32(0.0)
            if wid == cutlass.Int32(1):
                off_log = log0
                off_r, off_i = p0r, p0i
            if wid == cutlass.Int32(2):
                off_log = log0 + log1
                off_r, off_i = complex_mul(p1r, p1i, p0r, p0i)
            if wid == cutlass.Int32(3):
                off_log = log0 + log1 + log2
                p10r, p10i = complex_mul(p1r, p1i, p0r, p0i)
                off_r, off_i = complex_mul(p2r, p2i, p10r, p10i)
            s_dlp[cutlass.Int32(self.num_warps) + wid] = off_log
            s_phase[cutlass.Int32(self.num_warps) + wid, 0] = off_r
            s_phase[cutlass.Int32(self.num_warps) + wid, 1] = off_i
        cute.arch.barrier()

        off_log = cutlass.Float32(s_dlp[cutlass.Int32(self.num_warps) + warp])
        off_r = cutlass.Float32(s_phase[cutlass.Int32(self.num_warps) + warp, 0])
        off_i = cutlass.Float32(s_phase[cutlass.Int32(self.num_warps) + warp, 1])
        logp = logp + off_log
        ur, ui = complex_mul(ur, ui, off_r, off_i)

        if tidx < cutlass.Int32(self.L):
            row_scale = cute.math.exp2(
                logp * cutlass.Float32(TWO_LOG2_E), fastmath=False
            )
            s_row_scale[tidx] = row_scale
            s_inv_row_scale[tidx] = cutlass.Float32(1.0) / row_scale
            s_phase[tidx, 0] = ur
            s_phase[tidx, 1] = ui
            s_tap_prev[tidx, 0] = cutlass.Float32(mK[bidz, tidx, 0, 0])
            s_tap_prev[tidx, 1] = cutlass.Float32(mK[bidz, tidx, 0, 1])
        cute.arch.barrier()

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
        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        gmem_thr_copy_P = gmem_tiled_copy_P.get_slice(tidx)
        mcS = cute.make_identity_tensor((mU.shape[0], self.L, mU.shape[2], self.L))
        mcKD = cute.make_identity_tensor((mU.shape[0], self.L, mU.shape[2], Dp))
        mcU = cute.make_identity_tensor(mU.layout.shape)
        mcC = cute.make_identity_tensor(mC.layout.shape)
        mcDY = cute.make_identity_tensor(mDOut.layout.shape)
        mcS_full = mcS[bidz, None, 0, None]
        mcKD_full = mcKD[bidz, None, 0, None]
        tQsQ0 = gmem_thr_copy_D.partition_D(sQ0)
        cC0 = cute.local_tile(mcC[bidz, None, 0, None], (kv_tile, Dp), (0, 0))
        tCc0 = gmem_thr_copy_D.partition_S(cC0)
        tDYs0 = gmem_thr_copy_P.partition_D(sDY0)
        tDYs1 = gmem_thr_copy_P.partition_D(sDY1)
        tVsV0 = gmem_thr_copy_P.partition_D(sV_tile)
        tVsV1 = gmem_thr_copy_P.partition_D(sV1)

        tSrS_blk = thr_mma.make_fragment_A(thr_mma.partition_A(sS_blk))
        tSsS_blk = thr_copy_A.partition_S(sS_blk)
        tSrS_blk_view = thr_copy_A.retile(tSrS_blk)
        tSrS_cache = thr_mma.make_fragment_A(thr_mma.partition_A(sS_cache))
        tSsS_cache = thr_copy_A.partition_S(sS_cache)
        tSrS_cache_view = thr_copy_A.retile(tSrS_cache)
        tSrS_prev = thr_mma.make_fragment_A(thr_mma.partition_A(sS_prev))
        tSsS_prev = thr_copy_A.partition_S(sS_prev)
        tSrS_prev_view = thr_copy_A.retile(tSrS_prev)
        sQt_layout = cute.make_layout((Ds, kv_tile), stride=(kv_tile, 1))

        sQt = cute.composition(sQ0, sQt_layout)
        tSrQt = thr_mma.make_fragment_B(thr_mma.partition_B(sQt))
        tSsQt = thr_copy_BT.partition_S(sQt)
        tSrQt_view = thr_copy_BT.retile(tSrQt)

        acc_shape_blk = thr_mma.partition_shape_C((kv_tile, kv_tile))
        acc_shape_tileD = thr_mma.partition_shape_C((kv_tile, Ds))

        total_pairs_stage = int(kv_tile * nvec_stage)
        iters_pairs_stage = int(
            (total_pairs_stage + self.num_threads - 1) // self.num_threads
        )
        iters_d = int((self.D + self.num_threads - 1) // self.num_threads)
        tQp = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tQsQ0.shape[0][1],
                    cute.size(tQsQ0, mode=[1]),
                    cute.size(tQsQ0, mode=[2]),
                ),
                stride=(cute.size(tQsQ0, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tQp.shape[0]):
            for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                tQp[rest_v, 0, rest_k] = cute.elem_less(
                    tCc0[(0, rest_v), 0, rest_k][3], mC.layout.shape[3]
                )
        tDYp = None
        if cutlass.const_expr(self.P != Pp):
            tDYp = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tDYs0.shape[0][1],
                        cute.size(tDYs0, mode=[1]),
                        cute.size(tDYs0, mode=[2]),
                    ),
                    stride=(cute.size(tDYs0, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
        mcDB = cute.make_identity_tensor(mDB.layout.shape)
        iters_ds = int((Ds + self.num_threads - 1) // self.num_threads)
        for it in range((Dp + self.num_threads - 1) // self.num_threads):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(Dp):
                sDB_carry[d] = cutlass.Float32(0.0).to(mU.element_type)
        cute.arch.barrier()

        for n_tile_rev in cutlass.range_constexpr(n_tiles):
            n_tile = (n_tiles - 1) - n_tile_rev
            n0 = n_tile * kv_tile
            m_tiles = n_tiles - n_tile

            if tidx < cutlass.Int32(kv_tile):
                t = cutlass.Int32(n0) + tidx
                s_tap_curr[tidx, 0] = cutlass.Float32(mK[bidz, t, 1, 0])
                s_tap_curr[tidx, 1] = cutlass.Float32(mK[bidz, t, 1, 1])
                s_dM_curr[tidx, 0] = cutlass.Float32(0.0)
                s_dM_curr[tidx, 1] = cutlass.Float32(0.0)
                s_dM_prev[tidx, 0] = cutlass.Float32(0.0)
                s_dM_prev[tidx, 1] = cutlass.Float32(0.0)
            cute.arch.barrier()

            for d_stage_idx in cutlass.range_constexpr(d_stage_count):
                d_col_base = cutlass.Int32(d_stage_idx * Ds)
                acc_dK_curr = cute.make_rmem_tensor(acc_shape_tileD, cutlass.Float32)
                acc_dK_curr.fill(0.0)
                acc_dK_prev = cute.make_rmem_tensor(acc_shape_tileD, cutlass.Float32)
                acc_dK_prev.fill(0.0)

                for rest_v in cutlass.range_constexpr(tQp.shape[0]):
                    for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                        g_col = (
                            cutlass.Int32(tCc0[(0, rest_v), 0, rest_k][3]) + d_col_base
                        )
                        tQp[rest_v, 0, rest_k] = cute.elem_less(
                            g_col, mC.layout.shape[3]
                        )

                for mi in cutlass.range_constexpr(m_tiles):
                    m_tile = n_tile + mi
                    m0 = m_tile * kv_tile

                    gC = cute.local_tile(
                        mC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, d_stage_idx)
                    )
                    tCg = gmem_thr_copy_D.partition_S(gC)
                    cC = cute.local_tile(
                        mcC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, d_stage_idx)
                    )
                    tCc = gmem_thr_copy_D.partition_S(cC)
                    for di in cutlass.range_constexpr(cute.size(tQsQ0.shape[1])):
                        if cute.elem_less(tCc[0, di, 0][1], mC.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_copy_D,
                                tCg[None, di, None],
                                tQsQ0[None, di, None],
                                pred=tQp[None, di, None],
                            )
                        else:
                            tQsQ0[None, di, None].fill(0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    for it in cutlass.range_constexpr(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = cutlass.Int32(m0) + t_local
                            if t < cutlass.Int32(self.L):
                                d_local = vv * 2
                                x = cutlass.Float32(
                                    sQ0[t_local, d_local + 0].to(cutlass.Float32)
                                )
                                y = cutlass.Float32(
                                    sQ0[t_local, d_local + 1].to(cutlass.Float32)
                                )
                                pr = cutlass.Float32(s_phase[t, 0])
                                pi = cutlass.Float32(s_phase[t, 1])
                                rx, ry = conj_mul_phase(x, y, pr, pi)
                                sQ0[t_local, d_local + 0] = rx.to(mU.element_type)
                                sQ0[t_local, d_local + 1] = ry.to(mU.element_type)
                    cute.arch.barrier()

                    cS_blk = cute.local_tile(
                        mcS_full, (kv_tile, kv_tile), (m_tile, n_tile)
                    )
                    tScS_blk = thr_mma.partition_C(cS_blk)
                    tScS_blk_mn = self._make_acc_tensor_mn_view(tScS_blk)
                    if (not single_tile_cache) or d_stage_idx == 0:
                        acc_blk_curr = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_blk_curr.fill(0.0)
                        p_tile_idx = cutlass.Int32(0)
                        gDY = cute.local_tile(
                            mDOut[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (m_tile, p_tile_idx),
                        )
                        tDYg = gmem_thr_copy_P.partition_S(gDY)
                        cDY = cute.local_tile(
                            mcDY[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (m_tile, p_tile_idx),
                        )
                        tDYc = gmem_thr_copy_P.partition_S(cDY)
                        if cutlass.const_expr(self.P == Pp):
                            cute.copy(gmem_tiled_copy_P, tDYg, tDYs0)
                        else:
                            tDYp_s = cute.make_rmem_tensor(tDYp.layout, cutlass.Boolean)
                            for rest_v in cutlass.range_constexpr(tDYp_s.shape[0]):
                                for rest_k in cutlass.range_constexpr(tDYp_s.shape[2]):
                                    tDYp_s[rest_v, 0, rest_k] = cute.elem_less(
                                        tDYc[(0, rest_v), 0, rest_k][3],
                                        mDOut.layout.shape[3],
                                    )
                            for vi in cutlass.range_constexpr(
                                cute.size(tDYs0.shape[1])
                            ):
                                if cute.elem_less(
                                    tDYc[0, vi, 0][1], mDOut.layout.shape[1]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_P,
                                        tDYg[None, vi, None],
                                        tDYs0[None, vi, None],
                                        pred=tDYp_s[None, vi, None],
                                    )
                                else:
                                    tDYs0[None, vi, None].fill(0)
                        gV = cute.local_tile(
                            mU[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (n_tile, p_tile_idx),
                        )
                        tVg = gmem_thr_copy_P.partition_S(gV)
                        cV = cute.local_tile(
                            mcU[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (n_tile, p_tile_idx),
                        )
                        tVc = gmem_thr_copy_P.partition_S(cV)
                        if cutlass.const_expr(self.P == Pp):
                            cute.copy(gmem_tiled_copy_P, tVg, tVsV0)
                        else:
                            tVp_s = cute.make_rmem_tensor(tDYp.layout, cutlass.Boolean)
                            for rest_v in cutlass.range_constexpr(tVp_s.shape[0]):
                                for rest_k in cutlass.range_constexpr(tVp_s.shape[2]):
                                    tVp_s[rest_v, 0, rest_k] = cute.elem_less(
                                        tVc[(0, rest_v), 0, rest_k][3],
                                        mU.layout.shape[3],
                                    )
                            for vi in cutlass.range_constexpr(
                                cute.size(tVsV0.shape[1])
                            ):
                                if cute.elem_less(tVc[0, vi, 0][1], mU.layout.shape[1]):
                                    cute.copy(
                                        gmem_tiled_copy_P,
                                        tVg[None, vi, None],
                                        tVsV0[None, vi, None],
                                        pred=tVp_s[None, vi, None],
                                    )
                                else:
                                    tVsV0[None, vi, None].fill(0)
                        cute.arch.cp_async_commit_group()
                        for p_iter in cutlass.range_constexpr(n_p_tiles):
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.barrier()
                            if cutlass.const_expr(p_iter + 1 < n_p_tiles):
                                p_next = cutlass.Int32(p_iter + 1)
                                gDY_next = cute.local_tile(
                                    mDOut[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (m_tile, p_next),
                                )
                                tDYg_next = gmem_thr_copy_P.partition_S(gDY_next)
                                cDY_next = cute.local_tile(
                                    mcDY[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (m_tile, p_next),
                                )
                                tDYc_next = gmem_thr_copy_P.partition_S(cDY_next)
                                gV_next = cute.local_tile(
                                    mU[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (n_tile, p_next),
                                )
                                tVg_next = gmem_thr_copy_P.partition_S(gV_next)
                                cV_next = cute.local_tile(
                                    mcU[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (n_tile, p_next),
                                )
                                tVc_next = gmem_thr_copy_P.partition_S(cV_next)
                                if cutlass.const_expr((p_iter & 1) == 0):
                                    if cutlass.const_expr(self.P == Pp):
                                        cute.copy(gmem_tiled_copy_P, tDYg_next, tDYs1)
                                        cute.copy(gmem_tiled_copy_P, tVg_next, tVsV1)
                                    else:
                                        tDYp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        tVp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        for rest_v in cutlass.range_constexpr(
                                            tDYp_next.shape[0]
                                        ):
                                            for rest_k in cutlass.range_constexpr(
                                                tDYp_next.shape[2]
                                            ):
                                                tDYp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tDYc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mDOut.layout.shape[3],
                                                    )
                                                )
                                                tVp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tVc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mU.layout.shape[3],
                                                    )
                                                )
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tDYs1.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tDYc_next[0, vi, 0][1],
                                                mDOut.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tDYg_next[None, vi, None],
                                                    tDYs1[None, vi, None],
                                                    pred=tDYp_next[None, vi, None],
                                                )
                                            else:
                                                tDYs1[None, vi, None].fill(0)
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tVsV1.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tVc_next[0, vi, 0][1],
                                                mU.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tVg_next[None, vi, None],
                                                    tVsV1[None, vi, None],
                                                    pred=tVp_next[None, vi, None],
                                                )
                                            else:
                                                tVsV1[None, vi, None].fill(0)
                                else:
                                    if cutlass.const_expr(self.P == Pp):
                                        cute.copy(gmem_tiled_copy_P, tDYg_next, tDYs0)
                                        cute.copy(gmem_tiled_copy_P, tVg_next, tVsV0)
                                    else:
                                        tDYp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        tVp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        for rest_v in cutlass.range_constexpr(
                                            tDYp_next.shape[0]
                                        ):
                                            for rest_k in cutlass.range_constexpr(
                                                tDYp_next.shape[2]
                                            ):
                                                tDYp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tDYc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mDOut.layout.shape[3],
                                                    )
                                                )
                                                tVp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tVc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mU.layout.shape[3],
                                                    )
                                                )
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tDYs0.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tDYc_next[0, vi, 0][1],
                                                mDOut.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tDYg_next[None, vi, None],
                                                    tDYs0[None, vi, None],
                                                    pred=tDYp_next[None, vi, None],
                                                )
                                            else:
                                                tDYs0[None, vi, None].fill(0)
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tVsV0.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tVc_next[0, vi, 0][1],
                                                mU.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tVg_next[None, vi, None],
                                                    tVsV0[None, vi, None],
                                                    pred=tVp_next[None, vi, None],
                                                )
                                            else:
                                                tVsV0[None, vi, None].fill(0)
                                cute.arch.cp_async_commit_group()
                            if cutlass.const_expr((p_iter & 1) == 0):
                                sDY_p = sDY0
                                sV_p = sV_tile
                            else:
                                sDY_p = sDY1
                                sV_p = sV1
                            tSrDY_m = thr_mma.make_fragment_A(
                                thr_mma.partition_A(sDY_p)
                            )
                            tSsDY_m = thr_copy_A.partition_S(sDY_p)
                            tSrDY_m_view = thr_copy_A.retile(tSrDY_m)
                            tSrV_p = thr_mma.make_fragment_B(thr_mma.partition_B(sV_p))
                            tSsV_p = thr_copy_B.partition_S(sV_p)
                            tSrV_p_view = thr_copy_B.retile(tSrV_p)
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsDY_m[None, None, 0],
                                tSrDY_m_view[None, None, 0],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tSsV_p[None, None, 0],
                                tSrV_p_view[None, None, 0],
                            )
                            for k in cutlass.range_constexpr(
                                cute.size(tSsDY_m.shape[2])
                            ):
                                k_next = (k + 1) % cute.size(tSsDY_m.shape[2])
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tSsDY_m[None, None, k_next],
                                    tSrDY_m_view[None, None, k_next],
                                )
                                cute.copy(
                                    smem_tiled_copy_B,
                                    tSsV_p[None, None, k_next],
                                    tSrV_p_view[None, None, k_next],
                                )
                                cute.gemm(
                                    tiled_mma,
                                    acc_blk_curr,
                                    tSrDY_m[None, None, k],
                                    tSrV_p[None, None, k],
                                    acc_blk_curr,
                                )
                        acc_blk_curr_mn = self._make_acc_tensor_mn_view(acc_blk_curr)
                        sS_curr = sS_cache if single_tile_cache else sS_blk
                        for r in cutlass.range_constexpr(
                            cute.size(acc_blk_curr_mn.shape[0])
                        ):
                            row_idx = cutlass.Int32(tScS_blk_mn[r, 0][1])
                            row_local = row_idx - cutlass.Int32(m0)
                            for c in cutlass.range_constexpr(
                                cute.size(acc_blk_curr_mn.shape[1])
                            ):
                                col_idx = cutlass.Int32(tScS_blk_mn[0, c][3])
                                col_local = col_idx - cutlass.Int32(n0)
                                val = cutlass.Float32(0.0)
                                if cute.elem_less(col_idx, row_idx + 1):
                                    val = acc_blk_curr_mn[r, c]
                                sS_curr[col_local, row_local] = val.to(mU.element_type)
                        cute.arch.barrier()

                    for it in cutlass.range_constexpr(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = cutlass.Int32(m0) + t_local
                            if t < cutlass.Int32(self.L):
                                d_local = vv * 2
                                qx = cutlass.Float32(
                                    sQ0[t_local, d_local + 0].to(cutlass.Float32)
                                )
                                qy = cutlass.Float32(
                                    sQ0[t_local, d_local + 1].to(cutlass.Float32)
                                )
                                rs = cutlass.Float32(s_row_scale[t])
                                sQ0[t_local, d_local + 0] = (qx * rs).to(
                                    mU.element_type
                                )
                                sQ0[t_local, d_local + 1] = (qy * rs).to(
                                    mU.element_type
                                )
                    cute.arch.barrier()

                    if single_tile_cache:
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_cache[None, None, 0],
                            tSrS_cache_view[None, None, 0],
                        )
                    else:
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_blk[None, None, 0],
                            tSrS_blk_view[None, None, 0],
                        )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsQt[None, None, 0],
                        tSrQt_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsS_blk.shape[2])):
                        k_next = (k + 1) % cute.size(tSsS_blk.shape[2])
                        if single_tile_cache:
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsS_cache[None, None, k_next],
                                tSrS_cache_view[None, None, k_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsS_blk[None, None, k_next],
                                tSrS_blk_view[None, None, k_next],
                            )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsQt[None, None, k_next],
                            tSrQt_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dK_curr,
                            tSrS_cache[None, None, k]
                            if single_tile_cache
                            else tSrS_blk[None, None, k],
                            tSrQt[None, None, k],
                            acc_dK_curr,
                        )
                    cute.arch.barrier()

                    if (not single_tile_cache) or d_stage_idx == 0:
                        acc_blk_prev = cute.make_rmem_tensor(
                            acc_shape_blk, cutlass.Float32
                        )
                        acc_blk_prev.fill(0.0)
                        p_tile_idx = cutlass.Int32(0)
                        gDY = cute.local_tile(
                            mDOut[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (m_tile, p_tile_idx),
                        )
                        tDYg = gmem_thr_copy_P.partition_S(gDY)
                        cDY = cute.local_tile(
                            mcDY[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (m_tile, p_tile_idx),
                        )
                        tDYc = gmem_thr_copy_P.partition_S(cDY)
                        if cutlass.const_expr(self.P == Pp):
                            cute.copy(gmem_tiled_copy_P, tDYg, tDYs0)
                        else:
                            tDYp_s = cute.make_rmem_tensor(tDYp.layout, cutlass.Boolean)
                            for rest_v in cutlass.range_constexpr(tDYp_s.shape[0]):
                                for rest_k in cutlass.range_constexpr(tDYp_s.shape[2]):
                                    tDYp_s[rest_v, 0, rest_k] = cute.elem_less(
                                        tDYc[(0, rest_v), 0, rest_k][3],
                                        mDOut.layout.shape[3],
                                    )
                            for vi in cutlass.range_constexpr(
                                cute.size(tDYs0.shape[1])
                            ):
                                if cute.elem_less(
                                    tDYc[0, vi, 0][1], mDOut.layout.shape[1]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_P,
                                        tDYg[None, vi, None],
                                        tDYs0[None, vi, None],
                                        pred=tDYp_s[None, vi, None],
                                    )
                                else:
                                    tDYs0[None, vi, None].fill(0)
                        gV = cute.local_tile(
                            mU[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (n_tile, p_tile_idx),
                        )
                        gV = cute.domain_offset((-1, 0), gV)
                        gV = cute.make_tensor(gV.iterator.align(16), gV.layout)
                        tVg = gmem_thr_copy_P.partition_S(gV)
                        cV = cute.local_tile(
                            mcU[bidz, None, 0, None],
                            (kv_tile, p_tile),
                            (n_tile, p_tile_idx),
                        )
                        cV = cute.domain_offset((-1, 0), cV)
                        tVc = gmem_thr_copy_P.partition_S(cV)
                        if cutlass.const_expr(self.P == Pp):
                            if n_tile == cutlass.Int32(0):
                                for vi in cutlass.range_constexpr(
                                    cute.size(tVsV0.shape[1])
                                ):
                                    if cute.elem_less(
                                        cutlass.Int32(-1),
                                        cutlass.Int32(tVc[0, vi, 0][1]),
                                    ):
                                        cute.copy(
                                            gmem_tiled_copy_P,
                                            tVg[None, vi, None],
                                            tVsV0[None, vi, None],
                                        )
                                    else:
                                        tVsV0[None, vi, None].fill(0)
                            else:
                                cute.copy(gmem_tiled_copy_P, tVg, tVsV0)
                        else:
                            tVp_s = cute.make_rmem_tensor(tDYp.layout, cutlass.Boolean)
                            for rest_v in cutlass.range_constexpr(tVp_s.shape[0]):
                                for rest_k in cutlass.range_constexpr(tVp_s.shape[2]):
                                    tVp_s[rest_v, 0, rest_k] = cute.elem_less(
                                        tVc[(0, rest_v), 0, rest_k][3],
                                        mU.layout.shape[3],
                                    )
                            for vi in cutlass.range_constexpr(
                                cute.size(tVsV0.shape[1])
                            ):
                                if cute.elem_less(
                                    cutlass.Int32(-1), cutlass.Int32(tVc[0, vi, 0][1])
                                ) and cute.elem_less(
                                    cutlass.Int32(tVc[0, vi, 0][1]), mU.layout.shape[1]
                                ):
                                    cute.copy(
                                        gmem_tiled_copy_P,
                                        tVg[None, vi, None],
                                        tVsV0[None, vi, None],
                                        pred=tVp_s[None, vi, None],
                                    )
                                else:
                                    tVsV0[None, vi, None].fill(0)
                        cute.arch.cp_async_commit_group()
                        for p_iter in cutlass.range_constexpr(n_p_tiles):
                            if cutlass.const_expr((p_iter & 1) == 0):
                                sDY_p = sDY0
                                sV_p = sV_tile
                            else:
                                sDY_p = sDY1
                                sV_p = sV1
                            if n0 == 0:
                                p_base = cutlass.Int32(p_iter * p_tile)
                                iters_row0 = (
                                    p_tile + self.num_threads - 1
                                ) // self.num_threads
                                for it in range(iters_row0):
                                    p_local = tidx + cutlass.Int32(
                                        it * self.num_threads
                                    )
                                    if p_local < cutlass.Int32(p_tile):
                                        p = p_base + p_local
                                        if p < cutlass.Int32(self.P):
                                            if chunk == cutlass.Int32(0):
                                                sV_p[0, p_local] = mU_prev0[bh, p]
                                            else:
                                                sV_p[0, p_local] = mU[
                                                    bidz - cutlass.Int32(1),
                                                    cutlass.Int32(self.L - 1),
                                                    0,
                                                    p,
                                                ]
                                        else:
                                            sV_p[0, p_local] = cutlass.Float32(0.0).to(
                                                mU.element_type
                                            )
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.barrier()
                            if cutlass.const_expr(p_iter + 1 < n_p_tiles):
                                p_next = cutlass.Int32(p_iter + 1)
                                gDY_next = cute.local_tile(
                                    mDOut[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (m_tile, p_next),
                                )
                                tDYg_next = gmem_thr_copy_P.partition_S(gDY_next)
                                cDY_next = cute.local_tile(
                                    mcDY[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (m_tile, p_next),
                                )
                                tDYc_next = gmem_thr_copy_P.partition_S(cDY_next)
                                gV_next = cute.local_tile(
                                    mU[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (n_tile, p_next),
                                )
                                gV_next = cute.domain_offset((-1, 0), gV_next)
                                gV_next = cute.make_tensor(
                                    gV_next.iterator.align(16), gV_next.layout
                                )
                                tVg_next = gmem_thr_copy_P.partition_S(gV_next)
                                cV_next = cute.local_tile(
                                    mcU[bidz, None, 0, None],
                                    (kv_tile, p_tile),
                                    (n_tile, p_next),
                                )
                                cV_next = cute.domain_offset((-1, 0), cV_next)
                                tVc_next = gmem_thr_copy_P.partition_S(cV_next)
                                if cutlass.const_expr((p_iter & 1) == 0):
                                    if cutlass.const_expr(self.P == Pp):
                                        cute.copy(gmem_tiled_copy_P, tDYg_next, tDYs1)
                                        if n_tile == cutlass.Int32(0):
                                            for vi in cutlass.range_constexpr(
                                                cute.size(tVsV1.shape[1])
                                            ):
                                                if cute.elem_less(
                                                    cutlass.Int32(-1),
                                                    cutlass.Int32(
                                                        tVc_next[0, vi, 0][1]
                                                    ),
                                                ):
                                                    cute.copy(
                                                        gmem_tiled_copy_P,
                                                        tVg_next[None, vi, None],
                                                        tVsV1[None, vi, None],
                                                    )
                                                else:
                                                    tVsV1[None, vi, None].fill(0)
                                        else:
                                            cute.copy(
                                                gmem_tiled_copy_P, tVg_next, tVsV1
                                            )
                                    else:
                                        tDYp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        tVp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        for rest_v in cutlass.range_constexpr(
                                            tDYp_next.shape[0]
                                        ):
                                            for rest_k in cutlass.range_constexpr(
                                                tDYp_next.shape[2]
                                            ):
                                                tDYp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tDYc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mDOut.layout.shape[3],
                                                    )
                                                )
                                                tVp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tVc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mU.layout.shape[3],
                                                    )
                                                )
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tDYs1.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tDYc_next[0, vi, 0][1],
                                                mDOut.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tDYg_next[None, vi, None],
                                                    tDYs1[None, vi, None],
                                                    pred=tDYp_next[None, vi, None],
                                                )
                                            else:
                                                tDYs1[None, vi, None].fill(0)
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tVsV1.shape[1])
                                        ):
                                            if cute.elem_less(
                                                cutlass.Int32(-1),
                                                cutlass.Int32(tVc_next[0, vi, 0][1]),
                                            ) and cute.elem_less(
                                                cutlass.Int32(tVc_next[0, vi, 0][1]),
                                                mU.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tVg_next[None, vi, None],
                                                    tVsV1[None, vi, None],
                                                    pred=tVp_next[None, vi, None],
                                                )
                                            else:
                                                tVsV1[None, vi, None].fill(0)
                                else:
                                    if cutlass.const_expr(self.P == Pp):
                                        cute.copy(gmem_tiled_copy_P, tDYg_next, tDYs0)
                                        if n_tile == cutlass.Int32(0):
                                            for vi in cutlass.range_constexpr(
                                                cute.size(tVsV0.shape[1])
                                            ):
                                                if cute.elem_less(
                                                    cutlass.Int32(-1),
                                                    cutlass.Int32(
                                                        tVc_next[0, vi, 0][1]
                                                    ),
                                                ):
                                                    cute.copy(
                                                        gmem_tiled_copy_P,
                                                        tVg_next[None, vi, None],
                                                        tVsV0[None, vi, None],
                                                    )
                                                else:
                                                    tVsV0[None, vi, None].fill(0)
                                        else:
                                            cute.copy(
                                                gmem_tiled_copy_P, tVg_next, tVsV0
                                            )
                                    else:
                                        tDYp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        tVp_next = cute.make_rmem_tensor(
                                            tDYp.layout, cutlass.Boolean
                                        )
                                        for rest_v in cutlass.range_constexpr(
                                            tDYp_next.shape[0]
                                        ):
                                            for rest_k in cutlass.range_constexpr(
                                                tDYp_next.shape[2]
                                            ):
                                                tDYp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tDYc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mDOut.layout.shape[3],
                                                    )
                                                )
                                                tVp_next[rest_v, 0, rest_k] = (
                                                    cute.elem_less(
                                                        tVc_next[
                                                            (0, rest_v), 0, rest_k
                                                        ][3],
                                                        mU.layout.shape[3],
                                                    )
                                                )
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tDYs0.shape[1])
                                        ):
                                            if cute.elem_less(
                                                tDYc_next[0, vi, 0][1],
                                                mDOut.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tDYg_next[None, vi, None],
                                                    tDYs0[None, vi, None],
                                                    pred=tDYp_next[None, vi, None],
                                                )
                                            else:
                                                tDYs0[None, vi, None].fill(0)
                                        for vi in cutlass.range_constexpr(
                                            cute.size(tVsV0.shape[1])
                                        ):
                                            if cute.elem_less(
                                                cutlass.Int32(-1),
                                                cutlass.Int32(tVc_next[0, vi, 0][1]),
                                            ) and cute.elem_less(
                                                cutlass.Int32(tVc_next[0, vi, 0][1]),
                                                mU.layout.shape[1],
                                            ):
                                                cute.copy(
                                                    gmem_tiled_copy_P,
                                                    tVg_next[None, vi, None],
                                                    tVsV0[None, vi, None],
                                                    pred=tVp_next[None, vi, None],
                                                )
                                            else:
                                                tVsV0[None, vi, None].fill(0)
                                cute.arch.cp_async_commit_group()
                            tSrDY_m = thr_mma.make_fragment_A(
                                thr_mma.partition_A(sDY_p)
                            )
                            tSsDY_m = thr_copy_A.partition_S(sDY_p)
                            tSrDY_m_view = thr_copy_A.retile(tSrDY_m)
                            tSrV_p = thr_mma.make_fragment_B(thr_mma.partition_B(sV_p))
                            tSsV_p = thr_copy_B.partition_S(sV_p)
                            tSrV_p_view = thr_copy_B.retile(tSrV_p)
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsDY_m[None, None, 0],
                                tSrDY_m_view[None, None, 0],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tSsV_p[None, None, 0],
                                tSrV_p_view[None, None, 0],
                            )
                            for k in cutlass.range_constexpr(
                                cute.size(tSsDY_m.shape[2])
                            ):
                                k_next = (k + 1) % cute.size(tSsDY_m.shape[2])
                                cute.copy(
                                    smem_tiled_copy_A,
                                    tSsDY_m[None, None, k_next],
                                    tSrDY_m_view[None, None, k_next],
                                )
                                cute.copy(
                                    smem_tiled_copy_B,
                                    tSsV_p[None, None, k_next],
                                    tSrV_p_view[None, None, k_next],
                                )
                                cute.gemm(
                                    tiled_mma,
                                    acc_blk_prev,
                                    tSrDY_m[None, None, k],
                                    tSrV_p[None, None, k],
                                    acc_blk_prev,
                                )
                        acc_blk_prev_mn = self._make_acc_tensor_mn_view(acc_blk_prev)
                        sS_prev_target = sS_prev if single_tile_cache else sS_blk
                        for r in cutlass.range_constexpr(
                            cute.size(acc_blk_prev_mn.shape[0])
                        ):
                            row_idx = cutlass.Int32(tScS_blk_mn[r, 0][1])
                            row_local = row_idx - cutlass.Int32(m0)
                            for c in cutlass.range_constexpr(
                                cute.size(acc_blk_prev_mn.shape[1])
                            ):
                                col_idx = cutlass.Int32(tScS_blk_mn[0, c][3])
                                col_local = col_idx - cutlass.Int32(n0)
                                val = cutlass.Float32(0.0)
                                if cute.elem_less(col_idx, row_idx + 1):
                                    val = acc_blk_prev_mn[r, c]
                                sS_prev_target[col_local, row_local] = val.to(
                                    mU.element_type
                                )
                        cute.arch.barrier()

                    if single_tile_cache:
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_prev[None, None, 0],
                            tSrS_prev_view[None, None, 0],
                        )
                    else:
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_blk[None, None, 0],
                            tSrS_blk_view[None, None, 0],
                        )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsQt[None, None, 0],
                        tSrQt_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsS_blk.shape[2])):
                        k_next = (k + 1) % cute.size(tSsS_blk.shape[2])
                        if single_tile_cache:
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsS_prev[None, None, k_next],
                                tSrS_prev_view[None, None, k_next],
                            )
                        else:
                            cute.copy(
                                smem_tiled_copy_A,
                                tSsS_blk[None, None, k_next],
                                tSrS_blk_view[None, None, k_next],
                            )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsQt[None, None, k_next],
                            tSrQt_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dK_prev,
                            (
                                tSrS_prev[None, None, k]
                                if single_tile_cache
                                else tSrS_blk[None, None, k]
                            ),
                            tSrQt[None, None, k],
                            acc_dK_prev,
                        )
                    cute.arch.barrier()

                cKD_stage = cute.local_tile(
                    mcKD_full, (kv_tile, Ds), (n_tile, d_stage_idx)
                )
                tOcKD_stage = thr_mma.partition_C(cKD_stage)
                tOcKD_stage_mn = self._make_acc_tensor_mn_view(tOcKD_stage)
                tCsPrev = thr_mma.partition_C(sQ0)
                tCrPrev = cute.make_fragment_like(tCsPrev, mU.element_type)
                tCrPrev.fill(0.0)
                tCrPrev_mn = self._make_acc_tensor_mn_view(tCrPrev)
                tCsK = thr_mma.partition_C(sK_tile)
                tCrK_curr = cute.make_fragment_like(tCsK, mU.element_type)
                tCrK_curr.fill(0.0)
                tCrK_curr_mn = self._make_acc_tensor_mn_view(tCrK_curr)
                acc_dK_curr_mn = self._make_acc_tensor_mn_view(acc_dK_curr)
                acc_dK_prev_mn = self._make_acc_tensor_mn_view(acc_dK_prev)

                for r in cutlass.range_constexpr(cute.size(acc_dK_curr_mn.shape[0])):
                    row_idx = cutlass.Int32(tOcKD_stage_mn[r, 0][1])
                    if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                        inv_rs = cutlass.Float32(s_inv_row_scale[row_idx])
                        for c in cutlass.range(cute.size(acc_dK_curr_mn.shape[1])):
                            d = cutlass.Int32(tOcKD_stage_mn[0, c][3])
                            if (
                                d + cutlass.Int32(1) < cutlass.Int32(self.D)
                                and (d & 1) == 0
                            ):
                                gx = acc_dK_curr_mn[r, c] * inv_rs
                                gy = acc_dK_curr_mn[r, c + 1] * inv_rs
                                pr = cutlass.Float32(s_phase[row_idx, 0])
                                pi = cutlass.Float32(s_phase[row_idx, 1])
                                bxr, bxi = conj_mul_phase(gx, gy, pr, pi)
                                tCrK_curr_mn[r, c] = bxr.to(mU.element_type)
                                tCrK_curr_mn[r, c + 1] = bxi.to(mU.element_type)

                for r in cutlass.range_constexpr(cute.size(acc_dK_prev_mn.shape[0])):
                    row_idx = cutlass.Int32(tOcKD_stage_mn[r, 0][1])
                    if cute.elem_less(row_idx, cutlass.Int32(self.L)):
                        inv_rs = cutlass.Float32(s_inv_row_scale[row_idx])
                        for c in cutlass.range(cute.size(acc_dK_prev_mn.shape[1])):
                            d = cutlass.Int32(tOcKD_stage_mn[0, c][3])
                            if (
                                d + cutlass.Int32(1) < cutlass.Int32(self.D)
                                and (d & 1) == 0
                            ):
                                gx = acc_dK_prev_mn[r, c] * inv_rs
                                gy = acc_dK_prev_mn[r, c + 1] * inv_rs
                                pr = cutlass.Float32(s_phase[row_idx, 0])
                                pi = cutlass.Float32(s_phase[row_idx, 1])
                                bxr, bxi = conj_mul_phase(gx, gy, pr, pi)
                                tCrPrev_mn[r, c] = bxr.to(mU.element_type)
                                tCrPrev_mn[r, c + 1] = bxi.to(mU.element_type)
                cute.autovec_copy(tCrPrev, tCsPrev)
                cute.autovec_copy(tCrK_curr, tCsK)
                cute.arch.barrier()

                t_local = warp
                while t_local < cutlass.Int32(kv_tile):
                    row = cutlass.Int32(n0) + t_local
                    if row < cutlass.Int32(self.L):
                        dmy_curr0 = cutlass.Float32(0.0)
                        dmy_curr1 = cutlass.Float32(0.0)
                        dmy_prev0 = cutlass.Float32(0.0)
                        dmy_prev1 = cutlass.Float32(0.0)
                        pr = cutlass.Float32(s_phase[row, 0])
                        pi = cutlass.Float32(s_phase[row, 1])
                        vv = lane
                        while vv < cutlass.Int32(nvec_stage):
                            d_local = vv * 2
                            d0 = d_col_base + d_local
                            if d0 + cutlass.Int32(1) < cutlass.Int32(self.D):
                                bxr_curr = cutlass.Float32(
                                    sK_tile[t_local, d_local + 0].to(cutlass.Float32)
                                )
                                bxi_curr = cutlass.Float32(
                                    sK_tile[t_local, d_local + 1].to(cutlass.Float32)
                                )
                                gx_curr, gy_curr = conj_mul_phase(
                                    bxr_curr, bxi_curr, pr, pi
                                )
                                br_curr = cutlass.Float32(
                                    mB[bidz, row, 0, d0 + 0].to(cutlass.Float32)
                                )
                                bi_curr = cutlass.Float32(
                                    mB[bidz, row, 0, d0 + 1].to(cutlass.Float32)
                                )
                                dmy_curr0 = (
                                    dmy_curr0 + gx_curr * br_curr - gy_curr * bi_curr
                                )
                                dmy_curr1 = (
                                    dmy_curr1 + gx_curr * bi_curr + gy_curr * br_curr
                                )

                                bxr_prev = cutlass.Float32(
                                    sQ0[t_local, d_local + 0].to(cutlass.Float32)
                                )
                                bxi_prev = cutlass.Float32(
                                    sQ0[t_local, d_local + 1].to(cutlass.Float32)
                                )
                                gx_prev, gy_prev = conj_mul_phase(
                                    bxr_prev, bxi_prev, pr, pi
                                )
                                br_prev = cutlass.Float32(0.0)
                                bi_prev = cutlass.Float32(0.0)
                                if row > cutlass.Int32(0):
                                    br_prev = cutlass.Float32(
                                        mB[bidz, row - cutlass.Int32(1), 0, d0 + 0].to(
                                            cutlass.Float32
                                        )
                                    )
                                    bi_prev = cutlass.Float32(
                                        mB[bidz, row - cutlass.Int32(1), 0, d0 + 1].to(
                                            cutlass.Float32
                                        )
                                    )
                                else:
                                    if chunk == cutlass.Int32(0):
                                        br_prev = cutlass.Float32(
                                            mB_prev0[bh, d0 + 0].to(cutlass.Float32)
                                        )
                                        bi_prev = cutlass.Float32(
                                            mB_prev0[bh, d0 + 1].to(cutlass.Float32)
                                        )
                                    else:
                                        br_prev = cutlass.Float32(
                                            mB[
                                                bidz - cutlass.Int32(1),
                                                cutlass.Int32(self.L - 1),
                                                0,
                                                d0 + 0,
                                            ].to(cutlass.Float32)
                                        )
                                        bi_prev = cutlass.Float32(
                                            mB[
                                                bidz - cutlass.Int32(1),
                                                cutlass.Int32(self.L - 1),
                                                0,
                                                d0 + 1,
                                            ].to(cutlass.Float32)
                                        )
                                dmy_prev0 = (
                                    dmy_prev0 + gx_prev * br_prev - gy_prev * bi_prev
                                )
                                dmy_prev1 = (
                                    dmy_prev1 + gx_prev * bi_prev + gy_prev * br_prev
                                )
                            vv = vv + cutlass.Int32(32)

                        for off in (16, 8, 4, 2, 1):
                            dmy_curr0 = dmy_curr0 + cute.arch.shuffle_sync_bfly(
                                dmy_curr0, offset=off, mask=-1, mask_and_clamp=31
                            )
                            dmy_curr1 = dmy_curr1 + cute.arch.shuffle_sync_bfly(
                                dmy_curr1, offset=off, mask=-1, mask_and_clamp=31
                            )
                            dmy_prev0 = dmy_prev0 + cute.arch.shuffle_sync_bfly(
                                dmy_prev0, offset=off, mask=-1, mask_and_clamp=31
                            )
                            dmy_prev1 = dmy_prev1 + cute.arch.shuffle_sync_bfly(
                                dmy_prev1, offset=off, mask=-1, mask_and_clamp=31
                            )
                        if lane == cutlass.Int32(0):
                            s_dM_curr[t_local, 0] = cutlass.Float32(
                                s_dM_curr[t_local, 0] + dmy_curr0
                            )
                            s_dM_curr[t_local, 1] = cutlass.Float32(
                                s_dM_curr[t_local, 1] + dmy_curr1
                            )
                            s_dM_prev[t_local, 0] = cutlass.Float32(
                                s_dM_prev[t_local, 0] + dmy_prev0
                            )
                            s_dM_prev[t_local, 1] = cutlass.Float32(
                                s_dM_prev[t_local, 1] + dmy_prev1
                            )
                    t_local = t_local + cutlass.Int32(self.num_warps)
                cute.arch.barrier()

                for it in cutlass.range_constexpr(iters_pairs_stage):
                    idx = tidx + cutlass.Int32(it * self.num_threads)
                    if idx < cutlass.Int32(total_pairs_stage):
                        t_local = idx // nvec_stage
                        vv = idx - t_local * nvec_stage
                        row = cutlass.Int32(n0) + t_local
                        if row < cutlass.Int32(self.L):
                            d_local = vv * 2
                            bxr_curr = cutlass.Float32(
                                sK_tile[t_local, d_local + 0].to(cutlass.Float32)
                            )
                            bxi_curr = cutlass.Float32(
                                sK_tile[t_local, d_local + 1].to(cutlass.Float32)
                            )
                            kr_curr = cutlass.Float32(s_tap_curr[t_local, 0])
                            ki_curr = cutlass.Float32(s_tap_curr[t_local, 1])
                            currx, curry = apply_complex_tap_adjoint(
                                bxr_curr, bxi_curr, kr_curr, ki_curr
                            )
                            prevx = cutlass.Float32(0.0)
                            prevy = cutlass.Float32(0.0)
                            if row + cutlass.Int32(1) < cutlass.Int32(self.L):
                                if t_local + cutlass.Int32(1) < cutlass.Int32(kv_tile):
                                    row_next = row + cutlass.Int32(1)
                                    bxr_prev = cutlass.Float32(
                                        sQ0[t_local + cutlass.Int32(1), d_local + 0].to(
                                            cutlass.Float32
                                        )
                                    )
                                    bxi_prev = cutlass.Float32(
                                        sQ0[t_local + cutlass.Int32(1), d_local + 1].to(
                                            cutlass.Float32
                                        )
                                    )
                                    kr_prev = cutlass.Float32(s_tap_prev[row_next, 0])
                                    ki_prev = cutlass.Float32(s_tap_prev[row_next, 1])
                                    prevx, prevy = apply_complex_tap_adjoint(
                                        bxr_prev, bxi_prev, kr_prev, ki_prev
                                    )
                                else:
                                    prevx = cutlass.Float32(
                                        sDB_carry[d_col_base + d_local + 0].to(
                                            cutlass.Float32
                                        )
                                    )
                                    prevy = cutlass.Float32(
                                        sDB_carry[d_col_base + d_local + 1].to(
                                            cutlass.Float32
                                        )
                                    )
                            sK_tile[t_local, d_local + 0] = (currx + prevx).to(
                                mU.element_type
                            )
                            sK_tile[t_local, d_local + 1] = (curry + prevy).to(
                                mU.element_type
                            )
                cute.arch.barrier()

                gDB_stage = cute.local_tile(
                    mDB[bidz, None, 0, None], (kv_tile, Ds), (n_tile, d_stage_idx)
                )
                gmem_thr_store_B = gmem_tiled_store_D.get_slice(tidx)
                tBsK = gmem_thr_store_B.partition_S(sK_tile)
                tBgDB = gmem_thr_store_B.partition_D(gDB_stage)
                if cutlass.const_expr(self.D == Dp):
                    cute.copy(gmem_tiled_store_D, tBsK, tBgDB)
                else:
                    tBrB = cute.make_rmem_tensor_like(tBgDB, mU.element_type)
                    cute.copy(gmem_tiled_store_D, tBsK, tBrB)
                    cDB_stage = cute.local_tile(
                        mcDB[bidz, None, 0, None], (kv_tile, Ds), (n_tile, d_stage_idx)
                    )
                    tBcDB = gmem_thr_store_B.partition_D(cDB_stage)
                    tBpDB = cute.make_rmem_tensor(
                        cute.make_layout(
                            (tBgDB.shape[0][1], tBgDB.shape[1], tBgDB.shape[2]),
                            stride=(tBgDB.shape[2], 0, 1),
                        ),
                        cutlass.Boolean,
                    )
                    for rest_v in cutlass.range_constexpr(tBpDB.shape[0]):
                        for rest_n in cutlass.range_constexpr(
                            cute.size(tBpDB.shape[2])
                        ):
                            tBpDB[rest_v, 0, rest_n] = cute.elem_less(
                                tBcDB[(0, rest_v), 0, rest_n][3], mDB.layout.shape[3]
                            )
                    for rest_m in cutlass.range_constexpr(cute.size(tBpDB.shape[1])):
                        if cute.elem_less(tBcDB[0, rest_m, 0][1], mDB.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_store_D,
                                tBrB[None, rest_m, None],
                                tBgDB[None, rest_m, None],
                                pred=tBpDB[None, rest_m, None],
                            )

                for it in range(iters_ds):
                    d_local = tidx + cutlass.Int32(it * self.num_threads)
                    if d_local + cutlass.Int32(1) < cutlass.Int32(Ds) and (
                        d_local & cutlass.Int32(1)
                    ) == cutlass.Int32(0):
                        d = d_col_base + d_local
                        if d < cutlass.Int32(Dp):
                            bxr_head = cutlass.Float32(
                                sQ0[0, d_local + 0].to(cutlass.Float32)
                            )
                            bxi_head = cutlass.Float32(
                                sQ0[0, d_local + 1].to(cutlass.Float32)
                            )
                            kr_head = cutlass.Float32(s_tap_prev[n0, 0])
                            ki_head = cutlass.Float32(s_tap_prev[n0, 1])
                            headx, heady = apply_complex_tap_adjoint(
                                bxr_head, bxi_head, kr_head, ki_head
                            )
                            sDB_carry[d_col_base + d_local + 0] = headx.to(
                                mU.element_type
                            )
                            sDB_carry[d_col_base + d_local + 1] = heady.to(
                                mU.element_type
                            )
                cute.arch.barrier()

            if tidx < cutlass.Int32(kv_tile):
                row = cutlass.Int32(n0) + tidx
                if row < cutlass.Int32(self.L):
                    mDMcurr[bidz, row, 0] = s_dM_curr[tidx, 0]
                    mDMcurr[bidz, row, 1] = s_dM_curr[tidx, 1]
                    mDMprev[bidz, row, 0] = s_dM_prev[tidx, 0]
                    mDMprev[bidz, row, 1] = s_dM_prev[tidx, 1]
            cute.arch.barrier()
        for it in cutlass.range_constexpr(iters_d):
            d = tidx + cutlass.Int32(it * self.num_threads)
            if d < cutlass.Int32(self.D):
                mDB_prev[bidz, d] = sDB_carry[d]


__all__ = ["ChunkScanBwdDBAmpere"]
