"""CuTe backward ``du`` workhorse for the ``v2x2ssd`` chunk-scan stage.

This file is intentionally written in the same shape as the ``v3x3ssd``
``chunk_scan`` ``du`` workhorse:

- one monolithic stage-native kernel class
- direct public-stage inputs and outputs
- in-kernel prefix reconstruction
- tensor-core GEMMs for the dense contractions
- direct public ``dU`` / ``dU_prev`` writeout

The only changes are the ones forced by the scan algebra:

- SO(3) quaternion transport becomes SO(2) unit-complex transport
- 3-vectors become interleaved complex pairs
- FOH vector taps become complex-scalar taps
- ``trans`` becomes raw packed ``M``
"""

import cutlass
import cutlass.cute as cute

from .common import (
    LOG2_E,
    TWO_LOG2_E,
    apply_complex_tap,
    clamp_nonpositive_prefix_log,
    complex_mul,
    conj_mul_phase,
    safe_cast_to_dtype,
)


class ChunkScanBwdDUAmpere:
    """Ampere tensor-core kernel for ``chunk_scan`` backward value grads.

    Computes, per chunk (BHC):
      - ``dU`` and ``dU_prev``
      - placeholder ``dB`` / ``dB_prev`` / ``dlogprefix`` / ``dM`` outputs

    Notes:
      - This kernel recomputes the chunk-local prefix scan from raw packed
        complex transitions ``M`` and applies the packed complex taps on the fly.
      - The heavy contractions are the same FA2-style tensor-core GEMMs used in
        the ``v3`` workhorse, specialized to interleaved complex pairs.
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
        self.do_du = True
        self.do_db = False
        self.do_dlp = False

    @property
    def D_padded(self) -> int:
        return (self.D + 31) // 32 * 32

    def _d_stage_size(self) -> int:
        """Shared-memory D slice width used by the backward DU kernel."""
        Dp = self.D_padded
        if Dp <= 32:
            return Dp
        return 32

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
        swizzle_bits_P = 2
        sP_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits_P, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size_P), stride=(smem_k_block_size_P, 1)),
        )
        sDY_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))
        sV_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, p_tile), (0, 1))
        sDV_prev_layout = cute.tile_to_shape(sP_layout_atom, (kv_tile, Pp), (0, 1))

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
        tD_shape_dim_1 = sD_layout_atom.outer.shape[1] // async_elems_in
        tD_layout = cute.make_layout(
            (self.num_threads // tD_shape_dim_1, tD_shape_dim_1),
            stride=(tD_shape_dim_1, 1),
        )
        tP_shape_dim_1 = sP_layout_atom.outer.shape[1] // async_elems_in
        tP_layout = cute.make_layout(
            (self.num_threads // tP_shape_dim_1, tP_shape_dim_1),
            stride=(tP_shape_dim_1, 1),
        )
        v_in_layout = cute.make_layout((1, async_elems_in))
        gmem_tiled_copy_D = cute.make_tiled_copy_tv(
            atom_async_copy_in, tD_layout, v_in_layout
        )
        gmem_tiled_copy_P = cute.make_tiled_copy_tv(
            atom_async_copy_in, tP_layout, v_in_layout
        )

        store_elems = universal_copy_bits // mU.element_type.width
        atom_universal_copy_out = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mU.element_type,
            num_bits_per_copy=universal_copy_bits,
        )
        v_out_layout = cute.make_layout((1, store_elems))
        gmem_tiled_store_D = cute.make_tiled_copy_tv(
            atom_universal_copy_out, tD_layout, v_out_layout
        )
        gmem_tiled_store_P = cute.make_tiled_copy_tv(
            atom_universal_copy_out, tP_layout, v_out_layout
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
            sDV_prev: cute.struct.Align[
                cute.struct.MemRange[mU.element_type, 1],
                16,
            ]
            sS_blk: cute.struct.Align[
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
            sDV_carry: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, Pp], 8]
            s_dlp: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.L], 4]
            s_row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L], 4
            ]
            s_inv_row_scale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.L], 4
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
            sDV_prev_layout,
            sBlk_layout,
            gmem_tiled_copy_D,
            gmem_tiled_copy_P,
            gmem_tiled_store_D,
            gmem_tiled_store_P,
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
        sDV_prev_layout: cute.ComposedLayout,
        sBlk_layout: cute.ComposedLayout,
        gmem_tiled_copy_D: cute.TiledCopy,
        gmem_tiled_copy_P: cute.TiledCopy,
        gmem_tiled_store_D: cute.TiledCopy,
        gmem_tiled_store_P: cute.TiledCopy,
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
        kv_tile = self.kv_tile
        n_tiles = self.L // kv_tile

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ0 = storage.sQ.get_tensor(sQ_layout)
        sDY0 = storage.sDY.get_tensor(sDY_layout)
        sQ1 = sQ0
        sDY1 = sDY0
        sK_tile = storage.sK_tile.get_tensor(sK_layout)
        sV_tile = storage.sV_tile.get_tensor(sV_layout)
        sDV_carry = storage.sDV_carry.get_tensor(cute.make_layout((Pp,), stride=(1,)))
        sS_blk = storage.sS_blk.get_tensor(sBlk_layout)
        s_phase = storage.s_phase.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_prev = storage.s_tap_prev.get_tensor(
            cute.make_layout((self.L, 2), stride=(2, 1))
        )
        s_tap_curr = storage.s_tap_curr.get_tensor(
            cute.make_layout((kv_tile, 2), stride=(2, 1))
        )
        s_dlp = storage.s_dlp.get_tensor(cute.make_layout((self.L,), stride=(1,)))
        s_row_scale = storage.s_row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )
        s_inv_row_scale = storage.s_inv_row_scale.get_tensor(
            cute.make_layout((self.L,), stride=(1,))
        )

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        t_safe = cutlass.select_(
            tidx < cutlass.Int32(self.L), tidx, cutlass.Int32(self.L - 1)
        )

        if tidx < cutlass.Int32(self.L):
            s_phase[tidx, 0] = cutlass.Float32(mM[bidz, tidx, 0])
            s_phase[tidx, 1] = cutlass.Float32(mM[bidz, tidx, 1])
        cute.arch.barrier()

        mr = cutlass.Float32(s_phase[t_safe, 0])
        mi = cutlass.Float32(s_phase[t_safe, 1])
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
            orr = cute.arch.shuffle_sync_up(
                ur, offset=offset, mask=-1, mask_and_clamp=0
            )
            oii = cute.arch.shuffle_sync_up(
                ui, offset=offset, mask=-1, mask_and_clamp=0
            )
            pred = lane >= cutlass.Int32(offset)
            logp = cutlass.select_(pred, logp + other_log, logp)
            nr, ni = complex_mul(ur, ui, orr, oii)
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
            stable_logp = clamp_nonpositive_prefix_log(logp)
            row_scale = cute.math.exp2(
                stable_logp * cutlass.Float32(TWO_LOG2_E), fastmath=False
            )
            s_row_scale[tidx] = row_scale
            s_inv_row_scale[tidx] = cutlass.Float32(1.0) / row_scale
            tap_prev = mK[bidz, tidx, 0, None].load().to(cutlass.Float32)
            s_tap_prev[tidx, 0] = cutlass.Float32(tap_prev[0])
            s_tap_prev[tidx, 1] = cutlass.Float32(tap_prev[1])
            s_phase[tidx, 0] = ur
            s_phase[tidx, 1] = ui
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

        mcS = cute.make_identity_tensor((mB.shape[0], self.L, mB.shape[2], self.L))
        mcU = cute.make_identity_tensor((mB.shape[0], self.L, mB.shape[2], Pp))
        mcKD = cute.make_identity_tensor((mB.shape[0], self.L, mB.shape[2], Dp))
        mcS_full = mcS[bidz, None, 0, None]
        mcU_full = mcU[bidz, None, 0, None]
        mcKD_full = mcKD[bidz, None, 0, None]
        sDYt_layout = cute.make_layout((p_tile, kv_tile), stride=(kv_tile, 1))
        gmem_thr_copy_D = gmem_tiled_copy_D.get_slice(tidx)
        gmem_thr_copy_P = gmem_tiled_copy_P.get_slice(tidx)
        tQsQ0 = gmem_thr_copy_D.partition_D(sQ0)
        tDYs0 = gmem_thr_copy_P.partition_D(sDY0)
        tKsK = gmem_thr_copy_D.partition_D(sK_tile)

        mcC = cute.make_identity_tensor(mC.layout.shape)
        cC0 = cute.local_tile(mcC[bidz, None, 0, None], (kv_tile, Ds), (0, 0))
        tCc0 = gmem_thr_copy_D.partition_S(cC0)
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

        mcDY = cute.make_identity_tensor(mDOut.layout.shape)
        cDY0 = cute.local_tile(mcDY[bidz, None, 0, None], (kv_tile, p_tile), (0, 0))
        tDYc0 = gmem_thr_copy_P.partition_S(cDY0)
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
        for rest_v in cutlass.range_constexpr(tDYp.shape[0]):
            for rest_k in cutlass.range_constexpr(tDYp.shape[2]):
                tDYp[rest_v, 0, rest_k] = cute.elem_less(
                    tDYc0[(0, rest_v), 0, rest_k][3], mDOut.layout.shape[3]
                )

        acc_shape_blk = thr_mma.partition_shape_C((kv_tile, kv_tile))
        acc_shape_tileP = thr_mma.partition_shape_C((kv_tile, p_tile))
        thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        thr_copy_B = smem_tiled_copy_B.get_slice(tidx)
        thr_copy_BT = smem_tiled_copy_BT.get_slice(tidx)
        tSrS_blk = thr_mma.make_fragment_A(thr_mma.partition_A(sS_blk))
        tSsS_blk = thr_copy_A.partition_S(sS_blk)
        tSrS_blk_view = thr_copy_A.retile(tSrS_blk)
        tSrK_tile = thr_mma.make_fragment_B(thr_mma.partition_B(sK_tile))
        tSsK_tile = thr_copy_B.partition_S(sK_tile)
        tSrK_tile_view = thr_copy_B.retile(tSrK_tile)

        total_pairs_stage = kv_tile * nvec_stage
        iters_pairs_stage = (
            total_pairs_stage + self.num_threads - 1
        ) // self.num_threads
        iters_p_padded = (Pp + self.num_threads - 1) // self.num_threads
        iters_ds = (Ds + self.num_threads - 1) // self.num_threads
        for it in range(iters_p_padded):
            p = tidx + cutlass.Int32(it * self.num_threads)
            if p < cutlass.Int32(Pp):
                sDV_carry[p] = cutlass.Float32(0.0)
        cute.arch.barrier()

        for n_tile_rev in cutlass.range_constexpr(n_tiles):
            n_tile = (n_tiles - 1) - n_tile_rev
            n0 = n_tile * kv_tile
            acc_dV_tiles = []
            for _ in cutlass.range_constexpr(n_p_tiles):
                acc = cute.make_rmem_tensor(acc_shape_tileP, self.acc_dtype)
                acc.fill(0.0)
                acc_dV_tiles.append(acc)

            m_tiles = n_tiles - n_tile
            for mi in cutlass.range_constexpr(m_tiles):
                m_tile = n_tile + mi
                m0 = m_tile * kv_tile
                use_buf1 = False
                sDY_m = sDY1 if use_buf1 else sDY0
                sQ_m = sQ1 if use_buf1 else sQ0

                cS_blk = cute.local_tile(mcS_full, (kv_tile, kv_tile), (m_tile, n_tile))
                tScS_blk = thr_mma.partition_C(cS_blk)
                tScS_blk_mn = self._make_acc_tensor_mn_view(tScS_blk)
                acc_blk = cute.make_rmem_tensor(acc_shape_blk, self.acc_dtype)
                acc_blk.fill(0.0)

                cC_row = cute.local_tile(
                    mcC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, 0)
                )
                tCc_row = gmem_thr_copy_D.partition_S(cC_row)

                for d_stage_idx in cutlass.range_constexpr(d_stage_count):
                    d_col_base = cutlass.Int32(d_stage_idx * Ds)
                    for rest_v in cutlass.range_constexpr(tQp.shape[0]):
                        for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                            g_col = (
                                cutlass.Int32(tCc0[(0, rest_v), 0, rest_k][3])
                                + d_col_base
                            )
                            tQp[rest_v, 0, rest_k] = cute.elem_less(
                                g_col, mC.layout.shape[3]
                            )

                    gB = cute.local_tile(
                        mB[bidz, None, 0, None], (kv_tile, Ds), (n_tile, d_stage_idx)
                    )
                    tBg = gmem_thr_copy_D.partition_S(gB)
                    cB = cute.local_tile(
                        mcKD_full, (kv_tile, Ds), (n_tile, d_stage_idx)
                    )
                    tBc = gmem_thr_copy_D.partition_S(cB)
                    for di in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if cute.elem_less(tBc[0, di, 0][1], mB.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_copy_D,
                                tBg[None, di, None],
                                tKsK[None, di, None],
                                pred=tQp[None, di, None],
                            )
                        else:
                            tKsK[None, di, None].fill(0)

                    gC = cute.local_tile(
                        mC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, d_stage_idx)
                    )
                    tCg = gmem_thr_copy_D.partition_S(gC)
                    for di in cutlass.range_constexpr(cute.size(tQsQ0.shape[1])):
                        if cute.elem_less(tCc_row[0, di, 0][1], mC.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_copy_D,
                                tCg[None, di, None],
                                tQsQ0[None, di, None],
                                pred=tQp[None, di, None],
                            )
                        else:
                            tQsQ0[None, di, None].fill(0)

                    if tidx < cutlass.Int32(kv_tile):
                        t = cutlass.Int32(n0) + tidx
                        s_tap_curr[tidx, 0] = cutlass.Float32(mK[bidz, t, 1, 0])
                        s_tap_curr[tidx, 1] = cutlass.Float32(mK[bidz, t, 1, 1])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()

                    for it in range(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = t_local + cutlass.Int32(n0)
                            if t < cutlass.Int32(self.L):
                                d0 = vv * 2
                                bx = cutlass.Float32(
                                    sK_tile[t_local, d0 + 0].to(cutlass.Float32)
                                )
                                by = cutlass.Float32(
                                    sK_tile[t_local, d0 + 1].to(cutlass.Float32)
                                )
                                kr = cutlass.Float32(s_tap_curr[t_local, 0])
                                ki = cutlass.Float32(s_tap_curr[t_local, 1])
                                tr, ti = apply_complex_tap(bx, by, kr, ki)
                                pr = cutlass.Float32(s_phase[t, 0])
                                pi = cutlass.Float32(s_phase[t, 1])
                                kx, ky = conj_mul_phase(tr, ti, pr, pi)
                                sK_tile[t_local, d0 + 0] = safe_cast_to_dtype(
                                    kx, mU.element_type
                                )
                                sK_tile[t_local, d0 + 1] = safe_cast_to_dtype(
                                    ky, mU.element_type
                                )
                    cute.arch.barrier()

                    for it in range(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = cutlass.Int32(m0) + t_local
                            if t < cutlass.Int32(self.L):
                                d0 = vv * 2
                                x = cutlass.Float32(sQ_m[t_local, d0 + 0])
                                y = cutlass.Float32(sQ_m[t_local, d0 + 1])
                                pr = cutlass.Float32(s_phase[t, 0])
                                pi = cutlass.Float32(s_phase[t, 1])
                                rx, ry = conj_mul_phase(x, y, pr, pi)
                                sQ_m[t_local, d0 + 0] = safe_cast_to_dtype(
                                    rx, mU.element_type
                                )
                                sQ_m[t_local, d0 + 1] = safe_cast_to_dtype(
                                    ry, mU.element_type
                                )
                    cute.arch.barrier()

                    tSrQ_m = thr_mma.make_fragment_A(thr_mma.partition_A(sQ_m))
                    tSsQ_m = thr_copy_A.partition_S(sQ_m)
                    tSrQ_m_view = thr_copy_A.retile(tSrQ_m)
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsQ_m[None, None, 0],
                        tSrQ_m_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tSsK_tile[None, None, 0],
                        tSrK_tile_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsQ_m.shape[2])):
                        k_next = (k + 1) % cute.size(tSsQ_m.shape[2])
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsQ_m[None, None, k_next],
                            tSrQ_m_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tSsK_tile[None, None, k_next],
                            tSrK_tile_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_blk,
                            tSrQ_m[None, None, k],
                            tSrK_tile[None, None, k],
                            acc_blk,
                        )

                acc_S_blk_mn = self._make_acc_tensor_mn_view(acc_blk)
                for r in cutlass.range_constexpr(cute.size(acc_S_blk_mn.shape[0])):
                    row_idx = cutlass.Int32(tScS_blk_mn[r, 0][1])
                    row_local = row_idx - cutlass.Int32(m0)
                    rs = cutlass.Float32(s_row_scale[row_idx])
                    for c in cutlass.range_constexpr(cute.size(acc_S_blk_mn.shape[1])):
                        col_idx = cutlass.Int32(tScS_blk_mn[0, c][3])
                        col_local = col_idx - cutlass.Int32(n0)
                        inv_rs = cutlass.Float32(s_inv_row_scale[col_idx])
                        sc_f32 = cutlass.Float32(0.0)
                        if cute.elem_less(col_idx, row_idx + 1):
                            sc_f32 = acc_S_blk_mn[r, c] * (rs * inv_rs)
                        sS_blk[col_local, row_local] = safe_cast_to_dtype(
                            sc_f32, mU.element_type
                        )
                cute.arch.barrier()

                for p_tile_idx in cutlass.range_constexpr(n_p_tiles):
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
                        for vi in cutlass.range_constexpr(cute.size(tDYs0.shape[1])):
                            if cute.elem_less(tDYc[0, vi, 0][1], mDOut.layout.shape[1]):
                                cute.copy(
                                    gmem_tiled_copy_P,
                                    tDYg[None, vi, None],
                                    tDYs0[None, vi, None],
                                    pred=tDYp_s[None, vi, None],
                                )
                            else:
                                tDYs0[None, vi, None].fill(0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    sDYt_m = cute.composition(sDY_m, sDYt_layout)
                    tSrDYt_m = thr_mma.make_fragment_B(thr_mma.partition_B(sDYt_m))
                    tSsDYt_m = thr_copy_BT.partition_S(sDYt_m)
                    tSrDYt_m_view = thr_copy_BT.retile(tSrDYt_m)
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsS_blk[None, None, 0],
                        tSrS_blk_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsDYt_m[None, None, 0],
                        tSrDYt_m_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsS_blk.shape[2])):
                        k_next = (k + 1) % cute.size(tSsS_blk.shape[2])
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_blk[None, None, k_next],
                            tSrS_blk_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsDYt_m[None, None, k_next],
                            tSrDYt_m_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dV_tiles[p_tile_idx],
                            tSrS_blk[None, None, k],
                            tSrDYt_m[None, None, k],
                            acc_dV_tiles[p_tile_idx],
                        )

            acc_dV_prev_tiles = []
            for _ in cutlass.range_constexpr(n_p_tiles):
                acc = cute.make_rmem_tensor(acc_shape_tileP, self.acc_dtype)
                acc.fill(0.0)
                acc_dV_prev_tiles.append(acc)

            for mi in cutlass.range_constexpr(m_tiles):
                m_tile = n_tile + mi
                m0 = m_tile * kv_tile
                use_buf1 = False
                sDY_m = sDY1 if use_buf1 else sDY0
                sQ_m = sQ1 if use_buf1 else sQ0

                cS_blk = cute.local_tile(mcS_full, (kv_tile, kv_tile), (m_tile, n_tile))
                tScS_blk = thr_mma.partition_C(cS_blk)
                tScS_blk_mn = self._make_acc_tensor_mn_view(tScS_blk)
                acc_blk = cute.make_rmem_tensor(acc_shape_blk, self.acc_dtype)
                acc_blk.fill(0.0)

                cC_row = cute.local_tile(
                    mcC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, 0)
                )
                tCc_row = gmem_thr_copy_D.partition_S(cC_row)

                for d_stage_idx in cutlass.range_constexpr(d_stage_count):
                    d_col_base = cutlass.Int32(d_stage_idx * Ds)
                    for rest_v in cutlass.range_constexpr(tQp.shape[0]):
                        for rest_k in cutlass.range_constexpr(tQp.shape[2]):
                            g_col = (
                                cutlass.Int32(tCc0[(0, rest_v), 0, rest_k][3])
                                + d_col_base
                            )
                            tQp[rest_v, 0, rest_k] = cute.elem_less(
                                g_col, mC.layout.shape[3]
                            )

                    gB = cute.local_tile(
                        mB[bidz, None, 0, None], (kv_tile, Ds), (n_tile, d_stage_idx)
                    )
                    gB = cute.domain_offset((-1, 0), gB)
                    gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
                    tBg = gmem_thr_copy_D.partition_S(gB)
                    cB = cute.local_tile(
                        mcKD_full, (kv_tile, Ds), (n_tile, d_stage_idx)
                    )
                    cB = cute.domain_offset((-1, 0), cB)
                    tBc = gmem_thr_copy_D.partition_S(cB)
                    for di in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        row_idx = cutlass.Int32(tBc[0, di, 0][1])
                        if cute.elem_less(
                            cutlass.Int32(-1), row_idx
                        ) and cute.elem_less(row_idx, mB.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_copy_D,
                                tBg[None, di, None],
                                tKsK[None, di, None],
                                pred=tQp[None, di, None],
                            )
                        else:
                            tKsK[None, di, None].fill(0)

                    gC = cute.local_tile(
                        mC[bidz, None, 0, None], (kv_tile, Ds), (m_tile, d_stage_idx)
                    )
                    tCg = gmem_thr_copy_D.partition_S(gC)
                    for di in cutlass.range_constexpr(cute.size(tQsQ0.shape[1])):
                        if cute.elem_less(tCc_row[0, di, 0][1], mC.layout.shape[1]):
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
                    if n0 == 0:
                        for it in range(iters_ds):
                            d_local = tidx + cutlass.Int32(it * self.num_threads)
                            if d_local < cutlass.Int32(Ds):
                                d = d_col_base + d_local
                                val = cutlass.Float32(0.0).to(mU.element_type)
                                if d < cutlass.Int32(self.D):
                                    if chunk == cutlass.Int32(0):
                                        val = mB_prev0[bh, d]
                                    else:
                                        val = mB[
                                            bidz - cutlass.Int32(1),
                                            cutlass.Int32(self.L - 1),
                                            0,
                                            d,
                                        ]
                                sK_tile[0, d_local] = val
                    cute.arch.barrier()

                    for it in range(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = t_local + cutlass.Int32(n0)
                            if t < cutlass.Int32(self.L):
                                d0 = vv * 2
                                bx = cutlass.Float32(
                                    sK_tile[t_local, d0 + 0].to(cutlass.Float32)
                                )
                                by = cutlass.Float32(
                                    sK_tile[t_local, d0 + 1].to(cutlass.Float32)
                                )
                                kr = cutlass.Float32(s_tap_prev[t, 0])
                                ki = cutlass.Float32(s_tap_prev[t, 1])
                                tr, ti = apply_complex_tap(bx, by, kr, ki)
                                pr = cutlass.Float32(s_phase[t, 0])
                                pi = cutlass.Float32(s_phase[t, 1])
                                kx, ky = conj_mul_phase(tr, ti, pr, pi)
                                sK_tile[t_local, d0 + 0] = safe_cast_to_dtype(
                                    kx, mU.element_type
                                )
                                sK_tile[t_local, d0 + 1] = safe_cast_to_dtype(
                                    ky, mU.element_type
                                )
                    cute.arch.barrier()

                    for it in range(iters_pairs_stage):
                        idx = tidx + cutlass.Int32(it * self.num_threads)
                        if idx < cutlass.Int32(total_pairs_stage):
                            t_local = idx // nvec_stage
                            vv = idx - t_local * nvec_stage
                            t = cutlass.Int32(m0) + t_local
                            if t < cutlass.Int32(self.L):
                                d0 = vv * 2
                                x = cutlass.Float32(sQ_m[t_local, d0 + 0])
                                y = cutlass.Float32(sQ_m[t_local, d0 + 1])
                                pr = cutlass.Float32(s_phase[t, 0])
                                pi = cutlass.Float32(s_phase[t, 1])
                                rx, ry = conj_mul_phase(x, y, pr, pi)
                                sQ_m[t_local, d0 + 0] = safe_cast_to_dtype(
                                    rx, mU.element_type
                                )
                                sQ_m[t_local, d0 + 1] = safe_cast_to_dtype(
                                    ry, mU.element_type
                                )
                    cute.arch.barrier()

                    tSrQ_m = thr_mma.make_fragment_A(thr_mma.partition_A(sQ_m))
                    tSsQ_m = thr_copy_A.partition_S(sQ_m)
                    tSrQ_m_view = thr_copy_A.retile(tSrQ_m)
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsQ_m[None, None, 0],
                        tSrQ_m_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tSsK_tile[None, None, 0],
                        tSrK_tile_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsQ_m.shape[2])):
                        k_next = (k + 1) % cute.size(tSsQ_m.shape[2])
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsQ_m[None, None, k_next],
                            tSrQ_m_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tSsK_tile[None, None, k_next],
                            tSrK_tile_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_blk,
                            tSrQ_m[None, None, k],
                            tSrK_tile[None, None, k],
                            acc_blk,
                        )

                acc_S_blk_mn = self._make_acc_tensor_mn_view(acc_blk)
                for r in cutlass.range_constexpr(cute.size(acc_S_blk_mn.shape[0])):
                    row_idx = cutlass.Int32(tScS_blk_mn[r, 0][1])
                    row_local = row_idx - cutlass.Int32(m0)
                    rs = cutlass.Float32(s_row_scale[row_idx])
                    for c in cutlass.range_constexpr(cute.size(acc_S_blk_mn.shape[1])):
                        col_idx = cutlass.Int32(tScS_blk_mn[0, c][3])
                        col_local = col_idx - cutlass.Int32(n0)
                        inv_rs = cutlass.Float32(s_inv_row_scale[col_idx])
                        sc_f32 = cutlass.Float32(0.0)
                        if cute.elem_less(col_idx, row_idx + 1):
                            sc_f32 = acc_S_blk_mn[r, c] * (rs * inv_rs)
                        sS_blk[col_local, row_local] = safe_cast_to_dtype(
                            sc_f32, mU.element_type
                        )
                cute.arch.barrier()

                for p_tile_idx in cutlass.range_constexpr(n_p_tiles):
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
                        for vi in cutlass.range_constexpr(cute.size(tDYs0.shape[1])):
                            if cute.elem_less(tDYc[0, vi, 0][1], mDOut.layout.shape[1]):
                                cute.copy(
                                    gmem_tiled_copy_P,
                                    tDYg[None, vi, None],
                                    tDYs0[None, vi, None],
                                    pred=tDYp_s[None, vi, None],
                                )
                            else:
                                tDYs0[None, vi, None].fill(0)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.barrier()
                    sDYt_m = cute.composition(sDY_m, sDYt_layout)
                    tSrDYt_m = thr_mma.make_fragment_B(thr_mma.partition_B(sDYt_m))
                    tSsDYt_m = thr_copy_BT.partition_S(sDYt_m)
                    tSrDYt_m_view = thr_copy_BT.retile(tSrDYt_m)
                    cute.copy(
                        smem_tiled_copy_A,
                        tSsS_blk[None, None, 0],
                        tSrS_blk_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_BT,
                        tSsDYt_m[None, None, 0],
                        tSrDYt_m_view[None, None, 0],
                    )
                    for k in cutlass.range_constexpr(cute.size(tSsS_blk.shape[2])):
                        k_next = (k + 1) % cute.size(tSsS_blk.shape[2])
                        cute.copy(
                            smem_tiled_copy_A,
                            tSsS_blk[None, None, k_next],
                            tSrS_blk_view[None, None, k_next],
                        )
                        cute.copy(
                            smem_tiled_copy_BT,
                            tSsDYt_m[None, None, k_next],
                            tSrDYt_m_view[None, None, k_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            acc_dV_prev_tiles[p_tile_idx],
                            tSrS_blk[None, None, k],
                            tSrDYt_m[None, None, k],
                            acc_dV_prev_tiles[p_tile_idx],
                        )

            gmem_thr_store_U = gmem_tiled_store_P.get_slice(tidx)
            mcDU = cute.make_identity_tensor(mDU.layout.shape)
            iters_p_slice = (p_tile + self.num_threads - 1) // self.num_threads
            for p_tile_idx in cutlass.range_constexpr(n_p_tiles):
                p_base = cutlass.Int32(p_tile_idx * p_tile)
                cU_tile = cute.local_tile(
                    mcU_full, (kv_tile, p_tile), (n_tile, p_tile_idx)
                )
                tOcU_tile = thr_mma.partition_C(cU_tile)
                tOcU_tile_mn = self._make_acc_tensor_mn_view(tOcU_tile)
                sPrev_tile = sV_tile
                acc_prev_mn = self._make_acc_tensor_mn_view(
                    acc_dV_prev_tiles[p_tile_idx]
                )
                for r in cutlass.range_constexpr(cute.size(acc_prev_mn.shape[0])):
                    for c in cutlass.range_constexpr(cute.size(acc_prev_mn.shape[1])):
                        row_idx = cutlass.Int32(tOcU_tile_mn[r, c][1])
                        col_idx = cutlass.Int32(tOcU_tile_mn[r, c][3])
                        row_local = row_idx - cutlass.Int32(n0)
                        col_local = col_idx - p_base
                        if cute.elem_less(col_idx, cutlass.Int32(self.P)):
                            sPrev_tile[row_local, col_local] = acc_prev_mn[r, c].to(
                                mU.element_type
                            )
                cute.arch.barrier()

                acc_curr_mn = self._make_acc_tensor_mn_view(acc_dV_tiles[p_tile_idx])
                for r in cutlass.range_constexpr(cute.size(acc_curr_mn.shape[0])):
                    for c in cutlass.range_constexpr(cute.size(acc_curr_mn.shape[1])):
                        row_idx = cutlass.Int32(tOcU_tile_mn[r, c][1])
                        col_idx = cutlass.Int32(tOcU_tile_mn[r, c][3])
                        row_local = row_idx - cutlass.Int32(n0)
                        tp = row_idx + cutlass.Int32(1)
                        col_local = col_idx - p_base
                        if cute.elem_less(col_idx, cutlass.Int32(self.P)):
                            prev_f32 = cutlass.Float32(0.0)
                            if tp < cutlass.Int32(self.L):
                                if row_local < cutlass.Int32(kv_tile - 1):
                                    prev_f32 = cutlass.Float32(
                                        sPrev_tile[
                                            row_local + cutlass.Int32(1), col_local
                                        ].to(cutlass.Float32)
                                    )
                                else:
                                    prev_f32 = cutlass.Float32(sDV_carry[col_idx])
                            acc_curr_mn[r, c] = acc_curr_mn[r, c] + prev_f32.to(
                                self.acc_dtype
                            )
                cute.arch.barrier()
                for it in range(iters_p_slice):
                    p_local = tidx + cutlass.Int32(it * self.num_threads)
                    if p_local < cutlass.Int32(p_tile):
                        sDV_carry[p_base + p_local] = cutlass.Float32(
                            sPrev_tile[0, p_local].to(cutlass.Float32)
                        )
                cute.arch.barrier()

                if cutlass.const_expr(self.P == Pp):
                    tCsV = thr_mma.partition_C(sV_tile)
                    tCrOut = cute.make_fragment_like(tCsV, mU.element_type)
                    acc_dV_vals = acc_dV_tiles[p_tile_idx].load()
                    for i in cutlass.range_constexpr(cute.size(tCrOut)):
                        tCrOut[i] = safe_cast_to_dtype(acc_dV_vals[i], mU.element_type)
                    cute.autovec_copy(tCrOut, tCsV)
                else:
                    for r in cutlass.range_constexpr(cute.size(acc_curr_mn.shape[0])):
                        for c in cutlass.range_constexpr(
                            cute.size(acc_curr_mn.shape[1])
                        ):
                            row_idx = cutlass.Int32(tOcU_tile_mn[r, c][1])
                            col_idx = cutlass.Int32(tOcU_tile_mn[r, c][3])
                            row_local = row_idx - cutlass.Int32(n0)
                            col_local = col_idx - p_base
                            if cute.elem_less(col_idx, cutlass.Int32(self.P)):
                                curr_f32 = cutlass.Float32(acc_curr_mn[r, c])
                                sV_tile[row_local, col_local] = curr_f32.to(
                                    mU.element_type
                                )
                cute.arch.barrier()

                gDU = cute.local_tile(
                    mDU[bidz, None, 0, None], (kv_tile, p_tile), (n_tile, p_tile_idx)
                )
                tUsV = gmem_thr_store_U.partition_S(sV_tile)
                tUgDU = gmem_thr_store_U.partition_D(gDU)
                if cutlass.const_expr(self.P == Pp):
                    cute.copy(gmem_tiled_store_P, tUsV, tUgDU)
                else:
                    tUrU = cute.make_rmem_tensor_like(tUgDU, mU.element_type)
                    cute.copy(gmem_tiled_store_P, tUsV, tUrU)
                    cDU = cute.local_tile(
                        mcDU[bidz, None, 0, None],
                        (kv_tile, p_tile),
                        (n_tile, p_tile_idx),
                    )
                    tUcDU = gmem_thr_store_U.partition_D(cDU)
                    tUpDU = cute.make_rmem_tensor(
                        cute.make_layout(
                            (tUgDU.shape[0][1], tUgDU.shape[1], tUgDU.shape[2]),
                            stride=(tUgDU.shape[2], 0, 1),
                        ),
                        cutlass.Boolean,
                    )
                    for rest_v in cutlass.range_constexpr(tUpDU.shape[0]):
                        for rest_n in cutlass.range_constexpr(
                            cute.size(tUpDU.shape[2])
                        ):
                            tUpDU[rest_v, 0, rest_n] = cute.elem_less(
                                tUcDU[(0, rest_v), 0, rest_n][3], mDU.layout.shape[3]
                            )
                    for rest_m in cutlass.range_constexpr(cute.size(tUpDU.shape[1])):
                        if cute.elem_less(tUcDU[0, rest_m, 0][1], mDU.layout.shape[1]):
                            cute.copy(
                                gmem_tiled_store_P,
                                tUrU[None, rest_m, None],
                                tUgDU[None, rest_m, None],
                                pred=tUpDU[None, rest_m, None],
                            )
                if cutlass.const_expr(n_p_tiles > 1):
                    if cutlass.const_expr(p_tile_idx + 1 < n_p_tiles):
                        cute.arch.barrier()

        cute.arch.barrier()
        iters_p = (self.P + self.num_threads - 1) // self.num_threads
        for it in range(iters_p):
            p = tidx + cutlass.Int32(it * self.num_threads)
            if p < cutlass.Int32(self.P):
                mDU_prev[bidz, p] = safe_cast_to_dtype(
                    sDV_carry[p], mDU_prev.element_type
                )
        cute.arch.barrier()


__all__ = ["ChunkScanBwdDUAmpere"]
