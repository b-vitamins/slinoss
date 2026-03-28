"""CuTe backward ``dr`` post-pass for the ``v2x2ssd`` chunk-scan stage.

This kernel is intentionally lightweight and owns the low-arithmetic postprocessing
slice that follows the tensor-core ``dc`` contraction:

- reconstruct per-step unit-complex prefix phase from packed ``M``
- recover pre-transport ``dQ`` rows from public ``dC``
- accumulate 2x2 phase-matrix partials ``dR``

``dlogprefix`` is produced by the dedicated ``dlp`` kernel.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from .common import complex_mul, conj_mul_phase


def _make_shared_storage(phase_layout: cute.Layout):
    class SharedStorage:
        phase: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(phase_layout)],
            16,
        ]

    return cute.struct(SharedStorage)


class ChunkScanBwdDRAmpere:
    """Post-pass kernel for ``dR`` after the chunk-scan backward ``dc`` contraction."""

    def __init__(self, dtype, *, chunk_size, D, num_threads=128):
        self.ab_dtype = dtype
        self.L = int(chunk_size)
        self.D = int(D)
        self.num_threads = int(num_threads)
        if self.L <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.D % 2 != 0:
            raise ValueError("D must be divisible by 2 (flattened 2N).")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDC: cute.Tensor,
        mDR: cute.Tensor,
    ):
        if cutlass.const_expr(mC.element_type != mDC.element_type):
            raise TypeError("C and dC must share dtype.")
        if cutlass.const_expr(
            mC.element_type not in (cutlass.Float16, cutlass.BFloat16)
        ):
            raise TypeError("C and dC must be Float16/BFloat16.")
        if cutlass.const_expr(mM.element_type != cutlass.Float32):
            raise TypeError("M must be Float32.")
        if cutlass.const_expr(mDR.element_type != cutlass.Float32):
            raise TypeError("dR must be Float32.")

        if cutlass.const_expr(mC.shape[1] != self.L or mC.shape[2] != 1):
            raise ValueError("C must be (BHC, L, 1, D).")
        if cutlass.const_expr(mDC.shape[1] != self.L or mDC.shape[2] != 1):
            raise ValueError("dC must be (BHC, L, 1, D).")
        if cutlass.const_expr(mM.shape[1] != self.L or mM.shape[2] != 2):
            raise ValueError("M must be (BHC, L, 2).")
        if cutlass.const_expr(mDR.shape[1] != self.L or mDR.shape[2] != 4):
            raise ValueError("dR must be (BHC, L, 4).")
        if cutlass.const_expr(
            mC.shape[0] != mM.shape[0] or mC.shape[0] != mDR.shape[0]
        ):
            raise ValueError("C/M/dR must agree on BHC.")
        if cutlass.const_expr(mC.shape[3] != self.D or mDC.shape[3] != self.D):
            raise ValueError("C and dC must match D.")

        grid_z = cute.size(mC.shape[0])
        phase_layout = cute.make_layout((self.L, 2), stride=(2, 1))
        smem_bytes = int(_make_shared_storage(phase_layout).size_in_bytes())

        self.kernel(
            mC,
            mM,
            mDC,
            mDR,
            phase_layout,
        ).launch(
            grid=(1, 1, grid_z),
            block=[self.num_threads, 1, 1],
            smem=smem_bytes,
        )

    @cute.kernel
    def kernel(
        self,
        mC: cute.Tensor,
        mM: cute.Tensor,
        mDC: cute.Tensor,
        mDR: cute.Tensor,
        phase_layout: cute.Layout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        num_warps = self.num_threads // 32
        nvec = cutlass.Int32(self.D // 2)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_make_shared_storage(phase_layout))
        s_phase = storage.phase.get_tensor(phase_layout)

        eps = cutlass.Float32(1.0e-20)
        one = cutlass.Float32(1.0)
        zero = cutlass.Float32(0.0)

        if tidx == cutlass.Int32(0):
            pref_r = one
            pref_i = zero
            for t in cutlass.range(self.L, unroll=1):
                mr = cutlass.Float32(mM[bidz, t, 0])
                mi = cutlass.Float32(mM[bidz, t, 1])
                mag2 = mr * mr + mi * mi + eps
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2, fastmath=True))
                ur = mr * inv_mag
                ui = mi * inv_mag
                pref_r, pref_i = complex_mul(ur, ui, pref_r, pref_i)
                n2 = pref_r * pref_r + pref_i * pref_i + eps
                inv_n2 = cutlass.Float32(cute.math.rsqrt(n2, fastmath=True))
                pref_r = pref_r * inv_n2
                pref_i = pref_i * inv_n2
                s_phase[t, 0] = pref_r
                s_phase[t, 1] = pref_i
        cute.arch.barrier()

        t_local = warp
        while cute.elem_less(t_local, cutlass.Int32(self.L)):
            dR00 = zero
            dR01 = zero
            dR10 = zero
            dR11 = zero

            pr = cutlass.Float32(s_phase[t_local, 0])
            pi = cutlass.Float32(s_phase[t_local, 1])

            vv = lane
            while cute.elem_less(vv, nvec):
                d0 = vv * cutlass.Int32(2)
                dc0 = cutlass.Float32(mDC[bidz, t_local, 0, d0 + 0])
                dc1 = cutlass.Float32(mDC[bidz, t_local, 0, d0 + 1])
                dq0, dq1 = conj_mul_phase(dc0, dc1, pr, pi)

                c0 = cutlass.Float32(mC[bidz, t_local, 0, d0 + 0])
                c1 = cutlass.Float32(mC[bidz, t_local, 0, d0 + 1])
                dR00 = dR00 + dq0 * c0
                dR01 = dR01 + dq0 * c1
                dR10 = dR10 + dq1 * c0
                dR11 = dR11 + dq1 * c1
                vv = vv + cutlass.Int32(32)

            for off in (16, 8, 4, 2, 1):
                dR00 = dR00 + cute.arch.shuffle_sync_bfly(
                    dR00, offset=off, mask=-1, mask_and_clamp=31
                )
                dR01 = dR01 + cute.arch.shuffle_sync_bfly(
                    dR01, offset=off, mask=-1, mask_and_clamp=31
                )
                dR10 = dR10 + cute.arch.shuffle_sync_bfly(
                    dR10, offset=off, mask=-1, mask_and_clamp=31
                )
                dR11 = dR11 + cute.arch.shuffle_sync_bfly(
                    dR11, offset=off, mask=-1, mask_and_clamp=31
                )

            if lane == cutlass.Int32(0):
                mDR[bidz, t_local, 0] = dR00
                mDR[bidz, t_local, 1] = dR01
                mDR[bidz, t_local, 2] = dR10
                mDR[bidz, t_local, 3] = dR11

            t_local = t_local + cutlass.Int32(num_warps)


__all__ = ["ChunkScanBwdDRAmpere"]
