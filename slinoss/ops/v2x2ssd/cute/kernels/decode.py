# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Standalone CuTe decode kernel for the recurrent SLinOSS middle."""

from __future__ import annotations

import math
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from slinoss.ops.scanprep.cute.common import (
    SCANPREP_PARAM_DIM,
    complex_div,
    complex_mul,
    lerp,
    principal_angle,
    real_mul_conj,
    sigmoid,
    softplus,
)


class MixerDecodeStepFwd:
    """One-token recurrent middle for SLinOSS decode.

    Contract:
    - ``value`` / ``gate``: ``(B, H, P)``
    - ``params``: ``(B, H, 5)``
    - ``bc``: ``(B, H, 4, N)``
    - ``skip``: ``(H, P)``
    - ``state``: ``(B, H, P, 2N)``
    - ``b_prev``: ``(B, H, 2N)``
    - ``u_prev``: ``(B, H, P)``
    - outputs:
      - ``y``: ``(B, H, P)``
      - ``final_state``: ``(B, H, P, 2N)``
      - ``b_last``: ``(B, H, 2N)``
      - ``u_last``: ``(B, H, P)``
    """

    def __init__(
        self,
        *,
        heads: int,
        p_size: int,
        n_size: int,
        d_model: int,
        fuse_outproj: bool,
        state_align_bytes: int,
        u_prev_last_dim_contig: bool,
        tile_p: int,
        num_warps: int,
        vec_n: int,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        eps: float,
    ) -> None:
        self.heads = int(heads)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.d_size = int(2 * n_size)
        self.d_model = int(d_model)
        self.fuse_outproj = bool(fuse_outproj)

        self.tile_p = int(tile_p)
        self.num_warps = int(num_warps)
        self.num_threads = int(self.num_warps * cute.arch.WARP_SIZE)
        self.vec_n = int(vec_n)
        self.state_align_bytes = int(state_align_bytes)
        self.u_prev_last_dim_contig = bool(u_prev_last_dim_contig)
        self.use_state_cp_async = bool(
            self.state_align_bytes >= 16
            and self.tile_p >= 64
            and os.getenv("SLINOSS_MIXER_DECODE_CPASYNC", "0") == "1"
        )

        if self.tile_p < 1 or self.tile_p > self.p_size:
            raise ValueError(
                f"tile_p must be in [1, {self.p_size}], got {self.tile_p}."
            )
        if self.p_size % self.tile_p != 0:
            raise ValueError(
                f"tile_p={self.tile_p} must divide P={self.p_size} for decode."
            )
        if self.num_threads < self.tile_p or self.num_threads % self.tile_p != 0:
            raise ValueError(
                "num_warps must provide a whole number of thread groups per P-tile."
            )
        if self.vec_n < 1 or self.n_size % self.vec_n != 0:
            raise ValueError(
                f"vec_n must be positive and divide N={self.n_size}. Got {self.vec_n}."
            )
        if self.fuse_outproj and self.d_model < 1:
            raise ValueError(
                "d_model must be positive when fused out projection is on."
            )

        self.n_groups = int(self.num_threads // self.tile_p)
        self.n_vec_iters = int(
            (self.n_size + (self.n_groups * self.vec_n) - 1)
            // (self.n_groups * self.vec_n)
        )
        self.p_tiles = int((self.p_size + self.tile_p - 1) // self.tile_p)

        self.state_tile_elems = int(self.tile_p * self.d_size)
        self.out_proj_iters = int(
            (self.d_model + self.num_threads - 1) // self.num_threads
        )

        self.normalize_bc = bool(normalize_bc)

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    def _warp_reduce_sum(self, val: cutlass.Float32) -> cutlass.Float32:
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=16, mask=-1, mask_and_clamp=31
        )
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=8, mask=-1, mask_and_clamp=31
        )
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=4, mask=-1, mask_and_clamp=31
        )
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=2, mask=-1, mask_and_clamp=31
        )
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=1, mask=-1, mask_and_clamp=31
        )
        return val

    def _make_gmem_tiled_copy(
        self,
        dtype: type[cutlass.Numeric],
        major_mode_size: int,
        num_threads: int,
        *,
        is_async: bool,
    ) -> cute.TiledCopy:
        major_mode_size = int(major_mode_size)
        num_threads = int(num_threads)
        if major_mode_size <= 0 or num_threads <= 0:
            raise ValueError("major_mode_size and num_threads must be positive.")
        if is_async:
            max_copy_elems = max(1, int(128 // dtype.width))
            copy_elems = max(1, int(math.gcd(major_mode_size, max_copy_elems)))
            threads_per_row = max(1, int(major_mode_size // copy_elems))
        else:
            if num_threads < major_mode_size:
                # Keep copy atom scalar when the CTA has fewer lanes than the contiguous
                # state major mode; this avoids illegal vectorized gmem->smem layouts.
                threads_per_row = num_threads
                copy_elems = 1
            else:
                threads_per_row = math.gcd(major_mode_size, num_threads)
                if threads_per_row <= 0:
                    raise ValueError(
                        "Invalid copy partition: threads_per_row must be positive."
                    )
                copy_elems = max(1, int(major_mode_size // threads_per_row))
        copy_bits = int(copy_elems * dtype.width)
        if num_threads % threads_per_row != 0:
            raise ValueError(
                "num_threads must be divisible by threads_per_row for copy partitioning."
            )
        thread_layout = cute.make_ordered_layout(
            (num_threads // threads_per_row, threads_per_row),
            order=(1, 0),
        )
        value_layout = cute.make_layout((1, copy_elems))
        copy_op = (
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            )
            if is_async
            else cute.nvgpu.CopyUniversalOp()
        )
        copy_atom = cute.make_copy_atom(
            copy_op,
            dtype,
            num_bits_per_copy=copy_bits,
        )
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)

    @cute.jit
    def __call__(
        self,
        value: cute.Tensor,
        params: cute.Tensor,
        bc: cute.Tensor,
        gate: cute.Tensor,
        skip: cute.Tensor,
        state: cute.Tensor,
        b_prev: cute.Tensor,
        u_prev: cute.Tensor,
        dt_bias: cute.Tensor,
        gamma_bias: cute.Tensor,
        omega_bias: cute.Tensor,
        mix_r_bias: cute.Tensor,
        b_scale: cute.Tensor,
        c_scale: cute.Tensor,
        y: cute.Tensor,
        final_state: cute.Tensor,
        b_last: cute.Tensor,
        u_last: cute.Tensor,
        out_proj: cute.Tensor,
        projected: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch = cute.size(value, mode=[0])
        mValue = cute.make_tensor(
            value.iterator,
            cute.make_layout(
                (batch, self.heads, self.p_size),
                stride=(self.heads * self.p_size, self.p_size, 1),
            ),
        )
        mParams = cute.make_tensor(
            params.iterator,
            cute.make_layout(
                (batch, self.heads, SCANPREP_PARAM_DIM),
                stride=(
                    self.heads * SCANPREP_PARAM_DIM,
                    SCANPREP_PARAM_DIM,
                    1,
                ),
            ),
        )
        mBC = cute.make_tensor(
            bc.iterator,
            cute.make_layout(
                (batch, self.heads, 4, self.n_size),
                stride=(self.heads * 4 * self.n_size, 4 * self.n_size, self.n_size, 1),
            ),
        )
        mGate = cute.make_tensor(
            gate.iterator,
            cute.make_layout(
                (batch, self.heads, self.p_size),
                stride=(self.heads * self.p_size, self.p_size, 1),
            ),
        )
        mSkip = cute.make_tensor(
            skip.iterator,
            cute.make_layout((self.heads, self.p_size), stride=(self.p_size, 1)),
        )
        mState = state
        mBPrev = b_prev
        mUPrev = (
            cute.make_tensor(
                u_prev.iterator,
                cute.make_layout(
                    (batch, self.heads, self.p_size),
                    stride=(self.heads * self.p_size, self.p_size, 1),
                ),
            )
            if self.u_prev_last_dim_contig
            else u_prev
        )
        mDtBias = cute.make_tensor(
            dt_bias.iterator, cute.make_layout((self.heads,), stride=(1,))
        )
        mGammaBias = cute.make_tensor(
            gamma_bias.iterator, cute.make_layout((self.heads,), stride=(1,))
        )
        mOmegaBias = cute.make_tensor(
            omega_bias.iterator, cute.make_layout((self.heads,), stride=(1,))
        )
        mMixRBias = cute.make_tensor(
            mix_r_bias.iterator, cute.make_layout((self.heads,), stride=(1,))
        )
        mBScale = b_scale
        mCScale = c_scale
        mY = cute.make_tensor(
            y.iterator,
            cute.make_layout(
                (batch, self.heads, self.p_size),
                stride=(self.heads * self.p_size, self.p_size, 1),
            ),
        )
        mFinalState = final_state
        mBLast = b_last
        mULast = u_last
        mOutProj = cute.make_tensor(
            out_proj.iterator,
            cute.make_layout(
                (self.d_model if self.fuse_outproj else 1, self.heads, self.p_size),
                stride=(self.heads * self.p_size, self.p_size, 1),
            ),
        )
        mProjected = cute.make_tensor(
            projected.iterator,
            cute.make_layout(
                (
                    batch,
                    self.heads if self.fuse_outproj else 1,
                    self.d_model if self.fuse_outproj else 1,
                ),
                stride=(
                    (self.heads if self.fuse_outproj else 1)
                    * (self.d_model if self.fuse_outproj else 1),
                    self.d_model if self.fuse_outproj else 1,
                    1,
                ),
            ),
        )

        state_tile_layout = cute.make_ordered_layout(
            (self.d_size, self.tile_p),
            order=(1, 0),
        )
        p_layout = cute.make_layout((self.tile_p,))
        vec_layout = cute.make_layout((self.n_size,))
        bc_reduce_layout = cute.make_layout((4, self.num_warps))
        acc_layout = cute.make_layout((self.n_groups, self.tile_p))

        state_dtype = mState.element_type
        p_dtype = mValue.element_type
        skip_dtype = mSkip.element_type

        gmem_tiled_copy_state = self._make_gmem_tiled_copy(
            state_dtype,
            self.tile_p,
            self.num_threads,
            is_async=self.use_state_cp_async,
        )
        copy_elems_value = max(
            2, min(4, (self.tile_p + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE)
        )
        copy_bits_value = int(copy_elems_value * p_dtype.width)
        copy_atom_value = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            p_dtype,
            num_bits_per_copy=copy_bits_value,
        )
        gmem_tiled_copy_value = cute.make_tiled_copy_tv(
            copy_atom_value,
            cute.make_layout(self.tile_p // copy_elems_value),
            cute.make_layout(copy_elems_value),
        )
        copy_elems_u_prev = copy_elems_value if self.u_prev_last_dim_contig else 1
        copy_bits_u_prev = int(copy_elems_u_prev * p_dtype.width)
        copy_atom_u_prev = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            p_dtype,
            num_bits_per_copy=copy_bits_u_prev,
        )
        gmem_tiled_copy_u_prev = cute.make_tiled_copy_tv(
            copy_atom_u_prev,
            cute.make_layout(self.tile_p // copy_elems_u_prev),
            cute.make_layout(copy_elems_u_prev),
        )
        copy_elems_skip = 1
        copy_bits_skip = int(copy_elems_skip * skip_dtype.width)
        copy_atom_skip = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            skip_dtype,
            num_bits_per_copy=copy_bits_skip,
        )
        gmem_tiled_copy_skip = cute.make_tiled_copy_tv(
            copy_atom_skip,
            cute.make_layout(self.tile_p // copy_elems_skip),
            cute.make_layout(copy_elems_skip),
        )

        @cute.struct
        class SharedStorage:
            sState: cute.struct.Align[
                cute.struct.MemRange[state_dtype, cute.cosize(state_tile_layout)],
                1024,
            ]
            sValue: cute.struct.Align[
                cute.struct.MemRange[p_dtype, cute.cosize(p_layout)],
                128,
            ]
            sUPrev: cute.struct.Align[
                cute.struct.MemRange[p_dtype, cute.cosize(p_layout)],
                128,
            ]
            sGate: cute.struct.Align[
                cute.struct.MemRange[p_dtype, cute.cosize(p_layout)],
                128,
            ]
            sSkip: cute.struct.Align[
                cute.struct.MemRange[skip_dtype, cute.cosize(p_layout)],
                128,
            ]
            sY: cute.struct.Align[
                cute.struct.MemRange[p_dtype, cute.cosize(p_layout)],
                128,
            ]
            b_re: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            b_im: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            c_re: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            c_im: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            beta_prev_re: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            beta_prev_im: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            beta_curr_re: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            beta_curr_im: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(vec_layout)],
                128,
            ]
            inv: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 4], 16]
            bc_reduce: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(bc_reduce_layout)],
                32,
            ]
            acc_partial: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(acc_layout)],
                128,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            mValue,
            mParams,
            mBC,
            mGate,
            mSkip,
            mState,
            mBPrev,
            mUPrev,
            mDtBias,
            mGammaBias,
            mOmegaBias,
            mMixRBias,
            mBScale,
            mCScale,
            mY,
            mFinalState,
            mBLast,
            mULast,
            mOutProj,
            mProjected,
            state_tile_layout,
            p_layout,
            gmem_tiled_copy_state,
            gmem_tiled_copy_value,
            gmem_tiled_copy_u_prev,
            gmem_tiled_copy_skip,
        ).launch(
            grid=(self.p_tiles, self.heads, batch),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mValue: cute.Tensor,
        mParams: cute.Tensor,
        mBC: cute.Tensor,
        mGate: cute.Tensor,
        mSkip: cute.Tensor,
        mState: cute.Tensor,
        mBPrev: cute.Tensor,
        mUPrev: cute.Tensor,
        mDtBias: cute.Tensor,
        mGammaBias: cute.Tensor,
        mOmegaBias: cute.Tensor,
        mMixRBias: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mY: cute.Tensor,
        mFinalState: cute.Tensor,
        mBLast: cute.Tensor,
        mULast: cute.Tensor,
        mOutProj: cute.Tensor,
        mProjected: cute.Tensor,
        state_tile_layout: cute.Layout,
        p_layout: cute.Layout,
        gmem_tiled_copy_state: cute.TiledCopy,
        gmem_tiled_copy_value: cute.TiledCopy,
        gmem_tiled_copy_u_prev: cute.TiledCopy,
        gmem_tiled_copy_skip: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        bidp, bidh, bidb = cute.arch.block_idx()

        p_offset = bidp * self.tile_p

        # ------------------------------------------------------------------
        # CTA slice
        # ------------------------------------------------------------------
        gState = cute.local_tile(
            mState[bidb, bidh, None, None],
            (self.tile_p, self.d_size),
            (bidp, 0),
        )
        gFinalState = cute.local_tile(
            mFinalState[bidb, bidh, None, None],
            (self.tile_p, self.d_size),
            (bidp, 0),
        )
        gStateT = cute.composition(
            gState,
            cute.make_ordered_layout((self.d_size, self.tile_p), order=(1, 0)),
        )
        gFinalStateT = cute.composition(
            gFinalState,
            cute.make_ordered_layout((self.d_size, self.tile_p), order=(1, 0)),
        )
        gValue = cute.local_tile(mValue[bidb, bidh, None], (self.tile_p,), (bidp,))
        gUPrev = cute.local_tile(mUPrev[bidb, bidh, None], (self.tile_p,), (bidp,))
        gGate = cute.local_tile(mGate[bidb, bidh, None], (self.tile_p,), (bidp,))
        gSkip = cute.local_tile(mSkip[bidh, None], (self.tile_p,), (bidp,))
        gY = cute.local_tile(mY[bidb, bidh, None], (self.tile_p,), (bidp,))
        gULast = cute.local_tile(mULast[bidb, bidh, None], (self.tile_p,), (bidp,))

        # ------------------------------------------------------------------
        # Shared storage
        # ------------------------------------------------------------------
        vec_layout = cute.make_layout((self.n_size,))
        bc_reduce_layout = cute.make_layout((4, self.num_warps))
        acc_layout = cute.make_layout((self.n_groups, self.tile_p))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sState = storage.sState.get_tensor(state_tile_layout)
        sValue = storage.sValue.get_tensor(p_layout)
        sUPrev = storage.sUPrev.get_tensor(p_layout)
        sGate = storage.sGate.get_tensor(p_layout)
        sSkip = storage.sSkip.get_tensor(p_layout)
        sY = storage.sY.get_tensor(p_layout)

        b_re = storage.b_re.get_tensor(vec_layout)
        b_im = storage.b_im.get_tensor(vec_layout)
        c_re = storage.c_re.get_tensor(vec_layout)
        c_im = storage.c_im.get_tensor(vec_layout)
        beta_prev_re = storage.beta_prev_re.get_tensor(vec_layout)
        beta_prev_im = storage.beta_prev_im.get_tensor(vec_layout)
        beta_curr_re = storage.beta_curr_re.get_tensor(vec_layout)
        beta_curr_im = storage.beta_curr_im.get_tensor(vec_layout)

        inv_s = storage.inv.get_tensor(cute.make_layout((4,)))
        bc_reduce = storage.bc_reduce.get_tensor(bc_reduce_layout)
        acc_partial = storage.acc_partial.get_tensor(acc_layout)

        # ------------------------------------------------------------------
        # Copy partitioning + staged loads
        # ------------------------------------------------------------------
        thr_copy_state = gmem_tiled_copy_state.get_slice(tidx)
        tSgState = thr_copy_state.partition_S(gStateT)
        tSsState = thr_copy_state.partition_D(sState)
        cute.copy(gmem_tiled_copy_state, tSgState, tSsState)
        if self.use_state_cp_async:
            cute.arch.cp_async_commit_group()

        # Warp-specialized token-local loads (value/u_prev and gate/skip).
        thr_copy_value = gmem_tiled_copy_value.get_slice(lane_idx)
        tPgValue = thr_copy_value.partition_S(gValue)
        tPsValue = thr_copy_value.partition_D(sValue)
        tPgGate = thr_copy_value.partition_S(gGate)
        tPsGate = thr_copy_value.partition_D(sGate)
        thr_copy_u_prev = gmem_tiled_copy_u_prev.get_slice(lane_idx)
        tPgUPrev = thr_copy_u_prev.partition_S(gUPrev)
        tPsUPrev = thr_copy_u_prev.partition_D(sUPrev)

        load_elems_value = cute.size(tPgValue.shape[0][0])
        num_loads_value = self.tile_p // load_elems_value
        load_elems_u_prev = cute.size(tPgUPrev.shape[0][0])
        num_loads_u_prev = self.tile_p // load_elems_u_prev
        num_loads_skip = self.tile_p

        if warp_idx == 0:
            if lane_idx < num_loads_value:
                cute.copy(gmem_tiled_copy_value, tPgValue, tPsValue)
            if lane_idx < num_loads_u_prev:
                cute.copy(gmem_tiled_copy_u_prev, tPgUPrev, tPsUPrev)

        if warp_idx == 1:
            if lane_idx < num_loads_value:
                cute.copy(gmem_tiled_copy_value, tPgGate, tPsGate)
            skip_passes = (
                num_loads_skip + cute.arch.WARP_SIZE - 1
            ) // cute.arch.WARP_SIZE
            for skip_pass in cutlass.range_constexpr(skip_passes):
                skip_thread = lane_idx + skip_pass * cute.arch.WARP_SIZE
                if skip_thread < num_loads_skip:
                    thr_copy_skip = gmem_tiled_copy_skip.get_slice(skip_thread)
                    tPgSkip = thr_copy_skip.partition_S(gSkip)
                    tPsSkip = thr_copy_skip.partition_D(sSkip)
                    cute.copy(gmem_tiled_copy_skip, tPgSkip, tPsSkip)

        cute.arch.sync_threads()

        # ------------------------------------------------------------------
        # Parameter transform
        # ------------------------------------------------------------------
        rho_re = cutlass.Float32(0.0)
        rho_im = cutlass.Float32(0.0)
        tap_prev_re = cutlass.Float32(0.0)
        tap_prev_im = cutlass.Float32(0.0)
        tap_curr_re = cutlass.Float32(0.0)
        tap_curr_im = cutlass.Float32(0.0)

        if lane_idx == 0:
            dt_raw = cutlass.Float32(mParams[bidb, bidh, 0]) + cutlass.Float32(
                mDtBias[bidh]
            )
            gamma_raw = cutlass.Float32(mParams[bidb, bidh, 1]) + cutlass.Float32(
                mGammaBias[bidh]
            )
            omega_raw = cutlass.Float32(mParams[bidb, bidh, 2]) + cutlass.Float32(
                mOmegaBias[bidh]
            )
            r_raw = cutlass.Float32(mParams[bidb, bidh, 3])
            mix_r_raw = cutlass.Float32(mParams[bidb, bidh, 4]) + cutlass.Float32(
                mMixRBias[bidh]
            )

            dt_u = sigmoid(dt_raw)
            gamma = softplus(gamma_raw)
            omega = omega_raw
            r_direct_u = sigmoid(r_raw)
            mix_r = sigmoid(mix_r_raw)

            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            r_struct = cutlass.Float32(self.r_min) + cutlass.Float32(
                self.r_scale
            ) * cute_math.exp(-(gamma * dt))
            theta = principal_angle(omega * dt)
            r_direct = (
                cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * r_direct_u
            )

            r = lerp(r_direct, r_struct, mix_r)
            rho_re = r * cute_math.cos(theta)
            rho_im = r * cute_math.sin(theta)

            log_r = cute_math.log(r)
            z_re = log_r
            z_im = theta
            z2_re = z_re * z_re - z_im * z_im
            z2_im = cutlass.Float32(2.0) * z_re * z_im
            z_norm_sq = z_re * z_re + z_im * z_im

            kappa1_re = cutlass.Float32(0.0)
            kappa1_im = cutlass.Float32(0.0)
            kappa2_re = cutlass.Float32(0.0)
            kappa2_im = cutlass.Float32(0.0)

            if z_norm_sq < cutlass.Float32(self.z_thresh_sq):
                z3_re = z2_re * z_re - z2_im * z_im
                z3_im = z2_re * z_im + z2_im * z_re
                kappa1_re = (
                    cutlass.Float32(1.0)
                    + cutlass.Float32(0.5) * z_re
                    + z2_re / cutlass.Float32(6.0)
                    + z3_re / cutlass.Float32(24.0)
                )
                kappa1_im = (
                    cutlass.Float32(0.5) * z_im
                    + z2_im / cutlass.Float32(6.0)
                    + z3_im / cutlass.Float32(24.0)
                )
                kappa2_re = (
                    cutlass.Float32(0.5)
                    + z_re / cutlass.Float32(3.0)
                    + z2_re / cutlass.Float32(8.0)
                    + z3_re / cutlass.Float32(30.0)
                )
                kappa2_im = (
                    z_im / cutlass.Float32(3.0)
                    + z2_im / cutlass.Float32(8.0)
                    + z3_im / cutlass.Float32(30.0)
                )
            else:
                kappa1_re, kappa1_im = complex_div(
                    rho_re - cutlass.Float32(1.0),
                    rho_im,
                    z_re,
                    z_im,
                )
                num2_re = (
                    rho_re * (z_re - cutlass.Float32(1.0))
                    - rho_im * z_im
                    + cutlass.Float32(1.0)
                )
                num2_im = rho_re * z_im + rho_im * (z_re - cutlass.Float32(1.0))
                kappa2_re, kappa2_im = complex_div(num2_re, num2_im, z2_re, z2_im)

            k_prev_re = dt * kappa2_re
            k_prev_im = dt * kappa2_im
            k_curr_re = dt * kappa1_re - k_prev_re
            k_curr_im = dt * kappa1_im - k_prev_im

            tap_prev_re = k_prev_re
            tap_prev_im = k_prev_im
            tap_curr_re = k_curr_re
            tap_curr_im = k_curr_im

        rho_re = cute.arch.shuffle_sync(rho_re, 0)
        rho_im = cute.arch.shuffle_sync(rho_im, 0)
        tap_prev_re = cute.arch.shuffle_sync(tap_prev_re, 0)
        tap_prev_im = cute.arch.shuffle_sync(tap_prev_im, 0)
        tap_curr_re = cute.arch.shuffle_sync(tap_curr_re, 0)
        tap_curr_im = cute.arch.shuffle_sync(tap_curr_im, 0)

        # ------------------------------------------------------------------
        # BC normalization + beta preparation
        # ------------------------------------------------------------------
        norm0 = cutlass.Float32(0.0)
        norm1 = cutlass.Float32(0.0)
        norm2 = cutlass.Float32(0.0)
        norm3 = cutlass.Float32(0.0)
        bc0 = cutlass.Float32(0.0)
        bc1 = cutlass.Float32(0.0)
        bc2 = cutlass.Float32(0.0)
        bc3 = cutlass.Float32(0.0)
        active_bc_warps = (self.n_size + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE

        if tidx < self.n_size:
            bc0 = cutlass.Float32(mBC[bidb, bidh, 0, tidx])
            bc1 = cutlass.Float32(mBC[bidb, bidh, 1, tidx])
            bc2 = cutlass.Float32(mBC[bidb, bidh, 2, tidx])
            bc3 = cutlass.Float32(mBC[bidb, bidh, 3, tidx])
            norm0 = bc0 * bc0
            norm1 = bc1 * bc1
            norm2 = bc2 * bc2
            norm3 = bc3 * bc3

        if warp_idx < active_bc_warps:
            norm0 = self._warp_reduce_sum(norm0)
            norm1 = self._warp_reduce_sum(norm1)
            norm2 = self._warp_reduce_sum(norm2)
            norm3 = self._warp_reduce_sum(norm3)

        if lane_idx == 0 and warp_idx < active_bc_warps:
            bc_reduce[0, warp_idx] = norm0
            bc_reduce[1, warp_idx] = norm1
            bc_reduce[2, warp_idx] = norm2
            bc_reduce[3, warp_idx] = norm3

        cute.arch.barrier()

        if warp_idx == 0 and lane_idx < 4:
            total = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(active_bc_warps):
                total = total + bc_reduce[lane_idx, w]
            denom = cutlass.Float32(self.n_size)
            inv_s[lane_idx] = cute.rsqrt(total / denom + cutlass.Float32(1.0e-5))

        cute.arch.barrier()

        if tidx < self.n_size:
            n = tidx
            b_re_v = bc0 * inv_s[0]
            b_im_v = bc1 * inv_s[1]
            c_re_v = bc2 * inv_s[2]
            c_im_v = bc3 * inv_s[3]
            if self.normalize_bc:
                b_re_v = b_re_v * cutlass.Float32(mBScale[bidh, 0, n])
                b_im_v = b_im_v * cutlass.Float32(mBScale[bidh, 1, n])
                c_re_v = c_re_v * cutlass.Float32(mCScale[bidh, 0, n])
                c_im_v = c_im_v * cutlass.Float32(mCScale[bidh, 1, n])

            b_re[n] = b_re_v
            b_im[n] = b_im_v
            c_re[n] = c_re_v
            c_im[n] = c_im_v

            prev_re = cutlass.Float32(mBPrev[bidb, bidh, 2 * n])
            prev_im = cutlass.Float32(mBPrev[bidb, bidh, 2 * n + 1])
            beta_prev_re_v, beta_prev_im_v = complex_mul(
                tap_prev_re,
                tap_prev_im,
                prev_re,
                prev_im,
            )
            beta_curr_re_v, beta_curr_im_v = complex_mul(
                tap_curr_re,
                tap_curr_im,
                b_re_v,
                b_im_v,
            )
            beta_prev_re[n] = beta_prev_re_v
            beta_prev_im[n] = beta_prev_im_v
            beta_curr_re[n] = beta_curr_re_v
            beta_curr_im[n] = beta_curr_im_v

            if bidp == 0:
                mBLast[bidb, bidh, 2 * n] = b_re_v.to(mBLast.element_type)
                mBLast[bidb, bidh, 2 * n + 1] = b_im_v.to(mBLast.element_type)

        cute.arch.barrier()

        # ------------------------------------------------------------------
        # Register-resident recurrence update + writeback
        # ------------------------------------------------------------------
        if self.use_state_cp_async:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        worker = tidx // self.tile_p
        p_local = tidx - worker * self.tile_p
        p_global = p_offset + p_local

        if tidx < self.num_threads and p_global < self.p_size:
            u_prev = cutlass.Float32(sUPrev[p_local])
            u_curr = cutlass.Float32(sValue[p_local])
            acc = cutlass.Float32(0.0)

            for vec_iter in cutlass.range_constexpr(self.n_vec_iters):
                n0 = worker * self.vec_n + vec_iter * self.n_groups * self.vec_n
                re_frag = cute.make_rmem_tensor((self.vec_n,), cutlass.Float32)
                im_frag = cute.make_rmem_tensor((self.vec_n,), cutlass.Float32)

                for vi in cutlass.range_constexpr(self.vec_n):
                    n = n0 + vi
                    if n < self.n_size:
                        z_re = cutlass.Float32(sState[2 * n, p_local])
                        z_im = cutlass.Float32(sState[2 * n + 1, p_local])
                        mz_re, mz_im = complex_mul(rho_re, rho_im, z_re, z_im)
                        drive_re = u_prev * beta_prev_re[n] + u_curr * beta_curr_re[n]
                        drive_im = u_prev * beta_prev_im[n] + u_curr * beta_curr_im[n]
                        out_re = mz_re + drive_re
                        out_im = mz_im + drive_im
                        re_frag[vi] = out_re
                        im_frag[vi] = out_im

                for vi in cutlass.range_constexpr(self.vec_n):
                    n = n0 + vi
                    if n < self.n_size:
                        out_re = re_frag[vi]
                        out_im = im_frag[vi]
                        acc = acc + real_mul_conj(out_re, out_im, c_re[n], c_im[n])
                        gFinalStateT[2 * n, p_local] = out_re.to(
                            mFinalState.element_type
                        )
                        gFinalStateT[2 * n + 1, p_local] = out_im.to(
                            mFinalState.element_type
                        )

            acc_partial[worker, p_local] = acc

        cute.arch.barrier()

        # ------------------------------------------------------------------
        # Output contraction / writeback
        # ------------------------------------------------------------------
        if tidx < self.tile_p:
            p = tidx
            p_g = p_offset + p
            if p_g < self.p_size:
                acc = cutlass.Float32(0.0)
                for worker_idx in cutlass.range_constexpr(self.n_groups):
                    acc = acc + acc_partial[worker_idx, p]
                u_curr = cutlass.Float32(sValue[p])
                gate = cutlass.Float32(sGate[p])
                silu_gate = gate * sigmoid(gate)
                y = (acc + u_curr * cutlass.Float32(sSkip[p])) * silu_gate
                sY[p] = y.to(sY.element_type)
                gY[p] = y.to(mY.element_type)
                gULast[p] = u_curr.to(mULast.element_type)

        cute.arch.barrier()

        if self.fuse_outproj:
            for out_iter in cutlass.range_constexpr(self.out_proj_iters):
                out_idx = tidx + out_iter * self.num_threads
                if out_idx < self.d_model:
                    proj_acc = cutlass.Float32(0.0)
                    for p in cutlass.range_constexpr(self.tile_p):
                        p_g = p_offset + p
                        if p_g < self.p_size:
                            proj_acc = proj_acc + cutlass.Float32(
                                sY[p]
                            ) * cutlass.Float32(mOutProj[out_idx, bidh, p_g])
                    if self.p_tiles == 1:
                        mProjected[bidb, bidh, out_idx] = proj_acc.to(
                            mProjected.element_type
                        )
                    else:
                        out_offset = (
                            bidb * self.heads * self.d_model
                            + bidh * self.d_model
                            + out_idx
                        )
                        cute.arch.atomic_add(
                            (mProjected.iterator + out_offset).llvm_ptr,
                            proj_acc,
                        )


__all__ = ["MixerDecodeStepFwd"]
