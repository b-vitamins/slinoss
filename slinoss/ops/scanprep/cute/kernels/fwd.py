# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Split forward kernels for the CuTe scanprep backend."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    COEFF_AUX_DELTA_R,
    COEFF_AUX_DELTA_THETA,
    COEFF_AUX_DT,
    COEFF_AUX_DT_U,
    COEFF_AUX_EXP_TERM,
    COEFF_AUX_GAMMA,
    COEFF_AUX_GAMMA_SIGMOID,
    COEFF_AUX_KAPPA1_IM,
    COEFF_AUX_KAPPA1_RE,
    COEFF_AUX_KAPPA2_IM,
    COEFF_AUX_KAPPA2_RE,
    COEFF_AUX_K_CURR_TANH_IM,
    COEFF_AUX_K_CURR_TANH_RE,
    COEFF_AUX_K_PREV_TANH_IM,
    COEFF_AUX_K_PREV_TANH_RE,
    COEFF_AUX_LOG_R,
    COEFF_AUX_MIX_K_CURR,
    COEFF_AUX_MIX_K_PREV,
    COEFF_AUX_MIX_R,
    COEFF_AUX_MIX_THETA,
    COEFF_AUX_OMEGA,
    COEFF_AUX_R,
    COEFF_AUX_RHO_IM,
    COEFF_AUX_RHO_RE,
    COEFF_AUX_R_DIRECT_U,
    COEFF_AUX_THETA,
    COEFF_AUX_THETA_DIRECT_TANH,
    complex_div,
    lerp,
    principal_angle,
    sigmoid,
    softplus,
)


class ScanPrepFwdFused:
    """Two-stage forward launcher for row-pack and coefficient generation."""

    def __init__(
        self,
        *,
        h_size: int,
        p_size: int,
        n_size: int,
        normalize_bc: bool,
        store_rms_inv: bool,
        store_coeff_aux: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
        pack_warps_per_block: int = 8,
        coeff_block_size: int = 256,
    ) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.normalize_bc = bool(normalize_bc)
        self.store_rms_inv = bool(store_rms_inv)
        self.store_coeff_aux = bool(store_coeff_aux)

        self.pack_warps_per_block = int(pack_warps_per_block)
        if self.pack_warps_per_block <= 0:
            raise ValueError("pack_warps_per_block must be positive.")
        self.pack_block_size = self.pack_warps_per_block * 32

        self.coeff_block_size = int(coeff_block_size)
        if self.coeff_block_size <= 0 or self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a positive multiple of 32.")
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        if self.coeff_head_tile <= 0:
            raise ValueError("coeff_block_size must cover at least one warp.")
        self.coeff_smem_bytes = self.coeff_head_tile * 13 * (self.coeff_t_tile + 1) * 4

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    def _coeff_shared_storage(self):
        coeff_layout = cute.make_layout(
            (self.coeff_head_tile, 13, self.coeff_t_tile + 1),
            stride=(13 * (self.coeff_t_tile + 1), self.coeff_t_tile + 1, 1),
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sParams": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(coeff_layout)],
                16,
            ]
        }

        return cute.struct(SharedStorage)

    @cute.kernel
    def pack_kernel(
        self,
        mValue: cute.Tensor,
        mBC: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mU: cute.Tensor,
        mBOut: cute.Tensor,
        mCOut: cute.Tensor,
        mRmsInv: cute.Tensor,
        total_rows_,
        rows_per_batch_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = bidx * self.pack_warps_per_block + warp
        if row < total_rows_:
            b = row // rows_per_batch_
            rem = row - b * rows_per_batch_
            t = rem // self.h_size
            h = rem - t * self.h_size

            num_p_iters = (self.p_size + 31) // 32
            for p_iter in cutlass.range_constexpr(num_p_iters):
                p = lane + p_iter * 32
                if p < self.p_size:
                    mU[b, h, t, p] = mValue[b, h, t, p]

            if cutlass.const_expr(self.normalize_bc):
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                num_n_iters = (self.n_size + 31) // 32
                x0_cache = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                x1_cache = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                x2_cache = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                x3_cache = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * 32
                    if n < self.n_size:
                        x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                        x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                        x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                        x0_cache[n_iter] = x0
                        x1_cache[n_iter] = x1
                        x2_cache[n_iter] = x2
                        x3_cache[n_iter] = x3
                        s0 = s0 + x0 * x0
                        s1 = s1 + x1 * x1
                        s2 = s2 + x2 * x2
                        s3 = s3 + x3 * x3

                s0 = cute.arch.warp_reduction_sum(s0, threads_in_group=32)
                s1 = cute.arch.warp_reduction_sum(s1, threads_in_group=32)
                s2 = cute.arch.warp_reduction_sum(s2, threads_in_group=32)
                s3 = cute.arch.warp_reduction_sum(s3, threads_in_group=32)
                denom = cutlass.Float32(self.n_size)
                eps_bc = cutlass.Float32(1.0e-5)
                inv0 = cute.rsqrt(s0 / denom + eps_bc)
                inv1 = cute.rsqrt(s1 / denom + eps_bc)
                inv2 = cute.rsqrt(s2 / denom + eps_bc)
                inv3 = cute.rsqrt(s3 / denom + eps_bc)

                if cutlass.const_expr(self.store_rms_inv) and lane == 0:
                    mRmsInv[b, h, t, 0] = inv0
                    mRmsInv[b, h, t, 1] = inv1
                    mRmsInv[b, h, t, 2] = inv2
                    mRmsInv[b, h, t, 3] = inv3

                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * 32
                    if n < self.n_size:
                        b0 = x0_cache[n_iter] * inv0
                        b1 = x1_cache[n_iter] * inv1
                        c0 = x2_cache[n_iter] * inv2
                        c1 = x3_cache[n_iter] * inv3
                        mBOut[b, h, t, 2 * n] = (
                            b0 * cutlass.Float32(mBScale[h, 0, n])
                        ).to(mBOut.element_type)
                        mBOut[b, h, t, 2 * n + 1] = (
                            b1 * cutlass.Float32(mBScale[h, 1, n])
                        ).to(mBOut.element_type)
                        mCOut[b, h, t, 2 * n] = (
                            c0 * cutlass.Float32(mCScale[h, 0, n])
                        ).to(mCOut.element_type)
                        mCOut[b, h, t, 2 * n + 1] = (
                            c1 * cutlass.Float32(mCScale[h, 1, n])
                        ).to(mCOut.element_type)
            else:
                num_n_iters = (self.n_size + 31) // 32
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * 32
                    if n < self.n_size:
                        mBOut[b, h, t, 2 * n] = cutlass.Float32(mBC[b, t, h, 0, n]).to(
                            mBOut.element_type
                        )
                        mBOut[b, h, t, 2 * n + 1] = cutlass.Float32(
                            mBC[b, t, h, 1, n]
                        ).to(mBOut.element_type)
                        mCOut[b, h, t, 2 * n] = cutlass.Float32(mBC[b, t, h, 2, n]).to(
                            mCOut.element_type
                        )
                        mCOut[b, h, t, 2 * n + 1] = cutlass.Float32(
                            mBC[b, t, h, 3, n]
                        ).to(mCOut.element_type)

    @cute.kernel
    def coeff_kernel(
        self,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mGammaBias: cute.Tensor,
        mOmegaBias: cute.Tensor,
        mMixRBias: cute.Tensor,
        mMixThetaBias: cute.Tensor,
        mMixKPrevBias: cute.Tensor,
        mMixKCurrBias: cute.Tensor,
        mMOut: cute.Tensor,
        mKOut: cute.Tensor,
        mCoeffAux: cute.Tensor,
        total_bt_,
        t_size_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32

        smem = cutlass.utils.SmemAllocator()
        sParams = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.coeff_head_tile, 13, self.coeff_t_tile + 1),
                stride=(13 * (self.coeff_t_tile + 1), self.coeff_t_tile + 1, 1),
            ),
            16,
        )

        h_base = block_y * self.coeff_head_tile
        num_load_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_load_flat_iters = (self.coeff_head_tile * 13 + 31) // 32
        for load_t_iter in cutlass.range_constexpr(num_load_t_iters):
            load_t_local = warp + load_t_iter * self.coeff_head_tile
            load_bt = block_x * self.coeff_t_tile + load_t_local
            if load_t_local < self.coeff_t_tile and load_bt < total_bt_:
                load_b = load_bt // t_size_
                load_t = load_bt - load_b * t_size_
                for flat_iter in cutlass.range_constexpr(num_load_flat_iters):
                    flat = lane + flat_iter * 32
                    if flat < self.coeff_head_tile * 13:
                        local_h = flat // 13
                        param_idx = flat - local_h * 13
                        load_h = h_base + local_h
                        if load_h < self.h_size:
                            sParams[local_h, param_idx, load_t_local] = cutlass.Float32(
                                mParams[load_b, load_t, load_h, param_idx]
                            )

        cute.arch.sync_threads()

        bt = block_x * self.coeff_t_tile + lane
        h = block_y * self.coeff_head_tile + warp
        if bt < total_bt_ and h < self.h_size:
            b = bt // t_size_
            t = bt - b * t_size_

            dt_raw = sParams[warp, 0, lane] + cutlass.Float32(mDtBias[h])
            gamma_raw = sParams[warp, 1, lane] + cutlass.Float32(mGammaBias[h])
            omega_raw = sParams[warp, 2, lane] + cutlass.Float32(mOmegaBias[h])
            r_raw = sParams[warp, 3, lane]
            theta_raw = sParams[warp, 4, lane]
            mix_r_raw = sParams[warp, 5, lane] + cutlass.Float32(mMixRBias[h])
            mix_theta_raw = sParams[warp, 6, lane] + cutlass.Float32(mMixThetaBias[h])
            mix_k_prev_raw = sParams[warp, 7, lane] + cutlass.Float32(mMixKPrevBias[h])
            mix_k_curr_raw = sParams[warp, 8, lane] + cutlass.Float32(mMixKCurrBias[h])
            k_prev_re_raw = sParams[warp, 9, lane]
            k_prev_im_raw = sParams[warp, 10, lane]
            k_curr_re_raw = sParams[warp, 11, lane]
            k_curr_im_raw = sParams[warp, 12, lane]

            dt_u = sigmoid(dt_raw)
            gamma = softplus(gamma_raw)
            gamma_sigmoid = sigmoid(gamma_raw)
            omega = omega_raw
            r_direct_u = sigmoid(r_raw)
            theta_direct_tanh = cute_math.tanh(theta_raw)
            theta_direct = cutlass.Float32(self.theta_bound) * theta_direct_tanh
            mix_r = sigmoid(mix_r_raw)
            mix_theta = sigmoid(mix_theta_raw)
            mix_k_prev = sigmoid(mix_k_prev_raw)
            mix_k_curr = sigmoid(mix_k_curr_raw)
            k_prev_tanh_re = cute_math.tanh(k_prev_re_raw)
            k_prev_tanh_im = cute_math.tanh(k_prev_im_raw)
            k_curr_tanh_re = cute_math.tanh(k_curr_re_raw)
            k_curr_tanh_im = cute_math.tanh(k_curr_im_raw)
            k_prev_learned_re = cutlass.Float32(self.k_max) * k_prev_tanh_re
            k_prev_learned_im = cutlass.Float32(self.k_max) * k_prev_tanh_im
            k_curr_learned_re = cutlass.Float32(self.k_max) * k_curr_tanh_re
            k_curr_learned_im = cutlass.Float32(self.k_max) * k_curr_tanh_im

            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            exp_term = cute_math.exp(-(gamma * dt))
            r_struct = (
                cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * exp_term
            )
            theta_struct = omega * dt
            r_direct = (
                cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * r_direct_u
            )

            r = lerp(r_direct, r_struct, mix_r)
            theta = principal_angle(lerp(theta_direct, theta_struct, mix_theta))

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

            out_prev_re = lerp(k_prev_learned_re, k_prev_re, mix_k_prev)
            out_prev_im = lerp(k_prev_learned_im, k_prev_im, mix_k_prev)
            out_curr_re = lerp(k_curr_learned_re, k_curr_re, mix_k_curr)
            out_curr_im = lerp(k_curr_learned_im, k_curr_im, mix_k_curr)

            mMOut[b, h, t, 0] = rho_re
            mMOut[b, h, t, 1] = rho_im
            mKOut[b, h, t, 0, 0] = out_prev_re
            mKOut[b, h, t, 0, 1] = out_prev_im
            mKOut[b, h, t, 1, 0] = out_curr_re
            mKOut[b, h, t, 1, 1] = out_curr_im

            if cutlass.const_expr(self.store_coeff_aux):
                mCoeffAux[b, h, COEFF_AUX_DT_U, t] = dt_u
                mCoeffAux[b, h, COEFF_AUX_GAMMA_SIGMOID, t] = gamma_sigmoid
                mCoeffAux[b, h, COEFF_AUX_OMEGA, t] = omega
                mCoeffAux[b, h, COEFF_AUX_R_DIRECT_U, t] = r_direct_u
                mCoeffAux[b, h, COEFF_AUX_THETA_DIRECT_TANH, t] = theta_direct_tanh
                mCoeffAux[b, h, COEFF_AUX_MIX_R, t] = mix_r
                mCoeffAux[b, h, COEFF_AUX_MIX_THETA, t] = mix_theta
                mCoeffAux[b, h, COEFF_AUX_MIX_K_PREV, t] = mix_k_prev
                mCoeffAux[b, h, COEFF_AUX_MIX_K_CURR, t] = mix_k_curr
                mCoeffAux[b, h, COEFF_AUX_K_PREV_TANH_RE, t] = k_prev_tanh_re
                mCoeffAux[b, h, COEFF_AUX_K_PREV_TANH_IM, t] = k_prev_tanh_im
                mCoeffAux[b, h, COEFF_AUX_K_CURR_TANH_RE, t] = k_curr_tanh_re
                mCoeffAux[b, h, COEFF_AUX_K_CURR_TANH_IM, t] = k_curr_tanh_im
                mCoeffAux[b, h, COEFF_AUX_DT, t] = dt
                mCoeffAux[b, h, COEFF_AUX_GAMMA, t] = gamma
                mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t] = exp_term
                mCoeffAux[b, h, COEFF_AUX_DELTA_R, t] = r_struct - r_direct
                mCoeffAux[b, h, COEFF_AUX_DELTA_THETA, t] = theta_struct - theta_direct
                mCoeffAux[b, h, COEFF_AUX_R, t] = r
                mCoeffAux[b, h, COEFF_AUX_THETA, t] = theta
                mCoeffAux[b, h, COEFF_AUX_RHO_RE, t] = rho_re
                mCoeffAux[b, h, COEFF_AUX_RHO_IM, t] = rho_im
                mCoeffAux[b, h, COEFF_AUX_LOG_R, t] = log_r
                mCoeffAux[b, h, COEFF_AUX_KAPPA1_RE, t] = kappa1_re
                mCoeffAux[b, h, COEFF_AUX_KAPPA1_IM, t] = kappa1_im
                mCoeffAux[b, h, COEFF_AUX_KAPPA2_RE, t] = kappa2_re
                mCoeffAux[b, h, COEFF_AUX_KAPPA2_IM, t] = kappa2_im

    @cute.jit
    def __call__(
        self,
        value: cute.Tensor,
        bc: cute.Tensor,
        b_scale: cute.Tensor,
        c_scale: cute.Tensor,
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        gamma_bias: cute.Tensor,
        omega_bias: cute.Tensor,
        mix_r_bias: cute.Tensor,
        mix_theta_bias: cute.Tensor,
        mix_k_prev_bias: cute.Tensor,
        mix_k_curr_bias: cute.Tensor,
        u: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: cute.Tensor,
        k: cute.Tensor,
        rms_inv: cute.Tensor,
        coeff_aux: cute.Tensor,
    ):
        batch = cute.size(value, mode=[0])
        t_size = cute.size(value, mode=[1])
        total_rows = batch * t_size * self.h_size
        rows_per_batch = t_size * self.h_size
        total_bt = batch * t_size
        pack_grid_size = (
            total_rows + self.pack_warps_per_block - 1
        ) // self.pack_warps_per_block
        coeff_grid_x = (total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile
        coeff_grid_y = (self.h_size + self.coeff_head_tile - 1) // self.coeff_head_tile

        mValue = cute.make_tensor(
            value.iterator,
            cute.make_layout(
                (batch, self.h_size, t_size, self.p_size),
                stride=(
                    t_size * self.h_size * self.p_size,
                    self.p_size,
                    self.h_size * self.p_size,
                    1,
                ),
            ),
        )
        mParams = cute.make_tensor(
            params.iterator,
            cute.make_layout(
                (batch, t_size, self.h_size, 13),
                stride=(t_size * self.h_size * 13, self.h_size * 13, 13, 1),
            ),
        )
        coeff_smem_bytes = int(self._coeff_shared_storage().size_in_bytes())

        self.pack_kernel(
            mValue,
            bc,
            b_scale,
            c_scale,
            u,
            b,
            c,
            rms_inv,
            total_rows,
            rows_per_batch,
        ).launch(grid=(pack_grid_size, 1, 1), block=(self.pack_block_size, 1, 1))
        self.coeff_kernel(
            mParams,
            dt_bias,
            gamma_bias,
            omega_bias,
            mix_r_bias,
            mix_theta_bias,
            mix_k_prev_bias,
            mix_k_curr_bias,
            m,
            k,
            coeff_aux,
            total_bt,
            t_size,
        ).launch(
            grid=(coeff_grid_x, coeff_grid_y, 1),
            block=(self.coeff_block_size, 1, 1),
            smem=coeff_smem_bytes,
        )


__all__ = ["ScanPrepFwdFused"]
