# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Fused backward kernels for the CuTe scanprep backend."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    complex_div,
    complex_mul_conj,
    lerp,
    make_row_major_stride,
    principal_angle,
    real_mul_conj,
    sigmoid,
    softplus,
)


class ScanPrepBwdFused:
    """Thin host-JIT wrapper around the fused CuTe scanprep backward path."""

    def __init__(
        self,
        *,
        spec: tuple[int, int, int, int, int, int],
        du_stride: tuple[int, int, int, int] | None = None,
        params_in_stride: tuple[int, int, int] | None = None,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
        pack_block_size: int = 32,
        coeff_block_size: int = 128,
    ) -> None:
        batch, t_size, h_size, p_size, n_size, param_dim = spec
        self.batch = int(batch)
        self.t_size = int(t_size)
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.param_dim = int(param_dim)
        self.normalize_bc = bool(normalize_bc)

        self.du_shape = (self.batch, self.h_size, self.t_size, self.p_size)
        self.du_stride = (
            tuple(int(s) for s in du_stride)
            if du_stride is not None
            else make_row_major_stride(self.du_shape)
        )
        self.value_shape = (self.batch, self.t_size, self.h_size * self.p_size)
        self.value_stride = make_row_major_stride(self.value_shape)
        self.bc_shape = (self.batch, self.t_size, self.h_size, 4, self.n_size)
        self.bc_stride = make_row_major_stride(self.bc_shape)
        self.grad_shape = (self.batch, self.h_size, self.t_size, 2 * self.n_size)
        self.grad_stride = make_row_major_stride(self.grad_shape)
        self.bc_scale_shape = (self.h_size, 2, self.n_size)
        self.bc_scale_stride = make_row_major_stride(self.bc_scale_shape)
        self.scale_grad_shape = (self.h_size, 4, self.n_size)
        self.scale_grad_stride = make_row_major_stride(self.scale_grad_shape)
        self.param_flat_size = self.h_size * self.param_dim
        self.params_in_shape = (self.batch, self.t_size, self.param_flat_size)
        self.params_in_stride = (
            tuple(int(s) for s in params_in_stride)
            if params_in_stride is not None
            else make_row_major_stride(self.params_in_shape)
        )
        self.dparams_shape = (self.batch, self.t_size, self.param_flat_size)
        self.dparams_stride = make_row_major_stride(self.dparams_shape)
        self.m_shape = (self.batch, self.h_size, self.t_size, 2)
        self.m_stride = make_row_major_stride(self.m_shape)
        self.k_shape = (self.batch, self.h_size, self.t_size, 2, 2)
        self.k_stride = make_row_major_stride(self.k_shape)
        self.bias_shape = (self.h_size,)
        self.bias_stride = make_row_major_stride(self.bias_shape)
        self.bias_grad_shape = (self.h_size, 7)
        self.bias_grad_stride = make_row_major_stride(self.bias_grad_shape)

        self.total_rows = self.batch * self.h_size * self.t_size
        self.pack_block_size = int(pack_block_size)
        if self.pack_block_size % 32 != 0:
            raise ValueError("pack_block_size must be a multiple of 32.")
        self.pack_group_size = 16 if self.n_size <= 16 else 32
        if self.pack_block_size % self.pack_group_size != 0:
            raise ValueError("pack_block_size must be divisible by pack_group_size.")
        self.pack_groups_per_block = self.pack_block_size // self.pack_group_size
        self.pack_grid_size = (
            self.total_rows + self.pack_groups_per_block - 1
        ) // self.pack_groups_per_block
        self.coeff_block_size = int(coeff_block_size)
        if self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a multiple of 32.")
        self.coeff_grid_size = (
            self.total_rows + self.coeff_block_size - 1
        ) // self.coeff_block_size

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    @cute.kernel
    def pack_bc_kernel(
        self,
        mDU: cute.Tensor,
        mBC: cute.Tensor,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mValueGrad: cute.Tensor,
        mBCGrad: cute.Tensor,
        mScaleGrad: cute.Tensor,
        total_rows_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        subwarp_size = self.pack_group_size
        group = tidx // subwarp_size
        lane = tidx - group * subwarp_size
        row = bidx * self.pack_groups_per_block + group
        if row < total_rows_:
            rows_per_batch = self.h_size * self.t_size
            b = row // rows_per_batch
            rem = row - b * rows_per_batch
            h = rem // self.t_size
            t = rem - h * self.t_size

            base = h * self.p_size
            num_p_iters = (self.p_size + subwarp_size - 1) // subwarp_size
            for p_iter in cutlass.range_constexpr(num_p_iters):
                p = lane + p_iter * subwarp_size
                if p < self.p_size:
                    mValueGrad[b, t, base + p] = mDU[b, h, t, p]

            if self.normalize_bc:
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                num_n_iters = (self.n_size + subwarp_size - 1) // subwarp_size
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * subwarp_size
                    if n < self.n_size:
                        x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                        x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                        x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                        s0 = s0 + x0 * x0
                        s1 = s1 + x1 * x1
                        s2 = s2 + x2 * x2
                        s3 = s3 + x3 * x3
                s0 = cute.arch.warp_reduction_sum(s0, threads_in_group=subwarp_size)
                s1 = cute.arch.warp_reduction_sum(s1, threads_in_group=subwarp_size)
                s2 = cute.arch.warp_reduction_sum(s2, threads_in_group=subwarp_size)
                s3 = cute.arch.warp_reduction_sum(s3, threads_in_group=subwarp_size)
                denom = cutlass.Float32(self.n_size)
                eps_bc = cutlass.Float32(1.0e-5)
                inv0 = cute.rsqrt(s0 / denom + eps_bc)
                inv1 = cute.rsqrt(s1 / denom + eps_bc)
                inv2 = cute.rsqrt(s2 / denom + eps_bc)
                inv3 = cute.rsqrt(s3 / denom + eps_bc)
                inv0_cubed = inv0 * inv0 * inv0 / denom
                inv1_cubed = inv1 * inv1 * inv1 / denom
                inv2_cubed = inv2 * inv2 * inv2 / denom
                inv3_cubed = inv3 * inv3 * inv3 / denom

                dot0 = cutlass.Float32(0.0)
                dot1 = cutlass.Float32(0.0)
                dot2 = cutlass.Float32(0.0)
                dot3 = cutlass.Float32(0.0)
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * subwarp_size
                    if n < self.n_size:
                        db0 = cutlass.Float32(mDB[b, h, t, 2 * n])
                        db1 = cutlass.Float32(mDB[b, h, t, 2 * n + 1])
                        dc0 = cutlass.Float32(mDC[b, h, t, 2 * n])
                        dc1 = cutlass.Float32(mDC[b, h, t, 2 * n + 1])
                        x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                        x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                        x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                        scale0 = cutlass.Float32(mBScale[h, 0, n])
                        scale1 = cutlass.Float32(mBScale[h, 1, n])
                        scale2 = cutlass.Float32(mCScale[h, 0, n])
                        scale3 = cutlass.Float32(mCScale[h, 1, n])
                        dot0 = dot0 + (db0 * scale0) * x0
                        dot1 = dot1 + (db1 * scale1) * x1
                        dot2 = dot2 + (dc0 * scale2) * x2
                        dot3 = dot3 + (dc1 * scale3) * x3
                dot0 = cute.arch.warp_reduction_sum(dot0, threads_in_group=subwarp_size)
                dot1 = cute.arch.warp_reduction_sum(dot1, threads_in_group=subwarp_size)
                dot2 = cute.arch.warp_reduction_sum(dot2, threads_in_group=subwarp_size)
                dot3 = cute.arch.warp_reduction_sum(dot3, threads_in_group=subwarp_size)
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * subwarp_size
                    if n < self.n_size:
                        db0 = cutlass.Float32(mDB[b, h, t, 2 * n])
                        db1 = cutlass.Float32(mDB[b, h, t, 2 * n + 1])
                        dc0 = cutlass.Float32(mDC[b, h, t, 2 * n])
                        dc1 = cutlass.Float32(mDC[b, h, t, 2 * n + 1])
                        x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                        x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                        x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                        y0 = x0 * inv0
                        y1 = x1 * inv1
                        y2 = x2 * inv2
                        y3 = x3 * inv3
                        scale0 = cutlass.Float32(mBScale[h, 0, n])
                        scale1 = cutlass.Float32(mBScale[h, 1, n])
                        scale2 = cutlass.Float32(mCScale[h, 0, n])
                        scale3 = cutlass.Float32(mCScale[h, 1, n])
                        dy0 = db0 * scale0
                        dy1 = db1 * scale1
                        dy2 = dc0 * scale2
                        dy3 = dc1 * scale3
                        mBCGrad[b, t, h, 0, n] = (
                            inv0 * dy0 - x0 * inv0_cubed * dot0
                        ).to(mBCGrad.element_type)
                        mBCGrad[b, t, h, 1, n] = (
                            inv1 * dy1 - x1 * inv1_cubed * dot1
                        ).to(mBCGrad.element_type)
                        mBCGrad[b, t, h, 2, n] = (
                            inv2 * dy2 - x2 * inv2_cubed * dot2
                        ).to(mBCGrad.element_type)
                        mBCGrad[b, t, h, 3, n] = (
                            inv3 * dy3 - x3 * inv3_cubed * dot3
                        ).to(mBCGrad.element_type)
                        h_base = h * 4 * self.n_size
                        cute.arch.atomic_add(
                            (mScaleGrad.iterator + h_base + n).llvm_ptr, db0 * y0
                        )
                        cute.arch.atomic_add(
                            (mScaleGrad.iterator + h_base + self.n_size + n).llvm_ptr,
                            db1 * y1,
                        )
                        cute.arch.atomic_add(
                            (
                                mScaleGrad.iterator + h_base + 2 * self.n_size + n
                            ).llvm_ptr,
                            dc0 * y2,
                        )
                        cute.arch.atomic_add(
                            (
                                mScaleGrad.iterator + h_base + 3 * self.n_size + n
                            ).llvm_ptr,
                            dc1 * y3,
                        )
            else:
                num_n_iters = (self.n_size + subwarp_size - 1) // subwarp_size
                for n_iter in cutlass.range_constexpr(num_n_iters):
                    n = lane + n_iter * subwarp_size
                    if n < self.n_size:
                        mBCGrad[b, t, h, 0, n] = mDB[b, h, t, 2 * n]
                        mBCGrad[b, t, h, 1, n] = mDB[b, h, t, 2 * n + 1]
                        mBCGrad[b, t, h, 2, n] = mDC[b, h, t, 2 * n]
                        mBCGrad[b, t, h, 3, n] = mDC[b, h, t, 2 * n + 1]

    @cute.kernel
    def coeff_kernel(
        self,
        mParams: cute.Tensor,
        mDM: cute.Tensor,
        mDK: cute.Tensor,
        mDtBias: cute.Tensor,
        mGammaBias: cute.Tensor,
        mOmegaBias: cute.Tensor,
        mMixRBias: cute.Tensor,
        mMixThetaBias: cute.Tensor,
        mMixKPrevBias: cute.Tensor,
        mMixKCurrBias: cute.Tensor,
        mDParams: cute.Tensor,
        mBiasGrad: cute.Tensor,
        total_rows_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * self.coeff_block_size + tidx
        if row < total_rows_:
            rows_per_batch = self.h_size * self.t_size
            b = row // rows_per_batch
            rem = row - b * rows_per_batch
            h = rem // self.t_size
            t = rem - h * self.t_size

            p_base = h * self.param_dim
            dt_raw = cutlass.Float32(mParams[b, t, p_base + 0]) + cutlass.Float32(
                mDtBias[h]
            )
            gamma_raw = cutlass.Float32(mParams[b, t, p_base + 1]) + cutlass.Float32(
                mGammaBias[h]
            )
            omega_raw = cutlass.Float32(mParams[b, t, p_base + 2]) + cutlass.Float32(
                mOmegaBias[h]
            )
            r_raw = cutlass.Float32(mParams[b, t, p_base + 3])
            theta_raw = cutlass.Float32(mParams[b, t, p_base + 4])
            mix_r_raw = cutlass.Float32(mParams[b, t, p_base + 5]) + cutlass.Float32(
                mMixRBias[h]
            )
            mix_theta_raw = cutlass.Float32(
                mParams[b, t, p_base + 6]
            ) + cutlass.Float32(mMixThetaBias[h])
            mix_k_prev_raw = cutlass.Float32(
                mParams[b, t, p_base + 7]
            ) + cutlass.Float32(mMixKPrevBias[h])
            mix_k_curr_raw = cutlass.Float32(
                mParams[b, t, p_base + 8]
            ) + cutlass.Float32(mMixKCurrBias[h])
            k_prev_re_raw = cutlass.Float32(mParams[b, t, p_base + 9])
            k_prev_im_raw = cutlass.Float32(mParams[b, t, p_base + 10])
            k_curr_re_raw = cutlass.Float32(mParams[b, t, p_base + 11])
            k_curr_im_raw = cutlass.Float32(mParams[b, t, p_base + 12])

            dt_u = sigmoid(dt_raw)
            gamma = softplus(gamma_raw)
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

            k_prev_struct_re = dt * kappa2_re
            k_prev_struct_im = dt * kappa2_im
            k_curr_struct_re = dt * kappa1_re - k_prev_struct_re
            k_curr_struct_im = dt * kappa1_im - k_prev_struct_im

            g_rho_re = cutlass.Float32(mDM[b, h, t, 0])
            g_rho_im = cutlass.Float32(mDM[b, h, t, 1])
            g_k_prev_re = cutlass.Float32(mDK[b, h, t, 0, 0])
            g_k_prev_im = cutlass.Float32(mDK[b, h, t, 0, 1])
            g_k_curr_re = cutlass.Float32(mDK[b, h, t, 1, 0])
            g_k_curr_im = cutlass.Float32(mDK[b, h, t, 1, 1])

            one_minus_mix_k_prev = cutlass.Float32(1.0) - mix_k_prev
            one_minus_mix_k_curr = cutlass.Float32(1.0) - mix_k_curr
            g_k_prev_learned_re = g_k_prev_re * one_minus_mix_k_prev
            g_k_prev_learned_im = g_k_prev_im * one_minus_mix_k_prev
            g_k_prev_struct_re = g_k_prev_re * mix_k_prev
            g_k_prev_struct_im = g_k_prev_im * mix_k_prev
            g_mix_k_prev = real_mul_conj(
                g_k_prev_re,
                g_k_prev_im,
                k_prev_struct_re - k_prev_learned_re,
                k_prev_struct_im - k_prev_learned_im,
            )

            g_k_curr_learned_re = g_k_curr_re * one_minus_mix_k_curr
            g_k_curr_learned_im = g_k_curr_im * one_minus_mix_k_curr
            g_k_curr_struct_re = g_k_curr_re * mix_k_curr
            g_k_curr_struct_im = g_k_curr_im * mix_k_curr
            g_mix_k_curr = real_mul_conj(
                g_k_curr_re,
                g_k_curr_im,
                k_curr_struct_re - k_curr_learned_re,
                k_curr_struct_im - k_curr_learned_im,
            )

            kappa_diff_re = kappa1_re - kappa2_re
            kappa_diff_im = kappa1_im - kappa2_im
            g_dt = real_mul_conj(
                g_k_prev_struct_re,
                g_k_prev_struct_im,
                kappa2_re,
                kappa2_im,
            ) + real_mul_conj(
                g_k_curr_struct_re,
                g_k_curr_struct_im,
                kappa_diff_re,
                kappa_diff_im,
            )

            g_kappa1_re = g_k_curr_struct_re * dt
            g_kappa1_im = g_k_curr_struct_im * dt
            g_kappa2_re = (g_k_prev_struct_re - g_k_curr_struct_re) * dt
            g_kappa2_im = (g_k_prev_struct_im - g_k_curr_struct_im) * dt
            g_z_re = cutlass.Float32(0.0)
            g_z_im = cutlass.Float32(0.0)

            if z_norm_sq < cutlass.Float32(self.z_thresh_sq):
                deriv1_re = (
                    cutlass.Float32(0.5)
                    + z_re / cutlass.Float32(3.0)
                    + z2_re / cutlass.Float32(8.0)
                )
                deriv1_im = z_im / cutlass.Float32(3.0) + z2_im / cutlass.Float32(8.0)
                deriv2_re = (
                    cutlass.Float32(1.0) / cutlass.Float32(3.0)
                    + z_re / cutlass.Float32(4.0)
                    + z2_re / cutlass.Float32(10.0)
                )
                deriv2_im = z_im / cutlass.Float32(4.0) + z2_im / cutlass.Float32(10.0)
                add1_re, add1_im = complex_mul_conj(
                    g_kappa1_re, g_kappa1_im, deriv1_re, deriv1_im
                )
                add2_re, add2_im = complex_mul_conj(
                    g_kappa2_re, g_kappa2_im, deriv2_re, deriv2_im
                )
                g_z_re = g_z_re + add1_re + add2_re
                g_z_im = g_z_im + add1_im + add2_im
            else:
                inv_z_re, inv_z_im = complex_div(
                    cutlass.Float32(1.0),
                    cutlass.Float32(0.0),
                    z_re,
                    z_im,
                )
                add_rho_re, add_rho_im = complex_mul_conj(
                    g_kappa1_re, g_kappa1_im, inv_z_re, inv_z_im
                )
                g_rho_re = g_rho_re + add_rho_re
                g_rho_im = g_rho_im + add_rho_im

                neg_rho_minus_one_over_z2_re, neg_rho_minus_one_over_z2_im = (
                    complex_div(
                        -(rho_re - cutlass.Float32(1.0)),
                        -rho_im,
                        z2_re,
                        z2_im,
                    )
                )
                add_z1_re, add_z1_im = complex_mul_conj(
                    g_kappa1_re,
                    g_kappa1_im,
                    neg_rho_minus_one_over_z2_re,
                    neg_rho_minus_one_over_z2_im,
                )
                g_z_re = g_z_re + add_z1_re
                g_z_im = g_z_im + add_z1_im

                inv_z2_re, inv_z2_im = complex_div(
                    cutlass.Float32(1.0),
                    cutlass.Float32(0.0),
                    z2_re,
                    z2_im,
                )
                g_num2_re, g_num2_im = complex_mul_conj(
                    g_kappa2_re, g_kappa2_im, inv_z2_re, inv_z2_im
                )

                z4_re = z2_re * z2_re - z2_im * z2_im
                z4_im = cutlass.Float32(2.0) * z2_re * z2_im
                num2_re = (
                    rho_re * (z_re - cutlass.Float32(1.0))
                    - rho_im * z_im
                    + cutlass.Float32(1.0)
                )
                num2_im = rho_re * z_im + rho_im * (z_re - cutlass.Float32(1.0))
                neg_num2_over_z4_re, neg_num2_over_z4_im = complex_div(
                    -num2_re,
                    -num2_im,
                    z4_re,
                    z4_im,
                )
                g_denom2_re, g_denom2_im = complex_mul_conj(
                    g_kappa2_re,
                    g_kappa2_im,
                    neg_num2_over_z4_re,
                    neg_num2_over_z4_im,
                )

                add_rho2_re, add_rho2_im = complex_mul_conj(
                    g_num2_re,
                    g_num2_im,
                    z_re - cutlass.Float32(1.0),
                    z_im,
                )
                g_rho_re = g_rho_re + add_rho2_re
                g_rho_im = g_rho_im + add_rho2_im

                add_z2_re, add_z2_im = complex_mul_conj(
                    g_num2_re,
                    g_num2_im,
                    rho_re,
                    rho_im,
                )
                g_z_re = g_z_re + add_z2_re
                g_z_im = g_z_im + add_z2_im

                add_z3_re, add_z3_im = complex_mul_conj(
                    g_denom2_re,
                    g_denom2_im,
                    cutlass.Float32(2.0) * z_re,
                    cutlass.Float32(2.0) * z_im,
                )
                g_z_re = g_z_re + add_z3_re
                g_z_im = g_z_im + add_z3_im

            unit_re = rho_re / r
            unit_im = rho_im / r
            g_r = real_mul_conj(g_rho_re, g_rho_im, unit_re, unit_im)
            g_theta = real_mul_conj(g_rho_re, g_rho_im, -rho_im, rho_re)
            g_log_r = g_z_re
            g_theta = g_theta + g_z_im
            g_r = g_r + g_log_r / r

            one_minus_mix_theta = cutlass.Float32(1.0) - mix_theta
            g_theta_direct = g_theta * one_minus_mix_theta
            g_theta_struct = g_theta * mix_theta
            g_mix_theta = g_theta * (theta_struct - theta_direct)

            one_minus_mix_r = cutlass.Float32(1.0) - mix_r
            g_r_direct = g_r * one_minus_mix_r
            g_r_struct = g_r * mix_r
            g_mix_r = g_r * (r_struct - r_direct)

            g_r_direct_u = g_r_direct * cutlass.Float32(self.r_scale)
            g_exp_term = g_r_struct * cutlass.Float32(self.r_scale)
            g_x = g_exp_term * exp_term
            g_gamma = g_x * (-dt)
            g_dt = g_dt + g_x * (-gamma)
            g_omega = g_theta_struct * dt
            g_dt = g_dt + g_theta_struct * omega
            g_dt_u = g_dt * cutlass.Float32(self.dt_scale)

            d0 = g_dt_u * dt_u * (cutlass.Float32(1.0) - dt_u)
            d1 = g_gamma * sigmoid(gamma_raw)
            d2 = g_omega
            d3 = g_r_direct_u * r_direct_u * (cutlass.Float32(1.0) - r_direct_u)
            d4 = (
                g_theta_direct
                * cutlass.Float32(self.theta_bound)
                * (cutlass.Float32(1.0) - theta_direct_tanh * theta_direct_tanh)
            )
            d5 = g_mix_r * mix_r * (cutlass.Float32(1.0) - mix_r)
            d6 = g_mix_theta * mix_theta * (cutlass.Float32(1.0) - mix_theta)
            d7 = g_mix_k_prev * mix_k_prev * (cutlass.Float32(1.0) - mix_k_prev)
            d8 = g_mix_k_curr * mix_k_curr * (cutlass.Float32(1.0) - mix_k_curr)
            d9 = (
                g_k_prev_learned_re
                * cutlass.Float32(self.k_max)
                * (cutlass.Float32(1.0) - k_prev_tanh_re * k_prev_tanh_re)
            )
            d10 = (
                g_k_prev_learned_im
                * cutlass.Float32(self.k_max)
                * (cutlass.Float32(1.0) - k_prev_tanh_im * k_prev_tanh_im)
            )
            d11 = (
                g_k_curr_learned_re
                * cutlass.Float32(self.k_max)
                * (cutlass.Float32(1.0) - k_curr_tanh_re * k_curr_tanh_re)
            )
            d12 = (
                g_k_curr_learned_im
                * cutlass.Float32(self.k_max)
                * (cutlass.Float32(1.0) - k_curr_tanh_im * k_curr_tanh_im)
            )

            mDParams[b, t, p_base + 0] = d0.to(mDParams.element_type)
            mDParams[b, t, p_base + 1] = d1.to(mDParams.element_type)
            mDParams[b, t, p_base + 2] = d2.to(mDParams.element_type)
            mDParams[b, t, p_base + 3] = d3.to(mDParams.element_type)
            mDParams[b, t, p_base + 4] = d4.to(mDParams.element_type)
            mDParams[b, t, p_base + 5] = d5.to(mDParams.element_type)
            mDParams[b, t, p_base + 6] = d6.to(mDParams.element_type)
            mDParams[b, t, p_base + 7] = d7.to(mDParams.element_type)
            mDParams[b, t, p_base + 8] = d8.to(mDParams.element_type)
            mDParams[b, t, p_base + 9] = d9.to(mDParams.element_type)
            mDParams[b, t, p_base + 10] = d10.to(mDParams.element_type)
            mDParams[b, t, p_base + 11] = d11.to(mDParams.element_type)
            mDParams[b, t, p_base + 12] = d12.to(mDParams.element_type)
            h_bias = h * 7
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 0).llvm_ptr, d0)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 1).llvm_ptr, d1)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 2).llvm_ptr, d2)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 3).llvm_ptr, d5)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 4).llvm_ptr, d6)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 5).llvm_ptr, d7)
            cute.arch.atomic_add((mBiasGrad.iterator + h_bias + 6).llvm_ptr, d8)

    @cute.jit
    def __call__(
        self,
        du_ptr: cute.Pointer,
        bc_ptr: cute.Pointer,
        db_ptr: cute.Pointer,
        dc_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        c_scale_ptr: cute.Pointer,
        params_ptr: cute.Pointer,
        dm_ptr: cute.Pointer,
        dk_ptr: cute.Pointer,
        dt_bias_ptr: cute.Pointer,
        gamma_bias_ptr: cute.Pointer,
        omega_bias_ptr: cute.Pointer,
        mix_r_bias_ptr: cute.Pointer,
        mix_theta_bias_ptr: cute.Pointer,
        mix_k_prev_bias_ptr: cute.Pointer,
        mix_k_curr_bias_ptr: cute.Pointer,
        value_grad_ptr: cute.Pointer,
        bc_grad_ptr: cute.Pointer,
        dparams_ptr: cute.Pointer,
        scale_grad_ptr: cute.Pointer,
        bias_grad_ptr: cute.Pointer,
    ):
        mDU = cute.make_tensor(
            du_ptr, cute.make_layout(self.du_shape, stride=self.du_stride)
        )
        mBC = cute.make_tensor(
            bc_ptr, cute.make_layout(self.bc_shape, stride=self.bc_stride)
        )
        mDB = cute.make_tensor(
            db_ptr, cute.make_layout(self.grad_shape, stride=self.grad_stride)
        )
        mDC = cute.make_tensor(
            dc_ptr, cute.make_layout(self.grad_shape, stride=self.grad_stride)
        )
        mBScale = cute.make_tensor(
            b_scale_ptr,
            cute.make_layout(self.bc_scale_shape, stride=self.bc_scale_stride),
        )
        mCScale = cute.make_tensor(
            c_scale_ptr,
            cute.make_layout(self.bc_scale_shape, stride=self.bc_scale_stride),
        )
        mParams = cute.make_tensor(
            params_ptr,
            cute.make_layout(self.params_in_shape, stride=self.params_in_stride),
        )
        mDM = cute.make_tensor(
            dm_ptr, cute.make_layout(self.m_shape, stride=self.m_stride)
        )
        mDK = cute.make_tensor(
            dk_ptr, cute.make_layout(self.k_shape, stride=self.k_stride)
        )
        mDtBias = cute.make_tensor(
            dt_bias_ptr, cute.make_layout(self.bias_shape, stride=self.bias_stride)
        )
        mGammaBias = cute.make_tensor(
            gamma_bias_ptr, cute.make_layout(self.bias_shape, stride=self.bias_stride)
        )
        mOmegaBias = cute.make_tensor(
            omega_bias_ptr, cute.make_layout(self.bias_shape, stride=self.bias_stride)
        )
        mMixRBias = cute.make_tensor(
            mix_r_bias_ptr, cute.make_layout(self.bias_shape, stride=self.bias_stride)
        )
        mMixThetaBias = cute.make_tensor(
            mix_theta_bias_ptr,
            cute.make_layout(self.bias_shape, stride=self.bias_stride),
        )
        mMixKPrevBias = cute.make_tensor(
            mix_k_prev_bias_ptr,
            cute.make_layout(self.bias_shape, stride=self.bias_stride),
        )
        mMixKCurrBias = cute.make_tensor(
            mix_k_curr_bias_ptr,
            cute.make_layout(self.bias_shape, stride=self.bias_stride),
        )
        mValueGrad = cute.make_tensor(
            value_grad_ptr, cute.make_layout(self.value_shape, stride=self.value_stride)
        )
        mBCGrad = cute.make_tensor(
            bc_grad_ptr, cute.make_layout(self.bc_shape, stride=self.bc_stride)
        )
        mDParams = cute.make_tensor(
            dparams_ptr,
            cute.make_layout(self.dparams_shape, stride=self.dparams_stride),
        )
        mScaleGrad = cute.make_tensor(
            scale_grad_ptr,
            cute.make_layout(self.scale_grad_shape, stride=self.scale_grad_stride),
        )
        mBiasGrad = cute.make_tensor(
            bias_grad_ptr,
            cute.make_layout(self.bias_grad_shape, stride=self.bias_grad_stride),
        )

        self.pack_bc_kernel(
            mDU,
            mBC,
            mDB,
            mDC,
            mBScale,
            mCScale,
            mValueGrad,
            mBCGrad,
            mScaleGrad,
            self.total_rows,
        ).launch(grid=(self.pack_grid_size, 1, 1), block=(self.pack_block_size, 1, 1))
        self.coeff_kernel(
            mParams,
            mDM,
            mDK,
            mDtBias,
            mGammaBias,
            mOmegaBias,
            mMixRBias,
            mMixThetaBias,
            mMixKPrevBias,
            mMixKCurrBias,
            mDParams,
            mBiasGrad,
            self.total_rows,
        ).launch(grid=(self.coeff_grid_size, 1, 1), block=(self.coeff_block_size, 1, 1))


__all__ = ["ScanPrepBwdFused"]
