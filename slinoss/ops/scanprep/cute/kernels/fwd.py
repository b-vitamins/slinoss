# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Fused forward kernel for the CuTe scanprep backend."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    complex_div,
    lerp,
    make_row_major_stride,
    principal_angle,
    sigmoid,
    softplus,
)


class ScanPrepFwdFused:
    """Thin host-JIT wrapper around the fused scanprep forward row kernel."""

    def __init__(
        self,
        *,
        spec: tuple[int, int, int, int, int],
        params_in_stride: tuple[int, int, int] | None = None,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
        block_size: int = 96,
    ) -> None:
        batch, t_size, h_size, p_size, n_size = spec
        self.batch = int(batch)
        self.t_size = int(t_size)
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.normalize_bc = bool(normalize_bc)

        self.value_shape = (self.batch, self.t_size, self.h_size * self.p_size)
        self.value_stride = make_row_major_stride(self.value_shape)
        self.u_shape = (self.batch, self.h_size, self.t_size, self.p_size)
        self.u_stride = make_row_major_stride(self.u_shape)
        self.bc_shape = (self.batch, self.t_size, self.h_size, 4, self.n_size)
        self.bc_stride = make_row_major_stride(self.bc_shape)
        self.scale_shape = (self.h_size, 2, self.n_size)
        self.scale_stride = make_row_major_stride(self.scale_shape)
        self.bc_out_shape = (self.batch, self.h_size, self.t_size, 2 * self.n_size)
        self.bc_out_stride = make_row_major_stride(self.bc_out_shape)
        self.params_shape = (self.batch, self.t_size, self.h_size * 13)
        self.params_stride = (
            tuple(int(s) for s in params_in_stride)
            if params_in_stride is not None
            else make_row_major_stride(self.params_shape)
        )
        self.bias_shape = (self.h_size,)
        self.bias_stride = make_row_major_stride(self.bias_shape)
        self.m_shape = (self.batch, self.h_size, self.t_size, 2)
        self.m_stride = make_row_major_stride(self.m_shape)
        self.k_shape = (self.batch, self.h_size, self.t_size, 2, 2)
        self.k_stride = make_row_major_stride(self.k_shape)

        self.total_rows = self.batch * self.h_size * self.t_size
        self.block_size = int(block_size)
        self.grid_size = (self.total_rows + self.block_size - 1) // self.block_size

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    @cute.kernel
    def kernel(
        self,
        mValue: cute.Tensor,
        mBC: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mGammaBias: cute.Tensor,
        mOmegaBias: cute.Tensor,
        mMixRBias: cute.Tensor,
        mMixThetaBias: cute.Tensor,
        mMixKPrevBias: cute.Tensor,
        mMixKCurrBias: cute.Tensor,
        mU: cute.Tensor,
        mBOut: cute.Tensor,
        mCOut: cute.Tensor,
        mMOut: cute.Tensor,
        mKOut: cute.Tensor,
        total_rows_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * self.block_size + tidx
        if row < total_rows_:
            rows_per_batch = self.h_size * self.t_size
            b = row // rows_per_batch
            rem = row - b * rows_per_batch
            h = rem // self.t_size
            t = rem - h * self.t_size

            for p in cutlass.range_constexpr(self.p_size):
                mU[b, h, t, p] = mValue[b, t, h * self.p_size + p]

            if self.normalize_bc:
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                for n in cutlass.range_constexpr(self.n_size):
                    x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                    x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                    x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                    x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                    s0 = s0 + x0 * x0
                    s1 = s1 + x1 * x1
                    s2 = s2 + x2 * x2
                    s3 = s3 + x3 * x3
                denom = cutlass.Float32(self.n_size)
                eps_bc = cutlass.Float32(1.0e-5)
                inv0 = cute.rsqrt(s0 / denom + eps_bc)
                inv1 = cute.rsqrt(s1 / denom + eps_bc)
                inv2 = cute.rsqrt(s2 / denom + eps_bc)
                inv3 = cute.rsqrt(s3 / denom + eps_bc)
                for n in cutlass.range_constexpr(self.n_size):
                    b0 = cutlass.Float32(mBC[b, t, h, 0, n]) * inv0
                    b1 = cutlass.Float32(mBC[b, t, h, 1, n]) * inv1
                    c0 = cutlass.Float32(mBC[b, t, h, 2, n]) * inv2
                    c1 = cutlass.Float32(mBC[b, t, h, 3, n]) * inv3
                    mBOut[b, h, t, 2 * n] = (b0 * cutlass.Float32(mBScale[h, 0, n])).to(
                        mBOut.element_type
                    )
                    mBOut[b, h, t, 2 * n + 1] = (
                        b1 * cutlass.Float32(mBScale[h, 1, n])
                    ).to(mBOut.element_type)
                    mCOut[b, h, t, 2 * n] = (c0 * cutlass.Float32(mCScale[h, 0, n])).to(
                        mCOut.element_type
                    )
                    mCOut[b, h, t, 2 * n + 1] = (
                        c1 * cutlass.Float32(mCScale[h, 1, n])
                    ).to(mCOut.element_type)
            else:
                for n in cutlass.range_constexpr(self.n_size):
                    mBOut[b, h, t, 2 * n] = cutlass.Float32(mBC[b, t, h, 0, n]).to(
                        mBOut.element_type
                    )
                    mBOut[b, h, t, 2 * n + 1] = cutlass.Float32(mBC[b, t, h, 1, n]).to(
                        mBOut.element_type
                    )
                    mCOut[b, h, t, 2 * n] = cutlass.Float32(mBC[b, t, h, 2, n]).to(
                        mCOut.element_type
                    )
                    mCOut[b, h, t, 2 * n + 1] = cutlass.Float32(mBC[b, t, h, 3, n]).to(
                        mCOut.element_type
                    )

            p_base = h * 13
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
            theta_direct = cutlass.Float32(self.theta_bound) * cute_math.tanh(theta_raw)
            mix_r = sigmoid(mix_r_raw)
            mix_theta = sigmoid(mix_theta_raw)
            mix_k_prev = sigmoid(mix_k_prev_raw)
            mix_k_curr = sigmoid(mix_k_curr_raw)
            k_prev_learned_re = cutlass.Float32(self.k_max) * cute_math.tanh(
                k_prev_re_raw
            )
            k_prev_learned_im = cutlass.Float32(self.k_max) * cute_math.tanh(
                k_prev_im_raw
            )
            k_curr_learned_re = cutlass.Float32(self.k_max) * cute_math.tanh(
                k_curr_re_raw
            )
            k_curr_learned_im = cutlass.Float32(self.k_max) * cute_math.tanh(
                k_curr_im_raw
            )

            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            r_struct = cutlass.Float32(self.r_min) + cutlass.Float32(
                self.r_scale
            ) * cute_math.exp(-(gamma * dt))
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

    @cute.jit
    def __call__(
        self,
        value_ptr: cute.Pointer,
        bc_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        c_scale_ptr: cute.Pointer,
        params_ptr: cute.Pointer,
        dt_bias_ptr: cute.Pointer,
        gamma_bias_ptr: cute.Pointer,
        omega_bias_ptr: cute.Pointer,
        mix_r_bias_ptr: cute.Pointer,
        mix_theta_bias_ptr: cute.Pointer,
        mix_k_prev_bias_ptr: cute.Pointer,
        mix_k_curr_bias_ptr: cute.Pointer,
        u_ptr: cute.Pointer,
        m_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
    ):
        mValue = cute.make_tensor(
            value_ptr, cute.make_layout(self.value_shape, stride=self.value_stride)
        )
        mU = cute.make_tensor(
            u_ptr, cute.make_layout(self.u_shape, stride=self.u_stride)
        )
        mBC = cute.make_tensor(
            bc_ptr, cute.make_layout(self.bc_shape, stride=self.bc_stride)
        )
        mBScale = cute.make_tensor(
            b_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mCScale = cute.make_tensor(
            c_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mB = cute.make_tensor(
            b_ptr, cute.make_layout(self.bc_out_shape, stride=self.bc_out_stride)
        )
        mC = cute.make_tensor(
            c_ptr, cute.make_layout(self.bc_out_shape, stride=self.bc_out_stride)
        )
        mParams = cute.make_tensor(
            params_ptr, cute.make_layout(self.params_shape, stride=self.params_stride)
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
        mM = cute.make_tensor(
            m_ptr, cute.make_layout(self.m_shape, stride=self.m_stride)
        )
        mK = cute.make_tensor(
            k_ptr, cute.make_layout(self.k_shape, stride=self.k_stride)
        )
        self.kernel(
            mValue,
            mBC,
            mBScale,
            mCScale,
            mParams,
            mDtBias,
            mGammaBias,
            mOmegaBias,
            mMixRBias,
            mMixThetaBias,
            mMixKPrevBias,
            mMixKCurrBias,
            mU,
            mB,
            mC,
            mM,
            mK,
            self.total_rows,
        ).launch(grid=(self.grid_size, 1, 1), block=(self.block_size, 1, 1))


__all__ = ["ScanPrepFwdFused"]
