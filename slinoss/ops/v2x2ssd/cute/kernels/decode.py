# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Standalone CuTe decode kernel for the recurrent SLinOSS middle."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from slinoss.ops.scanprep.cute.common import (
    complex_div,
    complex_mul,
    lerp,
    make_row_major_stride,
    principal_angle,
    real_mul_conj,
    sigmoid,
    softplus,
)


class MixerDecodeStepFwd:
    """One-token recurrent middle for SLinOSS decode.

    Contract:
    - ``value`` / ``gate``: ``(B, H, P)``
    - ``params``: ``(B, H, 13)``
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
        spec: tuple[int, int, int, int],
        state_stride: tuple[int, int, int, int] | None = None,
        final_state_stride: tuple[int, int, int, int] | None = None,
        workers_per_p: int = 1,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
    ) -> None:
        batch, heads, p_size, n_size = spec
        self.batch = int(batch)
        self.heads = int(heads)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.d_size = int(2 * n_size)
        self.workers_per_p = int(workers_per_p)
        if self.workers_per_p < 1 or self.n_size % self.workers_per_p != 0:
            raise ValueError(
                "workers_per_p must be positive and divide the decode N dimension."
            )
        self.block_threads = int(self.p_size * self.workers_per_p)
        self.normalize_bc = bool(normalize_bc)

        self.value_shape = (self.batch, self.heads, self.p_size)
        self.value_stride = make_row_major_stride(self.value_shape)
        self.params_shape = (self.batch, self.heads, 13)
        self.params_stride = make_row_major_stride(self.params_shape)
        self.bc_shape = (self.batch, self.heads, 4, self.n_size)
        self.bc_stride = make_row_major_stride(self.bc_shape)
        self.gate_shape = self.value_shape
        self.gate_stride = self.value_stride
        self.skip_shape = (self.heads, self.p_size)
        self.skip_stride = make_row_major_stride(self.skip_shape)
        self.state_shape = (self.batch, self.heads, self.p_size, self.d_size)
        default_state_stride = make_row_major_stride(self.state_shape)
        self.state_stride = (
            default_state_stride
            if state_stride is None
            else tuple(int(v) for v in state_stride)
        )
        self.final_state_stride = (
            self.state_stride
            if final_state_stride is None
            else tuple(int(v) for v in final_state_stride)
        )
        self.prev_b_shape = (self.batch, self.heads, self.d_size)
        self.prev_b_stride = make_row_major_stride(self.prev_b_shape)
        self.prev_u_shape = self.value_shape
        self.prev_u_stride = self.value_stride
        self.y_shape = self.value_shape
        self.y_stride = self.value_stride
        self.bias_shape = (self.heads,)
        self.bias_stride = make_row_major_stride(self.bias_shape)
        self.scale_shape = (self.heads, 2, self.n_size)
        self.scale_stride = make_row_major_stride(self.scale_shape)

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    @cute.jit
    def __call__(
        self,
        value_ptr: cute.Pointer,
        params_ptr: cute.Pointer,
        bc_ptr: cute.Pointer,
        gate_ptr: cute.Pointer,
        skip_ptr: cute.Pointer,
        state_ptr: cute.Pointer,
        b_prev_ptr: cute.Pointer,
        u_prev_ptr: cute.Pointer,
        dt_bias_ptr: cute.Pointer,
        gamma_bias_ptr: cute.Pointer,
        omega_bias_ptr: cute.Pointer,
        mix_r_bias_ptr: cute.Pointer,
        mix_theta_bias_ptr: cute.Pointer,
        mix_k_prev_bias_ptr: cute.Pointer,
        mix_k_curr_bias_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        c_scale_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        final_state_ptr: cute.Pointer,
        b_last_ptr: cute.Pointer,
        u_last_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        mValue = cute.make_tensor(
            value_ptr, cute.make_layout(self.value_shape, stride=self.value_stride)
        )
        mParams = cute.make_tensor(
            params_ptr, cute.make_layout(self.params_shape, stride=self.params_stride)
        )
        mBC = cute.make_tensor(
            bc_ptr, cute.make_layout(self.bc_shape, stride=self.bc_stride)
        )
        mGate = cute.make_tensor(
            gate_ptr, cute.make_layout(self.gate_shape, stride=self.gate_stride)
        )
        mSkip = cute.make_tensor(
            skip_ptr, cute.make_layout(self.skip_shape, stride=self.skip_stride)
        )
        mState = cute.make_tensor(
            state_ptr, cute.make_layout(self.state_shape, stride=self.state_stride)
        )
        mBPrev = cute.make_tensor(
            b_prev_ptr, cute.make_layout(self.prev_b_shape, stride=self.prev_b_stride)
        )
        mUPrev = cute.make_tensor(
            u_prev_ptr, cute.make_layout(self.prev_u_shape, stride=self.prev_u_stride)
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
        mBScale = cute.make_tensor(
            b_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mCScale = cute.make_tensor(
            c_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mY = cute.make_tensor(
            y_ptr, cute.make_layout(self.y_shape, stride=self.y_stride)
        )
        mFinalState = cute.make_tensor(
            final_state_ptr,
            cute.make_layout(self.state_shape, stride=self.final_state_stride),
        )
        mBLast = cute.make_tensor(
            b_last_ptr, cute.make_layout(self.prev_b_shape, stride=self.prev_b_stride)
        )
        mULast = cute.make_tensor(
            u_last_ptr, cute.make_layout(self.prev_u_shape, stride=self.prev_u_stride)
        )

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
            mMixThetaBias,
            mMixKPrevBias,
            mMixKCurrBias,
            mBScale,
            mCScale,
            mY,
            mFinalState,
            mBLast,
            mULast,
        ).launch(
            grid=(self.batch, self.heads, 1),
            block=(self.block_threads, 1, 1),
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
        mMixThetaBias: cute.Tensor,
        mMixKPrevBias: cute.Tensor,
        mMixKCurrBias: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mY: cute.Tensor,
        mFinalState: cute.Tensor,
        mBLast: cute.Tensor,
        mULast: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, hidx, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        vec_layout = cute.make_layout((self.n_size,))
        scalar_layout = cute.make_layout((1,))
        acc_layout = cute.make_layout((self.workers_per_p, self.p_size))
        b_re = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        b_im = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        c_re = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        c_im = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        beta_prev_re = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        beta_prev_im = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        beta_curr_re = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        beta_curr_im = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=vec_layout,
            byte_alignment=16,
            swizzle=None,
        )
        rho_re_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        rho_im_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        inv0_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        inv1_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        inv2_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        inv3_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        tap_prev_re_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        tap_prev_im_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        tap_curr_re_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        tap_curr_im_s = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=scalar_layout,
            byte_alignment=4,
            swizzle=None,
        )
        acc_partial = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=acc_layout,
            byte_alignment=16,
            swizzle=None,
        )

        if tidx == 0:
            s0 = cutlass.Float32(0.0)
            s1 = cutlass.Float32(0.0)
            s2 = cutlass.Float32(0.0)
            s3 = cutlass.Float32(0.0)
            for n in cutlass.range(self.n_size, unroll_full=True):
                x0 = cutlass.Float32(mBC[bidx, hidx, 0, n])
                x1 = cutlass.Float32(mBC[bidx, hidx, 1, n])
                x2 = cutlass.Float32(mBC[bidx, hidx, 2, n])
                x3 = cutlass.Float32(mBC[bidx, hidx, 3, n])
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
            inv0_s[0] = inv0
            inv1_s[0] = inv1
            inv2_s[0] = inv2
            inv3_s[0] = inv3

            dt_raw = cutlass.Float32(mParams[bidx, hidx, 0]) + cutlass.Float32(
                mDtBias[hidx]
            )
            gamma_raw = cutlass.Float32(mParams[bidx, hidx, 1]) + cutlass.Float32(
                mGammaBias[hidx]
            )
            omega_raw = cutlass.Float32(mParams[bidx, hidx, 2]) + cutlass.Float32(
                mOmegaBias[hidx]
            )
            r_raw = cutlass.Float32(mParams[bidx, hidx, 3])
            theta_raw = cutlass.Float32(mParams[bidx, hidx, 4])
            mix_r_raw = cutlass.Float32(mParams[bidx, hidx, 5]) + cutlass.Float32(
                mMixRBias[hidx]
            )
            mix_theta_raw = cutlass.Float32(mParams[bidx, hidx, 6]) + cutlass.Float32(
                mMixThetaBias[hidx]
            )
            mix_k_prev_raw = cutlass.Float32(mParams[bidx, hidx, 7]) + cutlass.Float32(
                mMixKPrevBias[hidx]
            )
            mix_k_curr_raw = cutlass.Float32(mParams[bidx, hidx, 8]) + cutlass.Float32(
                mMixKCurrBias[hidx]
            )
            k_prev_re_raw = cutlass.Float32(mParams[bidx, hidx, 9])
            k_prev_im_raw = cutlass.Float32(mParams[bidx, hidx, 10])
            k_curr_re_raw = cutlass.Float32(mParams[bidx, hidx, 11])
            k_curr_im_raw = cutlass.Float32(mParams[bidx, hidx, 12])

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
            rho_re_s[0] = rho_re
            rho_im_s[0] = rho_im

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

            tap_prev_re = lerp(k_prev_learned_re, k_prev_re, mix_k_prev)
            tap_prev_im = lerp(k_prev_learned_im, k_prev_im, mix_k_prev)
            tap_curr_re = lerp(k_curr_learned_re, k_curr_re, mix_k_curr)
            tap_curr_im = lerp(k_curr_learned_im, k_curr_im, mix_k_curr)
            tap_prev_re_s[0] = tap_prev_re
            tap_prev_im_s[0] = tap_prev_im
            tap_curr_re_s[0] = tap_curr_re
            tap_curr_im_s[0] = tap_curr_im

        cute.arch.barrier()

        if tidx < self.n_size:
            n = tidx
            b_re_v = cutlass.Float32(mBC[bidx, hidx, 0, n]) * inv0_s[0]
            b_im_v = cutlass.Float32(mBC[bidx, hidx, 1, n]) * inv1_s[0]
            c_re_v = cutlass.Float32(mBC[bidx, hidx, 2, n]) * inv2_s[0]
            c_im_v = cutlass.Float32(mBC[bidx, hidx, 3, n]) * inv3_s[0]
            if self.normalize_bc:
                b_re_v = b_re_v * cutlass.Float32(mBScale[hidx, 0, n])
                b_im_v = b_im_v * cutlass.Float32(mBScale[hidx, 1, n])
                c_re_v = c_re_v * cutlass.Float32(mCScale[hidx, 0, n])
                c_im_v = c_im_v * cutlass.Float32(mCScale[hidx, 1, n])
            b_re[n] = b_re_v
            b_im[n] = b_im_v
            c_re[n] = c_re_v
            c_im[n] = c_im_v

            prev_re = cutlass.Float32(mBPrev[bidx, hidx, 2 * n])
            prev_im = cutlass.Float32(mBPrev[bidx, hidx, 2 * n + 1])
            beta_prev_re_v, beta_prev_im_v = complex_mul(
                tap_prev_re_s[0],
                tap_prev_im_s[0],
                prev_re,
                prev_im,
            )
            beta_curr_re_v, beta_curr_im_v = complex_mul(
                tap_curr_re_s[0],
                tap_curr_im_s[0],
                b_re_v,
                b_im_v,
            )
            beta_prev_re[n] = beta_prev_re_v
            beta_prev_im[n] = beta_prev_im_v
            beta_curr_re[n] = beta_curr_re_v
            beta_curr_im[n] = beta_curr_im_v
            mBLast[bidx, hidx, 2 * n] = b_re_v.to(mBLast.element_type)
            mBLast[bidx, hidx, 2 * n + 1] = b_im_v.to(mBLast.element_type)

        cute.arch.barrier()

        if tidx < self.block_threads:
            worker = tidx // self.p_size
            p = tidx - worker * self.p_size
            u_prev = cutlass.Float32(mUPrev[bidx, hidx, p])
            u_curr = cutlass.Float32(mValue[bidx, hidx, p])
            acc = cutlass.Float32(0.0)
            rho_re = rho_re_s[0]
            rho_im = rho_im_s[0]

            for n_iter in cutlass.range_constexpr(self.n_size // self.workers_per_p):
                n = worker + n_iter * self.workers_per_p
                z_re = cutlass.Float32(mState[bidx, hidx, p, 2 * n])
                z_im = cutlass.Float32(mState[bidx, hidx, p, 2 * n + 1])
                mz_re, mz_im = complex_mul(rho_re, rho_im, z_re, z_im)
                drive_re = u_prev * beta_prev_re[n] + u_curr * beta_curr_re[n]
                drive_im = u_prev * beta_prev_im[n] + u_curr * beta_curr_im[n]
                out_re = mz_re + drive_re
                out_im = mz_im + drive_im
                mFinalState[bidx, hidx, p, 2 * n] = out_re.to(mFinalState.element_type)
                mFinalState[bidx, hidx, p, 2 * n + 1] = out_im.to(
                    mFinalState.element_type
                )
                acc = acc + real_mul_conj(
                    out_re,
                    out_im,
                    c_re[n],
                    c_im[n],
                )

            acc_partial[worker, p] = acc

        if self.workers_per_p > 1:
            cute.arch.barrier()

        if tidx < self.p_size:
            p = tidx
            acc = cutlass.Float32(0.0)
            for worker in cutlass.range_constexpr(self.workers_per_p):
                acc = acc + acc_partial[worker, p]
            u_curr = cutlass.Float32(mValue[bidx, hidx, p])
            gate = cutlass.Float32(mGate[bidx, hidx, p])
            silu_gate = gate * sigmoid(gate)
            y = (acc + u_curr * cutlass.Float32(mSkip[hidx, p])) * silu_gate
            mY[bidx, hidx, p] = y.to(mY.element_type)
            mULast[bidx, hidx, p] = mValue[bidx, hidx, p]


__all__ = ["MixerDecodeStepFwd"]
