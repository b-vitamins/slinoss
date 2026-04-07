# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Split backward kernels for the CuTe scanprep backend."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    COEFF_AUX_DELTA_R,
    COEFF_AUX_DT,
    COEFF_AUX_EXP_TERM,
    COEFF_AUX_KAPPA1_IM,
    COEFF_AUX_KAPPA1_RE,
    COEFF_AUX_KAPPA2_IM,
    COEFF_AUX_KAPPA2_RE,
    COEFF_AUX_LOG_R,
    COEFF_AUX_MIX_R,
    COEFF_AUX_OMEGA,
    COEFF_AUX_OMEGA_DRIVE,
    COEFF_AUX_OMEGA_TANH,
    COEFF_AUX_R,
    COEFF_AUX_RHO_IM,
    COEFF_AUX_RHO_RE,
    COEFF_AUX_R_DIRECT_U,
    COEFF_AUX_THETA,
    COEFF_AUX_ZETA,
    complex_div,
    complex_mul_conj,
    real_mul_conj,
    safe_cast_to_dtype,
    sigmoid,
    softplus,
)


class ScanPrepBwdFused:
    """Backward launcher with explicit row kernels and coefficient gradients."""

    def __init__(
        self,
        *,
        h_size: int,
        p_size: int,
        n_size: int,
        param_dim: int,
        dt_min: float,
        dt_max: float,
        omega_min: float,
        zeta_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        value_warps_per_block: int = 8,
        pack_warps_per_block: int = 12,
        coeff_block_size: int = 512,
    ) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.param_dim = int(param_dim)

        self.value_warps_per_block = int(value_warps_per_block)
        if self.value_warps_per_block <= 0:
            raise ValueError("value_warps_per_block must be positive.")
        self.value_rows_per_round = self.value_warps_per_block
        self.value_block_size = self.value_warps_per_block * 32
        self.value_bt_tile = 256
        self.value_rounds = (
            self.value_bt_tile + self.value_rows_per_round - 1
        ) // self.value_rows_per_round

        self.pack_role_warps = 2
        self.pack_warps_per_block = int(pack_warps_per_block)
        if (
            self.pack_warps_per_block <= 0
            or self.pack_warps_per_block % self.pack_role_warps != 0
        ):
            raise ValueError(
                "pack_warps_per_block must be a positive multiple of pack_role_warps."
            )
        self.pack_rows_per_round = self.pack_warps_per_block // self.pack_role_warps
        self.pack_block_size = self.pack_warps_per_block * 32
        self.pack_bt_tile = 256
        self.pack_rounds = (
            self.pack_bt_tile + self.pack_rows_per_round - 1
        ) // self.pack_rows_per_round

        self.coeff_block_size = int(coeff_block_size)
        if self.coeff_block_size <= 0 or self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a positive multiple of 32.")
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        if self.coeff_head_tile <= 0:
            raise ValueError("coeff_block_size must cover at least one warp.")
        self.coeff_flat_tile = self.coeff_head_tile * self.param_dim
        self.coeff_flat_pad = self.coeff_flat_tile + 1
        self.coeff_smem_bytes = self.coeff_t_tile * self.coeff_flat_pad * 4

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.omega_min = float(omega_min)
        self.zeta_max = float(zeta_max)
        self.omega_mod_scale = 0.25
        self.r_scale = float(r_max - r_min)
        self.eps = float(eps)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    def _coeff_shared_storage(self):
        coeff_layout = cute.make_layout(
            (self.coeff_t_tile, self.coeff_flat_pad),
            stride=(self.coeff_flat_pad, 1),
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sDParams": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(coeff_layout)],
                16,
            ]
        }

        return cute.struct(SharedStorage)

    @cute.kernel
    def value_grad_kernel(
        self,
        mDU: cute.Tensor,
        mValueGrad: cute.Tensor,
        total_bt_,
        t_size_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_bt, h, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32

        if h < self.h_size:
            p_base = h * self.p_size
            num_p_iters = (self.p_size + 31) // 32
            for round_iter in cutlass.range_constexpr(self.value_rounds):
                bt = (
                    block_bt * self.value_bt_tile
                    + round_iter * self.value_rows_per_round
                    + warp
                )
                if bt < total_bt_:
                    b = bt // t_size_
                    t = bt - b * t_size_
                    for p_iter in cutlass.range_constexpr(num_p_iters):
                        p = lane + p_iter * 32
                        if p < self.p_size:
                            mValueGrad[b, t, p_base + p] = mDU[b, h, t, p]

    @cute.kernel
    def pack_grads_kernel(
        self,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mBCGrad: cute.Tensor,
        total_bt_,
        t_size_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_bt, h, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        role = warp // self.pack_rows_per_round
        row_local = warp - role * self.pack_rows_per_round

        if h < self.h_size and role < self.pack_role_warps:
            num_n_iters = (self.n_size + 31) // 32
            for round_iter in cutlass.range_constexpr(self.pack_rounds):
                bt = (
                    block_bt * self.pack_bt_tile
                    + round_iter * self.pack_rows_per_round
                    + row_local
                )
                if bt < total_bt_:
                    b = bt // t_size_
                    t = bt - b * t_size_
                    if role == 0:
                        for n_iter in cutlass.range_constexpr(num_n_iters):
                            n = lane + n_iter * 32
                            if n < self.n_size:
                                mBCGrad[b, t, h, 0, n] = mDB[b, h, t, 2 * n]
                                mBCGrad[b, t, h, 1, n] = mDB[b, h, t, 2 * n + 1]
                    else:
                        for n_iter in cutlass.range_constexpr(num_n_iters):
                            n = lane + n_iter * 32
                            if n < self.n_size:
                                mBCGrad[b, t, h, 2, n] = mDC[b, h, t, 2 * n]
                                mBCGrad[b, t, h, 3, n] = mDC[b, h, t, 2 * n + 1]

    @cute.kernel
    def coeff_grad_kernel(
        self,
        mCoeffAux: cute.Tensor,
        mDM: cute.Tensor,
        mDK: cute.Tensor,
        mDtBias: cute.Tensor,
        mOmegaNaturalBias: cute.Tensor,
        mOmegaSign: cute.Tensor,
        mDParams: cute.Tensor,
        mBiasGrad: cute.Tensor,
        total_bt_,
        t_size_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        bt = block_x * self.coeff_t_tile + lane
        h_base = block_y * self.coeff_head_tile
        h = h_base + warp

        smem = cutlass.utils.SmemAllocator()
        sDParams = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.coeff_t_tile, self.coeff_flat_pad),
                stride=(self.coeff_flat_pad, 1),
            ),
            16,
        )

        if bt < total_bt_ and h < self.h_size:
            b = bt // t_size_
            t = bt - b * t_size_

            zeta = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_ZETA, t])
            omega_tanh = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_OMEGA_TANH, t])
            omega_drive = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_OMEGA_DRIVE, t])
            omega = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_OMEGA, t])
            r_direct_u = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_R_DIRECT_U, t])
            mix_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_MIX_R, t])
            dt = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DT, t])
            exp_term = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t])
            delta_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DELTA_R, t])
            r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_R, t])
            theta = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_THETA, t])
            rho_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_RHO_RE, t])
            rho_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_RHO_IM, t])
            log_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_LOG_R, t])
            kappa1_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA1_RE, t])
            kappa1_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA1_IM, t])
            kappa2_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA2_RE, t])
            kappa2_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA2_IM, t])

            z_re = log_r
            z_im = theta
            z2_re = z_re * z_re - z_im * z_im
            z2_im = cutlass.Float32(2.0) * z_re * z_im
            z_norm_sq = z_re * z_re + z_im * z_im

            g_rho_re = cutlass.Float32(mDM[b, h, t, 0])
            g_rho_im = cutlass.Float32(mDM[b, h, t, 1])
            g_k_prev_re = cutlass.Float32(mDK[b, h, t, 0, 0])
            g_k_prev_im = cutlass.Float32(mDK[b, h, t, 0, 1])
            g_k_curr_re = cutlass.Float32(mDK[b, h, t, 1, 0])
            g_k_curr_im = cutlass.Float32(mDK[b, h, t, 1, 1])

            kappa_diff_re = kappa1_re - kappa2_re
            kappa_diff_im = kappa1_im - kappa2_im
            g_dt = real_mul_conj(
                g_k_prev_re,
                g_k_prev_im,
                kappa2_re,
                kappa2_im,
            ) + real_mul_conj(
                g_k_curr_re,
                g_k_curr_im,
                kappa_diff_re,
                kappa_diff_im,
            )

            g_kappa1_re = g_k_curr_re * dt
            g_kappa1_im = g_k_curr_im * dt
            g_kappa2_re = (g_k_prev_re - g_k_curr_re) * dt
            g_kappa2_im = (g_k_prev_im - g_k_curr_im) * dt
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

            one_minus_mix_r = cutlass.Float32(1.0) - mix_r
            g_r_direct = g_r * one_minus_mix_r
            g_r_struct = g_r * mix_r
            g_mix_r = g_r * delta_r

            g_r_direct_u = g_r_direct * cutlass.Float32(self.r_scale)
            g_exp_term = g_r_struct * cutlass.Float32(self.r_scale)
            g_x = g_exp_term * exp_term
            g_gamma = g_x * (-dt)
            gamma = zeta * omega_drive
            g_dt = g_dt + g_x * (-gamma)
            g_omega = g_theta * dt
            g_dt = g_dt + g_theta * omega

            sign = cutlass.Float32(mOmegaSign[h])
            under_raw = cutlass.Float32(1.0) - zeta * zeta
            sqrt_under = cutlass.max(cutlass.Float32(self.eps), under_raw)
            sqrt_under = cute_math.sqrt(sqrt_under)

            g_omega_drive = g_gamma * zeta + g_omega * sign * sqrt_under
            g_zeta = g_gamma * omega_drive
            if under_raw > cutlass.Float32(self.eps):
                g_zeta = g_zeta + g_omega * sign * omega_drive * (-zeta / sqrt_under)

            omega_natural_bias = cutlass.Float32(mOmegaNaturalBias[h])
            omega_natural = cutlass.Float32(self.omega_min) + softplus(
                omega_natural_bias
            )
            omega_scale = cute_math.exp(
                cutlass.Float32(self.omega_mod_scale) * omega_tanh
            )
            g_omega_natural = g_omega_drive * omega_scale
            g_omega_scale = g_omega_drive * omega_natural

            g_omega_mod_raw = (
                g_omega_scale
                * omega_scale
                * cutlass.Float32(self.omega_mod_scale)
                * (cutlass.Float32(1.0) - omega_tanh * omega_tanh)
            )
            zeta_sigmoid = zeta / cutlass.Float32(self.zeta_max)
            g_zeta_raw = (
                g_zeta
                * cutlass.Float32(self.zeta_max)
                * zeta_sigmoid
                * (cutlass.Float32(1.0) - zeta_sigmoid)
            )
            g_r_raw = g_r_direct_u * r_direct_u * (cutlass.Float32(1.0) - r_direct_u)
            g_mix_r_raw = g_mix_r * mix_r * (cutlass.Float32(1.0) - mix_r)

            dt_u = (dt - cutlass.Float32(self.dt_min)) / cutlass.Float32(self.dt_scale)
            g_dt_bias = (
                g_dt
                * cutlass.Float32(self.dt_scale)
                * dt_u
                * (cutlass.Float32(1.0) - dt_u)
            )
            g_omega_natural_bias = g_omega_natural * sigmoid(omega_natural_bias)

            flat_base = warp * self.param_dim
            sDParams[lane, flat_base + 0] = g_zeta_raw
            sDParams[lane, flat_base + 1] = g_omega_mod_raw
            sDParams[lane, flat_base + 2] = g_r_raw
            sDParams[lane, flat_base + 3] = g_mix_r_raw

            bias_base = h * 5
            cute.arch.atomic_add(
                (mBiasGrad.iterator + bias_base + 0).llvm_ptr, g_dt_bias
            )
            cute.arch.atomic_add(
                (mBiasGrad.iterator + bias_base + 1).llvm_ptr, g_zeta_raw
            )
            cute.arch.atomic_add(
                (mBiasGrad.iterator + bias_base + 2).llvm_ptr, g_omega_mod_raw
            )
            cute.arch.atomic_add(
                (mBiasGrad.iterator + bias_base + 3).llvm_ptr, g_omega_natural_bias
            )
            cute.arch.atomic_add(
                (mBiasGrad.iterator + bias_base + 4).llvm_ptr, g_mix_r_raw
            )

        cute.arch.sync_threads()

        num_store_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_store_flat_iters = (self.coeff_flat_tile + 31) // 32
        for store_t_iter in cutlass.range_constexpr(num_store_t_iters):
            store_t_local = warp + store_t_iter * self.coeff_head_tile
            store_bt = block_x * self.coeff_t_tile + store_t_local
            if store_t_local < self.coeff_t_tile and store_bt < total_bt_:
                store_b = store_bt // t_size_
                store_t = store_bt - store_b * t_size_
                p_base = h_base * self.param_dim
                for flat_iter in cutlass.range_constexpr(num_store_flat_iters):
                    flat = lane + flat_iter * 32
                    if flat < self.coeff_flat_tile:
                        store_h = h_base + flat // self.param_dim
                        if store_h < self.h_size:
                            mDParams[store_b, store_t, p_base + flat] = (
                                safe_cast_to_dtype(
                                    sDParams[store_t_local, flat],
                                    mDParams.element_type,
                                )
                            )

    @cute.jit
    def __call__(
        self,
        du: cute.Tensor,
        db: cute.Tensor,
        dc: cute.Tensor,
        coeff_aux: cute.Tensor,
        dm: cute.Tensor,
        dk: cute.Tensor,
        dt_bias: cute.Tensor,
        omega_natural_bias: cute.Tensor,
        omega_sign: cute.Tensor,
        value_grad: cute.Tensor,
        bc_grad: cute.Tensor,
        dparams: cute.Tensor,
        bias_grad: cute.Tensor,
    ):
        batch = cute.size(bc_grad, mode=[0])
        t_size = cute.size(bc_grad, mode=[1])
        total_bt = batch * t_size
        value_grid_x = (total_bt + self.value_bt_tile - 1) // self.value_bt_tile
        pack_grid_x = (total_bt + self.pack_bt_tile - 1) // self.pack_bt_tile
        coeff_grid_x = (total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile
        coeff_grid_y = (self.h_size + self.coeff_head_tile - 1) // self.coeff_head_tile

        mValueGrad = cute.make_tensor(
            value_grad.iterator,
            cute.make_layout(
                (batch, t_size, self.h_size * self.p_size),
                stride=(
                    t_size * self.h_size * self.p_size,
                    self.h_size * self.p_size,
                    1,
                ),
            ),
        )
        mDParams = cute.make_tensor(
            dparams.iterator,
            cute.make_layout(
                (batch, t_size, self.h_size * self.param_dim),
                stride=(
                    t_size * self.h_size * self.param_dim,
                    self.h_size * self.param_dim,
                    1,
                ),
            ),
        )
        mBiasGrad = cute.make_tensor(
            bias_grad.iterator,
            cute.make_layout((self.h_size, 5), stride=(5, 1)),
        )
        coeff_smem_bytes = int(self._coeff_shared_storage().size_in_bytes())

        self.value_grad_kernel(
            du,
            mValueGrad,
            total_bt,
            t_size,
        ).launch(
            grid=(value_grid_x, self.h_size, 1),
            block=(self.value_block_size, 1, 1),
        )
        self.pack_grads_kernel(
            db,
            dc,
            bc_grad,
            total_bt,
            t_size,
        ).launch(
            grid=(pack_grid_x, self.h_size, 1),
            block=(self.pack_block_size, 1, 1),
        )
        self.coeff_grad_kernel(
            coeff_aux,
            dm,
            dk,
            dt_bias,
            omega_natural_bias,
            omega_sign,
            mDParams,
            mBiasGrad,
            total_bt,
            t_size,
        ).launch(
            grid=(coeff_grid_x, coeff_grid_y, 1),
            block=(self.coeff_block_size, 1, 1),
            smem=coeff_smem_bytes,
        )


__all__ = ["ScanPrepBwdFused"]
