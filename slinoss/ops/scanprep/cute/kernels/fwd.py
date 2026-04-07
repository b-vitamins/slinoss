# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Split forward kernels for the CuTe scanprep backend."""

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
    SCANPREP_PARAM_DIM,
    complex_div,
    lerp,
    principal_angle,
    safe_cast_to_dtype,
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
        store_coeff_aux: bool,
        dt_min: float,
        dt_max: float,
        omega_min: float,
        zeta_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        pack_warps_per_block: int = 8,
        coeff_block_size: int = 256,
    ) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
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
        self.coeff_smem_bytes = (
            self.coeff_head_tile * SCANPREP_PARAM_DIM * (self.coeff_t_tile + 1) * 4
        )

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.omega_min = float(omega_min)
        self.zeta_max = float(zeta_max)
        self.omega_mod_scale = 0.25
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.eps = float(eps)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    @cute.kernel
    def pack_kernel(
        self,
        mValue: cute.Tensor,
        mBC: cute.Tensor,
        mU: cute.Tensor,
        mBOut: cute.Tensor,
        mCOut: cute.Tensor,
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

            num_n_iters = (self.n_size + 31) // 32
            for n_iter in cutlass.range_constexpr(num_n_iters):
                n = lane + n_iter * 32
                if n < self.n_size:
                    mBOut[b, h, t, 2 * n] = safe_cast_to_dtype(
                        cutlass.Float32(mBC[b, t, h, 0, n]),
                        mBOut.element_type,
                    )
                    mBOut[b, h, t, 2 * n + 1] = safe_cast_to_dtype(
                        cutlass.Float32(mBC[b, t, h, 1, n]),
                        mBOut.element_type,
                    )
                    mCOut[b, h, t, 2 * n] = safe_cast_to_dtype(
                        cutlass.Float32(mBC[b, t, h, 2, n]),
                        mCOut.element_type,
                    )
                    mCOut[b, h, t, 2 * n + 1] = safe_cast_to_dtype(
                        cutlass.Float32(mBC[b, t, h, 3, n]),
                        mCOut.element_type,
                    )

    @cute.kernel
    def coeff_kernel(
        self,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mZetaBias: cute.Tensor,
        mOmegaModBias: cute.Tensor,
        mOmegaNaturalBias: cute.Tensor,
        mMixRBias: cute.Tensor,
        mOmegaSign: cute.Tensor,
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
                (self.coeff_head_tile, SCANPREP_PARAM_DIM, self.coeff_t_tile + 1),
                stride=(
                    SCANPREP_PARAM_DIM * (self.coeff_t_tile + 1),
                    self.coeff_t_tile + 1,
                    1,
                ),
            ),
            16,
        )

        h_base = block_y * self.coeff_head_tile
        num_load_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_load_flat_iters = (self.coeff_head_tile * SCANPREP_PARAM_DIM + 31) // 32
        for load_t_iter in cutlass.range_constexpr(num_load_t_iters):
            load_t_local = warp + load_t_iter * self.coeff_head_tile
            load_bt = block_x * self.coeff_t_tile + load_t_local
            if load_t_local < self.coeff_t_tile and load_bt < total_bt_:
                load_b = load_bt // t_size_
                load_t = load_bt - load_b * t_size_
                for flat_iter in cutlass.range_constexpr(num_load_flat_iters):
                    flat = lane + flat_iter * 32
                    if flat < self.coeff_head_tile * SCANPREP_PARAM_DIM:
                        local_h = flat // SCANPREP_PARAM_DIM
                        param_idx = flat - local_h * SCANPREP_PARAM_DIM
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

            zeta_raw = sParams[warp, 0, lane] + cutlass.Float32(mZetaBias[h])
            omega_mod_raw = sParams[warp, 1, lane] + cutlass.Float32(mOmegaModBias[h])
            r_raw = sParams[warp, 2, lane]
            mix_r_raw = sParams[warp, 3, lane] + cutlass.Float32(mMixRBias[h])

            dt_u = sigmoid(cutlass.Float32(mDtBias[h]))
            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            zeta = cutlass.Float32(self.zeta_max) * sigmoid(zeta_raw)
            omega_tanh = cute_math.tanh(omega_mod_raw)
            omega_scale = cute_math.exp(
                cutlass.Float32(self.omega_mod_scale) * omega_tanh
            )
            omega_natural = cutlass.Float32(self.omega_min) + softplus(
                cutlass.Float32(mOmegaNaturalBias[h])
            )
            omega_drive = omega_natural * omega_scale
            gamma = zeta * omega_drive
            under = cutlass.max(
                cutlass.Float32(self.eps),
                cutlass.Float32(1.0) - zeta * zeta,
            )
            omega = cutlass.Float32(mOmegaSign[h]) * omega_drive * cute_math.sqrt(under)
            r_direct_u = sigmoid(r_raw)
            mix_r = sigmoid(mix_r_raw)

            exp_term = cute_math.exp(-(gamma * dt))
            r_struct = (
                cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * exp_term
            )
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

            mMOut[b, h, t, 0] = rho_re
            mMOut[b, h, t, 1] = rho_im
            mKOut[b, h, t, 0, 0] = k_prev_re
            mKOut[b, h, t, 0, 1] = k_prev_im
            mKOut[b, h, t, 1, 0] = k_curr_re
            mKOut[b, h, t, 1, 1] = k_curr_im

            if cutlass.const_expr(self.store_coeff_aux):
                mCoeffAux[b, h, COEFF_AUX_ZETA, t] = zeta
                mCoeffAux[b, h, COEFF_AUX_OMEGA_TANH, t] = omega_tanh
                mCoeffAux[b, h, COEFF_AUX_OMEGA_DRIVE, t] = omega_drive
                mCoeffAux[b, h, COEFF_AUX_OMEGA, t] = omega
                mCoeffAux[b, h, COEFF_AUX_R_DIRECT_U, t] = r_direct_u
                mCoeffAux[b, h, COEFF_AUX_MIX_R, t] = mix_r
                mCoeffAux[b, h, COEFF_AUX_DT, t] = dt
                mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t] = exp_term
                mCoeffAux[b, h, COEFF_AUX_DELTA_R, t] = r_struct - r_direct
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
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        zeta_bias: cute.Tensor,
        omega_mod_bias: cute.Tensor,
        omega_natural_bias: cute.Tensor,
        mix_r_bias: cute.Tensor,
        omega_sign: cute.Tensor,
        u: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: cute.Tensor,
        k: cute.Tensor,
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
                (batch, t_size, self.h_size, SCANPREP_PARAM_DIM),
                stride=(
                    t_size * self.h_size * SCANPREP_PARAM_DIM,
                    self.h_size * SCANPREP_PARAM_DIM,
                    SCANPREP_PARAM_DIM,
                    1,
                ),
            ),
        )
        coeff_smem_bytes = self.coeff_smem_bytes

        self.pack_kernel(
            mValue,
            bc,
            u,
            b,
            c,
            total_rows,
            rows_per_batch,
        ).launch(grid=(pack_grid_size, 1, 1), block=(self.pack_block_size, 1, 1))
        self.coeff_kernel(
            mParams,
            dt_bias,
            zeta_bias,
            omega_mod_bias,
            omega_natural_bias,
            mix_r_bias,
            omega_sign,
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
