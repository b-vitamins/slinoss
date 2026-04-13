"""CuTe forward kernels for the ``scanprep`` backend."""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    COEFF_AUX_DT,
    COEFF_AUX_EXP_TERM,
    COEFF_AUX_GAMMA,
    COEFF_AUX_KAPPA1_IM,
    COEFF_AUX_KAPPA1_RE,
    COEFF_AUX_KAPPA2_IM,
    COEFF_AUX_KAPPA2_RE,
    COEFF_AUX_LOG_R,
    COEFF_AUX_R,
    COEFF_AUX_RHO_IM,
    COEFF_AUX_RHO_RE,
    COEFF_AUX_THETA,
    COEFF_AUX_THETA_DRIVE,
    COEFF_AUX_THETA_TANH,
    SCANPREP_PARAM_DIM,
    complex_div,
    principal_angle,
    safe_cast_to_dtype,
    sigmoid,
)
from .common import _launchable, _make_layout, _size


class ScanPrepFwdFused:
    """Two-stage forward launcher for row packing and coefficient generation."""

    def __init__(
        self,
        *,
        h_size: int,
        p_size: int,
        n_size: int,
        store_coeff_aux: bool,
        dt_min: float,
        dt_max: float,
        theta_init_min: float,
        theta_init_max: float,
        gamma_min: float,
        gamma_max: float,
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
        self._validate_pack_warps_per_block()
        self.pack_block_size = self.pack_warps_per_block * 32

        self.coeff_block_size = int(coeff_block_size)
        self._validate_coeff_block_size()
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        self.coeff_smem_bytes = (
            self.coeff_head_tile * SCANPREP_PARAM_DIM * (self.coeff_t_tile + 1) * 4
        )

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.theta_init_min = float(theta_init_min)
        self.theta_span = float(max(theta_init_max - theta_init_min, 1.0e-6))
        self.gamma_min = float(gamma_min)
        self.gamma_span = float(gamma_max - gamma_min)
        self.theta_mod_scale = 0.25
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        self.eps = float(eps)
        self.z_thresh_sq = self._resolve_z_thresh_sq()

    def _validate_pack_warps_per_block(self) -> None:
        if self.pack_warps_per_block <= 0:
            raise ValueError("pack_warps_per_block must be positive.")

    def _validate_coeff_block_size(self) -> None:
        if self.coeff_block_size <= 0 or self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a positive multiple of 32.")
        if self.coeff_block_size < 32:
            raise ValueError("coeff_block_size must cover at least one warp.")

    def _resolve_z_thresh_sq(self) -> float:
        z_thresh = float(max(1.0e-4, (max(float(self.eps), 1.0e-12)) ** 0.5))
        return float(z_thresh * z_thresh)

    def _pack_grid_shape(self, *, total_rows) -> tuple[int, int, int]:
        return (
            (total_rows + self.pack_warps_per_block - 1) // self.pack_warps_per_block,
            1,
            1,
        )

    def _coeff_grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile,
            (self.h_size + self.coeff_head_tile - 1) // self.coeff_head_tile,
            1,
        )

    def _make_value_view(self, value: cute.Tensor, *, batch, time_steps):
        return cute.make_tensor(
            value.iterator,
            _make_layout(
                (batch, self.h_size, time_steps, self.p_size),
                stride=(
                    time_steps * self.h_size * self.p_size,
                    self.p_size,
                    self.h_size * self.p_size,
                    1,
                ),
            ),
        )

    def _make_param_view(self, params: cute.Tensor, *, batch, time_steps):
        return cute.make_tensor(
            params.iterator,
            _make_layout(
                (batch, time_steps, self.h_size, SCANPREP_PARAM_DIM),
                stride=(
                    time_steps * self.h_size * SCANPREP_PARAM_DIM,
                    self.h_size * SCANPREP_PARAM_DIM,
                    SCANPREP_PARAM_DIM,
                    1,
                ),
            ),
        )

    @cute.kernel
    def _pack_rows(
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
            for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                p = lane + p_iter * 32
                if p < self.p_size:
                    mU[b, h, t, p] = mValue[b, h, t, p]

            num_n_iters = (self.n_size + 31) // 32
            for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
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
    def _compute_coefficients(
        self,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mGammaBias: cute.Tensor,
        mThetaModBias: cute.Tensor,
        mThetaBias: cute.Tensor,
        mThetaSign: cute.Tensor,
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
            _make_layout(
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
        for load_t_iter in cutlass.range_constexpr(num_load_t_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
            load_t_local = warp + load_t_iter * self.coeff_head_tile
            load_bt = block_x * self.coeff_t_tile + load_t_local
            if load_t_local < self.coeff_t_tile and load_bt < total_bt_:
                load_b = load_bt // t_size_
                load_t = load_bt - load_b * t_size_
                for flat_iter in cutlass.range_constexpr(num_load_flat_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
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

            gamma_raw = sParams[warp, 0, lane] + cutlass.Float32(mGammaBias[h])
            theta_mod_raw = sParams[warp, 1, lane] + cutlass.Float32(mThetaModBias[h])

            dt_u = sigmoid(cutlass.Float32(mDtBias[h]))
            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            gamma = cutlass.Float32(self.gamma_min) + cutlass.Float32(
                self.gamma_span
            ) * sigmoid(gamma_raw)
            theta_tanh = cute_math.tanh(theta_mod_raw)
            theta_u = sigmoid(
                cutlass.Float32(mThetaBias[h])
                + cutlass.Float32(self.theta_mod_scale) * theta_tanh
            )
            theta_drive = (
                cutlass.Float32(self.theta_init_min)
                + cutlass.Float32(self.theta_span) * theta_u
            )
            theta = principal_angle(cutlass.Float32(mThetaSign[h]) * theta_drive)

            exp_term = cute_math.exp(-(gamma * dt))
            r_struct = (
                cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * exp_term
            )
            r = r_struct

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

            if cutlass.const_expr(self.store_coeff_aux):  # pyright: ignore[reportPrivateImportUsage]
                mCoeffAux[b, h, COEFF_AUX_GAMMA, t] = gamma
                mCoeffAux[b, h, COEFF_AUX_THETA_TANH, t] = theta_tanh
                mCoeffAux[b, h, COEFF_AUX_THETA_DRIVE, t] = theta_drive
                mCoeffAux[b, h, COEFF_AUX_DT, t] = dt
                mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t] = exp_term
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
        gamma_bias: cute.Tensor,
        theta_mod_bias: cute.Tensor,
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        u: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: cute.Tensor,
        k: cute.Tensor,
        coeff_aux: cute.Tensor,
    ):
        batch = _size(value, mode=[0])
        t_size = _size(value, mode=[1])
        total_rows = batch * t_size * self.h_size
        rows_per_batch = t_size * self.h_size
        total_bt = batch * t_size
        pack_grid = self._pack_grid_shape(total_rows=total_rows)
        coeff_grid = self._coeff_grid_shape(total_bt=total_bt)
        value_view = self._make_value_view(value, batch=batch, time_steps=t_size)
        param_view = self._make_param_view(params, batch=batch, time_steps=t_size)

        _launchable(
            self._pack_rows(
                value_view,
                bc,
                u,
                b,
                c,
                total_rows,
                rows_per_batch,
            )
        ).launch(grid=pack_grid, block=(self.pack_block_size, 1, 1))
        _launchable(
            self._compute_coefficients(
                param_view,
                dt_bias,
                gamma_bias,
                theta_mod_bias,
                theta_bias,
                theta_sign,
                m,
                k,
                coeff_aux,
                total_bt,
                t_size,
            )
        ).launch(
            grid=coeff_grid,
            block=(self.coeff_block_size, 1, 1),
            smem=self.coeff_smem_bytes,
        )


__all__ = ["ScanPrepFwdFused"]
