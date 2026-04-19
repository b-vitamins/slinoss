"""CuTe backward kernels for the ``scanprep`` backend."""

import cutlass
import cutlass.cute as cute

from ..common import (
    COEFF_AUX_DT,
    COEFF_AUX_EXP_TERM,
    COEFF_AUX_ALPHA,
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
    complex_div,
    complex_mul_conj,
    real_mul_conj,
    safe_cast_to_dtype,
)
from .common import (
    _cosize,
    _launchable,
    _llvm_ptr,
    _make_layout,
    _size,
    _struct,
)


class ScanPrepBwdFused:
    """Backward launcher with explicit row kernels and coefficient gradients."""

    def __init__(
        self,
        *,
        h_size: int,
        g_size: int,
        p_size: int,
        n_size: int,
        param_dim: int,
        dt_min: float,
        dt_max: float,
        theta_init_min: float,
        theta_init_max: float,
        alpha_min: float,
        alpha_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        value_warps_per_block: int = 8,
        pack_warps_per_block: int = 12,
        coeff_block_size: int = 512,
    ) -> None:
        self.h_size = int(h_size)
        self.g_size = int(g_size)
        self.p_size = int(p_size)
        self.n_size = int(n_size)
        self.param_dim = int(param_dim)
        self._validate_group_shape()

        self.value_warps_per_block = int(value_warps_per_block)
        self._validate_value_warps_per_block()
        self.value_rows_per_round = self.value_warps_per_block
        self.value_block_size = self.value_warps_per_block * 32
        self.value_bt_tile = 256
        self.value_rounds = (
            self.value_bt_tile + self.value_rows_per_round - 1
        ) // self.value_rows_per_round

        self.pack_role_warps = 2
        self.pack_warps_per_block = int(pack_warps_per_block)
        self._validate_pack_warps_per_block()
        self.pack_rows_per_round = self.pack_warps_per_block // self.pack_role_warps
        self.pack_block_size = self.pack_warps_per_block * 32
        self.pack_bt_tile = 256
        self.pack_rounds = (
            self.pack_bt_tile + self.pack_rows_per_round - 1
        ) // self.pack_rows_per_round

        self.coeff_block_size = int(coeff_block_size)
        self._validate_coeff_block_size()
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        self.coeff_flat_tile = self.coeff_head_tile * self.param_dim
        self.coeff_flat_pad = self.coeff_flat_tile + 1
        self.coeff_smem_bytes = self.coeff_t_tile * self.coeff_flat_pad * 4

        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.theta_init_min = float(theta_init_min)
        self.theta_span = float(max(theta_init_max - theta_init_min, 1.0e-6))
        self.alpha_min = float(alpha_min)
        self.alpha_span = float(alpha_max - alpha_min)
        self.theta_mod_scale = 0.25
        self.r_scale = float(r_max - r_min)
        self.eps = float(eps)
        self.z_thresh_sq = self._resolve_z_thresh_sq()

    def _validate_value_warps_per_block(self) -> None:
        if self.value_warps_per_block <= 0:
            raise ValueError("value_warps_per_block must be positive.")

    def _validate_group_shape(self) -> None:
        if self.g_size <= 0:
            raise ValueError("g_size must be positive.")
        if self.h_size % self.g_size != 0:
            raise ValueError(
                f"h_size must be divisible by g_size. Got {self.h_size}, {self.g_size}."
            )

    def _validate_pack_warps_per_block(self) -> None:
        if (
            self.pack_warps_per_block <= 0
            or self.pack_warps_per_block % self.pack_role_warps != 0
        ):
            raise ValueError(
                "pack_warps_per_block must be a positive multiple of pack_role_warps."
            )

    def _validate_coeff_block_size(self) -> None:
        if self.coeff_block_size <= 0 or self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a positive multiple of 32.")
        if self.coeff_block_size < 32:
            raise ValueError("coeff_block_size must cover at least one warp.")

    def _resolve_z_thresh_sq(self) -> float:
        z_thresh = float(max(1.0e-4, (max(float(self.eps), 1.0e-12)) ** 0.5))
        return float(z_thresh * z_thresh)

    def _value_grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.value_bt_tile - 1) // self.value_bt_tile,
            self.h_size,
            1,
        )

    def _pack_grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.pack_bt_tile - 1) // self.pack_bt_tile,
            self.g_size,
            1,
        )

    def _coeff_grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile,
            (self.h_size + self.coeff_head_tile - 1) // self.coeff_head_tile,
            1,
        )

    def _make_value_grad_view(self, value_grad: cute.Tensor, *, batch, time_steps):
        return cute.make_tensor(
            value_grad.iterator,
            _make_layout(
                (batch, time_steps, self.h_size * self.p_size),
                stride=(
                    time_steps * self.h_size * self.p_size,
                    self.h_size * self.p_size,
                    1,
                ),
            ),
        )

    def _make_param_grad_view(self, dparams: cute.Tensor, *, batch, time_steps):
        return cute.make_tensor(
            dparams.iterator,
            _make_layout(
                (batch, time_steps, self.h_size * self.param_dim),
                stride=(
                    time_steps * self.h_size * self.param_dim,
                    self.h_size * self.param_dim,
                    1,
                ),
            ),
        )

    def _make_bias_grad_view(self, bias_grad: cute.Tensor):
        return cute.make_tensor(
            bias_grad.iterator,
            _make_layout((self.h_size, 4), stride=(4, 1)),
        )

    def _make_coeff_shared_storage(self):
        coeff_layout = _make_layout(
            (self.coeff_t_tile, self.coeff_flat_pad),
            stride=(self.coeff_flat_pad, 1),
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sDParams": _struct().Align[
                _struct().MemRange[cutlass.Float32, _cosize(coeff_layout)],
                16,
            ]
        }

        return _struct()(SharedStorage)

    @cute.kernel
    def _unpack_value_grads(
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
            for round_iter in cutlass.range_constexpr(self.value_rounds):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                bt = (
                    block_bt * self.value_bt_tile
                    + round_iter * self.value_rows_per_round
                    + warp
                )
                if bt < total_bt_:
                    b = bt // t_size_
                    t = bt - b * t_size_
                    for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                        p = lane + p_iter * 32
                        if p < self.p_size:
                            mValueGrad[b, t, p_base + p] = mDU[b, h, t, p]

    @cute.kernel
    def _pack_bc_grads(
        self,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mBCGrad: cute.Tensor,
        total_bt_,
        t_size_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_bt, g, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        role = warp // self.pack_rows_per_round
        row_local = warp - role * self.pack_rows_per_round

        if g < self.g_size and role < self.pack_role_warps:
            num_n_iters = (self.n_size + 31) // 32
            for round_iter in cutlass.range_constexpr(self.pack_rounds):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                bt = (
                    block_bt * self.pack_bt_tile
                    + round_iter * self.pack_rows_per_round
                    + row_local
                )
                if bt < total_bt_:
                    b = bt // t_size_
                    t = bt - b * t_size_
                    if role == 0:
                        for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                            n = lane + n_iter * 32
                            if n < self.n_size:
                                mBCGrad[b, t, g, 0, n] = safe_cast_to_dtype(
                                    mDB[b, g, t, 2 * n], mBCGrad.element_type
                                )
                                mBCGrad[b, t, g, 1, n] = safe_cast_to_dtype(
                                    mDB[b, g, t, 2 * n + 1], mBCGrad.element_type
                                )
                    else:
                        for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                            n = lane + n_iter * 32
                            if n < self.n_size:
                                mBCGrad[b, t, g, 2, n] = safe_cast_to_dtype(
                                    mDC[b, g, t, 2 * n], mBCGrad.element_type
                                )
                                mBCGrad[b, t, g, 3, n] = safe_cast_to_dtype(
                                    mDC[b, g, t, 2 * n + 1], mBCGrad.element_type
                                )

    @cute.kernel
    def _accumulate_coeff_grads(
        self,
        mCoeffAux: cute.Tensor,
        mDM: cute.Tensor,
        mDK: cute.Tensor,
        mDtBias: cute.Tensor,
        mThetaBias: cute.Tensor,
        mThetaSign: cute.Tensor,
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
            _make_layout(
                (self.coeff_t_tile, self.coeff_flat_pad),
                stride=(self.coeff_flat_pad, 1),
            ),
            16,
        )

        if bt < total_bt_ and h < self.h_size:
            b = bt // t_size_
            t = bt - b * t_size_

            alpha = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_ALPHA, t])
            theta_tanh = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_THETA_TANH, t])
            theta_drive = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_THETA_DRIVE, t])
            dt = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DT, t])
            exp_term = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t])
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

            g_exp_term = g_r * cutlass.Float32(self.r_scale)
            g_alpha = g_exp_term * (-exp_term)

            sign = cutlass.Float32(mThetaSign[h])
            g_theta_drive = g_theta * sign

            theta_u = (
                theta_drive - cutlass.Float32(self.theta_init_min)
            ) / cutlass.Float32(self.theta_span)
            g_phase_logit = (
                g_theta_drive
                * cutlass.Float32(self.theta_span)
                * theta_u
                * (cutlass.Float32(1.0) - theta_u)
            )
            g_theta_mod_raw = (
                g_phase_logit
                * cutlass.Float32(self.theta_mod_scale)
                * (cutlass.Float32(1.0) - theta_tanh * theta_tanh)
            )
            alpha_sigmoid = (alpha - cutlass.Float32(self.alpha_min)) / cutlass.Float32(
                self.alpha_span
            )
            g_alpha_raw = (
                g_alpha
                * cutlass.Float32(self.alpha_span)
                * alpha_sigmoid
                * (cutlass.Float32(1.0) - alpha_sigmoid)
            )

            dt_u = (dt - cutlass.Float32(self.dt_min)) / cutlass.Float32(self.dt_scale)
            g_dt_raw = (
                g_dt
                * cutlass.Float32(self.dt_scale)
                * dt_u
                * (cutlass.Float32(1.0) - dt_u)
            )
            g_theta_bias = g_phase_logit

            flat_base = warp * self.param_dim
            sDParams[lane, flat_base + 0] = g_dt_raw
            sDParams[lane, flat_base + 1] = g_alpha_raw
            sDParams[lane, flat_base + 2] = g_theta_mod_raw

            bias_base = h * 4
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 0), g_dt_raw
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 1), g_alpha_raw
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 2), g_theta_mod_raw
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 3), g_theta_bias
            )

        cute.arch.sync_threads()

        num_store_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_store_flat_iters = (self.coeff_flat_tile + 31) // 32
        for store_t_iter in cutlass.range_constexpr(num_store_t_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
            store_t_local = warp + store_t_iter * self.coeff_head_tile
            store_bt = block_x * self.coeff_t_tile + store_t_local
            if store_t_local < self.coeff_t_tile and store_bt < total_bt_:
                store_b = store_bt // t_size_
                store_t = store_bt - store_b * t_size_
                p_base = h_base * self.param_dim
                for flat_iter in cutlass.range_constexpr(num_store_flat_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
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
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        value_grad: cute.Tensor,
        bc_grad: cute.Tensor,
        dparams: cute.Tensor,
        bias_grad: cute.Tensor,
    ):
        batch = _size(bc_grad, mode=[0])
        t_size = _size(bc_grad, mode=[1])
        total_bt = batch * t_size
        value_grid = self._value_grid_shape(total_bt=total_bt)
        pack_grid = self._pack_grid_shape(total_bt=total_bt)
        coeff_grid = self._coeff_grid_shape(total_bt=total_bt)
        value_grad_view = self._make_value_grad_view(
            value_grad,
            batch=batch,
            time_steps=t_size,
        )
        param_grad_view = self._make_param_grad_view(
            dparams,
            batch=batch,
            time_steps=t_size,
        )
        bias_grad_view = self._make_bias_grad_view(bias_grad)

        _launchable(
            self._unpack_value_grads(
                du,
                value_grad_view,
                total_bt,
                t_size,
            )
        ).launch(
            grid=value_grid,
            block=(self.value_block_size, 1, 1),
        )
        _launchable(
            self._pack_bc_grads(
                db,
                dc,
                bc_grad,
                total_bt,
                t_size,
            )
        ).launch(
            grid=pack_grid,
            block=(self.pack_block_size, 1, 1),
        )
        _launchable(
            self._accumulate_coeff_grads(
                coeff_aux,
                dm,
                dk,
                dt_bias,
                theta_bias,
                theta_sign,
                param_grad_view,
                bias_grad_view,
                total_bt,
                t_size,
            )
        ).launch(
            grid=coeff_grid,
            block=(self.coeff_block_size, 1, 1),
            smem=int(self._make_coeff_shared_storage().size_in_bytes()),
        )


__all__ = ["ScanPrepBwdFused"]
