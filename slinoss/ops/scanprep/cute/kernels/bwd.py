"""CuTe backward kernels for the live ``scanprep`` backend."""

import math

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math

from ..common import (
    SCANPREP_PARAM_DIM,
    complex_div,
    complex_mul_conj,
    principal_angle,
    real_mul_conj,
    safe_cast_to_dtype,
    sigmoid,
    softplus,
)
from .common import _launchable, _llvm_ptr, _make_layout, _size

_PHASE_LIMIT = float(math.pi)


def _warp_reduce_sum(val):
    total = cutlass.Float32(val)
    total = cutlass.Float32(
        total
        + cute.arch.shuffle_sync_bfly(total, offset=16, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=8, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=4, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=2, mask=-1, mask_and_clamp=31)
    )
    total = cutlass.Float32(
        total + cute.arch.shuffle_sync_bfly(total, offset=1, mask=-1, mask_and_clamp=31)
    )
    return total


class _UnpackValueGradFused:
    def __init__(self, *, h_size: int, p_size: int, warps_per_block: int = 8) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.warps_per_block = int(warps_per_block)
        if self.h_size <= 0 or self.p_size <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid unpack-value shape.")
        self.rows_per_round = self.warps_per_block
        self.block_size = self.warps_per_block * 32
        self.bt_tile = 256
        self.rounds = (self.bt_tile + self.rows_per_round - 1) // self.rows_per_round

    def _grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.bt_tile - 1) // self.bt_tile,
            self.h_size,
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
            for round_iter in cutlass.range_constexpr(self.rounds):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                bt = block_bt * self.bt_tile + round_iter * self.rows_per_round + warp
                if bt < total_bt_:
                    b = bt // t_size_
                    t = bt - b * t_size_
                    for p_iter in cutlass.range_constexpr(num_p_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                        p = lane + p_iter * 32
                        if p < self.p_size:
                            mValueGrad[b, t, p_base + p] = mDU[b, h, t, p]

    @cute.jit
    def __call__(self, du: cute.Tensor, value_grad: cute.Tensor):
        batch = _size(du, mode=[0])
        time_steps = _size(du, mode=[2])
        total_bt = batch * time_steps
        value_grad_view = self._make_value_grad_view(
            value_grad,
            batch=batch,
            time_steps=time_steps,
        )
        _launchable(
            self._unpack_value_grads(
                du,
                value_grad_view,
                total_bt,
                time_steps,
            )
        ).launch(
            grid=self._grid_shape(total_bt=total_bt), block=(self.block_size, 1, 1)
        )


class _RawBCGradFused:
    def __init__(
        self,
        *,
        g_size: int,
        n_size: int,
        eps: float,
        warps_per_block: int = 8,
    ) -> None:
        self.g_size = int(g_size)
        self.n_size = int(n_size)
        self.eps = float(eps)
        self.warps_per_block = int(warps_per_block)
        if self.g_size <= 0 or self.n_size <= 0 or self.warps_per_block <= 0:
            raise ValueError("Invalid raw-BC-grad shape.")
        self.block_size = self.warps_per_block * 32

    def _grid_shape(self, *, total_rows) -> tuple[int, int, int]:
        return ((total_rows + self.warps_per_block - 1) // self.warps_per_block, 1, 1)

    @cute.kernel
    def _raw_bc_grad(
        self,
        mBCRaw: cute.Tensor,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mBCGrad: cute.Tensor,
        total_rows_,
        rows_per_batch_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
        if row < total_rows_:
            b = row // rows_per_batch_
            rem = row - b * rows_per_batch_
            t = rem // self.g_size
            g = rem - t * self.g_size
            num_n_iters = (self.n_size + 31) // 32

            sum_sq_b = cutlass.Float32(0.0)
            sum_sq_c = cutlass.Float32(0.0)
            for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                n = lane + n_iter * 32
                if n < self.n_size:
                    b_amp = softplus(mBCRaw[b, t, g, 0, n])
                    c_amp = softplus(mBCRaw[b, t, g, 2, n])
                    sum_sq_b = sum_sq_b + b_amp * b_amp
                    sum_sq_c = sum_sq_c + c_amp * c_amp

            sum_sq_b = _warp_reduce_sum(sum_sq_b)
            sum_sq_c = _warp_reduce_sum(sum_sq_c)

            inv_n = cutlass.Float32(1.0 / float(self.n_size))
            mean_sq_b = sum_sq_b * inv_n
            mean_sq_c = sum_sq_c * inv_n
            floor_b = mean_sq_b
            floor_c = mean_sq_c
            b_clamped = False
            c_clamped = False
            if floor_b < cutlass.Float32(self.eps):
                floor_b = cutlass.Float32(self.eps)
                b_clamped = True
            if floor_c < cutlass.Float32(self.eps):
                floor_c = cutlass.Float32(self.eps)
                c_clamped = True
            rms_b = cute_math.sqrt(floor_b)
            rms_c = cute_math.sqrt(floor_c)
            inv_rms_b = cutlass.Float32(1.0) / rms_b
            inv_rms_c = cutlass.Float32(1.0) / rms_c

            dot_b = cutlass.Float32(0.0)
            dot_c = cutlass.Float32(0.0)
            for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                n = lane + n_iter * 32
                if n < self.n_size:
                    b_amp = softplus(mBCRaw[b, t, g, 0, n])
                    c_amp = softplus(mBCRaw[b, t, g, 2, n])
                    b_phase = cutlass.Float32(_PHASE_LIMIT) * cute_math.tanh(
                        cutlass.Float32(mBCRaw[b, t, g, 1, n])
                    )
                    c_phase = cutlass.Float32(_PHASE_LIMIT) * cute_math.tanh(
                        cutlass.Float32(mBCRaw[b, t, g, 3, n])
                    )
                    b_cos = cute_math.cos(b_phase)
                    b_sin = cute_math.sin(b_phase)
                    c_cos = cute_math.cos(c_phase)
                    c_sin = cute_math.sin(c_phase)
                    db_re = cutlass.Float32(mDB[b, g, t, 2 * n])
                    db_im = cutlass.Float32(mDB[b, g, t, 2 * n + 1])
                    dc_re = cutlass.Float32(mDC[b, g, t, 2 * n])
                    dc_im = cutlass.Float32(mDC[b, g, t, 2 * n + 1])
                    gy_b = db_re * b_cos + db_im * b_sin
                    gy_c = dc_re * c_cos + dc_im * c_sin
                    dot_b = dot_b + gy_b * b_amp
                    dot_c = dot_c + gy_c * c_amp

            dot_b = _warp_reduce_sum(dot_b)
            dot_c = _warp_reduce_sum(dot_c)
            inv_den_b = inv_n / (rms_b * rms_b * rms_b)
            inv_den_c = inv_n / (rms_c * rms_c * rms_c)

            for n_iter in cutlass.range_constexpr(num_n_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                n = lane + n_iter * 32
                if n < self.n_size:
                    b_amp_logit = cutlass.Float32(mBCRaw[b, t, g, 0, n])
                    b_phase_logit = cutlass.Float32(mBCRaw[b, t, g, 1, n])
                    c_amp_logit = cutlass.Float32(mBCRaw[b, t, g, 2, n])
                    c_phase_logit = cutlass.Float32(mBCRaw[b, t, g, 3, n])
                    b_amp = softplus(b_amp_logit)
                    c_amp = softplus(c_amp_logit)
                    b_tanh = cute_math.tanh(b_phase_logit)
                    c_tanh = cute_math.tanh(c_phase_logit)
                    b_phase = cutlass.Float32(_PHASE_LIMIT) * b_tanh
                    c_phase = cutlass.Float32(_PHASE_LIMIT) * c_tanh
                    b_cos = cute_math.cos(b_phase)
                    b_sin = cute_math.sin(b_phase)
                    c_cos = cute_math.cos(c_phase)
                    c_sin = cute_math.sin(c_phase)

                    db_re = cutlass.Float32(mDB[b, g, t, 2 * n])
                    db_im = cutlass.Float32(mDB[b, g, t, 2 * n + 1])
                    dc_re = cutlass.Float32(mDC[b, g, t, 2 * n])
                    dc_im = cutlass.Float32(mDC[b, g, t, 2 * n + 1])

                    gy_b = db_re * b_cos + db_im * b_sin
                    gy_c = dc_re * c_cos + dc_im * c_sin
                    gphase_b = (b_amp * inv_rms_b) * (-db_re * b_sin + db_im * b_cos)
                    gphase_c = (c_amp * inv_rms_c) * (-dc_re * c_sin + dc_im * c_cos)
                    gamp_b = gy_b * inv_rms_b
                    gamp_c = gy_c * inv_rms_c
                    if not b_clamped:
                        gamp_b = gamp_b - b_amp * dot_b * inv_den_b
                    if not c_clamped:
                        gamp_c = gamp_c - c_amp * dot_c * inv_den_c

                    mBCGrad[b, t, g, 0, n] = safe_cast_to_dtype(
                        gamp_b * sigmoid(b_amp_logit),
                        mBCGrad.element_type,
                    )
                    mBCGrad[b, t, g, 1, n] = safe_cast_to_dtype(
                        gphase_b
                        * cutlass.Float32(_PHASE_LIMIT)
                        * (cutlass.Float32(1.0) - b_tanh * b_tanh),
                        mBCGrad.element_type,
                    )
                    mBCGrad[b, t, g, 2, n] = safe_cast_to_dtype(
                        gamp_c * sigmoid(c_amp_logit),
                        mBCGrad.element_type,
                    )
                    mBCGrad[b, t, g, 3, n] = safe_cast_to_dtype(
                        gphase_c
                        * cutlass.Float32(_PHASE_LIMIT)
                        * (cutlass.Float32(1.0) - c_tanh * c_tanh),
                        mBCGrad.element_type,
                    )

    @cute.jit
    def __call__(
        self,
        bc_raw: cute.Tensor,
        db: cute.Tensor,
        dc: cute.Tensor,
        bc_grad: cute.Tensor,
    ):
        batch = _size(bc_raw, mode=[0])
        time_steps = _size(bc_raw, mode=[1])
        total_rows = batch * time_steps * self.g_size
        rows_per_batch = time_steps * self.g_size
        _launchable(
            self._raw_bc_grad(
                bc_raw,
                db,
                dc,
                bc_grad,
                total_rows,
                rows_per_batch,
            )
        ).launch(
            grid=self._grid_shape(total_rows=total_rows), block=(self.block_size, 1, 1)
        )


class _CoeffGradFused:
    def __init__(
        self,
        *,
        h_size: int,
        dt_min: float,
        dt_max: float,
        theta_init_min: float,
        theta_init_max: float,
        theta_mod_scale: float,
        alpha_min: float,
        alpha_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        coeff_block_size: int = 512,
    ) -> None:
        self.h_size = int(h_size)
        self.coeff_block_size = int(coeff_block_size)
        if (
            self.h_size <= 0
            or self.coeff_block_size <= 0
            or self.coeff_block_size % 32 != 0
        ):
            raise ValueError("Invalid coefficient-grad kernel shape.")
        self.param_dim = SCANPREP_PARAM_DIM
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        self.coeff_flat_tile = self.coeff_head_tile * self.param_dim
        self.coeff_flat_pad = self.coeff_flat_tile + 1
        self.param_smem_bytes = (
            self.coeff_head_tile * self.param_dim * (self.coeff_t_tile + 1) * 4
        )
        self.grad_smem_bytes = self.coeff_t_tile * self.coeff_flat_pad * 4
        self.coeff_smem_bytes = self.param_smem_bytes + self.grad_smem_bytes
        self.dt_min = float(dt_min)
        self.dt_scale = float(dt_max - dt_min)
        self.theta_init_min = float(theta_init_min)
        self.theta_span = float(max(theta_init_max - theta_init_min, 1.0e-6))
        self.theta_mod_scale = float(theta_mod_scale)
        self.alpha_min = float(alpha_min)
        self.alpha_span = float(alpha_max - alpha_min)
        self.r_min = float(r_min)
        self.r_scale = float(r_max - r_min)
        z_thresh = float(max(1.0e-4, math.sqrt(max(float(eps), 1.0e-12))))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    def _grid_shape(self, *, total_bt) -> tuple[int, int, int]:
        return (
            (total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile,
            (self.h_size + self.coeff_head_tile - 1) // self.coeff_head_tile,
            1,
        )

    def _make_param_view(self, params: cute.Tensor, *, batch, time_steps):
        return cute.make_tensor(
            params.iterator,
            _make_layout(
                (batch, time_steps, self.h_size, self.param_dim),
                stride=(
                    time_steps * self.h_size * self.param_dim,
                    self.h_size * self.param_dim,
                    self.param_dim,
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

    @cute.kernel
    def _accumulate_coeff_grads(
        self,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mAlphaBias: cute.Tensor,
        mThetaModBias: cute.Tensor,
        mThetaBias: cute.Tensor,
        mThetaSign: cute.Tensor,
        mDM: cute.Tensor,
        mDK: cute.Tensor,
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
        g_dt_raw = cutlass.Float32(0.0)
        g_alpha_raw = cutlass.Float32(0.0)
        g_theta_mod_raw = cutlass.Float32(0.0)
        g_phase_logit = cutlass.Float32(0.0)

        smem = cutlass.utils.SmemAllocator()
        sParams = smem.allocate_tensor(
            cutlass.Float32,
            _make_layout(
                (self.coeff_head_tile, self.param_dim, self.coeff_t_tile + 1),
                stride=(
                    self.param_dim * (self.coeff_t_tile + 1),
                    self.coeff_t_tile + 1,
                    1,
                ),
            ),
            16,
        )
        sDParams = smem.allocate_tensor(
            cutlass.Float32,
            _make_layout(
                (self.coeff_t_tile, self.coeff_flat_pad),
                stride=(self.coeff_flat_pad, 1),
            ),
            16,
        )

        num_load_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_load_flat_iters = (self.coeff_head_tile * self.param_dim + 31) // 32
        for load_t_iter in cutlass.range_constexpr(num_load_t_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
            load_t_local = warp + load_t_iter * self.coeff_head_tile
            load_bt = block_x * self.coeff_t_tile + load_t_local
            if load_t_local < self.coeff_t_tile and load_bt < total_bt_:
                load_b = load_bt // t_size_
                load_t = load_bt - load_b * t_size_
                for flat_iter in cutlass.range_constexpr(num_load_flat_iters):  # pyright: ignore[reportGeneralTypeIssues, reportPrivateImportUsage]
                    flat = lane + flat_iter * 32
                    if flat < self.coeff_head_tile * self.param_dim:
                        local_h = flat // self.param_dim
                        param_idx = flat - local_h * self.param_dim
                        load_h = h_base + local_h
                        if load_h < self.h_size:
                            sParams[local_h, param_idx, load_t_local] = cutlass.Float32(
                                mParams[load_b, load_t, load_h, param_idx]
                            )

        cute.arch.sync_threads()

        if bt < total_bt_ and h < self.h_size:
            b = bt // t_size_
            t = bt - b * t_size_
            dt_raw = sParams[warp, 0, lane] + cutlass.Float32(mDtBias[h])
            alpha_raw = sParams[warp, 1, lane] + cutlass.Float32(mAlphaBias[h])
            theta_mod_raw = sParams[warp, 2, lane] + cutlass.Float32(mThetaModBias[h])

            dt_u = sigmoid(dt_raw)
            dt = cutlass.Float32(self.dt_min) + cutlass.Float32(self.dt_scale) * dt_u
            alpha = cutlass.Float32(self.alpha_min) + cutlass.Float32(
                self.alpha_span
            ) * sigmoid(alpha_raw)
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

            exp_term = cute_math.exp(-alpha)
            r = cutlass.Float32(self.r_min) + cutlass.Float32(self.r_scale) * exp_term
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
        if bt < total_bt_ and h < self.h_size:
            flat_base = warp * self.param_dim
            sDParams[lane, flat_base + 0] = g_dt_raw
            sDParams[lane, flat_base + 1] = g_alpha_raw
            sDParams[lane, flat_base + 2] = g_theta_mod_raw

        bias_dt_sum = _warp_reduce_sum(g_dt_raw)
        bias_alpha_sum = _warp_reduce_sum(g_alpha_raw)
        bias_theta_mod_sum = _warp_reduce_sum(g_theta_mod_raw)
        bias_theta_sum = _warp_reduce_sum(g_phase_logit)
        if lane == 0 and h < self.h_size:
            bias_base = h * 4
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 0), bias_dt_sum
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 1), bias_alpha_sum
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 2), bias_theta_mod_sum
            )
            cute.arch.atomic_add(
                _llvm_ptr(mBiasGrad.iterator + bias_base + 3), bias_theta_sum
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
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        alpha_bias: cute.Tensor,
        theta_mod_bias: cute.Tensor,
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        dm: cute.Tensor,
        dk: cute.Tensor,
        dparams: cute.Tensor,
        bias_grad: cute.Tensor,
    ):
        batch = _size(params, mode=[0])
        time_steps = _size(params, mode=[1])
        total_bt = batch * time_steps
        param_view = self._make_param_view(params, batch=batch, time_steps=time_steps)
        dparam_view = self._make_param_grad_view(
            dparams, batch=batch, time_steps=time_steps
        )
        bias_grad_view = self._make_bias_grad_view(bias_grad)
        _launchable(
            self._accumulate_coeff_grads(
                param_view,
                dt_bias,
                alpha_bias,
                theta_mod_bias,
                theta_bias,
                theta_sign,
                dm,
                dk,
                dparam_view,
                bias_grad_view,
                total_bt,
                time_steps,
            )
        ).launch(
            grid=self._grid_shape(total_bt=total_bt),
            block=(self.coeff_block_size, 1, 1),
            smem=self.coeff_smem_bytes,
        )


class ScanPrepBwdFused:
    """Host wrapper that launches the live raw-BC backward phases."""

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
        theta_mod_scale: float,
        alpha_min: float,
        alpha_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        value_warps_per_block: int = 8,
        pack_warps_per_block: int = 8,
        coeff_block_size: int = 512,
    ) -> None:
        if int(param_dim) != SCANPREP_PARAM_DIM:
            raise ValueError(
                f"param_dim must be {SCANPREP_PARAM_DIM}. Got {param_dim}."
            )
        self._unpack_u = _UnpackValueGradFused(
            h_size=h_size,
            p_size=p_size,
            warps_per_block=value_warps_per_block,
        )
        self._raw_bc_grad = _RawBCGradFused(
            g_size=g_size,
            n_size=n_size,
            eps=eps,
            warps_per_block=pack_warps_per_block,
        )
        self._coeff_grad = _CoeffGradFused(
            h_size=h_size,
            dt_min=dt_min,
            dt_max=dt_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            theta_mod_scale=theta_mod_scale,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            coeff_block_size=coeff_block_size,
        )

    @cute.jit
    def __call__(
        self,
        du: cute.Tensor,
        bc: cute.Tensor,
        db: cute.Tensor,
        dc: cute.Tensor,
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        alpha_bias: cute.Tensor,
        theta_mod_bias: cute.Tensor,
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        dm: cute.Tensor,
        dk: cute.Tensor,
        value_grad: cute.Tensor,
        bc_grad: cute.Tensor,
        dparams: cute.Tensor,
        bias_grad: cute.Tensor,
    ):
        self._unpack_u(du, value_grad)
        self._raw_bc_grad(bc, db, dc, bc_grad)
        self._coeff_grad(
            params,
            dt_bias,
            alpha_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
            dm,
            dk,
            dparams,
            bias_grad,
        )


__all__ = ["ScanPrepBwdFused"]
