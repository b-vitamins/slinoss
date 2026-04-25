"""CuTe forward kernels for the ``scanprep`` backend."""

import math

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.pipeline as pipeline

from ..common import (
    SCANPREP_PARAM_DIM,
    complex_div,
    principal_angle,
    safe_cast_to_dtype,
    sigmoid,
    softplus,
)
from .common import _launchable, _make_layout, _size

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


class _PackUFused:
    def __init__(self, *, h_size: int, p_size: int, warps_per_block: int = 8) -> None:
        self.h_size = int(h_size)
        self.p_size = int(p_size)
        self.warps_per_block = int(warps_per_block)
        if self.warps_per_block <= 0:
            raise ValueError("warps_per_block must be positive.")
        self.block_size = self.warps_per_block * 32

    def _grid_shape(self, *, total_rows) -> tuple[int, int, int]:
        return ((total_rows + self.warps_per_block - 1) // self.warps_per_block, 1, 1)

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

    @cute.kernel
    def _pack_u(
        self, mValue: cute.Tensor, mU: cute.Tensor, total_rows_, rows_per_batch_
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        row = block_x * self.warps_per_block + warp
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

    @cute.jit
    def __call__(self, value: cute.Tensor, u: cute.Tensor):
        batch = _size(value, mode=[0])
        time_steps = _size(value, mode=[1])
        total_rows = batch * time_steps * self.h_size
        rows_per_batch = time_steps * self.h_size
        value_view = self._make_value_view(value, batch=batch, time_steps=time_steps)
        _launchable(
            self._pack_u(
                value_view,
                u,
                total_rows,
                rows_per_batch,
            )
        ).launch(
            grid=self._grid_shape(total_rows=total_rows), block=(self.block_size, 1, 1)
        )


class _PackRawBCFused:
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
            raise ValueError("g_size, n_size, and warps_per_block must be positive.")
        self.block_size = self.warps_per_block * 32

    def _grid_shape(self, *, total_rows) -> tuple[int, int, int]:
        return ((total_rows + self.warps_per_block - 1) // self.warps_per_block, 1, 1)

    @cute.kernel
    def _pack_bc(
        self,
        mBCRaw: cute.Tensor,
        mBOut: cute.Tensor,
        mCOut: cute.Tensor,
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

            sum_sq_b = cutlass.Float32(0.0)
            sum_sq_c = cutlass.Float32(0.0)
            num_n_iters = (self.n_size + 31) // 32
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
            if floor_b < cutlass.Float32(self.eps):
                floor_b = cutlass.Float32(self.eps)
            if floor_c < cutlass.Float32(self.eps):
                floor_c = cutlass.Float32(self.eps)
            inv_rms_b = cutlass.Float32(1.0) / cute_math.sqrt(floor_b)
            inv_rms_c = cutlass.Float32(1.0) / cute_math.sqrt(floor_c)

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
                    b_norm = b_amp * inv_rms_b
                    c_norm = c_amp * inv_rms_c
                    b_cos = cute_math.cos(b_phase)
                    b_sin = cute_math.sin(b_phase)
                    c_cos = cute_math.cos(c_phase)
                    c_sin = cute_math.sin(c_phase)
                    mBOut[b, g, t, 2 * n] = safe_cast_to_dtype(
                        b_norm * b_cos,
                        mBOut.element_type,
                    )
                    mBOut[b, g, t, 2 * n + 1] = safe_cast_to_dtype(
                        b_norm * b_sin,
                        mBOut.element_type,
                    )
                    mCOut[b, g, t, 2 * n] = safe_cast_to_dtype(
                        c_norm * c_cos,
                        mCOut.element_type,
                    )
                    mCOut[b, g, t, 2 * n + 1] = safe_cast_to_dtype(
                        c_norm * c_sin,
                        mCOut.element_type,
                    )

    @cute.jit
    def __call__(self, bc_raw: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        batch = _size(bc_raw, mode=[0])
        time_steps = _size(bc_raw, mode=[1])
        total_rows = batch * time_steps * self.g_size
        rows_per_batch = time_steps * self.g_size
        _launchable(
            self._pack_bc(
                bc_raw,
                b,
                c,
                total_rows,
                rows_per_batch,
            )
        ).launch(
            grid=self._grid_shape(total_rows=total_rows), block=(self.block_size, 1, 1)
        )


class _CoeffForwardFused:
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
        coeff_block_size: int = 256,
    ) -> None:
        self.h_size = int(h_size)
        self.coeff_block_size = int(coeff_block_size)
        if (
            self.h_size <= 0
            or self.coeff_block_size <= 0
            or self.coeff_block_size % 32 != 0
        ):
            raise ValueError("Invalid coefficient kernel shape.")
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        self.coeff_smem_bytes = (
            self.coeff_head_tile * SCANPREP_PARAM_DIM * (self.coeff_t_tile + 1) * 4
        )
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
    def _compute_coefficients(
        self,
        mParams: cute.Tensor,
        mDtBias: cute.Tensor,
        mAlphaBias: cute.Tensor,
        mThetaModBias: cute.Tensor,
        mThetaBias: cute.Tensor,
        mThetaSign: cute.Tensor,
        mMOut: cute.Tensor,
        mKOut: cute.Tensor,
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

        pipeline.sync()

        bt = block_x * self.coeff_t_tile + lane
        h = h_base + warp

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

    @cute.jit
    def __call__(
        self,
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        alpha_bias: cute.Tensor,
        theta_mod_bias: cute.Tensor,
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        m: cute.Tensor,
        k: cute.Tensor,
    ):
        batch = _size(params, mode=[0])
        time_steps = _size(params, mode=[1])
        total_bt = batch * time_steps
        param_view = self._make_param_view(params, batch=batch, time_steps=time_steps)
        _launchable(
            self._compute_coefficients(
                param_view,
                dt_bias,
                alpha_bias,
                theta_mod_bias,
                theta_bias,
                theta_sign,
                m,
                k,
                total_bt,
                time_steps,
            )
        ).launch(
            grid=self._grid_shape(total_bt=total_bt),
            block=(self.coeff_block_size, 1, 1),
            smem=self.coeff_smem_bytes,
        )


class ScanPrepFwdFused:
    """Host wrapper that launches the scanprep forward phases."""

    def __init__(
        self,
        *,
        h_size: int,
        g_size: int,
        p_size: int,
        n_size: int,
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
        pack_warps_per_block: int = 8,
        coeff_block_size: int = 256,
    ) -> None:
        self._pack_u = _PackUFused(
            h_size=h_size,
            p_size=p_size,
            warps_per_block=pack_warps_per_block,
        )
        self._pack_bc = _PackRawBCFused(
            g_size=g_size,
            n_size=n_size,
            eps=eps,
            warps_per_block=pack_warps_per_block,
        )
        self._coeff = _CoeffForwardFused(
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
        value: cute.Tensor,
        bc: cute.Tensor,
        params: cute.Tensor,
        dt_bias: cute.Tensor,
        alpha_bias: cute.Tensor,
        theta_mod_bias: cute.Tensor,
        theta_bias: cute.Tensor,
        theta_sign: cute.Tensor,
        u: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        m: cute.Tensor,
        k: cute.Tensor,
    ):
        self._pack_u(value, u)
        self._pack_bc(bc, b, c)
        self._coeff(
            params,
            dt_bias,
            alpha_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
            m,
            k,
        )


__all__ = ["ScanPrepFwdFused"]
