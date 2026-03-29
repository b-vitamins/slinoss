# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportOptionalSubscript=false, reportPossiblyUnboundVariable=false, reportGeneralTypeIssues=false
"""Split backward kernels for the CuTe scanprep backend."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from ..common import (
    COEFF_AUX_DELTA_R,
    COEFF_AUX_DELTA_THETA,
    COEFF_AUX_DT,
    COEFF_AUX_DT_U,
    COEFF_AUX_EXP_TERM,
    COEFF_AUX_FIELDS,
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
    complex_mul_conj,
    make_row_major_stride,
    real_mul_conj,
)


class ScanPrepBwdFused:
    """Backward launcher with row kernels and explicit reduction kernels."""

    def __init__(
        self,
        *,
        spec: tuple[int, int, int, int, int, int],
        du_stride: tuple[int, int, int, int] | None = None,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
        value_warps_per_block: int = 8,
        pack_warps_per_block: int = 12,
        coeff_block_size: int = 512,
        scale_block_size: int = 256,
        bias_block_size: int = 224,
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
        self.bc_shape = (self.batch, self.t_size, self.h_size, 4, self.n_size)
        self.bc_stride = make_row_major_stride(self.bc_shape)
        self.grad_shape = (self.batch, self.h_size, self.t_size, 2 * self.n_size)
        self.grad_stride = make_row_major_stride(self.grad_shape)
        self.value_shape = (self.batch, self.t_size, self.h_size * self.p_size)
        self.value_stride = make_row_major_stride(self.value_shape)
        self.dparams_shape = (self.batch, self.t_size, self.h_size * self.param_dim)
        self.dparams_stride = make_row_major_stride(self.dparams_shape)
        self.scale_shape = (self.h_size, 2, self.n_size)
        self.scale_stride = make_row_major_stride(self.scale_shape)
        self.scale_grad_shape = (self.h_size, 4, self.n_size)
        self.scale_grad_stride = make_row_major_stride(self.scale_grad_shape)
        self.bias_grad_shape = (self.h_size, 7)
        self.bias_grad_stride = make_row_major_stride(self.bias_grad_shape)
        self.m_shape = (self.batch, self.h_size, self.t_size, 2)
        self.m_stride = make_row_major_stride(self.m_shape)
        self.k_shape = (self.batch, self.h_size, self.t_size, 2, 2)
        self.k_stride = make_row_major_stride(self.k_shape)
        self.rms_inv_shape = (self.batch, self.h_size, self.t_size, 4)
        self.rms_inv_stride = make_row_major_stride(self.rms_inv_shape)
        self.coeff_aux_shape = (self.batch, self.h_size, COEFF_AUX_FIELDS, self.t_size)
        self.coeff_aux_stride = make_row_major_stride(self.coeff_aux_shape)

        self.total_rows = self.batch * self.t_size * self.h_size
        self.rows_per_batch = self.t_size * self.h_size
        self.total_bt = self.batch * self.t_size

        self.value_warps_per_block = int(value_warps_per_block)
        if self.value_warps_per_block <= 0:
            raise ValueError("value_warps_per_block must be positive.")
        self.value_rows_per_round = self.value_warps_per_block
        self.value_block_size = self.value_warps_per_block * 32
        self.value_bt_tile = 256
        self.value_rounds = (
            self.value_bt_tile + self.value_rows_per_round - 1
        ) // self.value_rows_per_round
        self.value_grid_x = (
            self.total_bt + self.value_bt_tile - 1
        ) // self.value_bt_tile

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
        self.pack_grid_x = (self.total_bt + self.pack_bt_tile - 1) // self.pack_bt_tile
        self.pack_scale_smem_stride = self.n_size + 1
        self.pack_scale_smem_bytes = (
            4 * self.pack_rows_per_round * self.pack_scale_smem_stride * 4
        )
        self.pack_scale_smem_bytes = self.pack_scale_smem_bytes + 64

        self.coeff_block_size = int(coeff_block_size)
        if self.coeff_block_size <= 0 or self.coeff_block_size % 32 != 0:
            raise ValueError("coeff_block_size must be a positive multiple of 32.")
        self.coeff_t_tile = 32
        self.coeff_head_tile = self.coeff_block_size // 32
        if self.coeff_head_tile <= 0:
            raise ValueError("coeff_block_size must cover at least one warp.")
        self.coeff_grid_x = (self.total_bt + self.coeff_t_tile - 1) // self.coeff_t_tile
        self.coeff_grid_y = (
            self.h_size + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        self.coeff_flat_tile = self.coeff_head_tile * self.param_dim
        self.coeff_flat_pad = self.coeff_flat_tile + 1
        self.coeff_smem_bytes = self.coeff_t_tile * self.coeff_flat_pad * 4

        self.scale_block_size = int(scale_block_size)
        if self.scale_block_size <= 0 or self.scale_block_size % 32 != 0:
            raise ValueError("scale_block_size must be a positive multiple of 32.")
        self.scale_warps_per_block = self.scale_block_size // 32
        if self.scale_warps_per_block <= 0:
            raise ValueError("scale_block_size must cover at least one warp.")
        self.scale_n_tile = 32
        self.scale_channel_pairs = 2
        self.scale_grid_x = (self.n_size + self.scale_n_tile - 1) // self.scale_n_tile
        self.scale_bt_tile = self.scale_warps_per_block * 32
        self.scale_grid_z = self.pack_grid_x
        self.scale_partial_shape = (
            self.scale_grid_z,
            self.h_size,
            4,
            self.n_size,
        )
        self.scale_partial_stride = make_row_major_stride(self.scale_partial_shape)
        self.scale_smem_stride = 33
        self.scale_reduce_smem_bytes = (
            self.scale_warps_per_block * self.scale_smem_stride * 4
        )

        self.bias_block_size = int(bias_block_size)
        if self.bias_block_size <= 0 or self.bias_block_size % 32 != 0:
            raise ValueError("bias_block_size must be a positive multiple of 32.")
        self.bias_warps_per_block = self.bias_block_size // 32
        if self.bias_warps_per_block < 7:
            raise ValueError("bias_block_size must cover at least seven warps.")
        self.bias_bt_tile = 256
        self.bias_grid_y = (self.total_bt + self.bias_bt_tile - 1) // self.bias_bt_tile
        self.bias_partial_shape = (self.bias_grid_y, self.h_size, 7)
        self.bias_partial_stride = make_row_major_stride(self.bias_partial_shape)

        self.dt_scale = float(dt_max - dt_min)
        self.r_scale = float(r_max - r_min)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        z_thresh = float(max(1.0e-4, (max(float(eps), 1.0e-12)) ** 0.5))
        self.z_thresh_sq = float(z_thresh * z_thresh)

    def _pack_scale_shared_storage(self):
        part_layout = cute.make_layout(
            (self.pack_rows_per_round, self.pack_scale_smem_stride),
            stride=(self.pack_scale_smem_stride, 1),
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sPartB0": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(part_layout)],
                16,
            ],
            "sPartB1": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(part_layout)],
                16,
            ],
            "sPartC0": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(part_layout)],
                16,
            ],
            "sPartC1": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(part_layout)],
                16,
            ],
        }

        return cute.struct(SharedStorage)

    def _scale_reduce_shared_storage(self):
        acc_layout = cute.make_layout(
            (self.scale_warps_per_block, self.scale_smem_stride),
            stride=(self.scale_smem_stride, 1),
        )

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "sAcc": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(acc_layout)],
                16,
            ]
        }

        return cute.struct(SharedStorage)

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
                if bt < self.total_bt:
                    b = bt // self.t_size
                    t = bt - b * self.t_size
                    for p_iter in cutlass.range_constexpr(num_p_iters):
                        p = lane + p_iter * 32
                        if p < self.p_size:
                            mValueGrad[b, t, p_base + p] = mDU[b, h, t, p]

    @cute.kernel
    def pack_grads_kernel(
        self,
        mBC: cute.Tensor,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mBScale: cute.Tensor,
        mCScale: cute.Tensor,
        mRmsInv: cute.Tensor,
        mBCGrad: cute.Tensor,
        mScalePartial: cute.Tensor,
        mScaleGrad: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_bt, h, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        role = warp // self.pack_rows_per_round
        row_local = warp - role * self.pack_rows_per_round

        if cutlass.const_expr(self.normalize_bc):
            smem = cutlass.utils.SmemAllocator()
            sPartB0 = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (self.pack_rows_per_round, self.pack_scale_smem_stride),
                    stride=(self.pack_scale_smem_stride, 1),
                ),
                16,
            )
            sPartB1 = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (self.pack_rows_per_round, self.pack_scale_smem_stride),
                    stride=(self.pack_scale_smem_stride, 1),
                ),
                16,
            )
            sPartC0 = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (self.pack_rows_per_round, self.pack_scale_smem_stride),
                    stride=(self.pack_scale_smem_stride, 1),
                ),
                16,
            )
            sPartC1 = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (self.pack_rows_per_round, self.pack_scale_smem_stride),
                    stride=(self.pack_scale_smem_stride, 1),
                ),
                16,
            )

        if h < self.h_size and role < self.pack_role_warps:
            num_n_iters = (self.n_size + 31) // 32
            if cutlass.const_expr(self.normalize_bc):
                denom = cutlass.Float32(self.n_size)
                if role == 0:
                    scale_acc0 = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                    scale_acc1 = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                    scale0_cache = cute.make_rmem_tensor(
                        (num_n_iters,), cutlass.Float32
                    )
                    scale1_cache = cute.make_rmem_tensor(
                        (num_n_iters,), cutlass.Float32
                    )
                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        scale_acc0[n_iter] = cutlass.Float32(0.0)
                        scale_acc1[n_iter] = cutlass.Float32(0.0)
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            scale0_cache[n_iter] = cutlass.Float32(mBScale[h, 0, n])
                            scale1_cache[n_iter] = cutlass.Float32(mBScale[h, 1, n])

                    for round_iter in cutlass.range_constexpr(self.pack_rounds):
                        bt = (
                            block_bt * self.pack_bt_tile
                            + round_iter * self.pack_rows_per_round
                            + row_local
                        )
                        if bt < self.total_bt:
                            b = bt // self.t_size
                            t = bt - b * self.t_size
                            inv0 = cutlass.Float32(mRmsInv[b, h, t, 0])
                            inv1 = cutlass.Float32(mRmsInv[b, h, t, 1])
                            inv0_cubed = inv0 * inv0 * inv0 / denom
                            inv1_cubed = inv1 * inv1 * inv1 / denom
                            dot0 = cutlass.Float32(0.0)
                            dot1 = cutlass.Float32(0.0)
                            x0_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            x1_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            dy0_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            dy1_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )

                            for n_iter in cutlass.range_constexpr(num_n_iters):
                                n = lane + n_iter * 32
                                if n < self.n_size:
                                    grad0 = cutlass.Float32(mDB[b, h, t, 2 * n])
                                    grad1 = cutlass.Float32(mDB[b, h, t, 2 * n + 1])
                                    x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                                    x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                                    scale0 = scale0_cache[n_iter]
                                    scale1 = scale1_cache[n_iter]
                                    dy0 = grad0 * scale0
                                    dy1 = grad1 * scale1
                                    x0_cache[n_iter] = x0
                                    x1_cache[n_iter] = x1
                                    dy0_cache[n_iter] = dy0
                                    dy1_cache[n_iter] = dy1
                                    dot0 = dot0 + dy0 * x0
                                    dot1 = dot1 + dy1 * x1
                                    scale_acc0[n_iter] = (
                                        scale_acc0[n_iter] + grad0 * x0 * inv0
                                    )
                                    scale_acc1[n_iter] = (
                                        scale_acc1[n_iter] + grad1 * x1 * inv1
                                    )

                            dot0 = cute.arch.warp_reduction_sum(
                                dot0, threads_in_group=32
                            )
                            dot1 = cute.arch.warp_reduction_sum(
                                dot1, threads_in_group=32
                            )

                            for n_iter in cutlass.range_constexpr(num_n_iters):
                                n = lane + n_iter * 32
                                if n < self.n_size:
                                    x0 = x0_cache[n_iter]
                                    x1 = x1_cache[n_iter]
                                    dy0 = dy0_cache[n_iter]
                                    dy1 = dy1_cache[n_iter]
                                    mBCGrad[b, t, h, 0, n] = (
                                        inv0 * dy0 - x0 * inv0_cubed * dot0
                                    ).to(mBCGrad.element_type)
                                    mBCGrad[b, t, h, 1, n] = (
                                        inv1 * dy1 - x1 * inv1_cubed * dot1
                                    ).to(mBCGrad.element_type)

                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            sPartB0[row_local, n] = scale_acc0[n_iter]
                            sPartB1[row_local, n] = scale_acc1[n_iter]
                else:
                    scale_acc2 = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                    scale_acc3 = cute.make_rmem_tensor((num_n_iters,), cutlass.Float32)
                    scale2_cache = cute.make_rmem_tensor(
                        (num_n_iters,), cutlass.Float32
                    )
                    scale3_cache = cute.make_rmem_tensor(
                        (num_n_iters,), cutlass.Float32
                    )
                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        scale_acc2[n_iter] = cutlass.Float32(0.0)
                        scale_acc3[n_iter] = cutlass.Float32(0.0)
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            scale2_cache[n_iter] = cutlass.Float32(mCScale[h, 0, n])
                            scale3_cache[n_iter] = cutlass.Float32(mCScale[h, 1, n])

                    for round_iter in cutlass.range_constexpr(self.pack_rounds):
                        bt = (
                            block_bt * self.pack_bt_tile
                            + round_iter * self.pack_rows_per_round
                            + row_local
                        )
                        if bt < self.total_bt:
                            b = bt // self.t_size
                            t = bt - b * self.t_size
                            inv2 = cutlass.Float32(mRmsInv[b, h, t, 2])
                            inv3 = cutlass.Float32(mRmsInv[b, h, t, 3])
                            inv2_cubed = inv2 * inv2 * inv2 / denom
                            inv3_cubed = inv3 * inv3 * inv3 / denom
                            dot2 = cutlass.Float32(0.0)
                            dot3 = cutlass.Float32(0.0)
                            x2_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            x3_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            dy2_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )
                            dy3_cache = cute.make_rmem_tensor(
                                (num_n_iters,), cutlass.Float32
                            )

                            for n_iter in cutlass.range_constexpr(num_n_iters):
                                n = lane + n_iter * 32
                                if n < self.n_size:
                                    grad2 = cutlass.Float32(mDC[b, h, t, 2 * n])
                                    grad3 = cutlass.Float32(mDC[b, h, t, 2 * n + 1])
                                    x2 = cutlass.Float32(mBC[b, t, h, 2, n])
                                    x3 = cutlass.Float32(mBC[b, t, h, 3, n])
                                    scale2 = scale2_cache[n_iter]
                                    scale3 = scale3_cache[n_iter]
                                    dy2 = grad2 * scale2
                                    dy3 = grad3 * scale3
                                    x2_cache[n_iter] = x2
                                    x3_cache[n_iter] = x3
                                    dy2_cache[n_iter] = dy2
                                    dy3_cache[n_iter] = dy3
                                    dot2 = dot2 + dy2 * x2
                                    dot3 = dot3 + dy3 * x3
                                    scale_acc2[n_iter] = (
                                        scale_acc2[n_iter] + grad2 * x2 * inv2
                                    )
                                    scale_acc3[n_iter] = (
                                        scale_acc3[n_iter] + grad3 * x3 * inv3
                                    )

                            dot2 = cute.arch.warp_reduction_sum(
                                dot2, threads_in_group=32
                            )
                            dot3 = cute.arch.warp_reduction_sum(
                                dot3, threads_in_group=32
                            )

                            for n_iter in cutlass.range_constexpr(num_n_iters):
                                n = lane + n_iter * 32
                                if n < self.n_size:
                                    x2 = x2_cache[n_iter]
                                    x3 = x3_cache[n_iter]
                                    dy2 = dy2_cache[n_iter]
                                    dy3 = dy3_cache[n_iter]
                                    mBCGrad[b, t, h, 2, n] = (
                                        inv2 * dy2 - x2 * inv2_cubed * dot2
                                    ).to(mBCGrad.element_type)
                                    mBCGrad[b, t, h, 3, n] = (
                                        inv3 * dy3 - x3 * inv3_cubed * dot3
                                    ).to(mBCGrad.element_type)

                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            sPartC0[row_local, n] = scale_acc2[n_iter]
                            sPartC1[row_local, n] = scale_acc3[n_iter]

                cute.arch.sync_threads()

                if role == 0 and row_local == 0:
                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            acc0 = cutlass.Float32(0.0)
                            acc1 = cutlass.Float32(0.0)
                            for r in cutlass.range_constexpr(self.pack_rows_per_round):
                                acc0 = acc0 + sPartB0[r, n]
                                acc1 = acc1 + sPartB1[r, n]
                            scale_base = (
                                h * self.scale_grad_stride[0]
                                + n * self.scale_grad_stride[2]
                            )
                            cute.arch.atomic_add(
                                (mScaleGrad.iterator + scale_base).llvm_ptr,
                                acc0,
                            )
                            cute.arch.atomic_add(
                                (
                                    mScaleGrad.iterator
                                    + scale_base
                                    + self.scale_grad_stride[1]
                                ).llvm_ptr,
                                acc1,
                            )
                elif role == 1 and row_local == 0:
                    for n_iter in cutlass.range_constexpr(num_n_iters):
                        n = lane + n_iter * 32
                        if n < self.n_size:
                            acc2 = cutlass.Float32(0.0)
                            acc3 = cutlass.Float32(0.0)
                            for r in cutlass.range_constexpr(self.pack_rows_per_round):
                                acc2 = acc2 + sPartC0[r, n]
                                acc3 = acc3 + sPartC1[r, n]
                            scale_base = (
                                h * self.scale_grad_stride[0]
                                + cutlass.Int32(2) * self.scale_grad_stride[1]
                                + n * self.scale_grad_stride[2]
                            )
                            cute.arch.atomic_add(
                                (mScaleGrad.iterator + scale_base).llvm_ptr,
                                acc2,
                            )
                            cute.arch.atomic_add(
                                (
                                    mScaleGrad.iterator
                                    + scale_base
                                    + self.scale_grad_stride[1]
                                ).llvm_ptr,
                                acc3,
                            )
            else:
                for round_iter in cutlass.range_constexpr(self.pack_rounds):
                    bt = (
                        block_bt * self.pack_bt_tile
                        + round_iter * self.pack_rows_per_round
                        + row_local
                    )
                    if bt < self.total_bt:
                        b = bt // self.t_size
                        t = bt - b * self.t_size
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
    def scale_partial_kernel(
        self,
        mBC: cute.Tensor,
        mDB: cute.Tensor,
        mDC: cute.Tensor,
        mRmsInv: cute.Tensor,
        mScalePartial: cute.Tensor,
        total_bt_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, block_z = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        n = block_x * self.scale_n_tile + lane
        h = block_y // self.scale_channel_pairs
        c_pair = block_y - h * self.scale_channel_pairs
        smem = cutlass.utils.SmemAllocator()
        sAcc0 = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.scale_warps_per_block, self.scale_smem_stride),
                stride=(self.scale_smem_stride, 1),
            ),
            16,
        )
        sAcc1 = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.scale_warps_per_block, self.scale_smem_stride),
                stride=(self.scale_smem_stride, 1),
            ),
            16,
        )
        if h < self.h_size and n < self.n_size:
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            bt_base = block_z * self.scale_bt_tile + warp * 32
            if c_pair == 0:
                for bt_step in cutlass.range_constexpr(32):
                    bt = bt_base + bt_step
                    if bt < total_bt_:
                        b = bt // self.t_size
                        t = bt - b * self.t_size
                        x0 = cutlass.Float32(mBC[b, t, h, 0, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 1, n])
                        inv0 = cutlass.Float32(mRmsInv[b, h, t, 0])
                        inv1 = cutlass.Float32(mRmsInv[b, h, t, 1])
                        grad0 = cutlass.Float32(mDB[b, h, t, 2 * n])
                        grad1 = cutlass.Float32(mDB[b, h, t, 2 * n + 1])
                        acc0 = acc0 + grad0 * x0 * inv0
                        acc1 = acc1 + grad1 * x1 * inv1
            else:
                for bt_step in cutlass.range_constexpr(32):
                    bt = bt_base + bt_step
                    if bt < total_bt_:
                        b = bt // self.t_size
                        t = bt - b * self.t_size
                        x0 = cutlass.Float32(mBC[b, t, h, 2, n])
                        x1 = cutlass.Float32(mBC[b, t, h, 3, n])
                        inv0 = cutlass.Float32(mRmsInv[b, h, t, 2])
                        inv1 = cutlass.Float32(mRmsInv[b, h, t, 3])
                        grad0 = cutlass.Float32(mDC[b, h, t, 2 * n])
                        grad1 = cutlass.Float32(mDC[b, h, t, 2 * n + 1])
                        acc0 = acc0 + grad0 * x0 * inv0
                        acc1 = acc1 + grad1 * x1 * inv1
            sAcc0[warp, lane] = acc0
            sAcc1[warp, lane] = acc1
        cute.arch.sync_threads()
        if warp == 0 and h < self.h_size and n < self.n_size:
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(self.scale_warps_per_block):
                acc0 = acc0 + sAcc0[w, lane]
                acc1 = acc1 + sAcc1[w, lane]
            c_base = 2 * c_pair
            mScalePartial[block_z, h, c_base, n] = acc0
            mScalePartial[block_z, h, c_base + 1, n] = acc1

    @cute.kernel
    def scale_reduce_kernel(
        self,
        mScalePartial: cute.Tensor,
        mScaleGrad: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        n = block_x * self.scale_n_tile + lane
        h = block_y // 4
        c = block_y - h * 4

        smem = cutlass.utils.SmemAllocator()
        sAcc = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (self.scale_warps_per_block, self.scale_smem_stride),
                stride=(self.scale_smem_stride, 1),
            ),
            16,
        )
        if h < self.h_size and n < self.n_size:
            acc = cutlass.Float32(0.0)
            tile = warp
            while tile < self.scale_grid_z:
                acc = acc + cutlass.Float32(mScalePartial[tile, h, c, n])
                tile += self.scale_warps_per_block
            sAcc[warp, lane] = acc
        cute.arch.sync_threads()
        if warp == 0 and h < self.h_size and n < self.n_size:
            acc = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(self.scale_warps_per_block):
                acc = acc + sAcc[w, lane]
            mScaleGrad[h, c, n] = acc

    @cute.kernel
    def coeff_grad_kernel(
        self,
        mCoeffAux: cute.Tensor,
        mDM: cute.Tensor,
        mDK: cute.Tensor,
        mDParams: cute.Tensor,
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

        if bt < self.total_bt and h < self.h_size:
            b = bt // self.t_size
            t = bt - b * self.t_size

            dt_u = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DT_U, t])
            gamma_sigmoid = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_GAMMA_SIGMOID, t])
            omega = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_OMEGA, t])
            r_direct_u = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_R_DIRECT_U, t])
            theta_direct_tanh = cutlass.Float32(
                mCoeffAux[b, h, COEFF_AUX_THETA_DIRECT_TANH, t]
            )
            mix_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_MIX_R, t])
            mix_theta = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_MIX_THETA, t])
            mix_k_prev = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_MIX_K_PREV, t])
            mix_k_curr = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_MIX_K_CURR, t])
            k_prev_tanh_re = cutlass.Float32(
                mCoeffAux[b, h, COEFF_AUX_K_PREV_TANH_RE, t]
            )
            k_prev_tanh_im = cutlass.Float32(
                mCoeffAux[b, h, COEFF_AUX_K_PREV_TANH_IM, t]
            )
            k_curr_tanh_re = cutlass.Float32(
                mCoeffAux[b, h, COEFF_AUX_K_CURR_TANH_RE, t]
            )
            k_curr_tanh_im = cutlass.Float32(
                mCoeffAux[b, h, COEFF_AUX_K_CURR_TANH_IM, t]
            )
            dt = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DT, t])
            gamma = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_GAMMA, t])
            exp_term = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_EXP_TERM, t])
            delta_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DELTA_R, t])
            delta_theta = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_DELTA_THETA, t])
            r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_R, t])
            theta = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_THETA, t])
            rho_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_RHO_RE, t])
            rho_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_RHO_IM, t])
            log_r = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_LOG_R, t])
            kappa1_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA1_RE, t])
            kappa1_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA1_IM, t])
            kappa2_re = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA2_RE, t])
            kappa2_im = cutlass.Float32(mCoeffAux[b, h, COEFF_AUX_KAPPA2_IM, t])

            k_prev_learned_re = cutlass.Float32(self.k_max) * k_prev_tanh_re
            k_prev_learned_im = cutlass.Float32(self.k_max) * k_prev_tanh_im
            k_curr_learned_re = cutlass.Float32(self.k_max) * k_curr_tanh_re
            k_curr_learned_im = cutlass.Float32(self.k_max) * k_curr_tanh_im

            z_re = log_r
            z_im = theta
            z2_re = z_re * z_re - z_im * z_im
            z2_im = cutlass.Float32(2.0) * z_re * z_im
            z_norm_sq = z_re * z_re + z_im * z_im

            k_prev_struct_re = dt * kappa2_re
            k_prev_struct_im = dt * kappa2_im
            k_curr_struct_re = dt * (kappa1_re - kappa2_re)
            k_curr_struct_im = dt * (kappa1_im - kappa2_im)

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
            g_mix_theta = g_theta * delta_theta

            one_minus_mix_r = cutlass.Float32(1.0) - mix_r
            g_r_direct = g_r * one_minus_mix_r
            g_r_struct = g_r * mix_r
            g_mix_r = g_r * delta_r

            g_r_direct_u = g_r_direct * cutlass.Float32(self.r_scale)
            g_exp_term = g_r_struct * cutlass.Float32(self.r_scale)
            g_x = g_exp_term * exp_term
            g_gamma = g_x * (-dt)
            g_dt = g_dt + g_x * (-gamma)
            g_omega = g_theta_struct * dt
            g_dt = g_dt + g_theta_struct * omega
            g_dt_u = g_dt * cutlass.Float32(self.dt_scale)

            d0 = g_dt_u * dt_u * (cutlass.Float32(1.0) - dt_u)
            d1 = g_gamma * gamma_sigmoid
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

            flat_base = warp * self.param_dim
            sDParams[lane, flat_base + 0] = d0
            sDParams[lane, flat_base + 1] = d1
            sDParams[lane, flat_base + 2] = d2
            sDParams[lane, flat_base + 3] = d3
            sDParams[lane, flat_base + 4] = d4
            sDParams[lane, flat_base + 5] = d5
            sDParams[lane, flat_base + 6] = d6
            sDParams[lane, flat_base + 7] = d7
            sDParams[lane, flat_base + 8] = d8
            sDParams[lane, flat_base + 9] = d9
            sDParams[lane, flat_base + 10] = d10
            sDParams[lane, flat_base + 11] = d11
            sDParams[lane, flat_base + 12] = d12

        cute.arch.sync_threads()

        num_store_t_iters = (
            self.coeff_t_tile + self.coeff_head_tile - 1
        ) // self.coeff_head_tile
        num_store_flat_iters = (self.coeff_flat_tile + 31) // 32
        for store_t_iter in cutlass.range_constexpr(num_store_t_iters):
            store_t_local = warp + store_t_iter * self.coeff_head_tile
            store_bt = block_x * self.coeff_t_tile + store_t_local
            if store_t_local < self.coeff_t_tile and store_bt < self.total_bt:
                store_b = store_bt // self.t_size
                store_t = store_bt - store_b * self.t_size
                p_base = h_base * self.param_dim
                for flat_iter in cutlass.range_constexpr(num_store_flat_iters):
                    flat = lane + flat_iter * 32
                    if flat < self.coeff_flat_tile:
                        store_h = h_base + flat // self.param_dim
                        if store_h < self.h_size:
                            mDParams[store_b, store_t, p_base + flat] = sDParams[
                                store_t_local, flat
                            ].to(mDParams.element_type)

    @cute.kernel
    def bias_partial_kernel(
        self,
        mDParams: cute.Tensor,
        mBiasPartial: cute.Tensor,
        mBiasGrad: cute.Tensor,
        total_bt_,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        h, tile_idx, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        if h < self.h_size and warp < 7:
            p_idx = cutlass.Int32(0)
            if warp == 0:
                p_idx = 0
            elif warp == 1:
                p_idx = 1
            elif warp == 2:
                p_idx = 2
            elif warp == 3:
                p_idx = 5
            elif warp == 4:
                p_idx = 6
            elif warp == 5:
                p_idx = 7
            else:
                p_idx = 8

            acc = cutlass.Float32(0.0)
            p_base = h * self.param_dim
            tile_start = tile_idx * self.bias_bt_tile
            num_bt_iters = (self.bias_bt_tile + 31) // 32
            for bt_iter in cutlass.range_constexpr(num_bt_iters):
                bt = tile_start + bt_iter * 32 + lane
                if bt < total_bt_:
                    b = bt // self.t_size
                    t = bt - b * self.t_size
                    acc = acc + cutlass.Float32(mDParams[b, t, p_base + p_idx])
            acc = cute.arch.warp_reduction_sum(acc, threads_in_group=32)
            if lane == 0:
                bias_offset = (
                    h * self.bias_grad_stride[0] + warp * self.bias_grad_stride[1]
                )
                cute.arch.atomic_add(
                    (mBiasGrad.iterator + bias_offset).llvm_ptr,
                    acc,
                )

    @cute.kernel
    def bias_reduce_kernel(
        self,
        mBiasPartial: cute.Tensor,
        mBiasGrad: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        h, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx - warp * 32
        if h < self.h_size and warp < 7:
            acc = cutlass.Float32(0.0)
            tile = lane
            while tile < self.bias_grid_y:
                acc = acc + cutlass.Float32(mBiasPartial[tile, h, warp])
                tile += 32
            acc = cute.arch.warp_reduction_sum(acc, threads_in_group=32)
            if lane == 0:
                mBiasGrad[h, warp] = acc

    @cute.jit
    def __call__(
        self,
        du_ptr: cute.Pointer,
        bc_ptr: cute.Pointer,
        db_ptr: cute.Pointer,
        dc_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        c_scale_ptr: cute.Pointer,
        rms_inv_ptr: cute.Pointer,
        coeff_aux_ptr: cute.Pointer,
        dm_ptr: cute.Pointer,
        dk_ptr: cute.Pointer,
        value_grad_ptr: cute.Pointer,
        bc_grad_ptr: cute.Pointer,
        dparams_ptr: cute.Pointer,
        scale_partial_ptr: cute.Pointer,
        scale_grad_ptr: cute.Pointer,
        bias_partial_ptr: cute.Pointer,
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
            b_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mCScale = cute.make_tensor(
            c_scale_ptr, cute.make_layout(self.scale_shape, stride=self.scale_stride)
        )
        mRmsInv = cute.make_tensor(
            rms_inv_ptr,
            cute.make_layout(self.rms_inv_shape, stride=self.rms_inv_stride),
        )
        mCoeffAux = cute.make_tensor(
            coeff_aux_ptr,
            cute.make_layout(self.coeff_aux_shape, stride=self.coeff_aux_stride),
        )
        mDM = cute.make_tensor(
            dm_ptr, cute.make_layout(self.m_shape, stride=self.m_stride)
        )
        mDK = cute.make_tensor(
            dk_ptr, cute.make_layout(self.k_shape, stride=self.k_stride)
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
        mScalePartial = cute.make_tensor(
            scale_partial_ptr,
            cute.make_layout(
                self.scale_partial_shape,
                stride=self.scale_partial_stride,
            ),
        )
        mScaleGrad = cute.make_tensor(
            scale_grad_ptr,
            cute.make_layout(self.scale_grad_shape, stride=self.scale_grad_stride),
        )
        mBiasPartial = cute.make_tensor(
            bias_partial_ptr,
            cute.make_layout(
                self.bias_partial_shape,
                stride=self.bias_partial_stride,
            ),
        )
        mBiasGrad = cute.make_tensor(
            bias_grad_ptr,
            cute.make_layout(self.bias_grad_shape, stride=self.bias_grad_stride),
        )
        pack_scale_smem_bytes = (
            int(self._pack_scale_shared_storage().size_in_bytes())
            if self.normalize_bc
            else 0
        )
        coeff_smem_bytes = int(self._coeff_shared_storage().size_in_bytes())

        self.value_grad_kernel(
            mDU,
            mValueGrad,
        ).launch(
            grid=(self.value_grid_x, self.h_size, 1),
            block=(self.value_block_size, 1, 1),
        )
        self.pack_grads_kernel(
            mBC,
            mDB,
            mDC,
            mBScale,
            mCScale,
            mRmsInv,
            mBCGrad,
            mScalePartial,
            mScaleGrad,
        ).launch(
            grid=(self.pack_grid_x, self.h_size, 1),
            block=(self.pack_block_size, 1, 1),
            smem=pack_scale_smem_bytes,
        )
        self.coeff_grad_kernel(
            mCoeffAux,
            mDM,
            mDK,
            mDParams,
        ).launch(
            grid=(self.coeff_grid_x, self.coeff_grid_y, 1),
            block=(self.coeff_block_size, 1, 1),
            smem=coeff_smem_bytes,
        )
        self.bias_partial_kernel(
            mDParams,
            mBiasPartial,
            mBiasGrad,
            self.total_bt,
        ).launch(
            grid=(self.h_size, self.bias_grid_y, 1), block=(self.bias_block_size, 1, 1)
        )


__all__ = ["ScanPrepBwdFused"]
