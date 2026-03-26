"""Training autograd wrapper for the CuTe scanprep operator."""

from __future__ import annotations

from typing import cast

import torch

from slinoss.ops.scanprep.cute.bwd import scanprep_bwd
from slinoss.ops.scanprep.cute.fwd import scanprep_fwd_cute_with_aux


class _ScanPrepCuTeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
        n_heads: int,
        d_state: int,
        d_head: int,
        normalize_bc: bool,
        dt_min: float,
        dt_max: float,
        r_min: float,
        r_max: float,
        theta_bound: float,
        k_max: float,
        eps: float,
        dt_bias: torch.Tensor,
        gamma_bias: torch.Tensor,
        omega_bias: torch.Tensor,
        mix_r_bias: torch.Tensor,
        mix_theta_bias: torch.Tensor,
        mix_k_prev_bias: torch.Tensor,
        mix_k_curr_bias: torch.Tensor,
        b_scale: torch.Tensor,
        c_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.n_heads = int(n_heads)
        ctx.d_state = int(d_state)
        ctx.d_head = int(d_head)
        ctx.normalize_bc = bool(normalize_bc)
        ctx.value_dtype = value.dtype
        ctx.params_dtype = params.dtype
        ctx.dt_min = float(dt_min)
        ctx.dt_max = float(dt_max)
        ctx.r_min = float(r_min)
        ctx.r_max = float(r_max)
        ctx.theta_bound = float(theta_bound)
        ctx.k_max = float(k_max)
        ctx.eps = float(eps)

        value_d = value.detach()
        params_d = params.detach()
        bc_d = bc.detach()
        dt_bias_d = dt_bias.detach()
        gamma_bias_d = gamma_bias.detach()
        omega_bias_d = omega_bias.detach()
        mix_r_bias_d = mix_r_bias.detach()
        mix_theta_bias_d = mix_theta_bias.detach()
        mix_k_prev_bias_d = mix_k_prev_bias.detach()
        mix_k_curr_bias_d = mix_k_curr_bias.detach()
        b_scale_d = b_scale.detach()
        c_scale_d = c_scale.detach()

        U, M, K, B, C, rms_inv, coeff_aux = scanprep_fwd_cute_with_aux(
            value_d,
            n_heads=n_heads,
            params=params_d,
            bc=bc_d,
            d_state=d_state,
            d_head=d_head,
            normalize_bc=normalize_bc,
            dt_min=dt_min,
            dt_max=dt_max,
            r_min=r_min,
            r_max=r_max,
            theta_bound=theta_bound,
            k_max=k_max,
            eps=eps,
            dt_bias=dt_bias_d,
            gamma_bias=gamma_bias_d,
            omega_bias=omega_bias_d,
            mix_r_bias=mix_r_bias_d,
            mix_theta_bias=mix_theta_bias_d,
            mix_k_prev_bias=mix_k_prev_bias_d,
            mix_k_curr_bias=mix_k_curr_bias_d,
            b_scale=b_scale_d if normalize_bc else None,
            c_scale=c_scale_d if normalize_bc else None,
        )

        ctx.save_for_backward(
            bc_d,
            rms_inv,
            coeff_aux,
            b_scale_d,
            c_scale_d,
        )
        return U, M, K, B, C

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        dU: torch.Tensor | None,
        dM: torch.Tensor | None,
        dK: torch.Tensor | None,
        dB: torch.Tensor | None,
        dC: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        (
            bc,
            rms_inv,
            coeff_aux,
            b_scale,
            c_scale,
        ) = ctx.saved_tensors

        (
            dvalue,
            dparams,
            dbc,
            d_dt_bias,
            d_gamma_bias,
            d_omega_bias,
            d_mix_r_bias,
            d_mix_theta_bias,
            d_mix_k_prev_bias,
            d_mix_k_curr_bias,
            d_b_scale,
            d_c_scale,
        ) = scanprep_bwd(
            bc=bc,
            coeff_aux=coeff_aux,
            rms_inv=rms_inv,
            dU=dU,
            dM=dM,
            dK=dK,
            dB=dB,
            dC=dC,
            n_heads=ctx.n_heads,
            d_head=ctx.d_head,
            d_state=ctx.d_state,
            normalize_bc=ctx.normalize_bc,
            value_dtype=ctx.value_dtype,
            params_dtype=ctx.params_dtype,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            r_min=ctx.r_min,
            r_max=ctx.r_max,
            theta_bound=ctx.theta_bound,
            k_max=ctx.k_max,
            eps=ctx.eps,
            b_scale=b_scale if ctx.normalize_bc else None,
            c_scale=c_scale if ctx.normalize_bc else None,
        )
        return (
            dvalue,
            dparams,
            dbc,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            d_dt_bias,
            d_gamma_bias,
            d_omega_bias,
            d_mix_r_bias,
            d_mix_theta_bias,
            d_mix_k_prev_bias,
            d_mix_k_curr_bias,
            d_b_scale,
            d_c_scale,
        )


def scanprep_cute_training_autograd(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    d_state: int,
    d_head: int,
    normalize_bc: bool,
    dt_min: float,
    dt_max: float,
    r_min: float,
    r_max: float,
    theta_bound: float,
    k_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    omega_bias: torch.Tensor,
    mix_r_bias: torch.Tensor,
    mix_theta_bias: torch.Tensor,
    mix_k_prev_bias: torch.Tensor,
    mix_k_curr_bias: torch.Tensor,
    b_scale: torch.Tensor | None,
    c_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b_scale_in = (
        b_scale
        if b_scale is not None
        else torch.empty((0,), device=value.device, dtype=value.dtype)
    )
    c_scale_in = (
        c_scale
        if c_scale is not None
        else torch.empty((0,), device=value.device, dtype=value.dtype)
    )
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _ScanPrepCuTeFn.apply(
            value,
            params,
            bc,
            int(n_heads),
            int(d_state),
            int(d_head),
            bool(normalize_bc),
            float(dt_min),
            float(dt_max),
            float(r_min),
            float(r_max),
            float(theta_bound),
            float(k_max),
            float(eps),
            dt_bias,
            gamma_bias,
            omega_bias,
            mix_r_bias,
            mix_theta_bias,
            mix_k_prev_bias,
            mix_k_curr_bias,
            b_scale_in,
            c_scale_in,
        ),
    )


__all__ = ["scanprep_cute_training_autograd"]
