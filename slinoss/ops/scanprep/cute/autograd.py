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
        dt_min: float,
        dt_max: float,
        theta_init_min: float,
        theta_init_max: float,
        gamma_min: float,
        gamma_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        dt_bias: torch.Tensor,
        gamma_bias: torch.Tensor,
        theta_mod_bias: torch.Tensor,
        theta_bias: torch.Tensor,
        theta_sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.n_heads = int(n_heads)
        ctx.d_state = int(d_state)
        ctx.d_head = int(d_head)
        ctx.value_dtype = value.dtype
        ctx.params_dtype = params.dtype
        ctx.dt_min = float(dt_min)
        ctx.dt_max = float(dt_max)
        ctx.theta_init_min = float(theta_init_min)
        ctx.theta_init_max = float(theta_init_max)
        ctx.gamma_min = float(gamma_min)
        ctx.gamma_max = float(gamma_max)
        ctx.r_min = float(r_min)
        ctx.r_max = float(r_max)
        ctx.eps = float(eps)

        value_d = value.detach()
        params_d = params.detach()
        bc_d = bc.detach()
        dt_bias_d = dt_bias.detach()
        gamma_bias_d = gamma_bias.detach()
        theta_mod_bias_d = theta_mod_bias.detach()
        theta_bias_d = theta_bias.detach()
        theta_sign_d = theta_sign.detach()

        U, M, K, B, C, coeff_aux = scanprep_fwd_cute_with_aux(
            value_d,
            params=params_d,
            bc=bc_d,
            n_heads=n_heads,
            d_state=d_state,
            d_head=d_head,
            dt_min=dt_min,
            dt_max=dt_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            dt_bias=dt_bias_d,
            gamma_bias=gamma_bias_d,
            theta_mod_bias=theta_mod_bias_d,
            theta_bias=theta_bias_d,
            theta_sign=theta_sign_d,
        )

        ctx.save_for_backward(
            bc_d,
            coeff_aux,
            dt_bias_d,
            theta_bias_d,
            theta_sign_d,
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
        bc, coeff_aux, dt_bias, theta_bias, theta_sign = ctx.saved_tensors

        (
            dvalue,
            dparams,
            dbc,
            d_dt_bias,
            d_gamma_bias,
            d_theta_mod_bias,
            d_theta_bias,
        ) = scanprep_bwd(
            bc=bc,
            coeff_aux=coeff_aux,
            dU=dU,
            dM=dM,
            dK=dK,
            dB=dB,
            dC=dC,
            n_heads=ctx.n_heads,
            d_head=ctx.d_head,
            d_state=ctx.d_state,
            value_dtype=ctx.value_dtype,
            params_dtype=ctx.params_dtype,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            theta_init_min=ctx.theta_init_min,
            theta_init_max=ctx.theta_init_max,
            gamma_min=ctx.gamma_min,
            gamma_max=ctx.gamma_max,
            r_min=ctx.r_min,
            r_max=ctx.r_max,
            eps=ctx.eps,
            dt_bias=dt_bias,
            theta_bias=theta_bias,
            theta_sign=theta_sign,
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
            None,
            d_dt_bias,
            d_gamma_bias,
            d_theta_mod_bias,
            d_theta_bias,
            None,
        )


def scanprep_cute_training_autograd(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _ScanPrepCuTeFn.apply(
            value,
            params,
            bc,
            int(n_heads),
            int(d_state),
            int(d_head),
            float(dt_min),
            float(dt_max),
            float(theta_init_min),
            float(theta_init_max),
            float(gamma_min),
            float(gamma_max),
            float(r_min),
            float(r_max),
            float(eps),
            dt_bias,
            gamma_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
        ),
    )


__all__ = ["scanprep_cute_training_autograd"]
