"""Autograd wrapper for the CuTe ``scanprep`` operator."""

from __future__ import annotations

from typing import cast

import torch

from slinoss.ops.scanprep.parameterization import (
    parameterize_scan_bc_rows,
    validate_scan_bc_raw,
    validate_scan_bc_rows,
)

from .kernels import _scanprep_bwd_cute_prevalidated, _scanprep_fwd_cute_prevalidated


class _ScanPrepCuTeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
        n_heads: int,
        bc_groups: int,
        d_state: int,
        d_head: int,
        dt_min: float,
        dt_max: float,
        theta_init_min: float,
        theta_init_max: float,
        alpha_min: float,
        alpha_max: float,
        r_min: float,
        r_max: float,
        eps: float,
        dt_bias: torch.Tensor,
        alpha_bias: torch.Tensor,
        theta_mod_bias: torch.Tensor,
        theta_bias: torch.Tensor,
        theta_sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.n_heads = int(n_heads)
        ctx.bc_groups = int(bc_groups)
        ctx.d_state = int(d_state)
        ctx.d_head = int(d_head)
        ctx.value_dtype = value.dtype
        ctx.params_dtype = params.dtype
        ctx.dt_min = float(dt_min)
        ctx.dt_max = float(dt_max)
        ctx.theta_init_min = float(theta_init_min)
        ctx.theta_init_max = float(theta_init_max)
        ctx.alpha_min = float(alpha_min)
        ctx.alpha_max = float(alpha_max)
        ctx.r_min = float(r_min)
        ctx.r_max = float(r_max)
        ctx.eps = float(eps)

        value_d = value.detach()
        params_d = params.detach()
        bc_d = bc.detach()
        dt_bias_d = dt_bias.detach()
        alpha_bias_d = alpha_bias.detach()
        theta_mod_bias_d = theta_mod_bias.detach()
        theta_bias_d = theta_bias.detach()
        theta_sign_d = theta_sign.detach()

        bc_rows = parameterize_scan_bc_rows(
            bc_d,
            bc_groups=int(bc_groups),
            d_state=int(d_state),
            eps=float(eps),
        )
        validate_scan_bc_rows(
            bc_rows,
            bc_groups=int(bc_groups),
            d_state=int(d_state),
        )
        outputs = _scanprep_fwd_cute_prevalidated(
            value_d,
            params_d,
            bc_rows,
            n_heads=n_heads,
            bc_groups=bc_groups,
            d_state=d_state,
            d_head=d_head,
            dt_min=dt_min,
            dt_max=dt_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            dt_bias=dt_bias_d,
            alpha_bias=alpha_bias_d,
            theta_mod_bias=theta_mod_bias_d,
            theta_bias=theta_bias_d,
            theta_sign=theta_sign_d,
            store_coeff_aux=False,
        )
        ctx.save_for_backward(
            bc_d,
            params_d,
            dt_bias_d,
            alpha_bias_d,
            theta_mod_bias_d,
            theta_bias_d,
            theta_sign_d,
        )
        return outputs.U, outputs.M, outputs.K, outputs.B, outputs.C

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
            params,
            dt_bias,
            alpha_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
        ) = ctx.saved_tensors

        with torch.enable_grad():
            bc_req = bc.detach().requires_grad_(True)
            bc_rows = parameterize_scan_bc_rows(
                bc_req,
                bc_groups=ctx.bc_groups,
                d_state=ctx.d_state,
                eps=ctx.eps,
            )
            validate_scan_bc_rows(
                bc_rows,
                bc_groups=ctx.bc_groups,
                d_state=ctx.d_state,
            )

        bc_rows_detached = bc_rows.detach()
        batch, time_steps = map(int, bc_rows_detached.shape[:2])
        value_dummy = torch.empty(
            (batch, time_steps, ctx.n_heads * ctx.d_head),
            device=bc_rows_detached.device,
            dtype=ctx.value_dtype,
        )
        coeff_aux = _scanprep_fwd_cute_prevalidated(
            value_dummy,
            params,
            bc_rows_detached,
            n_heads=ctx.n_heads,
            bc_groups=ctx.bc_groups,
            d_state=ctx.d_state,
            d_head=ctx.d_head,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            theta_init_min=ctx.theta_init_min,
            theta_init_max=ctx.theta_init_max,
            alpha_min=ctx.alpha_min,
            alpha_max=ctx.alpha_max,
            r_min=ctx.r_min,
            r_max=ctx.r_max,
            eps=ctx.eps,
            dt_bias=dt_bias,
            alpha_bias=alpha_bias,
            theta_mod_bias=theta_mod_bias,
            theta_bias=theta_bias,
            theta_sign=theta_sign,
            store_coeff_aux=True,
        ).coeff_aux
        outputs = _scanprep_bwd_cute_prevalidated(
            bc=bc_rows_detached,
            coeff_aux=coeff_aux,
            dU=dU,
            dM=dM,
            dK=dK,
            dB=dB,
            dC=dC,
            n_heads=ctx.n_heads,
            bc_groups=ctx.bc_groups,
            d_head=ctx.d_head,
            d_state=ctx.d_state,
            value_dtype=ctx.value_dtype,
            params_dtype=ctx.params_dtype,
            dt_min=ctx.dt_min,
            dt_max=ctx.dt_max,
            theta_init_min=ctx.theta_init_min,
            theta_init_max=ctx.theta_init_max,
            alpha_min=ctx.alpha_min,
            alpha_max=ctx.alpha_max,
            r_min=ctx.r_min,
            r_max=ctx.r_max,
            eps=ctx.eps,
            dt_bias=dt_bias,
            theta_bias=theta_bias,
            theta_sign=theta_sign,
        )

        with torch.enable_grad():
            (bc_grad,) = torch.autograd.grad(
                outputs=bc_rows,
                inputs=(bc_req,),
                grad_outputs=outputs.bc_grad,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )

        return (
            outputs.value_grad,
            outputs.dparams,
            bc_grad,
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
            None,
            outputs.bias_grad[:, 0].contiguous(),
            outputs.bias_grad[:, 1].contiguous(),
            outputs.bias_grad[:, 2].contiguous(),
            outputs.bias_grad[:, 3].contiguous(),
            None,
        )


def scanprep_cute_training_autograd(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int | None = None,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    theta_init_min: float,
    theta_init_max: float,
    alpha_min: float,
    alpha_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    alpha_bias: torch.Tensor,
    theta_mod_bias: torch.Tensor,
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resolved_bc_groups = n_heads if bc_groups is None else int(bc_groups)
    validate_scan_bc_raw(
        bc,
        bc_groups=resolved_bc_groups,
        d_state=int(d_state),
    )
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _ScanPrepCuTeFn.apply(
            value,
            params,
            bc,
            int(n_heads),
            int(resolved_bc_groups),
            int(d_state),
            int(d_head),
            float(dt_min),
            float(dt_max),
            float(theta_init_min),
            float(theta_init_max),
            float(alpha_min),
            float(alpha_max),
            float(r_min),
            float(r_max),
            float(eps),
            dt_bias,
            alpha_bias,
            theta_mod_bias,
            theta_bias,
            theta_sign,
        ),
    )


__all__ = ["scanprep_cute_training_autograd"]
