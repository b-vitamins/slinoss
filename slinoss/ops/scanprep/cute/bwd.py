# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Thin backward wrapper for the fused CuTe scanprep backend."""

from __future__ import annotations

from typing import cast

import torch
import cutlass.cute as cute

from .common import make_ptr_arg
from .kernels.bwd import ScanPrepBwdFused


_SCANPREP_BWD_CACHE: dict[tuple[object, ...], object] = {}


def scanprep_bwd(
    *,
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    d_state: int,
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    batch, t_size, width = map(int, value.shape)
    p_size = width // int(n_heads)

    du_in = (
        dU
        if dU is not None
        else torch.zeros(
            (batch, n_heads, t_size, p_size),
            device=value.device,
            dtype=value.dtype,
        )
    )
    db_in = (
        (dB if dB.is_contiguous() else dB.contiguous())
        if dB is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2 * d_state),
            device=bc.device,
            dtype=bc.dtype,
        )
    )
    dc_in = (
        (dC if dC.is_contiguous() else dC.contiguous())
        if dC is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2 * d_state),
            device=bc.device,
            dtype=bc.dtype,
        )
    )
    dm_in = (
        (dM if dM.is_contiguous() else dM.contiguous())
        if dM is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2),
            device=params.device,
            dtype=torch.float32,
        )
    )
    dk_in = (
        (dK if dK.is_contiguous() else dK.contiguous())
        if dK is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2, 2),
            device=params.device,
            dtype=torch.float32,
        )
    )

    value_grad = torch.empty_like(value)
    bc_grad = torch.empty_like(bc)
    dparams = torch.empty(
        (batch, t_size, n_heads * 13),
        device=params.device,
        dtype=params.dtype,
    )
    scale_grad = (
        torch.zeros(
            (n_heads, 4, d_state),
            device=bc.device,
            dtype=torch.float32,
        )
        if normalize_bc
        else torch.empty(
            (n_heads, 4, d_state),
            device=bc.device,
            dtype=torch.float32,
        )
    )
    bias_grad = torch.zeros((n_heads, 7), device=params.device, dtype=torch.float32)
    b_scale_in = (
        b_scale
        if normalize_bc
        else torch.empty((n_heads, 2, d_state), device=bc.device, dtype=bc.dtype)
    )
    c_scale_in = (
        c_scale
        if normalize_bc
        else torch.empty((n_heads, 2, d_state), device=bc.device, dtype=bc.dtype)
    )
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    du_stride = tuple(int(s) for s in du_in.stride())
    params_stride = tuple(int(s) for s in params.stride())

    du_ptr, du_align = make_ptr_arg(du_in)
    bc_ptr, bc_align = make_ptr_arg(bc_c)
    db_ptr, db_align = make_ptr_arg(db_in)
    dc_ptr, dc_align = make_ptr_arg(dc_in)
    b_scale_ptr, b_scale_align = make_ptr_arg(b_scale_in)
    c_scale_ptr, c_scale_align = make_ptr_arg(c_scale_in)
    params_ptr, params_align = make_ptr_arg(params)
    dm_ptr, dm_align = make_ptr_arg(dm_in)
    dk_ptr, dk_align = make_ptr_arg(dk_in)
    dt_bias_ptr, dt_bias_align = make_ptr_arg(dt_bias)
    gamma_bias_ptr, gamma_bias_align = make_ptr_arg(gamma_bias)
    omega_bias_ptr, omega_bias_align = make_ptr_arg(omega_bias)
    mix_r_bias_ptr, mix_r_bias_align = make_ptr_arg(mix_r_bias)
    mix_theta_bias_ptr, mix_theta_bias_align = make_ptr_arg(mix_theta_bias)
    mix_k_prev_bias_ptr, mix_k_prev_bias_align = make_ptr_arg(mix_k_prev_bias)
    mix_k_curr_bias_ptr, mix_k_curr_bias_align = make_ptr_arg(mix_k_curr_bias)
    value_grad_ptr, value_grad_align = make_ptr_arg(value_grad)
    bc_grad_ptr, bc_grad_align = make_ptr_arg(bc_grad)
    dparams_ptr, dparams_align = make_ptr_arg(dparams)
    scale_grad_ptr, scale_grad_align = make_ptr_arg(scale_grad)
    bias_grad_ptr, bias_grad_align = make_ptr_arg(bias_grad)

    spec = (batch, t_size, int(n_heads), int(p_size), int(d_state), 13)
    cache_key = (
        spec,
        int(value.device.index or 0),
        bool(normalize_bc),
        value.dtype,
        params.dtype,
        bc.dtype,
        du_in.dtype,
        db_in.dtype,
        dc_in.dtype,
        dm_in.dtype,
        dk_in.dtype,
        b_scale_in.dtype,
        c_scale_in.dtype,
        dt_bias.dtype,
        gamma_bias.dtype,
        omega_bias.dtype,
        mix_r_bias.dtype,
        mix_theta_bias.dtype,
        mix_k_prev_bias.dtype,
        mix_k_curr_bias.dtype,
        value_grad.dtype,
        bc_grad.dtype,
        dparams.dtype,
        scale_grad.dtype,
        bias_grad.dtype,
        du_stride,
        params_stride,
        du_align,
        bc_align,
        db_align,
        dc_align,
        b_scale_align,
        c_scale_align,
        params_align,
        dm_align,
        dk_align,
        dt_bias_align,
        gamma_bias_align,
        omega_bias_align,
        mix_r_bias_align,
        mix_theta_bias_align,
        mix_k_prev_bias_align,
        mix_k_curr_bias_align,
        value_grad_align,
        bc_grad_align,
        dparams_align,
        scale_grad_align,
        bias_grad_align,
        float(dt_min),
        float(dt_max),
        float(r_min),
        float(r_max),
        float(theta_bound),
        float(k_max),
        float(eps),
    )
    compiled = _SCANPREP_BWD_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            ScanPrepBwdFused(
                spec=spec,
                du_stride=du_stride,
                params_in_stride=params_stride,
                normalize_bc=normalize_bc,
                dt_min=dt_min,
                dt_max=dt_max,
                r_min=r_min,
                r_max=r_max,
                theta_bound=theta_bound,
                k_max=k_max,
                eps=eps,
            ),
            du_ptr,
            bc_ptr,
            db_ptr,
            dc_ptr,
            b_scale_ptr,
            c_scale_ptr,
            params_ptr,
            dm_ptr,
            dk_ptr,
            dt_bias_ptr,
            gamma_bias_ptr,
            omega_bias_ptr,
            mix_r_bias_ptr,
            mix_theta_bias_ptr,
            mix_k_prev_bias_ptr,
            mix_k_curr_bias_ptr,
            value_grad_ptr,
            bc_grad_ptr,
            dparams_ptr,
            scale_grad_ptr,
            bias_grad_ptr,
        )
        _SCANPREP_BWD_CACHE[cache_key] = compiled

    compiled(
        du_ptr,
        bc_ptr,
        db_ptr,
        dc_ptr,
        b_scale_ptr,
        c_scale_ptr,
        params_ptr,
        dm_ptr,
        dk_ptr,
        dt_bias_ptr,
        gamma_bias_ptr,
        omega_bias_ptr,
        mix_r_bias_ptr,
        mix_theta_bias_ptr,
        mix_k_prev_bias_ptr,
        mix_k_curr_bias_ptr,
        value_grad_ptr,
        bc_grad_ptr,
        dparams_ptr,
        scale_grad_ptr,
        bias_grad_ptr,
    )

    dvalue = (
        value_grad
        if value_grad.dtype == value.dtype
        else value_grad.to(dtype=value.dtype)
    )
    dbc = bc_grad if bc_grad.dtype == bc.dtype else bc_grad.to(dtype=bc.dtype)

    bias_dtype = dt_bias.dtype
    if (
        gamma_bias.dtype == bias_dtype
        and omega_bias.dtype == bias_dtype
        and mix_r_bias.dtype == bias_dtype
        and mix_theta_bias.dtype == bias_dtype
        and mix_k_prev_bias.dtype == bias_dtype
        and mix_k_curr_bias.dtype == bias_dtype
    ):
        d_dt_bias = bias_grad[:, 0]
        d_gamma_bias = bias_grad[:, 1]
        d_omega_bias = bias_grad[:, 2]
        d_mix_r_bias = bias_grad[:, 3]
        d_mix_theta_bias = bias_grad[:, 4]
        d_mix_k_prev_bias = bias_grad[:, 5]
        d_mix_k_curr_bias = bias_grad[:, 6]
    else:
        d_dt_bias = bias_grad[:, 0].to(dtype=dt_bias.dtype)
        d_gamma_bias = bias_grad[:, 1].to(dtype=gamma_bias.dtype)
        d_omega_bias = bias_grad[:, 2].to(dtype=omega_bias.dtype)
        d_mix_r_bias = bias_grad[:, 3].to(dtype=mix_r_bias.dtype)
        d_mix_theta_bias = bias_grad[:, 4].to(dtype=mix_theta_bias.dtype)
        d_mix_k_prev_bias = bias_grad[:, 5].to(dtype=mix_k_prev_bias.dtype)
        d_mix_k_curr_bias = bias_grad[:, 6].to(dtype=mix_k_curr_bias.dtype)

    if normalize_bc:
        if b_scale.dtype == c_scale.dtype:
            db_scale = scale_grad[:, :2, :]
            dc_scale = scale_grad[:, 2:, :]
        else:
            db_scale = scale_grad[:, :2, :].to(dtype=b_scale.dtype)
            dc_scale = scale_grad[:, 2:, :].to(dtype=c_scale.dtype)
    else:
        db_scale = None
        dc_scale = None
    return (
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
        cast(torch.Tensor | None, db_scale),
        cast(torch.Tensor | None, dc_scale),
    )


__all__ = ["scanprep_bwd"]
