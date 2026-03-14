# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Thin forward wrapper for the fused CuTe scanprep backend."""

from __future__ import annotations

import torch
import cutlass.cute as cute

from .common import make_ptr_arg
from .kernels.fwd import ScanPrepFwdFused


_SCANPREP_FWD_CACHE: dict[tuple[object, ...], object] = {}


def scanprep_fwd_cute(
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
    value_c = value if value.is_contiguous() else value.contiguous()
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    batch, t_size, width = map(int, value_c.shape)
    if width != int(n_heads * d_head):
        raise ValueError(f"value width must be {n_heads * d_head}. Got {width}.")
    if tuple(map(int, params.shape)) != (batch, t_size, int(n_heads * 13)):
        raise ValueError(
            f"params must be {(batch, t_size, int(n_heads * 13))}. Got {tuple(params.shape)}."
        )
    if tuple(map(int, bc_c.shape)) != (batch, t_size, int(n_heads), 4, int(d_state)):
        raise ValueError(
            f"bc must be {(batch, t_size, int(n_heads), 4, int(d_state))}. Got {tuple(bc_c.shape)}."
        )

    U = torch.empty(
        (batch, n_heads, t_size, d_head), device=value.device, dtype=value.dtype
    )
    M = torch.empty(
        (batch, n_heads, t_size, 2), device=value.device, dtype=torch.float32
    )
    K = torch.empty(
        (batch, n_heads, t_size, 2, 2), device=value.device, dtype=torch.float32
    )
    B = torch.empty(
        (batch, n_heads, t_size, 2 * d_state), device=value.device, dtype=bc.dtype
    )
    C = torch.empty_like(B)
    b_scale_c = (
        b_scale
        if b_scale is not None
        else torch.empty((n_heads, 2, d_state), device=value.device, dtype=bc.dtype)
    )
    c_scale_c = (
        c_scale
        if c_scale is not None
        else torch.empty((n_heads, 2, d_state), device=value.device, dtype=bc.dtype)
    )
    params_stride = tuple(int(s) for s in params.stride())

    value_ptr, value_align = make_ptr_arg(value_c)
    bc_ptr, bc_align = make_ptr_arg(bc_c)
    b_scale_ptr, b_scale_align = make_ptr_arg(b_scale_c)
    c_scale_ptr, c_scale_align = make_ptr_arg(c_scale_c)
    params_ptr, params_align = make_ptr_arg(params)
    dt_bias_ptr, dt_bias_align = make_ptr_arg(dt_bias)
    gamma_bias_ptr, gamma_bias_align = make_ptr_arg(gamma_bias)
    omega_bias_ptr, omega_bias_align = make_ptr_arg(omega_bias)
    mix_r_bias_ptr, mix_r_bias_align = make_ptr_arg(mix_r_bias)
    mix_theta_bias_ptr, mix_theta_bias_align = make_ptr_arg(mix_theta_bias)
    mix_k_prev_bias_ptr, mix_k_prev_bias_align = make_ptr_arg(mix_k_prev_bias)
    mix_k_curr_bias_ptr, mix_k_curr_bias_align = make_ptr_arg(mix_k_curr_bias)
    u_ptr, u_align = make_ptr_arg(U)
    m_ptr, m_align = make_ptr_arg(M)
    k_ptr, k_align = make_ptr_arg(K)
    b_ptr, b_align = make_ptr_arg(B)
    c_ptr, c_align = make_ptr_arg(C)

    spec = (batch, t_size, int(n_heads), int(d_head), int(d_state))
    cache_key = (
        spec,
        int(value.device.index or 0),
        bool(normalize_bc),
        value_c.dtype,
        params.dtype,
        bc_c.dtype,
        b_scale_c.dtype,
        c_scale_c.dtype,
        dt_bias.dtype,
        gamma_bias.dtype,
        omega_bias.dtype,
        mix_r_bias.dtype,
        mix_theta_bias.dtype,
        mix_k_prev_bias.dtype,
        mix_k_curr_bias.dtype,
        U.dtype,
        M.dtype,
        K.dtype,
        B.dtype,
        C.dtype,
        value_align,
        bc_align,
        b_scale_align,
        c_scale_align,
        params_stride,
        params_align,
        dt_bias_align,
        gamma_bias_align,
        omega_bias_align,
        mix_r_bias_align,
        mix_theta_bias_align,
        mix_k_prev_bias_align,
        mix_k_curr_bias_align,
        u_align,
        m_align,
        k_align,
        b_align,
        c_align,
        float(dt_min),
        float(dt_max),
        float(r_min),
        float(r_max),
        float(theta_bound),
        float(k_max),
        float(eps),
    )
    compiled = _SCANPREP_FWD_CACHE.get(cache_key)
    if compiled is None:
        compiled = cute.compile(
            ScanPrepFwdFused(
                spec=spec,
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
            value_ptr,
            bc_ptr,
            b_scale_ptr,
            c_scale_ptr,
            params_ptr,
            dt_bias_ptr,
            gamma_bias_ptr,
            omega_bias_ptr,
            mix_r_bias_ptr,
            mix_theta_bias_ptr,
            mix_k_prev_bias_ptr,
            mix_k_curr_bias_ptr,
            u_ptr,
            m_ptr,
            k_ptr,
            b_ptr,
            c_ptr,
        )
        _SCANPREP_FWD_CACHE[cache_key] = compiled
    compiled(
        value_ptr,
        bc_ptr,
        b_scale_ptr,
        c_scale_ptr,
        params_ptr,
        dt_bias_ptr,
        gamma_bias_ptr,
        omega_bias_ptr,
        mix_r_bias_ptr,
        mix_theta_bias_ptr,
        mix_k_prev_bias_ptr,
        mix_k_curr_bias_ptr,
        u_ptr,
        m_ptr,
        k_ptr,
        b_ptr,
        c_ptr,
    )
    return U, M, K, B, C


__all__ = ["scanprep_fwd_cute"]
