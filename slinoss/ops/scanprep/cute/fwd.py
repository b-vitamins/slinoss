# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Forward launcher for the split CuTe scanprep backend."""

from __future__ import annotations

from typing import cast

import torch
import cutlass.cute as cute

from slinoss.perf import note_cache_event

from .common import (
    COEFF_AUX_FIELDS,
    SCANPREP_PARAM_DIM,
    assumed_align,
    make_fake_tensor_arg,
)
from .kernels.fwd import ScanPrepFwdFused


_SCANPREP_FWD_CACHE: dict[tuple[object, ...], object] = {}
_SCANPREP_DUMMY_COEFF_AUX_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_SCANPREP_DUMMY_CACHE_LIMIT = 8


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_cold_capture_error(resource: str) -> None:
    raise RuntimeError(
        "CuTe scanprep forward "
        f"{resource} is cold during CUDA graph capture. "
        "Warm the same scanprep forward spec once outside capture before graph capture."
    )


def _cache_set(
    cache: dict[tuple[object, ...], object], key: tuple[object, ...], value
) -> None:
    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= _SCANPREP_DUMMY_CACHE_LIMIT:
        cache.pop(next(iter(cache)), None)
    cache[key] = value


def _get_dummy_coeff_aux(
    *,
    device: torch.device,
    batch: int,
    n_heads: int,
    t_size: int,
) -> torch.Tensor:
    key = (
        device.type,
        device.index if device.index is not None else -1,
        int(batch),
        int(n_heads),
        int(t_size),
    )
    cached = _SCANPREP_DUMMY_COEFF_AUX_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.scanprep.fwd.dummy_coeff_aux", hit=True)
        return cached
    note_cache_event("cute.scanprep.fwd.dummy_coeff_aux", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("dummy_coeff_aux cache")
    cached = torch.empty(
        (batch, n_heads, COEFF_AUX_FIELDS, t_size), device=device, dtype=torch.float32
    )
    _cache_set(_SCANPREP_DUMMY_COEFF_AUX_CACHE, key, cached)
    return cached


def _scanprep_fwd_impl(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    omega_min: float,
    zeta_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    zeta_bias: torch.Tensor,
    omega_mod_bias: torch.Tensor,
    omega_natural_bias: torch.Tensor,
    mix_r_bias: torch.Tensor,
    omega_sign: torch.Tensor,
    return_aux: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    value_c = value if value.is_contiguous() else value.contiguous()
    params_c = params if params.is_contiguous() else params.contiguous()
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    batch, t_size, width = map(int, value_c.shape)
    if width != int(n_heads * d_head):
        raise ValueError(f"value width must be {n_heads * d_head}. Got {width}.")
    if tuple(map(int, params.shape)) != (
        batch,
        t_size,
        int(n_heads * SCANPREP_PARAM_DIM),
    ):
        raise ValueError(
            "params must be "
            f"{(batch, t_size, int(n_heads * SCANPREP_PARAM_DIM))}. "
            f"Got {tuple(params.shape)}."
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
    if return_aux:
        coeff_aux = torch.empty(
            (batch, n_heads, COEFF_AUX_FIELDS, t_size),
            device=value.device,
            dtype=torch.float32,
        )
    else:
        coeff_aux = _get_dummy_coeff_aux(
            device=value.device, batch=batch, n_heads=n_heads, t_size=t_size
        )

    value_align = assumed_align(value_c)
    bc_align = assumed_align(bc_c)
    params_align = assumed_align(params_c)
    dt_bias_align = assumed_align(dt_bias)
    zeta_bias_align = assumed_align(zeta_bias)
    omega_mod_bias_align = assumed_align(omega_mod_bias)
    omega_natural_bias_align = assumed_align(omega_natural_bias)
    mix_r_bias_align = assumed_align(mix_r_bias)
    omega_sign_align = assumed_align(omega_sign)
    u_align = assumed_align(U)
    b_align = assumed_align(B)
    c_align = assumed_align(C)
    m_align = assumed_align(M)
    k_align = assumed_align(K)
    coeff_aux_align = assumed_align(coeff_aux)

    spec = (int(n_heads), int(d_head), int(d_state))
    cache_key = (
        spec,
        int(value.device.index or 0),
        value_c.dtype,
        params_c.dtype,
        bc_c.dtype,
        dt_bias.dtype,
        zeta_bias.dtype,
        omega_mod_bias.dtype,
        omega_natural_bias.dtype,
        mix_r_bias.dtype,
        omega_sign.dtype,
        U.dtype,
        M.dtype,
        K.dtype,
        B.dtype,
        C.dtype,
        coeff_aux.dtype,
        value_align,
        bc_align,
        params_align,
        dt_bias_align,
        zeta_bias_align,
        omega_mod_bias_align,
        omega_natural_bias_align,
        mix_r_bias_align,
        omega_sign_align,
        u_align,
        b_align,
        c_align,
        m_align,
        k_align,
        coeff_aux_align,
        bool(return_aux),
        float(dt_min),
        float(dt_max),
        float(omega_min),
        float(zeta_max),
        float(r_min),
        float(r_max),
        float(eps),
    )
    compiled = _SCANPREP_FWD_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cute.scanprep.fwd.compile", hit=False)
        if _is_cuda_graph_capturing(value.device):
            _raise_cold_capture_error("launcher cache")
        compiled = cute.compile(
            ScanPrepFwdFused(
                h_size=n_heads,
                p_size=d_head,
                n_size=d_state,
                store_coeff_aux=bool(return_aux),
                dt_min=dt_min,
                dt_max=dt_max,
                omega_min=omega_min,
                zeta_max=zeta_max,
                r_min=r_min,
                r_max=r_max,
                eps=eps,
            ),
            make_fake_tensor_arg(value_c, align=value_align),
            make_fake_tensor_arg(bc_c, align=bc_align),
            make_fake_tensor_arg(params_c, align=params_align),
            make_fake_tensor_arg(dt_bias, align=dt_bias_align),
            make_fake_tensor_arg(zeta_bias, align=zeta_bias_align),
            make_fake_tensor_arg(omega_mod_bias, align=omega_mod_bias_align),
            make_fake_tensor_arg(omega_natural_bias, align=omega_natural_bias_align),
            make_fake_tensor_arg(mix_r_bias, align=mix_r_bias_align),
            make_fake_tensor_arg(omega_sign, align=omega_sign_align),
            make_fake_tensor_arg(U, align=u_align),
            make_fake_tensor_arg(B, align=b_align),
            make_fake_tensor_arg(C, align=c_align),
            make_fake_tensor_arg(M, align=m_align),
            make_fake_tensor_arg(K, align=k_align),
            make_fake_tensor_arg(coeff_aux, align=coeff_aux_align),
            options="--enable-tvm-ffi",
        )
        _SCANPREP_FWD_CACHE[cache_key] = compiled
    else:
        note_cache_event("cute.scanprep.fwd.compile", hit=True)
    compiled(
        value_c,
        bc_c,
        params_c,
        dt_bias,
        zeta_bias,
        omega_mod_bias,
        omega_natural_bias,
        mix_r_bias,
        omega_sign,
        U,
        B,
        C,
        M,
        K,
        coeff_aux,
    )
    return U, M, K, B, C, coeff_aux


def scanprep_fwd_cute(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    omega_min: float,
    zeta_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    zeta_bias: torch.Tensor,
    omega_mod_bias: torch.Tensor,
    omega_natural_bias: torch.Tensor,
    mix_r_bias: torch.Tensor,
    omega_sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    U, M, K, B, C, _ = _scanprep_fwd_impl(
        value,
        params,
        bc,
        n_heads=n_heads,
        d_state=d_state,
        d_head=d_head,
        dt_min=dt_min,
        dt_max=dt_max,
        omega_min=omega_min,
        zeta_max=zeta_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        zeta_bias=zeta_bias,
        omega_mod_bias=omega_mod_bias,
        omega_natural_bias=omega_natural_bias,
        mix_r_bias=mix_r_bias,
        omega_sign=omega_sign,
        return_aux=False,
    )
    return U, M, K, B, C


def scanprep_fwd_cute_with_aux(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    d_state: int,
    d_head: int,
    dt_min: float,
    dt_max: float,
    omega_min: float,
    zeta_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    dt_bias: torch.Tensor,
    zeta_bias: torch.Tensor,
    omega_mod_bias: torch.Tensor,
    omega_natural_bias: torch.Tensor,
    mix_r_bias: torch.Tensor,
    omega_sign: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        _scanprep_fwd_impl(
            value,
            params,
            bc,
            n_heads=n_heads,
            d_state=d_state,
            d_head=d_head,
            dt_min=dt_min,
            dt_max=dt_max,
            omega_min=omega_min,
            zeta_max=zeta_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            dt_bias=dt_bias,
            zeta_bias=zeta_bias,
            omega_mod_bias=omega_mod_bias,
            omega_natural_bias=omega_natural_bias,
            mix_r_bias=mix_r_bias,
            omega_sign=omega_sign,
            return_aux=True,
        ),
    )


__all__ = ["scanprep_fwd_cute", "scanprep_fwd_cute_with_aux"]
