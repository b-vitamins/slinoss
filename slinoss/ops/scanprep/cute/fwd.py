# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportGeneralTypeIssues=false
"""Forward launcher for the split CuTe scanprep backend."""

from __future__ import annotations

from typing import cast

import torch
import cutlass.cute as cute

from slinoss.perf import note_cache_event

from .common import (
    COEFF_AUX_FIELDS,
    assumed_align,
    make_row_major_stride,
    torch_to_cutlass_dtype,
)
from .kernels.fwd import ScanPrepFwdFused


_SCANPREP_FWD_CACHE: dict[tuple[object, ...], object] = {}
_SCANPREP_DUMMY_RMS_INV_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
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


def _make_fake_tensor_arg(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    align: int | None = None,
    dynamic_stride: bool = False,
):
    fake_shape = tuple(
        int(dim) for dim in (shape if shape is not None else tensor.shape)
    )
    fake_stride = tuple(
        int(step) for step in (stride if stride is not None else tensor.stride())
    )
    assumed = int(align if align is not None else assumed_align(tensor))
    if not dynamic_stride and fake_stride == make_row_major_stride(fake_shape):
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        return cute.runtime.make_fake_compact_tensor(
            torch_to_cutlass_dtype(tensor.dtype),
            dynamic_shape,
            stride_order=tuple(reversed(range(len(fake_shape)))),
            assumed_align=assumed,
        )
    if dynamic_stride:
        dynamic_shape = tuple(cute.sym_int32() for _ in fake_shape)
        dynamic_fake_stride = tuple(
            0 if step == 0 else cute.sym_int32() for step in fake_stride
        )
        return cute.runtime.make_fake_tensor(
            torch_to_cutlass_dtype(tensor.dtype),
            dynamic_shape,
            stride=dynamic_fake_stride,
            assumed_align=assumed,
        )
    return cute.runtime.make_fake_tensor(
        torch_to_cutlass_dtype(tensor.dtype),
        fake_shape,
        stride=fake_stride,
        assumed_align=assumed,
    )


def _get_dummy_rms_inv(
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
    cached = _SCANPREP_DUMMY_RMS_INV_CACHE.get(key)
    if cached is not None:
        note_cache_event("cute.scanprep.fwd.dummy_rms_inv", hit=True)
        return cached
    note_cache_event("cute.scanprep.fwd.dummy_rms_inv", hit=False)
    if _is_cuda_graph_capturing(device):
        _raise_cold_capture_error("dummy_rms_inv cache")
    cached = torch.empty(
        (batch, n_heads, t_size, 4), device=device, dtype=torch.float32
    )
    _cache_set(_SCANPREP_DUMMY_RMS_INV_CACHE, key, cached)
    return cached


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
    return_aux: bool,
) -> tuple[
    torch.Tensor,
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
    if return_aux:
        rms_inv = torch.empty(
            (batch, n_heads, t_size, 4), device=value.device, dtype=torch.float32
        )
        coeff_aux = torch.empty(
            (batch, n_heads, COEFF_AUX_FIELDS, t_size),
            device=value.device,
            dtype=torch.float32,
        )
    else:
        rms_inv = _get_dummy_rms_inv(
            device=value.device, batch=batch, n_heads=n_heads, t_size=t_size
        )
        coeff_aux = _get_dummy_coeff_aux(
            device=value.device, batch=batch, n_heads=n_heads, t_size=t_size
        )

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
    value_align = assumed_align(value_c)
    bc_align = assumed_align(bc_c)
    b_scale_align = assumed_align(b_scale_c)
    c_scale_align = assumed_align(c_scale_c)
    params_align = assumed_align(params_c)
    dt_bias_align = assumed_align(dt_bias)
    gamma_bias_align = assumed_align(gamma_bias)
    omega_bias_align = assumed_align(omega_bias)
    mix_r_bias_align = assumed_align(mix_r_bias)
    mix_theta_bias_align = assumed_align(mix_theta_bias)
    mix_k_prev_bias_align = assumed_align(mix_k_prev_bias)
    mix_k_curr_bias_align = assumed_align(mix_k_curr_bias)
    u_align = assumed_align(U)
    b_align = assumed_align(B)
    c_align = assumed_align(C)
    m_align = assumed_align(M)
    k_align = assumed_align(K)
    rms_inv_align = assumed_align(rms_inv)
    coeff_aux_align = assumed_align(coeff_aux)

    spec = (int(n_heads), int(d_head), int(d_state))
    cache_key = (
        spec,
        int(value.device.index or 0),
        bool(normalize_bc),
        value_c.dtype,
        params_c.dtype,
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
        rms_inv.dtype,
        coeff_aux.dtype,
        value_align,
        bc_align,
        b_scale_align,
        c_scale_align,
        params_align,
        dt_bias_align,
        gamma_bias_align,
        omega_bias_align,
        mix_r_bias_align,
        mix_theta_bias_align,
        mix_k_prev_bias_align,
        mix_k_curr_bias_align,
        u_align,
        b_align,
        c_align,
        m_align,
        k_align,
        rms_inv_align,
        coeff_aux_align,
        bool(return_aux),
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
        note_cache_event("cute.scanprep.fwd.compile", hit=False)
        if _is_cuda_graph_capturing(value.device):
            _raise_cold_capture_error("launcher cache")
        compiled = cute.compile(
            ScanPrepFwdFused(
                h_size=n_heads,
                p_size=d_head,
                n_size=d_state,
                normalize_bc=normalize_bc,
                store_rms_inv=bool(return_aux and normalize_bc),
                store_coeff_aux=bool(return_aux),
                dt_min=dt_min,
                dt_max=dt_max,
                r_min=r_min,
                r_max=r_max,
                theta_bound=theta_bound,
                k_max=k_max,
                eps=eps,
            ),
            _make_fake_tensor_arg(value_c, align=value_align),
            _make_fake_tensor_arg(bc_c, align=bc_align),
            _make_fake_tensor_arg(b_scale_c, align=b_scale_align, dynamic_stride=True),
            _make_fake_tensor_arg(c_scale_c, align=c_scale_align, dynamic_stride=True),
            _make_fake_tensor_arg(params_c, align=params_align),
            _make_fake_tensor_arg(dt_bias, align=dt_bias_align),
            _make_fake_tensor_arg(gamma_bias, align=gamma_bias_align),
            _make_fake_tensor_arg(omega_bias, align=omega_bias_align),
            _make_fake_tensor_arg(mix_r_bias, align=mix_r_bias_align),
            _make_fake_tensor_arg(mix_theta_bias, align=mix_theta_bias_align),
            _make_fake_tensor_arg(mix_k_prev_bias, align=mix_k_prev_bias_align),
            _make_fake_tensor_arg(mix_k_curr_bias, align=mix_k_curr_bias_align),
            _make_fake_tensor_arg(U, align=u_align),
            _make_fake_tensor_arg(B, align=b_align),
            _make_fake_tensor_arg(C, align=c_align),
            _make_fake_tensor_arg(M, align=m_align),
            _make_fake_tensor_arg(K, align=k_align),
            _make_fake_tensor_arg(rms_inv, align=rms_inv_align),
            _make_fake_tensor_arg(coeff_aux, align=coeff_aux_align),
            options="--enable-tvm-ffi",
        )
        _SCANPREP_FWD_CACHE[cache_key] = compiled
    else:
        note_cache_event("cute.scanprep.fwd.compile", hit=True)
    compiled(
        value_c,
        bc_c,
        b_scale_c,
        c_scale_c,
        params_c,
        dt_bias,
        gamma_bias,
        omega_bias,
        mix_r_bias,
        mix_theta_bias,
        mix_k_prev_bias,
        mix_k_curr_bias,
        U,
        B,
        C,
        M,
        K,
        rms_inv,
        coeff_aux,
    )
    return U, M, K, B, C, rms_inv, coeff_aux


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
    U, M, K, B, C, _, _ = _scanprep_fwd_impl(
        value,
        params,
        bc,
        n_heads=n_heads,
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
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
        omega_bias=omega_bias,
        mix_r_bias=mix_r_bias,
        mix_theta_bias=mix_theta_bias,
        mix_k_prev_bias=mix_k_prev_bias,
        mix_k_curr_bias=mix_k_curr_bias,
        b_scale=b_scale,
        c_scale=c_scale,
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
]:
    return cast(
        tuple[
            torch.Tensor,
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
            normalize_bc=normalize_bc,
            dt_min=dt_min,
            dt_max=dt_max,
            r_min=r_min,
            r_max=r_max,
            theta_bound=theta_bound,
            k_max=k_max,
            eps=eps,
            dt_bias=dt_bias,
            gamma_bias=gamma_bias,
            omega_bias=omega_bias,
            mix_r_bias=mix_r_bias,
            mix_theta_bias=mix_theta_bias,
            mix_k_prev_bias=mix_k_prev_bias,
            mix_k_curr_bias=mix_k_curr_bias,
            b_scale=b_scale,
            c_scale=c_scale,
            return_aux=True,
        ),
    )


__all__ = ["scanprep_fwd_cute", "scanprep_fwd_cute_with_aux"]
