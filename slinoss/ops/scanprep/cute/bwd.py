# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Backward launcher for the split CuTe scanprep backend."""

from __future__ import annotations

import torch
import cutlass.cute as cute

from .common import make_ptr_arg
from .kernels.bwd import ScanPrepBwdFused


_SCANPREP_BWD_CACHE: dict[tuple[object, ...], object] = {}


def scanprep_bwd(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
    rms_inv: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    d_head: int,
    d_state: int,
    normalize_bc: bool,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
    dt_min: float,
    dt_max: float,
    r_min: float,
    r_max: float,
    theta_bound: float,
    k_max: float,
    eps: float,
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
    batch, t_size, _, _, _ = map(int, bc.shape)

    du_in = (
        dU
        if dU is not None
        else torch.zeros(
            (batch, n_heads, t_size, d_head),
            device=bc.device,
            dtype=value_dtype,
        )
    )
    if not du_in.is_contiguous():
        du_in = du_in.contiguous()
    db_in = (
        dB
        if dB is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2 * d_state), device=bc.device, dtype=bc.dtype
        )
    )
    if not db_in.is_contiguous():
        db_in = db_in.contiguous()
    dc_in = (
        dC
        if dC is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2 * d_state), device=bc.device, dtype=bc.dtype
        )
    )
    if not dc_in.is_contiguous():
        dc_in = dc_in.contiguous()
    dm_in = (
        dM
        if dM is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2), device=bc.device, dtype=torch.float32
        )
    )
    if not dm_in.is_contiguous():
        dm_in = dm_in.contiguous()
    dk_in = (
        dK
        if dK is not None
        else torch.zeros(
            (batch, n_heads, t_size, 2, 2), device=bc.device, dtype=torch.float32
        )
    )
    if not dk_in.is_contiguous():
        dk_in = dk_in.contiguous()

    value_grad = torch.empty(
        (batch, t_size, n_heads * d_head), device=bc.device, dtype=value_dtype
    )
    bc_grad = torch.empty_like(bc)
    dparams = torch.empty(
        (batch, t_size, n_heads * 13), device=bc.device, dtype=params_dtype
    )
    value_warps_per_block = 8
    pack_warps_per_block = 12
    coeff_block_size = 512
    scale_block_size = 256
    bias_block_size = 224
    pack_bt_tile = 256
    scale_grad = torch.zeros(
        (n_heads, 4, d_state), device=bc.device, dtype=torch.float32
    )
    scale_tile_count = (batch * t_size + (pack_bt_tile - 1)) // pack_bt_tile
    scale_partial = torch.empty(
        (scale_tile_count, n_heads, 4, d_state),
        device=bc.device,
        dtype=torch.float32,
    )
    bias_tile_count = (batch * t_size + (256 - 1)) // 256
    bias_partial = torch.empty(
        (bias_tile_count, n_heads, 7),
        device=bc.device,
        dtype=torch.float32,
    )
    bias_grad = torch.empty((n_heads, 7), device=bc.device, dtype=torch.float32)

    b_scale_in = (
        b_scale
        if normalize_bc and b_scale is not None
        else torch.empty((n_heads, 2, d_state), device=bc.device, dtype=bc.dtype)
    )
    c_scale_in = (
        c_scale
        if normalize_bc and c_scale is not None
        else torch.empty((n_heads, 2, d_state), device=bc.device, dtype=bc.dtype)
    )
    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    coeff_aux_c = coeff_aux if coeff_aux.is_contiguous() else coeff_aux.contiguous()
    rms_inv_c = rms_inv if rms_inv.is_contiguous() else rms_inv.contiguous()
    du_stride = tuple(int(s) for s in du_in.stride())

    du_ptr, du_align = make_ptr_arg(du_in)
    bc_ptr, bc_align = make_ptr_arg(bc_c)
    db_ptr, db_align = make_ptr_arg(db_in)
    dc_ptr, dc_align = make_ptr_arg(dc_in)
    b_scale_ptr, b_scale_align = make_ptr_arg(b_scale_in)
    c_scale_ptr, c_scale_align = make_ptr_arg(c_scale_in)
    rms_inv_ptr, rms_inv_align = make_ptr_arg(rms_inv_c)
    coeff_aux_ptr, coeff_aux_align = make_ptr_arg(coeff_aux_c)
    dm_ptr, dm_align = make_ptr_arg(dm_in)
    dk_ptr, dk_align = make_ptr_arg(dk_in)
    value_grad_ptr, value_grad_align = make_ptr_arg(value_grad)
    bc_grad_ptr, bc_grad_align = make_ptr_arg(bc_grad)
    dparams_ptr, dparams_align = make_ptr_arg(dparams)
    scale_partial_ptr, scale_partial_align = make_ptr_arg(scale_partial)
    scale_grad_ptr, scale_grad_align = make_ptr_arg(scale_grad)
    bias_partial_ptr, bias_partial_align = make_ptr_arg(bias_partial)
    bias_grad_ptr, bias_grad_align = make_ptr_arg(bias_grad)

    spec = (batch, t_size, int(n_heads), int(d_head), int(d_state), 13)
    cache_key = (
        spec,
        int(bc.device.index or 0),
        bool(normalize_bc),
        bc.dtype,
        du_in.dtype,
        db_in.dtype,
        dc_in.dtype,
        dm_in.dtype,
        dk_in.dtype,
        b_scale_in.dtype,
        c_scale_in.dtype,
        rms_inv_c.dtype,
        coeff_aux_c.dtype,
        value_grad.dtype,
        bc_grad.dtype,
        dparams.dtype,
        scale_partial.dtype,
        scale_grad.dtype,
        bias_partial.dtype,
        bias_grad.dtype,
        du_stride,
        du_align,
        bc_align,
        db_align,
        dc_align,
        b_scale_align,
        c_scale_align,
        rms_inv_align,
        coeff_aux_align,
        dm_align,
        dk_align,
        value_grad_align,
        bc_grad_align,
        dparams_align,
        scale_partial_align,
        scale_grad_align,
        bias_partial_align,
        bias_grad_align,
        int(value_warps_per_block),
        int(pack_warps_per_block),
        int(coeff_block_size),
        int(scale_block_size),
        int(bias_block_size),
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
                normalize_bc=normalize_bc,
                dt_min=dt_min,
                dt_max=dt_max,
                r_min=r_min,
                r_max=r_max,
                theta_bound=theta_bound,
                k_max=k_max,
                eps=eps,
                value_warps_per_block=value_warps_per_block,
                pack_warps_per_block=pack_warps_per_block,
                coeff_block_size=coeff_block_size,
                scale_block_size=scale_block_size,
                bias_block_size=bias_block_size,
            ),
            du_ptr,
            bc_ptr,
            db_ptr,
            dc_ptr,
            b_scale_ptr,
            c_scale_ptr,
            rms_inv_ptr,
            coeff_aux_ptr,
            dm_ptr,
            dk_ptr,
            value_grad_ptr,
            bc_grad_ptr,
            dparams_ptr,
            scale_partial_ptr,
            scale_grad_ptr,
            bias_partial_ptr,
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
        rms_inv_ptr,
        coeff_aux_ptr,
        dm_ptr,
        dk_ptr,
        value_grad_ptr,
        bc_grad_ptr,
        dparams_ptr,
        scale_partial_ptr,
        scale_grad_ptr,
        bias_partial_ptr,
        bias_grad_ptr,
    )

    d_b_scale = scale_grad[:, :2, :] if normalize_bc and b_scale is not None else None
    d_c_scale = scale_grad[:, 2:, :] if normalize_bc and c_scale is not None else None
    return (
        value_grad,
        dparams,
        bc_grad,
        bias_grad[:, 0],
        bias_grad[:, 1],
        bias_grad[:, 2],
        bias_grad[:, 3],
        bias_grad[:, 4],
        bias_grad[:, 5],
        bias_grad[:, 6],
        d_b_scale,
        d_c_scale,
    )


__all__ = ["scanprep_bwd"]
