# pyright: reportIndexIssue=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportGeneralTypeIssues=false
"""Backward launcher for the split CuTe scanprep backend."""

from __future__ import annotations

import torch
import cutlass.cute as cute

from slinoss.perf import note_cache_event

from .common import SCANPREP_PARAM_DIM, assumed_align, make_fake_tensor_arg
from .kernels.bwd import ScanPrepBwdFused


_SCANPREP_BWD_CACHE: dict[tuple[object, ...], object] = {}


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _raise_cold_capture_error(resource: str) -> None:
    raise RuntimeError(
        "CuTe scanprep backward "
        f"{resource} is cold during CUDA graph capture. "
        "Warm the same scanprep backward spec once outside capture before graph capture."
    )


def scanprep_bwd(
    *,
    bc: torch.Tensor,
    coeff_aux: torch.Tensor,
    dU: torch.Tensor | None,
    dM: torch.Tensor | None,
    dK: torch.Tensor | None,
    dB: torch.Tensor | None,
    dC: torch.Tensor | None,
    n_heads: int,
    d_head: int,
    d_state: int,
    value_dtype: torch.dtype,
    params_dtype: torch.dtype,
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
    theta_bias: torch.Tensor,
    theta_sign: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
        (batch, t_size, n_heads * SCANPREP_PARAM_DIM),
        device=bc.device,
        dtype=params_dtype,
    )
    bias_grad = torch.zeros((n_heads, 4), device=bc.device, dtype=torch.float32)

    bc_c = bc if bc.is_contiguous() else bc.contiguous()
    coeff_aux_c = coeff_aux if coeff_aux.is_contiguous() else coeff_aux.contiguous()

    du_align = assumed_align(du_in)
    bc_align = assumed_align(bc_c)
    db_align = assumed_align(db_in)
    dc_align = assumed_align(dc_in)
    coeff_aux_align = assumed_align(coeff_aux_c)
    dt_bias_align = assumed_align(dt_bias)
    theta_bias_align = assumed_align(theta_bias)
    theta_sign_align = assumed_align(theta_sign)
    dm_align = assumed_align(dm_in)
    dk_align = assumed_align(dk_in)
    value_grad_align = assumed_align(value_grad)
    bc_grad_align = assumed_align(bc_grad)
    dparams_align = assumed_align(dparams)
    bias_grad_align = assumed_align(bias_grad)

    spec = (int(n_heads), int(d_head), int(d_state), SCANPREP_PARAM_DIM)
    cache_key = (
        spec,
        int(bc.device.index or 0),
        bc.dtype,
        du_in.dtype,
        db_in.dtype,
        dc_in.dtype,
        dm_in.dtype,
        dk_in.dtype,
        coeff_aux_c.dtype,
        dt_bias.dtype,
        theta_bias.dtype,
        theta_sign.dtype,
        value_grad.dtype,
        bc_grad.dtype,
        dparams.dtype,
        bias_grad.dtype,
        du_align,
        bc_align,
        db_align,
        dc_align,
        coeff_aux_align,
        dt_bias_align,
        theta_bias_align,
        theta_sign_align,
        dm_align,
        dk_align,
        value_grad_align,
        bc_grad_align,
        dparams_align,
        bias_grad_align,
        float(dt_min),
        float(dt_max),
        float(theta_init_min),
        float(theta_init_max),
        float(gamma_min),
        float(gamma_max),
        float(r_min),
        float(r_max),
        float(eps),
    )
    compiled = _SCANPREP_BWD_CACHE.get(cache_key)
    if compiled is None:
        note_cache_event("cute.scanprep.bwd.compile", hit=False)
        if _is_cuda_graph_capturing(bc.device):
            _raise_cold_capture_error("launcher cache")
        compiled = cute.compile(
            ScanPrepBwdFused(
                h_size=n_heads,
                p_size=d_head,
                n_size=d_state,
                param_dim=SCANPREP_PARAM_DIM,
                dt_min=dt_min,
                dt_max=dt_max,
                theta_init_min=theta_init_min,
                theta_init_max=theta_init_max,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                r_min=r_min,
                r_max=r_max,
                eps=eps,
            ),
            make_fake_tensor_arg(du_in, align=du_align),
            make_fake_tensor_arg(db_in, align=db_align),
            make_fake_tensor_arg(dc_in, align=dc_align),
            make_fake_tensor_arg(coeff_aux_c, align=coeff_aux_align),
            make_fake_tensor_arg(dm_in, align=dm_align),
            make_fake_tensor_arg(dk_in, align=dk_align),
            make_fake_tensor_arg(dt_bias, align=dt_bias_align),
            make_fake_tensor_arg(theta_bias, align=theta_bias_align),
            make_fake_tensor_arg(theta_sign, align=theta_sign_align),
            make_fake_tensor_arg(value_grad, align=value_grad_align),
            make_fake_tensor_arg(bc_grad, align=bc_grad_align),
            make_fake_tensor_arg(dparams, align=dparams_align),
            make_fake_tensor_arg(bias_grad, align=bias_grad_align),
            options="--enable-tvm-ffi",
        )
        _SCANPREP_BWD_CACHE[cache_key] = compiled
    else:
        note_cache_event("cute.scanprep.bwd.compile", hit=True)

    compiled(
        du_in,
        db_in,
        dc_in,
        coeff_aux_c,
        dm_in,
        dk_in,
        dt_bias,
        theta_bias,
        theta_sign,
        value_grad,
        bc_grad,
        dparams,
        bias_grad,
    )

    return (
        value_grad,
        dparams,
        bc_grad,
        bias_grad[:, 0].contiguous(),
        bias_grad[:, 1].contiguous(),
        bias_grad[:, 2].contiguous(),
        bias_grad[:, 3].contiguous(),
    )


__all__ = ["scanprep_bwd"]
