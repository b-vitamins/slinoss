"""CuTe entry point for the ``scanprep`` operator."""

import torch

from slinoss.layers.backend import ScanInputs
from slinoss.ops.scanprep.parameterization import (
    RAW_BC_PARAM_ROWS,
    validate_scan_bc_raw,
)
from .common import SCANPREP_PARAM_DIM


def _match_scan_io_dtype(
    U: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_dtype = U.dtype
    if B.dtype != target_dtype:
        B = B.to(dtype=target_dtype)
    if C.dtype != target_dtype:
        C = C.to(dtype=target_dtype)
    return B, C


def _validate_scanprep_inputs(
    value: torch.Tensor,
    params: torch.Tensor,
    bc: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int,
    d_state: int,
    d_head: int,
) -> None:
    if (
        value.device.type != "cuda"
        or params.device.type != "cuda"
        or bc.device.type != "cuda"
    ):
        raise ValueError("CuTe scanprep requires CUDA tensors.")
    if n_heads <= 0 or bc_groups <= 0 or d_state <= 0 or d_head <= 0:
        raise ValueError(
            "Invalid scanprep dimensions: "
            f"n_heads={n_heads}, bc_groups={bc_groups}, d_state={d_state}, d_head={d_head}."
        )
    if n_heads % bc_groups != 0:
        raise ValueError(
            f"n_heads must be divisible by bc_groups. Got {n_heads}, {bc_groups}."
        )
    if value.ndim != 3 or params.ndim != 3 or bc.ndim != 5:
        raise ValueError(
            f"Expected value=(B,T,H*P), params=(B,T,H*{SCANPREP_PARAM_DIM}), "
            f"bc=(B,T,G,{RAW_BC_PARAM_ROWS},N). "
            f"Got {tuple(value.shape)}, {tuple(params.shape)}, {tuple(bc.shape)}."
        )
    validate_scan_bc_raw(
        bc,
        bc_groups=bc_groups,
        d_state=d_state,
    )
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if (
        value.dtype not in supported_dtypes
        or params.dtype not in supported_dtypes
        or bc.dtype not in supported_dtypes
    ):
        raise NotImplementedError(
            "CuTe scanprep supports only float16, bfloat16, and float32 inputs."
        )


def _should_use_scanprep_autograd(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def _make_scan_inputs(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> ScanInputs:
    B_out, C_out = _match_scan_io_dtype(U, B, C)
    return ScanInputs(U=U, M=M, K=K, B=B_out, C=C_out)


def scanprep_cute(
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
    theta_mod_scale: float,
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
) -> ScanInputs:
    """CuTe scanprep contract for stateless mixer execution."""
    resolved_bc_groups = n_heads if bc_groups is None else int(bc_groups)
    _validate_scanprep_inputs(
        value,
        params,
        bc,
        n_heads=n_heads,
        bc_groups=resolved_bc_groups,
        d_state=d_state,
        d_head=d_head,
    )
    if _should_use_scanprep_autograd(
        value,
        params,
        bc,
        dt_bias,
        alpha_bias,
        theta_mod_bias,
        theta_bias,
    ):
        from .autograd import scanprep_cute_training_autograd

        return _make_scan_inputs(
            *scanprep_cute_training_autograd(
                value,
                params,
                bc,
                n_heads=n_heads,
                bc_groups=resolved_bc_groups,
                d_state=d_state,
                d_head=d_head,
                dt_min=dt_min,
                dt_max=dt_max,
                theta_init_min=theta_init_min,
                theta_init_max=theta_init_max,
                theta_mod_scale=theta_mod_scale,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                r_min=r_min,
                r_max=r_max,
                eps=eps,
                dt_bias=dt_bias,
                alpha_bias=alpha_bias,
                theta_mod_bias=theta_mod_bias,
                theta_bias=theta_bias,
                theta_sign=theta_sign,
            )
        )

    from .kernels import _scanprep_fwd_cute_prevalidated

    outputs = _scanprep_fwd_cute_prevalidated(
        value,
        params,
        bc,
        n_heads=n_heads,
        bc_groups=resolved_bc_groups,
        d_state=d_state,
        d_head=d_head,
        dt_min=dt_min,
        dt_max=dt_max,
        theta_init_min=theta_init_min,
        theta_init_max=theta_init_max,
        theta_mod_scale=theta_mod_scale,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        r_min=r_min,
        r_max=r_max,
        eps=eps,
        dt_bias=dt_bias,
        alpha_bias=alpha_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    return _make_scan_inputs(
        outputs.U,
        outputs.M,
        outputs.K,
        outputs.B,
        outputs.C,
    )


__all__ = ["scanprep_cute"]
