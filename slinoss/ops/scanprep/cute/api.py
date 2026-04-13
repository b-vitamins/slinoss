"""CuTe entry point for the ``scanprep`` operator."""

import torch

from slinoss.layers.backend import ScanInputs
from slinoss.ops.scanprep.parameterization import parameterize_scan_bc_rows


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
    bc_complex_base: torch.Tensor,
    *,
    n_heads: int,
    bc_groups: int,
    d_state: int,
    d_head: int,
    bc_gain_max: float,
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
    if bc_gain_max <= 0.0:
        raise ValueError(f"bc_gain_max must be positive. Got {bc_gain_max}.")
    if value.ndim != 3 or params.ndim != 3 or bc.ndim != 5:
        raise ValueError(
            "Expected value=(B,T,H*P), params=(B,T,H*2), bc=(B,T,G,2,N). "
            f"Got {tuple(value.shape)}, {tuple(params.shape)}, {tuple(bc.shape)}."
        )
    expected_base_shape = (bc_groups, 2, d_state)
    if tuple(map(int, bc_complex_base.shape)) != expected_base_shape:
        raise ValueError(
            "Expected bc_complex_base shape "
            f"{expected_base_shape}. Got {tuple(map(int, bc_complex_base.shape))}."
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
    gamma_min: float,
    gamma_max: float,
    r_min: float,
    r_max: float,
    eps: float,
    bc_gain_max: float,
    bc_complex_base: torch.Tensor,
    dt_bias: torch.Tensor,
    gamma_bias: torch.Tensor,
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
        bc_complex_base,
        n_heads=n_heads,
        bc_groups=resolved_bc_groups,
        d_state=d_state,
        d_head=d_head,
        bc_gain_max=bc_gain_max,
    )
    if _should_use_scanprep_autograd(
        value,
        params,
        bc,
        bc_complex_base,
        dt_bias,
        gamma_bias,
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
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                r_min=r_min,
                r_max=r_max,
                eps=eps,
                bc_gain_max=bc_gain_max,
                bc_complex_base=bc_complex_base,
                dt_bias=dt_bias,
                gamma_bias=gamma_bias,
                theta_mod_bias=theta_mod_bias,
                theta_bias=theta_bias,
                theta_sign=theta_sign,
            )
        )

    from .kernels import scanprep_fwd_cute

    bc_rows = parameterize_scan_bc_rows(
        bc,
        bc_complex_base,
        bc_groups=resolved_bc_groups,
        d_state=d_state,
        eps=eps,
        bc_gain_max=bc_gain_max,
    )

    return _make_scan_inputs(
        *scanprep_fwd_cute(
            value,
            params,
            bc_rows,
            n_heads=n_heads,
            bc_groups=resolved_bc_groups,
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
            dt_bias=dt_bias,
            gamma_bias=gamma_bias,
            theta_mod_bias=theta_mod_bias,
            theta_bias=theta_bias,
            theta_sign=theta_sign,
        )
    )


__all__ = ["scanprep_cute"]
