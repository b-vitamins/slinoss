"""CuTe entry point for the scanprep operator."""

from __future__ import annotations

import torch

from slinoss.layers.backend import ScanInputs


def _match_scan_io_dtype(
    U: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enforce the v2x2ssd contract that U/B/C share dtype."""
    target_dtype = U.dtype
    if B.dtype != target_dtype:
        B = B.to(dtype=target_dtype)
    if C.dtype != target_dtype:
        C = C.to(dtype=target_dtype)
    return B, C


def scanprep_cute(
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
) -> ScanInputs:
    """CuTe scanprep contract for stateless mixer execution."""
    if (
        value.device.type != "cuda"
        or params.device.type != "cuda"
        or bc.device.type != "cuda"
    ):
        raise ValueError("CuTe scanprep requires CUDA tensors.")
    if n_heads <= 0 or d_state <= 0 or d_head <= 0:
        raise ValueError(
            f"Invalid scanprep dimensions: n_heads={n_heads}, d_state={d_state}, d_head={d_head}."
        )
    if value.ndim != 3 or params.ndim != 3 or bc.ndim != 5:
        raise ValueError(
            "Expected value=(B,T,H*P), params=(B,T,H*2), bc=(B,T,H,4,N). "
            f"Got {tuple(value.shape)}, {tuple(params.shape)}, {tuple(bc.shape)}."
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

    needs_autograd = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            value,
            params,
            bc,
            dt_bias,
            gamma_bias,
            theta_mod_bias,
            theta_bias,
        )
    )
    if needs_autograd:
        from slinoss.ops.scanprep.cute.autograd import scanprep_cute_training_autograd

        U, M, K, B, C = scanprep_cute_training_autograd(
            value,
            params,
            bc,
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
            dt_bias=dt_bias,
            gamma_bias=gamma_bias,
            theta_mod_bias=theta_mod_bias,
            theta_bias=theta_bias,
            theta_sign=theta_sign,
        )
        B, C = _match_scan_io_dtype(U, B, C)
        return ScanInputs(U=U, M=M, K=K, B=B, C=C)

    from slinoss.ops.scanprep.cute.fwd import scanprep_fwd_cute

    U, M, K, B, C = scanprep_fwd_cute(
        value,
        params,
        bc,
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
        dt_bias=dt_bias,
        gamma_bias=gamma_bias,
        theta_mod_bias=theta_mod_bias,
        theta_bias=theta_bias,
        theta_sign=theta_sign,
    )
    B, C = _match_scan_io_dtype(U, B, C)
    return ScanInputs(U=U, M=M, K=K, B=B, C=C)


__all__ = ["scanprep_cute"]
