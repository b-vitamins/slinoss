"""CuTe entry point for the scanprep operator."""

from __future__ import annotations

import torch

from slinoss.layers.backend import ScanInputs


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
            "Expected value=(B,T,H*P), params=(B,T,H*4), bc=(B,T,H,4,N). "
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
            zeta_bias,
            omega_mod_bias,
            omega_natural_bias,
            mix_r_bias,
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
        )
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
    )
    return ScanInputs(U=U, M=M, K=K, B=B, C=C)


__all__ = ["scanprep_cute"]
