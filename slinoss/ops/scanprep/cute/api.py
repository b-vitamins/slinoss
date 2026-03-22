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
) -> ScanInputs:
    """Training-only CuTe scanprep contract.

    Public contract:
    - ``value``: ``(B, T, H * P)`` post-conv/post-activation mixer values
    - ``params``: ``(B, T, H * 13)`` flat scanprep parameter stream
    - ``bc``: ``(B, T, H, 4, N)`` mixer-emitted BC tensor
    - output: packed scan-native ``(U, M, K, B, C)``

    Design constraints:
    - BC generation stays outside this backend
    - this boundary is training-only for now
    - the default eager/reference backend remains the source of truth until the
      fused CuTe implementation is complete
    """
    if (
        value.device.type != "cuda"
        or params.device.type != "cuda"
        or bc.device.type != "cuda"
    ):
        raise ValueError("CuTe scanprep requires CUDA tensors.")
    if normalize_bc and (b_scale is None or c_scale is None):
        raise ValueError("normalize_bc=True requires b_scale and c_scale tensors.")
    if n_heads <= 0 or d_state <= 0 or d_head <= 0:
        raise ValueError(
            f"Invalid scanprep dimensions: n_heads={n_heads}, d_state={d_state}, d_head={d_head}."
        )
    if value.ndim != 3 or params.ndim != 3 or bc.ndim != 5:
        raise ValueError(
            "Expected value=(B,T,H*P), params=(B,T,H*13), bc=(B,T,H,4,N). "
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

    grads_enabled = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            value,
            params,
            bc,
            dt_bias,
            gamma_bias,
            omega_bias,
            mix_r_bias,
            mix_theta_bias,
            mix_k_prev_bias,
            mix_k_curr_bias,
            b_scale,
            c_scale,
        )
    )
    if grads_enabled:
        from slinoss.ops.scanprep.cute.autograd import scanprep_cute_training_autograd

        U, M, K, B, C = scanprep_cute_training_autograd(
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
    )
    return ScanInputs(U=U, M=M, K=K, B=B, C=C)


__all__ = ["scanprep_cute"]
