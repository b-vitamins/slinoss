"""Reference scanprep algebra and shared utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def principal_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to the principal interval ``[-pi, pi)``."""
    theta_f = theta.to(torch.float32)
    two_pi = float(2.0 * math.pi)
    return torch.remainder(theta_f + math.pi, two_pi) - math.pi


def _pack_complex(x: torch.Tensor) -> torch.Tensor:
    packed = torch.view_as_real(x)
    if packed.dtype != torch.float32:
        packed = packed.to(torch.float32)
    return packed if packed.is_contiguous() else packed.contiguous()


def _foh_taps_from_normalized(
    dt_f: torch.Tensor,
    log_r_f: torch.Tensor,
    theta_f: torch.Tensor,
    rho: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FOH taps for already-normalized ``dt/log(r)/theta`` inputs."""

    z = torch.complex(log_r_f, theta_f)
    z_thresh = float(max(1.0e-4, math.sqrt(max(float(eps), 1.0e-12))))
    small = log_r_f.square() + theta_f.square() < (z_thresh * z_thresh)
    if bool(small.any()):
        safe_z = torch.where(small, torch.ones_like(z), z)

        kappa1 = (rho - 1.0) / safe_z
        kappa2 = (rho * (safe_z - 1.0) + 1.0) / (safe_z * safe_z)

        z2 = z * z
        z3 = z2 * z
        kappa1_taylor = 1.0 + 0.5 * z + z2 / 6.0 + z3 / 24.0
        kappa2_taylor = 0.5 + z / 3.0 + z2 / 8.0 + z3 / 30.0
        kappa1 = torch.where(small, kappa1_taylor, kappa1)
        kappa2 = torch.where(small, kappa2_taylor, kappa2)
    else:
        kappa1 = (rho - 1.0) / z
        kappa2 = (rho * (z - 1.0) + 1.0) / (z * z)

    k_prev = dt_f * kappa2
    k_curr = dt_f * kappa1 - k_prev
    return _pack_complex(k_prev), _pack_complex(k_curr)


def build_transition_from_polar(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Build packed complex transitions from polar parameters."""
    r_f = r.to(torch.float32).clamp_min(0.0)
    theta_f = principal_angle(theta)
    return _pack_complex(torch.polar(r_f, theta_f))


def foh_taps_from_polar(
    dt: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact FOH taps with a small-``lambda`` continuation."""

    dt_f = dt.to(torch.float32).clamp_min(max(1e-6, float(eps)))
    r_f = r.to(torch.float32).clamp(min=max(1e-12, float(eps)), max=1.0)
    theta_f = principal_angle(theta)

    rho = torch.polar(r_f, theta_f)
    log_r_f = torch.log(r_f)
    return _foh_taps_from_normalized(dt_f, log_r_f, theta_f, rho, eps=eps)


@dataclass(frozen=True)
class SLinOSSScanPrepCoefficients:
    """Structured oscillator coefficients for the scan backend."""

    M: torch.Tensor
    K: torch.Tensor
    dt: torch.Tensor
    r: torch.Tensor
    theta: torch.Tensor


def scanprep_scan_coeffs_from_flat_params(
    params: torch.Tensor,
    *,
    n_heads: int,
    param_dim: int,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``(M, K)`` generation from flat ``(B, T, H * param_dim)`` params."""
    if params.ndim != 3 or params.shape[-1] != n_heads * param_dim:
        raise ValueError(
            f"params must be (batch, T, {n_heads * param_dim}). Got {tuple(params.shape)}."
        )

    p = params.view(params.shape[0], params.shape[1], n_heads, param_dim)
    p = p.permute(0, 2, 1, 3).to(torch.float32)
    zero = torch.zeros_like(dt_bias)
    bias = torch.stack(
        (
            dt_bias,
            gamma_bias,
            omega_bias,
            zero,
            zero,
            mix_r_bias,
            mix_theta_bias,
            mix_k_prev_bias,
            mix_k_curr_bias,
            zero,
            zero,
            zero,
            zero,
        ),
        dim=-1,
    )
    p = p + bias.view(1, n_heads, 1, param_dim)

    dt_u = torch.sigmoid(p[..., 0])
    gamma = F.softplus(p[..., 1])
    omega = p[..., 2]
    r_direct_u = torch.sigmoid(p[..., 3])
    theta_direct = theta_bound * torch.tanh(p[..., 4])
    mix_r = torch.sigmoid(p[..., 5])
    mix_theta = torch.sigmoid(p[..., 6])
    mix_k_prev = torch.sigmoid(p[..., 7]).unsqueeze(-1)
    mix_k_curr = torch.sigmoid(p[..., 8]).unsqueeze(-1)
    k_prev_learned = k_max * torch.tanh(p[..., 9:11])
    k_curr_learned = k_max * torch.tanh(p[..., 11:13])

    dt = dt_min + (dt_max - dt_min) * dt_u
    r_struct = r_min + (r_max - r_min) * torch.exp(-gamma * dt)
    theta_struct = omega * dt
    r_direct = r_min + (r_max - r_min) * r_direct_u

    r = torch.lerp(r_direct, r_struct, mix_r)
    theta = principal_angle(torch.lerp(theta_direct, theta_struct, mix_theta))

    log_r_f = torch.log(r)
    rho = torch.polar(r, theta)
    k_prev_struct, k_curr_struct = _foh_taps_from_normalized(
        dt, log_r_f, theta, rho, eps=eps
    )
    k_prev = torch.lerp(k_prev_learned, k_prev_struct, mix_k_prev)
    k_curr = torch.lerp(k_curr_learned, k_curr_struct, mix_k_curr)

    M = _pack_complex(rho)
    K = torch.stack([k_prev, k_curr], dim=-2)
    return M, K


__all__ = [
    "SLinOSSScanPrepCoefficients",
    "build_transition_from_polar",
    "foh_taps_from_polar",
    "principal_angle",
    "scanprep_scan_coeffs_from_flat_params",
    "_foh_taps_from_normalized",
    "_pack_complex",
]
