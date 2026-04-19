"""Reference scanprep algebra and shared utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference ``(M, K)`` generation from flat ``(B, T, H * param_dim)`` params."""
    if params.ndim != 3 or params.shape[-1] != n_heads * param_dim:
        raise ValueError(
            f"params must be (batch, T, {n_heads * param_dim}). Got {tuple(params.shape)}."
        )

    p = params.view(params.shape[0], params.shape[1], n_heads, param_dim)
    p = p.permute(0, 2, 1, 3).to(torch.float32)
    bias = torch.stack((dt_bias, alpha_bias, theta_mod_bias), dim=-1)
    p = p + bias.view(1, n_heads, 1, param_dim)

    dt_raw = p[..., 0]
    alpha_raw = p[..., 1]
    theta_mod_raw = p[..., 2]

    dt = dt_min + (dt_max - dt_min) * torch.sigmoid(dt_raw)
    theta_span = float(max(theta_init_max - theta_init_min, 1.0e-6))
    theta_u = torch.sigmoid(
        theta_bias.view(1, n_heads, 1)
        + float(theta_mod_scale) * torch.tanh(theta_mod_raw)
    )
    theta_drive = theta_init_min + theta_span * theta_u
    alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(alpha_raw)
    theta = principal_angle(theta_sign.view(1, n_heads, 1) * theta_drive)
    r_struct = r_min + (r_max - r_min) * torch.exp(-alpha)
    r = r_struct
    log_r_f = torch.log(r)
    rho = torch.polar(r, theta)
    k_prev, k_curr = _foh_taps_from_normalized(dt, log_r_f, theta, rho, eps=eps)

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
