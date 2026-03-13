"""Paper-faithful discretization for the SLinOSS mixer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(p) - torch.log1p(-p)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


def principal_angle(theta: torch.Tensor) -> torch.Tensor:
    """Wraps angles to the principal interval ``[-pi, pi)``."""
    theta_f = theta.to(torch.float32)
    two_pi = float(2.0 * math.pi)
    return torch.remainder(theta_f + math.pi, two_pi) - math.pi


def _pack_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x).to(torch.float32).contiguous()


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
    z_abs = torch.abs(z)
    z_thresh = float(max(1.0e-4, math.sqrt(max(float(eps), 1.0e-12))))
    small = z_abs < z_thresh
    if bool(small.any()):
        safe_z = torch.where(small, torch.ones_like(z), z)

        # kappa1(z) = (exp(z) - 1) / z, kappa2(z) = (exp(z) * (z - 1) + 1) / z^2
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
    """Builds packed complex transitions from polar parameters."""
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
    """Evaluates exact FOH taps with a small-``lambda`` continuation.

    The closed-form taps use ``1 / lambda`` and ``1 / lambda^2`` where
    ``lambda = (log r + i theta) / dt``. Near ``lambda = 0`` those expressions
    are removable singularities, so we switch to the corresponding Taylor series
    instead of dividing by tiny complex numbers.
    """

    dt_f = dt.to(torch.float32).clamp_min(max(1e-6, float(eps)))
    r_f = r.to(torch.float32).clamp(min=max(1e-12, float(eps)), max=1.0)
    theta_f = principal_angle(theta)

    rho = torch.polar(r_f, theta_f)
    log_r_f = torch.log(r_f)
    return _foh_taps_from_normalized(dt_f, log_r_f, theta_f, rho, eps=eps)


@dataclass(frozen=True)
class SLinOSSDiscretizationOutput:
    """Structured oscillator coefficients for the scan backend."""

    M: torch.Tensor
    K: torch.Tensor
    dt: torch.Tensor
    r: torch.Tensor
    theta: torch.Tensor


class SLinOSSDiscretizer(nn.Module):
    """Maps per-token head parameters to ``(M, K)`` for ``v2x2ssd``.

    Per head the parameter stream is ordered as:
    ``[dt, gamma, omega, r, theta, mix_r, mix_theta, mix_k_prev, mix_k_curr,``
    ``k_prev_re, k_prev_im, k_curr_re, k_curr_im]``.
    """

    param_dim: int = 13

    def __init__(
        self,
        *,
        n_heads: int,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_bound: float = math.pi,
        k_max: float = 0.5,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _require(n_heads > 0, f"n_heads must be positive. Got {n_heads}.")
        _require(
            0.0 < dt_min < dt_max,
            f"Require 0 < dt_min < dt_max. Got {dt_min}, {dt_max}.",
        )
        _require(
            0.0 < dt_init_floor <= dt_max,
            f"Require 0 < dt_init_floor <= dt_max. Got {dt_init_floor}.",
        )
        _require(
            0.0 < r_min <= r_max <= 1.0,
            f"Require 0 < r_min <= r_max <= 1. Got {r_min}, {r_max}.",
        )
        _require(
            0.0 < theta_bound <= math.pi,
            f"theta_bound must be in (0, pi]. Got {theta_bound}.",
        )
        _require(k_max > 0.0, f"k_max must be positive. Got {k_max}.")

        self.n_heads = int(n_heads)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.theta_bound = float(theta_bound)
        self.k_max = float(k_max)
        self.eps = float(eps)

        fp32 = torch.float32
        self.dt_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.gamma_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.omega_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.mix_r_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.mix_theta_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.mix_k_prev_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.mix_k_curr_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dt_lo = max(self.dt_min, self.dt_init_floor)
        dt_hi = self.dt_max
        _require(dt_hi > dt_lo > 0.0, f"Bad dt init bounds: {dt_lo}, {dt_hi}.")

        dt0 = torch.exp(
            torch.rand((self.n_heads,), device=self.dt_bias.device, dtype=torch.float32)
            * (math.log(dt_hi) - math.log(dt_lo))
            + math.log(dt_lo)
        )
        dt_u0 = (dt0 - self.dt_min) / (self.dt_max - self.dt_min)
        gamma0 = torch.full_like(self.gamma_bias, 1.0)
        omega0 = torch.zeros_like(self.omega_bias)

        with torch.no_grad():
            self.dt_bias.copy_(_logit(dt_u0))
            self.gamma_bias.copy_(_inv_softplus(gamma0))
            self.omega_bias.copy_(omega0)
            self.mix_r_bias.copy_(_logit(torch.full_like(self.mix_r_bias, 0.9)))
            self.mix_theta_bias.copy_(_logit(torch.full_like(self.mix_theta_bias, 0.9)))
            self.mix_k_prev_bias.copy_(
                _logit(torch.full_like(self.mix_k_prev_bias, 0.95))
            )
            self.mix_k_curr_bias.copy_(
                _logit(torch.full_like(self.mix_k_curr_bias, 0.95))
            )

    def _compute_coefficients(
        self,
        params: torch.Tensor,
        *,
        include_aux: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if params.ndim != 4 or params.shape[-2] != self.n_heads:
            raise ValueError(
                f"Expected params shape (batch, T, {self.n_heads}, {self.param_dim}), "
                f"got {tuple(params.shape)}."
            )
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"Expected last dim {self.param_dim}, got {int(params.shape[-1])}."
            )

        # Run the coefficient algebra in the scan backend's native (B, H, T, *)
        # layout so we do not materialize transposed outputs afterward.
        p = params.to(torch.float32).permute(0, 2, 1, 3)
        (
            dt_raw,
            gamma_raw,
            omega_raw,
            r_raw,
            theta_raw,
            mix_r_raw,
            mix_theta_raw,
            mix_k_prev_raw,
            mix_k_curr_raw,
            k_prev_re,
            k_prev_im,
            k_curr_re,
            k_curr_im,
        ) = p.unbind(dim=-1)

        head = (1, self.n_heads, 1)
        sigmoid_inputs = torch.stack(
            [
                dt_raw + self.dt_bias.view(*head),
                r_raw,
                mix_r_raw + self.mix_r_bias.view(*head),
                mix_theta_raw + self.mix_theta_bias.view(*head),
                mix_k_prev_raw + self.mix_k_prev_bias.view(*head),
                mix_k_curr_raw + self.mix_k_curr_bias.view(*head),
            ],
            dim=-1,
        )
        (
            dt_u,
            r_direct_u,
            mix_r,
            mix_theta,
            mix_k_prev,
            mix_k_curr,
        ) = torch.sigmoid(sigmoid_inputs).unbind(dim=-1)
        dt = self.dt_min + (self.dt_max - self.dt_min) * dt_u
        gamma = F.softplus(gamma_raw + self.gamma_bias.view(*head))
        omega = omega_raw + self.omega_bias.view(*head)

        r_struct = self.r_min + (self.r_max - self.r_min) * torch.exp(-gamma * dt)
        theta_struct = omega * dt

        tanh_outputs = torch.tanh(
            torch.stack([theta_raw, k_prev_re, k_prev_im, k_curr_re, k_curr_im], dim=-1)
        )
        theta_direct = self.theta_bound * tanh_outputs[..., 0]
        k_prev_learned = self.k_max * tanh_outputs[..., 1:3]
        k_curr_learned = self.k_max * tanh_outputs[..., 3:5]
        r_direct = self.r_min + (self.r_max - self.r_min) * r_direct_u

        r = mix_r * r_struct + (1.0 - mix_r) * r_direct
        theta = principal_angle(
            mix_theta * theta_struct + (1.0 - mix_theta) * theta_direct
        )

        dt_f = dt.to(torch.float32).clamp_min(max(1e-6, float(self.eps)))
        r_f = r.to(torch.float32).clamp(min=max(1e-12, float(self.eps)), max=1.0)
        theta_f = theta
        log_r_f = torch.log(r_f)
        rho = torch.polar(r_f, theta_f)

        k_prev_struct, k_curr_struct = _foh_taps_from_normalized(
            dt_f, log_r_f, theta_f, rho, eps=self.eps
        )
        k_prev = (
            mix_k_prev.unsqueeze(-1) * k_prev_struct
            + (1.0 - mix_k_prev.unsqueeze(-1)) * k_prev_learned
        )
        k_curr = (
            mix_k_curr.unsqueeze(-1) * k_curr_struct
            + (1.0 - mix_k_curr.unsqueeze(-1)) * k_curr_learned
        )

        M = _pack_complex(rho)
        K = torch.stack([k_prev, k_curr], dim=-2)
        if not include_aux:
            return M, K, None, None, None
        return (
            M,
            K,
            dt,
            r,
            theta,
        )

    def scan_coeffs(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        M, K, _, _, _ = self._compute_coefficients(params, include_aux=False)
        return M, K

    def forward(self, params: torch.Tensor) -> SLinOSSDiscretizationOutput:  # type: ignore[override]
        M, K, dt, r, theta = self._compute_coefficients(params, include_aux=True)
        assert dt is not None and r is not None and theta is not None
        return SLinOSSDiscretizationOutput(
            M=M,
            K=K,
            dt=dt,
            r=r,
            theta=theta,
        )

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, dt=[{self.dt_min:g},{self.dt_max:g}], "
            f"r=[{self.r_min:g},{self.r_max:g}], theta_bound={self.theta_bound:g}"
        )


__all__ = [
    "SLinOSSDiscretizationOutput",
    "SLinOSSDiscretizer",
    "build_transition_from_polar",
    "foh_taps_from_polar",
    "principal_angle",
]
