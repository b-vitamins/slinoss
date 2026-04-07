"""Reference scanprep boundary for the SLinOSS mixer."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.ops.scanprep import (
    SLinOSSScanPrepCoefficients,
    build_transition_from_polar,
    foh_taps_from_polar,
    principal_angle,
    scanprep_cute,
)
from slinoss.ops.scanprep.reference import _foh_taps_from_normalized, _pack_complex

from .backend import (
    AutoScanPrepBackend,
    ScanInputs,
    ScanPrepBackend,
    ScanPrepInputs,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(p) - torch.log1p(-p)


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return y + torch.log(-torch.expm1(-y))


class SLinOSSScanPrep(nn.Module):
    """Builds canonical scan inputs from post-conv activations and parameter streams."""

    param_dim: int = 4
    bc_param_rows: int = 2
    omega_mod_scale: float = 0.25

    def __init__(
        self,
        *,
        n_heads: int,
        d_state: int,
        d_head: int,
        backend: ScanPrepBackend | None = None,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        omega_min: float = 0.1,
        zeta_max: float = 0.7,
        theta_init_min: float = 0.2,
        theta_init_max: float = 1.0,
        r_min: float = 0.9,
        r_max: float = 1.0,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _require(n_heads > 0, f"n_heads must be positive. Got {n_heads}.")
        _require(d_state > 0, f"d_state must be positive. Got {d_state}.")
        _require(d_head > 0, f"d_head must be positive. Got {d_head}.")
        _require(
            0.0 < dt_min < dt_max,
            f"Require 0 < dt_min < dt_max. Got {dt_min}, {dt_max}.",
        )
        _require(
            0.0 < dt_init_floor <= dt_max,
            f"Require 0 < dt_init_floor <= dt_max. Got {dt_init_floor}.",
        )
        _require(omega_min >= 0.0, f"Require omega_min >= 0. Got {omega_min}.")
        _require(
            0.0 < zeta_max < 1.0,
            f"Require 0 < zeta_max < 1. Got {zeta_max}.",
        )
        _require(
            0.0 < theta_init_min <= theta_init_max < math.pi,
            "Require 0 < theta_init_min <= theta_init_max < pi. Got "
            f"{theta_init_min}, {theta_init_max}.",
        )
        _require(
            0.0 < r_min <= r_max <= 1.0,
            f"Require 0 < r_min <= r_max <= 1. Got {r_min}, {r_max}.",
        )

        self.n_heads = int(n_heads)
        self.d_state = int(d_state)
        self.d_head = int(d_head)
        self.d_inner = int(self.n_heads * self.d_head)
        self.backend = AutoScanPrepBackend() if backend is None else backend

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)
        self.omega_min = float(omega_min)
        self.zeta_max = float(zeta_max)
        self.theta_init_min = float(theta_init_min)
        self.theta_init_max = float(theta_init_max)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.eps = float(eps)

        fp32 = torch.float32
        self.dt_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.zeta_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.omega_mod_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.omega_natural_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.mix_r_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.bc_complex_base = nn.Parameter(
            torch.empty(
                (self.n_heads, 2, self.d_state),
                device=device,
                dtype=torch.complex64,
            )
        )

        self.register_buffer(
            "omega_sign",
            torch.where(
                torch.arange(self.n_heads, device=device) % 2 == 0,
                torch.ones((self.n_heads,), device=device, dtype=fp32),
                -torch.ones((self.n_heads,), device=device, dtype=fp32),
            ),
            persistent=True,
        )
        self.register_buffer(
            "_zero_bias",
            torch.zeros((self.n_heads,), device=device, dtype=fp32),
            persistent=False,
        )
        self.reset_parameters()

    def _apply(self, fn):  # type: ignore[override]
        bc_complex_base = self._parameters.pop("bc_complex_base", None)
        super()._apply(fn)
        if bc_complex_base is None:
            return self

        probe = fn(torch.empty((), device=bc_complex_base.device, dtype=torch.float32))
        restored = nn.Parameter(
            bc_complex_base.detach().to(
                device=probe.device, dtype=bc_complex_base.dtype
            ),
            requires_grad=bc_complex_base.requires_grad,
        )
        if bc_complex_base.grad is not None:
            restored.grad = bc_complex_base.grad.detach().to(
                device=probe.device, dtype=bc_complex_base.dtype
            )
        self._parameters["bc_complex_base"] = restored
        return self

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

        theta_lo = max(self.omega_min * float(dt0.min()), self.theta_init_min)
        theta_hi = self.theta_init_max
        if self.n_heads == 1:
            theta0 = torch.full_like(self.omega_natural_bias, theta_hi)
        else:
            theta0 = torch.logspace(
                math.log10(theta_lo),
                math.log10(theta_hi),
                self.n_heads,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )
        omega_natural0 = theta0 / dt0.clamp_min(1.0e-6)
        zeta0 = (torch.ones_like(theta0) / omega_natural0.clamp_min(1.0e-6)).clamp(
            min=0.05,
            max=self.zeta_max * 0.8,
        )

        with torch.no_grad():
            self.dt_bias.copy_(_logit(dt_u0))
            self.zeta_bias.copy_(_logit(zeta0 / self.zeta_max))
            self.omega_mod_bias.zero_()
            self.omega_natural_bias.copy_(
                _inv_softplus((omega_natural0 - self.omega_min).clamp_min(1.0e-6))
            )
            self.mix_r_bias.copy_(_logit(torch.full_like(self.mix_r_bias, 0.9)))

            phase_grid = torch.linspace(
                -math.pi,
                math.pi,
                self.n_heads * 2 * self.d_state + 1,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )[:-1].view(self.n_heads, 2, self.d_state)
            self.bc_complex_base.copy_(
                torch.polar(torch.ones_like(phase_grid), phase_grid).to(torch.complex64)
            )

    def _flat_param_bias(self) -> torch.Tensor:
        zero = cast(torch.Tensor, self._zero_bias)
        return torch.stack(
            (
                self.zeta_bias,
                self.omega_mod_bias,
                zero,
                self.mix_r_bias,
            ),
            dim=-1,
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

        p = params.permute(0, 2, 1, 3).to(torch.float32)
        p = p + self._flat_param_bias().view(1, self.n_heads, 1, self.param_dim)
        zeta_raw, omega_mod_raw, r_raw, mix_r_raw = p.unbind(dim=-1)

        dt = (
            self.dt_min
            + (self.dt_max - self.dt_min)
            * torch.sigmoid(self.dt_bias).view(1, self.n_heads, 1)
        ).expand_as(zeta_raw)
        r_direct_u = torch.sigmoid(r_raw)
        mix_r = torch.sigmoid(mix_r_raw)

        omega_natural = self.omega_min + F.softplus(self.omega_natural_bias).view(
            1, self.n_heads, 1
        )
        omega_scale = torch.exp(self.omega_mod_scale * torch.tanh(omega_mod_raw))
        omega_drive = omega_natural * omega_scale
        zeta = self.zeta_max * torch.sigmoid(zeta_raw)
        gamma = zeta * omega_drive
        under = (1.0 - zeta.square()).clamp_min(self.eps)
        omega_sign = cast(torch.Tensor, self.omega_sign)
        omega = omega_sign.view(1, self.n_heads, 1) * omega_drive * torch.sqrt(under)

        r_struct = self.r_min + (self.r_max - self.r_min) * torch.exp(-gamma * dt)
        theta = principal_angle(omega * dt)
        r_direct = self.r_min + (self.r_max - self.r_min) * r_direct_u

        r = torch.lerp(r_direct, r_struct, mix_r)
        log_r_f = torch.log(r)
        rho = torch.polar(r, theta)
        k_prev, k_curr = _foh_taps_from_normalized(
            dt, log_r_f, theta, rho, eps=self.eps
        )

        M = _pack_complex(rho)
        K = torch.stack([k_prev, k_curr], dim=-2)
        if not include_aux:
            return M, K, None, None, None
        return M, K, dt, r, theta

    def scan_coeffs(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        M, K, _, _, _ = self._compute_coefficients(params, include_aux=False)
        return M, K

    def coefficients(self, params: torch.Tensor) -> SLinOSSScanPrepCoefficients:
        M, K, dt, r, theta = self._compute_coefficients(params, include_aux=True)
        assert dt is not None and r is not None and theta is not None
        return SLinOSSScanPrepCoefficients(M=M, K=K, dt=dt, r=r, theta=theta)

    def _parameterize_scan_bc_rows(self, bc: torch.Tensor) -> torch.Tensor:
        if bc.ndim != 5 or bc.shape[2:] != (
            self.n_heads,
            self.bc_param_rows,
            self.d_state,
        ):
            raise ValueError(
                "bc must be "
                f"(batch, T, heads, {self.bc_param_rows}, d_state). "
                f"Got {tuple(bc.shape)}."
            )
        amp = F.softplus(bc.to(torch.float32))
        complex_rows = amp.to(torch.complex64) * self.bc_complex_base.view(
            1, 1, self.n_heads, 2, self.d_state
        )
        bc_ri = torch.view_as_real(complex_rows).permute(0, 1, 2, 3, 5, 4).contiguous()
        return bc_ri.reshape(
            bc.shape[0],
            bc.shape[1],
            self.n_heads,
            4,
            self.d_state,
        ).to(dtype=bc.dtype)

    def _pack_scan_u(self, value: torch.Tensor, batch: int, T: int) -> torch.Tensor:
        if value.ndim != 3 or value.shape[-1] != self.d_inner:
            raise ValueError(
                f"value must be (batch, T, {self.d_inner}). Got {tuple(value.shape)}."
            )
        return (
            value.view(batch, T, self.n_heads, self.d_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _pack_scan_bc(
        self,
        bc: torch.Tensor,
        batch: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bc.ndim != 5 or bc.shape[2:] != (self.n_heads, 4, self.d_state):
            raise ValueError(
                f"bc must be (batch, T, heads, 4, d_state). Got {tuple(bc.shape)}."
            )
        packed = bc.permute(0, 2, 1, 4, 3).reshape(
            batch, self.n_heads, T, self.d_state, 4
        )
        B = packed[..., :2].reshape(batch, self.n_heads, T, 2 * self.d_state)
        C = packed[..., 2:].reshape(batch, self.n_heads, T, 2 * self.d_state)
        return B, C

    def _scan_coeffs_from_flat_params(
        self,
        params: torch.Tensor,
        batch: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.n_heads * self.param_dim
        if params.ndim != 3 or params.shape[-1] != expected:
            raise ValueError(
                f"params must be (batch, T, {expected}). Got {tuple(params.shape)}."
            )
        return self.scan_coeffs(params.view(batch, T, self.n_heads, self.param_dim))

    def _prepare_inputs_reference(self, inputs: ScanPrepInputs) -> ScanInputs:
        batch, T, _ = map(int, inputs.value.shape)
        U = self._pack_scan_u(inputs.value, batch, T)
        bc_rows = self._parameterize_scan_bc_rows(inputs.bc)
        B, C = self._pack_scan_bc(bc_rows, batch, T)
        M, K = self._scan_coeffs_from_flat_params(inputs.params, batch, T)
        return ScanInputs(U=U, M=M, K=K, B=B, C=C)

    def _prepare_inputs_cute(self, inputs: ScanPrepInputs) -> ScanInputs:
        bc_rows = self._parameterize_scan_bc_rows(inputs.bc)
        return scanprep_cute(
            inputs.value,
            inputs.params,
            bc_rows,
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            omega_min=self.omega_min,
            zeta_max=self.zeta_max,
            r_min=self.r_min,
            r_max=self.r_max,
            eps=self.eps,
            dt_bias=self.dt_bias,
            zeta_bias=self.zeta_bias,
            omega_mod_bias=self.omega_mod_bias,
            omega_natural_bias=self.omega_natural_bias,
            mix_r_bias=self.mix_r_bias,
            omega_sign=cast(torch.Tensor, self.omega_sign),
        )

    def forward(
        self,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
    ) -> ScanInputs:  # type: ignore[override]
        return self.backend(self, ScanPrepInputs(value=value, params=params, bc=bc))


__all__ = [
    "SLinOSSScanPrepCoefficients",
    "SLinOSSScanPrep",
    "build_transition_from_polar",
    "foh_taps_from_polar",
    "principal_angle",
]
