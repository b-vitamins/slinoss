"""Scan preparation layer definitions for the SLinOSS mixer."""

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.ops.scanprep import (
    SLinOSSScanPrepCoefficients,
    principal_angle,
    scanprep_cute,
)
from slinoss.ops.scanprep.reference import _foh_taps_from_normalized, _pack_complex

from ._validation import _require
from .backend import (
    AutoScanPrepBackend,
    ScanInputs,
    ScanPrepBackend,
    ScanPrepInputs,
)


def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=float(eps), max=1.0 - float(eps))
    return torch.log(p) - torch.log1p(-p)


class SLinOSSScanPrep(nn.Module):
    """Converts mixer value, parameter, and BC streams into scan inputs."""

    param_dim: int = 2
    bc_param_rows: int = 2
    theta_mod_scale: float = 0.25
    theta_sign: torch.Tensor

    def __init__(
        self,
        *,
        n_heads: int,
        d_state: int,
        d_head: int,
        backend: ScanPrepBackend | None = None,
        dt_min: float = 3e-2,
        dt_max: float = 1e-1,
        dt_init_floor: float = 3e-2,
        gamma_min: float = 2.0,
        gamma_max: float = 8.0,
        theta_init_min: float = 0.2,
        theta_init_max: float = 1.0,
        r_min: float = 0.9,
        r_max: float = 1.0,
        bc_gain_max: float = 2.0,
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
        _require(
            0.0 < gamma_min < gamma_max,
            f"Require 0 < gamma_min < gamma_max. Got {gamma_min}, {gamma_max}.",
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
        _require(bc_gain_max > 0.0, f"Require bc_gain_max > 0. Got {bc_gain_max}.")

        self.n_heads = int(n_heads)
        self.d_state = int(d_state)
        self.d_head = int(d_head)
        self.d_inner = int(self.n_heads * self.d_head)
        self.backend = AutoScanPrepBackend() if backend is None else backend

        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.theta_init_min = float(theta_init_min)
        self.theta_init_max = float(theta_init_max)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.bc_gain_max = float(bc_gain_max)
        self.eps = float(eps)

        fp32 = torch.float32
        self.dt_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.gamma_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.theta_mod_bias = nn.Parameter(
            torch.empty((self.n_heads,), device=device, dtype=fp32)
        )
        self.theta_bias = nn.Parameter(
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
            "theta_sign",
            torch.where(
                torch.arange(self.n_heads, device=device) % 2 == 0,
                torch.ones((self.n_heads,), device=device, dtype=fp32),
                -torch.ones((self.n_heads,), device=device, dtype=fp32),
            ),
            persistent=True,
        )
        self._theta_span = float(self.theta_init_max - self.theta_init_min)
        self.reset_parameters()

    def _head_init_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns per-head ``(dt, theta)`` initialization targets."""

        dt_lo = max(self.dt_min, self.dt_init_floor)
        dt_hi = self.dt_max
        _require(dt_hi > dt_lo > 0.0, f"Bad dt init bounds: {dt_lo}, {dt_hi}.")

        n_dt = max(1, int(math.floor(math.sqrt(self.n_heads))))
        n_theta = math.ceil(self.n_heads / n_dt)

        dt_grid = torch.logspace(
            math.log10(dt_lo),
            math.log10(dt_hi),
            n_dt,
            device=self.dt_bias.device,
            dtype=torch.float32,
        )
        theta_grid = torch.logspace(
            math.log10(self.theta_init_min),
            math.log10(self.theta_init_max),
            n_theta,
            device=self.dt_bias.device,
            dtype=torch.float32,
        )

        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for theta_idx, theta_val in enumerate(theta_grid):
            dt_order = range(n_dt) if theta_idx % 2 == 0 else range(n_dt - 1, -1, -1)
            for dt_idx in dt_order:
                pairs.append((dt_grid[dt_idx], theta_val))
        dt0 = torch.stack([pair[0] for pair in pairs[: self.n_heads]])
        theta0 = torch.stack([pair[1] for pair in pairs[: self.n_heads]])
        return dt0, theta0

    def _make_initial_phase_grid(self) -> torch.Tensor:
        return torch.linspace(
            -math.pi,
            math.pi,
            self.n_heads * 2 * self.d_state + 1,
            device=self.dt_bias.device,
            dtype=torch.float32,
        )[:-1].view(self.n_heads, 2, self.d_state)

    def reset_parameters(self) -> None:
        dt0, theta0 = self._head_init_targets()
        dt_u0 = (dt0 - self.dt_min) / (self.dt_max - self.dt_min)
        theta_u0 = (theta0 - self.theta_init_min) / max(self._theta_span, 1.0e-6)
        gamma0 = torch.full_like(theta0, 0.5 * (self.gamma_min + self.gamma_max))

        with torch.no_grad():
            self.dt_bias.copy_(_logit(dt_u0))
            self.gamma_bias.copy_(
                _logit((gamma0 - self.gamma_min) / (self.gamma_max - self.gamma_min))
            )
            self.theta_mod_bias.zero_()
            self.theta_bias.copy_(_logit(theta_u0))
            phase_grid = self._make_initial_phase_grid()
            self.bc_complex_base.copy_(
                torch.polar(torch.ones_like(phase_grid), phase_grid).to(torch.complex64)
            )

    def _apply(self, fn):  # type: ignore[override]
        bc_complex_base = self._parameters.pop("bc_complex_base", None)
        super()._apply(fn)
        if bc_complex_base is None:
            return self

        probe = fn(torch.empty((), device=bc_complex_base.device, dtype=torch.float32))
        restored = nn.Parameter(
            bc_complex_base.detach().to(
                device=probe.device,
                dtype=bc_complex_base.dtype,
            ),
            requires_grad=bc_complex_base.requires_grad,
        )
        if bc_complex_base.grad is not None:
            restored.grad = bc_complex_base.grad.detach().to(
                device=probe.device,
                dtype=bc_complex_base.dtype,
            )
        self._parameters["bc_complex_base"] = restored
        return self

    def coefficients(self, params: torch.Tensor) -> SLinOSSScanPrepCoefficients:
        M, K, dt, r, theta = self._make_scan_coefficients(params, include_aux=True)
        assert dt is not None and r is not None and theta is not None
        return SLinOSSScanPrepCoefficients(M=M, K=K, dt=dt, r=r, theta=theta)

    def _make_param_bias(self) -> torch.Tensor:
        return torch.stack((self.gamma_bias, self.theta_mod_bias), dim=-1)

    def _make_scan_coefficients(
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
        self._validate_coeff_params(params)

        p = params.permute(0, 2, 1, 3).to(torch.float32)
        p = p + self._make_param_bias().view(1, self.n_heads, 1, self.param_dim)
        gamma_raw, theta_mod_raw = p.unbind(dim=-1)

        dt = (
            self.dt_min
            + (self.dt_max - self.dt_min)
            * torch.sigmoid(self.dt_bias).view(1, self.n_heads, 1)
        ).expand_as(gamma_raw)
        theta_u = torch.sigmoid(
            self.theta_bias.view(1, self.n_heads, 1)
            + self.theta_mod_scale * torch.tanh(theta_mod_raw)
        )
        theta_drive = self.theta_init_min + self._theta_span * theta_u
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(
            gamma_raw
        )
        theta_sign = cast(torch.Tensor, self.theta_sign)
        theta = theta_sign.view(1, self.n_heads, 1) * theta_drive

        r = self.r_min + (self.r_max - self.r_min) * torch.exp(-(gamma * dt))
        log_r_f = torch.log(r)
        theta = principal_angle(theta)
        rho = torch.polar(r, theta)
        k_prev, k_curr = _foh_taps_from_normalized(
            dt,
            log_r_f,
            theta,
            rho,
            eps=self.eps,
        )

        M = _pack_complex(rho)
        K = torch.stack((k_prev, k_curr), dim=-2)
        if not include_aux:
            return M, K, None, None, None
        return M, K, dt, r, theta

    def _validate_coeff_params(self, params: torch.Tensor) -> None:
        expected = (self.n_heads, self.param_dim)
        _require(
            params.ndim == 4 and tuple(params.shape[-2:]) == expected,
            f"Expected params shape (batch, T, {self.n_heads}, {self.param_dim}), "
            f"got {tuple(params.shape)}.",
        )

    def _validate_scan_bc(self, bc: torch.Tensor) -> None:
        expected = (self.n_heads, self.bc_param_rows, self.d_state)
        _require(
            bc.ndim == 5 and tuple(bc.shape[2:]) == expected,
            "bc must be "
            f"(batch, T, heads, {self.bc_param_rows}, d_state). "
            f"Got {tuple(bc.shape)}.",
        )

    def _normalize_scan_bc_pairs(self, bc_pairs: torch.Tensor) -> torch.Tensor:
        mag_sq = bc_pairs.square().sum(dim=-1, dtype=torch.float32)
        row_rms = mag_sq.mean(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        row_rms_expanded = row_rms.unsqueeze(-1).to(dtype=bc_pairs.dtype)
        direction = bc_pairs / row_rms_expanded
        bounded_gain = self.bc_gain_max * torch.tanh(row_rms / self.bc_gain_max)
        return direction * bounded_gain.unsqueeze(-1).to(dtype=bc_pairs.dtype)

    def _parameterize_scan_bc_pairs(
        self,
        bc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_scan_bc(bc)

        bc_amplitude = F.softplus(bc)
        bc_complex_base = torch.view_as_real(self.bc_complex_base).to(
            device=bc.device,
            dtype=bc.dtype,
        )
        b_pairs = bc_amplitude[..., 0, :].unsqueeze(-1) * bc_complex_base[
            :, 0, :, :
        ].view(1, 1, self.n_heads, self.d_state, 2)
        c_pairs = bc_amplitude[..., 1, :].unsqueeze(-1) * bc_complex_base[
            :, 1, :, :
        ].view(1, 1, self.n_heads, self.d_state, 2)
        return (
            self._normalize_scan_bc_pairs(b_pairs).contiguous(),
            self._normalize_scan_bc_pairs(c_pairs).contiguous(),
        )

    def _parameterize_scan_bc_rows(self, bc: torch.Tensor) -> torch.Tensor:
        b_pairs, c_pairs = self._parameterize_scan_bc_pairs(bc)
        return torch.stack(
            (b_pairs[..., 0], b_pairs[..., 1], c_pairs[..., 0], c_pairs[..., 1]),
            dim=3,
        )

    def _pack_scan_u(
        self,
        value: torch.Tensor,
        *,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        _require(
            value.ndim == 3 and value.shape[-1] == self.d_inner,
            f"value must be (batch, T, {self.d_inner}). Got {tuple(value.shape)}.",
        )
        return (
            value.view(batch, T, self.n_heads, self.d_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def _pack_scan_bc(
        self,
        B_pairs: torch.Tensor,
        C_pairs: torch.Tensor,
        *,
        batch: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (batch, T, self.n_heads, self.d_state, 2)
        _require(
            tuple(map(int, B_pairs.shape)) == expected,
            f"B_pairs must be {expected}. Got {tuple(B_pairs.shape)}.",
        )
        _require(
            tuple(map(int, C_pairs.shape)) == expected,
            f"C_pairs must be {expected}. Got {tuple(C_pairs.shape)}.",
        )
        B = (
            B_pairs.permute(0, 2, 1, 3, 4)
            .reshape(batch, self.n_heads, T, 2 * self.d_state)
            .contiguous()
        )
        C = (
            C_pairs.permute(0, 2, 1, 3, 4)
            .reshape(batch, self.n_heads, T, 2 * self.d_state)
            .contiguous()
        )
        return B, C

    def _make_scan_coefficients_from_flat_params(
        self,
        params: torch.Tensor,
        *,
        batch: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.n_heads * self.param_dim
        _require(
            params.ndim == 3 and params.shape[-1] == expected,
            f"params must be (batch, T, {expected}). Got {tuple(params.shape)}.",
        )
        M, K, _, _, _ = self._make_scan_coefficients(
            params.view(batch, T, self.n_heads, self.param_dim),
            include_aux=False,
        )
        return M, K

    def _prepare_inputs_reference(self, inputs: ScanPrepInputs) -> ScanInputs:
        batch, T, _ = map(int, inputs.value.shape)
        U = self._pack_scan_u(inputs.value, batch=batch, T=T)
        b_pairs, c_pairs = self._parameterize_scan_bc_pairs(inputs.bc)
        B, C = self._pack_scan_bc(b_pairs, c_pairs, batch=batch, T=T)
        M, K = self._make_scan_coefficients_from_flat_params(
            inputs.params,
            batch=batch,
            T=T,
        )
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
            theta_init_min=self.theta_init_min,
            theta_init_max=self.theta_init_max,
            gamma_min=self.gamma_min,
            gamma_max=self.gamma_max,
            r_min=self.r_min,
            r_max=self.r_max,
            eps=self.eps,
            dt_bias=self.dt_bias,
            gamma_bias=self.gamma_bias,
            theta_mod_bias=self.theta_mod_bias,
            theta_bias=self.theta_bias,
            theta_sign=cast(torch.Tensor, self.theta_sign),
        )

    def forward(
        self,
        value: torch.Tensor,
        params: torch.Tensor,
        bc: torch.Tensor,
    ) -> ScanInputs:  # type: ignore[override]
        return self.backend(self, ScanPrepInputs(value=value, params=params, bc=bc))


__all__ = ["SLinOSSScanPrep"]
