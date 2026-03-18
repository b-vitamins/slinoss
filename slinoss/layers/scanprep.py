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

    param_dim: int = 13
    _sigmoid_param_idx = (0, 3, 5, 6, 7, 8)
    _tanh_param_idx = (4, 9, 10, 11, 12)

    def __init__(
        self,
        *,
        n_heads: int,
        d_state: int,
        d_head: int,
        normalize_bc: bool = True,
        backend: ScanPrepBackend | None = None,
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
            0.0 < r_min <= r_max <= 1.0,
            f"Require 0 < r_min <= r_max <= 1. Got {r_min}, {r_max}.",
        )
        _require(
            0.0 < theta_bound <= math.pi,
            f"theta_bound must be in (0, pi]. Got {theta_bound}.",
        )
        _require(k_max > 0.0, f"k_max must be positive. Got {k_max}.")

        self.n_heads = int(n_heads)
        self.d_state = int(d_state)
        self.d_head = int(d_head)
        self.d_inner = int(self.n_heads * self.d_head)
        self.normalize_bc = bool(normalize_bc)
        self.backend = AutoScanPrepBackend() if backend is None else backend

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
        if self.normalize_bc:
            self.b_scale = nn.Parameter(
                torch.ones((self.n_heads, 2, self.d_state), device=device, dtype=fp32)
            )
            self.c_scale = nn.Parameter(
                torch.ones((self.n_heads, 2, self.d_state), device=device, dtype=fp32)
            )
        else:
            self.b_scale = None
            self.c_scale = None

        self.register_buffer(
            "_zero_bias",
            torch.zeros((self.n_heads,), device=device, dtype=fp32),
            persistent=False,
        )
        self.register_buffer(
            "_sigmoid_idx_tensor",
            torch.tensor(self._sigmoid_param_idx, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_tanh_idx_tensor",
            torch.tensor(self._tanh_param_idx, device=device, dtype=torch.long),
            persistent=False,
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
            if self.b_scale is not None:
                self.b_scale.fill_(1.0)
            if self.c_scale is not None:
                self.c_scale.fill_(1.0)

    def _flat_param_bias(self) -> torch.Tensor:
        zero = cast(torch.Tensor, self._zero_bias)
        return torch.stack(
            (
                self.dt_bias,
                self.gamma_bias,
                self.omega_bias,
                zero,
                zero,
                self.mix_r_bias,
                self.mix_theta_bias,
                self.mix_k_prev_bias,
                self.mix_k_curr_bias,
                zero,
                zero,
                zero,
                zero,
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
            _k_prev_re,
            _k_prev_im,
            _k_curr_re,
            _k_curr_im,
        ) = p.unbind(dim=-1)

        sigmoid_idx = cast(torch.Tensor, self._sigmoid_idx_tensor)
        (
            dt_u,
            r_direct_u,
            mix_r,
            mix_theta,
            mix_k_prev,
            mix_k_curr,
        ) = torch.sigmoid(p.index_select(-1, sigmoid_idx)).unbind(dim=-1)
        dt = self.dt_min + (self.dt_max - self.dt_min) * dt_u
        gamma = F.softplus(gamma_raw)
        omega = omega_raw

        r_struct = self.r_min + (self.r_max - self.r_min) * torch.exp(-gamma * dt)
        theta_struct = omega * dt

        tanh_idx = cast(torch.Tensor, self._tanh_idx_tensor)
        tanh_outputs = torch.tanh(p.index_select(-1, tanh_idx))
        theta_direct = self.theta_bound * tanh_outputs[..., 0]
        k_prev_learned = self.k_max * tanh_outputs[..., 1:3]
        k_curr_learned = self.k_max * tanh_outputs[..., 3:5]
        r_direct = self.r_min + (self.r_max - self.r_min) * r_direct_u

        r = torch.lerp(r_direct, r_struct, mix_r)
        theta = principal_angle(torch.lerp(theta_direct, theta_struct, mix_theta))

        log_r_f = torch.log(r)
        rho = torch.polar(r, theta)
        k_prev_struct, k_curr_struct = _foh_taps_from_normalized(
            dt, log_r_f, theta, rho, eps=self.eps
        )
        k_prev = torch.lerp(k_prev_learned, k_prev_struct, mix_k_prev.unsqueeze(-1))
        k_curr = torch.lerp(k_curr_learned, k_curr_struct, mix_k_curr.unsqueeze(-1))

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

    def _normalize_bc(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_f = F.rms_norm(x.to(torch.float32), (self.d_state,), eps=1e-5)
        scaled = x_f.mul(scale.view(1, 1, self.n_heads, x.shape[-2], self.d_state))
        return scaled.to(dtype=x.dtype)

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

    def _normalize_scan_bc_rows(self, bc: torch.Tensor) -> torch.Tensor:
        if self.normalize_bc:
            assert self.b_scale is not None and self.c_scale is not None
            return self._normalize_bc(
                bc, torch.cat((self.b_scale, self.c_scale), dim=1)
            )
        return bc

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
        bc = self._normalize_scan_bc_rows(inputs.bc)
        B, C = self._pack_scan_bc(bc, batch, T)
        M, K = self._scan_coeffs_from_flat_params(inputs.params, batch, T)
        return ScanInputs(U=U, M=M, K=K, B=B, C=C)

    def _prepare_inputs_cute(self, inputs: ScanPrepInputs) -> ScanInputs:
        return scanprep_cute(
            inputs.value,
            inputs.params,
            inputs.bc,
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            normalize_bc=self.normalize_bc,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            r_min=self.r_min,
            r_max=self.r_max,
            theta_bound=self.theta_bound,
            k_max=self.k_max,
            eps=self.eps,
            dt_bias=self.dt_bias,
            gamma_bias=self.gamma_bias,
            omega_bias=self.omega_bias,
            mix_r_bias=self.mix_r_bias,
            mix_theta_bias=self.mix_theta_bias,
            mix_k_prev_bias=self.mix_k_prev_bias,
            mix_k_curr_bias=self.mix_k_curr_bias,
            b_scale=self.b_scale,
            c_scale=self.c_scale,
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
