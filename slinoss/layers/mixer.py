"""Reference SLinOSS mixer built around a hot-swappable scan backend."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.perf import call_region

from .backend import AutoScanBackend, ScanBackend, ScanInputs
from .discretization import SLinOSSDiscretizer
from .state import SLinOSSMixerState, ScanState


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _pack_interleaved_pairs(x: torch.Tensor) -> torch.Tensor:
    """Packs semantic ``(..., 2, N)`` pairs to canonical interleaved ``(..., 2N)``."""
    if x.ndim != 5 or x.shape[-2] != 2:
        raise ValueError(
            "Expected semantic pair tensor with shape (batch, T, heads, 2, N). "
            f"Got {tuple(x.shape)}."
        )
    batch, T, heads, _, N = map(int, x.shape)
    return x.permute(0, 2, 1, 4, 3).reshape(batch, heads, T, 2 * N).contiguous()


class SLinOSSMixer(nn.Module):
    """Selective linear oscillatory mixer with a backend-agnostic scan surface."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        expand: int = 2,
        d_head: int = 64,
        d_conv: int = 4,
        chunk_size: int = 64,
        dt_min: float = 1e-4,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        r_min: float = 0.9,
        r_max: float = 1.0,
        theta_bound: float = math.pi,
        k_max: float = 0.5,
        eps: float = 1e-8,
        normalize_bc: bool = True,
        backend: ScanBackend | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require(d_model > 0, f"d_model must be positive. Got {d_model}.")
        _require(d_state > 0, f"d_state must be positive. Got {d_state}.")
        _require(expand > 0, f"expand must be positive. Got {expand}.")
        _require(d_head > 0, f"d_head must be positive. Got {d_head}.")
        _require(d_conv > 0, f"d_conv must be positive. Got {d_conv}.")
        _require(chunk_size > 0, f"chunk_size must be positive. Got {chunk_size}.")

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = int(expand)
        self.d_head = int(d_head)
        self.d_conv = int(d_conv)
        self.chunk_size = int(chunk_size)
        self.normalize_bc = bool(normalize_bc)

        self.d_inner = int(self.expand * self.d_model)
        _require(
            self.d_inner % self.d_head == 0,
            f"expand * d_model = {self.d_inner} must be divisible by d_head = {self.d_head}.",
        )
        self.n_heads = int(self.d_inner // self.d_head)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.discretizer = SLinOSSDiscretizer(
            n_heads=self.n_heads,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            r_min=r_min,
            r_max=r_max,
            theta_bound=theta_bound,
            k_max=k_max,
            eps=eps,
            device=device,
        )
        self.backend = AutoScanBackend() if backend is None else backend

        param_dim = self.n_heads * self.discretizer.param_dim
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + param_dim,
            bias=False,
            **factory_kwargs,
        )
        self.dw_conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
            bias=True,
            **factory_kwargs,
        )
        self.bc_proj = nn.Linear(
            self.d_inner,
            self.n_heads * 4 * self.d_state,
            bias=False,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=False, **factory_kwargs
        )

        # The manuscript writes D ⊙ U_t, so the skip lives at the scan-channel level.
        self.skip = nn.Parameter(
            torch.ones((self.d_inner,), device=device, dtype=torch.float32)
        )
        if self.normalize_bc:
            self.b_scale = nn.Parameter(
                torch.ones(
                    (self.n_heads, 2, self.d_state), device=device, dtype=torch.float32
                )
            )
            self.c_scale = nn.Parameter(
                torch.ones(
                    (self.n_heads, 2, self.d_state), device=device, dtype=torch.float32
                )
            )
            self.output_norm: nn.Module | None = None
        else:
            self.b_scale = None
            self.c_scale = None
            self.output_norm = nn.RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.skip.fill_(1.0)
            if self.b_scale is not None:
                self.b_scale.fill_(1.0)
            if self.c_scale is not None:
                self.c_scale.fill_(1.0)
        self.discretizer.reset_parameters()

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSMixerState:
        _require(batch_size > 0, f"batch_size must be positive. Got {batch_size}.")
        if device is None:
            device = self.in_proj.weight.device
        if dtype is None:
            dtype = self.in_proj.weight.dtype
        return SLinOSSMixerState(
            conv=torch.zeros(
                (batch_size, self.d_inner, max(self.d_conv - 1, 0)),
                device=device,
                dtype=dtype,
            )
        )

    def _normalize_bc(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_f = F.rms_norm(x.to(torch.float32), (self.d_state,), eps=1e-5)
        scaled = x_f * scale.view(1, 1, self.n_heads, x.shape[-2], self.d_state)
        return scaled.to(dtype=x.dtype).contiguous()

    def _apply_causal_depthwise_conv(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, T, channels = map(int, x.shape)
        if channels != self.d_inner:
            raise ValueError(
                f"Expected conv input width {self.d_inner}, got {channels}."
            )

        state_len = max(self.d_conv - 1, 0)
        if state_len == 0:
            y = self.dw_conv(x.transpose(1, 2).contiguous())
            empty = x.new_empty((batch, self.d_inner, 0))
            return y.transpose(1, 2).contiguous(), empty

        if conv_state is None:
            prefix = torch.zeros(
                (batch, self.d_inner, state_len), device=x.device, dtype=x.dtype
            )
        else:
            if conv_state.shape != (batch, self.d_inner, state_len):
                raise ValueError(
                    f"conv_state must be {(batch, self.d_inner, state_len)}. "
                    f"Got {tuple(conv_state.shape)}."
                )
            prefix = conv_state.to(device=x.device, dtype=x.dtype)

        x_t = x.transpose(1, 2).contiguous()
        cat = torch.cat([prefix, x_t], dim=-1)
        y = self.dw_conv(cat)
        next_state = cat[..., -state_len:].contiguous()
        return y.transpose(1, 2).contiguous(), next_state

    def _build_scan_inputs(
        self,
        *,
        value: torch.Tensor,
        params: torch.Tensor,
    ) -> ScanInputs:
        batch, T, _ = map(int, value.shape)
        bc = call_region(
            "mixer.bc_proj",
            lambda value_: self.bc_proj(value_).view(
                batch, T, self.n_heads, 4, self.d_state
            ),
            value,
            capture_backward=False,
        )
        B_sem = bc[..., :2, :]
        C_sem = bc[..., 2:, :]
        if self.normalize_bc:
            assert self.b_scale is not None and self.c_scale is not None
            b_scale = self.b_scale
            c_scale = self.c_scale
            B_sem, C_sem = call_region(
                "mixer.bc_norm",
                lambda B_sem_, C_sem_: (
                    self._normalize_bc(B_sem_, b_scale),
                    self._normalize_bc(C_sem_, c_scale),
                ),
                B_sem,
                C_sem,
            )

        coeffs = call_region(
            "mixer.discretizer",
            lambda params_: self.discretizer.scan_coeffs(
                params_.view(batch, T, self.n_heads, -1)
            ),
            params,
        )
        return call_region(
            "mixer.scan_input_pack",
            lambda value_, coeffs_, B_sem_, C_sem_: ScanInputs(
                U=value_.view(batch, T, self.n_heads, self.d_head)
                .permute(0, 2, 1, 3)
                .contiguous(),
                M=coeffs_[0],
                K=coeffs_[1],
                B=_pack_interleaved_pairs(B_sem_),
                C=_pack_interleaved_pairs(C_sem_),
            ),
            value,
            coeffs,
            B_sem,
            C_sem,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSMixerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSMixerState]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}."
            )

        batch, T, _ = map(int, x.shape)
        if T == 0:
            empty = x.new_empty((batch, 0, self.d_model))
            next_state = SLinOSSMixerState() if state is None else state
            return (empty, next_state) if return_state else empty

        proj = call_region("mixer.in_proj", self.in_proj, x, capture_backward=False)
        gate, value_raw, params = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.n_heads * self.discretizer.param_dim],
            dim=-1,
        )
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = call_region(
            "mixer.dw_conv",
            lambda value_raw_: self._apply_causal_depthwise_conv(
                value_raw_, conv_state_in
            ),
            value_raw,
            capture_backward=False,
        )
        value = call_region("mixer.dw_conv_activation", F.silu, conv_out)

        scan_inputs = self._build_scan_inputs(value=value, params=params)
        scan_state_in = None if state is None else state.scan
        scan_result = call_region(
            "v2x2ssd.total",
            lambda scan_inputs_: self.backend(
                scan_inputs_,
                chunk_size=self.chunk_size,
                state=scan_state_in,
                return_state=return_state,
            ),
            scan_inputs,
        )
        if return_state:
            scan_y, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_y = cast(torch.Tensor, scan_result)
            scan_state = None

        gated = call_region(
            "mixer.gate_skip",
            self._apply_gate_skip,
            scan_y,
            value,
            gate,
            batch,
            T,
        )
        out = call_region(
            "mixer.out_proj",
            self.out_proj,
            gated,
            capture_backward=False,
        )

        if not return_state:
            return out

        next_state = SLinOSSMixerState(
            conv=conv_state, scan=cast(ScanState, scan_state)
        )
        return out, next_state

    def _apply_gate_skip(
        self,
        scan_y: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        scan_y = scan_y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner).contiguous()
        skip = self.skip.to(dtype=value.dtype).view(1, 1, self.d_inner)
        y = (scan_y + value * skip) * F.silu(gate).to(dtype=value.dtype)
        if self.output_norm is not None:
            y = self.output_norm(y)
        return y

    def step(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState | None = None,
    ) -> tuple[torch.Tensor, SLinOSSMixerState]:
        squeeze = False
        if x.ndim == 2:
            x = x.unsqueeze(1)
            squeeze = True
        elif x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(
                f"Expected x shape (batch, d_model) or (batch, 1, {self.d_model}), "
                f"got {tuple(x.shape)}."
            )

        y, next_state = self.forward(x, state=state, return_state=True)
        assert isinstance(next_state, SLinOSSMixerState)
        if squeeze:
            return y[:, 0, :], next_state
        return y, next_state


__all__ = ["SLinOSSMixer"]
