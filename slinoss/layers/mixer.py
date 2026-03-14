"""Reference SLinOSS mixer built around hot-swappable scanprep and scan backends."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.ops.cconv1d import cconv1d_cuda, cconv1d_cuda_supported

from .backend import (
    AutoCConv1dBackend,
    AutoScanBackend,
    CConv1dBackend,
    ScanBackend,
    ScanInputs,
    ScanPrepBackend,
)
from .scanprep import SLinOSSScanPrep
from .state import SLinOSSMixerState, ScanState


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


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
        scanprep_backend: ScanPrepBackend | None = None,
        cconv_backend: CConv1dBackend | None = None,
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
        self.scanprep = SLinOSSScanPrep(
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            normalize_bc=normalize_bc,
            backend=scanprep_backend,
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
        self.cconv_backend = (
            AutoCConv1dBackend() if cconv_backend is None else cconv_backend
        )

        param_dim = self.n_heads * self.scanprep.param_dim
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + param_dim,
            bias=False,
            **factory_kwargs,
        )
        self.dw_weight = nn.Parameter(
            torch.empty((self.d_inner, self.d_conv), **factory_kwargs)
        )
        self.dw_bias = nn.Parameter(torch.empty((self.d_inner,), **factory_kwargs))
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
            self.output_norm: nn.Module | None = None
        else:
            self.output_norm = nn.RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.skip.fill_(1.0)
        nn.init.kaiming_uniform_(
            self.dw_weight.view(self.d_inner, 1, self.d_conv), a=math.sqrt(5.0)
        )
        bound = 1.0 / math.sqrt(float(self.d_conv))
        nn.init.uniform_(self.dw_bias, -bound, bound)
        self.scanprep.reset_parameters()

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

    def _apply_causal_depthwise_conv(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cconv_backend(self, x, conv_state)

    def _dw_weight_3d(self) -> torch.Tensor:
        return self.dw_weight.unsqueeze(1)

    def _validate_conv_state(
        self,
        conv_state: torch.Tensor | None,
        *,
        batch: int,
        state_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if conv_state is None:
            return torch.zeros(
                (batch, self.d_inner, state_len), device=device, dtype=dtype
            )
        if conv_state.shape != (batch, self.d_inner, state_len):
            raise ValueError(
                f"conv_state must be {(batch, self.d_inner, state_len)}. "
                f"Got {tuple(conv_state.shape)}."
            )
        return conv_state.to(device=device, dtype=dtype)

    def _next_conv_state_from_input(
        self,
        x_t: torch.Tensor,
        *,
        state_len: int,
    ) -> torch.Tensor:
        if state_len == 0:
            return x_t.new_empty((x_t.shape[0], x_t.shape[1], 0))
        if x_t.shape[-1] >= state_len:
            return x_t[..., -state_len:].contiguous()
        prefix = x_t.new_zeros((x_t.shape[0], x_t.shape[1], state_len - x_t.shape[-1]))
        return torch.cat((prefix, x_t), dim=-1).contiguous()

    def _apply_cconv_reference(
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
        weight = self._dw_weight_3d()
        if state_len == 0:
            y = F.conv1d(
                x.transpose(1, 2).contiguous(),
                weight,
                self.dw_bias,
                groups=self.d_inner,
            )
            empty = x.new_empty((batch, self.d_inner, 0))
            return y.transpose(1, 2).contiguous(), empty

        prefix = self._validate_conv_state(
            conv_state,
            batch=batch,
            state_len=state_len,
            device=x.device,
            dtype=x.dtype,
        )
        x_t = x.transpose(1, 2).contiguous()
        cat = torch.cat([prefix, x_t], dim=-1)
        y = F.conv1d(cat, weight, self.dw_bias, groups=self.d_inner)
        next_state = cat[..., -state_len:].contiguous()
        return y.transpose(1, 2).contiguous(), next_state

    def _apply_cconv_cuda(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = map(int, x.shape)
        if channels != self.d_inner:
            raise ValueError(
                f"Expected conv input width {self.d_inner}, got {channels}."
            )

        state_len = max(self.d_conv - 1, 0)
        x_t = x.transpose(1, 2)
        weight = self.dw_weight
        if x_t.stride(1) == 1 and (
            self.d_inner % 8 != 0 or x_t.stride(2) % 8 != 0 or x_t.stride(0) % 8 != 0
        ):
            x_t = x_t.contiguous()

        if state_len == 0:
            if not cconv1d_cuda_supported(x_t, weight, activation=None):
                return self._apply_cconv_reference(x, conv_state)
            y = cconv1d_cuda(x_t, weight, self.dw_bias, activation=None)
            assert isinstance(y, torch.Tensor)
            return y.transpose(1, 2).contiguous(), x.new_empty((batch, self.d_inner, 0))

        if conv_state is None:
            if not cconv1d_cuda_supported(x_t, weight, activation=None):
                return self._apply_cconv_reference(x, conv_state)
            y = cconv1d_cuda(x_t, weight, self.dw_bias, activation=None)
            assert isinstance(y, torch.Tensor)
            next_state = self._next_conv_state_from_input(x_t, state_len=state_len)
            return y.transpose(1, 2).contiguous(), next_state

        if (
            x_t.stride(1) != 1
            or x_t.stride(2) % 8 != 0
            or x_t.stride(0) % 8 != 0
            or self.d_inner % 8 != 0
        ):
            return self._apply_cconv_reference(x, conv_state)

        init = self._validate_conv_state(
            conv_state,
            batch=batch,
            state_len=state_len,
            device=x.device,
            dtype=x.dtype,
        )
        init = init.transpose(1, 2).contiguous().transpose(1, 2)
        if not cconv1d_cuda_supported(
            x_t,
            weight,
            initial_states=init,
            activation=None,
        ):
            return self._apply_cconv_reference(x, conv_state)
        y_with_state = cconv1d_cuda(
            x_t,
            weight,
            self.dw_bias,
            initial_states=init,
            return_final_states=True,
            activation=None,
        )
        assert isinstance(y_with_state, tuple)
        y_t, next_state = y_with_state
        return y_t.transpose(1, 2).contiguous(), next_state.contiguous()

    def _build_scan_inputs(
        self,
        *,
        value: torch.Tensor,
        params: torch.Tensor,
    ) -> ScanInputs:
        batch, T, _ = map(int, value.shape)
        bc = self._project_bc(value, batch, T)
        return self.scanprep(value, params, bc)

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

        proj = self.in_proj(x)
        gate, value_raw, params = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.n_heads * self.scanprep.param_dim],
            dim=-1,
        )
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = self._apply_causal_depthwise_conv_with_state(
            value_raw, conv_state_in
        )
        value = F.silu(conv_out)

        scan_inputs = self._build_scan_inputs(value=value, params=params)
        scan_state_in = None if state is None else state.scan
        scan_result = self._run_scan_backend(scan_inputs, scan_state_in, return_state)
        if return_state:
            scan_y, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_y = cast(torch.Tensor, scan_result)
            scan_state = None

        gated = self._apply_gate_skip(scan_y, scan_inputs.U, gate, batch, T)
        out = self.out_proj(gated)

        if not return_state:
            return out

        next_state = SLinOSSMixerState(
            conv=conv_state, scan=cast(ScanState, scan_state)
        )
        return out, next_state

    def _apply_gate_skip(
        self,
        scan_y: torch.Tensor,
        scan_u: torch.Tensor,
        gate: torch.Tensor,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        skip = self.skip.to(dtype=scan_u.dtype).view(1, self.n_heads, 1, self.d_head)
        y = scan_y + scan_u * skip
        y = y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner)
        y = y * F.silu(gate).to(dtype=y.dtype)
        if self.output_norm is not None:
            y = self.output_norm(y)
        return y

    def _project_bc(self, value: torch.Tensor, batch: int, T: int) -> torch.Tensor:
        return self.bc_proj(value).view(batch, T, self.n_heads, 4, self.d_state)

    def _apply_causal_depthwise_conv_with_state(
        self,
        value_raw: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_causal_depthwise_conv(value_raw, conv_state)

    def _run_scan_backend(
        self,
        scan_inputs: ScanInputs,
        scan_state: ScanState | None,
        return_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        return self.backend(
            scan_inputs,
            chunk_size=self.chunk_size,
            state=scan_state,
            return_state=return_state,
        )

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
