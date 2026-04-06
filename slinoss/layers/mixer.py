"""Reference SLinOSS mixer built around hot-swappable scanprep and scan backends."""

from __future__ import annotations

import math
import os
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from slinoss.ops.cconv1d import cconv1d_cuda, cconv1d_cuda_supported
from slinoss.ops.decode_linear import decode_linear
from slinoss.ops.v2x2ssd.reference import v2x2ssm

from .backend import (
    AutoCConv1dBackend,
    AutoMixerDecodeBackend,
    AutoScanBackend,
    CConv1dBackend,
    CuteScanBackend,
    CuteMixerDecodeBackend,
    MixerDecodeBackend,
    MixerDecodeInputs,
    ScanBackend,
    ScanInputs,
    ScanPrepBackend,
    ScanPrepInputs,
)
from .scanprep import SLinOSSScanPrep
from .state import SLinOSSMixerState, ScanState


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _cute_scan_requires_d_state_multiple_of_8(d_state: int) -> None:
    if int(d_state) % 8 != 0:
        raise ValueError(
            "The current CuTe scan backend requires d_state to be a multiple of 8 "
            f"(got d_state={int(d_state)})."
        )


def _copy_scan_state_(dst: ScanState, src: ScanState) -> None:
    if dst.state is not None and src.state is not None:
        dst.state.copy_(src.state)
    if dst.b_prev is not None and src.b_prev is not None:
        dst.b_prev.copy_(src.b_prev)
    if dst.u_prev is not None and src.u_prev is not None:
        dst.u_prev.copy_(src.u_prev)


def _copy_mixer_state_(dst: SLinOSSMixerState, src: SLinOSSMixerState) -> None:
    if dst.conv is not None and src.conv is not None:
        dst.conv.copy_(src.conv)
    _copy_scan_state_(dst.scan, src.scan)


def _promote_dtypes(*dtypes: torch.dtype) -> torch.dtype:
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = torch.promote_types(result, dtype)
    return result


def _to_dtype_if_needed(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype:
        return x
    return x.to(dtype=dtype)


class _SplitMixerProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        d_inner: int,
        param_proj_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_inner = int(d_inner)
        param_proj_dim = int(param_proj_dim)
        gate_value_end = 2 * d_inner
        params_end = gate_value_end + param_proj_dim
        x_flat = x.reshape(-1, x.shape[-1])
        gate_value = torch.mm(x_flat, weight[:gate_value_end, :].t()).view(
            *x.shape[:-1], gate_value_end
        )
        params = torch.mm(x_flat, weight[gate_value_end:params_end, :].t()).view(
            *x.shape[:-1], param_proj_dim
        )
        bc_flat = torch.mm(x_flat, weight[params_end:, :].t()).view(
            *x.shape[:-1], weight.shape[0] - params_end
        )
        gate, value_raw = torch.split(gate_value, [d_inner, d_inner], dim=-1)
        ctx.save_for_backward(x, weight)
        ctx.d_inner = d_inner
        ctx.param_proj_dim = param_proj_dim
        return gate, value_raw, params, bc_flat

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx,
        grad_gate: torch.Tensor,
        grad_value: torch.Tensor,
        grad_params: torch.Tensor,
        grad_bc_flat: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        x, weight = ctx.saved_tensors
        d_inner = int(ctx.d_inner)
        param_proj_dim = int(ctx.param_proj_dim)
        gate_value_end = 2 * d_inner
        params_end = gate_value_end + param_proj_dim
        grad_x: torch.Tensor | None = None
        grad_weight: torch.Tensor | None = None
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None

        if ctx.needs_input_grad[0]:
            grad_input_dtype = _promote_dtypes(
                weight.dtype,
                grad_gate.dtype,
                grad_value.dtype,
                grad_params.dtype,
                grad_bc_flat.dtype,
            )
            grad_gate_flat = _to_dtype_if_needed(
                grad_gate.reshape(-1, d_inner), grad_input_dtype
            )
            grad_value_flat = _to_dtype_if_needed(
                grad_value.reshape(-1, d_inner), grad_input_dtype
            )
            grad_params_flat = _to_dtype_if_needed(
                grad_params.reshape(-1, param_proj_dim), grad_input_dtype
            )
            grad_bc_flat_2d = _to_dtype_if_needed(
                grad_bc_flat.reshape(-1, weight.shape[0] - params_end),
                grad_input_dtype,
            )
            grad_x_flat = torch.mm(
                grad_bc_flat_2d,
                _to_dtype_if_needed(weight[params_end:, :], grad_input_dtype),
            )
            grad_x_flat.addmm_(
                grad_gate_flat,
                _to_dtype_if_needed(weight[:d_inner, :], grad_input_dtype),
            )
            grad_x_flat.addmm_(
                grad_value_flat,
                _to_dtype_if_needed(
                    weight[d_inner:gate_value_end, :],
                    grad_input_dtype,
                ),
            )
            grad_x_flat.addmm_(
                grad_params_flat,
                _to_dtype_if_needed(
                    weight[gate_value_end:params_end, :],
                    grad_input_dtype,
                ),
            )
            grad_x_flat = _to_dtype_if_needed(grad_x_flat, x.dtype)
            grad_x = grad_x_flat.view_as(x)
        if ctx.needs_input_grad[1]:
            grad_weight_dtype = _promote_dtypes(
                x.dtype,
                grad_gate.dtype,
                grad_value.dtype,
                grad_params.dtype,
                grad_bc_flat.dtype,
            )
            x_flat = _to_dtype_if_needed(x.reshape(-1, x.shape[-1]), grad_weight_dtype)
            grad_gate_flat = _to_dtype_if_needed(
                grad_gate.reshape(-1, d_inner), grad_weight_dtype
            )
            grad_value_flat = _to_dtype_if_needed(
                grad_value.reshape(-1, d_inner), grad_weight_dtype
            )
            grad_params_flat = _to_dtype_if_needed(
                grad_params.reshape(-1, param_proj_dim), grad_weight_dtype
            )
            grad_bc_flat_2d = _to_dtype_if_needed(
                grad_bc_flat.reshape(-1, weight.shape[0] - params_end),
                grad_weight_dtype,
            )
            grad_weight = torch.empty_like(weight, dtype=grad_weight_dtype)
            grad_weight[:d_inner, :] = torch.mm(grad_gate_flat.t(), x_flat)
            grad_weight[d_inner:gate_value_end, :] = torch.mm(
                grad_value_flat.t(), x_flat
            )
            grad_weight[gate_value_end:params_end, :] = torch.mm(
                grad_params_flat.t(), x_flat
            )
            grad_weight[params_end:, :] = torch.mm(grad_bc_flat_2d.t(), x_flat)
            grad_weight = _to_dtype_if_needed(grad_weight, weight.dtype)
        return grad_x, grad_weight, None, None


class _MixerCudaGraphStepEngine:
    """Fixed-shape CUDA graph replay for one-token mixer decode."""

    def __init__(
        self,
        mixer: "SLinOSSMixer",
        state: SLinOSSMixerState,
        *,
        batch_size: int,
    ) -> None:
        self.mixer = mixer
        self.batch_size = int(batch_size)
        self.device = mixer.in_proj.weight.device
        self.dtype = mixer.in_proj.weight.dtype
        self.x_buffer = torch.zeros(
            (self.batch_size, mixer.d_model),
            device=self.device,
            dtype=self.dtype,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.static_y: torch.Tensor | None = None
        self._capture(state)

    @staticmethod
    def supported(
        mixer: "SLinOSSMixer",
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        if not isinstance(
            mixer.decode_backend,
            (AutoMixerDecodeBackend, CuteMixerDecodeBackend),
        ):
            return False
        return mixer._supports_cute_decode(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _capture(self, state: SLinOSSMixerState) -> None:
        snapshot = state.clone()
        stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(device=self.device)
        stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    _copy_mixer_state_(state, snapshot)
                    self.static_y = self.mixer._step_inplace(self.x_buffer, state)
            _copy_mixer_state_(state, snapshot)
            current_stream.wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.static_y = self.mixer._step_inplace(self.x_buffer, state)
        finally:
            current_stream.wait_stream(stream)
            _copy_mixer_state_(state, snapshot)

    def step(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState,
    ) -> tuple[torch.Tensor, SLinOSSMixerState]:
        self.x_buffer.copy_(x)
        self.graph.replay()
        if self.static_y is None:
            raise RuntimeError("Mixer decode graph did not materialize an output.")
        return self.static_y.clone(), state


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
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-3,
        r_min: float = 0.9,
        r_max: float = 0.9999,
        eps: float = 1e-8,
        normalize_bc: bool = True,
        backend: ScanBackend | None = None,
        decode_backend: MixerDecodeBackend | None = None,
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
            eps=eps,
            device=device,
        )
        self.backend = AutoScanBackend() if backend is None else backend
        self.decode_backend = (
            AutoMixerDecodeBackend() if decode_backend is None else decode_backend
        )
        self.cconv_backend = (
            AutoCConv1dBackend() if cconv_backend is None else cconv_backend
        )

        self.param_proj_dim = self.n_heads * self.scanprep.param_dim
        self.bc_proj_dim = self.n_heads * 4 * self.d_state
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + self.param_proj_dim + self.bc_proj_dim,
            bias=False,
            **factory_kwargs,
        )
        self.dw_weight = nn.Parameter(
            torch.empty((self.d_inner, self.d_conv), **factory_kwargs)
        )
        self.dw_bias = nn.Parameter(torch.empty((self.d_inner,), **factory_kwargs))
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
            ),
            scan=ScanState(
                state=torch.zeros(
                    (batch_size, self.n_heads, self.d_head, 2 * self.d_state),
                    device=device,
                    dtype=dtype,
                ),
                b_prev=torch.zeros(
                    (batch_size, self.n_heads, 2 * self.d_state),
                    device=device,
                    dtype=dtype,
                ),
                u_prev=torch.zeros(
                    (batch_size, self.n_heads, self.d_head),
                    device=device,
                    dtype=dtype,
                ),
            ),
        )

    def _make_decode_state_tensor(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            (batch_size, self.n_heads, 2 * self.d_state, self.d_head),
            device=device,
            dtype=dtype,
        ).transpose(-1, -2)

    def _ensure_fast_decode_state_layout(
        self,
        state: SLinOSSMixerState,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if state.scan.state is None:
            state.scan.state = self._make_decode_state_tensor(
                batch_size,
                device=device,
                dtype=dtype,
            )
            return
        if state.scan.state.shape != (
            batch_size,
            self.n_heads,
            self.d_head,
            2 * self.d_state,
        ):
            raise ValueError(
                "scan.state must match "
                f"{(batch_size, self.n_heads, self.d_head, 2 * self.d_state)}. "
                f"Got {tuple(state.scan.state.shape)}."
            )
        if state.scan.state.device != device or state.scan.state.dtype != dtype:
            state.scan.state = state.scan.state.to(device=device, dtype=dtype)
        if state.scan.state.stride()[-2:] == (1, self.d_head):
            return
        fast_state = self._make_decode_state_tensor(
            batch_size,
            device=device,
            dtype=dtype,
        )
        fast_state.copy_(state.scan.state)
        state.scan.state = fast_state

    def init_decode_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SLinOSSMixerState:
        state = self.init_state(batch_size, device=device, dtype=dtype)
        if device is None:
            device = self.in_proj.weight.device
        device_obj = torch.device(device)
        if dtype is None:
            dtype = self.in_proj.weight.dtype
        if self._supports_cute_decode(
            batch_size=batch_size,
            device=device_obj,
            dtype=dtype,
        ) and isinstance(
            self.decode_backend, (AutoMixerDecodeBackend, CuteMixerDecodeBackend)
        ):
            state.scan.state = self._make_decode_state_tensor(
                batch_size,
                device=device_obj,
                dtype=dtype,
            )
        return state

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

    def _project_mixer_branches(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            _SplitMixerProjectionFn.apply(
                x,
                self.in_proj.weight,
                self.d_inner,
                self.param_proj_dim,
            ),
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

        gate, value_raw, params, bc_flat = self._project_mixer_branches(x)
        conv_state_in = None if state is None else state.conv
        conv_out, conv_state = self._apply_causal_depthwise_conv(
            value_raw, conv_state_in
        )
        value = F.silu(conv_out)
        bc = self._reshape_bc(bc_flat, batch, T)

        scan_inputs = self.scanprep(value, params, bc)
        scan_state_in = None if state is None else state.scan
        scan_result = self._run_scan_backend(scan_inputs, scan_state_in, return_state)
        if return_state:
            scan_y, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_y = cast(torch.Tensor, scan_result)
            scan_state = None

        if self.output_norm is None:
            out = self._apply_gate_skip_and_project(
                scan_y,
                scan_inputs.U,
                gate,
                batch,
                T,
            )
        else:
            gated = self._apply_gate_skip(scan_y, scan_inputs.U, gate, batch, T)
            out = self.out_proj(gated)

        if not return_state:
            return out

        next_state = SLinOSSMixerState(
            conv=conv_state, scan=cast(ScanState, scan_state)
        )
        return out, next_state

    def _apply_gate_skip_headspace(
        self,
        scan_y: torch.Tensor,
        scan_u: torch.Tensor,
        gate: torch.Tensor,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        skip = self.skip.view(1, self.n_heads, 1, self.d_head)
        y_fp32 = scan_y.to(torch.float32)
        u_fp32 = scan_u.to(torch.float32)
        gate_fp32 = F.silu(gate.to(torch.float32)).view(
            batch, T, self.n_heads, self.d_head
        )
        gate_fp32 = gate_fp32.permute(0, 2, 1, 3)
        mixed = (y_fp32 + u_fp32 * skip.to(torch.float32)) * gate_fp32
        return mixed.to(dtype=scan_y.dtype)

    def _apply_gate_skip(
        self,
        scan_y: torch.Tensor,
        scan_u: torch.Tensor,
        gate: torch.Tensor,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        y = self._apply_gate_skip_headspace(scan_y, scan_u, gate, batch, T)
        y = y.permute(0, 2, 1, 3).reshape(batch, T, self.d_inner)
        if self.output_norm is not None:
            y = self.output_norm(y)
        return y

    def _project_headspace(self, y: torch.Tensor) -> torch.Tensor:
        weight = self.out_proj.weight.view(self.d_model, self.n_heads, self.d_head)
        return torch.einsum("bhtp,dhp->btd", y, weight)

    def _apply_gate_skip_and_project(
        self,
        scan_y: torch.Tensor,
        scan_u: torch.Tensor,
        gate: torch.Tensor,
        batch: int,
        T: int,
    ) -> torch.Tensor:
        return self._project_headspace(
            self._apply_gate_skip_headspace(scan_y, scan_u, gate, batch, T)
        )

    def _reshape_bc(self, bc: torch.Tensor, batch: int, T: int) -> torch.Tensor:
        expected = (batch, T, self.bc_proj_dim)
        if tuple(map(int, bc.shape)) != expected:
            raise ValueError(f"bc must be {expected}. Got {tuple(bc.shape)}.")
        return bc.view(batch, T, self.n_heads, 4, self.d_state)

    def _apply_causal_depthwise_conv_step(
        self,
        value_raw: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value_raw.ndim != 2 or value_raw.shape[-1] != self.d_inner:
            raise ValueError(
                f"Expected value_raw shape (batch, {self.d_inner}), "
                f"got {tuple(value_raw.shape)}."
            )

        state_len = max(self.d_conv - 1, 0)
        x = value_raw.unsqueeze(1)
        x_t = x.transpose(1, 2)
        if x_t.stride(1) == 1 and (
            self.d_inner % 8 != 0 or x_t.stride(2) % 8 != 0 or x_t.stride(0) % 8 != 0
        ):
            x_t = x_t.contiguous()

        if state_len == 0:
            if cconv1d_cuda_supported(x_t, self.dw_weight, activation="silu"):
                y = cconv1d_cuda(
                    x_t,
                    self.dw_weight,
                    self.dw_bias,
                    activation="silu",
                )
                assert isinstance(y, torch.Tensor)
                return y[..., 0].contiguous(), value_raw.new_empty(
                    (value_raw.shape[0], self.d_inner, 0)
                )

            y_seq, next_state = self._apply_cconv_reference(x, None)
            return F.silu(y_seq[:, 0, :]), next_state

        init = self._validate_conv_state(
            conv_state,
            batch=int(value_raw.shape[0]),
            state_len=state_len,
            device=value_raw.device,
            dtype=value_raw.dtype,
        )
        weight = self.dw_weight
        if (
            x_t.stride(1) != 1
            or x_t.stride(2) % 8 != 0
            or x_t.stride(0) % 8 != 0
            or self.d_inner % 8 != 0
        ):
            y_seq, next_state = self._apply_cconv_reference(x, init.contiguous())
            return F.silu(y_seq[:, 0, :]), next_state

        init = init.transpose(1, 2).contiguous().transpose(1, 2)
        if cconv1d_cuda_supported(
            x_t,
            weight,
            initial_states=init,
            activation="silu",
        ):
            y_with_state = cconv1d_cuda(
                x_t,
                weight,
                self.dw_bias,
                initial_states=init,
                return_final_states=True,
                activation="silu",
            )
            assert isinstance(y_with_state, tuple)
            y_t, next_state = y_with_state
            return y_t[..., 0].contiguous(), next_state.contiguous()

        y_seq, next_state = self._apply_cconv_reference(x, init.contiguous())
        return F.silu(y_seq[:, 0, :]), next_state

    def _run_scan_backend(
        self,
        scan_inputs: ScanInputs,
        scan_state: ScanState | None,
        return_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        use_cute_scan = scan_inputs.U.device.type == "cuda" and isinstance(
            self.backend, (AutoScanBackend, CuteScanBackend)
        )
        if use_cute_scan:
            _cute_scan_requires_d_state_multiple_of_8(self.d_state)
        return self.backend(
            scan_inputs,
            chunk_size=self.chunk_size,
            state=scan_state,
            return_state=return_state,
        )

    def _supports_cute_decode(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        if device.type != "cuda":
            return False
        if dtype not in (torch.float16, torch.bfloat16):
            return False
        if batch_size not in (1, 2, 4, 8, 16):
            return False
        if self.d_head != 64 or self.d_state != 64:
            return False
        if not self.normalize_bc or self.output_norm is not None:
            return False
        return True

    def _decode_step_reference(
        self,
        inputs: MixerDecodeInputs,
        state: ScanState,
    ) -> tuple[torch.Tensor, ScanState]:
        batch = int(inputs.value.shape[0])
        value = inputs.value.reshape(batch, 1, self.d_inner).contiguous()
        params = inputs.params.reshape(batch, 1, self.param_proj_dim).contiguous()
        bc = inputs.bc.reshape(batch, 1, self.n_heads, 4, self.d_state).contiguous()
        gate = inputs.gate.reshape(batch, 1, self.d_inner).contiguous()

        scan_inputs = self.scanprep._prepare_inputs_reference(
            ScanPrepInputs(value=value, params=params, bc=bc)
        )
        compute_dtype = (
            torch.float32
            if value.dtype in (torch.float16, torch.bfloat16)
            else value.dtype
        )
        scan_y, final_state, b_last, u_last = v2x2ssm(
            scan_inputs.U,
            scan_inputs.M,
            scan_inputs.K,
            scan_inputs.B,
            scan_inputs.C,
            initial_states=state.state,
            B_prev=state.b_prev,
            U_prev=state.u_prev,
            compute_dtype=compute_dtype,
            output_dtype=value.dtype,
        )
        gated = self._apply_gate_skip(scan_y, scan_inputs.U, gate, batch, 1)[:, 0, :]
        next_state = ScanState(state=final_state, b_prev=b_last, u_prev=u_last)
        return gated.contiguous(), next_state

    def _decode_step_cute(
        self,
        inputs: MixerDecodeInputs,
        state: ScanState,
        *,
        out_proj_weight: torch.Tensor | None = None,
        projected_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ScanState]:
        from slinoss.ops.v2x2ssd.cute.decode import mixer_decode_step_cute

        gated, final_state, b_last, u_last = mixer_decode_step_cute(
            inputs.value,
            inputs.params,
            inputs.bc,
            inputs.gate,
            inputs.skip,
            initial_states=state.state,
            B_prev=state.b_prev,
            U_prev=state.u_prev,
            dt_min=self.scanprep.dt_min,
            dt_max=self.scanprep.dt_max,
            r_min=self.scanprep.r_min,
            r_max=self.scanprep.r_max,
            eps=self.scanprep.eps,
            dt_bias=self.scanprep.dt_bias,
            gamma_bias=self.scanprep.gamma_bias,
            omega_bias=self.scanprep.omega_bias,
            mix_r_bias=self.scanprep.mix_r_bias,
            b_scale=self.scanprep.b_scale,
            c_scale=self.scanprep.c_scale,
            output_dtype=inputs.value.dtype,
            final_state_out=state.state,
            b_last_out=state.b_prev,
            u_last_out=state.u_prev,
            out_proj_weight=out_proj_weight,
            projected_out=projected_out,
        )
        next_state = ScanState(
            state=final_state,
            b_prev=b_last,
            u_prev=u_last,
        )
        return (projected_out if projected_out is not None else gated), next_state

    def _step_inplace(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState,
    ) -> torch.Tensor:
        batch = int(x.shape[0])
        gate, value_raw, params_flat, bc_flat = self._project_mixer_branches(x)
        value, conv_next = self._apply_causal_depthwise_conv_step(value_raw, state.conv)
        if state.conv is None or tuple(state.conv.shape) != tuple(conv_next.shape):
            state.conv = conv_next
        else:
            state.conv.copy_(conv_next)
        decode_inputs = MixerDecodeInputs(
            value=value.view(batch, self.n_heads, self.d_head).contiguous(),
            params=params_flat.view(
                batch, self.n_heads, self.scanprep.param_dim
            ).contiguous(),
            bc=bc_flat.view(batch, self.n_heads, 4, self.d_state).contiguous(),
            gate=gate.view(batch, self.n_heads, self.d_head).contiguous(),
            skip=self.skip.view(self.n_heads, self.d_head),
        )
        gated: torch.Tensor | None = None
        fused_out: torch.Tensor | None = None
        use_fused_outproj = (
            isinstance(
                self.decode_backend, (AutoMixerDecodeBackend, CuteMixerDecodeBackend)
            )
            and batch == 1
            and os.getenv("SLINOSS_MIXER_DECODE_FUSE_OUTPROJ", "0") == "1"
            and self._supports_cute_decode(
                batch_size=batch,
                device=x.device,
                dtype=x.dtype,
            )
            and self.out_proj.bias is None
        )
        if use_fused_outproj:
            projected_fp32 = torch.empty(
                (batch, self.d_model), device=x.device, dtype=torch.float32
            )
            _, scan_next = self._decode_step_cute(
                decode_inputs,
                state.scan,
                out_proj_weight=self.out_proj.weight,
                projected_out=projected_fp32,
            )
            fused_out = projected_fp32.to(dtype=x.dtype)
        else:
            gated, scan_next = self.decode_backend(self, decode_inputs, state.scan)
        if (
            state.scan.state is None
            or scan_next.state is None
            or tuple(state.scan.state.shape) != tuple(scan_next.state.shape)
        ):
            state.scan.state = scan_next.state
        elif state.scan.state is not scan_next.state:
            state.scan.state.copy_(scan_next.state)
        if (
            state.scan.b_prev is None
            or scan_next.b_prev is None
            or tuple(state.scan.b_prev.shape) != tuple(scan_next.b_prev.shape)
        ):
            state.scan.b_prev = scan_next.b_prev
        elif state.scan.b_prev is not scan_next.b_prev:
            state.scan.b_prev.copy_(scan_next.b_prev)
        if (
            state.scan.u_prev is None
            or scan_next.u_prev is None
            or tuple(state.scan.u_prev.shape) != tuple(scan_next.u_prev.shape)
        ):
            state.scan.u_prev = scan_next.u_prev
        elif state.scan.u_prev is not scan_next.u_prev:
            state.scan.u_prev.copy_(scan_next.u_prev)
        if fused_out is not None:
            return fused_out
        if gated is None:
            raise RuntimeError("CuTe decode path did not produce gated output.")
        return decode_linear(gated, self.out_proj)

    def step_cuda_fast(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState | None = None,
    ) -> tuple[torch.Tensor, SLinOSSMixerState]:
        if x.ndim != 2 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected x shape (batch, {self.d_model}), got {tuple(x.shape)}."
            )
        if torch.is_grad_enabled():
            raise ValueError("step_cuda_fast is inference-only.")
        batch = int(x.shape[0])
        if not isinstance(
            self.decode_backend, (AutoMixerDecodeBackend, CuteMixerDecodeBackend)
        ) or not self._supports_cute_decode(
            batch_size=batch,
            device=x.device,
            dtype=x.dtype,
        ):
            raise ValueError("Current inputs are unsupported for step_cuda_fast.")
        return self.step_inplace(x, state)

    def step_inplace(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState | None = None,
    ) -> tuple[torch.Tensor, SLinOSSMixerState]:
        if torch.is_grad_enabled():
            raise ValueError("step_inplace is inference-only.")

        squeeze = False
        if x.ndim == 2:
            x_t = x
            squeeze = True
        elif x.ndim == 3 and x.shape[1] == 1:
            x_t = x[:, 0, :]
        else:
            raise ValueError(
                f"Expected x shape (batch, d_model) or (batch, 1, {self.d_model}), "
                f"got {tuple(x.shape)}."
            )

        batch = int(x_t.shape[0])
        next_state = (
            self.init_decode_state(batch, device=x_t.device, dtype=x_t.dtype)
            if state is None
            else state
        )
        if _MixerCudaGraphStepEngine.supported(
            self,
            batch_size=batch,
            device=x_t.device,
            dtype=x_t.dtype,
        ):
            self._ensure_fast_decode_state_layout(
                next_state,
                batch_size=batch,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            engine = next_state._engine
            if (
                not isinstance(engine, _MixerCudaGraphStepEngine)
                or engine.mixer is not self
                or engine.batch_size != batch
                or engine.device != x_t.device
                or engine.dtype != x_t.dtype
            ):
                engine = _MixerCudaGraphStepEngine(
                    self,
                    next_state,
                    batch_size=batch,
                )
                next_state._engine = engine
                y_step, next_state = engine.step(x_t, next_state)
            else:
                y_step, next_state = engine.step(x_t, next_state)
        else:
            y_step = self._step_inplace(x_t, next_state)
        if squeeze:
            return y_step, next_state
        return y_step.unsqueeze(1), next_state

    def step(
        self,
        x: torch.Tensor,
        state: SLinOSSMixerState | None = None,
        *,
        inplace: bool | None = None,
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

        if inplace is None:
            inplace = not torch.is_grad_enabled()
        x_t = x[:, 0, :] if x.ndim == 3 else x
        if torch.is_grad_enabled():
            if inplace:
                raise ValueError(
                    "In-place decode is unsupported when gradients are enabled."
                )
            y, next_state = self.forward(x, state=state, return_state=True)
            assert isinstance(next_state, SLinOSSMixerState)
        else:
            if inplace:
                y_step, next_state = self.step_inplace(x_t, state)
            else:
                batch = int(x_t.shape[0])
                next_state = (
                    self.init_decode_state(batch, device=x_t.device, dtype=x_t.dtype)
                    if state is None
                    else state.clone()
                )
                y_step = self._step_inplace(x_t, next_state)
            y = y_step.unsqueeze(1)
        if squeeze:
            return y[:, 0, :], next_state
        return y, next_state


__all__ = ["SLinOSSMixer"]
