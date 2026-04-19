"""Mixer layer definitions for the SLinOSS model."""

import math
from typing import cast

import torch
from torch import nn

from slinoss.ops.mixer import mixer_tail, split_mixer_projection
from slinoss.ops.mixer.step import (
    MixerCudaGraphStepEngine as _MixerCudaGraphStepEngine,
    ensure_fast_decode_state_layout,
    make_decode_state_tensor,
    run_inplace_decode_step,
    supports_cute_decode,
)
from slinoss.layers.norm import RMSNorm

from .backend import (
    AutoCConv1dBackend,
    AutoMixerDecodeBackend,
    AutoScanBackend,
    CConv1dBackend,
    CuteMixerDecodeBackend,
    CuteScanBackend,
    MixerDecodeBackend,
    ScanBackend,
    ScanPrepBackend,
)
from ._validation import _require
from .scanprep import SLinOSSScanPrep
from .state import SLinOSSMixerState, ScanState


def _is_cuda_graph_capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


class SLinOSSMixer(nn.Module):
    """Mixer module with configurable scan, scanprep, conv, and decode backends."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        expand: float = 2,
        d_head: int = 64,
        d_conv: int = 4,
        chunk_size: int = 64,
        bc_groups: int | None = None,
        scanprep_backend: ScanPrepBackend | None = None,
        cconv_backend: CConv1dBackend | None = None,
        dt_min: float = 3e-2,
        dt_max: float = 1e-1,
        dt_init_floor: float = 3e-2,
        gamma_min: float = 2.0,
        gamma_max: float = 8.0,
        theta_init_min: float = 0.2,
        theta_init_max: float = 1.0,
        r_min: float = 0.9,
        r_max: float = 1.0,
        eps: float = 1e-8,
        scan_backend: ScanBackend | None = None,
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
        if bc_groups is not None:
            _require(
                bc_groups > 0,
                f"bc_groups must be positive when provided. Got {bc_groups}.",
            )

        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = float(expand)
        self.d_head = int(d_head)
        self.d_conv = int(d_conv)
        self.chunk_size = int(chunk_size)

        self.d_inner = self._quantize_inner_dim(
            self.expand * self.d_model,
            self.d_head,
        )
        _require(
            self.d_inner % self.d_head == 0,
            f"expand * d_model = {self.d_inner} must be divisible by d_head = {self.d_head}.",
        )
        self.n_heads = int(self.d_inner // self.d_head)
        self.bc_groups = self.n_heads if bc_groups is None else int(bc_groups)
        _require(
            self.bc_groups <= self.n_heads,
            "bc_groups must not exceed the realized head count. "
            f"Got bc_groups={self.bc_groups}, n_heads={self.n_heads}.",
        )
        _require(
            self.n_heads % self.bc_groups == 0,
            "bc_groups must divide n_heads so the contiguous head-to-group mapping "
            f"is well-defined. Got bc_groups={self.bc_groups}, n_heads={self.n_heads}.",
        )
        self.heads_per_bc_group = int(self.n_heads // self.bc_groups)

        factory_kwargs = {"device": device, "dtype": dtype}
        self.scanprep = SLinOSSScanPrep(
            n_heads=self.n_heads,
            bc_groups=self.bc_groups,
            d_state=self.d_state,
            d_head=self.d_head,
            backend=scanprep_backend,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            theta_init_min=theta_init_min,
            theta_init_max=theta_init_max,
            r_min=r_min,
            r_max=r_max,
            eps=eps,
            device=device,
        )
        self.scan_backend = AutoScanBackend() if scan_backend is None else scan_backend
        self.decode_backend = (
            AutoMixerDecodeBackend() if decode_backend is None else decode_backend
        )
        self.cconv_backend = (
            AutoCConv1dBackend() if cconv_backend is None else cconv_backend
        )

        self.param_proj_dim = self.n_heads * self.scanprep.param_dim
        self.bc_proj_dim = self.bc_groups * self.scanprep.bc_param_rows * self.d_state
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + self.param_proj_dim + self.bc_proj_dim,
            bias=False,
            **factory_kwargs,
        )
        self.d_skip = nn.Parameter(
            torch.ones((self.n_heads,), device=device, dtype=torch.float32)
        )
        setattr(self.d_skip, "_no_weight_decay", True)
        self.dw_weight = nn.Parameter(
            torch.empty((self.d_inner, self.d_conv), **factory_kwargs)
        )
        self.dw_bias = nn.Parameter(torch.empty((self.d_inner,), **factory_kwargs))
        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=False,
            **factory_kwargs,
        )
        self.out_norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.d_skip)
        nn.init.kaiming_uniform_(
            self.dw_weight.view(self.d_inner, 1, self.d_conv),
            a=math.sqrt(5.0),
        )
        bound = 1.0 / math.sqrt(float(self.d_conv))
        nn.init.uniform_(self.dw_bias, -bound, bound)
        self.scanprep.reset_parameters()

    @staticmethod
    def _quantize_inner_dim(raw_d_inner: float, d_head: int) -> int:
        """Rounds an inner width to the nearest multiple of ``d_head``."""

        _require(raw_d_inner > 0.0, f"raw_d_inner must be positive. Got {raw_d_inner}.")
        units = raw_d_inner / float(d_head)
        quantized_units = max(1, int(math.floor(units + 0.5)))
        return int(quantized_units * d_head)

    def _supports_fast_decode(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        return isinstance(
            self.decode_backend,
            (AutoMixerDecodeBackend, CuteMixerDecodeBackend),
        ) and supports_cute_decode(
            self,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

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
                    (batch_size, self.bc_groups, 2 * self.d_state),
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
        if dtype is None:
            dtype = self.in_proj.weight.dtype
        device_obj = torch.device(device)
        if self._supports_fast_decode(
            batch_size=batch_size,
            device=device_obj,
            dtype=dtype,
        ):
            state.scan.state = make_decode_state_tensor(
                self,
                batch_size,
                device=device_obj,
            )
        return state

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: SLinOSSMixerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SLinOSSMixerState]:
        _require(
            x.ndim == 3 and x.shape[-1] == self.d_model,
            f"Expected x shape (batch, T, {self.d_model}), got {tuple(x.shape)}.",
        )

        batch_size, time_steps, _ = map(int, x.shape)
        if time_steps == 0:
            empty = x.new_empty((batch_size, 0, self.d_model))
            next_state = SLinOSSMixerState() if state is None else state
            return (empty, next_state) if return_state else empty

        gate, value_proj, params_flat, bc_flat = split_mixer_projection(
            x,
            self.in_proj.weight,
            d_inner=self.d_inner,
            param_proj_dim=self.param_proj_dim,
        )
        conv_input_state = None if state is None else state.conv
        value_output, conv_state = self.cconv_backend(
            self, value_proj, conv_input_state
        )
        bc = bc_flat.view(
            batch_size,
            time_steps,
            self.bc_groups,
            self.scanprep.bc_param_rows,
            self.d_state,
        )

        scan_inputs = self.scanprep(value_output, params_flat, bc)
        use_cute_scan = scan_inputs.U.device.type == "cuda" and isinstance(
            self.scan_backend,
            (AutoScanBackend, CuteScanBackend),
        )
        if use_cute_scan:
            _require(
                self.d_state % 8 == 0,
                "The current CuTe scan backend requires d_state to be a multiple of 8 "
                f"(got d_state={self.d_state}).",
            )
        scan_result = self.scan_backend(
            scan_inputs,
            chunk_size=self.chunk_size,
            state=None if state is None else state.scan,
            return_state=return_state,
        )
        scan_state: ScanState | None = None
        if return_state:
            scan_output, scan_state = cast(tuple[torch.Tensor, ScanState], scan_result)
        else:
            scan_output = cast(torch.Tensor, scan_result)
        out = mixer_tail(
            scan_output,
            gate,
            self.out_norm,
            self.out_proj,
            skip_input=scan_inputs.U,
            d_skip=self.d_skip,
        )
        if scan_state is None:
            return out
        return out, SLinOSSMixerState(conv=conv_state, scan=scan_state)

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
        _require(
            x.ndim == 3 and x.shape[1] == 1 and x.shape[-1] == self.d_model,
            f"Expected x shape (batch, d_model) or (batch, 1, {self.d_model}), "
            f"got {tuple(x.shape)}.",
        )

        token = x[:, 0, :]
        if inplace is None:
            inplace = not torch.is_grad_enabled()

        if torch.is_grad_enabled():
            _require(
                not inplace,
                "In-place decode is unsupported when gradients are enabled.",
            )
            y, next_state = cast(
                tuple[torch.Tensor, SLinOSSMixerState],
                self.forward(x, state=state, return_state=True),
            )
        else:
            batch_size = int(token.shape[0])
            next_state = (
                self.init_decode_state(
                    batch_size, device=token.device, dtype=token.dtype
                )
                if state is None
                else (state if inplace else state.clone())
            )
            use_graph_engine = (
                inplace
                and self._supports_fast_decode(
                    batch_size=batch_size,
                    device=token.device,
                    dtype=token.dtype,
                )
                and not _is_cuda_graph_capturing(token.device)
            )
            if use_graph_engine:
                ensure_fast_decode_state_layout(
                    self,
                    next_state,
                    batch_size=batch_size,
                    device=token.device,
                )
                engine = next_state._engine
                if (
                    not isinstance(engine, _MixerCudaGraphStepEngine)
                    or engine.mixer is not self
                    or engine.batch_size != batch_size
                    or engine.device != token.device
                    or engine.dtype != token.dtype
                ):
                    engine = _MixerCudaGraphStepEngine(
                        self,
                        next_state,
                        batch_size=batch_size,
                    )
                    next_state._engine = engine
                token_output, next_state = engine.step(token, next_state)
            else:
                token_output = run_inplace_decode_step(self, token, next_state)
            y = token_output.unsqueeze(1)

        if squeeze:
            return y[:, 0, :], next_state
        return y, next_state


__all__ = ["SLinOSSMixer"]
