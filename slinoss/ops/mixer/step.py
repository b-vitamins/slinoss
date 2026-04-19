"""Mixer-specific decode-time execution helpers."""

from typing import TYPE_CHECKING, Protocol, cast

import torch

from slinoss.ops.decode import decode_linear
from slinoss.ops.v2x2ssd.reference import v2x2ssm

from .convolution import apply_causal_depthwise_conv_step
from .projection import split_mixer_projection
from .tail import _mixer_gated_hidden

if TYPE_CHECKING:
    from slinoss.layers.backend import MixerDecodeInputs, ScanInputs, ScanPrepInputs
    from slinoss.layers.mixer import SLinOSSMixer
    from slinoss.layers.state import SLinOSSMixerState, ScanState

    class _DecodeScanPrep(Protocol):
        @property
        def param_dim(self) -> int: ...

        @property
        def bc_param_rows(self) -> int: ...

        @property
        def dt_min(self) -> float: ...

        @property
        def dt_max(self) -> float: ...

        @property
        def theta_init_min(self) -> float: ...

        @property
        def theta_init_max(self) -> float: ...

        @property
        def theta_mod_scale(self) -> float: ...

        @property
        def alpha_min(self) -> float: ...

        @property
        def alpha_max(self) -> float: ...

        @property
        def r_min(self) -> float: ...

        @property
        def r_max(self) -> float: ...

        @property
        def eps(self) -> float: ...

        @property
        def dt_bias(self) -> torch.Tensor: ...

        @property
        def alpha_bias(self) -> torch.Tensor: ...

        @property
        def theta_mod_bias(self) -> torch.Tensor: ...

        @property
        def theta_bias(self) -> torch.Tensor: ...

        @property
        def theta_sign(self) -> torch.Tensor: ...

        def _prepare_inputs_reference(
            self,
            inputs: "ScanPrepInputs",
        ) -> "ScanInputs": ...

        def _parameterize_scan_bc_rows(self, bc: torch.Tensor) -> torch.Tensor: ...

    class _DecodeOwner(Protocol):
        @property
        def bc_groups(self) -> int: ...

        @property
        def d_head(self) -> int: ...

        @property
        def d_state(self) -> int: ...

        @property
        def d_inner(self) -> int: ...

        @property
        def param_proj_dim(self) -> int: ...

        @property
        def n_heads(self) -> int: ...

        @property
        def scanprep(self) -> "_DecodeScanPrep": ...

        @property
        def out_norm(self) -> torch.nn.Module: ...

        @property
        def d_skip(self) -> torch.Tensor: ...


def supports_cute_decode(
    mixer: "_DecodeOwner",
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    if mixer.d_skip is not None:
        return False
    if device.type != "cuda":
        return False
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if batch_size not in (1, 2, 4, 8, 16):
        return False
    return mixer.d_head == 64 and mixer.d_state == 64


def make_decode_state_tensor(
    mixer: "SLinOSSMixer",
    batch_size: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.zeros(
        (batch_size, mixer.n_heads, 2 * mixer.d_state, mixer.d_head),
        device=device,
        dtype=torch.float32,
    ).transpose(-1, -2)


def ensure_fast_decode_state_layout(
    mixer: "SLinOSSMixer",
    state: "SLinOSSMixerState",
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    if state.scan.state is None:
        state.scan.state = make_decode_state_tensor(
            mixer,
            batch_size,
            device=device,
        )
        return

    expected_shape = (batch_size, mixer.n_heads, mixer.d_head, 2 * mixer.d_state)
    if tuple(state.scan.state.shape) != expected_shape:
        raise ValueError(
            f"scan.state must match {expected_shape}. "
            f"Got {tuple(state.scan.state.shape)}."
        )
    if state.scan.state.device != device or state.scan.state.dtype != torch.float32:
        state.scan.state = state.scan.state.to(device=device, dtype=torch.float32)
    if state.scan.state.stride()[-2:] == (1, mixer.d_head):
        return

    fast_state = make_decode_state_tensor(
        mixer,
        batch_size,
        device=device,
    )
    fast_state.copy_(state.scan.state)
    state.scan.state = fast_state


def make_decode_inputs(
    mixer: "SLinOSSMixer",
    *,
    batch_size: int,
    value_token: torch.Tensor,
    param_token: torch.Tensor,
    bc_token: torch.Tensor,
    gate_token: torch.Tensor,
) -> "MixerDecodeInputs":
    from slinoss.layers.backend import MixerDecodeInputs

    return MixerDecodeInputs(
        value=value_token.view(batch_size, mixer.n_heads, mixer.d_head).contiguous(),
        params=param_token.view(
            batch_size,
            mixer.n_heads,
            mixer.scanprep.param_dim,
        ).contiguous(),
        bc=bc_token.view(
            batch_size,
            mixer.bc_groups,
            mixer.scanprep.bc_param_rows,
            mixer.d_state,
        ).contiguous(),
        gate=gate_token.view(batch_size, mixer.n_heads, mixer.d_head).contiguous(),
    )


def run_reference_decode_step(
    mixer: "_DecodeOwner",
    inputs,
    state: "ScanState",
) -> tuple[torch.Tensor, "ScanState"]:
    from slinoss.layers.backend import ScanPrepInputs
    from slinoss.layers.state import ScanState

    batch_size = int(inputs.value.shape[0])
    value_token = inputs.value.reshape(batch_size, 1, mixer.d_inner).contiguous()
    param_token = inputs.params.reshape(
        batch_size, 1, mixer.param_proj_dim
    ).contiguous()
    bc_token = inputs.bc.reshape(
        batch_size,
        1,
        mixer.bc_groups,
        mixer.scanprep.bc_param_rows,
        mixer.d_state,
    ).contiguous()
    gate_token = inputs.gate.reshape(batch_size, 1, mixer.d_inner).contiguous()

    scan_inputs = mixer.scanprep._prepare_inputs_reference(
        ScanPrepInputs(value=value_token, params=param_token, bc=bc_token)
    )
    compute_dtype = (
        torch.float32
        if value_token.dtype in (torch.float16, torch.bfloat16)
        else value_token.dtype
    )
    scan_output, final_state, b_last, u_last = v2x2ssm(
        scan_inputs.U,
        scan_inputs.M,
        scan_inputs.K,
        scan_inputs.B,
        scan_inputs.C,
        initial_states=state.state,
        B_prev=state.b_prev,
        U_prev=state.u_prev,
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
    )
    scan_output = scan_output.to(dtype=value_token.dtype)
    b_last = b_last.to(dtype=value_token.dtype)
    u_last = u_last.to(dtype=value_token.dtype)
    normalized_output = mixer.out_norm(
        _mixer_gated_hidden(
            scan_output,
            gate_token,
            skip_input=scan_inputs.U,
            d_skip=mixer.d_skip,
        )
    )[:, 0, :]
    return normalized_output.contiguous(), ScanState(
        state=final_state,
        b_prev=b_last,
        u_prev=u_last,
    )


def run_cute_decode_step(
    mixer: "_DecodeOwner",
    inputs,
    state: "ScanState",
) -> tuple[torch.Tensor, "ScanState"]:
    if mixer.d_skip is not None:
        return run_reference_decode_step(mixer, inputs, state)
    from slinoss.layers.state import ScanState
    from slinoss.ops.v2x2ssd.cute.step import mixer_decode_step_cute

    gated_output, final_state, b_last, u_last = mixer_decode_step_cute(
        inputs.value,
        inputs.params,
        mixer.scanprep._parameterize_scan_bc_rows(inputs.bc.unsqueeze(1))[:, 0, ...],
        inputs.gate,
        initial_states=state.state,
        B_prev=state.b_prev,
        U_prev=state.u_prev,
        dt_min=mixer.scanprep.dt_min,
        dt_max=mixer.scanprep.dt_max,
        theta_init_min=mixer.scanprep.theta_init_min,
        theta_init_max=mixer.scanprep.theta_init_max,
        theta_mod_scale=mixer.scanprep.theta_mod_scale,
        alpha_min=mixer.scanprep.alpha_min,
        alpha_max=mixer.scanprep.alpha_max,
        r_min=mixer.scanprep.r_min,
        r_max=mixer.scanprep.r_max,
        eps=mixer.scanprep.eps,
        dt_bias=mixer.scanprep.dt_bias,
        alpha_bias=mixer.scanprep.alpha_bias,
        theta_mod_bias=mixer.scanprep.theta_mod_bias,
        theta_bias=mixer.scanprep.theta_bias,
        theta_sign=cast(torch.Tensor, mixer.scanprep.theta_sign),
        output_dtype=inputs.value.dtype,
        final_state_out=state.state,
        b_last_out=state.b_prev,
        u_last_out=state.u_prev,
    )
    next_state = ScanState(
        state=final_state,
        b_prev=b_last,
        u_prev=u_last,
    )
    return mixer.out_norm(gated_output), next_state


def _adopt_conv_state_(
    state: "SLinOSSMixerState",
    next_conv_state: torch.Tensor,
) -> None:
    if (
        state.conv is None
        or tuple(state.conv.shape) != tuple(next_conv_state.shape)
        or state.conv.device != next_conv_state.device
        or state.conv.dtype != next_conv_state.dtype
    ):
        state.conv = next_conv_state
        return
    if state.conv is not next_conv_state:
        state.conv.copy_(next_conv_state)


def run_inplace_decode_step(
    mixer: "SLinOSSMixer",
    x: torch.Tensor,
    state: "SLinOSSMixerState",
) -> torch.Tensor:
    batch_size = int(x.shape[0])
    gate_token, value_proj, param_token, bc_token = split_mixer_projection(
        x,
        mixer.in_proj.weight,
        d_inner=mixer.d_inner,
        param_proj_dim=mixer.param_proj_dim,
    )
    value_token, next_conv_state = apply_causal_depthwise_conv_step(
        mixer,
        value_proj,
        state.conv,
    )
    _adopt_conv_state_(state, next_conv_state)

    decode_inputs = make_decode_inputs(
        mixer,
        batch_size=batch_size,
        value_token=value_token,
        param_token=param_token,
        bc_token=bc_token,
        gate_token=gate_token,
    )
    if mixer._supports_fast_decode(
        batch_size=batch_size,
        device=x.device,
        dtype=x.dtype,
    ):
        ensure_fast_decode_state_layout(
            mixer,
            state,
            batch_size=batch_size,
            device=x.device,
        )

    decode_output, next_scan_state = mixer.decode_backend(
        mixer,
        decode_inputs,
        state.scan,
    )
    state.scan.adopt_(next_scan_state)
    return decode_linear(decode_output, mixer.out_proj)


class MixerCudaGraphStepEngine:
    """Fixed-shape CUDA graph replay for one-token mixer decode."""

    def __init__(
        self,
        mixer: "SLinOSSMixer",
        state: "SLinOSSMixerState",
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

    def _capture(self, state: "SLinOSSMixerState") -> None:
        snapshot = state.clone()
        stream = torch.cuda.Stream(device=self.device)
        current_stream = torch.cuda.current_stream(device=self.device)
        stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    state.copy_(snapshot)
                    self.static_y = run_inplace_decode_step(
                        self.mixer,
                        self.x_buffer,
                        state,
                    )
            state.copy_(snapshot)
            current_stream.wait_stream(stream)
            with torch.cuda.graph(self.graph):
                self.static_y = run_inplace_decode_step(
                    self.mixer,
                    self.x_buffer,
                    state,
                )
        finally:
            current_stream.wait_stream(stream)
            state.copy_(snapshot)

    def step(
        self,
        x: torch.Tensor,
        state: "SLinOSSMixerState",
    ) -> tuple[torch.Tensor, "SLinOSSMixerState"]:
        self.x_buffer.copy_(x)
        self.graph.replay()
        if self.static_y is None:
            raise RuntimeError("Mixer decode graph did not materialize an output.")
        return self.static_y.clone(), state


__all__ = [
    "MixerCudaGraphStepEngine",
    "ensure_fast_decode_state_layout",
    "make_decode_inputs",
    "make_decode_state_tensor",
    "run_cute_decode_step",
    "run_inplace_decode_step",
    "run_reference_decode_step",
    "supports_cute_decode",
]
