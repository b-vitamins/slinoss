"""Mixer-specific causal depthwise conv helpers."""

from typing import TYPE_CHECKING, Protocol

import torch
from torch.nn import functional as F

from slinoss.ops.cconv1d import cconv1d_cuda, cconv1d_cuda_supported

if TYPE_CHECKING:

    class _CausalConvOwner(Protocol):
        @property
        def d_inner(self) -> int: ...

        @property
        def d_conv(self) -> int: ...

        @property
        def dw_weight(self) -> torch.Tensor: ...

        @property
        def dw_bias(self) -> torch.Tensor: ...


def _validate_conv_state(
    mixer: "_CausalConvOwner",
    conv_state: torch.Tensor | None,
    *,
    batch_size: int,
    state_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if conv_state is None:
        return torch.zeros(
            (batch_size, mixer.d_inner, state_len),
            device=device,
            dtype=dtype,
        )
    expected_shape = (batch_size, mixer.d_inner, state_len)
    if tuple(conv_state.shape) != expected_shape:
        raise ValueError(
            f"conv_state must be {expected_shape}. Got {tuple(conv_state.shape)}."
        )
    return conv_state.to(device=device, dtype=dtype)


def _next_conv_state_from_input(
    x_transposed: torch.Tensor,
    *,
    state_len: int,
) -> torch.Tensor:
    if state_len == 0:
        return x_transposed.new_empty((x_transposed.shape[0], x_transposed.shape[1], 0))
    if x_transposed.shape[-1] >= state_len:
        return x_transposed[..., -state_len:].contiguous()
    prefix = x_transposed.new_zeros(
        (
            x_transposed.shape[0],
            x_transposed.shape[1],
            state_len - x_transposed.shape[-1],
        )
    )
    return torch.cat((prefix, x_transposed), dim=-1).contiguous()


def apply_reference_causal_depthwise_conv(
    mixer: "_CausalConvOwner",
    x: torch.Tensor,
    conv_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _time_steps, channels = map(int, x.shape)
    if channels != mixer.d_inner:
        raise ValueError(f"Expected conv input width {mixer.d_inner}, got {channels}.")

    state_len = max(mixer.d_conv - 1, 0)
    weight = mixer.dw_weight.unsqueeze(1)
    if state_len == 0:
        output = F.conv1d(
            x.transpose(1, 2).contiguous(),
            weight,
            mixer.dw_bias,
            groups=mixer.d_inner,
        )
        empty_state = x.new_empty((batch_size, mixer.d_inner, 0))
        return output.transpose(1, 2).contiguous(), empty_state

    prefix = _validate_conv_state(
        mixer,
        conv_state,
        batch_size=batch_size,
        state_len=state_len,
        device=x.device,
        dtype=x.dtype,
    )
    x_transposed = x.transpose(1, 2).contiguous()
    stacked = torch.cat((prefix, x_transposed), dim=-1)
    output = F.conv1d(stacked, weight, mixer.dw_bias, groups=mixer.d_inner)
    next_state = stacked[..., -state_len:].contiguous()
    return output.transpose(1, 2).contiguous(), next_state


def apply_cuda_causal_depthwise_conv(
    mixer: "_CausalConvOwner",
    x: torch.Tensor,
    conv_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _time_steps, channels = map(int, x.shape)
    if channels != mixer.d_inner:
        raise ValueError(f"Expected conv input width {mixer.d_inner}, got {channels}.")

    state_len = max(mixer.d_conv - 1, 0)
    x_transposed = x.transpose(1, 2)
    weight = mixer.dw_weight
    if x_transposed.stride(1) == 1 and (
        mixer.d_inner % 8 != 0
        or x_transposed.stride(2) % 8 != 0
        or x_transposed.stride(0) % 8 != 0
    ):
        x_transposed = x_transposed.contiguous()

    if state_len == 0:
        if not cconv1d_cuda_supported(x_transposed, weight, activation=None):
            return apply_reference_causal_depthwise_conv(mixer, x, conv_state)
        output = cconv1d_cuda(x_transposed, weight, mixer.dw_bias, activation=None)
        assert isinstance(output, torch.Tensor)
        empty_state = x.new_empty((batch_size, mixer.d_inner, 0))
        return output.transpose(1, 2).contiguous(), empty_state

    if conv_state is None:
        if not cconv1d_cuda_supported(x_transposed, weight, activation=None):
            return apply_reference_causal_depthwise_conv(mixer, x, conv_state)
        output = cconv1d_cuda(x_transposed, weight, mixer.dw_bias, activation=None)
        assert isinstance(output, torch.Tensor)
        next_state = _next_conv_state_from_input(x_transposed, state_len=state_len)
        return output.transpose(1, 2).contiguous(), next_state

    if (
        x_transposed.stride(1) != 1
        or x_transposed.stride(2) % 8 != 0
        or x_transposed.stride(0) % 8 != 0
        or mixer.d_inner % 8 != 0
    ):
        return apply_reference_causal_depthwise_conv(mixer, x, conv_state)

    initial_state = _validate_conv_state(
        mixer,
        conv_state,
        batch_size=batch_size,
        state_len=state_len,
        device=x.device,
        dtype=x.dtype,
    )
    initial_state = initial_state.transpose(1, 2).contiguous().transpose(1, 2)
    if not cconv1d_cuda_supported(
        x_transposed,
        weight,
        initial_states=initial_state,
        activation=None,
    ):
        return apply_reference_causal_depthwise_conv(mixer, x, conv_state)
    output_with_state = cconv1d_cuda(
        x_transposed,
        weight,
        mixer.dw_bias,
        initial_states=initial_state,
        return_final_states=True,
        activation=None,
    )
    assert isinstance(output_with_state, tuple)
    output, next_state = output_with_state
    return output.transpose(1, 2).contiguous(), next_state.contiguous()


def apply_causal_depthwise_conv_step(
    mixer: "_CausalConvOwner",
    value_proj: torch.Tensor,
    conv_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value_proj.ndim != 2 or value_proj.shape[-1] != mixer.d_inner:
        raise ValueError(
            f"Expected value_proj shape (batch, {mixer.d_inner}), "
            f"got {tuple(value_proj.shape)}."
        )

    state_len = max(mixer.d_conv - 1, 0)
    x = value_proj.unsqueeze(1)
    x_transposed = x.transpose(1, 2)
    if x_transposed.stride(1) == 1 and (
        mixer.d_inner % 8 != 0
        or x_transposed.stride(2) % 8 != 0
        or x_transposed.stride(0) % 8 != 0
    ):
        x_transposed = x_transposed.contiguous()

    if state_len == 0:
        if cconv1d_cuda_supported(x_transposed, mixer.dw_weight, activation="silu"):
            output = cconv1d_cuda(
                x_transposed,
                mixer.dw_weight,
                mixer.dw_bias,
                activation="silu",
            )
            assert isinstance(output, torch.Tensor)
            empty_state = value_proj.new_empty((value_proj.shape[0], mixer.d_inner, 0))
            return output[..., 0].contiguous(), empty_state

        output_sequence, next_state = apply_reference_causal_depthwise_conv(
            mixer,
            x,
            None,
        )
        return torch.nn.functional.silu(output_sequence[:, 0, :]), next_state

    initial_state = _validate_conv_state(
        mixer,
        conv_state,
        batch_size=int(value_proj.shape[0]),
        state_len=state_len,
        device=value_proj.device,
        dtype=value_proj.dtype,
    )
    if (
        x_transposed.stride(1) != 1
        or x_transposed.stride(2) % 8 != 0
        or x_transposed.stride(0) % 8 != 0
        or mixer.d_inner % 8 != 0
    ):
        output_sequence, next_state = apply_reference_causal_depthwise_conv(
            mixer,
            x,
            initial_state.contiguous(),
        )
        return torch.nn.functional.silu(output_sequence[:, 0, :]), next_state

    initial_state = initial_state.transpose(1, 2).contiguous().transpose(1, 2)
    if cconv1d_cuda_supported(
        x_transposed,
        mixer.dw_weight,
        initial_states=initial_state,
        activation="silu",
    ):
        output_with_state = cconv1d_cuda(
            x_transposed,
            mixer.dw_weight,
            mixer.dw_bias,
            initial_states=initial_state,
            return_final_states=True,
            activation="silu",
        )
        assert isinstance(output_with_state, tuple)
        output, next_state = output_with_state
        return output[..., 0].contiguous(), next_state.contiguous()

    output_sequence, next_state = apply_reference_causal_depthwise_conv(
        mixer,
        x,
        initial_state.contiguous(),
    )
    return torch.nn.functional.silu(output_sequence[:, 0, :]), next_state


__all__ = [
    "apply_causal_depthwise_conv_step",
    "apply_cuda_causal_depthwise_conv",
    "apply_reference_causal_depthwise_conv",
]
