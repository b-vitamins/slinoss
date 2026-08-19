"""Decode state containers. Every raise in the public path is triggered here."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from slinoss.config import SLinOSSConfig
from slinoss.state import MixerState, StackState

CONFIG = SLinOSSConfig(d_model=64, d_state=48, n_layers=3)
"""d_inner 128, H 2, P 64, 3N 48, three layers."""

BATCH = 2


def mixer(dtype: torch.dtype = torch.bfloat16, device: str = "cpu") -> MixerState:
    return MixerState.allocate(CONFIG, BATCH, device=device, dtype=dtype)


def test_allocate_matches_config(device: torch.device) -> None:
    """Buffer shapes come from the config, not from the default shape.

    Catches a container that assumes ``d_head == 64`` or ``d_conv == 4``: any
    other config then allocates a state the scan cannot consume.
    """
    config = SLinOSSConfig(d_model=48, d_state=96, d_head=32, d_conv=3)
    state = MixerState.allocate(config, 3, device=device, dtype=torch.bfloat16)
    assert tuple(state.conv.shape) == (3, config.d_conv - 1, config.d_inner)
    assert tuple(state.ssm.shape) == (3, config.n_heads, config.d_head, config.d_state)
    assert state.ssm.shape[-1] == 3 * config.n_lanes
    assert state.ssm.is_contiguous() and state.conv.is_contiguous()
    assert not state.conv.any() and not state.ssm.any()
    assert state.batch == 3
    assert state.device.type == device.type


@pytest.mark.parametrize(
    ("dtype", "want"),
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_state_dtype_follows_the_pinning_policy(
    dtype: torch.dtype, want: torch.dtype
) -> None:
    """I4 on the recurrent state, and its float64 exception.

    Catches a state allocated at the activation dtype, which puts the recurrent
    decay in bfloat16, and a state hardcoded to float32, which downcasts a
    float64 gradcheck path without saying so.
    """
    state = mixer(dtype)
    assert state.conv.dtype is dtype
    assert state.ssm.dtype is want


def test_reset_keeps_the_buffer_addresses() -> None:
    """Catches a reset that rebinds a buffer.

    Graph replay writes the address recorded at capture, so a reallocation
    leaves the graph and the container pointing at different memory.
    """
    state = mixer(torch.float32)
    state.conv.fill_(1.0)
    state.ssm.fill_(2.0)
    before = (state.conv.data_ptr(), state.ssm.data_ptr())
    state.reset()
    assert (state.conv.data_ptr(), state.ssm.data_ptr()) == before
    assert not state.conv.any() and not state.ssm.any()


def test_clone_is_independent() -> None:
    """Catches a clone that returns views: writing the copy would corrupt the
    captured buffers it was taken from."""
    state = mixer(torch.float32)
    copy = state.clone()
    copy.conv.fill_(1.0)
    copy.ssm.fill_(2.0)
    assert not state.conv.any() and not state.ssm.any()
    assert copy.conv.dtype is state.conv.dtype
    assert copy.ssm.dtype is state.ssm.dtype
    assert tuple(copy.ssm.shape) == tuple(state.ssm.shape)


def test_stack_allocates_one_buffer_set_per_layer() -> None:
    """Catches ``(state,) * n_layers``: one buffer aliased across the stack
    makes every layer decode from its neighbour's state."""
    stack = StackState.allocate(CONFIG, BATCH, device="cpu", dtype=torch.bfloat16)
    assert len(stack.layers) == CONFIG.n_layers
    assert len({layer.conv.data_ptr() for layer in stack.layers}) == CONFIG.n_layers
    assert len({layer.ssm.data_ptr() for layer in stack.layers}) == CONFIG.n_layers
    assert stack.batch == BATCH
    assert stack.device.type == "cpu"


def test_stack_reset_and_clone_reach_every_layer() -> None:
    """Catches a reset or a clone that stops after layer 0."""
    stack = StackState.allocate(CONFIG, BATCH, device="cpu", dtype=torch.float32)
    for layer in stack.layers:
        layer.conv.fill_(1.0)
        layer.ssm.fill_(1.0)
    copy = stack.clone()
    stack.reset()
    assert all(not layer.conv.any() and not layer.ssm.any() for layer in stack.layers)
    assert all(
        bool(layer.conv.eq(1.0).all()) and bool(layer.ssm.eq(1.0).all())
        for layer in copy.layers
    )


def test_containers_are_frozen() -> None:
    """Catches a mutable container: a field swapped after capture leaves the
    graph writing a buffer nobody reads."""
    state = mixer()
    stack = StackState.allocate(CONFIG, BATCH, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.ssm = state.ssm.clone()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        stack.layers = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "exc", "match"),
    [
        (lambda s: (s.conv[0], s.ssm), ValueError, "conv must be"),
        (lambda s: (s.conv, s.ssm[0]), ValueError, "ssm must be"),
        (lambda s: (s.conv.to(torch.int64), s.ssm), TypeError, "conv has dtype"),
        (lambda s: (s.conv, s.ssm.to(torch.bfloat16)), TypeError, "float32-pinned"),
        (
            lambda s: (s.conv.to(torch.float64), s.ssm),
            ValueError,
            "ssm must be float64",
        ),
        (lambda s: (s.conv[:, :, :16], s.ssm), ValueError, "both are d_inner"),
        (lambda s: (s.conv[:1], s.ssm), ValueError, "one batch only"),
        (lambda s: (s.conv.to("meta"), s.ssm), ValueError, "one device only"),
    ],
)
def test_mixer_rejects_bad_buffers(
    mutate: Callable[[MixerState], tuple[Tensor, Tensor]],
    exc: type[Exception],
    match: str,
) -> None:
    """Every buffer-level raise, named by the message it must produce.

    Catches validation that trusts its caller: a rank, width, batch, device, or
    dtype error survives to a kernel launch, where it reads as a CUDA fault
    rather than as a shape mismatch.
    """
    conv, ssm = mutate(mixer())
    with pytest.raises(exc, match=match):
        MixerState(conv=conv, ssm=ssm)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: (), "at least one MixerState"),
        (
            lambda: (
                mixer(),
                MixerState.allocate(CONFIG, 3, device="cpu", dtype=torch.bfloat16),
            ),
            "one batch only",
        ),
        (lambda: (mixer(), mixer(device="meta")), "one device only"),
    ],
)
def test_stack_rejects_inconsistent_layers(
    build: Callable[[], tuple[MixerState, ...]], match: str
) -> None:
    """Catches a stack assembled from layers that disagree, and an empty stack
    whose ``batch`` and ``device`` would raise IndexError instead."""
    with pytest.raises(ValueError, match=match):
        StackState(layers=build())


@pytest.mark.parametrize("cls", [MixerState, StackState])
def test_allocate_rejects_nonpositive_batch(
    cls: type[MixerState] | type[StackState],
) -> None:
    """Catches a batch that reaches ``torch.zeros``, where 0 allocates an empty
    buffer the decode path silently reads as a no-op."""
    with pytest.raises(ValueError, match="batch must be positive"):
        cls.allocate(CONFIG, 0, device="cpu", dtype=torch.float32)
