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


def _fields(state: MixerState) -> dict[str, Tensor]:
    """Every buffer, by field name, read off ``state`` on each call."""
    return {
        "conv": state.conv,
        "ssm": state.ssm,
        "b_prev": state.b_prev,
        "u_prev": state.u_prev,
    }


def test_allocate_matches_config(device: torch.device) -> None:
    """Buffer shapes come from the config, not from the default shape.

    Catches a container that assumes ``d_head == 64`` or ``d_conv == 4``: any
    other config then allocates a state the scan cannot consume. The two carries
    are here because ``b_prev`` follows ``n_groups`` and ``u_prev`` follows
    ``n_heads``, so a carry sized from the other one passes at the default config.
    """
    config = SLinOSSConfig(d_model=48, d_state=96, d_head=32, d_conv=3)
    state = MixerState.allocate(config, 3, device=device, dtype=torch.bfloat16)
    assert tuple(state.conv.shape) == (3, config.d_conv - 1, config.d_inner)
    assert tuple(state.ssm.shape) == (3, config.n_heads, config.d_head, config.d_state)
    assert tuple(state.b_prev.shape) == (3, config.n_groups, config.d_state)
    assert tuple(state.u_prev.shape) == (3, config.n_heads, config.d_head)
    assert state.ssm.shape[-1] == 3 * config.n_lanes
    assert all(
        buffer.is_contiguous() and not buffer.any()
        for buffer in _fields(state).values()
    )
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
    for index, buffer in enumerate(_fields(state).values(), start=1):
        buffer.fill_(float(index))
    before = [buffer.data_ptr() for buffer in _fields(state).values()]
    state.reset()
    after = _fields(state)
    assert [buffer.data_ptr() for buffer in after.values()] == before
    assert not any(buffer.any() for buffer in after.values())


def test_clone_is_independent() -> None:
    """Catches a clone that returns views: writing the copy would corrupt the
    captured buffers it was taken from."""
    state = mixer(torch.float32)
    copy = state.clone()
    original = _fields(state)
    for index, buffer in enumerate(_fields(copy).values(), start=1):
        buffer.fill_(float(index))
    assert not any(buffer.any() for buffer in original.values())
    assert all(
        buffer.dtype is original[name].dtype
        and tuple(buffer.shape) == tuple(original[name].shape)
        for name, buffer in _fields(copy).items()
    )


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
        (lambda s: {"conv": s.conv[0]}, ValueError, "conv must be"),
        (lambda s: {"ssm": s.ssm[0]}, ValueError, "ssm must be"),
        (lambda s: {"b_prev": s.b_prev[0]}, ValueError, "b_prev must be"),
        (lambda s: {"u_prev": s.u_prev[0]}, ValueError, "u_prev must be"),
        (lambda s: {"conv": s.conv.to(torch.int64)}, TypeError, "conv has dtype"),
        (lambda s: {"ssm": s.ssm.to(torch.bfloat16)}, TypeError, "float32-pinned"),
        (
            lambda s: {"conv": s.conv.to(torch.float64)},
            ValueError,
            "ssm must be float64",
        ),
        # The carries are activation-dtype, so the pinned state's widening does not
        # reach them. Ordered after the rule above, which the same mutation would
        # otherwise mask.
        (
            lambda s: {"b_prev": s.b_prev.to(torch.float32)},
            ValueError,
            "one activation dtype only",
        ),
        (lambda s: {"conv": s.conv[:, :, :16]}, ValueError, "both are d_inner"),
        (lambda s: {"u_prev": s.u_prev[:, :, :16]}, ValueError, "u_prev holds"),
        (lambda s: {"b_prev": s.b_prev[:, :, :16]}, ValueError, "b_prev holds"),
        # ``G`` divides ``H``: head ``h`` reads group ``h // (H // G)``, so a group
        # count that does not divide sends some head past the end of the band.
        (
            lambda s: {"b_prev": s.b_prev.new_zeros(BATCH, 3, CONFIG.d_state)},
            ValueError,
            "does not divide",
        ),
        # Zero groups short-circuits ahead of the modulus, which would otherwise be
        # ZeroDivisionError rather than a reported shape error.
        (
            lambda s: {"b_prev": s.b_prev.new_zeros(BATCH, 0, CONFIG.d_state)},
            ValueError,
            "does not divide",
        ),
        (lambda s: {"conv": s.conv[:1]}, ValueError, "one batch only"),
        (lambda s: {"conv": s.conv.to("meta")}, ValueError, "one device only"),
    ],
)
def test_mixer_rejects_bad_buffers(
    mutate: Callable[[MixerState], dict[str, Tensor]],
    exc: type[Exception],
    match: str,
) -> None:
    """Every buffer-level raise, named by the message it must produce.

    Catches validation that trusts its caller: a rank, width, batch, device, or
    dtype error survives to a kernel launch, where it reads as a CUDA fault
    rather than as a shape mismatch.

    Batch and device are mutated on ``conv`` alone. Both rules compare all four
    buffers at once, so which one disagrees is not a separate failure mode.
    """
    state = mixer()
    fields = _fields(state)
    fields.update(mutate(state))
    with pytest.raises(exc, match=match):
        MixerState(**fields)


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
