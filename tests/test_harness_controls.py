"""The two controls that bracket a recurrence.

They have to be what they claim. ``conv`` is the star-free floor, so its receptive field must
be exactly ``d_conv`` positions and it must be causal; ``attention`` is the strongest thing
that reads the whole prefix, so it must be causal too. A non-causal control leaks the answer
and stops bounding anything.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from scripts.harness import CausalAttention, CausalConv, Rotary

D_MODEL = 32
MAX_LENGTH = 16


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: CausalConv(D_MODEL, d_conv=4), id="conv"),
        pytest.param(
            lambda: CausalAttention(D_MODEL, MAX_LENGTH, n_heads=4), id="attention"
        ),
    ],
)
def test_controls_are_causal(build: Callable[[], nn.Module]) -> None:
    """Perturbing a position moves no earlier output.

    A control that leaks the future bounds nothing: ``attention`` would stop being the
    strongest thing that reads the whole prefix, and ``conv`` would stop being the star-free
    floor.
    """
    mixer = build()
    mixer.eval()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(1, MAX_LENGTH, D_MODEL, generator=gen)
    perturbed = x.clone()
    perturbed[0, -1] += 10.0
    with torch.no_grad():
        before = mixer(x)
        after = mixer(perturbed)
    assert torch.allclose(before[0, :-1], after[0, :-1], atol=1e-5)
    assert not torch.allclose(before[0, -1], after[0, -1])


def test_conv_receptive_field_is_the_tap_count() -> None:
    """``conv`` reads ``d_conv`` positions and no more, which is what makes it the floor.

    Perturbing position ``t - d_conv`` leaves output ``t`` untouched; perturbing
    ``t - d_conv + 1`` does not. A wider field would let it solve a task by locality that an
    axis reads as carried state.
    """
    taps = 4
    mixer = CausalConv(D_MODEL, d_conv=taps)
    mixer.eval()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(1, MAX_LENGTH, D_MODEL, generator=gen)
    target = MAX_LENGTH - 1
    with torch.no_grad():
        before = mixer(x)
        outside = x.clone()
        outside[0, target - taps] += 10.0
        assert torch.allclose(before[0, target], mixer(outside)[0, target], atol=1e-5)
        inside = x.clone()
        inside[0, target - taps + 1] += 10.0
        assert not torch.allclose(before[0, target], mixer(inside)[0, target])


def test_rotary_refuses_a_sequence_past_its_table() -> None:
    """A batch wider than ``max_length`` is a configuration error, not a resize.

    Silently extending the tables would hide which arm was asked to extrapolate.
    """
    rotary = Rotary(8, MAX_LENGTH)
    assert rotary(torch.zeros(1, 2, MAX_LENGTH, 8)).shape == (1, 2, MAX_LENGTH, 8)
    with pytest.raises(ValueError, match="is over the rotary table's"):
        rotary(torch.zeros(1, 2, MAX_LENGTH + 1, 8))
    with pytest.raises(ValueError, match="d_head must be even"):
        Rotary(7, MAX_LENGTH)


def test_attention_head_count_must_divide_the_stream() -> None:
    """An indivisible head count would reshape silently under ``unflatten``."""
    with pytest.raises(ValueError, match="does not divide d_model"):
        CausalAttention(D_MODEL, MAX_LENGTH, n_heads=5)
    with pytest.raises(ValueError, match="d_conv must be positive"):
        CausalConv(D_MODEL, d_conv=0)


def test_position_blind_attention_is_available_as_a_control() -> None:
    """``rotary=false`` leaves the mixer order-blind, since the scaffold has no positions.

    That is the control separating a task whose answer depends on order from one whose
    answer depends only on content: without rotary the prefix is a set.
    """
    mixer = CausalAttention(D_MODEL, MAX_LENGTH, rotary=False)
    assert isinstance(mixer.rotary, nn.Identity)
    out = mixer(torch.zeros(1, 4, D_MODEL))
    assert out.shape == (1, 4, D_MODEL)
