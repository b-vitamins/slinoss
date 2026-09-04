"""The mixer registry and the two controls.

An arm is a name plus a list of ``key=value`` overrides, so the registry is where a typo
becomes a silently different measurement: an unknown setting that is accepted and dropped
would report the arm at its defaults. Overrides are refused rather than merged, and a value
is read at the type of the setting's default.

The two controls have to be what they claim. ``conv`` is the star-free floor, so its
receptive field must be exactly ``d_conv`` positions and it must be causal; ``attention`` is
the strongest thing that succeeds on the automaton half, so it must be causal too. A
non-causal control leaks the answer and stops bounding anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields

import pytest
import torch
from torch import nn

from scripts.state_tracking.mixers import (
    REGISTRY,
    CausalAttention,
    CausalConv,
    MixerEntry,
    Rotary,
    load_module,
    register,
    resolve,
    settings_from,
)

D_MODEL = 32
MAX_LENGTH = 16


def test_the_three_builtins_are_registered() -> None:
    """The operator under test and its two controls, under the names the CLI spells."""
    assert set(REGISTRY) >= {"slinoss", "attention", "conv"}
    for name in ("slinoss", "attention", "conv"):
        assert REGISTRY[name].defaults, f"{name} declares no settings"


def test_slinoss_settings_build_a_legal_config() -> None:
    """Every slinoss setting names a field, and the set of them validates at d_model 128.

    A setting that is not a field reaches the constructor as an unexpected keyword, and one
    that is a field at an illegal value reaches ``__post_init__``. Either way the failure
    lands on a GPU host after a split has been generated, so both are settled here.
    ``d_state`` is named in the registry because the config has no default for it.
    """
    from slinoss import SLinOSSConfig

    settings = resolve("slinoss").settings
    names = {field.name for field in fields(SLinOSSConfig)}
    stack_only = {
        "d_model",
        "n_layers",
        "ffn_ratio",
        "norm_eps",
        "vocab_size",
        "vocab_pad_multiple",
    }
    assert set(settings) == names - stack_only
    assert settings["d_state"] == 144
    config = SLinOSSConfig(d_model=128, **settings)
    assert config.d_state == 144
    assert config.d_model == 128


def test_slinoss_declares_length_unused_and_keeps_its_init_span() -> None:
    """Task length cannot silently choose SLinOSS's initialization."""
    from slinoss import SLinOSSMixer

    resolved = resolve("slinoss")
    mixer = resolved.factory(128, MAX_LENGTH)
    assert isinstance(mixer, SLinOSSMixer)
    assert resolved.max_length_policy == "unused"
    assert mixer.config.init_span == 4096
    assert resolved.constructions[0]["context"] == {
        "max_length_supplied": MAX_LENGTH,
        "max_length_policy": "unused",
        "max_length_consumed": None,
    }
    assert resolved.constructions[0]["effective_config"]["init_span"] == 4096


def test_overrides_are_read_at_the_defaults_type() -> None:
    """An override string is coerced to the type of the setting it replaces."""
    settings = settings_from("conv", ["d_conv=8", "expand=1.5"])
    assert settings == {"d_conv": 8, "expand": 1.5}
    assert isinstance(settings["d_conv"], int)
    assert isinstance(settings["expand"], float)
    flags = settings_from("attention", ["rotary=false"])
    assert flags["rotary"] is False
    assert settings_from("attention", ["rotary=1"])["rotary"] is True


def test_a_bad_override_is_refused_rather_than_dropped() -> None:
    """An unknown setting, a malformed pair or an unreadable value stops the arm.

    Accepting and ignoring any of the three would report an arm at settings it did not
    run.
    """
    with pytest.raises(ValueError, match="no setting n_head"):
        settings_from("attention", ["n_head=4"])
    with pytest.raises(ValueError, match="override must be key=value"):
        settings_from("conv", ["d_conv"])
    with pytest.raises(ValueError, match="d_conv is int"):
        settings_from("conv", ["d_conv=wide"])
    with pytest.raises(ValueError, match="rotary is a flag"):
        settings_from("attention", ["rotary=maybe"])
    with pytest.raises(KeyError, match="no mixer mamba"):
        resolve("mamba")


def test_resolve_closes_over_the_settings_it_reports() -> None:
    """The factory builds at the settings the record carries, and nothing else."""
    mixer = resolve("conv", ["d_conv=2"])
    assert mixer.name == "conv"
    assert mixer.settings["d_conv"] == 2
    built = mixer.factory(D_MODEL, MAX_LENGTH)
    assert isinstance(built, CausalConv)
    assert built.d_conv == 2
    assert mixer.factory(D_MODEL, MAX_LENGTH) is not built
    assert mixer.max_length_policy == "unused"
    assert [item["context"]["max_length_consumed"] for item in mixer.constructions] == [
        None,
        None,
    ]


def test_length_consumption_is_mandatory_and_fail_closed() -> None:
    """An entry cannot regain the old implicit accept-and-drop convention."""
    with pytest.raises(ValueError, match="max_length_policy"):
        MixerEntry(lambda d_model: nn.Identity(), "sometimes")  # type: ignore[arg-type]
    attention = resolve("attention")
    attention.factory(D_MODEL, MAX_LENGTH)
    assert attention.max_length_policy == "required"
    assert attention.constructions[0]["context"]["max_length_consumed"] == MAX_LENGTH


def test_a_name_cannot_be_re_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-registration is refused: it would silently change what an arm measured.

    The registry is module state, so a baseline module imported twice, or two modules
    claiming one name, has to fail loudly.
    """
    entry = MixerEntry(lambda d_model: nn.Identity(), "unused", {})
    monkeypatch.setitem(REGISTRY, "probe", entry)
    assert resolve("probe").settings == {}
    with pytest.raises(ValueError, match="mixer probe is already registered"):
        register("probe", entry)


def test_load_module_reports_a_missing_baseline() -> None:
    """``--mixer-module`` on a module that is not importable fails at the import."""
    with pytest.raises(ModuleNotFoundError):
        load_module("scripts.state_tracking.not_a_baseline")


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
    strongest thing that succeeds on the automaton half, and ``conv`` would stop being the
    star-free floor.
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
    ``t - d_conv + 1`` does not. A wider field would let it solve a task by locality that
    the axis reads as carried state.
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

    Silently extending the tables would hide which arm was asked to extrapolate, and the
    whole axis is an evaluation past the trained length.
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

    That is the control that separates a task's answer depending on order from its
    depending only on content: without rotary the prefix is a set.
    """
    mixer = CausalAttention(D_MODEL, MAX_LENGTH, rotary=False)
    assert isinstance(mixer.rotary, nn.Identity)
    out = mixer(torch.zeros(1, 4, D_MODEL))
    assert out.shape == (1, 4, D_MODEL)
