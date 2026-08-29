"""The mixer registry.

An arm is a name plus a list of ``key=value`` overrides, so the registry is where a typo
becomes a silently different measurement: an unknown setting that is accepted and dropped
would report the arm at its defaults. Overrides are refused rather than merged, and a value is
read at the type of the setting's default.

The registry is an object, so these tests build their own rather than mutating a shared one.
Two axes registering ``slinoss`` at different defaults is the case the class exists for, and a
test that reached into the language-modelling registry would be testing that axis's defaults
instead of the machinery.
"""

from __future__ import annotations

import pytest
from torch import nn

from scripts.harness import (
    CausalAttention,
    CausalConv,
    MixerEntry,
    Registry,
    load_module,
)

D_MODEL = 32
MAX_LENGTH = 16


@pytest.fixture
def registry() -> Registry:
    """A registry holding the two controls at known defaults."""
    fresh = Registry("probe")
    fresh.register(
        "attention", MixerEntry(CausalAttention, {"n_heads": 4, "rotary": True})
    )
    fresh.register("conv", MixerEntry(CausalConv, {"d_conv": 4, "expand": 2.0}))
    return fresh


def test_overrides_are_read_at_the_defaults_type(registry: Registry) -> None:
    """An override string is coerced to the type of the setting it replaces."""
    settings = registry.settings_from("conv", ["d_conv=8", "expand=1.5"])
    assert settings == {"d_conv": 8, "expand": 1.5}
    assert isinstance(settings["d_conv"], int)
    assert isinstance(settings["expand"], float)
    flags = registry.settings_from("attention", ["rotary=false"])
    assert flags["rotary"] is False
    assert registry.settings_from("attention", ["rotary=1"])["rotary"] is True


def test_a_bad_override_is_refused_rather_than_dropped(registry: Registry) -> None:
    """An unknown setting, a malformed pair or an unreadable value stops the arm.

    Accepting and ignoring any of the three would report an arm at settings it did not run.
    """
    with pytest.raises(ValueError, match="no setting n_head"):
        registry.settings_from("attention", ["n_head=4"])
    with pytest.raises(ValueError, match="override must be key=value"):
        registry.settings_from("conv", ["d_conv"])
    with pytest.raises(ValueError, match="d_conv is int"):
        registry.settings_from("conv", ["d_conv=wide"])
    with pytest.raises(ValueError, match="rotary is a flag"):
        registry.settings_from("attention", ["rotary=maybe"])
    with pytest.raises(KeyError, match="no probe mixer mamba"):
        registry.resolve("mamba")


def test_resolve_closes_over_the_settings_it_reports(registry: Registry) -> None:
    """The factory builds at the settings the record carries, and nothing else."""
    mixer = registry.resolve("conv", ["d_conv=2"])
    assert mixer.name == "conv"
    assert mixer.settings["d_conv"] == 2
    built = mixer.factory(D_MODEL, MAX_LENGTH)
    assert isinstance(built, CausalConv)
    assert built.d_conv == 2
    assert mixer.factory(D_MODEL, MAX_LENGTH) is not built


def test_a_name_cannot_be_re_registered(registry: Registry) -> None:
    """Re-registration is refused: it would silently change what an arm measured."""
    entry = MixerEntry(lambda d_model, max_length: nn.Identity(), {})
    registry.register("probe", entry)
    assert registry.resolve("probe").settings == {}
    with pytest.raises(ValueError, match="probe mixer probe is already registered"):
        registry.register("probe", entry)


def test_two_registries_hold_one_name_at_different_defaults() -> None:
    """The reason the registry is an object.

    ``slinoss`` on a 128-wide state-tracking stream and ``slinoss`` at a language-modelling
    width are different defaults under one name. A module-level dict would make importing
    both axes in one process a re-registration error.
    """
    first, second = Registry("first"), Registry("second")
    first.register("conv", MixerEntry(CausalConv, {"d_conv": 2, "expand": 1.0}))
    second.register("conv", MixerEntry(CausalConv, {"d_conv": 8, "expand": 4.0}))
    assert first.resolve("conv").settings["d_conv"] == 2
    assert second.resolve("conv").settings["d_conv"] == 8


def test_load_module_reports_a_missing_baseline() -> None:
    """``--mixer-module`` on a module that is not importable fails at the import."""
    with pytest.raises(ModuleNotFoundError):
        load_module("scripts.harness.not_a_baseline")
