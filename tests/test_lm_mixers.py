"""What an arm's name resolves to: the settings, the wrapper, and the baselines' absence.

Three failure modes here, none of which shows up until an arm is a day into a run.

A default named in this module that is not a field of the thing it builds raises at
construction, after the corpus is prepped and the width solved. The slinoss entry reads its
defaults off :class:`slinoss.SLinOSSConfig` for exactly that reason, so the check is that the
keys are the config's and that ``d_state`` -- the one value the config has no default for -- is
legal for the kernel.

:class:`scripts.lm.mixers.Unwrap` exists because the linear-attention baselines return a tuple.
Wrapping rather than adapting is a decision about parameter names: :func:`scripts.lm.groups.classify`
routes on the leaf name, and a wrapper that renamed leaves would move a baseline's whole mixer
into another optimizer group and change its rate.

The baselines' packages are absent on most hosts here. Resolving an arm's settings must not
need them, or no table could be planned on a host that does not run every arm.
"""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import Tensor, nn

from scripts.lm.groups import classify
from scripts.lm.mixers import REGISTRY, Unwrap
from slinoss import SLinOSSConfig

ARMS = ("slinoss", "gpt", "conv", "mamba2", "mamba3", "gdn2")


class _Tuple(nn.Module):
    """A mixer in the shape the linear-attention layers return."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, None, None]:
        return self.in_proj(x), None, None


class _Bare(nn.Module):
    """A mixer that returns one tensor, as the in-tree ones do."""

    def forward(self, x: Tensor) -> Tensor:
        return x * 2.0


def test_the_table_s_arms_are_all_registered() -> None:
    """The names a table's rows come from. A missing one is a typo found at launch."""
    assert set(ARMS) <= set(REGISTRY.names())
    assert REGISTRY.entry("gpt").max_length_policy == "required"
    assert REGISTRY.entry("slinoss").max_length_policy == "unused"
    assert all(
        REGISTRY.entry(name).max_length_policy == "unused"
        for name in set(ARMS) - {"gpt"}
    )


def test_a_baseline_s_settings_resolve_without_its_package() -> None:
    """Planning a table must not require every arm's package on the planning host.

    The import is inside the build, so ``resolve`` reports the settings and the failure
    arrives only when an arm is actually constructed. That is also what lets ``--set`` be
    validated against the defaults before anything is allocated.
    """
    for arm in ("mamba2", "mamba3", "gdn2"):
        resolved = REGISTRY.resolve(arm)
        assert resolved.settings
        assert resolved.name == arm


def test_the_slinoss_defaults_are_all_config_fields() -> None:
    """A default this module names and the config does not raises at construction.

    ``d_model`` is passed positionally by the builder, so naming it here would be a duplicate
    keyword; every other key has to be a field.
    """
    settings = REGISTRY.resolve("slinoss").settings
    names = {field.name for field in fields(SLinOSSConfig)}
    assert set(settings) <= names
    assert "d_model" not in settings
    assert "context_length" not in settings


def test_the_slinoss_defaults_are_the_config_s_own_values() -> None:
    """Read off the config, not restated, so a change there does not leave two truths.

    ``d_state`` is the exception and is named here because the config has no default for it.
    It must be a positive multiple of 48: the state is three rows of a rotation per lane, and
    the kernels tile the lane axis in sixteens.
    """
    settings = REGISTRY.resolve("slinoss").settings
    for key, value in settings.items():
        if key == "d_state":
            continue
        assert value == getattr(SLinOSSConfig, key)
    assert settings["d_state"] > 0
    assert settings["d_state"] % 48 == 0


def test_slinoss_declares_context_unused_and_keeps_its_init_span() -> None:
    """The fixed-span master mixer must not silently claim to consume context."""
    resolved = REGISTRY.resolve("slinoss", ["init_span=8192"])
    mixer = resolved.factory(128, 2048)
    assert mixer.config.init_span == 8192
    assert resolved.constructions[-1]["context"] == {
        "max_length_supplied": 2048,
        "max_length_policy": "unused",
        "max_length_consumed": None,
    }


def test_unwrap_takes_the_first_output_and_passes_a_bare_tensor_through() -> None:
    """One wrapper for both shapes, so no arm needs a per-baseline adapter."""
    x = torch.ones(2, 3, 4)
    wrapped = Unwrap(_Tuple())
    assert isinstance(wrapped(x), Tensor)
    assert wrapped(x).shape == (2, 3, 4)
    assert torch.equal(Unwrap(_Bare())(x), x * 2.0)


def test_unwrap_keeps_the_leaf_names_the_group_policy_routes_on() -> None:
    """A wrapper that renamed leaves would move a baseline's mixer to another rate.

    The names gain one ``inner.`` level and nothing else, so a projection weight inside a
    wrapped baseline lands in the same group as one inside an unwrapped mixer.
    """
    wrapped = Unwrap(_Tuple())
    leaves = [name for name, _ in wrapped.named_parameters()]
    assert leaves == ["inner.in_proj.weight"]
    param = nn.Parameter(torch.zeros(4, 4))
    inside = classify("blocks.0.mixer.inner.in_proj.weight", param)
    assert inside == classify("blocks.0.mixer.in_proj.weight", param)
    assert inside == "hidden"
