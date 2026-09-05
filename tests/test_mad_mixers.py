"""The registry, and the two controls the suite is read against.

A registry defect is silent in the worst way: an override the mixer never saw, or a
default that drifted off the config it claims to track, changes what an arm measured
without changing what its record says. So the surface is pinned here, and the controls
are held to the two properties that make them controls -- attention sees the whole prefix
and nothing after it, the convolution sees ``d_conv`` positions and nothing before them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from scripts.mad.mixers import (
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
from scripts.mad.train import TrainConfig, _autocast

STUB_DEFAULTS: dict[str, Any] = {
    "mode": "wide",
    "taps": 4,
    "gain": 2.0,
    "gated": True,
}
"""One setting of each type an override string is coerced to."""


class Stub(nn.Module):
    """A mixer that records the settings it was built with."""

    def __init__(self, d_model: int, max_length: int, **settings: Any) -> None:
        super().__init__()
        self.max_length = max_length
        self.settings = settings
        self.weight = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` unchanged."""
        return x


@contextmanager
def registered(name: str, entry: MixerEntry) -> Iterator[None]:
    """Add an entry for the duration of one test, then remove it.

    Args:
        name: Registry key. Must be free.
        entry: The entry.

    Yields:
        Nothing.
    """
    register(name, entry)
    try:
        yield
    finally:
        del REGISTRY[name]


def test_the_three_builtins_are_registered() -> None:
    """The operator under test and the two controls, under the names the flags use."""
    assert {"slinoss", "attention", "conv"} <= set(REGISTRY)


def test_an_unregistered_name_names_what_is_registered() -> None:
    """A typo has to be readable off the message, not chased through the registry."""
    with pytest.raises(KeyError, match="attention"):
        resolve("attetion")


def test_a_taken_name_is_refused() -> None:
    """Re-registration would silently change what every later arm measured."""
    with pytest.raises(ValueError, match="already registered"):
        register("conv", MixerEntry(Stub, "required"))


def test_overrides_are_read_at_the_type_of_the_default() -> None:
    """Each setting's default type is the rule, and the flag reads either spelling."""
    with registered("stub", MixerEntry(Stub, "required", dict(STUB_DEFAULTS))):
        settings = settings_from(
            "stub", ["mode=narrow", "taps=7", "gain=0.5", "gated=false"]
        )
        assert settings == {
            "mode": "narrow",
            "taps": 7,
            "gain": 0.5,
            "gated": False,
        }
        assert settings_from("stub", ["gated=1"])["gated"] is True
        assert settings_from("stub", ["gated=no"])["gated"] is False
        # bool is a subclass of int, so a flag read as int would take 0 and 1 and
        # then fail on every other spelling.
        assert settings_from("stub", [])["gated"] is True


@pytest.mark.parametrize(
    "override,message",
    [
        ("taps", "key=value"),
        ("tabs=4", "no setting tabs"),
        ("taps=many", "taps is int"),
        ("gain=high", "gain is float"),
        ("gated=maybe", "gated is a flag"),
    ],
)
def test_a_bad_override_is_refused_with_its_reason(override: str, message: str) -> None:
    """A setting a mixer does not have, or a value its type does not read, is an error.

    Dropping either would leave the record claiming a setting the mixer never saw.
    """
    with (
        registered("stub", MixerEntry(Stub, "required", dict(STUB_DEFAULTS))),
        pytest.raises(ValueError, match=message),
    ):
        settings_from("stub", [override])


def test_resolve_closes_over_the_settings_it_reports() -> None:
    """The factory builds with exactly the settings the record carries."""
    with registered("stub", MixerEntry(Stub, "required", dict(STUB_DEFAULTS))):
        mixer = resolve("stub", ["taps=9"])
        assert mixer.name == "stub"
        assert mixer.settings == {**STUB_DEFAULTS, "taps": 9}
        built = mixer.factory(8, 16)
        assert isinstance(built, Stub)
        assert built.settings == mixer.settings
        assert built.max_length == 16
        assert mixer.constructions[0]["context"]["max_length_consumed"] == 16


def test_length_consumption_is_mandatory_and_fail_closed() -> None:
    """An entry cannot regain the old implicit accept-and-drop convention."""
    with pytest.raises(ValueError, match="max_length_policy"):
        MixerEntry(Stub, "sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initialization_policy"):
        MixerEntry(Stub, "unused", {}, "sometimes")  # type: ignore[arg-type]
    resolved = resolve("slinoss")
    built = resolved.factory(128, 256)
    from slinoss import SLinOSSMixer

    assert isinstance(built, SLinOSSMixer)
    assert built.config.init_span == 4096
    assert built.config.context_length == 256
    assert resolved.max_length_policy == "required"
    assert resolved.initialization_policy == "constructor"
    assert resolved.constructions[0]["context"]["max_length_consumed"] == 256
    assert resolved.constructions[0]["initialization_policy"] == "constructor"


def test_slinoss_context_scales_resolve_from_configured_task_length() -> None:
    """The shared span law consumes task metadata, never observed tensor width."""
    resolved = resolve(
        "slinoss",
        ["init_period_context_scale=1", "init_decay_context_scale=16"],
    )
    built = resolved.factory(128, 32)
    from slinoss import SLinOSSMixer

    assert isinstance(built, SLinOSSMixer)
    assert built.config.context_length == 32
    assert built.config.resolved_init_period_span == 32
    assert built.config.resolved_init_decay_span == 512
    effective = resolved.constructions[0]["effective_config"]
    assert effective["resolved_init_period_span"] == 32
    assert effective["resolved_init_decay_span"] == 512


def test_slinoss_defaults_track_the_config() -> None:
    """The entry restates no default of its own except ``d_state``.

    A restated default drifts: the bound on the rotation norm was 2.0 once, which puts
    every order-2 element of the reachable group outside the ball, and an arm inheriting
    that from a stale copy would measure a different operator than its record names.
    """
    from slinoss import SLinOSSConfig

    defaults = REGISTRY["slinoss"].defaults
    fields = set(SLinOSSConfig.__dataclass_fields__)
    stack_only = {
        "d_model",
        "n_layers",
        "ffn_ratio",
        "norm_eps",
        "vocab_size",
        "vocab_pad_multiple",
        "context_length",
    }
    assert set(defaults) == fields - stack_only
    for key in set(defaults) - {"d_state"}:
        assert defaults[key] == getattr(SLinOSSConfig, key)
    assert defaults["w_max"] > 3.0
    assert defaults["d_state"] % 48 == 0
    # Constructible at the scaffold's width: d_inner must divide into whole heads.
    SLinOSSConfig(d_model=128, **defaults)


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_the_slinoss_entry_builds_and_runs_at_a_mad_width(precision: str) -> None:
    """The registry's own defaults, built and stepped at the width a task hands it.

    Nothing else here reaches the mixer: the defaults test constructs a config and stops.
    The width is 127, which is what ``icr`` at ``seq_len`` 128 produces -- odd, and not a
    multiple of the chunk. A recurrence that needed a whole chunk would fail on every
    recall arm and pass every test above this one.

    Both precisions run, in the loop's own autocast context. The mixer takes ``x`` in its
    parameters' dtype, so ``bf16`` reaches its kernel through autocast and nowhere else.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    config = TrainConfig(precision=precision, device="cuda")
    mixer = resolve("slinoss").factory(128, 128).cuda()
    x = torch.randn(2, 127, 128, device="cuda")
    with _autocast(config):
        out = mixer(x)
    assert out.shape == x.shape
    assert out.dtype == (torch.float32 if precision == "fp32" else torch.bfloat16)
    out.float().pow(2).sum().backward()
    assert all(param.grad is not None for param in mixer.parameters())


def test_load_module_reports_a_missing_module() -> None:
    """``--mixer-module`` on a path that does not import fails before any arm runs."""
    with pytest.raises(ModuleNotFoundError):
        load_module("scripts.mad.baselines.absent")


def test_attention_sees_the_prefix_and_nothing_after_it() -> None:
    """Perturbing a position moves that position's output and no earlier one.

    A mixer that leaks a later position solves every recall task for the wrong reason,
    and the leak is invisible in the accuracy it produces.
    """
    mixer = CausalAttention(16, 32, n_heads=4).eval()
    x = torch.randn(2, 8, 16)
    moved = x.clone()
    moved[:, 5] += 1.0
    with torch.no_grad():
        before, after = mixer(x), mixer(moved)
    assert before.shape == x.shape
    torch.testing.assert_close(before[:, :5], after[:, :5])
    assert not torch.allclose(before[:, 5], after[:, 5])


def test_attention_runs_without_rotary() -> None:
    """The ablation is a setting, and the scaffold carries no position code.

    So ``rotary=false`` is a position-blind mixer, which is the control for whether a
    task needs positions at all rather than a variant of the mixer.
    """
    mixer = CausalAttention(16, 32, n_heads=4, rotary=False).eval()
    assert isinstance(mixer.rotary, nn.Identity)
    assert mixer(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_attention_refuses_a_head_count_that_does_not_divide() -> None:
    """Silently rounding would give an arm a width its record does not name."""
    with pytest.raises(ValueError, match="does not divide"):
        CausalAttention(16, 32, n_heads=5)


def test_conv_reaches_exactly_d_conv_positions() -> None:
    """The star-free floor: the receptive field is ``d_conv`` and does not grow.

    A task the convolution solves is not testing memory, so the reach has to be the
    stated one -- one tap too many and the floor is no longer a floor.
    """
    taps = 3
    mixer = CausalConv(8, d_conv=taps).eval()
    x = torch.randn(1, 10, 8)
    moved = x.clone()
    moved[:, 4] += 1.0
    with torch.no_grad():
        before, after = mixer(x), mixer(moved)
    assert before.shape == x.shape
    torch.testing.assert_close(before[:, :4], after[:, :4])
    for offset in range(taps):
        assert not torch.allclose(before[:, 4 + offset], after[:, 4 + offset])
    torch.testing.assert_close(before[:, 4 + taps :], after[:, 4 + taps :])


def test_conv_refuses_a_tapless_setting() -> None:
    """Zero taps is a mixer that cannot see its own position."""
    with pytest.raises(ValueError, match="d_conv must be positive"):
        CausalConv(8, d_conv=0)


def test_rotary_turns_pairs_without_changing_their_length() -> None:
    """A rotation preserves the norm, and position zero is the identity."""
    rotary = Rotary(8, 16)
    x = torch.randn(2, 3, 6, 8)
    turned = rotary(x)
    torch.testing.assert_close(turned.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(turned[:, :, 0], x[:, :, 0])


def test_rotary_refuses_an_odd_head_width() -> None:
    """The half split has no pairing at an odd width, so it would fold silently."""
    with pytest.raises(ValueError, match="must be even"):
        Rotary(7, 16)
