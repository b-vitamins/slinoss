"""The registry, the per-layer cycle, and the two controls the suite is read against.

A registry defect is silent in the worst way: an override the mixer never saw, or a default
that drifted off the config it claims to track, changes what an arm measured without
changing what its record says. So the surface is pinned here.

The cycle is upstream's ``Hybrid``, and it is not a detail. Every architecture in zoology's
current MQAR config is a short convolution followed by the architecture under test, so a
slinoss number compared against those is only comparable if it is built the same way.

The controls are held to the properties that make them controls: attention sees the whole
prefix and nothing after it, the convolution sees ``kernel_size`` positions and nothing
before them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from scripts.mqar.mixers import (
    REGISTRY,
    CausalAttention,
    CausalConv,
    Setting,
    load_module,
    register,
    resolve,
    settings_from,
)
from scripts.mqar.model import LanguageModel, ModelConfig
from scripts.mqar.train import TrainConfig, _autocast

STUB_DEFAULTS: dict[str, Setting] = {
    "mode": "wide",
    "taps": 4,
    "gain": 2.0,
    "gated": True,
}
"""One setting of each type an override string is coerced to."""

SLINOSS_DEFAULTS: dict[str, Setting] = {
    "d_state": 144,
    "expand": 2.0,
    "d_head": 64,
    "n_groups": 1,
    "chunk_size": 64,
    "d_conv": 4,
    "key_conv": True,
    "init_span": 4096,
    "w_max": 3.141592502593994,
    "bias": False,
    "conv_bias": True,
}
"""The slinoss entry's declared settings, read off ``SLinOSSConfig`` at import.

Pinned as literals rather than against the dataclass, because the dataclass is what the
entry reads: a change there is a change to every arm's shape and has to be acknowledged
here. ``n_groups`` 1 is also the Mamba2 parity setting.
"""


class Stub(nn.Module):
    """A mixer that records the settings it was built with and returns its input."""

    def __init__(self, d_model: int, **settings: Any) -> None:
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` unchanged."""
        return x


def build_stub(d_model: int, **settings: Setting) -> nn.Module:
    """Build a :class:`Stub` with no context dependency."""
    return Stub(d_model, **settings)


@contextmanager
def registered(name: str, defaults: dict[str, Setting] | None = None) -> Iterator[None]:
    """Register a stub for the duration of one test, through the public entry point."""
    register(
        name,
        build_stub,
        STUB_DEFAULTS if defaults is None else defaults,
        layer_index_policy="unused",
        max_length_policy="unused",
    )
    try:
        yield
    finally:
        REGISTRY.pop(name, None)


def test_the_cycle_keys_per_layer_not_per_model() -> None:
    """``names[layer_idx % len(names)]``, which is upstream's ``Hybrid``.

    A cycle shorter than the model repeats, so ``conv slinoss`` at four layers is two
    hybrid pairs and not a conv followed by three of anything.
    """
    mixer = resolve(["conv", "attention"])
    assert mixer.name == "conv+attention"
    built = [mixer.factory(8, index, 16) for index in range(3)]
    assert [type(module) for module in built] == [
        CausalConv,
        CausalAttention,
        CausalConv,
    ]


def test_the_cycle_reaches_the_backbone_layer_by_layer() -> None:
    """The factory a model is built from is the one the registry resolved.

    Which is what makes ``--mixer conv slinoss`` a hybrid rather than a label.
    """
    config = ModelConfig(vocab_size=16, d_model=8, n_layers=3, max_length=4)
    model = LanguageModel(config, resolve(["conv", "attention"]).factory)
    assert [type(layer.sequence_mixer) for layer in model.layers] == [
        CausalConv,
        CausalAttention,
        CausalConv,
    ]


def test_settings_reach_the_module_and_land_in_the_record() -> None:
    """Every declared setting is passed, defaults included, and reported as resolved."""
    with registered("stub"):
        mixer = resolve(["stub"], ["taps=9"])
        assert mixer.settings == {"stub": {**STUB_DEFAULTS, "taps": 9}}
        built = mixer.factory(8, 0, 16)
        assert isinstance(built, Stub)
        assert built.settings == {**STUB_DEFAULTS, "taps": 9}
        assert mixer.constructions[0]["context"] == {
            "layer_index_supplied": 0,
            "layer_index_policy": "unused",
            "layer_index_consumed": None,
            "max_length_supplied": 16,
            "max_length_policy": "unused",
            "max_length_consumed": None,
        }


def test_context_policies_are_mandatory_and_consumed_only_when_declared() -> None:
    """Constructor context cannot disappear behind a positional compatibility shim."""
    with pytest.raises(TypeError):
        register("missing-policy", build_stub, {})  # type: ignore[call-arg]

    seen: list[tuple[int, int]] = []

    def contextual(d_model: int, layer_idx: int, max_length: int) -> nn.Module:
        seen.append((layer_idx, max_length))
        return Stub(d_model)

    register(
        "contextual",
        contextual,
        {},
        layer_index_policy="required",
        max_length_policy="required",
    )
    try:
        mixer = resolve(["contextual"])
        mixer.factory(8, 3, 16)
        assert seen == [(3, 16)]
        context = mixer.constructions[0]["context"]
        assert context["layer_index_consumed"] == 3
        assert context["max_length_consumed"] == 16
        with pytest.raises(ValueError, match="layer_idx must be non-negative"):
            mixer.factory(8, -1, 16)
        with pytest.raises(ValueError, match="max_length must be positive"):
            mixer.factory(8, 0, 0)
    finally:
        REGISTRY.pop("contextual", None)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ("mode=narrow", "narrow"),
        ("taps=9", 9),
        ("gain=0.5", 0.5),
        ("gated=false", False),
        ("gated=off", False),
        ("gated=1", True),
    ],
)
def test_an_override_is_typed_by_its_default(override: str, expected: Setting) -> None:
    """bool is read before int, because bool subclasses it and ``bool("false")`` is True."""
    key = override.split("=", 1)[0]
    with registered("stub"):
        assert settings_from("stub", [override])[key] == expected


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("taps=wide", "taps is int"),
        ("gain=wide", "gain is float"),
        ("gated=maybe", "gated is a flag"),
        ("taps", "not key=value"),
    ],
)
def test_a_malformed_override_names_the_setting(override: str, message: str) -> None:
    """The alternative is int's own message with no mention of which flag produced it."""
    with registered("stub"), pytest.raises(ValueError, match=message):
        settings_from("stub", [override])


def test_an_undeclared_setting_is_an_error_not_a_silent_extra() -> None:
    """A typo that is ignored is an arm that ran at a setting nobody chose."""
    with registered("stub"), pytest.raises(KeyError, match="has no setting depth"):
        settings_from("stub", ["depth=3"])


@pytest.mark.parametrize(
    ("names", "overrides", "kind", "message"),
    [
        ([], (), ValueError, "at least one name"),
        (["nope"], (), KeyError, "no mixer nope"),
        (["stub", "conv"], ("taps=9",), ValueError, "ambiguous"),
        (["stub"], ("conv.kernel_size=5",), KeyError, "not in the cycle"),
    ],
)
def test_resolve_rejects_an_unreadable_request(
    names: list[str], overrides: tuple[str, ...], kind: type[Exception], message: str
) -> None:
    """An unscoped override under several entries is ambiguous, not applied to all."""
    with registered("stub"), pytest.raises(kind, match=message):
        resolve(names, overrides)


def test_a_scoped_override_reaches_only_its_entry() -> None:
    """``entry.key=value``, which is how a hybrid is configured on one side."""
    with registered("stub"):
        mixer = resolve(["stub", "conv"], ["stub.taps=9", "conv.kernel_size=5"])
        assert mixer.settings["stub"]["taps"] == 9
        assert mixer.settings["conv"]["kernel_size"] == 5


@pytest.mark.parametrize("name", ["conv", "with.dot", "with+plus"])
def test_register_rejects_a_taken_or_reserved_name(name: str) -> None:
    """``.`` and ``+`` are the override and cycle syntaxes; a collision is silent."""
    with pytest.raises((KeyError, ValueError), match="mixer"):
        register(
            name,
            build_stub,
            {},
            layer_index_policy="unused",
            max_length_policy="unused",
        )


def test_a_baseline_slots_in_from_outside_the_tree(tmp_path: Path) -> None:
    """``--mixer-module`` imports a file so its ``register`` call runs.

    This is how a baseline arrives: one module, one call, no optional dependency imported
    at registry scope.
    """
    module = tmp_path / "outside.py"
    module.write_text(
        "from torch import nn\n"
        "from scripts.mqar.mixers import register\n"
        "def build(d_model, **settings):\n"
        "    return nn.Identity()\n"
        'register("outside", build, {"width": 3}, '
        'layer_index_policy="unused", max_length_policy="unused")\n',
        encoding="utf-8",
    )
    try:
        load_module(str(module))
        assert resolve(["outside"]).settings == {"outside": {"width": 3}}
    finally:
        REGISTRY.pop("outside", None)
    with pytest.raises(FileNotFoundError, match="no mixer module"):
        load_module(str(tmp_path / "absent.py"))


def test_the_published_control_defaults() -> None:
    """Attention's defaults are the figure-2 sweep's, not ``MHA``'s own.

    Dropout 0.1 rather than 0, which both published configs pass, and one head. The modern
    reproduction runs two heads, which is one override away.
    """
    assert REGISTRY["attention"].defaults == {
        "num_heads": 1,
        "bias": True,
        "dropout": 0.1,
    }
    assert REGISTRY["conv"].defaults == {"kernel_size": 3}
    assert all(
        entry.layer_index_policy == entry.max_length_policy == "unused"
        for entry in REGISTRY.values()
    )


def test_the_slinoss_entry_declares_the_operator_contract() -> None:
    """Every setting, at the value ``SLinOSSConfig`` declares."""
    assert REGISTRY["slinoss"].defaults == SLINOSS_DEFAULTS


def test_attention_reads_the_whole_prefix_and_nothing_after_it() -> None:
    """The recall ceiling. A leak here would make every recall number meaningless."""
    mixer = CausalAttention(d_model=8, num_heads=2, bias=True, dropout=0.0).eval()
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, -1] += 1.0
    with torch.no_grad():
        before, after = mixer(x), mixer(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-6)
    with torch.no_grad():
        early = mixer(x[:, :3])
    assert torch.allclose(early, before[:, :3], atol=1e-5)


def test_the_convolution_reads_its_own_kernel_and_no_further() -> None:
    """The star-free floor: ``kernel_size`` positions, left-padded, truncated.

    A task attention solves and this does not is a recall task; one both solve is
    measuring something other than recall.
    """
    mixer = CausalConv(d_model=4, kernel_size=3).eval()
    x = torch.randn(1, 6, 4)
    changed = x.clone()
    changed[:, 0] += 1.0
    with torch.no_grad():
        before, after = mixer(x), mixer(changed)
    assert not torch.allclose(before[:, :3], after[:, :3], atol=1e-6)
    assert torch.allclose(before[:, 3:], after[:, 3:], atol=1e-6)


def test_the_convolution_draws_its_projection_before_its_kernel() -> None:
    """Construction order decides which draw off the global generator each parameter gets.

    Upstream builds the projection first. The distribution is the same either way; the
    weights are not, and a comparison at equal seed against upstream's is.
    """
    torch.manual_seed(7)
    mixer = CausalConv(d_model=4, kernel_size=3)
    torch.manual_seed(7)
    first = nn.Linear(4, 4)
    assert torch.equal(mixer.projection.weight, first.weight)


def test_the_convolution_refuses_the_unported_long_variant() -> None:
    """Upstream selects a long convolution at ``kernel_size`` -1. No MQAR config does."""
    with pytest.raises(ValueError, match="long-convolution"):
        CausalConv(d_model=4, kernel_size=-1)


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_the_slinoss_entry_builds_and_steps_at_an_mqar_width(precision: str) -> None:
    """The registry's own defaults, built and stepped at a width the driver hands it.

    Nothing above this reaches the mixer: the defaults test compares a dictionary. Both
    precisions run in the loop's own autocast context, because the mixer takes ``x`` at its
    parameters' dtype and reaches bf16 there and nowhere else.

    The mixer must also carry the protection mark, or the backbone's blanket normal draw
    overwrites its parameterization and the arm measures a shape.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    mixer = resolve(["slinoss"]).factory(128, 0, 64).cuda()
    assert getattr(mixer, "_no_reinit", False) is True
    x = torch.randn(2, 64, 128, device="cuda")
    with _autocast(TrainConfig(precision=precision, device="cuda")):
        out = mixer(x)
    assert out.shape == x.shape
    assert out.dtype == (torch.float32 if precision == "fp32" else torch.bfloat16)
    out.float().pow(2).sum().backward()
    assert all(parameter.grad is not None for parameter in mixer.parameters())
