"""What an arm's name resolves to on this axis, and whether its defaults fit this axis's widths.

The generic registry machinery -- coercion, unknown settings, re-registration -- is covered in
``tests/test_harness_registry.py``. What is specific here is that this axis runs *narrow*. The
published configs use ``hidden_dim`` 16, 64 and 128, and a default head width or head count
carried over from a language-modelling registry does not divide 16. That failure arrives inside
:func:`scripts.tsc.model.build_model`'s probe, which is early, but only if some test actually
builds every in-tree mixer at every published width. This one does.

:func:`scripts.tsc.mixers.paper_overrides` is the other axis-specific piece: the published
per-dataset ``ssm_dim`` is an oscillator count and reaches the reference recurrence alone. Mapping
it onto another mixer's state width would make that baseline's capacity a harness choice, so the
test pins the empty list for every other name as firmly as it pins the value for the two.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch import Tensor, nn

from scripts.tsc.linoss import LinOSSRecurrence
from scripts.tsc.mixers import REGISTRY, Unwrap, paper_overrides
from scripts.tsc.model import ModelConfig, build_model
from scripts.tsc.protocol import DATASETS, setting_for
from slinoss import SLinOSSConfig

REGISTERED = (
    "attention",
    "conv",
    "gdn2",
    "linoss_im",
    "linoss_imex",
    "mamba2",
    "mamba3",
    "slinoss",
)

IN_TREE = ("attention", "conv", "linoss_im", "linoss_imex", "slinoss")
"""The mixers whose packages are this tree, so a test may build them."""

ON_CARD = ("slinoss",)
"""Of those, the ones whose operator refuses a host tensor.

The ``so3ssd`` reference path checks its operands are on a CUDA device, so this one's width probe
needs a card. The probe *is* the check here, which is why the requirement is carried as a marker
rather than worked around."""

ON_HOST = tuple(name for name in IN_TREE if name not in ON_CARD)
"""The rest. A name added to :data:`IN_TREE` lands here without an edit."""

WIDTHS = (16, 64, 128)
"""The ``hidden_dim`` values the six published configs use."""


def build_at(name: str, width: int, device: str) -> None:
    """Build one mixer into a one-block scaffold and run the scaffold's forward.

    Args:
        name: Registry name.
        width: The scaffold's ``hidden_dim``.
        device: Destination for the model and the input.
    """
    resolved = REGISTRY.resolve(name)
    config = ModelConfig(input_dim=2, hidden_dim=width, classes=3, blocks=1)
    model = build_model(config, [resolved.factory], max_length=8, device=device)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 8, 2, device=device))
    assert out.shape == (2, 3), (name, width, device)


class _Tuple(nn.Module):
    """A mixer in the shape the linear-attention layers return."""

    def forward(self, x: Tensor) -> tuple[Tensor, None, None]:
        return x * 2.0, None, None


def test_every_name_the_axis_offers_is_registered() -> None:
    """The eight arms, exactly. An extra one is dead weight; a missing one is a typo at launch."""
    assert REGISTRY.names() == list(REGISTERED)


@pytest.mark.parametrize("width", WIDTHS)
def test_every_host_mixer_builds_at_every_published_width(width: int) -> None:
    """A default head width or head count that does not divide 16 is what this catches.

    The build goes through :func:`scripts.tsc.model.build_model`, so each mixer's shape is probed
    the way an arm probes it rather than by constructing the layer alone.
    """
    for name in ON_HOST:
        build_at(name, width, "cpu")


@pytest.mark.cuda
@pytest.mark.parametrize("width", WIDTHS)
def test_the_axiss_own_mixer_builds_at_every_published_width(width: int) -> None:
    """``slinoss`` is the reason the previous test exists, and it needs a card to run at all.

    Its config's ``d_head`` is 64, which at ``hidden_dim`` 16 asks for a quarter of a head. The
    registry narrows it to 16 and this test is what holds that narrowing in place. It is also the
    only place the axis builds the CUDA-only operator through the scaffold, so it covers
    :func:`scripts.tsc.model.build_model` probing on the destination device instead of the host.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    for name in ON_CARD:
        build_at(name, width, "cuda")


def test_a_baselines_settings_resolve_without_its_package() -> None:
    """Planning a sweep must not require every arm's package on the planning host.

    The import is inside the build, so a lattice covering mamba3 can be enumerated, sharded and
    validated against the defaults on a host that cannot run it.
    """
    for name in ("mamba2", "mamba3", "gdn2"):
        resolved = REGISTRY.resolve(name)
        assert resolved.name == name
        assert resolved.settings


def test_the_slinoss_defaults_are_config_fields_at_the_configs_own_values() -> None:
    """Read off :class:`slinoss.SLinOSSConfig`, except the two this axis has to choose.

    A default named here that is not a field raises at construction. ``d_state`` has no default
    on the config and must be a positive multiple of 48 for the kernels; ``d_head`` is narrowed
    from 64 to fit ``hidden_dim`` 16, which is the whole reason this entry restates anything.
    """
    settings = REGISTRY.resolve("slinoss").settings
    assert set(settings) <= {field.name for field in fields(SLinOSSConfig)}
    assert "d_model" not in settings
    for key, value in settings.items():
        if key in {"d_state", "d_head"}:
            continue
        assert value == getattr(SLinOSSConfig, key), key
    assert settings["d_state"] > 0 and settings["d_state"] % 48 == 0
    assert settings["d_head"] == 16
    # The narrowing is a departure from the config and has to stay visible as one.
    assert SLinOSSConfig.d_head != settings["d_head"]


def test_the_published_state_width_reaches_the_reference_recurrence_alone() -> None:
    """``ssm_dim`` for the two ``linoss`` names, nothing for the other six.

    An oscillator count is not another mixer's state width. Translating it silently would put a
    baseline's capacity under the harness's control while the run record still read like the
    protocol's, which is the one way a comparison here can be wrong and look right.
    """
    for dataset in DATASETS:
        setting = setting_for(dataset)
        for name in ("linoss_im", "linoss_imex"):
            assert paper_overrides(name, setting) == [f"ssm_dim={setting.ssm_dim}"]
            resolved = REGISTRY.resolve(name, paper_overrides(name, setting))
            assert resolved.settings["ssm_dim"] == setting.ssm_dim
        for name in set(REGISTERED) - {"linoss_im", "linoss_imex"}:
            assert paper_overrides(name, setting) == []


def test_the_two_linoss_entries_differ_only_in_their_scheme() -> None:
    """The name carries the discretization, so the two arms cannot be confused in a record.

    Both names take ``ssm_dim`` and nothing else; a scheme reachable through ``--set`` would let
    one name report the other's numbers.
    """
    for name, scheme in (("linoss_im", "IM"), ("linoss_imex", "IMEX")):
        resolved = REGISTRY.resolve(name)
        assert set(resolved.settings) == {"ssm_dim"}
        layer = resolved.factory(4, 8)
        assert isinstance(layer, LinOSSRecurrence)
        assert layer.discretization == scheme


def test_unwrap_takes_the_first_output_and_passes_a_bare_tensor_through() -> None:
    """One wrapper for both shapes, so no baseline needs its own adapter in the scaffold."""
    x = torch.ones(2, 3, 4)
    assert torch.equal(Unwrap(_Tuple())(x), x * 2.0)
    assert torch.equal(Unwrap(nn.Identity())(x), x)
    assert [name for name, _ in Unwrap(nn.Linear(4, 4)).named_parameters()] == [
        "inner.weight",
        "inner.bias",
    ]
