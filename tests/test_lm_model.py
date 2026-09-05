"""The swap: that it happens, that it is the only thing that differs, and what it counts.

The whole comparison rests on one factory per block. If construction silently failed, every
arm could be slinoss under another name and the table would read as a set of near-ties. So the
assertions are on identity -- the module in the block is the one the factory made -- rather
than on a loss going down.

The counts are the other half. Arms are matched on non-embedding parameters, so that number has
to be the total less exactly the token table and the head, and the mixer count has to be the
mixers and nothing else. A count that quietly included the shared FFN would make two arms look
matched while their recurrences differed by a factor.

The mixers here are stand-ins: the identity times a gain, with a countable number of
parameters. A real baseline would put its own initialization and its own count between an
assertion and the thing asserted.
"""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from torch import Tensor, nn

from scripts.harness import MixerFactory
from scripts.lm.mixers import REGISTRY
from scripts.lm.model import (
    LMConfig,
    MixerLM,
    MixerResidualBlock,
    build_model,
    layer_factories,
    mixer_parameters,
    non_embedding_parameters,
    parameter_count,
)
from slinoss import SLinOSSConfig, SLinOSSMixer

D_MODEL = 64
N_LAYERS = 3
VOCAB = 64
MAX_LENGTH = 16


class Marked(nn.Module):
    """The identity times a gain, carrying ``rank * d_model`` parameters and a tag.

    Attributes:
        tag: Which factory made it, so a per-layer choice is readable off the built stack.
        gain: ``(rank, d_model)``, broadcast over the sequence.
    """

    def __init__(
        self, d_model: int, max_length: int, *, tag: int = 0, rank: int = 1
    ) -> None:
        del max_length
        super().__init__()
        self.tag = tag
        self.gain = nn.Parameter(torch.ones(rank, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gain.sum(dim=0)


def _factory(tag: int, rank: int = 1) -> MixerFactory:
    """A factory making :class:`Marked` mixers at one tag and rank."""

    def build(d_model: int, max_length: int) -> nn.Module:
        return Marked(d_model, max_length, tag=tag, rank=rank)

    return build


def _config() -> LMConfig:
    """The scaffold config these tests build on."""
    return LMConfig(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB)


def _blocks(model: MixerLM) -> list[MixerResidualBlock]:
    """The stack's blocks, typed."""
    return [block for block in model.blocks if isinstance(block, MixerResidualBlock)]


def _build(factories: list[MixerFactory]) -> MixerLM:
    """A stack with those factories, one per layer."""
    torch.manual_seed(0)
    return build_model(_config(), factories, max_length=MAX_LENGTH)


def test_lm_config_contains_only_scaffold_fields() -> None:
    assert {field.name for field in fields(LMConfig)} == {
        "d_model",
        "n_layers",
        "vocab_size",
        "bias",
        "norm_eps",
        "vocab_pad_multiple",
    }


def test_the_swap_replaces_every_block_s_mixer() -> None:
    """Every block holds the factory's module, and none holds the one it constructed.

    Identity, not shape: a scaffold that built a :class:`slinoss.SLinOSSMixer` and kept it
    would still run, still train, and still report a number under the baseline's name.
    """
    blocks = _blocks(_build(layer_factories(_factory(1), N_LAYERS)))
    assert len(blocks) == N_LAYERS
    for block in blocks:
        assert isinstance(block.mixer, Marked)
        assert block.mixer.tag == 1


def test_the_swapped_mixer_is_on_the_forward_path() -> None:
    """The block calls what was swapped in, not a reference to what it built.

    Checked through the stack's own forward with a mixer whose effect is visible: at a zero
    gain every mixer contributes nothing, so the residual stream at two positions holding the
    same token is the same and so are their logits. A block that kept its own mixer would mix
    across the sequence and the two would differ.
    """
    model = _build(layer_factories(_factory(0), N_LAYERS))
    for block in _blocks(model):
        mixer = block.mixer
        assert isinstance(mixer, Marked)
        with torch.no_grad():
            mixer.gain.zero_()
    ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64)
    with torch.no_grad():
        logits = model(ids)
    assert torch.allclose(logits[0, 0], logits[0, 3], atol=1e-5, rtol=0.0)


def test_the_hybrid_s_last_layer_differs_and_only_the_last() -> None:
    """``final`` is a per-layer choice, so the hybrid needs no second build path."""
    factories = layer_factories(_factory(1), N_LAYERS, final=_factory(2))
    tags = []
    for block in _blocks(_build(factories)):
        mixer = block.mixer
        assert isinstance(mixer, Marked)
        tags.append(mixer.tag)
    assert tags == [1, 1, 2]


def test_a_factory_count_that_is_not_the_depth_is_refused() -> None:
    """A short list would leave the tail of the stack running the scaffold's own mixer."""
    with pytest.raises(ValueError, match="2 factories for 3 layers"):
        _build([_factory(1)] * 2)
    with pytest.raises(ValueError, match="n_layers must be positive"):
        layer_factories(_factory(1), 0)


def test_a_mixer_built_at_another_width_is_refused() -> None:
    """A width mismatch is caught at the swap, not as a shape error mid-run.

    Only checkable for this tree's own mixer, which carries its config. It is also the one
    that can plausibly arrive at a stale width, since its settings come from the registry
    rather than from the scaffold.
    """

    def wrong(d_model: int, max_length: int) -> nn.Module:
        del d_model, max_length
        return SLinOSSMixer(SLinOSSConfig(d_model=2 * D_MODEL, d_state=48, d_head=16))

    with pytest.raises(ValueError, match=f"mixer built at d_model {2 * D_MODEL}"):
        _build(layer_factories(wrong, N_LAYERS))


def test_the_non_embedding_count_is_the_total_less_the_table_and_the_head() -> None:
    """The diagnostic count is a literal subtraction from the published total."""
    model = _build(layer_factories(_factory(1), N_LAYERS))
    embedding = model.embedding
    head = model.head
    assert embedding is not None
    assert head is not None
    table = embedding.weight.numel()
    padded = sum(param.numel() for param in head.parameters())
    assert non_embedding_parameters(model) == parameter_count(model) - table - padded
    assert padded >= VOCAB * D_MODEL


def test_the_mixer_count_is_the_mixers_and_nothing_else() -> None:
    """One gain row per mixer here, so the count is the depth times the width exactly.

    An off-by-one scaffold inclusion would be invisible on a real mixer with thousands of
    parameters, and this count is what separates the recurrence's contribution from the FFN's.
    """
    model = _build(layer_factories(_factory(1), N_LAYERS))
    assert mixer_parameters(model) == N_LAYERS * D_MODEL
    assert mixer_parameters(model) < non_embedding_parameters(model)


def test_two_arms_differ_by_their_mixers_and_by_nothing_else() -> None:
    """The property that makes the sizing search well posed.

    With the scaffold fixed, the parameter count is a function of the mixer and the width
    alone, so a difference between two arms at one width is the mixer's. Checked both ways:
    the non-mixer parameter names agree exactly, and the whole count difference is the mixer
    count difference.
    """
    thin = _build(layer_factories(_factory(1, rank=1), N_LAYERS))
    thick = _build(layer_factories(_factory(1, rank=4), N_LAYERS))
    outside = {
        name for name, _ in thin.named_parameters() if ".mixer." not in f".{name}"
    }
    assert outside == {
        name for name, _ in thick.named_parameters() if ".mixer." not in f".{name}"
    }
    assert mixer_parameters(thick) - mixer_parameters(thin) == 3 * N_LAYERS * D_MODEL
    assert parameter_count(thick) - parameter_count(thin) == mixer_parameters(
        thick
    ) - mixer_parameters(thin)


def test_the_published_gpt_widths_reproduce_the_two_total_parameter_scales() -> None:
    """The paper's 496/1360 widths identify the mixer-only total-count convention."""
    for width, target in ((496, 45_000_000), (1360, 180_000_000)):
        cfg = LMConfig(d_model=width, n_layers=12, vocab_size=32768)
        resolved = REGISTRY.resolve("gpt")
        model = build_model(cfg, layer_factories(resolved.factory, 12), max_length=2048)
        assert parameter_count(model) == pytest.approx(target, rel=0.02)


def test_only_the_token_table_is_stored_in_bfloat16() -> None:
    """The paper's storage exception must not become a bf16 compute path."""
    model = _build(layer_factories(_factory(1), N_LAYERS))
    assert model.embedding.weight.dtype is torch.bfloat16
    assert all(
        param.dtype is torch.float32
        for name, param in model.named_parameters()
        if not name.startswith("embedding.")
    )
    ids = torch.arange(8, dtype=torch.int64).remainder(VOCAB).reshape(2, 4)
    assert model(ids).dtype is torch.float32


def test_the_slinoss_arm_goes_through_the_same_swap() -> None:
    """The arm under test is built from the registry like any baseline.

    A special case for slinoss -- keeping the mixer the block constructed -- would give it the
    scaffold's settings while every baseline got the registry's, which is how a harness stops
    being a comparison. The scaffold is built at another ``d_state`` on purpose: the mixer in
    the block must carry the registry's, not the scaffold's.
    """
    resolved = REGISTRY.resolve("slinoss", ("d_state=144",))
    assert resolved.settings["d_state"] == 144
    for block in _blocks(_build(layer_factories(resolved.factory, N_LAYERS))):
        mixer = block.mixer
        assert isinstance(mixer, SLinOSSMixer)
        assert mixer.config.d_state == 144
