"""The backbone: the block form, initialization ownership, and the two asymmetries.

Three things here would be silent if wrong and would change every number: the residual
stream is carried beside the hidden state rather than added into it, the initialization
pass must not reach a mixer that parameterizes itself, and the rescaled draw must land on
exactly the three name suffixes upstream rescales. Each is pinned against a computation
rather than described.

The mixers used below are a one-position shift and a zero map. Both are causal by
construction and neither has anything to learn, so what fails is the scaffold.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from scripts.mqar.model import (
    RESCALED_SUFFIXES,
    Block,
    Embeddings,
    LanguageModel,
    ModelConfig,
    initialize,
    parameter_count,
    protect,
)

FILL = 0.25
"""Value a stub mixer's weight is filled with. Not a draw the init pass could produce."""


class Shift(nn.Module):
    """A causal mixer: every position reads the one before it, through a projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.proj.weight.fill_(FILL)
            self.proj.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` in and out, the first position reading zeros."""
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return self.proj(shifted)


class Zero(nn.Module):
    """A mixer returning zeros, so the block's residual arithmetic is all that is left."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_model)`` of zeros."""
        return x * self.scale


def config(**kwargs: object) -> ModelConfig:
    """A small backbone: vocabulary 16, width 8, two layers, no dropout."""
    settings: dict[str, object] = {
        "vocab_size": 16,
        "d_model": 8,
        "n_layers": 2,
        "max_length": 4,
        "embed_dropout": 0.0,
        "resid_dropout": 0.0,
    }
    settings.update(kwargs)
    return ModelConfig(**settings)  # pyright: ignore[reportArgumentType]


def shift_factory(d_model: int, layer_idx: int, max_length: int) -> nn.Module:
    """A :class:`Shift`, ignoring the layer index and the length."""
    del layer_idx, max_length
    return Shift(d_model)


def zero_factory(d_model: int, layer_idx: int, max_length: int) -> nn.Module:
    """A :class:`Zero`, ignoring the layer index and the length."""
    del layer_idx, max_length
    return Zero(d_model)


def blocks(model: LanguageModel) -> list[Block]:
    """The block list, narrowed off the module list."""
    narrowed: list[Block] = []
    for layer in model.layers:
        assert isinstance(layer, Block)
        narrowed.append(layer)
    return narrowed


def test_model_scores_every_position() -> None:
    """``(B, T)`` int64 in, ``(B, T, vocab_size)`` out."""
    model = LanguageModel(config(), shift_factory)
    ids = torch.randint(0, 16, (3, 4))
    assert model(ids).shape == (3, 4, 16)


def test_nothing_but_the_mixer_crosses_positions() -> None:
    """With a causal mixer the backbone is causal, so a later token moves nothing earlier.

    Every other part of the block is position-wise, and a norm that mixed positions or a
    residual that ran backwards would solve recall for the wrong reason.
    """
    model = LanguageModel(config(), shift_factory).eval()
    ids = torch.randint(0, 16, (2, 4))
    changed = ids.clone()
    changed[:, -1] = (ids[:, -1] + 1) % 16
    with torch.no_grad():
        before, after = model(ids), model(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1], atol=1e-6)


def test_residual_is_carried_beside_the_hidden_state() -> None:
    """The one place the block departs from the usual pre-norm form.

    A layer norm never sees its own output added back: the mixer's output enters the
    residual through the *next* block's first dropout, and the hidden state handed forward
    is a norm of the residual rather than a sum. Written out for a zero mixer and an
    identity state mixer, the whole backbone is the composition below.
    """
    model = LanguageModel(config(), zero_factory).eval()
    ids = torch.randint(0, 16, (2, 4))
    with torch.no_grad():
        residual = model.embeddings(ids)
        for layer in blocks(model):
            residual = layer.norm2(residual) + residual
        expected = model.lm_head(model.ln_f(residual))
        assert torch.allclose(model(ids), expected, atol=1e-6)


@pytest.mark.parametrize("protected", [True, False])
def test_protect_keeps_a_mixers_own_initialization(protected: bool) -> None:
    """A mixer that parameterizes itself is measured with that parameterization, or not.

    Upstream has no opt-out, so its blanket draw overwrites whatever a mixer set up. The
    unprotected half of this test is that behaviour, kept reachable because the two
    controls are measured under it.
    """

    def factory(d_model: int, layer_idx: int, max_length: int) -> nn.Module:
        built = shift_factory(d_model, layer_idx, max_length)
        return protect(built) if protected else built

    model = LanguageModel(config(), factory)
    mixer = blocks(model)[0].sequence_mixer
    assert isinstance(mixer, Shift)
    kept = bool(
        torch.equal(mixer.proj.weight, torch.full_like(mixer.proj.weight, FILL))
    )
    assert kept is protected


@pytest.mark.parametrize(
    ("init_type", "row_norm", "component_std"),
    [
        ("default", 0.02 * math.sqrt(64), 0.02),
        ("spherical", 1.0, 1.0 / math.sqrt(64)),
        ("normal", math.sqrt(64), 1.0),
    ],
)
def test_the_word_draw_reaches_the_table_and_survives_the_init_pass(
    init_type: str, row_norm: float, component_std: float
) -> None:
    """Each draw at its own scale, then untouched by the blanket normal pass.

    The scales differ by up to 50x, so a table the pass reached would be at ``init_std``
    whatever was asked for, and the mark is the only thing preventing that. The head is
    tied to whatever the table ended up holding.
    """
    shape = config(vocab_size=256, d_model=64, embedding_init_type=init_type)
    model = LanguageModel(shape, shift_factory)
    weight = model.embeddings.word_embeddings.weight
    table = weight.detach()
    assert float(table.norm(p=2, dim=1).mean()) == pytest.approx(row_norm, rel=0.05)
    assert float(table.std()) == pytest.approx(component_std, rel=0.05)
    assert model.lm_head.weight is weight


def test_a_drawn_table_is_taken_before_the_position_table() -> None:
    """Where upstream takes it, which fixes what every later draw gets.

    Reproduced by hand: the word table, then the spherical draw off the same generator,
    then the position table. A draw applied after construction would leave the position
    table holding the numbers asserted here for the word table.
    """
    torch.manual_seed(11)
    words = nn.Embedding(16, 8)
    vectors = torch.randn_like(words.weight)
    positions = nn.Embedding(4, 8)
    torch.manual_seed(11)
    built = Embeddings(
        config(embedding_init_type="spherical", max_position_embeddings=4)
    )
    table = built.position_embeddings
    assert table is not None
    expected = vectors / torch.norm(vectors, p=2, dim=1, keepdim=True)
    assert torch.allclose(built.word_embeddings.weight, expected, atol=1e-6)
    assert torch.allclose(built.word_embeddings.weight.norm(p=2, dim=1), torch.ones(16))
    assert torch.equal(table.weight, positions.weight)


@pytest.mark.parametrize("suffix", RESCALED_SUFFIXES)
def test_rescaled_suffixes_take_the_depth_draw(suffix: str) -> None:
    """``init_std / sqrt(2 * n_layers)`` on these three names and nowhere else.

    ``output_linear.0.weight`` names no module in this port; it is upstream's, and it stays
    in the list because a baseline slotted in from outside may carry it. The suffix is
    matched against the full parameter path, so a nested one has to work.
    """
    width = 128
    rescaled = nn.Linear(width, width, bias=False)
    plain = nn.Linear(width, width, bias=False)
    holder = nn.Module()
    if suffix == "output_linear.0.weight":
        holder.output_linear = nn.Sequential(rescaled)
    elif suffix == "out_proj.weight":
        holder.out_proj = rescaled
    else:
        holder.fc2 = rescaled
    holder.plain = plain
    initialize(holder, n_layers=2, init_std=0.02)
    assert float(plain.weight.detach().std()) == pytest.approx(0.02, rel=0.05)
    assert float(rescaled.weight.detach().std()) == pytest.approx(0.01, rel=0.05)


def test_a_tied_head_is_counted_once() -> None:
    """One weight, tied after the init pass so the embedding's draw is what both use."""
    model = LanguageModel(config(), shift_factory)
    assert model.lm_head.weight is model.embeddings.word_embeddings.weight
    duplicated = sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )
    assert duplicated - parameter_count(model) == model.lm_head.weight.numel()


def test_untied_embeddings_freeze_the_table_and_keep_a_separate_head() -> None:
    """The scaffold ablation: a frozen table, an untied head, the same trainable count."""
    model = LanguageModel(config(learnable_word_embeddings=False), shift_factory)
    assert model.lm_head.weight is not model.embeddings.word_embeddings.weight
    assert not model.embeddings.word_embeddings.weight.requires_grad
    assert parameter_count(model) == parameter_count(
        LanguageModel(config(), shift_factory)
    )


def test_mlp_state_mixer_adds_the_scaffold_the_published_configs_leave_out() -> None:
    """Every published MQAR config runs the identity, which is what isolates the mixer.

    ``mlp`` is upstream's own default block, kept for the ablation that separates what the
    scaffold contributes from what the recurrence does.
    """
    width, layers = 8, 2
    identity = LanguageModel(config(), shift_factory)
    scaffolded = LanguageModel(config(state_mixer="mlp"), shift_factory)
    per_layer = 8 * width * width + 5 * width
    assert parameter_count(scaffolded) - parameter_count(identity) == layers * per_layer
    assert isinstance(blocks(identity)[0].state_mixer, nn.Identity)


def test_positions_are_optional_and_follow_the_inputs_device() -> None:
    """A recurrent mixer is measured with no position embedding; attention is not.

    Upstream hardcodes ``device='cuda'`` here, so its backbone cannot run on CPU at all.
    """
    ids = torch.full((1, 2), 5, dtype=torch.long)
    without = LanguageModel(config(), shift_factory)
    assert without.embeddings.position_embeddings is None
    flat = without.embeddings(ids)
    assert torch.equal(flat[0, 0], flat[0, 1])
    with_positions = LanguageModel(config(max_position_embeddings=4), shift_factory)
    table = with_positions.embeddings.position_embeddings
    assert table is not None
    assert table.weight.shape == (4, 8)
    placed = with_positions.embeddings(ids)
    assert placed.device.type == "cpu"
    assert not torch.equal(placed[0, 0], placed[0, 1])


def test_dropout_is_the_embedding_rate_at_layer_zero_only() -> None:
    """Layer 0's first dropout is the embedding's; every other dropout is the residual's.

    Both published configs run ``embed_dropout`` 0.1 and ``resid_dropout`` 0, so this is
    the only dropout in the model and reading it as a residual rate would delete it.
    """
    model = LanguageModel(config(embed_dropout=0.1, resid_dropout=0.0), shift_factory)
    assert [layer.dropout1.p for layer in blocks(model)] == [0.1, 0.0]
    assert [layer.dropout2.p for layer in blocks(model)] == [0.0, 0.0]
    assert model.drop_f.p == 0.0


def test_only_the_final_norm_takes_the_configured_epsilon() -> None:
    """Upstream passes ``layer_norm_epsilon`` to ``ln_f`` and leaves the block norms alone."""
    model = LanguageModel(config(layer_norm_epsilon=1e-3), shift_factory)
    assert model.ln_f.eps == 1e-3
    assert [layer.norm1.eps for layer in blocks(model)] == [1e-5, 1e-5]
    assert [layer.norm2.eps for layer in blocks(model)] == [1e-5, 1e-5]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 0}, "d_model must be positive"),
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
        ({"max_length": 0}, "max_length must be positive"),
        ({"state_mixer": "glu"}, "state_mixer must be one of"),
        ({"embedding_init_type": "uniform"}, "embedding_init_type must be one of"),
        ({"max_position_embeddings": 2}, "is shorter than max_length"),
    ],
)
def test_out_of_contract_shapes_raise(kwargs: dict[str, object], message: str) -> None:
    """Caught at the config, before a parameter is allocated."""
    with pytest.raises(ValueError, match=message):
        config(**kwargs)
