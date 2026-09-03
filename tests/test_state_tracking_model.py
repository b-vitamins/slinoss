"""The scaffold: the block ordering, the parameter names, and the mixer boundary.

The scaffold is what makes two arms comparable, so its failure modes are all of the kind
that leave a run working and the comparison void.

The block is post-norm. Moving the norm to the front of the block changes what a
recurrence's output can do to the residual stream -- under post-norm it is rescaled after
every block and cannot dominate the skip by growing -- and every published number on this
axis was measured post-norm.

The embedding parameter is named ``embedding``. The optimizer's two weight-decay groups
split on that substring appearing in a parameter's name, so renaming the attribute moves
the token table from zero decay to 1e-2 with nothing failing.

The factory is called once per layer. A factory returning a shared module would tie the
layers' weights, halving the mixer's parameter count while the record still reports
``n_layers`` blocks.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from scripts.state_tracking.model import (
    Block,
    MixerFactory,
    ModelConfig,
    StateTracker,
    build_model,
    device_of,
    mixer_parameters,
    parameter_count,
)

D_MODEL = 16
VOCAB = 7
MAX_LENGTH = 32


def _linear_factory(calls: list[tuple[int, int]]) -> MixerFactory:
    """A mixer that is one bias-free linear, recording what it was built with.

    Args:
        calls: Appended to once per call, with the factory's arguments.

    Returns:
        The factory. Its parameter count is ``d_model**2`` per layer, which is what the
        mixer-boundary tests count.
    """

    def factory(d_model: int, max_length: int) -> nn.Module:
        calls.append((d_model, max_length))
        return nn.Linear(d_model, d_model, bias=False)

    return factory


def _block(model: StateTracker, index: int) -> Block:
    """One block, typed. ``nn.ModuleList`` indexing returns a bare ``Module``."""
    return cast(Block, model.blocks[index])


def test_block_is_post_norm() -> None:
    """The block is ``norm(mixer(x) + x)``, then dropout -- not ``mixer(norm(x)) + x``.

    Asserted against the block's own norm applied to the residual sum, so a reordering
    fails here rather than showing up as a different number a training run later.
    """
    block = Block(nn.Identity(), D_MODEL, dropout=0.0, use_glu=False)
    block.eval()
    x = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        got = block(x)
        expected = block.norm(x + x)
    assert torch.allclose(got, expected, atol=0.0)
    assert isinstance(block.norm, nn.LayerNorm)
    assert isinstance(block.drop, nn.Dropout)
    assert block.post_linear is None


def test_block_glu_branch_is_a_second_residual() -> None:
    """With ``use_glu`` the block adds ``glu(post_linear(y))`` before the norm.

    The projection is ``d_model -> 2 * d_model`` because glu halves it again; a projection
    to ``d_model`` would halve the stream and fail on the add.
    """
    block = Block(nn.Identity(), D_MODEL, dropout=0.0, use_glu=True)
    block.eval()
    assert isinstance(block.post_linear, nn.Linear)
    assert block.post_linear.out_features == 2 * D_MODEL
    x = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got = block(x)
        gated = nn.functional.glu(block.post_linear(x + x), dim=-1)
        expected = block.norm(x + x + gated)
    assert torch.allclose(got, expected, atol=0.0)


def test_dropout_is_the_last_thing_in_the_block() -> None:
    """Dropout follows the norm, so its zeros survive into the next block.

    A dropout applied before the norm leaves no exact zeros in the block's output, since
    the norm shifts by the mean. That is the discriminant used here, rather than the eval
    path, which cannot tell the two orderings apart.
    """
    block = Block(nn.Identity(), D_MODEL, dropout=0.5, use_glu=False)
    x = torch.randn(2, 5, D_MODEL, generator=torch.Generator().manual_seed(2))
    block.train()
    torch.manual_seed(0)
    with torch.no_grad():
        dropped = block(x)
    assert bool((dropped == 0.0).any())
    block.eval()
    with torch.no_grad():
        assert torch.allclose(block(x), block.norm(x + x), atol=0.0)
    assert isinstance(block.drop, nn.Dropout)
    assert block.drop.p == 0.5


def test_model_shape_and_layer_count() -> None:
    """Logits at every position, over the one shared vocabulary.

    Every position, not just the last: a group task supervises the whole trajectory and
    the loss selects with the batch's mask.
    """
    calls: list[tuple[int, int]] = []
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=3, dropout=0.0)
    model = build_model(config, _linear_factory(calls))
    assert isinstance(model, StateTracker)
    assert len(model.blocks) == 3
    assert calls == [(D_MODEL, MAX_LENGTH)] * 3
    out = model(torch.zeros(2, 5, dtype=torch.long))
    assert out.shape == (2, 5, VOCAB)
    assert model.head.out_features == VOCAB
    assert model.embedding.num_embeddings == VOCAB


def test_layers_do_not_share_a_mixer() -> None:
    """Each block holds its own mixer instance.

    A factory that returned one module would tie the layers. The parameter count is the
    only visible symptom, and it is reported per arm rather than compared to a reference.
    """
    calls: list[tuple[int, int]] = []
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=2)
    model = build_model(config, _linear_factory(calls))
    assert _block(model, 0).mixer is not _block(model, 1).mixer
    assert mixer_parameters(model) == 2 * D_MODEL * D_MODEL


def test_only_the_token_table_carries_the_embedding_name() -> None:
    """Exactly the embedding weight matches the optimizer's ``embedding`` substring.

    The zero-decay group is selected by that substring. A scaffold parameter that happened
    to contain it would silently stop being decayed, and the token table renamed would
    silently start being decayed.
    """
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=2)
    model = build_model(config, _linear_factory([]))
    matched = [name for name, _ in model.named_parameters() if "embedding" in name]
    assert matched == ["embedding.weight"]


def test_glu_costs_one_projection_per_block() -> None:
    """``use_glu`` adds exactly ``d_model * 2 * d_model + 2 * d_model`` per block.

    The flag is off in every published config on this axis, so the delta is stated here
    rather than left to be discovered when a parameter count moves.
    """
    plain = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=2)
    gated = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=2, use_glu=True)
    assert plain.use_glu is False
    gated_count = parameter_count(build_model(gated, _linear_factory([])))
    plain_count = parameter_count(build_model(plain, _linear_factory([])))
    assert gated_count - plain_count == 2 * (D_MODEL * 2 * D_MODEL + 2 * D_MODEL)


def test_parameter_count_excludes_frozen_parameters() -> None:
    """Only trainable parameters are counted, and mixer parameters are a subset.

    A mixer that freezes part of itself -- a fixed positional table, a buffer promoted to
    a parameter -- would otherwise inflate both numbers and void a matched-parameter
    claim.
    """
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=2)
    model = build_model(config, _linear_factory([]))
    total = parameter_count(model)
    assert 0 < mixer_parameters(model) < total
    for index in (0, 1):
        for param in _block(model, index).mixer.parameters():
            param.requires_grad_(False)
    assert mixer_parameters(model) == 0
    assert parameter_count(model) == total - 2 * D_MODEL * D_MODEL


def test_no_initialization_pass_touches_a_default() -> None:
    """The scaffold's parameters are torch defaults, drawn under the caller's seed.

    No upstream tree on this axis initializes, so the same seed must give the same
    parameters as constructing the modules directly in the same order. An initialization
    pass added here would move every published comparison.
    """
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=1)
    torch.manual_seed(11)
    model = build_model(config, _linear_factory([]))
    torch.manual_seed(11)
    assert torch.allclose(
        model.embedding.weight, nn.Embedding(VOCAB, D_MODEL).weight, atol=0.0
    )
    norm = _block(model, 0).norm
    assert torch.all(norm.weight == 1.0)
    assert torch.all(norm.bias == 0.0)


def test_model_config_validation() -> None:
    """A malformed shape is refused at construction rather than at the first forward."""
    ModelConfig(VOCAB, MAX_LENGTH, dropout=0.0)
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ModelConfig(0, MAX_LENGTH)
    with pytest.raises(ValueError, match="max_length must be positive"):
        ModelConfig(VOCAB, 0)
    with pytest.raises(ValueError, match="d_model must be positive"):
        ModelConfig(VOCAB, MAX_LENGTH, d_model=0)
    with pytest.raises(ValueError, match="n_layers must be positive"):
        ModelConfig(VOCAB, MAX_LENGTH, n_layers=0)
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\)"):
        ModelConfig(VOCAB, MAX_LENGTH, dropout=1.0)


def test_device_of_refuses_a_model_with_no_parameters() -> None:
    """A model with nothing to train has no device, and cannot be trained."""
    config = ModelConfig(VOCAB, MAX_LENGTH, d_model=D_MODEL, n_layers=1)
    model = build_model(config, _linear_factory([]))
    assert device_of(model) == torch.device("cpu")
    with pytest.raises(ValueError, match="model has no parameters"):
        device_of(nn.Identity())


def test_mixer_sees_the_stream_width_and_the_widest_batch() -> None:
    """The factory is handed ``(d_model, max_length)``, in that order.

    A mixer that sizes a positional table gets the widest batch the arm can produce, not
    the width of the batch in front of it, so an evaluation past the trained length does
    not resize anything mid-run.
    """
    seen: list[tuple[int, int]] = []

    class Echo(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros_like(x)

    def factory(d_model: int, max_length: int) -> nn.Module:
        seen.append((d_model, max_length))
        return Echo()

    build_model(ModelConfig(VOCAB, 256, d_model=D_MODEL, n_layers=2), factory)
    assert seen == [(D_MODEL, 256), (D_MODEL, 256)]
