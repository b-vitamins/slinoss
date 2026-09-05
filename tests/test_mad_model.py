"""The scaffold: shapes, the divergences it pins, and the two properties it must have.

Causality, because a scaffold that leaks a later position solves every recall task for
the wrong reason, and initialization ownership, because a mixer's own initialization is
part of the mixer and the scaffold's pass must not reach it.

The mixer here is a one-position shift, which is causal by construction and carries no
parameters worth initializing. What is under test is the scaffold around it.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from scripts.mad.model import (
    ModelConfig,
    RMSNorm,
    build_model,
    parameter_count,
    protect,
    sincos_positions,
)


class Shift(nn.Module):
    """A causal mixer: every position reads the one before it, through a projection.

    The projection is deliberately not what the scaffold's initialization pass would
    produce, so a pass that reached it would be visible.

    Args:
        d_model: Stream width.
        max_length: Ignored.
    """

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.proj.weight.fill_(0.25)
            self.proj.bias.fill_(0.5)

    def forward(self, x: Tensor) -> Tensor:
        """Shift by one position, then project.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``, the first position reading zeros.
        """
        shifted = torch.zeros_like(x)
        shifted[:, 1:] = x[:, :-1]
        return self.proj(shifted)


def config(**kwargs: object) -> ModelConfig:
    """A small scaffold config with task and observed lengths 8."""
    settings: dict[str, object] = {
        "vocab_size": 11,
        "task_length": 8,
        "observed_width": 8,
        "d_model": 16,
    }
    settings.update(kwargs)
    return ModelConfig(**settings)  # pyright: ignore[reportArgumentType]


def test_causal_model_scores_every_position() -> None:
    """``(B,T)`` int64 in, ``(B,T,vocab_size)`` out."""
    model = build_model(config(), Shift)
    ids = torch.randint(0, 11, (3, 8))
    assert model(ids).shape == (3, 8, 11)


def test_bottleneck_model_reconstructs_from_one_state() -> None:
    """Every output position comes from the last encoder position and its own code.

    So a change at the last input position moves every output, where the causal model
    moves only that position. That contrast is the whole difference between the two
    backbones, and compression is scored on it.
    """
    model = build_model(config(bottleneck=True), Shift)
    model.eval()
    ids = torch.randint(0, 11, (2, 8))
    moved = ids.clone()
    moved[:, -1] = (moved[:, -1] + 1) % 11
    with torch.no_grad():
        before, after = model(ids), model(moved)
    assert before.shape == (2, 8, 11)
    assert not torch.allclose(before[:, 0], after[:, 0])


def test_bottleneck_refuses_an_input_past_its_position_code() -> None:
    """The code is sized once, at ``width``; a longer input would silently truncate."""
    model = build_model(config(bottleneck=True), Shift)
    with pytest.raises(ValueError, match="positions"):
        model(torch.randint(0, 11, (1, 9)))


def test_causal_model_cannot_see_a_later_position() -> None:
    """Perturbing the last token leaves every earlier position's logits untouched."""
    model = build_model(config(n_layers=2), Shift)
    model.eval()
    ids = torch.randint(0, 11, (2, 8))
    moved = ids.clone()
    moved[:, -1] = (moved[:, -1] + 1) % 11
    with torch.no_grad():
        before, after = model(ids), model(moved)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_initialization_stops_at_the_mixer() -> None:
    """A mixer keeps whatever its own constructor chose.

    The pass would overwrite any weight it can reach. :func:`protect` is what a builder
    calls to say the mixer owns its own initialization, and the encoder calls it for
    every mixer it builds.
    """
    model = build_model(config(n_layers=2), Shift)
    mixers = [module for module in model.modules() if isinstance(module, Shift)]
    assert len(mixers) == 2
    for mixer in mixers:
        weight, bias = mixer.proj.weight, mixer.proj.bias
        torch.testing.assert_close(weight, torch.full_like(weight, 0.25))
        torch.testing.assert_close(bias, torch.full_like(bias, 0.5))


def test_configured_task_length_not_observed_width_reaches_the_mixer() -> None:
    """The autoregressive shift cannot silently shorten constructor context."""
    model = build_model(config(task_length=128, observed_width=127), Shift)
    mixers = [module for module in model.modules() if isinstance(module, Shift)]
    assert [mixer.max_length for mixer in mixers] == [128]


def test_scaffold_weights_are_normal_and_biases_are_zero() -> None:
    """Linear and embedding weights at ``init_std``, biases at zero."""
    model = build_model(config(vocab_size=257, d_model=128, init_std=0.02), Shift)
    head = model.head
    assert isinstance(head, nn.Linear)
    assert head.bias is not None
    torch.testing.assert_close(head.bias, torch.zeros_like(head.bias))
    assert abs(float(head.weight.detach().std()) - 0.02) < 0.002
    embeds = model.token_embeds
    assert isinstance(embeds, nn.Embedding)
    assert abs(float(embeds.weight.detach().std()) - 0.02) < 0.002


def test_protect_marks_every_parameter() -> None:
    """The exemption is per tensor, so a nested module is covered too."""
    inner = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    assert protect(inner) is inner
    for param in inner.parameters():
        assert getattr(param, "_no_reinit", False)


def test_scaffold_initialization_is_an_explicit_opt_in() -> None:
    """A published model-wide initializer runs only when its entry names that owner."""
    inner = nn.Linear(4, 4)
    inner._mad_initialization_policy = "scaffold"  # type: ignore[attr-defined]
    assert protect(inner) is inner
    assert not getattr(inner.weight, "_no_reinit", False)
    inner._mad_initialization_policy = "silent"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="initialization policy"):
        protect(inner)


def test_ffn_width_follows_the_rounding_rule() -> None:
    """352 at `mad-lab`'s rounding, 341 at KLA's, from the same ``8/3`` rule."""
    assert config(d_model=128).d_ffn == 352
    assert config(d_model=128, ffn_multiple_of=1).d_ffn == 341


def test_position_code_is_halves_not_interleaved() -> None:
    """Sines first, cosines second. Interleaving them is a different code.

    `mad-lab` concatenates the halves and KLA interleaves; the decoder reads whichever it
    was built with, so the layout is part of the compression task's definition.
    """
    code = sincos_positions(6, 8)
    assert code.shape == (6, 8)
    torch.testing.assert_close(code[0, :4], torch.zeros(4))
    torch.testing.assert_close(code[0, 4:], torch.ones(4))
    assert float(code[1, 0]) == pytest.approx(float(torch.sin(torch.ones(1))), abs=1e-6)
    with pytest.raises(ValueError, match="at least 4"):
        sincos_positions(4, 2)


def test_norm_reduces_in_float32() -> None:
    """A bf16 input is normalized against a float32 mean square.

    The reduction is over ``d_model`` values whose squares span several orders; taking it
    narrow loses bits of the scale, which is why both upstream norms widen it.
    """
    norm = RMSNorm(64)
    wide = torch.randn(2, 3, 64) * 30.0
    narrow = norm(wide.to(torch.bfloat16))
    assert narrow.dtype == torch.bfloat16
    torch.testing.assert_close(narrow.float(), norm(wide), rtol=0.02, atol=0.02)
    unit = torch.ones(1, 1, 64)
    torch.testing.assert_close(norm(unit), unit, rtol=1e-4, atol=1e-4)


def test_parameter_count_counts_trainable_only() -> None:
    """A frozen parameter is not part of the count a record reports."""
    model = build_model(config(), Shift)
    total = parameter_count(model)
    assert total > 0
    head = model.head
    assert isinstance(head, nn.Linear)
    head.weight.requires_grad_(False)
    assert parameter_count(model) == total - head.weight.numel()
