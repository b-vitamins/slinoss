"""The scaffold, held to the reference's composition rather than to a plausible one.

Every arm on this axis is scored inside this stack, so a difference here is a difference in every
number the axis produces, and none of the differences that matter are visible in a training curve.
The three pinned below are the ones a reimplementation gets wrong by default: torch's ``gelu`` is
the exact erf form while JAX's default is the tanh approximation, the residual skip is the block
input and not the normalized stream, and the norm is an unaffine batch norm over batch and time
rather than a layer norm over channels.

The construction-time refusals get a test of their own because they are what makes swapping a
mixer safe. A factory wired at the wrong width otherwise surfaces as a matmul error deep inside a
lane, after the corpus is loaded and the optimizer is built.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from scripts.tsc.linoss import LinOSSRecurrence
from scripts.tsc.model import (
    GLU,
    Block,
    ModelConfig,
    build_model,
    mixer_parameters,
    parameter_count,
)

CONFIG = ModelConfig(input_dim=3, hidden_dim=6, classes=4, blocks=2)


def identity(_d_model: int, _max_length: int) -> nn.Module:
    """A mixer that does nothing, so a test reads the scaffold alone."""
    return nn.Identity()


def test_the_scaffold_emits_probabilities_and_refuses_a_bad_stream() -> None:
    """``(B,L,input_dim)`` in, ``(B,classes)`` probabilities out, and nothing else accepted.

    The head emits probabilities and not logits because the reference's loss consumes them; a
    caller that fed logits to that loss would get a number that trains and means nothing. The rank
    refusal is not pedantic either: a ``(B,L)`` batch broadcasts through the encoder and trains.
    """
    torch.manual_seed(1)
    model = build_model(CONFIG, [identity] * CONFIG.blocks, max_length=5)
    model.eval()
    with torch.no_grad():
        probabilities = model(torch.randn(7, 5, 3))
    assert probabilities.shape == (7, 4)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(7), atol=1e-6)
    assert bool((probabilities >= 0).all())
    with pytest.raises(ValueError, match=r"expected \(B,L,3\)"):
        model(torch.randn(7, 5))
    with pytest.raises(ValueError, match=r"expected \(B,L,3\)"):
        model(torch.randn(7, 5, 4))


def test_a_block_is_the_reference_composition_with_the_tanh_gelu() -> None:
    """norm, mix, tanh-gelu, gate, add -- and the exact erf gelu is a different scaffold.

    Both halves are asserted: the block matches the tanh form and does *not* match the erf form.
    Without the second half the test passes under either activation, which is the mistake it is
    here to catch.
    """
    torch.manual_seed(2)
    block = Block(CONFIG).eval()
    x = torch.randn(4, 5, CONFIG.hidden_dim) * 3.0
    with torch.no_grad():
        found = block(x)
        normed = block.norm(x.transpose(1, 2)).transpose(1, 2)
        tanh = x + block.glu(nn.functional.gelu(normed, approximate="tanh"))
        erf = x + block.glu(nn.functional.gelu(normed, approximate="none"))
    assert torch.allclose(found, tanh, atol=1e-6)
    assert not torch.allclose(found, erf, atol=1e-6)


def test_the_skip_carries_the_block_input_and_not_the_normalized_stream() -> None:
    """With a mixer that emits zeros, the residual is the input plus a constant.

    A pre-norm residual -- the common arrangement, and what a reader would assume -- would leave
    the normalized stream in the trunk instead. That changes the scale every downstream block
    sees, so it changes results without changing shapes.
    """

    class Zero(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros_like(x)

    torch.manual_seed(3)
    block = Block(CONFIG).eval()
    block.add_module("mixer", Zero())
    x = torch.randn(4, 5, CONFIG.hidden_dim) * 3.0
    with torch.no_grad():
        found = block(x)
        normed = block.norm(x.transpose(1, 2)).transpose(1, 2)
    added = found - x
    assert torch.allclose(added, added[0, 0].expand_as(added), atol=1e-6)
    assert not torch.allclose(
        found - normed, (found - normed)[0, 0].expand_as(added), atol=1e-6
    )


def test_the_norm_is_unaffine_and_pools_over_batch_and_time() -> None:
    """No scale and no shift, and the running mean is the mean over batch and time per channel.

    A layer norm over channels would be the same shape and the same parameter count at
    ``affine=False``, and would normalize a different axis. The momentum is torch's weight on the
    *batch* statistic, which is one minus the reference's equinox default.
    """
    block = Block(CONFIG)
    assert block.norm.weight is None
    assert block.norm.bias is None
    assert block.norm.momentum == CONFIG.norm_momentum == 0.01
    torch.manual_seed(4)
    x = torch.randn(4, 5, CONFIG.hidden_dim) + 2.0
    block.train()
    block(x)
    running = block.norm.running_mean
    assert running is not None
    assert torch.allclose(running, CONFIG.norm_momentum * x.mean(dim=(0, 1)), atol=1e-6)


def test_a_mixer_that_does_not_preserve_the_stream_is_refused_at_construction() -> None:
    """The width probe and the factory count, both before any data is touched.

    The probe is the guard that makes the registry safe to point at an arbitrary module: a mixer
    built at the wrong width otherwise runs the encoder, the norm and the corpus load first and
    fails on a matmul that names neither the block nor the factory.
    """
    with pytest.raises(ValueError, match=r"mixer maps \(1, 2, 6\) to \(1, 2, 7\)"):
        build_model(CONFIG, [lambda d, _l: nn.Linear(d, d + 1)] * 2, max_length=5)
    with pytest.raises(ValueError, match="1 factories for 2 blocks"):
        build_model(CONFIG, [identity], max_length=5)


def test_mixer_parameters_separates_the_recurrence_from_the_scaffold() -> None:
    """The mixers' parameters only, which is the split every ablation on this axis reports.

    Counting the scaffold into a mixer's total makes a width change look like a capacity change:
    the encoder, the norms, the GLUs and the head are shared across arms at one width.
    """
    torch.manual_seed(5)
    scaffolded = build_model(CONFIG, [identity] * CONFIG.blocks, max_length=5)
    assert mixer_parameters(scaffolded) == 0

    def factory(d_model: int, _max_length: int) -> nn.Module:
        return LinOSSRecurrence(d_model, ssm_dim=4, discretization="IM")

    model = build_model(CONFIG, [factory] * CONFIG.blocks, max_length=5)
    per_block = sum(
        p.numel() for p in LinOSSRecurrence(CONFIG.hidden_dim, ssm_dim=4).parameters()
    )
    assert mixer_parameters(model) == CONFIG.blocks * per_block
    assert parameter_count(model) == parameter_count(scaffolded) + mixer_parameters(
        model
    )


def test_the_glu_is_the_references_two_full_width_projections() -> None:
    """``w1(x) * sigmoid(w2(x))``, both biased and both full width.

    The common half-width gate has the same signature and two thirds of the parameters, so
    substituting it would move every parameter count this axis reports.
    """
    torch.manual_seed(6)
    gate = GLU(5)
    x = torch.randn(3, 5)
    with torch.no_grad():
        assert torch.allclose(
            gate(x), gate.value(x) * torch.sigmoid(gate.gate(x)), atol=1e-6
        )
    assert sum(p.numel() for p in gate.parameters()) == 2 * (5 * 5 + 5)
    with pytest.raises(ValueError, match="width must be positive"):
        GLU(0)


def test_the_config_refuses_a_shape_that_cannot_be_built() -> None:
    """A non-positive size or a rate outside ``[0, 1)``, at the config and not at the first layer.

    This is where a swept value arrives, so refusing here names the setting instead of failing
    inside ``nn.Linear`` with no reference to the sweep point that produced it.
    """
    assert ModelConfig(input_dim=1, hidden_dim=1, classes=1, blocks=1).drop_rate == 0.05
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        ModelConfig(input_dim=1, hidden_dim=0, classes=2, blocks=1)
    with pytest.raises(ValueError, match=r"drop_rate must be in \[0, 1\)"):
        ModelConfig(input_dim=1, hidden_dim=2, classes=2, blocks=1, drop_rate=1.0)


def test_two_builds_under_one_seed_are_identical() -> None:
    """Every parameter is a default draw or a mixer's own initialization, so the seed fixes both.

    An arm that reseeded between the scaffold and the mixers, or built the mixers first, would not
    reproduce from its record even with the same seed written in it.
    """

    def build() -> list[Tensor]:
        torch.manual_seed(7)

        def factory(d_model: int, _max_length: int) -> nn.Module:
            return LinOSSRecurrence(d_model, ssm_dim=4)

        model = build_model(CONFIG, [factory] * CONFIG.blocks, max_length=5)
        return [p.detach().clone() for p in model.parameters()]

    for first, second in zip(build(), build(), strict=True):
        assert torch.equal(first, second)
