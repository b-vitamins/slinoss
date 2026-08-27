"""The protocol: the schedule, the groups, the batching rule, and the two accuracies.

Every number here is part of the comparison rather than an implementation choice, so each
is pinned against the Kalman Linear Attention driver's own setting. The two accuracies get
the most attention: micro and macro separate exactly when a task's supervised positions
are unbalanced across the vocabulary, which is most of MAD, and a run that reported one
under the other's name would beat or lose to a published bar for no reason at all.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scripts.mad.instances import IGNORE_INDEX
from scripts.mad.model import ModelConfig, build_model
from scripts.mad.tasks import Pool
from scripts.mad.train import (
    TrainConfig,
    _batches,
    evaluate,
    lr_at,
    parameter_groups,
    seed_all,
    train,
)

CPU: dict[str, Any] = {"device": "cpu"}
"""Every test here runs on the host: the protocol is the same program on either."""


class Oracle(nn.Module):
    """Predicts each position's own input token, with certainty.

    A model with a known argmax, so an accuracy is checkable by hand.

    Args:
        vocab_size: Classes.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, ids: Tensor) -> Tensor:
        """One-hot logits at the input id.

        Args:
            ids: ``(B,T)`` int64.

        Returns:
            ``(B,T,vocab_size)`` float32.
        """
        return nn.functional.one_hot(ids, self.vocab_size).float()


class Frozen(nn.Module):
    """A model whose logits do not depend on its parameter.

    Its loss is bit-identical every epoch, which is how the patience rule is tested
    without a tolerance: an implementation counting evaluations rather than epochs, or
    comparing against the wrong best, changes the stopping epoch.

    Args:
        vocab_size: Classes.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, ids: Tensor) -> Tensor:
        """Zero logits, carrying a gradient path that is identically zero.

        Args:
            ids: ``(B,T)`` int64.

        Returns:
            ``(B,T,vocab_size)`` float32.
        """
        flat = torch.zeros(*ids.shape, self.vocab_size)
        return flat + 0.0 * self.scale


class Grouped(nn.Module):
    """One parameter of each kind the decay split sorts on."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.transition = nn.Parameter(torch.zeros(4, 4))
        cast(Any, self.transition)._no_weight_decay = True
        self.frozen = nn.Parameter(torch.zeros(4, 4), requires_grad=False)


def copy_pool(num_train: int, num_test: int, width: int, vocab_size: int) -> Pool:
    """A pool whose target is its input, so a position-wise model can solve it.

    Args:
        num_train: Train examples.
        num_test: Test examples.
        width: Positions.
        vocab_size: Classes.

    Returns:
        The pool. Not a MAD task: what is under test is the loop, and a task a
        position-wise model cannot solve would confound a loop defect with a mixer's
        reach.
    """
    rng = np.random.default_rng(0)
    train = rng.integers(0, vocab_size, size=(num_train, width)).astype(np.int64)
    test = rng.integers(0, vocab_size, size=(num_test, width)).astype(np.int64)
    return Pool(train, train.copy(), test, test.copy(), 0.0)


def identity_model(vocab_size: int, width: int) -> nn.Module:
    """The scaffold with a mixer that carries nothing between positions."""
    config = ModelConfig(vocab_size=vocab_size, width=width, d_model=16, n_layers=1)
    return build_model(config, lambda d_model, max_length: nn.Identity())


@pytest.mark.parametrize(
    "field,value",
    [
        ("schedule", "linear"),
        ("precision", "fp16"),
        ("epochs", 0),
        ("batch_size", 0),
        ("log_every", 0),
        ("lr", 0.0),
        ("patience", -1),
        ("grad_clip", -1.0),
    ],
)
def test_a_setting_outside_the_protocol_is_refused(field: str, value: Any) -> None:
    """A protocol field is validated at construction, not at the step that uses it.

    An arm that fails 700 epochs in has already spent the time it was measuring.
    """
    with pytest.raises(ValueError, match=field):
        TrainConfig(**{field: value}, **CPU)


def test_the_default_protocol_is_the_published_one() -> None:
    """KLA's ``experiments/commands/mad.py``, spelled out.

    These are the numbers a MAD accuracy is comparable at. A default moving here moves
    every arm at once and is not visible in any single record.
    """
    config = TrainConfig()
    assert (config.epochs, config.patience, config.log_every) == (750, 70, 10)
    assert (config.batch_size, config.lr, config.weight_decay) == (128, 1e-3, 0.0)
    assert (config.schedule, config.precision, config.seed) == ("none", "fp32", 12345)
    assert config.grad_clip == 5.0


def test_the_flat_schedule_is_flat() -> None:
    """``none`` is the protocol's setting: no warmup, no decay, no step dependence."""
    config = TrainConfig(**CPU)
    assert [lr_at(config, step, 5) for step in (0, 1, 500, 5000)] == [config.lr] * 4


def test_the_cosine_schedule_warms_up_then_reaches_the_floor() -> None:
    """The ablation, when one is wanted: linear to ``lr``, half cosine to ``min_lr``."""
    config = TrainConfig(
        epochs=10, warmup_epochs=2, schedule="cosine", lr=1.0, min_lr=0.01, **CPU
    )
    assert lr_at(config, 0, 5) == pytest.approx(0.1)
    assert lr_at(config, 9, 5) == pytest.approx(1.0)
    assert lr_at(config, 10, 5) == pytest.approx(1.0)
    assert lr_at(config, 50, 5) == pytest.approx(config.min_lr)
    decay = [lr_at(config, step, 5) for step in range(10, 51)]
    assert decay == sorted(decay, reverse=True)


def test_decay_reaches_matrices_only() -> None:
    """A vector, and anything its own module exempted, sits at zero decay.

    A recurrence's transition parameters are shaped by their initialization; decaying
    them toward zero moves the operator rather than its scale.
    """
    groups = parameter_groups(Grouped(), 0.1)
    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    decayed, plain = (group["params"] for group in groups)
    assert len(decayed) == 1 and decayed[0].shape == (4, 4)
    assert {tuple(param.shape) for param in plain} == {(4,), (4, 4)}
    assert len(plain) == 2
    counted = sum(param.numel() for group in groups for param in group["params"])
    assert counted == 4 * 4 + 4 + 4 * 4


def test_training_drops_the_short_tail_and_evaluation_keeps_it() -> None:
    """``drop_last`` on the train split, whole coverage on the test split.

    The protocol's ``DataLoader`` drops the tail while training, and an evaluation that
    dropped it would score a split it does not report.
    """
    shuffled = list(_batches(10, 4, shuffle=True, generator=torch.Generator()))
    assert [int(batch.numel()) for batch in shuffled] == [4, 4]
    assert len(set(torch.cat(shuffled).tolist())) == 8
    whole = list(_batches(10, 4, shuffle=False, generator=None))
    assert [int(batch.numel()) for batch in whole] == [4, 4, 2]
    assert torch.cat(whole).tolist() == list(range(10))


def test_a_split_under_one_batch_is_refused() -> None:
    """Dropping the only batch would train on nothing and report the loss of nothing."""
    with pytest.raises(ValueError, match="under one batch"):
        list(_batches(3, 4, shuffle=True, generator=None))


def test_micro_and_macro_separate_on_an_unbalanced_split() -> None:
    """Micro counts positions, macro counts classes, and MAD's splits are unbalanced.

    Two supervised positions of class 0, one of class 9, and the oracle gets one of each
    wrong: micro is 1 of 3, macro is the mean of 1/2 and 0. Quoting either under the
    other's name is what this pins.
    """
    config = TrainConfig(batch_size=8, **CPU)
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[0, 0, IGNORE_INDEX, 9]])
    metrics = evaluate(Oracle(10), inputs, targets, vocab_size=10, config=config)
    assert metrics.micro == pytest.approx(1 / 3)
    assert metrics.macro == pytest.approx(0.25)


def test_evaluation_covers_the_whole_split() -> None:
    """The last, short batch counts.

    Four correct rows and one wrong one at batch size 2 is 0.8; a dropped tail reads 1.0
    and a solved task and an unsolved one look the same.
    """
    inputs = torch.arange(4).repeat(5, 1)
    targets = inputs.clone()
    targets[4] = (targets[4] + 1) % 4
    metrics = evaluate(
        Oracle(4),
        inputs,
        targets,
        vocab_size=4,
        config=TrainConfig(batch_size=2, **CPU),
    )
    assert metrics.micro == pytest.approx(0.8)


def test_a_fully_masked_split_scores_zero_rather_than_dividing_by_it() -> None:
    """No supervised position is a settings error, and it must not be a crash."""
    inputs = torch.arange(4).repeat(2, 1)
    metrics = evaluate(
        Oracle(4),
        inputs,
        torch.full_like(inputs, IGNORE_INDEX),
        vocab_size=4,
        config=TrainConfig(batch_size=2, **CPU),
    )
    assert metrics == (0.0, 0.0, 0.0)


def test_evaluation_leaves_the_model_as_it_found_it() -> None:
    """A mixer with a training-only path would otherwise stay switched off."""
    model = Oracle(4)
    model.train()
    evaluate(
        model,
        torch.arange(4).repeat(2, 1),
        torch.arange(4).repeat(2, 1),
        vocab_size=4,
        config=TrainConfig(batch_size=2, **CPU),
    )
    assert model.training


def test_the_loop_learns_a_task_a_position_wise_model_can_learn() -> None:
    """End to end: the loss falls, the accuracy rises, and every epoch runs.

    A loop that stepped the wrong parameters, evaluated the train split, or clipped the
    gradient to nothing would still produce a report; this is what distinguishes it from
    one that trained.
    """
    config = TrainConfig(
        epochs=40, batch_size=8, lr=0.05, log_every=10, patience=0, **CPU
    )
    seed_all(config.seed)
    model = identity_model(8, 4)
    report = train(model, copy_pool(32, 16, 4, 8), config, vocab_size=8)
    assert len(report.points) == 4
    assert report.epochs_run == 40
    assert not report.stopped_early
    assert report.points[-1].test.loss < report.points[0].test.loss
    assert report.best.micro > 0.9
    assert report.best.loss <= report.points[0].test.loss


def test_patience_is_counted_in_epochs() -> None:
    """A run whose loss never improves stops ``patience`` epochs after its best.

    Counted in epochs, not in evaluations: at ``log_every`` 10 the two differ by a factor
    of ten and the published patience is 70 epochs.
    """
    config = TrainConfig(
        epochs=20, batch_size=4, lr=0.1, log_every=1, patience=3, **CPU
    )
    report = train(Frozen(4), copy_pool(8, 4, 4, 4), config, vocab_size=4)
    assert report.best_epoch == 0
    assert report.epochs_run == 4
    assert report.stopped_early
    assert report.best == report.final


def test_bf16_leaves_the_master_weights_at_float32() -> None:
    """Autocast wraps the forward pass and nothing else.

    A mixer whose kernel wants a narrow dtype gets it that way; a run that cast the
    parameters would change the optimizer's arithmetic as well as the forward pass, and
    the loss is taken at float32 either way.
    """
    config = TrainConfig(epochs=1, batch_size=8, log_every=1, precision="bf16", **CPU)
    seed_all(config.seed)
    model = identity_model(8, 4)
    report = train(model, copy_pool(16, 8, 4, 8), config, vocab_size=8)
    assert all(param.dtype == torch.float32 for param in model.parameters())
    assert len(report.points) == 1
