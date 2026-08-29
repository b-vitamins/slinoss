"""The stopping rule, which decides which number this axis reports.

Every test here drives the loop with a model whose accuracy is scripted per evaluation, so the
rule is exercised on its own without a real model's noise. The rule is not obvious and it is not
a stopping rule in the usual sense: an evaluation that *ties* the running best both charges
patience and refreshes the reported test number, because the reference compares with ``<=`` and
then with ``>=``, in that order. The reported figure is therefore the test accuracy at the last
evaluation whose validation accuracy was at least the running best -- not the final one and not
the best one.

Three ways a plausible reimplementation gets it wrong, one test each: a tie treated as a plain
non-improvement never refreshes the test number, so the reported figure stays at the first
plateau; a halt that still refreshes reports a number measured after the run was over; and the
reference's ``step > 0`` guard, which makes the first evaluation observational, changes both
patience and the loss average at ``print_steps 1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scripts.tsc.batching import Batch, Loader
from scripts.tsc.split import Arrays
from scripts.tsc.train import (
    EPSILON,
    Splits,
    TrainConfig,
    accuracy,
    check_finite,
    loss_on,
    train,
)

SIZE = 4
"""Instances per split. The batch is the whole split, so one evaluation is one forward."""

TEST_PLAN = (0.25, 0.5, 0.75, 1.0)
"""Test accuracies, one per measurement, all distinct so the reported figure names its point."""


def loader_of(marker: int, size: int = SIZE) -> Loader:
    """A split whose every value is ``marker``, with every instance in class 0.

    Args:
        marker: 0 train, 1 validation, 2 test. :class:`Scripted` reads it to know which split it
            is being asked about.
        size: Instances.

    Returns:
        The loader.
    """
    inputs = np.full((size, 2, 1), float(marker), dtype=np.float32)
    targets = np.zeros((size, 2), dtype=np.float32)
    targets[:, 0] = 1.0
    return Loader(Arrays(inputs, targets))


def three_splits() -> Splits:
    """The three loaders, each distinguishable by its marker."""
    return Splits(loader_of(0), loader_of(1), loader_of(2))


class Scripted(nn.Module):
    """A model whose evaluation accuracy is read off a plan, one entry per measurement.

    In training mode it emits a real softmax of its one parameter, so the optimizer step and the
    loss are genuine and the loss decreases. In evaluation mode it predicts class 0 for a
    prescribed fraction of the rows, which makes the accuracy exactly that fraction.

    Args:
        val: Validation accuracy per measurement.
        test: Test accuracy per measurement.
    """

    def __init__(self, val: tuple[float, ...], test: tuple[float, ...]) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))
        self.plans = {0: (1.0,), 1: val, 2: test}
        self.calls = {0: 0, 1: 0, 2: 0}

    def forward(self, x: Tensor) -> Tensor:
        """Probabilities for a batch.

        Args:
            x: ``(B,L,1)``, every value the split's marker.

        Returns:
            ``(B,2)``.
        """
        size = int(x.shape[0])
        if self.training:
            return torch.softmax(self.bias.expand(size, 2), dim=-1)
        which = int(x[0, 0, 0].item())
        plan = self.plans[which]
        index = min(self.calls[which], len(plan) - 1)
        self.calls[which] += 1
        correct = round(plan[index] * size)
        out = torch.zeros(size, 2)
        out[:correct, 0] = 1.0
        out[correct:, 1] = 1.0
        return out


def test_a_tie_charges_patience_and_still_refreshes_the_reported_test_number() -> None:
    """A validation accuracy equal to the best fires both branches, in the reference's order.

    Four evaluations at one plateau: the first is observational, and the next three each refresh
    the test number while patience climbs to two. A reimplementation that treated a tie as a plain
    non-improvement would report the first plateau's test accuracy, 0.25, forever.
    """
    model = Scripted(val=(0.5,) * 4, test=TEST_PLAN)
    config = TrainConfig(
        lr=1e-2, batch_size=SIZE, num_steps=4, print_steps=1, patience=10
    )
    result = train(model, three_splits(), config)
    assert model.calls[2] == 3, "three of the four evaluations measured test"
    assert result.test_accuracy == TEST_PLAN[2]
    assert (result.best_step, result.steps, result.stopped_early) == (4, 4, False)
    assert [point.test_accuracy for point in result.evaluations] == [
        None,
        TEST_PLAN[0],
        TEST_PLAN[1],
        TEST_PLAN[2],
    ]


def test_a_patience_halt_suppresses_the_refresh_that_would_otherwise_fire() -> None:
    """The halting evaluation ties the best, so without the guard it would refresh test.

    That is the ``if not halt and ...`` in the loop and it is the difference between reporting a
    number measured while the run was live and one measured after patience ended it. Patience 1
    here, so the third tie halts.
    """
    model = Scripted(val=(0.5,) * 4, test=TEST_PLAN)
    config = TrainConfig(
        lr=1e-2, batch_size=SIZE, num_steps=6, print_steps=1, patience=1
    )
    result = train(model, three_splits(), config)
    assert model.calls[2] == 2
    assert (result.test_accuracy, result.best_step) == (TEST_PLAN[1], 3)
    assert (result.steps, result.stopped_early) == (4, True)
    last = result.evaluations[-1]
    assert last.val_accuracy == result.val_accuracy
    assert last.test_accuracy is None, "tied the best, and still no measurement"


def test_the_first_evaluation_is_observational_and_its_loss_carries_forward() -> None:
    """The reference's ``step > 0`` guard: no patience, no test, and no loss reset.

    It only bites at ``print_steps 1``, which is exactly where a smoke test runs. Patience 0 and a
    first validation accuracy of 0.0 -- equal to the initial best -- so an unguarded loop would
    charge patience and halt after one step. This one runs to the budget.
    """
    model = Scripted(val=(0.0, 0.5), test=TEST_PLAN)
    config = TrainConfig(
        lr=1e-2, batch_size=SIZE, num_steps=2, print_steps=1, patience=0
    )
    result = train(model, three_splits(), config)
    assert (result.steps, result.stopped_early) == (2, False)
    assert result.evaluations[0].test_accuracy is None
    assert model.calls[2] == 1
    # running_loss was not reset after the guarded point, so the second average holds both steps.
    assert result.evaluations[1].loss > result.evaluations[0].loss


def test_accuracy_counts_instances_rather_than_averaging_batches() -> None:
    """Three of five correct is 0.6, not the 0.5 a mean over batches of two would give.

    The trailing partial batch is where the two differ, and every reported accuracy on this axis
    comes through here. It also restores the training flag, so the loop can call it mid-run.
    """

    class Threshold(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            correct = x[:, 0, 0] > 0.0
            return torch.stack((correct.float(), 1.0 - correct.float()), dim=1)

    inputs = np.zeros((5, 2, 1), dtype=np.float32)
    inputs[:3] = 1.0
    targets = np.zeros((5, 2), dtype=np.float32)
    targets[:, 0] = 1.0
    loader = Loader(Arrays(inputs, targets))
    model = Threshold()
    model.train()
    assert accuracy(model, loader, 2) == pytest.approx(0.6)
    assert model.training, "the loop calls this mid-run and must come back training"


def test_the_loss_is_the_references_floored_log_and_not_a_fused_cross_entropy() -> None:
    """``-sum_c y_c log(p_c + 1e-8)``, so a probability of exactly zero is finite.

    The floor bounds the gradient on a confidently wrong example. ``log_softmax`` would give minus
    infinity here and a different trajectory on exactly the examples that dominate early training.
    """

    class Fixed(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            del x
            return torch.tensor([[0.5, 0.5], [0.0, 1.0]])

    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    found = loss_on(Fixed(), Batch(torch.zeros(2, 2, 1), targets))
    wanted = (-math.log(0.5 + EPSILON) - math.log(EPSILON)) / 2.0
    assert float(found) == pytest.approx(wanted, rel=1e-6)
    assert math.isfinite(float(found))


def test_a_corpus_the_unmasked_pool_cannot_survive_is_refused_up_front() -> None:
    """One missing value makes every prediction NaN, so it is caught before the model is built.

    The scaffold means over the whole sequence with no mask. The reference shares the hole and
    fills it with nothing; four of the archive's thirty arrive NaN-padded and would train to a
    NaN loss, so naming the count costs one pass over an array that is already resident.
    """
    inputs = np.zeros((2, 3, 1), dtype=np.float32)
    inputs[0, 1, 0] = np.nan
    targets = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1 of 6 inputs are not finite"):
        check_finite(Arrays(inputs, targets), dataset="Probe")
    targets[1, 1] = np.inf
    with pytest.raises(ValueError, match="targets are not finite"):
        check_finite(Arrays(np.zeros((2, 3, 1), np.float32), targets), dataset="Probe")


def test_a_batch_size_no_split_can_serve_stops_the_run_before_the_first_step() -> None:
    """All three splits are checked, so a validation split smaller than the batch is named early.

    That configuration exists in the published set -- a batch of 32 against a validation split of
    21 -- and the failure is otherwise an evaluation-time shape error minutes in.
    """
    splits = Splits(loader_of(0), loader_of(1, size=2), loader_of(2))
    config = TrainConfig(lr=1e-2, batch_size=SIZE, num_steps=2, print_steps=1)
    with pytest.raises(ValueError, match=r"batch_size 4 is over the split's 2"):
        train(Scripted(val=(0.5,), test=TEST_PLAN), splits, config)


def test_a_budget_shorter_than_one_interval_is_refused() -> None:
    """It would run to completion and report a test number that was never measured.

    The other refusals are ordinary bounds; this one is a configuration that succeeds and lies.
    """
    assert TrainConfig(lr=1e-3, batch_size=8).patience == 10
    with pytest.raises(ValueError, match="would never evaluate"):
        TrainConfig(lr=1e-3, batch_size=8, num_steps=500, print_steps=1000)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainConfig(lr=1e-3, batch_size=0)
    with pytest.raises(ValueError, match="patience must be non-negative"):
        TrainConfig(lr=1e-3, batch_size=8, patience=-1)
    with pytest.raises(ValueError, match="lr must be finite and positive"):
        TrainConfig(lr=0.0, batch_size=8)
    with pytest.raises(ValueError, match="lr must be finite and positive"):
        TrainConfig(lr=math.inf, batch_size=8)
