"""The loop, and the stopping rule that decides which number gets reported.

Adam at its defaults, a constant rate, no weight decay, no gradient clipping, no warmup,
float32 throughout. That is the reference's optimizer exactly, and every deviation from it
would move the bars in :data:`scripts.tsc.protocol.REFERENCE` by more than the differences this
axis is trying to measure.

The reported metric is the part that is easy to get wrong. It is not the final test accuracy and
it is not the best test accuracy. Every ``print_steps`` steps the loop evaluates train, then
validation, and then:

    validation <= best so far     charge one against patience; on the eleventh, stop, and do
                                  *not* touch the test number
    validation >= best so far     record the new best and re-measure test

Both branches fire when validation exactly equals the best, in that order, which is the
reference's own behaviour: an evaluation that ties the best both costs patience and refreshes
the reported test number. The reported figure is therefore the test accuracy at the last
evaluation whose validation accuracy was at least the running best.

The loss is the reference's, not a fused cross entropy::

    -sum_c y_c log(p_c + 1e-8)

with ``p`` the softmax the model already applied. The epsilon floors a confidently wrong
example's loss at ``-log(1e-8)`` and bounds its gradient; ``log_softmax`` does not, and swapping
it in changes the trajectory on exactly the examples that matter early.

What is not reproduced bit for bit: initialization, dropout masks and batch order are torch's.
The seed's job in this protocol is the partition, and that is
:func:`scripts.tsc.split.partition`'s.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from scripts.tsc.batching import Batch, Loader, generator_for
from scripts.tsc.protocol import NUM_STEPS, PATIENCE, PRINT_STEPS
from scripts.tsc.split import Arrays

__all__ = [
    "EPSILON",
    "Evaluation",
    "Splits",
    "TrainConfig",
    "TrainResult",
    "accuracy",
    "check_finite",
    "loss_on",
    "train",
]

EPSILON = 1e-8
"""The floor inside the log. The reference's, and load-bearing; see the module docstring."""


@dataclass(frozen=True)
class TrainConfig:
    """Everything a run needs that is not the model or the data.

    Attributes:
        lr: Constant Adam rate.
        batch_size: Instances per optimizer step, and per evaluation forward.
        num_steps: Cap on optimizer steps.
        print_steps: Steps between evaluations, and the unit patience counts in.
        patience: Non-improving evaluations tolerated. The run ends on the next one.
        seed: Fixes batch order through :func:`scripts.tsc.batching.generator_for`.
        betas: Adam betas.
        eps: Adam epsilon.

    Raises:
        ValueError: On a non-positive quantity, on a rate that is not finite and positive, or
            on a budget shorter than one evaluation interval. The last case would run to
            completion and report a test number that was never measured.
    """

    lr: float
    batch_size: int
    num_steps: int = NUM_STEPS
    print_steps: int = PRINT_STEPS
    patience: int = PATIENCE
    seed: int = 0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_steps", "print_steps"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.patience < 0:
            raise ValueError(f"patience must be non-negative, got {self.patience}")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and positive, got {self.lr}")
        if self.num_steps < self.print_steps:
            raise ValueError(
                f"num_steps {self.num_steps} is under print_steps {self.print_steps}, "
                f"so the run would never evaluate and would report nothing"
            )


class Splits(NamedTuple):
    """The three loaders.

    Attributes:
        train: Trains, and is also evaluated for the reported training accuracy.
        val: Selects. Both the stopping rule and the reported test point read it.
        test: Reported.
    """

    train: Loader
    val: Loader
    test: Loader


class Evaluation(NamedTuple):
    """One evaluation point.

    Attributes:
        step: Optimizer steps completed, so a multiple of ``print_steps``.
        loss: Mean training loss over the interval, the reference's
            ``running_loss / print_steps``.
        train_accuracy: Over the whole training split, in inference mode.
        val_accuracy: Over the whole validation split.
        test_accuracy: Over the whole test split when this evaluation was at least the best
            validation seen, else None. A None is not a missing measurement; it is the
            protocol declining to look.
        seconds: Wall time since the previous evaluation, the interval's training included.
    """

    step: int
    loss: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float | None
    seconds: float


class TrainResult(NamedTuple):
    """What a run reports.

    Attributes:
        test_accuracy: The reported figure: test accuracy at the last evaluation whose
            validation accuracy was at least the running best.
        val_accuracy: The validation accuracy at that same evaluation.
        best_step: Which step that was.
        steps: Optimizer steps actually taken.
        stopped_early: Whether patience ended the run rather than the budget.
        evaluations: Every point, in order.
    """

    test_accuracy: float
    val_accuracy: float
    best_step: int
    steps: int
    stopped_early: bool
    evaluations: list[Evaluation]


def check_finite(arrays: Arrays, *, dataset: str) -> None:
    """Refuse a dataset the scaffold cannot pool.

    The head means over the whole sequence with no mask, so one missing value anywhere in an
    instance makes that instance's prediction NaN, its loss NaN and every gradient NaN. The
    reference has the same hole and fills it with nothing. Four of the archive's thirty carry
    holes, all of them variable-length datasets padded with ``?`` in the ARFF:
    ``CharacterTrajectories``, ``InsectWingbeat``, ``JapaneseVowels`` and
    ``SpokenArabicDigits``. None of the six the protocol reports is among them, so refusing is
    free and catching it here costs one pass over an array that is already resident.

    Args:
        arrays: From :func:`scripts.tsc.split.prepare`.
        dataset: Name, for the message.

    Raises:
        ValueError: When any input or target is not finite, naming how many values.
    """
    for name, array in (("inputs", arrays.inputs), ("targets", arrays.targets)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        if bad:
            raise ValueError(
                f"{dataset}: {bad} of {array.size} {name} are not finite; the unmasked mean "
                f"pool would make every prediction NaN"
            )


def loss_on(model: nn.Module, batch: Batch) -> Tensor:
    """The reference's classification loss on one batch.

    Args:
        model: Emits probabilities, not logits.
        batch: The batch.

    Returns:
        A scalar: the mean over instances of ``-sum_c y_c log(p_c + 1e-8)``.
    """
    probabilities = model(batch.inputs)
    return -(batch.targets * torch.log(probabilities + EPSILON)).sum(dim=1).mean()


@torch.no_grad()
def accuracy(model: nn.Module, loader: Loader, batch_size: int) -> float:
    """Top-1 accuracy over a whole split, in inference mode.

    Inference mode means dropout off and batch norm on its running statistics, which is the
    reference's ``eqx.tree_inference``. The model's training flag is restored on the way out, so
    this is callable from inside the loop without leaving it evaluating.

    Args:
        model: The model.
        loader: The split.
        batch_size: Instances per forward.

    Returns:
        The fraction correct. Counted rather than averaged over batches, so a trailing partial
        batch is not over-weighted.

    Raises:
        ValueError: From :meth:`scripts.tsc.batching.Loader.epoch`.
    """
    was_training = model.training
    model.eval()
    try:
        correct = 0
        total = 0
        for batch in loader.epoch(batch_size):
            predicted = model(batch.inputs).argmax(dim=1)
            correct += int((predicted == batch.targets.argmax(dim=1)).sum().item())
            total += batch.size
    finally:
        model.train(was_training)
    return correct / total


def _steps(loader: Loader, config: TrainConfig) -> Iterator[Batch]:
    """The training batch stream.

    Args:
        loader: The training split.
        config: Holds the batch size and the seed.

    Returns:
        An endless iterator.
    """
    return loader.loop(config.batch_size, generator_for(config.seed))


def train(model: nn.Module, splits: Splits, config: TrainConfig) -> TrainResult:
    """Run the protocol on one model and one partition.

    Args:
        model: Built and already on the device the loaders are on.
        splits: The three loaders.
        config: The loop's settings.

    Returns:
        The result.

    Raises:
        ValueError: When a split cannot serve the configured batch size, which is checked on
            all three before the first step rather than at the first evaluation.
    """
    for loader in splits:
        loader.check_batch_size(config.batch_size)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps
    )
    batches = _steps(splits.train, config)
    evaluations: list[Evaluation] = []
    best_val = 0.0
    reported_test = 0.0
    reported_val = 0.0
    best_step = 0
    no_improvement = 0
    stopped_early = False
    running_loss = 0.0
    steps = 0
    start = time.monotonic()
    model.train()
    for step in range(config.num_steps):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_on(model, batch)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach())
        steps = step + 1
        if steps % config.print_steps != 0:
            continue

        train_accuracy = accuracy(model, splits.train, config.batch_size)
        val_accuracy = accuracy(model, splits.val, config.batch_size)
        elapsed = time.monotonic() - start
        start = time.monotonic()

        # The two comparisons are the reference's, in its order, and both fire on a tie: an
        # evaluation that equals the best costs patience and still refreshes the test number.
        # The `step > 0` guard is the reference's too. It only bites at print_steps 1, where it
        # makes the first evaluation observational: no patience charged, no test measured, and
        # the interval's loss carried into the next one.
        halt = False
        test_accuracy: float | None = None
        if step > 0:
            if val_accuracy <= best_val:
                no_improvement += 1
                halt = no_improvement > config.patience
            else:
                no_improvement = 0
            if not halt and val_accuracy >= best_val:
                best_val = val_accuracy
                test_accuracy = accuracy(model, splits.test, config.batch_size)
                reported_test = test_accuracy
                reported_val = val_accuracy
                best_step = steps
        evaluations.append(
            Evaluation(
                step=steps,
                loss=running_loss / config.print_steps,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                test_accuracy=test_accuracy,
                seconds=elapsed,
            )
        )
        if step > 0:
            running_loss = 0.0
        if halt:
            stopped_early = True
            break

    return TrainResult(
        test_accuracy=reported_test,
        val_accuracy=reported_val,
        best_step=best_step,
        steps=steps,
        stopped_early=stopped_early,
        evaluations=evaluations,
    )
