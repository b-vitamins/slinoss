"""The MAD training loop.

The default protocol is the published Kalman Linear Attention MAD protocol: 750 epochs of
AdamW at a flat 1e-3, batch 172, no weight decay, gradient norm clipped at 5, evaluation
every tenth epoch, and a stop when 70 epochs pass without the best test accuracy improving.
The named ``legacy-hybrid`` driver profile explicitly restores batch 128 for replaying the
repository implementation's deliberate departure. A task is solved or it is not, so
nothing here tunes: the same numbers run for every architecture.

Weight decay applies to matrices only. A parameter of dimension below two, or one its
own module marked ``_no_weight_decay``, sits in the second group at zero decay -- a
recurrence's transition parameters are shaped by their initialization and decaying them
toward zero moves the operator, not just its scale.

Accuracy is reported twice. Micro is the fraction of supervised positions predicted
correctly and is what KLA reports and what decides an arm; macro is the mean of the
per-class fractions over the classes that appear, which is what `mad-lab`'s torchmetrics
default reports. They separate when a task's supervised positions are unbalanced across
the vocabulary, and quoting one against the other's bar is the mistake this module exists
to make impossible.

Precision defaults to float32, the protocol's setting. ``bf16`` wraps the forward pass in
autocast and nothing else: master weights stay float32, the loss is taken at float32, and
no tensor is ever cast in place. A mixer whose kernel needs a narrow dtype gets it that
way, and the run record carries the setting because it is part of the arm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from scripts.mad.instances import IGNORE_INDEX
from scripts.mad.tasks import Pool

SCHEDULES = ("none", "cosine")
"""Learning-rate schedules. KLA's MAD config uses ``none``."""

PRECISIONS = ("fp32", "bf16")
"""Forward-pass precisions. The protocol uses ``fp32``."""


@dataclass(frozen=True)
class TrainConfig:
    """The protocol. Defaults are KLA's MAD settings.

    Attributes:
        epochs: Passes over the train split.
        batch_size: Examples per step, train and eval.
        lr: Peak learning rate, and the constant one under ``none``.
        min_lr: Floor the cosine decays to. Unused under ``none``.
        weight_decay: Decay on matrices. The protocol's is zero.
        schedule: A member of :data:`SCHEDULES`.
        warmup_epochs: Linear warmup from zero, cosine only.
        grad_clip: Gradient norm ceiling. Zero disables clipping.
        patience: Epochs without a strict test-accuracy improvement before stopping.
            Counted in epochs, not evaluations. Zero disables the stop.
        eval_every: Evaluate every this many epochs. The final epoch always evaluates.
        drop_last: Whether training omits a short final batch. Explicit because
            MAD-Lab keeps it while the KLA driver drops it. The paper's batch 172 does
            not divide the standard pools.
        float32_matmul_precision: PyTorch float32 matmul policy. MAD-Lab sets ``high``.
        precision: A member of :data:`PRECISIONS`.
        seed: Seeds the shuffle, and :func:`seed_all` for everything else.
        device: Where the model and the pool live.
    """

    epochs: int = 750
    batch_size: int = 172
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    schedule: str = "none"
    warmup_epochs: int = 5
    grad_clip: float = 5.0
    patience: int = 70
    eval_every: int = 10
    drop_last: bool = True
    float32_matmul_precision: str = "high"
    precision: str = "fp32"
    seed: int = 12345
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.schedule not in SCHEDULES:
            raise ValueError(
                f"schedule must be one of {SCHEDULES}, got {self.schedule}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision}"
            )
        for name in ("epochs", "batch_size", "eval_every"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("lr", "min_lr"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("weight_decay", "grad_clip", "patience", "warmup_epochs"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
        if self.float32_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "float32_matmul_precision must be highest, high, or medium, got "
                f"{self.float32_matmul_precision!r}"
            )


class Metrics(NamedTuple):
    """One evaluation of one split.

    Attributes:
        loss: Mean cross entropy over supervised positions.
        micro: Supervised positions predicted correctly, over their count.
        macro: Mean over the classes present of that class's own fraction.
    """

    loss: float
    micro: float
    macro: float


class Point(NamedTuple):
    """One evaluation, at the epoch it was taken.

    Attributes:
        epoch: Zero-based epoch just finished.
        step: Optimizer steps taken.
        train_loss: Mean train loss over that epoch's steps.
        test: The test split's metrics.
    """

    epoch: int
    step: int
    train_loss: float
    test: Metrics


class Report(NamedTuple):
    """What one arm produced.

    Attributes:
        best: Metrics at the evaluation with the highest test micro accuracy, ties going
            to the earliest. Selection is by accuracy because that is the reported
            quantity and the one the published bars are: past a task's loss plateau the
            two disagree, and a loss-selected arm reports the accuracy of a point every
            later evaluation beats.
        best_epoch: Its epoch.
        final: The last evaluation's metrics.
        points: Every evaluation, in order.
        epochs_run: Epochs actually run.
        stopped_early: Whether patience ended the run.
    """

    best: Metrics
    best_epoch: int
    final: Metrics
    points: tuple[Point, ...]
    epochs_run: int
    stopped_early: bool


def seed_all(seed: int) -> None:
    """Seed torch and numpy.

    Call before building the model: initialization draws from the torch generator, so
    the seed has to be set first for an arm to reproduce.

    Args:
        seed: The seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))


def lr_at(config: TrainConfig, step: int, steps_per_epoch: int) -> float:
    """Learning rate for one step.

    Args:
        config: The protocol.
        step: Zero-based optimizer step.
        steps_per_epoch: Steps in one epoch, for the warmup and decay horizons.

    Returns:
        Under ``none``, ``config.lr``. Under ``cosine``, a linear warmup over
        ``warmup_epochs`` then a half cosine from ``lr`` to ``min_lr`` over what remains.
    """
    if config.schedule == "none":
        return config.lr
    warmup = config.warmup_epochs * steps_per_epoch
    if step < warmup:
        return config.lr * (step + 1) / warmup
    total = config.epochs * steps_per_epoch
    span = max(total - warmup, 1)
    phase = min((step - warmup) / span, 1.0)
    scale = 0.5 * (1.0 + math.cos(math.pi * phase))
    return config.min_lr + (config.lr - config.min_lr) * scale


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into decayed and undecayed groups.

    Args:
        model: The model.
        weight_decay: Decay for the first group.

    Returns:
        Two AdamW groups. A parameter below dimension two, or one marked
        ``_no_weight_decay`` by its own module, goes in the undecayed group.
    """
    decay: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        exempt = getattr(param, "_no_weight_decay", False) or param.dim() < 2
        (plain if exempt else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def _to_device(array: NDArray[np.int64], device: str) -> Tensor:
    """One split's array as an int64 tensor on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _batches(
    count: int,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    generator: torch.Generator | None,
) -> Iterator[Tensor]:
    """Index batches over ``count`` examples.

    Args:
        count: Examples.
        batch_size: Examples per batch.
        shuffle: Whether to permute.
        drop_last: Whether to omit a short final batch.
        generator: Draw source for the permutation.

    Yields:
        Index tensors.

    Raises:
        ValueError: When ``drop_last`` would leave no batch.
    """
    if drop_last and count < batch_size:
        raise ValueError(f"{count} examples is under one batch of {batch_size}")
    order = (
        torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    )
    stop = count - count % batch_size if drop_last else count
    for start in range(0, stop, batch_size):
        yield order[start : min(start + batch_size, count)]


def _autocast(config: TrainConfig) -> torch.autocast:
    """Autocast context for the configured precision.

    Args:
        config: The protocol.

    Returns:
        A context that is inert under ``fp32``.
    """
    return torch.autocast(
        device_type=torch.device(config.device).type,
        dtype=torch.bfloat16,
        enabled=config.precision == "bf16",
    )


def _loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy over supervised positions, at float32.

    Args:
        logits: ``(B,T,V)``, any dtype.
        targets: ``(B,T)`` int64, ``IGNORE_INDEX`` where unsupervised.

    Returns:
        Scalar. Float32 whatever the forward pass ran at: a narrow softmax over a large
        vocabulary loses the tail the loss is measuring.
    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


class _Tally:
    """Accumulator over evaluation batches.

    Holds the supervised-position counts a split's metrics need: total loss weighted by
    position count, total hits, and per-class hits and support for the macro average.

    Args:
        vocab_size: Classes.
        device: Where the counters live.
    """

    def __init__(self, vocab_size: int, device: str) -> None:
        self.loss = torch.zeros((), dtype=torch.float64, device=device)
        self.positions = torch.zeros((), dtype=torch.float64, device=device)
        self.hits = torch.zeros(vocab_size, dtype=torch.float64, device=device)
        self.support = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    def add(self, logits: Tensor, targets: Tensor) -> None:
        """Fold one batch in.

        Args:
            logits: ``(B,T,V)``.
            targets: ``(B,T)`` int64.
        """
        flat = targets.reshape(-1)
        keep = flat != IGNORE_INDEX
        count = int(keep.sum())
        if count == 0:
            return
        gold = flat[keep]
        guess = logits.reshape(-1, logits.shape[-1])[keep].argmax(-1)
        self.loss += float(_loss(logits, targets)) * count
        self.positions += count
        self.support.index_add_(0, gold, torch.ones_like(gold, dtype=torch.float64))
        self.hits.index_add_(0, gold, (guess == gold).to(torch.float64))

    def metrics(self) -> Metrics:
        """The split's metrics.

        Returns:
            Zeros when no position was supervised, which a task's settings can produce
            only if its targets are entirely masked.
        """
        positions = float(self.positions)
        if positions == 0.0:
            return Metrics(0.0, 0.0, 0.0)
        present = self.support > 0
        per_class = self.hits[present] / self.support[present]
        return Metrics(
            float(self.loss) / positions,
            float(self.hits.sum()) / positions,
            float(per_class.mean()),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    vocab_size: int,
    config: TrainConfig,
) -> Metrics:
    """Score a split.

    Args:
        model: The model. Restored to its incoming train or eval mode.
        inputs: ``(N,T)`` int64 on ``config.device``.
        targets: ``(N,T)`` int64 on ``config.device``.
        vocab_size: Classes, for the macro average.
        config: The protocol. Supplies batch size, device and precision.

    Returns:
        The split's metrics, over every example including a short final batch.
    """
    was_training = model.training
    model.eval()
    tally = _Tally(vocab_size, config.device)
    for index in _batches(
        int(inputs.shape[0]),
        config.batch_size,
        shuffle=False,
        drop_last=False,
        generator=None,
    ):
        batch = index.to(config.device)
        with _autocast(config):
            logits = model(inputs[batch])
        tally.add(logits, targets[batch])
    model.train(was_training)
    return tally.metrics()


def train(
    model: nn.Module,
    pool: Pool,
    config: TrainConfig,
    *,
    vocab_size: int,
    on_point: Callable[[Point], None] | None = None,
) -> Report:
    """Run the protocol.

    Args:
        model: Built and on ``config.device``.
        pool: The task's fixed splits. Moved to the device whole: the largest MAD pool is
            a few tens of megabytes, and re-staging a batch per step would put a host
            copy on the critical path of a loop whose step is milliseconds.
        config: The protocol.
        vocab_size: Classes, for the macro average.
        on_point: Called with each evaluation as it happens, for a caller that streams
            progress. The report carries the same points.

    Returns:
        The report.

    Raises:
        ValueError: From :func:`_batches`, when the train split is under one batch.
    """
    device = config.device
    train_inputs = _to_device(pool.train_inputs, device)
    train_targets = _to_device(pool.train_targets, device)
    test_inputs = _to_device(pool.test_inputs, device)
    test_targets = _to_device(pool.test_targets, device)

    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.weight_decay), lr=config.lr
    )
    shuffle = torch.Generator()
    shuffle.manual_seed(config.seed)

    count = int(train_inputs.shape[0])
    steps_per_epoch = (
        count // config.batch_size
        if config.drop_last
        else math.ceil(count / config.batch_size)
    )
    points: list[Point] = []
    best = Metrics(math.inf, 0.0, 0.0)
    best_epoch = -1
    step = 0
    stopped_early = False
    epoch = 0

    model.train()
    for epoch in range(config.epochs):
        running = 0.0
        for index in _batches(
            count,
            config.batch_size,
            shuffle=True,
            drop_last=config.drop_last,
            generator=shuffle,
        ):
            batch = index.to(device)
            rate = lr_at(config, step, steps_per_epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            with _autocast(config):
                logits = model(train_inputs[batch])
            loss = _loss(logits, train_targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running += float(loss.detach())
            step += 1

        due = (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1
        if not due:
            continue
        test = evaluate(
            model,
            test_inputs,
            test_targets,
            vocab_size=vocab_size,
            config=config,
        )
        point = Point(epoch, step, running / max(steps_per_epoch, 1), test)
        points.append(point)
        if on_point is not None:
            on_point(point)
        if best_epoch < 0 or test.micro > best.micro:
            best, best_epoch = test, epoch
        if config.patience and epoch - best_epoch >= config.patience:
            stopped_early = True
            break

    # The final epoch always evaluates and the epoch count is validated positive, so
    # points is never empty and the patience break happens only after an append.
    return Report(
        best, best_epoch, points[-1].test, tuple(points), epoch + 1, stopped_early
    )
