"""The optimization protocol and the metrics.

Upstream's protocol, reproduced without the framework around it. Stated in full because
every one of these is a measurement decision and none of them is a default worth
rediscovering:

- ``AdamW`` over *every* parameter at weight decay 0.1, betas and epsilon at torch's
  defaults. No matrices-only split: norms, biases and embeddings all decay.
- Cosine schedule from the initial rate to exactly zero over ``max_epochs``, advanced once
  per epoch, and advanced only when the epoch did not trigger early stopping.
- No warmup. No gradient clipping.
- Cross entropy at ``ignore_index`` -100, so a step's loss is the mean over that batch's
  supervised positions and nothing else contributes.
- Early stopping the moment test example accuracy strictly exceeds 0.99. The check runs
  after each epoch's evaluation and before the schedule advances, so a run that stops at
  epoch 0 never decays its rate.
- Batches in a fixed order, never shuffled, tail batches kept. See
  :mod:`scripts.mqar.tasks`.

Accuracy is reported two ways because they differ and upstream reports only the first.
``example`` is the mean over examples of the per-example fraction of supervised positions
recovered -- upstream's number, and the one every published MQAR figure plots. ``position``
is the fraction of all supervised positions recovered. They coincide within a segment and
diverge across a pool whose segments carry different key-value counts, which is exactly
the multi-segment protocol. Loss is reported position-weighted; upstream averages
per-batch means unweighted, which differs on the same pools, and no published MQAR number
is a loss.

Upstream runs float32 throughout. ``precision="bf16"`` adds an autocast region around the
forward, and nothing else: parameters stay float32, and the loss is taken in float32. It
exists because a mixer's CUDA path may only be reachable at bf16, and a mixer measured on
a reference fallback is not the mixer.
"""

from __future__ import annotations

import contextlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from scripts.mqar.instances import IGNORE_INDEX
from scripts.mqar.tasks import Batch, Segment, batches

PRECISIONS = ("fp32", "bf16")
"""Admissible activation precisions. Parameters are float32 either way."""


@dataclass(frozen=True)
class TrainConfig:
    """The protocol.

    Attributes:
        max_epochs: Passes over the train pool, and the cosine schedule's period.
        batch_size: Train rows per batch.
        test_batch_size: Test rows per batch. 0 means the train value, which is what the
            figure-2 sweep uses; the modern repro uses an eighth of it.
        lr: Initial learning rate.
        weight_decay: AdamW decay, applied to every parameter.
        early_stopping_threshold: Stop once test example accuracy strictly exceeds this.
            A value of 1.0 or more never stops early.
        precision: One of :data:`PRECISIONS`.
        seed: Seeds python, numpy and torch before the model is built.
        device: Torch device string.
    """

    max_epochs: int = 64
    batch_size: int = 512
    test_batch_size: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.1
    early_stopping_threshold: float = 0.99
    precision: str = "fp32"
    seed: int = 123
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.test_batch_size < 0:
            raise ValueError(
                f"test_batch_size must be non-negative, got {self.test_batch_size}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision!r}"
            )

    @property
    def eval_batch_size(self) -> int:
        """Test rows per batch, resolving 0 to the train value."""
        return self.test_batch_size or self.batch_size


class Metrics(NamedTuple):
    """One evaluation of the test pool.

    Attributes:
        loss: Cross entropy per supervised position over the whole pool.
        example: Mean over examples of the per-example recovered fraction.
        position: Recovered fraction over all supervised positions.
        by_slice: ``{slice_key: {slice_value: example accuracy}}``. Slice values are
            strings so the record is JSON-clean.
    """

    loss: float
    example: float
    position: float
    by_slice: dict[str, dict[str, float]]


class Point(NamedTuple):
    """One epoch.

    Attributes:
        epoch: Zero-based index.
        lr: The rate that epoch trained at.
        train_loss: Mean of the epoch's per-batch losses.
        test: The evaluation that followed it.
    """

    epoch: int
    lr: float
    train_loss: float
    test: Metrics


class Report(NamedTuple):
    """A finished run.

    Attributes:
        best: The evaluation with the highest example accuracy.
        best_epoch: Its epoch.
        final: The last evaluation.
        points: Every epoch, in order.
        epochs_run: Epochs actually run, which is fewer than ``max_epochs`` when early
            stopping fired.
        stopped_early: Whether it fired.
    """

    best: Metrics
    best_epoch: int
    final: Metrics
    points: tuple[Point, ...]
    epochs_run: int
    stopped_early: bool


def seed_all(seed: int) -> None:
    """Seed python, numpy and torch, matching upstream's ``set_determinism``.

    Upstream does not set cudnn determinism and neither does this, so a run is
    reproducible up to the nondeterminism of the kernels it dispatches to.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(config: TrainConfig, epoch: int) -> float:
    """The rate epoch ``epoch`` trains at.

    The closed form of ``CosineAnnealingLR(T_max=max_epochs, eta_min=0.0)`` stepped once
    per epoch: ``lr * (1 + cos(pi * epoch / max_epochs)) / 2``. Epoch 0 trains at the
    initial rate and the schedule reaches zero only at ``max_epochs``, so the last epoch
    trains at a small nonzero rate.

    Args:
        config: The protocol.
        epoch: Zero-based epoch index.

    Returns:
        The rate.
    """
    return config.lr * (1.0 + math.cos(math.pi * epoch / config.max_epochs)) / 2.0


def train(
    model: nn.Module,
    train_segments: Sequence[Segment],
    test_segments: Sequence[Segment],
    config: TrainConfig,
) -> Report:
    """Run the protocol.

    Args:
        model: The model, moved to ``config.device`` here.
        train_segments: Train pool.
        test_segments: Test pool.
        config: The protocol.

    Returns:
        A :class:`Report`.
    """
    device = torch.device(config.device)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    points: list[Point] = []
    stopped_early = False
    for epoch in range(config.max_epochs):
        rate = lr_at(config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        losses: list[float] = []
        for batch in batches(train_segments, config.batch_size):
            inputs, labels = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config):
                logits = model(inputs)
            loss = loss_fn(logits.float().flatten(0, -2), labels.flatten())
            auxiliary = _auxiliary_loss(model)
            if auxiliary is not None:
                loss = loss + auxiliary
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, test_segments, config)
        points.append(
            Point(
                epoch=epoch,
                lr=rate,
                train_loss=sum(losses) / len(losses) if losses else math.nan,
                test=metrics,
            )
        )
        if metrics.example > config.early_stopping_threshold:
            stopped_early = True
            break
    best_epoch = max(range(len(points)), key=lambda index: points[index].test.example)
    return Report(
        best=points[best_epoch].test,
        best_epoch=best_epoch,
        final=points[-1].test,
        points=tuple(points),
        epochs_run=len(points),
        stopped_early=stopped_early,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, segments: Sequence[Segment], config: TrainConfig
) -> Metrics:
    """Score a pool.

    Args:
        model: The model, already on ``config.device``.
        segments: Segments to score.
        config: The protocol; supplies the eval batch size and the precision.

    Returns:
        A :class:`Metrics`.

    Raises:
        ValueError: If the pool holds no supervised position.
    """
    device = torch.device(config.device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    total_loss = 0.0
    correct = 0
    supervised = 0
    fractions: list[float] = []
    grouped: dict[str, dict[str, list[float]]] = {}
    for batch in batches(segments, config.eval_batch_size):
        inputs, labels = _to_device(batch, device)
        with _autocast(config):
            logits = model(inputs)
        flat = logits.float().flatten(0, -2)
        total_loss += loss_fn(flat, labels.flatten()).item()
        counted = labels != IGNORE_INDEX
        hits = (logits.argmax(dim=-1) == labels) & counted
        correct += int(hits.sum().item())
        supervised += int(counted.sum().item())
        per_example = (
            hits.sum(dim=-1).float() / counted.sum(dim=-1).clamp(min=1).float()
        )
        rows = [float(value) for value in per_example.tolist()]
        fractions.extend(rows)
        for key, value in batch.slices.items():
            grouped.setdefault(key, {}).setdefault(str(value), []).extend(rows)
    if supervised == 0:
        raise ValueError("the pool holds no supervised position")
    return Metrics(
        loss=total_loss / supervised,
        example=sum(fractions) / len(fractions),
        position=correct / supervised,
        by_slice={
            key: {value: sum(rows) / len(rows) for value, rows in values.items()}
            for key, values in grouped.items()
        },
    )


def _to_device(batch: Batch, device: torch.device) -> tuple[Tensor, Tensor]:
    inputs = torch.from_numpy(batch.inputs).to(device, non_blocking=True)
    labels = torch.from_numpy(batch.labels).to(device, non_blocking=True)
    return inputs, labels


def _autocast(config: TrainConfig) -> contextlib.AbstractContextManager[object]:
    """Autocast for ``bf16``, a null context for ``fp32``.

    Parameters stay float32 in both. A mixer whose kernel wants bf16 reaches it here and
    nowhere else, because it takes its input at its parameters' dtype.
    """
    if config.precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=torch.device(config.device).type, dtype=torch.bfloat16
    )


def _auxiliary_loss(model: nn.Module) -> Tensor | None:
    """Sum of every submodule's ``get_auxiliary_loss``, or None if none has one.

    Upstream's hook, kept: it is how a mixer carrying its own regularizer contributes to
    the objective without the trainer knowing about it.
    """
    terms: list[Tensor] = []
    for module in model.modules():
        hook = getattr(module, "get_auxiliary_loss", None)
        if callable(hook):
            term = hook()
            assert isinstance(term, Tensor), "get_auxiliary_loss must return a Tensor"
            terms.append(term)
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def batch_count(segments: Sequence[Segment], batch_size: int) -> int:
    """Batches one pass over ``segments`` yields, tail batches included."""
    return sum(math.ceil(segment.inputs.shape[0] / batch_size) for segment in segments)
