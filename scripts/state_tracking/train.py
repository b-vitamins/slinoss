"""The optimization protocol, and what an arm reports.

Every constant is `expressive-sparse-state-space-model`'s
``state_tracking_PyTorch/experiment_configs/*.json`` and the defaults of
``train.py::train_model``, which its twenty configs never move: 100001 steps of AdamW at
2e-3, batch 256, a linear warmup over the first tenth of the run then a cosine to 1e-5,
weight decay 1e-2 on everything but the embedding, evaluation every 5000 steps, and a stop
the first time validation accuracy passes 0.9995. Nothing here tunes. A task is tracked or
it is not, the same numbers run for every mixer, and the mixer is the only thing that
varies.

The learning-rate schedule is ``models/lr_scheduler.py``'s ``compute_lr`` to the letter,
including its edge: ``_LRScheduler.__init__`` steps once, so ``last_epoch`` is 0 at the
first update and the rate there is exactly zero. Optimizer step ``k`` runs at
``compute_lr(k)``. The first step of every run therefore moves nothing -- AdamW's decoupled
decay scales with the rate too -- and that is upstream's behaviour, one step in 100001.

Two upstream defects are not transcribed.

``train.py`` calls ``optimizer.step()`` twice per update, at lines 152 and 155, with
``optimizer.zero_grad()`` between them. Under modern PyTorch's default
``zero_grad(set_to_none=True)``, the second call sees every gradient as ``None`` and is
inert, including for AdamW's decoupled decay. The release leaves PyTorch unpinned, so an
older zero-to-tensor interpretation cannot be reconstructed as part of the protocol.
This module states the current executable semantics directly and steps once.

``optimizer.zero_grad()`` sits at the top of every micro-step, so under
``accumulation_steps > 1`` each micro-batch erases the last one's gradient and the update
is taken from the final micro-batch alone -- an accumulating run silently trains at
one-``accumulation_steps``-th of its nominal batch. Here the gradient is zeroed once per
optimizer step, before the window, and ``num_steps`` counts optimizer steps rather than
micro-batches. At the protocol's ``accumulation_steps`` of 1 the two readings coincide,
which is what every published config runs at.

The loss is masked, never keyed on an ignore index. On a group task label 0 is the
identity element as well as the pad token, so ``ignore_index=0`` would drop every position
whose running product happens to be the identity -- about one position in ``|G|`` of the
supervision, concentrated exactly where a tracker that has lost the state guesses.

Accuracy is banded by sequence length. The question on this axis is not whether an arm is
accurate but where in length it stops being accurate, and a single mean over a split
spanning lengths 40 to 256 hides the crossing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from scripts.state_tracking.instances import Batch, SplitConfig, batches
from scripts.state_tracking.tasks import Task

PRECISIONS = ("fp32", "bf16")
"""Forward-pass precisions. The protocol runs ``fp32``."""


@dataclass(frozen=True)
class TrainConfig:
    """The protocol. Defaults are `expressive-sparse-state-space-model`'s.

    Attributes:
        num_steps: Optimizer steps. Upstream's 100001.
        batch_size: Items per micro-batch.
        lr: Peak learning rate, at the end of warmup.
        final_lr: Floor the cosine decays to.
        warmup_fraction: Fraction of ``num_steps`` spent warming up from zero.
        weight_decay_embedding: Decay on any parameter whose name contains
            ``embedding``. Upstream's zero.
        weight_decay_others: Decay on everything else.
        early_stop_threshold: Stop the first time validation accuracy passes this.
        print_steps: Evaluate every this many steps. The final step always evaluates.
        accumulation_steps: Micro-batches per optimizer step. The effective batch is this
            times ``batch_size``.
        grad_clip: Gradient norm ceiling. Zero disables it, which is the protocol: no
            upstream tree on this axis clips.
        precision: A member of :data:`PRECISIONS`.
        band_width: Width of a length band in the report.
        seed: Seeds initialization and the split streams, through
            :func:`split_seeds`.
        device: Where the model and the batches live.
    """

    num_steps: int = 100001
    batch_size: int = 256
    lr: float = 0.002
    final_lr: float = 1e-5
    warmup_fraction: float = 0.1
    weight_decay_embedding: float = 0.0
    weight_decay_others: float = 1e-2
    early_stop_threshold: float = 0.9995
    print_steps: int = 5000
    accumulation_steps: int = 1
    grad_clip: float = 0.0
    precision: str = "fp32"
    band_width: int = 32
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"precision must be one of {PRECISIONS}, got {self.precision}"
            )
        for name in (
            "num_steps",
            "batch_size",
            "print_steps",
            "accumulation_steps",
            "band_width",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("lr", "final_lr"):
            value = getattr(self, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("weight_decay_embedding", "weight_decay_others", "grad_clip"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative, got {value}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(
                f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}"
            )

    @property
    def warmup_steps(self) -> int:
        """Steps of linear warmup, upstream's ``int(warmup_fraction * num_steps)``."""
        return int(self.warmup_fraction * self.num_steps)


class Band(NamedTuple):
    """Accuracy over the items of one length band.

    Attributes:
        low: Shortest length in the band.
        high: Longest length in the band, inclusive.
        positions: Supervised positions from items in it.
        accuracy: Their accuracy.
    """

    low: int
    high: int
    positions: int
    accuracy: float


class Metrics(NamedTuple):
    """One evaluation of one split.

    Attributes:
        loss: Mean cross entropy over supervised positions.
        accuracy: Supervised positions predicted correctly, over their count. This is
            upstream's number: ``val_acc / val_num``, pooled over positions and not over
            items, which matters on a group task where every position is supervised.
        positions: Supervised positions scored.
        bands: Per-length-band accuracy, in increasing length. An item's own length puts
            it in one band, so on a group task every position of an item shares a band.
    """

    loss: float
    accuracy: float
    positions: int
    bands: tuple[Band, ...]


class Point(NamedTuple):
    """One evaluation, at the step it was taken.

    Attributes:
        step: Zero-based optimizer step just finished.
        lr: The rate that step ran at.
        train_loss: Mean train loss per micro-batch since the previous evaluation.
        val: The validation split's metrics.
    """

    step: int
    lr: float
    train_loss: float
    val: Metrics


class Report(NamedTuple):
    """What one arm produced.

    Attributes:
        best: Metrics at the evaluation with the highest accuracy. Selection is by
            accuracy, not by loss: the threshold this axis reports against is an accuracy
            and a solved task's loss keeps falling after the accuracy has saturated.
        best_step: Its step.
        final: The last evaluation's metrics.
        points: Every evaluation, in order.
        steps_run: Optimizer steps taken.
        solved: Whether the early-stop threshold was reached.
    """

    best: Metrics
    best_step: int
    final: Metrics
    points: tuple[Point, ...]
    steps_run: int
    solved: bool


def split_seeds(seed: int) -> tuple[int, int]:
    """The train and validation split seeds for one run.

    Upstream runs the train split at ``seed`` and the validation split at ``2 * seed``.

    Args:
        seed: The run's seed. Note that at 0 both splits seed to 0 and the two streams
            coincide; upstream's own ``seed: 0`` configs have that property, and it is
            harmless because the two splits draw from disjoint length ranges.

    Returns:
        ``(train_seed, val_seed)``.
    """
    return seed, 2 * seed


def seed_all(seed: int) -> None:
    """Seed every generator the model draws from.

    Call before building the model: initialization is torch default draws, so the seed has
    to be set first for an arm to reproduce. The task streams do not read these
    generators -- each carries its own -- so a seed change moves the initialization and
    nothing about the data.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def lr_at(config: TrainConfig, step: int) -> float:
    """Learning rate for one optimizer step.

    ``models/lr_scheduler.py``'s ``compute_lr``, with its zero at step 0 and its floor
    from ``num_steps`` on.

    Args:
        config: The protocol.
        step: Zero-based optimizer step.

    Returns:
        The rate: ``lr * step / warmup_steps`` under warmup, then a half cosine from
        ``lr`` to ``final_lr`` over what remains, then ``final_lr``.
    """
    warmup = config.warmup_steps
    if step < warmup:
        return config.lr * step / warmup
    if step >= config.num_steps:
        return config.final_lr
    span = max(config.num_steps - warmup, 1)
    ratio = (step - warmup) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.final_lr + coeff * (config.lr - config.final_lr)


def parameter_groups(model: nn.Module, config: TrainConfig) -> list[dict[str, Any]]:
    """Split parameters into the two decay groups.

    The token embedding keeps upstream's name-based exemption. A parameter may also
    declare ``_no_weight_decay`` explicitly; this preserves the same transition
    operating-point treatment without coupling optimizer behavior to a parameter name.

    Args:
        model: The model.
        config: The protocol, for the two decays.

    Returns:
        Two AdamW groups, undecayed first.
    """
    plain: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        exempt = "embedding" in name or bool(getattr(param, "_no_weight_decay", False))
        (plain if exempt else decay).append(param)
    return [
        {"params": plain, "weight_decay": config.weight_decay_embedding},
        {"params": decay, "weight_decay": config.weight_decay_others},
    ]


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


def masked_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Cross entropy over supervised positions, at float32.

    Args:
        logits: ``(B,T,V)``, any dtype.
        targets: ``(B,T)`` int64.
        mask: ``(B,T)`` bool, True where supervised.

    Returns:
        Scalar, float32 whatever the forward pass ran at: a narrow softmax over the
        vocabulary loses the tail the loss is measuring.

    Raises:
        ValueError: When no position is supervised, which would return nan and train on
            it.
    """
    if not bool(mask.any()):
        raise ValueError("batch supervises no position")
    return nn.functional.cross_entropy(logits[mask].float(), targets[mask])


class _Tally:
    """Accumulator over evaluation batches.

    Args:
        band_width: Width of a length band.
    """

    def __init__(self, band_width: int) -> None:
        self.band_width = band_width
        self.loss = 0.0
        self.hits = 0
        self.positions = 0
        self.bands: dict[int, list[int]] = {}

    def add(self, logits: Tensor, batch: Batch) -> None:
        """Fold one batch in.

        Args:
            logits: ``(B,T,V)``.
            batch: The batch the logits came from.
        """
        mask = batch.mask
        count = int(mask.sum())
        if count == 0:
            return
        correct = (logits.argmax(-1) == batch.targets) & mask
        self.loss += float(masked_loss(logits, batch.targets, mask)) * count
        self.hits += int(correct.sum())
        self.positions += count
        rows = zip(
            batch.lengths.tolist(),
            correct.sum(-1).tolist(),
            mask.sum(-1).tolist(),
        )
        for length, hits, positions in rows:
            slot = self.bands.setdefault(length // self.band_width, [0, 0])
            slot[0] += hits
            slot[1] += positions

    def metrics(self) -> Metrics:
        """The split's metrics.

        Returns:
            Zeros on an empty split.
        """
        if self.positions == 0:
            return Metrics(0.0, 0.0, 0, ())
        bands = tuple(
            Band(
                index * self.band_width,
                (index + 1) * self.band_width - 1,
                positions,
                hits / positions if positions else 0.0,
            )
            for index, (hits, positions) in sorted(self.bands.items())
        )
        return Metrics(
            self.loss / self.positions,
            self.hits / self.positions,
            self.positions,
            bands,
        )


def stage(task: Task, split: SplitConfig, config: TrainConfig) -> tuple[Batch, ...]:
    """Generate a bounded split and put it on the device once.

    Upstream regenerates the split from its fixed seeds on every pass, which produces the
    identical items; holding them costs a few tens of megabytes and takes the generator off
    the evaluation path.

    Args:
        task: The task.
        split: The split. Must be bounded.
        config: The protocol, for the batch size and the device.

    Returns:
        The batches, on ``config.device``.

    Raises:
        ValueError: On an unbounded split, which has no last batch.
    """
    if split.count is None:
        raise ValueError(f"{task.name}: an unbounded split cannot be staged")
    return tuple(
        batch.to(config.device) for batch in batches(task, split, config.batch_size)
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, staged: tuple[Batch, ...], config: TrainConfig
) -> Metrics:
    """Score a staged split.

    Args:
        model: The model. Restored to its incoming train or eval mode.
        staged: What :func:`stage` returned.
        config: The protocol.

    Returns:
        The split's metrics.
    """
    was_training = model.training
    model.eval()
    tally = _Tally(config.band_width)
    for batch in staged:
        with _autocast(config):
            logits = model(batch.inputs)
        tally.add(logits, batch)
    model.train(was_training)
    return tally.metrics()


def _stream(task: Task, split: SplitConfig, config: TrainConfig) -> Iterator[Batch]:
    """The train batch stream, on the device.

    Args:
        task: The task.
        split: The train split, bounded or not.
        config: The protocol.

    Yields:
        Batches on ``config.device``.
    """
    for batch in batches(task, split, config.batch_size):
        yield batch.to(config.device)


def train(
    model: nn.Module,
    task: Task,
    train_split: SplitConfig,
    val_split: SplitConfig,
    config: TrainConfig,
    on_point: Callable[[Point], None] | None = None,
) -> Report:
    """Run the protocol.

    Args:
        model: Built, and already seeded.
        task: The task both splits draw from.
        train_split: Lengths 3 to 40 and unbounded, under the protocol.
        val_split: Lengths 40 to 256 and 8192 items, under the protocol. The two length
            ranges meeting only at 40 is what makes the number a length-generalization
            number.
        config: The protocol.
        on_point: Called with each evaluation as it happens, for a caller that streams
            progress. The report carries the same points.

    Returns:
        The report.

    Raises:
        ValueError: From :func:`stage` on an unbounded validation split, or from
            :func:`scripts.state_tracking.instances.batches` on a split under the task's
            length floor.
        StopIteration: When a bounded train split runs out mid-run. A run whose train
            split is bounded must supply at least ``num_steps * accumulation_steps``
            batches.
    """
    model.to(config.device)
    staged = stage(task, val_split, config)
    optimizer = torch.optim.AdamW(parameter_groups(model, config), lr=config.lr)
    stream = _stream(task, train_split, config)
    hook = getattr(model, "mask_grads", None)

    points: list[Point] = []
    best = Metrics(math.inf, -1.0, 0, ())
    best_step = -1
    window = 0.0
    micro = 0
    step = 0
    solved = False

    model.train()
    for step in range(config.num_steps):
        rate = lr_at(config, step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.accumulation_steps):
            batch = next(stream)
            with _autocast(config):
                logits = model(batch.inputs)
            loss = masked_loss(logits, batch.targets, batch.mask)
            (loss / config.accumulation_steps).backward()
            window += float(loss.detach())
            micro += 1
        if callable(hook):
            hook()
        if config.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        due = step % config.print_steps == 0 or step == config.num_steps - 1
        if not due:
            continue
        val = evaluate(model, staged, config)
        point = Point(step, rate, window / max(micro, 1), val)
        points.append(point)
        window = 0.0
        micro = 0
        if on_point is not None:
            on_point(point)
        if val.accuracy > best.accuracy:
            best, best_step = val, step
        if val.accuracy > config.early_stop_threshold:
            solved = True
            break

    # num_steps is validated positive and the final step always evaluates, so points is
    # never empty and the early-stop break happens only after an append.
    return Report(best, best_step, points[-1].val, tuple(points), step + 1, solved)
