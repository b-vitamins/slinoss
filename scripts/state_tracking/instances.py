"""Splits and batches: seeds in, padded tensors out.

`structured-linear-cdes`'s ``data_dir/dataloaders.py``. Upstream wraps each task in a
``torch.utils.data.Dataset`` whose ``__init__`` draws one seed per item from
``random.Random(seed)`` and whose ``__getitem__`` generates that item on demand, then hands
the dataset to a ``DataLoader`` with its ``collate_fn``. The seed scheme, the draw order
and the padding rule are transcribed; the ``Dataset``/``DataLoader`` pair is not, because
at ``num_workers=0`` it is a loop, and the loop is here.

Two divergences, both in the train split.

The first is the size. Upstream's train split is 25,600,000 items and its length is only
ever consulted through ``__len__``, one pass of shuffled iteration, which at batch 256 is
100,000 steps -- the protocol's step count, to within one batch. Materializing that seed
list costs about a gigabyte of Python integers to describe a stream that is drawn once and
never revisited. An unbounded split here draws its seeds lazily instead, sequentially from
the same ``random.Random(seed)``, so the k-th item is bit-identical to upstream's k-th
seed. What is lost is the shuffle, which permutes an independent, identically distributed
sequence and moves no distribution.

The second follows from the first: with the split unbounded there is no epoch, so nothing
here drops a short tail. Every training batch is full.

A finite split -- every evaluation split -- is materialized exactly, in upstream's order,
so its number is reproducible item for item and a parity test can pin it.

The padding rule is upstream's, restated because it is the one place a harness silently
truncates. Upstream appends a zero row of width ``padding_length`` to the batch, calls
``pad_sequence(padding_value=0)``, and drops the row, so the batch width is
``max(longest item, padding_length)`` and a longer item widens the batch rather than being
cut. `expressive-sparse-state-space-model` sets ``padding_length`` to 0 throughout, making
every batch as wide as its longest item and no wider.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

from scripts.state_tracking.tasks import PAD_TOKEN, Sample, Task

SEED_LIMIT = 2**32
"""Exclusive bound on a per-item seed, upstream's ``random.randint(0, 2**32 - 1)``."""


@dataclass(frozen=True)
class SplitConfig:
    """One split of one task.

    Attributes:
        min_length: Shortest sequence.
        max_length: Longest sequence, inclusive. Length generalization is exactly this
            pair moving between the train and the evaluation split.
        seed: Seeds the per-item seed stream. Upstream runs the train split at ``seed``
            and the evaluation split at ``2 * seed``, which keeps the two streams from
            sharing a prefix.
        count: Items, or None for an unbounded split. Only a bounded split is
            materialized, and only a bounded split is evaluated.
        pad_to: Floor on the batch width. Zero pads each batch to its own longest item,
            which is the protocol's setting. A longer item is never truncated.
    """

    min_length: int
    max_length: int
    seed: int
    count: int | None = None
    pad_to: int = 0

    def __post_init__(self) -> None:
        if self.min_length < 1:
            raise ValueError(f"min_length must be positive, got {self.min_length}")
        if self.max_length < self.min_length:
            raise ValueError(
                f"max_length {self.max_length} is under min_length {self.min_length}"
            )
        if self.count is not None and self.count < 1:
            raise ValueError(f"count must be positive or None, got {self.count}")
        if self.pad_to < 0:
            raise ValueError(f"pad_to must not be negative, got {self.pad_to}")


class Batch(NamedTuple):
    """One padded batch, on whatever device it was built for.

    Attributes:
        inputs: ``(B,W)`` int64 tokens, right-padded with
            :data:`scripts.state_tracking.tasks.PAD_TOKEN`.
        targets: ``(B,W)`` int64 labels, zero where unsupervised or padded. Never read
            outside ``mask``: on a group task 0 is a real label as well as the pad, so a
            loss keyed on an ignore index rather than on the mask would drop every
            position whose running product is the identity.
        mask: ``(B,W)`` bool, True at a supervised position of a real item.
        lengths: ``(B,)`` int64, each item's length before padding. Carried so an
            evaluation can band its accuracy by length, which is the whole question on
            this axis.
    """

    inputs: Tensor
    targets: Tensor
    mask: Tensor
    lengths: Tensor

    @property
    def width(self) -> int:
        """Padded sequence length."""
        return int(self.inputs.shape[1])

    def to(self, device: str | torch.device) -> Batch:
        """The batch on ``device``.

        Args:
            device: Destination.

        Returns:
            A new batch. Returns itself untouched when already there, so a CPU run pays
            no copy.
        """
        target = torch.device(device)
        if self.inputs.device == target:
            return self
        return Batch(
            self.inputs.to(target),
            self.targets.to(target),
            self.mask.to(target),
            self.lengths.to(target),
        )


def seed_stream(seed: int) -> Iterator[int]:
    """Unbounded stream of per-item seeds.

    Args:
        seed: Split seed.

    Yields:
        Upstream's ``random.randint(0, 2**32 - 1)`` draws, in order, forever.
    """
    rng = random.Random(seed)
    while True:
        yield rng.randint(0, SEED_LIMIT - 1)


def seeds(seed: int, count: int) -> tuple[int, ...]:
    """The first ``count`` per-item seeds.

    Args:
        seed: Split seed.
        count: Items.

    Returns:
        Upstream's seed list, entry for entry.

    Raises:
        ValueError: On a non-positive count.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    stream = seed_stream(seed)
    return tuple(next(stream) for _ in range(count))


def _check(task: Task, split: SplitConfig) -> None:
    """Refuse a split the task cannot generate.

    Args:
        task: The task.
        split: The split.

    Raises:
        ValueError: When the split's shortest sequence is under the task's floor.
            ``mod_arith_w_brack`` is the case: at length 1 its expression is the ``=``
            alone, which upstream's own assertion refuses mid-generation.
    """
    if split.min_length < task.min_length:
        raise ValueError(
            f"{task.name}: min_length {split.min_length} is under the task's "
            f"{task.min_length}"
        )


def materialize(task: Task, split: SplitConfig) -> tuple[Sample, ...]:
    """Generate a bounded split whole.

    Args:
        task: The task.
        split: The split. Its ``count`` must be set.

    Returns:
        The samples, in upstream's seed order.

    Raises:
        ValueError: On an unbounded split, or a split under the task's length floor.
    """
    if split.count is None:
        raise ValueError(f"{task.name}: an unbounded split cannot be materialized")
    _check(task, split)
    return tuple(
        task.sample(item_seed, split.min_length, split.max_length)
        for item_seed in seeds(split.seed, split.count)
    )


def collate(samples: Sequence[Sample], pad_to: int = 0) -> Batch:
    """Pad a list of samples into one batch.

    Args:
        samples: The items. Non-empty.
        pad_to: Floor on the width.

    Returns:
        The batch, width ``max(longest item, pad_to)``.

    Raises:
        ValueError: On an empty batch.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    lengths = [len(sample.ids) for sample in samples]
    width = max(max(lengths), pad_to)
    inputs = torch.full((len(samples), width), PAD_TOKEN, dtype=torch.long)
    targets = torch.zeros((len(samples), width), dtype=torch.long)
    mask = torch.zeros((len(samples), width), dtype=torch.bool)
    for row, sample in enumerate(samples):
        span = len(sample.ids)
        inputs[row, :span] = torch.tensor(sample.ids, dtype=torch.long)
        targets[row, :span] = torch.tensor(sample.targets, dtype=torch.long)
        mask[row, :span] = torch.tensor(sample.supervised, dtype=torch.bool)
    return Batch(inputs, targets, mask, torch.tensor(lengths, dtype=torch.long))


def batches(task: Task, split: SplitConfig, batch_size: int) -> Iterator[Batch]:
    """Iterate a split in batches.

    Args:
        task: The task.
        split: The split.
        batch_size: Items per batch.

    Yields:
        Batches on the CPU. A bounded split yields ``ceil(count / batch_size)`` of them,
        the last possibly short, so an evaluation covers every item exactly once. An
        unbounded split yields full batches forever.

    Raises:
        ValueError: On a non-positive batch size, or a split under the task's length
            floor.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    _check(task, split)
    if split.count is not None:
        pool = materialize(task, split)
        for start in range(0, len(pool), batch_size):
            yield collate(pool[start : start + batch_size], split.pad_to)
        return
    stream = seed_stream(split.seed)
    while True:
        chunk = [
            task.sample(next(stream), split.min_length, split.max_length)
            for _ in range(batch_size)
        ]
        yield collate(chunk, split.pad_to)
