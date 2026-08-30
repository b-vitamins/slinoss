"""Read a token file. Deterministically, with numpy and torch and nothing else.

The corpus is a flat ``uint16`` array. A window is ``seq_len + 1`` consecutive tokens read at
a stride of ``seq_len``, split into inputs and targets by an offset of one, so every token in
the shard is a target exactly once per epoch and no window straddles the end.

Order is a permutation of the window index, drawn per epoch from ``(seed, epoch)``. Two arms
at one seed therefore see the same text in the same order at every step, which is what makes
a paired comparison paired: a difference between them is the mixer and not the sample. The
permutation is a function of the epoch rather than of a generator's history, so a run resumed
at step ``k`` draws what an uninterrupted run would have drawn.

Validation is the whole shard in order, no permutation and no sampling, so the held-out
number is a sum and not an estimate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from scripts.lm.corpus import DTYPE

__all__ = ["Batch", "Shard", "batches", "val_batches", "window_count"]


class Batch(NamedTuple):
    """One batch of next-token prediction.

    Attributes:
        inputs: ``(B,T)`` int64 token ids.
        targets: ``(B,T)`` int64 token ids, the inputs shifted by one.
    """

    inputs: Tensor
    targets: Tensor

    def to(self, device: torch.device | str) -> Batch:
        """Move both tensors.

        Args:
            device: Destination.

        Returns:
            A batch on ``device``.
        """
        return Batch(self.inputs.to(device), self.targets.to(device))


def window_count(tokens: int, seq_len: int) -> int:
    """Windows a shard holds at a stride of ``seq_len``.

    A window needs ``seq_len + 1`` tokens because the last input's target is the token
    after it, so the count is one short of the naive division.

    Args:
        tokens: Tokens in the shard.
        seq_len: Sequence length.

    Returns:
        The count, possibly zero.

    Raises:
        ValueError: On a non-positive sequence length.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return max(0, (tokens - 1) // seq_len)


class Shard:
    """One token file, memory mapped.

    Args:
        path: The ``.bin``.
        tokens: What the manifest says it holds.

    Raises:
        ValueError: When the file's length is not ``tokens``. A shard that grew or was
            truncated under a run would otherwise change the window index silently.
    """

    def __init__(self, path: Path, tokens: int) -> None:
        found = path.stat().st_size // np.dtype(DTYPE).itemsize
        if found != tokens:
            raise ValueError(
                f"{path} holds {found} tokens and the manifest says {tokens}"
            )
        self.path = path
        self.tokens = tokens
        self._array = np.memmap(path, dtype=DTYPE, mode="r")

    def __len__(self) -> int:
        return self.tokens

    def window(self, index: int, seq_len: int) -> np.ndarray:
        """One window's ``seq_len + 1`` tokens.

        Args:
            index: Window index, in ``[0, window_count(len(self), seq_len))``.
            seq_len: Sequence length.

        Returns:
            ``(seq_len + 1,)`` int64. A copy: the caller gets a tensor whose storage is
            not the mapping.

        Raises:
            IndexError: On a window past the end.
        """
        count = window_count(self.tokens, seq_len)
        if not 0 <= index < count:
            raise IndexError(f"window {index} of {count}")
        start = index * seq_len
        return np.asarray(self._array[start : start + seq_len + 1], dtype=np.int64)


def _batch(shard: Shard, indices: np.ndarray, seq_len: int) -> Batch:
    """Stack windows into a batch.

    Args:
        shard: Source.
        indices: Window indices.
        seq_len: Sequence length.

    Returns:
        The batch, on the CPU.
    """
    block = torch.from_numpy(
        np.stack([shard.window(int(index), seq_len) for index in indices])
    )
    return Batch(block[:, :-1].contiguous(), block[:, 1:].contiguous())


def _permutation(count: int, seed: int, epoch: int) -> np.ndarray:
    """The window order for one epoch.

    Args:
        count: Windows.
        seed: Run seed.
        epoch: Pass over the shard, from zero.

    Returns:
        A permutation of ``range(count)``.
    """
    return np.random.default_rng([seed, epoch]).permutation(count)


def batches(
    shard: Shard,
    *,
    seq_len: int,
    batch_size: int,
    seed: int,
    steps: int,
    start: int = 0,
) -> Iterator[Batch]:
    """Training batches, in the order ``seed`` fixes.

    Args:
        shard: Source.
        seq_len: Sequence length.
        batch_size: Windows per batch. This is the micro batch; accumulation is the
            trainer's, not this iterator's.
        seed: Run seed.
        steps: Last batch index, exclusive. A run longer than one epoch wraps into a new
            permutation rather than repeating the first.
        start: First batch index. A run resumed here draws exactly what an uninterrupted
            run would have drawn at that index, because the window position is a function
            of the absolute index and the permutation a function of ``(seed, epoch)``.

    Yields:
        One :class:`Batch` per step in ``[start, steps)``, on the CPU.

    Raises:
        ValueError: On a non-positive batch size, a negative start, or a shard with no
            window.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if start < 0:
        raise ValueError(f"start must not be negative, got {start}")
    count = window_count(shard.tokens, seq_len)
    if count < 1:
        raise ValueError(
            f"shard of {shard.tokens} tokens holds no window of {seq_len + 1}"
        )
    epoch = -1
    order = np.empty(0, dtype=np.int64)
    for step in range(start, steps):
        indices = np.empty(batch_size, dtype=np.int64)
        for slot in range(batch_size):
            position = step * batch_size + slot
            want = position // count
            if want != epoch:
                epoch = want
                order = _permutation(count, seed, epoch)
            indices[slot] = order[position % count]
        yield _batch(shard, indices, seq_len)


def val_batches(shard: Shard, *, seq_len: int, batch_size: int) -> Iterator[Batch]:
    """Every window of a shard, in order, in batches of at most ``batch_size``.

    The last batch is short rather than dropped, so the score covers the shard.

    Args:
        shard: Source.
        seq_len: Sequence length.
        batch_size: Windows per batch.

    Yields:
        One :class:`Batch` per group, on the CPU.

    Raises:
        ValueError: On a non-positive batch size.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    count = window_count(shard.tokens, seq_len)
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        yield _batch(shard, indices, seq_len)
