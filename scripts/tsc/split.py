"""The seed's partition, the one-hot targets, and the time channel.

The protocol has no fixed train/validation/test split. The archive's own split does not survive
processing -- deduplication reorders the pool, see :mod:`scripts.tsc.corpus` -- so the reference
draws a partition per seed and reports the mean over five of them. Reproducing that partition is
the whole point of this module and of :mod:`scripts.tsc.prng`: a harness that drew its own would
produce numbers that are not comparable to the published bars, and the difference would read as
a modelling result.

The chain, from the seed to the permutation, is the reference's::

    key = PRNGKey(seed)
    datasetkey, modelkey, trainkey, key = split(key, 4)
    permkey, _ = split(datasetkey)
    order = permutation(permkey, instances)
    train, val, test = order[:0.7N], order[0.7N:0.85N], order[0.85N:]

with both bounds truncated by ``int``. Three consequences to know before reading any UEA number,
none of them this harness's choice:

    the partition is not stratified      a rare class can miss a split entirely
    the test split is 15% of the pool    at 259 instances that is 39 items, so one item is
                                         0.026 accuracy and a five-seed mean has a wide spread
    ``modelkey`` and ``trainkey`` go     initialization and dropout are torch's here, so the
    unused                               seed fixes the data and the harness fixes the rest

The time channel is prepended before the partition, so every split carries the same
``[0, T/L, ..., T(L-1)/L]``. It stops one step short of ``T``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from scripts.tsc.corpus import Corpus
from scripts.tsc.prng import permutation, prng_key
from scripts.tsc.prng import split as split_key

__all__ = [
    "Arrays",
    "Partition",
    "apply",
    "one_hot",
    "partition",
    "prepare",
    "with_time",
]

TRAIN_BOUND = 0.7
"""Fraction of the pool that trains."""

VAL_BOUND = 0.85
"""Fraction of the pool up to and including validation."""


class Partition(NamedTuple):
    """One seed's three index vectors, into the processed pool's row order.

    Attributes:
        train: Rows that train.
        val: Rows that select. Early stopping and the reported test point both read this.
        test: Rows that are reported.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Rows per split.

        Returns:
            Train, validation and test counts.
        """
        return (int(self.train.size), int(self.val.size), int(self.test.size))


class Arrays(NamedTuple):
    """A dataset in the shape the loop consumes.

    Attributes:
        inputs: ``(N, L, d)`` float32, time channel first when it is present.
        targets: ``(N, C)`` float32 one-hot.
    """

    inputs: np.ndarray
    targets: np.ndarray

    def take(self, rows: np.ndarray) -> Arrays:
        """Select rows.

        Args:
            rows: Index vector.

        Returns:
            The selection, contiguous.
        """
        return Arrays(
            np.ascontiguousarray(self.inputs[rows]),
            np.ascontiguousarray(self.targets[rows]),
        )


def partition(instances: int, seed: int) -> Partition:
    """The seed's partition.

    Args:
        instances: Rows in the processed pool.
        seed: The protocol seed.

    Returns:
        The three index vectors.

    Raises:
        ValueError: On fewer than three instances, which cannot fill three splits, or on a
            bound that leaves a split empty. The reference produces the empty split silently
            and fails later inside the loop with a shape error.
    """
    if instances < 3:
        raise ValueError(f"{instances} instances cannot fill three splits")
    root = prng_key(seed)
    datasetkey = split_key(root, 4)[0]
    permkey = split_key(datasetkey, 2)[0]
    order = permutation(permkey, instances)
    first = int(instances * TRAIN_BOUND)
    second = int(instances * VAL_BOUND)
    found = Partition(order[:first], order[first:second], order[second:])
    if min(found.sizes) == 0:
        raise ValueError(f"{instances} instances split to {found.sizes}")
    return found


def one_hot(labels: np.ndarray) -> np.ndarray:
    """One-hot targets over the classes the pool still holds.

    The width is the number of distinct labels in the *deduplicated* pool, which is the
    reference's ``len(jnp.unique(labels))``. It equals the class count only when no class was
    wholly removed by deduplication; when it does not, the reference writes a column that is
    not the label's, silently, because an out-of-range scatter index clamps in JAX. That is
    refused here rather than reproduced.

    Args:
        labels: ``(N,)`` integer class indices.

    Returns:
        ``(N, C)`` float32.

    Raises:
        ValueError: When the labels do not cover ``0..C-1``, which is the case the reference
            corrupts.
    """
    present = np.unique(labels)
    width = int(present.size)
    if width == 0 or int(present[-1]) != width - 1 or int(present[0]) != 0:
        raise ValueError(
            f"labels hold classes {present.tolist()}, which is not 0..{width - 1}; "
            f"deduplication removed a whole class and the target width would be wrong"
        )
    out = np.zeros((labels.shape[0], width), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def with_time(data: np.ndarray, horizon: float) -> np.ndarray:
    """Prepend the time channel.

    Args:
        data: ``(N, L, d)`` float32.
        horizon: The reference's ``T``, always 1 in the published configs.

    Returns:
        ``(N, L, d + 1)`` float32, channel 0 holding ``[0, T/L, ..., T(L-1)/L]``.

    Raises:
        ValueError: On an empty sequence axis.
    """
    length = int(data.shape[1])
    if length < 1:
        raise ValueError("data has no timepoints")
    # Float32 throughout, as the reference's is: a float64 ramp concatenated onto float32
    # instances would promote the whole array and change every value the model sees.
    step = np.float32(horizon / length)
    ramp = step * np.arange(length, dtype=np.float32)
    channel = np.broadcast_to(ramp[None, :, None], (data.shape[0], length, 1))
    return np.concatenate([channel, data], axis=2, dtype=np.float32)


def prepare(corpus: Corpus, *, include_time: bool, horizon: float = 1.0) -> Arrays:
    """Turn a processed dataset into inputs and targets.

    Before the partition, as the reference does it, so the ramp is identical in every split.

    Args:
        corpus: From :func:`scripts.tsc.corpus.load`.
        include_time: Whether to prepend the time channel. Per dataset in the published
            configs.
        horizon: The reference's ``T``.

    Returns:
        The arrays.

    Raises:
        ValueError: From :func:`one_hot`.
    """
    inputs = with_time(corpus.data, horizon) if include_time else corpus.data
    return Arrays(np.ascontiguousarray(inputs), one_hot(corpus.labels))


def apply(arrays: Arrays, rows: Partition) -> tuple[Arrays, Arrays, Arrays]:
    """Cut the three splits.

    Args:
        arrays: From :func:`prepare`.
        rows: From :func:`partition`.

    Returns:
        Train, validation and test.
    """
    return (arrays.take(rows.train), arrays.take(rows.val), arrays.take(rows.test))
