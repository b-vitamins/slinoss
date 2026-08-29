"""Batching, with the reference's two iteration shapes reproduced exactly.

A split is small enough to live on the device whole -- the widest of the six is 290 MB and the
narrowest is 14 MB -- so a loader is an index generator over two resident tensors and there is
no worker pool, no collation and no host round trip per batch.

The reference has two iterators and they do not agree, which is deliberate on its part and
reproduced here rather than corrected:

    ``loop``    training. Redraws a permutation each pass and yields ``while end < size``, so
                the tail is dropped, and on a size that is an exact multiple of the batch a
                whole final batch is dropped with it. Runs forever.
    ``epoch``   evaluation. In order, one pass, and the trailing partial batch *is* yielded,
                so coverage is complete and every reported accuracy is over the whole split.

The dropped training tail is not corrected because the permutation is redrawn every pass, so
which instances are skipped varies and nothing is systematically excluded; what changes is how
many distinct instances one pass touches. Correcting it would make this harness's effective
sampling differ from the one the published bars were produced under, at no gain. The evaluation
iterator is the one that has to be exhaustive and it is.

Batch order is torch's generator, not JAX's. It does not need to match: the seed's job in this
protocol is to fix the partition, and the partition is :mod:`scripts.tsc.split`'s.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from scripts.tsc.split import Arrays

__all__ = ["Batch", "Loader", "generator_for"]


class Batch(NamedTuple):
    """One batch.

    Attributes:
        inputs: ``(B,L,d)`` float32.
        targets: ``(B,C)`` float32 one-hot.
    """

    inputs: Tensor
    targets: Tensor

    @property
    def size(self) -> int:
        """Instances in the batch.

        Returns:
            The count.
        """
        return int(self.inputs.shape[0])


def generator_for(
    seed: int, device: torch.device | str | None = None
) -> torch.Generator:
    """A generator for batch order, separate from the global RNG.

    A loader that reads the process RNG makes batch order depend on how many parameters the
    model happened to draw, so two arms at one seed see different data orders. This is the
    fix and it is why nothing here calls :func:`torch.manual_seed`.

    Args:
        seed: The protocol seed.
        device: Where the generator lives. The default device, which is the host, is right
            for an index permutation.

    Returns:
        The generator.
    """
    generator = torch.Generator(device="cpu" if device is None else device)
    generator.manual_seed(seed)
    return generator


class Loader:
    """One split, resident, with the reference's two iterators.

    Args:
        arrays: The split, from :func:`scripts.tsc.split.apply`.
        device: Where the two tensors live.

    Raises:
        ValueError: On an empty split, or on inputs and targets of different lengths. The
            reference raises on the empty case only when an iterator is first pulled, which is
            thousands of steps after the mistake.
    """

    def __init__(
        self, arrays: Arrays, device: torch.device | str | None = None
    ) -> None:
        if arrays.inputs.shape[0] != arrays.targets.shape[0]:
            raise ValueError(
                f"{arrays.inputs.shape[0]} inputs and {arrays.targets.shape[0]} targets"
            )
        if arrays.inputs.shape[0] == 0:
            raise ValueError("split is empty")
        self.inputs = torch.from_numpy(np.ascontiguousarray(arrays.inputs)).to(device)
        self.targets = torch.from_numpy(np.ascontiguousarray(arrays.targets)).to(device)

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def length(self) -> int:
        """Timepoints per instance.

        Returns:
            The count.
        """
        return int(self.inputs.shape[1])

    @property
    def channels(self) -> int:
        """Channels per instance, the time channel included when present.

        Returns:
            The count.
        """
        return int(self.inputs.shape[2])

    @property
    def classes(self) -> int:
        """Target width.

        Returns:
            The count.
        """
        return int(self.targets.shape[1])

    def check_batch_size(self, batch_size: int) -> None:
        """Refuse a batch size the reference refuses.

        Public because the loop calls it on all three splits before building anything, which is
        the only place a 32-instance batch against a 21-instance validation split can be caught
        early enough to be cheap.

        Args:
            batch_size: Instances per batch.

        Raises:
            ValueError: On a non-positive size, or one over the split's own size. The latter
                is the case that ends a lane: a published batch of 32 on a validation split of
                21 instances is a configuration that cannot run, and it has to say so before
                the model is built rather than at the first evaluation.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > len(self):
            raise ValueError(f"batch_size {batch_size} is over the split's {len(self)}")

    def _batch(self, rows: Tensor) -> Batch:
        """Gather one batch.

        Args:
            rows: Index tensor.

        Returns:
            The batch.
        """
        return Batch(self.inputs[rows], self.targets[rows])

    def loop(self, batch_size: int, generator: torch.Generator) -> Iterator[Batch]:
        """Training batches, forever.

        Args:
            batch_size: Instances per batch.
            generator: Fixes the permutation sequence.

        Yields:
            Batches. A pass over the split drops its tail, and a split whose size is an exact
            multiple of the batch drops a whole final batch; see the module docstring.

        Raises:
            ValueError: From :meth:`check_batch_size`.
        """
        self.check_batch_size(batch_size)
        size = len(self)
        if batch_size == size:
            # The reference's own special case: the whole split, forever, unshuffled. With one
            # batch per pass a permutation would change nothing.
            while True:
                yield Batch(self.inputs, self.targets)
        while True:
            order = torch.randperm(size, generator=generator).to(self.inputs.device)
            start, end = 0, batch_size
            while end < size:
                yield self._batch(order[start:end])
                start, end = end, end + batch_size

    def epoch(self, batch_size: int) -> Iterator[Batch]:
        """Evaluation batches: one pass, in order, complete.

        Args:
            batch_size: Instances per batch.

        Yields:
            Batches covering every instance exactly once.

        Raises:
            ValueError: From :meth:`check_batch_size`.
        """
        self.check_batch_size(batch_size)
        size = len(self)
        if batch_size == size:
            yield Batch(self.inputs, self.targets)
            return
        start, end = 0, batch_size
        while end < size:
            yield Batch(self.inputs[start:end], self.targets[start:end])
            start, end = end, end + batch_size
        yield Batch(self.inputs[start:], self.targets[start:])
