"""The two iterators, including the reference's dropped training tail.

The reference's training iterator yields ``while end < size``, so a pass drops the tail and a
split whose size is an exact multiple of the batch drops a whole final batch. That is reproduced
rather than corrected, and it is pinned here because both halves of the argument for keeping it
are testable: nothing is systematically excluded, because the permutation is redrawn every pass,
and the *evaluation* iterator is exhaustive, because a reported accuracy has to be over the whole
split. A harness that quietly corrected the training tail would sample differently from the runs
the published bars came from, and one that inherited the drop into evaluation would report an
accuracy over a prefix.

The last test is the one that costs a day when it is missing: batch order must not come from the
process RNG. If it does, two arms at one seed see different data orders because their models drew
different numbers of parameters at construction, and the seed stops meaning anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.tsc.batching import Loader, generator_for
from scripts.tsc.split import Arrays


def make(size: int, *, length: int = 2, channels: int = 1, classes: int = 2) -> Loader:
    """A loader over ``size`` instances whose every value is the instance's own index.

    Args:
        size: Instances.
        length: Timepoints.
        channels: Channels.
        classes: Target width.

    Returns:
        The loader. ``inputs[i, 0, 0] == i``, so a batch names the rows it holds.
    """
    inputs = np.tile(
        np.arange(size, dtype=np.float32).reshape(size, 1, 1), (1, length, channels)
    )
    targets = np.zeros((size, classes), dtype=np.float32)
    targets[np.arange(size), np.arange(size) % classes] = 1.0
    return Loader(Arrays(inputs, targets))


def rows_of(batch: torch.Tensor) -> list[int]:
    """The instance indices a batch's inputs carry.

    Args:
        batch: ``(B,L,d)`` inputs from :func:`make`.

    Returns:
        The indices, in batch order.
    """
    return [int(value) for value in batch[:, 0, 0].tolist()]


def test_the_training_pass_drops_its_tail_and_redraws_the_permutation() -> None:
    """Nine of ten rows at batch 3, five of ten at batch 5, and every row seen across passes.

    The exact-multiple case is the surprising one: at batch 5 over 10 instances a pass yields one
    batch, not two. Both are the reference's ``while end < size``. The union over several passes is
    the whole split, which is the reason the drop is harmless and is asserted rather than argued.
    """
    for batch_size, per_pass in ((3, 9), (5, 5)):
        loader = make(10)
        stream = loader.loop(batch_size, generator_for(0))
        seen: set[int] = set()
        for _ in range(5):
            drawn = [
                row
                for _ in range(per_pass // batch_size)
                for row in rows_of(next(stream).inputs)
            ]
            assert len(drawn) == per_pass
            assert len(set(drawn)) == per_pass, (
                "a pass is a permutation, so no row repeats"
            )
            seen |= set(drawn)
        assert seen == set(range(10)), (batch_size, sorted(seen))


def test_a_whole_split_batch_is_the_references_unshuffled_special_case() -> None:
    """At ``batch_size == size`` the loop yields the split in file order, forever.

    One batch per pass makes a permutation a no-op on the loss, but not on the *order*, and the
    reference does not permute here. Reproduced so a small validation split's batch order is the
    reference's and not this harness's.
    """
    loader = make(6)
    stream = loader.loop(6, generator_for(0))
    for _ in range(3):
        assert rows_of(next(stream).inputs) == list(range(6))


@pytest.mark.parametrize("size", [1, 2, 5, 9, 10, 12])
def test_evaluation_covers_every_instance_exactly_once(size: int) -> None:
    """One pass, in order, complete, at every batch size the split admits.

    An accuracy over a prefix of a split is a number that looks like the reference's and is not,
    and the divisor and non-divisor cases take different branches -- the trailing partial batch is
    yielded after the loop, so an off-by-one there duplicates or loses the last instances.
    """
    loader = make(size)
    for batch_size in range(1, size + 1):
        drawn = [
            row for batch in loader.epoch(batch_size) for row in rows_of(batch.inputs)
        ]
        assert drawn == list(range(size)), (size, batch_size)


def test_a_batch_size_the_split_cannot_serve_is_refused_before_anything_is_built() -> (
    None
):
    """Non-positive, and over the split's own size.

    The second is the case that ends a lane: a published batch of 32 against a 21-instance
    validation split cannot run, and the loop checks all three splits up front so it says so
    before the model exists rather than at the first evaluation, minutes in.
    """
    loader = make(4)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        loader.check_batch_size(0)
    with pytest.raises(ValueError, match=r"batch_size 5 is over the split's 4"):
        loader.check_batch_size(5)
    with pytest.raises(ValueError, match="is over the split's"):
        next(loader.loop(5, generator_for(0)))
    with pytest.raises(ValueError, match="is over the split's"):
        next(loader.epoch(5))


def test_batch_order_is_fixed_by_the_seed_and_not_by_the_process_rng() -> None:
    """The same seed gives the same order however the global RNG has been used in between.

    This is what makes two arms at one seed comparable. A loader reading the process RNG ties data
    order to how many parameters a model drew at construction, so the wider arm sees different
    batches and the seed no longer fixes anything.
    """

    def order(seed: int, disturb: int) -> list[int]:
        torch.manual_seed(disturb)
        stream = make(10).loop(3, generator_for(seed))
        torch.randn(disturb + 1)
        return [row for _ in range(3) for row in rows_of(next(stream).inputs)]

    assert order(11, 0) == order(11, 7)
    assert order(11, 0) != order(12, 0)


def test_a_loader_refuses_a_split_it_cannot_iterate() -> None:
    """An empty split and a length mismatch, at construction.

    The reference raises on the empty case only when an iterator is first pulled. A validation
    split emptied by a truncated bound is the way that happens, and it is worth naming there.
    """
    with pytest.raises(ValueError, match="split is empty"):
        Loader(Arrays(np.zeros((0, 2, 1), np.float32), np.zeros((0, 2), np.float32)))
    with pytest.raises(ValueError, match="3 inputs and 2 targets"):
        Loader(Arrays(np.zeros((3, 2, 1), np.float32), np.zeros((2, 2), np.float32)))


def test_a_loader_reports_the_shape_the_model_is_built_from() -> None:
    """Length, channels and classes, which is where the model config comes from.

    A wrong ``channels`` builds an encoder at the wrong width and a wrong ``classes`` builds a
    head that cannot express the task, both of which train.
    """
    loader = make(5, length=7, channels=3, classes=4)
    assert (len(loader), loader.length, loader.channels, loader.classes) == (5, 7, 3, 4)
    batch = next(loader.epoch(2))
    assert batch.size == 2
    assert batch.inputs.dtype == torch.float32
    assert batch.targets.dtype == torch.float32
