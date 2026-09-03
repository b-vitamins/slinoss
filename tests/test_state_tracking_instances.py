"""Splits, seed streams and the padding rule.

Three failure modes here are silent. A seed stream that drifts changes which items an
evaluation is scored on, so two arms stop sharing a split. A padding rule that truncates
throws away the tail of exactly the long sequences the axis is about. A bounded split that
drops its last partial batch changes the denominator of every accuracy, by up to one batch
in the split.

The padding rule is pinned against widths measured from upstream's own ``collate_fn``,
which appends a zero row of ``padding_length``, calls
``torch.nn.utils.rnn.pad_sequence``, and drops the row: the width is
``max(longest item, padding_length)`` and a longer item widens the batch. At
``padding_length`` 0 -- what every published config on this axis runs -- that measured
identically to no padding argument at all.
"""

from __future__ import annotations

import random

import pytest
import torch

from scripts.state_tracking.instances import (
    SEED_LIMIT,
    SplitConfig,
    batches,
    collate,
    materialize,
    seed_stream,
    seeds,
)
from scripts.state_tracking.tasks import AUTOMATA, PAD_TOKEN, Sample, resolve

# Upstream's collate_fn over parity items at seeds 0, 1 and 7 from a split of lengths
# 3 to 40: item lengths 37, 10 and 16, measured widths 37 at padding_length 0, 128 at
# 128, and 37 with the argument absent.
COLLATE_SEEDS = (0, 1, 7)
COLLATE_LENGTHS = (37, 10, 16)
COLLATE_WIDTHS = {0: 37, 128: 128}


def test_seed_stream_is_the_stdlib_stream() -> None:
    """Per-item seeds are ``random.Random(seed).randint(0, 2**32 - 1)``, in order.

    The item at index k is a function of the k-th draw alone, so this fixes which items a
    split holds. Drawing from a different engine, or with a different bound, silently
    re-rolls the whole evaluation set.
    """
    for split_seed in (0, 1, 4096):
        rng = random.Random(split_seed)
        expected = [rng.randint(0, SEED_LIMIT - 1) for _ in range(8)]
        stream = seed_stream(split_seed)
        assert [next(stream) for _ in range(8)] == expected
        assert seeds(split_seed, 8) == tuple(expected)


def test_seed_prefixes_are_stable_under_the_count() -> None:
    """A longer split extends a shorter one; it does not re-roll it.

    Two arms at different ``--val-count`` share the smaller one's items, which is what
    makes a spot check at 512 items comparable with the protocol's 8192.
    """
    assert seeds(3, 16)[:5] == seeds(3, 5)
    with pytest.raises(ValueError, match="count must be positive"):
        seeds(3, 0)


def test_split_config_validation() -> None:
    """A malformed split is refused at construction.

    An inverted length range reaches the generator as a ``torch.randint`` with an empty
    interval, and a zero count reaches an evaluation as a division by zero.
    """
    SplitConfig(min_length=3, max_length=3, seed=0)
    with pytest.raises(ValueError, match="min_length must be positive"):
        SplitConfig(min_length=0, max_length=8, seed=0)
    with pytest.raises(ValueError, match="is under min_length"):
        SplitConfig(min_length=9, max_length=8, seed=0)
    with pytest.raises(ValueError, match="count must be positive or None"):
        SplitConfig(min_length=3, max_length=8, seed=0, count=0)
    with pytest.raises(ValueError, match="pad_to must not be negative"):
        SplitConfig(min_length=3, max_length=8, seed=0, pad_to=-1)


def _parity_items() -> tuple[Sample, ...]:
    """The three parity samples upstream's measured collate ran on."""
    task = AUTOMATA["parity"]
    return tuple(task.sample(seed, 3, 40) for seed in COLLATE_SEEDS)


def test_collate_width_is_the_longest_item_or_the_floor() -> None:
    """Width is ``max(longest item, pad_to)``, upstream's measured rule.

    A floor under the longest item does not truncate it. That is the one place a harness
    can quietly shorten a sequence, and on an axis whose whole question is length it would
    shorten exactly the items that carry the answer.
    """
    items = _parity_items()
    assert tuple(len(item.ids) for item in items) == COLLATE_LENGTHS
    for pad_to, width in COLLATE_WIDTHS.items():
        batch = collate(items, pad_to)
        assert batch.width == width
        assert batch.inputs.shape == (len(items), width)
    assert collate(items, 0).width == collate(items).width
    assert collate(items, 8).width == max(COLLATE_LENGTHS)


def test_collate_pads_with_the_pad_token_and_supervises_nothing_there() -> None:
    """Past an item's length the input is the pad token and the mask is False.

    The targets are zero there too, which on a group task is the identity element rather
    than a sentinel -- hence the mask. A True entry on a padded position would train the
    model on the pad.
    """
    items = _parity_items()
    batch = collate(items, 128)
    lengths = batch.lengths.tolist()
    assert lengths == list(COLLATE_LENGTHS)
    for row, length in enumerate(lengths):
        assert batch.inputs[row, :length].tolist() == list(items[row].ids)
        assert torch.all(batch.inputs[row, length:] == PAD_TOKEN)
        assert not bool(batch.mask[row, length:].any())
        assert torch.all(batch.targets[row, length:] == 0)
        assert int(batch.mask[row].sum()) == 1
        assert bool(batch.mask[row, length - 1])
    assert batch.inputs.dtype == torch.long
    assert batch.targets.dtype == torch.long
    assert batch.mask.dtype == torch.bool
    assert batch.lengths.dtype == torch.long


def test_collate_refuses_an_empty_batch() -> None:
    """An empty batch has no width, and its loss is nan rather than an error."""
    with pytest.raises(ValueError, match="cannot collate an empty batch"):
        collate([])


def test_group_task_collates_every_position_supervised() -> None:
    """A group split masks the whole item, not one position.

    The two supervision modes flow from the sample, not from the batcher, so this is the
    test that the batcher does not assume the automaton shape.
    """
    task = resolve("A5")
    split = SplitConfig(min_length=4, max_length=9, seed=0, count=6)
    for batch in batches(task, split, 6):
        for row, length in enumerate(batch.lengths.tolist()):
            assert int(batch.mask[row].sum()) == length


def test_bounded_split_covers_every_item_once() -> None:
    """A bounded split yields ``ceil(count / batch_size)`` batches, tail included.

    Dropping the tail would change an accuracy's denominator. Upstream's evaluation split
    is 8192 items at batch 256, which divides, so the defect would never show at the
    protocol's own numbers and would show at every spot check.
    """
    task = AUTOMATA["cycle_nav"]
    split = SplitConfig(min_length=4, max_length=12, seed=5, count=10)
    got = list(batches(task, split, 4))
    assert len(got) == 3
    assert [int(batch.inputs.shape[0]) for batch in got] == [4, 4, 2]
    pool = materialize(task, split)
    assert len(pool) == 10
    seen = [
        tuple(batch.inputs[row, : int(batch.lengths[row])].tolist())
        for batch in got
        for row in range(int(batch.inputs.shape[0]))
    ]
    assert seen == [item.ids for item in pool]


def test_unbounded_split_yields_full_batches_forever() -> None:
    """The train split has no epoch, so every training batch is full.

    It also does not repeat: consecutive batches come from consecutive draws of one
    stream, which is what replaces upstream's single shuffled pass.
    """
    task = AUTOMATA["parity"]
    split = SplitConfig(min_length=3, max_length=12, seed=1)
    stream = batches(task, split, 4)
    first, second, third = (next(stream) for _ in range(3))
    for batch in (first, second, third):
        assert int(batch.inputs.shape[0]) == 4
    expected = seeds(1, 8)
    got = [tuple(task.sample(item_seed, 3, 12).ids) for item_seed in expected]
    rows = [
        tuple(batch.inputs[row, : int(batch.lengths[row])].tolist())
        for batch in (first, second)
        for row in range(4)
    ]
    assert rows == got
    with pytest.raises(ValueError, match="batch_size must be positive"):
        next(batches(task, split, 0))


def test_unbounded_split_cannot_be_materialized() -> None:
    """Materializing an unbounded split would not terminate."""
    task = AUTOMATA["parity"]
    with pytest.raises(ValueError, match="cannot be materialized"):
        materialize(task, SplitConfig(min_length=3, max_length=8, seed=0))


def test_split_under_the_task_floor_is_refused() -> None:
    """``mod_arith_w_brack`` cannot generate at length 1, so the split is refused.

    Refused at the split rather than mid-generation, so the message names the task and the
    two numbers instead of surfacing as a recursion error a thousand items in.
    """
    task = AUTOMATA["mod_arith_w_brack"]
    split = SplitConfig(min_length=1, max_length=40, seed=0, count=4)
    with pytest.raises(ValueError, match="min_length 1 is under the task's 2"):
        materialize(task, split)
    with pytest.raises(ValueError, match="min_length 1 is under the task's 2"):
        next(batches(task, split, 2))


def test_batch_to_is_a_no_op_on_its_own_device() -> None:
    """A device move on the device the batch is already on returns the batch itself.

    Every training batch is moved, so a CPU run would otherwise pay a full copy of four
    tensors per step for nothing.
    """
    batch = collate(_parity_items())
    assert batch.to("cpu") is batch
    assert batch.to(torch.device("cpu")) is batch
