"""The window sampler: arithmetic at the end of the shard, and one order per seed.

Two failure modes, both silent. A window that overruns reads past the tokens the manifest
counted, which on a memmap is either a shorter row or another shard's bytes. An order that is
a function of a generator's history rather than of ``(seed, epoch)`` makes a resumed run a
different run, and makes two arms at one seed unpaired -- at which point a difference between
them is not the mixer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.lm.corpus import DTYPE
from scripts.lm.data import Shard, batches, val_batches, window_count

SEQ_LEN = 8


def _shard(tmp_path: Path, tokens: int) -> Shard:
    """A shard whose token at index ``i`` is ``i``."""
    path = tmp_path / "tokens.bin"
    np.arange(tokens, dtype=DTYPE).tofile(path)
    return Shard(path, tokens)


def _order(
    shard: Shard, *, seed: int, steps: int, start: int = 0, batch_size: int = 4
) -> list[torch.Tensor]:
    """The input tensors one run draws."""
    return [
        batch.inputs
        for batch in batches(
            shard,
            seq_len=SEQ_LEN,
            batch_size=batch_size,
            seed=seed,
            steps=steps,
            start=start,
        )
    ]


def test_window_count_leaves_room_for_the_last_target() -> None:
    """A window is ``seq_len + 1`` tokens, so the count is one short of the division.

    At exactly ``k * seq_len`` tokens the naive division gives ``k`` windows and the last
    one has no target for its last input.
    """
    assert window_count(2 * SEQ_LEN, SEQ_LEN) == 1
    assert window_count(2 * SEQ_LEN + 1, SEQ_LEN) == 2
    assert window_count(SEQ_LEN, SEQ_LEN) == 0
    assert window_count(0, SEQ_LEN) == 0
    with pytest.raises(ValueError, match="seq_len must be positive"):
        window_count(64, 0)


def test_the_last_window_does_not_overrun(tmp_path: Path) -> None:
    """Every window is full, and one past the last is refused rather than short."""
    shard = _shard(tmp_path, 3 * SEQ_LEN + 1)
    count = window_count(len(shard), SEQ_LEN)
    assert count == 3
    for index in range(count):
        window = shard.window(index, SEQ_LEN)
        assert window.shape == (SEQ_LEN + 1,)
        assert window[0] == index * SEQ_LEN
        assert int(window[-1]) < len(shard)
    with pytest.raises(IndexError, match="window 3 of 3"):
        shard.window(count, SEQ_LEN)


def test_a_shard_that_changed_size_is_refused(tmp_path: Path) -> None:
    """A grown or truncated file would move every window index silently."""
    shard = _shard(tmp_path, 64)
    with pytest.raises(ValueError, match="holds 64 tokens and the manifest says 65"):
        Shard(shard.path, 65)


def test_targets_are_the_inputs_shifted_by_one(tmp_path: Path) -> None:
    """Next-token prediction, and every token a target exactly once per epoch."""
    shard = _shard(tmp_path, 4 * SEQ_LEN + 1)
    batch = next(iter(batches(shard, seq_len=SEQ_LEN, batch_size=2, seed=0, steps=1)))
    assert batch.inputs.shape == (2, SEQ_LEN)
    assert batch.targets.shape == (2, SEQ_LEN)
    assert batch.inputs.dtype is torch.int64
    assert torch.equal(batch.inputs[:, 1:], batch.targets[:, :-1])


def test_one_seed_is_one_order(tmp_path: Path) -> None:
    """Two arms at one seed see the same text in the same order at every step.

    Without this a paired comparison is not paired and a margin between two arms carries
    the sample as well as the mixer.
    """
    shard = _shard(tmp_path, 32 * SEQ_LEN + 1)
    first = _order(shard, seed=7, steps=5)
    again = _order(shard, seed=7, steps=5)
    other = _order(shard, seed=8, steps=5)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_a_resumed_run_draws_what_an_uninterrupted_one_would(tmp_path: Path) -> None:
    """The order is a function of the absolute step, not of a generator's history.

    A resume that redrew from step zero would re-train on the batches already seen and
    report the result as a longer run.
    """
    shard = _shard(tmp_path, 32 * SEQ_LEN + 1)
    whole = _order(shard, seed=3, steps=6)
    resumed = _order(shard, seed=3, steps=6, start=3)
    assert len(resumed) == 3
    assert all(torch.equal(a, b) for a, b in zip(whole[3:], resumed, strict=True))


def test_an_epoch_boundary_draws_a_new_permutation(tmp_path: Path) -> None:
    """A run longer than one epoch reshuffles rather than repeating the first order."""
    shard = _shard(tmp_path, 4 * SEQ_LEN + 1)
    count = window_count(len(shard), SEQ_LEN)
    assert count == 4
    drawn = [
        int(row[0])
        for batch in batches(shard, seq_len=SEQ_LEN, batch_size=1, seed=1, steps=8)
        for row in batch.inputs
    ]
    first, second = drawn[:count], drawn[count:]
    assert sorted(first) == sorted(second)
    assert first != second


def test_validation_covers_the_shard_in_order(tmp_path: Path) -> None:
    """The held-out number is a sum over every window, not an estimate over a sample.

    The last batch is short rather than dropped, which is why the count divides unevenly
    here on purpose.
    """
    shard = _shard(tmp_path, 5 * SEQ_LEN + 1)
    seen = [
        int(row[0])
        for batch in val_batches(shard, seq_len=SEQ_LEN, batch_size=2)
        for row in batch.inputs
    ]
    assert seen == [index * SEQ_LEN for index in range(5)]
    sizes = [
        batch.inputs.shape[0]
        for batch in val_batches(shard, seq_len=SEQ_LEN, batch_size=2)
    ]
    assert sizes == [2, 2, 1]


def test_a_shard_with_no_window_is_refused(tmp_path: Path) -> None:
    """Training on a shard shorter than one window would yield nothing and report zero."""
    shard = _shard(tmp_path, SEQ_LEN)
    with pytest.raises(ValueError, match="holds no window"):
        next(iter(batches(shard, seq_len=SEQ_LEN, batch_size=1, seed=0, steps=1)))
