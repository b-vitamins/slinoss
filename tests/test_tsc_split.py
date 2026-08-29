"""The seed's partition, the targets, and the time channel.

The partition is the part of this axis that decides comparability. It comes from JAX's stream
through :mod:`scripts.tsc.prng`, and the bounds truncate rather than round, so a validation split
is 15% of the pool with the remainder falling to test. At 259 instances that is 39 items, and one
item is 0.026 accuracy -- which is why the partition is pinned against JAX here rather than
checked for plausibility.

The other two tests cover cases the reference gets wrong silently. A pool that lost a whole class
to deduplication makes its one-hot width too narrow, and JAX's out-of-range scatter clamps rather
than raising, so the reference writes a column that is not the label's. And the ramp is float32:
a float64 ramp concatenated onto float32 instances promotes the whole array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.tsc.corpus import Corpus, process
from scripts.tsc.protocol import HORIZON, SEEDS
from scripts.tsc.split import (
    TRAIN_BOUND,
    VAL_BOUND,
    Arrays,
    apply,
    one_hot,
    partition,
    prepare,
    with_time,
)
from tests.test_tsc_corpus import write_archive

# Pool sizes spanning the range the six protocol datasets reach, including 500 where both
# bounds land on an integer and truncation is invisible, and odd sizes where it is not.
POOL_SIZES = (39, 259, 275, 405, 500, 561, 1751)


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    """A 24-instance synthetic dataset, three classes, no missing values."""
    return process(write_archive(tmp_path / "archive"), "Probe")


def test_the_partition_is_a_disjoint_cover_with_truncated_bounds() -> None:
    """The three splits cover the pool exactly once, at ``int(0.7N)`` and ``int(0.85N)``.

    A bound that rounded instead of truncating moves one instance between validation and test at
    most sizes, and one instance is 2.6 accuracy points on the smallest of the six.
    """
    for size in POOL_SIZES:
        for seed in SEEDS:
            rows = partition(size, seed)
            joined = np.concatenate(rows)
            assert np.array_equal(np.sort(joined), np.arange(size)), (size, seed)
            first, second = int(size * TRAIN_BOUND), int(size * VAL_BOUND)
            assert rows.sizes == (first, second - first, size - second), (size, seed)


def test_the_partition_chain_matches_jax() -> None:
    """The whole chain from seed to index vectors is the reference's expression, bit for bit.

    ``split(key, 4)[0]`` then ``split(that)[0]`` then ``permutation``. Taking a different row of
    either split, or splitting once instead of twice, gives a valid-looking partition that is not
    the one the published bars were measured on.
    """
    jax = pytest.importorskip(
        "jax", reason="the partition's ground truth is JAX itself"
    )
    jax.config.update("jax_threefry_partitionable", False)
    random = jax.random
    for size in POOL_SIZES:
        for seed in SEEDS:
            key = random.PRNGKey(seed)
            datasetkey = random.split(key, 4)[0]
            permkey = random.split(datasetkey)[0]
            order = np.asarray(random.permutation(permkey, size))
            first, second = int(size * TRAIN_BOUND), int(size * VAL_BOUND)
            rows = partition(size, seed)
            assert np.array_equal(rows.train, order[:first]), (size, seed)
            assert np.array_equal(rows.val, order[first:second]), (size, seed)
            assert np.array_equal(rows.test, order[second:]), (size, seed)


def test_a_pool_too_small_for_three_splits_is_refused() -> None:
    """Two instances cannot fill three splits, and at three the bounds leave one split empty.

    The reference produces the empty split and fails thousands of steps later inside the loop
    with a shape error that does not name the cause.
    """
    with pytest.raises(ValueError, match="cannot fill three splits"):
        partition(2, SEEDS[0])
    # int(3 * 0.7) == int(3 * 0.85) == 2, so validation gets nothing.
    with pytest.raises(ValueError, match=r"3 instances split to \(2, 0, 1\)"):
        partition(3, SEEDS[0])


def test_one_hot_refuses_a_pool_that_lost_a_whole_class() -> None:
    """The width is the classes present, and labels that do not cover ``0..C-1`` stop the run.

    That gap is what a deduplication that removed a whole class leaves behind. The reference's
    width comes out too narrow and JAX clamps the out-of-range scatter index, so it writes the
    wrong column and reports a number.
    """
    targets = one_hot(np.array([2, 0, 1, 1], dtype=np.int32))
    assert targets.shape == (4, 3)
    assert targets.dtype == np.float32
    assert np.array_equal(targets.sum(axis=1), np.ones(4, dtype=np.float32))
    assert targets[0].tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="removed a whole class"):
        one_hot(np.array([0, 2, 2], dtype=np.int32))


def test_the_time_channel_is_a_float32_ramp_that_stops_short_of_the_horizon() -> None:
    """Channel 0 is ``[0, T/L, ..., T(L-1)/L]``, in float32, and the data follows it.

    The last value is not ``T``: the ramp is ``T/L * arange(L)``. A ramp that reached ``T`` would
    be a different input at every timepoint, and a float64 one would promote the whole array.
    """
    data = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    out = with_time(data, HORIZON)
    assert out.shape == (2, 4, 4)
    assert out.dtype == np.float32
    ramp = np.float32(HORIZON / 4) * np.arange(4, dtype=np.float32)
    assert np.array_equal(out[0, :, 0], ramp)
    assert np.array_equal(out[1, :, 0], ramp)
    assert out[0, -1, 0] < HORIZON
    assert np.array_equal(out[..., 1:], data)
    with pytest.raises(ValueError, match="no timepoints"):
        with_time(np.zeros((1, 0, 3), dtype=np.float32), HORIZON)


def test_prepare_and_apply_put_the_same_ramp_in_every_split(corpus: Corpus) -> None:
    """The ramp is prepended before the partition, so all three splits carry an identical one.

    Prepending after the cut would give each split its own ramp over its own row count, which is
    not what the reference does and not what the bars were measured under.
    """
    plain = prepare(corpus, include_time=False, horizon=HORIZON)
    timed = prepare(corpus, include_time=True, horizon=HORIZON)
    assert plain.inputs.shape == (24, 3, 2)
    assert timed.inputs.shape == (24, 3, 3)
    assert timed.targets.shape == (24, 3)

    rows = partition(corpus.manifest.instances, SEEDS[0])
    assert rows.sizes == (16, 4, 4)
    ramp = np.float32(HORIZON / 3) * np.arange(3, dtype=np.float32)
    for arrays, count in zip(apply(timed, rows), rows.sizes, strict=True):
        assert arrays.inputs.shape == (count, 3, 3)
        assert np.array_equal(arrays.inputs[:, :, 0], np.tile(ramp, (count, 1)))
    # Rows travel together: split k's targets are the pool's targets at split k's indices.
    for arrays, index in zip(apply(timed, rows), rows, strict=True):
        assert np.array_equal(arrays.targets, timed.targets[index])


def test_take_returns_contiguous_selections() -> None:
    """A fancy-indexed selection is made contiguous before it reaches ``torch.from_numpy``.

    ``from_numpy`` on a non-contiguous array raises, and the loader takes the split whole, so a
    view here would end a lane at the first batch.
    """
    arrays = Arrays(
        np.arange(24, dtype=np.float32).reshape(4, 3, 2),
        np.eye(4, dtype=np.float32),
    )
    taken = arrays.take(np.array([3, 1]))
    assert taken.inputs.flags["C_CONTIGUOUS"]
    assert taken.targets.flags["C_CONTIGUOUS"]
    assert np.array_equal(taken.inputs[0], arrays.inputs[3])
