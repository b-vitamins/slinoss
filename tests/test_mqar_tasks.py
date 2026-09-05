"""Pools: the seed derivation, the two published specs, the batching order, leakage.

The seed goldens come from running upstream's own two lines at the same seed. The rest is
protocol: which segments a preset holds, how a pool is cut into batches, and what a batch
is allowed to contain. None of it is a preference, and all of it changes a number, so each
is pinned rather than described.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.mqar.instances import IGNORE_INDEX, multiquery_ar
from scripts.mqar.tasks import (
    FIGURE2_CELLS,
    FIGURE2_EXAMPLES,
    MAX_SEED,
    REPRO_TEST,
    REPRO_TRAIN,
    PoolSpec,
    Segment,
    SegmentSpec,
    batch_size_for,
    batches,
    build_pool,
    figure2_spec,
    leaked_fraction,
    repro_spec,
    segment_seeds,
)

UPSTREAM_SEEDS = (
    (
        123,
        (843828734, 914636141, 1228959102, 1840268610, 974319580),
        (
            2967327842,
            2367878886,
            3088727057,
            3090095699,
            4256823402,
            3964712059,
            3350193721,
        ),
    ),
    (0, (209652396,), (2546248239,)),
    (999, (1303213504, 648491356, 118207333), (4277223905, 2659063496)),
)
"""``(seed, train_seeds, test_seeds)``. The first is the repro pool's own 5 and 7."""

LEAKY = PoolSpec(
    train=(SegmentSpec(input_seq_len=8, num_kv_pairs=2, num_examples=3_000),),
    test=(SegmentSpec(input_seq_len=8, num_kv_pairs=2, num_examples=3_000),),
    vocab_size=32,
    random_non_queries=False,
    seed=17,
)
"""A pool small enough to collide. 15*14 ordered key pairs, 16*15 value pairs, 2 gap
orders: 100,800 distinct rows, against 9,000,000 train-test row pairs."""

LEAKY_FRACTION = 0.030333333333333334
"""Measured leakage of :data:`LEAKY`. Fixed seeds, so this is a constant, not a rate."""


def test_figure2_cells_match_the_iclr24_release() -> None:
    """The published sweep reaches its longest 512-token, 64-pair cell."""
    assert FIGURE2_CELLS == ((64, 4), (128, 8), (256, 16), (512, 64))


def hand_segment(rows: list[list[int]], num_kv_pairs: int) -> Segment:
    """A segment built from literal rows, for the pool-level functions.

    The generator is pinned in :mod:`tests.test_mqar_instances`; what is under test here
    takes segments as given, so a literal one keeps the failure local.

    Args:
        rows: Input rows.
        num_kv_pairs: The slice value to carry.

    Returns:
        A :class:`Segment` whose labels supervise the last position of every row.
    """
    inputs = np.asarray(rows, dtype=np.int64)
    labels = np.full_like(inputs, IGNORE_INDEX)
    labels[:, -1] = inputs[:, 0]
    return Segment(
        spec=SegmentSpec(
            input_seq_len=inputs.shape[1],
            num_kv_pairs=num_kv_pairs,
            num_examples=inputs.shape[0],
        ),
        seed=0,
        inputs=inputs,
        labels=labels,
        slices={"num_kv_pairs": num_kv_pairs},
    )


@pytest.mark.parametrize(("seed", "train", "test"), UPSTREAM_SEEDS)
def test_segment_seeds_match_upstream(
    seed: int, train: tuple[int, ...], test: tuple[int, ...]
) -> None:
    """One generator, train seeds drawn first. The order is part of the fixture."""
    assert segment_seeds(seed, len(train), len(test)) == (train, test)


@pytest.mark.parametrize("seed", [0, 1, 123, 999, 2**31 - 1])
def test_seed_halves_stay_disjoint(seed: int) -> None:
    """The structural leakage guard: no test segment can be handed a train seed."""
    train, test = segment_seeds(seed, 8, 8)
    assert max(train) < MAX_SEED // 2 <= min(test)


def test_repro_spec_is_the_published_pool() -> None:
    """Five train segments to length 256, seven test segments to 1024, filler on.

    Lengths 512 and 1024 appear in the test pool only, which is what puts length
    generalization inside the protocol rather than beside it.
    """
    spec = repro_spec()
    assert spec.train == REPRO_TRAIN
    assert spec.test == REPRO_TEST
    assert spec.vocab_size == 8192
    assert spec.random_non_queries is True
    assert spec.max_length == 1024
    assert max(segment.input_seq_len for segment in spec.train) == 256
    assert [segment.input_seq_len for segment in spec.test][-2:] == [512, 1024]


@pytest.mark.parametrize(("length", "pairs"), FIGURE2_CELLS)
def test_figure2_spec_is_one_matched_cell(length: int, pairs: int) -> None:
    """One train and one test segment at the same length and key-value count, filler off.

    The filler is the difference between the two published task variants, so a cell that
    took the repro's setting would be measuring something else.
    """
    train_examples, test_examples = FIGURE2_EXAMPLES
    spec = figure2_spec(length, pairs)
    assert spec.train == (SegmentSpec(length, pairs, train_examples),)
    assert spec.test == (SegmentSpec(length, pairs, test_examples),)
    assert spec.vocab_size == 8192
    assert spec.random_non_queries is False
    assert spec.max_length == length


def test_power_override_reaches_every_segment() -> None:
    """A preset's exponent is one value, applied to all of its segments."""
    spec = repro_spec(power_a=0.5)
    assert {segment.power_a for segment in spec.train + spec.test} == {0.5}


def test_max_length_spans_both_pools() -> None:
    """A mixer is built for the longest sequence anywhere in the pool, not in training."""
    spec = PoolSpec(
        train=(SegmentSpec(16, 2, 4),),
        test=(SegmentSpec(64, 4, 4),),
        vocab_size=128,
    )
    assert spec.max_length == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train": ()}, "at least one train segment"),
        ({"test": ()}, "at least one test segment"),
        ({"vocab_size": 16}, "must exceed every segment length"),
    ],
)
def test_pool_spec_rejects_an_impossible_pool(
    kwargs: dict[str, object], message: str
) -> None:
    """Caught at the spec, before a segment is generated or a model is built."""
    settings: dict[str, object] = {
        "train": (SegmentSpec(16, 2, 4),),
        "test": (SegmentSpec(16, 2, 2),),
        "vocab_size": 64,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PoolSpec(**settings)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("filler", [True, False])
def test_pool_carries_the_filler_setting_it_was_built_at(filler: bool) -> None:
    """Both onto the pool and into the data, because a record that omits it is unreadable.

    At ``False`` the non-query positions are the padding id and a model can find a query by
    looking for a nonzero token; at ``True`` they are uniform tokens. Two different tasks.
    """
    spec = PoolSpec(
        train=(SegmentSpec(16, 2, 8),),
        test=(SegmentSpec(16, 2, 4),),
        vocab_size=64,
        random_non_queries=filler,
    )
    pool = build_pool(spec, measure_leakage=False)
    assert pool.random_non_queries is filler
    assert pool.vocab_size == 64
    assert pool.max_length == 16
    seed = segment_seeds(spec.seed, 1, 1)[0][0]
    expected = multiquery_ar(
        vocab_size=64,
        num_examples=8,
        input_seq_len=16,
        seed=seed,
        num_kv_pairs=2,
        random_non_queries=filler,
    )
    assert pool.train[0].seed == seed
    assert np.array_equal(pool.train[0].inputs, expected.inputs)
    assert np.array_equal(pool.train[0].labels, expected.labels)
    assert pool.train[0].slices == expected.slices


def test_each_segment_gets_its_own_seed() -> None:
    """Two identical specs in one pool must not produce identical data.

    They share every setting, so the only thing that can separate them is the derived
    seed. A pool that seeded segments alike would report a train pool of duplicates as
    five times the data.
    """
    spec = PoolSpec(
        train=(SegmentSpec(16, 2, 8), SegmentSpec(16, 2, 8)),
        test=(SegmentSpec(16, 2, 4),),
        vocab_size=64,
    )
    pool = build_pool(spec, measure_leakage=False)
    first, second = pool.train
    assert first.seed != second.seed
    assert not np.array_equal(first.inputs, second.inputs)


def test_leaked_fraction_counts_exact_rows() -> None:
    """Whole-row equality, over the whole train pool, and 0 on an empty test pool."""
    train = hand_segment([[1, 2], [3, 4]], num_kv_pairs=1)
    test = hand_segment([[3, 4], [5, 6]], num_kv_pairs=1)
    assert leaked_fraction([train], [test]) == 0.5
    assert leaked_fraction([train], []) == 0.0


def test_leakage_is_measured_on_real_data_and_can_be_skipped() -> None:
    """A pool small enough to collide does collide, and the flag turns the check off.

    The seeds are derived, not drawn, so the fraction is a constant. The skipped case must
    report 0.0 rather than a stale or partial number.
    """
    assert build_pool(LEAKY).leaked == pytest.approx(LEAKY_FRACTION)
    assert build_pool(LEAKY, measure_leakage=False).leaked == 0.0


def test_batches_never_span_a_segment_and_keep_the_tail() -> None:
    """Upstream's order: segments in order, contiguous slices, short tail kept.

    A batch that spanned two segments would mix lengths, and a dropped tail would silently
    shorten the pool.
    """
    first = hand_segment([[i, i] for i in range(5)], num_kv_pairs=1)
    second = hand_segment([[i, i, i] for i in range(3)], num_kv_pairs=2)
    produced = list(batches([first, second], 2))
    assert [batch.inputs.shape for batch in produced] == [
        (2, 2),
        (2, 2),
        (1, 2),
        (2, 3),
        (1, 3),
    ]
    assert [batch.slices["num_kv_pairs"] for batch in produced] == [1, 1, 1, 2, 2]
    assert np.array_equal(
        np.concatenate([batch.inputs for batch in produced[:3]]), first.inputs
    )


def test_batch_order_is_identical_every_epoch() -> None:
    """Nothing is shuffled, so epoch 2 sees exactly what epoch 1 saw.

    That is the protocol every published MQAR number was measured under. It is reproduced
    here, not improved on.
    """
    segment = hand_segment([[i, i] for i in range(7)], num_kv_pairs=1)
    first = [batch.inputs.tolist() for batch in batches([segment], 3)]
    second = [batch.inputs.tolist() for batch in batches([segment], 3)]
    assert first == second


def test_batches_rejects_a_nonpositive_size() -> None:
    """An empty batch stream would report a loss of nan and an accuracy of zero."""
    segment = hand_segment([[1, 1]], num_kv_pairs=1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(batches([segment], 0))


@pytest.mark.parametrize(
    ("length", "expected"), [(2048, 64), (1024, 64), (512, 128), (256, 256), (64, 512)]
)
def test_batch_size_ladder(length: int, expected: int) -> None:
    """The figure-2 ladder, keyed on the pool's longest sequence.

    It holds activation memory roughly fixed across the length sweep, so a length cell is
    not also a batch-size cell.
    """
    assert batch_size_for(length) == expected
