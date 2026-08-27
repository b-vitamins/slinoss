"""Segment pools, seed derivation, batching, and the leakage measure.

A pool is a list of train segments and a list of test segments. Each segment is one call
to the generator at its own seed, and a batch never mixes segments, so a pool can hold
several lengths and several key-value counts at once. That is how the published protocol
gets length generalization for free: the modern repro trains on lengths up to 256 and
tests up to 1024.

Seed derivation is upstream's, exactly. One generator seeded at ``PoolSpec.seed`` draws
train seeds from ``[0, 2**31)`` and then test seeds from ``[2**31, 2**32)``. The disjoint
halves are the leakage guard: no test segment can ever be handed a train segment's seed.
The draw order matters -- train first, then test -- because both come from one stream.

The two published pools live here rather than in the driver, so the parity gate reaches
them on a host with no torch: :func:`repro_spec` and :func:`figure2_spec`.

Numpy only. The batching order here is upstream's too, and it is worth stating plainly
because it is unusual: segments are visited in order, each is cut into contiguous slices,
the short tail slice is kept, and nothing is ever shuffled. Every epoch therefore sees
the identical batch sequence. That is the protocol every published MQAR number was
measured under; it is reproduced, not improved.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from scripts.mqar.instances import Instance, multiquery_ar

MAX_SEED = 2**32
"""Seed space. The low half is train, the high half is test."""

FIGURE2_CELLS = ((64, 4), (128, 8), (256, 16))
"""``(input_seq_len, num_kv_pairs)`` cells the ICLR24 figure-2 sweep reports."""

FIGURE2_EXAMPLES = (100_000, 3_000)
"""Train and test example counts of a figure-2 cell."""

FIGURE2_FILLER = False
"""Figure 2 sets ``random_non_queries`` False, so its filler is the padding id."""

REPRO_FILLER = True
"""The modern reproduction leaves ``random_non_queries`` at upstream's default True.

Not an oversight on either side: the ``random_non_queries=False`` variant of the same
pools is a separate config file upstream. The filler decides whether a query can be found
by looking for a nonzero token, so a number measured at one setting is not comparable to
one measured at the other.
"""


@dataclass(frozen=True)
class SegmentSpec:
    """One generator call.

    Attributes:
        input_seq_len: Sequence length, even.
        num_kv_pairs: Associations per example; ``4 * num_kv_pairs <= input_seq_len``.
        num_examples: Rows.
        power_a: Exponent for the query-offset power law. 0.01 in every published config.
    """

    input_seq_len: int
    num_kv_pairs: int
    num_examples: int
    power_a: float = 0.01


@dataclass(frozen=True)
class PoolSpec:
    """A train pool and a test pool.

    ``vocab_size`` and ``random_non_queries`` sit here rather than on
    :class:`SegmentSpec` because they must agree across segments: the vocabulary is
    shared with the model's embedding, and the filler setting is a property of the task
    variant, not of one segment.

    Attributes:
        train: Train segments, at least one.
        test: Test segments, at least one.
        vocab_size: Token count, larger than every segment's ``input_seq_len``.
        random_non_queries: Fill non-query positions with uniform tokens instead of the
            padding id 0. Upstream's default, kept here so the port's default is
            upstream's everywhere; figure 2 is the config that turns it off. See
            :data:`REPRO_FILLER`.
        seed: Seed for the segment-seed derivation, not for any segment.
    """

    train: tuple[SegmentSpec, ...]
    test: tuple[SegmentSpec, ...]
    vocab_size: int = 8192
    random_non_queries: bool = True
    seed: int = 123

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("a pool needs at least one train segment")
        if not self.test:
            raise ValueError("a pool needs at least one test segment")
        longest = max(spec.input_seq_len for spec in self.train + self.test)
        if self.vocab_size <= longest:
            raise ValueError(
                f"vocab_size must exceed every segment length, got {self.vocab_size} "
                f"<= {longest}"
            )

    @property
    def max_length(self) -> int:
        """Longest sequence in the pool, train and test together.

        This is the length a mixer is built for. Upstream computes the same maximum over
        both pools and passes it to every mixer as ``l_max``.
        """
        return max(spec.input_seq_len for spec in self.train + self.test)


REPRO_TRAIN = (
    SegmentSpec(input_seq_len=64, num_kv_pairs=4, num_examples=100_000),
    SegmentSpec(input_seq_len=128, num_kv_pairs=8, num_examples=20_000),
    SegmentSpec(input_seq_len=256, num_kv_pairs=16, num_examples=20_000),
    SegmentSpec(input_seq_len=256, num_kv_pairs=32, num_examples=20_000),
    SegmentSpec(input_seq_len=256, num_kv_pairs=64, num_examples=20_000),
)
"""Train pool of zoology's current MQAR reproduction config, verbatim."""

REPRO_TEST = (
    SegmentSpec(input_seq_len=64, num_kv_pairs=4, num_examples=1_000),
    SegmentSpec(input_seq_len=64, num_kv_pairs=8, num_examples=1_000),
    SegmentSpec(input_seq_len=64, num_kv_pairs=16, num_examples=1_000),
    SegmentSpec(input_seq_len=128, num_kv_pairs=32, num_examples=1_000),
    SegmentSpec(input_seq_len=256, num_kv_pairs=64, num_examples=1_000),
    SegmentSpec(input_seq_len=512, num_kv_pairs=128, num_examples=1_000),
    SegmentSpec(input_seq_len=1024, num_kv_pairs=256, num_examples=1_000),
)
"""Test pool of the same config. Lengths 512 and 1024 are never trained on."""


def repro_spec(seed: int = 123, power_a: float = 0.01) -> PoolSpec:
    """The modern reproduction's pool.

    Args:
        seed: Pool seed.
        power_a: Query-offset exponent, at the published 0.01 unless overridden.

    Returns:
        A :class:`PoolSpec` over :data:`REPRO_TRAIN` and :data:`REPRO_TEST`, filler on.
    """
    return PoolSpec(
        train=tuple(_at_power(spec, power_a) for spec in REPRO_TRAIN),
        test=tuple(_at_power(spec, power_a) for spec in REPRO_TEST),
        vocab_size=8192,
        random_non_queries=REPRO_FILLER,
        seed=seed,
    )


def figure2_spec(
    input_seq_len: int, num_kv_pairs: int, seed: int = 123, power_a: float = 0.01
) -> PoolSpec:
    """One ICLR24 figure-2 cell: matched train and test length, filler off.

    Args:
        input_seq_len: Cell length. The published cells are :data:`FIGURE2_CELLS`.
        num_kv_pairs: Cell key-value count.
        seed: Pool seed.
        power_a: Query-offset exponent. Figure 2 passes ``train_power_a`` and
            ``test_power_a``, which its config class does not declare and pydantic drops,
            so the value it actually ran at is the default 0.01.

    Returns:
        A :class:`PoolSpec` with one train and one test segment.
    """
    train_examples, test_examples = FIGURE2_EXAMPLES
    return PoolSpec(
        train=(
            SegmentSpec(input_seq_len, num_kv_pairs, train_examples, power_a=power_a),
        ),
        test=(
            SegmentSpec(input_seq_len, num_kv_pairs, test_examples, power_a=power_a),
        ),
        vocab_size=8192,
        random_non_queries=FIGURE2_FILLER,
        seed=seed,
    )


class Segment(NamedTuple):
    """One generated segment and the seed that produced it.

    Attributes:
        spec: The spec that produced it.
        seed: The derived seed handed to the generator.
        inputs: ``(num_examples, input_seq_len)`` int64.
        labels: ``(num_examples, input_seq_len)`` int64, ``IGNORE_INDEX`` off-query.
        slices: Grouping keys for sliced accuracy.
    """

    spec: SegmentSpec
    seed: int
    inputs: NDArray[np.int64]
    labels: NDArray[np.int64]
    slices: dict[str, int]


class Pool(NamedTuple):
    """A built pool.

    Attributes:
        train: Built train segments, in spec order.
        test: Built test segments, in spec order.
        vocab_size: Carried through from the spec.
        max_length: Longest sequence over both pools.
        random_non_queries: Carried through from the spec, because it decides which of the
            two MQAR variants a number was measured on and a record that omits it cannot
            say which.
        leaked: Fraction of test rows whose input row also occurs in the train pool.
    """

    train: tuple[Segment, ...]
    test: tuple[Segment, ...]
    vocab_size: int
    max_length: int
    random_non_queries: bool
    leaked: float = 0.0


class Batch(NamedTuple):
    """One batch, drawn from exactly one segment.

    Attributes:
        inputs: ``(batch, input_seq_len)`` int64.
        labels: ``(batch, input_seq_len)`` int64.
        slices: The source segment's grouping keys.
    """

    inputs: NDArray[np.int64]
    labels: NDArray[np.int64]
    slices: dict[str, int]


def segment_seeds(
    seed: int, num_train: int, num_test: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive per-segment seeds from one pool seed.

    Args:
        seed: Pool seed.
        num_train: Train segment count.
        num_test: Test segment count.

    Returns:
        ``(train_seeds, test_seeds)``, drawn in that order from one generator. Train
        seeds lie in ``[0, 2**31)`` and test seeds in ``[2**31, 2**32)``.
    """
    rng = np.random.RandomState(seed)
    train = rng.randint(0, MAX_SEED // 2, size=num_train, dtype=np.int64)
    test = rng.randint(MAX_SEED // 2, MAX_SEED, size=num_test, dtype=np.int64)
    return tuple(int(value) for value in train), tuple(int(value) for value in test)


def build_pool(spec: PoolSpec, measure_leakage: bool = True) -> Pool:
    """Generate every segment in a pool.

    Args:
        spec: The pool spec.
        measure_leakage: Compare test rows against train rows. Exact, and quadratic in
            neither: it hashes rows into a set. Costs about a second at 100k rows.

    Returns:
        A :class:`Pool`.
    """
    train_seeds, test_seeds = segment_seeds(spec.seed, len(spec.train), len(spec.test))
    train = tuple(
        _build(segment, seed, spec) for segment, seed in zip(spec.train, train_seeds)
    )
    test = tuple(
        _build(segment, seed, spec) for segment, seed in zip(spec.test, test_seeds)
    )
    return Pool(
        train=train,
        test=test,
        vocab_size=spec.vocab_size,
        max_length=spec.max_length,
        random_non_queries=spec.random_non_queries,
        leaked=leaked_fraction(train, test) if measure_leakage else 0.0,
    )


def leaked_fraction(train: Sequence[Segment], test: Sequence[Segment]) -> float:
    """Fraction of test rows whose input row occurs verbatim in the train pool.

    The disjoint seed halves make a whole-segment collision impossible, but individual
    rows can still coincide by chance, and at a small vocabulary or a small length they
    do. Upstream carries this check only in dead commented-out code; it runs here.

    Args:
        train: Train segments.
        test: Test segments.

    Returns:
        A fraction in ``[0, 1]``. 0 when the test pool is empty.
    """
    seen = {
        (row.shape[0], row.tobytes()) for segment in train for row in segment.inputs
    }
    total = sum(segment.inputs.shape[0] for segment in test)
    if total == 0:
        return 0.0
    hits = sum(
        1
        for segment in test
        for row in segment.inputs
        if (row.shape[0], row.tobytes()) in seen
    )
    return hits / total


def batches(segments: Sequence[Segment], batch_size: int) -> Iterator[Batch]:
    """Cut segments into batches, in order, without shuffling.

    Args:
        segments: Segments to visit, in order.
        batch_size: Rows per batch. The last batch of each segment may be shorter.

    Yields:
        One :class:`Batch` per slice. A batch never spans two segments.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for segment in segments:
        rows = segment.inputs.shape[0]
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            yield Batch(
                inputs=segment.inputs[start:stop],
                labels=segment.labels[start:stop],
                slices=segment.slices,
            )


def batch_size_for(input_seq_len: int) -> int:
    """The published batch-size ladder, keyed on the longest sequence in the pool.

    From the ICLR24 figure-2 sweep: 64 at length 1024, 128 at 512, 256 at 256, and 512
    below that. It exists to hold activation memory roughly fixed across the length
    sweep, so a length cell is not also a batch-size cell.

    Args:
        input_seq_len: Longest sequence in the pool.

    Returns:
        Rows per batch.
    """
    if input_seq_len >= 1024:
        return 64
    if input_seq_len >= 512:
        return 128
    if input_seq_len >= 256:
        return 256
    return 512


def _at_power(spec: SegmentSpec, power_a: float) -> SegmentSpec:
    return SegmentSpec(
        input_seq_len=spec.input_seq_len,
        num_kv_pairs=spec.num_kv_pairs,
        num_examples=spec.num_examples,
        power_a=power_a,
    )


def _build(spec: SegmentSpec, seed: int, pool: PoolSpec) -> Segment:
    instance: Instance = multiquery_ar(
        vocab_size=pool.vocab_size,
        num_examples=spec.num_examples,
        input_seq_len=spec.input_seq_len,
        seed=seed,
        num_kv_pairs=spec.num_kv_pairs,
        power_a=spec.power_a,
        random_non_queries=pool.random_non_queries,
    )
    return Segment(
        spec=spec,
        seed=seed,
        inputs=instance.inputs,
        labels=instance.labels,
        slices=instance.slices,
    )
