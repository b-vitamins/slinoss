"""The multi-query associative recall generator.

One example is a context of ``num_kv_pairs`` key-value pairs laid down in adjacent
positions, followed by a query region in which every key reappears exactly once at an
even offset. The label at a query's own position is that key's value; every other
position is ignored. Recall is therefore scored at exactly ``num_kv_pairs`` positions per
example, and a model must hold all ``num_kv_pairs`` associations at once.

Layout of one row, at ``input_seq_len`` 16 and ``num_kv_pairs`` 2::

    position  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    input    k0  v0  k1  v1  k0   .   .   .   .   .   .   .   .   .  k1   .
    label     .   .   .   .  v0   .   .   .   .   .   .   .   .   .  v1   .

Keys are drawn without replacement from ``[1, vocab_size // 2)`` and values from
``[vocab_size // 2, vocab_size)``, so the two are disjoint and 0 is neither. Query
offsets are drawn without replacement from ``range(space)`` under a power law, then
doubled, so a query never lands on an odd position.

Numpy in, numpy out. No torch anywhere in this module, so the parity gate against the
upstream generators runs on a host with no GPU and no torch install.

Parity, stated exactly. Every structural draw matches upstream bit for bit: the seeding,
the per-row draw order (all keys, then all values, then all gaps), the power law, the
placement, and the label alignment. Two divergences, both deliberate:

1. Upstream draws the ``random_non_queries`` filler from torch's *global* generator, and
   its trainer seeds that generator and then builds the model before building the data.
   Upstream data therefore depends on how many random numbers the model's initializer
   happened to consume, which changes when a mixer changes. Here the filler comes from
   the same numpy generator as everything else, drawn after the gaps. Structural content
   is unaffected: at equal seed the two agree at every non-filler position, and the
   labels are identical. The filler is not part of any published comparison for that
   reason -- upstream's own figures are not reproducible at the filler position either.
2. Upstream's ``num_passes`` (a JRT-era knob for repeating the context) is not ported. It
   is unreachable and broken in both trees: its config class declares no such field, so
   the generator always receives the default 1, and calling the generator with 2 raises
   ``ValueError: could not broadcast input array from shape (n,k) into shape (n,2k)``. No
   published number used it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

IGNORE_INDEX = -100
"""Label at every unsupervised position, and ``nn.CrossEntropyLoss``'s own default."""


class Instance(NamedTuple):
    """One generated segment.

    Attributes:
        inputs: ``(num_examples, input_seq_len)`` int64 token ids in ``[0, vocab_size)``.
        labels: ``(num_examples, input_seq_len)`` int64, either a value token or
            ``IGNORE_INDEX``. Exactly ``num_kv_pairs`` positions per row are supervised.
        slices: The per-segment grouping keys the report slices accuracy by.
    """

    inputs: NDArray[np.int64]
    labels: NDArray[np.int64]
    slices: dict[str, int]


def gap_weights(space: int, power_a: float) -> NDArray[np.float64]:
    """Query-offset distribution over ``range(space)``.

    ``p[i] ~ power_a * (i + 1) ** (power_a - 1)``, normalized: the density of a power law
    with exponent ``power_a``. At ``power_a`` 1 it is uniform; the published setting 0.01
    concentrates queries near the context, which is what makes short-range recall
    learnable before long-range recall is.

    Args:
        space: Number of admissible offsets, ``(input_seq_len - context_size) // 2``.
        power_a: Power-law exponent, positive.

    Returns:
        ``(space,)`` float64 summing to 1.
    """
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    return p / p.sum()


def multiquery_ar(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int = 8,
    power_a: float = 0.01,
    random_non_queries: bool = True,
) -> Instance:
    """Generate one MQAR segment.

    Args:
        vocab_size: Token count. Must exceed ``input_seq_len``; the lower half supplies
            keys and the upper half values.
        num_examples: Rows to generate, at least 1.
        input_seq_len: Sequence length, even.
        seed: Seed for this segment's generator. Segments must not share one.
        num_kv_pairs: Associations per example. ``4 * num_kv_pairs <= input_seq_len``,
            which is what leaves room for the context and for one even-offset query
            position per pair.
        power_a: Exponent for the query-offset power law, positive.
        random_non_queries: Replace filler zeros with uniform tokens over the whole
            vocabulary. True is upstream's default and the modern reproduction leaves it
            there; the ICLR24 figure-2 sweep sets it False, which leaves the filler as
            the padding id 0. The two are different tasks: at False a model can find a
            query by looking for a nonzero token.

    Returns:
        An :class:`Instance`.

    Raises:
        ValueError: On any violated shape or vocabulary bound.
    """
    if num_examples < 1:
        raise ValueError(f"num_examples must be at least 1, got {num_examples}")
    if input_seq_len < 2 or input_seq_len % 2 != 0:
        raise ValueError(
            f"input_seq_len must be even and at least 2, got {input_seq_len}"
        )
    if vocab_size <= input_seq_len:
        raise ValueError(
            f"vocab_size must exceed input_seq_len, got {vocab_size} <= {input_seq_len}"
        )
    if num_kv_pairs < 1:
        raise ValueError(f"num_kv_pairs must be at least 1, got {num_kv_pairs}")
    if 4 * num_kv_pairs > input_seq_len:
        raise ValueError(
            f"num_kv_pairs {num_kv_pairs} needs input_seq_len at least "
            f"{4 * num_kv_pairs}, got {input_seq_len}"
        )
    if power_a <= 0.0:
        raise ValueError(f"power_a must be positive, got {power_a}")
    # No key-exhaustion guard: the bounds above already imply one. Keys come from
    # [1, vocab_size // 2), so there are vocab_size // 2 - 1 of them, and
    # vocab_size > input_seq_len >= 4 * num_kv_pairs gives vocab_size // 2 - 1 >=
    # 2 * num_kv_pairs - 1, which is at least num_kv_pairs.
    key_vocab_size = vocab_size // 2

    rng = np.random.RandomState(seed)
    # Draw order is load bearing: all keys, then all values, then all gaps, one call per
    # row within each. Upstream reaches this order through np.apply_along_axis over a
    # tiled choice array, which calls np.random.choice once per row in index order.
    keys = _choose_rows(rng, np.arange(1, key_vocab_size), num_examples, num_kv_pairs)
    values = _choose_rows(
        rng, np.arange(key_vocab_size, vocab_size), num_examples, num_kv_pairs
    )
    context_size = num_kv_pairs * 2
    space = (input_seq_len - context_size) // 2
    gaps = _choose_rows(
        rng, np.arange(space), num_examples, num_kv_pairs, p=gap_weights(space, power_a)
    )

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)
    labels = np.full((num_examples, input_seq_len + 1), IGNORE_INDEX, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    # Width is input_seq_len + 1; the shift by one is what puts a value's label on its
    # own key's position rather than one past it.
    inputs = examples[:, :-1].copy()
    targets = labels[:, 1:].copy()
    if random_non_queries:
        filler = rng.randint(0, vocab_size, size=inputs.shape)
        inputs = np.where(inputs == 0, filler, inputs)
    return Instance(
        inputs=inputs,
        labels=targets,
        slices={"input_seq_len": input_seq_len, "num_kv_pairs": num_kv_pairs},
    )


def _choose_rows(
    rng: np.random.RandomState,
    choices: NDArray[np.int64],
    num_rows: int,
    size: int,
    p: NDArray[np.float64] | None = None,
) -> NDArray[np.int64]:
    """``(num_rows, size)`` int64, each row an independent draw without replacement."""
    rows = [rng.choice(choices, size=size, replace=False, p=p) for _ in range(num_rows)]
    return np.stack(rows).astype(np.int64, copy=False)
