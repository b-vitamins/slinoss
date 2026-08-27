"""Parity gate for the MQAR generator.

Every fixture below was captured from both upstream copies at the stated settings --
zoology's ``zoology/data/multiquery_ar.py`` and the ICLR24-era vendored
``synthetic_tasks/zoology/src/zoology/data/associative_recall.py``. The two agree on
every case, so a fixture is one number rather than two, and any change to
:mod:`scripts.mqar.instances` that alters a draw fails here. An MQAR number is comparable
to a published one only if the data is the same data.

The digest cases span the whole draw space that matters: the published exponent 0.01 and
the two extremes 0.5 and 1.0, lengths 8 through 1024, key-value counts 2 through 256, and
vocabularies 32, 64 and 8192. They are taken at ``random_non_queries=False`` because the
filler is the one draw the port makes differently; it is pinned separately, against the
port itself, below.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.mqar.instances import IGNORE_INDEX, gap_weights, multiquery_ar

SMALL: dict[str, Any] = {
    "vocab_size": 64,
    "num_examples": 3,
    "input_seq_len": 16,
    "num_kv_pairs": 2,
    "seed": 0,
}
"""The case whose arrays are written out in full, so a failure is readable."""

SMALL_ROWS = (
    (
        "3 42 30 38 3 0 0 0 0 0 0 0 0 0 30 0",
        "x x x x 42 x x x x x x x x x 38 x",
    ),
    (
        "14 45 7 44 0 0 14 0 7 0 0 0 0 0 0 0",
        "x x x x x x 45 x 44 x x x x x x x",
    ),
    (
        "22 47 27 52 27 0 22 0 0 0 0 0 0 0 0 0",
        "x x x x 52 x 47 x x x x x x x x x",
    ),
)
"""``(inputs, labels)`` of :data:`SMALL`, one pair per row. ``x`` is ``IGNORE_INDEX``.

Written as text rather than as nested tuples so a row stays on one line under the
formatter, and with ``x`` for the ignored positions so the supervised ones are visible at
a glance. The alignment is the point of the fixture: a value sits on its own key's
position, not one past it.
"""

SMALL_FILLED_DIGEST = "c516ac68e84958f2f3d1a6f1c1d0dabf3b5d1cd54509e10aa954b1834b052d95"
"""Digest of :data:`SMALL` with the filler on. The port's own, not upstream's.

Upstream draws the filler from torch's global generator after its trainer has built a
model off that same generator, so its filler depends on the mixer under test and is not a
fixture anyone can capture. This pins the port's replacement: one draw from the same numpy
generator, taken after the gaps.
"""

UPSTREAM_DIGESTS = (
    (
        {
            "vocab_size": 64,
            "num_examples": 3,
            "input_seq_len": 16,
            "num_kv_pairs": 2,
            "seed": 0,
            "power_a": 0.01,
        },
        "b8094d8e7bc1930d21ab604f1b4772e9561608056d1c82a176855b6bf079518e",
    ),
    (
        {
            "vocab_size": 32,
            "num_examples": 5,
            "input_seq_len": 8,
            "num_kv_pairs": 2,
            "seed": 7,
            "power_a": 1.0,
        },
        "5a16ae79a5c187e9c2599aa2148b90f3d67075d862d27ee611467df0e9592f2f",
    ),
    (
        {
            "vocab_size": 64,
            "num_examples": 4,
            "input_seq_len": 32,
            "num_kv_pairs": 4,
            "seed": 1,
            "power_a": 0.5,
        },
        "3743dff1faef1affa78197c9decbe101a0736e2a646d238f9911bba15e48a78c",
    ),
    (
        {
            "vocab_size": 8192,
            "num_examples": 4,
            "input_seq_len": 64,
            "num_kv_pairs": 4,
            "seed": 12345,
            "power_a": 0.01,
        },
        "67982e42024f0fc9390e3fc9a528e8674ecf1b83247722fd34fe89d7312f02cb",
    ),
    (
        {
            "vocab_size": 8192,
            "num_examples": 3,
            "input_seq_len": 256,
            "num_kv_pairs": 64,
            "seed": 99,
            "power_a": 0.01,
        },
        "02f1cc18a97abcdc49e51d29fb29e9f836181c4fa64b8fb09199613429e31a21",
    ),
    (
        {
            "vocab_size": 8192,
            "num_examples": 2,
            "input_seq_len": 1024,
            "num_kv_pairs": 256,
            "seed": 2024,
            "power_a": 0.01,
        },
        "8c3d897b271bdf64135cc4006a7ca067534e412bd7073b4e3e98e767b21dd9b9",
    ),
)
"""``(settings, sha256)`` at ``random_non_queries=False``, from both upstream copies."""

UPSTREAM_GAP_WEIGHTS = (
    (
        6,
        0.01,
        (
            0.4053657496500885,
            0.20409264570652952,
            0.13661456673621003,
            0.10275611115997471,
            0.08238852865038684,
            0.06878239809681035,
        ),
    ),
    (4, 1.0, (0.25, 0.25, 0.25, 0.25)),
)
"""``(space, power_a, weights)``, from upstream's own two lines."""


def digest(inputs: NDArray[np.int64], labels: NDArray[np.int64]) -> str:
    """sha256 over the two arrays, row by row, inputs then labels per row.

    Row-interleaved rather than array-at-a-time so a shifted label alignment cannot cancel
    against a shifted input.

    Args:
        inputs: ``(rows, length)``.
        labels: ``(rows, length)``.

    Returns:
        The hex digest.
    """
    running = hashlib.sha256()
    for row_in, row_lab in zip(inputs, labels):
        running.update(np.ascontiguousarray(row_in, dtype=np.int64).tobytes())
        running.update(np.ascontiguousarray(row_lab, dtype=np.int64).tobytes())
    return running.hexdigest()


def row(text: str) -> list[int]:
    """Read one fixture row; ``x`` is :data:`IGNORE_INDEX`."""
    return [IGNORE_INDEX if token == "x" else int(token) for token in text.split()]


def test_small_case_matches_upstream_arrays() -> None:
    """The whole array, written out, so a draw-order regression is readable."""
    instance = multiquery_ar(**SMALL, random_non_queries=False)
    assert instance.inputs.tolist() == [row(inputs) for inputs, _ in SMALL_ROWS]
    assert instance.labels.tolist() == [row(labels) for _, labels in SMALL_ROWS]


@pytest.mark.parametrize(("settings", "expected"), UPSTREAM_DIGESTS)
def test_generator_matches_upstream(settings: dict[str, Any], expected: str) -> None:
    """Both upstream copies produce this digest at these settings."""
    instance = multiquery_ar(**settings, random_non_queries=False)
    assert digest(instance.inputs, instance.labels) == expected


def test_filler_draw_is_pinned() -> None:
    """The port's filler, which upstream's is not reproducible against."""
    instance = multiquery_ar(**SMALL, random_non_queries=True)
    assert digest(instance.inputs, instance.labels) == SMALL_FILLED_DIGEST


def test_filler_touches_only_the_padding_positions() -> None:
    """The invariant that makes the filler divergence harmless.

    At equal seed the two settings agree at every position the structure occupies and the
    labels are identical, so the filler cannot move a key, a value, a query or a target.
    It is drawn after the gaps, so it also cannot perturb them.
    """
    padded = multiquery_ar(**SMALL, random_non_queries=False)
    filled = multiquery_ar(**SMALL, random_non_queries=True)
    occupied = padded.inputs != 0
    assert np.array_equal(filled.inputs[occupied], padded.inputs[occupied])
    assert np.array_equal(filled.labels, padded.labels)
    assert (filled.inputs[~occupied] != 0).any()


@pytest.mark.parametrize(("space", "power_a", "expected"), UPSTREAM_GAP_WEIGHTS)
def test_gap_weights_match_upstream(
    space: int, power_a: float, expected: tuple[float, ...]
) -> None:
    """The query-offset density, including that ``power_a`` 1 is uniform."""
    assert gap_weights(space, power_a).tolist() == pytest.approx(list(expected))


@pytest.mark.parametrize(
    ("vocab_size", "input_seq_len", "num_kv_pairs"),
    [(64, 16, 2), (8192, 256, 64), (32, 8, 2)],
)
def test_every_supervised_position_recalls_its_own_pair(
    vocab_size: int, input_seq_len: int, num_kv_pairs: int
) -> None:
    """The task's defining property, read off the arrays alone.

    Exactly ``num_kv_pairs`` positions per row are supervised, each sits at an even offset
    into the query region, and the label there is the value that followed that same key in
    the context. Checked without reference to the internals, so a placement that happened
    to match the digests but not the task would still fail.
    """
    instance = multiquery_ar(
        vocab_size=vocab_size,
        num_examples=8,
        input_seq_len=input_seq_len,
        seed=3,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=False,
    )
    context_size = 2 * num_kv_pairs
    for inputs, labels in zip(instance.inputs, instance.labels):
        supervised = np.flatnonzero(labels != IGNORE_INDEX)
        assert supervised.shape[0] == num_kv_pairs
        assert ((supervised - context_size) % 2 == 0).all()
        assert (supervised >= context_size).all()
        pairs = dict(zip(inputs[0:context_size:2], inputs[1:context_size:2]))
        assert len(pairs) == num_kv_pairs
        for position in supervised:
            assert labels[position] == pairs[inputs[position]]


def test_keys_and_values_come_from_disjoint_halves() -> None:
    """Keys from ``[1, vocab_size // 2)``, values from ``[vocab_size // 2, vocab_size)``.

    Which is what makes the padding id 0 neither, and what lets the filler be told apart
    from the structure in the test above.
    """
    vocab_size = 64
    instance = multiquery_ar(
        vocab_size=vocab_size,
        num_examples=16,
        input_seq_len=16,
        seed=5,
        num_kv_pairs=2,
        random_non_queries=False,
    )
    context = instance.inputs[:, :4]
    keys, values = context[:, 0::2], context[:, 1::2]
    assert keys.min() >= 1
    assert keys.max() < vocab_size // 2
    assert values.min() >= vocab_size // 2
    assert values.max() < vocab_size
    targets = instance.labels[instance.labels != IGNORE_INDEX]
    assert targets.min() >= vocab_size // 2


def test_slices_are_the_two_keys_the_port_reports() -> None:
    """Upstream also reports ``num_passes``, which is not ported.

    Its config class declares no such field, so upstream always passes the default 1, and
    calling upstream's generator with 2 raises on a broadcast. A slice key that is always
    the same constant groups nothing.
    """
    instance = multiquery_ar(**SMALL, random_non_queries=False)
    assert instance.slices == {"input_seq_len": 16, "num_kv_pairs": 2}
    assert "num_passes" not in inspect.signature(multiquery_ar).parameters


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"num_examples": 0}, "num_examples"),
        ({"input_seq_len": 15}, "even"),
        ({"input_seq_len": 0}, "even"),
        ({"vocab_size": 16}, "must exceed input_seq_len"),
        ({"num_kv_pairs": 0}, "num_kv_pairs must be at least 1"),
        ({"num_kv_pairs": 5}, "needs input_seq_len at least 20"),
        ({"power_a": 0.0}, "power_a must be positive"),
    ],
)
def test_out_of_contract_settings_raise(settings: dict[str, Any], message: str) -> None:
    """Every bound the generator relies on, checked at the call rather than in a kernel."""
    with pytest.raises(ValueError, match=message):
        multiquery_ar(**{**SMALL, **settings})


def test_dtypes_and_shapes_are_the_contract() -> None:
    """int64 throughout, both arrays ``(num_examples, input_seq_len)``."""
    instance = multiquery_ar(**SMALL, random_non_queries=True)
    assert instance.inputs.shape == (3, 16)
    assert instance.labels.shape == (3, 16)
    assert instance.inputs.dtype == np.int64
    assert instance.labels.dtype == np.int64
