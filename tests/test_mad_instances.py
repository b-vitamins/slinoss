"""Parity gate for the MAD generators.

Every fixture below was captured by running `mad-lab`'s ``mad/data/instances.py`` at the
stated settings and generator seed. A change to :mod:`scripts.mad.instances` that alters
any draw fails here, which is the point: a MAD number is comparable to a published one
only if the data is the same data.

Each case draws several instances from one generator, so a generator that consumes the
wrong number of draws fails from the second instance on even when its first is right.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from scripts.mad.instances import (
    IGNORE_INDEX,
    Instance,
    build_kv_map,
    compression,
    fuzzy_in_context_recall,
    in_context_recall,
    memorization,
    noisy_in_context_recall,
    selective_copying,
)

ICR_MQ_TRAIN = (
    ("5 12 2 10 0 8 0 8 1 14 5 12 7 12 1", "12 2 10 0 8 0 8 1 14 5 12 7 12 1 14"),
    ("5 13 4 12 7 10 6 13 0 11 6 13 4 12 5", "13 4 12 7 10 6 13 0 11 6 13 4 12 5 13"),
    ("5 14 1 8 6 8 4 8 2 11 3 11 0 8 5", "14 1 8 6 8 4 8 2 11 3 11 0 8 5 14"),
)
ICR_MQ_TEST = (
    (
        "5 12 2 10 0 8 0 8 1 14 5 12 7 12 1",
        "-100 -100 -100 -100 -100 -100 8 -100 -100 -100 12 -100 -100 -100 14",
    ),
    (
        "5 13 4 12 7 10 6 13 0 11 6 13 4 12 5",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 13 -100 12 -100 13",
    ),
    (
        "5 14 1 8 6 8 4 8 2 11 3 11 0 8 5",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 14",
    ),
)
ICR_SQ_TEST = (
    (
        "4 12 6 11 5 13 1 7 2 9 6 11 6 11 15 4",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 12",
    ),
    (
        "5 8 5 8 0 10 5 8 2 9 1 12 1 12 15 1",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 12",
    ),
    (
        "3 11 4 11 3 11 6 13 5 12 4 11 2 14 15 6",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 13",
    ),
)
NICR_TEST = (
    (
        "4 8 1 9 20 20 6 10 3 8 0 14 24 23 3",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 8",
    ),
    (
        "2 8 23 22 22 24 2 8 23 23 5 15 2 8 5",
        "-100 -100 -100 -100 -100 -100 8 -100 -100 -100 -100 -100 8 -100 15",
    ),
    (
        "20 21 7 10 0 11 6 14 30 27 24 21 0 11 6",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 11 -100 14",
    ),
)
FICR_TRAIN = (
    (
        "15 15 15 15 15 5 3 9 10 2 5 11 14 2 7 14 0 3 10 5 4 12 14 5 14 6 2 8 13 5 4 12",
        "15 15 15 15 5 3 9 10 2 5 11 14 2 7 14 0 3 10 5 4 12 14 5 14 6 2 8 13 5 4 12 14",
    ),
    (
        "15 15 1 14 3 12 11 0 10 11 2 9 2 9 6 11 6 11 0 10 11 6 4 8 0 10 11 2 3 7 10 6",
        "15 1 14 3 12 11 0 10 11 2 9 2 9 6 11 6 11 0 10 11 6 4 8 0 10 11 2 3 7 10 6 11",
    ),
)
FICR_TEST = (
    (
        "15 15 15 5 0 14 13 3 4 11 12 6 5 10 11 1 4 8 11 0 2 10 7 3 6 7 6 2 12 3 4 11",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 11 12",
    ),
    (
        "15 15 15 15 15 2 4 9 12 6 3 9 4 3 7 6 2 13 10 4 0 9 5 0 11 8 2 4 9 12 4 5",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 9 12 -100 -100 13",
    ),
)
MEM_TRAIN = (
    ("4 15 3 15 3 15 3 15", "-100 9 -100 11 -100 11 -100 11"),
    ("5 15 1 15 4 15 4 15", "-100 7 -100 8 -100 9 -100 9"),
    ("2 15 1 15 2 15 3 15", "-100 14 -100 8 -100 14 -100 11"),
)
COMP_TRAIN = (
    ("10 14 13 7 14 14 14 15", "10 14 13 7 14 14 14 15"),
    ("1 6 9 4 5 9 12 15", "1 6 9 4 5 9 12 15"),
)
SC_TRAIN = (
    (
        "14 14 14 9 14 14 11 14 14 14 14 0 15 14 14 14",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 9 11 0",
    ),
    (
        "14 14 14 14 14 11 14 14 14 6 14 7 15 14 14 14",
        "-100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 11 6 7",
    ),
)

MEM_KV_MAP = build_kv_map(15, seed=12345)


class Case(NamedTuple):
    """One captured stream.

    Attributes:
        fixture: Rows of ``(inputs, targets)``, whitespace-separated integers.
        seed: Seeds the generator the whole case draws from.
        settings: Generator settings.
        blank_seed: Seeds selective copying's second stream, when it has one.
    """

    fixture: tuple[tuple[str, str], ...]
    seed: int
    settings: dict[str, Any]
    blank_seed: int | None = None


CASES: dict[str, Case] = {
    "icr multi-query train": Case(
        ICR_MQ_TRAIN,
        0,
        {"vocab_size": 16, "seq_len": 16, "is_training": True, "multi_query": True},
    ),
    "icr multi-query test": Case(
        ICR_MQ_TEST,
        0,
        {"vocab_size": 16, "seq_len": 16, "is_training": False, "multi_query": True},
    ),
    "icr single-query test": Case(
        ICR_SQ_TEST,
        7,
        {"vocab_size": 16, "seq_len": 16, "is_training": False, "multi_query": False},
    ),
    "nicr test": Case(
        NICR_TEST,
        1,
        {
            "vocab_size": 32,
            "seq_len": 16,
            "is_training": False,
            "multi_query": True,
            "noise_vocab_size": 16,
            "frac_noise": 0.5,
        },
    ),
    "ficr train": Case(
        FICR_TRAIN,
        2,
        {
            "vocab_size": 16,
            "seq_len": 32,
            "is_training": True,
            "multi_query": True,
            "k_motif_size": 2,
            "v_motif_size": 2,
        },
    ),
    "ficr test": Case(
        FICR_TEST,
        2,
        {
            "vocab_size": 16,
            "seq_len": 32,
            "is_training": False,
            "multi_query": True,
            "k_motif_size": 2,
            "v_motif_size": 2,
        },
    ),
    "mem": Case(
        MEM_TRAIN,
        3,
        {
            "vocab_size": 16,
            "seq_len": 8,
            "kv_map": MEM_KV_MAP,
        },
    ),
    "comp": Case(COMP_TRAIN, 4, {"vocab_size": 16, "seq_len": 8}),
    "sc": Case(
        SC_TRAIN,
        5,
        {
            "vocab_size": 16,
            "seq_len": 16,
            "num_tokens_to_copy": 3,
        },
        blank_seed=5,
    ),
}

GENERATORS = {
    "icr multi-query train": in_context_recall,
    "icr multi-query test": in_context_recall,
    "icr single-query test": in_context_recall,
    "nicr test": noisy_in_context_recall,
    "ficr train": fuzzy_in_context_recall,
    "ficr test": fuzzy_in_context_recall,
    "mem": memorization,
    "comp": compression,
    "sc": selective_copying,
}


def ints(text: str) -> NDArray[np.int64]:
    """Read a whitespace-separated fixture row.

    Args:
        text: Integers, space separated.

    Returns:
        ``(n,)`` int64.
    """
    return np.array([int(token) for token in text.split()], dtype=np.int64)


@pytest.mark.parametrize("name", sorted(CASES))
def test_stream_matches_upstream(name: str) -> None:
    """Every draw of every instance equals `mad-lab`'s at the same generator state."""
    case = CASES[name]
    rng = np.random.default_rng(case.seed)
    settings = dict(case.settings)
    if case.blank_seed is not None:
        settings["blank_rng"] = np.random.RandomState(case.blank_seed)
    draw = partial(GENERATORS[name], **settings)
    for index, (inputs, targets) in enumerate(case.fixture):
        got = draw(rng)
        assert isinstance(got, Instance)
        assert got.inputs.dtype == np.int64
        assert got.targets.dtype == np.int64
        np.testing.assert_array_equal(got.inputs, ints(inputs), err_msg=f"#{index}")
        np.testing.assert_array_equal(got.targets, ints(targets), err_msg=f"#{index}")


def test_ignore_index_is_honoured_everywhere() -> None:
    """Upstream hardcodes ``-100`` at a probe position; here the argument decides.

    A masked target that ignores the argument would silently supervise the probe under
    any other index, so the two recall tasks are the ones checked.
    """
    rng = np.random.default_rng(0)
    for draw in (
        partial(
            in_context_recall,
            vocab_size=16,
            seq_len=16,
            is_training=False,
            multi_query=False,
            ignore_index=-1,
        ),
        partial(
            fuzzy_in_context_recall,
            vocab_size=16,
            seq_len=32,
            is_training=False,
            multi_query=False,
            k_motif_size=2,
            v_motif_size=2,
            ignore_index=-1,
        ),
    ):
        targets = draw(rng).targets
        assert IGNORE_INDEX not in targets.tolist()
        assert -1 in targets.tolist()


@pytest.mark.parametrize(
    "noise_vocab_size,frac_noise", [(0, 0.5), (16, 1.0), (16, -0.1)]
)
def test_noise_settings_are_refused_when_unusable(
    noise_vocab_size: int, frac_noise: float
) -> None:
    """A noise fraction with no noise vocabulary, or outside ``[0, 1)``, is an error.

    Upstream asserts the same, and a fraction of 1 would leave no key-value pair for the
    probe to recall.
    """
    with pytest.raises(ValueError):
        in_context_recall(
            np.random.default_rng(0),
            vocab_size=32,
            seq_len=16,
            is_training=True,
            multi_query=True,
            noise_vocab_size=noise_vocab_size,
            frac_noise=frac_noise,
        )


def test_selective_copying_blanks_come_from_the_second_stream() -> None:
    """The blank positions follow ``blank_rng``, not the instance generator.

    Upstream draws them from the ambient legacy generator, so the two streams are
    genuinely independent and an implementation that folded them into one would place
    the copied tokens differently.
    """
    settings = {
        "vocab_size": 16,
        "seq_len": 16,
        "num_tokens_to_copy": 3,
    }
    first = selective_copying(
        np.random.default_rng(0), blank_rng=np.random.RandomState(0), **settings
    )
    same = selective_copying(
        np.random.default_rng(0), blank_rng=np.random.RandomState(0), **settings
    )
    other_blanks = selective_copying(
        np.random.default_rng(0), blank_rng=np.random.RandomState(1), **settings
    )
    np.testing.assert_array_equal(first.inputs, same.inputs)
    assert not np.array_equal(first.inputs, other_blanks.inputs)
    # The tokens to copy are the instance generator's, so they survive the change.
    np.testing.assert_array_equal(first.targets[-3:], other_blanks.targets[-3:])


def test_kv_map_keys_and_values_are_disjoint() -> None:
    """Memorization's premise: a key is never also a value.

    The map splits the vocabulary in half, keys below values, and is injective. A
    collision would let a model read a value off the context instead of the weights.
    """
    kv_map = build_kv_map(64, seed=12345)
    keys = {key[0] for key in kv_map}
    values = {value[0] for value in kv_map.values()}
    assert len(kv_map) == 32
    assert keys.isdisjoint(values)
    assert max(keys) < min(values)
    assert build_kv_map(64, seed=12345) == kv_map
    assert build_kv_map(64, seed=1) != kv_map
