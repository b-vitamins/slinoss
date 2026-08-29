"""JAX's Threefry stream, reproduced in numpy.

The partition every published UEA bar was measured on comes out of this generator, so a
divergence here does not surface as an error: it surfaces as a different train/validation/test
split, which reads as a modelling result. There is exactly one ground truth for that and it is
JAX itself, so the pinning tests import it and skip where it is absent.

The flag is set explicitly in every one of those tests. ``jax_threefry_partitionable`` was False
when the reference results were produced and it changes the key schedule inside ``split`` and
``bits``, so a JAX whose default has moved must be a visible skip-or-fail and never a silent
divergence. The tests that need no JAX check the properties a permutation must have regardless.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from scripts.tsc.prng import permutation, prng_key, random_bits, split, threefry_2x32
from scripts.tsc.protocol import SEEDS

# Every UEA instance count the six datasets present, plus an odd size and a size either side of
# the 1,626-item boundary where jax.random.permutation goes from one sorting round to two.
SIZES = (1, 2, 3, 39, 259, 405, 1625, 1626, 1627, 3000)


@pytest.fixture
def jax_random() -> ModuleType:
    """``jax.random`` with the reference era's PRNG flag, or a skip."""
    jax = pytest.importorskip("jax", reason="the PRNG ground truth is JAX itself")
    jax.config.update("jax_threefry_partitionable", False)
    return jax.random


def test_key_chain_matches_jax(jax_random: ModuleType) -> None:
    """``PRNGKey`` and the reference's four-way then two-way split are bit-identical.

    This is the chain :func:`scripts.tsc.split.partition` walks, so a mismatch in any one row
    moves the partition even when the raw words look plausible.
    """
    for seed in SEEDS:
        root = prng_key(seed)
        assert np.array_equal(root, np.asarray(jax_random.PRNGKey(seed)))
        for num in (2, 4):
            want = np.asarray(jax_random.split(jax_random.PRNGKey(seed), num))
            assert np.array_equal(split(root, num), want), (seed, num)


def test_bits_match_jax_at_both_parities(jax_random: ModuleType) -> None:
    """32-bit draws match, including the odd-count padding.

    JAX pads an odd counter with one zero at the *end* of the flattened array and drops the last
    output word. Padding at the front instead gives a different stream for every odd draw, which
    is every permutation of an odd number of instances.
    """
    key = jax_random.PRNGKey(2345)
    for size in (1, 2, 3, 259, 3000):
        want = np.asarray(jax_random.bits(key, (size,), dtype=np.uint32))
        assert np.array_equal(random_bits(np.asarray(key), size), want)


def test_permutation_matches_jax(jax_random: ModuleType) -> None:
    """The permutation matches at every size the six datasets reach, and across the round bound.

    The round count is ``ceil(3 ln n / ln (2**32 - 1))``, so 1,626 and 1,627 sort a different
    number of times. Both sides of that boundary are checked because a wrong round count passes
    every small test.
    """
    for seed in (0, 2345, 6789):
        key = jax_random.PRNGKey(seed)
        for size in SIZES:
            want = np.asarray(jax_random.permutation(key, size))
            assert np.array_equal(permutation(np.asarray(key), size), want), (
                seed,
                size,
            )


def test_permutation_is_a_bijection_and_is_deterministic() -> None:
    """Every draw is a permutation of ``arange(n)`` and one key gives one answer.

    Holds without JAX, so a host with no JAX still catches a permutation that drops or repeats
    an index -- which would silently shrink or duplicate a split.
    """
    for seed in SEEDS:
        key = split(prng_key(seed), 2)[0]
        for size in SIZES:
            order = permutation(key, size)
            assert np.array_equal(np.sort(order), np.arange(size))
            assert np.array_equal(order, permutation(key, size))


def test_distinct_seeds_give_distinct_partitions() -> None:
    """The five protocol seeds do not collide at any dataset size.

    A generator that ignored part of its key would pass the bijection test and average one
    partition five times, reporting a spread of zero as a five-seed result.
    """
    for size in (39, 259, 405, 3000):
        orders = {
            permutation(split(prng_key(seed), 2)[0], size).tobytes() for seed in SEEDS
        }
        assert len(orders) == len(SEEDS), size


def test_a_short_key_is_refused() -> None:
    """A key that is not two words stops rather than being padded or truncated."""
    with pytest.raises(ValueError, match="two uint32 words"):
        threefry_2x32(np.zeros(3, dtype=np.uint32), np.arange(4, dtype=np.uint32))


def test_negative_counts_are_refused() -> None:
    """A negative seed, split count or draw size is a caller error, not an empty result."""
    with pytest.raises(ValueError, match="non-negative"):
        prng_key(-1)
    with pytest.raises(ValueError, match="positive"):
        split(prng_key(0), 0)
    with pytest.raises(ValueError, match="non-negative"):
        random_bits(prng_key(0), -1)
    with pytest.raises(ValueError, match="non-negative"):
        permutation(prng_key(0), -1)
