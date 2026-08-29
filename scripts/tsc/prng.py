"""JAX's Threefry-2x32 stream, in numpy, so a split can be reproduced without JAX.

The protocol this harness reproduces draws its train/validation/test partition from
``jax.random.permutation`` on a key chain rooted at the seed. That partition is not a detail:
the published UEA bars are means over five seeds of *those* five partitions, so a harness that
drew its own would produce numbers that are not comparable to them, and the difference would
look like a modelling result.

So the generator is transcribed rather than approximated. Three pieces, each matching one JAX
function:

    :func:`prng_key`      ``jax.random.PRNGKey``
    :func:`split`         ``jax.random.split``
    :func:`permutation`   ``jax.random.permutation`` on an integer

``jax.random.permutation`` sorts ``arange(n)`` by 32-bit random keys, once per round, with
``ceil(3 * ln n / ln (2**32 - 1))`` rounds. That is one round up to 1,626 items and two up to
2.6 million, and the sort is stable, so the whole thing is exact integer arithmetic and
reproduces bit for bit. Every UEA dataset the protocol uses is inside the first round.

This is the ``jax_threefry_partitionable = False`` stream, which is the default JAX shipped
when the reference results were produced. The flag changes the key schedule inside ``split``
and ``random_bits`` and therefore changes every partition; ``tests/test_tsc_prng.py`` pins
this module against JAX with the flag set explicitly, so a JAX whose default has moved is a
test failure rather than a silent divergence.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "permutation",
    "prng_key",
    "random_bits",
    "split",
    "threefry_2x32",
]

_PARITY = np.uint32(0x1BD11BDA)
"""Threefry's key-schedule parity constant."""

_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
"""The two alternating rotation sets, four rounds each, five groups: Threefry-2x32-20."""

_UINT32_MAX = float(np.iinfo(np.uint32).max)


def _rotate_left(value: np.ndarray, count: int) -> np.ndarray:
    """Rotate a uint32 vector left.

    Args:
        value: uint32 array.
        count: Bits, in 1..31.

    Returns:
        The rotation, uint32.
    """
    return (value << np.uint32(count)) | (value >> np.uint32(32 - count))


def threefry_2x32(key: np.ndarray, count: np.ndarray) -> np.ndarray:
    """The block cipher, over a flat counter.

    The odd-size handling is JAX's and is not cosmetic: an odd counter is padded with one zero
    *at the end of the flattened array*, split into halves, and the last output word dropped.
    Padding at the front, or splitting before padding, gives a different stream for every
    odd-length draw.

    Args:
        key: uint32 array of two words.
        count: uint32 counter of any shape.

    Returns:
        uint32 array of ``count``'s shape.

    Raises:
        ValueError: When the key is not two words.
    """
    words = np.asarray(key, dtype=np.uint32).ravel()
    if words.size != 2:
        raise ValueError(f"key must hold two uint32 words, got {words.size}")
    flat = np.asarray(count, dtype=np.uint32).ravel()
    odd = flat.size % 2
    if odd:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint32)])
    half = flat.size // 2
    schedule = (words[0], words[1], words[0] ^ words[1] ^ _PARITY)
    # Integer wraparound is the cipher's arithmetic, not an error. numpy warns on scalar
    # overflow and this runs on arrays, but the suppression is explicit so a future scalar
    # path cannot turn the cipher into a test failure under `filterwarnings = error`.
    with np.errstate(over="ignore"):
        left = flat[:half] + schedule[0]
        right = flat[half:] + schedule[1]
        for group in range(5):
            for rotation in _ROTATIONS[group % 2]:
                left = left + right
                right = left ^ _rotate_left(right, rotation)
            left = left + schedule[(group + 1) % 3]
            right = right + schedule[(group + 2) % 3] + np.uint32(group + 1)
    out = np.concatenate([left, right])
    return (out[:-1] if odd else out).reshape(np.shape(count))


def prng_key(seed: int) -> np.ndarray:
    """``jax.random.PRNGKey``.

    Args:
        seed: The seed. Split across two words high-first, so a seed under 2**32 gives a
            leading zero and not a zero-padded tail.

    Returns:
        uint32 array of two words.

    Raises:
        ValueError: On a negative seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.array([seed >> 32, seed & 0xFFFFFFFF], dtype=np.uint32)


def split(key: np.ndarray, num: int = 2) -> np.ndarray:
    """``jax.random.split``.

    Args:
        key: The key.
        num: Keys to derive.

    Returns:
        ``(num, 2)`` uint32. Row order is JAX's, so ``a, b = split(key)`` binds the same two
        keys as in the reference.

    Raises:
        ValueError: On a non-positive count.
    """
    if num < 1:
        raise ValueError(f"num must be positive, got {num}")
    return threefry_2x32(key, np.arange(num * 2, dtype=np.uint32)).reshape(num, 2)


def random_bits(key: np.ndarray, size: int) -> np.ndarray:
    """``jax.random.bits`` at 32 bits, as a flat vector.

    Args:
        key: The key.
        size: Words to draw.

    Returns:
        uint32 array of ``size`` words.

    Raises:
        ValueError: On a negative count.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")
    return threefry_2x32(key, np.arange(size, dtype=np.uint32))


def permutation(key: np.ndarray, n: int) -> np.ndarray:
    """``jax.random.permutation`` on an integer.

    Args:
        key: The key.
        n: Items to permute.

    Returns:
        int64 array holding a permutation of ``arange(n)``.

    Raises:
        ValueError: On a negative count.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    rounds = math.ceil(3.0 * math.log(max(1, n)) / math.log(_UINT32_MAX))
    order = np.arange(n, dtype=np.int64)
    for _ in range(rounds):
        key, subkey = split(key, 2)
        # Stable, because JAX's sort is. An unstable sort would differ only where two draws
        # collide, which is rare enough to pass a small test and wrong at every real size.
        order = order[np.argsort(random_bits(subkey, n), kind="stable")]
    return order
