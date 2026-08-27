"""The six MAD task generators.

Reimplemented from `mad-lab`'s ``mad/data/instances.py``: the same draws in the same
order from the same generator, so an instance is bit-identical to the upstream one at
equal generator state and equal settings. ``tests/test_mad_instances.py`` pins that
against fixtures captured from the upstream file and is the parity gate; nothing here
changes without it failing.

Every generator takes the draw source first and its settings by keyword, and returns an
:class:`Instance`. Width is fixed by the settings and never by the draw, so a split
stacks without padding:

    task               width       targets
    icr                seq_len-1   shifted inputs (train) / probed values (test)
    nicr               seq_len-1   same, noise motifs masked
                                   both are seq_len without ``multi_query``, which
                                   spends a position on the copy prefix
    ficr               seq_len     same, left-padded to width
    memorization       seq_len     the value after each insert token
    compression        seq_len     the inputs
    selective copying  seq_len     the trailing run, everything before it masked

Deviations from upstream, each at a point no MAD config reaches:

- ``ignore_index`` is honoured at every masked position. Upstream writes the literal
  ``-100`` at the probe's own positions of the two recall tasks, which differs only
  when the caller asks for another index. Every MAD config asks for ``-100``.
- Selective copying draws its blank positions from ``blank_rng`` rather than from the
  global generator behind ``np.random.randint``. One instance of the same legacy
  stream, advanced across a whole split, reproduces upstream draw for draw; taking it
  as an argument makes the second draw source visible rather than ambient.
- ``memorization`` takes its key-value map instead of rebuilding it per instance, and
  neither it nor ``compression`` takes the noise settings. Upstream's noise branch is
  unreachable for both -- no MAD config gives either task a noise vocabulary -- and
  ``compression``'s branch builds masked targets that its return then discards. The
  draws upstream spends before that branch are kept: a draw whose value is never read
  still advances the stream, and the pool shares one generator across every instance.
- The non-selective copying task, and every ``rng=None`` default, are dropped. The MAD
  suite uses neither.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

IGNORE_INDEX = -100
"""Target value the loss and the accuracy both skip. Upstream's ``target_ignore_idx``."""

KV_MOTIF_SIZE = 2
"""Tokens one key-value pair spends in the single-token recall and memorization streams."""

Motifs = list[tuple[Any, ...]]
"""Ordered motifs over a vocabulary. Elements are numpy integers, as ``permutations``
over an array yields them; a tuple of them keys a dict interchangeably with a tuple of
``int``."""


class Instance(NamedTuple):
    """One task example.

    Attributes:
        inputs: ``(width,)`` int64 token ids in ``[0, vocab_size)``.
        targets: ``(width,)`` int64, either a token id or ``ignore_index``.
    """

    inputs: NDArray[np.int64]
    targets: NDArray[np.int64]


def _reject_noise(frac_noise: float, noise_vocab_size: int) -> None:
    """Refuse a noise fraction the vocabulary split cannot serve.

    Args:
        frac_noise: Fraction of motifs replaced by noise. In ``[0, 1)``: at 1 the
            sequence carries no key-value pair to recall.
        noise_vocab_size: Symbols reserved for noise.

    Raises:
        ValueError: On a fraction outside the interval, or a positive fraction with an
            empty noise vocabulary.
    """
    if not 0.0 <= frac_noise < 1.0:
        raise ValueError(f"frac_noise must lie in [0, 1), got {frac_noise}")
    if frac_noise > 0.0 and noise_vocab_size <= 0:
        raise ValueError(
            f"frac_noise {frac_noise} needs a noise vocabulary, "
            f"got noise_vocab_size {noise_vocab_size}"
        )


def _shift(inputs: list[Any], targets: list[Any], is_training: bool) -> Instance:
    """Close a recall stream: drop the last input, drop the first target.

    The stream is built one token longer than the width so the shift costs no padding.
    Training reads the shifted inputs, which supervises every position; test reads the
    built targets, which supervise the probed positions only.

    Args:
        inputs: Token ids, ``width + 1`` of them.
        targets: Ids and ``ignore_index`` values, same length.
        is_training: Which of the two target streams to return.

    Returns:
        The instance, both arrays int64 of width ``len(inputs) - 1``.
    """
    xs = np.asarray(inputs, dtype=np.int64)
    ys = np.asarray(targets, dtype=np.int64)
    return Instance(xs[:-1], xs[1:] if is_training else ys[1:])


def vocab_permutations(
    vocab: NDArray[np.int64], motif_size: int, rng: np.random.Generator
) -> Motifs:
    """Every ordered motif of ``motif_size`` distinct symbols, shuffled.

    Args:
        vocab: ``(V,)`` symbols to draw motifs from.
        motif_size: Symbols per motif.
        rng: Draw source. Shuffled in place, so the draw is charged to ``rng`` whether
            or not the caller reads the order.

    Returns:
        ``V!/(V-motif_size)!`` tuples.
    """
    values: Motifs = list(permutations(vocab, motif_size))
    rng.shuffle(values)  # pyright: ignore[reportArgumentType]  # MutableSequence
    return values


def _motif(rng: np.random.Generator, motifs: Motifs) -> tuple[Any, ...]:
    """One motif from a table.

    ``choice`` over equal-length tuples draws a row, which is the draw upstream spends.

    Args:
        rng: Draw source.
        motifs: The table, every entry the same length.

    Returns:
        The drawn motif. ``atleast_1d`` carries a single-token motif, where the row is
        one element wide.
    """
    return tuple(np.atleast_1d(rng.choice(motifs)))


def build_kv_map(
    vocab_size: int,
    *,
    k_motif_size: int = 1,
    v_motif_size: int = 1,
    seed: int = 12345,
) -> dict[tuple[Any, ...], tuple[Any, ...]]:
    """The fixed key-to-value mapping the memorization task has to store in weights.

    Keys take the lower half of the vocabulary and values the upper, so the map is a
    bijection onto as many pairs as the shorter side admits. Its own generator, seeded
    here, keeps it stable across every instance of a split and across splits.

    Args:
        vocab_size: Symbols the two halves are carved from. Upstream passes the
            vocabulary less the insert token.
        k_motif_size: Tokens per key.
        v_motif_size: Tokens per value.
        seed: Seed for the mapping's own generator.

    Returns:
        ``min(len(keys), len(values))`` pairs, key motif to value motif.
    """
    rng = np.random.default_rng(seed)
    keys = vocab_permutations(np.arange(vocab_size // 2), k_motif_size, rng)
    values = vocab_permutations(
        np.arange(vocab_size // 2, vocab_size), v_motif_size, rng
    )
    return dict(zip(keys, values))


def in_context_recall(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    multi_query: bool,
    noise_vocab_size: int = 0,
    frac_noise: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> Instance:
    """One in-context recall instance: bigrams, then a key already seen.

    A stream of key-value bigrams over a vocabulary split in half, each key bound to
    one value for the whole instance. The stream closes on a key drawn from those
    already presented, so the value is recoverable from the context and from nothing
    else. ``multi_query`` supervises every repeat of a key rather than the last one
    alone, and spends the token the copy prefix would take on another pair.

    At ``frac_noise > 0`` a motif is replaced by two draws from a disjoint noise
    vocabulary, which is the noisy variant of the task; one motif index is held out of
    that so a pair always exists to probe.

    Args:
        rng: Draw source.
        vocab_size: Vocabulary, including the copy prefix at ``vocab_size - 1`` and
            the noise symbols.
        seq_len: Stream length. Even.
        is_training: Target stream to return, see :func:`_shift`.
        multi_query: Supervise every repeated key, and drop the copy prefix.
        noise_vocab_size: Symbols reserved for noise, taken off the top of the
            key-value range.
        frac_noise: Fraction of motifs replaced by noise.
        ignore_index: Target value at an unsupervised position.

    Returns:
        The instance, width ``seq_len - 1``.

    Raises:
        ValueError: On an odd ``seq_len``, or a noise setting :func:`_reject_noise`
            refuses.
    """
    _reject_noise(frac_noise, noise_vocab_size)
    if seq_len % KV_MOTIF_SIZE != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")

    copy_prefix = vocab_size - 1
    non_special = (vocab_size if multi_query else vocab_size - 1) - noise_vocab_size
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)
    noise_vocab = np.arange(non_special, non_special + noise_vocab_size)

    num_kv_pairs = seq_len // KV_MOTIF_SIZE
    bound: dict[Any, Any] = {}
    presented: dict[Any, Any] = {}
    inputs: list[Any] = []
    targets: list[Any] = []
    # Held out of the noise draw so at least one pair is present to probe.
    not_noise_idx = rng.choice(num_kv_pairs)
    # One pair short: the probe below is the last one.
    for i in range(num_kv_pairs - 1):
        noisy = (
            bool(rng.random() < frac_noise)
            if i != not_noise_idx and frac_noise > 0
            else False
        )
        if noisy:
            inputs += list(rng.choice(noise_vocab, size=KV_MOTIF_SIZE, replace=True))
            targets += [ignore_index] * KV_MOTIF_SIZE
            continue

        key = rng.choice(key_vocab)
        if key not in bound:
            bound[key] = rng.choice(value_vocab)
        value = bound[key]
        inputs += [key, value]
        targets.append(ignore_index)
        targets.append(value if multi_query and key in presented else ignore_index)
        presented[key] = value

    # Raises when noise took every motif, which needs a frac_noise far above the
    # 0.2 the noisy task is defined at. Upstream raises there too.
    k_probe = rng.choice(list(presented.keys()))
    v_probe = presented[k_probe]
    if not multi_query:
        inputs.append(copy_prefix)
        targets.append(ignore_index)
    inputs += [k_probe, v_probe]
    targets += [ignore_index, v_probe]
    return _shift(inputs, targets, is_training)


def noisy_in_context_recall(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    multi_query: bool,
    noise_vocab_size: int,
    frac_noise: float,
    ignore_index: int = IGNORE_INDEX,
) -> Instance:
    """In-context recall with a fraction of the motifs replaced by noise.

    The same task, with the noise arguments required rather than defaulted. Kept as its
    own name because the MAD suite counts it as its own task and reports it separately.

    Args:
        rng: Draw source.
        vocab_size: Vocabulary, including the noise symbols.
        seq_len: Stream length. Even.
        is_training: Target stream to return.
        multi_query: Supervise every repeated key.
        noise_vocab_size: Symbols reserved for noise. Positive.
        frac_noise: Fraction of motifs replaced by noise. Positive.
        ignore_index: Target value at an unsupervised position.

    Returns:
        The instance, width ``seq_len - 1``.
    """
    return in_context_recall(
        rng,
        vocab_size=vocab_size,
        seq_len=seq_len,
        is_training=is_training,
        multi_query=multi_query,
        noise_vocab_size=noise_vocab_size,
        frac_noise=frac_noise,
        ignore_index=ignore_index,
    )


def fuzzy_in_context_recall(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    multi_query: bool,
    k_motif_size: int,
    v_motif_size: int,
    noise_vocab_size: int = 0,
    frac_noise: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
) -> Instance:
    """In-context recall where a key and a value are motifs of variable length.

    Keys and values are tuples of up to ``k_motif_size`` and ``v_motif_size`` distinct
    symbols, so the boundary between one and the next is not at a fixed stride and has
    to be inferred. Training draws a motif length per pair; test pins every key to the
    maximum, which is the length the probe is always asked at. One pair is planted at a
    drawn offset and repeated at the end as the probe.

    The stream is left-padded to the width, so a shorter draw costs supervised
    positions at the front rather than a ragged batch.

    Args:
        rng: Draw source.
        vocab_size: Vocabulary, including the pad token at ``vocab_size - 1``
            (``vocab_size - 2`` without ``multi_query``, where the copy prefix takes
            the top symbol) and the noise symbols.
        seq_len: Width of the returned instance.
        is_training: Target stream to return, and whether key lengths vary.
        multi_query: Supervise every repeated key, and drop the copy prefix.
        k_motif_size: Longest key, in tokens.
        v_motif_size: Longest value, in tokens.
        noise_vocab_size: Symbols reserved for noise.
        frac_noise: Fraction of motifs replaced by noise.
        ignore_index: Target value at an unsupervised position.

    Returns:
        The instance, width ``seq_len``.

    Raises:
        ValueError: On a noise setting :func:`_reject_noise` refuses.
    """
    _reject_noise(frac_noise, noise_vocab_size)

    copy_prefix = vocab_size - 1
    pad_token = vocab_size - 1 if multi_query else vocab_size - 2
    non_special = (vocab_size - 1 if multi_query else vocab_size - 2) - noise_vocab_size
    key_vocab = np.arange(non_special // 2)
    value_vocab = np.arange(non_special // 2, non_special)
    noise_vocab = np.arange(non_special, non_special + noise_vocab_size)

    key_sizes = (
        range(1, k_motif_size + 1)
        if is_training
        else range(k_motif_size, k_motif_size + 1)
    )
    keys = {size: vocab_permutations(key_vocab, size, rng) for size in key_sizes}
    values = {
        size: vocab_permutations(value_vocab, size, rng)
        for size in range(1, v_motif_size + 1)
    }

    k_probe_size = int(rng.choice(list(keys.keys()))) if is_training else k_motif_size
    v_probe_size = int(rng.choice(list(values.keys())))
    k_probe = _motif(rng, keys[k_probe_size])
    v_probe = _motif(rng, values[v_probe_size])
    probe_size = k_probe_size + v_probe_size
    # Placed clear of the tail, so the probe's own repeat is not the first sight of it.
    probe_idx = rng.choice(seq_len - 2 * probe_size)
    probe_added = False

    bound: dict[int, dict[tuple[Any, ...], tuple[Any, ...]]] = {
        size: {} for size in range(1, k_motif_size + 1)
    }
    presented: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    inputs: list[Any] = []
    targets: list[Any] = []
    # Leaves room for the probe's repeat and for one more pair of maximum length, so
    # the stream never overruns the width and the pad below is never negative.
    while len(inputs) < seq_len - probe_size - (k_motif_size + v_motif_size):
        k_size = int(rng.choice(list(keys.keys()))) if is_training else k_motif_size
        v_size = int(rng.choice(list(values.keys())))

        if len(inputs) >= probe_idx and not probe_added:
            inputs += [*k_probe, *v_probe]
            targets += [ignore_index] * probe_size
            bound[k_probe_size][k_probe] = v_probe
            presented[k_probe] = v_probe
            probe_added = True
            continue

        noisy = bool(rng.random() < frac_noise) if frac_noise > 0 else False
        if noisy:
            size = k_size + v_size
            inputs += list(rng.choice(noise_vocab, size=size, replace=True))
            targets += [ignore_index] * size
            continue

        key = _motif(rng, keys[k_size])
        inputs += list(key)
        if key == k_probe:
            value = v_probe
            probe_added = True
        else:
            if key not in bound[k_size]:
                bound[k_size][key] = _motif(rng, values[v_size])
            value = bound[k_size][key]
        inputs += list(value)

        targets += [ignore_index] * k_size
        if multi_query and key in presented:
            targets += list(value)
        else:
            targets += [ignore_index] * len(value)
        presented[key] = value

    if not multi_query:
        inputs.append(copy_prefix)
        targets.append(ignore_index)
    inputs += [*k_probe, *v_probe]
    targets += [ignore_index] * k_probe_size
    targets += list(v_probe)

    n_pad = seq_len + 1 - len(inputs)
    if n_pad > 0:
        inputs = [pad_token] * n_pad + inputs
        targets = [ignore_index] * n_pad + targets
    return _shift(inputs, targets, is_training)


def memorization(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    kv_map: dict[tuple[Any, ...], tuple[Any, ...]],
    ignore_index: int = IGNORE_INDEX,
) -> Instance:
    """One memorization instance: keys, each followed by the insert token.

    The mapping is fixed across the whole task, so nothing in the context says what a
    key maps to and the value has to come from the weights. Both splits are drawn the
    same way and from the same map; ``is_training`` is accepted and ignored so every
    generator takes one call.

    Args:
        rng: Draw source.
        vocab_size: Vocabulary, including the insert token at ``vocab_size - 1``.
        seq_len: Width of the returned instance. Even.
        is_training: Ignored. The two splits differ only in their draws.
        kv_map: The fixed mapping, from :func:`build_kv_map`. Single-token keys and
            values, so a pair spends two positions.
        ignore_index: Target value at an unsupervised position.

    Returns:
        The instance, width ``seq_len``. Targets carry a value under every insert
        token and ``ignore_index`` everywhere else.

    Raises:
        ValueError: On an odd ``seq_len``.
    """
    del is_training
    if seq_len % KV_MOTIF_SIZE != 0:
        raise ValueError(f"seq_len must be even, got {seq_len}")

    insert_token = vocab_size - 1
    keys = list(kv_map.keys())
    pairs = seq_len // KV_MOTIF_SIZE
    # The draw upstream spends choosing the pair exempt from noise. No memorization
    # config sets a noise fraction, so the value is never read, but the draw advances a
    # stream one generator shares across a whole pool: dropping it shifts every
    # instance after the first.
    rng.choice(pairs)
    inputs: list[Any] = []
    targets: list[Any] = []
    for _ in range(pairs):
        key = _motif(rng, keys)
        inputs += [*key, insert_token]
        targets += [ignore_index, *kv_map[key]]
    return Instance(
        np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    )


def compression(
    rng: np.random.Generator,
    *,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
) -> Instance:
    """One compression instance: a random stream closed by the compression token.

    The target is the input, so the task is to reconstruct the whole stream from
    whatever the state holds at the compression token. Both splits are drawn the same
    way; ``is_training`` is accepted and ignored.

    Args:
        rng: Draw source.
        vocab_size: Vocabulary, including the compression token at ``vocab_size - 1``.
        seq_len: Width of the returned instance.
        is_training: Ignored. The two splits differ only in their draws.

    Returns:
        The instance, width ``seq_len``. Every position is supervised.
    """
    del is_training
    tokens = rng.choice(np.arange(vocab_size - 1), size=(seq_len - 1,), replace=True)
    inputs = np.concatenate([tokens.reshape(-1), np.array([vocab_size - 1])])
    inputs = inputs.astype(np.int64)
    return Instance(inputs, inputs)


def selective_copying(
    rng: np.random.Generator,
    *,
    blank_rng: np.random.RandomState,
    vocab_size: int,
    seq_len: int,
    is_training: bool,
    num_tokens_to_copy: int,
    ignore_index: int = IGNORE_INDEX,
) -> Instance:
    """One selective copying instance: a sparse run, then room to reproduce it.

    The tokens to copy are scattered through a field of blanks at drawn positions, so
    their offsets carry no information and the state has to select on content. The copy
    token then opens exactly as many blank positions as there are tokens to reproduce,
    and those are the only supervised positions. Both splits are drawn the same way;
    ``is_training`` is accepted and ignored.

    Args:
        rng: Draw source for the tokens.
        blank_rng: Draw source for the blank positions, a legacy-stream generator held
            across a whole split. See the module docstring.
        vocab_size: Vocabulary, including the copy token at ``vocab_size - 1`` and the
            blank at ``vocab_size - 2``.
        seq_len: Width of the returned instance.
        is_training: Ignored. The two splits differ only in their draws.
        num_tokens_to_copy: Tokens in the run.
        ignore_index: Target value at an unsupervised position.

    Returns:
        The instance, width ``seq_len``.

    Raises:
        ValueError: When the run and its copy leave no room for a blank.
    """
    del is_training
    if seq_len <= 2 * num_tokens_to_copy + 1:
        raise ValueError(
            f"seq_len must exceed 2 * num_tokens_to_copy + 1 = "
            f"{2 * num_tokens_to_copy + 1}, got {seq_len}"
        )

    copy_token = vocab_size - 1
    blank_token = vocab_size - 2
    num_blanks = seq_len - 2 * num_tokens_to_copy - 1
    to_copy = rng.choice(
        np.arange(vocab_size - 2), size=(num_tokens_to_copy,), replace=True
    ).reshape(-1)
    # Indices into the run, so duplicates stack blanks at one offset.
    where = blank_rng.randint(0, len(to_copy), num_blanks)
    scattered = np.insert(to_copy, where, [blank_token] * num_blanks).tolist()

    inputs = [*scattered, copy_token, *[blank_token] * num_tokens_to_copy]
    targets = [ignore_index] * (num_tokens_to_copy + num_blanks + 1) + list(to_copy)
    return Instance(
        np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)
    )
