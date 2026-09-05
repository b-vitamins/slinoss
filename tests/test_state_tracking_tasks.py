"""The five automaton generators and the group word problem, against upstream.

Every published number on this axis was measured on
`structured-linear-cdes`'s ``data_dir/fl_tasks/`` generators, which
`expressive-sparse-state-space-model` carries byte-identical. A transcription that drifts
by one draw produces a different length distribution, a different token distribution, or a
different label, and the arm still trains and still reports an accuracy -- against a
different task than the one the bars belong to. So the generators are pinned twice.

:data:`UPSTREAM` holds samples taken from upstream's own files, run under its own
``generate_sample`` and ``preprocess_data``, at four seeds spanning the ``2**32`` seed
space and both split length ranges. Those are compared token for token. The seeds include
0, which is the protocol's first, and ``2**32 - 1``, which is the largest a seed stream can
draw.

Then each label is re-derived from the tokens by a route the generator does not take -- the
two arithmetic tasks by evaluating the expression the tokens spell -- so a generator that
reproduces upstream's token stream and mislabels it is caught as well.

The group half needs no fixture: upstream's ``GroupCompositionDataset`` draws from
``random.Random`` and the stream is reproduced here directly.
"""

from __future__ import annotations

import random
from itertools import pairwise
from typing import NamedTuple

import pytest

from scripts.state_tracking.groups import parse
from scripts.state_tracking.tasks import (
    AUTOMATA,
    MODULUS,
    PAD_TOKEN,
    PDSSM_GROUP_TASKS,
    PDSSM_REGULAR_TASKS,
    Sample,
    Task,
    mod_arith_no_brack,
    mod_arith_w_brack,
    resolve,
    resolve_profile,
    word_problem,
)


class Case(NamedTuple):
    """One sample as upstream produced it.

    Attributes:
        seed: The item seed.
        min_length: Split floor.
        max_length: Split ceiling, inclusive.
        target: The label, at the last position.
        ids: Every input token.
    """

    seed: int
    min_length: int
    max_length: int
    target: int
    ids: tuple[int, ...]


UPSTREAM: dict[str, tuple[Case, ...]] = {
    "parity": (
        Case(
            0,
            3,
            40,
            1,
            (
                2,
                2,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                2,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                2,
                1,
                2,
            ),
        ),
        Case(1, 3, 40, 1, (2, 1, 1, 2, 2, 2, 2, 2, 1, 1)),
        Case(
            7,
            40,
            64,
            1,
            (
                1,
                2,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                2,
                2,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
            ),
        ),
        Case(
            4294967295,
            3,
            40,
            1,
            (1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1),
        ),
    ),
    "even_pairs": (
        Case(
            0,
            3,
            40,
            1,
            (
                2,
                2,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                1,
                2,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                2,
                1,
                2,
            ),
        ),
        Case(1, 3, 40, 2, (2, 1, 1, 2, 2, 2, 2, 2, 1, 1)),
        Case(
            7,
            40,
            64,
            2,
            (
                1,
                2,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                1,
                1,
                2,
                2,
                1,
                1,
                2,
                2,
                1,
                2,
                2,
                1,
                1,
                1,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                2,
            ),
        ),
        Case(
            4294967295,
            3,
            40,
            1,
            (1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1),
        ),
    ),
    "cycle_nav": (
        Case(
            0,
            3,
            40,
            4,
            (
                1,
                3,
                1,
                2,
                1,
                2,
                2,
                2,
                1,
                3,
                3,
                1,
                1,
                2,
                3,
                1,
                1,
                3,
                1,
                3,
                3,
                3,
                3,
                3,
                3,
                2,
                3,
                1,
                1,
                2,
                1,
                3,
                1,
                2,
                1,
                2,
                3,
            ),
        ),
        Case(1, 3, 40, 5, (3, 1, 3, 2, 2, 3, 3, 3, 1, 3)),
        Case(
            7,
            40,
            64,
            7,
            (
                2,
                2,
                3,
                2,
                1,
                3,
                3,
                2,
                3,
                2,
                2,
                3,
                1,
                3,
                2,
                1,
                2,
                1,
                1,
                1,
                2,
                3,
                1,
                2,
                3,
                3,
                2,
                1,
                3,
                1,
                2,
                1,
                2,
                2,
                1,
                1,
                3,
                3,
                3,
                2,
                3,
                1,
                3,
                3,
                2,
                2,
                3,
                3,
                2,
                1,
                3,
                3,
                1,
                1,
                1,
            ),
        ),
        Case(
            4294967295,
            3,
            40,
            4,
            (1, 3, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 1, 3, 1, 3, 2, 3, 1, 3, 3),
        ),
    ),
    "mod_arith_no_brack": (
        Case(
            0,
            3,
            40,
            8,
            (
                9,
                3,
                8,
                3,
                5,
                3,
                8,
                3,
                9,
                3,
                7,
                3,
                8,
                2,
                7,
                3,
                8,
                1,
                6,
                1,
                6,
                2,
                6,
                1,
                9,
                3,
                8,
                1,
                6,
                2,
                6,
                1,
                8,
                2,
                9,
                3,
                8,
                4,
            ),
        ),
        Case(1, 3, 40, 7, (9, 3, 9, 3, 8, 3, 8, 1, 8, 4)),
        Case(
            7,
            40,
            64,
            5,
            (
                7,
                3,
                6,
                1,
                6,
                2,
                8,
                1,
                7,
                2,
                7,
                2,
                9,
                1,
                8,
                1,
                6,
                3,
                8,
                3,
                6,
                3,
                8,
                2,
                7,
                3,
                7,
                1,
                5,
                3,
                6,
                3,
                7,
                2,
                8,
                2,
                8,
                3,
                6,
                3,
                5,
                2,
                5,
                1,
                6,
                3,
                8,
                3,
                8,
                1,
                8,
                1,
                7,
                1,
                5,
                4,
            ),
        ),
        Case(
            4294967295,
            3,
            40,
            8,
            (8, 2, 6, 3, 9, 1, 5, 3, 5, 1, 7, 3, 9, 2, 9, 3, 7, 1, 7, 3, 7, 4),
        ),
    ),
    "mod_arith_w_brack": (
        Case(
            0,
            3,
            40,
            3,
            (
                9,
                9,
                9,
                4,
                7,
                7,
                5,
                10,
                8,
                9,
                7,
                3,
                8,
                7,
                4,
                10,
                10,
                8,
                9,
                9,
                7,
                5,
                10,
                8,
                9,
                9,
                2,
                8,
                4,
                10,
                8,
                7,
                4,
                10,
                10,
                10,
                11,
            ),
        ),
        Case(1, 3, 40, 1, (9, 9, 4, 7, 4, 10, 8, 2, 10, 11)),
        Case(
            7,
            40,
            64,
            5,
            (
                9,
                9,
                9,
                4,
                8,
                3,
                10,
                8,
                9,
                9,
                9,
                7,
                4,
                7,
                9,
                7,
                4,
                8,
                7,
                3,
                10,
                10,
                7,
                7,
                2,
                10,
                6,
                9,
                9,
                2,
                8,
                9,
                1,
                10,
                10,
                7,
                9,
                4,
                8,
                4,
                10,
                10,
                10,
                10,
                7,
                9,
                9,
                7,
                1,
                10,
                7,
                2,
                10,
                10,
                11,
            ),
        ),
        Case(
            4294967295,
            3,
            40,
            1,
            (9, 7, 2, 8, 9, 9, 7, 1, 10, 7, 9, 3, 7, 9, 5, 7, 3, 10, 10, 10, 10, 11),
        ),
    ),
}
"""Samples taken from upstream's own generators, under torch 2.10.0."""

VOCAB = {
    "parity": 3,
    "even_pairs": 3,
    "cycle_nav": 9,
    "mod_arith_no_brack": 10,
    "mod_arith_w_brack": 12,
}
"""Upstream's ``vocab_size`` per task: tokens in, and classes on the head."""


def test_every_task_is_covered_by_a_fixture() -> None:
    """The fixture table names every registered automaton task.

    Without this a new generator could ship untested, since the parity tests iterate the
    fixture table rather than the registry.
    """
    assert set(UPSTREAM) == set(AUTOMATA)
    assert set(VOCAB) == set(AUTOMATA)


def test_upstream_tokens() -> None:
    """Every fixture sample reproduces upstream token for token.

    This is the parity claim. It covers the drawn length, the draw order and the token
    encoding at once: any of the three moving changes the tuple.
    """
    for name, cases in UPSTREAM.items():
        task = AUTOMATA[name]
        for case in cases:
            sample = task.sample(case.seed, case.min_length, case.max_length)
            assert sample.ids == case.ids, f"{name} at seed {case.seed}"


def test_upstream_supervision_is_the_last_position_alone() -> None:
    """One supervised position per automaton sample, at the end, carrying the label.

    Upstream's ``preprocess_data`` builds the targets with ``torch.zeros_like`` and writes
    one entry, so a target at any other position is zero and is never read. The mask is
    what the loss selects with; a mask that is True anywhere else would train the model to
    emit the label early.
    """
    for name, cases in UPSTREAM.items():
        task = AUTOMATA[name]
        assert task.supervision == "last"
        for case in cases:
            sample = task.sample(case.seed, case.min_length, case.max_length)
            width = len(sample.ids)
            assert sample.supervised == (False,) * (width - 1) + (True,)
            assert sample.targets == (0,) * (width - 1) + (case.target,)


def test_no_automaton_task_emits_the_pad_token() -> None:
    """Token 0 is unused by every automaton task, which is what lets it be the pad.

    A generator that emitted 0 would make a padded position indistinguishable from a real
    one, and the recurrence would read the pad as a symbol.
    """
    for name, task in AUTOMATA.items():
        assert task.vocab_size == VOCAB[name]
        for seed in range(32):
            sample = task.sample(seed, 2, 24)
            assert PAD_TOKEN not in sample.ids
            assert all(0 < token < task.vocab_size for token in sample.ids)
            assert 0 <= sample.targets[-1] < task.vocab_size


def test_length_range_is_honoured() -> None:
    """A drawn length sits in the requested range, with the two documented exceptions.

    ``mod_arith_no_brack`` rounds the drawn length up to even, so it can produce
    ``max_length + 1``; ``mod_arith_w_brack`` spends one token on the ``=`` and produces
    exactly the drawn length. Both are upstream's behaviour, and both are why the batcher
    cannot assume the split's ceiling is the widest item.
    """
    for seed in range(64):
        for name in ("parity", "even_pairs", "cycle_nav"):
            width = len(AUTOMATA[name].sample(seed, 5, 11).ids)
            assert 5 <= width <= 11
        flat = len(mod_arith_no_brack(seed, 5, 11).ids)
        assert flat % 2 == 0
        assert 5 <= flat <= 12
        bracketed = len(mod_arith_w_brack(seed, 5, 11).ids)
        assert 5 <= bracketed <= 11


def test_parity_label_counts_the_ones() -> None:
    """The label is 1 on an even count of the second symbol and 2 on an odd count."""
    for seed in range(64):
        sample = AUTOMATA["parity"].sample(seed, 3, 30)
        ones = sum(1 for token in sample.ids if token == 2)
        assert sample.targets[-1] == (1 if ones % 2 == 0 else 2)


def test_even_pairs_label_compares_the_ends() -> None:
    """The label is 1 when the first and last tokens agree, 2 when they do not."""
    for seed in range(64):
        sample = AUTOMATA["even_pairs"].sample(seed, 3, 30)
        ends_agree = sample.ids[0] == sample.ids[-1]
        assert sample.targets[-1] == (1 if ends_agree else 2)


def test_cycle_nav_label_is_the_position_on_the_five_cycle() -> None:
    """The label is ``4 + (forward - backward) mod 5``.

    Token 1 stays, 2 steps forward, 3 steps back, and the label codes sit above every
    input token so the head's classes do not collide with the vocabulary.
    """
    for seed in range(64):
        sample = AUTOMATA["cycle_nav"].sample(seed, 3, 30)
        forward = sum(1 for token in sample.ids if token == 2)
        backward = sum(1 for token in sample.ids if token == 3)
        assert sample.targets[-1] == 4 + (forward - backward) % MODULUS
        assert sample.targets[-1] >= 4


def _flat_text(ids: tuple[int, ...]) -> str:
    """``mod_arith_no_brack`` tokens read back as the expression they spell."""
    operators = {1: "+", 2: "-", 3: "*"}
    parts = []
    for token in ids[:-1]:
        parts.append(operators[token] if token < 5 else str(token - 5))
    return "".join(parts)


def _bracketed_text(ids: tuple[int, ...]) -> str:
    """``mod_arith_w_brack`` tokens read back as the expression they spell."""
    table = {
        MODULUS + 1: "+",
        MODULUS + 2: "-",
        MODULUS + 3: "*",
        MODULUS + 4: "(",
        MODULUS + 5: ")",
    }
    return "".join(
        table[token] if token > MODULUS else str(token - 1) for token in ids[:-1]
    )


def test_flat_arithmetic_label_obeys_bidmas() -> None:
    """The label is the expression's value mod 5, products binding before sums.

    The generator evaluates in two passes over its own token list. Here the tokens are
    read back into infix and evaluated by Python, which is an independent route to the
    same number and the only thing that catches a precedence bug: a left-to-right
    evaluation agrees with BIDMAS on every expression with no ``*`` before a ``+``.
    """
    seen_precedence = False
    for seed in range(256):
        sample = mod_arith_no_brack(seed, 6, 20)
        assert sample.ids[-1] == 4, "the last token is the equals sign"
        text = _flat_text(sample.ids)
        # Python's own precedence, on digits 0-4 and the three operators.
        assert sample.targets[-1] == eval(text) % MODULUS + 5
        if "*" in text and ("+" in text or "-" in text):
            seen_precedence = True
    assert seen_precedence, "no expression mixed a product with a sum"


def test_bracketed_arithmetic_label_evaluates_the_tree() -> None:
    """The label is the bracketed expression's value mod 5, brackets balanced.

    The generator computes the value as it builds the string, so evaluating the string
    afterwards is the independent check. Every intermediate is reduced mod 5 as it is
    formed, so the check reduces at every node too rather than only at the root.
    """
    for seed in range(256):
        sample = mod_arith_w_brack(seed, 6, 20)
        assert sample.ids[-1] == MODULUS + 6, "the last token is the equals sign"
        text = _bracketed_text(sample.ids)
        assert text.count("(") == text.count(")")
        # Every node of the tree is already reduced mod 5, so the root's residue is the
        # residue of the unreduced value: reduction commutes with +, - and *.
        assert sample.targets[-1] == eval(text) % MODULUS + 1


def test_generators_refuse_an_empty_length_range() -> None:
    """A split whose floor is over its ceiling is a configuration error, not an empty draw.

    ``torch.randint(low, high)`` with ``high <= low`` raises, but only after the generator
    has been seeded, and the message names neither the task nor the split.
    """
    for name in AUTOMATA:
        with pytest.raises(ValueError, match="is over max_length"):
            AUTOMATA[name].sample(0, 12, 4)


def test_bracketed_arithmetic_refuses_a_floor_under_two() -> None:
    """At length 1 the bracketed expression is the ``=`` alone and has no value.

    Upstream reaches that state and fails inside the recursion. Here it is refused at the
    boundary, which is why :class:`scripts.state_tracking.tasks.Task` carries a
    ``min_length`` at all.
    """
    assert AUTOMATA["mod_arith_w_brack"].min_length == 2
    with pytest.raises(ValueError, match="min_length must be at least 2"):
        mod_arith_w_brack(0, 1, 40)
    assert len(mod_arith_w_brack(0, 2, 2).ids) == 2


def test_group_task_reproduces_the_stdlib_stream() -> None:
    """The word problem's draws are ``random.Random(seed)``, in upstream's order.

    One length draw then one token draw per position. A reordering would still produce a
    uniform word, so only the stream itself catches it.
    """
    group = parse("A5")
    sample_fn = word_problem(group)
    for seed in (0, 1, 7, 2**32 - 1):
        sample = sample_fn(seed, 3, 40)
        rng = random.Random(seed)
        length = rng.randint(3, 40)
        expected = tuple(rng.randint(0, group.order - 1) for _ in range(length))
        assert sample.ids == expected
        assert len(sample.ids) == length


def test_group_task_supervises_every_position_with_the_running_product() -> None:
    """Each target is the product of the prefix ending there, from the identity.

    The whole state trajectory is the label, which is what makes the group half a state
    test rather than a classification: a model that recovers the final product only at the
    end scores ``1 / order`` on the positions before it.
    """
    group = parse("A5")
    sample_fn = word_problem(group)
    sample = sample_fn(3, 8, 8)
    assert sample.supervised == (True,) * 8
    state = 0
    for position, token in enumerate(sample.ids):
        state = group.compose(state, token)
        assert sample.targets[position] == state
    assert sample.targets == group.prefix(sample.ids)


def test_group_task_labels_reuse_the_token_vocabulary() -> None:
    """Tokens and labels are both element indices, so ``vocab_size`` is the order.

    Element 0 is the identity and the pad token at once. That collision is why the batcher
    carries an explicit mask: an ``ignore_index=0`` loss would drop every position whose
    running product is the identity.
    """
    task = resolve("A5")
    assert task.supervision == "all"
    assert task.group is not None
    assert task.vocab_size == task.group.order == 60
    sample = task.sample(0, 4, 4)
    assert all(0 <= token < task.vocab_size for token in sample.ids)
    assert all(0 <= target < task.vocab_size for target in sample.targets)


def test_resolve_falls_through_to_a_group_spec() -> None:
    """A name is an automaton key or a group spec, and nothing else is accepted."""
    assert resolve("parity") is AUTOMATA["parity"]
    assert resolve("Z60_x_Z2").vocab_size == 120
    with pytest.raises(ValueError, match="no task 'Q8'"):
        resolve("Q8")
    with pytest.raises(ValueError, match="no task 'A5 '"):
        resolve("A5 ")


def test_task_validates_its_own_fields() -> None:
    """A malformed task is refused at construction, not at the first batch."""
    ok: Sample = Sample((1,), (1,), (True,))
    with pytest.raises(ValueError, match="input_vocab_size must be positive"):
        Task("bad", 0, lambda s, lo, hi: ok, "last")
    with pytest.raises(ValueError, match="output_vocab_size must be positive"):
        Task("bad", 3, lambda s, lo, hi: ok, "last", output_vocab_size=0)
    with pytest.raises(ValueError, match="supervision must be last or all"):
        Task("bad", 3, lambda s, lo, hi: ok, "final")
    with pytest.raises(ValueError, match="min_length must be positive"):
        Task("bad", 3, lambda s, lo, hi: ok, "last", min_length=0)


def test_pdssm_a5_two_uses_two_inputs_and_sixty_output_states() -> None:
    """The released IBM A5 actions are inputs; all A5 elements remain output labels.

    This is the structural mismatch a single ``vocab_size`` concealed.  The action labels
    are computed independently from IBM's two released matrices in the source audit.
    """
    task = resolve("pdssm:A5:2")
    assert task.group is not None
    assert task.input_vocab_size == 2
    assert task.output_vocab_size == task.group.order == 60
    assert task.contract.fidelity == "cross-release-reconstruction"
    assert task.contract.generator_labels == ("12340", "10324")
    assert task.supervision == "last"
    with pytest.raises(ValueError, match="vocab_size is ambiguous"):
        _ = task.vocab_size

    sample = task.sample(7, 8, 8)
    assert all(0 <= token < 2 for token in sample.ids)
    assert sample.supervised == (False,) * 7 + (True,)
    elements = tuple(task.generator_elements[token] for token in sample.ids)
    assert sample.targets[-1] == task.group.prefix(elements)[-1]
    assert 0 <= sample.targets[-1] < 60


def test_pdssm_group_reconstructions_are_nested_distinct_and_labelled() -> None:
    """Every reconstructed table row states exactly which permutations it uses."""
    tasks = [resolve(name) for name in PDSSM_GROUP_TASKS]
    for task in tasks:
        assert task.contract.profile == "pdssm-groups-reconstruction"
        assert len(task.generator_elements) == task.input_vocab_size
        assert len(set(task.generator_elements)) == task.input_vocab_size
        assert 0 not in task.generator_elements
        assert task.group is not None
        assert task.contract.generator_labels == tuple(
            task.group.labels[index] for index in task.generator_elements
        )
        reached = {0}
        frontier = [0]
        while frontier:
            state = frontier.pop()
            for generator in task.generator_elements:
                successor = task.group.compose(state, generator)
                if successor not in reached:
                    reached.add(successor)
                    frontier.append(successor)
        assert len(reached) == task.group.order
    for group_name, counts in (("A5", (2, 6, 8, 12)), ("S5", (4, 8, 32))):
        labels = [
            resolve(f"pdssm:{group_name}:{count}").contract.generator_labels
            for count in counts
        ]
        assert all(
            long[: len(short)] == short for short, long in pairwise(labels)
        )


def test_profiles_are_fail_closed_and_the_default_is_the_released_four() -> None:
    """Task families cannot be mixed under a convenient but false benchmark label."""
    assert tuple(task.name for task in resolve_profile("pdssm-regular", None)) == (
        PDSSM_REGULAR_TASKS
    )
    assert tuple(
        task.name for task in resolve_profile("pdssm-groups-reconstruction", None)
    ) == PDSSM_GROUP_TASKS
    with pytest.raises(ValueError, match="another contract"):
        resolve_profile("pdssm-regular", ["A5"])
    with pytest.raises(ValueError, match="another contract"):
        resolve_profile("walker-group-prefix", ["pdssm:A5:2"])
    with pytest.raises(ValueError, match="not a published row"):
        resolve("pdssm:A5:4")
