"""The state-tracking tasks: five finite automata and the group word problem.

Every generator here is a transcription of one file under
`structured-linear-cdes`'s ``data_dir/fl_tasks/``, which
`expressive-sparse-state-space-model`'s ``state_tracking_PyTorch`` carries byte-identical.
The draw order and the draw shapes are upstream's, call for call, so a sample is
bit-identical to upstream's at the same seed; ``tests/test_state_tracking_tasks.py``
pins that against values taken from upstream itself.

Upstream splits each task into ``generate_sample`` and ``preprocess_data`` and hands the
pair to a ``Dataset``. Here one function per task returns the tensor-ready
:class:`Sample` directly: the intermediate character strings were only ever a step on the
way to the same three vectors, and collapsing them removes a per-item dictionary lookup
from the hot path.

One upstream defect is not transcribed. ``generate_sample`` calls the global
``torch.manual_seed(seed)`` and the dataset calls ``generate_sample`` once per item, so
with in-process loading every batch fetch reseeds the process generator that
initialization and dropout draw from -- the model's stochastic state becomes a function of
the last sample's index. Each generator here seeds a local :class:`torch.Generator`
instead, which consumes the same stream and touches nothing global.

    task                 vocab  supervised  what the state is
    parity                   3  last        one bit, the count of ``b`` mod 2
    even_pairs               3  last        the first token, held to the end
    cycle_nav                9  last        a position on ``Z_5``
    mod_arith_no_brack      10  last        a residue mod 5, under BIDMAS
    mod_arith_w_brack       12  last        a residue mod 5, over a bracket tree
    <group spec>      |G|<=512  all         the running product in the group

Token 0 is never emitted by an automaton task, which is what lets the batcher pad with
it. A group task's tokens are ``0..|G| - 1`` and 0 is the identity, so there the pad
token and the identity element coincide: a padded position feeds the recurrence the
identity, and the mask keeps it out of the loss either way.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

from scripts.state_tracking.groups import Group, parse

MAX_POSITION = 5
"""Positions on the cycle in ``cycle_nav``, upstream's ``max_position``."""

MODULUS = 5
"""Modulus of both arithmetic tasks, upstream's ``modulus``."""

PAD_TOKEN = 0
"""The batcher's fill. Emitted by no automaton task; the identity in a group task."""


class Sample(NamedTuple):
    """One example, ready for the batcher.

    Attributes:
        ids: Input tokens.
        targets: One target per position. Whatever sits at an unsupervised position is
            never read; it is zero, as upstream's ``torch.zeros_like`` leaves it.
        supervised: True where the position carries a target. Exactly one position for
            an automaton task, every position for a group task.
    """

    ids: tuple[int, ...]
    targets: tuple[int, ...]
    supervised: tuple[bool, ...]


SampleFn = Callable[[int, int, int], Sample]
"""``(seed, min_length, max_length) -> Sample``. The seed is the whole draw state."""


@dataclass(frozen=True)
class Task:
    """One task.

    Attributes:
        name: Spec that named it.
        vocab_size: Tokens, and classes on the head. Upstream's ``data_dim`` and
            ``label_dim`` are both this: input tokens and labels share one vocabulary.
        sample: The generator.
        supervision: ``last`` or ``all``. Reported in the record because it decides what
            an accuracy is an accuracy over.
        min_length: Shortest sequence the generator accepts.
        group: The group, for a word-problem task; None for an automaton task.
    """

    name: str
    vocab_size: int
    sample: SampleFn
    supervision: str
    min_length: int = 1
    group: Group | None = None

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"{self.name}: vocab_size must be positive")
        if self.supervision not in ("last", "all"):
            raise ValueError(f"{self.name}: supervision must be last or all")
        if self.min_length < 1:
            raise ValueError(f"{self.name}: min_length must be positive")


def _rng(seed: int) -> torch.Generator:
    """A CPU generator at ``seed``.

    ``torch.manual_seed`` seeds this same engine through the default generator, so a
    local one consumes the identical stream and upstream's draws reproduce exactly.

    Args:
        seed: Draw state.

    Returns:
        The generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _draw(generator: torch.Generator, low: int, high: int) -> int:
    """One integer from ``[low, high)``, upstream's ``torch.randint(low, high, (1,))``."""
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _check_lengths(min_length: int, max_length: int) -> None:
    """Refuse an empty length range, upstream's own guard.

    Args:
        min_length: Shortest sequence.
        max_length: Longest sequence, inclusive.

    Raises:
        ValueError: When the range is empty.
    """
    if min_length > max_length:
        raise ValueError(f"min_length {min_length} is over max_length {max_length}")


def _last(ids: Sequence[int], target: int) -> Sample:
    """A sample supervised at its final position only.

    Args:
        ids: Input tokens. Non-empty.
        target: The label, at the last position.

    Returns:
        The sample, targets zero everywhere but the end.
    """
    length = len(ids)
    targets = [0] * length
    targets[-1] = target
    supervised = [False] * length
    supervised[-1] = True
    return Sample(tuple(ids), tuple(targets), tuple(supervised))


def parity(seed: int, min_length: int, max_length: int) -> Sample:
    """Parity of the ``b`` count over an ``a``/``b`` word.

    ``fl_tasks/parity.py``. The state is one bit and the automaton is ``Z_2``: the task
    is the shortest test that a recurrence can hold a bit across the whole sequence.

    Args:
        seed: Draw state.
        min_length: Shortest sequence.
        max_length: Longest sequence, inclusive.

    Returns:
        Tokens in ``{1, 2}`` for ``a``, ``b``; the label is 1 on even, 2 on odd.

    Raises:
        ValueError: On an empty length range.
    """
    generator = _rng(seed)
    _check_lengths(min_length, max_length)
    length = _draw(generator, min_length, max_length + 1)
    bits = [_draw(generator, 0, 2) for _ in range(length)]
    return _last([1 + bit for bit in bits], 1 if sum(bits) % 2 == 0 else 2)


def even_pairs(seed: int, min_length: int, max_length: int) -> Sample:
    """Whether the first and last tokens of an ``a``/``b`` word agree.

    ``fl_tasks/even_pairs.py``. The state is the first token, so a recurrence has to
    carry one bit unchanged rather than update it; the failure mode it separates is a
    state that decays.

    Args:
        seed: Draw state.
        min_length: Shortest sequence.
        max_length: Longest sequence, inclusive.

    Returns:
        Tokens in ``{1, 2}``; the label is 1 when the ends agree, 2 when they do not.

    Raises:
        ValueError: On an empty length range.
    """
    generator = _rng(seed)
    _check_lengths(min_length, max_length)
    length = _draw(generator, min_length, max_length + 1)
    bits = [_draw(generator, 0, 2) for _ in range(length)]
    return _last([1 + bit for bit in bits], 1 if bits[0] == bits[-1] else 2)


def cycle_nav(seed: int, min_length: int, max_length: int) -> Sample:
    """Position on a five-cycle after a word of stay, forward and back.

    ``fl_tasks/cycle_nav.py``. The automaton is ``Z_5``, so the task is modular counting
    and the length-generalization question is whether the count survives past the trained
    length. The moves are drawn as one length-vector, as upstream draws them.

    Args:
        seed: Draw state.
        min_length: Shortest sequence.
        max_length: Longest sequence, inclusive.

    Returns:
        Tokens in ``{1, 2, 3}`` for stay, ``+1``, ``-1``; the label is ``4 + position``,
        so labels occupy ``4..8`` and never collide with an input token.

    Raises:
        ValueError: On an empty length range.
    """
    generator = _rng(seed)
    _check_lengths(min_length, max_length)
    length = _draw(generator, min_length, max_length + 1)
    moves = torch.randint(0, 3, (length,), generator=generator)
    forward = int((moves == 1).sum())
    backward = int((moves == 2).sum())
    position = (forward - backward) % MAX_POSITION
    return _last([1 + int(move) for move in moves.tolist()], 4 + position)


def mod_arith_no_brack(seed: int, min_length: int, max_length: int) -> Sample:
    """Value of a flat arithmetic expression modulo 5, under BIDMAS.

    ``fl_tasks/mod_arith_no_brack.py``. Products bind before sums, so the state is a
    residue and a pending factor: strictly more than ``cycle_nav`` carries and still a
    finite automaton. The drawn length is rounded up to even, since the expression is
    ``num (op num)* =``.

    Args:
        seed: Draw state.
        min_length: Shortest sequence before the rounding.
        max_length: Longest sequence, inclusive, before the rounding.

    Returns:
        Tokens with digits at ``5..9`` (the residue plus 5), operators ``1``, ``2``, ``3``
        for ``+``, ``-``, ``*``, and ``4`` for ``=`` at the final position. The label is
        the value plus 5, on the digits' own codes.

    Raises:
        ValueError: On an empty length range.
    """
    generator = _rng(seed)
    _check_lengths(min_length, max_length)
    drawn = _draw(generator, min_length, max_length + 1)
    length = drawn + 1 if drawn % 2 == 1 else drawn

    tokens = [0] * length
    for position in range(0, length, 2):
        tokens[position] = _draw(generator, 5, 10)
    for position in range(1, length - 1, 2):
        tokens[position] = _draw(generator, 1, 4)
    tokens[-1] = 4

    values = [tokens[0] - 5]
    operators: list[int] = []
    for position in range(1, length - 2, 2):
        operator = tokens[position]
        value = tokens[position + 1] - 5
        if operator == 3:
            values[-1] *= value
        else:
            values.append(value)
            operators.append(operator)

    total = values[0]
    for index, operator in enumerate(operators):
        if operator == 1:
            total += values[index + 1]
        else:
            total -= values[index + 1]
    return _last(tokens, total % MODULUS + 5)


def mod_arith_w_brack(seed: int, min_length: int, max_length: int) -> Sample:
    """Value of a bracketed arithmetic expression modulo 5.

    ``fl_tasks/mod_arith_w_brack.py``. The expression is a tree, so the language is
    deterministic context-free rather than regular and the state is a stack of residues;
    it sits above the four automaton tasks and below the non-solvable groups.

    The recursion spends its length budget as ``left + right + 3`` for the two brackets
    and the operator, with the left part drawn from ``[1, length - 4]``, so both parts are
    at least one token and the produced string is exactly the requested length. The four
    base cases are ``n``, ``-n``, ``(n)`` and ``(-n)``.

    Args:
        seed: Draw state.
        min_length: Shortest sequence. At least 2: at 1 the sequence is the ``=`` alone.
        max_length: Longest sequence, inclusive.

    Returns:
        Tokens with digits at ``1..5``, ``6``, ``7``, ``8`` for ``+``, ``-``, ``*``, ``9``
        and ``10`` for the brackets, and ``11`` for ``=`` at the final position. The
        label is the value plus 1, on the digits' own codes.

    Raises:
        ValueError: On an empty length range, or a ``min_length`` under 2.
    """
    generator = _rng(seed)
    _check_lengths(min_length, max_length)
    if min_length < 2:
        raise ValueError(f"min_length must be at least 2, got {min_length}")

    def terminal() -> tuple[str, int]:
        value = _draw(generator, 0, MODULUS)
        return str(value), value

    def expression(length: int) -> tuple[str, int]:
        if length == 1:
            text, value = terminal()
            return text, value
        if length == 2:
            text, value = terminal()
            return f"-{text}", (-value) % MODULUS
        if length == 3:
            text, value = terminal()
            return f"({text})", value % MODULUS
        if length == 4:
            text, value = terminal()
            return f"(-{text})", (-value) % MODULUS
        left_length = _draw(generator, 1, length - 3)
        left_text, left_value = expression(left_length)
        right_text, right_value = expression(length - (left_length + 3))
        operator = _draw(generator, 1, 4)
        if operator == 1:
            return f"({left_text}+{right_text})", (left_value + right_value) % MODULUS
        if operator == 2:
            return f"({left_text}-{right_text})", (left_value - right_value) % MODULUS
        return f"({left_text}*{right_text})", (left_value * right_value) % MODULUS

    vocab = {
        "+": MODULUS + 1,
        "-": MODULUS + 2,
        "*": MODULUS + 3,
        "(": MODULUS + 4,
        ")": MODULUS + 5,
    }
    for digit in range(MODULUS):
        vocab[str(digit)] = digit + 1

    length = _draw(generator, min_length, max_length + 1) - 1
    text, value = expression(length)
    ids = [vocab[character] for character in text]
    ids.append(MODULUS + 6)
    return _last(ids, vocab[str(value)])


def word_problem(group: Group) -> SampleFn:
    """The prefix-product generator for one group.

    `structured-linear-cdes`'s ``GroupCompositionDataset``. Every position is supervised
    with the product of the word so far, so the loss reads the whole state trajectory
    rather than one label, and the sequence distribution is uniform over the group.

    The draws are Python's, not torch's: upstream's group dataset already used
    ``random.Random(seed)`` per item, which is the pattern the automaton tasks needed and
    did not have.

    Args:
        group: The group.

    Returns:
        A generator over that group. Tokens are element indices and the target at a
        position is the running product from the identity.
    """

    def sample(seed: int, min_length: int, max_length: int) -> Sample:
        _check_lengths(min_length, max_length)
        rng = random.Random(seed)
        length = rng.randint(min_length, max_length)
        ids = tuple(rng.randint(0, group.order - 1) for _ in range(length))
        return Sample(ids, group.prefix(ids), (True,) * length)

    return sample


AUTOMATA: dict[str, Task] = {
    "parity": Task("parity", 3, parity, "last"),
    "even_pairs": Task("even_pairs", 3, even_pairs, "last"),
    "cycle_nav": Task("cycle_nav", MAX_POSITION + 4, cycle_nav, "last"),
    "mod_arith_no_brack": Task(
        "mod_arith_no_brack", MODULUS + 5, mod_arith_no_brack, "last"
    ),
    "mod_arith_w_brack": Task(
        "mod_arith_w_brack", MODULUS + 7, mod_arith_w_brack, "last", min_length=2
    ),
}
"""The five automaton tasks, keyed as upstream's module names.

`expressive-sparse-state-space-model`'s table 2 reports the first four; the fifth is the
bracketed arithmetic every state-tracking matrix since has carried. The thirteen further
files under upstream's ``fl_tasks/`` are Deletang's context-free and context-sensitive
transductions -- reversal, sorting, addition, square roots -- which measure a
transduction rather than a state, and none of the trees this harness is compared against
reports them on this axis."""


def resolve(name: str) -> Task:
    """The task one spec names.

    Args:
        name: An :data:`AUTOMATA` key, or a group spec such as ``A5``, ``S5``, ``Z60`` or
            ``A5_x_Z2``. Upstream spells the group task ``A5`` too.

    Returns:
        The task.

    Raises:
        ValueError: On a spec that is neither, naming the automaton keys. A group spec's
            own error carries through.
    """
    if name in AUTOMATA:
        return AUTOMATA[name]
    try:
        group = parse(name)
    except ValueError as exc:
        raise ValueError(
            f"no task {name!r}; automata are {sorted(AUTOMATA)}, or a group spec ({exc})"
        ) from exc
    return Task(group.name, group.order, word_problem(group), "all", group=group)
