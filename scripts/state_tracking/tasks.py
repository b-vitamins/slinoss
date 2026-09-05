"""State-tracking tasks, with the benchmark contract attached to every task.

The five named automaton generators are transcriptions of files under
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

    task                    inputs  outputs  supervised  what the state is
    parity                       3        3  last        one bit, ``b`` count mod 2
    even_pairs                   3        3  last        the first token
    cycle_nav                    9        9  last        a position on ``Z_5``
    mod_arith_no_brack          10       10  last        a residue mod 5
    mod_arith_w_brack           12       12  last        a bracket-stack residue
    <group spec>               |G|      |G|  all         every prefix product
    pdssm:<group>:<generators>    n      |G|  last        final generator-word state

Token 0 is never emitted by an automaton task, which is what lets the batcher pad with
it. In an all-elements group task, 0 is the identity, so padding is also a no-op. In a
generator-alphabet group task, 0 is a real action; right padding is nevertheless outside
the loss mask and occurs after every supervised position, so causality keeps it from
altering a scored state.

The distinction in the last two rows is essential.  Walker/Merrill's group word problem
uses every group element as an input and labels every prefix.  PD-SSM's non-solvable
table uses a small generator alphabet and a group-state output alphabet.  The released
PD-SSM repository does not include that table's data generator or its randomly selected
extra permutations.  This module therefore exposes the released regular tasks as exact,
the two-generator A5 task as a cross-release reconstruction from IBM's predecessor, and
the remaining PD-SSM group rows as explicitly labelled deterministic paper
reconstructions.  It never calls those unreleased rows exact.
"""

from __future__ import annotations

import hashlib
import random
import re
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

PDSSM_REVISION = "8682e78101be84f67ceb64702855e5d9e820f7d2"
"""Revision of IBM's released PD-SSM repository mirrored under ``.sources``."""

IBM_A5_REVISION = "5bdc7f7a6a7ad01c1db67ea1f68800810fe6cf19"
"""Revision of IBM's predecessor release containing the two-generator A5 task."""

WALKER_REVISION = "243cb30fcd85406a94f2810ec762c59e6e2bb1c7"
"""Revision of Walker's released all-elements, all-prefix group task."""

PDSSM_REGULAR_PROFILE = "pdssm-regular"
PDSSM_GROUP_PROFILE = "pdssm-groups-reconstruction"
WALKER_GROUP_PROFILE = "walker-group-prefix"
WALKER_EXTENSION_PROFILE = "walker-extension"


@dataclass(frozen=True)
class TaskContract:
    """The provenance and fidelity of a task definition.

    ``fidelity`` is deliberately categorical rather than a prose footnote.  A record
    consumer can reject reconstructed axes without guessing from a task name.
    """

    profile: str
    fidelity: str
    source: str
    revision: str
    implementation: str
    generator_selection: str = "not-applicable"
    generator_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fidelity not in {
            "source-exact",
            "cross-release-reconstruction",
            "paper-reconstruction",
            "extension",
        }:
            raise ValueError(f"unknown task-contract fidelity {self.fidelity!r}")


CUSTOM_CONTRACT = TaskContract(
    profile="custom",
    fidelity="extension",
    source="caller-defined",
    revision="unversioned",
    implementation="caller-defined Task",
)


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
        input_vocab_size: Number of input symbols.
        output_vocab_size: Number of head classes. Defaults to ``input_vocab_size``;
            it differs for generator-alphabet group tasks.
        sample: The generator.
        supervision: ``last`` or ``all``. Reported in the record because it decides what
            an accuracy is an accuracy over.
        min_length: Shortest sequence the generator accepts.
        group: The group, for a word-problem task; None for an automaton task.
        contract: Source and fidelity attached to every emitted record.
        generator_elements: Group-element indices represented by input symbols.
    """

    name: str
    input_vocab_size: int
    sample: SampleFn
    supervision: str
    min_length: int = 1
    group: Group | None = None
    output_vocab_size: int | None = None
    contract: TaskContract = CUSTOM_CONTRACT
    generator_elements: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.input_vocab_size < 1:
            raise ValueError(f"{self.name}: input_vocab_size must be positive")
        if self.output_vocab_size is None:
            object.__setattr__(self, "output_vocab_size", self.input_vocab_size)
        if self.output_vocab_size is None or self.output_vocab_size < 1:
            raise ValueError(f"{self.name}: output_vocab_size must be positive")
        if self.supervision not in ("last", "all"):
            raise ValueError(f"{self.name}: supervision must be last or all")
        if self.min_length < 1:
            raise ValueError(f"{self.name}: min_length must be positive")
        if self.generator_elements:
            if self.group is None:
                raise ValueError(f"{self.name}: generators require a group")
            if len(self.generator_elements) != self.input_vocab_size:
                raise ValueError(
                    f"{self.name}: {len(self.generator_elements)} generators for "
                    f"input_vocab_size {self.input_vocab_size}"
                )
            if len(set(self.generator_elements)) != len(self.generator_elements):
                raise ValueError(f"{self.name}: generator elements repeat")
            if any(
                not 0 <= element < self.group.order
                for element in self.generator_elements
            ):
                raise ValueError(f"{self.name}: generator element outside the group")
            if self.output_vocab_size != self.group.order:
                raise ValueError(
                    f"{self.name}: generator task needs {self.group.order} output classes"
                )

    @property
    def vocab_size(self) -> int:
        """The legacy shared vocabulary, only when input and output really share it.

        Raising on an asymmetric task prevents the exact bug this harness used to have:
        silently sizing both the embedding and classifier from the generator count.
        """
        if self.input_vocab_size != self.output_vocab_size:
            raise ValueError(
                f"{self.name}: vocab_size is ambiguous; use input_vocab_size or "
                "output_vocab_size"
            )
        return self.input_vocab_size


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


def generator_word_problem(
    group: Group, generator_elements: tuple[int, ...]
) -> SampleFn:
    """A final-state task over a small alphabet of group generators.

    Input symbol ``i`` applies ``generator_elements[i]``.  The classifier still predicts
    one of all ``|G|`` states, so this task requires distinct input and output vocabulary
    sizes.  Only the final state is supervised, matching IBM's released two-generator A5
    task and the final-label scaffold used by PD-SSM's released regular benchmark.
    """

    def sample(seed: int, min_length: int, max_length: int) -> Sample:
        _check_lengths(min_length, max_length)
        generator = _rng(seed)
        length = _draw(generator, min_length, max_length + 1)
        ids = tuple(
            int(token)
            for token in torch.randint(
                0, len(generator_elements), (length,), generator=generator
            ).tolist()
        )
        elements = tuple(generator_elements[token] for token in ids)
        return _last(ids, group.prefix(elements)[-1])

    return sample


def _contract(
    profile: str,
    fidelity: str,
    source: str,
    revision: str,
    implementation: str,
    *,
    generator_selection: str = "not-applicable",
    generator_labels: tuple[str, ...] = (),
) -> TaskContract:
    """Build a contract without repeating field names in the task table."""
    return TaskContract(
        profile=profile,
        fidelity=fidelity,
        source=source,
        revision=revision,
        implementation=implementation,
        generator_selection=generator_selection,
        generator_labels=generator_labels,
    )


_PDSSM_REGULAR_CONTRACT = _contract(
    PDSSM_REGULAR_PROFILE,
    "source-exact",
    "IBM/expressive-sparse-state-space-model:state_tracking_PyTorch",
    PDSSM_REVISION,
    "scripts.state_tracking.tasks; byte-pinned fixtures against released generators",
)

_WALKER_EXTENSION_CONTRACT = _contract(
    WALKER_EXTENSION_PROFILE,
    "extension",
    "Benjamin-Walker/structured-linear-cdes:data_dir/fl_tasks",
    WALKER_REVISION,
    "scripts.state_tracking.tasks.mod_arith_w_brack; not in PD-SSM Table 2",
)


AUTOMATA: dict[str, Task] = {
    "parity": Task("parity", 3, parity, "last", contract=_PDSSM_REGULAR_CONTRACT),
    "even_pairs": Task(
        "even_pairs", 3, even_pairs, "last", contract=_PDSSM_REGULAR_CONTRACT
    ),
    "cycle_nav": Task(
        "cycle_nav",
        MAX_POSITION + 4,
        cycle_nav,
        "last",
        contract=_PDSSM_REGULAR_CONTRACT,
    ),
    "mod_arith_no_brack": Task(
        "mod_arith_no_brack",
        MODULUS + 5,
        mod_arith_no_brack,
        "last",
        contract=_PDSSM_REGULAR_CONTRACT,
    ),
    "mod_arith_w_brack": Task(
        "mod_arith_w_brack",
        MODULUS + 7,
        mod_arith_w_brack,
        "last",
        min_length=2,
        contract=_WALKER_EXTENSION_CONTRACT,
    ),
}
"""The five automaton tasks, keyed as upstream's module names.

`expressive-sparse-state-space-model`'s table 2 reports the first four; the fifth is the
bracketed arithmetic every state-tracking matrix since has carried. The thirteen further
files under upstream's ``fl_tasks/`` are Deletang's context-free and context-sensitive
transductions -- reversal, sorting, addition, square roots -- which measure a
transduction rather than a state, and none of the trees this harness is compared against
reports them on this axis."""

PDSSM_REGULAR_TASKS = (
    "parity",
    "even_pairs",
    "cycle_nav",
    "mod_arith_no_brack",
)
"""Exactly the four tasks in PD-SSM's released state-tracking harness."""

PDSSM_GROUP_VARIANTS: dict[str, tuple[int, ...]] = {
    "A5": (2, 6, 8, 12),
    "S5": (4, 8, 32),
}
"""Rows in PD-SSM's non-solvable-group table; generator identities are unreleased."""

PDSSM_GROUP_TASKS = tuple(
    f"pdssm:{group_name}:{count}"
    for group_name, counts in PDSSM_GROUP_VARIANTS.items()
    for count in counts
)

PROFILE_DEFAULTS: dict[str, tuple[str, ...]] = {
    PDSSM_REGULAR_PROFILE: PDSSM_REGULAR_TASKS,
    PDSSM_GROUP_PROFILE: PDSSM_GROUP_TASKS,
    WALKER_GROUP_PROFILE: (),
    WALKER_EXTENSION_PROFILE: ("mod_arith_w_brack",),
}
"""Named, mutually exclusive task families used by the CLI's fail-closed gate."""

_PDSSM_GROUP_SPEC = re.compile(r"^pdssm:(A5|S5):([0-9]+)$")
_PDSSM_RECONSTRUCTION_KEY = "slinoss-pdssm-group-reconstruction-v1"


def _base_generator_labels(group: Group) -> tuple[str, str]:
    """Canonical two generators used as the reconstruction's connected base.

    A5's labels are the exact actions in IBM's released predecessor task: a five-cycle
    and the double transposition ``(01)(23)``.  S5 uses the same cycle and the standard
    transposition ``(01)``; the PD-SSM paper does not release its S5 generator identities.
    """
    if group.name == "A5":
        return ("12340", "10324")
    if group.name == "S5":
        return ("12340", "10234")
    raise ValueError(f"PD-SSM group reconstruction does not define {group.name}")


def _reconstruction_generators(group: Group, count: int) -> tuple[int, ...]:
    """Choose a nested, deterministic stand-in for PD-SSM's unreleased random set."""
    base_labels = _base_generator_labels(group)
    base = tuple(group.labels.index(label) for label in base_labels)
    candidates = [
        index
        for index in range(1, group.order)
        if index not in base
    ]
    candidates.sort(
        key=lambda index: hashlib.sha256(
            f"{_PDSSM_RECONSTRUCTION_KEY}:{group.name}:{group.labels[index]}".encode()
        ).digest()
    )
    return base + tuple(candidates[: count - len(base)])


def _resolve_pdssm_group(name: str, group_name: str, count: int) -> Task:
    """Resolve one explicitly reconstructed row of PD-SSM's group table."""
    allowed = PDSSM_GROUP_VARIANTS[group_name]
    if count not in allowed:
        raise ValueError(
            f"{name}: generator count is not a published row; {group_name} has {allowed}"
        )
    group = parse(group_name)
    elements = _reconstruction_generators(group, count)
    labels = tuple(group.labels[element] for element in elements)
    if group_name == "A5" and count == 2:
        fidelity = "cross-release-reconstruction"
        source = "IBM/selective-dense-state-space-model:tasks/regular/A5.py"
        revision = IBM_A5_REVISION
        selection = (
            "exact two action matrices from IBM predecessor; sample stream adapted to "
            "this harness"
        )
    else:
        fidelity = "paper-reconstruction"
        source = "PD-SSM paper Table nonsolvable; generator identities not released"
        revision = PDSSM_REVISION
        selection = (
            f"{_PDSSM_RECONSTRUCTION_KEY}; canonical generating pair followed by "
            "SHA-256-ranked distinct non-identity permutations"
        )
    contract = _contract(
        PDSSM_GROUP_PROFILE,
        fidelity,
        source,
        revision,
        "scripts.state_tracking.tasks.generator_word_problem",
        generator_selection=selection,
        generator_labels=labels,
    )
    return Task(
        name,
        count,
        generator_word_problem(group, elements),
        "last",
        group=group,
        output_vocab_size=group.order,
        contract=contract,
        generator_elements=elements,
    )


def resolve(name: str) -> Task:
    """The task one spec names.

    Args:
        name: An :data:`AUTOMATA` key; a Walker/Merrill all-elements group spec such as
            ``A5``; or an explicitly reconstructed PD-SSM table row such as
            ``pdssm:A5:2``.

    Returns:
        The task.

    Raises:
        ValueError: On a spec that is neither, naming the automaton keys. A group spec's
            own error carries through.
    """
    if name in AUTOMATA:
        return AUTOMATA[name]
    match = _PDSSM_GROUP_SPEC.fullmatch(name)
    if match is not None:
        return _resolve_pdssm_group(name, match.group(1), int(match.group(2)))
    if name.startswith("pdssm:"):
        raise ValueError(
            f"no reconstructed PD-SSM group task {name!r}; choices are "
            f"{PDSSM_GROUP_TASKS}"
        )
    try:
        group = parse(name)
    except ValueError as exc:
        raise ValueError(
            f"no task {name!r}; automata are {sorted(AUTOMATA)}, reconstructed PD-SSM "
            f"rows are {PDSSM_GROUP_TASKS}, or use a group spec ({exc})"
        ) from exc
    contract = _contract(
        WALKER_GROUP_PROFILE,
        "source-exact",
        "Benjamin-Walker/structured-linear-cdes:data_dir/dataloaders.py",
        WALKER_REVISION,
        "scripts.state_tracking.tasks.word_problem; group elements relabelled bijectively",
        generator_selection="all group elements, uniformly sampled",
        generator_labels=group.labels,
    )
    return Task(
        group.name,
        group.order,
        word_problem(group),
        "all",
        group=group,
        contract=contract,
        generator_elements=tuple(range(group.order)),
    )


def resolve_profile(profile: str, names: Sequence[str] | None) -> tuple[Task, ...]:
    """Resolve tasks through a named family and reject cross-family mixtures.

    The gate is intentionally strict.  A bare ``A5`` cannot be confused with
    ``pdssm:A5:2`` just because both have 60 output states.
    """
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"unknown profile {profile!r}; choices are {tuple(PROFILE_DEFAULTS)}")
    selected = PROFILE_DEFAULTS[profile] if names is None else tuple(names)
    if not selected:
        raise ValueError(f"profile {profile!r} requires at least one explicit task")
    tasks = tuple(resolve(name) for name in selected)
    wrong = [task.name for task in tasks if task.contract.profile != profile]
    if wrong:
        owners = {task.name: task.contract.profile for task in tasks if task.name in wrong}
        raise ValueError(
            f"profile {profile!r} cannot run tasks from another contract: {owners}"
        )
    return tasks
