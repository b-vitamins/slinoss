"""Finite groups by Cayley table: the word problem's ground truth.

Upstream builds its groups with `abstract_algebra`: ``generate_cyclic_group`` for
``Z_n``, ``generate_symmetric_group`` for ``S_n``, ``S_n.commutator_subalgebra()`` for
``A_n``, and ``*`` for the direct product, with the group named by a string such as
``A5`` or ``Z60_x_S3``. That package is not a dependency of this tree, so the tables are
built here from permutations and residues.

One divergence, stated once: the element order is this module's own. Element 0 is the
identity in every group, ``S_n`` and ``A_n`` list permutations lexicographically, and a
product lists pairs with the left factor major. Upstream's order is whatever
`abstract_algebra` enumerated and is not reproduced. Relabelling elements is a bijection
on the token vocabulary that carries the word problem to an isomorphic copy of itself, so
no accuracy moves under it; the group, the sequence distribution and the label function
are unchanged.

Composition is ``compose(p, q)[i] = p[q[i]]`` -- the right factor applies first -- and a
prefix product folds from the identity on the left, ``acc <- compose(acc, token)``, which
is upstream's ``acc := group_reduce(acc, x)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import permutations

MAX_ORDER = 512
"""Largest group a table is built for.

The order is the task's vocabulary size and the table is ``order**2`` entries, so this
bounds both. Every group the state-tracking literature runs is far below it: ``A5`` is
60, ``S5`` is 120, ``Z60`` is 60, ``A6`` is 360. ``S6`` at 720 is refused."""

_SPEC = re.compile(r"^([SZA])([0-9]+)$")
"""One factor of a group spec: a family letter and a degree."""

PRODUCT_SEPARATOR = "_x_"
"""Separator between the factors of a direct product, upstream's spelling."""


@dataclass(frozen=True)
class Group:
    """A finite group as a Cayley table.

    Attributes:
        name: Spec that built it, e.g. ``A5`` or ``Z60_x_S3``.
        labels: One label per element, for a record or a message. Never read by the
            task; the tokens are indices into this tuple.
        table: ``table[a][b]`` is the index of ``a . b``. Square, of side
            ``len(labels)``.

    Raises:
        ValueError: On an order outside ``[1, MAX_ORDER]``, a table that is not square,
            an entry out of range, or element 0 not acting as the identity. The identity
            check is what lets the prefix product start at index 0.
    """

    name: str
    labels: tuple[str, ...]
    table: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        order = len(self.labels)
        if not 1 <= order <= MAX_ORDER:
            raise ValueError(f"{self.name}: order {order} is outside [1, {MAX_ORDER}]")
        if len(self.table) != order:
            raise ValueError(
                f"{self.name}: table has {len(self.table)} rows for {order} elements"
            )
        for index, row in enumerate(self.table):
            if len(row) != order:
                raise ValueError(
                    f"{self.name}: row {index} has {len(row)} entries, not {order}"
                )
            for entry in row:
                if not 0 <= entry < order:
                    raise ValueError(f"{self.name}: entry {entry} is not an element")
        for element in range(order):
            if self.table[0][element] != element or self.table[element][0] != element:
                raise ValueError(f"{self.name}: element 0 is not the identity")

    @property
    def order(self) -> int:
        """Element count, and the task's vocabulary size."""
        return len(self.labels)

    def compose(self, left: int, right: int) -> int:
        """The product ``left . right``.

        Args:
            left: Element index.
            right: Element index.

        Returns:
            The product's index.

        Raises:
            IndexError: On an index outside the group.
        """
        return self.table[left][right]

    def prefix(self, tokens: Sequence[int]) -> tuple[int, ...]:
        """Running product of a word, one output per input position.

        Args:
            tokens: Element indices.

        Returns:
            ``out[i]`` is the product of ``tokens[:i + 1]`` in order, folded from the
            identity. Empty for an empty word.

        Raises:
            IndexError: On a token outside the group.
        """
        state = 0
        out: list[int] = []
        for token in tokens:
            state = self.table[state][token]
            out.append(state)
        return tuple(out)

    def inverse(self, element: int) -> int:
        """The inverse of one element.

        Args:
            element: Element index.

        Returns:
            The index ``b`` with ``element . b == 0``.

        Raises:
            ValueError: When the row holds no identity, which a validated table cannot
                do; the check is here so a hand-built table cannot pass silently.
        """
        row = self.table[element]
        for candidate, product in enumerate(row):
            if product == 0:
                return candidate
        raise ValueError(f"{self.name}: element {element} has no inverse")


def _inversions(perm: Sequence[int]) -> int:
    """Inversion count of a permutation, whose parity is its sign."""
    return sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )


def _check_order(name: str, order: int) -> None:
    """Refuse an order over :data:`MAX_ORDER` before anything is enumerated.

    :class:`Group` refuses it too, but only after a table has been built, and the table
    is quadratic: ``S7`` would enumerate 5040 permutations and 25 million products on the
    way to the message.

    Args:
        name: Group name, for the message.
        order: The order the constructor is about to build.

    Raises:
        ValueError: When the order is over :data:`MAX_ORDER`.
    """
    if order > MAX_ORDER:
        raise ValueError(f"{name}: order {order} is over {MAX_ORDER}")


def _from_permutations(name: str, perms: Sequence[tuple[int, ...]]) -> Group:
    """Build a permutation group's table.

    Args:
        name: Group name.
        perms: The elements, identity first. Must be closed under composition.

    Returns:
        The group, with ``labels`` the permutations written as digit strings.

    Raises:
        ValueError: When a product falls outside ``perms``, which is the closure
            failure. Nothing else proves the subset is a subgroup.
    """
    index = {perm: position for position, perm in enumerate(perms)}
    table: list[tuple[int, ...]] = []
    for left in perms:
        row: list[int] = []
        for right in perms:
            product = tuple(left[position] for position in right)
            if product not in index:
                raise ValueError(f"{name}: not closed, {left} . {right} is outside")
            row.append(index[product])
        table.append(tuple(row))
    labels = tuple("".join(str(position) for position in perm) for perm in perms)
    return Group(name, labels, tuple(table))


def cyclic(degree: int) -> Group:
    """``Z_n``, the integers modulo ``degree`` under addition.

    Args:
        degree: Order of the group. At least 1.

    Returns:
        The group. Element ``a`` is the residue ``a``, so it is already its own label.

    Raises:
        ValueError: On a degree below 1, or an order over :data:`MAX_ORDER`.
    """
    if degree < 1:
        raise ValueError(f"Z{degree}: degree must be positive")
    _check_order(f"Z{degree}", degree)
    labels = tuple(str(residue) for residue in range(degree))
    table = tuple(
        tuple((left + right) % degree for right in range(degree))
        for left in range(degree)
    )
    return Group(f"Z{degree}", labels, table)


def symmetric(degree: int) -> Group:
    """``S_n``, every permutation of ``degree`` points.

    Args:
        degree: Points permuted. At least 1. ``S6`` is order 720 and is refused by
            :data:`MAX_ORDER`.

    Returns:
        The group, permutations in lexicographic order so the identity is element 0.

    Raises:
        ValueError: On a degree below 1, or an order over :data:`MAX_ORDER`.
    """
    if degree < 1:
        raise ValueError(f"S{degree}: degree must be positive")
    _check_order(f"S{degree}", math.factorial(degree))
    return _from_permutations(f"S{degree}", tuple(permutations(range(degree))))


def alternating(degree: int) -> Group:
    """``A_n``, the even permutations of ``degree`` points.

    ``A_n`` for ``n >= 5`` is simple and non-solvable, which is the whole point of the
    non-solvable half of the state-tracking suite: its word problem is complete for
    ``NC^1`` and no solvable recurrence tracks it.

    Args:
        degree: Points permuted. At least 1. ``A1`` and ``A2`` are trivial.

    Returns:
        The group, even permutations in lexicographic order so the identity is element 0.

    Raises:
        ValueError: On a degree below 1, or an order over :data:`MAX_ORDER`.
    """
    if degree < 1:
        raise ValueError(f"A{degree}: degree must be positive")
    _check_order(f"A{degree}", max(math.factorial(degree) // 2, 1))
    even = tuple(
        perm for perm in permutations(range(degree)) if _inversions(perm) % 2 == 0
    )
    return _from_permutations(f"A{degree}", even)


def product(left: Group, right: Group) -> Group:
    """The direct product ``left x right``.

    Args:
        left: Left factor, major in the element order.
        right: Right factor.

    Returns:
        The group of order ``left.order * right.order``. Element ``i * right.order + j``
        is the pair ``(i, j)``, so element 0 is the pair of identities.

    Raises:
        ValueError: When the product's order is over :data:`MAX_ORDER`.
    """
    width = right.order
    name = f"{left.name}{PRODUCT_SEPARATOR}{right.name}"
    order = left.order * width
    _check_order(name, order)
    labels = tuple(
        f"{outer}:{inner}" for outer in left.labels for inner in right.labels
    )
    table = tuple(
        tuple(
            left.table[a // width][b // width] * width
            + right.table[a % width][b % width]
            for b in range(order)
        )
        for a in range(order)
    )
    return Group(name, labels, table)


def factor(spec: str) -> Group:
    """One factor of a group spec.

    Args:
        spec: A family letter in ``SZA`` and a degree, e.g. ``A5``, ``Z60``, ``S3``.

    Returns:
        The group.

    Raises:
        ValueError: On a spec the pattern does not read, or from the constructor.
    """
    match = _SPEC.match(spec)
    if match is None:
        raise ValueError(f"group must be S, Z or A and a degree, got {spec!r}")
    family, digits = match.groups()
    degree = int(digits)
    if family == "Z":
        return cyclic(degree)
    if family == "S":
        return symmetric(degree)
    return alternating(degree)


def parse(spec: str) -> Group:
    """Read a group spec, direct products included.

    Args:
        spec: Factors joined by :data:`PRODUCT_SEPARATOR`, e.g. ``A5``, ``S5``,
            ``Z60_x_Z2``. Upstream's spelling and upstream's fold order.

    Returns:
        The group.

    Raises:
        ValueError: On an unreadable factor or an order over :data:`MAX_ORDER`.
    """
    factors: Iterable[Group] = [
        factor(piece) for piece in spec.split(PRODUCT_SEPARATOR)
    ]
    return reduce(product, factors)
