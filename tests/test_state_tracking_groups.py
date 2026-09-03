"""The group tables the word problem's labels come from.

The label at a position is a prefix product, so a wrong table is a wrong label and an
arm scored against it measures nothing. Nothing else in the harness can catch that:
a permuted or non-associative table still trains, still converges, and still reports an
accuracy.

The tables are built here rather than taken from `abstract_algebra`, so what these tests
pin is that the construction produced a group: the axioms, the known orders, and the two
conventions the rest of the harness reads -- element 0 is the identity, and a direct
product lists pairs with the left factor major. The permutation tables are additionally
checked against composition of the permutations themselves, which is the ground truth the
table is a cache of.
"""

from __future__ import annotations

import math
import random
from itertools import permutations

import pytest

from scripts.state_tracking.groups import (
    MAX_ORDER,
    Group,
    alternating,
    cyclic,
    factor,
    parse,
    product,
    symmetric,
)


def _perm(label: str) -> tuple[int, ...]:
    """A permutation group's label read back as the permutation."""
    return tuple(int(character) for character in label)


def test_known_orders() -> None:
    """The four groups the suite names have the orders the literature names.

    An off-by-one in the parity filter or the enumeration would give ``A5`` order 120 or
    30, which is still a valid group and would still train.
    """
    assert cyclic(60).order == 60
    assert symmetric(5).order == 120
    assert alternating(4).order == 12
    assert alternating(5).order == 60


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(cyclic(6), id="Z6"),
        pytest.param(symmetric(3), id="S3"),
        pytest.param(alternating(4), id="A4"),
        pytest.param(product(cyclic(3), symmetric(3)), id="Z3_x_S3"),
    ],
)
def test_group_axioms(group: Group) -> None:
    """Closure, associativity, identity and inverses over the whole table.

    Held to groups small enough to check exhaustively; ``A5`` is checked by the
    permutation identity instead, which is stronger than a sample of triples.
    """
    order = group.order
    for a in range(order):
        assert group.table[0][a] == a
        assert group.table[a][0] == a
        assert group.compose(a, group.inverse(a)) == 0
        row = group.table[a]
        assert sorted(row) == list(range(order)), "a row must be a permutation"
        column = [group.table[b][a] for b in range(order)]
        assert sorted(column) == list(range(order)), "a column must be a permutation"
    for a in range(order):
        for b in range(order):
            for c in range(order):
                left = group.compose(group.compose(a, b), c)
                right = group.compose(a, group.compose(b, c))
                assert left == right


@pytest.mark.parametrize("degree", [3, 4, 5])
def test_permutation_table_is_composition(degree: int) -> None:
    """``table[a][b]`` is the index of ``perm[a] . perm[b]``, right factor first.

    The table is a cache of permutation composition and this is the only test that
    reads the thing it caches. The composition order is load-bearing: transposing it
    gives the opposite group, which is isomorphic but not equal, and the prefix product
    of a word would then be the product of its reverse.
    """
    group = alternating(degree)
    perms = [_perm(label) for label in group.labels]
    index = {perm: position for position, perm in enumerate(perms)}
    assert perms[0] == tuple(range(degree))
    for a, left in enumerate(perms):
        for b, right in enumerate(perms):
            composed = tuple(left[position] for position in right)
            assert group.table[a][b] == index[composed]


def test_alternating_is_the_even_half() -> None:
    """``A_n`` holds exactly the even permutations, in lexicographic order."""
    degree = 5
    group = alternating(degree)
    inversions = [
        sum(1 for i in range(degree) for j in range(i + 1, degree) if perm[i] > perm[j])
        for perm in (_perm(label) for label in group.labels)
    ]
    assert all(count % 2 == 0 for count in inversions)
    assert group.order == math.factorial(degree) // 2
    assert list(group.labels) == sorted(group.labels)


def test_cyclic_is_addition_modulo_the_degree() -> None:
    """``Z_n``'s table is residue addition, which fixes the token-to-element map."""
    group = cyclic(7)
    for a in range(7):
        for b in range(7):
            assert group.compose(a, b) == (a + b) % 7


def test_product_pairs_with_the_left_factor_major() -> None:
    """Element ``i * right.order + j`` is the pair ``(i, j)``.

    The convention is what makes element 0 the pair of identities. Reversing it would
    still give a group of the right order with a valid identity at 0, so only the
    decomposition catches it.
    """
    left, right = alternating(4), cyclic(3)
    both = product(left, right)
    width = right.order
    assert both.order == left.order * width
    assert both.name == "A4_x_Z3"
    for a in range(both.order):
        for b in range(both.order):
            expected = (
                left.table[a // width][b // width] * width
                + (right.table[a % width][b % width])
            )
            assert both.table[a][b] == expected


def test_prefix_folds_from_the_identity() -> None:
    """A prefix product is ``acc <- compose(acc, token)`` starting at element 0.

    The first entry is therefore the first token itself, which is what ties the label at
    position 0 to the token at position 0. Folding from the right instead would produce
    the reversed word's products, and every entry but the first would be wrong.
    """
    group = alternating(5)
    rng = random.Random(0)
    word = tuple(rng.randrange(group.order) for _ in range(16))
    running = group.prefix(word)
    assert len(running) == len(word)
    assert running[0] == word[0]
    state = 0
    for position, token in enumerate(word):
        state = group.compose(state, token)
        assert running[position] == state
    assert group.prefix(()) == ()


def test_order_over_the_cap_is_refused_before_enumeration() -> None:
    """``S7`` is refused by the pre-check, not by the constructor.

    The constructor's own guard fires only after ``order**2`` products have been built,
    which for ``S7`` is 25 million. The two messages differ, so the assertion pins which
    guard fired rather than merely that one did.
    """
    with pytest.raises(ValueError, match=r"S7: order 5040 is over 512"):
        symmetric(7)
    with pytest.raises(ValueError, match=r"A8: order 20160 is over 512"):
        alternating(8)
    with pytest.raises(ValueError, match=r"is over 512"):
        product(alternating(5), cyclic(60))
    assert symmetric(5).order <= MAX_ORDER


def test_a_table_whose_zero_is_not_the_identity_is_refused() -> None:
    """The prefix product starts at index 0, so that index has to be the identity."""
    swapped = ((1, 0), (0, 1))
    with pytest.raises(ValueError, match="element 0 is not the identity"):
        Group("bad", ("e", "a"), swapped)
    with pytest.raises(ValueError, match="row 1 has 1 entries"):
        Group("ragged", ("e", "a"), ((0, 1), (1,)))
    with pytest.raises(ValueError, match="entry 2 is not an element"):
        Group("wide", ("e", "a"), ((0, 1), (1, 2)))


def test_spec_parsing() -> None:
    """A spec names a group, a direct product folds left, and junk is refused."""
    assert parse("A5").name == "A5"
    assert parse("Z60_x_Z2").order == 120
    assert parse("Z2_x_Z3_x_Z5").name == "Z2_x_Z3_x_Z5"
    assert parse("Z2_x_Z3_x_Z5").order == 30
    assert factor("S4").order == 24
    for spec in ("Q8", "A", "5", "a5", "A5x Z2", ""):
        with pytest.raises(ValueError):
            parse(spec)
    with pytest.raises(ValueError, match="degree must be positive"):
        cyclic(0)


def test_trivial_and_degenerate_degrees() -> None:
    """Degree 1 and 2 give the trivial and two-element groups, not an error.

    ``A1`` and ``A2`` hold the identity alone, which is what ``permutations`` produces,
    and a task on them is a constant function. They are here because ``_check_order``
    takes ``max(n!/2, 1)`` and a zero there would refuse them.
    """
    assert alternating(1).order == 1
    assert alternating(2).order == 1
    assert symmetric(2).order == 2
    assert cyclic(1).order == 1
    assert len(tuple(permutations(range(1)))) == 1
