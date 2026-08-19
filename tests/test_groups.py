"""The group axis on ``B`` and ``C``.

``B`` and ``C`` carry ``G`` groups where every other operand carries ``H`` heads,
and head ``h`` reads group ``h // (H // G)``. Grouping is defined by one identity:
a grouped call equals the ungrouped call on the broadcast operands. That identity
is what these tests pin, forward and backward, rather than any particular sharing
arithmetic, because the identity is what the kernels have to reproduce and the
arithmetic is how they happen to reproduce it.

``G`` does not interact with the shape axes: the group index selects which slice
of ``B`` a block reads and changes nothing about the tiling, the chunking, or the
contraction extents. So it is swept here on its own, at the two structurally
distinct ends and one point between them, and not crossed with the shape sweeps
that live beside it.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from slinoss.config import SLinOSSConfig
from slinoss.ops.so3ssd import so3ssd_bwd_ref, so3ssd_ref
from slinoss.ops.so3ssd.reference import from_heads, to_heads
from tests.conftest import ScanInputs, make_inputs, max_err, rel_err

HEADS = 4
CHUNK = 16

# G = 1 and G = H are the two arcs the helpers special-case, and G = 2 is the only
# one that exercises the general reshape. Nothing else is a distinct arc.
GROUPS = [pytest.param(g, id=f"G{g}") for g in (1, 2, HEADS)]


def broadcast(inp: ScanInputs) -> ScanInputs:
    """The same call with ``B``, ``C``, and ``b_prev`` expanded onto ``HEADS``.

    Materialized, because at ``G = 1`` the broadcast is a stride-0 view and the
    operator refuses a non-contiguous operand rather than repacking it.
    """
    return inp._replace(
        B=to_heads(inp.B, HEADS).contiguous(),
        C=to_heads(inp.C, HEADS).contiguous(),
        b_prev=None if inp.b_prev is None else to_heads(inp.b_prev, HEADS).contiguous(),
    )


def test_from_heads_is_the_adjoint_of_to_heads() -> None:
    """``<from_heads(y), x> == <y, to_heads(x)>`` at every ``G``.

    The broadcast and the reduction are one linear map and its transpose, which is
    the whole content of the group axis: it is why autograd through the reference's
    broadcast produces the cross-head sum that the analytic backward writes out by
    hand, and why the two cannot disagree. Asserted directly rather than inferred
    from a gradient, so a failure names the map rather than the operator.
    """
    gen = torch.Generator().manual_seed(0)
    for groups in (1, 2, HEADS):
        x = torch.randn(3, groups, 5, generator=gen, dtype=torch.float64)
        y = torch.randn(3, HEADS, 5, generator=gen, dtype=torch.float64)
        lhs = float((from_heads(y, groups) * x).sum())
        rhs = float((y * to_heads(x, HEADS)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-12), f"G={groups}"


def test_to_heads_and_from_heads_reject_a_group_count_that_does_not_divide() -> None:
    """Both directions refuse ``G`` that leaves a remainder.

    With a remainder some head would index past the group axis, so the map is not
    defined; the reference cannot fall back to a partial group without silently
    dropping heads.
    """
    t = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="G=3 does not divide H=4"):
        to_heads(t, 4)
    with pytest.raises(ValueError, match="G=3 does not divide H=4"):
        from_heads(torch.zeros(2, 4, 4), 3)


@pytest.mark.parametrize("groups", GROUPS)
def test_grouped_call_matches_the_broadcast_ungrouped_call(groups: int) -> None:
    """A grouped forward equals the ungrouped forward on the broadcast operands.

    This is the definition of the group axis, so it is asserted exactly: the two
    calls perform the same float64 arithmetic on the same values in the same order,
    and the only difference is where those values were read from. A tolerance here
    would admit a genuine indexing error at a small ``G``.
    """
    inp = make_inputs(heads=HEADS, groups=groups, seqlen=40, dtype=torch.float64)
    wide = broadcast(inp)
    got = so3ssd_ref(*inp.args(), CHUNK, **inp.kw())
    want = so3ssd_ref(*wide.args(), CHUNK, **wide.kw())

    assert max_err(got.y, want.y) == 0.0
    assert max_err(got.state, want.state) == 0.0
    assert max_err(got.u_last, want.u_last) == 0.0
    # b_last is a time slice of the grouped B, so it stays grouped.
    assert tuple(got.b_last.shape) == (
        int(inp.B.shape[0]),
        groups,
        int(inp.B.shape[-1]),
    )
    assert max_err(to_heads(got.b_last, HEADS), want.b_last) == 0.0


@pytest.mark.parametrize("groups", GROUPS)
def test_grouped_gradients_are_the_ungrouped_ones_summed_over_each_group(
    groups: int,
) -> None:
    """Every cotangent of a grouped call, against the ungrouped call reduced.

    ``dB`` and ``dC`` reduce over the heads of their group; every other cotangent is
    unchanged, because no other operand is shared. Run through the analytic backward
    rather than autograd, since autograd through the reference gets the reduction
    from the broadcast for free and so cannot detect a wrong reduction in the
    hand-derived path.

    The seed for ``b_last`` is scattered rather than broadcast. ``b_last`` is an
    output, and the wide call's ``b_last`` is the grouped one broadcast, so the
    equivalent seed is any per-head tensor that reduces back to the grouped seed;
    broadcasting it would seed ``H // G`` copies instead of one.
    """
    inp = make_inputs(heads=HEADS, groups=groups, seqlen=40, dtype=torch.float64)
    wide = broadcast(inp)
    gen = torch.Generator().manual_seed(1)

    def cot(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    bsz, _, seqlen, rows = inp.U.shape
    dim = int(inp.B.shape[-1])
    dy = cot(bsz, HEADS, seqlen, rows)
    dstate = cot(bsz, HEADS, rows, dim)
    du_last = cot(bsz, HEADS, rows)
    db_last = cot(bsz, groups, dim)

    # One head of each group carries the seed and the rest carry zero, so the
    # per-head seed reduces back to db_last exactly.
    per_group = HEADS // groups
    wide_db_last = torch.zeros(bsz, HEADS, dim, dtype=torch.float64)
    wide_db_last[:, ::per_group] = db_last

    got = so3ssd_bwd_ref(dy, dstate, db_last, du_last, *inp.args(), CHUNK, **inp.kw())
    want = so3ssd_bwd_ref(
        dy,
        dstate,
        wide_db_last,
        du_last,
        *wide.args(),
        CHUNK,
        **wide.kw(),
    )

    # Bitwise on the unshared cotangents: the two calls run identical arithmetic.
    assert max_err(got.dU, want.dU) == 0.0
    assert max_err(got.dtrans, want.dtrans) == 0.0
    assert max_err(got.dK, want.dK) == 0.0
    assert got.dz0 is not None and want.dz0 is not None
    assert max_err(got.dz0, want.dz0) == 0.0
    assert got.du_prev is not None and want.du_prev is not None
    assert max_err(got.du_prev, want.du_prev) == 0.0

    # The reduction reassociates a sum over H // G terms, so it is exact only up to
    # float64 rounding of that sum: a few ulps of the largest entry, hence relative
    # and not bitwise. A wrong reduction is off by a whole term, which is O(1).
    ULPS = 1e-15
    for name, mine, theirs in (
        ("dB", got.dB, want.dB),
        ("dC", got.dC, want.dC),
    ):
        assert tuple(mine.shape) == (bsz, groups, seqlen, dim)
        assert rel_err(mine, from_heads(theirs, groups)) < ULPS, name
    assert got.db_prev is not None and want.db_prev is not None
    assert tuple(got.db_prev.shape) == (bsz, groups, dim)
    assert rel_err(got.db_prev, from_heads(want.db_prev, groups)) < ULPS


def test_reference_rejects_a_group_count_that_does_not_divide_heads() -> None:
    """The operator refuses a ``B`` whose group axis does not divide ``H``.

    Reached through the public entry point rather than the private checker, because
    it is the entry point that has to refuse: the group index is computed from
    ``H // G`` inside every kernel and a remainder would send some head past the end
    of ``B``.
    """
    inp = make_inputs(heads=HEADS, groups=HEADS, seqlen=16, dtype=torch.float64)
    bad = inp._replace(
        B=inp.B[:, :3].contiguous(),
        C=inp.C[:, :3].contiguous(),
        b_prev=None if inp.b_prev is None else inp.b_prev[:, :3].contiguous(),
    )
    with pytest.raises(ValueError, match="G=3, H=4"):
        so3ssd_ref(*bad.args(), CHUNK, **bad.kw())


def test_config_group_count_divides_the_head_count() -> None:
    """``n_groups`` is validated at construction and reported as heads per group.

    The kernels take ``H // G`` as a compile-time constant, so a config that
    reaches them with a remainder would have specialized a kernel on a division
    that is not exact.
    """
    cfg = SLinOSSConfig(d_model=256, d_state=48, d_head=64, n_groups=2)
    assert cfg.n_heads == 8
    assert cfg.heads_per_group == 4

    assert SLinOSSConfig(d_model=256, d_state=48, d_head=64).heads_per_group == 8

    with pytest.raises(ValueError, match="n_groups must be positive"):
        SLinOSSConfig(d_model=256, d_state=48, d_head=64, n_groups=0)
    with pytest.raises(ValueError, match="does not divide n_heads"):
        SLinOSSConfig(d_model=256, d_state=48, d_head=64, n_groups=3)
