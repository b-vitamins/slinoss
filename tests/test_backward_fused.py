"""The adjoint of the one-tap factorization against the adjoint of the two-tap one.

:func:`chunked_backward_fused` differentiates the same operator as
:func:`chunked_backward`, so the latter is its ground truth in float64. Three
failure modes belong to the reindex alone: the fused column, whose one cotangent
splits onto ``ap``, ``an`` and ``ls``; the two rank-one residues, which are the only
route to ``bnow``; and the ragged tail, where the shift cotangent has to move over
the padded axis before it is sliced.

Everything else in the derivation -- the offset term, the reverse chunk recurrence,
the table composition, the group reduction -- is shared with the two-tap backward
and tested there. Nothing here re-sweeps it.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from slinoss.ops.so3ssd import chunked_backward_fused, so3ssd_bwd_ref
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

# float64 analytic VJP against float64 analytic VJP. The two forms reassociate the
# same sums, so the gap is reordering roundoff. Worst measured over this file:
# 1.1e-15, on ``dtrans`` at T = 33, L = 16. Run with --tolerance-report to see every
# bound next to what it admitted.
BWD_REL = 1e-13

# The ``ls`` half of the split against autograd through the same nine-term
# reduction, whose order is the backend's. Worst measured: 0.0. The other two halves
# are one copy and one multiply, so they are asserted bitwise instead.
SPLIT_REL = 1e-15

GRAD_NAMES: tuple[str, ...] = (
    "dU",
    "dtrans",
    "dK",
    "dB",
    "dC",
    "dz0",
    "db_prev",
    "du_prev",
)

# name -> (chunk length, make_inputs keywords). One case per path the reindex can
# take, and no more:
#
# - hand, three chunks with no pad, the smallest geometry with a predecessor chunk;
# - grouped, H/G = 2, so the cross-head dB reduction runs under the reindex;
# - square, P == 3N, where a transposed 3x3 or score operand still typechecks;
# - nocarry, no z0/b_prev/u_prev, so slot 0 of chunk 0 shifts in a zero;
# - ragged33 and ragged40, ragged tails of one slot and of eight.
_CASES: dict[str, tuple[int, dict[str, Any]]] = {
    "hand": (4, {"bsz": 1, "heads": 2, "seqlen": 12, "rows": 16, "lanes": 16}),
    "grouped": (
        8,
        {"bsz": 2, "heads": 4, "groups": 2, "seqlen": 24, "rows": 16, "lanes": 16},
    ),
    "square": (
        8,
        {"bsz": 1, "heads": 2, "groups": 2, "seqlen": 32, "rows": 48, "lanes": 16},
    ),
    "nocarry": (
        4,
        {
            "bsz": 1,
            "heads": 2,
            "seqlen": 12,
            "rows": 16,
            "lanes": 16,
            "with_state": False,
            "streaming": False,
        },
    ),
    "ragged33": (16, {"bsz": 1, "heads": 2, "seqlen": 33, "rows": 16, "lanes": 32}),
    "ragged40": (16, {"bsz": 2, "heads": 2, "seqlen": 40, "rows": 16, "lanes": 16}),
}

Cotangents = tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]


def _inputs(name: str, *, seed: int) -> tuple[int, ScanInputs]:
    chunk, spec = _CASES[name]
    return chunk, make_inputs(seed=seed, **spec)


def _cotangents(inp: ScanInputs, seed: int) -> Cotangents:
    """One cotangent per operator output, in ``(dy, dstate, db_last, du_last)`` order."""
    bsz, heads, seqlen, rows = (int(d) for d in inp.U.shape)
    groups = int(inp.B.shape[1])
    state_dim = int(inp.B.shape[-1])
    gen = torch.Generator().manual_seed(seed)

    def like(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    return (
        like(bsz, heads, seqlen, rows),
        like(bsz, heads, rows, state_dim),
        like(bsz, groups, state_dim),
        like(bsz, heads, rows),
    )


@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_backward_matches_the_two_tap_backward(name: str) -> None:
    """Every gradient of both forms under one set of cotangents.

    The mandatory test: a forward parity constrains the sum the reindex rewrites, not
    the derivative of each factor in it, and this is the only check that every one of
    the eight cotangents is right. Two of the geometries are ragged, which is what
    makes the padded-axis shift adjoint load-bearing here.

    cpu only. The device axis sweeps reduction order, which does not interact with
    the reindex and is already swept for the fused forward and the two-tap backward.
    """
    chunk, inp = _inputs(name, seed=307)
    cot = _cotangents(inp, 311)
    want = so3ssd_bwd_ref(*cot, *inp.args(), chunk, **inp.kw())
    got = chunked_backward_fused(*cot, *inp.args(), chunk, **inp.kw())
    for grad_name, mine, theirs in zip(GRAD_NAMES, got, want):
        if theirs is None:
            assert mine is None, f"{grad_name} {name}: expected no gradient"
            continue
        assert mine is not None, f"{grad_name} {name}: missing gradient"
        assert_max_rel(mine, theirs, BWD_REL, f"fused {grad_name} {name}")


def test_afuse_split_matches_autograd() -> None:
    """The three-line split of ``dAfuse``, on its own primal.

    This is the formula the kernels copy, so it is checked against autograd rather
    than only inside the assembled gradient, where ``dap``, ``dan`` and ``dls`` all
    reach ``dtrans`` and ``dK`` through further contractions that could absorb a
    mis-scaled term. Written as an index loop, which is a different expression of the
    identity from the vectorized shift the implementation uses.
    """
    length = 5
    gen = torch.Generator().manual_seed(401)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    ap = rnd(length, 3, 3).requires_grad_(True)
    an = rnd(length, 3, 3).requires_grad_(True)
    # I1: ls <= 0, so every step decay lies in (0,1].
    ls = -rnd(length).abs().requires_grad_(True)
    step = torch.exp(2.0 * ls)
    an_shift = torch.cat([torch.zeros_like(an[:1]), an[:-1]], dim=0)
    afuse = ap + step[:, None, None] * an_shift

    dafuse = rnd(length, 3, 3)
    want_ap, want_an, want_ls = torch.autograd.grad(afuse, (ap, an, ls), dafuse)

    step_d, an_d = step.detach(), an.detach()
    got_ap = dafuse
    got_an = torch.zeros_like(an_d)
    got_ls = torch.zeros_like(step_d)
    for s in range(1, length):
        got_an[s - 1] = step_d[s] * dafuse[s]
        got_ls[s] = 2.0 * step_d[s] * (dafuse[s] * an_d[s - 1]).sum()

    # One copy and one multiply, so a tolerance there would hide a wrong index.
    assert torch.equal(got_ap, want_ap), "dAfuse split dap"
    assert torch.equal(got_an, want_an), "dAfuse split dan"
    assert_max_rel(got_ls, want_ls, SPLIT_REL, "dAfuse split dls")
    # Slot 0 is ap_0 alone and slot L-1 is nobody's predecessor, so the two ends of
    # the fused column take nothing. Exactly, not to a tolerance.
    assert float(want_ls[0]) == 0.0
    assert not bool(want_an[length - 1].any())


@pytest.mark.parametrize("name", ("ragged33", "ragged40"))
def test_state_cotangent_reaches_the_last_token_through_the_pad_slot(name: str) -> None:
    """The shift cotangent moves over the padded axis, then is sliced to ``T``.

    Under a state-only cotangent, token ``T-1`` reaches ``state`` through slot
    ``n = T mod L`` of the tail chunk and through nothing else: its own now-tap was
    reindexed there, the diagonal is silent without a ``y`` cotangent, and the
    rank-one residue sits at slot ``L-1``. So ``dU`` and ``dB`` at that token are
    exactly the reindexed term, and slicing before the move would leave both
    identically zero while every other token stayed correct.
    """
    chunk, inp = _inputs(name, seed=313)
    assert int(inp.U.shape[2]) % chunk, "the case must be ragged"
    dstate = _cotangents(inp, 317)[1]
    args, kw = inp.args(), inp.kw()
    want = so3ssd_bwd_ref(None, dstate, None, None, *args, chunk, **kw)
    got = chunked_backward_fused(None, dstate, None, None, *args, chunk, **kw)

    assert float(got.dU[:, :, -1].abs().max()) > 0.0, "the reindexed term is missing"
    assert float(got.dB[:, :, -1].abs().max()) > 0.0, "the reindexed term is missing"
    assert_max_rel(got.dU[:, :, -1], want.dU[:, :, -1], BWD_REL, f"tail dU {name}")
    assert_max_rel(got.dB[:, :, -1], want.dB[:, :, -1], BWD_REL, f"tail dB {name}")

    # And no other token pays for it: an off-by-one in the move corrupts the whole
    # axis, not just its last row.
    assert_max_rel(got.dU, want.dU, BWD_REL, f"tail whole dU {name}")
    assert_max_rel(got.dB, want.dB, BWD_REL, f"tail whole dB {name}")
