"""The one-tap factorization against the two-tap one.

:func:`chunked_forward_fused` reindexes the now-tap of token ``s-1`` onto slot
``s`` and folds both taps into one table column. It computes the same operator, so
:func:`chunked_forward` is its ground truth in float64. Three failure modes belong
to the reindex alone: the ragged tail, where a pad slot carries a real token's
forcing; the ``s = 0`` column, which has no predecessor inside the chunk; and the
gradient, which forward parity does not constrain.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from slinoss.ops.so3ssd import as_lanes, chunked_forward, chunked_forward_fused
from tests.conftest import ScanInputs, assert_max_rel, make_inputs, rel_err

# float64 end to end against float64 end to end. The two forms reassociate the
# same sums, so the gap is reordering roundoff. Worst measured over this file:
# 3.1e-15 on cuda at T = 192, L = 64, where cuBLAS reassociates the longest
# reduction. Run with --tolerance-report to see every bound next to what it
# admitted.
PARITY_REL = 1e-13

# float64 autograd against float64 autograd. Worst measured: 1.6e-15.
GRAD_REL = 1e-13

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

# The carry-ins reach slot 0 of chunk 0 and the state pass, neither of which the
# reindex touches, so their gradients must not move at all.
BITWISE: tuple[str, ...] = ("dz0", "db_prev", "du_prev")

# name -> (chunk length, make_inputs keywords). One case per path the reindex can
# take:
#
# - hand, the smallest geometry with a predecessor chunk;
# - grouped, H/G = 2, so two heads read one B/C pair;
# - asym, P = 64 against 3N = 240 over three chunks, the acceptance geometry;
# - square, P == 3N, where a transposed operand still typechecks;
# - strong, decay down to exp(2*ls) = 1e-5, the factor the fused column carries;
# - unit, exp(2*ls) = 1 exactly, where the fused column is ap + an;
# - nocarry, no z0/b_prev/u_prev, so slot 0 of chunk 0 shifts in a zero;
# - ragged40 and ragged33, ragged tails of eight slots and of one.
_CASES: dict[str, tuple[int, dict[str, Any]]] = {
    "hand": (4, {"bsz": 1, "heads": 2, "seqlen": 12, "rows": 16, "lanes": 16}),
    "grouped": (
        8,
        {"bsz": 2, "heads": 4, "groups": 2, "seqlen": 24, "rows": 16, "lanes": 16},
    ),
    "asym": (64, {"bsz": 1, "heads": 2, "seqlen": 192, "rows": 64, "lanes": 80}),
    "square": (
        8,
        {"bsz": 1, "heads": 2, "groups": 2, "seqlen": 32, "rows": 48, "lanes": 16},
    ),
    "strong": (
        8,
        {"bsz": 1, "heads": 2, "seqlen": 24, "rows": 16, "lanes": 16, "ls_bias": 3.0},
    ),
    "unit": (
        8,
        {
            "bsz": 1,
            "heads": 2,
            "seqlen": 24,
            "rows": 16,
            "lanes": 16,
            "ls_bias": -800.0,
        },
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
    "ragged40": (16, {"bsz": 2, "heads": 2, "seqlen": 40, "rows": 16, "lanes": 16}),
    "ragged33": (16, {"bsz": 1, "heads": 2, "seqlen": 33, "rows": 16, "lanes": 32}),
}


def _inputs(
    name: str,
    *,
    device: torch.device | str = "cpu",
    seed: int,
    requires_grad: bool = False,
) -> tuple[int, ScanInputs]:
    chunk, spec = _CASES[name]
    return chunk, make_inputs(
        device=device, seed=seed, requires_grad=requires_grad, **spec
    )


@pytest.mark.parametrize("name", tuple(_CASES))
def test_fused_matches_the_two_tap_form(name: str, device: torch.device) -> None:
    """One call of each form, over every geometry the reindex distinguishes.

    Crossed with the device because the two forms contract different extents and
    the BLAS behind each reduction is per device.
    """
    chunk, inp = _inputs(name, device=device, seed=101)
    want = chunked_forward(*inp.args(), chunk, **inp.kw())
    got = chunked_forward_fused(*inp.args(), chunk, **inp.kw())
    assert_max_rel(got.y, want.y, PARITY_REL, f"fused y {name}")
    assert_max_rel(got.state, want.state, PARITY_REL, f"fused state {name}")


@pytest.mark.parametrize("name", ("ragged40", "ragged33"))
def test_ragged_tail_shifts_the_padded_sequence(
    name: str, device: torch.device
) -> None:
    """Slot ``n = T mod L`` of a ragged tail chunk carries ``u_{T-1}``, ``b_{T-1}``.

    Shifting before the padding is what puts them there; padding a shift built on
    the unpadded sequence puts zero instead. That error never reaches ``y``,
    because a pad column enters rows ``t >= n`` alone and the tail slice discards
    those, so ``state`` is the only witness and it moves by O(1): 3.5e-01 at
    T = 40, 2.6e-01 at T = 33.
    """
    chunk, inp = _inputs(name, device=device, seed=103)
    want = chunked_forward(*inp.args(), chunk, **inp.kw())
    got = chunked_forward_fused(*inp.args(), chunk, **inp.kw())
    tail = got.seqlen % got.length
    assert tail, "the case must be ragged"

    # Slot tail - 1 of the last chunk is the last real token; slot tail is the
    # first pad token, and the reindex fills it from its predecessor.
    assert torch.equal(got.ushift[:, :, -1, tail], got.u[:, :, -1, tail - 1])
    assert torch.equal(got.bshift[:, :, -1, tail], got.b[:, :, -1, tail - 1])
    assert float(got.ushift[:, :, -1, tail].abs().max()) > 0.0
    assert float(got.bshift[:, :, -1, tail].abs().max()) > 0.0

    # The unshifted operands stay zero past the tail. The asymmetry is the point:
    # a pad token contributes through the shifted operands only.
    assert float(got.u[:, :, -1, tail:].abs().max()) == 0.0
    assert float(got.b[:, :, -1, tail:].abs().max()) == 0.0

    assert_max_rel(got.state, want.state, PARITY_REL, f"ragged state {name}")


@pytest.mark.parametrize("name", ("hand", "grouped", "strong", "ragged33"))
def test_analytic_gradients_agree(name: str, device: torch.device) -> None:
    """Every gradient of both forms under one pair of cotangents.

    A forward parity constrains the sum the reindex rewrites, not the derivative of
    each factor in it, so ``dtrans`` and ``dK`` are checked here for the first time:
    the fused column differentiates through ``exp(2*ls)`` as well as through ``an``.
    """
    chunk, ref_inp = _inputs(name, device=device, seed=107, requires_grad=True)
    _, fused_inp = _inputs(name, device=device, seed=107, requires_grad=True)
    want = chunked_forward(*ref_inp.args(), chunk, **ref_inp.kw())
    got = chunked_forward_fused(*fused_inp.args(), chunk, **fused_inp.kw())

    gen = torch.Generator(device=device).manual_seed(109)
    cotangents = tuple(
        torch.randn(t.shape, generator=gen, dtype=torch.float64, device=device)
        for t in (want.y, want.state)
    )
    want_g = torch.autograd.grad((want.y, want.state), ref_inp.leaves(), cotangents)
    got_g = torch.autograd.grad((got.y, got.state), fused_inp.leaves(), cotangents)

    for grad_name, mine, theirs in zip(GRAD_NAMES, got_g, want_g):
        assert_max_rel(mine, theirs, GRAD_REL, f"fused {grad_name} {name}")
        if grad_name in BITWISE:
            assert torch.equal(mine, theirs), f"{grad_name} moved at {name}"


def test_boundary_column_takes_no_cross_chunk_tap() -> None:
    """``Afuse_0 == ap_0``, and every other column shifts within its own chunk.

    The two identities are exact, not approximate: the table is built by one add
    and one scale. Injecting the previous chunk's ``an_{L-1}`` at ``s = 0`` is wrong
    rather than redundant, because that tap sits in the previous chunk's frame and
    its contribution already arrives through ``zstart``. Measured deviation in ``y``
    from the injection: 2.3e-01 here, 8.2e-02 at T = 24, L = 8.
    """
    chunk, inp = _inputs("hand", seed=113)
    got = chunked_forward_fused(*inp.args(), chunk, **inp.kw())
    ap, an = got.table.ap, got.table.an
    step = got.step[..., None, None]
    assert int(got.afuse.shape[2]) > 1, "there must be a previous chunk to draw from"

    assert torch.equal(got.afuse[..., :1, :, :], ap[..., :1, :, :])
    assert torch.equal(
        got.afuse[..., 1:, :, :],
        ap[..., 1:, :, :] + step[..., 1:, :, :] * an[..., :-1, :, :],
    )

    an_cross = torch.cat(
        [torch.zeros_like(an[:, :, :1, -1:]), an[:, :, :-1, -1:]], dim=2
    )
    injected = got.afuse.clone()
    injected[..., :1, :, :] = ap[..., :1, :, :] + step[..., :1, :, :] * an_cross
    bfuse = torch.einsum(
        "bhclij,bhclnj->bhclni", injected, as_lanes(got.bshift)
    ).flatten(-2, -1)
    score = got.crot.flatten(-2, -1) @ bfuse.transpose(-1, -2)
    y_diag = (score * got.dmask) @ got.ushift + got.dnow[..., None] * got.u
    y = (y_diag + got.y_off).flatten(2, 3)[:, :, : got.seqlen]
    assert rel_err(y, got.y) > 1e-3
