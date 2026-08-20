"""Cotangent contract and end-to-end parity of the backward driver.

Two things live here. The cotangent contract: the driver checks all four
cotangents before it launches anything, so a caller's mistake names the cotangent
rather than surfacing from whichever stage happens to read it. Two of those rules
cannot be checked anywhere else, because the dtype agreement spans two stages and
the all-absent call reaches no stage at all.

Then the composition. Each of the five launches has its own test module driving
it from synthetic inputs; none of them drives stage ``N`` from stage ``N-1``'s
real output, and none drives any stage from a chunk boundary the forward produced.
That seam is what the parity test covers, and it is the only place two buffers
consumed in place -- the chunk-start state over the increment, and the increment
cotangent over the chunk-start cotangent -- are shown to survive their passes. The
driver reaches that boundary two ways, rebuilt here and held by the caller, and
the last test in this module is what holds the two to the same bytes.

Authority. The float64 autograd of the operator is the ground truth for the
analytic reference, established in ``tests/test_reference_grad.py``. The CuTe
kernels have no float64 path, so the chain here is the second link: the reference
analytic backward, handed the float32 widening of the same values the kernels get,
against the kernels. Widening rather than passing the narrowed operands to both
makes the oracle the more accurate of the pair, so a measured error is the
kernels' own and not a sum of two roundings.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable
from typing import NamedTuple

from slinoss.ops.so3ssd.backward import so3ssd_bwd_ref
from slinoss.ops.so3ssd.cute.backward import so3ssd_bwd_cute
from slinoss.ops.so3ssd.cute.forward import so3ssd_fwd_cute
from slinoss.ops.so3ssd.cute.guard import check_cotangents
from tests.conftest import assert_max_rel, make_inputs, projection_band

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# (B, H, G, T, P, 3N), as check_shapes returns it. G divides H and 3N is a multiple
# of 48, so the record is a legal one and every rejection below is the mutation's
# own fault.
SHAPE = (2, 4, 2, 8, 16, 48)

Cotangents = dict[str, torch.Tensor | None]


def _ok() -> Cotangents:
    """All four cotangents, at the shapes ``SHAPE`` implies."""
    bsz, heads, groups, seqlen, rows, dim = SHAPE
    return {
        "dy": torch.empty(bsz, heads, seqlen, rows, dtype=torch.bfloat16),
        "dstate": torch.empty(bsz, heads, rows, dim, dtype=torch.float32),
        "db_last": torch.empty(bsz, groups, dim, dtype=torch.bfloat16),
        "du_last": torch.empty(bsz, heads, rows, dtype=torch.bfloat16),
    }


def _check(args: Cotangents) -> None:
    check_cotangents(
        args["dy"], args["dstate"], args["db_last"], args["du_last"], SHAPE
    )


# One case per expected shape rather than one per branch: the four share one loop,
# but the shape each is compared against is hand-written, and a wrong entry in that
# table is only reachable through its own cotangent.
WRONG_SHAPE = [
    pytest.param(
        "dy", lambda t: t[:, :, 0].contiguous(), r"dy must be \(2, 4, 8, 16\)"
    ),
    pytest.param("dstate", lambda t: t[..., :16].contiguous(), "dstate must be"),
    pytest.param("db_last", lambda t: t[:, :1].contiguous(), "db_last must be"),
    pytest.param("du_last", lambda t: t[:, :2].contiguous(), "du_last must be"),
]


@pytest.mark.parametrize(("name", "mutate", "match"), WRONG_SHAPE)
def test_rejects_a_cotangent_of_the_wrong_shape(
    name: str,
    mutate: Callable[[torch.Tensor], torch.Tensor],
    match: str,
) -> None:
    """A cotangent that does not match the forward output it belongs to."""
    args = _ok()
    current = args[name]
    assert current is not None
    args[name] = mutate(current)
    with pytest.raises(ValueError, match=match):
        _check(args)


def test_rejects_activation_cotangents_that_disagree_about_dtype() -> None:
    """The rule this check exists for.

    ``dy`` is read by the chunk-start stage and ``du_last`` by the boundary stage,
    so neither kernel sees both and a mixed pair would launch twice at two dtypes
    and return gradients in two dtypes with no error.
    """
    args = _ok()
    du_last = args["du_last"]
    assert du_last is not None
    args["du_last"] = du_last.half()
    with pytest.raises(TypeError, match="one dtype per call"):
        _check(args)


def test_rejects_a_call_with_no_cotangent() -> None:
    """Every cotangent absent asks the driver to run to produce zeros."""
    with pytest.raises(ValueError, match="at least one cotangent"):
        check_cotangents(None, None, None, None, SHAPE)


def test_accepts_every_cotangent() -> None:
    """The baseline the rejections mutate.

    Without it every rejection above would also pass against a check that refuses
    everything.
    """
    _check(_ok())


def test_accepts_a_state_cotangent_alone() -> None:
    """A state-only backward carries no activation cotangent.

    The dtype group is then empty, and a group of one dtype has nothing to agree on.
    """
    args = _ok()
    args["dy"] = None
    args["db_last"] = None
    args["du_last"] = None
    _check(args)


# ---------------------------------------------------------------------------
# End-to-end parity
# ---------------------------------------------------------------------------

ROWS = 16
LANES = 16
ACT = torch.bfloat16

# Bounds on the maximum relative error against the reference. Per gradient rather
# than one number, because they do not carry the same number of roundings: dU
# leaves a narrowing store, while dtrans and dK leave a float32 reduction over a
# chunk. Each is about twice the worst error measured over the cases below, which
# the comment beside it records. Run with --tolerance-report to print them again.
BOUNDS = {
    "dU": 1.1e-2,  # 5.413e-03
    "dtrans": 2.0e-2,  # 9.421e-03
    "dK": 7.0e-3,  # 3.184e-03
    "dB": 1.0e-2,  # 4.978e-03
    "dC": 1.0e-2,  # 4.634e-03
    "dz0": 9.0e-3,  # 4.409e-03
    "db_prev": 1.2e-2,  # 5.593e-03
    "du_prev": 1.1e-2,  # 5.424e-03
}


class Case(NamedTuple):
    """One structural branch of the driver.

    Attributes:
        label: Test id.
        bsz: ``B``.
        heads: ``H``.
        groups: ``G``. Equal to ``heads`` is the ungrouped case.
        seqlen: ``T``. Not a multiple of ``chunk`` is the ragged tail.
        chunk: ``L``.
        pitched: Hand ``B`` and ``C`` as bands of a wider buffer.
        streaming: Supply ``b_prev`` and ``u_prev``.
        with_state: Supply ``z0``.
        cotangents: Which of the four the call carries.
    """

    label: str
    bsz: int
    heads: int
    groups: int
    seqlen: int
    chunk: int
    pitched: bool
    streaming: bool
    with_state: bool
    cotangents: tuple[str, ...]


ALL_FOUR = ("dy", "dstate", "db_last", "du_last")

# One row per branch the driver takes, not one per shape that happens to be legal.
# Each row is the only case that reaches something: the tail predicate, the group
# fan-in, the single-chunk path where the inter-chunk recurrence has no step to
# take, the absent-dy path that skips a launch, and the training path, which is
# the one production runs.
CASES = [
    Case("exact", 2, 4, 4, 32, 16, False, True, True, ALL_FOUR),
    Case("pitched", 2, 4, 4, 32, 16, True, True, True, ALL_FOUR),
    Case("ragged", 2, 4, 4, 40, 16, True, True, True, ALL_FOUR),
    Case("grouped", 2, 4, 2, 32, 16, True, True, True, ALL_FOUR),
    Case("single_chunk", 1, 2, 1, 16, 16, True, True, True, ALL_FOUR),
    Case("state_only", 2, 4, 2, 32, 16, True, True, True, ("dstate",)),
    Case("training", 2, 4, 2, 32, 16, True, False, False, ("dy",)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.label)
def test_matches_the_reference_backward(case: Case) -> None:
    """Every gradient of the seven-launch driver against the analytic reference.

    The reference is handed the float32 widening of the same narrowed values, so
    it is the more accurate of the pair and the error is the kernels' own.
    """
    torch.manual_seed(0)
    inp = make_inputs(
        bsz=case.bsz,
        heads=case.heads,
        groups=case.groups,
        seqlen=case.seqlen,
        rows=ROWS,
        lanes=LANES,
        dtype=torch.float32,
        device="cuda",
        with_state=case.with_state,
        streaming=case.streaming,
    )
    dim = 3 * LANES
    opts = {"dtype": ACT, "device": "cuda"}
    U = inp.U.to(ACT)
    B = inp.B.to(ACT)
    C = inp.C.to(ACT)
    u_prev = None if inp.u_prev is None else inp.u_prev.to(ACT)
    b_prev = None if inp.b_prev is None else inp.b_prev.to(ACT)

    want = case.cotangents
    dy = (
        torch.randn(case.bsz, case.heads, case.seqlen, ROWS, **opts)
        if "dy" in want
        else None
    )
    dstate = (
        torch.randn(case.bsz, case.heads, ROWS, dim, device="cuda")
        if "dstate" in want
        else None
    )
    db_last = (
        torch.randn(case.bsz, case.groups, dim, **opts) if "db_last" in want else None
    )
    du_last = (
        torch.randn(case.bsz, case.heads, ROWS, **opts) if "du_last" in want else None
    )

    def wide(t: torch.Tensor | None) -> torch.Tensor | None:
        return None if t is None else t.float()

    ref = so3ssd_bwd_ref(
        wide(dy),
        dstate,
        wide(db_last),
        wide(du_last),
        U.float(),
        inp.trans,
        inp.K,
        B.float(),
        C.float(),
        case.chunk,
        z0=inp.z0,
        b_prev=wide(b_prev),
        u_prev=wide(u_prev),
    )
    got = so3ssd_bwd_cute(
        dy,
        dstate,
        db_last,
        du_last,
        U,
        inp.trans,
        inp.K,
        projection_band(B) if case.pitched else B,
        projection_band(C) if case.pitched else C,
        case.chunk,
        z0=inp.z0,
        b_prev=b_prev,
        u_prev=u_prev,
    )

    for name, bound in BOUNDS.items():
        mine = getattr(got, name)
        theirs = getattr(ref, name)
        if theirs is None:
            assert mine is None, f"{name} present where the reference has none"
            continue
        assert mine is not None, f"{name} absent where the reference has one"
        assert_max_rel(mine, theirs, bound, f"backward/{case.label}/{name}")


WIDE_HEADS = 18
WIDE_ROWS = 64
WIDE_LANES = 80


@pytest.mark.parametrize("groups", [1, WIDE_HEADS], ids=["folded", "ungrouped"])
def test_holds_a_wide_state_at_the_full_chunk(groups: int) -> None:
    """The driver launches at ``3N = 240``, ``P = 64``, ``L = 64``.

    Every case above runs at ``L = 16`` and ``3N = 48``, where the shared-memory
    live set is at its narrowest. That set grows with ``L``, ``P``, ``3N``, and the
    number of heads a group feeds, and it is wider in the backward than in the
    forward, so a shape the forward accepts is not a shape the backward can hold.
    Both folds are here because one launch sums over the fold and its footprint is
    the one that moves with it.

    Finiteness rather than parity: parity is the cases above, and what this shape
    reaches is the arena. Two chunks, because nothing here depends on the sequence
    length.
    """
    torch.manual_seed(0)
    seqlen = 128
    inp = make_inputs(
        bsz=1,
        heads=WIDE_HEADS,
        groups=groups,
        seqlen=seqlen,
        rows=WIDE_ROWS,
        lanes=WIDE_LANES,
        dtype=torch.float32,
        device="cuda",
        with_state=False,
        streaming=False,
    )
    dy = torch.randn(1, WIDE_HEADS, seqlen, WIDE_ROWS, dtype=ACT, device="cuda")
    got = so3ssd_bwd_cute(
        dy,
        None,
        None,
        None,
        inp.U.to(ACT),
        inp.trans,
        inp.K,
        inp.B.to(ACT),
        inp.C.to(ACT),
        64,
    )
    for name in ("dU", "dtrans", "dK", "dB", "dC"):
        grad = getattr(got, name)
        assert grad is not None, name
        assert bool(grad.isfinite().all()), name


# The dtype of a gradient is not a tolerance question: autograd raises when a
# gradient's dtype does not match its leaf, so a stage that stores float32 where
# the leaf is bfloat16 fails at the accumulation and never reaches a comparison.
# The two streaming carries are the ones at risk, because the kernel that writes
# them writes them from float32 carry buffers.
def test_every_gradient_carries_the_dtype_of_its_leaf() -> None:
    """Each gradient in the dtype autograd will accumulate it into."""
    torch.manual_seed(0)
    inp = make_inputs(
        bsz=2,
        heads=4,
        groups=2,
        seqlen=32,
        rows=ROWS,
        lanes=LANES,
        dtype=torch.float32,
        device="cuda",
        with_state=True,
        streaming=True,
    )
    U = inp.U.to(ACT)
    B = inp.B.to(ACT)
    C = inp.C.to(ACT)
    assert inp.b_prev is not None
    assert inp.u_prev is not None
    got = so3ssd_bwd_cute(
        torch.randn(2, 4, 32, ROWS, dtype=ACT, device="cuda"),
        None,
        None,
        None,
        U,
        inp.trans,
        inp.K,
        projection_band(B),
        projection_band(C),
        16,
        z0=inp.z0,
        b_prev=inp.b_prev.to(ACT),
        u_prev=inp.u_prev.to(ACT),
    )
    leaves = {
        "dU": ACT,
        "dB": ACT,
        "dC": ACT,
        "db_prev": ACT,
        "du_prev": ACT,
        "dtrans": torch.float32,
        "dK": torch.float32,
        "dz0": torch.float32,
    }
    for name, dtype in leaves.items():
        grad = getattr(got, name)
        assert grad is not None, name
        assert grad.dtype is dtype, f"{name} is {grad.dtype}, leaf is {dtype}"


def _inputs() -> tuple[torch.Tensor, ...]:
    """One legal call's operands, narrowed, at the shape the test below uses."""
    torch.manual_seed(0)
    inp = make_inputs(
        bsz=2,
        heads=4,
        groups=2,
        seqlen=32,
        rows=ROWS,
        lanes=LANES,
        dtype=torch.float32,
        device="cuda",
        with_state=False,
        streaming=False,
    )
    dy = torch.randn(2, 4, 32, ROWS, dtype=ACT, device="cuda")
    return dy, inp.U.to(ACT), inp.trans, inp.K, inp.B.to(ACT), inp.C.to(ACT)


# The reference's own destination tests live in tests/test_backward.py. This is a
# different failure: there the store is a copy_ the host issues, here it is the
# address a kernel's store instruction targets, and a kernel that allocated its own
# buffer and copied would pass the reference's test and still write twice.
def test_the_vector_destinations_are_the_buffers_the_kernels_store_to() -> None:
    """``dB`` and ``dC`` come back as the caller's own objects, written in full."""
    dy, U, trans, K, B, C = _inputs()
    dB = torch.full_like(B, float("nan"))
    dC = torch.full_like(C, float("nan"))
    got = so3ssd_bwd_cute(dy, None, None, None, U, trans, K, B, C, 16, dB=dB, dC=dC)
    assert got.dB is dB, "dB is not the buffer that was passed"
    assert got.dC is dC, "dC is not the buffer that was passed"
    # Every element overwritten: a partial store leaves a NaN, which no arithmetic
    # below would clear.
    assert not dB.isnan().any(), "dB kept a NaN, so some element was not stored"
    assert not dC.isnan().any(), "dC kept a NaN, so some element was not stored"


def test_the_forcing_seed_reaches_the_epilogue() -> None:
    """``dU_init`` is forwarded to the kernel that adds it.

    The arithmetic is not what is checked here. That the epilogue adds the seed once,
    at the right token, and ahead of the narrowing is established against a float64
    oracle in ``tests/test_cute_chunk_input_bwd.py``. What only the driver can drop is
    the keyword itself, and a driver that dropped it would pass every test in that
    module. So this is the weakest claim that catches it: two calls on identical
    operands, one seeded, differing by the seed.
    """
    dy, U, trans, K, B, C = _inputs()
    plain = so3ssd_bwd_cute(dy, None, None, None, U, trans, K, B, C, 16)
    # Scaled to dU rather than to one: a seed far below dU would be lost in dU's own
    # rounding and the test would pass against a driver that ignored it.
    seed = (torch.randn_like(U, dtype=torch.float32) * plain.dU.float().abs().max()).to(
        ACT
    )
    seeded = so3ssd_bwd_cute(dy, None, None, None, U, trans, K, B, C, 16, dU_init=seed)
    # Against the sum rather than the difference. Differencing two narrowed values
    # cancels dU away and leaves the ratio of two roundings.
    assert_max_rel(
        seeded.dU, plain.dU.float() + seed.float(), BOUNDS["dU"], "backward/seed/dU"
    )


def test_a_held_chunk_boundary_is_read_only_and_gives_the_rebuilt_gradients() -> None:
    """A supplied ``prologue`` is read, never written, and changes no gradient.

    While the backward rebuilt the boundary itself, a kernel that wrote over it was
    invisible: every call got a buffer nothing else would ever read. Held across the
    step, the same write corrupts the next backward, the forward's ``state``, or
    both, and no other test here reads those buffers after a backward has run.

    Bit-exact rather than toleranced. The two paths launch the same kernels over the
    same bytes, so a difference of any size is a defect in which tensor was read and
    not a rounding.
    """
    dy, U, trans, K, B, C = _inputs()
    out = so3ssd_fwd_cute(U, trans, K, B, C, 16)
    assert out.prologue is not None
    before = tuple(t.clone() for t in out.prologue)
    rebuilt = so3ssd_bwd_cute(dy, None, None, None, U, trans, K, B, C, 16)
    held = so3ssd_bwd_cute(
        dy, None, None, None, U, trans, K, B, C, 16, prologue=out.prologue
    )
    for name in BOUNDS:
        mine, theirs = getattr(held, name), getattr(rebuilt, name)
        if theirs is None:
            assert mine is None, f"{name} present where the rebuilt path has none"
            continue
        assert mine is not None, f"{name} absent where the rebuilt path has one"
        assert torch.equal(mine, theirs), f"{name} differs from the rebuilt path"
    for name, kept, was in zip(out.prologue._fields, out.prologue, before, strict=True):
        assert torch.equal(kept, was), f"the backward wrote over {name}"
