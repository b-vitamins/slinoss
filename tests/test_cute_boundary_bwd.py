"""Chunk-boundary and streaming epilogue of the backward.

A specification test of index arithmetic. The kernel adds and copies rows; it
computes nothing, so the authority here is the index expression restated in
float64 torch below, not a reference backward. The tie to
``slinoss/ops/so3ssd/backward.py`` -- that these are the terms the analytic
backward emits, at these tokens -- is an end-to-end comparison that belongs with
the dispatch wiring and lands with it. Until then this file pins the arithmetic
and nothing more.

Every operand is a cotangent and the kernel never reads what one means, only
where it goes. ``randn`` is therefore the strongest input available: any wrong
token, chunk slot, group, or column lands on an unrelated number and shows up.
Structured operands -- a ramp, one value per chunk -- would let a boundary error
land on an equal number and pass. This is not the forbidden pattern of
fabricating a derived intermediate: there is no derivation here to get wrong.

Operands are drawn in float32 and cast, never drawn twice at two dtypes: the
generator consumes a different number of raw words per element at each width, so
the same seed at two dtypes is two different problems.

The activation dtype is not an axis of the sweep. It selects the widen and narrow
pair around one add and interacts with no index, so the sweep runs in float32,
where a single add is exact and the comparison is bitwise, and one bfloat16 case
covers the conversion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from torch import Tensor

from slinoss.ops.so3ssd.cute.bwd.boundary import BoundaryStream, boundary_backward
from slinoss.ops.so3ssd.cute.common import THREADS
from tests.conftest import assert_max_rel, projection_band

pytestmark = [pytest.mark.cuda, pytest.mark.cute]


class Case(NamedTuple):
    """One shape and flag set. Every field has the base value as its default, so
    a case states only the axis it moves."""

    bsz: int = 2
    heads: int = 4
    groups: int = 4
    seqlen: int = 48
    chunk: int = 16
    rows: int = 16
    lanes: int = 16
    splits: int = 1
    du_last: bool = True
    db_last: bool = True
    want_prev: bool = True


WIDE = Case(rows=144, lanes=64)

CASES = [
    # Three chunks, two interior boundaries, every optional term present.
    pytest.param(Case(), id="three-chunks"),
    # One chunk: no boundary at all, so the carry is only a streaming carry-out.
    pytest.param(Case(seqlen=16), id="single-chunk"),
    # A tail of 8 tokens. T-1 is not a chunk boundary and must not be treated as
    # one, and the final chunk still owns the end-of-sequence taps.
    pytest.param(Case(seqlen=40), id="ragged-tail"),
    # Four heads on one group: the b side must run on the group's first head
    # only, or the same carry is added four times.
    pytest.param(Case(groups=1), id="one-group"),
    # Splits crossed with grouping deliberately. At G == H every head owns its
    # group and an unguarded split reduction is unobservable, since the extra
    # writes carry the same values. With four heads on one group it races the
    # boundary add on the row it just wrote, which is the failure the guard
    # exists for. Ragged, so the split pass also has a short final chunk.
    pytest.param(Case(seqlen=40, groups=1, splits=2), id="two-splits-one-group"),
    # Both row widths past the block width, so both stride loops take two steps.
    pytest.param(WIDE, id="rows-and-lanes-past-the-block"),
    # Neither end-of-sequence tap, then one: the two are separate compile-time
    # flags guarding separate adds, and a kernel that keys both off one flag
    # passes with both present and both absent.
    pytest.param(Case(du_last=False, db_last=False), id="no-end-taps"),
    pytest.param(Case(db_last=False), id="u-tap-only"),
    # No carry-out. The launch hands the placeholder the carry itself, so this is
    # also the case where a stray write would corrupt an input.
    pytest.param(Case(want_prev=False), id="no-carry-out"),
]


@dataclass
class Call:
    """One legal call. Mutable, so a rejection test perturbs one operand."""

    carry_u: Tensor
    carry_b: Tensor
    dU: Tensor
    dB: Tensor
    chunk_size: int
    partial_bc: Tensor | None
    du_last: Tensor | None
    db_last: Tensor | None
    want_prev: bool

    def run(self) -> BoundaryStream:
        """Apply the kernel. ``dU`` and ``dB`` are updated in place."""
        return boundary_backward(
            self.carry_u,
            self.carry_b,
            self.dU,
            self.dB,
            self.chunk_size,
            partial_bc=self.partial_bc,
            du_last=self.du_last,
            db_last=self.db_last,
            want_prev=self.want_prev,
        )


class Expected(NamedTuple):
    """The float64 specification. ``dU`` and ``dB`` are the whole tensors."""

    dU: Tensor
    dB: Tensor
    du_prev: Tensor | None
    db_prev: Tensor | None


def _build(case: Case, dtype: torch.dtype = torch.float32, seed: int = 0) -> Call:
    """Draw one operand set at ``case``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = 3 * case.lanes
    chunks = -(-case.seqlen // case.chunk)

    def rnd(*shape: int, low: bool = False) -> Tensor:
        out = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
        return out.to(dtype) if low else out

    return Call(
        carry_u=rnd(case.bsz, case.heads, chunks, case.rows),
        carry_b=rnd(case.bsz, case.groups, chunks, dim),
        dU=rnd(case.bsz, case.heads, case.seqlen, case.rows, low=True),
        dB=rnd(case.bsz, case.groups, case.seqlen, dim, low=True),
        chunk_size=case.chunk,
        partial_bc=(
            rnd(case.bsz, case.groups, case.splits, case.seqlen, dim)
            if case.splits > 1
            else None
        ),
        du_last=rnd(case.bsz, case.heads, case.rows, low=True)
        if case.du_last
        else None,
        db_last=rnd(case.bsz, case.groups, dim, low=True) if case.db_last else None,
        want_prev=case.want_prev,
    )


def _spec(call: Call) -> Expected:
    """The index arithmetic, stated in float64 independently of the kernel.

    Chunk ``c``'s first token sends its cotangent to the token before it, so the
    carry of slot ``c+1`` lands on the last token of chunk ``c``, and slot 0
    leaves the sequence. Whether a chunk boundary is a row of the tile that owns
    it is a kernel concern and does not appear here.
    """
    chunk = call.chunk_size
    seqlen = int(call.dU.shape[2])
    chunks = -(-seqlen // chunk)

    dU = call.dU.double()
    # A split producer writes no dB, so the sum over the splits is the row, not an
    # addend on top of one.
    dB = (
        call.dB.double() if call.partial_bc is None else call.partial_bc.double().sum(2)
    )

    for c in range(chunks - 1):
        token = (c + 1) * chunk - 1
        dU[:, :, token] += call.carry_u.double()[:, :, c + 1]
        dB[:, :, token] += call.carry_b.double()[:, :, c + 1]
    if call.du_last is not None:
        dU[:, :, seqlen - 1] += call.du_last.double()
    if call.db_last is not None:
        dB[:, :, seqlen - 1] += call.db_last.double()

    return Expected(
        dU=dU,
        dB=dB,
        du_prev=call.carry_u.double()[:, :, 0] if call.want_prev else None,
        db_prev=call.carry_b.double()[:, :, 0] if call.want_prev else None,
    )


@pytest.mark.parametrize("case", CASES)
def test_boundary_matches_the_index_specification(case: Case) -> None:
    """Every touched row, against the float64 index expression.

    Bitwise wherever the kernel performs one float32 add or one copy. With a
    split buffer the kernel narrows the sum before adding the boundary carry, so
    a boundary row carries one rounding the specification does not; that case is
    held to a float32 rounding bound instead.
    """
    call = _build(case)
    want = _spec(call)
    got = call.run()
    torch.cuda.synchronize()

    if call.partial_bc is None:
        assert torch.equal(call.dU, want.dU.float())
        assert torch.equal(call.dB, want.dB.float())
    else:
        # Two float32 roundings against the specification's one: 2**-23 each,
        # relative to the largest entry of the tensor.
        assert_max_rel(call.dU, want.dU, 1e-6, "cute-bwd-boundary.dU")
        assert_max_rel(call.dB, want.dB, 1e-6, "cute-bwd-boundary.dB")

    if case.want_prev:
        assert want.du_prev is not None and want.db_prev is not None
        assert got.du_prev is not None and got.db_prev is not None
        assert torch.equal(got.du_prev, want.du_prev.float())
        assert torch.equal(got.db_prev, want.db_prev.float())
    else:
        assert got.du_prev is None and got.db_prev is None


def test_low_precision_gradients_round_trip() -> None:
    """A bfloat16 gradient is widened, added in float32, and narrowed once.

    The bound is that one rounding: bfloat16 carries 8 mantissa bits, so a
    rounded sum is within 2**-9 of itself relative to its own magnitude, and the
    comparison normalizes by the largest entry of the tensor.
    """
    call = _build(Case(), dtype=torch.bfloat16)
    want = _spec(call)
    call.run()
    torch.cuda.synchronize()

    assert call.dU.dtype is torch.bfloat16
    assert_max_rel(call.dU, want.dU, 4e-3, "cute-bwd-boundary-bf16.dU")
    assert_max_rel(call.dB, want.dB, 4e-3, "cute-bwd-boundary-bf16.dB")


def test_only_the_boundary_and_final_rows_are_touched() -> None:
    """Three rows per sequence change and every other row is bitwise unchanged.

    The kernel read-modifies rows of a tensor another kernel owns, so a row it
    touches by accident is a silently wrong gradient rather than a crash. Asserted
    both ways: the three rows must move, or the test would pass on a kernel that
    does nothing.
    """
    case = Case()
    call = _build(case)
    before_u, before_b = call.dU.clone(), call.dB.clone()
    call.run()
    torch.cuda.synchronize()

    touched = {case.chunk - 1, 2 * case.chunk - 1, case.seqlen - 1}
    for token in range(case.seqlen):
        moved_u = not torch.equal(call.dU[:, :, token], before_u[:, :, token])
        moved_b = not torch.equal(call.dB[:, :, token], before_b[:, :, token])
        assert moved_u is (token in touched), f"dU token {token}"
        assert moved_b is (token in touched), f"dB token {token}"


def test_the_carries_are_not_written() -> None:
    """No carry-out requested means the carries are read-only.

    The launch hands the absent carry-out slots the ``c = 0`` slice of the carries
    themselves, so a write through a placeholder would corrupt the input rather
    than a scratch buffer, and would do it after the boundary rows were already
    read.
    """
    call = _build(Case(want_prev=False))
    before_u, before_b = call.carry_u.clone(), call.carry_b.clone()
    call.run()
    torch.cuda.synchronize()

    assert torch.equal(call.carry_u, before_u)
    assert torch.equal(call.carry_b, before_b)


def test_the_wide_case_is_wider_than_the_block() -> None:
    """The stride loops take more than one step only above the block width.

    Asserted rather than assumed: at or below it the wide case is the same trip
    count as every other case and that half of the sweep is vacuous.
    """
    assert WIDE.rows > THREADS
    assert 3 * WIDE.lanes > THREADS


def test_writes_a_band_of_the_fused_projection_gradient() -> None:
    """``dB`` ships pitched, and the kernel writes the band rather than a copy.

    ``B`` is a column band of one projection GEMM's output, so its gradient is a
    band of that GEMM's cotangent. Staging a contiguous ``dB`` and scattering it
    back afterwards is the copy the layout contract exists to refuse. Nothing about
    the arithmetic changes, so the two layouts must agree bit for bit rather than
    within a tolerance. The split case, because that is the one where the kernel
    writes every row of ``dB`` and not only the two boundary rows.
    """
    want = _build(SPLIT)
    got = _build(SPLIT)
    got.dB = projection_band(got.dB)
    want.run()
    got.run()
    torch.cuda.synchronize()
    assert torch.equal(got.dB, want.dB)
    assert torch.equal(got.dU, want.dU)


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------

OPERANDS = ["carry_u", "carry_b", "dU", "dB", "partial_bc", "du_last", "db_last"]
"""Every tensor the wrapper takes. The split case supplies all seven."""

CONTIGUOUS = [name for name in OPERANDS if name != "dB"]
"""The operands the wrapper requires contiguous. ``dB`` is pitched, so a wider
pitch is legal on it and it takes its own rejection below."""

SPLIT = Case(splits=2)


def _strided(t: Tensor) -> Tensor:
    """The same shape and dtype, a pitch twice the row width.

    Legal on a pitched operand and refused on a contiguous one, which is the whole
    difference between the two rules.
    """
    wide = torch.empty(
        *t.shape[:-1], 2 * int(t.shape[-1]), dtype=t.dtype, device=t.device
    )
    return wide[..., : int(t.shape[-1])]


@pytest.mark.parametrize("name", OPERANDS)
def test_rejects_a_host_operand(name: str) -> None:
    """A host tensor is refused before the launch, not during it.

    Launching against a host pointer raises inside CUDA and leaves the context
    unusable for the rest of the process, so every later launch fails too.
    """
    call = _build(SPLIT)
    setattr(call, name, getattr(call, name).cpu())
    with pytest.raises(ValueError, match="CUDA device"):
        call.run()


@pytest.mark.parametrize("name", CONTIGUOUS)
def test_rejects_a_non_contiguous_operand(name: str) -> None:
    """Nothing is repacked, so a strided operand is refused rather than fixed.

    Every index in the kernel is computed from the declared shape, so a strided
    operand is read at the wrong stride and returns a wrong gradient with no
    error of its own.
    """
    call = _build(SPLIT)
    setattr(call, name, _strided(getattr(call, name)))
    with pytest.raises(ValueError, match="contiguous"):
        call.run()


def test_rejects_a_gap_inside_a_row_of_db() -> None:
    """An arbitrary pitch is legal on ``dB``; a gap inside one row is not.

    A thread writes the three components of its 3-vector as adjacent elements. The
    remaining pitched rejections belong to the shared rule and are covered against
    it; this one says ``dB`` reaches that rule at all rather than the contiguous
    one.
    """
    call = _build(SPLIT)
    call.dB = call.dB[..., ::2]
    with pytest.raises(ValueError, match="dB must have unit stride"):
        call.run()


@pytest.mark.parametrize("name", ["carry_u", "carry_b", "partial_bc"])
def test_rejects_a_low_precision_carry(name: str) -> None:
    """I4 pins every carry, the split partials included.

    A carry is a sum over a chunk of a decayed cotangent; it is accumulated in
    float32 by the kernel that produces it and stays float32 across the boundary.
    """
    call = _build(SPLIT)
    setattr(call, name, getattr(call, name).bfloat16())
    with pytest.raises(ValueError, match=f"{name} must be float32"):
        call.run()


@pytest.mark.parametrize("name", ["dU", "dB", "du_last", "db_last"])
def test_rejects_a_gradient_dtype_with_no_kernel_path(name: str) -> None:
    """float64 has no kernel path, and a gradient in it is a caller bug."""
    call = _build(SPLIT)
    setattr(call, name, getattr(call, name).double())
    with pytest.raises(TypeError, match="kernel dtypes"):
        call.run()


@pytest.mark.parametrize("name", ["dB", "du_last", "db_last"])
def test_rejects_mixed_gradient_dtypes(name: str) -> None:
    """One activation dtype per call.

    The kernel widens and narrows through one type. Promoting an operand on the
    host would be the staging copy the kernels exist to avoid, and narrowing one
    would lose bits the caller still holds.
    """
    call = _build(SPLIT)
    setattr(call, name, getattr(call, name).bfloat16())
    with pytest.raises(TypeError, match="one dtype per call"):
        call.run()


SHAPES = [
    pytest.param("dU", lambda t: t.flatten(2, 3), "dU must be", id="du-rank"),
    pytest.param(
        "carry_u",
        lambda t: t[:, :, :-1].contiguous(),
        "carry_u must be",
        id="carry-u-chunk-count",
    ),
    pytest.param(
        "carry_u",
        lambda t: t[..., :-1].contiguous(),
        "carry_u must be",
        id="carry-u-rows",
    ),
    pytest.param(
        "carry_b", lambda t: t.flatten(2, 3), "carry_b must be", id="carry-b-rank"
    ),
    pytest.param(
        "carry_b", lambda t: t[:-1].contiguous(), "carry_b must be", id="carry-b-batch"
    ),
    pytest.param(
        "carry_b", lambda t: t[:, :3].contiguous(), "does not divide H", id="carry-b-g"
    ),
    # An empty group axis divides nothing and would be a division by zero rather
    # than a rejection.
    pytest.param(
        "carry_b",
        lambda t: t[:, :0].contiguous(),
        "does not divide H",
        id="carry-b-no-groups",
    ),
    # Narrowed by a whole alignment quantum, not by one 3-vector: ``dB`` is pitched,
    # and the pitch of a contiguous one is its row width, so a width that is not a
    # multiple of the 16-byte quantum is refused by the layout rule before the
    # shape rule is reached.
    pytest.param("dB", lambda t: t[..., :-16].contiguous(), "dB must be", id="db-dim"),
    pytest.param(
        "partial_bc",
        lambda t: t.flatten(1, 2),
        "partial_bc must be",
        id="partial-rank",
    ),
    pytest.param(
        "partial_bc",
        lambda t: t[:, :, :1].contiguous(),
        "at least two splits",
        id="partial-one-split",
    ),
    pytest.param(
        "partial_bc",
        lambda t: t[..., :-3].contiguous(),
        "partial_bc must be",
        id="partial-dim",
    ),
    pytest.param(
        "du_last",
        lambda t: t[..., :-1].contiguous(),
        "du_last must be",
        id="du-last-rows",
    ),
    pytest.param(
        "db_last",
        lambda t: t[..., :-1].contiguous(),
        "db_last must be",
        id="db-last-dim",
    ),
]
"""One entry per shape relation the wrapper asserts. Sliced then made contiguous,
so each reaches the shape check rather than the layout check."""


@pytest.mark.parametrize(("name", "bend", "message"), SHAPES)
def test_rejects_an_inconsistent_shape(
    name: str, bend: Callable[[Tensor], Tensor], message: str
) -> None:
    """Nothing broadcasts and nothing is inferred from one operand alone.

    ``dU`` fixes ``(B,H,T,P)`` and ``carry_b`` fixes ``G`` and ``3N``, so every
    other operand is checked against them. Each relation is listed once, because
    a mismatch in any of them sends some block past the end of a tensor.
    """
    call = _build(SPLIT)
    setattr(call, name, bend(getattr(call, name)))
    with pytest.raises(ValueError, match=message):
        call.run()


def test_rejects_an_empty_sequence() -> None:
    """A launch over zero tokens has no boundary and no final row.

    The grid would be empty and the call would silently do nothing, which is not
    distinguishable from a lost gradient.
    """
    call = _build(Case())
    call.dU = call.dU[:, :, :0].contiguous()
    with pytest.raises(ValueError, match="at least one token"):
        call.run()


def test_rejects_a_non_positive_chunk_size() -> None:
    """``C`` is derived from the chunk length, so zero is not a shape."""
    call = _build(Case())
    call.chunk_size = 0
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        call.run()


def test_rejects_a_chunk_size_the_carries_were_not_produced_at() -> None:
    """The chunk length is checked against the carry, not trusted.

    Every boundary token is ``(c+1)L-1``. A chunk length that disagrees with the
    one the producing kernels ran at puts every one of them on the wrong token,
    which no shape makes illegal on its own.
    """
    call = _build(Case())
    call.chunk_size = 2 * call.chunk_size
    with pytest.raises(ValueError, match="carry_u must be"):
        call.run()
