"""Inter-chunk recurrence against the float64 reference.

The kernel's three inputs are pipeline intermediates, so they are produced by a
float32 reference run rather than fabricated: a ``randn`` increment and a
``randn`` rotation would not test the frame change, which is the only nontrivial
thing here. Each one is read straight off the reference's own record, so nothing
is reconstructed from it and the two paths cannot disagree about what the operands
mean.

The authority is a full float64 ``chunked_forward``, not a float64 replay of the
float32 intermediates. That way the chunk endpoint, the rotation matrix, and the
recurrence itself are all inside the comparison.

Operands are built in float32 and upcast, never built twice at two dtypes: the
generator consumes a different number of raw words per element at each width, so
the same seed at two dtypes is two different problems.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE
from slinoss.ops.so3ssd import ChunkedForward, chunked_forward
from slinoss.ops.so3ssd.cute.common import THREADS
from slinoss.ops.so3ssd.cute.fwd.state_passing import state_passing_forward
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it a 64-token chunk reaches exp(2*lp) near 1e-54, the
# chunk decay is zero to float32, and the recurrence under test is the identity on
# the increment. The bias keeps every chunk transition normal and significant.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, with_state).
#
# This kernel never sees ``L`` or ``T``: its extents are ``C``, ``P``, ``N`` and
# whether an initial state was supplied. So the sweep varies only those. A ragged
# tail and a longer chunk reach the same instructions and are covered where they
# are actually a code path, in the chunk kernels.
#
# Three cases, one per distinct path: the recurrence with an initial state at the
# smallest tile count, the zero-start compile-time variant at eight tiles, and a
# single chunk, which is the shortest trip count the dynamic loop takes.
#
# A single chunk carries an initial state, and a zero initial state comes with
# several chunks. The two together would be vacuous: with one chunk and no state,
# `zstart` is the zero the kernel just wrote and the transition is never applied,
# so the recorded relative error is exactly zero whatever the matrix does.
SHAPES = [
    pytest.param(2, 3, 256, 64, 16, 16, True, id="four-chunks-two-tiles"),
    pytest.param(2, 2, 512, 64, 32, 32, False, id="eight-tiles-zero-start"),
    pytest.param(1, 1, 64, 64, 16, 16, True, id="single-chunk"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    with_state: bool,
) -> ScanInputs:
    """One float32 operand set. Streaming is off: this kernel sees neither tap."""
    return make_inputs(
        bsz=bsz,
        heads=heads,
        seqlen=seqlen,
        rows=rows,
        lanes=lanes,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
        with_state=with_state,
        streaming=False,
    )


def _upcast(inp: ScanInputs) -> ScanInputs:
    """The same operands in float64. Exact, so both paths see the same problem."""
    return ScanInputs(
        U=inp.U.double(),
        trans=inp.trans.double(),
        K=inp.K.double(),
        B=inp.B.double(),
        C=inp.C.double(),
        z0=None if inp.z0 is None else inp.z0.double(),
        b_prev=None if inp.b_prev is None else inp.b_prev.double(),
        u_prev=None if inp.u_prev is None else inp.u_prev.double(),
    )


def _pack(ref: ChunkedForward) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the kernel's three inputs out of a reference run.

    Returns ``(inc_local, cquat, cscale)``: the chunk increment in the
    chunk-local frame, the unit quaternion prefix at the end of each chunk, and
    the chunk decay ``exp(2*lp_{L-1})``.
    """
    return (
        ref.inc_local.contiguous(),
        ref.qprefix[..., -1, :].contiguous(),
        torch.exp(2.0 * ref.lprefix[..., -1]).contiguous(),
    )


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "with_state"), SHAPES
)
def test_state_passing_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    with_state: bool,
) -> None:
    """``zstart`` and the final state match a float64 forward end to end."""
    inp32 = _make(bsz, heads, seqlen, rows, lanes, with_state)
    inp64 = _upcast(inp32)
    inc, cquat, cscale = _pack(chunked_forward(*inp32.args(), chunk, **inp32.kw()))
    ref = chunked_forward(*inp64.args(), chunk, **inp64.kw())

    out = state_passing_forward(inc, cquat, cscale, inp32.z0)
    torch.cuda.synchronize()

    assert out.zstart.data_ptr() == inc.data_ptr(), "zstart must alias inc"
    tag = f"cute-state-passing[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}]"
    # Every chunk transition has norm at most one (I1), so the recurrence
    # neither amplifies nor accumulates: the bound is float32 rounding of the
    # increment and of the chunk endpoint, not a per-chunk growth term.
    assert_max_rel(out.zstart, ref.zstart.flatten(-2, -1), 4e-6, f"{tag}.zstart")
    assert_max_rel(out.state, ref.state.flatten(-2, -1), 4e-6, f"{tag}.state")
    # A comparison that only ever sees zeros passes whatever the transition does.
    # Every parameter must move at least one state through the matrix.
    assert torch.count_nonzero(out.zstart) > 0


def test_zero_start_is_not_read() -> None:
    """The zero-start variant ignores whatever stands in for the initial state.

    The kernel takes one signature and the no-state launch hands it the output
    buffer, which is uninitialized. Running the same increments twice must give
    the same answer, or that buffer is being read.
    """
    inp = _make(2, 2, 192, 16, 16, False)
    inc, cquat, cscale = _pack(chunked_forward(*inp.args(), 64, **inp.kw()))
    first = state_passing_forward(inc.clone(), cquat, cscale)
    second = state_passing_forward(inc.clone(), cquat, cscale)
    torch.cuda.synchronize()
    assert torch.equal(first.zstart, second.zstart)
    assert torch.equal(first.state, second.state)
    assert torch.count_nonzero(first.zstart[:, :, 0]) == 0


def _inputs(
    chunks: int = 3, rows: int = 16, lanes: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A legally shaped operand triple, for the rejection tests to perturb."""
    opts = {"device": "cuda", "dtype": torch.float32}
    return (
        torch.empty(2, 2, chunks, rows, 3 * lanes, **opts),
        torch.empty(2, 2, chunks, 4, **opts),
        torch.empty(2, 2, chunks, **opts),
    )


@pytest.mark.parametrize("operand", ["inc", "cquat", "cscale", "z0"])
def test_rejects_a_low_precision_operand(operand: str) -> None:
    """I4 pins the state, the rotation, and the decay alike."""
    inc, cquat, cscale = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    args = {"inc": inc, "cquat": cquat, "cscale": cscale, "z0": z0}
    args[operand] = args[operand].bfloat16()
    with pytest.raises(ValueError, match=f"{operand} must be float32"):
        state_passing_forward(args["inc"], args["cquat"], args["cscale"], args["z0"])


def test_rejects_wrong_rank() -> None:
    """The increment is chunked and row-major; a flat one is a caller bug."""
    inc, cquat, cscale = _inputs()
    with pytest.raises(ValueError, match="expected"):
        state_passing_forward(inc.flatten(3, 4), cquat, cscale)


@pytest.mark.parametrize("operand", ["cquat", "cscale", "z0"])
def test_rejects_a_mismatched_extent(operand: str) -> None:
    """One rotation and one decay per chunk, one initial state per ``(P,3N)``.

    Nothing broadcasts. Each operand is sliced along the axis it shares with
    ``inc``, and each carries a different rank, so one check cannot cover all
    three. Sliced then made contiguous, so this reaches the shape check rather
    than the layout check.
    """
    inc, cquat, cscale = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    args = {
        "cquat": cquat[:, :, :-1],
        "cscale": cscale[:, :, :-1],
        "z0": z0[:, :, :-1],
    }
    full = {"inc": inc, "cquat": cquat, "cscale": cscale, "z0": z0}
    full[operand] = args[operand].contiguous()
    with pytest.raises(ValueError, match=f"{operand} shape"):
        state_passing_forward(full["inc"], full["cquat"], full["cscale"], full["z0"])


def test_rejects_unlaunchable_shape() -> None:
    """The launch is exact, so an illegal ``(P, N)`` pair is refused, not padded.

    ``P`` a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE`` makes
    ``P*N`` a multiple of the block width. A shape that violates that reaches no
    kernel: the fix is the shape, not a tail predicate.
    """
    inc, cquat, cscale = _inputs(rows=3)
    with pytest.raises(ValueError, match="multiple of"):
        state_passing_forward(inc, cquat, cscale)


@pytest.mark.parametrize("operand", ["inc", "cquat", "cscale", "z0"])
def test_rejects_a_non_contiguous_operand(operand: str) -> None:
    """No repacking is done, so a strided operand is refused rather than fixed.

    The cases fail differently without the check: ``inc`` and ``z0`` reach an
    internal ``view`` and raise ``RuntimeError``, while ``cquat`` and ``cscale``
    are not reshaped at all and would be read at the wrong stride and return a
    wrong answer with no error.
    """
    inc, cquat, cscale = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    opts = {"device": "cuda", "dtype": torch.float32}
    wide = {
        "inc": torch.empty(2, 2, 3, 16, 96, **opts),
        "cquat": torch.empty(2, 2, 3, 8, **opts),
        "cscale": torch.empty(2, 2, 6, **opts),
        "z0": torch.empty(2, 2, 16, 96, **opts),
    }[operand]
    keep = {"inc": 48, "cquat": 4, "cscale": 3, "z0": 48}[operand]
    args = {"inc": inc, "cquat": cquat, "cscale": cscale, "z0": z0}
    args[operand] = wide[..., :keep]
    with pytest.raises(ValueError, match="contiguous"):
        state_passing_forward(args["inc"], args["cquat"], args["cscale"], args["z0"])


@pytest.mark.parametrize("operand", ["inc", "cquat", "cscale", "z0"])
def test_rejects_a_host_operand(operand: str) -> None:
    """A host tensor must be refused before the launch, not during it.

    Launching against a host pointer raises inside CUDA and leaves the context
    unusable for the rest of the process, so every later launch fails too. The
    check has to be on the host side of the call.
    """
    inc, cquat, cscale = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    args = {"inc": inc, "cquat": cquat, "cscale": cscale, "z0": z0}
    args[operand] = args[operand].cpu()
    with pytest.raises(ValueError, match="CUDA device"):
        state_passing_forward(args["inc"], args["cquat"], args["cscale"], args["z0"])


def test_block_width_divides_the_shape_multiples() -> None:
    """The exact launch depends on this, so it is asserted, not assumed.

    ``P*N`` is a multiple of ``HEAD_MULTIPLE * LANE_MULTIPLE``. That product
    being a multiple of the block width is what makes every launch exact.
    """
    assert (HEAD_MULTIPLE * LANE_MULTIPLE) % THREADS == 0
    assert THREADS % 32 == 0
