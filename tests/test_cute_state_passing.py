"""Inter-chunk recurrence against the float64 reference.

The kernel's two inputs are pipeline intermediates, so they are produced by a
float32 reference run rather than fabricated: a ``randn`` increment and a
``randn`` transition would not test the packing of the chunk transition into one
scaled quaternion, which is the only nontrivial thing here.

The authority is a full float64 ``chunked_forward``, not a float64 replay of the
float32 intermediates. That way the packing, the degree-two homogeneity of the
rotation matrix, and the recurrence itself are all inside the comparison.

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
from slinoss.ops.so3ssd import ChunkedForward, as_lanes, chunked_forward
from slinoss.ops.so3ssd.cute.common import THREADS
from slinoss.ops.so3ssd.cute.fwd.state_passing import state_passing_forward
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it a 64-token chunk reaches exp(2*lp) near 1e-54, the
# transition is zero to float32, and the recurrence under test is the identity on
# the increment. The bias keeps every chunk transition normal and significant.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, with_state). P and N are at and above
# their legal minima, so the vector count P*N runs from two blocks per (b,h) to
# eight, and the chunk count from one to seven.
#
# A single chunk carries an initial state, and a zero initial state comes with
# several chunks. The two together would be vacuous: with one chunk and no state,
# `zstart` is the zero the kernel just wrote and the transition is never applied,
# so the recorded relative error is exactly zero whatever the matrix does.
SHAPES = [
    pytest.param(2, 3, 256, 64, 16, 16, True, id="four-chunks"),
    pytest.param(2, 3, 200, 64, 16, 16, True, id="ragged-tail"),
    pytest.param(1, 1, 64, 64, 16, 16, True, id="single-chunk"),
    pytest.param(1, 2, 384, 128, 16, 32, True, id="L128-four-tiles"),
    pytest.param(2, 2, 100, 16, 32, 16, False, id="seven-chunks-zero-start"),
    pytest.param(1, 1, 128, 64, 32, 32, True, id="eight-tiles"),
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


def _pack(ref: ChunkedForward) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull the kernel's two inputs out of a reference run.

    Returns ``(inc, ctrans)``: the chunk increment already carried into the
    global frame, and the chunk transition packed as ``exp(lp_{L-1}) Q_{L-1}``.
    """
    chunk_rot = ref.table.rot[..., -1, :, :]
    inc = torch.einsum(
        "bhcij,bhcpnj->bhcpni", chunk_rot, as_lanes(ref.inc_local)
    ).flatten(-2, -1)
    ctrans = torch.exp(ref.lprefix[..., -1])[..., None] * ref.qprefix[..., -1, :]
    return inc.contiguous(), ctrans.contiguous()


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
    inc, ctrans = _pack(chunked_forward(*inp32.args(), chunk, **inp32.kw()))
    ref = chunked_forward(*inp64.args(), chunk, **inp64.kw())

    out = state_passing_forward(inc, ctrans, inp32.z0)
    torch.cuda.synchronize()

    assert out.zstart.data_ptr() == inc.data_ptr(), "zstart must alias inc"
    tag = f"cute-state-passing[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}]"
    # Every chunk transition has norm at most one (I1), so the recurrence
    # neither amplifies nor accumulates: the bound is float32 rounding of the
    # increment and of the packed transition, not a per-chunk growth term.
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
    inc, ctrans = _pack(chunked_forward(*inp.args(), 64, **inp.kw()))
    first = state_passing_forward(inc.clone(), ctrans)
    second = state_passing_forward(inc.clone(), ctrans)
    torch.cuda.synchronize()
    assert torch.equal(first.zstart, second.zstart)
    assert torch.equal(first.state, second.state)
    assert torch.count_nonzero(first.zstart[:, :, 0]) == 0


def _inputs(
    chunks: int = 3, rows: int = 16, lanes: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """A legally shaped operand pair, for the rejection tests to perturb."""
    return (
        torch.empty(2, 2, chunks, rows, 3 * lanes, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, chunks, 4, device="cuda", dtype=torch.float32),
    )


def test_rejects_low_precision_inc() -> None:
    """I4 pins the state, so a narrow increment has no kernel path."""
    inc, ctrans = _inputs()
    with pytest.raises(ValueError, match="float32"):
        state_passing_forward(inc.bfloat16(), ctrans)


def test_rejects_low_precision_ctrans() -> None:
    """I4 pins the transition too."""
    inc, ctrans = _inputs()
    with pytest.raises(ValueError, match="float32"):
        state_passing_forward(inc, ctrans.bfloat16())


def test_rejects_wrong_rank() -> None:
    """The increment is chunked and row-major; a flat one is a caller bug."""
    inc, ctrans = _inputs()
    with pytest.raises(ValueError, match="expected"):
        state_passing_forward(inc.flatten(3, 4), ctrans)


def test_rejects_mismatched_ctrans() -> None:
    """One transition per chunk, no broadcasting."""
    inc, ctrans = _inputs()
    # Contiguous, so this reaches the shape check rather than the layout check.
    with pytest.raises(ValueError, match="ctrans shape"):
        state_passing_forward(inc, ctrans[:, :, :-1].contiguous())


def test_rejects_unlaunchable_shape() -> None:
    """The launch is exact, so an illegal ``(P, N)`` pair is refused, not padded.

    ``P`` a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE`` makes
    ``P*N`` a multiple of the block width. A shape that violates that reaches no
    kernel: the fix is the shape, not a tail predicate.
    """
    inc, ctrans = _inputs(rows=3)
    with pytest.raises(ValueError, match="multiple of"):
        state_passing_forward(inc, ctrans)


def test_rejects_low_precision_z0() -> None:
    """The initial state is pinned by I4 as well."""
    inc, ctrans = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="float32 z0"):
        state_passing_forward(inc, ctrans, z0.bfloat16())


def test_rejects_mismatched_z0() -> None:
    """The initial state carries no chunk axis and must match ``(P,3N)``."""
    inc, ctrans = _inputs()
    z0 = torch.empty(2, 2, 32, 48, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="z0 shape"):
        state_passing_forward(inc, ctrans, z0)


@pytest.mark.parametrize("operand", ["inc", "ctrans", "z0"])
def test_rejects_a_non_contiguous_operand(operand: str) -> None:
    """No repacking is done, so a strided operand is refused rather than fixed.

    The three cases fail three different ways without the check: ``inc`` and
    ``z0`` reach an internal ``view`` and raise ``RuntimeError``, while
    ``ctrans`` is not reshaped at all and would be read at the wrong stride and
    return a wrong answer with no error.
    """
    inc, ctrans = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    wide = {
        "inc": torch.empty(2, 2, 3, 16, 96, device="cuda", dtype=torch.float32),
        "ctrans": torch.empty(2, 2, 3, 8, device="cuda", dtype=torch.float32),
        "z0": torch.empty(2, 2, 16, 96, device="cuda", dtype=torch.float32),
    }[operand]
    sliced = wide[..., : (4 if operand == "ctrans" else 48)]
    args = {"inc": inc, "ctrans": ctrans, "z0": z0}
    args[operand] = sliced
    with pytest.raises(ValueError, match="contiguous"):
        state_passing_forward(args["inc"], args["ctrans"], args["z0"])


@pytest.mark.parametrize("operand", ["inc", "ctrans", "z0"])
def test_rejects_a_host_operand(operand: str) -> None:
    """A host tensor must be refused before the launch, not during it.

    Launching against a host pointer raises inside CUDA and leaves the context
    unusable for the rest of the process, so every later launch fails too. The
    check has to be on the host side of the call.
    """
    inc, ctrans = _inputs()
    z0 = torch.empty(2, 2, 16, 48, device="cuda", dtype=torch.float32)
    args = {"inc": inc, "ctrans": ctrans, "z0": z0}
    args[operand] = args[operand].cpu()
    with pytest.raises(ValueError, match="cuda"):
        state_passing_forward(args["inc"], args["ctrans"], args["z0"])


def test_block_width_divides_the_shape_multiples() -> None:
    """The exact launch depends on this, so it is asserted, not assumed.

    ``P*N`` is a multiple of ``HEAD_MULTIPLE * LANE_MULTIPLE``. That product
    being a multiple of the block width is what makes every launch exact.
    """
    assert (HEAD_MULTIPLE * LANE_MULTIPLE) % THREADS == 0
    assert THREADS % 32 == 0
