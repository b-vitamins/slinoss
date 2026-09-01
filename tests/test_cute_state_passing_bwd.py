"""Reverse inter-chunk recurrence against the float64 reference.

The kernel's operands are pipeline intermediates, so each is read off a reference
run rather than fabricated. ``dzstart`` comes from
:func:`slinoss.ops.so3ssd.chunked_backward`, which names it and names this
kernel's two outputs beside it; ``cquat`` and ``cscale`` come from the same
reference forward the operator runs. A ``randn`` readout cotangent and a
``randn`` rotation would not test the reverse frame change, which is the only
nontrivial thing here.

The two cotangent seeds are the exception and are drawn: ``dy`` and ``dstate``
are cotangents of the operator's outputs, so upstream of them is a loss rather
than a pipeline stage.

The authority is a full float64 ``chunked_backward``, not a float64 replay of the
float32 intermediates. That way the chunk endpoint, the transposed rotation, and
the reverse recurrence are all inside the comparison.

Operands are built in float32 and upcast, never built twice at two dtypes: the
generator consumes a different number of raw words per element at each width, so
the same seed at two dtypes is two different problems.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss.ops.so3ssd import ChunkedForward, chunked_backward, chunked_forward
from slinoss.ops.so3ssd.cute.bwd.state_passing import state_passing_backward
from slinoss.ops.so3ssd.cute.fwd.chunk_increment import chunk_increment_forward
from slinoss.ops.so3ssd.cute.fwd.state_passing import state_passing_forward
from tests.conftest import LS_BIAS, ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# LS_BIAS keeps the chunk decay above float32 epsilon. Unbiased, the recurrence under
# test is the identity on the readout cotangent.

CHUNK = 64

# (bsz, heads, seqlen, rows, lanes, with_dstate, with_dy).
#
# This kernel never sees ``L`` or ``T``: its extents are ``C``, ``P``, ``N`` and
# which of the two cotangent seeds is present, so the sweep varies only those. A
# ragged tail reaches the same instructions and is covered where it is a code
# path, in the chunk kernels. ``z0`` is not an axis either: ``dz0`` is produced
# whether or not the forward carried an initial state, so one is passed
# throughout.
#
# Four cases, one per distinct path. Each compile-time seed is dropped once, and
# never both in the same case: with both absent every quantity is identically
# zero and the comparison would pass whatever the transition does.
SHAPES = [
    pytest.param(2, 3, 256, 16, 16, True, True, id="four-chunks-two-tiles"),
    pytest.param(2, 2, 512, 32, 32, False, True, id="eight-tiles-zero-dstate"),
    pytest.param(1, 1, 64, 16, 16, True, True, id="single-chunk"),
    pytest.param(2, 2, 192, 16, 16, True, False, id="zero-dzstart"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    *,
    activation: torch.dtype | None = None,
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
        with_state=True,
        streaming=False,
        u_dtype=activation,
        bc_dtype=activation,
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


def _seeds(
    inp: ScanInputs, rows: int, lanes: int, *, with_dy: bool, with_dstate: bool
) -> tuple[Tensor | None, Tensor | None]:
    """Cotangents of ``y`` and of ``state``, or ``None`` where absent.

    Drawn rather than read off a pipeline record: upstream of an output cotangent
    is a loss.
    """
    gen = torch.Generator(device="cuda").manual_seed(7)
    bsz, heads, seqlen = (int(n) for n in inp.U.shape[:3])

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda")

    dy = rnd(bsz, heads, seqlen, rows) if with_dy else None
    dstate = rnd(bsz, heads, rows, 3 * lanes) if with_dstate else None
    return dy, dstate


def _transition(ref: ChunkedForward) -> tuple[Tensor, Tensor]:
    """The chunk transition the forward hands this kernel.

    Returns ``(cquat, cscale)``: the unit quaternion prefix at the end of each
    chunk and the chunk decay ``exp(2*lp_{L-1})``.
    """
    return (
        ref.qprefix[..., -1, :].contiguous(),
        torch.exp(2.0 * ref.lprefix[..., -1]).contiguous(),
    )


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "rows", "lanes", "with_dstate", "with_dy"), SHAPES
)
def test_state_passing_bwd_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    with_dstate: bool,
    with_dy: bool,
) -> None:
    """``dinc`` and ``dz0`` match a float64 ``chunked_backward`` end to end.

    With ``dy`` absent the buffer handed in is filled with NaN rather than left
    uninitialized, so a kernel that reads it instead of dropping the load at
    compile time poisons every chunk and fails here.
    """
    inp32 = _make(bsz, heads, seqlen, rows, lanes)
    inp64 = _upcast(inp32)
    dy32, dstate32 = _seeds(
        inp32, rows, lanes, with_dy=with_dy, with_dstate=with_dstate
    )
    dy64 = None if dy32 is None else dy32.double()
    dstate64 = None if dstate32 is None else dstate32.double()

    ref32 = chunked_backward(
        dy32, dstate32, None, None, *inp32.args(), CHUNK, **inp32.kw()
    )
    ref64 = chunked_backward(
        dy64, dstate64, None, None, *inp64.args(), CHUNK, **inp64.kw()
    )
    cquat, cscale = _transition(chunked_forward(*inp32.args(), CHUNK, **inp32.kw()))

    buf = (
        ref32.dzstart.contiguous()
        if with_dy
        else torch.full_like(ref32.dzstart, float("nan")).contiguous()
    )
    out = state_passing_backward(buf, cquat, cscale, dstate32, has_dzstart=with_dy)
    torch.cuda.synchronize()

    assert out.dinc.data_ptr() == buf.data_ptr(), "dinc must alias dzstart"
    tag = f"cute-state-passing-bwd[{bsz}x{heads}x{seqlen}/P{rows}/N{lanes}]"
    # Every chunk transition has norm at most one (I1), so the reverse recurrence
    # neither amplifies nor accumulates: the bound is float32 rounding of the
    # readout cotangent and of the chunk endpoint, not a per-chunk growth term.
    # Worst measured over this sweep 5.6e-7 on dinc and 1.9e-6 on dz0.
    #
    # ``dz0`` carries the whole chain and is what the bound is set by. The single
    # chunk case records exactly zero on ``dinc`` and that is the answer: with one
    # chunk ``dinc`` is the seed verbatim, so the assert states the seed is copied
    # through unaltered and the transition is measured by ``dz0``.
    assert_max_rel(out.dinc, ref64.dinc, 4e-6, f"{tag}.dinc")
    assert_max_rel(out.dz0, ref64.dz0, 4e-6, f"{tag}.dz0")
    # A comparison that only ever sees zeros passes whatever the transition does.
    # Every case must move at least one cotangent through the matrix.
    assert torch.count_nonzero(out.dinc) > 0
    assert torch.count_nonzero(out.dz0) > 0


def test_connects_the_forward_fast_path() -> None:
    """The two kernels compose on the tensors they exchange.

    The forward fast path runs for real -- the chunk increment kernel, then the
    forward recurrence -- and this kernel is handed that path's own ``cquat`` and
    ``cscale`` rather than the reference's. Both directions are then compared
    against one float64 reference. A disagreement between the reference's chunk
    endpoint and the increment kernel's would show here and nowhere else: the
    forward parity file feeds the forward reference-supplied operands, and the
    sweep above feeds this kernel the same.

    ``U``, ``B`` and ``C`` are bfloat16, which is what the increment kernel takes.
    ``cquat`` and ``cscale`` are functions of ``trans`` alone, so the activation
    width does not enter the transition either direction uses.
    """
    rows, lanes = 16, 16
    inp32 = _make(2, 2, 192, rows, lanes, activation=torch.bfloat16)
    inp64 = _upcast(inp32)
    dy32, dstate32 = _seeds(inp32, rows, lanes, with_dy=True, with_dstate=True)

    increment = chunk_increment_forward(inp32.U, inp32.trans, inp32.K, inp32.B, CHUNK)
    passing = state_passing_forward(
        increment.inc, increment.cquat, increment.cscale, inp32.z0
    )
    fwd64 = chunked_forward(*inp64.args(), CHUNK, **inp64.kw())
    ref32 = chunked_backward(
        dy32, dstate32, None, None, *inp32.args(), CHUNK, **inp32.kw()
    )
    ref64 = chunked_backward(
        None if dy32 is None else dy32.double(),
        None if dstate32 is None else dstate32.double(),
        None,
        None,
        *inp64.args(),
        CHUNK,
        **inp64.kw(),
    )

    out = state_passing_backward(
        ref32.dzstart.contiguous(), increment.cquat, increment.cscale, dstate32
    )
    torch.cuda.synchronize()

    # The forward's bound is the increment kernel's, which rounds the rotated
    # forcing to bfloat16 on its way into shared memory: worst measured 2.0e-3,
    # which is bfloat16 epsilon carried through a 64-long sum. The backward's is
    # the sweep's, because nothing this kernel reads is an activation: worst
    # measured 4.2e-7 on dinc and 4.1e-7 on dz0.
    assert_max_rel(passing.zstart, fwd64.zstart.flatten(-2, -1), 3e-3, "connect.zstart")
    assert_max_rel(out.dinc, ref64.dinc, 4e-6, "connect.dinc")
    assert_max_rel(out.dz0, ref64.dz0, 4e-6, "connect.dz0")


def test_absent_dstate_is_not_read() -> None:
    """The zero-seed variant ignores whatever stands in for the final cotangent.

    The kernel takes one signature and the no-seed launch hands it the ``dz0``
    buffer, which is uninitialized. The last chunk's ``dinc`` is that seed
    unchanged, so it must be exactly zero, and two runs over the same cotangents
    must agree bitwise.
    """
    inp = _make(2, 2, 192, 16, 16)
    dy, _ = _seeds(inp, 16, 16, with_dy=True, with_dstate=False)
    ref = chunked_backward(dy, None, None, None, *inp.args(), CHUNK, **inp.kw())
    cquat, cscale = _transition(chunked_forward(*inp.args(), CHUNK, **inp.kw()))

    first = state_passing_backward(ref.dzstart.clone().contiguous(), cquat, cscale)
    second = state_passing_backward(ref.dzstart.clone().contiguous(), cquat, cscale)
    torch.cuda.synchronize()

    assert torch.count_nonzero(first.dinc[:, :, -1]) == 0
    assert torch.equal(first.dinc, second.dinc)
    assert torch.equal(first.dz0, second.dz0)


def _operands(
    chunks: int = 3, rows: int = 16, lanes: int = 16
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A legally shaped operand quad, for the rejection tests to perturb."""
    opts = {"device": "cuda", "dtype": torch.float32}
    return (
        torch.empty(2, 2, chunks, rows, 3 * lanes, **opts),
        torch.empty(2, 2, chunks, 4, **opts),
        torch.empty(2, 2, chunks, **opts),
        torch.empty(2, 2, rows, 3 * lanes, **opts),
    )


@pytest.mark.parametrize("operand", ["dzstart", "cquat", "cscale", "dstate"])
def test_rejects_a_low_precision_operand(operand: str) -> None:
    """I4 pins the cotangents, the rotation, and the decay alike.

    Swept over every operand because that is what establishes membership of the
    checked tuple: an operand left out of it reaches no check at all. The layout
    rule reads the same tuple, so it needs one case rather than four.
    """
    dzstart, cquat, cscale, dstate = _operands()
    args = {
        "dzstart": dzstart,
        "cquat": cquat,
        "cscale": cscale,
        "dstate": dstate,
    }
    args[operand] = args[operand].bfloat16()
    with pytest.raises(ValueError, match=f"{operand} must be float32"):
        state_passing_backward(
            args["dzstart"], args["cquat"], args["cscale"], args["dstate"]
        )


def test_rejects_a_host_operand() -> None:
    """A host tensor must be refused before the launch, not during it.

    Launching against a host pointer raises inside CUDA and leaves the context
    unusable for the rest of the process, so every later launch fails too. The
    check has to be on the host side of the call.
    """
    dzstart, cquat, cscale, _ = _operands()
    with pytest.raises(ValueError, match="CUDA device"):
        state_passing_backward(dzstart.cpu(), cquat, cscale)


def test_rejects_a_non_contiguous_operand() -> None:
    """No repacking is done, so a strided operand is refused rather than fixed.

    Without the check ``dzstart`` reaches an internal ``view`` and raises
    ``RuntimeError``, so it is the case that fails least visibly by hand.
    """
    _, cquat, cscale, _ = _operands()
    wide = torch.empty(2, 2, 3, 16, 96, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="contiguous"):
        state_passing_backward(wide[..., :48], cquat, cscale)


def test_rejects_wrong_rank() -> None:
    """The cotangent is chunked and row-major; a flat one is a caller bug."""
    dzstart, cquat, cscale, _ = _operands()
    with pytest.raises(ValueError, match="expected"):
        state_passing_backward(dzstart.flatten(3, 4), cquat, cscale)


@pytest.mark.parametrize("operand", ["cquat", "cscale", "dstate"])
def test_rejects_a_mismatched_extent(operand: str) -> None:
    """One rotation and one decay per chunk, one seed per ``(P,3N)``.

    Nothing broadcasts. Each operand is sliced along the axis it shares with
    ``dzstart``, and each carries a different rank, so one check cannot cover all
    three. Sliced then made contiguous, so this reaches the shape check rather
    than the layout check.
    """
    dzstart, cquat, cscale, dstate = _operands()
    sliced = {
        "cquat": cquat[:, :, :-1],
        "cscale": cscale[:, :, :-1],
        "dstate": dstate[:, :, :-1],
    }
    full = {
        "dzstart": dzstart,
        "cquat": cquat,
        "cscale": cscale,
        "dstate": dstate,
    }
    full[operand] = sliced[operand].contiguous()
    with pytest.raises(ValueError, match=f"{operand} shape"):
        state_passing_backward(
            full["dzstart"], full["cquat"], full["cscale"], full["dstate"]
        )


def test_rejects_unlaunchable_shape() -> None:
    """The launch is exact, so an illegal ``(P, N)`` pair is refused, not padded.

    ``P`` a multiple of ``HEAD_MULTIPLE`` and ``N`` of ``LANE_MULTIPLE`` makes
    ``P*N`` a multiple of the block width. A shape that violates that reaches no
    kernel: the fix is the shape, not a tail predicate.
    """
    dzstart, cquat, cscale, _ = _operands(rows=3)
    with pytest.raises(ValueError, match="multiple of"):
        state_passing_backward(dzstart, cquat, cscale)
