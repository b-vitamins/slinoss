"""The composed CuTe forward, through the public operator.

Each of the three kernels is swept over its own shape axes in its own test file.
What is only reachable here is the composition: the order of the launches, the
state handed between them, the four fields of the result, and the dispatch that
makes ``so3ssd`` select this path for a low-precision call. So the cases below
sweep the axes the composition owns -- carry present or absent, a ragged tail
against an exact division, both operand dtypes -- and not the axes the kernels
own.

Operands are built in float32 and cast, and the reference runs on an exact
float64 upcast of those same cast tensors, so the residual is the kernels' own
arithmetic rather than the rounding of the inputs.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss._cute import executor_count
from slinoss.ops.so3ssd import SO3SSDResult, resolve, so3ssd, so3ssd_ref
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it the decay mask underflows a few tokens in and most
# of the score contributes nothing.
LS_BIAS = -4.0

# ``y`` carries the chunk scan's bound and ``state`` carries the chunk
# increment's; both are one operand-dtype half-ulp against the largest entry, and
# the recurrence between them is float32 and contracting, so it adds nothing.
# Derived in tests/test_cute_chunk_scan.py and tests/test_cute_chunk_increment.py
# against the same operand construction.
BOUNDS = {torch.bfloat16: 6e-3, torch.float16: 8e-4}

# (bsz, heads, seqlen, chunk, rows, lanes, streaming, dtype).
SHAPES = [
    # Ragged tail, streaming carry, nonzero initial state, three chunks and one
    # short one: every field of the result is live and the last token is not the
    # last chunk slot.
    pytest.param(2, 2, 200, 64, 48, 16, True, torch.bfloat16, id="ragged-streaming"),
    # One chunk exactly, no carry, no initial state, smallest legal rows, the
    # second lane count, the second operand dtype.
    pytest.param(1, 1, 16, 16, 16, 32, False, torch.float16, id="single-chunk-fp16"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    streaming: bool,
    dtype: torch.dtype,
    *,
    requires_grad: bool = False,
) -> ScanInputs:
    """One operand set: float32 pinned tensors, ``dtype`` activations."""
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
        with_state=streaming,
        streaming=streaming,
        u_dtype=dtype,
        bc_dtype=dtype,
        requires_grad=requires_grad,
    )


def _reference(inp: ScanInputs, chunk: int) -> SO3SSDResult:
    """The float64 authority on the same cast operands."""
    return so3ssd_ref(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        chunk,
        z0=None if inp.z0 is None else inp.z0.double(),
        b_prev=None if inp.b_prev is None else inp.b_prev.double(),
        u_prev=None if inp.u_prev is None else inp.u_prev.double(),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_a_low_precision_call_on_cuda_selects_the_cute_backend() -> None:
    """The benchmarked path is the shipped path: nothing names this backend."""
    assert resolve(None, "cuda", torch.bfloat16).name == "cute"
    assert resolve(None, "cuda", torch.float16).name == "cute"


def test_a_float32_call_on_cuda_selects_the_reference() -> None:
    """The MMA atom is 16-bit, so float32 has no instantiation. Resolution routes
    around it rather than letting a kernel raise on an operand nobody chose it
    for."""
    assert resolve(None, "cuda", torch.float32).name == "reference"


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "streaming", "dtype"), SHAPES
)
def test_every_field_matches_the_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    streaming: bool,
    dtype: torch.dtype,
) -> None:
    """All four outputs of the composed path against a float64 forward."""
    inp = _make(bsz, heads, seqlen, rows, lanes, streaming, dtype)
    ref = _reference(inp, chunk)
    out = so3ssd(*inp.args(), chunk, **inp.kw(), backend="cute")
    torch.cuda.synchronize()

    tag = (
        f"cute-forward[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    assert out.y.shape == inp.U.shape and out.y.dtype is dtype
    assert out.state.shape == (bsz, heads, rows, 3 * lanes)
    assert out.state.dtype is torch.float32
    assert_max_rel(out.y, ref.y, BOUNDS[dtype], f"{tag}.y")
    assert_max_rel(out.state, ref.state, BOUNDS[dtype], f"{tag}.state")
    # A comparison against zeros passes whatever the GEMMs do.
    assert torch.count_nonzero(out.y) > 0

    # The carry is the last token, not the last chunk slot: a ragged tail pads
    # the chunk and a padded token is a no-op whose b and u are zero. Exact,
    # because both are slices of the caller's own tensors.
    assert torch.equal(out.b_last, inp.B[:, :, -1])
    assert torch.equal(out.u_last, inp.U[:, :, -1])


def test_the_streaming_split_reproduces_the_whole_sequence() -> None:
    """Feed the carry from one call into the next and rejoin.

    The split point is not a multiple of the chunk, so the head ends on a ragged
    chunk and the tail starts mid-chunk. This is the only path on which a token
    reads a previous tap that the kernel did not compute itself.
    """
    inp = _make(2, 2, 128, 48, 16, False, torch.bfloat16)
    chunk = 64
    whole = so3ssd(*inp.args(), chunk, backend="cute")
    hu, htrans, hk, hb, hc = (t[:, :, :40].contiguous() for t in inp.args())
    tu, ttrans, tk, tb, tc = (t[:, :, 40:].contiguous() for t in inp.args())
    head = so3ssd(hu, htrans, hk, hb, hc, chunk, backend="cute")
    tail = so3ssd(
        tu,
        ttrans,
        tk,
        tb,
        tc,
        chunk,
        z0=head.state,
        b_prev=head.b_last,
        u_prev=head.u_last,
        backend="cute",
    )
    torch.cuda.synchronize()

    joined = torch.cat([head.y, tail.y], dim=2)
    # Both sides run the same kernels at the same width, so the gap is the
    # reordering of one float32 chunk recurrence into two.
    bound = BOUNDS[torch.bfloat16]
    assert_max_rel(joined, whole.y, bound, "cute-split.y")
    assert_max_rel(tail.state, whole.state, bound, "cute-split.state")


# ---------------------------------------------------------------------------
# Compiled launch
# ---------------------------------------------------------------------------


def test_one_executor_per_kernel_serves_every_batch_head_and_length() -> None:
    """The shape is not in the executor cache key, and the chunk length is.

    ``dev_tensor`` marks every layout dynamic but the leading mode, so a tensor
    argument contributes its element type and its rank and nothing else. If the
    extents entered the key instead, a variable sequence length would retrace
    the host function on every call, and the trace is milliseconds against a
    kernel that is microseconds.

    The second half is the other side of the same claim: a compile-time argument
    must key the cache, or a second chunk length would silently run the first
    one's code.
    """
    warm = _make(2, 2, 128, 48, 16, False, torch.bfloat16)
    so3ssd(*warm.args(), 64, backend="cute")
    before = executor_count()

    other = _make(1, 3, 320, 48, 16, False, torch.bfloat16)
    so3ssd(*other.args(), 64, backend="cute")
    assert executor_count() == before

    so3ssd(*warm.args(), 32, backend="cute")
    assert executor_count() > before


# ---------------------------------------------------------------------------
# Autograd wiring
# ---------------------------------------------------------------------------


def test_the_backward_is_the_reference_backward_on_the_same_saved_inputs() -> None:
    """This backend's gradient is the reference's until the CuTe backward lands.

    Under one fixed cotangent the two backends must therefore agree bitwise: the
    saved set, the chunk size, and the backend name all reach the backward
    through the same context, and a divergence there is the only thing this can
    catch. The connection test -- fast forward, differentiate, compare end to end
    -- belongs to the backend whose backward is its own, and lives in
    tests/test_interface.py for the reference.
    """
    fast = _make(2, 2, 200, 48, 16, True, torch.bfloat16, requires_grad=True)
    ref = _make(2, 2, 200, 48, 16, True, torch.bfloat16, requires_grad=True)
    chunk = 64
    got = so3ssd(*fast.args(), chunk, **fast.kw(), backend="cute")
    want = so3ssd(*ref.args(), chunk, **ref.kw(), backend="reference")
    # One cotangent, not the outputs: squaring would feed each backward its own
    # forward error and turn a bitwise check into a tolerance.
    got.y.float().sum().backward()
    want.y.float().sum().backward()
    torch.cuda.synchronize()

    for name in ("U", "trans", "K", "B", "C", "z0", "b_prev", "u_prev"):
        a = getattr(fast, name).grad
        b = getattr(ref, name).grad
        assert a is not None and b is not None, name
        assert torch.equal(a, b), name
