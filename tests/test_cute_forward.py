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
from slinoss.ops.so3ssd.reference import to_heads
from tests.conftest import LS_BIAS, ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# LS_BIAS keeps the decay mask above float32 epsilon across the chunk. Unbiased, most
# of the score contributes nothing.

# ``y`` carries the chunk scan's bound and ``state`` carries the chunk
# increment's; both are one operand-dtype half-ulp against the largest entry, and
# the recurrence between them is float32 and contracting, so it adds nothing.
# Derived in tests/test_cute_chunk_scan.py and tests/test_cute_chunk_increment.py
# against the same operand construction.
BOUNDS = {torch.bfloat16: 6e-3, torch.float16: 8e-4}

# The streaming split's own bound, above the reference bound above it. The split
# carries ``b_last`` and ``u_last`` at the operand dtype, so the tail rounds a carry
# the whole-sequence path never materializes, and the split point is not a chunk
# multiple, so the tail's first chunk accumulates a prefix the whole sequence
# accumulated inside a longer one. Neither term is in the reference comparison, and
# sharing its figure was an accident of an operand set in which both were smaller
# than it. Worst measured: 6.849e-3, against 3.997e-3 for the same kernels' own
# reference gap at the nearest shape.
SPLIT_BOUND = 9e-3

# The gradient bound of the autograd wiring test. Two contraction orders over the
# same bf16 leaves, so it is the operand dtype's half-ulp against the largest
# entry, like the forward's. Worst measured: 7.092e-3, on ``dB``.
GRAD_BOUND = 8e-3

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
    groups: int | None = None,
    requires_grad: bool = False,
) -> ScanInputs:
    """One operand set: float32 pinned tensors, ``dtype`` activations."""
    return make_inputs(
        bsz=bsz,
        heads=heads,
        groups=groups,
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


def test_a_grouped_call_matches_the_broadcast_ungrouped_call() -> None:
    """``G < H`` against the same call on ``B`` and ``C`` broadcast to ``H``.

    All three kernels compute the group index from the same compile-time ``H // G``,
    so the composition is where it is checked once rather than three times. Compared
    against the ungrouped path, which the case above already holds to float64, and
    compared bitwise: the two runs issue the same instructions on the same values in
    the same order, and the only difference is the address each block reads ``b`` and
    ``c`` from. A tolerance here would admit a wrong group.

    ``G`` does not interact with the shape axes, so one shape is enough. It carries a
    ragged tail and a streaming carry, which is what makes ``b_prev`` grouped too.
    """
    heads, groups, chunk = 4, 2, 64
    inp = _make(2, heads, 200, 48, 16, True, torch.bfloat16, groups=groups)
    assert tuple(inp.B.shape[:2]) == (2, groups)
    # Materialized: at G = 1 the broadcast is a stride-0 view, and the operator
    # refuses a non-contiguous operand rather than repacking it.
    wide = inp._replace(
        B=to_heads(inp.B, heads).contiguous(),
        C=to_heads(inp.C, heads).contiguous(),
        b_prev=None if inp.b_prev is None else to_heads(inp.b_prev, heads).contiguous(),
    )
    got = so3ssd(*inp.args(), chunk, **inp.kw(), backend="cute")
    want = so3ssd(*wide.args(), chunk, **wide.kw(), backend="cute")
    torch.cuda.synchronize()

    assert torch.equal(got.y, want.y)
    assert torch.equal(got.state, want.state)
    assert torch.equal(got.u_last, want.u_last)
    # b_last is a time slice of the grouped B, so it stays grouped.
    assert torch.equal(to_heads(got.b_last, heads), want.b_last)
    # A comparison of two all-zero outputs passes whatever the group index does.
    assert torch.count_nonzero(got.y) > 0


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
    # Both sides run the same kernels at the same width, so the gap is the rounded
    # carry and the reordering of one float32 chunk recurrence into two.
    assert_max_rel(joined, whole.y, SPLIT_BOUND, "cute-split.y")
    assert_max_rel(tail.state, whole.state, SPLIT_BOUND, "cute-split.state")


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


def test_the_backward_reaches_every_leaf_the_forward_saved() -> None:
    """The context this backend's backward is driven from.

    The saved set, the chunk size and the streaming carry all reach the backward
    through the context, and a leaf that was saved wrong or a chunk that was
    recorded wrong lands far outside a rounding difference. What the two backends
    agree to is a tolerance rather than a bit pattern: since the CuTe backward
    landed, the gradient here is seven launches with their own contraction order,
    and only the reference's own accuracy chain in tests/test_cute_backward.py can
    call that order right. So the bound is the operand dtype's, and this test's
    subject is which leaves are connected and how well, not the arithmetic.

    One cotangent, not the outputs: squaring would feed each backward its own
    forward error on top of its own rounding.
    """
    fast = _make(2, 2, 200, 48, 16, True, torch.bfloat16, requires_grad=True)
    ref = _make(2, 2, 200, 48, 16, True, torch.bfloat16, requires_grad=True)
    chunk = 64
    got = so3ssd(*fast.args(), chunk, **fast.kw(), backend="cute")
    want = so3ssd(*ref.args(), chunk, **ref.kw(), backend="reference")
    got.y.float().sum().backward()
    want.y.float().sum().backward()
    torch.cuda.synchronize()

    for name in ("U", "trans", "K", "B", "C", "z0", "b_prev", "u_prev"):
        a = getattr(fast, name).grad
        b = getattr(ref, name).grad
        assert a is not None and b is not None, name
        assert_max_rel(a, b, GRAD_BOUND, f"cute autograd d{name}")
