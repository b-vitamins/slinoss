"""Chunk increment against the float64 reference.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors. So the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic: the float32 table, the float32 accumulator, and the one narrowing of
the rotated forcing on its way into shared memory.

The rotation carried at the chunk endpoint and the chunk decay are asserted here
too. Their arithmetic is covered by the device-math probe, which reaches the same
``chunk_endpoint`` on the same shared tiles; what is covered only here is that
this kernel writes them at the right index of the right tensor.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss._cute import smem_capacity
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import chunked_forward
from slinoss.ops.so3ssd.cute.fwd.chunk_increment import (
    chunk_increment_forward,
    increment_smem_bytes,
)
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it the chunk weight at the head of a 128-token chunk
# underflows and most of the K extent contributes nothing, which would leave the
# GEMM tested on a handful of trailing tokens.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, streaming, dtype).
#
# One case per distinct path through this kernel:
#
# - the K slice count, one at L=64 and two at L=128, which is the only loop whose
#   trip count changes what is staged;
# - a ragged tail, where the pad tap must zero the operand rather than the
#   predicate skipping a store;
# - the streaming split, which is the only way the previous tap reaches a token
#   before the sequence;
# - ``P`` padded up to the MMA tile against ``P`` exactly on it, which selects
#   between the predicated store and the vectorized one;
# - two ``N``, because the lane stride loop is per-thread;
# - both operand dtypes, because each is a different MMA atom.
SHAPES = [
    pytest.param(2, 2, 256, 64, 16, 16, True, torch.bfloat16, id="one-slice-streaming"),
    pytest.param(2, 2, 200, 64, 48, 32, False, torch.bfloat16, id="ragged-no-carry"),
    pytest.param(1, 1, 256, MAX_CHUNK, 64, 16, True, torch.bfloat16, id="two-slices"),
    pytest.param(2, 2, 64, 64, 16, 16, True, torch.float16, id="single-chunk-fp16"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    streaming: bool,
    dtype: torch.dtype,
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
        streaming=streaming,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "streaming", "dtype"), SHAPES
)
def test_chunk_increment_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    streaming: bool,
    dtype: torch.dtype,
) -> None:
    """The increment and the chunk transition match a float64 forward."""
    inp = _make(bsz, heads, seqlen, rows, lanes, streaming, dtype)
    ref = chunked_forward(
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
    out = chunk_increment_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        chunk,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
    )
    torch.cuda.synchronize()

    tag = (
        f"cute-increment[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    # The rotated forcing is rounded to the operand dtype once on its way into
    # shared memory, which the reference does not do, so the bound is that dtype's
    # epsilon carried through a K-long sum against the largest entry of the result.
    # Everything else is a float32 accumulation. Worst measured 2.7e-3 at bfloat16
    # and 2.7e-4 at float16, a ratio of ten that tracks the ratio of the two
    # significands, so each dtype carries its own bound rather than both sharing a
    # figure loose enough for the wider one.
    bound = 6e-3 if dtype is torch.bfloat16 else 8e-4
    assert_max_rel(out.inc, ref.inc_local, bound, f"{tag}.inc")
    # A comparison against zeros passes whatever the GEMM does.
    assert torch.count_nonzero(out.inc) > 0

    # Both carried quantities are float32 throughout, so they hold the prefix
    # bound rather than the operand bound.
    assert_max_rel(out.cquat, ref.qprefix[..., -1, :], 2e-6, f"{tag}.cquat")
    want_scale = torch.exp(2.0 * ref.lprefix[..., -1])
    assert bool((out.cscale > 0.0).all()) and bool((out.cscale <= 1.0).all()), "I1"
    # exp turns the float32 absolute error of the log prefix into a relative
    # error that grows with how far the prefix reaches.
    reach = float(ref.lprefix[..., -1].abs().max())
    eps = float(torch.finfo(torch.float32).eps)
    assert_max_rel(out.cscale, want_scale, 4e-6 + 4.0 * reach * eps, f"{tag}.cscale")


def test_previous_tap_reads_across_the_chunk_boundary() -> None:
    """Dropping the carry-in changes the answer, so the boundary is under test.

    Catches a kernel that stages the previous tap from the chunk's own first
    token, or zeroes it at every chunk start. Both are invisible in a parity test
    whose first chunk is the only one with a boundary, and both would leave the
    ragged and streaming cases above agreeing to within the operand bound.
    """
    inp = _make(2, 2, 128, 16, 16, True, torch.bfloat16)
    with_carry = chunk_increment_forward(
        inp.U, inp.trans, inp.K, inp.B, 64, u_prev=inp.u_prev, b_prev=inp.b_prev
    )
    without = chunk_increment_forward(inp.U, inp.trans, inp.K, inp.B, 64)
    torch.cuda.synchronize()
    # Chunk 0 sees the carry-in; chunk 1 reads its predecessor either way.
    assert not torch.equal(with_carry.inc[:, :, 0], without.inc[:, :, 0])
    assert torch.equal(with_carry.inc[:, :, 1], without.inc[:, :, 1])
    assert torch.equal(with_carry.cquat, without.cquat)


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The widest legal tiles are the binding case: ``MAX_CHUNK`` doubles every
    per-token tile and is the only shape that can overflow the carveout.
    """
    nbytes = increment_smem_bytes(MAX_CHUNK, 64, 96)
    assert nbytes <= smem_capacity()
    # Two blocks per SM at the widest shape is what keeps the DRAM pipe fed.
    assert 2 * nbytes <= smem_capacity()
    assert increment_smem_bytes(64, 16, 48) < nbytes


def _ok() -> dict[str, torch.Tensor]:
    """A legal operand set for the rejection table to perturb."""
    inp = _make(2, 2, 128, 16, 16, True, torch.bfloat16)
    assert inp.u_prev is not None and inp.b_prev is not None
    return {
        "U": inp.U,
        "trans": inp.trans,
        "K": inp.K,
        "B": inp.B,
        "u_prev": inp.u_prev,
        "b_prev": inp.b_prev,
    }


def _strided(tensor: torch.Tensor) -> torch.Tensor:
    """A view of ``tensor``'s shape whose last axis is strided."""
    shape = (*tensor.shape[:-1], 2 * int(tensor.shape[-1]))
    wide = torch.empty(shape, device=tensor.device, dtype=tensor.dtype)
    return wide[..., : tensor.shape[-1]]


Operands = dict[str, torch.Tensor]

REJECTIONS: list[tuple[Callable[[Operands], None], type[Exception], str]] = [
    (lambda a: a.update(U=a["U"].cpu()), ValueError, "U must be on a CUDA device"),
    (lambda a: a.update(B=_strided(a["B"])), ValueError, "B must be contiguous"),
    (lambda a: a.update(U=a["U"].float()), TypeError, "U has dtype"),
    (lambda a: a.update(B=a["B"].half()), TypeError, "one dtype per call"),
    (lambda a: a.update(trans=a["trans"].bfloat16()), ValueError, "trans must be"),
    (lambda a: a.update(K=a["K"].bfloat16()), ValueError, "K must be float32"),
    (
        lambda a: a.update(U=a["U"][:, :, 0].contiguous()),
        ValueError,
        r"U must be \(B,H,T,P\)",
    ),
    (lambda a: a.update(trans=a["trans"][..., :3].contiguous()), ValueError, "trans"),
    (lambda a: a.update(K=a["K"][..., :1, :].contiguous()), ValueError, "K must be"),
    (
        lambda a: a.update(B=a["B"][:, :, 0].contiguous()),
        ValueError,
        r"B must be \(B,G,T,3N\)",
    ),
    # A head count is no longer a shape violation: any G dividing H is legal. What
    # is left to refuse is a G that does not divide, which would send some head past
    # the end of B.
    (
        lambda a: a.update(B=a["B"][:, :1].repeat(1, 3, 1, 1)),
        ValueError,
        "does not divide H=2",
    ),
    (lambda a: a.update(u_prev=a["u_prev"][:1].contiguous()), ValueError, "u_prev"),
    (
        lambda a: a.update(b_prev=a["b_prev"][..., :3].contiguous()),
        ValueError,
        "b_prev",
    ),
]
"""Every ``raise`` reachable from the public path bar the unpaired stream, named
by its message.

Layout and dtype before shape, because the checks run in that order: a mutation
that changes two things at once must be matched by the first check it reaches.
"""


@pytest.mark.parametrize(("mutate", "exc", "match"), REJECTIONS)
def test_rejects_a_bad_operand(
    mutate: Callable[[Operands], None],
    exc: type[Exception],
    match: str,
) -> None:
    """A violation is refused on the host, not repacked and not launched.

    Launching against a host pointer or a wrong stride either faults inside CUDA,
    which leaves the context unusable for every later launch in the process, or
    returns a wrong answer with no error at all.
    """
    args = _ok()
    mutate(args)
    with pytest.raises(exc, match=match):
        chunk_increment_forward(
            args["U"],
            args["trans"],
            args["K"],
            args["B"],
            64,
            u_prev=args["u_prev"],
            b_prev=args["b_prev"],
        )


def test_rejects_an_unpaired_stream() -> None:
    """One tap without the other is a caller bug, not a half-streaming call.

    The two are read at the same token, so a call carrying only ``b_prev`` would
    pair it with a zero ``u`` and return a wrong answer with no error.
    """
    args = _ok()
    with pytest.raises(ValueError, match="supplied together"):
        chunk_increment_forward(
            args["U"], args["trans"], args["K"], args["B"], 64, b_prev=args["b_prev"]
        )


@pytest.mark.parametrize(
    ("chunk", "lanes", "match"),
    [
        (24, 16, "multiple of 16"),
        (64, 15, "3N must be"),
    ],
)
def test_rejects_an_extent_the_atom_cannot_cover(
    chunk: int, lanes: int, match: str
) -> None:
    """The fix for an illegal extent is the shape, never a padding path.

    ``L`` sets the MMA K extent and ``3N`` sets its N extent, and neither mode
    admits a tail.
    """
    inp = _make(2, 2, 128, 16, lanes, False, torch.bfloat16)
    with pytest.raises(ValueError, match=match):
        chunk_increment_forward(inp.U, inp.trans, inp.K, inp.B, chunk)
