"""Chunk-start state cotangent against the float64 reference.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors. So the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic: the float32 table, the float32 accumulator, and the one narrowing of
each operand on its way into shared memory.

``dzstart`` is a function of ``dy``, ``trans``, and ``C`` alone. ``U``, ``K``,
``B``, ``z0``, and the streaming carry-in do not reach it, so those axes are not
swept here: the kernel does not take them.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss._cute import smem_capacity
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import chunked_backward, chunked_forward
from slinoss.ops.so3ssd.cute.bwd.chunk_start import (
    chunk_start_backward,
    start_smem_bytes,
)
from tests.conftest import (
    ScanInputs,
    assert_max_rel,
    make_inputs,
    projection_band,
)

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it ``exp(2*lp)`` underflows a few tokens into the chunk
# and every later row of the K extent contributes nothing, which would leave the
# GEMM tested on the chunk's first handful of tokens.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, dtype).
#
# One case per distinct path through this kernel:
#
# - a ragged tail, where both staging passes must zero the rows past ``valid``
#   rather than a predicate skipping a store, together with three or more chunks;
# - a single chunk at the smallest legal ``N`` and the smallest ``P``;
# - ``MAX_CHUNK`` with ``B = H = 1`` and ``P`` exactly on the MMA tile, which
#   selects the vectorized store over the predicated one;
# - two ``N``, because the lane stride loop inside the rotated staging pass is
#   per-thread;
# - both operand dtypes, because each is a different MMA atom.
#
# ``L`` and ``P`` do not interact: ``L`` is the K extent and ``P`` is the M mode,
# rounded up in shared memory either way, so they are swept and not crossed.
SHAPES = [
    pytest.param(2, 2, 200, 64, 48, 32, torch.bfloat16, id="ragged-three-chunks"),
    pytest.param(2, 2, 64, 64, 16, 16, torch.bfloat16, id="single-chunk"),
    pytest.param(1, 1, 256, MAX_CHUNK, 64, 16, torch.bfloat16, id="max-chunk-b1h1"),
    pytest.param(2, 2, 128, 64, 16, 16, torch.float16, id="fp16"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
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
        u_dtype=dtype,
        bc_dtype=dtype,
        requires_grad=requires_grad,
    )


def _cotangent(inp: ScanInputs, dtype: torch.dtype, seed: int = 17) -> torch.Tensor:
    """``dy`` in the operand dtype. A loss gradient, not an intermediate."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(
        inp.U.shape, generator=gen, dtype=torch.float32, device=inp.U.device
    )
    return raw.to(dtype)


def _reference(inp: ScanInputs, dy: torch.Tensor, chunk: int) -> torch.Tensor:
    """``dzstart`` from the float64 reference backward, ``(B,H,C,P,3N)``."""
    ref = chunked_backward(
        dy.double(),
        None,
        None,
        None,
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
    return ref.dzstart


# The rotated readout and the weighted cotangent are each rounded to the operand
# dtype once on their way into shared memory, which the reference does not do, so
# the bound is that dtype's epsilon carried through an L-long sum against the
# largest entry of the result. Everything else is a float32 accumulation. Worst
# measured 4.1e-3 at bfloat16 over every shape below and 2.8e-4 at float16; at the
# one shape both dtypes run, 2.8e-3 against 2.8e-4, a ratio of ten that tracks the
# ratio of the two significands. So each dtype carries its own bound rather than
# both sharing a figure loose enough for the wider one.
BOUNDS = {torch.bfloat16: 8e-3, torch.float16: 6e-4}


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "dtype"), SHAPES
)
def test_chunk_start_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
) -> None:
    """The chunk-start cotangent matches a float64 reference backward."""
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype)
    dy = _cotangent(inp, dtype)
    want = _reference(inp, dy, chunk)
    got = chunk_start_backward(dy, inp.trans, inp.C, chunk)
    torch.cuda.synchronize()

    tag = (
        f"cute-chunk-start[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    assert got.shape == want.shape
    assert_max_rel(got, want, BOUNDS[dtype], tag)
    # A comparison against zeros passes whatever the GEMM does.
    assert torch.count_nonzero(got) > 0


@pytest.mark.parametrize("groups", [1, 4], ids=["one-group", "group-per-head"])
def test_grouped_readout_reads_its_own_group(groups: int) -> None:
    """Head ``h`` contracts against group ``h // (H // G)``.

    ``G == 1`` is the only case where every head reads one shared ``C``, and it is
    the case a missing divide silently passes at ``G == H``. An intermediate ``G``
    takes the same divide as ``G == 1``, so it is not swept.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=groups)
    dy = _cotangent(inp, torch.bfloat16)
    want = _reference(inp, dy, 64)
    got = chunk_start_backward(dy, inp.trans, inp.C, 64)
    torch.cuda.synchronize()
    assert_max_rel(got, want, BOUNDS[torch.bfloat16], f"cute-chunk-start[G{groups}]")


def test_reads_a_band_of_the_fused_projection() -> None:
    """``C`` ships pitched, and the kernel indexes the band rather than a copy of it.

    One projection GEMM feeds every consumer, so ``C`` is a column band of its
    output and never a buffer of its own. Recovering contiguity would be the staging
    copy the layout contract exists to refuse. Nothing about the arithmetic changes,
    so the two layouts must agree bit for bit rather than within a tolerance.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=2)
    dy = _cotangent(inp, torch.bfloat16)
    want = chunk_start_backward(dy, inp.trans, inp.C, 64)
    got = chunk_start_backward(dy, inp.trans, projection_band(inp.C), 64)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


def test_matches_autograd_through_the_forward() -> None:
    """``dzstart`` is the cotangent ``autograd`` sends into the forward's ``zstart``.

    The reference backward is a separate derivation from the forward it
    differentiates, so a shared error in both would pass
    :func:`test_chunk_start_matches_reference`. Differentiating the real forward
    with ``autograd`` shares nothing with either. The ragged shape is used because
    it is where the truncation to ``T`` decides which rows of the cotangent are
    zero.
    """
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16, requires_grad=True)
    dy = _cotangent(inp, torch.bfloat16)
    fw = chunked_forward(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        64,
        z0=None if inp.z0 is None else inp.z0.double(),
        b_prev=None if inp.b_prev is None else inp.b_prev.double(),
        u_prev=None if inp.u_prev is None else inp.u_prev.double(),
    )
    (want,) = torch.autograd.grad(fw.y, fw.zstart, dy.double())
    got = chunk_start_backward(dy, inp.trans, inp.C, 64)
    torch.cuda.synchronize()
    assert_max_rel(
        got, want.flatten(-2, -1), BOUNDS[torch.bfloat16], "cute-chunk-start[autograd]"
    )


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The widest legal tiles are the binding case: ``MAX_CHUNK`` doubles every
    per-token tile and is the only shape that can overflow the carveout.
    """
    nbytes = start_smem_bytes(MAX_CHUNK, 64, 96)
    assert nbytes <= smem_capacity()
    # The K extent is the whole chunk, so the operand tiles scale with L and the
    # widest legal shape is resident once per SM. Four blocks per SM at the shape
    # the class is declared against is what keeps the DRAM pipe fed there.
    assert 4 * start_smem_bytes(64, 48, 48) <= smem_capacity()
    assert start_smem_bytes(64, 16, 48) < nbytes


def test_rejects_a_shape_the_carveout_cannot_hold() -> None:
    """An oversized state width is refused on the host, not silently clipped.

    ``3N`` is legal at any multiple of 48, and the rotated readout tile grows with
    it, so the largest legal ``3N`` at ``MAX_CHUNK`` is set by the carveout and
    nothing else checks it.
    """
    inp = _make(1, 1, MAX_CHUNK, 16, 112, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    with pytest.raises(ValueError, match="chunk_start_bwd"):
        chunk_start_backward(dy, inp.trans, inp.C, MAX_CHUNK)


def _ok() -> dict[str, torch.Tensor]:
    """A legal operand set for the rejection table to perturb."""
    inp = _make(2, 2, 128, 16, 16, torch.bfloat16)
    return {"dy": _cotangent(inp, torch.bfloat16), "trans": inp.trans, "C": inp.C}


Operands = dict[str, torch.Tensor]

REJECTIONS: list[tuple[Callable[[Operands], None], type[Exception], str]] = [
    (lambda a: a.update(dy=a["dy"].cpu()), ValueError, "dy must be on a CUDA device"),
    # An arbitrary pitch between rows is legal on ``C``; a gap inside one is not,
    # since a thread reads the three components of its 3-vector as adjacent
    # elements. The other pitched rejections belong to the shared rule and are
    # covered against it; this one says ``C`` reaches that rule at all.
    (lambda a: a.update(C=a["C"][..., ::2]), ValueError, "C must have unit stride"),
    (lambda a: a.update(dy=a["dy"].float()), TypeError, "dy has dtype"),
    (lambda a: a.update(C=a["C"].half()), TypeError, "one dtype per call"),
    (lambda a: a.update(trans=a["trans"].bfloat16()), ValueError, "trans must be"),
    (
        lambda a: a.update(dy=a["dy"][:, :, 0].contiguous()),
        ValueError,
        r"dy must be \(B,H,T,P\)",
    ),
    (lambda a: a.update(trans=a["trans"][..., :3].contiguous()), ValueError, "trans"),
    (
        lambda a: a.update(C=a["C"][:, :, 0].contiguous()),
        ValueError,
        r"C must be \(B,G,T,3N\)",
    ),
    # Any G dividing H is legal. What is left to refuse is a G that does not
    # divide, which would send some head past the end of C.
    (
        lambda a: a.update(C=a["C"][:, :1].repeat(1, 3, 1, 1)),
        ValueError,
        "does not divide H=2",
    ),
]
"""Every ``raise`` reachable from the public path bar the two extents and the
carveout, named by its message.

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
        chunk_start_backward(args["dy"], args["trans"], args["C"], 64)


@pytest.mark.parametrize(
    ("chunk", "lanes", "match"),
    [
        (24, 16, "multiple of 16"),
        # 3N = 24 rather than any smaller miss: the pitch of a contiguous ``C`` is
        # ``3N``, so a 3N that is not a multiple of 8 elements is refused by the
        # pitched rule before the extent rule is reached. 24 clears alignment and
        # still fails the atom's N extent.
        (64, 8, "3N must be"),
    ],
)
def test_rejects_an_extent_the_atom_cannot_cover(
    chunk: int, lanes: int, match: str
) -> None:
    """The fix for an illegal extent is the shape, never a padding path.

    ``L`` sets the MMA K extent and ``3N`` sets its N extent, and neither mode
    admits a tail.
    """
    inp = _make(2, 2, 128, 16, lanes, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    with pytest.raises(ValueError, match=match):
        chunk_start_backward(dy, inp.trans, inp.C, chunk)
