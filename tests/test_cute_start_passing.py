"""Fused chunk-start cotangent and reverse recurrence.

Two authorities, because the fusion makes two claims. A float64
``chunked_backward`` says the kernel computes ``dinc`` and ``dz0``. The unfused
pair says the fusion is not an approximation of itself: every arithmetic step is
the same in the same order and only the storage of the intermediate moved, so the
two agree bit for bit.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors, so the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic.

``cquat`` and ``cscale`` are read off the reference forward rather than fabricated:
a ``randn`` rotation would not test the reverse frame change, which is the only
nontrivial thing the recurrence does. They are functions of ``trans`` alone, so
the activation width does not enter them.
"""

import inspect

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss._cute import smem_capacity
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import ChunkedForward, chunked_backward, chunked_forward
from slinoss.ops.so3ssd.cute.bwd.chunk_start import (
    chunk_start_backward,
    start_smem_bytes,
)
from slinoss.ops.so3ssd.cute.bwd.start_passing import (
    RESIDENT_MAX,
    SPLIT,
    fold_smem_bytes,
    start_passing_backward,
)
from slinoss.ops.so3ssd.cute.bwd.state_passing import state_passing_backward
from slinoss.ops.so3ssd.cute.common import WARPS
from slinoss.ops.so3ssd.cute.mma import WARPS_WIDE
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it exp(2*lp) underflows a few tokens into the chunk, the
# chunk decay is zero to float32, and the recurrence under test is the identity on
# the readout cotangent.
LS_BIAS = -4.0

CHUNK = 64

# (bsz, heads, seqlen, rows, lanes, groups, dtype, with_dstate).
#
# One case per distinct path through the fused kernel:
#
# - a ragged tail with three chunks at a ``P`` off the MMA tile, which selects the
#   predicated store into the shared tile;
# - one chunk at the smallest legal extents, with the zero seed;
# - five lane bands at ``P`` exactly on the MMA tile, which selects the vectorized
#   store and is the only case where the band offset is neither zero nor the whole
#   state;
# - two bands with a group shared by two heads, which crosses the band offset with
#   the group divide;
# - the other operand dtype, because each is a different MMA atom.
SHAPES = [
    pytest.param(2, 2, 200, 48, 16, None, torch.bfloat16, True, id="ragged-three"),
    pytest.param(2, 2, 64, 16, 16, None, torch.bfloat16, False, id="single-chunk"),
    pytest.param(1, 1, 256, 64, 80, None, torch.bfloat16, True, id="five-bands"),
    pytest.param(2, 4, 128, 16, 32, 2, torch.bfloat16, True, id="grouped-two-bands"),
    pytest.param(2, 2, 128, 16, 16, None, torch.float16, True, id="fp16"),
]

# The chunk-start GEMM rounds both operands to the activation dtype on their way
# into shared memory, which the reference does not do, so the bound is that dtype's
# epsilon carried through an L-long sum. The recurrence adds float32 rounding over
# ``C`` transitions of norm at most one (I1), which is three orders below that.
# Worst measured over the sweep below, from --tolerance-report on an A6000: 3.8e-3
# at bfloat16 and 5.6e-4 at float16, 38% and 28% of their bounds.
BOUNDS = {torch.bfloat16: 1e-2, torch.float16: 2e-3}


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
    groups: int | None = None,
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
        with_state=True,
        streaming=False,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


def _double(inp: ScanInputs) -> ScanInputs:
    """The same operands in float64. Exact, so both paths see one problem."""
    return ScanInputs(
        U=inp.U.double(),
        trans=inp.trans.double(),
        K=inp.K.double(),
        B=inp.B.double(),
        C=inp.C.double(),
        z0=None if inp.z0 is None else inp.z0.double(),
        b_prev=None,
        u_prev=None,
    )


def _seeds(
    inp: ScanInputs, rows: int, lanes: int, with_dstate: bool, dtype: torch.dtype
) -> tuple[Tensor, Tensor | None]:
    """Cotangents of ``y`` and of the final state, or ``None`` where absent.

    Drawn rather than read off a pipeline record: upstream of an output cotangent
    is a loss.
    """
    gen = torch.Generator(device="cuda").manual_seed(7)
    bsz, heads, seqlen = (int(n) for n in inp.U.shape[:3])
    dy = torch.randn(
        bsz, heads, seqlen, rows, generator=gen, dtype=torch.float32, device="cuda"
    )
    dstate = None
    if with_dstate:
        dstate = torch.randn(
            bsz,
            heads,
            rows,
            3 * lanes,
            generator=gen,
            dtype=torch.float32,
            device="cuda",
        )
    return dy.to(dtype), dstate


def _transition(fwd: ChunkedForward) -> tuple[Tensor, Tensor]:
    """``(cquat, cscale)``: the chunk-end quaternion prefix and the chunk decay."""
    return (
        fwd.qprefix[..., -1, :].float().contiguous(),
        torch.exp(2.0 * fwd.lprefix[..., -1]).float().contiguous(),
    )


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "rows", "lanes", "groups", "dtype", "with_dstate"),
    SHAPES,
)
def test_fused_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    groups: int | None,
    dtype: torch.dtype,
    with_dstate: bool,
) -> None:
    """``dinc`` and ``dz0`` match a float64 ``chunked_backward`` end to end."""
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype, groups)
    wide = _double(inp)
    dy, dstate = _seeds(inp, rows, lanes, with_dstate, dtype)
    ref = chunked_backward(
        dy.double(),
        None if dstate is None else dstate.double(),
        None,
        None,
        *wide.args(),
        CHUNK,
        **wide.kw(),
    )
    cquat, cscale = _transition(chunked_forward(*wide.args(), CHUNK, **wide.kw()))
    out = start_passing_backward(dy, inp.trans, inp.C, cquat, cscale, CHUNK, dstate)
    torch.cuda.synchronize()

    tag = (
        f"cute-start-passing[{bsz}x{heads}x{seqlen}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    assert out.dinc.shape == (bsz, heads, -(-seqlen // CHUNK), rows, 3 * lanes)
    assert out.dz0.shape == (bsz, heads, rows, 3 * lanes)
    assert_max_rel(out.dinc, ref.dinc, BOUNDS[dtype], f"{tag}.dinc")
    assert_max_rel(out.dz0, ref.dz0, BOUNDS[dtype], f"{tag}.dz0")
    # A comparison against zeros passes whatever the recurrence does, so the
    # residual alone is not evidence. The reference decides which side is empty:
    # ``dinc`` of a single chunk under a zero seed is the seed itself, and
    # demanding a nonzero there would fail a kernel that is right.
    for got, want, field in ((out.dinc, ref.dinc, "dinc"), (out.dz0, ref.dz0, "dz0")):
        assert (torch.count_nonzero(got) > 0) == (torch.count_nonzero(want) > 0), (
            f"{tag}.{field} is empty where the reference is not, or the reverse"
        )


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "rows", "lanes", "groups", "dtype", "with_dstate"),
    SHAPES,
)
def test_fused_matches_the_unfused_pair(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    groups: int | None,
    dtype: torch.dtype,
    with_dstate: bool,
) -> None:
    """The fusion changes where the intermediate lives and nothing else.

    Both paths run the same GEMM over the same staged tiles and the same reverse
    recurrence in the same order; the fused one keeps ``dzstart`` in shared memory
    instead of writing it to DRAM and reading it back. Both recurrences carry
    float32 and the fused store narrows the same value the unfused pair leaves wide,
    so the two results are equal bit for bit under one rounding and a tolerance
    would hide a real divergence in the band indexing.
    """
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype, groups)
    wide = _double(inp)
    dy, dstate = _seeds(inp, rows, lanes, with_dstate, dtype)
    cquat, cscale = _transition(chunked_forward(*wide.args(), CHUNK, **wide.kw()))

    dzstart = chunk_start_backward(dy, inp.trans, inp.C, CHUNK)
    want = state_passing_backward(dzstart, cquat, cscale, dstate)
    got = start_passing_backward(dy, inp.trans, inp.C, cquat, cscale, CHUNK, dstate)
    torch.cuda.synchronize()

    assert got.dinc.dtype is dtype
    assert torch.equal(got.dinc, want.dinc.to(dtype))
    assert torch.equal(got.dz0, want.dz0)


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "rows", "lanes", "groups", "dtype", "with_dstate"),
    [case for case in SHAPES if case.id in ("ragged-three", "five-bands")],
)
def test_narrow_block_matches_the_shipped_width(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    groups: int | None,
    dtype: torch.dtype,
    with_dstate: bool,
) -> None:
    """The block width moves the tile's N mode and no arithmetic.

    Warps past the first four are absorbed into the N atoms, so each accumulator
    element is still one K-long dot product summed in the same order and each
    3-vector's recurrence still runs on one thread. Only which thread owns which
    element changes, so the two widths agree bit for bit and a tolerance would hide
    a real divergence in the partition.

    Two shapes, one per store path: a ``P`` off the MMA tile selects the predicated
    store, and a ``P`` on it selects the vectorized one, whose partition is the
    width's own.

    The shipped width is the wide one, so the narrow arm is what this passes
    explicitly. The default is asserted here rather than in the driver's test,
    because a default that drifted back would leave the operator paying 65 us a call
    with every assertion still green.
    """
    assert (
        inspect.signature(start_passing_backward).parameters["warps"].default
        == WARPS_WIDE
    )
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype, groups)
    wide = _double(inp)
    dy, dstate = _seeds(inp, rows, lanes, with_dstate, dtype)
    cquat, cscale = _transition(chunked_forward(*wide.args(), CHUNK, **wide.kw()))

    args = (dy, inp.trans, inp.C, cquat, cscale, CHUNK, dstate)
    want = start_passing_backward(*args)
    got = start_passing_backward(*args, warps=WARPS)
    torch.cuda.synchronize()

    assert torch.equal(got.dinc, want.dinc)
    assert torch.equal(got.dz0, want.dz0)


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The band width is fixed, so the binding extent is ``L``: every staging tile is
    proportional to it and the state tile is not. The residency at the shape the
    class is declared against is what this kernel is short of, so the budget has to
    leave room for :data:`RESIDENT_MAX` blocks there.
    """
    assert fold_smem_bytes(MAX_CHUNK, 64, SPLIT) <= smem_capacity()
    assert RESIDENT_MAX * fold_smem_bytes(64, 64, SPLIT) <= smem_capacity()
    # The recurrence's tile costs nothing at that shape: it is smaller than the two
    # operand tiles it is overlaid on, so the fused block is the unfused GEMM's
    # footprint and the residency comes for free.
    assert fold_smem_bytes(64, 64, SPLIT) == start_smem_bytes(64, 64, SPLIT)
    # A short chunk inverts that, and the region has to follow the larger of the two
    # rather than the operands alone.
    assert fold_smem_bytes(16, 64, SPLIT) > start_smem_bytes(16, 64, SPLIT)


def test_rejects_a_band_the_launch_cannot_cover() -> None:
    """A band width is refused on the host, not rounded to one that fits.

    A band that is not a multiple of 3 splits a 3-vector across two blocks, whose
    recurrences would then rotate parts of one state independently.
    """
    inp = _make(2, 2, 128, 16, 16, torch.bfloat16)
    wide = _double(inp)
    dy, dstate = _seeds(inp, 16, 16, True, torch.bfloat16)
    cquat, cscale = _transition(chunked_forward(*wide.args(), CHUNK, **wide.kw()))
    with pytest.raises(ValueError, match="span must divide"):
        start_passing_backward(
            dy, inp.trans, inp.C, cquat, cscale, CHUNK, dstate, span=32
        )
