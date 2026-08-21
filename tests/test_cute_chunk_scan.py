"""Chunk scan against the float64 reference.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors. So the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic.

The parity cases feed the reference's own chunk-start state, upcast to float32, so
what they measure is this kernel alone. One further case runs the real producers --
the chunk increment and the state recurrence -- into it, which is what fixes that
the three kernels agree about the layout and the frame of that state.
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
from slinoss.ops.so3ssd.cute.fwd.chunk_scan import chunk_scan_forward, scan_smem_bytes
from slinoss.ops.so3ssd.cute.fwd.increment_passing import increment_passing_forward
from tests.conftest import (
    ScanInputs,
    assert_max_rel,
    make_inputs,
    projection_band,
)

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it the decay mask underflows a few tokens in and most of
# the score contributes nothing, which would leave the diagonal GEMM tested on a
# narrow band next to the diagonal.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, streaming, dtype).
#
# One case per distinct path through this kernel:
#
# - the score slice count, two at L=64, four at L=128, one at L=16, which is the
#   only loop whose trip count changes what is staged;
# - ``L`` on the MMA tile against ``L`` under it, which selects between an exact M
#   mode and one rounded up by four;
# - a ragged tail, where the store predicate is the only thing keeping the last
#   chunk inside the sequence;
# - the streaming split, the only way the previous tap reaches a token before the
#   sequence;
# - two ``N``, because the lane stride loop is per-thread;
# - both operand dtypes, because each is a different MMA atom.
SHAPES = [
    pytest.param(
        2, 2, 256, 64, 48, 16, True, torch.bfloat16, id="two-slices-streaming"
    ),
    pytest.param(2, 2, 200, 64, 64, 32, False, torch.bfloat16, id="ragged-wide"),
    pytest.param(1, 1, 256, MAX_CHUNK, 16, 16, True, torch.bfloat16, id="four-slices"),
    pytest.param(2, 2, 48, 16, 16, 16, True, torch.float16, id="padded-m-fp16"),
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


def _reference(inp: ScanInputs, chunk: int):
    """The float64 authority on the same cast operands."""
    return chunked_forward(
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


# The kernel narrows the rotated readout, both rotated forcings, the chunk-start
# state and the masked score to the operand dtype, and sums L of them in float32.
# The bound is that dtype's unit roundoff against the largest entry of the result.
# Worst measured 3.8e-3 at bfloat16 and 3.4e-4 at float16, a ratio of eleven that
# tracks the ratio of the two significands, so each dtype carries its own bound
# rather than both sharing a figure loose enough for the wider one. The bfloat16
# figure is one half-ulp of bfloat16, 2^-8 = 3.9e-3: the five narrowings do not
# compound, because each feeds a different term of a float32 sum.
BOUNDS = {torch.bfloat16: 6e-3, torch.float16: 8e-4}


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "streaming", "dtype"), SHAPES
)
def test_chunk_scan_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    streaming: bool,
    dtype: torch.dtype,
) -> None:
    """The output matches a float64 forward given the same chunk-start state."""
    inp = _make(bsz, heads, seqlen, rows, lanes, streaming, dtype)
    ref = _reference(inp, chunk)
    zstart = ref.zstart.flatten(-2, -1).to(dtype).contiguous()
    out = chunk_scan_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        inp.C,
        zstart,
        chunk,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
    )
    torch.cuda.synchronize()

    assert out.shape == inp.U.shape and out.dtype is dtype
    tag = (
        f"cute-scan[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    assert_max_rel(out, ref.y, BOUNDS[dtype], f"{tag}.y")
    # A comparison against zeros passes whatever the GEMMs do.
    assert torch.count_nonzero(out) > 0


def test_composes_with_the_real_chunk_start_state() -> None:
    """The two forward kernels agree about the state between them.

    The parity cases above hand the kernel the reference's state, so they would
    still pass if the prologue emitted its state in a different frame, if the
    recurrence wrote a different chunk index, or if the lane order disagreed. Run
    end to end, with more than one chunk and a nonzero initial state, none of those
    survive. The producer is the fused prologue, which is what the forward launches
    and what stores the state at the width this kernel reads.
    """
    inp = _make(2, 2, 200, 48, 16, True, torch.bfloat16)
    chunk = 64
    ref = _reference(inp, chunk)
    assert inp.z0 is not None
    passed = increment_passing_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        chunk,
        z0=inp.z0,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
    )
    out = chunk_scan_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        inp.C,
        passed.zstart,
        chunk,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
    )
    torch.cuda.synchronize()
    # Same bound as bfloat16 parity. The two producers add their own error, but the
    # recurrence that produces the state carries float32 and only its store narrows,
    # to the width this kernel would have narrowed it to anyway, and the chunk decay
    # attenuates it. So the residual stays inside one bfloat16 half-ulp: 2.9e-3
    # measured against 3.8e-3 for this kernel alone.
    assert_max_rel(out, ref.y, BOUNDS[torch.bfloat16], "cute-scan-composed.y")


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The widest legal tiles are the binding case: ``MAX_CHUNK`` doubles every
    per-token tile and is the only shape that can overflow the carveout.
    """
    nbytes = scan_smem_bytes(MAX_CHUNK, 64, 96)
    assert nbytes <= smem_capacity()
    assert scan_smem_bytes(64, 48, 48) < nbytes


def test_reads_bands_of_the_fused_projection() -> None:
    """``B`` and ``C`` both ship pitched, and both are indexed rather than copied.

    One projection GEMM feeds every consumer, so neither vector operand is a buffer
    of its own. Recovering contiguity would be the staging copy the layout contract
    exists to refuse. One test covers both because both reach the same rotate-and-
    stage path, so a break at either operand fails it. Nothing about the arithmetic
    changes, so the two layouts must agree bit for bit rather than to a tolerance.
    """
    inp = make_inputs(
        bsz=2,
        heads=4,
        groups=2,
        seqlen=128,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
        streaming=False,
        u_dtype=torch.bfloat16,
        bc_dtype=torch.bfloat16,
    )
    zstart = _reference(inp, 64).zstart.flatten(-2, -1).bfloat16().contiguous()
    want = chunk_scan_forward(inp.U, inp.trans, inp.K, inp.B, inp.C, zstart, 64)
    got = chunk_scan_forward(
        inp.U,
        inp.trans,
        inp.K,
        projection_band(inp.B),
        projection_band(inp.C),
        zstart,
        64,
    )
    torch.cuda.synchronize()
    assert torch.equal(got, want)


Operands = dict[str, torch.Tensor]

# The shared host guard is covered through the chunk increment. What is reachable
# only here is the second readout vector, ``P`` as an N extent, and the shape and
# dtype of the chunk-start state.
REJECTIONS: list[tuple[Callable[[Operands], None], type[Exception], str]] = [
    (lambda a: a.update(C=a["C"].float()), TypeError, "C has dtype"),
    (lambda a: a.update(C=a["C"][..., :48].contiguous()), ValueError, "C must be"),
    (
        lambda a: a.update(zstart=a["zstart"][:, :, :1].contiguous()),
        ValueError,
        "zstart",
    ),
    (lambda a: a.update(zstart=a["zstart"].float()), ValueError, "zstart must be"),
]


@pytest.mark.parametrize(("mutate", "exc", "match"), REJECTIONS)
def test_rejects_a_bad_operand(
    mutate: Callable[[Operands], None],
    exc: type[Exception],
    match: str,
) -> None:
    """A violation is refused on the host, not repacked and not launched."""
    inp = _make(2, 2, 128, 16, 32, True, torch.bfloat16)
    args = {
        "U": inp.U,
        "trans": inp.trans,
        "K": inp.K,
        "B": inp.B,
        "C": inp.C,
        "zstart": torch.zeros(2, 2, 2, 16, 96, dtype=torch.bfloat16, device="cuda"),
    }
    mutate(args)
    with pytest.raises(exc, match=match):
        chunk_scan_forward(
            args["U"],
            args["trans"],
            args["K"],
            args["B"],
            args["C"],
            args["zstart"],
            64,
        )


def test_rejects_a_head_width_the_atom_cannot_cover() -> None:
    """``P`` is an N extent here, unlike in the increment, where it is free.

    The fix for an illegal extent is the shape, never a padding path.
    """
    inp = _make(2, 2, 128, 24, 16, False, torch.bfloat16)
    zstart = torch.zeros(2, 2, 2, 24, 48, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="P must be a multiple of 16"):
        chunk_scan_forward(inp.U, inp.trans, inp.K, inp.B, inp.C, zstart, 64)
