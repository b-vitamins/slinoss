"""``dB``, ``dC``, the vector carry, and the transition parameters.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors. So the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic: the float32 table, the float32 accumulators, and the one narrowing of
each operand on its way into shared memory.

Four float32 inputs come from the reference rather than from a generator.
``dinc`` and ``zstart`` are what the two stages ahead of this one hand over, and
``dlogp``, ``dchunk_rot`` and ``dchunk_scale`` are the chunk-input stage's three
closing cotangents, which this kernel consumes and never recomputes. A fabricated
set would not compose the chunks.

``dB`` here is not the operator's ``dB``. The chunk-boundary rows carry the
current tap alone, because the previous tap at those rows belongs to the next
chunk's first token and
:func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds it there.
:func:`_expected_db` states that contract as a subtraction off the reference, so a
kernel that wrote those rows itself fails here rather than double-counting
downstream.

``dtrans`` and ``dK`` are complete: every producer of the rotation, log-scale and
tap cotangents is inside this kernel, ``dlogp`` included, so both are checked
against the reference's whole gradient.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable
from typing import NamedTuple

from torch import Tensor

from slinoss._cute import smem_capacity
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import chunked_backward, chunked_forward
from slinoss.ops.so3ssd.cute.bwd.chunk_vector import (
    ChunkVectorBwd,
    chunk_vector_backward,
    vblock,
    vector_smem_bytes,
)
from slinoss.ops.so3ssd.reference import from_heads
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it both ``exp(2*(lp_t - lp_r))`` and the increment
# weight underflow a few tokens into the chunk, and every earlier source token
# contributes nothing: the GEMMs would be tested on the chunk's last handful of
# tokens and the increment terms on almost none.
LS_BIAS = -4.0

# (bsz, heads, seqlen, chunk, rows, lanes, dtype).
#
# One case per distinct path through this kernel:
#
# - a ragged tail, where the staging passes zero the rows past ``valid`` and the
#   chunk transition closes on a token that is not ``L-1``, together with four
#   chunks so that the reverse recurrence's carry is not the same tensor twice;
# - a single chunk at the smallest legal ``P``, which is also the smallest ``N``:
#   there the transition closes with no predecessor chunk and ``dB``'s boundary
#   correction is empty;
# - two ``N`` past one lane tile, because every lane-indexed reduction strides over
#   ``N`` and one that dropped the stride passes at a single ``N``. That is also the
#   only shape here with a second lane tile, so it is where the two sums that cross
#   lanes are accumulated into rather than stored;
# - both operand dtypes, because each is a different MMA atom.
#
# ``MAX_CHUNK`` is absent because this kernel refuses it at every ``P``: the
# arena holds a chunk-square score fragment where the neighbouring kernels hold a
# row band. :func:`test_rejects_a_shape_the_carveout_cannot_hold` is that case.
SHAPES = [
    pytest.param(2, 2, 200, 64, 48, 16, torch.bfloat16, id="ragged-four-chunks"),
    pytest.param(2, 2, 64, 64, 16, 16, torch.bfloat16, id="single-chunk"),
    pytest.param(1, 1, 128, 64, 32, 32, torch.bfloat16, id="two-lanes"),
    pytest.param(2, 2, 128, 32, 16, 16, torch.float16, id="fp16"),
]


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
    groups: int | None = None,
    streaming: bool = True,
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
        streaming=streaming,
        w_scale=2.0,
        ls_bias=LS_BIAS,
        u_dtype=dtype,
        bc_dtype=dtype,
        requires_grad=requires_grad,
    )


def _cotangent(inp: ScanInputs, dtype: torch.dtype, seed: int = 17) -> Tensor:
    """``dy`` in the operand dtype. A loss gradient, not an intermediate."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    raw = torch.randn(
        inp.U.shape, generator=gen, dtype=torch.float32, device=inp.U.device
    )
    return raw.to(dtype)


def _dstate(inp: ScanInputs, seed: int = 23) -> Tensor:
    """``dstate`` in float64. A loss gradient, like ``dy``, not an intermediate.

    Every shape below passes one. The reverse chunk recurrence starts at ``dstate``,
    so without it the last chunk's increment cotangent is identically zero, and at a
    single chunk that is the only chunk: the tap gradients would then carry the
    diagonal term alone.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bsz, heads, _, rows = inp.U.shape
    return torch.randn(
        (int(bsz), int(heads), int(rows), int(inp.B.shape[-1])),
        generator=gen,
        dtype=torch.float64,
        device=inp.U.device,
    )


def _dbl(t: Tensor | None) -> Tensor | None:
    """Exact upcast, or ``None`` for an absent operand."""
    return None if t is None else t.double()


class Oracle(NamedTuple):
    """The float64 values the five outputs are checked against, and the five
    float32 inputs the kernel takes from the stages ahead of it.

    Attributes:
        dB: ``(B,G,T,3N)``, already corrected for the boundary rows this kernel
            does not write.
        dC: ``(B,G,T,3N)``.
        carry_b: ``(B,G,C,3N)``.
        dtrans: ``(B,H,T,4)``.
        dK: ``(B,H,T,2,4)``.
        dinc: ``(B,H,C,P,3N)`` float32, the recurrence carry in the global frame.
        zstart: ``(B,H,C,P,3N)`` float32, the state entering each chunk.
        dlogp: ``(B,H,C,L)`` float32, the chunk-input stage's log-scale half.
        dchunk_rot: ``(B,H,C,3,3)`` float32.
        dchunk_scale: ``(B,H,C)`` float32.
    """

    dB: Tensor
    dC: Tensor
    carry_b: Tensor
    dtrans: Tensor
    dK: Tensor
    dinc: Tensor
    zstart: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor


def _expected_db(dB: Tensor, carry_b: Tensor, chunk: int) -> Tensor:
    """The reference ``dB`` with the boundary contribution taken back out.

    The reference sums ``db(t) + dbshift(t+1)`` over the whole sequence. At
    ``t = cL-1`` the second term is chunk ``c``'s ``carry_b``, which lives in
    another block and is added by the boundary kernel, so it is subtracted here.
    """
    chunks = int(carry_b.shape[2])
    want = dB.clone()
    if chunks > 1:
        rows = torch.arange(1, chunks, device=want.device) * chunk - 1
        want[:, :, rows, :] -= carry_b[:, :, 1:, :]
    return want


def _oracle(
    inp: ScanInputs,
    dy: Tensor | None,
    chunk: int,
    dstate: Tensor | None = None,
) -> Oracle:
    """Run the float64 reference forward and backward once each.

    Args:
        inp: The operand set, in the operand dtype.
        dy: Cotangent of ``y``, or ``None`` for the state-only backward.
        chunk: ``L``.
        dstate: Cotangent of the final state, or ``None``.
    """
    U, trans, K, B, C = inp.args()
    z0, b_prev, u_prev = _dbl(inp.z0), _dbl(inp.b_prev), _dbl(inp.u_prev)
    fw = chunked_forward(
        U.double(),
        trans.double(),
        K.double(),
        B.double(),
        C.double(),
        chunk,
        z0=z0,
        b_prev=b_prev,
        u_prev=u_prev,
    )
    ref = chunked_backward(
        _dbl(dy),
        dstate,
        None,
        None,
        U.double(),
        trans.double(),
        K.double(),
        B.double(),
        C.double(),
        chunk,
        z0=z0,
        b_prev=b_prev,
        u_prev=u_prev,
    )
    return Oracle(
        dB=_expected_db(ref.grads.dB, ref.carry_b, chunk),
        dC=ref.grads.dC,
        carry_b=ref.carry_b,
        dtrans=ref.grads.dtrans,
        dK=ref.grads.dK,
        dinc=ref.dinc.float().contiguous(),
        zstart=fw.zstart.flatten(-2, -1).float().contiguous(),
        dlogp=ref.dlogp_scan.float().contiguous(),
        dchunk_rot=ref.dchunk_rot.float().contiguous(),
        dchunk_scale=ref.dchunk_scale.float().contiguous(),
    )


def _run(
    inp: ScanInputs,
    dy: Tensor,
    want: Oracle,
    chunk: int,
    dB: Tensor | None = None,
    dC: Tensor | None = None,
) -> ChunkVectorBwd:
    """One launch, against the float32 inputs the oracle carries."""
    got = chunk_vector_backward(
        dy,
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        inp.C,
        want.dinc,
        want.zstart,
        want.dlogp,
        want.dchunk_rot,
        want.dchunk_scale,
        chunk,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
        dB=dB,
        dC=dC,
    )
    torch.cuda.synchronize()
    return got


# Each output is a different depth of low-precision arithmetic, so each carries its
# own bound rather than all five sharing a figure loose enough for the deepest.
#
# ``dB``, ``dC`` and ``carry_b`` are one low-precision GEMM against a masked score
# that is itself a low-precision GEMM, followed by a float32 matvec against the
# table. ``dK`` reduces those same fragments over the lanes, so it inherits the
# error. ``dtrans`` is the deepest: its rotation half runs the lane reduction
# through the quaternion prefix VJP, whose chunk-local suffix product accumulates
# over ``L`` tokens, and its log-scale half is a suffix sum over the chunk.
#
#              bfloat16   float16
# dB           4.779e-3   3.524e-4
# dC           4.539e-3   4.318e-4
# carry_b      3.422e-3   3.317e-4
# dtrans       2.489e-3   1.426e-4
# dK           3.782e-3   2.035e-4
#
# Worst measured over every shape and every test in this file, read off a
# ``--tolerance-report`` run. Every bound below is about twice its figure. The
# float16 column runs 11 to 17 times tighter than the bfloat16 one, which brackets
# the factor of eight between the two significands, so the two dtypes do not share
# a bound.
BOUNDS: dict[torch.dtype, dict[str, float]] = {
    torch.bfloat16: {
        "dB": 1e-2,
        "dC": 1e-2,
        "carry_b": 7e-3,
        "dtrans": 5e-3,
        "dK": 8e-3,
    },
    torch.float16: {
        "dB": 7e-4,
        "dC": 9e-4,
        "carry_b": 7e-4,
        "dtrans": 3e-4,
        "dK": 4e-4,
    },
}
"""The two vector bounds reach ``1e-2``, twice what the forcing cotangent needs at
the same shapes: those two carry a float32 matvec against the table on top of the
two low-precision GEMMs, and the matvec's operand is a rotation, so all three of its
terms are the same order and none of them cancels."""


def _compare(
    got: ChunkVectorBwd,
    want: Oracle,
    dtype: torch.dtype,
    tag: str,
    zero: tuple[str, ...] = (),
) -> None:
    """Every output against the reference, each under its own bound.

    Args:
        got: One launch's outputs.
        want: The float64 reference.
        dtype: Activation dtype, which selects the bounds.
        tag: Label for the tolerance report.
        zero: Outputs the case under test makes identically zero. A comparison
            against zeros passes whatever the GEMMs did, so every other output is
            held to being nonzero and these are held to being exactly zero.
    """
    for name, bound in BOUNDS[dtype].items():
        mine: Tensor = getattr(got, name)
        theirs: Tensor = getattr(want, name)
        assert mine.shape == theirs.shape, name
        assert_max_rel(mine, theirs, bound, f"{tag}.{name}")
        if name in zero:
            assert torch.count_nonzero(mine) == 0, name
        else:
            assert torch.count_nonzero(mine) > 0, name


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "dtype"), SHAPES
)
def test_chunk_vector_matches_reference(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
) -> None:
    """All five outputs match a float64 reference backward."""
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype)
    dy = _cotangent(inp, dtype)
    want = _oracle(inp, dy, chunk, dstate=_dstate(inp))
    got = _run(inp, dy, want, chunk)

    tag = (
        f"cute-chunk-vector[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    _compare(got, want, dtype, tag)


@pytest.mark.parametrize("groups", [1, 4], ids=["one-group", "group-per-head"])
def test_grouped_vectors_sum_over_their_heads(groups: int) -> None:
    """``dB`` and ``dC`` at group ``g`` are the sum over the heads that read it.

    ``G == 1`` is the only case where every head contributes to one ``B``/``C``
    pair, and it is the case a missing reduction silently passes at ``G == H``. An
    intermediate ``G`` takes the same head fold, so it is not swept. The fold is
    the kernel's own loop rather than a grid axis, so ``G < H`` also runs the arena
    twice over without reallocating it, and the ``dB`` accumulator has to survive
    the head boundary that the last store crosses.

    The shape is also where the budget halves the source-token block, at ``G == 1``
    only, so it is the case that runs two blocks in the tap loop and the case where
    the halving and the second lane tile interact: the forcing sum is per lane tile
    and per group, and it has to be rezeroed on the tile boundary but not on the
    head boundary.
    """
    inp = _make(2, 4, 128, 64, 32, torch.bfloat16, groups=groups)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, f"cute-chunk-vector[G{groups}]")


@pytest.mark.parametrize("groups", [18, 1], ids=["group-per-head", "one-group"])
def test_holds_the_widest_state_the_mixer_configures(groups: int) -> None:
    """``3N = 240`` at the full chunk and the full row count, at both folds.

    The resident set is bounded by one lane tile, not by ``3N``, so this is the same
    launch path every narrower state takes rather than a second one, and the arena
    costs what it costs at ``3N = 48``. The geometry is the one the throughput
    targets are stated at: ``d_head 64``, ``L 64``, ``N 80``, ``H 18``. At ``G == 1``
    the whole fold runs inside the block and the budget halves the source-token
    block as well, so the two sums that cross lanes are accumulated over five lane
    tiles while the forcing sum crosses eighteen heads and two source blocks.
    """
    inp = _make(1, 18, 128, 64, 80, torch.bfloat16, groups=groups)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, f"cute-chunk-vector[3N240/G{groups}]")


def test_the_lane_slot_closure_reproduces_bit_for_bit() -> None:
    """Two launches at a tiled state width agree exactly, not within a bound.

    ``dtrans`` and ``dK`` are sums over lanes and the lane tile is a grid axis, so
    above one tile each tile writes its own slot row and a second launch sums them.
    That closure has no atomics and its order is fixed by the launch geometry, so the
    result is a function of the shape alone. A tolerance cannot tell that apart from
    an order that varies per run, which is what an atomic closure would give, so the
    two runs are compared exactly. The smallest shape with a second lane tile is
    used: the closure either has an order or it does not, and one tile past the first
    is enough to expose it.
    """
    inp = _make(1, 1, 128, 32, 32, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    first = _run(inp, dy, want, 64)
    second = _run(inp, dy, want, 64)
    for name in BOUNDS[torch.bfloat16]:
        assert torch.equal(getattr(first, name), getattr(second, name)), name


def test_without_the_streaming_carry_in() -> None:
    """The absent carry-in is a zero row, not an out-of-range read.

    ``has_prev`` is compile-time, so the two cases are two kernels. Chunk 0's first
    token has no predecessor either way: with the pair absent its previous tap sees
    zeros, and ``carry_b`` at chunk 0 is then a value nobody reads rather than the
    feedback gradient.
    """
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16, streaming=False)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, "cute-chunk-vector[no-stream]")


def test_increment_terms_alone_when_dy_is_zero() -> None:
    """Differentiating the final state alone leaves the increment terms.

    With ``dy`` zero every score term and the whole offset half drop out, and what
    is left is the increment's two tap contractions and the chunk transition. Those
    are a small part of the mixed result's magnitude, so a sign or a weight error in
    them sits under the tolerance of
    :func:`test_chunk_vector_matches_reference` and only shows up once the diagonal
    term is gone.

    ``dC`` vanishes exactly: the readout vector reaches the output through ``y``
    alone, so no increment term touches it, and a kernel that leaked one into it
    fails here.
    """
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16)
    dy = torch.zeros_like(inp.U)
    want = _oracle(inp, None, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, "cute-chunk-vector[no-dy]", zero=("dC",))


def test_matches_autograd_through_the_forward() -> None:
    """``dB``, ``dC`` and the carry are the cotangents ``autograd`` sends in.

    The reference backward is a separate derivation from the forward it
    differentiates, so a shared error in both passes
    :func:`test_chunk_vector_matches_reference`. Differentiating the real forward
    shares nothing with either. ``b`` and ``bshift`` are two chunked views of the
    same input, built independently from it, so their cotangents come back
    separately and the within-chunk shift this kernel owns can be assembled from
    them without any of the reference's algebra. The ragged shape is used because
    it is where the last valid token takes the current tap alone.

    Both differentiated outputs are pushed through together. ``y`` reaches the
    forcing vectors by the diagonal term and the final state reaches them by the
    increment term, and this kernel adds the two.

    ``dtrans`` and ``dK`` are left to the reference: the forward names ``w`` and
    ``tap`` per chunked token, not the operator's per-token parameters, so
    assembling them here would repeat the reference's own unchunking.
    """
    chunk, seqlen = 64, 200
    inp = _make(2, 2, seqlen, 48, 16, torch.bfloat16, requires_grad=True)
    dy = _cotangent(inp, torch.bfloat16)
    dstate = _dstate(inp)
    want = _oracle(inp, dy, chunk, dstate=dstate)
    fw = chunked_forward(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        chunk,
        z0=_dbl(inp.z0),
        b_prev=_dbl(inp.b_prev),
        u_prev=_dbl(inp.u_prev),
    )
    db, dbshift, dc = torch.autograd.grad(
        (fw.y, fw.state),
        (fw.b, fw.bshift, fw.c),
        (dy.double(), dstate.unflatten(-1, (-1, 3))),
    )
    groups = int(inp.B.shape[1])
    shifted = torch.zeros_like(dbshift)
    shifted[:, :, :, :-1, :] = dbshift[:, :, :, 1:, :]
    dB = from_heads((db + shifted).flatten(2, 3)[:, :, :seqlen], groups)
    dC = from_heads(dc.flatten(2, 3)[:, :, :seqlen], groups)
    carry_b = from_heads(dbshift[:, :, :, 0, :], groups)

    got = _run(inp, dy, want, chunk)
    tag = "cute-chunk-vector[autograd]"
    bounds = BOUNDS[torch.bfloat16]
    assert_max_rel(got.dB, dB, bounds["dB"], f"{tag}.dB")
    assert_max_rel(got.dC, dC, bounds["dC"], f"{tag}.dC")
    assert_max_rel(got.carry_b, carry_b, bounds["carry_b"], f"{tag}.carry_b")


def _wide_band(vec: Tensor, fill: float | None) -> tuple[Tensor, Tensor]:
    """``B`` or ``C`` as the mixer hands it over: one column band of a wider tensor.

    The fused projection is token-major, so the view the kernel receives strides by
    the projection width from one token to the next and by ``3N`` from one group to
    the next. The group axis therefore strides less than the axis before it, which a
    band cut out of a head-major buffer would not reproduce. Two bands sit ahead of
    it and one behind, so neither the offset nor the pitch is the one a dedicated
    buffer would have.

    Args:
        vec: The contiguous ``(B,G,T,3N)`` tensor whose geometry the band takes.
        fill: Value for the whole wide tensor, or None to copy ``vec`` into the band
            and leave the rest uninitialized. A filled band is a destination.

    Returns:
        The band, a pitched view of shape ``(B,G,T,3N)``, and the wide tensor it
        came from.
    """
    bsz, groups, seqlen, dim = vec.shape
    wide = torch.empty(bsz, seqlen, groups + 3, dim, dtype=vec.dtype, device=vec.device)
    band = wide[:, :, 2 : 2 + groups]
    if fill is None:
        band.copy_(vec.permute(0, 2, 1, 3))
    else:
        wide.fill_(fill)
    return band.permute(0, 2, 1, 3), wide


def test_reads_a_band_of_the_fused_projection() -> None:
    """``B`` and ``C`` ship pitched, and the kernel indexes the band.

    One projection GEMM feeds every consumer, so neither has a buffer of its own.
    Recovering contiguity would be the staging copy the layout contract exists to
    refuse. Nothing about the arithmetic changes, so the two layouts must agree bit
    for bit rather than within a tolerance. Both are checked in one call: they enter
    on different staging passes, and a pass that used ``3N`` as its token stride
    would read the wrong tokens only for its own operand.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=2)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    base = _run(inp, dy, want, 64)
    pitched = inp._replace(B=_wide_band(inp.B, None)[0], C=_wide_band(inp.C, None)[0])
    got = _run(pitched, dy, want, 64)
    for name in BOUNDS[torch.bfloat16]:
        assert torch.equal(getattr(got, name), getattr(base, name)), name


def test_writes_into_a_band_of_the_projection_gradient() -> None:
    """A supplied ``dB`` and ``dC`` are written in place, in full, and returned.

    The mixer's backward allocates one gradient for its fused projection and hands
    each operator the band its own cotangent belongs in. Allocating here and letting
    the caller assign afterwards would write every gradient byte twice on a
    DRAM-bound path. The buffer arrives filled with NaN rather than zeroed: a row
    the kernel skipped keeps its NaN, and so does a row it accumulated into instead
    of storing. The columns outside the band must still be NaN afterwards, which is
    what says every store is indexed through the runtime strides. The values agree
    with the allocating path bit for bit, the destination being the only difference.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=2)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    base = _run(inp, dy, want, 64)
    dest_b, wide_b = _wide_band(inp.B, float("nan"))
    dest_c, wide_c = _wide_band(inp.C, float("nan"))
    got = _run(inp, dy, want, 64, dB=dest_b, dC=dest_c)

    assert got.dB is dest_b
    assert got.dC is dest_c
    assert not got.dB.isnan().any()
    assert not got.dC.isnan().any()
    assert torch.equal(got.dB, base.dB)
    assert torch.equal(got.dC, base.dC)
    groups = int(inp.B.shape[1])
    for wide in (wide_b, wide_c):
        # The band is columns 2 .. 2+G of a (G+3)-wide token row; what is left is
        # two bands ahead of it and one behind.
        outside = torch.cat([wide[:, :, :2], wide[:, :, 2 + groups :]], dim=2)
        assert outside.isnan().all()


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The block over source tokens is what the budget buys: the full atom tile at
    the shape the DRAM-bound class is declared against, and half of it where the
    row count or the head fold would overflow the carveout. Halving the block
    doubles the ``U`` traffic and leaves the kernel memory bound either way, so the
    choice is the budget's and not the caller's. The widest legal ``L`` does not fit
    at any ``P`` and is refused on the host.

    ``3N`` is not one of the axes the budget scales with. Every tile that spans the
    state width spans one lane tile of it, so the arena is flat in ``3N`` and the
    widest state the mixer configures costs what the narrowest does.
    """
    assert vblock(64, 48, 48, 1) == 64
    assert vector_smem_bytes(64, 48, 48, 1, 64) <= smem_capacity()
    # The default head width halves the block, and the halved budget fits.
    assert vblock(64, 64, 48, 12) == 32
    assert vector_smem_bytes(64, 64, 48, 12, 32) <= smem_capacity()
    assert vector_smem_bytes(64, 64, 48, 12, 64) > smem_capacity()
    # ``MAX_CHUNK`` overflows at the narrowest ``P`` and ``3N``, so no shape saves
    # it and the halved block does not either.
    assert vector_smem_bytes(MAX_CHUNK, 16, 48, 1, vblock(MAX_CHUNK, 16, 48, 1)) > (
        smem_capacity()
    )
    # Three of the four axes scale the arena, so none of those three is free: a
    # budget that ignored one would compare equal on one of these.
    assert vector_smem_bytes(32, 48, 48, 1, 32) < vector_smem_bytes(64, 48, 48, 1, 32)
    assert vector_smem_bytes(64, 16, 48, 1, 32) < vector_smem_bytes(64, 48, 48, 1, 32)
    assert vector_smem_bytes(64, 48, 48, 1, 32) < vector_smem_bytes(64, 48, 48, 1, 64)
    assert vector_smem_bytes(64, 48, 48, 1, 32) < vector_smem_bytes(64, 48, 48, 2, 32)
    # The fourth is flat, and the state width the acceptance geometry asks for fits
    # at the full block and at the widest fold the mixer configures.
    assert vector_smem_bytes(64, 48, 48, 1, 32) == vector_smem_bytes(64, 48, 240, 1, 32)
    assert vblock(64, 64, 240, 1) == 64
    assert vector_smem_bytes(64, 64, 240, 1, 64) <= smem_capacity()
    assert vblock(64, 64, 240, 18) == 32
    assert vector_smem_bytes(64, 64, 240, 18, 32) <= smem_capacity()
    # ``L 64`` is the largest chunk the layout admits, at one resident block at both
    # row counts. The next legal chunk overflows at fold one and the halved block,
    # which is the cheapest corner it has, so no fold or block saves it.
    assert smem_capacity() < 2 * vector_smem_bytes(64, 48, 240, 1, 64)
    assert smem_capacity() < 2 * vector_smem_bytes(64, 64, 240, 18, 32)
    assert vector_smem_bytes(96, 48, 240, 1, vblock(96, 48, 240, 1)) > smem_capacity()
    assert vector_smem_bytes(96, 64, 240, 1, vblock(96, 64, 240, 1)) > smem_capacity()


def test_rejects_a_shape_the_carveout_cannot_hold() -> None:
    """An oversized triple is refused on the host, not silently clipped.

    ``L``, ``P`` and ``3N`` are each legal at the values below and only their
    product overflows the carveout, so nothing but the budget checks this. The
    neighbouring chunk kernels accept ``MAX_CHUNK``; this one holds a chunk-square
    score fragment as well as a row band, and refusing is the contract rather than
    a smaller block.
    """
    inp = _make(1, 1, MAX_CHUNK, 16, 16, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, MAX_CHUNK, dstate=_dstate(inp))
    with pytest.raises(ValueError, match="chunk_vector_bwd"):
        _run(inp, dy, want, MAX_CHUNK)


Operands = dict[str, Tensor | None]


def _t(entry: Tensor | None) -> Tensor:
    """Narrow a table entry the row it appears in requires."""
    assert entry is not None
    return entry


def _ok(chunk: int = 64) -> Operands:
    """A legal call for the rejection table to perturb, keyed by parameter name."""
    inp = _make(2, 2, 128, 16, 16, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, chunk)
    return {
        "dy": dy,
        "U": inp.U,
        "trans": inp.trans,
        "K": inp.K,
        "B": inp.B,
        "C": inp.C,
        "dinc": want.dinc,
        "zstart": want.zstart,
        "dlogp": want.dlogp,
        "dchunk_rot": want.dchunk_rot,
        "dchunk_scale": want.dchunk_scale,
        "u_prev": inp.u_prev,
        "b_prev": inp.b_prev,
        "dB": None,
        "dC": None,
    }


def _token_major(vec: Tensor) -> Tensor:
    """A buffer of the right shape whose trailing axis is the strided one.

    A destination laid out ``(B,G,3N,T)`` and viewed as ``(B,G,T,3N)`` carries the
    shape, dtype and device of its operand and nothing else. Every store would land
    at a stride the kernel never computed.
    """
    bsz, groups, seqlen, dim = vec.shape
    flat = torch.empty(bsz, groups, dim, seqlen, dtype=vec.dtype, device=vec.device)
    return flat.transpose(-1, -2)


def _call(args: Operands, chunk: int = 64) -> ChunkVectorBwd:
    """Apply the public path to a perturbed operand set."""
    return chunk_vector_backward(
        _t(args["dy"]),
        _t(args["U"]),
        _t(args["trans"]),
        _t(args["K"]),
        _t(args["B"]),
        _t(args["C"]),
        _t(args["dinc"]),
        _t(args["zstart"]),
        _t(args["dlogp"]),
        _t(args["dchunk_rot"]),
        _t(args["dchunk_scale"]),
        chunk,
        u_prev=args["u_prev"],
        b_prev=args["b_prev"],
        dB=args["dB"],
        dC=args["dC"],
    )


REJECTIONS: list[tuple[Callable[[Operands], None], type[Exception], str]] = [
    # The rows this kernel owns are the ones no shared rule covers: the cotangent
    # checked against ``U`` rather than against itself, the two float32 state
    # buffers whose chunk count is derived and not passed, the three closing
    # cotangents from the stage ahead, and the streaming pair.
    (
        lambda a: a.update(dy=_t(a["dy"])[:, :, :-1].contiguous()),
        ValueError,
        "dy must be",
    ),
    (
        lambda a: a.update(dinc=_t(a["dinc"])[:, :, :-1].contiguous()),
        ValueError,
        "dinc must be",
    ),
    (
        lambda a: a.update(zstart=_t(a["zstart"])[..., :-16].contiguous()),
        ValueError,
        "zstart must be",
    ),
    (
        lambda a: a.update(dlogp=_t(a["dlogp"])[..., :-16].contiguous()),
        ValueError,
        "dlogp must be",
    ),
    (
        lambda a: a.update(dchunk_rot=_t(a["dchunk_rot"])[..., :-1, :].contiguous()),
        ValueError,
        "dchunk_rot must be",
    ),
    (
        lambda a: a.update(dchunk_scale=_t(a["dchunk_scale"])[:, :, :-1].contiguous()),
        ValueError,
        "dchunk_scale must be",
    ),
    (lambda a: a.update(b_prev=None), ValueError, "supplied together"),
    (lambda a: a.update(u_prev=None), ValueError, "supplied together"),
    # One row per shared rule this kernel is the first to reach on an operand of
    # its own: ``U`` shares a dtype group with ``dy``, ``B`` and ``C``, and every
    # cotangent from a stage ahead is pinned by I4 and read as a dense tile.
    (lambda a: a.update(U=_t(a["U"]).half()), TypeError, "one dtype per call"),
    (lambda a: a.update(dlogp=_t(a["dlogp"]).double()), ValueError, "float32"),
    (
        lambda a: a.update(zstart=_t(a["zstart"]).transpose(-1, -2)),
        ValueError,
        "zstart must be contiguous",
    ),
    # A destination is held to the operand whose gradient it holds, in the order
    # shape, dtype, device, layout.
    (
        lambda a: a.update(dB=torch.empty_like(_t(a["B"]))[:, :, :-1]),
        ValueError,
        "dB must have shape",
    ),
    (
        lambda a: a.update(dC=torch.empty_like(_t(a["C"])).half()),
        TypeError,
        "dC must be",
    ),
    (lambda a: a.update(dB=_token_major(_t(a["B"]))), ValueError, "unit stride"),
]
"""Every ``raise`` this kernel's host path owns, named by its message.

Layout and dtype before shape, because the checks run in that order: a mutation
that changes two things at once must be matched by the first check it reaches."""


@pytest.mark.parametrize(("mutate", "exc", "match"), REJECTIONS)
def test_rejects_a_bad_operand(
    mutate: Callable[[Operands], None],
    exc: type[Exception],
    match: str,
) -> None:
    """A violation is refused on the host, not repacked and not launched.

    Launching against a wrong shape or stride either faults inside CUDA, which
    leaves the context unusable for every later launch in the process, or returns a
    wrong answer with no error at all.
    """
    args = _ok()
    mutate(args)
    with pytest.raises(exc, match=match):
        _call(args)


@pytest.mark.parametrize(
    ("chunk", "rows", "lanes", "groups", "match"),
    [
        # 48 is a multiple of 16 and so clears the atom's K extent; what it fails is
        # the source-token block, which is this kernel's own tiling and not the
        # atom's. It fails only where the budget halves that block, since the full
        # block is the chunk itself, and since the arena is flat in ``3N`` the only
        # axes that reach the halving are ``P`` and the fold. The public config
        # admits only powers of two, so no reachable configuration hits this; the
        # check is what keeps that true.
        (48, 128, 16, 1, "K slice"),
        # ``P`` is the N mode of the two increment contractions here, unlike the
        # kernels where it is only a row count, so it carries the atom's constraint.
        (64, 24, 16, None, "P must be"),
        (64, 16, 8, None, "3N must be"),
    ],
)
def test_rejects_an_extent_the_atom_cannot_cover(
    chunk: int, rows: int, lanes: int, groups: int | None, match: str
) -> None:
    """The fix for an illegal extent is the shape, never a padding path.

    The float32 inputs are zeros here rather than reference output. The reference
    refuses ``P = 24`` and ``3N = 24`` itself, so no pipeline can produce a matching
    set, and the extent check runs before any element is read.
    """
    seqlen = 128
    chunks = -(-seqlen // chunk)
    inp = _make(2, 2, seqlen, rows, lanes, torch.bfloat16, groups=groups)
    dy = _cotangent(inp, torch.bfloat16)
    state = torch.zeros(
        2, 2, chunks, rows, 3 * lanes, dtype=torch.float32, device="cuda"
    )
    dlogp = torch.zeros(2, 2, chunks, chunk, dtype=torch.float32, device="cuda")
    drot = torch.zeros(2, 2, chunks, 3, 3, dtype=torch.float32, device="cuda")
    dscale = torch.zeros(2, 2, chunks, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=match):
        chunk_vector_backward(
            dy,
            inp.U,
            inp.trans,
            inp.K,
            inp.B,
            inp.C,
            state,
            state,
            dlogp,
            drot,
            dscale,
            chunk,
            u_prev=inp.u_prev,
            b_prev=inp.b_prev,
        )
