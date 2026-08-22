"""``dU``, the input carry, the log-scale half and the chunk transition.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors. So the low-precision
rounding of the inputs is inside both paths and the residual is the kernel's own
arithmetic: the float32 table, the float32 accumulators, and the one narrowing of
each operand on its way into shared memory.

The two state buffers, ``dinc`` and ``zstart``, come from the reference at the
operand dtype: they are what the two stages ahead of this one hand over, and a
fabricated pair would not compose the chunks. ``dinc`` is the reverse chunk recurrence's carry in the global
frame, exactly as
:func:`slinoss.ops.so3ssd.cute.bwd.state_passing.state_passing_backward` leaves it.

``dchunk_scale`` contracts that pair and nothing else, so its oracle is
:func:`slinoss.ops.so3ssd.backward.chunk_transition_cotangents` on the two buffers
this file hands over, not on the reference's own float64 copies of them. The
reference's reverse chunk recurrence calls that same function, so float64 autograd
through the reference is still the authority; what changes is which pair the
authority is evaluated at, which is the docstring contract above.

``dU`` here is not the operator's ``dU``. The chunk-boundary rows carry the diagonal
term alone, because the shifted term at those rows belongs to the next chunk's first
token and
:func:`slinoss.ops.so3ssd.cute.bwd.boundary.boundary_backward` adds it there.
:func:`_expected_du` states that contract as a subtraction off the reference, so a
kernel that wrote those rows itself fails here rather than double-counting downstream.

``dlogp`` is the reference's ``dlogp_scan``, one of two halves of the log-scale
cotangent. The other half has another producer, so no autograd quantity names this
one alone and its oracle is the reference backward, which ``test_backward`` holds to
float64 ``gradcheck``.

The oracle is :func:`slinoss.ops.so3ssd.backward.chunked_backward_fused`, the
one-tap factorization this kernel implements. The two factorizations agree on every
operator gradient and disagree on ``dlogp_scan`` field for field: the fused column
gives ``ls`` a route that does not pass through the log-scale prefix, so mass moves
off ``dlogp`` and onto ``dls_step``, which
:func:`slinoss.ops.so3ssd.cute.bwd.chunk_vector.chunk_vector_backward` supplies. A
two-tap oracle fails a correct kernel here.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable
from typing import NamedTuple

from torch import Tensor

from slinoss._cute import smem_capacity, smem_residency
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import (
    ChunkedBackward,
    chunk_transition_cotangents,
    chunked_backward_fused,
    chunked_forward,
)
from slinoss.ops.so3ssd.cute.bwd.chunk_input import (
    LANE_MULTIPLE,
    LANE_THREADS,
    RESIDENT_MIN,
    ChunkInputBwd,
    chunk_input_backward,
    input_smem_bytes,
    input_threads,
    lane_threads,
    lblock,
)
from slinoss.ops.so3ssd.cute.common import THREADS
from slinoss.ops.so3ssd.cute.mma import THREADS_WIDE
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
#   last valid token takes ``du`` alone, together with four chunks so that the
#   reverse recurrence's carry is not the same tensor twice;
# - a single chunk at the smallest legal ``P``, which is also the smallest ``N``:
#   there the chunk transition closes on token ``L-1`` with no predecessor chunk;
# - ``MAX_CHUNK`` with ``B = H = 1``, the deepest target-token slice loop and the
#   smallest grid;
# - two ``N``, because every lane-indexed reduction in the epilogue strides over
#   ``N`` and one that dropped the stride passes at a single ``N``;
# - both operand dtypes, because each is a different MMA atom.
#
# ``L`` and ``P`` do not interact: ``L`` is the source-token M mode and the slice
# extent, ``P`` is a row count in three of the five contractions, so they are swept
# and not crossed.
SHAPES = [
    pytest.param(2, 2, 200, 64, 48, 32, torch.bfloat16, id="ragged-four-chunks"),
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
    single chunk that is the only chunk: the transition outputs would then be exact
    zeros on both sides and nothing about them would be tested.
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
    """The float64 values the five outputs are checked against, and the two state
    buffers the kernel takes from the stages ahead of it.

    Attributes:
        dU: ``(B,H,T,P)``, already corrected for the boundary rows this kernel
            does not write.
        carry_u: ``(B,H,C,P)``.
        dlogp: ``(B,H,C,L)``, the reference's ``dlogp_scan``.
        dchunk_rot: ``(B,H,C,3,3)``.
        dchunk_scale: ``(B,H,C)``, evaluated at ``dinc`` and ``zstart`` below.
        dinc: ``(B,H,C,P,3N)`` at the operand dtype, the recurrence carry in the
            global frame.
        zstart: ``(B,H,C,P,3N)`` at the operand dtype, the state entering each chunk.
    """

    dU: Tensor
    carry_u: Tensor
    dlogp: Tensor
    dchunk_rot: Tensor
    dchunk_scale: Tensor
    dinc: Tensor
    zstart: Tensor


def _expected_du(dU: Tensor, carry_u: Tensor, chunk: int) -> Tensor:
    """The reference ``dU`` with the boundary contribution taken back out.

    The reference sums ``du(t) + dushift(t+1)`` over the whole sequence. At
    ``t = cL-1`` the second term is chunk ``c``'s ``carry_u``, which lives in
    another block and is added by the boundary kernel, so it is subtracted here.
    """
    chunks = int(carry_u.shape[2])
    want = dU.clone()
    if chunks > 1:
        rows = torch.arange(1, chunks, device=want.device) * chunk - 1
        want[:, :, rows, :] -= carry_u[:, :, 1:, :]
    return want


def _expected_dchunk_scale(
    ref: ChunkedBackward, dinc: Tensor, zstart: Tensor
) -> Tensor:
    """``dchunk_scale`` off the two state buffers the kernel is handed.

    ``dchunk_scale`` contracts ``dinc`` and ``zstart`` and nothing else, so the
    rounding of the pair the kernel reads is the whole of its error unless the
    reference contracts that same pair. ``ref.dchunk_scale`` contracts the float64
    values instead, which charges the kernel for its own inputs. Reduction order
    follows the reference, which calls this per chunk.

    Args:
        ref: The float64 reference backward, for the transition's other two operands.
        dinc: ``(B,H,C,P,3N)``, as handed to the kernel.
        zstart: ``(B,H,C,P,3N)``, as handed to the kernel.

    Returns:
        ``(B,H,C)`` float64.
    """
    lanes = int(dinc.shape[-1]) // 3
    dinc_d = dinc.double().unflatten(-1, (lanes, 3))
    zstart_d = zstart.double().unflatten(-1, (lanes, 3))
    return torch.stack(
        [
            chunk_transition_cotangents(
                dinc_d[:, :, c],
                zstart_d[:, :, c],
                ref.chunk_rot[:, :, c],
                ref.chunk_scale[:, :, c],
            ).dchunk_scale
            for c in range(int(dinc_d.shape[2]))
        ],
        dim=2,
    )


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
    ref = chunked_backward_fused(
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
    dinc = ref.dinc.to(U.dtype).contiguous()
    zstart = fw.zstart.flatten(-2, -1).to(U.dtype).contiguous()
    return Oracle(
        dU=_expected_du(ref.grads.dU, ref.carry_u, chunk),
        carry_u=ref.carry_u,
        dlogp=ref.dlogp_scan,
        dchunk_rot=ref.dchunk_rot,
        dchunk_scale=_expected_dchunk_scale(ref, dinc, zstart),
        dinc=dinc,
        zstart=zstart,
    )


def _run(
    inp: ScanInputs,
    dy: Tensor,
    want: Oracle,
    chunk: int,
    du_init: Tensor | None = None,
    threads: int | None = None,
) -> ChunkInputBwd:
    """One launch, against the state buffers the oracle carries.

    ``threads`` defaults to the dispatched width, so every other test in this file
    runs whichever form the shape gets in production.
    """
    got = chunk_input_backward(
        dy,
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        inp.C,
        want.dinc,
        want.zstart,
        chunk,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
        du_init=du_init,
        threads=threads,
    )
    torch.cuda.synchronize()
    return got


# Each output is a different depth of low-precision arithmetic, so each carries its
# own bound rather than all five sharing a figure loose enough for the deepest.
#
# ``dU`` and ``carry_u`` are one low-precision GEMM against a masked score that is
# itself a low-precision GEMM, so the error is that dtype's epsilon carried through
# two contractions, of length ``L`` and ``3N``. ``dlogp`` and ``dchunk_rot`` are
# reductions of those same fragments, so they inherit the error rather than
# compounding it. ``dchunk_scale`` contracts only the two state buffers.
#
#              bfloat16   float16
# dU           4.136e-3   4.335e-4
# carry_u      4.220e-3   1.860e-4
# dlogp        2.784e-3   2.774e-4
# dchunk_rot   7.508e-3   2.772e-4
# dchunk_scale 5.079e-6   2.769e-7
#
# Worst measured over every shape and every test in this file, read off a
# ``--tolerance-report`` run. Every bound below is 1.2 to 2.2 times its figure.
#
# The one-tap column moved two of the ten. ``dchunk_rot`` in bfloat16 rose 1.62x, to
# 7.508e-3 at ``1x1x256/L128/P64/N16``: the fused column is ``Ap + e An``, so the two
# transitions are summed before the contraction that the two-tap form ran against each
# of them separately. ``dchunk_scale`` in float16 rose 1.45x, to 2.769e-7. The other
# eight fell or held.
# ``dchunk_scale`` is three orders tighter than the rest because it contracts two
# operands and nothing else and its oracle contracts the same pair, so what is left
# is the accumulation; its worst case is ``MAX_CHUNK``, the longest one.
# ``dchunk_rot`` is the closest to its bound of the five: its transition term
# contracts that same pair, but its increment term contracts ``inc_local`` too, so
# its oracle stays the reference's own value and the pair's rounding is inside the
# residual. That is where narrowing the pair to the operand dtype is paid for: 1.3x
# in bfloat16 and 1.9x in float16 on ``dchunk_rot``, under 2x on ``dchunk_scale``,
# and nothing measurable on the other three. Against the nearest comparable bfloat16
# shape the float16 column runs 3 to 16 times tighter, which brackets the factor of
# eight between the two significands, so the two dtypes do not share a bound.
BOUNDS: dict[torch.dtype, dict[str, float]] = {
    torch.bfloat16: {
        "dU": 8e-3,
        "carry_u": 8e-3,
        "dlogp": 5e-3,
        "dchunk_rot": 9.5e-3,
        "dchunk_scale": 1e-5,
    },
    torch.float16: {
        "dU": 8e-4,
        "carry_u": 4e-4,
        "dlogp": 5e-4,
        "dchunk_rot": 4e-4,
        "dchunk_scale": 4e-7,
    },
}
"""No bound reaches ``1e-2``, so none needs a justification beyond the measured
figure above it."""


def _compare(got: ChunkInputBwd, want: Oracle, dtype: torch.dtype, tag: str) -> None:
    """Every output against the reference, each under its own bound."""
    bounds = BOUNDS[dtype]
    for name, bound in bounds.items():
        mine: Tensor = getattr(got, name)
        theirs: Tensor = getattr(want, name)
        assert mine.shape == theirs.shape, name
        assert_max_rel(mine, theirs, bound, f"{tag}.{name}")
        # A comparison against zeros passes whatever the GEMMs did.
        assert torch.count_nonzero(mine) > 0, name


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "dtype"), SHAPES
)
def test_chunk_input_matches_reference(
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
        f"cute-chunk-input[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/{str(dtype).removeprefix('torch.')}]"
    )
    _compare(got, want, dtype, tag)


@pytest.mark.parametrize("groups", [1, 4], ids=["one-group", "group-per-head"])
def test_grouped_forcing_reads_its_own_group(groups: int) -> None:
    """Head ``h`` contracts against group ``h // (H // G)``.

    ``G == 1`` is the only case where every head reads one ``B``/``C`` pair, and it
    is the case a missing divide silently passes at ``G == H``. An intermediate
    ``G`` takes the same divide as ``G == 1``, so it is not swept. The streaming
    ``b_prev`` is grouped with ``B`` and is read on the same index.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=groups)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, f"cute-chunk-input[G{groups}]")


def test_without_the_streaming_carry_in() -> None:
    """The absent carry-in is a zero row, not an out-of-range read.

    ``has_prev`` is compile-time, so the two cases are two kernels. Chunk 0's first
    token has no predecessor either way: with the pair absent its shifted tap sees
    zeros, and ``carry_u`` at chunk 0 is then a value nobody reads rather than the
    feedback gradient.
    """
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16, streaming=False)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, "cute-chunk-input[no-stream]")


def test_increment_terms_alone_when_dy_is_zero() -> None:
    """Differentiating the final state alone leaves the increment terms.

    With ``dy`` zero every score term drops out and what is left is ``duw``,
    ``dupw``, ``dexpw`` and the chunk transition. Those four are a small part of
    the mixed result's magnitude, so a sign or a weight error in them sits under
    the tolerance of :func:`test_chunk_input_matches_reference` and only shows up
    once the diagonal term is gone.
    """
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16)
    dy = torch.zeros_like(inp.U)
    want = _oracle(inp, None, 64, dstate=_dstate(inp))
    got = _run(inp, dy, want, 64)
    _compare(got, want, torch.bfloat16, "cute-chunk-input[no-dy]")


def test_matches_autograd_through_the_forward() -> None:
    """``dU`` and ``carry_u`` are the cotangents ``autograd`` sends into the forward.

    The reference backward is a separate derivation from the forward it
    differentiates, so a shared error in both passes
    :func:`test_chunk_input_matches_reference`. Differentiating the real forward
    shares nothing with either. ``u`` and ``ushift`` are two chunked views of the
    same input, built independently from it, so their cotangents come back
    separately and the within-chunk shift this kernel owns can be assembled from
    them without any of the reference's algebra. The ragged shape is used because
    it is where the last valid token takes ``du`` alone.

    Both differentiated outputs are pushed through together. ``y`` reaches the
    forcing input by the diagonal term and the final state reaches it by the
    increment term, and this kernel adds the two.

    The other three outputs are halves of quantities the forward does not name, so
    ``autograd`` cannot address them and they are left to the reference.
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
    du, dushift = torch.autograd.grad(
        (fw.y, fw.state),
        (fw.u, fw.ushift),
        (dy.double(), dstate.unflatten(-1, (-1, 3))),
    )
    shifted = torch.zeros_like(dushift)
    shifted[:, :, :, :-1, :] = dushift[:, :, :, 1:, :]
    dU = (du + shifted).flatten(2, 3)[:, :, :seqlen]

    got = _run(inp, dy, want, chunk)
    tag = "cute-chunk-input[autograd]"
    assert_max_rel(got.dU, dU, BOUNDS[torch.bfloat16]["dU"], f"{tag}.dU")
    assert_max_rel(
        got.carry_u,
        dushift[:, :, :, 0, :],
        BOUNDS[torch.bfloat16]["carry_u"],
        f"{tag}.carry_u",
    )


def test_the_forcing_seed_joins_the_sum_before_the_narrowing() -> None:
    """A supplied ``du_init`` reaches ``dU`` once, at its own token.

    The seed is what lets a fused caller hand this kernel the forcing gradient its
    other terms already produced: one read of ``(B,H,T,P)`` instead of the read, read
    and write a host-side add would cost. It joins the float32 sum ahead of the one
    narrowing, so a seeded ``dU`` carries no rounding a bare one does not.

    Checked against the reference sum rather than against an unseeded launch: a
    dropped seed, a seed added twice, a seed read at the wrong token and a seed added
    after the narrowing all fail. Scaled to ``dU`` because a seed far below it would
    vanish into the store's rounding and pass against a kernel that ignored it.
    """
    chunk = 64
    inp = _make(2, 2, 200, 48, 16, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, chunk, dstate=_dstate(inp))
    gen = torch.Generator(device="cuda").manual_seed(31)
    raw = torch.randn(
        inp.U.shape, generator=gen, dtype=torch.float32, device=inp.U.device
    )
    seed = (raw * float(want.dU.abs().max())).to(inp.U.dtype)
    got = _run(inp, dy, want, chunk, du_init=seed)
    assert_max_rel(
        got.dU,
        want.dU + seed.double(),
        BOUNDS[torch.bfloat16]["dU"],
        "cute-chunk-input[seed].dU",
    )


def _projection_band(vec: Tensor) -> Tensor:
    """``B`` or ``C`` as the mixer hands it over: one column band of a wider tensor.

    The fused projection is token-major, so the view the kernel receives strides by
    the projection width from one token to the next and by ``3N`` from one group to
    the next. The group axis therefore strides less than the axis before it, which a
    band cut out of a head-major buffer would not reproduce. Two bands sit ahead of
    it and one behind, so neither the offset nor the pitch is the one a dedicated
    buffer would have.
    """
    bsz, groups, seqlen, dim = vec.shape
    wide = torch.empty(bsz, seqlen, groups + 3, dim, dtype=vec.dtype, device=vec.device)
    band = wide[:, :, 2 : 2 + groups]
    band.copy_(vec.permute(0, 2, 1, 3))
    return band.permute(0, 2, 1, 3)


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
    pitched = inp._replace(B=_projection_band(inp.B), C=_projection_band(inp.C))
    got = _run(pitched, dy, want, 64)
    for name in BOUNDS[torch.bfloat16]:
        assert torch.equal(getattr(got, name), getattr(base, name)), name


def test_shared_memory_budget_fits_the_queried_capacity() -> None:
    """The budget is computed from the layouts, not from a guard constant.

    Two blocks per SM at the shape the DRAM-bound class is declared against is what
    keeps the pipe fed, and the resident set plus the phase arena is what that
    depends on. The widest legal ``(L, P, 3N)`` triple does not fit and is refused
    on the host; the widest that does is ``MAX_CHUNK`` at the standard row count.

    ``3N`` is not one of the axes that moves the total. The three lane-dependent
    tiles hold one lane block, so the footprint is flat in ``3N`` above the first
    block that has to be split, and the widest state the config layer accepts still
    runs two blocks deep.
    """
    assert 2 * input_smem_bytes(64, 48, 48) <= smem_capacity()
    assert input_smem_bytes(MAX_CHUNK, 48, 48) <= smem_capacity()
    assert input_smem_bytes(64, 128, 96) <= smem_capacity()
    assert input_smem_bytes(MAX_CHUNK, 128, 96) > smem_capacity()
    # L and P scale every tile they appear in, so neither is free: a budget that
    # ignored one would compare equal on one of these.
    assert input_smem_bytes(16, 48, 48) < input_smem_bytes(64, 48, 48)
    assert input_smem_bytes(64, 16, 48) < input_smem_bytes(64, 48, 48)
    lane = lblock(64, 64, 240)
    assert lane % LANE_MULTIPLE == 0 and 240 % lane == 0
    assert input_smem_bytes(64, 64, 240) == input_smem_bytes(64, 64, lane)
    assert 2 * input_smem_bytes(64, 64, 240) <= smem_capacity()


# (L, P, 3N) and the width the shape gets. The wide form is taken where one lane block
# holds the whole lane extent and refused where the extent is banked across blocks: 3N =
# 48 is one block at every row count here, 96 and 240 are two and five. The last row is
# refused for the other reason -- one lane block, but 103,264 B of a 101,376 B carveout,
# which the narrow form clears at 90,736 B.
#
# ``L=64 P=64 3N=96`` is two blocks and still takes the wide form, through the
# residency fallback rather than the lane count. The wide arena at a 48-block is
# 50,528 B and residency two admits 50,176 B, so no candidate is held two deep and
# :func:`lblock` returns the widest that fits, the whole 96 at 74,080 B, which is one
# lane block. The one-tap column moved that shape across the cliff: the two-tap form
# cost 50,016 B at the same block and cleared it by 160 B. 3N = 240 is unaffected --
# its whole extent needs 129,376 B, so the fallback there returns a 48-block and the
# width stays narrow.
WIDTHS: list[tuple[int, int, int, int]] = [
    (64, 48, 48, THREADS_WIDE),
    (MAX_CHUNK, 48, 48, THREADS_WIDE),
    (64, 64, 96, THREADS_WIDE),
    (64, 64, 240, THREADS),
    (MAX_CHUNK, 64, 48, THREADS),
]


@pytest.mark.parametrize(("chunk", "rows", "dim", "want"), WIDTHS)
def test_the_block_width_follows_the_lane_extent(
    chunk: int, rows: int, dim: int, want: int
) -> None:
    """The dispatch reads the lane count, and its choice fits the carveout.

    The width is not free to pick: the wide form's arena carries two more scratch rows
    and, at one lane block, a staged copy of the masked score. A predicate that widened
    a shape whose arena then did not fit would be refused on the host instead of run, so
    the budget is asserted at the width the dispatch actually returns.
    """
    got = input_threads(chunk, rows, dim)
    assert got == want, (chunk, rows, dim)
    assert input_smem_bytes(chunk, rows, dim, warps=got // 32) <= smem_capacity()


def test_the_lane_extent_reaches_the_residency_it_is_chosen_for() -> None:
    """A block taken for residency reaches it, or the widest that fits is taken instead.

    ``smem_capacity() // RESIDENT_MIN`` is the wrong budget and sits 512 B above the
    residency-2 cliff on sm_86: the capacity has one
    :data:`slinoss._cute.SMEM_RESERVED` subtracted already and two blocks pay two.
    Inside that band a narrower block was returned for a residency it does not reach,
    and the lane extent was banked across blocks for nothing. ``L=32 P=80 3N=96`` at
    four warps is such a shape: 48 costs 50,896 B and runs one block deep, so the
    contract asks for 96 at 70,864 B, which is one block deep and one lane block.
    """
    chunk, rows, dim, warps = 32, 80, 96, 4
    lblk = lblock(chunk, rows, dim, warps=warps)
    nbytes = input_smem_bytes(chunk, rows, dim, lblk=lblk, warps=warps)
    assert nbytes <= smem_capacity()
    if smem_residency(nbytes) < RESIDENT_MIN:
        assert lblk == dim, (lblk, nbytes)


# (L, P, threads) and the run :func:`lane_threads` cuts the block into. One row per
# distinct outcome rather than per shape: the narrowest run the mapping admits, an
# intermediate, and the fallback, which is what the hardcoded form used at every shape.
#
# ``P`` and the width separate them, not ``L``. A run of one needs ``threads`` rows to
# divide both extents and no row here reaches it; ``P = 64`` admits 64 rows, so the narrow
# width maps a run of two and the wide one a run of four, and ``P = 48`` admits 16 and no
# more at either width.
RUNS: list[tuple[int, int, int, int]] = [
    (64, 64, THREADS, 2),
    (MAX_CHUNK, 64, THREADS, 2),
    (64, 64, THREADS_WIDE, 4),
    (64, 48, THREADS_WIDE, LANE_THREADS),
    (MAX_CHUNK, 48, THREADS_WIDE, LANE_THREADS),
]


@pytest.mark.parametrize(("chunk", "rows", "threads", "want"), RUNS)
def test_the_lane_run_is_the_widest_the_row_mapping_admits(
    chunk: int, rows: int, threads: int, want: int
) -> None:
    """The run maps rows exactly, lies in one warp, and no narrower run maps.

    Three failures, and only the first corrupts a result. A run whose ``threads // run``
    rows do not divide ``L`` or ``P`` walks a row step off the tile. A run that is not a
    power of two, or one over 32, breaks the butterfly, which needs an aligned run inside
    one warp. A run wider than the narrowest legal one is silent: total rounds go as
    ``run log2(run)``, so a run twice as wide costs more than twice the butterflies and
    changes nothing else, which is why minimality is asserted and not only legality.
    """
    got = lane_threads(chunk, rows, threads)
    assert got == want, (chunk, rows, threads)
    assert got & (got - 1) == 0, got
    assert got <= LANE_THREADS <= 32
    held = threads // got
    assert chunk % held == 0 and rows % held == 0, (held, chunk, rows)
    if got > 1:
        loose = threads // (got // 2)
        assert chunk % loose or rows % loose, (got // 2, loose)


def test_refuses_a_shape_no_lane_block_fits() -> None:
    """The narrowest block overflowing the carveout is an error, not a return value.

    :func:`lblock` promises a block that fits. ``L=16 P=224 3N=48`` has one candidate,
    48, and it needs 102,576 B of a 101,376 B carveout, so there is nothing to return
    and the caller cannot tell a fitting block from an overflowing one by its width.
    """
    with pytest.raises(ValueError, match="no lane block"):
        lblock(16, 224, 48, warps=4)


def test_the_two_block_widths_differ_only_in_warp_reduction_order() -> None:
    """Both forms compute the same function, and the wide one reduces over more warps.

    The wide form routes the masked score through shared memory rather than rereading
    the fragment, which puts two barriers and a staged tile on a path that had neither,
    so a width the dispatch selects has to be held to the width it replaces rather than
    to the reference alone: a race there lands well inside the reference bounds on most
    elements and nowhere near bitwise on the rest.

    ``dU`` and ``carry_u`` are exact. Every element of both is one thread's own float32
    accumulation over an unchanged K order, and widening the tiling splits N, so no
    element changes hands and no sum changes order. The three reductions are not exact:
    they cross the warps through the epilogue's scratch rows, and eight partial sums do
    not add in the order four do. 1e-6 is three orders inside their reference bounds and
    two above the measured 2.3e-7, and it is not a tolerance any other test shares.
    """
    inp = _make(2, 4, 128, 16, 16, torch.bfloat16, groups=2)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, 64, dstate=_dstate(inp))
    assert input_threads(64, 16, 48) == THREADS_WIDE
    wide = _run(inp, dy, want, 64, threads=THREADS_WIDE)
    narrow = _run(inp, dy, want, 64, threads=THREADS)
    for name in ("dU", "carry_u"):
        assert torch.equal(getattr(wide, name), getattr(narrow, name)), name
    for name in ("dlogp", "dchunk_rot", "dchunk_scale"):
        assert_max_rel(getattr(wide, name), getattr(narrow, name), 1e-6, f"wide.{name}")


def test_rejects_a_shape_the_carveout_cannot_hold() -> None:
    """An oversized triple is refused on the host, not silently clipped.

    ``L``, ``P`` and ``3N`` are each legal at the values below and only their
    product overflows the carveout, so nothing but the budget checks this.
    """
    inp = _make(1, 1, MAX_CHUNK, 128, 32, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    want = _oracle(inp, dy, MAX_CHUNK, dstate=_dstate(inp))
    with pytest.raises(ValueError, match="chunk_input_bwd"):
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
        "u_prev": inp.u_prev,
        "b_prev": inp.b_prev,
        "du_init": None,
    }


def _call(args: Operands, chunk: int = 64) -> ChunkInputBwd:
    """Apply the public path to a perturbed operand set."""
    return chunk_input_backward(
        _t(args["dy"]),
        _t(args["U"]),
        _t(args["trans"]),
        _t(args["K"]),
        _t(args["B"]),
        _t(args["C"]),
        _t(args["dinc"]),
        _t(args["zstart"]),
        chunk,
        u_prev=args["u_prev"],
        b_prev=args["b_prev"],
        du_init=args["du_init"],
    )


REJECTIONS: list[tuple[Callable[[Operands], None], type[Exception], str]] = [
    # The rows this kernel owns are the ones no shared rule covers: the cotangent
    # checked against ``U`` rather than against itself, the two state buffers whose
    # chunk count is derived and not passed, and the streaming pair.
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
    (lambda a: a.update(b_prev=None), ValueError, "supplied together"),
    (lambda a: a.update(u_prev=None), ValueError, "supplied together"),
    # The seed is held to ``U``, the operand whose gradient it is added to. One row
    # is enough: the band check itself is covered where it is defined, and what is
    # unproven here is that this host path reaches it at all.
    (
        lambda a: a.update(du_init=_t(a["U"])[:, :, :-1].contiguous()),
        ValueError,
        "du_init must have shape",
    ),
    # One row per shared rule this kernel is the first to reach on an operand of
    # its own: ``U`` shares a dtype group with ``dy``, ``B`` and ``C``, and the two
    # state buffers follow the activation dtype and are read as dense tiles. The
    # float32 case is the one that was legal before the buffers were narrowed.
    (lambda a: a.update(U=_t(a["U"]).half()), TypeError, "one dtype per call"),
    (lambda a: a.update(dinc=_t(a["dinc"]).float()), ValueError, "dinc must be"),
    (
        lambda a: a.update(zstart=_t(a["zstart"]).transpose(-1, -2)),
        ValueError,
        "zstart must be contiguous",
    ),
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
    ("chunk", "rows", "lanes", "match"),
    [
        # 48 is a multiple of 16 and so clears the atom's K extent; what it fails is
        # the target-token slice, which is this kernel's own tiling and not the
        # atom's. The public config admits only powers of two, so no reachable
        # configuration hits this; the check is what keeps that true.
        (48, 16, 16, "K slice"),
        # ``P`` is the N mode of the diagonal and increment GEMMs here, unlike the
        # kernels where it is only a row count, so it carries the atom's constraint.
        (64, 24, 16, "P must be"),
        (64, 16, 8, "3N must be"),
    ],
)
def test_rejects_an_extent_the_atom_cannot_cover(
    chunk: int, rows: int, lanes: int, match: str
) -> None:
    """The fix for an illegal extent is the shape, never a padding path.

    The two state buffers are zeros here rather than reference output. The
    reference refuses ``P = 24`` and ``3N = 24`` itself, so no pipeline can produce
    a matching pair, and the extent check runs before any element is read.
    """
    seqlen = 128
    inp = _make(2, 2, seqlen, rows, lanes, torch.bfloat16)
    dy = _cotangent(inp, torch.bfloat16)
    state = torch.zeros(
        2,
        2,
        -(-seqlen // chunk),
        rows,
        3 * lanes,
        dtype=torch.bfloat16,
        device="cuda",
    )
    with pytest.raises(ValueError, match=match):
        chunk_input_backward(
            dy,
            inp.U,
            inp.trans,
            inp.K,
            inp.B,
            inp.C,
            state,
            state,
            chunk,
            u_prev=inp.u_prev,
            b_prev=inp.b_prev,
        )
