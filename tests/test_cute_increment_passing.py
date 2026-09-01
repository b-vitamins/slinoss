"""Fused increment and inter-chunk recurrence against the float64 reference.

The two kernels this replaces are tested separately in
``test_cute_chunk_increment.py`` and ``test_cute_state_passing.py``, and both
comparisons are against an intermediate: the first stops at ``inc_local`` and the
second is handed one. Here the intermediate does not exist, so the only available
authority is the whole prologue at once, and the bound is the operand bound of the
GEMM carried through a recurrence whose every transition has norm at most one.

Operands are built in float32 and cast to the operand dtype, then the reference
runs on an exact float64 upcast of those same cast tensors, so the low-precision
rounding of the inputs is inside both paths.

What is new here and nowhere else is the band decomposition: the state width is cut
into blocks that each carry their own columns. A band that mislays its columns is
invisible at one band, so the sweep runs a width that gives several and a width that
gives one.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from slinoss._cute import smem_residency
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import chunked_forward
from slinoss.ops.so3ssd.cute.fwd import increment_passing
from slinoss.ops.so3ssd.cute.fwd.increment_passing import (
    RESIDENT_MAX,
    SPLIT,
    fused_kblock,
    fused_smem_bytes,
    increment_passing_forward,
)
from slinoss.ops.so3ssd.cute.mma import MMA_TILE_K
from tests.conftest import LS_BIAS, ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# LS_BIAS keeps the chunk decay above float32 epsilon. Unbiased, the recurrence under
# test is the identity on the increment.

# (bsz, heads, groups, seqlen, chunk, rows, lanes, span, kblk, streaming, state,
# dtype).
#
# One case per distinct path through the fused kernel:
#
# - several bands against one, which is the only new axis. At ``span`` equal to
#   ``3N`` there is one block per (batch, head) and the column arithmetic is an
#   identity; at :data:`SPLIT` it is not;
# - the K slice count, several against one, since the slice loop is what stages. A
#   ``kblk`` of None takes the shipped default, which is wider than one MMA step and
#   so tiles the GEMM's K mode; one row pins the atom's own extent to keep the
#   several-slice loop covered wherever the budget puts that default;
# - a ragged tail, where the pad tap must zero the operand;
# - the streaming split, the only way the previous tap reaches a token before the
#   sequence, and the initial state, which is a compile-time variant;
# - grouped heads, where the forcing band is shared and one head per group writes
#   the carry-out;
# - ``P`` padded up to the MMA tile against ``P`` on it, which selects between the
#   predicated store and the vectorized one, and with it the count of 3-vectors a
#   thread carries;
# - both operand dtypes, because each is a different MMA atom.
SHAPES = [
    pytest.param(
        2,
        2,
        2,
        200,
        64,
        16,
        16,
        SPLIT,
        None,
        True,
        True,
        torch.bfloat16,
        id="ragged-one-band",
    ),
    pytest.param(
        2,
        4,
        2,
        256,
        MAX_CHUNK,
        32,
        32,
        SPLIT,
        MMA_TILE_K,
        False,
        False,
        torch.float16,
        id="grouped-two-bands-zero-start",
    ),
    pytest.param(
        1,
        1,
        1,
        128,
        64,
        64,
        32,
        96,
        None,
        True,
        True,
        torch.bfloat16,
        id="whole-width-band",
    ),
]


def _make(
    bsz: int,
    heads: int,
    groups: int,
    seqlen: int,
    rows: int,
    lanes: int,
    streaming: bool,
    with_state: bool,
    dtype: torch.dtype,
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
        with_state=with_state,
        streaming=streaming,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


@pytest.mark.parametrize(
    (
        "bsz",
        "heads",
        "groups",
        "seqlen",
        "chunk",
        "rows",
        "lanes",
        "span",
        "kblk",
        "streaming",
        "with_state",
        "dtype",
    ),
    SHAPES,
)
def test_increment_passing_matches_reference(
    bsz: int,
    heads: int,
    groups: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    span: int,
    kblk: int | None,
    streaming: bool,
    with_state: bool,
    dtype: torch.dtype,
) -> None:
    """Every output of the prologue matches a float64 forward."""
    inp = _make(bsz, heads, groups, seqlen, rows, lanes, streaming, with_state, dtype)
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
    out = increment_passing_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        chunk,
        z0=inp.z0,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
        span=span,
        kblk=kblk,
    )
    torch.cuda.synchronize()

    tag = (
        f"cute-increment-passing[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}"
        f"/S{span}/K{kblk}/{str(dtype).removeprefix('torch.')}]"
    )
    # The rotated forcing is rounded to the operand dtype once on its way into
    # shared memory, which the reference does not do, so the bound is the increment
    # kernel's own operand bound. The recurrence adds nothing to it: every chunk
    # transition has norm at most one by I1, so the carried error neither amplifies
    # nor accumulates across chunks.
    bound = 6e-3 if dtype is torch.bfloat16 else 8e-4
    assert_max_rel(out.zstart, ref.zstart.flatten(-2, -1), bound, f"{tag}.zstart")
    assert_max_rel(out.state, ref.state.flatten(-2, -1), bound, f"{tag}.state")
    # A comparison against zeros passes whatever the GEMM and the recurrence do.
    assert torch.count_nonzero(out.zstart[:, :, -1]) > 0

    # Both carried quantities are float32 throughout, so they hold the prefix bound
    # rather than the operand bound.
    assert_max_rel(out.cquat, ref.qprefix[..., -1, :], 2e-6, f"{tag}.cquat")
    want_scale = torch.exp(2.0 * ref.lprefix[..., -1])
    assert bool((out.cscale > 0.0).all()) and bool((out.cscale <= 1.0).all()), "I1"
    # exp turns the float32 absolute error of the log prefix into a relative error
    # that grows with how far the prefix reaches.
    reach = float(ref.lprefix[..., -1].abs().max())
    eps = float(torch.finfo(torch.float32).eps)
    assert_max_rel(out.cscale, want_scale, 4e-6 + 4.0 * reach * eps, f"{tag}.cscale")

    # The carry-out is a copy, so it is exact whatever the band that wrote it.
    assert torch.equal(out.u_last, inp.U[:, :, seqlen - 1])
    assert torch.equal(out.b_last, inp.B[:, :, seqlen - 1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_start_state_carries_the_activation_dtype(dtype: torch.dtype) -> None:
    """``zstart`` is stored at the operand width; ``state`` stays float32.

    I4 pins float32 to the recurrence, which carries the state in registers and
    returns it, not to the per-chunk copy: that copy is read by one GEMM per chunk,
    which narrows it into shared memory whatever width it arrives at, so the store
    narrows instead. A copy at any other width reaches
    :func:`slinoss.ops.so3ssd.cute.guard.check_stored` at every consumer.

    Chunk 0's copy is the initial state before any increment enters it, so the store
    is checked bitwise against the narrowing of ``z0`` rather than under a tolerance.
    """
    inp = _make(2, 2, 2, 128, 16, 16, True, True, dtype)
    assert inp.z0 is not None
    out = increment_passing_forward(
        inp.U,
        inp.trans,
        inp.K,
        inp.B,
        64,
        z0=inp.z0,
        u_prev=inp.u_prev,
        b_prev=inp.b_prev,
    )
    torch.cuda.synchronize()
    assert out.zstart.dtype is dtype
    assert out.state.dtype is torch.float32
    assert torch.equal(out.zstart[:, :, 0], inp.z0.to(dtype))


# (span, warps, message). Everything else the host entry rejects is a guard shared
# with the unfused increment and is tested there; what is only here is the band
# width and the block width it has to partition.
TILINGS = [
    pytest.param(SPLIT + 3, 8, "span must divide", id="span-off-the-atom"),
    pytest.param(2 * SPLIT, 8, "span must divide", id="span-wider-than-3N"),
    pytest.param(SPLIT, 16, "warps", id="too-many-warps"),
]


@pytest.mark.parametrize(("span", "warps", "message"), TILINGS)
def test_rejects_an_uncoverable_tiling(span: int, warps: int, message: str) -> None:
    """A band the launch cannot cover exactly is refused, not padded.

    The grid is ``3N / span`` blocks and each owns ``P * span / 3`` 3-vectors split
    evenly over its threads. A width violating either is a caller bug: no predicate
    would make the recurrence's columns line up with ``zstart``.
    """
    inp = _make(2, 2, 2, 128, 16, 16, False, False, torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        increment_passing_forward(
            inp.U, inp.trans, inp.K, inp.B, 64, span=span, warps=warps
        )


# Every chunk `check_extents` admits, not only the powers of two `SLinOSSConfig`
# happens to allow. The kernel's own guard is the wider surface: a driver, a
# sweep, or a caller reaching past the config sees these lengths.
GUARD_CHUNKS = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 240, 256]


@pytest.mark.parametrize("chunk", GUARD_CHUNKS)
def test_slice_is_always_a_whole_number_of_mma_steps(chunk: int) -> None:
    """The chosen slice is a multiple of the atom's K extent, at every legal chunk.

    A slice that divides the chunk but not ``MMA_TILE_K`` splits the K loop across
    an MMA step. The atom reads its fragment whole, so the tail step reads operand
    columns the slice never staged, and the kernel returns NaN with no error
    raised: measured at ``L=144``, where the search returns 18, the launch produces
    55,349 NaN in ``zstart`` and 30,720 in ``state``, while slice 16 at the same
    shape is clean. Dividing the chunk is necessary and not sufficient.

    Held here rather than left to the config's power-of-two rule, which hides it by
    accident: every power of two at or above 16 halves down to 16, so the defect is
    unreachable through the public mixer and reachable through the guard.
    """
    kblk = fused_kblock(chunk, 64, SPLIT)
    assert kblk % MMA_TILE_K == 0, (chunk, kblk)
    assert chunk % kblk == 0, (chunk, kblk)
    assert kblk >= MMA_TILE_K, (chunk, kblk)


# (chunk, rows). One per residency the shipped budget reaches at ``SPLIT``: three
# blocks at ``L=64``, two at ``L=128``, one at ``L=256``.
RESIDENCIES = [(64, 48), (64, 64), (128, 48), (256, 48)]


@pytest.mark.parametrize(("chunk", "rows"), RESIDENCIES)
@pytest.mark.parametrize("asked", [None, RESIDENT_MAX + 2])
def test_the_launch_bound_never_asks_more_blocks_than_shared_memory_holds(
    monkeypatch: pytest.MonkeyPatch, chunk: int, rows: int, asked: int | None
) -> None:
    """The block count the bound asks for is clamped to the shared budget.

    ``min_blocks_per_mp`` caps a thread at ``65536 / (blocks * threads)`` registers
    whether or not the shared memory admits that many blocks, so a bound above the
    budget buys no residency and pays the register cap anyway. Measured on sm_86 at
    ``L=128 P=48 span=48``, where the budget holds two blocks: an unclamped bound of
    three compiles to 168 registers and spills 147,456 local load and 147,456 local
    store sectors a launch, against 236 registers and zero at two, and a bound of
    four to 128 registers and 469,248 local load sectors.

    Both ways in are covered. :data:`RESIDENT_MAX` is a constant a later tiling sweep
    may raise, and ``resident`` is a caller's number; neither reaches the launch
    without the clamp.

    The launcher is replaced rather than run, because what is under test is the
    argument and not the kernel.
    """
    captured: list[int] = []
    monkeypatch.setattr(
        increment_passing,
        "jit_launch",
        lambda fn, dynamic, static: captured.append(static[-1]),
    )
    inp = _make(1, 1, 1, chunk, rows, 16, False, False, torch.bfloat16)
    increment_passing_forward(inp.U, inp.trans, inp.K, inp.B, chunk, resident=asked)
    kblk = fused_kblock(chunk, rows, SPLIT, inp.U.element_size())
    held = smem_residency(
        fused_smem_bytes(chunk, rows, SPLIT, inp.U.element_size(), kblk=kblk)
    )
    assert len(captured) == 1, captured
    # Equality and not an upper bound: a count below the residency caps registers for
    # a block the SM would have run, which is the same defect in the other direction.
    assert captured == [min(asked or RESIDENT_MAX, held)], (chunk, rows, held, kblk)
