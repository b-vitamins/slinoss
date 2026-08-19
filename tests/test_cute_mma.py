"""The four scan contractions against a float32 oracle, at every legal extent.

One probe kernel stages two low-precision tiles into shared memory at the pitch
:func:`slinoss.ops.so3ssd.cute.mma.smem_pitch` prescribes, builds the operand view
each form needs, and runs the shipped helpers. The transposed forms are stride
swaps on the staged tile, so the probe also covers the claim that no form needs a
repack.

Operands are built in float32 and rounded once to the operand dtype, so the oracle
and the kernel see the same bits and every difference is float32 accumulation
order.

The fifth form is two contractions summed into one accumulator. That is how the
two forcing taps combine, and it is the only way to find out whether the helper
can be called twice on the same fragment.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from slinoss._cute import cute_dtype
from slinoss.config import HEAD_MULTIPLE, LANE_MULTIPLE, MIN_CHUNK
from slinoss.ops.so3ssd.cute.common import THREADS
from slinoss.ops.so3ssd.cute.mma import (
    MMA_TILE_K,
    MMA_TILE_M,
    MMA_TILE_N,
    SMEM_SEGMENT,
    make_mma,
    mma_acc,
    mma_gemm,
    mma_rows,
    mma_store,
    smem_pitch,
)
from tests.conftest import assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

INCREMENT = 0
SCORE = 1
DIAGONAL = 2
OFFSET = 3
TWO_TAP = 4

# (L, P, 3N). Every extent the operator can present: the default head width, a
# head width that does not divide the M tile, a doubled state, a chunk longer than
# the M tile, a chunk shorter than it, and both shape minima at once.
EXTENTS = [
    pytest.param(64, 64, 48, id="default"),
    pytest.param(64, 48, 48, id="P-under-tile"),
    pytest.param(64, 64, 96, id="wide-state"),
    pytest.param(128, 48, 48, id="L-over-tile"),
    pytest.param(32, 32, 48, id="L-under-tile"),
    pytest.param(MIN_CHUNK, HEAD_MULTIPLE, 3 * LANE_MULTIPLE, id="minima"),
]

# float32 accumulation over K in the MMA's order against cuBLAS's. The bound is
# 4e-6 rather than an epsilon multiple because K reaches 128 and the two orders
# differ; the recorded headroom shows it is not slack.
TOL = 4e-6


@cute.jit
def _stage(
    src: cute.Tensor,
    dst: cute.Tensor,
    tid: cutlass.Int32,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    ld: cutlass.Constexpr,
    pad_rows: cutlass.Constexpr,
) -> None:
    """Copy ``src(rows,cols)`` into ``dst(pad_rows,ld)``, zeroing the remainder.

    The padding participates in the MMA whenever it falls inside an operand view,
    so leaving it uninitialized admits whatever the allocator last held.
    """
    for i in cutlass.range(tid, pad_rows * ld, THREADS):
        r = i // ld
        c = i - r * ld
        # `&`, not `and`: the operands are device values, and `and` would force a
        # trace-time truth test on them.
        if (r < rows) & (c < cols):
            dst[r, c] = src[r, c]
        else:
            dst[r, c] = dst.element_type(0.0)


@cute.kernel
def _probe_kernel(
    ga0: cute.Tensor,
    gb0: cute.Tensor,
    ga1: cute.Tensor,
    gb1: cute.Tensor,
    gd: cute.Tensor,
    tiled_mma: cute.TiledMma,
    form: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
) -> None:
    """Run one contraction form and write the float32 accumulator out.

    Args:
        ga0: First left operand, low precision.
        gb0: First right operand, low precision.
        ga1: Second left operand. Read only by :data:`TWO_TAP`; the other forms
            are handed ``ga0`` so the signature has one form.
        gb1: Second right operand. Read only by :data:`TWO_TAP`.
        gd: Output, float32.
        tiled_mma: From :func:`make_mma`.
        form: Which contraction. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
    """
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    elem = ga0.element_type

    if cutlass.const_expr(form in (INCREMENT, TWO_TAP)):
        # M is P and is the stride-1 mode, so the pitch carries the rounding.
        mpad = mma_rows(rows)
        lda = smem_pitch(mpad)
        ldb = smem_pitch(dim)
        sa = smem.allocate_tensor(
            elem, cute.make_layout((chunk, lda), stride=(lda, 1)), SMEM_SEGMENT
        )
        sb = smem.allocate_tensor(
            elem, cute.make_layout((chunk, ldb), stride=(ldb, 1)), SMEM_SEGMENT
        )
        va = cute.make_tensor(
            sa.iterator, cute.make_layout((mpad, chunk), stride=(1, lda))
        )
        vb = cute.make_tensor(
            sb.iterator, cute.make_layout((dim, chunk), stride=(1, ldb))
        )
        acc = mma_acc(tiled_mma, tid, (mpad, dim))
        _stage(ga0, sa, tid, chunk, rows, lda, chunk)
        _stage(gb0, sb, tid, chunk, dim, ldb, chunk)
        cute.arch.sync_threads()
        mma_gemm(tiled_mma, tid, acc, va, vb, False, False)
        if cutlass.const_expr(form == TWO_TAP):
            # The second tap reuses the same two tiles. That is the shipped
            # pattern -- the increment kernel stages one tap at a time -- and the
            # sync ahead of the restage is what makes it legal.
            cute.arch.sync_threads()
            _stage(ga1, sa, tid, chunk, rows, lda, chunk)
            _stage(gb1, sb, tid, chunk, dim, ldb, chunk)
            cute.arch.sync_threads()
            mma_gemm(tiled_mma, tid, acc, va, vb, False, False)
        mma_store(tiled_mma, tid, acc, gd, (mpad, dim), rows)
        return

    # M is L and is the strided mode, so the tile carries the rounding in rows.
    mpad = mma_rows(chunk)
    if cutlass.const_expr(form == SCORE):
        ld = smem_pitch(dim)
        sa = smem.allocate_tensor(
            elem, cute.make_layout((mpad, ld), stride=(ld, 1)), SMEM_SEGMENT
        )
        sb = smem.allocate_tensor(
            elem, cute.make_layout((chunk, ld), stride=(ld, 1)), SMEM_SEGMENT
        )
        _stage(ga0, sa, tid, chunk, dim, ld, mpad)
        _stage(gb0, sb, tid, chunk, dim, ld, chunk)
        cute.arch.sync_threads()
        va = cute.make_tensor(
            sa.iterator, cute.make_layout((mpad, dim), stride=(ld, 1))
        )
        vb = cute.make_tensor(
            sb.iterator, cute.make_layout((chunk, dim), stride=(ld, 1))
        )
        acc = mma_acc(tiled_mma, tid, (mpad, chunk))
        mma_gemm(tiled_mma, tid, acc, va, vb, True, True)
        mma_store(tiled_mma, tid, acc, gd, (mpad, chunk), chunk)
    elif cutlass.const_expr(form == DIAGONAL):
        lda = smem_pitch(chunk)
        ldb = smem_pitch(rows)
        sa = smem.allocate_tensor(
            elem, cute.make_layout((mpad, lda), stride=(lda, 1)), SMEM_SEGMENT
        )
        sb = smem.allocate_tensor(
            elem, cute.make_layout((chunk, ldb), stride=(ldb, 1)), SMEM_SEGMENT
        )
        _stage(ga0, sa, tid, chunk, chunk, lda, mpad)
        _stage(gb0, sb, tid, chunk, rows, ldb, chunk)
        cute.arch.sync_threads()
        va = cute.make_tensor(
            sa.iterator, cute.make_layout((mpad, chunk), stride=(lda, 1))
        )
        vb = cute.make_tensor(
            sb.iterator, cute.make_layout((rows, chunk), stride=(1, ldb))
        )
        acc = mma_acc(tiled_mma, tid, (mpad, rows))
        mma_gemm(tiled_mma, tid, acc, va, vb, True, False)
        mma_store(tiled_mma, tid, acc, gd, (mpad, rows), chunk)
    else:
        ld = smem_pitch(dim)
        sa = smem.allocate_tensor(
            elem, cute.make_layout((mpad, ld), stride=(ld, 1)), SMEM_SEGMENT
        )
        sb = smem.allocate_tensor(
            elem, cute.make_layout((rows, ld), stride=(ld, 1)), SMEM_SEGMENT
        )
        _stage(ga0, sa, tid, chunk, dim, ld, mpad)
        _stage(gb0, sb, tid, rows, dim, ld, rows)
        cute.arch.sync_threads()
        va = cute.make_tensor(
            sa.iterator, cute.make_layout((mpad, dim), stride=(ld, 1))
        )
        vb = cute.make_tensor(
            sb.iterator, cute.make_layout((rows, dim), stride=(ld, 1))
        )
        acc = mma_acc(tiled_mma, tid, (mpad, rows))
        mma_gemm(tiled_mma, tid, acc, va, vb, True, True)
        mma_store(tiled_mma, tid, acc, gd, (mpad, rows), chunk)


@cute.jit
def _probe(
    ga0: cute.Tensor,
    gb0: cute.Tensor,
    ga1: cute.Tensor,
    gb1: cute.Tensor,
    gd: cute.Tensor,
    dtype: cutlass.Constexpr,
    form: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
) -> None:
    """Launch one block of :func:`_probe_kernel`."""
    _probe_kernel(
        ga0, gb0, ga1, gb1, gd, make_mma(dtype), form, chunk, rows, dim
    ).launch(grid=(1, 1, 1), block=(THREADS, 1, 1))


def _shapes(
    form: int, chunk: int, rows: int, dim: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """``(A, B, D)`` shapes for one form."""
    if form in (INCREMENT, TWO_TAP):
        return (chunk, rows), (chunk, dim), (rows, dim)
    if form == SCORE:
        return (chunk, dim), (chunk, dim), (chunk, chunk)
    if form == DIAGONAL:
        return (chunk, chunk), (chunk, rows), (chunk, rows)
    return (chunk, dim), (rows, dim), (chunk, rows)


def _oracle(
    form: int, a0: torch.Tensor, b0: torch.Tensor, a1: torch.Tensor, b1: torch.Tensor
) -> torch.Tensor:
    """The contraction in float32, from the same rounded bits the kernel reads."""
    if form == INCREMENT:
        return a0.float().T @ b0.float()
    if form == TWO_TAP:
        return a0.float().T @ b0.float() + a1.float().T @ b1.float()
    if form == DIAGONAL:
        return a0.float() @ b0.float()
    # Score and offset are the same expression at different extents: score
    # contracts two (L,3N) operands, offset an (L,3N) against a (P,3N).
    return a0.float() @ b0.float().T


def _run(
    form: int, chunk: int, rows: int, dim: int, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one form on the device and return ``(got, want)``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape_a, shape_b, shape_d = _shapes(form, chunk, rows, dim)

    def rnd(shape: tuple[int, int]) -> torch.Tensor:
        wide = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
        return wide.to(dtype)

    a0, b0 = rnd(shape_a), rnd(shape_b)
    a1, b1 = rnd(shape_a), rnd(shape_b)
    want = _oracle(form, a0, b0, a1, b1)
    got = torch.full(shape_d, float("nan"), device="cuda", dtype=torch.float32)
    _probe(
        *(from_dlpack(t, assumed_align=16) for t in (a0, b0, a1, b1, got)),
        cute_dtype(dtype),
        form,
        chunk,
        rows,
        dim,
    )
    torch.cuda.synchronize()
    return got, want


@pytest.mark.parametrize(("chunk", "rows", "dim"), EXTENTS)
@pytest.mark.parametrize(
    "form",
    [
        pytest.param(INCREMENT, id="increment"),
        pytest.param(SCORE, id="score"),
        pytest.param(DIAGONAL, id="diagonal"),
        pytest.param(OFFSET, id="offset"),
        pytest.param(TWO_TAP, id="two-tap"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_form_matches_the_oracle(
    form: int, chunk: int, rows: int, dim: int, dtype: torch.dtype
) -> None:
    """Every contraction, every extent, both operand dtypes."""
    got, want = _run(form, chunk, rows, dim, dtype, seed=form * 31 + chunk)
    assert torch.isfinite(got).all(), "an element of the output was never written"
    tag = f"cute-mma[{form}/L{chunk}/P{rows}/3N{dim}/{dtype}]"
    assert_max_rel(got, want, TOL, tag)


def test_padded_rows_do_not_reach_the_output() -> None:
    """The rounded M rows are dropped, not written past the end of the output.

    ``P = 48`` rounds to 64, so sixteen accumulator rows have nowhere to go. The
    output is allocated exactly and pre-filled with NaN; a store that ignored the
    predicate would either corrupt a neighbour or leave a NaN behind.
    """
    chunk, rows, dim = 64, 48, 48
    assert mma_rows(rows) == MMA_TILE_M
    got, want = _run(INCREMENT, chunk, rows, dim, torch.bfloat16, seed=7)
    assert tuple(got.shape) == (rows, dim)
    assert torch.isfinite(got).all()
    assert_max_rel(got, want, TOL, "cute-mma[predicated-store]")


def test_two_taps_accumulate_rather_than_overwrite() -> None:
    """A second contraction into the same accumulator adds to it.

    Checked against each contribution alone: the sum must differ from both, which
    a second call that reset the accumulator would fail.
    """
    chunk, rows, dim = 64, 64, 48
    both, want = _run(TWO_TAP, chunk, rows, dim, torch.bfloat16, seed=11)
    first, _ = _run(INCREMENT, chunk, rows, dim, torch.bfloat16, seed=11)
    assert_max_rel(both, want, TOL, "cute-mma[two-tap-sum]")
    assert not torch.allclose(both, first, rtol=1e-3, atol=1e-3)


def test_pitch_is_an_odd_number_of_segments() -> None:
    """Bank-conflict freedom rests on this, so it is asserted, not assumed."""
    for width in (8, 16, 24, 32, 48, 64, 96, 128, 144):
        for itemsize in (2, 4):
            unit = SMEM_SEGMENT // itemsize
            pitch = smem_pitch(width, itemsize)
            assert pitch >= width
            assert pitch % unit == 0
            assert (pitch // unit) % 2 == 1
            assert pitch - width < 2 * unit


def test_pitch_rejects_an_element_wider_than_the_segment() -> None:
    """A 32-byte element has no segment arithmetic; the message says so."""
    with pytest.raises(ValueError, match="does not divide"):
        smem_pitch(48, itemsize=SMEM_SEGMENT + 1)


def test_extents_the_operator_can_present_are_all_legal() -> None:
    """N and K divisibility is a device constraint, so the shape rules must imply it.

    Every N extent in the four forms is ``3N``, ``L`` or ``P``, and every K extent
    is ``3N`` or ``L``. If any of the three could fail to divide, a legal config
    would produce a kernel that does not compile.
    """
    assert (3 * LANE_MULTIPLE) % MMA_TILE_N == 0
    assert (3 * LANE_MULTIPLE) % MMA_TILE_K == 0
    assert HEAD_MULTIPLE % MMA_TILE_N == 0
    assert MIN_CHUNK % MMA_TILE_N == 0
    assert MIN_CHUNK % MMA_TILE_K == 0
