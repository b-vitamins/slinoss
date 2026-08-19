"""Block norm and activation: the CuTe kernels against the reference.

The authority is :mod:`slinoss.ops.block.reference` in float64. Operands are built
in float32, rounded once to the operand dtype, and only then upcast for the
oracle, so the kernel and the oracle read identical values and every difference
is arithmetic width and reduction order.

Every output buffer is poisoned with NaN and freed immediately before the call,
so the caching allocator hands the same block back to the kernel. Without that an
element the kernel never writes reads as whatever the allocator last held, and a
finiteness check passes on a kernel that skipped it. The backward's per-block
partial buffer is poisoned too: it is summed on the host, so one element the
epilogue skipped turns the whole weight gradient into a NaN.

Axes that do not interact are not crossed. Operand dtype changes the load and the
store width and nothing else, so it is swept once per kernel at one shape while
the shape sweep runs at one dtype. The three compile-time presence flags of the
fused backward do interact -- they gate branches of one kernel -- so they are
crossed in full, at one shape.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss._cute import assert_smem_fits, smem_bytes, smem_capacity
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.block import (
    rmsnorm_bwd_ref,
    rmsnorm_ref,
    rmsnorm_residual_bwd_ref,
    rmsnorm_residual_ref,
    swiglu_bwd_ref,
    swiglu_ref,
)
from slinoss.ops.block.cute import (
    ACT_THREADS,
    BWD_SLOTS,
    FWD_SLOTS,
    NORM_THREADS,
    VECTOR_BYTES,
    WARPS,
    norm_smem_bytes,
    reduce_tile,
    rmsnorm_backward,
    rmsnorm_forward,
    rmsnorm_residual_backward,
    rmsnorm_residual_forward,
    row_blocks,
    sm_count,
    swiglu_backward,
    swiglu_forward,
    total_tile,
)
from tests.conftest import assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

EPS = 1e-5

DTYPES = [torch.bfloat16, torch.float32]

# (rows, D). D is a constexpr, so every entry is a separate compilation: the
# smallest legal d_model, a D under the block width, a D that is neither a
# multiple of the block width nor of a warp, and a D above 2048. rows = 1 appears
# twice, once at each end of the D range.
NORM_SHAPES = [
    pytest.param(1, 1, id="single-row-single-column"),
    pytest.param(3, 48, id="D-under-block"),
    pytest.param(80, 300, id="ragged-D"),
    pytest.param(1, 4096, id="single-row-wide-D"),
    pytest.param(5, 2048, id="many-rows"),
]

# (rows, D) for the backwards, whose grid is fixed and strided over rows rather
# than one block per row. The D entries are the forward's, minus the 5x2048 case
# that adds nothing on this axis, plus a row count above twice the SM count so the
# stride loop runs several times and its trip count differs between blocks.
NORM_BWD_SHAPES = [
    pytest.param(1, 1, id="single-row-single-column"),
    pytest.param(3, 48, id="D-under-block"),
    pytest.param(80, 300, id="ragged-D"),
    pytest.param(1, 4096, id="single-row-wide-D"),
    pytest.param(500, 300, id="grid-stride-rows"),
]

# (rows, D) for the activation, which sees only the element count. 1 is a
# tail-only launch, 2107 is a count whose remainder is nonzero at both vector
# widths, and 1048576 is more vectors than the grid holds threads, so the stride
# loop runs several times.
ACT_SHAPES = [
    pytest.param(1, 1, id="one-element"),
    pytest.param(7, 301, id="ragged-tail"),
    pytest.param(80, 300, id="exact-vectors"),
    pytest.param(256, 4096, id="grid-stride"),
]

# Both kernels compute in float32 and store at the operand width, so both bounds
# are the same two terms.
#
# float32 is the hardware approximations the kernels call -- `rsqrt.approx.f32`
# for the row scale and `ex2.approx.f32` for the logistic, each under 2^-22
# relative -- plus a mean-square reduction in an order the oracle does not use.
F32_TOL = 4e-7

# bfloat16 keeps 8 significant bits, so one round-to-nearest store is under half
# an ulp, 2^-8, of the largest element compared. The oracle is unrounded float64,
# so that store is the whole remaining error and the bound is the analytic sum of
# the two terms with nothing added: a kernel that introduces any error of its own
# fails.
BF16_TOL = 2.0**-8 + F32_TOL

TOL = {torch.bfloat16: BF16_TOL, torch.float32: F32_TOL}

# The wide residual is one float32 add of the widened operands, which is what
# torch computes, so the two agree bit for bit and the bound is exact equality.

# The weight gradient is float32 whatever the operand width, and its operands are
# the rounded ones the oracle also reads, so the bfloat16 store term does not enter
# and one float32 bound covers both widths. Reducing over 500 rows instead of 80
# moved the measured error from 1.21e-7 to 1.33e-7, so the row axis is not the
# leading term; the row scale is, which is why the bound is the float32 one.
DW_TOL = F32_TOL

# `dx` at D = 1 is the conditioning of the map, not an error the kernel adds. The
# pullback is `r * (c_i - r^2 <c,s> s_i / D)`, and at D = 1 the cotangent is
# parallel to the row by construction, so it collapses to `r * c * eps / (x^2 +
# eps)`: a difference of two float32 quantities that agree to five digits. The
# float32 term is amplified by `(x^2 + eps) / eps`, which is 5e4 on this operand at
# eps = 1e-5, and 2^-21 * 5e4 is 2e-2. Measured 2.28e-2 at D = 1 against 4.09e-8 at
# D = 48, where the coupling term is 1/48 of the row and nothing cancels. Every
# other quantity at D = 1 keeps the ordinary bound.
DX1_TOL = 5e-2


def _tag(name: str, rows: int, width: int, dtype: torch.dtype) -> str:
    return f"cute-block.{name}[{rows}x{width}/{str(dtype).removeprefix('torch.')}]"


def _rnd(shape: tuple[int, ...], dtype: torch.dtype, *, seed: int) -> Tensor:
    """One operand on CUDA. Built in float32, then rounded once."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    wide = torch.randn(shape, generator=gen, dtype=torch.float32, device="cuda")
    return wide.to(dtype)


def _poison(*specs: tuple[tuple[int, ...], torch.dtype]) -> None:
    """Fill and free one block per output the next call allocates."""
    junk = [
        torch.full(shape, float("nan"), dtype=dtype, device="cuda")
        for shape, dtype in specs
    ]
    assert all(bool(t.isnan().all()) for t in junk)
    del junk


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "width"), NORM_SHAPES)
def test_rmsnorm_matches_the_reference(
    rows: int, width: int, dtype: torch.dtype
) -> None:
    """A row reduced by the wrong thread set, or a scale that never reached every
    thread, changes the answer.

    The cross-warp step is the failure mode this covers: a shuffle direction whose
    total lands in the wrong lane, or a broadcast read that races the write, both
    show up as a per-row scale that disagrees with the oracle.
    """
    shape = (1, rows, width)
    x = _rnd(shape, dtype, seed=1)
    weight = _rnd((width,), torch.float32, seed=2)
    want = rmsnorm_ref(x.double(), weight.double(), eps=EPS)

    _poison((shape, dtype))
    got = rmsnorm_forward(x, weight, eps=EPS)
    torch.cuda.synchronize()

    assert tuple(got.shape) == shape
    assert got.dtype is dtype
    assert got.is_contiguous()
    assert bool(torch.isfinite(got).all())
    assert_max_rel(got, want, TOL[dtype], _tag("norm", rows, width, dtype))


def _check_residual(rows: int, width: int, dtype: torch.dtype, stream: str) -> None:
    """The fused add must feed the norm the same sum it hands back.

    A second summation in the rescale pass, or a wide output written after it is
    read, would leave the two outputs describing different sums. The residual is
    compared for exact equality, which is what one float32 add of the widened
    operands owes; a bound there would hide a narrowed intermediate.
    """
    shape = (1, rows, width)
    x = _rnd(shape, dtype, seed=3)
    weight = _rnd((width,), torch.float32, seed=4)
    residual = None
    if stream != "none":
        res_dtype = dtype if stream == "same" else torch.float32
        residual = _rnd(shape, res_dtype, seed=5)

    want = rmsnorm_residual_ref(
        x.double(),
        None if residual is None else residual.double(),
        weight.double(),
        eps=EPS,
    )
    want_sum = x.float() if residual is None else x.float() + residual.float()

    _poison((shape, dtype), (shape, torch.float32))
    got = rmsnorm_residual_forward(x, residual, weight, eps=EPS)
    torch.cuda.synchronize()

    assert got.normed.dtype is dtype
    assert got.residual.dtype is torch.float32
    assert tuple(got.residual.shape) == shape
    assert got.normed.is_contiguous()
    assert got.residual.is_contiguous()
    assert torch.equal(got.residual, want_sum)
    assert_max_rel(
        got.normed,
        want.normed,
        TOL[dtype],
        _tag(f"residual-{stream}", rows, width, dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "width"), NORM_SHAPES)
def test_rmsnorm_residual_matches_the_reference(
    rows: int, width: int, dtype: torch.dtype
) -> None:
    """The fused kernel is compiled per ``D``, so it carries its own shape sweep.

    It shares only the reduction helper with the plain norm, not the loop bounds
    or the stores, so a shape-dependent fault here would not show up there.
    """
    _check_residual(rows, width, dtype, "same")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("stream", ["none", "wide"])
def test_rmsnorm_residual_carries_either_residual_width(
    stream: str, dtype: torch.dtype
) -> None:
    """The residual argument is optional and independently typed.

    ``none`` is the first block of a stack, where the sum is ``x`` alone.
    ``wide`` is every later block: a float32 residual against a low-precision
    branch output. Neither changes the reduction, so one ragged shape covers both;
    the shape sweep above runs at the matched width.
    """
    _check_residual(80, 300, dtype, stream)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("rows", "width"), ACT_SHAPES)
def test_swiglu_matches_the_reference(
    rows: int, width: int, dtype: torch.dtype
) -> None:
    """An element count that is not a whole number of vectors loses its remainder.

    The vector loop and the tail predicate partition the flat index space. An
    off-by-one in either leaves an element unwritten, which the poison turns into
    a NaN, or written twice, which the parity check catches.
    """
    shape = (1, rows, width)
    gate = _rnd(shape, dtype, seed=6)
    up = _rnd(shape, dtype, seed=7)
    want = swiglu_ref(gate.double(), up.double())

    _poison((shape, dtype))
    got = swiglu_forward(gate, up)
    torch.cuda.synchronize()

    assert tuple(got.shape) == shape
    assert got.dtype is dtype
    assert got.is_contiguous()
    assert bool(torch.isfinite(got).all())
    assert_max_rel(got, want, TOL[dtype], _tag("swiglu", rows, width, dtype))


# ---------------------------------------------------------------------------
# Backward parity
# ---------------------------------------------------------------------------


def _blocks(rows: int) -> int:
    """Partial-buffer rows the launch will allocate, for the poison."""
    return row_blocks(rows, torch.cuda.current_device())


def _dx_tol(width: int, dtype: torch.dtype) -> float:
    """Bound on ``dx``, widened at ``D = 1`` for the reason at :data:`DX1_TOL`."""
    return DX1_TOL if width == 1 else TOL[dtype]


def _norm_bwd(rows: int, width: int, dtype: torch.dtype) -> None:
    """One plain-norm backward against the float64 oracle.

    Both reductions of the backward are checked here at once: an error in the sum
    of squares moves the row scale and shows up in both gradients, while an error
    in the cotangent dot product moves only ``dx``.
    """
    shape = (1, rows, width)
    x = _rnd(shape, dtype, seed=8)
    weight = _rnd((width,), torch.float32, seed=9)
    dout = _rnd(shape, dtype, seed=10)
    want = rmsnorm_bwd_ref(dout.double(), x.double(), weight.double(), eps=EPS)

    _poison((shape, dtype), ((_blocks(rows), width), torch.float32))
    got = rmsnorm_backward(dout, x, weight, eps=EPS)
    torch.cuda.synchronize()

    assert tuple(got.dx.shape) == shape
    assert tuple(got.dweight.shape) == (width,)
    assert got.dx.dtype is dtype
    assert got.dweight.dtype is torch.float32
    assert got.dx.is_contiguous()
    assert got.dweight.is_contiguous()
    assert bool(torch.isfinite(got.dx).all())
    assert bool(torch.isfinite(got.dweight).all())
    assert_max_rel(
        got.dx, want.dx, _dx_tol(width, dtype), _tag("dx", rows, width, dtype)
    )
    assert_max_rel(
        got.dweight, want.dweight, DW_TOL, _tag("dweight", rows, width, dtype)
    )


@pytest.mark.parametrize(("rows", "width"), NORM_BWD_SHAPES)
def test_rmsnorm_backward_matches_the_reference(rows: int, width: int) -> None:
    """The shape sweep. ``D`` drives the segment count, rows drive the grid.

    A backward's column loop is unrolled over a compile-time segment count, so the
    ragged last segment is a distinct code path at every ``D`` that is not a
    multiple of the block width. The row axis is the grid: above twice the SM count
    the stride loop runs more than once, and a weight-gradient accumulator that
    was reset per row rather than held across the loop only fails there.
    """
    _norm_bwd(rows, width, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_backward_carries_every_operand_width(dtype: torch.dtype) -> None:
    """Operand width changes the loads and the ``dx`` store, not the arithmetic.

    Both reductions are float32 whatever the operand width, so the dtype does not
    interact with the shape and one shape covers it. The ragged one is used so the
    narrow store is exercised on a masked column too.
    """
    _norm_bwd(80, 300, dtype)


def _residual_bwd(
    rows: int,
    width: int,
    dtype: torch.dtype,
    *,
    stream: str,
    normed: bool,
    dres: bool,
) -> None:
    """One fused backward against the float64 oracle.

    Args:
        rows: Rows on the flattened axis.
        width: ``D``.
        dtype: Branch-output dtype.
        stream: ``none`` for the first block of a stack, ``same`` for a stream at
            the branch width, ``wide`` for the float32 stream every later block
            sees.
        normed: Whether ``normed`` carries a cotangent.
        dres: Whether the wide residual carries a cotangent.
    """
    shape = (1, rows, width)
    x = _rnd(shape, dtype, seed=11)
    weight = _rnd((width,), torch.float32, seed=12)
    residual = None
    if stream != "none":
        residual = _rnd(shape, dtype if stream == "same" else torch.float32, seed=13)
    dnormed = _rnd(shape, dtype, seed=14) if normed else None
    # The forward returns the residual at float32, so its cotangent is float32.
    dresidual = _rnd(shape, torch.float32, seed=15) if dres else None

    want = rmsnorm_residual_bwd_ref(
        None if dnormed is None else dnormed.double(),
        None if dresidual is None else dresidual.double(),
        x.double(),
        None if residual is None else residual.double(),
        weight.double(),
        eps=EPS,
    )

    specs: list[tuple[tuple[int, ...], torch.dtype]] = [(shape, dtype)]
    if residual is not None:
        specs.append((shape, residual.dtype))
    if dnormed is not None:
        specs.append(((_blocks(rows), width), torch.float32))
    _poison(*specs)
    got = rmsnorm_residual_backward(dnormed, dresidual, x, residual, weight, eps=EPS)
    torch.cuda.synchronize()

    label = f"{stream}-{'n' if normed else ''}{'r' if dres else ''}"
    assert want.dx is not None and got.dx is not None
    assert tuple(got.dx.shape) == shape
    assert got.dx.dtype is dtype
    assert got.dx.is_contiguous()
    assert bool(torch.isfinite(got.dx).all())
    assert_max_rel(got.dx, want.dx, TOL[dtype], _tag(f"dx-{label}", rows, width, dtype))

    if residual is None:
        assert got.dresidual is None
    else:
        assert want.dresidual is not None and got.dresidual is not None
        assert got.dresidual.dtype is residual.dtype
        assert got.dresidual.is_contiguous()
        assert_max_rel(
            got.dresidual,
            want.dresidual,
            TOL[residual.dtype],
            _tag(f"dres-{label}", rows, width, dtype),
        )

    if dnormed is None:
        assert got.dweight is None
        assert want.dweight is None
    else:
        assert want.dweight is not None and got.dweight is not None
        assert tuple(got.dweight.shape) == (width,)
        assert got.dweight.dtype is torch.float32
        assert bool(torch.isfinite(got.dweight).all())
        assert_max_rel(
            got.dweight,
            want.dweight,
            DW_TOL,
            _tag(f"dweight-{label}", rows, width, dtype),
        )


@pytest.mark.parametrize(("rows", "width"), NORM_BWD_SHAPES)
def test_rmsnorm_residual_backward_matches_the_reference(rows: int, width: int) -> None:
    """The fused backward carries its own shape sweep.

    It shares the reduction helper with the plain backward and nothing else: the
    loop bounds, the stores, and the recomputed sum are its own, so a
    shape-dependent fault here would not show up there.
    """
    _residual_bwd(rows, width, torch.float32, stream="same", normed=True, dres=True)


@pytest.mark.parametrize("stream", ["none", "same", "wide"])
@pytest.mark.parametrize(
    ("normed", "dres"), [(True, True), (True, False), (False, True)]
)
def test_rmsnorm_residual_backward_covers_every_presence(
    stream: str, normed: bool, dres: bool
) -> None:
    """The full cross of the three compile-time flags, at one shape.

    These interact: each gates a branch of the same kernel, and the combination
    decides which gradients exist, whether the row reduction runs at all, and which
    tensor slots are placeholders. Both cotangents absent is not a row here because
    it launches nothing; it has its own test. The shape is the ragged one so every
    combination also sees a masked column.
    """
    _residual_bwd(80, 300, torch.float32, stream=stream, normed=normed, dres=dres)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_residual_backward_carries_every_operand_width(
    dtype: torch.dtype,
) -> None:
    """A float32 stream against a low-precision branch output.

    The stream is independently typed, so ``dx`` and ``dresidual`` are the same
    value at two widths. Neither width changes the reduction, so one shape covers
    the axis.
    """
    _residual_bwd(80, 300, dtype, stream="wide", normed=True, dres=True)


def test_rmsnorm_residual_backward_without_cotangents_returns_nothing() -> None:
    """Neither output differentiated is no launch and no gradient.

    A zero tensor here would allocate three full-size buffers and a launch to fill
    them, all to report that nothing was differentiated.
    """
    x = _rnd((1, 4, 8), torch.float32, seed=16)
    got = rmsnorm_residual_backward(
        None,
        None,
        x,
        _rnd((1, 4, 8), torch.float32, seed=17),
        _rnd((8,), torch.float32, seed=40),
        eps=EPS,
    )
    assert got.dx is None
    assert got.dresidual is None
    assert got.dweight is None


def _swiglu_bwd(rows: int, width: int, dtype: torch.dtype) -> None:
    """One activation backward against the float64 oracle."""
    shape = (1, rows, width)
    gate = _rnd(shape, dtype, seed=18)
    up = _rnd(shape, dtype, seed=19)
    dout = _rnd(shape, dtype, seed=20)
    want = swiglu_bwd_ref(dout.double(), gate.double(), up.double())

    _poison((shape, dtype), (shape, dtype))
    got = swiglu_backward(dout, gate, up)
    torch.cuda.synchronize()

    assert tuple(got.dgate.shape) == shape
    assert tuple(got.dup.shape) == shape
    assert got.dgate.dtype is dtype
    assert got.dup.dtype is dtype
    assert got.dgate.is_contiguous()
    assert got.dup.is_contiguous()
    assert bool(torch.isfinite(got.dgate).all())
    assert bool(torch.isfinite(got.dup).all())
    assert_max_rel(got.dgate, want.dgate, TOL[dtype], _tag("dgate", rows, width, dtype))
    assert_max_rel(got.dup, want.dup, TOL[dtype], _tag("dup", rows, width, dtype))


@pytest.mark.parametrize(("rows", "width"), ACT_SHAPES)
def test_swiglu_backward_matches_the_reference(rows: int, width: int) -> None:
    """The element-count sweep, which is the only shape the activation sees.

    Two gradients share one index space, so a tail predicate that covers one and
    not the other leaves half the trailing elements poisoned.
    """
    _swiglu_bwd(rows, width, torch.float32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_backward_carries_every_operand_width(dtype: torch.dtype) -> None:
    """Operand width sets the vector length, which the element count already sweeps.

    The two axes do not interact: the tail is ``numel % V``, and the sweep above
    holds a nonzero remainder at both vector widths.
    """
    _swiglu_bwd(7, 301, dtype)


def test_swiglu_backward_stays_finite_at_the_saturated_ends() -> None:
    """``silu'`` is ``sig * (1 + g * (1 - sig))``, and both factors saturate.

    At a strongly negative gate the logistic underflows to zero and the derivative
    is an exact zero however large the gate; at a strongly positive gate it is one
    and the ``1 - sig`` term is an exact zero, leaving one. A kernel that formed
    the derivative as a quotient, or that clamped, diverges at one end or the
    other. ``dup`` is ``silu`` itself, which the forward already covers.
    """
    up = torch.full((2, 64), 3.0, dtype=torch.float32, device="cuda")
    gate = torch.empty_like(up)
    gate[0] = -800.0
    gate[1] = 800.0
    dout = torch.full_like(up, 0.5)

    got = swiglu_backward(dout, gate, up)
    torch.cuda.synchronize()

    assert bool(torch.isfinite(got.dgate).all())
    assert bool(torch.isfinite(got.dup).all())
    assert torch.equal(got.dgate[0], torch.zeros_like(got.dgate[0]))
    assert torch.equal(got.dup[0], torch.zeros_like(got.dup[0]))
    want = swiglu_bwd_ref(dout.double(), gate.double(), up.double())
    assert_max_rel(
        got.dgate, want.dgate, TOL[torch.float32], "cute-block.dgate[saturated]"
    )
    assert_max_rel(got.dup, want.dup, TOL[torch.float32], "cute-block.dup[saturated]")


# ---------------------------------------------------------------------------
# Forward and backward end to end
# ---------------------------------------------------------------------------


def test_norm_forward_and_backward_agree_with_one_reference_graph() -> None:
    """The kernel forward, then the kernel backward, against one autograd graph.

    A backward tested against a surrogate forward hides any disagreement between
    the surrogate and the shipped kernel. Here the output comes from
    :func:`rmsnorm_forward` and the gradients from :func:`rmsnorm_backward`, and
    all three are compared against a single float64 graph through the reference.
    """
    shape = (1, 80, 300)
    x = _rnd(shape, torch.float32, seed=21)
    weight = _rnd((300,), torch.float32, seed=22)
    dout = _rnd(shape, torch.float32, seed=23)

    out = rmsnorm_forward(x, weight, eps=EPS)
    grads = rmsnorm_backward(dout, x, weight, eps=EPS)
    torch.cuda.synchronize()

    leaves = (x.double().requires_grad_(True), weight.double().requires_grad_(True))
    ref = rmsnorm_ref(leaves[0], leaves[1], eps=EPS)
    ref.backward(dout.double())

    assert leaves[0].grad is not None and leaves[1].grad is not None
    assert_max_rel(out, ref, TOL[torch.float32], "cute-block.e2e-norm[out]")
    assert_max_rel(
        grads.dx, leaves[0].grad, TOL[torch.float32], "cute-block.e2e-norm[dx]"
    )
    assert_max_rel(
        grads.dweight,
        leaves[1].grad,
        DW_TOL,
        "cute-block.e2e-norm[dweight]",
    )


def test_residual_forward_and_backward_agree_with_one_reference_graph() -> None:
    """Both fused outputs and all three gradients, against one autograd graph.

    The residual cotangent handed to the backward is the one the forward's own wide
    output would carry, so the two halves of the fusion are connected rather than
    tested apart.
    """
    shape = (1, 80, 300)
    x = _rnd(shape, torch.bfloat16, seed=24)
    residual = _rnd(shape, torch.float32, seed=25)
    weight = _rnd((300,), torch.float32, seed=26)
    dnormed = _rnd(shape, torch.bfloat16, seed=27)
    dresidual = _rnd(shape, torch.float32, seed=28)

    out = rmsnorm_residual_forward(x, residual, weight, eps=EPS)
    grads = rmsnorm_residual_backward(dnormed, dresidual, x, residual, weight, eps=EPS)
    torch.cuda.synchronize()

    leaves = (
        x.double().requires_grad_(True),
        residual.double().requires_grad_(True),
        weight.double().requires_grad_(True),
    )
    ref = rmsnorm_residual_ref(leaves[0], leaves[1], leaves[2], eps=EPS)
    torch.autograd.backward(
        [ref.normed, ref.residual], [dnormed.double(), dresidual.double()]
    )

    assert grads.dx is not None and grads.dresidual is not None
    assert grads.dweight is not None
    want_dx, want_dres, want_dweight = (leaf.grad for leaf in leaves)
    assert want_dx is not None and want_dres is not None and want_dweight is not None
    assert_max_rel(
        out.normed, ref.normed, TOL[torch.bfloat16], "cute-block.e2e-residual[normed]"
    )
    assert torch.equal(out.residual, (x.float() + residual))
    assert_max_rel(
        grads.dx, want_dx, TOL[torch.bfloat16], "cute-block.e2e-residual[dx]"
    )
    assert_max_rel(
        grads.dresidual,
        want_dres,
        TOL[torch.float32],
        "cute-block.e2e-residual[dresidual]",
    )
    assert_max_rel(
        grads.dweight,
        want_dweight,
        DW_TOL,
        "cute-block.e2e-residual[dweight]",
    )


def test_swiglu_forward_and_backward_agree_with_one_reference_graph() -> None:
    """The activation output and both gradients, against one autograd graph."""
    shape = (1, 7, 301)
    gate = _rnd(shape, torch.float32, seed=29)
    up = _rnd(shape, torch.float32, seed=30)
    dout = _rnd(shape, torch.float32, seed=31)

    out = swiglu_forward(gate, up)
    grads = swiglu_backward(dout, gate, up)
    torch.cuda.synchronize()

    leaves = (gate.double().requires_grad_(True), up.double().requires_grad_(True))
    ref = swiglu_ref(*leaves)
    ref.backward(dout.double())

    assert leaves[0].grad is not None and leaves[1].grad is not None
    assert_max_rel(out, ref, TOL[torch.float32], "cute-block.e2e-swiglu[out]")
    assert_max_rel(
        grads.dgate, leaves[0].grad, TOL[torch.float32], "cute-block.e2e-swiglu[dgate]"
    )
    assert_max_rel(
        grads.dup, leaves[1].grad, TOL[torch.float32], "cute-block.e2e-swiglu[dup]"
    )


# ---------------------------------------------------------------------------
# Domain ends
# ---------------------------------------------------------------------------


def test_zero_row_normalizes_to_zero() -> None:
    """``eps`` is the only thing between an all-zero row and ``rsqrt(0)``.

    A kernel that folded ``eps`` in after the ``rsqrt``, or dropped it, returns
    infinity here rather than the exact zero the reference gives. Mixed with a
    nonzero row so a per-row scale cannot be shared by accident.
    """
    width = 300
    x = torch.zeros(4, width, dtype=torch.float32, device="cuda")
    x[2] = 1.0
    weight = torch.ones(width, dtype=torch.float32, device="cuda")

    got = rmsnorm_forward(x, weight, eps=EPS)
    fused = rmsnorm_residual_forward(x, None, weight, eps=EPS)
    torch.cuda.synchronize()

    want = rmsnorm_ref(x.double(), weight.double(), eps=EPS)
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got[0], torch.zeros(width, dtype=torch.float32, device="cuda"))
    assert_max_rel(got, want, TOL[torch.float32], "cute-block.norm[zero-row]")
    assert torch.equal(fused.normed, got)


def test_saturated_gate_stays_finite() -> None:
    """``exp(-g)`` overflows float32 below ``g = -104``, and the quotient is zero.

    A kernel that formed the logistic as ``1 / (1 + exp(-g))`` and multiplied
    would compute ``1 / inf`` the same way, but one that clamped, or that formed
    ``exp(g) / (1 + exp(g))`` at the positive end, diverges here. Both ends are
    checked against the float64 reference, which underflows to the same limits.
    """
    up = torch.full((2, 64), 3.0, dtype=torch.float32, device="cuda")
    gate = torch.empty_like(up)
    gate[0] = -800.0
    gate[1] = 800.0

    got = swiglu_forward(gate, up)
    torch.cuda.synchronize()

    assert bool(torch.isfinite(got).all())
    assert torch.equal(got[0], torch.zeros_like(got[0]))
    assert_max_rel(
        got,
        swiglu_ref(gate.double(), up.double()),
        TOL[torch.float32],
        "cute-block.swiglu[saturated]",
    )


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

Call = Callable[[], object]

SHAPE = (1, 3, 8)


def _x(shape: tuple[int, ...] = SHAPE, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.randn(shape, dtype=dtype, device="cuda")


def _w(width: int = 8, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.randn(width, dtype=dtype, device="cuda")


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        pytest.param(
            lambda: rmsnorm_forward(_x(()), _w(1), eps=EPS),
            ValueError,
            r"at least one axis",
            id="norm-scalar",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x((1, 0, 8)), _w(), eps=EPS),
            ValueError,
            r"at least one row",
            id="norm-empty",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(), _w(7), eps=EPS),
            ValueError,
            r"weight must be \(8,\)",
            id="norm-weight-shape",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(), _w(dtype=torch.bfloat16), eps=EPS),
            ValueError,
            r"weight must be float32",
            id="norm-weight-narrow",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(), _w(), eps=0.0),
            ValueError,
            r"eps must be positive",
            id="norm-eps-zero",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x().cpu(), _w(), eps=EPS),
            ValueError,
            r"x must be on a CUDA device",
            id="norm-x-host",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x().transpose(1, 2), _w(3), eps=EPS),
            ValueError,
            r"x must be contiguous",
            id="norm-x-strided",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(dtype=torch.float64), _w(), eps=EPS),
            TypeError,
            r"kernel dtypes",
            id="norm-x-float64",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(), _w().cpu(), eps=EPS),
            ValueError,
            r"weight must be on a CUDA device",
            id="norm-weight-host",
        ),
        pytest.param(
            lambda: rmsnorm_forward(_x(), _w(16)[::2], eps=EPS),
            ValueError,
            r"weight must be contiguous",
            id="norm-weight-strided",
        ),
        pytest.param(
            lambda: rmsnorm_residual_forward(_x(), None, _w(7), eps=EPS),
            ValueError,
            r"weight must be \(8,\)",
            id="residual-shares-the-norm-checks",
        ),
        pytest.param(
            lambda: rmsnorm_residual_forward(_x(), _x((1, 4, 8)), _w(), eps=EPS),
            ValueError,
            r"residual must be \(1, 3, 8\)",
            id="residual-shape",
        ),
        pytest.param(
            lambda: rmsnorm_residual_forward(_x(), _x().cpu(), _w(), eps=EPS),
            ValueError,
            r"residual must be on a CUDA device",
            id="residual-host",
        ),
        pytest.param(
            lambda: rmsnorm_residual_forward(
                _x(), _x((1, 8, 3)).transpose(1, 2), _w(), eps=EPS
            ),
            ValueError,
            r"residual must be contiguous",
            id="residual-strided",
        ),
        pytest.param(
            lambda: rmsnorm_residual_forward(
                _x(), _x(dtype=torch.float64), _w(), eps=EPS
            ),
            TypeError,
            r"kernel dtypes",
            id="residual-float64",
        ),
        pytest.param(
            lambda: swiglu_forward(_x(), _x((1, 3, 7))),
            ValueError,
            r"up must be \(1, 3, 8\)",
            id="swiglu-shape",
        ),
        pytest.param(
            lambda: swiglu_forward(_x(), _x(dtype=torch.bfloat16)),
            TypeError,
            r"one dtype per call",
            id="swiglu-mixed-dtypes",
        ),
        pytest.param(
            lambda: swiglu_forward(_x().cpu(), _x().cpu()),
            ValueError,
            r"gate must be on a CUDA device",
            id="swiglu-host",
        ),
        pytest.param(
            lambda: swiglu_forward(_x((1, 8, 3)).transpose(1, 2), _x()),
            ValueError,
            r"gate must be contiguous",
            id="swiglu-gate-strided",
        ),
        pytest.param(
            lambda: swiglu_forward(_x(), _x((1, 8, 3)).transpose(1, 2)),
            ValueError,
            r"up must be contiguous",
            id="swiglu-up-strided",
        ),
        pytest.param(
            lambda: swiglu_forward(_x(dtype=torch.float64), _x(dtype=torch.float64)),
            TypeError,
            r"kernel dtypes",
            id="swiglu-float64",
        ),
        pytest.param(
            lambda: swiglu_forward(_x((1, 0, 8)), _x((1, 0, 8))),
            ValueError,
            r"at least one element",
            id="swiglu-empty",
        ),
        pytest.param(
            lambda: rmsnorm_backward(_x(), _x(), _w(7), eps=EPS),
            ValueError,
            r"weight must be \(8,\)",
            id="norm-bwd-shares-the-norm-checks",
        ),
        pytest.param(
            lambda: rmsnorm_backward(_x((1, 4, 8)), _x(), _w(), eps=EPS),
            ValueError,
            r"dout must be \(1, 3, 8\)",
            id="norm-bwd-dout-shape",
        ),
        pytest.param(
            lambda: rmsnorm_backward(_x(dtype=torch.bfloat16), _x(), _w(), eps=EPS),
            TypeError,
            r"one dtype per call",
            id="norm-bwd-dout-dtype",
        ),
        pytest.param(
            lambda: rmsnorm_backward(_x().cpu(), _x(), _w(), eps=EPS),
            ValueError,
            r"dout must be on a CUDA device",
            id="norm-bwd-dout-host",
        ),
        pytest.param(
            lambda: rmsnorm_backward(
                _x((1, 8, 3)).transpose(1, 2), _x(), _w(), eps=EPS
            ),
            ValueError,
            r"dout must be contiguous",
            id="norm-bwd-dout-strided",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                _x(), None, _x(), _x((1, 4, 8)), _w(), eps=EPS
            ),
            ValueError,
            r"residual must be \(1, 3, 8\)",
            id="residual-bwd-shares-the-stream-checks",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                _x((1, 4, 8)), None, _x(), None, _w(), eps=EPS
            ),
            ValueError,
            r"dnormed must be \(1, 3, 8\)",
            id="residual-bwd-dnormed-shape",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                _x(dtype=torch.bfloat16), None, _x(), None, _w(), eps=EPS
            ),
            TypeError,
            r"one dtype per call",
            id="residual-bwd-dnormed-dtype",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                _x((1, 8, 3)).transpose(1, 2), None, _x(), None, _w(), eps=EPS
            ),
            ValueError,
            r"dnormed must be contiguous",
            id="residual-bwd-dnormed-strided",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                None, _x((1, 4, 8)), _x(), None, _w(), eps=EPS
            ),
            ValueError,
            r"dresidual must be \(1, 3, 8\)",
            id="residual-bwd-dresidual-shape",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                None, _x(dtype=torch.bfloat16), _x(), None, _w(), eps=EPS
            ),
            ValueError,
            r"dresidual must be float32",
            id="residual-bwd-dresidual-narrow",
        ),
        pytest.param(
            lambda: rmsnorm_residual_backward(
                None, _x((1, 8, 3)).transpose(1, 2), _x(), None, _w(), eps=EPS
            ),
            ValueError,
            r"dresidual must be contiguous",
            id="residual-bwd-dresidual-strided",
        ),
        pytest.param(
            lambda: swiglu_backward(_x((1, 3, 7)), _x(), _x()),
            ValueError,
            r"dout must be \(1, 3, 8\)",
            id="swiglu-bwd-dout-shape",
        ),
        pytest.param(
            lambda: swiglu_backward(_x(dtype=torch.bfloat16), _x(), _x()),
            TypeError,
            r"one dtype per call",
            id="swiglu-bwd-dout-dtype",
        ),
        pytest.param(
            lambda: swiglu_backward(_x((1, 8, 3)).transpose(1, 2), _x(), _x()),
            ValueError,
            r"dout must be contiguous",
            id="swiglu-bwd-dout-strided",
        ),
    ],
)
def test_rejects_before_launch(call: Call, error: type[Exception], match: str) -> None:
    """Every guard on all six entry points, triggered on the host.

    A host pointer or an unsupported width handed to a launch faults inside CUDA
    and leaves the context unusable for the rest of the process, so a rejection
    that happens late is not a rejection. Each row here returns rather than
    reaching a launch.
    """
    with pytest.raises(error, match=match):
        call()


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------


def test_launch_geometry_and_smem_budget_hold() -> None:
    """The reductions and the vector loads rest on these, so they are asserted.

    A block width that is not whole warps leaves a partial warp whose shuffle
    total is undefined; a partial tile count breaks the one-partial-per-warp
    layout; a vector width that does not divide the load span reads past the end
    of a thread's slice; and a budget over the queried capacity fails at launch
    rather than at build. The budget comes from the tile layouts, with no slop
    constant.
    """
    assert NORM_THREADS % 32 == 0
    assert ACT_THREADS % 32 == 0
    assert WARPS * 32 == NORM_THREADS
    assert (FWD_SLOTS, BWD_SLOTS) == (1, 2)

    for slots, name in ((FWD_SLOTS, "rmsnorm_fwd"), (BWD_SLOTS, "rmsnorm_bwd")):
        assert reduce_tile(slots).shape == (slots * WARPS,)
        assert total_tile(slots).shape == (slots,)
        budget = norm_smem_bytes(slots)
        assert budget == smem_bytes([(reduce_tile(slots), 4), (total_tile(slots), 4)])
        assert budget == 4 * slots * (WARPS + 1)
        assert assert_smem_fits(name, budget) == budget
        assert budget < smem_capacity()

    for dtype in KERNEL_DTYPES:
        itemsize = torch.empty(0, dtype=dtype).element_size()
        assert VECTOR_BYTES % itemsize == 0
        assert VECTOR_BYTES // itemsize in (4, 8)


def test_backward_grid_meets_the_block_floor() -> None:
    """The row stride is the grid, and the grid is the block-count floor.

    Twice the SM count is the floor every kernel is held to. Fewer rows than that
    caps the grid at the row count, which is the whole available parallelism: a row
    reduction cannot cross a grid barrier. A grid above the row count would launch
    blocks whose stride loop never runs and whose partial row is never written,
    which the host sum would read as NaN.
    """
    index = torch.cuda.current_device()
    floor = 2 * sm_count(index)
    assert floor > 0
    assert row_blocks(1, index) == 1
    assert row_blocks(floor - 1, index) == floor - 1
    assert row_blocks(floor, index) == floor
    assert row_blocks(10 * floor + 1, index) == floor
