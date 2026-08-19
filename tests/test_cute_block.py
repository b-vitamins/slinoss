"""Block norm and activation: the CuTe forward kernels against the reference.

The authority is :mod:`slinoss.ops.block.reference` in float64. Operands are built
in float32, rounded once to the operand dtype, and only then upcast for the
oracle, so the kernel and the oracle read identical values and every difference
is arithmetic width and reduction order.

Every output buffer is poisoned with NaN and freed immediately before the call,
so the caching allocator hands the same block back to the kernel. Without that an
element the kernel never writes reads as whatever the allocator last held, and a
finiteness check passes on a kernel that skipped it.
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
from slinoss.ops.block import rmsnorm_ref, rmsnorm_residual_ref, swiglu_ref
from slinoss.ops.block.cute import (
    ACT_THREADS,
    NORM_THREADS,
    PARTIAL_TILE,
    SCALE_TILE,
    VECTOR_BYTES,
    WARPS,
    norm_smem_bytes,
    rmsnorm_forward,
    rmsnorm_residual_forward,
    swiglu_forward,
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
    ],
)
def test_rejects_before_launch(call: Call, error: type[Exception], match: str) -> None:
    """Every guard on all three entry points, triggered on the host.

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
    assert PARTIAL_TILE.shape == (WARPS,)
    assert SCALE_TILE.shape == (1,)

    budget = norm_smem_bytes()
    assert budget == smem_bytes([(PARTIAL_TILE, 4), (SCALE_TILE, 4)])
    assert budget == 4 * (WARPS + 1)
    assert assert_smem_fits("rmsnorm_fwd", budget) == budget
    assert budget < smem_capacity()

    for dtype in KERNEL_DTYPES:
        itemsize = torch.empty(0, dtype=dtype).element_size()
        assert VECTOR_BYTES % itemsize == 0
        assert VECTOR_BYTES // itemsize in (4, 8)
