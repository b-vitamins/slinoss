"""The scan's parameter frontier: the CuTe kernels against the float64 reference.

The authority is :func:`slinoss.ops.scanprep.scanprep_ref` in float64, and the
gradient authority is :func:`slinoss.ops.scanprep.scanprep_bwd_ref`, which is
autograd through it. A hand-derived VJP would share its algebra with the kernel, so
an algebra error would pass silently. The float64 gradcheck that pins the authority
itself lives in ``tests/test_scanprep.py``, so it is not repeated here.

Operands are built once in float32 and cast down, never built twice at two dtypes:
the generator consumes a different number of raw words per element at each width,
so the same seed at two dtypes is two different problems. The oracle reads the cast
operands, so the kernel and the oracle evaluate the maps at identical values at
every width.

Both operands are cut out of one wider row at aligned column offsets, which is the
shipped layout: the mixer runs one projection GEMM and hands out views. The sweep
narrows them to their own allocation at the smallest legal row pitch, and one test
keeps them at the projection pitch and demands bitwise equality, because a kernel
that only handled one pitch would pass every other test here.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

import cutlass.cute as cute

from slinoss._guard import ALIGN_BYTES
from slinoss.config import STATE_MULTIPLE
from slinoss.ops.scanprep import (
    PARAM_COLS,
    ScanGrads,
    ScanParams,
    scanprep,
    scanprep_bwd_ref,
    scanprep_ref,
)
from slinoss.ops.scanprep.cute import (
    THREADS,
    TILE_TOKENS,
    scanprep_backward,
    scanprep_forward,
)
from tests.conftest import W_MAX, assert_max_rel, max_err

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

Shape = tuple[int, int, int, int, int]
Operands = tuple[Tensor, Tensor, Tensor]
Cotangents = tuple[Tensor, Tensor, Tensor, Tensor]
FwdMutator = Callable[[Tensor, Tensor, Tensor], Operands]
BwdCase = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
BwdMutator = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], BwdCase]

# (bsz, heads, seqlen, groups, state_dim). One block covers TILE_TOKENS tokens of
# one batch, so the sweep is over T against that tile and over the head and group
# counts the tile arithmetic is compiled for: one exact tile, a tail shorter than a
# tile, a ragged tail above three tiles at G = H, many exact tiles at a head and
# group count whose work counts divide the block width so the tail predicates
# elide, the widest state at the mixer's head count, and a single token. B = 1,
# H = 1, G = 1, and the smallest legal 3N all appear. Every T is written against
# TILE_TOKENS, so shrinking the tile cannot silently retire the ragged cases.
SHAPES: list[Shape] = [
    (1, 1, TILE_TOKENS, 1, STATE_MULTIPLE),
    (1, 1, TILE_TOKENS - 1, 1, STATE_MULTIPLE),
    (2, 3, 25 * TILE_TOKENS + 1, 3, STATE_MULTIPLE),
    (4, 32, 8 * TILE_TOKENS, 2, STATE_MULTIPLE),
    (2, 12, 50 * TILE_TOKENS, 1, 2 * STATE_MULTIPLE),
    (1, 1, 1, 1, STATE_MULTIPLE),
]
SHAPE_IDS = [
    "one-exact-tile",
    "tail-only",
    "ragged-tail-grouped",
    "many-tiles-exact",
    "wide-state",
    "single-token",
]

DTYPES = [torch.float32, torch.bfloat16]

# One representative shape for the tests whose subject is not the shape.
ONE = SHAPES[2]

W_SCALE = 1.5
"""Multiplies the raw rotation columns, so ``|w|`` sits near ``w_max`` rather than
near zero and the saturating part of the map is exercised."""

# The forward is float32 arithmetic over exactly representable inputs at every
# width, so the bound is float32 rounding of the bias add and the map. The rsqrt,
# the exp2 and the log are the hardware approximations, whose relative error is
# 2^-22, and the errors are read against the maximum magnitude of the tensor.
FWD_TOL = 1e-6

# The gradients are stored at the input width. bfloat16 keeps 8 significand bits,
# so a stored gradient carries up to a half-ulp, 2^-8 = 3.9e-3, of relative
# rounding; the bound is that, rounded up. Storage dominates, so there is no margin
# to spend on the arithmetic and none is needed: the arithmetic is float32. float32
# storage leaves only the arithmetic.
BWD_TOL = {torch.float32: 1e-6, torch.bfloat16: 4e-3}

# dparam_bias is reduced in float32 over the tokens of a tile and then over the
# tiles, against a float64 sum over every token in the reference. Both cotangents
# are float32 whatever the activation width, so the width of the terms is not the
# bound; the different summation order is.
BIAS_TOL = 1e-5

# The exact map lands in the closed ball of radius w_max; the computed vector is
# that value rounded twice, so its norm can sit a few ulp outside. Same argument
# and same constant as the reference's own bound.
BALL_BOUND = W_MAX * (1.0 + 3.0 * torch.finfo(torch.float32).eps)

EXTREME_RAWS = (-1e8, -1e4, -20.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 20.0, 1e4, 1e8)

ALIGN_ELEMS = ALIGN_BYTES // 2
"""Column multiple that keeps a slice's base address and row pitch aligned at every
activation width. The requirement is ``ALIGN_BYTES // itemsize`` elements, which is
eight at the narrowest activation and divides the four float32 needs, so one
constant covers the sweep."""


def _align(columns: int) -> int:
    """Round a column count up to :data:`ALIGN_ELEMS`."""
    return -(-columns // ALIGN_ELEMS) * ALIGN_ELEMS


def _narrow(view: Tensor) -> Tensor:
    """Copy one band into its own allocation at the narrowest legal row pitch.

    Not ``contiguous()``. The pitched contract requires the row pitch to step on the
    alignment as well as the base to start on it, so a contiguous ``(B,T,width)`` is
    legal only when ``width`` is already a multiple of :data:`ALIGN_ELEMS`; at
    ``H = 1`` it is not. The band is therefore padded, which is what the producer
    does to its projection width. The pitch still differs from the wide row's, so
    both ends of the runtime pitch path are covered.
    """
    width = int(view.shape[-1])
    own = torch.empty(
        *view.shape[:-1], _align(width), dtype=view.dtype, device=view.device
    )
    band = own[..., :width]
    band.copy_(view)
    return band


def _tag(shape: Shape, dtype: torch.dtype) -> str:
    bsz, heads, seqlen, groups, state_dim = shape
    width = str(dtype).removeprefix("torch.")
    return f"cute-scanprep[{bsz}x{heads}x{seqlen}/G{groups}/{state_dim}/{width}]"


def _operands(
    shape: Shape,
    dtype: torch.dtype = torch.float32,
    *,
    seed: int = 0,
    bias: float = 1.0,
    strided: bool = False,
) -> Operands:
    """``(params, bc, param_bias)`` on CUDA.

    Both projection slices are cut out of one row at aligned column offsets, with
    padding before, between, and after them, so a kernel that assumed a compact
    operand reads the wrong columns. ``strided`` leaves them as views; otherwise
    they are narrowed by :func:`_narrow`, and the two hold the same values.
    """
    bsz, heads, seqlen, groups, state_dim = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pwidth = heads * PARAM_COLS
    bwidth = 2 * groups * state_dim
    poff = ALIGN_ELEMS
    boff = _align(poff + pwidth)
    width = _align(boff + bwidth) + ALIGN_ELEMS
    row = torch.randn(
        bsz, seqlen, width, generator=gen, dtype=torch.float32, device="cuda"
    )
    row[..., poff : poff + 3] *= W_SCALE
    row = row.to(dtype)
    params = row[..., poff : poff + pwidth]
    bc = row[..., boff : boff + bwidth]
    if not strided:
        params, bc = _narrow(params), _narrow(bc)
    pbias = torch.randn(
        heads, PARAM_COLS, generator=gen, dtype=torch.float32, device="cuda"
    )
    return params, bc, pbias * bias


def _cotangents(shape: Shape, dtype: torch.dtype, *, seed: int) -> Cotangents:
    """``(dtrans, dK, dB, dC)``. The packed pair is float32 whatever the width,
    because I4 pins both packed outputs."""
    bsz, heads, seqlen, groups, state_dim = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*size: int) -> Tensor:
        return torch.randn(*size, generator=gen, dtype=torch.float32, device="cuda")

    return (
        rnd(bsz, heads, seqlen, 4),
        rnd(bsz, heads, seqlen, 2, 4),
        rnd(bsz, groups, seqlen, state_dim).to(dtype),
        rnd(bsz, groups, seqlen, state_dim).to(dtype),
    )


def _forward(ops: Operands, shape: Shape) -> ScanParams:
    return scanprep_forward(*ops, heads=shape[1], state_dim=shape[4], w_max=W_MAX)


def _oracle(ops: Operands, shape: Shape) -> ScanParams:
    params, bc, pbias = ops
    return scanprep_ref(
        params.double(),
        bc.double(),
        pbias.double(),
        heads=shape[1],
        state_dim=shape[4],
        w_max=W_MAX,
    )


def _backward(cots: Cotangents, ops: Operands, shape: Shape) -> ScanGrads:
    return scanprep_backward(
        *cots, ops[0], ops[2], heads=shape[1], state_dim=shape[4], w_max=W_MAX
    )


def _oracle_grads(cots: Cotangents, ops: Operands, shape: Shape) -> ScanGrads:
    dtrans, dK, dB, dC = cots
    return scanprep_bwd_ref(
        dtrans.double(),
        dK.double(),
        dB.double(),
        dC.double(),
        ops[0].double(),
        ops[2].double(),
        heads=shape[1],
        state_dim=shape[4],
        w_max=W_MAX,
    )


def _leaves(ops: Operands, *, double: bool) -> Operands:
    """The same operands as differentiable leaves. The upcast is exact.

    The two projection slices are rebuilt through :func:`_narrow` rather than
    ``clone``, which would give them a row pitch off the alignment at a head count
    whose row is not already a multiple of :data:`ALIGN_ELEMS`.
    """
    params, bc, pbias = ops
    if double:
        params, bc, pbias = params.double(), bc.double(), pbias.double()
    return (
        _narrow(params).detach().requires_grad_(),
        _narrow(bc).detach().requires_grad_(),
        pbias.detach().clone().requires_grad_(),
    )


def _grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_forward_matches_reference(shape: Shape, dtype: torch.dtype) -> None:
    """Every output the scan reads, against the float64 frontier.

    ``B`` and ``C`` are a permute of the operand, so their bound is equality: the
    kernel converts nothing on that path and a tolerance there would hide a
    misplaced column.
    """
    bsz, heads, seqlen, groups, state_dim = shape
    ops = _operands(shape, dtype, seed=1)
    want = _oracle(ops, shape)

    got = _forward(ops, shape)
    torch.cuda.synchronize()

    assert got.trans.shape == (bsz, heads, seqlen, 4)
    assert got.K.shape == (bsz, heads, seqlen, 2, 4)
    assert got.B.shape == (bsz, groups, seqlen, state_dim)
    assert got.C.shape == got.B.shape
    assert got.trans.dtype is torch.float32
    assert got.K.dtype is torch.float32
    assert got.B.dtype is dtype
    assert got.C.dtype is dtype
    assert all(t.is_contiguous() for t in got)

    tag = _tag(shape, dtype)
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    assert_max_rel(got.K, want.K, FWD_TOL, f"{tag}.K")
    bc = ops[1]
    half = groups * state_dim
    for grp in range(groups):
        lo = grp * state_dim
        assert torch.equal(got.B[:, grp], bc[..., lo : lo + state_dim])
        assert torch.equal(got.C[:, grp], bc[..., half + lo : half + lo + state_dim])


def test_lane_three_is_a_hard_zero() -> None:
    """Lane 3 is written by the kernel, not inherited from a zeroed allocation.

    The poison is freed immediately before the call so the caching allocator hands
    the same block back to the output; without it the check passes on any allocator
    that happens to return zeros.
    """
    bsz, heads, seqlen, _, _ = ONE
    poison = torch.full((bsz, heads, seqlen, 2, 4), float("nan"), device="cuda")
    assert bool(poison.isnan().all())
    del poison

    got = _forward(_operands(ONE, seed=6), ONE)
    torch.cuda.synchronize()
    assert torch.count_nonzero(got.K[..., 3]) == 0


def test_kernels_read_a_projection_slice_without_repacking_it() -> None:
    """Bitwise equality against narrowed operands, in both directions.

    The row pitch is taken from the operand at runtime, so the wide-pitch and the
    narrow-pitch call are the same executor and must produce identical bits. The
    backward reads ``params`` the same way, so it is checked in the same test
    rather than under a second fixture.
    """
    params, bc, pbias = _operands(ONE, seed=3, strided=True)
    assert params.stride(-1) == 1 and not params.is_contiguous()
    assert bc.stride(-1) == 1 and not bc.is_contiguous()
    compact = (_narrow(params), _narrow(bc), pbias)
    assert compact[0].stride(-2) != params.stride(-2)
    cots = _cotangents(ONE, torch.float32, seed=4)

    strided_out = _forward((params, bc, pbias), ONE)
    compact_out = _forward(compact, ONE)
    strided_grads = _backward(cots, (params, bc, pbias), ONE)
    compact_grads = _backward(cots, compact, ONE)
    torch.cuda.synchronize()

    for got, ref in zip(strided_out, compact_out, strict=True):
        assert torch.equal(got, ref)
    for got, ref in zip(strided_grads, compact_grads, strict=True):
        assert torch.equal(got, ref)


def test_outputs_stay_float32_under_autocast() -> None:
    """I4 pins the packed pair whatever the ambient autocast dtype. No
    ``custom_fwd``, so nothing casts the operands on the way in either."""
    ops = _operands(SHAPES[0], seed=7)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        got = _forward(ops, SHAPES[0])
    assert got.trans.dtype is torch.float32
    assert got.K.dtype is torch.float32
    assert got.B.dtype is torch.float32


def test_float16_takes_its_own_kernel_path() -> None:
    """float16 keys a distinct executor, so it is traced and checked once.

    The only dtype-dependent code is the widen on load and the narrow on store,
    neither of which interacts with the shape, so one shape covers the axis.
    """
    ops = _operands(SHAPES[0], torch.float16, seed=12)
    cots = _cotangents(SHAPES[0], torch.float16, seed=13)
    want = _oracle(ops, SHAPES[0])
    want_grads = _oracle_grads(cots, ops, SHAPES[0])

    got = _forward(ops, SHAPES[0])
    got_grads = _backward(cots, ops, SHAPES[0])
    torch.cuda.synchronize()

    assert got.B.dtype is torch.float16
    assert got_grads.dparams.dtype is torch.float16
    tag = _tag(SHAPES[0], torch.float16)
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    # float16 keeps 11 significand bits, so a stored gradient carries up to a
    # half-ulp, 2^-11 = 4.9e-4, of relative rounding; the bound is that, rounded up.
    assert_max_rel(got_grads.dparams, want_grads.dparams, 1e-3, f"{tag}.dparams")


def _extreme_operands() -> Operands:
    """One token per entry of :data:`EXTREME_RAWS`, in every parameter column.

    The bias is zero, so the biased row is the raw row exactly and the oracle
    evaluates the maps at the same float32 values the kernel does.
    """
    vals = torch.tensor(EXTREME_RAWS, dtype=torch.float32, device="cuda")
    count = int(vals.numel())
    params = vals[:, None].expand(count, PARAM_COLS).reshape(1, count, PARAM_COLS)
    return (
        _narrow(params),
        torch.zeros(1, count, 2 * STATE_MULTIPLE, dtype=torch.float32, device="cuda"),
        torch.zeros(1, PARAM_COLS, dtype=torch.float32, device="cuda"),
    )


def test_extreme_raws_match_the_reference() -> None:
    """Parity and both invariants across the reachable float32 domain.

    I1 and I2 are produced by this kernel, so they are asserted on its output, and
    the domain that matters is the reachable one rather than a normal draw.
    Magnitudes stop at 1e8 because ``|raw|^2`` overflows float32 near 1.8e19; past
    that the float32 map and the float64 oracle disagree by width, not by kernel,
    which is the next test.
    """
    shape: Shape = (1, 1, len(EXTREME_RAWS), 1, STATE_MULTIPLE)
    ops = _extreme_operands()
    want = _oracle(ops, shape)

    got = _forward(ops, shape)
    torch.cuda.synchronize()

    assert bool((got.trans[..., 3] <= 0.0).all())
    assert bool((got.trans[..., :3].double().norm(dim=-1) <= BALL_BOUND).all())
    assert bool(torch.isfinite(got.trans).all())
    assert bool(torch.isfinite(got.K).all())
    assert_max_rel(
        got.trans[..., :3], want.trans[..., :3], FWD_TOL, "cute-scanprep.extreme.w"
    )
    # The taps are the identity on the biased row, and the bias is zero here, so
    # the widening is exact and equality is the honest bound.
    assert torch.equal(got.K[..., :3].double(), want.K[..., :3])
    # The log-scale column spans 1e8 down to zero, so a bound relative to the
    # column maximum is vacuous. The absolute bound is float32 rounding of log1p
    # near its largest reachable argument.
    assert max_err(got.trans[..., 3], want.trans[..., 3]) < 1e-6


def test_overflowing_raw_norm_stays_finite() -> None:
    """``|raw|^2`` overflows float32 near 1.8e19, and ``rsqrt(inf)`` is zero.

    The map collapses to the centre of the ball rather than producing a NaN, which
    is all I2 claims. The float64 oracle does not overflow, so this is a property
    check and not a parity check.
    """
    shape: Shape = (1, 1, 8, 1, STATE_MULTIPLE)
    huge = torch.full((1, 8, PARAM_COLS), 1e30, dtype=torch.float32, device="cuda")
    ops = (
        _narrow(huge),
        torch.zeros(1, 8, 2 * STATE_MULTIPLE, dtype=torch.float32, device="cuda"),
        torch.zeros(1, PARAM_COLS, dtype=torch.float32, device="cuda"),
    )
    got = _forward(ops, shape)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(got.trans).all())
    assert bool((got.trans[..., 3] <= 0.0).all())
    assert float(got.trans[..., :3].double().norm(dim=-1).max()) <= BALL_BOUND


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_backward_matches_reference_autograd(shape: Shape, dtype: torch.dtype) -> None:
    """All three gradients against float64 autograd through the reference.

    ``dbc`` is the inverse permute of ``dB`` and ``dC``, so its bound is equality
    after widening: the kernel moves those bits and computes nothing.
    """
    bsz, heads, seqlen, groups, state_dim = shape
    ops = _operands(shape, dtype, seed=1)
    cots = _cotangents(shape, dtype, seed=2)
    want = _oracle_grads(cots, ops, shape)

    got = _backward(cots, ops, shape)
    torch.cuda.synchronize()

    assert got.dparams.shape == (bsz, seqlen, heads * PARAM_COLS)
    assert got.dbc.shape == (bsz, seqlen, 2 * groups * state_dim)
    assert got.dparam_bias.shape == (heads, PARAM_COLS)
    assert got.dparams.dtype is dtype
    assert got.dbc.dtype is dtype
    assert got.dparam_bias.dtype is torch.float32
    assert all(t.is_contiguous() for t in got)

    tag = _tag(shape, dtype)
    assert_max_rel(got.dparams, want.dparams, BWD_TOL[dtype], f"{tag}.dparams")
    assert_max_rel(got.dparam_bias, want.dparam_bias, BIAS_TOL, f"{tag}.dparam_bias")
    assert torch.equal(got.dbc.double(), want.dbc)


def test_backward_ignores_the_lane_three_cotangent() -> None:
    """Lane 3 of each tap is a constant zero, so its cotangent is the cotangent of
    nothing. A pullback that read it would leak into ``dparams``."""
    ops = _operands(ONE, seed=8)
    dtrans, dK, dB, dC = _cotangents(ONE, torch.float32, seed=9)
    loud = dK.clone()
    loud[..., 3] = 1e6

    quiet = _backward((dtrans, dK, dB, dC), ops, ONE)
    got = _backward((dtrans, loud, dB, dC), ops, ONE)
    torch.cuda.synchronize()

    assert torch.equal(got.dparams, quiet.dparams)
    assert torch.equal(got.dparam_bias, quiet.dparam_bias)
    assert float(quiet.dparams.abs().max()) > 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_backward_end_to_end(dtype: torch.dtype) -> None:
    """The fast forward, backpropagated through, against the float64 reference.

    The kernel forward feeds the kernel backward here, through the public operator
    and its registry dispatch, so a disagreement between the two cannot hide behind
    a surrogate forward.
    """
    heads, state_dim = ONE[1], ONE[4]
    ops = _operands(ONE, dtype, seed=10)
    dtrans, dK, dB, dC = _cotangents(ONE, dtype, seed=11)
    fast = _leaves(ops, double=False)
    oracle = _leaves(ops, double=True)

    got = scanprep(*fast, heads=heads, state_dim=state_dim, w_max=W_MAX, backend="cute")
    total = (got.trans * dtrans).sum() + (got.K * dK).sum()
    (total + (got.B * dB).sum() + (got.C * dC).sum()).backward()

    want = scanprep_ref(*oracle, heads=heads, state_dim=state_dim, w_max=W_MAX)
    ref = (want.trans * dtrans.double()).sum() + (want.K * dK.double()).sum()
    (ref + (want.B * dB.double()).sum() + (want.C * dC.double()).sum()).backward()
    torch.cuda.synchronize()

    tag = f"{_tag(ONE, dtype)}.e2e"
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    assert_max_rel(_grad(fast[0]), _grad(oracle[0]), BWD_TOL[dtype], f"{tag}.dparams")
    assert_max_rel(_grad(fast[2]), _grad(oracle[2]), BIAS_TOL, f"{tag}.dparam_bias")
    assert torch.equal(_grad(fast[1]).double(), _grad(oracle[1]))


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

REJECT: Shape = (2, 3, 8, 1, STATE_MULTIPLE)


def _unaligned(shape: Shape) -> Tensor:
    """A ``params``-shaped slice whose base address is one element in."""
    bsz, heads, seqlen, _, _ = shape
    width = heads * PARAM_COLS
    row = torch.randn(bsz, seqlen, width + 2, dtype=torch.float32, device="cuda")
    return row[..., 1 : 1 + width]


def _stride_two(shape: Shape) -> Tensor:
    """A ``params``-shaped view whose trailing stride is two."""
    bsz, heads, seqlen, _, _ = shape
    width = heads * PARAM_COLS
    row = torch.randn(bsz, seqlen, 2 * width, dtype=torch.float32, device="cuda")
    return row[..., ::2]


def _gapped(tensor: Tensor) -> Tensor:
    """The same values in a view that is not contiguous."""
    doubled = torch.cat([tensor, tensor], dim=-1)
    return doubled[..., : tensor.shape[-1]]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda p, b, pb: (p.double(), b.double(), pb.double()),
            TypeError,
            r"kernel dtypes",
        ),
        (
            lambda p, b, pb: (p.bfloat16(), b, pb),
            TypeError,
            r"one activation dtype per call",
        ),
        (
            lambda p, b, pb: (p.cpu(), b.cpu(), pb.cpu()),
            ValueError,
            r"must be on a CUDA device",
        ),
        (
            lambda p, b, pb: (_unaligned(REJECT), b, pb),
            ValueError,
            r"must start and step on a multiple of",
        ),
        (
            lambda p, b, pb: (p[:, :0], b[:, :0], pb),
            ValueError,
            r"at least one token",
        ),
        (
            lambda p, b, pb: (p, b, _gapped(pb)),
            ValueError,
            r"param_bias must be contiguous",
        ),
    ],
)
def test_forward_rejects(
    mutate: FwdMutator, error: type[Exception], match: str
) -> None:
    """Every guard the kernel host path owns: the narrower kernel dtype set, device
    residency, a launchable token count, and the contiguity of the one operand that
    is not a projection slice.

    Two cases here belong to contracts this path shares rather than owns, and each
    is one case, not that contract's table: the mixed-dtype case pins that this path
    runs the shared shape checker, and the misaligned case pins that it runs the
    pitched-layout checker. The pitched table itself, including the pitch multiple
    and the overlapping-row rejection, is ``tests/test_guard.py``'s. The rest of the
    shape contract is covered against the reference in ``test_scanprep.py``.
    """
    with pytest.raises(error, match=match):
        _forward(mutate(*_operands(REJECT, seed=3)), REJECT)


@pytest.mark.parametrize("w_max", [0.0, -1.0, 4.0, float("inf")])
def test_forward_rejects_illegal_bound(w_max: float) -> None:
    """I2 needs a bound strictly inside ``(0, pi)``."""
    params, bc, pbias = _operands(REJECT, seed=3)
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_forward(
            params, bc, pbias, heads=REJECT[1], state_dim=REJECT[4], w_max=w_max
        )


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda dt, dk, db, dc, p: (dt.bfloat16(), dk, db, dc, p),
            ValueError,
            r"dtrans must be float32",
        ),
        (
            lambda dt, dk, db, dc, p: (dt, dk.bfloat16(), db, dc, p),
            ValueError,
            r"dK must be float32",
        ),
        (
            lambda dt, dk, db, dc, p: (dt, dk, db.double(), dc, p),
            TypeError,
            r"kernel dtypes",
        ),
        (
            lambda dt, dk, db, dc, p: (dt[..., :3], dk, db, dc, p),
            ValueError,
            r"dtrans must be",
        ),
        (
            lambda dt, dk, db, dc, p: (dt, dk, _gapped(db), dc, p),
            ValueError,
            r"dB must be contiguous",
        ),
        (
            lambda dt, dk, db, dc, p: (dt, dk, db, dc, _stride_two(REJECT)),
            ValueError,
            r"params must have unit stride",
        ),
        (
            lambda dt, dk, db, dc, p: (dt, dk, db, dc, p.cpu()),
            ValueError,
            r"must be on a CUDA device",
        ),
        (
            lambda dt, dk, db, dc, p: (
                dt[:, :, :0],
                dk[:, :, :0],
                db[:, :, :0],
                dc[:, :, :0],
                p[:, :0],
            ),
            ValueError,
            r"at least one token",
        ),
    ],
)
def test_backward_rejects(
    mutate: BwdMutator, error: type[Exception], match: str
) -> None:
    """Every guard the backward owns. I4 pins both packed cotangents to float32, so
    a low-precision one is refused rather than promoted, and ``params`` is
    revalidated here because the backward reads it and the forward's checker does
    not run again."""
    ops = _operands(REJECT, seed=3)
    cots = _cotangents(REJECT, torch.float32, seed=4)
    dtrans, dK, dB, dC, params = mutate(*cots, ops[0])
    with pytest.raises(error, match=match):
        scanprep_backward(
            dtrans,
            dK,
            dB,
            dC,
            params,
            ops[2],
            heads=REJECT[1],
            state_dim=REJECT[4],
            w_max=W_MAX,
        )


def test_backward_rejects_illegal_bound() -> None:
    """The bound scales the gradient, so the backward checks it too."""
    ops = _operands(REJECT, seed=3)
    cots = _cotangents(REJECT, torch.float32, seed=4)
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_backward(
            *cots, ops[0], ops[2], heads=REJECT[1], state_dim=REJECT[4], w_max=4.0
        )


# ---------------------------------------------------------------------------
# Launch geometry
# ---------------------------------------------------------------------------


def test_the_launch_geometry_admits_the_bias_shuffle() -> None:
    """The two constraints the kernel's phases rest on, neither of them free.

    A block width off a warp multiple leaves a partial warp in every phase, because
    every phase divides its item count by ``THREADS`` and the remainder lands on one
    warp. A tile that does not divide the warp width splits a token run across two
    warps, and the backward's bias partial is reduced over that run by shuffle, which
    reaches only inside a warp; splitting it would drop the tokens on the far side.
    """
    assert THREADS % cute.arch.WARP_SIZE == 0
    assert cute.arch.WARP_SIZE % TILE_TOKENS == 0
