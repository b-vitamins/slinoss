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

``params`` is cut out of one wider row at an aligned column offset, which is the
shipped layout: the mixer runs one projection GEMM and hands out views. The sweep
narrows it to its own allocation at the smallest legal row pitch, and one test keeps
it at the projection pitch and demands bitwise equality, because a kernel that only
handled one pitch would pass every other test here. The backward's ``dparams``
destination is the same geometry on the store side and is checked the same way.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

import cutlass.cute as cute

from slinoss._guard import PROJ_ALIGN
from slinoss.ops.scanprep import (
    LS_MAX_MAG,
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
from slinoss.ops.so3ssd import tap_matrix
from tests.conftest import W_MAX, assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

Shape = tuple[int, int, int]
Operands = tuple[Tensor, Tensor]
Cotangents = tuple[Tensor, Tensor]
FwdMutator = Callable[[Tensor, Tensor], Operands]
BwdCase = tuple[Tensor, Tensor, Tensor]
BwdMutator = Callable[[Tensor, Tensor, Tensor], BwdCase]
Backward = Callable[..., ScanGrads]

BACKWARDS = [
    pytest.param(scanprep_backward, id="cute"),
    pytest.param(scanprep_bwd_ref, id="reference"),
]
"""Both backends' backward. The gradient destination is one contract across them,
and the reference's own tests run on the CPU, where the pitched-layout rule the
destination is held to does not apply."""

# (bsz, heads, seqlen). One block covers TILE_TOKENS tokens of one batch, so the
# sweep is over T against that tile and over the head counts the tile arithmetic is
# compiled for: one exact tile, a tail shorter than a tile, a ragged tail above
# three tiles, many exact tiles at a head count whose work counts divide the block
# width so the tail predicates elide, the mixer's own head count over many tiles,
# and a single token. B = 1 and H = 1 both appear. Every T is written against
# TILE_TOKENS, so shrinking the tile cannot silently retire the ragged cases.
SHAPES: list[Shape] = [
    (1, 1, TILE_TOKENS),
    (1, 1, TILE_TOKENS - 1),
    (2, 3, 25 * TILE_TOKENS + 1),
    (4, 32, 8 * TILE_TOKENS),
    (2, 12, 50 * TILE_TOKENS),
    (1, 1, 1),
]
SHAPE_IDS = [
    "one-exact-tile",
    "tail-only",
    "ragged-tail",
    "many-tiles-exact",
    "mixer-heads",
    "single-token",
]

DTYPES = [torch.float32, torch.bfloat16]

# One representative shape for the tests whose subject is not the shape.
ONE = SHAPES[2]

W_SCALE = 1.5
"""Multiplies both additive rotation operands, moving the chart away from zero."""

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

# dtransition_bias is reduced in float32 over the tokens of a tile and then over the
# tiles, against a float64 sum over every token in the reference. Both cotangents
# are float32 whatever the activation width, so the width of the terms is not the
# bound; the different summation order is.
BIAS_TOL = 1e-5

# The exact map lands in the closed ball of radius 2*w_max; the computed vector is
# that value rounded twice, so its norm can sit a few ulp outside. Same argument
# and same constant as the reference's own bound.
BALL_BOUND = 2.0 * W_MAX * (1.0 + 3.0 * torch.finfo(torch.float32).eps)

EXTREME_RAWS = (-1e8, -1e4, -20.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 20.0, 1e4, 1e8)


def _align(columns: int) -> int:
    """Round a column count up to :data:`slinoss._guard.PROJ_ALIGN`."""
    return -(-columns // PROJ_ALIGN) * PROJ_ALIGN


def _narrow(view: Tensor) -> Tensor:
    """Copy one band into its own allocation at the narrowest legal row pitch.

    Not ``contiguous()``. The pitched contract requires the row pitch to step on the
    alignment as well as the base to start on it, so a contiguous ``(B,T,width)`` is
    legal only when ``width`` is already a multiple of the padding column count; at
    ``H = 1`` it is not. The band is therefore padded, at the producer's own multiple:
    once the pitch exceeds the row width the operand is a strict band, which owes the
    sector and not the vector width. The pitch still differs from the wide row's, so
    both ends of the runtime pitch path are covered.
    """
    width = int(view.shape[-1])
    own = torch.empty(
        *view.shape[:-1], _align(width), dtype=view.dtype, device=view.device
    )
    band = own[..., :width]
    band.copy_(view)
    return band


def _matrices(out: ScanParams) -> Tensor:
    """Both tap operators as explicit float64 matrices, ``(B,H,T,2,3,3)``.

    ``K`` is compared here rather than on the chart. ``g`` carries ``1/|w|^2``, so
    what float32 holds of it is an absolute accuracy and not a relative one, while
    ``g * w w^T`` -- the only form the scan reads it in -- is regular as ``|w|``
    falls. A chart-level bound would be a claim about a corner no kernel downstream
    can observe.
    """
    return tap_matrix(out.K[..., :3].double(), out.trans[..., None, :3].double())


def _tag(shape: Shape, dtype: torch.dtype) -> str:
    bsz, heads, seqlen = shape
    width = str(dtype).removeprefix("torch.")
    return f"cute-scanprep[{bsz}x{heads}x{seqlen}/{width}]"


def _operands(
    shape: Shape,
    dtype: torch.dtype = torch.float32,
    *,
    seed: int = 0,
    bias: float = 1.0,
    strided: bool = False,
) -> Operands:
    """``(params, transition_bias)`` on CUDA.

    The projection slice is cut out of a wider row at an aligned column offset, with
    padding before and after it, so a kernel that assumed a compact operand reads
    the wrong columns. ``strided`` leaves it as a view; otherwise it is narrowed by
    :func:`_narrow`, and the two hold the same values.
    """
    bsz, heads, seqlen = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pwidth = heads * PARAM_COLS
    poff = PROJ_ALIGN
    width = _align(poff + pwidth) + PROJ_ALIGN
    row = torch.randn(
        bsz, seqlen, width, generator=gen, dtype=torch.float32, device="cuda"
    )
    row[..., poff : poff + 3] *= W_SCALE
    row = row.to(dtype)
    params = row[..., poff : poff + pwidth]
    if not strided:
        params = _narrow(params)
    pbias = torch.randn(
        heads, PARAM_COLS, generator=gen, dtype=torch.float32, device="cuda"
    )
    pbias[:, :3] *= W_SCALE
    return params, pbias * bias


def _cotangents(shape: Shape, *, seed: int) -> Cotangents:
    """``(dtrans, dK)``. Both are float32 whatever the activation width, because I4
    pins both packed outputs."""
    bsz, heads, seqlen = shape
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*size: int) -> Tensor:
        return torch.randn(*size, generator=gen, dtype=torch.float32, device="cuda")

    return rnd(bsz, heads, seqlen, 4), rnd(bsz, heads, seqlen, 2, 4)


def _forward(ops: Operands, shape: Shape) -> ScanParams:
    return scanprep_forward(*ops, heads=shape[1], w_max=W_MAX)


def _oracle(ops: Operands, shape: Shape) -> ScanParams:
    params, pbias = ops
    return scanprep_ref(params.double(), pbias.double(), heads=shape[1], w_max=W_MAX)


def _backward(cots: Cotangents, ops: Operands, shape: Shape) -> ScanGrads:
    return scanprep_backward(*cots, *ops, heads=shape[1], w_max=W_MAX)


def _oracle_grads(cots: Cotangents, ops: Operands, shape: Shape) -> ScanGrads:
    dtrans, dK = cots
    return scanprep_bwd_ref(
        dtrans.double(),
        dK.double(),
        ops[0].double(),
        ops[1].double(),
        heads=shape[1],
        w_max=W_MAX,
    )


def _leaves(ops: Operands, *, double: bool) -> Operands:
    """The same operands as differentiable leaves. The upcast is exact.

    The projection slice is rebuilt through :func:`_narrow` rather than ``clone``,
    which would give it a row pitch off the alignment at a head count whose row is
    not already a multiple of :data:`slinoss._guard.PROJ_ALIGN`.
    """
    params, pbias = ops
    if double:
        params, pbias = params.double(), pbias.double()
    return (
        _narrow(params).detach().requires_grad_(),
        pbias.detach().clone().requires_grad_(),
    )


def _grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def _band_dest(shape: Shape) -> Operands:
    """A NaN-filled wider buffer and the ``dparams``-shaped band inside it.

    The mixer's ``dproj``: the band sits at an aligned column offset with padding on
    both sides, so its row pitch exceeds its row width and any write outside it lands
    on a NaN column that is checked afterwards. float32 only, because what the store
    addresses through is the row pitch and not the element width.
    """
    bsz, heads, seqlen = shape
    width = heads * PARAM_COLS
    wide = torch.full(
        (bsz, seqlen, _align(width) + 2 * PROJ_ALIGN),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    return wide, wide[..., PROJ_ALIGN : PROJ_ALIGN + width]


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_forward_matches_reference(shape: Shape, dtype: torch.dtype) -> None:
    """Both outputs the scan reads, against the float64 frontier."""
    bsz, heads, seqlen = shape
    ops = _operands(shape, dtype, seed=1)
    want = _oracle(ops, shape)

    got = _forward(ops, shape)
    torch.cuda.synchronize()

    assert got.trans.shape == (bsz, heads, seqlen, 4)
    assert got.K.shape == (bsz, heads, seqlen, 2, 4)
    assert got.trans.dtype is torch.float32
    assert got.K.dtype is torch.float32
    assert all(t.is_contiguous() for t in got)

    tag = _tag(shape, dtype)
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    assert_max_rel(_matrices(got), _matrices(want), FWD_TOL, f"{tag}.K")


def test_lane_three_is_a_hard_zero() -> None:
    """Lane 3 is written by the kernel, not inherited from a zeroed allocation.

    The poison is freed immediately before the call so the caching allocator hands
    the same block back to the output; without it the check passes on any allocator
    that happens to return zeros.
    """
    bsz, heads, seqlen = ONE
    poison = torch.full((bsz, heads, seqlen, 2, 4), float("nan"), device="cuda")
    assert bool(poison.isnan().all())
    del poison

    got = _forward(_operands(ONE, seed=6), ONE)
    torch.cuda.synchronize()
    assert torch.count_nonzero(got.K[..., 3]) == 0


def test_kernels_read_a_projection_slice_without_repacking_it() -> None:
    """Bitwise equality against a narrowed operand, in both directions.

    The row pitch is taken from the operand at runtime, so the wide-pitch and the
    narrow-pitch call are the same executor and must produce identical bits. The
    backward reads ``params`` the same way, so it is checked in the same test
    rather than under a second fixture.
    """
    params, pbias = _operands(ONE, seed=3, strided=True)
    assert params.stride(-1) == 1 and not params.is_contiguous()
    compact = (_narrow(params), pbias)
    assert compact[0].stride(-2) != params.stride(-2)
    cots = _cotangents(ONE, seed=4)

    strided_out = _forward((params, pbias), ONE)
    compact_out = _forward(compact, ONE)
    strided_grads = _backward(cots, (params, pbias), ONE)
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


def test_float16_takes_its_own_kernel_path() -> None:
    """float16 keys a distinct executor, so it is traced and checked once.

    The only dtype-dependent code is the widen on load and the narrow on store,
    neither of which interacts with the shape, so one shape covers the axis.
    """
    ops = _operands(SHAPES[0], torch.float16, seed=12)
    cots = _cotangents(SHAPES[0], seed=13)
    want = _oracle(ops, SHAPES[0])
    want_grads = _oracle_grads(cots, ops, SHAPES[0])

    got = _forward(ops, SHAPES[0])
    got_grads = _backward(cots, ops, SHAPES[0])
    torch.cuda.synchronize()

    assert got_grads.dparams.dtype is torch.float16
    tag = _tag(SHAPES[0], torch.float16)
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    # float16 keeps 11 significand bits, so a stored gradient carries up to a
    # half-ulp, 2^-11 = 4.9e-4, of relative rounding; the bound is that, rounded up.
    assert_max_rel(got_grads.dparams, want_grads.dparams, 1e-3, f"{tag}.dparams")


def _sweep(vals: Tensor) -> tuple[Operands, Shape]:
    """One head per entry of ``vals``, in every parameter column, one token.

    The sweep goes in the bias and the band is zeroed, so the additive row is the
    swept value exactly. The oracle reads the same float32 values the kernel does.
    """
    count = int(vals.numel())
    zeros = torch.zeros(1, 1, count * PARAM_COLS, dtype=torch.float32, device="cuda")
    bias = vals[:, None].expand(count, PARAM_COLS).contiguous()
    return (_narrow(zeros), bias), (1, count, 1)


def test_extreme_raws_match_the_reference() -> None:
    """Parity and both invariants across the reachable float32 domain.

    I1 and I2 are produced by this kernel, so they are asserted on its output, and
    the domain that matters is the reachable one rather than a normal draw.
    Magnitudes stop at 1e8 because the squared radius overflows float32 near 1.8e19;
    past that the float32 map and the float64 oracle disagree by width, not by
    kernel, which is the next test.
    """
    vals = torch.tensor(EXTREME_RAWS, dtype=torch.float32, device="cuda")
    ops, shape = _sweep(vals)
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
    assert_max_rel(_matrices(got), _matrices(want), FWD_TOL, "cute-scanprep.extreme.K")
    # The log-scale column is bounded by LS_MAX_MAG at both ends now, so a bound
    # against the column maximum is a real claim rather than a vacuous one, and the
    # sigmoid saturates exactly at each end rather than approaching it.
    assert_max_rel(
        got.trans[..., 3], want.trans[..., 3], FWD_TOL, "cute-scanprep.extreme.ls"
    )
    assert float(got.trans[..., 3].min()) == -LS_MAX_MAG
    assert float(got.trans[..., 3].max()) == 0.0


def test_overflowing_radius_stays_finite() -> None:
    """An overflowing raw norm collapses through ``rsqrt(inf)`` without a NaN."""
    vals = torch.full((8,), 1e30, dtype=torch.float32, device="cuda")
    ops, shape = _sweep(vals)
    got = _forward(ops, shape)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(got.trans).all())
    assert bool(torch.isfinite(got.K).all())
    assert bool((got.trans[..., 3] <= 0.0).all())
    assert float(got.trans[..., :3].double().norm(dim=-1).max()) <= BALL_BOUND


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
def test_backward_matches_reference_autograd(shape: Shape, dtype: torch.dtype) -> None:
    """Both gradients against float64 autograd through the reference."""
    bsz, heads, seqlen = shape
    ops = _operands(shape, dtype, seed=1)
    cots = _cotangents(shape, seed=2)
    want = _oracle_grads(cots, ops, shape)

    got = _backward(cots, ops, shape)
    torch.cuda.synchronize()

    assert got.dparams.shape == (bsz, seqlen, heads * PARAM_COLS)
    assert got.dtransition_bias.shape == (heads, PARAM_COLS)
    assert got.dparams.dtype is dtype
    assert got.dtransition_bias.dtype is torch.float32
    assert all(t.is_contiguous() for t in got)

    tag = _tag(shape, dtype)
    assert_max_rel(got.dparams, want.dparams, BWD_TOL[dtype], f"{tag}.dparams")
    assert_max_rel(
        got.dtransition_bias, want.dtransition_bias, BIAS_TOL, f"{tag}.dtransition_bias"
    )


def test_backward_ignores_the_lane_three_cotangent() -> None:
    """Lane 3 of each tap is a constant zero, so its cotangent is the cotangent of
    nothing. A pullback that read it would leak into ``dparams``."""
    ops = _operands(ONE, seed=8)
    dtrans, dK = _cotangents(ONE, seed=9)
    loud = dK.clone()
    loud[..., 3] = 1e6

    quiet = _backward((dtrans, dK), ops, ONE)
    got = _backward((dtrans, loud), ops, ONE)
    torch.cuda.synchronize()

    assert torch.equal(got.dparams, quiet.dparams)
    assert torch.equal(got.dtransition_bias, quiet.dtransition_bias)
    assert float(quiet.dparams.abs().max()) > 0.0


@pytest.mark.parametrize("backward", BACKWARDS)
def test_backward_writes_dparams_into_a_supplied_band(backward: Backward) -> None:
    """A supplied destination is written in full and returned as itself.

    The mixer's backward allocates one ``dproj`` and hands each operator its band, so
    the gradient has to arrive at that pitch: a returned copy would be the second
    full write of every gradient byte the band destination exists to remove. Values
    are pinned against the default-allocation path, which is the same kernel reading
    the same operands at a different destination pitch, and the columns outside the
    band are pinned against the NaN they were filled with.
    """
    heads = ONE[1]
    ops = _operands(ONE, seed=14)
    cots = _cotangents(ONE, seed=15)
    want = backward(*cots, *ops, heads=heads, w_max=W_MAX)
    wide, dest = _band_dest(ONE)

    got = backward(*cots, *ops, heads=heads, w_max=W_MAX, dparams=dest)
    torch.cuda.synchronize()

    assert got.dparams is dest
    assert torch.equal(got.dparams, want.dparams)
    assert torch.equal(got.dtransition_bias, want.dtransition_bias)
    assert bool(wide[..., :PROJ_ALIGN].isnan().all())
    assert bool(wide[..., PROJ_ALIGN + heads * PARAM_COLS :].isnan().all())


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_backward_end_to_end(dtype: torch.dtype) -> None:
    """The fast forward, backpropagated through, against the float64 reference.

    The kernel forward feeds the kernel backward here, through the public operator
    and its registry dispatch, so a disagreement between the two cannot hide behind
    a surrogate forward.
    """
    heads = ONE[1]
    ops = _operands(ONE, dtype, seed=10)
    dtrans, dK = _cotangents(ONE, seed=11)
    fast = _leaves(ops, double=False)
    oracle = _leaves(ops, double=True)

    got = scanprep(*fast, heads=heads, w_max=W_MAX, backend="cute")
    ((got.trans * dtrans).sum() + (got.K * dK).sum()).backward()

    want = scanprep_ref(*oracle, heads=heads, w_max=W_MAX)
    ref = (want.trans * dtrans.double()).sum() + (want.K * dK.double()).sum()
    ref.backward()
    torch.cuda.synchronize()

    tag = f"{_tag(ONE, dtype)}.e2e"
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    assert_max_rel(_grad(fast[0]), _grad(oracle[0]), BWD_TOL[dtype], f"{tag}.dparams")
    assert_max_rel(
        _grad(fast[1]), _grad(oracle[1]), BIAS_TOL, f"{tag}.dtransition_bias"
    )


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

REJECT: Shape = (2, 3, 8)


def _unaligned(shape: Shape) -> Tensor:
    """A ``params``-shaped slice whose base address is one element in."""
    bsz, heads, seqlen = shape
    width = heads * PARAM_COLS
    row = torch.randn(bsz, seqlen, width + 2, dtype=torch.float32, device="cuda")
    return row[..., 1 : 1 + width]


def _stride_two(shape: Shape) -> Tensor:
    """A ``params``-shaped view whose trailing stride is two."""
    bsz, heads, seqlen = shape
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
            lambda p, pb: (p.double(), pb.double()),
            TypeError,
            r"kernel dtypes",
        ),
        (
            lambda p, pb: (p[..., :-PARAM_COLS], pb),
            ValueError,
            r"params must be \(B,T,",
        ),
        (
            lambda p, pb: (p.cpu(), pb.cpu()),
            ValueError,
            r"must be on a CUDA device",
        ),
        (
            lambda p, pb: (_unaligned(REJECT), pb),
            ValueError,
            r"must start and step on a multiple of",
        ),
        (
            lambda p, pb: (p[:, :0], pb),
            ValueError,
            r"at least one token",
        ),
        (
            lambda p, pb: (p, _gapped(pb)),
            ValueError,
            r"transition_bias must be contiguous",
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
    is one case, not that contract's table: the narrowed-width case pins that this
    path runs the shared shape checker, and the misaligned case pins that it runs the
    pitched-layout checker. The pitched table itself, including the pitch multiple
    and the overlapping-row rejection, is ``tests/test_guard.py``'s. The rest of the
    shape contract is covered against the reference in ``test_scanprep.py``.
    """
    with pytest.raises(error, match=match):
        _forward(mutate(*_operands(REJECT, seed=3)), REJECT)


@pytest.mark.parametrize("w_max", [0.0, -1.0, 3.14159265, 4.0, float("inf")])
def test_forward_rejects_illegal_bound(w_max: float) -> None:
    """I2 needs a bound strictly inside ``(0, pi)``."""
    params, pbias = _operands(REJECT, seed=3)
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_forward(params, pbias, heads=REJECT[1], w_max=w_max)


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (
            lambda dt, dk, p: (dt.bfloat16(), dk, p),
            ValueError,
            r"dtrans must be float32",
        ),
        (
            lambda dt, dk, p: (dt, dk.bfloat16(), p),
            ValueError,
            r"dK must be float32",
        ),
        (
            lambda dt, dk, p: (dt, dk, p.double()),
            TypeError,
            r"kernel dtypes",
        ),
        (
            lambda dt, dk, p: (dt[..., :3], dk, p),
            ValueError,
            r"dtrans must be",
        ),
        (
            lambda dt, dk, p: (_gapped(dt), dk, p),
            ValueError,
            r"dtrans must be contiguous",
        ),
        (
            lambda dt, dk, p: (dt, dk, _stride_two(REJECT)),
            ValueError,
            r"params must have unit stride",
        ),
        (
            lambda dt, dk, p: (dt, dk, p.cpu()),
            ValueError,
            r"must be on a CUDA device",
        ),
        (
            lambda dt, dk, p: (dt[:, :, :0], dk[:, :, :0], p[:, :0]),
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
    cots = _cotangents(REJECT, seed=4)
    dtrans, dK, params = mutate(*cots, ops[0])
    with pytest.raises(error, match=match):
        scanprep_backward(dtrans, dK, params, ops[1], heads=REJECT[1], w_max=W_MAX)


@pytest.mark.parametrize("backward", BACKWARDS)
def test_backward_rejects_a_misshaped_dparams_destination(backward: Backward) -> None:
    """Shape before layout, on the one operand a caller supplies rather than gets.

    The destination is one column short and starts one element into a sector, so a
    check that ran the alignment first would report the alignment and bury the shape.
    The rest of the destination's contract is the pitched-layout rule, whose table is
    ``tests/test_guard.py``'s.
    """
    ops = _operands(REJECT, seed=16)
    cots = _cotangents(REJECT, seed=17)
    dest = _band_dest(REJECT)[1][..., 1:]
    with pytest.raises(ValueError, match=r"dparams must be \(2, 8, 12\)"):
        backward(*cots, *ops, heads=REJECT[1], w_max=W_MAX, dparams=dest)


def test_backward_rejects_illegal_bound() -> None:
    """The bound scales the gradient, so the backward checks it too."""
    ops = _operands(REJECT, seed=3)
    cots = _cotangents(REJECT, seed=4)
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_backward(*cots, *ops, heads=REJECT[1], w_max=4.0)


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
