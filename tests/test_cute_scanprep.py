"""Bounded parameter maps: the CuTe kernels against the float64 reference.

The authority is :func:`slinoss.ops.scanprep.scanprep_ref` in float64, and the
gradient authority is float64 autograd through it. A hand-derived VJP shares its
derivation with the kernel, so a derivation error would pass silently; the
gradcheck here is what pins the authority itself.

Operands are built once in float32 and cast down, never built twice at two
dtypes: the generator consumes a different number of raw words per element at
each width, so the same seed at two dtypes is two different problems. The cast to
bfloat16 is exact on the way back up, so the kernel and the oracle see identical
values at every width.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss.ops.scanprep import scanprep_ref
from slinoss.ops.scanprep.cute import (
    THREADS,
    scanprep,
    scanprep_backward,
    scanprep_forward,
)
from tests.conftest import W_MAX, assert_max_rel, max_err

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

Triple = tuple[Tensor, Tensor, Tensor]

# (bsz, heads, seqlen). The kernel is one thread per token over B*H*T, so the
# sweep is over that product: one exact block, a tail-only block, a partial tail
# above several full blocks, many full blocks, and a single token. B = 1 and
# H = 1 appear, and so do two sequence lengths that are not multiples of the
# block width.
SHAPES = [
    pytest.param(1, 1, THREADS, id="one-exact-block"),
    pytest.param(1, 1, 40, id="tail-only"),
    pytest.param(2, 3, 100, id="ragged-tail"),
    pytest.param(4, 2, 512, id="many-blocks"),
    pytest.param(2, 1, 2000, id="long-ragged"),
    pytest.param(1, 1, 1, id="single-token"),
]

DTYPES = [torch.float32, torch.bfloat16]

# The forward is float32 arithmetic over exactly representable inputs at both
# widths, so the bound is float32 rounding of the map and nothing else.
FWD_TOL = 1e-6

# The gradients are stored at the input width. bfloat16 keeps 8 mantissa bits, so
# a stored gradient carries up to 2^-9 = 2.0e-3 of relative rounding; the bound
# is that with a factor of two of margin. float32 storage leaves only the
# arithmetic.
BWD_TOL = {torch.float32: 1e-6, torch.bfloat16: 4e-3}

# The exact map lands in the closed ball of radius w_max; the computed vector is
# that value rounded twice, so its norm can sit a few ulp outside. Same argument
# and same constant as the reference's own bound.
BALL_BOUND = W_MAX * (1.0 + 3.0 * torch.finfo(torch.float32).eps)

EXTREME_RAWS = (-1e8, -1e4, -20.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 20.0, 1e4, 1e8)


def _tag(bsz: int, heads: int, seqlen: int, dtype: torch.dtype) -> str:
    width = str(dtype).removeprefix("torch.")
    return f"cute-scanprep[{bsz}x{heads}x{seqlen}/{width}]"


def _raws(
    bsz: int,
    heads: int,
    seqlen: int,
    dtype: torch.dtype = torch.float32,
    *,
    seed: int = 0,
    w_scale: float = 1.5,
) -> Triple:
    """One operand triple on CUDA. Built in float32, then cast."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda")

    w_raw = rnd(bsz, heads, seqlen, 3) * w_scale
    ls_raw = rnd(bsz, heads, seqlen)
    tap_raw = rnd(bsz, heads, seqlen, 2, 3)
    return (w_raw.to(dtype), ls_raw.to(dtype), tap_raw.to(dtype))


def _cotangents(
    bsz: int, heads: int, seqlen: int, *, seed: int
) -> tuple[Tensor, Tensor]:
    """Cotangents of the packed outputs. float32, because I4 pins both outputs."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda")

    return rnd(bsz, heads, seqlen, 4), rnd(bsz, heads, seqlen, 2, 4)


def _leaves(raws: Triple, *, double: bool) -> Triple:
    """The same operands as differentiable leaves. The upcast is exact."""
    out = tuple(
        (t.double() if double else t).detach().clone().requires_grad_() for t in raws
    )
    return (out[0], out[1], out[2])


def _grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def _oracle_grads(raws: Triple, dtrans: Tensor, dK: Tensor) -> Triple:
    """Gradients from float64 autograd through the reference."""
    leaves = _leaves(raws, double=True)
    out = scanprep_ref(*leaves, w_max=W_MAX)
    grads = torch.autograd.grad(
        (out.trans, out.K), leaves, (dtrans.double(), dK.double())
    )
    return (grads[0], grads[1], grads[2])


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------


def test_reference_gradcheck_in_float64() -> None:
    """Every gradient the kernel is measured against, checked numerically."""
    leaves = _leaves(_raws(1, 1, 3, seed=5), double=True)

    def fn(w_raw: Tensor, ls_raw: Tensor, tap_raw: Tensor) -> tuple[Tensor, Tensor]:
        out = scanprep_ref(w_raw, ls_raw, tap_raw, w_max=W_MAX)
        return out.trans, out.K

    assert torch.autograd.gradcheck(fn, leaves)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("bsz", "heads", "seqlen"), SHAPES)
def test_forward_matches_reference(
    bsz: int, heads: int, seqlen: int, dtype: torch.dtype
) -> None:
    """Both packed outputs match the float64 maps, at both input widths."""
    raws = _raws(bsz, heads, seqlen, dtype, seed=1)
    want = scanprep_ref(*(t.double() for t in raws), w_max=W_MAX)

    got = scanprep_forward(*raws, w_max=W_MAX)
    torch.cuda.synchronize()

    assert got.trans.shape == (bsz, heads, seqlen, 4)
    assert got.K.shape == (bsz, heads, seqlen, 2, 4)
    assert got.trans.dtype is torch.float32
    assert got.K.dtype is torch.float32
    assert got.trans.is_contiguous()
    assert got.K.is_contiguous()
    assert_max_rel(
        got.trans, want.trans, FWD_TOL, f"{_tag(bsz, heads, seqlen, dtype)}.trans"
    )
    # The tap map is the identity and the widening is exact, so equality is the
    # honest bound here.
    assert torch.equal(got.K[..., :3], raws[2].float())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("bsz", "heads", "seqlen"), SHAPES)
def test_forward_holds_the_invariants(
    bsz: int, heads: int, seqlen: int, dtype: torch.dtype
) -> None:
    """I1 and I2 are produced by the kernel, so they are asserted on its output."""
    got = scanprep_forward(*_raws(bsz, heads, seqlen, dtype, seed=4), w_max=W_MAX)
    torch.cuda.synchronize()
    assert bool((got.trans[..., 3] <= 0.0).all())
    assert bool((got.trans[..., :3].double().norm(dim=-1) <= BALL_BOUND).all())
    assert bool(torch.isfinite(got.trans).all())
    assert bool(torch.isfinite(got.K).all())


def test_lane_three_is_a_hard_zero() -> None:
    """Lane 3 is written by the kernel, not inherited from a zeroed allocation.

    The poison is freed immediately before the call so the caching allocator
    hands the same block back to the output; without it the check passes on any
    allocator that happens to return zeros.
    """
    shape = (2, 3, 300)
    poison = torch.full((*shape, 2, 4), float("nan"), device="cuda")
    assert bool(poison.isnan().all())
    del poison

    got = scanprep_forward(*_raws(*shape, seed=6), w_max=W_MAX)
    torch.cuda.synchronize()
    assert torch.count_nonzero(got.K[..., 3]) == 0


def test_outputs_stay_float32_under_autocast() -> None:
    """I4 pins both outputs whatever the ambient autocast dtype."""
    raws = _raws(1, 2, 64, seed=7)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        got = scanprep_forward(*raws, w_max=W_MAX)
    assert got.trans.dtype is torch.float32
    assert got.K.dtype is torch.float32


def _extreme_raws() -> Triple:
    """One token per entry of :data:`EXTREME_RAWS`, in every operand."""
    vals = torch.tensor(EXTREME_RAWS, dtype=torch.float32, device="cuda")
    count = int(vals.numel())
    return (
        vals[:, None].expand(count, 3).reshape(1, 1, count, 3).contiguous(),
        vals.reshape(1, 1, count).contiguous(),
        vals[:, None, None].expand(count, 2, 3).reshape(1, 1, count, 2, 3).contiguous(),
    )


def test_extreme_raws_match_the_reference() -> None:
    """Parity across the reachable float32 domain.

    Magnitudes stop at 1e8 because ``|raw|^2`` overflows float32 near 1.8e19; past
    that the float32 map and the float64 oracle disagree by width, not by kernel,
    which is the next test.
    """
    raws = _extreme_raws()
    want = scanprep_ref(*(t.double() for t in raws), w_max=W_MAX)

    got = scanprep_forward(*raws, w_max=W_MAX)
    torch.cuda.synchronize()

    assert bool((got.trans[..., 3] <= 0.0).all())
    assert bool((got.trans[..., :3].double().norm(dim=-1) <= BALL_BOUND).all())
    assert_max_rel(
        got.trans[..., :3], want.trans[..., :3], FWD_TOL, "cute-scanprep.extreme.w"
    )
    # The log-scale column spans 1e8 down to zero, so a bound relative to the
    # column maximum is vacuous. The absolute bound is float32 rounding of
    # log1p near its largest reachable argument.
    assert max_err(got.trans[..., 3], want.trans[..., 3]) < 1e-6


def test_overflowing_raw_norm_stays_finite() -> None:
    """``|raw|^2`` overflows float32 near 1.8e19, and ``rsqrt(inf)`` is zero.

    The map collapses to the centre of the ball rather than producing a NaN,
    which is all I2 claims. The float64 oracle does not overflow, so this is a
    property check and not a parity check.
    """
    huge = torch.full((1, 1, 8, 3), 1e30, dtype=torch.float32, device="cuda")
    got = scanprep_forward(
        huge,
        huge[..., 0].contiguous(),
        huge[..., None, :].expand(1, 1, 8, 2, 3).contiguous(),
        w_max=W_MAX,
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(got.trans).all())
    assert bool((got.trans[..., 3] <= 0.0).all())
    assert float(got.trans[..., :3].double().norm(dim=-1).max()) <= BALL_BOUND


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("bsz", "heads", "seqlen"), SHAPES)
def test_backward_matches_reference_autograd(
    bsz: int, heads: int, seqlen: int, dtype: torch.dtype
) -> None:
    """All three gradients match float64 autograd through the reference."""
    raws = _raws(bsz, heads, seqlen, dtype, seed=1)
    dtrans, dK = _cotangents(bsz, heads, seqlen, seed=2)
    want_w, want_ls, want_tap = _oracle_grads(raws, dtrans, dK)

    got = scanprep_backward(dtrans, dK, raws[0], raws[1], w_max=W_MAX)
    torch.cuda.synchronize()

    assert got.dw_raw.shape == raws[0].shape
    assert got.dls_raw.shape == raws[1].shape
    assert got.dtap_raw.shape == raws[2].shape
    assert got.dw_raw.dtype is dtype
    assert got.dls_raw.dtype is dtype
    assert got.dtap_raw.dtype is dtype
    tag = _tag(bsz, heads, seqlen, dtype)
    bound = BWD_TOL[dtype]
    assert_max_rel(got.dw_raw, want_w, bound, f"{tag}.dw_raw")
    assert_max_rel(got.dls_raw, want_ls, bound, f"{tag}.dls_raw")
    assert_max_rel(got.dtap_raw, want_tap, bound, f"{tag}.dtap_raw")


def test_backward_ignores_the_lane_three_cotangent() -> None:
    """Lane 3 of ``K`` is a constant, so its cotangent reaches no input."""
    raws = _raws(2, 2, 96, seed=8)
    dtrans, dK = _cotangents(2, 2, 96, seed=9)
    quiet = dK.clone()
    quiet[..., 3] = 0.0

    first = scanprep_backward(dtrans, dK, raws[0], raws[1], w_max=W_MAX)
    second = scanprep_backward(dtrans, quiet, raws[0], raws[1], w_max=W_MAX)
    torch.cuda.synchronize()

    assert torch.equal(first.dw_raw, second.dw_raw)
    assert torch.equal(first.dls_raw, second.dls_raw)
    assert torch.equal(first.dtap_raw, second.dtap_raw)


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_backward_end_to_end(dtype: torch.dtype) -> None:
    """The fast forward, backpropagated through, against the float64 reference.

    The kernel forward feeds the kernel backward here, so a disagreement between
    the two cannot hide behind a surrogate forward.
    """
    raws = _raws(2, 3, 300, dtype, seed=10)
    dtrans, dK = _cotangents(2, 3, 300, seed=11)
    fast = _leaves(raws, double=False)
    oracle = _leaves(raws, double=True)

    got = scanprep(*fast, w_max=W_MAX)
    (got.trans * dtrans).sum().add((got.K * dK).sum()).backward()

    want = scanprep_ref(*oracle, w_max=W_MAX)
    (want.trans * dtrans.double()).sum().add((want.K * dK.double()).sum()).backward()
    torch.cuda.synchronize()

    tag = f"{_tag(2, 3, 300, dtype)}.e2e"
    assert_max_rel(got.trans, want.trans, FWD_TOL, f"{tag}.trans")
    bound = BWD_TOL[dtype]
    for leaf, ref, name in zip(fast, oracle, ("dw_raw", "dls_raw", "dtap_raw")):
        assert_max_rel(_grad(leaf), _grad(ref), bound, f"{tag}.{name}")


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

FwdMutator = Callable[[Tensor, Tensor, Tensor], Triple]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda w, ls, tap: (w[..., :2], ls, tap), ValueError, r"w_raw must be"),
        (lambda w, ls, tap: (w[0], ls, tap), ValueError, r"w_raw must be"),
        (
            lambda w, ls, tap: (w[:0], ls[:0], tap[:0]),
            ValueError,
            r"at least one token",
        ),
        (lambda w, ls, tap: (w, ls[..., :-1], tap), ValueError, r"ls_raw must be"),
        (lambda w, ls, tap: (w, ls, tap[..., :2]), ValueError, r"tap_raw must be"),
        (lambda w, ls, tap: (w, ls, tap[..., :1, :]), ValueError, r"tap_raw must be"),
        (
            lambda w, ls, tap: (w.double(), ls.double(), tap.double()),
            TypeError,
            r"kernel dtypes",
        ),
        (
            lambda w, ls, tap: (w.bfloat16(), ls, tap),
            TypeError,
            r"one dtype per call",
        ),
        (
            lambda w, ls, tap: (w.cpu(), ls.cpu(), tap.cpu()),
            ValueError,
            r"must be on a CUDA device",
        ),
        (
            lambda w, ls, tap: (w.transpose(0, 1), ls, tap),
            ValueError,
            r"w_raw must be contiguous",
        ),
    ],
)
def test_forward_rejects(
    mutate: FwdMutator, error: type[Exception], match: str
) -> None:
    """Each guard on the forward's operands, triggered.

    ``bsz == heads`` so that a transpose of the leading pair keeps the shape and
    loses only the layout.
    """
    with pytest.raises(error, match=match):
        scanprep_forward(*mutate(*_raws(2, 2, 8, seed=3)), w_max=W_MAX)


@pytest.mark.parametrize("w_max", [0.0, -1.0, 4.0, float("inf")])
def test_forward_rejects_illegal_bound(w_max: float) -> None:
    """I2 needs a bound strictly inside ``(0, pi)``."""
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_forward(*_raws(2, 2, 8, seed=3), w_max=w_max)


BwdMutator = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda dt, dk: (dt.bfloat16(), dk), r"dtrans must be float32"),
        (lambda dt, dk: (dt, dk.bfloat16()), r"dK must be float32"),
        (lambda dt, dk: (dt[..., :3], dk), r"dtrans must be"),
        (lambda dt, dk: (dt, dk[..., :1, :]), r"dK must be"),
        (lambda dt, dk: (dt.transpose(0, 1), dk), r"dtrans must be contiguous"),
    ],
)
def test_backward_rejects(mutate: BwdMutator, match: str) -> None:
    """Each guard on the backward's cotangents, triggered."""
    w_raw, ls_raw, _ = _raws(2, 2, 8, seed=3)
    with pytest.raises(ValueError, match=match):
        scanprep_backward(
            *mutate(*_cotangents(2, 2, 8, seed=4)), w_raw, ls_raw, w_max=W_MAX
        )


def test_backward_rejects_mismatched_raws() -> None:
    """The raw operands are validated on the way in, as on the forward."""
    w_raw, ls_raw, _ = _raws(2, 2, 8, seed=3)
    dtrans, dK = _cotangents(2, 2, 8, seed=4)
    with pytest.raises(ValueError, match=r"ls_raw must be"):
        scanprep_backward(dtrans, dK, w_raw, ls_raw[..., :-1], w_max=W_MAX)


def test_backward_rejects_illegal_bound() -> None:
    """The bound scales the gradient, so the backward checks it too."""
    w_raw, ls_raw, _ = _raws(2, 2, 8, seed=3)
    dtrans, dK = _cotangents(2, 2, 8, seed=4)
    with pytest.raises(ValueError, match=r"w_max must lie in \(0, pi\)"):
        scanprep_backward(dtrans, dK, w_raw, ls_raw, w_max=4.0)


def test_block_width_is_a_warp_multiple() -> None:
    """A tail predicate lets ``T`` be arbitrary; the block width stays whole warps."""
    assert THREADS % 32 == 0
