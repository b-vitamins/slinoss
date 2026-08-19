"""Fused mixer tail: the CuTe kernels against the float64 reference.

The authority is :func:`slinoss.ops.mixer.mixer_tail_ref` in float64, and the
gradient authority is float64 autograd through it. A hand-derived VJP shares its
derivation with the kernel, so a derivation error would pass silently; the
gradcheck here is what pins the authority itself.

Operands are built once in float32 and cast down, never built twice at two
dtypes: the generator consumes a different number of raw words per element at
each width, so the same seed at two dtypes is two different problems. The cast to
bfloat16 is exact on the way back up, so the kernel and the oracle see identical
values at every width.

``gate``, the output, and the output's cotangent are token-major, ``(B,T,H*P)``,
because in the block they are column bands of one projection. The parity sweep
builds them contiguous, which is the ``H*P == W`` case; one test builds a real
band of a wider buffer, which is the case the mixer hands over.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss._cute import assert_smem_fits, smem_bytes, smem_capacity
from slinoss.ops.mixer import mixer_tail, mixer_tail_ref
from slinoss.ops.mixer.cute import (
    ROWS,
    THREADS,
    mixer_tail_backward,
    mixer_tail_forward,
)
from slinoss.ops.mixer.cute.tail import (
    ROWS_PER_WARP,
    SLOTS,
    WARPS,
    param_tile,
)
from tests.conftest import assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

Operands = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

WARP = 32

EPS = 1e-5
"""The config default. Every parity call uses it, so the reciprocal square root
runs at the epsilon the model trains with."""

# (bsz, heads, seqlen, rows). The kernel is one warp per (b,h,t) row over B*T,
# with ceil(P/32) columns per lane, so the sweep is over those two axes: the
# smallest legal P with a masked tail, a P above the warp width, a P of four
# whole segments, and one P that is neither. B = 1, H = 1, a single token, a
# token count that is not a multiple of ROWS, and a token count that is appear
# across the four.
SHAPES = [
    pytest.param(1, 1, 1, 16, id="single-token-min-p"),
    pytest.param(2, 3, 40, 64, id="ragged-tiles"),
    pytest.param(1, 2, 129, 48, id="masked-tail-two-segments"),
    pytest.param(2, 1, 64, 128, id="exact-tiles-four-segments"),
]

DTYPES = [torch.float32, torch.bfloat16]

# Arithmetic is float32 on exactly representable inputs at both widths, so the
# bound is rounding and nothing else. float32: the row sum of squares, the
# reciprocal square root, and the sigmoid each round a few ulp. bfloat16: the
# store is the last operation and everything before it is float32, so the total
# is one bfloat16 rounding plus float32 noise; bfloat16 carries 8 significand
# bits, which puts unit roundoff at 2^-8 = 3.9e-3.
FWD_TOL = {torch.float32: 1e-6, torch.bfloat16: 4e-3}

# Gradients are stored at the width of the tensor they belong to, so the same two
# arguments apply. The float32 bound is wider than the forward's: the norm
# coupling is a difference of two terms of similar magnitude, and the parameter
# gradients reduce over B*T before they are stored.
BWD_TOL = {torch.float32: 3e-6, torch.bfloat16: 4e-3}


def _tag(bsz: int, heads: int, seqlen: int, rows: int, dtype: torch.dtype) -> str:
    width = str(dtype).removeprefix("torch.")
    return f"cute-mixer-tail[{bsz}x{heads}x{seqlen}x{rows}/{width}]"


def _operands(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    dtype: torch.dtype = torch.float32,
    *,
    param_dtype: torch.dtype | None = None,
    seed: int = 0,
) -> Operands:
    """One operand set on CUDA. Built in float32, then cast.

    ``gate`` is widened so both branches of the sigmoid are exercised, and
    ``weight`` sits around one, which is where a trained norm scale sits.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda")

    param = dtype if param_dtype is None else param_dtype
    y = rnd(bsz, heads, seqlen, rows)
    u = rnd(bsz, heads, seqlen, rows)
    gate = rnd(bsz, seqlen, heads * rows) * 3.0
    d_skip = rnd(heads, rows) * 0.5
    weight = 1.0 + 0.25 * rnd(heads, rows)
    return (
        y.to(dtype),
        u.to(dtype),
        gate.to(dtype),
        d_skip.to(param),
        weight.to(param),
    )


def _cotangent(
    bsz: int, heads: int, seqlen: int, rows: int, dtype: torch.dtype, *, seed: int
) -> Tensor:
    """Cotangent of the output, token-major, in the dtype the output carries."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        bsz, seqlen, heads * rows, generator=gen, dtype=torch.float32, device="cuda"
    ).to(dtype)


def _band(source: Tensor, pad: int) -> Tensor:
    """``source`` copied into a wider buffer and handed back as a column band.

    ``pad`` columns on each side, so the band is neither at the start of a row nor
    at the end of one and the pitch exceeds the width in both directions.
    """
    wide = torch.empty(
        *source.shape[:-1],
        int(source.shape[-1]) + 2 * pad,
        dtype=source.dtype,
        device=source.device,
    )
    band = wide[..., pad : pad + int(source.shape[-1])]
    band.copy_(source)
    return band


def _leaves(ops: Operands, *, double: bool) -> Operands:
    """The same operands as differentiable leaves. The upcast is exact."""
    out = tuple(
        (t.double() if double else t).detach().clone().requires_grad_() for t in ops
    )
    return (out[0], out[1], out[2], out[3], out[4])


def _grad(tensor: Tensor) -> Tensor:
    assert tensor.grad is not None
    return tensor.grad


def _oracle_grads(ops: Operands, dout: Tensor) -> Operands:
    """Gradients from float64 autograd through the reference."""
    leaves = _leaves(ops, double=True)
    out = mixer_tail_ref(*leaves, eps=EPS)
    grads = torch.autograd.grad(out, leaves, dout.double())
    return (grads[0], grads[1], grads[2], grads[3], grads[4])


NAMES = ("dy", "du", "dgate", "dd_skip", "dweight")


# ---------------------------------------------------------------------------
# The authority
# ---------------------------------------------------------------------------


def test_reference_gradcheck_in_float64() -> None:
    """Every gradient the kernel is measured against, checked numerically."""
    leaves = _leaves(_operands(1, 2, 3, 16, seed=5), double=True)

    def fn(
        y: Tensor, u: Tensor, gate: Tensor, d_skip: Tensor, weight: Tensor
    ) -> Tensor:
        return mixer_tail_ref(y, u, gate, d_skip, weight, eps=EPS)

    assert torch.autograd.gradcheck(fn, leaves)


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("bsz", "heads", "seqlen", "rows"), SHAPES)
def test_forward_matches_reference(
    bsz: int, heads: int, seqlen: int, rows: int, dtype: torch.dtype
) -> None:
    """The fused output matches the float64 tail, at both operand widths."""
    ops = _operands(bsz, heads, seqlen, rows, dtype, seed=1)
    want = mixer_tail_ref(*(t.double() for t in ops), eps=EPS)

    got = mixer_tail_forward(*ops, eps=EPS)
    torch.cuda.synchronize()

    assert got.shape == (bsz, seqlen, heads * rows)
    assert got.dtype is dtype
    assert got.is_contiguous()
    assert_max_rel(
        got, want, FWD_TOL[dtype], _tag(bsz, heads, seqlen, rows, dtype) + ".out"
    )


def test_float32_parameters_with_low_precision_operands() -> None:
    """Operand width and parameter width are independent.

    A kernel that read the parameters through the operand's element type would
    reinterpret float32 words as bfloat16 pairs, and a wrapper that demanded one
    dtype for all five would force the caller into a cast.
    """
    ops = _operands(2, 3, 40, 64, torch.bfloat16, param_dtype=torch.float32, seed=12)
    dout = _cotangent(2, 3, 40, 64, torch.bfloat16, seed=13)
    want = mixer_tail_ref(*(t.double() for t in ops), eps=EPS)
    want_grads = _oracle_grads(ops, dout)

    got = mixer_tail_forward(*ops, eps=EPS)
    grads = mixer_tail_backward(dout, *ops, eps=EPS)
    torch.cuda.synchronize()

    assert got.dtype is torch.bfloat16
    assert grads.dd_skip.dtype is torch.float32
    assert grads.dweight.dtype is torch.float32
    tag = "cute-mixer-tail[mixed-width]"
    assert_max_rel(got, want, FWD_TOL[torch.bfloat16], f"{tag}.out")
    for got_grad, want_grad, name in zip(grads, want_grads, NAMES):
        assert_max_rel(got_grad, want_grad, BWD_TOL[got_grad.dtype], f"{tag}.{name}")


@pytest.mark.parametrize(
    ("kill", "reason"),
    [
        (lambda ops: (ops[0] * 0.0, ops[1] * 0.0, *ops[2:]), "zero-value"),
        (lambda ops: (*ops[:2], torch.full_like(ops[2], -1e4), *ops[3:]), "shut-gate"),
    ],
)
def test_dead_row_stays_finite(
    kill: Callable[[Operands], Operands], reason: str
) -> None:
    """A row of exact zeros divides by ``eps`` alone and stays finite.

    ``zero-value`` reaches the epsilon: without it the reciprocal square root of
    zero is an infinity that the following multiply turns into a NaN.
    ``shut-gate`` reaches the sigmoid's negative branch, where ``exp(-gate)``
    overflows to infinity and the naive quotient is ``inf/inf``; the value is
    zero, so the whole row is again carried by the epsilon.
    """
    ops = kill(_operands(1, 2, 8, 64, seed=14))
    dout = _cotangent(1, 2, 8, 64, torch.float32, seed=15)

    got = mixer_tail_forward(*ops, eps=EPS)
    grads = mixer_tail_backward(dout, *ops, eps=EPS)
    torch.cuda.synchronize()

    assert torch.equal(got, torch.zeros_like(got)), reason
    for grad, name in zip(grads, NAMES):
        assert bool(torch.isfinite(grad).all()), f"{reason}: {name}"


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("bsz", "heads", "seqlen", "rows"), SHAPES)
def test_backward_matches_reference_autograd(
    bsz: int, heads: int, seqlen: int, rows: int, dtype: torch.dtype
) -> None:
    """All five gradients match float64 autograd through the reference.

    The two parameter gradients reduce over ``(B,T)`` inside the kernel's
    epilogue, so this is also what pins the cross-tile reduction: the sweep
    covers one tile, three tiles with a partial last, and four whole tiles.
    """
    ops = _operands(bsz, heads, seqlen, rows, dtype, seed=1)
    dout = _cotangent(bsz, heads, seqlen, rows, dtype, seed=2)
    want = _oracle_grads(ops, dout)

    got = mixer_tail_backward(dout, *ops, eps=EPS)
    torch.cuda.synchronize()

    tag = _tag(bsz, heads, seqlen, rows, dtype)
    for got_grad, want_grad, operand, name in zip(got, want, ops, NAMES):
        assert got_grad.shape == operand.shape
        assert got_grad.dtype is operand.dtype
        assert got_grad.is_contiguous()
        assert_max_rel(got_grad, want_grad, BWD_TOL[dtype], f"{tag}.{name}")


def test_every_output_element_is_written() -> None:
    """No output element is inherited from the allocation it landed in.

    The backward allocates four buffers without filling any of them, and the
    parameter partials are summed across tiles afterwards, so one unwritten word
    poisons a whole parameter row. The shape leaves seven of the eight warps with
    no row and the last segment half masked, which is where an unwritten word
    would come from. The NaN blocks are freed immediately before the call so the
    caching allocator hands them straight back; without that the check passes on
    any allocator that happens to return zeros.
    """
    poison = [torch.full((2, 1, 1, 16), float("nan"), device="cuda") for _ in range(6)]
    assert all(bool(block.isnan().all()) for block in poison)
    del poison

    ops = _operands(1, 1, 1, 16, seed=16)
    grads = mixer_tail_backward(
        _cotangent(1, 1, 1, 16, torch.float32, seed=17), *ops, eps=EPS
    )
    torch.cuda.synchronize()
    for grad, name in zip(grads, NAMES):
        assert bool(torch.isfinite(grad).all()), name


@pytest.mark.parametrize("dtype", DTYPES)
def test_pitched_band_matches_the_contiguous_call(dtype: torch.dtype) -> None:
    """``gate`` and ``dout`` as column bands of a wider buffer.

    In the block both are slices of one projection output, so a band is the layout
    the tail actually runs on. The result must be bit-identical to the contiguous
    call: the pitch changes a load address and nothing else, so a tolerance here
    would admit a kernel that had quietly repacked.
    """
    ops = _operands(2, 3, 40, 64, dtype, seed=19)
    dout = _cotangent(2, 3, 40, 64, dtype, seed=20)
    # Eight columns of padding: 32 B at float32 and 16 B at bfloat16, so the band
    # starts and steps on the 16 B boundary at both widths.
    banded: Operands = (ops[0], ops[1], _band(ops[2], 8), ops[3], ops[4])

    got = mixer_tail_forward(*banded, eps=EPS)
    want = mixer_tail_forward(*ops, eps=EPS)
    got_grads = mixer_tail_backward(_band(dout, 8), *banded, eps=EPS)
    want_grads = mixer_tail_backward(dout, *ops, eps=EPS)
    torch.cuda.synchronize()

    assert torch.equal(got, want)
    for got_grad, want_grad, name in zip(got_grads, want_grads, NAMES, strict=True):
        assert torch.equal(got_grad, want_grad), name


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_and_backward_end_to_end(dtype: torch.dtype) -> None:
    """The fast forward, backpropagated through, against the float64 reference.

    The kernel forward feeds the kernel backward here, so a disagreement between
    the two cannot hide behind a surrogate forward.
    """
    ops = _operands(2, 3, 40, 64, dtype, seed=10)
    dout = _cotangent(2, 3, 40, 64, dtype, seed=11)
    fast = _leaves(ops, double=False)
    oracle = _leaves(ops, double=True)

    got = mixer_tail(*fast, eps=EPS, backend="cute")
    (got * dout).sum().backward()

    want = mixer_tail_ref(*oracle, eps=EPS)
    (want * dout.double()).sum().backward()
    torch.cuda.synchronize()

    tag = f"{_tag(2, 3, 40, 64, dtype)}.e2e"
    assert_max_rel(got, want, FWD_TOL[dtype], f"{tag}.out")
    for leaf, ref, name in zip(fast, oracle, NAMES):
        assert_max_rel(_grad(leaf), _grad(ref), BWD_TOL[dtype], f"{tag}.{name}")


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

FwdMutator = Callable[[Operands], Operands]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda o: (o[0][0], *o[1:]), ValueError, r"y must be \(B,H,T,P\)"),
        (lambda o: (o[0], o[1][..., :-16], *o[2:]), ValueError, r"u must be"),
        (lambda o: (*o[:2], o[2][:1], *o[3:]), ValueError, r"gate must be"),
        (lambda o: (*o[:3], o[3][..., :-16], o[4]), ValueError, r"d_skip must be"),
        (lambda o: (*o[:4], o[4][:1]), ValueError, r"weight must be"),
        (
            lambda o: (o[0][:0], o[1][:0], o[2][:0], *o[3:]),
            ValueError,
            r"at least one element",
        ),
        (
            lambda o: (o[0].double(), o[1].double(), o[2].double(), o[3], o[4]),
            TypeError,
            r"kernel dtypes",
        ),
        (lambda o: (o[0].bfloat16(), *o[1:]), TypeError, r"u is"),
        (lambda o: (*o[:3], o[3].bfloat16(), o[4]), TypeError, r"weight is"),
        (
            lambda o: (o[0].cpu(), o[1].cpu(), o[2].cpu(), o[3].cpu(), o[4].cpu()),
            ValueError,
            r"y must be on a CUDA device",
        ),
        (
            lambda o: (o[0].transpose(0, 1), *o[1:]),
            ValueError,
            r"y must be contiguous",
        ),
        (
            lambda o: (*o[:2], _band(o[2], 1), *o[3:]),
            ValueError,
            r"gate must start and step on a multiple of",
        ),
    ],
)
def test_forward_rejects(
    mutate: FwdMutator, error: type[Exception], match: str
) -> None:
    """Each guard on the forward's operands, triggered.

    ``bsz == heads`` so that a transpose of the leading pair keeps the shape and
    loses only the layout. The last row is the one guard the head-major operands
    do not share: ``gate`` is allowed a pitch but not an unaligned one, which is a
    producer that handed out a band without padding the column offset.
    """
    with pytest.raises(error, match=match):
        mixer_tail_forward(*mutate(_operands(2, 2, 8, 64, seed=3)), eps=EPS)


BwdMutator = Callable[[Tensor, Operands], tuple[Tensor, Operands]]


@pytest.mark.parametrize(
    ("mutate", "error", "match"),
    [
        (lambda d, o: (d[..., :-16], o), ValueError, r"dout must be"),
        (lambda d, o: (d.bfloat16(), o), TypeError, r"dout is"),
        (
            lambda d, o: (_band(d, 1), o),
            ValueError,
            r"dout must start and step on a multiple of",
        ),
        (
            lambda d, o: (d, (o[0], o[1][..., :-16], *o[2:])),
            ValueError,
            r"u must be",
        ),
    ],
)
def test_backward_rejects(
    mutate: BwdMutator, error: type[Exception], match: str
) -> None:
    """Each guard the backward adds, plus proof it reruns the shared ones."""
    dout, ops = mutate(
        _cotangent(2, 2, 8, 64, torch.float32, seed=4), _operands(2, 2, 8, 64, seed=3)
    )
    with pytest.raises(error, match=match):
        mixer_tail_backward(dout, *ops, eps=EPS)


def test_rejects_non_positive_eps() -> None:
    """The epsilon is the only thing between a zero row and a division by zero,
    so both directions refuse a non-positive one."""
    ops = _operands(2, 2, 8, 64, seed=3)
    dout = _cotangent(2, 2, 8, 64, torch.float32, seed=4)
    with pytest.raises(ValueError, match=r"eps must be positive"):
        mixer_tail_forward(*ops, eps=0.0)
    with pytest.raises(ValueError, match=r"eps must be positive"):
        mixer_tail_backward(dout, *ops, eps=0.0)


def test_oversized_rows_are_refused() -> None:
    """``P`` is bounded by the partial tile against the queried capacity.

    The bound is read from the device rather than written down, so this fails on
    any host where the layout and the capacity disagree.
    """
    segments = smem_capacity() // (SLOTS * WARPS * WARP * 4) + 1
    rows = segments * WARP
    ops = _operands(1, 1, 1, rows, seed=18)
    with pytest.raises(ValueError, match=r"shared memory"):
        mixer_tail_forward(*ops, eps=EPS)


def test_partial_tile_budget_matches_the_layout() -> None:
    """The budget comes from the layout, with no guard constant.

    Both accesses to the tile are unit-stride across the lanes of a warp, so the
    trailing extent is what makes it conflict-free; a pitch added here would be
    the tell that something else is indexing it.
    """
    for segments in (1, 4):
        tile = param_tile(segments)
        assert tile.shape == (SLOTS, WARPS, WARP * segments)
        assert tile.stride[-1] == 1
        nbytes = smem_bytes([(tile, 4)])
        assert nbytes == SLOTS * WARPS * WARP * segments * 4
        assert assert_smem_fits(f"mixer_tail_bwd[{segments}]", nbytes) == nbytes
        assert nbytes <= smem_capacity()


def test_block_geometry_is_whole_warps() -> None:
    """The row index is ``tile*ROWS + step*WARPS + warp``.

    That covers ``[tile*ROWS, (tile+1)*ROWS)`` exactly once only under this
    factorization; a constant drifting out of step would skip rows or count them
    twice.
    """
    assert THREADS == WARPS * WARP
    assert ROWS == WARPS * ROWS_PER_WARP
