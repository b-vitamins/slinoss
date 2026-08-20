"""Causal depthwise conv1d: the CUDA kernels against the float64 reference.

The kernel is never the specification. Every figure here is compared against the
float64 oracle, or against the reference at the same storage dtype, and the
gradients are compared against float64 autograd through the reference rather than
against a written-out pullback.

The oracle and the operand builder are imported rather than restated. A second
copy of either would drift, and a drift in the oracle is a correctness bug that
passes silently.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from slinoss import _C
from slinoss._guard import ALIGN_BYTES

if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if not _C.is_available():
    pytest.skip(
        f"{_C.EXTENSION} is not built; run {_C.BUILD_COMMAND}",
        allow_module_level=True,
    )

from slinoss.ops.conv import (
    ConvStep,
    causal_conv1d,
    causal_conv1d_bwd_native,
    causal_conv1d_bwd_ref,
    causal_conv1d_fwd_native,
    causal_conv1d_update_ref,
    conv_output_shape,
    conv_state_shape,
    names,
    resolve,
)
from tests.conftest import assert_max_rel, max_err
from tests.test_conv_reference import (
    assert_bitwise,
    head_major_of,
    make_call,
    oracle,
)

pytestmark = [pytest.mark.cuda]

# (B, T, D, W). Sweeps T against both time tiles, which differ between the two
# directions, D against the 32-channel warp and the channel block, batch parity
# against the backward's stream interleaving, the smallest legal B, D, and T, both
# ends of the width range, and T below W-1, where the returned window straddles
# the incoming state. The ids name the property, not a tile count: the tiles are
# tuning constants and have changed.
SHAPES = [
    pytest.param(2, 40, 8, 4, id="base"),
    pytest.param(1, 1, 1, 4, id="single-token"),
    pytest.param(1, 1, 1, 1, id="pointwise-single"),
    pytest.param(2, 16, 3, 1, id="pointwise"),
    pytest.param(2, 5, 4, 8, id="short-wide"),
    pytest.param(1, 3, 2, 8, id="straddle"),
    pytest.param(3, 64, 32, 4, id="exact-tiles-one-warp-odd-batch"),
    pytest.param(2, 65, 33, 4, id="ragged-tile-ragged-warp"),
    pytest.param(2, 200, 128, 4, id="many-tiles-full-blocks"),
    pytest.param(2, 130, 129, 2, id="many-tiles-ragged-last-block"),
]

# Against the float64 oracle: one rounding of the output at the storage width.
# bfloat16 keeps 7 mantissa bits and float16 keeps 10, so half an ulp is at most
# 2^-8 and 2^-11 of the largest output element. float32 is dominated instead by
# accumulating a W-term sum, W <= 8, at 2^-24 per add.
ORACLE_DTYPES = [
    pytest.param(torch.float32, 2e-6, id="fp32"),
    pytest.param(torch.bfloat16, 5e-3, id="bf16"),
    pytest.param(torch.float16, 1e-3, id="fp16"),
]

# Against the reference at the same dtype: both accumulate in float32 and differ
# only in summation order, so the float32 gap is the same 2^-24 per add. That gap
# is far below a bfloat16 or float16 ulp, so it can only straddle one rounding
# boundary, which bounds the disagreement at exactly one ulp of the largest
# element: 2^-7 = 7.8e-3 and 2^-10 = 9.8e-4. These are ceilings, not estimates;
# the report shows the measured figures at 20% and 2% of them.
PARITY_DTYPES = [
    pytest.param(torch.float32, 2e-6, id="fp32"),
    pytest.param(torch.bfloat16, 7.8e-3, id="bf16"),
    pytest.param(torch.float16, 9.8e-4, id="fp16"),
]

FLAGS = [
    pytest.param(True, True, True, id="act-bias-state"),
    pytest.param(False, True, True, id="noact-bias-state"),
    pytest.param(True, False, True, id="act-nobias-state"),
    pytest.param(True, True, False, id="act-bias-nostate"),
    pytest.param(False, False, False, id="bare"),
]

# The gradients run in float32 against a float64 authority, so every bound is a
# count of float32 roundings at 2^-24 = 6.0e-8 each. dx sums W <= 8 terms, each
# carrying the rounding of the activation derivative and of its own W-term tap
# sum: 2*W*2^-24 = 1.0e-6. The parameter gradients reduce over B*T <= 400 terms
# of mixed sign, so their error grows as sqrt(400)*2^-24 = 1.2e-6, not linearly.
# Both bounds sit just above the estimate: the report shows them at 16% and 13%.
DX_BOUND = 2e-6
DPARAM_BOUND = 4e-6

OPERAND_NAMES = ("x", "weight", "bias", "initial_state")

# Operands that are buffers of their own, so the layout rule on them is contiguity.
# x is the exception: it is one band of the fused projection and is pitched.
CONTIGUOUS_OPERANDS = ("weight", "bias", "initial_state")

# Columns of padding on each side of a test band. A band whose pitch exceeds its row
# width is a strict band and owes :data:`slinoss._guard.SECTOR_BYTES`, not the vector
# width, so sixteen elements: 32 B at bfloat16 and 64 B at float32, and the pitch is
# a whole number of sectors at every channel count used here.
PAD = 16


def cuda_call(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    *,
    bias: bool = True,
    state: bool = True,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    requires_grad: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """``make_call`` on the default CUDA device."""
    return make_call(
        bsz,
        seqlen,
        channels,
        width,
        bias=bias,
        state=state,
        dtype=dtype,
        device="cuda",
        seed=seed,
        requires_grad=requires_grad,
    )


def _leaf(t: Tensor) -> Tensor:
    """``t`` as a float64 differentiable leaf."""
    return t.detach().double().requires_grad_(True)


def as_reference(
    operands: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """The same operands as float64 leaves, for the gradient authority."""
    x, weight, bias, state = operands
    return (
        _leaf(x),
        _leaf(weight),
        None if bias is None else _leaf(bias),
        None if state is None else _leaf(state),
    )


def cotangents(
    bsz: int, seqlen: int, channels: int, width: int, seed: int
) -> tuple[Tensor, Tensor]:
    """``(dy, dfinal_state)`` on the default CUDA device, float32."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        return torch.randn(
            *shape, generator=gen, dtype=torch.float32, device="cuda"
        ).contiguous()

    return rnd(bsz, seqlen, channels), rnd(*conv_state_shape(bsz, width, channels))


def _check_oracle(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    dtype: torch.dtype,
    bound: float,
    *,
    activation: bool = True,
    with_bias: bool = True,
    with_state: bool = True,
) -> None:
    """One native forward against the float64 oracle. The only copy of the body."""
    x, weight, bias, state = cuda_call(
        bsz, seqlen, channels, width, bias=with_bias, state=with_state, dtype=dtype
    )
    want = oracle(x, weight, bias, activation=activation, initial_state=state)
    got = causal_conv1d_fwd_native(
        x, weight, bias, activation=activation, initial_state=state
    )
    assert got.y.dtype == dtype
    assert got.y.is_contiguous()
    assert_max_rel(got.y, want.y, bound, f"conv/native/y/{dtype}")
    assert got.state.shape == want.state.shape
    # The window is a copy of the input, so it is exact at every dtype.
    assert_bitwise(got.state, want.state)


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
def test_forward_matches_the_oracle(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    """The tiling, at the widest storage width.

    Shape is the axis the kernel's decomposition depends on: the channel tile, the
    time tile, the tile prologue, and the final-window epilogue are all indexed
    from it. The other two axes do not interact with it. The dtype is a template
    parameter over scalar loads and a float32 accumulator, with no dtype-dependent
    tiling, so it is swept once below; the flags are epilogue terms, likewise.
    """
    _check_oracle(bsz, seqlen, channels, width, torch.float32, 2e-6)


@pytest.mark.parametrize(("dtype", "bound"), ORACLE_DTYPES)
@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width"),
    [
        pytest.param(2, 40, 8, 4, id="base"),
        pytest.param(2, 65, 33, 4, id="ragged-tile-ragged-warp"),
    ],
)
def test_forward_matches_the_oracle_at_every_width(
    bsz: int, seqlen: int, channels: int, width: int, dtype: torch.dtype, bound: float
) -> None:
    """Every instantiated storage width, at a whole shape and a ragged one.

    The ragged shape is here because the load and the store are the only
    dtype-dependent code the kernel has, and the tail of a partial tile is where a
    width-dependent addressing fault would land.
    """
    _check_oracle(bsz, seqlen, channels, width, dtype, bound)


@pytest.mark.parametrize(("activation", "with_bias", "with_state"), FLAGS)
def test_forward_flag_combinations_match_the_oracle(
    activation: bool, with_bias: bool, with_state: bool
) -> None:
    """Every combination of the three optional terms, at one shape.

    An absent bias and an absent state reach the kernel as a null pointer, so each
    is a separate branch there; the activation is the epilogue.
    """
    _check_oracle(
        2,
        40,
        8,
        4,
        torch.float32,
        2e-6,
        activation=activation,
        with_bias=with_bias,
        with_state=with_state,
    )


@pytest.mark.parametrize(("dtype", "bound"), PARITY_DTYPES)
def test_forward_matches_the_reference_at_the_same_dtype(
    dtype: torch.dtype, bound: float
) -> None:
    """Summation order against the reference, at each storage width.

    What this bounds is the gap between two orderings of one W-term sum, which is
    a property of the width and not of the tiling; the oracle sweep above is what
    covers the tiling.
    """
    bsz, seqlen, channels, width = 2, 200, 33, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width, dtype=dtype)
    want = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    got = causal_conv1d_fwd_native(x, weight, bias, initial_state=state)
    assert_max_rel(got.y, want.y, bound, f"conv/native-vs-ref/y/{dtype}")
    assert_bitwise(got.state, want.state)


# (B, T, D, W, P) with D = H*P. The layout is a store address, so what it interacts
# with is the store's own geometry: the head count, which sets how many contiguous
# runs a channel block's warps write; the time tiling, which the store is inside;
# and the width, which the epilogue is reached through. H = 1 is on the sweep
# because the head-major address collapses to the token-major one there -- a stride
# of P is a stride of D -- so a wrong stride still agrees at H = 1.
HEAD_MAJOR_SHAPES = [
    pytest.param(2, 40, 32, 4, 16, id="two-heads-one-warp"),
    pytest.param(2, 200, 128, 4, 16, id="many-tiles-two-blocks"),
    pytest.param(2, 65, 32, 4, 16, id="ragged-tile"),
    pytest.param(1, 3, 32, 8, 16, id="straddle-widest"),
    pytest.param(2, 16, 32, 1, 16, id="pointwise"),
    pytest.param(2, 40, 64, 4, 64, id="one-head"),
]


@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width", "d_head"), HEAD_MAJOR_SHAPES
)
@pytest.mark.parametrize("activation", [True, False])
def test_head_major_forward_is_the_token_major_forward_reindexed(
    bsz: int, seqlen: int, channels: int, width: int, d_head: int, activation: bool
) -> None:
    """Bitwise, because the layout is a store address and nothing else.

    Bitwise is the whole claim: a staging pass, a promoted intermediate, or a
    different accumulation order would show here as a rounding difference rather
    than as a wrong answer, and the token-major side is what the oracle sweep
    already pins. The activation is swept because the epilogue and the store are
    one expression in the kernel.
    """
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    want = causal_conv1d_fwd_native(
        x, weight, bias, activation=activation, initial_state=state
    )
    got = causal_conv1d_fwd_native(
        x, weight, bias, activation=activation, initial_state=state, d_head=d_head
    )
    assert got.y.shape == conv_output_shape(bsz, seqlen, channels, d_head)
    assert got.y.is_contiguous()
    assert_bitwise(got.y, head_major_of(want.y, d_head))
    assert_bitwise(got.state, want.state)


def _check_backward(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    *,
    activation: bool = True,
    with_bias: bool = True,
    with_state: bool = True,
    dtype: torch.dtype = torch.float32,
    bounds: tuple[float, float] = (DX_BOUND, DPARAM_BOUND),
    band: bool = False,
) -> None:
    """One native backward against float64 autograd. The only copy of the body."""
    operands = cuda_call(
        bsz, seqlen, channels, width, bias=with_bias, state=with_state, dtype=dtype
    )
    leaves = as_reference(operands)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=3)
    # The authority reads the rounded cotangents, so a two-byte call and its float64
    # image consume the same numbers and the gap left is accumulation alone.
    dy, dstate = dy.to(dtype), dstate.to(dtype)
    x = _band(operands[0]) if band else operands[0]
    dx_bound, param_bound = bounds

    got = causal_conv1d_bwd_native(
        dy,
        dstate,
        x,
        *operands[1:3],
        activation=activation,
        initial_state=operands[3],
    )
    want = causal_conv1d_bwd_ref(
        dy.double(),
        dstate.double(),
        *leaves[:3],
        activation=activation,
        initial_state=leaves[3],
    )
    assert_max_rel(got.dx, want.dx, dx_bound, "conv/native/dx")
    assert_max_rel(got.dweight, want.dweight, param_bound, "conv/native/dweight")
    # Present exactly when the forward carried a bias. The reduction reaches dbias
    # through one slice of its grid, so an absent bias is one fewer slice rather
    # than an output that is allocated and left unwritten.
    assert (got.dbias is None) == (not with_bias)
    if got.dbias is not None:
        assert want.dbias is not None
        assert_max_rel(got.dbias, want.dbias, param_bound, "conv/native/dbias")
    if got.dinitial_state is not None and got.dinitial_state.numel():
        assert want.dinitial_state is not None
        assert_max_rel(
            got.dinitial_state, want.dinitial_state, dx_bound, "conv/native/dstate"
        )


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
def test_backward_matches_float64_autograd_through_the_reference(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    """The backward tiling, with every gradient live.

    The backward's decomposition differs from the forward's: it walks the batch in
    interleaved groups inside a serial loop, reads a window that overhangs its tile
    by ``W-1``, and emits one parameter-gradient partial per time tile. Shape selects
    all three, so it carries the sweep; the flags are swept once below.

    Batch parity is part of what shape selects. A batch that is not a multiple of the
    interleaving width leaves the final group with dead streams, which address a
    clamped batch entry so that their loads stay in bounds and are held back from
    accumulating. An odd batch is what distinguishes a dead stream that is correctly
    silent from one that double-counts the entry it was clamped onto.
    """
    _check_backward(bsz, seqlen, channels, width)


@pytest.mark.parametrize("width", range(1, int(_C.extension().MAX_WIDTH) + 1))
def test_backward_matches_float64_autograd_at_every_instantiated_width(
    width: int,
) -> None:
    """Every width the backward instantiates, at one multi-tile shape.

    Width is a template parameter of the backward kernel, so each value is its own
    compiled instantiation reached through a switch. A case that launches the wrong
    instantiation is invisible to the shape sweep above, which only covers the
    widths its shapes happen to carry. Shape does not interact: the tiling is
    indexed from the sequence length and the channel count, and the width enters it
    only as the ``W-1`` overhang, so one shape wide enough to overhang carries all
    eight.
    """
    _check_backward(2, 40, 8, width)


# Gradients at a two-byte dtype. The kernel accumulates in float32 and rounds once
# on the store, so the gap to the float64 authority is that rounding: half an ulp of
# bfloat16 is 2^-9 = 2.0e-3 relative, and the float32 sum behind it adds
# B*T*2^-24 = 4.8e-6 at the shape below. The bound is that rounding with room for
# one boundary straddle.
BF16_GRAD_BOUND = 5e-3

# The backward stages x and dy through shared memory only at a two-byte dtype and
# only when every stride its 16 B request multiplies is a whole number of those
# requests; the gate is in the kernel file. The strip is a second instantiation
# reached at run time, and no case above reaches it, because every one of them is
# float32.
STAGED_CASES = [
    pytest.param(16, 4, False, id="staged"),
    pytest.param(16, 8, False, id="staged-widest-strip"),
    pytest.param(16, 4, True, id="staged-pitched-x"),
    pytest.param(12, 4, False, id="scalar-fallback"),
]


@pytest.mark.parametrize(("channels", "width", "band"), STAGED_CASES)
def test_backward_at_bfloat16_matches_float64_autograd(
    channels: int, width: int, band: bool
) -> None:
    """The staged strip and the walk it falls back to, at the dtype that reaches them.

    Four failure modes, one case each: the strip; its span at the widest width,
    which is ``2*kBwdTileT + 2*(W-1)`` timesteps and the only thing width changes
    about it; the row pitch, which the fill multiplies by a timestep index instead
    of walking; and the gate, which must refuse a channel count that is not a whole
    number of requests and fall back to the scalar walk at the same dtype. float16
    reaches the same instantiation over the same strides and is left to the
    forward's dtype sweep.
    """
    _check_backward(
        2,
        40,
        channels,
        width,
        dtype=torch.bfloat16,
        bounds=(BF16_GRAD_BOUND, BF16_GRAD_BOUND),
        band=band,
    )


@pytest.mark.parametrize(("activation", "with_bias", "with_state"), FLAGS)
def test_backward_flag_combinations_match_float64_autograd(
    activation: bool, with_bias: bool, with_state: bool
) -> None:
    """Every combination of the three optional terms, at a multi-tile shape.

    Absent bias means no ``dbias`` partials to reduce and absent state means no
    ``dinitial_state`` to write, so each flag removes an output rather than changing
    the tiling.
    """
    _check_backward(
        2, 200, 33, 4, activation=activation, with_bias=with_bias, with_state=with_state
    )


@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width", "d_head"), HEAD_MAJOR_SHAPES
)
def test_head_major_backward_is_the_token_major_backward_reindexed(
    bsz: int, seqlen: int, channels: int, width: int, d_head: int
) -> None:
    """Bitwise: a head-major cotangent is a load address and nothing else.

    Every gradient is compared rather than ``dx`` alone. The parameter gradients
    reduce over every ``(b,t)`` the block owns, so a ``dy`` load that reads the
    wrong head is visible there even at a shape where ``dx`` happens to agree.
    """
    operands = cuda_call(bsz, seqlen, channels, width)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=29)
    want = causal_conv1d_bwd_native(
        dy, dstate, *operands[:3], initial_state=operands[3]
    )
    got = causal_conv1d_bwd_native(
        head_major_of(dy, d_head), dstate, *operands[:3], initial_state=operands[3]
    )
    for have, expect in zip(got, want):
        assert have is not None and expect is not None
        assert_bitwise(have, expect)


@pytest.mark.parametrize(
    ("with_dy", "with_dstate"),
    [
        pytest.param(True, False, id="dy-only"),
        pytest.param(False, True, id="dstate-only"),
        pytest.param(False, False, id="neither"),
    ],
)
def test_absent_cotangents_match_the_reference(
    with_dy: bool, with_dstate: bool
) -> None:
    bsz, seqlen, channels, width = 2, 9, 3, 4
    operands = cuda_call(bsz, seqlen, channels, width)
    leaves = as_reference(operands)
    full_dy, full_dstate = cotangents(bsz, seqlen, channels, width, seed=5)
    dy = full_dy if with_dy else None
    dstate = full_dstate if with_dstate else None

    got = causal_conv1d_bwd_native(dy, dstate, *operands[:3], initial_state=operands[3])
    want = causal_conv1d_bwd_ref(
        None if dy is None else dy.double(),
        None if dstate is None else dstate.double(),
        *leaves[:3],
        initial_state=leaves[3],
    )
    assert_max_rel(got.dx, want.dx, DX_BOUND, "conv/native/dx/partial")
    assert want.dweight is not None
    assert_max_rel(
        got.dweight, want.dweight, DPARAM_BOUND, "conv/native/dweight/partial"
    )
    assert got.dbias is not None and want.dbias is not None
    assert_max_rel(got.dbias, want.dbias, DPARAM_BOUND, "conv/native/dbias/partial")
    assert got.dinitial_state is not None and want.dinitial_state is not None
    assert_max_rel(
        got.dinitial_state,
        want.dinitial_state,
        DX_BOUND,
        "conv/native/dstate/partial",
    )


# What the connection test catches is a wiring fault rather than a tiling fault: a
# saved tensor that is not the one the backward reads, or a gradient dropped on the
# way out. Both halves are swept by shape above, so three wirings suffice here --
# W = 1, where the incoming window is empty and one gradient is legitimately
# absent; a sub-tile sequence; and a multi-tile multi-block one.
CONNECTED_SHAPES = [
    pytest.param(2, 40, 8, 4, id="base"),
    pytest.param(2, 16, 3, 1, id="pointwise"),
    pytest.param(2, 130, 129, 2, id="many-tiles-ragged-last-block"),
]


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), CONNECTED_SHAPES)
def test_forward_and_backward_are_connected(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    # Runs the fast forward and then backpropagates through it, so the backward is
    # measured against the forward that ships rather than against a surrogate.
    operands = cuda_call(bsz, seqlen, channels, width, requires_grad=True)
    leaves = as_reference(operands)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=13)

    out = causal_conv1d(*operands[:3], initial_state=operands[3], backend="native")
    ((out.y * dy).sum() + (out.state * dstate).sum()).backward()

    ref = causal_conv1d_update_ref(*leaves[:3], initial_state=leaves[3])
    ((ref.y * dy.double()).sum() + (ref.state * dstate.double()).sum()).backward()

    for name, have, expect in zip(OPERAND_NAMES, operands, leaves):
        assert have is not None and expect is not None
        if have.numel() == 0:
            # At W = 1 the incoming window is empty and reaches no output.
            continue
        assert have.grad is not None and expect.grad is not None
        bound = DX_BOUND if name in ("x", "initial_state") else DPARAM_BOUND
        assert_max_rel(have.grad, expect.grad, bound, f"conv/native/connected/{name}")


@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width", "d_head"),
    [
        pytest.param(2, 40, 32, 4, 16, id="two-heads"),
        pytest.param(2, 200, 128, 4, 16, id="many-tiles-two-blocks"),
    ],
)
def test_head_major_forward_and_backward_are_connected(
    bsz: int, seqlen: int, channels: int, width: int, d_head: int
) -> None:
    """The public path at ``d_head`` against the float64 reference.

    Two shapes, because what this catches is wiring: ``d_head`` is not saved, so a
    keyword that reached the forward and left the backward reading the cotangent as
    token-major is only visible where autograd supplies the cotangent itself.
    """
    operands = cuda_call(bsz, seqlen, channels, width, requires_grad=True)
    leaves = as_reference(operands)
    gen = torch.Generator(device="cuda").manual_seed(31)
    dy = torch.randn(
        *conv_output_shape(bsz, seqlen, channels, d_head),
        generator=gen,
        dtype=torch.float32,
        device="cuda",
    )
    _, dstate = cotangents(bsz, seqlen, channels, width, seed=31)

    out = causal_conv1d(
        *operands[:3], initial_state=operands[3], d_head=d_head, backend="native"
    )
    ((out.y * dy).sum() + (out.state * dstate).sum()).backward()

    ref = causal_conv1d_update_ref(*leaves[:3], initial_state=leaves[3], d_head=d_head)
    ((ref.y * dy.double()).sum() + (ref.state * dstate.double()).sum()).backward()

    assert_max_rel(out.y, ref.y, 2e-6, "conv/native/head-major/y")
    for name, have, expect in zip(OPERAND_NAMES, operands, leaves):
        assert have is not None and expect is not None
        assert have.grad is not None and expect.grad is not None
        bound = DX_BOUND if name in ("x", "initial_state") else DPARAM_BOUND
        assert_max_rel(have.grad, expect.grad, bound, f"conv/native/head-major/{name}")


def _check_split(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    chunks: int,
    d_head: int | None = None,
) -> None:
    """One streaming split against the whole-sequence call. The only copy."""
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    whole = causal_conv1d_fwd_native(
        x, weight, bias, initial_state=state, d_head=d_head
    )
    # Uneven splits on purpose: an even split hides an off-by-one in the window
    # that is carried across a boundary.
    sizes = [seqlen // chunks] * chunks
    sizes[-1] += seqlen - sum(sizes)
    sizes = [s for s in sizes if s > 0]
    carry = state
    pieces: list[Tensor] = []
    for piece in torch.split(x, sizes, dim=1):
        out = causal_conv1d_fwd_native(
            piece.contiguous(), weight, bias, initial_state=carry, d_head=d_head
        )
        pieces.append(out.y)
        carry = out.state
    assert carry is not None
    # The token axis is -2 at both output layouts.
    joined = torch.cat(pieces, dim=-2)
    assert joined.shape == whole.y.shape
    # Exact, not approximate: the split changes which block owns a token, not the
    # order its tap sum is accumulated in.
    assert max_err(joined, whole.y) == 0.0
    assert_bitwise(carry, whole.state)


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
def test_streaming_split_reproduces_the_whole_sequence(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    """A two-piece split at every shape.

    Two pieces is the only split the short shapes admit, and it is the split that
    exercises the carry: the second call reads a window the first call wrote.
    """
    _check_split(bsz, seqlen, channels, width, 2)


@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width", "d_head"),
    [
        pytest.param(2, 40, 32, 4, 16, id="two-heads"),
        pytest.param(1, 3, 32, 8, 16, id="straddle-widest"),
    ],
)
def test_head_major_streaming_split_reproduces_the_whole_sequence(
    bsz: int, seqlen: int, channels: int, width: int, d_head: int
) -> None:
    """The identity at a layout the carry does not share.

    The carried window stays token-major while ``y`` does not, so this is what
    catches a state written in the output's layout: the second call would read the
    right elements from the wrong place and only the boundary tokens would move.
    Two shapes, not the whole sweep: the failure mode is which layout the carry is
    written in, which no head count reaches, and the straddle is the case where the
    carry crosses into the incoming state.
    """
    _check_split(bsz, seqlen, channels, width, 2, d_head=d_head)


def test_a_seven_piece_split_reproduces_the_whole_sequence() -> None:
    """One deep split: the carry is threaded six times.

    Only the long shapes can split seven ways at all, so this is a single case
    rather than a second axis on the sweep above.
    """
    _check_split(2, 200, 33, 4, 7)


def test_token_at_a_time_decode_reproduces_the_whole_sequence() -> None:
    bsz, seqlen, channels, width = 2, 40, 8, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    whole = causal_conv1d_fwd_native(x, weight, bias, initial_state=state)
    carry = state
    steps: list[Tensor] = []
    for t in range(seqlen):
        out = causal_conv1d_fwd_native(
            x[:, t : t + 1].contiguous(), weight, bias, initial_state=carry
        )
        steps.append(out.y)
        carry = out.state
    assert max_err(torch.cat(steps, dim=1), whole.y) == 0.0


def test_partial_count_stops_growing_with_the_sequence() -> None:
    """The stack the reduction reads is bounded by the launch target, not by T.

    A count that follows T puts the reduction over its class ceiling at the long
    shape; the bound is the whole point of the grid-stride walk. A loop, not a
    parametrize: the assertion is arithmetic on one host function, so the cases
    share everything except an integer and none can fail independently.
    """
    module = _C.extension()
    tile = int(module.BWD_TILE_T)
    per_block = int(module.BWD_TILES_PER_BLOCK)
    target = int(module.BWD_TARGET_BLOCKS)
    cpb = int(module.CHANNELS_PER_BLOCK)
    for channels in (16, 64, 576):
        rows = -(-channels // min(-(-channels // 32) * 32, cpb))
        # The floor in the stride division leaves at most one extra iteration's
        # worth of blocks over the target's share of the channel axis.
        ceiling = 2 * -(-target // rows)
        for seqlen in (1, tile - 1, tile, tile + 1, 200, 8192, 1 << 20):
            parts = int(module.bwd_parts(seqlen, channels))
            tiles = -(-seqlen // tile)
            groups = -(-tiles // per_block)
            assert 1 <= parts <= min(groups, ceiling)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_every_partial_slice_is_written(dtype: torch.dtype) -> None:
    """No slice of the stack is left at its allocated value.

    A block writes one slice for every tile group it walks, so a capped grid has
    blocks that walk several groups and lanes that own no tile in the group they
    are in. A slice nobody writes reaches the reduction as whatever the allocator
    left, and NaN is what makes that visible where zero would pass. Both dtypes:
    the staged path and the scalar walk leave the group loop by different routes.

    The geometry is the one that caps: 513 tiles is more than twice the target's
    share of a 576-channel grid, and the odd tile count is what leaves a lane in
    the last group with nothing of its own.
    """
    module = _C.extension()
    bsz, seqlen, channels, width = 1, 8200, 576, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width, dtype=dtype)
    assert state is not None
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=23)
    parts = int(module.bwd_parts(seqlen, channels))
    tiles = -(-seqlen // int(module.BWD_TILE_T))
    assert parts < -(-tiles // int(module.BWD_TILES_PER_BLOCK))
    nan = float("nan")
    dweight_parts = x.new_full((parts, width, channels), nan, dtype=torch.float32)
    dbias_parts = x.new_full((parts, channels), nan, dtype=torch.float32)
    module.bwd(
        dy.to(dtype),
        dstate.to(dtype),
        x,
        weight,
        bias,
        state,
        torch.empty_like(x),
        torch.empty_like(state),
        dweight_parts,
        dbias_parts,
        True,
    )
    assert bool(torch.isfinite(dweight_parts).all())
    assert bool(torch.isfinite(dbias_parts).all())


def test_more_than_one_partial_is_reduced() -> None:
    # A single-tile sequence cannot detect a dropped partial. This one spans many
    # backward tiles, so each parameter gradient is a sum over slices. The count is
    # asserted from the exported tile rather than written here, because the tile is a
    # tuning constant.
    bsz, seqlen, channels, width = 2, 200, 8, 4
    operands = cuda_call(bsz, seqlen, channels, width)
    leaves = as_reference(operands)
    assert _C.extension().bwd_parts(seqlen, channels) > 1
    dy, _ = cotangents(bsz, seqlen, channels, width, seed=17)

    got = causal_conv1d_bwd_native(dy, None, *operands[:3], initial_state=operands[3])
    want = causal_conv1d_bwd_ref(
        dy.double(), None, *leaves[:3], initial_state=leaves[3]
    )
    assert want.dbias is not None and got.dbias is not None
    assert_max_rel(
        got.dweight, want.dweight, DPARAM_BOUND, "conv/native/dweight/multi-tile"
    )
    assert_max_rel(got.dbias, want.dbias, DPARAM_BOUND, "conv/native/dbias/multi-tile")


def _bwd_partials(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    dtype: torch.dtype,
    *,
    with_bias: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """The backward kernel's parameter-gradient partials.

    Taken from the backward kernel rather than from ``randn``: the reduction's
    operand is that kernel's output, and a fabricated stack would not carry its
    layout or its magnitudes.

    Args:
        bsz: Batch.
        seqlen: Tokens. Long enough that the stack holds more than one slice.
        channels: Channels.
        width: Tap count.
        dtype: Storage dtype of the call.
        with_bias: Build a bias, which is what makes the bias partials exist.

    Returns:
        ``(dweight_parts, dbias_parts)``, float32, shapes ``(S,W,D)`` and
        ``(S,D)``. The second is None when the call carries no bias.
    """
    module = _C.extension()
    x, weight, bias, state = cuda_call(
        bsz, seqlen, channels, width, bias=with_bias, dtype=dtype
    )
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=41)
    assert state is not None
    parts = int(module.bwd_parts(seqlen, channels))
    assert parts > 1
    dweight_parts = x.new_empty((parts, width, channels), dtype=torch.float32)
    dbias_parts = (
        None if bias is None else x.new_empty((parts, channels), dtype=torch.float32)
    )
    module.bwd(
        dy.to(dtype),
        dstate.to(dtype),
        x,
        weight,
        bias,
        state,
        torch.empty_like(x),
        torch.empty_like(state),
        dweight_parts,
        dbias_parts,
        True,
    )
    return dweight_parts, dbias_parts


def _reduced(
    dweight_parts: Tensor, dbias_parts: Tensor | None, dtype: torch.dtype
) -> tuple[Tensor, Tensor | None]:
    """One ``bwd_reduce`` over a partial stack, into fresh outputs."""
    channels = int(dweight_parts.shape[2])
    width = int(dweight_parts.shape[1])
    dweight = dweight_parts.new_empty((channels, width), dtype=dtype)
    dbias = (
        None
        if dbias_parts is None
        else dweight_parts.new_empty((channels,), dtype=dtype)
    )
    _C.extension().bwd_reduce(dweight_parts, dbias_parts, dweight, dbias)
    return dweight, dbias


@pytest.mark.parametrize(("dtype", "bound"), PARITY_DTYPES)
def test_the_partial_reduction_matches_the_torch_tail(
    dtype: torch.dtype, bound: float
) -> None:
    """The reduction against the torch expression it replaces, at each dtype.

    The expression is ``parts.sum(0).t().contiguous().to(dtype)`` for the taps and
    ``parts.sum(0).to(dtype)`` for the bias: a reduction, a transpose copy, and a
    cast, which the kernel does in one launch and one store. Dtype is the axis
    swept because the store is where the layout and the narrowing happen and the
    kernel is one instantiation per output dtype; shape reaches only the grid, and
    the sweep above already carries it.

    The two sums differ in order, so the bound is the same one-ulp ceiling the
    forward parity test is held to and for the same reason.
    """
    bsz, seqlen, channels, width = 2, 200, 33, 4
    dweight_parts, dbias_parts = _bwd_partials(bsz, seqlen, channels, width, dtype)
    assert dbias_parts is not None
    want_dweight = dweight_parts.sum(0).t().contiguous().to(dtype)
    want_dbias = dbias_parts.sum(0).to(dtype)
    dweight, dbias = _reduced(dweight_parts, dbias_parts, dtype)
    assert dbias is not None
    assert dweight.shape == (channels, width) and dweight.is_contiguous()
    assert_max_rel(dweight, want_dweight, bound, f"conv/native/reduce/dweight/{dtype}")
    assert_max_rel(dbias, want_dbias, bound, f"conv/native/reduce/dbias/{dtype}")


def test_the_partial_reduction_is_reproducible() -> None:
    """Bitwise across two runs on one partial stack.

    What this pins is the reduction order. The stack is split across the block's
    rows, so a float atomic into the output or a combine that followed the
    scheduler rather than the row index would agree here only by luck. float32
    outputs, because a narrower store would round the differing bits away.
    """
    dweight_parts, dbias_parts = _bwd_partials(2, 200, 33, 4, torch.float32)
    first = _reduced(dweight_parts, dbias_parts, torch.float32)
    second = _reduced(dweight_parts, dbias_parts, torch.float32)
    for have, again in zip(first, second):
        assert have is not None and again is not None
        assert_bitwise(have, again)


def _band(t: Tensor) -> Tensor:
    """``t``'s values as one column band of a wider buffer.

    What the fused input projection hands over: the band's row pitch is the
    projection width, so neither the base offset nor the pitch is the one a
    dedicated buffer would have. :data:`PAD` columns sit on each side.
    """
    width = int(t.shape[-1])
    wide = t.new_empty((*t.shape[:-1], width + 2 * PAD))
    band = wide[..., PAD : PAD + width]
    band.copy_(t)
    return band


def test_the_forward_reads_a_band_of_the_projection() -> None:
    """``x`` ships pitched, and the kernel indexes the band rather than a copy.

    One projection GEMM feeds every consumer, so ``x`` is the value band of its
    output and never a buffer of its own; recovering contiguity would be the
    staging copy the layout contract exists to refuse. The pitch moves a load
    address and no arithmetic, so the two layouts agree bit for bit rather than
    within a tolerance.
    """
    x, weight, bias, state = cuda_call(2, 40, 8, 4)
    want = causal_conv1d_fwd_native(x, weight, bias, initial_state=state)
    got = causal_conv1d_fwd_native(_band(x), weight, bias, initial_state=state)
    assert_bitwise(got.y, want.y)
    assert_bitwise(got.state, want.state)


def test_the_backward_reads_and_writes_bands_of_the_projection() -> None:
    """``x`` read pitched and ``dx`` written pitched, against the contiguous call.

    Two addresses, so two halves. The read side runs through the backend, which is
    where the host rule on ``x`` lives, and covers the parameter gradients too:
    ``x`` is the operand the ``dweight`` sum contracts against, so a pitch honoured
    for ``dx`` and dropped there would show up in ``dweight`` alone. The store side
    runs through the extension, which is where the store contract is stated; the
    band's wider buffer is filled with NaN, so a store that ignored the pitch would
    both leave the band unwritten and land in a column the call does not own.
    """
    bsz, seqlen, channels, width = 2, 40, 8, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=23)
    want = causal_conv1d_bwd_native(dy, dstate, x, weight, bias, initial_state=state)
    got = causal_conv1d_bwd_native(
        dy, dstate, _band(x), weight, bias, initial_state=state
    )
    assert want.dbias is not None and got.dbias is not None
    assert want.dinitial_state is not None and got.dinitial_state is not None
    assert_bitwise(got.dx, want.dx)
    assert_bitwise(got.dweight, want.dweight)
    assert_bitwise(got.dbias, want.dbias)
    assert_bitwise(got.dinitial_state, want.dinitial_state)

    assert state is not None
    module = _C.extension()
    parts = int(module.bwd_parts(seqlen, channels))
    wide = x.new_full((bsz, seqlen, channels + 2 * PAD), float("nan"))
    band = wide[..., PAD : PAD + channels]
    module.bwd(
        dy,
        dstate,
        x,
        weight,
        bias,
        state,
        band,
        torch.empty_like(state),
        x.new_empty((parts, width, channels), dtype=torch.float32),
        x.new_empty((parts, channels), dtype=torch.float32),
        True,
    )
    assert_bitwise(band, want.dx)
    outside = torch.cat((wide[..., :PAD], wide[..., PAD + channels :]), dim=-1)
    assert bool(outside.isnan().all())


def test_a_supplied_dx_is_the_store_target() -> None:
    """The ``dx`` out parameter: the caller's band is the store target.

    One fused buffer holds the gradient of the whole projection, so a consumer owns
    a band of it and never a buffer of its own. The wider buffer is filled with NaN,
    so a store into an allocation of the backend's own would leave the band NaN and
    one that ignored the pitch would land in a column the call does not own. The
    pitch moves a store address and no arithmetic, so the band matches the
    allocating path bit for bit.
    """
    bsz, seqlen, channels, width = 2, 40, 8, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=29)
    want = causal_conv1d_bwd_native(dy, dstate, x, weight, bias, initial_state=state)
    wide = x.new_full((bsz, seqlen, channels + 2 * PAD), float("nan"))
    band = wide[..., PAD : PAD + channels]
    got = causal_conv1d_bwd_native(
        dy, dstate, x, weight, bias, initial_state=state, dx=band
    )
    assert got.dx is band
    assert_bitwise(got.dx, want.dx)
    outside = torch.cat((wide[..., :PAD], wide[..., PAD + channels :]), dim=-1)
    assert bool(outside.isnan().all())


def _vector_band(t: Tensor) -> Tensor:
    """``t``'s shape as a band that starts one vector width into a wider buffer.

    The offset a contiguous operand may carry and a strict band may not: the pitched
    rule holds a band to :data:`slinoss._guard.SECTOR_BYTES`, twice the vector width,
    so every row of this one overhangs into a sector nothing reads.
    """
    pad = ALIGN_BYTES // t.element_size()
    width = int(t.shape[-1])
    return t.new_empty((*t.shape[:-1], width + 2 * pad))[..., pad : pad + width]


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        pytest.param(
            lambda t: t.new_empty((*t.shape[:-1], int(t.shape[-1]) + 1)),
            r"dx must be \(2, 40, 8\)",
            id="shape",
        ),
        pytest.param(
            lambda t: t.to(torch.bfloat16), "dx must be torch.float32", id="dtype"
        ),
        pytest.param(lambda t: t.cpu(), "dx must be on cuda", id="device"),
        pytest.param(
            _vector_band, "dx must start and step on a multiple of", id="boundary"
        ),
    ],
)
def test_an_illegal_supplied_dx_is_rejected(
    damage: Callable[[Tensor], Tensor], match: str
) -> None:
    """The whole contract on a destination the caller owns.

    Every row is a store the kernel would otherwise make into memory it cannot
    address, or into a band whose rows each fetch a sector nothing reads. The order
    is the reporting order: the shape, dtype, and device rules precede the layout
    one, so a buffer that violates two is reported under the first.
    """
    bsz, seqlen, channels, width = 2, 40, 8, 4
    x, weight, bias, state = cuda_call(bsz, seqlen, channels, width)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=31)
    with pytest.raises(ValueError, match=match):
        causal_conv1d_bwd_native(
            dy, dstate, x, weight, bias, initial_state=state, dx=damage(x)
        )


def test_native_backend_is_registered_and_preferred() -> None:
    assert "native" in names()
    assert resolve(None, "cuda", torch.float32).name == "native"
    assert resolve(None, "cpu", torch.float32).name == "reference"


def test_float64_resolves_to_the_reference_rather_than_raising() -> None:
    # The kernel is instantiated per dtype and float64 has no instantiation, so
    # resolution routes around it. A float64 call is the oracle width; the caller
    # wants the answer, not an exception from inside a backend it did not name.
    assert resolve(None, "cuda", torch.float64).name == "reference"


def test_public_operator_selects_the_native_backend() -> None:
    x, weight, bias, state = cuda_call(2, 40, 8, 4)
    got = causal_conv1d(x, weight, bias, initial_state=state)
    want = causal_conv1d_fwd_native(x, weight, bias, initial_state=state)
    assert_bitwise(got.y, want.y)
    assert_bitwise(got.state, want.state)


def test_native_backend_does_not_run_on_the_cpu() -> None:
    with pytest.raises(ValueError, match="supports"):
        resolve("native", "cpu", torch.float32)


def test_width_above_the_kernel_bound_is_rejected() -> None:
    bound = int(_C.extension().MAX_WIDTH)
    x, weight, bias, state = cuda_call(2, 40, 8, bound + 1)
    with pytest.raises(ValueError, match=f"width <= {bound}"):
        causal_conv1d_fwd_native(x, weight, bias, initial_state=state)
    with pytest.raises(ValueError, match=f"width <= {bound}"):
        causal_conv1d_bwd_native(
            torch.ones_like(x), None, x, weight, bias, initial_state=state
        )


def test_float64_is_rejected() -> None:
    x, weight, bias, state = cuda_call(2, 40, 8, 4, dtype=torch.float64)
    with pytest.raises(TypeError, match="native backend supports"):
        causal_conv1d_fwd_native(x, weight, bias, initial_state=state)


def _uncontiguous(t: Tensor) -> Tensor:
    """``t``'s shape and values with a stride the kernel does not describe."""
    out = torch.repeat_interleave(t, 2, dim=0)[::2]
    assert not out.is_contiguous()
    return out


def _strided_channels(t: Tensor) -> Tensor:
    """``t``'s shape with a step along the channel axis, which no pitch describes."""
    width = int(t.shape[-1])
    out = t.new_empty((*t.shape[:-1], 2 * width))[..., ::2]
    assert out.shape == t.shape and out.stride(-1) != 1
    return out


def _spoil(name: str, damage: Callable[[Tensor], Tensor]) -> ConvStep:
    """Run the native forward with one operand replaced by ``damage(operand)``.

    Args:
        name: Which of :data:`OPERAND_NAMES` to replace.
        damage: Builds the replacement from the original operand.

    Returns:
        The forward result, on the runs that are not rejected.
    """
    operands: dict[str, Tensor | None] = dict(
        zip(OPERAND_NAMES, cuda_call(2, 40, 8, 4))
    )
    target = operands[name]
    assert target is not None
    operands[name] = damage(target)
    x, weight = operands["x"], operands["weight"]
    assert x is not None and weight is not None
    return causal_conv1d_fwd_native(
        x, weight, operands["bias"], initial_state=operands["initial_state"]
    )


@pytest.mark.parametrize("name", CONTIGUOUS_OPERANDS)
def test_a_noncontiguous_operand_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be contiguous"):
        _spoil(name, _uncontiguous)


def test_a_strided_channel_axis_is_rejected() -> None:
    """The rule x carries instead of contiguity.

    A wider buffer is what a band is, so widening x no longer rejects; a step
    along the channels does, because that is the one stride the kernel's address
    arithmetic cannot express.
    """
    with pytest.raises(ValueError, match="x must have unit stride"):
        _spoil("x", _strided_channels)


@pytest.mark.parametrize("name", ["weight", "bias", "initial_state"])
def test_a_mixed_dtype_operand_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be torch.float32"):
        _spoil(name, lambda t: t.to(torch.bfloat16))


def test_a_cpu_operand_is_rejected() -> None:
    x, weight, bias, state = make_call(2, 40, 8, 4, dtype=torch.float32)
    with pytest.raises(ValueError, match="x must be on a CUDA device"):
        causal_conv1d_fwd_native(x, weight, bias, initial_state=state)


@pytest.mark.parametrize("name", ["dy", "dfinal_state"])
def test_a_noncontiguous_cotangent_is_rejected(name: str) -> None:
    bsz, seqlen, channels, width = 2, 40, 8, 4
    operands = cuda_call(bsz, seqlen, channels, width)
    dy, dstate = cotangents(bsz, seqlen, channels, width, seed=19)
    if name == "dy":
        dy = _uncontiguous(dy)
    else:
        dstate = _uncontiguous(dstate)
    with pytest.raises(ValueError, match=f"{name} must be contiguous"):
        causal_conv1d_bwd_native(dy, dstate, *operands[:3], initial_state=operands[3])
