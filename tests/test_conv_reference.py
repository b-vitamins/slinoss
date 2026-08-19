"""Causal depthwise conv1d: reference, registry, and autograd entry point.

The forward is checked against an oracle that spells the index arithmetic out as
two explicit loops, so the reference's ``unfold`` view has to agree with direct
indexing rather than with itself. Gradients are checked by float64
:func:`torch.autograd.gradcheck`; no pullback is written out anywhere in this
file, because a hand-derived pullback shares its algebra with the forward it came
from and an algebra error would pass silently.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from slinoss import _C
from slinoss.ops.conv import (
    Backend,
    ConvDims,
    ConvStep,
    causal_conv1d,
    causal_conv1d_bwd_ref,
    causal_conv1d_ref,
    causal_conv1d_update_ref,
    check_operands,
    conv_state_shape,
    get,
    names,
    register,
    resolve,
)
from tests.conftest import assert_max_rel, max_err

# (B, T, D, W). Sweeps T against the kernel's 64-token tile, the smallest legal
# B, D, and T, both ends of the width range, and T below W-1, where the returned
# window straddles the incoming state.
SHAPES = [
    pytest.param(2, 40, 8, 4, id="base"),
    pytest.param(1, 1, 1, 4, id="single-token"),
    pytest.param(1, 1, 1, 1, id="pointwise-single"),
    pytest.param(2, 16, 3, 1, id="pointwise"),
    pytest.param(2, 5, 4, 8, id="short-wide"),
    pytest.param(1, 3, 2, 8, id="straddle"),
    pytest.param(3, 64, 4, 4, id="one-tile"),
    pytest.param(2, 65, 4, 4, id="tile-plus-one"),
    pytest.param(2, 130, 2, 2, id="three-tiles"),
]

FLAGS = [
    pytest.param(True, True, True, id="act-bias-state"),
    pytest.param(False, True, True, id="noact-bias-state"),
    pytest.param(True, False, True, id="act-nobias-state"),
    pytest.param(True, True, False, id="act-bias-nostate"),
    pytest.param(False, False, False, id="bare"),
]


def make_call(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    *,
    bias: bool = True,
    state: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    seed: int = 0,
    requires_grad: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Build ``(x, weight, bias, initial_state)`` for one call.

    Args:
        bsz: Batch.
        seqlen: Tokens.
        channels: Channels.
        width: Tap count.
        bias: Build a bias.
        state: Build an incoming window.
        dtype: Dtype of every tensor.
        device: Device of every tensor.
        seed: Generator seed.
        requires_grad: Mark every tensor a differentiable leaf.

    Returns:
        The four operands, contiguous.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> Tensor:
        out = torch.randn(*shape, generator=gen, dtype=dtype, device=device)
        return out.contiguous().requires_grad_(requires_grad)

    return (
        rnd(bsz, seqlen, channels),
        rnd(channels, width),
        rnd(channels) if bias else None,
        rnd(*conv_state_shape(bsz, width, channels)) if state else None,
    )


def oracle(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    activation: bool,
    initial_state: Tensor | None,
) -> ConvStep:
    """Causal conv1d written as explicit loops over time and tap index.

    Independent of the reference: the window is materialized and indexed
    directly instead of being formed by a strided view.

    Args:
        x: Activations, shape ``(B,T,D)``.
        weight: Taps, shape ``(D,W)``.
        bias: Per-channel bias, shape ``(D,)``, or None.
        activation: Apply SiLU.
        initial_state: Previous window, shape ``(B,W-1,D)``, or None.

    Returns:
        A :class:`ConvStep` in float64.
    """
    bsz, seqlen, channels = (int(d) for d in x.shape)
    width = int(weight.shape[1])
    taps = weight.detach().double()
    padded = torch.zeros(
        bsz, seqlen + width - 1, channels, dtype=torch.float64, device=x.device
    )
    if width > 1 and initial_state is not None:
        padded[:, : width - 1] = initial_state.detach().double()
    padded[:, width - 1 :] = x.detach().double()
    y = torch.zeros(bsz, seqlen, channels, dtype=torch.float64, device=x.device)
    for t in range(seqlen):
        for k in range(width):
            y[:, t] += padded[:, t + k] * taps[:, k]
    if bias is not None:
        y = y + bias.detach().double()
    if activation:
        y = y * torch.sigmoid(y)
    return ConvStep(y=y, state=padded[:, seqlen:].clone())


def assert_bitwise(got: Tensor, want: Tensor) -> None:
    """Assert two tensors agree exactly.

    A reduction over an empty tensor has no value, so an empty pair agrees on
    its shape alone. That case is reached at ``W = 1``, where the window is
    ``(B,0,D)``.
    """
    assert got.shape == want.shape
    if got.numel():
        assert max_err(got, want) == 0.0


def _check_oracle(
    bsz: int,
    seqlen: int,
    channels: int,
    width: int,
    *,
    activation: bool,
    with_bias: bool,
    with_state: bool,
    device: torch.device | str,
) -> None:
    """One reference call against the two-loop oracle. The only copy of the body."""
    x, weight, bias, state = make_call(
        bsz, seqlen, channels, width, bias=with_bias, state=with_state, device=device
    )
    want = oracle(x, weight, bias, activation=activation, initial_state=state)
    got = causal_conv1d_update_ref(
        x, weight, bias, activation=activation, initial_state=state
    )
    assert_max_rel(got.y, want.y, 1e-12, "conv/ref/y")
    assert got.state.shape == want.state.shape
    if got.state.numel():
        assert_max_rel(got.state, want.state, 1e-12, "conv/ref/state")


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
def test_forward_matches_oracle(
    bsz: int, seqlen: int, channels: int, width: int, device: torch.device
) -> None:
    """The ``unfold`` view against direct indexing, over the shape sweep.

    The window geometry is what varies with shape, so this axis carries the full
    sweep. The three flags do not touch it: each is a separate term in the
    epilogue or a separate prefix, so they are swept once, below.
    """
    _check_oracle(
        bsz,
        seqlen,
        channels,
        width,
        activation=True,
        with_bias=True,
        with_state=True,
        device=device,
    )


@pytest.mark.parametrize(("activation", "with_bias", "with_state"), FLAGS)
def test_forward_flag_combinations_match_oracle(
    activation: bool, with_bias: bool, with_state: bool
) -> None:
    """Every combination of the three optional terms, at one shape.

    An absent bias and an absent state are ``None`` rather than a zero tensor, so
    each one is a separate branch in the reference; the activation is the epilogue.
    Straddling ``T < W-1`` is covered by the shape sweep, which runs with the state
    present.
    """
    _check_oracle(
        2,
        40,
        8,
        4,
        activation=activation,
        with_bias=with_bias,
        with_state=with_state,
        device="cpu",
    )


# Three shapes rather than the sweep: the two entry points share `_contract`, so
# what is under test is the wiring, and W = 1 (which returns early from the
# padding) plus a straddling window are the only distinct wirings. Every entry
# has more than one token, which causality needs.
WIRING_SHAPES = [
    pytest.param(2, 40, 8, 4, id="base"),
    pytest.param(2, 16, 3, 1, id="pointwise"),
    pytest.param(1, 3, 2, 8, id="straddle"),
]


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), WIRING_SHAPES)
def test_whole_sequence_matches_streaming_form(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    x, weight, bias, state = make_call(bsz, seqlen, channels, width)
    plain = causal_conv1d_ref(x, weight, bias, initial_state=state)
    step = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    assert max_err(plain, step.y) == 0.0


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), WIRING_SHAPES)
def test_output_does_not_depend_on_the_future(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    x, weight, bias, state = make_call(bsz, seqlen, channels, width)
    cut = seqlen // 2
    perturbed = x.clone()
    perturbed[:, cut:] += 100.0
    base = causal_conv1d_ref(x, weight, bias, initial_state=state)
    moved = causal_conv1d_ref(perturbed, weight, bias, initial_state=state)
    assert max_err(base[:, :cut], moved[:, :cut]) == 0.0


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
@pytest.mark.parametrize("chunks", [2, 7])
def test_streaming_split_reproduces_the_whole_sequence(
    bsz: int, seqlen: int, channels: int, width: int, chunks: int
) -> None:
    """A split interacts with the shape through ``T``, so both axes are swept.

    Two pieces is the only split short sequences can take; seven leaves a long
    remainder on the last piece. ``chunks = 1`` is the whole-sequence call itself,
    so it is not a case.
    """
    x, weight, bias, state = make_call(bsz, seqlen, channels, width)
    whole = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    # Uneven splits on purpose: an even split hides an off-by-one in the window
    # that is carried across a boundary.
    sizes = [seqlen // chunks] * chunks
    sizes[-1] += seqlen - sum(sizes)
    sizes = [s for s in sizes if s > 0]
    carry = state
    pieces: list[Tensor] = []
    for piece in torch.split(x, sizes, dim=1):
        out = causal_conv1d_update_ref(piece, weight, bias, initial_state=carry)
        pieces.append(out.y)
        carry = out.state
    joined = torch.cat(pieces, dim=1)
    assert max_err(joined, whole.y) < 1e-14
    assert carry is not None
    assert_bitwise(carry, whole.state)


@pytest.mark.parametrize(("bsz", "seqlen", "channels", "width"), SHAPES)
def test_token_at_a_time_decode_reproduces_the_whole_sequence(
    bsz: int, seqlen: int, channels: int, width: int
) -> None:
    x, weight, bias, state = make_call(bsz, seqlen, channels, width)
    whole = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    carry = state
    steps: list[Tensor] = []
    for t in range(seqlen):
        out = causal_conv1d_update_ref(
            x[:, t : t + 1], weight, bias, initial_state=carry
        )
        steps.append(out.y)
        carry = out.state
    assert max_err(torch.cat(steps, dim=1), whole.y) < 1e-14


@pytest.mark.parametrize(
    ("bsz", "seqlen", "channels", "width"),
    [
        pytest.param(2, 6, 3, 4, id="base"),
        pytest.param(1, 1, 1, 4, id="single-token"),
        pytest.param(2, 3, 2, 8, id="straddle"),
        pytest.param(2, 4, 2, 1, id="pointwise"),
    ],
)
@pytest.mark.parametrize("activation", [True, False])
def test_gradcheck(
    bsz: int, seqlen: int, channels: int, width: int, activation: bool
) -> None:
    x, weight, bias, state = make_call(
        bsz, seqlen, channels, width, state=width > 1, requires_grad=True
    )
    leaves = [t for t in (x, weight, bias, state) if t is not None]

    def run(*args: Tensor) -> tuple[Tensor, ...]:
        xa, wa = args[0], args[1]
        ba = args[2]
        sa = args[3] if len(args) > 3 else None
        out = causal_conv1d_update_ref(
            xa, wa, ba, activation=activation, initial_state=sa
        )
        # A zero-numel output carries no gradient information and gradcheck
        # rejects one, so W = 1 is checked on y alone.
        return tuple(t for t in (out.y, out.state) if t.numel())

    assert torch.autograd.gradcheck(run, tuple(leaves), fast_mode=False)


@pytest.mark.parametrize(
    ("with_dy", "with_dstate"),
    [
        pytest.param(True, True, id="both"),
        pytest.param(True, False, id="dy-only"),
        pytest.param(False, True, id="dstate-only"),
        pytest.param(False, False, id="neither"),
    ],
)
def test_reference_backward_matches_autograd(with_dy: bool, with_dstate: bool) -> None:
    bsz, seqlen, channels, width = 2, 9, 3, 4
    x, weight, bias, state = make_call(bsz, seqlen, channels, width, requires_grad=True)
    gen = torch.Generator().manual_seed(7)
    dy = (
        torch.randn(bsz, seqlen, channels, generator=gen, dtype=torch.float64)
        if with_dy
        else None
    )
    dstate = (
        torch.randn(
            *conv_state_shape(bsz, width, channels), generator=gen, dtype=torch.float64
        )
        if with_dstate
        else None
    )
    out = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    total = torch.zeros((), dtype=torch.float64)
    if dy is not None:
        total = total + (out.y * dy).sum()
    if dstate is not None:
        total = total + (out.state * dstate).sum()
    leaves = [t for t in (x, weight, bias, state) if t is not None]
    if with_dy or with_dstate:
        # The returned window is a slice of the padded input, so with the window
        # cotangent alone the taps and the bias reach no output at all. The stub
        # types the result as Tensor, but allow_unused=True yields None for a leaf
        # that reaches no output, which is the case under test.
        found = cast(
            "tuple[Tensor | None, ...]",
            torch.autograd.grad(total, leaves, allow_unused=True),
        )
        want = tuple(
            torch.zeros_like(leaf) if grad is None else grad
            for leaf, grad in zip(leaves, found)
        )
    else:
        want = tuple(torch.zeros_like(t) for t in leaves)

    got = causal_conv1d_bwd_ref(dy, dstate, x, weight, bias, initial_state=state)
    for have, expect in zip((got.dx, got.dweight, got.dbias, got.dinitial_state), want):
        assert have is not None
        assert max_err(have, expect) < 1e-13


def test_backward_fills_an_unreachable_gradient_with_zeros() -> None:
    # At W = 1 the incoming window is empty and reaches no output, so autograd
    # reports nothing for it and the field is a zero tensor rather than None.
    bsz, seqlen, channels, width = 2, 4, 3, 1
    x, weight, bias, state = make_call(bsz, seqlen, channels, width)
    dy = torch.ones(bsz, seqlen, channels, dtype=torch.float64)
    got = causal_conv1d_bwd_ref(dy, None, x, weight, bias, initial_state=state)
    assert got.dinitial_state is not None
    assert got.dinitial_state.shape == (bsz, 0, channels)


def test_absent_optional_inputs_get_no_gradient() -> None:
    x, weight, _, _ = make_call(2, 6, 3, 4, bias=False, state=False)
    dy = torch.ones_like(x)
    got = causal_conv1d_bwd_ref(dy, None, x, weight)
    assert got.dbias is None
    assert got.dinitial_state is None


@pytest.mark.parametrize(
    ("dtype", "bound"),
    [
        # One rounding of the output at the storage width. bfloat16 keeps 7
        # mantissa bits and float16 keeps 10, so half an ulp is at most 2^-8 and
        # 2^-11 of the largest output element. float32 is dominated instead by
        # accumulating a W-term sum at 2^-24 per add.
        pytest.param(torch.bfloat16, 5e-3, id="bf16"),
        pytest.param(torch.float16, 1e-3, id="fp16"),
        pytest.param(torch.float32, 2e-6, id="fp32"),
    ],
)
def test_low_precision_forward_tracks_the_float64_oracle(
    dtype: torch.dtype, bound: float, device: torch.device
) -> None:
    bsz, seqlen, channels, width = 2, 40, 8, 4
    x, weight, bias, state = make_call(
        bsz, seqlen, channels, width, dtype=dtype, device=device
    )
    want = oracle(x, weight, bias, activation=True, initial_state=state)
    got = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    assert got.y.dtype == dtype
    assert got.state.dtype == dtype
    assert_max_rel(got.y, want.y, bound, f"conv/ref/y/{dtype}")


def test_conv_state_shape() -> None:
    assert conv_state_shape(3, 4, 8) == (3, 3, 8)
    assert conv_state_shape(3, 1, 8) == (3, 0, 8)


def test_check_operands_returns_the_extents() -> None:
    x, weight, bias, state = make_call(2, 40, 8, 4)
    assert check_operands(x, weight, bias, state) == ConvDims(
        batch=2, seqlen=40, channels=8, width=4
    )


def test_conv_step_is_named() -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    out = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    assert isinstance(out, ConvStep)
    assert out.y is out[0]
    assert out.state is out[1]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda t: t.reshape(2, -1), "x must be", id="x-rank"),
        pytest.param(lambda t: t[:, :0], "at least one element", id="x-empty"),
    ],
)
def test_x_is_rejected(mutate: Callable[[Tensor], Tensor], message: str) -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match=message):
        causal_conv1d_ref(mutate(x).contiguous(), weight, bias, initial_state=state)


def test_weight_rank_is_rejected() -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match="weight must be"):
        causal_conv1d_ref(x, weight.reshape(-1), bias, initial_state=state)


def test_weight_channel_count_is_rejected() -> None:
    x, weight, _, _ = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match="weight must be"):
        causal_conv1d_ref(x, weight[:2])


def test_zero_width_is_rejected() -> None:
    x, _, _, _ = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match="width must be positive"):
        causal_conv1d_ref(x, x.new_zeros(3, 0))


def test_bias_shape_is_rejected() -> None:
    x, weight, bias, _ = make_call(2, 6, 3, 4)
    assert bias is not None
    with pytest.raises(ValueError, match="bias must be"):
        causal_conv1d_ref(x, weight, bias[:2])


def test_initial_state_shape_is_rejected() -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    assert state is not None
    with pytest.raises(ValueError, match="initial_state must be"):
        causal_conv1d_ref(x, weight, bias, initial_state=state[:, :1])


@pytest.mark.parametrize("name", ["x", "weight", "bias", "initial_state"])
def test_unsupported_dtype_is_rejected(name: str) -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    assert bias is not None
    assert state is not None
    operands = {"x": x, "weight": weight, "bias": bias, "initial_state": state}
    operands[name] = operands[name].to(torch.int64)
    with pytest.raises(TypeError, match=f"{name} has dtype"):
        causal_conv1d_ref(
            operands["x"],
            operands["weight"],
            operands["bias"],
            initial_state=operands["initial_state"],
        )


def test_dy_shape_is_rejected() -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match="dy must be"):
        causal_conv1d_bwd_ref(
            torch.ones(2, 5, 3, dtype=torch.float64),
            None,
            x,
            weight,
            bias,
            initial_state=state,
        )


def test_dfinal_state_shape_is_rejected() -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    with pytest.raises(ValueError, match="dfinal_state must be"):
        causal_conv1d_bwd_ref(
            None,
            torch.ones(2, 2, 3, dtype=torch.float64),
            x,
            weight,
            bias,
            initial_state=state,
        )


@pytest.mark.parametrize("name", ["dy", "dfinal_state"])
def test_cotangent_dtype_is_rejected(name: str) -> None:
    x, weight, bias, state = make_call(2, 6, 3, 4)
    bad = torch.ones(2, 6, 3, dtype=torch.int64)
    if name == "dfinal_state":
        bad = torch.ones(2, 3, 3, dtype=torch.int64)
    args = (bad, None) if name == "dy" else (None, bad)
    with pytest.raises(TypeError, match=f"{name} has dtype"):
        causal_conv1d_bwd_ref(*args, x, weight, bias, initial_state=state)


def test_reference_backend_is_registered() -> None:
    assert "reference" in names()
    backend = get("reference")
    assert backend.priority == 0
    assert backend.device_types == ("cpu", "cuda")


def test_registering_a_name_twice_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register(get("reference"))


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get("nope")


def test_backend_that_does_not_support_the_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="supports"):
        resolve("reference", "meta")


def test_device_with_no_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="no backend supports"):
        resolve(None, "meta")


def test_resolve_prefers_the_highest_priority_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import slinoss.ops.conv.backends as backends

    reference = get("reference")
    faster = Backend(
        name="faster",
        forward=reference.forward,
        backward=reference.backward,
        device_types=("cpu",),
        priority=1,
    )
    monkeypatch.setitem(backends._REGISTRY, "faster", faster)
    assert resolve(None, "cpu").name == "faster"
    # An explicit name overrides priority.
    assert resolve("reference", "cpu").name == "reference"


def test_public_operator_matches_the_reference() -> None:
    x, weight, bias, state = make_call(2, 40, 8, 4)
    got = causal_conv1d(x, weight, bias, initial_state=state)
    want = causal_conv1d_update_ref(x, weight, bias, initial_state=state)
    assert_bitwise(got.y, want.y)
    assert_bitwise(got.state, want.state)


def _cloned_leaves(
    args: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """The same operands as an independent set of differentiable leaves."""

    def leaf(t: Tensor) -> Tensor:
        return t.detach().clone().requires_grad_(True)

    x, weight, bias, state = args
    return (
        leaf(x),
        leaf(weight),
        None if bias is None else leaf(bias),
        None if state is None else leaf(state),
    )


@pytest.mark.parametrize("activation", [True, False])
def test_public_operator_backward_matches_the_reference(activation: bool) -> None:
    bsz, seqlen, channels, width = 2, 9, 3, 4
    args = make_call(bsz, seqlen, channels, width, requires_grad=True)
    x, weight, bias, state = args
    other = _cloned_leaves(args)
    gen = torch.Generator().manual_seed(11)
    dy = torch.randn(bsz, seqlen, channels, generator=gen, dtype=torch.float64)
    dstate = torch.randn(
        *conv_state_shape(bsz, width, channels), generator=gen, dtype=torch.float64
    )

    out = causal_conv1d(x, weight, bias, activation=activation, initial_state=state)
    ((out.y * dy).sum() + (out.state * dstate).sum()).backward()

    ref = causal_conv1d_update_ref(
        other[0], other[1], other[2], activation=activation, initial_state=other[3]
    )
    ((ref.y * dy).sum() + (ref.state * dstate).sum()).backward()

    for have, expect in zip(args, other):
        assert have is not None and expect is not None
        assert have.grad is not None and expect.grad is not None
        assert max_err(have.grad, expect.grad) < 1e-13


def test_public_operator_needs_no_state_cotangent() -> None:
    # set_materialize_grads(False) means the unused window arrives as None in the
    # backward instead of as a zero tensor.
    x, weight, bias, state = make_call(2, 9, 3, 4, requires_grad=True)
    out = causal_conv1d(x, weight, bias, initial_state=state)
    out.y.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None


def test_unbuilt_extension_reports_the_build_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_C, "_MODULE", None)
    monkeypatch.setattr(_C, "_ERROR", ImportError("no such module"))
    assert not _C.is_available()
    with pytest.raises(RuntimeError, match=_C.BUILD_COMMAND):
        _C.extension()


def test_setup_names_the_same_extension() -> None:
    # Read rather than import: importing setup.py runs setuptools. A drift here
    # builds the module under a name nothing imports.
    source = (Path(__file__).resolve().parents[1] / "setup.py").read_text()
    assert f'MODULE = "{_C.EXTENSION}"' in source
