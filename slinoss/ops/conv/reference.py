"""Causal depthwise conv1d. Pure-PyTorch reference.

One tap bank per channel, no mixing across channels:

    s[b,t,d] = bias[d] + sum_{k=0}^{W-1} weight[d,k] * xp[b, t - (W-1) + k, d]
    y[b,t,d] = silu(s[b,t,d])   when the activation is on, else s[b,t,d]

``xp`` is ``x`` prefixed by the ``W-1`` timesteps that precede it. Those come
from ``initial_state`` when a stream is being continued and are zero otherwise,
so the operator is causal at every ``t`` with no future leak.

The streaming form returns the ``W-1`` trailing timesteps of ``xp`` as the next
state, which makes a decode step the same call at ``T = 1``. The state is the
raw input window, not a partial sum: taps are per token, so nothing about the
window can be folded away.

Layout. Time-major and channels-last, ``(B,T,D)``, contiguous. The taps are
``(D,W)`` so a channel's whole bank is contiguous. Nothing here transposes.

Precision. The tap contraction accumulates in float32, or float64 when any
operand is float64, and the output carries the dtype of ``x``. Accumulating a
``W``-term sum at the input width would put the reference and the kernel on
different footings.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import silu

from slinoss._precision import autocast_disabled, check_supported, pinned_dtype

__all__ = [
    "ConvDims",
    "ConvGrads",
    "ConvStep",
    "causal_conv1d_bwd_ref",
    "causal_conv1d_ref",
    "causal_conv1d_update_ref",
    "check_cotangents",
    "check_operands",
    "conv_state_shape",
]


def conv_state_shape(bsz: int, width: int, channels: int) -> tuple[int, int, int]:
    """Shape of the streaming state.

    Args:
        bsz: Batch.
        width: Tap count ``W``.
        channels: Channels ``D``.

    Returns:
        ``(B, W-1, D)``. Degenerate at ``W = 1``, where a causal conv1d is
        pointwise and carries nothing between steps.
    """
    return (bsz, width - 1, channels)


class ConvDims(NamedTuple):
    """Extents of a call.

    Attributes:
        batch: ``B``.
        seqlen: ``T``.
        channels: ``D``.
        width: Tap count ``W``.
    """

    batch: int
    seqlen: int
    channels: int
    width: int


def check_operands(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    initial_state: Tensor | None,
) -> ConvDims:
    """Validate a call. Shared by every backend so the error does not vary.

    Args:
        x: Activations, shape ``(B,T,D)``.
        weight: Taps, shape ``(D,W)``.
        bias: Per-channel bias, shape ``(D,)``, or None.
        initial_state: Previous window, shape ``(B,W-1,D)``, or None.

    Returns:
        A :class:`ConvDims`.

    Raises:
        ValueError: On a rank mismatch, an empty ``x``, a tap bank that is not
            ``(D,W)``, a non-positive ``W``, a bias that is not ``(D,)``, or a
            state that is not ``(B,W-1,D)``.
        TypeError: On an unsupported dtype.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (B,T,D), got {tuple(x.shape)}")
    bsz, seqlen, channels = (int(d) for d in x.shape)
    if bsz * seqlen * channels == 0:
        raise ValueError(f"x must hold at least one element, got {tuple(x.shape)}")
    if weight.ndim != 2 or int(weight.shape[0]) != channels:
        raise ValueError(f"weight must be ({channels},W), got {tuple(weight.shape)}")
    width = int(weight.shape[1])
    if width < 1:
        raise ValueError(f"width must be positive, got {width}")
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must be ({channels},), got {tuple(bias.shape)}")
    want = conv_state_shape(bsz, width, channels)
    if initial_state is not None and tuple(initial_state.shape) != want:
        raise ValueError(
            f"initial_state must be {want}, got {tuple(initial_state.shape)}"
        )
    check_supported(x, "x")
    check_supported(weight, "weight")
    if bias is not None:
        check_supported(bias, "bias")
    if initial_state is not None:
        check_supported(initial_state, "initial_state")
    return ConvDims(batch=bsz, seqlen=seqlen, channels=channels, width=width)


def _padded(
    x: Tensor,
    initial_state: Tensor | None,
    width: int,
    dtype: torch.dtype,
) -> Tensor:
    """``xp``: ``x`` prefixed by the ``W-1`` timesteps before it.

    Args:
        x: Activations, shape ``(B,T,D)``.
        initial_state: Previous window, shape ``(B,W-1,D)``, or None for zeros.
        width: Tap count ``W``.
        dtype: Accumulation dtype.

    Returns:
        Shape ``(B, T+W-1, D)`` in ``dtype``. Returned unchanged at ``W = 1``,
        where the prefix is empty.
    """
    wide = x.to(dtype)
    if width == 1:
        return wide
    if initial_state is None:
        prefix = wide.new_zeros(x.shape[0], width - 1, x.shape[2])
    else:
        prefix = initial_state.to(dtype)
    return torch.cat([prefix, wide], dim=1)


def _contract(
    padded: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    width: int,
    *,
    activation: bool,
    out_dtype: torch.dtype,
) -> Tensor:
    """Slide the tap bank over ``xp`` and apply the epilogue.

    ``unfold`` is a view, so the window axis costs no copy of ``xp``.

    Args:
        padded: ``xp``, shape ``(B,T+W-1,D)``, accumulation dtype.
        weight: Taps, shape ``(D,W)``.
        bias: Per-channel bias, shape ``(D,)``, or None.
        width: Tap count ``W``.
        activation: Apply SiLU.
        out_dtype: Dtype of the returned tensor.

    Returns:
        Shape ``(B,T,D)`` in ``out_dtype``.
    """
    dtype = padded.dtype
    # (B,T,D,W): component W of the window is xp[t + W], i.e. tap index k.
    windows = padded.unfold(1, width, 1)
    total = (windows * weight.to(dtype)).sum(-1)
    if bias is not None:
        total = total + bias.to(dtype)
    return (silu(total) if activation else total).to(out_dtype)


def causal_conv1d_ref(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
) -> Tensor:
    """Causal depthwise conv1d over a whole sequence.

    Args:
        x: Activations, shape ``(B,T,D)``, bf16/fp16/fp32/fp64.
        weight: Taps, shape ``(D,W)``. Tap ``k`` multiplies lag ``W-1-k``, so
            tap ``W-1`` is the current token.
        bias: Per-channel bias, shape ``(D,)``, or None.
        activation: Apply SiLU to the tap sum.
        initial_state: The ``W-1`` timesteps before ``x``, shape ``(B,W-1,D)``.
            Zero if omitted.

    Returns:
        Shape ``(B,T,D)``, dtype of ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, or a
            non-positive ``W``.
        TypeError: On an unsupported dtype.
    """
    width = check_operands(x, weight, bias, initial_state).width
    operands = [x, weight]
    if bias is not None:
        operands.append(bias)
    if initial_state is not None:
        operands.append(initial_state)
    dtype = pinned_dtype(*operands)
    with autocast_disabled(x.device.type):
        padded = _padded(x, initial_state, width, dtype)
        return _contract(
            padded,
            weight,
            bias,
            width,
            activation=activation,
            out_dtype=x.dtype,
        )


class ConvStep(NamedTuple):
    """Result of a streaming step.

    Attributes:
        y: Output, shape ``(B,T,D)``, dtype of ``x``.
        state: The ``W-1`` timesteps that follow ``x``, shape ``(B,W-1,D)``,
            dtype of ``x``. Feeds the next call's ``initial_state``. Empty at
            ``W = 1``.
    """

    y: Tensor
    state: Tensor


def causal_conv1d_update_ref(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
) -> ConvStep:
    """Causal depthwise conv1d that also returns the next window.

    Same map as :func:`causal_conv1d_ref`; splitting a sequence into consecutive
    calls threaded through ``state`` reproduces the whole-sequence result
    exactly. ``T = 1`` is the decode step.

    Args:
        x: Activations, shape ``(B,T,D)``.
        weight: Taps, shape ``(D,W)``.
        bias: Per-channel bias, shape ``(D,)``, or None.
        activation: Apply SiLU to the tap sum.
        initial_state: The ``W-1`` timesteps before ``x``, shape ``(B,W-1,D)``.
            Zero if omitted.

    Returns:
        A :class:`ConvStep`.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, or a
            non-positive ``W``.
        TypeError: On an unsupported dtype.
    """
    width = check_operands(x, weight, bias, initial_state).width
    operands = [x, weight]
    if bias is not None:
        operands.append(bias)
    if initial_state is not None:
        operands.append(initial_state)
    dtype = pinned_dtype(*operands)
    with autocast_disabled(x.device.type):
        padded = _padded(x, initial_state, width, dtype)
        y = _contract(
            padded,
            weight,
            bias,
            width,
            activation=activation,
            out_dtype=x.dtype,
        )
    # The trailing window of xp, which is the trailing window of x whenever
    # T >= W-1 and otherwise straddles the incoming state. Slicing xp covers
    # both without a branch.
    tail = padded.shape[1] - (width - 1)
    return ConvStep(y=y, state=padded[:, tail:, :].to(x.dtype).contiguous())


def check_cotangents(
    dy: Tensor | None,
    dfinal_state: Tensor | None,
    dims: ConvDims,
) -> None:
    """Validate the cotangents of a :class:`ConvStep`.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)``, or None.
        dfinal_state: Cotangent of the returned window, shape ``(B,W-1,D)``, or
            None.
        dims: Extents of the forward call.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    want_y = (dims.batch, dims.seqlen, dims.channels)
    if dy is not None:
        if tuple(dy.shape) != want_y:
            raise ValueError(f"dy must be {want_y}, got {tuple(dy.shape)}")
        check_supported(dy, "dy")
    want_state = conv_state_shape(dims.batch, dims.width, dims.channels)
    if dfinal_state is not None:
        if tuple(dfinal_state.shape) != want_state:
            raise ValueError(
                f"dfinal_state must be {want_state}, got {tuple(dfinal_state.shape)}"
            )
        check_supported(dfinal_state, "dfinal_state")


class ConvGrads(NamedTuple):
    """Cotangents of the operator inputs. Every field is contiguous.

    A field is ``None`` exactly when the corresponding input was ``None``, which
    is what :class:`torch.autograd.Function` expects for an absent optional
    argument.

    Attributes:
        dx: Shape ``(B,T,D)``, dtype of ``x``.
        dweight: Shape ``(D,W)``, dtype of ``weight``.
        dbias: Shape ``(D,)`` or ``None``.
        dinitial_state: Shape ``(B,W-1,D)`` or ``None``.
    """

    dx: Tensor
    dweight: Tensor
    dbias: Tensor | None
    dinitial_state: Tensor | None


def causal_conv1d_bwd_ref(
    dy: Tensor | None,
    dfinal_state: Tensor | None,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    /,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
) -> ConvGrads:
    """Pullback of :func:`causal_conv1d_update_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently; differentiating the forward itself cannot
    disagree with the forward. In float64 this is the gradient authority the
    kernels are measured against.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)``, or None.
        dfinal_state: Cotangent of the returned window, shape ``(B,W-1,D)``, or
            None.
        x: The forward's activations, shape ``(B,T,D)``.
        weight: The forward's taps, shape ``(D,W)``.
        bias: The forward's bias, shape ``(D,)``, or None.
        activation: The forward's activation flag.
        initial_state: The forward's incoming window, shape ``(B,W-1,D)``, or
            None.

    Returns:
        A :class:`ConvGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, or a
            non-positive ``W``.
        TypeError: On an unsupported dtype.
    """
    dims = check_operands(x, weight, bias, initial_state)
    check_cotangents(dy, dfinal_state, dims)

    xl = x.detach().requires_grad_(True)
    wl = weight.detach().requires_grad_(True)
    bl = None if bias is None else bias.detach().requires_grad_(True)
    sl = None if initial_state is None else initial_state.detach().requires_grad_(True)
    leaves: list[Tensor] = [xl, wl]
    if bl is not None:
        leaves.append(bl)
    if sl is not None:
        leaves.append(sl)

    with torch.enable_grad():
        out = causal_conv1d_update_ref(
            xl, wl, bl, activation=activation, initial_state=sl
        )
    outputs: list[Tensor] = []
    cotangents: list[Tensor] = []
    if dy is not None:
        outputs.append(out.y)
        cotangents.append(dy)
    if dfinal_state is not None:
        outputs.append(out.state)
        cotangents.append(dfinal_state)

    if outputs:
        # allow_unused because the incoming state reaches no output at W = 1 and
        # the bias reaches none when both cotangents are absent.
        found = torch.autograd.grad(
            outputs, leaves, cotangents, allow_unused=True, retain_graph=False
        )
    else:
        found = tuple(None for _ in leaves)
    filled = [
        torch.zeros_like(leaf) if grad is None else grad.contiguous()
        for leaf, grad in zip(leaves, found)
    ]
    return ConvGrads(
        dx=filled[0],
        dweight=filled[1],
        dbias=filled[2] if bl is not None else None,
        dinitial_state=filled[-1] if sl is not None else None,
    )
