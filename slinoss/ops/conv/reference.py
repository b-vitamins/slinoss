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

``d_head`` widens the output layout only. At ``d_head = P`` the output is
head-major, ``(B, D//P, T, P)``, with channel ``d = h*P + p`` landing at
``(b,h,t,p)``; the map, the taps, the state, and every input shape are unchanged.
The scan reads ``U`` head-major, so this is the one consumer's layout produced in
the store rather than in a repack afterwards. Here the head-major result is
computed by reshaping the operands and contracting into the head-major shape, so
it is the definition at that shape and not a permuted copy of the token-major one.

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
from slinoss.config import HEAD_MULTIPLE

__all__ = [
    "ConvDims",
    "ConvGrads",
    "ConvStep",
    "causal_conv1d_bwd_ref",
    "causal_conv1d_ref",
    "causal_conv1d_update_ref",
    "check_cotangents",
    "check_dx_out",
    "check_operands",
    "conv_output_shape",
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


def conv_output_shape(
    bsz: int, seqlen: int, channels: int, d_head: int | None
) -> tuple[int, ...]:
    """Shape of ``y``.

    Args:
        bsz: Batch ``B``.
        seqlen: Tokens ``T``.
        channels: Channels ``D``.
        d_head: Rows per head ``P``, for a head-major output, or None for a
            token-major one.

    Returns:
        ``(B,T,D)`` when ``d_head`` is None, else ``(B, D//P, T, P)``.

    Raises:
        ValueError: If ``P`` is not a positive multiple of
            :data:`slinoss.config.HEAD_MULTIPLE`, or does not divide ``D``. The
            multiple is the scan's rule, not the conv's: a ``P`` the scan cannot
            take is not a ``P`` worth writing.
    """
    if d_head is None:
        return (bsz, seqlen, channels)
    if d_head < 1 or d_head % HEAD_MULTIPLE != 0:
        raise ValueError(
            f"d_head must be a positive multiple of {HEAD_MULTIPLE}, got {d_head}"
        )
    if channels % d_head != 0:
        raise ValueError(f"d_head {d_head} must divide D={channels}")
    return (bsz, channels // d_head, seqlen, d_head)


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
    d_head: int | None,
) -> Tensor:
    """Slide the tap bank over ``xp`` and apply the epilogue.

    ``unfold`` is a view, so the window axis costs no copy of ``xp``. So is the
    head-major reshape of the operands: splitting the channel axis needs only its
    unit stride, and the transpose that follows moves no element. The contraction
    therefore writes the head-major output directly rather than permuting a
    token-major one.

    Args:
        padded: ``xp``, shape ``(B,T+W-1,D)``, accumulation dtype.
        weight: Taps, shape ``(D,W)``.
        bias: Per-channel bias, shape ``(D,)``, or None.
        width: Tap count ``W``.
        activation: Apply SiLU.
        out_dtype: Dtype of the returned tensor.
        d_head: Rows per head ``P``, or None for a token-major output.

    Returns:
        Shape ``(B,T,D)``, or ``(B, D//P, T, P)`` at ``d_head = P``, in
        ``out_dtype``.
    """
    dtype = padded.dtype
    taps = weight.to(dtype)
    shifted = None if bias is None else bias.to(dtype)
    if d_head is not None:
        # (B,H,T+W-1,P), (H,1,P,W), (H,1,P). Both reshapes are views: splitting the
        # channel axis needs only its unit stride and the transpose moves no
        # element. The token axis is -2 in either layout and the taps broadcast
        # against the trailing modes in either, so the contraction below is one
        # expression at both settings.
        padded = padded.unflatten(-1, (-1, d_head)).transpose(1, 2)
        taps = taps.unflatten(0, (-1, d_head))[:, None]
        if shifted is not None:
            shifted = shifted.unflatten(0, (-1, d_head))[:, None]
    # Component W of the window is xp[t + W], i.e. tap index k.
    total = (padded.unfold(-2, width, 1) * taps).sum(-1)
    if shifted is not None:
        total = total + shifted
    return (silu(total) if activation else total).to(out_dtype)


def causal_conv1d_ref(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
    d_head: int | None = None,
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
        d_head: Rows per head ``P``, which makes the output head-major, or None
            for the token-major output. Changes the output layout and nothing
            else.

    Returns:
        Shape ``(B,T,D)``, or ``(B, D//P, T, P)`` at ``d_head = P``, dtype of
        ``x``.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, a non-positive
            ``W``, or a ``d_head`` outside :func:`conv_output_shape`.
        TypeError: On an unsupported dtype.
    """
    dims = check_operands(x, weight, bias, initial_state)
    conv_output_shape(dims.batch, dims.seqlen, dims.channels, d_head)
    operands = [x, weight]
    if bias is not None:
        operands.append(bias)
    if initial_state is not None:
        operands.append(initial_state)
    dtype = pinned_dtype(*operands)
    with autocast_disabled(x.device.type):
        padded = _padded(x, initial_state, dims.width, dtype)
        return _contract(
            padded,
            weight,
            bias,
            dims.width,
            activation=activation,
            out_dtype=x.dtype,
            d_head=d_head,
        )


class ConvStep(NamedTuple):
    """Result of a streaming step.

    Attributes:
        y: Output, shape ``(B,T,D)``, or ``(B, D//P, T, P)`` when the call named
            ``d_head = P``, dtype of ``x``. The head-major form is the layout the
            scan reads ``U`` in: channel ``d = h*P + p`` is at ``(b,h,t,p)``.
        state: The ``W-1`` timesteps that follow ``x``, shape ``(B,W-1,D)``,
            dtype of ``x``. Feeds the next call's ``initial_state``. Empty at
            ``W = 1``. Token-major at both output layouts, because it is a window
            of ``x``.
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
    d_head: int | None = None,
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
        d_head: Rows per head ``P``, which makes ``y`` head-major, or None for the
            token-major ``y``. The returned window is token-major either way, so
            the streaming identity holds at both settings.

    Returns:
        A :class:`ConvStep`.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, a non-positive
            ``W``, or a ``d_head`` outside :func:`conv_output_shape`.
        TypeError: On an unsupported dtype.
    """
    dims = check_operands(x, weight, bias, initial_state)
    conv_output_shape(dims.batch, dims.seqlen, dims.channels, d_head)
    width = dims.width
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
            d_head=d_head,
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
) -> int | None:
    """Validate the cotangents of a :class:`ConvStep` and read off ``dy``'s layout.

    The layout is inferred from ``dy``'s rank rather than carried as a saved flag.
    ``dy`` is the only operand the output layout reaches -- ``dx`` is token-major
    because ``x`` is -- and its rank already states that layout, so a flag would be
    a second source of truth that can disagree with the tensor in hand. It would
    also protect nothing: the flag would be available only on the autograd path,
    where the cotangent's shape is already the output's, and not to a direct call
    on a backend's backward, which is public too. Rank plus the
    :func:`conv_output_shape` rules pin ``P`` and ``H`` exactly, so an inconsistent
    rank-4 cotangent is rejected rather than read the wrong way.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)`` or ``(B, D//P, T, P)``, or None.
        dfinal_state: Cotangent of the returned window, shape ``(B,W-1,D)``, or
            None.
        dims: Extents of the forward call.

    Returns:
        The ``d_head`` the forward was called with: ``P`` for a rank-4 ``dy``, None
        for a rank-3 one and for an absent one, which leaves the layout unread.

    Raises:
        ValueError: On a shape mismatch.
        TypeError: On an unsupported dtype.
    """
    d_head = None if dy is None or dy.ndim != 4 else int(dy.shape[-1])
    if dy is not None:
        want_y = conv_output_shape(dims.batch, dims.seqlen, dims.channels, d_head)
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
    return d_head


def check_dx_out(dx: Tensor, x: Tensor, dims: ConvDims) -> None:
    """Validate a caller-supplied ``dx`` destination. Shared by every backend.

    Shape, dtype, and device only. The layout rule belongs to the backend that
    stores through the buffer: the kernels take a row pitch and hold ``dx`` to the
    pitched contract, and the reference stores through ``copy_`` and holds it to
    nothing. Checked before that rule, so a buffer of the wrong extent or dtype is
    reported as such rather than as a stride.

    Args:
        dx: Destination for the activation gradient.
        x: The forward's activations, whose dtype and device ``dx`` shares.
        dims: Extents of the call.

    Raises:
        ValueError: If ``dx`` is not ``(B,T,D)``, or its dtype or device differs
            from ``x``'s. A cross-device destination would be a silent staging copy
            in the reference and an unaddressable store in a kernel.
    """
    want = (dims.batch, dims.seqlen, dims.channels)
    if tuple(dx.shape) != want:
        raise ValueError(f"dx must be {want}, got {tuple(dx.shape)}")
    if dx.dtype != x.dtype:
        raise ValueError(f"dx must be {x.dtype}, got {dx.dtype}")
    if dx.device != x.device:
        raise ValueError(f"dx must be on {x.device}, got {dx.device}")


class ConvGrads(NamedTuple):
    """Cotangents of the operator inputs.

    Every field is contiguous except a ``dx`` the caller supplied, which carries the
    layout it arrived with.

    A field is ``None`` exactly when the corresponding input was ``None``, which
    is what :class:`torch.autograd.Function` expects for an absent optional
    argument.

    Attributes:
        dx: Shape ``(B,T,D)``, dtype of ``x``. Token-major at both output layouts,
            because ``x`` is. The buffer the call named, when it named one.
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
    dx: Tensor | None = None,
) -> ConvGrads:
    """Pullback of :func:`causal_conv1d_update_ref`, by autograd through it.

    Autograd through the forward rather than a written-out VJP. A hand-derived
    pullback shares its algebra with the forward it was derived from, so an
    algebra error passes silently; differentiating the forward itself cannot
    disagree with the forward. In float64 this is the gradient authority the
    kernels are measured against.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)`` or ``(B, D//P, T, P)``, or None.
            Its rank is how the forward's output layout is recovered; see
            :func:`check_cotangents`.
        dfinal_state: Cotangent of the returned window, shape ``(B,W-1,D)``, or
            None.
        x: The forward's activations, shape ``(B,T,D)``.
        weight: The forward's taps, shape ``(D,W)``.
        bias: The forward's bias, shape ``(D,)``, or None.
        activation: The forward's activation flag.
        initial_state: The forward's incoming window, shape ``(B,W-1,D)``, or
            None.
        dx: Destination for the activation gradient, shape ``(B,T,D)``, dtype and
            device of ``x``, or None to allocate it. Written in full, so its
            incoming contents are unread and nothing is accumulated into it. The
            same parameter the native backend takes, so the two backends stay one
            signature.

    Returns:
        A :class:`ConvGrads`.

    Raises:
        ValueError: On a rank or shape mismatch, an empty ``x``, a non-positive
            ``W``, or a ``dx`` outside :func:`check_dx_out`.
        TypeError: On an unsupported dtype.
    """
    dims = check_operands(x, weight, bias, initial_state)
    d_head = check_cotangents(dy, dfinal_state, dims)
    if dx is not None:
        check_dx_out(dx, x, dims)

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
            xl, wl, bl, activation=activation, initial_state=sl, d_head=d_head
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
    grad_x = filled[0]
    if dx is not None:
        # A store into the caller's buffer, not a rebind: the caller reads the band
        # it handed in, so returning the allocation would leave that band unwritten.
        dx.copy_(grad_x)
        grad_x = dx
    return ConvGrads(
        dx=grad_x,
        dweight=filled[1],
        dbias=filled[2] if bl is not None else None,
        dinitial_state=filled[-1] if sl is not None else None,
    )
