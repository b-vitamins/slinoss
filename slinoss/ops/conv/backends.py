"""Backend registry for the causal depthwise conv1d.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is the conv's own: the two call signatures, the
native operand contract, and which implementations exist.

The native backend registers only when the compiled extension imported. An unbuilt
tree therefore resolves to the reference on every device instead of resolving to a
path that cannot run.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from slinoss import _C
from slinoss._guard import check_layout
from slinoss._precision import KERNEL_DTYPES, SUPPORTED_DTYPES
from slinoss._registry import Backend, Registry
from slinoss.ops.conv.reference import (
    ConvDims,
    ConvGrads,
    ConvStep,
    causal_conv1d_bwd_ref,
    causal_conv1d_update_ref,
    check_cotangents,
    check_operands,
    conv_output_shape,
    conv_state_shape,
)

__all__ = [
    "Backend",
    "ConvBackend",
    "ConvBackward",
    "ConvForward",
    "causal_conv1d_bwd_native",
    "causal_conv1d_fwd_native",
    "get",
    "names",
    "register",
    "resolve",
]

REFERENCE = "reference"
NATIVE = "native"


class ConvForward(Protocol):
    """Forward signature every backend implements."""

    def __call__(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
        *,
        activation: bool = True,
        initial_state: Tensor | None = None,
        d_head: int | None = None,
    ) -> ConvStep: ...


class ConvBackward(Protocol):
    """Backward signature every backend implements.

    No ``d_head``: the output layout reaches the backward only through ``dy``, and
    :func:`slinoss.ops.conv.reference.check_cotangents` reads it off there.
    """

    def __call__(
        self,
        dy: Tensor | None,
        dfinal_state: Tensor | None,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
        *,
        activation: bool = True,
        initial_state: Tensor | None = None,
    ) -> ConvGrads: ...


ConvBackend = Backend[ConvForward, ConvBackward]

_REGISTRY: Registry[ConvForward, ConvBackward] = Registry("conv")

register = _REGISTRY.register
names = _REGISTRY.names
get = _REGISTRY.get
resolve = _REGISTRY.resolve


def _check_native(
    named: tuple[tuple[str, Tensor | None], ...],
    dims: ConvDims,
    dtype: torch.dtype,
) -> None:
    """Enforce what the kernels assume beyond the shared operand contract.

    The kernels index raw pointers with the channels-last stride pattern and are
    instantiated per dtype, so a non-contiguous or differently typed operand is
    refused rather than repacked or promoted.

    The layout half is :func:`slinoss._guard.check_layout`, which every kernel
    path in the repo shares. Only the dtype rule is this backend's own: one dtype
    for the whole call, because the kernel is one template instantiation.

    Order is layout, then dtype, so an operand that violates both is reported
    under the layout rule.

    Args:
        named: ``(name, tensor)`` pairs; a ``None`` tensor is skipped.
        dims: Extents of the call.
        dtype: Dtype every operand must carry, i.e. the dtype of ``x``.

    Raises:
        ValueError: On a non-contiguous operand, a dtype that differs from
            ``dtype``, a tap count above the kernel bound, or a non-CUDA device.
        TypeError: If ``dtype`` has no kernel instantiation.
    """
    if dtype not in KERNEL_DTYPES:
        raise TypeError(
            f"the native backend supports {KERNEL_DTYPES}, got {dtype}; "
            f"use the reference backend for float64"
        )
    bound = int(_C.extension().MAX_WIDTH)
    if dims.width > bound:
        raise ValueError(
            f"the native backend supports width <= {bound}, got {dims.width}"
        )
    present = tuple((tensor, name) for name, tensor in named if tensor is not None)
    check_layout(present)
    for tensor, name in present:
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")


def causal_conv1d_fwd_native(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    /,
    *,
    activation: bool = True,
    initial_state: Tensor | None = None,
    d_head: int | None = None,
) -> ConvStep:
    """Causal depthwise conv1d on the CUDA kernel.

    ``d_head`` moves the store address and nothing else: the output is allocated at
    the head-major shape and the kernel's epilogue writes it there, so the layout
    the scan needs costs no pass over the largest activation in the step.

    Args:
        x: Activations, shape ``(B,T,D)``, contiguous, bf16/fp16/fp32.
        weight: Taps, shape ``(D,W)``, contiguous, dtype of ``x``.
        bias: Per-channel bias, shape ``(D,)``, contiguous, dtype of ``x``, or
            None.
        activation: Apply SiLU. Fused into the kernel epilogue.
        initial_state: The ``W-1`` timesteps before ``x``, shape ``(B,W-1,D)``,
            contiguous, dtype of ``x``. Zero if omitted.
        d_head: Rows per head ``P``, which makes ``y`` head-major, or None for the
            token-major ``y``.

    Returns:
        A :class:`ConvStep`.

    Raises:
        ValueError: On a shape, contiguity, dtype, device, width, or ``d_head``
            violation.
        TypeError: On an unsupported dtype.
        RuntimeError: If the extension is not built.
    """
    dims = check_operands(x, weight, bias, initial_state)
    shape = conv_output_shape(dims.batch, dims.seqlen, dims.channels, d_head)
    _check_native(
        (
            ("x", x),
            ("weight", weight),
            ("bias", bias),
            ("initial_state", initial_state),
        ),
        dims,
        x.dtype,
    )
    # empty, never a fill: the kernel writes every element of both outputs.
    y = x.new_empty(shape)
    state = x.new_empty(conv_state_shape(dims.batch, dims.width, dims.channels))
    _C.extension().fwd(x, weight, bias, initial_state, y, state, activation)
    return ConvStep(y=y, state=state)


def causal_conv1d_bwd_native(
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
    """Pullback of :func:`causal_conv1d_fwd_native` on the CUDA kernel.

    The parameter gradients arrive as one float32 partial per time tile and are
    reduced by a second launch that writes the weight's own layout and dtype
    directly. Both kernels write every element of every output they are given, so
    nothing is zeroed before either launch.

    A head-major cotangent moves the kernel's ``dy`` load address and nothing else.
    ``dx`` is token-major because ``x`` is.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)`` or ``(B, D//P, T, P)``,
            contiguous, or None for a cotangent that is identically zero. Its rank
            is how the forward's output layout is recovered.
        dfinal_state: Cotangent of the returned window, shape ``(B,W-1,D)``,
            contiguous, or None.
        x: The forward's activations, shape ``(B,T,D)``.
        weight: The forward's taps, shape ``(D,W)``.
        bias: The forward's bias, shape ``(D,)``, or None.
        activation: The forward's activation flag.
        initial_state: The forward's incoming window, shape ``(B,W-1,D)``, or
            None.

    Returns:
        A :class:`ConvGrads`.

    Raises:
        ValueError: On a shape, contiguity, dtype, device, or width violation.
        TypeError: On an unsupported dtype.
        RuntimeError: If the extension is not built.
    """
    dims = check_operands(x, weight, bias, initial_state)
    check_cotangents(dy, dfinal_state, dims)
    _check_native(
        (
            ("x", x),
            ("weight", weight),
            ("bias", bias),
            ("initial_state", initial_state),
            ("dy", dy),
            ("dfinal_state", dfinal_state),
        ),
        dims,
        x.dtype,
    )
    module = _C.extension()
    dx = torch.empty_like(x)
    dinitial_state = None if initial_state is None else torch.empty_like(initial_state)
    parts = int(module.bwd_parts(dims.seqlen))
    dweight_parts = x.new_empty((parts, dims.width, dims.channels), dtype=torch.float32)
    dbias_parts = (
        None
        if bias is None
        else x.new_empty((parts, dims.channels), dtype=torch.float32)
    )
    module.bwd(
        dy,
        dfinal_state,
        x,
        weight,
        bias,
        initial_state,
        dx,
        dinitial_state,
        dweight_parts,
        dbias_parts,
        activation,
    )
    # One launch for both parameter gradients. The partials stay tap-major so the
    # backward's stores coalesce, and the reduction transposes to the weight's
    # (D,W) and narrows to its dtype in its own store; the torch expression it
    # replaces is five launches, a reduction and a transpose copy and a cast per
    # gradient. csrc/causal_conv1d_kernel.cu holds the measurement.
    dweight = weight.new_empty((dims.channels, dims.width))
    dbias = None if bias is None else x.new_empty((dims.channels,))
    module.bwd_reduce(dweight_parts, dbias_parts, dweight, dbias)
    return ConvGrads(
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_state=dinitial_state,
    )


register(
    Backend(
        name=REFERENCE,
        forward=causal_conv1d_update_ref,
        backward=causal_conv1d_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)

if _C.is_available():
    register(
        Backend(
            name=NATIVE,
            forward=causal_conv1d_fwd_native,
            backward=causal_conv1d_bwd_native,
            device_types=("cuda",),
            dtypes=KERNEL_DTYPES,
            priority=10,
        )
    )
