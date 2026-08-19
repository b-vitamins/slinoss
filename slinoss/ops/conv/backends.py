"""Backend registry for the causal depthwise conv1d.

One entry point per implementation. A variant reachable from a benchmark and not
from the public path is a defect, so selection goes through :func:`resolve` and
nothing else.

Registration order does not decide anything. Each backend declares the device
types it runs on and a priority; :func:`resolve` picks the highest priority
backend that supports the requested device. That keeps the shipped path and the
benchmarked path identical without an import-order dependency.

The native backend registers only when the compiled extension imported. An
unbuilt tree therefore resolves to the reference on every device instead of
resolving to a path that cannot run.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
from torch import Tensor

from slinoss import _C
from slinoss._guard import check_layout
from slinoss._precision import KERNEL_DTYPES
from slinoss.ops.conv.reference import (
    ConvDims,
    ConvGrads,
    ConvStep,
    causal_conv1d_bwd_ref,
    causal_conv1d_update_ref,
    check_cotangents,
    check_operands,
    conv_state_shape,
)

__all__ = [
    "Backend",
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
    ) -> ConvStep: ...


class ConvBackward(Protocol):
    """Backward signature every backend implements."""

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


class Backend(NamedTuple):
    """One conv1d implementation.

    Attributes:
        name: Registry key.
        forward: Forward entry point.
        backward: Backward entry point.
        device_types: Torch device types this backend runs on, e.g. ``("cuda",)``.
        priority: Higher wins in :func:`resolve`. The reference is 0.
    """

    name: str
    forward: ConvForward
    backward: ConvBackward
    device_types: tuple[str, ...]
    priority: int


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Add a backend to the registry.

    Args:
        backend: The backend.

    Returns:
        The backend, so a module can register and bind in one statement.

    Raises:
        ValueError: If the name is already registered. Two implementations under
            one name is the defect this registry exists to prevent.
    """
    if backend.name in _REGISTRY:
        raise ValueError(f"backend {backend.name!r} is already registered")
    _REGISTRY[backend.name] = backend
    return backend


def names() -> tuple[str, ...]:
    """Registered backend names, sorted."""
    return tuple(sorted(_REGISTRY))


def get(name: str) -> Backend:
    """Look a backend up by name.

    Args:
        name: Registry key.

    Returns:
        The backend.

    Raises:
        ValueError: If no backend is registered under that name.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; registered: {names()}")
    return _REGISTRY[name]


def resolve(name: str | None, device_type: str) -> Backend:
    """Select a backend for a device.

    Args:
        name: Explicit backend name, or ``None`` to select automatically.
        device_type: Torch device type, e.g. ``"cuda"``.

    Returns:
        The backend.

    Raises:
        ValueError: If a named backend does not support the device, or if no
            registered backend supports it.
    """
    if name is not None:
        backend = get(name)
        if device_type not in backend.device_types:
            raise ValueError(
                f"backend {name!r} supports {backend.device_types}, not {device_type!r}"
            )
        return backend
    usable = [b for b in _REGISTRY.values() if device_type in b.device_types]
    if not usable:
        raise ValueError(f"no backend supports device type {device_type!r}")
    return max(usable, key=lambda b: b.priority)


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
) -> ConvStep:
    """Causal depthwise conv1d on the CUDA kernel.

    Args:
        x: Activations, shape ``(B,T,D)``, contiguous, bf16/fp16/fp32.
        weight: Taps, shape ``(D,W)``, contiguous, dtype of ``x``.
        bias: Per-channel bias, shape ``(D,)``, contiguous, dtype of ``x``, or
            None.
        activation: Apply SiLU. Fused into the kernel epilogue.
        initial_state: The ``W-1`` timesteps before ``x``, shape ``(B,W-1,D)``,
            contiguous, dtype of ``x``. Zero if omitted.

    Returns:
        A :class:`ConvStep`.

    Raises:
        ValueError: On a shape, contiguity, dtype, device, or width violation.
        TypeError: On an unsupported dtype.
        RuntimeError: If the extension is not built.
    """
    dims = check_operands(x, weight, bias, initial_state)
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
    y = torch.empty_like(x)
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
    summed here. The kernel writes every partial with a plain store, so nothing
    is zeroed before the launch.

    Args:
        dy: Cotangent of ``y``, shape ``(B,T,D)``, contiguous, or None for a
            cotangent that is identically zero.
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
    dweight_parts = x.new_empty((parts, dims.channels, dims.width), dtype=torch.float32)
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
    return ConvGrads(
        dx=dx,
        dweight=dweight_parts.sum(0).to(weight.dtype),
        dbias=None if dbias_parts is None else dbias_parts.sum(0).to(x.dtype),
        dinitial_state=dinitial_state,
    )


register(
    Backend(
        name=REFERENCE,
        forward=causal_conv1d_update_ref,
        backward=causal_conv1d_bwd_ref,
        device_types=("cpu", "cuda"),
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
            priority=10,
        )
    )
