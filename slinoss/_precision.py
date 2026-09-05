"""Float32-pinning policy.

The rotation state is float32 or wider everywhere, including under autocast:
``trans``, ``K``, the per-step quaternions, both chunk-local prefixes, the 3x3
transform table, and the recurrent state ``z``. Only ``U``, ``B``, ``C``, ``Y``,
the score matrix, and GEMM operands may be low precision.

Rotation error enters the rotation matrix squared and the chunk-local prefixes
accumulate over the whole chunk, so a low-precision transition is a correctness
defect rather than a speed/accuracy trade.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Self, cast

import torch
from torch import Tensor, nn

LOW_PRECISION_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16)
"""Admitted for ``U``, ``B``, ``C``, ``Y`` and GEMM operands only."""

WIDE_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64)
"""Admitted anywhere. float64 exists so the reference is the fp64 oracle."""

SUPPORTED_DTYPES: tuple[torch.dtype, ...] = LOW_PRECISION_DTYPES + WIDE_DTYPES

KERNEL_DTYPES: tuple[torch.dtype, ...] = (*LOW_PRECISION_DTYPES, torch.float32)
"""Dtypes a rowwise CuTe kernel can read and write.

Wider than :data:`LOW_PRECISION_DTYPES` because a rowwise kernel computes in
float32 whatever it loads, so a float32 operand costs it bandwidth and nothing
else. A GEMM operand is narrower than this: the tensor-core atom is 16-bit, so
the scan states its own set. float64 has no kernel path at all; it is the
reference oracle's width and runs in torch.
"""

PINNED_TENSORS: tuple[str, ...] = ("trans", "K", "q", "Q", "lp", "table", "z")
"""Names that must never carry a dtype from :data:`LOW_PRECISION_DTYPES`."""


class Float32Module(nn.Module):
    """Module whose named tensors stay float32 under low-precision casts."""

    _float32_names: tuple[str, ...] = ()

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        super()._apply(fn, recurse)
        for name in self._float32_names:
            tensor = cast(Tensor, getattr(self, name))
            if tensor.dtype in LOW_PRECISION_DTYPES:
                tensor.data = tensor.data.float()
        return self


def check_supported(tensor: Tensor, name: str) -> None:
    """Reject a dtype the operator does not implement.

    Args:
        tensor: Tensor to check.
        name: Name used in the error message.

    Raises:
        TypeError: If ``tensor.dtype`` is not in :data:`SUPPORTED_DTYPES`.
    """
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}; supported: {SUPPORTED_DTYPES}"
        )


def check_pinned(tensor: Tensor, name: str) -> None:
    """Reject a low-precision dtype on a pinned tensor.

    Args:
        tensor: Tensor to check.
        name: Name used in the error message. Expected to be in
            :data:`PINNED_TENSORS`.

    Raises:
        TypeError: If ``tensor.dtype`` is in :data:`LOW_PRECISION_DTYPES`.
    """
    check_supported(tensor, name)
    if tensor.dtype in LOW_PRECISION_DTYPES:
        raise TypeError(
            f"{name} is float32-pinned and cannot be {tensor.dtype}; "
            f"pinned tensors: {PINNED_TENSORS}"
        )


def pinned_dtype(*tensors: Tensor) -> torch.dtype:
    """Resolve the dtype the pinned math runs in.

    float64 whenever any operand is float64, so a float64 call is an fp64
    oracle end to end. float32 otherwise, including when every operand is
    bfloat16 or float16.

    Args:
        *tensors: Operands of the call.

    Returns:
        ``torch.float64`` or ``torch.float32``.

    Raises:
        ValueError: If no tensors are given.
    """
    if not tensors:
        raise ValueError("pinned_dtype needs at least one tensor")
    if any(t.dtype is torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def cast_to(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """``tensor`` in ``dtype``, and ``tensor`` itself when it is already in it.

    Only parameters and gradients reach this. Casting an activation band would copy
    the tensor a fused projection exists to keep in place.

    Args:
        tensor: Tensor to cast.
        dtype: Target dtype.

    Returns:
        ``tensor`` or a copy of it in ``dtype``.
    """
    return tensor if tensor.dtype is dtype else tensor.to(dtype)


def cast_opt(tensor: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    """:func:`cast_to` through an absent operand.

    Args:
        tensor: Tensor to cast, or None.
        dtype: Target dtype.

    Returns:
        None, or the cast tensor.
    """
    return None if tensor is None else cast_to(tensor, dtype)


def device_type_of(tensor: Tensor) -> str:
    """Return the autocast device-type string for ``tensor``."""
    return tensor.device.type


@contextlib.contextmanager
def autocast_disabled(device_type: str) -> Iterator[None]:
    """Run pinned math outside autocast.

    Autocast rewrites matmul operands to the autocast dtype. Every contraction
    over a pinned tensor is wrapped in this so the rewrite cannot reach the
    transition.

    Args:
        device_type: Device type string, e.g. ``"cuda"`` or ``"cpu"``.

    Yields:
        None.
    """
    with torch.amp.autocast(device_type=device_type, enabled=False):
        yield
