"""Backend registry for the SO(3) scan.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is the scan's own: the two call signatures, and
which implementations exist.

The CuTe backend registers only when the DSL imports and a CUDA device is visible.
Importing the DSL on a host without one raises, and an operator that cannot be
imported on a CPU host is not an operator, so the import is guarded and a tree with
no DSL resolves to the reference everywhere.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from slinoss._precision import LOW_PRECISION_DTYPES, SUPPORTED_DTYPES
from slinoss._registry import Backend, Registry
from slinoss.ops.so3ssd.backward import SO3SSDGrads, so3ssd_bwd_ref
from slinoss.ops.so3ssd.reference import SO3SSDResult, so3ssd_ref

__all__ = [
    "Backend",
    "ScanBackend",
    "ScanBackward",
    "ScanForward",
    "get",
    "names",
    "register",
    "resolve",
]

REFERENCE = "reference"
CUTE = "cute"


class ScanForward(Protocol):
    """Forward signature every backend implements."""

    def __call__(
        self,
        U: Tensor,
        trans: Tensor,
        K: Tensor,
        B: Tensor,
        C: Tensor,
        chunk_size: int,
        /,
        *,
        z0: Tensor | None = None,
        b_prev: Tensor | None = None,
        u_prev: Tensor | None = None,
    ) -> SO3SSDResult: ...


class ScanBackward(Protocol):
    """Backward signature every backend implements."""

    def __call__(
        self,
        dy: Tensor | None,
        dstate: Tensor | None,
        db_last: Tensor | None,
        du_last: Tensor | None,
        U: Tensor,
        trans: Tensor,
        K: Tensor,
        B: Tensor,
        C: Tensor,
        chunk_size: int,
        /,
        *,
        z0: Tensor | None = None,
        b_prev: Tensor | None = None,
        u_prev: Tensor | None = None,
    ) -> SO3SSDGrads: ...


ScanBackend = Backend[ScanForward, ScanBackward]

_REGISTRY: Registry[ScanForward, ScanBackward] = Registry("so3ssd")

register = _REGISTRY.register
names = _REGISTRY.names
get = _REGISTRY.get
resolve = _REGISTRY.resolve

register(
    Backend(
        name=REFERENCE,
        forward=so3ssd_ref,
        backward=so3ssd_bwd_ref,
        device_types=("cpu", "cuda"),
        dtypes=SUPPORTED_DTYPES,
        priority=0,
    )
)


def _register_cute() -> None:
    """Register the CuTe backend if this host can run it.

    The forward is the three-kernel tree; the backward is still the reference, so a
    training step on this backend runs a fast forward against a torch backward. That
    is a deliberate intermediate state, not a fallback: the forward is the half that
    is finished, and shipping it under its own name keeps the benchmarked path and
    the public path identical while the backward lands.
    """
    if not torch.cuda.is_available():
        return
    try:
        from slinoss.ops.so3ssd.cute.forward import so3ssd_fwd_cute
    except ImportError:
        return
    register(
        Backend(
            name=CUTE,
            forward=so3ssd_fwd_cute,
            backward=so3ssd_bwd_ref,
            device_types=("cuda",),
            dtypes=LOW_PRECISION_DTYPES,
            priority=10,
        )
    )


_register_cute()
