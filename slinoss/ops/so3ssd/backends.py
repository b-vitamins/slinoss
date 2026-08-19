"""Backend registry for the SO(3) scan.

The lookup itself is :class:`slinoss._registry.Registry`, which every operator
shares. This module holds only what is the scan's own: the two call signatures, and
which implementations exist.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from slinoss._precision import SUPPORTED_DTYPES
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
