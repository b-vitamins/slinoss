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
from slinoss.ops.so3ssd.reference import ScanPrologue, SO3SSDResult, so3ssd_ref

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
    """Backward signature every backend implements.

    ``dB`` and ``dC`` are destinations rather than allocations the backend makes. The
    mixer's backward allocates one ``dproj`` and hands each consumer the band its
    gradient belongs in, so a backend that allocated and let the caller assign would
    write every gradient byte twice on a DRAM-bound path. Each is written in full and
    returned as the same object; ``None`` allocates one.

    ``dU_init`` is not a destination. It is a read-only addend: the returned ``dU`` is
    it plus the cotangent of ``U``, and ``dU`` stays a buffer the backend allocates.
    The mixer tail's ``du`` and the scan's ``dU`` both feed the conv's ``dy``, and the
    scan builds ``dU`` by accumulation already, so seeding that accumulation replaces
    a separate read-read-write over ``(B,H,T,P)`` and narrows the sum once.

    ``prologue`` is the matching forward's, or ``None``. A backend is free to ignore
    it and rebuild what it needs from the inputs, and the reference does. It is in the
    signature rather than in one backend's because the autograd boundary holds one
    saved set for every backend and dispatches on a name.
    """

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
        dB: Tensor | None = None,
        dC: Tensor | None = None,
        dU_init: Tensor | None = None,
        prologue: ScanPrologue | None = None,
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

    Both directions are kernel trees: three launches forward, five backward. The
    backward reads the chunk boundary the forward left rather than rebuilding it, so a
    training step on this backend touches no torch fallback and launches no forward
    kernel twice.
    """
    if not torch.cuda.is_available():
        return
    try:
        from slinoss.ops.so3ssd.cute.backward import so3ssd_bwd_cute
        from slinoss.ops.so3ssd.cute.forward import so3ssd_fwd_cute
    except ImportError:
        return
    register(
        Backend(
            name=CUTE,
            forward=so3ssd_fwd_cute,
            backward=so3ssd_bwd_cute,
            device_types=("cuda",),
            dtypes=LOW_PRECISION_DTYPES,
            priority=10,
        )
    )


_register_cute()
