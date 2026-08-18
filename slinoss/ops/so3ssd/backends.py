"""Backend registry for the SO(3) scan.

One entry point per implementation. A variant reachable from a benchmark and not
from the public path is a defect, so selection goes through :func:`resolve` and
nothing else.

Registration order does not decide anything. Each backend declares the device
types it runs on and a priority; :func:`resolve` picks the highest priority
backend that supports the requested device. That keeps the shipped path and the
benchmarked path identical without an import-order dependency.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from torch import Tensor

from slinoss.ops.so3ssd.backward import SO3SSDGrads, so3ssd_bwd_ref
from slinoss.ops.so3ssd.reference import SO3SSDResult, so3ssd_ref

__all__ = [
    "Backend",
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


class Backend(NamedTuple):
    """One scan implementation.

    Attributes:
        name: Registry key.
        forward: Forward entry point.
        backward: Backward entry point.
        device_types: Torch device types this backend runs on, e.g. ``("cuda",)``.
        priority: Higher wins in :func:`resolve`. The reference is 0.
    """

    name: str
    forward: ScanForward
    backward: ScanBackward
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


register(
    Backend(
        name=REFERENCE,
        forward=so3ssd_ref,
        backward=so3ssd_bwd_ref,
        device_types=("cpu", "cuda"),
        priority=0,
    )
)
