"""Backend registry, shared by every operator.

One entry point per implementation. A variant reachable from a benchmark and not
from the public path is a defect, so selection goes through :meth:`Registry.resolve`
and nothing else.

Registration order does not decide anything. Each backend declares the device
types and the activation dtypes it runs on, plus a priority; :meth:`Registry.resolve`
picks the highest priority backend that supports both. That keeps the shipped path
and the benchmarked path identical without an import-order dependency, and it keeps
a fast path that cannot take an operand from being selected for it.

Declaring the dtypes rather than raising from inside the kernel matters: the scan's
tensor-core atom is 16-bit, so a float32 activation has no fast path at all, and a
caller who passes one wants the reference rather than an exception. Resolution is
the only place that decision belongs.

Shape is not a resolution axis. Every extent the kernels constrain -- ``P``, ``L``,
``3N`` -- is fixed by :mod:`slinoss.config` at construction, so a violating shape is
a caller error and raises from the operand guard rather than silently selecting a
slower path.

One implementation rather than one per operator: four operators with four copies of
the same lookup is four chances for the resolution rule to drift, and a drifted rule
means the benchmark and the shipped path can select differently.
"""

from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

import torch

__all__ = ["Backend", "Registry"]

Fwd = TypeVar("Fwd")
Bwd = TypeVar("Bwd")


class Backend(NamedTuple, Generic[Fwd, Bwd]):
    """One implementation of an operator.

    Attributes:
        name: Registry key.
        forward: Forward entry point. Its signature is the operator's own forward
            protocol.
        backward: Backward entry point. Its signature is the operator's own
            backward protocol.
        device_types: Torch device types this backend runs on, e.g. ``("cuda",)``.
        dtypes: Activation dtypes this backend accepts. The reference declares the
            operator's whole supported set; a kernel backend declares only what it
            has an instantiation for.
        priority: Higher wins in :meth:`Registry.resolve`. The reference is 0.
    """

    name: str
    forward: Fwd
    backward: Bwd
    device_types: tuple[str, ...]
    dtypes: tuple[torch.dtype, ...]
    priority: int


class Registry(Generic[Fwd, Bwd]):
    """The backends of one operator.

    Args:
        operator: Operator name, used in error messages so a message names which
            registry rejected the call.
    """

    def __init__(self, operator: str) -> None:
        self._operator = operator
        self._entries: dict[str, Backend[Fwd, Bwd]] = {}

    def register(self, backend: Backend[Fwd, Bwd]) -> Backend[Fwd, Bwd]:
        """Add a backend.

        Args:
            backend: The backend.

        Returns:
            The backend, so a module can register and bind in one statement.

        Raises:
            ValueError: If the name is already registered. Two implementations
                under one name is the defect this registry exists to prevent.
        """
        if backend.name in self._entries:
            raise ValueError(f"backend {backend.name!r} is already registered")
        self._entries[backend.name] = backend
        return backend

    def names(self) -> tuple[str, ...]:
        """Registered backend names, sorted."""
        return tuple(sorted(self._entries))

    def get(self, name: str) -> Backend[Fwd, Bwd]:
        """Look a backend up by name.

        Args:
            name: Registry key.

        Returns:
            The backend.

        Raises:
            ValueError: If no backend is registered under that name.
        """
        if name not in self._entries:
            raise ValueError(f"unknown backend {name!r}; registered: {self.names()}")
        return self._entries[name]

    def resolve(
        self, name: str | None, device_type: str, dtype: torch.dtype
    ) -> Backend[Fwd, Bwd]:
        """Select a backend for a device and an activation dtype.

        Args:
            name: Explicit backend name, or ``None`` to select automatically.
            device_type: Torch device type, e.g. ``"cuda"``.
            dtype: Activation dtype of the call.

        Returns:
            The backend.

        Raises:
            ValueError: If a named backend does not support the device or the
                dtype, or if no registered backend supports the pair. A named
                backend reports the device before the dtype, so a call that
                violates both is reported under the device rule.
        """
        if name is not None:
            backend = self.get(name)
            if device_type not in backend.device_types:
                raise ValueError(
                    f"backend {name!r} supports {backend.device_types}, "
                    f"not {device_type!r}"
                )
            if dtype not in backend.dtypes:
                raise ValueError(
                    f"backend {name!r} supports {backend.dtypes}, not {dtype}"
                )
            return backend
        on_device = [b for b in self._entries.values() if device_type in b.device_types]
        if not on_device:
            raise ValueError(
                f"no {self._operator} backend supports device type {device_type!r}"
            )
        usable = [b for b in on_device if dtype in b.dtypes]
        if not usable:
            raise ValueError(
                f"no {self._operator} backend supports {dtype} on {device_type!r}"
            )
        return max(usable, key=lambda b: b.priority)
