"""Memory forensics: what the autograd graph holds, and what the allocator peaked at.

The saved set is read with ``torch.autograd.graph.saved_tensors_hooks``, so the
count is what autograd actually holds rather than what the code intends to hold.
Deduplication is by storage identity, because autograd keeps a storage alive, not
a view: two views of one buffer cost one buffer.

``derived_bytes`` is the number the rematerialization policy is judged on. It is
the saved bytes that do not belong to any declared input, so a kernel that
stashes a workspace shows up here and nowhere else.

There is no generic summarizer in this module. A byte total and a duration do not
share a reduction, and a summarizer that names its output for one unit while
holding another is how a byte count ends up in a field called ``mean_ms``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated

import torch
from torch import Tensor

from slinoss.perf.device import device_ordinal
from slinoss.perf.timing import UNATTRIBUTED, active_recorder
from slinoss.perf.units import (
    INVARIANT,
    SUM,
    Bytes,
    Count,
    Mebibytes,
    PerfRecord,
    mib_from_bytes,
)

__all__ = [
    "MemoryPeaks",
    "RegionSaved",
    "SavedStorages",
    "SavedTensorProbe",
    "memory_peaks",
    "peak_window",
    "reset_memory_peaks",
    "saved_storage_bytes",
]

_StorageKey = tuple[str, int, int, int]


def _key(tensor: Tensor) -> _StorageKey:
    storage = tensor.untyped_storage()
    device = tensor.device
    return (
        device.type,
        device_ordinal(device),
        storage.data_ptr(),
        storage.nbytes(),
    )


@dataclass(frozen=True)
class RegionSaved(PerfRecord):
    """What one region contributed to the graph.

    Attributes:
        label: Region the tensor was saved inside, ``unattributed`` outside every
            region.
        storage_count: Distinct storages first seen in this region.
        save_event_count: Saves recorded in this region, repeats included.
        saved_bytes: Bytes of the storages first seen in this region.
    """

    label: str
    storage_count: Annotated[Count, SUM]
    save_event_count: Annotated[Count, SUM]
    saved_bytes: Annotated[Bytes, SUM]


@dataclass(frozen=True)
class SavedStorages(PerfRecord):
    """The whole graph's saved set.

    Attributes:
        label: What was probed.
        storage_count: Distinct storages held.
        save_event_count: Total saves, repeats included. Exceeds
            ``storage_count`` whenever one buffer is saved by several nodes.
        saved_bytes: Bytes of the distinct storages.
        input_bytes: Bytes of the saved storages that are declared inputs.
        derived_bytes: The rest. Zero is the target for a rematerializing
            backward.
        regions: Per-region attribution, in first-seen order.
    """

    label: str
    storage_count: Annotated[Count, SUM]
    save_event_count: Annotated[Count, SUM]
    saved_bytes: Annotated[Bytes, SUM]
    input_bytes: Annotated[Bytes, SUM]
    derived_bytes: Annotated[Bytes, SUM]
    regions: tuple[RegionSaved, ...]

    @property
    def saved_mib(self) -> Mebibytes:
        """Saved bytes as mebibytes, for a report line."""
        return mib_from_bytes(self.saved_bytes)


class SavedTensorProbe:
    """Records every tensor autograd saves while the probe is open.

    The probe does not change what is saved. It observes, then reports against a
    declared input set.

    Example:
        >>> probe = SavedTensorProbe()
        >>> with probe:
        ...     out = op(u, b, c)
        >>> report = probe.report("so3ssd", inputs=(u, b, c))
    """

    def __init__(self) -> None:
        self._seen: dict[_StorageKey, str] = {}
        self._bytes: dict[_StorageKey, int] = {}
        self._events: list[str] = []
        self._order: list[str] = []
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)

    def _pack(self, tensor: Tensor) -> Tensor:
        recorder = active_recorder()
        label = "" if recorder is None else recorder.current_label()
        # A blank label in a table is a broken label. A save taken outside every
        # region is unattributed, and the record says so.
        label = label or UNATTRIBUTED
        self._events.append(label)
        key = _key(tensor)
        if key not in self._seen:
            self._seen[key] = label
            self._bytes[key] = key[3]
            if label not in self._order:
                self._order.append(label)
        return tensor

    @staticmethod
    def _unpack(tensor: Tensor) -> Tensor:
        return tensor

    def __enter__(self) -> SavedTensorProbe:
        self._hooks.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._hooks.__exit__(*exc)

    def report(self, label: str, inputs: Iterable[Tensor] = ()) -> SavedStorages:
        """Summarize the saved set.

        Args:
            label: What was probed.
            inputs: Tensors the operator was called with. A saved storage
                belonging to one of these is not derived.

        Returns:
            The saved-storage record.
        """
        declared = {_key(t) for t in inputs}
        input_bytes = sum(n for k, n in self._bytes.items() if k in declared)
        total_bytes = sum(self._bytes.values())
        regions = tuple(
            RegionSaved(
                label=name,
                storage_count=Count(
                    sum(1 for owner in self._seen.values() if owner == name)
                ),
                save_event_count=Count(
                    sum(1 for owner in self._events if owner == name)
                ),
                saved_bytes=Bytes(
                    sum(n for k, n in self._bytes.items() if self._seen[k] == name)
                ),
            )
            for name in self._order
        )
        return SavedStorages(
            label=label,
            storage_count=Count(len(self._seen)),
            save_event_count=Count(len(self._events)),
            saved_bytes=Bytes(total_bytes),
            input_bytes=Bytes(input_bytes),
            derived_bytes=Bytes(total_bytes - input_bytes),
            regions=regions,
        )


@dataclass(frozen=True)
class MemoryPeaks(PerfRecord):
    """Allocator high-water marks over a window.

    ``reserved`` is what the process took from the driver; ``allocated`` is what
    tensors held. Reserved above allocated is fragmentation, not a leak.

    Attributes:
        label: What the window covered.
        peak_allocated_bytes: Peak of live tensor bytes.
        peak_reserved_bytes: Peak of allocator-reserved bytes.
    """

    label: str
    peak_allocated_bytes: Annotated[Bytes, INVARIANT]
    peak_reserved_bytes: Annotated[Bytes, INVARIANT]


def reset_memory_peaks(device: torch.device) -> None:
    """Clear the allocator's high-water marks. No effect on a CPU device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_peaks(label: str, device: torch.device) -> MemoryPeaks:
    """Read the allocator's high-water marks.

    Args:
        label: What the window covered.
        device: Device to read. A CPU device reports zeros, since the CUDA
            caching allocator is the only allocator instrumented here.

    Returns:
        The peaks.
    """
    if device.type != "cuda":
        return MemoryPeaks(
            label=label,
            peak_allocated_bytes=Bytes(0),
            peak_reserved_bytes=Bytes(0),
        )
    return MemoryPeaks(
        label=label,
        peak_allocated_bytes=Bytes(torch.cuda.max_memory_allocated(device)),
        peak_reserved_bytes=Bytes(torch.cuda.max_memory_reserved(device)),
    )


@contextmanager
def peak_window(label: str, device: torch.device) -> Iterator[list[MemoryPeaks]]:
    """Reset the peaks, run the body, and append the peaks to the yielded list.

    Args:
        label: What the window covers.
        device: Device to read.

    Yields:
        A one-element sink; the record lands in it on exit.
    """
    sink: list[MemoryPeaks] = []
    reset_memory_peaks(device)
    try:
        yield sink
    finally:
        sink.append(memory_peaks(label, device))


def saved_storage_bytes(tensors: Sequence[Tensor]) -> Bytes:
    """Distinct storage bytes behind a tensor sequence.

    Args:
        tensors: Tensors to measure.

    Returns:
        The sum over distinct storages. Aliases count once.
    """
    seen = {_key(t): _key(t)[3] for t in tensors}
    return Bytes(sum(seen.values()))
