"""Eager memory-forensics helpers for perf harnesses."""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from .runtime import register_region_observer

_UNLABELED_REGION = "<unlabeled>"


def current_memory_stats(device: torch.device | str) -> dict[str, int]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }
    torch.cuda.synchronize(resolved)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
    }


def peak_memory_stats(device: torch.device | str) -> dict[str, int]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    torch.cuda.synchronize(resolved)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
    }


def reset_peak_memory_stats(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)


class EagerMemoryForensics:
    """Collect region-exit and saved-tensor memory attribution for one eager step."""

    def __init__(self, *, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self._region_stack: list[str] = []
        self._region_exit_stats: dict[str, dict[str, int]] = {}
        self._saved_by_region: dict[str, dict[str, int]] = {}
        self._seen_storage_keys: set[tuple[Any, ...]] = set()

    def region_enter(self, label: str) -> None:
        self._region_stack.append(label)

    def region_exit(self, label: str) -> None:
        stats = self._region_exit_stats.setdefault(
            label,
            {
                "max_allocated_bytes": 0,
                "max_reserved_bytes": 0,
                "num_exits": 0,
            },
        )
        if self.device.type == "cuda" and torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated(self.device))
            reserved = int(torch.cuda.memory_reserved(self.device))
        else:
            allocated = 0
            reserved = 0
        stats["max_allocated_bytes"] = max(stats["max_allocated_bytes"], allocated)
        stats["max_reserved_bytes"] = max(stats["max_reserved_bytes"], reserved)
        stats["num_exits"] += 1

        if self._region_stack and self._region_stack[-1] == label:
            self._region_stack.pop()
            return
        for idx in range(len(self._region_stack) - 1, -1, -1):
            if self._region_stack[idx] == label:
                del self._region_stack[idx]
                return

    def _active_region_label(self) -> str:
        for label in reversed(self._region_stack):
            if not label.startswith("step."):
                return label
        try:
            return self._region_stack[-1]
        except IndexError:
            return _UNLABELED_REGION

    def _saved_bucket(self, label: str) -> dict[str, int]:
        return self._saved_by_region.setdefault(
            label,
            {
                "unique_saved_bytes": 0,
                "unique_storage_count": 0,
                "save_event_count": 0,
            },
        )

    def _storage_key(self, tensor: torch.Tensor) -> tuple[tuple[Any, ...], int]:
        try:
            storage = tensor.untyped_storage()
            key = (
                tensor.device.type,
                tensor.device.index,
                int(storage.data_ptr()),
                int(storage.nbytes()),
            )
            return key, int(storage.nbytes())
        except RuntimeError:
            byte_size = int(tensor.numel() * tensor.element_size())
            key = (
                tensor.device.type,
                tensor.device.index,
                "fallback",
                id(tensor),
                byte_size,
            )
            return key, byte_size

    def pack_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        label = self._active_region_label()
        bucket = self._saved_bucket(label)
        bucket["save_event_count"] += 1
        storage_key, byte_size = self._storage_key(tensor)
        if storage_key not in self._seen_storage_keys:
            self._seen_storage_keys.add(storage_key)
            bucket["unique_saved_bytes"] += byte_size
            bucket["unique_storage_count"] += 1
        return tensor

    def unpack_saved_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @contextmanager
    def capture(self) -> Iterator[None]:
        hooks = torch.autograd.graph.saved_tensors_hooks(
            self.pack_saved_tensor,
            self.unpack_saved_tensor,
        )
        with register_region_observer(self):
            with hooks:
                yield

    def top_region_exit_allocated(self, *, top_k: int) -> list[dict[str, int | str]]:
        rows = [
            {
                "label": label,
                "max_allocated_bytes": stats["max_allocated_bytes"],
                "max_reserved_bytes": stats["max_reserved_bytes"],
                "num_exits": stats["num_exits"],
            }
            for label, stats in self._region_exit_stats.items()
        ]
        rows.sort(key=lambda row: int(row["max_allocated_bytes"]), reverse=True)
        return rows[: max(0, int(top_k))]

    def saved_tensors_by_region(self) -> list[dict[str, int | str]]:
        rows = [
            {
                "label": label,
                "unique_saved_bytes": stats["unique_saved_bytes"],
                "unique_storage_count": stats["unique_storage_count"],
                "save_event_count": stats["save_event_count"],
            }
            for label, stats in self._saved_by_region.items()
        ]
        rows.sort(key=lambda row: int(row["unique_saved_bytes"]), reverse=True)
        return rows

    def saved_tensors_summary(self) -> dict[str, int | str]:
        rows = self.saved_tensors_by_region()
        return {
            "accounting": "unique_storage_first_save",
            "total_unique_saved_bytes": sum(
                int(row["unique_saved_bytes"]) for row in rows
            ),
            "total_unique_storage_count": sum(
                int(row["unique_storage_count"]) for row in rows
            ),
            "total_save_event_count": sum(int(row["save_event_count"]) for row in rows),
        }


def allocator_snapshot_metadata(
    *,
    device: torch.device | str,
    out_path: Path | None,
) -> tuple[dict[str, Any], AbstractContextManager[None]]:
    resolved = torch.device(device)
    if out_path is None:
        return {
            "requested": False,
            "captured": False,
            "path": None,
            "format": None,
        }, nullcontext()
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return {
            "requested": True,
            "captured": False,
            "path": str(out_path),
            "format": "json",
            "reason": "cuda_unavailable",
        }, nullcontext()
    if not hasattr(torch.cuda.memory, "_record_memory_history") or not hasattr(
        torch.cuda.memory, "_snapshot"
    ):
        return {
            "requested": True,
            "captured": False,
            "path": str(out_path),
            "format": "json",
            "reason": "allocator_snapshot_api_unavailable",
        }, nullcontext()

    @contextmanager
    def _capture() -> Iterator[None]:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            clear_history=True,
            device=resolved,
        )
        try:
            yield
        finally:
            snapshot = torch.cuda.memory._snapshot(device=resolved)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
            torch.cuda.memory._record_memory_history(
                enabled=None,
                clear_history=True,
                device=resolved,
            )

    return {
        "requested": True,
        "captured": True,
        "path": str(out_path),
        "format": "json",
    }, _capture()
