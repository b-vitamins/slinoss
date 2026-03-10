"""Backend boundary for SLinOSS scan operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute

from .state import ScanState


@dataclass(frozen=True)
class ScanInputs:
    """Canonical packed inputs for a v2x2 scan backend.

    Shapes:
    - ``U``: ``(batch, heads, T, P)``
    - ``M``: ``(batch, heads, T, 2)``
    - ``K``: ``(batch, heads, T, 2, 2)``
    - ``B, C``: ``(batch, heads, T, 2N)``
    """

    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor


class ScanBackend(Protocol):
    """Hot-swappable SLinOSS scan backend."""

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
    ) -> tuple[torch.Tensor, ScanState]: ...


def _default_compute_dtype(dtype: torch.dtype) -> torch.dtype | None:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return None


def _run_scan_op(
    scan_op,
    inputs: ScanInputs,
    *,
    chunk_size: int,
    state: ScanState | None,
    compute_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, ScanState]:
    scan_state = ScanState() if state is None else state
    output_dtype = inputs.U.dtype
    if compute_dtype is None:
        compute_dtype = _default_compute_dtype(output_dtype)

    y, final_state, b_last, u_last = scan_op(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        inputs.C,
        chunk_size=chunk_size,
        initial_states=scan_state.state,
        B_prev=scan_state.b_prev,
        U_prev=scan_state.u_prev,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )
    next_state = ScanState(state=final_state, b_prev=b_last, u_prev=u_last)
    return y, next_state


class ReferenceScanBackend:
    """Reference backend that wraps the staged ``v2x2ssd`` implementation."""

    def __init__(self, *, compute_dtype: torch.dtype | None = None) -> None:
        self.compute_dtype = compute_dtype

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
    ) -> tuple[torch.Tensor, ScanState]:
        return _run_scan_op(
            v2x2ssd,
            inputs,
            chunk_size=chunk_size,
            state=state,
            compute_dtype=self.compute_dtype,
        )


class CuteScanBackend:
    """CuTe backend wrapper for the staged ``v2x2ssd`` operator."""

    def __init__(self, *, compute_dtype: torch.dtype | None = None) -> None:
        self.compute_dtype = compute_dtype

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
    ) -> tuple[torch.Tensor, ScanState]:
        return _run_scan_op(
            v2x2ssd_cute,
            inputs,
            chunk_size=chunk_size,
            state=state,
            compute_dtype=self.compute_dtype,
        )


class AutoScanBackend:
    """Default backend that routes CUDA inputs to CuTe and others to reference."""

    def __init__(self, *, compute_dtype: torch.dtype | None = None) -> None:
        self.compute_dtype = compute_dtype
        self.reference = ReferenceScanBackend(compute_dtype=compute_dtype)
        self.cute = CuteScanBackend(compute_dtype=compute_dtype)

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
    ) -> tuple[torch.Tensor, ScanState]:
        backend = self.cute if inputs.U.device.type == "cuda" else self.reference
        return backend(inputs, chunk_size=chunk_size, state=state)


__all__ = [
    "ScanInputs",
    "ScanBackend",
    "ReferenceScanBackend",
    "CuteScanBackend",
    "AutoScanBackend",
]
