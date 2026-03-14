"""Backend boundaries for SLinOSS scan preparation and scan operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from slinoss.ops.cconv1d import cconv1d_cuda_supported, cconv1d_is_available
from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute

from .state import ScanState


@dataclass(frozen=True)
class ScanPrepInputs:
    """Canonical inputs for the scanprep backend.

    Shapes:
    - ``value``: ``(batch, T, heads * P)``
    - ``params``: ``(batch, T, heads * param_dim)``
    - ``bc``: ``(batch, T, heads, 4, N)``
    """

    value: torch.Tensor
    params: torch.Tensor
    bc: torch.Tensor


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


if TYPE_CHECKING:

    class _ScanPrepOwner(Protocol):
        def _prepare_inputs_reference(self, inputs: ScanPrepInputs) -> ScanInputs: ...
        def _prepare_inputs_cute(self, inputs: ScanPrepInputs) -> ScanInputs: ...

    class _CConvOwner(Protocol):
        d_conv: int

        @property
        def dw_weight(self) -> torch.Tensor: ...

        def _apply_cconv_reference(
            self,
            x: torch.Tensor,
            conv_state: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...

        def _apply_cconv_cuda(
            self,
            x: torch.Tensor,
            conv_state: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ScanPrepBackend(Protocol):
    """Hot-swappable SLinOSS scanprep backend."""

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs: ...


class ReferenceScanPrepBackend:
    """Reference backend for preparing scan-native ``(U, M, K, B, C)`` inputs."""

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        return owner._prepare_inputs_reference(inputs)


class CuteScanPrepBackend:
    """Explicit CuTe scanprep backend.

    This is not the default yet. The eager/reference path remains the source of
    truth until the fused CuTe implementation is complete.
    """

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        return owner._prepare_inputs_cute(inputs)


class AutoScanPrepBackend:
    """Default scanprep backend. CUDA fusion will route here later."""

    def __init__(self) -> None:
        self.reference = ReferenceScanPrepBackend()
        self.cute = CuteScanPrepBackend()

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        use_cute = (
            inputs.value.device.type == "cuda"
            and inputs.params.device.type == "cuda"
            and inputs.bc.device.type == "cuda"
            and inputs.value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and inputs.params.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and inputs.bc.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        backend = self.cute if use_cute else self.reference
        return backend(owner, inputs)


class CConv1dBackend(Protocol):
    """Hot-swappable depthwise causal conv1d backend."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ReferenceCConv1dBackend:
    """Reference depthwise causal conv1d backend."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return owner._apply_cconv_reference(x, conv_state)


class CudaCConv1dBackend:
    """CUDA depthwise causal conv1d backend."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cconv1d_is_available():
            return owner._apply_cconv_reference(x, conv_state)
        return owner._apply_cconv_cuda(x, conv_state)


class AutoCConv1dBackend:
    """Default depthwise causal conv1d backend."""

    def __init__(self) -> None:
        self.reference = ReferenceCConv1dBackend()
        self.cuda = CudaCConv1dBackend()

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_cuda = (
            x.device.type == "cuda"
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and owner.d_conv in (2, 3, 4)
            and cconv1d_cuda_supported(
                x.transpose(1, 2),
                owner.dw_weight,
                activation=None,
            )
        )
        backend = self.cuda if use_cuda else self.reference
        return backend(owner, x, conv_state)


class ScanBackend(Protocol):
    """Hot-swappable SLinOSS scan backend."""

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]: ...


def _default_compute_dtype(dtype: torch.dtype) -> torch.dtype | None:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return None


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
        return_state: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        if return_state is None:
            return_state = state is not None
        scan_state = ScanState() if state is None else state
        output_dtype = inputs.U.dtype
        compute_dtype = self.compute_dtype
        if compute_dtype is None:
            compute_dtype = _default_compute_dtype(output_dtype)

        y, final_state, b_last, u_last = v2x2ssd(
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
        if not return_state:
            return y
        next_state = ScanState(state=final_state, b_prev=b_last, u_prev=u_last)
        return y, next_state


class CuteScanBackend:
    """Training-only CuTe backend wrapper for the staged ``v2x2ssd`` operator."""

    def __init__(self, *, compute_dtype: torch.dtype | None = None) -> None:
        self.compute_dtype = compute_dtype

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool | None = None,
    ) -> torch.Tensor:
        if return_state is None:
            return_state = state is not None
        if state is not None or return_state:
            raise NotImplementedError(
                "CuTe scan backend currently supports only stateless training execution."
            )

        output_dtype = inputs.U.dtype
        compute_dtype = self.compute_dtype
        if compute_dtype is None:
            compute_dtype = _default_compute_dtype(output_dtype)

        return v2x2ssd_cute(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            inputs.C,
            chunk_size=chunk_size,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
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
        return_state: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        if return_state is None:
            return_state = state is not None
        backend = self.cute if inputs.U.device.type == "cuda" else self.reference
        return backend(
            inputs,
            chunk_size=chunk_size,
            state=state,
            return_state=return_state,
        )


__all__ = [
    "CConv1dBackend",
    "ReferenceCConv1dBackend",
    "CudaCConv1dBackend",
    "AutoCConv1dBackend",
    "ScanPrepInputs",
    "ScanPrepBackend",
    "ReferenceScanPrepBackend",
    "CuteScanPrepBackend",
    "AutoScanPrepBackend",
    "ScanInputs",
    "ScanBackend",
    "ReferenceScanBackend",
    "CuteScanBackend",
    "AutoScanBackend",
]
