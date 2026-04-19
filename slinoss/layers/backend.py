"""Backend protocols, adapters, and canonical inputs for SLinOSS layers."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch

from slinoss.ops.mixer.convolution import (
    apply_cuda_causal_depthwise_conv,
    apply_reference_causal_depthwise_conv,
)
from slinoss.ops.mixer.step import (
    run_cute_decode_step,
    run_reference_decode_step,
    supports_cute_decode,
)
from slinoss.ops.cconv1d import cconv1d_cuda_supported, cconv1d_is_available
from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute

from .state import ScanState


_CUDA_MATH_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@dataclass(frozen=True, slots=True)
class ScanPrepInputs:
    """Canonical inputs for scan preparation backends.

    Shapes:
    - ``value``: ``(batch, T, heads * P)``
    - ``params``: ``(batch, T, heads * param_dim)``
    - ``bc``: ``(batch, T, groups, 4, N)``

    The BC stream carries grouped raw control rows
    ``(B_amp, B_phase, C_amp, C_phase)`` across contiguous head ranges. If
    ``H`` is the value-head count and ``G`` is the BC-group count, then the
    grouped BC slice consumed by head ``h`` is ``bc[..., h // (H // G), :, :]``.
    """

    value: torch.Tensor
    params: torch.Tensor
    bc: torch.Tensor


@dataclass(frozen=True, slots=True)
class ScanInputs:
    """Canonical packed inputs for scan backends.

    Shapes:
    - ``U``: ``(batch, heads, T, P)``
    - ``M``: ``(batch, heads, T, 2)``
    - ``K``: ``(batch, heads, T, 2, 2)``
    - ``B, C``: ``(batch, groups, T, 2N)``

    The grouped BC tensors use the same contiguous head-to-group mapping as
    ``ScanPrepInputs.bc``.
    """

    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor


@dataclass(frozen=True, slots=True)
class MixerDecodeInputs:
    """Canonical per-token inputs for mixer decode backends.

    Shapes:
    - ``value``: ``(batch, heads, P)`` post-conv/post-activation value token
    - ``params``: ``(batch, heads, param_dim)`` flat scanprep parameter token
    - ``bc``: ``(batch, groups, 4, N)`` raw grouped BC control rows
    - ``gate``: ``(batch, heads, P)`` token-local gating vector
    """

    value: torch.Tensor
    params: torch.Tensor
    bc: torch.Tensor
    gate: torch.Tensor


if TYPE_CHECKING:
    from slinoss.ops.mixer.step import _DecodeOwner as _MixerDecodeOwner

    class _ScanPrepOwner(Protocol):
        def _prepare_inputs_reference(self, inputs: ScanPrepInputs) -> ScanInputs: ...
        def _prepare_inputs_cute(self, inputs: ScanPrepInputs) -> ScanInputs: ...

    class _CConvOwner(Protocol):
        @property
        def d_inner(self) -> int: ...

        @property
        def d_conv(self) -> int: ...

        @property
        def dw_weight(self) -> torch.Tensor: ...

        @property
        def dw_bias(self) -> torch.Tensor: ...


class ScanPrepBackend(Protocol):
    """Protocol for scan preparation backends."""

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs: ...


class ReferenceScanPrepBackend:
    """Reference implementation of scan preparation."""

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        return owner._prepare_inputs_reference(inputs)


class CuteScanPrepBackend:
    """CuTe implementation of scan preparation."""

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        return owner._prepare_inputs_cute(inputs)


class AutoScanPrepBackend:
    """Selects the scan preparation implementation from the input tensors."""

    def __init__(self) -> None:
        self.reference = ReferenceScanPrepBackend()
        self.cute = CuteScanPrepBackend()

    def __call__(
        self,
        owner: "_ScanPrepOwner",
        inputs: ScanPrepInputs,
    ) -> ScanInputs:
        backend = self.cute if _should_use_cute_scanprep(inputs) else self.reference
        return backend(owner, inputs)


class CConv1dBackend(Protocol):
    """Protocol for activated depthwise causal convolution backends."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class ReferenceCConv1dBackend:
    """Reference implementation of activated depthwise causal convolution."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_reference_causal_depthwise_conv(owner, x, conv_state)


class CudaCConv1dBackend:
    """CUDA implementation of activated depthwise causal convolution."""

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not cconv1d_is_available():
            return apply_reference_causal_depthwise_conv(owner, x, conv_state)
        return apply_cuda_causal_depthwise_conv(owner, x, conv_state)


class AutoCConv1dBackend:
    """Selects the depthwise causal convolution implementation."""

    def __init__(self) -> None:
        self.reference = ReferenceCConv1dBackend()
        self.cuda = CudaCConv1dBackend()

    def __call__(
        self,
        owner: "_CConvOwner",
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self.cuda if _should_use_cuda_cconv(owner, x) else self.reference
        return backend(owner, x, conv_state)


class ScanBackend(Protocol):
    """Protocol for scan backends."""

    def __call__(
        self,
        inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]: ...


class MixerDecodeBackend(Protocol):
    """Protocol for per-token mixer decode backends."""

    def supports(
        self,
        owner: "_MixerDecodeOwner",
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool: ...

    def __call__(
        self,
        owner: "_MixerDecodeOwner",
        inputs: MixerDecodeInputs,
        state: ScanState,
    ) -> tuple[torch.Tensor, ScanState]: ...


def _all_tensors_on_cuda(*tensors: torch.Tensor) -> bool:
    return all(tensor.device.type == "cuda" for tensor in tensors)


def _all_tensors_supported(
    *tensors: torch.Tensor,
    dtypes: tuple[torch.dtype, ...],
) -> bool:
    return all(tensor.dtype in dtypes for tensor in tensors)


def _default_compute_dtype(dtype: torch.dtype) -> torch.dtype | None:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return None


def _resolve_scan_call(
    *,
    inputs: ScanInputs,
    state: ScanState | None,
    return_state: bool | None,
    compute_dtype: torch.dtype | None,
) -> tuple[ScanState, bool, torch.dtype, torch.dtype | None]:
    resolved_return_state = state is not None if return_state is None else return_state
    output_dtype = inputs.U.dtype
    resolved_compute_dtype = compute_dtype
    if resolved_compute_dtype is None:
        resolved_compute_dtype = _default_compute_dtype(output_dtype)
    return (
        ScanState() if state is None else state,
        bool(resolved_return_state),
        output_dtype,
        resolved_compute_dtype,
    )


def _make_next_scan_state(
    *,
    final_state: torch.Tensor,
    b_last: torch.Tensor,
    u_last: torch.Tensor,
) -> ScanState:
    return ScanState(state=final_state, b_prev=b_last, u_prev=u_last)


def _should_use_cute_scanprep(inputs: ScanPrepInputs) -> bool:
    return _all_tensors_on_cuda(inputs.value, inputs.params, inputs.bc) and (
        _all_tensors_supported(
            inputs.value,
            inputs.params,
            inputs.bc,
            dtypes=_CUDA_MATH_DTYPES,
        )
    )


def _should_use_cuda_cconv(owner: "_CConvOwner", x: torch.Tensor) -> bool:
    return (
        x.device.type == "cuda"
        and x.dtype in _CUDA_MATH_DTYPES
        and owner.d_conv in (2, 3, 4)
        and cconv1d_cuda_supported(
            x.transpose(1, 2),
            owner.dw_weight,
            activation=None,
        )
    )


class ReferenceScanBackend:
    """Reference implementation of the scan operator."""

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
        scan_state, return_state, output_dtype, compute_dtype = _resolve_scan_call(
            inputs=inputs,
            state=state,
            return_state=return_state,
            compute_dtype=self.compute_dtype,
        )

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
        return y, _make_next_scan_state(
            final_state=final_state,
            b_last=b_last,
            u_last=u_last,
        )


class CuteScanBackend:
    """CuTe implementation of the scan operator."""

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
        scan_state, return_state, output_dtype, compute_dtype = _resolve_scan_call(
            inputs=inputs,
            state=state,
            return_state=return_state,
            compute_dtype=self.compute_dtype,
        )

        if not return_state:
            return cast(
                torch.Tensor,
                v2x2ssd_cute(
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
                    return_state=False,
                ),
            )
        y, final_state, b_last, u_last = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            v2x2ssd_cute(
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
                return_state=True,
            ),
        )
        return y, _make_next_scan_state(
            final_state=final_state,
            b_last=b_last,
            u_last=u_last,
        )


class AutoScanBackend:
    """Selects the scan implementation from the input tensors."""

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


class ReferenceMixerDecodeBackend:
    """Reference implementation of per-token mixer decode."""

    def supports(
        self,
        owner: "_MixerDecodeOwner",
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        del owner, batch_size, device, dtype
        return True

    def __call__(
        self,
        owner: "_MixerDecodeOwner",
        inputs: MixerDecodeInputs,
        state: ScanState,
    ) -> tuple[torch.Tensor, ScanState]:
        return run_reference_decode_step(owner, inputs, state)


class CuteMixerDecodeBackend:
    """CuTe implementation of per-token mixer decode."""

    def supports(
        self,
        owner: "_MixerDecodeOwner",
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        return supports_cute_decode(
            owner,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        owner: "_MixerDecodeOwner",
        inputs: MixerDecodeInputs,
        state: ScanState,
    ) -> tuple[torch.Tensor, ScanState]:
        return run_cute_decode_step(owner, inputs, state)


class AutoMixerDecodeBackend:
    """Selects the per-token mixer decode implementation."""

    def __init__(self) -> None:
        self.reference = ReferenceMixerDecodeBackend()
        self.cute = CuteMixerDecodeBackend()

    def supports(
        self,
        owner: "_MixerDecodeOwner",
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        return self.cute.supports(
            owner,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        owner: "_MixerDecodeOwner",
        inputs: MixerDecodeInputs,
        state: ScanState,
    ) -> tuple[torch.Tensor, ScanState]:
        backend = (
            self.cute
            if self.cute.supports(
                owner,
                batch_size=int(inputs.value.shape[0]),
                device=inputs.value.device,
                dtype=inputs.value.dtype,
            )
            else self.reference
        )
        return backend(owner, inputs, state)


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
    "MixerDecodeInputs",
    "MixerDecodeBackend",
    "ReferenceMixerDecodeBackend",
    "CuteMixerDecodeBackend",
    "AutoMixerDecodeBackend",
]
