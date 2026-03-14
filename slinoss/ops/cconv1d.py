"""Vendored CUDA depthwise causal conv1d wrapper."""

from __future__ import annotations

import importlib
from typing import Any, Final

import torch
from torch.nn import functional as F

try:
    _cconv1d_cuda = importlib.import_module("slinoss._C.cconv1d_cuda")
except Exception as exc:  # pragma: no cover - exercised in CPU-only envs
    _cconv1d_cuda = None
    _CCONV1D_LOAD_ERROR: Exception | None = exc
else:
    _CCONV1D_LOAD_ERROR = None


_SUPPORTED_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_SUPPORTED_WIDTHS: Final[frozenset[int]] = frozenset((2, 3, 4))
_SUPPORTED_ACTIVATIONS: Final[frozenset[str | None]] = frozenset(
    (None, "silu", "swish")
)


def cconv1d_is_available() -> bool:
    """Return ``True`` when the compiled CUDA extension is importable."""

    return _cconv1d_cuda is not None


def cconv1d_load_error() -> Exception | None:
    """Return the extension import error when CUDA op loading failed."""

    return _CCONV1D_LOAD_ERROR


def cconv1d_cuda_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    initial_states: torch.Tensor | None = None,
    activation: str | None = None,
) -> bool:
    """Return ``True`` if inputs can run through the CUDA kernel."""

    if not cconv1d_is_available():
        return False
    if activation not in _SUPPORTED_ACTIVATIONS:
        return False
    if x.device.type != "cuda" or weight.device.type != "cuda":
        return False
    if initial_states is not None and initial_states.device.type != "cuda":
        return False
    if x.dtype not in _SUPPORTED_DTYPES or weight.dtype not in _SUPPORTED_DTYPES:
        return False
    if int(weight.shape[-1]) not in _SUPPORTED_WIDTHS:
        return False
    return True


def _cconv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    final_states_out: torch.Tensor | None,
    silu_activation: bool,
) -> torch.Tensor:
    assert _cconv1d_cuda is not None
    out = torch.empty_like(x)
    _cconv1d_cuda.cconv1d_fwd(
        x,
        weight,
        bias,
        None,  # seq_idx
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )
    return out


def _cconv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    initial_states: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
    return_dinitial_states: bool,
    silu_activation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    assert _cconv1d_cuda is not None
    batch_size, dim = map(int, x.shape[:2])
    width = int(weight.shape[-1])
    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(
            (batch_size, width - 1, dim), device=x.device, dtype=x.dtype
        ).transpose(1, 2)

    _cconv1d_cuda.cconv1d_bwd(
        x,
        weight,
        bias,
        dout,
        None,  # seq_idx
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
    )

    dweight = dweight.to(dtype=weight.dtype)
    if dbias is not None and bias is not None:
        dbias = dbias.to(dtype=bias.dtype)
    return dx, dweight, dbias, dinitial_states


class _CConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        return_final_states: bool = False,
        activation: str | None = None,
    ) -> Any:
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise NotImplementedError("activation must be None, silu, or swish")
        if _cconv1d_cuda is None:
            raise RuntimeError("cconv1d CUDA extension is not available")
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if initial_states is not None and (
            initial_states.stride(2) != 1 and initial_states.stride(1) != 1
        ):
            initial_states = initial_states.contiguous()
        if return_final_states:
            if x.stride(1) != 1:
                raise ValueError(
                    "return_final_states requires channel-last layout (x.stride(1) == 1)"
                )
            batch, dim, _ = map(int, x.shape)
            width = int(weight.shape[1])
            final_states_out = torch.empty(
                (batch, width - 1, dim), device=x.device, dtype=x.dtype
            ).transpose(1, 2)
        else:
            final_states_out = None

        ctx.silu_activation = activation in ("silu", "swish")
        out = _cconv1d_fwd_function(
            x,
            weight,
            bias,
            initial_states,
            final_states_out,
            bool(ctx.silu_activation),
        )
        ctx.save_for_backward(x, weight)
        ctx.bias = bias
        ctx.initial_states = initial_states
        ctx.return_final_states = bool(return_final_states)
        ctx.return_dinitial_states = bool(
            initial_states is not None and initial_states.requires_grad
        )
        return out if not return_final_states else (out, final_states_out)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,
        dout: torch.Tensor,
        *args: Any,
    ) -> Any:
        x, weight = ctx.saved_tensors
        bias = ctx.bias
        initial_states = ctx.initial_states
        dfinal_states = args[0] if ctx.return_final_states else None
        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        dx, dweight, dbias, dinitial_states = _cconv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            initial_states,
            dfinal_states,
            bool(ctx.return_dinitial_states),
            bool(ctx.silu_activation),
        )
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            dinitial_states if initial_states is not None else None,
            None,
            None,
        )


def cconv1d_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    activation: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """CUDA depthwise causal conv1d, backed by the vendored extension."""

    if not cconv1d_cuda_supported(
        x,
        weight,
        initial_states=initial_states,
        activation=activation,
    ):
        raise ValueError("Inputs are unsupported for cconv1d CUDA execution")
    out: Any = _CConv1dFn.apply(
        x,
        weight,
        bias,
        initial_states,
        return_final_states,
        activation,
    )
    if return_final_states:
        if not isinstance(out, tuple):
            raise RuntimeError("cconv1d CUDA expected tuple output")
        y, final_states = out
        assert isinstance(y, torch.Tensor)
        assert isinstance(final_states, torch.Tensor)
        return y, final_states
    assert isinstance(out, torch.Tensor)
    return out


def cconv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    activation: str | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference path matching ``F.conv1d(..., groups=channels)`` semantics."""

    if activation not in _SUPPORTED_ACTIVATIONS:
        raise NotImplementedError("activation must be None, silu, or swish")

    dtype_in = x.dtype
    x_f = x.to(weight.dtype)
    seqlen = int(x.shape[-1])
    dim, width = map(int, weight.shape)
    if x_f.ndim != 3 or int(x_f.shape[1]) != dim:
        raise ValueError(f"x must be (batch, {dim}, seqlen); got {tuple(x_f.shape)}")
    if initial_states is None:
        out = F.conv1d(x_f, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x_f = torch.cat([initial_states.to(dtype=x_f.dtype), x_f], dim=-1)
        out = F.conv1d(x_f, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    final_states: torch.Tensor | None = None
    if return_final_states:
        final_states = F.pad(x_f, (width - 1 - x_f.shape[-1], 0)).to(dtype_in)
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    if not return_final_states:
        return out
    assert final_states is not None
    return out, final_states


def cconv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    activation: str | None = None,
    prefer_cuda: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to CUDA when available, else use the reference implementation."""

    if prefer_cuda and cconv1d_cuda_supported(
        x,
        weight,
        initial_states=initial_states,
        activation=activation,
    ):
        return cconv1d_cuda(
            x,
            weight,
            bias,
            initial_states=initial_states,
            return_final_states=return_final_states,
            activation=activation,
        )
    return cconv1d_reference(
        x,
        weight,
        bias,
        initial_states=initial_states,
        return_final_states=return_final_states,
        activation=activation,
    )


__all__ = [
    "cconv1d",
    "cconv1d_cuda",
    "cconv1d_reference",
    "cconv1d_cuda_supported",
    "cconv1d_is_available",
    "cconv1d_load_error",
]
