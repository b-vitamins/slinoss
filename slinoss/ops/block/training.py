"""Block channel-path training op for FFN memory rematerialization."""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, cast

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from slinoss.layers import RMSNorm, SLinOSSMLP

if TYPE_CHECKING:
    from slinoss.blocks.block import SLinOSSBlock

_INITIALIZED_CUBLAS_DEVICES: set[int] = set()


def _iter_channel_named_parameters(
    block: SLinOSSBlock,
) -> tuple[tuple[str, torch.Tensor], ...]:
    tensors: list[tuple[str, torch.Tensor]] = []
    if block.ffn_norm is not None:
        weight = getattr(block.ffn_norm, "weight", None)
        if isinstance(weight, torch.Tensor):
            tensors.append(("ffn_norm.weight", weight))
        bias = getattr(block.ffn_norm, "bias", None)
        if isinstance(bias, torch.Tensor):
            tensors.append(("ffn_norm.bias", bias))
    if block.ffn is not None:
        tensors.append(("ffn.in_proj.weight", block.ffn.in_proj.weight))
        if block.ffn.in_proj.bias is not None:
            tensors.append(("ffn.in_proj.bias", block.ffn.in_proj.bias))
        tensors.append(("ffn.out_proj.weight", block.ffn.out_proj.weight))
        if block.ffn.out_proj.bias is not None:
            tensors.append(("ffn.out_proj.bias", block.ffn.out_proj.bias))
    return tuple(tensors)


def _iter_channel_parameters(block: SLinOSSBlock) -> tuple[torch.Tensor, ...]:
    return tuple(tensor for _, tensor in _iter_channel_named_parameters(block))


def _channel_forward(
    block: SLinOSSBlock,
    residual: torch.Tensor,
    context: object | None,
) -> torch.Tensor:
    branch = block.forward_ffn_branch(residual, context=context)
    return block._residual_add(residual, block._drop_branch(branch))


def _supports_cute_ffn_rowwise(
    block: SLinOSSBlock,
    residual: torch.Tensor,
    context: object | None,
) -> bool:
    if residual.device.type != "cuda":
        return False
    if context is not None:
        return False
    if block.residual_dropout != 0.0:
        return False
    if not isinstance(block.ffn_norm, RMSNorm):
        return False
    if not isinstance(block.ffn, SLinOSSMLP):
        return False
    if block.ffn.kind not in {"gelu", "swiglu"}:
        return False
    norm_weight = block.ffn_norm.weight
    if norm_weight is None:
        return False
    tensors: list[torch.Tensor] = [
        residual,
        norm_weight,
        block.ffn.in_proj.weight,
        block.ffn.out_proj.weight,
    ]
    if block.ffn.in_proj.bias is not None:
        tensors.append(block.ffn.in_proj.bias)
    if block.ffn.out_proj.bias is not None:
        tensors.append(block.ffn.out_proj.bias)
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if any(tensor.device != residual.device for tensor in tensors):
        return False
    if any(tensor.dtype not in supported_dtypes for tensor in tensors):
        return False
    try:
        import cutlass  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_cuda_current_context(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.current_stream(device=device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    if device_index in _INITIALIZED_CUBLAS_DEVICES:
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attempting to run cuBLAS, but there was no current CUDA context!",
        )
        torch.cuda.current_blas_handle()
    _INITIALIZED_CUBLAS_DEVICES.add(device_index)


def _linear_input_grad(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    _ensure_cuda_current_context(grad_out.device)
    return F.linear(grad_out, weight.t())


def _linear_weight_grad(
    grad_out: torch.Tensor,
    input_: torch.Tensor,
) -> torch.Tensor:
    _ensure_cuda_current_context(grad_out.device)
    grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
    input_2d = input_.reshape(-1, input_.shape[-1])
    return grad_out_2d.transpose(0, 1).matmul(input_2d)


def _linear_bias_grad(
    grad_out: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor | None:
    if bias is None:
        return None
    return grad_out.reshape(-1, grad_out.shape[-1]).sum(dim=0, dtype=bias.dtype)


def _resolved_rms_eps(eps: float | None, dtype: torch.dtype) -> float:
    if eps is not None:
        return float(eps)
    return float(torch.finfo(dtype).eps)


def _channel_forward_cute(
    block: SLinOSSBlock,
    residual: torch.Tensor,
) -> torch.Tensor:
    from .cute import (
        _ffn_activation_fwd_cute_prevalidated,
        _ffn_rmsnorm_fwd_cute_prevalidated,
    )

    assert isinstance(block.ffn_norm, RMSNorm)
    assert block.ffn_norm.weight is not None
    assert isinstance(block.ffn, SLinOSSMLP)
    norm_eps = _resolved_rms_eps(block.ffn_norm.eps, block.ffn_norm.weight.dtype)

    normed = _ffn_rmsnorm_fwd_cute_prevalidated(
        residual,
        block.ffn_norm.weight,
        eps=norm_eps,
    )
    _ensure_cuda_current_context(normed.device)
    projected = F.linear(normed, block.ffn.in_proj.weight, block.ffn.in_proj.bias)
    hidden = _ffn_activation_fwd_cute_prevalidated(projected, kind=block.ffn.kind)
    branch = F.linear(hidden, block.ffn.out_proj.weight, block.ffn.out_proj.bias)
    return block._residual_add(residual, block._drop_branch(branch))


def _can_use_ffn_remat(
    block: SLinOSSBlock,
    residual: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    context: object | None,
) -> bool:
    if not torch.is_grad_enabled():
        return False
    if block.ffn is None or block.ffn_norm is None:
        return False
    if block.residual_dropout != 0.0:
        return False
    if isinstance(context, torch.Tensor) and context.requires_grad:
        return False
    if residual.requires_grad:
        return True
    return any(param.requires_grad for param in params)


class _BlockFFNResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        residual: torch.Tensor,
        block: SLinOSSBlock,
        context: object | None,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.device_type = residual.device.type
        ctx.autocast_enabled = bool(torch.is_autocast_enabled(ctx.device_type))
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        ctx.block = block
        ctx.context = context
        ctx.use_cute_rowwise = _supports_cute_ffn_rowwise(block, residual, context)
        ctx.param_names = tuple(
            name
            for name, tensor in _iter_channel_named_parameters(block)
            if tensor.requires_grad
        )
        with torch.no_grad():
            if ctx.use_cute_rowwise:
                out = _channel_forward_cute(block, residual)
            else:
                out = _channel_forward(block, residual, context)
        ctx.save_for_backward(residual, *params)
        return out

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        saved = ctx.saved_tensors
        residual = saved[0]
        params = saved[1:]

        if ctx.use_cute_rowwise:
            from .cute import (
                _ffn_activation_bwd_cute_prevalidated,
                _ffn_activation_fwd_cute_prevalidated,
                _ffn_rmsnorm_bwd_cute_prevalidated,
                _ffn_rmsnorm_fwd_cute_prevalidated,
            )

            block = ctx.block
            assert isinstance(block.ffn_norm, RMSNorm)
            assert block.ffn_norm.weight is not None
            assert isinstance(block.ffn, SLinOSSMLP)

            norm_weight = block.ffn_norm.weight
            in_proj_weight = block.ffn.in_proj.weight
            in_proj_bias = block.ffn.in_proj.bias
            out_proj_weight = block.ffn.out_proj.weight
            out_proj_bias = block.ffn.out_proj.bias
            norm_eps = _resolved_rms_eps(block.ffn_norm.eps, norm_weight.dtype)

            with torch.no_grad():
                normed = _ffn_rmsnorm_fwd_cute_prevalidated(
                    residual,
                    norm_weight,
                    eps=norm_eps,
                )
                projected = F.linear(normed, in_proj_weight, in_proj_bias)
                hidden = _ffn_activation_fwd_cute_prevalidated(
                    projected,
                    kind=block.ffn.kind,
                )
                d_out_proj_weight = _linear_weight_grad(grad_out, hidden)
                d_out_proj_bias = _linear_bias_grad(grad_out, out_proj_bias)
                del hidden
                d_hidden = _linear_input_grad(grad_out, out_proj_weight)
                d_projected = _ffn_activation_bwd_cute_prevalidated(
                    projected,
                    d_hidden,
                    kind=block.ffn.kind,
                )
                del d_hidden
                del projected
                d_in_proj_weight = _linear_weight_grad(d_projected, normed)
                d_in_proj_bias = _linear_bias_grad(d_projected, in_proj_bias)
                del normed
                d_normed = _linear_input_grad(d_projected, in_proj_weight)
                del d_projected
                norm_bwd = _ffn_rmsnorm_bwd_cute_prevalidated(
                    residual,
                    norm_weight,
                    d_normed,
                    eps=norm_eps,
                )
                del d_normed
                d_residual = grad_out + norm_bwd.d_input

            grad_by_name: dict[str, torch.Tensor] = {
                "ffn_norm.weight": norm_bwd.d_weight,
                "ffn.in_proj.weight": d_in_proj_weight,
                "ffn.out_proj.weight": d_out_proj_weight,
            }
            if d_in_proj_bias is not None:
                grad_by_name["ffn.in_proj.bias"] = d_in_proj_bias
            if d_out_proj_bias is not None:
                grad_by_name["ffn.out_proj.bias"] = d_out_proj_bias
            d_params = tuple(grad_by_name[name] for name in ctx.param_names)
            return (d_residual, None, None, *d_params)

        residual_r = residual.detach().requires_grad_(True)
        grad_inputs: list[torch.Tensor] = [residual_r, *params]

        autocast_ctx = (
            torch.autocast(
                device_type=ctx.device_type,
                dtype=ctx.autocast_dtype,
                enabled=bool(ctx.autocast_enabled),
            )
            if ctx.device_type in {"cpu", "cuda"}
            else nullcontext()
        )
        with torch.enable_grad():
            with autocast_ctx:
                out = _channel_forward(ctx.block, residual_r, ctx.context)
            grad_outputs = torch.autograd.grad(
                out,
                grad_inputs,
                grad_out,
                allow_unused=True,
            )

        d_residual = grad_outputs[0]
        d_params = grad_outputs[1:]
        return (d_residual, None, None, *d_params)


def _block_ffn_residual_impl(
    block: SLinOSSBlock,
    residual: torch.Tensor,
    *,
    context: object | None = None,
) -> torch.Tensor:
    """Apply pre-FFN norm -> FFN -> residual add with optional rematerialization."""

    if block.ffn is None or block.ffn_norm is None:
        raise RuntimeError("block_ffn_residual requires a block with an FFN branch.")

    params = _iter_channel_parameters(block)
    if not _can_use_ffn_remat(block, residual, params, context):
        if _supports_cute_ffn_rowwise(block, residual, context):
            return _channel_forward_cute(block, residual)
        return _channel_forward(block, residual, context)

    # We only thread parameters that participate in gradient propagation.
    trainable = tuple(param for param in params if param.requires_grad)
    return cast(
        torch.Tensor,
        _BlockFFNResidualFn.apply(residual, block, context, *trainable),
    )


block_ffn_residual = cast(
    Callable[..., Any],
    torch.compiler.disable(_block_ffn_residual_impl),
)


__all__ = ["block_ffn_residual"]
