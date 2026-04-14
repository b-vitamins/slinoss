"""Block channel-path training op for FFN memory rematerialization."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

import torch
from torch.autograd.function import once_differentiable

if TYPE_CHECKING:
    from slinoss.blocks.block import SLinOSSBlock


def _iter_channel_parameters(block: SLinOSSBlock) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    modules = []
    if block.ffn_norm is not None:
        modules.append(block.ffn_norm)
    if block.ffn is not None:
        modules.extend((block.ffn.in_proj, block.ffn.out_proj))
    for module in modules:
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            tensors.append(weight)
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            tensors.append(bias)
    return tuple(tensors)


def _channel_forward(
    block: SLinOSSBlock,
    residual: torch.Tensor,
    context: object | None,
) -> torch.Tensor:
    branch = block.forward_ffn_branch(residual, context=context)
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
        with torch.no_grad():
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


def block_ffn_residual(
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
        return _channel_forward(block, residual, context)

    # We only thread parameters that participate in gradient propagation.
    trainable = tuple(param for param in params if param.requires_grad)
    return cast(
        torch.Tensor,
        _BlockFFNResidualFn.apply(residual, block, context, *trainable),
    )


__all__ = ["block_ffn_residual"]
