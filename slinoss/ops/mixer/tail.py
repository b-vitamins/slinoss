"""Mixer tail op owning gate, output norm, and output projection."""

from __future__ import annotations

from contextlib import nullcontext
from typing import cast

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def _mixer_tail_dims(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
) -> tuple[int, int, int, int]:
    if scan_output.ndim != 4:
        raise ValueError(
            "scan_output must be 4-D; got "
            f"{scan_output.ndim}-D with shape {tuple(scan_output.shape)}."
        )
    batch_size, n_heads, time_steps, d_head = map(int, scan_output.shape)
    expected_gate = (batch_size, time_steps, n_heads * d_head)
    if gate.ndim != 3 or tuple(map(int, gate.shape)) != expected_gate:
        raise ValueError(
            f"gate must be {expected_gate}; got {tuple(map(int, gate.shape))}."
        )
    if gate.device != scan_output.device:
        raise ValueError("gate and scan_output must be on the same device.")
    return batch_size, n_heads, time_steps, d_head


def _validate_tail_parameters(
    out_norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    *,
    hidden_dim: int,
) -> None:
    if out_norm_weight.ndim != 1 or out_norm_weight.numel() != hidden_dim:
        raise ValueError(
            "out_norm_weight must be 1-D with length "
            f"{hidden_dim}; got {tuple(map(int, out_norm_weight.shape))}."
        )
    if out_proj_weight.ndim != 2 or int(out_proj_weight.shape[1]) != hidden_dim:
        raise ValueError(
            "out_proj_weight must be 2-D with shape "
            f"(*, {hidden_dim}); got {tuple(map(int, out_proj_weight.shape))}."
        )
    if out_proj_bias is not None and (
        out_proj_bias.ndim != 1 or out_proj_bias.numel() != out_proj_weight.shape[0]
    ):
        raise ValueError(
            "out_proj_bias must be 1-D with length "
            f"{out_proj_weight.shape[0]}; got {tuple(map(int, out_proj_bias.shape))}."
        )


def _mixer_gated_hidden(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, n_heads, time_steps, d_head = _mixer_tail_dims(scan_output, gate)
    if (skip_input is None) != (d_skip is None):
        raise ValueError(
            "skip_input and d_skip must either both be provided or both be None."
        )
    pre_gate = scan_output
    if skip_input is not None:
        expected_skip = (batch_size, n_heads, time_steps, d_head)
        if skip_input.ndim != 4 or tuple(map(int, skip_input.shape)) != expected_skip:
            raise ValueError(
                f"skip_input must be {expected_skip}; got {tuple(map(int, skip_input.shape))}."
            )
        if skip_input.device != scan_output.device:
            raise ValueError("skip_input and scan_output must be on the same device.")
        if d_skip is None:
            raise ValueError("d_skip must be provided when skip_input is provided.")
        if d_skip.ndim != 1 or d_skip.numel() != n_heads:
            raise ValueError(
                f"d_skip must be 1-D with length {n_heads}; got {tuple(map(int, d_skip.shape))}."
            )
        pre_gate = (
            scan_output.to(torch.float32)
            + skip_input.to(torch.float32)
            * d_skip.view(1, n_heads, 1, 1).to(torch.float32)
        ).to(dtype=scan_output.dtype)
    gate_head = gate.reshape(batch_size, time_steps, n_heads, d_head).permute(
        0,
        2,
        1,
        3,
    )
    gated_head = (pre_gate.to(torch.float32) * F.silu(gate_head.to(torch.float32))).to(
        dtype=pre_gate.dtype
    )
    return gated_head.permute(0, 2, 1, 3).reshape(
        batch_size,
        time_steps,
        n_heads * d_head,
    )


def _mixer_tail_forward(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm_weight: torch.Tensor,
    out_norm_eps: float | None,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
) -> torch.Tensor:
    _, n_heads, _, d_head = _mixer_tail_dims(scan_output, gate)
    hidden_dim = int(n_heads * d_head)
    _validate_tail_parameters(
        out_norm_weight,
        out_proj_weight,
        out_proj_bias,
        hidden_dim=hidden_dim,
    )
    gated = _mixer_gated_hidden(
        scan_output,
        gate,
        skip_input=skip_input,
        d_skip=d_skip,
    )
    normed = F.rms_norm(
        gated.to(dtype=out_norm_weight.dtype),
        (hidden_dim,),
        out_norm_weight,
        out_norm_eps,
    ).to(dtype=gated.dtype)
    return F.linear(normed, out_proj_weight, out_proj_bias)


class _MixerTailFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        scan_output: torch.Tensor,
        gate: torch.Tensor,
        skip_input: torch.Tensor | None,
        d_skip: torch.Tensor | None,
        out_norm_weight: torch.Tensor,
        out_norm_eps: float | None,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.device_type = scan_output.device.type
        ctx.autocast_enabled = bool(torch.is_autocast_enabled(ctx.device_type))
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        with torch.no_grad():
            out = _mixer_tail_forward(
                scan_output,
                gate,
                out_norm_weight,
                out_norm_eps,
                out_proj_weight,
                out_proj_bias,
                skip_input=skip_input,
                d_skip=d_skip,
            )
        if skip_input is None:
            ctx.save_for_backward(scan_output, gate)
        else:
            ctx.save_for_backward(scan_output, gate, skip_input)
        ctx.has_skip_input = bool(skip_input is not None)
        ctx.d_skip = d_skip
        ctx.out_norm_weight = out_norm_weight
        ctx.out_norm_eps = out_norm_eps
        ctx.out_proj_weight = out_proj_weight
        ctx.out_proj_bias = out_proj_bias
        return out

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        if ctx.has_skip_input:
            scan_output, gate, skip_input = ctx.saved_tensors
        else:
            scan_output, gate = ctx.saved_tensors
            skip_input = None
        scan_output_r = scan_output.detach().requires_grad_(True)
        gate_r = gate.detach().requires_grad_(True)
        skip_input_r = (
            None
            if skip_input is None
            else skip_input.detach().requires_grad_(skip_input.requires_grad)
        )
        d_skip = ctx.d_skip
        d_skip_r = (
            None
            if d_skip is None
            else d_skip.detach().requires_grad_(d_skip.requires_grad)
        )
        out_norm_weight_r = ctx.out_norm_weight.detach().requires_grad_(True)
        out_proj_weight_r = ctx.out_proj_weight.detach().requires_grad_(True)

        out_proj_bias = ctx.out_proj_bias
        out_proj_bias_r: torch.Tensor | None
        if out_proj_bias is None:
            out_proj_bias_r = None
        else:
            out_proj_bias_r = out_proj_bias.detach().requires_grad_(True)

        grad_inputs: list[torch.Tensor] = [
            scan_output_r,
            gate_r,
        ]
        if skip_input_r is not None:
            grad_inputs.append(skip_input_r)
        if d_skip_r is not None:
            grad_inputs.append(d_skip_r)
        grad_inputs.extend((out_norm_weight_r, out_proj_weight_r))
        if out_proj_bias_r is not None:
            grad_inputs.append(out_proj_bias_r)

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
                out = _mixer_tail_forward(
                    scan_output_r,
                    gate_r,
                    out_norm_weight_r,
                    ctx.out_norm_eps,
                    out_proj_weight_r,
                    out_proj_bias_r,
                    skip_input=skip_input_r,
                    d_skip=d_skip_r,
                )
            grad_outputs = torch.autograd.grad(
                out,
                grad_inputs,
                grad_out,
                allow_unused=True,
            )
        idx = 0
        d_scan_output = grad_outputs[idx]
        idx += 1
        d_gate = grad_outputs[idx]
        idx += 1
        d_skip_input = None
        if skip_input_r is not None:
            d_skip_input = grad_outputs[idx]
            idx += 1
        d_d_skip = None
        if d_skip_r is not None:
            d_d_skip = grad_outputs[idx]
            idx += 1
        d_norm_weight = grad_outputs[idx]
        idx += 1
        d_proj_weight = grad_outputs[idx]
        idx += 1
        d_proj_bias = grad_outputs[idx] if idx < len(grad_outputs) else None

        return (
            d_scan_output,
            d_gate,
            d_skip_input,
            d_d_skip,
            d_norm_weight,
            None,
            d_proj_weight,
            d_proj_bias,
        )


def mixer_tail(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    out_norm: nn.RMSNorm,
    out_proj: nn.Module,
    *,
    skip_input: torch.Tensor | None = None,
    d_skip: torch.Tensor | None = None,
) -> torch.Tensor:
    out_norm_weight = out_norm.weight
    if out_norm_weight is None:
        raise ValueError("mixer_tail requires an affine RMSNorm weight.")
    out_proj_weight = getattr(out_proj, "weight", None)
    if not isinstance(out_proj_weight, torch.Tensor):
        raise TypeError("mixer_tail requires out_proj to expose a tensor weight.")
    out_proj_bias = getattr(out_proj, "bias", None)
    if out_proj_bias is not None and not isinstance(out_proj_bias, torch.Tensor):
        raise TypeError("mixer_tail requires out_proj.bias to be a tensor or None.")

    requires_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (
            scan_output,
            gate,
            out_norm_weight,
            out_proj_weight,
            skip_input,
            d_skip,
        )
        if tensor is not None
    )
    if out_proj_bias is not None:
        requires_grad = requires_grad or out_proj_bias.requires_grad
    if not requires_grad:
        return _mixer_tail_forward(
            scan_output,
            gate,
            out_norm_weight,
            out_norm.eps,
            out_proj_weight,
            cast(torch.Tensor | None, out_proj_bias),
            skip_input=skip_input,
            d_skip=d_skip,
        )

    return cast(
        torch.Tensor,
        _MixerTailFn.apply(
            scan_output,
            gate,
            skip_input,
            d_skip,
            out_norm_weight,
            out_norm.eps,
            out_proj_weight,
            cast(torch.Tensor | None, out_proj_bias),
        ),
    )


__all__ = ["mixer_tail"]
