"""Custom autograd gate op for mixer scan outputs."""

from __future__ import annotations

from typing import cast

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


def _validate_gate_contract(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    batch_size: int,
    time_steps: int,
    n_heads: int,
    d_head: int,
) -> None:
    expected_scan = (batch_size, n_heads, time_steps, d_head)
    if scan_output.ndim != 4 or tuple(map(int, scan_output.shape)) != expected_scan:
        raise ValueError(
            "scan_output must be "
            f"{expected_scan}; got {tuple(map(int, scan_output.shape))}."
        )
    expected_gate = (batch_size, time_steps, n_heads * d_head)
    if gate.ndim != 3 or tuple(map(int, gate.shape)) != expected_gate:
        raise ValueError(
            f"gate must be {expected_gate}; got {tuple(map(int, gate.shape))}."
        )
    if gate.device != scan_output.device:
        raise ValueError("gate and scan_output must be on the same device.")


class _MixerGateFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        scan_output: torch.Tensor,
        gate: torch.Tensor,
        batch_size: int,
        time_steps: int,
        n_heads: int,
        d_head: int,
    ) -> torch.Tensor:
        batch_size = int(batch_size)
        time_steps = int(time_steps)
        n_heads = int(n_heads)
        d_head = int(d_head)
        _validate_gate_contract(
            scan_output,
            gate,
            batch_size=batch_size,
            time_steps=time_steps,
            n_heads=n_heads,
            d_head=d_head,
        )
        gate_head = gate.reshape(batch_size, time_steps, n_heads, d_head).permute(
            0, 2, 1, 3
        )
        gated_head = (
            scan_output.to(torch.float32) * F.silu(gate_head.to(torch.float32))
        ).to(dtype=scan_output.dtype)
        out = gated_head.permute(0, 2, 1, 3).reshape(
            batch_size,
            time_steps,
            n_heads * d_head,
        )
        ctx.save_for_backward(scan_output, gate)
        ctx.batch_size = batch_size
        ctx.time_steps = time_steps
        ctx.n_heads = n_heads
        ctx.d_head = d_head
        return out

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        scan_output, gate = ctx.saved_tensors
        batch_size = int(ctx.batch_size)
        time_steps = int(ctx.time_steps)
        n_heads = int(ctx.n_heads)
        d_head = int(ctx.d_head)

        grad_head = grad_out.reshape(batch_size, time_steps, n_heads, d_head).permute(
            0, 2, 1, 3
        )
        gate_head = gate.reshape(batch_size, time_steps, n_heads, d_head).permute(
            0, 2, 1, 3
        )
        gate_f = gate_head.to(torch.float32)
        grad_f = grad_head.to(torch.float32)

        d_scan = (grad_f * F.silu(gate_f)).to(dtype=scan_output.dtype)
        d_gate_head = torch.ops.aten.silu_backward.default(
            grad_f * scan_output, gate_f
        ).to(dtype=gate.dtype)
        d_gate = d_gate_head.permute(0, 2, 1, 3).reshape(
            batch_size, time_steps, n_heads * d_head
        )
        return d_scan, d_gate, None, None, None, None


def mixer_gate(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    *,
    batch_size: int,
    time_steps: int,
    n_heads: int,
    d_head: int,
) -> torch.Tensor:
    """Apply ``scan_output * silu(gate)`` with a low-save custom autograd surface."""
    return cast(
        torch.Tensor,
        _MixerGateFn.apply(
            scan_output,
            gate,
            int(batch_size),
            int(time_steps),
            int(n_heads),
            int(d_head),
        ),
    )


__all__ = ["mixer_gate"]
