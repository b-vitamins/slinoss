from __future__ import annotations

import torch
from torch.nn import functional as F

from slinoss.ops.gate import mixer_gate


def _reference_gate(scan_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    batch_size, n_heads, time_steps, d_head = map(int, scan_output.shape)
    gate_head = gate.reshape(batch_size, time_steps, n_heads, d_head).permute(
        0, 2, 1, 3
    )
    gated_head = (
        scan_output.to(torch.float32) * F.silu(gate_head.to(torch.float32))
    ).to(dtype=scan_output.dtype)
    return gated_head.permute(0, 2, 1, 3).reshape(
        batch_size, time_steps, n_heads * d_head
    )


def test_mixer_gate_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(0)
    batch_size, time_steps, n_heads, d_head = 2, 5, 3, 4
    scan_output = torch.randn(
        (batch_size, n_heads, time_steps, d_head),
        dtype=torch.float32,
        requires_grad=True,
    )
    gate = torch.randn(
        (batch_size, time_steps, n_heads * d_head),
        dtype=torch.float32,
        requires_grad=True,
    )

    scan_ref = scan_output.detach().clone().requires_grad_(True)
    gate_ref = gate.detach().clone().requires_grad_(True)

    out = mixer_gate(
        scan_output,
        gate,
        batch_size=batch_size,
        time_steps=time_steps,
        n_heads=n_heads,
        d_head=d_head,
    )
    out_ref = _reference_gate(scan_ref, gate_ref)
    torch.testing.assert_close(out, out_ref, atol=0.0, rtol=0.0)

    grad_out = torch.randn_like(out)
    dscan, dgate = torch.autograd.grad(out, (scan_output, gate), grad_out)
    dscan_ref, dgate_ref = torch.autograd.grad(out_ref, (scan_ref, gate_ref), grad_out)
    torch.testing.assert_close(dscan, dscan_ref, atol=2e-7, rtol=0.0)
    torch.testing.assert_close(dgate, dgate_ref, atol=3e-7, rtol=0.0)


def test_mixer_gate_saves_only_scan_and_gate_tensors() -> None:
    torch.manual_seed(0)
    batch_size, time_steps, n_heads, d_head = 2, 4, 3, 5
    scan_output = torch.randn(
        (batch_size, n_heads, time_steps, d_head),
        dtype=torch.float32,
        requires_grad=True,
    )
    gate = torch.randn(
        (batch_size, time_steps, n_heads * d_head),
        dtype=torch.float32,
        requires_grad=True,
    )
    saved: list[tuple[tuple[int, ...], torch.dtype]] = []

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        saved.append((tuple(map(int, t.shape)), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        out = mixer_gate(
            scan_output,
            gate,
            batch_size=batch_size,
            time_steps=time_steps,
            n_heads=n_heads,
            d_head=d_head,
        )
        torch.autograd.grad(out, (scan_output, gate), torch.randn_like(out))

    assert saved == [
        (tuple(map(int, scan_output.shape)), scan_output.dtype),
        (tuple(map(int, gate.shape)), gate.dtype),
    ]
