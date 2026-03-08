from __future__ import annotations

import math

import torch

from slinoss.layers import CuteScanBackend, ScanInputs, ScanState
from slinoss.layers import backend as backend_mod


def _make_inputs() -> tuple[ScanInputs, ScanState]:
    batch, heads, T, N, P = 2, 3, 7, 4, 5
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), dtype=torch.float32)
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, T), dtype=torch.float32
    ) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).contiguous()
    K_complex = (
        torch.randn((batch, heads, T, 2), dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).contiguous()
    U = torch.randn((batch, heads, T, P), dtype=torch.float32)
    B = torch.randn((batch, heads, T, 2 * N), dtype=torch.float32) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), dtype=torch.float32) * 0.1
    state = ScanState(
        state=torch.randn((batch, heads, P, 2 * N), dtype=torch.float32),
        b_prev=torch.randn((batch, heads, 2 * N), dtype=torch.float32),
        u_prev=torch.randn((batch, heads, P), dtype=torch.float32),
    )
    return ScanInputs(U=U, M=M, K=K, B=B, C=C), state


def test_cute_backend_preserves_scan_contract(monkeypatch) -> None:
    inputs, state = _make_inputs()
    calls: dict[str, object] = {}

    def fake_scan_op(
        U: torch.Tensor,
        M: torch.Tensor,
        K: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        *,
        chunk_size: int,
        initial_states: torch.Tensor | None = None,
        B_prev: torch.Tensor | None = None,
        U_prev: torch.Tensor | None = None,
        compute_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls["U"] = U
        calls["M"] = M
        calls["K"] = K
        calls["B"] = B
        calls["C"] = C
        calls["chunk_size"] = chunk_size
        calls["initial_states"] = initial_states
        calls["B_prev"] = B_prev
        calls["U_prev"] = U_prev
        calls["compute_dtype"] = compute_dtype
        calls["output_dtype"] = output_dtype

        y = torch.zeros_like(U)
        final_state = torch.full_like(initial_states, 3.0)  # type: ignore[arg-type]
        b_last = torch.full_like(B_prev, 5.0)  # type: ignore[arg-type]
        u_last = torch.full_like(U_prev, 7.0)  # type: ignore[arg-type]
        return y, final_state, b_last, u_last

    monkeypatch.setattr(backend_mod, "v2x2ssd_cute", fake_scan_op)

    backend = CuteScanBackend(compute_dtype=torch.float64)
    y, next_state = backend(inputs, chunk_size=4, state=state)

    assert calls["U"] is inputs.U
    assert calls["M"] is inputs.M
    assert calls["K"] is inputs.K
    assert calls["B"] is inputs.B
    assert calls["C"] is inputs.C
    assert calls["chunk_size"] == 4
    assert calls["initial_states"] is state.state
    assert calls["B_prev"] is state.b_prev
    assert calls["U_prev"] is state.u_prev
    assert calls["compute_dtype"] == torch.float64
    assert calls["output_dtype"] == inputs.U.dtype
    assert torch.equal(y, torch.zeros_like(inputs.U))
    assert next_state.state is not None
    assert next_state.b_prev is not None
    assert next_state.u_prev is not None
    assert torch.equal(next_state.state, torch.full_like(state.state, 3.0))  # type: ignore[arg-type]
    assert torch.equal(next_state.b_prev, torch.full_like(state.b_prev, 5.0))  # type: ignore[arg-type]
    assert torch.equal(next_state.u_prev, torch.full_like(state.u_prev, 7.0))  # type: ignore[arg-type]
