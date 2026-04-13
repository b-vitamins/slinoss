from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch

from slinoss.layers import AutoScanBackend, CuteScanBackend, ScanInputs, ScanState
from slinoss.layers import backend as backend_mod


def _make_inputs(
    *,
    heads: int = 3,
    bc_groups: int | None = None,
) -> tuple[ScanInputs, ScanState]:
    batch, T, N, P = 2, 7, 4, 5
    groups = heads if bc_groups is None else bc_groups
    assert heads % groups == 0
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
    B = torch.randn((batch, groups, T, 2 * N), dtype=torch.float32) * 0.1
    C = torch.randn((batch, groups, T, 2 * N), dtype=torch.float32) * 0.1
    state = ScanState(
        state=torch.randn((batch, heads, P, 2 * N), dtype=torch.float32),
        b_prev=torch.randn((batch, groups, 2 * N), dtype=torch.float32),
        u_prev=torch.randn((batch, heads, P), dtype=torch.float32),
    )
    return ScanInputs(U=U, M=M, K=K, B=B, C=C), state


def test_cute_backend_runs_stateless_training_contract(monkeypatch) -> None:
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
        return_state: bool = False,
    ) -> torch.Tensor:
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
        calls["return_state"] = return_state

        return torch.zeros_like(U)

    monkeypatch.setattr(backend_mod, "v2x2ssd_cute", fake_scan_op)

    backend = CuteScanBackend(compute_dtype=torch.float64)
    y = cast(torch.Tensor, backend(inputs, chunk_size=4))

    assert calls["U"] is inputs.U
    assert calls["M"] is inputs.M
    assert calls["K"] is inputs.K
    assert calls["B"] is inputs.B
    assert calls["C"] is inputs.C
    assert calls["chunk_size"] == 4
    assert calls["initial_states"] is None
    assert calls["B_prev"] is None
    assert calls["U_prev"] is None
    assert calls["compute_dtype"] == torch.float64
    assert calls["output_dtype"] == inputs.U.dtype
    assert calls["return_state"] is False
    assert torch.equal(y, torch.zeros_like(inputs.U))


@pytest.mark.parametrize(("heads", "bc_groups"), [(3, 3), (4, 2)])
def test_cute_backend_threads_stateful_execution(
    heads: int,
    bc_groups: int,
) -> None:
    inputs, state = _make_inputs(heads=heads, bc_groups=bc_groups)
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
        return_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del U, M, K, chunk_size, compute_dtype, output_dtype
        calls["B"] = B
        calls["C"] = C
        calls["initial_states"] = initial_states
        calls["B_prev"] = B_prev
        calls["U_prev"] = U_prev
        calls["return_state"] = return_state
        return (
            torch.zeros_like(inputs.U),
            state.state + 1.0,  # type: ignore[operator]
            state.b_prev + 2.0,  # type: ignore[operator]
            state.u_prev + 3.0,  # type: ignore[operator]
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(backend_mod, "v2x2ssd_cute", fake_scan_op)
    try:
        backend = CuteScanBackend(compute_dtype=torch.float64)
        y, next_state = cast(
            tuple[torch.Tensor, ScanState],
            backend(inputs, chunk_size=4, state=state, return_state=True),
        )
    finally:
        monkeypatch.undo()

    assert calls["B"] is inputs.B
    assert calls["C"] is inputs.C
    assert calls["initial_states"] is state.state
    assert calls["B_prev"] is state.b_prev
    assert calls["U_prev"] is state.u_prev
    assert calls["return_state"] is True
    assert tuple(inputs.B.shape) == (2, bc_groups, 7, 8)
    assert tuple(inputs.C.shape) == (2, bc_groups, 7, 8)
    assert state.b_prev is not None
    assert tuple(state.b_prev.shape) == (2, bc_groups, 8)
    assert torch.equal(y, torch.zeros_like(inputs.U))
    assert next_state.state is not None
    assert next_state.b_prev is not None
    assert next_state.u_prev is not None
    torch.testing.assert_close(next_state.state, state.state + 1.0)  # type: ignore[operator]
    torch.testing.assert_close(next_state.b_prev, state.b_prev + 2.0)  # type: ignore[operator]
    torch.testing.assert_close(next_state.u_prev, state.u_prev + 3.0)  # type: ignore[operator]


def test_auto_backend_routes_cpu_inputs_to_reference() -> None:
    inputs, state = _make_inputs()
    backend = AutoScanBackend()
    calls: list[str] = []

    def fake_reference(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        assert routed_inputs is inputs
        assert chunk_size == 4
        assert state is not None
        assert return_state is True
        calls.append("reference")
        return torch.zeros_like(inputs.U), state

    def fake_cute(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        del routed_inputs, chunk_size, state, return_state
        calls.append("cute")
        raise AssertionError("CPU inputs should not route to the CuTe backend.")

    backend_any = cast(Any, backend)
    backend_any.reference = fake_reference
    backend_any.cute = fake_cute

    y, next_state = backend(inputs, chunk_size=4, state=state, return_state=True)

    assert calls == ["reference"]
    assert torch.equal(y, torch.zeros_like(inputs.U))
    assert next_state is state


def test_auto_backend_routes_cuda_inputs_to_cute() -> None:
    backend = AutoScanBackend()
    calls: list[str] = []
    zeros = torch.zeros((1,), dtype=torch.float32)

    class _CudaLikeTensor:
        def __init__(self) -> None:
            self.device = torch.device("cuda")

    inputs = ScanInputs(
        U=cast(torch.Tensor, _CudaLikeTensor()),
        M=zeros,
        K=zeros,
        B=zeros,
        C=zeros,
    )

    def fake_reference(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        del routed_inputs, chunk_size, state, return_state
        calls.append("reference")
        raise AssertionError("CUDA inputs should not route to the reference backend.")

    def fake_cute(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        del chunk_size, state
        assert routed_inputs is inputs
        assert return_state is False
        calls.append("cute")
        return cast(torch.Tensor, routed_inputs.U)

    backend_any = cast(Any, backend)
    backend_any.reference = fake_reference
    backend_any.cute = fake_cute

    y = backend(inputs, chunk_size=7, state=None)

    assert calls == ["cute"]
    assert y is inputs.U


def test_auto_backend_routes_stateful_cuda_inputs_to_cute() -> None:
    backend = AutoScanBackend()
    calls: list[str] = []
    zeros = torch.zeros((1,), dtype=torch.float32)
    state = ScanState(
        state=torch.randn((1, 1, 5, 8), dtype=torch.float32),
        b_prev=torch.randn((1, 1, 8), dtype=torch.float32),
        u_prev=torch.randn((1, 1, 5), dtype=torch.float32),
    )

    class _CudaLikeTensor:
        def __init__(self) -> None:
            self.device = torch.device("cuda")

    inputs = ScanInputs(
        U=cast(torch.Tensor, _CudaLikeTensor()),
        M=zeros,
        K=zeros,
        B=zeros,
        C=zeros,
    )

    def fake_reference(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ScanState]:
        del routed_inputs, chunk_size, state, return_state
        calls.append("reference")
        raise AssertionError("CUDA inputs should not route to the reference backend.")

    def fake_cute(
        routed_inputs: ScanInputs,
        *,
        chunk_size: int,
        state: ScanState | None = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, ScanState]:
        assert routed_inputs is inputs
        assert chunk_size == 7
        assert state is not None
        assert state is not None and state.state is not None
        assert return_state is True
        calls.append("cute")
        return zeros, state

    backend_any = cast(Any, backend)
    backend_any.reference = fake_reference
    backend_any.cute = fake_cute

    y, next_state = backend(inputs, chunk_size=7, state=state, return_state=True)

    assert calls == ["cute"]
    assert y is zeros
    assert next_state is state
