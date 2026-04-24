from __future__ import annotations

from types import SimpleNamespace
from typing import cast
import math

import pytest
import torch
import cutlass.cute as cute

import slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan as chunk_scan_bwd_mod
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan import (
    chunk_scan_bwd_cute,
    compile_chunk_scan_bwd_kernels,
)
from slinoss.ops.v2x2ssd.reference import (
    chunk_increment,
    chunk_scan as ref_chunk_scan,
    state_passing,
)


def _public_from_chunked(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, L, F = map(int, x.shape)
    return x.reshape(B, H, C * L, F)[:, :, :T, :].to(dtype=torch.float32).contiguous()


def _fold_chunk_boundary_carries(x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    if int(out.shape[2]) > 1:
        out[:, :, :-1, -1, :] = out[:, :, :-1, -1, :] + x_prev[:, :, 1:, :]
    return out


def _public_from_param_scan(x: torch.Tensor, *, T: int) -> torch.Tensor:
    B, H, C, S, L, F = map(int, x.shape)
    assert S == 1
    return (
        x[:, :, :, 0, :, :]
        .reshape(B, H, C * L, F)[:, :, :T, :]
        .to(dtype=torch.float32)
        .contiguous()
    )


def _public_dk_from_parts(
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    *,
    T: int,
) -> torch.Tensor:
    dKprev_public = _public_from_param_scan(dKprev, T=T)
    dKcurr_public = _public_from_param_scan(dKcurr, T=T)
    return torch.stack((dKprev_public, dKcurr_public), dim=3).contiguous()


def _make_inputs(
    *,
    batch: int,
    heads: int,
    bc_groups: int | None = None,
    T: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bc_groups = heads if bc_groups is None else bc_groups
    assert heads % bc_groups == 0

    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    B = (
        torch.randn((batch, bc_groups, T, 2 * N), device=device, dtype=torch.float32)
        * 0.1
    )
    C = (
        torch.randn((batch, bc_groups, T, 2 * N), device=device, dtype=torch.float32)
        * 0.1
    )
    B_prev = (
        torch.randn((batch, bc_groups, 2 * N), device=device, dtype=torch.float32) * 0.1
    )
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, B_prev, U_prev


def _reference_du_grads(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    B_prev: torch.Tensor,
    U_prev: torch.Tensor,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    U_ref = U.detach().clone().requires_grad_(True)
    U_prev_ref = U_prev.detach().clone().requires_grad_(True)
    y_ref = ref_chunk_scan(
        U_ref,
        M,
        K,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev_ref,
        T=T,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    loss = (y_ref * d_out).sum()
    dU_ref, dU_prev_ref = torch.autograd.grad(loss, (U_ref, U_prev_ref))
    return (
        dU_ref.to(dtype=torch.float32).contiguous(),
        dU_prev_ref.to(dtype=torch.float32).contiguous(),
    )


def _reference_boundary_grads(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    B_prev: torch.Tensor,
    U_prev: torch.Tensor,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_starts_ref = chunk_starts.detach().clone().requires_grad_(True)
    B_prev_ref = B_prev.detach().clone().requires_grad_(True)
    U_prev_ref = U_prev.detach().clone().requires_grad_(True)
    y_ref = ref_chunk_scan(
        U,
        M,
        K,
        B,
        C,
        chunk_starts_ref,
        B_prev=B_prev_ref,
        U_prev=U_prev_ref,
        T=T,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    loss = (y_ref * d_out).sum()
    dZ0_ref, dB_prev_ref, dU_prev_ref = torch.autograd.grad(
        loss, (chunk_starts_ref, B_prev_ref, U_prev_ref)
    )
    return (
        dZ0_ref.to(dtype=torch.float32).contiguous(),
        dB_prev_ref.to(dtype=torch.float32).contiguous(),
        dU_prev_ref.to(dtype=torch.float32).contiguous(),
    )


def _reference_key_grads(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    B_prev: torch.Tensor,
    U_prev: torch.Tensor,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B_ref = B.detach().clone().requires_grad_(True)
    B_prev_ref = B_prev.detach().clone().requires_grad_(True)
    y_ref = ref_chunk_scan(
        U,
        M,
        K,
        B_ref,
        C,
        chunk_starts,
        B_prev=B_prev_ref,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    loss = (y_ref * d_out).sum()
    dB_ref, dB_prev_ref = torch.autograd.grad(loss, (B_ref, B_prev_ref))
    return (
        dB_ref.to(dtype=torch.float32).contiguous(),
        dB_prev_ref.to(dtype=torch.float32).contiguous(),
    )


def _reference_param_grads(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    B_prev: torch.Tensor,
    U_prev: torch.Tensor,
    T: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M_ref = M.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    y_ref = ref_chunk_scan(
        U,
        M_ref,
        K_ref,
        B,
        C,
        chunk_starts,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    loss = (y_ref * d_out).sum()
    dM_ref, dK_ref = torch.autograd.grad(loss, (M_ref, K_ref))
    return (
        dM_ref.to(dtype=torch.float32).contiguous(),
        dK_ref.to(dtype=torch.float32).contiguous(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_compile_entrypoint_matches_public_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    got_public = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    prepared = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    prepared.launch()

    got_compiled = prepared.outputs.public_outputs(
        T=T,
        value_dtype=U.dtype,
        key_dtype=B.dtype,
        query_dtype=C.dtype,
        return_prev_grads=True,
    )

    atol_by_slot = (
        0.0,
        5e-7,
        2e-7,
        2e-7,
        0.0,
        0.0,
        2e-7,
        0.0,
    )
    for got_tensor, want_tensor, atol in zip(
        got_compiled, got_public, atol_by_slot, strict=True
    ):
        torch.testing.assert_close(got_tensor, want_tensor, atol=atol, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("N", "P"), ((16, 64), (16, 96), (16, 128)))
def test_chunk_scan_bwd_matches_reference_when_value_axis_exceeds_state_axis(
    N: int,
    P: int,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T = 1, 1, 97
    chunk_size = 64
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    dU_ref, dU_prev_ref = _reference_du_grads(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
    )

    # Keep this as a short same-process repeatability check: the DU carry path
    # used to corrupt boundary rows intermittently when its shared carry buffer
    # was overwritten before every thread had finished reading it.
    for _attempt in range(3):
        prepared = compile_chunk_scan_bwd_kernels(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )
        prepared.launch()

        public_outputs = prepared.outputs.public_outputs(
            T=T,
            value_dtype=U.dtype,
            key_dtype=B.dtype,
            query_dtype=C.dtype,
            return_prev_grads=True,
        )
        dU_public = cast(torch.Tensor, public_outputs[0])
        dU_prev_public = cast(torch.Tensor, public_outputs[-1]).to(dtype=torch.float32)

        # Float32 stage inputs now preserve bf16 tensor-core staging instead of
        # narrowing to fp16, so the DU path follows a slightly different but
        # still well-behaved low-precision contract than the old test budget.
        torch.testing.assert_close(dU_public, dU_ref, atol=1e-3, rtol=0.0)
        torch.testing.assert_close(dU_prev_public, dU_prev_ref, atol=6e-4, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_param_outputs_match_reference() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    dM_ref, dK_ref = _reference_param_grads(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
    )

    _dU, dM, dK, _dB, _dC, _dZ0, _dB_prev, _dU_prev = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(dM, dM_ref, atol=7e-4, rtol=0.0)
    torch.testing.assert_close(dK, dK_ref, atol=1e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_matches_reference_dz0_for_realistic_stateful_shape() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 4, 17, 64, 64
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    dZ0_ref, dB_prev_ref, dU_prev_ref = _reference_boundary_grads(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
    )

    _dU, _dM, _dK, _dB, _dC, dZ0, dB_prev, dU_prev = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(dZ0, dZ0_ref, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(dB_prev, dB_prev_ref, atol=2e-3, rtol=0.0)
    torch.testing.assert_close(dU_prev, dU_prev_ref, atol=2e-4, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_matches_reference_db_for_multi_d_stage_shape() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 1, 1, 64, 64, 64
    chunk_size = 64
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    dB_ref, dB_prev_ref = _reference_key_grads(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
    )
    _dU, _dM, _dK, dB, _dC, _dZ0, dB_prev, _dU_prev = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    dB_error = (dB.to(dtype=torch.float32) - dB_ref).abs()
    assert float(dB_error.max()) < 1.25
    assert float(dB_error.mean()) < 0.05
    torch.testing.assert_close(
        dB_prev.to(dtype=torch.float32), dB_prev_ref, atol=2e-3, rtol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_cute_preserves_grouped_bc_public_contract() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, bc_groups, T, N, P = 1, 4, 2, 33, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        bc_groups=bc_groups,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    got_public = chunk_scan_bwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    prepared = compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    prepared.launch()
    got_compiled = prepared.outputs.public_outputs(
        T=T,
        value_dtype=U.dtype,
        key_dtype=B.dtype,
        query_dtype=C.dtype,
        return_prev_grads=True,
    )

    dB = cast(torch.Tensor, got_public[3])
    dC = cast(torch.Tensor, got_public[4])
    dB_prev = cast(torch.Tensor, got_public[6])
    dB_compiled = cast(torch.Tensor, got_compiled[3])
    dC_compiled = cast(torch.Tensor, got_compiled[4])
    dB_prev_compiled = cast(torch.Tensor, got_compiled[6])

    assert dB.shape == (batch, bc_groups, T, 2 * N)
    assert dC.shape == (batch, bc_groups, T, 2 * N)
    assert dB_prev.shape == (batch, bc_groups, 2 * N)
    assert dB_compiled.shape == (batch, bc_groups, T, 2 * N)
    assert dC_compiled.shape == (batch, bc_groups, T, 2 * N)
    assert dB_prev_compiled.shape == (batch, bc_groups, 2 * N)

    torch.testing.assert_close(dB_compiled, dB, atol=2e-7, rtol=0.0)
    torch.testing.assert_close(dC_compiled, dC, atol=0.0, rtol=0.0)
    torch.testing.assert_close(dB_prev_compiled, dB_prev, atol=2e-7, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_compile_entrypoint_reuses_cached_executors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    chunk_scan_bwd_mod._COMPILED_CACHE.clear()
    compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )

    def _unexpected_compile(*args, **kwargs):
        raise AssertionError("unexpected recompilation on cache hit")

    monkeypatch.setattr(chunk_scan_bwd_mod.cute, "compile", _unexpected_compile)
    compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=torch.float32,
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_compile_entrypoint_enables_tvm_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    compile_options: list[object] = []
    orig_compile = cute.compile

    def wrapped_compile(*args, **kwargs):
        compile_options.append(kwargs.get("options"))
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", wrapped_compile)
    chunk_scan_bwd_mod._COMPILED_CACHE.clear()
    try:
        compile_chunk_scan_bwd_kernels(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )
    finally:
        chunk_scan_bwd_mod._COMPILED_CACHE.clear()

    assert compile_options == ["--enable-tvm-ffi"] * 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_rejects_oversized_dcdr_shapes_before_launch() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 1, 1, 65, 1024, 512
    chunk_size = 64
    n_chunks = (T + chunk_size - 1) // chunk_size
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    chunk_starts = torch.zeros(
        (batch, heads, n_chunks, P, 2 * N),
        device=device,
        dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    with pytest.raises(
        ValueError,
        match=r"No supported chunk_scan backward dcdr kernel fits",
    ):
        compile_chunk_scan_bwd_kernels(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_rejects_unsupported_db_shapes_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    monkeypatch.setattr(
        chunk_scan_bwd_mod.ChunkScanBwdDBAmpere,
        "support_info",
        lambda self, in_dtype, *, device_index=None: SimpleNamespace(
            supported=False,
            required_smem_bytes=123456,
            smem_capacity_bytes=65536,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"No supported chunk_scan backward db kernel fits",
    ):
        compile_chunk_scan_bwd_kernels(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_bwd_rejects_unsupported_du_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, T, N, P = 2, 2, 65, 8, 16
    chunk_size = 32
    device = torch.device("cuda")

    U, M, K, B, C, B_prev, U_prev = _make_inputs(
        batch=batch,
        heads=heads,
        T=T,
        N=N,
        P=P,
        device=device,
    )
    inc, m_chunk = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=T,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    chunk_starts, _ = state_passing(
        inc,
        m_chunk,
        initial_states=None,
        compute_dtype=torch.float32,
    )
    d_out = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)

    monkeypatch.setattr(
        chunk_scan_bwd_mod.ChunkScanBwdDUAmpere,
        "support_info",
        lambda self, in_dtype, *, device_index=None: SimpleNamespace(
            supported=False,
            required_smem_bytes=131072,
            smem_capacity_bytes=65536,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"No supported chunk_scan backward du kernel fits",
    ):
        compile_chunk_scan_bwd_kernels(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=torch.float32,
        )
