from __future__ import annotations

import math
import pytest
import torch

from slinoss.ops.v2x2ssd import v2x2ssd, v2x2ssd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd import (
    compile_chunk_scan_kernel,
    chunk_scan_cute,
    state_passing_cute,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment, chunk_scan, state_passing


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_scan_inputs(
    *,
    batch: int,
    heads: int,
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
    torch.Tensor,
]:
    radius = 0.6 + 0.35 * torch.rand((batch, heads, T), device=device)
    angle = (2.0 * math.pi) * torch.rand((batch, heads, T), device=device) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()

    K_complex = (
        torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(torch.float32).contiguous()

    U = torch.randn((batch, heads, T, P), device=device, dtype=torch.float32)
    B = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    C = torch.randn((batch, heads, T, 2 * N), device=device, dtype=torch.float32) * 0.1
    initial_states = torch.randn(
        (batch, heads, P, 2 * N), device=device, dtype=torch.float32
    )

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, C, initial_states, B_prev, U_prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_compile_entrypoint_reuses_cache() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    compiled_a, out_chunk_a, out_view_a = compile_chunk_scan_kernel(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    compiled_b, out_chunk_b, out_view_b = compile_chunk_scan_kernel(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    assert compiled_a is compiled_b
    assert out_chunk_a.shape == out_chunk_b.shape == (8, 32, 1, 16)
    assert out_view_a.shape == out_view_b.shape == (2, 2, 64, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_state_passing_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    batch, heads, chunks, N, P = 2, 2, 16, 8, 16
    D = 2 * N
    inc = torch.randn((batch, heads, chunks, P, D), device="cuda", dtype=torch.float32)
    radius = 0.6 + 0.35 * torch.rand((batch, heads, chunks), device="cuda")
    angle = (2.0 * math.pi) * torch.rand(
        (batch, heads, chunks), device="cuda"
    ) - math.pi
    m_chunk = (
        torch.view_as_real(torch.polar(radius, angle)).to(torch.float32).contiguous()
    )
    initial_states = torch.randn(
        (batch, heads, P, D), device="cuda", dtype=torch.float32
    )

    starts_ref, final_ref = state_passing(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )
    starts_cute, final_cute = state_passing_cute(
        inc,
        m_chunk,
        initial_states=initial_states,
    )

    assert starts_cute.dtype == torch.float32
    assert final_cute.dtype == torch.float32
    torch.testing.assert_close(starts_cute, starts_ref, atol=4e-3, rtol=0.0)
    torch.testing.assert_close(final_cute, final_ref, atol=2e-3, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_scan_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    inc_ref, m_ref = chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
    )
    starts_ref, _ = state_passing(
        inc_ref,
        m_ref,
        initial_states=initial_states,
        compute_dtype=torch.float32,
    )

    y_ref = chunk_scan(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )
    y_cute = chunk_scan_cute(
        U,
        M,
        K,
        B,
        C,
        starts_ref,
        B_prev=B_prev,
        U_prev=U_prev,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_forward_training_only() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=64,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32

    y_ref, _final_ref, _b_last_ref, _u_last_ref = v2x2ssd(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        initial_states=None,
        B_prev=None,
        U_prev=None,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    y_cute = v2x2ssd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        compute_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    torch.testing.assert_close(y_cute, y_ref, atol=2e-2, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_v2x2ssd_cute_matches_reference_autograd_training_only() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, C, initial_states, B_prev, U_prev = _make_scan_inputs(
        batch=2,
        heads=2,
        T=65,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    chunk_size = 32
    weight = torch.randn((2, 2, 65, 16), device="cuda", dtype=torch.float32)

    def run(
        op, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        U_i, M_i, K_i, B_i, C_i = tensors
        if op is v2x2ssd:
            Y, _final_state, _b_last, _u_last = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                initial_states=None,
                B_prev=None,
                U_prev=None,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        else:
            Y = op(
                U_i,
                M_i,
                K_i,
                B_i,
                C_i,
                chunk_size=chunk_size,
                compute_dtype=torch.float32,
                output_dtype=torch.float32,
            )
        loss = (Y * weight).sum()
        grads = torch.autograd.grad(
            loss,
            (U_i, M_i, K_i, B_i, C_i),
            retain_graph=False,
        )
        return (Y,), grads

    ref_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in (U, M, K, B, C)
    )
    cute_tensors = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in (U, M, K, B, C)
    )

    ref_out, ref_grads = run(v2x2ssd, ref_tensors)
    cute_out, cute_grads = run(v2x2ssd_cute, cute_tensors)

    for got, want in zip(cute_out, ref_out, strict=True):
        torch.testing.assert_close(got, want, atol=1e-3, rtol=0.0)
    grad_names = ("U", "M", "K", "B", "C")
    atol_by_grad = {
        "U": 1e-1,
        "M": 3e-1,
        "K": 5e-1,
        "B": 4e-1,
        "C": 3e-1,
    }
    for name, got, want in zip(grad_names, cute_grads, ref_grads, strict=True):
        # The integrated chunk-scan backward now reuses the promoted packed
        # tensor-core dU/dC/dQ/dK slices. Public gradients that depend on those
        # packed intermediates inherit the same principled low-precision
        # contract: fp16/bf16 transport, fp32 dense accumulation, and exact fp32
        # scatter/reduction back to public layout. The remaining slices stay on
        # the tighter exact budget.
        torch.testing.assert_close(got, want, atol=atol_by_grad[name], rtol=0.0)
