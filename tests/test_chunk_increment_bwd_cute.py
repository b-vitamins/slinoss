from __future__ import annotations

import math

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment import (
    chunk_increment_bwd_cute,
    chunk_increment_bwd_prepared_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    chunk_increment_with_prepared_cute,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment as reference_chunk_increment


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_inputs(
    *,
    batch: int,
    heads: int,
    T: int,
    N: int,
    P: int,
    device: torch.device,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
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

    b_prev = (
        torch.randn((batch, heads, N), device=device, dtype=torch.float32)
        + 1j * torch.randn((batch, heads, N), device=device, dtype=torch.float32)
    ) * 0.1
    B_prev = _pack_complex_pairs(b_prev, real_dtype=torch.float32)
    U_prev = torch.randn((batch, heads, P), device=device, dtype=torch.float32)
    return U, M, K, B, B_prev, U_prev


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_cute_matches_autograd() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )
    U.requires_grad_(True)
    M.requires_grad_(True)
    K.requires_grad_(True)
    B.requires_grad_(True)
    B_prev.requires_grad_(True)
    U_prev.requires_grad_(True)

    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)
    loss = (inc * d_inc).sum() + (m_chunk * d_m_chunk).sum()

    dU_ref, dM_ref, dK_ref, dB_ref, dB_prev_ref, dU_prev_ref = torch.autograd.grad(
        loss,
        (U, M, K, B, B_prev, U_prev),
    )

    dU_cute, dM_cute, dK_cute, dB_cute, dB_prev_cute, dU_prev_cute = (
        chunk_increment_bwd_cute(
            U.detach(),
            M.detach(),
            K.detach(),
            B.detach(),
            d_inc=d_inc.detach(),
            d_m_chunk=d_m_chunk.detach(),
            chunk_size=32,
            B_prev=B_prev.detach(),
            U_prev=U_prev.detach(),
            compute_dtype=torch.float32,
        )
    )

    torch.testing.assert_close(dU_cute, dU_ref, atol=5e-5, rtol=0.0)
    torch.testing.assert_close(dM_cute, dM_ref, atol=5e-5, rtol=0.0)
    torch.testing.assert_close(dK_cute, dK_ref, atol=5e-5, rtol=0.0)
    torch.testing.assert_close(dB_cute, dB_ref, atol=5e-5, rtol=0.0)
    torch.testing.assert_close(dB_prev_cute, dB_prev_ref, atol=5e-5, rtol=0.0)
    torch.testing.assert_close(dU_prev_cute, dU_prev_ref, atol=5e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_bwd_prepared_entrypoint_matches_public_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)

    U, M, K, B, B_prev, U_prev = _make_inputs(
        batch=2,
        heads=2,
        T=33,
        N=8,
        P=16,
        device=torch.device("cuda"),
    )

    inc, m_chunk = reference_chunk_increment(
        U,
        M,
        K,
        B,
        B_prev=B_prev,
        U_prev=U_prev,
        T=U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    d_inc = torch.randn_like(inc)
    d_m_chunk = torch.randn_like(m_chunk)

    got_public = chunk_increment_bwd_cute(
        U.detach(),
        M.detach(),
        K.detach(),
        B.detach(),
        d_inc=d_inc.detach(),
        d_m_chunk=d_m_chunk.detach(),
        chunk_size=32,
        B_prev=B_prev.detach(),
        U_prev=U_prev.detach(),
        compute_dtype=torch.float32,
    )

    _, _m_chunk, prepared = chunk_increment_with_prepared_cute(
        U.detach(),
        M.detach(),
        K.detach(),
        B.detach(),
        chunk_size=32,
        B_prev=B_prev.detach(),
        U_prev=U_prev.detach(),
        compute_dtype=torch.float32,
    )
    got_prepared = chunk_increment_bwd_prepared_cute(
        U.detach(),
        M.detach(),
        K.detach(),
        B.detach(),
        A_main=prepared.A_main,
        B_main=prepared.B_main,
        u_head=prepared.u_head,
        b_head=prepared.b_head,
        d_inc=d_inc.detach(),
        d_m_chunk=d_m_chunk.detach(),
        chunk_size=32,
        B_prev=B_prev.detach(),
        U_prev=U_prev.detach(),
        compute_dtype=torch.float32,
    )

    for got_tensor, want_tensor in zip(got_prepared, got_public, strict=True):
        torch.testing.assert_close(got_tensor, want_tensor, atol=0.0, rtol=0.0)
