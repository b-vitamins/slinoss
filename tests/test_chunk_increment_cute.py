from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _chunk_increment_from_prepared_operands,
    _prepare_chunk_increment_operands,
    chunk_increment_cute,
)
from slinoss.ops.v2x2ssd.reference import chunk_increment as reference_chunk_increment


@dataclass(frozen=True)
class Inputs:
    U: torch.Tensor
    M: torch.Tensor
    K: torch.Tensor
    B: torch.Tensor
    B_prev: torch.Tensor | None
    U_prev: torch.Tensor | None


def _pack_complex_pairs(z: torch.Tensor, *, real_dtype: torch.dtype) -> torch.Tensor:
    return (
        torch.view_as_real(z)
        .reshape(*z.shape[:-1], z.shape[-1] * 2)
        .to(dtype=real_dtype)
        .contiguous()
    )


def _make_inputs(
    *,
    b: int,
    h: int,
    T: int,
    N: int,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
    streaming: bool,
) -> Inputs:
    radius = 0.6 + 0.35 * torch.rand((b, h, T), device=device, dtype=torch.float32)
    angle = (2.0 * math.pi) * torch.rand(
        (b, h, T), device=device, dtype=torch.float32
    ) - math.pi
    M = torch.view_as_real(torch.polar(radius, angle)).to(dtype=dtype).contiguous()

    K_complex = (
        torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
        + 1j * torch.randn((b, h, T, 2), device=device, dtype=torch.float32)
    ) * 0.1
    K = torch.view_as_real(K_complex).to(dtype=dtype).contiguous()

    U = torch.randn((b, h, T, P), device=device, dtype=dtype)
    B = torch.randn((b, h, T, 2 * N), device=device, dtype=dtype) * 0.1

    B_prev = None
    U_prev = None
    if streaming:
        b_prev = (
            torch.randn((b, h, N), device=device, dtype=torch.float32)
            + 1j * torch.randn((b, h, N), device=device, dtype=torch.float32)
        ) * 0.1
        B_prev = _pack_complex_pairs(b_prev, real_dtype=dtype)
        U_prev = torch.randn((b, h, P), device=device, dtype=dtype)

    return Inputs(U=U, M=M, K=K, B=B, B_prev=B_prev, U_prev=U_prev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_cute_matches_reference_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=True,
    )

    inc_ref, m_ref = reference_chunk_increment(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        T=inputs.U.shape[2],
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    inc_cute, m_cute = chunk_increment_cute(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        chunk_size=32,
        compute_dtype=torch.float32,
    )

    torch.testing.assert_close(inc_cute, inc_ref, atol=2e-5, rtol=0.0)
    torch.testing.assert_close(m_cute, m_ref, atol=2e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chunk_increment_prepared_entrypoint_matches_public_stage() -> None:
    pytest.importorskip("cutlass")
    torch.manual_seed(0)
    inputs = _make_inputs(
        b=2,
        h=2,
        T=33,
        N=8,
        P=16,
        dtype=torch.float32,
        device=torch.device("cuda"),
        streaming=True,
    )

    inc_public, m_public = chunk_increment_cute(
        inputs.U,
        inputs.M,
        inputs.K,
        inputs.B,
        B_prev=inputs.B_prev,
        U_prev=inputs.U_prev,
        chunk_size=32,
        compute_dtype=torch.float32,
    )
    A_main, B_main, u_head, b_head, m_chunk, batch_size, n_heads, n_chunks, P = (
        _prepare_chunk_increment_operands(
            inputs.U,
            inputs.M,
            inputs.K,
            inputs.B,
            chunk_size=32,
            B_prev=inputs.B_prev,
            U_prev=inputs.U_prev,
            compute_dtype=torch.float32,
        )
    )
    inc_prepared, m_prepared = _chunk_increment_from_prepared_operands(
        A_main,
        B_main,
        u_head,
        b_head,
        m_chunk,
        batch_size=batch_size,
        n_heads=n_heads,
        n_chunks=n_chunks,
        P=P,
    )

    torch.testing.assert_close(inc_prepared, inc_public, atol=0.0, rtol=0.0)
    torch.testing.assert_close(m_prepared, m_public, atol=0.0, rtol=0.0)
