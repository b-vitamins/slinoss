"""CuTe backward for the ``v2x2ssd`` chunk-increment stage.

Logical contract
----------------
- inputs:
  - ``U``: ``(B, H, T, P)``
  - ``M``: ``(B, H, T, 2)``
  - ``K``: ``(B, H, T, 2, 2)``
  - ``B``: ``(B, H, T, D)`` with ``D = 2N`` interleaved complex lanes
  - ``B_prev``: ``(B, H, D)``
  - ``U_prev``: ``(B, H, P)``
  - ``d_inc``: ``(B, H, C, P, D)``
  - ``d_m_chunk``: ``(B, H, C, 2)``
- outputs:
  - ``dU``: ``(B, H, T, P)``
  - ``dM``: ``(B, H, T, 2)``
  - ``dK``: ``(B, H, T, 2, 2)``
  - ``dB``: ``(B, H, T, D)``
  - ``dB_prev``: ``(B, H, D)``
  - ``dU_prev``: ``(B, H, P)``

Layout / launch
---------------
The dense backward contractions reuse the forward SGEMM kernel on prepared
operand views:

- ``dA_main = d_inc @ B_main^T`` via one batched GEMM
- ``dB_main = A_main @ d_inc`` via a second batched GEMM

Those prepared views keep the same feature-major stride contract as the forward
kernel, so no new CuTe layout family is introduced for this stage.

Numerical contract
------------------
The remaining backward pass through the operand preparation uses explicit
complex arithmetic in Torch without reciprocal prefix factors. Gradients through
``m_chunk`` and the suffix products are propagated by a direct scan over the
short chunk axis, so there is no division-by-zero hazard here.
"""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import (
    _get_compiled_gemm,
    _mark_output,
    _mark_prepared_input,
    _prepare_chunk_increment_operands,
)
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _complex_dtype_from_real,
    _pack_complex_pairs,
    _pad_time_partial,
    _resolve_dtypes,
    _resolve_prev0,
    _to_complex_scalar,
    _to_complex_taps,
    _validate_chunk_increment_inputs,
)

from .tail import chunk_increment_bwd_tail_exact_cute


def _prepared_dA_main_cute(
    d_inc_flat: torch.Tensor,
    B_main: torch.Tensor,
) -> torch.Tensor:
    """Compute ``dA_main`` in ``(BHC, L, P)`` by reusing the forward SGEMM."""
    dinc_t = d_inc_flat.transpose(1, 2).contiguous()
    B_main_t = B_main.transpose(1, 2).contiguous()

    A3 = dinc_t.permute(2, 1, 0)
    B3 = B_main_t.permute(2, 1, 0)
    dA_buf = torch.empty(
        (d_inc_flat.shape[0], d_inc_flat.shape[1], B_main.shape[1]),
        device=d_inc_flat.device,
        dtype=torch.float32,
    )
    C3 = dA_buf.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    compiled(_mark_prepared_input(A3), _mark_prepared_input(B3), _mark_output(C3))
    return dA_buf.transpose(1, 2).contiguous()


def _prepared_dB_main_cute(
    A_main: torch.Tensor,
    d_inc_flat: torch.Tensor,
) -> torch.Tensor:
    """Compute ``dB_main`` in ``(BHC, L, D)`` by reusing the forward SGEMM."""
    A_main_t = A_main.transpose(1, 2).contiguous()

    A3 = A_main_t.permute(2, 1, 0)
    B3 = d_inc_flat.permute(2, 1, 0)
    dB_main = torch.empty(
        (A_main.shape[0], A_main.shape[1], d_inc_flat.shape[-1]),
        device=A_main.device,
        dtype=torch.float32,
    )
    C3 = dB_main.permute(1, 2, 0)

    compiled = _get_compiled_gemm(A3, B3, C3)
    compiled(_mark_prepared_input(A3), _mark_prepared_input(B3), _mark_output(C3))
    return dB_main


def chunk_increment_bwd_stage_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    d_inc: torch.Tensor,
    d_m_chunk: torch.Tensor,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Low-level stage implementation for ``chunk_increment`` backward."""
    batch_size, n_heads, T, N, P = _validate_chunk_increment_inputs(
        U, M, K, B, B_prev, U_prev, int(U.shape[2])
    )
    if U.device.type != "cuda":
        raise ValueError("CuTe chunk_increment backward requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")
    D = 2 * N

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, d_inc.dtype, d_m_chunk.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe chunk_increment backward supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    for name, tensor in (
        ("d_inc", d_inc),
        ("d_m_chunk", d_m_chunk),
    ):
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; got strides {tensor.stride()}."
            )

    n_chunks = (T + int(chunk_size) - 1) // int(chunk_size)
    if d_inc.shape != (batch_size, n_heads, n_chunks, P, D):
        raise ValueError(
            "d_inc must be (batch, heads, chunks, P, 2N) = "
            f"{(batch_size, n_heads, n_chunks, P, D)}. Got {tuple(d_inc.shape)}."
        )
    if d_m_chunk.shape != (batch_size, n_heads, n_chunks, 2):
        raise ValueError(
            "d_m_chunk must be (batch, heads, chunks, 2) = "
            f"{(batch_size, n_heads, n_chunks, 2)}. Got {tuple(d_m_chunk.shape)}."
        )

    A_main, B_main, u_head, b_head, _, _, _, _, _ = _prepare_chunk_increment_operands(
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=rdtype,
    )
    BHC = int(A_main.shape[0])
    d_inc_flat = d_inc.reshape(BHC, P, D).contiguous()

    dA_main = _prepared_dA_main_cute(d_inc_flat, B_main)
    dB_main = _prepared_dB_main_cute(A_main, d_inc_flat)
    d_u_head = torch.einsum("bpd,bd->bp", d_inc_flat, b_head)
    d_b_head = torch.einsum("bpd,bp->bd", d_inc_flat, u_head)

    cplx_dtype = _complex_dtype_from_real(rdtype)
    device = U.device
    U_f, M_f, K_f, B_f, _, n_chunks = _pad_time_partial(
        U, M, K, B, chunk_size=chunk_size, real_dtype=rdtype
    )
    L = int(chunk_size)
    T_pad = int(U_f.shape[2])

    m = _to_complex_scalar(M_f, name="M").to(dtype=cplx_dtype)
    k = _to_complex_taps(K_f, name="K").to(dtype=cplx_dtype)
    k_prev, k_curr = k[..., 0], k[..., 1]
    b_t = _as_complex_pairs(B_f, name="B").to(dtype=cplx_dtype)

    b_prev0, u_prev0 = _resolve_prev0(
        B_prev,
        U_prev,
        batch_size=batch_size,
        n_heads=n_heads,
        D=D,
        P=P,
        device=device,
        real_dtype=rdtype,
        complex_dtype=cplx_dtype,
    )

    m_blk = m.reshape(batch_size, n_heads, n_chunks, L)
    k_prev_blk = k_prev.reshape(batch_size, n_heads, n_chunks, L)
    k_curr_blk = k_curr.reshape(batch_size, n_heads, n_chunks, L)
    b_blk = b_t.reshape(batch_size, n_heads, n_chunks, L, N)

    suffix_after = torch.ones_like(m_blk)
    if L > 1:
        suffix_after[..., :-1] = torch.flip(
            torch.cumprod(torch.flip(m_blk[..., 1:], dims=(-1,)), dim=-1),
            dims=(-1,),
        )

    d_alpha = _as_complex_pairs(
        dB_main.reshape(batch_size, n_heads, n_chunks, L, D), name="dB_main"
    ).to(dtype=cplx_dtype)
    d_boundary = _as_complex_pairs(
        d_b_head.reshape(batch_size, n_heads, n_chunks, D), name="d_b_head"
    ).to(dtype=cplx_dtype)

    b_prev_chunk0 = torch.empty(
        (batch_size, n_heads, n_chunks, N), device=device, dtype=cplx_dtype
    )
    b_prev_chunk0[:, :, 0, :] = b_prev0
    if n_chunks > 1:
        b_prev_chunk0[:, :, 1:, :] = b_blk[:, :, :-1, -1, :]

    dU_blk = dA_main.reshape(batch_size, n_heads, n_chunks, L, P)
    dU_prev_blk = d_u_head.reshape(batch_size, n_heads, n_chunks, P)
    dB_blk = torch.zeros(
        (batch_size, n_heads, n_chunks, L, N), device=device, dtype=cplx_dtype
    )
    dB_blk += (
        torch.conj(suffix_after.unsqueeze(-1) * k_curr_blk.unsqueeze(-1)) * d_alpha
    )

    if L > 1:
        d_alpha_shift = d_alpha[..., :-1, :]
        dB_blk[..., :-1, :] += (
            torch.conj(
                suffix_after[..., 1:].unsqueeze(-1) * k_prev_blk[..., 1:].unsqueeze(-1)
            )
            * d_alpha_shift
        )

    d_b_prev_chunk0 = (
        torch.conj(
            suffix_after[..., 0].unsqueeze(-1) * k_prev_blk[..., 0].unsqueeze(-1)
        )
        * d_boundary
    )
    dK_prev_blk_r, dK_curr_blk_r, _d_suffix_after_r, dM_blk_r = (
        chunk_increment_bwd_tail_exact_cute(
            torch.view_as_real(suffix_after.reshape(BHC, L)).to(dtype=rdtype).contiguous(),
            torch.view_as_real(k_prev_blk.reshape(BHC, L)).to(dtype=rdtype).contiguous(),
            torch.view_as_real(k_curr_blk.reshape(BHC, L)).to(dtype=rdtype).contiguous(),
            _pack_complex_pairs(
                b_blk.reshape(batch_size * n_heads * n_chunks, L, N),
                real_dtype=rdtype,
            ).reshape(batch_size * n_heads * n_chunks, L, D).contiguous(),
            _pack_complex_pairs(
                b_prev_chunk0.reshape(batch_size * n_heads * n_chunks, N),
                real_dtype=rdtype,
            ).reshape(batch_size * n_heads * n_chunks, D).contiguous(),
            _pack_complex_pairs(
                d_alpha.reshape(batch_size * n_heads * n_chunks, L, N),
                real_dtype=rdtype,
            ).reshape(batch_size * n_heads * n_chunks, L, D).contiguous(),
            _pack_complex_pairs(
                d_boundary.reshape(batch_size * n_heads * n_chunks, N),
                real_dtype=rdtype,
            ).reshape(batch_size * n_heads * n_chunks, D).contiguous(),
            torch.view_as_real(m_blk.reshape(BHC, L)).to(dtype=rdtype).contiguous(),
            d_m_chunk.reshape(batch_size * n_heads * n_chunks, 2)
            .to(dtype=rdtype)
            .contiguous(),
        )
    )
    dK_prev_blk = torch.view_as_complex(dK_prev_blk_r.contiguous()).reshape_as(k_prev_blk)
    dK_curr_blk = torch.view_as_complex(dK_curr_blk_r.contiguous()).reshape_as(k_curr_blk)

    dU_prev = dU_prev_blk[:, :, 0, :].contiguous()
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += dU_prev_blk[:, :, 1:, :]

    dB_prev_c = d_b_prev_chunk0[:, :, 0, :].contiguous()
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_b_prev_chunk0[:, :, 1:, :]

    dM_blk = torch.view_as_complex(dM_blk_r.contiguous()).reshape_as(m_blk)

    dU = dU_blk.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous()
    dM = torch.view_as_real(dM_blk.reshape(batch_size, n_heads, T_pad))
    dM = dM.to(dtype=rdtype)[:, :, :T, :].contiguous()

    dK_blk = torch.stack((dK_prev_blk, dK_curr_blk), dim=-1)
    dK = torch.view_as_real(dK_blk.reshape(batch_size, n_heads, T_pad, 2))
    dK = dK.reshape(batch_size, n_heads, T_pad, 2, 2).to(dtype=rdtype)
    dK = dK[:, :, :T, :, :].contiguous()

    dB = _pack_complex_pairs(
        dB_blk.reshape(batch_size, n_heads, T_pad, N), real_dtype=rdtype
    )
    dB = dB[:, :, :T, :].contiguous()
    dB_prev = _pack_complex_pairs(dB_prev_c, real_dtype=rdtype)

    return dU, dM, dK, dB, dB_prev, dU_prev.to(dtype=rdtype).contiguous()


__all__ = ["chunk_increment_bwd_stage_cute"]
