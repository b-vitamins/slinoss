"""Backward stage for the CuTe ``v2x2ssd`` chunk-increment operator."""

import torch

from .stage import chunk_increment_bwd_stage_cute


def chunk_increment_bwd_cute(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Canonical backward entrypoint for the public chunk-increment contract."""
    return chunk_increment_bwd_stage_cute(
        U,
        M,
        K,
        B,
        d_inc=d_inc,
        d_m_chunk=d_m_chunk,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


__all__ = ["chunk_increment_bwd_cute"]
