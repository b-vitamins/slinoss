"""CuTe forward stage contract for ``chunk_scan``.

Logical tensors:
- ``U``: ``(B, H, T, P)``
- ``M``: ``(B, H, T, 2)``
- ``K``: ``(B, H, T, 2, 2)``
- ``B, C``: ``(B, H, T, D)``
- ``chunk_starts``: ``(B, H, C, P, D)``

Layout/partitioning intent:
- CTA owns one ``(bhc, chunk)`` tile
- thread/value layout is 2D over ``(P_tile, D_tile)``
- causal, ``L`` tail, ``P`` tail, and ``D`` tail are explicitly predicated
- segment ratios must be formed directly; avoid reciprocal row-scale
  factorizations that invite non-finite behavior
"""

from __future__ import annotations

import torch


def chunk_scan_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del U, M, K, B, C, chunk_starts
    del chunk_size, B_prev, U_prev, output_dtype, compute_dtype
    raise NotImplementedError("CuTe chunk_scan is not implemented yet.")


__all__ = ["chunk_scan_cute"]
