"""CuTe forward stage contract for ``chunk_increment``.

Logical tensors:
- ``U``: ``(BHC, L, P)``
- ``M``: ``(BHC, L, 2)``
- ``K``: ``(BHC, L, 2, 2)``
- ``B``: ``(BHC, L, D)`` with ``D = 2N`` interleaved complex lanes

Layout/partitioning intent:
- CTA owns one ``(bhc, chunk)`` tile
- ``P`` and ``D`` are the hot contiguous axes
- thread/value layout is 2D over ``(P_tile, D_tile)``
- ``L``, ``P``, and ``D`` tails are explicitly predicated
"""

from __future__ import annotations

import torch


def chunk_increment_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del U, M, K, B, chunk_size, B_prev, U_prev, compute_dtype
    raise NotImplementedError("CuTe chunk_increment is not implemented yet.")


__all__ = ["chunk_increment_cute"]
