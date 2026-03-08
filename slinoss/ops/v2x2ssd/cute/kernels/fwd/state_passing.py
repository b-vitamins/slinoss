"""CuTe forward stage contract for ``state_passing``.

Logical tensors:
- ``inc``: ``(B, H, C, P, D)``
- ``m_chunk``: ``(B, H, C, 2)``
- ``chunk_starts``: ``(B, H, C, P, D)``
- ``final_state``: ``(B, H, P, D)``

Layout/partitioning intent:
- flatten ``S = P * D`` as the hot contiguous axis
- sequential over chunks, parallel over ``S``
- only the ``S`` tail requires predication
"""

from __future__ import annotations

import torch


def state_passing_cute(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del inc, m_chunk, initial_states, compute_dtype
    raise NotImplementedError("CuTe state_passing is not implemented yet.")


__all__ = ["state_passing_cute"]
