"""CuTe entry point for the v2x2ssd operator."""

from __future__ import annotations

import torch


def v2x2ssd_cute(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe-backed staged v2x2ssd forward path.

    Contract:
    - logical shapes and layouts match ``reference.v2x2ssd`` exactly
    - the CuTe path preserves the staged decomposition
      ``chunk_increment -> state_passing -> chunk_scan``
    - no model-side layout conversion or alternate public tensor contract is
      introduced here
    """

    del U, M, K, B, C
    del chunk_size, initial_states, B_prev, U_prev, compute_dtype, output_dtype

    # CuTe is a real runtime dependency for this path. Import it here so the
    # rest of the repo remains usable without the kernel toolchain.
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    from .kernels.fwd.chunk_increment import chunk_increment_cute
    from .kernels.fwd.chunk_scan import chunk_scan_cute
    from .kernels.fwd.state_passing import state_passing_cute

    del chunk_increment_cute, state_passing_cute, chunk_scan_cute

    raise NotImplementedError(
        "The CuTe v2x2ssd entry point is wired, but the forward stage kernels "
        "have not been implemented yet."
    )


__all__ = ["v2x2ssd_cute"]
