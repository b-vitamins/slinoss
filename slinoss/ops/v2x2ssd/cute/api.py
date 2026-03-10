"""CuTe entry point for the v2x2ssd operator."""

from __future__ import annotations

import torch

from slinoss.ops.v2x2ssd.reference import (
    _resolve_dtypes,
    _resolve_empty_outputs,
    _validate_inputs,
)


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
    # CuTe is a real runtime dependency for this path. Import it here so the
    # rest of the repo remains usable without the kernel toolchain.
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    from .kernels.fwd.chunk_increment import chunk_increment_cute
    from .kernels.fwd.chunk_scan import chunk_scan_cute
    from .kernels.fwd.state_passing import state_passing_cute

    batch_size, n_heads, T, N, P = _validate_inputs(
        U, M, K, B, C, initial_states, B_prev, U_prev
    )
    if U.device.type != "cuda":
        raise ValueError("CuTe v2x2ssd requires CUDA tensors.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive. Got {chunk_size}.")

    D = 2 * N
    rdtype, odtype = _resolve_dtypes(
        input_dtypes=[U.dtype, M.dtype, K.dtype, B.dtype, C.dtype],
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
        default_output_dtype=U.dtype,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe v2x2ssd forward path supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )

    if T == 0:
        return _resolve_empty_outputs(
            batch_size=batch_size,
            n_heads=n_heads,
            P=P,
            D=D,
            device=U.device,
            output_dtype=odtype,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
        )

    if any(
        tensor is not None and tensor.requires_grad
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
    ):
        from .autograd import v2x2ssd_cute_autograd

        return v2x2ssd_cute_autograd(
            U,
            M,
            K,
            B,
            C,
            chunk_size=chunk_size,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=rdtype,
            output_dtype=odtype,
        )

    B_last = B[:, :, -1, :].to(dtype=odtype).contiguous()
    U_last = U[:, :, -1, :].to(dtype=odtype).contiguous()

    inc, m_chunk = chunk_increment_cute(
        U,
        M,
        K,
        B,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=rdtype,
    )
    chunk_starts, final_state = state_passing_cute(
        inc,
        m_chunk,
        initial_states=initial_states,
        compute_dtype=rdtype,
    )
    Y = chunk_scan_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        output_dtype=odtype,
        compute_dtype=rdtype,
    )
    return Y, final_state.to(dtype=odtype).contiguous(), B_last, U_last


__all__ = ["v2x2ssd_cute"]
