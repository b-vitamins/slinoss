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
) -> torch.Tensor:
    """CuTe-backed stateless v2x2ssd forward path.

    Contract:
    - logical shapes and layouts match ``reference.v2x2ssd`` exactly
    - stateless forward numerics are mode-invariant across grad-enabled,
      ``torch.no_grad()``, and ``torch.inference_mode()``
    - gradients, when recorded, are layered on top of the same stateless
      forward contract
    - the CuTe path preserves the staged decomposition
      ``chunk_increment -> state_passing -> chunk_scan``
    - the current public entrypoint is stateless-only and returns only ``Y``
    """
    # CuTe is a real runtime dependency for this path. Import it here so the
    # rest of the repo remains usable without the kernel toolchain.
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    from .kernels.fwd import v2x2ssd_fwd_cute

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
    if initial_states is not None or B_prev is not None or U_prev is not None:
        raise NotImplementedError(
            "CuTe v2x2ssd currently supports only stateless execution."
        )

    if T == 0:
        empty_y, _final, _b_last, _u_last = _resolve_empty_outputs(
            batch_size=batch_size,
            n_heads=n_heads,
            P=P,
            D=D,
            device=U.device,
            output_dtype=odtype,
            initial_states=None,
            B_prev=None,
            U_prev=None,
        )
        return empty_y

    needs_autograd = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (U, M, K, B, C)
    )
    if needs_autograd:
        from .autograd import v2x2ssd_cute_training_autograd

        return v2x2ssd_cute_training_autograd(
            U,
            M,
            K,
            B,
            C,
            chunk_size=chunk_size,
            compute_dtype=rdtype,
            output_dtype=odtype,
        )

    Y, _m_chunk, _chunk_starts = v2x2ssd_fwd_cute(
        U,
        M,
        K,
        B,
        C,
        chunk_size=chunk_size,
        output_dtype=odtype,
        compute_dtype=rdtype,
    )
    return Y


__all__ = ["v2x2ssd_cute"]
