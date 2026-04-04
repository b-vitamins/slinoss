"""CuTe entry point for the v2x2ssd operator."""

from __future__ import annotations

from typing import Literal, overload, cast

import torch

from .common import _materialize_boundary_tensor
from slinoss.ops.v2x2ssd.reference import (
    _resolve_dtypes,
    _resolve_empty_outputs,
    _validate_inputs,
)


@overload
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
    return_state: Literal[False] = False,
) -> torch.Tensor: ...


@overload
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
    return_state: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


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
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe-backed v2x2ssd forward path."""
    # CuTe is a real runtime dependency for this path. Import it here so the
    # rest of the repo remains usable without the kernel toolchain.
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    from .kernels.fwd import _v2x2ssd_fwd_cute_prevalidated

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
        empty_y, final_state, b_last, u_last = _resolve_empty_outputs(
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
        if not return_state:
            return empty_y
        return empty_y, final_state, b_last, u_last

    needs_autograd = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (U, M, K, B, C, initial_states, B_prev, U_prev)
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
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=rdtype,
            output_dtype=odtype,
            return_state=return_state,
        )

    if not return_state:
        Y, _m_chunk, _chunk_starts = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            _v2x2ssd_fwd_cute_prevalidated(
                U,
                M,
                K,
                B,
                C,
                chunk_size=chunk_size,
                initial_states=initial_states,
                B_prev=B_prev,
                U_prev=U_prev,
                output_dtype=odtype,
                compute_dtype=rdtype,
                return_intermediates=False,
            ),
        )
        return Y

    Y, final_state, _m_chunk, _chunk_starts = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _v2x2ssd_fwd_cute_prevalidated(
            U,
            M,
            K,
            B,
            C,
            chunk_size=chunk_size,
            initial_states=initial_states,
            B_prev=B_prev,
            U_prev=U_prev,
            output_dtype=odtype,
            compute_dtype=rdtype,
            return_final_state=True,
            # Keep the boundary-return path on fresh intermediates. This
            # matches the autograd wrapper and avoids coupling public state
            # materialization to cached no-grad workspaces.
            return_intermediates=True,
        ),
    )
    Y_out = cast(
        torch.Tensor,
        _materialize_boundary_tensor(
            Y,
            dtype=odtype,
        ),
    )
    final_state_out = cast(
        torch.Tensor,
        _materialize_boundary_tensor(
            final_state,
            dtype=odtype,
        ),
    )
    B_last = cast(
        torch.Tensor,
        _materialize_boundary_tensor(
            B[:, :, -1, :],
            dtype=odtype,
        ),
    )
    U_last = cast(
        torch.Tensor,
        _materialize_boundary_tensor(
            U[:, :, -1, :],
            dtype=odtype,
        ),
    )
    return Y_out, final_state_out, B_last, U_last


__all__ = ["v2x2ssd_cute"]
