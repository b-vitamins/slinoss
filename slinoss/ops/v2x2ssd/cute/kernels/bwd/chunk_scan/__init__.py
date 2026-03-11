"""CuTe backward kernels for the ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

from typing import Callable

import torch

from .common import prepare_chunk_scan_bwd_dout, prepare_chunk_scan_bwd_packed_context
from .db import (
    _chunk_scan_bwd_dk_prepared_cute,
    chunk_scan_bwd_db_cute,
    chunk_scan_bwd_db_exact_cute,
    chunk_scan_bwd_dk_packed_cute,
    prepare_chunk_scan_bwd_db_operands,
)
from .dc import (
    chunk_scan_bwd_dc_cute,
    chunk_scan_bwd_dc_exact_cute,
    chunk_scan_bwd_dc_packed_cute,
    prepare_chunk_scan_bwd_dc_operands,
)
from .du import (
    _chunk_scan_bwd_du_prepared_cute,
    chunk_scan_bwd_du_cute,
    prepare_chunk_scan_bwd_du_operands,
)
from .dz0 import chunk_scan_bwd_dz0_cute, chunk_scan_bwd_dz0_packed_cute
from .param_scan import (
    chunk_scan_bwd_dlogprefix_exact_cute,
    chunk_scan_bwd_param_cute,
    chunk_scan_bwd_param_packed_cute,
    chunk_scan_bwd_param_scan_cute,
    chunk_scan_bwd_param_scan_packed_cute,
)


def _chunk_scan_dz0_to_chunk_starts(
    dZ0: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    n_chunks: int,
    P: int,
    D: int,
) -> torch.Tensor:
    N = D // 2
    return (
        torch.view_as_real(
            torch.conj(
                torch.view_as_complex(dZ0.reshape(batch_size * n_heads * n_chunks, P, N, 2).contiguous())
            ).resolve_conj()
        )
        .reshape(batch_size, n_heads, n_chunks, P, D)
        .to(dtype=torch.float32)
        .contiguous()
    )


def _run_chunk_scan_bwd_pipeline(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
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
    torch.Tensor,
    torch.Tensor,
]:
    ctx = prepare_chunk_scan_bwd_packed_context(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )
    d_out_padded, d_out_flat, d_out_rev = prepare_chunk_scan_bwd_dout(
        d_out,
        ctx=ctx,
        tc_dtype=ctx.Q.dtype,
    )

    dZ0 = chunk_scan_bwd_dz0_packed_cute(
        ctx.Q.contiguous(),
        ctx.logprefix_half.contiguous(),
        d_out_flat,
    )
    d_chunk_starts = _chunk_scan_dz0_to_chunk_starts(
        dZ0,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        n_chunks=ctx.n_chunks,
        P=ctx.P,
        D=ctx.D,
    )

    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = prepare_chunk_scan_bwd_du_operands(
        ctx.Q.contiguous(),
        ctx.Kprev.contiguous(),
        ctx.Kcurr.contiguous(),
        ctx.logprefix_half.contiguous(),
    )
    Q_rev_db, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev_db, phase = (
        prepare_chunk_scan_bwd_db_operands(
            ctx.Q.contiguous(),
            ctx.Vprev.contiguous(),
            ctx.Vcurr.contiguous(),
            ctx.logprefix_half.contiguous(),
            ctx.M_raw.contiguous(),
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )

    dU, dU_prev = _chunk_scan_bwd_du_prepared_cute(
        Q_rev,
        Kprev_rev,
        Kcurr_rev,
        neg_logprefix_half_rev,
        d_out_rev,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )

    z0_q = ctx.Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    dQ = chunk_scan_bwd_dc_packed_cute(
        ctx.Vprev.contiguous(),
        ctx.Kprev.contiguous(),
        ctx.Vcurr.contiguous(),
        ctx.Kcurr.contiguous(),
        ctx.logprefix_half.contiguous(),
        z0_q,
        d_out_padded,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )
    dC = chunk_scan_bwd_dc_exact_cute(
        dQ,
        phase,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )

    dK_prev_packed, dK_curr_packed = _chunk_scan_bwd_dk_prepared_cute(
        Q_rev_db,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev_db,
        d_out_rev,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
    )
    dB, dB_prev, dK = chunk_scan_bwd_db_exact_cute(
        dK_prev_packed.contiguous(),
        dK_curr_packed.contiguous(),
        phase,
        ctx.K_raw.to(dtype=torch.float32).contiguous(),
        ctx.B_raw.to(dtype=torch.float32).contiguous(),
        ctx.B_head.to(dtype=torch.float32).contiguous(),
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )
    dM = chunk_scan_bwd_param_scan_packed_cute(
        ctx.Q,
        ctx.Kprev,
        ctx.Vprev,
        ctx.Kcurr,
        ctx.Vcurr,
        ctx.logprefix_half,
        ctx.Z0,
        ctx.M_raw,
        d_out_flat,
        dQ,
        dK_prev_packed,
        dK_curr_packed,
        phase,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T_pad=ctx.T_pad,
    )
    return dU, dM[:, :, : ctx.T, :].contiguous(), dK, dB, dC, d_chunk_starts, dB_prev, dU_prev


def compile_chunk_scan_bwd_kernels(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None = None,
    U_prev: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
    return_launchers: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[], None],
    Callable[[], None],
]:
    """Compile the split chunk-scan backward pipeline on the public contract."""
    dU = torch.empty_like(U, dtype=torch.float32)
    dM = torch.empty_like(M, dtype=torch.float32)
    dK = torch.empty_like(K, dtype=torch.float32)
    dB = torch.empty_like(B, dtype=torch.float32)
    dC = torch.empty_like(C, dtype=torch.float32)
    d_chunk_starts = torch.empty_like(chunk_starts, dtype=torch.float32)
    dB_prev_out = (
        torch.empty_like(B_prev, dtype=torch.float32)
        if B_prev is not None
        else torch.empty((U.shape[0], U.shape[1], B.shape[-1]), device=U.device, dtype=torch.float32)
    )
    dU_prev_out = (
        torch.empty_like(U_prev, dtype=torch.float32)
        if U_prev is not None
        else torch.empty((U.shape[0], U.shape[1], U.shape[-1]), device=U.device, dtype=torch.float32)
    )

    def launch_sequential() -> None:
        got = _run_chunk_scan_bwd_pipeline(
            U,
            M,
            K,
            B,
            C,
            chunk_starts,
            d_out,
            chunk_size=chunk_size,
            B_prev=B_prev,
            U_prev=U_prev,
            compute_dtype=compute_dtype,
        )
        for out, value in zip(
            (dU, dM, dK, dB, dC, d_chunk_starts, dB_prev_out, dU_prev_out),
            got,
            strict=True,
        ):
            out.copy_(value)

    def launch_overlapped() -> None:
        # The current slice helpers reuse shared scratch buffers keyed by shape,
        # so the safe package-level launcher is the sequential pipeline.
        launch_sequential()

    if return_launchers:
        return (
            dU,
            dM,
            dK,
            dB,
            dC,
            d_chunk_starts,
            dB_prev_out,
            dU_prev_out,
            launch_sequential,
            launch_overlapped,
        )

    launch_sequential()
    return dU, dM, dK, dB, dC, d_chunk_starts, dB_prev_out, dU_prev_out


def chunk_scan_bwd_cute(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    d_out: torch.Tensor,
    *,
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
    torch.Tensor,
    torch.Tensor,
]:
    """Canonical backward entrypoint for the public ``chunk_scan`` contract."""
    return compile_chunk_scan_bwd_kernels(
        U,
        M,
        K,
        B,
        C,
        chunk_starts,
        d_out,
        chunk_size=chunk_size,
        B_prev=B_prev,
        U_prev=U_prev,
        compute_dtype=compute_dtype,
    )


__all__ = [
    "chunk_scan_bwd_cute",
    "compile_chunk_scan_bwd_kernels",
    "prepare_chunk_scan_bwd_db_operands",
    "chunk_scan_bwd_dk_packed_cute",
    "chunk_scan_bwd_db_cute",
    "chunk_scan_bwd_db_exact_cute",
    "prepare_chunk_scan_bwd_dc_operands",
    "chunk_scan_bwd_dc_packed_cute",
    "chunk_scan_bwd_dc_cute",
    "chunk_scan_bwd_dc_exact_cute",
    "chunk_scan_bwd_dlogprefix_exact_cute",
    "prepare_chunk_scan_bwd_du_operands",
    "chunk_scan_bwd_du_cute",
    "chunk_scan_bwd_dz0_packed_cute",
    "chunk_scan_bwd_dz0_cute",
    "chunk_scan_bwd_param_scan_cute",
    "chunk_scan_bwd_param_scan_packed_cute",
    "chunk_scan_bwd_param_cute",
    "chunk_scan_bwd_param_packed_cute",
]
