"""CuTe backward kernels for the ``v2x2ssd`` chunk-scan stage."""

from __future__ import annotations

from typing import Callable

import torch

from .common import prepare_chunk_scan_bwd_dout, prepare_chunk_scan_bwd_packed_context
from .db import (
    _chunk_scan_bwd_dk_prepared_cute,
    chunk_scan_bwd_db_cute,
    chunk_scan_bwd_db_exact_cute,
    chunk_scan_bwd_db_exact_with_meta_cute,
    chunk_scan_bwd_dk_packed_cute,
    prepare_chunk_scan_bwd_db_operands,
)
from .dc import (
    chunk_scan_bwd_dc_cute,
    chunk_scan_bwd_dc_exact_cute,
    chunk_scan_bwd_dq_meta_cute,
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
    chunk_scan_bwd_phase_scan_from_meta_cute,
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
    Q = ctx.Q.contiguous()
    Kprev = ctx.Kprev.contiguous()
    Vprev = ctx.Vprev.contiguous()
    Kcurr = ctx.Kcurr.contiguous()
    Vcurr = ctx.Vcurr.contiguous()
    logprefix_half = ctx.logprefix_half.contiguous()
    M_raw = ctx.M_raw.contiguous()
    K_raw = ctx.K_raw.to(dtype=torch.float32).contiguous()
    B_raw = ctx.B_raw.to(dtype=torch.float32).contiguous()
    B_head = ctx.B_head.to(dtype=torch.float32).contiguous()
    z0_q = ctx.Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = (
        prepare_chunk_scan_bwd_du_operands(Q, Kprev, Kcurr, logprefix_half)
    )
    Q_rev_db, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev_db, phase = (
        prepare_chunk_scan_bwd_db_operands(
            Q,
            Vprev,
            Vcurr,
            logprefix_half,
            M_raw,
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )

    return _run_chunk_scan_bwd_pipeline_prepared(
        ctx=ctx,
        Q=Q,
        Kprev=Kprev,
        Vprev=Vprev,
        Kcurr=Kcurr,
        Vcurr=Vcurr,
        logprefix_half=logprefix_half,
        M_raw=M_raw,
        K_raw=K_raw,
        B_raw=B_raw,
        B_head=B_head,
        z0_q=z0_q,
        Q_rev=Q_rev,
        Kprev_rev=Kprev_rev,
        Kcurr_rev=Kcurr_rev,
        neg_logprefix_half_rev=neg_logprefix_half_rev,
        Q_rev_db=Q_rev_db,
        Vprev_rev=Vprev_rev,
        Vcurr_rev=Vcurr_rev,
        neg_logprefix_half_rev_db=neg_logprefix_half_rev_db,
        phase=phase,
        d_out_padded=d_out_padded,
        d_out_flat=d_out_flat,
        d_out_rev=d_out_rev,
    )


def _run_chunk_scan_bwd_pipeline_prepared(
    *,
    ctx,
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    M_raw: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    z0_q: torch.Tensor,
    Q_rev: torch.Tensor,
    Kprev_rev: torch.Tensor,
    Kcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    Q_rev_db: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev_db: torch.Tensor,
    phase: torch.Tensor,
    d_out_padded: torch.Tensor,
    d_out_flat: torch.Tensor,
    d_out_rev: torch.Tensor,
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
    dZ0 = chunk_scan_bwd_dz0_packed_cute(Q, logprefix_half, d_out_flat)
    d_chunk_starts = _chunk_scan_dz0_to_chunk_starts(
        dZ0,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        n_chunks=ctx.n_chunks,
        P=ctx.P,
        D=ctx.D,
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

    dQ = chunk_scan_bwd_dc_packed_cute(
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        logprefix_half,
        z0_q,
        d_out_padded,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )
    d_phase_q, d_logprefix_q = chunk_scan_bwd_dq_meta_cute(Q, dQ, phase)
    dC = chunk_scan_bwd_dc_exact_cute(
        dQ,
        phase,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
    )

    dK_prev_packed_rev, dK_curr_packed_rev = _chunk_scan_bwd_dk_prepared_cute(
        Q_rev_db,
        Vprev_rev,
        Vcurr_rev,
        neg_logprefix_half_rev_db,
        d_out_rev,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        reverse_time=True,
    )
    dB, dB_prev, dK, d_phase_k, d_logprefix_k = chunk_scan_bwd_db_exact_with_meta_cute(
        dK_prev_packed_rev,
        dK_curr_packed_rev,
        Kprev.squeeze(2).contiguous(),
        Kcurr.squeeze(2).contiguous(),
        phase,
        K_raw,
        B_raw,
        B_head,
        batch_size=ctx.batch_size,
        n_heads=ctx.n_heads,
        T=ctx.T,
        reverse_time=True,
    )
    dM = chunk_scan_bwd_phase_scan_from_meta_cute(
        M_raw,
        phase,
        (d_phase_q + d_phase_k).contiguous(),
        (d_logprefix_q + d_logprefix_k).contiguous(),
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
    Q = ctx.Q.contiguous()
    Kprev = ctx.Kprev.contiguous()
    Vprev = ctx.Vprev.contiguous()
    Kcurr = ctx.Kcurr.contiguous()
    Vcurr = ctx.Vcurr.contiguous()
    logprefix_half = ctx.logprefix_half.contiguous()
    M_raw = ctx.M_raw.contiguous()
    K_raw = ctx.K_raw.to(dtype=torch.float32).contiguous()
    B_raw = ctx.B_raw.to(dtype=torch.float32).contiguous()
    B_head = ctx.B_head.to(dtype=torch.float32).contiguous()
    z0_q = ctx.Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    Q_rev, Kprev_rev, Kcurr_rev, neg_logprefix_half_rev = (
        prepare_chunk_scan_bwd_du_operands(Q, Kprev, Kcurr, logprefix_half)
    )
    Q_rev_db, Vprev_rev, Vcurr_rev, neg_logprefix_half_rev_db, phase = (
        prepare_chunk_scan_bwd_db_operands(
            Q,
            Vprev,
            Vcurr,
            logprefix_half,
            M_raw,
            Q_rev=Q_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
        )
    )

    if not return_launchers:
        return _run_chunk_scan_bwd_pipeline_prepared(
            ctx=ctx,
            Q=Q,
            Kprev=Kprev,
            Vprev=Vprev,
            Kcurr=Kcurr,
            Vcurr=Vcurr,
            logprefix_half=logprefix_half,
            M_raw=M_raw,
            K_raw=K_raw,
            B_raw=B_raw,
            B_head=B_head,
            z0_q=z0_q,
            Q_rev=Q_rev,
            Kprev_rev=Kprev_rev,
            Kcurr_rev=Kcurr_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
            Q_rev_db=Q_rev_db,
            Vprev_rev=Vprev_rev,
            Vcurr_rev=Vcurr_rev,
            neg_logprefix_half_rev_db=neg_logprefix_half_rev_db,
            phase=phase,
            d_out_padded=d_out_padded,
            d_out_flat=d_out_flat,
            d_out_rev=d_out_rev,
        )

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
        got = _run_chunk_scan_bwd_pipeline_prepared(
            ctx=ctx,
            Q=Q,
            Kprev=Kprev,
            Vprev=Vprev,
            Kcurr=Kcurr,
            Vcurr=Vcurr,
            logprefix_half=logprefix_half,
            M_raw=M_raw,
            K_raw=K_raw,
            B_raw=B_raw,
            B_head=B_head,
            z0_q=z0_q,
            Q_rev=Q_rev,
            Kprev_rev=Kprev_rev,
            Kcurr_rev=Kcurr_rev,
            neg_logprefix_half_rev=neg_logprefix_half_rev,
            Q_rev_db=Q_rev_db,
            Vprev_rev=Vprev_rev,
            Vcurr_rev=Vcurr_rev,
            neg_logprefix_half_rev_db=neg_logprefix_half_rev_db,
            phase=phase,
            d_out_padded=d_out_padded,
            d_out_flat=d_out_flat,
            d_out_rev=d_out_rev,
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
