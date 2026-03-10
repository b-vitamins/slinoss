"""Autograd wrapper for the CuTe ``v2x2ssd`` operator."""

from __future__ import annotations

from typing import Any, cast

import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.stage import (
    chunk_increment_bwd_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.common import (
    _scalar_grad_from_vec,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.param import (
    _chunk_scan_bwd_param_from_intermediates,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.dc import (
    chunk_scan_bwd_dc_exact_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.db import (
    chunk_scan_bwd_db_exact_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_scan.dlogprefix import (
    chunk_scan_bwd_dlogprefix_exact_cute,
)
from slinoss.ops.v2x2ssd.cute.kernels.bwd.state_passing import state_passing_bwd_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_increment import chunk_increment_cute
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_chunk_scan,
    _pack_chunk_scan_inner_inputs,
    _prepare_chunk_scan_small_operands,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.state_passing import state_passing_cute
def _run_chunk_scan_forward_and_pack(
    U: torch.Tensor,
    M: torch.Tensor,
    K: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_starts: torch.Tensor,
    *,
    chunk_size: int,
    B_prev: torch.Tensor | None,
    U_prev: torch.Tensor | None,
    compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix_half,
        Z0_raw,
        U_head,
        B_head,
        batch_size,
        n_heads,
        T,
        T_pad,
        odtype,
    ) = _prepare_chunk_scan_small_operands(
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
        output_dtype=output_dtype,
    )
    Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0 = _pack_chunk_scan_inner_inputs(
        U_raw,
        B_raw,
        C_raw,
        M_raw,
        K_raw,
        logprefix_half,
        Z0_raw,
        U_head,
        B_head,
    )

    out = torch.empty(
        (Q.shape[0], Q.shape[1], 1, Vprev.shape[-1]),
        device=U.device,
        dtype=odtype,
    )
    compiled = cast(
        Any,
        _get_compiled_chunk_scan(
        Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0, out
        ),
    )
    compiled(
        from_dlpack(Q, assumed_align=16),
        from_dlpack(Kprev, assumed_align=16),
        from_dlpack(Vprev, assumed_align=16),
        from_dlpack(Kcurr, assumed_align=16),
        from_dlpack(Vcurr, assumed_align=16),
        from_dlpack(logprefix_half, assumed_align=16),
        from_dlpack(Z0, assumed_align=16),
        from_dlpack(out, assumed_align=16),
    )

    L = int(chunk_size)
    n_chunks = int(T_pad // L)
    Y = out.squeeze(2).reshape(batch_size, n_heads, n_chunks, L, out.shape[-1])
    Y = Y.reshape(batch_size, n_heads, T_pad, out.shape[-1])[:, :, :T].contiguous()
    return Y, M_raw, K_raw, B_raw, B_head, Q, Kprev, Vprev, Kcurr, Vcurr, logprefix_half, Z0


def _zero_like_optional(tensor: torch.Tensor | None, *, dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    return torch.zeros_like(tensor, dtype=dtype)


def _packed_phase_prefix(M_raw: torch.Tensor) -> torch.Tensor:
    m_c = torch.view_as_complex(M_raw.contiguous())
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.cumprod(m_c / mag, dim=1)


def _packed_causal_scales(logprefix_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    L = int(logprefix_half.shape[1])
    t_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(1)
    s_idx = torch.arange(L, device=logprefix_half.device).unsqueeze(0)
    causal = (s_idx <= t_idx).unsqueeze(0)
    lp = logprefix_half.to(torch.float32)
    scale = torch.exp(2.0 * (lp.unsqueeze(-1) - lp.unsqueeze(1))).masked_fill(
        ~causal, 0.0
    )
    row_scale = torch.exp(2.0 * lp).unsqueeze(-1)
    return scale, row_scale


def _scatter_value_grads(
    dVprev: torch.Tensor,
    dVcurr: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    BHC, L, P = map(int, dVprev.shape)
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dU_blk = dVcurr.reshape(batch_size, n_heads, n_chunks, L, P).clone()
    dVprev_view = dVprev.reshape(batch_size, n_heads, n_chunks, L, P)
    if L > 1:
        dU_blk[..., :-1, :] += dVprev_view[..., 1:, :]

    d_head = dVprev_view[..., 0, :]
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += d_head[:, :, 1:, :]

    dU_prev = d_head[:, :, 0, :].contiguous()
    dU = dU_blk.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous()
    return dU, dU_prev


def _scatter_key_grads(
    dK_prev_packed: torch.Tensor,
    dK_curr_packed: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BHC, L, D = map(int, dK_prev_packed.shape)
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    N = D // 2

    phase_c = phase.unsqueeze(-1)
    dK_prev_c = torch.view_as_complex(
        dK_prev_packed.reshape(BHC, L, N, 2).contiguous()
    )
    dK_curr_c = torch.view_as_complex(
        dK_curr_packed.reshape(BHC, L, N, 2).contiguous()
    )
    d_beta_prev = phase_c * torch.conj(dK_prev_c)
    d_beta_curr = phase_c * torch.conj(dK_curr_c)

    b_curr = torch.view_as_complex(B_raw.reshape(BHC, L, N, 2).contiguous())
    b_head_c = torch.view_as_complex(B_head.reshape(BHC, N, 2).contiguous())
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if L > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]

    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous())
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous())

    dB_curr_c = torch.conj(k_curr_c).unsqueeze(-1) * d_beta_curr
    dB_prev_seq_c = torch.conj(k_prev_c).unsqueeze(-1) * d_beta_prev
    dK_prev_tap = _scalar_grad_from_vec(b_prev_seq, d_beta_prev)
    dK_curr_tap = _scalar_grad_from_vec(b_curr, d_beta_curr)

    dB_blk = dB_curr_c.reshape(batch_size, n_heads, n_chunks, L, N).clone()
    dB_prev_view = dB_prev_seq_c.reshape(batch_size, n_heads, n_chunks, L, N)
    if L > 1:
        dB_blk[..., :-1, :] += dB_prev_view[..., 1:, :]

    d_head = dB_prev_view[..., 0, :]
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_head[:, :, 1:, :]

    dB_prev = (
        torch.view_as_real(d_head[:, :, 0, :].contiguous())
        .reshape(batch_size, n_heads, D)
        .to(dtype=torch.float32)
        .contiguous()
    )
    dB = (
        torch.view_as_real(dB_blk)
        .reshape(batch_size, n_heads, T_pad, D)
        .to(dtype=torch.float32)[:, :, :T, :]
        .contiguous()
    )
    dK_prev_real = torch.view_as_real(
        dK_prev_tap.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK_curr_real = torch.view_as_real(
        dK_curr_tap.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK = (
        torch.stack((dK_prev_real, dK_curr_real), dim=4)
        .reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :]
        .contiguous()
    )
    return dB, dB_prev, dK


def _chunk_scan_bwd_exact_packed(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    M_raw: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    BHC, L, _, D = map(int, Q.shape)
    P = int(Vprev.shape[-1])
    BH = batch_size * n_heads
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    N = D // 2

    if T_pad != T:
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, T_pad - T, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_flat = d_out.reshape(BHC, L, P).to(torch.float32)
    Qf = Q.squeeze(2).to(torch.float32)
    Kprevf = Kprev.squeeze(2).to(torch.float32)
    Kcurrf = Kcurr.squeeze(2).to(torch.float32)
    Vprevf = Vprev.squeeze(2).to(torch.float32)
    Vcurrf = Vcurr.squeeze(2).to(torch.float32)
    Z0f = Z0.squeeze(2).to(torch.float32)

    scale, row_scale = _packed_causal_scales(logprefix_half)
    score_prev = torch.bmm(Qf, Kprevf.transpose(1, 2))
    score_curr = torch.bmm(Qf, Kcurrf.transpose(1, 2))
    Sprev = score_prev * scale
    Scurr = score_curr * scale
    dSprev = torch.bmm(d_out_flat, Vprevf.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, Vcurrf.transpose(1, 2))
    dScore_prev = dSprev * scale
    dScore_curr = dScurr * scale

    dZ0 = torch.bmm(
        (d_out_flat * row_scale).transpose(1, 2),
        Qf,
    )
    d_chunk_starts = (
        torch.view_as_real(
            torch.conj(
                torch.view_as_complex(dZ0.reshape(BHC, P, N, 2).contiguous())
            ).resolve_conj()
        )
        .reshape(batch_size, n_heads, n_chunks, P, D)
        .to(dtype=torch.float32)
        .contiguous()
    )

    dVprev = torch.bmm(Sprev.transpose(1, 2), d_out_flat)
    dVcurr = torch.bmm(Scurr.transpose(1, 2), d_out_flat)
    dU, dU_prev = _scatter_value_grads(
        dVprev,
        dVcurr,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )

    dQ = (
        torch.bmm(d_out_flat * row_scale, Z0f)
        + torch.bmm(dScore_prev, Kprevf)
        + torch.bmm(dScore_curr, Kcurrf)
    )
    dK_prev_packed = torch.bmm(dScore_prev.transpose(1, 2), Qf)
    dK_curr_packed = torch.bmm(dScore_curr.transpose(1, 2), Qf)
    y_off = torch.bmm(Qf, Z0f.transpose(1, 2)) * row_scale
    d_logprefix_half = chunk_scan_bwd_dlogprefix_exact_cute(
        score_prev,
        score_curr,
        dSprev,
        dScurr,
        logprefix_half.contiguous(),
        y_off.contiguous(),
        d_out_flat,
    )

    phase = _packed_phase_prefix(M_raw)
    dC = chunk_scan_bwd_dc_exact_cute(
        dQ.contiguous(),
        torch.view_as_real(phase).to(dtype=torch.float32).contiguous(),
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dB, dB_prev, dK = chunk_scan_bwd_db_exact_cute(
        dK_prev_packed.contiguous(),
        dK_curr_packed.contiguous(),
        torch.view_as_real(phase).to(dtype=torch.float32).contiguous(),
        K_raw.to(dtype=torch.float32).contiguous(),
        B_raw.to(dtype=torch.float32).contiguous(),
        B_head.to(dtype=torch.float32).contiguous(),
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )
    dM = _chunk_scan_bwd_param_from_intermediates(
        Qf,
        Kprevf,
        Kcurrf,
        phase,
        M_raw,
        dQ,
        dK_prev_packed,
        dK_curr_packed,
        d_logprefix_half,
        batch_size=batch_size,
        n_heads=n_heads,
    )[:, :, :T, :].contiguous()
    return dU, dM, dK, dB, dC, d_chunk_starts, dB_prev, dU_prev


class _V2x2SSDCuTeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        U: torch.Tensor,
        M: torch.Tensor,
        K: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_states: torch.Tensor | None,
        B_prev: torch.Tensor | None,
        U_prev: torch.Tensor | None,
        chunk_size: int,
        compute_dtype: torch.dtype | None,
        output_dtype: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = compute_dtype
        ctx.output_dtype = output_dtype
        ctx.has_initial_states = initial_states is not None
        ctx.has_prev = B_prev is not None

        U_d = U.detach()
        M_d = M.detach()
        K_d = K.detach()
        B_d = B.detach()
        C_d = C.detach()
        initial_states_d = initial_states.detach() if initial_states is not None else None
        B_prev_d = B_prev.detach() if B_prev is not None else None
        U_prev_d = U_prev.detach() if U_prev is not None else None

        B_last = B_d[:, :, -1, :].to(dtype=output_dtype or U.dtype).contiguous()
        U_last = U_d[:, :, -1, :].to(dtype=output_dtype or U.dtype).contiguous()

        inc, m_chunk = chunk_increment_cute(
            U_d,
            M_d,
            K_d,
            B_d,
            chunk_size=ctx.chunk_size,
            B_prev=B_prev_d,
            U_prev=U_prev_d,
            compute_dtype=compute_dtype,
        )
        chunk_starts, final_state = state_passing_cute(
            inc,
            m_chunk,
            initial_states=initial_states_d,
            compute_dtype=compute_dtype,
        )
        (
            Y,
            M_raw,
            K_raw,
            B_raw,
            B_head,
            Q,
            Kprev,
            Vprev,
            Kcurr,
            Vcurr,
            logprefix_half,
            Z0,
        ) = _run_chunk_scan_forward_and_pack(
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            chunk_starts,
            chunk_size=ctx.chunk_size,
            B_prev=B_prev_d,
            U_prev=U_prev_d,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )

        saved: list[torch.Tensor] = [
            U_d,
            M_d,
            K_d,
            B_d,
            C_d,
            m_chunk,
            chunk_starts,
            M_raw,
            K_raw,
            B_raw,
            B_head,
            Q,
            Kprev,
            Vprev,
            Kcurr,
            Vcurr,
            logprefix_half,
            Z0,
        ]
        if initial_states_d is not None:
            saved.append(initial_states_d)
        if B_prev_d is not None:
            saved.append(B_prev_d)
            saved.append(U_prev_d)  # type: ignore[arg-type]
        ctx.save_for_backward(*saved)
        return Y, final_state.to(dtype=output_dtype or U.dtype).contiguous(), B_last, U_last

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        dY: torch.Tensor | None,
        d_final_state: torch.Tensor | None,
        dB_last: torch.Tensor | None,
        dU_last: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
    ]:
        saved = ctx.saved_tensors
        idx = 0
        U = saved[idx]
        idx += 1
        M = saved[idx]
        idx += 1
        K = saved[idx]
        idx += 1
        B = saved[idx]
        idx += 1
        C = saved[idx]
        idx += 1
        m_chunk = saved[idx]
        idx += 1
        chunk_starts = saved[idx]
        idx += 1
        M_raw = saved[idx]
        idx += 1
        K_raw = saved[idx]
        idx += 1
        B_raw = saved[idx]
        idx += 1
        B_head = saved[idx]
        idx += 1
        Q = saved[idx]
        idx += 1
        Kprev = saved[idx]
        idx += 1
        Vprev = saved[idx]
        idx += 1
        Kcurr = saved[idx]
        idx += 1
        Vcurr = saved[idx]
        idx += 1
        logprefix_half = saved[idx]
        idx += 1
        Z0 = saved[idx]
        idx += 1

        initial_states = None
        if ctx.has_initial_states:
            initial_states = saved[idx]
            idx += 1

        B_prev = None
        U_prev = None
        if ctx.has_prev:
            B_prev = saved[idx]
            U_prev = saved[idx + 1]

        batch_size, n_heads, T, P = map(int, U.shape)
        D = int(B.shape[-1])
        odtype = ctx.output_dtype or U.dtype
        rdtype = torch.float32

        dU_scan = torch.zeros_like(U, dtype=rdtype)
        dM_scan = torch.zeros_like(M, dtype=rdtype)
        dK_scan = torch.zeros_like(K, dtype=rdtype)
        dB_scan = torch.zeros_like(B, dtype=rdtype)
        dC_scan = torch.zeros_like(C, dtype=rdtype)
        dB_prev_scan = _zero_like_optional(B_prev, dtype=rdtype)
        dU_prev_scan = _zero_like_optional(U_prev, dtype=rdtype)
        d_chunk_starts = torch.zeros_like(chunk_starts, dtype=rdtype)

        if dY is not None:
            dY = dY.contiguous()
            (
                dU_scan,
                dM_scan,
                dK_scan,
                dB_scan,
                dC_scan,
                d_chunk_starts,
                dB_prev_scan_raw,
                dU_prev_scan_raw,
            ) = _chunk_scan_bwd_exact_packed(
                Q,
                Kprev,
                Vprev,
                Kcurr,
                Vcurr,
                logprefix_half,
                Z0,
                M_raw,
                K_raw,
                B_raw,
                B_head,
                dY,
                batch_size=batch_size,
                n_heads=n_heads,
                T=T,
            )
            if dU_prev_scan is not None:
                dU_prev_scan = dU_prev_scan_raw
            if dB_prev_scan is not None:
                dB_prev_scan = dB_prev_scan_raw

        dU_inc = torch.zeros_like(U, dtype=rdtype)
        dM_inc = torch.zeros_like(M, dtype=rdtype)
        dK_inc = torch.zeros_like(K, dtype=rdtype)
        dB_inc = torch.zeros_like(B, dtype=rdtype)
        dB_prev_inc = _zero_like_optional(B_prev, dtype=rdtype)
        dU_prev_inc = _zero_like_optional(U_prev, dtype=rdtype)
        d_initial = _zero_like_optional(initial_states, dtype=rdtype)

        if dY is not None or d_final_state is not None:
            if d_final_state is None:
                d_final_state = torch.zeros(
                    (batch_size, n_heads, P, D),
                    device=U.device,
                    dtype=odtype,
                )
            d_inc, d_m_chunk, d_initial_raw = state_passing_bwd_cute(
                d_chunk_starts.contiguous(),
                d_final_state.contiguous(),
                chunk_starts,
                m_chunk,
            )
            dU_inc, dM_inc, dK_inc, dB_inc, dB_prev_inc_raw, dU_prev_inc_raw = (
                chunk_increment_bwd_cute(
                    U,
                    M,
                    K,
                    B,
                    d_inc=d_inc,
                    d_m_chunk=d_m_chunk,
                    chunk_size=ctx.chunk_size,
                    B_prev=B_prev,
                    U_prev=U_prev,
                    compute_dtype=ctx.compute_dtype,
                )
            )
            if d_initial is not None:
                d_initial = d_initial_raw
            if dB_prev_inc is not None:
                dB_prev_inc = dB_prev_inc_raw
            if dU_prev_inc is not None:
                dU_prev_inc = dU_prev_inc_raw

        dU_total = dU_inc + dU_scan
        dM_total = dM_inc + dM_scan
        dK_total = dK_inc + dK_scan
        dB_total = dB_inc + dB_scan
        dC_total = dC_scan

        if dB_last is not None:
            dB_total[:, :, -1, :] += dB_last.to(dtype=rdtype)
        if dU_last is not None:
            dU_total[:, :, -1, :] += dU_last.to(dtype=rdtype)

        dB_prev_total = None
        if B_prev is not None and dB_prev_inc is not None and dB_prev_scan is not None:
            dB_prev_total = dB_prev_inc + dB_prev_scan

        dU_prev_total = None
        if U_prev is not None and dU_prev_inc is not None and dU_prev_scan is not None:
            dU_prev_total = dU_prev_inc + dU_prev_scan

        return (
            dU_total.to(dtype=U.dtype),
            dM_total.to(dtype=M.dtype),
            dK_total.to(dtype=K.dtype),
            dB_total.to(dtype=B.dtype),
            dC_total.to(dtype=C.dtype),
            None if initial_states is None or d_initial is None else d_initial.to(dtype=initial_states.dtype),
            None if B_prev is None or dB_prev_total is None else dB_prev_total.to(dtype=B_prev.dtype),
            None if U_prev is None or dU_prev_total is None else dU_prev_total.to(dtype=U_prev.dtype),
            None,
            None,
            None,
        )


def v2x2ssd_cute_autograd(
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
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        _V2x2SSDCuTeFn.apply(
            U,
            M,
            K,
            B,
            C,
            initial_states,
            B_prev,
            U_prev,
            int(chunk_size),
            compute_dtype,
            output_dtype,
        ),
    )


__all__ = ["v2x2ssd_cute_autograd"]
