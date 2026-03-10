"""Parameter-side backward slice for ``chunk_scan`` gradients into ``M``.

This slice stays on the packed chunk contract and differentiates the dense
``Q/K`` algebra explicitly in Torch:

- exact packed ``dQ/dKprev/dKcurr`` from batched matrix products
- exact cumulative ``dlogprefix_half`` from the same packed contract
- short SO(2) reverse scan back to per-step ``M``

The chunk axis is intentionally kept explicit and small here. That keeps the
phase-sensitive path numerically transparent and avoids feeding approximate
packed ``dQ/dK`` slices into the final ``M`` reduction.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_CompiledPhaseReduceKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]
_CompiledPhaseScanKey = tuple[
    int,
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]
_COMPILED_PHASE_REDUCE: dict[_CompiledPhaseReduceKey, object] = {}
_COMPILED_PHASE_SCAN: dict[_CompiledPhaseScanKey, object] = {}


class _ChunkScanParamPhaseReduce:
    """Warp-cooperative packed phase reduction.

    Logical shape:
    - inputs: ``Q/Kprev/Kcurr/dQ/dKprev/dKcurr`` are ``(BHC, L, D)``
    - ``phase``: ``(BHC, L, 2)``
    - output ``d_phase``: ``(BHC, L, 2)``

    Layout / launch:
    - one warp owns one ``(bhc, t)`` item
    - lanes stride over the interleaved complex feature pairs ``N = D / 2``
    - grid: linear over ``BHC * L``

    This matches the saved packed layout: ``D`` is contiguous, so a warp sees
    coalesced loads while reducing the complex scalar gradient for the phase.
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDQ: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mQ.element_type
                == mKprev.element_type
                == mKcurr.element_type
                == mDQ.element_type
                == mDKprev.element_type
                == mDKcurr.element_type
                == mPhase.element_type
                == mDPhase.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("phase-reduce expects Float32 tensors.")
        if cutlass.const_expr(mQ.shape != mKprev.shape or mQ.shape != mKcurr.shape):
            raise ValueError("Q/K tensors must share the same shape.")
        if cutlass.const_expr(mQ.shape != mDQ.shape or mQ.shape != mDKprev.shape or mQ.shape != mDKcurr.shape):
            raise ValueError("Q and dQ/dK tensors must share the same shape.")
        if cutlass.const_expr(mPhase.shape[0] != mQ.shape[0] or mPhase.shape[1] != mQ.shape[1] or mPhase.shape[2] != 2):
            raise ValueError("phase must be (BHC, L, 2).")
        if cutlass.const_expr(mDPhase.shape != mPhase.shape):
            raise ValueError("d_phase must match phase shape.")

        BHC = cute.size(mQ.shape[0])
        L = cute.size(mQ.shape[1])
        warps_per_block = self.num_threads // 32
        total_items = BHC * L
        self.kernel(
            mQ,
            mKprev,
            mKcurr,
            mDQ,
            mDKprev,
            mDKcurr,
            mPhase,
            mDPhase,
            BHC,
            L,
            total_items,
        ).launch(
            grid=[cute.ceil_div(total_items, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKprev: cute.Tensor,
        mKcurr: cute.Tensor,
        mDQ: cute.Tensor,
        mDKprev: cute.Tensor,
        mDKcurr: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        BHC: cutlass.Int32,
        L: cutlass.Int32,
        total_items: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        item = bidx * warps_per_block + warp
        item_valid = cute.elem_less(item, total_items)
        item_safe = cutlass.min(item, total_items - cutlass.Int32(1))
        bhc = item_safe // L
        t = item_safe - bhc * L

        pr = cutlass.Float32(mPhase[bhc, t, 0])
        pi = cutlass.Float32(mPhase[bhc, t, 1])
        N = cute.size(mQ.shape[2]) // 2

        acc_re = cutlass.Float32(0.0)
        acc_im = cutlass.Float32(0.0)
        n = lane
        while n < N:
            col = n * 2
            qr = cutlass.Float32(mQ[bhc, t, col + 0])
            qi = cutlass.Float32(mQ[bhc, t, col + 1])
            kpr = cutlass.Float32(mKprev[bhc, t, col + 0])
            kpi = cutlass.Float32(mKprev[bhc, t, col + 1])
            kcr = cutlass.Float32(mKcurr[bhc, t, col + 0])
            kci = cutlass.Float32(mKcurr[bhc, t, col + 1])

            dqr = cutlass.Float32(mDQ[bhc, t, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, t, col + 1])
            dkpr = cutlass.Float32(mDKprev[bhc, t, col + 0])
            dkpi = cutlass.Float32(mDKprev[bhc, t, col + 1])
            dkcr = cutlass.Float32(mDKcurr[bhc, t, col + 0])
            dkci = cutlass.Float32(mDKcurr[bhc, t, col + 1])

            qbr = qr * pr + qi * pi
            qbi = qi * pr - qr * pi
            kpbr = kpr * pr + kpi * pi
            kpbi = kpi * pr - kpr * pi
            kcbr = kcr * pr + kci * pi
            kcbi = kci * pr - kcr * pi

            acc_re += dqr * qbr + dqi * qbi
            acc_im += -dqr * qbi + dqi * qbr
            acc_re += dkpr * kpbr + dkpi * kpbi
            acc_im += -dkpr * kpbi + dkpi * kpbr
            acc_re += dkcr * kcbr + dkci * kcbi
            acc_im += -dkcr * kcbi + dkci * kcbr
            n += 32

        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=16, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=16, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=8, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=8, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=4, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=4, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=2, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=2, mask=-1, mask_and_clamp=31
        )
        acc_re = acc_re + cute.arch.shuffle_sync_bfly(
            acc_re, offset=1, mask=-1, mask_and_clamp=31
        )
        acc_im = acc_im + cute.arch.shuffle_sync_bfly(
            acc_im, offset=1, mask=-1, mask_and_clamp=31
        )
        if item_valid and lane == 0:
            mDPhase[bhc, t, 0] = acc_re
            mDPhase[bhc, t, 1] = acc_im


class _ChunkScanParamPhaseScan:
    """Short reverse scan from ``(phase, d_phase, d_logprefix_half)`` to ``dM``.

    Logical shape:
    - ``M_raw`` / ``phase`` / ``d_phase`` / output ``dM``: ``(BHC, L, 2)``
    - ``d_logprefix_half``: ``(BHC, L)``

    Thread layout:
    - one warp owns one ``bhc`` sequence
    - lane 0 executes the scalar SO(2) reverse scan while other lanes stay idle

    The hot dense reduction already happened in ``_ChunkScanParamPhaseReduce``.
    This kernel removes the many tiny Torch launches in the remaining exact
    reverse scan without trying to invent a tensor-core shape for scalar work.
    """

    def __init__(self, *, num_threads: int = 128) -> None:
        self.num_threads = int(num_threads)
        if self.num_threads <= 0 or self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a positive multiple of 32.")

    @cute.jit
    def __call__(
        self,
        mMRaw: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        mDM: cute.Tensor,
    ) -> None:
        if cutlass.const_expr(
            not (
                mMRaw.element_type
                == mPhase.element_type
                == mDPhase.element_type
                == mDLogprefixHalf.element_type
                == mDM.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("phase-scan expects Float32 tensors.")
        if cutlass.const_expr(mMRaw.shape != mPhase.shape or mMRaw.shape != mDPhase.shape or mMRaw.shape != mDM.shape):
            raise ValueError("M_raw, phase, d_phase, and dM must share the same shape.")
        if cutlass.const_expr(mMRaw.shape[2] != 2):
            raise ValueError("M_raw/phase/d_phase/dM must be (BHC, L, 2).")
        if cutlass.const_expr(mDLogprefixHalf.shape[0] != mMRaw.shape[0] or mDLogprefixHalf.shape[1] != mMRaw.shape[1]):
            raise ValueError("d_logprefix_half must be (BHC, L).")

        BHC = cute.size(mMRaw.shape[0])
        warps_per_block = self.num_threads // 32
        self.kernel(
            mMRaw,
            mPhase,
            mDPhase,
            mDLogprefixHalf,
            mDM,
            BHC,
        ).launch(
            grid=[cute.ceil_div(BHC, warps_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mMRaw: cute.Tensor,
        mPhase: cute.Tensor,
        mDPhase: cute.Tensor,
        mDLogprefixHalf: cute.Tensor,
        mDM: cute.Tensor,
        BHC: cutlass.Int32,
    ) -> None:
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        warps_per_block = self.num_threads // 32
        bhc = bidx * warps_per_block + warp
        if cute.elem_less(bhc, BHC) and lane == 0:
            L = cute.size(mMRaw.shape[1])
            eps = cutlass.Float32(1.0e-20)

            carry_re = cutlass.Float32(0.0)
            carry_im = cutlass.Float32(0.0)
            dlogr_running = cutlass.Float32(0.0)

            for t_it in cutlass.range(L, unroll=1):
                t = (L - 1) - t_it

                dlogr_running += cutlass.Float32(mDLogprefixHalf[bhc, t])

                mr = cutlass.Float32(mMRaw[bhc, t, 0])
                mi = cutlass.Float32(mMRaw[bhc, t, 1])
                mag2 = mr * mr + mi * mi
                inv_mag = cutlass.Float32(cute.math.rsqrt(mag2 + eps))
                mag = mag2 * inv_mag
                ur = mr * inv_mag
                ui = mi * inv_mag

                dpr = cutlass.Float32(mDPhase[bhc, t, 0]) + carry_re
                dpi = cutlass.Float32(mDPhase[bhc, t, 1]) + carry_im

                ppr = cutlass.Float32(1.0)
                ppi = cutlass.Float32(0.0)
                if t == 0:
                    pass
                else:
                    ppr = cutlass.Float32(mPhase[bhc, t - 1, 0])
                    ppi = cutlass.Float32(mPhase[bhc, t - 1, 1])

                # Scalar complex gradient: d_unit = grad(total wrt p_prev * unit).
                d_unit_re = dpr * ppr + dpi * ppi
                d_unit_im = -dpr * ppi + dpi * ppr
                carry_re = dpr * ur + dpi * ui
                carry_im = -dpr * ui + dpi * ur

                dot = ur * d_unit_re + ui * d_unit_im
                dphase_m_re = (d_unit_re - ur * dot) / mag
                dphase_m_im = (d_unit_im - ui * dot) / mag

                scale = cutlass.Float32(0.5) * dlogr_running / (mag2 + eps)
                dmag_m_re = scale * mr
                dmag_m_im = scale * mi

                mDM[bhc, t, 0] = dphase_m_re + dmag_m_re
                mDM[bhc, t, 1] = dphase_m_im + dmag_m_im


def _get_compiled_phase_reduce(
    Q: torch.Tensor,
    phase: torch.Tensor,
    d_phase: torch.Tensor,
) -> object:
    device_index = 0 if Q.device.index is None else int(Q.device.index)
    key: _CompiledPhaseReduceKey = (
        device_index,
        tuple(int(x) for x in Q.shape),
        tuple(int(x) for x in phase.shape),
        tuple(int(x) for x in d_phase.shape),
    )
    compiled = _COMPILED_PHASE_REDUCE.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanParamPhaseReduce()
    compiled = cute.compile(
        kernel,
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(Q, assumed_align=Q.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
    )
    _COMPILED_PHASE_REDUCE[key] = compiled
    return compiled


def _get_compiled_phase_scan(
    M_raw: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    dM: torch.Tensor,
) -> object:
    device_index = 0 if M_raw.device.index is None else int(M_raw.device.index)
    key: _CompiledPhaseScanKey = (
        device_index,
        tuple(int(x) for x in M_raw.shape),
        tuple(int(x) for x in d_logprefix_half.shape),
        tuple(int(x) for x in dM.shape),
    )
    compiled = _COMPILED_PHASE_SCAN.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanParamPhaseScan()
    compiled = cute.compile(
        kernel,
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(d_logprefix_half, assumed_align=d_logprefix_half.element_size()),
        from_dlpack(dM, assumed_align=dM.element_size()),
    )
    _COMPILED_PHASE_SCAN[key] = compiled
    return compiled


def _packed_causal_scales(logprefix_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the stable packed-contract scale tensors.

    ``logprefix_half`` stores half of the cumulative log-magnitude prefix. The
    packed dense path uses:

    - ``row_scale[t] = exp(2 * lp[t])`` for the off-term
    - ``scale[t, s] = exp(2 * (lp[t] - lp[s]))`` for the causal diagonal terms

    The explicit causal mask keeps the would-be undefined upper triangle out of
    the computation entirely instead of relying on later multiplication by zero.
    """

    if logprefix_half.ndim != 2:
        raise ValueError(
            f"logprefix_half must be rank-2 packed metadata. Got {tuple(logprefix_half.shape)}."
        )
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


def _dlogprefix_half_packed(
    score_prev: torch.Tensor,
    score_curr: torch.Tensor,
    dSprev: torch.Tensor,
    dScurr: torch.Tensor,
    y_off: torch.Tensor,
    scale: torch.Tensor,
    d_out_flat: torch.Tensor,
) -> torch.Tensor:
    """Exact packed-contract gradient for cumulative ``logprefix_half``.

    For the diagonal terms, each ``lp[k]`` contributes with opposite signs to
    row ``k`` and column ``k`` of the stable segment-ratio matrix. Writing that
    contribution explicitly as row-sum minus column-sum avoids building an
    autograd graph for this short metadata path.
    """

    e_prev = dSprev * score_prev * scale
    e_curr = dScurr * score_curr * scale
    return (
        2.0 * (d_out_flat * y_off).sum(dim=-1)
        + 2.0 * (e_prev.sum(dim=2) - e_prev.sum(dim=1))
        + 2.0 * (e_curr.sum(dim=2) - e_curr.sum(dim=1))
    ).contiguous()


def _packed_phase_prefix(M_raw: torch.Tensor) -> torch.Tensor:
    """Build the unit-complex phase prefix from raw packed ``M``."""

    m_c = torch.view_as_complex(M_raw.contiguous())
    mag = m_c.abs().clamp_min(torch.finfo(torch.float32).tiny)
    unit = m_c / mag
    return torch.cumprod(unit, dim=1)


def _chunk_scan_bwd_param_from_intermediates(
    Qf: torch.Tensor,
    Kprevf: torch.Tensor,
    Kcurrf: torch.Tensor,
    phase: torch.Tensor,
    M_raw: torch.Tensor,
    dQ: torch.Tensor,
    dKprev: torch.Tensor,
    dKcurr: torch.Tensor,
    d_logprefix_half: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
) -> torch.Tensor:
    """Map packed exact intermediates onto public ``dM``."""
    BHC, L, _ = map(int, Qf.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Packed leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    phase_raw = torch.view_as_real(phase).to(dtype=torch.float32).contiguous()
    d_phase = torch.empty_like(phase_raw)
    compiled_reduce = _get_compiled_phase_reduce(Qf, phase_raw, d_phase)
    compiled_reduce(
        from_dlpack(Qf, assumed_align=Qf.element_size()),
        from_dlpack(Kprevf, assumed_align=Kprevf.element_size()),
        from_dlpack(Kcurrf, assumed_align=Kcurrf.element_size()),
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(dKprev, assumed_align=dKprev.element_size()),
        from_dlpack(dKcurr, assumed_align=dKcurr.element_size()),
        from_dlpack(phase_raw, assumed_align=phase_raw.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
    )

    dM = torch.empty_like(M_raw)
    compiled_scan = _get_compiled_phase_scan(M_raw, d_logprefix_half, dM)
    compiled_scan(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase_raw, assumed_align=phase_raw.element_size()),
        from_dlpack(d_phase, assumed_align=d_phase.element_size()),
        from_dlpack(d_logprefix_half, assumed_align=d_logprefix_half.element_size()),
        from_dlpack(dM, assumed_align=dM.element_size()),
    )
    return dM.reshape(batch_size, n_heads, T_pad, 2).to(dtype=torch.float32).contiguous()


def chunk_scan_bwd_param_cute(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Vprev: torch.Tensor,
    Kcurr: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
    M_raw: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Compute ``dM`` for ``chunk_scan`` from cached packed forward tensors."""

    tensors = (
        ("Q", Q),
        ("Kprev", Kprev),
        ("Vprev", Vprev),
        ("Kcurr", Kcurr),
        ("Vcurr", Vcurr),
        ("logprefix_half", logprefix_half),
        ("Z0", Z0),
        ("M_raw", M_raw),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q.ndim != 4 or Kprev.ndim != 4 or Kcurr.ndim != 4:
        raise ValueError("Q/K tensors must be rank-4 packed tensors.")
    if Q.shape != Kprev.shape or Q.shape != Kcurr.shape:
        raise ValueError(
            "Q, Kprev, and Kcurr must share the same packed D contract. Got "
            f"{tuple(Q.shape)}, {tuple(Kprev.shape)}, {tuple(Kcurr.shape)}."
        )
    if Vprev.shape != Vcurr.shape or Vprev.ndim != 4 or Vprev.shape[2] != 1:
        raise ValueError("Vprev/Vcurr must be packed as (BHC, L, 1, P).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError("logprefix_half must be (BHC, L) matching Q.")
    if Z0.ndim != 4 or Z0.shape[0] != Q.shape[0] or Z0.shape[2] != 1:
        raise ValueError("Z0 must be packed as (BHC, P, 1, D).")
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)}."
        )
    if d_out.ndim != 4 or d_out.shape[:2] != (batch_size, n_heads) or int(d_out.shape[2]) != T:
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P). Got "
            f"{tuple(d_out.shape)}."
        )

    BHC, L, _, _ = map(int, Q.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q."
        )

    P = int(Vprev.shape[-1])
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
    dSprev = torch.bmm(d_out_flat, Vprevf.transpose(1, 2))
    dScurr = torch.bmm(d_out_flat, Vcurrf.transpose(1, 2))
    dScore_prev = dSprev * scale
    dScore_curr = dScurr * scale
    y_off = torch.bmm(Qf, Z0f.transpose(1, 2)) * row_scale

    dQ = (
        torch.bmm(d_out_flat * row_scale, Z0f)
        + torch.bmm(dScore_prev, Kprevf)
        + torch.bmm(dScore_curr, Kcurrf)
    )
    dKprev = torch.bmm(dScore_prev.transpose(1, 2), Qf)
    dKcurr = torch.bmm(dScore_curr.transpose(1, 2), Qf)
    d_logprefix_half = _dlogprefix_half_packed(
        score_prev,
        score_curr,
        dSprev,
        dScurr,
        y_off,
        scale,
        d_out_flat,
    )

    phase = _packed_phase_prefix(M_raw)
    dM = _chunk_scan_bwd_param_from_intermediates(
        Qf,
        Kprevf,
        Kcurrf,
        phase,
        M_raw,
        dQ,
        dKprev,
        dKcurr,
        d_logprefix_half,
        batch_size=batch_size,
        n_heads=n_heads,
    )
    return dM[:, :, :T, :].contiguous()


__all__ = ["chunk_scan_bwd_param_cute"]
