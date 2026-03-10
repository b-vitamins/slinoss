"""CuTe backward slice for ``chunk_scan`` gradients into ``B`` and ``K``.

Logical contract
----------------
This slice consumes cached forward-packed tensors plus the lightweight raw
metadata needed to map packed key gradients back to the public operator inputs:

- ``Q_rev``: ``flip(Q, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Vprev_rev``: ``flip(Vprev, dim=1)``, shape ``(BHC, L, 1, P)``
- ``Vcurr_rev``: ``flip(Vcurr, dim=1)``, shape ``(BHC, L, 1, P)``
- ``neg_logprefix_half_rev``: ``-flip(logprefix_half, dim=1)``, shape ``(BHC, L)``
- ``phase``: ``(BHC, L, 2)``, the unit-complex phase prefix from ``M_raw``
- ``K_raw``: ``(BHC, L, 2, 2)``, raw public taps in packed-complex form
- ``B_raw``: ``(BHC, L, D)``, raw public ``B`` rows in interleaved ``2N``
- ``B_head``: ``(BHC, D)``, per-chunk boundary ``B`` input used at ``t = 0``
- ``d_out``: ``(B, H, T, P)``

Why this contract
-----------------
The packed-real key gradient is another causal attention-like contraction after:

- reversing time,
- swapping the forward value vectors into the query role,
- using reversed ``d_out`` as the key vectors,
- keeping the reversed/negated logprefix metadata.

After the dense packed ``dKprev/dKcurr`` work, the remaining map back to the
public ``(B, B_prev, K)`` contract is a short explicit complex scatter. That
host-side algebra stays readable and avoids inventing another CuTe layout
family for a non-dominant reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.bwd.chunk_increment.common import (
    _scalar_grad_from_vec,
)
from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_chunk_scan,
    _get_compiled_phase,
)
from slinoss.ops.v2x2ssd.reference import (
    _as_complex_pairs,
    _complex_dtype_from_real,
    _pack_complex_pairs,
)


@dataclass
class _ChunkScanBwdDBScratch:
    K_zero: torch.Tensor
    V_zero: torch.Tensor
    Z0_zero: torch.Tensor
    dKprev_rev: torch.Tensor
    dKcurr_rev: torch.Tensor


_ScratchKey = tuple[int, torch.dtype, int, int, int, int]
_SCRATCH_DB: dict[_ScratchKey, _ChunkScanBwdDBScratch] = {}


def prepare_chunk_scan_bwd_db_operands(
    Q: torch.Tensor,
    Vprev: torch.Tensor,
    Vcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    M_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cached reverse-time operands plus phase metadata for the ``dB`` slice."""
    if Q.ndim != 4 or Vprev.ndim != 4 or Vcurr.ndim != 4:
        raise ValueError("Q/Vprev/Vcurr must be rank-4 tensors.")
    if Q.shape[:3] != Vprev.shape[:3] or Q.shape[:3] != Vcurr.shape[:3]:
        raise ValueError(
            "Q/Vprev/Vcurr must agree on the leading packed dims. Got "
            f"{tuple(Q.shape)}, {tuple(Vprev.shape)}, {tuple(Vcurr.shape)}."
        )
    if Q.shape[2] != 1 or Vprev.shape[2] != 1 or Vcurr.shape[2] != 1:
        raise ValueError("Packed Q/V tensors must be shaped as (BHC, L, 1, feat).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if M_raw.shape != (*Q.shape[:2], 2):
        raise ValueError(
            "M_raw must be (BHC, L, 2) matching Q. Got "
            f"{tuple(M_raw.shape)} for Q shape {tuple(Q.shape)}."
        )
    if not (
        Q.is_contiguous()
        and Vprev.is_contiguous()
        and Vcurr.is_contiguous()
        and logprefix_half.is_contiguous()
        and M_raw.is_contiguous()
    ):
        raise ValueError(
            "Q, Vprev, Vcurr, logprefix_half, and M_raw must be contiguous cached "
            "forward tensors."
        )

    phase = torch.empty(
        (M_raw.shape[0], M_raw.shape[1], 2),
        device=M_raw.device,
        dtype=torch.float32,
    )
    compiled_phase = _get_compiled_phase(M_raw, phase)
    compiled_phase(
        from_dlpack(M_raw, assumed_align=M_raw.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
    )
    return (
        torch.flip(Q, dims=[1]).contiguous(),
        torch.flip(Vprev, dims=[1]).contiguous(),
        torch.flip(Vcurr, dims=[1]).contiguous(),
        (-torch.flip(logprefix_half, dims=[1])).contiguous(),
        phase,
    )


def _get_db_scratch(
    *,
    vprev_rev: torch.Tensor,
    D: int,
) -> _ChunkScanBwdDBScratch:
    device_index = 0 if vprev_rev.device.index is None else int(vprev_rev.device.index)
    BHC, L, _, P = map(int, vprev_rev.shape)
    key: _ScratchKey = (
        device_index,
        vprev_rev.dtype,
        BHC,
        L,
        P,
        D,
    )
    scratch = _SCRATCH_DB.get(key)
    if scratch is not None:
        return scratch

    K_zero = torch.zeros_like(vprev_rev)
    V_zero = torch.zeros((BHC, L, 1, D), device=vprev_rev.device, dtype=vprev_rev.dtype)
    Z0_zero = torch.zeros(
        (BHC, D, 1, P), device=vprev_rev.device, dtype=vprev_rev.dtype
    )
    dKprev_rev = torch.empty(
        (BHC, L, 1, D), device=vprev_rev.device, dtype=torch.float32
    )
    dKcurr_rev = torch.empty_like(dKprev_rev)
    scratch = _ChunkScanBwdDBScratch(
        K_zero=K_zero,
        V_zero=V_zero,
        Z0_zero=Z0_zero,
        dKprev_rev=dKprev_rev,
        dKcurr_rev=dKcurr_rev,
    )
    _SCRATCH_DB[key] = scratch
    return scratch


def chunk_scan_bwd_db_cute(
    Q_rev: torch.Tensor,
    Vprev_rev: torch.Tensor,
    Vcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    phase: torch.Tensor,
    K_raw: torch.Tensor,
    B_raw: torch.Tensor,
    B_head: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(dB, dB_prev, dK)`` for ``chunk_scan`` from cached forward packs."""
    tensors = (
        ("Q_rev", Q_rev),
        ("Vprev_rev", Vprev_rev),
        ("Vcurr_rev", Vcurr_rev),
        ("neg_logprefix_half_rev", neg_logprefix_half_rev),
        ("phase", phase),
        ("K_raw", K_raw),
        ("B_raw", B_raw),
        ("B_head", B_head),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Q_rev.ndim != 4 or Vprev_rev.ndim != 4 or Vcurr_rev.ndim != 4:
        raise ValueError("Q_rev/Vprev_rev/Vcurr_rev must be rank-4 tensors.")
    if Q_rev.shape[:3] != Vprev_rev.shape[:3] or Q_rev.shape[:3] != Vcurr_rev.shape[:3]:
        raise ValueError(
            "Q_rev/Vprev_rev/Vcurr_rev must share leading packed dims. Got "
            f"{tuple(Q_rev.shape)}, {tuple(Vprev_rev.shape)}, {tuple(Vcurr_rev.shape)}."
        )
    if Q_rev.shape[2] != 1 or Vprev_rev.shape[2] != 1 or Vcurr_rev.shape[2] != 1:
        raise ValueError("Packed reverse-time tensors must be (BHC, L, 1, feat).")
    if neg_logprefix_half_rev.shape != Q_rev.shape[:2]:
        raise ValueError(
            "neg_logprefix_half_rev must be (BHC, L) matching Q_rev. Got "
            f"{tuple(neg_logprefix_half_rev.shape)}."
        )
    if phase.shape != (*Q_rev.shape[:2], 2):
        raise ValueError(
            f"phase must be (BHC, L, 2) matching Q_rev. Got {tuple(phase.shape)}."
        )
    if K_raw.shape != (*Q_rev.shape[:2], 2, 2):
        raise ValueError(
            "K_raw must be (BHC, L, 2, 2). Got "
            f"{tuple(K_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_raw.shape != (*Q_rev.shape[:2], Q_rev.shape[-1]):
        raise ValueError(
            "B_raw must be (BHC, L, D) matching Q_rev. Got "
            f"{tuple(B_raw.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if B_head.shape != (Q_rev.shape[0], Q_rev.shape[-1]):
        raise ValueError(
            "B_head must be (BHC, D) matching Q_rev. Got "
            f"{tuple(B_head.shape)} for Q_rev shape {tuple(Q_rev.shape)}."
        )
    if (
        d_out.ndim != 4
        or d_out.shape[:2] != (batch_size, n_heads)
        or int(d_out.shape[2]) != T
    ):
        raise ValueError(
            "d_out must be (batch_size, n_heads, T, P). Got "
            f"{tuple(d_out.shape)} for {(batch_size, n_heads, T)}."
        )

    BHC, L, _, D = map(int, Q_rev.shape)
    P = int(Vprev_rev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Q_rev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Q_rev."
        )

    scratch = _get_db_scratch(vprev_rev=Vprev_rev, D=D)

    # This is the hot path in the eventual autograd wrapper. Like the other
    # CuTe backward stages, it assumes forward saved sane finite tensors and
    # avoids whole-tensor finite scans here.
    if T_pad != T:
        pad = T_pad - T
        d_out = torch.cat(
            [
                d_out,
                torch.zeros(
                    (batch_size, n_heads, pad, P),
                    device=d_out.device,
                    dtype=d_out.dtype,
                ),
            ],
            dim=2,
        )
    d_out_rev = torch.flip(
        d_out.reshape(BHC, L, 1, P).to(dtype=Vprev_rev.dtype), dims=[1]
    ).contiguous()
    # The packed key-gradient contraction needs the same inner-kernel prefix
    # renormalization as the packed ``dQ`` path: after reversing time, the
    # stable segment ratio is represented by half of the negated half-logprefix.
    half_neg_logprefix_half_rev = (0.5 * neg_logprefix_half_rev).contiguous()

    compiled_prev = _get_compiled_chunk_scan(
        Vprev_rev,
        d_out_rev,
        Q_rev,
        scratch.K_zero,
        scratch.V_zero,
        half_neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.dKprev_rev,
    )
    compiled_curr = _get_compiled_chunk_scan(
        Vcurr_rev,
        scratch.K_zero,
        scratch.V_zero,
        d_out_rev,
        Q_rev,
        half_neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.dKcurr_rev,
    )

    compiled_prev(
        from_dlpack(Vprev_rev, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(half_neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.dKprev_rev, assumed_align=16),
    )
    compiled_curr(
        from_dlpack(Vcurr_rev, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(d_out_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(half_neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.dKcurr_rev, assumed_align=16),
    )

    cplx_dtype = _complex_dtype_from_real(torch.float32)
    phase_c = (
        torch.view_as_complex(phase.contiguous()).to(dtype=cplx_dtype).unsqueeze(-1)
    )
    dKprev_packed = torch.flip(scratch.dKprev_rev.squeeze(2), dims=[1]).contiguous()
    dKcurr_packed = torch.flip(scratch.dKcurr_rev.squeeze(2), dims=[1]).contiguous()
    dKprev_c = _as_complex_pairs(dKprev_packed, name="dKprev_packed").to(
        dtype=cplx_dtype
    )
    dKcurr_c = _as_complex_pairs(dKcurr_packed, name="dKcurr_packed").to(
        dtype=cplx_dtype
    )

    # ``Kprev/Kcurr`` are packed ``conj(beta) * phase``. For the underlying
    # real 2x2 map, the gradient back to ``beta`` is ``phase * conj(dK)``.
    d_beta_prev = phase_c * torch.conj(dKprev_c)
    d_beta_curr = phase_c * torch.conj(dKcurr_c)

    b_curr = _as_complex_pairs(B_raw, name="B_raw").to(dtype=cplx_dtype)
    b_head_c = (
        _as_complex_pairs(B_head.unsqueeze(1), name="B_head")
        .squeeze(1)
        .to(dtype=cplx_dtype)
    )
    b_prev_seq = torch.empty_like(b_curr)
    b_prev_seq[:, 0, :] = b_head_c
    if L > 1:
        b_prev_seq[:, 1:, :] = b_curr[:, :-1, :]

    k_prev_c = torch.view_as_complex(K_raw[:, :, 0, :].contiguous()).to(
        dtype=cplx_dtype
    )
    k_curr_c = torch.view_as_complex(K_raw[:, :, 1, :].contiguous()).to(
        dtype=cplx_dtype
    )

    dB_curr_c = torch.conj(k_curr_c).unsqueeze(-1) * d_beta_curr
    dB_prev_seq_c = torch.conj(k_prev_c).unsqueeze(-1) * d_beta_prev
    dK_prev_tap_c = _scalar_grad_from_vec(b_prev_seq, d_beta_prev)
    dK_curr_tap_c = _scalar_grad_from_vec(b_curr, d_beta_curr)

    N = D // 2
    dB_blk = dB_curr_c.reshape(batch_size, n_heads, n_chunks, L, N).clone()
    dB_prev_view = dB_prev_seq_c.reshape(batch_size, n_heads, n_chunks, L, N)
    if L > 1:
        dB_blk[:, :, :, :-1, :] += dB_prev_view[:, :, :, 1:, :]

    d_head_c = dB_prev_view[:, :, :, 0, :]
    if n_chunks > 1:
        dB_blk[:, :, :-1, -1, :] += d_head_c[:, :, 1:, :]

    dB_prev0_c = d_head_c[:, :, 0, :].contiguous()
    dB = _pack_complex_pairs(
        dB_blk.reshape(batch_size, n_heads, T_pad, N),
        real_dtype=torch.float32,
    )[:, :, :T, :].contiguous()
    dB_prev = _pack_complex_pairs(dB_prev0_c, real_dtype=torch.float32)

    dK_prev_real = torch.view_as_real(
        dK_prev_tap_c.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK_curr_real = torch.view_as_real(
        dK_curr_tap_c.reshape(batch_size, n_heads, n_chunks, L)
    ).to(dtype=torch.float32)
    dK = (
        torch.stack((dK_prev_real, dK_curr_real), dim=4)
        .reshape(batch_size, n_heads, T_pad, 2, 2)[:, :, :T, :, :]
        .contiguous()
    )
    return dB, dB_prev, dK


__all__ = [
    "prepare_chunk_scan_bwd_db_operands",
    "chunk_scan_bwd_db_cute",
]
