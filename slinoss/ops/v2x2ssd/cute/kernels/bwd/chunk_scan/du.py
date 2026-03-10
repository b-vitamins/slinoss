"""CuTe backward slice for ``chunk_scan`` gradients into ``U`` and ``U_prev``.

Logical contract
----------------
This slice consumes cached forward-packed operands instead of the raw public
``U/M/K/B/C`` inputs:

- ``Q_rev``: ``flip(Q, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Kprev_rev``: ``flip(Kprev, dim=1)``, shape ``(BHC, L, 1, D)``
- ``Kcurr_rev``: ``flip(Kcurr, dim=1)``, shape ``(BHC, L, 1, D)``
- ``neg_logprefix_half_rev``: ``-flip(logprefix_half, dim=1)``, shape ``(BHC, L)``
- ``d_out``: ``(B, H, T, P)``

Why this contract
-----------------
For the packed-real inner kernel, the value gradient is another causal
attention-like pass after:

- reversing time,
- swapping the forward ``Q`` and ``K`` roles,
- negating the half-logprefix metadata.

Doing that transformation once in forward and saving the result is what keeps
this slice above the usefulness bar. Recomputing or repacking it in backward is
too expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import _get_compiled_chunk_scan


@dataclass
class _ChunkScanBwdDUScratch:
    K_zero: torch.Tensor
    V_zero: torch.Tensor
    Z0_zero: torch.Tensor
    out_prev_rev: torch.Tensor
    out_curr_rev: torch.Tensor


_ScratchKey = tuple[int, torch.dtype, int, int, int]
_SCRATCH_DU: dict[_ScratchKey, _ChunkScanBwdDUScratch] = {}


def prepare_chunk_scan_bwd_du_operands(
    Q: torch.Tensor,
    Kprev: torch.Tensor,
    Kcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the cached reverse-time contract for ``chunk_scan`` value grads."""
    if Q.ndim != 4 or Kprev.ndim != 4 or Kcurr.ndim != 4:
        raise ValueError("Q/Kprev/Kcurr must be rank-4 tensors.")
    if Q.shape != Kprev.shape or Q.shape != Kcurr.shape:
        raise ValueError(
            "Q, Kprev, and Kcurr must have the same packed-inner shape. Got "
            f"{tuple(Q.shape)}, {tuple(Kprev.shape)}, {tuple(Kcurr.shape)}."
        )
    if Q.shape[2] != 1:
        raise ValueError("Packed Q/K tensors must be shaped as (BHC, L, 1, D).")
    if logprefix_half.shape != Q.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching Q/K. Got "
            f"{tuple(logprefix_half.shape)} for Q shape {tuple(Q.shape)}."
        )
    if not (
        Q.is_contiguous()
        and Kprev.is_contiguous()
        and Kcurr.is_contiguous()
        and logprefix_half.is_contiguous()
    ):
        raise ValueError(
            "Q, Kprev, Kcurr, and logprefix_half must be contiguous cached "
            "forward tensors."
        )

    return (
        torch.flip(Q, dims=[1]).contiguous(),
        torch.flip(Kprev, dims=[1]).contiguous(),
        torch.flip(Kcurr, dims=[1]).contiguous(),
        (-torch.flip(logprefix_half, dims=[1])).contiguous(),
    )


def _get_du_scratch(
    *,
    q_rev: torch.Tensor,
    P: int,
) -> _ChunkScanBwdDUScratch:
    device_index = 0 if q_rev.device.index is None else int(q_rev.device.index)
    BHC, L, _, D = map(int, q_rev.shape)
    key: _ScratchKey = (
        device_index,
        q_rev.dtype,
        BHC,
        L,
        P,
    )
    scratch = _SCRATCH_DU.get(key)
    if scratch is not None:
        return scratch

    K_zero = torch.zeros_like(q_rev)
    V_zero = torch.zeros((BHC, L, 1, P), device=q_rev.device, dtype=q_rev.dtype)
    Z0_zero = torch.zeros((BHC, P, 1, D), device=q_rev.device, dtype=q_rev.dtype)
    out_prev_rev = torch.empty((BHC, L, 1, P), device=q_rev.device, dtype=torch.float32)
    out_curr_rev = torch.empty_like(out_prev_rev)
    scratch = _ChunkScanBwdDUScratch(
        K_zero=K_zero,
        V_zero=V_zero,
        Z0_zero=Z0_zero,
        out_prev_rev=out_prev_rev,
        out_curr_rev=out_curr_rev,
    )
    _SCRATCH_DU[key] = scratch
    return scratch


def chunk_scan_bwd_du_cute(
    Q_rev: torch.Tensor,
    Kprev_rev: torch.Tensor,
    Kcurr_rev: torch.Tensor,
    neg_logprefix_half_rev: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(dU, dU_prev)`` for ``chunk_scan`` from cached forward packs."""
    if (
        Q_rev.device.type != "cuda"
        or Kprev_rev.device.type != "cuda"
        or Kcurr_rev.device.type != "cuda"
        or neg_logprefix_half_rev.device.type != "cuda"
        or d_out.device.type != "cuda"
    ):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if not (
        Q_rev.is_contiguous()
        and Kprev_rev.is_contiguous()
        and Kcurr_rev.is_contiguous()
        and neg_logprefix_half_rev.is_contiguous()
        and d_out.is_contiguous()
    ):
        raise ValueError("chunk_scan backward cached operands and d_out must be contiguous.")
    if Q_rev.ndim != 4 or Kprev_rev.ndim != 4 or Kcurr_rev.ndim != 4:
        raise ValueError("Q_rev/Kprev_rev/Kcurr_rev must be rank-4 tensors.")
    if Q_rev.shape != Kprev_rev.shape or Q_rev.shape != Kcurr_rev.shape:
        raise ValueError(
            "Q_rev, Kprev_rev, and Kcurr_rev must have the same shape. Got "
            f"{tuple(Q_rev.shape)}, {tuple(Kprev_rev.shape)}, "
            f"{tuple(Kcurr_rev.shape)}."
        )
    if Q_rev.shape[2] != 1:
        raise ValueError("Packed reverse-time Q/K tensors must be (BHC, L, 1, D).")
    if neg_logprefix_half_rev.shape != Q_rev.shape[:2]:
        raise ValueError(
            "neg_logprefix_half_rev must be (BHC, L) matching Q_rev. Got "
            f"{tuple(neg_logprefix_half_rev.shape)} for Q_rev shape "
            f"{tuple(Q_rev.shape)}."
        )
    if d_out.ndim != 4:
        raise ValueError("d_out must be rank-4 (B, H, T, P).")
    if d_out.shape[:2] != (batch_size, n_heads):
        raise ValueError(
            "Leading d_out dims must match (batch_size, n_heads). Got "
            f"{tuple(d_out.shape[:2])} vs {(batch_size, n_heads)}."
        )
    if int(d_out.shape[2]) != T:
        raise ValueError(f"d_out T must match T={T}. Got {int(d_out.shape[2])}.")

    BHC, L, _, _ = map(int, Q_rev.shape)
    P = int(d_out.shape[-1])
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

    scratch = _get_du_scratch(q_rev=Q_rev, P=P)

    # This is the hot path in the eventual autograd wrapper. Like the other
    # CuTe backward stages, it assumes forward saved sane finite tensors and
    # does not scan whole tensors for non-finites here.
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
    V_rev = torch.flip(
        d_out.reshape(BHC, L, 1, P).to(dtype=Q_rev.dtype), dims=[1]
    ).contiguous()

    compiled_prev = _get_compiled_chunk_scan(
        Kprev_rev,
        Q_rev,
        V_rev,
        scratch.K_zero,
        scratch.V_zero,
        neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.out_prev_rev,
    )
    compiled_curr = _get_compiled_chunk_scan(
        scratch.K_zero,
        scratch.K_zero,
        scratch.V_zero,
        Kcurr_rev,
        V_rev,
        neg_logprefix_half_rev,
        scratch.Z0_zero,
        scratch.out_curr_rev,
    )

    compiled_prev(
        from_dlpack(Kprev_rev, assumed_align=16),
        from_dlpack(Q_rev, assumed_align=16),
        from_dlpack(V_rev, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.out_prev_rev, assumed_align=16),
    )
    compiled_curr(
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(Kcurr_rev, assumed_align=16),
        from_dlpack(V_rev, assumed_align=16),
        from_dlpack(neg_logprefix_half_rev, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.out_curr_rev, assumed_align=16),
    )

    dV_prev = torch.flip(scratch.out_prev_rev.squeeze(2), dims=[1]).contiguous()
    dV_curr = torch.flip(scratch.out_curr_rev.squeeze(2), dims=[1]).contiguous()

    dU_blk = dV_curr.reshape(batch_size, n_heads, n_chunks, L, P)
    dV_prev_blk = dV_prev.reshape(batch_size, n_heads, n_chunks, L, P)
    dU_blk[:, :, :, :-1, :] += dV_prev_blk[:, :, :, 1:, :]
    if n_chunks > 1:
        dU_blk[:, :, :-1, -1, :] += dV_prev_blk[:, :, 1:, 0, :]

    dU = dU_blk.reshape(batch_size, n_heads, T_pad, P)[:, :, :T, :].contiguous()
    dU_prev = dV_prev_blk[:, :, 0, 0, :].contiguous()
    return dU, dU_prev


__all__ = [
    "prepare_chunk_scan_bwd_du_operands",
    "chunk_scan_bwd_du_cute",
]
