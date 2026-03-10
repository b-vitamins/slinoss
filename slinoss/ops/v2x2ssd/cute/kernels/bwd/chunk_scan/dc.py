"""CuTe backward slice for ``chunk_scan`` gradients into ``C``.

Logical contract
----------------
This slice consumes cached forward-packed tensors plus a tiny metadata prep:

- ``Vprev, Vcurr``: ``(BHC, L, 1, P)``
- ``Kprev, Kcurr``: ``(BHC, L, 1, D)``
- ``logprefix_half``: ``(BHC, L)``
- ``half_logprefix_half``: ``0.5 * logprefix_half``
- ``Z0_q``: ``(BHC, D, 1, P)``, the off-term contract for the ``dQ`` analogue
- ``phase``: ``(BHC, L, 2)``, the unit-complex phase prefix used in forward

Why this contract
-----------------
The packed-real gradient into ``Q = pack(conj(C) * phase)`` splits into:

- a diagonal term, which is another packed attention-like contraction, and
- an off-term, which is the same inner kernel with only ``Z0`` active.

Keeping the transformed metadata cached is what keeps the hot path above the
usefulness bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.cute.kernels.fwd.chunk_scan import (
    _get_compiled_chunk_scan,
    _get_compiled_phase,
)


@dataclass
class _ChunkScanBwdDCScratch:
    K_zero: torch.Tensor
    V_zero: torch.Tensor
    Z0_zero: torch.Tensor
    dQ_diag: torch.Tensor
    dQ_off: torch.Tensor


_ScratchKey = tuple[int, torch.dtype, int, int, int, int]
_SCRATCH_DC: dict[_ScratchKey, _ChunkScanBwdDCScratch] = {}
_CompiledScatterKey = tuple[int, tuple[int, int, int], tuple[int, int, int]]
_COMPILED_DC_SCATTER: dict[_CompiledScatterKey, object] = {}


class _ChunkScanBwdDCScatter:
    """Exact float32 scatter from packed ``dQ`` into public ``dC``.

    Logical shape:
    - ``dQ``: ``(BHC, L, D)``, interleaved complex pairs in fp32
    - ``phase``: ``(BHC, L, 2)``, unit-complex prefix in fp32
    - output ``dC_pad``: ``(BH, T_pad, D)`` in fp32

    Major mode:
    - ``D`` is the contiguous hot axis.
    - each thread owns one complex pair for one row.

    Launch / mapping:
    - grid ``(pair_tiles, row_tiles, BHC)``
    - ``bhc = bh * n_chunks + chunk`` with ``global_t = chunk * L + row``
    - writes land directly in the public padded time layout, so trimming to ``T``
      is only a final cheap slice on the host.
    """

    def __init__(self, *, pair_tile: int, num_threads: int = 128) -> None:
        self.pair_tile = int(pair_tile)
        self.num_threads = int(num_threads)
        if self.pair_tile <= 0 or self.num_threads % self.pair_tile != 0:
            raise ValueError("num_threads must be divisible by pair_tile.")
        self.row_tile = self.num_threads // self.pair_tile

    @cute.jit
    def __call__(
        self,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(
            not (
                mDQ.element_type
                == mPhase.element_type
                == mDCPad.element_type
                == cutlass.Float32
            )
        ):
            raise TypeError("dC scatter expects Float32 tensors.")
        if cutlass.const_expr(mDQ.shape[0] != mPhase.shape[0] or mDQ.shape[1] != mPhase.shape[1]):
            raise ValueError("dQ and phase must agree on (BHC, L).")
        if cutlass.const_expr(mPhase.shape[2] != 2):
            raise ValueError("phase must be (BHC, L, 2).")

        BHC = cute.size(mDQ.shape[0])
        L = cute.size(mDQ.shape[1])
        pair_cols = cute.size(mDQ.shape[2]) // 2
        grid_x = cute.ceil_div(pair_cols, self.pair_tile)
        grid_y = cute.ceil_div(L, self.row_tile)
        self.kernel(mDQ, mPhase, mDCPad, n_chunks).launch(
            grid=[grid_x, grid_y, BHC],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        mDQ: cute.Tensor,
        mPhase: cute.Tensor,
        mDCPad: cute.Tensor,
        n_chunks: cutlass.Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        pair_tile_idx, row_tile_idx, bhc = cute.arch.block_idx()

        pair_local = tidx % self.pair_tile
        row_local = tidx // self.pair_tile
        row = row_tile_idx * self.row_tile + row_local
        pair_idx = pair_tile_idx * self.pair_tile + pair_local
        pair_cols = mDQ.shape[2] // 2

        if cute.elem_less(row, mDQ.shape[1]) and cute.elem_less(pair_idx, pair_cols):
            bh = bhc // n_chunks
            chunk = bhc - bh * n_chunks
            global_t = chunk * mDQ.shape[1] + row
            col = pair_idx * 2

            pr = cutlass.Float32(mPhase[bhc, row, 0])
            pi = cutlass.Float32(mPhase[bhc, row, 1])
            dqr = cutlass.Float32(mDQ[bhc, row, col + 0])
            dqi = cutlass.Float32(mDQ[bhc, row, col + 1])

            mDCPad[bh, global_t, col + 0] = dqr * pr + dqi * pi
            mDCPad[bh, global_t, col + 1] = dqr * pi - dqi * pr


def _get_compiled_dc_scatter(
    dQ: torch.Tensor,
    phase: torch.Tensor,
    dC_pad: torch.Tensor,
) -> object:
    device_index = 0 if dQ.device.index is None else int(dQ.device.index)
    key: _CompiledScatterKey = (
        device_index,
        tuple(int(x) for x in dQ.shape),
        tuple(int(x) for x in dC_pad.shape),
    )
    compiled = _COMPILED_DC_SCATTER.get(key)
    if compiled is not None:
        return compiled

    kernel = _ChunkScanBwdDCScatter(pair_tile=16)
    compiled = cute.compile(
        kernel,
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        int(dC_pad.shape[1] // dQ.shape[1]),
    )
    _COMPILED_DC_SCATTER[key] = compiled
    return compiled


def chunk_scan_bwd_dc_exact_cute(
    dQ: torch.Tensor,
    phase: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Scatter exact packed ``dQ`` into public ``dC`` with an fp32 CuTe kernel."""
    if dQ.device.type != "cuda" or phase.device.type != "cuda":
        raise ValueError("Exact CuTe dC scatter requires CUDA tensors.")
    if not dQ.is_contiguous() or not phase.is_contiguous():
        raise ValueError("dQ and phase must be contiguous.")
    if dQ.dtype != torch.float32 or phase.dtype != torch.float32:
        raise ValueError("Exact CuTe dC scatter expects float32 tensors.")
    if dQ.ndim != 3 or phase.shape != (*dQ.shape[:2], 2):
        raise ValueError(
            f"dQ must be (BHC, L, D) and phase must be (BHC, L, 2). Got {tuple(dQ.shape)} and {tuple(phase.shape)}."
        )

    BHC, L, D = map(int, dQ.shape)
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"dQ leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L

    dC_pad = torch.empty((BH, T_pad, D), device=dQ.device, dtype=torch.float32)
    compiled = _get_compiled_dc_scatter(dQ, phase, dC_pad)
    compiled(
        from_dlpack(dQ, assumed_align=dQ.element_size()),
        from_dlpack(phase, assumed_align=phase.element_size()),
        from_dlpack(dC_pad, assumed_align=dC_pad.element_size()),
        n_chunks,
    )
    return (
        dC_pad.reshape(batch_size, n_heads, T_pad, D)[:, :, :T, :].contiguous()
    )


def prepare_chunk_scan_bwd_dc_operands(
    M_raw: torch.Tensor,
    logprefix_half: torch.Tensor,
    Z0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cached metadata for the ``chunk_scan`` ``dC`` slice.

    Returns ``(phase, half_logprefix_half, Z0_q)``.
    """
    if M_raw.ndim != 3 or M_raw.shape[-1] != 2:
        raise ValueError(f"M_raw must be (BHC, L, 2). Got {tuple(M_raw.shape)}.")
    if logprefix_half.shape != M_raw.shape[:2]:
        raise ValueError(
            "logprefix_half must be (BHC, L) matching M_raw. Got "
            f"{tuple(logprefix_half.shape)} for M_raw shape {tuple(M_raw.shape)}."
        )
    if Z0.ndim != 4 or Z0.shape[0] != M_raw.shape[0] or Z0.shape[2] != 1:
        raise ValueError(
            "Z0 must be the packed forward tensor shaped as (BHC, P, 1, D). "
            f"Got {tuple(Z0.shape)}."
        )
    if not (
        M_raw.is_contiguous() and logprefix_half.is_contiguous() and Z0.is_contiguous()
    ):
        raise ValueError("M_raw, logprefix_half, and Z0 must be contiguous.")

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

    half_logprefix_half = (0.5 * logprefix_half).contiguous()
    Z0_q = Z0.squeeze(2).transpose(1, 2).unsqueeze(2).contiguous()
    return phase, half_logprefix_half, Z0_q


def _get_dc_scratch(
    *,
    Kprev: torch.Tensor,
    P: int,
) -> _ChunkScanBwdDCScratch:
    device_index = 0 if Kprev.device.index is None else int(Kprev.device.index)
    BHC, L, _, D = map(int, Kprev.shape)
    key: _ScratchKey = (
        device_index,
        Kprev.dtype,
        BHC,
        L,
        P,
        D,
    )
    scratch = _SCRATCH_DC.get(key)
    if scratch is not None:
        return scratch

    K_zero = torch.zeros((BHC, L, 1, P), device=Kprev.device, dtype=Kprev.dtype)
    V_zero = torch.zeros((BHC, L, 1, D), device=Kprev.device, dtype=Kprev.dtype)
    Z0_zero = torch.zeros((BHC, D, 1, P), device=Kprev.device, dtype=Kprev.dtype)
    dQ_diag = torch.empty((BHC, L, 1, D), device=Kprev.device, dtype=torch.float32)
    dQ_off = torch.empty_like(dQ_diag)
    scratch = _ChunkScanBwdDCScratch(
        K_zero=K_zero,
        V_zero=V_zero,
        Z0_zero=Z0_zero,
        dQ_diag=dQ_diag,
        dQ_off=dQ_off,
    )
    _SCRATCH_DC[key] = scratch
    return scratch


def chunk_scan_bwd_dc_cute(
    Vprev: torch.Tensor,
    Kprev: torch.Tensor,
    Vcurr: torch.Tensor,
    Kcurr: torch.Tensor,
    logprefix_half: torch.Tensor,
    half_logprefix_half: torch.Tensor,
    Z0_q: torch.Tensor,
    phase: torch.Tensor,
    d_out: torch.Tensor,
    *,
    batch_size: int,
    n_heads: int,
    T: int,
) -> torch.Tensor:
    """Compute ``dC`` for ``chunk_scan`` from cached packed forward tensors."""
    tensors = (
        ("Vprev", Vprev),
        ("Kprev", Kprev),
        ("Vcurr", Vcurr),
        ("Kcurr", Kcurr),
        ("logprefix_half", logprefix_half),
        ("half_logprefix_half", half_logprefix_half),
        ("Z0_q", Z0_q),
        ("phase", phase),
        ("d_out", d_out),
    )
    if any(t.device.type != "cuda" for _name, t in tensors):
        raise ValueError("CuTe chunk_scan backward requires CUDA tensors.")
    if any(not t.is_contiguous() for _name, t in tensors):
        raise ValueError(
            "chunk_scan backward cached operands and d_out must be contiguous."
        )
    if Vprev.shape != Vcurr.shape:
        raise ValueError(
            f"Vprev and Vcurr must have the same shape. Got {tuple(Vprev.shape)} "
            f"and {tuple(Vcurr.shape)}."
        )
    if Kprev.shape != Kcurr.shape:
        raise ValueError(
            f"Kprev and Kcurr must have the same shape. Got {tuple(Kprev.shape)} "
            f"and {tuple(Kcurr.shape)}."
        )
    if Vprev.ndim != 4 or Kprev.ndim != 4 or Vprev.shape[2] != 1 or Kprev.shape[2] != 1:
        raise ValueError("Packed V/K tensors must be rank-4 with a singleton dim2.")
    if (
        logprefix_half.shape != Kprev.shape[:2]
        or half_logprefix_half.shape != Kprev.shape[:2]
    ):
        raise ValueError("logprefix_half tensors must be (BHC, L) matching Kprev.")
    if phase.shape != (*Kprev.shape[:2], 2):
        raise ValueError(
            "phase must be (BHC, L, 2) matching Kprev. Got "
            f"{tuple(phase.shape)} for Kprev shape {tuple(Kprev.shape)}."
        )
    if Z0_q.ndim != 4 or Z0_q.shape[0] != Kprev.shape[0] or Z0_q.shape[2] != 1:
        raise ValueError(
            f"Z0_q must be shaped as (BHC, D, 1, P). Got {tuple(Z0_q.shape)}."
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

    BHC, L, _, D = map(int, Kprev.shape)
    P = int(Vprev.shape[-1])
    BH = int(batch_size) * int(n_heads)
    if BH <= 0 or BHC % BH != 0:
        raise ValueError(
            f"Kprev leading dim BHC={BHC} is not divisible by batch*heads={BH}."
        )
    n_chunks = BHC // BH
    T_pad = n_chunks * L
    if T > T_pad:
        raise ValueError(
            f"T={T} exceeds the cached padded length T_pad={T_pad} implied by Kprev."
        )
    if Z0_q.shape != (BHC, D, 1, P):
        raise ValueError(f"Z0_q must be {(BHC, D, 1, P)}. Got {tuple(Z0_q.shape)}.")

    scratch = _get_dc_scratch(Kprev=Kprev, P=P)

    # This path is meant to consume forward saved tensors directly. Like the
    # other CuTe backward slices, it assumes sane finite saved state and avoids
    # whole-tensor finite scans in the hot path.
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
    d_out_tc = d_out.reshape(BHC, L, 1, P).to(dtype=Kprev.dtype).contiguous()

    compiled_diag = _get_compiled_chunk_scan(
        d_out_tc,
        Vprev,
        Kprev,
        Vcurr,
        Kcurr,
        half_logprefix_half,
        scratch.Z0_zero,
        scratch.dQ_diag,
    )
    compiled_off = _get_compiled_chunk_scan(
        d_out_tc,
        scratch.K_zero,
        scratch.V_zero,
        scratch.K_zero,
        scratch.V_zero,
        logprefix_half,
        Z0_q,
        scratch.dQ_off,
    )

    compiled_diag(
        from_dlpack(d_out_tc, assumed_align=16),
        from_dlpack(Vprev, assumed_align=16),
        from_dlpack(Kprev, assumed_align=16),
        from_dlpack(Vcurr, assumed_align=16),
        from_dlpack(Kcurr, assumed_align=16),
        from_dlpack(half_logprefix_half, assumed_align=16),
        from_dlpack(scratch.Z0_zero, assumed_align=16),
        from_dlpack(scratch.dQ_diag, assumed_align=16),
    )
    compiled_off(
        from_dlpack(d_out_tc, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(scratch.K_zero, assumed_align=16),
        from_dlpack(scratch.V_zero, assumed_align=16),
        from_dlpack(logprefix_half, assumed_align=16),
        from_dlpack(Z0_q, assumed_align=16),
        from_dlpack(scratch.dQ_off, assumed_align=16),
    )

    dq = (scratch.dQ_diag.squeeze(2) + scratch.dQ_off.squeeze(2)).contiguous()
    return chunk_scan_bwd_dc_exact_cute(
        dq,
        phase,
        batch_size=batch_size,
        n_heads=n_heads,
        T=T,
    )


__all__ = [
    "prepare_chunk_scan_bwd_dc_operands",
    "chunk_scan_bwd_dc_cute",
    "chunk_scan_bwd_dc_exact_cute",
]
