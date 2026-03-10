"""CuTe forward kernel for the ``v2x2ssd`` state-passing stage.

Logical tensors
---------------
- ``inc``: ``(B, H, C, P, D)``
- ``m_chunk``: ``(B, H, C, 2)``
- ``chunk_starts``: ``(B, H, C, P, D)``
- ``final_state``: ``(B, H, P, D)``

The current CuTe path stores ``chunk_starts`` and ``final_state`` in a compact
transport dtype (currently fp16 when compute is fp32). The recurrence itself
still runs in fp32; the reduced-precision storage exists only to keep the
bandwidth-bound stage competitive and feeds directly into the half-precision
chunk-scan packing path.

Layout / launch contract
------------------------
- Flatten ``S = P * D`` as the hot contiguous axis.
- Grid is ``(ceil_div(S, tile), B * H, 1)``.
- Each CTA owns one contiguous ``S`` tile for one ``(batch, head)`` row.
- Thread/value layout is linear contiguous ownership:
  ``(num_threads, elems_per_thread)``.
- Predication is only required for the tail ``S`` tile.

Numerical contract
------------------
This stage applies the direct complex-scalar recurrence on adjacent
``(re, im)`` pairs. No divisions or reciprocal prefix factors are used.
"""

from __future__ import annotations

from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from slinoss.ops.v2x2ssd.reference import (
    _resolve_dtypes,
    _validate_state_passing_inputs,
)


_CompiledKey = tuple[
    int,
    torch.dtype,
    torch.dtype,
    torch.dtype,
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    bool,
]
_COMPILED_STATE_PASSING: dict[_CompiledKey, Callable[..., object]] = {}


def _elem_bits(dt: torch.dtype) -> int:
    if dt == torch.float32:
        return 32
    if dt in (torch.float16, torch.bfloat16):
        return 16
    raise TypeError(f"Unsupported dtype: {dt}")


def _choose_copy_bits_for_linear_tiles(
    t: torch.Tensor,
    tile_stride_elems: int,
    *,
    elems_per_thread: int,
    candidates_bits: tuple[int, ...] = (128, 64, 32),
) -> int:
    """Pick the widest CopyUniversalOp width safe for all tile starts."""
    eb = _elem_bits(t.dtype)
    elem_bytes = t.element_size()
    stride_bytes = tile_stride_elems * elem_bytes

    best = eb
    for bits in candidates_bits:
        if bits < eb or bits % eb != 0:
            continue
        vec_elems = bits // eb
        if elems_per_thread % vec_elems != 0:
            continue
        align = bits // 8
        if (t.data_ptr() % align) == 0 and (stride_bytes % align) == 0:
            best = bits
            break
    return best


class _StatePassingAmpere:
    """Bandwidth-oriented state-passing kernel with fp32 math."""

    def __init__(
        self,
        *,
        num_threads: int = 128,
        pairs_per_thread: int = 8,
        copy_bits_in: int,
        copy_bits_out: int,
        has_init: bool,
    ) -> None:
        self.num_threads = int(num_threads)
        self.pairs_per_thread = int(pairs_per_thread)
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive.")
        if self.num_threads % 32 != 0:
            raise ValueError("num_threads must be a multiple of 32.")
        if self.pairs_per_thread <= 0:
            raise ValueError("pairs_per_thread must be positive.")

        # Keep each thread on whole complex pairs.
        self.elems_per_thread = 2 * self.pairs_per_thread
        self.tile = self.num_threads * self.elems_per_thread

        self.copy_bits_in = int(copy_bits_in)
        self.copy_bits_out = int(copy_bits_out)
        self.has_init = bool(has_init)

    @cute.jit
    def __call__(
        self,
        inc: cute.Tensor,  # (B,H,C,P,D)
        m_chunk: cute.Tensor,  # (B,H,C,2)
        out_starts: cute.Tensor,  # (B,H,C,P,D) fp32
        out_final: cute.Tensor,  # (B,H,P,D) fp32
        init_or_dummy: cute.Tensor,  # (B,H,P,D) or ignored
    ) -> None:
        B, H, C, P, D = inc.shape
        BH = B * H
        S = P * D

        layout_bcs = cute.make_layout((BH, C, S), stride=(C * S, S, 1))
        layout_bcm = cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1))
        layout_bs = cute.make_layout((BH, S), stride=(S, 1))

        inc_flat = cute.make_tensor(inc.iterator, layout_bcs)
        m_flat = cute.make_tensor(m_chunk.iterator, layout_bcm)
        out_starts_flat = cute.make_tensor(out_starts.iterator, layout_bcs)
        out_final_flat = cute.make_tensor(out_final.iterator, layout_bs)
        init_flat = cute.make_tensor(init_or_dummy.iterator, layout_bs)

        tv_layout = cute.make_layout(
            (self.num_threads, self.elems_per_thread),
            stride=(self.elems_per_thread, 1),
        )

        idS = cute.make_identity_tensor(S)
        cS = cute.zipped_divide(idS, tiler=cute.make_layout(self.tile))

        grid_x = cute.ceil_div(S, self.tile)
        grid_y = BH

        self.kernel(
            inc_flat,
            m_flat,
            out_starts_flat,
            out_final_flat,
            init_flat,
            cS,
            tv_layout,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
        )

    @cute.kernel
    def kernel(
        self,
        inc_flat: cute.Tensor,
        m_flat: cute.Tensor,
        out_starts_flat: cute.Tensor,
        out_final_flat: cute.Tensor,
        init_flat: cute.Tensor,
        cS: cute.Tensor,
        tv_layout: cute.Layout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        bh = bidy
        tile_idx = bidx

        S = inc_flat.shape[2]
        C = inc_flat.shape[1]

        tile_start = cutlass.Int32(self.tile) * tile_idx
        residue = S - tile_start
        is_partial_tile = cute.elem_less(residue, cutlass.Int32(self.tile))

        cta_coord = (None, tile_idx)
        cta_crd = cS[cta_coord]
        tid_crd = cute.composition(cta_crd, tv_layout)
        thr_crd = tid_crd[tidx, None]

        copy_in_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            inc_flat.element_type,
            num_bits_per_copy=self.copy_bits_in,
        )
        copy_out_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            out_starts_flat.element_type,
            num_bits_per_copy=self.copy_bits_out,
        )
        copy_m = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            m_flat.element_type,
            num_bits_per_copy=m_flat.element_type.width * 2,
        )
        tile_layout = cute.make_layout(cS.shape[0])

        gInit = init_flat[bh, None]
        tInit = cute.zipped_divide(gInit, tiler=tile_layout)
        ctaInit = tInit[cta_coord]
        tidInit = cute.composition(ctaInit, tv_layout)
        thrInit = tidInit[tidx, None]

        accZ = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)
        accZ.fill(0.0)
        if cutlass.const_expr(self.has_init):
            for i in cutlass.range_constexpr(cute.size(accZ)):
                coord = cutlass.Int32(0)
                coord = thr_crd[i]
                if (not is_partial_tile) or cute.elem_less(coord, S):
                    accZ[i] = init_flat[bh, coord].to(cutlass.Float32)

        frgIn = cute.make_rmem_tensor_like(thrInit)
        frgOut = cute.make_rmem_tensor(thrInit.shape, cutlass.Float32)
        frgOutStore = cute.make_rmem_tensor(thrInit.shape, out_starts_flat.element_type)
        pairs_per_thread = cute.size(accZ) // 2

        for c in cutlass.range(C, unroll=1):
            gOut = out_starts_flat[bh, c, None]
            tOut = cute.zipped_divide(gOut, tiler=tile_layout)
            ctaOut = tOut[cta_coord]
            tidOut = cute.composition(ctaOut, tv_layout)
            thrOut = tidOut[tidx, None]

            frgOut.store(accZ.load())
            frgOutStore.store(frgOut.load().to(out_starts_flat.element_type))
            coord = cutlass.Int32(0)
            if is_partial_tile:
                for i in cutlass.range_constexpr(cute.size(frgOut)):
                    coord = thr_crd[i]
                    if cute.elem_less(coord, S):
                        out_starts_flat[bh, c, coord] = frgOutStore[i]
            else:
                cute.copy(copy_out_vec, frgOutStore, thrOut)

            gInc = inc_flat[bh, c, None]
            tInc = cute.zipped_divide(gInc, tiler=tile_layout)
            ctaInc = tInc[cta_coord]
            tidInc = cute.composition(ctaInc, tv_layout)
            thrInc = tidInc[tidx, None]

            frgIn.fill(0)
            coord = cutlass.Int32(0)
            if is_partial_tile:
                for i in cutlass.range_constexpr(cute.size(frgIn)):
                    coord = thr_crd[i]
                    if cute.elem_less(coord, S):
                        frgIn[i] = inc_flat[bh, c, coord]
            else:
                cute.copy(copy_in_vec, thrInc, frgIn)
            inc_f32 = frgIn.load().to(cutlass.Float32)

            gM = m_flat[bh, c, None]
            frgM = cute.make_rmem_tensor_like(gM)
            cute.copy(copy_m, gM, frgM)
            m_val = frgM.load().to(cutlass.Float32)
            mr, mi = m_val[0], m_val[1]

            for v in cutlass.range_constexpr(pairs_per_thread):
                base = v * 2
                zr = accZ[base + 0]
                zi = accZ[base + 1]

                accZ[base + 0] = (mr * zr - mi * zi) + inc_f32[base + 0]
                accZ[base + 1] = (mr * zi + mi * zr) + inc_f32[base + 1]

        gF = out_final_flat[bh, None]
        tF = cute.zipped_divide(gF, tiler=tile_layout)
        ctaF = tF[cta_coord]
        tidF = cute.composition(ctaF, tv_layout)
        thrF = tidF[tidx, None]

        frgOut.store(accZ.load())
        frgOutStore.store(frgOut.load().to(out_final_flat.element_type))
        coord = cutlass.Int32(0)
        if is_partial_tile:
            for i in cutlass.range_constexpr(cute.size(frgOut)):
                coord = thr_crd[i]
                if cute.elem_less(coord, S):
                    out_final_flat[bh, coord] = frgOutStore[i]
        else:
            cute.copy(copy_out_vec, frgOutStore, thrF)


def _compiled_key(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    out_starts: torch.Tensor,
    out_final: torch.Tensor,
    *,
    has_init: bool,
    device_index: int,
) -> _CompiledKey:
    return (
        device_index,
        inc.dtype,
        out_starts.dtype,
        out_final.dtype,
        tuple(int(x) for x in inc.shape),
        tuple(int(x) for x in inc.stride()),
        tuple(int(x) for x in m_chunk.shape),
        tuple(int(x) for x in m_chunk.stride()),
        has_init,
    )


def _get_compiled_state_passing(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    out_starts: torch.Tensor,
    out_final: torch.Tensor,
    init_or_dummy: torch.Tensor,
    *,
    has_init: bool,
) -> Callable[..., object]:
    if inc.device.type != "cuda":
        raise ValueError("CuTe state_passing requires CUDA tensors.")

    device_index = 0 if inc.device.index is None else int(inc.device.index)
    key = _compiled_key(
        inc,
        m_chunk,
        out_starts,
        out_final,
        has_init=has_init,
        device_index=device_index,
    )
    compiled = _COMPILED_STATE_PASSING.get(key)
    if compiled is not None:
        return compiled

    B, H, _, P, D = map(int, inc.shape)
    S = P * D
    pairs_per_thread = 8
    elems_per_thread = 2 * pairs_per_thread
    copy_bits_in = _choose_copy_bits_for_linear_tiles(
        inc, tile_stride_elems=S, elems_per_thread=elems_per_thread
    )
    copy_bits_out = _choose_copy_bits_for_linear_tiles(
        out_starts, tile_stride_elems=S, elems_per_thread=elems_per_thread
    )

    align_in = max(inc.element_size(), copy_bits_in // 8)
    align_out = max(out_starts.element_size(), copy_bits_out // 8)
    align_m = max(m_chunk.element_size(), 8)

    mInc = from_dlpack(inc, assumed_align=align_in)
    mM = from_dlpack(m_chunk, assumed_align=align_m)
    mOutStarts = from_dlpack(out_starts, assumed_align=align_out)
    mOutFinal = from_dlpack(out_final, assumed_align=align_out)
    mInit = from_dlpack(init_or_dummy, assumed_align=init_or_dummy.element_size())

    kernel = _StatePassingAmpere(
        num_threads=128,
        pairs_per_thread=pairs_per_thread,
        copy_bits_in=copy_bits_in,
        copy_bits_out=copy_bits_out,
        has_init=has_init,
    )
    compiled = cute.compile(kernel, mInc, mM, mOutStarts, mOutFinal, mInit)
    _COMPILED_STATE_PASSING[key] = compiled
    return compiled


def state_passing_cute(
    inc: torch.Tensor,
    m_chunk: torch.Tensor,
    *,
    initial_states: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagates state across chunk boundaries with a CuTe kernel."""
    batch_size, n_heads, n_chunks, N, P = _validate_state_passing_inputs(
        inc, m_chunk, initial_states
    )
    D = 2 * N

    rdtype, _ = _resolve_dtypes(
        input_dtypes=[inc.dtype, m_chunk.dtype],
        compute_dtype=compute_dtype,
        output_dtype=torch.float32,
        default_output_dtype=torch.float32,
    )
    if rdtype != torch.float32:
        raise ValueError(
            "The current CuTe state_passing kernel supports only float32 "
            f"compute. Got compute_dtype={rdtype}."
        )
    if inc.device.type != "cuda":
        raise ValueError("CuTe state_passing requires CUDA tensors.")

    inc_c = inc if inc.dtype == rdtype else inc.to(dtype=rdtype)
    m_c = m_chunk if m_chunk.dtype == rdtype else m_chunk.to(dtype=rdtype)
    inc_c = inc_c.contiguous()
    m_c = m_c.contiguous()

    if initial_states is None:
        init_c = torch.zeros(
            (batch_size, n_heads, P, D), device=inc.device, dtype=rdtype
        )
        has_init = True
    else:
        if initial_states.dtype != rdtype:
            init_c = initial_states.to(dtype=rdtype)
        else:
            init_c = initial_states
        init_c = init_c.contiguous()
        has_init = True

    storage_dtype = torch.float16 if rdtype == torch.float32 else rdtype
    chunk_starts = torch.empty(
        (batch_size, n_heads, n_chunks, P, D),
        device=inc.device,
        dtype=storage_dtype,
    )
    final_state = torch.empty(
        (batch_size, n_heads, P, D),
        device=inc.device,
        dtype=storage_dtype,
    )

    compiled = _get_compiled_state_passing(
        inc_c,
        m_c,
        chunk_starts,
        final_state,
        init_c,
        has_init=has_init,
    )

    mInc = from_dlpack(inc_c, assumed_align=inc_c.element_size())
    mM = from_dlpack(m_c, assumed_align=max(m_c.element_size(), 8))
    mOutStarts = from_dlpack(chunk_starts, assumed_align=chunk_starts.element_size())
    mOutFinal = from_dlpack(final_state, assumed_align=final_state.element_size())
    mInit = from_dlpack(init_c, assumed_align=init_c.element_size())

    compiled(mInc, mM, mOutStarts, mOutFinal, mInit)
    return chunk_starts, final_state


__all__ = ["state_passing_cute"]
