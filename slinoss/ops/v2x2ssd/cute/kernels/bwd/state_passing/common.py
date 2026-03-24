"""CuTe backward kernels for the ``v2x2ssd`` state-passing stage.

This stage is split into two bandwidth-oriented kernels:

1) Backprop through the chunk recurrence to produce gradients w.r.t. the
   per-chunk increments and the initial state:

   z_{c+1} = m_chunk[c] * z_c + inc[c]
   chunk_starts[c] = z_c

   Given upstream grads (d_chunk_starts, d_final), compute:
     d_inc[c] = d_z_{c+1}
     d_z_c = d_chunk_starts[c] + conj(m_chunk[c]) * d_z_{c+1}

2) Gradient w.r.t. the per-chunk complex transport parameters m_chunk[c].
   This is a reduction over the flattened state axis S=P*D. It consumes
   (chunk_starts, d_inc) and produces d_m_chunk.

Both kernels compute in fp32 and write fp32 outputs, matching the reference
math.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import torch


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


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
        if bits < eb:
            continue
        if bits % eb != 0:
            continue
        vec_elems = bits // eb
        if elems_per_thread % vec_elems != 0:
            continue
        align = bits // 8
        if (t.data_ptr() % align) == 0 and (stride_bytes % align) == 0:
            best = bits
            break
    return best


def _assumed_align(
    t: torch.Tensor,
    candidates_bytes: tuple[int, ...] = (16, 8, 4),
) -> int:
    """Return the widest safe assumed alignment for a tensor view."""
    elem_align = max(1, t.element_size())
    ptr = int(t.data_ptr())
    for align in candidates_bytes:
        if align < elem_align:
            continue
        if (ptr % align) == 0:
            return align
    return elem_align


@dataclass(frozen=True)
class _TileConfig:
    num_threads: int = 128
    pairs_per_thread: int = 8

    @property
    def elems_per_thread(self) -> int:
        return 2 * int(self.pairs_per_thread)

    @property
    def tile(self) -> int:
        return int(self.num_threads) * self.elems_per_thread
