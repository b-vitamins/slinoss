"""CuTe backward helpers for the fused ``v2x2ssd`` state-passing stage.

The fused kernel walks the backward chunk recurrence over one contiguous
``S = P * D`` tile per CTA:

1) It writes ``d_inc[c] = d_z_{c+1}``.
2) It accumulates the local ``d_m_chunk[c]`` contribution from
   ``(chunk_starts[c], d_z_{c+1})``.
3) It updates ``d_z_c = d_chunk_starts[c] + conj(m_chunk[c]) * d_z_{c+1}``.

The full ``d_m_chunk`` reduction is completed by atomically accumulating the
per-tile partials into the global output.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
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


@dataclass(frozen=True)
class StatePassingLayoutBundle:
    layout_bcs: object
    layout_bcm: object
    layout_bs: object
    tile_layout: object
    tv_layout: object


def _make_layout_bundle(
    *,
    BH: int,
    C: int,
    S: int,
    cfg: _TileConfig,
) -> StatePassingLayoutBundle:
    return StatePassingLayoutBundle(
        layout_bcs=cute.make_layout((BH, C, S), stride=(C * S, S, 1)),
        layout_bcm=cute.make_layout((BH, C, 2), stride=(C * 2, 2, 1)),
        layout_bs=cute.make_layout((BH, S), stride=(S, 1)),
        tile_layout=cute.make_layout(cfg.tile),
        tv_layout=cute.make_layout(
            (cfg.num_threads, cfg.elems_per_thread),
            stride=(cfg.elems_per_thread, 1),
        ),
    )


@cute.jit
def _thread_tile_view(
    g_tensor: cute.Tensor,
    tile_layout: cute.Layout,
    cta_coord,
    tv_layout: cute.Layout,
    tidx: cutlass.Int32,
):
    t_tensor = cute.zipped_divide(g_tensor, tiler=tile_layout)
    cta_tensor = t_tensor[cta_coord]
    tid_tensor = cute.composition(cta_tensor, tv_layout)
    return tid_tensor[tidx, None]
