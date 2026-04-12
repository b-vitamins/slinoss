"""CuTe backward kernels for the ``v2x2ssd`` chunk-increment stage.

This file implements the minimal shared surface for the staged backward split:

  - ``db``: tensor-core ``dB`` workhorse plus ``DMsumPart`` partial reductions
  - ``du``: tensor-core ``dU`` workhorse
  - ``boundary``: rank-1 boundary gradients plus ``DMp0``
  - ``param_scan``: per-chunk scan-backward for ``dM``, ``dKprev``, and ``dKcurr``

The package-level ``__init__.py`` owns orchestration. This module should stay
small and only hold truly shared helpers.
"""

import torch

import cutlass
import cutlass.utils as utils


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _default_tc_k_tile(k_extent: int) -> int:
    if k_extent >= 32 and k_extent % 32 == 0:
        return 32
    if k_extent >= 16 and k_extent % 16 == 0:
        return 16
    raise ValueError("Tensor-core K extent must be divisible by 16.")


def _default_async_copy_bits(
    *,
    dtype_width: int,
    major_mode: utils.LayoutEnum,
    tile_m: int,
    tile_k: int,
    num_threads: int,
) -> int:
    """Pick the widest async-copy width whose thread tiling is actually legal."""

    for copy_bits in (128, 64, 32, 16):
        if copy_bits < dtype_width or copy_bits % dtype_width != 0:
            continue
        copy_elems = copy_bits // dtype_width

        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            for tm in range(1, num_threads + 1):
                if num_threads % tm != 0:
                    continue
                tn = num_threads // tm
                tile_k_seg = tn * copy_elems
                if (int(tile_k) % tile_k_seg) != 0:
                    continue
                return copy_bits
            continue

        shape_dim_0 = (int(tile_m) + int(copy_elems) - 1) // int(copy_elems)
        if shape_dim_0 <= num_threads:
            for cand in range(shape_dim_0, num_threads + 1):
                if num_threads % cand == 0:
                    return copy_bits

    raise ValueError("Failed to find a legal async-copy width for the given tile.")


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


__all__ = [
    "_assumed_align",
    "_default_async_copy_bits",
    "_default_tc_k_tile",
    "_next_pow2",
    "_torch_to_cutlass_dtype",
]
