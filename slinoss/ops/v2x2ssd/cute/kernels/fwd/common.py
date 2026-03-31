"""Minimal shared helpers for the ``v2x2ssd`` forward CuTe package."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


_PTR_ARG_CACHE: dict[tuple[object, ...], tuple[object, int]] = {}
_PTR_ARG_CACHE_LIMIT = 32768


def _torch_to_cutlass_dtype(dt: torch.dtype) -> type[cutlass.Numeric]:
    if dt == torch.float16:
        return cutlass.Float16
    if dt == torch.bfloat16:
        return cutlass.BFloat16
    if dt == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported dtype: {dt}")


def _tc_input_dtype(
    input_dtype: torch.dtype, compute_dtype: torch.dtype | None
) -> torch.dtype:
    dt = input_dtype if compute_dtype is None else compute_dtype
    if dt in (torch.float16, torch.bfloat16):
        return dt
    if dt == torch.float32:
        return torch.float16
    raise TypeError(f"Unsupported compute dtype: {dt}")


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


def _ptr_arg_cache_key(t: torch.Tensor) -> tuple[object, ...]:
    device_index = (
        int(t.device.index)
        if t.device.type == "cuda" and t.device.index is not None
        else -1
    )
    return (
        t.device.type,
        device_index,
        int(id(t)),
        int(t.data_ptr()),
        t.dtype,
        tuple(int(dim) for dim in t.shape),
        tuple(int(step) for step in t.stride()),
    )


def _make_ptr_arg(t: torch.Tensor) -> tuple[object, int]:
    # Distinct tensor views can share a base address. Keep separate Pointer
    # instances for those views so multi-arg host wrappers do not collapse them.
    key = _ptr_arg_cache_key(t)
    cached = _PTR_ARG_CACHE.get(key)
    if cached is not None:
        return cached

    align = _assumed_align(t)
    cached = (
        make_ptr(
            _torch_to_cutlass_dtype(t.dtype),
            t.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=align,
        ),
        align,
    )
    if len(_PTR_ARG_CACHE) >= _PTR_ARG_CACHE_LIMIT:
        _PTR_ARG_CACHE.clear()
    _PTR_ARG_CACHE[key] = cached
    return cached


def _make_ptr_args(
    *tensors: torch.Tensor,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    ptrs: list[object] = []
    alignments: list[int] = []
    for tensor in tensors:
        ptr, align = _make_ptr_arg(tensor)
        ptrs.append(ptr)
        alignments.append(align)
    return tuple(ptrs), tuple(alignments)


def _pad_zero_time(
    tensor: torch.Tensor,
    *,
    T_pad: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(dtype=dtype).contiguous()
    T = int(tensor.shape[2])
    if T == T_pad:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=tensor.device, dtype=dtype)
    return torch.cat((tensor, pad), dim=2).contiguous()


def _pad_m_identity(M: torch.Tensor, *, T_pad: int) -> torch.Tensor:
    M = M.to(dtype=torch.float32).contiguous()
    T = int(M.shape[2])
    if T == T_pad:
        return M
    pad_shape = list(M.shape)
    pad_shape[2] = T_pad - T
    pad = torch.zeros(pad_shape, device=M.device, dtype=torch.float32)
    pad[..., 0] = 1.0
    return torch.cat((M, pad), dim=2).contiguous()


__all__ = [
    "_assumed_align",
    "_choose_copy_bits_for_linear_tiles",
    "_elem_bits",
    "_make_ptr_arg",
    "_make_ptr_args",
    "_pad_m_identity",
    "_pad_zero_time",
    "_tc_input_dtype",
    "_torch_to_cutlass_dtype",
]
