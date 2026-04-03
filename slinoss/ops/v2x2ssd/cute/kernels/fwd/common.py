"""Minimal shared helpers for the ``v2x2ssd`` forward CuTe package."""

from __future__ import annotations

import torch
import cutlass
import cutlass.cute as cute


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


def _ensure_min_alignment(t: torch.Tensor, *, min_align: int) -> torch.Tensor:
    """Materialize a fresh contiguous buffer when the current view is under-aligned.

    The forward CuTe fast paths use 128-bit cp.async staging for fp16/bf16
    activations. Arbitrary contiguous views can legally be only 2-byte aligned,
    so the launcher must normalize them before they cross the JIT boundary.
    """
    if _assumed_align(t) >= int(min_align):
        return t
    aligned = t.clone(memory_format=torch.contiguous_format)
    if _assumed_align(aligned) < int(min_align):
        raise RuntimeError(
            f"Failed to materialize a buffer with assumed_align >= {min_align}."
        )
    return aligned


def _guard_prev_time_base(t: torch.Tensor, *, min_align: int) -> torch.Tensor:
    """Prepend a valid guard row before the visible time axis without changing layout.

    The forward chunk-scan kernel forms a ``domain_offset(..., -1, ...)`` view over
    flattened time-major inputs so it can read the previous timestep directly. For
    the very first logical row, that shifted view would otherwise point before the
    allocation base. This helper materializes one guarded row of backing storage
    before the visible tensor while preserving the tensor's logical shape/stride.
    """
    base = _ensure_min_alignment(t, min_align=min_align)
    if not base.is_contiguous():
        raise ValueError("Expected a contiguous tensor before adding a guard row.")
    if base.ndim < 3:
        raise ValueError("Expected a tensor with an explicit time dimension.")

    row_elems = int(base.stride()[2])
    align_elems = max(1, int(min_align) // base.element_size())
    guard_elems = ((row_elems + align_elems - 1) // align_elems) * align_elems

    guarded_storage = torch.empty(
        base.numel() + guard_elems,
        device=base.device,
        dtype=base.dtype,
    )
    guarded_storage[:guard_elems].zero_()
    guarded_storage[guard_elems:].copy_(base.reshape(-1))
    guarded = torch.as_strided(
        guarded_storage,
        size=tuple(int(dim) for dim in base.shape),
        stride=tuple(int(step) for step in base.stride()),
        storage_offset=guard_elems,
    )
    if _assumed_align(guarded) < int(min_align):
        raise RuntimeError(
            f"Failed to materialize a guarded tensor with assumed_align >= {min_align}."
        )
    return guarded


def _make_fake_tensor_arg(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
    align: int | None = None,
):
    return cute.runtime.make_fake_tensor(
        _torch_to_cutlass_dtype(tensor.dtype),
        tuple(int(dim) for dim in (shape if shape is not None else tensor.shape)),
        stride=tuple(
            int(step) for step in (stride if stride is not None else tensor.stride())
        ),
        assumed_align=int(align if align is not None else _assumed_align(tensor)),
    )


def _compile_env_stream_placeholder():
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


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
    "_compile_env_stream_placeholder",
    "_assumed_align",
    "_choose_copy_bits_for_linear_tiles",
    "_elem_bits",
    "_ensure_min_alignment",
    "_guard_prev_time_base",
    "_make_fake_tensor_arg",
    "_pad_m_identity",
    "_pad_zero_time",
    "_tc_input_dtype",
    "_torch_to_cutlass_dtype",
]
