"""SwiGLU activation. CuTe DSL forward.

    y = silu(gate) * up,    silu(g) = g / (1 + exp(-g))

Elementwise, so the kernel has no notion of the trailing axis and both operands
are flattened on the host. ``silu`` is evaluated in float32 at every operand
width and the product is narrowed once, on the store.

Parallel decomposition. A grid-stride loop over vectors of ``V`` consecutive
elements, ``V = 8`` at two bytes and ``V = 4`` at four, so one thread step spans
16 bytes of each operand. The grid is twice the SM count, which is the
block-count floor, and the stride loop covers any element count from that one
launch shape. The trailing ``numel % V`` elements do not fill a vector and are
taken by the first warp of block 0, so the vector path carries no predicate.

The exponential argument is unbounded. A strongly negative gate drives
``exp(-g)`` past float32 range, and ``g / inf`` is a signed zero, which is the
correct limit; a strongly positive gate drives ``exp(-g)`` to zero and leaves
``g``. Neither end needs a clamp, so the whole path is branchless.

DRAM-bound. Analytic traffic is ``3 * numel * i`` bytes at operand itemsize
``i``: one read of ``gate``, one of ``up``, one write. No measured bandwidth is
claimed here.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import dev_tensor, narrow, sigmoid, silu, widen
from slinoss.ops.block.cute.norm import check_operand

__all__ = [
    "ACT_THREADS",
    "VECTOR_BYTES",
    "swiglu_forward",
    "swiglu_fwd",
    "swiglu_fwd_kernel",
]

ACT_THREADS = 256
"""Block width. Eight warps, and the grid-stride step is this times the grid."""

VECTOR_BYTES = 16
"""Bytes of one operand a thread covers per step: eight 16-bit or four 32-bit
elements."""


# ---------------------------------------------------------------------------
# Device math
# ---------------------------------------------------------------------------


# The logistic family is in slinoss._cute: the mixer tail rounds the same
# function, and two roundings of one activation is a divergence.


def _store_swiglu(
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gy: cute.Tensor,
    index: cutlass.Int32,
) -> None:
    """Write one element of ``silu(gate) * up``. The only copy of the body.

    Args:
        ggate: ``(numel,)`` gate, activation dtype.
        gup: ``(numel,)`` up, same dtype.
        gy: ``(numel,)`` output, same dtype.
        index: Flat element index. In range at every call site.
    """
    gate = widen(ggate[index], ggate.element_type)
    gy[index] = narrow(
        silu(gate, sigmoid(gate)) * widen(gup[index], gup.element_type),
        gy.element_type,
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def swiglu_fwd_kernel(
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gy: cute.Tensor,
    groups: cutlass.Int32,
    tail: cutlass.Int32,
    span: cutlass.Int32,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Apply the activation over whole vectors, then over the trailing elements.

    Args:
        ggate: ``(numel,)`` gate, activation dtype.
        gup: ``(numel,)`` up, same dtype.
        gy: ``(numel,)`` output, same dtype.
        groups: ``numel // vec``, whole vectors. Dynamic.
        tail: ``numel % vec``, below ``vec``. Dynamic.
        span: Vectors one grid step advances, ``blocks * threads``. Dynamic.
        vec: Elements per thread step. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        ``groups * vec + tail == numel``, so every element is written exactly
        once. The vector loop is unpredicated; only the tail carries a predicate,
        and it is confined to one warp of one block.
    """
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    for group in cutlass.range(bid * threads + tid, groups, span):
        base = group * vec
        for j in cutlass.range_constexpr(vec):
            _store_swiglu(ggate, gup, gy, base + j)

    # `&`, not `and`: both operands are device values.
    if (bid == 0) & (tid < tail):
        _store_swiglu(ggate, gup, gy, groups * vec + tid)


@cute.jit
def swiglu_fwd(
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gy: cute.Tensor,
    groups: cutlass.Int32,
    tail: cutlass.Int32,
    span: cutlass.Int32,
    blocks: cutlass.Int32,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`swiglu_fwd_kernel`.

    Only ``vec`` and ``threads`` are compile-time, so one compiled variant per
    operand dtype covers every element count.
    """
    swiglu_fwd_kernel(ggate, gup, gy, groups, tail, span, vec, threads).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1)
    )


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


@cache
def _sm_count(index: int) -> int:
    """Multiprocessors on one CUDA device. Cached: the grid is sized per launch."""
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def _vector_width(dtype: torch.dtype) -> int:
    """Elements of ``dtype`` in :data:`VECTOR_BYTES`."""
    return VECTOR_BYTES // dtype.itemsize


def swiglu_forward(gate: Tensor, up: Tensor) -> Tensor:
    """``silu(gate) * up``, in one launch.

    Args:
        gate: Shape ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        up: Same shape and dtype.

    Returns:
        Same shape, dtype and layout as ``up``.

    Raises:
        ValueError: On a shape mismatch, an empty operand, or a non-CUDA or
            non-contiguous operand. An empty operand has no element to write, so
            it is refused rather than launched over.
        TypeError: On a dtype with no kernel path, or on two operand dtypes.
    """
    if tuple(up.shape) != tuple(gate.shape):
        raise ValueError(f"up must be {tuple(gate.shape)}, got {tuple(up.shape)}")
    if up.dtype is not gate.dtype:
        raise TypeError(
            f"up is {up.dtype} and gate is {gate.dtype}; one dtype per call"
        )
    check_operand(gate, "gate")
    check_operand(up, "up")
    count = gate.numel()
    if count == 0:
        raise ValueError(
            f"gate must hold at least one element, got {tuple(gate.shape)}"
        )

    out = torch.empty_like(up)
    vec = _vector_width(gate.dtype)
    groups = count // vec
    blocks = 2 * _sm_count(gate.device.index)
    swiglu_fwd(
        dev_tensor(gate.view(count)),
        dev_tensor(up.view(count)),
        dev_tensor(out.view(count)),
        groups,
        count - groups * vec,
        blocks * ACT_THREADS,
        blocks,
        vec,
        ACT_THREADS,
    )
    return out
