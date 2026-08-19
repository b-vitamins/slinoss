"""SwiGLU activation. CuTe DSL both directions.

    y = silu(gate) * up,    silu(g) = g / (1 + exp(-g))

Elementwise, so the kernel has no notion of the trailing axis and both operands
are flattened on the host. ``silu`` is evaluated in float32 at every operand
width and the product is narrowed once, on the store.

The pullback is the same map's derivative, elementwise again:

    dgate = dout * up * silu'(gate)
    dup   = dout * silu(gate)

Both need the logistic of the same gate, so it is evaluated once per element and
handed to :func:`slinoss._cute.silu` and :func:`slinoss._cute.silu_grad`.

Parallel decomposition, both directions. A grid-stride loop over vectors of ``V``
consecutive elements, ``V = 8`` at two bytes and ``V = 4`` at four, so one thread
step spans 16 bytes of each operand. The grid is twice the SM count, which is the
block-count floor, and the stride loop covers any element count from that one
launch shape. The trailing ``numel % V`` elements do not fill a vector and are
taken by the first warp of block 0, so the vector path carries no predicate.

The exponential argument is unbounded. A strongly negative gate drives
``exp(-g)`` past float32 range, and ``g / inf`` is a signed zero, which is the
correct limit; a strongly positive gate drives ``exp(-g)`` to zero and leaves
``g``. Neither end needs a clamp, so the whole path is branchless. The derivative
inherits that: at either end the logistic is an exact 0 or 1 and
``sig * (1 + g * (1 - sig))`` is finite.

Nothing is saved from the forward. Both gradients are recomputed from ``gate``
and ``up``, which the forward's caller already holds.

DRAM-bound, both directions. Analytic traffic at operand itemsize ``i``, with no
measured bandwidth claimed here:

- ``swiglu_fwd``: ``3 * numel * i`` -- one read of ``gate``, one of ``up``, one
  write.
- ``swiglu_bwd``: ``5 * numel * i`` -- the cotangent, ``gate`` and ``up``, then
  ``dgate`` and ``dup``.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    cute_dtype,
    dev_tensor,
    jit_launch,
    narrow,
    sigmoid,
    silu,
    silu_grad,
    widen,
)
from slinoss.ops.block.cute.norm import check_operand, sm_count
from slinoss.ops.block.reference import SwiGLUGrads

__all__ = [
    "ACT_THREADS",
    "VECTOR_BYTES",
    "swiglu_backward",
    "swiglu_bwd",
    "swiglu_bwd_kernel",
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


def _store_swiglu_grads(
    gdout: cute.Tensor,
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gdgate: cute.Tensor,
    gdup: cute.Tensor,
    index: cutlass.Int32,
    dtype: cutlass.Constexpr,
) -> None:
    """Write one element of each gradient. The only copy of the body.

    Args:
        gdout: ``(numel,)`` cotangent of the output, ``dtype``.
        ggate: ``(numel,)`` gate, ``dtype``.
        gup: ``(numel,)`` up, ``dtype``.
        gdgate: ``(numel,)`` written, ``dtype``.
        gdup: ``(numel,)`` written, ``dtype``.
        index: Flat element index. In range at every call site.
        dtype: The one element type of all five tensors. Compile-time and passed
            rather than read off the tensor because it keys the executor cache in
            :func:`slinoss._cute.jit_launch`.
    """
    gate = widen(ggate[index], dtype)
    sig = sigmoid(gate)
    cot = widen(gdout[index], dtype)
    gdgate[index] = narrow(cot * widen(gup[index], dtype) * silu_grad(gate, sig), dtype)
    gdup[index] = narrow(cot * silu(gate, sig), dtype)


# ---------------------------------------------------------------------------
# Kernels
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


@cute.kernel
def swiglu_bwd_kernel(
    gdout: cute.Tensor,
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gdgate: cute.Tensor,
    gdup: cute.Tensor,
    groups: cutlass.Int32,
    tail: cutlass.Int32,
    span: cutlass.Int32,
    dtype: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Pull the cotangent back over whole vectors, then over the trailing elements.

    Args:
        gdout: ``(numel,)`` cotangent of the output, ``dtype``.
        ggate: ``(numel,)`` the forward's gate, ``dtype``.
        gup: ``(numel,)`` the forward's up, ``dtype``.
        gdgate: ``(numel,)`` written, ``dtype``.
        gdup: ``(numel,)`` written, ``dtype``.
        groups: ``numel // vec``, whole vectors. Dynamic.
        tail: ``numel % vec``, below ``vec``. Dynamic.
        span: Vectors one grid step advances, ``blocks * threads``. Dynamic.
        dtype: The one element type of all five tensors. Compile-time.
        vec: Elements per thread step. Compile-time.
        threads: Block width. Compile-time.

    Invariants:
        ``groups * vec + tail == numel``, so every element of both gradients is
        written exactly once. The index space is the forward's, so a shape the
        forward covers is covered here.
    """
    bid, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()

    for group in cutlass.range(bid * threads + tid, groups, span):
        base = group * vec
        for j in cutlass.range_constexpr(vec):
            _store_swiglu_grads(gdout, ggate, gup, gdgate, gdup, base + j, dtype)

    # `&`, not `and`: both operands are device values.
    if (bid == 0) & (tid < tail):
        _store_swiglu_grads(gdout, ggate, gup, gdgate, gdup, groups * vec + tid, dtype)


@cute.jit
def swiglu_bwd(
    gdout: cute.Tensor,
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gdgate: cute.Tensor,
    gdup: cute.Tensor,
    groups: cutlass.Int32,
    tail: cutlass.Int32,
    span: cutlass.Int32,
    blocks: cutlass.Int32,
    dtype: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`swiglu_bwd_kernel`."""
    swiglu_bwd_kernel(
        gdout, ggate, gup, gdgate, gdup, groups, tail, span, dtype, vec, threads
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1))


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def _vector_width(dtype: torch.dtype) -> int:
    """Elements of ``dtype`` in :data:`VECTOR_BYTES`."""
    return VECTOR_BYTES // dtype.itemsize


def _check_operands(gate: Tensor, up: Tensor) -> int:
    """Validate the two activation operands, shared by both directions.

    Args:
        gate: Gate operand.
        up: Up operand.

    Returns:
        The element count.

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
    return count


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
            non-contiguous operand.
        TypeError: On a dtype with no kernel path, or on two operand dtypes.
    """
    count = _check_operands(gate, up)
    out = torch.empty_like(up)
    vec = _vector_width(gate.dtype)
    groups = count // vec
    blocks = 2 * sm_count(gate.device.index)
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


def swiglu_backward(dout: Tensor, gate: Tensor, up: Tensor, /) -> SwiGLUGrads:
    """Pullback of :func:`swiglu_forward`, in one launch.

    Both gradients are written by the same launch: they share the logistic of the
    gate, and splitting them would read ``gate`` twice.

    Args:
        dout: Cotangent of the output, shape and dtype of ``gate``, contiguous
            CUDA.
        gate: The forward's gate operand, ``(..., D)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`.
        up: The forward's up operand, same shape and dtype.

    Returns:
        A :class:`slinoss.ops.block.SwiGLUGrads`, both fields shaped and typed like
        the operands.

    Raises:
        ValueError: On a shape mismatch, an empty operand, or a non-CUDA or
            non-contiguous operand.
        TypeError: On a dtype with no kernel path, or on more than one operand
            dtype.
    """
    count = _check_operands(gate, up)
    if tuple(dout.shape) != tuple(gate.shape):
        raise ValueError(f"dout must be {tuple(gate.shape)}, got {tuple(dout.shape)}")
    if dout.dtype is not gate.dtype:
        raise TypeError(
            f"dout is {dout.dtype} and gate is {gate.dtype}; one dtype per call"
        )
    check_operand(dout, "dout")

    dgate = torch.empty_like(gate)
    dup = torch.empty_like(up)
    vec = _vector_width(gate.dtype)
    groups = count // vec
    blocks = 2 * sm_count(gate.device.index)
    jit_launch(
        swiglu_bwd,
        (
            dev_tensor(dout.view(count)),
            dev_tensor(gate.view(count)),
            dev_tensor(up.view(count)),
            dev_tensor(dgate.view(count)),
            dev_tensor(dup.view(count)),
            groups,
            count - groups * vec,
            blocks * ACT_THREADS,
            blocks,
        ),
        (cute_dtype(gate.dtype), vec, ACT_THREADS),
    )
    return SwiGLUGrads(dgate=dgate, dup=dup)
