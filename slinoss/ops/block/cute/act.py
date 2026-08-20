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
step spans 16 bytes of each operand. The grid is
:func:`slinoss.ops.block.cute.norm.fill_blocks`, and the stride loop covers any
element count from that one launch shape. The trailing ``numel % V`` elements do
not fill a vector and are taken by the first warp of block 0, so the vector path
carries no predicate.

Each vector crosses the boundary as one access, through a register fragment and
:func:`cutlass.cute.autovec_copy`. ``V`` separate subscripts of the global tensor
are ``V`` separate requests at a 16-byte lane stride, which is eight times the
necessary L1TEX load traffic and, on the store side, eight partial writes per
32-byte sector: measured on sm_86 at 8x the sector count on both sides, and the
partial stores turned into 1.9x the necessary DRAM write traffic once the grid was
large enough to keep more of them live in L2 at once.

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
    Scalar,
    Stream,
    cute_dtype,
    jit_launch,
    narrow,
    sigmoid,
    silu,
    silu_grad,
    widen,
)
from slinoss.ops.block.cute.norm import check_operand, fill_blocks
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


def _swiglu(gate: Scalar, up: Scalar) -> Scalar:
    """``silu(gate) * up`` in float32. The only copy of the forward body.

    Args:
        gate: Gate element, widened.
        up: Up element, widened.

    Returns:
        The activation.
    """
    return silu(gate, sigmoid(gate)) * up


def _swiglu_grads(gate: Scalar, up: Scalar, cot: Scalar) -> tuple[Scalar, Scalar]:
    """``(dgate, dup)`` in float32. The only copy of the backward body.

    Args:
        gate: Gate element, widened.
        up: Up element, widened.
        cot: Cotangent of the output, widened.

    Returns:
        The gradient of the gate and the gradient of the up operand.
    """
    sig = sigmoid(gate)
    return cot * up * silu_grad(gate, sig), cot * silu(gate, sig)


def _vector(tensor: cute.Tensor, vec: cutlass.Constexpr) -> cute.Tensor:
    """Retile a flat tensor as ``(vec, groups)``, one whole vector per column.

    Args:
        tensor: ``(numel,)``, contiguous.
        vec: Elements per vector. Compile-time.

    Returns:
        The retiled view. Column ``g`` is statically shaped ``(vec,)``, which is
        what lets :func:`cutlass.cute.autovec_copy` pick the widest access. A
        trailing partial column exists when ``vec`` does not divide ``numel`` and
        is never addressed: the caller's loop bound is the whole-vector count.
    """
    return cute.zipped_divide(tensor, (vec,))


def _fragment(tensor: cute.Tensor, vec: cutlass.Constexpr) -> cute.Tensor:
    """A register fragment holding one vector of ``tensor``'s element type.

    Args:
        tensor: The tensor the fragment is copied to or from.
        vec: Elements per vector. Compile-time.

    Returns:
        The fragment.
    """
    return cute.make_fragment((vec,), tensor.element_type)


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

    vgate = _vector(ggate, vec)
    vup = _vector(gup, vec)
    vy = _vector(gy, vec)
    fgate = _fragment(ggate, vec)
    fup = _fragment(gup, vec)
    fy = _fragment(gy, vec)

    for group in cutlass.range(bid * threads + tid, groups, span):
        cute.autovec_copy(vgate[(None, group)], fgate)
        cute.autovec_copy(vup[(None, group)], fup)
        for j in cutlass.range_constexpr(vec):
            fy[j] = narrow(
                _swiglu(
                    widen(fgate[j], ggate.element_type),
                    widen(fup[j], gup.element_type),
                ),
                gy.element_type,
            )
        cute.autovec_copy(fy, vy[(None, group)])

    # `&`, not `and`: both operands are device values.
    if (bid == 0) & (tid < tail):
        index = groups * vec + tid
        gy[index] = narrow(
            _swiglu(
                widen(ggate[index], ggate.element_type),
                widen(gup[index], gup.element_type),
            ),
            gy.element_type,
        )


@cute.jit
def swiglu_fwd(
    ggate: cute.Tensor,
    gup: cute.Tensor,
    gy: cute.Tensor,
    groups: cutlass.Int32,
    tail: cutlass.Int32,
    span: cutlass.Int32,
    blocks: cutlass.Int32,
    stream: Stream,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`swiglu_fwd_kernel`.

    Only ``vec`` and ``threads`` are compile-time, so one compiled variant per
    operand dtype covers every element count.
    """
    swiglu_fwd_kernel(ggate, gup, gy, groups, tail, span, vec, threads).launch(
        grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream
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

    vdout = _vector(gdout, vec)
    vgate = _vector(ggate, vec)
    vup = _vector(gup, vec)
    vdgate = _vector(gdgate, vec)
    vdup = _vector(gdup, vec)
    fdout = _fragment(gdout, vec)
    fgate = _fragment(ggate, vec)
    fup = _fragment(gup, vec)
    fdgate = _fragment(gdgate, vec)
    fdup = _fragment(gdup, vec)

    for group in cutlass.range(bid * threads + tid, groups, span):
        cute.autovec_copy(vdout[(None, group)], fdout)
        cute.autovec_copy(vgate[(None, group)], fgate)
        cute.autovec_copy(vup[(None, group)], fup)
        for j in cutlass.range_constexpr(vec):
            dgate, dup = _swiglu_grads(
                widen(fgate[j], dtype), widen(fup[j], dtype), widen(fdout[j], dtype)
            )
            fdgate[j] = narrow(dgate, dtype)
            fdup[j] = narrow(dup, dtype)
        cute.autovec_copy(fdgate, vdgate[(None, group)])
        cute.autovec_copy(fdup, vdup[(None, group)])

    # `&`, not `and`: both operands are device values.
    if (bid == 0) & (tid < tail):
        index = groups * vec + tid
        dgate, dup = _swiglu_grads(
            widen(ggate[index], dtype),
            widen(gup[index], dtype),
            widen(gdout[index], dtype),
        )
        gdgate[index] = narrow(dgate, dtype)
        gdup[index] = narrow(dup, dtype)


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
    stream: Stream,
    dtype: cutlass.Constexpr,
    vec: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`swiglu_bwd_kernel`."""
    swiglu_bwd_kernel(
        gdout, ggate, gup, gdgate, gdup, groups, tail, span, dtype, vec, threads
    ).launch(grid=(blocks, 1, 1), block=(threads, 1, 1), stream=stream)


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
    blocks = fill_blocks(ACT_THREADS, gate.device.index)
    jit_launch(
        swiglu_fwd,
        (
            gate.view(count),
            up.view(count),
            out.view(count),
            groups,
            count - groups * vec,
            blocks * ACT_THREADS,
            blocks,
        ),
        (vec, ACT_THREADS),
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
    blocks = fill_blocks(ACT_THREADS, gate.device.index)
    jit_launch(
        swiglu_bwd,
        (
            dout.view(count),
            gate.view(count),
            up.view(count),
            dgate.view(count),
            dup.view(count),
            groups,
            count - groups * vec,
            blocks * ACT_THREADS,
            blocks,
        ),
        (cute_dtype(gate.dtype), vec, ACT_THREADS),
    )
    return SwiGLUGrads(dgate=dgate, dup=dup)
