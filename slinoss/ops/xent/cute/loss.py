"""Fused cross entropy. CuTe DSL forward and backward.

    lse  = logsumexp(logits[:, :classes])
    loss = mean(lse - logits[row, labels[row]])

The unfused expression is a pass over the logits per softmax stage plus a float32
copy of them, because ``aten``'s reduction accumulates at the operand's own width
below float32 and the copy is how a caller widens it. That copy is the largest
tensor in a language-model step. This kernel reads the logits once per direction and
accumulates in float32 at every operand width, so the copy has nowhere to be.

Parallel decomposition. The reduction runs over the class axis and never crosses a
row, so one row is one independent problem of length ``classes``. One block owns one
row: the grid is ``(rows, 1, 1)`` and rows are the only axis, which at any trained
shape is thousands of blocks. Thread ``t`` walks the columns from ``t`` at a stride
of the block width, so a warp's 32 loads are 32 consecutive elements, one coalesced
request whatever the class count is. The stride form rather than a contiguous
segment per thread: a segment would need the width divided among the threads, and a
vocabulary is not a multiple of anything.

Scalar loads, not vectorized. A vocabulary is odd as often as not -- 50,257 is --
so consecutive rows of a contiguous logits tensor start at odd element offsets and
no row past the first is aligned to a vector width. The instruction cost of the
scalar form is bounded well under the traffic it issues: one pass over the class
axis is one load, one fused multiply-add and one ``ex2.approx`` per element, and at
the standard shape the exponential's throughput accounts for a fifth of the pass's
DRAM time, which it overlaps.

Online normalizer, one pass. The forward tracks a running maximum and a running sum
rescaled to it, so it reads each logit once instead of once for the maximum and
again for the sum. The rescale is branchless: both the accumulated sum and the new
term are scaled to ``max(peak, x)``, which costs a second exponential per element
and no divergence. A thread whose stride reaches no column keeps
:data:`NO_COLUMN` as its peak and zero as its sum, and the block combine scales
that sum by an exponential that underflows, so it contributes exactly zero rather
than a special case.

Two block reductions, a maximum then a sum, over separate scratch words. Sharing one
tile would need a barrier between the second store and the first read, and the
sixty-four bytes the second tile costs are cheaper than the barrier.

The mean is closed by :func:`slinoss._reduce.reduce_partials` over the per-row
losses, one float32 per row. Summing inside the launch would need either an atomic,
which is not reproducible, or a second grid-wide barrier, which the launch does not
have.

``lse`` crosses to the backward at 4 B per row. The alternative is a second pass
over the logits to recompute it, which is another read of the largest tensor in the
step; see :mod:`slinoss.ops.xent.reference`.

Padding. ``classes`` and the operand width are separate arguments and the width is
the larger. A head padded to a tensor-core multiple emits columns no label indexes,
and the forward stops at ``classes`` while the backward writes zero over the rest,
so the gradient covers every column of its own allocation without a fill.

DRAM-bound, both directions. Per row the forward reads ``classes`` operand elements
and writes two float32; the backward reads ``classes`` operand elements and the two
saved words and writes ``width`` operand elements. At the standard shape, 8,192 rows
of 50,257 bfloat16 classes: 823.4 MB in and 65.5 KB out for the forward, 823.4 MB
plus 65.5 KB in and 823.4 MB out for the backward, 2.47 GB over the pair. The label
axis is 32 KB and rounds away.

Measured on sm_86, clocks not lockable, at that shape, two passes with the arm order
reversed: the forward 1.177 ms both times, 699.6 GB/s over its 823.4 MB, and the
backward 2.643 and 2.645 ms, 622.9 GB/s over its 1,646.8 MB. A device copy of that
size sustains 683 GB/s here, so the forward is at 102% of the copy law and the
backward at 91%. The row reduction closing the mean adds 0.026 ms. The aten
expression this replaces ran 27.715 ms over seven kernels at the same shape, 6.0 ms
of it the float32 copy of the logits and its narrowing back, and its peak allocation
was 6,282.9 MiB against 2,356.7 MiB here.
"""

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor

from slinoss._cute import (
    LOG2_E,
    Scalar,
    Stream,
    Tile,
    assert_smem_fits,
    block_reduce_add,
    f32,
    jit_launch,
    narrow,
    select,
    shuffle_xor,
    smem_bytes,
    widen,
)
from slinoss._guard import check_dtypes, check_layout
from slinoss._precision import KERNEL_DTYPES
from slinoss._reduce import reduce_partials
from slinoss.ops.xent.reference import (
    LABEL_DTYPES,
    XentGrads,
    XentState,
    xent_shape,
)

__all__ = [
    "NO_COLUMN",
    "XENT_THREADS",
    "warp_smem_bytes",
    "warp_tile",
    "xent_backward",
    "xent_bwd",
    "xent_bwd_kernel",
    "xent_forward",
    "xent_fwd",
    "xent_fwd_kernel",
]

XENT_THREADS = 256
"""Block width. One block owns one row.

Eight warps, so the block reduction is one butterfly and eight words. Wider would
shorten the per-thread walk over the class axis and lengthen the combine; the walk
is hundreds of elements at any trained vocabulary, so the combine is not what bounds
it.
"""

NO_COLUMN: float = -1.0e38
"""Peak of a thread whose stride reaches no column.

Finite, so the difference against a real peak does not produce a NaN, and low
enough that its exponential underflows to zero at every representable peak. The
sum it scales is zero either way; the value is what keeps the scale factor from
being an infinity.
"""


def warp_tile(threads: int) -> Tile:
    """One float32 per warp, for one block reduction.

    Args:
        threads: Block width. Compile-time.

    Returns:
        The tile.
    """
    return Tile((threads // cute.arch.WARP_SIZE,), (1,))


def warp_smem_bytes(threads: int) -> int:
    """Shared memory :func:`xent_fwd_kernel` holds, in bytes.

    Two tiles: the maximum's words and the sum's.

    Args:
        threads: Block width.

    Returns:
        Total bytes.
    """
    return smem_bytes([(warp_tile(threads), 4), (warp_tile(threads), 4)])


def _block_max(
    value: Scalar, smax: cute.Tensor, tid: cutlass.Int32, threads: int
) -> Scalar:
    """Maximum of one float32 over a whole block, result in every thread.

    The maximum counterpart of :func:`slinoss._cute.block_reduce_add`, same shape
    of reduction and same unpredicated warp store: a butterfly leaves the warp's
    maximum in all 32 lanes, so every lane writes one identical word and which one
    wins cannot change the result. Module-private because this is the only kernel
    in the tree that reduces a maximum.

    Entered by every thread of the block. One barrier.

    Args:
        value: The thread's contribution.
        smax: ``(threads // 32,)`` float32 scratch.
        tid: Thread index within the block.
        threads: Block width, a multiple of the warp width. Compile-time.

    Returns:
        The block maximum, bitwise identical in every thread.
    """
    reach = 1
    while reach < cute.arch.WARP_SIZE:
        value = cutlass.max(value, shuffle_xor(value, reach))
        reach *= 2
    smax[tid // cute.arch.WARP_SIZE] = value
    cute.arch.sync_threads()
    # Plain Python loop: an undecorated helper the DSL's preprocessor never
    # rewrites, and the bound is compile-time, which unrolls it during the trace.
    top = cutlass.Float32(NO_COLUMN)
    for warp in range(threads // cute.arch.WARP_SIZE):
        top = cutlass.max(top, smax[warp])
    return top


@cute.kernel
def xent_fwd_kernel(
    glogits: cute.Tensor,
    glabels: cute.Tensor,
    gloss: cute.Tensor,
    glse: cute.Tensor,
    classes: cutlass.Int32,
    inv_rows: Scalar,
    threads: cutlass.Constexpr,
) -> None:
    """One row's normalizer and its scaled loss term.

    Args:
        glogits: ``(rows, width)``, one of
            :data:`slinoss._precision.KERNEL_DTYPES`. The element type is read off
            the operand.
        glabels: ``(rows,)`` integer.
        gloss: ``(rows,)`` float32, the loss term already divided by the row count.
        glse: ``(rows,)`` float32, the log partition function over ``classes``.
        classes: Classes the labels index. Dynamic, so one variant covers every
            vocabulary.
        inv_rows: Reciprocal of the row count. Dynamic.
        threads: Block width. Compile-time.

    Invariants:
        The reduction order is fixed by the launch geometry alone: ascending column
        within a thread, then the butterfly, then ascending warp. No atomics, so one
        shape reproduces bit for bit. Columns at or past ``classes`` are never read.
        A label outside ``[0, classes)`` is clamped into it, so the kernel cannot
        fault on one; the loss it then returns is the clamped class's.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    smax = smem.allocate_tensor(cutlass.Float32, warp_tile(threads).layout(), 16)
    ssum = smem.allocate_tensor(cutlass.Float32, warp_tile(threads).layout(), 16)
    src = glogits.element_type

    # Both loads are uniform across the block and both are ahead of the one
    # divergent branch in the kernel: a load inside a divergent branch issues as
    # many requests as there are active lanes.
    label = cutlass.max(cutlass.min(cutlass.Int32(glabels[row]), classes - 1), 0)
    target = widen(glogits[row, label], src)

    peak = cutlass.Float32(NO_COLUMN)
    total = cutlass.Float32(0.0)
    for col in cutlass.range(tid, classes, threads):
        value = widen(glogits[row, col], src)
        # Branchless rescale. `peak` starts below every representable logit, so the
        # first column scales a zero sum by an underflowing factor and adds one.
        ahead = cutlass.max(peak, value)
        total = total * f32(cute.exp2((peak - ahead) * LOG2_E)) + f32(
            cute.exp2((value - ahead) * LOG2_E)
        )
        peak = ahead

    top = _block_max(peak, smax, tid, threads)
    total = block_reduce_add(
        total * f32(cute.exp2((peak - top) * LOG2_E)), ssum, tid, threads
    )
    lse = top + f32(cute.log(total))
    if tid == 0:
        glse[row] = lse
        gloss[row] = (lse - target) * inv_rows


@cute.jit
def xent_fwd(
    glogits: cute.Tensor,
    glabels: cute.Tensor,
    gloss: cute.Tensor,
    glse: cute.Tensor,
    rows: cutlass.Int32,
    classes: cutlass.Int32,
    inv_rows: Scalar,
    stream: Stream,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`xent_fwd_kernel`, one block per row."""
    xent_fwd_kernel(glogits, glabels, gloss, glse, classes, inv_rows, threads).launch(
        grid=(rows, 1, 1), block=(threads, 1, 1), stream=stream
    )


@cute.kernel
def xent_bwd_kernel(
    glogits: cute.Tensor,
    glabels: cute.Tensor,
    glse: cute.Tensor,
    gdloss: cute.Tensor,
    gdlogits: cute.Tensor,
    classes: cutlass.Int32,
    width: cutlass.Int32,
    inv_rows: Scalar,
    threads: cutlass.Constexpr,
) -> None:
    """One row's logit gradient, over the operand's whole width.

    Args:
        glogits: ``(rows, width)``, as the forward read them.
        glabels: ``(rows,)`` integer.
        glse: ``(rows,)`` float32, the forward's normalizer.
        gdloss: ``(1,)`` float32, the cotangent of the mean.
        gdlogits: ``(rows, width)``, the operand's element type.
        classes: Classes the labels index. Dynamic.
        width: Operand width, at least ``classes``. Dynamic.
        inv_rows: Reciprocal of the row count. Dynamic.
        threads: Block width. Compile-time.

    Invariants:
        No reduction and no shared memory: the row's normalizer is an input, so
        every element is one load and one store. Columns at or past ``classes``
        are written zero rather than left, because the destination is
        uninitialized memory of the operand's full width. A label outside
        ``[0, classes)`` is clamped, as in the forward.
    """
    row, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    src = glogits.element_type
    dst = gdlogits.element_type

    label = cutlass.max(cutlass.min(cutlass.Int32(glabels[row]), classes - 1), 0)
    lse = glse[row]
    scale = gdloss[0] * inv_rows

    for col in cutlass.range(tid, classes, threads):
        prob = f32(cute.exp2((widen(glogits[row, col], src) - lse) * LOG2_E))
        onehot = select(col == label, prob - 1.0, prob)
        gdlogits[row, col] = narrow(onehot * scale, dst)
    for col in cutlass.range(classes + tid, width, threads):
        gdlogits[row, col] = narrow(cutlass.Float32(0.0), dst)


@cute.jit
def xent_bwd(
    glogits: cute.Tensor,
    glabels: cute.Tensor,
    glse: cute.Tensor,
    gdloss: cute.Tensor,
    gdlogits: cute.Tensor,
    rows: cutlass.Int32,
    classes: cutlass.Int32,
    width: cutlass.Int32,
    inv_rows: Scalar,
    stream: Stream,
    threads: cutlass.Constexpr,
) -> None:
    """Launch :func:`xent_bwd_kernel`, one block per row."""
    xent_bwd_kernel(
        glogits, glabels, glse, gdloss, gdlogits, classes, width, inv_rows, threads
    ).launch(grid=(rows, 1, 1), block=(threads, 1, 1), stream=stream)


def _check_operands(logits: Tensor, labels: Tensor, classes: int) -> tuple[int, int]:
    """Validate what both directions share.

    Args:
        logits: ``(rows, width)``, contiguous CUDA.
        labels: ``(rows,)`` integer, contiguous CUDA.
        classes: Classes the labels index.

    Returns:
        ``(rows, width)``.

    Raises:
        ValueError: On a shape, device or layout violation.
        TypeError: On an operand dtype with no kernel path, or a label dtype that
            is not an integer.
    """
    rows, width = xent_shape(logits, labels, classes)
    check_dtypes(((logits, "logits"),), KERNEL_DTYPES, "kernel dtypes")
    check_dtypes(((labels, "labels"),), LABEL_DTYPES, "integer labels", "group")
    check_layout(((logits, "logits"), (labels, "labels")))
    assert_smem_fits("xent_fwd_kernel", warp_smem_bytes(XENT_THREADS))
    return rows, width


def xent_forward(logits: Tensor, labels: Tensor, /, *, classes: int) -> XentState:
    """Mean cross entropy in two launches: the rows, then their mean.

    Args:
        logits: ``(rows, width)``, contiguous CUDA, one of
            :data:`slinoss._precision.KERNEL_DTYPES`. Contiguity rather than a
            pitched band: a vocabulary is odd as often as not, so a row pitch
            carries no alignment a band could rely on.
        labels: ``(rows,)`` contiguous CUDA, int32 or int64, every entry in
            ``[0, classes)``.
        classes: Classes the labels index, at most the operand width. Never
            ``logits.shape[-1]``: a padded head emits columns no label indexes.

    Returns:
        The loss, 0-d float32, and the per-row normalizer, ``(rows,)`` float32.

    Raises:
        ValueError: On a shape, device or layout violation.
        TypeError: On an unsupported operand or label dtype.
    """
    rows, _ = _check_operands(logits, labels, classes)
    per_row = torch.empty(rows, dtype=torch.float32, device=logits.device)
    lse = torch.empty(rows, dtype=torch.float32, device=logits.device)
    jit_launch(
        xent_fwd,
        (logits, labels, per_row, lse, rows, classes, 1.0 / rows),
        (XENT_THREADS,),
    )
    # The row terms are already divided by the row count, so the sum is the mean.
    # Reduced by the shared row reduction rather than by torch: an aten reduction
    # here would be a kernel in the step's glue class for 32 KB of traffic.
    return XentState(loss=reduce_partials(per_row.view(1, rows, 1)).view(()), lse=lse)


def xent_backward(
    dloss: Tensor,
    logits: Tensor,
    labels: Tensor,
    lse: Tensor,
    /,
    *,
    classes: int,
) -> XentGrads:
    """Logit gradient in one launch.

    Args:
        dloss: 0-d float32 CUDA, the cotangent of the mean. Read on the device, so
            no host synchronization is on the path.
        logits: ``(rows, width)``, as the forward read them.
        labels: ``(rows,)``, as the forward read them.
        lse: ``(rows,)`` float32 contiguous CUDA, the forward's normalizer.
        classes: Classes the labels index.

    Returns:
        ``dlogits``, the operand's shape, dtype and layout, zero at every column at
        or past ``classes``.

    Raises:
        ValueError: On a shape, device or layout violation.
        TypeError: On an unsupported operand, label, cotangent or normalizer dtype.
    """
    rows, width = _check_operands(logits, labels, classes)
    if tuple(lse.shape) != (rows,):
        raise ValueError(f"lse must be {(rows,)}, got {tuple(lse.shape)}")
    if dloss.numel() != 1:
        raise ValueError(f"dloss must hold one element, got {tuple(dloss.shape)}")
    flat = dloss.reshape(1)
    check_dtypes(((flat, "dloss"), (lse, "lse")), (torch.float32,), "float32", "group")
    check_layout(((flat, "dloss"), (lse, "lse")))
    dlogits = torch.empty_like(logits)
    jit_launch(
        xent_bwd,
        (logits, labels, lse, flat, dlogits, rows, classes, width, 1.0 / rows),
        (XENT_THREADS,),
    )
    return XentGrads(dlogits=dlogits)
