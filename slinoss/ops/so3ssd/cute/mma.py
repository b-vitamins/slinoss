"""Warp-level tensor-core GEMM. One tiled MMA for every contraction in the scan.

After the rowwise change of basis into the chunk-local frame, the chunked scan is
four dense real GEMMs over shared-memory operands:

    increment  D(P,3N)  = U(L,P)^T  @ Bn(L,3N)     M=P,  N=3N, K=L
    score      D(L,L)   = Cr(L,3N)  @ Bn(L,3N)^T   M=L,  N=L,  K=3N
    diagonal   D(L,P)  += S(L,L)    @ U(L,P)       M=L,  N=P,  K=L
    offset     D(L,P)   = Cr(L,3N)  @ Z(P,3N)^T    M=L,  N=P,  K=3N

All four are covered by one atom, ``(16,8,16)`` low-precision times low-precision
into float32, with four warps partitioning M and a tile of ``(64,16,16)``. A
transposed operand is a stride swap on the same shared-memory iterator, so no form
needs a staging copy or a repack.

Divisibility, measured on the device rather than assumed:

- N must be a multiple of :data:`MMA_TILE_N`. An N extent of 8 or 24 fails IR
  verification; 16, 48, 64, 96 and 128 compile and give exact results. ``3N`` is a
  multiple of 48, ``L`` a power of two at or above 16, and ``P`` a multiple of
  ``HEAD_MULTIPLE``, so every N extent in the operator clears this.
- K must be a multiple of :data:`MMA_TILE_K`. ``L`` and ``3N`` both do.
- M is free. It is rounded up to :data:`MMA_TILE_M` in shared memory by
  :func:`mma_rows`, the rounded rows are zero-filled, and the store is predicated.
  This is the one padded mode in the operator, and it is unavoidable: the M tile is
  64 because four warps partition M, and no four-warp atom layout makes a multiple
  of 16 and a multiple of 48 both divide. It costs wasted tensor-core work in
  kernels that are DRAM-bound by a factor of seven, and no extra traffic.

Shared-memory operands are not swizzled. A row pitch that is an odd multiple of 16
bytes puts the eight threads of an ``ldmatrix`` phase in eight distinct banks
already, which is what :func:`smem_pitch` computes; padding a 48-wide row to 64
instead is the worst available choice, because a 128-byte pitch collapses all eight
onto one segment. A composed swizzle is not an alternative here: in the pinned DSL
version ``partition_B`` aborts on any swizzled layout whose stride-1 mode is sliced,
which is every transposed operand above.
"""

import cutlass
import cutlass.cute as cute

from slinoss._cute import Tile
from slinoss.ops.so3ssd.cute.common import WARPS

__all__ = [
    "MMA_INST",
    "MMA_TILE_K",
    "MMA_TILE_M",
    "MMA_TILE_N",
    "SMEM_SEGMENT",
    "make_mma",
    "mma_acc",
    "mma_coords",
    "mma_gemm",
    "mma_rows",
    "mma_store",
    "operand_tile",
    "smem_pitch",
]

MMA_INST: tuple[int, int, int] = (16, 8, 16)
"""Atom shape ``(M,N,K)``. The sm_80 16-bit tensor-core instruction."""

MMA_TILE_M: int = WARPS * MMA_INST[0]
"""M mode of the tile. Four warps partition M, so this is four atoms wide."""

MMA_TILE_N: int = 16
"""N mode of the tile. Every N extent must be a multiple of this."""

MMA_TILE_K: int = MMA_INST[2]
"""K mode of the tile. Every K extent must be a multiple of this."""

SMEM_SEGMENT: int = 16
"""Shared-memory segment in bytes. Eight threads of an ``ldmatrix`` phase read one
segment each, so the row pitch decides whether they collide."""


def mma_rows(extent: int) -> int:
    """Round an M extent up to the tile's M mode.

    Args:
        extent: Logical rows of the output, ``P`` or ``L``.

    Returns:
        Rows the shared-memory operand must expose. Equal to ``extent`` when it
        already divides.
    """
    return -(-extent // MMA_TILE_M) * MMA_TILE_M


def smem_pitch(width: int, itemsize: int = 2) -> int:
    """Row pitch in elements for a bank-conflict-free shared-memory tile.

    The pitch is rounded up to a whole number of 16-byte segments and then forced
    odd. An even count puts two or eight threads of a phase in the same segment; an
    odd count spreads them over all eight.

    Args:
        width: Elements that must fit in a row.
        itemsize: Bytes per element. 2 for the GEMM operands, 4 for float32 tiles.

    Returns:
        Pitch in elements. An odd multiple of ``SMEM_SEGMENT // itemsize``.

    Raises:
        ValueError: If ``itemsize`` does not divide the segment.
    """
    if SMEM_SEGMENT % itemsize != 0:
        raise ValueError(f"itemsize {itemsize} does not divide {SMEM_SEGMENT} bytes")
    unit = SMEM_SEGMENT // itemsize
    segments = -(-width // unit)
    return unit * (segments | 1)


def operand_tile(rows: int, width: int) -> Tile:
    """Row-major tile for one GEMM operand, ``(rows, smem_pitch(width))``.

    Every operand tile in the tree comes from here, so the pitch rule lives in one
    place. Whether the tile's rows are the M mode or the N mode is decided by the
    view built over it, not by the tile.

    Args:
        rows: Rows to allocate. Already rounded by :func:`mma_rows` when the tile's
            rows are the M mode of the output.
        width: Elements per row that carry data. The rest of the pitch is padding
            outside every view.
    """
    pitch = smem_pitch(width)
    return Tile((rows, pitch), (pitch, 1))


@cute.jit
def make_mma(dtype: cutlass.Constexpr) -> cute.TiledMma:
    """Build the one tiled MMA.

    Constructed on the host side and passed into the kernel, so a kernel holds no
    knowledge of the atom.

    Args:
        dtype: Operand element type. ``cutlass.BFloat16`` or ``cutlass.Float16``.

    Returns:
        The tiled MMA with tile ``(MMA_TILE_M, MMA_TILE_N, MMA_TILE_K)``.
    """
    return cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, MMA_INST),
        (WARPS, 1, 1),
        permutation_mnk=(MMA_TILE_M, MMA_TILE_N, MMA_TILE_K),
    )


@cute.jit
def mma_acc(
    tiled_mma: cute.TiledMma, tid: cutlass.Int32, shape_mn: cutlass.Constexpr
) -> cute.Tensor:
    """Allocate this thread's float32 accumulator and zero it.

    The zero is a register fill inside the kernel, never a global buffer.

    Args:
        tiled_mma: From :func:`make_mma`.
        tid: Thread index within the block.
        shape_mn: ``(M,N)`` of the output tile. ``M`` must already be rounded by
            :func:`mma_rows`.

    Returns:
        Register-backed float32 fragment.
    """
    thr = tiled_mma.get_slice(tid)
    acc = cute.make_fragment(thr.partition_shape_C(shape_mn), cutlass.Float32)
    acc.fill(0.0)
    return acc


@cute.jit
def mma_gemm(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    acc: cute.Tensor,
    va: cute.Tensor,
    vb: cute.Tensor,
    a_k_major: cutlass.Constexpr,
    b_k_major: cutlass.Constexpr,
) -> None:
    """Accumulate ``va @ vb^T`` into ``acc``.

    Called more than once on the same accumulator to sum contributions, which is
    how the two forcing taps combine without concatenating along K.

    Args:
        tiled_mma: From :func:`make_mma`.
        tid: Thread index within the block.
        acc: From :func:`mma_acc`. Updated in place.
        va: Shared-memory view of shape ``(M,K)``.
        vb: Shared-memory view of shape ``(N,K)``.
        a_k_major: Whether ``va``'s K mode is the stride-1 mode.
        b_k_major: Whether ``vb``'s K mode is the stride-1 mode.

    Invariants:
        ``ldmatrix`` transposes exactly when the operand's stride-1 mode is M or N
        rather than K, so the flag is the negation of the major-ness. A wrong flag
        is a compile-time IR verification failure, not a wrong answer.
    """
    thr = tiled_mma.get_slice(tid)
    fa = tiled_mma.make_fragment_A(thr.partition_A(va))
    fb = tiled_mma.make_fragment_B(thr.partition_B(vb))
    atom_a = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not a_k_major, 4), va.element_type
    )
    atom_b = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not b_k_major, 4), vb.element_type
    )
    copy_a = cute.make_tiled_copy_A(atom_a, tiled_mma)
    copy_b = cute.make_tiled_copy_B(atom_b, tiled_mma)
    thr_a = copy_a.get_slice(tid)
    thr_b = copy_b.get_slice(tid)
    cute.copy(copy_a, thr_a.partition_S(va), thr_a.retile(fa))
    cute.copy(copy_b, thr_b.partition_S(vb), thr_b.retile(fb))
    cute.gemm(tiled_mma, acc, fa, fb, acc)


@cute.jit
def mma_coords(
    tiled_mma: cute.TiledMma, tid: cutlass.Int32, shape_mn: cutlass.Constexpr
) -> cute.Tensor:
    """This thread's ``(m,n)`` coordinate for each accumulator element.

    The epilogue needs the coordinate to predicate the padded M rows and to index
    any per-row or per-column factor. Coordinates carry no storage.

    Args:
        tiled_mma: From :func:`make_mma`.
        tid: Thread index within the block.
        shape_mn: The same ``(M,N)`` passed to :func:`mma_acc`.

    Returns:
        Partitioned identity tensor, indexable in lockstep with the accumulator.
    """
    return tiled_mma.get_slice(tid).partition_C(cute.make_identity_tensor(shape_mn))


@cute.jit
def mma_store(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    acc: cute.Tensor,
    dst: cute.Tensor,
    shape_mn: cutlass.Constexpr,
    rows: cutlass.Constexpr,
) -> None:
    """Write the accumulator out, dropping the rows M was rounded up by.

    When no rows were added, the store is one vectorized copy. That path needs a
    static destination layout, and a tensor handed in from the host carries dynamic
    strides, so the extents and the row pitch are rebuilt here from ``shape_mn``.
    Sound because every destination is a contiguous row-major ``(rows, N)``
    sub-tensor, which the tensor contract already requires.

    Args:
        tiled_mma: From :func:`make_mma`.
        tid: Thread index within the block.
        acc: From :func:`mma_acc`.
        dst: Destination of shape ``(rows, N)``.
        shape_mn: The same ``(M,N)`` passed to :func:`mma_acc`.
        rows: Logical rows, before :func:`mma_rows` rounded them up.
    """
    if cutlass.const_expr(rows == shape_mn[0]):
        view = cute.make_tensor(
            dst.iterator, cute.make_layout(shape_mn, stride=(shape_mn[1], 1))
        )
        cute.autovec_copy(acc, tiled_mma.get_slice(tid).partition_C(view))
    else:
        crd = mma_coords(tiled_mma, tid, shape_mn)
        for i in cutlass.range_constexpr(cute.size(acc)):
            m, n = crd[i]
            if m < rows:
                dst[m, n] = acc[i]
