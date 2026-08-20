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

Block width. Four warps partitioning M is :func:`mma_atoms` at :data:`WARPS`.
Eight warps do not partition M further: the M mode stays :data:`WARPS` atoms wide
and the extra warps subdivide the tile's N mode, two warps to each M tile. So
:data:`MMA_TILE_M` is flat in the block width and every M-extent shared tile with
it, which is what puts 256 threads within reach of a kernel whose live set already
fills the carveout: one resident block of eight warps holds twice the resident
warps of one resident block of four at the same bytes. The tile's N mode spans
:data:`MMA_TILE_ATOMS_N` atoms and no more, so :data:`WARPS_WIDE` is the widest
block this atom admits with M pinned. Widening the N mode to take more would raise
the N divisibility requirement to 32, which ``3N`` at 48 fails.

Two consequences of splitting N, both read off the device rather than assumed:

- A warp group holds one atom of the B operand's N mode rather than two, so its
  ``ldmatrix`` moves two 8x8 matrices rather than four. A count the fragment does
  not divide fails IR verification, so :func:`mma_matrices` derives it from the
  tiling.
- :func:`mma_areg` does not survive the split, and a chained consumer takes its
  left operand from shared memory at :data:`WARPS_WIDE`. A thread's consecutive N
  steps are two atoms apart at two groups, so the N mode it would reread as K is
  not contiguous. Letting K absorb the warps instead keeps that reread intact and
  costs more than it saves: an atom layout with a K mode replicates the
  accumulator across the K warps rather than partitioning it, so every product
  needs a float32 ``M*N`` cross-warp reduction in the arena the wider block exists
  to leave alone.

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

from slinoss._cute import Tile, narrow
from slinoss.ops.so3ssd.cute.common import WARPS

__all__ = [
    "MMA_INST",
    "MMA_PAIR",
    "MMA_TILE_ATOMS_N",
    "MMA_TILE_K",
    "MMA_TILE_M",
    "MMA_TILE_N",
    "SMEM_SEGMENT",
    "THREADS_WIDE",
    "WARPS_WIDE",
    "fp32_tile",
    "make_mma",
    "mma_acc",
    "mma_areg",
    "mma_atoms",
    "mma_coords",
    "mma_gemm",
    "mma_gemm_areg",
    "mma_groups",
    "mma_matrices",
    "mma_offsets",
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

MMA_TILE_ATOMS_N: int = MMA_TILE_N // MMA_INST[1]
"""Atoms the tile's N mode spans, and the warp groups it can be split into."""

WARPS_WIDE: int = WARPS * MMA_TILE_ATOMS_N
"""Warps of the widest block the atom admits with the M mode pinned.

Two warp groups of :data:`WARPS`, each holding one atom of the tile's N mode. A
third group has no atom to hold, so this is a ceiling and not a default: the
four-warp form remains what every caller gets from :func:`make_mma` unasked."""

THREADS_WIDE: int = WARPS_WIDE * 32
"""Threads of a :data:`WARPS_WIDE` block, the wide sibling of
:data:`slinoss.ops.so3ssd.cute.common.THREADS`."""

MMA_PAIR: int = 2
"""Adjacent output columns one thread holds in the atom's C fragment.

The ``m16n8k16`` C layout gives a thread four values as two column pairs eight rows
apart, and consecutive flat indices of the accumulator are one pair. The pair is
contiguous in a row-major destination, so the epilogue's unit is two elements, not
one. Four lanes of a phase then cover 32 bytes of a float32 row, which is one whole
sector; one element at a time covers half of it and pays the other half anyway.
"""

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


def fp32_tile(rows: int, width: int) -> Tile:
    """Row-major float32 tile, ``(rows, smem_pitch(width, 4))``.

    The float32 sibling of :func:`operand_tile`, for a quantity that I4 pins to
    float32 and that therefore never becomes a GEMM operand. It exists because the
    pitch depends on the element size: taking the operand pitch for a four-byte
    element halves the segment count, and at ``3N = 48`` that lands on an even
    number of segments, which is the two-way conflict the odd rule exists to avoid.

    Args:
        rows: Rows to allocate.
        width: Elements per row that carry data. The rest of the pitch is padding
            outside every view.
    """
    pitch = smem_pitch(width, 4)
    return Tile((rows, pitch), (pitch, 1))


def mma_atoms(warps: int) -> tuple[int, int, int]:
    """Atom layout ``(M,N,K)`` for a block of ``warps`` warps.

    The M mode is :data:`WARPS` atoms at every legal block width, so
    :data:`MMA_TILE_M` and every shared tile whose rows or pitch carry an M extent
    are flat in the warp count. Warps past :data:`WARPS` go to the N mode, which
    holds :data:`MMA_TILE_ATOMS_N` atoms and therefore that many warp groups.

    The K mode is never given a warp. An atom layout with a K mode replicates the
    accumulator across those warps instead of partitioning it, so the partial sums
    need a cross-warp reduction that neither of the other two modes needs.

    Args:
        warps: Warps per block. A multiple of :data:`WARPS`, at most
            :data:`WARPS_WIDE`.

    Returns:
        The layout to hand :func:`cute.make_tiled_mma`.

    Raises:
        ValueError: If ``warps`` is not a legal block width.
    """
    if warps <= 0 or warps % WARPS or warps > WARPS_WIDE:
        raise ValueError(
            f"warps must be a multiple of {WARPS} at most {WARPS_WIDE}, got {warps}"
        )
    return (WARPS, warps // WARPS, 1)


def mma_groups(tiled_mma: cute.TiledMma) -> int:
    """Warp groups the tiling splits the tile's N mode into.

    One for the four-warp form, :data:`MMA_TILE_ATOMS_N` for the wide one. Read off
    the tiling rather than passed in, so a helper cannot be handed a count that
    disagrees with the tiled MMA it was given.

    Not a ``@cute.jit`` function: it is layout algebra on a static layout and it
    emits nothing.

    Args:
        tiled_mma: From :func:`make_mma`.

    Returns:
        The N extent of the tiling's thread layout.
    """
    return cute.size(tiled_mma.thr_layout_vmnk, mode=[2])


def mma_matrices(tiled_mma: cute.TiledMma) -> int:
    """8x8 matrices one ``ldmatrix`` of the B operand moves.

    A warp group holds ``MMA_TILE_N // groups`` columns over :data:`MMA_TILE_K`
    rows of B, which is that many 64-element matrices: four at one group, two at
    two. The op's count has to divide the fragment the copy retiles onto, and a
    count that does not fails IR verification rather than returning a wrong answer.

    The A operand is not split. Its patch is one atom of M over
    :data:`MMA_TILE_K`, four matrices at every block width, and the tiling
    broadcasts it across the N groups.

    Args:
        tiled_mma: From :func:`make_mma`.

    Returns:
        The count to hand :class:`cute.nvgpu.warp.LdMatrix8x8x16bOp`.
    """
    return MMA_TILE_N * MMA_TILE_K // (mma_groups(tiled_mma) * 64)


@cute.jit
def make_mma(
    dtype: cutlass.Constexpr, warps: cutlass.Constexpr = WARPS
) -> cute.TiledMma:
    """Build the one tiled MMA.

    Constructed on the host side and passed into the kernel, so a kernel holds no
    knowledge of the atom.

    Args:
        dtype: Operand element type. ``cutlass.BFloat16`` or ``cutlass.Float16``.
        warps: Warps per block, defaulting to :data:`WARPS`. :data:`WARPS_WIDE`
            asks for the wide form, whose tile is the same and whose block is
            twice as many warps.

    Returns:
        The tiled MMA with tile ``(MMA_TILE_M, MMA_TILE_N, MMA_TILE_K)``.
    """
    return cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, MMA_INST),
        mma_atoms(warps),
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

        B's matrix count comes from :func:`mma_matrices`, since a wide tiling gives
        a warp group half the N mode. A's is four at every block width.
    """
    thr = tiled_mma.get_slice(tid)
    fa = tiled_mma.make_fragment_A(thr.partition_A(va))
    fb = tiled_mma.make_fragment_B(thr.partition_B(vb))
    atom_a = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not a_k_major, 4), va.element_type
    )
    atom_b = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not b_k_major, mma_matrices(tiled_mma)),
        vb.element_type,
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


def mma_offsets(
    tiled_mma: cute.TiledMma, shape_mn: cutlass.Constexpr
) -> tuple[tuple[int, int], ...]:
    """Trace-time ``(m,n)`` offset of every accumulator element in the fragment.

    A thread's coordinate is its own base plus these offsets, and the base is one
    value for the whole fragment, so a residue of an offset is a trace-time constant
    where a residue of the coordinate is not. An epilogue whose destination index is
    a residue of the column uses this to keep that index static.

    Not a ``@cute.jit`` function: it is layout algebra on thread zero's slice, whose
    base is the origin, and it emits nothing.

    Args:
        tiled_mma: From :func:`make_mma`.
        shape_mn: The same ``(M,N)`` passed to :func:`mma_acc`.

    Returns:
        One ``(m,n)`` pair of Python ints per element, in fragment order.
    """
    origin = tiled_mma.get_slice(0).partition_C(cute.make_identity_tensor(shape_mn))
    return tuple((origin[i][0], origin[i][1]) for i in range(cute.size(origin.layout)))


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

    When no rows were added, the store is one vectorized copy over the whole
    fragment. When rows were added, the predicate is per row and the unit is the
    :data:`MMA_PAIR` columns the fragment already holds side by side, so a
    predicated store is one access rather than two.

    Both paths need a static destination layout, and a tensor handed in from the
    host carries dynamic strides, so the extents and the row pitch are rebuilt here
    from ``shape_mn``. Sound because every destination is a contiguous row-major
    ``(rows, N)`` sub-tensor, which the tensor contract already requires.

    Args:
        tiled_mma: From :func:`make_mma`.
        tid: Thread index within the block.
        acc: From :func:`mma_acc`.
        dst: Destination of shape ``(rows, N)``.
        shape_mn: The same ``(M,N)`` passed to :func:`mma_acc`.
        rows: Logical rows, before :func:`mma_rows` rounded them up.

    Invariants:
        ``N`` is a multiple of :data:`MMA_TILE_N`, so :data:`MMA_PAIR` divides both
        the row width and the column of every pair, and no pair straddles a row.

        The destination's base is aligned to :data:`MMA_PAIR` elements. Every
        destination is a sub-tensor of a ``(...,rows,N)`` tensor sliced on the
        leading modes, so its base is a whole multiple of ``rows * N`` elements and
        ``MMA_TILE_N`` divides ``N``.

        Flat accumulator index ``2i`` and ``2i+1`` are adjacent columns of one row,
        which is what makes a pair one access. This is the atom's C layout, whose
        innermost mode has extent :data:`MMA_PAIR`, so it holds for any ``MMA_M`` and
        ``MMA_N`` and is independent of :data:`WARPS` and :data:`MMA_TILE_M`. It
        would not survive a ``permutation_mnk`` that reorders the N mode, and it
        fails silently rather than loudly if it ever does.
    """
    # A sub-tensor taken at a dynamic index reports its iterator as aligned to one
    # element whatever the parent claimed, and cute.autovec_copy caps the access
    # width at the iterator's claim. Restating the alignment the invariant above
    # already gives is what makes either path below wider than one element; without
    # it both measured 1.99x the payload's store sectors on sm_86.
    base = dst.iterator.align(MMA_PAIR * (dst.element_type.width // 8))
    if cutlass.const_expr(rows == shape_mn[0]):
        view = cute.make_tensor(
            base, cute.make_layout(shape_mn, stride=(shape_mn[1], 1))
        )
        cute.autovec_copy(acc, tiled_mma.get_slice(tid).partition_C(view))
    else:
        cols = shape_mn[1]
        flat = cute.make_tensor(base, cute.make_layout((rows * cols,), stride=(1,)))
        pairs = cute.zipped_divide(flat, (MMA_PAIR,))
        frag = cute.make_fragment((MMA_PAIR,), dst.element_type)
        crd = mma_coords(tiled_mma, tid, shape_mn)
        for i in cutlass.range_constexpr(cute.size(acc) // MMA_PAIR):
            first = i * MMA_PAIR
            # Filled before the predicate: a value produced inside a dynamic branch
            # is not readable after it, and the fill costs nothing on the rows the
            # predicate drops, which are whole warps.
            for j in cutlass.range_constexpr(MMA_PAIR):
                frag[j] = narrow(acc[first + j], dst.element_type)
            m, n = crd[first]
            if m < rows:
                cute.autovec_copy(
                    frag, pairs[(None, m * (cols // MMA_PAIR) + n // MMA_PAIR)]
                )


@cute.jit
def mma_areg(frag: cute.Tensor) -> cute.Tensor:
    """View a fragment in the C layout as the A operand of the same tiled MMA.

    One warp's C fragment for the atom is four values over a 16x8 tile; its A
    fragment is eight over 16x16, and the eight are the two adjacent 16x8 C tiles
    with the N mode reread as K. The correspondence is per thread, so a product one
    contraction computed is the left operand of the next with no data movement, no
    ``ldmatrix``, and no barrier. FlashAttention-2's ``convert_layout_acc_Aregs``.

    Args:
        frag: Register fragment laid out as :func:`mma_acc` lays one out,
            ``((4), MMA_M, MMA_N)``, already narrowed to the operand dtype.

    Returns:
        A view over the same registers with layout ``((4,2), MMA_M, MMA_N // 2)``:
        the A fragment of a GEMM whose K extent is the N extent of ``frag``.

    Invariants:
        ``MMA_N`` must be even, so the N extent must be a multiple of twice the
        atom's 8 columns. Every N extent in the operator is a multiple of
        :data:`MMA_TILE_N`, which is 16, so this holds. The consuming GEMM's B
        operand must have that same K extent.

        The fragment must come from a tiling of one N group. At
        :data:`MMA_TILE_ATOMS_N` groups a thread's consecutive N steps are two
        atoms apart rather than adjacent, so the pair this reads as one 16-row K
        slab is two 8-row slabs eight rows apart. :func:`mma_gemm_areg` refuses
        that tiling, which is where the reread is caught.
    """
    split = cute.logical_divide(frag.layout, (None, None, 2))
    return cute.make_tensor(
        frag.iterator,
        cute.make_layout(
            ((split.shape[0], split.shape[2][0]), split.shape[1], split.shape[2][1]),
            stride=(
                (split.stride[0], split.stride[2][0]),
                split.stride[1],
                split.stride[2][1],
            ),
        ),
    )


@cute.jit
def mma_gemm_areg(
    tiled_mma: cute.TiledMma,
    tid: cutlass.Int32,
    acc: cute.Tensor,
    fa: cute.Tensor,
    vb: cute.Tensor,
    b_k_major: cutlass.Constexpr,
) -> None:
    """Accumulate ``fa @ vb^T`` into ``acc`` with A already in registers.

    The sibling of :func:`mma_gemm` for a left operand a previous contraction
    produced. Only B is loaded through ``ldmatrix``.

    Args:
        tiled_mma: From :func:`make_mma`. The same one that produced ``fa``, since
            the M partition across warps has to agree.
        tid: Thread index within the block.
        acc: From :func:`mma_acc`. Updated in place.
        fa: A fragment from :func:`mma_areg`.
        vb: Shared-memory view of shape ``(N,K)``.
        b_k_major: Whether ``vb``'s K mode is the stride-1 mode.

    Raises:
        ValueError: If the tiling splits the tile's N mode across warp groups.
            The reread :func:`mma_areg` performs is not contiguous in K there, and
            the product would be a plausible wrong answer rather than a failure.
            A chained consumer of a wide tiling stages its left operand through
            shared memory and takes :func:`mma_gemm`.
    """
    if mma_groups(tiled_mma) != 1:
        raise ValueError(
            f"mma_gemm_areg needs one N group, got {mma_groups(tiled_mma)}"
        )
    thr = tiled_mma.get_slice(tid)
    fb = tiled_mma.make_fragment_B(thr.partition_B(vb))
    atom_b = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(not b_k_major, 4), vb.element_type
    )
    copy_b = cute.make_tiled_copy_B(atom_b, tiled_mma)
    thr_b = copy_b.get_slice(tid)
    cute.copy(copy_b, thr_b.partition_S(vb), thr_b.retile(fb))
    cute.gemm(tiled_mma, acc, fa, fb, acc)
