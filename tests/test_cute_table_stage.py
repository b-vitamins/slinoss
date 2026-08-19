"""The transpose flag of ``stage_rotated``, and ``stage_matrix``, against float64.

Both helpers write only into shared memory on the shipped path, so one probe kernel
stages a chunk, runs the chunk-local prefixes, builds the 3x3 table, applies one
slot of it with the flag off and on into two operand tiles, applies one chunk-wide
matrix to a ``(P, 3N)`` state, and writes every tile out.

The table is not fabricated. It is built inside the probe by ``stage_chunk``,
``chunk_prefixes`` and ``build_table``, which is what fixes that the index triple
the flag permutes is the triple ``build_table`` wrote; a table from ``randn`` would
admit either permutation. The authority is then a float64 expression of ``A v``,
``A^T v`` and ``mat @ v`` over the table the probe read out, so the residual is the
staging arithmetic alone. The table itself is the subject of
``test_cute_device_math``, and the ``(P, 3N)`` state comes from the reference's own
chunk decomposition rather than from ``randn``.

Inputs are built in float32 and rounded once to the operand dtype, so both paths see
the same bits.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from torch import Tensor

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    dev_tensor,
    smem_bytes,
    smem_capacity,
)
from slinoss.config import MAX_CHUNK
from slinoss.ops.so3ssd import chunked_forward
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AP,
    THREADS,
    scalar_tile,
    table_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.mma import (
    SMEM_SEGMENT,
    fp32_tile,
    operand_tile,
)
from slinoss.ops.so3ssd.cute.prefix import chunk_prefixes
from slinoss.ops.so3ssd.cute.table import (
    build_table,
    stage_chunk,
    stage_matrix,
    stage_rotated,
)
from tests.conftest import ScanInputs, assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# scanprep maps the raw log scale through a negative softplus, so a negative bias is
# a weak decay. Without it the table entries past the first few tokens are scaled to
# nothing, which narrows the range the matvec is measured over.
LS_BIAS = -4.0

SENTINEL = -7.0
"""Prefill of the float32 tile. ``keep_fp32`` off must leave every word of it."""

# The slot the transpose is measured on. Ac is a rotation matrix: dense, and
# asymmetric at every reachable nonzero angle.
ROT_SLOT = TABLE_AC

# The slot the one chunk-wide matrix comes from, read at token 0, which exists in
# every launched chunk. Ap composes the rotation with a tap, so it is neither
# orthogonal nor symmetric and a wrong index triple cannot survive it.
MAT_SLOT = TABLE_AP


def _probe_tiles(
    chunk: int, rows: int, dim: int, itemsize: int
) -> list[tuple[Tile, int]]:
    """The probe's shared-memory allocations, in the order the kernel makes them."""
    return [
        (trans_tile(chunk), 4),
        (tap_tile(chunk), 4),
        (scalar_tile(chunk), 4),
        (trans_tile(chunk), 4),
        (table_tile(chunk, 3), 4),
        (operand_tile(chunk, dim), itemsize),
        (operand_tile(chunk, dim), itemsize),
        (operand_tile(rows, dim), itemsize),
        (fp32_tile(rows, dim), 4),
    ]


@cute.kernel
def _probe_kernel(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gv: cute.Tensor,
    gz: cute.Tensor,
    otable: cute.Tensor,
    ooff: cute.Tensor,
    oon: cute.Tensor,
    omat: cute.Tensor,
    omat32: cute.Tensor,
    seqlen: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    keep_fp32: cutlass.Constexpr,
) -> None:
    """Stage one chunk both ways and write every tile out.

    One block per ``(chunk, batch, head)``. The two :func:`stage_rotated` calls
    differ in the trailing flag and in nothing else, so the pair is comparable word
    for word.

    Args:
        gtrans: ``(B,H,T,4)`` float32 ``(w_x, w_y, w_z, ls)``.
        gtap: ``(B,H,T,2,4)`` float32 per-tap ``(kr, g, h, 0)``.
        gv: ``(B,H,T,3N)`` operand-dtype vectors the table is applied to.
        gz: ``(B,H,C,P,3N)`` float32 chunk-start states.
        otable: ``(B,H,C,3,L,9)`` float32, written with the table.
        ooff: ``(B,H,C,L,3N)`` operand-dtype, written with the untransposed tile.
        oon: ``(B,H,C,L,3N)`` operand-dtype, written with the transposed tile.
        omat: ``(B,H,C,P,3N)`` operand-dtype, written with the narrowed matrix tile.
        omat32: ``(B,H,C,P,3N)`` float32, written with the float32 matrix tile.
        seqlen: ``T``. Compile-time.
        threads: Block width. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        keep_fp32: Passed through to :func:`stage_matrix`. Compile-time.
    """
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()
    lanes = dim // 3

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, 3).layout(), 16)
    soff = smem.allocate_tensor(
        gv.element_type, operand_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    son = smem.allocate_tensor(
        gv.element_type, operand_tile(chunk, dim).layout(), SMEM_SEGMENT
    )
    sdst = smem.allocate_tensor(
        gv.element_type, operand_tile(rows, dim).layout(), SMEM_SEGMENT
    )
    sfp32 = smem.allocate_tensor(cutlass.Float32, fp32_tile(rows, dim).layout(), 16)

    t0 = cidx * chunk
    valid = cutlass.min(cutlass.Int32(chunk), cutlass.Int32(seqlen) - t0)
    stage_chunk(
        gtrans[bidx, hidx, None, None],
        gtap[bidx, hidx, None, None, None],
        strans,
        stap,
        t0,
        valid,
        tid,
        threads,
        chunk,
    )
    for i in cutlass.range(tid, rows * dim, threads):
        p = i // dim
        sfp32[p, i - p * dim] = cutlass.Float32(SENTINEL)
    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()
    build_table(strans, stap, squat, stable, tid, threads, chunk, 3)
    cute.arch.sync_threads()

    # ``slp`` is handed over as the scale tile and left unread. The scale multiplies
    # the result after the matvec, so it does not interact with the transpose; it is
    # swept by the forward parity tests instead.
    stage_rotated(
        gv,
        gv,
        soff,
        stable,
        slp,
        bidx,
        hidx,
        t0,
        cutlass.Int32(0),
        valid,
        tid,
        ROT_SLOT,
        0,
        threads,
        chunk,
        lanes,
        False,
        False,
    )
    stage_rotated(
        gv,
        gv,
        son,
        stable,
        slp,
        bidx,
        hidx,
        t0,
        cutlass.Int32(0),
        valid,
        tid,
        ROT_SLOT,
        0,
        threads,
        chunk,
        lanes,
        False,
        False,
        True,
    )
    # Nine words at one address for the whole block: the broadcast read a matrix
    # operand is required to be, hoisted out of the lane loop into registers.
    mat = (
        stable[MAT_SLOT, 0, 0],
        stable[MAT_SLOT, 0, 1],
        stable[MAT_SLOT, 0, 2],
        stable[MAT_SLOT, 0, 3],
        stable[MAT_SLOT, 0, 4],
        stable[MAT_SLOT, 0, 5],
        stable[MAT_SLOT, 0, 6],
        stable[MAT_SLOT, 0, 7],
        stable[MAT_SLOT, 0, 8],
    )
    stage_matrix(
        gz, sdst, sfp32, mat, bidx, hidx, cidx, tid, threads, rows, lanes, keep_fp32
    )
    cute.arch.sync_threads()

    for i in cutlass.range(tid, chunk * dim, threads):
        r = i // dim
        c = i - r * dim
        ooff[bidx, hidx, cidx, r, c] = soff[r, c]
        oon[bidx, hidx, cidx, r, c] = son[r, c]
    for i in cutlass.range(tid, rows * dim, threads):
        p = i // dim
        c = i - p * dim
        omat[bidx, hidx, cidx, p, c] = sdst[p, c]
        omat32[bidx, hidx, cidx, p, c] = sfp32[p, c]
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            for slot in cutlass.range_constexpr(3):
                for entry in cutlass.range_constexpr(9):
                    otable[bidx, hidx, cidx, slot, token, entry] = stable[
                        slot, token, entry
                    ]


@cute.jit
def _probe_launch(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gv: cute.Tensor,
    gz: cute.Tensor,
    otable: cute.Tensor,
    ooff: cute.Tensor,
    oon: cute.Tensor,
    omat: cute.Tensor,
    omat32: cute.Tensor,
    seqlen: cutlass.Constexpr,
    chunks: cutlass.Constexpr,
    bsz: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    keep_fp32: cutlass.Constexpr,
) -> None:
    _probe_kernel(
        gtrans,
        gtap,
        gv,
        gz,
        otable,
        ooff,
        oon,
        omat,
        omat32,
        seqlen,
        threads,
        chunk,
        rows,
        dim,
        keep_fp32,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


class Probe(NamedTuple):
    """What the probe wrote out.

    Attributes:
        table: ``(B,H,C,3,L,9)`` float32 transform table.
        off: ``(B,H,C,L,3N)`` the table applied without the transpose.
        on: ``(B,H,C,L,3N)`` the same table applied with it.
        mat: ``(B,H,C,P,3N)`` the chunk-wide matrix, narrowed to the operand dtype.
        mat32: ``(B,H,C,P,3N)`` the same result at float32, or :data:`SENTINEL`.
    """

    table: Tensor
    off: Tensor
    on: Tensor
    mat: Tensor
    mat32: Tensor


def _run_probe(inp: ScanInputs, zstart: Tensor, chunk: int, keep_fp32: bool) -> Probe:
    """Launch the probe over every chunk of ``inp``."""
    bsz, heads, seqlen, _ = inp.trans.shape
    chunks, rows, dim = zstart.shape[2], zstart.shape[3], zstart.shape[4]
    assert_smem_fits(
        f"table-stage[L{chunk}/P{rows}/3N{dim}]",
        smem_bytes(_probe_tiles(chunk, rows, dim, inp.C.element_size())),
    )
    wide = {"device": inp.trans.device, "dtype": torch.float32}
    thin = {"device": inp.trans.device, "dtype": inp.C.dtype}
    otable = torch.empty(bsz, heads, chunks, 3, chunk, 9, **wide)
    omat32 = torch.empty(bsz, heads, chunks, rows, dim, **wide)
    ooff = torch.empty(bsz, heads, chunks, chunk, dim, **thin)
    oon = torch.empty_like(ooff)
    omat = torch.empty(bsz, heads, chunks, rows, dim, **thin)
    _probe_launch(
        dev_tensor(inp.trans),
        dev_tensor(inp.K),
        dev_tensor(inp.C),
        dev_tensor(zstart),
        dev_tensor(otable),
        dev_tensor(ooff),
        dev_tensor(oon),
        dev_tensor(omat),
        dev_tensor(omat32),
        seqlen,
        chunks,
        bsz,
        heads,
        THREADS,
        chunk,
        rows,
        dim,
        keep_fp32,
    )
    torch.cuda.synchronize()
    return Probe(otable, ooff, oon, omat, omat32)


def _make(
    bsz: int,
    heads: int,
    seqlen: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
    w_scale: float,
) -> ScanInputs:
    """One operand set: float32 pinned tensors, ``dtype`` vectors."""
    return make_inputs(
        bsz=bsz,
        heads=heads,
        seqlen=seqlen,
        rows=rows,
        lanes=lanes,
        dtype=torch.float32,
        device="cuda",
        streaming=False,
        w_scale=w_scale,
        ls_bias=LS_BIAS,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


def _zstart(inp: ScanInputs, chunk: int) -> Tensor:
    """The reference's chunk-start state as a kernel reads it, ``(B,H,C,P,3N)``."""
    ref = chunked_forward(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        chunk,
        z0=None if inp.z0 is None else inp.z0.double(),
    )
    return ref.zstart.flatten(-2, -1).float().contiguous()


def _slot(table: Tensor, slot: int) -> Tensor:
    """One slot of the read-back table as float64 3x3 matrices, ``(B,H,C,L,3,3)``."""
    return table[:, :, :, slot].double().unflatten(-1, (3, 3))


def _rotated_oracle(
    table: Tensor, src: Tensor, chunk: int, chunks: int, transposed: bool
) -> Tensor:
    """``A v`` or ``A^T v`` per token, in float64, over the read-back table.

    Args:
        table: ``(B,H,C,3,L,9)`` as the probe wrote it.
        src: ``(B,H,T,3N)`` source vectors.
        chunk: ``L``.
        chunks: Chunks the sequence covers.
        transposed: Which of the two the kernel was asked for.

    Returns:
        ``(B,H,C,L,3N)``. Tokens past the sequence are zero, which is what the
        staging writes there.
    """
    padded = torch.nn.functional.pad(
        src.double(), (0, 0, 0, chunks * chunk - src.shape[-2])
    )
    vec = padded.unflatten(-2, (chunks, chunk)).unflatten(-1, (-1, 3))
    spec = "bhclji,bhclnj->bhclni" if transposed else "bhclij,bhclnj->bhclni"
    return torch.einsum(spec, _slot(table, ROT_SLOT), vec).flatten(-2, -1)


def _matrix_oracle(table: Tensor, zstart: Tensor) -> Tensor:
    """``mat @ v`` over the ``(P, 3N)`` state, in float64, one matrix per chunk.

    Args:
        table: ``(B,H,C,3,L,9)`` as the probe wrote it.
        zstart: ``(B,H,C,P,3N)`` float32 state.

    Returns:
        ``(B,H,C,P,3N)``.
    """
    mat = _slot(table, MAT_SLOT)[:, :, :, 0]
    vec = zstart.double().unflatten(-1, (-1, 3))
    return torch.einsum("bhcij,bhcpnj->bhcpni", mat, vec).flatten(-2, -1)


# (bsz, heads, seqlen, chunk, rows, lanes, dtype). Two cases, one value each of
# every axis that changes what is generated:
#
# - ``N``, because both helpers walk the lane dimension with a stride loop whose
#   trip count is per thread;
# - ``P`` above and below the block width, which is that loop for
#   :func:`stage_matrix` and the only axis that puts more than one prefetch group in
#   it;
# - a ragged tail, where zeroing the rows past the sequence is what keeps the
#   transposed matvec off a token that does not exist;
# - the operand dtype, because the narrowing into the tile is a different
#   instruction at each width.
#
# The axes not swept are the table slot, the vector's token offset ``back``, the
# streaming carry-in, and the per-token scale. None interacts with the transpose:
# the flag permutes the matrix operand's index triple and changes neither which
# vector is loaded nor what multiplies the result afterwards. All four are swept by
# the forward parity tests, which exercise the defaulted flag.
CASES = [
    pytest.param(2, 2, 256, 64, 16, 32, torch.bfloat16, id="bf16-exact-wide-state"),
    pytest.param(1, 2, 200, 64, 144, 16, torch.float16, id="fp16-ragged-wide-rows"),
]

# The kernel narrows the transformed vector once, on the way into the tile, and the
# oracle applies the same float64 matrix to the same bits, so each bound is one
# half-ulp of the tile's dtype at the bottom of a binade: 2^-8 = 3.9e-3 at bfloat16
# and 2^-11 = 4.9e-4 at float16. The float32 tile ``keep_fp32`` writes is not
# narrowed at all and carries the rounding of three products and two float32 sums,
# which is a small multiple of 1.2e-7. Measured 3.2e-03, 2.8e-04 and 5.5e-08.
BOUNDS = {torch.bfloat16: 4e-3, torch.float16: 6e-4, torch.float32: 2e-7}


@pytest.mark.parametrize(
    ("bsz", "heads", "seqlen", "chunk", "rows", "lanes", "dtype"), CASES
)
def test_staged_transforms_match_the_read_back_table(
    bsz: int,
    heads: int,
    seqlen: int,
    chunk: int,
    rows: int,
    lanes: int,
    dtype: torch.dtype,
) -> None:
    """Both flag values and the chunk-wide matrix match float64 on the same table."""
    inp = _make(bsz, heads, seqlen, rows, lanes, dtype, w_scale=2.0)
    zstart = _zstart(inp, chunk)
    got = _run_probe(inp, zstart, chunk, keep_fp32=True)
    chunks = -(-seqlen // chunk)
    tag = f"cute-stage[{bsz}x{heads}x{seqlen}/L{chunk}/P{rows}/N{lanes}]"

    # Without this the two oracles coincide and either index triple passes.
    for slot in (ROT_SLOT, MAT_SLOT):
        entries = _slot(got.table, slot)
        assert not torch.allclose(entries, entries.transpose(-2, -1), atol=1e-3)

    bound = BOUNDS[dtype]
    assert_max_rel(
        got.off,
        _rotated_oracle(got.table, inp.C, chunk, chunks, False),
        bound,
        f"{tag}.rotated",
    )
    assert_max_rel(
        got.on,
        _rotated_oracle(got.table, inp.C, chunk, chunks, True),
        bound,
        f"{tag}.rotated-transposed",
    )
    want = _matrix_oracle(got.table, zstart)
    assert_max_rel(got.mat, want, bound, f"{tag}.matrix")
    # The float32 copy is the result before the narrowing, not a widened operand, so
    # it carries the float32 bound whatever the operand dtype is.
    assert_max_rel(got.mat32, want, BOUNDS[torch.float32], f"{tag}.matrix-float32")
    # A comparison against zeros passes whatever the matvec does.
    assert torch.count_nonzero(got.off) > 0
    assert torch.count_nonzero(got.mat) > 0

    # Rows past the sequence are zeroed on the float32 input, so the nine FMAs are
    # exact there and no consumer of the tile needs a predicate.
    valid = seqlen - (chunks - 1) * chunk
    assert torch.count_nonzero(got.off[:, :, -1, valid:]) == 0
    assert torch.count_nonzero(got.on[:, :, -1, valid:]) == 0


def test_transposed_off_is_bitwise_the_untransposed_matvec() -> None:
    """At ``w = 0`` every table slot is diagonal, so both flag values must agree.

    The flag permutes an index triple during the trace and changes nothing else. On a
    symmetric table both permutations select the same nine words in the same order,
    so the two tiles are bitwise equal unless the transposed branch altered the
    arithmetic rather than the index. A tolerance here would admit exactly that.

    ``w = 0`` is reachable and exact: ``quat_exp(0)`` has a zero vector part, so
    every off-diagonal entry of the rotation matrix is a product with zero, and the
    tap chart at the origin is ``kr I``, which is where the polynomial form is
    defined and the axis normal form is not.
    """
    chunk = 64
    inp = _make(2, 2, 128, 16, 16, torch.bfloat16, w_scale=0.0)
    got = _run_probe(inp, _zstart(inp, chunk), chunk, keep_fp32=False)

    entries = _slot(got.table, ROT_SLOT)
    diagonal = torch.diag_embed(entries.diagonal(dim1=-2, dim2=-1))
    assert torch.count_nonzero(entries - diagonal) == 0
    assert torch.count_nonzero(diagonal) > 0
    assert torch.equal(got.off, got.on)
    assert torch.count_nonzero(got.off) > 0


def test_keep_fp32_off_leaves_the_float32_tile_untouched() -> None:
    """The second store is gated, so a consumer of one width can pass any tile.

    Asserted against a prefill: a store that ignored the flag would overwrite it, and
    a comparison against the oracle alone cannot see the extra traffic.
    """
    chunk = 64
    inp = _make(2, 2, 128, 16, 16, torch.bfloat16, w_scale=2.0)
    zstart = _zstart(inp, chunk)
    got = _run_probe(inp, zstart, chunk, keep_fp32=False)

    assert torch.equal(got.mat32, torch.full_like(got.mat32, SENTINEL))
    # The operand tile is still written, so the flag gates the float32 store alone.
    assert_max_rel(
        got.mat,
        _matrix_oracle(got.table, zstart),
        BOUNDS[torch.bfloat16],
        "cute-stage[keep-fp32-off].matrix",
    )


@pytest.mark.parametrize(("chunk", "rows", "dim"), [(64, 144, 48), (MAX_CHUNK, 16, 96)])
def test_shared_memory_budget_fits_the_queried_capacity(
    chunk: int, rows: int, dim: int
) -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The two binding cases are the widest ``P`` and the longest chunk. Both allocate
    an operand tile and a float32 tile over one ``(P, 3N)`` extent, which is what a
    kernel keeping both widths of a transformed state pays.
    """
    tiles = _probe_tiles(chunk, rows, dim, 2)
    nbytes = smem_bytes(tiles)
    assert assert_smem_fits(f"table-stage[L{chunk}]", nbytes) == nbytes
    assert nbytes <= smem_capacity()
    # The float32 tile is pitched at its own element width. Reusing the operand pitch
    # would put a 48-element row at 56 elements, an even number of float32 segments,
    # which is a two-way conflict on every row.
    unit = SMEM_SEGMENT // 4
    assert (fp32_tile(rows, dim).stride[0] // unit) % 2 == 1
