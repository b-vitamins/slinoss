"""Device-side transition math against the float64 reference.

One probe kernel stages a chunk, runs both chunk-local prefixes, and composes the
3x3 table, then writes every intermediate out. That covers, in one launch, the
quaternion exponential series, the composition order, the warp prefix scans, the
renormalization, the homogeneous rotation matrix, the tap chart, the 3x3 product
and transpose, the ragged-tail staging, and the chunk endpoint.

The probe exists because these quantities never reach global memory on the shipped
path. Comparing them requires a kernel written for the comparison; the alternative
is checking them only through the output of the kernels that consume them, which
localizes nothing.

Inputs are built in float32, then upcast for the reference, so both paths see the
same bits and every difference is float32 arithmetic.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    cute_dtype,
    dev_tensor,
    smem_bytes,
    smem_capacity,
)
from slinoss.config import MAX_CHUNK, MIN_CHUNK
from slinoss.ops.so3ssd import chunked_forward, chunked_forward_fused
from slinoss.ops.so3ssd.cute.common import (
    TABLE_AC,
    TABLE_AC_SOLE,
    TABLE_AFUSE,
    TABLE_AN,
    TABLE_AP,
    THREADS,
    table_tile,
    tap_tile,
    trans_tile,
    vec_tile,
)
from slinoss.ops.so3ssd.cute.prefix import (
    chunk_endpoint,
    chunk_prefixes,
)
from slinoss.ops.so3ssd.cute.table import build_table, stage_chunk
from tests.conftest import assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]


def _lp_tile(chunk: int) -> Tile:
    """The scalar log-prefix tile. Dense, one entry per token."""
    return Tile((chunk,), (1,))


# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it the prefix at the end of a 128-token chunk reaches
# exp(2*lp) near 4e-90, the chunk decay underflows to zero, and the assertion on
# it is skipped as subnormal rather than checked.
LS_BIAS = -4.0


@cute.kernel
def _probe_kernel(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    olp: cute.Tensor,
    oquat: cute.Tensor,
    otable: cute.Tensor,
    oend: cute.Tensor,
    oscale: cute.Tensor,
    seqlen: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    mats: cutlass.Constexpr,
    fused: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, _lp_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk, mats).layout(), 16)

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
    cute.arch.sync_threads()
    chunk_prefixes(strans, slp, squat, tid, chunk)
    cute.arch.sync_threads()
    build_table(strans, stap, squat, stable, tid, threads, chunk, mats, fused)
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            olp[bidx, hidx, cidx, token] = slp[token]
            for j in cutlass.range_constexpr(4):
                oquat[bidx, hidx, cidx, j, token] = squat[j, token]
            for mat in cutlass.range_constexpr(mats):
                for entry in cutlass.range_constexpr(9):
                    otable[bidx, hidx, cidx, mat, token, entry] = stable[
                        mat, token, entry
                    ]

    if tid == 0:
        quat, cscale = chunk_endpoint(squat, slp, chunk)
        for j in cutlass.range_constexpr(4):
            oend[bidx, hidx, cidx, j] = quat[j]
        oscale[bidx, hidx, cidx] = cscale


@cute.jit
def _probe_launch(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    olp: cute.Tensor,
    oquat: cute.Tensor,
    otable: cute.Tensor,
    oend: cute.Tensor,
    oscale: cute.Tensor,
    seqlen: cutlass.Constexpr,
    chunks: cutlass.Constexpr,
    bsz: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    mats: cutlass.Constexpr,
    fused: cutlass.Constexpr,
) -> None:
    _probe_kernel(
        gtrans,
        gtap,
        olp,
        oquat,
        otable,
        oend,
        oscale,
        seqlen,
        threads,
        chunk,
        mats,
        fused,
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


def _run_probe(
    trans: torch.Tensor,
    tap: torch.Tensor,
    chunk: int,
    mats: int = 3,
    fused: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Launch the probe and return ``(lp, quat, table, cquat, cscale)``."""
    bsz, heads, seqlen, _ = trans.shape
    chunks = (seqlen + chunk - 1) // chunk
    opts = {"device": trans.device, "dtype": torch.float32}
    olp = torch.empty(bsz, heads, chunks, chunk, **opts)
    oquat = torch.empty(bsz, heads, chunks, 4, chunk, **opts)
    otable = torch.empty(bsz, heads, chunks, mats, chunk, 9, **opts)
    oend = torch.empty(bsz, heads, chunks, 4, **opts)
    oscale = torch.empty(bsz, heads, chunks, **opts)
    _probe_launch(
        dev_tensor(trans),
        dev_tensor(tap),
        dev_tensor(olp),
        dev_tensor(oquat),
        dev_tensor(otable),
        dev_tensor(oend),
        dev_tensor(oscale),
        seqlen,
        chunks,
        bsz,
        heads,
        THREADS,
        chunk,
        mats,
        fused,
    )
    torch.cuda.synchronize()
    return olp, oquat, otable, oend, oscale


# The scan structure is set by two things and nothing else: the segment count
# ``ceil(L/32)``, which is the serial depth, and whether ``L`` is a warp multiple,
# which selects the clamp branch. One case per reachable pair, plus the ragged
# tail, which is a staging path rather than a scan path.
SHAPES = [
    pytest.param(2, 3, 256, 64, id="two-segments-exact"),
    pytest.param(2, 3, 200, 64, id="ragged-tail"),
    pytest.param(1, 2, 384, MAX_CHUNK, id="four-segments-exact"),
    pytest.param(2, 2, 96, 48, id="two-segments-inexact"),
    pytest.param(1, 2, 48, MIN_CHUNK, id="one-segment-under-warp"),
    pytest.param(2, 2, 64, 32, id="one-segment-exact"),
]


@pytest.mark.parametrize(("bsz", "heads", "seqlen", "chunk"), SHAPES)
def test_prefixes_and_table(bsz: int, heads: int, seqlen: int, chunk: int) -> None:
    """Every staged, scanned, and composed quantity matches the reference."""
    inp = make_inputs(
        bsz=bsz,
        heads=heads,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
    )
    lp, quat, table, end, cscale = _run_probe(inp.trans, inp.K, chunk)
    ref = chunked_forward(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        chunk,
    )

    tag = f"cute-device[{bsz}x{heads}x{seqlen}/L{chunk}]"
    # float32 Hillis-Steele over at most MAX_CHUNK tokens. The quaternion prefix is
    # the deepest chain and sets the bound; the table inherits it squared through
    # the rotation matrix, which is why the bound is not tighter.
    assert_max_rel(lp, ref.lprefix, 2e-6, f"{tag}.lprefix")
    assert_max_rel(quat, ref.qprefix.movedim(-1, -2), 2e-6, f"{tag}.qprefix")
    assert_max_rel(
        table[:, :, :, TABLE_AP],
        ref.table.ap.flatten(-2, -1),
        4e-6,
        f"{tag}.table.ap",
    )
    assert_max_rel(
        table[:, :, :, TABLE_AN],
        ref.table.an.flatten(-2, -1),
        4e-6,
        f"{tag}.table.an",
    )
    assert_max_rel(
        table[:, :, :, TABLE_AC],
        ref.table.ac.flatten(-2, -1),
        4e-6,
        f"{tag}.table.ac",
    )

    # I5 is a property of the prefix, not of its distance from the reference: a
    # comparison against float64 sees drift only after the rotation matrix has
    # squared it, and by then it is inside the bound above. The norm is what the
    # projection controls, so the norm is what is asserted. Measured 1.8e-07 with
    # the projection and 2.0e-06 without it at L=128, so this bound separates the
    # two by a factor of five in the direction that matters.
    drift = (quat.double().square().sum(dim=-2).sqrt() - 1.0).abs().max()
    assert float(drift) < 5e-7, f"{tag}: quaternion prefix norm drifted {drift:.3e}"

    # The chunk endpoint is the last token's prefix, split into a unit rotation
    # and its own decay. The rotation carries the prefix bound directly, with no
    # scale mixed into it, which is the point of not packing the two together.
    assert_max_rel(end, ref.qprefix[..., -1, :], 2e-6, f"{tag}.cquat")

    # The decay carries the absolute error of a float32 log prefix, and exp
    # turns that into a relative error that grows with the prefix magnitude,
    # because exp(2*(lp + e)) is exp(2*lp)*(1 + 2*e). Underflow is graceful by
    # I1, so the regime where the decay is subnormal is checked for boundedness
    # rather than for agreement.
    want_scale = torch.exp(2.0 * ref.lprefix[..., -1])
    assert torch.isfinite(cscale).all()
    assert bool((cscale >= 0.0).all()), "a chunk decay went negative (I1)"
    assert bool((cscale <= 1.0).all()), "a chunk decay exceeded one (I1)"
    normal = want_scale > 1e-30
    if bool(normal.any()):
        reach = float(ref.lprefix[..., -1].abs().max())
        bound = 4e-6 + 4.0 * reach * float(torch.finfo(torch.float32).eps)
        assert_max_rel(cscale[normal], want_scale[normal], bound, f"{tag}.cscale")
    if bool((~normal).any()):
        floor = torch.finfo(torch.float32).smallest_normal
        assert bool(
            (cscale[~normal].double() <= 4.0 * want_scale[~normal] + floor).all()
        )


def test_padded_tail_is_identity() -> None:
    """A chunk that overhangs the sequence stages the scan identity.

    ``quat_exp(0)`` is the identity quaternion and a zero log scale is a unit
    decay, so the prefix is constant across the pad and the pad tap is zero. This
    is what makes the ragged tail need no separate code path.

    The prefix is asserted at the same bound as the prefix itself, not bitwise: a
    pad token in a later lane reaches the same product through a different
    association than the last valid token in an earlier lane. The pad tap
    matrices are asserted exactly zero, because a zero tap makes every entry a
    product with zero.
    """
    chunk = 64
    seqlen = 100
    inp = make_inputs(
        bsz=1,
        heads=1,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
    )
    lp, quat, table, _, _ = _run_probe(inp.trans, inp.K, chunk)
    tail = seqlen - chunk

    assert_max_rel(
        lp[0, 0, 1, tail:],
        lp[0, 0, 1, tail - 1].expand(chunk - tail),
        2e-6,
        "cute-device.pad.lprefix",
    )
    assert_max_rel(
        quat[0, 0, 1, :, tail:],
        quat[0, 0, 1, :, tail - 1, None].expand(4, chunk - tail),
        4e-6,
        "cute-device.pad.qprefix",
    )
    assert torch.count_nonzero(table[0, 0, 1, TABLE_AP, tail:]) == 0
    assert torch.count_nonzero(table[0, 0, 1, TABLE_AN, tail:]) == 0


@pytest.mark.parametrize(("bsz", "heads", "seqlen", "chunk"), SHAPES)
def test_fused_slot_matches_the_reference_column(
    bsz: int, heads: int, seqlen: int, chunk: int
) -> None:
    """``fused`` writes the reference's one-tap column into the first slot.

    The failure mode is the reindex: an off-by-one in the shift, a factor taken from
    the prefix instead of the step, or the term landing in the wrong slot. Compared
    against ``chunked_forward_fused``, which is the authority for that column, at the
    same bound the table itself is held to, since the arithmetic is one extra
    rotation and one extra tap matrix in float32.
    """
    inp = make_inputs(
        bsz=bsz,
        heads=heads,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
    )
    table = _run_probe(inp.trans, inp.K, chunk, fused=True)[2]
    # The column depends on ``trans`` and ``K`` alone, so the streaming operands are
    # left off rather than widened.
    want = chunked_forward_fused(
        inp.U.double(),
        inp.trans.double(),
        inp.K.double(),
        inp.B.double(),
        inp.C.double(),
        chunk,
    ).afuse

    assert_max_rel(
        table[:, :, :, TABLE_AFUSE].unflatten(-1, (3, 3)),
        want,
        6e-6,
        "cute-device.fused.afuse",
    )


def test_fused_column_boundaries_are_the_reindex_edges() -> None:
    """Token 0 takes no shifted term, and the pad slot of a ragged tail takes one.

    Two edges, one launch. ``Afuse_0 == Ap_0`` bitwise: the previous chunk's
    ``An_{L-1}`` is in the previous chunk's frame and arrives through the carried
    state, so injecting it moves ``y``. Slot ``valid`` is the opposite error: the
    reindex puts the last real token's ``An`` there, and zeroing it as a pad row
    leaves ``y`` at roundoff while moving ``state`` by O(1), so no y-only assertion
    can see it.
    """
    chunk = 64
    seqlen = 100
    inp = make_inputs(
        bsz=1,
        heads=1,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
    )
    plain = _run_probe(inp.trans, inp.K, chunk)[2]
    table = _run_probe(inp.trans, inp.K, chunk, fused=True)[2]
    tail = seqlen - chunk

    assert torch.equal(table[:, :, :, TABLE_AFUSE, 0], plain[:, :, :, TABLE_AP, 0])
    # The pad token's log scale stages as zero, so the factor is one and the slot is
    # the previous token's An exactly.
    assert torch.equal(
        table[0, 0, 1, TABLE_AFUSE, tail], plain[0, 0, 1, TABLE_AN, tail - 1]
    )
    assert torch.count_nonzero(table[0, 0, 1, TABLE_AFUSE, tail]) > 0
    assert torch.count_nonzero(table[0, 0, 1, TABLE_AFUSE, tail + 1 :]) == 0


def test_reduced_slot_tables_match_the_three_slot_matrices() -> None:
    """``mats=2`` and ``mats=1`` write the same matrices at fewer slots.

    ``Ac`` is an intermediate of both taps and the taps are not intermediates of
    ``Ac``, so neither reduction may change what it keeps. Asserted bitwise: the
    arithmetic is identical in each case, so anything but equality means a slot
    index moved. ``mats=1`` is the case where the constant changes, from
    ``TABLE_AC`` to ``TABLE_AC_SOLE``, and reading the old one there would land on
    a slot that does not exist.
    """
    inp = make_inputs(
        bsz=2,
        heads=2,
        seqlen=192,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
        ls_bias=LS_BIAS,
    )
    full = _run_probe(inp.trans, inp.K, 64, mats=3)[2]
    taps = _run_probe(inp.trans, inp.K, 64, mats=2)[2]
    sole = _run_probe(inp.trans, inp.K, 64, mats=1)[2]
    assert tuple(taps.shape[3:]) == (2, 64, 9)
    assert tuple(sole.shape[3:]) == (1, 64, 9)
    assert torch.equal(taps[:, :, :, TABLE_AP], full[:, :, :, TABLE_AP])
    assert torch.equal(taps[:, :, :, TABLE_AN], full[:, :, :, TABLE_AN])
    assert torch.equal(sole[:, :, :, TABLE_AC_SOLE], full[:, :, :, TABLE_AC])
    assert table_tile(64, 2).words == table_tile(64, 3).words - 9 * 64
    assert table_tile(64, 1).words == table_tile(64, 3).words - 18 * 64


def test_table_rejects_a_slot_count_with_no_consumer() -> None:
    """One, two and three slots are the only shapes any kernel reads."""
    for mats in (0, 4):
        with pytest.raises(ValueError, match="1, 2 or 3 matrices"):
            table_tile(64, mats)


def _probe_smem_bytes(chunk: int, mats: int = 3) -> int:
    """Bytes the probe's shared-memory tiles add up to.

    The same five tiles :func:`_probe_kernel` allocates, in the same order.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (_lp_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk, mats), 4),
        ]
    )


@pytest.mark.parametrize("chunk", [MIN_CHUNK, 32, 64, MAX_CHUNK])
def test_shared_memory_budget_fits_the_queried_capacity(chunk: int) -> None:
    """The budget is computed from the layouts, not from a guard constant.

    The 48 KiB default is not the ceiling; the opt-in capacity is queried from the
    device's own architecture. At ``MAX_CHUNK`` the probe's tiles are the widest
    the tree stages, so this is the binding case.
    """
    nbytes = _probe_smem_bytes(chunk)
    assert assert_smem_fits(f"probe[L{chunk}]", nbytes) == nbytes
    assert nbytes <= smem_capacity()
    # A carveout of at least 64 KiB is what makes three of these resident per SM.
    assert smem_capacity() >= 64 * 1024


def test_shared_memory_budget_over_capacity_is_refused() -> None:
    """No slop constant: either the layouts fit or the layouts change."""
    with pytest.raises(ValueError, match="shared memory"):
        assert_smem_fits("oversized", smem_capacity() + 1)


def test_vector_tile_covers_width_times_chunk() -> None:
    """The per-token vector tile is dense: one entry per component per token."""
    assert smem_bytes([(vec_tile(64, 3), 4)]) == 4 * 3 * 64
    assert smem_bytes([(vec_tile(128, 48), 2)]) == 2 * 48 * 128


def test_dtype_map_rejects_a_dtype_with_no_kernel_path() -> None:
    """float64 has no tensor-core path, so it is refused rather than downcast."""
    assert cute_dtype(torch.bfloat16) is cutlass.BFloat16
    assert cute_dtype(torch.float32) is cutlass.Float32
    with pytest.raises(TypeError, match="no CuTe kernel path"):
        cute_dtype(torch.float64)
