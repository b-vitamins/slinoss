"""Device-side transition math against the float64 reference.

One probe kernel stages a chunk, runs both chunk-local prefixes, and composes the
3x3 table, then writes every intermediate out. That covers, in one launch, the
quaternion exponential series, the composition order, the warp prefix scans, the
renormalization, the homogeneous rotation matrix, the tap chart, the 3x3 product
and transpose, the ragged-tail staging, and the packed chunk endpoint.

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
from cutlass.cute.runtime import from_dlpack

from slinoss._cute import (
    Tile,
    assert_smem_fits,
    cute_dtype,
    smem_bytes,
    smem_capacity,
)
from slinoss.config import MAX_CHUNK, MIN_CHUNK
from slinoss.ops.so3ssd import chunked_forward
from slinoss.ops.so3ssd.cute.common import (
    THREADS,
    table_tile,
    tap_tile,
    trans_tile,
    vec_tile,
)
from slinoss.ops.so3ssd.cute.prefix import (
    chunk_prefixes,
    quat_prefix_endpoint,
)
from slinoss.ops.so3ssd.cute.table import build_table, stage_chunk
from tests.conftest import assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]


def _lp_tile(chunk: int) -> Tile:
    """The scalar log-prefix tile. Dense, one entry per token."""
    return Tile((chunk,), (1,))


# scanprep maps the raw log scale through a negative softplus, so a negative bias
# is a weak decay. Without it the prefix at the end of a 128-token chunk reaches
# exp(lp) near 2e-45, the packed chunk endpoint underflows to zero, and the
# assertion on it is skipped as subnormal rather than checked.
LS_BIAS = -4.0


@cute.kernel
def _probe_kernel(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    olp: cute.Tensor,
    oquat: cute.Tensor,
    otable: cute.Tensor,
    oend: cute.Tensor,
    seqlen: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, _lp_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stable = smem.allocate_tensor(cutlass.Float32, table_tile(chunk).layout(), 16)

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
    build_table(strans, stap, squat, stable, tid, threads, chunk)
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            olp[bidx, hidx, cidx, token] = slp[token]
            for j in cutlass.range_constexpr(4):
                oquat[bidx, hidx, cidx, j, token] = squat[j, token]
            for mat in cutlass.range_constexpr(3):
                for entry in cutlass.range_constexpr(9):
                    otable[bidx, hidx, cidx, mat, token, entry] = stable[
                        mat, token, entry
                    ]

    if tid == 0:
        endpoint = quat_prefix_endpoint(squat, slp, chunk)
        for j in cutlass.range_constexpr(4):
            oend[bidx, hidx, cidx, j] = endpoint[j]


@cute.jit
def _probe_launch(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    olp: cute.Tensor,
    oquat: cute.Tensor,
    otable: cute.Tensor,
    oend: cute.Tensor,
    seqlen: cutlass.Constexpr,
    chunks: cutlass.Constexpr,
    bsz: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    _probe_kernel(
        gtrans, gtap, olp, oquat, otable, oend, seqlen, threads, chunk
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


def _dev(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _run_probe(
    trans: torch.Tensor, tap: torch.Tensor, chunk: int
) -> tuple[torch.Tensor, ...]:
    """Launch the probe and return ``(lp, quat, table, endpoint)``."""
    bsz, heads, seqlen, _ = trans.shape
    chunks = (seqlen + chunk - 1) // chunk
    opts = {"device": trans.device, "dtype": torch.float32}
    olp = torch.empty(bsz, heads, chunks, chunk, **opts)
    oquat = torch.empty(bsz, heads, chunks, 4, chunk, **opts)
    otable = torch.empty(bsz, heads, chunks, 3, chunk, 9, **opts)
    oend = torch.empty(bsz, heads, chunks, 4, **opts)
    _probe_launch(
        _dev(trans),
        _dev(tap),
        _dev(olp),
        _dev(oquat),
        _dev(otable),
        _dev(oend),
        seqlen,
        chunks,
        bsz,
        heads,
        THREADS,
        chunk,
    )
    torch.cuda.synchronize()
    return olp, oquat, otable, oend


SHAPES = [
    pytest.param(2, 3, 256, 64, id="exact-4-chunks"),
    pytest.param(2, 3, 200, 64, id="ragged-tail"),
    pytest.param(1, 1, 128, MAX_CHUNK, id="single-chunk-seg4"),
    pytest.param(1, 2, 384, MAX_CHUNK, id="three-chunks-seg4"),
    pytest.param(2, 2, 96, 48, id="chunk-not-warp-multiple"),
    pytest.param(1, 2, 48, MIN_CHUNK, id="chunk-under-warp"),
    pytest.param(2, 2, 64, 32, id="chunk-one-per-lane"),
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
    lp, quat, table, end = _run_probe(inp.trans, inp.K, chunk)
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
        table[:, :, :, 0], ref.table.ac.flatten(-2, -1), 4e-6, f"{tag}.table.ac"
    )
    assert_max_rel(
        table[:, :, :, 1], ref.table.ap.flatten(-2, -1), 4e-6, f"{tag}.table.ap"
    )
    assert_max_rel(
        table[:, :, :, 2], ref.table.an.flatten(-2, -1), 4e-6, f"{tag}.table.an"
    )

    # I5 is a property of the prefix, not of its distance from the reference: a
    # comparison against float64 sees drift only after the rotation matrix has
    # squared it, and by then it is inside the bound above. The norm is what the
    # projection controls, so the norm is what is asserted. Measured 1.8e-07 with
    # the projection and 2.0e-06 without it at L=128, so this bound separates the
    # two by a factor of five in the direction that matters.
    drift = (quat.double().square().sum(dim=-2).sqrt() - 1.0).abs().max()
    assert float(drift) < 5e-7, f"{tag}: quaternion prefix norm drifted {drift:.3e}"

    # The packed chunk endpoint must reproduce the full chunk transition, scale
    # included, through the degree-two homogeneity of `rot_hom`.
    #
    # Packing the scale inside the quaternion halves the exponent it has to
    # carry: the stored value is exp(lp) and the transition it produces is
    # exp(2*lp). Whatever underflows in the packed form therefore corresponds to
    # a transition that is already exactly zero in float32, so the two regimes
    # are asserted separately. Direction and magnitude are checked together where
    # the magnitude is normal, and gracefulness is checked where it is not.
    scale = torch.exp(ref.lprefix[..., -1])
    want_end = ref.qprefix[..., -1, :] * scale[..., None]
    assert torch.isfinite(end).all()
    normal = scale > 1e-30
    if bool(normal.any()):
        # Dividing by the reference scale puts both sides at unit magnitude, so
        # one relative bound is meaningful. The direction carries the prefix
        # bound; the scale carries the absolute error of a float32 log prefix,
        # which grows with the prefix magnitude, because exp(lp + e) is
        # exp(lp)*(1 + e).
        reach = float(ref.lprefix[..., -1].abs().max())
        bound = 4e-6 + 2.0 * reach * float(torch.finfo(torch.float32).eps)
        assert_max_rel(
            end[normal].double() / scale[normal][:, None],
            ref.qprefix[..., -1, :][normal],
            bound,
            f"{tag}.endpoint",
        )
    if bool((~normal).any()):
        floor = torch.finfo(torch.float32).smallest_normal
        assert bool(
            (end[~normal].abs().double() <= 4.0 * want_end[~normal].abs() + floor).all()
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
    lp, quat, table, _ = _run_probe(inp.trans, inp.K, chunk)
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
    assert torch.count_nonzero(table[0, 0, 1, 1:, tail:]) == 0


def _probe_smem_bytes(chunk: int) -> int:
    """Bytes the probe's shared-memory tiles add up to.

    The same five tiles :func:`_probe_kernel` allocates, in the same order.
    """
    return smem_bytes(
        [
            (trans_tile(chunk), 4),
            (tap_tile(chunk), 4),
            (_lp_tile(chunk), 4),
            (trans_tile(chunk), 4),
            (table_tile(chunk), 4),
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
