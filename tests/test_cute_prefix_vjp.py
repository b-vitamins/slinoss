"""The two reverse chunk-local scans against the float64 adjoint of the reference.

``chunk_suffix`` and ``quat_suffix_vjp`` never reach global memory on the shipped
path, so one probe kernel stages a chunk, runs :func:`chunk_prefixes` to produce
the quaternion prefix the adjoint inverts, calls both reverse scans, and writes
their outputs. A fabricated prefix would test neither the composition the adjoint
inverts nor the renormalization its projection undoes.

Ground truth is float64 autograd through the reference forward, not
``quat_prefix_scan_vjp`` read as a formula: a closed form shares its derivation
with the kernel, so a derivation error would pass silently in both. The closed
form's own agreement with autograd is pinned in ``tests/test_backward.py`` and is
not restated here.

Inputs are built in float32, then upcast for the oracle, so both paths see the same
bits and every difference is float32 arithmetic. The cotangents are drawn rather
than produced by a forward: a cotangent is an input to the backward, not an
intermediate of the forward.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from slinoss.config import MAX_CHUNK, MIN_CHUNK
from slinoss.ops.so3ssd.cute.common import (
    THREADS,
    scalar_tile,
    tap_tile,
    trans_tile,
)
from slinoss.ops.so3ssd.cute.prefix import (
    chunk_prefixes,
    chunk_suffix,
    quat_suffix_vjp,
)
from slinoss.ops.so3ssd.cute.table import stage_chunk
from slinoss.ops.so3ssd.reference import quat_exp, quat_prefix_scan
from tests.conftest import assert_max_rel, make_inputs

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

# Batch, head, and chunk count do not interact with a chunk-local scan: every
# block runs one chunk and reads nothing outside it. They are fixed at two, which
# is enough to catch a block index dropped from an output address.
BSZ = 2
HEADS = 2
CHUNKS = 2


@cute.kernel
def _probe_kernel(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    osuf: cute.Tensor,
    oquat: cute.Tensor,
    odquat: cute.Tensor,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    cidx, bidx, hidx = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    strans = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    stap = smem.allocate_tensor(cutlass.Float32, tap_tile(chunk).layout(), 16)
    slp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    squat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdlp = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    ssuf = smem.allocate_tensor(cutlass.Float32, scalar_tile(chunk).layout(), 16)
    sdrot = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)
    sdquat = smem.allocate_tensor(cutlass.Float32, trans_tile(chunk).layout(), 16)

    t0 = cidx * chunk
    stage_chunk(
        gtrans[bidx, hidx, None, None],
        gtap[bidx, hidx, None, None, None],
        strans,
        stap,
        t0,
        cutlass.Int32(chunk),
        tid,
        threads,
        chunk,
    )
    # The probe runs whole chunks, so the cotangent staging needs no clamp. A
    # chunk that overhangs the sequence is staged by stage_chunk, whose pad path
    # tests/test_cute_device_math.py already pins.
    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            sdlp[token] = gdlp[bidx, hidx, t0 + token]
            for j in cutlass.range_constexpr(4):
                sdrot[j, token] = gdrot[bidx, hidx, t0 + token, j]
    cute.arch.sync_threads()

    chunk_prefixes(strans, slp, squat, tid, chunk)
    chunk_suffix(sdlp, ssuf, tid, chunk)
    cute.arch.sync_threads()

    # Separated by a barrier: the adjoint's store pass reads the prefix of the
    # previous token, which at a lane's first token another lane wrote.
    quat_suffix_vjp(squat, sdrot, sdquat, tid, chunk)
    cute.arch.sync_threads()

    for step in cutlass.range_constexpr((chunk + threads - 1) // threads):
        token = tid + step * threads
        if token < chunk:
            osuf[bidx, hidx, cidx, token] = ssuf[token]
            for j in cutlass.range_constexpr(4):
                oquat[bidx, hidx, cidx, j, token] = squat[j, token]
                odquat[bidx, hidx, cidx, j, token] = sdquat[j, token]


@cute.jit
def _probe_launch(
    gtrans: cute.Tensor,
    gtap: cute.Tensor,
    gdlp: cute.Tensor,
    gdrot: cute.Tensor,
    osuf: cute.Tensor,
    oquat: cute.Tensor,
    odquat: cute.Tensor,
    chunks: cutlass.Constexpr,
    bsz: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
) -> None:
    _probe_kernel(
        gtrans, gtap, gdlp, gdrot, osuf, oquat, odquat, threads, chunk
    ).launch(grid=(chunks, bsz, heads), block=(threads, 1, 1))


def _dev(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _run_probe(
    trans: torch.Tensor,
    tap: torch.Tensor,
    dlp: torch.Tensor,
    drot: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the probe and return ``(suffix, qprefix, dquat)``.

    ``suffix`` is ``(B,H,chunks,L)``; the other two are ``(B,H,chunks,4,L)``,
    component-major as the shared tiles hold them.
    """
    bsz, heads, seqlen, _ = trans.shape
    chunks = seqlen // chunk
    opts = {"device": trans.device, "dtype": torch.float32}
    osuf = torch.empty(bsz, heads, chunks, chunk, **opts)
    oquat = torch.empty(bsz, heads, chunks, 4, chunk, **opts)
    odquat = torch.empty(bsz, heads, chunks, 4, chunk, **opts)
    _probe_launch(
        _dev(trans),
        _dev(tap),
        _dev(dlp),
        _dev(drot),
        _dev(osuf),
        _dev(oquat),
        _dev(odquat),
        chunks,
        bsz,
        heads,
        THREADS,
        chunk,
    )
    torch.cuda.synchronize()
    return osuf, oquat, odquat


def _cotangents(seqlen: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A log-scale cotangent ``(B,H,T)`` and a prefix cotangent ``(B,H,T,4)``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (BSZ, HEADS, seqlen)
    return (
        torch.randn(*shape, generator=gen, dtype=torch.float32, device="cuda"),
        torch.randn(*shape, 4, generator=gen, dtype=torch.float32, device="cuda"),
    )


def _reverse_cumsum(value: torch.Tensor, chunk: int) -> torch.Tensor:
    """Reverse inclusive sum within each chunk of a ``(B,H,T)`` tensor, in float64."""
    return value.double().unflatten(-1, (-1, chunk)).flip(-1).cumsum(-1).flip(-1)


def _autograd_dquat(
    trans: torch.Tensor, drot: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Per-step quaternion cotangent, by float64 autograd through the reference.

    ``quat_prefix_scan`` renormalizes, so its adjoint carries the projection the
    kernel's first step applies. Returns ``(B,H,chunks,4,L)``.
    """
    w = trans[..., :3].double().unflatten(-2, (-1, chunk))
    quat = quat_exp(w).requires_grad_(True)
    prefix = quat_prefix_scan(quat)
    (dquat,) = torch.autograd.grad(
        prefix, quat, drot.double().unflatten(-2, (-1, chunk))
    )
    return dquat.movedim(-1, -2)


# The reverse scan's structure is set by the segment count ``ceil(L/32)``, which is
# its serial depth, and by whether ``L`` is a warp multiple, which selects the
# clamp branch. One case per reachable pair. ``L = 45`` also puts the ragged tail
# inside a lane's segment rather than only in whole idle lanes: lane 22 owns tokens
# 44 and 45, of which 45 does not exist, and that slot is the first one the reverse
# pass visits.
CHUNK_CASES = [
    pytest.param(MIN_CHUNK, id="one-segment-under-warp"),
    pytest.param(32, id="one-segment-exact"),
    pytest.param(45, id="two-segments-ragged-lane"),
    pytest.param(64, id="two-segments-exact"),
    pytest.param(MAX_CHUNK, id="four-segments-exact"),
]


@pytest.mark.parametrize("chunk", CHUNK_CASES)
def test_reverse_scans_match_the_float64_adjoint(chunk: int) -> None:
    """Both reverse scans, against float64 autograd through the reference."""
    seqlen = CHUNKS * chunk
    inp = make_inputs(
        bsz=BSZ,
        heads=HEADS,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
    )
    dlp, drot = _cotangents(seqlen, seed=chunk)
    suffix, _, dquat = _run_probe(inp.trans, inp.K, dlp, drot, chunk)

    tag = f"cute-suffix[L{chunk}]"
    # Both sides add the same float32 values; the segmented scan reassociates at
    # most MAX_CHUNK terms of order one. Measured 1.13e-07 at L = 128.
    assert_max_rel(suffix, _reverse_cumsum(dlp, chunk), 5e-7, f"{tag}.chunk_suffix")
    # The adjoint reads the float32 quaternion prefix, multiplies it in twice and
    # sums up to MAX_CHUNK terms of it. The prefix's own error is a rotation shared
    # by ``Q_l``, ``conj(Q_m)`` and ``conj(Q_{l-1})``, so most of it cancels and the
    # gap stays near the reassociation of the sum. Measured 5.12e-07 at L = 64.
    assert_max_rel(
        dquat, _autograd_dquat(inp.trans, drot, chunk), 2e-6, f"{tag}.quat_suffix_vjp"
    )


def test_a_radial_cotangent_leaves_no_gradient() -> None:
    """The projection of step one, pinned on its own.

    ``dQ_t = c_t Q_t`` is radial at the prefix the kernel itself produced, and the
    renormalized prefix does not move along its own radius, so the exact gradient
    is zero. Skipping the projection instead returns
    ``S_l = sum_{m>=l} c_m conj(Q_m) (*) Q_m = sum_{m>=l} c_m`` and therefore
    ``dq_l = (sum_{m>=l} c_m) q_l``, which is order one because ``|q_l| = 1``. The
    two answers are separated by the whole dynamic range, so without this case a
    later deletion of the projection would leave every other test passing.

    One shape: the projection is per token and has no interaction with the
    segmentation the sweep above covers.
    """
    chunk = 64
    seqlen = CHUNKS * chunk
    inp = make_inputs(
        bsz=BSZ,
        heads=HEADS,
        seqlen=seqlen,
        rows=16,
        lanes=16,
        dtype=torch.float32,
        device="cuda",
        w_scale=2.0,
    )
    dlp, drot = _cotangents(seqlen, seed=1)
    _, qprefix, _ = _run_probe(inp.trans, inp.K, dlp, drot, chunk)

    # One arbitrary scalar per token times the kernel's own prefix. The log-scale
    # draw serves as that scalar; a second draw would test nothing further.
    coeff = dlp
    radial = (
        (coeff.unflatten(-1, (-1, chunk)).unsqueeze(-1) * qprefix.movedim(-1, -2))
        .flatten(2, 3)
        .contiguous()
    )
    _, _, dquat = _run_probe(inp.trans, inp.K, dlp, radial, chunk)

    # The surrogate's magnitude is read off the coefficients, so this states what
    # the assertion below discriminates against rather than assuming it. Measured
    # 1.43e+01 against the 1.69e-06 below: seven orders of separation.
    surrogate = float(_reverse_cumsum(coeff, chunk).abs().max())
    assert surrogate > 0.5, f"the skipped-projection answer is only {surrogate:.3e}"
    # Absolute, because the exact answer is zero. What is left is the surrogate
    # times half the prefix's residual norm drift: ``|Q|^2 - 1`` measures 2.7e-07
    # after the renormalization of I5, so the radial direction is itself only
    # radial to that accuracy. Measured 1.69e-06.
    worst = float(dquat.abs().max())
    assert worst < 5e-6, f"a radial cotangent left a gradient of {worst:.3e}"
