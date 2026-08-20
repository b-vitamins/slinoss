"""What a block width costs in shared memory and buys in occupancy.

Three tilings of one contraction, on one problem, in one process. The chain is the
chunk scan's: a score GEMM, the score narrowed and staged, a second GEMM against a
forcing operand, and a float32 slab round trip that stands in for the resident
state every scan kernel carries. Nothing here is a shipped kernel. No operator
launches it and none is meant to; it exists so the trade between the M tile and the
warp count is a number rather than an argument.

The arms:

- ``narrow``, :func:`slinoss.ops.so3ssd.cute.mma.make_mma` at
  :data:`slinoss.ops.so3ssd.cute.common.WARPS`. The shipped tiling.
- ``wide``, the same at :data:`slinoss.ops.so3ssd.cute.mma.WARPS_WIDE`. Twice the
  warps, the same tile, so every M-extent allocation is the same size.
- ``tall``, built here rather than taken from the library, because the library does
  not offer it: an atom layout of ``2 * WARPS`` in M, which is what a block width
  that partitions M further means. Twice the warps and twice the M mode.

``tall`` is the arm the other two are read against. It is on the device only while
its footprint fits; past that the driver prints its bytes and skips it, which is the
measurement.

Two modes, for the reason ``scripts/perf/profile_boundary_bwd.py`` has two: one
capture window and one warmup policy, in one file, so they cannot drift. The
default drives; ``--window`` is the process NCU attaches to. Every arm that fits
runs inside one window, and the three kernels have three names so the profiler
attributes the counters without a second invocation.

    python3 scripts/perf/profile_mma_warps.py --slabs 2
    python3 scripts/perf/profile_mma_warps.py --slabs 1 --ncu /usr/local/cuda-12.3/bin/ncu

``--slabs`` sets how many float32 M-extent slabs a block holds, and it is the knob
that decides which occupancy regime the arms land in: the M-extent bytes are linear
in it and the flat bytes are not.
"""

# No `from __future__ import annotations`: the DSL reads a decorated function's
# annotations at trace time and PEP 563 turns them into strings, so every Constexpr
# parameter would be classified as a runtime argument. This is the only driver in
# this directory that declares kernels, so it is the only one the rule reaches.

import argparse
import sys
from collections.abc import Callable, Sequence

import cutlass
import cutlass.cute as cute
import torch

from slinoss._cute import (
    Stream,
    Tile,
    cute_dtype,
    jit_launch,
    smem_bytes,
    smem_capacity,
)
from slinoss.ops.so3ssd.cute.common import THREADS, WARPS
from slinoss.ops.so3ssd.cute.mma import (
    MMA_INST,
    MMA_TILE_K,
    MMA_TILE_M,
    MMA_TILE_N,
    SMEM_SEGMENT,
    THREADS_WIDE,
    WARPS_WIDE,
    make_mma,
    mma_acc,
    mma_coords,
    mma_gemm,
    mma_store,
    operand_tile,
    smem_pitch,
)
from slinoss.perf.capture import profiler_window
from slinoss.perf.device import (
    compute_apps_query,
    device_ordinal,
    require_cuda,
    smi_selector,
)
from slinoss.perf.ncu import (
    NCU_TABLES,
    SPILL_TABLE,
    NcuPass,
    SpillCounters,
    kernel_counters,
    run_ncu,
    spill_counters,
)
from slinoss.perf.timing import Timed, measure, on_device

KERNEL = "mma_warps_"
"""Regex NCU narrows to. Matches all three arms; the names keep them apart."""

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

NARROW = "narrow"
WIDE = "wide"
TALL = "tall"

ARM_WARPS = {NARROW: WARPS, WIDE: WARPS_WIDE, TALL: WARPS_WIDE}
"""Warps per block, per arm."""

ARM_TILE_M = {NARROW: MMA_TILE_M, WIDE: MMA_TILE_M, TALL: 2 * MMA_TILE_M}
"""M mode of the tile, per arm. The one figure the arms exist to separate."""

ARMS = (NARROW, WIDE, TALL)


def arm_rows(arm: str, chunk: int) -> int:
    """Rows the shared operand must expose for an M extent of ``chunk``.

    :func:`slinoss.ops.so3ssd.cute.mma.mma_rows` rounds to the shipped
    :data:`slinoss.ops.so3ssd.cute.mma.MMA_TILE_M`, which is what two of the three
    arms have; ``tall`` needs its own, so the rounding is written once here against
    the arm's own tile.
    """
    tile_m = ARM_TILE_M[arm]
    return -(-chunk // tile_m) * tile_m


def slab_tile(slabs: int, mpad: int, rows: int) -> Tile:
    """``(slabs, mpad, smem_pitch(rows, 4))``, the float32 slabs a block holds.

    Stands in for the resident state and the accumulated cotangents a scan kernel
    keeps live across a chunk: float32 by I4, one row per output row, so it scales
    with the M tile and nothing else.
    """
    pitch = smem_pitch(rows, 4)
    return Tile((slabs, mpad, pitch), (mpad * pitch, pitch, 1))


def arm_tiles(
    arm: str, chunk: int, rows: int, dim: int, slabs: int
) -> tuple[tuple[tuple[Tile, int], ...], tuple[tuple[Tile, int], ...]]:
    """The block's allocations, split by whether the M tile sizes them.

    One description of each allocation: the kernel builds its layout from the same
    call, so the printed footprint is the allocated footprint.

    Args:
        arm: One of :data:`ARMS`.
        chunk: ``L``, the M extent of both GEMMs.
        rows: ``P``, the N extent of the second GEMM and the slab width.
        dim: ``3N``, the K extent of the first GEMM.
        slabs: Float32 M-extent slabs.

    Returns:
        ``(m_extent, flat)``, each a tuple of ``(tile, itemsize)`` pairs on the
        footing :func:`slinoss._cute.smem_bytes` takes.
    """
    mpad = arm_rows(arm, chunk)
    m_extent = (
        (operand_tile(mpad, dim), 2),
        (operand_tile(mpad, chunk), 2),
        (slab_tile(slabs, mpad, rows), 4),
    )
    flat = ((operand_tile(chunk, dim), 2), (operand_tile(chunk, rows), 2))
    return m_extent, flat


@cute.jit
def _stage(
    src: cute.Tensor,
    dst: cute.Tensor,
    tid: cutlass.Int32,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
    ld: cutlass.Constexpr,
    pad_rows: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    """Copy ``src(rows,cols)`` into ``dst(pad_rows,ld)``, zeroing the remainder.

    The padding participates in the MMA whenever it falls inside an operand view,
    so leaving it uninitialized admits whatever the allocator last held.
    """
    for i in cutlass.range(tid, pad_rows * ld, threads):
        r = i // ld
        c = i - r * ld
        if (r < rows) & (c < cols):
            dst[r, c] = src[r, c]
        else:
            dst[r, c] = dst.element_type(0.0)


@cute.jit
def _arm_body(
    gv: cute.Tensor,
    gu: cute.Tensor,
    gd: cute.Tensor,
    tiled_mma: cute.TiledMma,
    arm: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    folds: cutlass.Constexpr,
) -> None:
    """The chain, once per block, for any of the three tilings.

    One body behind three kernel names: an arm that differed in anything but its
    tiling would not be a measurement of the tiling.

    The second GEMM takes its left operand from shared memory rather than from
    :func:`slinoss.ops.so3ssd.cute.mma.mma_areg`, because the register retile does
    not survive an N split and the arms must run the same instruction sequence.

    Args:
        gv: ``(blocks, L, 3N)``, the score operand, both sides.
        gu: ``(blocks, L, P)``, the forcing operand.
        gd: ``(blocks, L, P)`` float32 output.
        tiled_mma: The arm's tiling.
        arm: One of :data:`ARMS`. Compile-time.
        chunk: ``L``. Compile-time.
        rows: ``P``. Compile-time.
        dim: ``3N``. Compile-time.
        slabs: Float32 M-extent slabs. Compile-time.
        folds: Times the chain runs per block. Compile-time.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    threads = 32 * ARM_WARPS[arm]
    mpad = arm_rows(arm, chunk)
    elem = gv.element_type
    ldv = smem_pitch(dim)
    ldu = smem_pitch(rows)
    lds = smem_pitch(chunk)

    smem = cutlass.utils.SmemAllocator()
    sa = smem.allocate_tensor(elem, operand_tile(mpad, dim).layout(), SMEM_SEGMENT)
    sscore = smem.allocate_tensor(
        elem, operand_tile(mpad, chunk).layout(), SMEM_SEGMENT
    )
    sslab = smem.allocate_tensor(
        cutlass.Float32, slab_tile(slabs, mpad, rows).layout(), SMEM_SEGMENT
    )
    sb = smem.allocate_tensor(elem, operand_tile(chunk, dim).layout(), SMEM_SEGMENT)
    su = smem.allocate_tensor(elem, operand_tile(chunk, rows).layout(), SMEM_SEGMENT)

    pv = gv[bid, None, None]
    pu = gu[bid, None, None]
    _stage(pv, sa, tid, chunk, dim, ldv, mpad, threads)
    _stage(pv, sb, tid, chunk, dim, ldv, chunk, threads)
    _stage(pu, su, tid, chunk, rows, ldu, chunk, threads)
    cute.arch.sync_threads()

    va = cute.make_tensor(sa.iterator, cute.make_layout((mpad, dim), stride=(ldv, 1)))
    vb = cute.make_tensor(sb.iterator, cute.make_layout((chunk, dim), stride=(ldv, 1)))
    vu = cute.make_tensor(su.iterator, cute.make_layout((rows, chunk), stride=(1, ldu)))
    vs = cute.make_tensor(
        sscore.iterator, cute.make_layout((mpad, chunk), stride=(lds, 1))
    )

    total = mma_acc(tiled_mma, tid, (mpad, rows))
    crd = mma_coords(tiled_mma, tid, (mpad, rows))
    scrd = mma_coords(tiled_mma, tid, (mpad, chunk))
    # Rolled, not unrolled: a trip count of one costs local traffic on this shape
    # and the fold count is only here to keep the launch out of the duration.
    for _ in cutlass.range(folds):
        sacc = mma_acc(tiled_mma, tid, (mpad, chunk))
        mma_gemm(tiled_mma, tid, sacc, va, vb, True, True)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(cute.size(sacc)):
            m, n = scrd[i]
            vs[m, n] = sacc[i].to(elem)
        cute.arch.sync_threads()
        acc = mma_acc(tiled_mma, tid, (mpad, rows))
        mma_gemm(tiled_mma, tid, acc, vs, vu, True, False)
        for s in cutlass.range_constexpr(slabs):
            for i in cutlass.range_constexpr(cute.size(acc)):
                m, n = crd[i]
                sslab[s, m, n] = acc[i]
        cute.arch.sync_threads()
        # Read the slabs back in the reverse order they were written, so every
        # store is consumed by a load the barrier orders after it and no slab is
        # dead.
        for s in cutlass.range_constexpr(slabs):
            for i in cutlass.range_constexpr(cute.size(acc)):
                m, n = crd[i]
                total[i] += sslab[slabs - 1 - s, m, n]
        cute.arch.sync_threads()
    mma_store(tiled_mma, tid, total, gd[bid, None, None], (mpad, rows), chunk)


@cute.kernel
def mma_warps_narrow_kernel(
    gv: cute.Tensor,
    gu: cute.Tensor,
    gd: cute.Tensor,
    tiled_mma: cute.TiledMma,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    folds: cutlass.Constexpr,
) -> None:
    """:data:`NARROW`: the shipped tiling at :data:`THREADS` threads."""
    _arm_body(gv, gu, gd, tiled_mma, NARROW, chunk, rows, dim, slabs, folds)


@cute.kernel
def mma_warps_wide_kernel(
    gv: cute.Tensor,
    gu: cute.Tensor,
    gd: cute.Tensor,
    tiled_mma: cute.TiledMma,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    folds: cutlass.Constexpr,
) -> None:
    """:data:`WIDE`: :data:`THREADS_WIDE` threads, the same M tile."""
    _arm_body(gv, gu, gd, tiled_mma, WIDE, chunk, rows, dim, slabs, folds)


@cute.kernel
def mma_warps_tall_kernel(
    gv: cute.Tensor,
    gu: cute.Tensor,
    gd: cute.Tensor,
    tiled_mma: cute.TiledMma,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    folds: cutlass.Constexpr,
) -> None:
    """:data:`TALL`: :data:`THREADS_WIDE` threads and twice the M tile."""
    _arm_body(gv, gu, gd, tiled_mma, TALL, chunk, rows, dim, slabs, folds)


@cute.jit
def _launch(
    gv: cute.Tensor,
    gu: cute.Tensor,
    gd: cute.Tensor,
    stream: Stream,
    dtype: cutlass.Constexpr,
    arm: cutlass.Constexpr,
    blocks: cutlass.Constexpr,
    chunk: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    dim: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    folds: cutlass.Constexpr,
) -> None:
    """Launch one arm.

    ``narrow`` and ``wide`` take their tiling from
    :func:`slinoss.ops.so3ssd.cute.mma.make_mma`, so what is measured is the shipped
    construction and not a copy of it. ``tall`` is built here because the library
    refuses it: :func:`slinoss.ops.so3ssd.cute.mma.mma_atoms` never puts a warp in
    the M mode past :data:`slinoss.ops.so3ssd.cute.common.WARPS`.
    """
    if cutlass.const_expr(arm == TALL):
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, MMA_INST),
            (2 * WARPS, 1, 1),
            permutation_mnk=(2 * MMA_TILE_M, MMA_TILE_N, MMA_TILE_K),
        )
    else:
        tiled_mma = make_mma(dtype, ARM_WARPS[arm])
    args = (gv, gu, gd, tiled_mma, chunk, rows, dim, slabs, folds)
    grid = (blocks, 1, 1)
    block = (32 * ARM_WARPS[arm], 1, 1)
    if cutlass.const_expr(arm == NARROW):
        mma_warps_narrow_kernel(*args).launch(grid=grid, block=block, stream=stream)
    elif cutlass.const_expr(arm == WIDE):
        mma_warps_wide_kernel(*args).launch(grid=grid, block=block, stream=stream)
    else:
        mma_warps_tall_kernel(*args).launch(grid=grid, block=block, stream=stream)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=64, help="L, the M extent.")
    parser.add_argument("--rows", type=int, default=64, help="P, the second N.")
    parser.add_argument("--dim", type=int, default=48, help="3N, the first K.")
    parser.add_argument(
        "--slabs",
        type=int,
        default=2,
        help="Float32 M-extent slabs a block holds. Sets which occupancy regime "
        "the arms land in.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=8,
        help="Times a block runs the chain. Keeps the launch out of the duration.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=672,
        help="Grid. Eight waves of one block per SM on an 84-SM part.",
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--event-iters", type=int, default=20)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument(
        "--window",
        action="store_true",
        help="Run as the profiler's target: warm up, then launch every arm that "
        "fits inside the capture window. Emits nothing.",
    )
    return parser.parse_args(argv)


def target_argv(args: argparse.Namespace) -> list[str]:
    """The argv NCU attaches to. Carries every knob forward."""
    return [
        sys.executable,
        __file__,
        "--window",
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--chunk",
        str(args.chunk),
        "--rows",
        str(args.rows),
        "--dim",
        str(args.dim),
        "--slabs",
        str(args.slabs),
        "--folds",
        str(args.folds),
        "--blocks",
        str(args.blocks),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]


def check_shape(args: argparse.Namespace) -> None:
    """Reject a shape the atom cannot take.

    Raises:
        ValueError: If an extent does not divide the tile, or a count is not
            positive. The tall arm's M tile is twice the others', and
            :func:`arm_rows` pads to it, so no M extent is rejected here.
    """
    if args.chunk % MMA_TILE_N or args.chunk % MMA_TILE_K:
        raise ValueError(f"--chunk {args.chunk} must divide {MMA_TILE_N} and K")
    if args.rows % MMA_TILE_N:
        raise ValueError(f"--rows {args.rows} must be a multiple of {MMA_TILE_N}")
    if args.dim % MMA_TILE_K:
        raise ValueError(f"--dim {args.dim} must be a multiple of {MMA_TILE_K}")
    for name in ("slabs", "folds", "blocks", "iters", "event_iters"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def fitting_arms(args: argparse.Namespace) -> tuple[str, ...]:
    """The arms whose footprint the device admits, in :data:`ARMS` order."""
    capacity = smem_capacity()
    out = []
    for arm in ARMS:
        m_extent, flat = arm_tiles(arm, args.chunk, args.rows, args.dim, args.slabs)
        if smem_bytes(m_extent) + smem_bytes(flat) <= capacity:
            out.append(arm)
    return tuple(out)


def build_inputs(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """One operand set, shared by every arm.

    The arms must see the same bits or their outputs cannot be compared.
    """
    gen = torch.Generator(device=device).manual_seed(1)

    def randn(*size: int) -> torch.Tensor:
        return torch.randn(*size, dtype=dtype, device=device, generator=gen)

    return (
        randn(args.blocks, args.chunk, args.dim),
        randn(args.blocks, args.chunk, args.rows),
    )


def run_arm(
    arm: str,
    args: argparse.Namespace,
    gv: torch.Tensor,
    gu: torch.Tensor,
    out: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Launch one arm into ``out``."""
    jit_launch(
        _launch,
        (gv, gu, out),
        (
            cute_dtype(dtype),
            arm,
            args.blocks,
            args.chunk,
            args.rows,
            args.dim,
            args.slabs,
            args.folds,
        ),
    )


def _runner(
    arm: str,
    args: argparse.Namespace,
    gv: torch.Tensor,
    gu: torch.Tensor,
    out: torch.Tensor,
    dtype: torch.dtype,
) -> Callable[[], None]:
    """Bind one arm's launch arguments.

    A named closure rather than a lambda in a loop: a lambda would capture the loop
    variable and every arm would time the last one.
    """

    def run() -> None:
        run_arm(arm, args, gv, gu, out, dtype)

    return run


def time_arms(
    arms: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Timed], dict[str, float], dict[str, bool]]:
    """Time every arm and compare their outputs.

    The operands and the outputs live and die inside this call, so the device is
    empty again before NCU spawns a target that allocates the same tensors.

    Args:
        arms: The arms to run, first one taken as the reference for the diff.
        args: The parsed command line.
        device: Where to allocate.
        dtype: Operand dtype.

    Returns:
        ``(timed, diffs, finite)``: the timing per arm, the max absolute
        difference of each later arm against the first, and whether each output is
        finite everywhere. The arms partition the same output over their warps and
        run the same K loop, so a nonzero difference means an arm's tiling maps an
        element somewhere the others do not.
    """
    gv, gu = build_inputs(args, device, dtype)
    outs = {
        arm: torch.zeros(
            args.blocks, args.chunk, args.rows, dtype=torch.float32, device=device
        )
        for arm in arms
    }
    timed = {
        arm: measure(
            _runner(arm, args, gv, gu, outs[arm], dtype),
            label=f"mma_warps {arm}",
            iters=args.event_iters,
            warmup=args.warmup,
            device=device,
        )
        for arm in arms
    }
    torch.cuda.synchronize()
    reference = outs[arms[0]]
    diffs = {
        arm: (outs[arm] - reference).abs().max().item()
        for arm in arms
        if arm != arms[0]
    }
    finite = {arm: bool(torch.isfinite(outs[arm]).all()) for arm in arms}
    return timed, diffs, finite


def footprint_lines(args: argparse.Namespace) -> list[str]:
    """The host-side byte figures, one line per arm.

    Exact arithmetic over the allocations the kernel makes, not a model: the tiles
    come from the same calls the kernel builds its layouts from.
    """
    capacity = smem_capacity()
    lines = []
    for arm in ARMS:
        m_extent, flat = arm_tiles(arm, args.chunk, args.rows, args.dim, args.slabs)
        m_bytes = smem_bytes(m_extent)
        flat_bytes = smem_bytes(flat)
        total = m_bytes + flat_bytes
        resident = total <= capacity
        blocks = capacity // total if resident else 0
        warps = ARM_WARPS[arm]
        lines.append(
            f"  {arm:<7} warps={warps:<2} tile_m={ARM_TILE_M[arm]:<4} "
            f"mpad={arm_rows(arm, args.chunk):<4} "
            f"m_extent={m_bytes:>7} B flat={flat_bytes:>6} B "
            f"total={total:>7} B "
            f"blocks<={blocks} warps<={blocks * warps} "
            f"{'' if resident else 'OVER CAPACITY'}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, profile, and print.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
        ValueError: If the shape is illegal, or a counter table returned no value
            for a metric, which means the name is wrong for this driver.
    """
    args = parse_args(argv)
    check_shape(args)
    device = require_cuda(args.device)
    dtype = DTYPES[args.dtype]
    arms = fitting_arms(args)
    if not arms:
        raise ValueError("no arm fits the carveout at this shape")

    if args.window:
        gv, gu = build_inputs(args, device, dtype)
        outs = {
            arm: torch.zeros(
                args.blocks, args.chunk, args.rows, dtype=torch.float32, device=device
            )
            for arm in arms
        }
        with on_device(device):
            for arm in arms:
                for _ in range(args.warmup):
                    run_arm(arm, args, gv, gu, outs[arm], dtype)
            with profiler_window(device):
                for arm in arms:
                    for _ in range(args.iters):
                        run_arm(arm, args, gv, gu, outs[arm], dtype)
        return 0

    ordinal = device_ordinal(device)
    before = compute_apps_query(smi_selector(ordinal))
    timed, diffs, finite = time_arms(arms, args, device, dtype)

    tables = (*NCU_TABLES, SPILL_TABLE)
    narrow = ("--kernel-name", f"regex:{KERNEL}")
    passes: list[NcuPass] = []
    spills: tuple[SpillCounters, ...] = ()
    for table in tables:
        one = run_ncu(table, target_argv(args), ncu=args.ncu, extra=narrow)
        if one.missing_metrics:
            raise ValueError(
                f"ncu table {table.name!r} returned no value for "
                f"{list(one.missing_metrics)}; the metric names are wrong for "
                f"this driver"
            )
        if table is SPILL_TABLE:
            spills = spill_counters(one)
        else:
            passes.append(one)
    after = compute_apps_query(smi_selector(ordinal))

    print(f"shape        L={args.chunk} P={args.rows} 3N={args.dim}")
    print(f"knobs        slabs={args.slabs} folds={args.folds} blocks={args.blocks}")
    print(f"dtype        {args.dtype}")
    print(f"capacity     {smem_capacity()} B/block")
    print(f"threads      narrow={THREADS} wide={THREADS_WIDE}")
    print(f"smi before   {before}")
    print(f"smi after    {after}")
    print("footprint")
    for line in footprint_lines(args):
        print(line)
    print(f"arms run     {' '.join(arms)}")
    for arm in arms:
        one = timed[arm]
        print(
            f"  {arm:<7} med={one.total.median_duration_us:.1f} us "
            f"min={one.total.min_duration_us:.1f} "
            f"max={one.total.max_duration_us:.1f} "
            f"resolution={one.total.resolution_pct:.2f}% "
            f"clocks={one.clocks} finite={finite[arm]}"
        )
    for arm, diff in diffs.items():
        print(f"  {arm:<7} max_abs_diff_vs_{arms[0]}={diff:.3e}")

    for counters in kernel_counters(passes):
        launches = counters.launch_count
        print(f"kernel       {counters.kernel}")
        print(f"  launches   {launches}")
        print(f"  us/launch  {counters.duration_us / launches:.1f}")
        print(f"  regs       {counters.register_per_thread_count}")
        print(f"  smem       {counters.smem_bytes} B/block")
        print(
            f"  blocks     {counters.block_count} x {counters.thread_per_block_count}"
        )
        print(
            f"  occ        theo={counters.theoretical_occupancy_pct:.1f}% "
            f"achieved={counters.achieved_occupancy_pct:.1f}%"
        )
        print(f"  issue      {counters.issue_active_pct:.2f}%")
        print(
            f"  stall      {counters.dominant_stall} "
            f"{counters.dominant_stall_pct:.1f}% "
            f"barrier={counters.stall_barrier_pct:.1f}% "
            f"short_scoreboard={counters.stall_short_scoreboard_pct:.1f}% "
            f"mio_throttle={counters.stall_mio_throttle_pct:.1f}%"
        )
        print(
            f"  conflicts  ld={counters.shared_load_conflict_count} "
            f"st={counters.shared_store_conflict_count} "
            f"per_wavefront={counters.conflict_per_wavefront_ratio:.4f}"
        )
        # Shared traffic and warp-instruction count per launch: an N-mode split
        # replicates the A operand across the groups, so both rise with the group
        # count even though the M tile does not.
        print(
            f"  shared     wavefronts={counters.wavefront_count / launches:.0f}"
            f"/launch inst={counters.inst_count / launches:.0f}/launch"
        )
        print(
            f"  sol        sm={counters.sm_pct:.1f}% mem={counters.memory_pct:.1f}% "
            f"l1tex={counters.l1tex_pct:.1f}% l2={counters.l2_pct:.1f}% "
            f"tensor={counters.tensor_pipe_pct:.1f}%"
        )
        print(f"  GB/s       {counters.achieved_gbs:.1f}")
    for record in spills:
        launches = record.launch_count
        print(f"local        {record.kernel}")
        print(f"  ld sectors {record.local_load_sector_count / launches:.0f}/launch")
        print(f"  st sectors {record.local_store_sector_count / launches:.0f}/launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
