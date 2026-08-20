"""Measured ceilings. Nothing here comes from a product page.

A ceiling divides a measurement, so a modelled ceiling makes every ratio built on
it modelled too, and the fact is invisible by the time it reaches a table. Both
ceilings in this module are measured in the same process, on the same device, at
the same clocks as the kernel under test, which is the only way an unlocked clock
does not corrupt the ratio: numerator and denominator drift together.

Two ceilings, matching the two kernel classes:

- DRAM: a large device-to-device copy, counting a read and a write per byte.
- Tensor: a large square GEMM in the operand dtype, counting ``2*M*N*K`` flop.

The bars every kernel is held to live here beside the ceilings that divide them,
and two of them divide nothing. :data:`MIN_OCCUPANCY_PCT` and the block floor on
:attr:`slinoss.perf.device.DeviceInfo.block_floor_count` are launch-geometry rules:
they are read off the launch configuration and the warp census rather than against
a probe, and :class:`GeometryVerdict` carries them. A kernel passes on its class
floor and fails on its geometry independently, which is the case
``chunk_vector_bwd`` makes: one resident block hides no latency at any traffic
figure, at any block width.

A kernel may read slightly above the copy ceiling if its read/write mix is
friendlier than a copy's. That is a fact about the probe, not a licence to invent
a higher number, and it is left visible rather than clamped.

Both ceilings are taken from the fastest sample, not the median. A ceiling is a
property of the hardware, so the estimator wanted is the sample least perturbed
by anything else on the device, and that is the fastest one. The median is the
wrong estimator here and fails in the one direction that matters: it absorbs
foreign load into the denominator, which inflates every ratio built on it and
turns a slow kernel into a passing one. Taking the minimum inverts that. Foreign
load then depresses the kernel numerator while leaving the denominator right, so
a contended capture reports a kernel as slower than it is and a verdict fails
conservatively rather than passing spuriously.

The dispersion is still reported beside every ceiling, because it is what says
how much foreign load the probe saw. ``spread_pct`` is not a trust bar on the
minimum: intermittent load raises it while leaving a clean fastest sample, and
uniform load lowers it while corrupting every sample. Contention is read from
``device.sharing``, which is a direct probe, not inferred from dispersion.

A rate at one footprint is not a denominator for a kernel at another. A copy
carries a fixed cost, so its rate rises with its footprint, and a kernel moving a
fraction of what the copy moved is charged for a fixed cost the copy amortized
away. On an A6000 a 25 MB copy read 599 and 614 GB/s over two runs where an
805 MB copy read 681 and 682. :class:`DramTimeFloor` sweeps the copy over several
footprints and fits ``t = c + bytes/B`` to the fastest sample at each, and
:func:`dram_floor_verdict` divides the floor at the kernel's own traffic by the
kernel's own duration. Both fitted terms and every sample are reported, so the
fit is checked rather than trusted.

The sweep is sized from the queried L2 and every point is at least two L2
capacities per buffer. A copy whose buffers fit in L2 measures L2 and reads as a
DRAM rate, so the small-footprint floor comes from the fit and never from a small
copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Final

import torch

from slinoss.perf.device import DeviceInfo, device_info, device_ordinal
from slinoss.perf.timing import measure
from slinoss.perf.units import (
    INVARIANT,
    MEDIAN,
    Bytes,
    Count,
    GBPerSecond,
    Microseconds,
    Percent,
    PerfRecord,
    Ratio,
    Spread,
    TFlopsPerSecond,
    gbs_from_bytes_us,
    mib_from_bytes,
    pct_of,
    ratio_of,
    tflops_from_flop_us,
)

__all__ = [
    "BLOCK_FLOOR_EXEMPT_CLASSES",
    "CLASS_FLOOR_PCT",
    "DRAM_BOUND",
    "L2_MULTIPLES",
    "MIN_OCCUPANCY_PCT",
    "SERIAL_TINY",
    "TENSOR_BOUND",
    "Ceilings",
    "ClassVerdict",
    "CopySample",
    "DramCeiling",
    "DramTimeFloor",
    "GeometryVerdict",
    "TensorCeiling",
    "ceilings",
    "dram_ceiling",
    "dram_floor_verdict",
    "dram_time_floor",
    "geometry_verdict",
    "serial_verdict",
    "tensor_ceiling",
    "tensor_verdict",
]

DRAM_BOUND = "DRAM-bound"
TENSOR_BOUND = "TENSOR-bound"
SERIAL_TINY = "SERIAL-tiny"

CLASS_FLOOR_PCT: dict[str, Percent] = {
    DRAM_BOUND: Percent(85.0),
    TENSOR_BOUND: Percent(70.0),
    SERIAL_TINY: Percent(2.0),
}
"""What each class must clear. SERIAL-tiny is a ceiling on step share, not a floor."""

MIN_OCCUPANCY_PCT: Final[Percent] = Percent(50.0)
"""Achieved occupancy no kernel may run below, whatever its class.

Occupancy is resident warps against warp slots, so it is what says how much
latency the multiprocessor has left to hide. It is not a share of a ceiling and so
does not belong in :data:`CLASS_FLOOR_PCT`: the two rules are independent, and a
kernel far under this bar can still read high against a bandwidth. That is the
reading this bar exists to catch. ``chunk_vector_bwd`` on sm_86 runs one resident
block at 8.3% achieved occupancy at four warps and 16.6% at eight, and its measured
share of its own DRAM floor sat between 12 and 31% across three shapes, both spill
states and both widths, so no traffic figure separates it from a kernel that merely
moves more bytes. Its arena is above the two-block ceiling at either width, so the
50% bar is unreachable there until the arena falls.
"""

BLOCK_FLOOR_EXEMPT_CLASSES: Final[frozenset[str]] = frozenset((SERIAL_TINY,))
"""Classes the block-count floor does not apply to.

A kernel provably serial in its own extent has no blocks to add, and its bar is
already a share of the step rather than a rate. Nothing exempts a kernel from the
occupancy rule: the block count is a property of the shape the kernel was handed,
and occupancy is a property of what the kernel does with a multiprocessor.
"""

_MIB = 1 << 20

_ITERS: Final = 30
"""Timed iterations for every probe in this module."""

_WARMUP: Final = 5
"""Untimed iterations for every probe in this module."""


@dataclass(frozen=True)
class DramCeiling(PerfRecord):
    """Achievable device memory bandwidth, measured by a large copy.

    Attributes:
        label: Probe description.
        moved_bytes: Bytes crossing DRAM per iteration, read plus write.
        duration: Per-iteration dispersion of the copy.
        achieved_gbs: ``moved_bytes`` over the fastest duration.
    """

    label: str
    moved_bytes: Annotated[Bytes, INVARIANT]
    duration: Spread
    achieved_gbs: Annotated[GBPerSecond, MEDIAN]


@dataclass(frozen=True)
class TensorCeiling(PerfRecord):
    """Achievable tensor-core throughput, measured by a large GEMM.

    Attributes:
        label: Probe description, including the operand dtype.
        flop_count: Floating-point operations per iteration, ``2*M*N*K``.
        duration: Per-iteration dispersion of the GEMM.
        achieved_tflops: ``flop_count`` over the fastest duration.
    """

    label: str
    flop_count: Annotated[Count, INVARIANT]
    duration: Spread
    achieved_tflops: Annotated[TFlopsPerSecond, MEDIAN]


@dataclass(frozen=True)
class Ceilings(PerfRecord):
    """Both ceilings and the device they were measured on.

    Attributes:
        device: Queried device identity and clock state.
        dram: The copy ceiling.
        tensor: The GEMM ceiling.
    """

    device: DeviceInfo
    dram: DramCeiling
    tensor: TensorCeiling


def _buffer_bytes(device: torch.device, requested: int) -> int:
    free, _total = torch.cuda.mem_get_info(device)
    usable = int(free) // 4
    return max(_MIB, min(requested, usable) // _MIB * _MIB)


def dram_ceiling(
    device: torch.device,
    *,
    requested_bytes: int = 512 * _MIB,
    iters: int = _ITERS,
    warmup: int = _WARMUP,
) -> DramCeiling:
    """Measure achievable bandwidth with a device-to-device copy.

    The buffer is sized from free memory so the probe does not evict the workload
    it is a ceiling for, and it is far larger than L2 so the copy is a DRAM copy.

    Args:
        device: CUDA device.
        requested_bytes: Preferred size per buffer. Clamped to a quarter of free
            memory.
        iters: Timed iterations.
        warmup: Untimed iterations.

    Returns:
        The ceiling.

    Raises:
        RuntimeError: If the device is not CUDA.
    """
    if device.type != "cuda":
        raise RuntimeError("dram_ceiling needs a CUDA device")
    size = _buffer_bytes(device, requested_bytes)
    src = torch.empty(size, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    src.random_(0, 255)
    timed = measure(
        lambda: dst.copy_(src),
        label="dram copy",
        iters=iters,
        warmup=warmup,
        device=device,
    )
    moved = Bytes(2 * size)
    return DramCeiling(
        label=f"device-to-device copy, {size // _MIB} MiB per buffer",
        moved_bytes=moved,
        duration=timed.total,
        achieved_gbs=gbs_from_bytes_us(moved, timed.total.min_duration_us),
    )


def tensor_ceiling(
    device: torch.device,
    *,
    dim: int = 8192,
    dtype: torch.dtype = torch.bfloat16,
    iters: int = _ITERS,
    warmup: int = _WARMUP,
) -> TensorCeiling:
    """Measure achievable tensor-core throughput with a square GEMM.

    Args:
        device: CUDA device.
        dim: Square GEMM dimension.
        dtype: Operand dtype. The ceiling is dtype-specific and the label says so.
        iters: Timed iterations.
        warmup: Untimed iterations.

    Returns:
        The ceiling.

    Raises:
        RuntimeError: If the device is not CUDA.
    """
    if device.type != "cuda":
        raise RuntimeError("tensor_ceiling needs a CUDA device")
    lhs = torch.randn(dim, dim, dtype=dtype, device=device)
    rhs = torch.randn(dim, dim, dtype=dtype, device=device)
    out = torch.empty(dim, dim, dtype=dtype, device=device)
    timed = measure(
        lambda: torch.mm(lhs, rhs, out=out),
        label="gemm",
        iters=iters,
        warmup=warmup,
        device=device,
    )
    flop = Count(2 * dim * dim * dim)
    return TensorCeiling(
        label=f"{dim}x{dim}x{dim} {dtype} gemm",
        flop_count=flop,
        duration=timed.total,
        achieved_tflops=tflops_from_flop_us(flop, timed.total.min_duration_us),
    )


def ceilings(
    device: torch.device, *, iters: int = _ITERS, warmup: int = _WARMUP
) -> Ceilings:
    """Measure both ceilings and read the device identity.

    Args:
        device: CUDA device.
        iters: Timed iterations, applied to both probes.
        warmup: Untimed iterations, applied to both probes.

    Returns:
        The pair, with the device record.
    """
    return Ceilings(
        device=device_info(device_ordinal(device)),
        dram=dram_ceiling(device, iters=iters, warmup=warmup),
        tensor=tensor_ceiling(device, iters=iters, warmup=warmup),
    )


L2_MULTIPLES: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64)
"""Sweep footprints, as multiples of the queried L2 size, per buffer.

Derived from the device rather than written in mebibytes, so the smallest point
stays above L2 on any part and no architecture is named. A thirty-twofold span is
what pins the two fitted terms apart: a narrow sweep trades them off, and the
intercept is the term the small-footprint floor rests on.
"""


@dataclass(frozen=True)
class CopySample(PerfRecord):
    """One device-to-device copy, at one footprint.

    Attributes:
        moved_bytes: Bytes crossing DRAM per iteration, read plus write.
        duration: Per-iteration dispersion of the copy at this footprint.
        achieved_gbs: ``moved_bytes`` over the fastest duration.
        l2_multiple_ratio: Bytes per buffer over the queried L2 size. Above one by
            construction; see :func:`dram_time_floor`.
    """

    moved_bytes: Annotated[Bytes, INVARIANT]
    duration: Spread
    achieved_gbs: Annotated[GBPerSecond, MEDIAN]
    l2_multiple_ratio: Annotated[Ratio, MEDIAN]


@dataclass(frozen=True)
class DramTimeFloor(PerfRecord):
    """Fastest a copy of a given size can run, as a fitted time law.

    ``t = c + bytes/B`` over the sweep, by least squares on the fastest sample at
    each footprint. Both terms are measurements: ``c`` is the part of a copy's
    duration that does not scale with its size, ``B`` the rate the copy approaches
    once it does. Neither comes from a product page, and every sample they were
    fitted to travels with them.

    A fit is only a floor if it holds at the footprint it is used at.
    ``max_residual_pct`` is what says whether it does, and it is reported rather
    than gated: a sweep that does not fit a line is a fact about the device, and
    hiding it behind a threshold would leave the verdict resting on it anyway.

    The fit is far better pinned at the top of the sweep than below it. Over ten
    takes on an A6000 the fitted rate at the largest swept footprint agreed with a
    single-point copy measured there to within 0.11%, while the floor the same fits
    extrapolated to a 10 MB footprint ranged over 443 to 531 GB/s. The residual is
    what separates those takes: 0.16% on the tightest, 14.5% on the loosest. A
    small-footprint verdict is worth no more than the residual beside it.

    A negative ``fixed_duration_us`` is possible on a noisy sweep. It lowers the
    floor, which tightens every verdict built on it, so it is left visible rather
    than clamped; :meth:`floor_us` refuses only a floor that is not positive.

    Attributes:
        label: Probe description.
        l2_bytes: Queried L2 size, so the sweep's L2 multiples can be rechecked.
        fixed_duration_us: Fitted ``c``, the size-independent term.
        asymptotic_gbs: Fitted ``B``, the rate the law approaches.
        max_residual_pct: Largest fitted-against-measured duration disagreement
            over the samples, as a percentage of the measured duration.
        copies: Every sample, in ascending footprint order.
    """

    label: str
    l2_bytes: Annotated[Bytes, INVARIANT]
    fixed_duration_us: Annotated[Microseconds, MEDIAN]
    asymptotic_gbs: Annotated[GBPerSecond, MEDIAN]
    max_residual_pct: Annotated[Percent, MEDIAN]
    copies: tuple[CopySample, ...]

    @classmethod
    def of(
        cls, label: str, copies: Sequence[CopySample], *, l2_bytes: Bytes
    ) -> DramTimeFloor:
        """Fit the time law to measured copies.

        Args:
            label: Probe description.
            copies: Samples at two or more distinct footprints.
            l2_bytes: Queried L2 size.

        Returns:
            The fitted floor.

        Raises:
            ValueError: If fewer than two distinct footprints were measured, which
                leaves the two terms indistinguishable, or if the fitted slope is
                not positive, which means the sweep measured no bandwidth at all.
        """
        sizes = [float(one.moved_bytes) for one in copies]
        times = [float(one.duration.min_duration_us) for one in copies]
        if len(set(sizes)) < 2:
            raise ValueError(
                f"a time law needs two distinct footprints, got {len(set(sizes))}; "
                f"one footprint fits infinitely many lines"
            )
        count = len(sizes)
        mean_size = sum(sizes) / count
        mean_time = sum(times) / count
        span = sum((size - mean_size) ** 2 for size in sizes)
        covariance = sum(
            (size - mean_size) * (time - mean_time)
            for size, time in zip(sizes, times, strict=True)
        )
        slope = covariance / span
        if slope <= 0.0:
            raise ValueError(
                f"fitted slope {slope} is not positive; a copy that does not take "
                f"longer as it grows is not measuring bandwidth"
            )
        intercept = mean_time - slope * mean_size
        residual = max(
            pct_of(abs(intercept + slope * size - time), time)
            for size, time in zip(sizes, times, strict=True)
        )
        return cls(
            label=label,
            l2_bytes=l2_bytes,
            fixed_duration_us=Microseconds(intercept),
            # slope is us per byte; its reciprocal is bytes per us, and GB/s is
            # 1000-based, so the same 1e3 as gbs_from_bytes_us.
            asymptotic_gbs=GBPerSecond(1.0 / (slope * 1e3)),
            max_residual_pct=residual,
            copies=tuple(copies),
        )

    def floor_us(self, moved_bytes: Bytes) -> Microseconds:
        """Fastest one copy of ``moved_bytes`` can run, per the fitted law.

        Args:
            moved_bytes: Bytes crossing DRAM in one launch, read plus write.

        Returns:
            The floor duration.

        Raises:
            ValueError: If ``moved_bytes`` is not positive, or if the law puts the
                floor at or below zero, which no duration can be under.
        """
        if moved_bytes <= 0:
            raise ValueError(f"moved_bytes must be positive, got {moved_bytes}")
        floor = Microseconds(
            self.fixed_duration_us + moved_bytes / (1e3 * self.asymptotic_gbs)
        )
        if floor <= 0.0:
            raise ValueError(
                f"the fitted law puts the floor for {moved_bytes} bytes at "
                f"{floor} us; fixed_duration_us={self.fixed_duration_us} is too "
                f"negative for this footprint and the sweep needs widening"
            )
        return floor

    def floor_gbs(self, moved_bytes: Bytes) -> GBPerSecond:
        """Rate a copy of ``moved_bytes`` reaches at its own floor.

        This is the denominator the single-point ceiling should have been: the same
        probe, measured at the footprint the kernel has.

        Args:
            moved_bytes: Bytes crossing DRAM in one launch, read plus write.

        Returns:
            The size-matched rate.
        """
        return gbs_from_bytes_us(moved_bytes, self.floor_us(moved_bytes))


def dram_time_floor(
    device: torch.device,
    *,
    l2_multiples: Sequence[int] = L2_MULTIPLES,
    iters: int = _ITERS,
    warmup: int = _WARMUP,
) -> DramTimeFloor:
    """Sweep the copy over several footprints and fit its time law.

    One buffer pair is allocated at the largest footprint and the smaller copies
    run on prefixes of it. Every sample then comes from one allocation, at one set
    of clocks, in the process that measures the kernel, and the sweep costs the
    peak memory of its largest point rather than the sum of all of them.

    Args:
        device: CUDA device.
        l2_multiples: Footprints per buffer, as multiples of the queried L2 size.
        iters: Timed iterations per footprint.
        warmup: Untimed iterations per footprint.

    Returns:
        The fitted floor, with every sample.

    Raises:
        RuntimeError: If the device is not CUDA.
        ValueError: If free memory collapses the sweep to fewer than two distinct
            footprints, or if any footprint is not above L2. A copy inside L2 is
            not a DRAM copy, and its rate would enter the fit as one.
    """
    if device.type != "cuda":
        raise RuntimeError("dram_time_floor needs a CUDA device")
    ordinal = device_ordinal(device)
    l2 = Bytes(torch.cuda.get_device_properties(ordinal).L2_cache_size)
    sizes = sorted({_buffer_bytes(device, multiple * l2) for multiple in l2_multiples})
    if len(sizes) < 2:
        raise ValueError(
            f"the sweep collapsed to {len(sizes)} distinct footprints; free memory "
            f"clamps every buffer to the same size and no law can be fitted"
        )
    # Checked after the collapse: a device too full to hold two footprints clamps
    # every point to the same size, and the L2 message would name the symptom.
    inside = [size for size in sizes if size <= l2]
    if inside:
        raise ValueError(
            f"footprints {inside} are within the {l2}-byte L2; a copy that fits in "
            f"L2 does not measure DRAM and its rate is not a DRAM rate"
        )
    src = torch.empty(sizes[-1], dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    src.random_(0, 255)
    copies: list[CopySample] = []
    for size in sizes:
        timed = measure(
            partial(dst[:size].copy_, src[:size]),
            label=f"dram copy {mib_from_bytes(Bytes(size)):.0f} MiB",
            iters=iters,
            warmup=warmup,
            device=device,
        )
        moved = Bytes(2 * size)
        copies.append(
            CopySample(
                moved_bytes=moved,
                duration=timed.total,
                achieved_gbs=gbs_from_bytes_us(moved, timed.total.min_duration_us),
                l2_multiple_ratio=ratio_of(size, l2),
            )
        )
    return DramTimeFloor.of(
        f"device-to-device copy over {len(sizes)} footprints, "
        f"{mib_from_bytes(Bytes(sizes[0])):.0f} to "
        f"{mib_from_bytes(Bytes(sizes[-1])):.0f} MiB per buffer",
        copies,
        l2_bytes=l2,
    )


@dataclass(frozen=True)
class ClassVerdict(PerfRecord):
    """Whether a kernel meets the bar for the class it declares.

    A kernel that is none of the three classes is a defect, not a result, so
    there is no fourth verdict and no unclassified state.

    Attributes:
        kernel: Kernel name.
        declared: One of the three class names.
        achieved_pct: Achieved rate as a percentage of the measured ceiling, or
            step share for SERIAL-tiny.
        required_pct: The bar for the declared class.
        passed: Whether the bar is met, in the direction the class demands.
    """

    kernel: str
    declared: str
    achieved_pct: Annotated[Percent, MEDIAN]
    required_pct: Annotated[Percent, MEDIAN]
    passed: bool


def dram_floor_verdict(
    kernel: str,
    *,
    moved_bytes: Bytes,
    launch_count: Count,
    duration_us: Microseconds,
    floor: DramTimeFloor,
) -> ClassVerdict:
    """Judge a DRAM-bound kernel against the time floor at its own traffic.

    The bar is :data:`CLASS_FLOOR_PCT`, unchanged. What changes is the denominator:
    the kernel is compared against a copy of its own size rather than against the
    largest copy the device can run. ``floor_us / duration_us`` is the same number
    as ``achieved_gbs / floor_gbs``, so ``achieved_pct`` still reads as a share of
    a measured copy rate.

    The fixed term is charged once per launch. Both measurements sum over the
    launches in the capture window, and a window of ``n`` launches pays the fixed
    cost ``n`` times, so folding it in once would understate the floor by
    ``(n - 1) * c`` and score every multi-launch kernel low.

    ``moved_bytes`` is measured DRAM traffic, not an analytic byte count, so a
    kernel that moves more bytes raises its own floor. Traffic the byte model does
    not contain therefore does not fail here; a register spill is caught by the
    spill rule in :func:`slinoss.perf.declared.floor_audit` instead.

    Args:
        kernel: Kernel name.
        moved_bytes: Measured DRAM read plus write, summed over the launches.
        launch_count: Launches those two sums cover.
        duration_us: Measured kernel duration, summed over the same launches.
        floor: The fitted time floor, measured on the same device in the same
            process.

    Returns:
        The verdict.

    Raises:
        ValueError: If ``launch_count`` is not positive.
    """
    if launch_count <= 0:
        raise ValueError(f"launch_count must be positive, got {launch_count}")
    per_launch = Bytes(moved_bytes // launch_count)
    total_floor = Microseconds(launch_count * floor.floor_us(per_launch))
    share = pct_of(total_floor, duration_us)
    bar = CLASS_FLOOR_PCT[DRAM_BOUND]
    return ClassVerdict(
        kernel=kernel,
        declared=DRAM_BOUND,
        achieved_pct=share,
        required_pct=bar,
        passed=share >= bar,
    )


def tensor_verdict(
    kernel: str, achieved_tflops: TFlopsPerSecond, ceiling: TensorCeiling
) -> ClassVerdict:
    """Judge a TENSOR-bound kernel against the measured GEMM ceiling.

    Args:
        kernel: Kernel name.
        achieved_tflops: Kernel flop over kernel duration.
        ceiling: The GEMM ceiling, in the same operand dtype.

    Returns:
        The verdict.
    """
    share = pct_of(achieved_tflops, ceiling.achieved_tflops)
    floor = CLASS_FLOOR_PCT[TENSOR_BOUND]
    return ClassVerdict(
        kernel=kernel,
        declared=TENSOR_BOUND,
        achieved_pct=share,
        required_pct=floor,
        passed=share >= floor,
    )


def serial_verdict(kernel: str, share_of_step_pct: Percent) -> ClassVerdict:
    """Judge a SERIAL-tiny kernel against its share of the step.

    Args:
        kernel: Kernel name.
        share_of_step_pct: Kernel duration as a percentage of step duration.

    Returns:
        The verdict. The comparison is an upper bound, unlike the other two.
    """
    limit = CLASS_FLOOR_PCT[SERIAL_TINY]
    return ClassVerdict(
        kernel=kernel,
        declared=SERIAL_TINY,
        achieved_pct=share_of_step_pct,
        required_pct=limit,
        passed=share_of_step_pct <= limit,
    )


@dataclass(frozen=True)
class GeometryVerdict(PerfRecord):
    """Whether a kernel's launch geometry leaves the device latency to hide with.

    Two rules, in one record because one launch configuration decides both and a
    verdict naming one alone reads as though the other had been checked. Neither is
    a share of a ceiling, so neither belongs in :class:`ClassVerdict`: a kernel
    scores against a bandwidth with whatever geometry it happens to have, and the
    two verdicts fail independently.

    The geometry is judged whether or not the class floor could be. A launch whose
    traffic stayed inside L2 gets no bandwidth verdict, and its grid and its
    occupancy are measured all the same.

    Attributes:
        kernel: Kernel name.
        declared: One of the three class names. It decides the block-count
            exemption and nothing else here.
        block_count: Blocks the launch requested.
        block_floor_count: Fewest blocks the rule allows, twice the queried
            multiprocessor count.
        thread_per_block_count: Threads per block, so a block count is read
            against the width it was traded against.
        achieved_occupancy_pct: Resident warps against warp slots, measured.
        required_occupancy_pct: The occupancy bar, :data:`MIN_OCCUPANCY_PCT`.
        theoretical_occupancy_pct: Occupancy the launch configuration allows.
            Reported beside the achieved figure because the two separate a
            register or shared-memory limit from a grid that ran out of work.
        block_floor_exempt: Whether the declared class waives the block floor.
        occupancy_passed: Whether achieved occupancy reaches its bar.
        block_floor_passed: Whether the block count reaches its floor, or is
            exempt from it.
        passed: Both of the above.
        detail: Which rule failed, by how much, and at what geometry. A waived
            block floor says so rather than reading as a pass on the count.
    """

    kernel: str
    declared: str
    block_count: Annotated[Count, INVARIANT]
    block_floor_count: Annotated[Count, INVARIANT]
    thread_per_block_count: Annotated[Count, INVARIANT]
    achieved_occupancy_pct: Annotated[Percent, MEDIAN]
    required_occupancy_pct: Annotated[Percent, MEDIAN]
    theoretical_occupancy_pct: Annotated[Percent, MEDIAN]
    block_floor_exempt: bool
    occupancy_passed: bool
    block_floor_passed: bool
    passed: bool
    detail: str


def geometry_verdict(
    kernel: str,
    *,
    declared: str,
    block_count: Count,
    thread_per_block_count: Count,
    achieved_occupancy_pct: Percent,
    theoretical_occupancy_pct: Percent,
    device: DeviceInfo,
) -> GeometryVerdict:
    """Judge one kernel's launch geometry against the occupancy and block rules.

    The block floor is :attr:`slinoss.perf.device.DeviceInfo.block_floor_count`,
    twice the multiprocessor count the part reports. It is taken off the device
    record rather than passed in as a number, so no caller can judge a kernel
    against a floor the hardware did not set.

    Args:
        kernel: Kernel name.
        declared: The class the kernel declares. Only
            :data:`BLOCK_FLOOR_EXEMPT_CLASSES` changes the outcome.
        block_count: Blocks in the grid, from the launch configuration.
        thread_per_block_count: Threads per block, from the same.
        achieved_occupancy_pct: Measured resident warps against warp slots.
        theoretical_occupancy_pct: Occupancy the launch configuration allows.
        device: The part the kernel ran on, which sets the block floor.

    Returns:
        The verdict.

    Raises:
        ValueError: If either launch extent is not positive. A launch with no
            blocks or no threads did not happen, and judging one would report a
            broken profile as a geometry failure.
    """
    if block_count <= 0 or thread_per_block_count <= 0:
        raise ValueError(
            f"kernel {kernel!r} reports {block_count} blocks of "
            f"{thread_per_block_count} threads; a launch extent must be positive"
        )
    floor = device.block_floor_count
    exempt = declared in BLOCK_FLOOR_EXEMPT_CLASSES
    occupancy_passed = achieved_occupancy_pct >= MIN_OCCUPANCY_PCT
    block_floor_passed = exempt or block_count >= floor
    geometry = (
        f"{block_count} blocks x {thread_per_block_count} threads, "
        f"theoretical occupancy {theoretical_occupancy_pct:.2f}%"
    )
    reasons: list[str] = []
    if not occupancy_passed:
        reasons.append(
            f"achieved occupancy {achieved_occupancy_pct:.2f}% is "
            f"{MIN_OCCUPANCY_PCT - achieved_occupancy_pct:.2f} points under the "
            f"{MIN_OCCUPANCY_PCT:.1f}% bar"
        )
    if not block_floor_passed:
        reasons.append(
            f"{block_count} blocks is {floor - block_count} under the "
            f"{floor}-block floor, twice the {device.sm_count} multiprocessors "
            f"the part reports"
        )
    if reasons:
        detail = "; ".join(reasons) + f"; at {geometry}"
    elif exempt and block_count < floor:
        detail = (
            f"occupancy clears its bar; the {floor}-block floor is waived by "
            f"{declared}, at {geometry}"
        )
    else:
        detail = f"occupancy and the block floor both clear, at {geometry}"
    return GeometryVerdict(
        kernel=kernel,
        declared=declared,
        block_count=block_count,
        block_floor_count=floor,
        thread_per_block_count=thread_per_block_count,
        achieved_occupancy_pct=achieved_occupancy_pct,
        required_occupancy_pct=MIN_OCCUPANCY_PCT,
        theoretical_occupancy_pct=theoretical_occupancy_pct,
        block_floor_exempt=exempt,
        occupancy_passed=occupancy_passed,
        block_floor_passed=block_floor_passed,
        passed=occupancy_passed and block_floor_passed,
        detail=detail,
    )
