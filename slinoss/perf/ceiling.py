"""Measured ceilings. Nothing here comes from a product page.

A ceiling divides a measurement, so a modelled ceiling makes every ratio built on
it modelled too, and the fact is invisible by the time it reaches a table. Both
ceilings in this module are measured in the same process, on the same device, at
the same clocks as the kernel under test, which is the only way an unlocked clock
does not corrupt the ratio: numerator and denominator drift together.

Two ceilings, matching the two kernel classes:

- DRAM: a large device-to-device copy, counting a read and a write per byte.
- Tensor: a large square GEMM in the operand dtype, counting ``2*M*N*K`` flop.

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
"""

from __future__ import annotations

from dataclasses import dataclass
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
    Percent,
    PerfRecord,
    Spread,
    TFlopsPerSecond,
    gbs_from_bytes_us,
    pct_of,
    tflops_from_flop_us,
)

__all__ = [
    "CLASS_FLOOR_PCT",
    "DRAM_BOUND",
    "SERIAL_TINY",
    "TENSOR_BOUND",
    "Ceilings",
    "ClassVerdict",
    "DramCeiling",
    "TensorCeiling",
    "ceilings",
    "dram_ceiling",
    "dram_verdict",
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


def dram_verdict(
    kernel: str, achieved_gbs: GBPerSecond, ceiling: DramCeiling
) -> ClassVerdict:
    """Judge a DRAM-bound kernel against the measured copy ceiling.

    Args:
        kernel: Kernel name.
        achieved_gbs: Kernel bytes over kernel duration, both measured.
        ceiling: The copy ceiling.

    Returns:
        The verdict.
    """
    share = pct_of(achieved_gbs, ceiling.achieved_gbs)
    floor = CLASS_FLOOR_PCT[DRAM_BOUND]
    return ClassVerdict(
        kernel=kernel,
        declared=DRAM_BOUND,
        achieved_pct=share,
        required_pct=floor,
        passed=share >= floor,
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
