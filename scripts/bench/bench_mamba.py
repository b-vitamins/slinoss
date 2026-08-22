"""Bench Mamba2's chunked scan under the same harness, as the external floor.

Same timer, same iteration count, same spread discipline, same report schema as
``bench_op.py``. A comparison run under two different harnesses compares the
harnesses.

Two group configurations are measured, because they are two different claims:

- ``groups=heads`` gives every head its own ``B`` and ``C``, which is what the
  SO(3) operator does, so the two move the same bytes.
- ``groups=1`` shares ``B`` and ``C`` across heads, which is Mamba2's own default
  and moves fewer bytes. It is the harder number to beat and it is reported
  rather than omitted.

Requires ``mamba-ssm``. Absent, the script exits with a message naming the
package instead of a traceback.

    python3 scripts/bench/bench_mamba.py --shape standard --mode both

``--against-so3ssd`` runs both operators inside one loop and judges the
per-iteration difference. Two separate runs cannot be subtracted: their medians
scatter further than either run's own floor. This is the only comparison against
the floor that resolves anything.

    python3 scripts/bench/bench_mamba.py --shape standard --mode step \\
        --groups heads --against-so3ssd

``--seq`` holds one shape's geometry and varies its sequence length, which the six
names cannot: they differ in five other extents as well.

``--end-to-end`` measures the second acceptance clause instead of the first. The
first compares the scan operators at iso state dimension; the second compares the
whole layer at iso ``d_model``, so norm, projections and convolution are inside both
arms. The bar is 1.0x rather than 1.15x, and the arms are the shipped modules:
``mamba_ssm.modules.mamba2.Mamba2`` against :class:`slinoss.mixer.SLinOSSMixer`.

    python3 scripts/bench/bench_mamba.py --shape acceptance --groups one \\
        --mode step --end-to-end --mamba-chunk 256

``--mamba-chunk`` reaches the layer arm as well as the operator arm. Both arms hold
a chunk length that is an internal tiling parameter, and at the acceptance geometry
the two optima differ by a factor of four, so a layer ratio taken at one shared
length is not a ratio of two floors.

Mamba2 has two forward paths and they are two different baselines. Its default,
``use_mem_eff_path=True``, calls ``mamba_split_conv1d_scan_combined``, which fuses
the projection split, the convolution, the scan and the gated norm; that is what a
Mamba2 user runs. Setting it False runs the convolution, then the
``mamba_chunk_scan_combined`` of the first clause, then a separate gated norm. Both
are measured, because a ratio against the unfused path is not a ratio against
Mamba2.

The comparison states what it holds equal. :func:`mapping_of` fixes the geometry,
:func:`mamba_arithmetic` and :func:`so3ssd_arithmetic` count the GEMM flop of each
side at it, and :func:`parameter_counts` counts the parameters of the two layers the
two operators sit inside. All of it reaches the report notes and stdout, so the
comparison can be judged instead of taken.

Chunk length is an internal tiling parameter of both implementations and not a
model hyperparameter, so the two arms need not share one. ``--chunk`` moves both,
``--mamba-chunk`` moves the Mamba2 arm alone. The second exists because Mamba2's
library default is 256 while the SO(3) arena refuses everything above 64: holding
both at one length scores at least one arm off its own optimum, and a bar measured
there is not that arm's floor. Pinning one arm and sweeping the other also makes
the pinned arm an in-loop normalizer, so a ratio taken across two runs is a ratio
of two ratios and carries no session-to-session drift.

    python3 scripts/bench/bench_mamba.py --shape acceptance --groups one \\
        --mode step --against-so3ssd --chunk 64 --mamba-chunk 256
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from slinoss.perf.budget import assert_closed, budget
from slinoss.perf.device import (
    await_exclusive,
    device_info,
    device_ordinal,
    require_cuda,
)
from slinoss.perf.dispersion import PairedRow
from slinoss.perf.memory import (
    SavedStorages,
    SavedTensorProbe,
    memory_peaks,
    pool_retention,
    reset_memory_peaks,
)
from slinoss.perf.report import Report, rate_table, write_report
from slinoss.perf.timing import Throughput, Timed, measure, measure_paired, region
from slinoss.perf.workload import SHAPES, OpShape, shape_by_name
from slinoss.perf.workload import forward_only as so3ssd_forward_only
from slinoss.perf.workload import make_inputs as so3ssd_inputs
from slinoss.perf.workload import step as so3ssd_step

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "step")


def load_scan() -> Callable[..., Any]:
    """Import Mamba2's chunked scan.

    Returns:
        ``mamba_chunk_scan_combined``.

    Raises:
        SystemExit: If ``mamba-ssm`` is not installed.
    """
    try:
        from mamba_ssm.ops.triton.ssd_combined import (  # type: ignore[import-not-found]
            mamba_chunk_scan_combined,
        )
    except ImportError as exc:
        raise SystemExit(f"bench_mamba needs mamba-ssm: {exc}") from exc
    return mamba_chunk_scan_combined


def load_block() -> Any:
    """Import Mamba2's layer module.

    Returns:
        The ``Mamba2`` class.

    Raises:
        SystemExit: If ``mamba-ssm`` is not installed.
    """
    try:
        from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(f"bench_mamba needs mamba-ssm: {exc}") from exc
    return Mamba2


class Mapping(NamedTuple):
    """The geometry both operators are held at.

    Attributes:
        headdim: Mamba2's ``headdim``, the SO(3) ``P``.
        dstate: Mamba2's ``dstate``, the SO(3) ``3N``.
        ngroups: Mamba2's ``ngroups``.
        chunk: Mamba2's ``chunk_size``.
        so3_chunk: The SO(3) ``L``. Equal to ``chunk`` unless the two arms were
            deliberately run at their own tiling.
        state_elems: State elements one head carries, ``headdim * dstate``. Equal on
            both sides by construction, which is what the mapping fixes.
    """

    headdim: int
    dstate: int
    ngroups: int
    chunk: int
    so3_chunk: int
    state_elems: int

    @property
    def iso_chunk(self) -> bool:
        """Whether both arms tile at one chunk length."""
        return self.chunk == self.so3_chunk

    def describe(self) -> str:
        """One line for a report note."""
        tiling = (
            f"chunk_size={self.chunk}"
            if self.iso_chunk
            else f"chunk_size={self.chunk} against L={self.so3_chunk}, per-arm tiling"
        )
        return (
            f"mapping: headdim={self.headdim} dstate={self.dstate} "
            f"ngroups={self.ngroups} {tiling}, "
            f"{self.state_elems:,} state elements per head on both sides"
        )


def mapping_of(shape: OpShape, groups: int, chunk: int | None = None) -> Mapping:
    """Map one SO(3) shape onto Mamba2's four geometry arguments.

    ``dstate = 3N``, so a head carries ``P * 3N`` state elements on both sides and
    the chunk-state buffer is the same size in both. The alternative, ``dstate = N``,
    matches the lane count instead and hands Mamba2 a third of the state, which
    would make any win here an artefact of the mapping.

    The mapping is not neutral. At equal state elements the SO(3) operator does more
    arithmetic per element: its score is per head, because the rotation is, while
    Mamba2 shares one ``C B^T`` across a group, and its forcing has two taps.
    :func:`mamba_arithmetic` and :func:`so3ssd_arithmetic` count both sides so the
    gap is visible.

    Args:
        shape: The SO(3) shape.
        groups: Mamba2 group count. Equal to ``shape.groups`` for a matched run.
        chunk: Mamba2's chunk length, or None for the shape's own.

    Returns:
        The mapping.
    """
    return Mapping(
        headdim=shape.rows,
        dstate=shape.d_state,
        ngroups=groups,
        chunk=shape.chunk if chunk is None else chunk,
        so3_chunk=shape.chunk,
        state_elems=shape.rows * shape.d_state,
    )


class Arithmetic(NamedTuple):
    """GEMM flop of one operator call at one geometry. Counted, not measured.

    Both state passings are omitted from both sides: each is ``2 * P * dstate`` per
    chunk per head against ten times that per token, which is under one percent of
    either total at any admitted chunk length.

    Attributes:
        label: Which side.
        forward_flop: Flop of one forward call.
        backward_flop: Flop of one backward call, including whatever that side
            recomputes rather than saves.
    """

    label: str
    forward_flop: int
    backward_flop: int

    @property
    def step_flop(self) -> int:
        """Forward plus backward."""
        return self.forward_flop + self.backward_flop

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.label} flop: forward {self.forward_flop / 1e9:,.2f}G "
            f"backward {self.backward_flop / 1e9:,.2f}G "
            f"step {self.step_flop / 1e9:,.2f}G"
        )


def mamba_arithmetic(
    shape: OpShape, groups: int, chunk: int | None = None
) -> Arithmetic:
    """Count Mamba2's GEMM flop at one geometry.

    Forward, from ``_mamba_chunk_scan_combined_fwd``: ``_chunk_state_fwd`` contracts
    ``x`` against ``B`` over the chunk, ``_bmm_chunk_fwd`` forms ``C B^T`` once per
    group, and ``_chunk_scan_fwd`` applies that score to ``x`` and adds ``C`` against
    the incoming state.

    Backward, from ``_mamba_chunk_scan_combined_bwd``: two forward kernels are
    recomputed, and then ``_chunk_scan_bwd_dstates``, the two contractions inside
    ``_chunk_scan_chunk_state_bwd_dx``, ``_chunk_state_bwd_db``,
    ``_chunk_scan_bwd_dC``, ``_chunk_scan_bwd_dcb``, ``_chunk_scan_bwd_ddAcs_stable``
    and two ``_bmm_chunk_bwd`` calls. Five contractions over ``P * dstate`` per token
    per head, three over ``L * P``, three over ``L * dstate`` per token per group.

    Args:
        shape: The SO(3) shape the geometry is mapped from.
        groups: Mamba2 group count.
        chunk: Mamba2's chunk length, or None for the shape's own. The score and the
            diagonal terms are both linear in it, so a chunk sweep that left this at
            the shape's default would report one flop for four tilings.

    Returns:
        The count.
    """
    m = mapping_of(shape, groups, chunk)
    lanes = shape.bsz * shape.heads * shape.seq
    band = shape.bsz * groups * shape.seq
    state = 2 * m.headdim * m.dstate
    score = 2 * m.chunk * m.dstate
    diagonal = 2 * m.chunk * m.headdim
    return Arithmetic(
        label=f"mamba-g{groups}",
        forward_flop=lanes * (2 * state + diagonal) + band * score,
        backward_flop=lanes * (5 * state + 3 * diagonal) + band * 3 * score,
    )


def so3ssd_arithmetic(shape: OpShape) -> Arithmetic:
    """Count the SO(3) operator's GEMM flop at one geometry.

    The terms are the sweep's, so there is one flop model for this operator and a
    driver cannot carry a second. A rematerialized forward kernel counts against the
    backward, which is where it runs.

    Args:
        shape: The SO(3) shape.

    Returns:
        The count.
    """
    from scripts.perf.chunk_sweep import flop_terms, geometry_of

    lanes = shape.bsz * shape.heads * shape.seq
    forward = sum(
        term.flop
        for term in flop_terms(geometry_of(shape), shape.chunk)
        if term.kernel.endswith("_fwd")
    )
    total = sum(term.flop for term in flop_terms(geometry_of(shape), shape.chunk))
    return Arithmetic(
        label="so3ssd",
        forward_flop=lanes * forward,
        backward_flop=lanes * (total - forward),
    )


class Parameters(NamedTuple):
    """Parameters of one layer at the mapped geometry.

    Attributes:
        label: Which side.
        elements: Parameter elements. Not ``count``, which is a tuple method.
    """

    label: str
    elements: int


def d_model_of(shape: OpShape) -> int:
    """The residual-stream width both layers are held at.

    Read off the SO(3) layer's own config rather than recomputed. The second
    acceptance clause is stated at iso ``d_model``, so the equality it rests on has
    one definition and the Mamba2 arm is built from that one. A second copy of
    ``H * P / 2`` here could drift from the layer while the clause still read as iso.

    Args:
        shape: The SO(3) shape.

    Returns:
        ``d_model``.

    Raises:
        ValueError: If ``H*P`` is odd, so no integer width expands to it.
    """
    from slinoss.perf.workload import layer_config

    return layer_config(shape, groups=shape.groups).d_model


def fused_path_blocker() -> str | None:
    """Report why Mamba2's fused forward cannot run, if it cannot.

    ``mamba_split_conv1d_scan_combined`` calls ``causal_conv1d`` unconditionally and
    has no fallback, so the fused path raises ``TypeError`` on a ``None`` handle
    rather than degrading. The unfused path guards the same handle and falls back to
    ``nn.Conv1d``. Probe the handle instead of the package: what decides is the
    symbol ``ssd_combined`` bound at import.

    Returns:
        A one-line reason, or ``None`` if the fused path is callable.
    """
    try:
        from mamba_ssm.ops.triton import (  # type: ignore[import-not-found]
            ssd_combined,
        )
    except ImportError as exc:
        return f"mamba-ssm is not importable: {exc}"
    if getattr(ssd_combined, "causal_conv1d_fwd_function", None) is None:
        return (
            "the causal_conv1d package is absent, so "
            "mamba_split_conv1d_scan_combined calls a None handle"
        )
    return None


def unbuilt_stage_blocker(device: torch.device) -> str | None:
    """Report which layer stages have no kernel registered for this device.

    Every stage of the mixer dispatches, and a reference path exists for all of them
    so the layer runs wherever torch runs. That fallback is not a degradation to note
    afterwards, it is a different program: the convolution's reference is an
    ``unfold``, a ``cat`` and a reduction over the tap axis, which moves the window
    count's multiple of the band's bytes and costs several times the rest of the
    layer put together.

    A kernel backend registers only when what it needs imported, so an unbuilt
    extension shows up as a slower arm and not as an error. This turns it into one.

    What is checked is registration for the device type, not the resolution at one
    dtype. A stage whose kernel declares no instantiation for the requested dtype
    falls back for a reason no build changes, and float32 is such a case on the scan.

    Args:
        device: The device the comparison runs on.

    Returns:
        A one-line reason naming every stage with no kernel for the device, or
        ``None`` if every stage has one.
    """
    from slinoss.ops.conv import backends as conv_dispatch
    from slinoss.ops.mixer import backends as tail_dispatch
    from slinoss.ops.scanprep import backends as prep_dispatch
    from slinoss.ops.so3ssd import backends as scan_dispatch

    stages = (
        ("conv", conv_dispatch),
        ("scanprep", prep_dispatch),
        ("so3ssd", scan_dispatch),
        ("mixer", tail_dispatch),
    )
    absent = [
        label
        for label, registry in stages
        if not any(
            registry.get(name).name != "reference"
            and device.type in registry.get(name).device_types
            for name in registry.names()
        )
    ]
    if not absent:
        return None
    return (
        f"no {device.type} kernel is registered for {', '.join(absent)}, so the arm "
        f"would run a reference path; build the extensions with "
        f"`python3 setup.py build_ext --inplace` before comparing"
    )


def make_mamba_block(
    shape: OpShape,
    groups: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mem_eff_path: bool = True,
    chunk: int | None = None,
) -> Any:
    """Build a Mamba2 layer at the mapped geometry.

    Args:
        shape: The SO(3) shape.
        groups: Mamba2 group count.
        device: Where to allocate.
        dtype: Parameter dtype.
        mem_eff_path: Whether the layer takes its fused forward. False runs the
            convolution, ``mamba_chunk_scan_combined`` and the gated norm as three
            calls, which is the path whose scan the first clause measures.
        chunk: Chunk length for this layer's scan, or None for the shape's own.
            The layer arm needs the same per-arm tiling the operator arm has:
            Mamba2's library default is 256 while the SO(3) arena refuses
            everything above 64, so a layer comparison at one shared length
            scores the baseline off its own optimum.

    Returns:
        The layer.

    Raises:
        SystemExit: If ``mamba-ssm`` is not installed, or if the fused path is
            asked for and cannot run here.
    """
    if mem_eff_path:
        blocker = fused_path_blocker()
        if blocker is not None:
            raise SystemExit(f"--mamba-path fused is unavailable: {blocker}")
    mapping = mapping_of(shape, groups, chunk)
    return load_block()(
        d_model=d_model_of(shape),
        d_state=mapping.dstate,
        headdim=mapping.headdim,
        ngroups=mapping.ngroups,
        expand=2,
        chunk_size=mapping.chunk,
        use_mem_eff_path=mem_eff_path,
        device=device,
        dtype=dtype,
    )


def block_runner(
    module: Any,
    x: Tensor,
    dy: Tensor,
    *,
    grads: bool,
    prefix: str,
) -> Callable[[], None]:
    """Build the timed callable for one layer.

    The backward differentiates with respect to the input and every parameter,
    which is the gradient set a training step forms. Restricting it to the input
    would drop both projections' weight gradients, and those are the largest GEMMs
    in either layer.

    Args:
        module: The layer, already on the device.
        x: ``(B,T,d_model)`` input, requiring grad in step mode.
        dy: ``(B,T,d_model)`` cotangent seed.
        grads: Whether to run the backward.
        prefix: Region label prefix.

    Returns:
        The callable.

    Raises:
        ValueError: If a step is asked for and nothing requires grad, which would
            time a forward and call it a step.
    """
    if not grads:

        def run_forward() -> None:
            with torch.no_grad(), region(f"{prefix}.forward"):
                module(x)

        return run_forward

    targets = (x, *(p for p in module.parameters() if p.requires_grad))
    if not any(t.requires_grad for t in targets):
        raise ValueError(f"{prefix} step needs at least one tensor requiring grad")

    def run_step() -> None:
        with region(f"{prefix}.forward"):
            y = module(x)
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, targets, dy)

    return run_step


def block_stream(
    shape: OpShape,
    device: torch.device,
    *,
    dtype: torch.dtype,
    requires_grad: bool,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """The residual-stream input and cotangent both layers are fed.

    One seed, so the two arms see the same numbers. Each arm needs its own leaf to
    accumulate into, so the input is drawn once and cloned per arm by the caller.

    Args:
        shape: The SO(3) shape.
        device: Where to allocate.
        dtype: Activation dtype.
        requires_grad: Whether the input is a differentiable leaf.
        seed: Generator seed.

    Returns:
        The input and the cotangent, both ``(B,T,d_model)``.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    size = (shape.bsz, shape.seq, d_model_of(shape))
    with torch.no_grad():
        x = torch.randn(*size, dtype=dtype, device=device, generator=gen)
        dy = torch.randn(*size, dtype=dtype, device=device, generator=gen)
    return x.requires_grad_(requires_grad), dy


def parameter_counts(shape: OpShape, groups: int) -> tuple[Parameters, Parameters]:
    """Count the parameters of the two layers the two operators sit inside.

    Neither operator holds a parameter: both take per-token tensors a projection
    produced. The comparable count is therefore the layer's, and it is counted from
    the shipped modules rather than derived, so it cannot drift from either.

    ``d_model`` is ``H * P / 2``, the width that expands to the measured inner
    dimension at Mamba2's default expansion and at this project's.

    Both modules are built on the host. The count is a property of the layer, and
    building on the device under measurement would put two layers of parameters into
    its allocator.

    Args:
        shape: The SO(3) shape.
        groups: Group count, both sides.

    Returns:
        Mamba2's count and the SO(3) mixer's, in that order.

    Raises:
        SystemExit: If ``mamba-ssm`` is not installed.
    """
    from slinoss.mixer import SLinOSSMixer
    from slinoss.perf.workload import layer_config

    # ``use_mem_eff_path`` selects a forward and allocates nothing, so the count is
    # the same either way. Ask for the unfused path so a host without
    # ``causal_conv1d`` can still count parameters.
    theirs = make_mamba_block(
        shape, groups, device=torch.device("cpu"), mem_eff_path=False
    )
    ours = SLinOSSMixer(
        layer_config(shape, groups=groups), device="cpu", dtype=torch.float32
    )
    return (
        Parameters(f"mamba2-g{groups}", sum(p.numel() for p in theirs.parameters())),
        Parameters("slinoss-mixer", sum(p.numel() for p in ours.parameters())),
    )


class ArmTimes(NamedTuple):
    """One arm's median durations inside a paired loop.

    Attributes:
        label: Arm label.
        total_us: Median of the whole arm.
        forward_us: Median of its forward region.
        backward_us: Median of its backward region, or None in forward mode.
    """

    label: str
    total_us: float
    forward_us: float
    backward_us: float | None

    def describe(self) -> str:
        """One line for a report note."""
        tail = "-" if self.backward_us is None else f"{self.backward_us:,.1f}"
        return (
            f"{self.label}: total {self.total_us:,.1f} us "
            f"forward {self.forward_us:,.1f} us backward {tail} us"
        )


def arm_times(timed: Timed, label: str) -> ArmTimes:
    """Read one arm's three medians out of a paired measurement.

    Args:
        timed: The paired loop.
        label: The arm's region label. Its forward and backward are its children.

    Returns:
        The medians. The backward is None when the arm ran without gradients.

    Raises:
        KeyError: If the arm or its forward is not in the tree.
    """
    backward: float | None
    try:
        backward = float(timed.region(f"{label}.backward").spread.median_duration_us)
    except KeyError:
        backward = None
    return ArmTimes(
        label=label,
        total_us=float(timed.region(label).spread.median_duration_us),
        forward_us=float(timed.region(f"{label}.forward").spread.median_duration_us),
        backward_us=backward,
    )


class MambaInputs(NamedTuple):
    """Inputs to ``mamba_chunk_scan_combined``.

    Attributes:
        x: ``(batch, seqlen, nheads, headdim)``.
        dt: ``(batch, seqlen, nheads)``.
        A: ``(nheads,)``, float32. Mamba2 requires float32 here.
        B: ``(batch, seqlen, ngroups, dstate)``.
        C: ``(batch, seqlen, ngroups, dstate)``.
        dy: Output-gradient seed, shaped like ``x``.
    """

    x: Tensor
    dt: Tensor
    A: Tensor
    B: Tensor
    C: Tensor
    dy: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The five tensors gradients are taken with respect to."""
        return (self.x, self.dt, self.A, self.B, self.C)


def make_inputs(
    shape: OpShape,
    groups: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    requires_grad: bool,
    seed: int = 0,
) -> MambaInputs:
    """Build Mamba2 inputs matching one SO(3) shape.

    The geometry is :func:`mapping_of`: ``headdim`` is the SO(3) row count and
    ``dstate`` is its ``3N``, so a head carries the same state in both operators.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    x = randn(shape.bsz, shape.seq, shape.heads, shape.rows)
    return MambaInputs(
        x=x.requires_grad_(requires_grad),
        dt=randn(shape.bsz, shape.seq, shape.heads, dt=torch.float32).requires_grad_(
            requires_grad
        ),
        A=(-randn(shape.heads, dt=torch.float32).abs()).requires_grad_(requires_grad),
        B=randn(shape.bsz, shape.seq, groups, shape.d_state).requires_grad_(
            requires_grad
        ),
        C=randn(shape.bsz, shape.seq, groups, shape.d_state).requires_grad_(
            requires_grad
        ),
        dy=randn(shape.bsz, shape.seq, shape.heads, shape.rows),
    )


def runner(
    scan: Callable[..., Any],
    inputs: MambaInputs,
    chunk: int,
    *,
    grads: bool,
    prefix: str = "mamba",
) -> Callable[[], None]:
    """Build the timed callable for one mode.

    Args:
        scan: ``mamba_chunk_scan_combined``.
        inputs: Its inputs.
        chunk: Chunk length.
        grads: Whether to run the backward.
        prefix: Region label prefix. Two arms measured in one loop need two
            prefixes; see :func:`slinoss.perf.workload.forward_only`.

    Returns:
        The callable.
    """

    def forward() -> Tensor:
        return scan(inputs.x, inputs.dt, inputs.A, inputs.B, inputs.C, chunk)

    if not grads:

        def run_forward() -> None:
            with torch.no_grad(), region(f"{prefix}.forward"):
                forward()

        return run_forward

    def run_step() -> None:
        with region(f"{prefix}.forward"):
            y = forward()
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, inputs.differentiable, inputs.dy)

    return run_step


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        choices=[s.name for s in SHAPES],
        help="Shape to bench. Repeatable. Defaults to every standard shape.",
    )
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument(
        "--groups",
        action="append",
        choices=["heads", "one"],
        help="Group configuration. Repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--seq",
        action="append",
        type=int,
        help="Sequence length to hold each shape at. Repeatable. Defaults to the "
        "shape's own. The named shapes differ in H, P, N, L and G as well as T, so "
        "a table over the names is five geometries and not a sweep in T.",
    )
    parser.add_argument(
        "--chunk",
        action="append",
        type=int,
        help="Chunk length to hold each shape at, on both arms. Repeatable. "
        "Defaults to the shape's own. Powers of two only.",
    )
    parser.add_argument(
        "--mamba-chunk",
        type=int,
        default=None,
        help="Chunk length for the Mamba2 arm alone, overriding --chunk. Mamba2's "
        "library default is 256 and the SO(3) arena refuses everything above 64, so "
        "a single shared length scores at least one arm off its own tiling optimum. "
        "Reaches both the operator arm and the --end-to-end layer arm.",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device, cuda or cuda:N. There is no host path: every "
        "report names the part the numbers came from.",
    )
    parser.add_argument(
        "--against-so3ssd",
        action="store_true",
        help=(
            "Measure the SO(3) operator against Mamba2 inside one loop and judge "
            "the per-iteration difference. Needs an even --iters."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="SO(3) backend for the comparison arm. Default is the fastest one.",
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help=(
            "Measure the whole Mamba2 layer against the whole SO(3) mixer at iso "
            "d_model, which is the second acceptance clause and a 1.0x bar. Needs "
            "an even --iters."
        ),
    )
    parser.add_argument(
        "--mamba-path",
        choices=["fused", "unfused", "both"],
        default="both",
        help="Which Mamba2 forward the end-to-end baseline takes. `fused` is its "
        "default mem-eff path and what a Mamba2 user runs; `unfused` runs the "
        "convolution, the scan of the first clause, and the gated norm as three "
        "calls. They are two different baselines, so both are measured by default.",
    )
    parser.add_argument(
        "--require-idle",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Wait up to SECONDS for the device to read idle on five consecutive "
        "probes before measuring, and refuse to measure if it does not. A foreign "
        "process moves a median by more than 2x on this fleet without moving the "
        "spread, so a contended run reports as reproducible.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/bench-mamba"))
    return parser.parse_args(argv)


def group_counts(shape: OpShape, kinds: Sequence[str]) -> tuple[int, ...]:
    """Resolve group kinds to distinct group counts, in the order requested.

    Args:
        shape: The shape being benched. ``heads`` resolves against its head count.
        kinds: ``heads``, ``one``, or both.

    Returns:
        The group counts to measure. ``heads`` and ``one`` are the same
        configuration at ``heads == 1``, and one configuration is measured once.
    """
    counts = [shape.heads if kind == "heads" else 1 for kind in kinds]
    return tuple(dict.fromkeys(counts))


def seq_variants(shape: OpShape, lengths: Sequence[int]) -> tuple[OpShape, ...]:
    """One copy of a shape per requested sequence length.

    Only ``T`` moves. A ratio taken across the named shapes moves ``H``, ``P``,
    ``N``, ``L`` and ``G`` with it, so it says nothing about sequence length; this
    holds the geometry the headline ratio was measured at and varies the one extent.

    ``T`` reaches the shape name and therefore the report file name, so two lengths
    of one shape write two reports instead of the second overwriting the first.

    Args:
        shape: The named shape.
        lengths: Sequence lengths, or empty for the shape's own.

    Returns:
        The shapes to measure, in the order requested.

    Raises:
        ValueError: If a length is not positive.
    """
    if not lengths:
        return (shape,)
    for seq in lengths:
        if seq < 1:
            raise ValueError(f"a sequence length must be positive, got {seq}")
    return tuple(
        replace(shape, name=f"{shape.name}-t{seq}", seq=seq) for seq in lengths
    )


def chunk_variants(shape: OpShape, lengths: Sequence[int]) -> tuple[OpShape, ...]:
    """One copy of a shape per requested chunk length.

    Only ``L`` moves, and it moves on both arms unless ``--mamba-chunk`` overrides
    the Mamba2 one. ``L`` reaches the shape name and therefore the report file name,
    so two lengths write two reports.

    Args:
        shape: The named shape.
        lengths: Chunk lengths, or empty for the shape's own.

    Returns:
        The shapes to measure, in the order requested.

    Raises:
        ValueError: If a length is not a positive power of two. The config admits no
            other, so a non-power-of-two would be refused after the inputs were
            allocated rather than here.
    """
    if not lengths:
        return (shape,)
    for chunk in lengths:
        if chunk < 1 or chunk & (chunk - 1):
            raise ValueError(f"a chunk length must be a power of two, got {chunk}")
    return tuple(
        replace(shape, name=f"{shape.name}-l{chunk}", chunk=chunk) for chunk in lengths
    )


def mamba_chunk(shape: OpShape, override: int | None) -> int:
    """Chunk length the Mamba2 arm runs at.

    Args:
        shape: The shape, carrying the shared default.
        override: ``--mamba-chunk``, or None for the shape's own.

    Returns:
        The length.

    Raises:
        ValueError: If the override is not a positive power of two.
    """
    if override is None:
        return shape.chunk
    if override < 1 or override & (override - 1):
        raise ValueError(f"a chunk length must be a power of two, got {override}")
    return override


def mamba_tag(shape: OpShape, chunk: int) -> str:
    """Label suffix naming the Mamba2 arm's own chunk length, when it has one.

    Empty at iso-chunk, so a matched run's arm label and report name are what they
    were before per-arm tiling existed. Non-empty otherwise, because two runs that
    differ only in the Mamba2 chunk would otherwise carry one label and the second
    report would overwrite the first.

    Args:
        shape: The shape, carrying the shared length.
        chunk: The Mamba2 arm's length.

    Returns:
        ``-l<chunk>`` when the two differ, else the empty string.
    """
    return "" if chunk == shape.chunk else f"-l{chunk}"


def _saved(
    scan: Callable[..., Any],
    shape: OpShape,
    groups: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk: int,
) -> SavedStorages:
    """Probe what Mamba2's graph holds for one forward and backward.

    Runs under a recorder so each save attributes to the region it was taken in.
    Without one every row would read ``unattributed``.
    """
    inputs = make_inputs(shape, groups, device, dtype=dtype, requires_grad=True)
    probe = SavedTensorProbe()
    with probe:
        measure(
            runner(scan, inputs, chunk, grads=True),
            label=f"mamba {shape.name} saved",
            iters=1,
            warmup=0,
            device=device,
        )
    return probe.report(f"mamba {shape.name}", inputs.differentiable)


class Faceoff(NamedTuple):
    """One configuration's verdict together with what was held equal.

    Attributes:
        row: The verdict on the per-iteration differences.
        mapping: The geometry both arms ran at.
        flops: Mamba2's counted flop and the SO(3) operator's, in that order.
        params: Both layers' parameter counts, in the same order.
        arms: Both arms' medians, in the same order.
    """

    row: PairedRow
    mapping: Mapping
    flops: tuple[Arithmetic, Arithmetic]
    params: tuple[Parameters, Parameters]
    arms: tuple[ArmTimes, ArmTimes]

    def lines(self) -> tuple[str, ...]:
        """What was held equal and what each side paid for it.

        Returns:
            One line each for the mapping, the two flop counts, the parameter
            counts, and the two arms' medians.
        """
        return (
            self.mapping.describe(),
            *(count.describe() for count in self.flops),
            "parameters: "
            + ", ".join(f"{p.label} {p.elements:,}" for p in self.params),
            *(arm.describe() for arm in self.arms),
        )


class BlockFaceoff(NamedTuple):
    """One layer comparison's verdict together with what was held equal.

    No flop count. Mamba2 holds its convolution in an ``nn.Conv1d`` and the SO(3)
    mixer holds its own as a bare parameter, so a count that walked submodules would
    charge one side for a kernel it did not charge the other. An asymmetric flop
    model is worse than none, and the clause is stated in throughput.

    Attributes:
        row: The verdict on the per-iteration differences.
        d_model: The width both layers ran at.
        path: Which Mamba2 forward path the baseline arm took.
        chunk: The Mamba2 arm's chunk length.
        so3_chunk: The SO(3) arm's chunk length. Unequal to ``chunk`` when each arm
            was run at its own tiling optimum, which is the only way a layer ratio
            is a ratio of two floors.
        params: Both layers' parameter counts, Mamba2 first.
        arms: Both arms' medians, in the same order.
    """

    row: PairedRow
    d_model: int
    path: str
    chunk: int
    so3_chunk: int
    params: tuple[Parameters, Parameters]
    arms: tuple[ArmTimes, ArmTimes]

    def lines(self) -> tuple[str, ...]:
        """What was held equal and what each side paid for it."""
        theirs, ours = self.params
        excess = 100.0 * (ours.elements - theirs.elements) / theirs.elements
        tiling = (
            f"chunk_size={self.chunk} on both arms"
            if self.chunk == self.so3_chunk
            else f"mamba2 chunk_size={self.chunk} against so3ssd L={self.so3_chunk}, "
            f"per-arm tiling"
        )
        return (
            f"iso d_model={self.d_model} on both arms, mamba2 path={self.path}",
            tiling,
            "parameters: "
            + ", ".join(f"{p.label} {p.elements:,}" for p in self.params),
            f"the SO(3) layer carries {excess:+,.2f}% of Mamba2's parameters at "
            f"the same d_model",
            *(arm.describe() for arm in self.arms),
        )


def compare_block(
    shape: OpShape,
    groups: int,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
    *,
    mem_eff_path: bool,
) -> tuple[Report, BlockFaceoff]:
    """Measure the whole Mamba2 layer against the whole SO(3) mixer in one loop.

    This is the second acceptance clause. Both arms hold the same ``d_model`` and
    the same residual-stream input values, and both include the projections, the
    convolution and the norm. Mamba2 is the baseline arm, so ``speedup_ratio``
    above one means the SO(3) mixer is the faster of the two, and the bar is one.

    Args:
        shape: The problem size.
        groups: Mamba2 group count.
        mode: ``forward`` or ``step``.
        args: Parsed command line.
        device: Device to time on.
        mem_eff_path: Whether the Mamba2 arm takes its fused forward.

    Returns:
        The report and the face-off.
    """
    from slinoss.mixer import SLinOSSMixer
    from slinoss.perf.workload import layer_config

    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    path = "fused" if mem_eff_path else "unfused"
    mchunk = mamba_chunk(shape, args.mamba_chunk)
    a_label = f"mamba2-block-{path}{mamba_tag(shape, mchunk)}"
    b_label = "slinoss-mixer"
    theirs = make_mamba_block(
        shape,
        groups,
        device=device,
        dtype=dtype,
        mem_eff_path=mem_eff_path,
        chunk=mchunk,
    )
    ours = SLinOSSMixer(layer_config(shape, groups=groups), device=device, dtype=dtype)
    if not grads:
        theirs.requires_grad_(False)
        ours.requires_grad_(False)
    x, dy = block_stream(shape, device, dtype=dtype, requires_grad=grads)
    # One draw, one clone per arm: the same numbers, and a leaf each so neither
    # arm's backward accumulates into the other's.
    their_x = x.detach().clone().requires_grad_(grads)
    our_x = x.detach().clone().requires_grad_(grads)
    label = f"mamba2 block {path} vs slinoss mixer {shape.name} {mode} paired"
    reset_memory_peaks(device)
    out = measure_paired(
        a_label,
        block_runner(theirs, their_x, dy, grads=grads, prefix=a_label),
        b_label,
        block_runner(ours, our_x, dy, grads=grads, prefix=b_label),
        label=label,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    tree = budget(out.timed)
    assert_closed(tree)
    face = BlockFaceoff(
        row=out.comparison,
        d_model=d_model_of(shape),
        path=path,
        chunk=mchunk,
        so3_chunk=shape.chunk,
        params=parameter_counts(shape, groups),
        arms=(arm_times(out.timed, a_label), arm_times(out.timed, b_label)),
    )
    report = Report(
        title=f"bench: {label}",
        device=device_info(device_ordinal(device)),
        budget=tree,
        throughput=tuple(
            Throughput.of(name, shape.token_count, out.timed.region(name).spread)
            for name in (a_label, b_label)
        ),
        comparisons=(out.comparison,),
        peaks=memory_peaks(label, device),
        pool=pool_retention(label),
        notes=(
            shape.describe(),
            *face.lines(),
            f"so3ssd n_groups={shape.groups} mamba2 ngroups={groups}",
            f"mode={mode} dtype={args.dtype}",
            f"arm a={a_label} b={b_label}, one loop, order swapped each iteration",
            "both arms differentiate the input and every parameter",
            "each arm holds its own leaf of one draw; the memory peak covers both",
            f"iters={args.iters} warmup={args.warmup}",
            f"timer={out.timed.timer} clocks={out.timed.clocks}",
        ),
    )
    return report, face


def compare_so3ssd(
    scan: Callable[..., Any],
    shape: OpShape,
    groups: int,
    mode: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Report, Faceoff]:
    """Measure the SO(3) operator against Mamba2 in one loop at one configuration.

    Mamba2 is the baseline arm, so ``speedup_ratio`` above one means the SO(3)
    operator is the faster of the two.

    The geometry is :func:`mapping_of`, which equalizes the state elements per head
    and not the arithmetic. The counted flop of both sides and the parameter counts
    of both layers reach the notes, so the ratio can be read against what produced
    it.

    Args:
        scan: ``mamba_chunk_scan_combined``.
        shape: The problem size. Both arms carry the same state size per head.
        groups: Mamba2 group count. The SO(3) arm uses the shape's own, so a run
            with ``groups != shape.groups`` compares two group counts and says so.
        mode: ``forward`` or ``step``.
        args: Parsed command line.
        device: Device to time on.

    Returns:
        The report and the face-off.
    """
    dtype = DTYPES[args.dtype]
    grads = mode == "step"
    mchunk = mamba_chunk(shape, args.mamba_chunk)
    a_label = f"mamba-g{groups}{mamba_tag(shape, mchunk)}"
    b_label = f"so3ssd-{args.backend or 'auto'}"
    mapping = mapping_of(shape, groups, mchunk)
    flops = (mamba_arithmetic(shape, groups, mchunk), so3ssd_arithmetic(shape))
    params = parameter_counts(shape, groups)
    mamba = make_inputs(shape, groups, device, dtype=dtype, requires_grad=grads)
    ours = so3ssd_inputs(shape, device, dtype=dtype, requires_grad=grads)
    label = f"mamba g{groups} vs so3ssd {shape.name} {mode} paired"
    reset_memory_peaks(device)
    out = measure_paired(
        a_label,
        runner(scan, mamba, mchunk, grads=grads, prefix=a_label),
        b_label,
        (
            so3ssd_step(ours, shape.chunk, backend=args.backend, prefix=b_label)
            if grads
            else so3ssd_forward_only(
                ours, shape.chunk, backend=args.backend, prefix=b_label
            )
        ),
        label=label,
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    tree = budget(out.timed)
    assert_closed(tree)
    face = Faceoff(
        row=out.comparison,
        mapping=mapping,
        flops=flops,
        params=params,
        arms=(arm_times(out.timed, a_label), arm_times(out.timed, b_label)),
    )
    report = Report(
        title=f"bench: {label}",
        device=device_info(device_ordinal(device)),
        budget=tree,
        throughput=tuple(
            Throughput.of(name, shape.token_count, out.timed.region(name).spread)
            for name in (a_label, b_label)
        ),
        comparisons=(out.comparison,),
        peaks=memory_peaks(label, device),
        pool=pool_retention(label),
        notes=(
            shape.describe(),
            *face.lines(),
            f"so3ssd n_groups={shape.groups}",
            # Only when the arms tile differently. At iso-chunk the mapping line
            # already carries the one length, and a note restating it would say
            # nothing.
            *(
                ()
                if mchunk == shape.chunk
                else (f"mamba chunk_size={mchunk} against so3ssd L={shape.chunk}",)
            ),
            f"mode={mode} dtype={args.dtype}",
            f"arm a={a_label} b={b_label}, one loop, order swapped each iteration",
            # The two operators take different tensors, so the arms cannot share
            # inputs the way two backends of one operator do. The peak is the sum of
            # both arms' live tensors and belongs to neither.
            "each arm holds its own inputs; the memory peak covers both",
            f"iters={args.iters} warmup={args.warmup}",
            f"timer={out.timed.timer} clocks={out.timed.clocks}",
        ),
    )
    return report, face


def _run_blocks(
    shapes: Sequence[OpShape],
    modes: Sequence[str],
    wanted: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    """Run every layer comparison and print the verdicts.

    Returns:
        Process exit status.
    """
    paths = (
        (True, False) if args.mamba_path == "both" else (args.mamba_path == "fused",)
    )
    blocker = fused_path_blocker()
    if blocker is not None and args.mamba_path == "both":
        # Skipping is right only for ``both``, which asks for whatever runs. An
        # explicit ``--mamba-path fused`` must fail, not silently measure the other
        # path under the fused label.
        print(f"skipping the fused Mamba2 path: {blocker}")
        paths = (False,)
    rates: list[tuple[str, Throughput]] = []
    verdicts: list[BlockFaceoff] = []
    for shape in shapes:
        for groups in group_counts(shape, wanted):
            for mode in modes:
                for mem_eff in paths:
                    report, face = compare_block(
                        shape, groups, mode, args, device, mem_eff_path=mem_eff
                    )
                    base = args.out.with_name(
                        f"{args.out.name}-block-{face.path}-{shape.name}"
                        f"-g{groups}{mamba_tag(shape, face.chunk)}-{mode}"
                    )
                    md, _ = write_report(report, base, require_agreement=False)
                    rates += [
                        (f"{shape.name}/g{groups}/{mode}/{rate.label}", rate)
                        for rate in report.throughput
                    ]
                    verdicts.append(face)
                    print(f"wrote {md}")
    print()
    print(rate_table(rates, width=52))
    for face in verdicts:
        print()
        print(face.row.verdict())
        for line in face.lines():
            print(f"  {line}")
    return 0


def _run_comparisons(
    scan: Callable[..., Any],
    shapes: Sequence[OpShape],
    modes: Sequence[str],
    wanted: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    """Run every paired comparison against Mamba2 and print the verdicts.

    Returns:
        Process exit status.
    """
    rates: list[tuple[str, Throughput]] = []
    verdicts: list[Faceoff] = []
    for shape in shapes:
        for groups in group_counts(shape, wanted):
            for mode in modes:
                report, face = compare_so3ssd(scan, shape, groups, mode, args, device)
                tag = mamba_tag(shape, mamba_chunk(shape, args.mamba_chunk))
                base = args.out.with_name(
                    f"{args.out.name}-{shape.name}-g{groups}{tag}-{mode}-paired"
                )
                md, _ = write_report(report, base, require_agreement=False)
                rates += [
                    (f"{shape.name}/g{groups}/{mode}/{rate.label}", rate)
                    for rate in report.throughput
                ]
                verdicts.append(face)
                print(f"wrote {md}")
    print()
    print(rate_table(rates, width=52))
    for face in verdicts:
        print()
        print(face.row.verdict())
        for line in face.lines():
            print(f"  {line}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench.

    Returns:
        Process exit status.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    if args.require_idle is not None:
        # Before the first compile, so a wait does not sit between two arms of one
        # paired loop and desynchronize their warm state.
        await_exclusive(device_ordinal(device), timeout_s=args.require_idle)
    scan = load_scan()
    dtype = DTYPES[args.dtype]
    named = [shape_by_name(n) for n in (args.shape or [s.name for s in SHAPES])]
    shapes = [
        tiled
        for shape in named
        for variant in seq_variants(shape, args.seq or ())
        for tiled in chunk_variants(variant, args.chunk or ())
    ]
    modes = MODES if args.mode == "both" else (args.mode,)
    wanted = args.groups or ["heads", "one"]
    unbuilt = unbuilt_stage_blocker(device)
    if unbuilt is not None:
        # Not a warning. A ratio against Mamba2's kernels taken with one of these
        # stages on its reference path is a number about the build and not about
        # either implementation.
        raise SystemExit(unbuilt)
    if args.end_to_end:
        return _run_blocks(shapes, modes, wanted, args, device)
    if args.against_so3ssd:
        return _run_comparisons(scan, shapes, modes, wanted, args, device)
    info = device_info(device_ordinal(device))
    rows: list[tuple[str, Throughput]] = []
    for shape in shapes:
        # At heads=1 the two group kinds resolve to the same configuration. Running
        # both would time one thing twice, print it under one label, and have the
        # second report overwrite the first.
        for groups in group_counts(shape, wanted):
            for mode in modes:
                grads = mode == "step"
                mchunk = mamba_chunk(shape, args.mamba_chunk)
                inputs = make_inputs(
                    shape, groups, device, dtype=dtype, requires_grad=grads
                )
                label = f"mamba {shape.name} g{groups} {mode}"
                reset_memory_peaks(device)
                timed = measure(
                    runner(scan, inputs, mchunk, grads=grads),
                    label=label,
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                )
                peaks = memory_peaks(label, device)
                tree = budget(timed)
                assert_closed(tree)
                rate = Throughput.of(label, shape.token_count, timed.total)
                report = Report(
                    title=f"bench: {label}",
                    device=info,
                    budget=tree,
                    throughput=(rate,),
                    saved=(
                        _saved(scan, shape, groups, device, dtype, mchunk)
                        if grads
                        else None
                    ),
                    peaks=peaks,
                    pool=pool_retention(label),
                    notes=(
                        shape.describe(),
                        f"mamba2 ngroups={groups} headdim={shape.rows} "
                        f"dstate={shape.d_state}"
                        # The shape line already prints L, so name the tiling only
                        # when the arm was given one the shape does not carry.
                        + ("" if mchunk == shape.chunk else f" chunk_size={mchunk}"),
                        f"mode={mode} dtype={args.dtype}",
                        f"iters={args.iters} warmup={args.warmup}",
                        f"timer={timed.timer} clocks={timed.clocks}",
                    ),
                )
                tag = mamba_tag(shape, mchunk)
                base = args.out.with_name(
                    f"{args.out.name}-{shape.name}-g{groups}{tag}-{mode}"
                )
                md, _ = write_report(report, base, require_agreement=False)
                rows.append((f"{shape.name}/g{groups}{tag}/{mode}", rate))
                print(f"wrote {md}")
    print()
    print(rate_table(rows, width=28))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
