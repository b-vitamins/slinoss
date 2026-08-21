"""Which kernels each benchmarked arm launches, and the rule that a run judged them.

:mod:`slinoss.perf.declared` says what class a kernel is held to.
:func:`slinoss.perf.declared.floor_audit` judges the kernels a capture contained.
Neither can say that a capture contained the kernels it should have, so an audit over
an empty capture reports success: no verdict failed, because no verdict exists. That
is the failure this module closes.

The measured case: a conv audit exited zero having judged nothing. The compiled
extension had not been built in that environment, so
:mod:`slinoss.ops.conv.backends` never registered the native backend, the operator
resolved to its reference, and thirteen torch kernels were profiled and reported as
unjudged. Every rule held. Nothing was measured.

:data:`COVERAGE` names, per ``(op, mode)``, the declared kernels that arm launches.
:func:`coverage_verdict` compares that against what the audit judged and fails on a
shortfall, so a capture that holds fewer kernels than the arm declares is a nonzero
exit rather than a short table. Three shortfalls fail the same way: no capture at all,
a capture of the reference's kernels, and a capture missing one launch.

A kernel whose launch depends on the shape is declared :class:`Conditional` and
carries the condition. It is judged when the capture holds it and is not required when
it does not, which is the one thing this module cannot decide from the operator name
and the mode alone. Everything else is required, so an absence is a defect.

A kernel no arm launches at any shape is declared :data:`TARGETED` and carries the
driver that does launch it. That is the whole escape hatch: four entries, every one
named in the report, none excused from anything but :func:`unreachable`.

The table is data, not an import: naming a kernel here pulls in neither the operator
packages nor the DSL, so the completeness test runs on a host with no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

from slinoss.perf.declared import DECLARED, declared_key
from slinoss.perf.units import INVARIANT, Count, PerfRecord

__all__ = [
    "COVERAGE",
    "EXTENSION_GLOB",
    "MODES",
    "TARGETED",
    "Conditional",
    "CoverageVerdict",
    "DispatchVerdict",
    "OpCoverage",
    "RegistryChoice",
    "Targeted",
    "TreeProvenance",
    "coverage_of",
    "coverage_verdict",
    "tree_provenance",
    "unreachable",
]

MODES: Final[tuple[str, ...]] = ("forward", "step")
"""The two arms every operator is profiled under.

``forward`` runs under ``no_grad``; ``step`` runs the forward and the backward, so it
holds the forward's kernels too. One definition, because :data:`COVERAGE` is keyed by
mode and a driver offering a third mode would be judged against no entry."""


@dataclass(frozen=True)
class Conditional:
    """A declared kernel one arm launches at some shapes and not at others.

    Attributes:
        kernel: Key of :data:`slinoss.perf.declared.DECLARED`.
        condition: The shape property that makes the launch happen, as prose. Not a
            predicate: evaluating one would need the shape records, which live in
            :mod:`slinoss.perf.workload` and drag every operator package in behind
            them. The report quotes this so an absence is readable.
    """

    kernel: str
    condition: str


@dataclass(frozen=True)
class OpCoverage:
    """The declared kernels one operator launches in one mode.

    Attributes:
        required: Keys of :data:`slinoss.perf.declared.DECLARED` this arm launches at
            every shape. An audit that did not judge one of these failed.
        conditional: Kernels this arm launches at some shapes. Judged when present,
            not required when absent.
    """

    required: tuple[str, ...]
    conditional: tuple[Conditional, ...] = ()

    @property
    def kernels(self) -> tuple[str, ...]:
        """Every key this arm can launch, required and conditional, sorted."""
        return tuple(sorted({*self.required, *(c.kernel for c in self.conditional)}))


COVERAGE: Final[dict[tuple[str, str], OpCoverage]] = {
    ("so3ssd", "forward"): OpCoverage(
        required=(
            "increment_passing_fwd_kernel",
            "chunk_scan_fwd_kernel",
        )
    ),
    ("so3ssd", "step"): OpCoverage(
        required=(
            "increment_passing_fwd_kernel",
            "chunk_scan_fwd_kernel",
            "start_passing_bwd_kernel",
            "chunk_input_bwd_kernel",
            "chunk_prefix_bwd_kernel",
            "chunk_vector_bwd_kernel",
            "boundary_bwd_kernel",
        ),
        conditional=(
            Conditional(
                "reduce_rows_kernel",
                "3N above one lane tile, 48 columns: the vector backward writes "
                "dtrans and dK once per lane tile and closes the slots in one "
                "reduction, and at 3N == 48 there is one tile and no slot buffer",
            ),
            Conditional(
                "vector_reduce_kernel",
                "a head-sum depth above one: the vector backward shares a group's "
                "heads over that many blocks and a second launch closes the partials, "
                "and at H // G == 1 the depth is one and the kernel writes the summed "
                "outputs directly",
            ),
        ),
    ),
    ("conv", "forward"): OpCoverage(required=("conv1d_fwd_kernel",)),
    ("conv", "step"): OpCoverage(
        required=(
            "conv1d_fwd_kernel",
            "conv1d_bwd_kernel",
            "conv1d_reduce_parts_kernel",
        )
    ),
    ("scanprep", "forward"): OpCoverage(required=("scanprep_fwd_kernel",)),
    ("scanprep", "step"): OpCoverage(
        required=(
            "scanprep_fwd_kernel",
            "scanprep_bwd_kernel",
            "reduce_rows_kernel",
        )
    ),
    ("block", "forward"): OpCoverage(
        required=(
            "rmsnorm_residual_fwd_kernel",
            "swiglu_fwd_kernel",
            "rmsnorm_fwd_kernel",
        )
    ),
    ("block", "step"): OpCoverage(
        required=(
            "rmsnorm_residual_fwd_kernel",
            "swiglu_fwd_kernel",
            "rmsnorm_fwd_kernel",
            "rmsnorm_residual_bwd_kernel",
            "swiglu_bwd_kernel",
            "rmsnorm_bwd_kernel",
            "rmsnorm_dweight_kernel",
        )
    ),
    ("mixer", "forward"): OpCoverage(required=("mixer_tail_fwd_kernel",)),
    ("mixer", "step"): OpCoverage(
        required=(
            "mixer_tail_fwd_kernel",
            "mixer_tail_bwd_kernel",
            "reduce_rows_kernel",
        )
    ),
    ("xent", "forward"): OpCoverage(required=("xent_fwd_kernel", "reduce_rows_kernel")),
    ("xent", "step"): OpCoverage(
        required=("xent_fwd_kernel", "reduce_rows_kernel", "xent_bwd_kernel")
    ),
}
"""Every benchmarked arm, and the declared kernels it launches.

Read off the launch sites, not off a capture: a table derived from what a profiler
happened to report could not detect a missing launch, which is what it exists for.

The forward kernels recur in ``so3ssd``'s step when the backward recomputes the chunk
prologue rather than reading a saved one, so the step can hold one of them twice. One
key either way: a capture merges the launches of one symbol into one counter row.
``reduce_rows_kernel`` is shared by four arms: the
frontier's parameter-bias reduction, the fused tail's two parameter slots, the
loss's mean, and the vector backward's slot close. Only the last is conditional.

The step's two conditionals are separate shape properties and can differ. The slot
close follows the lane count, which the state width sets; ``vector_reduce_kernel``
follows the head-sum depth, which ``H // G`` sets. The benchmarked standard shape has
one lane tile and one head per group, so both are absent there and the acceptance
shape launches both.

``rmsnorm_dweight_kernel`` appears once in ``block``'s step and is launched twice
there, by the plain norm's backward and by the residual norm's. One key, because a
capture merges the launches of one symbol into one counter row."""


@dataclass(frozen=True)
class Targeted:
    """A declared kernel no benchmarked arm launches, and the driver that does.

    Attributes:
        kernel: Key of :data:`slinoss.perf.declared.DECLARED`.
        driver: Path of the script that launches it, with the flags that select it.
        reason: Why no arm in :data:`COVERAGE` reaches it.
    """

    kernel: str
    driver: str
    reason: str


TARGETED: Final[tuple[Targeted, ...]] = (
    Targeted(
        "chunk_start_bwd_kernel",
        "scripts/perf/profile_chunk_start_bwd.py",
        "the backward fused this GEMM into start_passing_bwd, so the operator no "
        "longer launches it on any path; it survives as the arm the fusion is "
        "ranked against and is declared here until it is deleted",
    ),
    Targeted(
        "chunk_increment_fwd_kernel",
        "scripts/perf/profile_increment_passing_fwd.py --arm pair",
        "the forward fused this GEMM into increment_passing_fwd, so the operator no "
        "longer launches it on any path; it survives as the arm the fusion is "
        "ranked against and is declared here until it is deleted",
    ),
    Targeted(
        "state_passing_fwd_kernel",
        "scripts/perf/profile_increment_passing_fwd.py --arm pair",
        "the forward recurrence now runs inside increment_passing_fwd over a shared "
        "increment, so nothing launches the standalone pass over an increment "
        "buffer; it survives as the other half of the arm the fusion is ranked "
        "against",
    ),
    Targeted(
        "state_passing_bwd_kernel",
        "scripts/perf/profile_start_passing_bwd.py --arm pair",
        "the operator launches the recurrence alone only when dy is absent, which "
        "is a caller differentiating the carried state and not the sequence; every "
        "arm in COVERAGE differentiates the output",
    ),
)
"""Declared kernels reached by a targeted driver rather than by a benchmarked arm.

The escape hatch, and it is narrow by construction: it excuses a kernel from
:func:`unreachable` and from nothing else, it names the command that does profile
the kernel, and every entry is a line in the report so the excuse is read every time
the audit runs. An entry whose kernel an arm does launch is a contradiction the
completeness test refuses.

No entry is a shape condition, which is why none is a :class:`Conditional`: three
kernels are off the operator's path entirely, two of them displaced by the forward's
fusion and one by the backward's, and the fourth is on a path no arm here takes. A
kernel that some shapes launch and others do not belongs in ``conditional`` instead,
where it is judged whenever the capture holds it.

A displaced kernel stays declared while it is still the arm a fusion is ranked
against. Deleting it deletes the comparison, so the entry is what keeps the losing
arm runnable rather than remembered."""


def coverage_of(op: str, mode: str) -> OpCoverage:
    """The kernels one arm launches.

    Args:
        op: Operator name, one of :data:`slinoss.perf.workload.OPS`.
        mode: One of :data:`MODES`.

    Returns:
        The entry.

    Raises:
        KeyError: If the pair has no entry. An arm no entry covers cannot be audited,
            because the audit would have nothing to be judged incomplete against.
    """
    entry = COVERAGE.get((op, mode))
    if entry is None:
        raise KeyError(
            f"no coverage entry for op {op!r} mode {mode!r}; have {sorted(COVERAGE)}"
        )
    return entry


def unreachable() -> tuple[str, ...]:
    """Declared kernels neither an arm nor a targeted driver launches, sorted.

    A class no driver reaches is a claim with no gate behind it: the audit judges what
    a capture contained, so a kernel nothing launches reads as verified. This is the
    static half of the coverage rule and needs no device.

    Returns:
        Keys of :data:`slinoss.perf.declared.DECLARED` absent from both
        :data:`COVERAGE` and :data:`TARGETED`.
    """
    claimed = {key for entry in COVERAGE.values() for key in entry.kernels}
    claimed |= {one.kernel for one in TARGETED}
    return tuple(sorted(set(DECLARED) - claimed))


EXTENSION_GLOB: Final = "_C/_conv1d*.so"
"""Where the compiled extension lands, relative to the package root.

Matched on the filesystem rather than imported: the question is what the tree
holds, and an import answers what ``sys.path`` found, which is the thing under
suspicion."""


@dataclass(frozen=True)
class TreeProvenance(PerfRecord):
    """Which source tree the measurement came out of.

    Attributes:
        package_root: Directory of the imported :mod:`slinoss` package, resolved.
        driver_root: Repository root the driver script lives under, resolved.
        same_tree: Whether ``package_root`` sits inside ``driver_root``. False means
            the scripts being run and the package being measured are two checkouts.
        extension: Path of the compiled conv extension inside ``package_root``, or
            ``absent``. Absent is what made a conv audit resolve to its reference.
        extension_stamp: The extension's mtime, UTC, to the second, or the empty
            string when absent. A build older than the source it wraps measures the
            old kernel and reports the new tree's name for it.
    """

    package_root: str
    driver_root: str
    same_tree: bool
    extension: str
    extension_stamp: str


def tree_provenance(driver: Path) -> TreeProvenance:
    """Record the tree the perf package was imported from, beside the driver's.

    A remote tree that accumulates files, or a ``PYTHONPATH`` pointing at a second
    checkout, measures one tree while the operator reads the scripts of another. The
    result is a green run about code nobody edited, which is the vacuous pass one
    layer below the coverage rule. Reported, not judged: a rule would need a
    declared expected tree, and nothing in this repo records one.

    Args:
        driver: ``__file__`` of the script that runs the audit. Its repository root
            is taken as two levels up, which is where ``scripts/perf/`` sits.

    Returns:
        The record. Every path is resolved, so a symlinked or relative invocation
        does not read as a mismatch.
    """
    package_root = Path(__file__).resolve().parents[1]
    driver_root = driver.resolve().parents[2]
    built = sorted(package_root.glob(EXTENSION_GLOB))
    stamp = ""
    if built:
        mtime = datetime.fromtimestamp(built[0].stat().st_mtime, tz=UTC)
        stamp = mtime.isoformat(timespec="seconds")
    return TreeProvenance(
        package_root=str(package_root),
        driver_root=str(driver_root),
        same_tree=package_root.parent == driver_root,
        extension=str(built[0]) if built else "absent",
        extension_stamp=stamp,
    )


@dataclass(frozen=True)
class RegistryChoice(PerfRecord):
    """Which backend one registry selected for the profiled device and dtype.

    Attributes:
        registry: Registry name, as :class:`slinoss._registry.Registry` was
            constructed with. An operator with three of them contributes three rows.
        backend: Name the registry resolved.
        is_reference: Whether that name is the reference implementation, which
            launches no kernel this repo compiles.
    """

    registry: str
    backend: str
    is_reference: bool


@dataclass(frozen=True)
class DispatchVerdict(PerfRecord):
    """Whether the profiled operator resolved to a kernel backend at all.

    Attributes:
        op: Operator name.
        device_type: Torch device type resolution was asked for.
        dtype: Activation dtype resolution was asked for, as its torch name.
        choices: One row per registry the operator selects through.
        reference_count: Registries that resolved to the reference.
        passed: True when no registry resolved to the reference.
        detail: Which registries fell back, or that none did.
    """

    op: str
    device_type: str
    dtype: str
    choices: tuple[RegistryChoice, ...]
    reference_count: Annotated[Count, INVARIANT]
    passed: bool
    detail: str


@dataclass(frozen=True)
class CoverageVerdict(PerfRecord):
    """Whether an audit judged every kernel its arm declares.

    Attributes:
        op: Operator name.
        mode: One of :data:`MODES`.
        required_count: Kernels the arm launches at every shape.
        judged_count: Distinct declared kernels the audit judged.
        missing: Required kernels the audit did not judge. Nonempty fails.
        absent: Conditional kernels the audit did not judge. Reported, not failed.
        unclaimed: Declared kernels the audit judged that this arm does not claim.
            Nonempty fails: the table and the workload disagree, and the table is
            what the missing check is measured against.
        narrowed: Whether an ``--kernel`` regex was in force. A narrowed capture
            holds a subset by construction, so completeness is not judged and only
            the vacuous case fails.
        passed: Whether the arm's coverage holds.
        detail: What was declared, what was judged, and what is missing.
    """

    op: str
    mode: str
    required_count: Annotated[Count, INVARIANT]
    judged_count: Annotated[Count, INVARIANT]
    missing: tuple[str, ...]
    absent: tuple[str, ...]
    unclaimed: tuple[str, ...]
    narrowed: bool
    passed: bool
    detail: str


def coverage_verdict(
    op: str,
    mode: str,
    judged: Sequence[str],
    *,
    narrowed: bool = False,
) -> CoverageVerdict:
    """Judge one audit's coverage against what its arm declares.

    Args:
        op: Operator name.
        mode: One of :data:`MODES`.
        judged: Kernel symbols the audit judged, as NCU reports them. Mapped to
            :data:`slinoss.perf.declared.DECLARED` keys, so two instantiations of one
            kernel count once and a symbol this repo does not compile counts not at
            all. Empty is the vacuous audit, which fails.
        narrowed: Whether the capture was narrowed to a kernel-name regex. A narrowed
            capture is judged only on the vacuous case and on ``unclaimed``.

    Returns:
        The verdict.

    Raises:
        KeyError: If ``(op, mode)`` has no :data:`COVERAGE` entry.
        ValueError: If a symbol in ``judged`` is one this repo compiles and matches no
            declared kernel, or matches two.
    """
    entry = coverage_of(op, mode)
    keys = {key for key in (declared_key(one) for one in judged) if key is not None}
    claimed = set(entry.kernels)
    missing = () if narrowed else tuple(k for k in entry.required if k not in keys)
    absent = tuple(c.kernel for c in entry.conditional if c.kernel not in keys)
    unclaimed = tuple(sorted(keys - claimed))
    vacuous = not keys
    passed = not vacuous and not missing and not unclaimed
    counted = Count(len(keys))
    required = Count(len(entry.required))
    if vacuous:
        detail = (
            f"judged nothing: {op} {mode} declares {required} kernels and the "
            f"capture held none of them; an audit with no verdict is not a pass"
        )
    elif missing:
        detail = f"judged {counted} of {required} declared: missing {list(missing)}"
    elif unclaimed:
        detail = (
            f"judged {list(unclaimed)}, which {op} {mode} does not declare; "
            f"the coverage table and the workload disagree"
        )
    elif narrowed:
        detail = f"narrowed capture: judged {counted}, completeness not judged"
    else:
        detail = f"judged every one of the {required} kernels {op} {mode} declares"
    if absent:
        conditions = "; ".join(
            f"{c.kernel} absent: {c.condition}"
            for c in entry.conditional
            if c.kernel in absent
        )
        detail = f"{detail}. {conditions}"
    return CoverageVerdict(
        op=op,
        mode=mode,
        required_count=required,
        judged_count=counted,
        missing=missing,
        absent=absent,
        unclaimed=unclaimed,
        narrowed=narrowed,
        passed=passed,
        detail=detail,
    )
