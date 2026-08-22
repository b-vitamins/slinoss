"""The rule that an audit judged something, and the two escapes from it.

The audit in :mod:`slinoss.perf.declared` judges the kernels a capture held, so a
capture that held nothing passes every rule it applies. What is pinned here is the
table that says what a capture should have held, its completeness against the
declaration table, and the verdict's five outcomes.

No device and no operator package: the table is data, which is what lets the
completeness check run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slinoss.perf.coverage import (
    COVERAGE,
    MODES,
    TARGETED,
    coverage_of,
    coverage_verdict,
    tree_provenance,
    unreachable,
)
from slinoss.perf.declared import DECLARED
from slinoss.perf.workload import OPS

SCAN_STEP = ("so3ssd", "step")
"""The one arm with conditional kernels, so it exercises both branches."""


def symbol(key: str) -> str:
    """The symbol NCU reports for a declared kernel, under CuTe's mangling."""
    return f"kernel_cutlass_{key}_bf16_Ampere_0"


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


def test_every_benchmarked_arm_has_an_entry_and_names_only_declared_kernels() -> None:
    """A missing entry is an unauditable arm, and a stray name is a rule on nothing.

    ``coverage_verdict`` measures what an audit judged against this table, so an
    operator absent from it cannot be judged incomplete, and a kernel named here that
    no longer exists would demand a launch that can never happen.
    """
    assert set(COVERAGE) == {(op, mode) for op in OPS for mode in MODES}
    for (op, mode), entry in COVERAGE.items():
        assert entry.required, f"{op} {mode} requires nothing"
        for key in entry.kernels:
            assert key in DECLARED, f"{op} {mode} names undeclared {key}"
        assert len(set(entry.required)) == len(entry.required)
        # A conditional carries the shape property, because the report quotes it in
        # place of the launch and an unexplained absence is what the rule catches.
        for one in entry.conditional:
            assert one.kernel not in entry.required
            assert len(one.condition) > 20


def test_every_declared_kernel_is_reached_by_an_arm_or_by_a_named_driver() -> None:
    """The static half of the rule: a class no driver reaches is a claim.

    A kernel nothing launches is never judged, so its declared class reads as
    verified. Either an arm launches it or :data:`TARGETED` names the command that
    does, and the second is a line in every report so the excuse gets read.
    """
    assert unreachable() == ()
    claimed = {key for entry in COVERAGE.values() for key in entry.kernels}
    for one in TARGETED:
        assert one.kernel in DECLARED
        # An entry whose kernel an arm does launch is a contradiction: it would be
        # excused from the reachability check while the arm's audit required it.
        assert one.kernel not in claimed, f"{one.kernel} is reached by an arm"
        assert one.driver.startswith("scripts/perf/")
        assert len(one.reason) > 40
    assert len({one.kernel for one in TARGETED}) == len(TARGETED)


def test_coverage_of_refuses_an_arm_it_cannot_judge() -> None:
    # A driver offering a mode the table does not know would otherwise be audited
    # against nothing, which is the vacuous pass one level up.
    with pytest.raises(KeyError, match="no coverage entry"):
        coverage_of("so3ssd", "decode")
    assert coverage_of(*SCAN_STEP).conditional[0].kernel == "reduce_rows_kernel"


# ---------------------------------------------------------------------------
# coverage_verdict
# ---------------------------------------------------------------------------


def test_an_audit_that_judged_nothing_fails() -> None:
    """The whole point. Not a shortfall of one: a capture with no verdict in it.

    This is what a conv audit did on a tree with no compiled extension, and it exited
    zero. The detail names the count declared, because a reader seeing only "failed"
    would look for a slow kernel.
    """
    verdict = coverage_verdict(*SCAN_STEP, ())
    assert verdict.passed is False
    assert verdict.judged_count == 0
    assert verdict.required_count == len(coverage_of(*SCAN_STEP).required)
    assert "judged nothing" in verdict.detail
    assert "an audit with no verdict is not a pass" in verdict.detail
    # A capture of the reference's kernels is the same failure: torch's symbols map
    # to no declared kernel, so the count of verdicts is still zero.
    torch_only = coverage_verdict(
        *SCAN_STEP, ("void at::native::vectorized_elementwise_kernel<4>(int)",)
    )
    assert torch_only.passed is False
    assert torch_only.judged_count == 0
    # And a narrowed capture does not excuse it: narrowing selects a subset of the
    # launches, not none of them.
    assert coverage_verdict(*SCAN_STEP, (), narrowed=True).passed is False


def test_a_capture_short_by_one_launch_fails_and_names_what_is_missing() -> None:
    entry = coverage_of(*SCAN_STEP)
    verdict = coverage_verdict(*SCAN_STEP, [symbol(k) for k in entry.required[:-1]])
    assert verdict.passed is False
    assert verdict.missing == (entry.required[-1],)
    assert (
        f"judged {len(entry.required) - 1} of {len(entry.required)}" in verdict.detail
    )
    assert entry.required[-1] in verdict.detail


def test_a_full_capture_passes_and_two_instantiations_of_one_kernel_count_once() -> (
    None
):
    """The count is of declared kernels, not of symbols.

    A capture holds one counter row per symbol, and one kernel compiled for two
    dtypes is two symbols. Counting symbols would let a doubly instantiated kernel
    cover for a missing one.
    """
    entry = coverage_of(*SCAN_STEP)
    judged = [symbol(k) for k in entry.required]
    verdict = coverage_verdict(*SCAN_STEP, [*judged, judged[0].replace("bf16", "fp16")])
    assert verdict.passed is True
    assert verdict.judged_count == len(entry.required)
    assert verdict.missing == ()
    assert f"every one of the {len(entry.required)} kernels" in verdict.detail
    # Every conditional is absent from a required-only capture, and that is reported
    # with the condition rather than failed. Read off the table rather than spelled
    # out: naming the kernels here would fail this test, which is about counting
    # instantiations, whenever the table gains or loses a conditional.
    assert entry.conditional, "a table with no conditional makes the rest vacuous"
    assert verdict.absent == tuple(c.kernel for c in entry.conditional)
    for cond in entry.conditional:
        assert f"{cond.kernel} absent: {cond.condition}" in verdict.detail
    # Present, each is judged and counted, and the report says nothing about it. The
    # two conditions are separate shape properties, so one can hold without the other.
    conditional = [symbol(c.kernel) for c in entry.conditional]
    both = coverage_verdict(*SCAN_STEP, [*judged, *conditional])
    assert both.absent == ()
    assert both.judged_count == len(entry.required) + len(conditional)
    assert both.passed is True


def test_judging_a_kernel_the_arm_does_not_declare_fails_the_table_not_the_kernel() -> (
    None
):
    """The table is what the missing check is measured against, so it must be right.

    A launch the arm makes and the table does not name means the shortfall count is
    computed against a stale declaration. Failing here is what keeps the table
    honest as the operator changes; the alternative is a rule that quietly weakens.
    """
    entry = coverage_of("mixer", "step")
    judged = [symbol(k) for k in entry.required] + [symbol("boundary_bwd_kernel")]
    verdict = coverage_verdict("mixer", "step", judged)
    assert verdict.passed is False
    assert verdict.unclaimed == ("boundary_bwd_kernel",)
    assert "does not declare" in verdict.detail
    # Narrowing does not excuse it either: a regex selects fewer kernels, never a
    # kernel from another operator.
    assert coverage_verdict("mixer", "step", judged, narrowed=True).passed is False


def test_a_narrowed_capture_is_judged_on_the_vacuous_case_alone() -> None:
    # `--kernel` holds a subset by construction, so requiring completeness would fail
    # on the narrowing rather than on a defect. Something was still judged.
    entry = coverage_of(*SCAN_STEP)
    verdict = coverage_verdict(*SCAN_STEP, [symbol(entry.required[0])], narrowed=True)
    assert verdict.passed is True
    assert verdict.narrowed is True
    assert verdict.missing == ()
    assert "completeness not judged" in verdict.detail


# ---------------------------------------------------------------------------
# tree_provenance
# ---------------------------------------------------------------------------


def test_the_stamp_names_the_tree_the_package_was_imported_from() -> None:
    """Reported, not judged. A rule would need a declared expected tree.

    A remote directory that accumulates files, or a ``PYTHONPATH`` naming a second
    checkout, measures code nobody edited and reads clean. The stamp is what makes
    that readable after the fact.
    """
    driver = Path(__file__).resolve().parents[1] / "scripts" / "perf" / "profile_op.py"
    stamp = tree_provenance(driver)
    assert Path(stamp.package_root).name == "slinoss"
    assert stamp.driver_root == str(driver.resolve().parents[2])
    assert stamp.same_tree is True
    # A driver from another checkout is the mismatch, and it is reported as one.
    other = tree_provenance(Path("/elsewhere/scripts/perf/profile_op.py"))
    assert other.same_tree is False
    assert other.driver_root == "/elsewhere"
    # The extension is a filesystem question, not an import: what the tree holds is
    # the subject, and an import answers what sys.path found.
    if stamp.extension == "absent":
        assert stamp.extension_stamp == ""
    else:
        assert stamp.extension.startswith(stamp.package_root)
        assert stamp.extension_stamp.endswith("+00:00")
