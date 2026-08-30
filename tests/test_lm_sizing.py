"""The width solver: the nearest grid point, and the gate that refuses mismatched arms.

Sizing is the one step whose failure is invisible in the output. A solver that returned the
grid point below the target rather than the nearest would size every arm small by up to one
grid step, consistently, and the table would still print. A solver that converged on a
non-monotone counter would return a width that depended on the bisection's path.

So the counter here is analytic, not a model: the property under test is the search, and a real
build would put a mixer's parameter shapes between the assertion and the arithmetic. Whether
the count off a real model is monotone is the counter's problem, and
:func:`scripts.lm.sizing.solve_width` checks it rather than assuming it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from scripts.lm.sizing import (
    Sizing,
    check_spread,
    solve_width,
    spread,
)

MULTIPLE = 64


def _linear(scale: int = 1000) -> tuple[list[int], Callable[[int], int]]:
    """A counter that is ``scale * d_model``, and the widths it was called at."""
    seen: list[int] = []

    def count(d_model: int) -> int:
        seen.append(d_model)
        return scale * d_model

    return seen, count


def test_the_solver_returns_the_nearer_grid_point_not_the_lower_one() -> None:
    """A target between two grid points goes to whichever is closer, over or under.

    Always rounding down would size every arm small by up to a grid step and the bias would
    be the same for every arm, so no comparison would look wrong.
    """
    _, count = _linear()
    below = solve_width(1000 * 64 + 1000 * 20, count, high=1024, multiple=MULTIPLE)
    assert below == 64
    above = solve_width(1000 * 64 + 1000 * 44, count, high=1024, multiple=MULTIPLE)
    assert above == 128


def test_a_target_on_a_grid_point_is_hit_exactly() -> None:
    """No off-by-one at the fixed points of the search."""
    _, count = _linear()
    for width in (64, 256, 1024):
        assert solve_width(1000 * width, count, high=1024, multiple=MULTIPLE) == width


def test_the_solver_is_logarithmic_in_the_grid() -> None:
    """Each call may build a model, so the call count is part of the contract.

    A search that walked the grid would build 64 models per arm at this bracket.
    """
    seen, count = _linear()
    solve_width(1000 * 2000, count, high=4096, multiple=MULTIPLE)
    assert len(seen) <= 12


def test_a_target_outside_the_bracket_clamps_to_its_end() -> None:
    """A target the grid cannot reach returns the nearest end rather than raising.

    The bracket is a legality bound, not a preference: a target under the smallest legal
    width has no answer but the smallest legal width, and reporting it lets the achieved
    count and its error say so.
    """
    _, count = _linear()
    assert solve_width(1, count, high=1024, multiple=MULTIPLE) == 64
    assert solve_width(10**12, count, high=1024, multiple=MULTIPLE) == 1024


def test_a_non_monotone_counter_is_refused_rather_than_converged_on() -> None:
    """Bisection on a non-monotone count returns a path-dependent width.

    The likely cause is a mixer setting that scales with the width instead of being held
    fixed, which would also make the arm a different arm at every width.
    """

    def flat(d_model: int) -> int:
        del d_model
        return 1000

    with pytest.raises(ValueError, match="the search needs it increasing"):
        solve_width(500, flat, high=1024, multiple=MULTIPLE)

    def falling(d_model: int) -> int:
        return -d_model

    with pytest.raises(ValueError, match="the search needs it increasing"):
        solve_width(500, falling, high=1024, multiple=MULTIPLE)


def test_the_solver_refuses_an_empty_or_illegal_bracket() -> None:
    """A bracket with no grid point in it would bisect over nothing."""
    _, count = _linear()
    with pytest.raises(ValueError, match="no multiple of 64 in"):
        solve_width(1000, count, low=65, high=127, multiple=MULTIPLE)
    with pytest.raises(ValueError, match="target must be positive"):
        solve_width(0, count, high=1024, multiple=MULTIPLE)
    with pytest.raises(ValueError, match="multiple must be positive"):
        solve_width(1000, count, high=1024, multiple=0)


def test_the_sizing_reports_the_achieved_count_and_its_signed_error() -> None:
    """The record carries what was built, not what was asked for.

    A record that carried the target would make every arm look exactly matched, which is the
    one thing the spread gate exists to check.
    """
    over = Sizing("a", 128, 12, 105, 100)
    under = Sizing("b", 64, 12, 95, 100)
    assert over.error == pytest.approx(0.05)
    assert under.error == pytest.approx(-0.05)


def test_the_spread_is_zero_for_one_arm_and_relative_for_more() -> None:
    """``(max - min) / mean``, so the tolerance is readable as a percentage."""
    assert spread({"a": 1000}) == 0.0
    assert spread({"a": 990, "b": 1010}) == pytest.approx(20 / 1000)
    with pytest.raises(ValueError, match="at least one arm"):
        spread({})


def test_the_gate_passes_matched_arms_and_names_the_offenders_otherwise() -> None:
    """The gate is what stops a table of differently sized columns from printing.

    The message names the widest and the narrowest arm, because the action is to re-size one
    of them and the search's own bracket is what has to move.
    """
    counts = {"slinoss": 44_800_000, "mamba2": 45_100_000, "gdn2": 44_950_000}
    assert check_spread(counts) == pytest.approx(300_000 / 44_950_000, rel=1e-3)
    counts["gdn2"] = 60_000_000
    with pytest.raises(ValueError, match=r"gdn2 at 60000000 and slinoss at 44800000"):
        check_spread(counts)


def test_the_tolerance_boundary_passes() -> None:
    """A spread exactly at the tolerance is allowed; only over it is refused."""
    assert check_spread({"a": 99, "b": 101}, tolerance=0.02) == pytest.approx(0.02)
    with pytest.raises(ValueError, match="is over"):
        check_spread({"a": 99, "b": 101}, tolerance=0.0199)
