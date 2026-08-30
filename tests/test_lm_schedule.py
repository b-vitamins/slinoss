"""The rate: the transfer factor, and the three points on the schedule that matter.

A rate is the one setting that is both shared across arms and derived per arm, so it is where a
comparison quietly stops being one. The transfer has to be a function of width and batch and
nothing else, and the schedule has to be exact at step zero, at the start of the warmdown, and
at the last step -- an off-by-one at either end changes the last few percent of training, which
is where the loss moves fastest.
"""

from __future__ import annotations

import math

import pytest

from scripts.lm.schedule import (
    REF_D_MODEL,
    REF_TOKEN_BATCH,
    lr_at,
    steps_for,
    transfer,
)

BASE = 4e-3
PEAK = 1.0
TOTAL = 100


def test_transfer_is_the_identity_at_the_reference_shape() -> None:
    """A rate tuned at the reference width and batch is carried unchanged."""
    assert transfer(
        BASE, d_model=REF_D_MODEL, token_batch=REF_TOKEN_BATCH
    ) == pytest.approx(BASE)


def test_transfer_falls_with_width_and_rises_with_batch() -> None:
    """``sqrt(ref/d_model) * sqrt(batch/ref)``, each factor independently.

    Both directions are checked because a sign error in either would still leave a
    plausible-looking number at every shape actually run.
    """
    wider = transfer(BASE, d_model=4 * REF_D_MODEL, token_batch=REF_TOKEN_BATCH)
    assert wider == pytest.approx(BASE / 2)
    bigger = transfer(BASE, d_model=REF_D_MODEL, token_batch=4 * REF_TOKEN_BATCH)
    assert bigger == pytest.approx(BASE * 2)
    both = transfer(BASE, d_model=4 * REF_D_MODEL, token_batch=4 * REF_TOKEN_BATCH)
    assert both == pytest.approx(BASE)


def test_transfer_refuses_a_non_positive_shape() -> None:
    """A zero width would divide, and a zero batch would report a zero rate."""
    with pytest.raises(ValueError, match="d_model must be positive"):
        transfer(BASE, d_model=0, token_batch=REF_TOKEN_BATCH)
    with pytest.raises(ValueError, match="token_batch must be positive"):
        transfer(BASE, d_model=REF_D_MODEL, token_batch=0)


def test_the_constant_phase_holds_the_peak() -> None:
    """No warmup: step zero is at the peak, and so is every step before the warmdown."""
    for step in (0, 1, 59):
        assert lr_at(
            step, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=0.4
        ) == pytest.approx(PEAK)


def test_the_warmdown_starts_where_the_fraction_says() -> None:
    """At ``(1 - fraction) * total`` the fall begins, and the first falling step is the peak.

    The schedule is linear from the peak at the hold point to the floor at ``total_steps``,
    so the boundary step is the peak by construction. An off-by-one here shortens or
    lengthens the fall by a step at every scale.
    """
    hold = round(TOTAL * 0.6)
    assert lr_at(
        hold, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=0.4
    ) == pytest.approx(PEAK)
    half = hold + (TOTAL - hold) // 2
    assert lr_at(
        half, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=0.4
    ) == pytest.approx(PEAK / 2)


def test_the_last_step_is_above_the_floor_and_the_floor_is_one_past_it() -> None:
    """The floor is reached at ``total_steps``, which is one past the last step taken.

    A schedule that hit zero on the last step would spend that step not training.
    """
    last = lr_at(TOTAL - 1, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=0.4)
    assert 0.0 < last < PEAK
    assert last == pytest.approx(PEAK / (TOTAL - round(TOTAL * 0.6)))
    assert lr_at(TOTAL, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=0.4) == 0.0


def test_a_floor_fraction_lands_at_the_floor() -> None:
    """``final_fraction`` sets where the fall ends, not where it starts."""
    assert lr_at(
        TOTAL,
        total_steps=TOTAL,
        peak_lr=PEAK,
        warmdown_fraction=0.4,
        final_fraction=0.1,
    ) == pytest.approx(0.1)
    assert lr_at(
        TOTAL + 50,
        total_steps=TOTAL,
        peak_lr=PEAK,
        warmdown_fraction=0.4,
        final_fraction=0.1,
    ) == pytest.approx(0.1)


def test_a_full_warmdown_falls_from_the_first_step() -> None:
    """``warmdown_fraction`` of one is a linear decay over the whole run."""
    assert lr_at(
        0, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=1.0
    ) == pytest.approx(PEAK)
    assert lr_at(
        TOTAL // 2, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=1.0
    ) == pytest.approx(PEAK / 2)


def test_the_schedule_refuses_a_fraction_outside_the_unit_interval() -> None:
    """A fraction over one would put the hold point before step zero."""
    for fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="must be in"):
            lr_at(0, total_steps=TOTAL, peak_lr=PEAK, warmdown_fraction=fraction)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        lr_at(0, total_steps=0, peak_lr=PEAK, warmdown_fraction=0.4)
    with pytest.raises(ValueError, match="peak_lr must be positive"):
        lr_at(0, total_steps=TOTAL, peak_lr=0.0, warmdown_fraction=0.4)


def test_steps_for_rounds_down_and_refuses_a_partial_step() -> None:
    """The budget is what is held fixed across arms, so a fractional step is not taken."""
    assert steps_for(1000, 100) == 10
    assert steps_for(1099, 100) == 10
    with pytest.raises(ValueError, match="under one step"):
        steps_for(99, 100)
    with pytest.raises(ValueError, match="must be positive"):
        steps_for(0, 100)


def test_the_published_shape_gives_the_published_step_count() -> None:
    """The protocol's two scales, as step counts, from the numbers as written.

    1.8B tokens at a 131,072-token batch and 10.9B at 524,288. Checked so a typo in either
    constant shows up as a step count nobody recognizes rather than as a shorter run.
    """
    assert steps_for(1_800_000_000, 1 << 17) == 13732
    assert steps_for(10_900_000_000, 1 << 19) == 20790
    assert math.isclose(
        transfer(BASE, d_model=496, token_batch=1 << 17),
        BASE * math.sqrt(768 / 496) * 0.5,
    )
