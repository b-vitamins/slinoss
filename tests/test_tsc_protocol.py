"""The published protocol, pinned so a drift in it is a test failure and not a modelling result.

Two things are checked and they are different in kind. The settings are transcribed constants, so
they are pinned field for field against the reference's own ``experiment_configs`` -- a wrong
``lr`` or a wrong ``include_time`` produces a number that looks comparable to a published bar and
is not.

The bars themselves are checked as arithmetic. :data:`PER_SEED` holds the five values the
reference shipped in ``outputs/LinOSS_<scheme>/<Dataset>/**/test_metric.npy``, and every mean and
spread in :data:`scripts.tsc.protocol.REFERENCE` has to be their mean and their *population*
standard deviation. A sample standard deviation reads about 12% larger at five seeds, so the
choice is not cosmetic: it decides whether a re-run of the baseline lands inside the bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.tsc.protocol import (
    DATASETS,
    DISCRETIZATIONS,
    HORIZON,
    NUM_STEPS,
    PATIENCE,
    PRINT_STEPS,
    REFERENCE,
    SEEDS,
    Setting,
    setting_for,
)

# dataset -> (batch_size, lr, blocks, hidden_dim, ssm_dim, discretization, include_time)
PUBLISHED = {
    "EigenWorms": (4, 1e-3, 2, 128, 64, "IM", True),
    "EthanolConcentration": (32, 1e-5, 4, 16, 16, "IM", False),
    "Heartbeat": (32, 1e-3, 6, 16, 16, "IM", True),
    "MotorImagery": (32, 1e-5, 2, 128, 16, "IM", False),
    "SelfRegulationSCP1": (32, 1e-4, 6, 128, 256, "IM", True),
    "SelfRegulationSCP2": (32, 1e-5, 6, 64, 256, "IMEX", True),
}

# The five ``test_metric.npy`` values the reference shipped, in SEEDS order. float32 values read
# back as float64, which is why the tolerance below is 5e-5 and not exact.
PER_SEED = {
    "EigenWorms": (1.0, 0.9166666865, 0.9444444776, 1.0, 0.8888888955),
    "EthanolConcentration": (
        0.3037974834,
        0.291139245,
        0.3037974834,
        0.291139245,
        0.3037974834,
    ),
    "Heartbeat": (
        0.7580645084,
        0.8064515591,
        0.7096773982,
        0.7903225422,
        0.7258064151,
    ),
    "MotorImagery": (
        0.5263158083,
        0.6140350699,
        0.7368420959,
        0.5438596606,
        0.5789473653,
    ),
    "SelfRegulationSCP1": (
        0.8823529482,
        0.9176470637,
        0.870588243,
        0.8823529482,
        0.8352941275,
    ),
    "SelfRegulationSCP2": (
        0.561403513,
        0.5263158083,
        0.6666666865,
        0.4912280738,
        0.7017543912,
    ),
}

TOLERANCE = 5e-5
"""What a bar rounded to four decimals can differ from the arithmetic by."""


def test_the_six_settings_are_the_published_ones() -> None:
    """Every field of every dataset's setting, against the reference's config files.

    These are the numbers that make a run comparable to a bar. A single wrong one -- most easily
    ``include_time``, which differs across the six -- gives a result that no published figure
    describes and nothing else would flag.
    """
    assert set(DATASETS) == set(PUBLISHED)
    for dataset, want in PUBLISHED.items():
        setting = setting_for(dataset)
        found = (
            setting.batch_size,
            setting.lr,
            setting.blocks,
            setting.hidden_dim,
            setting.ssm_dim,
            setting.discretization,
            setting.include_time,
        )
        assert found == want, dataset
        assert setting.dataset == dataset


def test_the_protocol_constants_are_the_published_ones() -> None:
    """Seeds, step budget, evaluation interval, patience and horizon.

    The budget is the one that was an earlier gap on its own: 100,000 steps is a cap and early
    stopping ends most runs, so a harness that stops at 10,000 measures a different quantity.
    """
    assert SEEDS == (2345, 3456, 4567, 5678, 6789)
    assert (NUM_STEPS, PRINT_STEPS, PATIENCE, HORIZON) == (100_000, 1000, 10, 1.0)
    assert DISCRETIZATIONS == ("IM", "IMEX")


def test_every_bar_is_the_mean_and_population_spread_of_its_five_runs() -> None:
    """Each bar reproduces the arithmetic over the reference's own per-seed files.

    ``ddof = 0``. At five seeds a sample standard deviation is 11.8% larger, which is enough to
    move a re-run from inside a bar to outside it, so the estimator is pinned and not assumed.
    """
    assert set(REFERENCE) == set(PUBLISHED)
    for dataset, bar in REFERENCE.items():
        values = np.asarray(PER_SEED[dataset], dtype=np.float64)
        assert values.size == bar.seeds == len(SEEDS)
        assert bar.mean == pytest.approx(values.mean(), abs=TOLERANCE), dataset
        assert bar.std == pytest.approx(values.std(ddof=0), abs=TOLERANCE), dataset
        # And the sample estimator is not what is recorded, wherever the two differ at all.
        if values.std(ddof=0) > TOLERANCE:
            assert bar.std != pytest.approx(values.std(ddof=1), abs=TOLERANCE), dataset


def test_each_bar_names_the_scheme_its_dataset_uses() -> None:
    """A bar's arm is the discretization that dataset's setting selects.

    Otherwise the recorded target for a dataset would be a different model's number: only
    SelfRegulationSCP2 is IMEX, and pairing it with the IM bar would set the wrong target on both.
    """
    for dataset, bar in REFERENCE.items():
        assert bar.mixer == f"LinOSS_{setting_for(dataset).discretization}", dataset


def test_a_dataset_outside_the_protocol_is_refused_by_name() -> None:
    """A typo does not fall through to a default setting.

    A default would run, finish, and report a number against a bar it has nothing to do with.
    """
    with pytest.raises(KeyError, match="SelfRegulationSCP3"):
        setting_for("SelfRegulationSCP3")


def test_a_setting_validates_on_construction() -> None:
    """A non-positive size, a non-positive rate or an unknown scheme is refused at the setting.

    This is where a swept value lands, so refusing here stops a bad sweep point before a model is
    built rather than at the first matmul.
    """
    fields = {
        "dataset": "Heartbeat",
        "batch_size": 32,
        "lr": 1e-3,
        "blocks": 6,
        "hidden_dim": 16,
        "ssm_dim": 16,
        "discretization": "IM",
        "include_time": True,
    }
    assert Setting(**fields).blocks == 6
    with pytest.raises(ValueError, match="blocks must be positive"):
        Setting(**{**fields, "blocks": 0})
    with pytest.raises(ValueError, match="lr must be positive"):
        Setting(**{**fields, "lr": 0.0})
    with pytest.raises(ValueError, match="discretization must be one of"):
        Setting(**{**fields, "discretization": "RK4"})
