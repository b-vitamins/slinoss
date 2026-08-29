"""The published protocol: which datasets, which seeds, which settings, when to stop.

Every constant here is transcribed from the reference's own artifacts, not from a paper's
prose. The per-dataset settings are its ``experiment_configs/repeats/LinOSS/<Dataset>.json``,
the seeds and step budget are in those same files, and the bars in :data:`REFERENCE` are the
mean and population standard deviation of the five ``test_metric.npy`` files the reference
shipped under ``outputs/LinOSS_<discretization>/<Dataset>/``. A bar quoted from a table cannot
be checked; these can, and :mod:`tests.test_tsc_protocol` checks the arithmetic.

Three parts of the protocol are easy to get wrong and each changes the number:

    the rate is constant           ``lr_scheduler`` is ``lambda lr: lr``. No warmup, no decay,
                                   no clipping, Adam at its defaults.
    the metric is not the last     the reported point is the test accuracy at the last
                                   evaluation whose validation accuracy was at least the best
                                   seen. See :func:`scripts.tsc.train.train`.
    the budget is rarely spent     100,000 steps is a cap; early stopping ends most runs, and
                                   a harness that stops at, say, 10,000 measures a different
                                   quantity. That was the whole of one earlier gap.

``ssm_blocks`` and ``scale`` appear in those config files and reach no LinOSS code path. They
are not recorded here; carrying a dead field forward invites an arm to sweep it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

__all__ = [
    "DATASETS",
    "DISCRETIZATIONS",
    "DROP_RATE",
    "HORIZON",
    "NUM_STEPS",
    "PATIENCE",
    "PRINT_STEPS",
    "REFERENCE",
    "SEEDS",
    "Bar",
    "Setting",
    "setting_for",
]

SEEDS = (2345, 3456, 4567, 5678, 6789)
"""The five protocol seeds, in the order the reference lists them.

The seed fixes the partition and nothing else here: the reference's ``modelkey`` and
``trainkey`` are JAX's, and initialization and dropout are torch's in this harness. So a
reported mean is over five *partitions*, and the spread it carries is mostly the partition's,
not the optimizer's."""

NUM_STEPS = 100_000
"""Optimizer steps, as a cap. Early stopping ends most runs well inside it."""

PRINT_STEPS = 1000
"""Steps between evaluations. Also the early-stopping clock: patience counts evaluations."""

PATIENCE = 10
"""Evaluations without improvement that are tolerated.

The reference breaks on ``no_val_improvement > 10``, so the eleventh non-improving evaluation
ends the run and ten are survived."""

HORIZON = 1.0
"""The reference's ``T``. Every published config sets it to 1."""

DROP_RATE = 0.05
"""Dropout inside a block. The reference's default and no config overrides it."""

DISCRETIZATIONS = ("IM", "IMEX")
"""The two LinOSS discretizations. Which one a dataset uses is part of its setting."""


@dataclass(frozen=True)
class Setting:
    """One dataset's published settings.

    Attributes:
        dataset: Archive folder name.
        batch_size: Sequences per optimizer step.
        lr: Constant Adam rate.
        blocks: Residual blocks.
        hidden_dim: Stream width, the reference's ``H``.
        ssm_dim: State size per block, the reference's ``ssm_size`` and ``P``. This is the
            oscillator count, not a channel count: the scan carries ``2 * ssm_dim`` reals.
        discretization: ``IM`` or ``IMEX``.
        include_time: Whether the time ramp is prepended as channel 0.

    Raises:
        ValueError: On a non-positive size or an unknown discretization.
    """

    dataset: str
    batch_size: int
    lr: float
    blocks: int
    hidden_dim: int
    ssm_dim: int
    discretization: str
    include_time: bool

    def __post_init__(self) -> None:
        for name in ("batch_size", "blocks", "hidden_dim", "ssm_dim"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(
                    f"{self.dataset}: {name} must be positive, got {value}"
                )
        if self.lr <= 0.0:
            raise ValueError(f"{self.dataset}: lr must be positive, got {self.lr}")
        if self.discretization not in DISCRETIZATIONS:
            raise ValueError(
                f"{self.dataset}: discretization must be one of {DISCRETIZATIONS}, "
                f"got {self.discretization!r}"
            )


class Bar(NamedTuple):
    """A reference result, as its own artifacts report it.

    Attributes:
        mixer: Which reference arm, ``LinOSS_IM`` or ``LinOSS_IMEX``.
        mean: Mean test accuracy over the seeds, as a fraction.
        std: Population standard deviation over the seeds, ``ddof = 0``, which is what the
            reference's post-processing computes. A sample standard deviation would read
            about 12% larger at five seeds and would not match the published spread.
        seeds: How many runs the bar averages.
    """

    mixer: str
    mean: float
    std: float
    seeds: int


DATASETS: dict[str, Setting] = {
    "EigenWorms": Setting(
        dataset="EigenWorms",
        batch_size=4,
        lr=1e-3,
        blocks=2,
        hidden_dim=128,
        ssm_dim=64,
        discretization="IM",
        include_time=True,
    ),
    "EthanolConcentration": Setting(
        dataset="EthanolConcentration",
        batch_size=32,
        lr=1e-5,
        blocks=4,
        hidden_dim=16,
        ssm_dim=16,
        discretization="IM",
        include_time=False,
    ),
    "Heartbeat": Setting(
        dataset="Heartbeat",
        batch_size=32,
        lr=1e-3,
        blocks=6,
        hidden_dim=16,
        ssm_dim=16,
        discretization="IM",
        include_time=True,
    ),
    "MotorImagery": Setting(
        dataset="MotorImagery",
        batch_size=32,
        lr=1e-5,
        blocks=2,
        hidden_dim=128,
        ssm_dim=16,
        discretization="IM",
        include_time=False,
    ),
    "SelfRegulationSCP1": Setting(
        dataset="SelfRegulationSCP1",
        batch_size=32,
        lr=1e-4,
        blocks=6,
        hidden_dim=128,
        ssm_dim=256,
        discretization="IM",
        include_time=True,
    ),
    "SelfRegulationSCP2": Setting(
        dataset="SelfRegulationSCP2",
        batch_size=32,
        lr=1e-5,
        blocks=6,
        hidden_dim=64,
        ssm_dim=256,
        discretization="IMEX",
        include_time=True,
    ),
}
"""The six UEA datasets the protocol reports, at their published settings.

Insertion order is the reference's alphabetical order, which is the column order of every
table this harness produces."""

REFERENCE: dict[str, Bar] = {
    "EigenWorms": Bar("LinOSS_IM", 0.9500, 0.0444, 5),
    "EthanolConcentration": Bar("LinOSS_IM", 0.2987, 0.0062, 5),
    "Heartbeat": Bar("LinOSS_IM", 0.7581, 0.0368, 5),
    "MotorImagery": Bar("LinOSS_IM", 0.6000, 0.0748, 5),
    "SelfRegulationSCP1": Bar("LinOSS_IM", 0.8776, 0.0264, 5),
    "SelfRegulationSCP2": Bar("LinOSS_IMEX", 0.5895, 0.0812, 5),
}
"""What the reference measured, from the five per-seed files it shipped.

Each bar's ``mixer`` is the discretization that dataset's setting names, so these are the
numbers a re-run of the reference baseline through this harness has to land on. A run that
misses one is a harness failure until it is shown otherwise."""


def setting_for(dataset: str) -> Setting:
    """One dataset's published setting.

    Args:
        dataset: Archive folder name.

    Returns:
        The setting.

    Raises:
        KeyError: On a dataset outside the protocol, naming the six that are in it. A
            typo that fell through to a default setting would produce a number that looks
            comparable and is not.
    """
    if dataset not in DATASETS:
        raise KeyError(f"{dataset} is not a protocol dataset; have {sorted(DATASETS)}")
    return DATASETS[dataset]
