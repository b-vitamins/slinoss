"""The sweep: one enumeration, cost-balanced slices, and a harvest that refuses to mix plans.

A sweep on this axis is run by several hosts at once and nothing coordinates them at runtime, so
every property that makes the result trustworthy has to be a property of the enumeration. Three of
them are tested here and each one is a way a parallel sweep silently reports the wrong thing.

The balance test asserts a bound that is a theorem rather than a measured threshold. Longest-
processing-time-first puts the maximum load within ``4/3`` of the optimum, and the optimum is at
least both the mean load and the largest single point, so ``max <= 4/3 * max(mean, largest)`` holds
for any lattice. Block and stride slicing violate it on the real six-dataset plan, which is the
point: a point on EigenWorms costs two orders of magnitude more than one on Heartbeat, so slicing
by count puts one host on a week and another on an hour.

The weighted case gets the other invariant LPT gives: the slice ratios ``load / weight`` cannot
differ by more than one point's cost over the smallest weight, because a slice only ever receives
a point while its ratio is the lowest.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.tsc.corpus import read_manifest
from scripts.tsc.protocol import DATASETS, SEEDS
from scripts.tsc.sweep import (
    LENGTHS,
    SCAFFOLD_KEYS,
    Axis,
    Lattice,
    Point,
    assign,
    cost,
    merge,
    plan_digest,
    points,
    setting_for_point,
    shard,
)

CORPUS_VARIABLE = "SLINOSS_TSC_CORPUS"
"""Environment variable naming a processed corpus root, for the one test that needs real data."""

SMALL = Lattice(
    datasets=("Heartbeat", "MotorImagery"),
    mixers=("linoss_im", "conv"),
    seeds=(2345, 3456),
    axes=(Axis.parse("blocks=2,4"),),
)


def loads_of(buckets: Sequence[Sequence[Point]], num_steps: int) -> list[float]:
    """Total estimated cost per slice.

    Args:
        buckets: From :func:`scripts.tsc.sweep.assign`.
        num_steps: Step cap the estimate used.

    Returns:
        One load per slice.
    """
    return [sum(cost(point, num_steps) for point in bucket) for bucket in buckets]


def test_the_lattice_enumerates_in_one_canonical_order_with_seeds_innermost() -> None:
    """dataset, then mixer, then each axis in its listed order, then seed.

    Seeds vary fastest so a truncated read of a plan still shows whole cells. The order is also
    what ``position`` means and what the plan digest covers, so it is pinned rather than left to
    ``itertools``.
    """
    found = points(SMALL)
    assert len(found) == 2 * 2 * 2 * 2
    assert [point.position for point in found] == list(range(16))
    assert len({point.key for point in found}) == 16
    assert [
        (point.dataset, point.mixer, point.scaffold[0][1], point.seed)
        for point in found[:6]
    ] == [
        ("Heartbeat", "linoss_im", "2", 2345),
        ("Heartbeat", "linoss_im", "2", 3456),
        ("Heartbeat", "linoss_im", "4", 2345),
        ("Heartbeat", "linoss_im", "4", 3456),
        ("Heartbeat", "conv", "2", 2345),
        ("Heartbeat", "conv", "2", 3456),
    ]


def test_a_points_key_survives_a_reordered_lattice_but_the_plan_digest_does_not() -> (
    None
):
    """The key addresses a run; the digest identifies the plan that produced it.

    Both are needed and they are not the same thing. A key that moved with axis order would make a
    record unfindable after a lattice was rewritten; a digest that did not move would let two
    hosts agree on ``--shard 3/8`` while disagreeing about what shard 3 contains.
    """
    forward = Lattice(
        datasets=("Heartbeat",),
        axes=(Axis.parse("blocks=2,4"), Axis.parse("hidden_dim=16,32")),
    )
    backward = Lattice(
        datasets=("Heartbeat",),
        axes=(Axis.parse("hidden_dim=16,32"), Axis.parse("blocks=2,4")),
    )
    assert {point.key for point in points(forward)} == {
        point.key for point in points(backward)
    }
    assert plan_digest(forward) != plan_digest(backward)


def test_a_fixed_override_belongs_to_the_plan_and_to_every_key() -> None:
    """It changes what every number in the sweep means, so it separates the keys too.

    Otherwise a run at ``lr=1e-4`` and the same run at the published rate would write to the same
    record file, and the second would be taken for the first by ``--skip-existing``.
    """
    plain = Lattice(datasets=("Heartbeat",), seeds=(2345,))
    held = Lattice(datasets=("Heartbeat",), seeds=(2345,), fixed=("lr=1e-4",))
    assert plan_digest(plain) != plan_digest(held)
    assert points(plain)[0].key != points(held)[0].key
    assert points(held)[0].scaffold == (("lr", "1e-4"),)
    assert setting_for_point(points(held)[0]).lr == 1e-4
    # A mixer override routes to the settings list instead, ready for the registry.
    mixed = Lattice(datasets=("Heartbeat",), seeds=(2345,), fixed=("ssm_dim=32",))
    assert points(mixed)[0].settings == ("ssm_dim=32",)
    assert points(mixed)[0].scaffold == ()


def test_a_swept_value_is_read_at_the_type_of_the_field_it_replaces() -> None:
    """Text in, the setting's own type out, and a value the type refuses stops the point.

    There is no type table in the sweep module: a scaffold key's type comes from
    :class:`scripts.tsc.protocol.Setting` and a mixer key's from the registry defaults, so nothing
    here can fall out of date. ``blocks=0`` is refused by the setting's own validation, before a
    model is built.
    """
    lattice = Lattice(
        datasets=("SelfRegulationSCP2",),
        seeds=(2345,),
        fixed=("include_time=false", "blocks=3", "lr=5e-4"),
    )
    setting = setting_for_point(points(lattice)[0])
    assert (setting.include_time, setting.blocks, setting.lr) == (False, 3, 5e-4)
    bad = Lattice(datasets=("Heartbeat",), seeds=(2345,), fixed=("blocks=0",))
    with pytest.raises(ValueError, match="blocks must be positive"):
        setting_for_point(points(bad)[0])
    worse = Lattice(datasets=("Heartbeat",), seeds=(2345,), fixed=("blocks=many",))
    with pytest.raises(ValueError, match="blocks is int"):
        setting_for_point(points(worse)[0])


def test_an_axis_specification_is_validated_where_it_is_written() -> None:
    """Malformed, empty, repeated, or naming a scaffold field that does not exist.

    An axis is the one thing a person types by hand for every sweep, and a repeated value would
    quietly double a cell's cost while a misspelled scaffold key would be sent to the mixer.
    """
    assert Axis.parse("lr=1e-3,1e-4") == Axis("lr", ("1e-3", "1e-4"), True)
    assert set(SCAFFOLD_KEYS) == {
        "batch_size",
        "lr",
        "blocks",
        "hidden_dim",
        "include_time",
    }
    with pytest.raises(ValueError, match="must be key=v1,v2"):
        Axis.parse("lr")
    with pytest.raises(ValueError, match="repeats a value"):
        Axis.parse("lr=1e-3,1e-3")
    with pytest.raises(ValueError, match="has no values"):
        Axis("lr", (), True)
    with pytest.raises(ValueError, match="hidden is not a scaffold setting"):
        Axis("hidden", ("16",), True)


def test_a_published_setting_the_scaffold_never_reads_is_routed_to_the_mixer() -> None:
    """``ssm_dim`` and ``discretization`` are recorded per dataset and consumed by neither.

    This is the one silent-duplicate failure the sweep can produce on its own. The scaffold reads
    five of :class:`scripts.tsc.protocol.Setting`'s fields; the published state width reaches a
    mixer through :func:`scripts.tsc.mixers.paper_overrides` and the scheme is spelled by the
    registry name. Routed as scaffold keys, both would sweep: two values, two keys, two plan
    digests, and one program run twice -- reported as a width or a scheme that does not matter.
    Routed to the mixer, one reaches the mixer that has it and the other is refused by name.
    """
    assert {"ssm_dim", "discretization"} & set(SCAFFOLD_KEYS) == set()
    assert Axis.parse("ssm_dim=16,64").scaffold is False
    swept = Lattice(
        datasets=("Heartbeat",), seeds=(2345,), axes=(Axis.parse("ssm_dim=16,32"),)
    )
    assert [point.settings for point in points(swept)] == [
        ("ssm_dim=16",),
        ("ssm_dim=32",),
    ]
    assert {point.scaffold for point in points(swept)} == {()}
    # The published value is still the default, and the sweep's value follows it and wins.
    assert cost(points(swept)[1], 1) > cost(points(swept)[0], 1)
    with pytest.raises(ValueError, match="linoss_im has no setting discretization"):
        Lattice(fixed=("discretization=IMEX",))
    with pytest.raises(ValueError, match="conv has no setting ssm_dim"):
        Lattice(mixers=("conv",), fixed=("ssm_dim=32",))


def test_a_lattice_that_cannot_run_is_refused_before_a_single_lane_launches() -> None:
    """Every whole-plan mistake, at enumeration, on the host that plans rather than the ones
    that run.

    The mixer-setting check is the one that pays for itself: a sweep over ``ssm_dim`` that lists
    ``conv`` among its mixers is a plausible command line, and without this it would launch, run
    every ``conv`` point to completion and fail on none of them.
    """
    with pytest.raises(KeyError, match="Nonesuch"):
        Lattice(datasets=("Nonesuch",))
    with pytest.raises(KeyError, match="no tsc mixer nonesuch"):
        Lattice(mixers=("nonesuch",))
    with pytest.raises(ValueError, match="conv has no setting ssm_dim"):
        Lattice(mixers=("linoss_im", "conv"), axes=(Axis.parse("ssm_dim=16,32"),))
    with pytest.raises(ValueError, match=r"\['lr'\] is both fixed and swept"):
        Lattice(fixed=("lr=1e-3",), axes=(Axis.parse("lr=1e-4,1e-5"),))
    with pytest.raises(ValueError, match="fixes a key twice"):
        Lattice(fixed=("lr=1e-3", "lr=1e-4"))
    with pytest.raises(ValueError, match="repeats a dataset"):
        Lattice(datasets=("Heartbeat", "Heartbeat"))
    with pytest.raises(ValueError, match="lattice has no seeds"):
        Lattice(seeds=())
    with pytest.raises(ValueError, match="must be key=value"):
        Lattice(fixed=("lr",))
    with pytest.raises(ValueError, match="num_steps must be positive"):
        Lattice(num_steps=0)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8, 30])
def test_the_slices_are_a_disjoint_cover_balanced_within_the_lpt_bound(
    count: int,
) -> None:
    """Every point exactly once, and a maximum load inside ``4/3 * max(mean, largest)``.

    The bound is LPT's guarantee against an optimum that is itself at least the mean and at least
    the largest single point, so it holds for any lattice and needs no threshold to be tuned. The
    real plan is what it is asserted on, because that is where the two-orders-of-magnitude spread
    across the six datasets lives -- block and stride slicing both fail this test on it.
    """
    plan = Lattice()
    every = points(plan)
    assert len(every) == len(DATASETS) * len(SEEDS)
    buckets = assign(every, count, num_steps=plan.num_steps)
    assert sorted(point.key for bucket in buckets for point in bucket) == sorted(
        point.key for point in every
    )
    loads = loads_of(buckets, plan.num_steps)
    largest = max(cost(point, plan.num_steps) for point in every)
    assert max(loads) <= (4.0 / 3.0) * max(sum(loads) / count, largest) + 1e-6
    # Most expensive first, so a host killed early has done the work that fits nowhere else.
    for bucket in buckets:
        costs = [cost(point, plan.num_steps) for point in bucket]
        assert costs == sorted(costs, reverse=True)


def test_an_uneven_fleet_takes_work_in_proportion_to_its_weights() -> None:
    """A card twice as fast takes twice the cost, from one list and no per-host configuration.

    The assertion is LPT's own invariant: a slice only receives a point while its ``load / weight``
    is the lowest, so the final ratios cannot differ by more than one point's cost over the
    smallest weight. That is exact, unlike any ratio a fleet's real speeds would suggest.
    """
    plan = Lattice()
    every = points(plan)
    weights = [1.0, 3.0, 0.5]
    buckets = assign(every, 3, num_steps=plan.num_steps, weights=weights)
    loads = loads_of(buckets, plan.num_steps)
    ratios = [load / weight for load, weight in zip(loads, weights, strict=True)]
    largest = max(cost(point, plan.num_steps) for point in every)
    assert max(ratios) - min(ratios) <= largest / min(weights) + 1e-6
    assert loads[1] > loads[0] > loads[2]
    with pytest.raises(ValueError, match="2 weights for 3 slices"):
        assign(every, 3, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="weights must be positive"):
        assign(every, 2, weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="count must be positive"):
        assign(every, 0)


def test_a_shard_carries_the_plan_digest_and_the_slices_tile_the_plan() -> None:
    """``shard(lattice, i, n)`` for every ``i`` is the whole plan, once, with one digest.

    This is what a worker is handed and what it writes next to its results, so
    :func:`scripts.tsc.sweep.merge` can refuse a harvest whose shards came from two lattices.
    """
    plan = Lattice(datasets=("Heartbeat", "EigenWorms"))
    slices = [shard(plan, index, 4) for index in range(4)]
    assert {piece.plan for piece in slices} == {plan_digest(plan)}
    assert sorted(point.key for piece in slices for point in piece.points) == sorted(
        point.key for point in points(plan)
    )
    with pytest.raises(ValueError, match=r"shard 4 is outside 0..3"):
        shard(plan, 4, 4)
    with pytest.raises(ValueError, match=r"shard -1 is outside"):
        shard(plan, -1, 4)


def test_a_harvest_that_spans_two_plans_is_refused_and_a_rerun_is_dropped() -> None:
    """Sorted by key, one plan, and no key with two different results.

    The dangerous case is two hosts running ``--shard 3/8`` against lattices that differ by one
    fixed override: both produce well-formed records, and merging them would report a mix of two
    experiments as one.
    """
    first = {"plan": "aaa", "key": "b", "test_accuracy": 0.5}
    second = {"plan": "aaa", "key": "a", "test_accuracy": 0.25}
    assert merge([first, second]) == [second, first]
    assert merge([first, second, dict(first)]) == [second, first]
    with pytest.raises(ValueError, match="records span two plans"):
        merge([first, {**second, "plan": "bbb"}])
    with pytest.raises(ValueError, match="two different results for b"):
        merge([first, {**first, "test_accuracy": 0.75}])
    with pytest.raises(ValueError, match="record carries no plan"):
        merge([{"key": "a"}])
    with pytest.raises(ValueError, match="record carries no key"):
        merge([{"plan": "aaa"}])


def test_the_recorded_lengths_match_a_processed_corpus() -> None:
    """:data:`scripts.tsc.sweep.LENGTHS` against the manifests, when real data is present.

    The lengths are archive facts used for the cost estimate only, so a stale one can unbalance a
    sweep and can never change a number -- which is why this is the one test here that skips when
    the corpus is absent instead of carrying a fixture.
    """
    root = os.environ.get(CORPUS_VARIABLE)
    if not root or not Path(root).is_dir():
        pytest.skip(f"{CORPUS_VARIABLE} does not name a processed corpus")
    assert set(LENGTHS) == set(DATASETS)
    for dataset, length in sorted(LENGTHS.items()):
        folder = Path(root) / dataset
        if not folder.is_dir():
            pytest.skip(f"{dataset} is not processed under {root}")
        assert read_manifest(folder).length == length, dataset
