"""Enumerate a sweep once, globally, then hand each host a slice of it.

A sweep here is a lattice: datasets x mixers x seeds x a value list per swept key. It is
enumerated deterministically in one place, and a worker is told which slice to run. No host
decides what to run; a host is told, and every host can check it was told the same plan.

Three properties, and each is a failure this shape prevents.

    one enumeration, one digest    :func:`plan_digest` hashes the whole lattice. A shard writes
                                   it with its results and :func:`merge` refuses results from
                                   two different plans. Two hosts disagreeing about the lattice
                                   while agreeing about ``--shard 3/8`` is the way a sweep
                                   silently reports the wrong cell.
    cost-balanced slices           A point on EigenWorms costs about two orders of magnitude
                                   more than one on EthanolConcentration, so slicing by count
                                   puts one host on a week and another on an hour.
                                   :func:`assign` orders points by an explicit cost estimate
                                   and fills the slice with the least load first.
    unequal hosts                  ``weights`` scales each slice's capacity, so a card that is
                                   twice as fast takes twice the work. One list, no per-host
                                   configuration anywhere in the tree.

The slice is not a contiguous block and not a stride. Both align with the lattice's own shape:
at eight shards over six datasets, a stride hands most of one dataset to one host. Cost-ordered
greedy filling has no such alignment.

Every value in a lattice is text. Types come from where the setting is defined -- a scaffold key
from :class:`scripts.tsc.protocol.Setting`, a mixer key from that mixer's registry defaults --
so this module has no type table to fall out of date.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

from scripts.tsc.mixers import REGISTRY
from scripts.tsc.protocol import DATASETS, NUM_STEPS, SEEDS, Setting, setting_for

__all__ = [
    "LENGTHS",
    "SCAFFOLD_KEYS",
    "Axis",
    "Lattice",
    "Point",
    "Shard",
    "assign",
    "cost",
    "merge",
    "plan_digest",
    "points",
    "setting_for_point",
    "shard",
]

_MIXER_OWNED = frozenset({"ssm_dim", "discretization"})
"""Published settings the scaffold never reads, so a sweep must route them to the mixer.

Both are recorded per dataset in :class:`scripts.tsc.protocol.Setting` because the reference's
config files record them, and neither reaches anything through the setting: the published
``ssm_dim`` is handed to a mixer by :func:`scripts.tsc.mixers.paper_overrides`, and the
discretization is spelled by the registry name, ``linoss_im`` or ``linoss_imex``. Left in
:data:`SCAFFOLD_KEYS` they would be sweepable and inert -- two keys, two plan digests, one
program -- and the sweep would report a width or a scheme as having no effect when it was never
varied. As mixer keys they reach the mixer that has them and are refused by name for a mixer that
does not."""

SCAFFOLD_KEYS = tuple(
    field.name
    for field in fields(Setting)
    if field.name != "dataset" and field.name not in _MIXER_OWNED
)
"""Keys a sweep may move that belong to the scaffold rather than to a mixer.

Read off :class:`scripts.tsc.protocol.Setting` so a field added there is sweepable without an
edit here, and so a typo is refused against the real field list. A field the scaffold does not
consume belongs in :data:`_MIXER_OWNED` instead."""

_DIGEST_WIDTH = 8


def _coerce(key: str, default: Any, text: str) -> Any:
    """Read a swept value at the type of the setting it overrides.

    Args:
        key: The setting, for the message.
        default: Its default, whose type is the target.
        text: The value as the command line spelled it.

    Returns:
        The value.

    Raises:
        ValueError: On text the type does not read.
    """
    if isinstance(default, bool):
        lowered = text.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"{key} is a flag, got {text!r}")
    for kind in (int, float, str):
        if isinstance(default, kind):
            try:
                return kind(text)
            except ValueError as exc:
                raise ValueError(f"{key} is {kind.__name__}, got {text!r}") from exc
    raise ValueError(f"{key} has no rule for {type(default).__name__}")


@dataclass(frozen=True)
class Axis:
    """One swept key and its values.

    Attributes:
        key: The setting's name.
        values: Its values, as text, in the order the lattice enumerates them.
        scaffold: True when the key is a :class:`scripts.tsc.protocol.Setting` field, False
            when it is a mixer setting. Which one it is decides where the value goes and what
            type it is read at, so it is recorded rather than guessed at use.

    Raises:
        ValueError: On an empty value list, on a duplicate value, or on a scaffold key that is
            not a settable field.
    """

    key: str
    values: tuple[str, ...]
    scaffold: bool

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"axis {self.key} has no values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"axis {self.key} repeats a value: {list(self.values)}")
        if self.scaffold and self.key not in SCAFFOLD_KEYS:
            raise ValueError(
                f"{self.key} is not a scaffold setting; have {list(SCAFFOLD_KEYS)}"
            )

    @classmethod
    def parse(cls, spec: str) -> Axis:
        """Read a ``key=v1,v2,v3`` specification.

        Whether the key is a scaffold key or a mixer key is decided by
        :data:`SCAFFOLD_KEYS`, so ``lr`` reaches the loop and ``ssm_dim`` reaches the mixer
        with no prefix to remember and no way to send one to the wrong place.

        Args:
            spec: The specification.

        Returns:
            The axis.

        Raises:
            ValueError: On a specification with no ``=``, or from :meth:`__post_init__`.
        """
        key, sep, body = spec.partition("=")
        if not sep:
            raise ValueError(f"sweep axis must be key=v1,v2, got {spec!r}")
        key = key.strip()
        values = tuple(part.strip() for part in body.split(","))
        return cls(key, values, key in SCAFFOLD_KEYS)


@dataclass(frozen=True)
class Lattice:
    """A whole sweep.

    Attributes:
        datasets: Protocol dataset names.
        mixers: Registry names.
        seeds: Protocol seeds.
        axes: The swept keys. Order is the enumeration's, so it is part of the plan digest.
        fixed: ``key=value`` overrides applied to every point, scaffold or mixer, classified by
            :data:`SCAFFOLD_KEYS` the same way an axis is. A fixed override belongs to the plan
            and not to the invocation: it changes what every number in the sweep means, so it
            is digested with the rest and it separates one point's key from the same point run
            at another setting.
        num_steps: Step cap, for the cost estimate and for the runs.

    Raises:
        ValueError: On an empty axis of the lattice itself, on an unknown dataset, on an
            unknown mixer, on a malformed fixed override, on a key that is both fixed and
            swept, on a mixer setting some listed mixer does not accept, or on a repeated
            dataset, mixer or seed. Every one of those is caught here, before a single lane
            launches, which is the point of enumerating centrally.
    """

    datasets: tuple[str, ...] = tuple(DATASETS)
    mixers: tuple[str, ...] = ("linoss_im",)
    seeds: tuple[int, ...] = SEEDS
    axes: tuple[Axis, ...] = ()
    fixed: tuple[str, ...] = ()
    num_steps: int = NUM_STEPS

    def __post_init__(self) -> None:
        for name, items in (
            ("datasets", self.datasets),
            ("mixers", self.mixers),
            ("seeds", self.seeds),
        ):
            if not items:
                raise ValueError(f"lattice has no {name}")
            if len(set(items)) != len(items):
                raise ValueError(f"lattice repeats a {name[:-1]}: {list(items)}")
        for dataset in self.datasets:
            setting_for(dataset)
        swept = [axis.key for axis in self.axes]
        if len(set(swept)) != len(swept):
            raise ValueError(f"lattice sweeps a key twice: {swept}")
        held = [key for key, _ in self._fixed_pairs()]
        if len(set(held)) != len(held):
            raise ValueError(f"lattice fixes a key twice: {held}")
        both = sorted(set(held) & set(swept))
        if both:
            raise ValueError(f"{both} is both fixed and swept")
        mixer_keys = [axis.key for axis in self.axes if not axis.scaffold]
        mixer_keys += [key for key in held if key not in SCAFFOLD_KEYS]
        for mixer in self.mixers:
            accepted = set(REGISTRY.entry(mixer).defaults)
            for key in mixer_keys:
                if key not in accepted:
                    raise ValueError(
                        f"{mixer} has no setting {key}; has {sorted(accepted)}"
                    )
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")

    def _fixed_pairs(self) -> tuple[tuple[str, str], ...]:
        """The fixed overrides, split.

        Returns:
            ``(key, value)`` in the given order.

        Raises:
            ValueError: On an override with no ``=``.
        """
        pairs: list[tuple[str, str]] = []
        for override in self.fixed:
            key, sep, text = override.partition("=")
            if not sep:
                raise ValueError(f"fixed override must be key=value, got {override!r}")
            pairs.append((key.strip(), text.strip()))
        return tuple(pairs)

    def held(self) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        """The fixed overrides, routed.

        Returns:
            Scaffold overrides as ``(key, value)`` pairs, and mixer overrides as ``key=value``
            strings, both in the given order.

        Raises:
            ValueError: From :meth:`_fixed_pairs`.
        """
        pairs = self._fixed_pairs()
        scaffold = tuple(pair for pair in pairs if pair[0] in SCAFFOLD_KEYS)
        settings = tuple(
            f"{key}={value}" for key, value in pairs if key not in SCAFFOLD_KEYS
        )
        return scaffold, settings

    def to_dict(self) -> dict[str, Any]:
        """The lattice as a run record carries it.

        Returns:
            Plain data, ordered, so :func:`plan_digest` is stable across processes.
        """
        return {
            "datasets": list(self.datasets),
            "mixers": list(self.mixers),
            "seeds": list(self.seeds),
            "axes": [
                {
                    "key": axis.key,
                    "values": list(axis.values),
                    "scaffold": axis.scaffold,
                }
                for axis in self.axes
            ],
            "fixed": list(self.fixed),
            "num_steps": self.num_steps,
        }


class Point(NamedTuple):
    """One run.

    Attributes:
        position: Place in the canonical enumeration, from zero. Stable for a given lattice and
            useful for reading a plan; it is *not* what decides the shard. See :func:`assign`.
            Spelled ``position`` rather than ``index`` because a named tuple's field would
            otherwise shadow ``tuple.index``.
        dataset: Protocol dataset.
        mixer: Registry name.
        seed: Protocol seed.
        scaffold: This point's scaffold overrides as ``(key, text)``: the lattice's fixed ones
            first, then the swept ones in axis order.
        settings: This point's mixer overrides as ``key=text``, same order, ready for
            :meth:`scripts.harness.Registry.resolve`.
        key: A stable, filesystem-safe identifier. Two lattices that contain this same run
            produce the same key, so a result file is addressable without the plan.
    """

    position: int
    dataset: str
    mixer: str
    seed: int
    scaffold: tuple[tuple[str, str], ...]
    settings: tuple[str, ...]
    key: str


def _key_for(
    dataset: str,
    mixer: str,
    seed: int,
    scaffold: tuple[tuple[str, str], ...],
    settings: tuple[str, ...],
) -> str:
    """The stable identifier for one run.

    Args:
        dataset: Protocol dataset.
        mixer: Registry name.
        seed: Protocol seed.
        scaffold: Scaffold overrides.
        settings: Mixer overrides.

    Returns:
        ``<dataset>-<mixer>-s<seed>-<digest>``. The digest covers the overrides sorted, so
        two lattices that list the same axes in different orders address the same run.
    """
    body = json.dumps(
        {"scaffold": sorted(scaffold), "settings": sorted(settings)}, sort_keys=True
    )
    digest = hashlib.blake2b(body.encode("utf-8"), digest_size=8).hexdigest()
    return f"{dataset}-{mixer}-s{seed}-{digest[:_DIGEST_WIDTH]}"


def points(lattice: Lattice) -> tuple[Point, ...]:
    """Enumerate the lattice, once, in canonical order.

    The order is dataset, then mixer, then each axis in its listed order, then seed. Seeds vary
    fastest so that a truncated read of a plan still shows whole cells.

    Args:
        lattice: The sweep.

    Returns:
        Every run, with ``index`` in this order. The lattice's fixed overrides are merged into
        every point ahead of its swept ones, so a point carries everything it needs and nothing
        downstream has to remember to apply the fixed set.

    Raises:
        ValueError: From :meth:`Lattice.held`.
    """
    found: list[Point] = []
    held_scaffold, held_settings = lattice.held()
    combinations = list(itertools.product(*(axis.values for axis in lattice.axes)))
    for dataset, mixer in itertools.product(lattice.datasets, lattice.mixers):
        for chosen in combinations:
            scaffold = held_scaffold + tuple(
                (axis.key, value)
                for axis, value in zip(lattice.axes, chosen, strict=True)
                if axis.scaffold
            )
            settings = held_settings + tuple(
                f"{axis.key}={value}"
                for axis, value in zip(lattice.axes, chosen, strict=True)
                if not axis.scaffold
            )
            for seed in lattice.seeds:
                found.append(
                    Point(
                        position=len(found),
                        dataset=dataset,
                        mixer=mixer,
                        seed=seed,
                        scaffold=scaffold,
                        settings=settings,
                        key=_key_for(dataset, mixer, seed, scaffold, settings),
                    )
                )
    return tuple(found)


def plan_digest(lattice: Lattice) -> str:
    """A digest of the whole plan.

    Args:
        lattice: The sweep.

    Returns:
        Hex blake2b over the lattice and every point's key. Covering the keys and not only the
        lattice fields means a change in enumeration order or in key construction also changes
        the digest, so a mixed harvest is caught rather than merged.
    """
    body = json.dumps(
        {
            "lattice": lattice.to_dict(),
            "keys": [point.key for point in points(lattice)],
        },
        sort_keys=True,
    )
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).hexdigest()


def setting_for_point(point: Point) -> Setting:
    """The scaffold setting one run uses.

    Args:
        point: The run.

    Returns:
        The dataset's published setting with the point's scaffold overrides applied, each read
        at the type of the field it replaces.

    Raises:
        KeyError: On a dataset outside the protocol.
        ValueError: On a value the field's type does not read, or on a setting the override
            makes invalid -- :class:`scripts.tsc.protocol.Setting` validates on construction,
            so a swept ``blocks=0`` is refused here and not at the model.
    """
    base = setting_for(point.dataset)
    if not point.scaffold:
        return base
    changes = {
        key: _coerce(key, getattr(base, key), text) for key, text in point.scaffold
    }
    return replace(base, **changes)


LENGTHS = {
    "EigenWorms": 17984,
    "EthanolConcentration": 1751,
    "Heartbeat": 405,
    "MotorImagery": 3000,
    "SelfRegulationSCP1": 896,
    "SelfRegulationSCP2": 1152,
}
"""Sequence length per protocol dataset, for the cost estimate only.

Archive facts, and a cost estimate is the one place a stale copy of one is harmless: it can only
unbalance a sweep, never change a number. The length that reaches a model comes from ``data.npy``
and never from here, and :mod:`tests.test_tsc_sweep` checks these against the manifests when
``SLINOSS_TSC_CORPUS`` names a processed corpus."""

_STATE_KEYS = frozenset({"ssm_dim", "d_state", "state_size"})
"""Mixer settings that name a state width, for the cost estimate only."""


def cost(point: Point, num_steps: int = NUM_STEPS) -> float:
    """A relative cost estimate for one run.

    The shape of the per-step work: ``batch_size * length * blocks * (hidden * state +
    hidden^2)``, times the step cap. ``length`` is the dataset's sequence length, the term that
    spreads cost across the six datasets by two orders of magnitude and therefore the whole
    reason a sweep cannot be sliced by count.

    Two things it deliberately does not model. Early stopping, which ends most runs and cannot
    be read off a configuration, so this is a full-budget upper bound and every point is
    over-estimated by a factor of the same kind. And the mixer's real arithmetic: its state
    width is taken from a swept ``ssm_dim``/``d_state``/``state_size`` when one is present and
    from the published ``ssm_dim`` otherwise. An exact per-mixer cost would have to be measured,
    and a measured number is a per-host number, which cannot live in a shared plan.

    Args:
        point: The run.
        num_steps: Step cap.

    Returns:
        A positive number in arbitrary units. Only ratios are used.

    Raises:
        KeyError: On a dataset outside the protocol or with no recorded length.
        ValueError: From :func:`setting_for_point`, or on a swept state width that is not a
            number.
    """
    setting = setting_for_point(point)
    if point.dataset not in LENGTHS:
        raise KeyError(
            f"no recorded length for {point.dataset}; have {sorted(LENGTHS)}"
        )
    length = float(LENGTHS[point.dataset])
    state = float(setting.ssm_dim)
    for override in point.settings:
        key, _, text = override.partition("=")
        if key in _STATE_KEYS:
            state = float(text)
    hidden = float(setting.hidden_dim)
    per_step = (
        setting.batch_size * length * setting.blocks * (hidden * state + hidden**2)
    )
    return float(num_steps) * per_step


class Shard(NamedTuple):
    """One host's slice.

    Attributes:
        position: Which slice, from zero. Spelled like :attr:`Point.position` and for the same
            reason: an ``index`` field would shadow ``tuple.index``.
        slices: Slices in total.
        plan: The lattice's digest, to be written with the results.
        points: The runs, in the order the host should take them: most expensive first, so a
            shard that is killed early has done the work that would not fit anywhere else.
    """

    position: int
    slices: int
    plan: str
    points: tuple[Point, ...]


def assign(
    every: Sequence[Point],
    count: int,
    *,
    num_steps: int = NUM_STEPS,
    weights: Sequence[float] | None = None,
) -> tuple[tuple[Point, ...], ...]:
    """Split points across slices by estimated cost.

    Longest-processing-time-first: order by cost descending, and give each point to the slice
    whose load over its weight is lowest. That is within 4/3 of optimal for equal machines and
    is what makes a heterogeneous fleet usable from one plan. Ties break on the point's key, so
    the assignment is a function of the plan alone and every host computes the same one.

    Args:
        every: The enumeration, from :func:`points`.
        count: Slices.
        num_steps: Step cap, for the cost estimate.
        weights: Relative capacity per slice, or None for equal. A slice with twice the weight
            takes about twice the cost.

    Returns:
        One tuple of points per slice, each ordered most expensive first.

    Raises:
        ValueError: On a non-positive count, on a weight list of the wrong length, or on a
            weight that is not positive.
    """
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if weights is None:
        weights = [1.0] * count
    if len(weights) != count:
        raise ValueError(f"{len(weights)} weights for {count} slices")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError(f"weights must be positive, got {list(weights)}")
    ordered = sorted(every, key=lambda point: (-cost(point, num_steps), point.key))
    loads = [0.0] * count
    buckets: list[list[Point]] = [[] for _ in range(count)]
    for point in ordered:
        target = min(range(count), key=lambda slot: (loads[slot] / weights[slot], slot))
        buckets[target].append(point)
        loads[target] += cost(point, num_steps)
    return tuple(tuple(bucket) for bucket in buckets)


def shard(
    lattice: Lattice,
    index: int,
    count: int,
    *,
    weights: Sequence[float] | None = None,
) -> Shard:
    """One slice of a lattice, with the plan digest that identifies it.

    Args:
        lattice: The sweep.
        index: Which slice, from zero.
        count: Slices.
        weights: Relative capacity per slice.

    Returns:
        The slice.

    Raises:
        ValueError: On an index outside the count, or from :func:`assign`.
    """
    if not 0 <= index < count:
        raise ValueError(f"shard {index} is outside 0..{count - 1}")
    buckets = assign(
        points(lattice), count, num_steps=lattice.num_steps, weights=weights
    )
    return Shard(index, count, plan_digest(lattice), buckets[index])


def merge(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect shard results, refusing a mixed harvest.

    Args:
        records: One dict per finished run, each carrying a ``plan`` and a ``key``.

    Returns:
        The records, sorted by key.

    Raises:
        ValueError: When a record carries no ``plan`` or no ``key``, when two records disagree
            about the plan, or when one key appears twice with different results. A duplicate
            key with identical results is a re-run and is dropped.
    """
    seen: dict[str, dict[str, Any]] = {}
    plan: str | None = None
    for record in records:
        for field_name in ("plan", "key"):
            if field_name not in record:
                raise ValueError(f"record carries no {field_name}: {sorted(record)}")
        if plan is None:
            plan = str(record["plan"])
        elif str(record["plan"]) != plan:
            raise ValueError(
                f"records span two plans, {plan} and {record['plan']}; they were enumerated "
                f"from different lattices and their shard indices do not mean the same thing"
            )
        key = str(record["key"])
        if key in seen and seen[key] != record:
            raise ValueError(f"two different results for {key}")
        seen[key] = record
    return [seen[key] for key in sorted(seen)]
