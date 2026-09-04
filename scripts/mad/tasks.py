"""The six MAD tasks and their data pools.

A :class:`TaskSpec` is one point of the benchmark: a generator, the settings it is
called with, and the split sizes. :data:`TASKS` holds the six baselines `mad-lab`'s
``configs/tasks/*.yml`` define, each carrying the difficulty ladder that file lists
under ``changes``. :func:`build_pool` draws a pool from a spec.

Pools are numpy, not torch: the data layer is exactly as portable as the generators it
calls, so its parity gate runs anywhere.

Draw order follows `mad-lab`'s ``generate_data``. One generator serves both splits and
the train split is drawn first, so a pool is reproduced by its seed alone and the test
split is not the train split's stream replayed. Recall tasks declare that their target
stream consumes the train/test role; the other three declare their generator invariant
to it and never receive an unused argument.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from scripts.mad.instances import (
    Instance,
    build_kv_map,
    compression,
    fuzzy_in_context_recall,
    in_context_recall,
    memorization,
    noisy_in_context_recall,
    selective_copying,
)

SUPPLIED = frozenset({"kv_map", "blank_rng"})
"""Generator arguments :func:`build_pool` constructs rather than the spec carrying."""

LEAKAGE_LIMIT = 0.001
"""Fraction of test inputs also drawn into train that `mad-lab` warns above. A task
whose pool exceeds it is measuring memorization of the pool, not the mechanism."""


@dataclass(frozen=True)
class TaskSpec:
    """One benchmark point.

    Frozen against rebinding, not against mutation of ``extra`` and ``ladder``. Use
    :meth:`override` to move along an axis.

    Attributes:
        name: Short key, as :data:`TASKS` and the command line use it.
        mad_name: The name MAD reports the task under.
        generator: Instance generator. ``split_policy`` declares whether it is
            called with an ``is_training`` keyword.
        split_policy: ``required`` when train and test targets differ by role;
            ``invariant`` when they differ only by their independent draws. This is
            explicit so an accepted role argument is never deleted inside a generator.
        vocab_size: Symbols, including whatever special tokens the generator reserves.
        seq_len: Generator length setting. Not the width, which the generator fixes;
            the two recall tasks with a copy prefix return ``seq_len - 1`` positions.
        num_train: Training examples.
        num_test: Test examples.
        bottleneck: Whether the target is the whole input reconstructed from one state,
            which needs the bottleneck backbone rather than the causal one.
        supplied: Generator arguments :func:`build_pool` constructs, a subset of
            :data:`SUPPLIED`.
        extra: Generator settings beyond ``vocab_size`` and ``seq_len``.
        ladder: The task's own difficulty axes, from its ``changes`` block. Data for a
            sweep; :meth:`override` does not consult it.
    """

    name: str
    mad_name: str
    generator: Callable[..., Instance]
    split_policy: Literal["required", "invariant"]
    vocab_size: int
    seq_len: int
    num_train: int
    num_test: int = 1280
    bottleneck: bool = False
    supplied: frozenset[str] = frozenset()
    extra: dict[str, Any] = field(default_factory=dict)
    ladder: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split_policy not in {"required", "invariant"}:
            raise ValueError(
                f"{self.name}: split_policy must be required or invariant, got "
                f"{self.split_policy!r}"
            )
        unknown = self.supplied - SUPPLIED
        if unknown:
            raise ValueError(f"{self.name}: cannot supply {sorted(unknown)}")
        overlap = self.extra.keys() & {"vocab_size", "seq_len", "is_training"}
        if overlap:
            raise ValueError(f"{self.name}: {sorted(overlap)} belongs to the spec")
        for axis in self.ladder:
            if axis not in self.axes:
                raise ValueError(f"{self.name}: ladder axis {axis} is not a setting")

    @property
    def axes(self) -> frozenset[str]:
        """Settings :meth:`override` accepts."""
        return frozenset(
            {"vocab_size", "seq_len", "num_train", "num_test", *self.extra}
        )

    @property
    def kwargs(self) -> dict[str, Any]:
        """Generator settings, less the ones :func:`build_pool` supplies."""
        return {"vocab_size": self.vocab_size, "seq_len": self.seq_len, **self.extra}

    def override(self, **axes: Any) -> TaskSpec:
        """This spec with one or more settings moved.

        Args:
            **axes: Settings to replace, each a member of :attr:`axes`. An axis the
                task does not have is refused rather than dropped: ``frac_noise`` on a
                task with no noise vocabulary, or ``num_tokens_to_copy`` on a task with
                nothing to copy, would otherwise report as applied.

        Returns:
            A new spec.

        Raises:
            ValueError: On an axis this task does not have.
        """
        unknown = axes.keys() - self.axes
        if unknown:
            raise ValueError(
                f"{self.name}: no axis {sorted(unknown)}; has {sorted(self.axes)}"
            )
        extra = {k: v for k, v in axes.items() if k in self.extra}
        direct = {k: v for k, v in axes.items() if k not in self.extra}
        return replace(self, extra={**self.extra, **extra}, **direct)


TASKS: dict[str, TaskSpec] = {
    "icr": TaskSpec(
        name="icr",
        mad_name="in-context-recall",
        generator=in_context_recall,
        split_policy="required",
        vocab_size=16,
        seq_len=128,
        num_train=12800,
        extra={"multi_query": True},
        ladder={
            "vocab_size": (32, 64, 128),
            "seq_len": (256, 512, 1024),
            "num_train": (6400, 3200, 1600, 800),
        },
    ),
    "nicr": TaskSpec(
        name="nicr",
        mad_name="noisy-in-context-recall",
        generator=noisy_in_context_recall,
        split_policy="required",
        # 32, not 16: the noise vocabulary is carved out of the same range, so 16 less
        # 16 noise symbols leaves no key-value vocabulary at all. 32 leaves the 16 the
        # other recall tasks get, and the ladder's vocabulary rungs each add the same
        # 16 on top.
        vocab_size=32,
        seq_len=128,
        num_train=12800,
        extra={"multi_query": True, "noise_vocab_size": 16, "frac_noise": 0.2},
        ladder={
            "vocab_size": (48, 80, 144),
            "seq_len": (256, 512, 1024),
            "num_train": (6400, 3200, 1600, 800),
            "frac_noise": (0.4, 0.6, 0.8),
        },
    ),
    "ficr": TaskSpec(
        name="ficr",
        mad_name="fuzzy-in-context-recall",
        generator=fuzzy_in_context_recall,
        split_policy="required",
        vocab_size=16,
        seq_len=128,
        num_train=12800,
        extra={"multi_query": True, "k_motif_size": 3, "v_motif_size": 3},
        ladder={
            "vocab_size": (32, 64, 128),
            "seq_len": (256, 512, 1024),
            "num_train": (6400, 3200, 1600, 800),
        },
    ),
    "mem": TaskSpec(
        name="mem",
        mad_name="memorization",
        generator=memorization,
        split_policy="invariant",
        vocab_size=256,
        seq_len=32,
        num_train=256,
        supplied=frozenset({"kv_map"}),
        ladder={"vocab_size": (512, 1024, 2048, 4096, 8192)},
    ),
    "comp": TaskSpec(
        name="comp",
        mad_name="compression",
        generator=compression,
        split_policy="invariant",
        vocab_size=16,
        seq_len=32,
        num_train=12800,
        bottleneck=True,
        ladder={
            "vocab_size": (32, 64, 128),
            "seq_len": (64, 128, 256),
            "num_train": (6400, 3200, 1600, 800),
        },
    ),
    "sc": TaskSpec(
        name="sc",
        mad_name="selective-copying",
        generator=selective_copying,
        split_policy="invariant",
        vocab_size=16,
        seq_len=256,
        num_train=12800,
        supplied=frozenset({"blank_rng"}),
        extra={"num_tokens_to_copy": 16},
        ladder={
            "vocab_size": (32, 64, 128),
            "seq_len": (512, 1024),
            "num_train": (6400, 3200, 1600, 800),
            "num_tokens_to_copy": (32, 64, 96),
        },
    ),
}
"""The six baselines, keyed by short name. Values are `mad-lab`'s task configs."""


class Pool(NamedTuple):
    """One task's fixed train and test sets.

    Attributes:
        train_inputs: ``(num_train, width)`` int64.
        train_targets: ``(num_train, width)`` int64, ids and ignore indices.
        test_inputs: ``(num_test, width)`` int64.
        test_targets: ``(num_test, width)`` int64.
        leakage: Fraction of distinct test inputs that also occur in train.
    """

    train_inputs: NDArray[np.int64]
    train_targets: NDArray[np.int64]
    test_inputs: NDArray[np.int64]
    test_targets: NDArray[np.int64]
    leakage: float

    @property
    def width(self) -> int:
        """Positions per example, both splits."""
        return int(self.train_inputs.shape[1])


def leakage(train_inputs: NDArray[np.int64], test_inputs: NDArray[np.int64]) -> float:
    """Fraction of distinct test inputs that also occur in train.

    Args:
        train_inputs: ``(n, width)`` int64.
        test_inputs: ``(m, width)`` int64.

    Returns:
        In ``[0, 1]``. Above :data:`LEAKAGE_LIMIT` the test split reports recall of the
        train split.
    """
    train = {row.tobytes() for row in train_inputs}
    test = {row.tobytes() for row in test_inputs}
    return 1.0 - len(test - train) / len(test)


def _draw(
    spec: TaskSpec,
    count: int,
    *,
    is_training: bool,
    rng: np.random.Generator,
    kwargs: dict[str, Any],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Draw one split.

    Args:
        spec: The task.
        count: Examples to draw.
        is_training: Which target stream the generator returns.
        rng: Draw source, advanced across both splits.
        kwargs: Generator settings, including whatever :func:`build_pool` supplied.

    Returns:
        ``(inputs, targets)``, both ``(count, width)`` int64.

    Raises:
        ValueError: On ``count < 1``, or on instances of unequal width, which would
            stack only under a padding rule the task does not define.
    """
    if count < 1:
        raise ValueError(f"{spec.name}: a split needs at least one example")
    if spec.split_policy == "required":
        pairs = [
            spec.generator(rng, is_training=is_training, **kwargs) for _ in range(count)
        ]
    else:
        pairs = [spec.generator(rng, **kwargs) for _ in range(count)]
    widths = {len(inputs) for inputs, _ in pairs}
    if len(widths) != 1:
        raise ValueError(f"{spec.name}: ragged instances, widths {sorted(widths)}")
    return (
        np.stack([inputs for inputs, _ in pairs]),
        np.stack([targets for _, targets in pairs]),
    )


def build_pool(spec: TaskSpec, *, seed: int) -> Pool:
    """Draw a task's train and test sets.

    Args:
        spec: The task.
        seed: Seeds the pool's generator, and any draw source in
            :attr:`TaskSpec.supplied`. A pool is a function of this and the spec.

    Returns:
        The pool, its two splits of equal width.

    Raises:
        ValueError: From :func:`_draw`, on an empty or a ragged split.
    """
    kwargs = dict(spec.kwargs)
    if "kv_map" in spec.supplied:
        # Less the insert token, and seeded here rather than per instance, so one
        # mapping serves every example of both splits.
        kwargs["kv_map"] = build_kv_map(spec.vocab_size - 1, seed=seed)
    if "blank_rng" in spec.supplied:
        kwargs["blank_rng"] = np.random.RandomState(seed)

    rng = np.random.default_rng(seed)
    train_inputs, train_targets = _draw(
        spec, spec.num_train, is_training=True, rng=rng, kwargs=kwargs
    )
    test_inputs, test_targets = _draw(
        spec, spec.num_test, is_training=False, rng=rng, kwargs=kwargs
    )
    if train_inputs.shape[1] != test_inputs.shape[1]:
        raise ValueError(
            f"{spec.name}: splits disagree on width, "
            f"{train_inputs.shape[1]} and {test_inputs.shape[1]}"
        )
    return Pool(
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        leakage(train_inputs, test_inputs),
    )
