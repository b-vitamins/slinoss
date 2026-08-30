"""Solve the width that puts two arms at one parameter count.

A table whose arms differ in size is not a comparison, and a table whose arms were sized by
hand is a comparison nobody can reproduce. So the width is solved: fix the depth, fix the
target non-embedding parameter count, and search ``d_model`` for the arm that lands nearest.

Non-embedding, because the token table and the head scale with the vocabulary and are
identical across arms at one width; counting them would let a wider mixer hide behind a shared
table. :func:`scripts.lm.model.non_embedding_parameters` is the count.

The search is a bisection over a grid, not over the integers. Every arm here constrains the
width: ``d_inner = expand * d_model`` must be a multiple of ``d_head``, ``d_state`` must be a
multiple of 48, and attention needs ``n_heads`` to divide ``d_model``. A grid of multiples of
:data:`WIDTH_MULTIPLE` satisfies all of them at the settings this axis registers, and a width
off the grid would build for one arm and raise for another. The published protocol's widths
(496 and 1360) are not on it; reproducing a width is not the contract, matching a count is.

The count is monotone increasing in the width for every arm, which is what makes the
bisection sound. :func:`solve_width` does not assume it silently -- it checks the bracket it
starts from and reports a non-monotone counter rather than converging on noise.

:func:`check_spread` is the gate. Sizing every arm to a target still leaves each one a grid
point away from it, so the run refuses to proceed when the spread across arms is over the
tolerance, rather than printing a table whose columns are different sizes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import NamedTuple

__all__ = [
    "SPREAD_TOLERANCE",
    "WIDTH_MULTIPLE",
    "Sizing",
    "check_spread",
    "size_arm",
    "solve_width",
    "spread",
]

WIDTH_MULTIPLE = 64
"""Grid step for ``d_model``.

64 divides into every width constraint the registered mixers impose: it makes ``d_inner`` at
``expand 2`` a multiple of 128 so any ``d_head`` up to 128 divides it, and it is divisible by
every head count an arm would use. A finer grid lands closer to a target and builds for fewer
arms.
"""

SPREAD_TOLERANCE = 0.02
"""Largest relative spread in non-embedding parameters a table may carry across arms."""

Counter = Callable[[int], int]
"""``d_model -> non-embedding parameters``. Monotone increasing."""


class Sizing(NamedTuple):
    """One arm's solved shape.

    Attributes:
        mixer: Registry name.
        d_model: Solved width, a multiple of :data:`WIDTH_MULTIPLE`.
        n_layers: Depth, the same for every arm.
        parameters: Non-embedding trainable parameters at that width. The achieved count,
            recorded rather than the target, because the target is not reachable on the
            grid.
        target: What was asked for.
    """

    mixer: str
    d_model: int
    n_layers: int
    parameters: int
    target: int

    @property
    def error(self) -> float:
        """Signed relative miss against the target.

        Returns:
            ``(parameters - target) / target``.
        """
        return (self.parameters - self.target) / self.target


def solve_width(
    target: int,
    count: Counter,
    *,
    low: int = WIDTH_MULTIPLE,
    high: int = 4096,
    multiple: int = WIDTH_MULTIPLE,
) -> int:
    """The grid width whose parameter count is nearest ``target``.

    Args:
        target: Non-embedding parameters wanted.
        count: Parameters at a width. Called once per bisection step, so it may build a
            model; the search is logarithmic in the grid size.
        low: Smallest width to consider.
        high: Largest width to consider.
        multiple: Grid step.

    Returns:
        The width. The nearer of the two grid points bracketing the target, by absolute
        difference in count, so the answer may be over the target as well as under.

    Raises:
        ValueError: On a non-positive target or step, on a bracket that is empty, or on a
            counter that is not increasing across the bracket. A non-monotone counter makes
            the bisection meaningless, and the likely cause is a setting that changes with
            the width rather than one held fixed.
    """
    if target < 1:
        raise ValueError(f"target must be positive, got {target}")
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}")
    first = -(-low // multiple)
    last = high // multiple
    if first > last or first < 1:
        raise ValueError(f"no multiple of {multiple} in [{low}, {high}]")
    at_first, at_last = count(first * multiple), count(last * multiple)
    if at_last <= at_first:
        raise ValueError(
            f"count is {at_first} at d_model {first * multiple} and {at_last} at "
            f"{last * multiple}; the search needs it increasing"
        )
    if target <= at_first:
        return first * multiple
    if target >= at_last:
        return last * multiple
    lower, upper = first, last
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if count(middle * multiple) <= target:
            lower = middle
        else:
            upper = middle
    below, above = count(lower * multiple), count(upper * multiple)
    if target - below <= above - target:
        return lower * multiple
    return upper * multiple


def size_arm(
    target: int,
    mixer: str,
    *,
    n_layers: int,
    vocab_size: int,
    max_length: int,
    overrides: Iterable[str] = (),
    low: int = WIDTH_MULTIPLE,
    high: int = 4096,
    multiple: int = WIDTH_MULTIPLE,
) -> Sizing:
    """Solve one arm's width by building it.

    The count is taken off a real model rather than a formula, so a mixer this harness did
    not write is sized correctly without anyone transcribing its parameter shapes. The cost
    is a build per bisection step, on the host, discarded.

    A baseline that refuses to construct off CUDA cannot be sized this way; size it on the
    host it runs on, or size the arm it is matched to and pass the width through.

    Args:
        target: Non-embedding parameters wanted.
        mixer: Registry name.
        n_layers: Depth.
        vocab_size: Tokens, for the scaffold. Does not affect the count.
        max_length: Longest sequence, passed to the factory.
        overrides: ``key=value`` mixer settings.
        low: Smallest width to consider.
        high: Largest width to consider.
        multiple: Grid step.

    Returns:
        The sizing.

    Raises:
        KeyError: On an unregistered mixer.
        ValueError: From :func:`solve_width`, or from the mixer at a width it refuses.
    """
    from scripts.lm.mixers import REGISTRY
    from scripts.lm.model import (
        build_model,
        layer_factories,
        non_embedding_parameters,
        scaffold_config,
    )

    resolved = REGISTRY.resolve(mixer, overrides)

    def count(d_model: int) -> int:
        config = scaffold_config(
            d_model=d_model, n_layers=n_layers, vocab_size=vocab_size
        )
        model = build_model(
            config,
            layer_factories(resolved.factory, n_layers),
            max_length=max_length,
        )
        return non_embedding_parameters(model)

    d_model = solve_width(target, count, low=low, high=high, multiple=multiple)
    return Sizing(mixer, d_model, n_layers, count(d_model), target)


def spread(counts: Mapping[str, int]) -> float:
    """Relative spread in non-embedding parameters.

    Args:
        counts: Parameters per arm. A mapping rather than a sequence of :class:`Sizing`,
            so a table of finished records checks the same way a set of solved widths
            does.

    Returns:
        ``(max - min) / mean``, or zero for fewer than two arms.

    Raises:
        ValueError: On an empty mapping.
    """
    if not counts:
        raise ValueError("spread needs at least one arm")
    values = list(counts.values())
    if len(values) < 2:
        return 0.0
    return (max(values) - min(values)) / (sum(values) / len(values))


def check_spread(
    counts: Mapping[str, int], tolerance: float = SPREAD_TOLERANCE
) -> float:
    """Refuse a set of arms that are not the same size.

    Args:
        counts: Parameters per arm.
        tolerance: Largest relative spread allowed.

    Returns:
        The spread.

    Raises:
        ValueError: When the spread is over the tolerance, naming the widest and narrowest
            arm and their counts.
    """
    found = spread(counts)
    if found > tolerance:
        widest = max(counts, key=lambda arm: counts[arm])
        narrowest = min(counts, key=lambda arm: counts[arm])
        raise ValueError(
            f"parameter spread {found:.4f} is over {tolerance:.4f}: "
            f"{widest} at {counts[widest]} and {narrowest} at {counts[narrowest]}"
        )
    return found
