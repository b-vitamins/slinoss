"""The learning rate: where it comes from and how it falls.

Two pure functions, both from the protocol this harness reproduces.

:func:`transfer` is the width and batch correction. A base rate is tuned once at a reference
width and token batch, and every other shape reads it through
``sqrt(ref_d_model / d_model) * sqrt(token_batch / ref_token_batch)``: the first factor is the
mu-P transfer for a rate tuned at one width, the second is Adam's batch scaling. This is why
a 45M arm and a 180M arm carry one tuned number between them and not two, and why a run
record stores the transferred rate rather than the typed one -- a replay that re-derives it
from a changed width would silently be a different run.

:func:`lr_at` is the schedule: constant, then linear down over the last ``warmdown_fraction``
of training. No warmup. The floor is reached at ``total_steps``, which is one past the last
step taken, so the last step runs at a small positive rate rather than at zero.
"""

from __future__ import annotations

import math

__all__ = ["lr_at", "steps_for", "transfer"]

REF_D_MODEL = 768
"""Width the base rate was tuned at."""

REF_TOKEN_BATCH = 1 << 19
"""Token batch the base rate was tuned at."""


def transfer(
    base_lr: float,
    *,
    d_model: int,
    token_batch: int,
    ref_d_model: int = REF_D_MODEL,
    ref_token_batch: int = REF_TOKEN_BATCH,
) -> float:
    """Carry a tuned rate to another width and batch size.

    Args:
        base_lr: Rate tuned at the reference shape.
        d_model: This arm's width.
        token_batch: This arm's tokens per optimizer step.
        ref_d_model: Width the base rate was tuned at.
        ref_token_batch: Token batch the base rate was tuned at.

    Returns:
        The peak rate for this shape.

    Raises:
        ValueError: On a non-positive width, batch or reference.
    """
    for name, value in (
        ("base_lr", base_lr),
        ("d_model", d_model),
        ("token_batch", token_batch),
        ("ref_d_model", ref_d_model),
        ("ref_token_batch", ref_token_batch),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    width = math.sqrt(ref_d_model / d_model)
    batch = math.sqrt(token_batch / ref_token_batch)
    return base_lr * width * batch


def lr_at(
    step: int,
    *,
    total_steps: int,
    peak_lr: float,
    warmdown_fraction: float,
    final_fraction: float = 0.0,
) -> float:
    """The rate at one step.

    Args:
        step: Step index, from zero.
        total_steps: Steps the run takes. The floor is reached here, one past the last
            step, so no step runs at the floor.
        peak_lr: The constant phase's rate.
        warmdown_fraction: Fraction of the run spent falling.
        final_fraction: Floor, as a fraction of ``peak_lr``.

    Returns:
        The rate. Clamped at the floor past ``total_steps`` so a resumed run that
        overshoots does not go negative.

    Raises:
        ValueError: On a non-positive step budget or peak, or a fraction outside
            ``[0, 1]``.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if peak_lr <= 0.0:
        raise ValueError(f"peak_lr must be positive, got {peak_lr}")
    for name, value in (
        ("warmdown_fraction", warmdown_fraction),
        ("final_fraction", final_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    floor = peak_lr * final_fraction
    hold = round(total_steps * (1.0 - warmdown_fraction))
    if step < hold:
        return peak_lr
    if step >= total_steps or hold >= total_steps:
        return floor
    fallen = (step - hold) / (total_steps - hold)
    return peak_lr + (floor - peak_lr) * fallen


def steps_for(token_budget: int, token_batch: int) -> int:
    """Optimizer steps a token budget buys.

    Args:
        token_budget: Tokens the run trains on.
        token_batch: Tokens per optimizer step.

    Returns:
        The step count, rounded down. A partial step is not taken: the budget is the
        thing held fixed across arms, and a fractional step would differ per width.

    Raises:
        ValueError: On a non-positive argument, or a budget under one step.
    """
    for name, value in (("token_budget", token_budget), ("token_batch", token_batch)):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    steps = token_budget // token_batch
    if steps < 1:
        raise ValueError(
            f"token_budget {token_budget} is under one step of {token_batch} tokens"
        )
    return steps
