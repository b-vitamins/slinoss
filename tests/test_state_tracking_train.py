"""The protocol: the schedule, the decay split, the masked loss, the length bands.

Nothing in this module is tuned, so its failure modes are all transcription errors, and
every one of them is silent. A schedule off by one step trains at a different rate than
every published bar. A decay split that misses the embedding decays the token table. A loss
keyed on an ignore index instead of the mask drops every group-task position whose running
product is the identity. A band boundary off by one moves the tail accuracy, which is the
number the axis is read on.

The two upstream defects the module refuses to transcribe -- the double
``optimizer.step()`` and the ``zero_grad`` inside the accumulation window -- are pinned
here as behaviour: one parameter update per step, and an accumulated step whose reported
loss is the mean over every micro-batch it saw.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn

from scripts.state_tracking.instances import SplitConfig, batches, collate
from scripts.state_tracking.model import ModelConfig, StateTracker, build_model
from scripts.state_tracking.tasks import AUTOMATA, Task, resolve
from scripts.state_tracking.train import (
    Band,
    Metrics,
    Report,
    TrainConfig,
    _Tally,
    evaluate,
    lr_at,
    masked_loss,
    parameter_groups,
    seed_all,
    split_seeds,
    stage,
    train,
)

D_MODEL = 16
PROTOCOL = TrainConfig()
TINY = TrainConfig(
    num_steps=4,
    batch_size=4,
    print_steps=2,
    early_stop_threshold=2.0,
    band_width=4,
    device="cpu",
)
"""The protocol shrunk to run whole in a test. Every field it does not name is upstream's.

``early_stop_threshold`` is over 1 so no run here stops early by luck, and ``device`` is the
CPU so the loop is tested where there is no GPU."""

TRAIN_SPLIT = SplitConfig(min_length=3, max_length=8, seed=0)
VAL_SPLIT = SplitConfig(min_length=8, max_length=12, seed=0, count=8)


def _factory(d_model: int, max_length: int) -> nn.Module:
    """A one-linear mixer, enough to make the scaffold trainable on the CPU."""
    del max_length
    return nn.Linear(d_model, d_model, bias=False)


def _model(vocab_size: int = 3, n_layers: int = 1) -> StateTracker:
    """The scaffold at zero dropout, so a loss is a function of the batch alone."""
    config = ModelConfig(
        vocab_size, 16, d_model=D_MODEL, n_layers=n_layers, dropout=0.0
    )
    return build_model(config, _factory)


def _run(config: TrainConfig) -> tuple[StateTracker, Report]:
    """Train the tiny protocol on parity and return the model and the report."""
    seed_all(config.seed)
    model = _model()
    report = train(model, AUTOMATA["parity"], TRAIN_SPLIT, VAL_SPLIT, config)
    return model, report


def test_protocol_defaults_are_upstreams() -> None:
    """The twenty published configs' constants, none of them a knob here.

    Restated as an assertion because a default moving is invisible: an arm's record would
    carry the new value and read as reproducible.
    """
    assert PROTOCOL.num_steps == 100001
    assert PROTOCOL.batch_size == 256
    assert PROTOCOL.lr == 0.002
    assert PROTOCOL.final_lr == 1e-5
    assert PROTOCOL.warmup_fraction == 0.1
    assert PROTOCOL.weight_decay_embedding == 0.0
    assert PROTOCOL.weight_decay_others == 1e-2
    assert PROTOCOL.early_stop_threshold == 0.9995
    assert PROTOCOL.print_steps == 5000
    assert PROTOCOL.accumulation_steps == 1
    assert PROTOCOL.precision == "fp32"
    assert PROTOCOL.grad_clip == 0.0
    assert PROTOCOL.warmup_steps == 10000


def test_first_step_runs_at_exactly_zero() -> None:
    """Upstream's scheduler steps once in its constructor, so step 0 runs at rate 0.

    That first update moves nothing, decoupled weight decay included, since the decay
    scales with the rate. One step in 100001, and it is upstream's, so it is transcribed
    rather than corrected. The zero comes from the warmup: with no warmup, step 0 is the
    peak.
    """
    assert lr_at(PROTOCOL, 0) == 0.0
    assert lr_at(PROTOCOL, 1) == pytest.approx(PROTOCOL.lr / PROTOCOL.warmup_steps)
    without_warmup = replace(PROTOCOL, warmup_fraction=0.0)
    assert without_warmup.warmup_steps == 0
    assert lr_at(without_warmup, 0) == PROTOCOL.lr


def test_schedule_peaks_at_the_end_of_warmup_and_floors_after_the_run() -> None:
    """Linear to ``lr`` at ``warmup_steps``, half a cosine to ``final_lr``, then flat.

    The peak lands exactly on ``lr`` and the floor exactly on ``final_lr``; an off-by-one
    at either end would put the whole cosine on a different interval.
    """
    warmup = PROTOCOL.warmup_steps
    assert lr_at(PROTOCOL, warmup) == PROTOCOL.lr
    assert lr_at(PROTOCOL, warmup - 1) < PROTOCOL.lr
    assert lr_at(PROTOCOL, PROTOCOL.num_steps) == PROTOCOL.final_lr
    assert lr_at(PROTOCOL, PROTOCOL.num_steps + 10**6) == PROTOCOL.final_lr
    assert lr_at(PROTOCOL, PROTOCOL.num_steps - 1) == pytest.approx(
        PROTOCOL.final_lr, rel=1e-6
    )
    rates = [lr_at(PROTOCOL, step) for step in range(warmup, PROTOCOL.num_steps, 997)]
    assert rates == sorted(rates, reverse=True)


def test_cosine_midpoint_is_the_midpoint() -> None:
    """At half the decay the rate is halfway between ``lr`` and ``final_lr``.

    Checked on a config whose warmup and decay spans are exact, so the assertion is
    arithmetic rather than a restatement of the implementation.
    """
    config = replace(PROTOCOL, num_steps=20, warmup_fraction=0.5)
    assert config.warmup_steps == 10
    assert lr_at(config, 10) == PROTOCOL.lr
    assert lr_at(config, 15) == pytest.approx(0.5 * (PROTOCOL.lr + PROTOCOL.final_lr))
    assert lr_at(config, 19) == pytest.approx(
        PROTOCOL.final_lr
        + 0.5 * (1 + math.cos(math.pi * 9 / 10)) * (PROTOCOL.lr - PROTOCOL.final_lr)
    )


def test_config_validation() -> None:
    """A malformed protocol is refused at construction."""
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainConfig(precision="fp16")
    with pytest.raises(ValueError, match="num_steps must be positive"):
        TrainConfig(num_steps=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="print_steps must be positive"):
        TrainConfig(print_steps=0)
    with pytest.raises(ValueError, match="accumulation_steps must be positive"):
        TrainConfig(accumulation_steps=0)
    with pytest.raises(ValueError, match="band_width must be positive"):
        TrainConfig(band_width=0)
    with pytest.raises(ValueError, match="lr must be positive"):
        TrainConfig(lr=0.0)
    with pytest.raises(ValueError, match="final_lr must be positive"):
        TrainConfig(final_lr=0.0)
    with pytest.raises(ValueError, match="grad_clip must not be negative"):
        TrainConfig(grad_clip=-1.0)
    with pytest.raises(ValueError, match=r"warmup_fraction must be in \[0, 1\)"):
        TrainConfig(warmup_fraction=1.0)


def test_split_seeds_separate_the_two_streams() -> None:
    """Train at ``seed``, validation at ``2 * seed``, upstream's scheme.

    They coincide at seed 0, which is upstream's first config, and that is harmless only
    because the two splits draw from disjoint length ranges.
    """
    assert split_seeds(3) == (3, 6)
    assert split_seeds(0) == (0, 0)


def test_weight_decay_split_is_by_parameter_name() -> None:
    """The embedding lands in the zero-decay group, everything else in the other.

    The rule is the substring ``embedding`` in a parameter's name, which is upstream's.
    """
    model = _model(vocab_size=7, n_layers=2)
    groups = parameter_groups(model, PROTOCOL)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.0
    assert groups[1]["weight_decay"] == 1e-2
    assert len(groups[0]["params"]) == 1
    assert groups[0]["params"][0] is model.embedding.weight
    total = sum(1 for _ in model.parameters())
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == total


def test_frozen_parameters_reach_no_group() -> None:
    """A frozen parameter is left out, so AdamW never decays what it cannot move.

    Decoupled decay is applied to every parameter in a group whether or not it has a
    gradient, so a frozen parameter inside a group would shrink toward zero over a run.
    """
    model = _model()
    model.embedding.weight.requires_grad_(False)
    groups = parameter_groups(model, PROTOCOL)
    assert groups[0]["params"] == []
    assert all(param.requires_grad for param in groups[1]["params"])


def test_masked_loss_scores_the_supervised_positions_only() -> None:
    """The loss is cross entropy over the masked positions, and nothing else.

    Equated against the gathered positions directly. An unmasked loss would train the
    model to emit the label at every position, and on the automaton tasks -- one
    supervised position in up to 256 -- it would be dominated by the padding.
    """
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, 4, generator=gen)
    targets = torch.randint(0, 4, (2, 5), generator=gen)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 4] = True
    mask[1, 2] = True
    expected = nn.functional.cross_entropy(
        torch.stack([logits[0, 4], logits[1, 2]]),
        torch.stack([targets[0, 4], targets[1, 2]]),
    )
    got = masked_loss(logits, targets, mask)
    assert torch.allclose(got, expected)
    assert got.dtype == torch.float32


def test_masked_loss_is_float32_under_a_narrow_forward() -> None:
    """The softmax runs at float32 whatever the forward pass ran at.

    A bf16 softmax over a 60-class group vocabulary loses the tail the loss is measuring,
    and the loss is the only signal the group half has.
    """
    logits = torch.randn(1, 3, 60, dtype=torch.bfloat16)
    targets = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = masked_loss(logits, targets, mask)
    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss))


def test_masked_loss_refuses_a_batch_it_cannot_score() -> None:
    """An all-False mask would return nan and train on it."""
    logits = torch.zeros(1, 2, 3)
    targets = torch.zeros(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="supervises no position"):
        masked_loss(logits, targets, torch.zeros(1, 2, dtype=torch.bool))


def test_bands_are_keyed_on_the_items_own_length() -> None:
    """A band holds the supervised positions of the items whose length falls in it.

    Band ``k`` is lengths ``[k * width, (k + 1) * width - 1]``. The accuracy is pooled
    over positions, not averaged over items, which is upstream's ``val_acc / val_num``.
    """
    task = resolve("A5")
    short, long = task.sample(0, 3, 3), task.sample(0, 9, 9)
    batch = collate([short, long])
    logits = torch.zeros(2, batch.width, 60)
    # Right on every position of the short item, wrong on every position of the long one.
    for position, target in enumerate(short.targets):
        logits[0, position, target] = 1.0
    for position, target in enumerate(long.targets):
        logits[1, position, (target + 1) % 60] = 1.0
    tally = _Tally(band_width=4)
    tally.add(logits, batch)
    metrics = tally.metrics()
    assert metrics.positions == 3 + 9
    assert metrics.accuracy == pytest.approx(3 / 12)
    assert metrics.bands == (Band(0, 3, 3, 1.0), Band(8, 11, 9, 0.0))


def test_empty_split_reports_zeros_rather_than_dividing_by_nothing() -> None:
    """A tally with no supervised position reports zeros and no bands."""
    assert _Tally(band_width=8).metrics() == Metrics(0.0, 0.0, 0, ())


def test_evaluate_restores_the_models_mode() -> None:
    """Evaluation leaves the model in the mode it was handed.

    Dropout is on during training and off during evaluation; a model left in eval mode
    after the first evaluation would train without dropout for the rest of the run.
    """
    model = _model()
    staged = stage(AUTOMATA["parity"], VAL_SPLIT, TINY)
    assert len(staged) == 2
    model.train()
    metrics = evaluate(model, staged, TINY)
    assert model.training is True
    assert metrics.positions == 8
    model.eval()
    evaluate(model, staged, TINY)
    assert model.training is False


def test_stage_refuses_an_unbounded_split() -> None:
    """An unbounded split has no last batch, so it cannot be staged or scored."""
    with pytest.raises(ValueError, match="cannot be staged"):
        stage(AUTOMATA["parity"], TRAIN_SPLIT, TINY)


def test_the_protocol_runs_and_reports_every_evaluation() -> None:
    """A whole run of the loop, at four steps, on the CPU.

    Evaluations land at ``step % print_steps == 0`` and at the last step, so at 4 steps
    and ``print_steps`` 2 that is steps 0, 2 and 3. ``final`` is the last of them and
    ``best`` is the highest-accuracy one, selected on accuracy rather than loss.
    """
    _, report = _run(TINY)
    assert report.steps_run == 4
    assert [point.step for point in report.points] == [0, 2, 3]
    assert report.solved is False
    assert report.final == report.points[-1].val
    accuracies = [point.val.accuracy for point in report.points]
    assert report.best.accuracy == max(accuracies)
    assert report.best_step in [point.step for point in report.points]
    assert all(point.val.positions == 8 for point in report.points)
    assert all(math.isfinite(point.train_loss) for point in report.points)
    assert all(point.lr > 0.0 for point in report.points)


def test_early_stop_breaks_after_the_first_evaluation_over_the_threshold() -> None:
    """The run stops the first time accuracy passes the threshold, and says so.

    Forced with a threshold below zero, so the break path is exercised without needing a
    solved task. The report still carries the evaluation that triggered it.
    """
    _, report = _run(replace(TINY, early_stop_threshold=-1.0))
    assert report.solved is True
    assert report.steps_run == 1
    assert [point.step for point in report.points] == [0]
    assert report.best_step == 0


def test_one_parameter_update_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optimizer steps once, not twice.

    Upstream steps twice per update with ``zero_grad`` between, so the second step applies
    the update again from stale momentum and charges decoupled decay twice. Counted on the
    optimizer itself: the loop must call it exactly ``num_steps`` times.
    """
    calls: list[int] = []
    original = torch.optim.AdamW.step

    def counted(self: torch.optim.AdamW, *args: object, **kwargs: object) -> None:
        calls.append(1)
        original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted)
    _run(replace(TINY, num_steps=3, print_steps=8))
    assert len(calls) == 3


def _micro_losses(task: Task, count: int) -> list[float]:
    """The first ``count`` train batches' losses at initialization.

    Args:
        task: The task.
        count: Batches.

    Returns:
        One loss per batch, all at the same weights, which is what an accumulation window
        sees before its single update.
    """
    seed_all(TINY.seed)
    model = _model()
    stream = batches(task, TRAIN_SPLIT, TINY.batch_size)
    out: list[float] = []
    with torch.no_grad():
        for _ in range(count):
            batch = next(stream)
            logits = model(batch.inputs)
            out.append(float(masked_loss(logits, batch.targets, batch.mask)))
    return out


def test_accumulation_sees_every_micro_batch() -> None:
    """The reported loss over an accumulated step is the mean over its micro-batches.

    Upstream zeroes the gradient inside the window, so an accumulating run trains on the
    last micro-batch alone and reports only its loss. Here the window is the whole step:
    at ``accumulation_steps`` 1 the first point is the first batch's loss, and at 2 it is
    the mean of the first two, both computed independently at initialization.
    """
    losses = _micro_losses(AUTOMATA["parity"], 2)
    single = _run(replace(TINY, num_steps=1, print_steps=1))[1]
    doubled = _run(replace(TINY, num_steps=1, print_steps=1, accumulation_steps=2))[1]
    assert single.points[0].train_loss == pytest.approx(losses[0], rel=1e-6)
    assert doubled.points[0].train_loss == pytest.approx(sum(losses) / 2, rel=1e-6)
    assert losses[0] != pytest.approx(losses[1], rel=1e-6)


def test_a_models_own_gradient_hook_is_called_every_step() -> None:
    """A model exposing ``mask_grads`` has it called after backward, before the step.

    `expressive-sparse-state-space-model`'s own model carries that hook, so a baseline
    slotted in here needs it honoured or its constrained parameters drift.
    """
    hits: list[int] = []

    class Hooked(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(3, D_MODEL)
            self.head = nn.Linear(D_MODEL, 3)

        def mask_grads(self) -> None:
            hits.append(1)

        def forward(self, tokens: Tensor) -> Tensor:
            return self.head(self.embedding(tokens))

    train(
        Hooked(),
        AUTOMATA["parity"],
        TRAIN_SPLIT,
        VAL_SPLIT,
        replace(TINY, num_steps=3, print_steps=8),
    )
    assert len(hits) == 3
