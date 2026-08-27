"""The protocol: the schedule, the stopping rule, and what the two accuracies mean.

Every number a run reports comes out of this module, so each of its decisions is pinned
against a computation rather than a description: the cosine rate against torch's own
scheduler, the stopping rule against a model that solves the task on the first epoch, and
the accuracy pair against a pool whose two segments carry different key-value counts, which
is the only configuration in which they differ.

The models below are hand-built rather than trained. One copies its input and one predicts
a fixed token, so what the assertions read is the loop and the metric, not an optimization.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from scripts.mqar.instances import IGNORE_INDEX
from scripts.mqar.tasks import Segment, SegmentSpec, batches
from scripts.mqar.train import (
    TrainConfig,
    _autocast,
    _auxiliary_loss,
    batch_count,
    evaluate,
    lr_at,
    train,
)

VOCAB = 8
"""Vocabulary of every pool below. Small enough that a loss is checkable by hand."""

LOGIT = 10.0
"""Logit a stub model puts on its prediction. Everything else stays at 0."""

PREDICTED = 5
"""The token :class:`Constant` always predicts."""

COPY_ROWS = (
    ("1 2 3", "x x 3"),
    ("4 5 6", "x x 6"),
)
"""``(inputs, labels)``, ``x`` for :data:`IGNORE_INDEX`. Solved exactly by :class:`Copy`."""

ONE_PAIR_ROWS = (
    ("1 2 1", "x x 5"),
    ("3 4 3", "x x 5"),
)
"""One supervised position per row, and :class:`Constant` gets it. Fraction 1."""

THREE_PAIR_ROWS = (
    ("1 2 3 4", "x 5 6 7"),
    ("2 3 4 1", "x 5 6 7"),
)
"""Three supervised positions per row, one of which :class:`Constant` gets. Fraction 1/3."""


class Copy(nn.Module):
    """Predicts its own input token at every position."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, ids: Tensor) -> Tensor:
        """``(B, T)`` int64 in, ``(B, T, VOCAB)`` out."""
        one_hot = nn.functional.one_hot(ids, VOCAB).float()
        return one_hot * LOGIT + self.gain * 0.0


class Constant(nn.Module):
    """Predicts :data:`PREDICTED` at every position, whatever its parameters become."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, ids: Tensor) -> Tensor:
        """``(B, T)`` int64 in, ``(B, T, VOCAB)`` out."""
        logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB)
        logits[..., PREDICTED] = LOGIT
        return logits + self.gain * 0.0


class WithPenalty(nn.Module):
    """A wrapper carrying a constant auxiliary term, which is upstream's mixer hook."""

    def __init__(self, inner: nn.Module, penalty: float) -> None:
        super().__init__()
        self.inner = inner
        self.penalty = penalty

    def forward(self, ids: Tensor) -> Tensor:
        """Delegate to the wrapped model."""
        return self.inner(ids)

    def get_auxiliary_loss(self) -> Tensor:
        """The constant term the loop is expected to add."""
        return torch.tensor(self.penalty)


def segment(rows: tuple[tuple[str, str], ...], num_kv_pairs: int) -> Segment:
    """A segment from literal rows.

    The generator is pinned in :mod:`tests.test_mqar_instances`; the loop takes segments as
    given, so literal ones keep a failure here local to the loop.

    Args:
        rows: ``(inputs, labels)`` per row, space-separated, ``x`` for the ignored label.
        num_kv_pairs: The slice value to carry, and the supervised count per row.

    Returns:
        A :class:`Segment`.
    """
    inputs = torch.tensor([[int(token) for token in row.split()] for row, _ in rows])
    labels = torch.tensor(
        [
            [IGNORE_INDEX if token == "x" else int(token) for token in row.split()]
            for _, row in rows
        ]
    )
    length = inputs.shape[1]
    return Segment(
        spec=SegmentSpec(
            input_seq_len=length, num_kv_pairs=num_kv_pairs, num_examples=len(rows)
        ),
        seed=0,
        inputs=inputs.numpy(),
        labels=labels.numpy(),
        slices={"input_seq_len": length, "num_kv_pairs": num_kv_pairs},
    )


def cpu_config(**kwargs: object) -> TrainConfig:
    """A protocol on the CPU at batch size 2, everything else at the tree's default."""
    settings: dict[str, object] = {"device": "cpu", "batch_size": 2, "max_epochs": 4}
    settings.update(kwargs)
    return TrainConfig(**settings)  # pyright: ignore[reportArgumentType]


def divergent_pool() -> list[Segment]:
    """Two segments at one and three supervised positions per row."""
    return [segment(ONE_PAIR_ROWS, 1), segment(THREE_PAIR_ROWS, 3)]


def test_lr_at_is_torchs_own_cosine_schedule() -> None:
    """The closed form of ``CosineAnnealingLR(T_max=max_epochs, eta_min=0.0)``.

    Read off the scheduler rather than restated, because the whole point of the closed form
    is that the loop does not carry a scheduler object whose state could desynchronize from
    the epoch it is reporting.
    """
    config = cpu_config(max_epochs=8, lr=1e-3)
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=0.0
    )
    observed: list[float] = []
    for _ in range(config.max_epochs):
        observed.append(float(optimizer.param_groups[0]["lr"]))
        # Train-then-step is the loop's own order, and torch warns about the other one.
        optimizer.step()
        scheduler.step()
    assert observed == pytest.approx([lr_at(config, e) for e in range(8)])
    assert lr_at(config, 0) == config.lr
    assert lr_at(config, 8) == 0.0
    assert lr_at(config, 7) > 0.0


def test_early_stopping_fires_before_the_rate_decays() -> None:
    """A run that solves the task on epoch 0 stops there, at the undecayed rate.

    The order is the measurement decision: evaluate, check, and only then advance the
    schedule. A run that stopped after decaying would report a rate it never trained at.
    """
    report = train(
        Copy(), [segment(COPY_ROWS, 1)], [segment(COPY_ROWS, 1)], cpu_config()
    )
    assert report.stopped_early is True
    assert report.epochs_run == 1
    assert report.points[0].lr == cpu_config().lr
    assert report.final.example == 1.0
    assert report.best_epoch == 0
    assert report.best is report.points[0].test
    assert report.final is report.points[-1].test


def test_the_stopping_rule_is_strict() -> None:
    """At threshold 1.0 a perfect run does not stop, which is how the sweep disables it."""
    config = cpu_config(max_epochs=3, early_stopping_threshold=1.0)
    report = train(Copy(), [segment(COPY_ROWS, 1)], [segment(COPY_ROWS, 1)], config)
    assert report.stopped_early is False
    assert report.epochs_run == 3
    assert [point.lr for point in report.points] == [lr_at(config, e) for e in range(3)]
    assert report.final.example == 1.0


def test_the_auxiliary_term_enters_the_objective() -> None:
    """A mixer's own regularizer is added to the batch loss, not merely collected.

    Upstream's hook, kept because it is how a mixer contributes a penalty without the loop
    knowing about it. The model solves the task, so the loss is the penalty and nothing else.
    """
    penalty = 7.0
    model = WithPenalty(Copy(), penalty)
    report = train(
        model, [segment(COPY_ROWS, 1)], [segment(COPY_ROWS, 1)], cpu_config()
    )
    assert report.points[0].train_loss == pytest.approx(penalty, abs=0.01)


def test_auxiliary_terms_sum_and_are_absent_when_nobody_declares_one() -> None:
    """Every submodule carrying the hook contributes; None when none does."""
    assert _auxiliary_loss(Copy()) is None
    nested = WithPenalty(WithPenalty(Copy(), 3.0), 2.0)
    total = _auxiliary_loss(nested)
    assert total is not None
    assert float(total) == pytest.approx(5.0)


def test_the_two_accuracies_diverge_across_unequal_segments() -> None:
    """``example`` weights a row, ``position`` weights a supervised position.

    They coincide inside a segment and separate across a pool whose segments carry
    different key-value counts, which is exactly the published multi-segment protocol.
    ``example`` is upstream's, and the one every MQAR figure plots.
    """
    metrics = evaluate(Constant(), divergent_pool(), cpu_config())
    assert metrics.example == pytest.approx((1.0 + 1.0 + 1 / 3 + 1 / 3) / 4)
    assert metrics.position == pytest.approx(4 / 8)
    assert metrics.example != pytest.approx(metrics.position)


def test_slices_group_by_every_key_a_segment_carries() -> None:
    """Sliced accuracy is the ``example`` number restricted to a slice value.

    Which is what makes a recall curve against key-value count readable off one run.
    """
    metrics = evaluate(Constant(), divergent_pool(), cpu_config())
    assert metrics.by_slice["num_kv_pairs"] == pytest.approx({"1": 1.0, "3": 1 / 3})
    assert metrics.by_slice["input_seq_len"] == pytest.approx({"3": 1.0, "4": 1 / 3})


def test_loss_is_weighted_by_supervised_position() -> None:
    """Not the unweighted mean of per-batch means, which upstream reports.

    The two differ on any pool whose batches hold different supervised counts, and both
    candidates are computed here so the divergence is shown rather than asserted.
    """
    segments = divergent_pool()
    config = cpu_config()
    model = Constant().eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    per_batch: list[float] = []
    weighted = 0.0
    supervised = 0
    for batch in batches(segments, config.eval_batch_size):
        inputs = torch.from_numpy(batch.inputs)
        labels = torch.from_numpy(batch.labels)
        with torch.no_grad():
            logits = model(inputs).float().flatten(0, -2)
        per_batch.append(float(loss_fn(logits, labels.flatten())))
        count = int((batch.labels != IGNORE_INDEX).sum())
        weighted += per_batch[-1] * count
        supervised += count
    metrics = evaluate(model, segments, config)
    assert metrics.loss == pytest.approx(weighted / supervised)
    assert metrics.loss != pytest.approx(sum(per_batch) / len(per_batch))


def test_evaluate_refuses_a_pool_with_nothing_to_score() -> None:
    """The alternative is a division by zero reported as an accuracy of zero."""
    unsupervised = segment((("1 2 3", "x x x"),), 1)
    with pytest.raises(ValueError, match="no supervised position"):
        evaluate(Constant(), [unsupervised], cpu_config())


def test_autocast_is_a_region_not_a_cast() -> None:
    """``bf16`` casts activations inside the forward and leaves parameters at float32.

    A mixer whose CUDA path is only reachable at bf16 gets there this way. Nothing else in
    the loop changes: the loss is taken in float32 either way.
    """
    x = torch.randn(2, 2)
    layer = nn.Linear(2, 2)
    with _autocast(cpu_config(precision="fp32")):
        assert layer(x).dtype == torch.float32
    with _autocast(cpu_config(precision="bf16")):
        assert layer(x).dtype == torch.bfloat16
    assert layer.weight.dtype == torch.float32


def test_batch_count_agrees_with_the_batches_it_counts() -> None:
    """Tails included. It sizes a progress estimate, so an off-by-one is visible."""
    segments = [segment(ONE_PAIR_ROWS, 1), segment(THREE_PAIR_ROWS, 3)]
    assert batch_count(segments, 1) == 4
    for size in (1, 2, 3):
        assert batch_count(segments, size) == len(list(batches(segments, size)))


def test_eval_batch_size_resolves_zero_to_the_train_size() -> None:
    """Figure 2 evaluates at the train batch size; the modern repro at an eighth of it."""
    assert cpu_config(batch_size=256, test_batch_size=0).eval_batch_size == 256
    assert cpu_config(batch_size=256, test_batch_size=32).eval_batch_size == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_epochs": 0}, "max_epochs must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"test_batch_size": -1}, "test_batch_size must be non-negative"),
        ({"lr": 0.0}, "lr must be positive"),
        ({"weight_decay": -0.1}, "weight_decay must be non-negative"),
        ({"precision": "fp16"}, "precision must be one of"),
    ],
)
def test_out_of_contract_protocols_raise(
    kwargs: dict[str, object], message: str
) -> None:
    """Caught at the config, before a pool is generated or an epoch is run."""
    with pytest.raises(ValueError, match=message):
        cpu_config(**kwargs)


def test_train_loss_is_finite_on_a_model_that_learns_nothing() -> None:
    """A constant model still produces a usable curve rather than a nan.

    Its prediction does not move, so every epoch reports the same loss and the same
    accuracy, and the run goes the distance. That is the shape a failed arm has, and a nan
    there would be read as a harness fault.
    """
    config = cpu_config(max_epochs=2, early_stopping_threshold=1.0)
    report = train(Constant(), divergent_pool(), divergent_pool(), config)
    assert report.epochs_run == 2
    assert all(math.isfinite(point.train_loss) for point in report.points)
    assert report.points[0].test.example == pytest.approx(report.points[1].test.example)
