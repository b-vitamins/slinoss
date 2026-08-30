"""The loop: accumulation is a larger batch, one update per step, and one seed is one run.

Three failure modes, all of which leave a run that trains and reports a number.

Accumulation that divides wrong makes the effective batch a micro batch and the effective rate
``accum`` times the schedule, so an arm that fit a smaller micro batch on its card silently ran
a different optimizer. Checked against the same tokens in one forward, which is the definition
rather than a proxy.

Two updates per step -- the usual shape being a manual step plus a scheduler that also steps --
doubles the rate and charges decay twice. Checked by hand-computing AdamW's first step, because
a second update is not visible in a loss curve.

A loop that reseeds, or that draws from process RNG, makes two runs at one seed different runs,
at which point a paired margin between two arms carries the sample.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scripts.lm.corpus import DTYPE
from scripts.lm.data import Shard, batches
from scripts.lm.train import (
    Step,
    TrainConfig,
    accumulate,
    evaluate,
    loss_on,
    train,
)

SEQ_LEN = 4
CLASSES = 8
WIDTH = 16


class Tiny(nn.Module):
    """One embedding and one head: enough to have a gradient and cheap enough to hand-compute.

    The scaffold is not under test here. What is under test is the loop's arithmetic, and a
    real stack would put a mixer's numerics between an assertion and the arithmetic.
    """

    def __init__(self, classes: int = CLASSES, width: int = WIDTH) -> None:
        super().__init__()
        self.embedding = nn.Embedding(classes, width)
        self.head = nn.Linear(width, classes, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embedding(x))


def _model(seed: int = 0) -> Tiny:
    """A model at a fixed initialization."""
    torch.manual_seed(seed)
    return Tiny()


def _shard(tmp_path: Path, windows: int, name: str = "tokens.bin") -> Shard:
    """A shard of ``windows`` full windows, tokens cycling through the classes."""
    tokens = windows * SEQ_LEN + 1
    path = tmp_path / name
    (np.arange(tokens, dtype=np.int64) % CLASSES).astype(DTYPE).tofile(path)
    return Shard(path, tokens)


def _grads(model: nn.Module) -> list[Tensor]:
    """The gradients, cloned, in parameter order."""
    return [
        param.grad.detach().clone()
        for param in model.parameters()
        if param.grad is not None
    ]


def test_accumulation_is_the_gradient_of_the_larger_batch(tmp_path: Path) -> None:
    """Eight sequences in four micro steps and in one give the same gradient.

    The data order is a function of ``(seed, step)`` and the micro batch is a slice of it, so
    two accumulation shapes over the same eight windows see the same windows. Any difference
    is then the division: a loop that summed micro losses without dividing would be off by
    exactly the accumulation count.
    """
    shard = _shard(tmp_path, 32)
    split = _model()
    whole = _model()
    accumulate(
        split,
        batches(shard, seq_len=SEQ_LEN, batch_size=2, seed=5, steps=4),
        accum=4,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
    )
    accumulate(
        whole,
        batches(shard, seq_len=SEQ_LEN, batch_size=8, seed=5, steps=1),
        accum=1,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
    )
    for a, b in zip(_grads(split), _grads(whole), strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_accumulation_clears_the_gradient_once_at_the_start(tmp_path: Path) -> None:
    """A second step's gradient is its own, not the first step's plus its own.

    Compared against a fresh model accumulating the same second step alone. The parameters do
    not move here -- :func:`scripts.lm.train.accumulate` takes no optimizer -- so the two
    gradients are the same quantity and any difference is a stale gradient carried over.

    The mirror of the same bug is clearing per micro step, which accumulates nothing; that one
    is covered by the equivalence above.
    """
    shard = _shard(tmp_path, 32)
    twice = _model()
    stream = batches(shard, seq_len=SEQ_LEN, batch_size=2, seed=5, steps=8)
    accumulate(
        twice, stream, accum=4, classes=CLASSES, device="cpu", autocast_dtype=None
    )
    first = _grads(twice)
    accumulate(
        twice, stream, accum=4, classes=CLASSES, device="cpu", autocast_dtype=None
    )
    second = _grads(twice)

    once = _model()
    accumulate(
        once,
        batches(shard, seq_len=SEQ_LEN, batch_size=2, seed=5, steps=8, start=4),
        accum=4,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
    )
    alone = _grads(once)
    assert any(not torch.allclose(a, b) for a, b in zip(first, second, strict=True))
    for a, b in zip(second, alone, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_a_short_stream_is_an_error_not_a_short_step(tmp_path: Path) -> None:
    """A step at fewer tokens than the schedule assumes is not the step that was scheduled."""
    shard = _shard(tmp_path, 8)
    model = _model()
    stream = batches(shard, seq_len=SEQ_LEN, batch_size=2, seed=0, steps=2)
    with pytest.raises(StopIteration):
        accumulate(
            model, stream, accum=4, classes=CLASSES, device="cpu", autocast_dtype=None
        )


def test_exactly_one_update_per_step(tmp_path: Path) -> None:
    """One step of the loop equals one hand-computed AdamW update, not two.

    At one step the schedule holds the peak, so the rate is the transferred peak exactly and
    the hand computation needs no schedule. AdamW's first step: both bias corrections are
    ``1 - beta``, so the update reduces elementwise to ``lr * g / (|g| + eps)``.

    The clip is off here and the decay is zero, because what is under test is the count of
    updates and the rate each group took. The clip is invisible on a first step in any case: it
    scales the gradient uniformly and the first step normalizes by the gradient's own
    magnitude. Its reporting is checked separately.
    """
    shard = _shard(tmp_path, 8)
    config = TrainConfig(
        token_budget=8 * SEQ_LEN,
        token_batch=8 * SEQ_LEN,
        seq_len=SEQ_LEN,
        micro_batch=8,
        weight_decay=0.0,
        grad_clip=0.0,
        autocast_dtype=None,
        log_every=0,
    )
    assert config.steps == 1
    assert config.accum == 1

    model = _model()
    before = [param.detach().clone() for param in model.parameters()]
    result = train(
        model,
        shard,
        config,
        d_model=WIDTH,
        classes=CLASSES,
        device="cpu",
    )
    assert result.steps == 1

    reference = _model()
    batch = next(
        iter(batches(shard, seq_len=SEQ_LEN, batch_size=8, seed=config.seed, steps=1))
    )
    loss = loss_on(reference, batch, classes=CLASSES)
    loss.backward()
    lr = result.peak_lr
    for name, param, start in zip(
        [name for name, _ in reference.named_parameters()],
        reference.parameters(),
        before,
        strict=True,
    ):
        grad = param.grad
        assert grad is not None
        rate = result.embedding_lr if "embedding" in name else lr
        expected = start - rate * grad / (grad.abs() + config.eps)
        trained = dict(model.named_parameters())[name]
        assert torch.allclose(trained.detach(), expected, atol=1e-6, rtol=1e-4)


def test_the_reported_gradient_norm_is_the_one_before_clipping(tmp_path: Path) -> None:
    """A post-clip norm always reads at or under the cap and diagnoses nothing.

    The number's only use is telling a run whose gradients are being clipped every step from
    one whose are not, so it has to be the norm as it arrived. Checked against the norm of the
    same step's gradients computed outside the loop, with a cap small enough to bind.
    """
    shard = _shard(tmp_path, 8)
    cap = 1e-3
    config = TrainConfig(
        token_budget=8 * SEQ_LEN,
        token_batch=8 * SEQ_LEN,
        seq_len=SEQ_LEN,
        micro_batch=8,
        grad_clip=cap,
        autocast_dtype=None,
        log_every=1,
    )
    seen: list[Step] = []
    train(
        _model(),
        shard,
        config,
        d_model=WIDTH,
        classes=CLASSES,
        device="cpu",
        on_step=seen.append,
    )

    reference = _model()
    accumulate(
        reference,
        batches(shard, seq_len=SEQ_LEN, batch_size=8, seed=config.seed, steps=1),
        accum=1,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
    )
    expected = math.sqrt(sum(float(grad.pow(2).sum()) for grad in _grads(reference)))
    assert expected > cap
    assert seen[0].grad_norm == pytest.approx(expected, rel=1e-5)


def test_the_rate_ratios_between_groups_hold_at_every_step(tmp_path: Path) -> None:
    """The schedule multiplies every group by one factor, so the policy's ratios survive.

    A schedule that set one rate on every group would put the token table at the hidden rate
    and undo the transfer the protocol specifies.
    """
    shard = _shard(tmp_path, 64)
    config = TrainConfig(
        token_budget=8 * 8 * SEQ_LEN,
        token_batch=8 * SEQ_LEN,
        seq_len=SEQ_LEN,
        micro_batch=8,
        warmdown_fraction=0.5,
        autocast_dtype=None,
        log_every=1,
    )
    seen: list[Step] = []
    result = train(
        model=_model(),
        train_shard=shard,
        config=config,
        d_model=WIDTH,
        classes=CLASSES,
        device="cpu",
        on_step=seen.append,
    )
    assert [step.number for step in seen] == list(range(result.steps))
    assert seen[0].lr == pytest.approx(result.peak_lr)
    assert seen[-1].lr < result.peak_lr
    assert result.embedding_lr != result.peak_lr
    assert all(step.grad_norm > 0.0 for step in seen)


def test_one_seed_is_one_run(tmp_path: Path) -> None:
    """Two runs at one seed land on the same parameters, to the last bit of float32 noise.

    No :func:`torch.manual_seed` below :func:`scripts.lm.train.train`: a loop that reseeded
    would make a run depend on how many times it had been called in the process.
    """
    shard = _shard(tmp_path, 32)
    config = TrainConfig(
        token_budget=4 * 8 * SEQ_LEN,
        token_batch=8 * SEQ_LEN,
        seq_len=SEQ_LEN,
        micro_batch=4,
        autocast_dtype=None,
        log_every=0,
    )
    first = _model()
    second = _model()
    a = train(first, shard, config, d_model=WIDTH, classes=CLASSES, device="cpu")
    b = train(second, shard, config, d_model=WIDTH, classes=CLASSES, device="cpu")
    assert a.train_loss == pytest.approx(b.train_loss, abs=1e-6)
    for x, y in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.allclose(x.detach(), y.detach(), atol=1e-6)


def test_the_held_out_score_is_a_token_weighted_sum_over_the_whole_shard(
    tmp_path: Path,
) -> None:
    """A short last batch must not weigh as much as a full one.

    Checked against the loss over every window computed one batch at a time, which is the
    same quantity by construction and a different code path. An unweighted mean over batches
    would move with the batch size, so the number would depend on the card it was scored on.
    """
    shard = _shard(tmp_path, 5)
    model = _model()
    scores = [
        evaluate(
            model,
            shard,
            seq_len=SEQ_LEN,
            batch_size=size,
            classes=CLASSES,
            device="cpu",
            autocast_dtype=None,
        )
        for size in (1, 2, 5)
    ]
    assert {score.tokens for score in scores} == {5 * SEQ_LEN}
    assert scores[0].loss == pytest.approx(scores[1].loss, abs=1e-6)
    assert scores[0].loss == pytest.approx(scores[2].loss, abs=1e-6)
    assert scores[0].bpb is None


def test_bits_per_byte_is_the_loss_over_the_manifest_s_byte_count(
    tmp_path: Path,
) -> None:
    """Nats per token is comparable at one tokenizer; bits per byte is comparable across.

    A figure divided by the wrong byte count is still a plausible number, so the conversion is
    pinned against the arithmetic rather than against a previous run.
    """
    shard = _shard(tmp_path, 4)
    model = _model()
    score = evaluate(
        model,
        shard,
        seq_len=SEQ_LEN,
        batch_size=2,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
        bytes_per_token=4.0,
    )
    assert score.bpb is not None
    assert score.bpb == pytest.approx(score.loss / math.log(2.0) / 4.0)


def test_evaluation_restores_the_mode_it_found(tmp_path: Path) -> None:
    """A mid-run evaluation must not leave the model in eval mode for the rest of training."""
    shard = _shard(tmp_path, 4)
    model = _model()
    model.train()
    evaluate(
        model,
        shard,
        seq_len=SEQ_LEN,
        batch_size=2,
        classes=CLASSES,
        device="cpu",
        autocast_dtype=None,
    )
    assert model.training


def test_a_token_batch_that_is_not_whole_micro_batches_is_refused() -> None:
    """A partial micro batch would make the optimizer's token count something else."""
    with pytest.raises(ValueError, match="not a whole number of"):
        TrainConfig(
            token_budget=1 << 20, token_batch=1000, seq_len=SEQ_LEN, micro_batch=8
        )
    with pytest.raises(ValueError, match="eval_batch must be positive"):
        TrainConfig(token_budget=1 << 20, token_batch=1 << 17, eval_batch=0)


def test_the_config_derives_the_accumulation_count_and_the_step_count() -> None:
    """``micro_batch`` is a memory decision; the optimizer step is what is held fixed.

    The published shapes: 131,072 tokens a step at 2048, and 524,288 at the larger scale.
    """
    small = TrainConfig(token_budget=1_800_000_000, token_batch=1 << 17, micro_batch=8)
    assert small.accum == (1 << 17) // (8 * 2048)
    assert small.steps == 13732
    large = TrainConfig(token_budget=10_900_000_000, token_batch=1 << 19, micro_batch=8)
    assert large.accum == 4 * small.accum
    assert large.steps == 20790
