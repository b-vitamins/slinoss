"""The loop. Accumulate, clip, step once, evaluate.

Imports torch and numpy and nothing else, by way of :mod:`scripts.lm.data` reading a plain
memmap. A loop that pulls in a dataset library is a loop whose behaviour depends on that
library's version; this one depends on a file with a digest.

Three shapes are avoided deliberately, all three found in trainers on the shelf.

    two ``step()`` calls per iteration      every update applied twice, decay charged twice,
                                            so the effective rate is 2x the printed schedule
    ``zero_grad()`` per micro-step          accumulation accumulates nothing
    ``manual_seed`` inside the loop         the process RNG is reseeded per item

So: one :meth:`torch.optim.Optimizer.zero_grad` before the micro-steps, each micro-loss
divided by the accumulation count, one clip over the summed gradient, one
:meth:`torch.optim.Optimizer.step`, and no global seeding anywhere below :func:`train`.

Precision is bf16 autocast over float32 parameters and float32 optimizer state. The protocol
this reproduces ran float32 throughout; float32 here would take the operator's reference path
and measure a different program, so the deviation is named rather than hidden, and it is the
same deviation for every arm.

The loss is :func:`slinoss.ops.xent.cross_entropy`, which takes the class count separately
from the operand width. That is the padded head exactly: the head emits
``padded_vocab_size`` columns and only the first ``vocab_size`` are classes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
from torch import Tensor, nn

from scripts.lm.data import Batch, Shard, batches, val_batches
from scripts.lm.groups import GroupPolicy, parameter_groups
from scripts.lm.schedule import lr_at, steps_for, transfer
from slinoss.ops.xent import cross_entropy

__all__ = [
    "Evaluation",
    "Step",
    "TrainConfig",
    "TrainResult",
    "accumulate",
    "evaluate",
    "loss_on",
    "train",
]

_LOG2E = 1.0 / math.log(2.0)


@dataclass(frozen=True)
class TrainConfig:
    """Everything a run needs that is not the model or the data.

    ``token_batch`` is the optimizer step's token count and is the quantity held fixed
    across arms; ``micro_batch`` is a memory decision and the accumulation count follows from
    the two. An arm that changes ``micro_batch`` to fit a card takes the same optimizer step.

    Attributes:
        token_budget: Tokens the run trains on.
        token_batch: Tokens per optimizer step.
        seq_len: Sequence length.
        micro_batch: Sequences per forward.
        base_lr: Hidden-group rate before transfer.
        embedding_base_lr: Token-table rate before transfer.
        ssm_multiplier: Multiple of the transferred rate the state-space group runs at.
        weight_decay: Decay for the decayed groups.
        betas: AdamW betas.
        eps: AdamW epsilon.
        grad_clip: Global gradient norm cap. Zero disables.
        warmdown_fraction: Fraction of the run spent falling.
        final_fraction: Floor, as a fraction of the peak.
        seed: Fixes the data order and, through the caller, the initialization.
        eval_batch: Sequences per forward during validation.
        log_every: Steps between progress callbacks. Zero silences them.
        autocast_dtype: Compute dtype under autocast, or None for none.
    """

    token_budget: int
    token_batch: int = 1 << 17
    seq_len: int = 2048
    micro_batch: int = 8
    base_lr: float = 4e-3
    embedding_base_lr: float = 0.3
    ssm_multiplier: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.8, 0.95)
    eps: float = 1e-10
    grad_clip: float = 3.0
    warmdown_fraction: float = 0.4
    final_fraction: float = 0.0
    seed: int = 0
    eval_batch: int = 8
    log_every: int = 50
    autocast_dtype: torch.dtype | None = field(default=torch.bfloat16)

    def __post_init__(self) -> None:
        window = self.micro_batch * self.seq_len
        if window < 1:
            raise ValueError(
                f"micro_batch {self.micro_batch} and seq_len {self.seq_len} must be "
                f"positive"
            )
        if self.token_batch % window != 0:
            raise ValueError(
                f"token_batch {self.token_batch} is not a whole number of "
                f"{self.micro_batch}x{self.seq_len} micro batches"
            )
        if self.eval_batch < 1:
            raise ValueError(f"eval_batch must be positive, got {self.eval_batch}")

    @property
    def accum(self) -> int:
        """Micro batches per optimizer step.

        Returns:
            The count, at least one.
        """
        return self.token_batch // (self.micro_batch * self.seq_len)

    @property
    def steps(self) -> int:
        """Optimizer steps the budget buys.

        Returns:
            The count.

        Raises:
            ValueError: From :func:`scripts.lm.schedule.steps_for`.
        """
        return steps_for(self.token_budget, self.token_batch)


class Step(NamedTuple):
    """One optimizer step's record, for the progress callback.

    Attributes:
        number: Which step, from zero. Not ``index``, which would shadow
            :meth:`tuple.index` on a named tuple.
        loss: Mean training loss over the step's micro batches, in nats.
        lr: The hidden group's rate at this step.
        grad_norm: Gradient norm before clipping.
    """

    number: int
    loss: float
    lr: float
    grad_norm: float


class Evaluation(NamedTuple):
    """A held-out score.

    Attributes:
        loss: Mean cross entropy in nats per token.
        bpb: Bits per byte of the original text, or None when the manifest carries no
            byte count. A cross-tokenizer comparison needs bits per byte; nats per token
            is only comparable at one tokenizer.
        tokens: Tokens scored.
    """

    loss: float
    bpb: float | None
    tokens: int


class TrainResult(NamedTuple):
    """What a finished run reports.

    Attributes:
        steps: Optimizer steps taken.
        tokens: Tokens consumed, ``steps * token_batch``.
        train_loss: Mean loss over the last :func:`train`-internal window of steps.
        val: The held-out score, or None when no validation shard was given.
        peak_lr: The transferred hidden rate. Recorded post-transfer so a replay at a
            changed width cannot re-derive a different number.
        embedding_lr: The transferred token-table rate.
        accum: Micro batches per step.
    """

    steps: int
    tokens: int
    train_loss: float
    val: Evaluation | None
    peak_lr: float
    embedding_lr: float
    accum: int


def loss_on(model: nn.Module, batch: Batch, *, classes: int) -> Tensor:
    """Cross entropy of one batch.

    Args:
        model: Maps ``(B,T)`` ids to ``(B,T,width)`` logits.
        batch: Inputs and targets, already on the model's device.
        classes: Tokens the labels index. Not the logits' width: the head is padded and a
            pad column is not a class.

    Returns:
        The mean loss, 0-d float32.
    """
    logits = model(batch.inputs)
    return cross_entropy(
        logits.flatten(0, 1).contiguous(),
        batch.targets.reshape(-1),
        classes=classes,
    )


def accumulate(
    model: nn.Module,
    stream: Iterator[Batch],
    *,
    accum: int,
    classes: int,
    device: torch.device | str,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
) -> float:
    """One optimizer step's worth of gradient, and nothing else.

    Clears the gradients once, pulls ``accum`` micro batches, and divides each micro loss by
    ``accum`` so the summed gradient is the gradient of the mean over the whole step. Takes
    no optimizer and applies no update: an accumulation that also stepped could not be
    checked against a single larger batch, which is the one property it has to have.

    Args:
        model: The model, in training mode.
        stream: Micro batches, on the CPU.
        accum: Micro batches in this step.
        classes: Tokens the labels index.
        device: Where to run.
        autocast_dtype: Compute dtype under autocast, or None for none.

    Returns:
        The mean loss over the micro batches, in nats.

    Raises:
        StopIteration: When the stream runs dry mid-step. A short step would be a step at
            fewer tokens than the schedule assumes.
    """
    model.zero_grad(set_to_none=True)
    device_type = torch.device(device).type
    total = 0.0
    for _ in range(accum):
        batch = next(stream).to(device)
        with torch.autocast(
            device_type=device_type,
            dtype=autocast_dtype or torch.float32,
            enabled=autocast_dtype is not None,
        ):
            loss = loss_on(model, batch, classes=classes)
        (loss / accum).backward()
        total += loss.detach().item()
    return total / accum


@torch.no_grad()
def evaluate(
    model: nn.Module,
    shard: Shard,
    *,
    seq_len: int,
    batch_size: int,
    classes: int,
    device: torch.device | str,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    bytes_per_token: float | None = None,
) -> Evaluation:
    """Score every window of a shard, in order.

    The whole shard, not a sample: the held-out number is the thing arms are ranked on, so
    it is a sum rather than an estimate. Batches are weighted by their token count, so a
    short last batch does not overweight its windows.

    Args:
        model: The model. Left in whatever mode it arrived in and restored on exit.
        shard: The validation shard.
        seq_len: Sequence length.
        batch_size: Sequences per forward.
        classes: Tokens the labels index.
        device: Where to run.
        autocast_dtype: Compute dtype under autocast, or None for none.
        bytes_per_token: Decoded UTF-8 bytes per token in this shard, from the manifest.
            None leaves ``bpb`` unset.

    Returns:
        The score.

    Raises:
        ValueError: When the shard holds no window of ``seq_len + 1``.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    device_type = torch.device(device).type
    try:
        for batch in val_batches(shard, seq_len=seq_len, batch_size=batch_size):
            on_device = batch.to(device)
            count = on_device.targets.numel()
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype or torch.float32,
                enabled=autocast_dtype is not None,
            ):
                loss = loss_on(model, on_device, classes=classes)
            total += float(loss) * count
            tokens += count
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError(f"{shard.path} holds no window of {seq_len + 1} tokens")
    loss = total / tokens
    bpb = None if bytes_per_token is None else loss * _LOG2E / bytes_per_token
    return Evaluation(loss, bpb, tokens)


def train(
    model: nn.Module,
    train_shard: Shard,
    config: TrainConfig,
    *,
    d_model: int,
    classes: int,
    device: torch.device | str,
    val_shard: Shard | None = None,
    bytes_per_token: float | None = None,
    on_step: Callable[[Step], None] | None = None,
) -> TrainResult:
    """Run the loop.

    Args:
        model: On ``device``, in float32, already seeded at construction.
        train_shard: Training tokens.
        config: The run.
        d_model: This arm's width, for the rate transfer.
        classes: Tokens the labels index.
        device: Where to run.
        val_shard: Held-out tokens, or None to skip validation.
        bytes_per_token: Decoded bytes per validation token, from the manifest.
        on_step: Called every ``config.log_every`` steps and on the last step.

    Returns:
        The result.

    Raises:
        ValueError: From the config, the schedule, or the group partition.
    """
    steps = config.steps
    accum = config.accum
    peak_lr = transfer(config.base_lr, d_model=d_model, token_batch=config.token_batch)
    embedding_lr = transfer(
        config.embedding_base_lr, d_model=d_model, token_batch=config.token_batch
    )
    policy = GroupPolicy(
        lr=peak_lr,
        embedding_lr=embedding_lr,
        ssm_multiplier=config.ssm_multiplier,
        weight_decay=config.weight_decay,
    )
    groups = parameter_groups(model, policy)
    optimizer = torch.optim.AdamW(
        groups, lr=peak_lr, betas=config.betas, eps=config.eps, fused=False
    )
    # The schedule multiplies every group by one factor, so the ratios the policy set --
    # the state-space group's 0.1x, the token table's own rate -- hold at every step.
    peaks = [float(group["lr"]) for group in optimizer.param_groups]

    stream = batches(
        train_shard,
        seq_len=config.seq_len,
        batch_size=config.micro_batch,
        seed=config.seed,
        steps=steps * accum,
    )
    model.train()
    recent: list[float] = []
    window = max(1, min(steps, 20))
    step_lr = peak_lr

    for step in range(steps):
        factor = (
            lr_at(
                step,
                total_steps=steps,
                peak_lr=peak_lr,
                warmdown_fraction=config.warmdown_fraction,
                final_fraction=config.final_fraction,
            )
            / peak_lr
        )
        for group, base in zip(optimizer.param_groups, peaks, strict=True):
            group["lr"] = base * factor
        step_lr = peak_lr * factor

        mean = accumulate(
            model,
            stream,
            accum=accum,
            classes=classes,
            device=device,
            autocast_dtype=config.autocast_dtype,
        )
        norm = (
            float(nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
            if config.grad_clip > 0.0
            else 0.0
        )
        optimizer.step()

        recent.append(mean)
        del recent[:-window]
        if on_step is not None and (
            step == steps - 1 or (config.log_every > 0 and step % config.log_every == 0)
        ):
            on_step(Step(step, mean, step_lr, norm))

    val = (
        None
        if val_shard is None
        else evaluate(
            model,
            val_shard,
            seq_len=config.seq_len,
            batch_size=config.eval_batch,
            classes=classes,
            device=device,
            autocast_dtype=config.autocast_dtype,
            bytes_per_token=bytes_per_token,
        )
    )
    return TrainResult(
        steps=steps,
        tokens=steps * config.token_batch,
        train_loss=sum(recent) / len(recent) if recent else math.nan,
        val=val,
        peak_lr=peak_lr,
        embedding_lr=embedding_lr,
        accum=accum,
    )
