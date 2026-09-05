"""CUDA integration at the two language-model scale shapes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from slinoss import _C

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
if not _C.is_available():
    pytest.skip(
        f"{_C.EXTENSION} is not built; run {_C.BUILD_COMMAND}",
        allow_module_level=True,
    )

from scripts.lm.corpus import DTYPE
from scripts.lm.data import Shard
from scripts.lm.mixers import REGISTRY
from scripts.lm.model import LMConfig, build_model, layer_factories
from scripts.lm.train import Step, TrainConfig, train
from slinoss.ops.conv import backends as conv_backends
from slinoss.ops.so3ssd import backends as scan_backends
from slinoss.ops.xent import backends as xent_backends

pytestmark = [pytest.mark.cuda, pytest.mark.cute, pytest.mark.slow]

SEQ_LEN = 2048
N_LAYERS = 12
VOCAB = 50277
STEPS = 3
BYTES_PER_TOKEN = 4.2

SCALES = [
    pytest.param(512, 8, 10, id="45M"),
    pytest.param(1344, 2, 14, id="180M"),
]
"""``(d_model, micro_batch, free GiB needed)`` per scale.

The widths are the grid points nearest the protocol's 496 and 1360; the grid is what every
registered arm builds at, and matching a count is the contract rather than reproducing a width.
The micro batch is a memory decision, so the larger scale takes a smaller one and the same
optimizer step.
"""


def _shard(path: Path, windows: int) -> Shard:
    """A shard of ``windows`` full windows, tokens cycling through the vocabulary."""
    tokens = windows * SEQ_LEN + 1
    (np.arange(tokens, dtype=np.int64) % VOCAB).astype(DTYPE).tofile(path)
    return Shard(path, tokens)


def test_dispatch_matches_the_declared_precision_contract() -> None:
    """Pin intentional fp32 dispatch and availability of every fast backend."""
    assert (
        scan_backends.resolve(None, "cuda", torch.float32).name
        == scan_backends.REFERENCE
    )
    assert (
        scan_backends.resolve(None, "cuda", torch.bfloat16).name == scan_backends.CUTE
    )
    assert (
        conv_backends.resolve(None, "cuda", torch.float32).name == conv_backends.NATIVE
    )
    assert xent_backends.resolve(None, "cuda", torch.float32).name == xent_backends.CUTE


@pytest.mark.parametrize(("d_model", "micro_batch", "needed_gib"), SCALES)
def test_three_steps_at_a_scale_s_shape(
    tmp_path: Path, d_model: int, micro_batch: int, needed_gib: int
) -> None:
    """Twelve layers, 2048 tokens, the real vocabulary, and a held-out score at the end.

    The first step's loss is the assertion that carries weight: an untrained model over random
    tokens predicts uniformly, so the loss is the vocabulary's entropy. A padded head scored
    against the padded width instead of the class count, or labels off by a position, both land
    somewhere else while still looking like a number a trainer would print.
    """
    free, _ = torch.cuda.mem_get_info()
    if free < needed_gib << 30:
        pytest.skip(f"{free / 2**30:.1f} GiB free, needs {needed_gib}")

    train_shard = _shard(tmp_path / "train.bin", STEPS * micro_batch)
    val_shard = _shard(tmp_path / "val.bin", 2)
    config = TrainConfig(
        token_budget=STEPS * micro_batch * SEQ_LEN,
        token_batch=micro_batch * SEQ_LEN,
        seq_len=SEQ_LEN,
        micro_batch=micro_batch,
        eval_batch=micro_batch,
        log_every=1,
    )
    assert config.steps == STEPS
    assert config.accum == 1

    torch.manual_seed(0)
    resolved = REGISTRY.resolve("slinoss")
    stack = build_model(
        LMConfig(d_model=d_model, n_layers=N_LAYERS, vocab_size=VOCAB),
        layer_factories(resolved.factory, N_LAYERS),
        max_length=SEQ_LEN,
        device="cuda",
        dtype=torch.float32,
    )

    seen: list[Step] = []
    try:
        result = train(
            stack,
            train_shard,
            config,
            d_model=d_model,
            classes=VOCAB,
            device="cuda",
            val_shard=val_shard,
            bytes_per_token=BYTES_PER_TOKEN,
            on_step=seen.append,
        )
    finally:
        del stack
        torch.cuda.empty_cache()

    assert [step.number for step in seen] == list(range(STEPS))
    assert all(math.isfinite(step.loss) for step in seen)
    assert all(math.isfinite(step.grad_norm) and step.grad_norm > 0.0 for step in seen)
    assert seen[0].loss == pytest.approx(math.log(VOCAB), abs=1.0)
    assert result.val is not None
    assert math.isfinite(result.val.loss)
    assert result.val.bpb == pytest.approx(
        result.val.loss / math.log(2.0) / BYTES_PER_TOKEN
    )
    assert result.val.tokens == 2 * SEQ_LEN
    assert result.tokens == STEPS * config.token_batch
