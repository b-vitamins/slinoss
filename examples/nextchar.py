#!/usr/bin/env python3
"""Small next-character training example built on the reference SLinOSS mixer.

By default the script keeps all artifacts under ``/tmp/nextchar``:

- dataset: ``/tmp/nextchar/enwik8``
- download cache: ``/tmp/nextchar/enwik8.zip``
- logs: ``/tmp/nextchar/train.log``
- metrics: ``/tmp/nextchar/metrics.json``

The model intentionally stays small and explicit so it doubles as a usage
example for collaborators who need an end-to-end reference before the CuTe
backend lands.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from slinoss.layers import SLinOSSMixer
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from slinoss.layers import SLinOSSMixer


ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
DEFAULT_ROOT = Path("/tmp/nextchar")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_enwik8(*, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "enwik8"
    zip_path = root / "enwik8.zip"
    if raw_path.exists():
        return raw_path

    if not zip_path.exists():
        print(f"[data] downloading {ENWIK8_URL} -> {zip_path}", flush=True)
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)

    print(f"[data] extracting {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("enwik8") as f_in, raw_path.open("wb") as f_out:
            f_out.write(f_in.read())
    return raw_path


def load_enwik8_chars(
    data_path: Path | None = None,
    *,
    root: Path = DEFAULT_ROOT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    raw_path = data_path if data_path is not None else download_enwik8(root=root)
    raw = raw_path.read_bytes()
    raw_np = np.frombuffer(raw, dtype=np.uint8)
    unique_bytes = np.unique(raw_np)
    vocab_size = int(unique_bytes.size)

    lut = torch.full((256,), -1, dtype=torch.long)
    lut[torch.tensor(unique_bytes.tolist(), dtype=torch.long)] = torch.arange(
        vocab_size, dtype=torch.long
    )
    # Avoid the NumPy -> DLPack bridge here. The current Guix torch/NumPy stack
    # rejects readonly NumPy buffers in this path, while a writable bytearray
    # gives the same byte-level view without the failure.
    data_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)
    data = lut[data_bytes].contiguous()

    train = data[:90_000_000]
    val = data[90_000_000:95_000_000]
    test = data[95_000_000:100_000_000]
    return train, val, test, vocab_size


def sample_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts.tolist()], dim=0)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts.tolist()], dim=0)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: torch.device,
) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()

    n_blocks = min(eval_batches * batch_size, (data.numel() - 1) // block_size)
    _require(n_blocks > 0, "Not enough data to form an evaluation block.")

    data = data[: n_blocks * block_size + 1]
    x_all = data[:-1].view(n_blocks, block_size)
    y_all = data[1:].view(n_blocks, block_size)

    total_loss = 0.0
    total_tokens = 0
    for i in range(0, n_blocks, batch_size):
        xb = x_all[i : i + batch_size].to(device, non_blocking=True)
        yb = y_all[i : i + batch_size].to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        tokens = yb.numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens

    mean_nll = total_loss / total_tokens
    bpc = mean_nll / math.log(2.0)
    ppl = math.exp(mean_nll)
    if was_training:
        model.train()
    return mean_nll, bpc, ppl


def configure_optim(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias" not in name and "norm" not in name.lower():
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, *, mult: int = 4) -> None:
        super().__init__()
        hidden = mult * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class NextCharBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = SLinOSSMixer(
            d_model,
            d_state=d_state,
            expand=expand,
            d_head=d_head,
            d_conv=d_conv,
            chunk_size=chunk_size,
            normalize_bc=True,
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm1 = self.norm1(x)
        x = x + self.mixer(norm1)
        norm2 = self.norm2(x)
        x = x + self.ff(norm2)
        return x


class NextCharLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        d_state: int,
        expand: int,
        d_head: int,
        d_conv: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(1, self.block_size, d_model))
        self.blocks = nn.ModuleList(
            [
                NextCharBlock(
                    d_model,
                    d_state=d_state,
                    expand=expand,
                    d_head=d_head,
                    d_conv=d_conv,
                    chunk_size=chunk_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError(f"Expected idx shape (batch, T), got {tuple(idx.shape)}.")
        if idx.shape[1] > self.block_size:
            raise ValueError(
                f"Sequence length {idx.shape[1]} exceeds block_size {self.block_size}."
            )
        x = self.token_embed(idx)
        x = x + self.pos_embed[:, : idx.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        return self.lm_head(x)


@dataclass(frozen=True)
class EvalPoint:
    step: int
    train_loss: float
    val_nll: float
    val_bpc: float
    val_ppl: float
    elapsed_s: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reference SLinOSS next-character example on enwik8."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=1_000)
    p.add_argument("--eval-batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    _require(
        args.chunk_size <= args.block_size,
        f"chunk_size={args.chunk_size} must be <= block_size={args.block_size}.",
    )
    _require(
        (args.expand * args.d_model) % args.d_head == 0,
        "expand * d_model must be divisible by d_head.",
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.root.mkdir(parents=True, exist_ok=True)
    log_path = args.root / "train.log"
    metrics_path = args.root / "metrics.json"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    train, val, test, vocab_size = load_enwik8_chars(args.data, root=args.root)
    model = NextCharLM(
        vocab_size=vocab_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        d_head=args.d_head,
        d_conv=args.d_conv,
        chunk_size=args.chunk_size,
    ).to(device)
    optimizer = configure_optim(model, lr=args.lr, weight_decay=args.weight_decay)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    start = time.time()
    log(f"[setup] device={device} vocab={vocab_size} params={params_m:.3f}M")
    log(
        "[setup] "
        f"steps={args.steps} batch_size={args.batch_size} block_size={args.block_size} "
        f"d_model={args.d_model} n_layers={args.n_layers} d_state={args.d_state}"
    )

    init_nll, init_bpc, init_ppl = evaluate(
        model,
        val,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=device,
    )
    log(
        f"[eval] step=0 train_loss=nan val_nll={init_nll:.4f} "
        f"val_bpc={init_bpc:.4f} val_ppl={init_ppl:.4f}"
    )

    history: list[EvalPoint] = []
    last_loss = float("nan")
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = sample_batch(
            train,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        last_loss = float(loss.detach())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {last_loss}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            log(
                f"[train] step={step:6d} loss={last_loss:.4f} "
                f"elapsed_s={time.time() - start:.1f}"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            val_nll, val_bpc, val_ppl = evaluate(
                model,
                val,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_batches=args.eval_batches,
                device=device,
            )
            point = EvalPoint(
                step=step,
                train_loss=last_loss,
                val_nll=val_nll,
                val_bpc=val_bpc,
                val_ppl=val_ppl,
                elapsed_s=time.time() - start,
            )
            history.append(point)
            log(
                f"[eval] step={step:6d} train_loss={last_loss:.4f} "
                f"val_nll={val_nll:.4f} val_bpc={val_bpc:.4f} val_ppl={val_ppl:.4f}"
            )

    test_nll, test_bpc, test_ppl = evaluate(
        model,
        test,
        batch_size=args.batch_size,
        block_size=args.block_size,
        eval_batches=args.eval_batches,
        device=device,
    )
    log(
        f"[test] final_train_loss={last_loss:.4f} test_nll={test_nll:.4f} "
        f"test_bpc={test_bpc:.4f} test_ppl={test_ppl:.4f}"
    )

    metrics = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "device": str(device),
        "params_m": params_m,
        "initial_val": {"nll": init_nll, "bpc": init_bpc, "ppl": init_ppl},
        "history": [asdict(row) for row in history],
        "final_test": {"nll": test_nll, "bpc": test_bpc, "ppl": test_ppl},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
