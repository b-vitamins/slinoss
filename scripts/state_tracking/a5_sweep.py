"""Fixed-length A5 depth sweep used by the Merrill/Walker Figure-1 protocol.

The ordinary state-tracking runner measures length generalization.  This program instead
trains a fresh model at each fixed length and reports the minimum depth whose validation
accuracy reaches 90 percent.  It uses the official Merrill CSVs, Walker's 100k-step
training contract, and KLA's five-seed ``any seed solves`` decision rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.request
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scripts.provenance import capture as capture_provenance
from scripts.state_tracking.mixers import load_module, resolve
from scripts.state_tracking.model import (
    ModelConfig,
    build_model,
    mixer_parameters,
    parameter_count,
)
from scripts.state_tracking.train import seed_all

MERRILL_REPOSITORY = "https://github.com/jopetty/word-problem"
MERRILL_REVISION = "8f910f92e1c70455dcd9376f56032dfc55126188"
WALKER_REPOSITORY = "https://github.com/Benjamin-Walker/structured-linear-cdes"
WALKER_REVISION = "243cb30fcd85406a94f2810ec762c59e6e2bb1c7"
KLA_REPOSITORY = "https://github.com/vaisakh-shaj/kalman-linear-attention"
KLA_REVISION = "173f8382044512705c742dfdc9c9888652898339"

LENGTHS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20)
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD = 0.90
NUM_STEPS = 100_000
BATCH_SIZE = 256
AUX_BATCH_SIZE = BATCH_SIZE // 10
PEAK_LR = 1e-3
FINAL_LR = 1e-5
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 1e-2
DROPOUT = 0.1
EVAL_EVERY = 10_000
TRAIN_FRACTION = 0.8


@dataclass(frozen=True)
class SourceFile:
    """Pinned Git-LFS payload metadata for one official CSV."""

    sha256: str
    size: int


SOURCE_FILES = {
    2: SourceFile(
        "5fa1b247ecf078d027a71fff7c486d6e4f12df4d0f8ff4526bc91cf759bbf307", 66_018
    ),
    3: SourceFile(
        "b6deb8732e380bf6f78e7d8b312c49ef9415afb00b225e5383273585c3662433", 5_184_018
    ),
    4: SourceFile(
        "caf2a3e7381c26b6c85bba0e401ba36f32ffc83c01d7ddd900f2ba9d6c9cea53", 29_667_856
    ),
    5: SourceFile(
        "67edb76df072854c767b5fd7d697022b75c1c4a07b5b4beb4507c5bd9112a007", 35_333_790
    ),
    6: SourceFile(
        "1e441715a6138fb8c1669a37a1463d680bb5a18e8ccc19fb41c411660dfa74e6", 40_999_379
    ),
    7: SourceFile(
        "0c0a1ec07462fbf86acbe3654df4f3df7c044b453ef884331e21d6679d407fe1", 46_665_582
    ),
    8: SourceFile(
        "751f10bdd0b56a0331211ade376c779641ccaf9171ddefcb026a5552ef26798b", 52_332_838
    ),
    9: SourceFile(
        "0ef43e8f7871c52d1252254b6460cc8a063c2344909805e90b63651881f5d2d1", 57_998_967
    ),
    10: SourceFile(
        "2420bf67ec445620ebc606df34b8dd8e111d6dbafa37a5febd1d45d0817498d5", 63_667_048
    ),
    11: SourceFile(
        "04cc19634e1bebc2f89e1530063238f151d8023e3928f439f0aa3b6317f3014a", 69_332_824
    ),
    12: SourceFile(
        "d7f373aa1c7a35a8eb478dfa79d9fbb60f79ca84dfc3ca8c78ed30e77be6af7d", 74_999_612
    ),
    13: SourceFile(
        "faa2bdfb16c195ff530abe1f3700718ffa107c88629c91c5389c7d3790c24d18", 80_667_900
    ),
    14: SourceFile(
        "82bbc02229fcf1c25d93bc2f62b68236961f0e94bd78f10c078f17abf382e7f1", 86_334_749
    ),
    15: SourceFile(
        "8dea4a9ccd41910fe2390d8df579d486c4d89e67321fa8d45946131019be2978", 92_001_387
    ),
    16: SourceFile(
        "8729a39418471ae30ecfe37caf2f0aa510297bbd3beaa03f8531816658d5c4fb", 97_668_616
    ),
    17: SourceFile(
        "8ce255fb7c48262b28280ba12b1d1b77c2d20d8fd88bf6716e8a610b22018b66", 103_331_498
    ),
    18: SourceFile(
        "846615ad6dfc5c0903ede05d16b3b5ef6cba9c345c68a2b98e345680d8abba7f", 109_000_662
    ),
    19: SourceFile(
        "cb3f30051bd6dc791f5ad5375dbd92184b14a57904a2dcc0a7aba1c7c9f088c3", 114_666_139
    ),
    20: SourceFile(
        "61a06b9ad5b00b647f8fb90d69870616b0cc5afc66cafa304c1339e04d458926", 120_334_884
    ),
}


class FixedData(NamedTuple):
    """One official fixed-length table in compact CPU storage."""

    inputs: Tensor
    targets: Tensor
    sha256: str


class Split(NamedTuple):
    """Pinned train/validation row indices."""

    train: Tensor
    validation: Tensor


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, source: SourceFile) -> None:
    """Reject a missing, truncated, or substituted source payload."""
    if not path.is_file():
        raise FileNotFoundError(f"missing official A5 table: {path}")
    size = path.stat().st_size
    if size != source.size:
        raise ValueError(f"{path}: expected {source.size} bytes, found {size}")
    found = _digest(path)
    if found != source.sha256:
        raise ValueError(f"{path}: expected sha256 {source.sha256}, found {found}")


def source_url(length: int) -> str:
    """Pinned Git-LFS media URL for one source table."""
    if length not in SOURCE_FILES:
        raise ValueError(f"no pinned A5 source for length {length}")
    return (
        "https://media.githubusercontent.com/media/jopetty/word-problem/"
        f"{MERRILL_REVISION}/data/A5%3D{length}.csv"
    )


def prepare(root: Path, lengths: Iterable[int]) -> None:
    """Download and verify official tables, replacing no valid payload."""
    root.mkdir(parents=True, exist_ok=True)
    for length in dict.fromkeys((2, *lengths)):
        source = SOURCE_FILES[length]
        path = root / f"A5={length}.csv"
        try:
            verify_file(path, source)
        except (FileNotFoundError, ValueError):
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=root)
            os.close(fd)
            tmp = Path(temporary)
            try:
                with (
                    urllib.request.urlopen(source_url(length)) as response,
                    tmp.open("wb") as output,
                ):
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                verify_file(tmp, source)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        print(f"{path} {source.size} {source.sha256}", flush=True)


def load_csv(path: Path, length: int, source: SourceFile | None = None) -> FixedData:
    """Parse one source table and enforce its schema and value domain."""
    if source is not None:
        verify_file(path, source)
    raw_inputs = array("B")
    raw_targets = array("B")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows, None)
        if header != ["seed", "input", "target"]:
            raise ValueError(f"{path}: expected header seed,input,target; got {header}")
        for line, row in enumerate(rows, 2):
            if len(row) != 3:
                raise ValueError(f"{path}:{line}: expected three columns")
            try:
                int(row[0])
                inputs = tuple(map(int, row[1].split()))
                targets = tuple(map(int, row[2].split()))
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: non-integer field") from exc
            if len(inputs) != length or len(targets) != length:
                raise ValueError(
                    f"{path}:{line}: expected {length} input and target tokens"
                )
            if any(not 0 <= value < 60 for value in (*inputs, *targets)):
                raise ValueError(f"{path}:{line}: A5 token outside [0, 60)")
            raw_inputs.extend(inputs)
            raw_targets.extend(targets)
    if not raw_inputs:
        raise ValueError(f"{path}: empty table")
    inputs = torch.frombuffer(raw_inputs, dtype=torch.uint8).clone().view(-1, length)
    targets = torch.frombuffer(raw_targets, dtype=torch.uint8).clone().view(-1, length)
    return FixedData(inputs, targets, _digest(path))


def split_rows(rows: int, seed: int) -> Split:
    """The source runner's seeded 80/20 row split."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(rows, generator=generator)
    cut = int(TRAIN_FRACTION * rows)
    if cut == 0 or cut == rows:
        raise ValueError(f"cannot split {rows} rows 80/20")
    return Split(order[:cut], order[cut:])


class BatchStream:
    """Endless, independently seeded shuffle-by-epoch stream over row indices."""

    def __init__(self, rows: Tensor, batch_size: int, seed: int) -> None:
        self.rows = rows
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.offset = 0

    def __iter__(self) -> BatchStream:
        return self

    def __next__(self) -> Tensor:
        if self.offset >= self.order.numel():
            permutation = torch.randperm(self.rows.numel(), generator=self.generator)
            self.order = self.rows[permutation]
            self.offset = 0
        end = min(self.offset + self.batch_size, self.order.numel())
        batch = self.order[self.offset : end]
        self.offset = end
        return batch


def _batch(data: FixedData, rows: Tensor, device: str) -> tuple[Tensor, Tensor]:
    return (
        data.inputs[rows].to(device=device, dtype=torch.long, non_blocking=True),
        data.targets[rows].to(device=device, dtype=torch.long, non_blocking=True),
    )


def _lr(step: int) -> float:
    warmup = int(NUM_STEPS * WARMUP_FRACTION)
    if step < warmup:
        return PEAK_LR * (step + 1) / warmup
    phase = (step - warmup) / max(NUM_STEPS - warmup - 1, 1)
    return FINAL_LR + 0.5 * (PEAK_LR - FINAL_LR) * (1.0 + math.cos(math.pi * phase))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: FixedData,
    rows: Tensor,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Token accuracy on every held-out row."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for start in range(0, rows.numel(), batch_size):
        inputs, targets = _batch(data, rows[start : start + batch_size], device)
        prediction = model(inputs).argmax(dim=-1)
        correct += int((prediction == targets).sum())
        total += targets.numel()
    model.train(was_training)
    return correct / total


def run_arm(
    *,
    data: FixedData,
    auxiliary: FixedData,
    length: int,
    depth: int,
    seed: int,
    mixer_name: str,
    overrides: Sequence[str],
    d_model: int,
    device: str,
    provenance: dict[str, Any],
    quiet: bool,
) -> dict[str, Any]:
    """Run one ``(length, depth, seed)`` arm under the locked protocol."""
    seed_all(seed)
    mixer = resolve(mixer_name, overrides)
    model_config = ModelConfig(
        input_vocab_size=60,
        output_vocab_size=60,
        max_length=length,
        d_model=d_model,
        n_layers=depth,
        dropout=DROPOUT,
        use_glu=True,
    )
    model = build_model(model_config, mixer.factory).to(device)
    split = split_rows(data.inputs.shape[0], seed + 1)
    main = BatchStream(split.train, BATCH_SIZE, seed + 3)
    auxiliary_rows = torch.arange(auxiliary.inputs.shape[0])
    aux = BatchStream(auxiliary_rows, AUX_BATCH_SIZE, seed + 4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY
    )

    points: list[dict[str, float | int]] = []
    best = 0.0
    best_step = -1
    train_window = 0.0
    started = time.perf_counter()
    model.train()
    step = -1
    for step in range(NUM_STEPS):
        rate = _lr(step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = _batch(data, next(main), device)
        aux_inputs, aux_targets = _batch(auxiliary, next(aux), device)
        loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        aux_loss = F.cross_entropy(
            model(aux_inputs).flatten(0, 1), aux_targets.flatten()
        )
        total_loss = loss + aux_loss
        total_loss.backward()
        optimizer.step()
        train_window += float(total_loss.detach())

        if step % EVAL_EVERY != 0 and step != NUM_STEPS - 1:
            continue
        accuracy = evaluate(model, data, split.validation, device)
        point = {
            "step": step,
            "lr": rate,
            "train_loss": train_window / (1 if not points else EVAL_EVERY),
            "validation_accuracy": accuracy,
        }
        points.append(point)
        train_window = 0.0
        if accuracy > best:
            best, best_step = accuracy, step
        if not quiet:
            print(
                f"A5 L{length} D{depth} seed {seed} step {step} "
                f"acc {accuracy:.4f} loss {point['train_loss']:.4f}",
                file=sys.stderr,
                flush=True,
            )
        if accuracy >= THRESHOLD:
            break

    elapsed = time.perf_counter() - started
    return {
        "task": "A5",
        "length": length,
        "depth": depth,
        "seed": seed,
        "solved": best >= THRESHOLD,
        "threshold": THRESHOLD,
        "best_accuracy": best,
        "best_step": best_step,
        "steps_run": step + 1,
        "wall_seconds": elapsed,
        "points": points,
        "mixer": mixer.name,
        "mixer_settings": mixer.settings,
        "mixer_constructions": mixer.constructions,
        "model": asdict(model_config),
        "parameters": parameter_count(model),
        "mixer_parameters": mixer_parameters(model),
        "protocol": protocol_record(),
        "data": {
            "repository": MERRILL_REPOSITORY,
            "revision": MERRILL_REVISION,
            "main": {
                "file": f"A5={length}.csv",
                "sha256": data.sha256,
                "rows": data.inputs.shape[0],
                "split_seed": seed + 1,
                "train_rows": split.train.numel(),
                "validation_rows": split.validation.numel(),
            },
            "auxiliary": {
                "file": "A5=2.csv",
                "sha256": auxiliary.sha256,
                "rows": auxiliary.inputs.shape[0],
                "policy": "all rows; independent shuffled batches",
            },
        },
        "provenance": provenance,
    }


def protocol_record() -> dict[str, Any]:
    """The immutable experimental contract carried by every record."""
    return {
        "name": "walker-a5-fixed-v1",
        "lengths": list(LENGTHS),
        "num_steps": NUM_STEPS,
        "batch_size": BATCH_SIZE,
        "length_2_batch_size": AUX_BATCH_SIZE,
        "peak_lr": PEAK_LR,
        "final_lr": FINAL_LR,
        "warmup_fraction": WARMUP_FRACTION,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "eval_every": EVAL_EVERY,
        "train_fraction": TRAIN_FRACTION,
        "threshold": THRESHOLD,
        "seed_rule": "five seeds; a depth solves a length if any seed reaches threshold",
        "source_fidelity": {
            "data": "source-exact Merrill Git-LFS payloads and seeded 80/20 split",
            "training": "Walker paper contract; auxiliary size follows released JAX runner batch_size//10",
            "kla": (
                "decision rule and paper hyperparameters inspected; released KLA repo "
                "contains no A5 runner"
            ),
            "walker_repository": WALKER_REPOSITORY,
            "walker_revision": WALKER_REVISION,
            "kla_repository": KLA_REPOSITORY,
            "kla_revision": KLA_REVISION,
        },
    }


def minimum_depths(records: Sequence[dict[str, Any]]) -> dict[int, int | None]:
    """Minimum solved depth per requested length under the any-seed rule."""
    lengths = sorted({int(record["length"]) for record in records})
    answer: dict[int, int | None] = {}
    for length in lengths:
        solved = [
            int(record["depth"])
            for record in records
            if record["length"] == length and record["solved"]
        ]
        answer[length] = min(solved) if solved else None
    return answer


def _run(args: argparse.Namespace, argv: list[str] | None) -> int:
    provenance = capture_provenance(
        "scripts/state_tracking", argv, module="scripts.state_tracking.a5_sweep"
    )
    for module in args.mixer_module:
        load_module(module)
    auxiliary = load_csv(args.data_root / "A5=2.csv", 2, SOURCE_FILES[2])
    records: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as output:
        for length in args.length:
            data = load_csv(
                args.data_root / f"A5={length}.csv", length, SOURCE_FILES[length]
            )
            for depth in sorted(args.depth):
                depth_solved = False
                for seed in args.seed:
                    record = run_arm(
                        data=data,
                        auxiliary=auxiliary,
                        length=length,
                        depth=depth,
                        seed=seed,
                        mixer_name=args.mixer,
                        overrides=args.settings,
                        d_model=args.d_model,
                        device=args.device,
                        provenance=provenance,
                        quiet=args.quiet,
                    )
                    line = json.dumps(record)
                    print(line, flush=True)
                    output.write(line + "\n")
                    output.flush()
                    records.append(record)
                    depth_solved |= bool(record["solved"])
                    if depth_solved and args.short_circuit_seeds:
                        break
                if depth_solved:
                    break
    print(json.dumps({"minimum_depth": minimum_depths(records)}), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts.state_tracking.a5_sweep")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="download and verify official Merrill CSVs")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--length", type=int, nargs="+", default=list(LENGTHS))

    run = sub.add_parser("run", help="run fixed-length depth arms")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--length", type=int, nargs="+", default=list(LENGTHS))
    run.add_argument("--depth", type=int, nargs="+", default=[1, 2, 3, 4])
    run.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    run.add_argument("--mixer", default="slinoss")
    run.add_argument("--mixer-module", action="append", default=[])
    run.add_argument("--set", action="append", default=[], dest="settings")
    run.add_argument("--d-model", type=int, default=128)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--short-circuit-seeds", action="store_true")
    run.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bad_lengths = sorted(set(args.length) - set(LENGTHS))
    if bad_lengths:
        parser.error(f"lengths outside the Figure-1 grid: {bad_lengths}")
    if args.command == "prepare":
        prepare(args.data_root, args.length)
        return 0
    if any(depth < 1 for depth in args.depth):
        parser.error("depths must be positive")
    return _run(args, argv)


if __name__ == "__main__":
    raise SystemExit(main())
