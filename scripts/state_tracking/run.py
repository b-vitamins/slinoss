"""Run one state-tracking arm, or a list of them, and write one record per arm.

    python3 -m scripts.state_tracking.run --task parity --mixer slinoss
    python3 -m scripts.state_tracking.run --task A5 --mixer attention --seed 0 1 2
    python3 -m scripts.state_tracking.run --task cycle_nav --val-max-length 512

Records go to stdout as JSON lines and the summary table to stderr, so a redirect keeps the
data and leaves the table on the terminal. A record carries everything an arm is: the task
and its supervision, both splits' length ranges, the mixer and every resolved setting, the
protocol, the parameter count split into scaffold and mixer, and every evaluation with its
length bands. Nothing about the host goes in.

A baseline whose package is not a dependency of this tree registers itself in a module of
its own; ``--mixer-module`` imports it before the name is resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from scripts.state_tracking.instances import SplitConfig
from scripts.state_tracking.mixers import REGISTRY, load_module, resolve
from scripts.state_tracking.model import (
    ModelConfig,
    build_model,
    mixer_parameters,
    parameter_count,
)
from scripts.state_tracking.tasks import AUTOMATA, Task
from scripts.state_tracking.tasks import resolve as resolve_task
from scripts.state_tracking.train import (
    Point,
    Report,
    TrainConfig,
    seed_all,
    split_seeds,
    train,
)


def splits(args: argparse.Namespace, seed: int) -> tuple[SplitConfig, SplitConfig]:
    """The train and validation splits for one arm.

    Args:
        args: Parsed command line.
        seed: The arm's seed.

    Returns:
        ``(train_split, val_split)``. The train split is unbounded unless
        ``--train-count`` bounds it; the validation split is always bounded, since an
        accuracy needs a denominator.
    """
    train_seed, val_seed = split_seeds(seed)
    return (
        SplitConfig(
            min_length=args.train_min_length,
            max_length=args.train_max_length,
            seed=train_seed,
            count=args.train_count,
            pad_to=args.pad_to,
        ),
        SplitConfig(
            min_length=args.val_min_length,
            max_length=args.val_max_length,
            seed=val_seed,
            count=args.val_count,
            pad_to=args.pad_to,
        ),
    )


def run_arm(
    task: Task,
    mixer_name: str,
    overrides: list[str],
    model_args: dict[str, Any],
    config: TrainConfig,
    train_split: SplitConfig,
    val_split: SplitConfig,
    *,
    quiet: bool,
) -> dict[str, Any]:
    """Build a model, run the protocol, and return the record.

    Args:
        task: The task.
        mixer_name: Registry key.
        overrides: Mixer settings, as ``key=value`` strings.
        model_args: Scaffold settings beyond what the task fixes.
        config: The protocol. Its seed drives initialization only.
        train_split: The train split.
        val_split: The validation split.
        quiet: Suppress the per-evaluation lines on stderr.

    Returns:
        The record, JSON-ready.
    """
    mixer = resolve(mixer_name, overrides)
    model_config = ModelConfig(
        vocab_size=task.vocab_size,
        max_length=max(val_split.max_length, train_split.max_length),
        **model_args,
    )
    seed_all(config.seed)
    model = build_model(model_config, mixer.factory)

    def echo(point: Point) -> None:
        head = ""
        if point.val.bands:
            last = point.val.bands[-1]
            head = f" tail[{last.low}-{last.high}] {last.accuracy:.4f}"
        print(
            f"  {task.name} step {point.step} lr {point.lr:.2e} "
            f"train {point.train_loss:.4f} val {point.val.loss:.4f} "
            f"acc {point.val.accuracy:.4f}{head}",
            file=sys.stderr,
            flush=True,
        )

    report = train(
        model,
        task,
        train_split,
        val_split,
        config,
        on_point=None if quiet else echo,
    )
    return _record(
        task,
        mixer.name,
        mixer.settings,
        model_config,
        config,
        train_split,
        val_split,
        report,
        parameter_count(model),
        mixer_parameters(model),
    )


def _record(
    task: Task,
    mixer_name: str,
    settings: dict[str, Any],
    model_config: ModelConfig,
    config: TrainConfig,
    train_split: SplitConfig,
    val_split: SplitConfig,
    report: Report,
    parameters: int,
    mixer_params: int,
) -> dict[str, Any]:
    """Assemble one arm's record.

    Args:
        task: The task.
        mixer_name: Registry key.
        settings: Resolved mixer settings.
        model_config: Scaffold shape.
        config: The protocol.
        train_split: The train split.
        val_split: The validation split.
        report: What the run produced.
        parameters: Trainable parameters.
        mixer_params: Trainable parameters inside the mixers.

    Returns:
        A JSON-ready dict.
    """
    return {
        "task": task.name,
        "supervision": task.supervision,
        "vocab_size": task.vocab_size,
        "group_order": None if task.group is None else task.group.order,
        "mixer": mixer_name,
        "mixer_settings": settings,
        "model": asdict(model_config),
        "protocol": asdict(config),
        "train_split": asdict(train_split),
        "val_split": asdict(val_split),
        "parameters": parameters,
        "mixer_parameters": mixer_params,
        "best": _metrics(report.best),
        "best_step": report.best_step,
        "final": _metrics(report.final),
        "steps_run": report.steps_run,
        "solved": report.solved,
        "points": [
            [p.step, p.lr, p.train_loss, p.val.loss, p.val.accuracy]
            for p in report.points
        ],
    }


def _metrics(metrics: Any) -> dict[str, Any]:
    """One evaluation as a dict, bands included."""
    out = metrics._asdict()
    out["bands"] = [band._asdict() for band in metrics.bands]
    return out


def table(records: list[dict[str, Any]]) -> str:
    """One row per record, plus the mean accuracy.

    Args:
        records: What :func:`run_arm` returned, in order.

    Returns:
        The table. ``tail`` is the longest band's accuracy, which is the number the axis
        is about: a mean over lengths 40 to 256 can sit high while the tail is at chance.
    """
    head = (
        f"{'task':<20} {'mixer':<10} {'seed':>5} {'acc':>8} {'tail':>8} "
        f"{'loss':>8} {'steps':>7} {'solved':>7} {'params':>9}"
    )
    lines = [head, "-" * len(head)]
    for record in records:
        best = record["best"]
        bands = best["bands"]
        tail = bands[-1]["accuracy"] if bands else 0.0
        lines.append(
            f"{record['task']:<20} {record['mixer']:<10} "
            f"{record['protocol']['seed']:>5} {best['accuracy']:>8.4f} "
            f"{tail:>8.4f} {best['loss']:>8.4f} {record['steps_run']:>7} "
            f"{record['solved']!s:>7} {record['parameters']:>9}"
        )
    if len(records) > 1:
        mean = sum(r["best"]["accuracy"] for r in records) / len(records)
        lines.append("-" * len(head))
        lines.append(f"{'mean':<20} {'':<10} {'':>5} {mean:>8.4f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser. Every protocol field is a flag so an arm is reproducible from its
        record, and the defaults are the protocol's.
    """
    protocol = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="scripts.state_tracking.run", description="Run state-tracking arms."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=sorted(AUTOMATA),
        metavar="NAME",
        help=f"tasks to run, from {sorted(AUTOMATA)} or a group spec such as A5",
    )
    parser.add_argument(
        "--mixer", default="attention", help=f"one of {sorted(REGISTRY)}"
    )
    parser.add_argument(
        "--set",
        dest="settings",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="mixer settings",
    )
    parser.add_argument(
        "--mixer-module",
        nargs="*",
        default=[],
        metavar="MODULE",
        help="import before resolving, for a mixer registered outside this tree",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[protocol.seed],
        help="one arm per seed; seeds the initialization and both split streams",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument(
        "--use-glu", action="store_true", help="gated-linear branch in every block"
    )
    parser.add_argument("--train-min-length", type=int, default=3)
    parser.add_argument("--train-max-length", type=int, default=40)
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="bound the train split; unbounded by default",
    )
    parser.add_argument("--val-min-length", type=int, default=40)
    parser.add_argument("--val-max-length", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=8192)
    parser.add_argument(
        "--pad-to",
        type=int,
        default=0,
        help="floor on the batch width; zero pads each batch to its longest item",
    )
    parser.add_argument("--num-steps", type=int, default=protocol.num_steps)
    parser.add_argument("--batch-size", type=int, default=protocol.batch_size)
    parser.add_argument("--lr", type=float, default=protocol.lr)
    parser.add_argument("--final-lr", type=float, default=protocol.final_lr)
    parser.add_argument(
        "--warmup-fraction", type=float, default=protocol.warmup_fraction
    )
    parser.add_argument(
        "--wd-embedding", type=float, default=protocol.weight_decay_embedding
    )
    parser.add_argument("--wd-others", type=float, default=protocol.weight_decay_others)
    parser.add_argument(
        "--early-stop", type=float, default=protocol.early_stop_threshold
    )
    parser.add_argument("--print-steps", type=int, default=protocol.print_steps)
    parser.add_argument(
        "--accumulation-steps", type=int, default=protocol.accumulation_steps
    )
    parser.add_argument("--grad-clip", type=float, default=protocol.grad_clip)
    parser.add_argument(
        "--precision", default=protocol.precision, choices=("fp32", "bf16")
    )
    parser.add_argument("--band-width", type=int, default=protocol.band_width)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where the model and the batches live",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="append records here as well as stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no per-evaluation lines on stderr"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run every requested task at every requested seed.

    Args:
        argv: Command line, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status. Zero unless an arm raised.
    """
    args = build_parser().parse_args(argv)
    for module in args.mixer_module:
        load_module(module)

    model_args = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "use_glu": args.use_glu,
    }
    base = TrainConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        final_lr=args.final_lr,
        warmup_fraction=args.warmup_fraction,
        weight_decay_embedding=args.wd_embedding,
        weight_decay_others=args.wd_others,
        early_stop_threshold=args.early_stop,
        print_steps=args.print_steps,
        accumulation_steps=args.accumulation_steps,
        grad_clip=args.grad_clip,
        precision=args.precision,
        band_width=args.band_width,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    handle = args.out.open("a") if args.out is not None else None
    try:
        for seed in args.seed:
            train_split, val_split = splits(args, seed)
            for name in args.task:
                record = run_arm(
                    resolve_task(name),
                    args.mixer,
                    args.settings,
                    model_args,
                    replace(base, seed=seed),
                    train_split,
                    val_split,
                    quiet=args.quiet,
                )
                line = json.dumps(record)
                print(line, flush=True)
                if handle is not None:
                    handle.write(line + "\n")
                    handle.flush()
                records.append(record)
    finally:
        if handle is not None:
            handle.close()

    print(table(records), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
