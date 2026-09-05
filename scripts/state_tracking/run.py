"""Run one state-tracking arm, or a list of them, and write one record per arm.

    python3 -m scripts.state_tracking.run --task parity --mixer slinoss
    python3 -m scripts.state_tracking.run --profile walker-group-prefix --task A5
    python3 -m scripts.state_tracking.run --profile pdssm-groups-reconstruction

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

from scripts.provenance import capture as capture_provenance
from scripts.provenance import identity
from scripts.state_tracking.instances import SplitConfig
from scripts.state_tracking.mixers import REGISTRY, load_module, resolve
from scripts.state_tracking.model import (
    ModelConfig,
    build_model,
    mixer_parameters,
    parameter_count,
)
from scripts.state_tracking.tasks import (
    AUTOMATA,
    PROFILE_DEFAULTS,
    Task,
    resolve_profile,
)
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
    provenance: dict[str, Any] | None = None,
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
        provenance: Source/harness/command identity captured before the arm starts.
            Direct Python callers may omit it; the CLI captures one context and
            passes it to every arm.

    Returns:
        The record, JSON-ready.
    """
    mixer = resolve(mixer_name, overrides)
    if task.output_vocab_size is None:  # narrowed by Task.__post_init__
        raise AssertionError("resolved task output vocabulary is missing")
    model_config = ModelConfig(
        input_vocab_size=task.input_vocab_size,
        output_vocab_size=task.output_vocab_size,
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
        mixer.max_length_policy,
        mixer.constructions,
        capture_provenance(
            "scripts/state_tracking",
            ["<python-api>"],
            module="scripts.state_tracking.run",
        )
        if provenance is None
        else provenance,
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
    max_length_policy: str,
    constructions: list[dict[str, Any]],
    provenance: dict[str, Any],
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
        max_length_policy: Declared consumption of the scaffold length context.
        constructions: Effective per-layer mixer configurations.
        provenance: Source, harness, dirty tree, and command identity.

    Returns:
        A JSON-ready dict.
    """
    task_contract = asdict(task.contract)
    task_data = {
        "task": task.name,
        "supervision": task.supervision,
        "input_vocab_size": task.input_vocab_size,
        "output_vocab_size": task.output_vocab_size,
        "group_order": None if task.group is None else task.group.order,
        "benchmark_contract": task_contract,
    }
    train_data = {**task_data, "split": asdict(train_split)}
    val_data = {**task_data, "split": asdict(val_split)}
    init_lattices = [
        {
            key: construction["effective_config"].get(key)
            for key in (
                "context_length",
                "init_period_context_scale",
                "init_decay_context_scale",
                "resolved_init_period_span",
                "resolved_init_decay_span",
            )
        }
        for construction in constructions
        if "resolved_init_period_span" in construction["effective_config"]
    ]
    train_width_max = max(train_split.max_length, train_split.pad_to)
    val_width_max = max(val_split.max_length, val_split.pad_to)
    return {
        "task": task.name,
        "supervision": task.supervision,
        "vocab_size": (
            task.input_vocab_size
            if task.input_vocab_size == task.output_vocab_size
            else None
        ),
        "input_vocab_size": task.input_vocab_size,
        "output_vocab_size": task.output_vocab_size,
        "group_order": None if task.group is None else task.group.order,
        "benchmark_contract": task_contract,
        "mixer": mixer_name,
        "mixer_settings": settings,
        "mixer_contract": {
            "max_length_policy": max_length_policy,
            "initialization": "mixer_constructor; no scaffold reinitialization",
        },
        "mixer_constructions": constructions,
        "model": asdict(model_config),
        "protocol": asdict(config),
        "train_split": asdict(train_split),
        "val_split": asdict(val_split),
        "lengths": {
            "configured_task_length": None,
            "training_ceiling": train_split.max_length,
            "evaluation_ceiling": val_split.max_length,
            "observed_tensor_width": {
                "train": {"min": train_split.min_length, "max": train_width_max},
                "evaluation": {"min": val_split.min_length, "max": val_width_max},
            },
            "mixer_initialization_lattice": init_lattices or None,
        },
        "seeds": {
            "model": config.seed,
            "train_data": train_split.seed,
            "evaluation_data": val_split.seed,
        },
        "data": {
            "train": {**train_data, "identity": identity(train_data)},
            "evaluation": {**val_data, "identity": identity(val_data)},
        },
        "initialization": {
            "scaffold": "framework constructor defaults; no post-construction pass",
            "mixer": "owned by each mixer constructor",
        },
        "provenance": provenance,
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
        "--profile",
        default="pdssm-regular",
        choices=tuple(PROFILE_DEFAULTS),
        help="benchmark contract; tasks from another contract are rejected",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            f"tasks within --profile; automata are {sorted(AUTOMATA)}, "
            "Walker groups use A5/S5, PD reconstructions use pdssm:A5:2"
        ),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tasks = resolve_profile(args.profile, args.task)
    except ValueError as exc:
        parser.error(str(exc))
    provenance = capture_provenance(
        "scripts/state_tracking", argv, module="scripts.state_tracking.run"
    )
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
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("a") if args.out is not None else None
    try:
        for seed in args.seed:
            train_split, val_split = splits(args, seed)
            for task in tasks:
                record = run_arm(
                    task,
                    args.mixer,
                    args.settings,
                    model_args,
                    replace(base, seed=seed),
                    train_split,
                    val_split,
                    quiet=args.quiet,
                    provenance=provenance,
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
