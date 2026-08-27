"""Run one MAD arm, or the six-task suite, and write one record per task.

    python3 -m scripts.mad.run --mixer slinoss --set d_state=192
    python3 -m scripts.mad.run --mixer attention --task icr ficr --seed 0 1 2
    python3 -m scripts.mad.run --mixer conv --task sc --axis num_tokens_to_copy=64

Records go to stdout as JSON lines and the summary table to stderr, so a redirect keeps
the data and leaves the table on the terminal. A record carries everything an arm is: the
task and any axis moved off its baseline, the mixer and every resolved setting, the
protocol, the pool's leakage and width, the parameter count, and each evaluation. Nothing
about the host goes in.

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

from scripts.mad.mixers import REGISTRY, load_module, resolve
from scripts.mad.model import ModelConfig, build_model, parameter_count
from scripts.mad.tasks import LEAKAGE_LIMIT, TASKS, Pool, TaskSpec, build_pool
from scripts.mad.train import Point, Report, TrainConfig, seed_all, train


def parse_axes(pairs: list[str]) -> dict[str, Any]:
    """Read ``axis=value`` strings into task settings.

    Args:
        pairs: Strings of the form ``axis=value``. Values are read as int, then float,
            then the literal ``true``/``false``, then string.

    Returns:
        The axes, for :meth:`scripts.mad.tasks.TaskSpec.override`.

    Raises:
        ValueError: On a string with no ``=``.
    """
    axes: dict[str, Any] = {}
    for pair in pairs:
        key, sep, text = pair.partition("=")
        if not sep:
            raise ValueError(f"axis must be key=value, got {pair!r}")
        axes[key] = _literal(text)
    return axes


def _literal(text: str) -> Any:
    """Read a command-line value at its narrowest type."""
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


def _spec(name: str, axes: dict[str, Any]) -> TaskSpec:
    """One task's spec with the requested axes moved.

    Args:
        name: A key of :data:`scripts.mad.tasks.TASKS`.
        axes: Settings to move. An axis the task does not have is an error, not a no-op.

    Returns:
        The spec.

    Raises:
        KeyError: On an unknown task, naming the six.
        ValueError: From :meth:`scripts.mad.tasks.TaskSpec.override`.
    """
    if name not in TASKS:
        raise KeyError(f"no task {name}; have {sorted(TASKS)}")
    spec = TASKS[name]
    return spec.override(**axes) if axes else spec


def run_task(
    spec: TaskSpec,
    mixer_name: str,
    overrides: list[str],
    model_args: dict[str, Any],
    config: TrainConfig,
    *,
    data_seed: int,
    quiet: bool,
) -> dict[str, Any]:
    """Build a pool, build a model, run the protocol, and return the record.

    Args:
        spec: The task, baseline or with axes moved.
        mixer_name: Registry key.
        overrides: Mixer settings, as ``key=value`` strings.
        model_args: Scaffold settings beyond what the task fixes: ``d_model``,
            ``n_layers``, and the divergence flags of
            :class:`scripts.mad.model.ModelConfig`.
        config: The protocol. Its seed drives initialization and the shuffle.
        data_seed: Seeds the pool. Held apart from the model seed so a paired
            comparison can share a pool across arms and vary only the initialization.
        quiet: Suppress the per-evaluation lines on stderr.

    Returns:
        The record, JSON-ready.
    """
    pool = build_pool(spec, seed=data_seed)
    mixer = resolve(mixer_name, overrides)
    model_config = ModelConfig(
        vocab_size=spec.vocab_size,
        width=pool.width,
        bottleneck=spec.bottleneck,
        **model_args,
    )
    seed_all(config.seed)
    model = build_model(model_config, mixer.factory).to(config.device)

    def echo(point: Point) -> None:
        print(
            f"  {spec.name} epoch {point.epoch + 1} "
            f"train {point.train_loss:.4f} test {point.test.loss:.4f} "
            f"micro {point.test.micro:.4f} macro {point.test.macro:.4f}",
            file=sys.stderr,
            flush=True,
        )

    report = train(
        model,
        pool,
        config,
        vocab_size=spec.vocab_size,
        on_point=None if quiet else echo,
    )
    return _record(
        spec,
        mixer.name,
        mixer.settings,
        model_config,
        config,
        pool,
        report,
        parameter_count(model),
    )


def _record(
    spec: TaskSpec,
    mixer_name: str,
    settings: dict[str, Any],
    model_config: ModelConfig,
    config: TrainConfig,
    pool: Pool,
    report: Report,
    parameters: int,
) -> dict[str, Any]:
    """Assemble one arm's record.

    Args:
        spec: The task.
        mixer_name: Registry key.
        settings: Resolved mixer settings.
        model_config: Scaffold shape.
        config: The protocol.
        pool: The pool, for its leakage and width.
        report: What the run produced.
        parameters: Trainable parameter count.

    Returns:
        A JSON-ready dict. ``leaky`` flags a pool over
        :data:`scripts.mad.tasks.LEAKAGE_LIMIT`, which invalidates the arm rather than
        merely warning about it.
    """
    return {
        "task": spec.name,
        "mad_task": spec.mad_name,
        "task_settings": {
            "vocab_size": spec.vocab_size,
            "seq_len": spec.seq_len,
            "num_train": spec.num_train,
            "num_test": spec.num_test,
            **spec.extra,
        },
        "mixer": mixer_name,
        "mixer_settings": settings,
        "model": asdict(model_config),
        "protocol": asdict(config),
        "width": pool.width,
        "leakage": pool.leakage,
        "leaky": pool.leakage > LEAKAGE_LIMIT,
        "parameters": parameters,
        "best": report.best._asdict(),
        "best_epoch": report.best_epoch,
        "final": report.final._asdict(),
        "epochs_run": report.epochs_run,
        "stopped_early": report.stopped_early,
        "points": [
            [p.epoch, p.step, p.train_loss, p.test.loss, p.test.micro, p.test.macro]
            for p in report.points
        ],
    }


def table(records: list[dict[str, Any]]) -> str:
    """One row per record, plus the mean of the accuracies.

    Args:
        records: What :func:`run_task` returned, in order.

    Returns:
        The table. Accuracy is the best evaluation's, micro first because micro is what
        decides an arm.
    """
    head = f"{'task':<6} {'mixer':<12} {'seed':>5} {'micro':>8} {'macro':>8} {'loss':>8} {'epochs':>7} {'params':>9}"
    lines = [head, "-" * len(head)]
    for record in records:
        best = record["best"]
        lines.append(
            f"{record['task']:<6} {record['mixer']:<12} "
            f"{record['protocol']['seed']:>5} {best['micro']:>8.4f} "
            f"{best['macro']:>8.4f} {best['loss']:>8.4f} "
            f"{record['epochs_run']:>7} {record['parameters']:>9}"
        )
    if len(records) > 1:
        micro = sum(r["best"]["micro"] for r in records) / len(records)
        macro = sum(r["best"]["macro"] for r in records) / len(records)
        lines.append("-" * len(head))
        lines.append(f"{'mean':<6} {'':<12} {'':>5} {micro:>8.4f} {macro:>8.4f}")
    leaky = [r["task"] for r in records if r["leaky"]]
    if leaky:
        lines.append(f"leaked pools: {sorted(set(leaky))}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser. Every protocol field is a flag so an arm is reproducible from its
        record, and the defaults are the protocol's.
    """
    protocol = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="scripts.mad.run", description="Run MAD arms."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=sorted(TASKS),
        metavar="NAME",
        help=f"tasks to run, from {sorted(TASKS)}; default all six",
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
        "--axis",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="move a task off its baseline, e.g. seq_len=256",
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
        help="model and shuffle seeds; one arm per seed",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="pool seed; defaults to each arm's own seed",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=protocol.epochs)
    parser.add_argument("--batch-size", type=int, default=protocol.batch_size)
    parser.add_argument("--lr", type=float, default=protocol.lr)
    parser.add_argument("--weight-decay", type=float, default=protocol.weight_decay)
    parser.add_argument(
        "--schedule", default=protocol.schedule, choices=("none", "cosine")
    )
    parser.add_argument("--grad-clip", type=float, default=protocol.grad_clip)
    parser.add_argument("--patience", type=int, default=protocol.patience)
    parser.add_argument("--log-every", type=int, default=protocol.log_every)
    parser.add_argument(
        "--precision", default=protocol.precision, choices=("fp32", "bf16")
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="where the model and the pool live",
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
        Process exit status. Nonzero when any pool leaked, since such an arm reports
        recall of its own train split.
    """
    args = build_parser().parse_args(argv)
    for module in args.mixer_module:
        load_module(module)

    axes = parse_axes(args.axis)
    model_args = {"d_model": args.d_model, "n_layers": args.n_layers}
    base = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        schedule=args.schedule,
        grad_clip=args.grad_clip,
        patience=args.patience,
        log_every=args.log_every,
        precision=args.precision,
        device=args.device,
    )

    records: list[dict[str, Any]] = []
    handle = args.out.open("a") if args.out is not None else None
    try:
        for seed in args.seed:
            for name in args.task:
                spec = _spec(name, axes)
                record = run_task(
                    spec,
                    args.mixer,
                    args.settings,
                    model_args,
                    replace(base, seed=seed),
                    data_seed=seed if args.data_seed is None else args.data_seed,
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
    return 1 if any(r["leaky"] for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
