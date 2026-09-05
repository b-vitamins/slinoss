"""Run one MAD arm, or the six-task suite, and write one record per task.

    python3 -m scripts.mad.run --mixer slinoss --set d_state=192
    python3 -m scripts.mad.run --mixer attention --task icr ficr --seed 0 1 2
    python3 -m scripts.mad.run --mixer conv --task sc --axis num_tokens_to_copy=64

Records go to stdout as JSON lines and the summary table to stderr, so a redirect keeps
the data and leaves the table on the terminal. A record carries everything an arm is: the
task and any axis moved off its baseline, the mixer and every resolved setting, the
protocol, the pool's leakage and width, the parameter count, and each evaluation. Nothing
about the runtime environment or credentials goes in; source and command provenance do.

A baseline whose package is not a dependency of this tree registers itself in a module of
its own; ``--mixer-module`` imports it before the name is resolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from scripts.mad.mixers import REGISTRY, load_module, resolve
from scripts.mad.model import ModelConfig, build_model, parameter_count
from scripts.mad.profiles import PROFILES, HarnessProfile, get_profile
from scripts.mad.tasks import LEAKAGE_LIMIT, TASKS, Pool, TaskSpec, build_pool
from scripts.mad.train import Point, Report, TrainConfig, seed_all, train
from scripts.provenance import capture as capture_provenance
from scripts.provenance import identity


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


def _task_baselines() -> dict[str, dict[str, Any]]:
    """Serialize all six baseline task contracts into every named profile record."""
    return {
        name: {
            "mad_name": spec.mad_name,
            "vocab_size": spec.vocab_size,
            "seq_len": spec.seq_len,
            "num_train": spec.num_train,
            "num_test": spec.num_test,
            "bottleneck": spec.bottleneck,
            "split_policy": spec.split_policy,
            **spec.extra,
        }
        for name, spec in sorted(TASKS.items())
    }


def run_task(
    spec: TaskSpec,
    mixer_name: str,
    overrides: list[str],
    model_args: dict[str, Any],
    config: TrainConfig,
    *,
    data_seed: int,
    quiet: bool,
    provenance: dict[str, Any] | None = None,
    profile_record: dict[str, Any] | None = None,
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
        provenance: Source/harness/command identity captured before the task starts.
        profile_record: Atomic scaffold/protocol profile resolved by the CLI. The
            programmatic API records itself explicitly when omitted.

    Returns:
        The record, JSON-ready.
    """
    pool = build_pool(spec, seed=data_seed)
    mixer = resolve(mixer_name, overrides)
    model_config = ModelConfig(
        vocab_size=spec.vocab_size,
        task_length=spec.seq_len,
        observed_width=pool.width,
        bottleneck=spec.bottleneck,
        **model_args,
    )
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
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
        data_seed,
        mixer.max_length_policy,
        mixer.constructions,
        capture_provenance("scripts/mad", ["<python-api>"], module="scripts.mad.run")
        if provenance is None
        else provenance,
        {
            "name": model_config.scaffold_profile,
            "locked": False,
            "references": ["programmatic API; inspect resolved model and protocol"],
        }
        if profile_record is None
        else profile_record,
    )


def _pool_identity(pool: Pool) -> str:
    """Hash the exact train/test arrays, including shapes and dtypes."""
    digest = hashlib.sha256()
    for name, array in (
        ("train_inputs", pool.train_inputs),
        ("train_targets", pool.train_targets),
        ("test_inputs", pool.test_inputs),
        ("test_targets", pool.test_targets),
    ):
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(array.shape).encode() + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _record(
    spec: TaskSpec,
    mixer_name: str,
    settings: dict[str, Any],
    model_config: ModelConfig,
    config: TrainConfig,
    pool: Pool,
    report: Report,
    parameters: int,
    data_seed: int,
    max_length_policy: str,
    constructions: list[dict[str, Any]],
    provenance: dict[str, Any],
    profile_record: dict[str, Any],
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
        data_seed: Seed from which the exact pool was drawn.
        max_length_policy: Declared consumption of configured task length.
        constructions: Effective per-layer mixer configurations.
        provenance: Source, harness, dirty tree, and command identity.
        profile_record: Named atomic harness contract.

    Returns:
        A JSON-ready dict. ``leaky`` flags a pool over
        :data:`scripts.mad.tasks.LEAKAGE_LIMIT`, which invalidates the arm rather than
        merely warning about it.
    """
    task_settings = {
        "vocab_size": spec.vocab_size,
        "seq_len": spec.seq_len,
        "num_train": spec.num_train,
        "num_test": spec.num_test,
        **spec.extra,
    }
    initialization_policies = {
        construction["initialization_policy"] for construction in constructions
    }
    if len(initialization_policies) != 1:
        raise RuntimeError(
            "one mixer resolved to inconsistent initialization policies: "
            f"{sorted(initialization_policies)}"
        )
    initialization_policy = next(iter(initialization_policies))
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
    return {
        "task": spec.name,
        "mad_task": spec.mad_name,
        "task_settings": task_settings,
        "task_contract": {
            "split_policy": spec.split_policy,
            "generator": f"{spec.generator.__module__}.{spec.generator.__qualname__}",
        },
        "mixer": mixer_name,
        "mixer_settings": settings,
        "mixer_contract": {
            "max_length_policy": max_length_policy,
            "initialization_policy": initialization_policy,
            "initialization": (
                "mixer constructor; protected from scaffold reinitialization"
                if initialization_policy == "constructor"
                else "explicit scaffold pass over nested Linear/Embedding parameters"
            ),
        },
        "mixer_constructions": constructions,
        "harness_profile": profile_record,
        "model": asdict(model_config),
        "protocol": asdict(config),
        "selection": {
            "split": "test",
            "metric": "micro_accuracy",
            "evaluation_interval_epochs": config.eval_every,
        },
        "width": pool.width,
        "lengths": {
            "configured_task_length": spec.seq_len,
            "training_ceiling": int(pool.train_inputs.shape[1]),
            "evaluation_ceiling": int(pool.test_inputs.shape[1]),
            "observed_tensor_width": {
                "train": int(pool.train_inputs.shape[1]),
                "evaluation": int(pool.test_inputs.shape[1]),
            },
            "mixer_initialization_lattice": init_lattices or None,
        },
        "seeds": {
            "model": config.seed,
            "shuffle": config.seed,
            "data": data_seed,
        },
        "pool": {
            "identity": _pool_identity(pool),
            "spec_identity": identity(task_settings),
            "train_examples": int(pool.train_inputs.shape[0]),
            "test_examples": int(pool.test_inputs.shape[0]),
        },
        "initialization": {
            "scaffold": f"normal std={model_config.init_std}; mixer parameters exempt",
            "mixer": "owned by each mixer constructor",
        },
        "provenance": provenance,
        "leakage": pool.leakage,
        "leaky": pool.leakage > LEAKAGE_LIMIT,
        "parameters": parameters,
        "best": report.best._asdict(),
        "best_epoch": report.best_epoch + 1,
        "epoch_indexing": "one_based",
        "final": report.final._asdict(),
        "epochs_run": report.epochs_run,
        "stopped_early": report.stopped_early,
        "points": [
            [p.epoch + 1, p.step, p.train_loss, p.test.loss, p.test.micro, p.test.macro]
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
        "--profile",
        default="legacy-hybrid",
        choices=sorted(PROFILES),
        help=(
            "atomic scaffold/protocol contract; kla-paper-v2 is a locked textual "
            "reconstruction, while legacy-hybrid permits explicit overrides for replay"
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
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--schedule", default=None, choices=("none", "cosine"))
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="drop a short final training batch; KLA does, MAD-Lab does not",
    )
    parser.add_argument(
        "--float32-matmul-precision",
        default=None,
        choices=("highest", "high", "medium"),
    )
    parser.add_argument("--precision", default=None, choices=("fp32", "bf16"))
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


def _resolved_profile(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[HarnessProfile, dict[str, Any], dict[str, Any]]:
    """Resolve optional CLI values against one atomic profile.

    A locked profile accepts a repeated value, which keeps fully explicit replay
    commands valid, but refuses a conflicting value instead of quietly ceasing to be
    that profile.
    """
    profile = get_profile(args.profile)
    model_args = profile.model_args()
    train_args = profile.train_args()
    model_cli = {"d_model": "--d-model", "n_layers": "--n-layers"}
    train_cli = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "schedule": "--schedule",
        "grad_clip": "--grad-clip",
        "patience": "--patience",
        "eval_every": "--eval-every",
        "drop_last": "--drop-last/--no-drop-last",
        "float32_matmul_precision": "--float32-matmul-precision",
        "precision": "--precision",
    }

    for field, flag in model_cli.items():
        supplied = getattr(args, field)
        if supplied is None:
            continue
        expected = model_args[field]
        if profile.locked and supplied != expected:
            parser.error(
                f"profile {profile.name} locks {flag} to {expected!r}; got {supplied!r}"
            )
        model_args[field] = supplied

    for field, flag in train_cli.items():
        supplied = getattr(args, field)
        if supplied is None:
            continue
        expected = train_args[field]
        if profile.locked and supplied != expected:
            parser.error(
                f"profile {profile.name} locks {flag} to {expected!r}; got {supplied!r}"
            )
        train_args[field] = supplied

    train_args["device"] = args.device
    return profile, model_args, train_args


def main(argv: list[str] | None = None) -> int:
    """Run every requested task at every requested seed.

    Args:
        argv: Command line, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status. Nonzero when any pool leaked, since such an arm reports
        recall of its own train split.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    profile, model_args, train_args = _resolved_profile(args, parser)
    if not profile.published_table_eligible:
        print(
            f"  profile {profile.name!r} is {profile.contract_status} and is not "
            "eligible by itself for a published-table claim",
            file=sys.stderr,
            flush=True,
        )
    provenance = capture_provenance("scripts/mad", argv, module="scripts.mad.run")
    provenance["mixer_modules"] = list(args.mixer_module)
    provenance["harness_profile"] = profile.name
    for module in args.mixer_module:
        load_module(module)

    axes = parse_axes(args.axis)
    base = TrainConfig(**train_args)
    profile_record = profile.record()
    task_baselines = _task_baselines()
    profile_record["task_baselines"] = task_baselines
    profile_record["task_baselines_identity"] = identity(task_baselines)
    profile_record["identity"] = identity(profile_record)

    records: list[dict[str, Any]] = []
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
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
                    provenance=provenance,
                    profile_record=profile_record,
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
