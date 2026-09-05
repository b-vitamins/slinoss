"""The driver: one point, or one host's slice of a plan.

Two modes, one code path under them. ``--dataset X --mixer Y --seed S`` runs a single point;
``--shard i/n`` runs the slice :mod:`scripts.tsc.sweep` assigns to host ``i`` of ``n``. Both build
a one-point-or-many :class:`scripts.tsc.sweep.Lattice` first, so a single run carries the same
plan digest machinery as a sweep and its record merges with one.

A run record is a JSON file named for the point's key. It carries the plan digest, the resolved
setting, the mixer's settings after overrides, the corpus manifest, the partition sizes, the
parameter counts and every evaluation point. Enough to answer what produced a number without the
process that produced it, and enough for :func:`scripts.tsc.sweep.merge` to refuse a harvest that
spans two plans.

``--skip-existing`` makes a shard resumable: a killed host is restarted with the same command and
picks up where it stopped. The check is the record file's presence, so a partial write would be
taken for a finished run; the record is written by rename from a temporary in the same directory,
which on a POSIX filesystem makes it appear whole or not at all.

The seed does two jobs and they are separated on purpose. The partition is JAX's stream, from
:mod:`scripts.tsc.prng`, because the published bars depend on it. Initialization and dropout are
torch's, seeded here; batch order is a dedicated generator in :mod:`scripts.tsc.batching`. So an
arm is reproducible on this harness and its data split is the reference's.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from scripts.harness import load_module
from scripts.provenance import capture as capture_provenance
from scripts.provenance import identity
from scripts.tsc import model as model_module
from scripts.tsc.batching import Loader
from scripts.tsc.corpus import load
from scripts.tsc.mixers import REGISTRY, paper_overrides
from scripts.tsc.protocol import (
    DATASETS,
    HORIZON,
    NUM_STEPS,
    PATIENCE,
    PRINT_STEPS,
    REFERENCE,
    SEEDS,
)
from scripts.tsc.split import apply, partition, prepare
from scripts.tsc.sweep import (
    Axis,
    Lattice,
    Point,
    plan_digest,
    points,
    setting_for_point,
    shard,
)
from scripts.tsc.train import Splits, TrainConfig, check_finite, train

__all__ = [
    "build_parser",
    "execute",
    "lattice_from",
    "main",
    "record_path",
    "run_point",
]


def record_path(out: Path, point: Point) -> Path:
    """Where one point's record goes.

    Args:
        out: Output directory.
        point: The run.

    Returns:
        The path. Named for the key alone, so two shards of one plan can write into one
        directory without coordinating.
    """
    return out / f"{point.key}.json"


def _write_record(path: Path, record: dict[str, Any]) -> None:
    """Write a record so it appears whole or not at all.

    Args:
        path: Destination.
        record: The record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(temporary, path)


def _device_for(name: str | None) -> torch.device:
    """Pick the device.

    Args:
        name: An explicit device, or None to take a card when one is visible.

    Returns:
        The device.
    """
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_point(
    point: Point,
    corpus_root: Path,
    *,
    device: torch.device,
    plan: str,
    num_steps: int = NUM_STEPS,
    print_steps: int = PRINT_STEPS,
    patience: int = PATIENCE,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and train one point, and return its record.

    Args:
        point: The run.
        corpus_root: Directory holding one processed dataset per folder, from
            :func:`scripts.tsc.corpus.process`.
        device: Where the model and both split tensors live.
        plan: The plan digest to record.
        num_steps: Step cap.
        print_steps: Evaluation interval.
        patience: Non-improving evaluations tolerated.
        provenance: Source, harness, dirty tree, and exact command identity.

    Returns:
        The record.

    Raises:
        FileNotFoundError: When the dataset is not processed under ``corpus_root``.
        KeyError: On a dataset outside the protocol or a mixer outside the registry.
        ValueError: From the corpus digest check, from :func:`scripts.tsc.train.check_finite`,
            from the mixer's settings, or from the loop's batch size check.
    """
    setting = setting_for_point(point)
    corpus = load(corpus_root / point.dataset)
    arrays = prepare(corpus, include_time=setting.include_time, horizon=HORIZON)
    check_finite(arrays, dataset=point.dataset)
    rows = partition(corpus.manifest.instances, point.seed)
    train_arrays, val_arrays, test_arrays = apply(arrays, rows)
    splits = Splits(
        Loader(train_arrays, device),
        Loader(val_arrays, device),
        Loader(test_arrays, device),
    )

    # The published state width first, then the point's own overrides, so a swept ssm_dim wins
    # over the paper's for that dataset and a mixer with no published width takes only the sweep.
    overrides = [*paper_overrides(point.mixer, setting), *point.settings]
    mixer = REGISTRY.resolve(point.mixer, overrides)
    config = model_module.ModelConfig(
        input_dim=splits.train.channels,
        hidden_dim=setting.hidden_dim,
        classes=splits.train.classes,
        blocks=setting.blocks,
    )
    torch.manual_seed(point.seed)
    built = model_module.build_model(
        config,
        [mixer.factory] * setting.blocks,
        max_length=splits.train.length,
        device=device,
    )

    started = time.monotonic()
    result = train(
        built,
        splits,
        TrainConfig(
            lr=setting.lr,
            batch_size=setting.batch_size,
            num_steps=num_steps,
            print_steps=print_steps,
            patience=patience,
            seed=point.seed,
        ),
    )
    reference = REFERENCE.get(point.dataset)
    source = {
        "corpus": asdict(corpus.manifest),
        "partition_seed": point.seed,
        "split_sizes": list(rows.sizes),
    }
    parameter = next(built.parameters())
    return {
        "plan": plan,
        "key": point.key,
        "position": point.position,
        "dataset": point.dataset,
        "mixer": mixer.name,
        "seed": point.seed,
        "setting": asdict(setting),
        "overrides": list(overrides),
        "mixer_settings": mixer.settings,
        "mixer_contract": {
            "max_length_policy": mixer.max_length_policy,
            "initialization": "mixer_constructor; no scaffold reinitialization",
        },
        "mixer_constructions": mixer.constructions,
        "corpus": asdict(corpus.manifest),
        "split_sizes": list(rows.sizes),
        "input_dim": splits.train.channels,
        "length": splits.train.length,
        "lengths": {
            "configured_task_length": corpus.manifest.length,
            "training_ceiling": splits.train.length,
            "evaluation_ceiling": max(splits.val.length, splits.test.length),
            "observed_tensor_width": {
                "train": splits.train.length,
                "validation": splits.val.length,
                "test": splits.test.length,
            },
        },
        "seeds": {
            "model": point.seed,
            "partition": point.seed,
            "batch_order": point.seed,
        },
        "data": {**source, "identity": identity(source)},
        "initialization": {
            "scaffold": "framework constructor defaults; no post-construction pass",
            "mixer": "owned by each mixer constructor",
        },
        "precision": {
            "parameter_dtype": str(parameter.dtype),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast": False,
        },
        "provenance": (
            capture_provenance(
                "scripts/tsc", ["<python-api>"], module="scripts.tsc.run"
            )
            if provenance is None
            else provenance
        ),
        "classes": splits.train.classes,
        "parameters": model_module.parameter_count(built),
        "mixer_parameters": model_module.mixer_parameters(built),
        "device": str(device),
        "test_accuracy": result.test_accuracy,
        "val_accuracy": result.val_accuracy,
        "best_step": result.best_step,
        "steps": result.steps,
        "stopped_early": result.stopped_early,
        "seconds": time.monotonic() - started,
        "evaluations": [ev._asdict() for ev in result.evaluations],
        "reference": None if reference is None else reference._asdict(),
    }


def lattice_from(options: argparse.Namespace) -> Lattice:
    """The plan the invocation names.

    In single-point mode the lattice holds exactly that dataset, mixer and seed, so the point
    gets a plan digest and a key like any other and its record merges with a sweep's.

    Args:
        options: Parsed arguments.

    Returns:
        The lattice.

    Raises:
        ValueError: From :class:`scripts.tsc.sweep.Lattice`, or when single-point mode is given
            more than one of anything.
    """
    datasets = tuple(options.datasets or DATASETS)
    mixers = tuple(options.mixers or ("linoss_im",))
    seeds = tuple(options.seeds or SEEDS)
    if options.shard is None and (
        len(datasets) != 1 or len(mixers) != 1 or len(seeds) != 1
    ):
        raise ValueError(
            "without --shard the invocation must name one dataset, one mixer and one seed; "
            f"got {len(datasets)}, {len(mixers)} and {len(seeds)}"
        )
    return Lattice(
        datasets=datasets,
        mixers=mixers,
        seeds=seeds,
        axes=tuple(Axis.parse(spec) for spec in options.sweep),
        fixed=tuple(options.set),
        num_steps=options.num_steps,
    )


def execute(
    options: argparse.Namespace, *, provenance: dict[str, Any] | None = None
) -> int:
    """Run what the invocation asks for.

    Args:
        options: Parsed arguments.
        provenance: One source/harness/command identity shared by every selected point.

    Returns:
        A process exit status: zero unless a point raised under ``--keep-going``.

    Raises:
        ValueError: From :func:`lattice_from` or from :func:`scripts.tsc.sweep.shard`.
    """
    for path in options.mixer_module:
        load_module(path)
    if provenance is None:
        provenance = capture_provenance(
            "scripts/tsc", ["<python-api>"], module="scripts.tsc.run"
        )
    lattice = lattice_from(options)
    plan = plan_digest(lattice)
    if options.shard is None:
        selected = points(lattice)
    else:
        index, _, count = options.shard.partition("/")
        selected = shard(
            lattice, int(index), int(count), weights=options.weights
        ).points

    if options.list:
        print(f"plan {plan}")
        for point in selected:
            print(f"{point.position}\t{point.key}\t{' '.join(point.settings) or '-'}")
        return 0

    out = Path(options.out)
    device = _device_for(options.device)
    failures = 0
    for nth, point in enumerate(selected, start=1):
        path = record_path(out, point)
        if options.skip_existing and path.is_file():
            print(f"[{nth}/{len(selected)}] {point.key} already recorded")
            continue
        print(f"[{nth}/{len(selected)}] {point.key} on {device}", flush=True)
        try:
            record = run_point(
                point,
                Path(options.corpus),
                device=device,
                plan=plan,
                num_steps=options.num_steps,
                print_steps=options.print_steps,
                patience=options.patience,
                provenance=provenance,
            )
        except Exception as exc:
            if not options.keep_going:
                raise
            # A shard of forty points must not lose thirty-nine to one bad configuration, and
            # the failure has to be visible in the harvest rather than only in a log.
            failures += 1
            _write_record(
                path,
                {
                    "plan": plan,
                    "key": point.key,
                    "position": point.position,
                    "dataset": point.dataset,
                    "mixer": point.mixer,
                    "seed": point.seed,
                    "provenance": provenance,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        _write_record(path, record)
        print(
            f"  test {record['test_accuracy']:.4f} val {record['val_accuracy']:.4f} "
            f"at step {record['best_step']} of {record['steps']}",
            flush=True,
        )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="scripts.tsc.run",
        description="Run one time series classification point, or one shard of a sweep.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="directory of processed datasets, one folder each",
    )
    parser.add_argument("--out", default="runs/tsc", help="where run records go")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        metavar="NAME",
        help="protocol dataset; repeatable, defaults to all six",
    )
    parser.add_argument(
        "--mixer",
        dest="mixers",
        action="append",
        default=[],
        metavar="NAME",
        help=f"registry mixer; repeatable. registered: {', '.join(REGISTRY.names())}",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help="seed; repeatable, defaults to the protocol's five",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override held fixed across the plan, scaffold or mixer; repeatable",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="swept axis, scaffold or mixer; repeatable",
    )
    parser.add_argument(
        "--shard",
        metavar="I/N",
        help="run shard I of N, cost-balanced over the whole plan",
    )
    parser.add_argument(
        "--weights",
        type=lambda text: [float(part) for part in text.split(",")],
        help="relative capacity per shard, comma separated, for an uneven fleet",
    )
    parser.add_argument(
        "--mixer-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="import before resolving, for an out-of-tree mixer; repeatable",
    )
    parser.add_argument("--device", help="torch device; defaults to cuda when visible")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--print-steps", type=int, default=PRINT_STEPS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave points that already have a record, so a shard resumes",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record a failed point and continue instead of stopping the shard",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the selection and run nothing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments, or None for the process's.

    Returns:
        A process exit status.
    """
    provenance = capture_provenance("scripts/tsc", argv, module="scripts.tsc.run")
    return execute(build_parser().parse_args(argv), provenance=provenance)


if __name__ == "__main__":
    raise SystemExit(main())
