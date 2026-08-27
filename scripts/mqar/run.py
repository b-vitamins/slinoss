"""MQAR driver. One JSON record per cell, on stdout.

A cell is one ``(d_model, lr, seed)`` point. The published protocol reports the maximum
over the learning-rate sweep for each width, so a width's number is only defined once its
whole ``--lr`` grid has run; the driver emits every point and leaves that maximum to
whatever reads the records.

The ICLR24 figure-2 cell, which is one train segment and one test segment at matched
length and key-value count, filler off, 64 epochs::

    python3 -m scripts.mqar.run --preset figure2 --cell 128:8 \\
        --mixer slinoss --d-model 64 128 256 512 --lr 1e-4 4.64e-4 2.15e-3 1e-2

The modern repro, which trains to length 256 and tests to 1024 so length generalization is
inside the protocol rather than beside it, filler on, 32 epochs at batch 256/32::

    python3 -m scripts.mqar.run --preset repro --mixer conv slinoss \\
        --d-model 32 64 128 --lr 1e-3 3.16e-3 1e-2 3.16e-2

The repro pool ships in two variants, and the second one differs from the first in exactly
two settings::

    python3 -m scripts.mqar.run --preset repro --no-random-non-queries \\
        --embedding-init spherical --mixer conv slinoss ...

Both published sweeps take the maximum over their learning-rate grid, and those grids are
the ones written above: ``logspace(-4, -2, 4)`` for figure 2 and ``logspace(-3, -1.5, 4)``
for the repro.

Ad-hoc pools come from ``--train`` and ``--test``, repeatable, each
``length:kv_pairs:examples``. Their defaults are the generator's: vocabulary 8192, power
law 0.01, filler on. A preset overrides that with whatever its own config ran.

No position embedding by default. ``--positions length`` turns on the learned absolute
position table, which figure 2 gives attention and nothing else.

``--dry-run`` builds the pool and the model, reports the shapes, the parameter count and
the measured train-test leakage, and trains nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.mqar import mixers as mixer_registry
from scripts.mqar.model import (
    EMBEDDING_INITS,
    LanguageModel,
    ModelConfig,
    parameter_count,
)
from scripts.mqar.tasks import (
    FIGURE2_CELLS,
    Pool,
    PoolSpec,
    Segment,
    SegmentSpec,
    batch_size_for,
    build_pool,
    figure2_spec,
    repro_spec,
)
from scripts.mqar.train import (
    Metrics,
    Report,
    TrainConfig,
    batch_count,
    seed_all,
    train,
)

PRESETS = ("repro", "figure2")
"""Named published protocols.

Each fixes the pool, the filler, the epoch count and the batch size to what its own config
ran: ``repro`` 32 epochs at batch 256/32 with the filler on, ``figure2`` 64 epochs at the
length ladder with the filler off. ``figure2`` needs ``--cell``.
"""

PROTOCOL_EPOCHS = {"repro": 32, "figure2": 64}
"""Epoch count per preset. 64 is also the default for an ad-hoc pool."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    parser = argparse.ArgumentParser(
        prog="scripts.mqar.run", description="Multi-query associative recall."
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--train",
        action="append",
        default=[],
        metavar="LEN:KV:N",
        help="a train segment; repeatable",
    )
    data.add_argument(
        "--test",
        action="append",
        default=[],
        metavar="LEN:KV:N",
        help="a test segment; repeatable",
    )
    data.add_argument(
        "--preset",
        choices=PRESETS,
        help="a published pool, protocol and filler setting",
    )
    data.add_argument(
        "--cell",
        metavar="LEN:KV",
        help=f"figure-2 cell; the published ones are {list(FIGURE2_CELLS)}",
    )
    data.add_argument("--vocab-size", type=int, default=8192)
    data.add_argument("--power-a", type=float, default=0.01)
    data.add_argument(
        "--random-non-queries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fill non-query positions with uniform tokens instead of the padding id; "
        "default is the preset's, or on for an ad-hoc pool",
    )
    data.add_argument("--data-seed", type=int, default=123)
    data.add_argument(
        "--no-leakage-check",
        action="store_true",
        help="skip the train-test row comparison",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--mixer",
        nargs="+",
        default=["slinoss"],
        metavar="NAME",
        help="one name per layer position, cycled; e.g. --mixer conv slinoss",
    )
    model.add_argument(
        "--set",
        nargs="*",
        default=[],
        dest="settings",
        metavar="KEY=VALUE",
        help="mixer setting; scope it as MIXER.KEY=VALUE when several are named",
    )
    model.add_argument(
        "--mixer-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="import a module so its register() calls run; dotted name or .py path",
    )
    model.add_argument("--d-model", nargs="+", type=int, default=[128])
    model.add_argument("--n-layers", type=int, default=2)
    model.add_argument("--state-mixer", default="identity", choices=("identity", "mlp"))
    model.add_argument("--hidden-mult", type=int, default=4)
    model.add_argument(
        "--positions",
        default="none",
        choices=("none", "length"),
        help="learned absolute position embeddings; 'length' sizes them to the pool",
    )
    model.add_argument(
        "--embedding-init",
        default="default",
        choices=EMBEDDING_INITS,
        help="word-embedding draw; the repro sweep with the filler off runs 'spherical'",
    )
    model.add_argument("--embed-dropout", type=float, default=0.1)
    model.add_argument("--resid-dropout", type=float, default=0.0)
    model.add_argument("--init-std", type=float, default=0.02)
    model.add_argument(
        "--untied-embeddings",
        action="store_true",
        help="freeze the word embedding and leave the head untied",
    )

    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--lr", nargs="+", type=float, default=[1e-3])
    protocol.add_argument("--seed", nargs="+", type=int, default=[123])
    protocol.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="0 selects the protocol default: 64, or 32 under --preset repro",
    )
    protocol.add_argument(
        "--batch-size", type=int, default=0, help="0 selects the published ladder"
    )
    protocol.add_argument(
        "--test-batch-size", type=int, default=0, help="0 matches the train batch"
    )
    protocol.add_argument("--weight-decay", type=float, default=0.1)
    protocol.add_argument("--early-stop", type=float, default=0.99)
    protocol.add_argument("--precision", default="fp32", choices=("fp32", "bf16"))
    protocol.add_argument("--device", default="cuda")

    output = parser.add_argument_group("output")
    output.add_argument("--out", help="append records here as well as to stdout")
    output.add_argument(
        "--points", action="store_true", help="include every epoch in the record"
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="build the pool and the model, report, train nothing",
    )
    return parser


def parse_cell(text: str) -> tuple[int, int]:
    """Read ``LEN:KV``.

    Args:
        text: The cell.

    Returns:
        ``(input_seq_len, num_kv_pairs)``.

    Raises:
        ValueError: On the wrong field count or a non-integer field.
    """
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"cell {text!r} is not LEN:KV")
    try:
        length, pairs = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"cell {text!r} has a non-integer field") from error
    return length, pairs


def parse_segment(text: str, power_a: float = 0.01) -> SegmentSpec:
    """Read ``LEN:KV:N``.

    Args:
        text: The spec.
        power_a: Query-offset exponent to build it at.

    Returns:
        A :class:`scripts.mqar.tasks.SegmentSpec`.

    Raises:
        ValueError: On the wrong field count or a non-integer field.
    """
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"segment {text!r} is not LEN:KV:N")
    try:
        length, pairs, examples = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"segment {text!r} has a non-integer field") from error
    return SegmentSpec(
        input_seq_len=length,
        num_kv_pairs=pairs,
        num_examples=examples,
        power_a=power_a,
    )


def pool_spec(args: argparse.Namespace) -> PoolSpec:
    """Assemble the pool spec from the parsed arguments.

    A preset supplies the pool, the vocabulary and the filler; ``--random-non-queries`` or
    ``--no-random-non-queries`` overrides the filler and nothing else overrides a preset.

    Raises:
        ValueError: If neither a preset nor both segment lists were given, if both were, if
            ``--cell`` is given without ``--preset figure2`` or missing with it.
    """
    if args.preset and (args.train or args.test):
        raise ValueError("--preset and --train/--test are mutually exclusive")
    if args.cell and args.preset != "figure2":
        raise ValueError("--cell belongs to --preset figure2")
    if args.preset and args.vocab_size != 8192:
        raise ValueError(
            f"--preset {args.preset} fixes the vocabulary at 8192; --vocab-size "
            f"{args.vocab_size} would be silently ignored"
        )
    if args.preset == "figure2":
        if not args.cell:
            raise ValueError(f"--preset figure2 needs --cell, one of {FIGURE2_CELLS}")
        length, pairs = parse_cell(args.cell)
        spec = figure2_spec(length, pairs, seed=args.data_seed, power_a=args.power_a)
    elif args.preset == "repro":
        spec = repro_spec(seed=args.data_seed, power_a=args.power_a)
    else:
        if not args.train or not args.test:
            raise ValueError("give --preset, or at least one --train and one --test")
        spec = PoolSpec(
            train=tuple(parse_segment(text, args.power_a) for text in args.train),
            test=tuple(parse_segment(text, args.power_a) for text in args.test),
            vocab_size=args.vocab_size,
            seed=args.data_seed,
        )
    if args.random_non_queries is None:
        return spec
    return replace(spec, random_non_queries=bool(args.random_non_queries))


def train_config(
    args: argparse.Namespace, pool: Pool, lr: float, seed: int
) -> TrainConfig:
    """Assemble one cell's protocol.

    Two defaults come from the preset rather than from the parser, because they are part
    of the published protocol and not preferences: ``repro`` runs 32 epochs at batch
    256/32, while ``figure2`` and any ad-hoc pool run 64 epochs at the batch-size ladder
    keyed on the pool's longest sequence. An explicit flag overrides either.
    """
    repro = args.preset == "repro"
    batch = args.batch_size or (256 if repro else batch_size_for(pool.max_length))
    test_batch = args.test_batch_size or (32 if repro and not args.batch_size else 0)
    return TrainConfig(
        max_epochs=args.max_epochs or PROTOCOL_EPOCHS.get(args.preset, 64),
        batch_size=batch,
        test_batch_size=test_batch,
        lr=lr,
        weight_decay=args.weight_decay,
        early_stopping_threshold=args.early_stop,
        precision=args.precision,
        seed=seed,
        device=args.device,
    )


def model_config(args: argparse.Namespace, pool: Pool, d_model: int) -> ModelConfig:
    """Assemble one cell's model shape."""
    return ModelConfig(
        vocab_size=pool.vocab_size,
        d_model=d_model,
        n_layers=args.n_layers,
        max_length=pool.max_length,
        max_position_embeddings=pool.max_length if args.positions == "length" else 0,
        learnable_word_embeddings=not args.untied_embeddings,
        embedding_init_type=args.embedding_init,
        state_mixer=args.state_mixer,
        hidden_mult=args.hidden_mult,
        embed_dropout=args.embed_dropout,
        resid_dropout=args.resid_dropout,
        init_std=args.init_std,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every cell in the sweep.

    Args:
        argv: Command line, or None for ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    for spec in args.mixer_module:
        mixer_registry.load_module(spec)
    mixer = mixer_registry.resolve(args.mixer, args.settings)
    pool = build_pool(pool_spec(args), measure_leakage=not args.no_leakage_check)
    with contextlib.ExitStack() as stack:
        sink = (
            stack.enter_context(Path(args.out).open("a", encoding="utf-8"))
            if args.out
            else None
        )
        for seed in args.seed:
            for width in args.d_model:
                for lr in args.lr:
                    record = _run_cell(args, pool, mixer, width, lr, seed)
                    line = json.dumps(record, sort_keys=True)
                    print(line, flush=True)
                    if sink is not None:
                        sink.write(line + "\n")
                        sink.flush()
    return 0


def _run_cell(
    args: argparse.Namespace,
    pool: Pool,
    mixer: mixer_registry.Mixer,
    width: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    protocol = train_config(args, pool, lr, seed)
    seed_all(seed)
    shape = model_config(args, pool, width)
    model = LanguageModel(shape, mixer.factory)
    record: dict[str, Any] = {
        "task": "mqar",
        "mixer": mixer.name,
        "settings": mixer.settings,
        "d_model": width,
        "lr": lr,
        "seed": seed,
        "data_seed": int(args.data_seed),
        "parameters": parameter_count(model),
        "leaked": pool.leaked,
        "vocab_size": pool.vocab_size,
        "max_length": pool.max_length,
        "preset": args.preset,
        "random_non_queries": pool.random_non_queries,
        "train_segments": [_segment_record(segment) for segment in pool.train],
        "test_segments": [_segment_record(segment) for segment in pool.test],
        "steps_per_epoch": batch_count(pool.train, protocol.batch_size),
        "protocol": {
            "max_epochs": protocol.max_epochs,
            "batch_size": protocol.batch_size,
            "test_batch_size": protocol.eval_batch_size,
            "weight_decay": protocol.weight_decay,
            "early_stopping_threshold": protocol.early_stopping_threshold,
            "precision": protocol.precision,
            "state_mixer": shape.state_mixer,
            "n_layers": shape.n_layers,
            "max_position_embeddings": shape.max_position_embeddings,
            "embedding_init_type": shape.embedding_init_type,
            "learnable_word_embeddings": shape.learnable_word_embeddings,
        },
    }
    if args.dry_run:
        record["dry_run"] = True
        return record
    start = time.perf_counter()
    report = train(model, pool.train, pool.test, protocol)
    record["seconds"] = round(time.perf_counter() - start, 3)
    record.update(_report_record(report, include_points=args.points))
    return record


def _segment_record(segment: Segment) -> dict[str, Any]:
    return {
        "input_seq_len": segment.spec.input_seq_len,
        "num_kv_pairs": segment.spec.num_kv_pairs,
        "num_examples": segment.spec.num_examples,
        "power_a": segment.spec.power_a,
        "seed": segment.seed,
    }


def _report_record(report: Report, include_points: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "best": _metrics_record(report.best),
        "best_epoch": report.best_epoch,
        "final": _metrics_record(report.final),
        "epochs_run": report.epochs_run,
        "stopped_early": report.stopped_early,
    }
    if include_points:
        record["points"] = [
            {
                "epoch": point.epoch,
                "lr": point.lr,
                "train_loss": point.train_loss,
                "test": _metrics_record(point.test),
            }
            for point in report.points
        ]
    return record


def _metrics_record(metrics: Metrics) -> dict[str, Any]:
    return {
        "loss": metrics.loss,
        "example": metrics.example,
        "position": metrics.position,
        "by_slice": metrics.by_slice,
    }


if __name__ == "__main__":
    sys.exit(main())
