"""The command line: prepare a corpus, train an arm, absorb its zero-shot scores, print.

Four subcommands, one artifact each.

    prep    a corpus directory: two token files and a manifest with their digests
    size    a width per arm at a target non-embedding parameter count
    train   a checkpoint and a record
    merge   an lm-eval results file folded into a record
    table   the eight-column table, from records

A record is the unit. One JSON file per arm carrying what it was, what it cost, what it
scored, and which corpus it read; a table is a list of records and nothing else. Nothing is
recomputed at print time, so a table is reproducible from files that already exist and a row
cannot quietly come from a different run than the one next to it.

:func:`table` refuses a set of records whose corpus digest or precision differ. That is the
standing rule -- no bits-per-byte figure crosses hosts -- as a check rather than a habit.
Absolute values here will not equal any published table: the tokenizer is this harness's, so
only the ordering within a table is comparable.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from scripts.lm import corpus as corpus_mod
from scripts.lm import model as model_mod
from scripts.lm.checkpoint import save
from scripts.lm.data import Shard
from scripts.lm.groups import group_counts
from scripts.lm.mixers import REGISTRY
from scripts.lm.sizing import check_spread, size_arm
from scripts.lm.train import Step, TrainConfig, train
from scripts.provenance import capture as capture_provenance
from scripts.provenance import identity

__all__ = ["PRECISION", "TASKS", "Record", "main", "run_arm", "table"]

PRECISION = "bf16-autocast/fp32-state"
"""The one precision every arm runs at.

A named deviation from the protocol, which ran float32 throughout. Float32 here would take
the operator's reference path and measure a different program. Recorded per row so a table
cannot mix two.
"""

TASKS: tuple[tuple[str, str, str], ...] = (
    ("lambada_openai", "acc", "LAMBADA"),
    ("hellaswag", "acc_norm", "HellaSwag"),
    ("piqa", "acc", "PIQA"),
    ("arc_easy", "acc", "ARC-e"),
    ("arc_challenge", "acc_norm", "ARC-c"),
    ("winogrande", "acc", "WinoGrande"),
    ("openbookqa", "acc_norm", "OBQA"),
    ("boolq", "acc", "BoolQ"),
)
"""The eight columns: lm-eval task, its metric, and the header to print.

The metric per task is the one the literature reports for it, which is length-normalized
accuracy where the continuations differ in length and plain accuracy where they do not. Fixed
here so no row is read at a different metric than the row above it.
"""

RECORD_NAME = "record.json"
"""Record file name inside a run directory."""


@dataclass(frozen=True)
class Record:
    """One arm's row.

    Attributes:
        arm: Display name. Defaults to the mixer name; distinct arms of one mixer differ
            here.
        mixer: Registry name.
        mixer_settings: Settings the mixer was built at.
        mixer_contract: Declared context and initialization contract.
        mixer_constructions: Effective configuration of each constructed layer.
        hybrid_final: Registry name of the last layer's mixer, or None.
        d_model: Width.
        n_layers: Depth.
        parameters: Non-embedding trainable parameters. What arms are matched on.
        total_parameters: All trainable parameters, padding columns included.
        mixer_parameters: Trainable parameters inside the mixers.
        group_parameters: Trainable parameters per optimizer group.
        seq_len: Sequence length.
        token_batch: Tokens per optimizer step.
        token_budget: Tokens asked for.
        steps: Optimizer steps taken.
        tokens: Tokens consumed.
        peak_lr: Transferred hidden rate.
        embedding_lr: Transferred token-table rate.
        seed: Run seed.
        precision: :data:`PRECISION`.
        tokenizer: Tokenizer id.
        dataset: Dataset id.
        train_sha256: Digest of the training token file.
        val_sha256: Digest of the validation token file.
        val_loss: Held-out nats per token.
        val_bpb: Held-out bits per byte.
        train_loss: Trailing mean training loss.
        zero_shot: Accuracy per lm-eval task in ``[0, 1]``, or None before a merge.
    """

    arm: str
    mixer: str
    mixer_settings: dict[str, Any]
    mixer_contract: dict[str, Any]
    mixer_constructions: list[dict[str, Any]]
    hybrid_final: str | None
    d_model: int
    n_layers: int
    parameters: int
    total_parameters: int
    mixer_parameters: int
    group_parameters: dict[str, int]
    seq_len: int
    token_batch: int
    token_budget: int
    steps: int
    tokens: int
    peak_lr: float
    embedding_lr: float
    seed: int
    precision: str
    precision_details: dict[str, Any]
    tokenizer: str
    dataset: str
    train_sha256: str
    val_sha256: str
    val_loss: float | None
    val_bpb: float | None
    train_loss: float
    lengths: dict[str, Any]
    seeds: dict[str, int]
    data: dict[str, Any]
    initialization: dict[str, str]
    provenance: dict[str, Any]
    zero_shot: dict[str, float] | None = field(default=None)

    def average(self) -> float | None:
        """Mean accuracy over the tasks this record scored.

        Returns:
            The mean over :data:`TASKS` present in ``zero_shot``, or None when none are.
            Over the tasks present, not over the eight, so a partial evaluation reports a
            mean of what it ran rather than one silently dragged down by absences.
        """
        if not self.zero_shot:
            return None
        found = [self.zero_shot[task] for task, _, _ in TASKS if task in self.zero_shot]
        return sum(found) / len(found) if found else None


def read_record(path: Path) -> Record:
    """Read a record.

    Args:
        path: The JSON file.

    Returns:
        The record.

    Raises:
        FileNotFoundError: When the file is absent.
        TypeError: On a file missing a field or carrying an unknown one.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no record at {path}")
    return Record(**json.loads(path.read_text(encoding="utf-8")))


def write_record(path: Path, record: Record) -> None:
    """Write a record.

    Args:
        path: Destination JSON file. Its parent is created.
        record: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(record), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def scores_from(results: dict[str, Any]) -> dict[str, float]:
    """Pull the eight accuracies out of an lm-eval results file.

    lm-eval keys a metric by name and filter, as ``acc,none``. Both spellings are read, and
    a task whose metric is absent is left out rather than defaulted: a missing number is not
    a zero.

    Args:
        results: The parsed JSON, or its ``results`` mapping.

    Returns:
        Accuracy per task, in ``[0, 1]``.

    Raises:
        ValueError: When the file carries no ``results`` mapping.
    """
    table_ = results.get("results", results)
    if not isinstance(table_, dict):
        raise ValueError("results file carries no results mapping")
    scores: dict[str, float] = {}
    for task, metric, _ in TASKS:
        row = table_.get(task)
        if not isinstance(row, dict):
            continue
        for key in (f"{metric},none", metric):
            if key in row:
                scores[task] = float(row[key])
                break
    return scores


def _cell(value: float | None, width: int, scale: float = 100.0) -> str:
    """One right-aligned numeric cell.

    Args:
        value: The number, or None for an absent one.
        width: Field width.
        scale: Multiplier applied before formatting.

    Returns:
        The cell, or dashes when the value is absent or not finite.
    """
    if value is None or not math.isfinite(value):
        return "-".rjust(width)
    return f"{value * scale:.2f}".rjust(width)


def table(records: Sequence[Record]) -> str:
    """The eight-column zero-shot table.

    Args:
        records: One per arm, in the order to print.

    Returns:
        The table as text: arm, non-embedding parameters, held-out bits per byte, the eight
        accuracies in percent, and their mean.

    Raises:
        ValueError: On an empty sequence, on rows from different corpora, or on rows at
            different precisions. Either would make the columns incomparable, which is the
            one thing this function exists to prevent.
    """
    if not records:
        raise ValueError("table needs at least one record")
    for key in ("train_sha256", "val_sha256", "precision", "tokenizer"):
        found = {getattr(record, key) for record in records}
        if len(found) > 1:
            named = ", ".join(
                f"{record.arm}={getattr(record, key)}" for record in records
            )
            raise ValueError(f"rows differ in {key}: {named}")
    spread = check_spread({record.arm: record.parameters for record in records})
    arm_width = max(len("arm"), *(len(record.arm) for record in records))
    header = [
        "arm".ljust(arm_width),
        "params".rjust(9),
        "bpb".rjust(7),
        *(name.rjust(max(6, len(name))) for _, _, name in TASKS),
        "avg".rjust(6),
    ]
    lines = ["  ".join(header)]
    for record in records:
        row = [
            record.arm.ljust(arm_width),
            f"{record.parameters / 1e6:.1f}M".rjust(9),
            _cell(record.val_bpb, 7, scale=1.0),
            *(
                _cell(
                    None if not record.zero_shot else record.zero_shot.get(task),
                    max(6, len(name)),
                )
                for task, _, name in TASKS
            ),
            _cell(record.average(), 6),
        ]
        lines.append("  ".join(row))
    lines.append(
        f"\n{len(records)} arms at {records[0].tokens:,} tokens, "
        f"{records[0].precision}, parameter spread {spread:.4f}"
    )
    return "\n".join(lines)


def run_arm(
    *,
    out: Path,
    corpus_root: Path,
    mixer: str,
    overrides: Sequence[str],
    d_model: int,
    config: TrainConfig,
    device: str,
    arm: str | None = None,
    hybrid_final: str | None = None,
    hybrid_overrides: Sequence[str] = (),
    n_layers: int = 12,
    quiet: bool = False,
    provenance: dict[str, Any] | None = None,
) -> Record:
    """Train one arm and write its checkpoint and record.

    The seed is set once here, before the model is built, and never again: the data order is
    a function of ``(seed, epoch)`` inside :mod:`scripts.lm.data`, so the loop needs no
    process RNG and reseeding inside it would only decouple the two.

    Args:
        out: Run directory. Receives ``model.pt`` and ``record.json``.
        corpus_root: Corpus directory, with a manifest.
        mixer: Registry name.
        overrides: ``key=value`` mixer settings.
        d_model: Width. From :func:`scripts.lm.sizing.size_arm` or from the command line.
        config: The run.
        device: Where to run.
        arm: Display name, defaulting to the mixer name.
        hybrid_final: Registry name for the last layer only.
        hybrid_overrides: That mixer's settings.
        n_layers: Depth.
        quiet: Suppress progress lines.
        provenance: Source, harness, dirty tree, and exact command identity.

    Returns:
        The record, already written.

    Raises:
        KeyError: On an unregistered mixer.
        ValueError: From the corpus, the config, the sizing, or the group partition.
    """
    manifest = corpus_mod.read_manifest(corpus_root)
    resolved = REGISTRY.resolve(mixer, overrides)
    final = REGISTRY.resolve(hybrid_final, hybrid_overrides) if hybrid_final else None

    torch.manual_seed(config.seed)
    scaffold = model_mod.scaffold_config(
        d_model=d_model, n_layers=n_layers, vocab_size=manifest.vocab_size
    )
    stack = model_mod.build_model(
        scaffold,
        model_mod.layer_factories(
            resolved.factory, n_layers, None if final is None else final.factory
        ),
        max_length=config.seq_len,
        device=device,
        dtype=torch.float32,
    )
    constructions = [*resolved.constructions]
    if final is not None:
        constructions.extend(final.constructions)

    train_shard = Shard(
        corpus_mod.shard_path(corpus_root, "train"), manifest.train.tokens
    )
    val_shard = Shard(corpus_mod.shard_path(corpus_root, "val"), manifest.val.tokens)

    def report(step: Step) -> None:
        print(
            f"step {step.number:>7}  loss {step.loss:.4f}  "
            f"lr {step.lr:.3e}  |g| {step.grad_norm:.3f}",
            flush=True,
        )

    result = train(
        stack,
        train_shard,
        config,
        d_model=d_model,
        classes=manifest.vocab_size,
        device=device,
        val_shard=val_shard,
        bytes_per_token=manifest.val_bytes_per_token,
        on_step=None if quiet else report,
    )

    save(
        out / "model.pt",
        stack,
        config=scaffold,
        mixer=mixer,
        mixer_settings=resolved.settings,
        max_length=config.seq_len,
        step=result.steps,
        lr=result.peak_lr,
        embedding_lr=result.embedding_lr,
        seed=config.seed,
        manifest=manifest,
        hybrid_final=hybrid_final,
        hybrid_final_settings=None if final is None else final.settings,
    )
    data = corpus_mod.to_dict(manifest)
    record = Record(
        arm=arm or mixer,
        mixer=mixer,
        mixer_settings=resolved.settings,
        mixer_contract={
            "base_max_length_policy": resolved.max_length_policy,
            "hybrid_final_max_length_policy": (
                None if final is None else final.max_length_policy
            ),
            "initialization": "mixer_constructor; no scaffold reinitialization",
        },
        mixer_constructions=constructions,
        hybrid_final=hybrid_final,
        d_model=d_model,
        n_layers=n_layers,
        parameters=model_mod.non_embedding_parameters(stack),
        total_parameters=model_mod.parameter_count(stack),
        mixer_parameters=model_mod.mixer_parameters(stack),
        group_parameters=group_counts(stack),
        seq_len=config.seq_len,
        token_batch=config.token_batch,
        token_budget=config.token_budget,
        steps=result.steps,
        tokens=result.tokens,
        peak_lr=result.peak_lr,
        embedding_lr=result.embedding_lr,
        seed=config.seed,
        precision=PRECISION,
        precision_details={
            "parameter_dtype": str(next(stack.parameters()).dtype),
            "autocast_dtype": "torch.bfloat16",
            "recurrent_state_dtype": "torch.float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        tokenizer=manifest.tokenizer,
        dataset=manifest.dataset,
        train_sha256=manifest.train.digest,
        val_sha256=manifest.val.digest,
        val_loss=None if result.val is None else result.val.loss,
        val_bpb=None if result.val is None else result.val.bpb,
        train_loss=result.train_loss,
        lengths={
            "configured_task_length": config.seq_len,
            "training_ceiling": config.seq_len,
            "evaluation_ceiling": config.seq_len,
            "observed_tensor_width": {
                "train": config.seq_len,
                "validation": config.seq_len,
            },
            "mixer_initialization_span": [
                construction["effective_config"].get("init_span")
                for construction in constructions
                if "init_span" in construction["effective_config"]
            ]
            or None,
        },
        seeds={"model": config.seed, "data_order": config.seed},
        data={**data, "identity": identity(data)},
        initialization={
            "scaffold": "framework constructor defaults; no post-construction pass",
            "mixer": "owned by each mixer constructor",
        },
        provenance=(
            capture_provenance("scripts/lm", ["<python-api>"], module="scripts.lm.run")
            if provenance is None
            else provenance
        ),
    )
    write_record(out / RECORD_NAME, record)
    return record


def _parser() -> argparse.ArgumentParser:
    """The command line.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(prog="scripts.lm.run", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prep", help="tokenize a corpus")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    prep.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    prep.add_argument("--dataset-config", default="sample-10BT")
    prep.add_argument("--dataset-split", default="train")
    prep.add_argument("--text-field", default="text")
    prep.add_argument("--train-tokens", type=int, required=True)
    prep.add_argument("--val-tokens", type=int, default=1 << 24)

    size = sub.add_parser("size", help="solve a width per arm")
    size.add_argument("--target", type=int, required=True)
    size.add_argument("--mixer", action="append", required=True)
    size.add_argument("--n-layers", type=int, default=12)
    size.add_argument("--vocab-size", type=int, default=50432)
    size.add_argument("--seq-len", type=int, default=2048)

    run = sub.add_parser("train", help="train one arm")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--mixer", required=True)
    run.add_argument("--set", action="append", default=[], dest="overrides")
    run.add_argument("--arm")
    run.add_argument("--hybrid-final")
    run.add_argument(
        "--hybrid-set", action="append", default=[], dest="hybrid_overrides"
    )
    run.add_argument("--d-model", type=int)
    run.add_argument("--target-parameters", type=int)
    run.add_argument("--n-layers", type=int, default=12)
    run.add_argument("--seq-len", type=int, default=2048)
    run.add_argument("--micro-batch", type=int, default=8)
    run.add_argument("--token-batch", type=int, default=1 << 17)
    run.add_argument("--token-budget", type=int, required=True)
    run.add_argument("--base-lr", type=float, default=4e-3)
    run.add_argument("--embedding-base-lr", type=float, default=0.3)
    run.add_argument("--grad-clip", type=float, default=3.0)
    run.add_argument("--warmdown-fraction", type=float, default=0.4)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--eval-batch", type=int, default=8)
    run.add_argument("--log-every", type=int, default=50)
    run.add_argument("--device", default="cuda")
    run.add_argument("--quiet", action="store_true")

    merge = sub.add_parser("merge", help="fold lm-eval results into a record")
    merge.add_argument("--record", type=Path, required=True)
    merge.add_argument("--results", type=Path, required=True)

    show = sub.add_parser("table", help="print the table")
    show.add_argument("--record", type=Path, action="append", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand.

    Args:
        argv: Arguments, or None for ``sys.argv[1:]``.

    Returns:
        Process exit status.

    Raises:
        KeyError: On an unregistered mixer.
        ValueError: From any stage that refuses its input.
    """
    args = _parser().parse_args(argv)

    if args.command == "prep":
        manifest = corpus_mod.build(
            args.root,
            tokenizer=args.tokenizer,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            train_tokens=args.train_tokens,
            val_tokens=args.val_tokens,
        )
        print(
            f"train {manifest.train.tokens:,} tokens {manifest.train.digest[:12]}\n"
            f"val   {manifest.val.tokens:,} tokens {manifest.val.digest[:12]}\n"
            f"val bytes/token {manifest.val_bytes_per_token:.4f}"
        )
        return 0

    if args.command == "size":
        sizings = [
            size_arm(
                args.target,
                mixer,
                n_layers=args.n_layers,
                vocab_size=args.vocab_size,
                max_length=args.seq_len,
            )
            for mixer in args.mixer
        ]
        for sizing in sizings:
            print(
                f"{sizing.mixer:<10} d_model {sizing.d_model:>5}  "
                f"params {sizing.parameters:>12,}  miss {sizing.error:+.4f}"
            )
        spread = check_spread({sizing.mixer: sizing.parameters for sizing in sizings})
        print(f"spread {spread:.4f}")
        return 0

    if args.command == "train":
        if (args.d_model is None) == (args.target_parameters is None):
            raise ValueError("give exactly one of --d-model and --target-parameters")
        d_model = args.d_model
        if d_model is None:
            manifest = corpus_mod.read_manifest(args.corpus)
            d_model = size_arm(
                args.target_parameters,
                args.mixer,
                n_layers=args.n_layers,
                vocab_size=manifest.vocab_size,
                max_length=args.seq_len,
                overrides=args.overrides,
            ).d_model
        config = TrainConfig(
            token_budget=args.token_budget,
            token_batch=args.token_batch,
            seq_len=args.seq_len,
            micro_batch=args.micro_batch,
            base_lr=args.base_lr,
            embedding_base_lr=args.embedding_base_lr,
            grad_clip=args.grad_clip,
            warmdown_fraction=args.warmdown_fraction,
            seed=args.seed,
            eval_batch=args.eval_batch,
            log_every=args.log_every,
        )
        record = run_arm(
            out=args.out,
            corpus_root=args.corpus,
            mixer=args.mixer,
            overrides=args.overrides,
            d_model=d_model,
            config=config,
            device=args.device,
            arm=args.arm,
            hybrid_final=args.hybrid_final,
            hybrid_overrides=args.hybrid_overrides,
            n_layers=args.n_layers,
            quiet=args.quiet,
            provenance=capture_provenance("scripts/lm", argv, module="scripts.lm.run"),
        )
        line = f"{record.arm}  params {record.parameters:,}"
        if record.val_loss is not None:
            line += f"  val loss {record.val_loss:.4f}"
        if record.val_bpb is not None:
            line += f"  val bpb {record.val_bpb:.4f}"
        print(line)
        return 0

    if args.command == "merge":
        record = read_record(args.record)
        results = json.loads(args.results.read_text(encoding="utf-8"))
        merged = Record(**{**asdict(record), "zero_shot": scores_from(results)})
        write_record(args.record, merged)
        print(f"{merged.arm}: {len(merged.zero_shot or {})} tasks")
        return 0

    print(table([read_record(path) for path in args.record]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
