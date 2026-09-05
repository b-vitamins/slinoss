"""The driver: what a record carries, and what the table reports.

The record is the only durable artefact of an arm, so a field missing from it is a
measurement that cannot be replayed, and a field the command line fails to wire is an arm
that ran at a default while its record says otherwise. Both are pinned here: the protocol in
the record is compared field for field against the config the flags spell, and both splits
are compared against their dataclass's own field list.

The table's ``tail`` column is the longest length band's accuracy. A mean over lengths 40 to
256 sits high while the tail is at chance, so the tail is the number the axis is read on and
it has to come from the last band, not the first.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest

from scripts.state_tracking.instances import SplitConfig
from scripts.state_tracking.run import build_parser, main, splits, table
from scripts.state_tracking.tasks import PDSSM_REGULAR_TASKS
from scripts.state_tracking.train import TrainConfig

TINY = [
    "--mixer",
    "conv",
    "--set",
    "d_conv=2",
    "--d-model",
    "16",
    "--n-layers",
    "1",
    "--dropout",
    "0.0",
    "--train-min-length",
    "3",
    "--train-max-length",
    "8",
    "--val-min-length",
    "8",
    "--val-max-length",
    "12",
    "--val-count",
    "8",
    "--num-steps",
    "2",
    "--batch-size",
    "4",
    "--print-steps",
    "1",
    "--early-stop",
    "2.0",
    "--band-width",
    "4",
    "--device",
    "cpu",
    "--quiet",
]
"""The protocol shrunk to run whole on the CPU. Every arm below is this plus its own flags."""

COLUMNS = ["task", "mixer", "seed", "acc", "tail", "loss", "steps", "solved", "params"]


def _run(extra: list[str], capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """Run ``main`` and return the records it wrote to stdout.

    Args:
        extra: Flags beyond :data:`TINY`.
        capsys: Capture fixture.

    Returns:
        One record per arm, parsed, in the order they were written.
    """
    assert main(TINY + extra) == 0
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line]


def test_defaults_are_the_published_protocol() -> None:
    """The flags' defaults are the protocol's, so a bare invocation is the published arm.

    The defaults live in two places -- :class:`TrainConfig` and the parser -- so they are
    equated rather than restated: one of the two moving alone is the failure mode.
    """
    args = build_parser().parse_args([])
    protocol = TrainConfig()
    assert args.profile == "pdssm-regular"
    assert args.task is None
    assert args.num_steps == protocol.num_steps
    assert args.batch_size == protocol.batch_size
    assert args.lr == protocol.lr
    assert args.final_lr == protocol.final_lr
    assert args.warmup_fraction == protocol.warmup_fraction
    assert args.wd_embedding == protocol.weight_decay_embedding
    assert args.wd_others == protocol.weight_decay_others
    assert args.early_stop == protocol.early_stop_threshold
    assert args.print_steps == protocol.print_steps
    assert args.accumulation_steps == protocol.accumulation_steps
    assert args.grad_clip == protocol.grad_clip
    assert args.precision == protocol.precision
    assert args.band_width == protocol.band_width
    assert args.seed == [protocol.seed]
    assert (args.d_model, args.n_layers, args.dropout, args.use_glu) == (
        128,
        2,
        0.01,
        False,
    )
    assert (args.train_min_length, args.train_max_length, args.train_count) == (
        3,
        40,
        None,
    )
    assert (args.val_min_length, args.val_max_length, args.val_count) == (40, 256, 8192)
    assert args.pad_to == 0


def test_splits_are_the_published_ranges_at_the_two_seeds() -> None:
    """Train unbounded over 3 to 40, validation bounded over 40 to 256 at ``2 * seed``.

    Every published bar on this axis is an evaluation strictly past the trained length, so
    the two ranges must not overlap: a validation split reaching down into the training
    range would report length generalization it never measured.
    """
    args = build_parser().parse_args([])
    train_split, val_split = splits(args, 3)
    assert train_split == SplitConfig(min_length=3, max_length=40, seed=3)
    assert val_split == SplitConfig(min_length=40, max_length=256, seed=6, count=8192)
    assert train_split.count is None
    assert val_split.min_length >= train_split.max_length
    padded = build_parser().parse_args(["--pad-to", "256", "--train-count", "512"])
    train_split, val_split = splits(padded, 0)
    assert train_split.pad_to == val_split.pad_to == 256
    assert train_split.count == 512


def test_every_flag_reaches_the_protocol(capsys: pytest.CaptureFixture[str]) -> None:
    """A record's protocol is the config its flags spell, field for field.

    Asserted at a non-default value on every field, so a flag the driver forgets to wire
    shows up here rather than as a silently different run. ``seed`` arrives through the
    per-arm replace and ``precision`` is fixed by ``choices``.
    """
    records = _run(
        [
            "--task",
            "parity",
            "--seed",
            "3",
            "--lr",
            "0.001",
            "--final-lr",
            "1e-06",
            "--warmup-fraction",
            "0.5",
            "--wd-embedding",
            "0.1",
            "--wd-others",
            "0.2",
            "--accumulation-steps",
            "2",
            "--grad-clip",
            "0.5",
        ],
        capsys,
    )
    assert len(records) == 1
    expected = TrainConfig(
        num_steps=2,
        batch_size=4,
        lr=0.001,
        final_lr=1e-06,
        warmup_fraction=0.5,
        weight_decay_embedding=0.1,
        weight_decay_others=0.2,
        early_stop_threshold=2.0,
        print_steps=1,
        accumulation_steps=2,
        grad_clip=0.5,
        band_width=4,
        seed=3,
        device="cpu",
    )
    assert records[0]["protocol"] == asdict(expected)
    assert records[0]["mixer_settings"] == {"d_conv": 2, "expand": 2.0}
    assert records[0]["train_split"]["seed"] == 3
    assert records[0]["val_split"]["seed"] == 6


def test_record_carries_every_field_a_replay_needs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One record per arm, JSON-clean, holding both splits whole and every evaluation.

    The splits are compared against :class:`SplitConfig`'s own field list, so a field added
    to a split without being recorded fails here. Both supervision modes are run, since the
    group half is what carries the group order.
    """
    records = _run(["--task", "parity"], capsys)
    records += _run(
        ["--profile", "walker-group-prefix", "--task", "A5"], capsys
    )
    assert [record["task"] for record in records] == ["parity", "A5"]
    split_fields = {field.name for field in fields(SplitConfig)}
    for record in records:
        assert set(record["train_split"]) == split_fields
        assert set(record["val_split"]) == split_fields
        assert json.loads(json.dumps(record)) == record
        assert record["steps_run"] == 2
        assert len(record["points"]) == 2
        assert all(len(point) == 5 for point in record["points"])
        assert record["parameters"] > record["mixer_parameters"] > 0
        assert record["best_step"] in [point[0] for point in record["points"]]
        assert record["solved"] is False
        assert record["mixer_contract"]["max_length_policy"] == "unused"
        assert record["benchmark_contract"]["fidelity"] == "source-exact"
        assert len(record["mixer_constructions"]) == record["model"]["n_layers"]
        assert record["lengths"]["training_ceiling"] == 8
        assert record["lengths"]["evaluation_ceiling"] == 12
        assert record["lengths"]["mixer_initialization_span"] is None
        assert record["seeds"] == {
            "model": 0,
            "train_data": 0,
            "evaluation_data": 0,
        }
        assert len(record["data"]["train"]["identity"]) == 64
        assert len(record["data"]["evaluation"]["identity"]) == 64
        provenance = record["provenance"]
        assert len(provenance["repository_commit"]) == 40
        assert len(provenance["source"]["tree"]) == 40
        assert len(provenance["harness"]["tree"]) == 40
        assert len(provenance["dirty_diff_sha256"]) == 64
        assert provenance["command_argv"]
    parity, group = records
    assert (parity["supervision"], parity["vocab_size"], parity["group_order"]) == (
        "last",
        3,
        None,
    )
    assert (group["supervision"], group["vocab_size"], group["group_order"]) == (
        "all",
        60,
        60,
    )
    assert parity["best"]["positions"] == 8
    assert group["best"]["positions"] > 8


def test_default_run_is_exactly_the_released_pdssm_regular_suite(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare run excludes the bracketed extension and every reconstructed group row."""
    records = _run([], capsys)
    assert tuple(record["task"] for record in records) == PDSSM_REGULAR_TASKS
    assert all(
        record["benchmark_contract"]["profile"] == "pdssm-regular"
        for record in records
    )


def test_asymmetric_group_record_carries_both_vocabularies_and_generators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The record cannot collapse a two-symbol input into its 60-state classifier."""
    record = _run(
        [
            "--profile",
            "pdssm-groups-reconstruction",
            "--task",
            "pdssm:A5:2",
        ],
        capsys,
    )[0]
    assert record["vocab_size"] is None
    assert (record["input_vocab_size"], record["output_vocab_size"]) == (2, 60)
    assert (record["model"]["input_vocab_size"], record["model"]["output_vocab_size"]) == (
        2,
        60,
    )
    contract = record["benchmark_contract"]
    assert contract["fidelity"] == "cross-release-reconstruction"
    assert contract["generator_labels"] == ["12340", "10324"]
    assert record["data"]["train"]["benchmark_contract"] == contract
    assert record["best"]["positions"] == 8


def test_the_mixer_is_sized_for_the_evaluation_not_the_training_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``max_length`` is the wider of the two splits, so nothing resizes mid-run.

    The axis trains short and evaluates long. A mixer sized at the train ceiling would
    refuse the first evaluation batch, or worse, quietly extend a positional table under
    it. Run on ``attention`` because it is the mixer here that holds such a table.
    """
    records = _run(
        ["--task", "parity", "--mixer", "attention", "--set", "n_heads=4"], capsys
    )
    assert records[0]["model"]["max_length"] == 12
    assert records[0]["train_split"]["max_length"] == 8
    assert records[0]["mixer_settings"] == {"n_heads": 4, "rotary": True}
    construction = records[0]["mixer_constructions"][0]
    assert construction["context"]["max_length_policy"] == "required"
    assert construction["context"]["max_length_consumed"] == 12


def test_out_file_holds_the_same_lines_as_stdout(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``--out`` appends the records it printed, so a killed run keeps its finished arms.

    Appends rather than truncates: a seed bank is filled by several invocations writing to
    one file.
    """
    out = tmp_path / "nested" / "records.jsonl"
    first = _run(["--task", "parity", "--out", str(out)], capsys)
    second = _run(["--task", "even_pairs", "--out", str(out)], capsys)
    lines = out.read_text().splitlines()
    assert [json.loads(line) for line in lines] == first + second


def test_table_rows_and_the_mean(capsys: pytest.CaptureFixture[str]) -> None:
    """One row per record, in order, and a mean row only when there are several.

    The mean is what a seed bank is read on, so a mean row printed for a single arm would
    be mistaken for one.
    """
    records = _run(["--task", "parity", "even_pairs"], capsys)
    lines = table(records).splitlines()
    assert lines[0].split() == COLUMNS
    assert len(lines) == 6
    for line, record in zip(lines[2:4], records):
        row = line.split()
        assert row[0] == record["task"]
        assert row[1] == record["mixer"]
        assert row[3] == f"{record['best']['accuracy']:.4f}"
        assert row[-1] == str(record["parameters"])
    mean = sum(record["best"]["accuracy"] for record in records) / 2
    assert lines[-1].startswith("mean")
    assert f"{mean:.4f}" in lines[-1]
    single = table(records[:1]).splitlines()
    assert len(single) == 3
    assert "mean" not in single[-1]


def test_table_tail_column_is_the_last_band(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``tail`` is the longest band, and an evaluation with no band reports zero.

    Forged onto a real record with two bands at different accuracies, because a real run's
    bands can agree and then the first and the last are indistinguishable. The no-band case
    is reachable from a validation split that scored nothing, which the table should not be
    the first thing to report.
    """
    record = _run(["--task", "parity"], capsys)[0]
    graded = copy.deepcopy(record)
    graded["best"]["bands"] = [
        {"low": 8, "high": 11, "positions": 4, "accuracy": 0.25},
        {"low": 12, "high": 15, "positions": 4, "accuracy": 0.75},
    ]
    assert table([graded]).splitlines()[2].split()[4] == "0.7500"
    empty = copy.deepcopy(record)
    empty["best"]["bands"] = []
    assert table([empty]).splitlines()[2].split()[4] == "0.0000"
