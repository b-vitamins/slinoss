"""The driver: argument parsing, the preset protocols, and the record.

The record is the only thing that survives a run, so what it carries is a contract: a cell
whose settings are not in its record cannot be compared to anything later. The preset
protocols are the other half -- epoch count and batch size come from the published config
rather than from the parser, and an explicit flag has to override them.

The end-to-end case at the bottom runs one epoch on eight examples. It is not a
measurement; it is the only way to catch a driver that assembles a pool and a model
correctly and then fails to connect them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.mqar.run import (
    build_parser,
    main,
    model_config,
    parse_cell,
    parse_segment,
    pool_spec,
    train_config,
)
from scripts.mqar.tasks import Pool

TINY: tuple[str, ...] = (
    "--train",
    "16:2:8",
    "--test",
    "16:2:4",
    "--vocab-size",
    "64",
    "--mixer",
    "conv",
    "--d-model",
    "16",
    "--device",
    "cpu",
)
"""The smallest pool and model the driver accepts. Eight train rows, one batch."""


def parsed(argv: list[str]) -> argparse.Namespace:
    """Parse a command line through the driver's own parser."""
    return build_parser().parse_args(argv)


def pool_at(max_length: int, vocab_size: int = 8192) -> Pool:
    """A pool carrying only what the config assemblers read off one."""
    return Pool(
        train=(),
        test=(),
        vocab_size=vocab_size,
        max_length=max_length,
        random_non_queries=True,
    )


def records(captured: str) -> list[dict[str, Any]]:
    """Parse the driver's stdout, one JSON record per line."""
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("text", "expected"), [("64:4", (64, 4)), ("1024:256", (1024, 256))]
)
def test_parse_cell(text: str, expected: tuple[int, int]) -> None:
    """``LEN:KV``."""
    assert parse_cell(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [("64", "not LEN:KV"), ("64:4:8", "not LEN:KV"), ("64:x", "non-integer")],
)
def test_parse_cell_rejects_a_malformed_cell(text: str, message: str) -> None:
    """A cell that silently read as something else would run the wrong shape."""
    with pytest.raises(ValueError, match=message):
        parse_cell(text)


def test_parse_segment_carries_the_exponent() -> None:
    """``LEN:KV:N`` at the pool's ``power_a``, which is not part of the syntax."""
    spec = parse_segment("128:8:2000", power_a=0.5)
    assert (spec.input_seq_len, spec.num_kv_pairs, spec.num_examples) == (128, 8, 2000)
    assert spec.power_a == 0.5
    assert parse_segment("128:8:2000").power_a == 0.01


@pytest.mark.parametrize(
    ("text", "message"),
    [("16:2", "not LEN:KV:N"), ("16:2:8:1", "not LEN:KV:N"), ("16:2:n", "non-integer")],
)
def test_parse_segment_rejects_a_malformed_segment(text: str, message: str) -> None:
    """Caught before a pool is generated."""
    with pytest.raises(ValueError, match=message):
        parse_segment(text)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--preset", "repro", "--train", "16:2:8"], "mutually exclusive"),
        (["--cell", "64:4"], "--cell belongs to --preset figure2"),
        (["--preset", "figure2"], "needs --cell"),
        (["--preset", "repro", "--vocab-size", "64"], "fixes the vocabulary"),
        ([], "give --preset"),
        (["--train", "16:2:8"], "give --preset"),
        (["--test", "16:2:4"], "give --preset"),
    ],
)
def test_pool_spec_rejects_an_unreadable_request(argv: list[str], message: str) -> None:
    """A flag a preset ignores is refused rather than dropped.

    ``--vocab-size`` under a preset is the case that matters: the preset fixes 8192, and a
    run that accepted 64 and then trained at 8192 would be unreadable from its record.
    """
    with pytest.raises(ValueError, match=message):
        pool_spec(parsed(argv))


def test_presets_reach_their_published_pools() -> None:
    """One cell for figure 2, the five-and-seven ladder for the repro."""
    figure2 = pool_spec(parsed(["--preset", "figure2", "--cell", "128:8"]))
    assert len(figure2.train) == 1
    assert len(figure2.test) == 1
    assert figure2.train[0].input_seq_len == 128
    assert figure2.train[0].num_kv_pairs == 8
    assert figure2.vocab_size == 8192
    repro = pool_spec(parsed(["--preset", "repro"]))
    assert (len(repro.train), len(repro.test)) == (5, 7)
    assert repro.max_length == 1024


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--preset", "repro"], True),
        (["--preset", "repro", "--no-random-non-queries"], False),
        (["--preset", "figure2", "--cell", "64:4"], False),
        (["--preset", "figure2", "--cell", "64:4", "--random-non-queries"], True),
    ],
)
def test_the_filler_flag_is_three_valued(argv: list[str], expected: bool) -> None:
    """Absent means the preset's, and the two presets disagree.

    The flag is the only way to run either task variant against the other's protocol, and
    the default has to be the preset's or a repro number silently changes task.
    """
    assert pool_spec(parsed(argv)).random_non_queries is expected


def test_the_ad_hoc_pool_takes_the_flags_it_is_given() -> None:
    """Vocabulary, exponent and data seed all reach the spec; the filler defaults on."""
    spec = pool_spec(parsed([*TINY, "--power-a", "0.5", "--data-seed", "7"]))
    assert spec.vocab_size == 64
    assert spec.seed == 7
    assert spec.random_non_queries is True
    assert {segment.power_a for segment in spec.train + spec.test} == {0.5}


def test_repro_runs_its_own_epoch_count_and_split_batch() -> None:
    """32 epochs at batch 256 with an eighth of that for evaluation. Not the ladder."""
    config = train_config(parsed(["--preset", "repro"]), pool_at(1024), 1e-3, 5)
    assert config.max_epochs == 32
    assert config.batch_size == 256
    assert config.eval_batch_size == 32
    assert (config.lr, config.seed) == (1e-3, 5)


def test_an_explicit_train_batch_drops_the_repros_split() -> None:
    """The split is 256/32; at any other train batch the eval batch matches it again.

    Carrying 32 over to a hand-picked train batch would be an eighth of a number nobody
    chose.
    """
    config = train_config(
        parsed(["--preset", "repro", "--batch-size", "128"]), pool_at(1024), 1e-3, 5
    )
    assert config.batch_size == 128
    assert config.eval_batch_size == 128
    split = train_config(
        parsed(["--preset", "repro", "--batch-size", "128", "--test-batch-size", "64"]),
        pool_at(1024),
        1e-3,
        5,
    )
    assert split.eval_batch_size == 64


@pytest.mark.parametrize(
    ("argv", "max_length", "epochs", "batch"),
    [
        (["--preset", "figure2", "--cell", "256:16"], 256, 64, 256),
        (["--preset", "figure2", "--cell", "64:4"], 64, 64, 512),
        (list(TINY), 16, 64, 512),
        ([*TINY, "--max-epochs", "3", "--batch-size", "4"], 16, 3, 4),
    ],
)
def test_figure2_and_ad_hoc_pools_run_the_ladder(
    argv: list[str], max_length: int, epochs: int, batch: int
) -> None:
    """64 epochs, batch off :func:`scripts.mqar.tasks.batch_size_for`, eval batch matched."""
    config = train_config(parsed(argv), pool_at(max_length), 1e-3, 5)
    assert config.max_epochs == epochs
    assert config.batch_size == batch
    assert config.eval_batch_size == batch


def test_the_protocol_flags_reach_the_protocol() -> None:
    """Everything the loop reads that is not a preset decision."""
    config = train_config(
        parsed(
            [
                *TINY,
                "--weight-decay",
                "0.05",
                "--early-stop",
                "1.0",
                "--precision",
                "bf16",
            ]
        ),
        pool_at(16, vocab_size=64),
        1e-2,
        11,
    )
    assert config.weight_decay == 0.05
    assert config.early_stopping_threshold == 1.0
    assert config.precision == "bf16"
    assert config.device == "cpu"


def test_positions_are_off_unless_asked_for() -> None:
    """Figure 2 gives attention a position table and gives nothing else one.

    ``length`` sizes it to the pool's longest sequence, which is the test pool's length
    under the repro protocol, not the train pool's.
    """
    off = model_config(parsed(list(TINY)), pool_at(256), 128)
    assert off.max_position_embeddings == 0
    on = model_config(parsed([*TINY, "--positions", "length"]), pool_at(256), 128)
    assert on.max_position_embeddings == 256
    assert on.max_length == 256


def test_untied_embeddings_reach_the_model_shape() -> None:
    """The scaffold ablation is a flag, not an edit."""
    tied = model_config(parsed(list(TINY)), pool_at(16, vocab_size=64), 32)
    assert tied.learnable_word_embeddings is True
    untied = model_config(
        parsed([*TINY, "--untied-embeddings"]), pool_at(16, vocab_size=64), 32
    )
    assert untied.learnable_word_embeddings is False


def test_the_word_embedding_draw_is_a_flag() -> None:
    """The repro variant with the filler off runs ``spherical``; everything else default.

    It is the second of the two settings that separate the two published repro configs, so
    a run that took one without the other would be neither of them.
    """
    pool = pool_at(16, vocab_size=64)
    assert model_config(parsed(list(TINY)), pool, 32).embedding_init_type == "default"
    spherical = model_config(parsed([*TINY, "--embedding-init", "spherical"]), pool, 32)
    assert spherical.embedding_init_type == "spherical"


def test_a_dry_run_reports_the_whole_cell_and_trains_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every setting that decides what a number means, before any of it is spent.

    The sweep is the product of ``--seed``, ``--d-model`` and ``--lr`` in that nesting, and
    a record carries its own coordinates, so the records need no external ordering to be
    read.
    """
    argv = [*TINY, "--d-model", "16", "32", "--lr", "1e-3", "1e-2", "--dry-run"]
    assert main(argv) == 0
    emitted = records(capsys.readouterr().out)
    assert [(record["d_model"], record["lr"]) for record in emitted] == [
        (16, 1e-3),
        (16, 1e-2),
        (32, 1e-3),
        (32, 1e-2),
    ]
    first = emitted[0]
    assert first["task"] == "mqar"
    assert first["mixer"] == "conv"
    assert first["settings"] == {"conv": {"kernel_size": 3}}
    assert first["mixer_contracts"] == {
        "conv": {"layer_index_policy": "unused", "max_length_policy": "unused"}
    }
    assert len(first["mixer_constructions"]) == 2
    assert all(
        construction["context"]["max_length_consumed"] is None
        for construction in first["mixer_constructions"]
    )
    assert first["dry_run"] is True
    assert first["preset"] is None
    assert first["random_non_queries"] is True
    assert first["vocab_size"] == 64
    assert first["max_length"] == 16
    assert first["lengths"]["training_ceiling"] == 16
    assert first["lengths"]["evaluation_ceiling"] == 16
    assert first["lengths"]["mixer_initialization_span"] is None
    assert first["seeds"] == {"model": 123, "data": 123}
    assert len(first["data"]["identity"]) == 64
    assert first["precision"]["parameter_dtype"] == "torch.float32"
    provenance = first["provenance"]
    assert len(provenance["repository_commit"]) == 40
    assert len(provenance["source"]["tree"]) == 40
    assert len(provenance["harness"]["tree"]) == 40
    assert len(provenance["dirty_diff_sha256"]) == 64
    assert provenance["command_argv"]
    assert first["leaked"] == 0.0
    assert first["parameters"] > 0
    assert first["steps_per_epoch"] == 1
    assert first["protocol"]["max_epochs"] == 64
    assert first["protocol"]["state_mixer"] == "identity"
    assert first["protocol"]["embedding_init_type"] == "default"
    assert first["protocol"]["learnable_word_embeddings"] is True
    assert first["train_segments"][0]["num_examples"] == 8
    assert first["train_segments"][0]["seed"] != first["test_segments"][0]["seed"]
    assert "best" not in first
    assert "seconds" not in first


def test_out_appends_rather_than_truncates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sweep is run in pieces across hosts, so a second invocation must not erase one."""
    out = tmp_path / "records.jsonl"
    for _ in range(2):
        assert main([*TINY, "--dry-run", "--out", str(out)]) == 0
    capsys.readouterr()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["task"] == "mqar" for line in lines)


def test_one_epoch_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """The driver, the pool, the model and the loop, connected.

    One epoch on eight examples with the convolution control. Not a measurement: what it
    catches is a driver that assembles every part correctly and then hands the loop the
    wrong one.
    """
    argv = [
        *TINY,
        "--max-epochs",
        "1",
        "--lr",
        "1e-3",
        "--early-stop",
        "1.0",
        "--points",
    ]
    assert main(argv) == 0
    record = records(capsys.readouterr().out)[0]
    assert record["epochs_run"] == 1
    assert record["stopped_early"] is False
    assert record["best_epoch"] == 0
    assert record["seconds"] >= 0.0
    assert record["best"]["example"] == record["final"]["example"]
    assert record["final"]["by_slice"]["num_kv_pairs"] == {
        "2": record["final"]["example"]
    }
    assert len(record["points"]) == 1
    assert record["points"][0]["lr"] == 1e-3
    assert record["points"][0]["train_loss"] > 0.0
