"""The record and the table: what is written, what is read back, and what is refused.

A record is the only thing that survives a run, and a table is a list of records. So the
failure modes here are all about a number arriving in a table without its provenance. A record
that does not round trip loses a field silently -- JSON has no schema and a missing key becomes
a default. A table that prints rows from two corpora is not a comparison and looks exactly like
one. A metric read at the wrong key silently becomes a zero, which reads as a broken arm rather
than as a broken parse.

The lm-eval side is pinned against its actual output shape, ``"acc,none"`` for a metric with no
filter, because that is the spelling this harness has to survive and the plain spelling is what
an older version wrote.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.lm.run import (
    PRECISION,
    TASKS,
    Record,
    read_record,
    scores_from,
    table,
    write_record,
)

DIGEST = "a" * 64
VAL_DIGEST = "b" * 64


def _record(arm: str, **overrides: Any) -> Record:
    """A record with every field filled, so a round trip exercises all of them."""
    base = Record(
        arm=arm,
        mixer=arm,
        mixer_settings={"d_state": 96, "expand": 2.0, "band_conv": True},
        hybrid_final=None,
        d_model=512,
        n_layers=12,
        parameters=45_000_000,
        total_parameters=71_000_000,
        mixer_parameters=12_000_000,
        group_parameters={"embedding": 26_000_000, "hidden": 18_000_000},
        seq_len=2048,
        token_batch=1 << 17,
        token_budget=1_800_000_000,
        steps=13732,
        tokens=1_799_819_264,
        peak_lr=1.5e-3,
        embedding_lr=0.11,
        seed=0,
        precision=PRECISION,
        tokenizer="EleutherAI/gpt-neox-20b",
        dataset="HuggingFaceFW/fineweb-edu",
        train_sha256=DIGEST,
        val_sha256=VAL_DIGEST,
        val_loss=3.1,
        val_bpb=0.92,
        train_loss=3.2,
    )
    return replace(base, **overrides)


def _scored(arm: str, **overrides: Any) -> Record:
    """A record with all eight tasks scored."""
    scores = {task: 0.25 + index / 100 for index, (task, _, _) in enumerate(TASKS)}
    return _record(arm, zero_shot=scores, **overrides)


def test_a_record_round_trips_through_json(tmp_path: Path) -> None:
    """Every field, including the nested settings and the scores.

    JSON has no schema, so a field lost on the way out becomes a default on the way in and the
    table prints a plausible row. The comparison is on the whole record for that reason.
    """
    written = _scored("slinoss")
    path = tmp_path / "record.json"
    write_record(path, written)
    assert read_record(path) == written
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["mixer_settings"]["d_state"] == 96
    assert set(raw["zero_shot"]) == {task for task, _, _ in TASKS}


def test_a_missing_record_is_not_an_empty_one(tmp_path: Path) -> None:
    """A table built from a path that does not exist must name the path."""
    with pytest.raises(FileNotFoundError, match="no record at"):
        read_record(tmp_path / "absent.json")


def test_a_record_with_an_unknown_field_is_refused(tmp_path: Path) -> None:
    """A record written by another version of this harness is not silently readable."""
    path = tmp_path / "record.json"
    write_record(path, _record("slinoss"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["invented"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TypeError):
        read_record(path)


def test_the_average_is_over_the_tasks_present_not_over_the_eight() -> None:
    """A partial evaluation reports the mean of what it ran.

    Averaging over eight with absences counted as zero would drag a partial row down and make
    it look like a bad arm instead of an incomplete one.
    """
    assert _record("slinoss").average() is None
    assert _record("slinoss", zero_shot={}).average() is None
    two = _record("slinoss", zero_shot={"piqa": 0.6, "boolq": 0.4})
    assert two.average() == pytest.approx(0.5)
    unknown = _record("slinoss", zero_shot={"piqa": 0.6, "not_a_task": 1.0})
    assert unknown.average() == pytest.approx(0.6)


def test_the_metric_is_read_at_both_spellings_and_absence_is_not_zero() -> None:
    """lm-eval writes ``acc,none``; older versions wrote ``acc``.

    A task whose metric is absent is left out. Defaulting to zero would put a real number in
    the table for an evaluation that did not happen.
    """
    results = {
        "results": {
            "piqa": {"acc,none": 0.61, "acc_stderr,none": 0.01},
            "boolq": {"acc": 0.55},
            "hellaswag": {"acc,none": 0.3},
            "arc_easy": {"acc_norm,none": 0.4},
        }
    }
    scores = scores_from(results)
    assert scores == {"piqa": 0.61, "boolq": 0.55}
    assert "arc_easy" not in scores
    assert "winogrande" not in scores


def test_the_normalized_metric_is_read_where_the_task_uses_it() -> None:
    """Length-normalized accuracy where the continuations differ in length, plain where not.

    Fixed per task in :data:`scripts.lm.run.TASKS` so no row is read at a different metric
    than the row above it.
    """
    metrics = {task: metric for task, metric, _ in TASKS}
    assert metrics["hellaswag"] == "acc_norm"
    assert metrics["arc_challenge"] == "acc_norm"
    assert metrics["openbookqa"] == "acc_norm"
    assert metrics["piqa"] == "acc"
    results = {
        "results": {
            "hellaswag": {"acc,none": 0.31, "acc_norm,none": 0.34},
            "piqa": {"acc,none": 0.61, "acc_norm,none": 0.62},
        }
    }
    assert scores_from(results) == {"hellaswag": 0.34, "piqa": 0.61}


def test_a_results_file_with_no_results_mapping_is_refused() -> None:
    """A failed lm-eval run writes something; it must not read as eight absences."""
    with pytest.raises(ValueError, match="no results mapping"):
        scores_from({"results": []})


def test_a_flat_results_mapping_is_accepted() -> None:
    """The ``results`` sub-mapping on its own, which is what a hand-extracted file carries."""
    assert scores_from({"piqa": {"acc,none": 0.61}}) == {"piqa": 0.61}


def test_the_table_refuses_rows_from_different_corpora() -> None:
    """The standing rule as a check: no bits-per-byte figure crosses corpora or hosts.

    Both digests are checked, because an arm trained on one training shard and scored on
    another validation shard is the more likely accident.
    """
    rows = [_scored("slinoss"), _scored("mamba2", train_sha256="c" * 64)]
    with pytest.raises(ValueError, match="rows differ in train_sha256"):
        table(rows)
    rows = [_scored("slinoss"), _scored("mamba2", val_sha256="c" * 64)]
    with pytest.raises(ValueError, match="rows differ in val_sha256"):
        table(rows)


def test_the_table_refuses_rows_at_different_precisions_or_tokenizers() -> None:
    """A row at float32 and a row at bf16 are two programs; a table names one precision."""
    rows = [_scored("slinoss"), _scored("mamba2", precision="fp32")]
    with pytest.raises(ValueError, match="rows differ in precision"):
        table(rows)
    rows = [_scored("slinoss"), _scored("mamba2", tokenizer="gpt2")]
    with pytest.raises(ValueError, match="rows differ in tokenizer"):
        table(rows)


def test_the_table_refuses_rows_that_are_not_the_same_size() -> None:
    """Sizing is enforced at print time as well as at solve time.

    The record carries the achieved count, so a table is the last place a mis-sized arm can be
    caught before it is read as a result.
    """
    rows = [_scored("slinoss"), _scored("mamba2", parameters=60_000_000)]
    with pytest.raises(ValueError, match="parameter spread"):
        table(rows)


def test_the_table_prints_a_row_per_arm_with_the_eight_headers() -> None:
    """The shape of the output, and that an unscored row prints dashes rather than zeros."""
    rows = [_scored("slinoss"), _record("mamba2", parameters=45_100_000)]
    text = table(rows)
    lines = text.splitlines()
    assert all(name in lines[0] for _, _, name in TASKS)
    assert lines[1].startswith("slinoss")
    assert lines[2].startswith("mamba2")
    assert lines[2].count("-") >= len(TASKS)
    assert "parameter spread" in lines[-1]
    assert "1,799,819,264 tokens" in lines[-1]


def test_the_table_needs_a_record() -> None:
    """An empty table would print a header and read as a run with no arms."""
    with pytest.raises(ValueError, match="at least one record"):
        table([])


def test_an_absent_or_non_finite_number_prints_as_a_dash() -> None:
    """A NaN loss from a diverged run must not print as a number."""
    text = table([_record("slinoss", val_bpb=float("nan"))])
    assert text.splitlines()[1].count("-") >= len(TASKS) + 1
