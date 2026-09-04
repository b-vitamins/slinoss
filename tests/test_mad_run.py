"""The driver: what a record carries, and that the whole path runs.

A record is the only durable trace of an arm, so the test that matters is the one that
runs the driver end to end and reads the record back: task settings, mixer settings,
protocol, width, leakage, parameter count, every evaluation. A field missing from it is an
arm that cannot be reproduced, which is the failure the whole harness exists to prevent.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from scripts.mad.run import _literal, _spec, main, parse_axes, table

RECORD_KEYS = {
    "task",
    "mad_task",
    "task_settings",
    "task_contract",
    "mixer",
    "mixer_settings",
    "mixer_contract",
    "mixer_constructions",
    "model",
    "protocol",
    "selection",
    "width",
    "lengths",
    "seeds",
    "pool",
    "initialization",
    "provenance",
    "leakage",
    "leaky",
    "parameters",
    "best",
    "best_epoch",
    "epoch_indexing",
    "final",
    "epochs_run",
    "stopped_early",
    "points",
}
"""Everything an arm is. A record short of these is not reproducible."""


def test_a_command_line_value_takes_its_narrowest_type() -> None:
    """Integers stay integers, so a width does not arrive as a float and reshape."""
    assert _literal("128") == 128 and isinstance(_literal("128"), int)
    assert _literal("0.25") == 0.25
    assert _literal("true") is True
    assert _literal("False") is False
    assert _literal("last") == "last"


def test_axes_are_read_as_a_mapping() -> None:
    """``--axis seq_len=256 frac_noise=0.4`` is two settings, not one string."""
    assert parse_axes(["seq_len=256", "frac_noise=0.4"]) == {
        "seq_len": 256,
        "frac_noise": 0.4,
    }
    with pytest.raises(ValueError, match="key=value"):
        parse_axes(["seq_len"])


def test_an_unknown_task_names_the_six() -> None:
    """A typo costs nothing when the message lists what is available."""
    with pytest.raises(KeyError, match="icr"):
        _spec("recall", {})


def test_a_spec_carries_the_axis_it_was_asked_for() -> None:
    """The axis reaches the spec, and a baseline is left untouched."""
    assert _spec("icr", {"seq_len": 256}).seq_len == 256
    assert _spec("icr", {}) is _spec("icr", {})


def test_the_table_flags_a_leaked_pool() -> None:
    """A leak invalidates the arm, so it appears next to the number it invalidates."""
    record: dict[str, Any] = {
        "task": "comp",
        "mixer": "conv",
        "protocol": {"seed": 0},
        "best": {"micro": 0.5, "macro": 0.5, "loss": 1.0},
        "epochs_run": 10,
        "parameters": 1234,
        "leaky": True,
    }
    rendered = table([record])
    assert "comp" in rendered and "conv" in rendered
    assert "leaked pools" in rendered
    assert "leaked pools" not in table([{**record, "leaky": False}])


def test_the_driver_runs_both_backbones_and_records_them() -> None:
    """One invocation, a causal task and the bottleneck task, records read back.

    Small enough to be a unit test and wide enough to cover the whole path: two task
    specs off their baselines, a registered mixer with settings, both model backbones, the
    protocol, the JSON line on stdout and the appended file, and the exit status.
    """
    out = io.StringIO()
    err = io.StringIO()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "records.jsonl"
        argv = [
            "--task",
            "icr",
            "comp",
            "--mixer",
            "conv",
            "--set",
            "d_conv=3",
            "--axis",
            "seq_len=16",
            "num_train=64",
            "num_test=32",
            "--seed",
            "0",
            "--d-model",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--eval-every",
            "1",
            "--device",
            "cpu",
            "--out",
            str(path),
            "--quiet",
        ]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main(argv)
        written = path.read_text().splitlines()

    assert status == 0
    records = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [json.loads(line) for line in written] == records
    assert [record["task"] for record in records] == ["icr", "comp"]
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record["mixer"] == "conv"
        assert record["mixer_settings"] == {"d_conv": 3, "expand": 2.0}
        assert record["task_settings"]["seq_len"] == 16
        assert record["task_settings"]["num_train"] == 64
        assert record["protocol"]["seed"] == 0
        assert record["protocol"]["device"] == "cpu"
        assert record["protocol"]["eval_every"] == 1
        assert record["protocol"]["drop_last"] is True
        assert record["protocol"]["float32_matmul_precision"] == "high"
        assert record["model"]["d_model"] == 16
        assert record["model"]["task_length"] == 16
        assert record["parameters"] > 0
        assert record["leakage"] == 0.0
        assert record["leaky"] is False
        assert record["epochs_run"] == 1
        assert len(record["points"]) == 1
        assert record["points"][0][0] == 1
        assert record["best_epoch"] == 1
        assert record["epoch_indexing"] == "one_based"
        assert record["selection"] == {
            "split": "test",
            "metric": "micro_accuracy",
            "evaluation_interval_epochs": 1,
        }
        assert record["seeds"] == {"model": 0, "shuffle": 0, "data": 0}
        assert len(record["pool"]["identity"]) == 64
        assert len(record["pool"]["spec_identity"]) == 64
        assert record["mixer_contract"]["max_length_policy"] == "unused"
        construction = record["mixer_constructions"][0]
        assert construction["context"] == {
            "max_length_supplied": 16,
            "max_length_policy": "unused",
            "max_length_consumed": None,
        }
        assert record["provenance"]["command_argv"][-1] == "--quiet"
        assert "--eval-every" in record["provenance"]["command_argv"]
        assert len(record["provenance"]["dirty_diff_sha256"]) == 64
    icr, comp = records
    # The copy prefix spends a position, so multi-query recall is a position short.
    assert (icr["width"], comp["width"]) == (15, 16)
    assert (icr["model"]["observed_width"], comp["model"]["observed_width"]) == (
        15,
        16,
    )
    assert icr["lengths"]["configured_task_length"] == 16
    assert icr["lengths"]["observed_tensor_width"] == {
        "train": 15,
        "evaluation": 15,
    }
    assert icr["task_contract"]["split_policy"] == "required"
    assert comp["task_contract"]["split_policy"] == "invariant"
    assert icr["model"]["bottleneck"] is False
    assert comp["model"]["bottleneck"] is True
    assert "conv" in err.getvalue()
