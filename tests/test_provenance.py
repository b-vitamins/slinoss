"""Replay identity shared by the experiment harnesses."""

from __future__ import annotations

from scripts.provenance import capture, identity


def test_json_identity_is_order_independent_and_value_sensitive() -> None:
    """Equivalent records hash together; a moved experimental value does not."""
    assert identity({"a": 1, "b": [2, 3]}) == identity({"b": [2, 3], "a": 1})
    assert identity({"a": 1}) != identity({"a": 2})


def test_capture_pins_source_harness_dirty_overlay_and_command() -> None:
    """A record carries both committed trees plus the exact working-tree overlay."""
    record = capture(
        "scripts/state_tracking",
        ["--task", "A5"],
        module="scripts.state_tracking.run",
    )
    assert record["source"]["path"] == "slinoss"
    assert record["harness"]["path"] == "scripts/state_tracking"
    assert record["source"]["commit"] == record["repository_commit"]
    assert record["harness"]["commit"] == record["repository_commit"]
    assert len(record["source"]["tree"]) == 40
    assert len(record["harness"]["tree"]) == 40
    assert len(record["dirty_diff_sha256"]) == 64
    assert record["command_argv"][-2:] == ["--task", "A5"]
    assert "scripts.state_tracking.run" in record["command"]
