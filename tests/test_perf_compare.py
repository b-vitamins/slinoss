from __future__ import annotations

import pytest

from slinoss.perf.compare import compare_budget_trees, rank_budget_deltas


def test_compare_budget_trees_and_rank_deltas() -> None:
    before = {
        "step": {"__stats__": {"mean_ms": 10.0}},
        "forward": {
            "__stats__": {"mean_ms": 4.0},
            "mixer": {"__stats__": {"mean_ms": 2.0}},
        },
        "backward": {
            "__stats__": {"mean_ms": 6.0},
            "v2x2ssd": {"__stats__": {"mean_ms": 3.0}},
        },
    }
    after = {
        "step": {"__stats__": {"mean_ms": 9.0}},
        "forward": {
            "__stats__": {"mean_ms": 3.5},
            "mixer": {"__stats__": {"mean_ms": 1.5}},
        },
        "backward": {
            "__stats__": {"mean_ms": 5.5},
            "v2x2ssd": {"__stats__": {"mean_ms": 3.2}},
        },
    }

    rows = compare_budget_trees(before, after)
    row_by_label = {str(row["label"]): row for row in rows}
    assert row_by_label["step"]["delta_ms"] == pytest.approx(-1.0)
    assert row_by_label["forward.mixer"]["delta_ms"] == pytest.approx(-0.5)
    assert row_by_label["backward.v2x2ssd"]["delta_ms"] == pytest.approx(0.2)

    ranked = rank_budget_deltas(rows, top_k=2)
    assert [str(row["label"]) for row in ranked["regressions"]] == ["backward.v2x2ssd"]
    improvement_labels = [str(row["label"]) for row in ranked["improvements"]]
    assert improvement_labels[0] == "step"
    assert len(improvement_labels) == 2
    assert improvement_labels[1] in {"forward", "backward", "forward.mixer"}
