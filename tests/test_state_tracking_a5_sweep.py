from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.state_tracking.a5_sweep import (
    BatchStream,
    SourceFile,
    load_csv,
    minimum_depths,
    source_url,
    split_rows,
    verify_file,
)


def _table(path: Path) -> None:
    path.write_text(
        "seed,input,target\n"
        "7,0 1 2,0 3 4\n"
        "8,2 1 0,2 5 5\n"
        "9,3 4 5,3 6 7\n"
        "10,5 4 3,5 8 9\n"
        "11,9 8 7,9 10 11\n",
        encoding="utf-8",
    )


def test_load_csv_and_verify_payload(tmp_path: Path) -> None:
    path = tmp_path / "A5=3.csv"
    _table(path)
    payload = path.read_bytes()
    source = SourceFile(hashlib.sha256(payload).hexdigest(), len(payload))
    verify_file(path, source)
    data = load_csv(path, 3, source)
    assert data.inputs.dtype == torch.uint8
    assert data.inputs.tolist() == [
        [0, 1, 2],
        [2, 1, 0],
        [3, 4, 5],
        [5, 4, 3],
        [9, 8, 7],
    ]
    assert data.targets.shape == (5, 3)

    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match=r"expected .* bytes"):
        verify_file(path, source)


def test_load_csv_rejects_schema_and_token_domain(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("input,target\n0 1,0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected header"):
        load_csv(path, 2)
    path.write_text("seed,input,target\n0,0 60,0 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_csv(path, 2)


def test_split_and_batch_stream_are_seeded() -> None:
    split = split_rows(10, 7)
    assert split.train.numel() == 8
    assert split.validation.numel() == 2
    assert set(split.train.tolist()).isdisjoint(split.validation.tolist())
    left = BatchStream(split.train, 3, 9)
    right = BatchStream(split.train, 3, 9)
    assert [next(left).tolist() for _ in range(5)] == [
        next(right).tolist() for _ in range(5)
    ]


def test_minimum_depth_uses_any_seed() -> None:
    records = [
        {"length": 3, "depth": 1, "solved": False},
        {"length": 3, "depth": 1, "solved": True},
        {"length": 3, "depth": 2, "solved": True},
        {"length": 4, "depth": 1, "solved": False},
    ]
    assert minimum_depths(records) == {3: 1, 4: None}


def test_source_url_is_revision_pinned() -> None:
    url = source_url(20)
    assert "/8f910f92e1c70455dcd9376f56032dfc55126188/" in url
    with pytest.raises(ValueError, match="no pinned"):
        source_url(21)
