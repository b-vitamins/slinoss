from __future__ import annotations

import json
from pathlib import Path

from slinoss._wheel_aot import stage_cute_forward_aot_payload


def test_stage_cute_forward_aot_payload_copies_manifest_and_dirs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    artifact_dir = source_root / "artifacts"
    runtime_dir = source_root / "runtime"
    artifact_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    (source_root / "manifest.json").write_text(
        json.dumps({"artifacts": [{"kind": "v2x2ssd_fwd"}]})
    )
    (artifact_dir / "payload.json").write_text("{}")
    (artifact_dir / "payload.so").write_text("so")
    (runtime_dir / "runtime.so").write_text("rt")

    package_root = tmp_path / "package"
    stage_cute_forward_aot_payload(source_root, package_root)

    assert json.loads((package_root / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "v2x2ssd_fwd"}]
    }
    assert (package_root / "artifacts" / "payload.json").read_text() == "{}"
    assert (package_root / "artifacts" / "payload.so").read_text() == "so"
    assert (package_root / "runtime" / "runtime.so").read_text() == "rt"
