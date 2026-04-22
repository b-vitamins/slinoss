from __future__ import annotations

import json
from pathlib import Path

from slinoss._wheel_aot import stage_cute_aot_bundle, stage_cute_aot_payload


def test_stage_cute_aot_payload_copies_manifest_and_dirs(
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
    stage_cute_aot_payload(source_root, package_root)

    assert json.loads((package_root / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "v2x2ssd_fwd"}]
    }
    assert (package_root / "artifacts" / "payload.json").read_text() == "{}"
    assert (package_root / "artifacts" / "payload.so").read_text() == "so"
    assert (package_root / "runtime" / "runtime.so").read_text() == "rt"


def test_stage_cute_aot_bundle_stages_known_payload_roots(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "bundle"
    for name, kind in (
        ("v2x2ssd", "v2x2ssd_fwd"),
        ("scanprep", "scanprep_fwd"),
        ("mixer", "mixer_tail_rowwise_fwd"),
        ("block", "ffn_norm_fwd"),
    ):
        payload_root = source_root / name
        artifact_dir = payload_root / "artifacts"
        runtime_dir = payload_root / "runtime"
        artifact_dir.mkdir(parents=True)
        runtime_dir.mkdir(parents=True)
        (payload_root / "manifest.json").write_text(
            json.dumps({"artifacts": [{"kind": kind}]})
        )
        (artifact_dir / "payload.so").write_text(name)
        (runtime_dir / "runtime.so").write_text(f"{name}-rt")

    package_roots = {
        "v2x2ssd": tmp_path / "pkg-v2",
        "scanprep": tmp_path / "pkg-scanprep",
        "mixer": tmp_path / "pkg-mixer",
        "block": tmp_path / "pkg-block",
    }
    stage_cute_aot_bundle(source_root, package_roots)

    assert json.loads((package_roots["v2x2ssd"] / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "v2x2ssd_fwd"}]
    }
    assert json.loads((package_roots["scanprep"] / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "scanprep_fwd"}]
    }
    assert json.loads((package_roots["mixer"] / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "mixer_tail_rowwise_fwd"}]
    }
    assert json.loads((package_roots["block"] / "manifest.json").read_text()) == {
        "artifacts": [{"kind": "ffn_norm_fwd"}]
    }
    assert (
        package_roots["v2x2ssd"] / "artifacts" / "payload.so"
    ).read_text() == "v2x2ssd"
    assert (
        package_roots["scanprep"] / "artifacts" / "payload.so"
    ).read_text() == "scanprep"
    assert (package_roots["mixer"] / "artifacts" / "payload.so").read_text() == "mixer"
    assert (package_roots["block"] / "artifacts" / "payload.so").read_text() == "block"
