from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping


def stage_cute_aot_payload(
    source_root: str | Path,
    package_root: str | Path,
) -> None:
    source_root = Path(source_root)
    package_root = Path(package_root)
    manifest = source_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing CuTe AOT manifest at {manifest}")

    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, package_root / "manifest.json")

    for subdir_name in ("artifacts", "runtime"):
        source_dir = source_root / subdir_name
        if not source_dir.exists():
            continue
        target_dir = package_root / subdir_name
        shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)


def stage_cute_forward_aot_payload(
    source_root: str | Path,
    package_root: str | Path,
) -> None:
    stage_cute_aot_payload(source_root, package_root)


def stage_cute_aot_bundle(
    source_root: str | Path,
    package_roots: Mapping[str, str | Path],
) -> None:
    source_root = Path(source_root)
    for name, package_root in package_roots.items():
        payload_root = source_root / name
        if not payload_root.is_dir():
            continue
        stage_cute_aot_payload(payload_root, package_root)


__all__ = [
    "stage_cute_aot_bundle",
    "stage_cute_aot_payload",
    "stage_cute_forward_aot_payload",
]
