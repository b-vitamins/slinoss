from __future__ import annotations

import shutil
from pathlib import Path


def stage_cute_forward_aot_payload(
    source_root: str | Path,
    package_root: str | Path,
) -> None:
    source_root = Path(source_root)
    package_root = Path(package_root)
    manifest = source_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing forward AOT manifest at {manifest}")

    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, package_root / "manifest.json")

    for subdir_name in ("artifacts", "runtime"):
        source_dir = source_root / subdir_name
        if not source_dir.exists():
            continue
        target_dir = package_root / subdir_name
        shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)


__all__ = ["stage_cute_forward_aot_payload"]
