"""Persistent tuning database for the CuTe ``v2x2ssd`` forward stack."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from slinoss._cute_runtime import _default_cute_cache_dir

from .types import HardwareFingerprint

_DB_VERSION = 1


def _autotune_root() -> Path:
    root = os.environ.get("SLINOSS_CUTE_AUTOTUNE_CACHE_DIR")
    if root:
        path = Path(root).expanduser()
    else:
        path = _default_cute_cache_dir() / "autotune"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tuning_db_path() -> Path:
    """Return the persistent JSON database path for forward autotuning."""

    return _autotune_root() / "forward_tuning_db.json"


def load_cute_tuning_db() -> dict[str, Any]:
    """Load the persisted tuning DB, returning an empty schema on first use."""

    path = tuning_db_path()
    if not path.exists():
        return {"version": _DB_VERSION, "records": {}}
    payload = json.loads(path.read_text())
    if int(payload.get("version", -1)) != _DB_VERSION:
        raise ValueError(f"Unsupported tuning DB version in {path}.")
    records = payload.get("records")
    if not isinstance(records, dict):
        raise ValueError(f"Malformed tuning DB payload in {path}.")
    return payload


def save_cute_tuning_db(payload: dict[str, Any]) -> None:
    """Atomically persist the tuning DB to disk."""

    payload = dict(payload)
    payload["version"] = _DB_VERSION
    path = tuning_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _record_key(
    *,
    scope: str,
    hardware: HardwareFingerprint,
    problem_key: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "scope": str(scope),
            "hardware": hardware.to_record(),
            "problem": problem_key,
        },
        sort_keys=True,
    )


def lookup_tuning_record(
    *,
    scope: str,
    hardware: HardwareFingerprint,
    problem_key: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a persisted record for ``scope/problem_key`` if present."""

    payload = load_cute_tuning_db()
    records = payload.setdefault("records", {})
    record = records.get(
        _record_key(scope=scope, hardware=hardware, problem_key=problem_key)
    )
    if record is None or not isinstance(record, dict):
        return None
    return dict(record)


def store_tuning_record(
    *,
    scope: str,
    hardware: HardwareFingerprint,
    problem_key: dict[str, Any],
    config_record: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist one winning tuning record."""

    payload = load_cute_tuning_db()
    records = payload.setdefault("records", {})
    records[_record_key(scope=scope, hardware=hardware, problem_key=problem_key)] = {
        "config": dict(config_record),
        "metadata": {} if metadata is None else dict(metadata),
    }
    save_cute_tuning_db(payload)


__all__ = [
    "load_cute_tuning_db",
    "lookup_tuning_record",
    "save_cute_tuning_db",
    "store_tuning_record",
    "tuning_db_path",
]
