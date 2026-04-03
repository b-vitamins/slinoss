from __future__ import annotations

import os
from pathlib import Path

from slinoss._cute_runtime import ensure_cute_runtime_env


def test_ensure_cute_runtime_env_sets_xdg_cache_dir_and_tvm_ffi(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUTE_DSL_ENABLE_TVM_FFI", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    ensure_cute_runtime_env()

    assert Path(tmp_path, "slinoss", "cute-dsl").is_dir()
    assert Path(os.environ["CUTE_DSL_CACHE_DIR"]) == Path(
        tmp_path, "slinoss", "cute-dsl"
    )
    assert os.environ["CUTE_DSL_ENABLE_TVM_FFI"] == "1"


def test_ensure_cute_runtime_env_preserves_explicit_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom-cute-cache"
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(custom))
    monkeypatch.setenv("CUTE_DSL_ENABLE_TVM_FFI", "0")

    ensure_cute_runtime_env()

    assert custom.is_dir()
    assert os.environ["CUTE_DSL_CACHE_DIR"] == str(custom)
    assert os.environ["CUTE_DSL_ENABLE_TVM_FFI"] == "0"
