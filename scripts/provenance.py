"""Fail-visible provenance shared by the experiment harnesses.

The numerical record is the experiment.  A score without the exact source tree,
harness tree, command, and dirty state is only a note about a machine that once
ran.  This module records those facts without host names or environment values.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> bytes:
    """Run one read-only Git query at ``root`` and return its stdout."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _root(start: Path | None = None) -> Path:
    """Resolve the enclosing Git worktree."""
    cwd = Path.cwd() if start is None else start
    return Path(_git(cwd, "rev-parse", "--show-toplevel").decode().strip())


def _dirty_digest(root: Path) -> tuple[bool, str, list[str]]:
    """Hash every tracked patch and every non-ignored untracked input.

    Ignored data caches and local source mirrors are intentionally outside this
    identity.  A non-ignored untracked module is not: its path and contents enter
    the digest, so importing it cannot leave the record claiming a clean source.
    """
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    changed = sorted(
        line[3:].decode(errors="surrogateescape")
        for line in status.splitlines()
        if line
    )
    digest = hashlib.sha256()
    digest.update(_git(root, "diff", "--binary", "HEAD", "--"))
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    )
    for raw in sorted(path for path in untracked if path):
        relative = raw.decode(errors="surrogateescape")
        path = root / relative
        digest.update(b"untracked\0" + raw + b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode(errors="surrogateescape"))
        elif path.is_file():
            digest.update(path.read_bytes())
    return bool(status), digest.hexdigest(), changed


def identity(payload: Any) -> str:
    """Stable SHA-256 identity for a JSON-clean experiment input."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture(
    harness_path: str,
    argv: Sequence[str] | None,
    *,
    module: str,
    source_path: str = "slinoss",
) -> dict[str, Any]:
    """Capture replay-critical source and invocation provenance.

    ``commit`` names the complete checked-out repository.  The two tree object IDs
    independently pin the library and harness subtrees; ``dirty_diff_sha256`` adds
    any working-tree overlay.  ``argv`` is recorded both structurally and in a
    shell-escaped display string.
    """
    import shlex

    root = _root()
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    source_tree = _git(root, "rev-parse", f"HEAD:{source_path}").decode().strip()
    harness_tree = _git(root, "rev-parse", f"HEAD:{harness_path}").decode().strip()
    dirty, dirty_digest, dirty_files = _dirty_digest(root)
    command_argv = (
        list(sys.argv) if argv is None else [sys.executable, "-m", module, *list(argv)]
    )
    return {
        "repository_commit": head,
        "source": {"path": source_path, "commit": head, "tree": source_tree},
        "harness": {"path": harness_path, "commit": head, "tree": harness_tree},
        "dirty": dirty,
        "dirty_files": dirty_files,
        "dirty_diff_sha256": dirty_digest,
        "cwd": str(Path.cwd()),
        "command_argv": command_argv,
        "command": shlex.join(command_argv),
    }
