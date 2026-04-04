#!/usr/bin/env python3
"""Build packaged forward AOT artifacts for the CuTe ``v2x2ssd`` backend."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_aot_build_helpers():
    from slinoss._cute_runtime import ensure_cute_runtime_env
    from slinoss.ops.v2x2ssd.cute.aot import (
        DEFAULT_FORWARD_AOT_SPECS,
        build_default_forward_aot_package,
    )

    return (
        ensure_cute_runtime_env,
        DEFAULT_FORWARD_AOT_SPECS,
        build_default_forward_aot_package,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, export, and package default forward AOT artifacts for the "
            "CuTe v2x2ssd backend."
        )
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("slinoss/ops/v2x2ssd/cute/aot"),
        help="Package directory that will receive manifest/runtime/artifact files.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep any existing packaged AOT files instead of rebuilding from scratch.",
    )
    parser.add_argument(
        "--list-default-specs",
        action="store_true",
        help="Print the default packaged forward AOT specs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        ensure_cute_runtime_env,
        default_forward_aot_specs,
        build_default_forward_aot_package,
    ) = _import_aot_build_helpers()
    if args.list_default_specs:
        for spec in default_forward_aot_specs:
            print(spec)
        return

    ensure_cute_runtime_env()
    exported = build_default_forward_aot_package(
        package_root=args.package_root,
        clean=not args.no_clean,
    )
    print(
        "built-forward-aot",
        args.package_root,
        "count",
        len(exported),
    )
    for module in exported:
        print(module.module_id, module.shared_library)


if __name__ == "__main__":
    main()
