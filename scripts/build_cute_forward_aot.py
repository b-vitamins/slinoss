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
        build_forward_aot_search_space_package,
        default_forward_aot_specs,
    )

    return (
        ensure_cute_runtime_env,
        default_forward_aot_specs,
        build_forward_aot_search_space_package,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, export, and package the full forward autotune AOT search space for the "
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
        help="Print the packaged forward AOT search-space specs and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    (
        ensure_cute_runtime_env,
        default_forward_aot_specs_fn,
        build_forward_aot_search_space_package,
    ) = _import_aot_build_helpers()
    if args.list_default_specs:
        for spec in default_forward_aot_specs_fn():
            print(spec)
        return

    ensure_cute_runtime_env()
    exported = build_forward_aot_search_space_package(
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
