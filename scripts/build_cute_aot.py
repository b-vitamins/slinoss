#!/usr/bin/env python3
"""Build packaged CuTe AOT artifacts for the CuTe training stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_aot_build_helpers():
    from slinoss._cute_runtime import ensure_cute_runtime_env
    from slinoss.ops.scanprep.cute.aot import (
        build_default_backward_aot_package as build_default_scanprep_backward_aot_package,
    )
    from slinoss.ops.scanprep.cute.aot import (
        build_default_cute_aot_package as build_default_scanprep_cute_aot_package,
    )
    from slinoss.ops.scanprep.cute.aot import (
        build_default_forward_aot_package as build_default_scanprep_forward_aot_package,
    )
    from slinoss.ops.scanprep.cute.aot import (
        default_backward_aot_specs as default_scanprep_backward_aot_specs,
    )
    from slinoss.ops.scanprep.cute.aot import (
        default_forward_aot_specs as default_scanprep_forward_aot_specs,
    )
    from slinoss.ops.mixer.cute.aot import (
        build_default_backward_aot_package as build_default_mixer_backward_aot_package,
    )
    from slinoss.ops.mixer.cute.aot import (
        build_default_cute_aot_package as build_default_mixer_cute_aot_package,
    )
    from slinoss.ops.mixer.cute.aot import (
        build_default_forward_aot_package as build_default_mixer_forward_aot_package,
    )
    from slinoss.ops.mixer.cute.aot import (
        default_backward_aot_specs as default_mixer_backward_aot_specs,
    )
    from slinoss.ops.mixer.cute.aot import (
        default_forward_aot_specs as default_mixer_forward_aot_specs,
    )
    from slinoss.ops.v2x2ssd.cute.aot import (
        build_default_backward_aot_package,
        build_default_cute_aot_package,
        build_default_forward_aot_package,
        build_forward_aot_search_space_package,
        default_backward_aot_specs,
        default_forward_aot_specs,
        search_space_forward_aot_specs,
    )

    return (
        ensure_cute_runtime_env,
        default_forward_aot_specs,
        default_backward_aot_specs,
        search_space_forward_aot_specs,
        default_scanprep_forward_aot_specs,
        default_scanprep_backward_aot_specs,
        default_mixer_forward_aot_specs,
        default_mixer_backward_aot_specs,
        build_default_forward_aot_package,
        build_default_backward_aot_package,
        build_default_cute_aot_package,
        build_forward_aot_search_space_package,
        build_default_scanprep_forward_aot_package,
        build_default_scanprep_backward_aot_package,
        build_default_scanprep_cute_aot_package,
        build_default_mixer_forward_aot_package,
        build_default_mixer_backward_aot_package,
        build_default_mixer_cute_aot_package,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile, export, and package the curated CuTe AOT payload for the "
            "full CuTe training stack. By default this builds both forward and "
            "backward default payloads for v2x2ssd, scanprep, and mixer-tail rowwise."
        )
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("build/cute-aot"),
        help=(
            "Bundle directory that will receive sibling v2x2ssd/, scanprep/, and mixer/ "
            "payload trees."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep any existing packaged AOT files instead of rebuilding from scratch.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Build only the forward payload.",
    )
    parser.add_argument(
        "--backward-only",
        action="store_true",
        help="Build only the backward payload.",
    )
    parser.add_argument(
        "--search-space-forward",
        action="store_true",
        help="Build the full forward autotune search-space payload instead of forward defaults.",
    )
    parser.add_argument(
        "--list-default-forward-specs",
        action="store_true",
        help="Print the curated default forward AOT specs and exit.",
    )
    parser.add_argument(
        "--list-default-backward-specs",
        action="store_true",
        help="Print the curated default backward AOT specs and exit.",
    )
    parser.add_argument(
        "--list-search-space-forward-specs",
        action="store_true",
        help="Print the full forward autotune search-space specs and exit.",
    )
    args = parser.parse_args()
    if args.forward_only and args.backward_only:
        parser.error("--forward-only and --backward-only are mutually exclusive")
    return args


def main() -> None:
    args = _parse_args()
    (
        ensure_cute_runtime_env,
        default_v2_forward_aot_specs,
        default_v2_backward_aot_specs,
        search_space_v2_forward_aot_specs,
        default_scanprep_forward_aot_specs,
        default_scanprep_backward_aot_specs,
        default_mixer_forward_aot_specs,
        default_mixer_backward_aot_specs,
        build_default_v2_forward_aot_package,
        build_default_v2_backward_aot_package,
        build_default_v2_cute_aot_package,
        build_v2_forward_aot_search_space_package,
        build_default_scanprep_forward_aot_package,
        build_default_scanprep_backward_aot_package,
        build_default_scanprep_cute_aot_package,
        build_default_mixer_forward_aot_package,
        build_default_mixer_backward_aot_package,
        build_default_mixer_cute_aot_package,
    ) = _import_aot_build_helpers()
    if args.list_default_forward_specs:
        print("[v2x2ssd]")
        for spec in default_v2_forward_aot_specs():
            print(spec)
        print("[scanprep]")
        for spec in default_scanprep_forward_aot_specs():
            print(spec)
        print("[mixer]")
        for spec in default_mixer_forward_aot_specs():
            print(spec)
        return
    if args.list_default_backward_specs:
        print("[v2x2ssd]")
        for spec in default_v2_backward_aot_specs():
            print(spec)
        print("[scanprep]")
        for spec in default_scanprep_backward_aot_specs():
            print(spec)
        print("[mixer]")
        for spec in default_mixer_backward_aot_specs():
            print(spec)
        return
    if args.list_search_space_forward_specs:
        for spec in search_space_v2_forward_aot_specs():
            print(spec)
        return

    ensure_cute_runtime_env()
    v2_package_root = args.package_root / "v2x2ssd"
    scanprep_package_root = args.package_root / "scanprep"
    mixer_package_root = args.package_root / "mixer"
    if not args.no_clean:
        shutil.rmtree(args.package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(args.package_root / "runtime", ignore_errors=True)
        (args.package_root / "manifest.json").unlink(missing_ok=True)
    if args.forward_only:
        exported = []
        exported += (
            build_v2_forward_aot_search_space_package(
                package_root=v2_package_root,
                clean=not args.no_clean,
            )
            if args.search_space_forward
            else build_default_v2_forward_aot_package(
                package_root=v2_package_root,
                clean=not args.no_clean,
            )
        )
        exported += build_default_scanprep_forward_aot_package(
            package_root=scanprep_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_mixer_forward_aot_package(
            package_root=mixer_package_root,
            clean=not args.no_clean,
        )
    elif args.backward_only:
        exported = []
        exported += build_default_v2_backward_aot_package(
            package_root=v2_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_scanprep_backward_aot_package(
            package_root=scanprep_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_mixer_backward_aot_package(
            package_root=mixer_package_root,
            clean=not args.no_clean,
        )
    elif args.search_space_forward:
        exported = []
        exported += build_v2_forward_aot_search_space_package(
            package_root=v2_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_scanprep_forward_aot_package(
            package_root=scanprep_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_mixer_forward_aot_package(
            package_root=mixer_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_v2_backward_aot_package(
            package_root=v2_package_root,
            clean=False,
        )
        exported += build_default_scanprep_backward_aot_package(
            package_root=scanprep_package_root,
            clean=False,
        )
        exported += build_default_mixer_backward_aot_package(
            package_root=mixer_package_root,
            clean=False,
        )
    else:
        exported = []
        exported += build_default_v2_cute_aot_package(
            package_root=v2_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_scanprep_cute_aot_package(
            package_root=scanprep_package_root,
            clean=not args.no_clean,
        )
        exported += build_default_mixer_cute_aot_package(
            package_root=mixer_package_root,
            clean=not args.no_clean,
        )
    print("built-cute-aot", args.package_root, "count", len(exported))
    for module in exported:
        print(module.module_id, module.shared_library)


if __name__ == "__main__":
    main()
