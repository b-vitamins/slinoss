"""Extension build hooks for vendored CUDA operators."""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except Exception:
    BuildExtension = None  # type: ignore[assignment]
    CUDAExtension = None  # type: ignore[assignment]
    CUDA_HOME = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
CSRC_DIR = ROOT / "csrc" / "causal_conv1d"
CSRC_DIR_REL = CSRC_DIR.relative_to(ROOT)


def _want_cuda_extension() -> bool:
    if os.environ.get("SLINOSS_SKIP_CUDA_BUILD", "0") == "1":
        return False
    if BuildExtension is None or CUDAExtension is None:
        return False
    if CUDA_HOME is None:
        warnings.warn(
            "Skipping CUDA extension build: CUDA_HOME is not set. "
            "Set SLINOSS_SKIP_CUDA_BUILD=1 to silence this warning.",
            stacklevel=2,
        )
        return False
    return True


def _want_cute_aot() -> bool:
    return (
        os.environ.get(
            "SLINOSS_BUILD_CUTE_AOT",
            os.environ.get("SLINOSS_BUILD_CUTE_FORWARD_AOT", "0"),
        )
        == "1"
    )


def _cute_aot_payload_dir() -> Path | None:
    payload_dir = os.environ.get("SLINOSS_CUTE_AOT_PAYLOAD_DIR", "").strip()
    if not payload_dir:
        payload_dir = os.environ.get("SLINOSS_CUTE_FORWARD_AOT_PAYLOAD_DIR", "").strip()
    if not payload_dir:
        return None
    return Path(payload_dir)


def _build_cute_aot(package_root: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from slinoss._cute_runtime import ensure_cute_runtime_env
    from slinoss.ops.scanprep.cute.aot import (
        build_default_cute_aot_package as build_default_scanprep_cute_aot_package,
    )
    from slinoss.ops.v2x2ssd.cute.aot import (
        build_default_cute_aot_package as build_default_v2x2ssd_cute_aot_package,
    )

    ensure_cute_runtime_env()
    build_default_v2x2ssd_cute_aot_package(
        package_root=package_root / "v2x2ssd",
        clean=True,
    )
    build_default_scanprep_cute_aot_package(
        package_root=package_root / "scanprep",
        clean=True,
    )


def _stage_cute_aot_payload(source_root: Path, package_root: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from slinoss._wheel_aot import stage_cute_aot_payload

    shutil.rmtree(package_root / "artifacts", ignore_errors=True)
    shutil.rmtree(package_root / "runtime", ignore_errors=True)
    (package_root / "manifest.json").unlink(missing_ok=True)
    stage_cute_aot_payload(source_root, package_root)


def _stage_cute_aot_bundle(
    source_root: Path,
    package_roots: dict[str, Path],
) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from slinoss._wheel_aot import stage_cute_aot_bundle

    for package_root in package_roots.values():
        shutil.rmtree(package_root / "artifacts", ignore_errors=True)
        shutil.rmtree(package_root / "runtime", ignore_errors=True)
        (package_root / "manifest.json").unlink(missing_ok=True)
    stage_cute_aot_bundle(source_root, package_roots)


class BuildPyWithCuteAOT(_build_py):
    """Optionally package prebuilt CuTe AOT artifacts into wheels."""

    def run(self):
        super().run()
        package_roots = {
            "v2x2ssd": (
                Path(self.build_lib) / "slinoss" / "ops" / "v2x2ssd" / "cute" / "aot"
            ),
            "scanprep": (
                Path(self.build_lib) / "slinoss" / "ops" / "scanprep" / "cute" / "aot"
            ),
        }
        for package_root in package_roots.values():
            package_root.mkdir(parents=True, exist_ok=True)
        payload_dir = _cute_aot_payload_dir()
        if payload_dir is not None:
            if any(
                (payload_dir / name / "manifest.json").is_file()
                for name in package_roots
            ):
                _stage_cute_aot_bundle(payload_dir, package_roots)
            else:
                _stage_cute_aot_payload(payload_dir, package_roots["v2x2ssd"])
            return
        if _want_cute_aot():
            _build_cute_aot(Path(self.build_lib) / "slinoss" / "ops")


ext_modules = []
cmdclass: dict[str, Any] = {"build_py": BuildPyWithCuteAOT}

if _want_cuda_extension():
    assert CUDAExtension is not None
    assert BuildExtension is not None
    ext_modules.append(
        CUDAExtension(
            name="slinoss._C.cconv1d_cuda",
            sources=[
                str(CSRC_DIR_REL / "causal_conv1d.cpp"),
                str(CSRC_DIR_REL / "causal_conv1d_fwd.cu"),
                str(CSRC_DIR_REL / "causal_conv1d_bwd.cu"),
                str(CSRC_DIR_REL / "causal_conv1d_update.cu"),
            ],
            include_dirs=[str(CSRC_DIR_REL)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--threads",
                    "4",
                ],
            },
        )
    )
    cmdclass["build_ext"] = BuildExtension


setup(ext_modules=ext_modules, cmdclass=cmdclass)
