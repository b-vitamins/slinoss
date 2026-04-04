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


def _want_cute_forward_aot() -> bool:
    return os.environ.get("SLINOSS_BUILD_CUTE_FORWARD_AOT", "0") == "1"


def _cute_forward_aot_payload_dir() -> Path | None:
    payload_dir = os.environ.get("SLINOSS_CUTE_FORWARD_AOT_PAYLOAD_DIR", "").strip()
    if not payload_dir:
        return None
    return Path(payload_dir)


def _build_cute_forward_aot(package_root: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from slinoss._cute_runtime import ensure_cute_runtime_env
    from slinoss.ops.v2x2ssd.cute.aot import build_forward_aot_search_space_package

    ensure_cute_runtime_env()
    build_forward_aot_search_space_package(package_root=package_root, clean=True)


def _stage_cute_forward_aot_payload(source_root: Path, package_root: Path) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from slinoss._wheel_aot import stage_cute_forward_aot_payload

    shutil.rmtree(package_root / "artifacts", ignore_errors=True)
    shutil.rmtree(package_root / "runtime", ignore_errors=True)
    (package_root / "manifest.json").unlink(missing_ok=True)
    stage_cute_forward_aot_payload(source_root, package_root)


class BuildPyWithCuteForwardAOT(_build_py):
    """Optionally package prebuilt CuTe forward AOT artifacts into wheels."""

    def run(self):
        super().run()
        package_root = (
            Path(self.build_lib) / "slinoss" / "ops" / "v2x2ssd" / "cute" / "aot"
        )
        package_root.mkdir(parents=True, exist_ok=True)
        payload_dir = _cute_forward_aot_payload_dir()
        if payload_dir is not None:
            _stage_cute_forward_aot_payload(payload_dir, package_root)
            return
        if _want_cute_forward_aot():
            _build_cute_forward_aot(package_root)


ext_modules = []
cmdclass: dict[str, Any] = {"build_py": BuildPyWithCuteForwardAOT}

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
