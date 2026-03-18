"""Extension build hooks for vendored CUDA operators."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from setuptools import setup

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


ext_modules = []
cmdclass: dict[str, object] = {}

if _want_cuda_extension():
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
