"""Build the causal conv1d CUDA extension.

Package metadata lives in ``pyproject.toml``. This file exists only because the
extension needs ``torch.utils.cpp_extension``, which cannot be expressed there.

Build in place:

    python3 setup.py build_ext --inplace

The extension list is empty when torch is absent or when the toolchain has no
CUDA, so a CPU-only install still succeeds and the pure-PyTorch reference still
runs. :mod:`slinoss._C` is what reports the missing extension, at call time.

No architecture string is hardcoded. nvcc targets whatever
``TORCH_CUDA_ARCH_LIST`` or the detected devices ask for.
"""

from __future__ import annotations

from typing import Any

from setuptools import setup

MODULE = "slinoss._C._conv1d"
"""Import path of the compiled module. Mirrors ``slinoss._C.EXTENSION``."""

# Distinct stems. The object name is the source name with the extension
# replaced, so two sources called causal_conv1d in one directory collide.
#
# Relative, and passed through as written. ``build_ext --inplace`` accepts an
# absolute path but ``bdist_wheel`` refuses one, so absolutizing these against
# this file's parent leaves a tree that compiles in place and has no
# distribution -- and therefore no way to ship a payload from
# ``slinoss.aot.PAYLOAD_DIR``. Every build backend invokes this script with its
# own directory as the working directory, which is what these are relative to.
SOURCES = ["csrc/causal_conv1d.cpp", "csrc/causal_conv1d_kernel.cu"]

INCLUDE_DIRS = ["csrc"]
"""Header search path. Relative for the same reason as :data:`SOURCES`."""

CXX_FLAGS = ["-O3"]

# -lineinfo so the profiler can attribute a counter to a source line. No
# --use_fast_math: it would replace expf in the SiLU epilogue with a lower
# accuracy intrinsic and put the kernel and the reference on different footings.
NVCC_FLAGS = ["-O3", "-lineinfo"]


def _build() -> tuple[list[Any], dict[str, Any]]:
    """Extension modules and command classes for this environment.

    Returns:
        ``(ext_modules, cmdclass)``. Both empty when the extension cannot be
        built here.
    """
    try:
        from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    except ImportError:
        return [], {}
    if CUDA_HOME is None:
        return [], {}
    extension = CUDAExtension(
        name=MODULE,
        sources=SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = _build()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
