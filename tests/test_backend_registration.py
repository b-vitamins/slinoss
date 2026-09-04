"""Every kernel backend a host that can run it must have registered.

Registration is silent on failure. Each family calls ``_register_cute()`` at import
and that function returns rather than raises when the kernel import fails, so a tree
whose kernels do not import runs the reference under the kernel path's name. Nothing
in the tree held any family to registering: the shape of the guard was documented and
unenforced, and a reference fallback has already voided a whole measurement campaign
here.

The resolved name is asserted, not membership in ``names()``. Presence says a backend
object exists; resolution is what the public path calls. A backend registered at a
priority the reference outranks, or over a dtype the call does not carry, is present
and unreachable, and every dtype a family declares is asserted because a partial
declaration routes one call to the kernel and its neighbour to the reference.

The skip is on host capability alone -- a CUDA device, and the DSL importable -- never
on the registry. A guard derived from the registry converts the exact failure this
file exists to catch into a skip.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from slinoss import _C
from slinoss._precision import KERNEL_DTYPES, LOW_PRECISION_DTYPES
from slinoss._registry import Backend
from slinoss.ops.block import (
    rmsnorm_residual_resolve,
    rmsnorm_resolve,
    swiglu_resolve,
)
from slinoss.ops.conv import resolve as conv_resolve
from slinoss.ops.decode import resolve as decode_resolve
from slinoss.ops.mixer import resolve as mixer_resolve
from slinoss.ops.scanprep import resolve as scanprep_resolve
from slinoss.ops.so3ssd import resolve as so3ssd_resolve
from slinoss.ops.xent import resolve as xent_resolve

try:
    import cutlass.cute  # noqa: F401
except ImportError:
    DSL_PRESENT = False
else:
    DSL_PRESENT = True
"""Whether the DSL the kernels are written in is installed.

The one condition under which an unregistered CuTe backend is not a defect. Probed
by importing it, so a package that is present and broken counts as absent from the
DSL's side and present from the kernels' -- which is the case that must fail, and it
fails inside the family's own ``try`` instead.
"""

needs_kernels = pytest.mark.skipif(
    not (torch.cuda.is_available() and DSL_PRESENT),
    reason="host cannot run the CuTe kernels: no CUDA device or no DSL",
)

Resolve = Callable[[str | None, str, torch.dtype], Backend[Any, Any]]

CUTE_FAMILIES = [
    pytest.param(xent_resolve, KERNEL_DTYPES, id="cross_entropy"),
    pytest.param(decode_resolve, KERNEL_DTYPES, id="decode"),
    pytest.param(scanprep_resolve, KERNEL_DTYPES, id="scanprep"),
    pytest.param(mixer_resolve, KERNEL_DTYPES, id="mixer_tail"),
    pytest.param(rmsnorm_resolve, KERNEL_DTYPES, id="rmsnorm"),
    pytest.param(rmsnorm_residual_resolve, KERNEL_DTYPES, id="rmsnorm_residual"),
    pytest.param(swiglu_resolve, KERNEL_DTYPES, id="swiglu"),
    pytest.param(so3ssd_resolve, LOW_PRECISION_DTYPES, id="so3ssd"),
]
"""Every registry with a CuTe backend, against the dtype set that backend declares.

``so3ssd`` declares only the low-precision pair: its MMA atom is 16-bit, so float32
resolving to the reference there is the contract rather than a hole.
"""


@needs_kernels
@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.parametrize(("resolve", "dtypes"), CUTE_FAMILIES)
def test_a_capable_host_resolves_every_family_to_the_kernel_backend(
    resolve: Resolve, dtypes: tuple[torch.dtype, ...]
) -> None:
    """The shipped path is the kernel path on a host that can run kernels."""
    for dtype in dtypes:
        assert resolve(None, "cuda", dtype).name == "cute", (
            f"{dtype} resolved to the reference; the CuTe backend did not register"
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_cuda_host_resolves_the_convolution_to_the_compiled_extension() -> None:
    """The convolution's fallback is the one that has already voided a campaign.

    Its guard is the compiled extension rather than the DSL, and the extension is
    built from this tree: an unbuilt checkout is a build step not taken, not a host
    without a dependency, so it is asserted on every CUDA host rather than skipped
    around. tests/test_conv_cuda.py skips its whole module on the same condition.
    """
    assert _C.is_available(), (
        f"{_C.EXTENSION} is not built; run {_C.BUILD_COMMAND!r} from the repository "
        f"root. Every measurement on this tree is a reference measurement until it is"
    )
    for dtype in KERNEL_DTYPES:
        assert conv_resolve(None, "cuda", dtype).name == "native", (
            f"{dtype} resolved to the reference convolution"
        )
