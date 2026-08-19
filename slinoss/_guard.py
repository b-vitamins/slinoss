"""Host-side operand checks every kernel path shares. No operator owns these.

The layout contract is one rule for the whole repo: every kernel operand is a
contiguous tensor on a CUDA device. That rule is checked once, here, rather than
once per operator. Three operators disagreeing about the wording of the same
rejection is three chances for a caller to hit the disagreement.

Dtype policy is not here, because it legitimately differs: the scan's GEMM
operands have no float32 tensor-core path, while the mixer tail and the block
kernels run float32 natively. Each operator's guard module states its own.

Every check runs before the launch. A host pointer handed to a kernel faults
inside CUDA and leaves the context unusable for the rest of the process, and a
strided operand is either misread with no error or fails later inside an internal
reshape. Repacking instead of raising would be the staging copy the kernels exist
to avoid.

No CuTe DSL import. A reference path can use this.
"""

from __future__ import annotations

from torch import Tensor

__all__ = ["Named", "check_layout"]

Named = tuple[tuple[Tensor, str], ...]
"""Operands paired with the name to report them under."""


def check_layout(named: Named) -> None:
    """Check the layout contract on every operand in order.

    Args:
        named: ``(tensor, name)`` pairs. Order is the reporting order: an operand
            that violates both conditions is reported under the device one.

    Raises:
        ValueError: If any operand is off CUDA or not contiguous.
    """
    for tensor, name in named:
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
