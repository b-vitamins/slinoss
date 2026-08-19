"""Host-side operand checks every kernel path shares. No operator owns these.

The layout contract is two rules for the whole repo. Most operands are contiguous
tensors on a CUDA device. The exception is an operand that is one column band of a
wider tensor -- a slice of the fused input projection -- which is pitched rather
than contiguous. Both rules are checked here rather than once per operator. Three
operators disagreeing about the wording of the same rejection is three chances for
a caller to hit the disagreement.

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

__all__ = ["ALIGN_BYTES", "Named", "check_layout", "check_pitched"]

Named = tuple[tuple[Tensor, str], ...]
"""Operands paired with the name to report them under."""

ALIGN_BYTES = 16
"""Byte alignment every kernel operand is declared to carry.

The CuTe views are built with ``assumed_align=16``, which is a claim about the
address a load starts from. A contiguous allocation satisfies it at the base. A
column band of a wider tensor satisfies it only if the band's offset and the pitch
between one row and the next both land on the boundary, which is the producer's
job: it pads the column offsets it hands out.
"""


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


def check_pitched(named: Named) -> None:
    """Check the pitched-layout contract on every operand in order.

    A pitched operand is one column band of a wider tensor: unit stride along the
    trailing axis, an arbitrary pitch from one row to the next, and no constraint
    on the leading modes, which the kernels index through a dynamic layout. That is
    what a slice of one fused projection output is, so demanding contiguity would
    force either a projection per consumer or a staging copy.

    Args:
        named: ``(tensor, name)`` pairs. Order is the reporting order: an operand
            that violates two conditions is reported under the first.

    Raises:
        ValueError: If any operand is off CUDA, has no row axis, has a strided
            trailing axis, has rows that overlap, or starts or steps on an address
            :data:`ALIGN_BYTES` does not cover.
    """
    for tensor, name in named:
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA device, got {tensor.device}")
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have a row axis, got {tuple(tensor.shape)}")
        width, pitch = int(tensor.shape[-1]), int(tensor.stride(-2))
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have unit stride on its trailing axis, "
                f"got {tensor.stride(-1)}"
            )
        # A pitch under the row width means two rows share elements, which is what
        # an expanded or transposed view looks like from here.
        if pitch < width:
            raise ValueError(
                f"{name} rows overlap: pitch {pitch} is below the row width {width}"
            )
        multiple = ALIGN_BYTES // tensor.element_size()
        if tensor.data_ptr() % ALIGN_BYTES != 0 or pitch % multiple != 0:
            raise ValueError(
                f"{name} must start and step on a multiple of {multiple} elements; "
                f"got byte offset {tensor.data_ptr() % ALIGN_BYTES} and pitch {pitch}"
            )
