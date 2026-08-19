"""Host-side operand checks every kernel path shares. No operator owns these.

The layout contract is two rules for the whole repo. Most operands are contiguous
tensors on a CUDA device. The exception is an operand that is one column band of a
wider tensor -- a slice of the fused input projection -- which is pitched rather
than contiguous. Both rules are checked here rather than once per operator. Three
operators disagreeing about the wording of the same rejection is three chances for
a caller to hit the disagreement.

Dtype policy is not here, because it legitimately differs: the scan's GEMM
operands have no float32 tensor-core path, while the mixer tail and the block
kernels run float32 natively. Each operator's guard module states its own set and
hands it in; only the loop over it is shared.

Every check runs before the launch. A host pointer handed to a kernel faults
inside CUDA and leaves the context unusable for the rest of the process, and a
strided operand is either misread with no error or fails later inside an internal
reshape. Repacking instead of raising would be the staging copy the kernels exist
to avoid.

No CuTe DSL import. A reference path can use this.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "ALIGN_BYTES",
    "PROJ_ALIGN",
    "SECTOR_BYTES",
    "Named",
    "check_dtypes",
    "check_layout",
    "check_pitched",
]

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

SECTOR_BYTES = 32
"""The device's memory sector. Alignment below this costs traffic, not
correctness.

A load is served in whole sectors. A band row that starts mid-sector spans one
more sector than its length needs, and the extra sector is fetched and discarded.
Alignment to :data:`ALIGN_BYTES` leaves that case reachable: at 2 bytes an element,
a pitch of 8 elements is 16 bytes, so every second row starts mid-sector. Measured
on sm_86 at ``3N`` 48 and 12 groups that is 5.0% more DRAM traffic than the same
kernel reading the same bytes from an aligned band, and no bandwidth counter shows
it, because the sectors are genuinely fetched.
"""

PROJ_ALIGN = SECTOR_BYTES // 2
"""Elements a band offset and a projection width are padded to.

Sixteen rather than ``SECTOR_BYTES // itemsize``, because a producer pads once at
construction and the same buffer is read at every activation dtype. Two bytes is
the narrowest element any kernel here takes, so its multiple covers the wider ones:
16 elements is 32 bytes at 2 and 64 at 4.
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


def check_dtypes(
    named: Named,
    allowed: tuple[torch.dtype, ...],
    label: str,
    unit: str = "call",
) -> torch.dtype:
    """Check a group against a dtype set, then against itself.

    Both halves are one rule everywhere: an operand outside the set has no kernel
    path, and a group that mixes dtypes would need more than one widening type
    inside the kernel. Only the set is per-operator, so the set is an argument and
    the loop is not.

    Args:
        named: ``(tensor, name)`` pairs. Order is the reporting order.
        allowed: Dtypes the caller's kernel path accepts.
        label: What ``allowed`` is called in the rejection, for a caller whose set
            is narrower than the repo's and whose reason for that is its own.
        unit: What the group spans. ``"call"`` when every operand of the call shares
            one dtype, ``"group"`` when a call carries two independent groups.

    Returns:
        The shared dtype.

    Raises:
        TypeError: If a dtype is outside ``allowed``, or if the group mixes dtypes.
    """
    for tensor, name in named:
        if tensor.dtype not in allowed:
            raise TypeError(f"{name} has dtype {tensor.dtype}; {label}: {allowed}")
    head, head_name = named[0]
    for tensor, name in named[1:]:
        if tensor.dtype is not head.dtype:
            raise TypeError(
                f"{name} is {tensor.dtype} and {head_name} is {head.dtype}; "
                f"one dtype per {unit}"
            )
    return head.dtype
