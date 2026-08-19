"""Host-side operand checks for the scan kernels.

Every check runs before the launch. A host pointer handed to a kernel faults
inside CUDA and leaves the context unusable for the rest of the process, and a
strided operand is either misread with no error or fails later inside an internal
reshape. Repacking instead of raising would be the staging copy the kernels exist
to avoid.

One implementation rather than one per kernel. Two kernels of the same operator
that disagree about what a legal call is have a caller that can reach the
disagreement, and the check order is part of the contract: layout, then dtype,
then shape, then extent. A mutation that violates two at once must be reported by
the first check it reaches.

The layout half of that contract is repo-wide, and so is the loop that checks a
dtype group, so both come from :mod:`slinoss._guard` and are re-exported here: a
scan kernel imports one guard module, not two. Only the dtype sets below are this
operator's own.
"""

from __future__ import annotations

import torch
from torch import Tensor

from slinoss._guard import Named, check_dtypes, check_layout
from slinoss._precision import LOW_PRECISION_DTYPES
from slinoss.ops.so3ssd.cute.mma import MMA_TILE_K, MMA_TILE_N

__all__ = [
    "OPERAND_DTYPES",
    "Named",
    "check_dtypes",
    "check_extents",
    "check_layout",
    "check_operands",
    "check_pinned",
    "check_rows",
    "check_shapes",
    "check_stream",
]

OPERAND_DTYPES: tuple[torch.dtype, ...] = LOW_PRECISION_DTYPES
"""Activation dtypes with a tensor-core path. The atom is 16-bit times 16-bit into
float32, so a float32 activation resolves to the reference backend rather than
being downcast behind the caller. Narrower than
:data:`slinoss._precision.KERNEL_DTYPES`, which is what a rowwise kernel takes."""


def check_operands(named: Named) -> torch.dtype:
    """The activation dtype a GEMM kernel of this operator was called at.

    Args:
        named: ``(tensor, name)`` pairs holding the activations.

    Returns:
        The shared activation dtype.

    Raises:
        TypeError: If an activation dtype has no tensor-core path, or if the
            activations do not share one dtype.
    """
    return check_dtypes(named, OPERAND_DTYPES, "tensor-core operand dtypes")


def check_pinned(named: Named) -> None:
    """Raises:
    ValueError: If a float32-pinned operand is not float32 (I4).
    """
    for tensor, name in named:
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32 (I4), got {tensor.dtype}")


def check_shapes(
    rowwise: Tensor,
    trans: Tensor,
    K: Tensor | None,
    *vectors: tuple[Tensor, str],
    label: str = "U",
) -> tuple[int, int, int, int, int, int]:
    """Check the per-token operands against the ``(B,H,T,P)`` one.

    ``G`` is read off the first vector rather than passed in. It is a property of
    the operands, so a caller cannot claim one grouping and hand over another, and
    ``G == H`` needs no separate signature.

    Args:
        rowwise: ``(B,H,T,P)``. Sets the leading shape. ``U`` on a forward path and
            its cotangent on a backward one.
        trans: ``(B,H,T,4)``.
        K: ``(B,H,T,2,4)``, or None for a kernel that reads no tap. A kernel with
            no tap matrix to build does not take ``K``, so there is nothing to
            check.
        vectors: ``(tensor, name)`` pairs, each ``(B,G,T,3N)``. The first one sets
            ``G`` and ``3N``.
        label: Name of ``rowwise`` in a rejection.

    Returns:
        ``(B, H, G, T, P, 3N)``.

    Raises:
        ValueError: On a rank or shape mismatch, or a ``G`` that does not divide
            ``H``. A group holds a whole number of heads: with a remainder some
            head would index past the group axis.
    """
    if rowwise.ndim != 4:
        raise ValueError(f"{label} must be (B,H,T,P), got {tuple(rowwise.shape)}")
    bsz, heads, seqlen, rows = (int(d) for d in rowwise.shape)
    lead = (bsz, heads, seqlen)
    if tuple(trans.shape) != (*lead, 4):
        raise ValueError(f"trans must be {(*lead, 4)}, got {tuple(trans.shape)}")
    if K is not None and tuple(K.shape) != (*lead, 2, 4):
        raise ValueError(f"K must be {(*lead, 2, 4)}, got {tuple(K.shape)}")
    head, head_name = vectors[0]
    if head.ndim != 4 or int(head.shape[0]) != bsz or int(head.shape[2]) != seqlen:
        raise ValueError(
            f"{head_name} must be (B,G,T,3N) with B={bsz} and T={seqlen}, "
            f"got {tuple(head.shape)}"
        )
    groups = int(head.shape[1])
    if groups < 1 or heads % groups != 0:
        raise ValueError(
            f"{head_name} carries G={groups}, which does not divide H={heads}"
        )
    glead = (bsz, groups, seqlen)
    dim = int(head.shape[3])
    for tensor, name in vectors[1:]:
        if tuple(tensor.shape) != (*glead, dim):
            raise ValueError(
                f"{name} must be {(*glead, dim)}, got {tuple(tensor.shape)}"
            )
    return bsz, heads, groups, seqlen, rows, dim


def check_extents(chunk_size: int, dim: int, slice_size: int) -> None:
    """Raises:
    ValueError: If ``L``, ``3N``, or the K slice is an extent the atom cannot
        cover. The fix for any of these is the shape, never a padding path.
    """
    if chunk_size < MMA_TILE_K or chunk_size % MMA_TILE_K != 0:
        raise ValueError(
            f"chunk_size must be a positive multiple of {MMA_TILE_K}, got {chunk_size}"
        )
    if chunk_size % slice_size != 0:
        raise ValueError(
            f"chunk_size {chunk_size} is not a multiple of its K slice {slice_size}"
        )
    if dim % 3 != 0 or dim % MMA_TILE_N != 0:
        raise ValueError(f"3N must be a multiple of 3 and of {MMA_TILE_N}, got {dim}")


def check_rows(rows: int) -> None:
    """Raises:
    ValueError: If ``P`` is an N extent the atom cannot cover. Only kernels that
        read out over ``P`` need this; where ``P`` is the M mode instead it is free.
    """
    if rows % MMA_TILE_N != 0:
        raise ValueError(f"P must be a multiple of {MMA_TILE_N}, got {rows}")


def check_stream(
    u_prev: Tensor | None,
    b_prev: Tensor | None,
    shape: tuple[int, int, int, int, int],
) -> bool:
    """Check the streaming carry-in pair.

    ``b_prev`` is a time slice of ``B``, so it is grouped exactly as ``B`` is.

    Args:
        u_prev: ``(B,H,P)`` or ``None``.
        b_prev: ``(B,G,3N)`` or ``None``.
        shape: ``(B, H, G, P, 3N)``.

    Returns:
        Whether the pair was supplied.

    Raises:
        ValueError: If one tap is supplied without the other, or on a shape
            mismatch. The two are read at the same token, so a call carrying only
            one would pair it with a zero and return a wrong answer with no error.
    """
    if (u_prev is None) != (b_prev is None):
        raise ValueError("u_prev and b_prev are supplied together or not at all")
    if u_prev is None or b_prev is None:
        return False
    bsz, heads, groups, rows, dim = shape
    if tuple(u_prev.shape) != (bsz, heads, rows):
        raise ValueError(
            f"u_prev must be {(bsz, heads, rows)}, got {tuple(u_prev.shape)}"
        )
    if tuple(b_prev.shape) != (bsz, groups, dim):
        raise ValueError(
            f"b_prev must be {(bsz, groups, dim)}, got {tuple(b_prev.shape)}"
        )
    return True
