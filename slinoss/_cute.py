"""CuTe DSL shims and shared-memory policy. No operator owns these.

Every operator's ``cute`` package needs the same handful of things: a scalar
retype at the boundary of a DSL math call, a branchless select, a warp shuffle
with the right clamp, the dtype map, the DLPack wrapper, and a way to state a
shared-memory budget. They live here rather than in one operator's module so a
second operator does not have to import the first one's internals.

Operator math does not live here. The quaternion algebra, the tap chart, and the
3x3 composition belong to :mod:`slinoss.ops.so3ssd.cute.common`, which is their
only device-side implementation.

Importing this module imports the CuTe DSL. Nothing on a reference path imports
it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import get_smem_capacity_in_bytes

__all__ = [
    "LOG2_E",
    "TWO_LOG2_E",
    "Scalar",
    "Tile",
    "assert_smem_fits",
    "cute_dtype",
    "decay",
    "dev_tensor",
    "f32",
    "narrow",
    "select",
    "shuffle_up",
    "smem_bytes",
    "smem_capacity",
    "widen",
]

# exp(x) == exp2(x * LOG2_E), exp(2*x) == exp2(x * TWO_LOG2_E). One multiply
# ahead of one ex2.approx.f32.
LOG2_E: float = math.log2(math.e)
TWO_LOG2_E: float = 2.0 * LOG2_E

Scalar = cutlass.Float32
"""The only scalar width any of this repo's device math runs at (I4)."""

_TORCH_TO_CUTE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
}


def cute_dtype(dtype: torch.dtype) -> type:
    """Map a torch dtype to the CuTe numeric type.

    Args:
        dtype: One of bfloat16, float16, float32.

    Returns:
        The corresponding ``cutlass`` numeric type.

    Raises:
        TypeError: If the dtype has no kernel path. float64 is the reference
            oracle's width and reaches no kernel.
    """
    try:
        return _TORCH_TO_CUTE[dtype]
    except KeyError:
        raise TypeError(f"no CuTe kernel path for {dtype}") from None


def dev_tensor(tensor: torch.Tensor) -> cute.Tensor:
    """Wrap a contiguous torch tensor for a kernel launch.

    Only the trailing mode is declared contiguous; every other stride stays
    dynamic, so one compiled kernel serves every batch, head, and chunk count.
    Every tensor contract in this repo is time-major and contiguous, so the
    16-byte alignment claim holds for any allocation torch returns.

    The detach is not optional. ``from_dlpack`` refuses a tensor that requires
    grad, and inside a :class:`torch.autograd.Function` the saved operands still
    carry the flag even though grad mode is off. Detaching aliases the same
    storage, so it is not a staging copy.

    Args:
        tensor: A contiguous CUDA tensor.

    Returns:
        The CuTe view of it.
    """
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


# ---------------------------------------------------------------------------
# Shared-memory tiles and capacity
# ---------------------------------------------------------------------------


class Tile(NamedTuple):
    """One shared-memory allocation, as a shape and a stride.

    A tile is the single description of an allocation: the kernel builds its
    layout from :meth:`layout`, and the budget test counts its bytes from
    :attr:`words`. Two descriptions of one allocation drift.

    The tile exists rather than a bare ``cute.Layout`` because a layout cannot be
    built on the host. ``cute.make_layout`` outside a decorated body needs an MLIR
    context that the DSL only stands up inside one; supplying an explicit context
    lets the first call through and corrupts the process heap on the second.
    """

    shape: tuple[int, ...]
    """Extents, outermost first."""

    stride: tuple[int, ...]
    """Element stride of each extent. A pitch wider than the extent it precedes
    is padding, and :attr:`words` counts it."""

    def layout(self) -> cute.Layout:
        """The CuTe layout. Callable only from inside a decorated body."""
        return cute.make_layout(self.shape, stride=self.stride)

    @property
    def words(self) -> int:
        """Elements the tile spans, padding included.

        The span, not the product of the extents: a padded pitch makes the two
        differ, and the allocator advances by the span.
        """
        return 1 + sum(
            (extent - 1) * step
            for extent, step in zip(self.shape, self.stride, strict=True)
        )


def smem_bytes(tiles: Iterable[tuple[Tile, int]]) -> int:
    """Bytes a kernel's shared-memory tiles occupy.

    Args:
        tiles: ``(tile, itemsize)`` pairs, one per allocation the kernel makes.

    Returns:
        Total bytes.
    """
    return sum(itemsize * tile.words for tile, itemsize in tiles)


def smem_capacity() -> int:
    """Opt-in shared-memory capacity per block, in bytes.

    Queried from the DSL's own architecture, so no architecture string appears
    here or in any caller. The 48 KiB default is not the budget: the DSL attaches
    the dynamic-shared-memory opt-in attribute to every kernel it generates.

    Returns:
        Capacity in bytes.
    """
    return get_smem_capacity_in_bytes()


def assert_smem_fits(name: str, nbytes: int) -> int:
    """Check one kernel's shared-memory budget against the queried capacity.

    Args:
        name: Kernel name, for the message.
        nbytes: Bytes the kernel's tiles add up to.

    Returns:
        ``nbytes``, so a caller can use this inline.

    Raises:
        ValueError: If the budget exceeds capacity. There is no slop constant:
            either the tiles fit or the tiles change.
    """
    capacity = smem_capacity()
    if nbytes > capacity:
        raise ValueError(
            f"{name} needs {nbytes} B of shared memory, capacity is {capacity} B"
        )
    return nbytes


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def f32(value: object) -> Scalar:
    """Retype the result of a ``cute`` math call as a float32 scalar.

    The scalar path of the DSL's math ops returns a raw MLIR value rather than a
    numeric. Arithmetic on it still works, but a second math call on it and a
    ``.to()`` both reject it, and the failure surfaces at the second call rather
    than the first. Every boundary out of ``cute.exp2``, ``cute.rsqrt`` and their
    siblings goes through here.

    Args:
        value: The result of a ``cute`` scalar math op.

    Returns:
        The same value as a ``cutlass.Float32``.
    """
    return cutlass.Float32(value)


def widen(value: object, src: object) -> Scalar:
    """Read a tensor element as float32. Identity when the tensor is float32.

    ``.to()`` on a value already at the target width still emits a conversion in
    the DSL's IR, so the identity case is taken in Python rather than on the
    device.

    Args:
        value: An element read from a tensor.
        src: The tensor's element type, a compile-time ``cutlass`` numeric type.

    Returns:
        The value at float32.
    """
    return value if src is cutlass.Float32 else value.to(cutlass.Float32)  # type: ignore[attr-defined]


def narrow(value: Scalar, dst: object) -> object:
    """Write a float32 result at a tensor's width. Identity when that is float32.

    Args:
        value: A float32 scalar.
        dst: The destination tensor's element type, compile-time.

    Returns:
        The value at the destination width.
    """
    return value if dst is cutlass.Float32 else value.to(dst)


def select(cond: cutlass.Boolean, if_true: Scalar, if_false: Scalar) -> Scalar:
    """Branchless float32 select.

    ``cute.where`` is a tensor operation and rejects two scalar operands, so the
    scalar form goes through the DSL's conditional expression and is retyped.
    Lowers to one ``arith.select``, which is one predicated move: no divergence,
    so average active threads per warp is unaffected.

    Args:
        cond: A dynamic predicate. A compile-time predicate belongs in an
            ``if cutlass.const_expr(...)``, not here.
        if_true: Value taken where the predicate holds.
        if_false: Value taken elsewhere.

    Returns:
        The selected value.
    """
    return cutlass.Float32(cutlass.select_(cond, if_true, if_false))


def shuffle_up(value: Scalar, offset: int) -> Scalar:
    """``shfl.sync.up`` across a full warp: lane ``l`` reads lane ``l - offset``.

    Lanes below ``offset`` keep their own value, which is the identity the scans
    here rely on.

    The clamp field of the shuffle's packed operand is a lower bound on the
    source lane for the ``up`` direction, so a full-warp up-shuffle needs zero
    there. The DSL default is ``31``, which makes every lane read itself and
    turns a scan into a doubling.

    Args:
        value: The value to shift.
        offset: Lane distance. Compile-time in every caller.

    Returns:
        Lane ``l - offset``'s value, or the lane's own below ``offset``.
    """
    return cute.arch.shuffle_sync_up(value, offset, mask_and_clamp=0)


def decay(log_diff: Scalar) -> Scalar:
    """``exp(2 * log_diff)``.

    Args:
        log_diff: A difference of log-scale prefixes, never a sum of two
            separately exponentiated terms (I3). Non-positive wherever a segment
            decay is formed, so the result lies in ``(0, 1]`` and overflow is
            unreachable (I1).

    Returns:
        The decay factor.
    """
    return f32(cute.exp2(log_diff * TWO_LOG2_E))
