"""Warp primitives in :mod:`slinoss._cute`: the two shuffle directions and the
block reduction.

The permutations are pinned as exact integers rather than through a consumer.
The shuffle's clamp field is a lower bound on the source lane in the ``up``
direction and an upper bound in the other two, so the three helpers do not share
a constant; a wrong one is a permutation that still returns plausible floats and
would surface as a numerical disagreement inside whichever scan consumed it. One
probe kernel returning lane indices localizes it here.

The lanes that fall off the end of a shift keep their own value rather than
clamping to the end lane, so the sweep runs distances above one, where those two
readings differ.

``block_reduce_add`` is checked against a torch sum at three block widths, because
its cross-warp pass is a loop over ``threads // 32`` and a single width would not
distinguish it from a warp reduction.
"""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute

from slinoss._cute import (
    Tile,
    block_reduce_add,
    dev_tensor,
    shuffle_down,
    shuffle_up,
    shuffle_xor,
)

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

WARP = 32

UP = 0
DOWN = 1
XOR = 2


@cute.kernel
def _shuffle_kernel(
    gout: cute.Tensor, mode: cutlass.Constexpr, offset: cutlass.Constexpr
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    value = cutlass.Float32(0.0) + tid.to(cutlass.Float32)
    if cutlass.const_expr(mode == UP):
        result = shuffle_up(value, offset)
    elif cutlass.const_expr(mode == DOWN):
        result = shuffle_down(value, offset)
    else:
        result = shuffle_xor(value, offset)
    gout[tid] = result


@cute.jit
def _shuffle_launch(
    gout: cute.Tensor, mode: cutlass.Constexpr, offset: cutlass.Constexpr
) -> None:
    _shuffle_kernel(gout, mode, offset).launch(grid=(1, 1, 1), block=(WARP, 1, 1))


@cute.kernel
def _reduce_kernel(
    gin: cute.Tensor, gout: cute.Tensor, threads: cutlass.Constexpr
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sred = smem.allocate_tensor(
        cutlass.Float32, Tile((threads // WARP,), (1,)).layout(), 16
    )
    gout[tid] = block_reduce_add(gin[tid], sred, tid, threads)


@cute.jit
def _reduce_launch(
    gin: cute.Tensor, gout: cute.Tensor, threads: cutlass.Constexpr
) -> None:
    _reduce_kernel(gin, gout, threads).launch(grid=(1, 1, 1), block=(threads, 1, 1))


def _shuffle(mode: int, offset: int) -> list[int]:
    """Lane indices as permuted by one shuffle across a full warp."""
    out = torch.empty(WARP, dtype=torch.float32, device="cuda")
    _shuffle_launch(dev_tensor(out), mode, offset)
    torch.cuda.synchronize()
    return out.int().tolist()


@pytest.mark.parametrize("offset", [1, 8, 16])
def test_up_holds_the_bottom_lanes(offset: int) -> None:
    """Lane ``l`` reads ``l - offset``; the bottom ``offset`` lanes keep their own.

    Keeping their own value is not the same as clamping to lane 0, and the two
    coincide at ``offset = 1``, so the sweep runs the wider distances a 32-lane
    scan reaches. The held value is the scan's identity contribution only because
    the caller predicates the combine on ``lane >= offset``, which is what
    :mod:`slinoss.ops.so3ssd.cute.prefix` does.
    """
    want = [lane - offset if lane >= offset else lane for lane in range(WARP)]
    assert _shuffle(UP, offset) == want


@pytest.mark.parametrize("offset", [1, 8, 16])
def test_down_holds_the_top_lanes(offset: int) -> None:
    """Lane ``l`` reads ``l + offset``; the top ``offset`` lanes keep their own.

    The mirror of the up case, held value included, and the identity the backward's
    suffix scans rely on.
    """
    want = [lane + offset if lane + offset < WARP else lane for lane in range(WARP)]
    assert _shuffle(DOWN, offset) == want


@pytest.mark.parametrize("mask", [1, 8, 16])
def test_xor_exchanges_every_lane(mask: int) -> None:
    """Lane ``l`` reads ``l ^ mask``, with no lane held.

    Every lane both sends and receives, which is what lets a butterfly leave a
    warp total in all 32 lanes and removes the broadcast step that a shift-based
    reduction needs.
    """
    assert _shuffle(XOR, mask) == [lane ^ mask for lane in range(WARP)]


@pytest.mark.parametrize("threads", [WARP, 128, 256])
def test_block_reduce_add_totals_in_every_thread(threads: int) -> None:
    """The block sum, identical in every thread, against a float32 torch sum.

    One warp exercises the degenerate cross-warp loop; the two wider widths make
    it a real loop. Both sides add the same float32 values, so the only difference
    is summation order, and the bound is the reassociation of at most 256 terms of
    order one.

    The broadcast is asserted bitwise. Every thread reads the same shared words in
    the same order, so anything short of exact equality means a thread took a
    different path through the reduction.
    """
    gen = torch.Generator(device="cuda").manual_seed(threads)
    values = torch.randn(threads, generator=gen, dtype=torch.float32, device="cuda")
    out = torch.empty_like(values)

    _reduce_launch(dev_tensor(values), dev_tensor(out), threads)
    torch.cuda.synchronize()

    assert bool((out == out[0]).all()), "the total is not broadcast to every thread"
    assert float(out[0]) == pytest.approx(float(values.sum()), rel=1e-6)
