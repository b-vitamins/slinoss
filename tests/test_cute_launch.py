"""The two caches behind :func:`slinoss._cute.jit_launch`.

Both exist because the DSL charges for repetition. Tracing a ``@cute.jit``
function is 316 ms and building a ``cute.Tensor`` is 8.4 us on sm_86, against
launchers that trace one function and build nine descriptors per call, so
:func:`jit_launch` keeps a compiled executor per key and a descriptor per layout
and re-points the descriptor's base word. Every test here names one way that
reuse turns into a wrong launch.

One probe kernel computes ``out = a - b`` over three tensors of one layout. The
aliasing case is then visible in the output rather than only in the pool: two
arguments sharing a descriptor gives an exact zero, and the launch goes through
the shipped path, raw tensors into :func:`jit_launch`, not through a private
helper.

The pool is keyed on layout and not on address, so its axes are address freshness
and borrow order. The executor is keyed on what shapes the generated code, so its
axis is dtype. Shape is not swept: it is a resolution axis for neither cache, and
the kernels' own parity tests sweep it.
"""

import threading

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from torch import Tensor

from slinoss import _cute
from slinoss._cute import clear_dev_pool, dev_tensor, jit_launch

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

N = 128
"""Probe length. One block, one thread per element."""


@cute.kernel
def _combine_kernel(ga: cute.Tensor, gb: cute.Tensor, gout: cute.Tensor) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gout[tid] = ga[tid] - gb[tid]


@cute.jit
def _combine_launch(
    ga: cute.Tensor, gb: cute.Tensor, gout: cute.Tensor, n: cutlass.Constexpr
) -> None:
    _combine_kernel(ga, gb, gout).launch(grid=(1, 1, 1), block=(n, 1, 1))


def _combine(a: Tensor, b: Tensor) -> Tensor:
    """``a - b`` through the shipped launch path."""
    out = torch.empty_like(a)
    jit_launch(_combine_launch, (a, b, out), (a.numel(),))
    torch.cuda.synchronize()
    return out


def _pool() -> dict:
    """This thread's pool. Empty dict if nothing has been borrowed yet."""
    return getattr(_cute._POOLS, "slots", {})


def _operands(n: int = N) -> tuple[Tensor, Tensor]:
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    return a, a * 3.0


def test_two_arguments_of_one_layout_get_separate_descriptors() -> None:
    """One launch with two same-layout arguments must read two buffers.

    ``a`` and ``b`` share dtype, shape and stride, so they share a pool key. A
    slot handed out twice in one launch would leave both arguments pointing at
    whichever tensor was converted last, and ``a - b`` would be exactly zero.
    That is why the borrow count advances per argument and drops only when the
    launch ends, rather than when the tensor it was built from dies.
    """
    a, b = _operands()
    out = _combine(a, b)
    assert torch.equal(out, a - b)
    assert out.abs().sum() > 0


def test_a_pooled_descriptor_reads_the_current_tensor_address() -> None:
    """A second launch must read its own arguments, not the first launch's.

    Both pairs are alive at once, so they sit at different addresses. A
    descriptor that kept the address it was built with would make the second
    launch recompute the first result.
    """
    a1, b1 = _operands()
    a2, b2 = _operands()
    assert a1.data_ptr() != a2.data_ptr()
    _combine(a1, b1)
    out = _combine(a2 + 5.0, b2)
    assert torch.equal(out, (a2 + 5.0) - b2)


def test_a_recycled_address_is_not_reachable_through_a_pooled_descriptor() -> None:
    """Freeing an argument and reallocating its address must not resurrect it.

    The caching allocator hands the identical block back for a same-size request
    on the same stream, which is what makes a cache keyed on ``data_ptr()`` a
    use-after-free. The pool key is ``(dtype, device, shape, stride)`` and holds
    no address, so the recycled tensor is a plain cache hit on layout with the
    base word re-read from it.

    The recycling assertion is part of the test: without it the hazard is not
    exercised and the rest of the test proves nothing.
    """
    a, b = _operands()
    _combine(a, b)
    ptr = a.data_ptr()
    del a
    fresh = torch.full((N,), 7.0, dtype=torch.float32, device="cuda")
    assert fresh.data_ptr() == ptr, "allocator did not recycle; hazard not exercised"
    out = _combine(fresh, torch.zeros_like(fresh))
    assert torch.equal(out, fresh)


def test_repeated_launches_build_one_descriptor_per_argument() -> None:
    """Six conversions over two launches must build three descriptors.

    This is the reuse the host-cost cut rests on, and it is also what says the
    borrows of the first launch were released: a launch that kept them would
    take slots 3, 4 and 5. A length of its own keeps the count independent of
    the other tests.
    """
    n = 64
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = a * 3.0
    _combine(a, b)
    _combine(a, b)
    key = (a.dtype, a.device, a.shape, a.stride())
    assert len(_pool()[key].views) == 3


def test_two_operand_dtypes_do_not_share_one_executor() -> None:
    """An executor is specialized on its operands' element type.

    A kernel is free to read its element type off the tensor instead of taking it
    as a :class:`cutlass.Constexpr`, so ``static`` need not name the dtype and for
    several entry points here it does not. Keyed without it, the second dtype's
    launch runs the first dtype's compiled code: the executor takes a base pointer
    and a dynamic layout, so nothing at call time contradicts it and the kernel
    reads the buffer at the wrong element width.

    Two dtypes of one length, so ``static`` is identical across the pair and the
    dtype is the only thing separating them.
    """
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    assert torch.equal(_combine(a, a * 3.0), a - a * 3.0)
    h = a.to(torch.float16)
    assert torch.equal(_combine(h, h * 3.0), h - h * 3.0)


def test_pools_are_per_thread() -> None:
    """A second thread gets its own pool.

    Autograd runs a backward on its own thread. A borrow count shared across
    threads would let two concurrent launches take the same slot, and the loser
    would launch against the winner's base pointer.
    """
    a, b = _operands()
    _combine(a, b)
    seen: list[object] = []

    def run() -> None:
        seen.append(_pool())
        torch.cuda.set_device(0)
        assert torch.equal(_combine(a, b), a - b)
        seen.append(_pool())

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert seen[0] == {}
    assert seen[1] is not _pool()


def test_the_memref_base_word_is_the_pointer_field() -> None:
    """The pool's one ABI assumption, checked against the installed DSL.

    A borrow re-points the leading word of the memref descriptor. The pool
    verifies that word against ``data_ptr()`` when it fills a slot and disables
    pooling for the process if it disagrees, so a DSL that moved the field would
    be correct and slow. This test is what reports that it went slow.
    """
    a, b = _operands()
    _combine(a, b)
    assert _cute._POOLABLE
    assert _pool()


def test_a_caller_built_descriptor_is_passed_through() -> None:
    """A descriptor already in ``dynamic`` must reach the launch unconverted.

    An optional operand is a sentinel descriptor rather than a tensor, and a
    launcher that needs one view in two argument slots builds it itself. A
    conversion that rejected a non-tensor argument, or tried to borrow one,
    would break both.
    """
    a, b = _operands()
    out = torch.empty_like(a)
    jit_launch(_combine_launch, (dev_tensor(a), b, out), (a.numel(),))
    torch.cuda.synchronize()
    assert torch.equal(out, a - b)


def test_pooling_off_still_launches() -> None:
    """The fallback the ABI check drops to must run.

    Reached for the whole process when the memref's leading word is not the base
    pointer. It is the slow path, so nothing else exercises it, and an untested
    fallback is not a fallback.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_cute, "_POOLABLE", False)
        a, b = _operands()
        assert torch.equal(_combine(a, b), a - b)


def test_a_layout_past_the_key_cap_is_not_pooled() -> None:
    """The cap must drop pooling, not correctness.

    Retention is one pinned allocation per filled slot, so the key count is
    capped. Past it a borrow is built unpooled and the launch is unaffected.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_cute, "_POOL_KEYS", 1)
        clear_dev_pool()
        a, b = _operands()
        _combine(a, b)
        assert len(_pool()) == 1
        wide, narrow = _operands(N * 2)
        assert torch.equal(_combine(wide, narrow), wide - narrow)
        assert len(_pool()) == 1


def test_clear_dev_pool_drops_what_the_pool_pins() -> None:
    """Clearing must release the descriptors and leave the path working.

    A pooled descriptor pins the allocation of the tensor it was built from
    through its DLPack capsule, so a caller that sweeps layouts needs a way to
    give them back. The next borrow rebuilds.
    """
    a, b = _operands()
    _combine(a, b)
    assert _pool()
    clear_dev_pool()
    assert _pool() == {}
    assert torch.equal(_combine(a, b), a - b)
