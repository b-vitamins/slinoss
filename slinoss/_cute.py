"""CuTe DSL shims and shared-memory policy. No operator owns these.

Every operator's ``cute`` package needs the same handful of things: a scalar
retype at the boundary of a DSL math call, a branchless select, a warp shuffle
with the right clamp, the dtype map, the DLPack wrapper, and a way to state a
shared-memory budget. They live here rather than in one operator's module so a
second operator does not have to import the first one's internals.

A scalar function shared by more than one operator lives here, as its only
device-side implementation: the decay exponential and the logistic family. Two
operators rounding the same activation two ways is a divergence, and the
divergence is a correctness bug.

Per-operator algebra does not live here. The quaternion algebra, the tap chart,
and the 3x3 composition belong to :mod:`slinoss.ops.so3ssd.cute.common`, which is
their only device-side implementation.

Importing this module imports the CuTe DSL. Nothing on a reference path imports
it.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Final, NamedTuple, Protocol

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda_driver
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import get_smem_capacity_in_bytes

__all__ = [
    "LOG2_E",
    "SMEM_RESERVED",
    "TWO_LOG2_E",
    "CacheEvents",
    "Compiled",
    "DevPoolUse",
    "LaunchPayload",
    "Scalar",
    "Stream",
    "Tile",
    "assert_smem_fits",
    "block_reduce_add",
    "cache_events",
    "clear_dev_pool",
    "compiled_launches",
    "current_stream",
    "cute_dtype",
    "decay",
    "dev_pool_use",
    "dev_tensor",
    "executor_count",
    "f32",
    "jit_launch",
    "narrow",
    "reset_cache_events",
    "select",
    "shuffle_down",
    "shuffle_up",
    "shuffle_xor",
    "sigmoid",
    "silu",
    "silu_grad",
    "smem_bytes",
    "smem_capacity",
    "smem_residency",
    "use_payload",
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


# ---------------------------------------------------------------------------
# Device tensor descriptors
# ---------------------------------------------------------------------------


def dev_tensor(tensor: torch.Tensor) -> cute.Tensor:
    """Wrap a torch tensor for a kernel launch.

    Only the trailing mode is declared contiguous; every other stride stays
    dynamic, so one compiled kernel serves every batch, head, and chunk count, and
    a band of the mixer's fused projection needs no kernel change to read at its
    own pitch. The 16-byte alignment claim is what a caller owes:
    :func:`slinoss._guard.check_pitched` is the rule, and it is checked on the
    host for every operand that arrives pitched.

    The detach is not optional. ``from_dlpack`` refuses a tensor that requires
    grad, and inside a :class:`torch.autograd.Function` the saved operands still
    carry the flag even though grad mode is off. Detaching aliases the same
    storage, so it is not a staging copy.

    Building a descriptor costs 8.4 us on sm_86, so a launcher does not call this
    per operand: it hands its torch tensors to :func:`jit_launch`, which converts
    them through the descriptor pool. This is the unpooled builder, for a caller
    that holds a descriptor across launches or invokes a ``@cute.jit`` function
    directly.

    Args:
        tensor: A CUDA tensor with unit stride on its trailing axis, aligned to 16
            bytes at its base and on every other stride.

    Returns:
        The CuTe view of it, valid for as long as ``tensor`` is.
    """
    return from_dlpack(tensor.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


# ---------------------------------------------------------------------------
# Compiled launch
# ---------------------------------------------------------------------------

Stream = cuda_driver.CUstream
"""The launch stream's type, as a ``@cute.jit`` launcher declares it.

Every launcher takes one, last of its runtime arguments, and hands it to
``launch(stream=...)``. It is not optional and it has no default: a launch with no
stream goes to the legacy default stream, which is neither the stream the operands
were produced on nor a stream a graph can capture. Measured: a launch without it
runs during capture instead of being recorded, and the graph replays as a no-op.
"""

_STREAMS: dict[int, Stream] = {}
"""Stream handles by raw pointer. Wrapping one costs a construction per launch and
a process sees at most a handful, so they are interned rather than rebuilt."""


def current_stream() -> Stream:
    """The handle of torch's current stream on the current device.

    Every CuTe launch takes this. Torch produced the operands on this stream and
    consumes the results on it, so a kernel on any other stream is ordered against
    neither; the legacy default stream survives that only by implicitly
    synchronizing every blocking stream, which serializes what it appears to
    overlap and cannot be captured into a graph at all.

    Returns:
        A ``cuda.bindings.driver.CUstream`` for the current stream, interned.
    """
    raw = torch.cuda.current_stream().cuda_stream
    handle = _STREAMS.get(raw)
    if handle is None:
        handle = _STREAMS[raw] = Stream(raw)
    return handle


_Key = tuple[Callable[..., None], tuple[Hashable, ...], int, tuple[Hashable, ...]]
"""An executor cache key: launcher, compile-time arguments, device, declared runtime
types. Everything that shapes the generated code, plus the device the code was
built for. The stream is not in it; see :func:`jit_launch`."""

_EXECUTORS: dict[_Key, Any] = {}

_POOL_DEPTH = 4
"""Pooled descriptors per layout. Four is the largest number of arguments of one
layout any launch here has -- ``U``, ``B``, ``C`` and ``Y`` of the chunk scan
coincide where ``P == 3N``. Past it a borrow is built unpooled."""

_POOL_KEYS = 256
"""Layouts pooled per thread. A filled slot pins one allocation, so an unbounded
key count is an unbounded leak in a process that sweeps shapes. Past the cap a
borrow is built unpooled, and :func:`clear_dev_pool` gives back what is held."""

_POOLS = threading.local()
"""Per-thread pools. Autograd runs a backward on its own thread, and a borrow
count shared across threads would hand one descriptor to two launches."""

_POOLABLE = True
"""Cleared for the process if the memref's leading word is not the base pointer.
A borrow re-points that word, so the check is what makes the patch legal."""


class _Pooled(NamedTuple):
    """One pooled descriptor.

    Attributes:
        view: The CuTe view.
        word: One-word ctypes view of the leading word of the memref, which is the
            descriptor's base pointer. Borrowing re-points it.
        storage: ``(data_ptr, nbytes)`` of the allocation the view's DLPack capsule
            pins. Recorded at build time, which is the allocation that stays
            pinned: the capsule outlives the tensor the slot was built from.
    """

    view: cute.Tensor
    word: Any
    storage: tuple[int, int]


class _Slots:
    """The pooled descriptors of one layout, and the live borrow count."""

    __slots__ = ("taken", "views")

    def __init__(self) -> None:
        self.taken = 0
        self.views: list[_Pooled] = []


def _borrow(tensor: torch.Tensor) -> cute.Tensor:
    """A descriptor for one launch argument, from the pool where possible.

    Valid until :func:`_release`, which is why :func:`jit_launch` is the only
    caller: a borrow taken outside a launch has no scope that ends, and a
    descriptor that outlived its launch would be handed out again while the first
    launch still held it, putting two arguments on one base pointer.

    The key is ``(dtype, device, shape, stride)`` and holds no address, so a
    recycled allocation is not reachable through a stale descriptor: the base
    pointer is re-read from the live tensor on every borrow. Two arguments of one
    launch that share a layout get two slots, because the count advances per
    argument and drops only when the launch ends.

    Args:
        tensor: A contiguous CUDA tensor.

    Returns:
        The CuTe view of it.
    """
    global _POOLABLE
    if not _POOLABLE:
        return dev_tensor(tensor)

    slots: dict[Hashable, _Slots] | None = getattr(_POOLS, "slots", None)
    if slots is None:
        slots = _POOLS.slots = {}
        _POOLS.borrowed = []

    key = (tensor.dtype, tensor.device, tensor.shape, tensor.stride())
    entry = slots.get(key)
    if entry is None:
        if len(slots) >= _POOL_KEYS:
            return dev_tensor(tensor)
        entry = slots[key] = _Slots()
    index = entry.taken
    entry.taken = index + 1
    _POOLS.borrowed.append(entry)

    if index < len(entry.views):
        pooled = entry.views[index]
        pooled.word[0] = tensor.data_ptr()
        return pooled.view

    view = dev_tensor(tensor)
    if index >= _POOL_DEPTH:
        return view

    word = (ctypes.c_int64 * 1).from_address(view.__c_pointers__()[0])
    if word[0] != tensor.data_ptr():
        _POOLABLE = False
        slots.clear()
        _POOLS.borrowed.clear()
        return view
    storage = tensor.untyped_storage()
    entry.views.append(_Pooled(view, word, (storage.data_ptr(), storage.nbytes())))
    return view


def _release() -> None:
    """Drop the borrow counts of the launch that is ending.

    Every borrow on the list belongs to that launch: :func:`jit_launch` converts
    its own arguments and is the only caller of :func:`_borrow`, so no borrow
    spans a call out into other code and none can belong to an enclosing launch.
    """
    borrowed: list[_Slots] | None = getattr(_POOLS, "borrowed", None)
    if not borrowed:
        return
    for entry in borrowed:
        entry.taken -= 1
    borrowed.clear()


def clear_dev_pool() -> None:
    """Drop every pooled descriptor on this thread.

    A pooled descriptor holds the DLPack capsule of the tensor it was built from,
    which pins that allocation. The capsule cannot be dropped: the DSL wrapper
    reads the element type and the MLIR type through it, and a compile against a
    stripped descriptor fails. So the pool retains one allocation per filled slot
    whose tensor the caller does not hold itself, which is a launcher's own
    outputs -- 14.03 MiB for the chunk increment at the standard shape on sm_86,
    one copy of its increment tensor.

    Correctness does not depend on this: a dropped layout is rebuilt on its next
    borrow. It exists for a caller that sweeps layouts and then stops, such as a
    shape sweep in the perf harness. Not callable from inside a launch.
    :func:`dev_pool_use` reads what is held before and after.
    """
    slots: dict[Hashable, _Slots] | None = getattr(_POOLS, "slots", None)
    if slots is not None:
        slots.clear()
        _POOLS.borrowed.clear()


class DevPoolUse(NamedTuple):
    """What the descriptor pool holds on the calling thread.

    Attributes:
        layout_count: Layouts with a slot list. Bounded by the pool's key cap; past
            it a borrow is built unpooled and retains nothing.
        descriptor_count: Filled slots, summed over layouts.
        retained_bytes: Distinct storage the filled slots pin. Two layouts of one
            allocation are two slots and one storage, so it counts once.
    """

    layout_count: int
    descriptor_count: int
    retained_bytes: int


def dev_pool_use() -> DevPoolUse:
    """Read the pool's occupancy and what it retains.

    The retained figure is the cost side of the pool: it is what
    :func:`clear_dev_pool` gives back, and without it those bytes read as a leak
    or as autograd's saved set in a memory report.

    Returns:
        The counts on this thread. All zero before the first launch, and in a
        process where the descriptor patch was refused.
    """
    slots: dict[Hashable, _Slots] | None = getattr(_POOLS, "slots", None)
    if slots is None:
        return DevPoolUse(0, 0, 0)
    pinned = {pooled.storage for entry in slots.values() for pooled in entry.views}
    return DevPoolUse(
        layout_count=len(slots),
        descriptor_count=sum(len(entry.views) for entry in slots.values()),
        retained_bytes=sum(nbytes for _, nbytes in pinned),
    )


def executor_count() -> int:
    """Number of compiled executors held. For tests that assert cache reuse."""
    return len(_EXECUTORS)


# ---------------------------------------------------------------------------
# Where an executor came from
# ---------------------------------------------------------------------------


class LaunchPayload(Protocol):
    """An ahead-of-time payload, as :func:`jit_launch` consults it.

    Implemented by :class:`slinoss.aot.Payload`. A protocol and a slot rather
    than an import, because the payload module imports this one to name a key.

    Attributes:
        strict: Raise on a key the payload does not hold instead of compiling it.
    """

    strict: bool

    def lookup(
        self,
        fn: Callable[..., None],
        static: tuple[Hashable, ...],
        signature: tuple[Hashable, ...],
    ) -> Any | None:
        """The entry for one launch key, or None if the payload holds none.

        Args:
            fn: The ``@cute.jit`` launcher.
            static: Its compile-time arguments, in order.
            signature: What its runtime arguments declared.

        Returns:
            Something callable on the dynamic arguments alone, exactly as the
            value ``cute.compile`` returns is, or None.
        """
        ...


_PAYLOAD: LaunchPayload | None = None
"""The payload consulted before a compile. None until one is loaded, and with
none loaded the path is what it was before payloads existed."""


def use_payload(payload: LaunchPayload | None) -> None:
    """Install the payload :func:`jit_launch` consults, or remove it.

    Process-wide, and not retroactive: an executor already in the cache stays,
    since a payload entry and a freshly compiled one are the same code. Two
    payloads at once is not a case -- the second call replaces the first.

    Args:
        payload: The payload, or None to go back to compiling every key.
    """
    global _PAYLOAD
    _PAYLOAD = payload


class _Events:
    """The process's compile-cache event counts. One instance."""

    __slots__ = ("compile_ns", "compiled", "hits", "payload_hits", "payload_misses")

    def __init__(self) -> None:
        self.compiled = 0
        self.compile_ns = 0
        self.hits = 0
        self.payload_hits = 0
        self.payload_misses = 0


_EVENTS = _Events()


class CacheEvents(NamedTuple):
    """Where the process's launches got their executors.

    The counters exist so that "the cold pass was actually cold" is a
    measurement. Reading them costs a namedtuple construction and no profiler.

    Attributes:
        compiled: Executors built by ``cute.compile``. A second pass over work a
            first pass already did adds none.
        hits: Launches the executor cache served outright, which is every launch
            past the first of its key.
        payload_hits: Executors taken from a loaded payload rather than compiled.
        payload_misses: Keys a loaded payload did not hold. Zero while none is
            loaded, and in strict mode the first one raises instead of counting.
        compile_us: Microseconds spent inside ``cute.compile``, summed. Trace,
            MLIR passes and ptxas, which is the whole of what a payload removes.
        dsl_hits: The DSL's own ``cache_hits``, or None when it is not counting.
        dsl_misses: The DSL's own ``cache_misses``, or None. The DSL defines both
            only under ``CUTE_DSL_JIT_TIME_PROFILING``, and ``cute.compile`` sets
            ``no_cache`` unconditionally, so the miss count advances once per
            compile and the hit count cannot advance at all. They are surfaced to
            say that, not as a reuse signal.
    """

    compiled: int
    hits: int
    payload_hits: int
    payload_misses: int
    compile_us: float
    dsl_hits: int | None
    dsl_misses: int | None


_DSL: Any = None
"""The DSL singleton, memoized. Its accessor constructs the class."""


def _dsl_cache_events() -> tuple[int | None, int | None]:
    """The DSL's own cache counters, or ``(None, None)`` if it holds none."""
    global _DSL
    if _DSL is None:
        _DSL = BaseDSL._get_dsl()
    return getattr(_DSL, "cache_hits", None), getattr(_DSL, "cache_misses", None)


def cache_events() -> CacheEvents:
    """Read the compile-cache counters.

    Returns:
        The counts for this process. All zero before the first launch.
    """
    dsl_hits, dsl_misses = _dsl_cache_events()
    return CacheEvents(
        compiled=_EVENTS.compiled,
        hits=_EVENTS.hits,
        payload_hits=_EVENTS.payload_hits,
        payload_misses=_EVENTS.payload_misses,
        compile_us=_EVENTS.compile_ns / 1000.0,
        dsl_hits=dsl_hits,
        dsl_misses=dsl_misses,
    )


def reset_cache_events() -> None:
    """Zero the counters, for a caller framing one interval.

    The executors themselves are not dropped, so a zeroed count followed by a
    nonzero ``hits`` and a zero ``compiled`` is the warm case. The DSL's own
    counters are not this module's to zero.
    """
    _EVENTS.compiled = 0
    _EVENTS.compile_ns = 0
    _EVENTS.hits = 0
    _EVENTS.payload_hits = 0
    _EVENTS.payload_misses = 0


class Compiled(NamedTuple):
    """One held executor and the parts of its key that name it.

    Attributes:
        fn: The ``@cute.jit`` launcher.
        static: Its compile-time arguments, in order.
        signature: What its runtime arguments declared: dtype and rank per tensor,
            element type or None per anything else.
        executor: The compiled function, callable on the dynamic arguments alone.
    """

    fn: Callable[..., None]
    static: tuple[Hashable, ...]
    signature: tuple[Hashable, ...]
    executor: Any


def compiled_launches() -> tuple[Compiled, ...]:
    """Every executor the process holds, with the key that reached it.

    The executor cache is the record of what a workload compiled: an entry is
    added on the first launch of a key and never evicted, so one pass over a step
    leaves exactly the set that step needs. An ahead-of-time build reads this
    rather than a hand-written key list, which drifts the moment a kernel gains a
    :class:`cutlass.Constexpr`.

    The device index is dropped. It keys the executor because an executor binds
    to a device context, and it shapes no generated code.

    Returns:
        One entry per held executor, in the order they were built.
    """
    return tuple(
        Compiled(fn=fn, static=static, signature=signature, executor=executor)
        for (fn, static, _device, signature), executor in _EXECUTORS.items()
    )


def _executor_for(
    fn: Callable[..., None],
    args: list[Any],
    static: tuple[Hashable, ...],
    signature: tuple[Hashable, ...],
) -> Any:
    """The executor for a key the cache does not hold: from the payload, or built.

    Off the hot path by construction -- reached once per key -- so the payload
    consult costs a launch nothing after its first.

    Args:
        fn: The ``@cute.jit`` launcher.
        args: Its runtime arguments, stream included, as the launch will pass
            them.
        static: Its compile-time arguments, in order.
        signature: What the runtime arguments declared.

    Returns:
        The executor.

    Raises:
        KeyError: If a strict payload is loaded and does not hold the key.
    """
    payload = _PAYLOAD
    if payload is not None:
        entry = payload.lookup(fn, static, signature)
        if entry is not None:
            _EVENTS.payload_hits += 1
            return entry
        if payload.strict:
            raise KeyError(
                f"the payload holds no entry for {fn.__module__}.{fn.__qualname__} "
                f"static={static} signature={signature}, and strict mode does not "
                f"compile"
            )
        _EVENTS.payload_misses += 1
    start = time.perf_counter_ns()
    executor = cute.compile(fn, *args, *static)
    _EVENTS.compile_ns += time.perf_counter_ns() - start
    _EVENTS.compiled += 1
    return executor


def jit_launch(
    fn: Callable[..., None],
    dynamic: Sequence[Any],
    static: tuple[Hashable, ...],
) -> None:
    """Launch a ``@cute.jit`` entry point through a cached executor.

    Calling a ``@cute.jit`` function retraces it. The trace is not the kernel
    compile -- that is cached by the DSL -- so it is paid on every call and it
    dominates everything the kernel does. Measured on sm_86 at the standard
    shape: 316 ms per direct call of the chunk increment against 0.126 ms
    through the executor, over a kernel whose DRAM floor is tens of
    microseconds. ``cute.compile`` traces once and returns a callable that takes
    the dynamic arguments alone.

    Torch tensors in ``dynamic`` are converted here rather than by the caller.
    Building a descriptor is a DLPack export, a wrapper construction and a memref
    materialization, 8.4 us on sm_86, so the nine a launch needs put the host
    ahead of a kernel that runs in tens of microseconds; a pooled borrow
    re-points the memref's base word instead. Host cost per launch at the
    standard shape on sm_86, clocks unlocked: the chunk increment 197 us to
    120 us, the state passing 113 us to 65 us. Converting here rather than in the
    caller is what bounds a pooled descriptor's life to one launch.

    The cache key must name every property that shapes the generated code, and
    ``static`` is not all of it. A tensor argument contributes its element type
    and its rank -- and nothing else, because :func:`dev_tensor` marks every
    layout dynamic except the leading mode -- and a kernel is free to read its
    element type off the tensor rather than take it as a
    :class:`cutlass.Constexpr`, in which case ``static`` names no dtype at all.
    So the key is built from ``static`` and from what the arguments themselves
    declare. Leaving that to the caller is silent: an executor takes a base
    pointer and a dynamic layout, so a second dtype launched through the first
    dtype's code reads the buffer at the wrong element width and returns.

    The launch stream is appended to ``dynamic`` here, so every launcher takes a
    :data:`Stream` last of its runtime arguments. See :func:`current_stream` for
    why a launcher may not be called without one. It shapes no generated code, so
    it is neither in the key nor baked into an ahead-of-time entry: a payload
    entry replayed on another stream launches on that stream.

    A key the cache does not hold goes to :func:`_executor_for`, which consults a
    loaded payload before it compiles. With no payload loaded that is a compile,
    as it has always been. :func:`cache_events` counts both.

    Args:
        fn: The ``@cute.jit`` launcher.
        dynamic: Its leading run of runtime arguments, stream excluded. A torch
            tensor is converted to a pooled descriptor; anything else -- a scalar,
            or a descriptor the caller built itself -- is passed through.
        static: Its trailing run of compile-time arguments, in order.

    Raises:
        TypeError: If ``static`` holds an unhashable value, which would mean a
            compile-time argument that cannot key the cache.
        KeyError: If a strict payload is loaded and holds no entry for the key.
    """
    try:
        args: list[Any] = []
        # Flat rather than one entry per argument: the arity is fixed by ``fn``,
        # so a pair per tensor costs a tuple allocation and a nested hash per
        # launch and buys nothing.
        signature: list[Hashable] = []
        for arg in dynamic:
            if isinstance(arg, torch.Tensor):
                args.append(_borrow(arg))
                signature.append(arg.dtype)
                signature.append(arg.ndim)
            else:
                # A scalar declares nothing: its type is fixed by the entry
                # point's own signature. A descriptor the caller built declares
                # its element type the same way a tensor does.
                args.append(arg)
                signature.append(getattr(arg, "element_type", None))
        # The stream is appended here rather than taken from the caller: it is the
        # same for every launcher, it is not the caller's to choose, and a launcher
        # that could be called without one would be a launch on the default stream.
        # It shapes no generated code, so it stays out of the key.
        args.append(current_stream())
        declared = tuple(signature)
        key = (fn, static, torch.cuda.current_device(), declared)
        executor = _EXECUTORS.get(key)
        if executor is None:
            executor = _EXECUTORS[key] = _executor_for(fn, args, static, declared)
        else:
            _EVENTS.hits += 1
        executor(*args)
    finally:
        _release()


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


SMEM_RESERVED: Final = 1024
"""Shared memory the driver reserves per block, in bytes, on every supported arch.

Not part of :func:`smem_capacity`, which reports what one block may allocate:
the reservation is already subtracted there. It reappears once a second block is
asked for, because each block pays it, and it is what makes the residency divisor
larger than the tiles.
"""


def smem_residency(nbytes: int) -> int:
    """Blocks per SM a shared-memory budget allows, at least one.

    ``smem_capacity() // nbytes`` is the wrong divisor and reads one block too
    high near a boundary: the capacity has one reservation subtracted from it
    already, and ``k`` blocks pay ``k`` of them. Measured on sm_86, a 34,304 B
    arena computes as three by that formula and runs as two, which cost a lane 25%.

    Args:
        nbytes: Bytes one block's tiles add up to. Zero or less yields the
            occupancy limit no shared memory implies, which is not a limit at all,
            so the caller's own cap decides.

    Returns:
        Blocks the carveout holds, floor at one. One is returned for a budget that
        does not fit at all; :func:`assert_smem_fits` is what refuses that.
    """
    carveout = smem_capacity() + SMEM_RESERVED
    if nbytes <= 0:
        return carveout
    return max(1, carveout // (nbytes + SMEM_RESERVED))


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


def shuffle_down(value: Scalar, offset: int) -> Scalar:
    """``shfl.sync.down`` across a full warp: lane ``l`` reads lane ``l + offset``.

    Lanes within ``offset`` of the top keep their own value, which is the identity
    the suffix scans here rely on.

    The clamp is the mirror of :func:`shuffle_up`'s: an upper bound on the source
    lane rather than a lower one, so the full-warp value is the top lane index and
    not zero. The two directions therefore take opposite constants, and zero here
    is the identity permutation rather than a shorter reach.

    Args:
        value: The value to shift.
        offset: Lane distance. Compile-time in every caller.

    Returns:
        Lane ``l + offset``'s value, or the lane's own within ``offset`` of the top.
    """
    return cute.arch.shuffle_sync_down(
        value, offset, mask_and_clamp=cute.arch.WARP_SIZE - 1
    )


def shuffle_xor(value: Scalar, mask: int) -> Scalar:
    """``shfl.sync.bfly`` across a full warp: lane ``l`` reads lane ``l ^ mask``.

    Every lane both sends and receives, so a butterfly of ``log2(32)`` rounds
    leaves the total in all 32 lanes with no separate broadcast. The clamp is the
    top lane index, as for :func:`shuffle_down`; zero there confines the exchange
    to lanes below ``mask`` and leaves the rest reading themselves.

    Args:
        value: The value to exchange.
        mask: Lane bit to flip. Compile-time in every caller.

    Returns:
        Lane ``l ^ mask``'s value.
    """
    return cute.arch.shuffle_sync_bfly(
        value, mask, mask_and_clamp=cute.arch.WARP_SIZE - 1
    )


def block_reduce_add(
    value: Scalar, sred: cute.Tensor, tid: cutlass.Int32, threads: int
) -> Scalar:
    """Sum one float32 over a whole block, result in every thread.

    A butterfly inside each warp, one word per warp through shared memory, then
    every thread sums those words. The butterfly leaves the warp total in all 32
    lanes and the final pass is a broadcast read, so no thread is a designated
    reducer and no second barrier is needed to publish the answer.

    The store is unpredicated. All 32 lanes of a warp hold bitwise-identical
    values after the butterfly and write one address, which the ISA resolves to a
    single write by an unspecified lane with no bank conflict, so which lane wins
    cannot change the result. Predicating it on lane zero would add a branch and a
    divergent instruction for no effect.

    Entered by every thread of the block. One barrier, ordering the warp writes
    against the reads. Reusing ``sred`` for a second reduction needs a barrier
    from the caller first, on the same principle as
    :func:`slinoss.ops.so3ssd.cute.prefix.chunk_prefixes`: only the caller knows
    what else is live in shared memory. A caller reducing several values at once
    should widen ``sred`` to one word per warp per value, which pays one barrier
    for all of them.

    Args:
        value: The thread's contribution.
        sred: ``(threads // 32,)`` float32 scratch, or one such row of a wider
            tile.
        tid: Thread index within the block.
        threads: Block width, a multiple of the warp width. Compile-time.

    Returns:
        The block sum, bitwise identical in every thread.
    """
    reach = 1
    while reach < cute.arch.WARP_SIZE:
        value = value + shuffle_xor(value, reach)
        reach *= 2
    sred[tid // cute.arch.WARP_SIZE] = value
    cute.arch.sync_threads()
    # Plain Python loops, not cutlass.range_constexpr: this is an undecorated
    # helper, so the DSL's preprocessor never rewrites its body, and both bounds
    # are compile-time anyway, which is what unrolls them during the trace.
    total = cutlass.Float32(0.0)
    for warp in range(threads // cute.arch.WARP_SIZE):
        total = total + sred[warp]
    return total


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


def sigmoid(value: Scalar) -> Scalar:
    """``sigmoid(value)``, evaluated so no intermediate exceeds one.

    ``e = exp(-|value|)`` lies in ``(0, 1]`` at every finite input, and the result
    is ``1 / (1 + e)`` above zero and ``e / (1 + e)`` below it. The naive form
    exponentiates ``-value`` directly, which overflows to infinity on the negative
    side and then divides infinity by infinity. The absolute value is a select, so
    the two cases cost one predicated move and no divergence.

    Args:
        value: A float32 scalar. Unbounded: the mixer's gate and the block's
            activation both reach the saturated ends.

    Returns:
        The logistic, in ``(0, 1)``.
    """
    positive = value > cutlass.Float32(0.0)
    small = f32(cute.exp2(select(positive, -value, value) * LOG2_E))
    return select(positive, cutlass.Float32(1.0), small) / (small + 1.0)


def silu(value: Scalar, sig: Scalar) -> Scalar:
    """``silu(value) = value * sigmoid(value)``.

    Takes the sigmoid rather than recomputing it: a backward needs both the
    activation and its derivative from one evaluation, and a forward that
    recomputed would be a second implementation.

    Args:
        value: A float32 scalar.
        sig: :func:`sigmoid` of the same value.

    Returns:
        The activation.
    """
    return value * sig


def silu_grad(value: Scalar, sig: Scalar) -> Scalar:
    """``d silu / d value = sigmoid * (1 + value * (1 - sigmoid))``.

    Args:
        value: A float32 scalar.
        sig: :func:`sigmoid` of the same value.

    Returns:
        The derivative.
    """
    return sig * (1.0 + value * (1.0 - sig))
