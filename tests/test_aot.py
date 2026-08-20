"""Ahead-of-time payloads: identity, keying, and what a loaded entry runs.

A payload replaces a compile with a file read, so every test here names one way
that substitution can be wrong rather than slow. The identity tests cover a payload
accepted when it should not be; the key tests cover a launch matched to the wrong
entry or to none; the launch test covers an entry that loads and computes something
else; the counter test covers a compile that happens without being reported, which
is what would make a cold-start number a lie.

One probe kernel computes ``out = a - b * n`` over three tensors, with ``n`` a
:class:`cutlass.Constexpr`, so two specializations of one launcher exist and a
payload built for one must not serve the other. The result is a function of ``n``,
so an entry matched to the wrong specialization shows up in the output.

The executor cache and the installed payload are process state. Every test that
touches either restores both, so order does not decide an outcome.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from torch import Tensor

from slinoss import _cute, aot, graph
from slinoss._cute import Stream, cache_events, compiled_launches, jit_launch

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

N = 128
"""Probe length. One block, one thread per element."""

ARCH = "sm_86"
"""A fixed architecture for the identity tests, so they do not depend on the part."""


@cute.kernel
def _scale_kernel(
    ga: cute.Tensor, gb: cute.Tensor, gout: cute.Tensor, n: cutlass.Constexpr
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    gout[tid] = ga[tid] - gb[tid] * cutlass.Float32(n)


@cute.jit
def _scale_launch(
    ga: cute.Tensor,
    gb: cute.Tensor,
    gout: cute.Tensor,
    stream: Stream,
    n: cutlass.Constexpr,
    threads: cutlass.Constexpr,
) -> None:
    _scale_kernel(ga, gb, gout, n).launch(
        grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
    )


def _scale(a: Tensor, b: Tensor, n: int) -> Tensor:
    """``a - b * n`` through the shipped launch path."""
    out = torch.zeros_like(a)
    jit_launch(_scale_launch, (a, b, out), (n, a.numel()))
    torch.cuda.synchronize()
    return out


def _other_launch(*args: object) -> None:
    """A second launcher name for the key test. Never launched."""


def _operands() -> tuple[Tensor, Tensor]:
    a = torch.arange(N, dtype=torch.float32, device="cuda")
    return a, a * 3.0 + 1.0


@pytest.fixture
def isolated() -> Iterator[None]:
    """Run with this launcher's executors and the installed payload restored after.

    Only this launcher's keys are dropped, not the whole cache: a test that emptied
    it would make every later test in the session recompile.
    """
    _cute.use_payload(None)
    try:
        yield
    finally:
        _cute.use_payload(None)
        _drop_probe_executors()


def _drop_probe_executors() -> None:
    """Forget every executor compiled for the probe launcher."""
    for key in [key for key in _cute._EXECUTORS if key[0] is _scale_launch]:
        del _cute._EXECUTORS[key]


def _probe_launches() -> tuple[_cute.Compiled, ...]:
    """The probe launcher's entries in the executor cache, and nothing else's."""
    return tuple(item for item in compiled_launches() if item.fn is _scale_launch)


def _build(path: Path, *specs: int) -> aot.Manifest:
    """Compile the probe at each ``n`` and export a payload holding exactly those.

    The executors are dropped afterwards as well as before, so no test launches
    through an executor another test exported.
    """
    a, b = _operands()
    _drop_probe_executors()
    for n in specs:
        _scale(a, b, n)
    manifest = aot.build(_probe_launches(), path=path)
    _drop_probe_executors()
    return manifest


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_module_id_is_the_same_in_another_process() -> None:
    """An id that depends on the process cannot gate a payload built by another one.

    The build runs in one process and the load in the next, so an id built from
    anything the interpreter chooses per run -- a dict order, an ``id()``, a hash
    seed -- would refuse every payload it produced.
    """
    here = aot.module_id(arch=ARCH)
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from slinoss import aot; print(aot.module_id(arch={ARCH!r}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout.strip() == here


def test_module_id_covers_kernel_source_and_not_host_source(tmp_path: Path) -> None:
    """One changed kernel byte must refuse a stale payload; a host edit must not.

    The two halves are one fact. An id blind to a kernel edit runs yesterday's cubin
    against today's launcher, which is a wrong result rather than a slow one. An id
    that moves on any edit anywhere invalidates the payload on every commit, and a
    payload nobody can keep valid is not used.
    """
    kernel = tmp_path / "ops" / "so3ssd" / "cute" / "fwd" / "chunk.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("x = 1\n")
    (tmp_path / "_cute.py").write_text("y = 2\n")
    before = aot._source_digest(tmp_path)

    (tmp_path / "config.py").write_text("z = 3\n")
    (tmp_path / "ops" / "so3ssd" / "forward.py").write_text("w = 4\n")
    assert aot._source_digest(tmp_path) == before

    kernel.write_text("x = 2\n")
    assert aot._source_digest(tmp_path) != before


def test_a_payload_from_another_tree_is_refused_by_field(tmp_path: Path) -> None:
    """A refusal must name what differs, and must not fire on a matching payload.

    A payload is refused far from where it was built. ``the payload does not match``
    sends a reader to the wrong file; the field says whether a kernel changed, the
    DSL moved, or the object is for another part.
    """
    live = aot.identity()
    _build(tmp_path, 3)
    assert aot.load(tmp_path).identity == live

    for field, value in (("source_digest", "0" * 64), ("arch", "sm_1")):
        body = json.loads((tmp_path / aot.MANIFEST).read_text())
        body["identity"] = {**live._asdict(), field: value}
        (tmp_path / aot.MANIFEST).write_text(json.dumps(body))
        with pytest.raises(ValueError, match=field):
            aot.load(tmp_path)


def test_a_manifest_of_another_format_is_refused(tmp_path: Path) -> None:
    """A manifest this reader does not understand is an error, not an empty payload.

    Silently reading zero entries out of a newer manifest turns a version skew into
    a process that compiles everything and reports nothing wrong.
    """
    _build(tmp_path, 3)
    body = json.loads((tmp_path / aot.MANIFEST).read_text())
    body["format"] = aot.FORMAT + 1
    (tmp_path / aot.MANIFEST).write_text(json.dumps(body))
    with pytest.raises(ValueError, match="format"):
        aot.load(tmp_path)


# ---------------------------------------------------------------------------
# The textual key
# ---------------------------------------------------------------------------


def test_the_textual_key_separates_what_the_live_key_separates() -> None:
    """Two launches that compile different code must not share one text key.

    The live key holds a function object, a tuple of compile-time arguments and what
    the runtime arguments declared. The text form must collide on none of them: a
    collision hands a launch the wrong specialization's cubin, and nothing downstream
    can tell.
    """
    keys = {
        aot.launch_key(_scale_launch, (3, N), (torch.float32, 1)),
        aot.launch_key(_scale_launch, (5, N), (torch.float32, 1)),
        aot.launch_key(_scale_launch, (3, N), (torch.bfloat16, 1)),
        aot.launch_key(_scale_launch, (3, N), (torch.float32, 2)),
        aot.launch_key(_scale_launch, (3, N), (torch.float32, 1, None)),
        aot.launch_key(_other_launch, (3, N), (torch.float32, 1)),
    }
    assert len(keys) == 6
    assert aot.launch_key(_scale_launch, (3, N), ()) == aot.launch_key(
        _scale_launch, (3, N), ()
    )
    # A flag and the integer it is stored as are different specializations.
    assert aot.launch_key(_scale_launch, (True,), ()) != aot.launch_key(
        _scale_launch, (1,), ()
    )
    assert aot.launch_key(_scale_launch, (cutlass.Float32,), ()) != aot.launch_key(
        _scale_launch, (cutlass.BFloat16,), ()
    )


def test_a_key_with_no_text_form_raises() -> None:
    """A key that cannot be written down is a bug, not a payload miss.

    Falling back would be silent: the kernel would compile on every cold start and
    the miss would read as a payload that simply does not cover it, so a new
    compile-time argument type could remove the payload's whole point without
    failing anything.
    """
    with pytest.raises(TypeError, match="no text form"):
        aot.launch_key(_scale_launch, (object(),), ())
    with pytest.raises(TypeError, match="no text form"):
        aot.launch_key(_scale_launch, (), (torch.zeros(1),))


# ---------------------------------------------------------------------------
# Launching from a payload
# ---------------------------------------------------------------------------


def test_a_payload_entry_computes_what_the_compile_computed(
    tmp_path: Path, isolated: None
) -> None:
    """A loaded entry must be the same kernel, bit for bit.

    This is the whole claim. The object file was written by a compile in another
    process, is JIT-linked rather than assembled, and is called with a stream that
    was not the one it was compiled against; any of those going wrong shows up here
    as a different number.
    """
    a, b = _operands()
    expected = _scale(a, b, 3)
    _build(tmp_path, 3)
    payload = aot.use(tmp_path, strict=True)
    assert payload is not None
    before = cache_events()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        got = _scale(a, b, 3)
    after = cache_events()

    assert torch.equal(got, expected)
    assert after.compiled == before.compiled
    assert after.payload_hits == before.payload_hits + 1


def test_a_strict_payload_raises_on_a_key_it_does_not_hold(
    tmp_path: Path, isolated: None
) -> None:
    """A miss under strict must fail, and under the default must compile and count.

    A benchmark that silently compiles reports a first step two orders of magnitude
    slow and calls it the kernel. A process that must not stall reads the miss
    counter instead.
    """
    a, b = _operands()
    _build(tmp_path, 3)
    aot.use(tmp_path, strict=True)
    with pytest.raises(KeyError, match="strict"):
        _scale(a, b, 5)

    aot.use(tmp_path, strict=False)
    before = cache_events()
    assert torch.equal(_scale(a, b, 5), a - b * 5.0)
    after = cache_events()
    assert after.payload_misses == before.payload_misses + 1
    assert after.compiled == before.compiled + 1


def test_the_counters_separate_a_compile_from_a_reuse(isolated: None) -> None:
    """The compile count must move once per key and never again.

    Deliverable of the counters themselves: a cold-start number is the difference
    between a pass that compiles and a pass that does not, so a counter that missed
    a compile, or charged a cached launch for one, would make that number up.
    """
    a, b = _operands()
    _drop_probe_executors()

    cold = cache_events()
    _scale(a, b, 7)
    once = cache_events()
    assert once.compiled == cold.compiled + 1
    assert once.compile_us > cold.compile_us

    _scale(a, b, 7)
    twice = cache_events()
    assert twice.compiled == once.compiled
    assert twice.hits == once.hits + 1
    assert twice.compile_us == once.compile_us


def test_reset_leaves_the_executors_alone(isolated: None) -> None:
    """Resetting the counters must not read as an emptied cache.

    The counters are a measurement window and the cache is state. A reset that
    dropped executors would make a second window inside one process compile again,
    which is the opposite of what the window is for.
    """
    a, b = _operands()
    _scale(a, b, 11)
    held = _cute.executor_count()
    _cute.reset_cache_events()
    assert cache_events().compiled == 0
    assert _cute.executor_count() == held
    _scale(a, b, 11)
    assert cache_events().compiled == 0
    assert cache_events().hits == 1


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def test_a_capture_that_compiles_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing inside a capture must fail rather than record a graph missing a launch.

    Host work is not recorded, so the traced launch happens once, at capture time,
    and never on a replay. The graph then replays without it and the output holds
    whatever the warmup left, which no comparison against an eager step catches
    reliably.
    """
    counter = iter((0, 1))
    monkeypatch.setattr(graph, "_compiled", lambda: next(counter))
    with pytest.raises(RuntimeError, match="compiled 1 executors"):
        graph.capture(lambda x: x + 1.0, torch.zeros(4, device="cuda"))
