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

The second half of the module is the decode step's own coverage, which the probe
cannot stand in for. What a payload is for there is a CUDA graph: tracing is host
work, a graph records none of it, and a capture that traced runs the traced launch
once at capture time and never on a replay. The decode tests therefore name the four
ways that can be wrong -- a key the build did not export, a payload loaded and
consulted by nothing, a cold process that compiles inside its first capture, and a
replayed entry that is not arithmetically the compiled one -- plus the coverage
question, which is which shapes one payload is a payload for.

The executor cache and the installed payload are process state. Every test that
touches either restores both, so order does not decide an outcome.
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

import cutlass
import cutlass.cute as cute
from torch import Tensor

import slinoss
from slinoss import _cute, aot, graph
from slinoss._cute import Stream, cache_events, compiled_launches, jit_launch
from slinoss.ops.decode import decode_step
from slinoss.ops.decode.backends import CUTE
from slinoss.ops.decode.cute.step import decode_carry, decode_fwd, row_group
from slinoss.ops.decode.reference import TOKENS
from slinoss.ops.so3ssd.cute.common import THREADS
from tests.conftest import LS_BIAS, ScanInputs, make_inputs

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


def _child_env() -> dict[str, str]:
    """The environment a child interpreter needs to import this process's package.

    Returns:
        A copy of the environment with ``PYTHONPATH`` prefixed by the directory
        holding the ``slinoss`` package this process imported.

    The package is not installed in the tree the suite runs from, so a child gets it
    only off ``sys.path``. ``python -c`` puts the working directory there and
    ``python script.py`` puts the script's directory there instead, so a child
    running a script written outside every tree sees no ``slinoss`` at all. The root
    comes off the imported module rather than off ``__file__`` or the working
    directory, because the fact under test is that the child agrees with this
    process, and a second tree would answer a different question.
    """
    env = dict(os.environ)
    root = str(Path(slinoss.__file__).resolve().parents[1])
    held = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{held}" if held else root
    return env


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


# ---------------------------------------------------------------------------
# The decode step
# ---------------------------------------------------------------------------

LANES = 32
"""``N`` the decode payload is built at. ``3N = 96``, a whole-warp row group."""

OTHER_LANES = 48
"""``N`` on the other side of the row-group split. ``3N = 144``, a half-warp group.

Adjacent to :data:`LANES` in the legal set and on the other side of
``N % 32 == 0``, so the pair separates the two packed segment operands rather than
two arbitrary state widths.
"""

DTYPE = torch.bfloat16
"""Activation dtype the decode payload is built at."""

OTHER_DTYPE = torch.float16
"""A second activation dtype in :data:`slinoss._precision.KERNEL_DTYPES`."""

PINNED = torch.float32
"""Dtype of ``trans``, ``K`` and the state at every kernel activation width.

:func:`slinoss.ops.decode.reference.check_operands` pins the state to the call's
pinned dtype, and float64 -- the only other one -- has no kernel path.
"""

DECODE_LAUNCHERS = (decode_fwd, decode_carry)
"""Every ``@cute.jit`` entry point one call of the decode step reaches."""


def _decode_inputs(
    *,
    lanes: int = LANES,
    dtype: torch.dtype = DTYPE,
    bsz: int = 1,
    heads: int = 2,
    groups: int = 1,
    rows: int = 16,
    seed: int = 0,
) -> ScanInputs:
    """One valid decode call, from the real parameter maps, on CUDA.

    Args:
        lanes: ``N``. ``3N`` is the state width.
        dtype: Activation dtype. The pinned tensors and the state stay float32.
        bsz: Batch.
        heads: Heads.
        groups: Groups sharing one ``B``/``C`` pair. Divides ``heads``.
        rows: ``P``.
        seed: Generator seed. The same seed gives the same bytes, which is what the
            cold-start comparison needs.

    Returns:
        A :class:`tests.conftest.ScanInputs` at ``TOKENS`` tokens, so
        :func:`slinoss.ops.decode.decode_step` accepts it.
    """
    return make_inputs(
        bsz=bsz,
        heads=heads,
        groups=groups,
        seqlen=TOKENS,
        rows=rows,
        lanes=lanes,
        dtype=PINNED,
        device="cuda",
        seed=seed,
        w_scale=2.0,
        ls_bias=LS_BIAS,
        u_dtype=dtype,
        bc_dtype=dtype,
    )


def _decode_carries(inp: ScanInputs) -> dict[str, Tensor]:
    """Fresh copies of the three in-place carries, keyed by parameter name."""
    assert inp.z0 is not None and inp.b_prev is not None and inp.u_prev is not None
    return {
        "ssm": inp.z0.clone(),
        "b_prev": inp.b_prev.clone(),
        "u_prev": inp.u_prev.clone(),
    }


def _decode(inp: ScanInputs, carries: dict[str, Tensor]) -> Tensor:
    """One decode step over ``carries``, asserting the kernel answered it.

    The reference backend answers every call a registry with no kernel holds, so a
    payload test that did not check this would pass on a tree where the DSL import
    failed and nothing compiled at all.
    """
    # By name rather than ``**carries``: a str-keyed mapping can reach every
    # keyword-only parameter, ``backend`` included, and a checker says so.
    out = decode_step(
        *inp.args(),
        ssm=carries["ssm"],
        b_prev=carries["b_prev"],
        u_prev=carries["u_prev"],
    )
    assert out.backend == CUTE, f"{out.backend} answered; the kernel did not register"
    return out.y


def _drop_decode_executors() -> None:
    """Forget every executor compiled for a decode launcher.

    Only the decode keys, for the reason :func:`_drop_probe_executors` gives. Also
    what makes a payload observable: :func:`slinoss._cute.jit_launch` consults a
    payload on a cache miss alone, so a process that already holds the executor
    launches through it and a loaded payload is never asked.
    """
    for key in [key for key in _cute._EXECUTORS if key[0] in DECODE_LAUNCHERS]:
        del _cute._EXECUTORS[key]


def _decode_launches() -> tuple[_cute.Compiled, ...]:
    """The decode launchers' entries in the executor cache, and nothing else's."""
    return tuple(item for item in compiled_launches() if item.fn in DECODE_LAUNCHERS)


def _expected_keys(lanes: int = LANES, dtype: torch.dtype = DTYPE) -> tuple[str, str]:
    """The two textual keys one decode call at ``lanes`` and ``dtype`` forms.

    Written out rather than read back from the executor cache. The claim is the map
    from a call to a key, and a key taken from the thing under test satisfies it by
    construction; spelling the signature here is also the enumeration of what a
    payload entry is specialized on, so an argument that gains or loses a
    :class:`cutlass.Constexpr` moves this literal.

    Ranks and element types only. :func:`slinoss._cute.dev_tensor` marks every
    layout dynamic except the leading mode, so no extent, stride or pitch reaches a
    key; a scalar declares nothing and contributes ``None``.

    Args:
        lanes: ``N``.
        dtype: Activation dtype.

    Returns:
        ``(decode_fwd, decode_carry)``, in that order.
    """
    group = row_group(lanes)
    forward = aot.launch_key(
        decode_fwd,
        (THREADS, group, lanes // group),
        (
            dtype, 4,  # U
            PINNED, 4,  # trans
            PINNED, 5,  # K
            dtype, 4,  # B
            dtype, 4,  # C
            PINNED, 4,  # ssm
            dtype, 3,  # b_prev
            dtype, 3,  # u_prev
            dtype, 4,  # y
            None,  # heads_per_group
            None,  # tiles
            None,  # heads
            None,  # bsz
        ),
    )  # fmt: skip
    carry = aot.launch_key(
        decode_carry,
        (THREADS,),
        (
            dtype, 4,  # B
            dtype, 3,  # b_prev
            None,  # width
            None,  # groups
            None,  # bsz
        ),
    )  # fmt: skip
    return forward, carry


@pytest.fixture
def decode_isolated() -> Iterator[None]:
    """Run with the decode executors and the installed payload restored after.

    Dropped on the way in as well as out: an earlier test in the session leaves the
    executors cached, and a cached executor is never matched against a payload.
    """
    _cute.use_payload(None)
    _drop_decode_executors()
    try:
        yield
    finally:
        _cute.use_payload(None)
        _drop_decode_executors()


def _build_decode(path: Path, *inputs: ScanInputs) -> aot.Manifest:
    """Run the decode step on each of ``inputs`` and export what it compiled.

    Args:
        path: Payload directory.
        *inputs: Operand sets to run, one call each.

    Returns:
        The manifest written, holding the decode launches of those calls and nothing
        else.
    """
    _drop_decode_executors()
    for inp in inputs:
        _decode(inp, _decode_carries(inp))
    torch.cuda.synchronize()
    manifest = aot.build(_decode_launches(), path=path)
    _drop_decode_executors()
    return manifest


def test_the_build_discovers_the_decode_launches_and_keys_them(
    tmp_path: Path, decode_isolated: None
) -> None:
    """A decode step must reach a payload with no registration step, under its key.

    Two halves. :func:`slinoss._cute.compiled_launches` is the whole of what a build
    reads, so a launcher it does not report is a kernel no payload can hold however
    the build is invoked -- and the failure is silent, because a short payload is a
    fallback compile rather than an error. And the key is the only thing a launch is
    matched on, so an entry filed under a key no call forms is an entry nothing ever
    loads.
    """
    inp = _decode_inputs()
    _decode(inp, _decode_carries(inp))
    torch.cuda.synchronize()
    discovered = {
        aot.launch_key(item.fn, item.static, item.signature)
        for item in compiled_launches()
    }
    assert set(_expected_keys()) <= discovered

    manifest = _build_decode(tmp_path, inp)
    assert {entry.key for entry in manifest.entries} == set(_expected_keys())
    for entry in manifest.entries:
        assert (tmp_path / entry.file).stat().st_size > 0


def test_a_capture_over_the_decode_step_runs_a_payload_and_compiles_nothing(
    tmp_path: Path, decode_isolated: None
) -> None:
    """A strict payload must serve a capture, and a short one must stop it.

    One fact in two halves, and the second is what makes the first a measurement. A
    capture that compiles records a graph the traced launch is not in, so it replays
    as a no-op over that launch and the output holds whatever the warmup left;
    counting compiles across the capture is the only detector, and a counter that
    cannot go up detects nothing. Removing the forward entry from the payload is the
    injected fault: the raise proves the capture was consulting the payload rather
    than launching through an executor it already held.
    """
    inp = _decode_inputs()
    carries = _decode_carries(inp)
    _build_decode(tmp_path, inp)

    payload = aot.use(tmp_path, strict=True)
    assert payload is not None

    def recorded(u: Tensor) -> Tensor:
        """One decode step over the closed-over operands and carries."""
        return decode_step(
            u,
            inp.trans,
            inp.K,
            inp.B,
            inp.C,
            ssm=carries["ssm"],
            b_prev=carries["b_prev"],
            u_prev=carries["u_prev"],
        ).y

    before = cache_events()
    step = graph.capture(recorded, inp.U)
    step(inp.U)
    torch.cuda.synchronize()
    after = cache_events()
    # Across the warmup, the recording and the replay: nothing traced anywhere. One
    # payload hit per key, since a key reaches the payload once and the cache after.
    assert after.compiled == before.compiled
    assert after.payload_misses == before.payload_misses
    assert after.payload_hits == before.payload_hits + len(DECODE_LAUNCHERS)
    assert step.outputs.shape == inp.U.shape

    # The same capture against a payload the forward entry was removed from.
    forward, _ = _expected_keys()
    body = json.loads((tmp_path / aot.MANIFEST).read_text())
    body["entries"] = [e for e in body["entries"] if e["key"] != forward]
    assert len(body["entries"]) == len(payload) - 1
    (tmp_path / aot.MANIFEST).write_text(json.dumps(body))
    aot.use(tmp_path, strict=True)
    _drop_decode_executors()
    with pytest.raises(KeyError, match="strict"):
        graph.capture(recorded, inp.U)


def test_the_payload_covers_the_free_axes_and_no_other_width_or_dtype(
    tmp_path: Path, decode_isolated: None
) -> None:
    """A payload's coverage must be exactly the axes no key names.

    The honest set, from :func:`_expected_keys`: one entry pair per
    ``(activation dtype, N)``. ``B``, ``H``, ``G`` and ``P`` are dynamic arguments
    that declare nothing, and no extent, stride or pitch reaches a key, so one pair
    serves every batch, head count, grouping, row count and vector layout at that
    dtype and width. ``N`` and the activation dtype are not free: ``N`` sets the row
    group and the lane walk, whose packed segment operands differ, and the dtype sets
    the element width every load widens from.

    Both halves matter and neither implies the other. Coverage claimed and not held
    is a fallback compile on a shape the build thought it had covered; coverage held
    and not claimed is a payload built over a shape list nobody needed.
    """
    built = _decode_inputs()
    manifest = _build_decode(tmp_path, built)
    assert len(manifest.entries) == len(DECODE_LAUNCHERS)

    aot.use(tmp_path, strict=True)
    # Every axis no key names, moved at once and away from the built shape.
    free = _decode_inputs(bsz=3, heads=4, groups=2, rows=32, seed=1)
    before = cache_events()
    _decode(free, _decode_carries(free))
    torch.cuda.synchronize()
    served = cache_events()
    assert served.compiled == before.compiled
    assert served.payload_hits == before.payload_hits + len(DECODE_LAUNCHERS)

    for outside in (
        _decode_inputs(lanes=OTHER_LANES),
        _decode_inputs(dtype=OTHER_DTYPE),
    ):
        _drop_decode_executors()
        with pytest.raises(KeyError, match="strict"):
            _decode(outside, _decode_carries(outside))


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

COLD_SCRIPT = '''\
"""Load a payload, capture one decode step, replay it. A child process.

Written by ``tests/test_aot.py`` outside every tree and run as
``python3 cold.py PAYLOAD OPERANDS OUT``. Nothing runs the step before the capture,
so the payload is the only thing that can keep the capture from tracing.
"""

import json
import sys

import torch

from slinoss import aot, graph
from slinoss._cute import cache_events
from slinoss.ops.decode import decode_step

payload_dir, operands, out = sys.argv[1:4]
held = torch.load(operands, map_location="cuda", weights_only=True)
U, trans, K, B, C = (held[name] for name in ("U", "trans", "K", "B", "C"))
start = {name: held[name] for name in ("ssm", "b_prev", "u_prev")}
carries = {name: value.clone() for name, value in start.items()}

payload = aot.use(payload_dir, strict=True)
assert payload is not None, f"no payload at {payload_dir}"
before = cache_events()
step = graph.capture(lambda u: decode_step(u, trans, K, B, C, **carries), U)
# The warmup and the recording each advanced the carries. Rewind them in place, in
# the buffers the graph holds the addresses of, so the replay below is the first
# token from the state the parent steps from.
for name, value in start.items():
    carries[name].copy_(value)
result = step(U)
torch.cuda.synchronize()
after = cache_events()

torch.save(
    {"y": result.y.clone(), **{n: v.clone() for n, v in carries.items()}}, out
)
print(
    json.dumps(
        {
            "entries": len(payload),
            "backend": result.backend,
            "compiled": after.compiled - before.compiled,
            "compile_us": after.compile_us - before.compile_us,
            "payload_hits": after.payload_hits - before.payload_hits,
            "payload_misses": after.payload_misses - before.payload_misses,
        }
    )
)
'''
"""The cold-start child, as source.

A file rather than ``-c``: it needs a docstring and a module body, and a traceback
from it should name a line.
"""


class ColdRun(NamedTuple):
    """One cold capture and the eager step it is compared against.

    Attributes:
        events: The child's :func:`slinoss._cute.cache_events` deltas and the
            backend that answered, as JSON.
        replayed: ``y`` and the three carries after the child's replay, on CUDA.
        eager: The same four, from one step in this process with no payload loaded
            and no executor held. The comparison is exact rather than bounded: a
            payload entry is the object file that compile emitted, so the two runs
            are one program over one set of bytes.
    """

    events: dict[str, object]
    replayed: dict[str, Tensor]
    eager: dict[str, Tensor]


@pytest.fixture(scope="module")
def cold_run(tmp_path_factory: pytest.TempPathFactory) -> ColdRun:
    """Build a decode payload, replay it in a fresh process, and step eagerly here.

    Module-scoped because the child is a whole interpreter and torch import, and the
    two facts it establishes -- that a cold capture compiles nothing, and that its
    replay is arithmetically the compiled kernel -- are read off one run of it.

    The payload the parent installs and the executors it holds are restored before
    the child starts, so nothing here decides what a later test sees.
    """
    root = tmp_path_factory.mktemp("decode-cold")
    payload_dir, operands, replayed = (
        root / "payload",
        root / "operands.pt",
        root / "replayed.pt",
    )
    script = root / "cold.py"
    script.write_text(COLD_SCRIPT)

    inp = _decode_inputs()
    installed = _cute._PAYLOAD
    try:
        _cute.use_payload(None)
        _build_decode(payload_dir, inp)
        torch.save(
            {
                "U": inp.U,
                "trans": inp.trans,
                "K": inp.K,
                "B": inp.B,
                "C": inp.C,
                **_decode_carries(inp),
            },
            operands,
        )
        # The reference: no payload installed and no executor held, so this step
        # compiles its own and the comparison is against a compile rather than
        # against another read of the same file.
        carries = _decode_carries(inp)
        y = _decode(inp, carries)
        torch.cuda.synchronize()
        eager = {"y": y.clone(), **{n: v.clone() for n, v in carries.items()}}
    finally:
        _cute.use_payload(installed)
        _drop_decode_executors()

    done = subprocess.run(
        [sys.executable, str(script), str(payload_dir), str(operands), str(replayed)],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(),
    )
    assert done.returncode == 0, f"the cold child failed:\n{done.stdout}\n{done.stderr}"
    return ColdRun(
        events=json.loads(done.stdout.strip().splitlines()[-1]),
        replayed=torch.load(replayed, map_location="cuda", weights_only=True),
        eager=eager,
    )


def test_a_cold_process_captures_the_decode_step_without_compiling(
    cold_run: ColdRun,
) -> None:
    """The case the payload exists for: capture is the first thing the process does.

    Every other test here runs a step before it captures, so the executor cache is
    warm and a capture would not trace whatever the payload does. A generation
    process does not: it loads, it captures, it replays. If the payload does not
    cover the step, the trace happens inside the recording, is not recorded, and the
    graph replays without the launch it traced.
    """
    assert cold_run.events["backend"] == CUTE
    assert cold_run.events["entries"] == len(DECODE_LAUNCHERS)
    assert cold_run.events["compiled"] == 0
    assert cold_run.events["compile_us"] == 0.0
    assert cold_run.events["payload_misses"] == 0
    assert cold_run.events["payload_hits"] == len(DECODE_LAUNCHERS)


def test_the_replayed_decode_step_is_the_compiled_one_bit_for_bit(
    cold_run: ColdRun,
) -> None:
    """A replayed payload entry must equal a compile exactly, on every buffer.

    Exactly, not closely. The object file is what that compile emitted, so any gap
    at all is one of the substitutions the payload makes going wrong -- the wrong
    entry matched, the JIT link resolving a different symbol, or a graph replaying
    over an address the recording did not hold -- and a tolerance would hide each of
    them. The state buffers are checked beside ``y`` because a step whose output is
    right and whose carry is wrong reads correct for exactly one token.
    """
    assert set(cold_run.replayed) == set(cold_run.eager)
    for name, want in cold_run.eager.items():
        got = cold_run.replayed[name]
        assert got.dtype is want.dtype and got.shape == want.shape, name
        assert torch.equal(got, want), name


# ---------------------------------------------------------------------------
# Where a payload lives
# ---------------------------------------------------------------------------


def test_the_payload_directory_is_inside_the_package_and_is_not_committed() -> None:
    """A payload must sit where an install can carry it, and never reach a commit.

    One placement decision with two halves. Outside the package the path is relative
    to nothing an installed wheel knows, so no install could hold a payload at all.
    Committed, the entries are architecture- and DSL-version-specific binaries, and a
    checkout on another part loads one built for a card it does not have --
    :func:`slinoss.aot.load` refuses it by field, so the tree would carry a file
    whose only effect is to raise.
    """
    package = Path(slinoss.__file__).resolve().parent
    assert aot.PAYLOAD_DIR.parent == package
    ignored = (package.parent / ".gitignore").read_text().splitlines()
    assert f"{package.name}/{aot.PAYLOAD_DIR.name}/" in ignored


def test_the_package_imports_with_a_payload_present_and_absent(
    decode_isolated: None,
) -> None:
    """Importing the package must not depend on whether a payload is there.

    A payload is loaded by an explicit :func:`slinoss.aot.use`, so the import may
    neither look for one nor refuse one. An import that did would turn a payload left
    inside the package by an earlier build into an ImportError on any host that only
    wanted the reference path, and that directory is exactly what a build leaves
    behind.
    """
    if aot.PAYLOAD_DIR.exists():
        pytest.skip(f"{aot.PAYLOAD_DIR} already holds a payload; not overwriting it")
    probe = [sys.executable, "-c", "import slinoss; print(slinoss.__version__)"]
    env = _child_env()
    absent = subprocess.run(probe, capture_output=True, text=True, check=False, env=env)
    assert absent.returncode == 0, absent.stderr

    inp = _decode_inputs()
    try:
        _build_decode(aot.PAYLOAD_DIR, inp)
        assert (aot.PAYLOAD_DIR / aot.MANIFEST).is_file()
        present = subprocess.run(
            probe, capture_output=True, text=True, check=False, env=env
        )
    finally:
        shutil.rmtree(aot.PAYLOAD_DIR, ignore_errors=True)
    assert present.returncode == 0, present.stderr
    assert present.stdout == absent.stdout
