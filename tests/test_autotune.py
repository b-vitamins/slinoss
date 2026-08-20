"""Launch-geometry selection: what it promises when there is nothing to select.

Most of what this module pins is the absence of behaviour. A tuning cache sits on
the launch path of every block kernel, so the properties that matter are that it
changes nothing until it has a measurement, that it cannot break a forward pass
however wrong the file is, and that it costs one memoized lookup rather than one
file read per call. Those are testable and are tested here rather than asserted in
a docstring.

The device identity is faked in the host-only tests. A record is addressed by the
part it was measured on, so a fixture that hardcoded an A6000 would pass on one
host and skip on every other; substituting :func:`slinoss.autotune.device_key`
makes the resolution logic the subject and the part irrelevant.

One positive control runs on a device: every geometry a kernel declares has to
compute what its default computes, to
:data:`slinoss.autotune.AGREEMENT_TOL`. Without it the whole mechanism is a
licence to launch an untested kernel.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

from slinoss import autotune
from slinoss.autotune import (
    AGREEMENT_TOL,
    Attempt,
    DeviceKey,
    Record,
    ShapeKey,
    Variants,
)

requires_cute = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="CuTe DSL not installed"
)

BLOCK_KERNELS = (
    "rmsnorm_fwd",
    "rmsnorm_residual_fwd",
    "rmsnorm_bwd",
    "rmsnorm_residual_bwd",
    "rmsnorm_dweight",
    "swiglu_fwd",
    "swiglu_bwd",
)
"""Every kernel the block tree declares geometries for."""

PART = DeviceKey(name="test part", capability="8.6", sm_count=84)
OTHER = DeviceKey(name="test part", capability="8.6", sm_count=108)
"""One part, and a part differing only in SM count, which is a different launch."""

DEFAULT = (256, 0)
FASTER = (512, 1)
THIRD = (128, 2)
UNDECLARED = (64, 3)

SHAPE = ShapeKey.of(2048, 384, 2)
CALL = (2048, 384, 2, 0)
"""``(rows, width, itemsize, index)`` resolving to :data:`SHAPE`."""

SAMPLES = (9.0, 10.0, 11.0)


def _variants() -> Variants[tuple[int, ...]]:
    """A declaration standing in for a kernel's. Not registered: a test must not
    add a name the tuning driver would then sweep."""
    return Variants(
        kernel="probe", default=DEFAULT, candidates=(DEFAULT, FASTER, THIRD)
    )


def _record(
    geometry: tuple[int, ...] = FASTER,
    *,
    device: DeviceKey = PART,
    shape: ShapeKey = SHAPE,
    kernel: str = "probe",
) -> Record:
    """A record naming ``geometry`` as the winner, with a measurement behind it."""
    return Record(
        kernel=kernel,
        shape=shape,
        device=device,
        winner=Attempt.of(geometry, SAMPLES),
        runners_up=(Attempt.of(DEFAULT, (20.0, 21.0, 22.0)),),
        repeat_count=len(SAMPLES),
        probe_count=3,
        torch_version="2.10.0",
        cutlass_version="4.4.2",
        conditions="clocks unlocked",
    )


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty per-test cache and a fixed device identity.

    The module-level table and memo are process state, so they are cleared on the
    way in and on the way out: a test that installed a record must not decide the
    next test's launches.
    """
    path = tmp_path / "tuning.json"
    monkeypatch.setenv(autotune.CACHE_ENV, str(path))
    monkeypatch.setattr(autotune, "device_key", lambda index: PART)
    autotune.reset()
    yield path
    autotune.reset()


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------


def test_with_no_cache_a_kernel_launches_its_default(cache: Path) -> None:
    """The whole of the zero-cost-absence promise, at the resolution level."""
    variants = _variants()
    assert variants.select(*CALL) == DEFAULT
    assert not cache.exists(), "resolution created a file it was only reading"


def test_the_file_is_read_on_the_first_miss_and_not_before(cache: Path) -> None:
    """Lazy, so an import costs no file IO and a process that never launches a
    tunable kernel never touches the disk."""
    autotune.save([_record()], cache)
    assert not autotune.is_loaded()
    assert _variants().select(*CALL) == FASTER
    assert autotune.is_loaded()


@requires_cute
def test_importing_the_package_reads_no_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a fresh interpreter, so it cannot pass because this one already imported.

    A populated cache is in place: if any module read it at import, the table would
    be loaded before the subprocess asks.
    """
    path = tmp_path / "tuning.json"
    autotune.save([_record()], path)
    monkeypatch.setenv(autotune.CACHE_ENV, str(path))
    probe = textwrap.dedent(
        """
        import slinoss
        from slinoss import autotune
        print(autotune.is_loaded())
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert done.stdout.strip() == "False", done.stdout


def test_a_second_call_at_one_shape_does_not_resolve_again(cache: Path) -> None:
    """The steady-state path is the memo. Only a miss reaches the resolver, so
    ``witnessed`` recording once over two calls is the memoization."""
    autotune.save([_record()], cache)
    variants = _variants()
    variants.select(*CALL)
    with autotune.witnessed() as seen:
        assert variants.select(*CALL) == FASTER
        assert variants.select(*CALL) == FASTER
    assert seen == [("probe", SHAPE)], seen


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_a_measured_record_changes_the_launch(cache: Path) -> None:
    """The positive control for the mechanism: a record is not decoration."""
    autotune.install([_record()])
    assert _variants().select(*CALL) == FASTER
    assert not cache.exists()


def test_the_device_key_holds_no_slot_identity(cache: Path) -> None:
    """Two identical parts in one host share one tuning, so the key names the part
    and not the slot."""
    assert DeviceKey._fields == ("name", "capability", "sm_count")
    autotune.install([_record()])
    variants = _variants()
    rows, width, itemsize, _ = CALL
    assert variants.select(rows, width, itemsize, 0) == FASTER
    assert variants.select(rows, width, itemsize, 1) == FASTER


def test_a_record_from_a_different_part_is_not_used(cache: Path) -> None:
    """An SM count is occupancy arithmetic, so a record measured against another
    one is not evidence about this one."""
    autotune.install([_record(device=OTHER)])
    assert _variants().select(*CALL) == DEFAULT


@pytest.mark.parametrize(
    ("rows", "expected"),
    [(2004, FASTER), (2048, FASTER), (1025, FASTER), (1024, DEFAULT), (4096, DEFAULT)],
)
def test_row_counts_inside_one_bucket_share_a_record(
    cache: Path, rows: int, expected: tuple[int, ...]
) -> None:
    """A grid-strided launch sees ``min(rows, capacity)``, so the exact row count
    is not a resolution axis and the ragged sequence length is not a second
    record."""
    autotune.install([_record()])
    _, width, itemsize, index = CALL
    assert _variants().select(rows, width, itemsize, index) == expected


def test_width_and_itemsize_are_exact(cache: Path) -> None:
    """Both are compile-time arguments of the launch, so neither is bucketed."""
    autotune.install([_record()])
    rows, width, itemsize, index = CALL
    assert _variants().select(rows, width + 1, itemsize, index) == DEFAULT
    assert _variants().select(rows, width, itemsize * 2, index) == DEFAULT


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------


def test_a_geometry_the_kernel_no_longer_declares_is_ignored(cache: Path) -> None:
    """The one check that subsumes a stale file: a rewritten kernel drops a
    geometry and the record addressing it stops being usable."""
    autotune.install([_record(UNDECLARED)])
    assert _variants().select(*CALL) == DEFAULT


def test_a_pin_outside_the_declaration_is_ignored(cache: Path) -> None:
    """A typo in a driver is a fallback, not an unheld launch."""
    variants = _variants()
    with autotune.pinned({"probe": UNDECLARED}):
        assert variants.select(*CALL) == DEFAULT
    with autotune.pinned({"probe": THIRD}):
        assert variants.select(*CALL) == THIRD


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param('{"schema": 1, "records": [', id="truncated"),
        pytest.param("[]", id="not-an-object"),
        pytest.param('{"schema": 1}', id="no-records"),
        pytest.param('{"schema": 99, "records": []}', id="future-schema"),
        pytest.param('{"schema": 1, "records": [{"kernel": "probe"}]}', id="no-fields"),
    ],
)
def test_an_unusable_file_resolves_to_the_default(cache: Path, text: str) -> None:
    """Every way a file can be wrong has one outcome. A cache that can raise on the
    launch path is worse than no cache."""
    cache.write_text(text, encoding="utf-8")
    assert autotune.load(cache) == ()
    assert _variants().select(*CALL) == DEFAULT


def test_statistics_that_their_own_samples_do_not_support_are_dropped(
    cache: Path,
) -> None:
    """A record is a measurement. Hand-editing the median leaves the samples behind
    to contradict it, and the record goes; the file's other records survive, so one
    tampered entry is not a lost cache."""
    autotune.save([_record(), _record(kernel="other")], cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    payload["records"][0]["winner"]["median_duration_us"] = 0.001
    cache.write_text(json.dumps(payload), encoding="utf-8")
    kept = autotune.load(cache)
    assert [r.kernel for r in kept] == ["other"]
    assert _variants().select(*CALL) == DEFAULT


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_record_needs_a_measurement_behind_it() -> None:
    """Three ways to construct provenance with nothing under it, all refused."""
    with pytest.raises(ValueError, match="no samples"):
        Attempt.of(DEFAULT, ())
    with pytest.raises(ValueError, match="behind its winner"):
        Record(
            kernel="probe",
            shape=SHAPE,
            device=PART,
            winner=Attempt.refused(DEFAULT, "would not launch"),
            repeat_count=8,
        )
    with pytest.raises(ValueError, match="repeats"):
        Record(
            kernel="probe",
            shape=SHAPE,
            device=PART,
            winner=Attempt.of(DEFAULT, SAMPLES),
            repeat_count=0,
        )


def test_a_stored_record_carries_its_provenance_and_no_clock(cache: Path) -> None:
    """The stored fields, exactly. No wall-clock stamp: an unlocked-clock host makes
    one misleading, and the repeats it was measured over are the useful thing."""
    autotune.save([_record()], cache)
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema"] == autotune.SCHEMA
    assert set(payload["records"][0]) == {
        "kernel",
        "shape",
        "device",
        "winner",
        "runners_up",
        "repeat_count",
        "probe_count",
        "torch_version",
        "cutlass_version",
        "conditions",
    }
    assert set(payload["records"][0]["winner"]) == {
        "geometry",
        "median_duration_us",
        "min_duration_us",
        "max_duration_us",
        "samples_duration_us",
        "note",
    }
    assert autotune.load(cache) == (_record(),)


def test_re_tuning_one_shape_keeps_the_others_and_their_order(cache: Path) -> None:
    """A sweep of one kernel must not reorder the file or drop what it did not
    measure."""
    first = _record(kernel="a")
    second = _record(kernel="b")
    autotune.save([first, second], cache)
    fresh = _record(THIRD, kernel="a")
    merged = autotune.merge(autotune.load(cache), [fresh, _record(kernel="c")])
    assert [r.kernel for r in merged] == ["a", "b", "c"]
    assert merged[0].winner.geometry == THIRD


def test_witnessed_reports_the_shape_each_kernel_resolved_at(cache: Path) -> None:
    """How the driver learns a key it cannot derive: the tail's row count is the
    grid the reducing kernel ran at. The memo is cleared on entry, so a call site
    already warm in this process is still reported."""
    variants = _variants()
    variants.select(*CALL)
    with autotune.witnessed() as seen:
        variants.select(*CALL)
        variants.select(64, 384, 2, 0)
    assert seen == [("probe", SHAPE), ("probe", ShapeKey.of(64, 384, 2))]


# ---------------------------------------------------------------------------
# The declarations
# ---------------------------------------------------------------------------


@requires_cute
def test_the_block_kernels_declare_the_geometry_they_shipped_with() -> None:
    """An untuned tree launches what it launched before this module existed, which
    means every default is the constant the kernel used to read."""
    from slinoss.ops.block.cute import act, norm

    declared = autotune.registered()
    assert set(BLOCK_KERNELS) <= set(declared)
    assert norm.FWD_VARIANTS.default == norm.NormGeometry(norm.NORM_THREADS, norm.FILL)
    assert norm.BWD_VARIANTS.default == norm.NormGeometry(norm.NORM_THREADS, norm.FILL)
    assert norm.RESIDUAL_FWD_VARIANTS.default == norm.RowGeometry(norm.NORM_THREADS)
    assert norm.RESIDUAL_BWD_VARIANTS.default == norm.NormGeometry(
        norm.NORM_THREADS, norm.FILL
    )
    assert norm.DWEIGHT_VARIANTS.default == norm.DweightGeometry(
        norm.DWEIGHT_THREADS, norm.DWEIGHT_COLS
    )
    shipped = act.ActGeometry(act.ACT_THREADS, act.VECTOR_BYTES, norm.FILL)
    assert act.FWD_VARIANTS.default == shipped
    assert act.BWD_VARIANTS.default == shipped
    for variants in declared.values():
        assert variants.default in variants.candidates


@requires_cute
def test_two_kernels_may_not_share_a_name() -> None:
    """They would share a record, and one of them would launch the other's
    measurement."""
    held = autotune.registered()["swiglu_fwd"]
    autotune.register(held)
    with pytest.raises(ValueError, match="already registered"):
        autotune.register(
            Variants(kernel="swiglu_fwd", default=DEFAULT, candidates=(DEFAULT,))
        )


# ---------------------------------------------------------------------------
# Every variant on a device
# ---------------------------------------------------------------------------

ROWS = 512
WIDTH = 384
HIDDEN = 1024
EPS = 1e-5


def _outputs(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(t for item in value for t in _outputs(item))
    return ()


def _arm(kernel: str, dtype: torch.dtype) -> Callable[[], object]:
    """A callable launching ``kernel``, and the tensors to compare.

    The pullbacks build their graph once, outside the callable, so a candidate and
    the default are pulled back over the same forward.
    """
    from slinoss.ops.block import rmsnorm, rmsnorm_residual, swiglu

    device = torch.device("cuda")

    def make(*size: int, dt: torch.dtype = dtype, grad: bool = False) -> torch.Tensor:
        out = torch.randn(*size, dtype=dt, device=device)
        return out.requires_grad_(True) if grad else out

    if kernel == "rmsnorm_fwd":
        x, weight = make(ROWS, WIDTH), make(WIDTH, dt=torch.float32)
        return lambda: rmsnorm(x, weight, eps=EPS, backend="cute")
    if kernel == "rmsnorm_residual_fwd":
        x = make(ROWS, WIDTH)
        residual = make(ROWS, WIDTH, dt=torch.float32)
        weight = make(WIDTH, dt=torch.float32)
        return lambda: rmsnorm_residual(x, residual, weight, eps=EPS, backend="cute")
    if kernel == "swiglu_fwd":
        gate, up = make(ROWS, HIDDEN), make(ROWS, HIDDEN)
        return lambda: swiglu(gate, up, backend="cute")
    if kernel in ("rmsnorm_bwd", "rmsnorm_dweight"):
        x = make(ROWS, WIDTH, grad=True)
        weight = make(WIDTH, dt=torch.float32, grad=True)
        cot = make(ROWS, WIDTH)
        out = rmsnorm(x, weight, eps=EPS, backend="cute")
        return lambda: torch.autograd.grad(out, (x, weight), cot, retain_graph=True)
    if kernel == "rmsnorm_residual_bwd":
        x = make(ROWS, WIDTH, grad=True)
        residual = make(ROWS, WIDTH, dt=torch.float32, grad=True)
        weight = make(WIDTH, dt=torch.float32, grad=True)
        cots = (make(ROWS, WIDTH), make(ROWS, WIDTH, dt=torch.float32))
        got = rmsnorm_residual(x, residual, weight, eps=EPS, backend="cute")
        return lambda: torch.autograd.grad(
            (got.normed, got.residual),
            (x, residual, weight),
            cots,
            retain_graph=True,
        )
    gate = make(ROWS, HIDDEN, grad=True)
    up = make(ROWS, HIDDEN, grad=True)
    cot = make(ROWS, HIDDEN)
    out = swiglu(gate, up, backend="cute")
    return lambda: torch.autograd.grad(out, (gate, up), cot, retain_graph=True)


@requires_cute
@pytest.mark.cuda
@pytest.mark.cute
@pytest.mark.slow
@pytest.mark.parametrize("kernel", BLOCK_KERNELS)
def test_every_declared_variant_computes_what_the_default_computes(
    kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A geometry says which thread reaches which element and nothing else.

    Held over every candidate, not a sample of them: the tuner is free to select
    any of them on a part nobody has run yet, and a geometry that is only ever
    launched by a cache hit is otherwise untested.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from tests.conftest import assert_max_rel

    dtype = torch.bfloat16
    monkeypatch.setenv(autotune.CACHE_ENV, str(Path("/nonexistent/tuning.json")))
    autotune.reset()
    variants = autotune.registered()[kernel]
    torch.manual_seed(0)
    run = _arm(kernel, dtype)
    with autotune.pinned({kernel: variants.default}):
        want = _outputs(run())
    for geometry in variants.candidates:
        with autotune.pinned({kernel: geometry}):
            got = _outputs(run())
        assert len(got) == len(want)
        for index, (a, b) in enumerate(zip(got, want)):
            assert_max_rel(
                a, b, AGREEMENT_TOL[dtype], f"{kernel} {tuple(geometry)} out{index}"
            )
    autotune.reset()
