"""The shared row reduction of a per-block partial buffer.

The authority is the same sum in float64. A float32 reference would share the
kernel's rounding only by accident, since torch picks its own reduction order, so
it could not tell a wrong order from a different one.

Two properties are pinned bitwise instead of to a tolerance, because at one shape
the kernel does one fixed chain of adds and one store: a rerun reproduces its own
result, and a narrowed destination differs from the float32 one by exactly the
store's rounding.

The axes swept are the row extent against the slot count and the width against the
column tile, one at a time. The ragged width is the case the clamped read exists
for: a lane past the width reads a column that is in bounds and stores nothing.
"""

import pytest
import torch
from torch import Tensor

pytest.importorskip("cutlass.cute", reason="CuTe DSL not installed")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from collections.abc import Callable

from slinoss._cute import assert_smem_fits, smem_bytes, smem_capacity
from slinoss._guard import SECTOR_BYTES
from slinoss._precision import KERNEL_DTYPES
from slinoss._reduce import (
    REDUCE_COLS,
    REDUCE_THREADS,
    reduce_partials,
    slot_smem_bytes,
    slot_tile,
)
from tests.conftest import assert_max_rel

pytestmark = [pytest.mark.cuda, pytest.mark.cute]

SLOTS = REDUCE_THREADS // REDUCE_COLS

# A sum of zero-mean terms is smaller than the terms, so the reference magnitude
# this is taken against is the sum's own.
REDUCE_TOL = 1e-6


def _partial(slabs: int, rows: int, width: int, seed: int = 0) -> Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        slabs, rows, width, generator=generator, device="cuda", dtype=torch.float32
    )


@pytest.mark.parametrize(
    "rows",
    # Below one slot each, exactly one, one plus a remainder, and a frontier-sized
    # extent: the slot loop's trip count is 0, 1, 2 and many.
    [1, SLOTS - 1, SLOTS, SLOTS + 1, 2 * SLOTS + 3, 2004],
)
def test_rows_parity(rows: int) -> None:
    """A row extent that does not divide the slot count still sums every row."""
    partial = _partial(1, rows, 120, seed=rows)
    got = reduce_partials(partial)
    assert got.shape == (1, 120)
    assert got.dtype is torch.float32
    assert_max_rel(got, partial.double().sum(1), REDUCE_TOL, f"reduce rows={rows}")


@pytest.mark.parametrize(
    "width",
    # One lane of one tile, a whole tile, a tile plus one lane, and both call sites'
    # widths: 120 is ragged against 8 and 576 is not.
    [1, REDUCE_COLS, REDUCE_COLS + 1, 120, 576],
)
def test_width_parity(width: int) -> None:
    """A width that does not fill its last column tile is neither clipped nor fed
    the clamped column the out-of-range lanes read."""
    partial = _partial(1, 257, width, seed=width)
    got = reduce_partials(partial)
    assert got.shape == (1, width)
    assert_max_rel(got, partial.double().sum(1), REDUCE_TOL, f"reduce width={width}")


def test_slabs_are_independent() -> None:
    """Each slab reduces its own rows: a dropped slab index would sum the first."""
    partial = _partial(3, 65, 120, seed=3)
    got = reduce_partials(partial)
    assert got.shape == (3, 120)
    assert_max_rel(got, partial.double().sum(1), REDUCE_TOL, "reduce slabs")
    # The slabs must also differ, or agreement above would prove nothing.
    assert not torch.equal(got[0], got[1])


def test_rerun_is_bitwise() -> None:
    """No atomics and a fixed launch geometry: one shape reproduces itself."""
    partial = _partial(2, 511, 120, seed=7)
    assert torch.equal(reduce_partials(partial), reduce_partials(partial))


@pytest.mark.parametrize("dtype", KERNEL_DTYPES)
def test_narrowing_store_is_the_only_rounding(dtype: torch.dtype) -> None:
    """A narrow destination carries the float32 total rounded once, on the store.

    The reference is this kernel at float32, not torch's sum: what is under test is
    the store, and a second reduction order would confound it.
    """
    partial = _partial(2, 300, 120, seed=11)
    got = reduce_partials(partial, out_dtype=dtype)
    assert got.dtype is dtype
    assert torch.equal(got, reduce_partials(partial).to(dtype))


def test_out_is_written_in_full_and_returned() -> None:
    """A supplied destination is the result object, and no element of it survives."""
    partial = _partial(1, 300, 120, seed=13)
    out = torch.full((1, 120), float("nan"), dtype=torch.float32, device="cuda")
    got = reduce_partials(partial, out=out)
    assert got is out
    assert torch.equal(got, reduce_partials(partial))


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda p: reduce_partials(p[0]), ValueError, "must be"),
        (
            lambda p: reduce_partials(p.new_empty(1, 0, 120)),
            ValueError,
            "at least one element",
        ),
        (lambda p: reduce_partials(p.double()), TypeError, "float32"),
        (
            lambda p: reduce_partials(p, out=p.new_empty(1, 8)),
            ValueError,
            "out must be",
        ),
        (
            lambda p: reduce_partials(p, out=p.new_empty(1, 120).double()),
            TypeError,
            "kernel dtypes",
        ),
        (
            lambda p: reduce_partials(p, out=p.new_empty(1, 120), out_dtype=p.dtype),
            ValueError,
            "one or the other",
        ),
        (lambda p: reduce_partials(p.transpose(1, 2)), ValueError, "contiguous"),
        (lambda p: reduce_partials(p.cpu()), ValueError, "CUDA"),
    ],
)
def test_rejections(
    call: Callable[[Tensor], Tensor], error: type[Exception], match: str
) -> None:
    """Every raise on the host path is reachable, and none of them is the kernel
    reading past an operand."""
    with pytest.raises(error, match=match):
        call(_partial(1, 33, 120))


def test_geometry_and_shared_memory() -> None:
    """The block is a whole number of warps and of column tiles, the column tile is
    one sector, and the accumulator tile fits with room to spare."""
    assert REDUCE_THREADS % 32 == 0
    assert REDUCE_THREADS % REDUCE_COLS == 0
    assert REDUCE_COLS * 4 == SECTOR_BYTES
    assert slot_tile(REDUCE_THREADS).shape == (REDUCE_THREADS,)
    budget = slot_smem_bytes(REDUCE_THREADS)
    assert budget == smem_bytes([(slot_tile(REDUCE_THREADS), 4)])
    assert budget == 4 * REDUCE_THREADS
    assert assert_smem_fits("reduce_rows", budget) == budget
    assert budget < smem_capacity()
