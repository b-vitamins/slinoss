"""Tests for the chunk-vector backward counter driver.

The driver narrows NCU to more than one kernel, because the operator's second
pass reduces the workspace partials the first pass writes and a figure for the
first alone would credit work moved into the second. Everything the driver sums
must therefore stay separated by kernel.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import pathlib
import sys
import types

import pytest
import torch

pytest.importorskip("cutlass")

from slinoss.ops.so3ssd.cute.bwd.chunk_vector import chunk_vector_backward
from slinoss.ops.so3ssd.cute.common import WARPS
from slinoss.ops.so3ssd.cute.mma import WARPS_WIDE
from slinoss.perf.ncu import NcuInvocation, NcuPass
from slinoss.perf.workload import shape_by_name

DRIVER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "perf"
    / "profile_chunk_vector_bwd.py"
)


def load_driver() -> types.ModuleType:
    """Import the driver by path, since ``scripts`` is not a package.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location("_profile_chunk_vector_bwd", DRIVER)
    if spec is None or spec.loader is None:
        pytest.fail(f"no import spec for {DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_sectors_separates_kernels() -> None:
    """Two kernels in one pass keep separate local-sector totals.

    Summing them into one figure would let a candidate move spill out of the
    first kernel and into the second at no apparent cost.
    """
    module = load_driver()
    metric = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum"
    one = NcuPass(
        table="local",
        command=(),
        invocations=(
            NcuInvocation(
                launch_id="0", kernel="chunk_vector_bwd_kernel", values={metric: 100.0}
            ),
            NcuInvocation(
                launch_id="1", kernel="chunk_vector_bwd_kernel", values={metric: 40.0}
            ),
            NcuInvocation(
                launch_id="2", kernel="reduce_rows_kernel", values={metric: 7.0}
            ),
        ),
        missing_metrics=(),
    )
    got = module.local_sectors(one)
    assert got == {
        ("chunk_vector_bwd_kernel", metric): 140.0,
        ("reduce_rows_kernel", metric): 7.0,
    }


def test_the_default_groups_is_what_the_operands_are_allocated_at() -> None:
    """``--groups`` defaults to the shape's own ``G``, not to ``H``.

    ``make_inputs`` allocates ``B`` and ``C`` at the shape's ``G`` and the operator
    reads ``G`` off those operands, so a default of ``H`` left the driver reporting
    fold one and a zero workspace for the only shape whose fold is above one while the
    run underneath it summed eighteen heads through a 143.77 MB partial.
    """
    module = load_driver()
    acceptance = shape_by_name("acceptance")
    assert acceptance.groups == 1
    assert acceptance.heads == 18
    assert module.requested_groups(acceptance, None) == 1
    assert module.requested_groups(acceptance, 2) == 2
    standard = shape_by_name("standard")
    assert module.requested_groups(standard, None) == standard.heads


def test_the_default_width_is_the_one_the_operator_ships() -> None:
    """``--warps`` defaults to the operator's own block width, not to the narrow one.

    ``chunk_vector_backward`` ships ``WARPS_WIDE``. A driver defaulting to ``WARPS``
    printed every counter for a 128-thread block while the step launches 256, so the
    duration, the register count and the arena all belonged to a width no caller runs.
    """
    module = load_driver()
    signature = inspect.signature(chunk_vector_backward)
    assert signature.parameters["warps"].default == WARPS_WIDE
    assert module.requested_warps(None) == WARPS_WIDE
    assert module.requested_warps(WARPS) == WARPS
    assert inspect.signature(module.build_runner).parameters["warps"].default == (
        WARPS_WIDE
    )


def test_the_atomic_probe_prices_the_closure_s_own_geometry() -> None:
    """The probe's destination and fold are the ones an atomic close would face.

    The tax it reports is only readable against the pass it would replace, so a
    destination of the wrong extent or a fold of the wrong depth prices a different
    question at the same shape name. The destination is ``dB``'s own ``(B*G*T, 3N)``
    and the fold is the head-sum depth, and both arms move the same bytes, which is
    what makes their difference the atomic and not the traffic.
    """
    module = load_driver()
    shape = dataclasses.replace(
        shape_by_name("acceptance"), name="probe", seq=8, chunk=4, lanes=16, heads=6
    )
    price = module.atomic_price(
        shape, 1, torch.device("cpu"), None, order="blocked", iters=3, warmup=1
    )
    assert price.rows == shape.bsz * shape.seq
    assert price.fold == shape.heads
    assert price.bytes_moved == price.rows * price.fold * shape.d_state * 4
    assert price.atomic_us - price.plain_us == pytest.approx(price.tax_us)
    shallow = module.atomic_price(
        shape, 1, torch.device("cpu"), 2, order="adjacent", iters=3, warmup=1
    )
    assert shallow.fold == 2
    with pytest.raises(ValueError, match="order must be one of"):
        module.atomic_price(shape, 1, torch.device("cpu"), None, order="strided")


def test_kernel_regex_admits_every_reduce_pass() -> None:
    """The narrowing regex matches every kernel the operator launches.

    The names are the mangled symbols NCU reports, truncated after the first
    operand. A regex written against the Python function name would miss a reduce
    pass, and a pass the regex drops leaves its cost out of the capture entirely
    rather than failing: at the model geometry ``vector_reduce_kernel`` is 221.5 us
    of a 3,559.4 us call, so a silent omission understates the operator by 6%.
    """
    import re

    module = load_driver()
    pattern = re.compile(module.KERNEL)
    assert pattern.search("kernel_cutlass_chunk_vector_bwd_kernel_tensorptrbf16_0")
    assert pattern.search("kernel_cutlass_vector_reduce_kernel_tensorptrf32_0")
    assert pattern.search("kernel_cutlass_reduce_rows_kernel_tensorptrf32_0")
