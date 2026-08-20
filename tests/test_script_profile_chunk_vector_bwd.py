"""Tests for the chunk-vector backward counter driver.

The driver narrows NCU to more than one kernel, because the operator's second
pass reduces the workspace partials the first pass writes and a figure for the
first alone would credit work moved into the second. Everything the driver sums
must therefore stay separated by kernel.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

from slinoss.perf.ncu import NcuInvocation, NcuPass

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


def test_kernel_regex_admits_every_reduce_pass() -> None:
    """The narrowing regex matches every kernel the operator launches.

    The names are the mangled symbols NCU reports, truncated after the first
    operand. A regex written against the Python function name would miss a reduce
    pass, and a pass the regex drops leaves its cost out of the capture entirely
    rather than failing: at the model geometry ``vector_reduce_kernel`` is 432.8 us
    of a 3,811.3 us call, so a silent omission understates the operator by 11%.
    """
    import re

    module = load_driver()
    pattern = re.compile(module.KERNEL)
    assert pattern.search("kernel_cutlass_chunk_vector_bwd_kernel_tensorptrbf16_0")
    assert pattern.search("kernel_cutlass_vector_reduce_kernel_tensorptrf32_0")
    assert pattern.search("kernel_cutlass_reduce_rows_kernel_tensorptrf32_0")
