"""The GEMM census driver: what it attributes a launch to, and what it reports.

The driver's numbers come from a device. The failure mode that matters is a row
whose numbers are not what the hardware did, and there are two ways to reach one:
attribute a launch to the wrong role, or turn a launch's duration into the wrong
per-step figure. Both are host arithmetic and both are pinned here.

The role table is held against a built stack rather than against literals, so a
width or a stored orientation that changes in the library fails here instead of
producing a census that assigns nothing. The stack is built on the host: only the
parameter shapes are read.

The profile is a fake. A real one is a device, a kernel name from cuBLAS and a
recorded shape, and none of the three is what the assignment can get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import pytest
from torch import Tensor

from scripts.perf.gemm_census import (
    DGRAD,
    FWD,
    NN,
    NT,
    TN,
    WGRAD,
    GemmShape,
    census,
    gemm_shapes,
    linear_maps,
)
from slinoss.blocks import SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.perf.ceiling import TensorCeiling
from slinoss.perf.units import Count, Microseconds, Spread, TFlopsPerSecond
from slinoss.stack import SLinOSSStack

CENSUS_CONFIG = SLinOSSConfig(
    d_model=32,
    d_state=48,
    d_head=16,
    n_groups=2,
    chunk_size=16,
    n_layers=2,
    ffn_ratio=2.0,
    vocab_size=17,
)
"""Two layers, so a per-layer call count is not also a total."""

TOKENS = 256
"""``B*T`` for the shape table."""

CEILING = TensorCeiling(
    label="fake",
    flop_count=Count(1),
    duration=Spread.of([Microseconds(1.0)]),
    achieved_tflops=TFlopsPerSecond(100.0),
)
"""A round denominator, so a percentage of it is checkable by hand."""

LAUNCH_US = 10.0
"""Duration of every fabricated launch. One value, so a median is not a question."""

OUT_PROJ_TFLOPS = 0.1048576
"""``2*256*32*64`` flop over :data:`LAUNCH_US`, in TFLOPS. By hand, not by helper."""


def _weights(stack: SLinOSSStack) -> dict[str, Tensor]:
    """The stored weight of every role, by the name the census's table uses.

    Args:
        stack: A built stack.

    Returns:
        Role to the parameter as the module tree stores it, layer zero for a
        per-layer role.
    """
    block = stack.blocks[0]
    assert isinstance(block, SLinOSSBlock)
    assert stack.head is not None
    return {
        "in_proj": block.mixer.in_proj.weight,
        "out_proj": block.mixer.out_proj.weight,
        "ffn_gate": block.ffn_gate.weight,
        "ffn_up": block.ffn_up.weight,
        "ffn_out": block.ffn_out_weight,
        "head": stack.head.weight,
    }


def test_the_role_table_is_the_stacks_own_weights() -> None:
    """Every entry against the parameter it claims, stored orientation included.

    A launch is joined to a role by its extents, so a stale width assigns nothing
    and a stale orientation assigns a real launch to the wrong transpose case. The
    driver's output shows neither: the role reads as missing and the launch as
    unclaimed, and the class total stays plausible.
    """
    stack = SLinOSSStack(CENSUS_CONFIG)
    stored = _weights(stack)
    table = linear_maps(CENSUS_CONFIG)
    assert {one.name for one in table} == set(stored)
    for one in table:
        rows, cols = (
            (one.fan_in, one.fan_out) if one.transposed else (one.fan_out, one.fan_in)
        )
        assert stored[one.name].shape == (rows, cols), one.name
        assert one.call_count == (1 if one.name == "head" else CENSUS_CONFIG.n_layers)


def test_a_transposed_weight_swaps_the_two_forward_transpose_cases() -> None:
    """``ffn_out``, stored ``(I,O)``, against ``ffn_gate``, stored ``(O,I)``.

    The stored orientation fixes all three cases at once, and every shape it
    produces is a shape another map already runs: ``ffn_out``'s dgrad is
    ``ffn_gate``'s forward and its wgrad is ``ffn_gate``'s wgrad, one cuBLAS kernel
    for each pair. Same stage and same shape is one row; same shape at different
    stages is two rows over one launch stream, and a table keyed by shape alone
    loses the stage the row is read for.
    """
    cfg = CENSUS_CONFIG
    table = {(one.label, one.stage): one for one in gemm_shapes(cfg, TOKENS)}
    gate, out = "ffn_gate+ffn_up", "ffn_out"
    assert [table[gate, stage].layout for stage in (FWD, DGRAD)] == [TN, NN]
    assert [table[out, stage].layout for stage in (FWD, DGRAD)] == [NN, TN]
    assert table[out, DGRAD].key == table[gate, FWD].key
    assert table[out, FWD].key == table[gate, DGRAD].key
    # The weight gradient comes out in the stored shape, which is the point of the
    # orientation: it is what decides how many tiles the token reduction covers.
    # Transposed, ffn_out's is the shape the two gate projections already ran, so the
    # three are one row and its launch count is the sum.
    assert (gate, WGRAD) not in table and (out, WGRAD) not in table
    merged = table[f"{gate}+{out}", WGRAD]
    assert merged.layout == NT
    assert (merged.m, merged.n, merged.k) == (cfg.d_ffn, cfg.d_model, TOKENS)
    assert merged.call_count == 3 * cfg.n_layers


class FakeKernel(NamedTuple):
    """What the profiler reports for one device launch."""

    name: str
    duration: float


@dataclass(frozen=True)
class FakeEvent:
    """One ``aten::mm`` and the kernels it launched."""

    name: str
    input_shapes: tuple[tuple[int, ...], ...]
    kernels: tuple[FakeKernel, ...]


@dataclass(frozen=True)
class FakeProfile:
    """A finished profile, as far as the driver's reader of one reads it."""

    recorded: tuple[FakeEvent, ...]

    def events(self) -> tuple[FakeEvent, ...]:
        """Every recorded operator event."""
        return self.recorded


def _profile(shapes: list[GemmShape], iters: int) -> FakeProfile:
    """One ``aten::mm`` event per launch of every shape, all at :data:`LAUNCH_US`.

    The recorded operand shapes are ``(m,k)`` and ``(k,n)`` whatever the transpose
    case: a transposed operand reaches the profiler as the view, and the view is
    conformable. That is why the case has to come off the kernel name.

    Args:
        shapes: What ran.
        iters: Steps in the profile.

    Returns:
        The fake profile.
    """
    events: list[FakeEvent] = []
    for one in shapes:
        kernel = f"ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_{one.layout}"
        events.extend(
            FakeEvent(
                name="aten::mm",
                input_shapes=((one.m, one.k), (one.k, one.n)),
                kernels=(FakeKernel(kernel, LAUNCH_US),),
            )
            for _ in range(one.call_count * iters)
        )
    return FakeProfile(tuple(events))


def test_a_rows_figures_are_the_launches_behind_it() -> None:
    """Per-launch durations into a per-step total, a rate, and a percentage.

    Three ways to report a number the hardware did not produce: divide by the wrong
    launch count, count a shape's launches once for each role that claims it, or
    fold a kernel that is not a tiled GEMM into a row whose flop count is not its
    own. The shared shape here is the one the stack actually produces, and the
    ``gemv`` is what cuBLAS falls back to on a thin extent.
    """
    iters = 3
    table = {(one.label, one.stage): one for one in gemm_shapes(CENSUS_CONFIG, TOKENS)}
    pair = [table["ffn_out", DGRAD], table["ffn_gate+ffn_up", FWD]]
    shared = census(pair, _profile(pair, iters), iters, CEILING)
    assert shared.unassigned == ()
    layers = CENSUS_CONFIG.n_layers
    by_stage = {row.stage: row for row in shared.rows}
    assert by_stage[DGRAD].step_duration_us == pytest.approx(LAUNCH_US * layers)
    assert by_stage[FWD].step_duration_us == pytest.approx(LAUNCH_US * 2 * layers)

    alone = table["out_proj", FWD]
    tiled = _profile([alone], 1)
    with_gemv = FakeProfile(
        (
            *tiled.recorded,
            FakeEvent(
                name="aten::mm",
                input_shapes=((alone.m, alone.k), (alone.k, alone.n)),
                kernels=(FakeKernel("gemv2T_kernel_val", 4.0),),
            ),
        )
    )
    got = census([alone], with_gemv, 1, CEILING)
    assert len(got.rows) == 1
    row = got.rows[0]
    assert row.duration.median_duration_us == pytest.approx(LAUNCH_US)
    assert row.step_duration_us == pytest.approx(LAUNCH_US * layers)
    assert row.achieved_tflops == pytest.approx(OUT_PROJ_TFLOPS)
    assert row.ceiling_pct == pytest.approx(OUT_PROJ_TFLOPS)
    assert any("gemv2T_kernel_val" in line for line in got.unassigned)
