"""Launch and traffic census of the routed one-token step, stage by stage.

The recurrence kernel is closed at its own roofline. What is not known is what the
rest of the routed step costs, so a fusion decision rests on a number rather than
on which boundary looks untidy.

The subject is one :meth:`slinoss.mixer.SLinOSSMixer.step` at ``T == 1``: input
projection, value convolution, key convolution, parameter frontier, recurrence,
tail, output projection, and the two window carries. That is one layer, so every
launch figure here is per layer by construction and no depth divides it.

:func:`stage_program` reproduces that step stage by stage, through the same
dispatch entry points in the same order. It is a harness and not a second
implementation: ``tests/test_script_decode_census.py`` holds its output against
:meth:`slinoss.mixer.SLinOSSMixer.step` on the same inputs and the same state, so
a stage that drifted from the routed program fails the suite instead of
mis-attributing a kernel.

Three passes, because they take three instruments and a contended sample voids
only its own:

``provenance``
    Which tree, which torch, which backend answered every registry, what the card
    was doing, and whether the steady-state step keeps the hard contract.
    ``decode`` resolving to ``reference`` voids the census, so it is printed and
    refused rather than assumed.
``launch``
    Nsight Systems over the whole step. Per-kernel launch count, duration, share,
    the copy count the two carries land in, and the gap census that per-launch idle
    comes off.
``traffic``
    Nsight Compute over each stage. Measured DRAM bytes against the compulsory
    figure :class:`StageOperands` states, and against a DRAM floor fitted in the
    same process at each kernel's own footprint.

``target`` is the profiled child the other two attach to.

Every pass writes one JSON artifact per cell and skips a cell already banked under
``--resume``, so a short quiet window banks rows instead of losing them.

    python3 scripts/perf/decode_census.py --pass provenance --shape acceptance
    python3 scripts/perf/decode_census.py --pass launch --shape acceptance --batch 128
    python3 scripts/perf/decode_census.py --pass traffic --shape acceptance --batch 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor
from torch.nn.functional import linear

from slinoss._precision import cast_opt, cast_to
from slinoss.mixer import SLinOSSMixer
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.decode import TOKENS, decode_step
from slinoss.ops.decode import backends as decode_dispatch
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.perf.capture import profiler_window
from slinoss.perf.ceiling import DramTimeFloor, dram_floor_verdict, dram_time_floor
from slinoss.perf.device import (
    ClockPolicy,
    Contention,
    DeviceInfo,
    clock_policy,
    contention,
    device_info,
    device_ordinal,
    require_cuda,
)
from slinoss.perf.ncu import NcuPass, NcuTable, run_ncu
from slinoss.perf.nsys import (
    nsys_report_texts,
    occupancy,
    parse_gpu_events,
    parse_gpu_trace,
)
from slinoss.perf.timing import measure, on_device
from slinoss.perf.tools import resolve_tool
from slinoss.perf.units import (
    Bytes,
    Count,
    GBPerSecond,
    Microseconds,
    Percent,
    Ratio,
    gbs_from_bytes_us,
)
from slinoss.perf.workload import decode_shape_by_name, layer_config
from slinoss.state import MixerState

DTYPES: Final[dict[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

PASSES: Final[tuple[str, ...]] = ("provenance", "launch", "traffic", "target")

IN_PROJ: Final = "in_proj"
VALUE_CONV: Final = "value_conv"
KEY_CONV: Final = "key_conv"
PREP: Final = "prep"
RECURRENCE: Final = "recurrence"
TAIL: Final = "tail"
OUT_PROJ: Final = "out_proj"
CARRY_CONV: Final = "carry_conv"
CARRY_KEYS: Final = "carry_keys"

ALL_STAGES: Final = "all"

STAGE_ORDER: Final[tuple[str, ...]] = (
    IN_PROJ,
    VALUE_CONV,
    KEY_CONV,
    PREP,
    RECURRENCE,
    TAIL,
    OUT_PROJ,
    CARRY_CONV,
    CARRY_KEYS,
)
"""Call order, which is the order :meth:`SLinOSSMixer.step` runs them in.

The two carries land after the output projection because the step advances each
window only after its last read of the window that one replaces."""

FUSION_CANDIDATE: Final[dict[str, str]] = {
    IN_PROJ: "none -- vendor GEMM",
    VALUE_CONV: "value conv state update and activation",
    KEY_CONV: "B/C key conv state update",
    PREP: "SO(3) parameter preparation and FOH taps",
    RECURRENCE: "recurrence/readout -- closed at its roofline",
    TAIL: "skip/gate/RMS-normalization tail",
    OUT_PROJ: "none -- vendor GEMM",
    CARRY_CONV: "value conv state update and activation",
    CARRY_KEYS: "B/C key conv state update",
}
"""Which Phase 3 fusion candidate each stage is, or why it is none.

Neither carry is a candidate of its own: each is the second half of the
convolution state update it belongs to, and a fusion that folds that convolution
in deletes its carry with it."""

COPY_ONLY_STAGES: Final[frozenset[str]] = frozenset({CARRY_CONV, CARRY_KEYS})
"""Stages that launch no kernel.

Each is one :meth:`torch.Tensor.copy_` between contiguous buffers of one dtype,
which the runtime serves as a device-to-device memcpy. NCU profiles kernels: a
window holding only copies makes it print ``No kernels were profiled`` and emit no
CSV at all, which reaches the caller as a parse failure rather than as an empty
pass. The launch trace carries them instead, in its copy column."""

_DURATION: Final = "gpu__time_duration.sum"
_GRID: Final = "launch__grid_size"
_BLOCK: Final = "launch__block_size"
_REGISTERS: Final = "launch__registers_per_thread"
_DRAM_READ: Final = "dram__bytes_read.sum"
_DRAM_WRITE: Final = "dram__bytes_write.sum"
_L2_READ: Final = "lts__t_sectors_op_read.sum"
_L2_WRITE: Final = "lts__t_sectors_op_write.sum"
_L2_READ_HIT: Final = "lts__t_sector_op_read_hit_rate.pct"
_L2_WRITE_HIT: Final = "lts__t_sector_op_write_hit_rate.pct"
_GLOBAL_LD_REQ: Final = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
_GLOBAL_ST_REQ: Final = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
_GLOBAL_LD_SEC: Final = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
_GLOBAL_ST_SEC: Final = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"
_LOCAL_LD_SEC: Final = "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"
_LOCAL_ST_SEC: Final = "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"

_RATE_METRICS: Final[tuple[str, ...]] = (_L2_READ_HIT, _L2_WRITE_HIT)
"""Metrics that are rates, so a site averages them over its launches.

Every other metric requested here is extensive and sums."""

_PER_LAUNCH_METRICS: Final[tuple[str, ...]] = (_GRID, _BLOCK, _REGISTERS)
"""Launch properties, so a site averages them too rather than summing.

Grid and block are what define a site, so their average over that site is the
value itself; the register count is the one figure this actually reduces."""

CENSUS_TABLE: Final = NcuTable(
    "census",
    (
        _DURATION,
        _GRID,
        _BLOCK,
        _REGISTERS,
        _DRAM_READ,
        _DRAM_WRITE,
        _L2_READ,
        _L2_WRITE,
        _L2_READ_HIT,
        _L2_WRITE_HIT,
        _GLOBAL_LD_REQ,
        _GLOBAL_ST_REQ,
        _GLOBAL_LD_SEC,
        _GLOBAL_ST_SEC,
        _LOCAL_LD_SEC,
        _LOCAL_ST_SEC,
    ),
)
"""One pass carrying every counter a census row needs.

One pass rather than :data:`slinoss.perf.ncu.NCU_TABLES`, because each pass costs a
process start and an interpreter's import per stage and no metric here is wanted at
a different granularity from the rest.

The L2 metrics are requested beside the DRAM ones and not instead. ``dram__bytes``
is device-wide and a co-resident process's traffic lands in it. A kernel cannot
read DRAM it did not miss on, so L2 sectors times the miss rate bounds the kernel's
own traffic and the gap names contamination; see
:func:`contamination_residual_pct`.
"""

VERDICT_METRICS: Final[tuple[str, ...]] = (
    _DURATION,
    _DRAM_READ,
    _DRAM_WRITE,
    _L2_READ,
    _L2_WRITE,
    _L2_READ_HIT,
    _L2_WRITE_HIT,
)
"""The metrics a verdict rests on. Any of these missing voids the stage."""

SECTOR_BYTES: Final = 32
"""Bytes in one L2 sector on every part this repo targets."""

CONTAMINATION_CEILING_PCT: Final = 10.0
"""Residual above which a DRAM byte figure is not treated as the kernel's.

A threshold and not a measurement: it sits between a clean pass and one taken
beside a foreign kernel, and a row above it is reported void rather than
caveated."""

UNJUDGED: Final = "unjudged -- cache-served"
"""Verdict for a kernel whose measured traffic falls short of its compulsory figure.

``footprint > L2`` is not the crossover condition. A shape at 1.42x the cache
measured 0.967x compulsory and was still cache-served, which inflated its apparent
floor percentage. The honest test is the measured traffic ratio, which is a direct
observation."""

WARMUP: Final = 20
ITERS: Final = 200
CAPTURE_ITERS: Final = 3
CONTRACT_STEPS: Final = 32


# ---------------------------------------------------------------------------
# the staged step


@dataclass
class _Slots:
    """The intermediates one step passes between its stages.

    Every field is written by one stage and read by later ones, so a stage run on
    its own after a priming pass reads what the routed program would have handed
    it. All ``None`` before that pass; see :func:`prime`.
    """

    proj: Tensor | None = None
    conv_y: Tensor | None = None
    conv_state: Tensor | None = None
    keys_y: Tensor | None = None
    keys_state: Tensor | None = None
    trans: Tensor | None = None
    tap: Tensor | None = None
    y: Tensor | None = None
    tail: Tensor | None = None
    out: Tensor | None = None


def _held(tensor: Tensor | None, name: str) -> Tensor:
    """Return a populated slot, or name the stage that never ran.

    Args:
        tensor: The slot.
        name: Slot name, for the message.

    Returns:
        The tensor.

    Raises:
        RuntimeError: If the slot is empty, which is a stage run before the one
            that fills its input.
    """
    if tensor is None:
        raise RuntimeError(
            f"slot {name!r} is empty: run the whole program once before running a "
            f"single stage, so every stage reads what the routed step hands it"
        )
    return tensor


def _bytes_of(view: Tensor) -> Bytes:
    """Compulsory bytes of a tensor or a band: its own elements at its own width.

    Args:
        view: The tensor or band view.

    Returns:
        ``numel * element_size``.
    """
    return Bytes(view.numel() * view.element_size())


@dataclass(frozen=True)
class StageOperands:
    """The distinct tensors one stage reads and writes.

    Compulsory traffic is one pass over each of these and nothing else: a byte a
    kernel reads twice is a cache question, not a compulsory one. The figures come
    off the live tensors rather than a width formula, so a layout change moves them
    without editing this file.

    Attributes:
        reads: ``(label, bytes)`` per distinct tensor or band read.
        writes: ``(label, bytes)`` per distinct tensor or band written.
    """

    reads: tuple[tuple[str, Bytes], ...]
    writes: tuple[tuple[str, Bytes], ...]

    @property
    def read_bytes(self) -> Bytes:
        """Compulsory bytes in."""
        return Bytes(sum(int(size) for _, size in self.reads))

    @property
    def write_bytes(self) -> Bytes:
        """Compulsory bytes out."""
        return Bytes(sum(int(size) for _, size in self.writes))

    @property
    def total_bytes(self) -> Bytes:
        """Compulsory bytes moved."""
        return Bytes(int(self.read_bytes) + int(self.write_bytes))


@dataclass(frozen=True)
class Stage:
    """One stage of the routed step, callable on its own.

    Attributes:
        name: One of :data:`STAGE_ORDER`.
        candidate: The fusion candidate this stage is, or why it is none.
        run: Runs the stage against the shared slots, in place. Call it inside
            ``no_grad``: the decode boundary refuses an operand that requires a
            gradient. :meth:`StepProgram.run` and :func:`no_grad_on` both provide it.
        operands: What it reads and writes. Valid once the program has been primed.
        copy_only: Whether the stage launches no kernel. See
            :data:`COPY_ONLY_STAGES`.
    """

    name: str
    candidate: str
    run: Callable[[], None]
    operands: Callable[[], StageOperands]
    copy_only: bool = False


@dataclass(frozen=True)
class StepProgram:
    """The routed one-token step, decomposed into stages over one input set.

    Attributes:
        mixer: The layer, in the activation dtype, on the profiled device.
        x: ``(B,1,d_model)`` input token, in the state's activation dtype.
        state: The layer's decode state, advanced in place by every whole step.
        stages: The stages, in call order.
        slots: The intermediates the stages pass between them.
    """

    mixer: SLinOSSMixer
    x: Tensor
    state: MixerState
    stages: tuple[Stage, ...]
    slots: _Slots

    def stage(self, name: str) -> Stage:
        """Look up one stage.

        Args:
            name: Stage name.

        Returns:
            The stage.

        Raises:
            KeyError: If no stage carries that name.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"no stage {name!r}; have {[s.name for s in self.stages]}")

    def run(self) -> None:
        """Run every stage once, in call order. One routed step.

        Under ``no_grad``, which :meth:`slinoss.mixer.SLinOSSMixer.step` carries as a
        decorator. Not a convenience: the decode boundary refuses an operand that
        requires a gradient, so a staged step outside that mode is not the routed
        step but a raise. One context manager per step, entered outside every launch.
        """
        with torch.no_grad():
            for stage in self.stages:
                stage.run()

    def output(self) -> Tensor:
        """The step's output.

        Returns:
            ``(B,1,d_model)``.

        Raises:
            RuntimeError: If the program has not been run.
        """
        return _held(self.slots.out, "out")


def stage_program(mixer: SLinOSSMixer, x: Tensor, state: MixerState) -> StepProgram:
    """Decompose one routed step into individually callable stages.

    The stage bodies are :meth:`slinoss.mixer.SLinOSSMixer.step` at ``T == 1``,
    split at each dispatch call and nowhere else: the same registries, the same
    order, the same operands, the same in-place writes to ``state``. Splitting is
    what lets one stage be narrowed under a profiler.

    Args:
        mixer: The layer to step.
        x: ``(B,1,d_model)``, in the state's activation dtype.
        state: The layer's decode state, advanced in place.

    Returns:
        The program. Nothing has run, so no slot is populated and
        :meth:`StepProgram.output` refuses until :meth:`StepProgram.run` has.

    Raises:
        ValueError: On a rank, width, token-extent, batch, or dtype disagreement
            with ``state``.
    """
    cfg, layout = mixer.config, mixer.layout
    if x.ndim != 3 or x.shape[2] != cfg.d_model:
        raise ValueError(f"expected (B,1,{cfg.d_model}), got {tuple(x.shape)}")
    if x.shape[1] != TOKENS:
        raise ValueError(f"the census subject is T={TOKENS}, got T={int(x.shape[1])}")
    if x.shape[0] != state.batch:
        raise ValueError(
            f"x holds batch {int(x.shape[0])} and state holds {state.batch}"
        )
    if x.dtype is not state.conv.dtype:
        raise ValueError(
            f"the token is {x.dtype} and the state is {state.conv.dtype}; cast the "
            f"module, not the state"
        )
    slots = _Slots()

    def run_in_proj() -> None:
        slots.proj = linear(x, mixer.in_proj.weight, mixer.in_proj.bias)

    def operands_in_proj() -> StageOperands:
        reads = [
            ("x", _bytes_of(x)),
            ("in_proj.weight", _bytes_of(mixer.in_proj.weight)),
        ]
        # Read through getattr: a Linear built with bias=False holds None, which the
        # annotated attribute type does not admit.
        bias: Tensor | None = getattr(mixer.in_proj, "bias", None)
        if bias is not None:
            reads.append(("in_proj.bias", _bytes_of(bias)))
        return StageOperands(
            tuple(reads), (("proj", _bytes_of(_held(slots.proj, "proj"))),)
        )

    def run_value_conv() -> None:
        proj = _held(slots.proj, "proj")
        backend = conv_dispatch.resolve(None, proj.device.type, proj.dtype)
        result = backend.forward(
            layout.value(proj),
            cast_to(mixer.conv_weight, proj.dtype),
            cast_opt(mixer.conv_bias, proj.dtype),
            activation=True,
            initial_state=state.conv,
            d_head=cfg.d_head,
        )
        slots.conv_y, slots.conv_state = result.y, result.state

    def operands_value_conv() -> StageOperands:
        proj = _held(slots.proj, "proj")
        reads = [
            ("proj.value", _bytes_of(layout.value(proj))),
            ("state.conv", _bytes_of(state.conv)),
            ("conv_weight", _bytes_of(mixer.conv_weight)),
        ]
        if mixer.conv_bias is not None:
            reads.append(("conv_bias", _bytes_of(mixer.conv_bias)))
        return StageOperands(
            tuple(reads),
            (
                ("conv.y", _bytes_of(_held(slots.conv_y, "conv_y"))),
                ("conv.state", _bytes_of(_held(slots.conv_state, "conv_state"))),
            ),
        )

    def run_key_conv() -> None:
        if mixer.key_weight is None:
            return
        proj = _held(slots.proj, "proj")
        backend = conv_dispatch.resolve(None, proj.device.type, proj.dtype)
        result = backend.forward(
            layout.keys(proj),
            cast_to(mixer.key_weight, proj.dtype),
            None,
            activation=False,
            initial_state=state.keys,
        )
        slots.keys_y, slots.keys_state = result.y, result.state

    def operands_key_conv() -> StageOperands:
        if mixer.key_weight is None:
            return StageOperands((), ())
        proj = _held(slots.proj, "proj")
        return StageOperands(
            (
                ("proj.keys", _bytes_of(layout.keys(proj))),
                ("state.keys", _bytes_of(state.keys)),
                ("key_weight", _bytes_of(mixer.key_weight)),
            ),
            (
                ("keys.y", _bytes_of(_held(slots.keys_y, "keys_y"))),
                ("keys.state", _bytes_of(_held(slots.keys_state, "keys_state"))),
            ),
        )

    def run_prep() -> None:
        proj = _held(slots.proj, "proj")
        backend = prep_dispatch.resolve(None, proj.device.type, proj.dtype)
        result = backend.forward(
            layout.params(proj),
            mixer.transition_bias,
            heads=cfg.n_heads,
            w_max=cfg.w_max,
        )
        slots.trans, slots.tap = result.trans, result.K

    def operands_prep() -> StageOperands:
        proj = _held(slots.proj, "proj")
        return StageOperands(
            (
                ("proj.params", _bytes_of(layout.params(proj))),
                ("transition_bias", _bytes_of(mixer.transition_bias)),
            ),
            (
                ("trans", _bytes_of(_held(slots.trans, "trans"))),
                ("K", _bytes_of(_held(slots.tap, "tap"))),
            ),
        )

    def bands() -> tuple[Tensor, Tensor]:
        """The ``B`` and ``C`` bands the recurrence reads, from whichever producer."""
        proj = _held(slots.proj, "proj")
        if mixer.key_weight is None:
            return layout.b(proj), layout.c(proj)
        keys_y = _held(slots.keys_y, "keys_y")
        return layout.key_b(keys_y), layout.key_c(keys_y)

    def run_recurrence() -> None:
        b_band, c_band = bands()
        slots.y = decode_step(
            _held(slots.conv_y, "conv_y"),
            _held(slots.trans, "trans"),
            _held(slots.tap, "tap"),
            b_band,
            c_band,
            ssm=state.ssm,
            b_prev=state.b_prev,
            u_prev=state.u_prev,
        ).y

    def operands_recurrence() -> StageOperands:
        b_band, c_band = bands()
        return StageOperands(
            (
                ("U", _bytes_of(_held(slots.conv_y, "conv_y"))),
                ("trans", _bytes_of(_held(slots.trans, "trans"))),
                ("K", _bytes_of(_held(slots.tap, "tap"))),
                ("B", _bytes_of(b_band)),
                ("C", _bytes_of(c_band)),
                ("state.ssm", _bytes_of(state.ssm)),
            ),
            (
                ("y", _bytes_of(_held(slots.y, "y"))),
                ("state.ssm", _bytes_of(state.ssm)),
                ("state.b_prev", _bytes_of(state.b_prev)),
                ("state.u_prev", _bytes_of(state.u_prev)),
            ),
        )

    def run_tail() -> None:
        proj = _held(slots.proj, "proj")
        backend = tail_dispatch.resolve(None, proj.device.type, proj.dtype)
        slots.tail = backend.forward(
            _held(slots.y, "y"),
            _held(slots.conv_y, "conv_y"),
            layout.gate(proj),
            mixer.skip_gain,
            mixer.norm_weight,
            eps=cfg.norm_eps,
        )

    def operands_tail() -> StageOperands:
        proj = _held(slots.proj, "proj")
        return StageOperands(
            (
                ("y", _bytes_of(_held(slots.y, "y"))),
                ("conv.y", _bytes_of(_held(slots.conv_y, "conv_y"))),
                ("proj.gate", _bytes_of(layout.gate(proj))),
                ("skip_gain", _bytes_of(mixer.skip_gain)),
                ("norm_weight", _bytes_of(mixer.norm_weight)),
            ),
            (("tail", _bytes_of(_held(slots.tail, "tail"))),),
        )

    def run_out_proj() -> None:
        slots.out = linear(
            _held(slots.tail, "tail"), mixer.out_proj.weight, mixer.out_proj.bias
        )

    def operands_out_proj() -> StageOperands:
        reads = [
            ("tail", _bytes_of(_held(slots.tail, "tail"))),
            ("out_proj.weight", _bytes_of(mixer.out_proj.weight)),
        ]
        bias: Tensor | None = getattr(mixer.out_proj, "bias", None)
        if bias is not None:
            reads.append(("out_proj.bias", _bytes_of(bias)))
        return StageOperands(
            tuple(reads), (("out", _bytes_of(_held(slots.out, "out"))),)
        )

    def run_carry_conv() -> None:
        state.conv.copy_(_held(slots.conv_state, "conv_state"))

    def operands_carry_conv() -> StageOperands:
        return StageOperands(
            (("conv.state", _bytes_of(_held(slots.conv_state, "conv_state"))),),
            (("state.conv", _bytes_of(state.conv)),),
        )

    def run_carry_keys() -> None:
        if slots.keys_state is None:
            return
        state.keys.copy_(slots.keys_state)

    def operands_carry_keys() -> StageOperands:
        if mixer.key_weight is None:
            return StageOperands((), ())
        return StageOperands(
            (("keys.state", _bytes_of(_held(slots.keys_state, "keys_state"))),),
            (("state.keys", _bytes_of(state.keys)),),
        )

    bodies: dict[str, tuple[Callable[[], None], Callable[[], StageOperands]]] = {
        IN_PROJ: (run_in_proj, operands_in_proj),
        VALUE_CONV: (run_value_conv, operands_value_conv),
        KEY_CONV: (run_key_conv, operands_key_conv),
        PREP: (run_prep, operands_prep),
        RECURRENCE: (run_recurrence, operands_recurrence),
        TAIL: (run_tail, operands_tail),
        OUT_PROJ: (run_out_proj, operands_out_proj),
        CARRY_CONV: (run_carry_conv, operands_carry_conv),
        CARRY_KEYS: (run_carry_keys, operands_carry_keys),
    }
    stages = tuple(
        Stage(
            name,
            FUSION_CANDIDATE[name],
            bodies[name][0],
            bodies[name][1],
            copy_only=name in COPY_ONLY_STAGES,
        )
        for name in STAGE_ORDER
    )
    return StepProgram(mixer=mixer, x=x, state=state, stages=stages, slots=slots)


def build_layer(
    shape_name: str,
    batch: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    seed: int = 0,
) -> StepProgram:
    """Allocate one layer, one token and one state, and stage the step over them.

    The geometry is the shared decode shape's, so a census figure and a training
    figure at one name were taken over one layer. The batch is this driver's own
    axis: at ``T == 1`` it is the only extensive one left, and it is what carries
    the footprint across the cache.

    Args:
        shape_name: A :data:`slinoss.perf.workload.DECODE_SHAPES` name. Its scan
            geometry and group count are used and its batch is not.
        batch: ``B``.
        device: Where to allocate.
        dtype: Activation dtype.
        seed: Seed for the parameters and the token.

    Returns:
        The program, not yet run.
    """
    shape = decode_shape_by_name(shape_name)
    cfg = layer_config(shape.scan, groups=shape.groups)
    torch.manual_seed(seed)
    mixer = SLinOSSMixer(cfg, device=device).to(dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        batch, TOKENS, cfg.d_model, device=device, dtype=dtype, generator=gen
    )
    state = MixerState.allocate(cfg, batch, device=device, dtype=dtype)
    return stage_program(mixer, x, state)


def geometry(program: StepProgram) -> dict[str, Any]:
    """The extents a report names, read off the allocated layer.

    Args:
        program: The staged step.

    Returns:
        A JSON-ready mapping.
    """
    cfg, layout = program.mixer.config, program.mixer.layout
    state = program.state
    return {
        "batch": int(program.x.shape[0]),
        "d_model": cfg.d_model,
        "d_inner": cfg.d_inner,
        "heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "d_state": cfg.d_state,
        "groups": cfg.n_groups,
        "d_conv": cfg.d_conv,
        "key_conv": cfg.key_conv,
        "proj_width": layout.width,
        "activation_dtype": str(program.x.dtype),
        "state_dtype": str(state.ssm.dtype),
        "ssm_state_bytes": int(_bytes_of(state.ssm)),
    }


@contextmanager
def no_grad_on(device: torch.device) -> Iterator[None]:
    """Hold a region to the mode and the device a decode step runs in.

    Args:
        device: The device every launch goes to.

    Yields:
        None.
    """
    with torch.no_grad(), on_device(device):
        yield


def prime(program: StepProgram, *, warmup: int = WARMUP) -> None:
    """Run whole steps until every slot is populated and nothing compiles again.

    Args:
        program: The staged step.
        warmup: Whole steps to run, at least one.

    Raises:
        ValueError: If ``warmup`` is not positive. A stage cannot be run in
            isolation from empty slots.
    """
    if warmup <= 0:
        raise ValueError(f"priming needs at least one whole step, got {warmup}")
    device = program.x.device
    with no_grad_on(device):
        for _ in range(warmup):
            program.run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# provenance and the steady-state contract


def registry_names() -> dict[str, tuple[str, ...]]:
    """Every registry on the routed step's path, and what it holds.

    Returns:
        Registry name to the backend names registered under it.
    """
    return {
        "conv": conv_dispatch.names(),
        "scanprep": prep_dispatch.names(),
        "so3ssd": scan_dispatch.names(),
        "mixer_tail": tail_dispatch.names(),
        "decode": decode_dispatch.names(),
    }


def resolved_backends(device_type: str, dtype: torch.dtype) -> dict[str, str]:
    """What each registry answers with at one device type and dtype.

    Args:
        device_type: ``"cuda"`` or ``"cpu"``.
        dtype: Activation dtype.

    Returns:
        Registry name to the resolved backend name.
    """
    return {
        "conv": conv_dispatch.resolve(None, device_type, dtype).name,
        "scanprep": prep_dispatch.resolve(None, device_type, dtype).name,
        "so3ssd": scan_dispatch.resolve(None, device_type, dtype).name,
        "mixer_tail": tail_dispatch.resolve(None, device_type, dtype).name,
        "decode": decode_dispatch.resolve(None, device_type, dtype).name,
    }


def require_kernel_path(device_type: str, dtype: torch.dtype) -> dict[str, str]:
    """Resolve every registry and refuse a reference decode.

    A registry whose kernel import failed answers every call with the reference, so
    a census taken without this check is a census of whichever implementation
    happened to be reachable. An extension build does not cover it: the DSL is a
    separate import and fails separately.

    Args:
        device_type: ``"cuda"`` or ``"cpu"``.
        dtype: Activation dtype.

    Returns:
        Registry name to the resolved backend name.

    Raises:
        RuntimeError: If ``decode`` resolved to the reference.
    """
    resolved = resolved_backends(device_type, dtype)
    if resolved["decode"] == decode_dispatch.REFERENCE:
        raise RuntimeError(
            "decode resolved to 'reference': the CuTe DSL is not importable, so this "
            "would census the reference program. Put the DSL on PYTHONPATH and "
            "re-run; build_ext does not cover it"
        )
    return resolved


def cuda_build_version() -> str:
    """The CUDA toolkit this torch was built against.

    Read through ``getattr``: ``torch.version`` is a submodule torch populates at
    import and the type stubs do not declare it, so the attribute path does not
    type-check while the value it holds is the one the provenance block needs.

    Returns:
        The version, or ``"unknown"``.
    """
    module = getattr(torch, "version", None)
    return str(getattr(module, "cuda", None) or "unknown")


@dataclass(frozen=True)
class Provenance:
    """Which program ran, on which card, in which state.

    Attributes:
        slinoss_file: ``slinoss.__file__``, so the tree is named and not assumed.
        torch_version: Torch version and the CUDA it was built against.
        registries: Every registry's ``names()``.
        resolved: The backend each registry answered with.
        device: Device properties, including the L2 capacity a footprint is placed
            against.
        clocks: Clock policy at entry.
        before: Contention sampled immediately before the timed region.
        after: Contention sampled immediately after it.
    """

    slinoss_file: str
    torch_version: str
    registries: dict[str, tuple[str, ...]]
    resolved: dict[str, str]
    device: DeviceInfo
    clocks: ClockPolicy
    before: Contention
    after: Contention


def provenance(device: torch.device, dtype: torch.dtype) -> Provenance:
    """Collect the provenance block, resolving the kernel path or refusing.

    Args:
        device: The profiled device.
        dtype: Activation dtype.

    Returns:
        The block. ``before`` and ``after`` both hold the entry sample; a caller
        that runs a timed region replaces ``after`` when the region ends.

    Raises:
        RuntimeError: If ``decode`` resolves to the reference.
    """
    import slinoss

    ordinal = device_ordinal(device)
    resolved = require_kernel_path(device.type, dtype)
    entry = contention(ordinal)
    return Provenance(
        slinoss_file=str(slinoss.__file__),
        torch_version=f"{torch.__version__} cuda {cuda_build_version()}",
        registries=registry_names(),
        resolved=resolved,
        device=device_info(ordinal),
        clocks=clock_policy(ordinal),
        before=entry,
        after=entry,
    )


@dataclass(frozen=True)
class ContractCheck:
    """Whether the steady-state step keeps the hard contract.

    Attributes:
        steps: Steps run under the check.
        allocated_delta_bytes: Change in outstanding allocator bytes. Nonzero on an
            eager step is expected and is stated rather than judged: the step
            allocates each intermediate it computes, which is what capture is for.
        segment_count_delta: Change in allocator segments, which counts the
            ``cudaMalloc`` calls that outlived the window. Nonzero is a defect: a
            steady step whose pool grows has no steady cost.
        alloc_retry_delta: Change in allocator retries. Each is a synchronize and a
            cache flush inside a step that must have neither.
        sync_violation: The synchronizing or device-reading call the step made, or
            empty. Any is a defect.
    """

    steps: Count
    allocated_delta_bytes: int
    segment_count_delta: int
    alloc_retry_delta: int
    sync_violation: str

    @property
    def passed(self) -> bool:
        """Whether nothing the contract forbids happened."""
        return (
            self.segment_count_delta == 0
            and self.alloc_retry_delta == 0
            and not self.sync_violation
        )


def contract_check(
    program: StepProgram, *, steps: int = CONTRACT_STEPS
) -> ContractCheck:
    """Run steady-state steps and report every contract violation.

    ``set_sync_debug_mode("error")`` raises on any call that synchronizes or reads a
    device value on the host, which is the whole class the contract forbids and is
    stricter than inferring it from a timeline. Compilation falls in the same
    window: a step that compiled would allocate and synchronize to do it.

    Args:
        program: The staged step, already primed.
        steps: Steps to run inside the window.

    Returns:
        The check.

    Raises:
        ValueError: If ``steps`` is not positive.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    device = program.x.device
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)
    previous = torch.cuda.get_sync_debug_mode()
    violation = ""
    torch.cuda.set_sync_debug_mode("error")
    try:
        with no_grad_on(device):
            for _ in range(steps):
                program.run()
    except RuntimeError as error:
        violation = str(error).splitlines()[0]
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)

    def delta(key: str) -> int:
        return int(after.get(key, 0)) - int(before.get(key, 0))

    return ContractCheck(
        steps=Count(steps),
        allocated_delta_bytes=delta("allocated_bytes.all.current"),
        segment_count_delta=delta("segment.all.current"),
        alloc_retry_delta=delta("num_alloc_retries"),
        sync_violation=violation,
    )


# ---------------------------------------------------------------------------
# launch census


@dataclass(frozen=True)
class LaunchRow:
    """One kernel's launch footprint in the routed step.

    Attributes:
        kernel: Kernel name as nsys reports it.
        launches_per_step: Launches per step, which is per layer: the subject is
            one layer, so no depth divides this.
        duration_us: Device time per step, summed over those launches.
        share_pct: That time as a percentage of the traced device time.
    """

    kernel: str
    launches_per_step: Ratio
    duration_us: Microseconds
    share_pct: Percent


@dataclass(frozen=True)
class LaunchCensus:
    """The whole step's launch and gap census.

    Attributes:
        reps: Steps inside the profiler window, and the divisor of every per-step
            figure here.
        step_wall_us: One step by CUDA events, median over the timed samples.
        device_us: Device time of one step: kernels, copies and fills.
        kernel_us: Device time of one step in kernels alone.
        copies_per_step: Copies per step. A carry between same-dtype contiguous
            buffers is a copy and not a kernel, so counting kernels alone makes it
            invisible.
        copy_us: Device time those copies took, per step.
        launches_per_step: Kernels launched per step.
        idle_us: Device idle strictly inside the traced span, per step.
        idle_pct: That idle as a percentage of the span.
        gaps_per_step: Idle intervals per step.
        max_gap_us: The longest single one, not divided.
        per_launch_idle_us: ``idle_us`` over ``launches_per_step``. The figure a
            launch count would be a lever against, or would not.
        host_us: ``step_wall_us - device_us``. What the step costs beyond the
            device: launch submission and the host program between launches.
        rows: One row per kernel, by descending duration.
    """

    reps: Count
    step_wall_us: Microseconds
    device_us: Microseconds
    kernel_us: Microseconds
    copies_per_step: Ratio
    copy_us: Microseconds
    launches_per_step: Ratio
    idle_us: Microseconds
    idle_pct: Percent
    gaps_per_step: Ratio
    max_gap_us: Microseconds
    per_launch_idle_us: Microseconds
    host_us: Microseconds
    rows: tuple[LaunchRow, ...]


def launch_census(
    text: str,
    *,
    step_wall_us: Microseconds,
    reps: int,
    report_path: str = "",
) -> LaunchCensus:
    """Reduce one GPU trace to a per-kernel launch census of a single step.

    Args:
        text: ``nsys stats --report cuda_gpu_trace`` stdout.
        step_wall_us: One step by CUDA events, measured in the same session.
        reps: Steps inside the profiler window.
        report_path: The report the text came from, recorded on the trace.

    Returns:
        The census, per step.

    Raises:
        ValueError: If ``reps`` is not positive.
    """
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    trace = parse_gpu_trace(text, label="step", report_path=report_path)
    gaps = occupancy("step", parse_gpu_events(text))
    per = float(reps)
    launches = sum(int(kernel.launch_count) for kernel in trace.kernels)
    launches_per_step = launches / per
    device_us = float(trace.device_sum_duration_us) / per
    idle_us = float(gaps.idle_us) / per
    rows = tuple(
        LaunchRow(
            kernel=kernel.kernel,
            launches_per_step=Ratio(int(kernel.launch_count) / per),
            duration_us=Microseconds(float(kernel.duration_us) / per),
            share_pct=Percent(float(kernel.share_pct)),
        )
        for kernel in trace.kernels
    )
    return LaunchCensus(
        reps=Count(reps),
        step_wall_us=step_wall_us,
        device_us=Microseconds(device_us),
        kernel_us=Microseconds(float(trace.kernel_sum_duration_us) / per),
        copies_per_step=Ratio(int(trace.memcpy_count) / per),
        copy_us=Microseconds(float(trace.memcpy_sum_duration_us) / per),
        launches_per_step=Ratio(launches_per_step),
        idle_us=Microseconds(idle_us),
        idle_pct=Percent(float(gaps.idle_pct)),
        gaps_per_step=Ratio(int(gaps.gap_count) / per),
        max_gap_us=Microseconds(float(gaps.max_gap_us)),
        per_launch_idle_us=Microseconds(
            idle_us / launches_per_step if launches_per_step else 0.0
        ),
        host_us=Microseconds(float(step_wall_us) - device_us),
        rows=rows,
    )


def loop_wall_us(
    body: Callable[[], object], *, iters: int, device: torch.device
) -> Microseconds:
    """Host wall clock per step, over a whole loop rather than per sample.

    A CUDA event pair brackets device work, so a per-sample median reports the
    device timeline and not what the step costs when the host is the thing behind.
    The samples go bimodal: several short intervals while the host drains a queue it
    already filled, then one long one while it refills. Measured at ``B`` 1: the
    per-sample median read 38.912 us against a mean of 600 over the same 200
    samples, and the median is what a per-step figure would otherwise be read from.
    A wall over the whole loop with one synchronize at each end has no such mode.

    Args:
        body: One step. Must not synchronize.
        iters: Steps in the loop.
        device: Device to synchronize.

    Returns:
        Wall time per step.

    Raises:
        ValueError: If ``iters`` is not positive.
    """
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        body()
    torch.cuda.synchronize(device)
    return Microseconds((time.perf_counter() - start) * 1e6 / iters)


@dataclass(frozen=True)
class ReplayTiming:
    """One step's wall time with its host program deleted.

    Attributes:
        wall_us: Median replay of one step by CUDA events, or zero on failure.
        idle_us: ``wall_us`` minus the step's device time, floored at zero. What
            the launches themselves cost once no Python runs between them.
        per_launch_idle_us: That idle over the step's launch count. The figure a
            launch count is or is not a lever against.
        error: Why capture failed, or empty. A cell carrying one is void.
        eager_loop_wall_us: The eager step by host wall clock over a whole loop.
            The figure the eager host cost is read from; see :func:`loop_wall_us`.
        replay_loop_wall_us: The same clock over replays, so the two are comparable
            and their difference is what the host program costs.
    """

    wall_us: Microseconds
    idle_us: Microseconds
    per_launch_idle_us: Microseconds
    error: str
    eager_loop_wall_us: Microseconds
    replay_loop_wall_us: Microseconds


def replay_timing(
    program: StepProgram,
    census: LaunchCensus,
    *,
    iters: int,
    warmup: int,
    device: torch.device,
    clocks: ClockPolicy | None = None,
) -> ReplayTiming:
    """Time one step under CUDA-graph replay and price its launches.

    Replay is the same launches with the host program removed: one graph launch, no
    dispatcher, no Python between kernels. So the eager wall minus this is what the
    host costs, and this minus device time is what the launches cost. Only the second
    is what a launch count could buy back, which is why the eager per-launch idle is
    the wrong figure to refuse the lever on.

    One capture per process. A second decode capture in one process fires a
    device-side ``indexSelectSmallIndex`` assert at some shapes and a device assert
    poisons the context, so a failure is returned rather than raised and the caller
    reports the cell void instead of measuring past it.

    Args:
        program: The staged step. Captured as it stands, so it must already be
            primed: capture refuses a recording that compiles.
        census: The launch census of the same program, for its device time and
            launch count.
        iters: Timed replays.
        warmup: Replays before timing.
        device: The device both were measured on.
        clocks: Clock policy recorded with the timing, or None to probe it.

    Returns:
        The timing, or one carrying ``error``.
    """
    from slinoss.graph import capture

    zero = Microseconds(0.0)
    try:
        step = capture(program.run)
    except (RuntimeError, ValueError, torch.cuda.CudaError) as exc:
        return ReplayTiming(
            zero, zero, zero, f"{type(exc).__name__}: {exc}", zero, zero
        )
    timed = measure(
        step,
        label="replay",
        iters=iters,
        warmup=warmup,
        device=device,
        clocks=clocks,
    )
    wall = float(timed.total.median_duration_us)
    idle = max(0.0, wall - float(census.device_us))
    launches = float(census.launches_per_step)
    for _ in range(warmup):
        program.run()
    eager_loop = loop_wall_us(program.run, iters=iters, device=device)
    replay_loop = loop_wall_us(step, iters=iters, device=device)
    return ReplayTiming(
        Microseconds(wall),
        Microseconds(idle),
        Microseconds(idle / launches if launches else 0.0),
        "",
        eager_loop,
        replay_loop,
    )


# ---------------------------------------------------------------------------
# traffic census


def contamination_residual_pct(
    dram_bytes: Bytes,
    read_sectors: float,
    write_sectors: float,
    read_hit_pct: float,
    write_hit_pct: float,
) -> Percent:
    """How far reported DRAM traffic sits above what this kernel can account for.

    A kernel cannot read DRAM it did not miss on, so read sectors times the read
    miss rate times :data:`SECTOR_BYTES` bounds its own read traffic. The write side
    does not work that way and the miss rate is the wrong factor for it: L2 is
    write-back, so a store that finds its line resident is counted a hit and the
    dirty line still reaches DRAM on eviction. Measured on the recurrence kernel at
    ``B`` 128: write hit rate 99.91% against 141,843,072 DRAM bytes written, which
    the miss-rate form explains 129,920 of. Write sectors times the sector size
    explains it to 1.77%, and the read miss form lands within 0.52%.

    So the write term carries no miss rate, which makes the estimate an upper bound
    on the write side rather than a two-sided model: a line written twice, or written
    and never evicted inside the window, moves fewer DRAM bytes than its sectors. The
    residual is therefore signed and only its positive side means anything. Traffic
    above the bound is a co-resident process's, landing in a device-wide counter;
    traffic below it is the kernel's own writes still sitting in cache, which is the
    regime :data:`UNJUDGED` already names.

    Args:
        dram_bytes: ``dram__bytes_read`` plus ``dram__bytes_write``.
        read_sectors: L2 read sectors.
        write_sectors: L2 write sectors.
        read_hit_pct: L2 read hit rate, percent.
        write_hit_pct: L2 write hit rate, percent. Unused, and named so that a reader
            reaching for it finds why.

    Returns:
        Reported minus explained, as a signed percentage of ``dram_bytes``, and zero
        when that is zero.
    """
    del write_hit_pct
    implied = SECTOR_BYTES * (
        read_sectors * (1.0 - read_hit_pct / 100.0) + write_sectors
    )
    reported = float(dram_bytes)
    if reported <= 0.0:
        return Percent(0.0)
    return Percent((reported - implied) / reported * 100.0)


@dataclass(frozen=True)
class Site:
    """One kernel at one launch geometry, which is one place in the program.

    A counter reported as a kernel-wide average hides its outlier: a kernel-wide
    sectors-per-request figure has already concealed a 3x per-tensor amplification
    in this tree. Grid and block are what separate the two convolution launches,
    and the two projections, inside one kernel name, so a row keyed on them is a
    row about one site.

    Attributes:
        kernel: Demangled kernel name.
        grid: Blocks launched.
        block: Threads per block.
    """

    kernel: str
    grid: int
    block: int

    @property
    def label(self) -> str:
        """The site, as one string a table cell can hold."""
        return f"{self.kernel} g{self.grid} b{self.block}"


@dataclass(frozen=True)
class TrafficRow:
    """One site's measured traffic against the bytes its stage must move.

    Attributes:
        stage: The stage of the routed step that launched it.
        candidate: The fusion candidate that stage is, or why it is none.
        site: Kernel and launch geometry.
        launches_per_step: Launches per step, per layer.
        duration_us: Device time per step. NCU serializes, so this is the site's own
            cost and not a share of a wall.
        dram_read_bytes: Measured, per step.
        dram_write_bytes: Measured, per step.
        compulsory_bytes: One pass over each distinct operand the whole stage
            touches, per step.
        traffic_ratio: The stage's measured DRAM bytes over its compulsory bytes.
            At or above one the footprint is not being served by cache and the
            kernel is judgeable; below one it is, and no bandwidth verdict exists
            at that shape.
        achieved_gbs: Measured bytes over measured duration.
        floor_gbs: The fitted floor's rate at the same per-launch byte count, which
            is the size-matched denominator.
        floor_pct: ``achieved_gbs`` over ``floor_gbs``.
        roofline_class: The verdict, or :data:`UNJUDGED` where the traffic test says
            the footprint is cache-served.
        sectors_per_load_request: Global load sectors over load requests, at this
            site rather than across the kernel. Above four is an uncoalesced read.
        sectors_per_store_request: The same for stores.
        local_sector_count: Local-memory sectors, per step. Nonzero is a register
            spill, and a spilled kernel is not entitled to a bandwidth verdict.
        register_per_thread_count: Registers per thread, which is what a fusion
            proposal has to fit inside.
        residual_pct: Contamination residual, signed. Above
            :data:`CONTAMINATION_CEILING_PCT` the byte figures are not this
            kernel's and the row is void; a negative value is the kernel's own
            writes still held in cache and carries no verdict of its own. See
            :func:`contamination_residual_pct`.
    """

    stage: str
    candidate: str
    site: Site
    launches_per_step: Ratio
    duration_us: Microseconds
    dram_read_bytes: Bytes
    dram_write_bytes: Bytes
    compulsory_bytes: Bytes
    traffic_ratio: Ratio
    achieved_gbs: GBPerSecond
    floor_gbs: GBPerSecond
    floor_pct: Percent
    roofline_class: str
    sectors_per_load_request: Ratio
    sectors_per_store_request: Ratio
    local_sector_count: int
    register_per_thread_count: int
    residual_pct: Percent

    @property
    def dram_bytes(self) -> Bytes:
        """Measured bytes moved per step, read plus write."""
        return Bytes(int(self.dram_read_bytes) + int(self.dram_write_bytes))

    @property
    def contaminated(self) -> bool:
        """Whether the DRAM counters carry traffic that is not this kernel's."""
        return float(self.residual_pct) > CONTAMINATION_CEILING_PCT

    @property
    def spilled(self) -> bool:
        """Whether the site touched local memory."""
        return self.local_sector_count > 0


def sum_by_site(one: NcuPass) -> dict[Site, dict[str, float]]:
    """Reduce a pass's per-launch metrics onto its sites.

    Rates and launch properties are averaged over a site's launches; every other
    metric is extensive and sums.

    Args:
        one: The pass.

    Returns:
        Site to metric name to value, with ``"launches"`` carrying the count.
    """
    out: dict[Site, dict[str, float]] = {}
    for invocation in one.invocations:
        values = invocation.values
        site = Site(
            kernel=invocation.kernel,
            grid=int(values.get(_GRID, 0.0)),
            block=int(values.get(_BLOCK, 0.0)),
        )
        row = out.setdefault(site, {"launches": 0.0})
        row["launches"] += 1.0
        for metric, value in values.items():
            row[metric] = row.get(metric, 0.0) + value
    for row in out.values():
        for metric in (*_RATE_METRICS, *_PER_LAUNCH_METRICS):
            if metric in row:
                row[metric] /= row["launches"]
    return out


def missing_verdict_metrics(one: NcuPass) -> tuple[str, ...]:
    """The metrics a verdict rests on that this pass did not carry.

    Args:
        one: The pass.

    Returns:
        The missing names, in :data:`VERDICT_METRICS` order.
    """
    absent = set(one.missing_metrics)
    return tuple(metric for metric in VERDICT_METRICS if metric in absent)


def stage_rows(
    stage: str,
    candidate: str,
    one: NcuPass,
    *,
    compulsory: Bytes,
    reps: int,
    floor: DramTimeFloor,
) -> tuple[TrafficRow, ...]:
    """Join one stage's counters, its compulsory figure, and the fitted floor.

    The compulsory figure belongs to the stage and not to a site, but the traffic
    ratio and the verdict it gates are read per site: this site's measured traffic
    over the stage's compulsory bytes. Reading the ratio on the stage's total
    instead lets one site's traffic buy a verdict for another's. Measured: the
    recurrence stage's two sites move 284,624,341 B and 170 B, and the stage ratio
    1.0012 declared the 170 B site DRAM-bound at 197.57% of a floor it never asked
    for. Per site that one reads 6e-7x compulsory and is named unjudged.

    Args:
        stage: Stage name.
        candidate: The fusion candidate the stage is, or why it is none.
        one: The census pass over this stage.
        compulsory: Compulsory bytes for the whole stage, one step's worth.
        reps: Steps inside the profiler window.
        floor: The DRAM floor fitted in this session on this device.

    Returns:
        One row per site, by descending duration.

    Raises:
        ValueError: If ``reps`` is not positive.
        RuntimeError: If the pass is missing a metric a verdict rests on.
    """
    if reps <= 0:
        raise ValueError(f"reps must be positive, got {reps}")
    absent = missing_verdict_metrics(one)
    if absent:
        raise RuntimeError(f"stage {stage!r}: NCU did not report {absent}")
    by_site = sum_by_site(one)
    per = float(reps)
    rows: list[TrafficRow] = []
    for site, row in by_site.items():
        launches = int(row["launches"])
        window_bytes = Bytes(int(row.get(_DRAM_READ, 0.0) + row.get(_DRAM_WRITE, 0.0)))
        window_us = Microseconds(row.get(_DURATION, 0.0) / 1000.0)
        ratio = Ratio(
            (int(window_bytes) / per) / float(compulsory) if compulsory else 0.0
        )
        judged = float(ratio) >= 1.0
        # A site that reached DRAM for nothing at all has no floor to be held to:
        # the law is a time per byte moved and it moved none. That is the extreme of
        # the cache-served case and it is named the same way, not judged against a
        # bandwidth it never asked for.
        verdict = (
            dram_floor_verdict(
                site.label,
                moved_bytes=window_bytes,
                launch_count=Count(launches),
                duration_us=window_us,
                floor=floor,
            )
            if int(window_bytes) > 0
            else None
        )
        per_launch = Bytes(int(window_bytes) // launches)
        load_requests = row.get(_GLOBAL_LD_REQ, 0.0)
        store_requests = row.get(_GLOBAL_ST_REQ, 0.0)
        rows.append(
            TrafficRow(
                stage=stage,
                candidate=candidate,
                site=site,
                launches_per_step=Ratio(launches / per),
                duration_us=Microseconds(float(window_us) / per),
                dram_read_bytes=Bytes(int(row.get(_DRAM_READ, 0.0) / per)),
                dram_write_bytes=Bytes(int(row.get(_DRAM_WRITE, 0.0) / per)),
                compulsory_bytes=compulsory,
                traffic_ratio=ratio,
                achieved_gbs=(
                    gbs_from_bytes_us(window_bytes, window_us)
                    if float(window_us) > 0.0
                    else GBPerSecond(0.0)
                ),
                floor_gbs=(
                    floor.floor_gbs(per_launch)
                    if int(per_launch) > 0
                    else GBPerSecond(0.0)
                ),
                floor_pct=(
                    Percent(float(verdict.achieved_pct))
                    if verdict is not None
                    else Percent(0.0)
                ),
                roofline_class=(
                    f"{verdict.declared} {'pass' if verdict.passed else 'FAIL'}"
                    if judged and verdict is not None
                    else UNJUDGED
                ),
                sectors_per_load_request=Ratio(
                    row.get(_GLOBAL_LD_SEC, 0.0) / load_requests
                    if load_requests
                    else 0.0
                ),
                sectors_per_store_request=Ratio(
                    row.get(_GLOBAL_ST_SEC, 0.0) / store_requests
                    if store_requests
                    else 0.0
                ),
                local_sector_count=int(
                    (row.get(_LOCAL_LD_SEC, 0.0) + row.get(_LOCAL_ST_SEC, 0.0)) / per
                ),
                register_per_thread_count=int(row.get(_REGISTERS, 0.0)),
                residual_pct=contamination_residual_pct(
                    window_bytes,
                    row.get(_L2_READ, 0.0),
                    row.get(_L2_WRITE, 0.0),
                    row.get(_L2_READ_HIT, 0.0),
                    row.get(_L2_WRITE_HIT, 0.0),
                ),
            )
        )
    return tuple(sorted(rows, key=lambda row: float(row.duration_us), reverse=True))


# ---------------------------------------------------------------------------
# artifacts


def payload(obj: object) -> Any:
    """Convert a record tree to JSON-ready data.

    Args:
        obj: A dataclass, mapping, sequence, or scalar.

    Returns:
        Dicts, lists and scalars.
    """
    fields = getattr(obj, "__dataclass_fields__", None)
    if fields is not None:
        return {name: payload(getattr(obj, name)) for name in fields}
    if isinstance(obj, dict):
        return {str(key): payload(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [payload(item) for item in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def write_cell(out: Path, key: str, data: object) -> Path:
    """Write one cell's artifact atomically.

    A census that dies mid-shape has to leave the rows it banked readable, so the
    file appears whole or not at all.

    Args:
        out: Artifact directory, created if absent.
        key: Cell key, which is the file stem.
        data: The record to serialize.

    Returns:
        The written path.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{key}.json"
    handle, temporary = tempfile.mkstemp(dir=str(out), suffix=".tmp")
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload(data), sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)
    return path


def cell_key(kind: str, shape: str, batch: int, dtype: str) -> str:
    """The artifact name one cell owns.

    Args:
        kind: Pass name.
        shape: Shape name.
        batch: ``B``.
        dtype: Activation dtype name.

    Returns:
        The key.
    """
    return f"{kind}-{shape}-b{batch}-{dtype}"


# ---------------------------------------------------------------------------
# tables


def launch_table(census: LaunchCensus) -> str:
    """Render the launch census as one markdown table.

    Args:
        census: The census.

    Returns:
        The table, newline terminated.
    """
    lines = [
        "| kernel | launches/step | us/step | share % |",
        "| --- | --- | --- | --- |",
    ]
    for row in census.rows:
        lines.append(
            f"| {row.kernel} | {float(row.launches_per_step):.3g} | "
            f"{float(row.duration_us):.3f} | {float(row.share_pct):.2f} |"
        )
    lines.append(
        f"| device total | {float(census.launches_per_step):.3g} | "
        f"{float(census.device_us):.3f} | 100.00 |"
    )
    return "\n".join(lines) + "\n"


def traffic_table(rows: Sequence[TrafficRow]) -> str:
    """Render the traffic census as one markdown table.

    Args:
        rows: The rows.

    Returns:
        The table, newline terminated.
    """
    columns = (
        "stage",
        "site",
        "l/step",
        "us",
        "read B",
        "write B",
        "compulsory B",
        "meas/comp",
        "GB/s",
        "floor GB/s",
        "% floor",
        "class",
        "ld sec/req",
        "local sec",
        "reg",
        "resid %",
        "candidate",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            f"| {row.stage} | {row.site.label} | "
            f"{float(row.launches_per_step):.3g} | {float(row.duration_us):.3f} | "
            f"{int(row.dram_read_bytes)} | {int(row.dram_write_bytes)} | "
            f"{int(row.compulsory_bytes)} | {float(row.traffic_ratio):.4f} | "
            f"{float(row.achieved_gbs):.1f} | {float(row.floor_gbs):.1f} | "
            f"{float(row.floor_pct):.2f} | {row.roofline_class} | "
            f"{float(row.sectors_per_load_request):.2f} | {row.local_sector_count} | "
            f"{row.register_per_thread_count} | {float(row.residual_pct):.2f} | "
            f"{row.candidate} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# passes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        The namespace.
    """
    parser = argparse.ArgumentParser(description="Census of the routed one-token step.")
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    parser.add_argument(
        "--shape",
        default="acceptance",
        help="Decode shape name. Its scan geometry and group count are used; its "
        "batch is not, because batch is this driver's own axis.",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device. There is no host path: every figure names its part.",
    )
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument(
        "--capture-iters",
        type=int,
        default=CAPTURE_ITERS,
        help="Steps inside the profiler window, and the divisor of every per-step "
        "figure. The parent passes its own value to the child, so the two cannot "
        "disagree.",
    )
    parser.add_argument(
        "--stage",
        default=ALL_STAGES,
        help="Target pass only: which stage runs inside the window, or all.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/decode-census"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a cell whose artifact already exists.",
    )
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def target_argv(args: argparse.Namespace, stage: str) -> list[str]:
    """The child command a profiler attaches to.

    Args:
        args: Parsed command line.
        stage: Stage to run inside the window, or :data:`ALL_STAGES`.

    Returns:
        The command.
    """
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pass",
        "target",
        "--shape",
        args.shape,
        "--batch",
        str(args.batch),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--stage",
        stage,
        "--capture-iters",
        str(args.capture_iters),
        "--warmup",
        str(args.warmup),
    ]


def run_target(args: argparse.Namespace) -> int:
    """Warm up outside the window, then run the measured steps inside it.

    Args:
        args: Parsed command line.

    Returns:
        Process exit status.
    """
    device = require_cuda(args.device)
    dtype = DTYPES[args.dtype]
    require_kernel_path(device.type, dtype)
    program = build_layer(args.shape, args.batch, device, dtype=dtype)
    prime(program, warmup=args.warmup)
    body = program.run if args.stage == ALL_STAGES else program.stage(args.stage).run
    with no_grad_on(device):
        with profiler_window(device):
            for _ in range(args.capture_iters):
                body()
        torch.cuda.synchronize(device)
    return 0


def run_provenance(args: argparse.Namespace) -> int:
    """Print and bank the provenance block, the compulsory model, and the contract.

    Args:
        args: Parsed command line.

    Returns:
        Process exit status.
    """
    device = require_cuda(args.device)
    dtype = DTYPES[args.dtype]
    block = provenance(device, dtype)
    program = build_layer(args.shape, args.batch, device, dtype=dtype)
    prime(program, warmup=args.warmup)
    contract = contract_check(program)
    record = {
        "provenance": payload(block),
        "geometry": geometry(program),
        "contract": payload(contract),
        "compulsory": {
            stage.name: payload(stage.operands()) for stage in program.stages
        },
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    for registry, registered in block.registries.items():
        print(f"registry {registry}: {registered} -> {block.resolved[registry]}")
    print(f"slinoss {block.slinoss_file}")
    print(f"torch {block.torch_version}")
    print(f"contract passed: {contract.passed} {contract.sync_violation}")
    write_cell(
        args.out, cell_key("provenance", args.shape, args.batch, args.dtype), record
    )
    return 0


def run_launch(args: argparse.Namespace) -> int:
    """Time the step, then trace it and reduce the trace to a launch census.

    Args:
        args: Parsed command line.

    Returns:
        Process exit status.
    """
    key = cell_key("launch", args.shape, args.batch, args.dtype)
    if args.resume and (args.out / f"{key}.json").exists():
        print(f"resume: {key} already banked")
        return 0
    device = require_cuda(args.device)
    dtype = DTYPES[args.dtype]
    block = provenance(device, dtype)
    nsys = resolve_tool(args.nsys)
    program = build_layer(args.shape, args.batch, device, dtype=dtype)
    prime(program, warmup=args.warmup)
    with no_grad_on(device):
        timed = measure(
            program.run,
            label="step",
            iters=args.iters,
            warmup=args.warmup,
            device=device,
            clocks=block.clocks,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / key
    texts = nsys_report_texts(
        target_argv(args, ALL_STAGES),
        base,
        ("cuda_gpu_trace",),
        nsys=nsys,
        timeout_s=args.timeout_s,
    )
    census = launch_census(
        texts["cuda_gpu_trace"],
        step_wall_us=Microseconds(timed.total.median_duration_us),
        reps=args.capture_iters,
        report_path=str(base.with_name(base.name + ".nsys-rep")),
    )
    # After the trace, so a capture that faults cannot corrupt the census it would
    # otherwise be reported beside.
    with no_grad_on(device):
        replay = replay_timing(
            program,
            census,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
            clocks=block.clocks,
        )
    record = {
        "provenance": payload(replace(block, after=contention(device_ordinal(device)))),
        "geometry": geometry(program),
        "wall": payload(timed.total),
        "census": payload(census),
        "replay": payload(replay),
    }
    print(launch_table(census))
    print(
        f"step wall {float(census.step_wall_us):.3f} us, device "
        f"{float(census.device_us):.3f} us, host {float(census.host_us):.3f} us, "
        f"{float(census.launches_per_step):.3g} launches and "
        f"{float(census.copies_per_step):.3g} copies per step, idle "
        f"{float(census.idle_us):.3f} us ({float(census.idle_pct):.2f}%), "
        f"per-launch idle {float(census.per_launch_idle_us):.4f} us"
    )
    print(
        f"replay wall {float(replay.wall_us):.3f} us, launch idle "
        f"{float(replay.idle_us):.3f} us, per-launch "
        f"{float(replay.per_launch_idle_us):.4f} us"
        + (f", CAPTURE FAILED {replay.error}" if replay.error else "")
    )
    print(
        f"loop wall: eager {float(replay.eager_loop_wall_us):.3f} us, replay "
        f"{float(replay.replay_loop_wall_us):.3f} us, host program "
        f"{float(replay.eager_loop_wall_us) - float(replay.replay_loop_wall_us):.3f} us"
    )
    write_cell(args.out, key, record)
    return 0


def run_traffic(args: argparse.Namespace) -> int:
    """Fit the floor, then count each stage's bytes against its compulsory figure.

    Args:
        args: Parsed command line.

    Returns:
        Process exit status.
    """
    key = cell_key("traffic", args.shape, args.batch, args.dtype)
    if args.resume and (args.out / f"{key}.json").exists():
        print(f"resume: {key} already banked")
        return 0
    device = require_cuda(args.device)
    dtype = DTYPES[args.dtype]
    block = provenance(device, dtype)
    ncu = resolve_tool(args.ncu)
    program = build_layer(args.shape, args.batch, device, dtype=dtype)
    prime(program, warmup=args.warmup)
    operands = {stage.name: stage.operands() for stage in program.stages}
    floor = dram_time_floor(device)
    rows: list[TrafficRow] = []
    silent: list[str] = []
    for stage in program.stages:
        compulsory = operands[stage.name].total_bytes
        if int(compulsory) == 0:
            continue
        if stage.copy_only:
            # Declared kernelless, so NCU is never asked: it would print
            # ``No kernels were profiled``, emit no CSV, and exit zero.
            silent.append(stage.name)
            continue
        one = run_ncu(
            CENSUS_TABLE,
            target_argv(args, stage.name),
            ncu=ncu,
            timeout_s=args.timeout_s,
        )
        if not one.invocations:
            silent.append(stage.name)
            continue
        rows.extend(
            stage_rows(
                stage.name,
                stage.candidate,
                one,
                compulsory=compulsory,
                reps=args.capture_iters,
                floor=floor,
            )
        )
    record = {
        "provenance": payload(replace(block, after=contention(device_ordinal(device)))),
        "geometry": geometry(program),
        "floor": payload(floor),
        "operands": {name: payload(value) for name, value in operands.items()},
        "no_kernel_stages": silent,
        "rows": payload(rows),
    }
    print(traffic_table(rows))
    print(
        f"floor {float(floor.fixed_duration_us):.4f} us + bytes / "
        f"{float(floor.asymptotic_gbs):.3f} GB/s, max residual "
        f"{float(floor.max_residual_pct):.2f}%"
    )
    if silent:
        print(f"stages launching no kernel: {silent}")
    void = [row.site.label for row in rows if row.contaminated]
    if void:
        print(f"VOID -- contaminated DRAM counters at {void}")
    write_cell(args.out, key, record)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one pass.

    Args:
        argv: Arguments, or ``None`` for ``sys.argv``.

    Returns:
        Process exit status.

    Raises:
        ValueError: If ``--batch``, ``--iters``, ``--capture-iters`` or ``--warmup``
            is not positive. Warmup is not optional: a stage cannot be measured in
            isolation until a whole step has filled the slots it reads.
    """
    args = parse_args(argv)
    if args.batch <= 0:
        raise ValueError(f"--batch must be positive, got {args.batch}")
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.capture_iters <= 0:
        raise ValueError(f"--capture-iters must be positive, got {args.capture_iters}")
    if args.warmup <= 0:
        raise ValueError(f"--warmup must be positive, got {args.warmup}")
    if args.pass_name == "target":
        return run_target(args)
    if args.pass_name == "provenance":
        return run_provenance(args)
    if args.pass_name == "launch":
        return run_launch(args)
    return run_traffic(args)


if __name__ == "__main__":
    raise SystemExit(main())
