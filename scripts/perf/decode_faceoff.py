"""One-token autoregressive decode latency, SLinOSS against Mamba3, and the verdict.

    python3 scripts/perf/decode_faceoff.py --smoke
    python3 scripts/perf/decode_faceoff.py --json /tmp/faceoff.json
    python3 scripts/perf/decode_faceoff.py --bank /tmp/bank --boundary recurrence --only 3

The third form is how a run survives a fleet that offers minutes. Every measured cell is
written to its own JSON file under ``--bank``, under a temporary name and renamed, so an
interrupted invocation loses at most the cell in flight; the next invocation reads what is
there, skips it and continues. ``--order decisive`` puts the rows that can settle the
question first, so a four-minute window banks evidence rather than cell one of the grid.

Two boundaries, reported separately and never blended, because they are limited by
different resources:

- ``recurrence``: the ``T = 1`` state update alone. :func:`slinoss.ops.decode.decode_step`
  against Mamba3's ``mamba3_step_fn``. No projection, no convolution, no parameter map,
  no gate on either side.
- ``whole_step``: one token in, one token's output out, through the whole mixer step.
  :meth:`slinoss.mixer.SLinOSSMixer.step` against ``Mamba3.step``. The embedding and the
  language-model head are outside both sides, so neither pays a vocabulary. That method
  routes ``T == 1`` to :func:`slinoss.ops.decode.decode_step` behind an explicit branch, so
  a whole-step row reaches the decode boundary and the chunked scan is not on it. See
  :data:`ROUTING_DISCLOSURE`.

Fourteen disclosures, each of which makes a row mean something other than what it appears to
say. They print with every table and they are not footnotes.

:data:`VERSION_DISCLOSURE`. No figure crosses hosts, and none crosses torch versions
either. Every number in the table comes from this process, on this interpreter, on the one
card pinned for the run. Host dispatch cost moves with a torch minor version and the eager
rows are dominated by exactly that term, so a latency measured under another torch is not a
reference point here and none is quoted as one.

:data:`CONV_DISCLOSURE`. SLinOSS runs two causal convolutions and carries two convolution
state buffers; Mamba3 has no short convolution at all. That is an architectural cost
SLinOSS pays, it is a column in every whole-step table, and it is not dropped to level the
comparison whichever side wins.

:data:`FP32_DISCLOSURE`. ``so3ssd`` falls back to the reference at float32, and before
routing that made every float32 whole-step row a mixed path. It no longer does: the routed
``T == 1`` branch does not call ``so3ssd``, and conv, scanprep, decode and the mixer tail all
resolve to declared kernels at float32, so both boundaries are kernel paths at both dtypes.
Every row still carries its own resolved backend tuple rather than a single per-run claim.

:data:`EAGER_HOST_DISCLOSURE`. An eager host-bound row is recorded and never judged: its
median carries a half-width of 20 to 42 percent against the 10 percent margin a verdict
turns on. Those rows print ``jdg=n`` and enter no verdict.

:data:`INTERPRETER_DISCLOSURE`. The interpreter is a number in the table, not a caveat under
it, and its measured deltas bound the confound rather than converting between versions.

:data:`EAGER_DISCLOSURE`. The eager step at small batch is host enqueue, not device work,
and a graph replay removes host-induced idle and nothing else. So an eager small-batch row
compares two Python wrappers. The magnitude is torch-version-specific and is read off this
run's own eager-against-graph delta, never carried in. The graph rows are where an
architectural claim can live.

:data:`GRAPH_LAUNCH_DISCLOSURE`. At the graph boundary per-launch idle is bounded by a
fraction of a percent of the replay, so launch count is not a differentiator for either
architecture and a gap there is a bytes-and-kernel-quality gap.

:data:`SUB_L2_DISCLOSURE`. Below a measured batch of 10.6 to 24.4 the state crossing fits
in L2 and the fixed cost is weight streaming, so batches 1 and 8 are not attributable to
the recurrence at either boundary.

:data:`MAMBA_FP32_STATE_DISCLOSURE` and :data:`MAMBA_FUSION_DISCLOSURE` state what does not
shrink on Mamba3's side at bf16, and why the recurrence number is not a subset of the
whole-step number.

``d_state`` never matches across the two architectures. SLinOSS's is a multiple of 48 and
Mamba3's step kernel asserts one of 32, 64 or 128, so no value is legal on both sides and
"matched" is printed rather than assumed. :class:`MatchReport` carries what was held equal
and what could not be, and every row carries state bytes per token for both sides.

The verdict is computed here and not in prose. :func:`verdict` consumes the measured ratios
and returns one of :data:`DOMINATES`, :data:`COMPETITIVE`, :data:`NEITHER` plus the shape
class it quantified over. It refuses every positive word when a primary batch is missing.

:data:`CAPTURE_HAZARD_DISCLOSURE`. Building a second decode graph capture in one process
fires a device-side assert at some shapes, so a graph cell is measured one per process and
the loop lives outside. A device assert poisons the CUDA context, so a cell that dies on one
voids the process rather than the cell: nothing is banked, no table is printed, and the
invocation exits nonzero.

The card a row was measured on is a property of the row. :func:`admit` grades it: a card the
house gate admits is :data:`EXCLUSIVE_WITNESS` and owes one window, and a card at or below
the same utilization ceiling holding foreign memory above the same floor is
:data:`RESIDENCY_WITNESS` and owes :data:`RESIDENCY_REPLICATES` disjoint windows whose
medians must agree inside the sum of their half-widths. Neither threshold moves; the second
case pays in evidence instead. A card above the utilization ceiling is refused outright, and
a cell whose windows disagree is discarded rather than averaged or caveated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import torch
from torch import Tensor

from slinoss.config import HEAD_MULTIPLE, STATE_MULTIPLE, SLinOSSConfig
from slinoss.mixer import SLinOSSMixer
from slinoss.perf.ceiling import DramTimeFloor, dram_time_floor
from slinoss.perf.device import (
    FOREIGN_MIB_FLOOR,
    ClockPolicy,
    ContendedDevice,
    Contention,
    DeviceInfo,
    await_exclusive,
    clock_policy,
    contention,
    device_info,
    device_ordinal,
    require_cuda,
)
from slinoss.perf.timing import measure_paired
from slinoss.perf.units import Bytes, Microseconds, Percent
from slinoss.state import MixerState

# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------

PRIMARY_BATCHES: Final[tuple[int, ...]] = (1, 8, 32, 64, 128)
"""The batches the verdict quantifies over.

A verdict is refused unless every one of these carries a measured cell, so a run that lost
a batch to memory or to a shape guard cannot report a positive word over the rest.
"""

D_MODELS: Final[tuple[int, ...]] = (512, 1024, 2048)
SL_D_STATES: Final[tuple[int, ...]] = (96, 144, 192)
M3_SISO_D_STATES: Final[tuple[int, ...]] = (64, 128)
M3_MIMO_RANK: Final = 4

DTYPES: Final[dict[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

EXPAND: Final = 2
"""Inner width multiplier, held equal on both sides. Mamba3's own default.

An int and not 2.0: Mamba3 types the argument as an int, so a float here would either be
truncated at that call or refused by the checker, and a truncated expand gives the two sides
different inner widths. SLinOSS takes either.
"""

D_HEAD: Final = 64
"""Rows per head, the shipped value and Mamba3's ``headdim`` default.

Held fixed across the grid so ``d_head`` is not confounded with the head/group sharing
axis. See :data:`MEASURED_HEAD_ROWS` for why a literal ``n_heads = 4`` cannot replace it.
"""

MEASURED_HEAD_ROWS: Final[frozenset[int]] = frozenset((16, 48, 64, 96, 128))
"""``d_head`` values the scan's MMA N mode is measured to compile at.

:data:`slinoss.config.HEAD_MULTIPLE` admits every multiple of 16 and the tiled MMA fails IR
verification rather than padding at a width outside this list, so the config's guard is
necessary and not sufficient. 8 and 24 are measured to fail; 16, 48, 64, 96 and 128 to
pass. A ``d_head`` a multiple of 16 but absent here is rejected by name.

Compiling is forward legality and is not trainability, and the two part at 128. At
``3N = 144`` a ``d_head`` of 128 has a legal backward at ``chunk_size`` 16 and 32 with a
ceiling of 192 in both directions, and no legal backward at 64 or 128, so that shape trains
only at ``chunk_size <= 32`` and not at the shipped 64. It is not inference-only, and a row
carrying it would owe the ``chunk_size`` at which it trains rather than a caveat. No row
here carries it: :data:`D_HEAD` is 64 at every grid point, so the constraint bounds what
this list may be read to mean and describes no measured cell.
"""

MAMBA_D_STATES: Final[frozenset[int]] = frozenset((32, 64, 128))
"""``d_state`` values Mamba3's step kernel asserts.

``mamba3_step_fn`` asserts ``dstate in [32, 64, 128]``. Disjoint from every multiple of 48,
which is why no ``d_state`` is legal on both sides at once.
"""

ROPE_FRACTION: Final = 0.5
"""Mamba3's rope fraction, its own default. Sizes the angle state."""

D_CONV: Final = 4
"""SLinOSS causal convolution width. Mamba3 has no counterpart."""

G1: Final = "G1"
GMID: Final = "Gmid"
GHEAD: Final = "GH"
SHARINGS: Final[tuple[str, ...]] = (G1, GMID, GHEAD)
"""Head/group sharing cases, read as sharing and not as literal head counts.

``H4/G1`` and ``H8/G4`` name sharing regimes, not shapes: at ``d_model 2048`` with
``expand 2`` a literal ``n_heads = 4`` forces ``d_head = 1024``, outside
:data:`MEASURED_HEAD_ROWS`. The regimes are carried at the shipped ``d_head`` instead --
all heads sharing one ``B``/``C`` pair, four heads per group, and one pair per head -- and
the literal shape is carried only where it is legal, which :func:`literal_head_rejection`
reports is nowhere in this grid.
"""

HEADS_PER_GROUP_MID: Final = 4
"""Heads per group in the intermediate sharing case. The ``H8/G4`` ratio."""

LITERAL_N_HEADS: Final = 4
"""The literal head count ``H4/G1`` names, checked for legality per ``d_model``."""

SISO: Final = "siso"
MIMO: Final = "mimo"
MODES: Final[tuple[str, ...]] = (SISO, MIMO)

RECURRENCE: Final = "recurrence"
WHOLE_STEP: Final = "whole_step"
BOUNDARIES: Final[tuple[str, ...]] = (RECURRENCE, WHOLE_STEP)

EAGER: Final = "eager"
GRAPH: Final = "graph"
EXECUTIONS: Final[tuple[str, ...]] = (EAGER, GRAPH)

SLINOSS: Final = "slinoss"
MAMBA3: Final = "mamba3"

REFERENCE: Final = "reference"
"""The name every operator's reference backend registers under."""

KERNEL_PATH: Final = "kernel"
MIXED_PATH: Final = "mixed"
REFERENCE_PATH: Final = "reference"
PATHS: Final[tuple[str, ...]] = (KERNEL_PATH, MIXED_PATH, REFERENCE_PATH)
"""How much of a row ran in a declared kernel. See :func:`path_class`."""

HOST_REGIME: Final = "host"
SUB_L2_REGIME: Final = "sub_l2"
DRAM_REGIME: Final = "dram"
REGIMES: Final[tuple[str, ...]] = (HOST_REGIME, SUB_L2_REGIME, DRAM_REGIME)
"""What limits a row before its operator does. See :func:`regime`."""

REPLAY_IDLE_PCT_HIGH: Final = 1.68
"""Per-launch idle as a percent of one graph replay at batch 1, measured on this hardware.

A count of launches, not a duration, so it survives a torch version bump; a replay's gap
structure is recorded into the graph and is not re-enqueued per step.
"""

REPLAY_IDLE_PCT_LOW: Final = 0.05
"""The same fraction at batch 128. See :data:`REPLAY_IDLE_PCT_HIGH`."""

PRIOR_DRAM_CEILING_GBS: Final = 681.336
"""Prior single-point DRAM ceiling on this hardware, in GB/s.

Held only to check this run's own in-process fit. A ceiling is a property of the card, not
of the interpreter, so a disagreement with the in-process fit means the card is contended or
clocked differently, which is a reason to discard the samples rather than to keep them.
"""

PRIOR_DRAM_FIXED_US: Final = 4.1877
"""Fixed term of the prior DRAM time-law fit, in microseconds, charged once per launch."""

PRIOR_DRAM_RATE_GBS: Final = 684.640
"""Asymptotic term of the prior DRAM time-law fit, in GB/s."""

FIT_AGREEMENT_PCT: Final = 15.0
"""How far this run's fit may sit from :data:`PRIOR_DRAM_RATE_GBS` before it is called out.

Wide, because the prior fit was taken under a different torch and a different clock stamp
and only the card is shared; a disagreement beyond this is a card problem, not noise.
"""

EXCLUSIVE_WITNESS: Final = "exclusive"
RESIDENCY_WITNESS: Final = "residency"
WITNESSES: Final[tuple[str, ...]] = (EXCLUSIVE_WITNESS, RESIDENCY_WITNESS)
"""How a row earned the card it was taken on. Never blended, and printed per row.

:data:`EXCLUSIVE_WITNESS` is a row the house gate admitted outright. :data:`RESIDENCY_WITNESS`
is a row taken on a card at or below the gate's utilization ceiling but holding foreign
*residency* above its memory floor, which the gate refuses and which is idle in the only sense
that moves a latency. Such a row pays for the card in evidence instead: two disjoint windows,
each with its own build, capture and warmup, whose medians must agree inside the sum of their
half-widths. Neither threshold is relaxed to admit it.
"""

EXCLUSIVE_REPLICATES: Final = 1
RESIDENCY_REPLICATES: Final = 2
"""Windows a residency-witnessed cell must survive. Two, and they must agree."""

REPLICATE_GAP_S: Final = 5.0
"""Seconds between two replicate windows, with a fresh contention probe between them.

Disjoint means the second window shares no allocation, no graph and no warmup with the
first, and is separated by a probe that can still void the pair.
"""

BANK_SCHEMA: Final = 3
"""Version of the per-cell artifact layout. A bank at another version is refused, not read.

An appended field whose absent reading is documented is not a version change. ``schema`` is
in :data:`MATCHED_PROVENANCE`, so a bump refuses every banked cell; refusing a cell that
reads correctly, to announce a field it does not carry, costs the whole bank. A bump is for a
field whose meaning changed under a name that did not."""

GRAPH_CELLS_PER_PROCESS: Final = 1
"""Graph cells one process may measure.

Building a second decode graph capture in one process fires a device-side assert,
``indexSelectSmallIndex: srcIndex < srcSelectDimSize``, at some shapes, in an unmodified
tree. One is the budget until the rule is known; a cell over it is deferred to the next
invocation, so the driver loop lives outside the process. Raising this exposes the run.
"""

VOID_SUFFIX: Final = ".void.json"
"""Suffix of a void marker.

A void cell is recorded and not skipped: the marker names why it died and the cell stays
pending, so the next invocation measures it again. Read back as a record, never as a cell.
"""

POISON_MARKS: Final[tuple[str, ...]] = (
    "device-side assert",
    "indexselectsmallindex",
    "cuda error",
    "illegal memory access",
    "illegal instruction",
    "misaligned address",
)
"""Substrings that mark a failure as having poisoned the CUDA context.

Lowercase; matched against a lowercased message. Once one of these fires, a later launch in
the same process can still return numbers, and they mean nothing.
"""

DECISIVE: Final = "decisive"
NESTED: Final = "nested"
ORDERS: Final[tuple[str, ...]] = (DECISIVE, NESTED)
"""Cell order. :data:`DECISIVE` first emits the rows that can carry a verdict.

The fleet offers minutes, not hours, so the first window that opens has to buy a conclusion
rather than cell one of three hundred and sixty. :data:`NESTED` is the enumeration order, kept
so a run can reproduce the grid as written.
"""

# --------------------------------------------------------------------------
# The disclosures
# --------------------------------------------------------------------------

CONV_DISCLOSURE: Final = (
    "SLinOSS runs two causal convolutions (the value band, and the B/C key bands) and "
    "carries two convolution state buffers; Mamba3 has no short convolution and no conv "
    "state at all -- no d_conv, no causal_conv1d, and a d_conv passed to its constructor "
    "is swallowed by **kwargs. That cost is SLinOSS's, it is the cvbuf column, and it is "
    "not removed to level the comparison whichever side wins."
)

VERSION_DISCLOSURE: Final = (
    "NO FIGURE IN THIS TABLE CROSSES A HOST OR A TORCH VERSION. Every latency here was "
    "taken in this process, on the interpreter named in the liveness proof, on the one "
    "card pinned for the run. Host dispatch cost moves with a torch minor version and the "
    "eager rows are dominated by that term, so no latency measured under another torch is "
    "quoted as a reference point; the only cross-version quantities used at all are the "
    "structural ones -- the L2 crossover batch, the compulsory-traffic ratios, and the "
    "prior DRAM fit, which is printed beside this run's own in-process fit as a check and "
    "never substituted for it."
)

FP32_DISCLOSURE: Final = (
    "FLOAT32 IS NO LONGER A MIXED PATH AT EITHER BOUNDARY, MEASURED IN THIS TREE AFTER "
    "ROUTING. so3ssd registers cute and reference and resolves REFERENCE at float32, but the "
    "routed T=1 branch does not call so3ssd, and conv resolves native, scanprep cute, decode "
    "cute and the mixer tail cute at float32, so dispatch_verdict(decode) passes at float32 "
    "and both boundaries are kernel paths. so3ssd's missing float32 instantiation is a "
    "prefill property and describes no row here. The prediction that an fp32 whole-step row "
    "would be a mixed path refused a dominates was written before routing landed and is "
    "false in this tree. Only a row whose own resolved backend tuple names reference is "
    "refused a dominates."
)

EAGER_DISCLOSURE: Final = (
    "The eager step at small batch is host enqueue, not device work: the step sits above "
    "the CUDA driver as Python dispatch, and a graph replay removes host-induced idle and "
    "nothing else. An eager small-batch row therefore compares two Python wrappers, not "
    "two operators; if Mamba3's wrapper is thinner then SLinOSS loses that row for a "
    "reason with nothing to do with the recurrence. How large that term is depends on the "
    "torch version, so it is read off this run's own eager-against-graph delta and never "
    "carried in from another. The graph rows are where an architectural claim can live."
)

EAGER_HOST_DISCLOSURE: Final = (
    "NO EAGER ROW IS JUDGED, AT ALL, AND THE BATCH WAS THE WRONG PROXY FOR WHY. Measured "
    "here on the decisive class: the eager arms sit on flat host floors, SLinOSS 160-163 us "
    "and Mamba3 28-29 us at the recurrence boundary and 620-643 against 908-1,003 us at the "
    "whole step, so a row whose device work is under its own floor prices two Python "
    "wrappers however DRAM-bound the batch crossover calls it -- eager batch 8 read 8.53x "
    "its own DRAM floor against its graph twin's 1.04x. Above the floor the graph row of the "
    "same cell carries the same ratio to within 1.7 percent, so nothing is lost by refusing "
    "them all. The earlier form of this rule, which judged an eager row above the crossover, "
    "was falsified by that batch-8 pair. AN EAGER HOST-BOUND ROW IS NOT JUDGED EITHER. Measured by the instrument lane on this "
    "hardware and this torch version, exclusive, 600 iterations after 1,500 warmup: the "
    "host-bound eager cells carry median half-widths of 20 to 42 percent of their own "
    "medians, while the same cells under graph replay carry under 0.7 percent. A noise floor "
    "several times the 10 percent margin the verdict vocabulary has to discriminate cannot "
    "resolve any of the three words, and no iteration count fixes it because the variance is "
    "host scheduling and not device work. Such rows print for the record and are excluded "
    "from every verdict, which costs the eager class its primary batches and so refuses it a "
    "positive word by the missing-batch rule. Mamba3's step is capturable, so the graph rows "
    "capture both sides or neither; capturing one side only would void the comparison. The "
    "alternative admissible form for these rows is separate host-enqueue and device-busy "
    "columns, which this driver does not take."
)

SAMPLE_COUNT_DISCLOSURE_LABEL: Final = "SAMPLE COUNT IS NOT THE LEVER ON A WIDE BAND"

SAMPLE_COUNT_DISCLOSURE: Final = (
    f"{SAMPLE_COUNT_DISCLOSURE_LABEL}, MEASURED AND NOT ASSUMED. The prediction was that a "
    "band scales as one over the square root of the count, so twenty times the iterations "
    "would take the widest recurrence band from 30.4 percent to about 6.8 and inside the 10 "
    "percent margin. Re-measured at 1,000 iterations against 50, exclusive, same card and "
    "same process shape: the bf16 batch-8 recurrence band went 30.43 to 20.08 percent and "
    "the fp32 one 29.06 to 25.91, a 1.5x and a 1.1x narrowing against the 4.5x predicted. "
    "Over the same twenty-fold increase the full sample range widened rather than settled, "
    "SLinOSS 52.6 to 128.6 percent of its median and Mamba3 50.0 to 168.2, and the SLinOSS "
    "median itself moved 19.456 to 21.504 us, which is 10.5 percent and larger than the "
    "whole margin. Two samples of one cell that disagree by more than the margin do not "
    "average, so the band is a property of a heavy-tailed per-replay distribution and not of "
    "the sample size, and the row is refused rather than re-sampled. This falsifies the "
    "instruction that produced it: raising the count until the half-width clears the margin "
    "is not achievable at these shapes, and no wording in this driver promises it."
)

INTERPRETER_DISCLOSURE: Final = (
    "The interpreter is stated, not corrected for. Measured on one card, one tree, sixteen "
    "cells: against torch 2.6.0+cu124, torch 2.7.1+cu126 is 5 to 17 percent cheaper on the "
    "host-bound cells with every cell agreeing in sign, and 0.2 to 1.2 percent slower on the "
    "device-bound cells, also unanimous. The host term is torch's own dispatch and not a "
    "different op count: per-step CUDA entry-point counts are identical, driver-side total is "
    "flat within 0.3 percent, and the Python call count is bit-identical. Mamba3 forces "
    "2.7.1, so this run takes a slightly slower device and a cheaper host than any earlier "
    "SLinOSS figure. That direction is against SLinOSS on the device axis and so cannot "
    "manufacture a win. These deltas bound the confound; they are not a conversion factor "
    "and no number in this table is adjusted by them."
)

GRAPH_LAUNCH_DISCLOSURE: Final = (
    "At the graph boundary launch count is not a differentiator: per-launch replay idle "
    f"is bounded on this hardware by {REPLAY_IDLE_PCT_HIGH}% of the rep at batch 1 and "
    f"{REPLAY_IDLE_PCT_LOW}% at batch 128. A Mamba3-against-SLinOSS gap at a graph row is "
    "therefore a bytes-and-kernel-quality gap, and attributing one to launch overhead is "
    "refused by that bound."
)

SUB_L2_DISCLOSURE: Final = (
    "Two crossovers, one per boundary, because the two move different bytes. The decode "
    "kernel alone crosses out of L2 between batch 2.8 by full compulsory traffic and 5.7 by "
    "the tree's state-only formula. The whole step streams the parameter map every token "
    "and does not cross until batch 10.6 to 24.4, where DRAM traffic goes from 0.87x "
    "compulsory at batch 1 to 2.51x at batch 128. Below its own crossover a row's fixed "
    "cost is weight streaming, not state traffic, and docs/kernels.md gives no roofline "
    "verdict at a footprint under the cache: batch 1 is unjudged at both boundaries, batch "
    "8 is judged at the recurrence boundary and unjudged at the whole step."
)

MAMBA_FP32_STATE_DISCLOSURE: Final = (
    "Mamba3 pins angle_dt_state and ssm_state to float32 at every model dtype, so its "
    "dominant state term does not shrink at bf16 while SLinOSS's activation-dtype carries "
    "do. Its dt_bias, B_bias, C_bias, D and mimo projections are also built with a bare "
    "device= and no dtype=, so they are float32 as constructed at every requested dtype."
)

MAMBA_FUSION_DISCLOSURE: Final = (
    "The two boundaries use different Mamba3 fusion settings, so the recurrence number is "
    "NOT a subset of the whole-step number. At its shipped is_outproj_norm=False the step "
    "kernel also fuses the SiLU gate and the mimo_o down-reduction, which is Mamba3's "
    "fastest whole step and is what the whole-step rows measure. Only at "
    "is_outproj_norm=True do the gate and the reduction move out into _postprocess, "
    "leaving the kernel a bare state update, which is the only setting matching SLinOSS's "
    "recurrence boundary; that is what the recurrence rows measure. The triton rotary sits "
    "before the kernel in both settings and so is outside the recurrence boundary, "
    "matching slinoss.ops.scanprep being outside SLinOSS's."
)

ROUTING_DISCLOSURE: Final = (
    "ROUTED: SLinOSSMixer.step reaches the decode boundary at T=1, behind an explicit "
    "branch at the call site rather than inside a registry, and the branch also drops the "
    "three carry copies the T-token path ends with. So a whole-step row prices the decode "
    "kernel plus the projections, the two convolutions, the parameter map and the tail, and "
    "the backends column names decode rather than chunked_scan. Every whole-step figure "
    "taken before routing is void and not superseded: the routing lane measured 17,013.760 "
    "us chunked against 6,729.728 us routed at batch 128, -60.448 percent, 2.5281x, four "
    "repetitions agreeing to 60.430/60.440/60.458/60.448 and a paired interval excluding "
    "zero, with -41.58 percent at batch 8 and -21.86 percent at batch 1. Those figures are "
    "that lane's on this hardware, not this run's."
)

CAPTURE_HAZARD_DISCLOSURE: Final = (
    "ONE GRAPH CELL PER PROCESS. Building a second decode graph capture in one process "
    "fires a device-side assert, indexSelectSmallIndex: srcIndex < srcSelectDimSize, at "
    "some shapes. It reproduces in an unmodified tree, so it is not caused by routing and "
    "it is not caused by this driver. Until the rule is known the run is protected rather "
    "than worked around: a graph cell over the budget is deferred to the next invocation, "
    "never measured second. A device assert poisons the CUDA context, so any cell that dies "
    "on one voids the whole process -- nothing banked, no table, nonzero exit -- because "
    "every later cell in that process would produce numbers that look ordinary and are not."
)

BANK_DISCLOSURE: Final = (
    "A TABLE MAY BE ASSEMBLED ACROSS WINDOWS, NEVER ACROSS TREES, AND BOTH TREES COUNT. A "
    "banked cell carries the torch version, the card name and, for the slinoss package and "
    "the resolved mamba_ssm package alike, the directory the import machinery answered with "
    "and a digest of every .py file under it. A record that differs on any of those is "
    "refused and not corrected, so a kernel edit, a competitor source edit, a host change or "
    "a torch change empties the bank instead of blending two trees into one table. The "
    "competitor digest is taken from the imported module's own file and not from a path "
    "passed on the command line, so it names what answered. Each record also carries, read "
    "and not keyed, the copy's per-file sha256 manifest against the immutable .sources tree "
    "it came from, its commit if it carries one, and the dependency set that took the "
    "numbers: interpreter, environment root, torch, triton and apache-tvm-ffi. Cells are read "
    "back in queue order, so a table filled over several windows reads the same as one filled "
    "in a single pass."
)

WITNESS_DISCLOSURE: Final = (
    "THE CARD IS A PROPERTY OF THE ROW, IN THE wit COLUMN. exclusive1 is a row the house "
    "gate admitted: no foreign compute and foreign memory under the floor, one window. "
    "residency2 is a row on a card at or below the same 5 percent utilization ceiling that "
    "held foreign memory above the same 512 MiB floor, which that gate refuses. Neither "
    "threshold was moved to admit it and no contended sample is stamped: a card above the "
    "utilization ceiling is refused outright. The row pays for the weaker card in evidence "
    "instead, in two disjoint windows with their own modules, capture and warmup, whose "
    "medians must agree inside the sum of their own half-widths or one 1.024 us CUDA "
    "event tick, whichever is larger; the tick is the timer's own step, read off this "
    "tree's medians, and it is the only tolerance in the rule. A cell whose windows disagree is discarded; it is not averaged, "
    "it does not get a caveat, and it is not banked, so it is measured again. Two windows "
    "that agree are stronger evidence than one window that passed a memory threshold."
)

DISCLOSURES: Final[tuple[str, ...]] = (
    VERSION_DISCLOSURE,
    INTERPRETER_DISCLOSURE,
    ROUTING_DISCLOSURE,
    CONV_DISCLOSURE,
    FP32_DISCLOSURE,
    EAGER_DISCLOSURE,
    EAGER_HOST_DISCLOSURE,
    SAMPLE_COUNT_DISCLOSURE,
    GRAPH_LAUNCH_DISCLOSURE,
    SUB_L2_DISCLOSURE,
    MAMBA_FP32_STATE_DISCLOSURE,
    MAMBA_FUSION_DISCLOSURE,
    WITNESS_DISCLOSURE,
    CAPTURE_HAZARD_DISCLOSURE,
    BANK_DISCLOSURE,
)

L2_CROSSOVER_LOW: Final = 10.6
L2_CROSSOVER_HIGH: Final = 24.4
"""Measured batch range where the whole step's footprint leaves L2.

The whole step streams the parameter map every token, so its footprint is weight-dominated
and stays cache-masked long after the state crossing does not. See
:data:`SUB_L2_DISCLOSURE`.
"""

DECODE_CROSSOVER_LOW: Final = 2.8
DECODE_CROSSOVER_HIGH: Final = 5.7
"""The same crossover for the decode kernel alone, which moves state and nothing else.

The low edge is by full compulsory traffic, the high edge by the tree's state-only formula.
Batch 1 is below both and gets no roofline verdict; batch 8 is above both and is the first
primary batch that does.
"""

HOST_BOUND_RESOURCE: Final = (
    "host enqueue: at batch 1 the eager step is Python and launch marshaling rather than "
    "device work, so the row prices two wrappers; its size is torch-version-specific and "
    "is read off this run's own eager-against-graph delta at the same cell, not carried in"
)
DECODE_DRAM_RESOURCE: Final = (
    "DRAM bandwidth: the decode kernel moves 1.0009x compulsory bytes per launch and "
    "reaches 669.75 GB/s, 98.94% of the fitted DRAM floor, so it PASSES its declared "
    "DRAM-bound class and the residual 1.06% is not addressable in the kernel. An earlier "
    "profile of the same kernel read 357.64 GB/s and 52.77% and FAILED the class; an "
    "addressing transpose closed it, so that figure and every step time taken against it "
    "are void rather than superseded. Profiled by the kernel lane on this hardware at bf16 "
    "batch 128, 64 registers and zero spills, not by this run"
)
REFERENCE_RESOURCE: Final = (
    "none, and none is available: the scan resolved to the reference path, so the figure "
    "prices torch rather than a kernel and no limiting resource of the shipped operator "
    "can be read off it"
)

# --------------------------------------------------------------------------
# The verdict thresholds
# --------------------------------------------------------------------------

DOMINATES_RATIO: Final = 0.90
"""SLinOSS latency over Mamba3's, at or below which a point dominates. Inclusive."""

COMPETITIVE_GEOMEAN_RATIO: Final = 1.10
"""Geometric-mean ratio at or below which a class is competitive. Inclusive."""

WORST_POINT_RATIO: Final = 1.20
"""Worst single primary point a competitive class may carry. Inclusive."""

MARGIN_PCT: Final = 100.0 * (COMPETITIVE_GEOMEAN_RATIO - 1.0)
"""The tightest margin the three words turn on, as a percent.

Not a threshold of its own and never compared against a ratio. It is what a row's own
uncertainty is held to before the row is allowed to move a verdict: a vocabulary that
separates competitive from neither at ten percent cannot be adjudicated by a row whose ratio
is uncertain by more than ten percent, whichever way that uncertainty arrives. See
:func:`unresolved`.
"""

DOMINATES: Final = "dominates"
COMPETITIVE: Final = "competitive"
NEITHER: Final = "neither"
VERDICTS: Final[tuple[str, ...]] = (DOMINATES, COMPETITIVE, NEITHER)

IDLE_CEILING_PCT: Final = 5.0
"""Utilization a clean contention probe may report. Matches the gate default."""


class Witness(NamedTuple):
    """How one row earned the card it was measured on.

    Attributes:
        stamp: :data:`EXCLUSIVE_WITNESS`, :data:`RESIDENCY_WITNESS`, or "" when no probe
            was recorded. Printed per row; never averaged across rows.
        foreign_mib: Foreign device memory at admission, in MiB. Zero under an exclusive
            witness by construction.
        replicates: Timed windows behind the row. One under exclusive,
            :data:`RESIDENCY_REPLICATES` under residency.
        agrees: True when every replicate's median agreed with the first inside the sum of
            their half-widths. Always True on a row that exists, because a row whose
            replicates disagreed is discarded rather than recorded.
        detail: One sentence naming the stamp, the foreign memory and the agreement.
    """

    stamp: str
    foreign_mib: float
    replicates: int
    agrees: bool
    detail: str


NO_WITNESS: Final = Witness(
    stamp="",
    foreign_mib=0.0,
    replicates=0,
    agrees=False,
    detail="no contention witness recorded for this row",
)
"""The witness of a row nobody probed a card for. Prints as a dash.

Deliberately not :data:`EXCLUSIVE_WITNESS`: a default that claimed the strongest card would
make an unprobed row read as the best-evidenced one.
"""


def admit(
    probe: Contention,
    *,
    ceiling_pct: float = IDLE_CEILING_PCT,
    mib_floor: float = FOREIGN_MIB_FLOOR,
    exclusive_only: bool = False,
) -> tuple[str, int, str]:
    """Which witness a card qualifies for, and how many windows it then owes.

    Neither threshold moves. A card the house gate admits is admitted here unchanged and
    owes one window. A card at or below the utilization ceiling but holding foreign memory
    above the floor is refused by that gate and is admitted here only against a heavier
    requirement: :data:`RESIDENCY_REPLICATES` disjoint windows whose medians must agree. A
    card above the utilization ceiling is refused outright, which is the case this must
    never admit.

    Args:
        probe: A contention probe.
        ceiling_pct: Utilization ceiling. The house default, unchanged.
        mib_floor: Foreign-memory floor in MiB. The house default, unchanged.
        exclusive_only: Refuse the residency witness, leaving only the house gate.

    Returns:
        The witness stamp or "" when refused, the windows owed, and the reason either way.
    """
    if not probe.probed:
        return "", 0, f"refused: the contention probe returned nothing ({probe.detail})"
    if probe.quiet(ceiling_pct=ceiling_pct, mib_floor=mib_floor):
        return (
            EXCLUSIVE_WITNESS,
            EXCLUSIVE_REPLICATES,
            f"exclusive: {probe.stamp}, inside a gate of {ceiling_pct:.0f}% utilization "
            f"and {mib_floor:,.0f} MiB foreign memory",
        )
    if probe.utilization_pct > ceiling_pct:
        return (
            "",
            0,
            f"refused: {probe.stamp} is above the {ceiling_pct:.0f}% utilization ceiling, "
            f"so the card is running foreign compute and a sample on it would be void",
        )
    if exclusive_only:
        return (
            "",
            0,
            f"refused: {probe.stamp} carries foreign memory above {mib_floor:,.0f} MiB and "
            f"--exclusive-only was given, so the residency witness is not available",
        )
    return (
        RESIDENCY_WITNESS,
        RESIDENCY_REPLICATES,
        f"residency: {probe.stamp}. The card is idle in the only sense that moves a "
        f"latency -- no foreign compute -- but holds foreign memory above "
        f"{mib_floor:,.0f} MiB, so the house gate refuses it. Neither threshold is "
        f"relaxed: the row instead pays {RESIDENCY_REPLICATES} disjoint windows whose "
        f"medians must agree inside the sum of their half-widths, and is discarded if "
        f"they do not",
    )


CLOSE_SETTLE_S: Final = 1.0
"""Seconds to wait after the last timed region before probing the card.

The utilization the probe reads is the device's, not this process's share of it, and it is
averaged over the driver's own sampling period. Measured on this card: immediately after
``torch.cuda.synchronize`` the probe reads 93 to 100 percent utilization with zero foreign
processes, which is this run's own work, and voids the run against itself; the reading is
back to 0 percent 0.25 s later and stays there. This is four times that, and it buys the
closing check back rather than relaxing it: no threshold moves.
"""


def closing_probe(
    ordinal: int,
    *,
    device: torch.device,
    settle_s: float = CLOSE_SETTLE_S,
    probe: Callable[[int], Contention] = contention,
    rest: Callable[[float], None] = time.sleep,
    sync: Callable[[torch.device], None] | None = None,
) -> Contention:
    """Probe the card after the last timed region, once the reading is attributable.

    Args:
        ordinal: Device ordinal.
        device: The device to drain first, so queued work is not still running.
        settle_s: Seconds between the drain and the probe. See :data:`CLOSE_SETTLE_S`.
        probe: Contention probe, injected for testing.
        rest: Sleep, injected for testing.
        sync: Device drain, injected for testing. Defaults to ``torch.cuda.synchronize``.

    Returns:
        The closing probe. Read by :func:`sample_void`.
    """
    drain = torch.cuda.synchronize if sync is None else sync
    drain(device)
    rest(settle_s)
    return probe(ordinal)


def sample_void(
    after: Contention,
    *,
    witness: Witness = NO_WITNESS,
    ceiling_pct: float = IDLE_CEILING_PCT,
    mib_floor: float = FOREIGN_MIB_FLOOR,
    sampled: bool = True,
) -> str:
    """Why this run's samples do not stand, or the empty string when they do.

    The gate is armed once, before the first timed region, so it cannot see a foreign job
    that lands mid-run. This reads the closing probe against the same thresholds and voids
    the whole run when the card stopped being exclusive, because a sample taken on a
    contended card is void rather than caveated.

    A process that took no sample is not voided. The condition is about samples taken here,
    and a run that only reads a bank and renders has none: every row it prints already
    carries the witness of the window that measured it. Voiding a render would make a table
    unprintable whenever any tenant is on the card, which loses the table without protecting
    a number.

    The probe must come from :func:`closing_probe` and not from :func:`contention` directly:
    a probe taken the instant the last region ends reads this process's own utilization and
    voids every run against itself.

    Under a residency witness the closing probe cannot be quiet by construction, since the
    foreign memory that made it a residency witness is still there. The condition then is
    the one it was admitted under and no weaker: no foreign compute, and no new tenant, read
    as foreign memory that did not grow by more than ``mib_floor`` over its admitted value.

    Args:
        after: The contention probe taken after the last timed region.
        witness: What the run was admitted as. See :func:`admit`.
        ceiling_pct: Utilization the closing probe may report.
        mib_floor: Foreign device memory the closing probe may report, in MiB, and under a
            residency witness the growth it may report over the admitted value.
        sampled: Whether this process took a timed sample. False for a render out of a bank.

    Returns:
        A reason naming the closing stamp, or "" when the run stands.
    """
    if not sampled:
        return ""
    if witness.stamp == RESIDENCY_WITNESS:
        ceiling = witness.foreign_mib + mib_floor
        if (
            after.probed
            and after.utilization_pct <= ceiling_pct
            and after.foreign_memory_mib <= ceiling
        ):
            return ""
        return (
            f"VOID: the card admitted on residency did not hold that condition to the end "
            f"({after.stamp}), against no foreign compute above {ceiling_pct:.0f}% "
            f"utilization and foreign memory no higher than {ceiling:,.0f} MiB, being the "
            f"{witness.foreign_mib:,.0f} MiB admitted plus {mib_floor:,.0f} MiB of growth. "
            f"Every figure above is void and must be retaken, not caveated. No verdict is "
            f"reported."
        )
    if after.quiet(ceiling_pct=ceiling_pct, mib_floor=mib_floor):
        return ""
    return (
        f"VOID: the card was not exclusive when the last timed region ended "
        f"({after.stamp}), against a gate of {ceiling_pct:.0f}% utilization and "
        f"{mib_floor:,.0f} MiB foreign memory. Every figure above is void and must be "
        f"retaken on a quiet card, not caveated. No verdict is reported."
    )


TIMER_QUANTUM_US: Final = 1.024
"""The CUDA event timer's own step on this card, in microseconds.

Read off this tree's first banked window and not assumed: seven of the eight medians in it
were exact integer multiples of this value (227, 199, 116, 102, 19, 16 and 12 of them), the
eighth being the mean of two adjacent order statistics. It is the floor on how finely two
windows can be said to differ at all, so it is the floor on the replicate agreement reach in
:func:`agrees_within_half_widths`. Two medians one step apart are one measurement landing on
two adjacent ticks, and a check that called that a disagreement would discard every cell
whose duration sits near a tick boundary, forever.
"""


def agrees_within_half_widths(
    a_duration_us: float,
    a_resolution_pct: float,
    b_duration_us: float,
    b_resolution_pct: float,
    *,
    quantum_us: float = TIMER_QUANTUM_US,
) -> bool:
    """Whether two medians agree inside the sum of their own half-widths.

    The test is on the medians' own dispersion and one instrument limit: two windows that
    overlap on their reported intervals are one measurement taken twice, and two that do not
    are two different cards. The reach never falls below ``quantum_us``, because a half-width
    narrower than the timer's own step claims a resolution the timer does not have, and two
    medians on adjacent ticks are one measurement. That floor is a property of the clock, not
    of the card's occupancy: contention shows up as tens of percent, which is three orders of
    magnitude above one tick, so nothing contended passes because of it.

    Args:
        a_duration_us: First window's median, in microseconds.
        a_resolution_pct: Half-width of the interval on it, as a percent of it.
        b_duration_us: Second window's median.
        b_resolution_pct: Half-width of the interval on it.
        quantum_us: The timer's own step. See :data:`TIMER_QUANTUM_US`.

    Returns:
        True when the two intervals meet.
    """
    reach = (
        a_duration_us * a_resolution_pct + b_duration_us * b_resolution_pct
    ) / 100.0
    gap = abs(a_duration_us - b_duration_us)
    bound = max(reach, quantum_us)
    # Inclusive, and inclusive under float subtraction: a gap of exactly one tick comes out
    # of two float medians a few parts in 1e16 above the tick, and a bare <= would then
    # discard the one case the floor exists for.
    return gap <= bound or math.isclose(gap, bound, rel_tol=1e-9)


ESTIMATOR: Final = (
    "median over per-iteration CUDA-event samples; dispersion is the half-width of the "
    "distribution-free confidence interval on that median, read off the order statistics "
    "at 95% nominal coverage (slinoss.perf.units.Spread.resolution_pct), with the full "
    "range reported beside it as an outlier detector. The cross-architecture delta is "
    "slinoss.perf.dispersion.paired: both arms in one loop with the launch order swapped "
    "every iteration, so drift common to the pair cancels out of the difference."
)
"""How every figure in the table was reduced. Named because a margin without a dispersion
is not a margin, and a dispersion without an estimator is not a bound."""


def order_statistic_us(
    samples: Sequence[Microseconds], quantile: float
) -> Microseconds:
    """One quantile of a sample, as an order statistic.

    Args:
        samples: Timed samples, in any order. Not mutated.
        quantile: In ``[0, 1]``.

    Returns:
        The nearest-rank order statistic: the sorted sample at ``ceil(q*n) - 1``, clamped
        into range.

    Raises:
        ValueError: On an empty sample or a quantile outside ``[0, 1]``. An empty sample has
            no quantile and a zero would print as a measured duration.

    Nearest-rank and not interpolated, so every printed figure is a duration that was
    actually observed. An interpolated quantile of a bimodal sample -- which is what host
    run-ahead produces here -- lands between the two modes, at a latency the loop never ran.
    """
    if not samples:
        raise ValueError("no samples, so no quantile")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile {quantile} is outside [0, 1]")
    ordered = sorted(samples)
    rank = min(max(math.ceil(quantile * len(ordered)) - 1, 0), len(ordered) - 1)
    return Microseconds(ordered[rank])


# --------------------------------------------------------------------------
# Legality
# --------------------------------------------------------------------------


class Rejection(NamedTuple):
    """One grid point that was not enumerated, and why.

    Attributes:
        detail: What was rejected and which rule refused it. Names the offending field, so
            a reader can tell a ``d_state`` refusal from a ``d_head`` one without
            recomputing the arithmetic.
    """

    detail: str


def slinoss_rejection(
    *, d_model: int, d_state: int, d_head: int, n_groups: int
) -> str | None:
    """Why SLinOSS cannot run this shape, or None if it can.

    Checked here rather than by catching :class:`slinoss.config.SLinOSSConfig`'s own guard,
    for two reasons. The config admits every ``d_head`` that is a multiple of 16 and the
    MMA N mode does not, so one rule is missing there; and a grid that enumerates by
    construction has to state its refusals, which an exception does not carry to a table.

    Args:
        d_model: Residual-stream width.
        d_state: Per-head state width ``3N``.
        d_head: Rows per head ``P``.
        n_groups: Groups sharing one ``B``/``C`` pair.

    Returns:
        A sentence naming the offending field and the rule, or None.
    """
    if d_state < STATE_MULTIPLE or d_state % STATE_MULTIPLE != 0:
        return (
            f"d_state {d_state} is not a positive multiple of {STATE_MULTIPLE}: d_state "
            f"is 3N with N a multiple of {HEAD_MULTIPLE}"
        )
    if d_head < HEAD_MULTIPLE or d_head % HEAD_MULTIPLE != 0:
        return (
            f"d_head {d_head} is not a positive multiple of {HEAD_MULTIPLE}: P is the N "
            f"mode of two scan GEMMs"
        )
    if d_head not in MEASURED_HEAD_ROWS:
        return (
            f"d_head {d_head} is a multiple of {HEAD_MULTIPLE} but outside the measured "
            f"MMA N-mode list {sorted(MEASURED_HEAD_ROWS)}: the tiled MMA fails IR "
            f"verification rather than padding"
        )
    d_inner = round(EXPAND * d_model)
    if d_inner % d_head != 0:
        return f"d_inner {d_inner} is not divisible by d_head {d_head}"
    n_heads = d_inner // d_head
    if n_groups < 1 or n_heads % n_groups != 0:
        return (
            f"n_groups {n_groups} does not divide n_heads {n_heads}: a group holds a "
            f"whole number of heads"
        )
    return None


def mamba_rejection(
    *, d_model: int, d_state: int, d_head: int, n_groups: int
) -> str | None:
    """Why Mamba3 cannot run this shape, or None if it can.

    Args:
        d_model: Residual-stream width.
        d_state: Mamba3's ``d_state``.
        d_head: Mamba3's ``headdim``.
        n_groups: Mamba3's ``ngroups``.

    Returns:
        A sentence naming the offending field and the rule, or None.
    """
    if d_state not in MAMBA_D_STATES:
        return (
            f"d_state {d_state} is outside {sorted(MAMBA_D_STATES)}: mamba3_step_fn "
            f"asserts dstate in [32, 64, 128]"
        )
    d_inner = round(EXPAND * d_model)
    if d_inner % d_head != 0:
        return f"d_inner {d_inner} is not divisible by headdim {d_head}"
    n_heads = d_inner // d_head
    if n_groups not in (1, n_heads):
        return (
            f"ngroups {n_groups} is neither 1 nor nheads {n_heads}: _preprocess expands B "
            f"and C from a size-ngroups axis to nheads, and expand admits only those two"
        )
    return None


def literal_head_rejection(d_model: int) -> str | None:
    """Why the literal ``n_heads = 4`` shape is illegal at this ``d_model``, or None.

    Reported rather than silently dropped: the grid is asked for ``H4/G1``, and the honest
    answer at every width here is that the shape does not compile.

    Args:
        d_model: Residual-stream width.

    Returns:
        A sentence naming the forced ``d_head``, or None if the shape is legal.
    """
    d_inner = round(EXPAND * d_model)
    if d_inner % LITERAL_N_HEADS != 0:
        return (
            f"literal n_heads={LITERAL_N_HEADS} at d_model {d_model}: d_inner {d_inner} "
            f"is not divisible by {LITERAL_N_HEADS}"
        )
    forced = d_inner // LITERAL_N_HEADS
    if forced in MEASURED_HEAD_ROWS:
        return None
    return (
        f"literal n_heads={LITERAL_N_HEADS} at d_model {d_model} forces d_head {forced}, "
        f"outside the measured MMA N-mode list {sorted(MEASURED_HEAD_ROWS)}; carried as "
        f"the {G1} sharing case at d_head {D_HEAD} instead"
    )


def group_count(sharing: str, n_heads: int) -> int:
    """Groups for one sharing case.

    Args:
        sharing: One of :data:`SHARINGS`.
        n_heads: Heads in the layer.

    Returns:
        ``G``. One for :data:`G1`, ``n_heads`` for :data:`GHEAD`, and ``n_heads // 4`` for
        :data:`GMID`, clamped to at least one so a narrow layer collapses onto :data:`G1`
        rather than producing a zero.

    Raises:
        ValueError: On an unknown sharing case.
    """
    if sharing == G1:
        return 1
    if sharing == GHEAD:
        return n_heads
    if sharing == GMID:
        return max(1, n_heads // HEADS_PER_GROUP_MID)
    raise ValueError(f"unknown sharing {sharing!r}; have {list(SHARINGS)}")


# --------------------------------------------------------------------------
# State bytes per token
# --------------------------------------------------------------------------


class StateBytes(NamedTuple):
    """Persistent decode state one layer holds for one sequence, itemized.

    One token per sequence per step, so this is also the state one token's step must read
    and write, which is what makes it the quantity a ``d_state`` mismatch is audited
    against. A model over the allocated shapes and dtypes, not a measurement: no counter is
    read here and no verdict divides by it.

    Attributes:
        recurrent_bytes: The recurrent state proper. SLinOSS's ``ssm``; Mamba3's
            ``ssm_state``. Both float32 at every low-precision activation dtype.
        conv_bytes: Convolution state. SLinOSS's ``conv`` and ``keys`` together. Exactly
            zero for Mamba3, which has no short convolution.
        carry_bytes: Everything else the step carries. SLinOSS's ``b_prev`` and
            ``u_prev``; Mamba3's ``angle_dt_state``, ``k_state`` and ``v_state``.
        total_bytes: The three above.
        conv_buffer_count: Convolution state buffers. Two for SLinOSS, zero for Mamba3. An
            integer so the disclosure is a column and not only a sentence.
    """

    recurrent_bytes: Bytes
    conv_bytes: Bytes
    carry_bytes: Bytes
    total_bytes: Bytes
    conv_buffer_count: int


def state_dtype_bytes(dtype: torch.dtype) -> int:
    """Element size of the recurrent state at an activation dtype.

    :meth:`slinoss.state.MixerState.allocate` holds ``ssm`` in float32 at every activation
    dtype below it, and in float64 only when the activations are float64.

    Args:
        dtype: Activation dtype.

    Returns:
        Bytes per recurrent-state element.
    """
    return 8 if dtype is torch.float64 else 4


def slinoss_state_bytes(config: SLinOSSConfig, *, dtype: torch.dtype) -> StateBytes:
    """SLinOSS decode state for one sequence in one layer.

    Every term is the shape :meth:`slinoss.state.MixerState.allocate` allocates::

        ssm     d_inner * d_state           state dtype
        conv    (d_conv-1) * d_inner        activation dtype
        keys    (d_conv-1) * 2*G*d_state    activation dtype
        b_prev  G * d_state                 activation dtype
        u_prev  d_inner                     activation dtype

    ``keys`` is allocated whatever ``key_conv`` says, and is counted here only when
    ``key_conv`` is on, because that is when the step reads and writes it.

    Args:
        config: Shape contract.
        dtype: Activation dtype.

    Returns:
        The itemized bytes.
    """
    activation = torch.empty((), dtype=dtype).element_size()
    state = state_dtype_bytes(dtype)
    window = config.d_conv - 1
    recurrent = state * config.d_inner * config.d_state
    conv = activation * window * config.d_inner
    keys = (
        activation * window * 2 * config.n_groups * config.d_state
        if config.key_conv
        else 0
    )
    carry = activation * (config.n_groups * config.d_state + config.d_inner)
    return StateBytes(
        recurrent_bytes=Bytes(recurrent),
        conv_bytes=Bytes(conv + keys),
        carry_bytes=Bytes(carry),
        total_bytes=Bytes(recurrent + conv + keys + carry),
        conv_buffer_count=2 if config.key_conv else 1,
    )


def rope_angles(d_state: int, *, fraction: float = ROPE_FRACTION) -> int:
    """Angle-state width Mamba3 derives from ``d_state``.

    ``s = int(d_state*fraction)``, decremented to even, halved. Reproduced rather than
    approximated: the angle state is the one Mamba3 term that is not a plain product of the
    widths, and rounding it would move a reported byte count.

    Args:
        d_state: Mamba3's ``d_state``.
        fraction: Mamba3's ``rope_fraction``.

    Returns:
        ``num_rope_angles``.
    """
    span = int(d_state * fraction)
    if span % 2:
        span -= 1
    return span // 2


def mamba_state_bytes(
    *,
    d_model: int,
    d_state: int,
    d_head: int,
    dtype: torch.dtype,
    rank: int = 1,
    fraction: float = ROPE_FRACTION,
) -> StateBytes:
    """Mamba3 decode state for one sequence in one layer.

    Every term is the shape ``Mamba3.allocate_inference_cache`` allocates::

        angle_dt_state  nheads * num_rope_angles     float32, pinned
        ssm_state       nheads * headdim * d_state    float32, pinned
        k_state         R * nheads * d_state          model dtype
        v_state         nheads * headdim              model dtype

    ``ngroups`` does not appear: ``B`` and ``C`` are expanded to ``nheads`` on both paths,
    so sharing changes the projection width and not the state. The two float32 pins are
    what :data:`MAMBA_FP32_STATE_DISCLOSURE` reports.

    Args:
        d_model: Residual-stream width.
        d_state: Mamba3's ``d_state``.
        d_head: Mamba3's ``headdim``.
        dtype: Model dtype, carried by ``k_state`` and ``v_state`` only.
        rank: ``mimo_rank``, or one under SISO.
        fraction: Mamba3's ``rope_fraction``.

    Returns:
        The itemized bytes. ``conv_bytes`` is zero and ``conv_buffer_count`` is zero:
        Mamba3 has no short convolution.
    """
    model = torch.empty((), dtype=dtype).element_size()
    n_heads = round(EXPAND * d_model) // d_head
    angle = 4 * n_heads * rope_angles(d_state, fraction=fraction)
    recurrent = 4 * n_heads * d_head * d_state
    keys = model * rank * n_heads * d_state
    values = model * n_heads * d_head
    return StateBytes(
        recurrent_bytes=Bytes(recurrent),
        conv_bytes=Bytes(0),
        carry_bytes=Bytes(angle + keys + values),
        total_bytes=Bytes(recurrent + angle + keys + values),
        conv_buffer_count=0,
    )


# --------------------------------------------------------------------------
# Compulsory traffic and the floor
# --------------------------------------------------------------------------


class MovedBytes(NamedTuple):
    """Compulsory DRAM traffic for one decode step, itemized.

    Compulsory and not measured: the least any implementation of this step could move, so a
    floor built on it is a lower bound and a measured duration can only sit above it. Read
    the terms, not just the total; the total is a sum of three assumptions.

    Attributes:
        weight_bytes: Parameters read once. Zero at the recurrence boundary, where the
            operands are already formed and no parameter map is read.
        state_bytes: Recurrent state read and written once, for the whole batch.
        activation_bytes: The token in and the token out, for the whole batch. Zero at the
            recurrence boundary, where the operand set is not modeled.
        total_bytes: The sum.
    """

    weight_bytes: int
    state_bytes: int
    activation_bytes: int
    total_bytes: int


def moved_bytes(
    *,
    boundary: str,
    param_bytes: int,
    state: StateBytes,
    batch: int,
    d_model: int,
    dtype: torch.dtype,
) -> MovedBytes:
    """Compulsory traffic for one step at one boundary.

    Args:
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        param_bytes: Parameter bytes of the module under test, at its constructed dtypes.
        state: Per-sequence state bytes for the same side.
        batch: Sequences stepped at once.
        d_model: Residual-stream width.
        dtype: Activation dtype.

    Returns:
        The itemization.

    Raises:
        ValueError: On an unknown boundary, or a non-positive batch.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    # The activation element size, not state_dtype_bytes: the token in and out are in the
    # activation dtype, while the recurrent state is pinned wider and is already counted
    # inside state.total_bytes.
    element = torch.empty((), dtype=dtype).element_size()
    crossing = 2 * batch * int(state.total_bytes)
    if boundary == RECURRENCE:
        return MovedBytes(
            weight_bytes=0,
            state_bytes=crossing,
            activation_bytes=0,
            total_bytes=crossing,
        )
    activations = 2 * batch * d_model * element
    total = param_bytes + crossing + activations
    return MovedBytes(
        weight_bytes=param_bytes,
        state_bytes=crossing,
        activation_bytes=activations,
        total_bytes=total,
    )


class FloorPair(NamedTuple):
    """Both sides' distance from their own DRAM floor, or the reason there is none.

    Attributes:
        available: True only when a floor was fitted in this process and the row is in the
            :data:`DRAM_REGIME`. A footprint under L2 gets no roofline verdict.
        slinoss_moved_bytes: SLinOSS compulsory bytes.
        slinoss_floor_us: The fitted floor for those bytes.
        slinoss_x_floor: Measured over floor. One means at the floor.
        mamba_moved_bytes: The same for Mamba3.
        mamba_floor_us: The same for Mamba3.
        mamba_x_floor: The same for Mamba3.
        detail: One sentence, printed whichever side wins.
    """

    available: bool
    slinoss_moved_bytes: int
    slinoss_floor_us: Microseconds
    slinoss_x_floor: float
    mamba_moved_bytes: int
    mamba_floor_us: Microseconds
    mamba_x_floor: float
    detail: str


def floor_pair(
    *,
    regime_name: str,
    slinoss_bytes: MovedBytes,
    mamba_bytes: MovedBytes,
    slinoss_duration_us: Microseconds,
    mamba_duration_us: Microseconds,
    fit: DramTimeFloor | None,
) -> FloorPair:
    """Price both sides against their own compulsory-traffic floor.

    Each side is judged against its own footprint and not against the other's: the two
    architectures move different numbers of bytes, so one shared floor would credit the side
    that moves fewer.

    Args:
        regime_name: One of :data:`REGIMES`.
        slinoss_bytes: SLinOSS compulsory traffic.
        mamba_bytes: Mamba3 compulsory traffic.
        slinoss_duration_us: Measured SLinOSS median.
        mamba_duration_us: Measured Mamba3 median.
        fit: The floor fitted in this process, or None when none was.

    Returns:
        The pair, with ``available`` False and a reason in ``detail`` when no floor applies.

    Raises:
        ValueError: On an unknown regime.
    """
    if regime_name not in REGIMES:
        raise ValueError(f"unknown regime {regime_name!r}; have {list(REGIMES)}")
    empty = FloorPair(
        available=False,
        slinoss_moved_bytes=slinoss_bytes.total_bytes,
        slinoss_floor_us=Microseconds(0.0),
        slinoss_x_floor=0.0,
        mamba_moved_bytes=mamba_bytes.total_bytes,
        mamba_floor_us=Microseconds(0.0),
        mamba_x_floor=0.0,
        detail="",
    )
    if fit is None:
        return empty._replace(
            detail="no floor: none was fitted in this process, and a fit from another "
            "process or another host does not price this row"
        )
    if regime_name != DRAM_REGIME:
        return empty._replace(
            detail=f"no floor: the row is {regime_name}, and docs/kernels.md gives no "
            f"roofline verdict at a footprint under the cache or under a host-bound step"
        )
    sl_floor = fit.floor_us(Bytes(slinoss_bytes.total_bytes))
    m3_floor = fit.floor_us(Bytes(mamba_bytes.total_bytes))
    sl_x = float(slinoss_duration_us) / float(sl_floor)
    m3_x = float(mamba_duration_us) / float(m3_floor)
    return FloorPair(
        available=True,
        slinoss_moved_bytes=slinoss_bytes.total_bytes,
        slinoss_floor_us=sl_floor,
        slinoss_x_floor=sl_x,
        mamba_moved_bytes=mamba_bytes.total_bytes,
        mamba_floor_us=m3_floor,
        mamba_x_floor=m3_x,
        detail=(
            f"slinoss is {sl_x:.2f}x its own floor at "
            f"{slinoss_bytes.total_bytes:,} compulsory bytes; mamba3 is {m3_x:.2f}x its "
            f"own floor at {mamba_bytes.total_bytes:,}"
        ),
    )


def fit_cross_check(fit: DramTimeFloor | None) -> str:
    """Compare this run's DRAM fit with the prior fit on the same hardware.

    Args:
        fit: The in-process fit, or None.

    Returns:
        One line. Names a disagreement past :data:`FIT_AGREEMENT_PCT`, which indicts the
        card rather than the samples.
    """
    if fit is None:
        return "dram fit: none taken in this process, so no row carries a floor"
    rate = float(fit.asymptotic_gbs)
    off = 100.0 * (rate - PRIOR_DRAM_RATE_GBS) / PRIOR_DRAM_RATE_GBS
    verdict_word = "agrees with" if abs(off) <= FIT_AGREEMENT_PCT else "DISAGREES WITH"
    return (
        f"dram fit here: {float(fit.fixed_duration_us):.4f} us + bytes / {rate:.3f} GB/s, "
        f"max residual {float(fit.max_residual_pct):.2f}%; {verdict_word} the prior fit "
        f"{PRIOR_DRAM_FIXED_US} us + bytes / {PRIOR_DRAM_RATE_GBS} GB/s "
        f"(ceiling {PRIOR_DRAM_CEILING_GBS} GB/s) by {off:+.2f}% on the rate"
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def nearest_mamba_d_state(d_state: int, *, legal: Sequence[int] | None = None) -> int:
    """The Mamba3 ``d_state`` closest to a SLinOSS one, by ratio.

    By ratio and not by difference: 96 sits 32 from both 64 and 128 in absolute terms, and
    a tie broken arbitrarily would be a silent choice. On the ratio 128 is nearer at every
    value this grid carries, which is what makes SLinOSS 144 against Mamba3 128 the nearest
    pair.

    Args:
        d_state: SLinOSS's ``d_state``.
        legal: Candidate Mamba3 widths. Defaults to :data:`MAMBA_D_STATES`.

    Returns:
        The nearest candidate.

    Raises:
        ValueError: On a non-positive ``d_state``, or an empty candidate list.
    """
    if d_state < 1:
        raise ValueError(f"d_state must be positive, got {d_state}")
    options = sorted(MAMBA_D_STATES) if legal is None else sorted(legal)
    if not options:
        raise ValueError("no candidate Mamba3 d_state to match against")
    return min(options, key=lambda other: abs(math.log(other / d_state)))


class MatchReport(NamedTuple):
    """What one row held equal across the two architectures, and what it could not.

    ``d_state_matched`` is never True over this grid: SLinOSS's width is a multiple of 48
    and Mamba3's is one of 32, 64 or 128, and those sets are disjoint. The field exists so
    the mismatch is a value a reader and a test can both read, rather than a fact the table
    leaves out.

    Attributes:
        d_model: Held equal.
        dtype: Held equal, as a string.
        batch: Held equal.
        layers: Held equal.
        held: Field names held equal, for the row's own record.
        slinoss_d_state: SLinOSS's ``3N``.
        mamba_d_state: Mamba3's ``d_state``.
        d_state_ratio: SLinOSS's over Mamba3's.
        d_state_matched: True only if the two widths are equal.
        slinoss_state_bytes: SLinOSS state per token per layer.
        mamba_state_bytes: Mamba3 state per token per layer.
        state_bytes_ratio: SLinOSS's total over Mamba3's.
        slinoss_param_count: Parameters in the measured SLinOSS layer, or zero when no
            module was built.
        mamba_param_count: Parameters in the measured Mamba3 layer, or zero.
        param_matched: True only if the two counts are equal and non-zero.
        detail: One line stating the mismatch. Always names ``d_state`` and both values; a
            row whose detail did not would be hiding the thing this record exists to
            report.
    """

    d_model: int
    dtype: str
    batch: int
    layers: int
    held: tuple[str, ...]
    slinoss_d_state: int
    mamba_d_state: int
    d_state_ratio: float
    d_state_matched: bool
    slinoss_state_bytes: StateBytes
    mamba_state_bytes: StateBytes
    state_bytes_ratio: float
    slinoss_param_count: int
    mamba_param_count: int
    param_matched: bool
    detail: str


def match_shapes(
    config: SLinOSSConfig,
    *,
    dtype: torch.dtype,
    batch: int,
    mamba_d_state: int,
    mamba_d_head: int = D_HEAD,
    rank: int = 1,
    slinoss_param_count: int = 0,
    mamba_param_count: int = 0,
) -> MatchReport:
    """Pair one SLinOSS shape with one Mamba3 shape and report the mismatch.

    Args:
        config: The SLinOSS layer, one layer.
        dtype: Activation dtype, held equal.
        batch: Sequences stepped at once, held equal.
        mamba_d_state: Mamba3's ``d_state``.
        mamba_d_head: Mamba3's ``headdim``.
        rank: Mamba3's ``mimo_rank``, or one under SISO.
        slinoss_param_count: Parameters in the built SLinOSS layer.
        mamba_param_count: Parameters in the built Mamba3 layer.

    Returns:
        The report. ``detail`` names ``d_state`` and both widths whenever they differ.
    """
    slinoss_bytes = slinoss_state_bytes(config, dtype=dtype)
    mamba_bytes = mamba_state_bytes(
        d_model=config.d_model,
        d_state=mamba_d_state,
        d_head=mamba_d_head,
        dtype=dtype,
        rank=rank,
    )
    d_state_matched = config.d_state == mamba_d_state
    param_matched = (
        slinoss_param_count == mamba_param_count and slinoss_param_count != 0
    )
    ratio = config.d_state / mamba_d_state
    if d_state_matched:
        detail = f"d_state matched at {config.d_state}"
    else:
        detail = (
            f"d_state NOT matched: slinoss 3N={config.d_state} against mamba3 "
            f"d_state={mamba_d_state}, ratio {ratio:.4f}; the two legality sets are "
            f"disjoint (multiples of {STATE_MULTIPLE} against {sorted(MAMBA_D_STATES)}), "
            f"so no value is legal on both sides. State bytes per token per layer: "
            f"slinoss {int(slinoss_bytes.total_bytes):,} against mamba3 "
            f"{int(mamba_bytes.total_bytes):,}, ratio "
            f"{slinoss_bytes.total_bytes / mamba_bytes.total_bytes:.4f}"
        )
    if not param_matched:
        detail += (
            f". Parameter count NOT matched: slinoss {slinoss_param_count:,} against "
            f"mamba3 {mamba_param_count:,}"
        )
    return MatchReport(
        d_model=config.d_model,
        dtype=str(dtype),
        batch=batch,
        layers=config.n_layers,
        held=("d_model", "dtype", "batch", "layers", "expand", "d_head"),
        slinoss_d_state=config.d_state,
        mamba_d_state=mamba_d_state,
        d_state_ratio=ratio,
        d_state_matched=d_state_matched,
        slinoss_state_bytes=slinoss_bytes,
        mamba_state_bytes=mamba_bytes,
        state_bytes_ratio=slinoss_bytes.total_bytes / mamba_bytes.total_bytes,
        slinoss_param_count=slinoss_param_count,
        mamba_param_count=mamba_param_count,
        param_matched=param_matched,
        detail=detail,
    )


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """One cell of the faceoff, legal on both sides by construction.

    Attributes:
        d_model: Residual-stream width, held equal.
        dtype_name: Key of :data:`DTYPES`.
        batch: Sequences stepped at once, held equal.
        slinoss_d_state: SLinOSS's ``3N``.
        mamba_d_state: Mamba3's ``d_state``.
        sharing: One of :data:`SHARINGS`.
        mode: :data:`SISO` or :data:`MIMO`.
    """

    d_model: int
    dtype_name: str
    batch: int
    slinoss_d_state: int
    mamba_d_state: int
    sharing: str
    mode: str

    @property
    def dtype(self) -> torch.dtype:
        """The activation dtype."""
        return DTYPES[self.dtype_name]

    @property
    def rank(self) -> int:
        """Mamba3's ``mimo_rank``. One under SISO."""
        return M3_MIMO_RANK if self.mode == MIMO else 1

    @property
    def n_heads(self) -> int:
        """Heads in the layer, shared by both sides at :data:`D_HEAD`."""
        return round(EXPAND * self.d_model) // D_HEAD

    @property
    def n_groups(self) -> int:
        """``G``, from the sharing case."""
        return group_count(self.sharing, self.n_heads)

    @property
    def config(self) -> SLinOSSConfig:
        """The SLinOSS layer this cell measures.

        One layer, and no vocabulary: the boundaries exclude the embedding and the head, so
        a vocabulary would add a GEMM neither side is being compared on.
        """
        return SLinOSSConfig(
            d_model=self.d_model,
            d_state=self.slinoss_d_state,
            expand=EXPAND,
            d_head=D_HEAD,
            n_groups=self.n_groups,
            d_conv=D_CONV,
            key_conv=True,
            n_layers=1,
            vocab_size=None,
        )

    @property
    def shape_class(self) -> str:
        """The class a verdict quantifies over. Every axis but the batch."""
        return (
            f"{self.dtype_name}/d_model={self.d_model}/sl_3N={self.slinoss_d_state}"
            f"/m3_d_state={self.mamba_d_state}/{self.sharing}/G={self.n_groups}"
            f"/H={self.n_heads}/P={D_HEAD}/{self.mode}"
        )

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.shape_class} B={self.batch} rank={self.rank} "
            f"d_inner={round(EXPAND * self.d_model)}"
        )


class Grid(NamedTuple):
    """Every enumerated cell, and every combination that was refused.

    Attributes:
        points: Cells legal on both sides.
        rejections: Refusals, each naming the offending field and the rule.
    """

    points: tuple[GridPoint, ...]
    rejections: tuple[Rejection, ...]


def enumerate_grid(
    *,
    d_models: Sequence[int] = D_MODELS,
    dtype_names: Sequence[str] = tuple(DTYPES),
    batches: Sequence[int] = PRIMARY_BATCHES,
    slinoss_d_states: Sequence[int] = SL_D_STATES,
    mamba_d_states: Sequence[int] = M3_SISO_D_STATES,
    sharings: Sequence[str] = SHARINGS,
    modes: Sequence[str] = MODES,
) -> Grid:
    """Enumerate the cells legal on both sides, and record every refusal.

    A cell is emitted only when both :func:`slinoss_rejection` and :func:`mamba_rejection`
    return None, so nothing downstream has to re-check a shape. Every refusal is kept: a
    grid that dropped illegal combinations silently would report a verdict over a set
    nobody can reconstruct.

    Args:
        d_models: Residual-stream widths.
        dtype_names: Keys of :data:`DTYPES`.
        batches: Batches.
        slinoss_d_states: SLinOSS ``3N`` values.
        mamba_d_states: Mamba3 ``d_state`` values.
        sharings: Head/group sharing cases.
        modes: :data:`SISO`, :data:`MIMO`, or both.

    Returns:
        The grid.

    Raises:
        ValueError: On an unknown dtype name, sharing case, or mode. A typo that enumerated
            nothing would otherwise read as a grid with no legal cell.
    """
    for name in dtype_names:
        if name not in DTYPES:
            raise ValueError(f"unknown dtype {name!r}; have {list(DTYPES)}")
    for sharing in sharings:
        if sharing not in SHARINGS:
            raise ValueError(f"unknown sharing {sharing!r}; have {list(SHARINGS)}")
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; have {list(MODES)}")

    points: list[GridPoint] = []
    rejections: list[Rejection] = []
    seen: set[str] = set()

    for d_model in d_models:
        literal = literal_head_rejection(d_model)
        if literal is not None and literal not in seen:
            seen.add(literal)
            rejections.append(Rejection(literal))

    for d_model in d_models:
        n_heads = round(EXPAND * d_model) // D_HEAD
        for sharing in sharings:
            groups = group_count(sharing, n_heads)
            for sl_state in slinoss_d_states:
                sl_bad = slinoss_rejection(
                    d_model=d_model, d_state=sl_state, d_head=D_HEAD, n_groups=groups
                )
                if sl_bad is not None:
                    key = f"slinoss d_model={d_model} {sharing} 3N={sl_state}: {sl_bad}"
                    if key not in seen:
                        seen.add(key)
                        rejections.append(Rejection(key))
                    continue
                for m3_state in mamba_d_states:
                    m3_bad = mamba_rejection(
                        d_model=d_model,
                        d_state=m3_state,
                        d_head=D_HEAD,
                        n_groups=groups,
                    )
                    if m3_bad is not None:
                        key = (
                            f"mamba3 d_model={d_model} {sharing} d_state={m3_state}: "
                            f"{m3_bad}"
                        )
                        if key not in seen:
                            seen.add(key)
                            rejections.append(Rejection(key))
                        continue
                    for mode in modes:
                        for dtype_name in dtype_names:
                            for batch in batches:
                                points.append(
                                    GridPoint(
                                        d_model=d_model,
                                        dtype_name=dtype_name,
                                        batch=batch,
                                        slinoss_d_state=sl_state,
                                        mamba_d_state=m3_state,
                                        sharing=sharing,
                                        mode=mode,
                                    )
                                )
    return Grid(points=tuple(points), rejections=tuple(rejections))


# --------------------------------------------------------------------------
# What actually ran, and in what regime
# --------------------------------------------------------------------------


class Resolved(NamedTuple):
    """Which backend every stage of one row resolved to, and what that makes the row.

    Printed per row and not once per run: the resolution depends on the dtype, and at
    float32 the scan falls back while the conv does not, so one name would describe neither
    path. See :data:`FP32_DISCLOSURE`.

    Attributes:
        names: ``stage=backend`` for every stage on the measured boundary, in call order.
        path: :data:`KERNEL_PATH` when no stage fell back, :data:`REFERENCE_PATH` when the
            operator did, :data:`MIXED_PATH` when some other stage did but the operator
            did not.
        detail: One sentence naming the fallbacks, or stating there were none.
    """

    names: tuple[str, ...]
    path: str
    detail: str


def path_class(stages: Mapping[str, str]) -> str:
    """Classify a row by how much of it ran in a declared kernel.

    The scan is singled out because it is the operator under test. A row whose scan fell
    back prices torch, and no amount of native conv around it makes the figure a statement
    about the shipped operator.

    Args:
        stages: Backend name per stage. A ``scan``, ``chunked_scan``, ``so3ssd`` or
            ``decode`` key names the operator. ``chunked_scan`` is retained so a record
            banked before routing classifies the same way it did when it was measured.

    Returns:
        :data:`KERNEL_PATH`, :data:`MIXED_PATH` or :data:`REFERENCE_PATH`.
    """
    operator = [
        name
        for stage, name in stages.items()
        if stage in ("scan", "chunked_scan", "so3ssd", "decode")
    ]
    if any(name == REFERENCE for name in operator):
        return REFERENCE_PATH
    if any(name == REFERENCE for name in stages.values()):
        return MIXED_PATH
    return KERNEL_PATH


def resolve_stages(*, boundary: str, dtype: torch.dtype) -> Resolved:
    """Ask every registry on one boundary what it resolves for this dtype.

    Args:
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        dtype: Activation dtype the row runs at.

    Returns:
        The resolution.

    Raises:
        ValueError: On an unknown boundary.
    """
    import slinoss.ops.conv.backends as conv_backends
    import slinoss.ops.decode.backends as decode_backends
    import slinoss.ops.mixer.backends as mixer_backends
    import slinoss.ops.scanprep.backends as prep_backends

    if boundary == RECURRENCE:
        wanted = (("decode", decode_backends),)
    elif boundary == WHOLE_STEP:
        # decode and not chunked_scan, and so3ssd is absent rather than resolved: the T=1
        # branch calls the decode boundary, so the scan registry is off this row and asking
        # it would print a backend the row never runs. See ROUTING_DISCLOSURE.
        wanted = (
            ("conv", conv_backends),
            ("prep", prep_backends),
            ("decode", decode_backends),
            ("tail", mixer_backends),
        )
    else:
        raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")

    stages = {
        stage: module.resolve(None, "cuda", dtype).name for stage, module in wanted
    }
    fell_back = [stage for stage, name in stages.items() if name == REFERENCE]
    if fell_back:
        detail = (
            f"{list(fell_back)} resolved to {REFERENCE} at {dtype}; this row prices torch "
            f"for those stages, not the shipped operator"
        )
    else:
        detail = f"every stage resolved to a declared kernel at {dtype}"
    if boundary == WHOLE_STEP:
        detail += (
            "; ROUTED: the operator here is the decode kernel, reached by the T=1 "
        )
        detail += (
            "branch in SLinOSSMixer.step, plus the projections, the two convolutions, "
        )
        detail += "the parameter map and the tail"
    return Resolved(
        names=tuple(f"{stage}={name}" for stage, name in stages.items()),
        path=path_class(stages),
        detail=detail,
    )


def crossover(boundary: str) -> tuple[float, float]:
    """The batch range below which a row's footprint is cache-resident.

    Two ranges, not one, because the two boundaries move different bytes. The decode
    boundary moves state and nothing else and crosses out of L2 early; the whole step
    streams the parameter map every token and stays cache-masked far longer.

    Args:
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.

    Returns:
        Low and high edge of the measured crossover.

    Raises:
        ValueError: On an unknown boundary.
    """
    if boundary == RECURRENCE:
        return DECODE_CROSSOVER_LOW, DECODE_CROSSOVER_HIGH
    if boundary == WHOLE_STEP:
        return L2_CROSSOVER_LOW, L2_CROSSOVER_HIGH
    raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")


def regime(*, batch: int, execution: str, boundary: str = WHOLE_STEP) -> str:
    """What limits a row before its operator does.

    Args:
        batch: Sequences stepped at once.
        execution: :data:`EAGER` or :data:`GRAPH`.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`, which pick different measured
            crossovers. Defaults to the later-crossing one, so an omitted boundary
            withholds a roofline verdict rather than granting one.

    Returns:
        A short label. ``host`` when the eager wrapper dominates, ``sub_l2`` when the state
        crossing is cache-resident, ``dram`` otherwise.
    """
    high = crossover(boundary)[1]
    if execution == EAGER and batch <= high:
        return HOST_REGIME
    if batch <= high:
        return SUB_L2_REGIME
    return DRAM_REGIME


def default_resource(
    *, path: str, batch: int, execution: str, boundary: str = WHOLE_STEP
) -> str:
    """The limiting resource a row is already known to sit behind.

    Supplied so a :data:`NEITHER` is never printed with an unnamed resource when the
    resource is already measured on this fleet. Every string it returns names its own
    provenance, because a figure from another lane's profile is not a profile taken here.

    Args:
        path: :data:`KERNEL_PATH`, :data:`MIXED_PATH` or :data:`REFERENCE_PATH`.
        batch: Sequences stepped at once.
        execution: :data:`EAGER` or :data:`GRAPH`.
        boundary: Which crossover applies. See :func:`crossover`.

    Returns:
        The resource, or the empty string when none is known and a profile is owed.
    """
    low, high = crossover(boundary)
    if path != KERNEL_PATH:
        return REFERENCE_RESOURCE
    if execution == EAGER and batch <= high:
        return HOST_BOUND_RESOURCE
    if batch <= high:
        return (
            f"weight bytes: the batch is below the measured crossover of {low}-{high} for "
            f"the {boundary} boundary, where the state crossing is cache-resident and the "
            f"fixed cost is weight streaming"
        )
    if boundary == RECURRENCE:
        return DECODE_DRAM_RESOURCE
    return ""


def measured_resource(cell: Cell) -> str:
    """Name the resource behind a gap from the row's own floor, not from a profile.

    The whole-step boundary has no NCU profile on this fleet, so :func:`default_resource`
    owes it a string and returns none. The row itself carries enough to name the resource
    without one: each side's measured distance from its own fitted DRAM floor, and each
    side's compulsory bytes. A gap whose byte term times its floor-distance term reproduces
    the measured ratio is a bandwidth gap, and the factorization says how much of it is
    state size rather than kernel efficiency. The residual is printed rather than hidden,
    because a factorization that misses the measurement is not an explanation of it.

    Args:
        cell: The worst-ratio row of the class. Its floor must be available; a row under L2
            or from a process that took no fit gets no resource here.

    Returns:
        The resource, or the empty string when the row carries no floor, which leaves the
        verdict to print that a profile is owed.
    """
    floor = cell.floor
    if (
        not floor.available
        or floor.mamba_moved_bytes <= 0
        or floor.mamba_x_floor <= 0.0
    ):
        return ""
    byte_term = floor.slinoss_moved_bytes / floor.mamba_moved_bytes
    reach_term = floor.slinoss_x_floor / floor.mamba_x_floor
    predicted = byte_term * reach_term
    residual = Percent(100.0 * (cell.ratio / predicted - 1.0))
    return (
        f"DRAM bandwidth, priced on this row and not read from another lane's profile: at "
        f"batch {cell.point.batch} slinoss sits {floor.slinoss_x_floor:.2f}x its own fitted "
        f"floor and mamba3 {floor.mamba_x_floor:.2f}x its own, so the measured "
        f"{cell.ratio:.4f}x factors into a byte term {byte_term:.4f}x "
        f"({floor.slinoss_moved_bytes:,} against {floor.mamba_moved_bytes:,} compulsory "
        f"bytes) and a floor-distance term {reach_term:.4f}x, whose product "
        f"{predicted:.4f}x reproduces the measurement to {residual:+.2f}%. The byte term is "
        f"the state the two operators carry, so it moves only by changing d_state, not by "
        f"changing the kernel"
    )


def judgeable(*, batch: int, execution: str, boundary: str) -> str:
    """Why a row may not enter a verdict, or the empty string when it may.

    No eager row is judged. The batch was the proxy for host-bound and this run measured it
    wrong: at the recurrence boundary the eager arms sit on flat host floors of 160-163 us
    for SLinOSS against 28-29 us for Mamba3, so an eager row below that floor prices two
    Python wrappers at any batch the crossover calls DRAM-bound, and the eager batch-8 row
    read 8.53x its own DRAM floor while its graph twin read 1.04x. Above the floor the graph
    row of the same cell carries the ratio anyway, within 1.7 percent. So the eager rows are
    recorded, and the architectural claim lives on the graph rows.
    See :data:`EAGER_HOST_DISCLOSURE`.

    Args:
        batch: Sequences stepped at once.
        execution: :data:`EAGER` or :data:`GRAPH`.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.

    Returns:
        A reason, or "" when the row is judgeable.
    """
    if execution != EAGER:
        return ""
    if regime(batch=batch, execution=execution, boundary=boundary) == HOST_REGIME:
        return (
            f"not judged: {execution} at batch {batch} is host-bound at the {boundary} "
            f"boundary, where the median's half-width is 20-42% against a 10% margin"
        )
    floors = (
        "160-163 us against mamba3's 28-29"
        if boundary == RECURRENCE
        else "620-643 us against mamba3's 908-1,003"
    )
    return (
        f"not judged: {execution} at batch {batch} sits on a measured host floor at the "
        f"{boundary} boundary, {floors} us, so the row prices two Python wrappers wherever "
        f"the device work is under it; the graph row of this cell carries the ratio"
    )


def unresolved(cell: Cell, *, margin_pct: float = MARGIN_PCT) -> str:
    """Why the instrument cannot adjudicate this row's ratio, or "" when it can.

    :func:`judgeable` refuses a row for what it is: a regime, known before the row runs. This
    refuses a row for what it measured. Both sources of uncertainty are converted to the same
    thing, a percentage of the ratio, and held to the same margin the three words turn on:

    - The clock. Two medians inside one :data:`TIMER_QUANTUM_US` tick are one measurement
      landing on two adjacent ticks, and the ratio between them is then set by the tick and
      by whatever fixed cost both arms share. Priced onto both medians, one tick is
      ``100 * tick * (1/a + 1/b)`` percent of the ratio, and where that exceeds the margin
      the row is refused. No iteration count moves it: the tick is the clock's step, not a
      sample count. Both conditions are needed. A tick is negligible on a long row, so a
      long row whose two arms tie is a measured tie and stays; and a large gap is a measured
      gap however coarse the clock, so it stays too.
    - The order statistics. The two half-widths sum to the ratio's own band, and where that
      band exceeds the margin the row is refused. This one an iteration count does move,
      so the reason says what count the row carries.

    The gate only ever removes a row from a verdict. It cannot admit one, and it cannot
    produce a positive word: a refused primary batch refuses :data:`DOMINATES` outright in
    :func:`verdict`, exactly as an unmeasured one does.

    Args:
        cell: The measured row.
        margin_pct: The margin the vocabulary discriminates. See :data:`MARGIN_PCT`.

    Returns:
        A reason, or "" when the row may enter a verdict.
    """
    slinoss, mamba = float(cell.slinoss_duration_us), float(cell.mamba_duration_us)
    gap = abs(slinoss - mamba)
    # Inclusive under float subtraction, as in agrees_within_half_widths: two medians exactly
    # one tick apart differ by a few parts in 1e16 more than the tick, and a bare <= would
    # then admit the case this refuses.
    adjacent = gap <= TIMER_QUANTUM_US or math.isclose(
        gap, TIMER_QUANTUM_US, rel_tol=1e-9
    )
    quantum_pct = 100.0 * TIMER_QUANTUM_US * (1.0 / slinoss + 1.0 / mamba)
    if adjacent and quantum_pct > margin_pct:
        return (
            f"not judged: the two medians lie within one {TIMER_QUANTUM_US:.3f} us timer "
            f"tick ({slinoss:,.3f} against {mamba:,.3f} us), where one tick priced onto "
            f"both is {quantum_pct:.2f}% of the ratio against the {margin_pct:.0f}% margin "
            f"the vocabulary discriminates, so the {cell.ratio:.4f} it prints is the "
            f"timer's step and not a measured difference; no iteration count moves it"
        )
    band_pct = float(cell.slinoss_resolution_pct) + float(cell.mamba_resolution_pct)
    if band_pct > margin_pct:
        count = (
            f"{cell.iters:,d} timed iterations" if cell.iters else "an unrecorded count"
        )
        return (
            f"not judged: the two half-widths sum to {band_pct:.2f}% of the ratio "
            f"({cell.slinoss_resolution_pct:.2f}% and {cell.mamba_resolution_pct:.2f}%) "
            f"against the {margin_pct:.0f}% margin the vocabulary discriminates, at "
            f"{count}, over full sample ranges of {cell.slinoss_spread_pct:.1f}% and "
            f"{cell.mamba_spread_pct:.1f}% of the two medians. The count is on the record "
            f"so a second sample at another count can be set against this one; no count is "
            f"asserted to close the band, because the half-width is set by the dispersion "
            f"and not by n alone. See {SAMPLE_COUNT_DISCLOSURE_LABEL}"
        )
    return ""


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


class Verdict(NamedTuple):
    """One of three words over one shape class at one boundary, and the arithmetic.

    Attributes:
        word: :data:`DOMINATES`, :data:`COMPETITIVE` or :data:`NEITHER`.
        shape_class: What the word applies to. Printed with the word, always: a verdict
            without its class is a claim about an unstated set.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        path: :data:`KERNEL_PATH`, :data:`MIXED_PATH` or :data:`REFERENCE_PATH`. Anything
            but the first refuses :data:`DOMINATES`.
        batches: Primary batches that carried a measured ratio and were judged, ascending.
        missing_batches: Primary batches with no measured ratio. Non-empty refuses every
            positive word.
        refused_batches: Primary batches measured and not judged, by :func:`unresolved` or
            :func:`judgeable`. Non-empty refuses :data:`DOMINATES` only.
        geomean_ratio: Geometric mean of SLinOSS over Mamba3 across ``batches``.
        worst_ratio: Largest single ratio.
        worst_batch: The batch carrying it.
        best_ratio: Smallest single ratio.
        gap_pct: ``geomean_ratio - 1`` as a percentage. Positive means SLinOSS is slower.
            Reported for every word, since :data:`NEITHER` has to carry the exact gap.
        limiting_resource: The resource behind a :data:`NEITHER`.
        caveats: Every disclosure bearing on this class. A :data:`DOMINATES` carries none
            by construction, which is what the brief's "with no caveat" means here.
        detail: The sentence a report prints.
    """

    word: str
    shape_class: str
    boundary: str
    execution: str
    path: str
    batches: tuple[int, ...]
    missing_batches: tuple[int, ...]
    geomean_ratio: float
    worst_ratio: float
    worst_batch: int
    best_ratio: float
    gap_pct: Percent
    limiting_resource: str
    caveats: tuple[str, ...]
    detail: str
    refused_batches: tuple[int, ...] = ()


def verdict(
    ratios: Mapping[int, float],
    *,
    shape_class: str,
    boundary: str,
    execution: str,
    path: str = KERNEL_PATH,
    primary: Sequence[int] = PRIMARY_BATCHES,
    limiting_resource: str = "",
    refused: Mapping[int, str] | None = None,
) -> Verdict:
    """Decide which of the three words a measured shape class earns.

    The rules, in the order they are applied:

    - A missing primary batch refuses every positive word. Quantifying over the cells that
      happened to run is the failure mode this exists to prevent, so the refusal is not
      limited to :data:`DOMINATES`: a geometric mean over a subset is exactly as misleading
      as a universal claim over one.
    - A primary batch ``refused`` by the instrument refuses :data:`DOMINATES`, and only
      that. It is not the same as a missing one and the two are not merged: a missing batch
      is a hole in coverage, where what the class does at that shape is unknown, while a
      refused batch is a measured row whose ratio the instrument cannot place against a
      margin. The class is covered, so a geometric mean over the rest is a statement about
      measured shapes, and it prints with the refusal and its reason attached. The universal
      claim is still gone, because "at every primary batch" cannot rest on a row that was
      not judged.
    - :data:`DOMINATES` when every primary ratio is at or below :data:`DOMINATES_RATIO`
      and ``path`` is :data:`KERNEL_PATH`. Inclusive, so exactly 0.90 dominates. The path
      condition is the brief's "with no caveat" made mechanical: a win recorded while
      SLinOSS ran its reference implementation is a fact about torch.
    - :data:`COMPETITIVE` when the geometric mean is at or below
      :data:`COMPETITIVE_GEOMEAN_RATIO` and no primary ratio exceeds
      :data:`WORST_POINT_RATIO`. Both inclusive.
    - :data:`NEITHER` otherwise, carrying the exact gap and a named limiting resource.

    Args:
        ratios: SLinOSS latency over Mamba3's, keyed by batch. Entries outside ``primary``
            are ignored: the thresholds are defined over the primary batches, and a
            non-primary cell that moved a geometric mean would move a verdict nobody
            quantified over.
        shape_class: What the verdict applies to.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        path: How much of the class ran in a declared kernel.
        primary: Batches the verdict quantifies over.
        limiting_resource: Resource behind a gap. Filled from :func:`default_resource`
            when omitted and one is already known.
        refused: Why the instrument would not judge a primary batch, keyed by batch. From
            :func:`unresolved`. A batch here is expected to be absent from ``ratios``; one
            that appears in both is refused, since a row that was not judged may not move a
            number either.

    Returns:
        The verdict.

    Raises:
        ValueError: If no primary batch carries a ratio, which leaves nothing to quantify
            over; if any ratio is not positive, which leaves the geometric mean undefined;
            or on an unknown ``path``.
    """
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}; have {list(PATHS)}")
    unfit = dict(refused or {})
    present = tuple(
        batch for batch in sorted(primary) if batch in ratios and batch not in unfit
    )
    missing = tuple(
        batch for batch in sorted(primary) if batch not in ratios and batch not in unfit
    )
    declined = tuple(batch for batch in sorted(primary) if batch in unfit)
    if not present:
        raise ValueError(
            f"no primary batch carries a ratio; primary is {sorted(primary)} and the "
            f"measured batches are {sorted(ratios)}"
        )
    values = [ratios[batch] for batch in present]
    for batch, value in zip(present, values, strict=True):
        if value <= 0.0:
            raise ValueError(
                f"ratio at batch {batch} is {value}, not positive; a latency ratio of "
                f"zero or less is not a measurement"
            )
    geomean = math.exp(sum(math.log(value) for value in values) / len(values))
    worst = max(values)
    worst_batch = present[values.index(worst)]
    best = min(values)
    gap = Percent(100.0 * (geomean - 1.0))

    caveats: list[str] = []
    if path != KERNEL_PATH:
        caveats.append(FP32_DISCLOSURE if "fp32" in shape_class else REFERENCE_RESOURCE)
    if execution == EAGER:
        caveats.append(EAGER_DISCLOSURE)
    if any(batch <= crossover(boundary)[1] for batch in present):
        caveats.append(SUB_L2_DISCLOSURE)
    if boundary == WHOLE_STEP:
        caveats.append(CONV_DISCLOSURE)
        caveats.append(ROUTING_DISCLOSURE)
    caveats.extend(unfit[batch] for batch in declined)
    caveats.append(MAMBA_FUSION_DISCLOSURE)

    excluded = (
        ""
        if not declined
        else (
            f" Primary batches {list(declined)} were measured and not judged, so nothing "
            f"here quantifies over them and {DOMINATES} is refused whatever the remaining "
            f"ratios say."
        )
    )
    resource = limiting_resource or default_resource(
        path=path, batch=worst_batch, execution=execution, boundary=boundary
    )
    if missing:
        word = NEITHER
        detail = (
            f"{NEITHER} over {shape_class} at {boundary}/{execution}: no verdict is "
            f"quantified, because primary batches {list(missing)} carry no measured cell. "
            f"Measured batches {list(present)} give a geometric mean of {geomean:.4f} and "
            f"a worst point of {worst:.4f} at batch {worst_batch}, and neither licenses a "
            f"word over the primary set.{excluded}"
        )
    elif (
        all(value <= DOMINATES_RATIO for value in values)
        and path == KERNEL_PATH
        and not declined
    ):
        word = DOMINATES
        detail = (
            f"{DOMINATES} over {shape_class} at {boundary}/{execution}: slinoss is at or "
            f"below {DOMINATES_RATIO:.2f}x mamba3 at every primary batch {list(present)}; "
            f"worst {worst:.4f} at batch {worst_batch}, best {best:.4f}, geometric mean "
            f"{geomean:.4f}. Path {path}."
        )
    elif geomean <= COMPETITIVE_GEOMEAN_RATIO and worst <= WORST_POINT_RATIO:
        word = COMPETITIVE
        blocked = (
            ""
            if path == KERNEL_PATH
            else (
                f" {DOMINATES} was refused on path {path} whatever the ratios say: "
                f"slinoss did not run its kernel here."
            )
        )
        detail = (
            f"{COMPETITIVE} over {shape_class} at {boundary}/{execution}: geometric mean "
            f"{geomean:.4f} is within "
            f"{100.0 * (COMPETITIVE_GEOMEAN_RATIO - 1.0):.0f}% and no primary batch "
            f"exceeds {WORST_POINT_RATIO:.2f}x; worst {worst:.4f} at batch {worst_batch}, "
            f"best {best:.4f}.{blocked}{excluded}"
        )
    else:
        word = NEITHER
        named = resource or (
            "UNNAMED -- a gap without a profile-backed limiting resource is not a report"
        )
        detail = (
            f"{NEITHER} over {shape_class} at {boundary}/{execution}: geometric mean "
            f"{geomean:.4f} ({gap:+.2f}%), worst {worst:.4f} at batch {worst_batch}, best "
            f"{best:.4f}. Limiting resource: {named}.{excluded}"
        )
    return Verdict(
        word=word,
        shape_class=shape_class,
        boundary=boundary,
        execution=execution,
        path=path,
        batches=present,
        missing_batches=missing,
        geomean_ratio=geomean,
        worst_ratio=worst,
        worst_batch=worst_batch,
        best_ratio=best,
        gap_pct=gap,
        limiting_resource=resource,
        caveats=() if word == DOMINATES else tuple(caveats),
        detail=detail,
        refused_batches=declined,
    )


def unjudged(
    *,
    shape_class: str,
    boundary: str,
    execution: str,
    refused: Mapping[int, str] | None = None,
) -> Verdict:
    """The verdict for a class not one of whose rows was fit to be judged.

    :func:`verdict` raises when no primary batch carries a ratio, which is right for a lost
    cell and wrong here: nothing was lost, the rows exist and are unfit to be judged. The
    word is :data:`NEITHER` because the vocabulary has no fourth word, and the detail says
    that no comparison was attempted rather than that one failed.

    Args:
        shape_class: What would have been judged.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        refused: Per-batch reasons from :func:`unresolved`, when the instrument gate is what
            emptied the class. Omitted, the class is the eager one and the reason is its
            regime, which is known without measuring anything.

    Returns:
        A verdict carrying no arithmetic.
    """
    unfit = dict(refused or {})
    if unfit:
        reasons = "; ".join(unfit[batch] for batch in sorted(unfit))
        detail = (
            f"{NEITHER} over {shape_class} at {boundary}/{execution}: NO COMPARISON WAS "
            f"ATTEMPTED. Every row in this class was measured and none was fit to be "
            f"judged: {reasons}."
        )
    else:
        detail = (
            f"{NEITHER} over {shape_class} at {boundary}/{execution}: NO COMPARISON WAS "
            f"ATTEMPTED. Every row in this class is host-bound eager, whose median "
            f"half-width is 20-42% against the 10% margin the vocabulary discriminates, so "
            f"none entered a verdict. Read the graph rows of the same class instead."
        )
    return Verdict(
        word=NEITHER,
        shape_class=shape_class,
        boundary=boundary,
        execution=execution,
        path=KERNEL_PATH,
        batches=(),
        missing_batches=() if unfit else tuple(sorted(PRIMARY_BATCHES)),
        geomean_ratio=float("nan"),
        worst_ratio=float("nan"),
        worst_batch=0,
        best_ratio=float("nan"),
        gap_pct=Percent(float("nan")),
        limiting_resource="" if unfit else HOST_BOUND_RESOURCE,
        caveats=tuple(unfit[batch] for batch in sorted(unfit))
        or (EAGER_HOST_DISCLOSURE,),
        detail=detail,
        refused_batches=tuple(sorted(unfit)),
    )


# --------------------------------------------------------------------------
# Liveness
# --------------------------------------------------------------------------


class Liveness(NamedTuple):
    """What every registry resolved to, in the process that took the numbers.

    Dispatch falls back to a reference path silently and answers every call, so a table
    without this is a table that may have measured torch. Printed verbatim and checked:
    :attr:`live` is False when any registry resolved to its reference at ``dtype``.

    Attributes:
        lines: One line per registry, naming its registered backends and what it resolved
            to.
        live: True only if every stage of every measured boundary resolved to a declared
            kernel at ``dtype``, and the dispatch verdict passed, and a decode kernel is
            registered. Computed over the boundaries the run measures and not over every
            registry: after routing ``so3ssd`` is on neither boundary, so its float32
            reference would otherwise mark a fully kernel row as not live.
        loaded: True if a decode kernel backend is registered at all. False means the
            extension did not build or did not import, which is a broken tree rather than a
            dtype with no instantiation, and no row taken in that process means anything.
        recurrence_live: True if the decode registry resolved to a kernel at ``dtype``.
            Independent of :attr:`live`: the decode backend registers for float32 while
            so3ssd does not, so the two boundaries are not live at the same dtypes.
        slinoss_package: Resolved package directory, so the report stamps the tree it
            measured rather than whichever checkout was first on the path.
        torch_version: The torch the numbers were taken on.
        detail: The dispatch verdict's own sentence.
    """

    lines: tuple[str, ...]
    live: bool
    loaded: bool
    recurrence_live: bool
    slinoss_package: str
    torch_version: str
    detail: str


def liveness(*, dtype: torch.dtype, boundaries: Sequence[str] = BOUNDARIES) -> Liveness:
    """Ask every registry on the decode path what it resolves for this device.

    Imported here rather than at module scope: this module is importable, and
    :mod:`slinoss.perf.dispatch` pulls in every operator's backend module.

    Args:
        dtype: Activation dtype the measurement runs at. A kernel backend declares the
            dtypes it has an instantiation for, so a dtype with no fast path resolves to
            the reference and this reports it.
        boundaries: The boundaries this run measures. Decides which resolutions
            :attr:`Liveness.live` is computed over; every registry is printed either way.

    Returns:
        The proof.
    """
    import slinoss
    import slinoss.ops.conv.backends as conv_backends
    import slinoss.ops.decode.backends as decode_backends
    import slinoss.ops.mixer.backends as mixer_backends
    import slinoss.ops.scanprep.backends as prep_backends
    import slinoss.ops.so3ssd.backends as scan_backends
    from slinoss.perf.dispatch import dispatch_verdict
    from slinoss.perf.workload import DECODE

    lines: list[str] = []
    recurrence_live = False
    # Every registry, including one no boundary is on: the proof is what the tree
    # registered, and so3ssd resolving its reference at float32 is worth printing even
    # where it prices nothing.
    for label, module in (
        ("so3ssd", scan_backends),
        ("conv", conv_backends),
        ("prep", prep_backends),
        ("decode", decode_backends),
        ("tail", mixer_backends),
    ):
        chosen = module.resolve(None, "cuda", dtype)
        lines.append(
            f"{label}.names()={module.names()}  resolve->{chosen.name}  "
            f"prio{chosen.priority}"
        )
        if label == "decode":
            recurrence_live = chosen.name != REFERENCE
    live = True
    for boundary in boundaries:
        resolved = resolve_stages(boundary=boundary, dtype=dtype)
        lines.append(
            f"{boundary}: path={resolved.path}  stages={resolved.names}  {resolved.detail}"
        )
        live = live and resolved.path == KERNEL_PATH
    # Printed as its own line: a decode CuTe backend is registered at priority 10, so a
    # decode registry reporting the reference alone means the kernel did not load and every
    # recurrence row taken in this process is void rather than merely slow.
    registered = tuple(name for name in decode_backends.names() if name != REFERENCE)
    lines.append(
        f"decode kernel backends registered: {registered}"
        if registered
        else "decode kernel backends registered: NONE -- every recurrence row is VOID"
    )
    live = live and bool(registered)
    step = dispatch_verdict(DECODE, device_type="cuda", dtype=dtype)
    lines.append(f"dispatch_verdict({DECODE}) passed={step.passed}  {step.detail}")
    package = slinoss.__file__ or "<unknown>"
    lines.append(f"slinoss package: {package}")
    # The interpreter is a measured term, not a footnote: a torch minor moves host dispatch
    # by 5-17% and the toolkit moves device time by up to 1.2%, so both are stamped.
    # getattr twice: torch.version is attached at import and is not on the type stub, so a
    # direct reference is a type error while the attribute is always there at runtime.
    toolkit = getattr(getattr(torch, "version", None), "cuda", None) or "unknown"
    lines.append(
        f"interpreter: torch {torch.__version__}  cuda {toolkit}  dtype {dtype}"
    )
    return Liveness(
        lines=tuple(lines),
        live=live and step.passed,
        loaded=bool(registered),
        recurrence_live=recurrence_live and bool(registered),
        slinoss_package=package,
        torch_version=str(torch.__version__),
        detail=step.detail,
    )


def kernel_gate(live: Liveness) -> str:
    """Why no row may be taken in this process at all, or "" when rows may be taken.

    Only an unloaded kernel closes the gate. A registry that resolved to its reference for
    the requested dtype is a dtype with no instantiation, which is a reportable property of
    the operator: the row is measured, labelled a reference or mixed path, and refused a
    dominates by :func:`verdict`. A kernel that never registered is a tree that did not
    build, and every row it would produce is a fact about the build.

    Args:
        live: The liveness proof from the measuring process.

    Returns:
        A reason, or "".
    """
    if live.loaded:
        return ""
    return (
        "no decode kernel backend is registered in this process, so the extension did not "
        "build or did not import. Run setup.py build_ext --inplace in this tree; every row "
        "taken here would price torch under the kernel's name."
    )


# --------------------------------------------------------------------------
# The arms
# --------------------------------------------------------------------------


class Arm(NamedTuple):
    """One measurable callable and the parameter count of the layer behind it.

    Attributes:
        call: Takes no arguments, returns a tensor, advances its state in place.
            Consecutive calls are consecutive tokens, which is the steady state a decode
            loop runs in. Returns a tensor and not a named tuple because
            :func:`slinoss.graph.capture` records the outputs it is handed.
        param_count: Parameters in the layer.
        param_bytes: Bytes those parameters occupy, at the dtypes they were constructed
            with. Not ``param_count`` times the activation element size: Mamba3 pins several
            of its parameters to float32 at every model dtype, so the product would understate
            its weight traffic.
    """

    call: Callable[[], Tensor]
    param_count: int
    param_bytes: int


def build_slinoss(
    point: GridPoint, device: torch.device, *, boundary: str, seed: int = 0
) -> Arm:
    """Build one SLinOSS layer and the arm for one boundary.

    Args:
        point: The cell.
        device: Where to allocate and run.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        seed: Generator seed. Module initialization draws from the default generator, so
            this is seeded globally.

    Returns:
        The arm.

    Raises:
        ValueError: On an unknown boundary.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")
    from slinoss.ops.decode import decode_step

    config = point.config
    dtype = point.dtype
    torch.manual_seed(seed)
    mixer = SLinOSSMixer(config, device=device).to(dtype)
    state = MixerState.allocate(config, point.batch, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        point.batch, 1, config.d_model, device=device, dtype=dtype, generator=gen
    )
    params = sum(p.numel() for p in mixer.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in mixer.parameters())

    if boundary == WHOLE_STEP:

        def whole_step() -> Tensor:
            return mixer.step(x, state)

        return Arm(call=whole_step, param_count=params, param_bytes=param_bytes)

    operands = _slinoss_recurrence_operands(mixer, x, state)

    def recurrence() -> Tensor:
        return decode_step(*operands.args, **operands.carries).y

    return Arm(call=recurrence, param_count=params, param_bytes=param_bytes)


class RecurrenceOperands(NamedTuple):
    """The five reads and three carries :func:`slinoss.ops.decode.decode_step` takes.

    Attributes:
        args: ``(U, trans, K, B, C)``, in call order.
        carries: ``ssm``, ``b_prev`` and ``u_prev``, written in place by every call.
    """

    args: tuple[Tensor, ...]
    carries: dict[str, Tensor]


def _slinoss_recurrence_operands(
    mixer: SLinOSSMixer, x: Tensor, state: MixerState
) -> RecurrenceOperands:
    """Run everything outside the recurrence once, and keep what crosses the boundary.

    The projection, both convolutions and the parameter maps run here, once. What they
    produce is exactly the operator's five reads, so the timed region is the state update
    and nothing else. A fabricated ``trans`` or ``K`` would not carry the invariants the
    kernel relies on, and the recurrence's cost does not depend on their values, so the
    real maps cost one call and remove the question.

    Cloned carries: the returned operands are stepped in place by every measured call, and
    sharing them with ``state`` would leave a whole-step arm reading a state this arm had
    advanced.

    Args:
        mixer: The layer.
        x: ``(B,1,d_model)`` in the activation dtype.
        state: A state of the layer's shape, read but not advanced.

    Returns:
        The operands.
    """
    from slinoss._precision import cast_opt, cast_to
    from slinoss.mixer import _resolve
    from slinoss.ops.conv import backends as conv_dispatch
    from slinoss.ops.scanprep import backends as prep_dispatch

    config, layout = mixer.config, mixer.layout
    with torch.no_grad():
        proj = torch.nn.functional.linear(x, mixer.in_proj.weight, mixer.in_proj.bias)
        picks = _resolve(proj)
        conv = conv_dispatch.get(picks.conv).forward(
            layout.value(proj),
            cast_to(mixer.conv_weight, proj.dtype),
            cast_opt(mixer.conv_bias, proj.dtype),
            activation=True,
            initial_state=state.conv,
            d_head=config.d_head,
        )
        keys = (
            None
            if mixer.key_weight is None
            else conv_dispatch.get(picks.conv).forward(
                layout.keys(proj),
                cast_to(mixer.key_weight, proj.dtype),
                None,
                activation=False,
                initial_state=state.keys,
            )
        )
        prep = prep_dispatch.get(picks.prep).forward(
            layout.params(proj),
            mixer.transition_bias,
            heads=config.n_heads,
            w_max=config.w_max,
        )
        b_band = layout.b(proj) if keys is None else layout.key_b(keys.y)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys.y)
    return RecurrenceOperands(
        args=(
            conv.y.contiguous(),
            prep.trans.contiguous(),
            prep.K.contiguous(),
            b_band.contiguous(),
            c_band.contiguous(),
        ),
        carries={
            "ssm": state.ssm.clone(),
            "b_prev": state.b_prev.clone(),
            "u_prev": state.u_prev.clone(),
        },
    )


MAMBA_IMPORT_HINT: Final = (
    "Mamba3 needs torch 2.7.1, triton >= 3.5.0, nvidia-cutlass-dsl, apache-tvm-ffi "
    "<= 0.1.9 and, for MIMO, tilelang on PYTHONPATH; mamba_ssm's package initializer also "
    "drags in transformers and huggingface_hub. Both guards in "
    "mamba_ssm/modules/mamba3.py catch only ImportError, so a cutlass ABI failure is "
    "swallowed and degrades into an assert far from its cause."
)


def build_mamba(
    point: GridPoint, device: torch.device, *, boundary: str, seed: int = 0
) -> Arm:
    """Build one Mamba3 layer and the arm for one boundary.

    ``Mamba3.step`` is called directly with a 2-D ``u``. The ``inference_params`` route
    through ``Mamba3.forward`` is not used: it hands ``step`` a 3-D ``u`` and raises in
    ``rearrange``, so it cannot decode at all as shipped.

    The fusion setting follows the boundary, and that asymmetry is what
    :data:`MAMBA_FUSION_DISCLOSURE` reports. The whole step runs at Mamba3's shipped
    ``is_outproj_norm=False``, which fuses the SiLU gate and the ``mimo_o`` reduction into
    the kernel and is its fastest configuration, so no advantage is taken. The recurrence
    runs at ``is_outproj_norm=True``, the only setting in which the kernel is a bare state
    update and therefore the only one matching SLinOSS's recurrence boundary.

    The four state tensors are the ones ``allocate_inference_cache`` returns, in the order
    ``step`` consumes them. The original tuple is re-fed every call and the returns are
    discarded: ``step`` updates all four in place, which is what a capture needs, and
    re-feeding the returns instead would rebind buffers a graph recorded.

    Args:
        point: The cell.
        device: Where to allocate and run.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        seed: Generator seed.

    Returns:
        The arm.

    Raises:
        ValueError: On an unknown boundary.
        ImportError: If ``mamba_ssm`` or its kernel dependencies are missing. Deliberately
            unguarded so the traceback survives; see :data:`MAMBA_IMPORT_HINT`.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")
    # Reached only through PYTHONPATH on a host that carries the package, which a
    # static checker cannot see; the tree's other mamba import is suppressed the same way.
    from mamba_ssm.modules.mamba3 import Mamba3  # type: ignore[import-not-found]

    dtype = point.dtype
    torch.manual_seed(seed)
    layer = Mamba3(
        d_model=point.d_model,
        d_state=point.mamba_d_state,
        expand=EXPAND,
        headdim=D_HEAD,
        ngroups=point.n_groups,
        rope_fraction=ROPE_FRACTION,
        is_outproj_norm=boundary == RECURRENCE,
        is_mimo=point.mode == MIMO,
        mimo_rank=point.rank,
        layer_idx=0,
        device=device,
        dtype=dtype,
    )
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(
        point.batch, point.d_model, device=device, dtype=dtype, generator=gen
    )
    state = layer.allocate_inference_cache(point.batch, 1, device=device, dtype=dtype)
    params = sum(p.numel() for p in layer.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())

    if boundary == WHOLE_STEP:

        def whole_step() -> Tensor:
            return layer.step(u, *state)[0]

        return Arm(call=whole_step, param_count=params, param_bytes=param_bytes)

    return Arm(
        call=_mamba_recurrence(layer, u, state),
        param_count=params,
        param_bytes=param_bytes,
    )


def _mamba_recurrence(
    layer: Any, u: Tensor, state: Sequence[Tensor]
) -> Callable[[], Tensor]:
    """Isolate ``mamba3_step_fn`` by recording the arguments ``Mamba3.step`` hands it.

    Recorded rather than reconstructed. Reproducing the prologue means reimplementing
    ``_preprocess``, both gated ``B``/``C`` norms, the two bias rearranges and the triton
    rotary, and any drift between that copy and the module would surface as a latency
    difference attributed to the operator. One real ``step`` call produces exactly the
    tensors the module passes, including the ones the factory ``dtype`` never reached:
    ``dt_bias``, ``D`` and the ``mimo`` projections are built with a bare ``device=`` and so
    are float32 whatever dtype was requested, and ``xproj.dtype`` is part of the kernel's
    compile key.

    Everything the recording captured stays alive and is re-fed unchanged, so the timed
    region allocates nothing. The kernel writes ``ssm_state`` in place, because
    ``state_out`` is None, and writes ``out``; ``angle_dt_state``, ``k_state`` and
    ``v_state`` are advanced by python ``copy_`` calls after the kernel and so do not
    advance here. That is also true of the authors' own ``full_step_fn``, which re-feeds one
    state tuple every iteration, and it does not change the work the kernel does.

    Args:
        layer: A ``Mamba3`` built with ``is_outproj_norm=True``, so the gate and the
            ``mimo_o`` reduction sit outside the kernel.
        u: ``(B,d_model)`` in the model dtype.
        state: ``(angle_dt_state, ssm_state, k_state, v_state)``.

    Returns:
        A callable running one ``mamba3_step_fn`` and nothing else, returning ``out``.

    Raises:
        RuntimeError: If the step kernel is absent, or if ``Mamba3.step`` returned without
            reaching it, either of which would leave the recorded boundary empty.
    """
    import mamba_ssm.modules.mamba3 as module  # type: ignore[import-not-found]

    real = module.mamba3_step_fn
    if real is None:
        raise RuntimeError(
            "mamba_ssm.modules.mamba3.mamba3_step_fn is None: the CuTe step kernel import "
            f"was swallowed by the module's ImportError guard. {MAMBA_IMPORT_HINT}"
        )
    seen: dict[str, Any] = {}

    def recorder(*args: Any, **kwargs: Any) -> Any:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return real(*args, **kwargs)

    module.mamba3_step_fn = recorder
    try:
        with torch.no_grad():
            layer.step(u, *state)
    finally:
        module.mamba3_step_fn = real
    if "args" not in seen:
        raise RuntimeError(
            "Mamba3.step returned without calling mamba3_step_fn, so the recurrence "
            "boundary recorded nothing to measure"
        )
    args, kwargs = seen["args"], seen["kwargs"]
    out = kwargs["out"]

    def recurrence() -> Tensor:
        real(*args, **kwargs)
        return out

    return recurrence


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


class Cell(NamedTuple):
    """One measured comparison at one boundary and one execution.

    Attributes:
        point: The cell's shape.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        resolved: Which backend every stage resolved to, and what that makes the row.
        regime: What limits the row before its operator does.
        slinoss_duration_us: Median SLinOSS latency.
        slinoss_resolution_pct: Half-width of the interval on that median.
        slinoss_spread_pct: Full range over that median.
        mamba_duration_us: Median Mamba3 latency.
        mamba_resolution_pct: Half-width of the interval on that median.
        mamba_spread_pct: Full range over that median.
        ratio: SLinOSS over Mamba3. Below one means SLinOSS is faster.
        paired_delta_us: Median per-iteration difference, position cost removed.
        paired_low_us: Lower bound on that median.
        paired_high_us: Upper bound on that median.
        paired_resolves: True only if that interval excludes zero. The only field that
            licenses a claim about this cell.
        match: What was held equal and what could not be.
        floor: Each side against its own compulsory-traffic floor, or the reason there is
            none for this row.
        witness: How the row earned its card. See :class:`Witness`.
        iters: Timed iterations behind the two medians. Zero means the record predates the
            field, and the column prints a dash rather than a count: a table whose rows were
            taken at different iteration counts has to say per row which count it carries,
            because the half-width the row is judged on is a function of that count.
        slinoss_samples_duration_us: Every timed SLinOSS sample, in measurement order.
            Retained so a reader recomputes the median, the quantiles and the half-width
            rather than taking three summary floats on trust, and sees drift across the
            loop that no summary shows. Empty means the record predates the field.
        mamba_samples_duration_us: The same for Mamba3, from the same paired loop, so the
            two orders line up pairwise and a difference is read per iteration.
    """

    point: GridPoint
    boundary: str
    execution: str
    resolved: Resolved
    regime: str
    slinoss_duration_us: Microseconds
    slinoss_resolution_pct: Percent
    slinoss_spread_pct: Percent
    mamba_duration_us: Microseconds
    mamba_resolution_pct: Percent
    mamba_spread_pct: Percent
    ratio: float
    paired_delta_us: Microseconds
    paired_low_us: Microseconds
    paired_high_us: Microseconds
    paired_resolves: bool
    match: MatchReport
    floor: FloorPair
    witness: Witness = NO_WITNESS
    iters: int = 0
    slinoss_samples_duration_us: tuple[Microseconds, ...] = ()
    mamba_samples_duration_us: tuple[Microseconds, ...] = ()


def measure_cell(
    point: GridPoint,
    *,
    boundary: str,
    execution: str,
    device: torch.device,
    iters: int,
    warmup: int,
    clocks: ClockPolicy | None = None,
    fit: DramTimeFloor | None = None,
    witness: Witness = NO_WITNESS,
) -> Cell:
    """Measure one cell, both architectures in one paired loop.

    One loop and not two: a paired loop swaps which arm runs first every iteration, so a
    clock excursion or a foreign job arriving hits both arms of a pair and cancels out of
    the difference. Two separate loops would compare two medians taken at different times.

    Args:
        point: The cell.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        device: Where to run.
        iters: Timed iterations. Even, so the order swap balances.
        warmup: Untimed iterations first.
        clocks: Clock policy to stamp. Probed by the timer if omitted.
        fit: DRAM time law fitted in this process. Omitted, the row carries no floor; a fit
            from another process or another host does not price this card.
        witness: How this window earned the card, recorded on the row. One window; the
            replicate rule lives in :func:`measure_replicated`.

    Returns:
        The measured cell.

    Raises:
        ValueError: On an unknown boundary or execution.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; have {list(BOUNDARIES)}")
    if execution not in EXECUTIONS:
        raise ValueError(f"unknown execution {execution!r}; have {list(EXECUTIONS)}")

    slinoss = build_slinoss(point, device, boundary=boundary)
    mamba = build_mamba(point, device, boundary=boundary)
    match = match_shapes(
        point.config,
        dtype=point.dtype,
        batch=point.batch,
        mamba_d_state=point.mamba_d_state,
        rank=point.rank,
        slinoss_param_count=slinoss.param_count,
        mamba_param_count=mamba.param_count,
    )
    sl_call, m3_call = slinoss.call, mamba.call
    if execution == GRAPH:
        sl_call, m3_call = _capture(sl_call), _capture(m3_call)

    out = measure_paired(
        MAMBA3,
        m3_call,
        SLINOSS,
        sl_call,
        label=f"{boundary}/{execution}/{point.shape_class}/B={point.batch}",
        iters=iters,
        warmup=warmup,
        device=device,
        clocks=clocks,
    )
    sl = out.timed.region(SLINOSS).spread
    m3 = out.timed.region(MAMBA3).spread
    where = regime(batch=point.batch, execution=execution, boundary=boundary)
    return Cell(
        point=point,
        boundary=boundary,
        execution=execution,
        resolved=resolve_stages(boundary=boundary, dtype=point.dtype),
        regime=where,
        slinoss_duration_us=sl.median_duration_us,
        slinoss_resolution_pct=sl.resolution_pct,
        slinoss_spread_pct=sl.spread_pct,
        slinoss_samples_duration_us=tuple(sl.samples_duration_us),
        mamba_duration_us=m3.median_duration_us,
        mamba_resolution_pct=m3.resolution_pct,
        mamba_spread_pct=m3.spread_pct,
        mamba_samples_duration_us=tuple(m3.samples_duration_us),
        ratio=sl.median_duration_us / m3.median_duration_us,
        paired_delta_us=out.comparison.delta_median_duration_us,
        paired_low_us=out.comparison.delta_low_duration_us,
        paired_high_us=out.comparison.delta_high_duration_us,
        paired_resolves=out.comparison.resolves,
        match=match,
        floor=floor_pair(
            regime_name=where,
            slinoss_bytes=moved_bytes(
                boundary=boundary,
                param_bytes=slinoss.param_bytes,
                state=match.slinoss_state_bytes,
                batch=point.batch,
                d_model=point.d_model,
                dtype=point.dtype,
            ),
            mamba_bytes=moved_bytes(
                boundary=boundary,
                param_bytes=mamba.param_bytes,
                state=match.mamba_state_bytes,
                batch=point.batch,
                d_model=point.d_model,
                dtype=point.dtype,
            ),
            slinoss_duration_us=sl.median_duration_us,
            mamba_duration_us=m3.median_duration_us,
            fit=fit,
        ),
        witness=witness,
        iters=iters,
    )


def _capture(arm: Callable[[], Tensor]) -> Callable[[], Tensor]:
    """Record one arm and return a replay of it.

    :func:`slinoss.graph.capture` copies the inputs it is given into the buffers the graph
    recorded; these arms close over their own buffers and take no arguments, so the graph
    is recorded over an empty input list and a replay reads exactly what the capture
    recorded.

    Args:
        arm: A callable taking no arguments and returning a tensor, whose state is written
            in place.

    Returns:
        A callable that replays the recorded graph.
    """
    from slinoss.graph import capture

    step = capture(arm)
    return lambda: step()


def reconcile(first: Cell, second: Cell) -> str:
    """Why two windows on one cell disagree, or "" when they agree.

    Both arms are checked, not the ratio: two medians can each move and leave the ratio
    where it was, and a row is a claim about two latencies before it is a claim about their
    quotient.

    Args:
        first: The window whose numbers would be reported.
        second: The corroborating window.

    Returns:
        A sentence naming the arm that disagreed and by how much, or "".
    """
    for arm, a_us, a_pct, b_us, b_pct in (
        (
            SLINOSS,
            float(first.slinoss_duration_us),
            float(first.slinoss_resolution_pct),
            float(second.slinoss_duration_us),
            float(second.slinoss_resolution_pct),
        ),
        (
            MAMBA3,
            float(first.mamba_duration_us),
            float(first.mamba_resolution_pct),
            float(second.mamba_duration_us),
            float(second.mamba_resolution_pct),
        ),
    ):
        if not agrees_within_half_widths(a_us, a_pct, b_us, b_pct):
            reach = max((a_us * a_pct + b_us * b_pct) / 100.0, TIMER_QUANTUM_US)
            return (
                f"{arm} disagreed across the two windows: {a_us:,.3f} us against "
                f"{b_us:,.3f} us, a gap of {abs(a_us - b_us):,.3f} us against a reach of "
                f"{reach:,.3f} us, being the combined half-width or the "
                f"{TIMER_QUANTUM_US:,.3f} us timer step, whichever is larger"
            )
    return ""


def measure_replicated(
    point: GridPoint,
    *,
    boundary: str,
    execution: str,
    device: torch.device,
    iters: int,
    warmup: int,
    clocks: ClockPolicy | None = None,
    fit: DramTimeFloor | None = None,
    witness_stamp: str,
    replicates: int,
    foreign_mib: float,
    ordinal: int,
    gap_s: float = REPLICATE_GAP_S,
    probe: Callable[[int], Contention] = contention,
    rest: Callable[[float], None] = time.sleep,
    measure: Callable[..., Cell] = measure_cell,
) -> tuple[Cell | None, str]:
    """Measure one cell in as many disjoint windows as its card owes, or discard it.

    Under an exclusive witness this is one window and the answer is that window. Under a
    residency witness it is :data:`RESIDENCY_REPLICATES` windows, each with its own modules,
    its own capture and its own warmup, separated by a probe that can still refuse the card.
    The first window is the reported number and the second is corroboration; nothing is
    averaged, because averaging two windows that disagree hides exactly the thing the second
    window was taken to detect.

    Args:
        point: The cell.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
        device: Where to run.
        iters: Timed iterations per window.
        warmup: Untimed iterations per window.
        clocks: Clock policy to stamp.
        fit: DRAM time law fitted in this process, or None.
        witness_stamp: :data:`EXCLUSIVE_WITNESS` or :data:`RESIDENCY_WITNESS`.
        replicates: Windows owed. See :func:`admit`.
        foreign_mib: Foreign memory at admission, recorded on the row.
        ordinal: Device ordinal to re-probe between windows.
        gap_s: Seconds between windows. Also what makes the between-window probe
            attributable: it is 20 times the 0.25 s the utilization reading takes to settle
            after this process's own work. See :data:`CLOSE_SETTLE_S`.
        probe: Contention probe, injected for testing.
        rest: Sleep, injected for testing.
        measure: One window's measurement, injected for testing.

    Returns:
        The cell and "" when it stands, or None and the reason when it is discarded. A
        discarded cell is not banked, so the next invocation measures it again.
    """
    windows: list[Cell] = []
    for index in range(max(1, replicates)):
        if index:
            rest(gap_s)
            between = probe(ordinal)
            stamp, _, why = admit(between)
            if stamp != witness_stamp:
                return (
                    None,
                    f"discarded: the card changed between windows {index} and {index + 1} "
                    f"({why}), so the two windows are not one card measured twice",
                )
        windows.append(
            measure(
                point,
                boundary=boundary,
                execution=execution,
                device=device,
                iters=iters,
                warmup=warmup,
                clocks=clocks,
                fit=fit,
            )
        )
    for other in windows[1:]:
        disagreement = reconcile(windows[0], other)
        if disagreement:
            return (
                None,
                f"discarded: {disagreement}. A cell whose replicates disagree is void; it "
                f"is not averaged and it does not get a caveat",
            )
    agreed = (
        f"; the {len(windows)} windows agreed inside the sum of their half-widths"
        if len(windows) > 1
        else ""
    )
    return (
        windows[0]._replace(
            witness=Witness(
                stamp=witness_stamp,
                foreign_mib=foreign_mib,
                replicates=len(windows),
                agrees=True,
                detail=(
                    f"{witness_stamp} card, {foreign_mib:,.0f} MiB foreign memory, "
                    f"{len(windows)} window{'s' if len(windows) > 1 else ''}{agreed}"
                ),
            )
        ),
        "",
    )


# --------------------------------------------------------------------------
# The bank
# --------------------------------------------------------------------------


class Task(NamedTuple):
    """One cell to measure, at one boundary and one execution.

    Attributes:
        point: The shape.
        boundary: :data:`RECURRENCE` or :data:`WHOLE_STEP`.
        execution: :data:`EAGER` or :data:`GRAPH`.
    """

    point: GridPoint
    boundary: str
    execution: str

    @property
    def key(self) -> str:
        """The cell's identity in a bank. Stable, and a legal file name."""
        point = self.point
        return (
            f"{self.boundary}-{self.execution}-{point.dtype_name}"
            f"-dm{point.d_model}-sl{point.slinoss_d_state}-m3{point.mamba_d_state}"
            f"-{point.sharing}-{point.mode}-B{point.batch}"
        )


def boundary_rank(boundary: str) -> int:
    """Recurrence before whole step: only the first is measuring the decode kernel."""
    return 0 if boundary == RECURRENCE else 1


def decisiveness(task: Task) -> tuple[int, ...]:
    """Sort key putting the rows that can settle the question first.

    The fleet offers minutes. So the order is: the recurrence boundary, under graph capture,
    at the shapes where the two ``d_state`` values are closest, at bf16 before float32; and
    inside one class, the batches above the L2 crossover before the ones under it, largest
    first. Every field before the last two is a property of the class, so a class's cells
    stay contiguous and a verdict completes rather than a hundred classes each losing a
    batch.

    Args:
        task: The cell.

    Returns:
        A tuple to sort ascending.
    """
    point = task.point
    return (
        boundary_rank(task.boundary),
        0 if task.execution == GRAPH else 1,
        abs(point.slinoss_d_state - point.mamba_d_state),
        # The two widths themselves, not only their distance: two different pairs can be
        # equidistant, and without these their cells interleave and neither class finishes.
        point.slinoss_d_state,
        point.mamba_d_state,
        list(DTYPES).index(point.dtype_name),
        point.d_model,
        SHARINGS.index(point.sharing),
        MODES.index(point.mode),
        # From here on the key varies inside one class, so these two fields order the
        # class's own cells and never split it.
        1
        if judgeable(
            batch=point.batch, execution=task.execution, boundary=task.boundary
        )
        else 0,
        0
        if regime(batch=point.batch, execution=task.execution, boundary=task.boundary)
        == DRAM_REGIME
        else 1,
        -point.batch,
    )


def tasks(
    grid: Grid,
    *,
    boundaries: Sequence[str],
    executions: Sequence[str],
    order: str = DECISIVE,
) -> tuple[Task, ...]:
    """Every cell to measure, in the requested order.

    Args:
        grid: The enumerated grid.
        boundaries: Boundaries to measure.
        executions: Executions to measure.
        order: :data:`DECISIVE` or :data:`NESTED`.

    Returns:
        The cells.

    Raises:
        ValueError: On an unknown order.
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; have {list(ORDERS)}")
    flat = tuple(
        Task(point=point, boundary=boundary, execution=execution)
        for boundary in boundaries
        for execution in executions
        for point in grid.points
    )
    return flat if order == NESTED else tuple(sorted(flat, key=decisiveness))


def defers(task: Task, *, graphed: int, budget: int = GRAPH_CELLS_PER_PROCESS) -> bool:
    """Whether this cell must wait for the next process.

    Args:
        task: The cell.
        graphed: Graph cells this process has already built a capture for, counted whether
            or not the cell survived its replicates.
        budget: Graph cells one process may measure. See
            :data:`GRAPH_CELLS_PER_PROCESS`.

    Returns:
        True when the cell is a graph cell and the budget is spent. Eager cells never defer:
        they build no capture, so the hazard does not reach them.
    """
    return task.execution == GRAPH and graphed >= budget


def poisons(error: BaseException) -> bool:
    """Whether a failure leaves the CUDA context untrustworthy.

    Args:
        error: The exception a cell died on.

    Returns:
        True when the message names a device-side assert or another CUDA fault. A True here
        voids the process, not the cell: a later launch can still return a plausible number.
    """
    text = str(error).lower()
    return any(mark in text for mark in POISON_MARKS)


def source_digest(package: str) -> str:
    """Digest every ``.py`` file of one measured package, either side's.

    The CuTe kernels are Python, so a kernel edit changes this digest and empties the bank.
    That is the point: an addressing change took the decode kernel from 357.64 to 669.75
    GB/s, and a bank keyed only on host and torch would have merged both into one table. The
    competitor is digested by this same function for the same reason in the other direction:
    a Mamba3 source edit must void a banked Mamba3 cell exactly as a kernel edit voids ours,
    and a digest of our tree alone cannot see one.

    Args:
        package: The ``__file__`` of the package's top-level module, from the measuring
            process.

    Returns:
        Sixteen hex characters, or ``"unreadable"`` when the directory cannot be walked.
    """
    root = os.path.dirname(package)
    if not os.path.isdir(root):
        return "unreadable"
    digest = hashlib.sha256()
    for base, directories, names in os.walk(root):
        # Sorted in place so os.walk visits in one order on every filesystem, and caches
        # excluded so a bytecode write does not change the digest of unchanged source.
        directories[:] = sorted(one for one in directories if one != "__pycache__")
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            digest.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()[:16]


MAMBA_PACKAGE: Final = "mamba_ssm"
"""Import name of the competitor package.

Digested the way :func:`source_digest` digests ours, so a competitor source edit voids a
banked competitor cell exactly as a kernel edit voids one of ours.
"""

SOURCES_ORIGIN: Final = "~/projects/slinoss/.sources/code/mamba/mamba_ssm"
"""Where the disposable competitor copy came from.

Immutable and read only. Recorded rather than read at measurement time: the tree is on
another filesystem than the card, so the copy is what answers and the copy is what is
digested. :func:`competitor_provenance` states the origin and the manifest that lets the two
be compared file by file; it does not assert they are equal.
"""

TVM_FFI_DISTRIBUTION: Final = "apache-tvm-ffi"
TVM_FFI_MODULE: Final = "tvm_ffi"

MATCHED_PROVENANCE: Final[tuple[str, ...]] = (
    "schema",
    "torch",
    "device",
    "slinoss_package",
    "slinoss_sources",
    "mamba_package",
    "mamba_sources",
)
"""The provenance fields a banked cell must match before it is read.

Both trees, both paths. The fields outside this tuple are recorded and not compared: the
dependency set and the copy's file manifest describe the run rather than key it, and the
manifest in particular is a longer statement of what ``mamba_sources`` already keys on.
"""


def file_manifest(root: str) -> tuple[dict[str, str], ...]:
    """Per-file sha256 of every ``.py`` under one package directory.

    The digest in :func:`source_digest` says whether two trees differ; this says which file
    differs, which is what makes a disposable copy auditable against the immutable tree it
    was copied out of.

    Args:
        root: Package directory.

    Returns:
        One mapping per file, path relative to ``root`` with forward slashes, sorted by path.
        Empty when the directory cannot be walked.
    """
    if not os.path.isdir(root):
        return ()
    found: list[dict[str, str]] = []
    for base, directories, names in os.walk(root):
        directories[:] = sorted(one for one in directories if one != "__pycache__")
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, "rb") as handle:
                    body = handle.read()
            except OSError:
                continue
            found.append(
                {
                    "path": os.path.relpath(path, root).replace(os.sep, "/"),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
    return tuple(sorted(found, key=lambda one: one["path"]))


def git_commit(root: str) -> str:
    """The commit a copy carries, read from ``.git`` without running ``git``.

    A subprocess in the measuring process is a side effect on the thing being measured, and
    the host need not carry a ``git`` binary at all. Both this copy and the ``.sources`` tree
    it came from are bare file trees, so the expected answer here is ``"absent"``; the code
    exists because a copy that does carry ``.git`` must name its commit rather than be
    described by a manifest alone.

    Args:
        root: Package directory. ``.git`` is searched for at this directory and above, since
            a package directory sits inside its repository.

    Returns:
        Forty hex characters, a branch-relative description when the ref cannot be resolved,
        or ``"absent"`` when the directory does not exist or no ``.git`` is found. The
        existence check is what keeps an unresolved package from walking up out of the working
        directory and reporting this repository's commit as the competitor's.
    """
    if not os.path.isdir(root):
        return "absent"
    here = os.path.abspath(root)
    while True:
        candidate = os.path.join(here, ".git")
        if os.path.exists(candidate):
            break
        parent = os.path.dirname(here)
        if parent == here:
            return "absent"
        here = parent
    try:
        with open(os.path.join(candidate, "HEAD")) as handle:
            head = handle.read().strip()
    except OSError as broken:
        return f"unreadable ({broken})"
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    try:
        with open(os.path.join(candidate, ref)) as handle:
            return handle.read().strip()
    except OSError:
        pass
    try:
        with open(os.path.join(candidate, "packed-refs")) as handle:
            for line in handle:
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    return fields[0]
    except OSError:
        pass
    return f"unresolved ref {ref}"


def resolved_package(name: str) -> tuple[str, str]:
    """Where an import of one package would resolve, and by which route.

    Resolved through the import machinery and never from a path passed on the command line,
    so the field names what actually answers. An already-imported module is read from
    :data:`sys.modules`; otherwise the finder is asked without executing the package, because
    ``mamba_ssm``'s initializer drags in transformers and huggingface_hub and this runs in
    processes that never build a Mamba3 layer. Both routes consult the same ``sys.path`` in
    the same order, and the route is recorded so a disagreement between them is visible
    rather than assumed away.

    Args:
        name: Top-level import name.

    Returns:
        The package directory and the route, or ``("unimportable", reason)``.
    """
    module = sys.modules.get(name)
    origin = getattr(module, "__file__", None) if module is not None else None
    if origin:
        return os.path.dirname(os.path.abspath(origin)), "sys.modules"
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as broken:
        return "unimportable", f"find_spec raised ({broken})"
    if spec is None or not spec.origin:
        return "unimportable", "no spec on sys.path"
    return os.path.dirname(os.path.abspath(spec.origin)), "find_spec"


def installed_version(distribution: str, module: str) -> str:
    """One dependency's version, without importing it.

    Metadata first: a ``--target`` install carries its ``dist-info`` and is found this way.
    A bare directory copy carries none, so the finder answers presence alone. Neither route
    executes the package, which is what keeps this callable from a measuring process.

    Args:
        distribution: Distribution name, as ``importlib.metadata`` knows it.
        module: Import name, which need not match the distribution name.

    Returns:
        A version string, ``"present, version unknown"``, or ``"absent"``.
    """
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return "absent"
    return "present, version unknown" if spec is not None else "absent"


def dependency_set() -> dict[str, str]:
    """The interpreter and the packages a Mamba3 figure does not survive a change of.

    Mamba3's step kernel is CuTe DSL over triton and apache-tvm-ffi, and it refuses to import
    at all outside a narrow pin set, so a figure taken under one set is not comparable with
    one taken under another. Recorded in the artifact rather than left to a report sentence,
    and deliberately outside :data:`MATCHED_PROVENANCE`: the MIMO and SISO paths differ in
    exactly this mapping, so keying on it would refuse a MIMO cell on a SISO render.

    Returns:
        A JSON-ready mapping. ``prefix`` is the environment root, which is the venv path when
        the interpreter is a venv's.
    """
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "torch": str(torch.__version__),
        "triton": installed_version("triton", "triton"),
        TVM_FFI_DISTRIBUTION: installed_version(TVM_FFI_DISTRIBUTION, TVM_FFI_MODULE),
    }


def competitor_provenance() -> dict[str, Any]:
    """What the competitor side of a banked figure is keyed and audited on.

    Returns:
        ``package`` and ``digest`` are the two matched fields, formed exactly as the slinoss
        pair is. ``origin`` is the audit block: the route the package resolved by, the copy's
        commit if it carries one, the count and per-file sha256 of what was copied, and the
        immutable tree the copy came from. ``attested`` says the digest was taken in the
        process that took the numbers, which is what separates a measured record from one a
        later migration stamped.
    """
    directory, route = resolved_package(MAMBA_PACKAGE)
    manifest = file_manifest(directory)
    return {
        "package": directory,
        "digest": source_digest(os.path.join(directory, "__init__.py")),
        "origin": {
            "route": route,
            "git": git_commit(directory),
            "copied_from": SOURCES_ORIGIN,
            "file_count": len(manifest),
            "files": [dict(one) for one in manifest],
            "attested": "process",
        },
    }


def card_identity(ordinal: int, *, name: str) -> str:
    """Name one physical card, not one model of card.

    Two cards of the same model on one host report the same marketing name, so a bank keyed
    on the name alone would merge a figure taken on one into a table built from the other.
    The UUID is what separates them.

    Args:
        ordinal: Device ordinal, as visible to this process.
        name: Marketing name, from :func:`slinoss.perf.device.device_info`.

    Returns:
        ``"name uuid"``, or ``"name ordinal N"`` where torch exposes no UUID.
    """
    try:
        found = getattr(torch.cuda.get_device_properties(ordinal), "uuid", None)
    except (AssertionError, RuntimeError):
        found = None
    return f"{name} {found}" if found else f"{name} ordinal {ordinal}"


def provenance(live: Liveness, *, device_name: str) -> dict[str, Any]:
    """What a banked cell must match before it is read back.

    A bank is the one place a figure could cross a host, a torch version or a tree without
    anyone noticing, since it is read from disk rather than measured. So the fields a figure
    may never cross are stored with every cell and compared on the way back in, both trees
    among them: see :func:`source_digest`. Digesting ours alone would have let a banked
    competitor cell survive a competitor source edit, which is the difference between an
    artifact that is reproducible and one that asserts it is.

    Args:
        live: The liveness proof from the measuring process.
        device_name: The card, from :func:`card_identity`.

    Returns:
        A JSON-ready mapping. :data:`MATCHED_PROVENANCE` names the subset compared on read;
        ``deps`` and ``competitor_origin`` are recorded and not compared.
    """
    competitor = competitor_provenance()
    return {
        "schema": BANK_SCHEMA,
        "torch": live.torch_version,
        "device": device_name,
        "slinoss_package": live.slinoss_package,
        "slinoss_sources": source_digest(live.slinoss_package),
        "mamba_package": competitor["package"],
        "mamba_sources": competitor["digest"],
        "deps": dependency_set(),
        "competitor_origin": competitor["origin"],
    }


def state_record(state: StateBytes) -> dict[str, Any]:
    """One side's state itemization, serialized."""
    return {
        "recurrent_bytes": int(state.recurrent_bytes),
        "conv_bytes": int(state.conv_bytes),
        "carry_bytes": int(state.carry_bytes),
        "total_bytes": int(state.total_bytes),
        "conv_buffer_count": state.conv_buffer_count,
    }


def cell_record(cell: Cell, *, stored: Mapping[str, Any]) -> dict[str, Any]:
    """One measured cell, serialized whole and reconstructible.

    Args:
        cell: The cell.
        stored: The provenance to store with it. See :func:`provenance`.

    Returns:
        A JSON-ready mapping.
    """
    point = cell.point
    return {
        "provenance": dict(stored),
        "key": Task(point=point, boundary=cell.boundary, execution=cell.execution).key,
        "point": {
            "d_model": point.d_model,
            "dtype_name": point.dtype_name,
            "batch": point.batch,
            "slinoss_d_state": point.slinoss_d_state,
            "mamba_d_state": point.mamba_d_state,
            "sharing": point.sharing,
            "mode": point.mode,
        },
        "boundary": cell.boundary,
        "execution": cell.execution,
        "resolved": {
            "names": list(cell.resolved.names),
            "path": cell.resolved.path,
            "detail": cell.resolved.detail,
        },
        "regime": cell.regime,
        "iters": cell.iters,
        "slinoss_duration_us": float(cell.slinoss_duration_us),
        "slinoss_resolution_pct": float(cell.slinoss_resolution_pct),
        "slinoss_spread_pct": float(cell.slinoss_spread_pct),
        "slinoss_samples_duration_us": [
            float(one) for one in cell.slinoss_samples_duration_us
        ],
        "mamba_duration_us": float(cell.mamba_duration_us),
        "mamba_resolution_pct": float(cell.mamba_resolution_pct),
        "mamba_spread_pct": float(cell.mamba_spread_pct),
        "mamba_samples_duration_us": [
            float(one) for one in cell.mamba_samples_duration_us
        ],
        "ratio": float(cell.ratio),
        "paired_delta_us": float(cell.paired_delta_us),
        "paired_low_us": float(cell.paired_low_us),
        "paired_high_us": float(cell.paired_high_us),
        "paired_resolves": cell.paired_resolves,
        "slinoss_param_count": cell.match.slinoss_param_count,
        "mamba_param_count": cell.match.mamba_param_count,
        "slinoss_state_bytes": state_record(cell.match.slinoss_state_bytes),
        "mamba_state_bytes": state_record(cell.match.mamba_state_bytes),
        "floor": {
            "available": cell.floor.available,
            "slinoss_moved_bytes": cell.floor.slinoss_moved_bytes,
            "slinoss_floor_us": float(cell.floor.slinoss_floor_us),
            "slinoss_x_floor": float(cell.floor.slinoss_x_floor),
            "mamba_moved_bytes": cell.floor.mamba_moved_bytes,
            "mamba_floor_us": float(cell.floor.mamba_floor_us),
            "mamba_x_floor": float(cell.floor.mamba_x_floor),
            "detail": cell.floor.detail,
        },
        "witness": {
            "stamp": cell.witness.stamp,
            "foreign_mib": cell.witness.foreign_mib,
            "replicates": cell.witness.replicates,
            "agrees": cell.witness.agrees,
            "detail": cell.witness.detail,
        },
    }


def cell_from_record(record: Mapping[str, Any]) -> Cell:
    """Rebuild a cell from its banked record.

    The match report is not stored as a blob and reread; it is recomputed by
    :func:`match_shapes` from the stored shape and the stored parameter counts, which is
    pure. A stored blob could drift from what the current code would say about the same
    shape, and then a banked row and a fresh row in one table would be describing their
    match by two different rules.

    Args:
        record: One cell's record.

    Returns:
        The cell.

    Raises:
        KeyError: On a record missing a field, which is a corrupt bank rather than a stale
            one and is not swallowed.
    """
    point = GridPoint(**record["point"])
    floor = record["floor"]
    stored = record["witness"]
    return Cell(
        point=point,
        boundary=record["boundary"],
        execution=record["execution"],
        resolved=Resolved(
            names=tuple(record["resolved"]["names"]),
            path=record["resolved"]["path"],
            detail=record["resolved"]["detail"],
        ),
        regime=record["regime"],
        # Absent in a record written before the field existed. Zero, not a guess at the
        # driver's default: a count read off today's default would be attributed to a run
        # that never reported one.
        iters=int(record.get("iters", 0)),
        slinoss_duration_us=Microseconds(record["slinoss_duration_us"]),
        slinoss_resolution_pct=Percent(record["slinoss_resolution_pct"]),
        slinoss_spread_pct=Percent(record["slinoss_spread_pct"]),
        # Empty in a record written before the field existed, on the same rule as `iters`:
        # the summary floats it does carry are not resampled into a sample list, because a
        # fabricated list would read as the measured one.
        slinoss_samples_duration_us=tuple(
            Microseconds(one) for one in record.get("slinoss_samples_duration_us", ())
        ),
        mamba_duration_us=Microseconds(record["mamba_duration_us"]),
        mamba_resolution_pct=Percent(record["mamba_resolution_pct"]),
        mamba_spread_pct=Percent(record["mamba_spread_pct"]),
        mamba_samples_duration_us=tuple(
            Microseconds(one) for one in record.get("mamba_samples_duration_us", ())
        ),
        ratio=record["ratio"],
        paired_delta_us=Microseconds(record["paired_delta_us"]),
        paired_low_us=Microseconds(record["paired_low_us"]),
        paired_high_us=Microseconds(record["paired_high_us"]),
        paired_resolves=record["paired_resolves"],
        match=match_shapes(
            point.config,
            dtype=point.dtype,
            batch=point.batch,
            mamba_d_state=point.mamba_d_state,
            rank=point.rank,
            slinoss_param_count=record["slinoss_param_count"],
            mamba_param_count=record["mamba_param_count"],
        ),
        floor=FloorPair(
            available=floor["available"],
            slinoss_moved_bytes=floor["slinoss_moved_bytes"],
            slinoss_floor_us=Microseconds(floor["slinoss_floor_us"]),
            slinoss_x_floor=floor["slinoss_x_floor"],
            mamba_moved_bytes=floor["mamba_moved_bytes"],
            mamba_floor_us=Microseconds(floor["mamba_floor_us"]),
            mamba_x_floor=floor["mamba_x_floor"],
            detail=floor["detail"],
        ),
        witness=Witness(
            stamp=stored["stamp"],
            foreign_mib=stored["foreign_mib"],
            replicates=stored["replicates"],
            agrees=stored["agrees"],
            detail=stored["detail"],
        ),
    )


def write_record(
    directory: str, key: str, record: Mapping[str, Any], *, suffix: str = ".json"
) -> str:
    """Write one bank record, atomically.

    Written under a temporary name in the same directory and renamed onto the final name,
    because :func:`os.replace` is atomic within a filesystem and a half-written JSON file
    read by the next invocation would be a corrupt bank. An interrupted run therefore loses
    the cell in flight and nothing else.

    Args:
        directory: The bank. Created if absent.
        key: The cell key. See :attr:`Task.key`.
        record: JSON-ready payload.
        suffix: ``".json"`` for a cell, :data:`VOID_SUFFIX` for a void marker.

    Returns:
        The path written.
    """
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, f"{key}{suffix}")
    handle, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{key}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w") as opened:
            json.dump(record, opened, indent=2)
            opened.flush()
            os.fsync(opened.fileno())
        os.replace(temporary, final)
    except BaseException:
        # A failed write leaves no partial file behind for the next invocation to read as a
        # banked cell.
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return final


def write_cell(directory: str, cell: Cell, *, stored: Mapping[str, Any]) -> str:
    """Bank one measured cell.

    Args:
        directory: The bank. Created if absent.
        cell: The cell to write.
        stored: Provenance. See :func:`provenance`.

    Returns:
        The path written.
    """
    key = Task(point=cell.point, boundary=cell.boundary, execution=cell.execution).key
    return write_record(directory, key, cell_record(cell, stored=stored))


def write_void(
    directory: str, key: str, reason: str, *, stored: Mapping[str, Any]
) -> str:
    """Record that a cell died, and why.

    A void marker is not a skip: the cell stays pending and the next invocation measures it
    again. It exists so a cell that died is a documented result rather than a hole in the
    grid.

    Args:
        directory: The bank. Created if absent.
        key: The cell key. See :attr:`Task.key`.
        reason: What it died on, verbatim.
        stored: Provenance. See :func:`provenance`.

    Returns:
        The path written.
    """
    return write_record(
        directory,
        key,
        {"key": key, "void": reason, "provenance": dict(stored)},
        suffix=VOID_SUFFIX,
    )


def read_voids(directory: str) -> tuple[str, ...]:
    """Every void marker in the bank, as one sentence each.

    Args:
        directory: The bank. Absent is not an error.

    Returns:
        ``"key: reason"`` per marker, sorted by key. Provenance is not filtered on: a cell
        that died under another tree is still worth printing next to one that died here.
    """
    if not os.path.isdir(directory):
        return ()
    lines: list[str] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(VOID_SUFFIX):
            continue
        try:
            with open(os.path.join(directory, name)) as handle:
                record = json.load(handle)
        except (OSError, ValueError) as broken:
            lines.append(f"{name}: unreadable ({broken})")
            continue
        lines.append(f"{record.get('key', name)}: {record.get('void', '(no reason)')}")
    return tuple(lines)


def read_bank(
    directory: str, *, stored: Mapping[str, Any]
) -> tuple[dict[str, Cell], tuple[str, ...]]:
    """Read the cells a previous invocation banked, refusing the ones that do not match.

    Compared over :data:`MATCHED_PROVENANCE` and not over every field the record carries: the
    dependency set and the copy's file manifest are recorded to be read, not keyed, and the
    manifest restates what ``mamba_sources`` already keys on. A field this process does not
    supply is not compared, so a caller keying on a subset gets that subset.

    Args:
        directory: The bank. Absent is not an error; it is an empty bank.
        stored: The provenance this process would write. See :func:`provenance`.

    Returns:
        The cells by key, and one sentence per refused file.
    """
    banked: dict[str, Cell] = {}
    refused: list[str] = []
    if not os.path.isdir(directory):
        return banked, ()
    for name in sorted(os.listdir(directory)):
        # A void marker is a record and not a cell. Read by read_voids, never here.
        if not name.endswith(".json") or name.endswith(VOID_SUFFIX):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path) as handle:
                record = json.load(handle)
        except (OSError, ValueError) as broken:
            refused.append(f"{name}: unreadable ({broken})")
            continue
        found = record.get("provenance", {})
        differs = [
            f"{field} {found.get(field)!r} against {stored[field]!r}"
            for field in MATCHED_PROVENANCE
            if field in stored and found.get(field) != stored[field]
        ]
        if differs:
            # Refused and not corrected: no figure crosses a host, a card, a torch version
            # or a tree, and a bank is the one path by which one silently could.
            refused.append(f"{name}: provenance differs on {'; '.join(differs)}")
            continue
        try:
            cell = cell_from_record(record)
        except (KeyError, TypeError, ValueError) as broken:
            refused.append(f"{name}: corrupt record ({broken})")
            continue
        banked[record["key"]] = cell
    return banked, tuple(refused)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def group_verdicts(
    cells: Sequence[Cell], *, resources: Mapping[str, str] | None = None
) -> tuple[Verdict, ...]:
    """One verdict per shape class, boundary and execution.

    Args:
        cells: Measured cells, in any order.
        resources: Limiting resource per shape class, for the classes that earned
            :data:`NEITHER`.

    Returns:
        The verdicts, in first-seen order.
    """
    named = resources or {}
    keys: list[tuple[str, str, str]] = []
    ratios: dict[tuple[str, str, str], dict[int, float]] = {}
    refused: dict[tuple[str, str, str], dict[int, str]] = {}
    paths: dict[tuple[str, str, str], str] = {}
    worst: dict[tuple[str, str, str], Cell] = {}
    for cell in cells:
        key = (cell.point.shape_class, cell.boundary, cell.execution)
        if key not in ratios:
            ratios[key] = {}
            refused[key] = {}
            keys.append(key)
        # An unjudgeable row still reaches the table; it just carries no ratio into the
        # verdict, so the class loses that primary batch and the missing-batch rule applies.
        if judgeable(
            batch=cell.point.batch, execution=cell.execution, boundary=cell.boundary
        ):
            continue
        # The instrument gate is separate from the regime rule and is not merged into it: a
        # row it refuses was measured, so the class keeps its coverage and loses only the
        # universal claim. See unresolved and the refused argument of verdict.
        unfit = unresolved(cell)
        if unfit:
            refused[key][cell.point.batch] = unfit
            continue
        ratios[key][cell.point.batch] = cell.ratio
        held = worst.get(key)
        if held is None or cell.ratio > held.ratio:
            worst[key] = cell
        # The worst path in the class wins. One reference-path cell makes the class's
        # geometric mean a statement about torch, so it cannot be averaged away.
        paths[key] = (
            cell.resolved.path
            if cell.resolved.path != KERNEL_PATH
            else paths.get(key, cell.resolved.path)
        )
    return tuple(
        verdict(
            ratios[key],
            shape_class=key[0],
            boundary=key[1],
            execution=key[2],
            path=paths.get(key, KERNEL_PATH),
            limiting_resource=_resource_for(
                key, named=named, paths=paths, worst=worst, ratios=ratios
            ),
            refused=refused[key],
        )
        if ratios[key]
        else unjudged(
            shape_class=key[0],
            boundary=key[1],
            execution=key[2],
            refused=refused[key],
        )
        for key in keys
    )


def _resource_for(
    key: tuple[str, str, str],
    *,
    named: Mapping[str, str],
    paths: Mapping[tuple[str, str, str], str],
    worst: Mapping[tuple[str, str, str], Cell],
    ratios: Mapping[tuple[str, str, str], Mapping[int, float]],
) -> str:
    """The limiting resource for one class, by precedence.

    An explicitly supplied string wins, then a resource already measured on this fleet for
    that regime, then the worst row's own floor. The order matters: a caller who names the
    resource has evidence this function does not, and a fleet profile of the recurrence
    kernel is stronger than a two-term factorization of one row.

    Args:
        key: Shape class, boundary, execution.
        named: Caller-supplied resources, keyed by shape class.
        paths: Worst path per key.
        worst: Worst-ratio judged cell per key.
        ratios: Judged ratios per key, used only to locate the worst batch.

    Returns:
        The resource, or "" when none of the three sources has one.
    """
    supplied = named.get(key[0], "")
    if supplied:
        return supplied
    values = ratios[key]
    if not values:
        return ""
    worst_batch = max(values, key=lambda batch: values[batch])
    known = default_resource(
        path=paths.get(key, KERNEL_PATH),
        batch=worst_batch,
        execution=key[2],
        boundary=key[1],
    )
    if known:
        return known
    row = worst.get(key)
    return "" if row is None else measured_resource(row)


def witness_mark(one: Witness) -> str:
    """The witness as a table cell: the stamp abbreviated, with the windows behind it."""
    if not one.stamp:
        return "-"
    return f"{one.stamp[:4]}{one.replicates}"


def dispersion_lines(cells: Sequence[Cell]) -> tuple[str, ...]:
    """The quantiles of each row's own samples, one line per arm.

    Args:
        cells: Measured cells, in table order.

    Returns:
        Lines, empty when no cell in the sequence banked its samples.

    The table carries a median and a half-width per arm, which is what a verdict is read
    off. This block carries what those two summarize: the sample count, ``p10``, the median,
    ``p90`` and the extremes, every one an order statistic of the samples the row banked, so
    a reader recomputes the summary rather than taking it. A row banked before the samples
    were retained is named here rather than dropped, because a block holding fewer rows than
    the table would read as a block over all of them.
    """
    banked = [
        cell
        for cell in cells
        if cell.slinoss_samples_duration_us and cell.mamba_samples_duration_us
    ]
    if not banked:
        return ()
    header = (
        f"{'boundary':11s} {'exec':6s} {'B':>4s} {'arm':7s} {'n':>6s} {'p10':>11s} "
        f"{'median':>11s} {'p90':>11s} {'min':>11s} {'max':>11s}"
    )
    lines = [
        "dispersion, over each row's own samples. p10, p50 and p90 are nearest-rank order "
        "statistics, so every figure is a latency the loop observed.",
        header,
        "-" * len(header),
    ]
    for cell in banked:
        for arm, samples in (
            (SLINOSS, cell.slinoss_samples_duration_us),
            (MAMBA3, cell.mamba_samples_duration_us),
        ):
            lines.append(
                f"{cell.boundary:11s} {cell.execution:6s} {cell.point.batch:4d} "
                f"{arm:7s} {len(samples):6,d} "
                f"{order_statistic_us(samples, 0.10):11,.3f} "
                f"{order_statistic_us(samples, 0.50):11,.3f} "
                f"{order_statistic_us(samples, 0.90):11,.3f} "
                f"{min(samples):11,.3f} {max(samples):11,.3f}"
            )
    missing = len(cells) - len(banked)
    if missing:
        lines.append(
            f"{missing} of {len(cells)} rows banked no samples and are absent from this "
            f"block: their records predate the field. Their medians and half-widths stand; "
            f"only the recomputation does not."
        )
    return tuple(lines)


def render(
    cells: Sequence[Cell],
    verdicts: Sequence[Verdict],
    live: Liveness,
    *,
    fit: DramTimeFloor | None = None,
    void: str = "",
) -> tuple[str, ...]:
    """The table, its disclosures, and the verdicts.

    Args:
        cells: Measured cells.
        verdicts: One per class, boundary and execution.
        live: The liveness proof, printed above the table.
        fit: The DRAM law fitted in this process, cross-checked against the prior fit under
            the table. None when no floor was taken.
        void: Why the run does not stand, from :func:`sample_void`. Non-empty replaces
            every verdict with the reason.

    Returns:
        Lines, ready to print.
    """
    lines: list[str] = ["liveness proof, from the process that took the numbers:"]
    lines.extend(f"  {line}" for line in live.lines)
    lines.append("")
    lines.append(f"estimator: {ESTIMATOR}")
    lines.extend(f"disclosure: {one}" for one in DISCLOSURES)
    lines.append("")
    header = (
        f"{'boundary':11s} {'exec':6s} {'dtype':5s} {'d_model':>7s} {'sl3N':>5s} "
        f"{'m3ds':>5s} {'G':>4s} {'B':>4s} {'it':>6s} {'sl_us':>11s} {'sl+-%':>7s} "
        f"{'m3_us':>11s} {'m3+-%':>7s} {'ratio':>8s} {'path':>9s} {'regime':>7s} "
        f"{'sl_Bpt':>12s} {'m3_Bpt':>12s} {'cvbuf':>6s} {'sl_xfl':>7s} "
        f"{'m3_xfl':>7s} {'res':>4s} {'jdg':>4s} {'wit':>5s}  backends"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for cell in cells:
        point = cell.point
        # A dash and not a zero where no floor applies: a zero in a ratio column reads as
        # a measurement that came out at zero.
        sl_x = (
            f"{cell.floor.slinoss_x_floor:7.2f}"
            if cell.floor.available
            else f"{'-':>7s}"
        )
        m3_x = (
            f"{cell.floor.mamba_x_floor:7.2f}" if cell.floor.available else f"{'-':>7s}"
        )
        count = f"{cell.iters:,d}" if cell.iters else "-"
        unfit = judgeable(
            batch=point.batch, execution=cell.execution, boundary=cell.boundary
        ) or unresolved(cell)
        lines.append(
            f"{cell.boundary:11s} {cell.execution:6s} {point.dtype_name:5s} "
            f"{point.d_model:7d} {point.slinoss_d_state:5d} {point.mamba_d_state:5d} "
            f"{point.n_groups:4d} {point.batch:4d} {count:>6s} "
            f"{cell.slinoss_duration_us:11,.3f} {cell.slinoss_resolution_pct:7.3f} "
            f"{cell.mamba_duration_us:11,.3f} {cell.mamba_resolution_pct:7.3f} "
            f"{cell.ratio:8.4f} {cell.resolved.path:>9s} {cell.regime:>7s} "
            f"{int(cell.match.slinoss_state_bytes.total_bytes):12,d} "
            f"{int(cell.match.mamba_state_bytes.total_bytes):12,d} "
            f"{cell.match.slinoss_state_bytes.conv_buffer_count:2d}/"
            f"{cell.match.mamba_state_bytes.conv_buffer_count:<3d} "
            f"{sl_x} {m3_x} "
            f"{'y' if cell.paired_resolves else 'n':>4s} "
            f"{'n' if unfit else 'y':>4s} "
            f"{witness_mark(cell.witness):>5s}  "
            f"{','.join(cell.resolved.names)}"
        )
    if cells:
        lines.append("")
        lines.append(
            "it: timed iterations behind the row's two medians, which is what the "
            "half-width is a function of; a dash is a record written before the count was "
            "stored. A row re-measured at a higher count is a further sample of the same "
            "cell, not a correction to the earlier one."
        )
        # A regime reason prints once per distinct reason, because it is a property of the
        # regime and not of the row, and one line per row would bury the table it qualifies.
        # An instrument reason prints per row and names it: it quotes that row's own medians,
        # and a refused row that is documented is a result while a silent one is a hole.
        refused: list[str] = []
        for cell in cells:
            regime_unfit = judgeable(
                batch=cell.point.batch,
                execution=cell.execution,
                boundary=cell.boundary,
            )
            if regime_unfit:
                if regime_unfit not in refused:
                    refused.append(regime_unfit)
                    lines.append(f"jdg=n: {regime_unfit}")
                continue
            measured_unfit = unresolved(cell)
            if measured_unfit:
                lines.append(
                    f"jdg=n {cell.boundary}/{cell.execution}/B={cell.point.batch}: "
                    f"{measured_unfit}"
                )
        witnessed: list[str] = []
        for cell in cells:
            if cell.witness.detail not in witnessed:
                witnessed.append(cell.witness.detail)
                lines.append(f"wit={witness_mark(cell.witness)}: {cell.witness.detail}")
        lines.append(f"matching: {cells[0].match.detail}")
        lines.append(f"floor: {fit_cross_check(fit)}")
        # A missing floor is a property of the regime, so its reason prints once. A present
        # floor prints per row only where SLinOSS is ahead: a win is not reportable without
        # both sides' distance from their own floor, and a loss already reads off the ratio.
        seen: list[str] = []
        for cell in cells:
            if cell.floor.available and cell.ratio < 1.0:
                lines.append(
                    f"floor {cell.boundary}/{cell.execution}/B={cell.point.batch}: "
                    f"{cell.floor.detail}"
                )
            elif not cell.floor.available and cell.floor.detail not in seen:
                seen.append(cell.floor.detail)
                lines.append(f"floor {cell.regime} rows: {cell.floor.detail}")
        spread = dispersion_lines(cells)
        if spread:
            lines.append("")
            lines.extend(spread)
    lines.append("")
    if void:
        lines.append(void)
        return tuple(lines)
    for one in verdicts:
        lines.append(one.detail)
        lines.extend(f"  caveat: {caveat}" for caveat in one.caveats)
    return tuple(lines)


def as_json(
    cells: Sequence[Cell],
    verdicts: Sequence[Verdict],
    live: Liveness,
    grid: Grid,
    *,
    info: DeviceInfo | None,
    clocks: ClockPolicy | None,
    quiet: Contention | None,
    fit: DramTimeFloor | None = None,
    void: str = "",
    bank: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything the run measured, serialized.

    A pass that ran and found something reaches the machine-readable output and not only
    the prose, so the rejections, the liveness lines and the contention stamp are all
    fields here.

    Args:
        cells: Measured cells.
        verdicts: The verdicts.
        live: The liveness proof.
        grid: The enumerated grid and its refusals.
        info: Device record, or None when nothing was measured.
        clocks: Clock stamp, or None.
        quiet: The contention probe that opened the gate, or None.
        fit: The DRAM law fitted in this process, or None.
        void: Why the run does not stand, from :func:`sample_void`. Empty when it does.
        bank: Which cells this invocation read, measured and discarded, or None when no
            bank was used.

    Returns:
        A JSON-ready mapping.
    """
    return {
        "void": void,
        "bank": None if bank is None else dict(bank),
        "estimator": ESTIMATOR,
        "disclosures": {
            "version": VERSION_DISCLOSURE,
            "interpreter": INTERPRETER_DISCLOSURE,
            "eager_host": EAGER_HOST_DISCLOSURE,
            "routing": ROUTING_DISCLOSURE,
            "capture_hazard": CAPTURE_HAZARD_DISCLOSURE,
            "bank": BANK_DISCLOSURE,
            "conv": CONV_DISCLOSURE,
            "fp32": FP32_DISCLOSURE,
            "eager": EAGER_DISCLOSURE,
            "graph_launch": GRAPH_LAUNCH_DISCLOSURE,
            "sub_l2": SUB_L2_DISCLOSURE,
            "mamba_fp32_state": MAMBA_FP32_STATE_DISCLOSURE,
            "mamba_fusion": MAMBA_FUSION_DISCLOSURE,
            "witness": WITNESS_DISCLOSURE,
        },
        "dram_fit": None
        if fit is None
        else {
            "fixed_duration_us": float(fit.fixed_duration_us),
            "asymptotic_gbs": float(fit.asymptotic_gbs),
            "max_residual_pct": float(fit.max_residual_pct),
            "prior_fixed_duration_us": PRIOR_DRAM_FIXED_US,
            "prior_asymptotic_gbs": PRIOR_DRAM_RATE_GBS,
            "prior_ceiling_gbs": PRIOR_DRAM_CEILING_GBS,
            "cross_check": fit_cross_check(fit),
        },
        "liveness": {
            "lines": list(live.lines),
            "live": live.live,
            "loaded": live.loaded,
            "recurrence_live": live.recurrence_live,
            "slinoss_package": live.slinoss_package,
            "torch": live.torch_version,
        },
        "device": None
        if info is None
        else {
            "name": info.name,
            "capability": info.capability,
            "sm_count": int(info.sm_count),
            "l2_bytes": int(info.l2_bytes),
            "total_memory_bytes": int(info.total_memory_bytes),
            "clocks": info.clocks.stamp,
            "sharing": info.sharing.stamp,
        },
        "clocks": None if clocks is None else clocks.stamp,
        "contention": None if quiet is None else quiet.stamp,
        "primary_batches": list(PRIMARY_BATCHES),
        "grid": {
            "enumerated": [point.describe() for point in grid.points],
            "rejections": [one.detail for one in grid.rejections],
        },
        "cells": [
            {
                "shape_class": cell.point.shape_class,
                "boundary": cell.boundary,
                "execution": cell.execution,
                "batch": cell.point.batch,
                "backends": list(cell.resolved.names),
                "path": cell.resolved.path,
                "path_detail": cell.resolved.detail,
                "regime": cell.regime,
                "slinoss_duration_us": float(cell.slinoss_duration_us),
                "slinoss_resolution_pct": float(cell.slinoss_resolution_pct),
                "slinoss_spread_pct": float(cell.slinoss_spread_pct),
                "slinoss_samples_duration_us": [
                    float(one) for one in cell.slinoss_samples_duration_us
                ],
                "mamba_duration_us": float(cell.mamba_duration_us),
                "mamba_resolution_pct": float(cell.mamba_resolution_pct),
                "mamba_spread_pct": float(cell.mamba_spread_pct),
                "mamba_samples_duration_us": [
                    float(one) for one in cell.mamba_samples_duration_us
                ],
                "ratio": cell.ratio,
                "paired_delta_us": float(cell.paired_delta_us),
                "paired_low_us": float(cell.paired_low_us),
                "paired_high_us": float(cell.paired_high_us),
                "paired_resolves": cell.paired_resolves,
                "slinoss_state_bytes_per_token": int(
                    cell.match.slinoss_state_bytes.total_bytes
                ),
                "mamba_state_bytes_per_token": int(
                    cell.match.mamba_state_bytes.total_bytes
                ),
                "slinoss_conv_buffers": cell.match.slinoss_state_bytes.conv_buffer_count,
                "mamba_conv_buffers": cell.match.mamba_state_bytes.conv_buffer_count,
                "floor_available": cell.floor.available,
                "slinoss_moved_bytes": cell.floor.slinoss_moved_bytes,
                "slinoss_floor_us": float(cell.floor.slinoss_floor_us),
                "slinoss_x_floor": cell.floor.slinoss_x_floor,
                "mamba_moved_bytes": cell.floor.mamba_moved_bytes,
                "mamba_floor_us": float(cell.floor.mamba_floor_us),
                "mamba_x_floor": cell.floor.mamba_x_floor,
                "floor_detail": cell.floor.detail,
                "held_equal": list(cell.match.held),
                "match_detail": cell.match.detail,
                "iters": cell.iters,
                "unjudged_reason": judgeable(
                    batch=cell.point.batch,
                    execution=cell.execution,
                    boundary=cell.boundary,
                )
                or unresolved(cell),
                "witness_stamp": cell.witness.stamp,
                "witness_foreign_mib": cell.witness.foreign_mib,
                "witness_replicates": cell.witness.replicates,
                "witness_agrees": cell.witness.agrees,
                "witness_detail": cell.witness.detail,
            }
            for cell in cells
        ],
        "verdicts": [
            {
                "word": one.word,
                "shape_class": one.shape_class,
                "boundary": one.boundary,
                "execution": one.execution,
                "path": one.path,
                "batches": list(one.batches),
                "missing_batches": list(one.missing_batches),
                "refused_batches": list(one.refused_batches),
                "geomean_ratio": one.geomean_ratio,
                "worst_ratio": one.worst_ratio,
                "worst_batch": one.worst_batch,
                "best_ratio": one.best_ratio,
                "gap_pct": float(one.gap_pct),
                "limiting_resource": one.limiting_resource,
                "caveats": list(one.caveats),
                "detail": one.detail,
            }
            # Empty under a void run, not merely flagged: a consumer that reads the word
            # without reading the void field must not find one to read.
            for one in (() if void else verdicts)
        ],
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, nargs="+", default=list(D_MODELS))
    parser.add_argument(
        "--dtype", nargs="+", choices=list(DTYPES), default=list(DTYPES)
    )
    parser.add_argument("--batch", type=int, nargs="+", default=list(PRIMARY_BATCHES))
    parser.add_argument(
        "--d-state", type=int, nargs="+", default=list(SL_D_STATES), help="SLinOSS 3N."
    )
    parser.add_argument(
        "--mamba-d-state",
        type=int,
        nargs="+",
        default=list(M3_SISO_D_STATES),
        help="Mamba3 d_state.",
    )
    parser.add_argument(
        "--sharing", nargs="+", choices=list(SHARINGS), default=list(SHARINGS)
    )
    parser.add_argument("--mode", nargs="+", choices=list(MODES), default=[SISO])
    parser.add_argument(
        "--boundary", nargs="+", choices=list(BOUNDARIES), default=list(BOUNDARIES)
    )
    parser.add_argument(
        "--execution", nargs="+", choices=list(EXECUTIONS), default=list(EXECUTIONS)
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--json", default="", help="Write the full record here. Outside the tree."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enumerate the grid, print the shapes and the state bytes, and exit. Takes "
        "no timed sample and needs no idle card.",
    )
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help="Skip the in-process DRAM fit. Rows then carry no floor: a fit from another "
        "process or another host does not price this card.",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=600.0,
        help="Longest wait for an idle card before giving up. A contended sample is void, "
        "so the driver waits rather than stamping one.",
    )
    parser.add_argument(
        "--bank",
        default="",
        help="Directory of per-cell JSON artifacts, outside the tree. Cells already there "
        "are read and skipped, cells measured here are written to it one at a time, so an "
        "interrupted run loses at most the cell in flight.",
    )
    parser.add_argument(
        "--order",
        choices=list(ORDERS),
        default=DECISIVE,
        help="Cell order. 'decisive' measures the rows that can carry a verdict first; "
        "'nested' walks the grid as enumerated.",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=0,
        help="Stop after this many cells are measured in this invocation, banking each. "
        "Zero measures every unbanked cell.",
    )
    parser.add_argument(
        "--close-settle-s",
        type=float,
        default=CLOSE_SETTLE_S,
        help="Seconds between the last timed region and the closing probe. Below the "
        "settle time the probe reads this run's own utilization and voids the run against "
        "itself.",
    )
    parser.add_argument(
        "--graph-cells-per-process",
        type=int,
        default=GRAPH_CELLS_PER_PROCESS,
        help="Graph cells this invocation may measure. A cell over the budget is deferred "
        "to the next invocation, not refused: a second decode graph capture in one process "
        "fires a device-side assert at some shapes. Raising this exposes the run.",
    )
    parser.add_argument(
        "--replicate-gap-s",
        type=float,
        default=REPLICATE_GAP_S,
        help="Seconds between the two windows of a residency-witnessed cell.",
    )
    parser.add_argument(
        "--exclusive-only",
        action="store_true",
        help="Refuse the residency witness. Only a card the house gate admits is measured, "
        "which on a shared fleet usually means nothing is measured.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Print the table from the bank and take no sample. Every pending cell is "
        "deferred and the idle-card gate is skipped, because the gate protects samples and "
        "this invocation takes none. It admits no contended sample; it admits no sample.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Enumerate the grid, measure what a card allows, and print the verdict.

    Returns:
        Process exit status. Zero when every verdict is a positive word; one when a verdict
        is :data:`NEITHER`, a primary batch is missing, the card was contended when the last
        sample ended, no decode kernel backend is registered at all, or a cell poisoned the
        CUDA context. The last case prints no table and writes no JSON.
    """
    args = parse_args(argv)
    grid = enumerate_grid(
        d_models=args.d_model,
        dtype_names=args.dtype,
        batches=args.batch,
        slinoss_d_states=args.d_state,
        mamba_d_states=args.mamba_d_state,
        sharings=args.sharing,
        modes=args.mode,
    )
    print(f"enumerated {len(grid.points)} cells, refused {len(grid.rejections)}:")
    for one in grid.rejections:
        print(f"  refused: {one.detail}")

    if args.smoke:
        for point in grid.points:
            match = match_shapes(
                point.config,
                dtype=point.dtype,
                batch=point.batch,
                mamba_d_state=point.mamba_d_state,
                rank=point.rank,
            )
            print(
                f"  {point.describe()}  "
                f"sl_state={int(match.slinoss_state_bytes.total_bytes):,}B  "
                f"m3_state={int(match.mamba_state_bytes.total_bytes):,}B  "
                f"cvbuf {match.slinoss_state_bytes.conv_buffer_count}/"
                f"{match.mamba_state_bytes.conv_buffer_count}  "
                # Per boundary, not per execution alone: the two boundaries cross out of L2
                # at different batches, so one pair of labels would describe one of them.
                f"regime rec "
                f"{regime(batch=point.batch, execution=EAGER, boundary=RECURRENCE)}/"
                f"{regime(batch=point.batch, execution=GRAPH, boundary=RECURRENCE)}"
                f" step "
                f"{regime(batch=point.batch, execution=EAGER, boundary=WHOLE_STEP)}/"
                f"{regime(batch=point.batch, execution=GRAPH, boundary=WHOLE_STEP)}"
            )
        for one in DISCLOSURES:
            print(f"disclosure: {one}")
        return 0

    device = require_cuda(args.device)
    ordinal = device_ordinal(device)
    live = liveness(dtype=DTYPES[args.dtype[0]], boundaries=args.boundary)
    for line in live.lines:
        print(f"  {line}")
    closed = kernel_gate(live)
    if closed:
        print(f"gate closed: {closed}")
        return 1
    if not live.live:
        # Not an abort: a dtype with no instantiation is a property of the operator, and the
        # row carries it as a mixed or reference path that is refused a dominates. The
        # decode registry admits float32 while so3ssd does not, so the recurrence boundary
        # can be a kernel row at a dtype where the whole step cannot.
        print(
            f"dispatch resolved to a reference path at {args.dtype[0]}; recurrence "
            f"kernel live: {live.recurrence_live}. {FP32_DISCLOSURE}"
        )

    before = contention(ordinal)
    stamp, replicates, why = admit(before, exclusive_only=args.exclusive_only)
    if args.render_only:
        # The gate protects samples. This invocation takes none, so waiting on a busy card
        # would only withhold a table already computed from banked numbers. The reading is
        # still printed and still stored, since the table names the card it was taken on.
        quiet = before
        # The empty stamp is what admit returns for a card it refuses, and it is right here
        # for the other reason: no row this invocation prints was measured by it.
        stamp, replicates = NO_WITNESS.stamp, 0
        why = (
            f"render only: no sample taken in this invocation. Card read {before.stamp}"
        )
    elif not stamp:
        # The card is running foreign compute. Wait for it: a contended sample is void, and
        # waiting is correct where substituting is not.
        try:
            quiet = await_exclusive(ordinal, timeout_s=args.idle_timeout_s)
        except ContendedDevice as shut:
            print(f"measurement pending: no card idled, so nothing was sampled. {shut}")
            print(f"admission: {why}")
            return 1
        stamp, replicates, why = admit(quiet, exclusive_only=args.exclusive_only)
        if not stamp:
            print(f"measurement pending: nothing was sampled. {why}")
            return 1
    else:
        quiet = before
    clocks = clock_policy(ordinal)
    info = device_info(ordinal)
    print(f"device {ordinal}  {info.name}  {clocks.stamp}")
    print(f"contention before {before.stamp}")
    print(f"admission: {why}")

    stored = provenance(live, device_name=card_identity(ordinal, name=info.name))
    origin = stored["competitor_origin"]
    print(
        f"competitor {stored['mamba_package']}  digest {stored['mamba_sources']}  "
        f"route {origin['route']}  git {origin['git']}  files {origin['file_count']}  "
        f"copied from {origin['copied_from']}"
    )
    deps = stored["deps"]
    print("deps " + "  ".join(f"{key} {value}" for key, value in deps.items()))
    banked, unusable = read_bank(args.bank, stored=stored) if args.bank else ({}, ())
    for one in unusable:
        print(f"bank refused: {one}")
    queue = tasks(
        grid,
        boundaries=args.boundary,
        executions=args.execution,
        order=args.order,
    )
    pending = [one for one in queue if one.key not in banked]
    print(
        f"bank {args.bank or '(none)'}: {len(banked)} cells read, {len(pending)} pending, "
        f"order {args.order}, cap {args.only or 'none'}"
    )

    # Fitted here, on this card, in this process. The prior fit is only ever a check on it.
    fit = None if args.no_floor else dram_time_floor(device)
    print(fit_cross_check(fit))

    admitted = Witness(
        stamp=stamp,
        foreign_mib=float(before.foreign_memory_mib),
        replicates=replicates,
        agrees=True,
        detail=why,
    )
    taken: list[str] = []
    discarded: list[str] = []
    deferred: list[str] = []
    voided: list[str] = []
    graphed = 0
    for task in pending:
        if args.only and len(taken) >= args.only:
            break
        if args.render_only:
            # Deferred rather than refused, as an over-budget graph cell is: the cell is
            # measurable, this invocation is simply not the one that measures it.
            deferred.append(task.key)
            continue
        if defers(task, graphed=graphed, budget=args.graph_cells_per_process):
            # Deferred and not refused: the second decode capture in one process is the
            # hazard, so the loop over cells belongs outside the process.
            deferred.append(task.key)
            continue
        if task.execution == GRAPH:
            # Counted before the measurement and not after: the capture is built whether or
            # not the cell survives its replicates.
            graphed += 1
        try:
            cell, lost = measure_replicated(
                task.point,
                boundary=task.boundary,
                execution=task.execution,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                clocks=clocks,
                fit=fit,
                witness_stamp=stamp,
                replicates=replicates,
                foreign_mib=float(before.foreign_memory_mib),
                ordinal=ordinal,
                gap_s=args.replicate_gap_s,
            )
        except Exception as failure:
            reason = f"{type(failure).__name__}: {failure}"
            voided.append(f"{task.key}: {reason}")
            if args.bank:
                print(
                    f"cell {task.key}: VOID, marker at "
                    f"{write_void(args.bank, task.key, reason, stored=stored)}"
                )
            else:
                print(f"cell {task.key}: VOID -- {reason}")
            if poisons(failure):
                # No table and no JSON from a poisoned process. A later launch here can
                # still return an ordinary-looking number, and it would mean nothing.
                print(
                    f"aborting: the CUDA context is poisoned, so every later cell in this "
                    f"process is untrustworthy even if it produces numbers. Measured "
                    f"{len(taken)} before it. {CAPTURE_HAZARD_DISCLOSURE}"
                )
                return 1
            continue
        if cell is None:
            print(f"cell {task.key}: {lost}")
            discarded.append(f"{task.key}: {lost}")
            continue
        banked[task.key] = cell
        taken.append(task.key)
        if args.bank:
            print(
                f"cell {task.key}: banked at {write_cell(args.bank, cell, stored=stored)}"
            )
        else:
            print(f"cell {task.key}: measured")
    # Ordered by the queue and not by arrival, so a table built from a bank filled over
    # several windows reads the same as one filled in a single pass.
    cells = [banked[one.key] for one in queue if one.key in banked]

    after = closing_probe(ordinal, device=device, settle_s=args.close_settle_s)
    # The fit counts: it is timed on this card, and its rate is printed and cross-checked.
    sampled = bool(taken or discarded or voided) or fit is not None
    void = sample_void(after, witness=admitted, sampled=sampled)
    verdicts = group_verdicts(cells)
    for line in render(cells, verdicts, live, fit=fit, void=void):
        print(line)
    print(f"contention after {after.stamp}")
    print(
        f"this invocation measured {len(taken)}, discarded {len(discarded)}, voided "
        f"{len(voided)}, deferred {len(deferred)}, table carries {len(cells)} of "
        f"{len(queue)}"
    )
    for one in voided:
        print(f"void: {one}")
    for one in read_voids(args.bank) if args.bank else ():
        print(f"bank void marker: {one}")
    if deferred:
        print(
            f"deferred to the next invocation, graph budget {args.graph_cells_per_process} "
            f"spent: {deferred[0]} and {len(deferred) - 1} more"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                as_json(
                    cells,
                    verdicts,
                    live,
                    grid,
                    info=info,
                    clocks=clocks,
                    quiet=quiet,
                    fit=fit,
                    void=void,
                    bank={
                        "directory": args.bank,
                        "order": args.order,
                        "schema": BANK_SCHEMA,
                        "provenance": stored,
                        "measured_here": taken,
                        "discarded_here": discarded,
                        "voided_here": voided,
                        "deferred_here": deferred,
                        "void_markers": list(read_voids(args.bank))
                        if args.bank
                        else [],
                        "refused_records": list(unusable),
                        "carried": len(cells),
                        "enumerated": len(queue),
                    },
                ),
                handle,
                indent=2,
            )
        print(f"wrote {args.json}")
    if void:
        return 1
    return 0 if all(one.word != NEITHER for one in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
