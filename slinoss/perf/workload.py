"""The benchmarked workloads. One definition per operator, shared by every driver.

The bench, the NCU target, and the NSYS target all call the same functions here,
so the benchmarked path is the shipped path: :func:`slinoss.ops.so3ssd.so3ssd` and
:func:`slinoss.ops.conv.causal_conv1d`, with no variant reachable from a script
and not from the public API.

Five shape vocabularies, because the five operators have five. The scan is indexed
by ``(B,H,T,P,N,L)``, the causal conv1d by ``(B,T,D,W)``, the parameter frontier by
``(B,T,H,3N,G)``, the block by ``(B,T,d_model,d_ffn)`` and the fused tail by the
scan shape plus the width of the projection its gate is a band of. One name denotes
one layer measured in five places: the conv's ``D`` is the scan's ``H*P``, and the
frontier, the block and the tail hold the scan shape itself and read their widths
off the :class:`slinoss.config.SLinOSSConfig` it implies, so none can drift from it.

``trans`` and ``K`` are produced by the real parameter maps and then detached, so
the numerical invariants hold on the benchmarked tensors -- ``ls <= 0`` and
``|w| <= w_max`` -- while the measurement covers the scan alone. Fabricating them
from ``randn`` would put ``ls > 0`` into a decay prefix and measure a kernel that
cannot run in training.

The backward is measured with :func:`torch.autograd.grad` rather than
``loss.backward()``: no reduction is added to the graph, and no gradient is
accumulated into a ``.grad`` buffer, so an ``aten::fill_`` cannot contaminate the
backward bucket.

This module is deliberately absent from ``slinoss.perf``'s exports. It depends on
:mod:`slinoss.ops.so3ssd`, which depends on the timing primitives, so importing
it from the package initializer would build a cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple

import torch
from torch import Tensor

from slinoss._guard import PROJ_ALIGN
from slinoss.config import SLinOSSConfig
from slinoss.ops.block import rmsnorm, rmsnorm_residual, swiglu
from slinoss.ops.conv import causal_conv1d, conv_output_shape, conv_state_shape
from slinoss.ops.mixer import mixer_tail
from slinoss.ops.scanprep import scanprep
from slinoss.ops.scanprep.reference import PARAM_COLS, pack_params, scanprep_ref
from slinoss.ops.so3ssd import so3ssd
from slinoss.perf.timing import region
from slinoss.perf.units import Count

__all__ = [
    "BLOCK",
    "BLOCK_SHAPES",
    "CONV",
    "CONV_SHAPES",
    "MIXER",
    "MIXER_SHAPES",
    "OPS",
    "PREP_SHAPES",
    "SCANPREP",
    "SHAPES",
    "SHAPE_NAMES",
    "SO3SSD",
    "W_MAX",
    "BlockInputs",
    "BlockShape",
    "ConvInputs",
    "ConvShape",
    "MixerInputs",
    "MixerShape",
    "OpInputs",
    "OpShape",
    "PrepInputs",
    "PrepShape",
    "block_forward_only",
    "block_shape_by_name",
    "block_step",
    "conv_forward_only",
    "conv_shape_by_name",
    "conv_step",
    "forward_only",
    "layer_config",
    "make_block_inputs",
    "make_conv_inputs",
    "make_inputs",
    "make_mixer_inputs",
    "make_prep_inputs",
    "mixer_forward_only",
    "mixer_shape_by_name",
    "mixer_step",
    "prep_forward_only",
    "prep_shape_by_name",
    "prep_step",
    "shape_by_name",
    "step",
]

W_MAX: Final = 3.0
"""Rotation-vector bound used by every benchmark. Below pi, as I2 requires."""

SO3SSD: Final = "so3ssd"
CONV: Final = "conv"
SCANPREP: Final = "scanprep"
BLOCK: Final = "block"
MIXER: Final = "mixer"
OPS: Final[tuple[str, ...]] = (SO3SSD, CONV, SCANPREP, BLOCK, MIXER)
"""Benchmarkable operators. The whole registry every driver dispatches on.

Appended to, never reordered: the first entry is every driver's default, so moving
it would change what a command with no ``--op`` profiles."""


@dataclass(frozen=True)
class OpShape:
    """One benchmarked problem size.

    Attributes:
        name: Shape name, used on the command line and in the report.
        bsz: Batch, ``B``.
        heads: Heads, ``H``.
        seq: Sequence length, ``T``. Not required to be a multiple of ``chunk``.
        rows: Rows per head, ``P``. Multiple of 16.
        lanes: Independent 3-vectors, ``N``. Multiple of 16.
        chunk: Chunk length ``L``.
    """

    name: str
    bsz: int
    heads: int
    seq: int
    rows: int
    lanes: int
    chunk: int

    @property
    def d_state(self) -> int:
        """Trailing state width, ``3N``."""
        return 3 * self.lanes

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.bsz * self.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.bsz} H={self.heads} T={self.seq} "
            f"P={self.rows} N={self.lanes} 3N={self.d_state} L={self.chunk}"
        )


SHAPES: Final[tuple[OpShape, ...]] = (
    OpShape("tiny", bsz=1, heads=1, seq=256, rows=16, lanes=16, chunk=64),
    OpShape("standard", bsz=4, heads=12, seq=2048, rows=48, lanes=16, chunk=64),
    OpShape("wide", bsz=4, heads=12, seq=2048, rows=64, lanes=32, chunk=64),
    OpShape("long", bsz=2, heads=12, seq=8192, rows=48, lanes=16, chunk=128),
    OpShape("ragged", bsz=4, heads=12, seq=2004, rows=48, lanes=16, chunk=64),
)
"""The standard sizes. Every optimization is measured at all of them, before and
after, with the same commands. ``ragged`` has a sequence length that is not a
multiple of the chunk, so a tail-handling regression shows up in the bench and
not only in the tests."""


def shape_by_name(name: str) -> OpShape:
    """Look up a standard shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`SHAPES`.
    """
    for shape in SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no shape {name!r}; have {[s.name for s in SHAPES]}")


class OpInputs(NamedTuple):
    """Operator inputs at one shape.

    Attributes:
        U: ``(B,H,T,P)``, low precision.
        trans: ``(B,H,T,4)``, float32, packing ``(w_x, w_y, w_z, ls)``.
        K: ``(B,H,T,2,4)``, float32, packing ``(kr, g, h, 0)`` per tap.
        B: ``(B,H,T,3N)``, low precision.
        C: ``(B,H,T,3N)``, low precision.
        dy: ``(B,H,T,P)`` output-gradient seed, preallocated so the backward
            measurement contains no allocation of its own.
    """

    U: Tensor
    trans: Tensor
    K: Tensor
    B: Tensor
    C: Tensor
    dy: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The five tensors gradients are taken with respect to."""
        return (self.U, self.trans, self.K, self.B, self.C)


def make_inputs(
    shape: OpShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = True,
    seed: int = 0,
) -> OpInputs:
    """Build operator inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype for ``U``, ``B``, ``C``, and ``dy``. ``trans`` and ``K`` are
            float32 regardless, as I4 requires.
        requires_grad: Whether the five differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    lead = (shape.bsz, shape.heads, shape.seq)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    with torch.no_grad():
        params = scanprep_ref(
            pack_params(
                randn(*lead, 3, dt=torch.float32),
                randn(*lead, dt=torch.float32),
                randn(*lead, 2, 3, dt=torch.float32),
            ),
            # trans and K do not depend on bc, and B and C are drawn below, so bc
            # is the narrowest legal placeholder: one group, outputs discarded.
            torch.zeros(
                shape.bsz,
                shape.seq,
                2 * shape.d_state,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(shape.heads, PARAM_COLS, dtype=torch.float32, device=device),
            heads=shape.heads,
            state_dim=shape.d_state,
            w_max=W_MAX,
        )
    trans = params.trans.detach().requires_grad_(requires_grad)
    K = params.K.detach().requires_grad_(requires_grad)
    return OpInputs(
        U=randn(*lead, shape.rows).requires_grad_(requires_grad),
        trans=trans,
        K=K,
        B=randn(*lead, shape.d_state).requires_grad_(requires_grad),
        C=randn(*lead, shape.d_state).requires_grad_(requires_grad),
        dy=randn(*lead, shape.rows),
    )


def forward_only(
    inputs: OpInputs, chunk: int, *, backend: str | None = None, prefix: str = "op"
) -> Callable[[], None]:
    """A callable that runs the forward under ``no_grad``.

    Args:
        inputs: Operator inputs.
        chunk: Chunk length.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. Two arms measured in one loop need two
            prefixes, or the recorder sums their regions into one and the inner
            tree describes neither.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad(), region(f"{prefix}.forward"):
            so3ssd(
                inputs.U,
                inputs.trans,
                inputs.K,
                inputs.B,
                inputs.C,
                chunk,
                backend=backend,
            )

    return run


def step(
    inputs: OpInputs,
    chunk: int,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "op",
) -> Callable[[], None]:
    """A callable that runs the forward and the backward.

    Args:
        inputs: Operator inputs. The five differentiable ones must require grad.
        chunk: Chunk length.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all five.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("step needs at least one input requiring grad")

    def run() -> None:
        with region(f"{prefix}.forward"):
            y = so3ssd(
                inputs.U,
                inputs.trans,
                inputs.K,
                inputs.B,
                inputs.C,
                chunk,
                backend=backend,
            ).y
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, targets, inputs.dy)

    return run


@dataclass(frozen=True)
class ConvShape:
    """One benchmarked causal conv1d size.

    Attributes:
        name: Shape name, used on the command line and in the report.
        bsz: Batch, ``B``.
        seq: Sequence length, ``T``. Not required to be a multiple of the
            kernel's time tile.
        channels: Channels, ``D``. The mixer's ``d_inner``, i.e. the scan's
            ``H*P``.
        width: Tap count, ``W``.
    """

    name: str
    bsz: int
    seq: int
    channels: int
    width: int

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Streaming-state shape, ``(B,W-1,D)``."""
        return conv_state_shape(self.bsz, self.width, self.channels)

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.bsz * self.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.bsz} T={self.seq} D={self.channels} W={self.width}"
        )


CONV_SHAPES: Final[tuple[ConvShape, ...]] = (
    ConvShape("tiny", bsz=1, seq=256, channels=16, width=4),
    ConvShape("standard", bsz=4, seq=2048, channels=576, width=4),
    ConvShape("wide", bsz=4, seq=2048, channels=768, width=8),
    ConvShape("long", bsz=2, seq=8192, channels=576, width=4),
    ConvShape("ragged", bsz=4, seq=2004, channels=576, width=4),
)
"""The standard conv sizes. One name per entry of :data:`SHAPES`, and the same
names, so one ``--shape`` table serves both operators: ``D`` is that shape's
``H*P``, which is what the mixer feeds the conv. ``wide`` also takes the widest
tap bank the kernel instantiates, and ``ragged`` a sequence length that is not a
multiple of either time tile. The two directions tile time differently, so a
length ragged against one and exact against the other would leave the bench blind
to a tail regression in that direction."""

SHAPE_NAMES: Final[tuple[str, ...]] = tuple(s.name for s in SHAPES)
"""One shape-name table for every operator, so a driver offers one ``--shape``
list whatever ``--op`` it was handed and a typo is rejected before anything is
allocated. A test holds :data:`CONV_SHAPES` to the same names."""


def conv_shape_by_name(name: str) -> ConvShape:
    """Look up a standard conv shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`CONV_SHAPES`.
    """
    for shape in CONV_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no conv shape {name!r}; have {[s.name for s in CONV_SHAPES]}")


class ConvInputs(NamedTuple):
    """Causal conv1d inputs at one shape.

    One dtype throughout: the native backend is one template instantiation per
    dtype and refuses a mixed-dtype call rather than promoting an operand.

    Attributes:
        x: ``(B,T,D)``.
        weight: ``(D,W)``.
        bias: ``(D,)``.
        initial_state: ``(B,W-1,D)``, the window before ``x``. Present, because
            the decode path always carries one and its pullback is a separate
            kernel arc.
        dy: Output-gradient seed at the output's own shape, ``(B,T,D)`` or
            ``(B,H,T,P)``, preallocated so the backward measurement contains no
            allocation of its own.
        d_head: The output layout ``dy`` was allocated at, carried here rather
            than passed to the runners: the seed's shape and the forward's layout
            are one fact, and two arguments for it can disagree.
    """

    x: Tensor
    weight: Tensor
    bias: Tensor
    initial_state: Tensor
    dy: Tensor
    d_head: int | None = None

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The four tensors gradients are taken with respect to."""
        return (self.x, self.weight, self.bias, self.initial_state)

    @property
    def tensors(self) -> tuple[Tensor, ...]:
        """Every tensor, in field order. ``d_head`` is not one."""
        return (*self.differentiable, self.dy)


def make_conv_inputs(
    shape: ConvShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = True,
    seed: int = 0,
    d_head: int | None = None,
) -> ConvInputs:
    """Build causal conv1d inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype of every operand.
        requires_grad: Whether the four differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.
        d_head: Output layout. ``dy`` is allocated at the shape that layout
            produces, so the backward is handed the cotangent the forward's output
            would carry, and the runners read the layout back off the inputs.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*size: int) -> Tensor:
        return torch.randn(*size, dtype=dtype, device=device, generator=gen)

    return ConvInputs(
        x=randn(shape.bsz, shape.seq, shape.channels).requires_grad_(requires_grad),
        weight=randn(shape.channels, shape.width).requires_grad_(requires_grad),
        bias=randn(shape.channels).requires_grad_(requires_grad),
        initial_state=randn(*shape.state_shape).requires_grad_(requires_grad),
        dy=randn(*conv_output_shape(shape.bsz, shape.seq, shape.channels, d_head)),
        d_head=d_head,
    )


def conv_forward_only(
    inputs: ConvInputs, *, backend: str | None = None, prefix: str = "conv"
) -> Callable[[], None]:
    """A callable that runs the conv forward under ``no_grad``.

    Args:
        inputs: Conv inputs. Their ``d_head`` selects the output layout.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad(), region(f"{prefix}.forward"):
            causal_conv1d(
                inputs.x,
                inputs.weight,
                inputs.bias,
                initial_state=inputs.initial_state,
                d_head=inputs.d_head,
                backend=backend,
            )

    return run


def conv_step(
    inputs: ConvInputs,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "conv",
) -> Callable[[], None]:
    """A callable that runs the conv forward and backward.

    The returned window is dropped, so its cotangent is absent rather than zero
    and the backward skips the state pullback. That is the training path.

    Args:
        inputs: Conv inputs. The four differentiable ones must require grad, and
            their ``d_head`` selects the output layout ``dy`` was allocated at.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all four.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("conv step needs at least one input requiring grad")

    def run() -> None:
        with region(f"{prefix}.forward"):
            y = causal_conv1d(
                inputs.x,
                inputs.weight,
                inputs.bias,
                initial_state=inputs.initial_state,
                d_head=inputs.d_head,
                backend=backend,
            ).y
        with region(f"{prefix}.backward"):
            torch.autograd.grad(y, targets, inputs.dy)

    return run


def layer_config(shape: OpShape, *, groups: int = 1) -> SLinOSSConfig:
    """The layer whose scan runs at ``shape``.

    ``d_inner`` is ``H*P``, so at the default expansion the residual stream is half
    of it and the FFN hidden is four times the stream. Every width the frontier and
    the block are measured at is read off the returned config, so a driver restates
    none of them and none can drift from the scan shape.

    Args:
        shape: The scan shape.
        groups: ``G``, groups sharing one ``B``/``C`` pair.

    Returns:
        The config.

    Raises:
        ValueError: If ``H*P`` is odd, so no integer ``d_model`` expands to it, or
            if any shape invariant of :class:`slinoss.config.SLinOSSConfig` fails.
    """
    inner = shape.heads * shape.rows
    if inner % 2 != 0:
        raise ValueError(f"H*P must be even to halve, got {inner}")
    return SLinOSSConfig(
        d_model=inner // 2,
        d_state=shape.d_state,
        expand=2.0,
        d_head=shape.rows,
        n_groups=groups,
        chunk_size=shape.chunk,
    )


@dataclass(frozen=True)
class PrepShape:
    """One benchmarked parameter-frontier size.

    Attributes:
        scan: The scan this frontier feeds. ``B``, ``T``, ``H`` and ``3N`` are read
            off it.
        groups: ``G``, groups sharing one ``B``/``C`` pair. Divides ``H``.
    """

    scan: OpShape
    groups: int

    @property
    def name(self) -> str:
        """Shape name. The scan's, because it is the same layer."""
        return self.scan.name

    @property
    def params_width(self) -> int:
        """Parameter-band width, ``10*H``."""
        return PARAM_COLS * self.scan.heads

    @property
    def bc_width(self) -> int:
        """``B``/``C`` band width, ``2*G*3N``."""
        return 2 * self.groups * self.scan.d_state

    @property
    def bc_offset(self) -> int:
        """First column of the ``B``/``C`` band, ``2*d_inner``. The value and gate
        bands precede it."""
        return 2 * layer_config(self.scan, groups=self.groups).d_inner

    @property
    def params_offset(self) -> int:
        """First column of the parameter band. Last of the four."""
        return self.bc_offset + self.bc_width

    @property
    def proj_width(self) -> int:
        """Fused-projection width: the four bands, padded up to :data:`PROJ_ALIGN`.

        The row pitch of a band is the whole projection width, so the width has to
        clear :data:`slinoss._guard.PROJ_ALIGN` even though the sum of the bands need
        not: ``10*H`` is a multiple of it only for ``H`` a multiple of 8. Padding the
        projection costs a few columns of its GEMM and keeps every ``H`` reachable,
        and a band whose pitch misses the sector is a refusal, not a slow path.
        """
        total = self.params_offset + self.params_width
        return -(-total // PROJ_ALIGN) * PROJ_ALIGN

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.scan.bsz * self.scan.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.scan.bsz} T={self.scan.seq} H={self.scan.heads} "
            f"3N={self.scan.d_state} G={self.groups} W={self.proj_width}"
        )


PREP_SHAPES: Final[tuple[PrepShape, ...]] = tuple(
    PrepShape(shape, groups=4 if shape.name == "wide" else 1) for shape in SHAPES
)
"""The standard frontier sizes, one per entry of :data:`SHAPES` and the same names.

``G`` is 1 at every name but ``wide``, which is the config default and therefore
the headline case. ``wide`` takes ``G = 4`` against ``H = 12``, so three heads share
a pair: neither ``G = 1`` nor ``G = H`` exercises the general ``h // (H//G)``
address, and the ``B``/``C`` band's share of the projection is a function of ``G``.
"""


def prep_shape_by_name(name: str) -> PrepShape:
    """Look up a standard frontier shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`PREP_SHAPES`.
    """
    for shape in PREP_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no prep shape {name!r}; have {[s.name for s in PREP_SHAPES]}")


class PrepInputs(NamedTuple):
    """Parameter-frontier inputs at one shape.

    ``params`` and ``bc`` are detached slices of ``proj``, so each is a leaf with
    the row pitch the fused projection gives it. Slicing a leaf instead would put
    a pullback into a zeroed ``(B,T,W)`` buffer inside the backward measurement,
    which is the projection's cost and not the frontier's.

    Attributes:
        proj: ``(B,T,W)`` projection output, contiguous. Held only to keep the two
            slices alive and to name the pitch; not differentiated.
        params: ``(B,T,10H)`` pitched slice, activation dtype.
        bc: ``(B,T,2*G*3N)`` pitched slice, same dtype.
        param_bias: ``(H,10)``, float32.
        dtrans: ``(B,H,T,4)`` float32 cotangent seed.
        dK: ``(B,H,T,2,4)`` float32 cotangent seed.
        dB: ``(B,G,T,3N)`` cotangent seed, activation dtype.
        dC: ``(B,G,T,3N)`` cotangent seed, activation dtype.
    """

    proj: Tensor
    params: Tensor
    bc: Tensor
    param_bias: Tensor
    dtrans: Tensor
    dK: Tensor
    dB: Tensor
    dC: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The three tensors gradients are taken with respect to."""
        return (self.params, self.bc, self.param_bias)

    @property
    def cotangents(self) -> tuple[Tensor, ...]:
        """The four output-gradient seeds, in output order."""
        return (self.dtrans, self.dK, self.dB, self.dC)


def make_prep_inputs(
    shape: PrepShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = True,
    seed: int = 0,
) -> PrepInputs:
    """Build parameter-frontier inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype of the projection. ``param_bias`` is float32 regardless, as
            I4 requires.
        requires_grad: Whether the three differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scan = shape.scan
    lead = (scan.bsz, scan.heads, scan.seq)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    with torch.no_grad():
        proj = randn(scan.bsz, scan.seq, shape.proj_width)
    # Each band by offset and width, not to the end: the padding that squares the
    # projection width with the alignment sits past the last band.
    bc_columns = slice(shape.bc_offset, shape.bc_offset + shape.bc_width)
    param_columns = slice(shape.params_offset, shape.params_offset + shape.params_width)
    return PrepInputs(
        proj=proj,
        params=proj[..., param_columns].detach().requires_grad_(requires_grad),
        bc=proj[..., bc_columns].detach().requires_grad_(requires_grad),
        param_bias=torch.zeros(
            scan.heads, PARAM_COLS, dtype=torch.float32, device=device
        ).requires_grad_(requires_grad),
        dtrans=randn(*lead, 4, dt=torch.float32),
        dK=randn(*lead, 2, 4, dt=torch.float32),
        dB=randn(scan.bsz, shape.groups, scan.seq, scan.d_state),
        dC=randn(scan.bsz, shape.groups, scan.seq, scan.d_state),
    )


def prep_forward_only(
    inputs: PrepInputs,
    shape: PrepShape,
    *,
    backend: str | None = None,
    prefix: str = "prep",
) -> Callable[[], None]:
    """A callable that runs the frontier forward under ``no_grad``.

    Args:
        inputs: Frontier inputs.
        shape: The problem size, for ``H`` and ``3N``.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad(), region(f"{prefix}.forward"):
            scanprep(
                inputs.params,
                inputs.bc,
                inputs.param_bias,
                heads=shape.scan.heads,
                state_dim=shape.scan.d_state,
                w_max=W_MAX,
                backend=backend,
            )

    return run


def prep_step(
    inputs: PrepInputs,
    shape: PrepShape,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "prep",
) -> Callable[[], None]:
    """A callable that runs the frontier forward and backward.

    Args:
        inputs: Frontier inputs. The three differentiable ones must require grad.
        shape: The problem size, for ``H`` and ``3N``.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all three.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("prep step needs at least one input requiring grad")

    def run() -> None:
        with region(f"{prefix}.forward"):
            out = scanprep(
                inputs.params,
                inputs.bc,
                inputs.param_bias,
                heads=shape.scan.heads,
                state_dim=shape.scan.d_state,
                w_max=W_MAX,
                backend=backend,
            )
        with region(f"{prefix}.backward"):
            torch.autograd.grad(tuple(out), targets, inputs.cotangents)

    return run


@dataclass(frozen=True)
class BlockShape:
    """One benchmarked block size.

    Two kernels, so two widths: the fused residual add and norm run on the
    residual stream and the activation on the FFN hidden. Both are read off
    :func:`layer_config`.

    Attributes:
        scan: The scan of the block this measures.
    """

    scan: OpShape

    @property
    def name(self) -> str:
        """Shape name. The scan's, because it is the same layer."""
        return self.scan.name

    @property
    def width(self) -> int:
        """Residual-stream width, ``d_model``."""
        return layer_config(self.scan).d_model

    @property
    def hidden(self) -> int:
        """FFN hidden width, ``d_ffn``. The activation's row length."""
        return layer_config(self.scan).d_ffn

    @property
    def eps(self) -> float:
        """Norm epsilon."""
        return layer_config(self.scan).norm_eps

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.scan.bsz * self.scan.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.scan.bsz} T={self.scan.seq} "
            f"d_model={self.width} d_ffn={self.hidden}"
        )


BLOCK_SHAPES: Final[tuple[BlockShape, ...]] = tuple(BlockShape(s) for s in SHAPES)
"""The standard block sizes, one per entry of :data:`SHAPES` and the same names."""


def block_shape_by_name(name: str) -> BlockShape:
    """Look up a standard block shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`BLOCK_SHAPES`.
    """
    for shape in BLOCK_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no block shape {name!r}; have {[s.name for s in BLOCK_SHAPES]}")


class BlockInputs(NamedTuple):
    """Block norm and activation inputs at one shape.

    The incoming residual, the norm weight, and the residual output's cotangent
    are float32 and the rest are the activation dtype: that is every block of a
    stack but the first, where the stream has already been widened once and is
    never narrowed again.

    Attributes:
        x: ``(B,T,d_model)`` branch output.
        residual: ``(B,T,d_model)`` incoming stream, float32.
        weight: ``(d_model,)`` norm weight, float32. Shared by the fused norm and
            the plain one: at ``d_model`` elements the parameter is a kilobyte, so
            which module owns it is not a measurement.
        gate: ``(B,T,d_ffn)`` activation gate.
        up: ``(B,T,d_ffn)`` activation value.
        dnormed: ``(B,T,d_model)`` cotangent seed of the normed output.
        dresidual: ``(B,T,d_model)`` float32 cotangent seed of the stream output.
        dout: ``(B,T,d_ffn)`` cotangent seed of the activation output.
        prehead: ``(B,T,d_model)`` input of the plain norm, its own buffer and not
            a second use of ``x``: a second read of ``x`` would come out of L2 and
            understate the plain norm's DRAM traffic.
        dprehead: ``(B,T,d_model)`` cotangent seed of the plain norm's output.
    """

    x: Tensor
    residual: Tensor
    weight: Tensor
    gate: Tensor
    up: Tensor
    dnormed: Tensor
    dresidual: Tensor
    dout: Tensor
    prehead: Tensor
    dprehead: Tensor

    @property
    def fused(self) -> tuple[Tensor, ...]:
        """Operands of the fused norm and the activation, in output order."""
        return (self.x, self.residual, self.weight, self.gate, self.up)

    @property
    def plain(self) -> tuple[Tensor, ...]:
        """Operands of the plain norm."""
        return (self.prehead, self.weight)

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """Every tensor gradients are taken with respect to, across both arms.

        ``weight`` belongs to both and appears once.
        """
        return (*self.fused, self.prehead)


def make_block_inputs(
    shape: BlockShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = True,
    seed: int = 0,
) -> BlockInputs:
    """Build block inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype of the branch output and the activation operands.
        requires_grad: Whether the six differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    stream = (shape.scan.bsz, shape.scan.seq, shape.width)
    ffn = (shape.scan.bsz, shape.scan.seq, shape.hidden)

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    # The plain norm's two tensors are drawn last, so every tensor the fused arms
    # read holds the values it held before that arm existed and a figure taken at
    # one seed stays comparable across this addition.
    return BlockInputs(
        x=randn(*stream).requires_grad_(requires_grad),
        residual=randn(*stream, dt=torch.float32).requires_grad_(requires_grad),
        weight=randn(shape.width, dt=torch.float32).requires_grad_(requires_grad),
        gate=randn(*ffn).requires_grad_(requires_grad),
        up=randn(*ffn).requires_grad_(requires_grad),
        dnormed=randn(*stream),
        dresidual=randn(*stream, dt=torch.float32),
        dout=randn(*ffn),
        prehead=randn(*stream).requires_grad_(requires_grad),
        dprehead=randn(*stream),
    )


def block_forward_only(
    inputs: BlockInputs,
    shape: BlockShape,
    *,
    backend: str | None = None,
    prefix: str = "block",
) -> Callable[[], None]:
    """A callable that runs all three block kernels forward under ``no_grad``.

    The plain norm records under ``<prefix>.rmsnorm.forward`` and reads its own
    input, so the fused arms' bucket holds what it held before that arm existed and
    the aggregate grows by a separately readable row.

    Args:
        inputs: Block inputs.
        shape: The problem size, for the norm epsilon.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad():
            with region(f"{prefix}.forward"):
                rmsnorm_residual(
                    inputs.x,
                    inputs.residual,
                    inputs.weight,
                    eps=shape.eps,
                    backend=backend,
                )
                swiglu(inputs.gate, inputs.up, backend=backend)
            with region(f"{prefix}.rmsnorm.forward"):
                rmsnorm(inputs.prehead, inputs.weight, eps=shape.eps, backend=backend)

    return run


def _named(targets: Sequence[Tensor], operands: Sequence[Tensor]) -> tuple[Tensor, ...]:
    """The operands of one arm that ``targets`` names, in arm order.

    Membership by identity: ``in`` on a tensor compares elementwise and two
    distinct operands of one shape would test equal.

    Args:
        targets: Tensors the step differentiates.
        operands: One arm's operands.

    Returns:
        The intersection, empty if the arm was not named.
    """
    return tuple(t for t in operands if any(t is named for named in targets))


def block_step(
    inputs: BlockInputs,
    shape: BlockShape,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "block",
) -> Callable[[], None]:
    """A callable that runs all three block kernels forward and backward.

    Both norm outputs carry a cotangent: the normed one feeds the projection and
    the stream one feeds the next block, so a stack seeds both and a measurement
    that dropped either would skip an arc of the pullback.

    The plain norm is a second arm, under ``<prefix>.rmsnorm.*`` and a second
    :func:`torch.autograd.grad` call over its own operands, so the fused arms'
    buckets hold what they held before it existed. The aggregate step grows by it.

    Args:
        inputs: Block inputs. The six differentiable ones must require grad.
        shape: The problem size, for the norm epsilon.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all six. An arm
            no named tensor belongs to is not run, so a subset naming one arm
            measures that arm alone.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("block step needs at least one input requiring grad")
    fused = _named(targets, inputs.fused)
    plain = _named(targets, inputs.plain)

    def run() -> None:
        if fused:
            with region(f"{prefix}.forward"):
                normed, stream = rmsnorm_residual(
                    inputs.x,
                    inputs.residual,
                    inputs.weight,
                    eps=shape.eps,
                    backend=backend,
                )
                out = swiglu(inputs.gate, inputs.up, backend=backend)
            with region(f"{prefix}.backward"):
                torch.autograd.grad(
                    (normed, stream, out),
                    fused,
                    (inputs.dnormed, inputs.dresidual, inputs.dout),
                )
        if plain:
            with region(f"{prefix}.rmsnorm.forward"):
                head = rmsnorm(
                    inputs.prehead, inputs.weight, eps=shape.eps, backend=backend
                )
            with region(f"{prefix}.rmsnorm.backward"):
                torch.autograd.grad(head, plain, inputs.dprehead)

    return run


@dataclass(frozen=True)
class MixerShape:
    """One benchmarked fused-tail size.

    Attributes:
        prep: The frontier reading the other bands of the projection this tail's
            gate is one band of. ``B``, ``H``, ``T``, ``P`` and the projection width
            come off it, so the gate's pitch is the width the fused projection has
            at this shape rather than a second number that can disagree with it.
    """

    prep: PrepShape

    @property
    def scan(self) -> OpShape:
        """The scan whose output this tail consumes."""
        return self.prep.scan

    @property
    def name(self) -> str:
        """Shape name. The scan's, because it is the same layer."""
        return self.scan.name

    @property
    def width(self) -> int:
        """Band width, ``d_inner``, which is ``H*P``."""
        return layer_config(self.scan).d_inner

    @property
    def gate_offset(self) -> int:
        """First column of the gate band. The value band precedes it and is the
        same width, so the gate starts at ``d_inner``."""
        return self.width

    @property
    def proj_width(self) -> int:
        """Row pitch of the gate band: the whole fused-projection width."""
        return self.prep.proj_width

    @property
    def eps(self) -> float:
        """Norm epsilon."""
        return layer_config(self.scan).norm_eps

    @property
    def token_count(self) -> Count:
        """Tokens per call, ``B*T``."""
        return Count(self.scan.bsz * self.scan.seq)

    def describe(self) -> str:
        """One line for a report note."""
        return (
            f"{self.name}: B={self.scan.bsz} H={self.scan.heads} T={self.scan.seq} "
            f"P={self.scan.rows} d_inner={self.width} W={self.proj_width}"
        )


MIXER_SHAPES: Final[tuple[MixerShape, ...]] = tuple(
    MixerShape(prep) for prep in PREP_SHAPES
)
"""The standard fused-tail sizes, one per entry of :data:`SHAPES` and the same names.

The pitch is the frontier's projection width at the same name, so a tail figure and
a frontier figure at one name were taken against one projection rather than against
two independently chosen widths."""


def mixer_shape_by_name(name: str) -> MixerShape:
    """Look up a standard fused-tail shape.

    Args:
        name: Shape name.

    Returns:
        The shape.

    Raises:
        KeyError: If the name is not one of :data:`MIXER_SHAPES`.
    """
    for shape in MIXER_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"no mixer shape {name!r}; have {[s.name for s in MIXER_SHAPES]}")


class MixerInputs(NamedTuple):
    """Fused mixer-tail inputs at one shape.

    ``y`` and ``u`` are head-major and contiguous, which is what the scan and the
    conv write. ``gate`` and the output cotangent are token-major column bands of
    two wider buffers, which is what the projections around the tail hand over, so
    both carry a row pitch above their width and the kernels reach them through the
    dynamic layout they were written for. A contiguous fixture would measure a
    layout the mixer never produces.

    ``gate`` is a detached band rather than a slice of a live leaf: slicing a leaf
    would put a pullback into a zeroed ``(B,T,W)`` buffer inside the backward
    measurement, which is the projection's cost and not the tail's.

    Attributes:
        proj: ``(B,T,W)`` input projection, contiguous. Held to keep ``gate`` alive
            and to name the pitch; not differentiated.
        y: ``(B,H,T,P)`` scan output.
        u: ``(B,H,T,P)`` scan input, source of the skip term.
        gate: ``(B,T,H*P)`` pitched band of ``proj``.
        d_skip: ``(H,P)`` skip scale, parameter dtype.
        weight: ``(H,P)`` norm scale, parameter dtype.
        dproj: ``(B,T,W)`` buffer the cotangent seed is a band of, contiguous. Held
            for the same two reasons as ``proj``.
        dout: ``(B,T,H*P)`` pitched band of ``dproj``, the output-gradient seed,
            preallocated so the backward measurement contains no allocation of its
            own.
    """

    proj: Tensor
    y: Tensor
    u: Tensor
    gate: Tensor
    d_skip: Tensor
    weight: Tensor
    dproj: Tensor
    dout: Tensor

    @property
    def differentiable(self) -> tuple[Tensor, ...]:
        """The five tensors gradients are taken with respect to."""
        return (self.y, self.u, self.gate, self.d_skip, self.weight)

    @property
    def bands(self) -> tuple[Tensor, ...]:
        """The two operands the kernels index through a dynamic layout."""
        return (self.gate, self.dout)


def make_mixer_inputs(
    shape: MixerShape,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    param_dtype: torch.dtype | None = None,
    requires_grad: bool = True,
    seed: int = 0,
) -> MixerInputs:
    """Build fused mixer-tail inputs at one shape.

    Args:
        shape: The problem size.
        device: Where to allocate.
        dtype: Dtype of ``y``, ``u``, the gate band and the cotangent seed.
        param_dtype: Dtype of ``d_skip`` and ``weight``, or None for ``dtype``.
            Operand width and parameter width are independent in the kernel, so
            float32 parameters against low-precision activations is one call and a
            benchmark that could not express it would leave that call unmeasured.
        requires_grad: Whether the five differentiable inputs carry gradients.
        seed: Generator seed, so two runs benchmark the same numbers.

    Returns:
        The inputs.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    scan = shape.scan
    lead = (scan.bsz, scan.heads, scan.seq)
    param = dtype if param_dtype is None else param_dtype

    def randn(*size: int, dt: torch.dtype = dtype) -> Tensor:
        return torch.randn(*size, dtype=dt, device=device, generator=gen)

    with torch.no_grad():
        proj = randn(scan.bsz, scan.seq, shape.proj_width)
        dproj = randn(scan.bsz, scan.seq, shape.proj_width)
    # By offset and width, not to the end: the value band precedes the gate and the
    # bands the frontier reads follow it.
    band = slice(shape.gate_offset, shape.gate_offset + shape.width)
    return MixerInputs(
        proj=proj,
        y=randn(*lead, scan.rows).requires_grad_(requires_grad),
        u=randn(*lead, scan.rows).requires_grad_(requires_grad),
        gate=proj[..., band].detach().requires_grad_(requires_grad),
        d_skip=randn(scan.heads, scan.rows, dt=param).requires_grad_(requires_grad),
        weight=randn(scan.heads, scan.rows, dt=param).requires_grad_(requires_grad),
        dproj=dproj,
        dout=dproj[..., band].detach(),
    )


def mixer_forward_only(
    inputs: MixerInputs,
    shape: MixerShape,
    *,
    backend: str | None = None,
    prefix: str = "mixer",
) -> Callable[[], None]:
    """A callable that runs the fused tail forward under ``no_grad``.

    Args:
        inputs: Fused-tail inputs.
        shape: The problem size, for the norm epsilon.
        backend: Backend name, or None for the fastest registered one.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.
    """

    def run() -> None:
        with torch.no_grad(), region(f"{prefix}.forward"):
            mixer_tail(
                inputs.y,
                inputs.u,
                inputs.gate,
                inputs.d_skip,
                inputs.weight,
                eps=shape.eps,
                backend=backend,
            )

    return run


def mixer_step(
    inputs: MixerInputs,
    shape: MixerShape,
    *,
    backend: str | None = None,
    wrt: Sequence[Tensor] | None = None,
    prefix: str = "mixer",
) -> Callable[[], None]:
    """A callable that runs the fused tail forward and backward.

    Args:
        inputs: Fused-tail inputs. The five differentiable ones must require grad.
        shape: The problem size, for the norm epsilon.
        backend: Backend name, or None for the fastest registered one.
        wrt: Tensors to differentiate with respect to. Defaults to all five.
        prefix: Region label prefix. See :func:`forward_only`.

    Returns:
        The callable, timed by :func:`slinoss.perf.timing.measure`.

    Raises:
        ValueError: If no input requires grad, which would time a forward and
            call it a step.
    """
    targets = tuple(inputs.differentiable if wrt is None else wrt)
    if not any(t.requires_grad for t in targets):
        raise ValueError("mixer step needs at least one input requiring grad")

    def run() -> None:
        with region(f"{prefix}.forward"):
            out = mixer_tail(
                inputs.y,
                inputs.u,
                inputs.gate,
                inputs.d_skip,
                inputs.weight,
                eps=shape.eps,
                backend=backend,
            )
        with region(f"{prefix}.backward"):
            torch.autograd.grad(out, targets, inputs.dout)

    return run
