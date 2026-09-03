"""The sequence mixer and the column bands of its fused input projection.

One GEMM produces every per-token operand of a mixer step: the value the
convolution filters, the gate, the two state vectors, and the four scan parameters
per head. Each consumer reads its own column band of that one output at the
projection's pitch. Nothing is copied out of it, and nothing gets a projection of
its own.

The band geometry is here rather than in each consumer's guard because it is one
statement about one buffer. A consumer checks that what it received is a legal
band; this module decides where the bands are.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

from slinoss._guard import PROJ_ALIGN
from slinoss._linear import linear_backward
from slinoss._precision import LOW_PRECISION_DTYPES, cast_opt, cast_to
from slinoss.config import SLinOSSConfig
from slinoss.ops.conv import backends as conv_dispatch
from slinoss.ops.mixer import backends as tail_dispatch
from slinoss.ops.scanprep import LS_COLUMN, LS_MAX_MAG, PARAM_COLS, ROTVEC_COLUMNS
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd.reference import ScanPrologue
from slinoss.state import MixerState, oscillator_basis

__all__ = [
    "FALLBACK_SPAN",
    "ProjectionLayout",
    "SLinOSSMixer",
    "fibonacci_axes",
    "head_band",
    "head_grid",
    "head_lattice",
]


def _align_up(width: int) -> int:
    """``width`` rounded up to a multiple of :data:`slinoss._guard.PROJ_ALIGN`."""
    return -(-width // PROJ_ALIGN) * PROJ_ALIGN


@dataclass(frozen=True)
class ProjectionLayout:
    """Where each consumer's band sits in the fused projection.

    Band order is value, gate, ``B``, ``C``, parameters, then the columns the
    padding adds. The order is what keeps every offset aligned without padding
    between bands: the three activation widths are multiples of
    :data:`slinoss._guard.PROJ_ALIGN` already, ``d_inner`` because it is a whole
    number of heads of a width that is a multiple of 16, and each state band
    because ``3N`` is a multiple of 48. Only the parameter band is a free width, so
    it goes last and the padding lands past every band.

    Attributes:
        d_inner: Width of the value band, and of the gate band.
        heads: ``H``. The parameter band is :data:`PARAM_COLS` columns per head.
        groups: ``G``. Each state band is ``groups * state_dim`` columns.
        state_dim: ``3N``.
        width: Projected width. Every band steps by this from one token to the
            next.
    """

    d_inner: int
    heads: int
    groups: int
    state_dim: int
    width: int

    @classmethod
    def from_config(cls, cfg: SLinOSSConfig) -> ProjectionLayout:
        """The layout a configuration implies.

        Args:
            cfg: The mixer's configuration.

        Returns:
            The layout.
        """
        bands = 2 * cfg.d_inner + 2 * cfg.n_groups * cfg.d_state
        return cls(
            d_inner=cfg.d_inner,
            heads=cfg.n_heads,
            groups=cfg.n_groups,
            state_dim=cfg.d_state,
            width=bands + _align_up(PARAM_COLS * cfg.n_heads),
        )

    def __post_init__(self) -> None:
        for name, offset in (
            ("gate", self.gate_off),
            ("B", self.b_off),
            ("C", self.c_off),
            ("params", self.params_off),
            ("width", self.width),
        ):
            if offset % PROJ_ALIGN != 0:
                raise ValueError(
                    f"{name} lands on column {offset}, which is not a multiple of "
                    f"{PROJ_ALIGN}: a band row would start mid-sector"
                )
        if self.width < self.params_off + PARAM_COLS * self.heads:
            raise ValueError(
                f"width {self.width} is below the {self.params_off} columns of "
                f"bands plus {PARAM_COLS * self.heads} of parameters"
            )

    @property
    def gate_off(self) -> int:
        """First column of the gate band."""
        return self.d_inner

    @property
    def b_off(self) -> int:
        """First column of the ``B`` band."""
        return 2 * self.d_inner

    @property
    def c_off(self) -> int:
        """First column of the ``C`` band."""
        return self.b_off + self.groups * self.state_dim

    @property
    def params_off(self) -> int:
        """First column of the parameter band."""
        return self.c_off + self.groups * self.state_dim

    @property
    def pad_width(self) -> int:
        """Columns past the last band. They belong to no consumer.

        A cotangent buffer must still zero them: the projection's own pullback
        reads the whole width, so a column no band wrote is a column of garbage
        rather than a column of zero.
        """
        return self.width - self.params_off - PARAM_COLS * self.heads

    def value(self, proj: Tensor) -> Tensor:
        """The value band, ``(B,T,d_inner)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``.
        """
        return proj[..., : self.d_inner]

    def gate(self, proj: Tensor) -> Tensor:
        """The gate band, ``(B,T,d_inner)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``.
        """
        return proj[..., self.gate_off : self.b_off]

    def b(self, proj: Tensor) -> Tensor:
        """The ``B`` band, ``(B,G,T,3N)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``. The group axis strides by ``3N`` and the token axis
            by ``width``, so the group axis strides less than the axis before it.
        """
        return self._vectors(proj, self.b_off)

    def c(self, proj: Tensor) -> Tensor:
        """The ``C`` band, ``(B,G,T,3N)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``, laid out like :meth:`b`.
        """
        return self._vectors(proj, self.c_off)

    def keys(self, proj: Tensor) -> Tensor:
        """The ``B`` and ``C`` bands as one band, ``(B,T,2*G*3N)``, pitched.

        They are adjacent by :attr:`c_off`, so the pair is one slice and the key
        convolution is one call over it rather than two over the halves.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``.
        """
        return proj[..., self.b_off : self.params_off]

    def params(self, proj: Tensor) -> Tensor:
        """The parameter band, ``(B,T,H*PARAM_COLS)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``. The padding past it is excluded, so the width is
            the one scanprep unflattens by head.
        """
        stop = self.params_off + PARAM_COLS * self.heads
        return proj[..., self.params_off : stop]

    def pad(self, proj: Tensor) -> Tensor:
        """The columns no band owns, ``(B,T,pad_width)``, pitched.

        Args:
            proj: ``(B,T,width)``.

        Returns:
            A view of ``proj``, empty when :attr:`pad_width` is zero. See
            :attr:`pad_width` for why a cotangent buffer zeroes it.
        """
        return proj[..., self.params_off + PARAM_COLS * self.heads :]

    def key_b(self, keys: Tensor) -> Tensor:
        """``B`` inside a keys buffer, ``(B,G,T,3N)``.

        Args:
            keys: ``(B,T,2*G*3N)``, as :meth:`keys` cuts it or as the key
                convolution returns it.

        Returns:
            A view of ``keys``, laid out like :meth:`b`.
        """
        return self._vectors(keys, 0)

    def key_c(self, keys: Tensor) -> Tensor:
        """``C`` inside a keys buffer, ``(B,G,T,3N)``.

        Args:
            keys: ``(B,T,2*G*3N)``.

        Returns:
            A view of ``keys``, laid out like :meth:`c`.
        """
        return self._vectors(keys, self.groups * self.state_dim)

    def _vectors(self, proj: Tensor, offset: int) -> Tensor:
        """One state band as the scan reads it.

        ``unflatten`` of a unit-stride trailing axis and ``permute`` are both views,
        so the group-major shape costs no copy.
        """
        band = proj[..., offset : offset + self.groups * self.state_dim]
        return band.unflatten(-1, (self.groups, self.state_dim)).permute(0, 2, 1, 3)


FALLBACK_SPAN: float = 4096.0
"""Slow end of the lattice when :attr:`SLinOSSConfig.seq_len` is None.

A stack that does not state the sequence it trains on gets the widest band the
harnesses in this tree ever ask for. Nothing derives this: it is the absence of an
answer, and every configuration that carries ``seq_len`` reads
:func:`head_band` instead.
"""

_MAP_HEADROOM = 0.5
"""Fraction of the chart scale the fastest grid rung asks for.

Half preserves a four-token fastest period at the default scale and keeps the
initialized bank well inside both the canonical SO(3) ball and the chart's
``2*w_max`` asymptote.
"""


def head_band(config: SLinOSSConfig) -> tuple[float, float]:
    """Band in tokens the lattice spans, both ends derived, ``(fast, slow)``.

    The fast end is the period at :data:`_MAP_HEADROOM` of the chart scale,
    ``2*pi / (_MAP_HEADROOM * w_max)``. This keeps initialization away from both
    the canonical half-turn boundary and the chart's asymptote; training still has
    the rest of the chart available. A two-token period is the alternating sign no
    sampled rotation resolves.

    The slow end is one turn across the trained sequence. Past it a head is a
    constant: it neither completes a turn, so it carries no phase, nor decays, so it
    separates no timescale. The previous constant was four times the longest
    sequence any harness ran, which put most of the bank past the slow end of every
    shorter task -- at 8 heads and 32 tokens, six of the eight rungs.

    A horizon and a period read the same band. The per-token amplitude factor is
    ``exp(2*ls)``, so a horizon of ``h`` tokens is ``ls = -0.5/h``; conjugation
    turns by ``|w|`` per token, ``quat_exp`` building the half angle and the
    conjugation doubling it back, so a period of ``p`` tokens is ``|w| = 2*pi/p``.
    One band, two grids over it, and not one grid: turns per amplitude lifetime is
    the ratio ``h/p``, so a lattice over both sweeps it across the square of the
    band while a single grid pins it to the diagonal ``h == p``. The corners are
    what the diagonal has no room for -- a head that decays without turning at all,
    which is the scalar transition, and a narrowband resonator that turns many times
    inside its own memory.

    Args:
        config: Supplies ``w_max`` and ``seq_len``.

    Returns:
        ``(fast, slow)`` in tokens, with ``slow`` at least ``fast``. A ``seq_len``
        under the fast end collapses the band to one rung rather than inverting it.
    """
    fast = 2.0 * math.pi / (_MAP_HEADROOM * config.w_max)
    slow = FALLBACK_SPAN if config.seq_len is None else float(config.seq_len)
    return (fast, max(fast, slow))


_MAX_MAP_FRACTION = 0.9
"""Cap on the initialized target as a fraction of either map's named scale.

The log-scale inverse diverges at one; the rotation inverse does not diverge until
two under the finite-half-turn chart. One common cap keeps both initial rows away
from a saturated coordinate. It is inactive at the defaults, where the shortest
period and horizon each ask for exactly one half of their named scale.
"""


def head_grid(bounds: tuple[float, float], heads: int) -> Tensor:
    """One value per head, log-spaced over ``bounds``, endpoints included.

    Log-spaced because the quantity is a scale: a linear grid over a ratio of 1024
    puts most heads at the high end.

    Both endpoints come out exactly, which ``logspace`` does not give: it exponentiates
    a base-10 logarithm, and neither the log nor the power is exact, so the lowest rung
    lands 1e-7 above ``low``. The ratio is raised directly instead, and its zeroth and
    first powers are exact by IEEE, so the range means what it says and the inverse
    through ``logit`` is the one the bound was derived at.

    Args:
        bounds: ``(low, high)``, both positive.
        heads: ``H``. One head takes ``low``.

    Returns:
        ``(H,)``, float32, on the CPU.
    """
    low, high = bounds
    if heads < 2:
        return torch.full((heads,), low, dtype=torch.float32)
    ramp = torch.arange(heads, dtype=torch.float64) / (heads - 1)
    return (low * (high / low) ** ramp).float()


def head_lattice(heads: int, band: tuple[float, float]) -> tuple[Tensor, Tensor]:
    """One ``(horizon, period)`` pair per head, boustrophedon over the two grids.

    ``isqrt(H)`` horizons against ``ceil(H / isqrt(H))`` periods, as square as ``H``
    allows, so neither axis collapses to one rung while the other takes them all.
    The pairs run period-major with the horizon order reversed on every second
    period, so consecutive heads differ in one coordinate: the head axis is the axis
    a grouped operand shares along, and a snake keeps neighbours in it neighbours
    here.

    ``H`` need not factor. The lattice holds ``n_h * n_p >= H`` points and the first
    ``H`` are taken, so at most ``n_h - 1`` points of the last period go unused and
    no pair repeats.

    Args:
        heads: ``H``, at least one.
        band: ``(fast, slow)`` in tokens, from :func:`head_band`. Both grids span
            it.

    Returns:
        ``(horizon, period)``, each ``(H,)`` float32 on the CPU.
    """
    n_h = max(1, math.isqrt(heads))
    n_p = -(-heads // n_h)
    rows = head_grid(band, n_h).expand(n_p, n_h).clone()
    rows[1::2] = rows[1::2].flip(-1)
    period = head_grid(band, n_p).repeat_interleave(n_h)
    return rows.reshape(-1)[:heads], period[:heads]


def fibonacci_axes(heads: int) -> Tensor:
    """One unit rotation axis per head, a spherical Fibonacci set.

    The height marches uniformly through ``(-1, 1)`` and the azimuth advances by the
    golden angle. That is the lowest-discrepancy sphere set of a given size a closed
    form gives: no two axes coincide, the set has no accumulation direction, and it
    is a function of ``H`` alone. A pseudo-random draw needs a seed to be
    reproducible at all and still clusters at the head counts this operator runs.

    Args:
        heads: ``H``, at least one.

    Returns:
        ``(H,3)`` float32 on the CPU, rows of unit norm.
    """
    index = torch.arange(heads, dtype=torch.float64) + 0.5
    height = 1.0 - 2.0 * index / heads
    radius = (1.0 - height * height).clamp(min=0.0).sqrt()
    azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
    plane = torch.stack((azimuth.cos(), azimuth.sin()), dim=-1) * radius[:, None]
    return torch.cat((plane, height[:, None]), dim=-1).float()


def _param_bias_init(config: SLinOSSConfig) -> Tensor:
    """Rows of ``param_bias`` at initialization, ``(H, PARAM_COLS)``, float32, CPU.

    Both bounded maps are inverted, so the lattice states what the recurrence does
    at step zero rather than what an unbounded parameterization would do. Each row
    reads its own axis of :func:`head_lattice`:

    - ``bounded_logscale(raw) = -LS_MAX_MAG*sigmoid(raw)``, so a horizon of ``h``
      tokens is ``ls = -0.5/h`` at ``raw = logit(0.5/(h*LS_MAX_MAG))``.
    - ``bounded_rotvec(raw, w_max) = raw*w_max*rsqrt(1 + |raw|^2/4)`` gives
      ``|w| = s*w_max`` at ``|raw| = s*rsqrt(1 - s*s/4)``, so a period of ``p``
      tokens is ``|w| = 2*pi/p`` at that radius along the head's axis. Unlike the
      old unit-asymptote chart, ``s = 1`` is finite.

    The taps take no row at all: they are the first-order-hold moments of the
    transition these rows set, so the lattice reaches them already.

    Args:
        config: Supplies ``H`` and ``w_max``.

    Returns:
        ``(H, PARAM_COLS)``, float32.
    """
    heads = config.n_heads
    rows = torch.zeros(heads, PARAM_COLS, dtype=torch.float32)
    horizon, period = head_lattice(heads, head_band(config))
    decay = (0.5 / (horizon * LS_MAX_MAG)).clamp(max=_MAX_MAP_FRACTION)
    rows[:, LS_COLUMN] = torch.logit(decay)
    angle = (2.0 * math.pi / (period * config.w_max)).clamp(max=_MAX_MAP_FRACTION)
    radius = angle * torch.rsqrt(1.0 - 0.25 * angle * angle)
    rows[:, ROTVEC_COLUMNS] = radius[:, None] * fibonacci_axes(heads)
    return rows


class _Backends(NamedTuple):
    """Backend name each stage resolved to, in call order.

    Attributes:
        conv: Key in :mod:`slinoss.ops.conv.backends`.
        prep: Key in :mod:`slinoss.ops.scanprep.backends`.
        scan: Key in :mod:`slinoss.ops.so3ssd.backends`.
        tail: Key in :mod:`slinoss.ops.mixer.backends`.
    """

    conv: str
    prep: str
    scan: str
    tail: str


def _resolve(proj: Tensor) -> _Backends:
    """Resolve every stage against the projection's device and dtype.

    Against the projection rather than the input or the parameters, because under
    autocast the activation dtype is neither of those.

    Args:
        proj: The projection output.

    Returns:
        The four names, to be saved and reused by the backward so the two passes
        cannot select differently.
    """
    device, dtype = proj.device.type, proj.dtype
    return _Backends(
        conv=conv_dispatch.resolve(None, device, dtype).name,
        prep=prep_dispatch.resolve(None, device, dtype).name,
        scan=scan_dispatch.resolve(None, device, dtype).name,
        tail=tail_dispatch.resolve(None, device, dtype).name,
    )


_Grads = tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    None,
    None,
    None,
]


class _SLinOSSMixerFunction(torch.autograd.Function):
    """The whole mixer as one autograd node: input projection to output projection.

    One node rather than six is what buys the single gradient buffer. Autograd
    would give each operator its own ``dB``, ``dC``, ``dgate``, ``dparams`` and
    ``dvalue``, then read all five again to accumulate them into the projection's
    cotangent; here each backend stores its band of one ``dproj`` and the
    projection's pullback reads that buffer once.

    Backends are called directly rather than through each operator's public
    :class:`torch.autograd.Function`, because grad is disabled inside
    :meth:`forward` and a nested node would record nothing.

    Saved: ``x``, ``proj``, the convolution output, ``trans``, ``K``, the scan
    output, the tail output, every parameter, and the scan's chunk boundary. Saving
    ``trans`` and ``K`` rather than rematerializing scanprep in the backward is this
    commit's choice; the two trade ``(B,H,T,4)`` plus ``(B,H,T,2,4)`` of float32
    against one extra elementwise pass, and neither has been measured here. The
    chunk boundary is measured: it trades one ``(B,H,C,P,3N)`` float32 buffer per
    layer against the scan's first two launches, which the scan's backward would
    otherwise run again, and ``scripts/perf/profile_prologue.py`` prices both arms.

    No :func:`torch.amp.custom_fwd`. It casts every input to the autocast dtype,
    which would demote ``param_bias``, ``trans``, and ``K`` and break I4.

    Every gradient is produced unconditionally. ``ctx.needs_input_grad`` would
    branch the composition, and every band of ``dproj`` is written by the same
    backends that produce the parameter gradients, so nothing is saved by skipping
    one.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        in_weight: Tensor,
        in_bias: Tensor | None,
        conv_weight: Tensor,
        conv_bias: Tensor | None,
        key_weight: Tensor | None,
        param_bias: Tensor,
        d_skip: Tensor,
        norm_weight: Tensor,
        out_weight: Tensor,
        out_bias: Tensor | None,
        initial_state: Tensor,
        layout: ProjectionLayout,
        config: SLinOSSConfig,
    ) -> Tensor:
        proj = linear(x, in_weight, in_bias)
        picks = _resolve(proj)
        # The taps carry the activation dtype: a kernel backend holds every operand
        # of the convolution to one dtype, and the reference widens either way.
        step = conv_dispatch.get(picks.conv).forward(
            layout.value(proj),
            cast_to(conv_weight, proj.dtype),
            cast_opt(conv_bias, proj.dtype),
            activation=True,
            d_head=config.d_head,
        )
        # No activation on the keys and no bias: a state vector's direction is what
        # the scan contracts against, and a rectifier on its components confines
        # every key to one octant. Token-major, because the scan reads B and C
        # group-major from a flat band either way.
        keys = (
            None
            if key_weight is None
            else conv_dispatch.get(picks.conv)
            .forward(
                layout.keys(proj),
                cast_to(key_weight, proj.dtype),
                None,
                activation=False,
            )
            .y
        )
        b_band = layout.b(proj) if keys is None else layout.key_b(keys)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys)
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj), param_bias, heads=config.n_heads, w_max=config.w_max
        )
        z0 = initial_state.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
        scan = scan_dispatch.get(picks.scan).forward(
            step.y,
            params.trans,
            params.K,
            b_band,
            c_band,
            config.chunk_size,
            z0=z0,
        )
        tail = tail_dispatch.get(picks.tail).forward(
            scan.y, step.y, layout.gate(proj), d_skip, norm_weight, eps=config.norm_eps
        )
        ctx.save_for_backward(
            x,
            proj,
            step.y,
            params.trans,
            params.K,
            scan.y,
            tail,
            in_weight,
            in_bias,
            conv_weight,
            conv_bias,
            param_bias,
            d_skip,
            norm_weight,
            out_weight,
            out_bias,
            key_weight,
            keys,
            z0,
            # Last, so the parameter slices above stay fixed. Three Nones from a
            # backend whose backward rebuilds the boundary instead.
            *((None, None, None) if scan.prologue is None else scan.prologue),
        )
        ctx.layout = layout
        ctx.config = config
        ctx.picks = picks
        return linear(tail, out_weight, out_bias)

    @staticmethod
    def backward(ctx: Any, dout: Tensor) -> _Grads:  # type: ignore[override]
        saved = ctx.saved_tensors
        x, proj, conv_y, trans, K, scan_y, tail = saved[:7]
        in_weight, in_bias, conv_weight, conv_bias, param_bias = saved[7:12]
        d_skip, norm_weight, out_weight, out_bias = saved[12:16]
        key_weight, keys, z0 = saved[16:19]
        zstart, cquat, cscale = saved[19:]
        layout: ProjectionLayout = ctx.layout
        config: SLinOSSConfig = ctx.config
        picks: _Backends = ctx.picks

        out_grads = linear_backward(
            dout, tail, out_weight, has_bias=out_bias is not None
        )
        # One buffer for every band's cotangent, uninitialized: each consumer writes
        # its own band in full. Only the columns no consumer owns are zeroed, and
        # the projection's pullback then reads the whole width.
        dproj = torch.empty(proj.shape, dtype=proj.dtype, device=proj.device)
        layout.pad(dproj).zero_()

        tail_grads = tail_dispatch.get(picks.tail).backward(
            out_grads.dinput,
            scan_y,
            conv_y,
            layout.gate(proj),
            d_skip,
            norm_weight,
            eps=config.norm_eps,
            dgate=layout.gate(dproj),
        )
        # With a key convolution the scan's B and C cotangents belong to its output,
        # not to the projection's band, so they land in their own buffer and the
        # convolution's own pullback carries them the rest of the way. Without one
        # the buffer is dproj itself and the band is written in place.
        dkeys = dproj if keys is None else torch.empty_like(keys)
        b_band = layout.b(proj) if keys is None else layout.key_b(keys)
        c_band = layout.c(proj) if keys is None else layout.key_c(keys)
        db_band = layout.b(dproj) if keys is None else layout.key_b(dkeys)
        dc_band = layout.c(dproj) if keys is None else layout.key_c(dkeys)
        # The tail's du is the skip path's share of the scan's dU, handed over as the
        # scan's addend. The returned dU is the sum, so nothing adds it afterwards.
        scan_grads = scan_dispatch.get(picks.scan).backward(
            tail_grads.dy,
            None,
            None,
            None,
            conv_y,
            trans,
            K,
            b_band,
            c_band,
            config.chunk_size,
            z0=z0,
            dB=db_band,
            dC=dc_band,
            dU_init=tail_grads.du,
            prologue=(None if zstart is None else ScanPrologue(zstart, cquat, cscale)),
        )
        key_dweight: Tensor | None = None
        if key_weight is not None:
            key_dweight = cast_to(
                conv_dispatch.get(picks.conv)
                .backward(
                    dkeys,
                    None,
                    layout.keys(proj),
                    cast_to(key_weight, proj.dtype),
                    None,
                    activation=False,
                    dx=layout.keys(dproj),
                )
                .dweight,
                key_weight.dtype,
            )
        prep_grads = prep_dispatch.get(picks.prep).backward(
            scan_grads.dtrans,
            scan_grads.dK,
            layout.params(proj),
            param_bias,
            heads=config.n_heads,
            w_max=config.w_max,
            dparams=layout.params(dproj),
        )
        conv_grads = conv_dispatch.get(picks.conv).backward(
            scan_grads.dU,
            None,
            layout.value(proj),
            cast_to(conv_weight, proj.dtype),
            cast_opt(conv_bias, proj.dtype),
            activation=True,
            dx=layout.value(dproj),
        )
        in_grads = linear_backward(dproj, x, in_weight, has_bias=in_bias is not None)
        return (
            cast_to(in_grads.dinput, x.dtype),
            cast_to(in_grads.dweight, in_weight.dtype),
            cast_opt(in_grads.dbias, in_weight.dtype),
            cast_to(conv_grads.dweight, conv_weight.dtype),
            cast_opt(conv_grads.dbias, conv_weight.dtype),
            key_dweight,
            prep_grads.dparam_bias,
            tail_grads.dd_skip,
            tail_grads.dweight,
            cast_to(out_grads.dweight, out_weight.dtype),
            cast_opt(out_grads.dbias, out_weight.dtype),
            None,
            None,
            None,
        )


class SLinOSSMixer(nn.Module):
    """Oscillatory state-space sequence mixer.

    One GEMM projects every per-token operand, one autograd node spans the
    projection through the output projection, and one buffer carries every band's
    cotangent back to it. Nothing between the two projections is copied: each
    consumer reads and writes its own column band in place.

    ``x`` is ``(B,T,d_model)`` and the result is ``(B,T,d_model)``, both in the
    activation dtype. The transition embedding, skip gain, and fixed initial state
    are float32 at every low-precision module dtype.

    CUDA. Every operand between the two projections is a column band, and
    :func:`slinoss._guard.check_pitched` holds a band to a device rule, so a CPU
    call raises from the first consumer that checks one.

    :meth:`forward` mixes a whole sequence and threads no state. :meth:`step`
    continues one: same composition, same backends, with the four carries
    :class:`slinoss.state.MixerState` holds read at the front and written at the
    back. It takes no gradient.

    Initialization is principled where the scale sets the recurrence and residual.
    The per-head transition rows form ``transition_embedding``: a direct embedding
    of heads into the operator chart, inverted through both bounded maps onto the
    two-axis lattice :func:`head_lattice` states. It carries the standard no-decay
    marker and its semantic name also places it with embeddings in trainers that use
    the usual name-based parameter groups. Decay of a chart operating point changes
    the represented operator rather than regularizing a weight. ``d_skip`` starts at
    one. The recurrent state starts on a deterministic cyclic basis, because zero is
    a fixed point of every homogeneous rotation. The input projection's parameter and
    forcing-B bands are zeroed, and the output projection is zeroed, so the residual
    branch starts as an exact no-op; all three remain trainable. Other input rows keep
    :meth:`torch.nn.Linear.reset_parameters`, and the convolution taps take the same
    uniform bound over ``d_conv``. No depth scaling.

    Args:
        config: Shape and parameterization contract.
        device: Device for every parameter.
        dtype: Dtype for every parameter except those in
            :data:`CRITICAL_FP32_TENSORS`.
    """

    _version = 2
    """State-dict version introducing the finite chart and transition embedding."""

    #: Tensors a module-wide cast may not demote. ``param_bias`` is read by scanprep
    #: (I4), the initial state is the scan accumulator, and ``d_skip`` is a per-head
    #: scalar whose gradient reduces over ``B*T*P``.
    CRITICAL_FP32_TENSORS: tuple[str, ...] = (
        "param_bias",
        "d_skip",
        "initial_state",
    )

    def __init__(
        self,
        config: SLinOSSConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = ProjectionLayout.from_config(config)
        # out_features is the padded width, so the parameter band's tail is columns
        # the GEMM computes and no band reads. reset_parameters zeroes their rows.
        self.in_proj = nn.Linear(
            config.d_model,
            self.layout.width,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(config.d_inner, config.d_conv, device=device, dtype=dtype)
        )
        self.conv_bias: Tensor | None = (
            nn.Parameter(torch.empty(config.d_inner, device=device, dtype=dtype))
            if config.conv_bias
            else None
        )
        # One weight over both state bands, because they are adjacent columns, and a
        # separate tap row per channel, so B and C are convolved independently.
        self.key_weight: Tensor | None = (
            nn.Parameter(
                torch.empty(
                    2 * config.n_groups * config.d_state,
                    config.d_conv,
                    device=device,
                    dtype=dtype,
                )
            )
            if config.key_conv
            else None
        )
        init_device = device if device is not None else "cpu"
        self.transition_embedding = nn.Embedding(
            config.n_heads,
            PARAM_COLS,
            _weight=_param_bias_init(config).to(device=init_device),
        )
        self.d_skip = nn.Parameter(
            torch.empty(config.n_heads, device=device, dtype=torch.float32)
        )
        cast(Any, self.transition_embedding.weight)._no_weight_decay = True
        cast(Any, self.d_skip)._no_weight_decay = True
        self.register_buffer(
            "initial_state",
            oscillator_basis(config, device=init_device, dtype=torch.float32),
            persistent=False,
        )
        self.norm_weight = nn.Parameter(
            torch.empty(config.n_heads, config.d_head, device=device, dtype=dtype)
        )
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place.

        Called by the constructor. See the class docstring for which of these is a
        choice and which is a framework default.
        """
        cfg = self.config
        with torch.no_grad():
            self.in_proj.reset_parameters()
            # Consume the projection's ordinary reset before replacing its value.
            # This keeps a local no-op initialization from shifting the random draws
            # of later layers in an enclosing stack.
            self.out_proj.reset_parameters()
            self.out_proj.weight.zero_()
            if self.out_proj.bias is not None:
                self.out_proj.bias.zero_()
            # One slice, two reasons. The parameter band, because a default draw
            # there is a per-token rotation the lattice cannot survive: on unit-RMS
            # input a column fluctuates by 1/sqrt(3) whatever d_model is, so the
            # anchored drive is a vector of norm about 0.8 of the head's own bias
            # radius pointing where the input picks, which tilts the axis some 30
            # degrees a token at every timescale, and a head whose axis wanders that
            # far holds no phase. Zeroed, the recurrence starts as the oscillator
            # bank the lattice states, and input dependence is not pinned there: the
            # pullback to a row is sum_t dL/dband_t x_t and carries no factor of the
            # row itself. The pad columns after it, because they belong to no band, so
            # zeros are what a misaddressed band should read rather than plausible
            # numbers, and the cotangent's zeroed pad band has to agree with them.
            self.in_proj.weight[self.layout.b_off : self.layout.c_off].zero_()
            self.in_proj.weight[self.layout.params_off :].zero_()
            if cfg.bias:
                self.in_proj.bias[self.layout.b_off : self.layout.c_off].zero_()
                self.in_proj.bias[self.layout.params_off :].zero_()
            bound = 1.0 / math.sqrt(cfg.d_conv)
            self.conv_weight.uniform_(-bound, bound)
            if self.conv_bias is not None:
                self.conv_bias.zero_()
            if self.key_weight is not None:
                # The delta, tap W-1 being the current token: the initialized mixer
                # is the one with no key convolution at all, so the lattice the
                # parameter band states is still what step zero does, and a key that
                # spans more than its own token is only ever learned into. A draw
                # like the value band's would instead mix three neighbours into every
                # key before the first step, which is a different operator at
                # initialization and not a generalization of this one.
                self.key_weight.zero_()
                self.key_weight[:, -1] = 1.0
            self.d_skip.fill_(1.0)
            self.norm_weight.fill_(1.0)
            self.param_bias.copy_(
                _param_bias_init(cfg).to(
                    device=self.param_bias.device, dtype=self.param_bias.dtype
                )
            )
            self.initial_state.copy_(
                oscillator_basis(
                    cfg,
                    device=self.initial_state.device,
                    dtype=self.initial_state.dtype,
                )
            )

    @property
    def param_bias(self) -> Tensor:
        """Direct per-head chart rows, stored as a transition embedding."""
        return self.transition_embedding.weight

    @property
    def transition_bias(self) -> Tensor:
        """Effective per-head transition rows."""
        return self.param_bias

    @property
    def skip_gain(self) -> Tensor:
        """Effective direct-path gain."""
        return self.d_skip

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> SLinOSSMixer:
        """Apply ``fn`` to every parameter, then undo a demoted critical one.

        :meth:`torch.nn.Module.to` casts every floating-point parameter, and
        ``mixer.to(torch.bfloat16)`` is how the module is meant to reach a kernel
        dtype. The parameters :data:`CRITICAL_FP32_TENSORS` names cannot follow. Only
        a demotion is undone, so a widening cast is left alone and a float64 module
        keeps a float64 oracle end to end.

        Args:
            fn: The per-tensor operation :meth:`torch.nn.Module._apply` applies.
            recurse: Whether to descend into submodules.

        Returns:
            This module.
        """
        super()._apply(fn, recurse)
        for name in self.CRITICAL_FP32_TENSORS:
            param = cast(Tensor, getattr(self, name))
            if param.dtype in LOW_PRECISION_DTYPES:
                param.data = param.data.to(torch.float32)
        return self

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Translate pre-v2 physical rows into the finite-chart embedding.

        The initial state is a deterministic non-persistent buffer, so the serialized
        key set grows only by replacing ``param_bias`` with the explicitly named
        transition embedding. Version 1 used the unit-asymptote rotation chart;
        ``r / sqrt(1 + .75*|r|^2)`` is the raw vector in the new chart that produces
        the same physical rotation.
        """
        if int(local_metadata.get("version", 1)) < 2:
            old_key = prefix + "param_bias"
            new_key = prefix + "transition_embedding.weight"
            if old_key in state_dict:
                effective = state_dict.pop(old_key).clone()
                rotation = effective[:, ROTVEC_COLUMNS]
                radius_sq = (rotation * rotation).sum(-1, keepdim=True)
                effective[:, ROTVEC_COLUMNS] = rotation * torch.rsqrt(
                    1.0 + 0.75 * radius_sq
                )
                state_dict[new_key] = effective
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Mix one sequence.

        Args:
            x: ``(B,T,d_model)``, bf16/fp16/fp32/fp64.

        Returns:
            ``(B,T,d_model)``, in the dtype the projections produce: ``x``'s, or the
            autocast dtype under autocast.

        Raises:
            ValueError: From a consumer's guard, on a device or a layout its
                operand rule refuses.
            TypeError: From a consumer's guard, on an unsupported dtype.
        """
        return cast(
            "Tensor",
            _SLinOSSMixerFunction.apply(
                x,
                self.in_proj.weight,
                self.in_proj.bias,
                self.conv_weight,
                self.conv_bias,
                self.key_weight,
                self.transition_bias,
                self.skip_gain,
                self.norm_weight,
                self.out_proj.weight,
                self.out_proj.bias,
                self.initial_state,
                self.layout,
                self.config,
            ),
        )

    @torch.no_grad()
    def step(self, x: Tensor, state: MixerState) -> Tensor:
        """Mix ``T`` tokens continuing from ``state``, and advance ``state`` in place.

        ``T = 1`` is a decode step and ``T > 1`` is a prefill. Both run the
        composition :meth:`forward` runs, on the same backends, with the four carries
        read at the front and written at the back, so stepping a sequence in any
        partition from a freshly allocated state reproduces the whole-sequence result.

        Not an autograd node. The backends are called directly, so no graph is
        recorded whatever the caller's grad mode, and the four writes are in place:
        a captured graph holds those addresses, and a rebound buffer leaves replay
        writing memory no consumer reads.

        Cast the module rather than run under autocast. Autocast makes the
        projection's dtype the autocast dtype while the state keeps the parameter
        dtype, and casting the state per step would allocate the buffers a graph is
        supposed to own. The guard below reports that pair rather than the shape
        error a narrowed carry would raise two operators later.

        Args:
            x: ``(B,T,d_model)``, in the state's activation dtype.
            state: This layer's decode state, advanced by ``T`` tokens.

        Returns:
            ``(B,T,d_model)``, in the dtype the projections produce.

        Raises:
            ValueError: On a rank, width, batch, or dtype disagreement with
                ``state``, or from a consumer's guard.
            TypeError: From a consumer's guard, on an unsupported dtype.
        """
        cfg, layout = self.config, self.layout
        if x.ndim != 3 or x.shape[2] != cfg.d_model:
            raise ValueError(f"expected (B,T,{cfg.d_model}), got {tuple(x.shape)}")
        if x.shape[0] != state.batch:
            raise ValueError(
                f"x holds batch {int(x.shape[0])} and state holds {state.batch}"
            )
        proj = linear(x, self.in_proj.weight, self.in_proj.bias)
        if proj.dtype is not state.conv.dtype:
            raise ValueError(
                f"the projection is {proj.dtype} and the state is "
                f"{state.conv.dtype}; cast the module, not the state"
            )
        picks = _resolve(proj)
        conv = conv_dispatch.get(picks.conv).forward(
            layout.value(proj),
            cast_to(self.conv_weight, proj.dtype),
            cast_opt(self.conv_bias, proj.dtype),
            activation=True,
            initial_state=state.conv,
            d_head=cfg.d_head,
        )
        keys = (
            None
            if self.key_weight is None
            else conv_dispatch.get(picks.conv).forward(
                layout.keys(proj),
                cast_to(self.key_weight, proj.dtype),
                None,
                activation=False,
                initial_state=state.keys,
            )
        )
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj),
            self.transition_bias,
            heads=cfg.n_heads,
            w_max=cfg.w_max,
        )
        scan = scan_dispatch.get(picks.scan).forward(
            conv.y,
            params.trans,
            params.K,
            layout.b(proj) if keys is None else layout.key_b(keys.y),
            layout.c(proj) if keys is None else layout.key_c(keys.y),
            cfg.chunk_size,
            z0=state.ssm,
            b_prev=state.b_prev,
            u_prev=state.u_prev,
        )
        tail = tail_dispatch.get(picks.tail).forward(
            scan.y,
            conv.y,
            layout.gate(proj),
            self.skip_gain,
            self.norm_weight,
            eps=cfg.norm_eps,
        )
        out = linear(tail, self.out_proj.weight, self.out_proj.bias)
        # After every read: the scan starts from state.ssm itself, and the
        # convolution's incoming window is state.conv.
        state.conv.copy_(conv.state)
        if keys is not None:
            state.keys.copy_(keys.state)
        state.ssm.copy_(scan.state)
        state.b_prev.copy_(scan.b_last)
        state.u_prev.copy_(scan.u_last)
        return out
