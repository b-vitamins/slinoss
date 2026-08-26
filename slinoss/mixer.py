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
from slinoss.ops.scanprep import LS_COLUMN, PARAM_COLS, ROTVEC_COLUMNS
from slinoss.ops.scanprep import backends as prep_dispatch
from slinoss.ops.so3ssd import backends as scan_dispatch
from slinoss.ops.so3ssd.reference import ScanPrologue
from slinoss.state import MixerState

__all__ = [
    "HORIZON_RANGE",
    "ProjectionLayout",
    "SLinOSSMixer",
    "head_grid",
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

    def _vectors(self, proj: Tensor, offset: int) -> Tensor:
        """One state band as the scan reads it.

        ``unflatten`` of a unit-stride trailing axis and ``permute`` are both views,
        so the group-major shape costs no copy.
        """
        band = proj[..., offset : offset + self.groups * self.state_dim]
        return band.unflatten(-1, (self.groups, self.state_dim)).permute(0, 2, 1, 3)


HORIZON_RANGE: tuple[float, float] = (4.0, 256.0)
"""Tokens a head turns in and remembers over, at initialization.

One head per point of a log-spaced grid over this range, so the initial scales
cover it geometrically instead of clustering at one end. One range and not one per
row: a head's rotation period and its decay time constant are the same number, so
one turn takes one amplitude lifetime. Two ranges over the same geometric center
differ only in span, and their ratio is then a log-linear sweep of turns per
lifetime that nothing asks for.

The low end is bounded twice. A two-token period is the alternating sign, which no
sampled rotation resolves, and its ``2*pi/2`` exceeds
:data:`_MAX_ANGLE_FRACTION` of any legal ``w_max``. The high end is free, and is
the one number that moves the whole grid.
"""

_AXIS_SEED = 0
"""Seed of the rotation-axis draw.

Fixed, and not the ambient generator: the axis grid is part of what
initialization states, so it must not move with global RNG state. Every layer of
a stack therefore draws the same axes.
"""

_MAX_ANGLE_FRACTION = 0.9
"""Cap on ``|w| / w_max`` at initialization.

The inverse of :func:`slinoss.ops.scanprep.bounded_rotvec` diverges as the
requested norm approaches the bound. Inactive at the default ``w_max``, where the
shortest horizon in :data:`HORIZON_RANGE` asks for ``pi/2``, one half of it.
"""


def head_grid(bounds: tuple[float, float], heads: int) -> Tensor:
    """One value per head, log-spaced over ``bounds``, endpoints included.

    Log-spaced because the quantity is a scale: a linear grid over a ratio of 64
    puts most heads at the high end.

    Args:
        bounds: ``(low, high)``, both positive.
        heads: ``H``. One head takes ``low``.

    Returns:
        ``(H,)``, float32, on the CPU.
    """
    low, high = bounds
    return torch.logspace(math.log10(low), math.log10(high), heads, dtype=torch.float32)


def _param_bias_init(config: SLinOSSConfig) -> Tensor:
    """Rows of ``param_bias`` at initialization, ``(H, PARAM_COLS)``, float32, CPU.

    Both bounded maps are inverted, so the grid states what the recurrence does at
    step zero rather than what an unbounded parameterization would do. Both scale
    rows read the one grid, at ``h`` tokens for the head:

    - ``bounded_logscale(raw) = -softplus(raw)`` and the per-token amplitude factor
      is ``exp(2*ls)``, so an amplitude that falls by ``exp(-1)`` over ``h``
      tokens is ``ls = -1/(2*h)`` at ``raw = log(expm1(1/(2*h)))``.
    - ``bounded_rotvec(raw, w_max) = raw * w_max * rsqrt(1 + |raw|^2)`` gives
      ``|w| = s * w_max`` at ``|raw| = s * rsqrt(1 - s*s)``. Conjugation turns by
      ``|w|`` per token: ``quat_exp`` builds the half-angle quaternion and the
      conjugation doubles the half-angle back. A turn every ``h`` tokens is
      therefore ``|w| = 2*pi/h``.

    The taps take no row. They are the first-order-hold moments of the transition
    these two rows set, so the grid reaches them already.

    Args:
        config: Supplies ``H`` and ``w_max``.

    Returns:
        ``(H, PARAM_COLS)``, float32.
    """
    heads = config.n_heads
    rows = torch.zeros(heads, PARAM_COLS, dtype=torch.float32)
    horizon = head_grid(HORIZON_RANGE, heads)
    rows[:, LS_COLUMN] = torch.log(torch.expm1(0.5 / horizon))
    angle = 2.0 * math.pi / horizon
    fraction = (angle / config.w_max).clamp(max=_MAX_ANGLE_FRACTION)
    radius = fraction * torch.rsqrt(1.0 - fraction * fraction)
    axis = torch.randn(
        heads,
        3,
        generator=torch.Generator().manual_seed(_AXIS_SEED),
        dtype=torch.float32,
    )
    rows[:, ROTVEC_COLUMNS] = radius[:, None] * axis / axis.norm(dim=-1, keepdim=True)
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
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
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
        param_bias: Tensor,
        d_skip: Tensor,
        norm_weight: Tensor,
        out_weight: Tensor,
        out_bias: Tensor | None,
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
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj), param_bias, heads=config.n_heads, w_max=config.w_max
        )
        scan = scan_dispatch.get(picks.scan).forward(
            step.y,
            params.trans,
            params.K,
            layout.b(proj),
            layout.c(proj),
            config.chunk_size,
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
        zstart, cquat, cscale = saved[16:]
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
            layout.b(proj),
            layout.c(proj),
            config.chunk_size,
            dB=layout.b(dproj),
            dC=layout.c(dproj),
            dU_init=tail_grads.du,
            prologue=(None if zstart is None else ScanPrologue(zstart, cquat, cscale)),
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
            prep_grads.dparam_bias,
            tail_grads.dd_skip,
            tail_grads.dweight,
            cast_to(out_grads.dweight, out_weight.dtype),
            cast_opt(out_grads.dbias, out_weight.dtype),
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
    activation dtype. ``param_bias`` is float32 at every module dtype and stays
    float32 through a module-wide cast, because the scan parameters it biases are
    pinned (I4).

    CUDA. Every operand between the two projections is a column band, and
    :func:`slinoss._guard.check_pitched` holds a band to a device rule, so a CPU
    call raises from the first consumer that checks one.

    :meth:`forward` mixes a whole sequence and threads no state. :meth:`step`
    continues one: same composition, same backends, with the four carries
    :class:`slinoss.state.MixerState` holds read at the front and written at the
    back. It takes no gradient.

    Initialization is principled where the scale sets the recurrence and the
    framework default elsewhere. ``param_bias`` is inverted through both bounded
    maps onto one horizon grid over :data:`HORIZON_RANGE`, with an isotropic axis per
    head; ``d_skip`` and ``norm_weight`` are ones. The two
    projections keep :meth:`torch.nn.Linear.reset_parameters`, which is a default
    and not a choice, and the convolution taps take the same uniform bound over
    ``d_conv``. No depth scaling.

    Args:
        config: Shape and parameterization contract.
        device: Device for every parameter.
        dtype: Dtype for every parameter except ``param_bias``.
    """

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
        self.param_bias = nn.Parameter(
            torch.empty(config.n_heads, PARAM_COLS, device=device, dtype=torch.float32)
        )
        self.d_skip = nn.Parameter(
            torch.empty(config.n_heads, config.d_head, device=device, dtype=dtype)
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
            self.out_proj.reset_parameters()
            # The pad columns belong to no band. Zeroed rows keep them zero for
            # every input, so a misaddressed band reads zeros rather than plausible
            # numbers, and the cotangent's zeroed pad band is consistent with them.
            stop = self.layout.params_off + PARAM_COLS * cfg.n_heads
            self.in_proj.weight[stop:].zero_()
            if cfg.bias:
                self.in_proj.bias[stop:].zero_()
            bound = 1.0 / math.sqrt(cfg.d_conv)
            self.conv_weight.uniform_(-bound, bound)
            if self.conv_bias is not None:
                self.conv_bias.zero_()
            self.d_skip.fill_(1.0)
            self.norm_weight.fill_(1.0)
            self.param_bias.copy_(_param_bias_init(cfg))

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> SLinOSSMixer:
        """Apply ``fn`` to every parameter, then undo a demoted ``param_bias``.

        :meth:`torch.nn.Module.to` casts every floating-point parameter, and
        ``mixer.to(torch.bfloat16)`` is how the module is meant to reach a kernel
        dtype. ``param_bias`` is the one parameter that cannot follow: scanprep
        raises on a low-precision one (I4). A widening cast is left alone, so a
        float64 module keeps a float64 oracle end to end.

        Args:
            fn: The per-tensor operation :meth:`torch.nn.Module._apply` applies.
            recurse: Whether to descend into submodules.

        Returns:
            This module.
        """
        super()._apply(fn, recurse)
        if self.param_bias.dtype in LOW_PRECISION_DTYPES:
            self.param_bias.data = self.param_bias.data.to(torch.float32)
        return self

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
                self.param_bias,
                self.d_skip,
                self.norm_weight,
                self.out_proj.weight,
                self.out_proj.bias,
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
        partition from a zeroed state reproduces the whole-sequence result.

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
        params = prep_dispatch.get(picks.prep).forward(
            layout.params(proj), self.param_bias, heads=cfg.n_heads, w_max=cfg.w_max
        )
        scan = scan_dispatch.get(picks.scan).forward(
            conv.y,
            params.trans,
            params.K,
            layout.b(proj),
            layout.c(proj),
            cfg.chunk_size,
            z0=state.ssm,
            b_prev=state.b_prev,
            u_prev=state.u_prev,
        )
        tail = tail_dispatch.get(picks.tail).forward(
            scan.y,
            conv.y,
            layout.gate(proj),
            self.d_skip,
            self.norm_weight,
            eps=cfg.norm_eps,
        )
        out = linear(tail, self.out_proj.weight, self.out_proj.bias)
        # After every read: the scan starts from state.ssm itself, and the
        # convolution's incoming window is state.conv.
        state.conv.copy_(conv.state)
        state.ssm.copy_(scan.state)
        state.b_prev.copy_(scan.b_last)
        state.u_prev.copy_(scan.u_last)
        return out
