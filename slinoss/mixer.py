"""The sequence mixer and the column bands of its fused input projection.

One GEMM produces every per-token operand of a mixer step: the value the
convolution filters, the gate, the two state vectors, and the ten scan parameters
per head. Each consumer reads its own column band of that one output at the
projection's pitch. Nothing is copied out of it, and nothing gets a projection of
its own.

The band geometry is here rather than in each consumer's guard because it is one
statement about one buffer. A consumer checks that what it received is a legal
band; this module decides where the bands are.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from slinoss._guard import PROJ_ALIGN
from slinoss.config import SLinOSSConfig
from slinoss.ops.scanprep import PARAM_COLS

__all__ = ["ProjectionLayout"]


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

    def _vectors(self, proj: Tensor, offset: int) -> Tensor:
        """One state band as the scan reads it.

        ``unflatten`` of a unit-stride trailing axis and ``permute`` are both views,
        so the group-major shape costs no copy.
        """
        band = proj[..., offset : offset + self.groups * self.state_dim]
        return band.unflatten(-1, (self.groups, self.state_dim)).permute(0, 2, 1, 3)
