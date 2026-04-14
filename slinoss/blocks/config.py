"""Typed configuration for SLinOSS blocks and stacks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from slinoss.layers._validation import _require
from slinoss.layers.mlp import SLinOSSMLPConfig

NormKind = Literal["rmsnorm", "layernorm"]
FinalNormKind = Literal["rmsnorm", "layernorm", "none"]


def _lerp(start: float, end: float, index: int, count: int) -> float:
    if count <= 1:
        return float(start)
    alpha = float(index) / float(count - 1)
    return float(start + (end - start) * alpha)


@dataclass(frozen=True, slots=True)
class SLinOSSMixerConfig:
    """Configuration for the token-mixing path of a block."""

    d_state: int = 128
    expand: float = 2.0
    d_head: int = 64
    d_conv: int = 4
    chunk_size: int = 64
    bc_groups: int | None = None
    dt_min: float = 3.0e-2
    dt_max: float = 1.0e-1
    dt_init_floor: float = 3.0e-2
    gamma_min: float = 2.0
    gamma_max: float = 8.0
    theta_init_min: float = 0.2
    theta_init_max: float = 1.0
    r_min: float = 0.9
    r_max: float = 1.0
    bc_gain_max: float = 2.0
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        _require(self.d_state > 0, f"d_state must be positive. Got {self.d_state}.")
        _require(self.expand > 0.0, f"expand must be positive. Got {self.expand}.")
        _require(self.d_head > 0, f"d_head must be positive. Got {self.d_head}.")
        _require(self.d_conv > 0, f"d_conv must be positive. Got {self.d_conv}.")
        _require(
            self.chunk_size > 0,
            f"chunk_size must be positive. Got {self.chunk_size}.",
        )
        if self.bc_groups is not None:
            _require(
                self.bc_groups > 0,
                f"bc_groups must be positive. Got {self.bc_groups}.",
            )

    def build_kwargs(self) -> dict[str, object]:
        return {
            "d_state": self.d_state,
            "expand": self.expand,
            "d_head": self.d_head,
            "d_conv": self.d_conv,
            "chunk_size": self.chunk_size,
            "bc_groups": self.bc_groups,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "theta_init_min": self.theta_init_min,
            "theta_init_max": self.theta_init_max,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "bc_gain_max": self.bc_gain_max,
            "eps": self.eps,
        }


@dataclass(frozen=True, slots=True)
class SLinOSSBlockConfig:
    """Configuration for a single serial residual block."""

    d_model: int
    mixer: SLinOSSMixerConfig = field(default_factory=SLinOSSMixerConfig)
    ffn: SLinOSSMLPConfig | None = field(default_factory=SLinOSSMLPConfig)
    norm_kind: NormKind = "rmsnorm"
    norm_eps: float = 1.0e-5
    residual_in_fp32: bool = True
    residual_dropout: float = 0.0

    def __post_init__(self) -> None:
        _require(self.d_model > 0, f"d_model must be positive. Got {self.d_model}.")
        _require(
            self.norm_kind in ("rmsnorm", "layernorm"),
            f"Unsupported norm_kind: {self.norm_kind!r}.",
        )
        _require(
            self.norm_eps > 0.0, f"norm_eps must be positive. Got {self.norm_eps}."
        )
        _require(
            0.0 <= self.residual_dropout < 1.0,
            f"residual_dropout must lie in [0, 1). Got {self.residual_dropout}.",
        )


@dataclass(frozen=True, slots=True)
class SLinOSSStackConfig:
    """Configuration for a stack of SLinOSS blocks."""

    blocks: tuple[SLinOSSBlockConfig, ...]
    final_norm_kind: FinalNormKind = "rmsnorm"
    final_norm_eps: float = 1.0e-5
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        _require(bool(self.blocks), "SLinOSSStackConfig requires at least one block.")
        d_model = self.blocks[0].d_model
        for index, block in enumerate(self.blocks[1:], start=1):
            _require(
                block.d_model == d_model,
                "All block configs in a stack must share d_model. "
                f"Block 0 has {d_model}, block {index} has {block.d_model}.",
            )
        _require(
            self.final_norm_kind in ("rmsnorm", "layernorm", "none"),
            f"Unsupported final_norm_kind: {self.final_norm_kind!r}.",
        )
        _require(
            self.final_norm_eps > 0.0,
            f"final_norm_eps must be positive. Got {self.final_norm_eps}.",
        )

    @property
    def d_model(self) -> int:
        return self.blocks[0].d_model

    @classmethod
    def uniform(
        cls,
        block: SLinOSSBlockConfig,
        *,
        n_layers: int,
        final_norm_kind: FinalNormKind = "rmsnorm",
        final_norm_eps: float = 1.0e-5,
        gradient_checkpointing: bool = False,
    ) -> "SLinOSSStackConfig":
        return cls(
            blocks=uniform_block_schedule(block, n_layers=n_layers),
            final_norm_kind=final_norm_kind,
            final_norm_eps=final_norm_eps,
            gradient_checkpointing=gradient_checkpointing,
        )


def uniform_block_schedule(
    block: SLinOSSBlockConfig,
    *,
    n_layers: int,
) -> tuple[SLinOSSBlockConfig, ...]:
    _require(n_layers > 0, f"n_layers must be positive. Got {n_layers}.")
    return tuple(block for _ in range(n_layers))


def sandwich_block_schedule(
    *,
    stem: SLinOSSBlockConfig,
    middle: SLinOSSBlockConfig,
    tail: SLinOSSBlockConfig,
    n_layers: int,
) -> tuple[SLinOSSBlockConfig, ...]:
    _require(n_layers > 0, f"n_layers must be positive. Got {n_layers}.")
    if n_layers == 1:
        return (stem,)
    if n_layers == 2:
        return (stem, tail)
    return (stem, *(middle for _ in range(n_layers - 2)), tail)


def scaled_budget_schedule(
    base: SLinOSSBlockConfig,
    *,
    n_layers: int,
    mixer_expand_range: tuple[float, float] | None = None,
    ffn_expand_range: tuple[float, float] | None = None,
    residual_dropout_range: tuple[float, float] | None = None,
) -> tuple[SLinOSSBlockConfig, ...]:
    _require(n_layers > 0, f"n_layers must be positive. Got {n_layers}.")
    if ffn_expand_range is not None and base.ffn is not None:
        _require(
            base.ffn.hidden_dim is None,
            "scaled_budget_schedule can only scale FFN expand when "
            "base.ffn.hidden_dim is not fixed.",
        )

    blocks: list[SLinOSSBlockConfig] = []
    for index in range(n_layers):
        mixer = base.mixer
        if mixer_expand_range is not None:
            mixer = replace(
                mixer,
                expand=_lerp(
                    mixer_expand_range[0],
                    mixer_expand_range[1],
                    index,
                    n_layers,
                ),
            )

        ffn = base.ffn
        if ffn_expand_range is not None and ffn is not None:
            ffn = replace(
                ffn,
                expand=_lerp(
                    ffn_expand_range[0],
                    ffn_expand_range[1],
                    index,
                    n_layers,
                ),
            )

        residual_dropout = base.residual_dropout
        if residual_dropout_range is not None:
            residual_dropout = _lerp(
                residual_dropout_range[0],
                residual_dropout_range[1],
                index,
                n_layers,
            )

        blocks.append(
            replace(
                base,
                mixer=mixer,
                ffn=ffn,
                residual_dropout=residual_dropout,
            )
        )
    return tuple(blocks)


__all__ = [
    "FinalNormKind",
    "NormKind",
    "SLinOSSBlockConfig",
    "SLinOSSMLPConfig",
    "SLinOSSMixerConfig",
    "SLinOSSStackConfig",
    "sandwich_block_schedule",
    "scaled_budget_schedule",
    "uniform_block_schedule",
]
