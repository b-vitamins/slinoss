"""Validated mixer and stack configuration."""

from __future__ import annotations

from dataclasses import dataclass

LANE_MULTIPLE = 16
"""``N`` is a multiple of this, so ``3N`` is a multiple of 48 and of 16."""

STATE_MULTIPLE = 3 * LANE_MULTIPLE
"""``d_state`` is ``3N`` and therefore a multiple of 48."""

HEAD_MULTIPLE = 16
"""MMA width of the scan GEMMs' ``P`` dimension."""

VOCAB_MULTIPLE = 8
"""Bfloat16 head columns covered by one sixteen-byte operand load."""

ROTATION_CHART_SCALE_MAX = 3.141592502593994
"""Largest float32 strictly below pi.

The parameter frontier pins its scalar arithmetic to float32. A decimal merely
below :data:`math.pi` can round upward when it enters a kernel; this exact value
cannot, so twice the chart scale remains strictly below ``2*pi`` there as well.
"""

MIN_CHUNK = 16
"""Shortest chunk that amortizes the transform table over the lane tile."""

MAX_CHUNK = 128
"""Longest chunk whose lane block fits one conflict-free vector load."""


@dataclass(frozen=True)
class SLinOSSMixerConfig:
    """Mixer shape and parameterization contract.

    Attributes:
        d_model: Residual-stream width.
        d_state: Per-head state width ``3N``. Multiple of 48.
        expand: Inner width multiplier. ``d_inner = round(expand * d_model)``.
        d_head: Rows per head, ``P``. Multiple of 16.
        n_groups: Groups sharing one ``B``/``C`` pair, ``G``. Divides ``n_heads``.
            Head ``h`` reads group ``h // (n_heads // n_groups)``. At ``n_groups
            == n_heads`` every head carries its own pair; at 1 all heads share
            one. Only ``B`` and ``C`` are shared, never ``U``, the transition, the
            taps, or the state.
        chunk_size: Scan chunk length ``L``. Power of two in [16, 128]; the
            quaternion prefix scan is a shuffle scan over ``log2(L)`` steps.
        d_conv: Causal depthwise convolution width.
        key_conv: Convolve the ``B`` and ``C`` bands as well as the value band.
            Their taps start at the delta, so the initialized mixer is the one
            without it and a key motif is only ever learned into.
        bias: Bias on the linear projections.
        conv_bias: Bias on the causal convolution.
        norm_eps: RMS norm epsilon.
    """

    d_model: int
    d_state: int
    expand: float = 2.0
    d_head: int = 64
    n_groups: int = 1
    chunk_size: int = 64
    d_conv: int = 4
    key_conv: bool = True
    bias: bool = False
    conv_bias: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_state < STATE_MULTIPLE or self.d_state % STATE_MULTIPLE != 0:
            raise ValueError(
                f"d_state is 3N with N a multiple of {LANE_MULTIPLE}, so it must be "
                f"a positive multiple of {STATE_MULTIPLE}; got {self.d_state}"
            )
        if self.expand <= 0.0:
            raise ValueError(f"expand must be positive, got {self.expand}")
        if self.d_head < HEAD_MULTIPLE or self.d_head % HEAD_MULTIPLE != 0:
            raise ValueError(
                f"d_head must be a positive multiple of {HEAD_MULTIPLE}, "
                f"got {self.d_head}"
            )
        if self.d_inner % self.d_head != 0:
            raise ValueError(
                f"d_inner {self.d_inner} is not divisible by d_head {self.d_head}"
            )
        if self.n_groups < 1:
            raise ValueError(f"n_groups must be positive, got {self.n_groups}")
        if self.n_heads % self.n_groups != 0:
            raise ValueError(
                f"n_groups {self.n_groups} does not divide n_heads {self.n_heads}; "
                f"a group holds a whole number of heads"
            )
        if not MIN_CHUNK <= self.chunk_size <= MAX_CHUNK:
            raise ValueError(
                f"chunk_size must lie in [{MIN_CHUNK}, {MAX_CHUNK}], "
                f"got {self.chunk_size}"
            )
        if self.chunk_size & (self.chunk_size - 1) != 0:
            raise ValueError(
                f"chunk_size must be a power of two, got {self.chunk_size}"
            )
        if self.d_conv < 1:
            raise ValueError(f"d_conv must be positive, got {self.d_conv}")
        if self.norm_eps <= 0.0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")

    @property
    def d_inner(self) -> int:
        """Expanded mixer width."""
        return round(self.expand * self.d_model)

    @property
    def n_heads(self) -> int:
        """Number of heads."""
        return self.d_inner // self.d_head

    @property
    def heads_per_group(self) -> int:
        """Heads sharing one B/C pair."""
        return self.n_heads // self.n_groups

    @property
    def n_lanes(self) -> int:
        """Independent 3-vectors per recurrent row."""
        return self.d_state // 3


@dataclass(frozen=True)
class SLinOSSConfig(SLinOSSMixerConfig):
    """Block/stack settings layered on :class:`SLinOSSMixerConfig`."""

    n_layers: int = 1
    ffn_ratio: float = 4.0
    vocab_size: int | None = None
    vocab_pad_multiple: int = VOCAB_MULTIPLE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.ffn_ratio <= 0.0:
            raise ValueError(f"ffn_ratio must be positive, got {self.ffn_ratio}")
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ValueError(
                f"vocab_size must be positive or None, got {self.vocab_size}"
            )
        if self.vocab_pad_multiple < 1:
            raise ValueError(
                f"vocab_pad_multiple must be positive, got {self.vocab_pad_multiple}"
            )
        if (
            self.vocab_pad_multiple != 1
            and self.vocab_pad_multiple % VOCAB_MULTIPLE != 0
        ):
            raise ValueError(
                f"vocab_pad_multiple is 1 for no padding, else a multiple of "
                f"{VOCAB_MULTIPLE}: a head width that is not a whole number of "
                f"operand loads stays on the narrow-load kernel, so a multiple in "
                f"between costs parameters and buys nothing; got "
                f"{self.vocab_pad_multiple}"
            )

    @property
    def d_ffn(self) -> int:
        """FFN hidden width."""
        return round(self.ffn_ratio * self.d_model)

    @property
    def padded_vocab_size(self) -> int | None:
        """``vocab_size`` rounded up to ``vocab_pad_multiple``, or None."""
        if self.vocab_size is None:
            return None
        multiple = self.vocab_pad_multiple
        return -(-self.vocab_size // multiple) * multiple
