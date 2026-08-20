"""Configuration for the mixer, the block, and the stack.

Every shape constraint the kernels rely on is validated here, at construction,
so no kernel needs a padding path or a guard. Frozen: an invariant checked once
cannot be invalidated later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LANE_MULTIPLE = 16
"""``N`` is a multiple of this, so ``3N`` is a multiple of 48 and of 16."""

STATE_MULTIPLE = 3 * LANE_MULTIPLE
"""``d_state`` is ``3N`` and therefore a multiple of 48."""

HEAD_MULTIPLE = 16
"""``P`` is a multiple of this because it is the N mode of two of the scan GEMMs.

The MMA tile's N mode is 16 wide. An N extent that is not a multiple of 16 does
not compile: the tiled MMA fails IR verification rather than padding. 8 and 24
were measured to fail, 16, 48, 64, 96 and 128 to pass. The M mode is free; it is
rounded up inside the kernel and the store is predicated.
"""

MIN_CHUNK = 16
"""Shortest legal chunk. Below this the chunked form loses to the streaming one:
the per-chunk transform table costs order 120 FMA per token and amortizes over
``N`` lanes only from ``N = 16``."""

MAX_CHUNK = 128
"""Longest legal chunk, set by the widest block one vector load covers.

The prefix scan gives one lane a block of ``ceil(L/32)`` consecutive tokens. Up to
four words the compiler folds that block into one vector load, so the access is
conflict-free by construction, and both shared bank-conflict counters read zero at
the chunk sizes the bench covers. At eight words, which is ``L = 256``, the block
is wider than any shared vector load and the construction no longer holds. Bank
conflicts are a defect rather than a tradeoff, so the shape is constrained
instead. Nothing is lost: at ``L = 256`` the score tile alone is 256 KB
of float32 accumulator, four times the register file of a block."""


@dataclass(frozen=True)
class SLinOSSConfig:
    """Shape and parameterization contract.

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
        w_max: Bound on the rotation-vector norm. Strictly below pi so
            ``quat_exp`` is one branchless polynomial over the reachable domain.
        bias: Bias on the linear projections.
        conv_bias: Bias on the causal convolution.
        n_layers: Blocks in the stack.
        ffn_ratio: FFN hidden width as a multiple of ``d_model``.
        norm_eps: RMS norm epsilon.
        vocab_size: Embedding and head width, or None for an embedding-free
            stack.
    """

    d_model: int
    d_state: int
    expand: float = 2.0
    d_head: int = 64
    n_groups: int = 1
    chunk_size: int = 64
    d_conv: int = 4
    w_max: float = 3.0
    bias: bool = False
    conv_bias: bool = True
    n_layers: int = 1
    ffn_ratio: float = 4.0
    norm_eps: float = 1e-5
    vocab_size: int | None = None

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
        if not 0.0 < self.w_max < math.pi:
            raise ValueError(
                f"w_max must lie in (0, pi) so quat_exp stays polynomial, "
                f"got {self.w_max}"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.ffn_ratio <= 0.0:
            raise ValueError(f"ffn_ratio must be positive, got {self.ffn_ratio}")
        if self.norm_eps <= 0.0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ValueError(
                f"vocab_size must be positive or None, got {self.vocab_size}"
            )

    @property
    def d_inner(self) -> int:
        """Inner width. ``round`` so a float ``expand`` cannot truncate."""
        return round(self.expand * self.d_model)

    @property
    def n_heads(self) -> int:
        """Heads per layer, ``H``."""
        return self.d_inner // self.d_head

    @property
    def heads_per_group(self) -> int:
        """Heads sharing one ``B``/``C`` pair, ``H // G``."""
        return self.n_heads // self.n_groups

    @property
    def n_lanes(self) -> int:
        """Independent 3-vectors per head, ``N``."""
        return self.d_state // 3

    @property
    def d_ffn(self) -> int:
        """FFN hidden width."""
        return round(self.ffn_ratio * self.d_model)
