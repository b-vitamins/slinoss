"""The MAD model scaffold: everything around the sequence mixer.

One embedding, ``n_layers`` pre-norm residual pairs of a sequence mixer and a SwiGLU
channel mixer, a final norm, and an untied head. No positional embedding: the sequence
mixer is the only thing that can tell one position from another, which is what makes the
comparison about the mixer.

The shape is `mad-lab`'s ``LanguageModel``, and :class:`BottleneckModel` is its
``AutoEncoder`` at ``global_pool='last'`` -- the backbone compression needs, whose target
is the whole input reconstructed from the state at one position. Where `mad-lab` and the
Kalman Linear Attention driver differ:

    piece            mad-lab                    KLA repository
    norm             RMSNorm, eps 1e-5          same
    channel mixer    separate SwiGLU, width 352 fused SwiGLU, width 341
    causal head      Linear with bias            Linear, no bias
    compression head Linear with bias            Linear with bias
    encoder-final    absent in AutoEncoder       RMSNorm
    decoder posemb   sincos, half-half concat    interleaved
    mixer length     seq_len (benchmark.py)      configured task seq_len

The model config selects either shape explicitly. ``kla-paper-v2`` is a locked textual
reconstruction: it uses the KLA v0.0.1 repository scaffold where the paper is silent and
replaces that repository's ``[128, 128]`` compression decoder with the paper's stated
``[240, 120]`` widths. The released repository did not execute this combination, so the
profile records that it is not independently eligible for a published-table claim.

The initialization pass reaches the scaffold only. Every mixer keeps whatever its own
constructor chose, so a mixer arrives with its authors' initialization rather than this
file's, and :func:`protect` is how a builder says so.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

MixerFactory = Callable[[int, int], nn.Module]
"""``(d_model, configured_task_length) -> module``.

MAD-Lab passes the generator's configured ``seq_len`` to every mixer constructor.
That is not always the observed tensor width: autoregressive recall shifts a
128-token generated stream into 127 model inputs.
"""


@dataclass(frozen=True)
class ModelConfig:
    """Scaffold shape.

    Attributes:
        vocab_size: Embedding rows and head columns.
        task_length: Generator's configured ``seq_len``. Handed to the mixer factory
            exactly once, matching MAD-Lab's constructor contract.
        observed_width: Positions in the tensors the generator actually returned.
            Sizes the bottleneck decoder's position code; it must not replace
            ``task_length`` at the mixer boundary.
        d_model: Residual stream width.
        n_layers: Mixer and channel-mixer pairs.
        bottleneck: Route through :class:`BottleneckModel` rather than
            :class:`CausalModel`.
        scaffold_profile: Name of the atomic harness profile that selected this shape.
        ffn_multiple_of: Rounding of the SwiGLU hidden width. 16 is `mad-lab`'s and
            gives 352 at ``d_model`` 128; 1 gives KLA's 341.
        fused_ffn_input: Use KLA's single ``w12`` input projection rather than
            MAD-Lab's separate ``w1`` and ``w2`` projections.
        head_bias: Bias on the causal LM head. `mad-lab` has one, KLA does not.
        bottleneck_head_bias: Bias on the compression unembed projection.
        bottleneck_encoder_norm: Apply KLA's final encoder norm before selecting the
            bottleneck state. MAD-Lab's autoencoder selects the raw residual stream.
        decoder_widths: The two hidden widths after the compression bottleneck.
        position_layout: ``half`` for MAD-Lab's concatenated sine/cosine code or
            ``interleaved`` for the KLA scaffold's code.
        norm_eps: RMS norm epsilon.
        init_std: Standard deviation of the scaffold's normal initialization.
    """

    vocab_size: int
    task_length: int
    observed_width: int
    d_model: int = 128
    n_layers: int = 1
    bottleneck: bool = False
    scaffold_profile: str = "legacy-hybrid"
    ffn_multiple_of: int = 16
    fused_ffn_input: bool = False
    head_bias: bool = True
    bottleneck_head_bias: bool = True
    bottleneck_encoder_norm: bool = False
    decoder_widths: tuple[int, int] = (128, 128)
    position_layout: Literal["half", "interleaved"] = "half"
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "task_length",
            "observed_width",
            "d_model",
            "n_layers",
            "ffn_multiple_of",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.norm_eps <= 0.0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")
        if not self.scaffold_profile:
            raise ValueError("scaffold_profile must not be empty")
        if len(self.decoder_widths) != 2 or any(
            width < 1 for width in self.decoder_widths
        ):
            raise ValueError(
                "decoder_widths must contain two positive widths, got "
                f"{self.decoder_widths}"
            )
        if self.position_layout not in {"half", "interleaved"}:
            raise ValueError(
                "position_layout must be 'half' or 'interleaved', got "
                f"{self.position_layout!r}"
            )

    @property
    def d_ffn(self) -> int:
        """SwiGLU hidden width: ``8/3`` of the stream, rounded up."""
        inner = int(2 * self.d_model * 4 / 3)
        multiple = self.ffn_multiple_of
        return multiple * ((inner + multiple - 1) // multiple)


class RMSNorm(nn.Module):
    """Root-mean-square norm, reduction in float32.

    The reduction is float32 whatever the input dtype, which is what `mad-lab`'s Triton
    norm and KLA's eager one both do; at bf16 a narrow reduction over ``d_model`` loses
    several bits of the scale.

    Args:
        width: Trailing extent to normalize over.
        eps: Added to the mean square.
    """

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the trailing axis.

        Args:
            x: ``(..., width)``.

        Returns:
            The same shape and dtype.
        """
        wide = x.float()
        wide = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + self.eps)
        return (wide * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """Gated channel mixer, ``w3(silu(w1 x) * w2 x)``, no bias.

    `mad-lab`'s ``SwiGLU``. Three projections at ``8/3`` of the stream cost the same
    parameters as a two-projection MLP at 4x.

    Args:
        d_model: Stream width.
        d_ffn: Hidden width.
        fused_input: Whether gate and value share one KLA-style input projection.
    """

    def __init__(self, d_model: int, d_ffn: int, *, fused_input: bool) -> None:
        super().__init__()
        self.fused_input = fused_input
        if fused_input:
            self.w12 = nn.Linear(d_model, 2 * d_ffn, bias=False)
        else:
            self.w1 = nn.Linear(d_model, d_ffn, bias=False)
            self.w2 = nn.Linear(d_model, d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Mix channels.

        Args:
            x: ``(B,T,d_model)``.

        Returns:
            ``(B,T,d_model)``.
        """
        if self.fused_input:
            gate, value = self.w12(x).chunk(2, dim=-1)
        else:
            gate, value = self.w1(x), self.w2(x)
        return self.w3(nn.functional.silu(gate) * value)


class Residual(nn.Module):
    """``x + inner(norm(x))``.

    Args:
        inner: The mixer, sequence or channel.
        width: Stream width, for the norm.
        eps: Norm epsilon.
    """

    def __init__(self, inner: nn.Module, width: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(width, eps)
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        """Add the inner module's output to its input.

        Args:
            x: ``(B,T,width)``.

        Returns:
            ``(B,T,width)``.
        """
        return x + self.inner(self.norm(x))


def protect(module: nn.Module) -> nn.Module:
    """Apply a mixer's explicit ownership rule for scaffold initialization.

    Constructor ownership is the default: a mixer's initialization is part of the
    mixer, and the scaffold may not overwrite it. An external published implementation
    can instead opt into ``scaffold`` ownership through its registry entry. That choice
    is attached by the resolved factory and recorded with every construction; guessing
    from a class or silently swallowing the model-wide pass is forbidden.

    Args:
        module: The mixer.

    Returns:
        ``module``, for use as an expression.

    Raises:
        ValueError: On an initialization policy the registry does not define.
    """
    policy = getattr(module, "_mad_initialization_policy", "constructor")
    if policy == "scaffold":
        return module
    if policy != "constructor":
        raise ValueError(f"unknown mixer initialization policy {policy!r}")
    for param in module.parameters():
        cast(Any, param)._no_reinit = True
    return module


def _fresh(tensor: Tensor) -> bool:
    """Whether the scaffold's initialization pass owns ``tensor``."""
    return not getattr(tensor, "_no_reinit", False)


def _init_scaffold(module: nn.Module, std: float) -> None:
    """Initialize one scaffold module in place.

    Normal weights on linear and embedding layers, zero biases, norms left at ones.
    KLA's ``_init_weights``, with :func:`protect`'s exemption honoured per tensor.

    Args:
        module: Candidate. Anything else is left alone.
        std: Standard deviation of the normal draw.
    """
    if isinstance(module, nn.Linear):
        if _fresh(module.weight):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        bias = cast(Tensor | None, getattr(module, "bias", None))
        if bias is not None and _fresh(bias):
            nn.init.zeros_(bias)
    elif isinstance(module, nn.Embedding) and _fresh(module.weight):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def sincos_positions(
    length: int,
    width: int,
    layout: Literal["half", "interleaved"] = "half",
) -> Tensor:
    """Sinusoidal position code from either maintained MAD scaffold.

    Sines in the first half of the width and cosines in the second, not interleaved, and
    the frequency ladder is spaced by ``log(10000)/(half - 1)`` so the slowest column is
    exactly one period over 10000 positions. An odd width takes a zero column.

    Args:
        length: Positions.
        width: Channels. At 2 the half-layout ladder is degenerate.
            divides by ``half - 1``.
        layout: ``half`` for MAD-Lab or ``interleaved`` for the KLA repository.

    Returns:
        ``(length, width)`` float32.

    Raises:
        ValueError: On ``width < 4``, where the ladder's spacing is degenerate.
    """
    if layout == "half":
        if width < 4:
            raise ValueError(f"width must be at least 4, got {width}")
        half = width // 2
        ladder = torch.exp(
            torch.arange(half, dtype=torch.float32) * -(math.log(10000) / (half - 1))
        )
        angle = torch.arange(length, dtype=torch.float32).unsqueeze(
            1
        ) * ladder.unsqueeze(0)
        code = torch.cat([angle.sin(), angle.cos()], dim=1)
        if width % 2 == 1:
            code = torch.cat([code, torch.zeros(length, 1)], dim=1)
        return code
    if layout == "interleaved":
        if width < 1:
            raise ValueError(f"width must be positive, got {width}")
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        ladder = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        code = torch.zeros(length, width, dtype=torch.float32)
        code[:, 0::2] = torch.sin(position * ladder)
        if width > 1:
            code[:, 1::2] = torch.cos(position * ladder[: width // 2])
        return code
    raise ValueError(f"unknown position layout {layout!r}")


def _encoder(config: ModelConfig, mixer: MixerFactory) -> nn.Sequential:
    """The residual stack: one mixer and one channel mixer per layer.

    Args:
        config: Scaffold shape.
        mixer: Sequence-mixer factory, called once per layer.

    Returns:
        ``n_layers * 2`` residual layers, mixer first.
    """
    layers: list[nn.Module] = []
    for _ in range(config.n_layers):
        built = protect(mixer(config.d_model, config.task_length))
        layers.append(Residual(built, config.d_model, config.norm_eps))
        layers.append(
            Residual(
                SwiGLU(
                    config.d_model,
                    config.d_ffn,
                    fused_input=config.fused_ffn_input,
                ),
                config.d_model,
                config.norm_eps,
            )
        )
    return nn.Sequential(*layers)


class CausalModel(nn.Module):
    """Embedding, residual stack, norm, head.

    `mad-lab`'s ``LanguageModel``. Every task but compression reads this: the target at
    a position is a function of that position and the ones before it.

    Args:
        config: Scaffold shape.
        mixer: Sequence-mixer factory.
    """

    def __init__(self, config: ModelConfig, mixer: MixerFactory) -> None:
        super().__init__()
        self.config = config
        self.token_embeds = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = _encoder(config, mixer)
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=config.head_bias)
        self.apply(lambda m: _init_scaffold(m, config.init_std))

    def forward(self, ids: Tensor) -> Tensor:
        """Score every position.

        Args:
            ids: ``(B,T)`` int64 token ids.

        Returns:
            ``(B,T,vocab_size)`` logits.
        """
        return self.head(self.norm(self.encoder(self.token_embeds(ids))))


class BottleneckModel(nn.Module):
    """Embedding, residual stack, the last position, position code, MLP decoder, head.

    `mad-lab`'s ``AutoEncoder`` at ``global_pool='last'``. The whole sequence has to be
    reconstructed from one ``d_model`` vector, so the state at the final position is the
    only thing the decoder sees and the task measures what that state retained. The
    decoder is not residual. Its two widths are profile-owned because the published
    KLA table specifies ``[240, 120]`` while the current KLA repository uses
    ``[128, 128]``.

    Args:
        config: Scaffold shape. ``observed_width`` sizes the position code.
        mixer: Sequence-mixer factory.
    """

    def __init__(self, config: ModelConfig, mixer: MixerFactory) -> None:
        super().__init__()
        self.config = config
        self.token_embeds = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = _encoder(config, mixer)
        self.encoder_norm = (
            RMSNorm(config.d_model, config.norm_eps)
            if config.bottleneck_encoder_norm
            else nn.Identity()
        )
        first, second = config.decoder_widths
        self.decoder = nn.Sequential(
            RMSNorm(config.d_model, config.norm_eps),
            nn.Linear(config.d_model, first),
            nn.GELU(),
            RMSNorm(first, config.norm_eps),
            nn.Linear(first, second),
            nn.GELU(),
        )
        self.norm = RMSNorm(second, config.norm_eps)
        self.head = nn.Linear(
            second, config.vocab_size, bias=config.bottleneck_head_bias
        )
        self.register_buffer(
            "positions",
            sincos_positions(
                config.observed_width, config.d_model, config.position_layout
            ),
            persistent=False,
        )
        self.apply(lambda m: _init_scaffold(m, config.init_std))

    def forward(self, ids: Tensor) -> Tensor:
        """Reconstruct every position from the last state.

        Args:
            ids: ``(B,T)`` int64 token ids, ``T`` at most
                ``config.observed_width``.

        Returns:
            ``(B,T,vocab_size)`` logits.

        Raises:
            ValueError: On an input wider than the position code.
        """
        length = int(ids.shape[1])
        if length > self.config.observed_width:
            raise ValueError(
                f"input has {length} positions, code carries "
                f"{self.config.observed_width}"
            )
        state = self.encoder_norm(self.encoder(self.token_embeds(ids)))[:, -1:, :]
        code = cast(Tensor, self.positions)[:length]
        return self.head(self.norm(self.decoder(state + code)))


def build_model(config: ModelConfig, mixer: MixerFactory) -> nn.Module:
    """The backbone the task calls for.

    Args:
        config: Scaffold shape.
        mixer: Sequence-mixer factory.

    Returns:
        :class:`BottleneckModel` when ``config.bottleneck``, else :class:`CausalModel`.
        Both map ``(B,T)`` int64 to ``(B,T,vocab_size)``.
    """
    if config.bottleneck:
        return BottleneckModel(config, mixer)
    return CausalModel(config, mixer)


def parameter_count(model: nn.Module) -> int:
    """Trainable parameters.

    Args:
        model: Any module.

    Returns:
        Their total element count.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
