"""Named, atomic MAD harness profiles and their comparison eligibility.

A published number should be a contract, not a bag of defaults. A named profile binds
the model scaffold and optimization protocol together so a command cannot silently use
half of one upstream implementation and half of another. It also says whether an
executable upstream artifact establishes that contract. A locked textual reconstruction
is reproducible, but that does not make it eligible for a published-table claim.
Runtime placement and random seeds remain outside the profile; they do not change the
mathematical experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HarnessProfile:
    """One complete scaffold/protocol contract.

    ``locked`` means command-line values may repeat these settings but may not replace
    them.  The legacy profile is deliberately unlocked so old commands remain
    replayable, while a record names them honestly as the historical hybrid.
    """

    name: str
    locked: bool
    contract_status: str
    published_table_eligible: bool
    limitations: tuple[str, ...]

    # Scaffold.
    d_model: int
    n_layers: int
    ffn_multiple_of: int
    fused_ffn_input: bool
    head_bias: bool
    bottleneck_head_bias: bool
    bottleneck_encoder_norm: bool
    decoder_widths: tuple[int, int]
    position_layout: str
    norm_eps: float
    init_std: float

    # Optimization protocol.
    epochs: int
    batch_size: int
    lr: float
    min_lr: float
    weight_decay: float
    schedule: str
    warmup_epochs: int
    grad_clip: float
    patience: int
    eval_every: int
    drop_last: bool
    float32_matmul_precision: str
    precision: str

    # Human-auditable provenance for the choices above.
    references: tuple[str, ...]

    def model_args(self) -> dict[str, Any]:
        """Resolved arguments for :class:`scripts.mad.model.ModelConfig`."""
        return {
            "scaffold_profile": self.name,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "ffn_multiple_of": self.ffn_multiple_of,
            "fused_ffn_input": self.fused_ffn_input,
            "head_bias": self.head_bias,
            "bottleneck_head_bias": self.bottleneck_head_bias,
            "bottleneck_encoder_norm": self.bottleneck_encoder_norm,
            "decoder_widths": self.decoder_widths,
            "position_layout": self.position_layout,
            "norm_eps": self.norm_eps,
            "init_std": self.init_std,
        }

    def train_args(self) -> dict[str, Any]:
        """Resolved arguments for :class:`scripts.mad.train.TrainConfig`."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "schedule": self.schedule,
            "warmup_epochs": self.warmup_epochs,
            "grad_clip": self.grad_clip,
            "patience": self.patience,
            "eval_every": self.eval_every,
            "drop_last": self.drop_last,
            "float32_matmul_precision": self.float32_matmul_precision,
            "precision": self.precision,
        }

    def record(self) -> dict[str, Any]:
        """JSON-ready full profile, including its evidence labels."""
        return asdict(self)


LEGACY_HYBRID = HarnessProfile(
    name="legacy-hybrid",
    locked=False,
    contract_status="historical-hybrid",
    published_table_eligible=False,
    limitations=(
        "MAD-Lab scaffold combined with KLA optimizer defaults; no upstream source "
        "defines this composite contract",
    ),
    d_model=128,
    n_layers=1,
    ffn_multiple_of=16,
    fused_ffn_input=False,
    head_bias=True,
    bottleneck_head_bias=True,
    bottleneck_encoder_norm=False,
    decoder_widths=(128, 128),
    position_layout="half",
    norm_eps=1e-5,
    init_std=0.02,
    epochs=750,
    batch_size=128,
    lr=1e-3,
    min_lr=1e-6,
    weight_decay=0.0,
    schedule="none",
    warmup_epochs=5,
    grad_clip=5.0,
    patience=70,
    eval_every=10,
    drop_last=True,
    float32_matmul_precision="high",
    precision="fp32",
    references=(
        "historical in-tree runs: MAD-Lab scaffold plus KLA optimizer defaults",
    ),
)


KLA_PAPER_V2 = HarnessProfile(
    name="kla-paper-v2",
    locked=True,
    contract_status="textual-reconstruction",
    published_table_eligible=False,
    limitations=(
        "No released KLA source tree or result record reproduces the paper's MAD table",
        "KLA v0.0.1 has no implementation/config for the paper's stated encoder MLP "
        "hidden dimension 120",
        "KLA v0.0.1 deliberately uses batch 128 and compression decoder [128,128], "
        "while the paper states 172 and [240,120]",
    ),
    d_model=128,
    n_layers=1,
    ffn_multiple_of=1,
    fused_ffn_input=True,
    head_bias=False,
    bottleneck_head_bias=True,
    bottleneck_encoder_norm=True,
    decoder_widths=(240, 120),
    position_layout="interleaved",
    norm_eps=1e-5,
    init_std=0.02,
    epochs=750,
    batch_size=172,
    lr=1e-3,
    min_lr=1e-6,
    weight_decay=0.0,
    schedule="none",
    warmup_epochs=5,
    grad_clip=5.0,
    patience=70,
    eval_every=10,
    drop_last=True,
    float32_matmul_precision="high",
    precision="fp32",
    references=(
        "KLA arXiv:2602.10743v2 Table 8 (MAD task settings)",
        "KLA arXiv:2602.10743v2 Table 10 and Appendix G.3 (training)",
        "KLA arXiv:2602.10743v2 Appendix F.1.2 (compression decoder 240,120)",
        "KLA v0.0.1 repository scaffold for details not fixed by the paper; this is "
        "a documented inference, not an executable reproduction artifact",
        "official MAD-Lab noisy-recall vocab 32 corrects Table 8's impossible 16",
    ),
)


PROFILES = {profile.name: profile for profile in (LEGACY_HYBRID, KLA_PAPER_V2)}
"""Profiles accepted by the driver, keyed exactly as the record spells them."""


def get_profile(name: str) -> HarnessProfile:
    """Resolve a profile name, refusing typos rather than falling back."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"no MAD profile {name}; have {sorted(PROFILES)}") from None
