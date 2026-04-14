"""First-class residual blocks and backbone stacks built on SLinOSS layers."""

from slinoss.layers import SLinOSSMLPConfig

from .block import SLinOSSBlock
from .config import (
    FinalNormKind,
    NormKind,
    SLinOSSBlockConfig,
    SLinOSSMixerConfig,
    SLinOSSStackConfig,
    sandwich_block_schedule,
    scaled_budget_schedule,
    uniform_block_schedule,
)
from .stack import SLinOSSStack
from .state import SLinOSSBlockState, SLinOSSStackState

__all__ = [
    "FinalNormKind",
    "NormKind",
    "SLinOSSBlock",
    "SLinOSSBlockConfig",
    "SLinOSSBlockState",
    "SLinOSSMLPConfig",
    "SLinOSSMixerConfig",
    "SLinOSSStack",
    "SLinOSSStackConfig",
    "SLinOSSStackState",
    "sandwich_block_schedule",
    "scaled_budget_schedule",
    "uniform_block_schedule",
]
