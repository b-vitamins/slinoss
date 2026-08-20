"""Oscillatory state-space sequence mixer with an SO(3) operator."""

from slinoss.blocks import BlockOutput, SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.decode import DecodeOutput, generate
from slinoss.mixer import SLinOSSMixer
from slinoss.stack import SLinOSSStack
from slinoss.state import MixerState, StackState

__version__ = "0.1.0"

__all__ = [
    "BlockOutput",
    "DecodeOutput",
    "MixerState",
    "SLinOSSBlock",
    "SLinOSSConfig",
    "SLinOSSMixer",
    "SLinOSSStack",
    "StackState",
    "__version__",
    "generate",
]
