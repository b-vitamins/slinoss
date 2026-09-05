"""Oscillatory state-space sequence mixer with an SO(3) operator."""

from slinoss.blocks import BlockOutput, SLinOSSBlock
from slinoss.config import SLinOSSConfig, SLinOSSMixerConfig
from slinoss.decode import DecodeOutput, generate
from slinoss.graph import GraphedStep, capture, capture_decode
from slinoss.mixer import SLinOSSMixer
from slinoss.stack import SLinOSSStack
from slinoss.state import MixerState, StackState

__version__ = "0.1.0"

__all__ = [
    "BlockOutput",
    "DecodeOutput",
    "GraphedStep",
    "MixerState",
    "SLinOSSBlock",
    "SLinOSSConfig",
    "SLinOSSMixer",
    "SLinOSSMixerConfig",
    "SLinOSSStack",
    "StackState",
    "__version__",
    "capture",
    "capture_decode",
    "generate",
]
