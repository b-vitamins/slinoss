"""Model definitions built on top of SLinOSS layers."""

from .nextchar import (
    FeedForward,
    NextCharBlock,
    NextCharDecodeState,
    NextCharLM,
    configure_optim,
)

__all__ = [
    "FeedForward",
    "NextCharBlock",
    "NextCharDecodeState",
    "NextCharLM",
    "configure_optim",
]
