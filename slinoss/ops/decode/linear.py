"""Decode-specialized small-batch linear helpers."""

import torch
from torch import nn


def decode_linear(x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
    """Run a decode-time linear on the fastest practical path for the batch."""

    if x.ndim < 2:
        raise ValueError(
            f"decode_linear expects at least 2 dims with batch leading, got {tuple(x.shape)}."
        )
    if x.ndim != 2:
        return linear(x)
    if x.shape[0] != 1:
        return linear(x)
    x0 = x[0]
    if linear.bias is None:
        return torch.mv(linear.weight, x0).unsqueeze(0)
    return torch.addmv(linear.bias, linear.weight, x0).unsqueeze(0)


__all__ = ["decode_linear"]
