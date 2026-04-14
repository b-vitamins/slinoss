"""Normalization layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.RMSNorm):
    """RMSNorm with explicit input/weight dtype control."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            return F.rms_norm(x, self.normalized_shape, None, self.eps)
        out = F.rms_norm(
            x.to(self.weight.dtype),
            self.normalized_shape,
            self.weight,
            self.eps,
        )
        return out.to(dtype=x.dtype)
