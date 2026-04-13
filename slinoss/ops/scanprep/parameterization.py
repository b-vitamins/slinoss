"""Shared BC parameterization helpers for scanprep."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _validate_scan_bc_raw(
    bc: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
) -> None:
    expected_tail = (int(bc_groups), 2, int(d_state))
    if bc.ndim != 5 or tuple(map(int, bc.shape[2:])) != expected_tail:
        raise ValueError(
            "bc must be "
            f"(batch, T, groups, 2, d_state). Got {tuple(map(int, bc.shape))}."
        )


def _normalize_scan_bc_pairs(
    bc_pairs: torch.Tensor,
    *,
    eps: float,
    bc_gain_max: float,
) -> torch.Tensor:
    mag_sq = bc_pairs.square().sum(dim=-1, dtype=torch.float32)
    row_rms = mag_sq.mean(dim=-1, keepdim=True).clamp_min(float(eps)).sqrt()
    row_rms_expanded = row_rms.unsqueeze(-1).to(dtype=bc_pairs.dtype)
    direction = bc_pairs / row_rms_expanded
    bounded_gain = float(bc_gain_max) * torch.tanh(row_rms / float(bc_gain_max))
    return direction * bounded_gain.unsqueeze(-1).to(dtype=bc_pairs.dtype)


def parameterize_scan_bc_pairs(
    bc: torch.Tensor,
    bc_complex_base: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
    eps: float,
    bc_gain_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized complex B/C pairs from raw grouped BC amplitudes."""
    _validate_scan_bc_raw(
        bc,
        bc_groups=int(bc_groups),
        d_state=int(d_state),
    )
    if bc_complex_base.ndim != 3 or tuple(map(int, bc_complex_base.shape)) != (
        int(bc_groups),
        2,
        int(d_state),
    ):
        raise ValueError(
            "bc_complex_base must be "
            f"({int(bc_groups)}, 2, {int(d_state)}). "
            f"Got {tuple(map(int, bc_complex_base.shape))}."
        )

    bc_amplitude = F.softplus(bc)
    complex_base = torch.view_as_real(bc_complex_base).to(
        device=bc.device, dtype=bc.dtype
    )
    b_pairs = bc_amplitude[..., 0, :].unsqueeze(-1) * complex_base[:, 0, :, :].view(
        1, 1, int(bc_groups), int(d_state), 2
    )
    c_pairs = bc_amplitude[..., 1, :].unsqueeze(-1) * complex_base[:, 1, :, :].view(
        1, 1, int(bc_groups), int(d_state), 2
    )
    return (
        _normalize_scan_bc_pairs(
            b_pairs,
            eps=float(eps),
            bc_gain_max=float(bc_gain_max),
        ).contiguous(),
        _normalize_scan_bc_pairs(
            c_pairs,
            eps=float(eps),
            bc_gain_max=float(bc_gain_max),
        ).contiguous(),
    )


def parameterize_scan_bc_rows(
    bc: torch.Tensor,
    bc_complex_base: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
    eps: float,
    bc_gain_max: float,
) -> torch.Tensor:
    """Build packed real BC rows ``(B_re, B_im, C_re, C_im)`` for CuTe scanprep."""
    b_pairs, c_pairs = parameterize_scan_bc_pairs(
        bc,
        bc_complex_base,
        bc_groups=int(bc_groups),
        d_state=int(d_state),
        eps=float(eps),
        bc_gain_max=float(bc_gain_max),
    )
    return torch.stack(
        (b_pairs[..., 0], b_pairs[..., 1], c_pairs[..., 0], c_pairs[..., 1]),
        dim=3,
    )


__all__ = [
    "parameterize_scan_bc_pairs",
    "parameterize_scan_bc_rows",
]
