"""Direct token-emitted complex BC parameterization helpers for scanprep."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

RAW_BC_PARAM_ROWS = 4
PACKED_BC_ROWS = 4
PHASE_LIMIT = math.pi


def validate_scan_bc_raw(
    bc: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
) -> None:
    expected_tail = (int(bc_groups), RAW_BC_PARAM_ROWS, int(d_state))
    if bc.ndim != 5 or tuple(map(int, bc.shape[2:])) != expected_tail:
        raise ValueError(
            "bc must be "
            f"(batch, T, groups, {RAW_BC_PARAM_ROWS}, d_state). "
            f"Got {tuple(map(int, bc.shape))}."
        )


def validate_scan_bc_rows(
    bc_rows: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
) -> None:
    expected_tail = (int(bc_groups), PACKED_BC_ROWS, int(d_state))
    if bc_rows.ndim != 5 or tuple(map(int, bc_rows.shape[2:])) != expected_tail:
        raise ValueError(
            "Packed bc rows must be "
            f"(batch, T, groups, {PACKED_BC_ROWS}, d_state). "
            f"Got {tuple(map(int, bc_rows.shape))}."
        )


def _normalize_scan_bc_pairs(
    bc_pairs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    mag_sq = bc_pairs.square().sum(dim=-1, dtype=torch.float32)
    row_rms = mag_sq.mean(dim=-1, keepdim=True).clamp_min(float(eps)).sqrt()
    return bc_pairs / row_rms.unsqueeze(-1).to(dtype=bc_pairs.dtype)


def _phase_rotor(phase_logits: torch.Tensor) -> torch.Tensor:
    phase = PHASE_LIMIT * torch.tanh(phase_logits.to(torch.float32))
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def parameterize_scan_bc_pairs(
    bc: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized complex B/C pairs directly from token-emitted polar fields.

    Raw grouped BC rows are interpreted as:
    - ``bc[..., 0, :]``: B amplitude logits
    - ``bc[..., 1, :]``: B phase logits
    - ``bc[..., 2, :]``: C amplitude logits
    - ``bc[..., 3, :]``: C phase logits
    """
    validate_scan_bc_raw(
        bc,
        bc_groups=int(bc_groups),
        d_state=int(d_state),
    )

    b_amp = F.softplus(bc[..., 0, :].to(torch.float32))
    c_amp = F.softplus(bc[..., 2, :].to(torch.float32))
    b_phase = _phase_rotor(bc[..., 1, :])
    c_phase = _phase_rotor(bc[..., 3, :])

    b_pairs = b_amp.unsqueeze(-1) * b_phase
    c_pairs = c_amp.unsqueeze(-1) * c_phase

    pair_dtype = (
        bc.dtype
        if bc.dtype in (torch.float16, torch.bfloat16, torch.float32)
        else torch.float32
    )
    return (
        _normalize_scan_bc_pairs(
            b_pairs,
            eps=float(eps),
        )
        .to(dtype=pair_dtype)
        .contiguous(),
        _normalize_scan_bc_pairs(
            c_pairs,
            eps=float(eps),
        )
        .to(dtype=pair_dtype)
        .contiguous(),
    )


def parameterize_scan_bc_rows(
    bc: torch.Tensor,
    *,
    bc_groups: int,
    d_state: int,
    eps: float,
) -> torch.Tensor:
    """Build packed real BC rows ``(B_re, B_im, C_re, C_im)`` for CuTe scanprep."""
    b_pairs, c_pairs = parameterize_scan_bc_pairs(
        bc,
        bc_groups=int(bc_groups),
        d_state=int(d_state),
        eps=float(eps),
    )
    rows = torch.stack(
        (b_pairs[..., 0], b_pairs[..., 1], c_pairs[..., 0], c_pairs[..., 1]),
        dim=3,
    )
    validate_scan_bc_rows(
        rows,
        bc_groups=int(bc_groups),
        d_state=int(d_state),
    )
    return rows


__all__ = [
    "PACKED_BC_ROWS",
    "PHASE_LIMIT",
    "RAW_BC_PARAM_ROWS",
    "parameterize_scan_bc_pairs",
    "parameterize_scan_bc_rows",
    "validate_scan_bc_raw",
    "validate_scan_bc_rows",
]
