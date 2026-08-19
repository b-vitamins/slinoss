"""Shape contract. Every raise in :class:`SLinOSSConfig` is triggered here."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest

from slinoss.config import (
    HEAD_MULTIPLE,
    LANE_MULTIPLE,
    MAX_CHUNK,
    MIN_CHUNK,
    STATE_MULTIPLE,
    SLinOSSConfig,
)

BASE: dict[str, Any] = {"d_model": 64, "d_state": 48}


def cfg(**overrides: Any) -> SLinOSSConfig:
    return SLinOSSConfig(**{**BASE, **overrides})


def test_multiples_are_consistent() -> None:
    assert STATE_MULTIPLE == 3 * LANE_MULTIPLE
    assert STATE_MULTIPLE % 16 == 0
    assert MIN_CHUNK <= MAX_CHUNK


def test_defaults() -> None:
    c = cfg()
    assert (c.d_inner, c.n_heads, c.n_lanes, c.d_ffn) == (128, 2, 16, 256)
    assert c.chunk_size == 64
    assert c.vocab_size is None


def test_fractional_expand_rounds() -> None:
    c = cfg(expand=1.5, d_head=16)
    assert c.d_inner == 96
    assert c.n_heads == 6


def test_fractional_ffn_ratio_rounds() -> None:
    assert cfg(ffn_ratio=2.5).d_ffn == 160


def test_frozen() -> None:
    c = cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.d_model = 128  # type: ignore[misc]


@pytest.mark.parametrize("d_state", [48, 96, 480])
def test_legal_state_widths(d_state: int) -> None:
    c = cfg(d_state=d_state)
    assert c.n_lanes * 3 == d_state
    assert c.n_lanes % LANE_MULTIPLE == 0


@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_legal_chunk_sizes(chunk_size: int) -> None:
    assert cfg(chunk_size=chunk_size).chunk_size == chunk_size


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"d_model": 0}, "d_model must be positive"),
        ({"d_model": -1}, "d_model must be positive"),
        ({"d_state": 0}, "d_state"),
        ({"d_state": 47}, "d_state"),
        ({"d_state": 49}, "d_state"),
        ({"d_state": 45}, "d_state"),
        ({"expand": 0.0}, "expand must be positive"),
        ({"expand": -2.0}, "expand must be positive"),
        ({"d_head": 0}, "d_head"),
        ({"d_head": 4}, "d_head"),
        ({"d_head": 12}, "d_head"),
        ({"d_head": 48}, "not divisible by d_head"),
        ({"chunk_size": MIN_CHUNK - 1}, "chunk_size must lie in"),
        ({"chunk_size": MAX_CHUNK + 1}, "chunk_size must lie in"),
        ({"chunk_size": 48}, "power of two"),
        ({"chunk_size": 100}, "power of two"),
        ({"d_conv": 0}, "d_conv must be positive"),
        ({"w_max": 0.0}, "w_max must lie in"),
        ({"w_max": -1.0}, "w_max must lie in"),
        ({"w_max": math.pi}, "w_max must lie in"),
        ({"w_max": 4.0}, "w_max must lie in"),
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"ffn_ratio": 0.0}, "ffn_ratio must be positive"),
        ({"norm_eps": 0.0}, "norm_eps must be positive"),
        ({"norm_eps": -1e-5}, "norm_eps must be positive"),
        ({"vocab_size": 0}, "vocab_size must be positive"),
    ],
)
def test_rejects(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cfg(**overrides)


@pytest.mark.parametrize("d_head", [16, 32, 64, 128])
def test_heads_tile_the_inner_width(d_head: int) -> None:
    # d_head is P, the N mode of two scan GEMMs. The MMA N tile is 16 wide.
    c = cfg(d_head=d_head)
    assert d_head % HEAD_MULTIPLE == 0
    assert c.n_heads * c.d_head == c.d_inner
