from __future__ import annotations

import pytest
import torch

from slinoss.layers import SLinOSSMLP, SLinOSSMLPConfig


def test_mlp_config_resolves_hidden_dim_from_expand_and_multiple() -> None:
    config = SLinOSSMLPConfig(expand=2.5, multiple_of=32)

    assert config.resolve_hidden_dim(65) == 192


def test_mlp_config_rounds_explicit_hidden_dim_to_multiple() -> None:
    config = SLinOSSMLPConfig(hidden_dim=191, multiple_of=64)

    assert config.resolve_hidden_dim(128) == 192


@pytest.mark.parametrize(
    ("kind", "expected_in_proj_dim"),
    [
        ("swiglu", 384),
        ("gelu", 192),
    ],
)
def test_mlp_builds_expected_projection_dims(
    kind: str,
    expected_in_proj_dim: int,
) -> None:
    mlp = SLinOSSMLP(
        96,
        kind=kind,  # type: ignore[arg-type]
        hidden_dim=191,
        multiple_of=64,
    )

    assert mlp.hidden_dim == 192
    assert mlp.in_proj.in_features == 96
    assert mlp.in_proj.out_features == expected_in_proj_dim
    assert mlp.out_proj.in_features == 192
    assert mlp.out_proj.out_features == 96


@pytest.mark.parametrize("kind", ["swiglu", "gelu"])
def test_mlp_forward_preserves_model_width(kind: str) -> None:
    torch.manual_seed(0)
    mlp = SLinOSSMLP(
        128,
        kind=kind,  # type: ignore[arg-type]
        hidden_dim=192,
        multiple_of=64,
        bias=False,
    )
    x = torch.randn(2, 5, 128)

    y = mlp(x)

    assert y.shape == (2, 5, 128)


@pytest.mark.parametrize("kind", ["swiglu", "gelu"])
def test_mlp_decode_one_matches_forward_for_single_token(kind: str) -> None:
    torch.manual_seed(0)
    mlp = SLinOSSMLP(
        64,
        kind=kind,  # type: ignore[arg-type]
        hidden_dim=96,
        multiple_of=32,
        bias=True,
    )
    x = torch.randn(1, 64)

    expected = mlp(x)
    actual = mlp.decode_one(x)

    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_mlp_from_config_builds_equivalent_module() -> None:
    config = SLinOSSMLPConfig(
        kind="swiglu",
        hidden_dim=160,
        multiple_of=64,
        bias=True,
    )

    mlp = SLinOSSMLP.from_config(96, config)

    assert mlp.config == config
    assert mlp.hidden_dim == 192
    assert mlp.in_proj.out_features == 384
