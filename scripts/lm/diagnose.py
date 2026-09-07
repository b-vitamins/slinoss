"""Bounded LM differential diagnosis for current, Mamba-3, and legacy SLinOSS.

This is an experimental runner, not a second training protocol.  It uses the
same corpus windows, model scaffold, parameter groups, learning-rate transfer,
schedule, token batch, and full held-out shard as :mod:`scripts.lm.run`.  The
only deliberate substitution is an explicit PyTorch cross entropy shared by
all three operators, which lets the legacy package run in the same process
without importing the current package's fused loss.

The output is one self-describing JSON receipt.  Telemetry is sampled before
global clipping and the time spent taking snapshots is excluded from reported
training throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from scripts.lm import corpus as corpus_mod
from scripts.lm.data import Batch, Shard, batches, val_batches
from scripts.lm.groups import GroupPolicy, group_counts, parameter_groups
from scripts.lm.schedule import lr_at, transfer

ARMS = (
    "current",
    "mamba3",
    "mamba3-heavy-tail",
    "official-mamba3",
    "official-mamba3-depth-scaled",
    "old-v2x2",
    "zero-z0",
    "paired-bc",
    "zero-z0-paired-bc",
    "normalized-transition",
    "normalized-transition-no-key-conv",
    "normalized-transition-depth-scaled",
    "v2-lift-so3",
    "r10-old-bc",
    "r10-mamba3-bc",
    "r10-mamba3-bc-unit",
    "r10-mamba3-bc-unit-zero-bias",
    "r10-mamba3-bc-unit-frozen-bias",
    "r10-mamba3-bc-unit-frozen-token",
    "r10-mamba3-bc-unit-key-conv",
)
SNAPSHOT_STEPS = frozenset({0, 1, 4, 9, 19, 24, 49, 74, 99})
DEEP_STEPS = frozenset({-1, 9, 19, 99})
_LOG2E = 1.0 / math.log(2.0)


def _rmsnorm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    dtype = torch.promote_types(x.dtype, torch.float32)
    scale = torch.rsqrt(x.to(dtype).square().mean(-1, keepdim=True) + eps)
    return (x.to(dtype) * scale * weight.to(dtype)).to(x.dtype)


def _unwrap(value: Any) -> Tensor:
    return cast(Tensor, value[0] if isinstance(value, tuple) else value)


def _stats(value: Tensor) -> dict[str, float | bool]:
    x = value.detach().float()
    return {
        "rms": float(x.square().mean().sqrt()),
        "mean": float(x.mean()),
        "max_abs": float(x.abs().max()),
        "finite": bool(torch.isfinite(x).all()),
    }


def _cosine(a: Tensor, b: Tensor) -> float:
    x, y = a.detach().float().flatten(), b.detach().float().flatten()
    denom = x.norm() * y.norm()
    return 0.0 if float(denom) == 0.0 else float(torch.dot(x, y) / denom)


def _loss(model: nn.Module, batch: Batch, classes: int) -> Tensor:
    logits = model(batch.inputs)
    acc = logits.flatten(0, 1)[:, :classes].float()
    labels = batch.targets.reshape(-1).long()
    return (torch.logsumexp(acc, -1) - acc.gather(1, labels[:, None])[:, 0]).mean()


class _OldBlock(nn.Module):
    def __init__(self, d_model: int, mixer: nn.Module, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_eps = eps
        self.norm_weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.mixer = mixer

    def forward(self, x: Tensor) -> Tensor:
        return x + _unwrap(self.mixer(_rmsnorm(x, self.norm_weight, self.norm_eps)))


class _OldLM(nn.Module):
    """The current mixer-only scaffold around the legacy v2x2 operator."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        mixers: Sequence[nn.Module],
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            d_model=d_model, n_layers=n_layers, vocab_size=vocab_size, norm_eps=1e-5
        )
        padded = -(-vocab_size // 8) * 8
        self.embedding = nn.Embedding(vocab_size, d_model, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList(_OldBlock(d_model, mixer) for mixer in mixers)
        self.norm_weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.head = nn.Linear(d_model, padded, bias=False, dtype=torch.float32)

    def _apply(self, fn: Any, recurse: bool = True) -> _OldLM:
        super()._apply(fn, recurse)
        self.embedding.weight.data = self.embedding.weight.data.to(torch.bfloat16)
        self.norm_weight.data = self.norm_weight.data.float()
        for block in self.blocks:
            cast(_OldBlock, block).norm_weight.data = cast(
                _OldBlock, block
            ).norm_weight.data.float()
        return self

    def forward(self, ids: Tensor) -> Tensor:
        hidden = self.embedding(ids).to(self.head.weight.dtype)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(_rmsnorm(hidden, self.norm_weight, 1e-5))


class _PackedRowScale(nn.Module):
    """Apply a fixed functional scale to one packed projection row band."""

    def __init__(
        self,
        rows: int,
        start: int,
        stop: int,
        factor: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        scale = torch.ones(rows, 1, device=device, dtype=dtype)
        scale[start:stop] = factor
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, weight: Tensor) -> Tensor:
        return weight * self.scale


def _build_old(
    *,
    old_root: Path,
    old_commit: str,
    d_model: int,
    n_layers: int,
    vocab_size: int,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    if not old_root.is_dir():
        raise FileNotFoundError(f"legacy source root does not exist: {old_root}")
    sys.path.insert(0, str(old_root))
    try:
        from slinoss.layers import SLinOSSMixer as OldMixer
    except Exception:
        sys.path.remove(str(old_root))
        raise
    mixers = [
        OldMixer(
            d_model,
            d_state=48,  # 48 complex coordinates == current 96 real coordinates.
            expand=2.0,
            d_head=64,
            d_conv=4,
            chunk_size=64,
            bc_groups=1,
        )
        for _ in range(n_layers)
    ]
    model = _OldLM(
        d_model=d_model, n_layers=n_layers, vocab_size=vocab_size, mixers=mixers
    ).to(device=device, dtype=torch.float32)
    return model, {
        "operator": "legacy-v2x2",
        "d_state_complex": 48,
        "d_state_real": 96,
        "expand": 2.0,
        "d_head": 64,
        "bc_groups": 1,
        "source_commit": old_commit,
    }


def _build_current(
    *, arm: str, d_model: int, n_layers: int, vocab_size: int, device: str
) -> tuple[nn.Module, dict[str, Any]]:
    from scripts.lm import model as model_mod
    from scripts.lm.mixers import REGISTRY, Unwrap

    if arm in {"official-mamba3", "official-mamba3-depth-scaled"}:
        from mamba_ssm.modules.mamba3 import Mamba3 as OfficialMamba3

        scaffold = model_mod.LMConfig(
            d_model=d_model, n_layers=n_layers, vocab_size=vocab_size
        )

        def factory(width: int, _max_length: int) -> nn.Module:
            return OfficialMamba3(
                d_model=width,
                d_state=96,
                expand=2,
                headdim=64,
                ngroups=1,
                chunk_size=64,
            )

        model = model_mod.build_model(
            scaffold,
            model_mod.layer_factories(factory, n_layers),
            max_length=2048,
            device=device,
            dtype=torch.float32,
        )
        if arm == "official-mamba3-depth-scaled":
            residual_scale = 1.0 / math.sqrt(n_layers)
            with torch.no_grad():
                for block in model.blocks:
                    block.mixer.out_proj.weight.mul_(residual_scale)
        else:
            residual_scale = 1.0
        return model, {
            "operator": "official-mamba3",
            "implementation": "mamba_ssm.modules.mamba3.Mamba3",
            "settings": {
                "d_state": 96,
                "expand": 2,
                "headdim": 64,
                "ngroups": 1,
                "chunk_size": 64,
            },
            "mutation": (
                "none"
                if residual_scale == 1.0
                else f"out_proj.weight *= 1/sqrt(n_layers) = {residual_scale:.17g}"
            ),
        }

    if arm == "mamba3-heavy-tail":
        from fla.layers.mamba3 import Mamba3 as FLAMamba3
        from mamba_ssm.modules.mamba3 import heavy_tail_activation

        class HeavyTailFLAMamba3(FLAMamba3):
            def _compute_a(self, dd_a: Tensor) -> Tensor:
                return (-heavy_tail_activation(dd_a.float())).clamp(
                    max=-self.A_floor
                )

        scaffold = model_mod.LMConfig(
            d_model=d_model, n_layers=n_layers, vocab_size=vocab_size
        )

        def factory(width: int, _max_length: int) -> nn.Module:
            return Unwrap(
                HeavyTailFLAMamba3(
                    hidden_size=width,
                    state_size=96,
                    expand=2,
                    head_dim=64,
                    n_groups=1,
                    chunk_size=64,
                )
            )

        model = model_mod.build_model(
            scaffold,
            model_mod.layer_factories(factory, n_layers),
            max_length=2048,
            device=device,
            dtype=torch.float32,
        )
        return model, {
            "operator": "mamba3-heavy-tail",
            "implementation": "fla.layers.mamba3.Mamba3",
            "settings": {
                "state_size": 96,
                "expand": 2,
                "head_dim": 64,
                "n_groups": 1,
                "chunk_size": 64,
            },
            "mutation": "replace FLA negative-softplus A with official heavy-tail A",
        }

    mixer_name = "mamba3" if arm == "mamba3" else "slinoss"
    resolved = REGISTRY.resolve(mixer_name, ())
    factory = resolved.factory
    resolved_settings = dict(resolved.settings)
    if arm == "normalized-transition-no-key-conv":
        from scripts.harness import build_slinoss

        resolved_settings["key_conv"] = False

        def factory(width: int, _max_length: int) -> nn.Module:
            return build_slinoss(width, **resolved_settings)

    scaffold = model_mod.LMConfig(
        d_model=d_model, n_layers=n_layers, vocab_size=vocab_size
    )
    if arm == "v2-lift-so3":
        from scripts.harness import slinoss_defaults
        from scripts.harness.v2_lift_so3 import build_v2_lift_so3

        settings = slinoss_defaults(96)
        settings["key_conv"] = False

        def factory(width: int, _max_length: int) -> nn.Module:
            return build_v2_lift_so3(width, **settings)

        model = model_mod.build_model(
            scaffold,
            model_mod.layer_factories(factory, n_layers),
            max_length=2048,
            device=device,
            dtype=torch.float32,
        )
        return model, {
            "operator": "v2-lift-so3",
            "settings": settings,
            "mutation": (
                "production v2x2ssd learning geometry lifted to SO(3): live fan-in "
                "projection; U-only convolution; polar row-RMS B/C; global post-gate "
                "RMSNorm; normal D; live output; fan-in-normalized full-reach transition; "
                "current deterministic cyclic z0"
            ),
        }
    if arm in {
        "r10-old-bc",
        "r10-mamba3-bc",
        "r10-mamba3-bc-unit",
        "r10-mamba3-bc-unit-zero-bias",
        "r10-mamba3-bc-unit-frozen-bias",
        "r10-mamba3-bc-unit-frozen-token",
        "r10-mamba3-bc-unit-key-conv",
    }:
        from scripts.harness import slinoss_defaults
        from scripts.harness.prior_bc_so3 import (
            build_mamba3_bc,
            build_old_slinoss_bc,
            build_unit_mamba3_bc,
            build_unit_mamba3_bc_key_conv,
            build_unit_zero_bias_mamba3_bc,
        )

        settings = slinoss_defaults(96)
        settings["key_conv"] = False
        builder = {
            "r10-old-bc": build_old_slinoss_bc,
            "r10-mamba3-bc": build_mamba3_bc,
            "r10-mamba3-bc-unit": build_unit_mamba3_bc,
            "r10-mamba3-bc-unit-zero-bias": build_unit_zero_bias_mamba3_bc,
            "r10-mamba3-bc-unit-frozen-bias": build_unit_mamba3_bc,
            "r10-mamba3-bc-unit-frozen-token": build_unit_mamba3_bc,
            "r10-mamba3-bc-unit-key-conv": build_unit_mamba3_bc_key_conv,
        }[arm]

        def factory(width: int, _max_length: int) -> nn.Module:
            return builder(width, **settings)

        model = model_mod.build_model(
            scaffold,
            model_mod.layer_factories(factory, n_layers),
            max_length=2048,
            device=device,
            dtype=torch.float32,
        )
        source = "old-v2x2" if arm == "r10-old-bc" else "mamba3"
        magnitude = (
            "; realized vectors rescaled to unit L2 norm as a magnitude-only control"
            if arm in {
                "r10-mamba3-bc-unit",
                "r10-mamba3-bc-unit-zero-bias",
                "r10-mamba3-bc-unit-frozen-bias",
                "r10-mamba3-bc-unit-frozen-token",
                "r10-mamba3-bc-unit-key-conv",
            }
            else ""
        )
        alignment = (
            "; B_bias and C_bias initialized to zero but left trainable"
            if arm == "r10-mamba3-bc-unit-zero-bias"
            else ""
        )
        if arm == "r10-mamba3-bc-unit-frozen-bias":
            for block in model.blocks:
                block.mixer.transition_bias.requires_grad_(False)
            alignment += "; transition_bias frozen"
        if arm == "r10-mamba3-bc-unit-frozen-token":
            for block in model.blocks:
                mixer = block.mixer
                start = mixer.layout.params_off
                stop = mixer.layout.out_features

                def zero_token_transition(
                    grad: Tensor, *, start: int = start, stop: int = stop
                ) -> Tensor:
                    grad[start:stop].zero_()
                    return grad

                mixer.in_proj.weight.register_hook(zero_token_transition)
            alignment += "; token-transition projection rows frozen at zero"
        if arm == "r10-mamba3-bc-unit-key-conv":
            alignment += "; B/C causal convolution retained (delta initialized)"
        return model, {
            "operator": arm,
            "settings": settings,
            "mutation": (
                "R10 transition and current SO(3)/cyclic-z0 body; bias-free default "
                "fan-in B/C projection; no B/C convolution; source-faithful "
                f"{source} B/C realization{magnitude}{alignment}"
            ),
        }
    model = model_mod.build_model(
        scaffold,
        model_mod.layer_factories(factory, n_layers),
        max_length=2048,
        device=device,
        dtype=torch.float32,
    )
    mutation = "none"
    if arm in {"zero-z0", "zero-z0-paired-bc"}:
        with torch.no_grad():
            for block in model.blocks:
                block.mixer.initial_state.zero_()
        mutation = "initial_state.zero_()"
    if arm in {"paired-bc", "zero-z0-paired-bc"}:
        with torch.no_grad():
            for block in model.blocks:
                mixer = block.mixer
                layout = mixer.layout
                weight = mixer.in_proj.weight
                b = weight[layout.b_off : layout.c_off]
                c = weight[layout.c_off : layout.params_off]
                if b.shape != c.shape:
                    raise ValueError(
                        f"B/C shapes differ: {tuple(b.shape)} vs {tuple(c.shape)}"
                    )
                c.copy_(b)
        mutation = (
            "C_projection_rows.copy_(B_projection_rows); per-row and total B/C "
            "initial variance unchanged"
            if mutation == "none"
            else mutation + "; C_projection_rows.copy_(B_projection_rows)"
        )
    if arm in {
        "normalized-transition",
        "normalized-transition-no-key-conv",
        "normalized-transition-depth-scaled",
    }:
        factor = 1.0 / math.sqrt(d_model)
        for block in model.blocks:
            mixer = block.mixer
            layout = mixer.layout
            weight = mixer.in_proj.weight
            parametrize.register_parametrization(
                mixer.in_proj,
                "weight",
                _PackedRowScale(
                    weight.shape[0],
                    layout.params_off,
                    layout.out_features,
                    factor,
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            )
        mutation = (
            "token-transition projection output scaled by "
            f"1/sqrt(d_model)={factor:.17g}; reachable set unchanged"
        )
    if arm == "normalized-transition-no-key-conv":
        mutation += "; key_conv=False"
    if arm == "normalized-transition-depth-scaled":
        residual_scale = 1.0 / math.sqrt(n_layers)
        with torch.no_grad():
            for block in model.blocks:
                block.mixer.out_proj.weight.mul_(residual_scale)
        mutation += (
            "; out_proj.weight *= 1/sqrt(n_layers) = "
            f"{residual_scale:.17g}"
        )
    return model, {
        "operator": mixer_name,
        "settings": resolved_settings,
        "mutation": mutation,
    }


def _family(name: str) -> str:
    name = re.sub(r"^blocks\.\d+\.", "blocks.*.", name)
    return name.replace(".inner.", ".")


def _metric_rows(rows: Iterable[tuple[str, Tensor]]) -> dict[str, dict[str, float]]:
    accum: dict[str, list[float]] = {}
    for name, tensor in rows:
        x = tensor.detach().float()
        slot = accum.setdefault(name, [0.0, 0.0, 0.0])
        slot[0] += float(x.square().sum())
        slot[1] += x.numel()
        slot[2] = max(slot[2], float(x.abs().max()))
    return {
        name: {
            "norm": math.sqrt(total),
            "rms": math.sqrt(total / count),
            "max_abs": maximum,
            "numel": count,
        }
        for name, (total, count, maximum) in sorted(accum.items())
    }


def _parameter_metrics(model: nn.Module) -> dict[str, dict[str, float]]:
    return _metric_rows(
        (_family(name), param) for name, param in model.named_parameters()
    )


def _gradient_metrics(model: nn.Module) -> dict[str, Any]:
    generic = _metric_rows(
        (_family(name), cast(Tensor, param.grad))
        for name, param in model.named_parameters()
        if param.grad is not None
    )
    leaves = _metric_rows(
        (name, cast(Tensor, param.grad))
        for name, param in model.named_parameters()
        if param.grad is not None
    )
    result: dict[str, Any] = {"families": generic, "leaves": leaves}

    # The current mixer packs five causally different paths into in_proj.  A
    # whole-matrix norm would hide the exact path monopolising global clipping.
    try:
        from slinoss.mixer import SLinOSSMixer
    except (ImportError, ModuleNotFoundError):
        return result
    sliced: list[tuple[str, Tensor]] = []
    transition_heads: list[dict[str, float | int]] = []
    from slinoss.mixer import _head_lattice

    for layer, module in enumerate(
        item for item in model.modules() if isinstance(item, SLinOSSMixer)
    ):
        if parametrize.is_parametrized(module.in_proj, "weight"):
            grad = module.in_proj.parametrizations.weight.original.grad
        else:
            grad = module.in_proj.weight.grad
        if grad is None:
            continue
        layout = module.layout
        sliced.extend(
            (
                ("value", grad[: layout.gate_off]),
                ("gate", grad[layout.gate_off : layout.b_off]),
                ("B", grad[layout.b_off : layout.c_off]),
                ("C", grad[layout.c_off : layout.params_off]),
                ("token_transition", grad[layout.params_off : layout.out_features]),
            )
        )
        bias_grad = module.transition_bias.grad
        if bias_grad is not None:
            horizon, period = _head_lattice(module.config.n_heads)
            token = grad[layout.params_off : layout.out_features].unflatten(
                0, (module.config.n_heads, 4)
            )
            for head in range(module.config.n_heads):
                transition_heads.append(
                    {
                        "layer": layer,
                        "head": head,
                        "initial_horizon": float(horizon[head]),
                        "initial_period": float(period[head]),
                        "bias_rotation_grad_norm": float(
                            bias_grad[head, :3].float().norm()
                        ),
                        "bias_decay_grad_abs": float(bias_grad[head, 3].float().abs()),
                        "token_rotation_grad_norm": float(
                            token[head, :3].float().norm()
                        ),
                        "token_decay_grad_norm": float(token[head, 3].float().norm()),
                    }
                )
    if sliced:
        result["slinoss_in_proj_bands"] = _metric_rows(sliced)
    if transition_heads:
        result["slinoss_transition_heads"] = transition_heads
    return result


@torch.no_grad()
def _inspect_current_mixer(mixer: nn.Module, x: Tensor) -> dict[str, Any]:
    from slinoss._precision import cast_opt, cast_to
    from slinoss.config import ROTATION_CHART_SCALE_MAX
    from slinoss.mixer import _aligned_linear, _l2_normalize_, _resolve
    from slinoss.ops.conv import backends as conv_dispatch
    from slinoss.ops.scanprep import backends as prep_dispatch
    from slinoss.ops.so3ssd import backends as scan_dispatch
    from slinoss.ops.so3ssd.reference import tap_matrix

    cfg, layout = mixer.config, mixer.layout
    proj = _aligned_linear(x, mixer.in_proj.weight, mixer.in_proj.bias, layout)
    picks = _resolve(proj)
    step = conv_dispatch.get(picks.conv).forward(
        layout.value(proj),
        cast_to(mixer.conv_weight, proj.dtype),
        cast_opt(mixer.conv_bias, proj.dtype),
        activation=True,
        d_head=cfg.d_head,
    )
    keys = (
        None
        if mixer.key_weight is None
        else conv_dispatch.get(picks.conv)
        .forward(
            layout.keys(proj),
            cast_to(mixer.key_weight, proj.dtype),
            None,
            activation=False,
        )
        .y
    )
    b_raw = (layout.b(proj) if keys is None else layout.key_b(keys)).clone()
    c_raw = (layout.c(proj) if keys is None else layout.key_c(keys)).clone()
    if hasattr(mixer, "_vectors"):
        b, c = mixer._vectors(b_raw, c_raw)
        vector_rule = type(mixer).__name__
    else:
        b, c = b_raw.clone(), c_raw.clone()
        _l2_normalize_(b)
        _l2_normalize_(c)
        vector_rule = "l2-unit"
    raw_params = layout.params(proj)
    tangent_scale = float(getattr(mixer, "transition_tangent_scale", 1.0))
    params = prep_dispatch.get(picks.prep).forward(
        raw_params * tangent_scale,
        mixer.transition_bias,
        heads=cfg.n_heads,
        w_max=ROTATION_CHART_SCALE_MAX,
    )
    z0 = mixer.initial_state.unsqueeze(0).expand(x.shape[0], -1, -1, -1).contiguous()
    scan = scan_dispatch.get(picks.scan)
    full = scan.forward(step.y, params.trans, params.K, b, c, cfg.chunk_size, z0=z0)
    carrier = scan.forward(
        torch.zeros_like(step.y), params.trans, params.K, b, c, cfg.chunk_size, z0=z0
    )
    write = scan.forward(
        step.y, params.trans, params.K, b, c, cfg.chunk_size, z0=torch.zeros_like(z0)
    )
    skip = mixer.d_skip[None, :, None, None] * step.y
    closure = (full.y - carrier.y - write.y).float()
    denom = full.y.float().norm().clamp_min(1e-30)

    heads_per_group = cfg.n_heads // cfg.n_groups
    bh = (
        b
        if b.shape[1] == cfg.n_heads
        else b.repeat_interleave(heads_per_group, dim=1)
    ).unflatten(-1, (-1, 3))
    ch = (
        c
        if c.shape[1] == cfg.n_heads
        else c.repeat_interleave(heads_per_group, dim=1)
    ).unflatten(-1, (-1, 3))
    coupling: dict[str, dict[str, float | bool]] = {}
    for tap_index, name in enumerate(("previous", "current")):
        matrix = tap_matrix(params.K[..., tap_index, :3], params.trans[..., :3])
        kb = torch.einsum("bhtij,bhtnj->bhtni", matrix, bh)
        score = (ch * kb).sum((-1, -2))
        coupling[name] = _stats(score)

    decay = torch.exp(2.0 * params.trans[..., 3])
    return {
        "vector_rule": vector_rule,
        "transition_tangent_scale": tangent_scale,
        "raw_B_vector_norm": _stats(torch.linalg.vector_norm(b_raw.float(), dim=-1)),
        "raw_C_vector_norm": _stats(torch.linalg.vector_norm(c_raw.float(), dim=-1)),
        "realized_B_vector_norm": _stats(torch.linalg.vector_norm(b.float(), dim=-1)),
        "realized_C_vector_norm": _stats(torch.linalg.vector_norm(c.float(), dim=-1)),
        "normalized_BC_dot": _stats((b.float() * c.float()).sum(-1)),
        "C_K_B": coupling,
        "transition_decay": _stats(decay),
        "transition_angle": _stats(
            torch.linalg.vector_norm(params.trans[..., :3], dim=-1)
        ),
        "U": _stats(step.y),
        "carrier_read": _stats(carrier.y),
        "write_read": _stats(write.y),
        "scan_read": _stats(full.y),
        "skip": _stats(skip),
        "pre_tail": _stats(full.y + skip),
        "linearity_relative_error": float(closure.norm() / denom),
    }


@torch.no_grad()
def _inspect_mamba3_mixer(mixer: nn.Module, x: Tensor) -> dict[str, Any]:
    inner = getattr(mixer, "inner", mixer)
    if not _is_mamba3(inner):
        raise TypeError(f"not a Mamba3 mixer: {type(inner)!r}")

    projected = inner.in_proj(x)
    d_inner = int(getattr(inner, "d_inner", getattr(inner, "intermediate_size", 0)))
    d_state = int(getattr(inner, "d_state", getattr(inner, "ssm_state_size", 0)))
    n_heads = int(getattr(inner, "nheads", getattr(inner, "num_heads", 0)))
    n_groups = int(getattr(inner, "num_bc_heads", getattr(inner, "n_groups", 0)))
    head_dim = int(getattr(inner, "headdim", getattr(inner, "head_dim", 0)))
    rank = int(inner.mimo_rank)
    angles = int(inner.num_rope_angles)
    z, value, b_raw, c_raw, dd_dt, dd_a, trap, _angle = torch.split(
        projected,
        [
            d_inner,
            d_inner,
            d_state * n_groups * rank,
            d_state * n_groups * rank,
            n_heads,
            n_heads,
            n_heads,
            angles,
        ],
        dim=-1,
    )
    shape = (*b_raw.shape[:-1], rank, n_groups, d_state)
    b_norm = inner.B_norm(b_raw.view(shape))
    c_norm = inner.C_norm(c_raw.view(shape))
    heads_per_group = n_heads // n_groups
    b_head = b_norm.repeat_interleave(heads_per_group, dim=-2)
    c_head = c_norm.repeat_interleave(heads_per_group, dim=-2)
    b_bias = inner.B_bias.permute(1, 0, 2)
    c_bias = inner.C_bias.permute(1, 0, 2)
    b_real = b_head + b_bias
    c_real = c_head + c_bias

    dt = torch.nn.functional.softplus(dd_dt.float() + inner.dt_bias.float())
    if hasattr(inner, "_compute_a"):
        a = inner._compute_a(dd_a)
        a_rule = "negative-softplus"
    else:
        from mamba_ssm.modules.mamba3 import heavy_tail_activation

        a = -heavy_tail_activation(dd_a.float())
        a = a.clamp(max=-float(inner.A_floor))
        a_rule = "negative-heavy-tail"
    alpha = torch.exp(a * dt)
    lam = torch.sigmoid(trap.float())
    gamma = dt * lam
    beta = dt * (1.0 - lam) * alpha
    qk = (b_real.float() * c_real.float()).sum(-1)
    value = value.unflatten(-1, (n_heads, head_dim))
    z = z.unflatten(-1, (n_heads, head_dim))
    return {
        "implementation": f"{type(inner).__module__}.{type(inner).__name__}",
        "A_rule": a_rule,
        "raw_B": _stats(b_raw),
        "raw_C": _stats(c_raw),
        "normalized_B_vector_norm": _stats(
            torch.linalg.vector_norm(b_norm.float(), dim=-1)
        ),
        "normalized_C_vector_norm": _stats(
            torch.linalg.vector_norm(c_norm.float(), dim=-1)
        ),
        "realized_B_vector_norm": _stats(
            torch.linalg.vector_norm(b_real.float(), dim=-1)
        ),
        "realized_C_vector_norm": _stats(
            torch.linalg.vector_norm(c_real.float(), dim=-1)
        ),
        "C_B": _stats(qk),
        "dt": _stats(dt),
        "A": _stats(a),
        "alpha": _stats(alpha),
        "beta": _stats(beta),
        "gamma": _stats(gamma),
        "gamma_C_B": _stats(gamma.unsqueeze(-2) * qk),
        "beta_C_B": _stats(beta.unsqueeze(-2) * qk),
        "value": _stats(value),
        "gate": _stats(torch.nn.functional.silu(z)),
    }


def _is_mamba3(module: nn.Module) -> bool:
    return all(
        hasattr(module, name)
        for name in ("num_rope_angles", "mimo_rank", "B_norm", "C_norm", "dt_bias")
    )


def _capture_first_operand(destination: dict[str, Tensor]):
    def hook(_module: nn.Module, operands: tuple[Tensor, ...]) -> None:
        destination["value"] = operands[0].detach()

    return hook


@torch.no_grad()
def _activation_probe(
    model: nn.Module, batch: Batch, classes: int, *, deep: bool
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    hidden = model.embedding(batch.inputs).to(model.head.weight.dtype)
    layers: list[dict[str, Any]] = []
    try:
        from slinoss.mixer import SLinOSSMixer
    except (ImportError, ModuleNotFoundError):
        SLinOSSMixer = ()  # type: ignore[assignment,misc]
    for index, block in enumerate(model.blocks):
        before = hidden
        normed = _rmsnorm(before, block.norm_weight, block.norm_eps)
        inner = getattr(block.mixer, "inner", block.mixer)
        out_proj = getattr(inner, "out_proj", None)
        captured: dict[str, Tensor] = {}
        handle = None
        if isinstance(out_proj, nn.Linear):
            handle = out_proj.register_forward_pre_hook(
                _capture_first_operand(captured)
            )
        try:
            mixed = _unwrap(block.mixer(normed))
        finally:
            if handle is not None:
                handle.remove()
        hidden = before + mixed
        row: dict[str, Any] = {
            "layer": index,
            "input": _stats(before),
            "normed": _stats(normed),
            "mixed": _stats(mixed),
            "output": _stats(hidden),
            "input_mixer_cosine": _cosine(before, mixed),
        }
        if "value" in captured:
            row["pre_out_proj"] = _stats(captured["value"])
        if isinstance(out_proj, nn.Linear):
            row["out_proj_weight"] = _stats(out_proj.weight)
        if deep and isinstance(block.mixer, SLinOSSMixer):
            row["slinoss"] = _inspect_current_mixer(block.mixer, normed[:1])
        if deep and _is_mamba3(inner):
            row["mamba3"] = _inspect_mamba3_mixer(block.mixer, normed[:1])
        layers.append(row)
    final = _rmsnorm(hidden, model.norm_weight, model.config.norm_eps)
    probe_loss = _loss(model, batch, classes)
    model.train(was_training)
    return {"loss": float(probe_loss), "final_hidden": _stats(final), "layers": layers}


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    shard: Shard,
    *,
    seq_len: int,
    batch_size: int,
    classes: int,
    device: str,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    for batch in val_batches(shard, seq_len=seq_len, batch_size=batch_size):
        batch = batch.to(device)
        count = batch.targets.numel()
        total += float(_loss(model, batch, classes)) * count
        tokens += count
    model.train(was_training)
    return total / tokens, tokens


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old-root", type=Path)
    parser.add_argument("--old-commit", default="")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--token-batch", type=int, default=131072)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=4e-3)
    parser.add_argument("--embedding-base-lr", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--abort-grad-norm", type=float, default=math.inf)
    parser.add_argument("--warmdown-fraction", type=float, default=0.4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.token_batch % (args.micro_batch * args.seq_len):
        raise ValueError("token batch must be a whole number of micro batches")
    accum = args.token_batch // (args.micro_batch * args.seq_len)
    manifest = corpus_mod.read_manifest(args.corpus)
    train_shard = Shard(
        corpus_mod.shard_path(args.corpus, "train"), manifest.train.tokens
    )
    val_shard = Shard(corpus_mod.shard_path(args.corpus, "val"), manifest.val.tokens)
    probe = next(val_batches(val_shard, seq_len=args.seq_len, batch_size=1)).to(
        args.device
    )

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    if args.arm == "old-v2x2":
        if args.old_root is None:
            raise ValueError("--old-root is required for old-v2x2")
        if not args.old_commit:
            raise ValueError("--old-commit is required for old-v2x2")
        model, construction = _build_old(
            old_root=args.old_root,
            old_commit=args.old_commit,
            d_model=args.d_model,
            n_layers=args.n_layers,
            vocab_size=manifest.vocab_size,
            device=args.device,
        )
    else:
        model, construction = _build_current(
            arm=args.arm,
            d_model=args.d_model,
            n_layers=args.n_layers,
            vocab_size=manifest.vocab_size,
            device=args.device,
        )

    peak_lr = transfer(args.base_lr, d_model=args.d_model, token_batch=args.token_batch)
    embedding_lr = transfer(
        args.embedding_base_lr, d_model=args.d_model, token_batch=args.token_batch
    )
    policy = GroupPolicy(
        lr=peak_lr, embedding_lr=embedding_lr, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, policy),
        lr=peak_lr,
        betas=(0.8, 0.95),
        eps=1e-10,
        fused=False,
    )
    peaks = [float(group["lr"]) for group in optimizer.param_groups]
    stream = batches(
        train_shard,
        seq_len=args.seq_len,
        batch_size=args.micro_batch,
        seed=args.seed,
        steps=args.steps * accum,
    )

    payload: dict[str, Any] = {
        "status": "running",
        "arm": args.arm,
        "construction": construction,
        "protocol": {
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "seq_len": args.seq_len,
            "micro_batch": args.micro_batch,
            "token_batch": args.token_batch,
            "steps": args.steps,
            "tokens": args.steps * args.token_batch,
            "seed": args.seed,
            "base_lr": args.base_lr,
            "peak_lr": peak_lr,
            "embedding_base_lr": args.embedding_base_lr,
            "embedding_lr": embedding_lr,
            "weight_decay": args.weight_decay,
            "betas": [0.8, 0.95],
            "eps": 1e-10,
            "grad_clip": args.grad_clip,
            "warmdown_fraction": args.warmdown_fraction,
            "precision": "fp32-compute/bf16-token-embedding",
            "loss": "explicit logsumexp over logical vocabulary",
        },
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "group_parameters": group_counts(model),
        "data": {
            "tokenizer": manifest.tokenizer,
            "train_sha256": manifest.train.digest,
            "val_sha256": manifest.val.digest,
            "vocab_size": manifest.vocab_size,
        },
        "provenance": {
            "diagnostic_commit": args.source_commit,
            "baseline_commit": args.baseline_commit,
            "diagnostic_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "argv": list(sys.argv),
        },
        "initial_parameters": _parameter_metrics(model),
        "snapshots": {
            "-1": _activation_probe(model, probe, manifest.vocab_size, deep=True)
        },
        "steps": [],
        "gradient_snapshots": {},
    }
    _write(args.out, payload)

    model.train()
    recent: list[float] = []
    clipped = 0
    clip_scales: list[float] = []
    training_seconds = 0.0
    telemetry_seconds = 0.0
    torch.cuda.synchronize()
    for step in range(args.steps):
        factor = (
            lr_at(
                step,
                total_steps=args.steps,
                peak_lr=peak_lr,
                warmdown_fraction=args.warmdown_fraction,
            )
            / peak_lr
        )
        for group, base in zip(optimizer.param_groups, peaks, strict=True):
            group["lr"] = base * factor

        train_start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        mean = 0.0
        for _ in range(accum):
            batch = next(stream).to(args.device)
            loss = _loss(model, batch, manifest.vocab_size)
            (loss / accum).backward()
            mean += float(loss.detach()) / accum
        if step in SNAPSHOT_STEPS:
            payload["gradient_snapshots"][str(step)] = _gradient_metrics(model)
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        scale = min(1.0, args.grad_clip / max(grad_norm, 1e-30))
        if not math.isfinite(grad_norm) or grad_norm >= args.abort_grad_norm:
            payload["steps"].append(
                {
                    "step": step,
                    "loss": mean,
                    "lr": peak_lr * factor,
                    "grad_norm_preclip": grad_norm,
                    "clip_scale": scale,
                }
            )
            payload.update(
                {
                    "status": "aborted",
                    "abort_reason": (
                        f"pre-clip gradient norm {grad_norm} reached "
                        f"threshold {args.abort_grad_norm}"
                    ),
                    "training_seconds": training_seconds,
                    "telemetry_seconds": telemetry_seconds,
                }
            )
            _write(args.out, payload)
            print(f"ABORT arm={args.arm} step={step} grad={grad_norm:.3e}", flush=True)
            return 2
        clipped += int(scale < 1.0)
        clip_scales.append(scale)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - train_start

        recent.append(mean)
        del recent[:-20]
        payload["steps"].append(
            {
                "step": step,
                "loss": mean,
                "lr": peak_lr * factor,
                "grad_norm_preclip": grad_norm,
                "clip_scale": scale,
            }
        )
        if step in SNAPSHOT_STEPS:
            telemetry_start = time.perf_counter()
            payload["snapshots"][str(step)] = _activation_probe(
                model, probe, manifest.vocab_size, deep=step in DEEP_STEPS
            )
            torch.cuda.synchronize()
            telemetry_seconds += time.perf_counter() - telemetry_start
            payload["training_seconds"] = training_seconds
            payload["telemetry_seconds"] = telemetry_seconds
            _write(args.out, payload)
        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"{args.arm} step={step:03d} loss={mean:.6f} "
                f"grad={grad_norm:.3e} clip={scale:.3e}",
                flush=True,
            )

    validation_start = time.perf_counter()
    val_loss, val_tokens = _evaluate(
        model,
        val_shard,
        seq_len=args.seq_len,
        batch_size=args.eval_batch,
        classes=manifest.vocab_size,
        device=args.device,
    )
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - validation_start
    payload.update(
        {
            "status": "complete",
            "train_loss": sum(recent) / len(recent),
            "val_loss": val_loss,
            "val_bpb": val_loss * _LOG2E / manifest.val_bytes_per_token,
            "val_tokens": val_tokens,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "telemetry_seconds": telemetry_seconds,
            "tokens_per_second": args.steps * args.token_batch / training_seconds,
            "clipping": {
                "steps_clipped": clipped,
                "fraction": clipped / args.steps,
                "minimum_scale": min(clip_scales),
                "geometric_mean_scale": math.exp(
                    sum(math.log(max(value, 1e-300)) for value in clip_scales)
                    / len(clip_scales)
                ),
            },
            "final_parameters": _parameter_metrics(model),
        }
    )
    _write(args.out, payload)
    print(
        f"RESULT arm={args.arm} val_loss={val_loss:.6f} "
        f"val_bpb={payload['val_bpb']:.6f} tok_s={payload['tokens_per_second']:.1f} "
        f"clip_fraction={payload['clipping']['fraction']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
