from __future__ import annotations

from slinoss.perf.budget import build_tree, derive_nextchar_budget
from slinoss.perf.schema import (
    validate_nextchar_bench_payload,
    validate_nextchar_profile_payload,
)


def _sample_regions() -> dict[str, float]:
    return {
        "step.forward_loss": 10.0,
        "step.backward": 20.0,
        "step.clip": 1.0,
        "step.optim": 2.0,
        "step.zero_grad": 0.5,
        "forward.v2x2ssd.total": 3.0,
        "backward.v2x2ssd.total": 8.0,
        "forward.embed.token": 1.0,
        "forward.embed.pos": 0.5,
        "backward.embed.token": 0.75,
        "backward.embed.pos": 0.25,
        "forward.norms.pre_mixer": 0.1,
        "forward.norms.pre_ffn": 0.2,
        "forward.norms.final": 0.3,
        "backward.norms.pre_mixer": 0.4,
        "backward.norms.pre_ffn": 0.5,
        "backward.norms.final": 0.6,
        "forward.mixer.in_proj": 0.7,
        "forward.mixer.dw_conv": 0.8,
        "forward.mixer.bc_emit": 0.9,
        "forward.mixer.dw_conv_activation": 0.25,
        "forward.mixer.scanprep.total": 2.25,
        "forward.mixer.scanprep.pack_u": 0.4,
        "forward.mixer.scanprep.bc_norm": 0.15,
        "forward.mixer.scanprep.coefficients": 1.0,
        "forward.mixer.scanprep.pack_bc": 0.7,
        "forward.mixer.gate_skip": 1.2,
        "forward.mixer.out_proj": 0.3,
        "backward.mixer.in_proj": 1.3,
        "backward.mixer.dw_conv": 1.4,
        "backward.mixer.bc_emit": 1.5,
        "backward.mixer.dw_conv_activation": 0.45,
        "backward.mixer.scanprep.total": 3.65,
        "backward.mixer.scanprep.pack_u": 0.5,
        "backward.mixer.scanprep.bc_norm": 0.35,
        "backward.mixer.scanprep.coefficients": 1.6,
        "backward.mixer.scanprep.pack_bc": 1.2,
        "backward.mixer.gate_skip": 1.8,
        "backward.mixer.out_proj": 0.4,
        "forward.ffn": 0.9,
        "backward.ffn": 1.9,
        "forward.residual.mixer": 0.11,
        "forward.residual.ffn": 0.12,
        "backward.residual.mixer": 0.21,
        "backward.residual.ffn": 0.22,
        "forward.head.logits": 0.4,
        "forward.head.loss": 0.2,
        "backward.head.logits": 0.6,
        "backward.head.loss": 0.7,
        "forward.v2x2ssd.chunk_increment.total": 1.0,
        "forward.v2x2ssd.state_passing.total": 0.5,
        "forward.v2x2ssd.chunk_scan.total": 1.25,
        "backward.v2x2ssd.chunk_increment.total": 2.0,
        "backward.v2x2ssd.state_passing.total": 1.0,
        "backward.v2x2ssd.chunk_scan.total": 3.0,
        "backward.v2x2ssd.chunk_increment.db": 0.4,
        "backward.v2x2ssd.chunk_increment.du": 0.5,
        "backward.v2x2ssd.chunk_increment.boundary": 0.3,
        "backward.v2x2ssd.chunk_increment.param_scan": 0.2,
        "backward.v2x2ssd.state_passing.state": 0.4,
        "backward.v2x2ssd.state_passing.m": 0.2,
        "backward.v2x2ssd.chunk_scan.dz0": 0.3,
        "backward.v2x2ssd.chunk_scan.du": 0.4,
        "backward.v2x2ssd.chunk_scan.db": 0.5,
        "backward.v2x2ssd.chunk_scan.dcdr": 0.6,
        "backward.v2x2ssd.chunk_scan.param_scan": 0.7,
    }


def _sample_tree() -> dict[str, object]:
    derived = derive_nextchar_budget(_sample_regions())
    summaries = {label: {"mean_ms": value} for label, value in derived.items()}
    return build_tree(summaries)


def test_validate_nextchar_bench_payload_accepts_expected_schema() -> None:
    tree = _sample_tree()
    payload = {
        "kind": "bench_nextchar",
        "schema_version": 1,
        "device_name": "Fake GPU",
        "suite": "single",
        "cases": {
            "default": {
                "config": {"batch_size": 4},
                "workload": {
                    "cute": {
                        "backend": "cute",
                        "config": {"batch_size": 4},
                        "tokens_per_step": 256,
                        "methodology": {
                            "deterministic_fixture": True,
                            "fixture_model_seed": 0,
                            "fixture_batch_seed": 1,
                            "warmup_steps": 10,
                            "steps_per_repeat": 20,
                            "workload_repeat": 5,
                        },
                        "cold": {
                            "regions": {},
                            "budget": {},
                            "tree": tree,
                            "cache_events": {},
                        },
                        "warm": {
                            "step": {"mean_ms": 10.0},
                            "repeat_step": {"mean_ms": 10.2},
                            "tokens_per_s": {"mean": 1000.0},
                            "repeat_tokens_per_s": {"mean": 980.0},
                            "regions": {},
                            "budget": {},
                            "tree": tree,
                            "cache_events": {},
                        },
                    }
                },
                "stage_suite": {"rows": [], "config": {}},
            }
        },
    }
    validate_nextchar_bench_payload(payload)


def test_validate_nextchar_profile_payload_accepts_expected_schema() -> None:
    payload = {
        "kind": "profile_nextchar",
        "schema_version": 1,
        "backend": "cute",
        "config": {"batch_size": 4},
        "regions": {},
        "budget": {},
        "tree": _sample_tree(),
        "trace_out": None,
    }
    validate_nextchar_profile_payload(payload)
