from __future__ import annotations

from slinoss.perf.budget import build_tree, derive_training_budget
from slinoss.perf.schema import (
    validate_training_bench_payload,
    validate_training_memory_payload,
    validate_training_profile_payload,
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
        "forward.mixer.scanprep.total": 2.25,
        "forward.mixer.scanprep.pack_u": 0.4,
        "forward.mixer.scanprep.bc_norm": 0.15,
        "forward.mixer.scanprep.coefficients": 1.0,
        "forward.mixer.scanprep.pack_bc": 0.7,
        "forward.mixer.tail": 1.5,
        "backward.mixer.in_proj": 1.3,
        "backward.mixer.dw_conv": 1.4,
        "backward.mixer.bc_emit": 1.5,
        "backward.mixer.scanprep.total": 3.65,
        "backward.mixer.scanprep.pack_u": 0.5,
        "backward.mixer.scanprep.bc_norm": 0.35,
        "backward.mixer.scanprep.coefficients": 1.6,
        "backward.mixer.scanprep.pack_bc": 1.2,
        "backward.mixer.tail": 2.2,
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
    }


def _sample_tree() -> dict[str, object]:
    derived = derive_training_budget(_sample_regions())
    summaries = {label: {"mean_ms": value} for label, value in derived.items()}
    return build_tree(summaries)


def test_validate_training_bench_payload_accepts_expected_schema() -> None:
    tree = _sample_tree()
    payload = {
        "kind": "bench_training",
        "schema_version": 2,
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
                            "timing": "cuda_event_per_step",
                            "deterministic_fixture": True,
                            "fixture_model_seed": 0,
                            "fixture_batch_seed": 1,
                            "batch_count": 20,
                            "warmup_steps": 10,
                            "steps_per_repeat": 20,
                            "workload_repeat": 5,
                            "warm_execution": "bench_loop",
                            "profile_execution": "eager_single_post_bench_replay",
                            "memory_measurement": "bench_path_step_peaks",
                            "memory_forensics": "use profile_training_memory.py for eager attribution",
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
                "v2x2ssd_suite": {"rows": [], "config": {}},
            }
        },
    }
    validate_training_bench_payload(payload)


def test_validate_training_profile_payload_accepts_expected_schema() -> None:
    payload = {
        "kind": "profile_training",
        "schema_version": 1,
        "backend": "cute",
        "config": {"batch_size": 4},
        "methodology": {
            "execution": "eager_profiled_training_step",
            "memory_mode": "torch_profiler_profile_memory",
        },
        "regions": {},
        "budget": {},
        "tree": _sample_tree(),
        "trace_out": None,
    }
    validate_training_profile_payload(payload)


def test_validate_training_memory_payload_accepts_expected_schema() -> None:
    payload = {
        "kind": "profile_training_memory",
        "schema_version": 1,
        "backend": "cute",
        "config": {"batch_size": 4},
        "methodology": {
            "execution": "eager_training_step",
            "baseline_scope": "warmed_model_plus_inputs",
            "warmup_steps": 1,
            "top_k": 20,
            "memory_metric_primary": "peak_allocated_bytes",
            "allocator_snapshot_requested": False,
        },
        "baseline_memory": {
            "allocated_bytes": 1,
            "reserved_bytes": 2,
        },
        "step_memory": {
            "peak_allocated_bytes": 3,
            "peak_reserved_bytes": 4,
            "end_allocated_bytes": 5,
            "end_reserved_bytes": 6,
        },
        "regions": {},
        "budget": {},
        "tree": _sample_tree(),
        "top_region_exit_allocated": [
            {
                "label": "forward.head.loss",
                "max_allocated_bytes": 7,
                "max_reserved_bytes": 8,
                "num_exits": 1,
            }
        ],
        "saved_tensors_by_region": [
            {
                "label": "forward.ffn",
                "unique_saved_bytes": 9,
                "unique_storage_count": 1,
                "save_event_count": 2,
            }
        ],
        "saved_tensors_summary": {
            "accounting": "unique_storage_first_save",
            "total_unique_saved_bytes": 9,
            "total_unique_storage_count": 1,
            "total_save_event_count": 2,
        },
        "allocator_snapshot": {
            "requested": False,
            "captured": False,
            "path": None,
            "format": None,
        },
    }
    validate_training_memory_payload(payload)


def test_validate_training_memory_payload_requires_full_methodology() -> None:
    payload = {
        "kind": "profile_training_memory",
        "schema_version": 1,
        "backend": "cute",
        "config": {"batch_size": 4},
        "methodology": {
            "execution": "eager_training_step",
        },
        "baseline_memory": {
            "allocated_bytes": 1,
            "reserved_bytes": 2,
        },
        "step_memory": {
            "peak_allocated_bytes": 3,
            "peak_reserved_bytes": 4,
            "end_allocated_bytes": 5,
            "end_reserved_bytes": 6,
        },
        "regions": {},
        "budget": {},
        "tree": _sample_tree(),
        "top_region_exit_allocated": [],
        "saved_tensors_by_region": [],
        "saved_tensors_summary": {
            "accounting": "unique_storage_first_save",
            "total_unique_saved_bytes": 0,
            "total_unique_storage_count": 0,
            "total_save_event_count": 0,
        },
        "allocator_snapshot": {
            "requested": False,
            "captured": False,
            "path": None,
            "format": None,
        },
    }

    try:
        validate_training_memory_payload(payload)
    except ValueError as exc:
        assert "baseline_scope" in str(exc)
    else:
        raise AssertionError("Expected validate_training_memory_payload to fail.")
