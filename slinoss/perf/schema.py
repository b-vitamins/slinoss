"""Schema validation for perf harness payloads."""

from __future__ import annotations

from typing import Any


def _expect(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing key: {key}")
    return mapping[key]


def _expect_dict(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _expect(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict at key: {key}")
    return value


def _expect_path(root: dict[str, Any], path: str) -> Any:
    node: Any = root
    parts = path.split(".")
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Missing path: {'.'.join(traversed)}")
        node = node[part]
    return node


def _validate_memory_summary(memory: dict[str, Any]) -> None:
    for label in ("peak_allocated_bytes", "peak_reserved_bytes"):
        summary = _expect_dict(memory, label)
        for key in (
            "mean_bytes",
            "median_bytes",
            "min_bytes",
            "max_bytes",
            "stdev_bytes",
            "num_samples",
        ):
            _expect(summary, key)


def _validate_nextchar_budget_tree(
    tree: dict[str, Any],
    *,
    require_stage_breakdown: bool,
) -> None:
    for path in (
        "step.__stats__",
        "forward.__stats__",
        "backward.__stats__",
        "forward.v2x2ssd.__stats__",
        "backward.v2x2ssd.__stats__",
        "forward.other.__stats__",
        "backward.other.__stats__",
        "forward.other.unattributed.__stats__",
        "backward.other.unattributed.__stats__",
        "forward.mixer.__stats__",
        "backward.mixer.__stats__",
        "forward.mixer.in_proj.__stats__",
        "backward.mixer.in_proj.__stats__",
        "forward.mixer.dw_conv.__stats__",
        "backward.mixer.dw_conv.__stats__",
        "forward.mixer.dw_conv_activation.__stats__",
        "backward.mixer.dw_conv_activation.__stats__",
        "forward.mixer.bc_emit.__stats__",
        "backward.mixer.bc_emit.__stats__",
        "forward.mixer.scanprep.__stats__",
        "backward.mixer.scanprep.__stats__",
        "forward.mixer.tail.__stats__",
        "backward.mixer.tail.__stats__",
        "forward.embed.__stats__",
        "backward.embed.__stats__",
        "forward.norms.__stats__",
        "backward.norms.__stats__",
        "forward.ffn.__stats__",
        "backward.ffn.__stats__",
        "forward.residual.__stats__",
        "backward.residual.__stats__",
        "forward.head.__stats__",
        "backward.head.__stats__",
    ):
        _expect_path(tree, path)
    if require_stage_breakdown:
        _expect_path(tree, "backward.v2x2ssd.chunk_increment.__stats__")
        _expect_path(tree, "backward.v2x2ssd.state_passing.__stats__")
        _expect_path(tree, "backward.v2x2ssd.chunk_scan.__stats__")


def validate_nextchar_bench_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "bench_nextchar":
        raise ValueError("Expected kind=bench_nextchar.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "device_name")

    cases = _expect_dict(payload, "cases")
    if not cases:
        raise ValueError("Expected at least one case.")

    for case_name, case_payload in cases.items():
        if not isinstance(case_payload, dict):
            raise ValueError(f"Case {case_name} must be a dict.")
        _expect(case_payload, "config")
        workloads = _expect_dict(case_payload, "workload")
        _expect(case_payload, "stage_suite")
        for backend_name, workload in workloads.items():
            if backend_name not in {"reference", "cute"}:
                raise ValueError(f"Unsupported backend key: {backend_name}")
            if not isinstance(workload, dict):
                raise ValueError(f"Workload {case_name}/{backend_name} must be a dict.")
            _expect(workload, "backend")
            _expect(workload, "config")
            _expect(workload, "tokens_per_step")
            methodology = _expect_dict(workload, "methodology")
            for key in (
                "timing",
                "deterministic_fixture",
                "fixture_model_seed",
                "fixture_batch_seed",
                "batch_count",
                "warmup_steps",
                "steps_per_repeat",
                "workload_repeat",
                "warm_execution",
                "profile_execution",
                "memory_measurement",
                "memory_forensics",
            ):
                _expect(methodology, key)

            warm = _expect_dict(workload, "warm")
            cold = _expect_dict(workload, "cold")
            for section in (warm, cold):
                _expect(section, "budget")
                tree = _expect_dict(section, "tree")
                _expect_dict(section, "regions")
                _expect_dict(section, "cache_events")
                memory = section.get("memory")
                if memory is not None:
                    if not isinstance(memory, dict):
                        raise ValueError(
                            f"Memory summary for {case_name}/{backend_name} must be a dict."
                        )
                    _validate_memory_summary(memory)
                _validate_nextchar_budget_tree(
                    tree,
                    require_stage_breakdown=False,
                )

            _expect_dict(warm, "step")
            _expect_dict(warm, "tokens_per_s")
            repeat_step = warm.get("repeat_step")
            if repeat_step is not None and not isinstance(repeat_step, dict):
                raise ValueError(
                    f"Warm repeat_step for {case_name}/{backend_name} must be a dict."
                )
            repeat_tps = warm.get("repeat_tokens_per_s")
            if repeat_tps is not None and not isinstance(repeat_tps, dict):
                raise ValueError(
                    f"Warm repeat_tokens_per_s for {case_name}/{backend_name} must be a dict."
                )
            _validate_nextchar_budget_tree(
                warm["tree"],
                require_stage_breakdown=True,
            )


def validate_nextchar_profile_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "profile_nextchar":
        raise ValueError("Expected kind=profile_nextchar.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "backend")
    _expect(payload, "config")
    methodology = _expect_dict(payload, "methodology")
    _expect(methodology, "execution")
    _expect(methodology, "memory_mode")
    _expect(payload, "regions")
    _expect(payload, "budget")
    tree = _expect_dict(payload, "tree")
    _validate_nextchar_budget_tree(tree, require_stage_breakdown=False)


def validate_nextchar_memory_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "profile_nextchar_memory":
        raise ValueError("Expected kind=profile_nextchar_memory.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "backend")
    _expect(payload, "config")
    methodology = _expect_dict(payload, "methodology")
    for key in (
        "execution",
        "baseline_scope",
        "warmup_steps",
        "top_k",
        "memory_metric_primary",
        "allocator_snapshot_requested",
    ):
        _expect(methodology, key)
    _expect(payload, "baseline_memory")
    _expect(payload, "step_memory")
    _expect(payload, "regions")
    _expect(payload, "budget")
    tree = _expect_dict(payload, "tree")
    _validate_nextchar_budget_tree(tree, require_stage_breakdown=False)

    baseline = _expect_dict(payload, "baseline_memory")
    _expect(baseline, "allocated_bytes")
    _expect(baseline, "reserved_bytes")

    step_memory = _expect_dict(payload, "step_memory")
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "end_allocated_bytes",
        "end_reserved_bytes",
    ):
        _expect(step_memory, key)

    top_region_exit_allocated = _expect(payload, "top_region_exit_allocated")
    if not isinstance(top_region_exit_allocated, list):
        raise ValueError("Expected top_region_exit_allocated to be a list.")
    for row in top_region_exit_allocated:
        if not isinstance(row, dict):
            raise ValueError("Each top_region_exit_allocated row must be a dict.")
        for key in (
            "label",
            "max_allocated_bytes",
            "max_reserved_bytes",
            "num_exits",
        ):
            _expect(row, key)

    saved_tensors = _expect(payload, "saved_tensors_by_region")
    if not isinstance(saved_tensors, list):
        raise ValueError("Expected saved_tensors_by_region to be a list.")
    for row in saved_tensors:
        if not isinstance(row, dict):
            raise ValueError("Each saved_tensors_by_region row must be a dict.")
        for key in (
            "label",
            "unique_saved_bytes",
            "unique_storage_count",
            "save_event_count",
        ):
            _expect(row, key)

    saved_summary = _expect_dict(payload, "saved_tensors_summary")
    for key in (
        "accounting",
        "total_unique_saved_bytes",
        "total_unique_storage_count",
        "total_save_event_count",
    ):
        _expect(saved_summary, key)

    allocator_snapshot = _expect_dict(payload, "allocator_snapshot")
    for key in ("requested", "captured", "path", "format"):
        _expect(allocator_snapshot, key)


def validate_nextchar_decode_bench_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "bench_nextchar_decode":
        raise ValueError("Expected kind=bench_nextchar_decode.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "backend")
    _expect(payload, "device_name")
    rows = _expect(payload, "rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected non-empty rows list.")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each decode bench row must be a dict.")
        _expect(row, "batch_size")
        _expect_dict(_expect_dict(row, "persistent"), "summary")
        _expect_dict(_expect_dict(row, "eager"), "summary")


def validate_nextchar_decode_profile_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "profile_nextchar_decode":
        raise ValueError("Expected kind=profile_nextchar_decode.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "backend")
    _expect(payload, "mode")
    _expect(payload, "config")
