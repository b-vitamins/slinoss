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
            methodology = workload.get("methodology")
            if methodology is not None and not isinstance(methodology, dict):
                raise ValueError(
                    f"Workload {case_name}/{backend_name} methodology must be a dict."
                )

            warm = _expect_dict(workload, "warm")
            cold = _expect_dict(workload, "cold")
            for section in (warm, cold):
                _expect(section, "budget")
                tree = _expect_dict(section, "tree")
                _expect_dict(section, "regions")
                _expect_dict(section, "cache_events")
                _expect_path(tree, "step.__stats__")
                _expect_path(tree, "forward.__stats__")
                _expect_path(tree, "backward.__stats__")
                _expect_path(tree, "forward.v2x2ssd.__stats__")
                _expect_path(tree, "backward.v2x2ssd.__stats__")
                _expect_path(tree, "forward.other.__stats__")
                _expect_path(tree, "backward.other.__stats__")
                _expect_path(tree, "forward.other.unattributed.__stats__")
                _expect_path(tree, "backward.other.unattributed.__stats__")
                _expect_path(tree, "forward.mixer.__stats__")
                _expect_path(tree, "backward.mixer.__stats__")
                _expect_path(tree, "forward.mixer.scanprep.__stats__")
                _expect_path(tree, "backward.mixer.scanprep.__stats__")
                _expect_path(tree, "forward.embed.__stats__")
                _expect_path(tree, "backward.embed.__stats__")
                _expect_path(tree, "forward.norms.__stats__")
                _expect_path(tree, "backward.norms.__stats__")
                _expect_path(tree, "forward.head.__stats__")
                _expect_path(tree, "backward.head.__stats__")

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
            _expect_path(warm["tree"], "backward.v2x2ssd.chunk_increment.__stats__")
            _expect_path(warm["tree"], "backward.v2x2ssd.state_passing.__stats__")
            _expect_path(warm["tree"], "backward.v2x2ssd.chunk_scan.__stats__")


def validate_nextchar_profile_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict.")
    if _expect(payload, "kind") != "profile_nextchar":
        raise ValueError("Expected kind=profile_nextchar.")
    if int(_expect(payload, "schema_version")) != 1:
        raise ValueError("Unsupported schema_version.")
    _expect(payload, "backend")
    _expect(payload, "config")
    _expect(payload, "regions")
    _expect(payload, "budget")
    tree = _expect_dict(payload, "tree")
    _expect_path(tree, "step.__stats__")
    _expect_path(tree, "forward.__stats__")
    _expect_path(tree, "backward.__stats__")
    _expect_path(tree, "forward.mixer.scanprep.__stats__")
    _expect_path(tree, "backward.mixer.scanprep.__stats__")


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
