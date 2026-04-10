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
        "forward.mixer.gate.__stats__",
        "backward.mixer.gate.__stats__",
        "forward.mixer.out_proj.__stats__",
        "backward.mixer.out_proj.__stats__",
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
    _expect(payload, "regions")
    _expect(payload, "budget")
    tree = _expect_dict(payload, "tree")
    _validate_nextchar_budget_tree(tree, require_stage_breakdown=False)


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
