from __future__ import annotations

import pytest

from scripts.perf.ncu_report import parse_ncu_launch_summaries, parse_ncu_summary


def test_parse_ncu_launch_summaries_splits_multi_launch_output() -> None:
    output = """
[123] python@127.0.0.1
  kernel_cutlass_auxiliary_example_0 (1, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         us       325.76
    DRAM Throughput                   %        59.03
    ----------------------- ----------- ------------
    Section: Launch Statistics
    -------------------------------- --------------- ------------
    Metric Name                      Metric Unit     Metric Value
    -------------------------------- --------------- ------------
    Registers Per Thread             register/thread          137
    Dynamic Shared Memory Per Block  Kbyte/block            7.06
    Static Shared Memory Per Block   byte/block                0
    Driver Shared Memory Per Block   Kbyte/block            1.02
    -------------------------------- --------------- ------------
  kernel_cutlass_kernel_example_1 (1, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    Duration                         ms         3.63
    DRAM Throughput                   %        49.62
    ----------------------- ----------- ------------
    Section: Launch Statistics
    -------------------------------- --------------- ------------
    Metric Name                      Metric Unit     Metric Value
    -------------------------------- --------------- ------------
    Registers Per Thread             register/thread          160
    Dynamic Shared Memory Per Block  Kbyte/block           24.90
    Static Shared Memory Per Block   byte/block                0
    Driver Shared Memory Per Block   Kbyte/block            1.02
    -------------------------------- --------------- ------------
"""

    launches = parse_ncu_launch_summaries(output)

    assert len(launches) == 2
    assert launches[0]["kernel_label"] == "launch_0"
    assert launches[0]["duration_ms"] == 0.32576
    assert launches[0]["duration_source"] == "ncu_replay_profiled"
    assert launches[0]["registers_per_thread"] == 137
    assert launches[0]["smem_total_kib"] == 8.08
    assert launches[1]["kernel_label"] == "main"
    assert launches[1]["duration_ms"] == 3.63
    assert launches[1]["duration_source"] == "ncu_replay_profiled"
    assert launches[1]["registers_per_thread"] == 160
    assert launches[1]["smem_total_kib"] == pytest.approx(25.92)


def test_parse_ncu_summary_uses_first_flat_metric_match() -> None:
    output = """
    Registers Per Thread             register/thread          137
    Registers Per Thread             register/thread          160
"""

    summary = parse_ncu_summary(output)

    assert summary["registers_per_thread"] == 137


def test_parse_ncu_summary_converts_raw_duration_metric_units() -> None:
    output = """
    gpu__time_duration.sum       325.76 us
"""

    summary = parse_ncu_summary(output)

    assert summary["duration_ms"] == pytest.approx(0.32576)
