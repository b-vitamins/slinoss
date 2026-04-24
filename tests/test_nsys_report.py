from __future__ import annotations

from scripts.perf.nsys_report import parse_nsys_cuda_gpu_trace_csv


def test_parse_nsys_cuda_gpu_trace_csv_keeps_chronological_launch_order() -> None:
    csv_text = """Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,Device,Ctx,GreenCtx,Strm,Name
9961685,2707098,67,1,1,2304,128,1,1,160,0.000,0.025,,,,,GPU,1,,7,kernel_cutlass_kernel_example_1
9695093,265888,65,1,1,2304,128,1,1,137,0.000,0.007,,,,,GPU,1,,7,kernel_cutlass_auxiliary_example_0
"""

    launches = parse_nsys_cuda_gpu_trace_csv(csv_text)

    assert len(launches) == 2
    assert launches[0]["kernel_label"] == "launch_0"
    assert launches[0]["duration_ms"] == 0.265888
    assert launches[0]["duration_source"] == "nsys_cuda_gpu_trace_hot"
    assert launches[0]["registers_per_thread"] == 137
    assert launches[1]["kernel_label"] == "main"
    assert launches[1]["duration_ms"] == 2.707098
    assert launches[1]["registers_per_thread"] == 160
