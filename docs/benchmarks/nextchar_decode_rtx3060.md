# NextChar Decode Benchmarks (RTX 3060)

Device:

- `NVIDIA GeForce RTX 3060`
- `sm_86`

Model family:

- `vocab_size=4096`
- `block_size=512`
- `d_model=256`
- `n_layers=6`
- `d_state=64`
- `expand=2`
- `d_head=64`
- `d_conv=4`
- `chunk_size=32`
- `dtype=fp16`

Method:

- `scripts/perf/bench_nextchar_decode.py`
- `warmup_tokens=16`
- `active_tokens=256`
- `repeat=5`
- `backend=cute`

Results:

| B | persistent us/token | eager us/token | speedup | t_lower us/token | efficiency |
|---:|---:|---:|---:|---:|---:|
| 1 | 430.089 | 3406.004 | 7.919x | 4.599 | 0.011 |
| 2 | 288.397 | 1872.520 | 6.493x | 9.198 | 0.032 |
| 4 | 164.809 | 928.797 | 5.636x | 18.394 | 0.112 |
| 8 | 98.186 | 462.517 | 4.711x | 36.786 | 0.375 |
| 16 | 85.160 | 233.142 | 2.738x | 73.570 | 0.864 |

Interpretation:

- The persistent path is materially faster than the eager token loop across the
  entire supported batch grid.
- The local lower-bound efficiency improves with batch size and gets close to
  the bound at `B=16`.
- Small batches are far from the bound on this GPU. The eager decode profile
  shows that the remaining time is dominated by CUDA GEMM (`aten::mm`,
  `aten::addmm`) and depthwise conv (`aten::cudnn_convolution`) kernels rather
  than by the decode graph plumbing itself.
- No H100 was available in the local environment, so this table is not an H100
  claim. The decode harness accepts an H100 preset for target-platform runs.
