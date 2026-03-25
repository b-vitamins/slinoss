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
| 1 | 299.170 | 3312.410 | 11.072x | 4.599 | 0.015 |
| 2 | 212.030 | 1915.486 | 9.034x | 9.198 | 0.043 |
| 4 | 112.643 | 900.226 | 7.992x | 18.394 | 0.163 |
| 8 | 66.540 | 455.699 | 6.848x | 36.786 | 0.553 |
| 16 | 37.237 | 223.092 | 5.991x | 73.570 | 1.976 |

Interpretation:

- The persistent path is materially faster than the eager token loop across the
  entire supported batch grid.
- Relative to the `split-N=2` decode checkpoint, persistent decode improved by
  about `1.12x` at `B=1`, `1.11x` at `B=2`, `1.02x` at `B=4`, and stayed
  effectively flat at `B=8` and `B=16`.
- The largest wins came from making the tiny `B*H` decode path use four workers
  per `P` row and parallelizing the `B/C` norm reduction inside the recurrent
  kernel.
- Nsight Systems on the current `B=1` whole-model persistent path shows the
  CuTe recurrent decode kernel down to about `16.3%` of GPU time, with the next
  biggest remaining buckets now being small projection kernels and decode-step
  conv.
- Small batches are still far from the proxy bound on this GPU.
- The `B=16` efficiency proxy exceeds `1.0`, which means the simple HBM traffic
  model is now overcounting off-chip traffic on this steady-state path. Treat
  the reported efficiency here as a comparative proxy, not a strict physical
  bound, on this card.
- No H100 was available in the local environment, so this table is not an H100
  claim. The decode harness accepts an H100 preset for target-platform runs.
