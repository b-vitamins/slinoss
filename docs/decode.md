# SLinOSS Decode Architecture

## Goal

Issue `#8` adds an inference-only autoregressive decode path that is separate from
the chunked sequence kernels used for training and prefill.

The landing has three explicit requirements:

- a one-token mixer decode contract
- a standalone CuTe recurrent-middle step kernel
- a model-level persistent decode path so the host does not launch one step per
  layer

## Public Surface

`SLinOSSMixer.step_inplace()` is the real public in-place decode entry point on
inference workloads. On supported CUDA shapes it does not route through
`forward(T=1)`.

`SLinOSSMixer.step(..., inplace=False)` remains as a convenience wrapper that
preserves the caller state by cloning it first.

`NextCharLM` exposes:

- `init_decode_state(batch_size, ...)`
- `decode_one(idx, state=None)`

`NextCharDecodeState` owns:

- per-layer mixer state
- the current decode position
- a position buffer for the persistent CUDA graph path

## Supported Fast Path Grid

The first landing keeps the fast path narrow on purpose:

- CUDA only
- `dtype in {fp16, bf16}`
- `batch in {1, 2, 4, 8, 16}`
- `d_head == 64`
- `d_state == 64`
- `normalize_bc == True`
- no `output_norm`

Unsupported shapes, CPU, and explicit reference decode backends fall back to the
eager/reference path.

## Architecture

The decode stack is split into two layers:

1. Per-layer decode:
   - `SLinOSSMixer._apply_causal_depthwise_conv_step(...)` performs the token-step
     causal depthwise convolution with persistent conv state.
   - `slinoss.ops.v2x2ssd.cute.decode.mixer_decode_step_cute(...)` runs the
     recurrent middle on a standalone one-token CuTe kernel.
   - `SLinOSSMixer.step_inplace(...)` exposes the real fast layer decode surface
     and mutates the passed decode state directly.
   - The kernel consumes token-local post-conv activations, flat params, raw
     `B/C`, gate and skip data, plus recurrent state, and returns gated output
     together with the next `state / b_prev / u_prev`.

2. Model-level persistence:
   - `NextCharLM.decode_one(...)` uses a CUDA-graph-backed engine on the supported
     CuTe grid.
   - The graph captures the full block stack, final norm, and LM head for one
     token.
   - Replay advances persistent per-layer state in-place, so the host performs
     one replay per token instead of a Python loop over layers.

The graph path is only enabled when the model is actually configured for the fast
CuTe decode backend. Explicit reference decode backends stay on the eager path.

## Correctness Notes

- The decode kernel is launched on the active CUDA stream so it participates in
  CUDA graph capture.
- The graph capture path does not reset recurrent state on replay.
- The recurrent decode kernel writes `state / b_prev / u_prev` directly back
  into the stable decode buffers, which keeps the persistent graph state valid
  across replays without extra host-side state copies.
- Public `step_inplace()` follows the same in-place decode contract as the
  persistent model path. The convenience `step(..., inplace=False)` wrapper is
  intentionally separate so callers can opt into cloning semantics when needed.

## Perf Tooling

Decode-specific tooling lives in `scripts/perf/`:

- `bench_nextchar_decode.py`: steady-state `us/token`, eager-vs-persistent
  comparison, and a lower-bound efficiency proxy
- `profile_nextchar_decode.py`: torch-profiler trace for steady-state decode
- `run_nsys_decode_report.sh`: Nsight Systems helper
- `run_ncu_decode_report.sh`: Nsight Compute helper

Recommended commands:

```bash
./scripts/guix-run python3 scripts/perf/bench_nextchar_decode.py \
  --backend cute \
  --batch-sizes 1,2,4,8,16 \
  --json-out /tmp/nextchar_decode.json
```

```bash
./scripts/guix-run python3 scripts/perf/profile_nextchar_decode.py \
  --backend cute \
  --mode persistent \
  --batch-size 1 \
  --trace-out /tmp/nextchar_decode_trace.json
```

```bash
scripts/perf/run_nsys_decode_report.sh \
  --backend cute \
  --mode persistent \
  --batch-size 1
```

## Lower-Bound Efficiency

`bench_nextchar_decode.py` reports:

`efficiency = t_lower / t_measured`

with

`t_lower = max(bytes_hbm / peak_bw, flops_tc / peak_tc_flops, flops_simt / peak_simt_flops, launches * launch_floor)`

The current implementation uses an analytic proxy built from the real nextchar
shape and the dominant decode traffic terms:

- token/position row reads
- logits writes
- tensor-core and SIMT work estimates from the actual model dimensions
- one host-visible graph replay per token on the persistent path

For cache-friendly steady-state decode, especially on the local RTX 3060, this
proxy can overestimate compulsory off-chip traffic and therefore report an
efficiency slightly above `1.0`. Treat it as a stable comparative metric rather
than a hard physical bound unless it is recalibrated on the target platform.

Checked-in measurements are from the local RTX 3060 development machine. The
harness also accepts an H100 preset for future runs on the target platform.
