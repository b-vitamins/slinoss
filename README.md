# slinoss

Oscillatory state-space sequence mixer with an SO(3) operator.

The scan's homogeneous step is a 3D rotation plus an isotropic scale, applied by
quaternion conjugation, with two-tap first-order-hold forcing:

```
z_t = exp(2*ls_t) * R(q_t) z_{t-1}
    + outer(u_{t-1}, Kprev_t(b_{t-1}))
    + outer(u_t,     Kcurr_t(b_t))
y_t = <C_t, z_t>
```

`q_t` is the unit quaternion of rotation vector `w_t`. It acts identically on
each of the `N` 3-vectors in the `3N`-dimensional per-head state and on each of
the `P` state rows, so the transition is four numbers per token. `ls_t <= 0` is a
log-scale. `Kprev`, `Kcurr` are first-order-hold taps in the commutant of the
rotation.

Status: under construction. The reference implementation defines correctness and
the CUDA path is built against it. Performance claims appear here only when
`scripts/bench/` reproduces them.

## Chunk form

Per chunk, with prefixes

```
Q_t  = q_t (*) ... (*) q_0     non-commutative, unit
lp_t = sum_{j<=t} ls_j         commutative, non-positive
```

rotate `C` and the tap-applied `B` into the chunk-local frame by `R(Q)^T`. Every
contraction becomes a real GEMM under a separable scalar mask:

```
score(t,s) = <crot_t, brot_s> * exp(2*(lp_t - lp_s)) * [s <= t]
```

No complex arithmetic, no interleaved storage, no packed-real matrix-multiply.
The chunked form is four dense real GEMMs at state dim `3N`. Both prefixes are
cheap enough to recompute per kernel rather than store.

## Install

Guix manages dependencies.

```
guix shell -m manifest.scm -- python3 -m pip install --no-deps -e .
```

Python 3.11+, PyTorch 2.1+. The CUDA path needs SM80 or newer and
`nvidia-cutlass-dsl`.

## Use

```python
import torch
from slinoss import SLinOSSMixer, SLinOSSConfig

cfg = SLinOSSConfig(d_model=576, d_state=48, expand=2.0, d_head=48, chunk_size=64)
mixer = SLinOSSMixer(cfg).cuda().to(torch.bfloat16)

x = torch.randn(4, 2048, 576, device="cuda", dtype=torch.bfloat16)
y = mixer(x)
```

The backward kernels walk the state in 48-lane blocks, so their shared-memory
footprint does not grow with `3N` and the trade is `chunk_size` against `d_head`.
Measured on a 101,376 B carveout, at every `3N`: `chunk_size` 16 and 32 fit at
every `d_head`; `chunk_size = 64` fits to `d_head = 64`; `chunk_size = 128` does
not fit. The forward accepts every legal shape, so one the backward cannot hold
raises from the backward rather than from the forward.

A whole model is `SLinOSSStack`, and one layer of it `SLinOSSBlock`: a fused
pre-norm around the mixer, and a second around a SwiGLU FFN of width
`ffn_ratio * d_model`. A block hands its branch output back unadded, so the add
is the next fused norm's first operation and the stack's final norm is the add
the last block did not do. The residual stream is float32 from the first norm to
the last, and the three norm weights stay float32 through a module-wide cast.

```python
from slinoss import SLinOSSStack

cfg = SLinOSSConfig(d_model=576, d_state=48, d_head=48, n_layers=13, vocab_size=50257)
model = SLinOSSStack(cfg).cuda().to(torch.bfloat16)

ids = torch.randint(0, 50257, (4, 2048), device="cuda")
logits = model(ids)
```

`vocab_size=None` drops the embedding and the head, and the stack then takes and
returns `(B,T,d_model)` activations.

The head is wider than `vocab_size`. All three of its GEMMs carry the output width
on the mode that decides which kernel cuBLAS picks, so an unaligned width costs
every one of them its wide load: 10.1 ms of the 59.3 ms the GEMM class takes in a
456 ms step, measured at the shape above.
`vocab_pad_multiple`, 8 by default, rounds the head up, so `logits` is
`(B,T,cfg.padded_vocab_size)` and 50257 runs at 50264. The columns past
`vocab_size` hold `finfo(dtype).min`, which is exactly zero under a softmax and
below every reachable logit, so the loss, every gradient, every argmax and every
sample are the unpadded ones. Set `vocab_pad_multiple=1` to keep the head at
`vocab_size` and pay for it. The embedding is a gather and is never padded.

Decode threads a `StackState`, which holds four carries per layer: the scan state,
the convolution window, and the previous token's `B` and `U`, which the two-tap
forcing needs. Every buffer is written in place, so a captured graph keeps its
addresses. Prefill and a single token are the same call at two sequence lengths.

```python
from slinoss import StackState

state = StackState.allocate(cfg, 4, device="cuda", dtype=torch.bfloat16)
logits = model(ids[:, :-1], state)  # prefill
logits = model(ids[:, -1:], state)  # one token
```

Cast the module to the state's dtype rather than decoding under autocast.

`generate` is that loop: prefill, then one call per token, sampling on the device.
`temperature=0` is greedy, `top_k` truncates the distribution, and `stop_token_id`
holds a finished sequence at its stop token so the returned block stays
rectangular. The returned state has consumed every generated token but the last, so
a continuation prompts with that one.

```python
from slinoss import generate

out = generate(model, ids, max_new_tokens=64, temperature=0.8, top_k=50)
more = generate(model, out.tokens[:, -1:], max_new_tokens=64, state=out.state)
```

The operator is callable directly. `so3ssm` is the sequential reference and is
float64-capable; `so3ssd` is the chunked, autograd-complete, CUDA-accelerated
path.

```python
from slinoss.ops.so3ssd import so3ssd, so3ssm

y, state, b_last, u_last = so3ssd(U, trans, K, B, C, chunk_size=64)
```

## Tensor contracts

Time-major. `N` is a multiple of 16, so `3N` is a multiple of 48.

| tensor  | shape         | dtype          | contents                |
|---------|---------------|----------------|-------------------------|
| `U`     | `(B,H,T,P)`   | bf16/fp16/fp32 | input weights           |
| `trans` | `(B,H,T,4)`   | fp32           | `(w_x, w_y, w_z, ls)`   |
| `K`     | `(B,H,T,2,4)` | fp32           | per tap `(kr, g, h, 0)` |
| `B`     | `(B,G,T,3N)`  | bf16/fp16/fp32 | input vectors           |
| `C`     | `(B,G,T,3N)`  | bf16/fp16/fp32 | output vectors          |
| `z`     | `(B,H,P,3N)`  | fp32           | state                   |
| `Y`     | `(B,H,T,P)`   | bf16/fp16/fp32 | output                  |

Contiguous, except `B` and `C`, which may be pitched: unit trailing stride,
non-overlapping rows, 16-byte aligned base and pitch. That is what lets them be
column bands of one fused projection rather than buffers of their own.

`B` and `C` are grouped. `G` divides `H` and head `h` reads group
`h // (H // G)`, so `G == H` is the ungrouped case and not a second signature.

The trailing `3N` is `N` 3-vectors in lane-major order: element `3n+i` is
component `i` of 3-vector `n`.

Taps act as `K(v) = kr*v + g*(w.v)*w + h*(w x v)`. The chart is polynomial in
`w`, hence analytic at `w = 0`; the axis-angle normal form is not. Tap lane 3 is
a hard zero kept for alignment.

`b_prev (B,G,3N)` and `u_prev (B,H,P)` supply the time-0 previous values for
streaming. Pass both or neither.

## Development

```
guix shell -m manifest.scm -- ruff format . && ruff check .
guix shell -m manifest.scm -- pyright
guix shell -m manifest.scm -- python3 -m pytest -xvs
```

Benchmark and profile drivers are in `scripts/bench/` and `scripts/perf/`.
`CLAUDE.md` holds the rules the CUDA path is held to.

## License

MIT.
