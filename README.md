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

The backward kernels hold whole `3N` extents in shared memory, so `d_state`
trades against `chunk_size` and `d_head`. Measured on a 101,376 B carveout:
`3N = 48` fits at every `chunk_size` and `d_head`; `3N = 96` needs
`chunk_size <= 32`, and `chunk_size = 16` at `d_head = 64`; `3N = 144` needs
`chunk_size = 16`; `3N = 240` does not fit. The forward accepts every legal
shape, so one the backward cannot hold raises from the backward rather than from
the forward.

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
