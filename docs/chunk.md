# Chunk length

`L` is pinned by shared memory, not chosen by a sweep. `scripts/perf/chunk_sweep.py`
holds the four questions apart: `--mode arena` for legality, `--mode traffic` for the
byte and arithmetic model, `--mode numerics` for the invariants, `--mode step` for the
measured step.

## What scales which way

Per token, at a fixed geometry:

| quantity | scaling in `L` | terms |
| --- | --- | --- |
| chunk-state buffers | `1/L` | `inc`, `zstart`, `dinc`, `dzstart` at `(B,H,C,P,3N)` fp32 |
| chunk scalars | `1/L` | `cquat`, `cscale`, `carry_u`, `carry_b`, `dchunk_rot`, `dchunk_scale` |
| shifted spans | `(L+1)/L` | `U` and `B`, one extra row per chunk for the previous tap |
| token operands | flat | `dy`, `C`, `trans`, `K`, `y`, `dU`, `dB`, `dC`, `dtrans`, `dK` |
| `dlogp` | flat | `(B,H,C,L)` is `T` scalars however `L` is cut |
| score and diagonal | `L` | arithmetic only; the `L x L` score never reaches global memory |

No global-memory term grows with `L`. The `L x L` score is retiled in registers in
`chunk_scan_fwd` and banked in shared memory in `chunk_vector_bwd`, so lengthening the
chunk buys traffic and costs arithmetic, with no interior byte minimum.

Compulsory traffic and arithmetic of one whole step of one operator call at
`B 4 H 18 T 2048 P 64 3N 240 G 1`, bf16, computed by `--mode traffic`:

| `L` | total MB | GFLOP | flop/byte |
| --- | --- | --- | --- |
| 16 | 8,235 | 71.8 | 8.7 |
| 32 | 4,263 | 84.7 | 19.9 |
| 64 | 2,277 | 110.5 | 48.5 |
| 128 | 1,284 | 162.2 | 126.3 |
| 256 | 787 | 265.5 | 337.2 |

Against 685.22 GB/s and 112 TFLOPS the two floors cross between 128 and 256, so the
model's optimum is `L = 128`, which is `MAX_CHUNK`. A model is not a measurement.

## What refuses it

Five kernels allocate shared memory. Both state passings and the boundary allocate
none, so `L` cannot refuse them. At the geometry above, against a 101,376 B carveout:

| kernel | `L 16` | `L 32` | `L 64` | `L 128` | `L 256` |
| --- | --- | --- | --- | --- | --- |
| `chunk_increment_fwd` | 12,688 | 25,232 | 19,600 | 28,816 | 47,248 |
| `chunk_scan_fwd` | 68,752 | 73,872 | 79,504 | 122,512 | 208,528 |
| `chunk_start_bwd` | 11,392 | 22,784 | 45,568 | 91,136 | 182,272 |
| `chunk_input_bwd` | 42,416 | 44,528 | 48,752 | 89,968 | 172,400 |
| `chunk_vector_bwd` fold 1 | 53,648 | 64,848 | 91,344 | 136,144 | 248,272 |
| `chunk_vector_bwd` fold 18 | 56,976 | 71,504 | 93,392 | 162,768 | 301,520 |

Legal `L` is 16, 32, 64. `L 128` is refused by `chunk_vector_bwd` and
`chunk_scan_fwd`; `L 256` exceeds `MAX_CHUNK` and is refused by four of the five.
No `L` reaches two resident blocks in every kernel: `chunk_scan_fwd` and
`chunk_vector_bwd` hold one block at every legal `L`.

The two refusals are not the same kind. `chunk_scan_fwd` is refused by `3N`: its
`mma_rows(L) x 3N` operand tile is 63,488 B of the 122,512 at `L 128`, and slicing that
tile by a lane block, as `chunk_input_bwd` already does, gives 73,360 B at a 120-wide
slice and 48,784 B at 48 wide. `chunk_vector_bwd` is refused by `L` alone: at the
narrowest legal widths, `P 16` and `3N 48`, it still spans 120,528 B at `L 128`. Its
three `L`-spanning fp32 accumulators are 53,456 B of the 162,768, and no single tile
carries the 61,392 B of excess.

## What it measures

Whole step, 13 layers, at the geometry above, on one RTX A6000 (sm_86) with clocks
unlocked and no foreign process on the device beyond the MPS daemon. Medians over six
launches, each figure measured twice.

| `L` | median us | range us | tokens/s | in-package kernels us |
| --- | --- | --- | --- | --- |
| 16 | 563,616 | 563,258 - 564,257 | 14,535 | 410,621 |
| 32 | 449,405 | 449,124 - 450,243 | 18,210 | 295,950 |
| 64 | 461,166 | 460,256 - 462,166 | 17,782 | 307,652 |

Run to run the same `L` reproduces to 0.20%, so the 2.55% between 32 and 64 resolves.
Per kernel, microseconds per step over 13 calls:

| kernel | `L 16` | `L 32` | `L 64` |
| --- | --- | --- | --- |
| `chunk_vector_bwd` | 121,784 | 128,767 | 204,905 |
| `chunk_input_bwd` | 150,100 | 82,850 | 46,486 |
| `state_passing_fwd` | 43,977 | 22,073 | 11,134 |
| `chunk_increment_fwd` | 27,505 | 15,551 | 9,446 |
| `state_passing_bwd` | 21,863 | 10,984 | 5,478 |
| `chunk_scan_fwd` | 15,454 | 11,661 | 8,723 |
| `chunk_start_bwd` | 13,020 | 6,510 | 3,265 |

Every kernel but one falls with `L`, at or near the `1/L` the byte model predicts.
`chunk_vector_bwd` rises: 1.59x from `L 32` to `L 64` against 1.46x more arithmetic and
0.54x the bytes, so its time tracks its arithmetic and not its traffic. At `L 64` it
moves 4.58 GB per step against a 6,676 us byte floor and reaches 2.1% of the tensor
pipe, so it is bound by neither and its 1.59x is what makes the shorter chunk win.

## What the invariants do

`--mode numerics` against a float64 oracle, `B 1 H 2 T 1024 P 16 3N 48`:

| `L` | `ref/f32` y | `ref/bf16` y | `cute/bf16` y | chunk decay | prefix drift |
| --- | --- | --- | --- | --- | --- |
| 16 | 4.23e-07 | 3.265e-03 | 6.210e-03 | 1.8e-16 | 5.36e-07 |
| 32 | 8.28e-07 | 3.265e-03 | 4.800e-03 | 2.1e-30 | 7.75e-07 |
| 64 | 1.25e-06 | 3.265e-03 | 4.800e-03 | 2.6e-52 | 1.07e-06 |
| 128 | 3.87e-06 | 3.265e-03 | 4.800e-03 | 9.6e-101 | 1.55e-06 |
| 256 | 7.41e-06 | 3.265e-03 | 4.800e-03 | 4.3e-191 | 2.38e-06 |

I1 bounds the chunk decay, and a longer chunk multiplies more of them: past `L 32` the
chunk-end decay is below the float32 subnormal floor and flushes to zero. That is not a
defect. I3 forbids factoring `exp(2lp_t) * exp(-2lp_s)`, so nothing divides by the
decay and a flushed exponent contributes exactly zero, which is what the map says a
token that far back contributes.

I2 is a per-step bound on `|w|` and does not see `L`.

I5 renormalizes the quaternion prefix once per chunk. The drift that renormalization
absorbs grows as the square root of `L`, not linearly: 1.41x per doubling measured
against 1.414 predicted for a random walk. At `L 256` it is 2.38e-06, three orders
below the bf16 operand epsilon of 3.9e-03, so one renormalization per chunk still
suffices.

The bf16 arms are flat in `L` to every printed digit. Accuracy does not constrain the
chunk length; shared memory does.

## The default

64. The measured 2.55% at `L 32` comes from one kernel that is 30x off its own byte
floor, and every other kernel is between 1.3x and 2.0x faster at 64 than at 32. Cutting
`L` to hold `chunk_vector_bwd` down encodes that kernel's cost model in a global
default and reverses once the kernel reaches its class. The byte model falls
monotonically in `L`, so the direction the default should eventually move is up.
