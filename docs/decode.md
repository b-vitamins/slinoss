# Decode notes

One-token autoregressive inference. The chunked scan is the training operator and
answers every `T`; it is the wrong program at `T = 1`, and this document records what
replaced it, what the replacement is held to, and where the remaining cost is.

`docs/operator.md` holds the map and the tensor contracts, `docs/kernels.md` the GEMM
forms and the roofline classes, `docs/structure.md` the five parts and dispatch. This
file holds the decode stage only.

## Why a second boundary

At `T = 1` the chunked program still launches a whole chunk pipeline: a change of
basis, four GEMM forms, chunk-local prefixes, a scan over one chunk. Per-chunk work
does not shrink with the token count, so the fraction of it that a single token needs
falls with `L`. Counting carry traffic alone predicted 46.0% of the step was
recoverable. Measured, routing one token to its own boundary took the batch-128
captured step from 17,000.448 us to 6,727.680 us, **-60.414%, 2.5269x**, paired
interval `[-10,273.791, -10,269.696] us` at 95.996% over 200 order-swapped pairs. The
prediction was low because it priced traffic and the deleted quantity was program.

That delta is not decomposable into a routing term and a kernel term. The
pre-transpose kernel deregisters at the routed shapes and the boundary reads
`('reference',)` there, measuring 95,746 us, so there is no arm that isolates one of
the two. What is measurable is the reverse: the routed step's latency moves 0.018
points across kernel versions once both register, so the 60.414% is dominated by the
boundary change and not by the kernel behind it.

## The fusion boundary

The boundary of this version is the `T = 1` scan recurrence and nothing else. Outside
it, in producer order: the fused input projection, the value convolution, the key
convolution, the parameter maps of `slinoss.ops.scanprep`, and the mixer tail with its
output projection. The call reads `U`, `trans`, `K`, `B`, `C` from global memory and
writes `y` back to it.

Three buffers carry one step and only three: `ssm`, `b_prev`, `u_prev`. They are
written in place, in the caller's storage. Two measured reasons, not stylistic ones.

At `B 1, H 16, P 64, 3N 96, G 1`, bfloat16 activations, float32 state, the call moves
793,920 B of which the two state passes are 786,432 B: **99.06%**. A boundary that
returned a fresh state for the caller to copy into the carry would run four passes over
that 99% instead of two. And CUDA-graph capture records addresses, so a rebound field
would leave replay writing memory nobody reads.

The update is legal in place because at `T = 1` it is lane-local: a lane reads its
three components, rotates and scales within them, writes the same three addresses. No
term crosses a lane or a row. That holds at one token only and for inference only.
`slinoss.ops.so3ssd.so3ssd` keeps `z0` for its backward and cannot take this signature.
`slinoss.ops.decode.reference.decode_ref` forms the new state out of place and copies it
in at the end, so the oracle cannot hide a lane-crossing error behind aliasing the
kernel is permitted.

Every later fusion moves the boundary outward without touching state semantics. The
value convolution adds `conv` as a fourth carry and drops `U`; the key convolution adds
`keys` and drops `B` and `C`; the parameter maps drop `trans` and `K`; the tail replaces
`y` with the mixed token. None of them changes the three carries.

## The launch graph

One layer, one token, routed. Eight kernels and two device-to-device copies at batch 1
and 128. **Not invariant in batch**, and the part that moves is not this operator's:
the six kernels the tree owns -- `decode_fwd`, `decode_carry`, `conv1d_fwd` twice,
`scanprep_fwd`, `mixer_tail_fwd` -- are the same at every batch, width and dtype, while
the two projections are cuBLAS's choice and it picks a different program per shape. At
batch 8, 32 and 64 it adds a `cublasLt::splitKreduce_kernel` and the step is nine
kernels; at batch 1 one projection is a `gemvx` GEMV rather than a GEMM. The tile
differs at every batch measured. A launch count for this step therefore has to name its
batch.

`acceptance` (`d_model 576, H 18, P 64, 3N 240, G 1`), batch 128, bfloat16, one card,
floor fitted in that session at `4.5392 us + bytes / 684.356 GB/s`, max residual 0.97%:

| stage | kernel | us | share | traffic / compulsory | % of fitted floor | class |
| --- | --- | --- | --- | --- | --- | --- |
| in_proj | `ampere_bf16_s16816gemm_128x64` | 10.752 | 2.25% | 0.389x | 64.3 | unjudged |
| value conv | `conv1d_fwd` | 9.696 | 2.03% | 0.747x | 73.5 | unjudged |
| key conv | `conv1d_fwd` | 5.803 | 1.21% | 0.390x | 87.9 | unjudged |
| prep | `scanprep_fwd` | 3.189 | 0.67% | 0.842x | n/a | unjudged |
| **recurrence** | **`decode_fwd`** | **426.453** | **89.25%** | **1.0013x** | **98.60** | **DRAM_BOUND** |
| carry | `decode_carry` | 2.571 | 0.54% | 6e-7x | n/a | unjudged |
| tail | `mixer_tail_fwd` | 6.571 | 1.38% | 0.092x | 71.5 | unjudged |
| out_proj | `cutlass_80_wmma_32x32` | 12.789 | 2.68% | 0.143x | 38.4 | unjudged |
| conv window | D2D copy | 2.99 | 0.63% | 1.00x | -- | unjudged |
| key window | D2D copy | 2.98 | 0.62% | 1.00x | -- | unjudged |

The recurrence's share of device time is 15.2 / 46.1 / 73.8 / 83.7 / 88.7% at batch
1 / 8 / 32 / 64 / 128, 13.9% and 55.0% at batch 8 and 128 at `d_model 288`, and 82.4% at
float32 where the two sgemms take 10.6%.

**No other site in the census carries a traffic verdict at any batch or either width.**
The only stage that ever moves compulsory traffic is the one already closed, so
everything a later fusion may touch -- both convolutions, prep, the carry kernel, the
tail, the two window copies -- is 30.8 us of 477.5, a 6.45% ceiling, and 1.25 MB of
countable traffic against the step's 284 MB.

Launch count is not a lever on this graph. Under replay total launch idle is 2.77-5.34
us/step and per-launch 0.32-0.67 us, 0.58% of device time at batch 128; deleting seven
of eight launches buys under 2.4 us. Both projections stay vendor GEMMs, and negatively:
in_proj's 10.752 us against the DRAM time for its compulsory bytes is 99.0%, out_proj
leaves 5.66 us (1.18%) but is bound by neither traffic (0.143x compulsory, weights
L2-resident) nor arithmetic (170 MFLOP at 13.3 TFLOP/s, 8.6% of peak) at 96 CTAs, so
beating cuBLAS there means out-scheduling it at `M = B`.

The one arm still worth costing is the value and key convolutions fused with their two
window copies: four launches to one, 19.3 us at batch 128 and 12.2 us of 79.7 at
`d_model 288`. It is mostly program deletion -- three launches and three wave tails,
about 9 us of fixed cost -- plus a traffic deletion of exactly 1,253,376 B, the window
read a copy performs and a convolution holding its window in registers would not.

## Not an autograd node

No `torch.autograd.Function`, at any `T`. A function whose backward raised would still
record a node and defer the failure to `.backward()`, which is a training step that
fails after the forward rather than at it. Every operand is refused if it requires a
gradient rather than detached: detaching returns a tensor whose gradient is silently
zero, which is a training run that reports a number.

A sequence start is spelled as zero carries, not as an omitted one. Zeros are not a
branch in the arithmetic -- the previous tap is linear in `b_prev` and scaled by
`u_prev`, so a zero carry annihilates the term exactly, at every tap value including
`w = 0`.

## The kernel and its class

12 flop per state element against 8 bytes is 1.5 flop/byte against a machine balance of
163 on this part, so arithmetic is under 1% of the roofline and the class is
`DRAM_BOUND`: at least 85% of the bandwidth measured at its own footprint. The carry
stage is `SERIAL_TINY`.

The first version read 52.77% of its fitted floor while moving 1.0008x compulsory
traffic. Bytes were therefore not the binder; memory-level parallelism was. A 3-vector
is 12 bytes wide and touches three sectors per request however lanes are dealt, so the
fix was a transpose of the addressing rather than of the assignment: `gssm` is read and
written by component plane at `run + k*group + slot`, covering the group's `3*group`
consecutive floats sector-exactly, with three `shfl.sync` per direction converting
plane layout to lane layout inside the row group. The packed segment field
`((32 - G) << 8) | 31` supplies the group-width wrap for free. No barrier and no shared
memory were added.

Store sectors fell 67,092,480 -> 22,855,680 and load sectors 118,793,544 -> 73,159,608.
The 45.6M load reduction was not predicted: the kernel-wide load ratio read 4.4757
because the fp32 state term is averaged in with bfloat16 requests and broadcasts, which
concealed a 3x per-tensor amplification. Account a sector ratio per tensor, never
kernel-wide.

Measured against the fitted floor, `4.5036 us + bytes / 684.898 GB/s` on this part:

| shape | % of fitted DRAM floor |
| --- | --- |
| B16 | 100.80 |
| B32 | 99.76 |
| B64 | 99.29 |
| B128 | 98.72 |
| B128, float32 | 98.17 |
| B128, `3N 288` | 98.40 |

64 registers and zero local sectors at every rung. Floors are fitted per session, so
the arm's own paired render reads 52.77% -> 98.94% at B128 against this ladder's 98.72%;
each figure belongs to its own fit and the two are not interchangeable.

Batch 8 reads 102.79%, which is above the floor and therefore requires an account
rather than a caveat: its measured traffic is 0.967x compulsory, so part of the working
set is served by L2. **The crossover is not `footprint > L2`.** Batch 8's footprint is
1.42x L2 and its traffic is still under compulsory. The honest test is measured traffic
over compulsory at or above 1.0, and only shapes passing it carry a DRAM verdict. Below
L2 there is no verdict and the kernel is reported unjudged, which is why batch 1 is
never the judged shape.

Residual sector excess sits on the L2-resident band tensors, where it costs tag work
rather than DRAM bytes. No further arm on this kernel is warranted.

## Numerical validation

`max |got - want|` over `max |want|` against a float64 autograd oracle over the same
operands. Every residual agreed to every printed digit under torch 2.6.0+cu124 and
2.7.1+cu126, so the bounds are the arithmetic's and not a version's.

Absolute drift at `(2,4,2,16,32)`, one token then the chained horizon, against an oracle
whose `y` reaches 3.43e+01 at one token and 8.37e+01 over the chain, state 5.00e+00 and
1.02e+01:

| dtype | `y`, one token -> chained | state, one token -> chained |
| --- | --- | --- |
| bfloat16 | 6.232e-02 -> 2.491e-01 | 9.770e-07 -> 1.585e-05 |
| float16 | 7.550e-03 -> 2.864e-02 | 1.032e-06 -> 1.699e-05 |
| float32 | 4.546e-06 -> 2.290e-04 | 1.105e-06 -> 1.638e-05 |

The horizon costs a factor of 21 in relative `y` error at float32 and 1.2 at float16,
because a 16-bit store rounds away everything the chain accumulated. Single-step and
chained bounds are therefore separate constants: one pair sized by the chain would leave
float32 at a twentieth of its bound at one token, which is a bound nothing can fail.
`--tolerance-report` prints each bound beside the error measured under it; no bound sits
above 2.5x its worst case.

Continuing from a state this boundary last wrote reproduces the whole-sequence result of
`slinoss.ops.so3ssd.so3ssd` over the same tokens, so stepping a sequence in any
partition from a freshly allocated `slinoss.state.MixerState` is the same operator.
That is asserted token by token and as a prefill followed by a decode, at a kernel width
rather than only at float64.

## Graph integration

`GraphedStep` carries a required `recorded` field. A CUDA graph addresses parameters and
state by pointer, so a caller that retained only `graph`, `inputs` and `outputs` let
those blocks return to the allocator; the embedding gather then read reclaimed float
bytes as ids. Symptoms ranged over a device-side assert, an illegal memory access, and
-- with nothing else allocated -- no error at all and silently non-finite logits. The
third is the reason this is a required field and not a documented caution.

Replay is allocation, compile and synchronization free, asserted rather than asserted
of. Over 8 shape cells at 32 steps each: allocator retries 0, allocated bytes delta 0,
segment count delta 0, no synchronization under `set_sync_debug_mode("error")`, and no
compile inside capture with the AOT payload loaded. Each property fails under its own
injected fault -- a per-call `torch.empty`, an `.item()`, a stripped payload key.

## The payload

A capture that compiles inside itself is not a captured step, so the kernels a decode
step launches are exported ahead of time. `decode_fwd` specializes on activation dtype
and `N`; `decode_carry` on activation dtype alone, since no address in it depends on the
state width. Nothing else is an axis: batch, head count and group count enter as runtime
extents.

The whole ladder is 24 cells, **42 entries, 1,019,608 B**, built in 78.17 s. 42 and not
the recurrence's 27, because a decode step through the stack reaches four more
launchers -- `rmsnorm_residual_fwd` 6 entries, `mixer_tail_fwd` 3, `scanprep_fwd` 3,
`swiglu_fwd` 3, none width-specialized. The 27-entry, 685,656 B figure is exactly the
`decode_fwd` plus `decode_carry` subset; sizing a payload to it leaves the other four
compiling inside the first capture.

Each cell is verified in its own process under a strict payload, reading `compiled 0`
against `payload hits 7`. Strict and not permissive: a miss that falls through to a
compile is the failure the payload exists to prevent, and it is silent. An install
carries it -- the built wheel holds 43 `_aot` members over 1,033,247 B and loads all 42
entries strictly with the source tree off `PYTHONPATH`.

**Below batch 128 the eager step is host-bound, not device-bound.** The eager host wall
is 610.7-632.5 us per layer step at every batch and both widths while device time is
27.0-520.7 us, so the host program costs 114-582 us of it and replay deletes that to
within 1.5-3.0 us of device time. Which kernel to fuse is not the question at batch 64
and below; whether the step is captured is.

## The matched comparison

Against Mamba3, one card, one layer, one token, CUDA-graph replay on both sides,
torch 2.7.1+cu126. `dominates` is latency at or under 0.90x on every primary batch;
`competitive` is geomean within 10% with no primary point past 1.20x. `sB` is state
bytes per token per layer, this operator over Mamba3; `lch` is device launches per
step. Ten matched pairs, each judged at both boundaries.

| pair | boundary | verdict | geomean | worst | lch | sB |
| --- | --- | --- | --- | --- | --- | --- |
| **`3N 144` / `d_state 128`, bf16, `d_model 512`** | recurrence | neither | 1.1525 | 1.1698 @ B32 | 2:1 | 1.1269 |
| **the same pair** | **step** | **competitive** | **0.8859** | **1.0251 @ B128** | **10:23** | **1.1269** |
| `3N 144` / 128, float32 | recurrence | neither | 1.1428 | 1.1698 @ B32 | 2:1 | 1.1330 |
| `3N 144` / 128, float32 | step | competitive | 0.9011 | 1.0314 @ B128 | 10:23 | 1.1330 |
| `3N 144` / 128, `d_model 1024` | recurrence | neither | 1.1536 | 1.2143 @ B8 | 2:1 | 1.1250 |
| `3N 144` / 128, `d_model 1024` | step | competitive | 0.9412 | 1.0575 @ B128 | 10:23 | 1.1250 |
| `3N 144` / 128, `d_model 2048` | recurrence | neither | 1.1252 | 1.1731 @ B8 | 2:1 | 1.1240 |
| `3N 144` / 128, `d_model 2048` | step | competitive | 0.9983 | 1.0654 @ B128 | 11:24 | 1.1240 |
| `3N 144` / 128, `ngroups = nheads` | recurrence | neither | 1.1777 | 1.2115 @ B32 | 2:1 | 1.1837 |
| `3N 144` / 128, `ngroups = nheads` | step | competitive | 0.9342 | 1.1268 @ B128 | 10:25 | 1.1837 |
| `3N 144` / 128, MIMO | recurrence | neither | 1.1014 | 1.1091 @ B32 | 2:1 | 1.1014 |
| `3N 144` / 128, MIMO | step | competitive | 0.8008 | 0.9268 @ B128 | 10:27 | 1.1014 |
| `3N 96` / 128 | recurrence | competitive | 0.8401 | 0.8571 @ B32 | 2:1 | 0.7564 |
| `3N 96` / 128 | step | dominates | 0.7458 | 0.7915 @ B128 | 10:23 | 0.7564 |
| `3N 96` / 64 | recurrence | neither | 1.6095 | 1.6176 @ B128 | 2:1 | 1.5069 |
| `3N 96` / 64 | step | neither | 0.9673 | 1.2893 @ B128 | 10:23 | 1.5069 |
| `3N 192` / 128 | recurrence | neither | 1.5057 | 1.5094 @ B32 | 2:1 | 1.4974 |
| `3N 192` / 128 | step | neither | 1.0330 | 1.3127 @ B128 | 10:23 | 1.4974 |
| `3N 192` / 64 | recurrence | neither | 2.8916 | 2.9314 @ B128 | 2:1 | 2.9832 |
| `3N 192` / 64 | step | neither | 1.3291 | 2.1187 @ B128 | 10:23 | 2.9832 |

One `dominates`, six `competitive`, thirteen `neither`. **The verdict of this version
is `competitive` at the whole layer step and `neither` at the recurrence**, read off
the nearest matched pair and held by five further step pairs across two widths, two
dtypes, both group modes and MIMO. The single `dominates` is the favourable pairing
and is not the headline.

110 cells were measured over 10 shape classes of the 144 the full enumeration admits,
every one on an exclusive card in one window. No verdict is claimed over the 134
unmeasured classes. Nine of the 720 legal cells are refused: three because `n_heads = 4`
forces a `d_head` of 256 to 1024, outside the measured MMA N-mode list, and six because
`ngroups` of 4, 8 or 16 is neither 1 nor `nheads` and Mamba3 admits only those two.

## Remaining bottleneck, and the limits

**The recurrence gap is state bytes, not kernel work.** Against Mamba3 at matched
capture, both operators sit 1.00-1.09x their own in-process fitted DRAM floor at the
recurrence, and the measured latency ratio factors into a state-byte term times a
floor-distance term reproducing it to -0.74% (`1.3127x = 1.4848x * 0.8907x`) and -2.46%
on a second shape class. There is no kernel deficit left to find at that boundary.
`d_state` is the only lever.

The table says the same thing a second way: the recurrence ratio tracks the state-byte
ratio across every pair -- 0.7564 bytes to 0.8401 time, 1.1269 to 1.1525, 1.4974 to
1.5057, 1.5069 to 1.6095, 2.9832 to 2.8916 -- and the kernel is already at 98.94% of its
fitted floor.

The step goes the other way from the recurrence, and not for the reason the two
convolutions suggest. At batch 1 and 128 this step is 10 device launches against
Mamba3's 23 to 27, of which 14 of Mamba3's are elementwise, fill and normalization glue;
the two convolutions are 2 of these 10. Mamba3's step sits 1.27-1.94x its own floor
against this one's 1.16-1.44x, and this one pays two convolutions and two convolution
state buffers Mamba3 does not have and is still ahead at the geomean.

One action follows, and it is not the carry. `decode_carry` is the recurrence's second
launch, so deleting it is the arm the launch count suggests; priced, it cannot move a
verdict. It runs 1.451 to 2.699 us over batch 1 to 128 and is launch-bound at 6e-7x its
compulsory traffic, so it does not scale with width and transfers across shape classes.
Against the headline pair's own recurrence medians that
is 3.90 / 2.25 / 1.06% of the boundary at batch 32 / 64 / 128, and deleting it entirely
takes the geomean 1.1525 to 1.1247 against a 1.10 bar while moving no point across 1.20.
The verdict stays `neither`. It is not a byte lever either: `b_prev` is 2,336 B of the
600,032 B that boundary carries at batch 8. The launch exists for a cross-block ordering
reason -- every block whose head maps to group `g` reads the `b_prev` a different block
wrote, and a grid has no barrier -- so fusing it is a correctness change with no measured
return.

`d_state` is the lever, and it is quantized. The legal widths are multiples of 48 here
and `[32, 64, 128]` there, so against `d_state = 128` the reachable byte ratios are
`96/128 = 0.75` and `144/128 = 1.125` with no rung between. The recurrence axis is won at
`3N = 96` and lost at `3N = 144`, nothing sits at parity, and a claim has to name its
width.

Limits that constrain what any of these figures can be used for:

- **`d_state` can never match.** The legality sets are disjoint: multiples of 48 here,
  `[32, 64, 128]` there. Every comparison is state-byte-asymmetric and must be printed
  with its ratio.
- **The `lch` column names two batches, not the column's row.** It was counted at batch 1
  and 128 only. Two of the step's ten launches are cuBLAS's choice of projection program,
  and at batch 8, 32 and 64 cuBLAS adds a `cublasLt::splitKreduce_kernel`, so the step is
  11 launches there. The ratio is reported at the two batches counted and no launch count
  is claimed for the other three.
- **A favourable pairing is not a result.** `3N = 96` against `d_state = 128` wins on
  every axis and hands this operator 0.756x the state. It is the least honest cell
  available, not the headline.
- **No eager row is judged, at any batch.** An eager sample is not a measurement of the
  operator: eager at batch 8 read 8.53x its own fitted DRAM floor against its captured
  twin's 1.04x. A per-sample median is not the eager cost either -- CUDA-event samples
  go bimodal under host run-ahead, and at batch 1 the median read 38.912 us against a
  mean near 600 over the same 200 samples. Capture both sides, or report host enqueue
  and device busy separately as whole-loop walls.
- **Iteration count does not buy a judgeable interval here.** Raising it 20x moved the
  worst recurrence half-width 30.43% to 20.08% in bf16 and 29.06% to 25.91% in float32,
  not to the ~6.8% a `1/sqrt(n)` model predicts, while the full range widened 52.6% to
  128.6% and the median itself moved 19.456 to 21.504 us -- 10.5%, larger than the
  whole margin under test. The judged graph rows carry half-widths at or under 1.111% of
  their medians; refused rows carry 3.35% to 20.43%. A row that cannot be resolved is
  reported unjudged with its count, not resolved by more samples.
- **The batch-1 recurrence is below the instrument.** Both medians land on the same
  1.024 us CUDA-event tick, which priced onto both is 14.29% of the ratio against a 10%
  margin. That is a tick-quantization limit, not a variance one, so it is unjudged at
  any iteration count.
- **The comparison is not one dependency set.** SISO rows were taken under
  `apache-tvm-ffi` 0.1.9 and the MIMO row under 0.1.12. The dependency set is recorded
  and deliberately not part of the bank key: keying it would have refused every MIMO
  cell on a SISO render, and recording it unkeyed is what exposed the split.
- **`% of fitted floor` is uninformative below the fit's intercept.** The fixed term is
  4.3-5.0 us, so a stage shorter than that reads above 100% by being short. Only the
  traffic test judges those rows, and it returns unjudged for every one of them.
- **An L2 contamination check is signed, and write-back on the store side.** Scaling
  write sectors by a miss rate declared this kernel's closed cell void at 49.79%
  residual; write sectors times sector size is the bound, and the corrected form lands
  within 0.52-1.77%.
  It voids three cells outright: `acceptance` batch 1, 8 and 32 in bfloat16 carry
  +51.3 / +50.7 / +12.3% residual at the in_proj GEMM, so in_proj's traffic ratio is
  unattributable at batch 32 and below.
- **No figure crosses a host or a torch version.** The 2.6.0 -> 2.7.1 delta is measured
  and opposite in sign on the two axes: host-bound cells 5-17% cheaper, device-bound
  cells 0.2-1.2% slower. That bounds the confound; it is not a conversion factor.
- **`slinoss.decode.generate` is not the latency path.** It reaches this boundary -- it
  steps the stack with `token[:, None]`, so `T = 1` -- but its early-stop check reads
  `finished.all()` on the host once per step, which is a device synchronization per
  token whenever `stop_token_id` is set. A per-token latency figure comes from the
  captured step.
- **Dispatch falls back per boundary, not per tree.** `decode` resolves its kernel at
  float32 while `so3ssd` falls to the reference at the same call. A measurement that
  does not print the backend that answered cannot tell a kernel number from a torch one.
