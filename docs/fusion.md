# Fusion

Every boundary in the operator's launch tree, costed. Fusion is the only class that
has produced a large win here -- `07bbb05` took the forward pair to
`increment_passing_fwd` and `7cc2a03` took the two-tap column to one -- so the
boundaries were never enumerated. This is the enumeration. One boundary survives it.

## The tree, as it launches

Thirteen device operations a step at `B 4 H 18 T 2048 P 64 3N 240 L 64 G 1`, bf16.
Five are easy to miss: `reduce_rows`, which is `close_slots` on `dK`, and four
`aten` element-wise launches, one of which is the operator's own.

| kernel | grid | blocks | threads | regs | smem dyn B | occ_S | occ_R | waves | local ld/st sectors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `increment_passing_fwd` | (18,5,4) | 360 | 128 | 152 | 27,488 | 3 | 3 | 1.43 | 0 / 0 |
| `chunk_scan_fwd` | (18,4,32) | 2,304 | 256 | 216 | 87,040 | 1 | 1 | 27.43 | 0 / 0 |
| `chunk_prefix_bwd` | (576,4,1) | 2,304 | 128 | 38 | 2,304 | 19 | 12 | 2.29 | 0 / 0 |
| `start_passing_bwd` | (18,5,4) | 360 | 256 | 64 | 21,760 | 4 | 4 | 1.07 | 0 / 0 |
| `chunk_input_bwd` | (18,4,32) | 2,304 | 128 | 255 | 49,264 | 2 | 2 | 13.71 | 2,469,888 / 884,736 |
| `chunk_vector_bwd` | (2880,4,1) | 11,520 | 256 | 140 | 98,736 | 1 | 1 | 137.14 | 0 / 0 |
| `reduce_rows` | (64,72,1) | 4,608 | 256 | 40 | 0 | 8 | 6 | 9.14 | 0 / 0 |
| `vector_reduce` | (2048,4,1) | 8,192 | 128 | 56 | 0 | 16 | 9 | 10.84 | 0 / 0 |
| `boundary_bwd` | (32,4,18) | 2,304 | 128 | 18 | 0 | 16 | 21 | 2.29 | 0 / 0 |

`occ_S` and `occ_R` are `launch__occupancy_limit_shared_mem` and
`launch__occupancy_limit_registers` in blocks. Two kernels run one resident block on
both bars at once, `chunk_vector_bwd` and `chunk_scan_fwd`. One kernel spills at the
architectural register cap, `chunk_input_bwd`, 148.63 MB a launch.

## The regime, which decides whether deleted bytes are worth anything

A deleted byte converts at 0.334 of its time at the bus only on a kernel near its
bandwidth roof. Measured on a kernel at 31.4% of memory speed-of-light the byte term
bought nothing: a pre-registered -62.5 us for 127.69 MB collected zero of it. So
the regime is read per kernel before any byte is credited.

| kernel | mem SOL % | DRAM % | sm % | issue % | L2 % | DRAM read MB | DRAM write MB | regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `increment_passing_fwd` | 54.97 | 54.97 | 36.04 | 33.73 | 43.87 | 37.18 | 75.81 | latency |
| `chunk_scan_fwd` | 49.93 | 47.62 | 35.51 | 28.69 | 35.64 | 104.61 | 18.90 | latency |
| `chunk_prefix_bwd` | 40.96 | 12.02 | 40.96 | 49.59 | 72.12 | 0.00 | 0.47 | serial |
| `start_passing_bwd` | 65.72 | 65.72 | 37.72 | 39.10 | 55.47 | 43.98 | 75.21 | latency |
| `chunk_input_bwd` | 65.64 | 65.64 | 55.57 | 34.00 | 41.53 | 210.87 | 44.31 | latency |
| `chunk_vector_bwd` | 46.36 | 38.66 | 46.36 | 37.25 | 24.07 | 197.89 | 196.47 | latency |
| `reduce_rows` | 90.66 | 90.66 | 11.87 | 12.29 | 26.30 | 23.59 | 4.73 | roof |
| `vector_reduce` | 94.74 | 94.74 | 14.76 | 13.31 | 47.61 | 143.79 | 10.00 | roof |
| `boundary_bwd` | 40.47 | 40.47 | 12.68 | 19.65 | 23.85 | 0.99 | 0.47 | tiny |

Two kernels are at the roof and both are second-launch closures. Every kernel that
computes anything is latency-bound. **No fusion between two computing kernels may be
credited for its deleted bytes.** At 685.22 GB/s a deleted megabyte is 1.459 us at
the bus and 0.487 us at 0.334; in the latency regime it is 0. `120eb0f` is the
sharpest case: -29.14% of `chunk_vector_bwd`'s global store sectors bought -2.0 to
-3.0 us on a 1,820 us launch, because nothing waits on a global store and deleting
its sectors at 38% issue returns the issue slots and nothing else.

## The launch-overhead column is 9 us for the whole tree

Measured with `scripts/perf/profile_launch_gap.py`, not assumed: device idle inside
one step is **8.9 to 9.1 us over twelve gaps, 0.81 to 0.82 us a gap, 0.20% of a
4,150 us step**. Idle plus everything outside the device span bounds it at ~111 us,
2.4%. CUDA-graph replay moves the operator -29.1 to +12.8 us, an interval containing
zero. The paired-interval resolution floor is ~2.5 us, three launches wide.

So a fusion that deletes a launch and nothing else is worth 0.8 us and cannot be
measured. Both landed fusions deleted work and removed a launch incidentally:
`07bbb05` deleted 334.85 MB and a whole recurrence pass, `7cc2a03` deleted a tap of
the score GEMM. **Rank fusions by deleted work. The launch column is noise.**

## Two structural facts that kill most of the board before arithmetic

**Three interior tensors have two consumers each.** `dinc` is read by
`chunk_input_bwd` and by `chunk_vector_bwd`; `prefix_lp` and `prefix_q` are read by
`start_passing_bwd` and by `chunk_vector_bwd`. A pairwise fusion cannot delete the
write of a tensor a third launch still reads, so at most the one consumer's read
falls -- half a round trip, in the latency regime, for nothing.

**The chunk axis is serial in one kernel of every interesting pair.**
`increment_passing_fwd` and `start_passing_bwd` carry the chunk as an in-block loop
because their recurrences are serial; `chunk_scan_fwd`, `chunk_input_bwd` and
`chunk_vector_bwd` carry it on the grid. A consumer block indexed by chunk needs
what a producer block produced at one iteration of its serial loop, so the dependency
is grid-wide, not block-to-block. Making the consumer serial too collapses its grid
by `C = 32`.

## The board

Predictions were written before any arm was built or measured. Nothing in this table
is a measured delta.

| rank | boundary | crossing | bytes | L2 absorbs | producer regime | byte credit | fused smem | fits | residency effect | registers | grid | prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `chunk_vector_bwd` -> `reduce_rows` | `dK` slot rows, (4,18,10240,2,4) f32 | 23.59 MB w, 23.59 MB r, 4.72 MB w | none, to the sector | consumer at roof 90.66% | pass, not bytes | no new shared | yes | none, both bars stay 1 | 140 -> 140..144, cap 255, no spill | clean, 5 tiles are consecutive `x` | **-40.3 us** |
| 2 | `chunk_vector_bwd` -> `vector_reduce` | `dB`/`dC`/carry partials | 143.77 MB | none | consumer at roof 94.74% | real, the only one | -- | -- | -- | -- | -- | -170 us, costed separately |
| 3 | `chunk_input_bwd` -> `arrived` fill | `arrived`, (4,18,32) i32 | 9,216 B | -- | fill, 1.86 us | -- | none | yes | none | +0 | exact match, both (18,4,32) | -1.9 us |
| 4 | `chunk_input_bwd`/`chunk_vector_bwd` -> `boundary_bwd` | `carry_u` 0.59 MB, `carry_b` 0.12 MB | 1.42 MB round trip | -- | 40.47% | 0 | 0 | yes | none | 18 | two producers | dead, ceiling -5.2 us |
| 5 | `reduce_rows` -> `vector_reduce` | nothing crosses | 0 | -- | -- | 0 | 0 | yes | none | -- | 256 vs 128 threads | dead, -0.8 us |
| -- | `chunk_prefix_bwd` -> `start_passing_bwd` | `prefix_lp` 0.59, `prefix_q` 2.36 MB | 2.95 MB | 84% of the write already | 40.96% / 12.02% | 0 | 24,064 B | yes | none | 64 -> 72 loses the 4th block | serial vs grid | **dead, +25.1 to +26.1 us measured** |
| -- | `chunk_prefix_bwd` -> `chunk_vector_bwd` | same | 2.95 MB | same | same | 0 | 101,040 B | no | -- | -- | scan runs 5x, one per lane tile | dead, this is the fission that landed |
| -- | `start_passing_bwd` -> `chunk_input_bwd` | `dinc`, (4,18,32,64,240) bf16 | 70.78 MB | none | 65.72% | 0 to -34.5 us | 71,024 B | yes | **cib 2 -> 1, +294 us** | 255 and spilling, +24 live | serial reverse vs grid | dead, +294 us or worse |
| -- | `chunk_input_bwd` -> `chunk_vector_bwd` | `dlogp` 0.59, `dchunk_rot` 0.083, `dchunk_scale` 0.009 MB | 0.682 MB | yes, 5 consecutive blocks | 65.64% | -2.0 us at 0.334, ~0 real | 148,000 B | **no, short 47,648 B** | -- | 255 + 140 under one 255 cap | 128 vs 256 threads, lane looped vs gridded | dead, +204.5 to +2,650 us |
| -- | `increment_passing_fwd` -> `chunk_scan_fwd` | `zstart`, (4,18,32,64,240) bf16 | 70.78 MB, 141.56 MB round trip | **none, largest unabsorbed crossing in the tree** | 54.97% | 0 | 114,528 B | no | -- | 152 + 216 | fan-in 5, fan-out 32, whole-grid dependency | dead, unbuildable |

### Rank 1, in full

`chunk_vector_bwd` writes `held.dest = open_slots(dK, tiles=5, axis=-3)`, a
`(4,18,10240,2,4)` float32 slot buffer of 23,592,960 B, one row per lane tile;
`reduce_rows` sums the five slots into `dK` `(4,18,2048,2,4)` float32, 4,718,592 B.
`reduce_rows` reads 23,593,088 DRAM bytes against a 23,592,960 B buffer, so L2
absorbs none of it, and it runs at 90.66% of memory speed-of-light and 96% of the
41.3 us floor its own 28.31 MB implies. Its bytes are real and its pass is at its
ceiling: it cannot be optimized, only deleted.

Deleting it is the same arm that already closed `dtrans`. Every lane tile publishes
`PART_WORDS = 9` float32 a token to the chart, increments `arrived`, and the tile
that reads the last value sums the slots and runs `dtrans`' maps once. `dK`'s slot
rows are post-map, so closing them needs no map at all, only a float32 add of five
rows in slot order. `PART_WORDS` 9 -> 17 grows the chart from 26.54 to 50.14 MB and
the slot buffer's 23.59 MB of writes moves into it, a wash. The read-back is free:
the five tiles of one `(chunk, shard)` are consecutive `x` indices with `jstep`
innermost, and the producer's `dram__bytes_read.sum` moves +0.011 MB for 23.6 MB of
published words. At seventeen words the ninety consecutive blocks of one chunk hold
1.96 MB of chart against a 6 MB L2, where nine words hold 1.04 MB. Both resident.

Cost, at the 1:1 instruction conversion this kernel has three confirmations of
(311,145,984 warp-instructions to 1,706.7 us, 5.485 us a million). **The publish is
now a wash.** Before `120eb0f` the slot row went out one component at a time and
eight more words a token would have cost 0.18 M; that commit made both dK taps one
`STG.E.128` each, so publishing them into the chart instead of into the slot buffer
is the same two stores at a different address. What is left is the close, on one tile
of `tiles`: each of the 256 threads owns 2 of the 512 float32 in a `(b,h,c)` record,
so it reads back 5 slots x 2 words as 10 loads, adds 8, and stores 2 -- and the 2
stores already existed. Over 18,432 closing warps that is 0.18 M loads and 0.15 M
adds, **+0.33 M warp-instructions, +1.8 us.** Corroborated by that commit's own
freshly measured LSU rate on this kernel, 5.7 us a million.

Feasibility is the reason this is rank 1. It needs **no shared memory at all** -- the
chart is global, `chunk_vector_bwd`'s 98,736 B arena and its 1,616 B of headroom are
untouched, and both occupancy bars stay at the 1 they already read. Registers cannot
cost residency on a kernel already at one block on both bars, and there are 115
registers to the cap with zero local sectors today. The arrival counter, the fence,
the closing branch and the barrier are already built and already run. No grid changes.

Fidelity: five float32 partials summed in slot order before and after, no width
change and no order change. `dtrans[..., 3]` is untouched -- the existing nine chart
words and their maps are unchanged and the eight new words are read only by `dK`'s
own store.

**Prediction: -41.3 (pass, at its own floor) -0.8 (launch) +1.8 (instructions) =
-40.3 us, band -36 to -44.** The pass is priced at its DRAM floor rather than at a
duration, because NCU's `gpu__time_duration` runs 2.8x under to 1.6x over here and
has disagreed in sign. Corroboration only: 42.8 and 43.2 us on that counter, 43.8
and 44.2 us event-timed in `slinoss/_reduce.py` and `bwd/chunk_vector.py`.

### Two refusals, re-checked against today's arenas

**`chunk_input_bwd` + `chunk_vector_bwd` is refused, and the changed register
environment does not reopen it.** The recorded refusal is 142,400 B against a
101,376 B carveout, short by 41,024 B. Today's arenas are 49,264 and 98,736 B, so
the sum is **148,000 B against 100,352 B usable, short by 47,648 B** -- the tap
fusion made it worse by 5,600 B. The max form is no better: `chunk_vector_bwd`'s
five GEMM operands plus the transform table are 55,440 B live in one barrier
interval with zero staging and no cut separates them, so it leaves 1,616 B against
`chunk_input_bwd`'s 49,264 B. The register premise does not apply: the 120-registers-
no-spill figure is `chunk_vector_bwd` alone at eight warps, while
`chunk_input_bwd` sits at the 255 architectural cap and spills 2,469,888 load and
884,736 store sectors at this same shape today. Two bodies cannot be under one cap
when one saturates it. **The refusal was never about registers**, so a register
change cannot lift it. The grids refuse it independently: 128 against 256 threads,
2,304 against 11,520 blocks, and `chunk_input_bwd` walks the lane axis in a 5-trip
loop where `chunk_vector_bwd` puts it on the grid. Gridding the lane axis makes five
blocks recompute one `dU`, `carry_u` and `dlogp`, four fifths of an 883.0 us kernel
wasted; rolling `chunk_vector_bwd`'s lane tile into a loop carries a recorded
+204.5 us entry fee and changes `dtrans[..., 3]`'s reduction order. The crossing that
would be deleted is 0.682 MB, -2.0 us at 0.334 and near zero in the measured regime.

**Passing `dm` is still refused, and it is not a fusion.** The record is a
`(B,H,C,L,L)` float32 pair, 151 MB round trip against 107.6 us of compute deleted,
refused by 2x. The cost term was a byte figure and both endpoints are latency-bound,
so it is smaller than 221 us -- but the arm materializes an intermediate and deletes
no launch, which is backwards on the only metric that pays here. Off the board as a
de-fusion, not carried as a fusion.

## What the census closes

The reachable total is **-40.3 us for rank 1 and -1.9 for rank 3, -42.2 us**, or
-212.2 with the `vector_reduce` arm. Against a -2,068 us gap that is
2.0% and 10.3%. The arithmetic settles the class: `07bbb05` and
`7cc2a03` consumed the tree's fusable adjacencies, and what is left is one closure
and one tail. Seven boundaries are dead, five of them on shared capacity or grid
shape, which no register environment can reach.

Fusion is no longer the class to work. `chunk_vector_bwd` is 41% of the step at
46.36% of memory speed-of-light, 37.25% issue, one resident block on both bars, and
it converts instruction deletion 1:1 -- every 1% of its 311,145,984 instructions is
-17 us. The remaining gap is inside that kernel, not between kernels.

## Provenance

RTX A6000 sm_86, 84 SMs, 101,376 B opt-in carveout less 1,024 B driver.
Two NCU passes, `--clock-control none --cache-control none --replay-mode kernel`,
against `scripts/perf/profile_target.py --op so3ssd --shape acceptance --mode step`.
Device 0 at 68-100% utilization with 41,056-45,422 MiB foreign resident throughout,
so **every wall-clock absolute here is contaminated 1.6-2.2x**. Instruction,
register, shared-byte, occupancy-limit, sector and grid counts are contention-immune
and are what the refusals rest on. No arm was built and no delta was measured.
