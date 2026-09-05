# SLinOSS experimental closure record — 2026-09-06

Status: in progress. This note records only completed benchmark rows. It will be
updated with the remaining state-tracking, MAD, and LM results, then stamped with
the final master commit.

## Candidate identity

- Final master commit: **TBD**
- Experiment source and harness commit: `3c64ddac2188811957aa8ef8473b930aa18c7640`
- Closure branch tip at this snapshot: `50ea79f`
- State-tracking geometry: `d_head=32`, `n_groups=8`, one layer for A5 and two
  layers for native S5/regular tasks.
- MAD geometry: `d_head=32`, `n_groups=4`, using the same mixer recurrence and
  initialization.
- Every raw receipt embeds the command argv, effective mixer construction,
  parameter count, dataset identity, source/harness tree hashes, and protocol.

## State tracking

Completed: **20/28 benchmark rows**. Solved: **20/20 completed rows**.

### Walker/KLA Figure-1-style fixed-length A5 sweep

The paper rule is used: a `(length, depth)` cell is solved if any of five seeds
reaches 90%. These are one-layer results with the source-exact Merrill payloads,
100,000-step allowance, batch 256, a length-2 auxiliary batch each step, peak
learning rate `1e-3`, dropout 0.1, and evaluation every 10,000 steps.

| Length | Best accuracy | Step | Status |
| ---: | ---: | ---: | :--- |
| 3 | 100.0000% | 10,000 | solved, depth 1 |
| 4 | 100.0000% | 10,000 | solved, depth 1 |
| 5 | 100.0000% | 10,000 | solved, depth 1 |
| 6 | 100.0000% | 10,000 | solved, depth 1 |
| 7 | 100.0000% | 10,000 | solved, depth 1 |
| 8 | 100.0000% | 10,000 | solved, depth 1 |
| 9 | 100.0000% | 10,000 | solved, depth 1 |
| 10 | 100.0000% | 10,000 | solved, depth 1 |
| 11 | 100.0000% | 10,000 | solved, depth 1 |
| 12 | 100.0000% | 10,000 | solved, depth 1 |
| 14 | 99.99996% | 10,000 | solved, depth 1 |
| 16 | 100.0000% | 10,000 | solved, depth 1 |
| 18 | 99.99989% | 10,000 | solved, depth 1 |
| 20 | 99.99975% | 10,000 | solved, depth 1 |

The fixed-length model has 456,804 total parameters, of which 408,104 are in
the single mixer. Length 14 is legitimately solved by seed 0 under the published
"any seed" rule; a later seed-1 attempt in the same raw receipt did not solve.

All 14 fixed-length cells are solved at depth 1.

### Native long-range/group protocols

- Walker A5, one layer: **99.97860%** overall at step 20,000; the longest
  `256–287` band is **100.0000%**. Total/mixer parameters: 423,780/408,104.
- PD-SSM reconstructed `A5:2`: **100.0000%** at step 5,000, one layer.
- PD-SSM reconstructed `A5:6`: **100.0000%** at step 5,000, one layer.
- PD-SSM reconstructed `A5:8`: **100.0000%** at step 5,000, one layer.
- PD-SSM reconstructed `A5:12`: **100.0000%** at step 10,000, one layer.
- PD-SSM reconstructed `S5:4`: **100.0000%** at step 5,000, two layers, with
  **100.0000% in every reported length band**. Total/mixer parameters:
  832,712/816,208.

Remaining native rows: PD-SSM `S5:8`, `S5:32`; Walker S5; four released regular
tasks; and the Walker arithmetic-with-brackets extension.

## MAD-Lab

Protocol: `kla-paper-v2`, seed 12345, one run per task. Three of six tasks are
complete.

| Task | Best accuracy | Best epoch | Final/stop | Published comparison |
| :--- | ---: | ---: | :--- | :--- |
| Memorization | 100.0000% | 20 | 100.0000% at epoch 90 | above KLA+ 99.94% |
| Compression | 83.7451% | 220 | 82.9785% at epoch 290 | below KLA+ 88.87% |
| Fuzzy recall | 41.5355% | 70 | 38.6823% at epoch 140 | below KLA 45.70%; near KLA+ 43.32% |

Pending: context recall, noisy context recall, and selective copy.

## Language modelling

Pending: one synchronized 10-minute SLinOSS probe and one matched-parameter
Mamba3 probe. The final record will include matched-token validation loss and
wall-clock throughput.

## Raw receipts

The point-in-time receipt bundle is under
`.sources/receipts/closure-2026-09-06/`. `SHA256SUMS` seals this snapshot.
Receipt files can contain duplicate completed attempts from supervisor retries;
the benchmark ledger above deduplicates by protocol cell and reports the best
valid seed according to that protocol's declared selection rule.

At final closure, live result files will be pulled again, this note will be
completed, and `Final master commit` will be replaced with the landed commit.
