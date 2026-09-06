# SLinOSS experimental closure record — 2026-09-06

Status: complete. This note records the frozen candidate, every requested
benchmark row, and the content-addressed evidence used to land it.

## Candidate identity

- Frozen closure commit: `37f04d5a46d79bd66f96ce2397c2670082fc3e11`
- Benchmarked mixer source tree: `5c340024befe5dfc9627b77d9ba8443776518683`.
  Every result receipt carries this exact tree, irrespective of the later
  harness-only commit from which it was launched.
- Final harness trees: state tracking
  `768670d70a900f1f2a389db6892f22d6dc07b021`, MAD-Lab
  `dc268e11cfdcac2a9033fbb1def7968ef097b41b`, language modelling
  `9e1eb46258477011764cf696ffdb23cbcaa31275`, and MQAR
  `6a9f5b3cae7258f40289cac78e0dba42f4161814`.
- State-tracking geometry: `d_head=32`, `n_groups=8`, one layer for A5 and two
  layers for native S5/regular tasks.
- MAD geometry: `d_head=32`, `n_groups=4`, using the same mixer recurrence and
  initialization.
- Every raw receipt embeds the command argv, effective mixer construction,
  parameter count, dataset identity, source/harness tree hashes, and protocol.

## State tracking

Completed: **28/28 benchmark rows**. Solved: **22/28 rows**.

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
- PD-SSM reconstructed `S5:8`: **2.0264%** at step 100,000, two layers; the
  longest reported band is **0.0000%**. This cell is not solved.
- PD-SSM reconstructed `S5:32`: **1.0376%** at step 35,000, two layers; the
  longest reported band is **3.5714%**. Its final step-100,000 accuracy is
  **0.8423%**. This cell is not solved.
- Walker S5: **5.7610%** at step 100,000, two layers; the longest reported
  band is **3.5301%**. This cell is not solved. Total/mixer parameters:
  847,560/816,208.
- Released regular `cycle_nav`: **100.0000%** at step 90,000, two layers,
  with **100.0000% in every reported length band**. Total/mixer parameters:
  819,033/816,208.
- Released regular `parity`: **56.6650%** at the full 100,000-step endpoint,
  two layers; the longest reported band is **50.0000%**. This cell is not
  solved. Total/mixer parameters: 817,491/816,208.
- Released regular `even_pairs`: **100.0000%** at step 25,000, two layers,
  with **100.0000% in every reported length band**. Total/mixer parameters:
  817,491/816,208.
- Released regular `mod_arith_no_brack`: **71.6309%** at step 80,000, two
  layers; the longest reported band is **21.8750%**. Its final step-100,000
  accuracy is **69.1528%**. This cell is not solved. Total/mixer parameters:
  819,290/816,208.
- Walker extension `mod_arith_w_brack`: **33.6060%** at the full step-100,000
  endpoint, two layers; the longest reported band is **21.4286%**. This cell is
  not solved. Total/mixer parameters: 819,804/816,208.

## MAD-Lab

Protocol: `kla-paper-v2`, seed 12345, one run per task. All six tasks are
complete.

| Task | Best accuracy | Best epoch | Final/stop | Published comparison |
| :--- | ---: | ---: | :--- | :--- |
| Memorization | 100.0000% | 20 | 100.0000% at epoch 90 | above KLA+ 99.94% |
| Context recall | 99.9986% | 20 | 99.9121% at epoch 90 | above KLA+ 99.94% |
| Noisy context recall | 100.0000% | 20 | 99.9051% at epoch 90 | above KLA+ 99.95% |
| Compression | 83.7451% | 220 | 82.9785% at epoch 290 | below KLA+ 88.87% |
| Fuzzy recall | 41.5355% | 70 | 38.6823% at epoch 140 | below KLA 45.70%; near KLA+ 43.32% |
| Selective copy | 72.7246% | 470 | 72.6953% at epoch 540 | below KLA+ 91.45% and the historical SLinOSS 94.71% |

Selective copy is a material regression and does not satisfy the campaign's
"respectable MAD" gate.

## MQAR

This is the Zoology ICLR-24 Figure-2 task. The authoritative paper and
[`iclr24` release](https://github.com/HazyResearch/zoology/releases/tag/iclr24)
contain four matched train/test cells: `64:4`, `128:8`, `256:16`, and
`512:64`. The current Zoology `main` config comments out the last cell; the
in-tree preset inherited that omission until `ef93e4a`, which restores it.

The paper and its current config specify two layers, while the release-tag
script conditionally set four layers for non-attention mixers. These receipts
use the paper-text/current-config interpretation: two layers. This is a bounded
point profile at `d_model=128`, learning rate `1e-3`, and model/data seed 123,
not the published selection over four widths and four learning rates. Each cell
uses vocabulary 8,192, 100,000 training rows, 3,000 test rows, padding filler,
64 epochs maximum, weight decay 0.1, and early stopping above 99% accuracy.
The model has 1,329,704 parameters and uses the default SLinOSS constructor
geometry (`d_head=64`, `n_groups=1`) in both layers.

| Cell (length:pairs) | Best example accuracy | Best position accuracy | Best epoch | Runtime | Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `64:4` | 99.9500% | 99.9500% | 2 | 36.390 s | complete; zero leakage |
| `128:8` | 99.9375% | 99.9375% | 2 | 66.062 s | complete; zero leakage |
| `256:16` | 99.0604% | 99.0604% | 18 | 819.350 s | complete; zero leakage |
| `512:64` | 99.7661% | 99.7661% | 31 | 2,749.766 s | complete; zero leakage |

All four canonical Figure-2 cells are complete and exceed 99% example accuracy.

## Language modelling

Both synchronized, bounded arms are complete. They use the same GPT-NeoX
tokenizer, 2,048-token context,
13,107,200-token budget, 100 optimizer steps, seed 0, and approximately 40.5M
parameters.

- SLinOSS: 40,501,976 parameters; validation loss **8.556816**; validation BPB
  **2.634495**; 306.659 s synchronized training time; 6.091 s validation time;
  **42,741.9 tokens/s**.
- Mamba3: 40,525,424 parameters; validation loss **7.470294**; validation BPB
  **2.299974**; 200.472 s synchronized training time; 2.146 s validation time;
  **65,381.6 tokens/s**.

The result is finite and materially better than the old-master faceplant
(validation loss 9.9262), but its logged pre-clip gradient norm grew from 1.84
at step 0 to roughly 0.5M at the end and exceeded 1.0M mid-run. At a parameter
spread of only 23,448 parameters (0.058%), SLinOSS trails Mamba3 by **1.086523
validation-loss points** and delivers **65.37% of Mamba3 throughput**. It is no
longer a NaN/divergence faceplant, but it is not yet competitive or healthy.

## Raw receipts

The complete receipt bundle is under
`.sources/receipts/closure-2026-09-06/`. `SHA256SUMS` seals this snapshot.
Receipt files can contain duplicate completed attempts from supervisor retries;
the benchmark ledger above deduplicates by protocol cell and reports the best
valid seed according to that protocol's declared selection rule.

Every one of the 48 result objects parses as JSON and identifies mixer source
tree `5c340024befe5dfc9627b77d9ba8443776518683`. Receipts whose provenance is
marked dirty list only result/export files; no source or harness file differs
from its recorded tree. The first bracketed-arithmetic attempt was terminated
externally; its supervisor record preserves two attempts and one failed attempt,
while the reported result is the uninterrupted 97.03-minute second attempt.

The hardware-specific runtime and routing ledger is
`.sources/notes/EXPERIMENT_RUNTIME_MATRIX.md`; its end-to-end timing records are
sealed alongside the benchmark receipts under `timings/`.

## Verification

- `shasum -a 256 -c SHA256SUMS`: every raw result and timing receipt passes.
- Full local suite: **1,622 passed, 230 environment-gated skips, 0 failed**.
- `ruff check .`: passed.
- `git diff --check`: passed.
- Final repository trees agree with the tree identities recorded above.
