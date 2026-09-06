# Experiment runtime and routing matrix

This is the scheduling ledger for the frozen candidate benchmarked on NVIDIA
H200 NVL GPUs. Times are end-to-end job wall times unless marked as payload
times. The corresponding supervisor job records are archived under
`.sources/receipts/closure-2026-09-06/timings/`.

Sonata's externally imposed allocation boundary is 30 minutes. Use a
conservative admission rule:

- **Sonata:** only a completed, matched job with a worst observed wall time of
  at most 20 minutes. Keep at least ten minutes for launch variance.
- **Automation:** anything over 20 minutes, any 100k-step failure path, and any
  cell without a completed matched timing.
- Never infer eligibility from a partial trajectory. Update this ledger after a
  new cell completes, then route future copies from the measured wall time.

## State tracking

| Protocol/job unit | Completed work | Observed wall time | Route | Scheduling note |
| --- | ---: | ---: | --- | --- |
| Fixed-length A5, one cell, seed 0 solves at 10k | lengths 3--20 | 4.45--5.04 min/cell | Sonata | A shard of four known-solved cells completed in 19.15--19.44 min; do not make a larger shard. |
| Fixed-length A5, adverse seed, solves at 30k | length 16, seed 1 | 14.85 min | Sonata, one cell | Do not bundle with another uncertain seed. |
| Fixed-length A5, adverse seed, full 100k failure | length 14, seed 1 | 48.92 min | Automation | A new candidate or unsolved seed can take this path; the five-seed sweep is not generically Sonata-safe. |
| KLA/Walker Figure-1 shard | four cells | 19.15--19.44 min | Sonata | This is the largest demonstrated safe shard for the frozen candidate. |
| KLA/Walker Figure-1 shard | two cells | 10.41 min | Sonata | Comfortable margin. |
| PD-SSM A5:2/6/8/12 bundle, one layer | four cells | 13.08 min | Sonata | Completed bundle timing. |
| Walker A5 variable-length row | one cell, solved at 20k | not isolated | Automation | The benchmark receipt has no wall-time field; do not use a reaper-interrupted wrapper as a timer. |
| Walker S5 variable-length row, two layers | one full 100k cell | 87.97 min | Automation | Completed stable timing. |
| PD-SSM S5:4/8/32 | mixed bundle | 180.02 min | Automation | The bundle contains two 100k failure paths and is far beyond a Sonata window. |
| Regular parity + even-pairs bundle, two layers | parity 100k + even pairs 25k | 127.25 min | Automation | One uninterrupted bundle; parity exhausted the full budget and even pairs solved. |
| Remaining regular/Walker arithmetic | pending | pending | Automation | Promote a cell to Sonata only after this closure records an isolated wall time below 20 minutes. |

## MAD-Lab

| Protocol/job unit | Epochs run | Observed wall time | Route |
| --- | ---: | ---: | --- |
| Memorization + fuzzy recall bundle | 90 + 140 | 5.78 min | Sonata |
| Context recall | 90 | 3.32 min | Sonata |
| Noisy context recall | 90 | 3.45 min | Sonata |
| Compression | 290 | 6.19 min | Sonata |
| Selective copy | 540 | 34.59 min | Automation |

Do not submit all six MAD cells as one Sonata job. Put selective copy on
Automation and group the other five into bounded Sonata jobs.

## MQAR Figure 2

| Cell | Steps/epoch | Epochs run | Payload / job wall time | Route |
| --- | ---: | ---: | ---: | --- |
| 64:4 | 196 | 3 | 36.39 s / 47.40 s | Sonata |
| 128:8 | 196 | 3 | 66.06 s / 77.57 s | Sonata |
| 256:16 | 391 | 19 | 819.35 s / 856.98 s | Sonata |
| 512:64 | 782 | 32 | 2,749.77 s / 2,762.46 s | Automation |

The 512 cell is not a modest extension of the 256 cell: its published batch
ladder doubles the steps per epoch and each epoch contains twice as many
tokens. Even though it stopped early at epoch 31, it took 46.04 minutes
end-to-end and must never be sent to Sonata.

## Bounded language modelling

| Arm | Token budget | Payload train + validation | Job wall time | Route |
| --- | ---: | ---: | ---: | --- |
| SLinOSS | 13,107,200 | 312.75 s | 5.78 min | Sonata |
| Mamba3 | 13,107,200 | 202.62 s | 4.22 min | Sonata |

## Timing coverage

Fixed-length A5 records contain `wall_seconds`; MQAR records contain `seconds`;
LM records contain train and validation seconds. Native state-tracking and MAD
benchmark rows currently omit wall time, so their archived supervisor records
are the timing authority. After the frozen snapshot is landed, those two
harnesses should record `wall_seconds` directly so every future benchmark row
is self-timing.
