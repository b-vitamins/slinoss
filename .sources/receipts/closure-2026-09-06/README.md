# Closure receipt bundle

This directory is a point-in-time snapshot of completed JSONL result rows for
the SLinOSS closure campaign begun on 2026-09-06.

- `sonata/`: fixed-length A5, Walker/PD A5, completed MAD rows, bounded LM
  records, and completed MQAR cells.
- `automation/`: fixed-length A5 length 14 and native PD-SSM S5 rows.
- `timings/`: immutable supervisor job records used for end-to-end wall times in
  `.sources/notes/EXPERIMENT_RUNTIME_MATRIX.md`.
- `SHA256SUMS`: content hashes for the raw receipts.

The JSONL rows are the primary evidence. Each row includes its exact command,
effective configuration, protocol, parameter accounting, dataset identity, and
repository provenance. Empty/live result files were not copied into this
snapshot. The companion ledger is
`.sources/notes/EXPERIMENTAL_CLOSURE_2026-09-06.md`.
