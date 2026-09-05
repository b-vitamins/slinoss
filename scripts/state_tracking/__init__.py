"""State tracking: source-labelled benchmark contracts, past the trained length.

An arm trains on lengths 3 to 40 and is scored on lengths 40 to 256, so what is measured
is whether the recurrence's state survives beyond the horizon it was fit on. A bare CLI
invocation is exactly `expressive-sparse-state-space-model`'s released four-task suite:
``parity``, ``even_pairs``, ``cycle_nav`` and ``mod_arith_no_brack``. Other task families
must be selected through a named profile.

Two upstream trees were read to build this one, and neither is imported. The task
generators, the seed scheme, the padding rule and the residual block come from
`structured-linear-cdes` (``data_dir/fl_tasks/``, ``data_dir/dataloaders.py``,
``models/mamba.py``); the optimization protocol, its constants and its schedule come from
`expressive-sparse-state-space-model`'s ``state_tracking_PyTorch``
(``train.py``, ``experiment_configs/``, ``models/lr_scheduler.py``), which carries the
generators byte-identical.

There are two different group benchmarks in the literature. Walker/Merrill sample every
group element and label every prefix; use ``--profile walker-group-prefix --task A5``.
PD-SSM uses a small generator alphabet and a 60- or 120-state output alphabet; use
``--profile pdssm-groups-reconstruction``. PD-SSM did not release the group generator or
the randomly selected extra permutations. Those seven rows are therefore recorded as
reconstructions, never exact: the two-generator A5 actions come from IBM's predecessor
release and every extra set is a deterministic, fully recorded stand-in. Bare ``A5`` and
``pdssm:A5:2`` cannot be mixed under one profile.

Every divergence from upstream is stated in the docstring of the module that owns it: the
element order in ``groups.py``, source and reconstruction status in ``tasks.py``, the
local generator and absent shuffle in ``instances.py``, the ``use_glu`` default in
``model.py``, and the corrected accumulation window in ``train.py``.

Seven modules, one job each:

- ``groups.py`` -- finite groups by Cayley table. No torch, no dependency.
- ``tasks.py`` -- generators plus fail-closed benchmark contracts and provenance.
- ``instances.py`` -- split configuration, seed streams, padding, batching.
- ``model.py`` -- the post-norm scaffold every arm is scored on.
- ``mixers.py`` -- the name-keyed sequence-mixer registry.
- ``train.py`` -- the protocol, the metrics, the length bands.
- ``run.py`` -- the CLI driver; emits one JSON record per arm.

What upstream carries and this does not: wandb, hydra, fire, pandas, polars, einops,
numpy, ``abstract_algebra``, ``torch.utils.data``, worker processes, checkpoint files, the
``A5={length}.csv`` corpus under ``data_dir/illusion/`` (empty in the mirror, so that path
is unreachable), and the thirteen further generators under ``fl_tasks/`` -- Deletang's
context-free and context-sensitive transductions, which measure a transduction rather than
a carried state. The missing PD-SSM group generator is not silently filled in: its
reconstruction status, exact generator labels and selection rule are in every record.
"""
