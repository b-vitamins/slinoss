"""State tracking: finite automata and group word problems, past the trained length.

An arm trains on lengths 3 to 40 and is scored on lengths 40 to 256, so what is measured
is whether the recurrence's state survives beyond the horizon it was fit on. The suite is
`expressive-sparse-state-space-model`'s table 2 -- ``parity``, ``even_pairs``,
``cycle_nav``, ``mod_arith_no_brack`` -- plus the bracketed arithmetic and the group word
problems, which cross out of the solvable regime at ``A_5``.

Two upstream trees were read to build this one, and neither is imported. The task
generators, the seed scheme, the padding rule and the residual block come from
`structured-linear-cdes` (``data_dir/fl_tasks/``, ``data_dir/dataloaders.py``,
``models/mamba.py``); the optimization protocol, its constants and its schedule come from
`expressive-sparse-state-space-model`'s ``state_tracking_PyTorch``
(``train.py``, ``experiment_configs/``, ``models/lr_scheduler.py``), which carries the
generators byte-identical.

Every divergence from upstream is stated in the docstring of the module that owns it: the
element order in ``groups.py``, the local generator in ``tasks.py``, the unbounded train
split and its absent shuffle in ``instances.py``, the ``use_glu`` default in ``model.py``,
the single optimizer step and the corrected accumulation window in ``train.py``.

Seven modules, one job each:

- ``groups.py`` -- finite groups by Cayley table. No torch, no dependency.
- ``tasks.py`` -- the five automaton generators and the group word problem.
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
a carried state and which no tree this harness is compared against reports on this axis.
None of them changes a number reported here.
"""
