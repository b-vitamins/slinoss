"""Multi-query associative recall: generator, model, protocol, driver.

The task and every protocol constant come from zoology (Arora, Eyuboglu et al.,
"Zoology: Measuring and Improving Recall in Efficient Language Models"). Two copies of
that tree were read to build this one: the standalone fork, whose generator lives in
``zoology/data/multiquery_ar.py``, and the ICLR24-era vendored copy, whose generator
lives in ``zoology/data/associative_recall.py``. The two generators are bit-identical on
their shared surface and their optimization protocols are the same code; ``instances.py``
names the one place they differ.

Every divergence from upstream is stated in the docstring of the module that owns it: the
generator's two in ``instances.py``, the backbone's three plus its unported config fields
in ``model.py``, the loss reduction in ``train.py``.

Six modules, one job each:

- ``instances.py`` -- the generator. numpy only, no torch, so it runs anywhere.
- ``tasks.py`` -- segment pools, the split-seed scheme, batching, the leakage measure.
- ``model.py`` -- the two-layer pre-norm backbone the task is scored on.
- ``mixers.py`` -- the name-keyed sequence-mixer registry.
- ``train.py`` -- the optimization protocol and the metrics.
- ``run.py`` -- the CLI driver; emits one JSON record per cell.

What upstream carries and this does not: wandb, pydantic, hydra, tqdm, pandas, einops,
stochastic depth, the continuous-input model, the non-cross-entropy losses, and the
compositional, forgetting and stacked task variants. None of them is reachable from an
MQAR config, and none changes a number this harness reports.
"""
