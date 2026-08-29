"""The time series classification axis: the UEA benchmark, in house.

The data path, in the order it runs. Everything here is prep time except the last line.

    :mod:`scripts.tsc.reader`     the archive's ARFF and ``.ts`` files, parsed here
    :mod:`scripts.tsc.corpus`     the reference preprocessing, into ``.npy`` plus a manifest,
                                  with a CLI: ``python3 -m scripts.tsc.corpus``
    :mod:`scripts.tsc.prng`       JAX's Threefry stream, so a seed's partition reproduces
    :mod:`scripts.tsc.split`      the partition, the targets, the time channel
    :mod:`scripts.tsc.batching`   the reference's two iteration shapes, over resident tensors

The run.

    :mod:`scripts.tsc.protocol`   the published per-dataset settings and the reference bars
    :mod:`scripts.tsc.model`      the scaffold the bars were produced on, mixer swapped in
    :mod:`scripts.tsc.linoss`     the reference recurrence, in torch, as one such mixer
    :mod:`scripts.tsc.mixers`     which mixer goes in, and what a baseline may be given
    :mod:`scripts.tsc.train`      the loop and the stopping rule
    :mod:`scripts.tsc.sweep`      a lattice of runs, enumerated once and sliced by cost
    :mod:`scripts.tsc.run`        the driver: ``python3 -m scripts.tsc.run --shard i/n``

Nothing here imports JAX, ``sktime``, ``pandas``, ``arff``, ``pickle``, a logging service, or a
training framework. The data path is numpy and the loop is torch.
"""

from __future__ import annotations
