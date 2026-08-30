"""Language-modelling harness: pretrain an arm, score it, put it in one table.

Six parts, each usable alone.

    corpus      tokenize a stream once into two files and a manifest that names them
    data        read those files, deterministically, with numpy and nothing else
    sizing      choose the width that puts an arm at a target parameter count
    model       the scaffold, identical across arms, with the mixer swapped in
    train       the loop: accumulate, clip, step, evaluate, record
    run         the command line, one arm per invocation, and the table

Two more sit at the boundary. ``checkpoint`` writes an arm in a form that names its own
corpus and configuration, and ``shim`` hands that checkpoint to ``lm_eval`` for the eight
zero-shot benchmarks. The zero-shot half is the only place this harness takes an outside
dependency, and it takes it deliberately: the task files encode the prompt format, the split
and the accuracy convention per benchmark, and a reimplementation of those is where
comparability with published numbers dies.

Training imports ``torch`` and ``numpy``. Nothing else.
"""
