from __future__ import annotations

import importlib

import pytest
import torch


def test_scanprep_bwd_runner_matches_fused_launcher_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass")
    ncu_kernels = importlib.import_module("scripts.perf._ncu_kernels")

    compile_args: list[tuple[object, ...]] = []
    launch_args: list[tuple[torch.Tensor, ...]] = []

    def fake_compile(_kernel: object, *args: object, **_: object):
        compile_args.append(args)

        def compiled(*runtime_args: torch.Tensor) -> None:
            launch_args.append(runtime_args)

        return compiled

    monkeypatch.setattr(
        ncu_kernels,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.ScanPrepPerfConfig(
        batch=2,
        heads=2,
        T=5,
        P=4,
        N=8,
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_scanprep_bwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 16

    runner.prepare()
    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 16
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.T, cfg.heads, 4, cfg.N)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])
