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


def test_mixer_tail_rowwise_fwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.mixer_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.MixerTailPerfConfig(
        batch=2,
        heads=3,
        T=5,
        P=4,
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_mixer_tail_rowwise_fwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 6

    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 6
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.heads, cfg.T, cfg.P)
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert tuple(launch_args[0][4].shape) == (cfg.heads,)
    assert launch_args[0][4].dtype == cfg.d_skip_dtype
    assert tuple(launch_args[0][5].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])


def test_mixer_tail_rowwise_bwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.mixer_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.MixerTailPerfConfig(
        batch=2,
        heads=3,
        T=5,
        P=4,
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_mixer_tail_rowwise_bwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 11

    runner.prepare()
    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 11
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.heads, cfg.T, cfg.P)
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert tuple(launch_args[0][5].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert launch_args[0][4].dtype == cfg.d_skip_dtype
    assert tuple(launch_args[0][9].shape) == (cfg.heads,)
    assert tuple(launch_args[0][10].shape) == (cfg.hidden_dim,)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])


def test_ffn_norm_fwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.block_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.FfnPerfConfig(
        batch=2,
        T=5,
        d_model=12,
        hidden_dim=16,
        kind="gelu",
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_ffn_norm_fwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 3

    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 3
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.T, cfg.d_model)
    assert tuple(launch_args[0][1].shape) == (cfg.d_model,)
    assert tuple(launch_args[0][2].shape) == (cfg.batch, cfg.T, cfg.d_model)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])


def test_ffn_norm_bwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.block_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.FfnPerfConfig(
        batch=2,
        T=5,
        d_model=12,
        hidden_dim=16,
        kind="gelu",
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_ffn_norm_bwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 5

    runner.prepare()
    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 5
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.T, cfg.d_model)
    assert tuple(launch_args[0][2].shape) == (cfg.batch, cfg.T, cfg.d_model)
    assert tuple(launch_args[0][4].shape) == (cfg.d_model,)
    assert launch_args[0][4].dtype == torch.float32
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])


def test_ffn_activation_fwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.block_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.FfnPerfConfig(
        batch=2,
        T=5,
        d_model=12,
        hidden_dim=16,
        kind="swiglu",
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_ffn_activation_fwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 2

    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 2
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.T, cfg.projected_dim)
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])


def test_ffn_activation_bwd_runner_matches_fused_launcher_contract(
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
        ncu_kernels.block_cute_common,
        "make_fake_tensor_arg",
        lambda tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(ncu_kernels.cute, "compile", fake_compile)

    cfg = ncu_kernels.FfnPerfConfig(
        batch=2,
        T=5,
        d_model=12,
        hidden_dim=16,
        kind="swiglu",
        dtype=torch.float16,
        device="cpu",
    )

    runner = ncu_kernels._build_ffn_activation_bwd_runner(cfg)

    assert len(compile_args) == 1
    assert len(compile_args[0]) == 3

    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 3
    assert tuple(launch_args[0][0].shape) == (cfg.batch, cfg.T, cfg.projected_dim)
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.T, cfg.hidden_dim)
    assert tuple(launch_args[0][2].shape) == (cfg.batch, cfg.T, cfg.projected_dim)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])
