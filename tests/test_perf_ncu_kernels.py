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

    def fake_scanprep_fwd_cute_with_aux(
        value: torch.Tensor,
        params_flat: torch.Tensor,
        bc: torch.Tensor,
        *,
        n_heads: int,
        d_state: int,
        d_head: int,
        **unused_kwargs: object,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del params_flat, bc, unused_kwargs
        batch_size, t_size, value_width = map(int, value.shape)
        u = torch.empty(
            (batch_size, n_heads, t_size, d_head),
            device=value.device,
            dtype=value.dtype,
        )
        m = torch.empty(
            (batch_size, n_heads, t_size, 2),
            device=value.device,
            dtype=torch.float32,
        )
        k = torch.empty(
            (batch_size, n_heads, t_size, 2, 2),
            device=value.device,
            dtype=torch.float32,
        )
        b = torch.empty(
            (batch_size, n_heads, t_size, 2 * d_state),
            device=value.device,
            dtype=value.dtype,
        )
        c = torch.empty_like(b)
        assert value_width > 0
        coeff_aux = torch.empty(
            (batch_size, n_heads, ncu_kernels.COEFF_AUX_FIELDS, t_size),
            device=value.device,
            dtype=torch.float32,
        )
        return u, m, k, b, c, coeff_aux

    def fake_compile(_kernel: object, *args: object, **_: object):
        compile_args.append(args)

        def compiled(*runtime_args: torch.Tensor) -> None:
            launch_args.append(runtime_args)

        return compiled

    monkeypatch.setattr(
        ncu_kernels,
        "scanprep_fwd_cute_with_aux",
        fake_scanprep_fwd_cute_with_aux,
    )
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
    assert len(compile_args[0]) == 13

    runner.prepare()
    runner.launch()

    assert len(launch_args) == 1
    assert len(launch_args[0]) == 13
    assert tuple(launch_args[0][1].shape) == (cfg.batch, cfg.heads, cfg.T, 2 * cfg.N)
    assert runner.effective_bytes == ncu_kernels._tensor_bytes(*launch_args[0])
