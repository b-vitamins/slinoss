from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "blocks_lm.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("blocks_lm_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocks_lm_example_builds_and_runs_forward() -> None:
    mod = _load_example_module()
    model = mod.build_demo_model()
    x = torch.randint(0, 4096, (2, 8))

    logits = model(x)

    assert logits.shape == (2, 8, 4096)
