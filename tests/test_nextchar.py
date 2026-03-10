from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "nextchar.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("nextchar_example", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_nextchar_defaults_match_reference_run() -> None:
    mod = _load_example_module()
    args = mod.build_parser().parse_args([])

    assert args.root == Path("/tmp/nextchar")
    assert args.steps == 80_000
    assert args.log_interval == 100
    assert args.eval_interval == 1_000
    assert args.eval_batches == 64
    assert args.batch_size == 12
    assert args.block_size == 128
    assert args.d_model == 96
    assert args.n_layers == 2
    assert args.d_state == 16
    assert args.expand == 2
    assert args.d_head == 32
    assert args.d_conv == 4
    assert args.chunk_size == 32
    assert args.lr == 3e-4
    assert args.weight_decay == 0.05
    assert args.grad_clip == 1.0


def test_nextchar_model_forward_shape() -> None:
    mod = _load_example_module()
    model = mod.NextCharLM(
        vocab_size=32,
        block_size=16,
        d_model=32,
        n_layers=2,
        d_state=8,
        expand=2,
        d_head=16,
        d_conv=4,
        chunk_size=8,
    )
    idx = torch.randint(0, 32, (2, 16), dtype=torch.long)

    logits = model(idx)

    assert logits.shape == (2, 16, 32)
    assert torch.isfinite(logits).all()


def test_nextchar_load_enwik8_chars_reads_small_local_file(tmp_path: Path) -> None:
    mod = _load_example_module()
    data_path = tmp_path / "tiny-enwik8"
    data_path.write_bytes(b"babc")

    train, val, test, vocab_size = mod.load_enwik8_chars(data_path)

    assert vocab_size == 3
    assert train.dtype == torch.long
    assert train.tolist() == [1, 0, 1, 2]
    assert val.numel() == 0
    assert test.numel() == 0
