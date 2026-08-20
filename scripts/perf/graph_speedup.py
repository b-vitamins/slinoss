"""What CUDA-graph capture is worth on a whole step.

Two arms in one paired loop: the eager step, and a replay of the same step
captured. The arms run the same kernels over the same shapes, so the difference is
per-launch host cost and nothing else, and it is reported as an interval that either
excludes zero or licenses no claim.

    python3 scripts/perf/graph_speedup.py --mode decode
    python3 scripts/perf/graph_speedup.py --mode train --iters 10

The geometry defaults to the one the throughput target is stated at. Decode is one
token per sequence, which is launch-bound by construction; train is forward,
backward, and a capturable optimizer over a full sequence.

Neither arm copies its arguments: the replay is handed the buffers it captured, so
what is timed is the step and not a token copy a real loop would do either way.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from slinoss.config import SLinOSSConfig
from slinoss.graph import capture, capture_decode
from slinoss.perf.device import device_ordinal, require_cuda
from slinoss.perf.timing import Throughput, measure_paired
from slinoss.perf.units import Count
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

MODES = ("decode", "train")
DTYPE = torch.bfloat16
"""The dtype the kernel path runs. float32 falls back to the reference scan."""

EAGER = "eager"
GRAPH = "graph"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="decode")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--seqlen", type=int, default=2048, help="Tokens per train step."
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=64,
        help="Tokens the decode state is advanced by before the step is captured, so "
        "the measured step runs against carries a real loop would have.",
    )
    parser.add_argument("--d-model", type=int, default=576)
    parser.add_argument("--d-state", type=int, default=240)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--layers", type=int, default=13)
    parser.add_argument("--vocab", type=int, default=50257)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SLinOSSConfig:
    """The geometry both arms run."""
    return SLinOSSConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        d_head=args.d_head,
        chunk_size=args.chunk,
        n_layers=args.layers,
        ffn_ratio=4.0,
        vocab_size=args.vocab,
    )


def build_decode(
    args: argparse.Namespace, config: SLinOSSConfig, device: torch.device
) -> tuple[Callable[[], object], Callable[[], object], Count]:
    """One decode step, eager and captured.

    Each arm owns its state: the two advance independently over the loop and a
    replay writes the buffers its capture recorded.

    Args:
        args: The command line.
        config: The geometry.
        device: Where to run.

    Returns:
        The eager arm, the graph arm, and tokens per call.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(config, device=device).to(DTYPE)
    vocab = config.vocab_size
    assert vocab is not None
    prompt = torch.randint(0, vocab, (args.batch, args.prefill), device=device)
    token = torch.randint(0, vocab, (args.batch, 1), device=device)

    eager_state = StackState.allocate(config, args.batch, device=device, dtype=DTYPE)
    graph_state = StackState.allocate(config, args.batch, device=device, dtype=DTYPE)
    stack(prompt, eager_state)
    stack(prompt, graph_state)
    step = capture_decode(stack, graph_state)

    def eager() -> object:
        return stack(token, eager_state)

    return eager, lambda: step(*step.inputs), Count(args.batch)


def build_train(
    args: argparse.Namespace, config: SLinOSSConfig, device: torch.device
) -> tuple[Callable[[], object], Callable[[], object], Count]:
    """One training step, eager and captured.

    ``capturable=True`` keeps the optimizer's step count on the device, without
    which the update reads a host value a replay would never advance. The learning
    rate is zero so that the parameters both arms share do not drift over the loop;
    the update does the same work at any rate.

    Args:
        args: The command line.
        config: The geometry.
        device: Where to run.

    Returns:
        The eager arm, the graph arm, and tokens per call.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(config, device=device).to(DTYPE)
    vocab = config.vocab_size
    assert vocab is not None
    ids = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    labels = torch.randint(0, vocab, (args.batch, args.seqlen), device=device)
    optimizer = torch.optim.AdamW(stack.parameters(), lr=0.0, capturable=True)

    def train(x: Tensor, target: Tensor) -> Tensor:
        optimizer.zero_grad(set_to_none=False)
        logits = stack(x)
        # Classes come from the config, never from the logits' last extent: an
        # aligned head pads its output width past the vocabulary, and a pad
        # column is not a class a label indexes.
        # The float32 copy is not removable through aten. log_softmax(dtype=)
        # reaches the kernel that reads low precision and accumulates in float32
        # for float16 only; bfloat16 casts first.
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1)[:, :vocab].float(), target.flatten()
        )
        loss.backward()
        optimizer.step()
        # Detached: a returned loss keeps its autograd graph alive across replays,
        # and the AccumulateGrad nodes in it then belong to the capture stream. The
        # next eager backward finds that mismatch and synchronizes for it, which
        # would show up as the eager arm being slower than it is.
        return loss.detach()

    step = capture(train, ids, labels, warmup=3)

    def eager() -> object:
        return train(ids, labels)

    return eager, lambda: step(*step.inputs), Count(args.batch * args.seqlen)


def build_arms(
    args: argparse.Namespace, config: SLinOSSConfig, device: torch.device
) -> tuple[Callable[[], object], Callable[[], object], Count]:
    """Dispatch on ``--mode``."""
    if args.mode == "decode":
        return build_decode(args, config, device)
    return build_train(args, config, device)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure one mode.

    Returns:
        Process exit status. Nonzero if the comparison licenses no claim.

    Raises:
        RuntimeError: If the requested device is not a usable CUDA device.
    """
    args = parse_args(argv)
    device = require_cuda(args.device)
    config = build_config(args)
    eager, graphed, tokens = build_arms(args, config, device)
    out = measure_paired(
        EAGER,
        eager,
        GRAPH,
        graphed,
        label=f"graph/{args.mode}",
        iters=args.iters,
        warmup=args.warmup,
        device=device,
    )
    print(f"device {device_ordinal(device)}  {out.timed.clocks}")
    print(
        f"geometry {config.n_layers} layers  d_model {config.d_model}  "
        f"3N {config.d_state}  d_head {config.d_head}  chunk {config.chunk_size}  "
        f"heads {config.n_heads}  batch {args.batch}"
    )
    for name in (EAGER, GRAPH):
        rate = Throughput.of(name, tokens, out.timed.region(name).spread)
        print(
            f"{name:6s} {rate.duration_us:12,.3f} us  "
            f"spread {rate.spread_pct:6,.3f}%  {rate.throughput_tps:14,.1f} tok/s"
        )
    print(f"speedup {out.comparison.speedup_ratio:.4f}x")
    print(out.comparison.verdict())
    return 0 if out.comparison.resolves else 1


if __name__ == "__main__":
    raise SystemExit(main())
