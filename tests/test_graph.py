"""Capture and replay of one step.

What a graph can get wrong is not arithmetic. The kernels are the same kernels; a
replay either issues them over the addresses the capture recorded or it does not.
So each test here is one way the recording can be wrong while every kernel is
right: work that ran during capture instead of being recorded, a warmup whose side
effects stay in the caller's state, a replay fed an argument the graph never had a
buffer for.

The comparison is bitwise. Two runs of one kernel sequence over one shape agree to
the bit, so a tolerance here would only hide a graph that replayed something else.

Ground truth is the eager step. It is taken before the capture, from a snapshot of
the same state, so the two runs start from identical carries.

The empty capture is the failure that is otherwise silent: nothing is recorded, the
replay is a no-op, the output buffer still holds what the warmup left in it, and a
comparison against the eager result passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from slinoss.config import SLinOSSConfig
from slinoss.graph import GraphedStep, capture, capture_decode
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

pytestmark = pytest.mark.cuda

GRAPH_CONFIG = SLinOSSConfig(
    d_model=32,
    d_state=48,
    d_head=16,
    n_groups=2,
    chunk_size=16,
    n_layers=2,
    ffn_ratio=2.0,
    vocab_size=17,
)
"""Two layers, so a graph that records only the first one shows."""

BATCH = 2
PROMPT = 5

DTYPE = torch.bfloat16
"""bfloat16, because that is the dtype the kernel path runs.

float32 falls back to the reference scan, so a capture in it would record torch ops
and say nothing about whether a CuTe launch reaches the graph.
"""


def _cuda() -> torch.device:
    """The first visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def _stack(device: torch.device) -> SLinOSSStack:
    """A seeded stack, cast to the state dtype."""
    torch.manual_seed(0)
    return SLinOSSStack(GRAPH_CONFIG, device=device).to(DTYPE)


def _ids(device: torch.device, length: int) -> Tensor:
    """``(BATCH, length)`` ids from a generator of their own."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    vocab = GRAPH_CONFIG.vocab_size
    assert vocab is not None
    return torch.randint(0, vocab, (BATCH, length), generator=generator).to(device)


def _buffers(state: StackState) -> list[tuple[str, Tensor]]:
    """Every carry in the state, labelled, in layer order."""
    return [
        (f"layer {index} {name}", getattr(layer, name))
        for index, layer in enumerate(state.layers)
        for name in ("conv", "ssm", "b_prev", "u_prev")
    ]


@dataclass(frozen=True)
class Decode:
    """One captured decode step and the eager step it is compared against."""

    stack: SLinOSSStack
    token: Tensor
    step: GraphedStep
    eager_logits: Tensor
    eager_state: StackState
    entry_state: StackState
    graph_state: StackState
    entry_pointers: tuple[int, ...]


@pytest.fixture(scope="module")
def decode() -> Decode:
    """Prefill, take the eager step, then capture from the same carries.

    Module scope because a capture traces every kernel in the step, which costs
    more than the assertions that read it. The tests below do not write into what
    they are given except through the replay under test.
    """
    device = _cuda()
    stack = _stack(device)
    state = StackState.allocate(GRAPH_CONFIG, BATCH, device=device, dtype=DTYPE)
    stack(_ids(device, PROMPT), state)

    eager_state = state.clone()
    graph_state = state.clone()
    entry_state = state.clone()
    entry_pointers = tuple(b.data_ptr() for _, b in _buffers(graph_state))

    token = _ids(device, PROMPT + 1)[:, -1:]
    eager_logits = stack(token, eager_state).clone()
    step = capture_decode(stack, graph_state)
    return Decode(
        stack=stack,
        token=token,
        step=step,
        eager_logits=eager_logits,
        eager_state=eager_state,
        entry_state=entry_state,
        graph_state=graph_state,
        entry_pointers=entry_pointers,
    )


def test_the_capture_leaves_the_state_it_warmed_on_where_it_found_it(
    decode: Decode,
) -> None:
    """Warmup advances the state, so capture must put it back, in place.

    Both halves matter. Values, because a caller's carries are mid-sequence and
    warmup tokens are not part of that sequence. Addresses, because the graph
    recorded these buffers: restoring by rebinding a field would leave every replay
    writing memory nobody reads.
    """
    for (name, restored), (_, entry) in zip(
        _buffers(decode.graph_state), _buffers(decode.entry_state), strict=True
    ):
        assert torch.equal(restored, entry), f"warmup left its tokens in {name}"
    pointers = tuple(b.data_ptr() for _, b in _buffers(decode.graph_state))
    assert pointers == decode.entry_pointers, "the restore rebound a buffer"


def test_a_replayed_step_is_the_eager_step(decode: Decode) -> None:
    """The replay issues the step's kernels over the step's carries.

    Bitwise on both the logits and the four carries per layer. A graph missing a
    launch, or holding one whose operands moved, fails one of them.
    """
    logits = decode.step(decode.token)
    assert logits.shape == decode.eager_logits.shape
    assert torch.equal(logits, decode.eager_logits), "the replay is not the step"
    for (name, replayed), (_, eager) in zip(
        _buffers(decode.graph_state), _buffers(decode.eager_state), strict=True
    ):
        assert torch.equal(replayed, eager), f"the replay left a different {name}"


def test_a_replay_names_an_argument_the_graph_has_no_buffer_for(
    decode: Decode,
) -> None:
    """Shape and dtype are checked rather than passed to ``copy_``.

    ``copy_`` broadcasts and casts, so an argument of the wrong shape or dtype is
    silently accepted and the replay computes something the caller did not ask for.
    """
    with pytest.raises(ValueError, match="captured 1 inputs"):
        decode.step(decode.token, decode.token)
    with pytest.raises(ValueError, match="captured buffer is"):
        decode.step(decode.token[:1])
    with pytest.raises(ValueError, match="captured buffer is"):
        decode.step(decode.token.to(torch.int32))


def test_a_capture_that_records_nothing_is_an_error() -> None:
    """The one failure a comparison against the eager result cannot see."""
    device = _cuda()
    x = torch.ones(4, device=device)
    with pytest.raises(RuntimeError, match="recorded no work"):
        capture(lambda _: None, x)


def test_a_captured_train_step_reproduces_the_eager_gradients() -> None:
    """Forward and backward in one graph, over the parameters' own ``grad`` buffers.

    The backward runs on the autograd engine's thread, which is not the thread that
    opened the capture. Every launch it issues reaches the graph only if it goes to
    the stream that thread finds current, so this is the test that a launcher taking
    the current stream is enough to capture a backward.

    ``zero_grad(set_to_none=False)`` is inside the step: a replay cannot allocate,
    and a ``None`` grad would be allocated on the first backward after it.
    """
    device = _cuda()
    stack = _stack(device)
    ids = _ids(device, PROMPT)
    target = torch.zeros(BATCH, PROMPT, dtype=torch.int64, device=device)

    def step(x: Tensor, labels: Tensor) -> Tensor:
        stack.zero_grad(set_to_none=False)
        logits = stack(x)
        loss = torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten()
        )
        loss.backward()
        # Detached, as a captured step's output must be: a returned loss keeps its
        # autograd graph alive, and the AccumulateGrad nodes in it then belong to
        # whichever stream produced it rather than to the one running.
        return loss.detach()

    eager_loss = step(ids, target).clone()
    eager_grads = {
        name: p.grad.clone()
        for name, p in stack.named_parameters()
        if p.grad is not None
    }
    assert eager_grads, "the eager step produced no gradients"

    graphed = capture(step, ids, target)
    graphed(ids, target)

    assert torch.equal(graphed.outputs, eager_loss), "the replayed loss is not the loss"
    for name, eager in eager_grads.items():
        replayed = stack.get_parameter(name).grad
        assert replayed is not None, f"the replay left no gradient on {name}"
        assert torch.equal(replayed, eager), f"the replay left a different d{name}"
