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

Equivalence is the weakest of the properties a replay owes. A replay that allocated
on every step, or synchronized on every step, would reproduce the eager step exactly
and would not be a decode path: both put back the per-token host cost capture exists
to remove. Both are asserted below, over several replays, because a cost paid once
per capture and a cost paid once per step are the same reading after one replay.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import NamedTuple, cast

import pytest
import torch
from torch import Tensor

from slinoss.blocks import SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.graph import GraphedStep, capture, capture_decode
from slinoss.stack import SLinOSSStack
from slinoss.state import MixerState, StackState

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
    """A seeded stack, cast to the state dtype, with every zeroed band drawn.

    ``SLinOSSMixer.reset_parameters`` zeroes ``out_proj``, the forcing-B band and the
    parameter band, and sets ``key_weight`` to the last-tap delta. Left that way four
    of the five carries a layer holds are identically zero after a prefill and the
    mixer output is exactly zero, so a bitwise comparison of replay against eager is
    a comparison of zeros: it holds for a reason that has nothing to do with the
    recorded addresses.
    """
    torch.manual_seed(0)
    stack = SLinOSSStack(GRAPH_CONFIG, device=device).to(DTYPE)
    for module in stack.blocks:
        mixer = cast("SLinOSSBlock", module).mixer
        mixer.in_proj.reset_parameters()
        mixer.out_proj.reset_parameters()
        if mixer.key_weight is not None:
            torch.nn.init.normal_(mixer.key_weight, std=0.5)
    return stack


def _ids(device: torch.device, length: int) -> Tensor:
    """``(BATCH, length)`` ids from a generator of their own."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    vocab = GRAPH_CONFIG.vocab_size
    assert vocab is not None
    return torch.randint(0, vocab, (BATCH, length), generator=generator).to(device)


CARRIES = tuple(carry.name for carry in fields(MixerState))
"""Every buffer name in a layer's state, read off the dataclass.

Not written out here. The restore under test enumerates the same set, and a literal
list in the test is a second enumeration free to drop the same field as the first.
That is how a restore that never copied ``keys`` passed this file: both lists named
the same four buffers, so the test agreed with the defect instead of the contract.
"""


def _buffers(state: StackState) -> list[tuple[str, Tensor]]:
    """Every carry in the state, labelled, in layer order."""
    return [
        (f"layer {index} {name}", getattr(layer, name))
        for index, layer in enumerate(state.layers)
        for name in CARRIES
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

    Bitwise on the logits and on every carry a layer holds. A graph missing a launch,
    or holding one whose operands moved, fails one of them. The key convolution is
    reached only through ``keys``, so it is unverified by any comparison that skips
    that buffer.
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


# ---------------------------------------------------------------------------
# What a replay must not do
# ---------------------------------------------------------------------------

REPLAYS = 4
"""Replays inside each region under test.

More than one, because a cost paid once per capture and a cost paid once per step are
the same reading after a single replay.
"""


@dataclass(frozen=True)
class Replay:
    """A captured step the two tests below are free to advance.

    Its own capture, not the ``decode`` fixture's. Those tests compare the state a
    replay advanced against the eager step, so repeated replays on the same state
    would make this file's result depend on the order it ran in.

    Attributes:
        step: The captured step.
        token: ``(BATCH, 1)`` ids to replay over.
    """

    step: GraphedStep
    token: Tensor


@pytest.fixture(scope="module")
def replay() -> Replay:
    """Prefill, capture, and replay once, leaving the allocator at its steady state."""
    device = _cuda()
    stack = _stack(device)
    state = StackState.allocate(GRAPH_CONFIG, BATCH, device=device, dtype=DTYPE)
    stack(_ids(device, PROMPT), state)
    step = capture_decode(stack, state)
    token = _ids(device, PROMPT + 1)[:, -1:]
    step(token)
    return Replay(step=step, token=token)


class Counters(NamedTuple):
    """Two allocator readings, taken together.

    Attributes:
        current: Bytes the caching allocator currently holds out to tensors.
        calls: Allocations it has served since the process started.
    """

    current: int
    calls: int


def _counters() -> Counters:
    """Both readings, so a failure names which one moved.

    ``current`` alone cannot see an allocation and a free in the same region, which is
    what a replay that allocates per step looks like once its temporary is dropped.
    ``calls`` is cumulative and sees it.
    """
    return Counters(
        current=int(torch.cuda.memory_allocated()),
        calls=int(torch.cuda.memory_stats()["allocation.all.allocated"]),
    )


def test_a_replay_allocates_nothing_in_the_steady_state(replay: Replay) -> None:
    """The claim is the steady state of replay, not the absence of any allocation.

    The captured function does allocate: the mixer's output is a fresh tensor per
    call. It is allocated once, during the recording, out of the graph's private pool,
    at an address every replay reuses -- which is why a replay must not reach the
    allocator again. So the reading is taken after a replay has already run, and what
    is asserted is that further replays move neither counter. An assertion that no
    allocation happens anywhere would be false and would be deleted by whoever read
    it next.
    """
    before = _counters()
    for _ in range(REPLAYS):
        replay.step(replay.token)
    assert _counters() == before, "a replay reached the allocator"


def test_a_replay_synchronizes_nothing(replay: Replay) -> None:
    """A step that waits on the device once per token is not a decode path.

    ``set_sync_debug_mode("error")`` raises from any call that waits, so the property
    is asserted by running the region rather than by measuring it. The mode is process
    wide and a failure inside the region would otherwise leave it set for every test
    after this one, hence the ``finally``.
    """
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(REPLAYS):
            replay.step(replay.token)
    finally:
        torch.cuda.set_sync_debug_mode("default")
