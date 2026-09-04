"""The generate loop over the stack.

The loop itself is host arithmetic over a state the stack advances, so what can go
wrong here is not numerical: which logits row the sampler reads, whether the state
a continuation resumes from is one token behind or ahead, whether a finished
sequence's freeze reaches a row that did not finish. Each of those returns ids of
the right shape from a plausible-looking distribution.

Ground truth for the loop is the stack's own whole-sequence call, teacher-forced on
what the loop produced. That is a stronger statement than a self-comparison: it
says every generated token is the argmax the model assigns at that position, which
fails if the prefill hands over at the wrong offset or the state stops advancing.

Greedy throughout, so the comparison is over ids rather than tolerances. float64,
because a split run and a whole-sequence run agree to a few ulp rather than
bitwise, and the tests assert that the argmax survives that; the top-two gap is
checked so a near-tie reports as a tie rather than as a decode bug.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from slinoss._precision import SUPPORTED_DTYPES
from slinoss.blocks import SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.decode import _sample, generate
from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

DECODE_CONFIG = SLinOSSConfig(
    d_model=32,
    d_state=48,
    d_head=16,
    n_groups=2,
    chunk_size=16,
    n_layers=2,
    ffn_ratio=2.0,
    vocab_size=17,
)
"""Two layers, so a state that fails to reach the second one shows.

``vocab_size`` is prime and small enough that a teacher-forced argmax is cheap to
compare. ``PROMPT`` and ``PROMPT + NEW`` both sit inside one ``chunk_size``, so
every call runs a partial chunk and none of the agreement below comes from landing
on a chunk boundary.
"""

BATCH = 2
PROMPT = 5
NEW = 6

TIE_MARGIN = 1e-9
"""Least top-two logit gap a greedy comparison is read as decisive.

A split run and a whole-sequence run disagree by order 1e-15 (see
``tests/test_stack.py``), so a gap above this bound cannot be closed by the
disagreement and the argmax is the same on both sides. Below it the comparison
reports a tie rather than a decode defect. Measured smallest gap over the positions
compared: 3.643e-3, six orders above the bound.
"""


@pytest.fixture
def cuda() -> torch.device:
    """The first visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def _stack(device: torch.device) -> SLinOSSStack:
    """A seeded float64 stack. Seeded because the ids compared below are one draw."""
    torch.manual_seed(0)
    stack = SLinOSSStack(DECODE_CONFIG, device=device).to(torch.float64)
    # The mixer's residual branch intentionally starts as an exact no-op. Activate
    # only its final projection: unactivated, every logit is a function of its own
    # token alone, so a state that stops advancing still reproduces the argmax.
    for module in stack.blocks:
        cast("SLinOSSBlock", module).mixer.out_proj.reset_parameters()
    return stack


def _prompt(device: torch.device) -> torch.Tensor:
    """``(BATCH, PROMPT)`` ids, from a generator of their own."""
    gen = torch.Generator(device=device).manual_seed(1)
    return torch.randint(0, 17, (BATCH, PROMPT), generator=gen, device=device)


@pytest.mark.parametrize(
    "dtype", SUPPORTED_DTYPES, ids=lambda d: str(d).removeprefix("torch.")
)
@pytest.mark.parametrize(
    ("temperature", "top_k"),
    [(0.0, None), (1.0, None), (0.7, 4), (1.0, 4 * 17)],
    ids=["argmax", "untruncated", "top_k", "top_k_past_the_vocabulary"],
)
def test_a_padding_column_is_never_argmaxed_or_sampled(
    dtype: torch.dtype, temperature: float, top_k: int | None
) -> None:
    """Both sampler branches against a head wider than its vocabulary.

    The head is ``padded_vocab_size`` wide, so every id past ``vocab_size`` is a
    column the tokenizer cannot name and the model never trained. Returning one is
    silent: it is an integer in the tensor's range, it indexes the embedding, and
    the next step continues from it. What keeps it out is only the fill
    :meth:`slinoss.SLinOSSStack.forward` writes, so this drives the sampler
    directly rather than through a stack, which is the only way to reach every
    dtype the stack admits without a kernel for each.

    Adversarial on purpose: every real logit is below zero, so a padding column
    left at zero, left at a framework init, or filled with anything finite and
    positive wins the argmax outright and carries sampling mass. ``top_k`` above
    the padded width is the case where padding columns do enter the truncation.
    """
    vocab = DECODE_CONFIG.vocab_size
    padded = DECODE_CONFIG.padded_vocab_size
    assert vocab is not None and padded is not None
    assert padded > vocab, "this config no longer pads, so the test proves nothing"
    rows = 256
    gen = torch.Generator().manual_seed(3)
    logits = torch.full((rows, padded), torch.finfo(dtype).min, dtype=dtype)
    logits[:, :vocab] = (-1.0 - torch.rand(rows, vocab, generator=gen)).to(dtype)
    for seed in range(4):
        ids = _sample(
            logits,
            temperature=temperature,
            top_k=top_k,
            generator=torch.Generator().manual_seed(seed),
        )
        assert ids.shape == (rows,)
        assert bool((ids < vocab).all()), f"a padding id was returned, seed {seed}"


@pytest.mark.cuda
def test_greedy_decode_is_the_teacher_forced_argmax(cuda: torch.device) -> None:
    """Every generated id against the whole-sequence logits at its own position.

    The loop reads ``logits[:, -1]``, which is the last position of the prefill on
    the first step and the only position after it. An off-by-one there, a state that
    stops advancing, or a prefill whose last token is dropped all keep generating
    ids that look like the model's; none of them stay the argmax of the sequence
    they claim to have produced.
    """
    stack = _stack(cuda)
    prompt = _prompt(cuda)
    out = generate(stack, prompt, max_new_tokens=NEW)

    assert out.tokens.shape == (BATCH, NEW)
    assert out.tokens.dtype is prompt.dtype

    whole = torch.cat((prompt, out.tokens), dim=1)
    # No grad: the loop records nothing, so a teacher-forced graph here would be the
    # only one in the test, and the margin below reads a scalar off it.
    with torch.no_grad():
        logits = stack(whole[:, :-1])[:, PROMPT - 1 :]
    top2 = logits.topk(2, dim=-1).values
    margin = float((top2[..., 0] - top2[..., 1]).min())
    print(f"decode top-two margin {margin:.3e}")
    assert margin > TIE_MARGIN
    assert torch.equal(logits.argmax(dim=-1), out.tokens)


@pytest.mark.cuda
def test_a_continued_state_resumes_where_the_block_ended(cuda: torch.device) -> None:
    """Two calls of ``NEW`` against one of ``2 * NEW``.

    The returned state has consumed every generated token but the last, so a
    continuation prompts with that one. Off by a token either way the streams
    diverge from the first resumed id: fed nothing, the model predicts from a state
    one token stale, and fed the whole block, it sees ``NEW`` tokens twice.
    """
    stack = _stack(cuda)
    prompt = _prompt(cuda)
    once = generate(stack, prompt, max_new_tokens=2 * NEW)

    first = generate(stack, prompt, max_new_tokens=NEW)
    assert torch.equal(first.tokens, once.tokens[:, :NEW])
    second = generate(
        stack, first.tokens[:, -1:], max_new_tokens=NEW, state=first.state
    )
    assert torch.equal(second.tokens, once.tokens[:, NEW:])
    assert second.state is first.state


@pytest.mark.cuda
def test_top_k_of_one_is_greedy(cuda: torch.device) -> None:
    """A truncated distribution of width one leaves the sampler no choice.

    The only test of the sampler, and a sampler drawing from the wrong support is
    invisible downstream: an inverted mask keeps everything but the top logit, a
    ``topk`` over the wrong axis keeps the wrong support entirely, and both return
    ids in range from a distribution nothing here would otherwise inspect. At
    ``top_k = 1`` the softmax is a point mass, so the draw is the argmax whatever
    the generator holds.
    """
    stack = _stack(cuda)
    prompt = _prompt(cuda)
    greedy = generate(stack, prompt, max_new_tokens=NEW)
    gen = torch.Generator(device=cuda).manual_seed(2)
    sampled = generate(
        stack, prompt, max_new_tokens=NEW, temperature=0.7, top_k=1, generator=gen
    )
    assert torch.equal(sampled.tokens, greedy.tokens)


@pytest.mark.cuda
def test_a_stop_freezes_one_row_and_leaves_the_others_running(
    cuda: torch.device,
) -> None:
    """One row's stop token, and what the rest of the batch does about it.

    The stop mask is the one place the loop couples the rows, and coupling them
    wrongly is silent: a mask that broadcasts over the batch ends every sequence at
    the first one to finish, and a tail left at whatever ``torch.empty`` held
    returns ids no vocabulary check would reject.
    """
    stack = _stack(cuda)
    prompt = _prompt(cuda)
    base = generate(stack, prompt, max_new_tokens=NEW).tokens
    stop = int(base[0, 0])
    assert int(base[1, 0]) != stop, "the seed no longer gives the rows distinct starts"

    out = generate(stack, prompt, max_new_tokens=NEW, stop_token_id=stop)
    # Row 0 stops on its first token, so every later position is the fill.
    assert torch.equal(out.tokens[0], out.tokens.new_full((NEW,), stop))
    # Row 1 is unaffected by row 0's freeze up to its own stop, if it has one.
    hit = (base[1] == stop).nonzero()
    end = NEW if hit.numel() == 0 else int(hit[0]) + 1
    assert torch.equal(out.tokens[1, :end], base[1, :end])
    assert torch.equal(out.tokens[1, end:], out.tokens.new_full((NEW - end,), stop))


def test_a_request_generate_cannot_serve_is_named_before_the_prefill() -> None:
    """Six rejections, all host arithmetic, all before the stack runs.

    A missing head surfaces as a shape mismatch inside the final norm, and the four
    numeric arguments surface later or not at all: ``max_new_tokens = 0`` returns an
    empty block, a negative temperature inverts the distribution, and
    ``top_k = 0`` masks every logit and leaves ``multinomial`` with no support.
    """
    headless = SLinOSSStack(replace(DECODE_CONFIG, vocab_size=None))
    ids = torch.zeros(BATCH, PROMPT, dtype=torch.long)
    with pytest.raises(ValueError, match="needs a stack with a head"):
        generate(headless, ids, max_new_tokens=NEW)

    stack = SLinOSSStack(DECODE_CONFIG)
    with pytest.raises(ValueError, match=r"must be \(B,T\) with T >= 1"):
        generate(stack, ids[:, :0], max_new_tokens=NEW)
    with pytest.raises(ValueError, match=r"must be \(B,T\) with T >= 1"):
        generate(stack, ids[0], max_new_tokens=NEW)
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        generate(stack, ids, max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature must not be negative"):
        generate(stack, ids, max_new_tokens=NEW, temperature=-1.0)
    with pytest.raises(ValueError, match="top_k must be positive"):
        generate(stack, ids, max_new_tokens=NEW, top_k=0)

    off = StackState.allocate(
        DECODE_CONFIG, BATCH + 1, device="cpu", dtype=torch.float32
    )
    with pytest.raises(ValueError, match=f"state holds {BATCH + 1}"):
        generate(stack, ids, max_new_tokens=NEW, state=off)
