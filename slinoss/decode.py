"""Autoregressive decoding over :class:`slinoss.SLinOSSStack`.

    logits = stack(prompt, state)          # prefill, one call over T tokens
    repeat: token = sample(logits[:, -1]); logits = stack(token, state)

One state, advanced in place. Per step the loop allocates the logits, the sampled
token, and whatever the sampler needs; the carries are the buffers the state was
built with, which is what lets a captured graph replay the step.

Sampling is on the device throughout. ``stop_token_id`` is the one option that
costs a host synchronization, once per step, to decide whether to stop early; the
sequences that finished earlier are held at the stop token rather than dropped, so
the returned block stays rectangular.

The prompt is one length for the whole batch. Ragged prompts belong to a scheduler
that owns padding and a position mask, and neither is this module's.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn.functional import softmax

from slinoss.stack import SLinOSSStack
from slinoss.state import StackState

__all__ = ["DecodeOutput", "generate"]


class DecodeOutput(NamedTuple):
    """What one :func:`generate` call produces.

    Attributes:
        tokens: ``(B, max_new_tokens)`` int64 sampled ids. Positions past a
            sequence's stop token hold that token.
        state: The state the loop advanced. It has consumed the prompt and every
            generated token but the last, which was sampled and never fed back, so
            a continuation passes ``tokens[:, -1:]`` as its prompt. Feeding the
            whole block again would apply those tokens twice.
    """

    tokens: Tensor
    state: StackState


def _sample(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> Tensor:
    """One token per row of ``logits``.

    Args:
        logits: ``(B,V)``, any float dtype.
        temperature: Zero for argmax. Positive scales the logits before the
            softmax.
        top_k: Keep the ``top_k`` highest logits per row, or None for all of them.
        generator: Source for :func:`torch.multinomial`, or None for the global one.

    Returns:
        ``(B,)`` int64 ids.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    # float32 for the softmax whatever the head produced: a bf16 exponential over a
    # vocabulary-wide sum is a sampler with a floor under its own resolution.
    scaled = logits.float() / temperature
    if top_k is not None:
        kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    return torch.multinomial(softmax(scaled, dim=-1), 1, generator=generator)[:, 0]


def generate(
    stack: SLinOSSStack,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    state: StackState | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    stop_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> DecodeOutput:
    """Decode ``max_new_tokens`` tokens after ``prompt``.

    Args:
        stack: A stack with an embedding and a head, so ``vocab_size`` is set.
        prompt: ``(B,T)`` int64 ids, one length for the batch. At least one token,
            since the first sampled token reads the prefill's last logits.
        max_new_tokens: Tokens to generate. Positive.
        state: A state to continue, advanced in place, or None to allocate a zeroed
            one at ``prompt``'s batch on the stack's device and activation dtype. A
            continued state expects the last token of the previous call as the
            prompt; see :class:`DecodeOutput`.
        temperature: Zero for greedy decoding, otherwise a positive scale.
        top_k: Truncate the distribution to its ``top_k`` highest logits. Ignored
            at ``temperature = 0``, where argmax is already the top one.
        stop_token_id: Stop once every sequence has produced this id. Costs one
            host synchronization per step.
        generator: Source of randomness for the sampler.

    Returns:
        A :class:`DecodeOutput`.

    Raises:
        ValueError: On a stack with no head, a prompt that is not ``(B,T)`` with
            ``T >= 1``, a non-positive ``max_new_tokens``, a negative
            ``temperature``, a non-positive ``top_k``, or a state whose batch is
            not the prompt's.
    """
    if stack.head is None or stack.embedding is None:
        raise ValueError("generate needs a stack with a head; set config.vocab_size")
    if prompt.ndim != 2 or prompt.shape[1] < 1:
        raise ValueError(f"prompt must be (B,T) with T >= 1, got {tuple(prompt.shape)}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must not be negative, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")

    batch = int(prompt.shape[0])
    if state is None:
        weight = stack.embedding.weight
        state = StackState.allocate(
            stack.config, batch, device=weight.device, dtype=weight.dtype
        )
    elif state.batch != batch:
        raise ValueError(f"prompt holds batch {batch} and state holds {state.batch}")

    # Filled rather than empty: an early stop leaves the tail of every row at the
    # stop token, so the block is rectangular without a second pass over it.
    tokens = prompt.new_full(
        (batch, max_new_tokens), 0 if stop_token_id is None else stop_token_id
    )
    finished = torch.zeros(batch, dtype=torch.bool, device=prompt.device)
    # Once, outside the loop: a scalar built from a Python int is a host-to-device
    # copy, which is the one thing per step that has no reason to be there.
    stop = None if stop_token_id is None else prompt.new_full((), stop_token_id)

    logits = stack(prompt, state)
    for index in range(max_new_tokens):
        token = _sample(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        if stop is not None:
            token = torch.where(finished, stop, token)
            finished |= token == stop
        tokens[:, index] = token
        if index + 1 == max_new_tokens:
            break
        if stop is not None and bool(finished.all()):
            break
        logits = stack(token[:, None], state)

    return DecodeOutput(tokens=tokens, state=state)
