"""The block and the stack around the mixer.

One file for both modules, because every property worth a test is a property of
the composition: the block alone is two fused norms and an FFN around a mixer that
has its own file, and what can go wrong is which stream is threaded where.

The parity half needs float64 and a device, and float64 has no kernel path, so it
judges the composition against the reference backend. Whether the kernel norm
matches that backend is a different question with its own file. The dtype halves do
need a kernel, since the point of them is what a module-wide cast and autocast do
to a norm weight and to the residual stream. The two guards are host arithmetic
over ranks.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, silu

from slinoss.blocks import SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.stack import SLinOSSStack
from tests.conftest import assert_max_rel

STACK_CONFIG = SLinOSSConfig(
    d_model=32,
    d_state=48,
    d_head=16,
    n_groups=2,
    chunk_size=16,
    n_layers=2,
    ffn_ratio=2.0,
    bias=True,
)
"""Two layers, because one cannot get the handover between blocks wrong.

Small everywhere else, and ``SEQLEN`` is not a multiple of ``chunk_size``, so the
scan runs a partial chunk. ``bias=True`` puts a bias on all five projections.
"""

VOCAB = 17
"""Embedding and head width. Prime, so no shape below it divides it."""

BATCH = 2
SEQLEN = 40

PARITY_TOL = 1e-15
"""Bound on the stack against the pre-norm residual composition, at float64.

Measured: 0.0 for both outputs and for all 58 gradients. float64 has no kernel
path, so the fused norm resolves to the reference backend, whose reduction is the
one :func:`_norm` states; both paths then issue the same operations on the same
operands and agree bitwise. The bound is a few float64 ulp, left as a tolerance
rather than an equality so that a reordered reduction reports a number.
"""


@pytest.fixture
def cuda() -> torch.device:
    """The first visible CUDA device, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def _activations(
    cfg: SLinOSSConfig, device: torch.device, dtype: torch.dtype, *, seed: int = 0
) -> Tensor:
    """One batch of activations, ``(BATCH, SEQLEN, d_model)``, in ``dtype``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        BATCH, SEQLEN, cfg.d_model, generator=gen, device=device, dtype=dtype
    )


def _tokens(vocab: int, device: torch.device, *, seed: int = 0) -> Tensor:
    """One batch of token ids, ``(BATCH, SEQLEN)`` int64."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, vocab, (BATCH, SEQLEN), generator=gen, device=device)


def _norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMS norm over the trailing axis, stated rather than dispatched."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def _reference(stack: SLinOSSStack, x: Tensor) -> Tensor:
    """The textbook pre-norm residual stack over the same parameters.

    Ground truth for the stack's stream handling. Every branch is added to the
    stream where it is produced, so nothing here mirrors the block's convention of
    handing its branch output back unadded; if that convention loses or duplicates
    an add, this disagrees.

    Args:
        stack: Supplies the parameters and the configuration.
        x: ``(B,T)`` ids or ``(B,T,d_model)`` activations, matching the stack.

    Returns:
        What :meth:`slinoss.SLinOSSStack.forward` returns.
    """
    eps = stack.config.norm_eps
    hidden = x if stack.embedding is None else stack.embedding(x)
    for module in stack.blocks:
        assert isinstance(module, SLinOSSBlock)
        hidden = hidden + module.mixer(_norm(hidden, module.mixer_norm_weight, eps))
        pre = _norm(hidden, module.ffn_norm_weight, eps)
        gated = silu(module.ffn_gate(pre)) * module.ffn_up(pre)
        hidden = hidden + module.ffn_out(gated)
    normed = _norm(hidden, stack.norm_weight, eps)
    return normed if stack.head is None else stack.head(normed)


@pytest.mark.cuda
@pytest.mark.parametrize("vocab", [None, VOCAB], ids=["activations", "tokens"])
def test_the_stack_is_the_prenorm_residual_stack(
    cuda: torch.device, vocab: int | None
) -> None:
    """Forward and every gradient against the composition, at float64.

    The block returns its branch output unadded and the stack's final norm is the
    add the last block did not do. Both are invisible in a shape or a dtype: a
    dropped add, a stream threaded past a branch, or a final norm over the wrong
    summand all give the right shape and finite numbers.
    """
    cfg = replace(STACK_CONFIG, vocab_size=vocab)
    stack = SLinOSSStack(cfg, device=cuda).to(torch.float64)
    names = [name for name, _ in stack.named_parameters()]
    params = list(stack.parameters())

    if vocab is None:
        fused_x = _activations(cfg, cuda, torch.float64).requires_grad_(True)
        ref_x = fused_x.detach().clone().requires_grad_(True)
        fused_leaves, ref_leaves = [fused_x, *params], [ref_x, *params]
        labels = ["x", *names]
    else:
        fused_x = ref_x = _tokens(vocab, cuda)
        fused_leaves, ref_leaves = params, params
        labels = names

    fused = stack(fused_x)
    ref = _reference(stack, ref_x)
    tag = "activations" if vocab is None else "tokens"
    assert_max_rel(fused, ref, PARITY_TOL, f"stack forward {tag}")

    gen = torch.Generator(device=cuda).manual_seed(1)
    dout = torch.randn(fused.shape, generator=gen, device=cuda, dtype=fused.dtype)
    fused_grads = torch.autograd.grad(fused, fused_leaves, dout)
    ref_grads = torch.autograd.grad(ref, ref_leaves, dout)
    for label, got, want in zip(labels, fused_grads, ref_grads, strict=True):
        assert_max_rel(got, want, PARITY_TOL, f"stack grad {tag} {label}")


@pytest.mark.cuda
@pytest.mark.cute
def test_a_block_hands_back_a_wide_stream_and_a_narrow_branch(
    cuda: torch.device,
) -> None:
    """The stream is float32 at a bf16 module, and stays float32 across a handover.

    A stack narrows its residual once per block if the fused norm's wide output is
    demoted on the way out, and nothing in a shape or a loss curve names that as
    the cause.
    """
    cfg = STACK_CONFIG
    block = SLinOSSBlock(cfg, device=cuda, dtype=torch.bfloat16)
    first = block(_activations(cfg, cuda, torch.bfloat16))
    assert first.hidden.dtype is torch.bfloat16
    assert first.residual.dtype is torch.float32
    second = block(first.hidden, first.residual)
    assert second.hidden.dtype is torch.bfloat16
    assert second.residual.dtype is torch.float32


@pytest.mark.cuda
@pytest.mark.cute
def test_the_stack_trains_under_bf16_autocast(cuda: torch.device) -> None:
    """Every parameter of a float32 stack takes a finite gradient under autocast.

    The dtype the kernels see under autocast is neither the parameter dtype nor the
    input dtype, and three parameters here are pinned float32 against an autocast
    that casts on sight. A demotion raises from a guard, and a parameter left out
    of the graph comes back with no gradient at all.
    """
    cfg = replace(STACK_CONFIG, vocab_size=VOCAB)
    stack = SLinOSSStack(cfg, device=cuda)
    ids = _tokens(VOCAB, cuda)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = stack(ids)
    assert logits.dtype is torch.bfloat16
    cross_entropy(logits.float().flatten(0, 1), ids.flatten()).backward()
    for name, param in stack.named_parameters():
        assert param.grad is not None, name
        assert bool(param.grad.isfinite().all()), name


def test_norm_weights_stay_float32_through_a_module_cast() -> None:
    """A module-wide demotion must not take a norm weight with it.

    ``stack.to(torch.bfloat16)`` is how the module reaches a kernel dtype, and the
    fused norm refuses a low-precision weight (I4). Demoted, the cast succeeds and
    the next forward raises. One cast covers both overrides: the stack's runs on
    its own weight and each block's runs on its two.
    """
    stack = SLinOSSStack(replace(STACK_CONFIG, vocab_size=VOCAB)).to(torch.bfloat16)
    block = stack.blocks[0]
    assert isinstance(block, SLinOSSBlock)
    assert stack.norm_weight.dtype is torch.float32
    assert block.mixer_norm_weight.dtype is torch.float32
    assert block.ffn_norm_weight.dtype is torch.float32
    assert block.ffn_gate.weight.dtype is torch.bfloat16
    # A widening cast is left alone, so a float64 oracle stays float64 end to end.
    wide = SLinOSSStack(STACK_CONFIG).to(torch.float64)
    assert wide.norm_weight.dtype is torch.float64


def test_the_input_form_follows_vocab_size() -> None:
    """One flag decides both ends, so a mismatched input is named where it enters.

    Without this check an activation tensor reaches the embedding, which reports an
    index error, and an id tensor reaches the fused norm, which reports a dtype it
    was never handed.
    """
    tokens = SLinOSSStack(replace(STACK_CONFIG, vocab_size=VOCAB))
    with pytest.raises(ValueError, match="token ids"):
        tokens(torch.zeros(BATCH, SEQLEN, STACK_CONFIG.d_model))
    plain = SLinOSSStack(STACK_CONFIG)
    with pytest.raises(ValueError, match=r"expected \(B,T,32\)"):
        plain(torch.zeros(BATCH, SEQLEN, dtype=torch.long))
