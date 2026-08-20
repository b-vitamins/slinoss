"""The block and the stack around the mixer.

One file for both modules, because every property worth a test is a property of
the composition: the block alone is two fused norms and an FFN around a mixer that
has its own file, and what can go wrong is which stream is threaded where.

The parity half needs float64 and a device, and float64 has no kernel path, so it
judges the composition against the reference backend. Whether the kernel norm
matches that backend is a different question with its own file. The dtype halves do
need a kernel, since the point of them is what a module-wide cast and autocast do
to a norm weight and to the residual stream. The guards are host arithmetic over
ranks, shapes and dtypes.

The decode half is judged against this file's own whole-sequence call rather than
against a second reference: what a split can get wrong is which of the four carries
it threads, and the whole-sequence call is already ground truth for the composition
by the time it is used that way.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, linear, silu

from slinoss.blocks import SLinOSSBlock
from slinoss.config import SLinOSSConfig
from slinoss.stack import SLinOSSStack
from slinoss.state import MixerState, StackState
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

DECODE_TOL = 1e-14
"""Bound on a split decode against one whole-sequence call, at float64.

Measured: 3.3e-15 stepwise, 2.7e-15 chunked, 2.0e-15 prefill. Not bitwise, unlike
the parity bound: a split restarts the scan's chunk prefixes at every boundary, so
the two runs reduce the same recurrence in a different order. Stepwise is the worst
of the three because it restarts the most often.
"""

PARITY_TOL = 1e-15
"""Bound on the stack against the pre-norm residual composition, at float64.

Measured: 0.0 for both outputs and for every gradient of both leaf sets, 38 from
the token entry point and 36 from the activation one. float64 has no kernel
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
        # Through the framework's own linear over the transposed weight. The block
        # stores that weight (d_ffn, d_model) and contracts it directly, so this
        # states the map the nn.Linear it replaced stated, and the gradient reaching
        # the stored parameter through this view is the parity between the two call
        # forms as well as the parity of the composition.
        out = linear(gated, module.ffn_out_weight.t(), module.ffn_out_bias)
        hidden = hidden + out
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


def test_the_ffn_output_weight_is_stored_transposed() -> None:
    """The stored orientation of the output projection, and the fan_in with it.

    ``(d_ffn, d_model)`` is a throughput contract, not a convention: the weight
    gradient comes out in the stored shape, and in the :class:`torch.nn.Linear`
    orientation that shape covers 0.64 of a wave on the device the block is written
    for. A return to the framework orientation changes no value and no dtype.

    Neither does initializing over the stored shape instead of over its transpose,
    which is why the draw is checked against the framework's own and not only
    against the extents. The default is uniform on ``+-1/sqrt(fan_in)`` with
    ``fan_in`` the contraction extent, so reading ``fan_in`` off the wrong axis
    rescales every weight in the layer by ``sqrt(d_ffn/d_model)`` and every bias
    with it. The variance distinguishes that from the right draw; the bound alone
    does not, since a wider uniform still fits under a maximum over finitely many
    samples. The comparison is against a live :class:`torch.nn.Linear` of the map
    this replaces, so it holds whatever the framework's default becomes.
    """
    cfg = STACK_CONFIG
    # Seeded: the asserts below are sample statistics, and the tolerances are chosen
    # against the sample count rather than against a run of luck.
    torch.manual_seed(0)
    block = SLinOSSBlock(cfg)
    framework = nn.Linear(cfg.d_ffn, cfg.d_model, bias=cfg.bias)
    assert block.ffn_out_bias is not None
    assert framework.bias is not None
    # Detached because every statistic below reads a leaf as a scalar.
    weight = block.ffn_out_weight.detach()
    bias = block.ffn_out_bias.detach()
    ref_weight = framework.weight.detach()
    ref_bias = framework.bias.detach()
    assert weight.shape == (cfg.d_ffn, cfg.d_model)
    assert weight.shape == ref_weight.t().shape
    assert bias.shape == ref_bias.shape
    # A uniform's variance is a third of the square of its bound, so matching the
    # framework's variance over 2,048 samples pins the bound, and with it the axis
    # fan_in was read from. The relative standard error of the estimate is 2%; the
    # error it has to catch is a factor of d_ffn/d_model, which is 4.
    assert float(weight.var()) == pytest.approx(float(ref_weight.var()), rel=0.1)
    assert float(bias.var()) == pytest.approx(float(ref_bias.var()), rel=0.2)
    bound = cfg.d_ffn**-0.5
    assert float(weight.abs().max()) <= bound
    assert float(bias.abs().max()) <= bound


DECODE_SPLITS = [
    pytest.param("stepwise", (1,) * SEQLEN, id="stepwise"),
    pytest.param("chunked", (17, 23), id="chunked"),
    pytest.param("prefill", (17, *(1,) * (SEQLEN - 17)), id="prefill"),
]
"""How the sequence is partitioned across calls. Every length sums to ``SEQLEN``.

Three calling shapes, not three input values: single-token calls throughout, two
multi-token calls, and the decode pattern of one multi-token call then single ones.
``17`` against ``chunk_size = 16`` puts every boundary inside a chunk, so a
continuation that only holds on chunk boundaries fails here.
"""


@pytest.mark.cuda
@pytest.mark.parametrize(("label", "splits"), DECODE_SPLITS)
def test_a_split_decode_reproduces_the_whole_sequence(
    cuda: torch.device, label: str, splits: tuple[int, ...]
) -> None:
    """A partitioned run against one whole-sequence call, at float64.

    Four carries, and a decode that drops any of them still returns finite logits of
    the right shape: without ``ssm`` the recurrence restarts at every boundary,
    without ``b_prev`` and ``u_prev`` each continuation loses one token of the
    two-tap forcing, and without ``conv`` it loses the tap window.
    """
    cfg = replace(STACK_CONFIG, vocab_size=VOCAB)
    # Seeded: the parameters come from the global generator, and the bound below is
    # the error of one draw. Unseeded, the splits would not be comparable either.
    torch.manual_seed(0)
    stack = SLinOSSStack(cfg, device=cuda).to(torch.float64)
    ids = _tokens(VOCAB, cuda)
    whole = stack(ids)

    state = StackState.allocate(cfg, BATCH, device=cuda, dtype=torch.float64)
    parts: list[Tensor] = []
    offset = 0
    for length in splits:
        parts.append(stack(ids[:, offset : offset + length], state))
        offset += length
    assert offset == SEQLEN
    assert_max_rel(torch.cat(parts, dim=1), whole, DECODE_TOL, f"decode {label}")


@pytest.mark.cuda
@pytest.mark.cute
def test_a_step_advances_the_state_in_place_and_records_no_graph(
    cuda: torch.device,
) -> None:
    """One bf16 step, at the kernel backends, through the buffers it was handed.

    Two properties and one dtype, all of which a parity test at float64 misses. A
    rebound buffer leaves a captured graph writing memory no consumer reads, and
    that replays as a state frozen at its first token. A recorded graph is a leak
    per step whose gradient reaches no mixer parameter, since the mixer's step
    records nothing either way. And ``T = 1`` is the shape the chunked kernels are
    least likely to admit: one partial chunk, no full one.
    """
    cfg = replace(STACK_CONFIG, vocab_size=VOCAB)
    stack = SLinOSSStack(cfg, device=cuda).to(torch.bfloat16)
    state = StackState.allocate(cfg, BATCH, device=cuda, dtype=torch.bfloat16)
    layer = state.layers[0]
    buffers = {
        "conv": layer.conv,
        "ssm": layer.ssm,
        "b_prev": layer.b_prev,
        "u_prev": layer.u_prev,
    }
    before = {name: buf.data_ptr() for name, buf in buffers.items()}

    logits = stack(_tokens(VOCAB, cuda)[:, :1], state)

    assert logits.shape == (BATCH, 1, VOCAB)
    assert logits.dtype is torch.bfloat16
    assert not logits.requires_grad
    for name, buf in buffers.items():
        assert buf.data_ptr() == before[name], name
        assert bool(buf.isfinite().all()), name
    assert bool(layer.ssm.any()), "the recurrent state was not written"


def test_a_state_the_stack_cannot_use_is_named_where_it_enters() -> None:
    """Depth, batch, and dtype, each reported before an operator sees it.

    Unnamed, a short state runs the layers it has against the parameters of the
    layers it does not, a mismatched batch surfaces as the convolution's window
    shape, and a mismatched dtype surfaces two operators later as whichever operand
    the kernel checked first.
    """
    cfg = replace(STACK_CONFIG, vocab_size=VOCAB)
    stack = SLinOSSStack(cfg)
    ids = torch.zeros(BATCH, 1, dtype=torch.long)
    one = MixerState.allocate(cfg, BATCH, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="state has depth 1"):
        stack(ids, StackState(layers=(one,)))
    off_batch = StackState.allocate(cfg, BATCH + 1, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match=f"state holds {BATCH + 1}"):
        stack(ids, off_batch)
    wide = StackState.allocate(cfg, BATCH, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="cast the module"):
        stack(ids, wide)


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
