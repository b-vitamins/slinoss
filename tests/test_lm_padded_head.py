"""The padded head, from the evaluator's side: a pad column is unreachable, not merely unused.

The head is wider than the vocabulary so the GEMM aligns. That is the tree's decision and the
tree tests the gradients through it; what this file tests is the one property the language-model
harness leans on, which is that nothing downstream needs to slice.

``lm_eval`` reads the whole logit row: it takes a log-softmax over it and gathers the
continuation's tokens, and for the multiple-choice tasks it ranks. If a pad column carried a
finite score, the normalizer would include it and every likelihood would be shifted by an amount
that changed with the padding width -- which is a function of the vocabulary and the alignment
multiple, so two arms at one tokenizer would be shifted identically and the bug would be
invisible in a comparison and wrong in an absolute number. If it carried a large one, a greedy
argmax would return a token that does not exist.

So: the pad band is the dtype's minimum, the softmax mass on it is zero, and a ranking over the
padded width is the ranking over the slice.
"""

from __future__ import annotations

import torch

from scripts.lm.mixers import REGISTRY
from scripts.lm.model import MixerLM, build_model, layer_factories, scaffold_config

D_MODEL = 64
N_LAYERS = 2
VOCAB = 100
MAX_LENGTH = 16


def _stack() -> MixerLM:
    """A two-layer stack with a vocabulary that does not divide the alignment multiple."""
    torch.manual_seed(0)
    config = scaffold_config(
        d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB, d_state=48, d_head=16
    )
    resolved = REGISTRY.resolve("conv")
    return build_model(
        config, layer_factories(resolved.factory, N_LAYERS), max_length=MAX_LENGTH
    )


def _logits(model: MixerLM) -> torch.Tensor:
    """Logits over a fixed batch."""
    ids = torch.arange(8, dtype=torch.int64).remainder(VOCAB).reshape(2, 4)
    with torch.no_grad():
        return model(ids)


def test_the_head_is_wider_than_the_vocabulary() -> None:
    """The premise. A vocabulary already on the multiple would make every assertion vacuous."""
    model = _stack()
    head = model.head
    assert head is not None
    assert head.out_features > VOCAB
    assert _logits(model).shape[-1] == head.out_features


def test_the_pad_band_is_the_dtype_minimum() -> None:
    """Not zero and not merely small: zero would outrank a confidently negative real token."""
    model = _stack()
    logits = _logits(model)
    pad = logits[..., VOCAB:]
    assert pad.numel() > 0
    assert torch.all(pad == torch.finfo(logits.dtype).min)
    assert torch.all(torch.isfinite(logits[..., :VOCAB]))


def test_the_softmax_puts_no_mass_on_a_pad_column() -> None:
    """The normalizer an evaluator divides by is the vocabulary's, not the padded width's.

    A pad column with any mass shifts every log-likelihood by ``log(1 + pad_mass)``, so an
    absolute bits-per-byte or accuracy figure would be wrong while a comparison between two
    arms at one width stayed unchanged.
    """
    model = _stack()
    logits = _logits(model)
    probs = logits.softmax(dim=-1)
    assert torch.all(probs[..., VOCAB:] == 0.0)
    assert torch.allclose(probs[..., :VOCAB].sum(dim=-1), torch.ones(2, 4), atol=1e-6)
    sliced = logits[..., :VOCAB].log_softmax(dim=-1)
    assert torch.allclose(logits.log_softmax(dim=-1)[..., :VOCAB], sliced, atol=1e-6)


def test_a_ranking_over_the_padded_width_is_the_ranking_over_the_slice() -> None:
    """What the multiple-choice tasks do, and what a greedy decode does.

    ``argmax`` over the padded row must never land in the pad band, and the order of the real
    tokens must be the order the slice gives, so the harness can hand the full row to
    ``lm_eval`` unsliced.
    """
    model = _stack()
    logits = _logits(model)
    assert torch.all(logits.argmax(dim=-1) < VOCAB)
    padded_order = logits.argsort(dim=-1, descending=True, stable=True)[..., :VOCAB]
    sliced_order = logits[..., :VOCAB].argsort(dim=-1, descending=True, stable=True)
    assert torch.equal(padded_order, sliced_order)


def test_the_band_is_the_minimum_of_the_dtype_the_gemm_produced() -> None:
    """Under autocast the logits are not float32, and a float32 constant would not fit.

    A band written at the wrong dtype saturates to negative infinity, which is not wrong under
    a softmax but makes a log-softmax over an all-pad row a NaN rather than a large negative.
    Training and evaluation both run under autocast, so this is the path that is measured.
    """
    model = _stack()
    ids = torch.arange(8, dtype=torch.int64).remainder(VOCAB).reshape(2, 4)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(ids)
    assert torch.all(logits[..., VOCAB:] == torch.finfo(logits.dtype).min)
    assert torch.all(torch.isfinite(logits))
    assert torch.all(logits.softmax(dim=-1)[..., VOCAB:] == 0.0)
