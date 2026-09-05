"""The bridge to ``lm_eval``: one override, the attributes it reads off ``self``, and the
number that comes out.

Three failure modes, all silent.

The wrapper does not call ``HFLM.__init__`` -- it cannot, since there is no Hugging Face model
to construct -- so every attribute the tokenization and batching paths read has to be set here.
A missing one surfaces as an ``AttributeError`` after the first task has already been scored,
and a wrong one is worse: ``add_bos_token`` or ``logits_cache`` set the other way changes every
likelihood by a fixed amount, which shifts an absolute number and leaves a comparison between
two arms at that setting looking clean. Inherited methods that read those same unset attributes
belong to this mode too, and the one the driver calls unconditionally is checked here.

The third is the scoring chain between the two. ``HFLM`` sorts, batches, pads, shifts and
gathers around the one call this class overrides, and every misalignment in that path returns
a plausible log-probability, so one pair is scored end to end against the stack's own numbers.

The other is the row itself. ``lm_eval`` takes a log-softmax over the whole logit row and
gathers the continuation's tokens, so the padded columns are part of the normalizer unless they
are unreachable. This file checks the row the wrapper actually hands over, at the wrapper's own
dtype, which is a different path from the one ``tests/test_lm_padded_head.py`` checks.

The tokenizer is stubbed. :meth:`transformers.AutoTokenizer.from_pretrained` fetches over the
network, and a test that needs a download is a test that fails for a reason that is not the
code's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
# The submodule, not the package: it imports accelerate, which lm_eval declares only in an
# extra, so a host with lm_eval alone raises here instead of skipping.
pytest.importorskip("lm_eval.models.huggingface")

from lm_eval.api.registry import get_model  # type: ignore[import-not-found]

from scripts.lm import shim
from scripts.lm.checkpoint import save
from scripts.lm.corpus import CorpusManifest, ShardCounts
from scripts.lm.mixers import REGISTRY
from scripts.lm.model import LMConfig, MixerLM, build_model, layer_factories
from scripts.lm.shim import SLinOSSEvalWrapper

D_MODEL = 64
N_LAYERS = 2
VOCAB = 100
MAX_LENGTH = 16
BATCH = 4


class _Tokenizer:
    """The tokenizer surface the wrapper touches.

    ``eos_token_id`` is read twice: once for the pad slot the batcher writes, once for the
    document separator ``lm_eval`` joins contexts with. Nothing else is reached before a task
    runs, so a stub is enough to build the wrapper and keeps the assertions on the wiring.
    """

    name: str = ""
    eos_token_id: int = 0
    pad_token_id: int | None = None

    @classmethod
    def from_pretrained(cls, name: str) -> _Tokenizer:
        """Return a stub, recording the name it was asked for."""
        instance = cls()
        instance.name = name
        return instance


def _manifest() -> CorpusManifest:
    """A manifest naming the tokenizer and the vocabulary the head was sized for."""
    return CorpusManifest(
        tokenizer="EleutherAI/gpt-neox-20b",
        vocab_size=VOCAB,
        eot_token_id=0,
        dataset="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
        revision=None,
        text_field="text",
        dtype="uint16",
        train=ShardCounts(tokens=1024, text_bytes=4096, digest="a" * 64),
        val=ShardCounts(tokens=256, text_bytes=1024, digest="b" * 64),
    )


def _save(path: Path, *, manifest: CorpusManifest | None) -> MixerLM:
    """Write an arm and return the stack that was written.

    The control mixer, because the wrapper is mixer-agnostic by construction: the checkpoint
    names its own mixer and the registry rebuilds it. Which mixer that is
    ``tests/test_lm_checkpoint.py`` covers.
    """
    torch.manual_seed(0)
    config = LMConfig(d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB)
    resolved = REGISTRY.resolve("conv")
    stack = build_model(
        config, layer_factories(resolved.factory, N_LAYERS), max_length=MAX_LENGTH
    )
    save(
        path,
        stack,
        config=config,
        mixer="conv",
        mixer_settings=resolved.settings,
        max_length=MAX_LENGTH,
        step=11,
        lr=1.5e-3,
        embedding_lr=0.11,
        seed=0,
        manifest=manifest,
    )
    return stack


def _wrap(
    path: Path, monkeypatch: pytest.MonkeyPatch, *, dtype: str = "float32"
) -> SLinOSSEvalWrapper:
    """Build the wrapper with the tokenizer stubbed out."""
    monkeypatch.setattr(shim, "AutoTokenizer", _Tokenizer)
    return SLinOSSEvalWrapper(
        str(path),
        max_length=MAX_LENGTH,
        batch_size=BATCH,
        device="cpu",
        dtype=dtype,
    )


def _ids() -> torch.Tensor:
    """A fixed batch of token ids."""
    return torch.arange(2 * 8, dtype=torch.int64).remainder(VOCAB).reshape(2, 8)


def test_the_wrapper_is_what_the_model_name_resolves_to() -> None:
    """``--model slinoss`` has to reach this class.

    The registration is a call after the class body rather than a decorator on it, because the
    decorator erases the class type. A call is easy to lose in a refactor and nothing else here
    would notice: ``lm_eval`` would report an unknown model name and every arm would be
    unscored.
    """
    assert get_model("slinoss") is SLinOSSEvalWrapper


def test_the_model_call_returns_the_stack_s_logits_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole override, against the stack that was saved.

    A tolerance here would admit a dtype demotion or a cast on the way through the wrapper,
    and a zero-shot score computed on a demoted model is not the score of the model that
    trained. The output must also carry no grad: eight tasks scored with an autograd graph
    alive is an out-of-memory failure, not a wrong number, but it is one this checks for free.
    """
    path = tmp_path / "model.pt"
    saved = _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    ids = _ids()
    saved.eval()
    with torch.no_grad():
        expected = saved(ids)
    produced = wrapper._model_call(ids)
    assert torch.equal(produced, expected)
    assert not produced.requires_grad


def test_the_row_handed_over_needs_no_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The padded columns are unreachable in the row ``lm_eval`` normalizes over.

    If the fill ever became ``0.0`` every ranking task would degrade silently: the normalizer
    would grow by the padding width, which is a function of the vocabulary and the alignment
    multiple, so two arms at one tokenizer would shift identically and the comparison would
    look unchanged while every absolute figure was wrong.
    """
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    logits = wrapper._model_call(_ids())
    assert logits.shape[-1] > wrapper.vocab_size
    assert torch.all(logits[..., VOCAB:] == torch.finfo(logits.dtype).min)
    assert torch.all(torch.isfinite(logits[..., :VOCAB]))
    assert torch.all(logits.softmax(dim=-1)[..., VOCAB:] == 0.0)


def test_a_scored_pair_is_the_stack_s_own_log_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scoring chain, from ``lm_eval``'s entry to the number a task ranks on.

    Everything above this test checks one attribute or one call. This one runs the batching,
    the padding, the shift and the gather that ``HFLM`` does between them, and compares the
    result to the same quantity computed off the saved stack: the continuation's tokens read
    out of a log-softmax over the row that precedes each of them.

    The value is what makes it worth running. Every alignment error available here -- a logit
    row read one position late, a continuation slice taken from the wrong end, a normalizer
    over the wrong width -- returns a number of the right sign and magnitude, and the ranking
    it produces looks like a measurement.

    Entered below ``loglikelihood`` because tokenization is not this harness's code: the ids
    are given, so what is under test is the part that touches the model.
    """
    path = tmp_path / "model.pt"
    saved = _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    context = [3, 5, 7, 11]
    continuation = [13, 17]
    scored = wrapper._loglikelihood_tokens(
        [(("context", "continuation"), context, continuation)]
    )
    assert len(scored) == 1
    logprob, is_greedy = scored[0]

    ids = torch.tensor([context + continuation], dtype=torch.int64)
    saved.eval()
    with torch.no_grad():
        rows = saved(ids[:, :-1])[0, -len(continuation) :].log_softmax(dim=-1)
    wanted = torch.tensor(continuation, dtype=torch.int64)
    expected = rows.gather(1, wanted[:, None]).sum().item()
    assert logprob == pytest.approx(expected, abs=1e-4)
    assert is_greedy == bool((rows.argmax(dim=-1) == wanted).all())


def test_the_model_info_recorded_is_the_checkpoint_s_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``simple_evaluate`` calls this on every model, after all scoring is done.

    ``HFLM``'s version reads four attributes its own ``__init__`` sets and then asks the Hub
    for a model SHA, so the inherited one raises an ``AttributeError`` at the end of a run that
    has already spent its compute. Overridden here to report what actually identifies a local
    arm: which mixer, which step, and which corpus digest the numbers belong to.
    """
    path = tmp_path / "model.pt"
    saved = _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    info = wrapper.get_model_info()
    assert info["model_num_parameters"] == sum(p.numel() for p in saved.parameters())
    assert info["model_dtype"] == str(torch.float32)
    assert info["mixer"] == "conv"
    assert info["step"] == 11
    assert info["train_sha256"] == _manifest().train.digest


def test_every_attribute_hflm_reads_off_self_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HFLM.__init__`` never runs, so none of these has a default here.

    Read rather than merely present: a batch size that was detected would differ per card, a
    context length that was not the training one would be an extrapolation measurement, and
    ``add_bos_token`` or ``logits_cache`` flipped would shift every likelihood.
    """
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    assert wrapper.backend == "causal"
    assert wrapper.add_bos_token is False
    assert wrapper.truncation is False
    assert wrapper.logits_cache is True
    assert wrapper.batch_sizes == {}
    assert wrapper.custom_prefix_token_id is None
    assert wrapper.softmax_dtype is torch.float32
    assert wrapper.batch_size == BATCH
    assert wrapper.max_length == MAX_LENGTH
    assert wrapper.max_gen_toks == 256
    assert wrapper.device == torch.device("cpu")
    assert wrapper.vocab_size == VOCAB


def test_the_tokenizer_is_the_corpus_s_own_with_a_pad_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest names the tokenizer, so scoring cannot use a different one than training.

    A tokenizer chosen at evaluation time would retokenize the tasks against a vocabulary the
    head was not sized for, and the pad slot has to exist because the batcher writes it.
    """
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    assert isinstance(wrapper.tokenizer, _Tokenizer)
    assert wrapper.tokenizer.name == _manifest().tokenizer
    assert wrapper.tokenizer.pad_token_id == wrapper.tokenizer.eos_token_id
    assert wrapper.eot_token_id == _Tokenizer.eos_token_id


def test_the_compute_dtype_is_the_one_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cast happens at load, and the short spelling is the same setting as the long one.

    A dtype argument that was accepted and dropped would report a bf16 number as a float32
    one. Checked on the parameters rather than through a forward, since what is under test is
    the plumbing and not the operator's behaviour at bf16.

    Not every parameter follows. :meth:`slinoss.SLinOSSStack._apply` and the block's undo a
    demotion of their RMSNorm weights, so those stay float32 by the operator's own contract
    and an assertion that every leaf is bf16 would be asserting a bug. Everything the cast is
    for -- the token table, the head, the projections -- has to move.
    """
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch, dtype="bf16")
    leaves = dict(wrapper.model.named_parameters())
    pinned = {name for name, param in leaves.items() if param.dtype is torch.float32}
    assert pinned == {name for name in leaves if name.endswith("norm_weight")}
    assert {leaves[name].dtype for name in leaves.keys() - pinned} == {torch.bfloat16}
    assert leaves["embedding.weight"].dtype is torch.bfloat16
    assert leaves["head.weight"].dtype is torch.bfloat16


def test_an_unknown_dtype_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused before the checkpoint is read, and the message lists what is accepted."""
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    with pytest.raises(ValueError, match="dtype must be one of"):
        _wrap(path, monkeypatch, dtype="float8")


def test_a_checkpoint_with_no_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a manifest the tokenizer is a guess, and a guessed tokenizer is a wrong table."""
    path = tmp_path / "model.pt"
    _save(path, manifest=None)
    with pytest.raises(ValueError, match="carries no manifest"):
        _wrap(path, monkeypatch)


def test_generation_is_refused_rather_than_approximated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The eight tasks rank log-likelihoods. A generation path here would be untested code.

    Refusing is the point: a task list that accidentally included a generative task would
    otherwise produce a number off an untested decode.
    """
    path = tmp_path / "model.pt"
    _save(path, manifest=_manifest())
    wrapper = _wrap(path, monkeypatch)
    with pytest.raises(NotImplementedError, match="scores likelihoods"):
        wrapper._model_generate(_ids(), max_length=4, stop=None)
