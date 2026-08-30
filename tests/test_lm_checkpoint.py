"""The checkpoint: rebuilt from the registry, loaded strictly, readable without a class.

Everything downstream of training reads a checkpoint, so a checkpoint that rebuilds the wrong
model is a table of wrong numbers with no symptom. Three properties make it not that.

It reloads bit for bit. A tolerance here would hide a dtype demotion on the way out, and a
zero-shot score computed on a demoted model is not the score of the model that trained.

It rebuilds from the stored settings and not from the registry's current defaults. A default
that moves between the run and the evaluation would silently score a different arm.

It loads strictly. A state dict that does not match the rebuilt module is a different program,
and a partial load leaves a freshly initialized layer in the middle of a trained stack.

The manifest travels with it, because the digest is what lets a zero-shot score and the
bits-per-byte next to it be known to come from one corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.lm.checkpoint import FORMAT, load, load_model, save
from scripts.lm.corpus import CorpusManifest, ShardCounts
from scripts.lm.mixers import REGISTRY
from scripts.lm.model import build_model, layer_factories, scaffold_config
from slinoss import SLinOSSBlock, SLinOSSConfig, SLinOSSMixer, SLinOSSStack

D_MODEL = 64
N_LAYERS = 2
VOCAB = 64
MAX_LENGTH = 16
D_STATE = 96


def _manifest() -> CorpusManifest:
    """A manifest with both shards counted."""
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


def _config() -> SLinOSSConfig:
    """The scaffold config."""
    return scaffold_config(
        d_model=D_MODEL, n_layers=N_LAYERS, vocab_size=VOCAB, d_state=48, d_head=16
    )


def _stack(mixer: str = "slinoss", overrides: tuple[str, ...] = ()) -> SLinOSSStack:
    """A built arm, and the settings it was built at."""
    torch.manual_seed(0)
    resolved = REGISTRY.resolve(mixer, overrides)
    return build_model(
        _config(), layer_factories(resolved.factory, N_LAYERS), max_length=MAX_LENGTH
    )


def _mixers(model: SLinOSSStack) -> list[nn.Module]:
    """The mixer in each block, in order."""
    return [block.mixer for block in model.blocks if isinstance(block, SLinOSSBlock)]


def _save(
    path: Path, mixer: str = "slinoss", overrides: tuple[str, ...] = ()
) -> SLinOSSStack:
    """Build an arm, perturb it so its weights are not an initialization, and write it."""
    stack = _stack(mixer, overrides)
    with torch.no_grad():
        for param in stack.parameters():
            param.add_(torch.full_like(param, 0.01))
    resolved = REGISTRY.resolve(mixer, overrides)
    save(
        path,
        stack,
        config=_config(),
        mixer=mixer,
        mixer_settings=resolved.settings,
        max_length=MAX_LENGTH,
        step=17,
        lr=1.5e-3,
        embedding_lr=0.11,
        seed=3,
        manifest=_manifest(),
    )
    return stack


def test_the_checkpoint_reads_back_without_unpickling_a_class(tmp_path: Path) -> None:
    """Every stored value is a scalar, a string, a plain dict or a tensor.

    :func:`torch.load` runs with ``weights_only=True``, so a payload holding a dataclass or a
    namedtuple would raise on load rather than at save time. That is why the config and the
    manifest are stored as dicts and reconstructed.
    """
    path = tmp_path / "model.pt"
    _save(path)
    checkpoint = load(path)
    assert checkpoint.config == _config()
    assert checkpoint.manifest == _manifest()
    assert checkpoint.mixer == "slinoss"
    assert checkpoint.mixer_settings["d_state"] == D_STATE
    assert checkpoint.step == 17
    assert checkpoint.seed == 3
    assert checkpoint.max_length == MAX_LENGTH
    assert checkpoint.hybrid_final is None


def test_the_reloaded_model_is_the_saved_one_bit_for_bit(tmp_path: Path) -> None:
    """No tolerance. A demoted parameter would score a different program than trained."""
    path = tmp_path / "model.pt"
    saved = _save(path)
    loaded, _ = load_model(path)
    original = saved.state_dict()
    for name, tensor in loaded.state_dict().items():
        assert tensor.dtype is original[name].dtype
        assert torch.equal(tensor, original[name])


def test_the_reloaded_model_gives_the_same_logits(tmp_path: Path) -> None:
    """The end-to-end property, since the state dict is a means to it.

    The control mixer, and the property is not weaker for it: the stack holds no buffers, so
    the parameters are its whole state and which mixer consumes them is not what is under
    test. The reason it is not the slinoss arm is that the mixer hands the scan a pitched band
    of one fused projection, and that operand contract is CUDA-only in both paths, so a
    slinoss forward on the CPU raises rather than falling back.
    """
    path = tmp_path / "model.pt"
    saved = _save(path, "conv")
    loaded, _ = load_model(path)
    ids = torch.arange(8, dtype=torch.int64).remainder(VOCAB).reshape(2, 4)
    saved.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(saved(ids), loaded(ids))


def test_the_mixer_is_rebuilt_at_the_stored_settings(tmp_path: Path) -> None:
    """Not at the registry's current defaults.

    A default that moved between the run and the evaluation would rebuild a different arm and
    the strict load would then fail on a shape -- or, at a setting that does not change a
    shape, succeed and score the wrong thing.
    """
    path = tmp_path / "model.pt"
    _save(path, overrides=("d_state=144",))
    loaded, checkpoint = load_model(path)
    assert checkpoint.mixer_settings["d_state"] == 144
    assert REGISTRY.entry("slinoss").defaults["d_state"] == D_STATE
    for mixer in _mixers(loaded):
        assert isinstance(mixer, SLinOSSMixer)
        assert mixer.config.d_state == 144


def test_a_hybrid_s_last_layer_survives_the_round_trip(tmp_path: Path) -> None:
    """The per-layer choice is part of what the checkpoint has to name.

    A hybrid reloaded as a uniform stack would fail the strict load only if the two mixers
    disagree on a parameter shape, so the layer identity has to be checked directly.
    """
    path = tmp_path / "model.pt"
    final = REGISTRY.resolve("conv")
    hybrid = build_model(
        _config(),
        layer_factories(REGISTRY.resolve("slinoss").factory, N_LAYERS, final.factory),
        max_length=MAX_LENGTH,
    )
    save(
        path,
        hybrid,
        config=_config(),
        mixer="slinoss",
        mixer_settings=REGISTRY.resolve("slinoss").settings,
        max_length=MAX_LENGTH,
        step=1,
        lr=1e-3,
        embedding_lr=0.1,
        seed=0,
        manifest=_manifest(),
        hybrid_final="conv",
        hybrid_final_settings=final.settings,
    )
    loaded, checkpoint = load_model(path)
    assert checkpoint.hybrid_final == "conv"
    mixers = _mixers(loaded)
    assert len(mixers) == N_LAYERS
    assert isinstance(mixers[0], SLinOSSMixer)
    assert not isinstance(mixers[-1], SLinOSSMixer)


def test_a_state_dict_that_does_not_match_is_refused(tmp_path: Path) -> None:
    """Strict, so a missing key is an error and not a freshly initialized layer.

    A partial load is the worst failure available here: it runs, it produces logits, and the
    numbers are those of a stack with one untrained block in it.
    """
    path = tmp_path / "model.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    dropped = next(iter(payload["state_dict"]))
    del payload["state_dict"][dropped]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)


def test_another_layout_version_is_refused_not_guessed_at(tmp_path: Path) -> None:
    """A file from another version of this harness is not readable by accident."""
    path = tmp_path / "model.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format"] = FORMAT + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match=f"format is {FORMAT + 1}"):
        load(path)


def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    """An evaluation pointed at nothing must say so rather than report a fresh model."""
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        load(tmp_path / "absent.pt")


def test_an_unregistered_mixer_is_refused_at_load(tmp_path: Path) -> None:
    """A baseline's checkpoint needs that baseline's package, and says which."""
    path = tmp_path / "model.pt"
    _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["mixer"] = "not_a_mixer"
    torch.save(payload, path)
    with pytest.raises(KeyError, match="no lm mixer not_a_mixer"):
        load_model(path)
