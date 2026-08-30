"""The bridge to ``lm_eval``: one wrapper, one overridden call, no evaluation logic.

``lm_eval`` owns the tasks, the prompts, the metrics and the ranking. This module supplies a
model it can call and nothing else, because a ranking computed here would be a ranking nobody
else's numbers are comparable to.

:class:`HFLM` reads ``self.model(inps).logits``. :meth:`slinoss.SLinOSSStack.forward` returns
a bare tensor, so :meth:`SLinOSSEvalWrapper._model_call` is the one override. Everything else
is attribute wiring: the checkpoint's tokenizer, the padded batch token, the context length.

The padded head needs no slice. :meth:`slinoss.SLinOSSStack.forward` fills the columns past
``vocab_size`` with ``finfo(dtype).min``, which is exactly zero under a softmax and below every
reachable logit, so a ranking over the padded width is the ranking over the first
``vocab_size`` columns. This is a contract of the operator, and
``tests/test_lm_shim.py`` pins it: if the fill ever becomes ``0.0`` every ranking task
degrades silently, and a silent degradation is the failure this harness is built to refuse.

Run it as the harness's own entry point:

    python3 -m scripts.lm.shim --model slinoss \\
        --model_args pretrained=runs/slinoss/model.pt,batch_size=16 \\
        --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,boolq \\
        --output_path runs/slinoss/zero_shot.json

then fold the result into the arm's record with ``scripts.lm.run merge``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import transformers  # type: ignore[import-not-found]
from lm_eval.__main__ import cli_evaluate  # type: ignore[import-not-found]
from lm_eval.api.model import LM  # type: ignore[import-not-found]
from lm_eval.api.registry import register_model  # type: ignore[import-not-found]
from lm_eval.models.huggingface import HFLM  # type: ignore[import-not-found]
from torch import Tensor
from transformers import AutoTokenizer  # type: ignore[import-not-found]

from scripts.lm.checkpoint import load_model

__all__ = ["SLinOSSEvalWrapper"]

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


_HFLMBase: Any = HFLM
"""The base, stated as untyped because that is what it is.

``lm_eval`` decorates :class:`HFLM` with ``register_model``, whose inner function is
unannotated and asserts ``issubclass(cls, LM)``. A type checker therefore reads ``HFLM`` as
``type[LM]``, and every member this class inherits -- the tokenizer paths, the batching, the
context-length property -- is gone from that type along with ``HFLM.__init__``. Aliasing the
base is one statement of the fact; the alternative is a suppression at each inherited
attribute, which would also hide a real one going missing.
"""


class SLinOSSEvalWrapper(_HFLMBase):
    """A trained arm, in the shape ``lm_eval`` calls.

    Every arm goes through this wrapper, baselines included: the checkpoint names its own
    mixer and :func:`scripts.lm.checkpoint.load_model` rebuilds it, so one registered model
    covers the whole table and no arm is evaluated by a different code path than the arm it
    is compared to.

    Args:
        pretrained: Path to the arm's ``.pt``.
        max_length: Context the tasks are scored at. The training context by default; a
            longer one is an extrapolation measurement and has to be asked for.
        batch_size: Sequences per forward.
        device: Where to run.
        dtype: Compute dtype. Float32 by default, which is the dtype the checkpoint holds;
            a lower one is a different program from the one that trained.

    Raises:
        FileNotFoundError: When the checkpoint is absent.
        ValueError: On an unknown dtype name, or on a checkpoint with no manifest, which
            cannot name its tokenizer.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained: str,
        max_length: int = 2048,
        batch_size: int | str | None = None,
        device: str = "cuda",
        dtype: str = "float32",
    ) -> None:
        LM.__init__(self)
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        model, checkpoint = load_model(
            Path(pretrained), device=device, dtype=_DTYPES[dtype]
        )
        if checkpoint.manifest is None:
            raise ValueError(
                f"{pretrained} carries no manifest, so its tokenizer is unknown"
            )
        model.eval()
        self._model = model
        self._checkpoint = checkpoint
        self._manifest = checkpoint.manifest
        self._dtype = _DTYPES[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint.manifest.tokenizer)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = checkpoint.manifest.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 16
        self._max_length = max_length
        self._device = torch.device(device)
        # Attributes HFLM's tokenization and batching paths read off self. Set explicitly
        # rather than inherited: HFLM sets them in an __init__ this class does not call, so
        # a default that moves upstream would surface as a missing attribute mid-run.
        self.backend = "causal"
        self.add_bos_token = False
        self.truncation = False
        self.logits_cache = True
        self.batch_sizes = {}
        self.custom_prefix_token_id = None
        self._max_gen_toks = 256
        # The dtype the log-softmax over the padded row is taken in, float32 whatever the
        # model runs at: a ranking that moved with the compute dtype would not be comparable
        # across arms evaluated at different ones.
        self.softmax_dtype = torch.float32

    @property
    def batch_size(self) -> int:
        """Sequences per forward.

        Returns:
            The batch size. Fixed, never detected: a detected size would differ per card
            and per arm, and a rank the batch size moved is not a rank.
        """
        return self._batch_size

    def get_model_info(self) -> dict[str, Any]:
        """What ``lm_eval`` records next to the scores.

        Overridden, not inherited. ``simple_evaluate`` calls this on every model it is handed,
        and ``HFLM``'s reads ``self.revision``, ``self.pretrained``, ``self.peft`` and
        ``self.delta`` -- four attributes its own ``__init__`` sets -- and then asks the Hub
        for a model SHA. Here the arm is a local file and its provenance is the checkpoint's,
        so this reports that instead of raising after every task has been scored.

        Returns:
            The parameter count, the compute dtype, and the three fields that identify which
            arm and which corpus the numbers belong to.
        """
        return {
            "model_num_parameters": sum(
                param.numel() for param in self._model.parameters()
            ),
            "model_dtype": str(self._dtype),
            "mixer": self._checkpoint.mixer,
            "step": self._checkpoint.step,
            "train_sha256": self._manifest.train.digest,
        }

    def _model_call(
        self, inps: Tensor, attn_mask: Any = None, labels: Any = None
    ) -> Tensor:
        """Logits for one batch of token ids.

        The whole override. ``HFLM`` expects a model whose call returns an object with a
        ``logits`` field; this one returns the tensor.

        Args:
            inps: ``(B,T)`` int64 token ids.
            attn_mask: Unused. The stack is causal by construction and every sequence in a
                batch is the same length here.
            labels: Unused.

        Returns:
            ``(B,T,padded_vocab_size)`` logits. The columns past ``vocab_size`` hold
            ``finfo(dtype).min``, so no slice is needed.
        """
        del attn_mask, labels
        with torch.no_grad():
            return self._model(inps)

    def _model_generate(
        self, context: Tensor, max_length: int, stop: Any, **generation_kwargs: Any
    ) -> Tensor:
        """Not implemented.

        The eight zero-shot tasks are ranking tasks and need log-likelihoods, not samples.
        A generation path would need the decode state and would be untested here.

        Args:
            context: Prompt ids.
            max_length: Requested length.
            stop: Stop strings.
            **generation_kwargs: Sampler settings.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "this harness scores likelihoods, it does not generate"
        )


# Registered after the class rather than as a decorator on it. ``register_model``'s inner
# function is unannotated and asserts ``issubclass(cls, LM)``, so a decorated class narrows to
# ``type[LM]`` and every attribute this one adds, its constructor included, disappears from the
# public type. The registration is the same either way: the name is what ``--model`` resolves.
register_model("slinoss")(SLinOSSEvalWrapper)


if __name__ == "__main__":
    cli_evaluate()
