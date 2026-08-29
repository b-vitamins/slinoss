;; Guix manifest for slinoss.
;;
;; Usage:
;;   guix shell -m manifest.scm -- python3 -m pytest -xvs
;;   guix shell -m manifest.scm -- ruff format . && ruff check .
;;   guix shell -m manifest.scm -- pyright

(specifications->manifest
 '(;; Python and the core numeric stack.
   "python"
   "python-pytorch-cuda"
   "python-numpy@1"
   "python-einops"

   ;; CuTe DSL and CUDA toolchain.
   "python-nvidia-cutlass-dsl"
   "python-cuda-python"
   "cuda-toolkit"
   "cutlass-headers"
   "cutlass-tools"
   "cudnn"
   "onednn"

   ;; Profiling and instrumentation.
   "python-nvtx"
   "python-cupti-python"
   "python-ncu-report"
   "python-nsight-python"

   ;; Native build toolchain for the causal conv1d extension.
   "gcc-toolchain@14"
   "cmake"
   "ninja"
   "pybind11"

   ;; Corpus preparation and zero-shot evaluation for scripts/lm. The tokenizer
   ;; and the dataset stream are used once, by `scripts.lm.run prep`, and write a
   ;; token file with a digest; the loop itself reads that file and imports
   ;; neither. lm-eval owns the eight tasks, their prompts and their metrics: a
   ;; ranking computed here would be a ranking nobody else's numbers compare to.
   "python-datasets"
   "python-transformers"
   "python-tokenizers"
   "python-lm-eval"
   ;; lm-eval's HuggingFace model module imports accelerate, which lm-eval itself
   ;; declares only in an extra.
   "python-accelerate"

   ;; Ground truth for the UEA partition. scripts/tsc reproduces JAX's Threefry
   ;; stream in numpy because the published bars were measured on it; the pinning
   ;; tests import JAX itself and skip where it is absent, so this is test-only and
   ;; nothing under scripts/ imports it.
   "python-jax"

   ;; Test, lint, and type checking.
   "python-pytest"
   "python-pytest-cov"
   "python-ruff"
   "node-pyright"

   ;; Plotting for assets/ (300 DPI minimum).
   "python-matplotlib"

   ;; Misc utilities.
   "unzip"))
