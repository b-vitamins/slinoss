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

   ;; Test, lint, and type checking.
   "python-pytest"
   "python-pytest-cov"
   "python-ruff"
   "node-pyright"

   ;; Plotting for assets/ (300 DPI minimum).
   "python-matplotlib"

   ;; Misc utilities.
   "unzip"))
