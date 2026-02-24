"""
ShinkaEvolve initial program for evolving FlashInfer kernels.

This is the ShinkaEvolve analogue of "candidate program text" in OpenEvolve:
OpenEvolve evolves a kernel as a raw string; ShinkaEvolve evolves a Python file,
mutating the region between `EVOLVE-BLOCK-START/END`.

Contract:
  - ShinkaEvolve evaluation script calls `run_experiment(**kwargs)`.
  - This program returns a dict containing at least `combined_score` (float, maximize).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# EVOLVE-BLOCK-START
def candidate_kernel() -> Dict[str, Any]:
    """
    Return the candidate kernel spec to evaluate.

    Code mode (what you evolve towards):
       {
         "mode": "code",
         "language": "cuda" | "triton" | "python",
         "code": "<inline source text OR a filesystem path to a source file>",
       }

       For CUDA you may also provide KernelGenerator XML with exactly 3 files:
         <header_file name="kernel.h"> ... </header_file>
         <cuda_file name="kernel.cu"> ... </cuda_file>
         <cpp_file name="main.cpp"> ... </cpp_file>
    """

    # Start "empty": returning empty code will fail fast with helpful feedback.
    # Evolution should quickly mutate this into a valid CUDA XML candidate.
    return {"mode": "code", "language": "cuda", "code": ""}

# EVOLVE-BLOCK-END


def run_experiment(**kwargs) -> Dict[str, Any]:
    """
    Entry point called by ShinkaEvolve evaluation.

    Expected kwargs (provided by evaluator config):
      - dataset_path (str)
      - definition (str)
      - language (optional)
      - target_gpu (optional)
      - warmup_runs / iterations / num_trials / rtol / atol / timeout_seconds
      - use_isolated_runner / parallel_workloads / feedback_workloads / ...
      - baseline_solution (optional): also used as the starting baseline in EVOLVE-BLOCK
    """
    from flashinfer_shinka_evaluator import FlashInferShinkaEvaluatorConfig, evaluate_candidate

    cfg = FlashInferShinkaEvaluatorConfig.from_config(kwargs)
    # Candidate is always a "code" candidate; baseline is only used during final evaluation
    # (see run_evolve.py's --final-eval-all-workloads path).
    cand = candidate_kernel()
    return evaluate_candidate(cfg=cfg, candidate=cand)

