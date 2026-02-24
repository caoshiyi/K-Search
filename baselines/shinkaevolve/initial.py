"""
Toy ShinkaEvolve program: fit a quadratic.

The evaluator (evaluate.py) will test the candidate "model" on a few x-values and
score it against the target function:

    target(x) = (x - 3)^2

ShinkaEvolve should evolve the block between EVOLVE-BLOCK markers.
"""

from __future__ import annotations

from typing import Dict, List


# EVOLVE-BLOCK-START
def predict(x: float) -> float:
    """
    Candidate model.

    Starting point is intentionally weak; evolution should improve it.
    """
    # Constant predictor (bad baseline).
    return 0.0


def model_name() -> str:
    return "constant_0"

# EVOLVE-BLOCK-END


def run_experiment(*, xs: List[float]) -> Dict[str, object]:
    """
    Entry point called by ShinkaEvolve evaluation.

    Must be importable and callable as `run_experiment(**kwargs)` where kwargs are
    provided by `evaluate.py` via `get_experiment_kwargs`.
    """
    preds = [float(predict(float(x))) for x in xs]
    return {
        "model": str(model_name()),
        "preds": preds,
    }

