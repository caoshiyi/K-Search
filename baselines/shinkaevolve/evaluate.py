"""
Toy ShinkaEvolve evaluator for `baselines/shinkaevolve/initial.py`.

This follows the pattern shown in ShinkaEvolve's README:
`evaluate.py` uses `shinka.core.run_shinka_eval(...)` to run the candidate
program's `run_experiment` multiple times, validate outputs, and aggregate into
metrics including a single `combined_score` (higher is better).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _target(x: float) -> float:
    return (x - 3.0) ** 2


def get_kwargs(run_idx: int) -> dict:
    # Deterministic small suite of test points (varies per run).
    suites = [
        [-5.0, -2.0, 0.0, 2.0, 5.0],
        [-10.0, -1.0, 1.0, 3.0, 7.0],
        [0.5, 1.5, 2.5, 3.5, 4.5],
    ]
    xs = suites[run_idx % len(suites)]
    return {"xs": xs}


def validate_fn(result: Any, *, xs: List[float]) -> Tuple[bool, str]:
    """
    Validate the program output for one run.

    `result` is whatever `run_experiment(**kwargs)` returned from the candidate program.
    """
    if not isinstance(result, dict):
        return False, f"Expected dict result, got {type(result).__name__}"
    preds = result.get("preds", None)
    if not isinstance(preds, list):
        return False, f"Expected result['preds'] to be a list, got {type(preds).__name__}"
    if len(preds) != len(xs):
        return False, f"Expected {len(xs)} preds, got {len(preds)}"
    for i, p in enumerate(preds):
        if not isinstance(p, (int, float)):
            return False, f"preds[{i}] is not a number: {type(p).__name__}"
        pf = float(p)
        if not math.isfinite(pf):
            return False, f"preds[{i}] is not finite: {pf}"
    return True, ""


def _score_one_run(result: Dict[str, Any], *, xs: List[float]) -> Dict[str, float]:
    preds = [float(x) for x in result["preds"]]
    ys = [_target(float(x)) for x in xs]
    mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / float(len(xs) or 1)
    # Higher is better; use negative MSE.
    score = -float(mse)
    return {"mse": float(mse), "score": float(score)}


def aggregate_fn(results: List[Any]) -> dict:
    """
    Aggregate across runs.

    ShinkaEvolve expects a dict with at least:
      - combined_score: float (maximize)
    """
    # `results` are raw values returned by the candidate program on each run.
    per_run: List[Dict[str, float]] = []
    for run_idx, r in enumerate(results):
        xs = get_kwargs(run_idx)["xs"]
        if isinstance(r, dict) and "preds" in r:
            per_run.append(_score_one_run(r, xs=xs))
        else:
            per_run.append({"mse": float("inf"), "score": -1e30})

    mean_mse = sum(m["mse"] for m in per_run) / float(len(per_run) or 1)
    mean_score = sum(m["score"] for m in per_run) / float(len(per_run) or 1)

    return {
        "combined_score": float(mean_score),
        "public": {"mean_mse": float(mean_mse), "mean_score": float(mean_score)},
        "private": {},
        "extra_data": {"per_run": per_run},
        "text_feedback": (
            f"Mean MSE={mean_mse:.6f}. "
            "Improve predict(x) to better match target(x)=(x-3)^2 across the test points."
        ),
    }


def main(*, program_path: str, results_dir: str, num_runs: int) -> int:
    try:
        from shinka.core import run_shinka_eval  # type: ignore
    except Exception as e:
        print("ERROR: ShinkaEvolve (shinka) is not installed in this environment.")
        print("Install it following upstream docs:", "https://github.com/SakanaAI/ShinkaEvolve")
        print("Import error:", repr(e))
        return 2

    program_path_p = Path(program_path).expanduser().resolve()
    results_dir_p = Path(results_dir).expanduser().resolve()
    results_dir_p.mkdir(parents=True, exist_ok=True)

    metrics, correct, err = run_shinka_eval(
        program_path=str(program_path_p),
        results_dir=str(results_dir_p),
        experiment_fn_name="run_experiment",
        num_runs=int(num_runs),
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )

    print("correct:", correct)
    if err:
        print("err:", err)
    print("metrics:", metrics)
    return 0 if correct else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a ShinkaEvolve program (toy quadratic fit).")
    p.add_argument("--program-path", type=str, default=str(Path(__file__).with_name("initial.py")))
    p.add_argument("--results-dir", type=str, default=str(Path(__file__).with_name("results_smoke")))
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()
    raise SystemExit(main(program_path=args.program_path, results_dir=args.results_dir, num_runs=args.num_runs))

