"""
ShinkaEvolve evaluation entrypoint for FlashInfer kernel evolution.

This mirrors the OpenEvolve pattern:
  - OpenEvolve: evaluator is a callable object; run_evolve.py wires it
  - ShinkaEvolve: evaluator is an external script invoked with (program_path, results_dir)

This script:
  - loads a YAML evaluator config (dataset_path, definition, benchmark knobs)
  - calls `shinka.core.run_shinka_eval(...)` to run the program's `run_experiment`
  - aggregates results into `combined_score` (maximize)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_mapping(path: str) -> dict:
    """Load a small YAML mapping file for evaluator settings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix not in (".yaml", ".yml"):
        raise ValueError("Unsupported evaluator config file type (use .yaml/.yml)")
    try:
        from omegaconf import OmegaConf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OmegaConf is required. Install with: pip install omegaconf") from e
    cfg = OmegaConf.load(str(p))
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Evaluator config must be a mapping")
    return data


def get_kwargs_factory(base_cfg: Dict[str, Any]):
    def get_kwargs(run_idx: int) -> dict:
        # Pass run_idx through (useful if the evolved code wants deterministic per-run variation).
        return {**base_cfg, "run_idx": int(run_idx)}

    return get_kwargs


def validate_fn(result: Any, **_: Any) -> Tuple[bool, str]:
    if not isinstance(result, dict):
        return False, f"Expected dict result, got {type(result).__name__}"
    cs = result.get("combined_score", None)
    if not isinstance(cs, (int, float)):
        return False, "Missing/invalid combined_score (must be a number)"
    return True, ""


def aggregate_fn(results: List[Any]) -> dict:
    vals: List[float] = []
    best_trace_log = ""
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("combined_score", None), (int, float)):
            vals.append(float(r["combined_score"]))
            tl = r.get("trace_log", "")
            if isinstance(tl, str) and tl and not best_trace_log:
                best_trace_log = tl
        else:
            vals.append(0.0)

    mean_score = sum(vals) / float(len(vals) or 1)
    return {
        "combined_score": float(mean_score),
        "public": {"mean_combined_score": float(mean_score)},
        "private": {},
        "extra_data": {"per_run_combined_score": vals},
        "text_feedback": (best_trace_log[:8000] if best_trace_log else ""),
    }


def _tail_text(path: Path, *, max_chars: int = 6000, max_lines: int = 200) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = txt.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    out = "\n".join(lines).strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[-max_chars:]
    return out


def _extract_error_line_snippet(err: str, program_path: Path) -> str:
    """
    Best-effort: when Python errors mention `line N`, include a small code excerpt.
    """
    import re

    m = re.search(r"\bline\s+(\d+)\b", err or "")
    if not m:
        return ""
    try:
        line_no = int(m.group(1))
    except Exception:
        return ""
    if line_no <= 0:
        return ""
    try:
        lines = program_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    idx = line_no - 1
    lo = max(0, idx - 2)
    hi = min(len(lines), idx + 3)
    excerpt = "\n".join(f"{i+1}: {lines[i]}" for i in range(lo, hi)).rstrip()
    return excerpt


def _augment_text_feedback(
    *,
    metrics: Dict[str, Any],
    correct: bool,
    err: str | None,
    program_path: Path,
    results_dir: Path,
) -> None:
    """
    Ensure failures produce actionable feedback for the next LLM prompt.

    Why: if evaluation fails before `run_experiment()` returns (e.g., Python syntax/import
    errors), there may be no `trace_log`, so the model would otherwise get no signal.
    """
    existing = metrics.get("text_feedback", "")
    existing = str(existing) if existing is not None else ""

    # If we're correct and already have feedback, keep it as-is.
    if correct and existing.strip():
        metrics["text_feedback"] = existing[:8000]
        return

    parts: List[str] = []
    if existing.strip():
        parts.append(existing.strip())

    # Top-level error from run_shinka_eval (includes syntax/import errors).
    if err:
        parts.append(f"[error]\n{str(err).strip()}")
        snippet = _extract_error_line_snippet(str(err), program_path)
        if snippet:
            parts.append("[error_context]\nOffending code excerpt:\n" + snippet)

    # Validation errors recorded by run_shinka_eval (exception path populates this).
    try:
        ve = metrics.get("all_validation_errors", None)
        if isinstance(ve, list) and ve:
            parts.append("[validation_errors]\n" + "\n".join(str(x) for x in ve[:10]))
    except Exception:
        pass

    # Also include the captured stderr/stdout logs of this evaluator process (tail only).
    # These often include compiler errors, stack traces, and subprocess logs.
    for name, max_chars in (("job_log.err", 6000), ("job_log.out", 3000)):
        p = results_dir / name
        if p.exists() and p.is_file():
            tail = _tail_text(p, max_chars=max_chars, max_lines=220)
            if tail:
                parts.append(f"[{name} tail]\n{tail}")

    out = "\n\n".join(p for p in parts if p and p.strip()).strip()
    metrics["text_feedback"] = out[:8000] if out else ""


def main(argv: List[str] | None = None) -> int:
    # Ensure the K-Search repo root is importable even when this evaluator is launched
    # from a different working directory (ShinkaEvolve runs from results_*/...).
    # This is required because the evolved program imports `flashinfer_bench` from this repo.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="ShinkaEvolve evaluator for FlashInfer kernel evolution")
    # ShinkaEvolve JobScheduler passes `--program_path` and `--results_dir` (underscore),
    # so accept both hyphen and underscore spellings.
    p.add_argument("--program-path", "--program_path", dest="program_path", type=str, required=True)
    p.add_argument("--results-dir", "--results_dir", dest="results_dir", type=str, required=True)
    p.add_argument("--evaluator-config", type=str, required=True, help="YAML mapping (dataset_path, definition, ...)")
    p.add_argument("--num-runs", type=int, default=1, help="Num runs passed to run_shinka_eval (usually 1; bench is expensive)")
    args = p.parse_args(argv)

    try:
        from shinka.core import run_shinka_eval  # type: ignore
    except Exception as e:
        print("ERROR: ShinkaEvolve (shinka) is not installed in this environment.")
        print("Install it following upstream docs:", "https://github.com/SakanaAI/ShinkaEvolve")
        print("Import error:", repr(e))
        return 2

    base_cfg = _load_mapping(args.evaluator_config)
    get_kwargs = get_kwargs_factory(base_cfg)

    metrics, correct, err = run_shinka_eval(
        program_path=str(Path(args.program_path).expanduser().resolve()),
        results_dir=str(Path(args.results_dir).expanduser().resolve()),
        experiment_fn_name="run_experiment",
        num_runs=int(args.num_runs),
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )

    # Post-process: ensure failures still have meaningful text_feedback (syntax/import errors,
    # compilation logs, etc.). Then overwrite metrics.json so ShinkaEvolve picks it up.
    prog_path = Path(args.program_path).expanduser().resolve()
    res_dir = Path(args.results_dir).expanduser().resolve()
    try:
        _augment_text_feedback(
            metrics=metrics,
            correct=bool(correct),
            err=(str(err) if err else None),
            program_path=prog_path,
            results_dir=res_dir,
        )
        metrics_file = res_dir / "metrics.json"
        if metrics_file.exists():
            metrics_file.write_text(json.dumps(metrics, indent=4) + "\n", encoding="utf-8")
    except Exception:
        # Never fail evaluation due to feedback generation.
        pass

    print("correct:", correct)
    if err:
        print("err:", err)
    print("metrics:", metrics)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
