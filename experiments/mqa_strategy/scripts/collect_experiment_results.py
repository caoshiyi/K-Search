#!/usr/bin/env python3
"""Collect and compare results from strategy-form experiments.

Scans the experiment artifacts directories for eval reports, solution JSONs,
and world model snapshots, then produces a comparative summary.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def find_experiment_dirs(root: Path) -> list[dict]:
    """Find all experiment artifact directories."""
    results = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(".ksearch-exp-"):
            continue
        info = {"dir": d, "name": d.name}
        # Try to extract experiment type from name
        if "baseline_llm" in d.name:
            info["type"] = "baseline_llm"
        elif "strat_nl" in d.name:
            info["type"] = "natural_language"
        elif "strat_sp" in d.name:
            info["type"] = "structured_params"
        elif "strat_dsl" in d.name:
            info["type"] = "dsl"
        else:
            info["type"] = "unknown"
        results.append(info)
    return sorted(results, key=lambda x: x["name"])


def collect_eval_data(exp_dir: Path) -> dict:
    """Collect evaluation data from an experiment directory."""
    data = {
        "final_speedup": None,
        "final_latency_ms": None,
        "best_score": None,
        "solutions": [],
        "wm_snapshots": [],
        "eval_reports": [],
    }

    # Find eval reports
    eval_dir = exp_dir / "eval"
    if eval_dir.exists():
        for f in eval_dir.rglob("*.json"):
            try:
                report = json.loads(f.read_text(encoding="utf-8"))
                data["eval_reports"].append({
                    "path": str(f),
                    "report": report,
                })
                # Extract final speedup
                if isinstance(report, dict):
                    for key in ("speedup_factor", "mean_vs_baseline_factor"):
                        val = report.get(key)
                        if isinstance(val, (int, float)):
                            if data["final_speedup"] is None or val > data["final_speedup"]:
                                data["final_speedup"] = float(val)
                    for key in ("latency_ms", "mean_latency_ms"):
                        val = report.get(key)
                        if isinstance(val, (int, float)):
                            if data["final_latency_ms"] is None:
                                data["final_latency_ms"] = float(val)
            except Exception:
                pass

    # Find solutions
    sol_dir = exp_dir / "solutions"
    if sol_dir.exists():
        for f in sol_dir.rglob("*.json"):
            try:
                sol = json.loads(f.read_text(encoding="utf-8"))
                data["solutions"].append({
                    "path": str(f),
                    "name": sol.get("name", ""),
                    "score": sol.get("score", None),
                })
            except Exception:
                pass

    # Find WM snapshots
    wm_dir = exp_dir / "world_model"
    if wm_dir.exists():
        for f in wm_dir.rglob("world_model.json"):
            try:
                wm = json.loads(f.read_text(encoding="utf-8"))
                data["wm_snapshots"].append({
                    "path": str(f),
                    "nodes_count": len(wm.get("decision_tree", {}).get("nodes", [])),
                    "best_score": wm.get("best_score", None),
                })
                if isinstance(wm.get("best_score"), (int, float)):
                    if data["best_score"] is None or float(wm["best_score"]) > data["best_score"]:
                        data["best_score"] = float(wm["best_score"])
            except Exception:
                pass

    return data


def main():
    root = Path(__file__).resolve().parent.parent
    experiments = find_experiment_dirs(root)

    if not experiments:
        print("No experiment directories found in", root)
        sys.exit(1)

    print(f"\nFound {len(experiments)} experiment directories:")
    for exp in experiments:
        print(f"  - {exp['name']} (type: {exp['type']})")

    print("\n--- Experiment Results Summary ---")
    print(f"{'Experiment':<25} {'Type':<20} {'Speedup':<12} {'Latency(ms)':<14} {'Best Score':<12}")
    print("-" * 83)

    all_results = {}
    for exp in experiments:
        data = collect_eval_data(exp["dir"])
        sp = data["final_speedup"]
        lat = data["final_latency_ms"]
        sc = data["best_score"]

        sp_str = f"{sp:.3f}x" if sp is not None else "N/A"
        lat_str = f"{lat:.4f}" if lat is not None else "N/A"
        sc_str = f"{sc:.3f}" if sc is not None else "N/A"

        print(f"{exp['name']:<25} {exp['type']:<20} {sp_str:<12} {lat_str:<14} {sc_str:<12}")
        all_results[exp["type"]] = data

    # Comparative analysis
    print("\n--- Comparative Analysis ---")
    baseline_sp = all_results.get("baseline_llm", {}).get("final_speedup")
    strategy_sps = {}
    for t in ("natural_language", "structured_params", "dsl"):
        strategy_sps[t] = all_results.get(t, {}).get("final_speedup")

    if baseline_sp is not None:
        print(f"\nBaseline (pure LLM WM): {baseline_sp:.3f}x speedup")
        for t, sp in strategy_sps.items():
            if sp is not None:
                improvement = (sp - baseline_sp) / baseline_sp * 100 if baseline_sp > 0 else 0
                print(f"  {t}: {sp:.3f}x speedup ({improvement:+.1f}% vs baseline)")
            else:
                print(f"  {t}: No results yet")

    # Best strategy form
    best_form = None
    best_sp = -1
    for t, sp in strategy_sps.items():
        if sp is not None and sp > best_sp:
            best_sp = sp
            best_form = t

    if best_form:
        print(f"\nBest strategy form: {best_form} ({best_sp:.3f}x speedup)")
    else:
        print("\nNo strategy experiments completed yet.")

    print("\n--- WM Node Statistics ---")
    for exp in experiments:
        data = all_results.get(exp["type"], {})
        wm_snaps = data.get("wm_snapshots", [])
        if wm_snaps:
            last_wm = wm_snaps[-1]
            print(f"  {exp['type']}: {last_wm['nodes_count']} nodes, best_score={last_wm['best_score']}")


if __name__ == "__main__":
    main()