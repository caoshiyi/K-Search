import argparse
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, List, Optional
import json

def _persist_ksearch_solution(
    solution: Any, *, definition_name: str, artifacts_dir: Optional[str]
) -> Optional[Path]:
    """
    Persist a k-search task_base.Solution JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir
    except Exception:
        return None
    try:
        from k_search.tasks.task_base import Solution as KSearchSolution
    except Exception:
        KSearchSolution = None  # type: ignore

    try:
        # Note: base_dir is provided by caller; default remains ./ .ksearch
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or "")
        ).resolve()
        out_dir = root / "solutions" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(solution, "name", "") or "solution")
        dest = out_dir / f"{name}.json"
        if KSearchSolution is not None and isinstance(solution, KSearchSolution):
            obj = solution.to_dict()
        else:
            obj = solution.__dict__ if hasattr(solution, "__dict__") else {"solution": str(solution)}
        dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving k-search solution: {e}")
        import traceback
        traceback.print_exc()
        return None


def _persist_ksearch_eval_report(
    report: dict[str, Any],
    *,
    definition_name: str,
    solution_name: Optional[str],
    artifacts_dir: Optional[str],
) -> Optional[Path]:
    """
    Persist a final-eval report JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir
    except Exception:
        return None
    try:
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or "")
        ).resolve()
        out_dir = root / "eval" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sn = str(solution_name or "").strip()
        safe_sn = "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in sn]) if sn else ""
        suffix = f"_{safe_sn}" if safe_sn else ""
        dest = out_dir / f"eval_report_{ts}{suffix}.json"
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    except Exception as e:
        print(f"Error saving eval report: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_and_evaluate(
    task: Any,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    language: str,
    target_gpu: str,
    max_opt_rounds: int,
    save_solutions: bool,
    save_results: bool,
    continue_from_solution: Optional[str] = None,
    num_eval_workload: Optional[int] = None,
    # W&B options
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    # World model prompting
    enable_world_model: bool = False,
    wm_stagnation_window: int = 5,
    artifacts_dir: Optional[str] = None,
) -> None:
    """
    Generate exactly one solution for the task, then run final evaluation.
    """
    
    # Generator
    from k_search.kernel_generators.kernel_generator import KernelGenerator
    # Optional Weights & Biases support
    try:
        import wandb  # type: ignore
    except Exception:  # pragma: no cover
        wandb = None

    # Initialize wandb if enabled
    wb_run = None
    if enable_wandb and wandb is not None:
        print(f"Initializing wandb with project: {wandb_project} and name: {run_name}")
        try:
            task_cfg = task.get_config_for_logging()
        except Exception:
            task_cfg = {}
        wb_run = wandb.init(
            project=wandb_project or os.getenv("WANDB_PROJECT", "flashinfer-bench"),
            name=run_name or os.getenv("RUN_NAME"),
            config={
                "task": task_cfg,
                "generator": {
                    "model_name": model_name,
                    "language": language,
                    "target_gpu": target_gpu,
                },
                "max_opt_rounds": int(max_opt_rounds),
                "continue_from_solution": continue_from_solution,
                "enable_world_model": bool(enable_world_model),
                "wm_stagnation_window": int(wm_stagnation_window),
                "save_results": bool(save_results),
                "save_solutions": bool(save_solutions),
                "num_eval_workload": num_eval_workload,
                "artifacts_dir": artifacts_dir,
            },
            reinit=True,
        )

    def _eval_and_report_one(*, sol: Any) -> None:
        def_name = str(getattr(task, "name", "") or "")
        sol_name = str(getattr(sol, "name", "") or "")

        report = task.run_final_evaluation(
            solutions=[sol],
            config=None,
            dump_traces=bool(save_results),
            workload_limit=num_eval_workload,
        )
        if save_results:
            saved = _persist_ksearch_eval_report(
                report,
                definition_name=def_name,
                solution_name=sol_name,
                artifacts_dir=artifacts_dir,
            )
            if saved:
                print(f"[{def_name}] Saved eval report to: {saved}")

    if enable_world_model:
        # World-model mode uses the WM generator (task-driven).
        from k_search.kernel_generators.kernel_generator_world_model import WorldModelKernelGeneratorWithBaseline

        generator = WorldModelKernelGeneratorWithBaseline(
            model_name=model_name,
            language=language,
            target_gpu=target_gpu,
            api_key=api_key,
            base_url=base_url,
            artifacts_dir=artifacts_dir,
        )
    else:
        # Non-world-model mode: baseline-style generator (task-driven).
        from k_search.kernel_generators.kernel_generator import KernelGenerator

        generator = KernelGenerator(
            model_name=model_name,
            language=language,
            target_gpu=target_gpu,
            api_key=api_key,
            base_url=base_url,
        )

    # Generate exactly one solution.
    if enable_world_model:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=int(wm_stagnation_window),
            continue_from_solution=continue_from_solution,
        )
    else:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            continue_from_solution=continue_from_solution,
        )

    # Append timestamp and uid to ensure uniqueness and traceability
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    solution.name = f"{solution.name}_{ts}_{uid}"
    # Optional: reflect in description
    try:
        solution.description = (solution.description or "") + f" (generated {ts} uid={uid})"
    except Exception:
        pass

    # Optionally persist to disk (k-search solution type)
    if save_solutions:
        saved_path = _persist_ksearch_solution(
            solution, definition_name=str(getattr(task, "name", "") or ""), artifacts_dir=artifacts_dir
        )
        if saved_path:
            print(f"  ✓ Saved solution to: {saved_path}")
        else:
            print(f"  ✗ Failed to save solution")

    def_name = str(getattr(task, "name", "") or "")
    print(f"[{def_name}] Generated solution: {solution.name}")

    # Final eval: evaluate ONLY the solution(s) returned by the generator, one at a time.
    # This keeps the logic simple and avoids comparing multiple generated solutions in one report.
    _eval_and_report_one(sol=solution)

    # Cleanly close W&B run if it was opened (prevents BrokenPipe in Ray workers)
    if wb_run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Generate kernels with GPT/Gemini (OpenAI-compatible) and evaluate via flashinfer-bench.")
    parser.add_argument("--local", required=True, help="Path to flashinfer-trace dataset root")
    parser.add_argument(
        "--task-source",
        choices=["flashinfer"],
        default="flashinfer",
        help="Task backend to use (currently only: flashinfer).",
    )
    parser.add_argument(
        "--task-path",
        default=None,
        help="Task source path/identifier. For --task-source=flashinfer, this is the dataset root path (defaults to --local).",
    )
    parser.add_argument("--definition", default=None, help="Single definition name to target (required)")
    parser.add_argument("--model-name", required=True, help="LLM model name (e.g., gpt-4.1, gpt-5, gemini-2.5-pro via compatible endpoint)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for non-OpenAI providers (e.g. Gemini proxy)")
    parser.add_argument("--api-key", default=None, help="API key; if omitted, uses LLM_API_KEY env var")
    parser.add_argument("--language", default="triton", choices=["triton", "python", "cuda"], help="Target language for generated kernel")
    parser.add_argument("--target-gpu", default="H100", help="Target GPU architecture hint for prompts")
    parser.add_argument("--max-opt-rounds", type=int, default=5, help="Max optimization rounds for each solution generation")

    # Benchmark configuration
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--use-isolated-runner", action="store_true")
    parser.add_argument(
        "--parallel-workloads",
        action="store_true",
        help="Enable workload-parallel scheduling in Benchmark (useful when evaluating only a small number of solutions).",
    )
    parser.add_argument(
        "--max-parallel-workloads",
        type=int,
        default=0,
        help="Max concurrent workloads when --parallel-workloads is enabled (0 = auto based on visible CUDA devices).",
    )
    parser.add_argument("--no-save-results", action="store_true", help="Do not write traces to dataset")
    parser.add_argument(
        "--save-solutions",
        action="store_true",
        help="Persist generated solutions JSON into the k-search artifacts dir (see --artifacts-dir)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=".ksearch",
        help="Base directory for k-search artifacts (solutions, world model snapshots, eval reports).",
    )
    parser.add_argument("--baseline-solution", default=None, help="Optional baseline solution name to compare against; if absent, 'vs_base' is omitted")
    parser.add_argument("--num-eval-workload", type=int, default=None, help="If set, evaluate only this many workloads per definition; default uses all workloads")
    # Continue optimization options
    parser.add_argument("--continue-from-solution", default=None, help="Resume optimization from an existing solution name in the dataset")
    parser.add_argument("--feedback-workloads", nargs="+", default=None, help="Explicit workload UUIDs to use for optimization feedback rounds")
    # Nsight Compute
    parser.add_argument("--feedback-trace-policy", default="first", choices=["first", "random"], help="Policy for selecting feedback traces")
    parser.add_argument(
        "--world-model",
        action="store_true",
        help="Enable world-model prompting (maintain a persistent world model across rounds and inject it into prompts).",
    )
    parser.add_argument(
        "--wm-stagnation-window",
        type=int,
        default=5,
        help="World-model mode: end an action cycle after this many consecutive non-improving rounds (>=1).",
    )
    parser.add_argument(
        "--per-workload-feedback",
        action="store_true",
        help="Include per-workload performance feedback in optimization prompts (more verbose, can help targeted tuning)",
    )
    # W&B options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"), help="W&B project")
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME"), help="W&B run name")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("API key is required (pass --api-key or set LLM_API_KEY)")

    task_source = str(args.task_source or "flashinfer")
    task_path = str(args.task_path or args.local)
    if task_source == "flashinfer":
        from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask

        if not args.definition:
            raise ValueError("--definition is required")
        def_name = str(args.definition)

        task = FlashInferBenchTask.from_cli_args(
            task_path=task_path,
            definition_name=str(def_name),
            warmup_runs=args.warmup_runs,
            iterations=args.iterations,
            num_trials=args.num_trials,
            rtol=args.rtol,
            atol=args.atol,
            use_isolated_runner=args.use_isolated_runner,
            parallel_workloads=args.parallel_workloads,
            max_parallel_workloads=args.max_parallel_workloads,
            baseline_solution=args.baseline_solution,
            feedback_workloads=args.feedback_workloads,
            feedback_trace_policy=args.feedback_trace_policy,
            num_feedback_workloads=5,
        )
    else:
        raise ValueError(f"Unsupported task_source: {task_source}")
    generate_and_evaluate(
        task=task,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=api_key,
        language=args.language,
        target_gpu=args.target_gpu,
        max_opt_rounds=args.max_opt_rounds,
        save_solutions=args.save_solutions,
        save_results=not args.no_save_results,
        num_eval_workload=args.num_eval_workload,
        continue_from_solution=args.continue_from_solution,
        enable_world_model=args.world_model,
        wm_stagnation_window=args.wm_stagnation_window,
        artifacts_dir=args.artifacts_dir,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()


