import argparse
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Optional
import json


def _resolve_llm_config_from_args(args: Any) -> tuple[str, Optional[str]]:
    from k_search.kernel_generators.llm_clients import normalize_llm_provider

    llm_provider = normalize_llm_provider(getattr(args, "llm_provider", None))
    if llm_provider == "claude-agent":
        return llm_provider, getattr(args, "api_key", None)

    api_key = getattr(args, "api_key", None) or os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required for --llm-provider openai (pass --api-key or set LLM_API_KEY)"
        )
    return llm_provider, api_key

def _persist_ksearch_solution(
    solution: Any, *, definition_name: str, artifacts_dir: Optional[str], run_id: Optional[str] = None
) -> Optional[Path]:
    """
    Persist a k-search task_base.Solution JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir, get_run_id
    except Exception:
        return None
    try:
        from k_search.tasks.task_base import Solution as KSearchSolution
    except Exception:
        KSearchSolution = None  # type: ignore

    try:
        # Note: base_dir is provided by caller; default remains ./ .ksearch
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or ""), run_id=run_id or get_run_id()
        ).resolve()
        out_dir = root / "solutions" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(solution, "name", "") or "solution")
        # Sanitize the filename so model names like "org/model" don't become directories.
        safe_name = "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name]).strip("_")
        if not safe_name:
            safe_name = "solution"
        dest = out_dir / f"{safe_name}.json"
        if KSearchSolution is not None and isinstance(solution, KSearchSolution):
            obj = solution.to_dict()
        else:
            obj = solution.__dict__ if hasattr(solution, "__dict__") else {"solution": str(solution)}

        payload = json.dumps(obj, ensure_ascii=False, indent=2)
        dest.write_text(payload, encoding="utf-8")

        # Backward-compatibility: also persist under the original unsanitized
        # solution name so loaders that resolve by "<solution_ref>.json" keep
        # working for names that contain characters sanitized above.
        if name != safe_name:
            legacy_dest = (out_dir / f"{name}.json")
            legacy_dest_resolved = legacy_dest.resolve()
            try:
                legacy_dest_resolved.relative_to(out_dir.resolve())
            except ValueError:
                legacy_dest_resolved = None

            if legacy_dest_resolved is not None and legacy_dest_resolved != dest.resolve():
                legacy_dest_resolved.parent.mkdir(parents=True, exist_ok=True)
                legacy_dest_resolved.write_text(payload, encoding="utf-8")
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
    run_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Persist a final-eval report JSON under the k-search artifacts dir.
    """
    try:
        from k_search.utils.paths import get_ksearch_artifacts_dir, get_run_id
    except Exception:
        return None
    try:
        root = get_ksearch_artifacts_dir(
            base_dir=artifacts_dir, task_name=str(definition_name or ""), run_id=run_id or get_run_id()
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
    llm_provider: str,
    language: str,
    target_gpu: str,
    max_opt_rounds: int,
    save_solutions: bool,
    save_results: bool,
    continue_from_solution: Optional[str] = None,
    continue_from_world_model: Optional[str] = None,
    continue_from_run: Optional[str] = None,  # New: resume from historical run
    run_id: Optional[str] = None,  # New: explicit run_id override
    num_eval_workload: Optional[int] = None,
    # W&B options
    enable_wandb: bool = False,
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None,
    # World model prompting
    enable_world_model: bool = False,
    wm_stagnation_window: int = 5,
    wm_max_difficulty: Optional[int] = None,
    artifacts_dir: Optional[str] = None,
    # Strategy injection
    strategy_file: Optional[str] = None,
    strategy_form: Optional[str] = None,
) -> None:
    """
    Generate exactly one solution for the task, then run final evaluation.
    """
    from k_search.utils.paths import get_ksearch_artifacts_dir, get_run_id

    # Determine run_id
    if continue_from_run:
        effective_run_id = continue_from_run
    elif run_id:
        effective_run_id = run_id
    else:
        effective_run_id = get_run_id()

    task_name = str(getattr(task, "name", "") or "")

    # Write run_meta.json
    run_root = get_ksearch_artifacts_dir(
        base_dir=artifacts_dir,
        task_name=task_name,
        run_id=effective_run_id,
    )
    run_meta_path = run_root / "run_meta.json"
    run_meta = {
        "run_id": effective_run_id,
        "start_time": datetime.utcnow().isoformat() + "Z",
        "task_name": task_name,
        "model_name": model_name,
        "language": language,
        "target_gpu": target_gpu,
        "llm_provider": llm_provider,
        "max_opt_rounds": max_opt_rounds,
        "enable_world_model": enable_world_model,
        "wm_stagnation_window": wm_stagnation_window,
        "wm_max_difficulty": wm_max_difficulty,
        "continue_from_solution": continue_from_solution,
        "continue_from_world_model": continue_from_world_model,
        "continue_from_run": continue_from_run,
        "strategy_file": strategy_file,
        "strategy_form": strategy_form,
        "status": "running",
    }
    try:
        run_meta_path.parent.mkdir(parents=True, exist_ok=True)
        run_meta_path.write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[RUN] Run ID: {effective_run_id}")
        print(f"[RUN] Run metadata: {run_meta_path}")
    except Exception as e:
        print(f"[WARN] Failed to write run_meta.json: {e}")

    # Store run_id in task for downstream artifact and telemetry lineage.
    setattr(task, "_ksearch_run_id", effective_run_id)

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
                    "llm_provider": llm_provider,
                },
                "max_opt_rounds": int(max_opt_rounds),
                "continue_from_solution": continue_from_solution,
                "continue_from_world_model": continue_from_world_model,
                "enable_world_model": bool(enable_world_model),
                "wm_stagnation_window": int(wm_stagnation_window),
                "wm_max_difficulty": wm_max_difficulty,
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
                run_id=getattr(task, "_ksearch_run_id", None),
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
            wm_max_difficulty=wm_max_difficulty,
            llm_provider=llm_provider,
            strategy_file=strategy_file,
            strategy_form=strategy_form,
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
            llm_provider=llm_provider,
        )

    # Generate exactly one solution.
    if enable_world_model:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=int(wm_stagnation_window),
            continue_from_solution=continue_from_solution,
            continue_from_world_model=continue_from_world_model,
            continue_from_run=continue_from_run,
            run_id=effective_run_id,
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


def _build_task_from_args(args: Any) -> Any:
    task_source = str(getattr(args, "task_source", None) or "flashinfer")
    task_path = str(getattr(args, "task_path", None) or (getattr(args, "local", None) or ""))
    default_language = "ascendc" if task_source == "ascendc" else "triton"
    language = str(getattr(args, "language", default_language) or default_language).strip().lower()

    if task_source == "flashinfer":
        from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask

        if language in {"mlx", "ascendc"}:
            raise ValueError(f"--language {language} is not supported with --task-source=flashinfer")

        if not task_path:
            raise ValueError("--local or --task-path is required for --task-source=flashinfer")
        if not getattr(args, "definition", None):
            raise ValueError("--definition is required")
        def_name = str(args.definition)

        return FlashInferBenchTask.from_cli_args(
            task_path=task_path,
            definition_name=str(def_name),
            warmup_runs=getattr(args, "warmup_runs", 10),
            iterations=getattr(args, "iterations", 10),
            num_trials=getattr(args, "num_trials", 1),
            rtol=getattr(args, "rtol", 1e-2),
            atol=getattr(args, "atol", 1e-2),
            use_isolated_runner=bool(getattr(args, "use_isolated_runner", False)),
            parallel_workloads=bool(getattr(args, "parallel_workloads", False)),
            max_parallel_workloads=getattr(args, "max_parallel_workloads", 0),
            baseline_solution=getattr(args, "baseline_solution", None),
            feedback_workloads=getattr(args, "feedback_workloads", None),
            feedback_trace_policy=getattr(args, "feedback_trace_policy", "first"),
            num_feedback_workloads=5,
            artifacts_dir=getattr(args, "artifacts_dir", None),
        )
    if task_source == "gpumode":
        from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

        if language in {"mlx", "ascendc"}:
            raise ValueError(f"--language {language} is not supported with --task-source=gpumode")

        return GpuModeTriMulTask(
            mode=str(getattr(args, "gpumode_mode", None) or "benchmark"),
            keep_tmp=bool(getattr(args, "gpumode_keep_tmp", False)),
            task_dir=(str(getattr(args, "gpumode_task_dir", "")) if getattr(args, "gpumode_task_dir", None) else None),
            artifacts_dir=getattr(args, "artifacts_dir", None),
        )
    if task_source == "kernelbench":
        from k_search.tasks.kernelbench_task import KernelBenchTask

        if language in {"mlx", "ascendc"}:
            raise ValueError(f"--language {language} is not supported with --task-source=kernelbench")

        return KernelBenchTask(
            level=getattr(args, "kernelbench_level", 1),
            problem_id=getattr(args, "kernelbench_problem_id", 1),
            eval_mode=getattr(args, "kernelbench_eval_mode", "local"),
            gpu=getattr(args, "target_gpu", "H100"),
            num_correct_trials=getattr(args, "kernelbench_num_correct_trials", 5),
            num_perf_trials=getattr(args, "kernelbench_num_perf_trials", 100),
            artifacts_dir=getattr(args, "artifacts_dir", None),
            backend=language,
        )
    if task_source == "mlx":
        def_name = str(getattr(args, "definition", None) or "mlx_mamba_selective_scan_fwd").strip()
        if def_name in (
            "mlx_mamba",
            "mlx_mamba_selective_scan_fwd",
        ):
            from k_search.tasks.mlx_mamba_task import MlxMambaSelectiveScanFwdTask

            return MlxMambaSelectiveScanFwdTask(
                warmup_runs=getattr(args, "warmup_runs", 10),
                iterations=getattr(args, "iterations", 10),
                rtol=getattr(args, "rtol", 1e-2),
                atol=getattr(args, "atol", 1e-2),
                timeout_seconds=300,
                artifacts_dir=getattr(args, "artifacts_dir", None),
                name="mlx_mamba_selective_scan_fwd",
            )
        raise ValueError(
            "Unknown MLX definition. Use --definition one of: "
            "mlx_mamba_selective_scan_fwd. "
            f"Got {def_name!r}."
        )
    if task_source == "ascendc":
        from k_search.tasks.ascendc_task import AscendCTask

        if language != "ascendc":
            raise ValueError("--task-source=ascendc requires --language=ascendc")
        if not task_path:
            raise ValueError("--task-path is required for --task-source=ascendc")
        return AscendCTask(
            task_path=task_path,
            definition_name=(str(getattr(args, "definition", "") or "") or None),
            build_cmd=getattr(args, "ascendc_build_cmd", None),
            test_cmd=getattr(args, "ascendc_test_cmd", None),
            bench_cmd=getattr(args, "ascendc_bench_cmd", None),
            timeout_seconds=int(getattr(args, "ascendc_timeout_seconds", 600) or 600),
            reference_latency_ms=getattr(args, "ascendc_reference_latency_ms", None),
            artifacts_dir=getattr(args, "artifacts_dir", None),
            codegen_mode=getattr(args, "ascendc_codegen_mode", None),
        )
    raise ValueError(f"Unsupported task_source: {task_source}")


def main():
    parser = argparse.ArgumentParser(description="Generate kernels with GPT/Gemini (OpenAI-compatible) and evaluate via task backends.")
    parser.add_argument("--local", required=False, default=None, help="Path to flashinfer-trace dataset root (flashinfer only)")
    parser.add_argument(
        "--task-source",
        choices=["flashinfer", "gpumode", "kernelbench", "mlx", "ascendc"],
        default="flashinfer",
        help="Task backend to use.",
    )
    parser.add_argument(
        "--task-path",
        default=None,
        help="Task source path/identifier. For --task-source=flashinfer, this is the dataset root path (defaults to --local).",
    )
    parser.add_argument("--definition", default=None, help="Single definition name to target (required)")
    parser.add_argument("--model-name", required=True, help="LLM model name (e.g., gpt-4.1, gpt-5, gemini-2.5-pro via compatible endpoint)")
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "claude-agent"],
        help="LLM backend: openai for OpenAI-compatible APIs, claude-agent for Claude Agent SDK.",
    )
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for non-OpenAI providers (e.g. Gemini proxy)")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the OpenAI-compatible provider; if omitted, uses LLM_API_KEY env var.",
    )
    parser.add_argument(
        "--language",
        default="triton",
        choices=["triton", "python", "cuda", "mlx", "ascendc"],
        help="Target language for generated kernel",
    )
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
    parser.add_argument(
        "--continue-from-world-model",
        default=None,
        help=(
            "Resume world-model prompting state from a JSON file path. "
            "Use 'auto' to load <artifacts>/<task>/world_model/world_model.json if present."
        ),
    )
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
        "--wm-max-difficulty",
        type=int,
        default=None,
        help="World-model mode: max difficulty (1-5) for action selection. Actions above this are deferred. Default: use policy default (4).",
    )
    # W&B options
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT"), help="W&B project")
    parser.add_argument("--run-name", default=os.getenv("RUN_NAME"), help="W&B run name")

    # GPUMode options
    parser.add_argument("--gpumode-mode", default="benchmark", help="GPUMode eval mode (e.g., benchmark/test/leaderboard/profile)")
    parser.add_argument("--gpumode-keep-tmp", action="store_true", help="Keep GPUMode temp working dir for debugging")
    parser.add_argument("--gpumode-task-dir", default=None, help="Override GPUMode task dir (defaults to vendored trimul task)")

    # KernelBench options
    parser.add_argument("--kernelbench-level", type=int, default=1, help="KernelBench level (1, 2, or 3)")
    parser.add_argument("--kernelbench-problem-id", type=int, default=1, help="Problem ID within the level")
    parser.add_argument("--kernelbench-eval-mode", default="local", choices=["local", "modal"], help="Evaluation mode")
    parser.add_argument("--kernelbench-num-correct-trials", type=int, default=5, help="Number of correctness trials")
    parser.add_argument("--kernelbench-num-perf-trials", type=int, default=100, help="Number of performance trials")

    # AscendC options
    parser.add_argument("--ascendc-build-cmd", default=None, help="Shell command that compiles the AscendC candidate in the candidate project root")
    parser.add_argument("--ascendc-test-cmd", default=None, help="Shell command that validates AscendC correctness in the candidate project root")
    parser.add_argument("--ascendc-bench-cmd", default=None, help="Shell command that benchmarks the AscendC candidate and prints latency_ms=<float>")
    parser.add_argument("--ascendc-timeout-seconds", type=int, default=600, help="Timeout per AscendC build/test/bench command")
    parser.add_argument("--ascendc-reference-latency-ms", type=float, default=None, help="Optional baseline latency used to score speedup")
    parser.add_argument(
        "--ascendc-codegen-mode",
        choices=["auto", "full", "patch"],
        default=None,
        help=(
            "AscendC codegen response format. 'auto' (default) emits patches when a baseline "
            "is available and full containers otherwise. 'full' forces the legacy full "
            "multi-file container every round (regression-safe). 'patch' forces unified-diff "
            "responses. Falls back to env var KSEARCH_ASCENDC_CODEGEN_MODE."
        ),
    )

    # Strategy injection options
    parser.add_argument(
        "--strategy-file",
        default=None,
        help=(
            "Path to a strategy catalog JSON file. When provided alongside --world-model, "
            "the WM decision tree is seeded with strategy-derived action nodes instead of "
            "LLM-generated ones. Enables controlled strategy-form experiments."
        ),
    )
    parser.add_argument(
        "--strategy-form",
        choices=["natural_language", "structured_params", "dsl"],
        default=None,
        help=(
            "Strategy rendering form for action text injection. "
            "'natural_language' renders plain English descriptions. "
            "'structured_params' renders JSON parameter specifications. "
            "'dsl' renders domain-specific language specifications. "
            "Must be used with --strategy-file."
        ),
    )

    args = parser.parse_args()

    # Pin a single output base + run id for the whole process so that llm logs,
    # telemetry and the narrative summary all land under the same
    # <base>/logs/<task>/<run_id>/ tree (and never drift apart across calls).
    from k_search.utils.paths import get_run_id, resolve_output_base

    os.environ.setdefault(
        "KSEARCH_ARTIFACTS_DIR", str(resolve_output_base(getattr(args, "artifacts_dir", None)))
    )
    os.environ.setdefault("KSEARCH_RUN_ID", get_run_id())

    # MLX runs on Apple Silicon; the CUDA-style --target-gpu hint is not meaningful.
    # If Metal is available, replace it with an auto-detected device name
    if str(getattr(args, "task_source", "")).strip().lower() == "mlx":
        try:
            from k_search.utils.metal_gpu_info import get_metal_device_name

            detected = get_metal_device_name().strip()
            if detected:
                args.target_gpu = detected
            else:
                args.target_gpu = "AppleSilicon"
        except Exception:
            args.target_gpu = "AppleSilicon"
    if str(getattr(args, "task_source", "")).strip().lower() == "ascendc":
        if str(getattr(args, "target_gpu", "") or "").strip() == "H100":
            args.target_gpu = "ascend_910b"
        if str(getattr(args, "language", "") or "").strip().lower() == "triton":
            args.language = "ascendc"

    llm_provider, api_key = _resolve_llm_config_from_args(args)

    task = _build_task_from_args(args)

    generate_and_evaluate(
        task=task,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=api_key,
        llm_provider=llm_provider,
        language=args.language,
        target_gpu=args.target_gpu,
        max_opt_rounds=args.max_opt_rounds,
        save_solutions=args.save_solutions,
        save_results=not args.no_save_results,
        num_eval_workload=args.num_eval_workload,
        continue_from_solution=args.continue_from_solution,
        continue_from_world_model=args.continue_from_world_model,
        enable_world_model=args.world_model,
        wm_stagnation_window=args.wm_stagnation_window,
        wm_max_difficulty=args.wm_max_difficulty,
        artifacts_dir=args.artifacts_dir,
        enable_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        strategy_file=args.strategy_file,
        strategy_form=args.strategy_form,
    )


if __name__ == "__main__":
    main()
