import argparse
import os
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
# Maps --provider names to (base_url, env_var_for_api_key, default_model).
# base_url=None means use the SDK's default (OpenAI, Anthropic).
# api_style controls which SDK / calling convention to use:
#   "anthropic"         → Anthropic Messages API (anthropic SDK)
#   "openai_responses"  → OpenAI Responses API  (openai SDK, for reasoning models)
#   "openai_chat"       → OpenAI Chat Completions API (openai SDK, works with all compatible providers)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.2",
        "api_style": "openai_responses",
    },
    "anthropic": {
        "base_url": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-6",
        "api_style": "anthropic",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-3-pro-preview",
        "api_style": "openai_chat",
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "default_model": "grok-3",
        "api_style": "openai_chat",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "api_style": "openai_chat",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4",
        "api_style": "openai_chat",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "default_model": "qwen-max",
        "api_style": "openai_chat",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-128k",
        "api_style": "openai_chat",
    },
}

VALID_API_STYLES = {"anthropic", "openai_responses", "openai_chat"}


def _load_env_file(path: str = ".env.local") -> None:
    """Load key=value pairs from a .env file into os.environ (does not override existing vars)."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_provider(
    *,
    provider: Optional[str],
    model_name: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    api_style: Optional[str],
) -> Tuple[str, str, Optional[str], str]:
    """Resolve (model_name, api_key, base_url, api_style) from provider + explicit overrides.

    Priority: explicit flags > provider defaults > model-name inference > env fallback.
    """
    resolved_api_style = api_style  # may be None; resolved below

    if provider:
        p = provider.lower()
        if p not in PROVIDERS:
            available = ", ".join(sorted(PROVIDERS.keys()))
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
        prov = PROVIDERS[p]
        model_name = model_name or prov["default_model"]
        base_url = base_url or prov["base_url"]
        api_key = api_key or os.getenv(prov["api_key_env"]) or os.getenv("LLM_API_KEY")
        resolved_api_style = resolved_api_style or prov["api_style"]
    else:
        # No provider: try to infer from model name prefix
        if model_name and not api_key:
            if model_name.startswith("claude"):
                api_key = os.getenv("ANTHROPIC_API_KEY")
                resolved_api_style = resolved_api_style or "anthropic"
            elif model_name.startswith("gemini"):
                api_key = os.getenv("GOOGLE_API_KEY")
                base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/"
            elif model_name.startswith("grok"):
                api_key = os.getenv("XAI_API_KEY")
                base_url = base_url or "https://api.x.ai/v1"
            elif model_name.startswith("deepseek"):
                api_key = os.getenv("DEEPSEEK_API_KEY")
                base_url = base_url or "https://api.deepseek.com"
        api_key = api_key or os.getenv("LLM_API_KEY")

    # Final fallback
    resolved_api_style = resolved_api_style or "openai_chat"

    if resolved_api_style not in VALID_API_STYLES:
        raise ValueError(
            f"Unknown --api-style '{resolved_api_style}'. "
            f"Valid: {', '.join(sorted(VALID_API_STYLES))}"
        )
    if not model_name:
        raise ValueError(
            "Model name is required. Pass --model-name or --provider to use a default."
        )
    if not api_key:
        hint = f" ({PROVIDERS[provider.lower()]['api_key_env']})" if provider else ""
        raise ValueError(
            f"API key is required. Pass --api-key, set the provider env var{hint}, or set LLM_API_KEY."
        )
    return model_name, api_key, base_url, resolved_api_style

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
    continue_from_world_model: Optional[str] = None,
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
    api_style: str = "openai_chat",
) -> None:
    """
    Generate exactly one solution for the task, then run final evaluation.
    """
    
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
            api_style=api_style,
            artifacts_dir=artifacts_dir,
            wm_max_difficulty=wm_max_difficulty,
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
            api_style=api_style,
        )

    # Generate exactly one solution.
    if enable_world_model:
        solution = generator.generate(
            task=task,
            max_opt_rounds=max_opt_rounds,
            wm_stagnation_window=int(wm_stagnation_window),
            continue_from_solution=continue_from_solution,
            continue_from_world_model=continue_from_world_model,
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
    provider_names = ", ".join(sorted(PROVIDERS.keys()))
    parser = argparse.ArgumentParser(description="Generate kernels with LLMs and evaluate via task backends.")
    parser.add_argument("--local", required=False, default=None, help="Path to flashinfer-trace dataset root (flashinfer only)")
    parser.add_argument(
        "--task-source",
        choices=["flashinfer", "gpumode"],
        default="flashinfer",
        help="Task backend to use.",
    )
    parser.add_argument(
        "--task-path",
        default=None,
        help="Task source path/identifier. For --task-source=flashinfer, this is the dataset root path (defaults to --local).",
    )
    parser.add_argument("--definition", default=None, help="Single definition name to target (required)")
    parser.add_argument(
        "--provider",
        default=None,
        help=f"LLM provider — auto-sets base URL and API key env var. Available: {provider_names}",
    )
    parser.add_argument("--model-name", default=None, help="LLM model name (e.g., claude-sonnet-4-6, gpt-5.2, gemini-3-pro-preview). If --provider is set and this is omitted, uses the provider default.")
    parser.add_argument("--base-url", default=None, help="Override base URL (auto-set by --provider if not given)")
    parser.add_argument("--api-key", default=None, help="API key; if omitted, resolved from --provider env var or LLM_API_KEY")
    parser.add_argument("--env-file", default=".env.local", help="Path to .env file to load (default: .env.local)")
    parser.add_argument(
        "--api-style",
        default=None,
        choices=sorted(VALID_API_STYLES),
        help="Override the API calling convention (default: auto from --provider). "
             "anthropic=Anthropic Messages API, openai_responses=OpenAI Responses API, "
             "openai_chat=OpenAI Chat Completions API (works with all compatible providers).",
    )
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

    args = parser.parse_args()

    # Load .env file before resolving provider keys
    _load_env_file(args.env_file)

    model_name, api_key, base_url, api_style = resolve_provider(
        provider=args.provider,
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        api_style=args.api_style,
    )
    print(f"[config] model={model_name} provider={args.provider or '(auto)'} api_style={api_style} base_url={base_url or '(default)'}")

    task_source = str(args.task_source or "flashinfer")
    task_path = str(args.task_path or (args.local or ""))
    if task_source == "flashinfer":
        from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask

        if not task_path:
            raise ValueError("--local or --task-path is required for --task-source=flashinfer")
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
            artifacts_dir=args.artifacts_dir,
        )
    elif task_source == "gpumode":
        from k_search.tasks.gpu_mode_task import GpuModeTriMulTask

        task = GpuModeTriMulTask(
            mode=str(args.gpumode_mode or "benchmark"),
            keep_tmp=bool(args.gpumode_keep_tmp),
            task_dir=(str(args.gpumode_task_dir) if args.gpumode_task_dir else None),
            artifacts_dir=args.artifacts_dir,
        )
    else:
        raise ValueError(f"Unsupported task_source: {task_source}")
    generate_and_evaluate(
        task=task,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
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
        api_style=api_style,
    )


if __name__ == "__main__":
    main()


