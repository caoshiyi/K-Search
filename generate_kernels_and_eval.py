import argparse
import os
import sys
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Optional
import json

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
_ENV_ALLOWLIST = {
    "KSEARCH_ARTIFACTS_DIR",
    "KSEARCH_TASK_ID",
    "KSEARCH_TASK_START",
    "KSEARCH_RUN_ID",
    "KSEARCH_RUN_START",
    "KSEARCH_TELEMETRY",
    "KSEARCH_TELEMETRY_DIR",
    "KSEARCH_LLM_LOG_DIR",
    "KSEARCH_LLM_LOG_JSON",
    "KSEARCH_LLM_TIMEOUT_SECONDS",
    "KSEARCH_ENABLE_CODE_MAP",
    "KSEARCH_ENABLE_CURATOR",
    "KSEARCH_KEEP_EVAL_WORKDIRS",
    "KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW",
    "KSEARCH_USE_AGENT_TOOL_ALLOWLIST",
    "KSEARCH_ALLOW_UNKNOWN_AGENT_TOOL_INPUT",
    "KSEARCH_STRICT_HANDOFF_VALIDATION",
    "KSEARCH_REVIEW_FEEDBACK_RETRY_ROUNDS",
    "CLAUDE_AGENT_MAX_TURNS",
    "CLAUDE_AGENT_THINKING",
    "CLAUDE_AGENT_TIMEOUT_SECONDS",
    "API_TIMEOUT_MS",
    "WANDB_PROJECT",
    "RUN_NAME",
}


def _is_secret_key(key: str) -> bool:
    upper = str(key or "").upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _redact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[str(k)] = (
                "<redacted:present>"
                if _is_secret_key(str(k)) and v not in (None, "")
                else _redact_mapping(v)
            )
        return out
    if isinstance(value, (list, tuple)):
        return [_redact_mapping(v) for v in value]
    return value


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _capture_env_payload() -> dict[str, Any]:
    env = {}
    redacted_present = []
    for key in sorted(
        set(_ENV_ALLOWLIST) | {k for k in os.environ if _is_secret_key(k)}
    ):
        if key not in os.environ:
            continue
        value = os.environ.get(key)
        if _is_secret_key(key):
            if value:
                redacted_present.append(key)
            continue
        env[key] = value
    return {
        "schema_version": 1,
        "captured_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "policy": "allowlist_with_secret_redaction",
        "env": env,
        "redacted_present": sorted(redacted_present),
    }


def _serialize_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _update_run_meta(path: Path, **updates: Any) -> None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not isinstance(meta, dict):
            meta = {}
        meta.update({k: v for k, v in updates.items() if v is not None})
        _write_json_file(path, meta)
    except Exception as e:
        print(f"[WARN] Failed to update run_meta.json: {e}")


def _parse_strategy_form(value: str) -> str:
    form = str(value or "").strip().lower()
    if form != "natural_language":
        raise argparse.ArgumentTypeError(
            "Only natural_language strategy form is supported. "
            "Use markdown_ref in strategy catalog."
        )
    return form


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


def _checkpoint_requested(args: Any) -> bool:
    return bool(
        getattr(args, "checkpoint_enable", False)
        or getattr(args, "checkpoint_v3", False)
        or getattr(args, "resume_from_checkpoint", None)
    )


def _validate_checkpoint_args(args: Any, *, llm_provider: str) -> None:
    if not _checkpoint_requested(args):
        return
    if not bool(getattr(args, "world_model", False)):
        raise ValueError("checkpoint requires --world-model")
    if str(llm_provider or "").strip() != "claude-agent":
        raise ValueError(
            "checkpoint currently supports only --llm-provider claude-agent"
        )
    if str(getattr(args, "language", "") or "").strip().lower() != "ascendc":
        raise ValueError("checkpoint currently supports only --language ascendc")
    if str(getattr(args, "task_source", "") or "").strip().lower() != "ascendc":
        raise ValueError("checkpoint currently supports only --task-source ascendc")


def _build_checkpoint_config_from_args(args: Any) -> Any:
    from k_search.kernel_generators.checkpoint import CheckpointConfig

    v3_enabled = bool(getattr(args, "checkpoint_v3", False))
    return CheckpointConfig(
        enabled=bool(getattr(args, "checkpoint_enable", False)),
        every=str(getattr(args, "checkpoint_every", "cycle") or "cycle"),
        checkpoint_dir=getattr(args, "checkpoint_dir", None),
        keep=int(getattr(args, "checkpoint_keep", 5)),
        resume_from=(
            None if v3_enabled else getattr(args, "resume_from_checkpoint", None)
        ),
        resume_mode=str(getattr(args, "resume_mode", "new-run") or "new-run"),
        resume_policy=str(getattr(args, "resume_policy", "latest") or "latest"),
        include_project_snapshot_payload=bool(
            getattr(args, "checkpoint_include_project_snapshot", True)
        ),
        enable_claude_file_checkpointing=bool(
            getattr(args, "checkpoint_enable_claude_file_checkpointing", False)
        ),
        resume_claude_session=bool(
            getattr(args, "checkpoint_resume_claude_session", False)
        ),
        retry_failed_attempt=bool(
            getattr(args, "checkpoint_retry_failed_attempt", False)
        ),
    )


def _stage_checkpoint_config_from_args(args: Any, *, llm_provider: str) -> Any:
    from k_search.kernel_generators.checkpoint_v3 import (
        StageCheckpointConfig,
        validate_stage_checkpoint_config,
    )

    v3_enabled = bool(getattr(args, "checkpoint_v3", False))
    v3_resume_from = (
        str(getattr(args, "resume_from_checkpoint", "") or "").strip() or None
        if v3_enabled
        else None
    )
    config = StageCheckpointConfig(
        enabled=v3_enabled,
        resume_from=v3_resume_from,
        save_stage_start=bool(getattr(args, "checkpoint_stage_start", True)),
        save_stage_completed=bool(getattr(args, "checkpoint_stage_boundary", True)),
        resume_claude_session=bool(
            getattr(args, "checkpoint_resume_claude_session", False)
        ),
        claude_session_required=bool(
            getattr(args, "checkpoint_claude_session_required", False)
        ),
        subagent_resume=bool(getattr(args, "checkpoint_subagent_resume", False)),
        require_subagent_agent_id=bool(
            getattr(args, "checkpoint_require_subagent_agent_id", False)
        ),
        enable_claude_file_checkpointing=bool(
            getattr(args, "checkpoint_enable_claude_file_checkpointing", False)
        ),
        session_store_kind=str(
            getattr(args, "checkpoint_session_store_kind", "none") or "none"
        ),
        session_store_config=getattr(args, "checkpoint_session_store_config", None),
        resume_stage_policy=getattr(
            args, "checkpoint_resume_stage_policy", "next-pending"
        ),
        file_state_source=getattr(
            args, "checkpoint_file_state_source", "project-snapshot"
        ),
    )
    validate_stage_checkpoint_config(
        config,
        task_source=str(getattr(args, "task_source", "") or ""),
        language=str(getattr(args, "language", "") or ""),
        llm_provider=llm_provider,
        world_model=bool(getattr(args, "world_model", False)),
    )
    return config


def _persist_ksearch_solution(
    solution: Any,
    *,
    definition_name: str,
    artifacts_dir: Optional[str],
    run_id: Optional[str] = None,
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
            base_dir=artifacts_dir,
            task_name=str(definition_name or ""),
            run_id=run_id or get_run_id(),
        ).resolve()
        out_dir = root / "solutions" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(solution, "name", "") or "solution")
        # Sanitize the filename so model names like "org/model" don't become directories.
        safe_name = "".join(
            [c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in name]
        ).strip("_")
        if not safe_name:
            safe_name = "solution"
        dest = out_dir / f"{safe_name}.json"
        if KSearchSolution is not None and isinstance(solution, KSearchSolution):
            obj = solution.to_dict()
        else:
            obj = (
                solution.__dict__
                if hasattr(solution, "__dict__")
                else {"solution": str(solution)}
            )

        payload = json.dumps(obj, ensure_ascii=False, indent=2)
        dest.write_text(payload, encoding="utf-8")

        # Backward-compatibility: also persist under the original unsanitized
        # solution name so loaders that resolve by "<solution_ref>.json" keep
        # working for names that contain characters sanitized above.
        if name != safe_name:
            legacy_dest = out_dir / f"{name}.json"
            legacy_dest_resolved = legacy_dest.resolve()
            try:
                legacy_dest_resolved.relative_to(out_dir.resolve())
            except ValueError:
                legacy_dest_resolved = None

            if (
                legacy_dest_resolved is not None
                and legacy_dest_resolved != dest.resolve()
            ):
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
            base_dir=artifacts_dir,
            task_name=str(definition_name or ""),
            run_id=run_id or get_run_id(),
        ).resolve()
        out_dir = root / "eval" / str(definition_name or "__unknown__")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        sn = str(solution_name or "").strip()
        safe_sn = (
            "".join([c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in sn])
            if sn
            else ""
        )
        suffix = f"_{safe_sn}" if safe_sn else ""
        dest = out_dir / f"eval_report_{ts}{suffix}.json"
        dest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
    stage_checkpoint_config: Any | None = None,
    # Checkpointing (AscendC + Claude Agent SDK + world model only)
    checkpoint_enable: bool = False,
    checkpoint_every: str = "cycle",
    checkpoint_dir: Optional[str] = None,
    checkpoint_keep: int = 5,
    resume_from_checkpoint: Optional[str] = None,
    resume_mode: str = "new-run",
    resume_policy: str = "latest",
    checkpoint_include_project_snapshot: bool = True,
    checkpoint_enable_claude_file_checkpointing: bool = False,
    checkpoint_resume_claude_session: bool = False,
    checkpoint_retry_failed_attempt: bool = False,
    input_argv: Optional[list[str]] = None,
    parsed_args: Optional[dict[str, Any]] = None,
) -> None:
    """
    Generate exactly one solution for the task, then run final evaluation.
    """
    from k_search.utils.paths import (
        get_ksearch_run_dir,
        get_ksearch_task_dir,
        get_run_id,
        get_task_id,
    )

    # Determine run_id
    if continue_from_run:
        effective_run_id = continue_from_run
    elif run_id:
        effective_run_id = run_id
    else:
        effective_run_id = get_run_id()
    effective_task_id = get_task_id()

    task_name = str(getattr(task, "name", "") or "")
    start_time = datetime.utcnow().isoformat() + "Z"

    # Write task_meta.json
    task_root = get_ksearch_task_dir(
        base_dir=artifacts_dir,
        task_name=task_name,
        task_id=effective_task_id,
    )
    task_meta_path = task_root / "task_meta.json"
    task_meta = {
        "task_id": effective_task_id,
        "task_name": task_name,
        "start_time": start_time,
        "artifacts_dir": str(task_root.parent.parent),
    }
    try:
        task_meta_path.parent.mkdir(parents=True, exist_ok=True)
        task_meta_path.write_text(
            json.dumps(task_meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[TASK] Task ID: {effective_task_id}")
        print(f"[TASK] Task metadata: {task_meta_path}")
    except Exception as e:
        print(f"[WARN] Failed to write task_meta.json: {e}")

    # Write run_meta.json
    run_root = get_ksearch_run_dir(
        base_dir=artifacts_dir,
        task_name=task_name,
        task_id=effective_task_id,
        run_id=effective_run_id,
    )
    run_meta_path = run_root / "run_meta.json"
    run_meta = {
        "run_id": effective_run_id,
        "task_id": effective_task_id,
        "start_time": start_time,
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
        "checkpoint_enable": bool(checkpoint_enable),
        "checkpoint_every": checkpoint_every,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_keep": int(checkpoint_keep),
        "resume_from_checkpoint": resume_from_checkpoint,
        "resume_mode": resume_mode,
        "resume_policy": resume_policy,
        "strategy_file": strategy_file,
        "strategy_form": strategy_form,
        "checkpoint_v3": bool(getattr(stage_checkpoint_config, "enabled", False)),
        "input_args_path": "input_args.json",
        "task_config_path": "task_config.json",
        "env_path": "env.json",
        "status": "running",
    }
    try:
        run_meta_path.parent.mkdir(parents=True, exist_ok=True)
        run_meta_path.write_text(
            json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[RUN] Run ID: {effective_run_id}")
        print(f"[RUN] Run metadata: {run_meta_path}")
    except Exception as e:
        print(f"[WARN] Failed to write run_meta.json: {e}")

    try:
        _write_json_file(
            run_root / "input_args.json",
            {
                "schema_version": 1,
                "argv": list(input_argv or sys.argv),
                "parsed_args": _redact_mapping(dict(parsed_args or {})),
            },
        )
        task_cfg = (
            task.get_config_for_logging()
            if callable(getattr(task, "get_config_for_logging", None))
            else {}
        )
        _write_json_file(
            run_root / "task_config.json", _redact_mapping(dict(task_cfg or {}))
        )
        _write_json_file(run_root / "env.json", _capture_env_payload())
    except Exception as e:
        print(f"[WARN] Failed to write reproducibility metadata: {e}")

    status = "running"
    run_error = None

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
        from k_search.kernel_generators.kernel_generator_world_model import (
            WorldModelKernelGeneratorWithBaseline,
        )
        from k_search.kernel_generators.checkpoint import CheckpointConfig

        checkpoint_config = CheckpointConfig(
            enabled=bool(checkpoint_enable),
            every=checkpoint_every,  # type: ignore[arg-type]
            checkpoint_dir=checkpoint_dir,
            keep=int(checkpoint_keep),
            resume_from=resume_from_checkpoint,
            resume_mode=resume_mode,  # type: ignore[arg-type]
            resume_policy=resume_policy,  # type: ignore[arg-type]
            include_project_snapshot_payload=bool(checkpoint_include_project_snapshot),
            enable_claude_file_checkpointing=bool(
                checkpoint_enable_claude_file_checkpointing
            ),
            resume_claude_session=bool(checkpoint_resume_claude_session),
            retry_failed_attempt=bool(checkpoint_retry_failed_attempt),
        )

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
            stage_checkpoint_config=stage_checkpoint_config,
            checkpoint_config=checkpoint_config,
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
    if resume_from_checkpoint:
        if continue_from_solution:
            print(
                "[WARN] --resume-from-checkpoint specified; ignoring --continue-from-solution"
            )
            continue_from_solution = None
        if continue_from_world_model:
            print(
                "[WARN] --resume-from-checkpoint specified; ignoring --continue-from-world-model"
            )
            continue_from_world_model = None

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
        solution.description = (
            solution.description or ""
        ) + f" (generated {ts} uid={uid})"
    except Exception:
        pass

    # Optionally persist to disk (k-search solution type)
    if save_solutions:
        saved_path = _persist_ksearch_solution(
            solution,
            definition_name=str(getattr(task, "name", "") or ""),
            artifacts_dir=artifacts_dir,
            run_id=effective_run_id,
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
    _update_run_meta(
        run_meta_path,
        status="completed",
        end_time=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        error=None,
    )


def _build_task_from_args(args: Any) -> Any:
    task_source = str(getattr(args, "task_source", None) or "flashinfer")
    task_path = str(
        getattr(args, "task_path", None) or (getattr(args, "local", None) or "")
    )
    default_language = "ascendc" if task_source == "ascendc" else "triton"
    language = (
        str(getattr(args, "language", default_language) or default_language)
        .strip()
        .lower()
    )

    if task_source == "flashinfer":
        from k_search.tasks.flashinfer_bench_task import FlashInferBenchTask

        if language in {"mlx", "ascendc"}:
            raise ValueError(
                f"--language {language} is not supported with --task-source=flashinfer"
            )

        if not task_path:
            raise ValueError(
                "--local or --task-path is required for --task-source=flashinfer"
            )
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
            raise ValueError(
                f"--language {language} is not supported with --task-source=gpumode"
            )

        return GpuModeTriMulTask(
            mode=str(getattr(args, "gpumode_mode", None) or "benchmark"),
            keep_tmp=bool(getattr(args, "gpumode_keep_tmp", False)),
            task_dir=(
                str(getattr(args, "gpumode_task_dir", ""))
                if getattr(args, "gpumode_task_dir", None)
                else None
            ),
            artifacts_dir=getattr(args, "artifacts_dir", None),
        )
    if task_source == "kernelbench":
        from k_search.tasks.kernelbench_task import KernelBenchTask

        if language in {"mlx", "ascendc"}:
            raise ValueError(
                f"--language {language} is not supported with --task-source=kernelbench"
            )

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
        def_name = str(
            getattr(args, "definition", None) or "mlx_mamba_selective_scan_fwd"
        ).strip()
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
    parser = argparse.ArgumentParser(
        description="Generate kernels with GPT/Gemini (OpenAI-compatible) and evaluate via task backends."
    )
    parser.add_argument(
        "--local",
        required=False,
        default=None,
        help="Path to flashinfer-trace dataset root (flashinfer only)",
    )
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
    parser.add_argument(
        "--definition", default=None, help="Single definition name to target (required)"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="LLM model name (e.g., gpt-4.1, gpt-5, gemini-2.5-pro via compatible endpoint)",
    )
    parser.add_argument(
        "--llm-provider",
        default="openai",
        choices=["openai", "claude-agent"],
        help="LLM backend: openai for OpenAI-compatible APIs, claude-agent for Claude Agent SDK.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for non-OpenAI providers (e.g. Gemini proxy)",
    )
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
    parser.add_argument(
        "--target-gpu", default="H100", help="Target GPU architecture hint for prompts"
    )
    parser.add_argument(
        "--max-opt-rounds",
        type=int,
        default=5,
        help="Max optimization rounds for each solution generation",
    )

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
    parser.add_argument(
        "--no-save-results", action="store_true", help="Do not write traces to dataset"
    )
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
    parser.add_argument(
        "--baseline-solution",
        default=None,
        help="Optional baseline solution name to compare against; if absent, 'vs_base' is omitted",
    )
    parser.add_argument(
        "--num-eval-workload",
        type=int,
        default=None,
        help="If set, evaluate only this many workloads per definition; default uses all workloads",
    )
    # Continue optimization options
    parser.add_argument(
        "--continue-from-solution",
        default=None,
        help="Resume optimization from an existing solution name in the dataset",
    )
    parser.add_argument(
        "--continue-from-world-model",
        default=None,
        help=(
            "Resume world-model prompting state from a JSON file path. "
            "Use 'auto' to load <base>/<task>/<task_id>/runs/<run_id>/artifacts/world_model/world_model.json "
            "if present, falling back to <base>/<task>/<task_id>/artifacts/world_model/world_model.json."
        ),
    )
    parser.add_argument(
        "--feedback-workloads",
        nargs="+",
        default=None,
        help="Explicit workload UUIDs to use for optimization feedback rounds",
    )
    parser.add_argument(
        "--checkpoint-enable",
        action="store_true",
        help="Enable AscendC Claude Agent world-model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        choices=["cycle", "attempt"],
        default="cycle",
        help="Checkpoint save boundary. V1 supports cycle; attempt is reserved for V2.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint directory (default: run artifacts/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=5,
        help="Number of checkpoint directories to retain",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from checkpoint ref: latest, checkpoint id, or manifest path",
    )
    parser.add_argument(
        "--resume-mode", choices=["same-run", "new-run"], default="new-run"
    )
    parser.add_argument(
        "--resume-policy", choices=["latest", "stable"], default="latest"
    )
    parser.add_argument(
        "--checkpoint-include-project-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include project snapshot payload in checkpoints when available",
    )
    parser.add_argument(
        "--checkpoint-enable-claude-file-checkpointing",
        action="store_true",
        help="Reserved V2 option: request Claude file checkpointing metadata when supported",
    )
    parser.add_argument(
        "--checkpoint-resume-claude-session",
        action="store_true",
        help="Reserved V2 option: try to resume Claude session metadata when safe",
    )
    parser.add_argument(
        "--checkpoint-retry-failed-attempt",
        action="store_true",
        help="Reserved V2 option for retrying a failed attempt after restore",
    )
    # Nsight Compute
    parser.add_argument(
        "--feedback-trace-policy",
        default="first",
        choices=["first", "random"],
        help="Policy for selecting feedback traces",
    )
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
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", default=os.getenv("WANDB_PROJECT"), help="W&B project"
    )
    parser.add_argument(
        "--run-name", default=os.getenv("RUN_NAME"), help="W&B run name"
    )

    # GPUMode options
    parser.add_argument(
        "--gpumode-mode",
        default="benchmark",
        help="GPUMode eval mode (e.g., benchmark/test/leaderboard/profile)",
    )
    parser.add_argument(
        "--gpumode-keep-tmp",
        action="store_true",
        help="Keep GPUMode temp working dir for debugging",
    )
    parser.add_argument(
        "--gpumode-task-dir",
        default=None,
        help="Override GPUMode task dir (defaults to vendored trimul task)",
    )

    # KernelBench options
    parser.add_argument(
        "--kernelbench-level",
        type=int,
        default=1,
        help="KernelBench level (1, 2, or 3)",
    )
    parser.add_argument(
        "--kernelbench-problem-id",
        type=int,
        default=1,
        help="Problem ID within the level",
    )
    parser.add_argument(
        "--kernelbench-eval-mode",
        default="local",
        choices=["local", "modal"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--kernelbench-num-correct-trials",
        type=int,
        default=5,
        help="Number of correctness trials",
    )
    parser.add_argument(
        "--kernelbench-num-perf-trials",
        type=int,
        default=100,
        help="Number of performance trials",
    )

    # AscendC options
    parser.add_argument(
        "--ascendc-build-cmd",
        default=None,
        help="Shell command that compiles the AscendC candidate in the candidate project root",
    )
    parser.add_argument(
        "--ascendc-test-cmd",
        default=None,
        help="Shell command that validates AscendC correctness in the candidate project root",
    )
    parser.add_argument(
        "--ascendc-bench-cmd",
        default=None,
        help="Shell command that benchmarks the AscendC candidate and prints latency_ms=<float>",
    )
    parser.add_argument(
        "--ascendc-timeout-seconds",
        type=int,
        default=600,
        help="Timeout per AscendC build/test/bench command",
    )
    parser.add_argument(
        "--ascendc-reference-latency-ms",
        type=float,
        default=None,
        help="Optional baseline latency used to score speedup",
    )
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

    # Checkpoint V3 options
    parser.add_argument(
        "--checkpoint-v3",
        action="store_true",
        help="Enable V3 stage-boundary checkpoints for AscendC Claude world-model runs",
    )
    parser.add_argument(
        "--checkpoint-stage-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save stage_completed checkpoints",
    )
    parser.add_argument(
        "--checkpoint-stage-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save stage_start checkpoints",
    )
    parser.add_argument(
        "--checkpoint-claude-session-required",
        action="store_true",
        help="Fail restore if Claude session resume fails",
    )
    parser.add_argument(
        "--checkpoint-subagent-resume",
        action="store_true",
        help="Enable best-effort completed subagent follow-up resume metadata",
    )
    parser.add_argument(
        "--checkpoint-require-subagent-agent-id",
        action="store_true",
        help="Require captured subagent agentId for strict subagent resume debugging",
    )
    parser.add_argument(
        "--checkpoint-session-store-kind",
        choices=["none", "custom"],
        default="none",
        help="Optional Claude SessionStore kind",
    )
    parser.add_argument(
        "--checkpoint-session-store-config",
        default=None,
        help="Config path for a custom Claude SessionStore adapter",
    )
    parser.add_argument(
        "--checkpoint-resume-stage-policy",
        choices=["next-pending"],
        default="next-pending",
    )
    parser.add_argument(
        "--checkpoint-file-state-source",
        choices=["project-snapshot"],
        default="project-snapshot",
    )

    # Strategy injection options
    parser.add_argument(
        "--strategy-file",
        default=None,
        help=(
            "Path to a strategy catalog JSON file. When provided alongside --world-model, "
            "the WM decision tree is seeded with strategy-derived action nodes instead of "
            "LLM-generated ones. Catalog entries must reference markdown strategy files."
        ),
    )
    parser.add_argument(
        "--strategy-form",
        type=_parse_strategy_form,
        default="natural_language",
        help=(
            "Strategy rendering form. Only 'natural_language' is supported. "
            "Use --strategy-file with a JSON catalog whose entries reference "
            "markdown strategy files."
        ),
    )

    args = parser.parse_args()

    # Pin a single output base plus task/run ids for the whole process so that artifacts,
    # llm logs, telemetry and the narrative summary all land under the same
    # <base>/<task>/<task_id>/runs/<run_id>/ tree (and never drift apart across calls).
    from k_search.utils.paths import get_run_id, get_task_id, resolve_output_base

    os.environ.setdefault(
        "KSEARCH_ARTIFACTS_DIR",
        str(resolve_output_base(getattr(args, "artifacts_dir", None))),
    )
    os.environ.setdefault("KSEARCH_TASK_ID", get_task_id())
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
    _validate_checkpoint_args(args, llm_provider=llm_provider)
    stage_checkpoint_config = _stage_checkpoint_config_from_args(
        args, llm_provider=llm_provider
    )

    task = _build_task_from_args(args)

    try:
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
            stage_checkpoint_config=stage_checkpoint_config,
            checkpoint_enable=args.checkpoint_enable,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_keep=args.checkpoint_keep,
            resume_from_checkpoint=args.resume_from_checkpoint,
            resume_mode=args.resume_mode,
            resume_policy=args.resume_policy,
            checkpoint_include_project_snapshot=args.checkpoint_include_project_snapshot,
            checkpoint_enable_claude_file_checkpointing=args.checkpoint_enable_claude_file_checkpointing,
            checkpoint_resume_claude_session=args.checkpoint_resume_claude_session,
            checkpoint_retry_failed_attempt=args.checkpoint_retry_failed_attempt,
            input_argv=list(sys.argv),
            parsed_args=vars(args),
        )
    except KeyboardInterrupt as exc:
        from k_search.utils.paths import get_ksearch_run_dir

        run_meta_path = (
            get_ksearch_run_dir(
                base_dir=args.artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                run_id=os.getenv("KSEARCH_RUN_ID"),
            )
            / "run_meta.json"
        )
        _update_run_meta(
            run_meta_path,
            status="interrupted",
            end_time=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            error=_serialize_error(exc),
        )
        raise
    except BaseException as exc:
        from k_search.utils.paths import get_ksearch_run_dir

        run_meta_path = (
            get_ksearch_run_dir(
                base_dir=args.artifacts_dir,
                task_name=str(getattr(task, "name", "") or ""),
                run_id=os.getenv("KSEARCH_RUN_ID"),
            )
            / "run_meta.json"
        )
        _update_run_meta(
            run_meta_path,
            status="failed",
            end_time=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            error=_serialize_error(exc),
        )
        raise


if __name__ == "__main__":
    main()
