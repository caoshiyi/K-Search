from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.agentic_candidate_artifacts import write_agentic_candidate_artifacts
from k_search.kernel_generators.agentic_worktree import create_agentic_worktree
from k_search.kernel_generators.candidate_patch import CandidatePatch
from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
    ClaudeProjectEditorSession,
)
from k_search.kernel_generators.project_snapshot import ProjectSnapshot, create_project_snapshot
from k_search.tasks.task_base import EvalResult, Solution
from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder
from k_search.utils.path_sanitize import sanitize_worktree_paths
from k_search.utils.paths import get_ksearch_artifacts_dir, get_run_id


AgenticMode = Literal["generate", "action", "debug", "improve"]


@dataclass
class AscendCAgenticCodegenRequest:
    definition_text: str
    action_text: str
    trace_logs: str
    perf_summary: str
    target_gpu: str
    round_num: int
    attempt_idx: int
    mode: AgenticMode
    run_id: str | None = None  # New: run_id for organizing runs
    task_name: str | None = None  # New: task_name for artifacts directory
    parent_candidate_id: str | None = None
    action_node_id: str | None = None


@dataclass
class AscendCAgenticCodegenResult:
    solution: Solution
    eval_result: EvalResult
    raw: str
    cleaned: dict[str, str]
    transcript: str
    prompt: str
    prompt_chars: int
    changed_paths: list[str]
    diff_text: str
    project_path: str
    diff_after_eval: str | None = None
    evaluator_mutated_project: bool = False
    candidate_patch: CandidatePatch | None = None
    project_snapshot: ProjectSnapshot | None = None
    artifact_paths: dict[str, str] | None = None
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
    code_map_text: str | None = None
    editor_session: ClaudeProjectEditorSession | None = None
    worktree_session: Any = None


def _truncate(text: str, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 40)].rstrip() + "\n[truncated for agentic prompt budget]"


def _build_fix_prompt(eval_result: EvalResult, fix_round: int, max_chars: int = 6000) -> str:
    """Build a short fix prompt from an EvalResult, for multi-turn session follow-ups."""
    status = eval_result.status
    log = _truncate(eval_result.log_excerpt, max_chars)
    log = sanitize_worktree_paths(log)

    if status == "compile_failed":
        return (
            f"The candidate you just wrote failed to compile (fix attempt {fix_round}).\n"
            f"Build error output:\n\n{log}\n\n"
            f"Fix the compilation error. Read the relevant source files first, "
            f"identify the root cause, and make minimal edits."
        )
    elif status == "failed":
        return (
            f"The candidate compiled but failed correctness/precision testing (fix attempt {fix_round}).\n"
            f"Test error output:\n\n{log}\n\n"
            f"Fix the correctness failure. Read the relevant source files and test logs, "
            f"identify what assertion or output mismatch caused the failure, and make minimal edits."
        )
    elif status == "benchmark_failed":
        return (
            f"The candidate passed correctness but the benchmark failed (fix attempt {fix_round}).\n"
            f"Benchmark error output:\n\n{log}\n\n"
            f"Fix the benchmark failure while maintaining correctness."
        )
    elif status == "timeout":
        return (
            f"The candidate timed out during evaluation (fix attempt {fix_round}).\n"
            f"Timeout log:\n\n{log}\n\n"
            f"Fix the timeout — possible infinite loop or excessive computation."
        )
    else:
        return (
            f"Evaluation failed with status '{status}' (fix attempt {fix_round}).\n"
            f"Details:\n\n{log}\n\nPlease fix this."
        )


def _edit_project_with_optional_telemetry(
    editor_client: Any,
    *,
    project_dir: Path,
    prompt: str,
    telemetry_recorder: Any,
) -> ClaudeProjectEditResult:
    try:
        return editor_client.edit_project(
            project_dir=project_dir,
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
        )
    except TypeError as exc:
        if "telemetry_recorder" not in str(exc):
            raise
        return editor_client.edit_project(project_dir=project_dir, prompt=prompt)


class AscendCAgenticPromptBuilder:
    def __init__(self, *, max_chars: int | None = None) -> None:
        if max_chars is None:
            raw = os.getenv("KSEARCH_AGENTIC_PROMPT_MAX_CHARS", "").strip()
            max_chars = int(raw) if raw.isdigit() and int(raw) > 0 else 20_000
        self.max_chars = int(max_chars)

    def build(self, request: AscendCAgenticCodegenRequest, *, has_code_map: bool = False, task_path: str | None = None) -> str:
        sections = {
            "definition": _truncate(request.definition_text, 5000),
            "action": _truncate(request.action_text, 3000),
            "perf_summary": _truncate(request.perf_summary, 2500),
            "trace_logs": _truncate(request.trace_logs, 4000),
        }
        if has_code_map:
            inspect_line = (
                "A CODE_MAP.md at the project root describes file roles, kernel structure, "
                "tiling, buffers, and contracts. Read it first instead of grepping the whole project. "
                "After editing code, update the affected sections of CODE_MAP.md to keep it accurate.\n"
            )
        else:
            inspect_line = "First inspect the project with Glob, Grep, and Read. Then edit only necessary files.\n"
        prompt = (
            "You are an AscendC performance optimization agent working inside a candidate project directory.\n"
            "IMPORTANT: You must ONLY edit files inside the current project directory (CWD). Do NOT use absolute paths from any external directories.\n"
            f"Target GPU: {request.target_gpu}\n"
            f"Mode: {request.mode}\n"
            f"Round: {int(request.round_num)}\n"
            f"Attempt: {int(request.attempt_idx)}\n\n"
            "Available tools: Read/Grep/Glob/Edit/Write. Bash is disabled.\n"
            + inspect_line
            + "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
            "Preserve operator semantics, public entry points, host tiling contract, correctness harness behavior, and build layout.\n"
            "Do not return a full source container. Modify files in the project directory.\n"
            "End with a concise summary and changed-file list.\n\n"
            "Task specification:\n"
            f"{sections['definition']}\n\n"
            "Chosen action or debug intent:\n"
            f"{sections['action']}\n\n"
            "Performance summary:\n"
            f"{sections['perf_summary'] or '(none)'}\n\n"
            "Recent failure or trace excerpt:\n"
            f"{sections['trace_logs'] or '(none)'}\n"
        )
        # 不变量:送达 LLM 的文本不得携带物理路径(worktree 或原始任务目录),统一抹成语义占位符。
        prompt = sanitize_worktree_paths(prompt, task_path=task_path)
        if len(prompt) > self.max_chars:
            sizes = ", ".join(f"{name}={len(value)}" for name, value in sorted(sections.items()))
            raise ValueError(
                f"agentic prompt exceeded {self.max_chars} chars: prompt={len(prompt)}, sections: {sizes}"
            )
        return prompt


class AscendCAgenticCodegenRunner:
    def __init__(
        self,
        *,
        model_name: str,
        editor_client: Any | None = None,
        reader_editor_client: Any | None = None,
        prompt_builder: AscendCAgenticPromptBuilder | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.editor_client = editor_client or ClaudeAgentProjectEditorClient(model_name=self.model_name)
        self.reader_editor_client = reader_editor_client
        self.prompt_builder = prompt_builder or AscendCAgenticPromptBuilder()

    def run(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
    ) -> AscendCAgenticCodegenResult:
        session = create_agentic_worktree(task_path=getattr(task, "task_path", None))
        try:
            overlay = getattr(task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=session.project_dir, solution=base_solution)
                session.commit_all("ksearch agentic overlay baseline")

            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = False
            if store is not None:
                from k_search.kernel_generators.agents import CodeReaderAgent

                if store.load(CODE_MAP) is None:
                    try:
                        reader = CodeReaderAgent(
                            model_name=self.model_name, editor_client=self.reader_editor_client
                        )
                        reader.run(
                            project_dir=session.project_dir,
                            context={"definition_text": request.definition_text, "task_path": str(getattr(task, "task_path", "") or "")},
                        )
                        produced = store.read_from_worktree(CODE_MAP, session.project_dir)
                        if produced:
                            store.save(CODE_MAP, produced)
                    except Exception as exc:  # noqa: BLE001 - code map is best-effort
                        import warnings

                        warnings.warn(f"code_map reader failed, continuing without it: {exc}")
                has_code_map = store.materialize(CODE_MAP, session.project_dir)

            from k_search.kernel_generators.agents import CodegenAgent  # lazy import: avoid module-top cycle

            codegen = CodegenAgent(
                model_name=self.model_name,
                editor_client=self.editor_client,
                prompt_builder=self.prompt_builder,
            )
            codegen_context = {"request": request, "has_code_map": has_code_map, "task_path": str(getattr(task, "task_path", "") or "")}
            prompt = codegen.build_prompt(codegen_context)
            # Replace absolute task-path references so the LLM only sees <PROJECT_ROOT>.
            task_path = getattr(task, "task_path", None)
            if task_path is not None:
                prompt = prompt.replace(str(Path(task_path).expanduser().resolve()), "<PROJECT_ROOT>")
            telemetry_context = TelemetryContext(
                task_name=getattr(task, "definition_name", None),
                definition=getattr(task, "definition_name", None),
                flow="agentic_codegen",
                stage=request.mode,
                round_index=request.round_num,
                attempt_index=request.attempt_idx,
                model_name=self.model_name,
                provider="claude-agent",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
            try:
                agent_result = codegen.run(
                    project_dir=session.project_dir,
                    context=codegen_context,
                    telemetry_recorder=telemetry_recorder,
                )
                edit_result = agent_result.edit_result
            finally:
                telemetry_recorder.close()
            code_map_text = store.read_from_worktree(CODE_MAP, session.project_dir) if store is not None else None
            if store is not None:
                (session.project_dir / CODE_MAP.filename).unlink(missing_ok=True)
            project_changed_paths = session.project_changed_paths()
            changed_paths = project_changed_paths or session.changed_paths()
            changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
            if not changed_paths:
                # The LLM may have written edits to absolute paths outside the
                # worktree (e.g. the original task directory).  Sync those
                # changes into the worktree so change-detection can find them.
                task_path = getattr(task, "task_path", None)
                if task_path is not None:
                    import shutil as _shutil
                    from pathlib import Path as _P
                    src_root = _P(task_path).expanduser().resolve()
                    dst_root = session.project_dir
                    ignore = _shutil.ignore_patterns(".git", "__pycache__", "build", "cmake-build-debug", "logs")
                    try:
                        _shutil.copytree(src_root, dst_root, dirs_exist_ok=True, ignore=ignore)
                    except Exception as exc:  # noqa: BLE001
                        import logging
                        logging.getLogger(__name__).warning("mirror sync failed: %s", exc)
                    session.commit_all("ksearch sync external edits")
                    project_changed_paths = session.project_changed_paths()
                    changed_paths = project_changed_paths or session.changed_paths()
                    changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
            if not changed_paths:
                raise RuntimeError(
                    "Claude agentic AscendC codegen did not change any files "
                    f"(round={request.round_num}, attempt={request.attempt_idx})"
                )
            diff_text = session.project_diff_text()
            run_in_project_dir = getattr(task, "run_benchmark_in_project_dir", None)
            if not callable(run_in_project_dir):
                raise RuntimeError("AscendC agentic task does not support run_benchmark_in_project_dir")
            eval_result = run_in_project_dir(project_dir=session.project_dir, round_num=request.round_num)
            diff_after_eval = session.project_diff_text()
            evaluator_mutated_project = diff_after_eval != diff_text
            solution = task.make_solution_from_project_dir(
                project_dir=session.project_dir,
                changed_paths=changed_paths,
                raw_agent_output=edit_result.text,
                round_num=request.round_num,
                model_name=self.model_name,
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            cleaned = {src.path: src.content for src in solution.sources or []}
            candidate_id = f"round_{int(request.round_num):04d}_attempt_{int(request.attempt_idx):02d}"
            snapshot_id = f"{candidate_id}_snapshot"
            task_name = request.task_name or getattr(task, "definition_name", None) or getattr(task, "name", "ascendc")
            run_id = request.run_id or get_run_id()
            artifacts_dir = getattr(task, "artifacts_dir", None)
            snapshot_archive_dir = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(task_name), run_id=run_id) / "snapshots"
            project_snapshot = create_project_snapshot(
                project_dir=session.project_dir,
                snapshot_id=snapshot_id,
                parent_snapshot_id=None,
                base_commit=session.baseline_commit,
                created_by_round=request.round_num,
                eval_result=eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000),
                diff_from_parent=diff_text,
                archive_dir=snapshot_archive_dir,
                run_id=run_id,
            )
            candidate_patch, artifact_paths = write_agentic_candidate_artifacts(
                artifacts_dir=artifacts_dir,
                task_name=str(task_name),
                run_id=run_id,
                round_num=request.round_num,
                attempt_idx=request.attempt_idx,
                prompt=prompt,
                transcript=edit_result.transcript,
                changed_paths=changed_paths,
                diff_text=diff_text,
                eval_result=eval_result,
                project_snapshot=project_snapshot,
                parent_candidate_id=request.parent_candidate_id,
                base_ref=session.baseline_commit,
                project_rel_path=session.project_rel_path(),
                action_node_id=request.action_node_id,
                model_name=self.model_name,
                metadata={
                    "target_gpu": request.target_gpu,
                    "mode": request.mode,
                    "project_path": str(session.project_dir),
                    "evaluator_mutated_project": evaluator_mutated_project,
                },
            )
            return AscendCAgenticCodegenResult(
                solution=solution,
                eval_result=eval_result,
                raw=task.code_for_world_model_from_raw(raw=cleaned, language="ascendc"),
                cleaned=cleaned,
                transcript=edit_result.transcript,
                prompt=prompt,
                prompt_chars=len(prompt),
                changed_paths=changed_paths,
                diff_text=diff_text,
                project_path=str(session.project_dir),
                diff_after_eval=diff_after_eval,
                evaluator_mutated_project=evaluator_mutated_project,
                candidate_patch=candidate_patch,
                project_snapshot=project_snapshot,
                artifact_paths=artifact_paths,
                trace_path=edit_result.trace_path or telemetry_recorder.artifacts.trace_path,
                timeline_path=edit_result.timeline_path or telemetry_recorder.artifacts.timeline_path,
                cost_path=edit_result.cost_path or telemetry_recorder.artifacts.cost_path,
                session_id=edit_result.session_id,
                total_cost_usd=edit_result.total_cost_usd,
                usage=edit_result.usage,
                model_usage=edit_result.model_usage,
                num_turns=edit_result.num_turns,
                duration_ms=edit_result.duration_ms,
                code_map_text=code_map_text,
                editor_session=None,
                worktree_session=None,
            )
        finally:
            session.cleanup()

    def run_multi_turn(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
        max_fix_rounds: int = 3,
    ) -> AscendCAgenticCodegenResult:
        """Run agentic codegen with multi-turn fix loop in a single SDK session.

        After the initial code generation attempt, if evaluation fails (compile
        or precision), follow-up fix prompts are sent within the same session
        up to ``max_fix_rounds`` times.
        """
        max_fix_rounds = int(os.getenv("KSEARCH_AGENTIC_MAX_FIX_ROUNDS", str(max_fix_rounds)))
        wt_session = create_agentic_worktree(task_path=getattr(task, "task_path", None))
        editor_session: ClaudeProjectEditorSession | None = None
        try:
            overlay = getattr(task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=wt_session.project_dir, solution=base_solution)
                wt_session.commit_all("ksearch agentic overlay baseline")

            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0", "false", "no", "off",
            }
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = False
            if store is not None:
                from k_search.kernel_generators.agents import CodeReaderAgent

                if store.load(CODE_MAP) is None:
                    try:
                        reader = CodeReaderAgent(
                            model_name=self.model_name, editor_client=self.reader_editor_client
                        )
                        reader.run(
                            project_dir=wt_session.project_dir,
                            context={"definition_text": request.definition_text},
                        )
                        produced = store.read_from_worktree(CODE_MAP, wt_session.project_dir)
                        if produced:
                            store.save(CODE_MAP, produced)
                    except Exception as exc:  # noqa: BLE001
                        import warnings
                        warnings.warn(f"code_map reader failed, continuing without it: {exc}")
                has_code_map = store.materialize(CODE_MAP, wt_session.project_dir)

            # Open a multi-turn SDK session
            editor_session = self.editor_client.open_session(
                project_dir=wt_session.project_dir,
            )

            # Attempt 1: send full prompt
            prompt = self.prompt_builder.build(request, has_code_map=has_code_map)
            prompt = sanitize_worktree_paths(prompt)
            # Replace absolute task-path references so the LLM only sees <PROJECT_ROOT>.
            task_path = getattr(task, "task_path", None)
            if task_path is not None:
                prompt = prompt.replace(str(Path(task_path).expanduser().resolve()), "<PROJECT_ROOT>")
            telemetry_context = TelemetryContext(
                task_name=getattr(task, "definition_name", None),
                definition=getattr(task, "definition_name", None),
                flow="agentic_codegen_multi_turn",
                stage=request.mode,
                round_index=request.round_num,
                attempt_index=request.attempt_idx,
                model_name=self.model_name,
                provider="claude-agent",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
            try:
                edit_result = self.editor_client.send_prompt(
                    editor_session, prompt=prompt,
                    telemetry_recorder=telemetry_recorder,
                )
            finally:
                telemetry_recorder.close()

            code_map_text = store.read_from_worktree(CODE_MAP, wt_session.project_dir) if store is not None else None
            if store is not None:
                (wt_session.project_dir / CODE_MAP.filename).unlink(missing_ok=True)

            project_changed_paths = wt_session.project_changed_paths()
            changed_paths = project_changed_paths or wt_session.changed_paths()
            changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
            if not changed_paths:
                # The LLM may have written edits to absolute paths outside the
                # worktree (e.g. the original task directory).  Sync those
                # changes into the worktree so change-detection can find them.
                task_path = getattr(task, "task_path", None)
                if task_path is not None:
                    import shutil as _shutil
                    from pathlib import Path as _P
                    src_root = _P(task_path).expanduser().resolve()
                    dst_root = wt_session.project_dir
                    ignore = _shutil.ignore_patterns(".git", "__pycache__", "build", "cmake-build-debug", "logs")
                    try:
                        _shutil.copytree(src_root, dst_root, dirs_exist_ok=True, ignore=ignore)
                    except Exception as exc:  # noqa: BLE001
                        import logging
                        logging.getLogger(__name__).warning("mirror sync failed: %s", exc)
                    wt_session.commit_all("ksearch sync external edits")
                    project_changed_paths = wt_session.project_changed_paths()
                    changed_paths = project_changed_paths or wt_session.changed_paths()
                    changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
            if not changed_paths:
                raise RuntimeError(
                    "Claude agentic AscendC codegen did not change any files "
                    f"(round={request.round_num}, attempt={request.attempt_idx})"
                )

            diff_text = wt_session.project_diff_text()
            run_in_project_dir = getattr(task, "run_benchmark_in_project_dir", None)
            if not callable(run_in_project_dir):
                raise RuntimeError("AscendC agentic task does not support run_benchmark_in_project_dir")
            eval_result = run_in_project_dir(project_dir=wt_session.project_dir, round_num=request.round_num)
            diff_after_eval = wt_session.project_diff_text()
            evaluator_mutated_project = diff_after_eval != diff_text

            # Fix loop: send short fix prompts in the same session
            for fix_round in range(1, max_fix_rounds + 1):
                if eval_result.is_passed():
                    break

                # Commit evaluator mutations before next fix attempt
                if evaluator_mutated_project:
                    wt_session.commit_all(f"ksearch eval mutations (fix round {fix_round})")
                    diff_text = wt_session.project_diff_text()

                fix_prompt = _build_fix_prompt(eval_result, fix_round)
                fix_prompt = sanitize_worktree_paths(fix_prompt)

                fix_telemetry_context = TelemetryContext(
                    task_name=getattr(task, "definition_name", None),
                    definition=getattr(task, "definition_name", None),
                    flow="agentic_codegen_multi_turn",
                    stage="fix",
                    round_index=request.round_num,
                    attempt_index=request.attempt_idx,
                    fix_round_index=fix_round,
                    model_name=self.model_name,
                    provider="claude-agent",
                    target_gpu=request.target_gpu,
                    language="ascendc",
                )
                fix_telemetry_recorder = build_file_recorder(context=fix_telemetry_context, prompt=fix_prompt)
                try:
                    edit_result = self.editor_client.send_prompt(
                        editor_session, prompt=fix_prompt,
                        telemetry_recorder=fix_telemetry_recorder,
                    )
                finally:
                    fix_telemetry_recorder.close()

                # Check for changes after fix
                project_changed_paths = wt_session.project_changed_paths()
                changed_paths = project_changed_paths or wt_session.changed_paths()
                changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
                if not changed_paths:
                    break  # LLM didn't change anything, stop trying

                # Re-evaluate
                eval_result = run_in_project_dir(project_dir=wt_session.project_dir, round_num=request.round_num)
                diff_after_eval = wt_session.project_diff_text()
                evaluator_mutated_project = diff_after_eval != diff_text

            # Build final result (same as run())
            solution = task.make_solution_from_project_dir(
                project_dir=wt_session.project_dir,
                changed_paths=changed_paths,
                raw_agent_output=edit_result.text,
                round_num=request.round_num,
                model_name=self.model_name,
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            cleaned = {src.path: src.content for src in solution.sources or []}
            candidate_id = f"round_{int(request.round_num):04d}_attempt_{int(request.attempt_idx):02d}"
            snapshot_id = f"{candidate_id}_snapshot"
            task_name = request.task_name or getattr(task, "definition_name", None) or getattr(task, "name", "ascendc")
            run_id = request.run_id or get_run_id()
            artifacts_dir = getattr(task, "artifacts_dir", None)
            snapshot_archive_dir = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(task_name), run_id=run_id) / "snapshots"
            project_snapshot = create_project_snapshot(
                project_dir=wt_session.project_dir,
                snapshot_id=snapshot_id,
                parent_snapshot_id=None,
                base_commit=wt_session.baseline_commit,
                created_by_round=request.round_num,
                eval_result=eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000),
                diff_from_parent=diff_text,
                archive_dir=snapshot_archive_dir,
                run_id=run_id,
            )
            candidate_patch, artifact_paths = write_agentic_candidate_artifacts(
                artifacts_dir=artifacts_dir,
                task_name=str(task_name),
                run_id=run_id,
                round_num=request.round_num,
                attempt_idx=request.attempt_idx,
                prompt=prompt,
                transcript=edit_result.transcript,
                changed_paths=changed_paths,
                diff_text=diff_text,
                eval_result=eval_result,
                project_snapshot=project_snapshot,
                parent_candidate_id=request.parent_candidate_id,
                base_ref=wt_session.baseline_commit,
                project_rel_path=wt_session.project_rel_path(),
                action_node_id=request.action_node_id,
                model_name=self.model_name,
                metadata={
                    "target_gpu": request.target_gpu,
                    "mode": request.mode,
                    "project_path": str(wt_session.project_dir),
                    "evaluator_mutated_project": evaluator_mutated_project,
                },
            )
            return AscendCAgenticCodegenResult(
                solution=solution,
                eval_result=eval_result,
                raw=task.code_for_world_model_from_raw(raw=cleaned, language="ascendc"),
                cleaned=cleaned,
                transcript=edit_result.transcript,
                prompt=prompt,
                prompt_chars=len(prompt),
                changed_paths=changed_paths,
                diff_text=diff_text,
                project_path=str(wt_session.project_dir),
                diff_after_eval=diff_after_eval,
                evaluator_mutated_project=evaluator_mutated_project,
                candidate_patch=candidate_patch,
                project_snapshot=project_snapshot,
                artifact_paths=artifact_paths,
                trace_path=edit_result.trace_path or telemetry_recorder.artifacts.trace_path,
                timeline_path=edit_result.timeline_path or telemetry_recorder.artifacts.timeline_path,
                cost_path=edit_result.cost_path or telemetry_recorder.artifacts.cost_path,
                session_id=edit_result.session_id,
                total_cost_usd=edit_result.total_cost_usd,
                usage=edit_result.usage,
                model_usage=edit_result.model_usage,
                num_turns=edit_result.num_turns,
                duration_ms=edit_result.duration_ms,
                code_map_text=code_map_text,
                editor_session=editor_session,
                worktree_session=wt_session,
            )
        except Exception:
            # On error, close the editor session so it doesn't leak
            if editor_session is not None and not editor_session._closed:
                self.editor_client.close_session(editor_session)
                editor_session = None
            raise
        finally:
            # Only cleanup worktree if we're NOT returning it for further use
            # (caller will cleanup when done with the session)
            if editor_session is None:
                wt_session.cleanup()

    def continue_fix(
        self,
        *,
        task: Any,
        editor_session: ClaudeProjectEditorSession,
        wt_session: Any,
        fix_prompt: str,
        request: AscendCAgenticCodegenRequest,
    ) -> AscendCAgenticCodegenResult:
        """Send a fix prompt in an existing session and evaluate the result.

        Used by the world model path for attempt 2+ within the same cycle.
        The editor_session and wt_session must come from a previous run_multi_turn()
        or continue_fix() call.
        """
        fix_prompt = sanitize_worktree_paths(fix_prompt)

        telemetry_context = TelemetryContext(
            task_name=getattr(task, "definition_name", None),
            definition=getattr(task, "definition_name", None),
            flow="agentic_codegen_multi_turn",
            stage="fix",
            round_index=request.round_num,
            attempt_index=request.attempt_idx,
            model_name=self.model_name,
            provider="claude-agent",
            target_gpu=request.target_gpu,
            language="ascendc",
        )
        telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=fix_prompt)
        try:
            edit_result = self.editor_client.send_prompt(
                editor_session, prompt=fix_prompt,
                telemetry_recorder=telemetry_recorder,
            )
        finally:
            telemetry_recorder.close()

        project_changed_paths = wt_session.project_changed_paths()
        changed_paths = project_changed_paths or wt_session.changed_paths()
        if not changed_paths:
            raise RuntimeError(
                "Claude agentic fix did not change any files "
                f"(round={request.round_num}, attempt={request.attempt_idx})"
            )

        diff_text = wt_session.project_diff_text()
        run_in_project_dir = getattr(task, "run_benchmark_in_project_dir", None)
        if not callable(run_in_project_dir):
            raise RuntimeError("AscendC agentic task does not support run_benchmark_in_project_dir")
        eval_result = run_in_project_dir(project_dir=wt_session.project_dir, round_num=request.round_num)
        diff_after_eval = wt_session.project_diff_text()
        evaluator_mutated_project = diff_after_eval != diff_text

        solution = task.make_solution_from_project_dir(
            project_dir=wt_session.project_dir,
            changed_paths=changed_paths,
            raw_agent_output=edit_result.text,
            round_num=request.round_num,
            model_name=self.model_name,
            target_gpu=request.target_gpu,
            language="ascendc",
        )
        cleaned = {src.path: src.content for src in solution.sources or []}
        candidate_id = f"round_{int(request.round_num):04d}_attempt_{int(request.attempt_idx):02d}"
        snapshot_id = f"{candidate_id}_snapshot"
        task_name = getattr(task, "definition_name", None) or getattr(task, "name", "ascendc")
        artifacts_dir = getattr(task, "artifacts_dir", None)
        snapshot_archive_dir = get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(task_name)) / "snapshots"
        project_snapshot = create_project_snapshot(
            project_dir=wt_session.project_dir,
            snapshot_id=snapshot_id,
            parent_snapshot_id=None,
            base_commit=wt_session.baseline_commit,
            created_by_round=request.round_num,
            eval_result=eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000),
            diff_from_parent=diff_text,
            archive_dir=snapshot_archive_dir,
        )
        candidate_patch, artifact_paths = write_agentic_candidate_artifacts(
            artifacts_dir=artifacts_dir,
            task_name=str(task_name),
            round_num=request.round_num,
            attempt_idx=request.attempt_idx,
            prompt=fix_prompt,
            transcript=edit_result.transcript,
            changed_paths=changed_paths,
            diff_text=diff_text,
            eval_result=eval_result,
            project_snapshot=project_snapshot,
            parent_candidate_id=request.parent_candidate_id,
            base_ref=wt_session.baseline_commit,
            project_rel_path=wt_session.project_rel_path(),
            action_node_id=request.action_node_id,
            model_name=self.model_name,
            metadata={
                "target_gpu": request.target_gpu,
                "mode": request.mode,
                "project_path": str(wt_session.project_dir),
                "evaluator_mutated_project": evaluator_mutated_project,
            },
        )
        return AscendCAgenticCodegenResult(
            solution=solution,
            eval_result=eval_result,
            raw=task.code_for_world_model_from_raw(raw=cleaned, language="ascendc"),
            cleaned=cleaned,
            transcript=edit_result.transcript,
            prompt=fix_prompt,
            prompt_chars=len(fix_prompt),
            changed_paths=changed_paths,
            diff_text=diff_text,
            project_path=str(wt_session.project_dir),
            diff_after_eval=diff_after_eval,
            evaluator_mutated_project=evaluator_mutated_project,
            candidate_patch=candidate_patch,
            project_snapshot=project_snapshot,
            artifact_paths=artifact_paths,
            trace_path=edit_result.trace_path or telemetry_recorder.artifacts.trace_path,
            timeline_path=edit_result.timeline_path or telemetry_recorder.artifacts.timeline_path,
            cost_path=edit_result.cost_path or telemetry_recorder.artifacts.cost_path,
            session_id=edit_result.session_id,
            total_cost_usd=edit_result.total_cost_usd,
            usage=edit_result.usage,
            model_usage=edit_result.model_usage,
            num_turns=edit_result.num_turns,
            duration_ms=edit_result.duration_ms,
            code_map_text=None,
            editor_session=editor_session,
            worktree_session=wt_session,
        )
