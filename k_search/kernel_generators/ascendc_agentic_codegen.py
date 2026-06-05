from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.agentic_candidate_artifacts import write_agentic_candidate_artifacts
from k_search.kernel_generators.agentic_worktree import create_agentic_worktree
from k_search.kernel_generators.candidate_patch import CandidatePatch
from k_search.kernel_generators.claude_assets import NATIVE_HANDOFF_FILES, materialize_claude_project_assets
from k_search.kernel_generators.memory import CODE_MAP, KNOWLEDGE, MemoryStore
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
    ClaudeProjectEditorSession,
)
from k_search.kernel_generators.project_snapshot import ProjectSnapshot, create_project_snapshot
from k_search.kernel_generators.subagent_orchestration import (
    SubagentFlowConfig,
    load_subagent_flows,
    run_configured_subagent_flow,
    supports_configured_subagent_flow,
)
from k_search.tasks.task_base import EvalResult, Solution
from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder
from k_search.utils.path_sanitize import sanitize_worktree_paths
from k_search.utils.paths import get_ksearch_artifacts_dir, get_run_id


AgenticMode = Literal["generate", "action", "debug", "improve"]

DEBUG_EVIDENCE_FILES = {
    "debug_packet.json",
    "debug_log.md",
}

CURATOR_CONTEXT_FILES = set(NATIVE_HANDOFF_FILES) | {KNOWLEDGE.filename} | set(DEBUG_EVIDENCE_FILES)


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
    knowledge_text: str | None = None
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
    subagent_flow: SubagentFlowConfig | None = None,
) -> ClaudeProjectEditResult:
    if subagent_flow is not None and supports_configured_subagent_flow(editor_client):
        return run_configured_subagent_flow(
            editor_client=editor_client,
            project_dir=project_dir,
            base_prompt=prompt,
            flow=subagent_flow,
            telemetry_recorder=telemetry_recorder,
        )
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


def _is_native_handoff_path(path: str) -> bool:
    rel = str(path or "").replace("\\", "/").strip()
    if not rel:
        return True
    if rel.startswith(".claude/"):
        return True
    # KNOWLEDGE.md is a persisted memory file (like CODE_MAP.md), materialized into
    # the worktree for plan/codegen to read. It is not a candidate source artifact.
    if rel == KNOWLEDGE.filename:
        return True
    if rel in DEBUG_EVIDENCE_FILES:
        return True
    return rel in NATIVE_HANDOFF_FILES


def _candidate_changed_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not _is_native_handoff_path(path)]


def _field_value(text: str, field: str) -> str | None:
    prefix = f"{field.lower()}:"
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return stripped[len(prefix) :].strip()
    return None


def _empty_required_fixes(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    return normalized in {"", "[]", "none", "no", "n/a", "null", "false"}


def _validate_review_notes(review_text: str) -> None:
    status = (_field_value(review_text, "status") or "").strip().lower()
    eval_ready = (_field_value(review_text, "eval_ready") or "").strip().lower()
    required_fixes = _field_value(review_text, "required_fixes")
    if status != "ok" or eval_ready != "true" or not _empty_required_fixes(required_fixes):
        raise RuntimeError(
            "Claude native reviewer did not mark candidate eval-ready in REVIEW_NOTES.md "
            f"(status={status or 'missing'}, eval_ready={eval_ready or 'missing'})"
        )


def _flow_handoff_files(flow: SubagentFlowConfig) -> set[str]:
    required: set[str] = set()
    for stage in flow.stages:
        required.update(path for path in stage.required_files if path in NATIVE_HANDOFF_FILES)
    return required or set(NATIVE_HANDOFF_FILES)


def _require_native_handoff_files(project_dir: Path, required_files: set[str] | None = None) -> dict[str, str]:
    required = set(required_files or NATIVE_HANDOFF_FILES)
    missing = [name for name in sorted(required) if not (project_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Claude native subagent flow did not produce required handoff file(s): {', '.join(missing)}")
    handoffs = {
        name: (project_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in sorted(required)
    }
    if "REVIEW_NOTES.md" in required:
        _validate_review_notes(handoffs.get("REVIEW_NOTES.md", ""))
    return handoffs


def _code_map_from_handoffs(handoffs: dict[str, str]) -> str | None:
    text = handoffs.get(CODE_MAP.filename)
    return text if text and text.strip() else None


def _remove_native_handoff_files(project_dir: Path) -> None:
    for name in NATIVE_HANDOFF_FILES:
        (project_dir / name).unlink(missing_ok=True)


def _capture_project_files(project_dir: Path, names: set[str]) -> dict[str, str]:
    captured: dict[str, str] = {}
    for name in sorted(names):
        if name not in CURATOR_CONTEXT_FILES:
            continue
        p = project_dir / name
        if p.is_file():
            captured[name] = p.read_text(encoding="utf-8", errors="replace")
    return captured


def _remove_project_files(project_dir: Path, names: set[str]) -> None:
    for name in sorted(names):
        if name not in CURATOR_CONTEXT_FILES:
            continue
        (project_dir / name).unlink(missing_ok=True)


def _capture_and_remove_non_candidate_files(project_dir: Path) -> dict[str, str]:
    """Remove memory/debug files from candidate worktree while preserving curator context."""
    names = {KNOWLEDGE.filename, *DEBUG_EVIDENCE_FILES}
    captured = _capture_project_files(project_dir, names)
    _remove_project_files(project_dir, names)
    return captured


def _native_metadata() -> dict[str, Any]:
    return {
        "native_claude_agents": True,
        "native_handoff_files": sorted(NATIVE_HANDOFF_FILES),
    }


def _configured_flow_or_default(flow_set: Any, name: str) -> SubagentFlowConfig:
    try:
        return flow_set.get(name)
    except KeyError:
        return flow_set.get()


def _materialize_native_assets_baseline(wt_session: Any) -> None:
    materialize_claude_project_assets(wt_session.project_dir)
    wt_session.commit_all("ksearch native claude assets baseline")


def _materialize_existing_code_map(store: MemoryStore | None, project_dir: Path) -> bool:
    if store is None:
        return False
    return store.materialize(CODE_MAP, project_dir)


def _materialize_existing_knowledge(store: MemoryStore | None, project_dir: Path) -> bool:
    """Copy accumulated KNOWLEDGE.md into the worktree so plan/codegen can read it."""
    if store is None:
        return False
    return store.materialize(KNOWLEDGE, project_dir)


def _curator_enabled() -> bool:
    return os.getenv("KSEARCH_ENABLE_CURATOR", "1").strip().lower() not in {"0", "false", "no", "off"}


def _build_curator_prompt(eval_result: Any, has_knowledge: bool) -> str:
    """Prompt for the post-eval knowledge-curator subagent (single-turn edit_project)."""
    status = getattr(eval_result, "status", "unknown")
    log = ""
    excerpt = getattr(eval_result, "log_excerpt", "")
    if excerpt:
        log = sanitize_worktree_paths(_truncate(excerpt, 4000))
    knowledge_note = (
        "KNOWLEDGE.md already exists at the project root: update it, do not duplicate.\n"
        if has_knowledge
        else "KNOWLEDGE.md does not exist yet: create it only if there is a durable lesson.\n"
    )
    return (
        "Use the knowledge-curator subagent. Do not invoke any other subagent.\n"
        "The framework has finished evaluating this candidate. Distill durable, reusable, "
        "root-cause-level AscendC lessons into KNOWLEDGE.md at the project root.\n"
        f"Evaluation status: {status}\n"
        f"{knowledge_note}"
        "Read REVIEW_NOTES.md and debug_packet.json / debug_log.md at the project root if present. "
        "If there is no lesson worth keeping, write nothing and report files_written: none.\n"
        "The final message must be short: status, files_written, next.\n\n"
        "Evaluation log excerpt:\n"
        f"{log or '(none)'}\n"
    )


def _curator_action_node_id(action_node_id: str | None) -> str:
    base = str(action_node_id or "").strip()
    return f"{base}__curator" if base else "curator"


def _build_curator_telemetry_context(
    *,
    task: Any,
    request: AscendCAgenticCodegenRequest,
    model_name: str,
    flow: str,
) -> TelemetryContext:
    return TelemetryContext(
        run_id=request.run_id,
        task_name=getattr(task, "definition_name", None),
        definition=getattr(task, "definition_name", None),
        flow=flow,
        stage="curator",
        round_index=request.round_num,
        attempt_index=request.attempt_idx,
        action_node_id=_curator_action_node_id(request.action_node_id),
        model_name=model_name,
        provider="claude-agent",
        target_gpu=request.target_gpu,
        language="ascendc",
    )


def _run_curator_after_eval(
    *,
    editor_client: Any,
    project_dir: Path,
    eval_result: Any,
    store: MemoryStore | None,
    context_files: dict[str, str] | None = None,
    telemetry_recorder: Any | None = None,
    telemetry_context: TelemetryContext | None = None,
) -> str | None:
    """Run the knowledge-curator after evaluation; return updated KNOWLEDGE.md text.

    Best-effort: never let curation failure break the codegen result. The gated
    write-back to MemoryStore happens in the caller (only when adopted).
    """
    if not _curator_enabled():
        return None
    with tempfile.TemporaryDirectory(prefix="ksearch_curator_") as tmp:
        curator_dir = Path(tmp).resolve()
        materialize_claude_project_assets(curator_dir)
        for name, text in sorted((context_files or {}).items()):
            if name not in CURATOR_CONTEXT_FILES:
                continue
            target = curator_dir / name
            target.write_text(str(text or ""), encoding="utf-8")
        has_knowledge = (curator_dir / KNOWLEDGE.filename).is_file()
        prompt = _build_curator_prompt(eval_result, has_knowledge)
        owned_telemetry_recorder = None
        if telemetry_recorder is None and telemetry_context is not None:
            owned_telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
            telemetry_recorder = owned_telemetry_recorder
        try:
            editor_client.edit_project(
                project_dir=curator_dir,
                prompt=prompt,
                telemetry_recorder=telemetry_recorder,
            )
        except TypeError as exc:
            if "telemetry_recorder" in str(exc):
                try:
                    editor_client.edit_project(project_dir=curator_dir, prompt=prompt)
                except Exception:
                    pass
        except Exception:
            # Curation is non-critical; swallow and fall back to any file already written.
            pass
        finally:
            if owned_telemetry_recorder is not None:
                owned_telemetry_recorder.close()
        return _read_knowledge(curator_dir)


def _read_knowledge(project_dir: Path) -> str | None:
    p = project_dir / KNOWLEDGE.filename
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    return text if text.strip() else None


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
        code_map_status = "yes" if has_code_map else "no"
        code_map_instruction = (
            "CODE_MAP.md already exists: yes. Read it first and instruct plan/codegen/reviewer to read it before acting. "
            "After editing code, update the affected sections of CODE_MAP.md to keep it accurate.\n"
            if has_code_map
            else "CODE_MAP.md already exists: no. Use the code-reader subagent to create CODE_MAP.md before planning.\n"
        )
        prompt = (
            "You are the main K-Search AscendC orchestration agent working inside a candidate project directory.\n"
            "IMPORTANT: You must ONLY edit files inside the current project directory (CWD). Do NOT use absolute paths from external directories.\n"
            f"Target GPU: {request.target_gpu}\n"
            f"Mode: {request.mode}\n"
            f"Round: {int(request.round_num)}\n"
            f"Attempt: {int(request.attempt_idx)}\n"
            f"CODE_MAP.md already exists: {code_map_status}\n\n"
            "Available tools: Read/Grep/Glob/Edit/Write, Skill, and Agent. Bash is disabled.\n"
            "Use the ascendc-codegen and ascendc-api-reference skills when relevant.\n"
            "Required native subagent flow: code-reader -> plan -> codegen -> reviewer.\n"
            "The bug-fixer subagent is reserved for future eval-failure repair and must not be invoked in this release.\n"
            + code_map_instruction
            + "The plan subagent must write IMPLEMENTATION_PLAN.md.\n"
            "The reviewer subagent must write REVIEW_NOTES.md.\n"
            "CODE_MAP.md, IMPLEMENTATION_PLAN.md, and REVIEW_NOTES.md are the only trusted cross-subagent handoff.\n"
            "Every subagent final message must be short and contain only status, files_written, and next.\n"
            "Do not paste CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md, or source files into final messages.\n"
            + "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
            "Preserve operator semantics, public entry points, host tiling contract, correctness harness behavior, and build layout.\n"
            "End with a concise summary and changed-file list after reviewer says eval_ready is true.\n\n"
            "Task specification:\n"
            f"{sections['definition']}\n\n"
            "Chosen strategy/action/debug intent:\n"
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
        subagent_flow: SubagentFlowConfig | None = None,
        repair_subagent_flow: SubagentFlowConfig | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.editor_client = editor_client or ClaudeAgentProjectEditorClient(model_name=self.model_name)
        self.reader_editor_client = reader_editor_client
        self.prompt_builder = prompt_builder or AscendCAgenticPromptBuilder()
        flow_set = load_subagent_flows()
        self.subagent_flow = subagent_flow or _configured_flow_or_default(flow_set, "initial_codegen")
        try:
            default_repair_flow = flow_set.get("eval_failure_repair")
        except KeyError:
            default_repair_flow = self.subagent_flow
        self.repair_subagent_flow = repair_subagent_flow or default_repair_flow

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
            _materialize_native_assets_baseline(session)

            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = _materialize_existing_code_map(store, session.project_dir)
            _materialize_existing_knowledge(store, session.project_dir)

            prompt = self.prompt_builder.build(
                request,
                has_code_map=has_code_map,
                task_path=str(getattr(task, "task_path", "") or ""),
            )
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
                edit_result = _edit_project_with_optional_telemetry(
                    self.editor_client,
                    project_dir=session.project_dir,
                    prompt=prompt,
                    telemetry_recorder=telemetry_recorder,
                    subagent_flow=self.subagent_flow,
                )
            finally:
                telemetry_recorder.close()
            handoff_texts = _require_native_handoff_files(session.project_dir, _flow_handoff_files(self.subagent_flow))
            code_map_text = _code_map_from_handoffs(handoff_texts)
            if store is not None and code_map_text:
                store.save(CODE_MAP, code_map_text)
            _remove_native_handoff_files(session.project_dir)
            curator_context = dict(handoff_texts)
            curator_context.update(_capture_and_remove_non_candidate_files(session.project_dir))
            project_changed_paths = session.project_changed_paths()
            changed_paths = _candidate_changed_paths(project_changed_paths or session.changed_paths())
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
                    changed_paths = _candidate_changed_paths(project_changed_paths or session.changed_paths())
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
            knowledge_text = _run_curator_after_eval(
                editor_client=self.editor_client,
                project_dir=session.project_dir,
                eval_result=eval_result,
                store=store,
                context_files=curator_context,
                telemetry_context=_build_curator_telemetry_context(
                    task=task,
                    request=request,
                    model_name=self.model_name,
                    flow="agentic_codegen",
                ),
            )
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
                handoff_files=handoff_texts,
                metadata={
                    "target_gpu": request.target_gpu,
                    "mode": request.mode,
                    "project_path": str(session.project_dir),
                    "evaluator_mutated_project": evaluator_mutated_project,
                    **_native_metadata(),
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
                knowledge_text=knowledge_text,
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
        # The native bug-fixer subagent is reserved but not active in this release.
        # Keep retry ownership in the caller/world-model cycle instead of running
        # hidden generic fix turns inside this method.
        max_fix_rounds = 0
        wt_session = create_agentic_worktree(task_path=getattr(task, "task_path", None))
        editor_session: ClaudeProjectEditorSession | None = None
        try:
            overlay = getattr(task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=wt_session.project_dir, solution=base_solution)
                wt_session.commit_all("ksearch agentic overlay baseline")
            _materialize_native_assets_baseline(wt_session)

            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0", "false", "no", "off",
            }
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = _materialize_existing_code_map(store, wt_session.project_dir)
            _materialize_existing_knowledge(store, wt_session.project_dir)

            # Open a multi-turn SDK session
            editor_session = self.editor_client.open_session(
                project_dir=wt_session.project_dir,
            )

            # Attempt 1: send full prompt
            prompt = self.prompt_builder.build(
                request,
                has_code_map=has_code_map,
                task_path=str(getattr(task, "task_path", "") or ""),
            )
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
                edit_result = run_configured_subagent_flow(
                    editor_client=self.editor_client,
                    project_dir=wt_session.project_dir,
                    base_prompt=prompt,
                    flow=self.subagent_flow,
                    telemetry_recorder=telemetry_recorder,
                    session=editor_session,
                    close_session_on_exit=False,
                )
            finally:
                telemetry_recorder.close()

            handoff_texts = _require_native_handoff_files(wt_session.project_dir, _flow_handoff_files(self.subagent_flow))
            code_map_text = _code_map_from_handoffs(handoff_texts)
            if store is not None and code_map_text:
                store.save(CODE_MAP, code_map_text)
            _remove_native_handoff_files(wt_session.project_dir)
            curator_context = dict(handoff_texts)
            curator_context.update(_capture_and_remove_non_candidate_files(wt_session.project_dir))

            project_changed_paths = wt_session.project_changed_paths()
            changed_paths = _candidate_changed_paths(project_changed_paths or wt_session.changed_paths())
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
                    changed_paths = _candidate_changed_paths(project_changed_paths or wt_session.changed_paths())
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

                _materialize_existing_code_map(store, wt_session.project_dir)
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
                    edit_result = run_configured_subagent_flow(
                        editor_client=self.editor_client,
                        project_dir=wt_session.project_dir,
                        base_prompt=fix_prompt,
                        flow=self.repair_subagent_flow,
                        telemetry_recorder=fix_telemetry_recorder,
                        session=editor_session,
                        close_session_on_exit=False,
                    )
                finally:
                    fix_telemetry_recorder.close()
                fix_handoff_texts = _require_native_handoff_files(
                    wt_session.project_dir,
                    _flow_handoff_files(self.repair_subagent_flow),
                )
                produced_code_map = _code_map_from_handoffs(fix_handoff_texts)
                if store is not None and produced_code_map:
                    code_map_text = produced_code_map
                    store.save(CODE_MAP, produced_code_map)
                handoff_texts = fix_handoff_texts
                _remove_native_handoff_files(wt_session.project_dir)
                curator_context = dict(fix_handoff_texts)
                curator_context.update(_capture_and_remove_non_candidate_files(wt_session.project_dir))

                # Check for changes after fix
                project_changed_paths = wt_session.project_changed_paths()
                changed_paths = _candidate_changed_paths(project_changed_paths or wt_session.changed_paths())
                if not changed_paths:
                    break  # LLM didn't change anything, stop trying

                # Re-evaluate
                eval_result = run_in_project_dir(project_dir=wt_session.project_dir, round_num=request.round_num)
                diff_after_eval = wt_session.project_diff_text()
                evaluator_mutated_project = diff_after_eval != diff_text

            # Build final result (same as run())
            knowledge_text = _run_curator_after_eval(
                editor_client=self.editor_client,
                project_dir=wt_session.project_dir,
                eval_result=eval_result,
                store=store,
                context_files=curator_context,
                telemetry_context=_build_curator_telemetry_context(
                    task=task,
                    request=request,
                    model_name=self.model_name,
                    flow="agentic_codegen_multi_turn",
                ),
            )
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
                handoff_files=handoff_texts,
                metadata={
                    "target_gpu": request.target_gpu,
                    "mode": request.mode,
                    "project_path": str(wt_session.project_dir),
                    "evaluator_mutated_project": evaluator_mutated_project,
                    **_native_metadata(),
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
                knowledge_text=knowledge_text,
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
        code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }
        store = MemoryStore.for_task(task) if code_map_enabled else None
        has_code_map = _materialize_existing_code_map(store, wt_session.project_dir)
        _materialize_existing_knowledge(store, wt_session.project_dir)
        fix_prompt = sanitize_worktree_paths(fix_prompt)
        action_with_fix_context = (
            f"{request.action_text}\n\nFix context from previous evaluation:\n{fix_prompt}"
        ).strip()
        native_request = AscendCAgenticCodegenRequest(
            definition_text=request.definition_text,
            action_text=action_with_fix_context,
            trace_logs=request.trace_logs,
            perf_summary=request.perf_summary,
            target_gpu=request.target_gpu,
            round_num=request.round_num,
            attempt_idx=request.attempt_idx,
            mode=request.mode,
            run_id=request.run_id,
            task_name=request.task_name,
            parent_candidate_id=request.parent_candidate_id,
            action_node_id=request.action_node_id,
        )
        prompt = self.prompt_builder.build(
            native_request,
            has_code_map=has_code_map,
            task_path=str(getattr(task, "task_path", "") or ""),
        )
        prompt = sanitize_worktree_paths(prompt)
        task_path = getattr(task, "task_path", None)
        if task_path is not None:
            prompt = prompt.replace(str(Path(task_path).expanduser().resolve()), "<PROJECT_ROOT>")

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
        telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
        try:
            edit_result = run_configured_subagent_flow(
                editor_client=self.editor_client,
                project_dir=wt_session.project_dir,
                base_prompt=prompt,
                flow=self.repair_subagent_flow,
                telemetry_recorder=telemetry_recorder,
                session=editor_session,
                close_session_on_exit=False,
            )
        finally:
            telemetry_recorder.close()

        handoff_texts = _require_native_handoff_files(
            wt_session.project_dir,
            _flow_handoff_files(self.repair_subagent_flow),
        )
        code_map_text = _code_map_from_handoffs(handoff_texts)
        if store is not None and code_map_text:
            store.save(CODE_MAP, code_map_text)
        _remove_native_handoff_files(wt_session.project_dir)
        curator_context = dict(handoff_texts)
        curator_context.update(_capture_and_remove_non_candidate_files(wt_session.project_dir))

        project_changed_paths = wt_session.project_changed_paths()
        changed_paths = _candidate_changed_paths(project_changed_paths or wt_session.changed_paths())
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

        knowledge_text = _run_curator_after_eval(
            editor_client=self.editor_client,
            project_dir=wt_session.project_dir,
            eval_result=eval_result,
            store=store,
            context_files=curator_context,
            telemetry_context=_build_curator_telemetry_context(
                task=task,
                request=request,
                model_name=self.model_name,
                flow="agentic_codegen_multi_turn",
            ),
        )
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
            handoff_files=handoff_texts,
            metadata={
                "target_gpu": request.target_gpu,
                "mode": request.mode,
                "project_path": str(wt_session.project_dir),
                "evaluator_mutated_project": evaluator_mutated_project,
                **_native_metadata(),
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
            knowledge_text=knowledge_text,
            editor_session=editor_session,
            worktree_session=wt_session,
        )
