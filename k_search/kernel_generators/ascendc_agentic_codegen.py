from __future__ import annotations

import json
import os
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.agentic_candidate_artifacts import (
    get_agentic_candidate_artifact_dir,
    write_agentic_candidate_artifacts,
)
from k_search.kernel_generators.agentic_worktree import create_agentic_worktree
from k_search.kernel_generators.candidate_patch import CandidatePatch
from k_search.kernel_generators.claude_assets import materialize_claude_project_assets
from k_search.kernel_generators.memory import CODE_MAP, KNOWLEDGE, MemoryStore
from k_search.kernel_generators.code_map_lineage import (
    CodeMapReuseContext,
    evaluate_code_map_reuse,
)
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
    ClaudeProjectEditorSession,
)
from k_search.kernel_generators.project_snapshot import ProjectSnapshot, create_project_snapshot
from k_search.kernel_generators.stage_prompt_artifacts import StagePromptSink
from k_search.kernel_generators.runtime_artifacts import (
    NATIVE_DEBUG_EVIDENCE_FILES,
    NATIVE_HANDOFF_FILES,
    NATIVE_RUNTIME_DIRS,
    NATIVE_RUNTIME_FILES,
    is_native_runtime_path,
)
from k_search.kernel_generators.subagent_orchestration import (
    SubagentFlowConfig,
    load_subagent_flows,
    run_configured_subagent_flow,
    supports_configured_subagent_flow,
)
from k_search.kernel_generators.eval_context import build_eval_context_for_llm
from k_search.kernel_generators.worktree_context import (
    WorktreeContextPaths,
    assert_no_absolute_paths_for_llm,
    materialize_worktree_context,
)
from k_search.tasks.task_base import EvalResult, Solution
from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder
from k_search.utils.path_sanitize import sanitize_worktree_paths
from k_search.utils.paths import get_ksearch_artifacts_dir, get_ksearch_worktrees_dir, get_run_id


logger = logging.getLogger(__name__)

AgenticMode = Literal["generate", "action", "debug", "improve"]

DEBUG_EVIDENCE_FILES = set(NATIVE_DEBUG_EVIDENCE_FILES)

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
    eval_result: EvalResult | None = None
    canonical_strategy_markdown_path: Path | None = None
    strategy_summary: str | None = None
    eval_summary: dict[str, Any] | None = None
    eval_log: str | None = None
    context_paths: WorktreeContextPaths | None = None
    strategy_context: dict[str, Any] | None = None
    blocked_strategy_nodes: list[dict[str, Any]] | None = None


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
    eval_project_path: str | None = None
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


def _truncate(text: str, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 40)].rstrip() + "\n[truncated for agentic prompt budget]"


def _extract_bounded_strategy_summary(action_text: str, *, limit: int = 1200) -> str:
    text = str(action_text or "").strip()
    if not text:
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Full natural-language strategy markdown:":
            break
        if stripped.startswith("Full natural-language strategy markdown:"):
            break
        if stripped == "Referenced strategy document:":
            break
        lines.append(line)
    summary = "\n".join(lines).strip() or text[:limit].strip()
    if len(summary) > limit:
        summary = summary[: max(0, limit - 24)].rstrip() + "\n[summary truncated]"
    return summary


def _render_strategy_dependency_status(strategy_context: dict[str, Any] | None) -> str:
    if not isinstance(strategy_context, dict) or not strategy_context:
        return ""
    strategy_id = str(strategy_context.get("strategy_id") or "").strip()
    requires = strategy_context.get("requires") or []
    if isinstance(requires, str):
        requires_list = [requires]
    elif isinstance(requires, (list, tuple)):
        requires_list = [str(item).strip() for item in requires if str(item).strip()]
    else:
        requires_list = []
    satisfied = bool(strategy_context.get("dependencies_satisfied", False))
    parent_solution_id = str(strategy_context.get("parent_solution_id") or "").strip()
    parent_lineage = strategy_context.get("parent_strategy_lineage") or []
    if isinstance(parent_lineage, str):
        parent_lineage_list = [parent_lineage]
    elif isinstance(parent_lineage, (list, tuple)):
        parent_lineage_list = [str(item).strip() for item in parent_lineage if str(item).strip()]
    else:
        parent_lineage_list = []
    lines = [
        "Dependency check:",
        f"- strategy_id: {strategy_id or '(unknown)'}",
        f"- requires: {', '.join(requires_list) if requires_list else '(none)'}",
        f"- dependency_status: {'satisfied' if satisfied else 'unsatisfied'}",
    ]
    if parent_lineage_list:
        lines.append(f"- parent_strategy_lineage: {', '.join(parent_lineage_list)}")
    if parent_solution_id:
        lines.append(f"- parent_solution_id: {parent_solution_id}")
    return "\n".join(lines)


def _render_eval_summary_for_prompt(eval_summary: dict[str, Any] | None, *, max_chars: int = 1800) -> str:
    if not isinstance(eval_summary, dict):
        return "(none)"
    payload: dict[str, Any] = {}
    for key in (
        "eval_context_status",
        "has_prior_candidate_eval",
        "has_eval_log",
        "compile_passed",
        "correctness_passed",
        "performance_available",
        "diagnostic_kind",
        "baseline",
        "performance",
        "message_for_llm",
    ):
        if key in eval_summary:
            payload[key] = eval_summary[key]
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _truncate(text, max_chars)


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


def _wrap_repair_prompt(fix_prompt: str) -> str:
    fix_text = sanitize_worktree_paths(str(fix_prompt or "").strip())
    header = (
        "This is an eval_failure_repair attempt. Follow the configured repair flow exactly.\n"
        "If the active stage is bug-fixer, invoke the bug-fixer subagent exactly once.\n"
        "Do not perform bug fixing in the parent agent context."
    )
    if fix_text.startswith("This is an eval_failure_repair attempt."):
        return fix_text
    return f"{header}\n\n{fix_text}".strip()


def _build_repair_prompt(eval_result: EvalResult, fix_round: int, max_chars: int = 6000) -> str:
    return _wrap_repair_prompt(_build_fix_prompt(eval_result, fix_round=fix_round, max_chars=max_chars))


def _render_legacy_single_agent_prompt(base_prompt: str, flow: SubagentFlowConfig | None) -> str:
    if flow is None:
        return str(base_prompt or "")
    stage_names = " -> ".join(stage.agent for stage in flow.stages)
    return (
        "Legacy single-agent compatibility mode is explicitly enabled by "
        "KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW.\n"
        f"Required native subagent flow: {stage_names}.\n"
        "If CODE_MAP.md is missing, Use the code-reader subagent to create CODE_MAP.md before detailed design.\n"
        "Run these responsibilities yourself in order and write all required handoff files.\n\n"
        f"{str(base_prompt or '').strip()}"
    ).strip()


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
    if subagent_flow is not None:
        if not _env_truthy("KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW"):
            raise RuntimeError("Configured subagent flow is required in native subagent mode")
        prompt = _render_legacy_single_agent_prompt(prompt, subagent_flow)
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
    return is_native_runtime_path(path)


def _candidate_changed_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not _is_native_handoff_path(path)]


def _path_from_diff_header(line: str) -> str | None:
    text = str(line or "")
    if text.startswith("diff --git "):
        parts = text.split()
        if len(parts) >= 4:
            path = parts[3]
            return path[2:] if path.startswith("b/") else path
    if text.startswith("+++ b/"):
        return text[len("+++ b/") :].strip()
    return None


def _candidate_diff_text(diff_text: str) -> str:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in str(diff_text or "").splitlines():
        starts_block = line.startswith("diff --git ") or line.startswith("--- /dev/null")
        if starts_block and current:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)

    kept: list[str] = []
    for block in blocks:
        path = None
        for line in block[:6]:
            candidate = _path_from_diff_header(line)
            if candidate:
                path = candidate
        if path and is_native_runtime_path(path):
            continue
        kept.extend(block)
    return "\n".join(kept)


def _markdown_field_candidate(line: str) -> str:
    candidate = str(line or "").strip().lstrip("#").strip()
    for bullet in ("- ", "* "):
        if candidate.startswith(bullet):
            return candidate[2:].strip()
    return candidate


def _split_field_candidate(candidate: str, field: str) -> tuple[bool, str | None]:
    match = re.fullmatch(
        rf"{re.escape(field)}\s*(?:(:|=)\s*(.*))?",
        str(candidate or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return False, None
    if match.group(1) is None:
        return True, None
    return True, (match.group(2) or "").strip()


def _strip_wrapping_scalar_quotes(value: str) -> str:
    stripped = str(value or "").strip()
    if len(stripped) >= 2 and stripped[0] in {"'", '"', "`"} and stripped[-1] == stripped[0]:
        return stripped[1:-1].strip()
    return stripped


def _is_fence_line(stripped: str) -> bool:
    return stripped.startswith("```") or stripped.startswith("~~~")


def _looks_like_field_assignment(candidate: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*[:=]", str(candidate or "")))


def _following_field_value(lines: list[str], start: int) -> str | None:
    in_fence = False
    for line in lines[start:]:
        stripped = str(line or "").strip()
        if _is_fence_line(stripped):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        candidate = _markdown_field_candidate(stripped)
        if stripped.startswith("#") or _looks_like_field_assignment(candidate):
            return None
        return _strip_wrapping_scalar_quotes(candidate)
    return None


def _field_value(text: str, field: str) -> str | None:
    lines = str(text or "").splitlines()
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _is_fence_line(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        candidate = _markdown_field_candidate(stripped)
        matched, value = _split_field_candidate(candidate, field)
        if matched:
            if value:
                return _strip_wrapping_scalar_quotes(value)
            return _following_field_value(lines, index + 1)
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


def _strict_handoff_validation() -> bool:
    return os.getenv("KSEARCH_STRICT_HANDOFF_VALIDATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fallback_agentic_task_name(task: Any) -> str:
    return str(
        getattr(task, "name", "")
        or getattr(task, "definition_name", "")
        or "ascendc"
    ).strip() or "ascendc"


def _resolve_agentic_run_context(
    *,
    request: AscendCAgenticCodegenRequest,
    task: Any,
) -> tuple[str, str]:
    request_run_id = str(request.run_id or "").strip()
    request_task_name = str(request.task_name or "").strip()
    missing = []
    if not request_run_id:
        missing.append("run_id")
    if not request_task_name:
        missing.append("task_name")
    if missing and not _env_truthy("KSEARCH_ALLOW_MISSING_AGENTIC_RUN_CONTEXT"):
        raise RuntimeError(
            "AscendC agentic request missing "
            + ", ".join(missing)
            + "; pass explicit run_id and task_name, or set "
            "KSEARCH_ALLOW_MISSING_AGENTIC_RUN_CONTEXT=1 for temporary compatibility"
        )
    if missing:
        logger.warning(
            "AscendC agentic request missing %s; falling back because "
            "KSEARCH_ALLOW_MISSING_AGENTIC_RUN_CONTEXT=1",
            ", ".join(missing),
        )
    task_name = request_task_name or _fallback_agentic_task_name(task)
    run_id = request_run_id or get_run_id()
    return task_name, run_id


def _validate_code_map(text: str) -> None:
    if len(str(text or "").strip()) < 100:
        raise RuntimeError("CODE_MAP.md is too short to be useful")


def _validate_ascendc_design(text: str) -> None:
    if len(str(text or "").strip()) < 100:
        raise RuntimeError("ASCENDC_DESIGN.md is too short")


def _validate_execution_plan(text: str) -> None:
    if len(str(text or "").strip()) < 50:
        raise RuntimeError("IMPLEMENTATION_EXECUTION_PLAN.md is too short")


def _validate_implementation_handoff(text: str) -> None:
    if len(str(text or "").strip()) < 50:
        raise RuntimeError("IMPLEMENTATION_HANDOFF.md is too short")


def _validate_optional_handoff(text: str, validator: Any) -> None:
    try:
        validator(text)
    except Exception as exc:
        if _strict_handoff_validation():
            raise
        logger.warning(str(exc))


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
    present_optional = {name for name in NATIVE_HANDOFF_FILES - required if (project_dir / name).is_file()}
    collected = required | present_optional
    handoffs = {
        name: (project_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in sorted(collected)
    }
    if "CODE_MAP.md" in required:
        _validate_optional_handoff(handoffs.get("CODE_MAP.md", ""), _validate_code_map)
    if "ASCENDC_DESIGN.md" in required:
        _validate_optional_handoff(handoffs.get("ASCENDC_DESIGN.md", ""), _validate_ascendc_design)
    if "IMPLEMENTATION_EXECUTION_PLAN.md" in required:
        _validate_optional_handoff(
            handoffs.get("IMPLEMENTATION_EXECUTION_PLAN.md", ""),
            _validate_execution_plan,
        )
    if "IMPLEMENTATION_HANDOFF.md" in required:
        _validate_optional_handoff(handoffs.get("IMPLEMENTATION_HANDOFF.md", ""), _validate_implementation_handoff)
    if "REVIEW_NOTES.md" in required:
        _validate_review_notes(handoffs.get("REVIEW_NOTES.md", ""))
    return handoffs


def _code_map_from_handoffs(handoffs: dict[str, str]) -> str | None:
    text = handoffs.get(CODE_MAP.filename)
    return text if text and text.strip() else None


def _remove_native_handoff_files(project_dir: Path) -> None:
    for name in NATIVE_HANDOFF_FILES:
        (project_dir / name).unlink(missing_ok=True)


def _write_runtime_file(project_dir: Path, name: str, text: str | None) -> bool:
    if name not in NATIVE_RUNTIME_FILES:
        raise ValueError(f"not a native runtime file: {name}")
    if not text or not str(text).strip():
        return False
    target = project_dir / name
    target.write_text(str(text), encoding="utf-8")
    return True


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


def _copy_project_for_eval(candidate_dir: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmp = tempfile.TemporaryDirectory(prefix="ksearch_eval_")
    eval_dir = Path(tmp.name).resolve() / "project"
    ignore = shutil.ignore_patterns(
        ".git",
        ".claude",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "logs",
        "llm_logs",
        *NATIVE_RUNTIME_DIRS,
        *NATIVE_RUNTIME_FILES,
    )
    shutil.copytree(candidate_dir, eval_dir, symlinks=False, ignore=ignore)
    return tmp, eval_dir


def _capture_eval_debug_evidence(eval_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in DEBUG_EVIDENCE_FILES:
        p = eval_dir / name
        if p.is_file():
            out[name] = p.read_text(encoding="utf-8", errors="replace")
    return out


def _run_eval_in_isolated_copy(
    *,
    task: Any,
    candidate_project_dir: Path,
    round_num: int,
) -> tuple[EvalResult, str | None]:
    run_in_project_dir = getattr(task, "run_benchmark_in_project_dir", None)
    if not callable(run_in_project_dir):
        raise RuntimeError("AscendC agentic task does not support run_benchmark_in_project_dir")
    tmp, eval_dir = _copy_project_for_eval(candidate_project_dir)
    keep_eval_dir = os.getenv("KSEARCH_KEEP_EVAL_WORKDIRS", "").strip().lower() in {"1", "true", "yes", "on"}
    try:
        eval_result = run_in_project_dir(project_dir=eval_dir, round_num=round_num)
        setattr(eval_result, "_ksearch_debug_evidence", _capture_eval_debug_evidence(eval_dir))
        return eval_result, str(eval_dir)
    finally:
        if keep_eval_dir:
            try:
                tmp._finalizer.detach()  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            tmp.cleanup()


def _native_metadata() -> dict[str, Any]:
    return {
        "native_claude_agents": True,
        "native_handoff_files": sorted(NATIVE_HANDOFF_FILES),
        "native_runtime_files": sorted(NATIVE_RUNTIME_FILES),
    }


def _configured_flow_or_default(flow_set: Any, name: str) -> SubagentFlowConfig:
    try:
        return flow_set.get(name)
    except KeyError:
        return flow_set.get()


def _materialize_native_assets_baseline(wt_session: Any) -> None:
    materialize_claude_project_assets(wt_session.project_dir)
    wt_session.commit_all("ksearch native claude assets baseline")


def _code_map_reuse_context_for_request(request: AscendCAgenticCodegenRequest) -> CodeMapReuseContext:
    strategy_context = request.strategy_context if isinstance(request.strategy_context, dict) else {}
    parent_solution_id = str(strategy_context.get("parent_solution_id") or "").strip() or None
    parent_branch_id = (
        str(strategy_context.get("parent_branch_id") or strategy_context.get("parent_action_node_id") or "").strip()
        or None
    )
    return CodeMapReuseContext(
        mode="action",
        parent_solution_id=parent_solution_id,
        parent_branch_id=parent_branch_id,
        branch_id=str(strategy_context.get("action_node_id") or request.action_node_id or "").strip() or None,
    )


def _materialize_existing_code_map(
    store: MemoryStore | None,
    project_dir: Path,
    *,
    reuse_context: CodeMapReuseContext | None = None,
) -> bool:
    if store is None:
        return False
    return store.materialize(CODE_MAP, project_dir, code_map_reuse_context=reuse_context)


def _materialize_existing_knowledge(store: MemoryStore | None, project_dir: Path) -> bool:
    """Copy accumulated KNOWLEDGE.md into the worktree so designer/codegen can read it."""
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

    def build(
        self,
        request: AscendCAgenticCodegenRequest,
        *,
        has_code_map: bool = False,
        task_path: str | None = None,
    ) -> str:
        sections = {
            "definition": _truncate(request.definition_text, 5000),
            "action": _truncate(request.action_text, 3000),
            "perf_summary": _truncate(request.perf_summary, 2500),
            "trace_logs": _truncate(request.trace_logs, 4000),
        }
        context_paths = request.context_paths
        strategy_summary = _truncate(
            request.strategy_summary or _extract_bounded_strategy_summary(request.action_text),
            1200,
        )
        rendered_eval_summary = _render_eval_summary_for_prompt(request.eval_summary)
        code_map_status = "yes" if has_code_map else "no"
        code_map_instruction = (
            "CODE_MAP.md already exists: yes. Read it first in current stages that need project structure. "
            "After editing code, update the affected sections of CODE_MAP.md to keep it accurate.\n"
            if has_code_map
            else "CODE_MAP.md already exists: no. The configured stage prompt will create CODE_MAP.md when needed.\n"
        )
        if context_paths is not None:
            dependency_status = _render_strategy_dependency_status(request.strategy_context)
            dependency_block = f"{dependency_status}\n\n" if dependency_status else ""
            context_block = (
                "K-Search context files, relative to the candidate project root:\n"
                f"- Strategy document: {context_paths.strategy_md}\n"
                f"- Strategy summary: {context_paths.strategy_summary_md}\n"
                f"- Evaluation summary: {context_paths.eval_summary_json}\n"
                f"- Evaluation log: {context_paths.eval_log_md}\n"
                f"- Context manifest: {context_paths.manifest_json}\n\n"
                "Required reads before design/code edits:\n"
                f"- Read {context_paths.strategy_md}.\n"
                f"- Read {context_paths.eval_summary_json}.\n"
                f"- Read {context_paths.eval_log_md} only if EVAL_SUMMARY.json has has_eval_log=true.\n"
                "- Do not inspect raw evaluation logs.\n"
                "- Do not paste these files into final messages.\n"
                "- Do not use absolute paths.\n\n"
                "Strategy summary:\n"
                f"{strategy_summary or '(none)'}\n\n"
                f"{dependency_block}"
                "Evaluation summary:\n"
                f"{rendered_eval_summary or '(none)'}\n"
            )
            action_block = context_block
            perf_block = "(see .ksearch/context/EVAL_SUMMARY.json)"
            trace_block = "(see .ksearch/context/EVAL_LOG.md only when has_eval_log=true)"
        else:
            action_block = sections["action"]
            perf_block = sections["perf_summary"] or "(none)"
            trace_block = sections["trace_logs"] or "(none)"
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
            + code_map_instruction
            + "Runtime handoff files include CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md, optional IMPLEMENTATION_DEVIATIONS.md, and REVIEW_NOTES.md.\n"
            "Agent final messages must be short and contain only status, files_written, and next.\n"
            "Do not paste CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md, IMPLEMENTATION_DEVIATIONS.md, REVIEW_NOTES.md, or source files into final messages.\n"
            + "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
            "Preserve operator semantics, public entry points, host tiling contract, correctness harness behavior, and build layout.\n"
            "End with a concise summary and changed-file list after reviewer says eval_ready is true.\n\n"
            "Task specification:\n"
            f"{sections['definition']}\n\n"
            "Chosen strategy/action/debug intent:\n"
            f"{action_block}\n\n"
            "Performance summary:\n"
            f"{perf_block}\n\n"
            "Evaluation diagnostic context:\n"
            f"{trace_block}\n"
        )
        # 不变量:送达 LLM 的文本不得携带物理路径(worktree 或原始任务目录),统一抹成语义占位符。
        prompt = sanitize_worktree_paths(prompt, task_path=task_path)
        if context_paths is not None:
            assert_no_absolute_paths_for_llm(prompt)
            assert_no_absolute_paths_for_llm(strategy_summary)
            assert_no_absolute_paths_for_llm(rendered_eval_summary)
        if len(prompt) > self.max_chars:
            sizes = ", ".join(f"{name}={len(value)}" for name, value in sorted(sections.items()))
            raise ValueError(
                f"agentic prompt exceeded {self.max_chars} chars: prompt={len(prompt)}, sections: {sizes}"
            )
        return prompt


class AscendCAgenticCycle:
    """Own one candidate worktree and, when available, one Claude SDK session."""

    def __init__(
        self,
        *,
        runner: "AscendCAgenticCodegenRunner",
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
    ) -> None:
        self.runner = runner
        self.task = task
        self.request = request
        self.task_name, self.run_id = _resolve_agentic_run_context(request=request, task=task)
        self.base_solution = base_solution
        self.wt_session: Any | None = None
        self.editor_session: ClaudeProjectEditorSession | Any | None = None
        self.store: MemoryStore | None = None
        self.has_code_map: bool = False
        self.code_map_text: str | None = None
        self.curator_context: dict[str, str] = {}
        self.last_handoff_texts: dict[str, str] = {}
        self.last_edit_result: ClaudeProjectEditResult | None = None
        self.code_map_reuse_manifest: dict[str, Any] = {"reused": False, "reason": "not_checked"}
        self.last_stage_prompt_records: list[dict[str, Any]] = []
        self._closed = False
        self._initial_has_run = False

    def __enter__(self) -> "AscendCAgenticCycle":
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False

    def _open(self) -> None:
        if self.wt_session is not None:
            return
        try:
            self.wt_session = create_agentic_worktree(
                task_path=getattr(self.task, "task_path", None),
                worktree_parent_dir=get_ksearch_worktrees_dir(
                    base_dir=getattr(self.task, "artifacts_dir", None),
                    task_name=self.task_name,
                    run_id=self.run_id,
                ),
            )
            overlay = getattr(self.task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=self.wt_session.project_dir, solution=self.base_solution)
                self.wt_session.commit_all("ksearch agentic overlay baseline")
            _materialize_native_assets_baseline(self.wt_session)

            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            self.store = MemoryStore.for_task(self.task) if code_map_enabled else None
            reuse_context = _code_map_reuse_context_for_request(self.request)
            meta = self.store.load_meta(CODE_MAP) if self.store is not None else None
            decision = evaluate_code_map_reuse(meta, reuse_context) if self.store is not None else None
            self.code_map_reuse_manifest = (
                decision.to_manifest()
                if decision is not None
                else {"reused": False, "reason": "code_map_disabled"}
            )
            self.has_code_map = _materialize_existing_code_map(
                self.store,
                self.wt_session.project_dir,
                reuse_context=reuse_context,
            )
            _materialize_existing_knowledge(self.store, self.wt_session.project_dir)

            if supports_configured_subagent_flow(self.runner.editor_client):
                self.editor_session = self.runner.editor_client.open_session(
                    project_dir=self.wt_session.project_dir,
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        if self.editor_session is not None:
            try:
                if not bool(getattr(self.editor_session, "_closed", False)):
                    self.runner.editor_client.close_session(self.editor_session)
            except BaseException as exc:
                errors.append(exc)
            finally:
                self.editor_session = None
        if self.wt_session is not None:
            try:
                self.wt_session.cleanup()
            except BaseException as exc:
                errors.append(exc)
            finally:
                self.wt_session = None
        if errors:
            import logging

            log = logging.getLogger(__name__)
            for exc in errors:
                log.warning("failed to cleanup Claude agentic cycle resource: %s", exc)

    def _require_open(self) -> None:
        if self._closed or self.wt_session is None:
            raise RuntimeError("AscendCAgenticCycle is not open")

    def _task_path_text(self) -> str:
        return str(getattr(self.task, "task_path", "") or "")

    def _sanitize_prompt(self, prompt: str) -> str:
        prompt = sanitize_worktree_paths(prompt)
        task_path = getattr(self.task, "task_path", None)
        if task_path is not None:
            prompt = prompt.replace(str(Path(task_path).expanduser().resolve()), "<PROJECT_ROOT>")
        assert_no_absolute_paths_for_llm(prompt)
        return prompt

    def _telemetry_context(self, *, stage: str, extra: dict[str, Any] | None = None) -> TelemetryContext:
        return TelemetryContext(
            run_id=self.run_id,
            task_name=self.task_name,
            definition=getattr(self.task, "definition_name", None) or self.task_name,
            flow="agentic_codegen_multi_turn",
            stage=stage,
            round_index=self.request.round_num,
            attempt_index=self.request.attempt_idx,
            action_node_id=self.request.action_node_id,
            model_name=self.runner.model_name,
            provider="claude-agent",
            target_gpu=self.request.target_gpu,
            language="ascendc",
            extra=extra,
        )

    def _stage_prompt_sink(self) -> StagePromptSink:
        return StagePromptSink(
            get_agentic_candidate_artifact_dir(
                artifacts_dir=getattr(self.task, "artifacts_dir", None),
                task_name=self.task_name,
                run_id=self.run_id,
                round_num=self.request.round_num,
                attempt_idx=self.request.attempt_idx,
            )
        )

    def _curator_telemetry_context(self) -> TelemetryContext:
        return _build_curator_telemetry_context(
            task=self.task,
            request=self.request,
            model_name=self.runner.model_name,
            flow="agentic_codegen_multi_turn",
        )

    def _build_prompt(self, request: AscendCAgenticCodegenRequest, *, has_code_map: bool) -> str:
        assert self.wt_session is not None
        if request.eval_summary is None or request.eval_log is None:
            eval_summary, eval_log = build_eval_context_for_llm(
                eval_result=request.eval_result,
                task=self.task,
            )
        else:
            eval_summary = dict(request.eval_summary)
            eval_log = str(request.eval_log)
        strategy_summary = str(
            request.strategy_summary or _extract_bounded_strategy_summary(request.action_text)
        ).strip()
        assert_no_absolute_paths_for_llm(strategy_summary)
        assert_no_absolute_paths_for_llm(eval_log)
        paths = materialize_worktree_context(
            project_dir=self.wt_session.project_dir,
            canonical_strategy_markdown_path=request.canonical_strategy_markdown_path,
            strategy_summary=strategy_summary,
            eval_summary=eval_summary,
            eval_log=eval_log,
            strategy_context=request.strategy_context,
        )
        request = replace(
            request,
            strategy_summary=strategy_summary,
            eval_summary=eval_summary,
            eval_log=eval_log,
            context_paths=paths,
        )
        prompt = self.runner.prompt_builder.build(
            request,
            has_code_map=has_code_map,
            task_path=self._task_path_text(),
        )
        return self._sanitize_prompt(prompt)

    def _run_flow(
        self,
        *,
        prompt: str,
        flow: SubagentFlowConfig,
        telemetry_recorder: Any,
        stage_prompt_sink: StagePromptSink,
    ) -> ClaudeProjectEditResult:
        assert self.wt_session is not None
        if supports_configured_subagent_flow(self.runner.editor_client):
            return run_configured_subagent_flow(
                editor_client=self.runner.editor_client,
                project_dir=self.wt_session.project_dir,
                base_prompt=prompt,
                flow=flow,
                telemetry_recorder=telemetry_recorder,
                session=self.editor_session,
                close_session_on_exit=False,
                stage_prompt_sink=stage_prompt_sink,
                prompt_hygiene_known_paths=[getattr(self.task, "task_path", "")],
            )
        return _edit_project_with_optional_telemetry(
            self.runner.editor_client,
            project_dir=self.wt_session.project_dir,
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
            subagent_flow=flow,
        )

    def run_initial(self) -> AscendCAgenticCodegenResult:
        self._require_open()
        if self._initial_has_run:
            raise RuntimeError("run_initial() may only be called once per AscendCAgenticCycle")
        self._initial_has_run = True
        prompt = self._build_prompt(self.request, has_code_map=bool(self.has_code_map))
        telemetry_recorder = build_file_recorder(
            context=self._telemetry_context(stage=self.request.mode),
            prompt=prompt,
        )
        stage_prompt_sink = self._stage_prompt_sink()
        try:
            edit_result = self._run_flow(
                prompt=prompt,
                flow=self.runner.subagent_flow,
                telemetry_recorder=telemetry_recorder,
                stage_prompt_sink=stage_prompt_sink,
            )
        finally:
            telemetry_recorder.close()
        self.last_stage_prompt_records = stage_prompt_sink.records
        return self._finalize_attempt_result(
            edit_result=edit_result,
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
            flow=self.runner.subagent_flow,
            mode=self.request.mode,
        )

    def continue_fix(self, fix_prompt: str) -> AscendCAgenticCodegenResult:
        self._require_open()
        if not self._initial_has_run:
            raise RuntimeError("continue_fix() requires run_initial() first")
        if self.editor_session is None or not supports_configured_subagent_flow(self.runner.editor_client):
            raise RuntimeError("continue_fix() requires an open Claude agentic session")
        self.task_name, self.run_id = _resolve_agentic_run_context(request=self.request, task=self.task)
        assert self.wt_session is not None
        if not _write_runtime_file(self.wt_session.project_dir, CODE_MAP.filename, self.code_map_text):
            _materialize_existing_code_map(self.store, self.wt_session.project_dir)
        if not _write_runtime_file(
            self.wt_session.project_dir,
            KNOWLEDGE.filename,
            self.curator_context.get(KNOWLEDGE.filename),
        ):
            _materialize_existing_knowledge(self.store, self.wt_session.project_dir)

        fix_prompt = _wrap_repair_prompt(fix_prompt)
        action_with_fix_context = (
            f"{self.request.action_text}\n\nFix context from previous evaluation:\n{fix_prompt}"
        ).strip()
        native_request = replace(self.request, action_text=action_with_fix_context)
        prompt = self._build_prompt(
            native_request,
            has_code_map=(self.wt_session.project_dir / CODE_MAP.filename).is_file(),
        )
        telemetry_recorder = build_file_recorder(
            context=self._telemetry_context(stage="fix"),
            prompt=prompt,
        )
        stage_prompt_sink = self._stage_prompt_sink()
        try:
            edit_result = self._run_flow(
                prompt=prompt,
                flow=self.runner.repair_subagent_flow,
                telemetry_recorder=telemetry_recorder,
                stage_prompt_sink=stage_prompt_sink,
            )
        finally:
            telemetry_recorder.close()
        self.last_stage_prompt_records = stage_prompt_sink.records
        return self._finalize_attempt_result(
            edit_result=edit_result,
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
            flow=self.runner.repair_subagent_flow,
            mode="fix",
        )

    def run_repair_loop(
        self,
        first_result: AscendCAgenticCodegenResult,
        max_fix_rounds: int,
    ) -> AscendCAgenticCodegenResult:
        result = first_result
        for fix_round in range(1, max(0, int(max_fix_rounds or 0)) + 1):
            if result.eval_result.is_passed():
                break
            self.request = replace(self.request, eval_result=result.eval_result)
            result = self.continue_fix(_build_repair_prompt(result.eval_result, fix_round))
        return result

    def _run_eval(self) -> tuple[EvalResult, str | None]:
        assert self.wt_session is not None
        return _run_eval_in_isolated_copy(
            task=self.task,
            candidate_project_dir=self.wt_session.project_dir,
            round_num=self.request.round_num,
        )

    def _finalize_attempt_result(
        self,
        *,
        edit_result: ClaudeProjectEditResult,
        prompt: str,
        telemetry_recorder: Any,
        flow: SubagentFlowConfig,
        mode: str,
    ) -> AscendCAgenticCodegenResult:
        assert self.wt_session is not None
        handoff_texts = _require_native_handoff_files(
            self.wt_session.project_dir,
            _flow_handoff_files(flow),
        )
        produced_code_map = _code_map_from_handoffs(handoff_texts)
        if produced_code_map:
            self.code_map_text = produced_code_map
        _remove_native_handoff_files(self.wt_session.project_dir)
        self.curator_context = dict(handoff_texts)
        self.curator_context.update(_capture_and_remove_non_candidate_files(self.wt_session.project_dir))
        self.last_handoff_texts = handoff_texts
        self.last_edit_result = edit_result

        project_changed_paths = self.wt_session.project_changed_paths()
        changed_paths = _candidate_changed_paths(project_changed_paths or self.wt_session.changed_paths())
        if not changed_paths:
            raise RuntimeError(
                "Claude agentic codegen did not change any files inside the candidate worktree. "
                "Rejecting this attempt instead of importing external task_path changes."
            )

        diff_text = _candidate_diff_text(self.wt_session.project_diff_text())
        eval_result, eval_project_path = self._run_eval()
        self.curator_context.update(getattr(eval_result, "_ksearch_debug_evidence", {}) or {})
        knowledge_text = _run_curator_after_eval(
            editor_client=self.runner.editor_client,
            project_dir=self.wt_session.project_dir,
            eval_result=eval_result,
            store=self.store,
            context_files=self.curator_context,
            telemetry_context=self._curator_telemetry_context(),
        )
        solution = self.task.make_solution_from_project_dir(
            project_dir=self.wt_session.project_dir,
            changed_paths=changed_paths,
            raw_agent_output=edit_result.text,
            round_num=self.request.round_num,
            model_name=self.runner.model_name,
            target_gpu=self.request.target_gpu,
            language="ascendc",
        )
        cleaned = {src.path: src.content for src in solution.sources or []}
        candidate_id = f"round_{int(self.request.round_num):04d}_attempt_{int(self.request.attempt_idx):02d}"
        snapshot_id = f"{candidate_id}_snapshot"
        task_name = self.task_name
        run_id = self.run_id
        artifacts_dir = getattr(self.task, "artifacts_dir", None)
        snapshot_archive_dir = (
            get_ksearch_artifacts_dir(base_dir=artifacts_dir, task_name=str(task_name), run_id=run_id)
            / "snapshots"
        )
        project_snapshot = create_project_snapshot(
            project_dir=self.wt_session.project_dir,
            snapshot_id=snapshot_id,
            parent_snapshot_id=None,
            base_commit=self.wt_session.baseline_commit,
            created_by_round=self.request.round_num,
            eval_result=eval_result.to_dict(include_log_excerpt=True, max_log_chars=8000),
            diff_from_parent=diff_text,
            archive_dir=snapshot_archive_dir,
            run_id=run_id,
        )
        candidate_patch, artifact_paths = write_agentic_candidate_artifacts(
            artifacts_dir=artifacts_dir,
            task_name=str(task_name),
            run_id=run_id,
            round_num=self.request.round_num,
            attempt_idx=self.request.attempt_idx,
            prompt=prompt,
            transcript=edit_result.transcript,
            changed_paths=changed_paths,
            diff_text=diff_text,
            eval_result=eval_result,
            project_snapshot=project_snapshot,
            parent_candidate_id=self.request.parent_candidate_id,
            base_ref=self.wt_session.baseline_commit,
            project_rel_path=self.wt_session.project_rel_path(),
            action_node_id=self.request.action_node_id,
            model_name=self.runner.model_name,
            handoff_files=handoff_texts,
            stage_prompt_records=self.last_stage_prompt_records,
            metadata={
                "run_id": run_id,
                "task_name": task_name,
                "action_node_id": self.request.action_node_id,
                "blocked_strategy_nodes": list(self.request.blocked_strategy_nodes or []),
                "parent_candidate_id": self.request.parent_candidate_id,
                "round_num": self.request.round_num,
                "attempt_idx": self.request.attempt_idx,
                "mode": self.request.mode,
                "artifact_mode": mode,
                "target_gpu": self.request.target_gpu,
                "strategy": dict(self.request.strategy_context or {}),
                "strategy_id": (
                    (self.request.strategy_context or {}).get("strategy_id")
                    if isinstance(self.request.strategy_context, dict)
                    else None
                ),
                "strategy_requires": (
                    list((self.request.strategy_context or {}).get("requires") or [])
                    if isinstance(self.request.strategy_context, dict)
                    else []
                ),
                "parent_solution_id": (
                    (self.request.strategy_context or {}).get("parent_solution_id")
                    if isinstance(self.request.strategy_context, dict)
                    else None
                ),
                "parent_strategy_lineage": (
                    list((self.request.strategy_context or {}).get("parent_strategy_lineage") or [])
                    if isinstance(self.request.strategy_context, dict)
                    else []
                ),
                "adopted": False,
                "adoption_reason": "not_selected_as_parent",
                "code_map_reuse": dict(self.code_map_reuse_manifest),
                "project_path": str(self.wt_session.project_dir),
                "eval_project_path": eval_project_path,
                "evaluator_mutated_project": False,
                **_native_metadata(),
            },
        )
        return AscendCAgenticCodegenResult(
            solution=solution,
            eval_result=eval_result,
            raw=self.task.code_for_world_model_from_raw(raw=cleaned, language="ascendc"),
            cleaned=cleaned,
            transcript=edit_result.transcript,
            prompt=prompt,
            prompt_chars=len(prompt),
            changed_paths=changed_paths,
            diff_text=diff_text,
            project_path=str(self.wt_session.project_dir),
            eval_project_path=eval_project_path,
            diff_after_eval=diff_text,
            evaluator_mutated_project=False,
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
            code_map_text=self.code_map_text,
            knowledge_text=knowledge_text,
        )


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

    def open_cycle(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
    ) -> AscendCAgenticCycle:
        return AscendCAgenticCycle(
            runner=self,
            task=task,
            request=request,
            base_solution=base_solution,
        )

    def run_one_shot_closed(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
        max_fix_rounds: int = 3,
    ) -> AscendCAgenticCodegenResult:
        with self.open_cycle(task=task, request=request, base_solution=base_solution) as cycle:
            result = cycle.run_initial()
            return cycle.run_repair_loop(result, max_fix_rounds=max_fix_rounds)

    def run(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
    ) -> AscendCAgenticCodegenResult:
        return self.run_one_shot_closed(
            task=task,
            request=request,
            base_solution=base_solution,
            max_fix_rounds=0,
        )

    def run_multi_turn(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
        max_fix_rounds: int = 3,
    ) -> AscendCAgenticCodegenResult:
        return self.run_one_shot_closed(
            task=task,
            request=request,
            base_solution=base_solution,
            max_fix_rounds=max_fix_rounds,
        )

    def continue_fix(self, **kwargs: Any) -> AscendCAgenticCodegenResult:
        raise RuntimeError(
            "continue_fix() now belongs to AscendCAgenticCycle. "
            "Use `with runner.open_cycle(...) as cycle: cycle.continue_fix(...)`."
        )
