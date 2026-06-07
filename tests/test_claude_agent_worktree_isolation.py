import asyncio
import shutil
from pathlib import Path

import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenRunner,
)
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeProjectEditResult,
    _make_project_tool_permission_callback,
)
from k_search.tasks.ascendc_task import AscendCTask
from k_search.tasks.task_base import EvalResult


def _permission_behavior(result):
    if isinstance(result, dict):
        return result.get("behavior") or result.get("decision")
    name = type(result).__name__.lower()
    if "deny" in name:
        return "deny"
    if "allow" in name:
        return "allow"
    return getattr(result, "behavior", None)


def _permission_updated_input(result):
    if isinstance(result, dict):
        return result.get("updated_input")
    return getattr(result, "updated_input", None)


def _permission_interrupt(result):
    if isinstance(result, dict):
        return result.get("interrupt")
    return getattr(result, "interrupt", None)


def _callback(worktree: Path):
    return _make_project_tool_permission_callback(
        project_root=worktree,
        allowed_tools={"Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "Glob", "Grep", "Agent", "Skill"},
        allowed_agents={"code-reader"},
        allowed_skills={"ascendc-codegen"},
    )


def _write_native_handoffs(root: Path) -> None:
    (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel.cpp\n", encoding="utf-8")
    (root / "ASCENDC_DESIGN.md").write_text("# design\n", encoding="utf-8")
    (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text("# execution\n", encoding="utf-8")
    (root / "IMPLEMENTATION_HANDOFF.md").write_text("# handoff\n", encoding="utf-8")
    (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")


def test_can_use_tool_allows_worktree_relative_path(tmp_path):
    callback = _callback(tmp_path)

    result = asyncio.run(callback("Write", {"file_path": "kernel.cpp", "content": "x"}, None))

    assert _permission_behavior(result) == "allow"
    assert _permission_updated_input(result)["file_path"] == "kernel.cpp"


def test_can_use_tool_denies_absolute_external_path(tmp_path):
    callback = _callback(tmp_path)

    result = asyncio.run(callback("Write", {"file_path": "/tmp/outside/kernel.cpp", "content": "x"}, None))

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True


def test_can_use_tool_denies_parent_directory_escape(tmp_path):
    callback = _callback(tmp_path)

    result = asyncio.run(callback("Edit", {"file_path": "../outside.cpp", "old_string": "a", "new_string": "b"}, None))

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True


def test_can_use_tool_denies_symlink_escape(tmp_path):
    worktree = tmp_path / "worktree"
    outside = tmp_path / "outside"
    worktree.mkdir()
    outside.mkdir()
    (worktree / "link").symlink_to(outside, target_is_directory=True)
    callback = _callback(worktree)

    result = asyncio.run(callback("Write", {"file_path": "link/evil.cpp", "content": "x"}, None))

    assert _permission_behavior(result) == "deny"
    assert _permission_interrupt(result) is True


def test_can_use_tool_rewrites_omitted_grep_path_to_project_root(tmp_path):
    callback = _callback(tmp_path)

    result = asyncio.run(callback("Grep", {"pattern": "kernel"}, None))

    assert _permission_behavior(result) == "allow"
    assert _permission_updated_input(result)["path"] == "."


def test_runner_fails_without_syncing_external_task_path_when_no_files_changed(tmp_path, monkeypatch):
    import k_search.kernel_generators.ascendc_agentic_codegen as codegen

    monkeypatch.setenv("KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW", "1")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel.cpp").write_text("original\n", encoding="utf-8")
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    class NoChangeClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            _write_native_handoffs(Path(project_dir))
            return ClaudeProjectEditResult(
                text="no changes",
                transcript="no changes",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    class FakeSession:
        project_dir = worktree
        baseline_commit = "baseline"

        def commit_all(self, message):
            return self.baseline_commit

        def project_changed_paths(self):
            return []

        def changed_paths(self):
            return []

        def cleanup(self):
            pass

    copied_task_back = False
    real_copytree = shutil.copytree

    def guard_copytree(src, dst, *args, **kwargs):
        nonlocal copied_task_back
        if Path(src).resolve() == task_dir.resolve() and Path(dst).resolve() == worktree.resolve():
            copied_task_back = True
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(
        codegen,
        "create_agentic_worktree",
        lambda *, task_path, worktree_parent_dir=None: FakeSession(),
    )
    monkeypatch.setattr(codegen, "_materialize_native_assets_baseline", lambda session: None)
    monkeypatch.setattr(shutil, "copytree", guard_copytree)

    task = AscendCTask(task_path=task_dir, definition_name="x")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NoChangeClient())

    with pytest.raises(RuntimeError, match="Rejecting this attempt instead of importing external task_path changes"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec",
                action_text="change code",
                trace_logs="",
                perf_summary="",
                target_gpu="ascend_910b",
                round_num=1,
                attempt_idx=1,
                mode="action",
            ),
            base_solution=None,
        )
    assert copied_task_back is False


def test_agentic_changed_paths_allowlist(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    for rel in ["kernel.cpp", "CMakeLists.txt", "config.json"]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")

    task._validate_agentic_changed_paths(
        project_dir=tmp_path,
        changed_paths=["kernel.cpp", "CMakeLists.txt", "config.json"],
    )

    for rel in [".claude/agents/x.md", "build/generated.cpp", "logs/a.txt", "../evil.cpp", "image.png"]:
        with pytest.raises(ValueError):
            task._validate_agentic_changed_paths(project_dir=tmp_path, changed_paths=[rel])


def test_eval_copy_mutation_does_not_pollute_candidate_solution(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW", "1")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel.cpp").write_text("original\n", encoding="utf-8")

    candidate_content = "candidate before eval\n"
    eval_mutated_content = "eval mutated source\n"

    class MutatingEvalTask(AscendCTask):
        def run_benchmark_in_project_dir(self, *, project_dir, round_num=None, dump_traces=False):
            del dump_traces
            eval_dir = Path(project_dir)
            (eval_dir / "kernel.cpp").write_text(eval_mutated_content, encoding="utf-8")
            return EvalResult(
                status="passed",
                log_excerpt="eval mutated its private copy",
                metrics={"workdir": str(eval_dir), "round": round_num},
            )

    class EditingClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            _write_native_handoffs(root)
            (root / "kernel.cpp").write_text(candidate_content, encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited kernel.cpp",
                transcript="edited kernel.cpp",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    task = MutatingEvalTask(task_path=task_dir, definition_name="x")
    result = AscendCAgenticCodegenRunner(model_name="claude", editor_client=EditingClient()).run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change kernel",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=1,
            attempt_idx=1,
            mode="action",
        ),
        base_solution=None,
    )

    sources = {src.path: src.content for src in result.solution.sources}
    assert sources["kernel.cpp"] == candidate_content
    assert sources["kernel.cpp"] != eval_mutated_content
    assert result.eval_result.metrics["workdir"] != result.project_path
    assert result.evaluator_mutated_project is False
