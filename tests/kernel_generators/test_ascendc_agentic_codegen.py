import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenRunner,
    AscendCAgenticPromptBuilder,
    _build_review_feedback_retry_prompt,
    _parse_review_notes,
)
from k_search.kernel_generators.checkpoint_v3 import StageCheckpointConfig
from k_search.kernel_generators.checkpoint_v3 import StageCheckpointManager
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.kernel_generators.subagent_orchestration import load_subagent_flows
from k_search.tasks.ascendc_task import AscendCTask
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _write_native_handoffs(root: Path, code_map: str = "# CODE_MAP\nkernel/foo.h\n") -> None:
    (root / "CODE_MAP.md").write_text(code_map, encoding="utf-8")
    (root / "ASCENDC_DESIGN.md").write_text(
        "# ASCENDC_DESIGN\n"
        + "Detailed design line for source inspection, workspace, sync, dtype, tail, and offsets.\n" * 3,
        encoding="utf-8",
    )
    (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
        "# IMPLEMENTATION_EXECUTION_PLAN\nApply the requested focused source edit after source inspection.\n",
        encoding="utf-8",
    )
    (root / "IMPLEMENTATION_HANDOFF.md").write_text(
        "# IMPLEMENTATION_HANDOFF\nChanged kernel/foo.h and preserved source contracts.\n",
        encoding="utf-8",
    )
    (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")


def test_review_notes_fixed_status_with_eval_ready_is_accepted():
    state = _parse_review_notes(
        "status: fixed\n"
        "eval_ready: true\n"
        "fix_applied: reviewer applied the required source fix\n"
    )

    assert state.is_eval_ready


def test_review_retry_prompt_requires_execution_plan_regeneration():
    prompt = _build_review_feedback_retry_prompt(
        base_prompt="Base prompt",
        review_text="status: needs_fix\neval_ready: false\nrequired_fixes: fix offsets\n",
        retry_round=1,
    )

    assert "IMPLEMENTATION_EXECUTION_PLAN.md" in prompt
    assert "must regenerate all required files" in prompt


@pytest.fixture(autouse=True)
def _code_map_disabled_by_default(monkeypatch):
    # Keep memory-store tests hermetic unless they explicitly opt into persisted code_map behavior.
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    # Most legacy runner tests in this file use a one-shot edit_project test double.
    # Product code must still fail closed by default; dedicated tests delete this env.
    monkeypatch.setenv("KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW", "1")


class EditingClient:
    def __init__(self, new_text: str):
        self.new_text = new_text
        self.calls = []

    def edit_project(self, *, project_dir, prompt):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        _write_native_handoffs(root)
        target = root / "kernel" / "foo.h"
        target.write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="edited kernel/foo.h",
            transcript="located and edited kernel/foo.h",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


class NativeEditingClient:
    def __init__(
        self,
        new_text: str = "alpha\nBETA\ngamma\n",
        *,
        write_code_map: bool = True,
        write_design: bool = True,
        write_execution_plan: bool = True,
        write_handoff: bool = True,
        write_deviations: bool = False,
        write_review: bool = True,
        review_text: str = "status: ok\neval_ready: true\n",
    ):
        self.new_text = new_text
        self.write_code_map = write_code_map
        self.write_design = write_design
        self.write_execution_plan = write_execution_plan
        self.write_handoff = write_handoff
        self.write_deviations = write_deviations
        self.write_review = write_review
        self.review_text = review_text
        self.calls = []
        self.assets_seen: dict[str, bool] = {}

    def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        self.assets_seen = {
            "code_reader": (root / ".claude" / "agents" / "code-reader.md").exists(),
            "designer": (root / ".claude" / "agents" / "designer.md").exists(),
            "legacy_plan_removed": not (root / ".claude" / "agents" / "plan.md").exists(),
            "codegen": (root / ".claude" / "agents" / "codegen.md").exists(),
            "reviewer": (root / ".claude" / "agents" / "reviewer.md").exists(),
            "bug_fixer": (root / ".claude" / "agents" / "bug-fixer.md").exists(),
            "ascendc_codegen_skill": (root / ".claude" / "skills" / "ascendc-codegen" / "SKILL.md").exists(),
            "ascendc_api_reference_skill": (root / ".claude" / "skills" / "ascendc-api-reference" / "SKILL.md").exists(),
        }
        if self.write_code_map:
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h is the kernel file\n", encoding="utf-8")
        if self.write_design:
            (root / "ASCENDC_DESIGN.md").write_text(
                "# ASCENDC_DESIGN\n"
                + "Detailed design line for source inspection, workspace, sync, dtype, tail, and offsets.\n" * 3,
                encoding="utf-8",
            )
        if self.write_execution_plan:
            (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                "# IMPLEMENTATION_EXECUTION_PLAN\nChange beta to BETA in kernel/foo.h after source inspection.\n",
                encoding="utf-8",
            )
        if self.write_handoff:
            (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                "# IMPLEMENTATION_HANDOFF\nChanged kernel/foo.h and preserved source contracts.\n",
                encoding="utf-8",
            )
        if self.write_deviations:
            (root / "IMPLEMENTATION_DEVIATIONS.md").write_text("D2: equivalent source-local adjustment.\n", encoding="utf-8")
        if self.write_review:
            (root / "REVIEW_NOTES.md").write_text(self.review_text, encoding="utf-8")
        (root / "kernel" / "foo.h").write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md, REVIEW_NOTES.md\nnext: python_eval",
            transcript="native subagents completed",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


class NativeSessionClient:
    def __init__(self):
        self.prompts = []
        self.project_dir: Path | None = None

    def open_session(self, *, project_dir, telemetry_recorder=None):
        from types import SimpleNamespace

        self.project_dir = Path(project_dir)
        return SimpleNamespace(_closed=False, project_dir=self.project_dir)

    def send_prompt(self, session, *, prompt, telemetry_recorder=None):
        self.prompts.append(prompt)
        root = Path(session.project_dir)
        _write_native_handoffs(root)
        if prompt.startswith("Stage 1/3: improvement-assessor"):
            (root / "IMPROVEMENT_ASSESSMENT.md").write_text(
                "status: improve\n"
                "files_written: IMPROVEMENT_ASSESSMENT.md\n"
                "next: codegen\n\n"
                "- strategy_alignment: current implementation follows the selected strategy.\n"
                "- design_alignment: implementation and design are consistent.\n"
                "- implementation_deviation_analysis: no D1/D2/D3 deviation found.\n"
                "- remaining_opportunity: one source-local latency cleanup remains.\n"
                "- edit_scope: kernel/foo.h\n"
                "- risk_checks: correctness, dtype, tail, offsets, workspace, synchronization checked.\n",
                encoding="utf-8",
            )
        text = "alpha\nBETA\ngamma\n" if len(self.prompts) == 1 else "alpha\nGAMMA\ngamma\n"
        (root / "kernel" / "foo.h").write_text(text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md, REVIEW_NOTES.md\nnext: python_eval",
            transcript="native session completed",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )

    def close_session(self, session):
        session._closed = True


class NoOpImproveSessionClient:
    def __init__(self, *, assessment_status: str = "no_op"):
        self.prompts = []
        self.project_dir: Path | None = None
        self.assessment_status = assessment_status

    def open_session(self, *, project_dir, telemetry_recorder=None):
        from types import SimpleNamespace

        self.project_dir = Path(project_dir)
        return SimpleNamespace(_closed=False, project_dir=self.project_dir)

    def send_prompt(self, session, *, prompt, telemetry_recorder=None):
        self.prompts.append(prompt)
        root = Path(session.project_dir)
        _write_native_handoffs(root)
        if prompt.startswith("Stage 3/4: codegen"):
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        if prompt.startswith("Stage 1/3: improvement-assessor"):
            (root / "IMPROVEMENT_ASSESSMENT.md").write_text(
                f"status: {self.assessment_status}\n"
                "files_written: IMPROVEMENT_ASSESSMENT.md\n"
                "next: codegen\n\n"
                "- strategy_alignment: current implementation already follows the selected strategy.\n"
                "- design_alignment: implementation and design are consistent.\n"
                "- implementation_deviation_analysis: no implementation deviation found.\n"
                "- remaining_opportunity: none; no evidence-backed edit exists.\n"
                "- edit_scope: none\n"
                "- risk_checks: correctness, dtype, tail, offsets, workspace, synchronization checked.\n",
                encoding="utf-8",
            )
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: handoffs\nnext: python_eval",
            transcript="native session completed",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )

    def close_session(self, session):
        session._closed = True


class NoChangeClient:
    def edit_project(self, *, project_dir, prompt):
        _write_native_handoffs(Path(project_dir), code_map="# CODE_MAP\nkernel.cpp\n")
        return ClaudeProjectEditResult(
            text="no changes",
            transcript="no changes",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


def test_prompt_builder_omits_full_project_container_and_includes_action():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x\nSpecification:\nVector add.",
        action_text="Increase tile length within UB capacity.",
        trace_logs="compile ok",
        perf_summary="- last_attempt_mean_latency_ms: 1.2",
        target_gpu="ascend_910b",
        round_num=2,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request)

    assert "Increase tile length" in prompt
    assert "ascend_910b" in prompt
    assert "compile ok" in prompt
    assert "<ascendc_project>" not in prompt
    assert "Read/Grep/Glob/Edit/Write" in prompt
    assert "Required native subagent flow" not in prompt
    assert "code-reader -> designer -> codegen -> reviewer" not in prompt
    assert "ASCENDC_DESIGN.md" in prompt
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" in prompt
    assert "IMPLEMENTATION_HANDOFF.md" in prompt
    assert "REVIEW_NOTES.md" in prompt
    assert "bug-fixer" not in prompt
    assert "must not be invoked" not in prompt
    assert "Initial codegen flow agents" not in prompt
    assert "Eval-failure repair flow agents" not in prompt


def test_prompt_builder_raises_section_aware_error_when_budget_exceeded():
    builder = AscendCAgenticPromptBuilder(max_chars=200)
    request = AscendCAgenticCodegenRequest(
        definition_text="D" * 500,
        action_text="A" * 500,
        trace_logs="T" * 500,
        perf_summary="P" * 500,
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="debug",
    )

    with pytest.raises(ValueError, match="agentic prompt exceeded"):
        builder.build(request)


def test_runner_edits_worktree_and_returns_solution(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")
    client = EditingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text=task.get_agentic_definition_text(language="ascendc"),
            action_text="Change beta to BETA.",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=1,
            mode="action",
        ),
        base_solution=None,
    )

    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
    assert result.changed_paths == ["kernel/foo.h"]
    assert "-beta" in result.diff_text
    assert "+BETA" in result.diff_text
    assert client.calls
    assert "<ascendc_project>" not in client.calls[0][1]


def test_runner_materializes_strategy_from_canonical_path_not_action_text(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    canonical_strategy = tmp_path / "catalog_strategy.md"
    canonical_body = "# Canonical Strategy\n\n" + "full strategy line\n" * 200
    canonical_strategy.write_text(canonical_body, encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    class StrategyInspectingClient(EditingClient):
        def edit_project(self, *, project_dir, prompt):
            root = Path(project_dir)
            strategy_text = (root / ".ksearch" / "context" / "STRATEGY.md").read_text(
                encoding="utf-8"
            )
            assert strategy_text == canonical_body
            assert "[truncated strategy markdown]" not in strategy_text
            assert "[truncated for agentic prompt budget]" not in strategy_text
            assert "[truncated strategy markdown]" not in prompt
            assert "Full natural-language strategy markdown" not in prompt
            return super().edit_project(project_dir=project_dir, prompt=prompt)

    client = StrategyInspectingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text=task.get_agentic_definition_text(language="ascendc"),
            action_text=(
                "Change beta to BETA.\n\n"
                "Full natural-language strategy markdown:\n"
                "stale prompt section\n\n"
                "[truncated strategy markdown]"
            ),
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=1,
            mode="action",
            canonical_strategy_markdown_path=canonical_strategy,
        ),
        base_solution=None,
    )

    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


def test_runner_evaluates_worktree_and_persists_project_snapshot_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-artifact")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (task_dir / "kernel" / "large_header.hpp").write_text("x" * 210_000, encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel/foo.h').read_text() == 'alpha\\nBETA\\ngamma\\n'; "
            "assert Path('kernel/large_header.hpp').exists(); "
            "print('build saw edited complete worktree')"
        ),
        test_cmd=_py_cmd("print('correctness passed')"),
        bench_cmd=_py_cmd("print('latency_ms=4.0')"),
        reference_latency_ms=8.0,
        timeout_seconds=30,
    )
    client = EditingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text=task.get_agentic_definition_text(language="ascendc"),
            action_text="Change beta to BETA.",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=1,
            mode="action",
            run_id="artifact-run",
            task_name="x",
            parent_candidate_id="parent-7",
            action_node_id="A-12",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "passed"
    assert result.eval_result.metrics["score"] == 2.0
    assert result.eval_project_path is not None
    assert result.eval_result.metrics["workdir"] == result.eval_project_path
    assert result.eval_result.metrics["workdir"] != result.project_path
    assert str(
        tmp_path / "artifacts" / "x" / "task-artifact" / "runs" / "artifact-run" / "worktrees"
    ) in result.project_path
    assert result.evaluator_mutated_project is False
    assert "build saw edited complete worktree" in result.eval_result.log_excerpt
    assert result.candidate_patch is not None
    assert result.candidate_patch.action_node_id == "A-12"
    assert result.candidate_patch.parent_candidate_id == "parent-7"
    assert result.project_snapshot is not None
    assert "kernel/large_header.hpp" in result.project_snapshot.manifest
    assert result.artifact_paths is not None
    assert str(
        tmp_path
        / "artifacts"
        / "x"
        / "task-artifact"
        / "runs"
        / "artifact-run"
        / "artifacts"
        / "candidates"
    ) in result.artifact_paths["manifest_path"]
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == result.candidate_patch.candidate_id
    assert manifest["snapshot_id"] == result.project_snapshot.snapshot_id
    assert manifest["run_id"] == "artifact-run"
    assert manifest["task_name"] == "x"
    assert manifest["action_node_id"] == "A-12"
    assert manifest["parent_candidate_id"] == "parent-7"
    assert manifest["round_num"] == 3
    assert manifest["attempt_idx"] == 1
    assert manifest["mode"] == "action"
    assert Path(result.artifact_paths["diff_path"]).read_text(encoding="utf-8") == result.diff_text
    assert json.loads(Path(result.artifact_paths["eval_path"]).read_text(encoding="utf-8"))["status"] == "passed"


def test_runner_evaluation_remaps_absolute_harness_path_to_isolated_worktree_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-absolute-harness")
    repo = tmp_path / "repo"
    task_dir = repo / "agent_workdir" / "flash_attention"
    scripts_dir = repo / "agent_workdir" / "scripts"
    (task_dir / "kernel").mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    (task_dir / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (task_dir / "kernel" / "foo.h").write_text("baseline-good\n", encoding="utf-8")
    (scripts_dir / "evaluate_ascendc.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "SCRIPT_DIR=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\n"
        "WORKDIR=$(cd \"$SCRIPT_DIR/..\" && pwd)\n"
        "TASK_NAME=${1:?task name required}\n"
        "TARGET=\"$WORKDIR/$TASK_NAME/kernel/foo.h\"\n"
        "echo \"checking $TARGET\"\n"
        "if grep -q candidate-broken \"$TARGET\"; then\n"
        "  echo \"candidate precision failed\"\n"
        "  exit 9\n"
        "fi\n"
        "echo \"baseline precision passed\"\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "ksearch@example.invalid")
    _git(repo, "config", "user.name", "K Search Tests")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    task = AscendCTask(
        task_path=task_dir,
        definition_name="flash_attention",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=f"bash {shlex.quote(str(scripts_dir / 'evaluate_ascendc.sh'))} flash_attention basic",
        bench_cmd=_py_cmd("print('latency_ms=4.0')"),
        reference_latency_ms=8.0,
        timeout_seconds=30,
    )
    client = NativeEditingClient(new_text="candidate-broken\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text=task.get_agentic_definition_text(language="ascendc"),
            action_text="Make candidate fail correctness.",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=1,
            mode="action",
            run_id="absolute-harness-run",
            task_name="flash_attention",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "failed"
    assert "candidate precision failed" in result.eval_result.log_excerpt
    assert "baseline precision passed" not in result.eval_result.log_excerpt


def test_runner_requires_explicit_run_id_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KSEARCH_ALLOW_MISSING_AGENTIC_RUN_CONTEXT", raising=False)
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=EditingClient("alpha\nBETA\ngamma\n"))

    with pytest.raises(RuntimeError, match="missing run_id"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec",
                action_text="change beta",
                trace_logs="",
                perf_summary="",
                target_gpu="ascend_910b",
                round_num=1,
                attempt_idx=1,
                mode="action",
                task_name="x",
            ),
            base_solution=None,
        )


def test_runner_allows_missing_run_context_with_escape_hatch(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ALLOW_MISSING_AGENTIC_RUN_CONTEXT", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-fallback")
    monkeypatch.setenv("KSEARCH_RUN_ID", "fallback-run")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=EditingClient("alpha\nBETA\ngamma\n"))

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=1,
            attempt_idx=1,
            mode="action",
            task_name="x",
        ),
        base_solution=None,
    )

    assert result.artifact_paths is not None
    assert str(
        tmp_path / "artifacts" / "x" / "task-fallback" / "runs" / "fallback-run" / "artifacts"
    ) in result.artifact_paths["manifest_path"]


def test_runner_fails_when_agent_makes_no_file_changes(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NoChangeClient())

    with pytest.raises(RuntimeError, match="did not change any files"):
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


def test_runner_overlays_base_solution_before_editing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    base_solution = Solution(
        name="base",
        definition="x",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel/foo.h::run",
        ),
        sources=[SourceFile(path="kernel/foo.h", content="overlaid_base\n")],
    )

    class OverlayCheckClient:
        def __init__(self):
            self.project_dirs = []

        def edit_project(self, *, project_dir, prompt):
            root = Path(project_dir)
            self.project_dirs.append(root)
            pre_overlay = (root / "kernel" / "foo.h").read_text(encoding="utf-8")
            assert pre_overlay == "overlaid_base\n"
            _write_native_handoffs(root)
            (root / "kernel" / "foo.h").write_text("overlaid_base\nBETA\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    client = OverlayCheckClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="overlay then edit",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=2,
            attempt_idx=1,
            mode="improve",
        ),
        base_solution=base_solution,
    )

    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


def test_runner_creates_attempt_telemetry_files(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-1")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    class TelemetryAwareClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            _write_native_handoffs(root)
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            if telemetry_recorder is not None:
                from k_search.telemetry.events import TelemetryEvent

                telemetry_recorder.emit(TelemetryEvent(event_type="llm_start", provider="claude-agent", model_name="claude"))
                telemetry_recorder.emit(
                    TelemetryEvent(
                        event_type="llm_result",
                        provider="claude-agent",
                        model_name="claude",
                        session_id="sess-runner",
                        total_cost_usd=0.5,
                        num_turns=2,
                        duration_ms=100,
                    )
                )
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
                trace_path=telemetry_recorder.artifacts.trace_path if telemetry_recorder else None,
                timeline_path=telemetry_recorder.artifacts.timeline_path if telemetry_recorder else None,
                cost_path=telemetry_recorder.artifacts.cost_path if telemetry_recorder else None,
                session_id="sess-runner",
                total_cost_usd=0.5,
                num_turns=2,
                duration_ms=100,
            )

    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=TelemetryAwareClient())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=2,
            mode="action",
        ),
        base_solution=None,
    )

    assert result.trace_path is not None
    assert result.timeline_path is not None
    assert result.cost_path is not None
    assert Path(result.trace_path).exists()
    assert Path(result.timeline_path).exists()
    assert Path(result.cost_path).exists()
    assert Path(result.trace_path).parent.name == "attempt_0002"
    assert Path(result.trace_path).parent.parent.name == "action_unknown"
    assert result.session_id == "sess-runner"
    assert result.total_cost_usd == 0.5


def test_runner_disables_telemetry_with_env(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_TELEMETRY", "0")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    class Client:
        def edit_project(self, *, project_dir, prompt):
            root = Path(project_dir)
            _write_native_handoffs(root)
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=Client())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=1,
            attempt_idx=1,
            mode="action",
        ),
        base_solution=None,
    )

    assert result.trace_path is None
    assert result.timeline_path is None
    assert result.cost_path is None
    assert result.changed_paths == ["kernel/foo.h"]


def test_prompt_builder_replaces_paths_with_placeholder(tmp_path):
    """Prompt must NOT inject physical worktree paths; they become a placeholder."""
    # 本次范围:仅净化 worktree 临时路径(来自 trace_logs);definition 中用户 task_path 的原始绝对路径不在本次根治范围(见 spec §4.4)。
    original_dir = tmp_path / "original_project"

    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text=(
            f"Task: x\nSpecification source: {original_dir}/ksearch_task.md\n"
            f"Specification:\nSee {original_dir}/kernel/foo.h for details."
        ),
        action_text="Optimize the kernel.",
        trace_logs="[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/kernel/x.cpp",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request)

    assert "ksearch_agentic_worktree_02ut4r9r" not in prompt
    assert "<PROJECT_ROOT>" in prompt
    assert "Specification source:" in prompt


def test_prompt_builder_includes_cwd_only_instruction():
    """Prompt must contain the CWD-only constraint instruction."""
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x",
        action_text="change code",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request)

    assert "ONLY edit files inside the current project directory" in prompt


def test_prompt_builder_uses_code_map_branch_when_present():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x",
        action_text="optimize",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    with_map = builder.build(request, has_code_map=True)
    without_map = builder.build(request, has_code_map=False)

    assert "CODE_MAP.md" in with_map
    assert "Read it first" in with_map
    assert "update the affected sections" in with_map
    assert "CODE_MAP.md already exists: yes" in with_map
    assert "CODE_MAP.md already exists: no" in without_map
    assert "Use the code-reader subagent to create CODE_MAP.md" not in without_map


def test_prompt_builder_requires_file_handoff_and_short_subagent_summaries():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x",
        action_text="optimize",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request, has_code_map=False)

    assert "CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md, IMPLEMENTATION_HANDOFF.md" in prompt
    assert "optional IMPLEMENTATION_DEVIATIONS.md" in prompt
    assert "status, files_written, and next" in prompt
    assert "Do not paste CODE_MAP.md, ASCENDC_DESIGN.md, IMPLEMENTATION_EXECUTION_PLAN.md" in prompt


def test_runner_returns_code_map_for_adopted_writeback_on_first_round(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-native-map")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class ReaderClient:
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(project_dir)
            _write_native_handoffs(root, code_map="# CODE_MAP\nfoo.h is the kernel\n")
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            text = "native subagents completed"
            return ClaudeProjectEditResult(
                text=text, transcript=text, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = ReaderClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude", editor_client=client, reader_editor_client=client
    )
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert len(client.prompts) == 1
    assert "Use the code-reader subagent to create CODE_MAP.md" in client.prompts[0]
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore, save_code_map_if_adopted
    store = MemoryStore.for_task(task)
    assert result.code_map_text is not None
    assert store.load(CODE_MAP) is None
    save_code_map_if_adopted(task=task, code_map_text=result.code_map_text, adopted=True)
    assert store.load(CODE_MAP) is not None
    assert "CODE_MAP.md" not in result.changed_paths
    assert "kernel/foo.h" in result.changed_paths


def test_runner_does_not_persist_code_map_from_failed_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-failed-map-not-adopted")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("import sys; print('compile failed'); sys.exit(1)"),
        test_cmd=_py_cmd("print('not reached')"),
        bench_cmd=_py_cmd("print('not reached')"),
        timeout_seconds=30,
    )

    class FailedCandidateClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            _write_native_handoffs(root, code_map="# CODE_MAP\nfailed candidate map\n")
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    result = AscendCAgenticCodegenRunner(model_name="claude", editor_client=FailedCandidateClient()).run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore

    assert result.eval_result.status == "compile_failed"
    assert result.code_map_text == "# CODE_MAP\nfailed candidate map\n"
    assert MemoryStore.for_task(task).load(CODE_MAP) is None


def test_runner_runs_curator_after_eval_and_persists_knowledge(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "1")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-curator")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class CuratorAwareClient:
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(project_dir)
            if "knowledge-curator" in prompt:
                # Post-eval curator turn: distill a lesson.
                (root / "KNOWLEDGE.md").write_text(
                    "## 1: missing MTE3/MTE2 sync between GM->UB->GM\n", encoding="utf-8"
                )
                text = "curation done"
            else:
                _write_native_handoffs(root, code_map="# CODE_MAP\nfoo.h is the kernel\n")
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "native subagents completed"
            return ClaudeProjectEditResult(
                text=text, transcript=text, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = CuratorAwareClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude", editor_client=client, reader_editor_client=client
    )
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    # Two edit_project calls: codegen flow + post-eval curator.
    assert len(client.prompts) == 2
    assert "knowledge-curator" in client.prompts[1]
    # Curator output captured on the result.
    assert result.knowledge_text is not None
    assert "MTE3/MTE2" in result.knowledge_text
    # KNOWLEDGE.md is a memory file, never a candidate source artifact.
    assert "KNOWLEDGE.md" not in result.changed_paths
    assert "KNOWLEDGE.md" not in result.diff_text
    if result.project_snapshot is not None:
        assert "KNOWLEDGE.md" not in result.project_snapshot.manifest

    # Gated persistence: hook saves only when adopted.
    from k_search.kernel_generators.memory import KNOWLEDGE, MemoryStore, save_knowledge_if_adopted
    store = MemoryStore.for_task(task)
    assert store.load(KNOWLEDGE) is None
    save_knowledge_if_adopted(task=task, knowledge_text=result.knowledge_text, adopted=True)
    assert store.load(KNOWLEDGE) is not None
    # Next round materializes it back into a fresh worktree.
    from k_search.kernel_generators.ascendc_agentic_codegen import _materialize_existing_knowledge
    wt = tmp_path / "wt"
    wt.mkdir()
    assert _materialize_existing_knowledge(store, wt) is True
    assert (wt / "KNOWLEDGE.md").read_text(encoding="utf-8").startswith("## 1:")


def test_runner_records_curator_telemetry_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "1")
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-curator-telemetry")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class TelemetryCuratorClient:
        def __init__(self):
            self.telemetry_contexts = []
            self.cost_paths = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            if telemetry_recorder is not None:
                from k_search.telemetry.events import TelemetryEvent

                self.telemetry_contexts.append(telemetry_recorder.context)
                self.cost_paths.append(telemetry_recorder.artifacts.cost_path)
                cost = 0.05 if "knowledge-curator" in prompt else 0.5
                telemetry_recorder.emit(
                    TelemetryEvent(event_type="llm_start", provider="claude-agent", model_name="claude")
                )
                telemetry_recorder.emit(
                    TelemetryEvent(
                        event_type="llm_result",
                        provider="claude-agent",
                        model_name="claude",
                        total_cost_usd=cost,
                    )
                )
            if "knowledge-curator" in prompt:
                (root / "KNOWLEDGE.md").write_text("## telemetry lesson\n", encoding="utf-8")
                text = "curation done"
            else:
                _write_native_handoffs(root, code_map="# CODE_MAP\nfoo.h is the kernel\n")
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "native subagents completed"
            return ClaudeProjectEditResult(
                text=text, transcript=text, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = TelemetryCuratorClient()
    result = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client).run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.trace_path is not None
    assert len(client.telemetry_contexts) == 2
    assert client.telemetry_contexts[0].stage == "action"
    assert client.telemetry_contexts[1].stage == "curator"
    assert client.cost_paths[0] != client.cost_paths[1]
    assert Path(client.cost_paths[0]).parent.parent.name == "action_unknown"
    assert Path(client.cost_paths[1]).parent.parent.name == "action_curator"
    curator_cost = json.loads(Path(client.cost_paths[1]).read_text(encoding="utf-8"))
    assert curator_cost["summary"]["total_cost_usd"] == 0.05
    curator_prompt = (Path(client.cost_paths[1]).parent / "prompt.md").read_text(encoding="utf-8")
    assert "knowledge-curator" in curator_prompt


def test_runner_runs_curator_in_isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "1")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-curator-isolated")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class MutatingCuratorClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            if "knowledge-curator" in prompt:
                (root / "kernel").mkdir(exist_ok=True)
                (root / "kernel" / "foo.h").write_text("alpha\nCURATOR_MUTATION\ngamma\n", encoding="utf-8")
                (root / "KNOWLEDGE.md").write_text("## isolated lesson\n", encoding="utf-8")
                text = "curation done"
            else:
                _write_native_handoffs(root, code_map="# CODE_MAP\nfoo.h is the kernel\n")
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "native subagents completed"
            return ClaudeProjectEditResult(
                text=text, transcript=text, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    result = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=MutatingCuratorClient(),
    ).run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    foo = next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
    assert foo == "alpha\nBETA\ngamma\n"
    assert "CURATOR_MUTATION" not in result.diff_text
    assert "isolated lesson" in str(result.knowledge_text)


def test_runner_removes_materialized_knowledge_when_curator_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-knowledge-not-candidate")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    from k_search.kernel_generators.memory import KNOWLEDGE, MemoryStore
    MemoryStore.for_task(task).save(KNOWLEDGE, "SECRET_LESSON_SHOULD_NOT_BE_CANDIDATE_DIFF\n")

    class CodegenOnlyClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            assert (root / "KNOWLEDGE.md").is_file()
            _write_native_handoffs(root, code_map="# CODE_MAP\nfoo.h is the kernel\n")
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited", transcript="edited", prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    result = AscendCAgenticCodegenRunner(model_name="claude", editor_client=CodegenOnlyClient()).run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert "KNOWLEDGE.md" not in result.changed_paths
    assert "SECRET_LESSON" not in result.diff_text
    if result.project_snapshot is not None:
        assert "KNOWLEDGE.md" not in result.project_snapshot.manifest
    assert "KNOWLEDGE.md" not in {src.path for src in result.solution.sources}



def test_runner_reuses_existing_code_map_without_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "test-preseeded-map")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    MemoryStore.for_task(task).save(
        CODE_MAP,
        "# CODE_MAP\npreseeded\n",
        meta={
            "schema_version": 1,
            "solution_id": "sol_parent",
            "branch_id": "root/s1",
            "adopted": True,
        },
    )

    class CodegenOnlyClient:
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(project_dir)
            assert (root / "CODE_MAP.md").read_text(encoding="utf-8") == "# CODE_MAP\npreseeded\n"
            _write_native_handoffs(root, code_map="# CODE_MAP\npreseeded\n")
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited", transcript="edited", prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = CodegenOnlyClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude", editor_client=client, reader_editor_client=client
    )
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=2, attempt_idx=1, mode="improve",
            strategy_context={
                "parent_solution_id": "sol_parent",
                "parent_branch_id": "root/s1",
                "action_node_id": "s2",
            },
        ),
        base_solution=None,
    )
    assert len(client.prompts) == 1
    assert "CODE_MAP.md already exists: yes" in client.prompts[0]
    assert "Read it first" in client.prompts[0]
    assert "CODE_MAP.md" not in result.changed_paths


def test_runner_code_map_disabled_skips_memory_persistence_only(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    client = EditingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )
    assert "CODE_MAP.md already exists: no" in client.calls[0][1]
    assert "Use the code-reader subagent to create CODE_MAP.md" in client.calls[0][1]
    assert result.code_map_text is not None


def test_runner_code_map_not_in_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class C:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            _write_native_handoffs(root, code_map="# CODE_MAP\nmapped\n")
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            t = "edited"
            return ClaudeProjectEditResult(
                text=t, transcript=t, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    c = C()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=c, reader_editor_client=c)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )
    assert "CODE_MAP.md" not in result.diff_text
    assert "kernel/foo.h" in result.diff_text
    if result.project_snapshot is not None:
        assert "CODE_MAP.md" not in result.project_snapshot.manifest


def test_runner_materializes_native_assets_and_uses_single_project_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    client = NativeEditingClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert len(client.calls) == 1
    project_dir, prompt = client.calls[0]
    assert project_dir
    assert all(client.assets_seen.values())
    assert "Required native subagent flow" in prompt
    assert "code-reader -> designer -> codegen -> reviewer" in prompt
    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


def test_native_flow_requires_configured_flow_support_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("KSEARCH_ALLOW_LEGACY_SINGLE_AGENT_FLOW", raising=False)
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    with pytest.raises(RuntimeError, match="Configured subagent flow is required"):
        AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient()).run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_accepts_split_heading_review_notes(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    client = NativeEditingClient(
        review_text=(
            "## status\n\n\n"
            "ok\n\n"
            "## eval_ready\n\n"
            "true\n"
        ),
    )
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "passed"
    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


def test_runner_uses_configured_subagent_stages_in_one_session(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class StagedSessionClient:
        def __init__(self):
            self.open_count = 0
            self.closed = False
            self.prompts: list[str] = []
            self.project_dir: Path | None = None

        def open_session(self, *, project_dir, telemetry_recorder=None):
            from types import SimpleNamespace

            self.open_count += 1
            self.project_dir = Path(project_dir)
            return SimpleNamespace(_closed=False, project_dir=self.project_dir)

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            if "Stage 1/4: code-reader" in prompt:
                assert "Use the code-reader subagent" in prompt
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
                text = "reader done"
            elif "Stage 2/4: designer" in prompt:
                assert (root / "CODE_MAP.md").exists()
                assert "Use the designer subagent" in prompt
                (root / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
                text = "designer done"
            elif "Stage 3/4: codegen" in prompt:
                assert (root / "ASCENDC_DESIGN.md").exists()
                assert "Use the codegen subagent" in prompt
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
                (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                    "# execution\nchange beta after source inspection\n",
                    encoding="utf-8",
                )
                (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                    "# handoff\nchanged kernel/foo.h and preserved contracts\n",
                    encoding="utf-8",
                )
                text = "codegen done"
            elif "Stage 4/4: reviewer" in prompt:
                assert "Use the reviewer subagent" in prompt
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "review done"
            else:
                raise AssertionError(f"unexpected stage prompt: {prompt}")
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            self.closed = True
            session._closed = True

    client = StagedSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert client.open_count == 1
    assert client.closed is True
    assert [prompt.splitlines()[0] for prompt in client.prompts] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: designer",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
    ]
    assert result.changed_paths == ["kernel/foo.h"]
    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
    assert result.artifact_paths is not None
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    stage_paths = manifest["stage_prompt_paths"]
    assert [item["stage"] for item in stage_paths] == ["code-reader", "designer", "codegen", "reviewer"]
    for item in stage_paths:
        assert item["path"].startswith("stage_prompts/")
        assert not Path(item["path"]).is_absolute()
        saved_prompt = Path(result.artifact_paths["manifest_path"]).parent / item["path"]
        assert saved_prompt.is_file()
        assert saved_prompt.read_text(encoding="utf-8") == client.prompts[item["index"] - 1]
        assert item["hygiene"]["contains_absolute_path"] is False
        assert item["hygiene"]["contains_global_flow_policy"] is False


def test_runner_does_not_import_old_python_project_agents(tmp_path, monkeypatch):
    from types import ModuleType

    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    forbidden = ModuleType("k_search.kernel_generators.agents")

    def _blocked_getattr(name):
        raise AssertionError(f"old Python agent import used: {name}")

    forbidden.__getattr__ = _blocked_getattr
    monkeypatch.setitem(sys.modules, "k_search.kernel_generators.agents", forbidden)

    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.changed_paths == ["kernel/foo.h"]


def test_runner_fails_when_ascendc_design_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_design=False))

    with pytest.raises(RuntimeError, match="ASCENDC_DESIGN.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_fails_when_execution_plan_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=NativeEditingClient(write_execution_plan=False),
    )

    with pytest.raises(RuntimeError, match="IMPLEMENTATION_EXECUTION_PLAN.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_fails_when_implementation_handoff_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_handoff=False))

    with pytest.raises(RuntimeError, match="IMPLEMENTATION_HANDOFF.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_fails_when_review_notes_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_review=False))

    with pytest.raises(RuntimeError, match="REVIEW_NOTES.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_retries_codegen_when_reviewer_marks_candidate_not_eval_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "review-retry")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
    )

    class ReviewRetryClient:
        def __init__(self):
            self.prompts: list[str] = []

        def open_session(self, *, project_dir, telemetry_recorder=None):
            from types import SimpleNamespace

            return SimpleNamespace(_closed=False, project_dir=Path(project_dir))

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            first = prompt.splitlines()[0]
            if first == "Stage 1/4: code-reader":
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
                text = "reader done"
            elif first == "Stage 2/4: designer":
                (root / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
                text = "designer done"
            elif first == "Stage 3/4: codegen":
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
                (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                    "# execution\nchange beta after source inspection\n",
                    encoding="utf-8",
                )
                (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                    "# handoff\nchanged kernel/foo.h and preserved contracts\n",
                    encoding="utf-8",
                )
                text = "codegen done"
            elif first == "Stage 4/4: reviewer":
                (root / "REVIEW_NOTES.md").write_text(
                    "status: needs_fix\neval_ready: false\nrequired_fixes: fix tiling contract\n",
                    encoding="utf-8",
                )
                text = "review needs fix"
            elif first == "Stage 1/2: codegen":
                assert "required_fixes: fix tiling contract" in prompt
                (root / "kernel" / "foo.h").write_text("alpha\nREVIEWED\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h reviewed\n", encoding="utf-8")
                (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                    "# execution\napply reviewer feedback after source inspection\n",
                    encoding="utf-8",
                )
                (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                    "# handoff\naddressed reviewer feedback in kernel/foo.h\n",
                    encoding="utf-8",
                )
                text = "codegen review retry done"
            elif first == "Stage 2/2: reviewer":
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "review ready"
            else:
                raise AssertionError(first)
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    client = ReviewRetryClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="review-retry",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "passed"
    assert [prompt.splitlines()[0] for prompt in client.prompts] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: designer",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
        "Stage 1/2: codegen",
        "Stage 2/2: reviewer",
    ]
    assert "REVIEWED" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


def test_runner_fails_when_code_map_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_code_map=False))

    with pytest.raises(RuntimeError, match="CODE_MAP.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_filters_handoff_and_claude_asset_paths_from_candidate_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_deviations=True))

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.changed_paths == ["kernel/foo.h"]
    assert "CODE_MAP.md" not in result.diff_text
    assert "ASCENDC_DESIGN.md" not in result.diff_text
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" not in result.diff_text
    assert "IMPLEMENTATION_HANDOFF.md" not in result.diff_text
    assert "IMPLEMENTATION_DEVIATIONS.md" not in result.diff_text
    assert "REVIEW_NOTES.md" not in result.diff_text
    assert ".claude/agents" not in result.diff_text
    assert result.project_snapshot is not None
    assert "CODE_MAP.md" not in result.project_snapshot.manifest
    assert "ASCENDC_DESIGN.md" not in result.project_snapshot.manifest
    assert "IMPLEMENTATION_EXECUTION_PLAN.md" not in result.project_snapshot.manifest
    assert "IMPLEMENTATION_HANDOFF.md" not in result.project_snapshot.manifest
    assert "IMPLEMENTATION_DEVIATIONS.md" not in result.project_snapshot.manifest
    assert "REVIEW_NOTES.md" not in result.project_snapshot.manifest
    assert not any(path.startswith(".claude/") for path in result.project_snapshot.manifest)
    assert result.artifact_paths is not None
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    handoff_paths = manifest["native_handoff_paths"]
    assert sorted(handoff_paths) == [
        "ASCENDC_DESIGN.md",
        "CODE_MAP.md",
        "IMPLEMENTATION_DEVIATIONS.md",
        "IMPLEMENTATION_EXECUTION_PLAN.md",
        "IMPLEMENTATION_HANDOFF.md",
        "REVIEW_NOTES.md",
    ]
    assert "Change beta to BETA" in Path(handoff_paths["IMPLEMENTATION_EXECUTION_PLAN.md"]).read_text(encoding="utf-8")
    assert "D2:" in Path(handoff_paths["IMPLEMENTATION_DEVIATIONS.md"]).read_text(encoding="utf-8")


def test_continue_fix_uses_native_prompt_and_run_scoped_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-native-continue")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-continue")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = NativeSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    first_request = AscendCAgenticCodegenRequest(
        definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-continue",
    )

    with runner.open_cycle(task=task, request=first_request, base_solution=None) as cycle:
        first = cycle.run_initial()
        second_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=2, mode="debug", run_id="native-continue",
        )
        cycle.request = second_request
        second = cycle.continue_fix("raw compile fix context")

    assert len(client.prompts) == 6
    assert [prompt.splitlines()[0] for prompt in client.prompts[:4]] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: designer",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
    ]
    assert [prompt.splitlines()[0] for prompt in client.prompts[4:]] == [
        "Stage 1/2: bug-fixer",
        "Stage 2/2: reviewer",
    ]
    assert "raw compile fix context" in client.prompts[4]
    assert "Use the bug-fixer subagent" in client.prompts[4]
    assert second.artifact_paths is not None
    assert str(
        tmp_path / "artifacts" / "x" / "task-native-continue" / "runs" / "native-continue" / "artifacts"
    ) in second.artifact_paths["manifest_path"]


def test_runner_writes_checkpoint_v3_stage_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-checkpoint-v3")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = NativeSessionClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=client,
        stage_checkpoint_config=StageCheckpointConfig(enabled=True),
    )

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=2,
            attempt_idx=1,
            mode="action",
            run_id="checkpoint-v3-run",
            task_name="x",
        ),
        base_solution=None,
    )

    checkpoints_dir = (
        tmp_path
        / "artifacts"
        / "x"
        / "task-checkpoint-v3"
        / "runs"
        / "checkpoint-v3-run"
        / "artifacts"
        / "checkpoints"
    )
    latest = json.loads((checkpoints_dir / "latest.json").read_text(encoding="utf-8"))
    manifest_path = checkpoints_dir / latest["latest_checkpoint_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_state = json.loads((manifest_path.parent / "stage_state.json").read_text(encoding="utf-8"))

    assert result.eval_result.status == "passed"
    assert manifest["checkpoint_kind"] == "stage_completed"
    assert manifest["position"]["last_completed_stage_name"] == "reviewer"
    assert manifest["position"]["next_stage_index"] is None
    assert [item["status"] for item in stage_state["stages"]] == ["completed", "completed", "completed", "completed"]
    assert (manifest_path.parent / "stage" / "handoff" / "REVIEW_NOTES.md").is_file()


def test_runner_resumes_checkpoint_v3_from_designer_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-checkpoint-v3-resume")
    checkpoint_project = tmp_path / "checkpoint_project"
    (checkpoint_project / "kernel").mkdir(parents=True)
    (checkpoint_project / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (checkpoint_project / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
    (checkpoint_project / "ASCENDC_DESIGN.md").write_text("# ASCENDC_DESIGN\nuse uppercase beta\n", encoding="utf-8")
    flow = load_subagent_flows().get("initial_codegen")
    manager = StageCheckpointManager(
        artifacts_dir=tmp_path / "previous_artifacts",
        task_name="x",
        task_id="previous-task",
        run_id="previous-run",
        config=StageCheckpointConfig(enabled=True),
    )
    designer_manifest = manager.save_stage_completed(
        task=SimpleNamespace(name="x", definition_name="x", task_path=str(checkpoint_project)),
        project_dir=checkpoint_project,
        flow=flow,
        stage=flow.stages[1],
        stage_index=2,
        round_num=2,
        attempt_idx=1,
        prompt="designer prompt",
        result=ClaudeProjectEditResult(
            text="designer done",
            transcript="designer done",
            prompt="designer prompt",
            prompt_chars=len("designer prompt"),
            prompt_lines=1,
        ),
        session=None,
        telemetry_recorder=None,
        runtime_state={},
    )

    class ResumeSessionClient(NativeSessionClient):
        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            if prompt.startswith("Stage 1/2: codegen"):
                assert (root / "CODE_MAP.md").is_file()
                assert (root / "ASCENDC_DESIGN.md").is_file()
                (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                    "# IMPLEMENTATION_EXECUTION_PLAN\nChange beta to BETA.\n",
                    encoding="utf-8",
                )
                (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                    "# IMPLEMENTATION_HANDOFF\nChanged kernel/foo.h.\n",
                    encoding="utf-8",
                )
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "codegen done"
            elif prompt.startswith("Stage 2/2: reviewer"):
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "reviewer done"
            else:
                raise AssertionError(f"unexpected prompt: {prompt.splitlines()[0]}")
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = ResumeSessionClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=client,
        stage_checkpoint_config=StageCheckpointConfig(
            enabled=True,
            resume_from=str(designer_manifest),
        ),
    )

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="resume same action",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=2,
            attempt_idx=1,
            mode="action",
            run_id="checkpoint-v3-resume-run",
            task_name="x",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "passed"
    assert [prompt.splitlines()[0] for prompt in client.prompts] == [
        "Stage 1/2: codegen",
        "Stage 2/2: reviewer",
    ]


def test_continue_improve_uses_codegen_flow_without_bug_fixer(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-improve")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = NativeSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    first_request = AscendCAgenticCodegenRequest(
        definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-improve",
    )

    with runner.open_cycle(task=task, request=first_request, base_solution=None) as cycle:
        cycle.run_initial()
        improve_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="last_attempt.status=passed",
            target_gpu="ascend_910b", round_num=2, attempt_idx=2, mode="improve", run_id="native-improve",
        )
        cycle.request = improve_request
        cycle.continue_improve("continue from passed candidate and seek a smaller latency change")

    assert [prompt.splitlines()[0] for prompt in client.prompts[4:]] == [
        "Stage 1/3: improvement-assessor",
        "Stage 2/3: codegen",
        "Stage 3/3: reviewer",
    ]
    assert "bug-fixer" not in client.prompts[4]
    assert "continue_improve" in client.prompts[4]
    assert "IMPROVEMENT_ASSESSMENT.md" in client.prompts[4]


def test_continue_improve_no_op_preserves_current_candidate_without_source_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-improve-no-op")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = NoOpImproveSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    first_request = AscendCAgenticCodegenRequest(
        definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-improve-no-op",
    )

    with runner.open_cycle(task=task, request=first_request, base_solution=None) as cycle:
        first = cycle.run_initial()
        assert first.changed_paths == ["kernel/foo.h"]
        assert cycle.wt_session is not None
        cycle.wt_session.commit_all("accepted candidate baseline")
        improve_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="last_attempt.status=passed",
            target_gpu="ascend_910b", round_num=2, attempt_idx=2, mode="improve", run_id="native-improve-no-op",
        )
        cycle.request = improve_request
        second = cycle.continue_improve("continue from passed candidate only if evidence supports another edit")

    assert second.eval_result.status == "passed"
    assert second.changed_paths == []
    assert second.diff_text == ""
    assert "BETA" in next(src.content for src in second.solution.sources if src.path == "kernel/foo.h")


def test_continue_improve_requires_source_diff_when_assessor_requests_improvement(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-improve-requires-edit")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    client = NoOpImproveSessionClient(assessment_status="improve")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    first_request = AscendCAgenticCodegenRequest(
        definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-improve-requires-edit",
    )

    with runner.open_cycle(task=task, request=first_request, base_solution=None) as cycle:
        first = cycle.run_initial()
        assert first.changed_paths == ["kernel/foo.h"]
        assert cycle.wt_session is not None
        cycle.wt_session.commit_all("accepted candidate baseline")
        improve_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="last_attempt.status=passed",
            target_gpu="ascend_910b", round_num=2, attempt_idx=2, mode="improve", run_id="native-improve-requires-edit",
        )
        cycle.request = improve_request
        with pytest.raises(RuntimeError, match="did not change any files"):
            cycle.continue_improve("continue from passed candidate because assessor found an edit")


def test_run_multi_turn_uses_repair_flow_when_eval_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-repair-loop")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "text = Path('kernel/foo.h').read_text(); "
            "assert 'GAMMA' in text, text; "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )

    class RepairLoopClient:
        def __init__(self):
            self.prompts: list[str] = []

        def open_session(self, *, project_dir, telemetry_recorder=None):
            from types import SimpleNamespace

            return SimpleNamespace(_closed=False, project_dir=Path(project_dir))

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            first = prompt.splitlines()[0]
            if first == "Stage 1/4: code-reader":
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
                text = "reader done"
            elif first == "Stage 2/4: designer":
                (root / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
                text = "designer done"
            elif first == "Stage 3/4: codegen":
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
                (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text(
                    "# execution\nchange beta after source inspection\n",
                    encoding="utf-8",
                )
                (root / "IMPLEMENTATION_HANDOFF.md").write_text(
                    "# handoff\nchanged kernel/foo.h and preserved contracts\n",
                    encoding="utf-8",
                )
                text = "codegen done"
            elif first == "Stage 4/4: reviewer":
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "review done"
            elif first == "Stage 1/2: bug-fixer":
                assert (root / "CODE_MAP.md").is_file()
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\nGAMMA\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h fixed\n", encoding="utf-8")
                text = "bug fix done"
            elif first == "Stage 2/2: reviewer":
                (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
                text = "repair review done"
            else:
                raise AssertionError(first)
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    client = RepairLoopClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run_multi_turn(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-repair-loop",
        ),
        base_solution=None,
        max_fix_rounds=1,
    )
    assert result.eval_result.status == "passed"
    assert not hasattr(result, "editor_session")
    assert not hasattr(result, "worktree_session")
    assert [prompt.splitlines()[0] for prompt in client.prompts] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: designer",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
        "Stage 1/2: bug-fixer",
        "Stage 2/2: reviewer",
    ]
    assert "GAMMA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
    assert "+GAMMA" in result.diff_text


def test_continue_fix_filters_debug_evidence_files_from_candidate_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "native-debug-filter")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd("print('build ok')"),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )

    class DebugEvidenceSessionClient(NativeSessionClient):
        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            result = super().send_prompt(session, prompt=prompt, telemetry_recorder=telemetry_recorder)
            root = Path(session.project_dir)
            if "Stage 1/2: bug-fixer" in prompt:
                (root / "debug_packet.json").write_text('{"secret":"debug"}\n', encoding="utf-8")
                (root / "debug_log.md").write_text("debug notes\n", encoding="utf-8")
            return result

    client = DebugEvidenceSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    first_request = AscendCAgenticCodegenRequest(
        definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action", run_id="native-debug-filter",
    )

    with runner.open_cycle(task=task, request=first_request, base_solution=None) as cycle:
        first = cycle.run_initial()
        second_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=2, mode="debug", run_id="native-debug-filter",
        )
        cycle.request = second_request
        second = cycle.continue_fix("raw compile fix context")

    assert second.changed_paths == ["kernel/foo.h"]
    assert "debug_packet.json" not in second.diff_text
    assert "debug_log.md" not in second.diff_text
    assert "debug_packet.json" not in {src.path for src in second.solution.sources}
    if second.project_snapshot is not None:
        assert "debug_packet.json" not in second.project_snapshot.manifest
        assert "debug_log.md" not in second.project_snapshot.manifest
