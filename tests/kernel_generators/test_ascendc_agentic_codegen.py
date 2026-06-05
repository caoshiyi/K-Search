import json
import shlex
import sys
from pathlib import Path

import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenRunner,
    AscendCAgenticPromptBuilder,
)
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.tasks.ascendc_task import AscendCTask
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


@pytest.fixture(autouse=True)
def _code_map_disabled_by_default(monkeypatch):
    # Keep agentic runner tests hermetic and fast: with code_map enabled the runner
    # would fire a real CodeReaderAgent SDK call on the first round. Tests that
    # exercise code_map opt back in with monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1").
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")


class EditingClient:
    def __init__(self, new_text: str):
        self.new_text = new_text
        self.calls = []

    def edit_project(self, *, project_dir, prompt):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        target = root / "kernel" / "foo.h"
        target.write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="edited kernel/foo.h",
            transcript="located and edited kernel/foo.h",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


class NoChangeClient:
    def edit_project(self, *, project_dir, prompt):
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
    assert "code-reader -> plan -> codegen -> reviewer" in prompt
    assert "IMPLEMENTATION_PLAN.md" in prompt
    assert "REVIEW_NOTES.md" in prompt
    assert "bug-fixer" in prompt
    assert "must not be invoked" in prompt


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


def test_runner_evaluates_worktree_and_persists_project_snapshot_candidate(tmp_path):
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
            action_node_id="A-12",
        ),
        base_solution=None,
    )

    assert result.eval_result.status == "passed"
    assert result.eval_result.metrics["score"] == 2.0
    assert result.eval_result.metrics["workdir"] == result.project_path
    assert "build saw edited complete worktree" in result.eval_result.log_excerpt
    assert result.candidate_patch is not None
    assert result.candidate_patch.action_node_id == "A-12"
    assert result.project_snapshot is not None
    assert "kernel/large_header.hpp" in result.project_snapshot.manifest
    assert result.artifact_paths is not None
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == result.candidate_patch.candidate_id
    assert manifest["snapshot_id"] == result.project_snapshot.snapshot_id
    assert Path(result.artifact_paths["diff_path"]).read_text(encoding="utf-8") == result.diff_text
    assert json.loads(Path(result.artifact_paths["eval_path"]).read_text(encoding="utf-8"))["status"] == "passed"


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
            Path(project_dir, "kernel", "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
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
    assert "Use the code-reader subagent to create CODE_MAP.md" in without_map


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

    assert "CODE_MAP.md, IMPLEMENTATION_PLAN.md, and REVIEW_NOTES.md are the only trusted cross-subagent handoff" in prompt
    assert "status, files_written, and next" in prompt
    assert "Do not paste CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md, or source files" in prompt


def test_runner_generates_and_persists_code_map_on_first_round(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
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
            if "CODE_MAP.md using EXACTLY" in prompt:
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nfoo.h is the kernel\n", encoding="utf-8")
                text = "wrote CODE_MAP.md"
            else:
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "edited"
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

    assert any("CODE_MAP.md using EXACTLY" in p for p in client.prompts)
    assert any("Read it first instead of grepping" in p for p in client.prompts)
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    store = MemoryStore.for_task(task)
    assert store.load(CODE_MAP) is not None
    assert result.code_map_text is not None
    assert "CODE_MAP.md" not in result.changed_paths
    assert "kernel/foo.h" in result.changed_paths


def test_runner_reuses_existing_code_map_without_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    MemoryStore.for_task(task).save(CODE_MAP, "# CODE_MAP\npreseeded\n")

    class CodegenOnlyClient:
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            assert "CODE_MAP.md using EXACTLY" not in prompt
            root = Path(project_dir)
            assert (root / "CODE_MAP.md").read_text(encoding="utf-8") == "# CODE_MAP\npreseeded\n"
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
        ),
        base_solution=None,
    )
    assert len(client.prompts) == 1
    assert "Read it first instead of grepping" in client.prompts[0]
    assert "CODE_MAP.md" not in result.changed_paths


def test_runner_code_map_disabled_keeps_legacy_behavior(tmp_path, monkeypatch):
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
    assert "First inspect the project" in client.calls[0][1]
    assert "CODE_MAP.md" not in client.calls[0][1]
    assert result.code_map_text is None


def test_runner_code_map_not_in_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class C:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            if "CODE_MAP.md using EXACTLY" in prompt:
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nmapped\n", encoding="utf-8")
                t = "wrote map"
            else:
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
