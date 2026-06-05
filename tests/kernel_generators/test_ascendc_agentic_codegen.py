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


def _write_native_handoffs(root: Path, code_map: str = "# CODE_MAP\nkernel/foo.h\n") -> None:
    (root / "CODE_MAP.md").write_text(code_map, encoding="utf-8")
    (root / "IMPLEMENTATION_PLAN.md").write_text(
        "# IMPLEMENTATION_PLAN\nApply the requested focused source edit.\n",
        encoding="utf-8",
    )
    (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _code_map_disabled_by_default(monkeypatch):
    # Keep memory-store tests hermetic unless they explicitly opt into persisted code_map behavior.
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")


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
        write_plan: bool = True,
        write_review: bool = True,
        review_text: str = "status: ok\neval_ready: true\n",
    ):
        self.new_text = new_text
        self.write_code_map = write_code_map
        self.write_plan = write_plan
        self.write_review = write_review
        self.review_text = review_text
        self.calls = []
        self.assets_seen: dict[str, bool] = {}

    def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        self.assets_seen = {
            "code_reader": (root / ".claude" / "agents" / "code-reader.md").exists(),
            "plan": (root / ".claude" / "agents" / "plan.md").exists(),
            "codegen": (root / ".claude" / "agents" / "codegen.md").exists(),
            "reviewer": (root / ".claude" / "agents" / "reviewer.md").exists(),
            "bug_fixer": (root / ".claude" / "agents" / "bug-fixer.md").exists(),
            "ascendc_codegen_skill": (root / ".claude" / "skills" / "ascendc-codegen" / "SKILL.md").exists(),
            "ascendc_api_reference_skill": (root / ".claude" / "skills" / "ascendc-api-reference" / "SKILL.md").exists(),
        }
        if self.write_code_map:
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h is the kernel file\n", encoding="utf-8")
        if self.write_plan:
            (root / "IMPLEMENTATION_PLAN.md").write_text(
                "# IMPLEMENTATION_PLAN\nChange beta to BETA in kernel/foo.h.\n",
                encoding="utf-8",
            )
        if self.write_review:
            (root / "REVIEW_NOTES.md").write_text(self.review_text, encoding="utf-8")
        (root / "kernel" / "foo.h").write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval",
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
        text = "alpha\nBETA\ngamma\n" if len(self.prompts) == 1 else "alpha\nGAMMA\ngamma\n"
        (root / "kernel" / "foo.h").write_text(text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval",
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
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    store = MemoryStore.for_task(task)
    assert store.load(CODE_MAP) is not None
    assert result.code_map_text is not None
    assert "CODE_MAP.md" not in result.changed_paths
    assert "kernel/foo.h" in result.changed_paths


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
    MemoryStore.for_task(task).save(CODE_MAP, "# CODE_MAP\npreseeded\n")

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
    assert "code-reader -> plan -> codegen -> reviewer" in prompt
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
            elif "Stage 2/4: plan" in prompt:
                assert (root / "CODE_MAP.md").exists()
                assert "Use the plan subagent" in prompt
                (root / "IMPLEMENTATION_PLAN.md").write_text("# plan\nchange beta\n", encoding="utf-8")
                text = "plan done"
            elif "Stage 3/4: codegen" in prompt:
                assert (root / "IMPLEMENTATION_PLAN.md").exists()
                assert "Use the codegen subagent" in prompt
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
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
        "Stage 2/4: plan",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
    ]
    assert result.changed_paths == ["kernel/foo.h"]
    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")


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


def test_runner_fails_when_implementation_plan_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_plan=False))

    with pytest.raises(RuntimeError, match="IMPLEMENTATION_PLAN.md"):
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


def test_runner_fails_when_reviewer_marks_candidate_not_eval_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=NativeEditingClient(
            review_text="status: needs_fix\neval_ready: false\nrequired_fixes: fix tiling contract\n",
        ),
    )

    with pytest.raises(RuntimeError, match="REVIEW_NOTES.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


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
    assert "CODE_MAP.md" not in result.diff_text
    assert "IMPLEMENTATION_PLAN.md" not in result.diff_text
    assert "REVIEW_NOTES.md" not in result.diff_text
    assert ".claude/agents" not in result.diff_text
    assert result.project_snapshot is not None
    assert "CODE_MAP.md" not in result.project_snapshot.manifest
    assert "IMPLEMENTATION_PLAN.md" not in result.project_snapshot.manifest
    assert "REVIEW_NOTES.md" not in result.project_snapshot.manifest
    assert not any(path.startswith(".claude/") for path in result.project_snapshot.manifest)
    assert result.artifact_paths is not None
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    handoff_paths = manifest["native_handoff_paths"]
    assert sorted(handoff_paths) == ["CODE_MAP.md", "IMPLEMENTATION_PLAN.md", "REVIEW_NOTES.md"]
    assert "Change beta to BETA" in Path(handoff_paths["IMPLEMENTATION_PLAN.md"]).read_text(encoding="utf-8")


def test_continue_fix_uses_native_prompt_and_run_scoped_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
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

    first = runner.run_multi_turn(task=task, request=first_request, base_solution=None, max_fix_rounds=0)
    try:
        second_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=2, mode="debug", run_id="native-continue",
        )
        second = runner.continue_fix(
            task=task,
            editor_session=first.editor_session,
            wt_session=first.worktree_session,
            fix_prompt="raw compile fix context",
            request=second_request,
        )
    finally:
        if first.editor_session is not None:
            client.close_session(first.editor_session)
        if first.worktree_session is not None:
            first.worktree_session.cleanup()

    assert len(client.prompts) == 6
    assert [prompt.splitlines()[0] for prompt in client.prompts[:4]] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: plan",
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
    assert "/runs/native-continue/" in second.artifact_paths["manifest_path"]


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

    first = runner.run_multi_turn(task=task, request=first_request, base_solution=None, max_fix_rounds=0)
    try:
        second_request = AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="continue action", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=2, mode="debug", run_id="native-debug-filter",
        )
        second = runner.continue_fix(
            task=task,
            editor_session=first.editor_session,
            wt_session=first.worktree_session,
            fix_prompt="raw compile fix context",
            request=second_request,
        )
    finally:
        if first.editor_session is not None:
            client.close_session(first.editor_session)
        if first.worktree_session is not None:
            first.worktree_session.cleanup()

    assert second.changed_paths == ["kernel/foo.h"]
    assert "debug_packet.json" not in second.diff_text
    assert "debug_log.md" not in second.diff_text
    assert "debug_packet.json" not in {src.path for src in second.solution.sources}
    if second.project_snapshot is not None:
        assert "debug_packet.json" not in second.project_snapshot.manifest
        assert "debug_log.md" not in second.project_snapshot.manifest
