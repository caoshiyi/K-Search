import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import k_search.kernel_generators.ascendc_agentic_codegen as codegen
from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenResult,
    AscendCAgenticCodegenRunner,
)
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.tasks.ascendc_task import AscendCTask
from k_search.tasks.task_base import EvalResult


def _request(*, run_id: str = "lifecycle", attempt_idx: int = 1, mode: str = "action") -> AscendCAgenticCodegenRequest:
    return AscendCAgenticCodegenRequest(
        definition_text="spec",
        action_text="change kernel",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=attempt_idx,
        mode=mode,  # type: ignore[arg-type]
        run_id=run_id,
    )


def _make_task(tmp_path: Path, *, statuses: list[str] | None = None) -> AscendCTask:
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    status_queue = list(statuses or ["passed"])

    class LifecycleTask(AscendCTask):
        def run_benchmark_in_project_dir(self, *, project_dir, round_num=None, dump_traces=False):
            del project_dir, round_num, dump_traces
            status = status_queue.pop(0) if status_queue else "passed"
            return EvalResult(status=status, log_excerpt=f"{status} log", metrics={"score": 1.0})

    return LifecycleTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))


class FakeWorktree:
    def __init__(self, project_dir: Path):
        self.worktree_root = project_dir
        self.project_dir = project_dir
        self.baseline_commit = "baseline"
        self.cleaned = False

    def commit_all(self, message):
        self.baseline_commit = f"commit:{message}"
        return self.baseline_commit

    def changed_paths(self):
        return ["kernel/foo.h"]

    def project_changed_paths(self):
        return ["kernel/foo.h"]

    def diff_text(self):
        return self.project_diff_text()

    def project_diff_text(self):
        return "--- a/kernel/foo.h\n+++ b/kernel/foo.h\n@@\n-beta\n+BETA\n"

    def project_rel_path(self):
        return "."

    def cleanup(self):
        self.cleaned = True


class FakeSessionClient:
    def __init__(self, *, fail_flow: bool = False):
        self.fail_flow = fail_flow
        self.sessions: list[SimpleNamespace] = []
        self.closed_sessions: list[SimpleNamespace] = []
        self.prompts: list[str] = []
        self.send_count = 0

    def open_session(self, *, project_dir, telemetry_recorder=None):
        del telemetry_recorder
        session = SimpleNamespace(_closed=False, project_dir=Path(project_dir))
        self.sessions.append(session)
        return session

    def send_prompt(self, session, *, prompt, telemetry_recorder=None):
        del telemetry_recorder
        self.send_count += 1
        self.prompts.append(prompt)
        if self.fail_flow:
            raise RuntimeError("flow failed")
        root = Path(session.project_dir)
        first_line = prompt.splitlines()[0] if prompt else ""
        if "Stage 1/" in first_line and "code-reader" in first_line:
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
            text = "reader"
        elif "Stage 2/" in first_line and "designer" in first_line:
            (root / "ASCENDC_DESIGN.md").write_text("# design\n", encoding="utf-8")
            text = "design"
        elif "codegen" in first_line:
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
            (root / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text("# execution\n", encoding="utf-8")
            (root / "IMPLEMENTATION_HANDOFF.md").write_text("# handoff\n", encoding="utf-8")
            text = "codegen"
        elif "bug-fixer" in first_line:
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\nGAMMA\n", encoding="utf-8")
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h fixed\n", encoding="utf-8")
            text = "fix"
        elif "reviewer" in first_line:
            (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
            text = "review"
        else:
            raise AssertionError(first_line)
        return ClaudeProjectEditResult(
            text=text,
            transcript=text,
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )

    def close_session(self, session):
        session._closed = True
        self.closed_sessions.append(session)


@pytest.fixture
def lifecycle_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")


def _install_fake_worktree(monkeypatch, tmp_path: Path) -> FakeWorktree:
    worktree_dir = tmp_path / "worktree"
    (worktree_dir / "kernel").mkdir(parents=True)
    (worktree_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    worktree = FakeWorktree(worktree_dir)
    monkeypatch.setattr(
        codegen,
        "create_agentic_worktree",
        lambda *, task_path, worktree_parent_dir=None: worktree,
    )
    monkeypatch.setattr(codegen, "_materialize_native_assets_baseline", lambda wt_session: None)
    return worktree


def test_result_dataclass_does_not_hold_live_resources():
    fields = {field.name for field in dataclasses.fields(AscendCAgenticCodegenResult)}

    assert "editor_session" not in fields
    assert "worktree_session" not in fields


def test_run_one_shot_closed_closes_session_and_cleans_worktree_on_success(tmp_path, monkeypatch, lifecycle_env):
    worktree = _install_fake_worktree(monkeypatch, tmp_path)
    client = FakeSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run_one_shot_closed(
        task=_make_task(tmp_path),
        request=_request(),
        base_solution=None,
        max_fix_rounds=0,
    )

    assert not hasattr(result, "editor_session")
    assert not hasattr(result, "worktree_session")
    assert len(client.sessions) == 1
    assert client.closed_sessions == client.sessions
    assert worktree.cleaned is True


def test_run_one_shot_closed_closes_session_and_cleans_worktree_on_exception(tmp_path, monkeypatch, lifecycle_env):
    monkeypatch.setenv("KSEARCH_TASK_ID", "taskid")
    worktree = _install_fake_worktree(monkeypatch, tmp_path)
    client = FakeSessionClient(fail_flow=True)
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    with pytest.raises(RuntimeError, match="flow failed"):
        runner.run_one_shot_closed(
            task=_make_task(tmp_path),
            request=_request(),
            base_solution=None,
            max_fix_rounds=0,
        )

    assert len(client.sessions) == 1
    assert client.closed_sessions == client.sessions
    assert worktree.cleaned is True

    manifest_path = (
        tmp_path
        / "artifacts"
        / "x"
        / "taskid"
        / "runs"
        / "lifecycle"
        / "artifacts"
        / "candidates"
        / "round_0001_attempt_01"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"]["stage"] == "action"
    assert manifest["failure"]["error_type"] == "RuntimeError"
    assert "flow failed" in manifest["failure"]["error_message"]
    assert manifest["stage_prompt_paths"]


def test_open_cycle_reuses_session_until_context_exit(tmp_path, monkeypatch, lifecycle_env):
    worktree = _install_fake_worktree(monkeypatch, tmp_path)
    client = FakeSessionClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    with runner.open_cycle(task=_make_task(tmp_path), request=_request(), base_solution=None) as cycle:
        first = cycle.run_initial()
        second = cycle.continue_fix("raw compile fix context")
        assert first.eval_result.status == "passed"
        assert second.eval_result.status == "passed"
        assert len(client.sessions) == 1
        assert client.closed_sessions == []
        assert worktree.cleaned is False

    assert client.closed_sessions == client.sessions
    assert worktree.cleaned is True
