import json
import shlex
import sys

from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.kernel_generators.llm_clients import ClaudeAgentLLMClient
from k_search.tasks.ascendc_task import AscendCTask
from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder
from k_search.testing import (
    MockClaudeMessage,
    install_mock_claude_agent_sdk,
)


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_mock_claude_agent_sdk_records_options_and_streams_messages(monkeypatch):
    sdk = install_mock_claude_agent_sdk(
        monkeypatch,
        responses=[
            [
                MockClaudeMessage(content=[{"type": "text", "text": "assistant chunk"}]),
                MockClaudeMessage(result="final result"),
            ]
        ],
    )
    client = ClaudeAgentLLMClient(
        model_name="claude-sonnet-4-6",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
    )

    assert client.generate("optimize this") == "final result"

    assert sdk.calls[0].prompt == "optimize this"
    assert sdk.calls[0].options.kwargs["model"] == "claude-sonnet-4-6"
    assert sdk.calls[0].options.kwargs["allowed_tools"] == ["Read"]
    assert sdk.calls[0].options.kwargs["disallowed_tools"] == ["Bash"]


def test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools(monkeypatch, tmp_path):
    from pathlib import Path
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        target = project_dir / "kernel" / "foo.h"
        target.write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        return [
            MockClaudeMessage(content=[{"type": "text", "text": "edited foo.h"}]),
            MockClaudeMessage(result="final summary"),
        ]

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude-sonnet-4-6", timeout_seconds=30)

    result = client.edit_project(project_dir=tmp_path, prompt="Please edit the project.")

    assert result.text == "final summary"
    assert result.transcript == "edited foo.h\nfinal summary"
    assert (tmp_path / "kernel" / "foo.h").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert len(sdk.client_calls) == 1
    call = sdk.client_calls[0]
    assert call.prompt == "Please edit the project."
    assert call.options.kwargs["cwd"] == str(tmp_path)
    assert call.options.kwargs["allowed_tools"] == ["Read", "Grep", "Glob", "Edit", "Write", "Skill", "Agent"]
    assert call.options.kwargs["disallowed_tools"][0] == "Bash"
    assert call.options.kwargs["permission_mode"] == "acceptEdits"
    assert call.options.kwargs["model"] == "claude-sonnet-4-6"


def test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    from pathlib import Path

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (tmp_path / "spec.md").write_text("Optimize a tiny AscendC project.", encoding="utf-8")
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    def first_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
        (project_dir / "IMPLEMENTATION_PLAN.md").write_text("# IMPLEMENTATION_PLAN\nInitial safe edit.\n", encoding="utf-8")
        (project_dir / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
        (project_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n// initial agent edit\n", encoding="utf-8")
        return "status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval"

    def second_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
        (project_dir / "IMPLEMENTATION_PLAN.md").write_text("# IMPLEMENTATION_PLAN\nChange beta to BETA.\n", encoding="utf-8")
        (project_dir / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        return "status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval"

    sdk = install_mock_claude_agent_sdk(
        monkeypatch,
        responses=[first_edit, first_edit, first_edit, first_edit, second_edit, second_edit, second_edit, second_edit],
    )

    task = AscendCTask(
        task_path=tmp_path,
        definition_name="mock_ascendc",
        codegen_mode="auto",
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel/foo.h').exists(); "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd(
            "from pathlib import Path; "
            "text = Path('kernel/foo.h').read_text(); "
            "print('latency_ms=0.5' if 'BETA' in text else 'latency_ms=1.0')"
        ),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    generator = KernelGenerator(
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
    )

    solution = generator.generate(task=task, max_opt_rounds=2)

    foo = next(src for src in solution.sources if src.path == "kernel/foo.h")
    assert "BETA" in foo.content
    assert len(sdk.client_calls) == 8
    assert sdk.calls == []
    assert all("<ascendc_project>" not in call.prompt for call in sdk.client_calls)
    assert [call.prompt.splitlines()[0] for call in sdk.client_calls[:4]] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: plan",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
    ]
    assert [call.prompt.splitlines()[0] for call in sdk.client_calls[4:]] == [
        "Stage 1/4: code-reader",
        "Stage 2/4: plan",
        "Stage 3/4: codegen",
        "Stage 4/4: reviewer",
    ]
    assert sdk.client_calls[0].options.kwargs["cwd"]
    assert sdk.client_calls[0].options.kwargs["setting_sources"] == ["project"]
    assert "Skill" in sdk.client_calls[0].options.kwargs["allowed_tools"]
    assert "Agent" in sdk.client_calls[0].options.kwargs["allowed_tools"]


def test_claude_project_editor_writes_tool_timeline_and_cost(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "project" / "kernel").mkdir(parents=True)
    (tmp_path / "project" / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return [
            MockClaudeMessage(content=[SimpleNamespace(type="tool_use", id="toolu_1", name="Glob", input={"pattern": "**/*.h"})]),
            MockClaudeMessage(content=[SimpleNamespace(type="tool_result", tool_use_id="toolu_1", content="kernel/foo.h", is_error=False)]),
            MockClaudeMessage(content=[SimpleNamespace(type="tool_use", id="toolu_2", name="Read", input={"file_path": "kernel/foo.h"})]),
            MockClaudeMessage(content=[{"type": "text", "text": "edited foo.h"}]),
            MockClaudeMessage(
                result="final summary",
                session_id="sess-1",
                total_cost_usd=0.125,
                usage={"input_tokens": 10},
                model_usage={"claude": {"output_tokens": 5}},
                duration_ms=1000,
                duration_api_ms=800,
                num_turns=3,
            ),
        ]

    install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    recorder = build_file_recorder(
        context=TelemetryContext(task_name="task", run_id="run", round_index=1, attempt_index=1, model_name="claude"),
        prompt="Please edit the project.",
        root=tmp_path / "telemetry",
    )
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    result = client.edit_project(
        project_dir=tmp_path / "project",
        prompt="Please edit the project.",
        telemetry_recorder=recorder,
    )
    recorder.close()

    assert result.text == "final summary"
    assert result.trace_path == recorder.artifacts.trace_path
    assert result.timeline_path == recorder.artifacts.timeline_path
    assert result.cost_path == recorder.artifacts.cost_path
    assert result.session_id == "sess-1"
    assert result.total_cost_usd == 0.125
    rows = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows if row["event_type"] == "tool_use"] == ["tool_use", "tool_use"]
    assert "tool_use: Glob" in Path(result.timeline_path).read_text(encoding="utf-8")
    assert json.loads(Path(result.cost_path).read_text(encoding="utf-8"))["summary"]["session_id"] == "sess-1"


def test_claude_project_editor_enables_project_skills_and_agent_tool(monkeypatch, tmp_path):
    from pathlib import Path
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return "edited"

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    client.edit_project(project_dir=tmp_path, prompt="Use native subagents.")

    options = sdk.client_calls[0].options.kwargs
    assert options["setting_sources"] == ["project"]
    assert options["skills"] == ["ascendc-codegen", "ascendc-api-reference"]
    assert "Skill" in options["allowed_tools"]
    assert "Agent" in options["allowed_tools"]
    assert "Bash" in options["disallowed_tools"]


def test_claude_project_editor_session_uses_same_native_options(monkeypatch, tmp_path):
    from pathlib import Path
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return "edited"

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    session = client.open_session(project_dir=tmp_path)
    try:
        client.send_prompt(session, prompt="Use native subagents.")
    finally:
        client.close_session(session)

    options = sdk.client_calls[0].options.kwargs
    assert options["setting_sources"] == ["project"]
    assert options["skills"] == ["ascendc-codegen", "ascendc-api-reference"]
    assert "Skill" in options["allowed_tools"]
    assert "Agent" in options["allowed_tools"]
