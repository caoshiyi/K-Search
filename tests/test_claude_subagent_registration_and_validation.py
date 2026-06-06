from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import asyncio

import pytest


def test_parse_native_agent_markdown_frontmatter_lists_and_prompt():
    from k_search.kernel_generators.claude_assets.agent_definitions import _spec_from_markdown

    spec = _spec_from_markdown(
        "code-reader.md",
        """
---
name: code-reader
description: Reads code
tools: Read, Grep, Glob, Write
model: sonnet
skills: ascendc-codegen, ascendc-api-reference
---
Prompt body here.
""".lstrip(),
    )

    assert spec.name == "code-reader"
    assert spec.description == "Reads code"
    assert spec.tools == ["Read", "Grep", "Glob", "Write"]
    assert spec.skills == ["ascendc-codegen", "ascendc-api-reference"]
    assert spec.model == "sonnet"
    assert spec.prompt == "Prompt body here."


def test_load_native_agent_specs_parses_block_style_skills():
    from k_search.kernel_generators.claude_assets.agent_definitions import load_native_agent_specs

    specs = load_native_agent_specs()

    assert "designer" in specs
    assert specs["designer"].skills is not None
    assert "ascendc-hardware" in specs["designer"].skills
    assert "ascendc-dev-knowledge" in specs["designer"].skills


def test_agent_definition_from_spec_filters_unknown_dataclass_fields(monkeypatch):
    from k_search.kernel_generators.claude_assets import agent_definitions
    from k_search.kernel_generators.claude_assets.agent_definitions import NativeAgentSpec

    @dataclass
    class FakeAgentDefinition:
        description: str
        prompt: str
        tools: list[str] | None = None

    monkeypatch.setattr(agent_definitions, "_build_agent_definition_cls", lambda: FakeAgentDefinition)

    definition = agent_definitions._agent_definition_from_spec(
        NativeAgentSpec(
            name="codegen",
            description="Writes code",
            prompt="Implement the design.",
            tools=["Read", "Write"],
            model="ignored-by-fake",
        )
    )

    assert definition == FakeAgentDefinition(
        description="Writes code",
        prompt="Implement the design.",
        tools=["Read", "Write"],
    )


def test_require_agent_tool_invocation_accepts_agent_tool_event():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(
                event_type="tool_use",
                tool_name="Agent",
                tool_input={"subagent_type": "codegen"},
            )
        ]
    )

    _require_agent_tool_invocation(
        telemetry_recorder=recorder,
        event_start=0,
        stage=SubagentStageConfig(name="codegen", agent="codegen", instruction="Write code."),
    )


def test_require_agent_tool_invocation_accepts_task_tool_event():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(
                event_type="tool_use",
                tool_name="Task",
                tool_input={"subagent_type": "codegen"},
            )
        ]
    )

    _require_agent_tool_invocation(
        telemetry_recorder=recorder,
        event_start=0,
        stage=SubagentStageConfig(name="codegen", agent="codegen", instruction="Write code."),
    )


def test_require_agent_tool_invocation_accepts_nested_agent_name():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(
                event_type="tool_use",
                tool_name="Agent",
                tool_input={"input": {"agent_name": "reviewer"}},
            )
        ]
    )

    _require_agent_tool_invocation(
        telemetry_recorder=recorder,
        event_start=0,
        stage=SubagentStageConfig(name="reviewer", agent="reviewer", instruction="Review."),
    )


def test_require_agent_tool_invocation_reports_wrong_subagent():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(
                event_type="tool_use",
                tool_name="Agent",
                tool_input={"subagent_type": "designer"},
            )
        ]
    )

    with pytest.raises(RuntimeError) as exc:
        _require_agent_tool_invocation(
            telemetry_recorder=recorder,
            event_start=0,
            stage=SubagentStageConfig(name="codegen", agent="codegen", instruction="Write code."),
        )

    message = str(exc.value)
    assert "stage='codegen'" in message
    assert "expected_agent='codegen'" in message
    assert "observed_calls" in message
    assert "designer" in message


def test_require_agent_tool_invocation_accepts_duplicate_matching_calls():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(event_type="tool_use", tool_name="Agent", tool_input={"subagent_type": "reviewer"}),
            SimpleNamespace(event_type="tool_use", tool_name="Agent", tool_input={"agent": "plugin:reviewer"}),
        ]
    )

    _require_agent_tool_invocation(
        telemetry_recorder=recorder,
        event_start=0,
        stage=SubagentStageConfig(name="reviewer", agent="reviewer", instruction="Review."),
    )


def test_require_agent_tool_invocation_rejects_mixed_subagents_in_stage():
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _require_agent_tool_invocation,
    )

    recorder = SimpleNamespace(
        events=[
            SimpleNamespace(event_type="tool_use", tool_name="Agent", tool_input={"subagent_type": "reviewer"}),
            SimpleNamespace(event_type="tool_use", tool_name="Agent", tool_input={"subagent_type": "codegen"}),
        ]
    )

    with pytest.raises(RuntimeError) as exc:
        _require_agent_tool_invocation(
            telemetry_recorder=recorder,
            event_start=0,
            stage=SubagentStageConfig(name="reviewer", agent="reviewer", instruction="Review."),
        )

    message = str(exc.value)
    assert "matching_calls=1" in message
    assert "total_subagent_calls=2" in message
    assert "codegen" in message


def test_claude_project_editor_build_options_injects_programmatic_agents(monkeypatch, tmp_path):
    import k_search.kernel_generators.claude_agent_project_editor as editor_module
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    fake_agent_def = object()
    monkeypatch.delenv("KSEARCH_DISABLE_PROGRAMMATIC_AGENTS", raising=False)
    monkeypatch.setattr(
        editor_module,
        "load_native_agent_definitions",
        lambda *, enabled_agent_names=None: {"codegen": fake_agent_def},
    )

    client = ClaudeAgentProjectEditorClient(
        model_name="claude",
        native_agents=["codegen"],
        timeout_seconds=30,
    )

    options = client._build_options_kwargs(tmp_path)

    assert options["agents"]["codegen"] is fake_agent_def
    assert any(tool.startswith("Agent(") or tool == "Agent" for tool in options["tools"])
    assert any(tool.startswith("Agent(") or tool == "Agent" for tool in options["allowed_tools"])


def test_project_tool_permission_callback_accepts_task_tool_and_nested_agent(monkeypatch, tmp_path):
    import k_search.kernel_generators.claude_agent_project_editor as editor_module
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    monkeypatch.setenv("KSEARCH_DISABLE_PROGRAMMATIC_AGENTS", "1")
    monkeypatch.setattr(
        editor_module,
        "load_native_agent_definitions",
        lambda *, enabled_agent_names=None: pytest.fail("programmatic agent loader should not be called"),
    )
    client = ClaudeAgentProjectEditorClient(
        model_name="claude",
        native_agents=["reviewer"],
        timeout_seconds=30,
    )
    can_use_tool = client._build_options_kwargs(tmp_path)["can_use_tool"]

    allowed = asyncio.run(can_use_tool("Task", {"input": {"agent_name": "reviewer"}}, None))
    denied = asyncio.run(can_use_tool("Task", {"params": {"agent_name": "general-purpose"}}, None))

    assert _permission_behavior(allowed) == "allow"
    assert _permission_behavior(denied) == "deny"


def test_claude_project_editor_build_options_omits_agents_when_disabled(monkeypatch, tmp_path):
    import k_search.kernel_generators.claude_agent_project_editor as editor_module
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    monkeypatch.setenv("KSEARCH_DISABLE_PROGRAMMATIC_AGENTS", "1")
    monkeypatch.setattr(
        editor_module,
        "load_native_agent_definitions",
        lambda *, enabled_agent_names=None: pytest.fail("programmatic agent loader should not be called"),
    )

    options = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)._build_options_kwargs(tmp_path)

    assert "agents" not in options


def test_claude_project_editor_programmatic_agents_fallback_and_strict(monkeypatch, tmp_path):
    import k_search.kernel_generators.claude_agent_project_editor as editor_module
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    def fail_loader(*, enabled_agent_names=None):
        raise RuntimeError("agent definition unavailable")

    monkeypatch.delenv("KSEARCH_DISABLE_PROGRAMMATIC_AGENTS", raising=False)
    monkeypatch.delenv("KSEARCH_REQUIRE_PROGRAMMATIC_AGENTS", raising=False)
    monkeypatch.setattr(editor_module, "load_native_agent_definitions", fail_loader)

    options = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)._build_options_kwargs(tmp_path)
    assert "agents" not in options

    monkeypatch.setenv("KSEARCH_REQUIRE_PROGRAMMATIC_AGENTS", "1")
    with pytest.raises(RuntimeError, match="agent definition unavailable"):
        ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)._build_options_kwargs(tmp_path)


def test_stage_completion_warns_for_missing_marker_by_default(tmp_path, caplog, monkeypatch):
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _validate_stage_completion,
    )

    monkeypatch.delenv("KSEARCH_REQUIRE_STAGE_MARKERS", raising=False)
    (tmp_path / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")

    _validate_stage_completion(
        project_root=tmp_path,
        stage=SubagentStageConfig(
            name="reviewer",
            agent="reviewer",
            instruction="Review.",
            required_files=("REVIEW_NOTES.md",),
        ),
        telemetry_recorder=SimpleNamespace(enabled=False, events=[]),
        event_start=0,
        require_agent_tool_use=True,
    )

    assert "stage handoff marker missing" in caplog.text
    assert "reviewer" in caplog.text


def test_stage_completion_strict_marker_env_fails(tmp_path, monkeypatch):
    from k_search.kernel_generators.subagent_orchestration import (
        SubagentStageConfig,
        _validate_stage_completion,
    )

    monkeypatch.setenv("KSEARCH_REQUIRE_STAGE_MARKERS", "1")
    (tmp_path / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stage handoff marker missing"):
        _validate_stage_completion(
            project_root=tmp_path,
            stage=SubagentStageConfig(
                name="reviewer",
                agent="reviewer",
                instruction="Review.",
                required_files=("REVIEW_NOTES.md",),
            ),
            telemetry_recorder=SimpleNamespace(enabled=False, events=[]),
            event_start=0,
            require_agent_tool_use=True,
        )


def test_native_handoff_validation_accepts_review_notes_markdown_heading_fields(tmp_path):
    from k_search.kernel_generators.ascendc_agentic_codegen import _require_native_handoff_files

    (tmp_path / "REVIEW_NOTES.md").write_text(
        "## status: ok\n\n"
        "## changed_files_reviewed\n"
        "- kernel/foo.h\n\n"
        "### required_fixes: none\n\n"
        "## eval_ready: true\n",
        encoding="utf-8",
    )

    handoffs = _require_native_handoff_files(
        tmp_path,
        required_files={"REVIEW_NOTES.md"},
    )

    assert "## status: ok" in handoffs["REVIEW_NOTES.md"]


def test_native_handoff_validation_rejects_markdown_heading_not_eval_ready(tmp_path):
    from k_search.kernel_generators.ascendc_agentic_codegen import _require_native_handoff_files

    (tmp_path / "REVIEW_NOTES.md").write_text(
        "## status: needs_fix\n\n"
        "## required_fixes: fix tiling contract\n\n"
        "## eval_ready: false\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="REVIEW_NOTES.md"):
        _require_native_handoff_files(
            tmp_path,
            required_files={"REVIEW_NOTES.md"},
        )


def test_native_handoff_validation_warns_for_short_code_map_by_default(tmp_path, caplog, monkeypatch):
    from k_search.kernel_generators.ascendc_agentic_codegen import _require_native_handoff_files

    monkeypatch.delenv("KSEARCH_STRICT_HANDOFF_VALIDATION", raising=False)
    (tmp_path / "CODE_MAP.md").write_text("tiny\n", encoding="utf-8")
    (tmp_path / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
    (tmp_path / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text("# execution\nedit x\n", encoding="utf-8")
    (tmp_path / "IMPLEMENTATION_HANDOFF.md").write_text("# handoff\nchanged x\n", encoding="utf-8")
    (tmp_path / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")

    handoffs = _require_native_handoff_files(
        tmp_path,
        required_files={
            "CODE_MAP.md",
            "ASCENDC_DESIGN.md",
            "IMPLEMENTATION_EXECUTION_PLAN.md",
            "IMPLEMENTATION_HANDOFF.md",
            "REVIEW_NOTES.md",
        },
    )

    assert handoffs["CODE_MAP.md"] == "tiny\n"
    assert "CODE_MAP.md is too short to be useful" in caplog.text


def test_native_handoff_validation_strict_env_fails_for_short_execution_plan(tmp_path, monkeypatch):
    from k_search.kernel_generators.ascendc_agentic_codegen import _require_native_handoff_files

    monkeypatch.setenv("KSEARCH_STRICT_HANDOFF_VALIDATION", "1")
    (tmp_path / "CODE_MAP.md").write_text("# CODE_MAP\n" + "file entry contract\n" * 8, encoding="utf-8")
    (tmp_path / "ASCENDC_DESIGN.md").write_text("# design\n" + "detail\n" * 20, encoding="utf-8")
    (tmp_path / "IMPLEMENTATION_EXECUTION_PLAN.md").write_text("tiny\n", encoding="utf-8")
    (tmp_path / "IMPLEMENTATION_HANDOFF.md").write_text("# handoff\n" + "changed\n" * 8, encoding="utf-8")
    (tmp_path / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="IMPLEMENTATION_EXECUTION_PLAN.md is too short"):
        _require_native_handoff_files(
            tmp_path,
            required_files={
                "CODE_MAP.md",
                "ASCENDC_DESIGN.md",
                "IMPLEMENTATION_EXECUTION_PLAN.md",
                "IMPLEMENTATION_HANDOFF.md",
                "REVIEW_NOTES.md",
            },
        )


def _permission_behavior(result):
    if isinstance(result, dict):
        return result.get("behavior") or result.get("decision")
    name = type(result).__name__.lower()
    if "deny" in name:
        return "deny"
    if "allow" in name:
        return "allow"
    return getattr(result, "behavior", None)
