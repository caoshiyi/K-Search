from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
    extract_agent_id_from_tool_result,
)


def test_extract_agent_id_from_tool_result_text_content():
    block = {"content": [{"text": "subagent finished\nagentId: agent-abc-123\n"}]}

    assert extract_agent_id_from_tool_result(block) == "agent-abc-123"


def test_claude_project_edit_result_exposes_checkpoint_v3_metadata_fields():
    result = ClaudeProjectEditResult(
        text="ok",
        transcript="ok",
        prompt="prompt",
        prompt_chars=6,
        prompt_lines=1,
        file_checkpoint_uuid="user-message-1",
        user_message_uuids=["user-message-1"],
        subagent_agent_ids=["agent-1"],
        subagent_invocations=[{"agent": "designer", "agent_id": "agent-1"}],
    )

    assert result.file_checkpoint_uuid == "user-message-1"
    assert result.user_message_uuids == ["user-message-1"]
    assert result.subagent_agent_ids == ["agent-1"]
    assert result.subagent_invocations == [{"agent": "designer", "agent_id": "agent-1"}]


def test_claude_agent_options_include_resume_and_file_checkpointing(tmp_path):
    client = ClaudeAgentProjectEditorClient(
        model_name="claude",
        resume_session_id="session-123",
        continue_conversation=True,
        fork_session=True,
        enable_file_checkpointing=True,
    )

    kwargs = client._build_options_kwargs(tmp_path)

    assert kwargs["resume"] == "session-123"
    assert kwargs["continue_conversation"] is True
    assert kwargs["fork_session"] is True
    assert kwargs["enable_file_checkpointing"] is True
    assert kwargs["extra_args"]["replay-user-messages"] is None


def test_claude_agent_options_include_custom_session_store(tmp_path):
    session_store = object()
    client = ClaudeAgentProjectEditorClient(
        model_name="claude",
        session_store=session_store,
        session_store_flush="immediate",
    )

    kwargs = client._build_options_kwargs(tmp_path)

    assert kwargs["session_store"] is session_store
    assert kwargs["session_store_flush"] == "immediate"
