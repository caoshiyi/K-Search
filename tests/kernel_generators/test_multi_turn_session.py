"""Tests for multi-turn session and fix prompt generation."""

import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    _build_fix_prompt,
    _truncate,
)
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeProjectEditorSession,
    ClaudeProjectEditResult,
)
from k_search.tasks.task_base import EvalResult


class TestBuildFixPrompt:
    """Test _build_fix_prompt() for different EvalResult status types."""

    def test_compile_failed(self):
        er = EvalResult(
            status="compile_failed",
            log_excerpt="[build exit_code=1]\n[build stderr]\nerror: undefined reference to foo",
        )
        prompt = _build_fix_prompt(er, fix_round=1)
        assert "failed to compile" in prompt
        assert "fix attempt 1" in prompt
        assert "Build error output" in prompt
        assert "undefined reference to foo" in prompt
        assert "Fix the compilation error" in prompt

    def test_correctness_failed(self):
        er = EvalResult(
            status="failed",
            log_excerpt="[correctness exit_code=1]\n[correctness stderr]\nAssertionError: expected 0.5 got 0.3",
        )
        prompt = _build_fix_prompt(er, fix_round=2)
        assert "failed correctness/precision testing" in prompt
        assert "fix attempt 2" in prompt
        assert "Test error output" in prompt
        assert "AssertionError" in prompt
        assert "Fix the correctness failure" in prompt

    def test_benchmark_failed(self):
        er = EvalResult(
            status="benchmark_failed",
            log_excerpt="[benchmark exit_code=1]\n[benchmark stderr]\nbenchmark crashed",
        )
        prompt = _build_fix_prompt(er, fix_round=1)
        assert "benchmark failed" in prompt
        assert "Fix the benchmark failure" in prompt

    def test_timeout(self):
        er = EvalResult(
            status="timeout",
            log_excerpt="[timeout] command exceeded 900s",
        )
        prompt = _build_fix_prompt(er, fix_round=1)
        assert "timed out" in prompt
        assert "infinite loop" in prompt

    def test_unknown_status(self):
        er = EvalResult(
            status="unknown_error",
            log_excerpt="something went wrong",
        )
        prompt = _build_fix_prompt(er, fix_round=1)
        assert "unknown_error" in prompt
        assert "something went wrong" in prompt

    def test_log_truncation(self):
        er = EvalResult(
            status="compile_failed",
            log_excerpt="x" * 10_000,
        )
        prompt = _build_fix_prompt(er, fix_round=1, max_chars=500)
        # The log should be truncated via _truncate (prompt template + truncated log)
        assert len(prompt) < 800

    def test_fix_round_numbering(self):
        er = EvalResult(status="compile_failed", log_excerpt="build error")
        prompt_1 = _build_fix_prompt(er, fix_round=1)
        prompt_3 = _build_fix_prompt(er, fix_round=3)
        assert "fix attempt 1" in prompt_1
        assert "fix attempt 3" in prompt_3


class TestClaudeProjectEditorSession:
    """Test ClaudeProjectEditorSession dataclass and lifecycle."""

    def test_session_creation(self):
        session = ClaudeProjectEditorSession(
            client=None,
            project_dir="/tmp/project",
            model_name="claude-test",
        )
        assert not session._closed
        assert session.chunks == []
        assert session.model_name == "claude-test"

    def test_session_closed_flag(self):
        session = ClaudeProjectEditorSession(
            client=None,
            project_dir="/tmp/project",
            model_name="claude-test",
        )
        session._closed = True
        assert session._closed


class TestSessionClientMock:
    """Test session management methods with mock ClaudeAgentProjectEditorClient."""

    def test_send_prompt_on_closed_session_raises(self):
        from k_search.kernel_generators.claude_agent_project_editor import (
            ClaudeAgentProjectEditorClient,
        )

        client = ClaudeAgentProjectEditorClient(model_name="claude-test")
        session = ClaudeProjectEditorSession(
            client=None,
            project_dir="/tmp/project",
            model_name="claude-test",
            _closed=True,
        )
        with pytest.raises(RuntimeError, match="closed session"):
            client.send_prompt(session, prompt="test")


class TestTruncate:
    """Test _truncate() helper used by _build_fix_prompt."""

    def test_short_text_not_truncated(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("x" * 200, 50)
        assert len(result) <= 50
        assert "[truncated" in result

    def test_empty_text(self):
        assert _truncate("", 100) == ""