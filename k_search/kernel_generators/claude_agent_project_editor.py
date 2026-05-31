"""Claude Agent SDK project-editor client for agentic codegen.

This client uses ClaudeSDKClient as an async context manager that can
Read/Grep/Glob/Edit/Write files inside a worktree, in contrast to
ClaudeAgentLLMClient which uses query() as a prompt-to-text backend.

Also supports multi-turn sessions via open_session/send_prompt/close_session,
keeping a single ClaudeSDKClient connection alive across multiple prompts.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import os

from k_search.kernel_generators.llm_clients import (
    ClaudeAgentLLMClient,
    LLMProviderFatalError,
    _as_provider_exception,
    _default_claude_agent_max_turns,
    _default_claude_agent_thinking_enabled,
    _default_claude_agent_timeout_seconds,
    _log_llm_interaction,
)
from k_search.telemetry.claude_sdk_adapter import event_from_claude_message
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.recorder import TelemetryRecorder, noop_recorder

DEFAULT_PROJECT_EDITOR_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]


@dataclass
class ClaudeProjectEditorSession:
    """Active ClaudeSDKClient connection for multi-turn project editing."""
    client: Any  # claude_agent_sdk.ClaudeSDKClient
    project_dir: Path
    model_name: str
    chunks: list[str] = field(default_factory=list)
    _closed: bool = False


@dataclass
class ClaudeProjectEditResult:
    text: str
    transcript: str
    prompt: str
    prompt_chars: int
    prompt_lines: int
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None


@dataclass
class ClaudeAgentProjectEditorClient:
    model_name: str
    max_turns: Optional[int] = field(default_factory=_default_claude_agent_max_turns)
    allowed_tools: list[str] = field(default_factory=lambda: list(DEFAULT_PROJECT_EDITOR_TOOLS))
    disallowed_tools: list[str] = field(default_factory=lambda: ["Bash"])
    thinking_enabled: bool = field(default_factory=_default_claude_agent_thinking_enabled)
    timeout_seconds: float = field(default_factory=_default_claude_agent_timeout_seconds)

    def edit_project(self, *, project_dir: str | Path, prompt: str, telemetry_recorder: TelemetryRecorder | None = None) -> ClaudeProjectEditResult:
        try:
            import claude_agent_sdk  # type: ignore
        except ImportError as exc:
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name,
                prompt=prompt, response="", error=str(exc),
            )
            raise RuntimeError(
                "Claude Agent SDK provider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

        recorder = telemetry_recorder or noop_recorder()

        async def _run_edit() -> ClaudeProjectEditResult:
            project_root = Path(project_dir).expanduser().resolve()
            prompt_text = str(prompt or "")
            options_kwargs: dict[str, Any] = {
                "cwd": str(project_root),
                "allowed_tools": list(self.allowed_tools),
                "disallowed_tools": list(self.disallowed_tools),
                "permission_mode": os.getenv("CLAUDE_AGENT_PERMISSION_MODE", "acceptEdits"),
                "model": self.model_name,
            }
            if self.max_turns is not None:
                options_kwargs["max_turns"] = self.max_turns
            if not self.thinking_enabled:
                options_kwargs["thinking"] = {"type": "disabled"}
            options = claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)
            chunks: list[str] = []
            final_text = ""
            try:
                async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
                    recorder.emit(
                        TelemetryEvent(
                            event_type="llm_start",
                            provider="claude-agent",
                            model_name=self.model_name,
                        )
                    )
                    await client.query(prompt_text)
                    result_event: TelemetryEvent | None = None
                    async for message in client.receive_response():
                        for event in event_from_claude_message(message):
                            event.provider = event.provider or "claude-agent"
                            event.model_name = event.model_name or self.model_name
                            recorder.emit(event)
                            if event.event_type == "llm_result":
                                result_event = event
                        is_result_message = hasattr(message, "result")
                        if is_result_message:
                            ClaudeAgentLLMClient._ensure_successful_result_message(message)
                        text = ClaudeAgentLLMClient._extract_message_text(message)
                        if not text:
                            continue
                        chunks.append(text)
                        if is_result_message:
                            final_text = text
            except LLMProviderFatalError:
                raise
            except Exception as exc:
                recorder.emit(
                    TelemetryEvent(
                        event_type="llm_error",
                        provider="claude-agent",
                        model_name=self.model_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                provider_exc = _as_provider_exception(
                    provider="claude-agent", model_name=self.model_name, exc=exc,
                )
                if isinstance(provider_exc, LLMProviderFatalError):
                    raise provider_exc from exc
                raise RuntimeError(f"Claude Agent SDK project editor failed: {exc}") from exc

            recorder.emit(
                TelemetryEvent(
                    event_type="llm_end",
                    provider="claude-agent",
                    model_name=self.model_name,
                )
            )

            transcript = "\n".join(chunks).strip()
            result_text = (final_text or transcript).strip()
            if not result_text:
                raise RuntimeError("Claude Agent SDK project editor returned empty text")
            return ClaudeProjectEditResult(
                text=result_text,
                transcript=transcript,
                prompt=prompt_text,
                prompt_chars=len(prompt_text),
                prompt_lines=(prompt_text.count("\n") + 1 if prompt_text else 0),
                trace_path=recorder.artifacts.trace_path,
                timeline_path=recorder.artifacts.timeline_path,
                cost_path=recorder.artifacts.cost_path,
                session_id=result_event.session_id if result_event else None,
                total_cost_usd=result_event.total_cost_usd if result_event else None,
                usage=result_event.usage if result_event else None,
                model_usage=result_event.model_usage if result_event else None,
                num_turns=result_event.num_turns if result_event else None,
                duration_ms=result_event.duration_ms if result_event else None,
            )

        try:
            result = self._run_async(_run_edit)
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name,
                prompt=prompt, response=result.transcript,
            )
            return result
        except Exception as exc:
            recorder.emit(
                TelemetryEvent(
                    event_type="llm_error",
                    provider="claude-agent",
                    model_name=self.model_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            provider_exc = _as_provider_exception(
                provider="claude-agent", model_name=self.model_name, exc=exc,
            )
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name,
                prompt=prompt, response="", error=str(provider_exc),
            )
            if provider_exc is exc:
                raise
            raise provider_exc from exc

    def _run_async(self, coro_factory: Any) -> ClaudeProjectEditResult:
        timeout = float(self.timeout_seconds or 0)
        started = time.monotonic()

        async def _timed_run() -> ClaudeProjectEditResult:
            if timeout <= 0:
                return await coro_factory()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        def _timeout_error(exc: BaseException) -> TimeoutError:
            return TimeoutError(
                f"Claude Agent SDK provider timed out after {timeout:g}s. "
                "Set KSEARCH_LLM_TIMEOUT_SECONDS or API_TIMEOUT_MS to adjust this limit."
            )

        def _looks_like_timeout_cancel(exc: BaseException) -> bool:
            if timeout <= 0:
                return False
            elapsed = time.monotonic() - started
            text = str(exc)
            return elapsed >= (timeout * 0.9) and "exit code 143" in text

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(_timed_run())
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise

        def _runner() -> ClaudeProjectEditResult:
            return asyncio.run(_timed_run())

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return executor.submit(_runner).result()
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise

    # -- Multi-turn session management ------------------------------------------

    def _build_options(self, project_root: Path) -> Any:
        """Build ClaudeAgentOptions for a given project root."""
        import claude_agent_sdk  # type: ignore
        options_kwargs: dict[str, Any] = {
            "cwd": str(project_root),
            "allowed_tools": list(self.allowed_tools),
            "disallowed_tools": list(self.disallowed_tools),
            "permission_mode": os.getenv("CLAUDE_AGENT_PERMISSION_MODE", "acceptEdits"),
            "model": self.model_name,
        }
        if self.max_turns is not None:
            options_kwargs["max_turns"] = self.max_turns
        if not self.thinking_enabled:
            options_kwargs["thinking"] = {"type": "disabled"}
        return claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)

    def open_session(
        self,
        *,
        project_dir: str | Path,
        telemetry_recorder: TelemetryRecorder | None = None,
    ) -> ClaudeProjectEditorSession:
        """Open a long-lived ClaudeSDKClient session for multi-turn editing.

        The returned session can be used with send_prompt() for multiple
        query/receive cycles, then closed with close_session().
        """
        try:
            import claude_agent_sdk  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Claude Agent SDK provider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

        recorder = telemetry_recorder or noop_recorder()
        project_root = Path(project_dir).expanduser().resolve()
        options = self._build_options(project_root)

        async def _connect() -> ClaudeProjectEditorSession:
            client = claude_agent_sdk.ClaudeSDKClient(options=options)
            await client.connect()
            recorder.emit(
                TelemetryEvent(
                    event_type="llm_start",
                    provider="claude-agent",
                    model_name=self.model_name,
                )
            )
            return ClaudeProjectEditorSession(
                client=client,
                project_dir=project_root,
                model_name=self.model_name,
            )

        return self._run_async_session(_connect)

    def send_prompt(
        self,
        session: ClaudeProjectEditorSession,
        *,
        prompt: str,
        telemetry_recorder: TelemetryRecorder | None = None,
    ) -> ClaudeProjectEditResult:
        """Send a prompt in an existing session and collect the response.

        The session must have been opened with open_session() and not yet closed.
        """
        if session._closed:
            raise RuntimeError("Cannot send prompt to a closed session")

        recorder = telemetry_recorder or noop_recorder()
        prompt_text = str(prompt or "")

        async def _query_and_collect() -> ClaudeProjectEditResult:
            chunks: list[str] = list(session.chunks)
            final_text = ""
            result_event: TelemetryEvent | None = None

            await session.client.query(prompt_text)
            async for message in session.client.receive_response():
                for event in event_from_claude_message(message):
                    event.provider = event.provider or "claude-agent"
                    event.model_name = event.model_name or session.model_name
                    recorder.emit(event)
                    if event.event_type == "llm_result":
                        result_event = event
                is_result_message = hasattr(message, "result")
                if is_result_message:
                    ClaudeAgentLLMClient._ensure_successful_result_message(message)
                text = ClaudeAgentLLMClient._extract_message_text(message)
                if not text:
                    continue
                chunks.append(text)
                if is_result_message:
                    final_text = text

            session.chunks = chunks

            transcript = "\n".join(chunks).strip()
            result_text = (final_text or transcript).strip()
            if not result_text:
                raise RuntimeError("Claude Agent SDK project editor returned empty text in session")
            return ClaudeProjectEditResult(
                text=result_text,
                transcript=transcript,
                prompt=prompt_text,
                prompt_chars=len(prompt_text),
                prompt_lines=(prompt_text.count("\n") + 1 if prompt_text else 0),
                trace_path=recorder.artifacts.trace_path,
                timeline_path=recorder.artifacts.timeline_path,
                cost_path=recorder.artifacts.cost_path,
                session_id=result_event.session_id if result_event else None,
                total_cost_usd=result_event.total_cost_usd if result_event else None,
                usage=result_event.usage if result_event else None,
                model_usage=result_event.model_usage if result_event else None,
                num_turns=result_event.num_turns if result_event else None,
                duration_ms=result_event.duration_ms if result_event else None,
            )

        try:
            result = self._run_async_session(_query_and_collect)
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name,
                prompt=prompt, response=result.transcript,
            )
            return result
        except Exception as exc:
            recorder.emit(
                TelemetryEvent(
                    event_type="llm_error",
                    provider="claude-agent",
                    model_name=self.model_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            provider_exc = _as_provider_exception(
                provider="claude-agent", model_name=self.model_name, exc=exc,
            )
            _log_llm_interaction(
                provider="claude-agent", model_name=self.model_name,
                prompt=prompt, response="", error=str(provider_exc),
            )
            if provider_exc is exc:
                raise
            raise provider_exc from exc

    def close_session(self, session: ClaudeProjectEditorSession) -> None:
        """Close a session, disconnecting the ClaudeSDKClient."""
        if session._closed:
            return

        async def _disconnect() -> None:
            await session.client.disconnect()
            session._closed = True

        try:
            self._run_async_session_void(_disconnect)
        except Exception:
            # Best-effort disconnect; don't propagate errors from cleanup.
            session._closed = True

    def _run_async_session(self, coro_factory: Any) -> Any:
        """Run an async coroutine that returns a value, compatible with existing or no event loop."""
        timeout = float(self.timeout_seconds or 0)

        async def _timed_run() -> Any:
            if timeout <= 0:
                return await coro_factory()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(_timed_run())
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Claude Agent SDK session timed out after {timeout:g}s."
                ) from exc

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return executor.submit(lambda: asyncio.run(_timed_run())).result()
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Claude Agent SDK session timed out after {timeout:g}s."
                ) from exc

    def _run_async_session_void(self, coro_factory: Any) -> None:
        """Run an async coroutine that returns None (e.g. disconnect)."""
        self._run_async_session(coro_factory)
