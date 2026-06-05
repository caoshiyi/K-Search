"""Claude Agent SDK project-editor client for agentic codegen.

This client uses ClaudeSDKClient as an async context manager that can
Read/Grep/Glob/Edit/Write files inside a worktree, in contrast to
ClaudeAgentLLMClient which uses query() as a prompt-to-text backend.

Also supports multi-turn sessions via open_session/send_prompt/close_session,
keeping a single ClaudeSDKClient connection alive across multiple prompts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from k_search.kernel_generators.claude_assets import NATIVE_AGENT_FILES, NATIVE_AGENT_TOOL_NAMES, NATIVE_SKILLS
from k_search.kernel_generators.claude_assets.agent_definitions import (
    load_native_agent_definitions,
    require_programmatic_agents,
    should_use_programmatic_agents,
)
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

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_EDITOR_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]
DEFAULT_CLAUDE_NATIVE_TOOLS = ["Skill", *NATIVE_AGENT_TOOL_NAMES]
DEFAULT_NATIVE_AGENT_NAMES = [Path(name).stem for name in NATIVE_AGENT_FILES]
AGENT_TOOL_NAMES = {"Agent", "Task"}
PATH_KEYS_BY_TOOL = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "MultiEdit": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Glob": ("path",),
    "Grep": ("path",),
}


def _dedupe_tools(tools: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tool in tools:
        name = str(tool or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _with_claude_native_tools(tools: list[str]) -> list[str]:
    return _dedupe_tools(list(tools or []) + list(DEFAULT_CLAUDE_NATIVE_TOOLS))


def _agent_tool_allowlist_expr(agent_names: list[str]) -> str:
    names = [str(name).strip() for name in agent_names if str(name).strip()]
    if not names:
        return "Agent"
    return "Agent(" + ", ".join(names) + ")"


def _use_agent_tool_allowlist() -> bool:
    return os.getenv("KSEARCH_USE_AGENT_TOOL_ALLOWLIST", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _canonical_permission_tool_names(tools: list[str]) -> list[str]:
    out: list[str] = []
    for tool in tools:
        name = str(tool or "").strip()
        if name.startswith("Agent(") or name.startswith("Task("):
            out.extend(["Agent", "Task"])
        else:
            out.append(name)
    return _dedupe_tools(out)


def _normalize_agent_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.startswith("@agent-"):
        s = s[len("@agent-") :]
    if s.endswith(" (agent)"):
        s = s[: -len(" (agent)")]
    return s.strip() or None


def _tool_input_value(input_data: Any, *keys: str) -> str | None:
    if not isinstance(input_data, dict):
        return None
    for key in keys:
        value = _normalize_agent_name(input_data.get(key))
        if value:
            return value
    for nested_key in ("input", "arguments", "params"):
        nested = input_data.get(nested_key)
        if isinstance(nested, dict):
            for key in keys:
                value = _normalize_agent_name(nested.get(key))
                if value:
                    return value
    return None


def _agent_name_allowed(observed: str, allowed_agents: set[str]) -> bool:
    name = str(observed or "").strip()
    if not name:
        return False
    for allowed in allowed_agents:
        expected = str(allowed or "").strip()
        if name == expected or name.endswith(":" + expected) or name.endswith("/" + expected):
            return True
    return False


def _resolve_tool_path_under_root(project_root: Path, raw_path: str) -> Path:
    root = project_root.expanduser().resolve(strict=True)
    p = Path(str(raw_path or "")).expanduser()
    lexical_path = p if p.is_absolute() else root / p
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes project root: {raw_path!r} -> {resolved}")

    cur = lexical_path
    while cur != root:
        if cur.exists() and cur.is_symlink():
            raise ValueError(f"symlink path is not allowed: {cur}")
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return resolved


def _permission_allow(updated_input: dict[str, Any] | None = None) -> Any:
    try:
        from claude_agent_sdk import PermissionResultAllow  # type: ignore

        return PermissionResultAllow(updated_input=updated_input)
    except Exception:
        out = {"behavior": "allow"}
        if updated_input is not None:
            out["updated_input"] = updated_input
        return out


def _permission_deny(message: str, *, interrupt: bool = True) -> Any:
    try:
        from claude_agent_sdk import PermissionResultDeny  # type: ignore

        return PermissionResultDeny(message=message, interrupt=interrupt)
    except Exception:
        return {"behavior": "deny", "message": message, "interrupt": interrupt}


def _make_project_tool_permission_callback(
    *,
    project_root: Path,
    allowed_tools: set[str],
    allowed_agents: set[str],
    allowed_skills: set[str],
) -> Any:
    root = project_root.expanduser().resolve(strict=True)

    async def _can_use_tool(tool: str, input_data: dict[str, Any], context: Any) -> Any:
        tool_name = str(tool or "").strip()
        if tool_name not in allowed_tools:
            return _permission_deny(
                f"K-Search denied unavailable Claude SDK tool: {tool_name or '<empty>'}",
                interrupt=True,
            )
        updated_input = dict(input_data or {})
        for key in PATH_KEYS_BY_TOOL.get(tool_name, ()):
            raw = updated_input.get(key)
            if tool_name in {"Grep", "Glob"} and not raw:
                raw = "."
                updated_input[key] = "."
            if isinstance(raw, str) and raw.strip():
                try:
                    safe_path = _resolve_tool_path_under_root(root, raw)
                except Exception as exc:
                    return _permission_deny(
                        f"K-Search denied path outside candidate worktree for {tool_name}.{key}: {exc}",
                        interrupt=True,
                    )
                updated_input[key] = str(safe_path.relative_to(root))
        if tool_name in AGENT_TOOL_NAMES:
            subagent = _tool_input_value(
                updated_input,
                "subagent_type",
                "agent",
                "name",
                "subagent",
                "agent_name",
                "type",
            )
            if subagent is not None and not _agent_name_allowed(subagent, allowed_agents):
                return _permission_deny(f"K-Search denied unavailable native subagent: {subagent}", interrupt=True)
        if tool_name == "Skill":
            skill = _tool_input_value(updated_input, "skill", "skill_name", "name", "skill_id")
            if skill is not None and skill not in allowed_skills:
                return _permission_deny(f"K-Search denied unavailable native skill: {skill}", interrupt=True)
        return _permission_allow(updated_input=updated_input)

    return _can_use_tool


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
    allowed_tools: list[str] = field(default_factory=lambda: _with_claude_native_tools(list(DEFAULT_PROJECT_EDITOR_TOOLS)))
    disallowed_tools: list[str] = field(default_factory=lambda: ["Bash", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"])
    setting_sources: list[str] = field(default_factory=lambda: ["project"])
    skills: list[str] | str | None = field(default_factory=lambda: list(NATIVE_SKILLS))
    native_agents: list[str] = field(default_factory=lambda: list(DEFAULT_NATIVE_AGENT_NAMES))
    require_agent_tool_use: bool = True
    thinking_enabled: bool = field(default_factory=_default_claude_agent_thinking_enabled)
    timeout_seconds: float = field(default_factory=_default_claude_agent_timeout_seconds)

    def _build_options_kwargs(self, project_root: Path) -> dict[str, Any]:
        native_agent_names = list(self.native_agents)
        agent_tool_expr = _agent_tool_allowlist_expr(native_agent_names) if _use_agent_tool_allowlist() else "Agent"
        base_tools = [tool for tool in _with_claude_native_tools(list(self.allowed_tools)) if str(tool).strip() not in AGENT_TOOL_NAMES]
        tool_names = _dedupe_tools(base_tools + [agent_tool_expr])
        skill_names = set(NATIVE_SKILLS if self.skills is None or isinstance(self.skills, str) else self.skills)
        options_kwargs: dict[str, Any] = {
            "cwd": str(project_root),
            "tools": list(tool_names),
            "allowed_tools": list(tool_names),
            "disallowed_tools": list(self.disallowed_tools),
            "permission_mode": "dontAsk",
            "can_use_tool": _make_project_tool_permission_callback(
                project_root=project_root,
                allowed_tools=set(_canonical_permission_tool_names(tool_names)),
                allowed_agents=set(native_agent_names),
                allowed_skills=skill_names,
            ),
            "model": self.model_name,
            "setting_sources": list(self.setting_sources),
        }
        if should_use_programmatic_agents():
            try:
                programmatic_agents = load_native_agent_definitions(enabled_agent_names=native_agent_names)
                if programmatic_agents:
                    options_kwargs["agents"] = programmatic_agents
            except Exception:
                if require_programmatic_agents():
                    raise
                logger.warning(
                    "failed to build programmatic Claude subagent definitions; falling back to .claude/agents",
                    exc_info=True,
                )
        if self.skills is not None:
            options_kwargs["skills"] = self.skills if isinstance(self.skills, str) else list(self.skills)
        if self.max_turns is not None:
            options_kwargs["max_turns"] = self.max_turns
        if not self.thinking_enabled:
            options_kwargs["thinking"] = {"type": "disabled"}
        return options_kwargs

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
            options_kwargs = self._build_options_kwargs(project_root)
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

        return claude_agent_sdk.ClaudeAgentOptions(**self._build_options_kwargs(project_root))

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

        return self._run_on_session_loop(_connect)

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
            result = self._run_on_session_loop(_query_and_collect)
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
            self._run_on_session_loop(_disconnect)
        except Exception:
            # Best-effort disconnect; don't propagate errors from cleanup.
            session._closed = True

    # -- Persistent event-loop for multi-turn sessions -------------------------
    # Each call to asyncio.run() creates a fresh loop, which invalidates
    # WebSocket connections created in the previous loop.  To keep a
    # ClaudeSDKClient session alive across open_session / send_prompt /
    # close_session, all three must run on the SAME loop.  We achieve this
    # with a dedicated thread that hosts a long-lived asyncio event loop.

    _session_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _session_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def _ensure_session_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily create a persistent asyncio event loop on a background thread."""
        if self._session_loop is not None and self._session_loop.is_running():
            return self._session_loop
        loop = asyncio.new_event_loop()
        self._session_loop = loop
        self._session_thread = threading.Thread(
            target=loop.run_forever,
            name="claude-agent-session-loop",
            daemon=True,
        )
        self._session_thread.start()
        return loop

    def _stop_session_loop(self) -> None:
        """Stop the persistent event-loop thread (best-effort)."""
        if self._session_loop is not None:
            self._session_loop.call_soon_threadsafe(self._session_loop.stop)
            self._session_loop = None
            self._session_thread = None

    def _run_on_session_loop(self, coro_factory: Any) -> Any:
        """Run an async coroutine on the persistent session loop with timeout."""
        loop = self._ensure_session_loop()
        timeout = float(self.timeout_seconds or 0)

        async def _timed_run() -> Any:
            if timeout <= 0:
                return await coro_factory()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        future = asyncio.run_coroutine_threadsafe(_timed_run(), loop)
        try:
            return future.result(timeout=timeout if timeout > 0 else None)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Claude Agent SDK session timed out after {timeout:g}s."
            ) from exc

    def _run_async_session_void(self, coro_factory: Any) -> None:
        """Run an async coroutine that returns None (e.g. disconnect)."""
        self._run_on_session_loop(coro_factory)
