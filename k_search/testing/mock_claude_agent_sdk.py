"""A tiny in-process mock for the subset of claude_agent_sdk K-Search uses.

Tests can install this into ``sys.modules["claude_agent_sdk"]`` so the real
``ClaudeAgentLLMClient`` runs without the external SDK package or network calls.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable


_MISSING = object()


class MockClaudeMessage:
    """Message object with only explicitly supplied SDK-like attributes."""

    def __init__(
        self,
        *,
        result: Any = _MISSING,
        text: Any = _MISSING,
        content: Any = _MISSING,
        is_error: Any = _MISSING,
        subtype: Any = _MISSING,
        **extra: Any,
    ) -> None:
        if result is not _MISSING:
            self.result = result
        if text is not _MISSING:
            self.text = text
        if content is not _MISSING:
            self.content = content
        if is_error is not _MISSING:
            self.is_error = is_error
        if subtype is not _MISSING:
            self.subtype = subtype
        for key, value in extra.items():
            setattr(self, key, value)


@dataclass
class MockClaudeAgentOptions:
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)


@dataclass
class MockAgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    disallowedTools: list[str] | None = None
    model: str | None = None
    skills: list[str] | None = None
    maxTurns: int | None = None
    background: bool | None = None
    permissionMode: str | None = None


@dataclass
class MockClaudeCall:
    prompt: str
    options: MockClaudeAgentOptions


MockResponse = (
    str
    | MockClaudeMessage
    | dict[str, Any]
    | Iterable[str | MockClaudeMessage | dict[str, Any]]
)
MockResponseFactory = Callable[..., MockResponse]


class MockClaudeAgentSDK:
    """Queue-backed fake module implementing ClaudeAgentOptions, query(), and ClaudeSDKClient."""

    def __init__(self, responses: Iterable[MockResponse | MockResponseFactory]):
        self.responses = list(responses)
        self.calls: list[MockClaudeCall] = []
        self.options: list[MockClaudeAgentOptions] = []
        self.client_calls: list[MockClaudeCall] = []
        self._response_index = 0

    def as_module(self) -> Any:
        sdk = self

        class ClaudeAgentOptions(MockClaudeAgentOptions):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                sdk.options.append(self)

        async def query(prompt: str, options: MockClaudeAgentOptions):
            call_index = len(sdk.calls)
            sdk.calls.append(MockClaudeCall(prompt=str(prompt or ""), options=options))
            response = sdk._next_response(
                prompt=str(prompt or ""),
                options=options,
                call_index=call_index,
            )
            if isinstance(response, BaseException):
                raise response
            for message in sdk._coerce_messages(response):
                yield message

        class ClaudeSDKClient:
            def __init__(self, options: MockClaudeAgentOptions) -> None:
                self.options = options
                self._prompt = ""
                self._messages: list[MockClaudeMessage] = []
                self._connected = False

            async def __aenter__(self):
                await self.connect()
                return self

            async def __aexit__(self, exc_type, exc, tb):
                await self.disconnect()
                return False

            async def connect(self, prompt: Any = None) -> None:
                self._connected = True

            async def disconnect(self) -> None:
                self._connected = False

            async def query(self, prompt: str) -> None:
                call_index = len(sdk.client_calls)
                self._prompt = str(prompt or "")
                sdk.client_calls.append(MockClaudeCall(prompt=self._prompt, options=self.options))
                response = sdk._next_response(
                    prompt=self._prompt,
                    options=self.options,
                    call_index=call_index,
                )
                if isinstance(response, BaseException):
                    raise response
                self._messages = sdk._coerce_messages(response)

            async def receive_response(self):
                for message in self._messages:
                    yield message

        return SimpleNamespace(
            AgentDefinition=MockAgentDefinition,
            ClaudeAgentOptions=ClaudeAgentOptions,
            ClaudeSDKClient=ClaudeSDKClient,
            query=query,
        )

    def _next_response(
        self,
        *,
        prompt: str,
        options: MockClaudeAgentOptions,
        call_index: int,
    ) -> Any:
        if self._response_index >= len(self.responses):
            excerpt = prompt[:120].replace("\n", "\\n")
            raise AssertionError(
                f"No queued Claude Agent SDK mock response for call {call_index + 1}: {excerpt}"
            )

        response = self.responses[self._response_index]
        self._response_index += 1
        if callable(response):
            return response(prompt=prompt, options=options, call_index=call_index)
        return response

    def _coerce_messages(self, response: Any) -> list[MockClaudeMessage]:
        if isinstance(response, MockClaudeMessage):
            return [response]
        if isinstance(response, str):
            return [MockClaudeMessage(result=response)]
        if isinstance(response, dict):
            return [MockClaudeMessage(**response)]
        if response is None:
            return []
        if isinstance(response, Iterable):
            messages: list[MockClaudeMessage] = []
            for item in response:
                messages.extend(self._coerce_messages(item))
            return messages
        return [MockClaudeMessage(result=str(response))]


def install_mock_claude_agent_sdk(
    monkeypatch: Any,
    responses: Iterable[MockResponse | MockResponseFactory],
) -> MockClaudeAgentSDK:
    """Install a queued mock as the importable ``claude_agent_sdk`` module."""
    sdk = MockClaudeAgentSDK(responses=responses)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk.as_module())
    return sdk
