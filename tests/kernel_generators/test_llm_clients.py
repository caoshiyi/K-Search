import builtins
import asyncio
import json
import re
import sys
from types import SimpleNamespace

import pytest

from k_search.kernel_generators.llm_clients import (
    ClaudeAgentLLMClient,
    LLMAuthenticationError,
    OpenAICompatibleLLMClient,
    build_llm_client,
    _log_llm_interaction,
    llm_log_context,
    normalize_llm_provider,
)


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="responses text")


class _FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="chat text"))]
        )


class _FakeOpenAIClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = _FakeResponses()
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


class _FakeOpenAIModule:
    def __init__(self):
        self.instances = []

    def OpenAI(self, **kwargs):
        client = _FakeOpenAIClient(**kwargs)
        self.instances.append(client)
        return client


def _install_fake_claude_sdk(monkeypatch, query_func):
    fake_module = SimpleNamespace(
        ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        query=query_func,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)
    return fake_module


def test_normalize_llm_provider_accepts_aliases():
    assert normalize_llm_provider(None) == "openai"
    assert normalize_llm_provider("openai") == "openai"
    assert normalize_llm_provider("openai-compatible") == "openai"
    assert normalize_llm_provider("claude_agent") == "claude-agent"
    assert normalize_llm_provider("claude") == "claude-agent"


def test_normalize_llm_provider_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        normalize_llm_provider("local-model")


def test_build_llm_client_defaults_to_openai(monkeypatch):
    fake_openai = _FakeOpenAIModule()
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = build_llm_client(
        llm_provider=None,
        model_name="gpt-5.2",
        api_key="key",
        base_url="https://api.example/v1",
        reasoning_effort="high",
    )

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert fake_openai.instances[0].kwargs == {
        "api_key": "key",
        "base_url": "https://api.example/v1",
    }


def test_openai_reasoning_model_uses_responses_api():
    fake_openai = _FakeOpenAIModule()
    client = OpenAICompatibleLLMClient(
        model_name="gpt-5.2",
        api_key="key",
        base_url=None,
        reasoning_effort="high",
        openai_module=fake_openai,
    )

    assert client.generate("prompt") == "responses text"

    instance = fake_openai.instances[0]
    assert instance.responses.calls == [
        {
            "model": "gpt-5.2",
            "input": "prompt",
            "reasoning": {"effort": "high"},
        }
    ]
    assert instance.chat.completions.calls == []


def test_openai_non_reasoning_model_uses_chat_completions_api():
    fake_openai = _FakeOpenAIModule()
    client = OpenAICompatibleLLMClient(
        model_name="gemini-3-pro-preview",
        api_key="key",
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        openai_module=fake_openai,
    )

    assert client.generate("prompt") == "chat text"

    instance = fake_openai.instances[0]
    assert instance.responses.calls == []
    assert instance.chat.completions.calls == [
        {
            "model": "gemini-3-pro-preview",
            "messages": [{"role": "user", "content": "prompt"}],
        }
    ]


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatibleLLMClient(model_name="gpt-5.2")


def test_llm_interaction_logger_writes_hierarchical_human_readable_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("KSEARCH_LLM_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_LLM_LOG_JSON", "1")

    with llm_log_context(
        operator="multi/query attention",
        flow="world model",
        round_index=3,
        stage="debug codegen",
        action_node_id="n/3",
    ):
        _log_llm_interaction(
            provider="claude-agent",
            model_name="claude/sonnet:4.6",
            prompt="line one\nline two\n```python\nprint(1)\n```",
            response="answer one\nanswer two",
            error="timeout",
        )

    json_logs = list(tmp_path.rglob("*.json"))
    markdown_logs = list(tmp_path.rglob("*.md"))
    assert len(json_logs) == 1
    assert len(markdown_logs) == 1
    rel_parts = json_logs[0].relative_to(tmp_path).parts
    # Operator and run are encoded by the run logs dir; under an explicit
    # KSEARCH_LLM_LOG_DIR override the per-call sub-structure is flow/round/stage.
    assert rel_parts[0:3] == ("world_model", "round_0003", "debug_codegen")

    raw_json = json_logs[0].read_text(encoding="utf-8")
    payload = json.loads(raw_json)
    readable = markdown_logs[0].read_text(encoding="utf-8")
    assert "line one\\nline two" in raw_json
    assert "line one\nline two" in readable
    assert "line one\\nline two" not in readable
    assert "````text\nline one" in readable
    assert "## Response\n\n```text\nanswer one\nanswer two\n```" in readable
    assert "- error: timeout" in readable
    assert payload["log_context"]["operator"] == "multi/query attention"
    assert payload["log_context"]["round_index"] == 3
    assert "- operator: multi/query attention" in readable
    assert "- action_node_id: n/3" in readable


def test_llm_interaction_logger_defaults_under_task_run_logs(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_LLM_LOG_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_ARTIFACTS_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-llm")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-llm")

    with llm_log_context(
        operator="multi/query attention",
        flow="world model",
        round_index=3,
        stage="debug codegen",
    ):
        _log_llm_interaction(
            provider="claude-agent",
            model_name="claude/sonnet:4.6",
            prompt="prompt",
            response="response",
        )

    markdown_logs = list((tmp_path / "out").rglob("*.md"))
    assert len(markdown_logs) == 1
    rel_parts = markdown_logs[0].relative_to(tmp_path / "out").parts
    assert rel_parts[0:9] == (
        "multi_query_attention",
        "task-llm",
        "runs",
        "run-llm",
        "logs",
        "llm",
        "world_model",
        "round_0003",
        "debug_codegen",
    )


def test_llm_interaction_logger_uses_unknown_hierarchy_without_context(monkeypatch, tmp_path):
    monkeypatch.setenv("KSEARCH_LLM_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_LLM_LOG_JSON", "1")

    _log_llm_interaction(
        provider="openai",
        model_name="gpt-5.2",
        prompt="prompt",
        response="response",
    )

    json_logs = list(tmp_path.rglob("*.json"))
    assert len(json_logs) == 1
    rel_parts = json_logs[0].relative_to(tmp_path).parts
    assert rel_parts[0:3] == ("direct", "global", "llm_call")


def test_llm_interaction_logger_writes_markdown_only_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("KSEARCH_LLM_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("KSEARCH_LLM_LOG_JSON", raising=False)

    _log_llm_interaction(
        provider="openai",
        model_name="gpt-5.2",
        prompt="prompt",
        response="response",
    )

    # json+md double-write is merged: markdown only unless KSEARCH_LLM_LOG_JSON is set.
    assert list(tmp_path.rglob("*.md"))
    assert list(tmp_path.rglob("*.json")) == []


def test_codegen_logging_context_keeps_outer_round_and_stage(monkeypatch, tmp_path):
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    monkeypatch.setenv("KSEARCH_LLM_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_LLM_LOG_JSON", "1")

    class LoggingFakeLLMClient:
        def generate(self, prompt):
            _log_llm_interaction(
                provider="fake",
                model_name="fake-model",
                prompt=prompt,
                response="print('hello')",
            )
            return "print('hello')"

    class FakeTask:
        name = "vec/add"

    generator = KernelGenerator(
        model_name="fake-model",
        language="python",
        target_gpu="H100",
        api_key=None,
        llm_client=LoggingFakeLLMClient(),
    )

    with llm_log_context(
        operator=FakeTask.name,
        flow="world_model",
        round_index=7,
        stage="debug_codegen",
    ):
        result = generator._generate_code_from_prompt("make code", task=FakeTask())

    assert result["cleaned"] == "print('hello')"
    json_logs = list(tmp_path.rglob("*.json"))
    assert len(json_logs) == 1
    rel_parts = json_logs[0].relative_to(tmp_path).parts
    assert rel_parts[0:3] == ("world_model", "round_0007", "debug_codegen")
    payload = json.loads(json_logs[0].read_text(encoding="utf-8"))
    assert payload["log_context"]["attempt"] == 1
    assert payload["log_context"]["max_attempts"] == 1


def test_build_llm_client_can_create_claude_agent_client(monkeypatch):
    async def fake_query(prompt, options):
        yield SimpleNamespace(result="generated by claude")

    _install_fake_claude_sdk(monkeypatch, fake_query)

    client = build_llm_client(
        llm_provider="claude-agent",
        model_name="claude-sonnet-4-6",
        api_key=None,
        base_url=None,
    )

    assert isinstance(client, ClaudeAgentLLMClient)
    assert client.generate("prompt") == "generated by claude"


def test_claude_agent_client_passes_model_and_disables_tools(monkeypatch):
    seen = {}

    async def fake_query(prompt, options):
        seen["prompt"] = prompt
        seen["options"] = options
        yield SimpleNamespace(result="claude text")

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    assert client.generate("hello") == "claude text"
    assert seen["prompt"] == "hello"
    assert seen["options"].kwargs["model"] == "claude-sonnet-4-6"
    assert seen["options"].kwargs["tools"] == []
    assert seen["options"].kwargs["allowed_tools"] == []
    assert seen["options"].kwargs["permission_mode"] == "dontAsk"


def test_claude_agent_client_extracts_assistant_content_when_result_is_absent(monkeypatch):
    async def fake_query(prompt, options):
        yield SimpleNamespace(
            content=[
                SimpleNamespace(text="part one"),
                {"type": "text", "text": "part two"},
            ]
        )

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    assert client.generate("prompt") == "part one\npart two"


def test_claude_agent_client_rejects_error_result_message(monkeypatch):
    async def fake_query(prompt, options):
        yield SimpleNamespace(
            is_error=True,
            subtype="error_during_execution",
            result="failed but textual",
        )

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    with pytest.raises(
        RuntimeError,
        match="Claude Agent SDK.*error_during_execution.*failed but textual",
    ):
        client.generate("prompt")


def test_claude_agent_client_raises_authentication_error_for_403_result(monkeypatch):
    async def fake_query(prompt, options):
        yield SimpleNamespace(
            is_error=True,
            subtype="success",
            result=(
                "Failed to authenticate. API Error: 403 Access terminated. "
                "Contact support."
            ),
        )

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    with pytest.raises(LLMAuthenticationError, match="403 Access terminated"):
        client.generate("prompt")


def test_claude_agent_client_rejects_empty_text(monkeypatch):
    async def fake_query(prompt, options):
        yield SimpleNamespace(result="   ")

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    with pytest.raises(RuntimeError, match="Claude Agent SDK returned empty text"):
        client.generate("prompt")


def test_claude_agent_timeout_defaults_from_api_timeout_ms(monkeypatch):
    monkeypatch.delenv("KSEARCH_LLM_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CLAUDE_AGENT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("API_TIMEOUT_MS", "123000")

    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    assert client.timeout_seconds == 123.0


def test_claude_agent_timeout_reports_clear_error_when_sdk_child_is_cancelled(monkeypatch):
    async def fake_query(prompt, options):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise RuntimeError("Command failed with exit code 143") from None
        yield SimpleNamespace(result="unreachable")

    _install_fake_claude_sdk(monkeypatch, fake_query)
    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6", timeout_seconds=0.01)

    with pytest.raises(TimeoutError, match="timed out after 0.01s"):
        client.generate("prompt")


def test_claude_agent_client_reports_missing_sdk(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise ImportError("missing claude sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    with pytest.raises(RuntimeError, match="pip install claude-agent-sdk"):
        client.generate("prompt")


def test_kernel_generator_uses_injected_llm_client_for_prompt_generation():
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    class FakeLLMClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return "```python\nprint('hello')\n```"

    fake_client = FakeLLMClient()
    generator = KernelGenerator(
        model_name="fake-model",
        language="python",
        target_gpu="H100",
        api_key=None,
        llm_client=fake_client,
    )

    result = generator._generate_code_from_prompt("make code")

    assert fake_client.prompts == ["make code"]
    assert result["raw"] == "```python\nprint('hello')\n```"
    assert result["cleaned"] == "print('hello')"


def test_kernel_generator_strips_plain_text_from_injected_llm_client():
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    class FakeLLMClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return "  print('hello')  "

    fake_client = FakeLLMClient()
    generator = KernelGenerator(
        model_name="fake-model",
        language="python",
        target_gpu="H100",
        api_key=None,
        llm_client=fake_client,
    )

    result = generator._generate_code_from_prompt("make code")

    assert fake_client.prompts == ["make code"]
    assert result["raw"] == "print('hello')"
    assert result["cleaned"] == "print('hello')"


def test_world_model_generator_routes_llm_calls_through_injected_client():
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )

    class FakeLLMClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return "wm text"

    fake_client = FakeLLMClient()
    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake-model",
        language="python",
        target_gpu="H100",
        api_key=None,
        llm_client=fake_client,
    )

    assert generator._wm._llm_call("wm prompt") == "wm text"
    assert fake_client.prompts == ["wm prompt"]


def test_claude_agent_llm_client_still_uses_query_not_sdk_client(monkeypatch):
    seen = {"query": 0, "client": 0}

    class FakeClient:
        def __init__(self, options):
            seen["client"] += 1

    async def fake_query(prompt, options):
        seen["query"] += 1
        yield SimpleNamespace(result="query text")

    fake_module = SimpleNamespace(
        ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        ClaudeSDKClient=FakeClient,
        query=fake_query,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)

    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    assert client.generate("prompt") == "query text"
    assert seen == {"query": 1, "client": 0}


def test_world_model_ascendc_root_action_prompt_uses_task_baseline():
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )

    class StopAfterPrompt(Exception):
        pass

    class FakeTask:
        name = "mqa"

        def get_definition_text(self, language):
            return "spec"

        def get_baseline_targets_text(self):
            return ""

        def get_code_format_text(self, language, target_gpu):
            return "<ascendc_patch>"

        def get_baseline_code_for_codegen(self, language):
            return "<ascendc_project>BASELINE_CODE</ascendc_project>"

        def code_for_world_model_from_raw(self, raw, language):
            return str(raw or "")

    class FakeWorldModel:
        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "optimize",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

    captured = {}

    class FakeLLMClient:
        def generate(self, prompt):
            return ""

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake-model",
        language="ascendc",
        target_gpu="Ascend910B3",
        api_key=None,
        llm_client=FakeLLMClient(),
    )
    generator._wm = FakeWorldModel()
    generator._solution_db = None

    def capture_prompt(prompt, task):
        captured["prompt"] = prompt
        raise StopAfterPrompt

    generator._generate_code_from_prompt = capture_prompt

    with pytest.raises(StopAfterPrompt):
        generator._generate_world_model_cycles_v2(
            task=FakeTask(),
            max_opt_rounds=1,
            wm_stagnation_window=1,
            max_dai=1,
        )

    assert "BASELINE_CODE" in captured["prompt"]
    assert "(no base code; start from spec)" not in captured["prompt"]


def test_world_model_codegen_failure_marks_action_too_hard_and_returns_last_solution():
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    from k_search.tasks.task_base import EvalResult

    class FakeTask:
        name = "mqa"

        def get_definition_text(self, language):
            return "spec"

        def get_baseline_targets_text(self):
            return ""

        def get_code_format_text(self, language, target_gpu):
            return "<ascendc_patch>"

        def get_baseline_code_for_codegen(self, language):
            return "<ascendc_project>BASELINE_CODE</ascendc_project>"

        def code_for_world_model_from_raw(self, raw, language):
            return str(raw or "")

        def get_last_round_trace_logs_for_prompt(self):
            return "build failed"

        def run_benchmark(self, solution, dump_traces=False, round_num=0):
            return EvalResult(status="failed", log_excerpt="build failed")

    class FakeWorldModel:
        def __init__(self):
            self.too_hard_calls = []

        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "optimize",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def note_action_too_hard(self, **kwargs):
            self.too_hard_calls.append(kwargs)

    class FakeLLMClient:
        def generate(self, prompt):
            return ""

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake-model",
        language="ascendc",
        target_gpu="Ascend910B3",
        api_key=None,
        llm_client=FakeLLMClient(),
    )
    fake_wm = FakeWorldModel()
    generator._wm = fake_wm
    generator._solution_db = None

    calls = {"count": 0}

    def generate_or_timeout(prompt, task):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"raw": "candidate code", "cleaned": "candidate code"}
        raise TimeoutError("Claude Agent SDK provider timed out after 1200s")

    generator._generate_code_from_prompt = generate_or_timeout

    solution = generator._generate_world_model_cycles_v2(
        task=FakeTask(),
        max_opt_rounds=2,
        wm_stagnation_window=5,
        max_dai=5,
    )

    assert solution.name == "fake-model_mqa_ascendc_optimized_r1"
    assert len(fake_wm.too_hard_calls) == 1
    assert fake_wm.too_hard_calls[0]["eval_result"].status == "codegen_failed"


def test_world_model_codegen_fatal_provider_error_propagates_without_marking_action():
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )

    class FakeTask:
        name = "mqa"

        def get_definition_text(self, language):
            return "spec"

        def get_baseline_targets_text(self):
            return ""

        def get_code_format_text(self, language, target_gpu):
            return "<ascendc_patch>"

        def get_baseline_code_for_codegen(self, language):
            return "<ascendc_project>BASELINE_CODE</ascendc_project>"

        def code_for_world_model_from_raw(self, raw, language):
            return str(raw or "")

    class FakeWorldModel:
        def __init__(self):
            self.too_hard_calls = []

        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "optimize",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def note_action_too_hard(self, **kwargs):
            self.too_hard_calls.append(kwargs)

    class FakeLLMClient:
        def generate(self, prompt):
            return ""

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake-model",
        language="ascendc",
        target_gpu="Ascend910B3",
        api_key=None,
        llm_client=FakeLLMClient(),
    )
    fake_wm = FakeWorldModel()
    generator._wm = fake_wm
    generator._solution_db = None

    def fail_with_auth(prompt, task):
        raise LLMAuthenticationError("403 Access terminated")

    generator._generate_code_from_prompt = fail_with_auth

    with pytest.raises(LLMAuthenticationError, match="403 Access terminated"):
        generator._generate_world_model_cycles_v2(
            task=FakeTask(),
            max_opt_rounds=1,
            wm_stagnation_window=1,
            max_dai=1,
        )

    assert fake_wm.too_hard_calls == []


def test_world_model_debug_prompt_uses_applied_code_after_patch_response():
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    from k_search.tasks.task_base import EvalResult

    class StopAfterSecondPrompt(Exception):
        pass

    class FakeTask:
        name = "mqa"

        def get_definition_text(self, language, include_sources=True, include_format=True):
            return "spec"

        def get_baseline_targets_text(self):
            return ""

        def get_code_format_text(self, language, target_gpu):
            return "<ascendc_patch>"

        def get_baseline_code_for_codegen(self, language):
            return "<ascendc_project>BASELINE_CODE</ascendc_project>"

        def code_for_world_model_from_raw(self, raw, language):
            if raw == "RAW_PATCH_ONLY":
                return "<ascendc_project>APPLIED_FULL_CODE</ascendc_project>"
            return str(raw or "")

        def get_last_round_trace_logs_for_prompt(self):
            return "build failed"

        def run_benchmark(self, solution, dump_traces=False, round_num=0):
            return EvalResult(status="failed", log_excerpt="build failed")

    class FakeWorldModel:
        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "optimize",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

    class FakeLLMClient:
        def generate(self, prompt):
            return ""

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake-model",
        language="ascendc",
        target_gpu="Ascend910B3",
        api_key=None,
        llm_client=FakeLLMClient(),
    )
    generator._wm = FakeWorldModel()
    generator._solution_db = None

    prompts = []

    def capture_second_prompt(prompt, task):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"raw": "RAW_PATCH_ONLY", "cleaned": "RAW_PATCH_ONLY"}
        raise StopAfterSecondPrompt

    generator._generate_code_from_prompt = capture_second_prompt

    with pytest.raises(StopAfterSecondPrompt):
        generator._generate_world_model_cycles_v2(
            task=FakeTask(),
            max_opt_rounds=2,
            wm_stagnation_window=5,
            max_dai=5,
        )

    assert len(prompts) == 2
    assert "APPLIED_FULL_CODE" in prompts[1]
    assert "RAW_PATCH_ONLY" not in prompts[1]


def test_baseline_ascendc_agentic_failure_can_fallback_to_legacy(monkeypatch, tmp_path):
    from k_search.kernel_generators.kernel_generator import KernelGenerator
    from k_search.tasks.ascendc_task import AscendCTask, format_ascendc_project_files

    class FailingRunner:
        def run_one_shot_closed(self, *, task, request, base_solution, max_fix_rounds=0):
            del max_fix_rounds
            raise RuntimeError("agentic unavailable")

        def run(self, *, task, request, base_solution):
            return self.run_one_shot_closed(task=task, request=request, base_solution=base_solution)

    class LegacyClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return format_ascendc_project_files({"kernel.cpp": "void run() {}\n"})

    (tmp_path / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    legacy_client = LegacyClient()
    generator = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=legacy_client,
    )
    generator._ascendc_agentic_runner = FailingRunner()
    monkeypatch.setenv("KSEARCH_ASCENDC_AGENTIC_FALLBACK", "legacy")

    solution = generator.generate(task=task, max_opt_rounds=1)

    assert {src.path for src in solution.sources} == {"kernel.cpp"}
    assert legacy_client.prompts
    assert "<ascendc_project>" in legacy_client.prompts[0]


def test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction(tmp_path):
    from types import SimpleNamespace
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenResult
    from k_search.tasks.ascendc_task import AscendCTask
    from k_search.tasks.task_base import EvalResult

    (tmp_path / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    class FakeAgenticRunner:
        def __init__(self):
            self.requests = []

        def run(self, *, task, request, base_solution):
            self.requests.append(request)
            attempt_no = len(self.requests)
            (tmp_path / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            solution = task.make_solution_from_project_dir(
                project_dir=tmp_path,
                changed_paths=["kernel/foo.h"],
                raw_agent_output="edited",
                round_num=request.round_num,
                model_name="fake",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            return AscendCAgenticCodegenResult(
                solution=solution,
                eval_result=EvalResult(
                    status="passed",
                    latency_ms=1.0,
                    metrics={
                        "score": 1.0 if attempt_no == 1 else 0.5,
                        "score_name": "inv_latency",
                        "workdir": str(tmp_path),
                    },
                ),
                raw=task.code_for_world_model_from_raw(raw={src.path: src.content for src in solution.sources}, language="ascendc"),
                cleaned={src.path: src.content for src in solution.sources},
                transcript="edited",
                prompt="compact prompt",
                prompt_chars=14,
                changed_paths=["kernel/foo.h"],
                diff_text="diff",
                project_path=str(tmp_path),
                candidate_patch=SimpleNamespace(candidate_id=f"candidate-{attempt_no}"),
                artifact_paths={"manifest_path": str(tmp_path / f"candidate-{attempt_no}.json")},
            )

        def run_multi_turn(self, *, task, request, base_solution, max_fix_rounds=0):
            return self.run(task=task, request=request, base_solution=base_solution)

        def open_cycle(self, *, task, request, base_solution):
            runner = self

            class FakeCycle:
                def __init__(self):
                    self.request = request

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def run_initial(self):
                    return runner.run(task=task, request=self.request, base_solution=base_solution)

                def continue_fix(self, fix_prompt):
                    del fix_prompt
                    return runner.run(task=task, request=self.request, base_solution=None)

            return FakeCycle()

        class editor_client:
            @staticmethod
            def close_session(session):
                pass

    class FakeWorldModel:
        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "capitalize beta",
                    "description": "Change beta to BETA.",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def attach_solution_to_active_leaf(self, **kwargs):
            return None

        def refine(self, **kwargs):
            return None

        def note_action_too_hard(self, **kwargs):
            return None

    task = AscendCTask(
        task_path=tmp_path,
        definition_name="x",
        build_cmd="",
        test_cmd="",
        bench_cmd="python -c \"raise SystemExit('should not re-evaluate Solution.sources')\"",
        timeout_seconds=30,
    )
    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
    )
    fake_runner = FakeAgenticRunner()
    generator._wm = FakeWorldModel()
    generator._solution_db = None
    generator._ascendc_agentic_runner = fake_runner

    solution = generator._generate_world_model_cycles_v2(
        task=task,
        max_opt_rounds=2,
        wm_stagnation_window=2,
        max_dai=1,
        run_id="wm-effective-run",
    )

    assert "BETA" in next(src.content for src in solution.sources if src.path == "kernel/foo.h")
    assert len(fake_runner.requests) == 2
    assert fake_runner.requests[0].action_text
    assert "<ascendc_project>" not in fake_runner.requests[0].action_text
    assert fake_runner.requests[0].run_id == "wm-effective-run"
    assert fake_runner.requests[0].task_name == "x"
    assert fake_runner.requests[0].action_node_id == "n1"
    assert fake_runner.requests[0].parent_candidate_id is None
    assert fake_runner.requests[1].run_id == "wm-effective-run"
    assert fake_runner.requests[1].task_name == "x"
    assert fake_runner.requests[1].action_node_id == "n1"
    assert fake_runner.requests[1].parent_candidate_id == "candidate-1"


def test_world_model_narrative_logger_uses_effective_run_id(tmp_path, monkeypatch):
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    import k_search.kernel_generators.kernel_generator_world_model as wm_module

    class StopAfterInit(Exception):
        pass

    class FakeTask:
        name = "lineage_task"

        def get_definition_text(self, language):
            return "spec"

    class FakeWorldModel:
        def ensure_initialized(self, **kwargs):
            return "{}"

        def get(self, definition_name):
            return "{}"

    captured = {}

    class FakeNarrativeLogger:
        def __init__(self, root, *, meta):
            captured["root"] = root
            captured["meta"] = meta

        def run_start(self):
            captured["run_started"] = True

    monkeypatch.setenv("KSEARCH_RUN_ID", "global-run")
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-lineage")
    monkeypatch.setattr(wm_module, "RunNarrativeLogger", FakeNarrativeLogger)

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    generator._wm = FakeWorldModel()
    monkeypatch.setattr(
        generator,
        "_generate_world_model_cycles_v2",
        lambda **kwargs: (_ for _ in ()).throw(StopAfterInit),
    )

    with pytest.raises(StopAfterInit):
        generator.generate(task=FakeTask(), max_opt_rounds=1, run_id="effective-run")

    assert captured["meta"]["run_id"] == "effective-run"
    assert captured["run_started"] is True
    assert captured["root"] == (
        tmp_path
        / "artifacts"
        / "lineage_task"
        / "task-lineage"
        / "runs"
        / "effective-run"
        / "logs"
    )


def test_world_model_snapshot_uses_generated_effective_run_id(tmp_path, monkeypatch):
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    import k_search.utils.paths as paths_module

    class StopAfterInit(Exception):
        pass

    class FakeTask:
        name = "lineage_task"

        def get_definition_text(self, language):
            return "spec"

    class FakeWorldModel:
        def ensure_initialized(self, **kwargs):
            return '{"decision_tree": {"nodes": []}}'

        def get(self, definition_name):
            return '{"decision_tree": {"nodes": []}}'

    run_ids = iter(["effective-run", "late-run"])

    def fake_get_run_id():
        return next(run_ids)

    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_START", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-lineage")
    monkeypatch.setattr(paths_module, "get_run_id", fake_get_run_id)

    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    generator._wm = FakeWorldModel()
    monkeypatch.setattr(
        generator,
        "_generate_world_model_cycles_v2",
        lambda **kwargs: (_ for _ in ()).throw(StopAfterInit),
    )

    with pytest.raises(StopAfterInit):
        generator.generate(task=FakeTask(), max_opt_rounds=1)

    expected = (
        tmp_path
        / "artifacts"
        / "lineage_task"
        / "task-lineage"
        / "runs"
        / "effective-run"
        / "artifacts"
        / "world_model"
        / "world_model.json"
    )
    wrong = (
        tmp_path
        / "artifacts"
        / "lineage_task"
        / "task-lineage"
        / "runs"
        / "late-run"
        / "artifacts"
        / "world_model"
        / "world_model.json"
    )
    assert expected.is_file()
    assert not wrong.exists()


def test_baseline_agentic_memory_writeback_only_for_new_best(tmp_path, monkeypatch):
    from k_search.kernel_generators.kernel_generator import KernelGenerator
    from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenResult
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    from k_search.tasks.task_base import BuildSpec, EvalResult, Solution, SourceFile, SupportedLanguages

    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    class FakeTask:
        name = "x"
        definition_name = "x"
        task_path = task_dir
        artifacts_dir = str(tmp_path / "artifacts")

        def get_definition_text(self, language):
            return "spec"

        def get_agentic_definition_text(self, *, language):
            return "agentic spec"

        def get_baseline_targets_text(self):
            return ""

        def make_solution_from_project_dir(self, **kwargs):
            raise AssertionError("fake runner supplies solutions directly")

        def code_for_world_model_from_raw(self, raw, language):
            return str(raw)

        def get_last_round_trace_logs_for_prompt(self):
            return ""

    def solution(label):
        return Solution(
            name=f"sol-{label}",
            definition="x",
            author="fake",
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=["ascend_910b"],
                entry_point="kernel/foo.h::run",
            ),
            sources=[SourceFile(path="kernel/foo.h", content=f"{label}\n")],
        )

    class FakeAgenticRunner:
        def __init__(self):
            self.calls = 0

        def run(self, *, task, request, base_solution):
            self.calls += 1
            score = 2.0 if self.calls == 1 else 1.0
            label = "best" if self.calls == 1 else "worse"
            sol = solution(label)
            raw = {src.path: src.content for src in sol.sources}
            return AscendCAgenticCodegenResult(
                solution=sol,
                eval_result=EvalResult(
                    status="passed",
                    latency_ms=1.0 / score,
                    metrics={"score": score, "score_name": "inv_latency"},
                ),
                raw=raw,
                cleaned=raw,
                transcript=label,
                prompt=label,
                prompt_chars=len(label),
                changed_paths=["kernel/foo.h"],
                diff_text=label,
                project_path=str(task_dir),
                code_map_text=f"# CODE_MAP\n{label}\n",
            )

        def run_multi_turn(self, *, task, request, base_solution, max_fix_rounds=0):
            return self.run(task=task, request=request, base_solution=base_solution)

        def run_one_shot_closed(self, *, task, request, base_solution, max_fix_rounds=0):
            return self.run(task=task, request=request, base_solution=base_solution)

    task = FakeTask()
    generator = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: ""),
    )
    generator._ascendc_agentic_runner = FakeAgenticRunner()

    result = generator.generate(task=task, max_opt_rounds=2)

    assert next(src.content for src in result.sources if src.path == "kernel/foo.h") == "best\n"
    assert MemoryStore.for_task(task).load(CODE_MAP) == "# CODE_MAP\nbest\n"
