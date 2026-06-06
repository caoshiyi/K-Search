from types import SimpleNamespace

import pytest

from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.kernel_generators.llm_clients import LLMAuthenticationError


class _FakeLLMClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(str(prompt))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _PreviewTask:
    def __init__(self, failures=None):
        self._failures = dict(failures or {})
        self.raw_codes = []

    def preview_parse_generated_code(self, *, raw_code):
        self.raw_codes.append(raw_code)
        exc = self._failures.get(raw_code)
        if exc is not None:
            raise exc


def test_ascendc_codegen_retry_includes_parse_error_feedback():
    llm = _FakeLLMClient(["bad patch", "good patch"])
    task = _PreviewTask(
        failures={"bad patch": ValueError("patch context mismatch near cube.h")}
    )
    generator = KernelGenerator(
        model_name="test-model",
        language="ascendc",
        llm_provider="openai",
        llm_client=llm,
    )

    result = generator._generate_code_from_prompt("base prompt", task=task)

    assert result["raw"] == "good patch"
    assert len(llm.prompts) == 2
    assert "patch context mismatch near cube.h" in llm.prompts[1]


def test_ascendc_codegen_retries_timeout_before_failing():
    llm = _FakeLLMClient([TimeoutError("timed out"), "good patch"])
    task = _PreviewTask()
    generator = KernelGenerator(
        model_name="test-model",
        language="ascendc",
        llm_provider="openai",
        llm_client=llm,
    )

    result = generator._generate_code_from_prompt("base prompt", task=task)

    assert result["raw"] == "good patch"
    assert len(llm.prompts) == 2


def test_ascendc_codegen_does_not_retry_authentication_failure():
    llm = _FakeLLMClient(
        [
            LLMAuthenticationError(
                "Claude Agent SDK provider failed: API Error: 403 Access terminated"
            ),
            "good patch",
        ]
    )
    task = _PreviewTask()
    generator = KernelGenerator(
        model_name="test-model",
        language="ascendc",
        llm_provider="openai",
        llm_client=llm,
    )

    with pytest.raises(LLMAuthenticationError, match="403 Access terminated"):
        generator._generate_code_from_prompt("base prompt", task=task)

    assert len(llm.prompts) == 1


def test_ascendc_codegen_retry_refreshes_task_code_format_feedback():
    llm = _FakeLLMClient(["bad patch", "good full project"])

    class SwitchingFormatTask:
        def __init__(self):
            self.mode = "patch"

        def preview_parse_generated_code(self, *, raw_code):
            if raw_code == "bad patch":
                self.mode = "full"
                raise ValueError("patch failures reached fallback threshold")

        def get_code_format_text(self, *, language, target_gpu):
            if self.mode == "full":
                return "FULL_PROJECT_FORMAT"
            return "PATCH_FORMAT"

    generator = KernelGenerator(
        model_name="test-model",
        language="ascendc",
        target_gpu="Ascend910B3",
        llm_provider="openai",
        llm_client=llm,
    )

    result = generator._generate_code_from_prompt("base prompt", task=SwitchingFormatTask())

    assert result["raw"] == "good full project"
    assert len(llm.prompts) == 2
    assert "FULL_PROJECT_FORMAT" in llm.prompts[1]


def test_baseline_agentic_request_includes_explicit_run_and_task_context():
    captured = {}

    class FakeTask:
        name = "task_name"
        definition_name = "definition_name"
        _ksearch_run_id = "explicit-run"

        def get_agentic_definition_text(self, *, language):
            return f"agentic spec for {language}"

    class FakeAgenticRunner:
        def run_one_shot_closed(self, *, task, request, base_solution, max_fix_rounds):
            captured["request"] = request
            return SimpleNamespace(
                prompt_chars=17,
                changed_paths=["kernel/foo.h"],
                project_path="/tmp/project",
            )

    generator = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: ""),
    )
    generator._ascendc_agentic_runner = FakeAgenticRunner()

    generator._generate_ascendc_solution_agentically(
        task=FakeTask(),
        action_text="change code",
        trace_logs="trace",
        perf_summary="perf",
        round_num=2,
        attempt_idx=3,
        mode="action",
        base_solution=None,
    )

    request = captured["request"]
    assert request.run_id == "explicit-run"
    assert request.task_name == "task_name"
    assert request.round_num == 2
    assert request.attempt_idx == 3
