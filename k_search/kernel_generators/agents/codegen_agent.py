from __future__ import annotations

from typing import Any

from k_search.kernel_generators.agents.project_agent import ProjectAgent
from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticPromptBuilder,
)


class CodegenAgent(ProjectAgent):
    """Codegen role: edits the project to implement an action, optionally guided by code_map."""

    allowed_tools = ["Read", "Grep", "Glob", "Edit", "Write"]

    def __init__(self, *, model_name: str, editor_client: Any | None = None,
                 prompt_builder: AscendCAgenticPromptBuilder | None = None) -> None:
        super().__init__(model_name=model_name, editor_client=editor_client)
        self.prompt_builder = prompt_builder or AscendCAgenticPromptBuilder()

    def build_prompt(self, context: Any) -> str:
        request: AscendCAgenticCodegenRequest = context["request"]
        has_code_map = bool(context.get("has_code_map", False))
        task_path = context.get("task_path")
        return self.prompt_builder.build(request, has_code_map=has_code_map, task_path=task_path)
