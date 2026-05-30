from __future__ import annotations

import os
from typing import Any

from k_search.kernel_generators.agents.project_agent import ProjectAgent
from k_search.utils.path_sanitize import sanitize_worktree_paths

_READER_PROMPT_TEMPLATE = """\
You are a code-understanding agent working inside an AscendC operator project directory.
Your ONLY job is to read and understand the project, then write a concise CODE_MAP.md
at the project root (CWD). You MUST NOT modify, create, or delete any source file —
the only file you may write is CODE_MAP.md.

Available tools: Read, Grep, Glob, Write. Edit and Bash are disabled.
Do not read .git, build directories, caches, generated logs, or large artifacts.
Keep CODE_MAP.md under {max_chars} characters — it is a navigation map, not a copy of the code.
Describe what exists and where; do NOT propose optimizations or changes (that is another agent's job).

Operator task specification (for context only):
{definition_text}

Inspect the project with Glob/Grep/Read, then write CODE_MAP.md using EXACTLY this template:

# CODE_MAP

## Files
<relative/path> — <one-line role> (kernel / host tiling / InferShape / op-def / test harness / build / other)
... one line per source file ...

## Entry Points & Call Chain
- Public entry point(s): <symbol> in <file>
- Host → kernel launch path: <how host tiling reaches the kernel>

## Host <-> Kernel Contract
- Tiling struct / fields passed host->kernel: <...>
- Workspace / sync / block-dim assumptions: <...>

## Kernel Design
- Core split strategy: <...>
- Tiling formula (main/tail, tile sizes): <...>
- Buffer allocation (UB/L1, queue depth, double-buffer): <...>
- Pipeline (CopyIn -> Compute -> CopyOut) stages: <...>

## Invariants & Constraints
- Semantics that MUST be preserved: <...>
- Alignment / dtype / shape constraints: <...>
- Build layout assumptions: <...>

Finish by writing CODE_MAP.md. Output a one-line confirmation; do not paste the file back.
"""


def _default_code_map_max_chars() -> int:
    raw = os.getenv("KSEARCH_CODE_MAP_MAX_CHARS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 8000


class CodeReaderAgent(ProjectAgent):
    """Read-only role that produces the code_map memory (CODE_MAP.md)."""

    allowed_tools = ["Read", "Grep", "Glob", "Write"]
    disallowed_tools = ["Bash", "Edit"]

    def __init__(self, *, model_name: str, editor_client: Any | None = None, max_chars: int | None = None) -> None:
        super().__init__(model_name=model_name, editor_client=editor_client)
        self.max_chars = int(max_chars) if max_chars else _default_code_map_max_chars()

    def build_prompt(self, context: Any) -> str:
        definition_text = ""
        task_path = None
        if isinstance(context, dict):
            definition_text = str(context.get("definition_text", "") or "")
            task_path = context.get("task_path")
        prompt = _READER_PROMPT_TEMPLATE.format(
            max_chars=self.max_chars,
            definition_text=definition_text.strip() or "(no specification provided)",
        )
        prompt = sanitize_worktree_paths(prompt, task_path=task_path)
        return prompt
