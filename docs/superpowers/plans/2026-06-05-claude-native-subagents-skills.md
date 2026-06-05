# Claude Native Subagents and Skills Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace K-Search's Python `CodeReaderAgent` / `CodegenAgent` runtime path with Claude-native project subagents and skills for AscendC agentic codegen.

**Architecture:** Add repo-owned Claude assets under `k_search/kernel_generators/claude_assets/`, materialize them into each candidate worktree, and let one main Claude SDK session explicitly orchestrate `code-reader -> plan -> codegen -> reviewer`. Python keeps lifecycle ownership: worktree, memory, SDK options, eval, snapshots, artifacts, and validation of file handoff outputs.

**Tech Stack:** Python 3.10+, dataclasses, `importlib.resources`, pytest, Claude Agent SDK Python options, markdown filesystem assets.

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `k_search/kernel_generators/claude_assets/__init__.py` | Export native asset constants and materializer API. |
| `k_search/kernel_generators/claude_assets/materializer.py` | Copy K-Search-managed `.claude/agents` and `.claude/skills` assets into a worktree with overwrite safety. |
| `k_search/kernel_generators/claude_assets/agents/code-reader.md` | Native read/map subagent. |
| `k_search/kernel_generators/claude_assets/agents/plan.md` | Native strategy implementation planning subagent. |
| `k_search/kernel_generators/claude_assets/agents/codegen.md` | Native code editing subagent. |
| `k_search/kernel_generators/claude_assets/agents/reviewer.md` | Native review subagent writing `REVIEW_NOTES.md`. |
| `k_search/kernel_generators/claude_assets/agents/bug-fixer.md` | Reserved native bug-fix subagent file, not active in this release. |
| `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md` | AscendC native-agent workflow skill. |
| `k_search/kernel_generators/claude_assets/skills/ascendc-api-reference/SKILL.md` | Reference lookup skill for `references/`. |
| `k_search/kernel_generators/claude_agent_project_editor.py` | Add native `Skill` / `Agent` tool options, `setting_sources`, and `skills` support in both one-shot and session paths. |
| `k_search/kernel_generators/ascendc_agentic_codegen.py` | Materialize assets, remove old Python agent calls from `run()`, validate handoff files, filter handoff files from candidate source changes. |
| `k_search/kernel_generators/agents/` | Remove old runtime agent modules after runner migration; no production code should import them. |
| `tests/kernel_generators/test_claude_assets.py` | Materializer and asset content tests. |
| `tests/kernel_generators/test_claude_agent_sdk_mock.py` | SDK options tests. |
| `tests/kernel_generators/test_ascendc_agentic_codegen.py` | Runner migration and handoff validation tests. |

## Shared Constants

Use these exact handoff file constants in implementation and tests:

```python
NATIVE_HANDOFF_FILES = {
    "CODE_MAP.md",
    "IMPLEMENTATION_PLAN.md",
    "REVIEW_NOTES.md",
}

NATIVE_SKILLS = ["ascendc-codegen", "ascendc-api-reference"]
NATIVE_AGENT_TOOL_NAMES = ["Agent"]
```

`Task` is not included in `NATIVE_AGENT_TOOL_NAMES` by default. If old SDK compatibility is required later, add an env-based opt-in helper rather than passing `Task` unconditionally.

---

### Task 1: Add Claude Asset Materializer Tests

**Files:**
- Create: `tests/kernel_generators/test_claude_assets.py`
- Create later: `k_search/kernel_generators/claude_assets/__init__.py`
- Create later: `k_search/kernel_generators/claude_assets/materializer.py`

- [ ] **Step 1: Write failing tests for materialization**

Create `tests/kernel_generators/test_claude_assets.py` with:

```python
from pathlib import Path

import pytest


def test_materialize_claude_project_assets_writes_agents_and_skills(tmp_path):
    from k_search.kernel_generators.claude_assets import (
        NATIVE_AGENT_FILES,
        NATIVE_SKILL_FILES,
        materialize_claude_project_assets,
    )

    result = materialize_claude_project_assets(tmp_path)

    assert result.claude_dir == tmp_path / ".claude"
    assert sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in result.written_paths) == sorted(
        [f".claude/agents/{name}" for name in NATIVE_AGENT_FILES]
        + [f".claude/skills/{path}" for path in NATIVE_SKILL_FILES]
    )
    assert (tmp_path / ".claude" / "agents" / "code-reader.md").exists()
    assert (tmp_path / ".claude" / "agents" / "plan.md").exists()
    assert (tmp_path / ".claude" / "agents" / "codegen.md").exists()
    assert (tmp_path / ".claude" / "agents" / "reviewer.md").exists()
    assert (tmp_path / ".claude" / "agents" / "bug-fixer.md").exists()
    assert (tmp_path / ".claude" / "skills" / "ascendc-codegen" / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "skills" / "ascendc-api-reference" / "SKILL.md").exists()


def test_materializer_refuses_to_overwrite_unmanaged_files(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    target = tmp_path / ".claude" / "agents" / "code-reader.md"
    target.parent.mkdir(parents=True)
    target.write_text("user-owned agent file\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite unmanaged Claude asset"):
        materialize_claude_project_assets(tmp_path)


def test_materializer_can_refresh_managed_files(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    first = materialize_claude_project_assets(tmp_path)
    managed_file = tmp_path / ".claude" / "agents" / "code-reader.md"
    managed_file.write_text("<!-- K-Search managed Claude asset -->\ncorrupted\n", encoding="utf-8")

    second = materialize_claude_project_assets(tmp_path)

    assert first.claude_dir == second.claude_dir
    text = managed_file.read_text(encoding="utf-8")
    assert "name: code-reader" in text
    assert "corrupted" not in text


def test_asset_files_contain_required_handoff_contracts(tmp_path):
    from k_search.kernel_generators.claude_assets import materialize_claude_project_assets

    materialize_claude_project_assets(tmp_path)

    code_reader = (tmp_path / ".claude" / "agents" / "code-reader.md").read_text(encoding="utf-8")
    planner = (tmp_path / ".claude" / "agents" / "plan.md").read_text(encoding="utf-8")
    codegen = (tmp_path / ".claude" / "agents" / "codegen.md").read_text(encoding="utf-8")
    reviewer = (tmp_path / ".claude" / "agents" / "reviewer.md").read_text(encoding="utf-8")
    bug_fixer = (tmp_path / ".claude" / "agents" / "bug-fixer.md").read_text(encoding="utf-8")

    assert "CODE_MAP.md" in code_reader
    assert "IMPLEMENTATION_PLAN.md" in planner
    assert "IMPLEMENTATION_PLAN.md" in codegen
    assert "REVIEW_NOTES.md" in reviewer
    assert "reserved" in bug_fixer.lower()
    assert "not active" in bug_fixer.lower()
    for text in (code_reader, planner, codegen, reviewer, bug_fixer):
        assert "status" in text
        assert "files_written" in text
        assert "next" in text
        assert "Do not paste" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_claude_assets.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'k_search.kernel_generators.claude_assets'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/kernel_generators/test_claude_assets.py
git commit -m "test: specify claude native asset materialization"
```

---

### Task 2: Implement Claude Asset Materializer and Assets

**Files:**
- Create: `k_search/kernel_generators/claude_assets/__init__.py`
- Create: `k_search/kernel_generators/claude_assets/materializer.py`
- Create: `k_search/kernel_generators/claude_assets/agents/code-reader.md`
- Create: `k_search/kernel_generators/claude_assets/agents/plan.md`
- Create: `k_search/kernel_generators/claude_assets/agents/codegen.md`
- Create: `k_search/kernel_generators/claude_assets/agents/reviewer.md`
- Create: `k_search/kernel_generators/claude_assets/agents/bug-fixer.md`
- Create: `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md`
- Create: `k_search/kernel_generators/claude_assets/skills/ascendc-api-reference/SKILL.md`
- Test: `tests/kernel_generators/test_claude_assets.py`

- [ ] **Step 1: Create `__init__.py`**

```python
from k_search.kernel_generators.claude_assets.materializer import (
    CLAUDE_ASSET_MANAGED_MARKER,
    NATIVE_AGENT_FILES,
    NATIVE_HANDOFF_FILES,
    NATIVE_SKILL_FILES,
    NATIVE_SKILLS,
    ClaudeAssetMaterializationResult,
    materialize_claude_project_assets,
)

__all__ = [
    "CLAUDE_ASSET_MANAGED_MARKER",
    "NATIVE_AGENT_FILES",
    "NATIVE_HANDOFF_FILES",
    "NATIVE_SKILL_FILES",
    "NATIVE_SKILLS",
    "ClaudeAssetMaterializationResult",
    "materialize_claude_project_assets",
]
```

- [ ] **Step 2: Create `materializer.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


CLAUDE_ASSET_MANAGED_MARKER = "<!-- K-Search managed Claude asset -->"

NATIVE_AGENT_FILES = [
    "code-reader.md",
    "plan.md",
    "codegen.md",
    "reviewer.md",
    "bug-fixer.md",
]

NATIVE_SKILL_FILES = [
    "ascendc-codegen/SKILL.md",
    "ascendc-api-reference/SKILL.md",
]

NATIVE_SKILLS = ["ascendc-codegen", "ascendc-api-reference"]

NATIVE_HANDOFF_FILES = {
    "CODE_MAP.md",
    "IMPLEMENTATION_PLAN.md",
    "REVIEW_NOTES.md",
}


@dataclass(frozen=True)
class ClaudeAssetMaterializationResult:
    claude_dir: Path
    written_paths: list[Path]


def _asset_text(relative_path: str) -> str:
    root = files("k_search.kernel_generators.claude_assets")
    return (root / relative_path).read_text(encoding="utf-8")


def _write_managed_text(target: Path, text: str) -> None:
    if target.exists():
        current = target.read_text(encoding="utf-8", errors="replace")
        if not current.startswith(CLAUDE_ASSET_MANAGED_MARKER):
            raise FileExistsError(f"refusing to overwrite unmanaged Claude asset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    body = text if text.startswith(CLAUDE_ASSET_MANAGED_MARKER) else f"{CLAUDE_ASSET_MANAGED_MARKER}\n{text}"
    target.write_text(body, encoding="utf-8")


def materialize_claude_project_assets(project_dir: str | Path) -> ClaudeAssetMaterializationResult:
    root = Path(project_dir).expanduser().resolve()
    claude_dir = root / ".claude"
    written: list[Path] = []

    for name in NATIVE_AGENT_FILES:
        text = _asset_text(f"agents/{name}")
        target = claude_dir / "agents" / name
        _write_managed_text(target, text)
        written.append(target)

    for rel in NATIVE_SKILL_FILES:
        text = _asset_text(f"skills/{rel}")
        target = claude_dir / "skills" / rel
        _write_managed_text(target, text)
        written.append(target)

    return ClaudeAssetMaterializationResult(claude_dir=claude_dir, written_paths=written)
```

- [ ] **Step 3: Create native agent files**

Create `k_search/kernel_generators/claude_assets/agents/code-reader.md`:

```markdown
---
name: code-reader
description: Read-only AscendC project mapper. Use first when CODE_MAP.md is missing or stale.
tools: Read, Grep, Glob, Write
---

You are the K-Search AscendC code-reader subagent.

Read the current project and write or update CODE_MAP.md at the project root.
Do not modify source files. Do not propose optimizations. Do not run Bash.

CODE_MAP.md must include:
- source file roles
- public entry points and call chain
- host tiling to kernel launch contract
- tiling structs and fields
- workspace, block dim, dtype, shape, and alignment constraints
- kernel split strategy, buffers, and pipeline stages
- invariants that codegen must preserve

Final message contract:
- status: ok, needs_fix, or failed
- files_written: CODE_MAP.md
- next: plan

Do not paste CODE_MAP.md in the final message. The file is the handoff.
```

Create `k_search/kernel_generators/claude_assets/agents/plan.md`:

```markdown
---
name: plan
description: AscendC strategy implementation planner. Use after code-reader and before codegen.
tools: Read, Grep, Glob, Write
---

You are the K-Search AscendC planning subagent.

Read CODE_MAP.md, the current strategy text from the parent prompt, relevant trace or performance context, and referenced API docs under references/.
Write IMPLEMENTATION_PLAN.md at the project root.

IMPLEMENTATION_PLAN.md must include:
- the selected implementation objective for this attempt
- files expected to change
- APIs and constraints that must be checked
- exact edit sequence for codegen
- risks and invariants
- verification expectations for reviewer

Keep the plan scoped to one attempt. Prefer one focused change and one or two source files.
Do not edit source files. Do not run Bash.

Final message contract:
- status: ok, needs_fix, or failed
- files_written: IMPLEMENTATION_PLAN.md
- next: codegen

Do not paste IMPLEMENTATION_PLAN.md in the final message. The file is the handoff.
```

Create `k_search/kernel_generators/claude_assets/agents/codegen.md`:

```markdown
---
name: codegen
description: AscendC implementation agent. Use after plan to edit project files according to IMPLEMENTATION_PLAN.md.
tools: Read, Grep, Glob, Edit, Write
---

You are the K-Search AscendC codegen subagent.

Read IMPLEMENTATION_PLAN.md and CODE_MAP.md before editing.
Follow IMPLEMENTATION_PLAN.md. Do not invent a different strategy.
Edit only files inside the current project directory.
Do not edit .git, build directories, caches, logs, or generated artifacts.
Do not run Bash.

After source edits, update affected sections of CODE_MAP.md so later agents see the current project structure and contracts.

Final message contract:
- status: ok, needs_fix, or failed
- files_written: list of modified source files and CODE_MAP.md if updated
- next: reviewer

Do not paste source files or CODE_MAP.md in the final message. Files are the handoff.
```

Create `k_search/kernel_generators/claude_assets/agents/reviewer.md`:

```markdown
---
name: reviewer
description: AscendC candidate reviewer. Use after codegen to check scope, contracts, and handoff files.
tools: Read, Grep, Glob, Write
---

You are the K-Search AscendC reviewer subagent.

Review the current worktree after codegen. Read CODE_MAP.md and IMPLEMENTATION_PLAN.md.
Inspect the changed source files. Check:
- changed files match the implementation plan
- public entry points and host/kernel contract are preserved
- tiling fields, workspace, block dim, dtype, shape, alignment, and build layout remain coherent
- CODE_MAP.md matches the edited code
- no edits were made outside the intended candidate project scope

Write REVIEW_NOTES.md at the project root with:
- status: ok, needs_fix, or failed
- changed_files_reviewed
- contract_risks
- required_fixes
- eval_ready: true or false

If eval_ready is false, explain the minimal fix in REVIEW_NOTES.md. Do not edit source files yourself.
Do not run Bash.

Final message contract:
- status: ok, needs_fix, or failed
- files_written: REVIEW_NOTES.md
- next: python_eval when eval_ready is true, otherwise codegen

Do not paste REVIEW_NOTES.md in the final message. The file is the handoff.
```

Create `k_search/kernel_generators/claude_assets/agents/bug-fixer.md`:

```markdown
---
name: bug-fixer
description: Reserved K-Search AscendC bug-fix agent. Not active in the current flow.
tools: Read, Grep, Glob
---

This subagent is reserved for a future K-Search flow:
Python eval failure -> bug-fixer -> reviewer -> Python re-eval.

It is not active in the current K-Search native subagent flow.
If invoked in the current release, do not modify files.

Final message contract:
- status: failed
- files_written: []
- next: python_eval

Do not paste file contents in the final message.
```

- [ ] **Step 4: Create native skill files**

Create `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md`:

```markdown
---
name: ascendc-codegen
description: Use for K-Search AscendC candidate optimization attempts that need code-reader, plan, codegen, and reviewer coordination inside a project worktree.
---

# AscendC Codegen Workflow

You are working inside a K-Search candidate worktree.

Use this workflow:
1. Use code-reader when CODE_MAP.md is missing or stale.
2. Use plan to write IMPLEMENTATION_PLAN.md.
3. Use codegen to edit source files according to IMPLEMENTATION_PLAN.md.
4. Use reviewer to write REVIEW_NOTES.md.
5. Return only a concise final summary to the parent.

Cross-agent state must be written to files:
- CODE_MAP.md
- IMPLEMENTATION_PLAN.md
- REVIEW_NOTES.md

Do not paste those files into final messages.
Do not run Bash.
Do not edit outside the current working directory.
Preserve public entry points, host tiling contract, kernel launch behavior, correctness harness behavior, and build layout.
```

Create `k_search/kernel_generators/claude_assets/skills/ascendc-api-reference/SKILL.md`:

```markdown
---
name: ascendc-api-reference
description: Use when AscendC API signatures, constraints, examples, or anti-patterns are needed from K-Search references/ docs.
---

# AscendC API Reference Lookup

K-Search stores AscendC API documentation under references/.
When strategy text lists a path like references/api_reference_docs/.../RowMuls.md, read that file before editing code that uses the API.

Rules:
- Do not guess AscendC API signatures from general C++ experience.
- Prefer strategy-provided API summaries first, then read full docs for details.
- Treat anti-pattern warnings in strategy text as hard constraints.
- If a signature or constraint is missing, report it in IMPLEMENTATION_PLAN.md instead of inventing a call.
```

- [ ] **Step 5: Run materializer tests**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_claude_assets.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit materializer and assets**

```bash
git add k_search/kernel_generators/claude_assets tests/kernel_generators/test_claude_assets.py
git commit -m "feat: add claude native subagent and skill assets"
```

---

### Task 3: Add SDK Options Tests for Native Agents and Skills

**Files:**
- Modify: `tests/kernel_generators/test_claude_agent_sdk_mock.py`
- Modify later: `k_search/kernel_generators/claude_agent_project_editor.py`

- [ ] **Step 1: Add failing test for one-shot options**

Append to `tests/kernel_generators/test_claude_agent_sdk_mock.py`:

```python
def test_claude_project_editor_enables_project_skills_and_agent_tool(monkeypatch, tmp_path):
    from pathlib import Path
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return "edited"

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    client.edit_project(project_dir=tmp_path, prompt="Use native subagents.")

    options = sdk.client_calls[0].options.kwargs
    assert options["setting_sources"] == ["project"]
    assert options["skills"] == ["ascendc-codegen", "ascendc-api-reference"]
    assert "Skill" in options["allowed_tools"]
    assert "Agent" in options["allowed_tools"]
    assert "Bash" in options["disallowed_tools"]
```

- [ ] **Step 2: Add failing test for session options**

Append to the same file:

```python
def test_claude_project_editor_session_uses_same_native_options(monkeypatch, tmp_path):
    from pathlib import Path
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return "edited"

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    session = client.open_session(project_dir=tmp_path)
    try:
        client.send_prompt(session, prompt="Use native subagents.")
    finally:
        client.close_session(session)

    options = sdk.client_calls[0].options.kwargs
    assert options["setting_sources"] == ["project"]
    assert options["skills"] == ["ascendc-codegen", "ascendc-api-reference"]
    assert "Skill" in options["allowed_tools"]
    assert "Agent" in options["allowed_tools"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_enables_project_skills_and_agent_tool tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_session_uses_same_native_options -v
```

Expected: FAIL because `setting_sources`, `skills`, `Skill`, and `Agent` are not yet present in options.

- [ ] **Step 4: Commit failing SDK option tests**

```bash
git add tests/kernel_generators/test_claude_agent_sdk_mock.py
git commit -m "test: require native claude sdk options"
```

---

### Task 4: Implement Native SDK Options

**Files:**
- Modify: `k_search/kernel_generators/claude_agent_project_editor.py`
- Test: `tests/kernel_generators/test_claude_agent_sdk_mock.py`

- [ ] **Step 1: Import native skill constants**

At the top of `k_search/kernel_generators/claude_agent_project_editor.py`, after existing imports, add:

```python
from k_search.kernel_generators.claude_assets import NATIVE_SKILLS
```

- [ ] **Step 2: Add tool helpers and defaults**

Replace:

```python
DEFAULT_PROJECT_EDITOR_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]
```

with:

```python
DEFAULT_PROJECT_EDITOR_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]
DEFAULT_CLAUDE_NATIVE_TOOLS = ["Skill", "Agent"]


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
```

- [ ] **Step 3: Add dataclass fields**

In `ClaudeAgentProjectEditorClient`, change the `allowed_tools` field and add `setting_sources` and `skills`:

```python
    allowed_tools: list[str] = field(default_factory=lambda: _with_claude_native_tools(list(DEFAULT_PROJECT_EDITOR_TOOLS)))
    disallowed_tools: list[str] = field(default_factory=lambda: ["Bash", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet"])
    setting_sources: list[str] = field(default_factory=lambda: ["project"])
    skills: list[str] | str | None = field(default_factory=lambda: list(NATIVE_SKILLS))
```

Keep `thinking_enabled` and `timeout_seconds` after these fields.

- [ ] **Step 4: Add `_build_options_kwargs` helper**

Inside `ClaudeAgentProjectEditorClient`, before `edit_project`, add:

```python
    def _build_options_kwargs(self, project_root: Path) -> dict[str, Any]:
        options_kwargs: dict[str, Any] = {
            "cwd": str(project_root),
            "allowed_tools": _with_claude_native_tools(list(self.allowed_tools)),
            "disallowed_tools": list(self.disallowed_tools),
            "permission_mode": os.getenv("CLAUDE_AGENT_PERMISSION_MODE", "acceptEdits"),
            "model": self.model_name,
            "setting_sources": list(self.setting_sources),
        }
        if self.skills is not None:
            options_kwargs["skills"] = self.skills if isinstance(self.skills, str) else list(self.skills)
        if self.max_turns is not None:
            options_kwargs["max_turns"] = self.max_turns
        if not self.thinking_enabled:
            options_kwargs["thinking"] = {"type": "disabled"}
        return options_kwargs
```

- [ ] **Step 5: Use helper in `edit_project`**

In `edit_project`, replace the inline `options_kwargs` block at lines 93-103 with:

```python
            options_kwargs = self._build_options_kwargs(project_root)
```

Keep:

```python
            options = claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)
```

- [ ] **Step 6: Use helper in `_build_options`**

Replace `_build_options` body with:

```python
    def _build_options(self, project_root: Path) -> Any:
        """Build ClaudeAgentOptions for a given project root."""
        import claude_agent_sdk  # type: ignore

        return claude_agent_sdk.ClaudeAgentOptions(**self._build_options_kwargs(project_root))
```

- [ ] **Step 7: Run SDK tests**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_claude_agent_sdk_mock.py -v
```

Expected: PASS. Existing test `test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools` must be updated to expect `["Read", "Grep", "Glob", "Edit", "Write", "Skill", "Agent"]` if it fails on the old exact list.

- [ ] **Step 8: Commit SDK option implementation**

```bash
git add k_search/kernel_generators/claude_agent_project_editor.py tests/kernel_generators/test_claude_agent_sdk_mock.py
git commit -m "feat: enable claude native agents and skills in sdk options"
```

---

### Task 5: Add Native Prompt Builder Tests

**Files:**
- Modify: `tests/kernel_generators/test_ascendc_agentic_codegen.py`
- Modify later: `k_search/kernel_generators/ascendc_agentic_codegen.py`

- [ ] **Step 1: Update prompt builder tests**

In `test_prompt_builder_omits_full_project_container_and_includes_action`, add:

```python
    assert "code-reader -> plan -> codegen -> reviewer" in prompt
    assert "IMPLEMENTATION_PLAN.md" in prompt
    assert "REVIEW_NOTES.md" in prompt
    assert "bug-fixer" in prompt
    assert "must not be invoked" in prompt
```

In `test_prompt_builder_uses_code_map_branch_when_present`, replace:

```python
    assert "CODE_MAP.md" not in without_map
    assert "First inspect the project with Glob, Grep, and Read" in without_map
```

with:

```python
    assert "CODE_MAP.md already exists: yes" in with_map
    assert "CODE_MAP.md already exists: no" in without_map
    assert "Use the code-reader subagent to create CODE_MAP.md" in without_map
```

- [ ] **Step 2: Add final-message handoff test**

Append:

```python
def test_prompt_builder_requires_file_handoff_and_short_subagent_summaries():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x",
        action_text="optimize",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request, has_code_map=False)

    assert "CODE_MAP.md, IMPLEMENTATION_PLAN.md, and REVIEW_NOTES.md are the only trusted cross-subagent handoff" in prompt
    assert "status, files_written, and next" in prompt
    assert "Do not paste CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md, or source files" in prompt
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_omits_full_project_container_and_includes_action tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_uses_code_map_branch_when_present tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_requires_file_handoff_and_short_subagent_summaries -v
```

Expected: FAIL because current prompt does not include native subagent flow or strict handoff text.

- [ ] **Step 4: Commit failing prompt tests**

```bash
git add tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "test: require native subagent orchestration prompt"
```

---

### Task 6: Implement Native Prompt Builder

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Add handoff constants**

After imports in `ascendc_agentic_codegen.py`, add:

```python
IMPLEMENTATION_PLAN_FILENAME = "IMPLEMENTATION_PLAN.md"
REVIEW_NOTES_FILENAME = "REVIEW_NOTES.md"
NATIVE_HANDOFF_FILENAMES = {CODE_MAP.filename, IMPLEMENTATION_PLAN_FILENAME, REVIEW_NOTES_FILENAME}
```

- [ ] **Step 2: Replace inspect-line logic in `AscendCAgenticPromptBuilder.build`**

Inside `build`, replace the `if has_code_map` / `else` block with:

```python
        code_map_status = "yes" if has_code_map else "no"
        code_map_instruction = (
            "CODE_MAP.md already exists: yes. Instruct plan/codegen/reviewer to read it before acting.\n"
            if has_code_map
            else "CODE_MAP.md already exists: no. Use the code-reader subagent to create CODE_MAP.md before planning.\n"
        )
```

- [ ] **Step 3: Replace prompt body with native orchestration wording**

Replace the prompt string assembly with:

```python
        prompt = (
            "You are the main K-Search AscendC orchestration agent working inside a candidate project directory.\n"
            "IMPORTANT: You must ONLY edit files inside the current project directory (CWD). Do NOT use absolute paths from external directories.\n"
            f"Target GPU: {request.target_gpu}\n"
            f"Mode: {request.mode}\n"
            f"Round: {int(request.round_num)}\n"
            f"Attempt: {int(request.attempt_idx)}\n"
            f"CODE_MAP.md already exists: {code_map_status}\n\n"
            "Available tools include Read/Grep/Glob/Edit/Write, Skill, and Agent. Bash is disabled.\n"
            "Use the ascendc-codegen and ascendc-api-reference skills when relevant.\n"
            "Required native subagent flow: code-reader -> plan -> codegen -> reviewer.\n"
            "The bug-fixer subagent is reserved for future eval-failure repair and must not be invoked in this release.\n"
            + code_map_instruction
            + "The plan subagent must write IMPLEMENTATION_PLAN.md.\n"
            "The reviewer subagent must write REVIEW_NOTES.md.\n"
            "CODE_MAP.md, IMPLEMENTATION_PLAN.md, and REVIEW_NOTES.md are the only trusted cross-subagent handoff.\n"
            "Every subagent final message must be short and contain only status, files_written, and next.\n"
            "Do not paste CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md, or source files into final messages.\n"
            "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
            "Preserve operator semantics, public entry points, host tiling contract, correctness harness behavior, and build layout.\n"
            "End with a concise summary and changed-file list after reviewer says eval_ready is true.\n\n"
            "Task specification:\n"
            f"{sections['definition']}\n\n"
            "Chosen strategy/action/debug intent:\n"
            f"{sections['action']}\n\n"
            "Performance summary:\n"
            f"{sections['perf_summary'] or '(none)'}\n\n"
            "Recent failure or trace excerpt:\n"
            f"{sections['trace_logs'] or '(none)'}\n"
        )
```

- [ ] **Step 4: Run prompt tests**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_omits_full_project_container_and_includes_action tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_uses_code_map_branch_when_present tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_requires_file_handoff_and_short_subagent_summaries -v
```

Expected: PASS.

- [ ] **Step 5: Commit prompt builder implementation**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: build native subagent orchestration prompt"
```

---

### Task 7: Add Runner Migration Tests

**Files:**
- Modify: `tests/kernel_generators/test_ascendc_agentic_codegen.py`
- Modify later: `k_search/kernel_generators/ascendc_agentic_codegen.py`

- [ ] **Step 1: Add native editing client test helper**

Append this helper near existing `EditingClient`:

```python
class NativeEditingClient:
    def __init__(self, new_text: str = "alpha\nBETA\ngamma\n", *, write_code_map: bool = True, write_plan: bool = True, write_review: bool = True):
        self.new_text = new_text
        self.write_code_map = write_code_map
        self.write_plan = write_plan
        self.write_review = write_review
        self.calls = []

    def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        if self.write_code_map:
            (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h is the kernel file\n", encoding="utf-8")
        if self.write_plan:
            (root / "IMPLEMENTATION_PLAN.md").write_text("# IMPLEMENTATION_PLAN\nChange beta to BETA in kernel/foo.h.\n", encoding="utf-8")
        if self.write_review:
            (root / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
        (root / "kernel" / "foo.h").write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval",
            transcript="native subagents completed",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )
```

- [ ] **Step 2: Add test that runner materializes assets and calls one SDK session**

Append:

```python
def test_runner_materializes_native_assets_and_uses_single_project_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    client = NativeEditingClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert len(client.calls) == 1
    project_dir, prompt = client.calls[0]
    assert (project_dir / ".claude" / "agents" / "code-reader.md").exists()
    assert (project_dir / ".claude" / "agents" / "plan.md").exists()
    assert (project_dir / ".claude" / "agents" / "codegen.md").exists()
    assert (project_dir / ".claude" / "agents" / "reviewer.md").exists()
    assert (project_dir / ".claude" / "agents" / "bug-fixer.md").exists()
    assert (project_dir / ".claude" / "skills" / "ascendc-codegen" / "SKILL.md").exists()
    assert "code-reader -> plan -> codegen -> reviewer" in prompt
    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
```

- [ ] **Step 3: Add test that old Python agents are not imported**

Append:

```python
def test_runner_does_not_import_old_python_project_agents(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    forbidden = ModuleType("k_search.kernel_generators.agents")

    def _blocked_getattr(name):
        raise AssertionError(f"old Python agent import used: {name}")

    forbidden.__getattr__ = _blocked_getattr
    monkeypatch.setitem(sys.modules, "k_search.kernel_generators.agents", forbidden)

    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.changed_paths == ["kernel/foo.h"]
```

- [ ] **Step 4: Add tests for required handoff files**

Append:

```python
def test_runner_fails_when_implementation_plan_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_plan=False))

    with pytest.raises(RuntimeError, match="IMPLEMENTATION_PLAN.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_fails_when_review_notes_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_review=False))

    with pytest.raises(RuntimeError, match="REVIEW_NOTES.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )


def test_runner_fails_when_code_map_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient(write_code_map=False))

    with pytest.raises(RuntimeError, match="CODE_MAP.md"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
                target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
            ),
            base_solution=None,
        )
```

- [ ] **Step 5: Add test that handoff files do not count as candidate source changes**

Append:

```python
def test_runner_filters_handoff_and_claude_asset_paths_from_changed_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NativeEditingClient())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    assert result.changed_paths == ["kernel/foo.h"]
    assert "IMPLEMENTATION_PLAN.md" in result.diff_text
    assert "REVIEW_NOTES.md" in result.diff_text
    assert ".claude/agents" not in result.diff_text
```

- [ ] **Step 6: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_materializes_native_assets_and_uses_single_project_edit tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_does_not_import_old_python_project_agents tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_fails_when_implementation_plan_missing tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_fails_when_review_notes_missing tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_fails_when_code_map_missing tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_filters_handoff_and_claude_asset_paths_from_changed_paths -v
```

Expected: FAIL because runner still imports old agents, does not materialize assets, and does not validate native handoff files.

- [ ] **Step 7: Commit failing runner migration tests**

```bash
git add tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "test: require native subagent runner flow"
```

---

### Task 8: Implement Runner Native Flow

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Import materializer**

At the top of `ascendc_agentic_codegen.py`, add:

```python
from k_search.kernel_generators.claude_assets import (
    NATIVE_HANDOFF_FILES,
    materialize_claude_project_assets,
)
```

Remove runtime imports of `CodeReaderAgent` and `CodegenAgent` inside `run()`.

- [ ] **Step 2: Add handoff helpers**

After `_edit_project_with_optional_telemetry`, add:

```python
def _is_native_handoff_path(path: str) -> bool:
    rel = str(path or "").replace("\\", "/").strip()
    if not rel:
        return True
    if rel.startswith(".claude/"):
        return True
    return rel in NATIVE_HANDOFF_FILES


def _candidate_changed_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not _is_native_handoff_path(path)]


def _require_native_handoff_files(project_dir: Path) -> None:
    missing = [name for name in sorted(NATIVE_HANDOFF_FILES) if not (project_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Claude native subagent flow did not produce required handoff file(s): {', '.join(missing)}")
```

- [ ] **Step 3: Materialize assets and commit them as baseline**

In `AscendCAgenticCodegenRunner.run()`, after overlay/baseline commit and before memory handling, insert:

```python
            materialize_claude_project_assets(session.project_dir)
            session.commit_all("ksearch native claude assets baseline")
```

- [ ] **Step 4: Replace code_map reader block**

Replace lines 222-249 in `run()` with:

```python
            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = False
            if store is not None:
                has_code_map = store.materialize(CODE_MAP, session.project_dir)
```

Do not instantiate `CodeReaderAgent`.

- [ ] **Step 5: Replace CodegenAgent prompt/call block**

Replace lines 251-283 in `run()` with:

```python
            prompt = self.prompt_builder.build(
                request,
                has_code_map=has_code_map,
                task_path=str(getattr(task, "task_path", "") or ""),
            )
            task_path = getattr(task, "task_path", None)
            if task_path is not None:
                prompt = prompt.replace(str(Path(task_path).expanduser().resolve()), "<PROJECT_ROOT>")
            telemetry_context = TelemetryContext(
                task_name=getattr(task, "definition_name", None),
                definition=getattr(task, "definition_name", None),
                flow="agentic_codegen",
                stage=request.mode,
                round_index=request.round_num,
                attempt_index=request.attempt_idx,
                model_name=self.model_name,
                provider="claude-agent",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
            try:
                edit_result = _edit_project_with_optional_telemetry(
                    self.editor_client,
                    project_dir=session.project_dir,
                    prompt=prompt,
                    telemetry_recorder=telemetry_recorder,
                )
            finally:
                telemetry_recorder.close()
            _require_native_handoff_files(session.project_dir)
```

- [ ] **Step 6: Save and remove `CODE_MAP.md` before diff**

Keep current `code_map_text` read and unlink behavior:

```python
            code_map_text = (session.project_dir / CODE_MAP.filename).read_text(encoding="utf-8", errors="replace")
            if store is not None:
                store.save(CODE_MAP, code_map_text)
            (session.project_dir / CODE_MAP.filename).unlink(missing_ok=True)
```

This replaces the existing `code_map_text = store.read_from_worktree(...) if store is not None else None` block.

- [ ] **Step 7: Filter changed paths**

Where `changed_paths` is computed from `project_changed_paths` or `session.changed_paths()`, replace filtering:

```python
            changed_paths = _candidate_changed_paths(project_changed_paths or session.changed_paths())
```

Apply the same replacement in the mirror-sync retry block.

- [ ] **Step 8: Add native metadata**

In `write_agentic_candidate_artifacts(... metadata={...})`, add:

```python
                    "native_claude_agents": True,
                    "native_handoff_files": sorted(NATIVE_HANDOFF_FILES),
```

- [ ] **Step 9: Run runner tests**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v
```

Expected: PASS after updating existing tests that used `EditingClient` to write `CODE_MAP.md`, `IMPLEMENTATION_PLAN.md`, and `REVIEW_NOTES.md`, or replacing those clients with `NativeEditingClient`.

- [ ] **Step 10: Commit runner native flow**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: run ascendc codegen through native claude subagents"
```

---

### Task 9: Update End-to-End Mock Test for Native Flow

**Files:**
- Modify: `tests/kernel_generators/test_claude_agent_sdk_mock.py`

- [ ] **Step 1: Update mock edit functions to write handoff files**

In `test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization`, update `first_edit` and `second_edit`:

```python
    def first_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
        (project_dir / "IMPLEMENTATION_PLAN.md").write_text("# IMPLEMENTATION_PLAN\nInitial safe edit.\n", encoding="utf-8")
        (project_dir / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
        (project_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n// initial agent edit\n", encoding="utf-8")
        return "status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval"

    def second_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
        (project_dir / "IMPLEMENTATION_PLAN.md").write_text("# IMPLEMENTATION_PLAN\nChange beta to BETA.\n", encoding="utf-8")
        (project_dir / "REVIEW_NOTES.md").write_text("status: ok\neval_ready: true\n", encoding="utf-8")
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        return "status: ok\nfiles_written: kernel/foo.h, CODE_MAP.md, IMPLEMENTATION_PLAN.md, REVIEW_NOTES.md\nnext: python_eval"
```

- [ ] **Step 2: Add SDK options assertions**

After existing `assert sdk.client_calls[0].options.kwargs["cwd"]`, add:

```python
    assert sdk.client_calls[0].options.kwargs["setting_sources"] == ["project"]
    assert "Skill" in sdk.client_calls[0].options.kwargs["allowed_tools"]
    assert "Agent" in sdk.client_calls[0].options.kwargs["allowed_tools"]
```

- [ ] **Step 3: Run updated E2E mock test**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization -v
```

Expected: PASS.

- [ ] **Step 4: Commit E2E mock update**

```bash
git add tests/kernel_generators/test_claude_agent_sdk_mock.py
git commit -m "test: update claude agent mock flow for native handoff files"
```

---

### Task 10: Remove Old Python Agent Runtime Modules

**Files:**
- Delete: `k_search/kernel_generators/agents/code_reader_agent.py`
- Delete: `k_search/kernel_generators/agents/codegen_agent.py`
- Delete: `k_search/kernel_generators/agents/project_agent.py`
- Modify: `k_search/kernel_generators/agents/__init__.py`
- Delete: `tests/kernel_generators/test_project_agents.py`

- [ ] **Step 1: Verify no production imports remain**

Run:

```bash
rg -n "CodeReaderAgent|CodegenAgent|ProjectAgent|k_search\\.kernel_generators\\.agents" k_search tests
```

Expected before deletion: references only in `k_search/kernel_generators/agents/` and `tests/kernel_generators/test_project_agents.py`. If `ascendc_agentic_codegen.py` appears, finish Task 8 first.

- [ ] **Step 2: Delete old modules**

Use `apply_patch` or equivalent file delete for:

```text
k_search/kernel_generators/agents/code_reader_agent.py
k_search/kernel_generators/agents/codegen_agent.py
k_search/kernel_generators/agents/project_agent.py
tests/kernel_generators/test_project_agents.py
```

- [ ] **Step 3: Replace package `__init__.py` with compatibility message**

Set `k_search/kernel_generators/agents/__init__.py` to:

```python
"""Deprecated Python agent role package.

K-Search AscendC agentic codegen now uses Claude-native project assets under
``k_search.kernel_generators.claude_assets``. Runtime code must not import
Python CodeReaderAgent/CodegenAgent roles.
"""

__all__: list[str] = []
```

- [ ] **Step 4: Verify imports are gone**

Run:

```bash
rg -n "CodeReaderAgent|CodegenAgent|ProjectAgent|k_search\\.kernel_generators\\.agents" k_search tests
```

Expected: no output except the deprecation docstring if it includes the old names. If the docstring causes output, verify no executable imports remain:

```bash
rg -n "from k_search\\.kernel_generators\\.agents|import k_search\\.kernel_generators\\.agents" k_search tests
```

Expected: no output.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
python3 -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_claude_assets.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit old agent removal**

```bash
git add -A k_search/kernel_generators/agents tests/kernel_generators/test_project_agents.py
git commit -m "refactor: remove python project agent runtime roles"
```

---

### Task 11: Full Regression and Spec Coverage Check

**Files:**
- No code files.
- Use existing tests.

- [ ] **Step 1: Run focused kernel generator regression**

Run:

```bash
python3 -m pytest tests/kernel_generators -v
```

Expected: PASS.

- [ ] **Step 2: Run supporting telemetry and utils tests touched by this path**

Run:

```bash
python3 -m pytest tests/telemetry tests/utils -v
```

Expected: PASS.

- [ ] **Step 3: Inspect final diff for scope**

Run:

```bash
git diff --stat main..HEAD
git diff --name-only main..HEAD
```

Expected changed paths are limited to:

```text
docs/superpowers/specs/2026-06-05-claude-native-subagents-skills-design.md
docs/superpowers/plans/2026-06-05-claude-native-subagents-skills.md
k_search/kernel_generators/claude_agent_project_editor.py
k_search/kernel_generators/ascendc_agentic_codegen.py
k_search/kernel_generators/claude_assets/
k_search/kernel_generators/agents/
tests/kernel_generators/
```

- [ ] **Step 4: Commit plan completion note only if tests required code adjustments**

If regression failures required additional test or implementation fixes, commit those fixes:

```bash
git add k_search tests
git commit -m "fix: stabilize native claude subagent regression tests"
```

If no fixes were required, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Native `.claude/agents` and `.claude/skills` assets: Tasks 1-2.
- SDK `setting_sources`, `skills`, `Skill`, `Agent`: Tasks 3-4.
- `code-reader -> plan -> codegen -> reviewer`: Tasks 5-8.
- `bug-fixer` reserved file only: Task 2.
- File handoff via `CODE_MAP.md`, `IMPLEMENTATION_PLAN.md`, `REVIEW_NOTES.md`: Tasks 5-8.
- Short subagent final summaries: Tasks 2 and 5-6.
- No old Python agent runtime dependency: Tasks 7-10.
- Memory and artifact preservation: Task 8 and Task 11.
- No Bash / no eval by Claude: Task 2 assets and Task 4 SDK disallowed tools.

Placeholder scan:

- No banned placeholder tokens remain in executable plan steps.
- No step asks the implementer to invent missing behavior.
- No step asks the implementer to invent missing behavior.

Type consistency:

- `materialize_claude_project_assets(project_dir: str | Path) -> ClaudeAssetMaterializationResult` is used consistently.
- `setting_sources` and `skills` use Python SDK snake_case option names.
- Handoff filenames are shared through `NATIVE_HANDOFF_FILES`.
- `Agent` is the default native subagent tool name; `Task` is not passed by default.
