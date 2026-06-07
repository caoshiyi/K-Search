# Custom Subagent Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AscendC Claude subagent flow stage order and handoff artifacts configurable through `KSEARCH_SUBAGENT_FLOW_CONFIG`, including minimal flows such as `code-reader -> codegen`.

**Architecture:** Keep the existing JSON flow model and staged SDK session driver. Thread the active flow's configured `required_files` through prompt rendering, handoff capture, cleanup, eval-copy filtering, source import, and snapshot creation so configured artifacts are treated as runtime handoffs rather than candidate source files. Keep the built-in default flow behavior unchanged and make Claude asset instructions defer to the active stage prompt.

**Tech Stack:** Python dataclasses, stdlib path handling, existing Claude Agent SDK editor abstraction, pytest.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `k_search/kernel_generators/runtime_artifacts.py` | Add reusable helpers for combining built-in runtime files with flow-configured handoff files. |
| `k_search/kernel_generators/project_snapshot.py` | Accept per-call extra runtime files so custom handoffs are excluded from snapshot manifests and archives. |
| `k_search/tasks/ascendc_task.py` | Accept per-call extra runtime files so custom handoffs are excluded from solution sources and rejected as changed source paths. |
| `k_search/kernel_generators/ascendc_agentic_codegen.py` | Derive prompt text, required handoffs, cleanup, changed-path filtering, eval copy filtering, snapshot filtering, and metadata from the active flow configs. |
| `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md` | Replace default four-stage workflow language with active-flow language. |
| `k_search/kernel_generators/claude_assets/agents/codegen.md` | Make `ASCENDC_DESIGN.md` optional unless the stage prompt requires it. |
| `k_search/kernel_generators/claude_assets/agents/reviewer.md` | Make default handoff files optional unless the active flow requires them. |
| `k_search/kernel_generators/claude_assets/agents/knowledge-curator.md` | Read available evidence and avoid assuming `REVIEW_NOTES.md` always exists. |
| `README.md` | Document `KSEARCH_SUBAGENT_FLOW_CONFIG` with a minimal custom-flow example. |
| `tests/kernel_generators/test_runtime_artifacts.py` | New focused tests for runtime path helpers. |
| `tests/kernel_generators/test_project_snapshot.py` | Extend snapshot skip tests for configured handoff files. |
| `tests/test_ascendc_task.py` | Extend source import and changed-path validation tests for configured handoff files. |
| `tests/test_claude_subagent_flow_policy.py` | Extend prompt-policy tests for custom flows. |
| `tests/kernel_generators/test_ascendc_agentic_codegen.py` | Add end-to-end staged custom-flow test with custom handoff artifact. |
| `tests/kernel_generators/test_claude_assets.py` | Extend asset text tests so assets no longer hardcode the default flow as global policy. |

## Task 1: Runtime Helpers and Snapshot Filtering

**Files:**
- Modify: `k_search/kernel_generators/runtime_artifacts.py`
- Modify: `k_search/kernel_generators/project_snapshot.py`
- Create: `tests/kernel_generators/test_runtime_artifacts.py`
- Modify: `tests/kernel_generators/test_project_snapshot.py`

- [ ] **Step 1: Write failing runtime helper tests**

Create `tests/kernel_generators/test_runtime_artifacts.py`:

```python
from k_search.kernel_generators.runtime_artifacts import (
    native_runtime_files,
    is_native_runtime_path,
    normalize_runtime_file_names,
)


def test_runtime_file_set_includes_configured_handoff_files():
    files = native_runtime_files(extra_files={"CODEGEN_NOTES.md", "reports/codegen-summary.md"})

    assert "CODE_MAP.md" in files
    assert "CODEGEN_NOTES.md" in files
    assert "reports/codegen-summary.md" in files


def test_is_native_runtime_path_accepts_configured_handoff_files():
    extra = {"CODEGEN_NOTES.md", "reports/codegen-summary.md"}

    assert is_native_runtime_path("CODEGEN_NOTES.md", extra_files=extra)
    assert is_native_runtime_path("reports/codegen-summary.md", extra_files=extra)
    assert is_native_runtime_path(".claude/agents/codegen.md", extra_files=extra)
    assert not is_native_runtime_path("kernel/foo.h", extra_files=extra)


def test_normalize_runtime_file_names_rejects_unsafe_paths():
    try:
        normalize_runtime_file_names({"../escape.md"})
    except ValueError as exc:
        assert "unsafe runtime file path" in str(exc)
    else:
        raise AssertionError("unsafe runtime file path was accepted")
```

- [ ] **Step 2: Write failing snapshot filtering test**

Append to `tests/kernel_generators/test_project_snapshot.py`:

```python
def test_project_snapshot_skips_configured_runtime_files(tmp_path):
    from k_search.kernel_generators.project_snapshot import create_project_snapshot

    project = tmp_path / "project"
    project.mkdir()
    (project / "kernel").mkdir()
    (project / "kernel" / "foo.h").write_text("alpha\n", encoding="utf-8")
    (project / "CODEGEN_NOTES.md").write_text("handoff\n", encoding="utf-8")
    archive_dir = tmp_path / "snapshots"

    snapshot = create_project_snapshot(
        project_dir=project,
        snapshot_id="s1",
        parent_snapshot_id=None,
        base_commit="base",
        created_by_round=1,
        archive_dir=archive_dir,
        extra_skip_files={"CODEGEN_NOTES.md"},
    )

    assert "kernel/foo.h" in snapshot.manifest
    assert "CODEGEN_NOTES.md" not in snapshot.manifest
    assert (archive_dir / "s1" / "kernel" / "foo.h").is_file()
    assert not (archive_dir / "s1" / "CODEGEN_NOTES.md").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/kernel_generators/test_runtime_artifacts.py tests/kernel_generators/test_project_snapshot.py::test_project_snapshot_skips_configured_runtime_files -q
```

Expected: FAIL because `native_runtime_files`, `normalize_runtime_file_names`, and the `extra_skip_files` parameter do not exist.

- [ ] **Step 4: Implement runtime helper functions**

In `k_search/kernel_generators/runtime_artifacts.py`, add `PurePosixPath` and `Iterable` imports and these helpers:

```python
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def _normalize_runtime_file_name(path: str | Path) -> str:
    rel = str(path or "").strip().replace("\\", "/")
    if not rel:
        return ""
    posix = PurePosixPath(rel)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe runtime file path: {path!r}")
    return rel


def normalize_runtime_file_names(extra_files: Iterable[str | Path] | None = None) -> set[str]:
    out: set[str] = set()
    for item in extra_files or ():
        rel = _normalize_runtime_file_name(item)
        if rel:
            out.add(rel)
    return out


def native_runtime_files(extra_files: Iterable[str | Path] | None = None) -> set[str]:
    return set(NATIVE_RUNTIME_FILES) | normalize_runtime_file_names(extra_files)
```

Then replace `is_native_runtime_path()` with:

```python
def is_native_runtime_path(path: str | Path, *, extra_files: Iterable[str | Path] | None = None) -> bool:
    rel = _normalize_runtime_file_name(path)
    if not rel:
        return True
    parts = tuple(part for part in rel.split("/") if part)
    if any(part in NATIVE_RUNTIME_DIRS for part in parts):
        return True
    return rel in native_runtime_files(extra_files)
```

- [ ] **Step 5: Implement per-call snapshot filtering**

In `k_search/kernel_generators/project_snapshot.py`, import `Iterable` and `normalize_runtime_file_names`:

```python
from collections.abc import Iterable
from k_search.kernel_generators.runtime_artifacts import (
    NATIVE_RUNTIME_DIRS,
    NATIVE_RUNTIME_FILES,
    normalize_runtime_file_names,
)
```

Replace `_is_skipped()` with:

```python
def _snapshot_skip_names(extra_skip_files: Iterable[str | Path] | None = None) -> set[str]:
    return set(SNAPSHOT_SKIP_DIRS) | normalize_runtime_file_names(extra_skip_files)


def _is_skipped(rel: Path, *, extra_skip_files: Iterable[str | Path] | None = None) -> bool:
    names = _snapshot_skip_names(extra_skip_files)
    rel_text = str(rel).replace("\\", "/")
    return rel_text in names or any(part in names for part in rel.parts)
```

Change `_copy_snapshot_payload()` signature and calls:

```python
def _copy_snapshot_payload(src: Path, dst: Path, *, extra_skip_files: Iterable[str | Path] | None = None) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if _is_skipped(rel, extra_skip_files=extra_skip_files):
            continue
        target = dst / rel
        if p.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(p), target)
        elif p.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        elif p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
```

Change `create_project_snapshot()` signature and its two `_is_skipped()` / `_copy_snapshot_payload()` calls:

```python
def create_project_snapshot(
    *,
    project_dir: str | Path,
    snapshot_id: str,
    parent_snapshot_id: str | None,
    base_commit: str | None,
    created_by_round: int,
    eval_result: dict[str, Any] | None = None,
    diff_from_parent: str | None = None,
    archive_dir: str | Path | None = None,
    run_id: str | None = None,
    extra_skip_files: Iterable[str | Path] | None = None,
) -> ProjectSnapshot:
```

Inside the existing manifest loop, change:

```python
if _is_skipped(rel_path):
    continue
```

to:

```python
if _is_skipped(rel_path, extra_skip_files=extra_skip_files):
    continue
```

Inside the existing archive block, change:

```python
_copy_snapshot_payload(root, archive_dst)
```

to:

```python
_copy_snapshot_payload(root, archive_dst, extra_skip_files=extra_skip_files)
```

Change `materialize_project_snapshot()` to preserve current behavior:

```python
    _copy_snapshot_payload(src, dst)
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/kernel_generators/test_runtime_artifacts.py tests/kernel_generators/test_project_snapshot.py::test_project_snapshot_skips_configured_runtime_files -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add k_search/kernel_generators/runtime_artifacts.py k_search/kernel_generators/project_snapshot.py tests/kernel_generators/test_runtime_artifacts.py tests/kernel_generators/test_project_snapshot.py
git commit -m "feat: add configurable runtime artifact filters"
```

## Task 2: AscendC Task Source Filtering

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`
- Modify: `tests/test_ascendc_task.py`

- [ ] **Step 1: Write failing source filtering tests**

Append to `tests/test_ascendc_task.py`:

```python
def test_make_solution_from_project_dir_excludes_configured_runtime_files(tmp_path):
    from k_search.tasks.ascendc_task import AscendCTask

    project = tmp_path / "project"
    (project / "kernel").mkdir(parents=True)
    (project / "kernel" / "foo.h").write_text("alpha\n", encoding="utf-8")
    (project / "CODEGEN_NOTES.md").write_text("handoff\n", encoding="utf-8")
    task = AscendCTask(task_path=project, definition_name="x")

    solution = task.make_solution_from_project_dir(
        project_dir=project,
        changed_paths=["kernel/foo.h"],
        raw_agent_output="ok",
        round_num=1,
        model_name="claude",
        target_gpu="ascend_910b",
        language="ascendc",
        extra_runtime_files={"CODEGEN_NOTES.md"},
    )

    assert {src.path for src in solution.sources} == {"kernel/foo.h"}


def test_make_solution_from_project_dir_rejects_configured_runtime_changed_path(tmp_path):
    from k_search.tasks.ascendc_task import AscendCTask

    project = tmp_path / "project"
    (project / "kernel").mkdir(parents=True)
    (project / "kernel" / "foo.h").write_text("alpha\n", encoding="utf-8")
    (project / "CODEGEN_NOTES.md").write_text("handoff\n", encoding="utf-8")
    task = AscendCTask(task_path=project, definition_name="x")

    try:
        task.make_solution_from_project_dir(
            project_dir=project,
            changed_paths=["CODEGEN_NOTES.md"],
            raw_agent_output="ok",
            round_num=1,
            model_name="claude",
            target_gpu="ascend_910b",
            language="ascendc",
            extra_runtime_files={"CODEGEN_NOTES.md"},
        )
    except ValueError as exc:
        assert "forbidden agentic changed path" in str(exc)
    else:
        raise AssertionError("configured runtime changed path was accepted")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ascendc_task.py::test_make_solution_from_project_dir_excludes_configured_runtime_files tests/test_ascendc_task.py::test_make_solution_from_project_dir_rejects_configured_runtime_changed_path -q
```

Expected: FAIL because `make_solution_from_project_dir()` does not accept `extra_runtime_files`.

- [ ] **Step 3: Implement extra runtime filtering in AscendCTask**

Update imports in `k_search/tasks/ascendc_task.py`:

```python
from collections.abc import Iterable
from k_search.kernel_generators.runtime_artifacts import (
    NATIVE_RUNTIME_DIRS,
    NATIVE_RUNTIME_FILES,
    is_native_runtime_path,
)
```

Change helper signatures and runtime checks:

```python
def _is_forbidden_agentic_changed_path(
    path: str,
    *,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> bool:
    rel = str(path or "").strip().replace("\\", "/")
    if not rel:
        return True
    parts = tuple(p for p in rel.split("/") if p)
    return is_native_runtime_path(rel, extra_files=extra_runtime_files) or any(
        part in FORBIDDEN_AGENTIC_PATH_PARTS for part in parts
    )


def _collect_project_sources(
    root: Path,
    *,
    max_files: int = 80,
    max_bytes_per_file: int = 200_000,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> list[SourceFile]:
```

Inside the existing `_collect_project_sources()` loop, replace:

```python
if rel in NATIVE_RUNTIME_FILES or any(part in FORBIDDEN_AGENTIC_PATH_PARTS for part in rel_path.parts):
    continue
```

with:

```python
if is_native_runtime_path(rel, extra_files=extra_runtime_files) or any(
    part in FORBIDDEN_AGENTIC_PATH_PARTS for part in rel_path.parts
):
    continue
```

Apply the same `extra_runtime_files` parameter to:

```python
def _remove_project_source_candidates(
    root: Path,
    *,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> None:

def _validate_agentic_changed_paths(
    self,
    *,
    project_dir: str | Path,
    changed_paths: list[str] | None,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> None:

def _validate_solution_sources_under_root(
    self,
    root: Path,
    sources: list[SourceFile],
    *,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> None:
```

Inside those three existing function bodies, replace each `rel in NATIVE_RUNTIME_FILES` path check with:

```python
is_native_runtime_path(rel, extra_files=extra_runtime_files)
```

Preserve the existing forbidden path-part checks and existing error messages.

Change `make_solution_from_project_dir()` signature and calls:

```python
def make_solution_from_project_dir(
    self,
    *,
    project_dir: str | Path,
    changed_paths: list[str] | None,
    raw_agent_output: str,
    round_num: int,
    model_name: str,
    target_gpu: str,
    language: str,
    extra_runtime_files: Iterable[str | Path] | None = None,
) -> Solution:
    root = Path(project_dir).expanduser().resolve()
    self._validate_agentic_changed_paths(
        project_dir=root,
        changed_paths=changed_paths,
        extra_runtime_files=extra_runtime_files,
    )
    sources = _collect_project_sources(root, extra_runtime_files=extra_runtime_files)
    if not sources:
        raise ValueError(f"agentic project produced no source files: {root}")
    self._validate_solution_sources_under_root(root, sources, extra_runtime_files=extra_runtime_files)
    return Solution(
        name=f"{model_name}_{self.name}_ascendc_agentic_r{int(round_num)}",
        definition=self.name,
        author=str(model_name),
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=[str(target_gpu or "ascend")],
            entry_point=_default_entry_point(sources),
        ),
        sources=sources,
        description=(
            f"{model_name} agentic AscendC project for {self.name} "
            f"(round {int(round_num)}): {str(raw_agent_output or '').strip()[:500]}"
        ),
    )
```

Leave existing callers unchanged; `extra_runtime_files` defaults to `None`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ascendc_task.py::test_make_solution_from_project_dir_excludes_configured_runtime_files tests/test_ascendc_task.py::test_make_solution_from_project_dir_rejects_configured_runtime_changed_path -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat: filter configured runtime files from AscendC solutions"
```

## Task 3: Flow-Derived Prompt Rendering

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Modify: `tests/test_claude_subagent_flow_policy.py`
- Modify: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Write failing custom-flow prompt policy test**

Append to `tests/test_claude_subagent_flow_policy.py`:

```python
def test_prompt_builder_uses_custom_flow_outputs_without_default_stage_artifacts():
    initial = SubagentFlowConfig(
        name="minimal-initial-codegen",
        description="minimal",
        stages=(
            SubagentStageConfig(
                name="code-reader",
                agent="code-reader",
                instruction="Write CODE_MAP.md.",
                required_files=("CODE_MAP.md",),
                run_when_missing_files=("CODE_MAP.md",),
            ),
            SubagentStageConfig(
                name="codegen",
                agent="codegen",
                instruction="Implement and write CODEGEN_NOTES.md.",
                required_files=("CODE_MAP.md", "CODEGEN_NOTES.md"),
            ),
        ),
    )

    prompt = AscendCAgenticPromptBuilder(max_chars=20_000).build(
        _request(),
        initial_flow=initial,
        repair_flow=initial,
    )

    assert "Required native subagent flow: code-reader -> codegen." in prompt
    assert "CODEGEN_NOTES.md" in prompt
    assert "designer subagent must write ASCENDC_DESIGN.md" not in prompt
    assert "reviewer subagent must write REVIEW_NOTES.md" not in prompt
    assert "code-reader -> designer -> codegen -> reviewer" not in prompt
    assert "Eval-failure repair flow agents: code-reader, codegen." in prompt
```

Update `tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_omits_full_project_container_and_includes_action` to keep asserting the default flow still includes:

```python
assert "Required native subagent flow: code-reader -> designer -> codegen -> reviewer." in prompt
assert "Configured stage outputs:" in prompt
assert "Trusted handoff files:" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_claude_subagent_flow_policy.py::test_prompt_builder_uses_custom_flow_outputs_without_default_stage_artifacts tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_omits_full_project_container_and_includes_action -q
```

Expected: FAIL because `AscendCAgenticPromptBuilder.build()` does not accept `initial_flow` or `repair_flow`, and the prompt still hardcodes default stage artifacts.

- [ ] **Step 3: Implement flow-derived prompt helpers**

In `k_search/kernel_generators/ascendc_agentic_codegen.py`, replace `_render_flow_policy()` with a version that keeps its public name but renders arbitrary flows:

```python
def _flow_agent_list(flow: SubagentFlowConfig) -> str:
    return ", ".join(stage.agent for stage in flow.stages) or "(none)"


def _flow_required_files(flow: SubagentFlowConfig) -> set[str]:
    return {path for stage in flow.stages for path in stage.required_files}


def _flow_stage_outputs_text(flow: SubagentFlowConfig, *, label: str) -> str:
    lines = [f"{label} stages:"]
    for stage in flow.stages:
        required = ", ".join(stage.required_files) if stage.required_files else "(none)"
        lines.append(f"- {stage.name} [{stage.agent}]: {required}")
    return "\n".join(lines)


def _render_flow_policy(*, initial_flow: SubagentFlowConfig, repair_flow: SubagentFlowConfig) -> str:
    initial_agents = {stage.agent for stage in initial_flow.stages}
    repair_agents = {stage.agent for stage in repair_flow.stages}
    lines = [
        "Subagent usage policy:",
        f"- Flow names: initial_codegen={initial_flow.name}; eval_failure_repair={repair_flow.name}.",
        f"- Initial codegen flow agents: {_flow_agent_list(initial_flow)}.",
        f"- Eval-failure repair flow agents: {_flow_agent_list(repair_flow)}.",
    ]
    if "bug-fixer" not in initial_agents:
        lines.append("- Do not invoke bug-fixer during initial_codegen.")
    if "bug-fixer" in repair_agents:
        lines.append("- Invoke bug-fixer during eval_failure_repair when the active repair stage requests it.")
    lines.extend(
        [
            "- Invoke exactly the subagent requested by the active stage.",
            "- Do not invoke subagents outside the active stage's configured flow.",
        ]
    )
    return "\n".join(lines) + "\n"
```

Add:

```python
def _render_flow_requirements(
    *,
    initial_flow: SubagentFlowConfig,
    repair_flow: SubagentFlowConfig,
    has_code_map: bool,
) -> str:
    initial_agents = {stage.agent for stage in initial_flow.stages}
    handoffs = sorted(_flow_required_files(initial_flow) | _flow_required_files(repair_flow))
    lines = [
        f"Required native subagent flow: {' -> '.join(stage.agent for stage in initial_flow.stages)}.",
        "Configured stage outputs:",
        _flow_stage_outputs_text(initial_flow, label="initial_codegen"),
        _flow_stage_outputs_text(repair_flow, label="eval_failure_repair"),
        f"Trusted handoff files: {', '.join(handoffs) if handoffs else '(none)'}."
    ]
    if "designer" in initial_agents:
        lines.append("The designer subagent must write the files required by its configured stage.")
    if "codegen" in initial_agents:
        lines.append("The codegen subagent must write the files required by its configured stage before handing off.")
    if "reviewer" in initial_agents:
        lines.append("The reviewer subagent must write the files required by its configured stage.")
    if "code-reader" in initial_agents and not has_code_map:
        lines.append("Use the code-reader subagent to create CODE_MAP.md before later configured stages when CODE_MAP.md is required.")
    lines.append("Every subagent final message must be short and contain only status, files_written, and next.")
    if handoffs:
        lines.append(f"Do not paste {', '.join(handoffs)} or source files into final messages.")
    return "\n".join(lines)
```

- [ ] **Step 4: Update prompt builder signature and body**

Change `AscendCAgenticPromptBuilder.build()` signature:

```python
def build(
    self,
    request: AscendCAgenticCodegenRequest,
    *,
    has_code_map: bool = False,
    task_path: str | None = None,
    flow_policy_text: str | None = None,
    initial_flow: SubagentFlowConfig | None = None,
    repair_flow: SubagentFlowConfig | None = None,
) -> str:
```

Inside `build()`, before `code_map_instruction`, resolve defaults:

```python
if initial_flow is None or repair_flow is None:
    flow_set = load_subagent_flows()
    initial_flow = initial_flow or _configured_flow_or_default(flow_set, "initial_codegen")
    try:
        repair_flow = repair_flow or flow_set.get("eval_failure_repair")
    except KeyError:
        repair_flow = repair_flow or initial_flow
```

Replace the hardcoded flow text with:

```python
flow_policy = str(flow_policy_text or "").strip() or _render_flow_policy(
    initial_flow=initial_flow,
    repair_flow=repair_flow,
).strip()
flow_requirements = _render_flow_requirements(
    initial_flow=initial_flow,
    repair_flow=repair_flow,
    has_code_map=has_code_map,
)
```

Replace `code_map_instruction` with:

```python
code_map_instruction = (
    "CODE_MAP.md already exists: yes. Read it first before configured stages that depend on it. "
    "After editing code, update affected sections of CODE_MAP.md when project contracts changed.\n"
    if has_code_map
    else "CODE_MAP.md already exists: no. If the active configured flow includes code-reader or requires CODE_MAP.md, create CODE_MAP.md before later configured stages.\n"
)
```

In the prompt string, remove hardcoded lines for `Required native subagent flow`, designer, codegen, reviewer, trusted handoff list, and hardcoded "Do not paste" list. Insert:

```python
f"{flow_policy}\n"
f"{flow_requirements}\n"
+ code_map_instruction
```

Keep the existing tools, safety, task specification, action, perf, and trace sections unchanged.

- [ ] **Step 5: Update cycle prompt call**

In `AscendCAgenticCycle._build_prompt()`, pass the actual flows:

```python
prompt = self.runner.prompt_builder.build(
    request,
    has_code_map=has_code_map,
    task_path=self._task_path_text(),
    flow_policy_text=flow_policy_text,
    initial_flow=self.runner.subagent_flow,
    repair_flow=self.runner.repair_subagent_flow,
)
```

- [ ] **Step 6: Run prompt tests to verify they pass**

Run:

```bash
python -m pytest tests/test_claude_subagent_flow_policy.py tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_omits_full_project_container_and_includes_action -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/test_claude_subagent_flow_policy.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: render agentic prompts from subagent flow config"
```

## Task 4: Runner Handoff Capture and Filtering

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Modify: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Write failing end-to-end custom-flow test**

Append to `tests/kernel_generators/test_ascendc_agentic_codegen.py`:

```python
def test_runner_custom_flow_captures_and_filters_configured_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    monkeypatch.setenv("KSEARCH_ENABLE_CURATOR", "0")
    monkeypatch.setenv("KSEARCH_RUN_ID", "custom-flow")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        artifacts_dir=str(tmp_path / "artifacts"),
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel/foo.h').read_text() == 'alpha\\nBETA\\ngamma\\n'; "
            "assert not Path('CODEGEN_NOTES.md').exists(); "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd("print('latency_ms=1.0')"),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    flow = SubagentFlowConfig(
        name="minimal-initial-codegen",
        description="minimal",
        stages=(
            SubagentStageConfig(
                name="code-reader",
                agent="code-reader",
                instruction="Write CODE_MAP.md.",
                required_files=("CODE_MAP.md",),
                run_when_missing_files=("CODE_MAP.md",),
            ),
            SubagentStageConfig(
                name="codegen",
                agent="codegen",
                instruction="Implement and write CODEGEN_NOTES.md.",
                required_files=("CODE_MAP.md", "CODEGEN_NOTES.md"),
            ),
        ),
    )

    class MinimalFlowClient:
        def __init__(self):
            self.prompts: list[str] = []

        def open_session(self, *, project_dir, telemetry_recorder=None):
            from types import SimpleNamespace

            return SimpleNamespace(_closed=False, project_dir=Path(project_dir))

        def send_prompt(self, session, *, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(session.project_dir)
            first = prompt.splitlines()[0]
            if first == "Stage 1/2: code-reader":
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h\n", encoding="utf-8")
                text = "reader done"
            elif first == "Stage 2/2: codegen":
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nkernel/foo.h updated\n", encoding="utf-8")
                (root / "CODEGEN_NOTES.md").write_text("# notes\nchanged beta\n", encoding="utf-8")
                text = "codegen done"
            else:
                raise AssertionError(first)
            return ClaudeProjectEditResult(
                text=text,
                transcript=text,
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

        def close_session(self, session):
            session._closed = True

    client = MinimalFlowClient()
    runner = AscendCAgenticCodegenRunner(
        model_name="claude",
        editor_client=client,
        subagent_flow=flow,
        repair_subagent_flow=flow,
    )
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=1,
            attempt_idx=1,
            mode="action",
            run_id="custom-flow",
        ),
        base_solution=None,
    )

    assert [prompt.splitlines()[0] for prompt in client.prompts] == [
        "Stage 1/2: code-reader",
        "Stage 2/2: codegen",
    ]
    assert result.changed_paths == ["kernel/foo.h"]
    assert "CODEGEN_NOTES.md" not in result.diff_text
    assert "CODEGEN_NOTES.md" not in {src.path for src in result.solution.sources}
    assert result.project_snapshot is not None
    assert "CODEGEN_NOTES.md" not in result.project_snapshot.manifest
    assert result.artifact_paths is not None
    manifest = json.loads(Path(result.artifact_paths["manifest_path"]).read_text(encoding="utf-8"))
    assert "CODEGEN_NOTES.md" in manifest["native_handoff_paths"]
```

Add imports at the top of the file if missing:

```python
from k_search.kernel_generators.subagent_orchestration import SubagentFlowConfig, SubagentStageConfig
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_custom_flow_captures_and_filters_configured_handoff -q
```

Expected: FAIL because `CODEGEN_NOTES.md` is not captured as a native handoff and is still treated as a changed/source file.

- [ ] **Step 3: Implement configured handoff helpers in runner**

In `k_search/kernel_generators/ascendc_agentic_codegen.py`, import `Iterable` if it is not already imported:

```python
from collections.abc import Iterable
```

Replace `_flow_handoff_files()`:

```python
def _flow_handoff_files(flow: SubagentFlowConfig | None) -> set[str]:
    if flow is None:
        return set()
    return {path for stage in flow.stages for path in stage.required_files}


def _configured_flow_handoff_files(*flows: SubagentFlowConfig | None) -> set[str]:
    out: set[str] = set()
    for flow in flows:
        out.update(_flow_handoff_files(flow))
    return out
```

Change `_require_native_handoff_files()`:

```python
def _require_native_handoff_files(
    project_dir: Path,
    required_files: set[str] | None = None,
    *,
    optional_files: Iterable[str] | None = None,
) -> dict[str, str]:
    required = set(NATIVE_HANDOFF_FILES if required_files is None else required_files)
    optional = set(NATIVE_HANDOFF_FILES) | set(optional_files or ())
    missing = [name for name in sorted(required) if not (project_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Claude native subagent flow did not produce required handoff file(s): {', '.join(missing)}")
    present_optional = {name for name in optional - required if (project_dir / name).is_file()}
    collected = required | present_optional
    handoffs = {
        name: (project_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in sorted(collected)
    }
```

Keep the existing filename-specific validator block immediately after this new `handoffs` construction.

Change `_remove_native_handoff_files()`:

```python
def _remove_native_handoff_files(project_dir: Path, *, handoff_files: Iterable[str] | None = None) -> None:
    for name in set(NATIVE_HANDOFF_FILES) | set(handoff_files or ()):
        (project_dir / name).unlink(missing_ok=True)
```

Change `_is_native_handoff_path()` and `_candidate_changed_paths()`:

```python
def _is_native_handoff_path(path: str, *, extra_runtime_files: Iterable[str] | None = None) -> bool:
    return is_native_runtime_path(path, extra_files=extra_runtime_files)


def _candidate_changed_paths(paths: list[str], *, extra_runtime_files: Iterable[str] | None = None) -> list[str]:
    return [path for path in paths if not _is_native_handoff_path(path, extra_runtime_files=extra_runtime_files)]
```

- [ ] **Step 4: Thread configured files through eval, snapshot, and solution creation**

Change `_copy_project_for_eval()`:

```python
def _copy_project_for_eval(
    candidate_dir: Path,
    *,
    extra_runtime_files: Iterable[str] | None = None,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    tmp = tempfile.TemporaryDirectory(prefix="ksearch_eval_")
    eval_dir = Path(tmp.name).resolve() / "project"
    ignore = shutil.ignore_patterns(
        ".git",
        ".claude",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "logs",
        "llm_logs",
        *NATIVE_RUNTIME_DIRS,
        *native_runtime_files(extra_runtime_files),
    )
```

Keep the existing `shutil.copytree(candidate_dir, eval_dir, symlinks=False, ignore=ignore)` and `return tmp, eval_dir` lines after the ignore block.

Import `native_runtime_files` from `runtime_artifacts`.

Change `_run_eval_in_isolated_copy()` and `AscendCAgenticCycle._run_eval()`:

```python
def _run_eval_in_isolated_copy(
    *,
    task: Any,
    candidate_project_dir: Path,
    round_num: int,
    extra_runtime_files: Iterable[str] | None = None,
) -> tuple[EvalResult, str | None]:
    tmp, eval_dir = _copy_project_for_eval(candidate_project_dir, extra_runtime_files=extra_runtime_files)


def _run_eval(self, *, extra_runtime_files: Iterable[str] | None = None) -> tuple[EvalResult, str | None]:
    assert self.wt_session is not None
    return _run_eval_in_isolated_copy(
        task=self.task,
        candidate_project_dir=self.wt_session.project_dir,
        round_num=self.request.round_num,
        extra_runtime_files=extra_runtime_files,
    )
```

Leave the existing `keep_eval_dir`, `try`, debug evidence capture, and cleanup logic unchanged after the new `tmp, eval_dir` line in `_run_eval_in_isolated_copy()`.

Add a cycle helper:

```python
def _configured_handoff_files(self) -> set[str]:
    return _configured_flow_handoff_files(
        self.runner.subagent_flow,
        self.runner.repair_subagent_flow,
    )
```

In `_finalize_attempt_result()`, compute:

```python
configured_handoff_files = self._configured_handoff_files()
required_handoff_files = _flow_handoff_files(flow)
handoff_texts = _require_native_handoff_files(
    self.wt_session.project_dir,
    required_handoff_files,
    optional_files=configured_handoff_files,
)
produced_code_map = _code_map_from_handoffs(handoff_texts)
if produced_code_map:
    self.code_map_text = produced_code_map
_remove_native_handoff_files(self.wt_session.project_dir, handoff_files=configured_handoff_files)
self.curator_context = dict(handoff_texts)
self.curator_context.update(_capture_and_remove_non_candidate_files(self.wt_session.project_dir))
project_changed_paths = self.wt_session.project_changed_paths()
changed_paths = _candidate_changed_paths(
    project_changed_paths or self.wt_session.changed_paths(),
    extra_runtime_files=configured_handoff_files,
)
eval_result, eval_project_path = self._run_eval(extra_runtime_files=configured_handoff_files)
```

In the existing `self.task.make_solution_from_project_dir` call, add this keyword argument:

```python
extra_runtime_files=configured_handoff_files,
```

In the existing `create_project_snapshot` call, add this keyword argument:

```python
extra_skip_files=configured_handoff_files,
```

Change `_native_metadata()`:

```python
def _native_metadata(*, configured_handoff_files: Iterable[str] | None = None) -> dict[str, Any]:
    configured = sorted(set(configured_handoff_files or ()))
    return {
        "native_claude_agents": True,
        "native_handoff_files": sorted(NATIVE_HANDOFF_FILES),
        "configured_handoff_files": configured,
        "native_runtime_files": sorted(native_runtime_files(configured)),
    }
```

Pass `**_native_metadata(configured_handoff_files=configured_handoff_files)` in artifact metadata.

- [ ] **Step 5: Run custom-flow test to verify it passes**

Run:

```bash
python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_custom_flow_captures_and_filters_configured_handoff -q
```

Expected: PASS.

- [ ] **Step 6: Run focused lifecycle tests**

Run:

```bash
python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_uses_configured_subagent_stages_in_one_session tests/kernel_generators/test_ascendc_agentic_codegen.py::test_run_multi_turn_uses_repair_flow_when_eval_fails tests/kernel_generators/test_ascendc_agentic_codegen.py::test_continue_fix_filters_debug_evidence_files_from_candidate_outputs -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: capture configured subagent handoff artifacts"
```

## Task 5: Claude Asset Text and README

**Files:**
- Modify: `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md`
- Modify: `k_search/kernel_generators/claude_assets/agents/codegen.md`
- Modify: `k_search/kernel_generators/claude_assets/agents/reviewer.md`
- Modify: `k_search/kernel_generators/claude_assets/agents/knowledge-curator.md`
- Modify: `README.md`
- Modify: `tests/kernel_generators/test_claude_assets.py`
- Modify: `tests/kernel_generators/test_subagent_orchestration.py`

- [ ] **Step 1: Write failing asset text tests**

Append to `tests/kernel_generators/test_claude_assets.py`:

```python
def test_ascendc_codegen_skill_defers_to_active_configured_flow():
    repo_root = Path(__file__).resolve().parents[2]
    text = (
        repo_root
        / "k_search"
        / "kernel_generators"
        / "claude_assets"
        / "skills"
        / "ascendc-codegen"
        / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert "active configured flow" in text
    assert "Use this workflow:\n1. Use code-reader" not in text


def test_codegen_agent_treats_design_as_configured_or_optional():
    repo_root = Path(__file__).resolve().parents[2]
    text = (
        repo_root
        / "k_search"
        / "kernel_generators"
        / "claude_assets"
        / "agents"
        / "codegen.md"
    ).read_text(encoding="utf-8")

    assert "If `ASCENDC_DESIGN.md` exists or the active stage prompt requires it" in text
    assert "Read these files in order:\n\n1. `ASCENDC_DESIGN.md`" not in text
```

Update `tests/kernel_generators/test_subagent_orchestration.py::test_readme_documents_configured_native_stage_order` or add:

```python
def test_readme_documents_custom_subagent_flow_config_example():
    repo_root = Path(__file__).resolve().parents[2]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")

    assert "KSEARCH_SUBAGENT_FLOW_CONFIG" in readme
    assert "CODEGEN_NOTES.md" in readme
    assert "code-reader` and `codegen" in readme
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/kernel_generators/test_claude_assets.py::test_ascendc_codegen_skill_defers_to_active_configured_flow tests/kernel_generators/test_claude_assets.py::test_codegen_agent_treats_design_as_configured_or_optional tests/kernel_generators/test_subagent_orchestration.py::test_readme_documents_custom_subagent_flow_config_example -q
```

Expected: FAIL because the asset and README text still hardcode the default flow globally.

- [ ] **Step 3: Update `ascendc-codegen` skill**

Replace the body after the H1 in `k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md` with:

```markdown
You are working inside a K-Search candidate worktree.

Follow the active configured flow from the parent prompt and the current stage
prompt. The flow, stage order, active subagent, and required handoff files are
provided by K-Search at runtime.

For every stage:
- invoke only the subagent requested by the stage prompt
- write the files required by the active stage prompt
- read configured handoff files that exist before depending on their contents
- keep final messages concise: status, files_written, next

Cross-agent state must be written to the configured handoff files, not pasted
into final messages.

Do not run Bash.
Do not edit outside the current working directory.
Preserve public entry points, host tiling contract, kernel launch behavior,
correctness harness behavior, and build layout.
```

- [ ] **Step 4: Update `codegen.md` input and output language**

In `k_search/kernel_generators/claude_assets/agents/codegen.md`, replace the opening paragraph and input contract with:

```markdown
You are the K-Search AscendC codegen subagent. You implement the active stage
inside the current candidate project. The parent stage prompt is the source of
truth for required handoff files.

You do not run Bash. Evaluation is performed by the framework after the flow.

## Input Contract

Read the active stage prompt first. It tells you which files must be produced.

Then read:

1. `CODE_MAP.md` if it exists or the active stage prompt requires it.
2. `ASCENDC_DESIGN.md` if it exists or the active stage prompt requires it.
3. `KNOWLEDGE.md` if present, to apply distilled patterns and avoid known pitfalls.
4. The real source files referenced by the prompt, design, map, or strategy text,
   using Glob/Grep/Read before any Edit.

If `ASCENDC_DESIGN.md` exists or the active stage prompt requires it, use it as
the detailed design target. If it is not present and not required, derive the
implementation plan from the base attempt context, strategy/action text, CODE_MAP
when present, KNOWLEDGE when present, and real source inspection.
```

Replace the paragraph that starts with `Edit only files inside the current project directory.` with:

```markdown
Edit only files inside the current project directory. Do not edit `.git`,
`.claude`, build directories, caches, logs, generated artifacts, or debug
evidence files. Native handoff files may be edited only when the active stage
prompt requires them or explicitly instructs you to update them.
```

Change "Step 1" heading to:

```markdown
## Step 1: Write Configured Planning Handoffs Before Editing Source

If the active stage prompt requires `IMPLEMENTATION_EXECUTION_PLAN.md`, write it
before editing any source file.
```

Change final message contract files line:

```markdown
- files_written: list of modified source files and every handoff file required by the active stage prompt
```

- [ ] **Step 5: Update reviewer and curator text**

In `k_search/kernel_generators/claude_assets/agents/reviewer.md`, replace the "Use the active flow" bullet block with:

```markdown
Use the active flow from the stage prompt. Read every configured handoff file
that the reviewer stage requires. Also read default implementation handoff files
such as `ASCENDC_DESIGN.md`, `IMPLEMENTATION_EXECUTION_PLAN.md`,
`IMPLEMENTATION_HANDOFF.md`, and `IMPLEMENTATION_DEVIATIONS.md` when they exist.
Do not fail solely because a default handoff file is absent from a custom flow
that did not require it.
```

In `k_search/kernel_generators/claude_assets/agents/knowledge-curator.md`, replace the input list entries for `REVIEW_NOTES.md` with:

```markdown
- Configured handoff files captured by the framework, if present. `REVIEW_NOTES.md`
  is useful when a reviewer stage exists, but it may be absent in minimal custom
  flows.
```

- [ ] **Step 6: Update README**

In `README.md`, after the `KSEARCH_SUBAGENT_FLOW_CONFIG` environment variable table row or immediately after the table, add:

```markdown
`KSEARCH_SUBAGENT_FLOW_CONFIG` can point to a custom flow JSON. For example, a
minimal flow can keep only `code-reader` and `codegen`; any files listed in
stage `required_files` are treated as native handoff artifacts and are excluded
from candidate source import, eval copies, and snapshots:

```json
{
  "version": 1,
  "name": "minimal-code-reader-codegen",
  "description": "Minimal custom flow.",
  "default_flow": "initial_codegen",
  "flows": {
    "initial_codegen": {
      "name": "minimal-initial-codegen",
      "description": "Read code, then implement.",
      "stages": [
        {
          "name": "code-reader",
          "agent": "code-reader",
          "instruction": "Read the candidate project and write CODE_MAP.md. Do not edit source files.",
          "required_files": ["CODE_MAP.md"],
          "run_when_missing_files": ["CODE_MAP.md"]
        },
        {
          "name": "codegen",
          "agent": "codegen",
          "instruction": "Read CODE_MAP.md, the attempt context, and the real source files. Implement the requested change and write CODEGEN_NOTES.md.",
          "required_files": ["CODE_MAP.md", "CODEGEN_NOTES.md"]
        }
      ]
    }
  }
}
```
```

- [ ] **Step 7: Run asset and README tests**

Run:

```bash
python -m pytest tests/kernel_generators/test_claude_assets.py::test_ascendc_codegen_skill_defers_to_active_configured_flow tests/kernel_generators/test_claude_assets.py::test_codegen_agent_treats_design_as_configured_or_optional tests/kernel_generators/test_subagent_orchestration.py::test_readme_documents_custom_subagent_flow_config_example -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add k_search/kernel_generators/claude_assets/skills/ascendc-codegen/SKILL.md k_search/kernel_generators/claude_assets/agents/codegen.md k_search/kernel_generators/claude_assets/agents/reviewer.md k_search/kernel_generators/claude_assets/agents/knowledge-curator.md README.md tests/kernel_generators/test_claude_assets.py tests/kernel_generators/test_subagent_orchestration.py
git commit -m "docs: make Claude assets follow configured subagent flow"
```

## Task 6: Full Regression

**Files:**
- No new files.

- [ ] **Step 1: Run focused custom-flow and native subagent tests**

Run:

```bash
python -m pytest tests/test_claude_subagent_flow_policy.py tests/kernel_generators/test_subagent_orchestration.py tests/kernel_generators/test_ascendc_agentic_codegen.py -q
```

Expected: PASS.

- [ ] **Step 2: Run artifact, snapshot, and task tests**

Run:

```bash
python -m pytest tests/kernel_generators/test_runtime_artifacts.py tests/kernel_generators/test_project_snapshot.py tests/test_ascendc_task.py -q
```

Expected: PASS.

- [ ] **Step 3: Run SDK mock and asset tests**

Run:

```bash
python -m pytest tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_claude_assets.py tests/test_launcher_docs.py -q
```

Expected: PASS.

- [ ] **Step 4: Run whitespace and status checks**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. `git status --short` shows only intentional tracked changes before the final commit, plus any pre-existing unrelated user changes that were present before implementation.
