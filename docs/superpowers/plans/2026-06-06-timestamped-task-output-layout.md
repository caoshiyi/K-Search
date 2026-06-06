# Timestamped Task Output Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a task-level timestamp directory so each task invocation has isolated task-scoped artifacts and run-scoped outputs.

**Architecture:** Extend `k_search.utils.paths` with a task identity layer (`KSEARCH_TASK_ID` / `KSEARCH_TASK_START`) between the sanitized task name and run directories. Keep existing helper names working by routing them through the new task directory. Move cross-run persistent artifacts to task-scoped artifacts and keep per-run candidates/logs/worktrees under the run directory.

**Tech Stack:** Python 3, pathlib, pytest, existing K-Search path helpers and task/generator classes.

---

## File Structure

- Modify `k_search/utils/paths.py`: own all task id, task dir, run dir, artifacts, logs, and worktree path construction.
- Modify `generate_kernels_and_eval.py`: pin `KSEARCH_TASK_ID`, write `task_meta.json`, update help text.
- Modify `k_search/kernel_generators/memory/memory_store.py`: read/write task-scoped memory through `include_run=False`.
- Modify `k_search/kernel_generators/kernel_generator_world_model.py`: keep run snapshots under run artifacts and top-level snapshots under task artifacts.
- Review `k_search/kernel_generators/ascendc_agentic_codegen.py`: it should keep using `get_ksearch_worktrees_dir`, which will inherit the task timestamp automatically.
- Modify tests in `tests/utils/test_paths.py`, `tests/test_generate_kernels_cli.py`, `tests/kernel_generators/test_memory_store.py`, `tests/kernel_generators/test_ascendc_agentic_codegen.py`, and related world-model tests if path expectations fail.

## Task 1: Add Task Identity Path Helpers

**Files:**
- Modify: `k_search/utils/paths.py`
- Test: `tests/utils/test_paths.py`

- [ ] **Step 1: Write failing tests for task id and timestamped layout**

Add these imports to `tests/utils/test_paths.py`:

```python
from k_search.utils.paths import (
    get_ksearch_task_artifacts_dir,
    get_ksearch_task_dir,
    get_ksearch_run_dir,
    get_ksearch_artifacts_dir,
    get_ksearch_worktrees_dir,
    get_run_id,
    get_run_logs_dir,
    get_task_id,
    resolve_output_base,
    safe_path_component,
)
```

Add tests:

```python
def test_get_task_id_prefers_task_id_over_task_start(monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-1")
    monkeypatch.setenv("KSEARCH_TASK_START", "task-2")
    assert get_task_id() == "task-1"


def test_get_task_id_falls_back_to_task_start(monkeypatch):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_START", "20260606_101112")
    assert get_task_id() == "20260606_101112"


def test_get_task_id_falls_back_to_local_timestamp(monkeypatch):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.delenv("KSEARCH_TASK_START", raising=False)
    result = get_task_id()
    assert len(result) == 15
    assert result[8] == "_"


def test_task_dir_uses_task_timestamp_between_task_and_runs(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_ARTIFACTS_DIR", raising=False)
    monkeypatch.setenv("KSEARCH_TASK_ID", "task:alpha")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:1")

    task_root = get_ksearch_task_dir(base_dir=tmp_path, task_name="vec/add")
    run_root = get_ksearch_run_dir(base_dir=tmp_path, task_name="vec/add")

    assert task_root == (tmp_path / "vec_add" / "task_alpha").resolve()
    assert run_root == task_root / "runs" / "run_1"
    assert get_ksearch_task_artifacts_dir(base_dir=tmp_path, task_name="vec/add") == task_root / "artifacts"
    assert get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "artifacts"
    assert get_ksearch_artifacts_dir(base_dir=tmp_path, task_name="vec/add", include_run=False) == task_root / "artifacts"
    assert get_run_logs_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "logs"
    assert get_ksearch_worktrees_dir(base_dir=tmp_path, task_name="vec/add") == run_root / "worktrees"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. pytest tests/utils/test_paths.py -q
```

Expected: FAIL with import errors for `get_task_id`, `get_ksearch_task_dir`, and `get_ksearch_task_artifacts_dir`, or layout assertion failures.

- [ ] **Step 3: Implement task identity helpers**

Update `k_search/utils/paths.py` with this structure:

```python
def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_task_id() -> str:
    """Task-level identifier shared by all runs inside one task invocation.

    Priority: KSEARCH_TASK_ID > KSEARCH_TASK_START > local wall-clock timestamp.
    """
    for name in ("KSEARCH_TASK_ID", "KSEARCH_TASK_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return safe_path_component(raw, default="task")
    return _timestamp_id()


def get_run_id() -> str:
    """Unified run identifier shared by llm logs, telemetry and the narrative log.

    Priority: KSEARCH_RUN_ID > KSEARCH_RUN_START > local wall-clock timestamp.
    """
    for name in ("KSEARCH_RUN_ID", "KSEARCH_RUN_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return safe_path_component(raw, default="run")
    return _timestamp_id()


def get_ksearch_task_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Path:
    base = resolve_output_base(base_dir)
    task = safe_path_component(task_name, default="__unknown__")
    tid = safe_path_component(task_id or get_task_id(), default="task")
    return base / task / tid


def get_ksearch_task_artifacts_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Path:
    return get_ksearch_task_dir(base_dir=base_dir, task_name=task_name, task_id=task_id) / "artifacts"
```

Change `get_ksearch_artifacts_dir`, `get_ksearch_run_dir`, `get_run_logs_dir`, and `get_ksearch_worktrees_dir` to call the new task helpers:

```python
def get_ksearch_artifacts_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    run_id: Optional[str] = None,
    include_run: bool = True,
) -> Path:
    if include_run:
        return get_ksearch_run_dir(base_dir=base_dir, task_name=task_name, run_id=run_id) / "artifacts"
    return get_ksearch_task_artifacts_dir(base_dir=base_dir, task_name=task_name)


def get_ksearch_run_dir(
    *,
    base_dir: Optional[PathLike] = None,
    task_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    rid = safe_path_component(run_id or get_run_id(), default="run")
    return get_ksearch_task_dir(base_dir=base_dir, task_name=task_name) / "runs" / rid
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=. pytest tests/utils/test_paths.py -q
```

Expected: PASS.

## Task 2: Pin Task Id and Write Task Metadata

**Files:**
- Modify: `generate_kernels_and_eval.py`
- Test: `tests/test_generate_kernels_cli.py`

- [ ] **Step 1: Write failing CLI/task metadata tests**

In `tests/test_generate_kernels_cli.py`, import `main` if needed:

```python
from generate_kernels_and_eval import (
    _parse_strategy_form,
    _build_task_from_args,
    _resolve_llm_config_from_args,
    generate_and_evaluate,
    main,
)
```

Add assertions to `test_generate_and_evaluate_sets_task_run_id_unconditionally`:

```python
    task_meta_path = tmp_path / "artifacts" / "lineage_task" / "task-alpha" / "task_meta.json"
    run_meta_path = task_meta_path.parent / "runs" / "run-meta" / "run_meta.json"
    assert json.loads(task_meta_path.read_text(encoding="utf-8"))["task_id"] == "task-alpha"
    assert json.loads(run_meta_path.read_text(encoding="utf-8"))["run_id"] == "run-meta"
```

Set `KSEARCH_TASK_ID` in that test before calling `generate_and_evaluate`:

```python
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-alpha")
```

Add a CLI pinning test:

```python
def test_main_pins_ksearch_task_id(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_TASK_ID", raising=False)
    monkeypatch.delenv("KSEARCH_TASK_START", raising=False)
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-main")

    class FakeTask:
        name = "pin_task"

        def get_config_for_logging(self):
            return {}

    seen = {}

    def fake_build_task(args):
        seen["artifacts_dir"] = args.artifacts_dir
        return FakeTask()

    def fake_generate_and_evaluate(**kwargs):
        seen["task_id"] = __import__("os").environ.get("KSEARCH_TASK_ID")

    monkeypatch.setattr("generate_kernels_and_eval._resolve_llm_config_from_args", lambda args: ("claude-agent", None))
    monkeypatch.setattr("generate_kernels_and_eval._build_task_from_args", fake_build_task)
    monkeypatch.setattr("generate_kernels_and_eval.generate_and_evaluate", fake_generate_and_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_kernels_and_eval.py",
            "--model-name",
            "fake",
            "--artifacts-dir",
            str(tmp_path / "out"),
        ],
    )

    main()

    assert seen["task_id"]
    assert len(seen["task_id"]) == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=. pytest tests/test_generate_kernels_cli.py -q
```

Expected: FAIL because `task_meta.json` is not written and `main()` does not pin `KSEARCH_TASK_ID`.

- [ ] **Step 3: Implement task id pinning and metadata**

In `generate_kernels_and_eval.py`, import and use `get_ksearch_task_dir`, `get_task_id`, and `resolve_output_base`.

At CLI startup, pin:

```python
from k_search.utils.paths import get_run_id, get_task_id, resolve_output_base

os.environ.setdefault(
    "KSEARCH_ARTIFACTS_DIR", str(resolve_output_base(getattr(args, "artifacts_dir", None)))
)
os.environ.setdefault("KSEARCH_TASK_ID", get_task_id())
os.environ.setdefault("KSEARCH_RUN_ID", get_run_id())
```

In `generate_and_evaluate`, write `task_meta.json` before `run_meta.json`:

```python
from k_search.utils.paths import get_ksearch_run_dir, get_ksearch_task_dir, get_run_id, get_task_id

effective_task_id = get_task_id()
task_root = get_ksearch_task_dir(
    base_dir=artifacts_dir,
    task_name=task_name,
    task_id=effective_task_id,
)
task_meta_path = task_root / "task_meta.json"
task_meta = {
    "task_id": effective_task_id,
    "task_name": task_name,
    "start_time": datetime.utcnow().isoformat() + "Z",
    "artifacts_dir": artifacts_dir,
}
task_meta_path.parent.mkdir(parents=True, exist_ok=True)
task_meta_path.write_text(json.dumps(task_meta, indent=2, sort_keys=True), encoding="utf-8")
```

Keep `run_meta.json` under `get_ksearch_run_dir(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
PYTHONPATH=. pytest tests/test_generate_kernels_cli.py -q
```

Expected: PASS.

## Task 3: Move Cross-Run Persistent Artifacts to Task Artifacts

**Files:**
- Modify: `k_search/kernel_generators/memory/memory_store.py`
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`
- Test: `tests/kernel_generators/test_memory_store.py`
- Test: existing world model generator tests touched by layout assertions

- [ ] **Step 1: Write failing memory layout test**

In `tests/kernel_generators/test_memory_store.py`, add:

```python
def test_memory_store_uses_task_scoped_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TASK_ID", "task-one")
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-one")
    store = MemoryStore(artifacts_dir=str(tmp_path / "out"), task_name="opx")

    store.save(CODE_MAP, "# CODE_MAP\nshared\n")

    task_path = tmp_path / "out" / "opx" / "task-one" / "artifacts" / "memory" / "code_map" / "CODE_MAP.md"
    run_path = tmp_path / "out" / "opx" / "task-one" / "runs" / "run-one" / "artifacts" / "memory" / "code_map" / "CODE_MAP.md"
    assert task_path.read_text(encoding="utf-8") == "# CODE_MAP\nshared\n"
    assert not run_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=. pytest tests/kernel_generators/test_memory_store.py -q
```

Expected: FAIL because memory currently uses run-scoped artifacts.

- [ ] **Step 3: Implement task-scoped memory**

Change `MemoryStore._path()`:

```python
def _path(self, kind: MemoryKind) -> Path:
    base = get_ksearch_artifacts_dir(
        base_dir=self._artifacts_dir,
        task_name=self._task_name,
        include_run=False,
    )
    return base / "memory" / kind.name / kind.filename
```

Review `WorldModelKernelGeneratorWithBaseline._default_world_model_path()`. It already passes `include_run=False` for top-level snapshots; after Task 1, that path becomes task-scoped automatically. Keep the current run snapshot path unchanged.

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest tests/kernel_generators/test_memory_store.py tests/kernel_generators/test_ascendc_agentic_codegen.py -q
```

Expected: PASS after updating any hard-coded path expectations to include the task id.

## Task 4: Verify Agentic Worktrees and Telemetry Under Timestamped Runs

**Files:**
- Modify tests only unless failures show production code gaps:
  `tests/kernel_generators/test_agentic_worktree.py`,
  `tests/kernel_generators/test_ascendc_agentic_codegen.py`,
  `tests/telemetry/test_recorder_sinks.py`,
  `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Update path expectations for timestamped run roots**

In tests that assert paths contain:

```text
/runs/<run_id>/artifacts/
```

update them to assert the task id segment:

```text
/<task>/<task_id>/runs/<run_id>/artifacts/
```

For example, in `tests/kernel_generators/test_ascendc_agentic_codegen.py`, set:

```python
monkeypatch.setenv("KSEARCH_TASK_ID", "task-artifact")
```

and assert:

```python
assert str(tmp_path / "artifacts" / "x" / "task-artifact" / "runs" / "artifact-run" / "worktrees") in result.project_path
assert str(tmp_path / "artifacts" / "x" / "task-artifact" / "runs" / "artifact-run" / "artifacts" / "candidates") in result.artifact_paths["manifest_path"]
```

- [ ] **Step 2: Run focused tests to expose gaps**

Run:

```bash
PYTHONPATH=. pytest tests/utils/test_paths.py tests/telemetry/test_recorder_sinks.py tests/kernel_generators/test_agentic_worktree.py tests/kernel_generators/test_ascendc_agentic_codegen.py tests/kernel_generators/test_llm_clients.py -q
```

Expected: PASS after test updates. If telemetry defaults still omit `task_id`, fix `build_attempt_dir()` by using `get_run_logs_dir(task_name=task_name, run_id=run_id)`; after Task 1 that helper includes the task id automatically.

- [ ] **Step 3: Run full suite**

Run:

```bash
PYTHONPATH=. pytest -q
```

Expected: PASS. Existing warnings about deprecated strategy inline mode and `datetime.utcnow()` are acceptable if the warning count matches the current baseline.

## Task 5: Commit Implementation

**Files:**
- Stage all modified production and test files from Tasks 1-4.

- [ ] **Step 1: Check status**

Run:

```bash
git status --short
```

Expected: only files touched by this plan are modified.

- [ ] **Step 2: Commit**

Run:

```bash
git add k_search/utils/paths.py generate_kernels_and_eval.py k_search/kernel_generators/memory/memory_store.py k_search/kernel_generators/kernel_generator_world_model.py tests/utils/test_paths.py tests/test_generate_kernels_cli.py tests/kernel_generators/test_memory_store.py tests/kernel_generators/test_ascendc_agentic_codegen.py tests/telemetry/test_recorder_sinks.py tests/kernel_generators/test_llm_clients.py
git commit -m "feat: add timestamped task output roots"
```

Expected: commit succeeds.

## Self-Review

Spec coverage:

- Timestamped task directory: Task 1.
- Task identity env vars and CLI pinning: Task 2.
- Cross-run persistent artifacts under task artifacts: Task 3.
- Run-scoped artifacts/logs/worktrees under timestamped task runs: Tasks 1 and 4.
- Tests for focused and full verification: Tasks 1-4.

Placeholder scan: no placeholder tokens or open-ended implementation steps remain.

Type consistency: helper names are consistent across tasks: `get_task_id`, `get_ksearch_task_dir`, `get_ksearch_task_artifacts_dir`, `get_ksearch_run_dir`, `get_ksearch_artifacts_dir`, `get_run_logs_dir`, and `get_ksearch_worktrees_dir`.
