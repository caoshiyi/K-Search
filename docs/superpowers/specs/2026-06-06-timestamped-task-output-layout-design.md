# Timestamped Task Output Layout Design

## Goal

Each K-Search task invocation gets an isolated timestamped task directory, while
all outputs for that invocation remain easy to discover from one root.

## Target Layout

The output tree should be:

```text
<base>/<task>/<task_ts>/
  task_meta.json
  artifacts/
  runs/
    <run_id>/
      run_meta.json
      artifacts/
      logs/
      worktrees/
```

`<base>` is still resolved from explicit `--artifacts-dir`, then
`KSEARCH_ARTIFACTS_DIR`, then the current working directory's `.ksearch`.

`<task>` is the sanitized task or definition name.

`<task_ts>` is a task-level timestamp or explicit task id. It is stable for the
whole process and defaults to local wall-clock `YYYYMMDD_HHMMSS`.

`<run_id>` remains the run identifier. Multiple runs can live under the same
timestamped task directory when a caller intentionally reuses the same task id.

## Identity Rules

Add a task-level identity helper, separate from run identity:

- `get_task_id()` prefers `KSEARCH_TASK_ID`, then `KSEARCH_TASK_START`, then a
  local timestamp.
- CLI startup pins `KSEARCH_TASK_ID` once, just like it already pins
  `KSEARCH_RUN_ID`.
- The task id is sanitized with the same path component rules as task names and
  run ids.

This keeps run ids free to describe individual optimization/eval runs, while
task ids describe the larger task invocation directory.

## Path Semantics

Path helpers should centralize all layout decisions:

- `get_ksearch_task_dir(...)` returns `<base>/<task>/<task_id>`.
- `get_ksearch_task_artifacts_dir(...)` returns
  `<base>/<task>/<task_id>/artifacts`.
- `get_ksearch_run_dir(...)` returns
  `<base>/<task>/<task_id>/runs/<run_id>`.
- `get_ksearch_artifacts_dir(..., include_run=True)` returns
  `<run_dir>/artifacts`.
- `get_ksearch_artifacts_dir(..., include_run=False)` returns
  `<task_dir>/artifacts`.
- `get_run_logs_dir(...)` and `get_ksearch_worktrees_dir(...)` remain siblings
  under the run directory.

The public helper names that already exist should keep working; callers should
not hand-build paths.

## Cross-Run Persistent Artifacts

Cross-run persistent artifacts belong to the timestamped task artifacts root:

```text
<base>/<task>/<task_ts>/artifacts/
  world_model/world_model.json
  memory/code_map/CODE_MAP.md
  memory/knowledge/KNOWLEDGE.md
```

Run-scoped artifacts remain under:

```text
<base>/<task>/<task_ts>/runs/<run_id>/artifacts/
```

The world model snapshot should still be written to both the current run and
the task-level artifacts root. `--continue-from-world-model auto` should search
the current run first, then the timestamped task-level artifacts root.

Memory should intentionally become task-scoped instead of run-scoped so a later
run inside the same timestamped task directory can reuse adopted CODE_MAP and
KNOWLEDGE.

## Backward Compatibility

This change is a layout migration for new runs. Existing absolute paths still
work when passed explicitly, because loaders already accept direct JSON paths.

Name-based lookup through K-Search artifacts should use the new timestamped
task directory. It does not need to scan legacy directories automatically.

## Tests

Focused tests should verify:

- path helpers produce `<base>/<task>/<task_ts>/runs/<run_id>/{artifacts,logs,worktrees}`;
- `include_run=False` returns the task-level artifacts root;
- CLI startup pins `KSEARCH_TASK_ID`;
- world model top-level snapshots land under timestamped task artifacts;
- MemoryStore reads/writes task-scoped memory;
- AscendC agentic worktrees still land under the run's `worktrees/` directory.
