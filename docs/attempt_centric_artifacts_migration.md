# Attempt-centric artifact layout migration

New K-Search runs write run-level logs and artifacts directly under:

```text
<artifacts_dir>/<task_name>/<task_id>/runs/<run_id>/
```

Key locations:

- `summary.md`, `events.jsonl`, `run_meta.json`: run-level summary, event stream, and status metadata.
- `input_args.json`, `task_config.json`, `env.json`: reproducibility metadata with secret redaction.
- `world_model/`: `world_model.json`, `strategy_state.json`, `blocked_actions.jsonl`, and `solution_db.jsonl`.
- `attempts/r0001_a01_s1/`: prompt, stage prompts, telemetry trace, timeline, cost, diff, eval, handoff files, and snapshot tarball for a single attempt.
- `checkpoints/checkpoint_index.json`: queryable checkpoint summary index; `latest.json` remains the latest pointer.

Breaking changes for new runs:

- New runs no longer write attempt artifacts to `artifacts/candidates/`.
- New runs no longer write telemetry to `logs/telemetry/` by default.
- New runs no longer write world-model state to `artifacts/world_model/`.
- Historical run directories are not migrated automatically.

Migration guidance:

- Use `k_search.utils.paths.get_attempt_dir()`, `get_run_world_model_dir()`, and `get_run_checkpoints_dir()` instead of hand-built paths.
- Prefer `runs/<run_id>/artifact_index.json` when available for navigation, or read `attempts/*/manifest.json` directly.
- Use `checkpoints/checkpoint_index.json` for checkpoint lookup by round, attempt, stage, or eval status.
