# Custom Subagent Flow Design

> Date: 2026-06-07
> Scope: AscendC Claude native subagent orchestration

## Background

K-Search already has a configurable staged Claude native subagent driver:

- `k_search/kernel_generators/claude_assets/subagent_flow.json` defines named flows.
- `KSEARCH_SUBAGENT_FLOW_CONFIG` can point to an external JSON flow config.
- `SubagentFlowConfig` stages define `agent`, `instruction`, `required_files`,
  `run_when_missing_files`, and `include_base_prompt`.
- `run_configured_subagent_flow()` executes active stages one-by-one inside one
  Claude SDK session and validates stage output files.

The remaining problem is that several code paths still assume the built-in
four-stage flow:

```text
code-reader -> designer -> codegen -> reviewer
```

Those assumptions appear in the main prompt, flow policy text, artifact handling,
tests, README text, and Claude asset instructions. A custom flow that keeps only
`code-reader` and `codegen` should not be asked to create `ASCENDC_DESIGN.md` or
`REVIEW_NOTES.md` unless those files are declared by that custom flow.

## Goals

1. Support custom subagent flows through `KSEARCH_SUBAGENT_FLOW_CONFIG` only.
2. Let flow stage order and handoff artifact names come from the active flow
   config, not from hardcoded default-stage assumptions.
3. Keep the built-in default flow behavior unchanged.
4. Allow a minimal flow such as `code-reader -> codegen`.
5. Prevent configured handoff artifacts from leaking into candidate solutions,
   eval copies, or project snapshots.
6. Keep semantic validation for known files such as `REVIEW_NOTES.md`, while
   allowing new configured handoff filenames.

## Non-Goals

- No new CLI flags for flow config or flow selection.
- No new schema for arbitrary runtime file groups beyond stage `required_files`.
- No removal or renaming of built-in Claude agents or skills.
- No change to non-AscendC or non-Claude-agent codegen paths.
- No change to the Claude SDK permission model.

## Design

### 1. Configuration Entry

`KSEARCH_SUBAGENT_FLOW_CONFIG` remains the single user-facing entrypoint.

`load_subagent_flows()` continues to support:

- a single-flow JSON object
- a multi-flow JSON object with `flows`
- package default config when the env var is unset

`AscendCAgenticCodegenRunner` keeps the current flow selection behavior:

- `initial_codegen` is used when present.
- If `eval_failure_repair` is missing, repair falls back to the initial flow.

No command-line option is added.

### 2. Flow-Derived Prompt Rendering

`AscendCAgenticPromptBuilder` should receive the actual initial and repair flow
configs, or a pre-rendered object derived from them, instead of relying on
default-flow text.

The prompt should render these sections from the active config:

- `Required native subagent flow`: `stage.agent` joined by ` -> `
- `Configured stage outputs`: one line per stage with its `required_files`
- `Trusted handoff files`: unique configured handoff files from the initial and
  repair flows
- `Subagent usage policy`: flow names and configured agent lists

Agent-specific instructions must be conditional:

- If `designer` is not in the flow, do not say that designer must write
  `ASCENDC_DESIGN.md`.
- If `reviewer` is not in the flow, do not say that reviewer must write
  `REVIEW_NOTES.md`.
- If `bug-fixer` is not in the repair flow, do not instruct the model to invoke
  `bug-fixer`.
- If `code-reader` is present and `CODE_MAP.md` is missing, keep the code-map
  guidance, but phrase it as "before later configured stages" rather than
  "before detailed design."

The default package flow should still render the same effective requirements as
today:

```text
code-reader -> designer -> codegen -> reviewer
```

The two-stage custom flow should render as:

```text
code-reader -> codegen
```

and should not mention missing default-only stages or files unless the custom
JSON declares them.

### 3. Configured Handoff Artifact Names

Stage `required_files` is the authoritative source for flow-specific handoff
artifact names.

For a custom flow, Python must use configured `required_files` to:

- render stage prompts
- validate that each stage produced its required outputs
- validate flow-level required handoffs after the flow completes
- capture handoff files into `handoff_files`
- pass captured handoffs to the curator context
- remove captured handoffs from the candidate worktree
- exclude those files from eval copies, snapshots, and imported candidate sources

`NATIVE_HANDOFF_FILES` remains useful as the built-in default set and as the
source of known semantic validators, but it must not be the only recognized
handoff set for custom flows.

Known semantic validators remain filename-based:

- `CODE_MAP.md`: optional length/usefulness validation
- `ASCENDC_DESIGN.md`: optional design length validation
- `IMPLEMENTATION_EXECUTION_PLAN.md`: optional execution-plan validation
- `IMPLEMENTATION_HANDOFF.md`: optional handoff validation
- `REVIEW_NOTES.md`: `status` and `eval_ready` validation

Unknown configured handoff files are validated for existence, captured, cleaned,
and excluded, but do not receive extra semantic validation.

### 4. Runtime Artifact Filtering

Runtime filtering should be based on:

- built-in native runtime files
- configured required files from the active initial flow
- configured required files from the active repair flow
- debug evidence files
- memory files such as `KNOWLEDGE.md`
- native runtime directories such as `.claude` and `.ksearch`

This prevents a custom required file such as `CODEGEN_NOTES.md` from becoming a
candidate source, appearing in project snapshots, or being copied into eval
workdirs.

Implementation should avoid global mutable state for custom runtime file names.
Prefer passing an `extra_runtime_files` or `handoff_files` set through helper
calls where candidate filtering, eval copying, snapshot creation, and cleanup
need the active flow context.

### 5. Flow Completion and Cleanup

`_flow_handoff_files(flow)` should return the exact configured required file
names for the flow, not only names that appear in `NATIVE_HANDOFF_FILES`.

`_require_native_handoff_files(project_dir, required_files=...)` should:

- require exactly the configured required files
- include any present built-in handoff files as optional captured context
- include any present configured handoff files from the active flow
- run semantic validators only for known filenames

Cleanup should remove:

- built-in native handoff files
- configured handoff files from the active flow
- configured handoff files from the repair flow when relevant
- debug evidence and memory files where the current helper already owns them

### 6. Claude Asset Text

The materializer can keep copying all built-in agents and skills. Execution is
controlled by the configured flow, not by which asset files exist.

The asset instructions should stop treating the default four-stage flow as a
global rule.

`ascendc-codegen` skill:

- say to follow the active configured flow from the parent/stage prompt
- say to write files required by the active stage prompt
- keep worktree, Bash, and contract-preservation constraints

`codegen.md`:

- do not require `ASCENDC_DESIGN.md` unconditionally
- read `ASCENDC_DESIGN.md` if present or required by the stage prompt
- otherwise use base attempt context, strategy/action text, `CODE_MAP.md` if
  present, `KNOWLEDGE.md` if present, and real source files
- write the files required by the stage prompt
- keep the default execution-plan and implementation-handoff discipline when
  those files are required by the flow

`reviewer.md`:

- read configured handoff files when present or required
- do not fail solely because default initial handoff files are absent in a custom
  flow that does not require them

`knowledge-curator.md`:

- read what exists and record gaps
- do not assume `REVIEW_NOTES.md` always exists
- do not edit native runtime or configured handoff files

### 7. Example Two-Stage Flow

An external flow file can keep only `code-reader` and `codegen`:

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

With this config:

- prompt flow order is `code-reader -> codegen`
- `ASCENDC_DESIGN.md` and `REVIEW_NOTES.md` are not required
- `CODEGEN_NOTES.md` is required, captured, cleaned, and excluded from candidate
  source import
- default code-map memory behavior remains intact because `CODE_MAP.md` is still
  configured

## Error Handling

- Missing configured required files fail the current stage or flow.
- Unknown configured handoff filenames are allowed, but they are treated as
  handoff artifacts, not candidate source files.
- Known handoff semantic validation remains best-effort unless strict validation
  is enabled, matching current behavior.
- If the active flow has no active stages after `run_when_missing_files` checks,
  the existing "no active stages" failure remains.

## Tests

Add or update focused tests for:

1. Prompt rendering with default flow still mentions the default stages and files.
2. Prompt rendering with a custom two-stage flow omits designer/reviewer and
   default-only files.
3. A custom required file such as `CODEGEN_NOTES.md` is required, captured into
   handoff artifacts, removed from the worktree, excluded from solution sources,
   and excluded from snapshots/eval copies.
4. Known validators still run only when their known file is required.
5. README documents `KSEARCH_SUBAGENT_FLOW_CONFIG` with a custom-flow example.
6. Claude asset text no longer requires the default four-stage flow globally.

## Risks

- Passing active flow-specific runtime file sets through all artifact helpers may
  touch several call sites. Keep helper signatures narrow and default to current
  behavior when no custom set is passed.
- Claude asset wording must remain strong enough for the default flow. The stage
  prompt should carry exact required outputs so asset text can be generic without
  weakening enforcement.
- If a custom flow omits `REVIEW_NOTES.md`, Python cannot use reviewer
  `eval_ready` as a gate. That is acceptable because the configured flow is the
  contract, and Python evaluation still remains the final correctness gate.
