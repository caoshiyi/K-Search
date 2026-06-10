# Checkpoint V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add K-Search V3 stage-boundary checkpoints for AscendC Claude native subagent flows.

**Architecture:** Introduce `checkpoint_v3.py` as the stage checkpoint schema, persistence, restore, and CLI validation boundary. Wire it into `run_configured_subagent_flow()` so each stage can save `stage_start` before Claude execution and `stage_completed` after required-file validation. Keep K-Search `ProjectSnapshot` as the authoritative file state; Claude session metadata remains optional metadata.

**Tech Stack:** Python dataclasses, pathlib/json/shutil atomic writes, existing `ProjectSnapshot`, pytest.

---

### Task 1: Stage Checkpoint Manager

**Files:**
- Create: `k_search/kernel_generators/checkpoint_v3.py`
- Test: `tests/test_checkpoint_v3_stage_state.py`

- [ ] **Step 1: Write failing tests**

Add tests that create a fake AscendC project, save `stage_start`, save `stage_completed`, verify `latest.json`, `manifest.json`, `runtime_state.json`, `stage_state.json`, prompt/result artifacts, Claude metadata, and ProjectSnapshot payloads.

Run: `pytest tests/test_checkpoint_v3_stage_state.py -q`
Expected: FAIL because `k_search.kernel_generators.checkpoint_v3` does not exist.

- [ ] **Step 2: Implement minimal manager**

Implement `StageCheckpointConfig`, `StageRecord`, `StageRuntimeState`, `RestoredStageCheckpoint`, `StageCheckpointManager`, JSON atomic write helpers, snapshot copy/load helpers, stage state construction, and restore resolution.

- [ ] **Step 3: Verify manager tests**

Run: `pytest tests/test_checkpoint_v3_stage_state.py -q`
Expected: PASS.

### Task 2: Subagent Flow Integration

**Files:**
- Modify: `k_search/kernel_generators/subagent_orchestration.py`
- Test: `tests/kernel_generators/test_subagent_orchestration.py`

- [ ] **Step 1: Write failing tests**

Add tests that inject a fake stage checkpoint manager into `run_configured_subagent_flow()` and verify save order. Add tests for `filter_stages_after_restore()` skipping completed stages and rerunning `running` stages.

Run: `pytest tests/kernel_generators/test_subagent_orchestration.py -q`
Expected: FAIL because the new parameters and filter helper do not exist.

- [ ] **Step 2: Implement integration**

Extend `run_configured_subagent_flow()` with optional `stage_checkpoint_manager`, `restored_stage_state`, `checkpoint_task`, `round_num`, `attempt_idx`, and `runtime_state`. Save start/completed checkpoints around each stage and filter active stages after restore.

- [ ] **Step 3: Verify subagent tests**

Run: `pytest tests/kernel_generators/test_subagent_orchestration.py -q`
Expected: PASS.

### Task 3: Claude Metadata And Session Options

**Files:**
- Modify: `k_search/kernel_generators/claude_agent_project_editor.py`
- Test: `tests/test_checkpoint_v3_claude_metadata.py`

- [ ] **Step 1: Write failing tests**

Add tests for `extract_agent_id_from_tool_result()`, `ClaudeProjectEditResult` V3 fields, and `_build_options_kwargs()` mapping resume/session/file-checkpointing options.

Run: `pytest tests/test_checkpoint_v3_claude_metadata.py -q`
Expected: FAIL because fields/helpers/options are missing.

- [ ] **Step 2: Implement metadata capture**

Add V3 result fields, client config fields, option mappings, user message UUID capture, and best-effort subagent agentId extraction.

- [ ] **Step 3: Verify metadata tests**

Run: `pytest tests/test_checkpoint_v3_claude_metadata.py -q`
Expected: PASS.

### Task 4: Runner And CLI Wiring

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Modify: `k_search/kernel_generators/kernel_generator.py`
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`
- Modify: `generate_kernels_and_eval.py`
- Test: `tests/test_generate_kernels_cli.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Write failing tests**

Add CLI validation tests for V3 gating and incompatible flags. Add runner tests that enabling checkpoint V3 creates stage checkpoint directories during native subagent execution.

Run: `pytest tests/test_generate_kernels_cli.py tests/kernel_generators/test_ascendc_agentic_codegen.py -q`
Expected: FAIL because CLI flags and runner plumbing are missing.

- [ ] **Step 2: Implement wiring**

Parse V3 flags, validate AscendC/world-model/Claude requirements, pass `StageCheckpointConfig` to the world-model generator and agentic runner, instantiate `StageCheckpointManager` per agentic cycle, and pass it into the subagent flow.

- [ ] **Step 3: Verify runner and CLI tests**

Run: `pytest tests/test_generate_kernels_cli.py tests/kernel_generators/test_ascendc_agentic_codegen.py -q`
Expected: PASS.

### Task 5: Focused Regression

**Files:**
- All changed implementation and tests.

- [ ] **Step 1: Run focused test suite**

Run: `pytest tests/test_checkpoint_v3_stage_state.py tests/test_checkpoint_v3_claude_metadata.py tests/kernel_generators/test_subagent_orchestration.py tests/kernel_generators/test_ascendc_agentic_codegen.py tests/test_generate_kernels_cli.py -q`
Expected: PASS.

- [ ] **Step 2: Inspect git diff**

Run: `git diff --stat` and `git diff --check`
Expected: no whitespace errors and changes limited to checkpoint V3 scope.
