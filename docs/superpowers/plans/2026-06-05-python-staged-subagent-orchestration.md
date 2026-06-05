# Python Staged Subagent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a configurable Python orchestration framework that drives Claude native subagents stage-by-stage inside one SDK session.

**Architecture:** Add a package-owned JSON flow config beside the Claude assets. A new orchestrator module loads and validates that flow, builds one prompt per stage, sends each prompt through the existing `ClaudeAgentProjectEditorClient.open_session/send_prompt` API, and runs stage-level file/diff/review checks before returning a combined edit result to the AscendC runner.

**Tech Stack:** Python dataclasses, stdlib `json`, existing Claude Agent SDK editor client abstraction, pytest.

---

### Task 1: Config Model

**Files:**
- Create: `k_search/kernel_generators/subagent_orchestration.py`
- Create: `k_search/kernel_generators/claude_assets/subagent_flow.json`
- Test: `tests/kernel_generators/test_subagent_orchestration.py`

- [ ] **Step 1: Write tests for loading a package default flow config.**
- [ ] **Step 2: Run the new test and confirm it fails because the module does not exist.**
- [ ] **Step 3: Implement config dataclasses and default JSON loading.**
- [ ] **Step 4: Run the new config tests and confirm they pass.**

### Task 2: Staged Session Driver

**Files:**
- Modify: `k_search/kernel_generators/subagent_orchestration.py`
- Test: `tests/kernel_generators/test_subagent_orchestration.py`

- [ ] **Step 1: Write tests for sending configured stages through one opened editor session.**
- [ ] **Step 2: Run the staged driver test and confirm it fails because the driver is missing.**
- [ ] **Step 3: Implement per-stage prompt rendering, stage validation, transcript aggregation, and safe session cleanup.**
- [ ] **Step 4: Run the staged driver tests and confirm they pass.**

### Task 3: AscendC Runner Integration

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Write or update tests proving `run()` uses configured staged prompts and one session.**
- [ ] **Step 2: Confirm the test fails against the existing single-prompt path.**
- [ ] **Step 3: Replace the single `edit_project()` call with the staged orchestrator while preserving handoff filtering, memory persistence, telemetry, artifacts, and existing result fields.**
- [ ] **Step 4: Run focused AscendC agentic tests and update compatibility tests for the new staged behavior.**

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run `python -m pytest tests/kernel_generators/test_subagent_orchestration.py tests/kernel_generators/test_ascendc_agentic_codegen.py -q`.**
- [ ] **Step 2: Run `python -m pytest tests/kernel_generators/test_claude_agent_sdk_mock.py tests/test_launcher_docs.py -q`.**
- [ ] **Step 3: Inspect `git diff --stat` and `git diff --check`.**
