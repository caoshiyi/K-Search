# K-Search AscendC Claude Native Subagent 自动 Rollback 设计方案

## 1. 背景

K-Search AscendC + Claude Agent SDK 路径当前采用 native subagent workflow：

```text
initial_codegen:
  code-reader -> designer -> codegen -> reviewer

eval_failure_repair:
  bug-fixer -> reviewer

continue_improve_assessment:
  improvement-assessor

continue_improve_codegen:
  codegen -> reviewer
```

当前流程已经具备 stage 边界、required handoff 校验、stage checkpoint、candidate artifact、evaluation、review feedback retry 等机制。但当 subagent 出现异常时，例如 timeout、max_turns、handoff 缺失、工具协议错误、写错路径、scope 违规等，当前 stage 可能留下半成品文件，影响后续 stage 或后续 retry。

因此需要新增自动 rollback 功能：当 stage 产物不可信时，自动恢复到该 stage 开始前的干净状态，并使用 fresh Claude session 重试同一 stage。

---

## 2. 设计目标

### 2.1 核心目标

1. 保证长跑过程中 candidate workspace 状态干净。
2. stage 失败时自动恢复到 `stage_start` checkpoint。
3. rollback 后不复用脏 Claude session，使用 fresh session 重试。
4. 保留 rollback 前现场，避免根因不可定位。
5. 不改变现有 subagent workflow 的语义。
6. 架构保持简单，避免针对每种异常写主流程 `if/else`。
7. 后续新增 recovery 例外场景时易扩展。

### 2.2 非目标

1. 第一版不做 Claude session rollback。
2. 第一版不使用 Claude SDK file checkpointing 作为权威恢复源。
3. 第一版不做源码级 recovery。
4. 第一版不做复杂证据结构化分析。
5. 第一版不自动进入新 outer attempt。
6. 第一版不使用 `git reset` / `git checkout` 作为 rollback fallback。

---

## 3. 总体设计原则

### 3.1 Stage 即 Transaction

每个 subagent stage 被视为一个事务：

```text
begin stage
  -> save stage_start checkpoint
  -> run subagent
  -> validate stage
  -> optional controlled recovery
  -> decide action
commit or rollback
```

事务成功：

```text
save stage_completed checkpoint
continue next stage
```

事务失败：

```text
backup failure scene
restore project from stage_start checkpoint
close current Claude session
open fresh Claude session
retry same stage according to RetryBudgetPolicy
```

### 3.2 Project Snapshot 是唯一权威恢复源

rollback 只从 `stage_start` checkpoint 的 project snapshot 恢复。

MVP 不使用以下机制作为权威恢复源：

- `git reset`
- `git checkout`
- reverse patch
- Claude session rollback
- Claude file checkpointing

如果 `stage_start` checkpoint 创建失败，则 stage 不启动。

如果 rollback restore 失败，则标记 `ROLLBACK_FAILED`，并 fail 当前 run。

### 3.3 Claude Session Fresh Retry

正常路径下继续保持当前 session 复用，减少对现有 workflow 的影响。

只有发生 rollback 时：

1. close old Claude session；
2. restore project snapshot；
3. open fresh Claude session；
4. retry same stage。

不尝试恢复旧 session，不使用 session rollback。

### 3.4 先备份现场，再 rollback

rollback 前必须备份失败现场。

如果现场备份失败，则不执行 rollback，直接 fail 当前 run，并保留原失败 worktree。

目标是：加 rollback 功能前后，可见日志和产物内容保持一致，只是失败现场从原路径迁移到 failure scene 备份路径。

---

## 4. 架构变更

### 4.1 新增 `StageTransactionExecutor`

新增模块：

```text
k_search/kernel_generators/stage_transaction.py
```

负责单个 stage 的事务执行。

职责：

1. 创建 `stage_start` checkpoint。
2. 执行 subagent stage。
3. 调用 Detector 检测 violation。
4. 调用 RecoveryHandler 尝试受控恢复。
5. 调用 Policy 决定动作。
6. rollback 前备份现场。
7. 执行 `stage_start` snapshot restore。
8. rollback 后 fresh session 重试。
9. 成功后保存 `stage_completed` checkpoint。

`run_configured_subagent_flow()` 保持主流程简洁，只负责遍历 active stages：

```python
for stage in active_stages:
    result = stage_transaction_executor.run_stage(...)
```

### 4.2 新增 ViolationDetector / RecoveryHandler / Policy 三层扩展

为了支持后续增加更多例外场景，采用三层结构：

```text
ViolationDetector:
  只发现问题，不修复。

RecoveryHandler:
  只处理受控恢复，不决定最终 action。

RollbackPolicy:
  根据 violation 和 recovery result 决定 COMMIT / RECOVERED_AND_CONTINUE /
  KEEP_AND_CONTINUE / ROLLBACK_AND_RETRY / ROLLBACK_AND_FAIL。
```

禁止把异常处理写成主流程 `if/else`。

---

## 5. 核心数据结构

### 5.1 `StageContext`

```python
@dataclass
class StageContext:
    project_root: Path
    flow: SubagentFlowConfig
    stage: SubagentStageConfig
    stage_index: int
    round_num: int
    attempt_idx: int
    stage_retry_index: int
    run_id: str
    task_name: str
    action_node_id: str | None
    candidate_id: str | None
    checkpoint_manifest_path: Path
    telemetry_recorder: Any | None
    session: Any | None
    runtime_state: dict[str, Any]
```

### 5.2 `StageViolation`

```python
@dataclass(frozen=True)
class StageViolation:
    code: str
    severity: str
    stage: str
    agent: str
    message: str
    evidence: dict[str, Any]
```

常见 `code`：

- `MISSING_REQUIRED_OUTPUT`
- `INVALID_HANDOFF`
- `SUBAGENT_INVOCATION_VIOLATION`
- `PATH_ESCAPE`
- `SCOPE_VIOLATION`
- `DIFF_POLICY_VIOLATION`
- `SUBAGENT_TIMEOUT`
- `MAX_TURNS`
- `EMPTY_MODEL_RESULT`
- `TOOL_PROTOCOL_ERROR`
- `SOURCE_POLLUTION`
- `ROLLBACK_FAILED`
- `SCENE_BACKUP_FAILED`

### 5.3 `RecoveryResult`

```python
@dataclass(frozen=True)
class RecoveryResult:
    recovered: bool
    recovery_code: str
    source_violation_code: str
    changed_paths: list[str]
    cleanup_paths: list[str]
    evidence: dict[str, Any]
```

### 5.4 `StageAction`

```python
class StageAction(Enum):
    COMMIT = "commit"
    KEEP_AND_CONTINUE = "keep_and_continue"
    RECOVERED_AND_CONTINUE = "recovered_and_continue"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    ROLLBACK_AND_FAIL = "rollback_and_fail"
    FAIL_WITHOUT_ROLLBACK = "fail_without_rollback"
```

---

## 6. `StageTransactionExecutor` 流程

### 6.1 正常流程

```python
def run_stage(ctx):
    checkpoint = checkpoint_manager.save_stage_start(...)
    try:
        result = run_one_subagent_stage(...)
        violations = detector_registry.detect(ctx, result)
        recovery_results = recovery_registry.try_recover(ctx, violations)
        if recovery_results:
            violations = detector_registry.detect(ctx, result)
        action = rollback_policy.decide(ctx, violations, recovery_results)
        if action in {COMMIT, RECOVERED_AND_CONTINUE}:
            checkpoint_manager.save_stage_completed(...)
            return result
        if action == KEEP_AND_CONTINUE:
            return result
        if action == ROLLBACK_AND_RETRY:
            backup_scene_or_fail(ctx)
            rollback_to_stage_start(ctx, checkpoint)
            close_session_and_open_fresh(ctx)
            retry_same_stage(ctx)
        if action == ROLLBACK_AND_FAIL:
            backup_scene_or_fail(ctx)
            rollback_to_stage_start(ctx, checkpoint)
            fail_run(ctx)
    except Exception as exc:
        violations = classify_exception(exc)
        action = rollback_policy.decide(ctx, violations, [])
        handle_action(action)
```

### 6.2 rollback 后重试语义

rollback 到 `stage_start` 不算新 outer attempt。

保持不变：

- `round_num`
- `attempt_idx`
- `candidate_id`

新增或更新：

- `stage_retry_index + 1`
- `rollback_count + 1`
- `rollback_reason`

---

## 7. MVP Detector 范围

MVP 覆盖状态可信性核心 Detector。

### 7.1 `RequiredOutputDetector`

检查 required handoff 是否存在。

异常示例：

- `CODE_MAP.md` 缺失
- `ASCENDC_DESIGN.md` 缺失
- `IMPLEMENTATION_EXECUTION_PLAN.md` 缺失
- `IMPLEMENTATION_HANDOFF.md` 缺失
- `REVIEW_NOTES.md` 缺失
- `IMPROVEMENT_ASSESSMENT.md` 缺失

输出：`MISSING_REQUIRED_OUTPUT`。

### 7.2 `HandoffContentDetector`

检查 handoff 内容是否满足协议。

异常示例：

- `CODE_MAP.md` 太短
- `ASCENDC_DESIGN.md` 太短
- `IMPLEMENTATION_EXECUTION_PLAN.md` 太短
- `IMPLEMENTATION_HANDOFF.md` 太短
- `REVIEW_NOTES.md` 缺 `eval_ready`
- `REVIEW_NOTES.md` 的 `eval_ready` 值不可解析
- `IMPROVEMENT_ASSESSMENT.md` 的 `status` 不支持

注意：

- `REVIEW_NOTES.md` 缺 `eval_ready` = `INVALID_HANDOFF`，rollback reviewer stage。
- `REVIEW_NOTES.md` 中 `eval_ready=false` = 正常质量门，不 rollback，进入 review-feedback retry。

### 7.3 `SubagentInvocationDetector`

检查是否调用了指定 subagent。

异常示例：

- 没有调用 Agent / Task。
- 调用了错误 subagent。
- 调用了多个不该调用的 subagent。
- parent agent 直接完成 stage。

输出：`SUBAGENT_INVOCATION_VIOLATION`。

### 7.4 `PathEscapeDetector`

检查 required handoff 是否写到了 project root 外部。

异常示例：

- 写到父目录。
- 写到历史 run。
- 写到 archive。
- 写到绝对路径。
- 写到 project root 外部。

输出：`PATH_ESCAPE`。

对于 handoff 文件，可交给 `HandoffPathRecoveryHandler` 尝试受控恢复。

### 7.5 `StageScopeDetector`

检查 stage 是否改了不该改的文件。

示例：

- `code-reader` 改源码。
- `designer` 改源码。
- `reviewer` 改源码。
- `improvement-assessor` 改源码。
- `evaluator` / `curator` 修改候选源码。
- `bug-fixer` / `codegen` 修改非候选路径。

输出：`SCOPE_VIOLATION`。

### 7.6 `CandidateDiffDetector`

检查实现类 stage 的 candidate diff 是否合理。

异常示例：

- `codegen` / `bug-fixer` 本应改源码但没有 candidate diff。
- diff 只有 handoff / runtime 文件。
- diff 涉及 `.claude` / `build` / `cache` / `log` / `artifact` 等非候选路径。
- diff 删除关键入口文件。

输出：`DIFF_POLICY_VIOLATION`。

---

## 8. MVP RecoveryHandler

### 8.1 `HandoffPathRecoveryHandler`

MVP 唯一启用的 RecoveryHandler。

适用场景：required handoff 缺失，但 telemetry 显示 subagent 把同名 handoff 写到了 project root 外部。

处理流程：

1. 根据 telemetry 查找 Write / Edit 事件。
2. 确认外部文件名与缺失 handoff 文件名一致。
3. 确认只有一个候选来源。
4. 复制外部 handoff 到 project root 正确位置。
5. 运行 handoff validator。
6. 如果 validator 通过，记录 recovery result。
7. 清理可证明由本 stage 写出的外部 handoff 文件。
8. 继续后续 stage validation。

允许恢复：

- `CODE_MAP.md`
- `ASCENDC_DESIGN.md`
- `IMPLEMENTATION_EXECUTION_PLAN.md`
- `IMPLEMENTATION_HANDOFF.md`
- `IMPLEMENTATION_DEVIATIONS.md`
- `REVIEW_NOTES.md`
- `IMPROVEMENT_ASSESSMENT.md`

禁止恢复：

- 源码文件
- 构建文件
- 测试文件
- benchmark 文件
- runtime / cache / build / log 文件
- 多候选来源文件
- 内容非法 handoff
- 不确定来源文件
- 历史 run / archive / home 等高风险路径文件

清理策略：只清理可证明由本 stage 写出的 handoff 错误文件。不清理源码、非 handoff、不确定来源、高风险路径文件。

---

## 9. `RollbackPolicy`

### 9.1 默认规则

默认策略：stage 已开始后，除正常质量门和受控 handoff recovery 外，失败默认 rollback。

### 9.2 Action 映射

| 情况 | Action |
| --- | --- |
| 无 violation | `COMMIT` |
| handoff recovery 成功且重新校验通过 | `RECOVERED_AND_CONTINUE` |
| reviewer `eval_ready=false` | `KEEP_AND_CONTINUE` |
| eval `compile_failed` / `accuracy_failed` / `performance_not_improved` | `KEEP_AND_CONTINUE` |
| provider transient before edit | retry without rollback 或 no-op rollback |
| missing required output | `ROLLBACK_AND_RETRY` |
| invalid handoff | `ROLLBACK_AND_RETRY` |
| subagent timeout / max_turns | `ROLLBACK_AND_RETRY` |
| empty model result | `ROLLBACK_AND_RETRY` |
| tool protocol error | `ROLLBACK_AND_RETRY` |
| scope violation | `ROLLBACK_AND_RETRY` |
| diff policy violation | `ROLLBACK_AND_RETRY` |
| path escape not recoverable | `ROLLBACK_AND_FAIL` |
| secret leak | `ROLLBACK_AND_FAIL` |
| git state corrupted | `ROLLBACK_AND_FAIL` |
| rollback failed | `FAIL_WITHOUT_ROLLBACK` / fail run |
| scene backup failed | `FAIL_WITHOUT_ROLLBACK` / fail run |

---

## 10. `RetryBudgetPolicy`

采用按 failure class 配置 retry budget。

MVP 默认表：

```python
DEFAULT_RETRY_BUDGET = {
    "PROVIDER_TRANSIENT_NO_EDIT": 3,
    "MISSING_REQUIRED_OUTPUT": 1,
    "INVALID_HANDOFF": 1,
    "TOOL_PROTOCOL_ERROR": 1,
    "SUBAGENT_TIMEOUT": 1,
    "MAX_TURNS": 1,
    "EMPTY_MODEL_RESULT": 1,
    "SOURCE_POLLUTION": 1,
    "SCOPE_VIOLATION": 1,
    "DIFF_POLICY_VIOLATION": 1,
    "PATH_ESCAPE": 0,
    "SECRET_LEAK": 0,
    "GIT_STATE_CORRUPTED": 0,
    "ROLLBACK_FAILED": 0,
    "SCENE_BACKUP_FAILED": 0,
}
```

stage retry 耗尽后：

- fail 当前 run；
- 不自动进入新 attempt；
- 不自动放弃当前 attempt 后重试 action。

这有利于 meta harness 接管并定位 K-Search 自身问题。

---

## 11. Failure Scene Backup

### 11.1 备份原则

rollback 前简单直接地备份现场。

目标是：加 rollback 功能前后，能看到的日志和产物内容一致，只是路径不同。

### 11.2 备份范围

备份范围包括：

- active worktree `project_dir`
- 当前 attempt / stage artifacts
- telemetry trace / timeline / cost
- stage checkpoint / stage prompt
- `scene_manifest.json`

目录示例：

```text
<run_dir>/failures/
  round_0003_attempt_01_codegen_retry_00/
    scene_manifest.json
    raw/
      project_before_rollback/
      attempt_artifacts/
      telemetry/
      checkpoint/
```

### 11.3 `scene_manifest.json`

只做轻量索引，不做复杂结构化 evidence。

示例：

```json
{
  "schema_version": 1,
  "run_id": "20260612_xxx",
  "round_num": 3,
  "attempt_idx": 1,
  "candidate_id": "round_0003_attempt_01",
  "stage": "codegen",
  "agent": "codegen",
  "stage_retry_index": 0,
  "rollback_reason": "MISSING_REQUIRED_OUTPUT",
  "checkpoint_id": "ckpt_000012_stage_r0003_a01_codegen_start",
  "backup_paths": {
    "project_before_rollback": "raw/project_before_rollback",
    "attempt_artifacts": "raw/attempt_artifacts",
    "telemetry": "raw/telemetry",
    "checkpoint": "raw/checkpoint"
  }
}
```

### 11.4 备份失败处理

如果 scene backup 失败：

1. 不执行 rollback；
2. fail 当前 run；
3. 保留 active worktree 原现场。

原因：不能因为 rollback 把唯一失败现场清掉。

---

## 12. Rollback Restore

### 12.1 恢复源

唯一恢复源：`stage_start` checkpoint project snapshot。

### 12.2 restore 流程

1. 读取 `stage_start` checkpoint manifest。
2. 找到 `pre_stage_project_snapshot`。
3. 清空 active `project_dir` 中除 `.git` 之外的文件。
4. materialize project snapshot 到 active `project_dir`。
5. 恢复 checkpoint 中保存的 handoff 文件。
6. 校验恢复结果。
7. 写 `rollback_result.json`。

### 12.3 restore 失败

如果恢复失败：

1. 标记 `ROLLBACK_FAILED`；
2. fail 当前 run；
3. 不继续自动 retry。

---

## 13. Session 策略

### 13.1 正常路径

正常路径保持现有 session 复用：

```text
code-reader -> designer -> codegen -> reviewer
```

在同一个 cycle / session 中继续执行。

### 13.2 rollback 路径

rollback 后：

1. close old session；
2. open fresh session；
3. retry same stage。

不做：

- session rollback
- session resume
- session fork
- Claude SDK file rewind

---

## 14. 自动启用策略

自动 rollback 默认启用，但限制在：

- `task_source = ascendc`
- `language = ascendc`
- `llm_provider = claude-agent`
- configured native subagent flow enabled

如果满足以上条件：

1. 自动启用 checkpoint v3 `stage_start` project snapshot；
2. 自动启用 stage auto rollback。

如果 `stage_start` checkpoint 创建失败：preflight fail，不启动 stage。

---

## 15. 与现有流程的关系

### 15.1 不改变 subagent workflow

保留现有 flow：

- `initial_codegen`
- `eval_failure_repair`
- `continue_improve_assessment`
- `continue_improve_codegen`
- review feedback retry

rollback 只是包在每个 stage 外层。

### 15.2 不改变 eval failure 语义

以下情况不 rollback candidate：

- `compile_failed`
- `accuracy_failed`
- `benchmark_failed`
- `performance_not_improved`
- blocked / no executable

这些是候选质量问题，不是 stage 状态污染问题。保留 candidate、diff、eval log，进入 `bug-fixer` / improve。

### 15.3 不改变 reviewer feedback 语义

`REVIEW_NOTES.md` 中 `eval_ready=false` 表示 reviewer 正常质量反馈，不 rollback candidate，进入 review-feedback retry。

但 `REVIEW_NOTES.md` 缺 `eval_ready` 是 handoff 协议异常，需要 rollback reviewer stage。

---

## 16. 实现计划

### Step 1：新增 `StageTransactionExecutor`

新增文件：

```text
k_search/kernel_generators/stage_transaction.py
```

实现：

- `StageTransactionExecutor`
- `StageContext`
- `StageAction`
- `StageViolation`
- `RecoveryResult`

### Step 2：`StageCheckpointManager` 增加 restore 到当前 worktree 能力

新增方法：

```python
def restore_stage_start_to_project(
    self,
    checkpoint_manifest_path: Path,
    target_project_dir: Path,
) -> RollbackRestoreResult:
    ...
```

要求：

- 只从 `pre_stage_project_snapshot` 恢复。
- 恢复失败直接抛错。
- 不使用 git fallback。

### Step 3：改造 `run_configured_subagent_flow`

当前 per-stage loop 改为调用：

```python
executor.run_stage(...)
```

`run_configured_subagent_flow()` 保持：

- active stages 计算；
- session 生命周期管理；
- flow transcript merge；
- final result 聚合。

### Step 4：实现 `DetectorRegistry`

MVP Detector：

- `RequiredOutputDetector`
- `HandoffContentDetector`
- `SubagentInvocationDetector`
- `PathEscapeDetector`
- `StageScopeDetector`
- `CandidateDiffDetector`

### Step 5：实现 `HandoffPathRecoveryHandler`

只支持 handoff-only controlled recovery。

恢复成功后：

1. copy handoff；
2. validate；
3. safe cleanup；
4. record recovery result。

### Step 6：实现 `RollbackPolicy` 和 `RetryBudgetPolicy`

实现：

```python
RollbackPolicy.decide(...)
RetryBudgetPolicy.max_retries(...)
```

默认 retry 表使用偏保守策略。

### Step 7：实现 `FailureSceneBackup`

新增模块：

```text
k_search/kernel_generators/rollback_scene_backup.py
```

实现：

```python
backup_failure_scene(ctx, reason) -> SceneBackupResult
```

备份：

- `project_before_rollback`
- `attempt_artifacts`
- `telemetry`
- `checkpoint`
- `scene_manifest.json`

备份失败时抛出 `SceneBackupFailed`。上层捕获后 fail run，不 rollback。

### Step 8：fresh session retry

`StageTransactionExecutor` 在 rollback 后：

1. close current session；
2. open fresh session；
3. retry same stage。

正常路径不改变 session 复用。

### Step 9：测试

Unit tests：

- `test_missing_required_output_triggers_rollback`
- `test_invalid_review_notes_missing_eval_ready_triggers_rollback`
- `test_review_eval_ready_false_does_not_rollback`
- `test_handoff_path_recovery_success`
- `test_handoff_path_recovery_invalid_content_rollbacks`
- `test_scope_violation_rollbacks`
- `test_diff_policy_violation_rollbacks`
- `test_backup_failure_preserves_scene_and_fails`
- `test_rollback_restore_failure_fails_run`
- `test_retry_budget_exhaustion_fails_run`

Integration tests：

- `initial_codegen`: codegen missing handoff -> rollback -> fresh retry。
- reviewer `eval_ready=false` -> review feedback retry, no rollback。
- bug-fixer writes `CODE_MAP.md` outside project -> recover -> validate -> continue。
- code-reader modifies source -> rollback -> retry。
- tool protocol error -> backup -> rollback -> fresh retry。

---

## 17. 风险与缓解

### 风险 1：rollback 误删现场

缓解：rollback 前必须 scene backup；backup 失败不 rollback。

### 风险 2：retry 循环消耗 token

缓解：`RetryBudgetPolicy` 有明确上限；retry 耗尽 fail run。

### 风险 3：RecoveryHandler 泛化导致复杂

缓解：MVP 只启用 handoff-only recovery；源码 recovery 禁止。后续新增 recovery 必须单独 handler + 单测。

### 风险 4：checkpoint 创建失败后仍继续执行

缓解：auto rollback 默认启用时，`stage_start` checkpoint 是 preflight 必需条件。创建失败直接 fail，不启动 stage。

### 风险 5：Claude session 文件状态不一致

缓解：rollback 后 fresh session；不做 session rollback。

---

## 18. 最终语义总结

自动 rollback 的最终语义：

- 每个 native subagent stage 都是事务。
- `stage_start` checkpoint 是唯一恢复源。
- stage 成功则 commit。
- 正常质量门失败则保留候选走原 workflow。
- handoff 写错位置且可证明可恢复时，走 handoff-only controlled recovery。
- 其他不可信 stage 产物默认 rollback。
- rollback 前原样备份失败现场。
- rollback 后 fresh session 重试同 stage。
- 重试耗尽后 fail 当前 run，交由 meta harness 接管。
