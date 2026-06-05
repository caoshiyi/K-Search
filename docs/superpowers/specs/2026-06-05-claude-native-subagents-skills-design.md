# Claude Native Subagents and Skills Design

> 日期: 2026-06-05
> 分支: `feat/claude-agent-subagents-skills`

## 背景

K-Search 的 AscendC agentic codegen 已经通过 `ClaudeAgentProjectEditorClient`
把 Claude Agent SDK 当作项目编辑器使用:Python 侧创建临时 worktree,叠加
candidate,调用 `ClaudeSDKClient`,收集 diff,运行评测并写 artifacts。

当前仍有一套 Python 内部 agent 角色:

- `ProjectAgent`
- `CodeReaderAgent`
- `CodegenAgent`

这套角色与 Claude Code 原生 subagent/skill 能力形成平行架构。用户希望下一步直接全量替换为更简单干净的架构:角色定义迁移到 Claude 原生
`.claude/agents/` 和 `.claude/skills/`,Python 只保留最薄的生命周期编排。

用户提供的 Claude Agent SDK 研究报告给出关键边界:

- Agent SDK 的主入口是 session/agent loop,不是 `createAgent()` 风格对象。
- 本地 skills 是文件系统能力包,通过 `.claude/skills/<name>/SKILL.md`
  和 `setting_sources=["project"]` 被发现。
- skills 不是函数式注册 API,也没有独立返回 schema。
- 显式限制 tools 时,需要把 `Skill` 工具放入可用工具列表。
- subagents 属于 `.claude/agents/` / Agent tool 委派能力。
- 权限控制仍在 SDK session options 层,而不是 skill 文件内部。

## 目标

把 AscendC Claude Agent SDK 路径改为 Claude 原生 subagent/skill 架构:

1. 运行时不再依赖 Python `CodeReaderAgent` / `CodegenAgent` 子类来表达角色。
2. 每个 agentic worktree 都包含项目级 `.claude/agents/` 和 `.claude/skills/` 资产。
3. 主 Claude session 使用原生 subagents 完成:
   `code-reader -> plan -> codegen -> reviewer -> Python eval`。
4. `bug-fixer` subagent 本期只预留空文件,后续用于 codegen 失败后的修复。
5. Python 继续负责 worktree、memory、strategy injection、评测、artifact、telemetry、session 生命周期。

## 非目标

- 不让 Claude 直接运行 build/test/bench。
- 不启用 Bash。
- 不把 skills 当作严格 JSON 输入输出工具。
- 不引入 MCP 或 custom tools。
- 不重构非 AscendC、非 Claude Agent SDK 路径。
- 不在本期接入 `bug-fixer` 修复流程。

## 架构

### 总体边界

Python 侧只保留生命周期和结果判定:

- 创建和清理 isolated worktree。
- 叠加 base solution 并提交 baseline。
- materialize memory 和 Claude assets。
- 调用 `ClaudeAgentProjectEditorClient`。
- 读取 worktree diff 和 `CODE_MAP.md`。
- 运行 K-Search 评测。
- 生成 `Solution`、snapshot、candidate artifacts、telemetry。

Claude 原生资产负责 agent 内部认知和编辑工作流:

- 项目理解。
- 策略实施方案。
- AscendC API 文档查阅。
- 首次代码修改。
- 改动审查。

### 内置资产目录

新增 repo 内置模板目录:

```text
k_search/kernel_generators/claude_assets/
  agents/
    code-reader.md
    plan.md
    codegen.md
    reviewer.md
    bug-fixer.md
  skills/
    ascendc-codegen/
      SKILL.md
    ascendc-api-reference/
      SKILL.md
```

运行时 materialize 到 candidate worktree:

```text
<worktree>/.claude/
  agents/
    code-reader.md
    plan.md
    codegen.md
    reviewer.md
    bug-fixer.md
  skills/
    ascendc-codegen/
      SKILL.md
    ascendc-api-reference/
      SKILL.md
```

### Subagents

`code-reader`

- 读取 AscendC operator project。
- 生成或更新 worktree 根目录的 `CODE_MAP.md`。
- 不提出优化建议。
- 不修改源码。

`plan`

- 读取策略文本、`CODE_MAP.md`、trace/perf context、API references。
- 生成本轮实施方案 `IMPLEMENTATION_PLAN.md`。
- 计划应限制改动范围,优先单点、低风险、可评测变更。
- 类似 superpowers 的 spec/plan 思维,但输出服务于当前一次 codegen attempt。

`codegen`

- 严格按 `IMPLEMENTATION_PLAN.md` 修改候选工程。
- 不重新发散策略。
- 只编辑必要源码或配置文件。
- 修改后更新受影响的 `CODE_MAP.md` section。

`reviewer`

- 审查 codegen 改动是否越界。
- 检查 AscendC host/kernel contract、tiling、dtype、shape、entry point、build layout。
- 如果发现风险,要求主 agent 修正后再结束。

`bug-fixer`

- 本期只创建占位 subagent 文件。
- 文件内容明确标注 reserved / not active in current K-Search flow。
- 主 prompt 不调用它,避免 Claude 提前误用。
- 后续接入失败修复流:
  `Python eval failure -> bug-fixer -> reviewer -> Python re-eval`。

### Skills

`ascendc-codegen`

- 描述 AscendC 优化的全局工作流。
- 强调 worktree 内编辑、禁止 Bash、禁止大范围改动、保留 operator contract。
- 指导 code-reader/plan/codegen/reviewer 的协作方式。

`ascendc-api-reference`

- 指导 Claude 如何使用 repo 内 `references/` 知识库。
- 说明策略 action_text 中的 `references/<doc_path>` 是优先查阅路径。
- 要求缺失签名或约束时先读文档,不要按通用 C++ 经验猜 API。

skills 提供工作流和知识查阅方法,不承担强约束执行。权限仍由
`ClaudeAgentProjectEditorClient` 的 session options 控制。

## SDK 配置

`ClaudeAgentProjectEditorClient` 作为唯一 SDK 入口,新增原生资产相关配置:

- `setting_sources=["project"]`
- `skills=["ascendc-codegen", "ascendc-api-reference"]`
- 显式 tools 时自动包含 `Skill`
- 保持 `cwd=<candidate worktree>`
- 保持 `permission_mode="acceptEdits"`
- 保持 `Bash` 禁用
- 保持 model、max_turns、thinking、timeout、telemetry 现有行为

如果 SDK 需要的 subagent 工具名在 Python 包版本中表现为 `Agent` 或 `Task`
等工具名,实现应把工具名集中到一个 helper 中,避免散落硬编码。

## 数据流

首次 attempt:

1. `AscendCAgenticCodegenRunner` 创建临时 candidate worktree。
2. Runner 叠加 base solution 并提交 baseline。
3. `ClaudeAssetMaterializer` 写入 `.claude/agents/` 和 `.claude/skills/`。
4. Runner 从 `MemoryStore` materialize `CODE_MAP.md`;若没有,由主 agent 委派
   `code-reader` 生成。
5. Runner 构造主 prompt,要求 Claude 使用:
   `code-reader -> plan -> codegen -> reviewer`。
6. `plan` subagent 写 `IMPLEMENTATION_PLAN.md`。
7. `codegen` subagent 按计划修改文件。
8. `reviewer` subagent 审查并推动必要修正。
9. Python 收集 changed files、diff、transcript、telemetry。
10. Python 运行评测。
11. Python 读取 `CODE_MAP.md` 并保存回 `MemoryStore`。
12. Python 写 candidate artifacts 和 project snapshot。

失败修复预留流:

1. Python eval 失败。
2. 后续版本把失败摘要发送给 `bug-fixer`。
3. `bug-fixer` 修改。
4. `reviewer` 审查。
5. Python re-eval。

本期不执行失败修复流。

## Prompt 策略

主 prompt 只描述一次 attempt 的目标、上下文、约束和必须使用的原生 agent flow。
它不再内联 `CodeReaderAgent` 或 `CodegenAgent` 的完整角色模板。

主 prompt 必须包含:

- Target GPU、mode、round、attempt。
- Strategy/action text。
- Perf summary。
- Trace excerpt。
- `CODE_MAP.md` 是否已存在。
- 必须使用 `code-reader -> plan -> codegen -> reviewer`。
- `IMPLEMENTATION_PLAN.md` 是 plan subagent 的输出。
- `bug-fixer` 本期不得调用。
- 结束时给出 concise summary 和 changed-file list。

## 迁移策略

这是一次硬切换:

- `AscendCAgenticCodegenRunner.run()` 不再实例化 `CodeReaderAgent` 或
  `CodegenAgent`。
- `run_multi_turn()` 和 `continue_fix()` 的长期目标是映射到 `bug-fixer`,
  但本期不接入 `bug-fixer`。
- 旧 `k_search/kernel_generators/agents/` 模块可以短期保留为兼容导入层,
  但运行时路径不得依赖它。
- 测试应明确防止旧 Python agent 被调用。

## 错误处理

- 资产写入失败:当前 attempt 失败,不静默降级。
- skill 未发现或不可用:当前 attempt 失败并保留 transcript。
- subagent 不可用:当前 attempt 失败,不回退到旧 Python agents。
- 应生成 `CODE_MAP.md` 但缺失:当前 attempt 失败。
- `IMPLEMENTATION_PLAN.md` 缺失:当前 attempt 失败。
- reviewer 报告未修正风险:当前 attempt 失败。
- Claude 未修改任何候选源码:沿用现有 `did not change any files` 失败逻辑。
- SDK/timeout/auth 错误:沿用现有 provider exception 和 telemetry 逻辑。

## 测试计划

Materializer:

- 写入 `.claude/agents/*.md`。
- 写入 `.claude/skills/*/SKILL.md`。
- 不覆盖 worktree 中已有用户文件,除非明确由 K-Search 管理。

SDK options:

- `cwd` 指向 worktree。
- `setting_sources == ["project"]`。
- `skills` 包含 `ascendc-codegen` 和 `ascendc-api-reference`。
- `allowed_tools` 包含 `Skill`。
- `Bash` 仍在 `disallowed_tools`。

Runner:

- mock SDK 响应写入 `CODE_MAP.md`、`IMPLEMENTATION_PLAN.md` 和源码改动。
- 验证 runner 不调用旧 `CodeReaderAgent` / `CodegenAgent`。
- 验证 prompt 包含原生 agent flow。
- 验证缺失 `IMPLEMENTATION_PLAN.md` 时失败。
- 验证缺失 `CODE_MAP.md` 时按规则失败。

Memory:

- 新生成或更新的 `CODE_MAP.md` 回写 `MemoryStore`。
- 已有 memory materialize 到 worktree 后可跳过初次 code-reader 生成要求。

Artifacts:

- `IMPLEMENTATION_PLAN.md` 保留在 snapshot/diff 中。
- candidate artifact metadata 可记录 native agents enabled。

Compatibility:

- 现有 mock Claude Agent SDK 流程继续通过。
- 非 Claude Agent SDK provider 不受影响。
- 非 AscendC language 不受影响。

## 验收标准

- AscendC Claude Agent SDK 路径运行时不依赖 Python `CodeReaderAgent` /
  `CodegenAgent`。
- 每个 agentic worktree 都有项目级 `.claude/agents` 和 `.claude/skills`。
- SDK options 正确启用 project setting source 和 skills。
- 主 prompt 指挥原生 `code-reader -> plan -> codegen -> reviewer`。
- `bug-fixer` 占位文件存在但不在本期 flow 中调用。
- 失败时不回退旧 Python agent 体系。
- 现有 worktree、telemetry、artifact、evaluation 行为保持。
