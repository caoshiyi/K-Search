---
name: ascendc-dumptensor
description: |
  用于DumpTensor进行AscendC算子精度的调试方法论。
  Use when:
  - AscendC kernel / 算子精度失败，结果不对，数值错误，部分位置错误，或 NaN/Inf
  - 需要用 DumpTensor / dumptensor / dump tensor 看中间结果、GM/UB/Workspace 数据
  - 需要分段定位 `Cube输入/中间/输出`、`Vector输入/中间/输出` 哪一段出问题
---

# AscendC DumpTensor 精度调试方法论

> 在 K-Search 中，子代理**不运行验证命令、不使用 Bash**；编译与验证由框架（Python）执行。
> 本 skill 提供 DumpTensor 调试的**思考纪律与插桩方法**：子代理负责在 kernel 源码中插入
> `DumpTensor` 调用、规划 dump 点位、并在框架回传日志后做 golden 比对分析。
> 工件统一写到项目根的 `artifacts/`（不是 cv 仓的 `current_task/artifacts/`）。

## 核心规则

1. **精度问题必须 DumpTensor，禁止只读代码猜。** 发现精度错误后的下一步必须是规划插桩 dump 点位，不是继续空读代码。
2. **必须先有 CPU Golden 心智模型。** 明确每个 dump 点位对应的期望中间结果（用 PyTorch 语义推导），desc 编号与点位一一对应。golden 计算脚本可写到 `artifacts/cpu_golden.py`。
3. **必须从输入开始顺序 dump。** 禁止跳过输入直接 dump 中间/输出。沿数据流逐段验证。
4. **发现异常必须回溯直接输入。** 不要只盯着一个异常点空想，沿数据流向前查一跳或多跳。
5. **定位完成后移除 DumpTensor。** 禁止在交付代码中保留（有性能开销）。

## dump 点位规划（强制顺序）

每次调试遵循，禁止跳步：

1. **缩小 case**：单 batch、小 head、最小可复现 seq/block
2. **建立 CPU Golden 心智模型**：列出与 dump 点位对应的期望中间结果及 desc 编号
3. **从输入开始顺序规划 dump 点**：首核/首轮/首 slot，`dumpSize` 先用 8/16/32
4. **在 kernel 源码插入 DumpTensor 调用**：每个点位标注 desc 编号、所在阶段、数据来源
5. **框架回传日志后逐段对比**：输入 → 计算 → 输出，沿数据流依次验证
6. **异常时回溯直接输入**：不要只围绕异常点猜测
7. **生产端正确、消费端错误** → 优先查同步、slot 管理、DataCopy 参数

> 验证日志由框架回传。分析时用 `desc=` 编号定位对应点位的实际数值，与 golden 期望比对。

## 算子类型判定

| 类型 | 特征 | 示例 |
|------|------|------|
| 类型 A - 简单 Vector | 单核、无 Cube、无跨核同步、线性数据流 | Add、Mul、Softmax、LayerNorm |
| 类型 B - 复杂 Cube+Vector | 双核分离、Mmad/Matmul、跨核同步、Workspace 中间数据 | FlashAttention、Matmul fusion |

## 参考资料

按需读取，不要一次性全读：

- `references/api-reference.md` — DumpTensor API 参数说明
- `references/debug-workflow-simple.md` — 类型 A 详细调试步骤
- `references/debug-workflow-complex.md` — 类型 B 详细调试步骤
- `references/error-patterns.md` — 常见错误模式与根因对照表
