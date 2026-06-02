# MQA策略变体实验结果报告

## 实验概述

**实验时间**: 2026-06-02
**实验目标**: 探索不同策略因素组合对K-Search优化效果的影响
**任务**: Multi-Query Attention (MQA) 算子优化 on Ascend910B3

## 实验设计

### 策略变体列表

已生成16个策略变体文件，位于 `/mnt/workspace/K-Search/strategies/mqa_experiments/`:

| 变体ID | 变体名称 | 策略长度 | 包含因素 |
|--------|---------|---------|---------|
| S1-V1 | Enlarged Tile Sizes - Minimal | 860 chars | basic_fields |
| S1-V2 | Enlarged Tile Sizes - DSL | 1191 chars | basic_fields + DSL |
| S1-V4 | Enlarged Tile Sizes - Usage Scenarios | 1812 chars | basic_fields + usage_scenarios + hardware_constraints |
| S1-V5 | Enlarged Tile Sizes - Anti Patterns | 1931 chars | basic_fields + anti_patterns + precautions |
| S2-V1 | WorkspaceQueue Pattern - Minimal | 1106 chars | basic_fields |
| S2-V3 | WorkspaceQueue Pattern - Priorities | 1479 chars | basic_fields + implementation_priorities |
| S2-V7 | WorkspaceQueue Pattern - Principles | 2563 chars | basic_fields + principles + performance_bottlenecks |
| S4-V1 | Vectorized RowMuls/RowDivs - Minimal | 866 chars | basic_fields |
| S4-V5 | Vectorized RowMuls/RowDivs - Anti Patterns | 2221 chars | basic_fields + anti_patterns + precautions |
| S4-V6 | Vectorized RowMuls/RowDivs - Code Examples | 2346 chars | basic_fields + code_examples + api_references |
| S8-V1 | Targeted SetWaitFlag - Minimal | 1124 chars | basic_fields |
| S8-V4 | Targeted SetWaitFlag - Usage Scenarios | 2082 chars | basic_fields + usage_scenarios + hardware_constraints |
| S8-V5 | Targeted SetWaitFlag - Anti Patterns | 2130 chars | basic_fields + anti_patterns + precautions |
| S10-V1 | Multi-level KV Outer Tiling - Minimal | 990 chars | basic_fields |
| S10-V4 | Multi-level KV Outer Tiling - Usage Scenarios | 1988 chars | basic_fields + usage_scenarios + hardware_constraints |
| S10-V7 | Multi-level KV Outer Tiling - Principles | 2225 chars | basic_fields + principles + performance_bottlenecks |

### 因素类型说明

- **basic_fields**: natural_language + structured_params + expected_speedup (所有策略都包含)
- **DSL**: DSL/伪代码描述
- **implementation_priorities**: 实现优先级 (P0/P1/P2)
- **usage_scenarios**: 使用场景和适用条件
- **hardware_constraints**: 硬件约束和限制
- **anti_patterns**: 反模式和常见错误
- **precautions**: 使用注意事项
- **code_examples**: 示例代码片段
- **api_references**: API参考和文档
- **principles**: 策略原理和机制
- **performance_bottlenecks**: 性能瓶颈分析

## 实验准备

### Baseline性能测量

**待执行步骤**:
```bash
cd /mnt/workspace/cv_agent/tile2asc/multi_query_attention
python3 /mnt/workspace/cv_agent/tile2asc/utils/run_perf.py \
    --phase bench --task-dir . --case-type basic
# 记录输出的 mean_us 值
export BASELINE_MS=<mean_us/1000>
```

**测试配置**:
- batch_size: 4
- n_heads: 8
- seq_len: 512
- d_k: 128
- dtype: float16

### 实验环境

- **硬件**: Ascend910B3
- **框架**: AscendC
- **LLM**: Claude Agent SDK (glm-5.1)
- **优化轮数上限**: 10轮
- **超时**: 900秒/轮

## 实验执行

### 示范实验（推荐先执行）

建议先运行3-5个代表性变体验证流程：

```bash
# 设置baseline
export BASELINE_MS=<baseline_ms>

# 运行示范实验
chmod +x /mnt/workspace/K-Search/scripts/run_mqa_strategy_experiments.sh
./scripts/run_mqa_strategy_experiments.sh --baseline-ms <baseline_ms> --sample
```

示范变体：
- S1-V1 (最小版)
- S1-V5 (反模式版)
- S4-V6 (示例代码版)
- S8-V1 (最小版)
- S10-V4 (使用场景版)

### 完整实验

验证流程后，运行完整实验：

```bash
./scripts/run_mqa_strategy_experiments.sh --baseline-ms <baseline_ms>
```

预计耗时：每个变体约1-2小时，完整实验约16-32小时

## 实验结果记录表

### 结果表格（待填写）

| 变体ID | 策略长度(chars) | 优化轮数 | 最终加速比 | 效果得分 | 状态 | 备注 |
|--------|----------------|---------|-----------|---------|------|------|
| S1-V1 | 860 | - | - | - | pending | - |
| S1-V2 | 1191 | - | - | - | pending | - |
| S1-V4 | 1812 | - | - | - | pending | - |
| S1-V5 | 1931 | - | - | - | pending | - |
| S2-V1 | 1106 | - | - | - | pending | - |
| S2-V3 | 1479 | - | - | - | pending | - |
| S2-V7 | 2563 | - | - | - | pending | - |
| S4-V1 | 866 | - | - | - | pending | - |
| S4-V5 | 2221 | - | - | - | pending | - |
| S4-V6 | 2346 | - | - | - | pending | - |
| S8-V1 | 1124 | - | - | - | pending | - |
| S8-V4 | 2082 | - | - | - | pending | - |
| S8-V5 | 2130 | - | - | - | pending | - |
| S10-V1 | 990 | - | - | - | pending | - |
| S10-V4 | 1988 | - | - | - | pending | - |
| S10-V7 | 2225 | - | - | - | pending | - |

**评估指标计算公式**:
```
效果得分 = (1 / k-search实现策略需要的轮数) * 最终的加速比
```

### 结果分析维度

实验完成后，从以下维度分析：

1. **因素重要性分析**: 哪些因素对效果得分贡献最大？
2. **长度-效果权衡**: 更长的策略是否总是更好？
3. **策略类别差异**: tiling/memory/compute/pipeline类策略对因素的需求是否不同？
4. **失败模式分析**: 缺少哪些因素可能导致失败？

## 预期结果假设

### 主要假设

1. **反模式假设**: 包含anti_patterns的策略成功率更高，减少实现错误
2. **示例代码假设**: 包含code_examples的策略轮数更少，LLM更容易理解实现方式
3. **DSL假设**: DSL形式可能比自然语言更精确，但可读性可能影响理解
4. **长度假设**: 策略长度与效果正相关，但可能存在边际效益递减

### 预期排名（待验证）

根据因素价值推测：
- **高效果组**: S4-V6 (示例代码), S1-V5 (反模式), S4-V5 (反模式)
- **中效果组**: S1-V4 (使用场景), S2-V7 (原理), S10-V4 (使用场景)
- **待验证组**: S1-V1, S2-V1, S4-V1, S8-V1, S10-V1 (最小版)

## 下一步行动

1. **测量Baseline**: 运行性能测试获取BASELINE_MS
2. **示范实验**: 运行5个代表性变体验证流程
3. **完整实验**: 运行全部16个变体
4. **结果分析**: 填写结果表格，分析最佳策略表现形式
5. **总结报告**: 生成最终实验报告和策略设计建议

## 附录

### 策略文件位置
- 策略变体测试文件: `/mnt/workspace/K-Search/strategies/mqa_strategy_variants_test.json`
- 单策略文件目录: `/mnt/workspace/K-Search/strategies/mqa_experiments/`
- 实验脚本: `/mnt/workspace/K-Search/scripts/run_mqa_strategy_experiments.sh`
- 策略提取脚本: `/mnt/workspace/K-Search/scripts/extract_strategy_variants.py`

### 实验输出目录
- 示范实验: `/mnt/workspace/K-Search/.ksearch-exp-mqa-strategies/`
- 每个变体的输出: `.ksearch-exp-mqa-strategies/<variant_id>_natural_language/`

### 参考文档
- 实验方案: `/mnt/workspace/K-Search/docs/mqa_strategy_experiment_plan.md`
- 硬件架构知识: `/mnt/workspace/K-Search/references/basic_knowledge_docs/`
- API参考: `/mnt/workspace/K-Search/references/api_reference_docs/`
- 源码对比分析: 见任务#2的subagent输出
- 硬件知识提取: 见任务#3的subagent输出