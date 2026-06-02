# MQA策略变体实验方案

## 实验目标
探索不同策略因素组合对K-Search优化效果的影响，找出最优的策略表现形式。

## 实验设计

### 1. 策略变体分类

我们为5个关键策略创建了15个变体，每个策略3种不同的因素组合：

#### 策略S1: Enlarged Tile Sizes (增大块大小)
- **S1-V1** (最小版): 仅包含 natural_language + structured_params + expected_speedup
- **S1-V2** (DSL版): V1 + DSL描述
- **S1-V4** (使用场景版): V1 + usage_scenarios + hardware_constraints
- **S1-V5** (反模式版): V1 + anti_patterns + precautions

#### 策略S2: WorkspaceQueue Pattern (工作空间队列模式)
- **S2-V1** (最小版): 仅包含 basic fields
- **S2-V3** (优先级版): V1 + implementation_priorities
- **S2-V7** (原理版): V1 + principles + performance_bottlenecks

#### 策略S4: Vectorized RowMuls/RowDivs (向量化行操作)
- **S4-V1** (最小版): 仅包含 basic fields
- **S4-V5** (反模式版): V1 + anti_patterns + precautions
- **S4-V6** (示例代码版): V1 + code_examples + api_references

#### 策略S8: Targeted SetWaitFlag (精准同步)
- **S8-V1** (最小版): 仅包含 basic fields
- **S8-V4** (使用场景版): V1 + usage_scenarios + hardware_constraints
- **S8-V5** (反模式版): V1 + anti_patterns + precautions

#### 策略S10: Multi-level KV Outer Tiling (多层KV外层切分)
- **S10-V1** (最小版): 仅包含 basic fields
- **S10-V4** (使用场景版): V1 + usage_scenarios + hardware_constraints
- **S10-V7** (原理版): V1 + principles + performance_bottlenecks

### 2. 因素类型总结

| 因素类型 | 说明 | 对策略效果的可能影响 |
|---------|------|---------------------|
| **basic_fields** | natural_language + structured_params + expected_speedup | 基础信息，必需 |
| **DSL** | DSL/伪代码描述 | 可能提高LLM理解精度，但可能过于抽象 |
| **implementation_priorities** | 实现优先级 (P0/P1/P2) | 可能引导LLM按正确顺序实现，减少错误 |
| **usage_scenarios** | 使用场景和适用条件 | 可能帮助LLM判断何时应用策略，避免误用 |
| **hardware_constraints** | 硬件约束和限制 | 可能帮助LLM理解底层限制，生成可行代码 |
| **anti_patterns** | 反模式和常见错误 | 可能帮助LLM避免已知错误，提高成功率 |
| **precautions** | 使用注意事项 | 可能减少实现错误 |
| **code_examples** | 示例代码片段 | 可能帮助LLM理解具体实现方式 |
| **api_references** | API参考和文档 | 可能帮助LLM正确使用API |
| **principles** | 策略原理和机制 | 可能帮助LLM理解为什么这样做，提高实现质量 |
| **performance_bottlenecks** | 性能瓶颈分析 | 可能帮助LLM理解优化目标 |

### 3. 实验假设

**主要假设**：
- 包含更多因素的策略可能提高成功率，但可能增加策略长度
- 反模式和示例代码可能对成功率影响最大
- DSL形式可能比自然语言更精确，但可读性可能较差

**评估指标**：
```
效果得分 = (1 / k-search实现策略需要的轮数) * 最终的加速比
```

对于效果相同的策略，优先选择长度更短的策略。

### 4. 实验流程

#### 步骤1: 准备策略文件
为每个策略变体创建单独的策略文件：
```
strategies/mqa_experiments/S1-V1.json
strategies/mqa_experiments/S1-V2.json
...
strategies/mqa_experiments/S10-V7.json
```

每个文件仅包含一个策略，便于独立测试。

#### 步骤2: 运行优化实验
对每个策略变体执行：
```bash
./scripts/ascendc_mqa_wm.sh \
    --strategy-file strategies/mqa_experiments/<variant>.json \
    --strategy-form natural_language \
    --max-rounds 10 \
    --artifacts-dir .ksearch-exp-mqa_<variant>
```

#### 步骤3: 收集日志并评估
解析日志提取：
- 优化轮数 (rounds)
- 最终加速比 (speedup)
- 策略长度 (character count)

计算评估指标并排序。

### 5. Baseline要求

需要先测量baseline性能：
```bash
cd /mnt/workspace/cv_agent/tile2asc/multi_query_attention
python3 /mnt/workspace/cv_agent/tile2asc/utils/run_perf.py \
    --phase bench --task-dir . --case-type basic
# 记录 mean_us 值，转换为 BASELINE_MS = mean_us / 1000
export BASELINE_MS=<baseline_ms>
```

### 6. 实验输出目录结构

```
.ksearch-exp-mqa_<variant>_natural_language_YYYYMMDD_HHMMSS/
├── world_model/
│   ├── world_model.json          # World model状态
│   ├── solution_db.jsonl         # Solution历史
│   └── llm_logs/                 # LLM交互日志
├── solutions/
│   └── multi_query_attention/
│       └── solution_<round>.json  # 每轮的solution
├── eval/
│   └── multi_query_attention/
│       └── eval_report.json       # 最终评估报告
└── code/
    └── kernel/                    # 生成的kernel代码
```

### 7. 数据收集表格

实验完成后填写：

| 策略变体 | 策略长度(chars) | 优化轮数 | 最终加速比 | 效果得分 | 备注 |
|---------|----------------|---------|-----------|---------|------|
| S1-V1   |                |         |           |         |      |
| S1-V2   |                |         |           |         |      |
| ...     |                |         |           |         |      |

### 8. 预期分析维度

1. **因素重要性排序**：哪些因素对效果得分贡献最大？
2. **长度-效果权衡**：更长的策略是否总是更好？
3. **策略类别差异**：tiling/memory/compute/pipeline类策略对因素的需求是否不同？
4. **失败模式分析**：缺少哪些因素可能导致失败？

## 下一步行动

1. 创建单独的策略文件（每个变体一个文件）
2. 测量baseline性能
3. 运行小规模示范实验（选择3-5个代表性变体）
4. 根据示范结果调整实验方案
5. 运行完整实验
6. 分析结果并总结最佳策略表现形式