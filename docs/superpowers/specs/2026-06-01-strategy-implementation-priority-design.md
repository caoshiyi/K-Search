# 策略层改进设计：Implementation Priority 与术语修正

> 设计日期：2026-06-01
> 目标：解决自然语言策略中"多维度叠加变更失败率高"的问题（文档：nl_strategy_problem_analysis.md）

---

## 一、问题背景

### 核心问题

实验数据表明：
- **单维变更通过率 100%（2/2）**：仅改 BLOCK_M/N/K=128 (R3) 和仅改 VEC2_M_CHUNK=16 (R12) 都成功
- **多维变更通过率 0%（0/6）**：tile放大 + DMA预取、Duplicate+Mul + VEC2_CHUNK 等都失败

### 根因分析

策略描述采用全面阐述风格，暗示"可以同时改多个维度"，但缺少：
1. **优先级排序**：哪些变更应该先做，哪些应该后做
2. **变更约束**：建议每轮修改的文件数量上限
3. **术语准确**：部分策略将用户自定义函数误称为 "hardware API"

---

## 二、设计目标

1. 在策略渲染输出开头注入 **Implementation Priority** 提示，引导 LLM 单步骤变更
2. 新增 `implementation_priorities` 字段表达变更优先级和依赖关系
3. 新增 `expected_speedup_interval` 字段（保留但不渲染，供 WM 内部决策）
4. 修正 S4 策略中的术语问题（"hardware API" → "vectorized helper functions"）

---

## 三、新增字段定义

### 3.1 `implementation_priorities` 字段

```json
"implementation_priorities": [
  {
    "priority": "P0",
    "action": "Change tile size constants (BLOCK_M, BLOCK_N, BASE_K) from 64 to 128",
    "dependency": null
  },
  {
    "priority": "P1",
    "action": "Adjust L1 buffer allocations for larger tile sizes",
    "dependency": "P0 verified"
  },
  {
    "priority": "P2",
    "action": "Reduce loop iteration count and sync overhead",
    "dependency": "P0+P1 verified"
  }
]
```

**设计要点**：
- `priority`：优先级标签（P0 = 最高/核心变更）
- `action`：变更描述（不包含具体文件名，保持通用性）
- `dependency`：依赖关系（如 "P0 verified" 表示需要 P0 先验证通过）
- **不包含 `files` 字段**：策略是通用的，文件名由 Task/Definition 提供

### 3.2 `expected_speedup_interval` 字段

```json
"expected_speedup_interval": {
  "min": 1.30,
  "likely": 1.47,
  "max": 1.50,
  "rationale": "min accounts for dim=512 L1 pressure fallback; likely based on dim=128 basic case实测; max is ideal L0 utilization"
}
```

**设计要点**：
- **不渲染到 prompt**：仅用于 WM 内部决策（判断是否继续尝试）
- 提供置信区间而非单一乐观值，避免 WM 过度追求"补足差距"

---

## 四、渲染函数改进

### 4.1 新增渲染函数 `_render_priorities_section`

```python
def _render_priorities_section(priorities: list[dict[str, Any]]) -> str:
    """Render implementation priorities into a structured section."""
    
    lines = ["=== Implementation Priority ==="]
    lines.append("Apply ONE change at a time, verify before stacking:")
    
    for p in priorities:
        priority = p.get("priority", "P?")
        action = p.get("action", "unknown action")
        dep = p.get("dependency")
        
        lines.append(f"{priority}: {action}")
        if dep:
            lines.append(f"     → Requires: {dep}")
    
    lines.append("")
    lines.append("Constraint: Recommended 1-2 files per round.")
    lines.append("→ Single-file changes have 100% success rate in experiments.")
    lines.append("→ Do NOT combine P0+P1+P2 in one round. Each round should change ONE priority level.")
    
    return "\n".join(lines)
```

### 4.2 修改 `render_strategy_as_action_text`

```python
def render_strategy_as_action_text(
    strategy: dict[str, Any],
    form: str = "natural_language",
) -> str:
    """Render a single strategy into action_text for WM codegen prompts."""
    
    # ... 现有的 header + natural_language 构建逻辑 ...
    # base_text = header + natural_language 内容
    
    # === Implementation Priorities Section（放在策略描述后面）===
    # 设计决策：先让 LLM 理解策略整体意图，再给出细分步骤约束
    priorities = strategy.get("implementation_priorities")
    if isinstance(priorities, list) and priorities:
        priority_section = _render_priorities_section(priorities)
        base_text = base_text + "\n\n" + priority_section  # 放在后面
    
    # === 新增：expected_speedup_interval 不渲染，仅用于 WM 内部 ===
    # 该字段在 _build_action_node 中提取并设置到 action.expected_vs_baseline_factor
    
    # === 现有的 API References 和 Anti-Patterns 逻辑保持不变（放在最后）===
    api_references = strategy.get("api_references")
    if isinstance(api_references, list) and api_references:
        # ... 现有逻辑 ...
    
    return base_text
```

### 4.3 修改 `_build_action_node`

```python
def _build_action_node(...) -> dict[str, Any]:
    # ... 现有逻辑 ...
    
    expected_speedup = None
    
    # 从 expected_speedup_interval 提取 likely 值（而非 max）
    interval = strategy.get("expected_speedup_interval")
    if isinstance(interval, dict):
        expected_speedup = float(interval.get("likely", 1.0))
    
    # ... 后续逻辑使用 expected_speedup ...
```

---

## 五、策略目录修正（P0）

### 5.1 S4 策略术语修正

**修正前**：
```
Replace scalar RowMulsImpl/RowDivsImpl (which use GetValue per-row loops) with hardware RowMuls/RowDivs API calls.
The hardware RowMuls API takes a scale source tensor and broadcasts it across each row...
```

**修正后**：
```
Replace scalar RowMulsImpl/RowDivsImpl (which use GetValue per-row loops) with vectorized RowMuls/RowDivs helper functions.

Important: RowMuls and RowDivs are NOT AscendC built-in APIs. They are user-defined __aicore__ inline functions that wrap Mul/Div with BinaryRepeatParams to achieve row-wise broadcast scaling at vector width (64 fp16/cycle) instead of scalar width (1/cycle). You must define or import these functions before calling them. Reference implementation: references/row_ops_source_reference/vector_common_row_ops.h

The scalar loops process one element per cycle while the vector unit sits idle. The vectorized helper functions take a scale source tensor and broadcast it across each row of the destination tensor, processing 64 fp16 elements per cycle. Similarly RowDivs divides each row by the corresponding element from a source tensor.

This change requires: (1) In Vec2, prepare a scale tensor (expStateUb) from softmax state cache; (2) Call RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim) where the 4th argument is the number of rows, 5th is aligned dim, and 6th is actual dim; (3) Call RowDivs(oNewUb, oNewUb, sumStateUb, dealRows, dim, actualDim) for final normalization. The state tensors must be properly sized: each row's scale factor is at offset row*BRCB_NUM in the state buffer.
```

### 5.2 关键修正点

| 修正前 | 修正后 |
|--------|--------|
| "hardware RowMuls/RowDivs API calls" | "vectorized RowMuls/RowDivs helper functions" |
| "The hardware RowMuls API takes..." | "The vectorized helper functions take..." |
| 无警告 | 新增 "Important: RowMuls and RowDivs are NOT AscendC built-in APIs..." |
| 无源码引用 | 新增 "Reference implementation: references/row_ops_source_reference/..." |

---

## 六、渲染效果示例

### 6.1 S1 策略渲染结果

```
Strategy S1: Enlarged Tile Sizes (category=tiling, impact=high, difficulty=2)

Increase BLOCK_M, BLOCK_N, and BASE_K from 64 to 128. Larger tile sizes amortize loop overhead, improve L0A/L0B utilization from 25% to 50%, and reduce the number of Mmad iterations per tile. This is the most impactful single change for bandwidth-bound attention because each Mmad processes more data, reducing per-tile sync and loop overhead...
[原有完整 natural_language 描述]

=== Implementation Priority ===
Apply ONE change at a time, verify before stacking:

P0: Change tile size constants (BLOCK_M, BLOCK_N, BASE_K) from 64 to 128

P1: Adjust L1 buffer allocations for larger tile sizes
     → Requires: P0 verified

P2: Reduce loop iteration count and sync overhead
     → Requires: P0+P1 verified

Constraint: Recommended 1-2 files per round.
→ Single-file changes have 100% success rate in experiments.
→ Do NOT combine P0+P1+P2 in one round. Each round should change ONE priority level.
```

### 6.2 S4 策略渲染结果

```
Strategy S4: Vectorized RowMuls/RowDivs (category=compute, impact=high, difficulty=2)

Replace scalar RowMulsImpl/RowDivsImpl (which use GetValue per-row loops) with vectorized RowMuls/RowDivs helper functions.

Important: RowMuls and RowDivs are NOT AscendC built-in APIs. They are user-defined __aicore__ inline functions that wrap Mul/Div with BinaryRepeatParams to achieve row-wise broadcast scaling at vector width (64 fp16/cycle) instead of scalar width (1/cycle). You must define or import these functions before calling them. Reference implementation: references/row_ops_source_reference/vector_common_row_ops.h

The scalar loops process one element per cycle while the vector unit sits idle...
[完整 natural_language 描述]

=== Implementation Priority ===
Apply ONE change at a time, verify before stacking:

P0: Replace scalar RowMulsImpl loops with vectorized RowMuls helper functions

P1: Replace scalar RowDivsImpl loops with vectorized RowDivs helper functions
     → Requires: P0 verified

Constraint: Recommended 1-2 files per round.
→ Single-file changes have 100% success rate in experiments.
→ Do NOT combine P0+P1 in one round.

=== AscendC API Reference ===
- RowMuls: RowMuls不是AscendC框架内置API，而是用户自定义__aicore__ inline函数...
[API详细信息]

=== Anti-Pattern Warnings ===
[AP-S4-ap1] Do NOT use Duplicate+Mul替代RowMuls...
[Anti-Pattern列表]
```

### 6.3 Section 顺序设计决策

| Section | 位置 | 理由 |
|--------|------|------|
| Strategy Header + Natural Language | **最前面** | 先让 LLM 理解策略整体意图和背景 |
| Implementation Priority | **中间** | 给出细分步骤约束，引导 LLM 单步骤变更 |
| API Reference | **后面** | 提供具体 API 签名，帮助 LLM 正确调用 |
| Anti-Pattern Warnings | **最后** | 提示不等价陷阱，防止 LLM 自探索失败 |

---

## 七、设计决策记录

### 7.1 为什么不包含 `files` 字段

**问题**：策略是通用的，不能针对某个算子预写具体文件名。

**决策**：策略不指定文件范围，仅描述变更类型（如 "Change tile size constants"），LLM 自行判断涉及哪些文件。

### 7.2 为什么移除单步骤 `expected_speedup`

**问题**：单步骤预期加速也是算子特定的（FlashAttention 的 P0 预期 1.5x，MatMul 可能不同）。

**决策**：移除 `implementation_priorities[].expected_speedup`，仅保留整体 `expected_speedup_interval`（不渲染，供 WM 内部使用）。

### 7.3 为什么 `expected_speedup_interval` 不渲染

**问题**：整体加速预期同样是算子特定的，渲染给 LLM 可能造成误导。

**决策**：保留字段但不渲染到 prompt，仅用于 WM 内部决策（如判断是否继续尝试补足差距）。WM 可通过其他方式（如 Task 注入）获取算子特定的预期值。

### 7.4 文件数约束为何是"建议"而非"硬限制"

**问题**：某些优先级步骤可能涉及 3-4 个文件，硬约束会导致无法完成。

**决策**：约束表述为 "Recommended 1-2 files per round"，不强制限制，但明确提示"单文件变更成功率最高"。

---

## 八、实施范围

### 修改文件列表

| 文件 | 修改内容 |
|------|---------|
| `k_search/kernel_generators/strategy_injection.py` | 新增 `_render_priorities_section`，修改 `render_strategy_as_action_text`，修改 `_build_action_node` |
| `strategies/mqa_strategies_catalog.json` | 为策略 S1-S12 添加 `implementation_priorities` 和 `expected_speedup_interval` 字段，修正 S4 的 `natural_language` 术语 |

### 不修改内容

- 不修改 `api_references` 和 `anti_patterns` 现有渲染逻辑
- 不修改框架层代码（WM cycle 管理、连续失败自动回退等）

---

## 九、预期效果

| 问题 | 改进措施 | 预期效果 |
|------|---------|---------|
| 多维叠加变更失败率高 | Implementation Priority 提示 | 减少 R4-R8 类型失败，从 5 轮浪费缩减为 2-3 轮 |
| 术语误导导致 API 误用 | S4 natural_language 修正 | 消除 R1/R2 类型编译失败 |
| WM 过度追求补足差距 | expected_speedup_interval (likely 值) | WM 遇到 likely 值时不再浪费 rounds |

**综合预期**：通过率从 33% 提升到约 50-60%（策略层改进），配合框架层改进可达 75%。