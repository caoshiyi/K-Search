# AscendC API 信息注入策略 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 K-Search 策略提供 AscendC API 精确信息（签名摘要 + 文档路径 + anti-pattern），消除因LLM猜测API用法导致的编译失败和精度校验失败。

**Architecture:** 策略JSON新增 `api_references` 字段（api_name + doc_path + summary + anti_patterns）。`strategy_injection.py` 渲染时将摘要和文档路径追加到 action_text。anti-pattern 支持运行时从失败轮次自动学习追加。知识库拷贝到项目内，缺失API创建占位文件。

**Tech Stack:** Python, JSON, pytest

---

## File Structure

| 文件 | 责任 | 状态 |
|------|------|------|
| `K-Search/references/api_reference_docs/` | AscendC API参考文档知识库（864个MD文件） | 新增（拷贝） |
| `K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md` | RowMuls API占位文档 | 新增 |
| `K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md` | RowDivs API占位文档 | 新增 |
| `K-Search/strategies/mqa_strategies_catalog.json` | 策略catalog，新增api_references字段 | 修改 |
| `K-Search/k_search/kernel_generators/strategy_injection.py` | 策略渲染，追加API Reference和Anti-Pattern section | 修改 |
| `K-Search/k_search/kernel_generators/world_model_manager.py` | WM管理，新增anti-pattern学习逻辑 | 修改 |
| `K-Search/tests/kernel_generators/test_strategy_injection.py` | strategy_injection 单元测试 | 新增 |
| `K-Search/tests/kernel_generators/test_strategy_api_injection.py` | API注入集成测试 | 新增 |

---

### Task 1: 拷贝 AscendC API 知识库到 K-Search 项目

**Files:**
- Create: `K-Search/references/` (整个目录)

- [ ] **Step 1: 拷贝references目录**

```bash
cp -r /home/developer/.claude/skills/ascendc-dev-knowledge/references/ /mnt/workspace/K-Search/references/
```

- [ ] **Step 2: 验证拷贝完整性**

```bash
# 验证文件数量
find /mnt/workspace/K-Search/references/api_reference_docs/ -name "*.md" | wc -l
# 预期: 864

# 验证关键文件存在
ls /mnt/workspace/K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/Muls.md
ls /mnt/workspace/K-Search/references/api_reference_docs/基础API/矢量计算/数据填充/Brcb.md
ls /mnt/workspace/K-Search/references/api_reference_docs/基础数据结构/LocalTensor/GetValue.md
ls /mnt/workspace/K-Search/references/api-reference.md
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/workspace/K-Search
git add references/
git commit -m "feat: add AscendC API reference knowledge base to K-Search project"
```

---

### Task 2: 创建 RowMuls 和 RowDivs 占位文档

**Files:**
- Create: `K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md`
- Create: `K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md`

- [ ] **Step 1: 创建 RowMuls.md 占位文件**

```markdown
# RowMuls

> ⚠️ 此文档为占位文件，内容待补充完整。当前信息来源于MQA优化实验的问题分析。

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明
逐行向量乘法：将每行的元素与scale张量中对应行的标量因子相乘。这是AscendC的硬件级专有指令，用于替代标量循环（GetValue + Muls）的行级缩放操作。

#### 函数原型

```
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dst, LocalTensor<T> src,
                                LocalTensor<T> scale, int32_t dealRows,
                                int32_t cols, int32_t actualCols)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目标张量，存储逐行乘法结果 |
| src | 输入 | 源张量 |
| scale | 输入 | 缩放因子张量（每行一个因子，位于 offset row*BRCB_NUM 处） |
| dealRows | 输入 | 处理行数 |
| cols | 输入 | 对齐列数 |
| actualCols | 输入 | 实际有效列数（用于padding区域处理） |

#### 关键约束
- scale参数是 LocalTensor<T>，不是 float 标量。每行的缩放因子存储在 offset row*BRCB_NUM 处。
- dealRows 是行数参数，不是列数。
- actualCols 用于正确处理padding区域：有效区域使用actualCols宽度计算，padding区域不受影响。
- BRCB_NUM = 32/sizeof(T)，即 fp32 时为 8。

#### Anti-Pattern（已知误用模式）

1. **Do NOT use Duplicate(scalar,dimAlign)+Mul(dst,src,buf,dimAlign) 作为替代**
   - Reason: Duplicate仅填充dim个有效元素为scalar值，padding区域(dimAlign-dim)保留垃圾值不清零。Mul操作整个dimAlign宽度包括padding区域的垃圾值。且Duplicate和Mul在同一PIPE_V上执行，无PipeBarrier分隔，存在RAW冒险。额外赋值步骤引入精度误差累积。
   - 正确替代：直接使用 RowMuls API。

2. **Do NOT use LocalTensor[] 下标运算符作为 float 标量参数传入 Muls**
   - Reason: LocalTensor[] 返回子张量视图 LocalTensor<T>，不是标量 T。使用 GetValue(tensor, index) 提取标量值。

3. **Do NOT use Brcb(dst,src,count) 只传3个参数**
   - Reason: AscendC Brcb 需要4个参数: Brcb(dst, src, repeatTimes, dstStride)。正确的行级广播乘法替代是 RowMuls API，不是 Brcb+Mulcs 组合。

#### 调用示例

```cpp
// softmax state rescale using RowMuls
LocalTensor<float> expStateUb = expStateBuf_.Get<float>();
RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim);
```
```

- [ ] **Step 2: 创建 RowDivs.md 占位文件**

```markdown
# RowDivs

> ⚠️ 此文档为占位文件，内容待补充完整。当前信息来源于MQA优化实验的问题分析。

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明
逐行向量除法：将每行的元素与scale张量中对应行的标量因子相除。与RowMuls对称，用于softmax最终归一化等场景。

#### 函数原型

```
template <typename T>
__aicore__ inline void RowDivs(LocalTensor<T> dst, LocalTensor<T> src,
                                LocalTensor<T> scale, int32_t dealRows,
                                int32_t cols, int32_t actualCols)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst | 输出 | 目标张量，存储逐行除法结果 |
| src | 输入 | 源张量 |
| scale | 输入 | 除法因子张量（每行一个因子，位于 offset row*BRCB_NUM 处） |
| dealRows | 输入 | 处理行数 |
| cols | 输入 | 对齐列数 |
| actualCols | 输入 | 实际有效列数（用于padding区域处理） |

#### 关键约束
- scale参数是 LocalTensor<T>，不是 float 标量。
- BRCB_NUM = 32/sizeof(T)，即 fp32 时为 8。
- actualCols 用于正确处理padding区域。

#### 调用示例

```cpp
// softmax final normalization using RowDivs
LocalTensor<float> sumStateUb = sumStateBuf_.Get<float>();
RowDivs(oNewUb, oNewUb, sumStateUb, dealRows, dim, actualDim);
```
```

- [ ] **Step 3: 验证占位文件存在**

```bash
ls /mnt/workspace/K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md
ls /mnt/workspace/K-Search/references/api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md
```

- [ ] **Step 4: Commit**

```bash
cd /mnt/workspace/K-Search
git add references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md
git add references/api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md
git commit -m "feat: add placeholder docs for RowMuls and RowDivs APIs"
```

---

### Task 3: 为策略catalog JSON新增 api_references 字段

**Files:**
- Modify: `K-Search/strategies/mqa_strategies_catalog.json`

- [ ] **Step 1: 为 S4 策略添加 api_references**

在 S4 策略dict中（当前最后一个字段 `dsl` 之后），新增 `api_references` 数组。S4涉及4个API：RowMuls、RowDivs、GetValue、Duplicate、Brcb。

需添加的JSON片段（位于S4策略dict末尾，`dsl` 字段之后）：

```json
"api_references": [
  {
    "api_name": "RowMuls",
    "doc_path": "api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md",
    "summary": "RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32). scale参数是LocalTensor而非float标量。dealRows是行数，actualCols是实际列数（含padding处理）。BRCB_NUM=32/sizeof(T)=8 for fp32。",
    "anti_patterns": [
      {
        "id": "ap1",
        "pattern": "Duplicate(scalar,dimAlign)+Mul(dst,src,buf,dimAlign) 替代 RowMuls",
        "reason": "PIPE_V RAW冒险(Duplicate写buf后Mul立即读同一PIPE_V无PipeBarrier) + padding区域垃圾值不清零(Duplicate仅填充dim个位置, Mul操作整个dimAlign宽度) + 累积精度误差(1次额外Duplicate赋值舍入ε)",
        "source": "round_9_failure",
        "discovered_at": 9
      },
      {
        "id": "ap2",
        "pattern": "LocalTensor[] 下标运算符作为 float 标量参数传入 Muls",
        "reason": "LocalTensor[] 返回子张量视图 LocalTensor<T>，不是标量 T。使用 GetValue(tensor, index) 提取标量值。",
        "source": "round_2_failure",
        "discovered_at": 2
      },
      {
        "id": "ap3",
        "pattern": "Brcb(dst,src,count) 只传3个参数替代 RowMuls",
        "reason": "AscendC Brcb 需要4个参数: Brcb(dst,src,repeatTimes,dstStride)。Mulcs在CANN 9.0中不存在(应为Muls)。正确的行级广播乘法是RowMuls API。",
        "source": "round_1_failure",
        "discovered_at": 1
      }
    ]
  },
  {
    "api_name": "RowDivs",
    "doc_path": "api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md",
    "summary": "RowDivs(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32). 与RowMuls对称的逐行除法。scale参数是LocalTensor而非float标量。",
    "anti_patterns": []
  },
  {
    "api_name": "GetValue",
    "doc_path": "api_reference_docs/基础数据结构/LocalTensor/GetValue.md",
    "summary": "GetValue(index: uint32) → PrimType. 获取LocalTensor指定索引的标量值。仅在VECIN/VECCALC/VECOUT位置支持。用于从LocalTensor提取float标量（而非[]下标运算符）。",
    "anti_patterns": []
  },
  {
    "api_name": "Duplicate",
    "doc_path": "api_reference_docs/基础API/矢量计算/数据填充/Duplicate.md",
    "summary": "Duplicate(dst, src, count). 广播复制：将src单个元素复制count次到dst。仅填充count个有效元素，padding区域(dimAlign-dim)保留垃圾值不清零。",
    "anti_patterns": []
  },
  {
    "api_name": "Brcb",
    "doc_path": "api_reference_docs/基础API/矢量计算/数据填充/Brcb.md",
    "summary": "Brcb(dst, src, repeatTimes, dstStride). 需要4参数，不可只传3个。用于行级广播复制。",
    "anti_patterns": []
  }
]
```

- [ ] **Step 2: 验证JSON格式正确**

```bash
cd /mnt/workspace/K-Search
python3 -c "import json; data=json.load(open('strategies/mqa_strategies_catalog.json')); s4=data['strategy_catalog'][3]; refs=s4.get('api_references',[]); print(f'S4 api_references count: {len(refs)}'); print(f'APIs: {[r[\"api_name\"] for r in refs]}')"
# 预期: S4 api_references count: 5, APIs: ['RowMuls', 'RowDivs', 'GetValue', 'Duplicate', 'Brcb']
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/workspace/K-Search
git add strategies/mqa_strategies_catalog.json
git commit -m "feat: add api_references field to S4 strategy with summaries and anti-patterns"
```

---

### Task 4: 编写 strategy_injection.py 的 API 渲染逻辑 — 先写测试

**Files:**
- Create: `K-Search/tests/kernel_generators/test_strategy_injection.py`

- [ ] **Step 1: 写测试文件**

```python
"""Tests for strategy_injection module, including API reference rendering."""

import json
import pytest
from k_search.kernel_generators.strategy_injection import (
    load_strategy_catalog,
    render_strategy_as_action_text,
    _render_api_references_section,
    _render_anti_patterns_section,
)


# --- Minimal strategy fixtures ---

def _make_strategy_with_api_refs():
    """A strategy with api_references for testing."""
    return {
        "id": "S4",
        "name": "Vectorized RowOps",
        "category": "compute",
        "difficulty": 2,
        "impact": "high",
        "natural_language": "Replace scalar RowMulsImpl with hardware RowMuls API calls.",
        "api_references": [
            {
                "api_name": "RowMuls",
                "doc_path": "api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md",
                "summary": "RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32). scale is LocalTensor not float scalar.",
                "anti_patterns": [
                    {
                        "id": "ap1",
                        "pattern": "Duplicate+Mul as substitute for RowMuls",
                        "reason": "PIPE_V RAW hazard + padding pollution + precision drift",
                        "source": "round_9_failure",
                        "discovered_at": 9,
                    }
                ],
            },
            {
                "api_name": "GetValue",
                "doc_path": "api_reference_docs/基础数据结构/LocalTensor/GetValue.md",
                "summary": "GetValue(index: uint32) → PrimType. Extract scalar from LocalTensor.",
                "anti_patterns": [],
            },
        ],
    }


def _make_strategy_without_api_refs():
    """A strategy without api_references for testing."""
    return {
        "id": "S1",
        "name": "Enlarged Tiles",
        "category": "tiling",
        "difficulty": 2,
        "impact": "high",
        "natural_language": "Increase BLOCK_M, BLOCK_N, and BASE_K from 64 to 128.",
    }


# --- Test: render with no api_references ---

def test_render_strategy_without_api_refs():
    """Strategies without api_references should render unchanged."""
    strategy = _make_strategy_without_api_refs()
    result = render_strategy_as_action_text(strategy, "natural_language")
    assert "Strategy S1: Enlarged Tiles" in result
    assert "AscendC API Reference" not in result
    assert "Anti-Pattern Warnings" not in result
    assert "Increase BLOCK_M" in result


# --- Test: render with api_references ---

def test_render_strategy_with_api_refs():
    """Strategies with api_references should include API Reference and Anti-Pattern sections."""
    strategy = _make_strategy_with_api_refs()
    result = render_strategy_as_action_text(strategy, "natural_language")
    assert "Strategy S4: Vectorized RowOps" in result
    assert "=== AscendC API Reference ===" in result
    assert "RowMuls" in result
    assert "RowMuls(dst: LocalTensor<T>" in result
    assert "scale is LocalTensor not float scalar" in result
    assert "api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md" in result
    assert "GetValue" in result
    assert "=== Anti-Pattern Warnings ===" in result
    assert "[AP-S4-ap1]" in result
    assert "Duplicate+Mul as substitute" in result
    assert "PIPE_V RAW hazard" in result


# --- Test: anti-pattern section only appears when anti_patterns exist ---

def test_render_api_refs_with_empty_anti_patterns():
    """Anti-Pattern section should not appear when all anti_patterns arrays are empty."""
    strategy = _make_strategy_with_api_refs()
    # Clear all anti_patterns
    for ref in strategy["api_references"]:
        ref["anti_patterns"] = []
    result = render_strategy_as_action_text(strategy, "natural_language")
    assert "=== AscendC API Reference ===" in result
    assert "=== Anti-Pattern Warnings ===" not in result


# --- Test: doc_path hint for reading ---

def test_render_includes_doc_path_hint():
    """Rendered text should include a hint about reading doc files for more details."""
    strategy = _make_strategy_with_api_refs()
    result = render_strategy_as_action_text(strategy, "natural_language")
    assert "If you need more details" in result
    assert "read the doc files" in result


# --- Test: _render_api_references_section standalone ---

def test_render_api_references_section_format():
    """Test the standalone API references rendering function."""
    api_refs = _make_strategy_with_api_refs()["api_references"]
    result = _render_api_references_section(api_refs)
    assert "RowMuls" in result
    assert "Signature:" in result or "RowMuls(dst:" in result
    assert "Full doc:" in result
    assert "api_reference_docs/" in result
    assert "GetValue" in result


# --- Test: _render_anti_patterns_section standalone ---

def test_render_anti_patterns_section_format():
    """Test the standalone anti-patterns rendering function."""
    api_refs = _make_strategy_with_api_refs()["api_references"]
    strategy_id = "S4"
    result = _render_anti_patterns_section(api_refs, strategy_id)
    assert "[AP-S4-ap1]" in result
    assert "Duplicate+Mul as substitute for RowMuls" in result
    assert "PIPE_V RAW hazard" in result
    assert "round_9_failure" in result


# --- Test: anti-pattern id numbering ---

def test_anti_pattern_id_format():
    """Anti-pattern IDs should use [AP-{strategy_id}-{anti_pattern.id}] format."""
    strategy = _make_strategy_with_api_refs()
    result = render_strategy_as_action_text(strategy, "natural_language")
    assert "[AP-S4-ap1]" in result
```

- [ ] **Step 2: Run tests to verify they fail (functions not yet implemented)**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/test_strategy_injection.py -v 2>&1 | head -30
```

预期：大部分测试FAIL，因为 `_render_api_references_section` 和 `_render_anti_patterns_section` 函数以及 `render_strategy_as_action_text` 的API注入逻辑尚未实现。

- [ ] **Step 3: Commit**

```bash
cd /mnt/workspace/K-Search
git add tests/kernel_generators/test_strategy_injection.py
git commit -m "test: add unit tests for strategy API reference rendering"
```

---

### Task 5: 实现 strategy_injection.py 的 API 渲染逻辑

**Files:**
- Modify: `K-Search/k_search/kernel_generators/strategy_injection.py`

- [ ] **Step 1: 添加 `_render_api_references_section` 函数**

在 `strategy_injection.py` 的 `render_strategy_as_action_text` 函数之后（第88行 `raise ValueError` 之后），添加以下两个新函数：

```python
def _render_api_references_section(api_references: list[dict[str, Any]]) -> str:
    """Render api_references into an API Reference section for prompt injection.

    Each API shows: summary (signature + key constraints) + doc_path for deep reading.
    """
    if not api_references:
        return ""

    lines: list[str] = []
    for ref in api_references:
        api_name = str(ref.get("api_name", "") or "").strip()
        summary = str(ref.get("summary", "") or "").strip()
        doc_path = str(ref.get("doc_path", "") or "").strip()
        if not api_name:
            continue
        lines.append(f"{api_name}:")
        if summary:
            lines.append(f"  {summary}")
        if doc_path:
            lines.append(f"  Full doc: {doc_path}")
        lines.append("")  # blank line between APIs

    # Add hint for deep reading
    lines.append("If you need more details (supported types, examples, alignment constraints), read the doc files at the paths above.")

    return "\n".join(lines)


def _render_anti_patterns_section(
    api_references: list[dict[str, Any]],
    strategy_id: str,
) -> str:
    """Render anti-pattern warnings from all api_references into a unified section.

    Anti-pattern IDs use [AP-{strategy_id}-{anti_pattern.id}] format.
    Only renders if at least one anti_pattern exists across all api_references.
    """
    all_anti_patterns: list[tuple[str, dict[str, Any]]] = []
    for ref in api_references:
        anti_patterns = ref.get("anti_patterns", [])
        if not isinstance(anti_patterns, list):
            continue
        for ap in anti_patterns:
            if isinstance(ap, dict):
                all_anti_patterns.append((str(ref.get("api_name", "")), ap))

    if not all_anti_patterns:
        return ""

    lines: list[str] = []
    for api_name, ap in all_anti_patterns:
        ap_id = str(ap.get("id", "unknown")).strip()
        pattern = str(ap.get("pattern", "") or "").strip()
        reason = str(ap.get("reason", "") or "").strip()
        source = str(ap.get("source", "") or "").strip()
        discovered_at = ap.get("discovered_at")
        tag = f"[AP-{strategy_id}-{ap_id}]"
        lines.append(f"{tag} Do NOT use {pattern}.")
        if reason:
            lines.append(f"  Reason: {reason}")
        if source:
            at_str = f" (Discovered from {source})" if not discovered_at else f" (Discovered from {source}, round {discovered_at})"
            lines.append(f"  {at_str}")
        lines.append("")  # blank line between warnings

    return "\n".join(lines)
```

- [ ] **Step 2: 修改 `render_strategy_as_action_text` 函数，追加API信息渲染**

修改 `render_strategy_as_action_text` 函数（当前在第36-88行）。需要在函数末尾（各 form 的 return 之后），增加一个统一的API信息追加逻辑。

修改方式：将函数体改为先渲染基础文本，然后追加API sections。替换整个函数体：

```python
def render_strategy_as_action_text(
    strategy: dict[str, Any],
    form: str = "natural_language",
) -> str:
    """Render a single strategy into action_text for WM codegen prompts.

    Args:
        strategy: A strategy dict from the catalog (must have id, name, and
                  at least one of natural_language/structured_params/dsl).
        form: One of "natural_language", "structured_params", "dsl".

    Returns:
        A string suitable for injection as action_text in WM action prompts.
    """
    form = form.strip().lower()
    if form not in STRATEGY_FORMS:
        raise ValueError(f"Invalid strategy form '{form}'; must be one of {STRATEGY_FORMS}")

    sid = strategy.get("id", "unknown")
    name = strategy.get("name", "unknown")
    category = strategy.get("category", "unknown")
    difficulty = strategy.get("difficulty", 3)
    impact = strategy.get("impact", "medium")

    header = f"Strategy {sid}: {name} (category={category}, impact={impact}, difficulty={difficulty})\n"

    # Render base strategy text based on form
    if form == "natural_language":
        text = str(strategy.get("natural_language", "") or "").strip()
        if not text:
            raise ValueError(f"Strategy {sid} has no natural_language field")
        base_text = header + text

    elif form == "structured_params":
        params = strategy.get("structured_params")
        if not isinstance(params, dict):
            raise ValueError(f"Strategy {sid} has no structured_params dict")
        intro = (
            f"Apply this optimization using the following structured parameters.\n"
            f"Each parameter specifies the exact change to make, with constraints and expected impact.\n"
        )
        base_text = header + intro + json.dumps(params, indent=2, ensure_ascii=False)

    elif form == "dsl":
        text = str(strategy.get("dsl", "") or "").strip()
        if not text:
            raise ValueError(f"Strategy {sid} has no dsl field")
        intro = (
            f"Apply this optimization expressed as a domain-specific language (DSL) specification.\n"
            f"The DSL defines the exact transformation, constraints, and expected outcome.\n"
        )
        base_text = header + intro + text

    else:
        raise ValueError(f"Unhandled form: {form}")

    # Append API Reference and Anti-Pattern sections if api_references exists
    api_references = strategy.get("api_references", [])
    if isinstance(api_references, list) and api_references:
        api_section = _render_api_references_section(api_references)
        if api_section:
            base_text += "\n\n=== AscendC API Reference ===\n\n" + api_section

        ap_section = _render_anti_patterns_section(api_references, str(sid))
        if ap_section:
            base_text += "\n\n=== Anti-Pattern Warnings (DO NOT do the following) ===\n\n" + ap_section

    return base_text
```

- [ ] **Step 3: 运行测试验证通过**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/test_strategy_injection.py -v
```

预期：所有测试PASS。

- [ ] **Step 4: Commit**

```bash
cd /mnt/workspace/K-Search
git add k_search/kernel_generators/strategy_injection.py
git commit -m "feat: add API reference and anti-pattern rendering to strategy_injection"
```

---

### Task 6: 编写 anti-pattern 学习逻辑 — 先写测试

**Files:**
- Create: `K-Search/tests/kernel_generators/test_strategy_api_injection.py`

- [ ] **Step 1: 写测试文件**

```python
"""Tests for anti-pattern learning from optimization failures."""

import json
import pytest
from pathlib import Path
from k_search.kernel_generators.strategy_injection import (
    learn_anti_pattern_from_failure,
    _text_similarity_ratio,
)


# --- Test: text similarity ---

def test_text_similarity_identical():
    assert _text_similarity_ratio("abc", "abc") == 1.0


def test_text_similarity_similar():
    ratio = _text_similarity_ratio(
        "Duplicate+Mul as substitute for RowMuls",
        "Duplicate and Mul as substitute for RowMuls"
    )
    assert ratio > 0.8


def test_text_similarity_different():
    ratio = _text_similarity_ratio("RowMuls API", "LocalTensor subscript")
    assert ratio < 0.5


# --- Test: learn anti-pattern from failure ---

def test_learn_anti_pattern_basic():
    """Test basic anti-pattern learning from a failure."""
    strategy_catalog = [
        {
            "id": "S4",
            "name": "Vectorized RowOps",
            "natural_language": "Replace scalar RowMulsImpl...",
            "api_references": [
                {
                    "api_name": "RowMuls",
                    "doc_path": "api_reference_docs/.../RowMuls.md",
                    "summary": "RowMuls signature...",
                    "anti_patterns": [
                        {
                            "id": "ap1",
                            "pattern": "Duplicate+Mul substitute",
                            "reason": "RAW hazard",
                            "source": "round_9_failure",
                            "discovered_at": 9,
                        }
                    ],
                },
            ],
        }
    ]

    new_ap = {
        "pattern": "Brcb 3-param call instead of RowMuls",
        "reason": "Brcb requires 4 arguments, not 3",
        "api_name_hint": "RowMuls",
    }

    updated = learn_anti_pattern_from_failure(
        strategy_catalog=strategy_catalog,
        strategy_id="S4",
        api_name="RowMuls",
        new_anti_pattern=new_ap,
        round_index=12,
    )

    # Find S4 and RowMuls in updated catalog
    s4 = updated[3] if len(updated) > 3 else None
    # strategy_id S4 should be at index 3 (S1=0, S2=1, S3=2, S4=3)
    # But for this test, S4 is the only strategy, so index 0
    s4 = updated[0]
    row_muls_refs = [r for r in s4["api_references"] if r["api_name"] == "RowMuls"]
    assert len(row_muls_refs) == 1
    anti_patterns = row_muls_refs[0]["anti_patterns"]
    assert len(anti_patterns) == 2  # original ap1 + new learned pattern
    new_learned = anti_patterns[1]
    assert new_learned["pattern"] == "Brcb 3-param call instead of RowMuls"
    assert new_learned["reason"] == "Brcb requires 4 arguments, not 3"
    assert new_learned["source"] == "round_12_failure"
    assert new_learned["discovered_at"] == 12


# --- Test: dedup anti-pattern (skip if similar already exists) ---

def test_learn_anti_pattern_dedup():
    """Similar anti-patterns should not be added twice."""
    strategy_catalog = [
        {
            "id": "S4",
            "name": "Vectorized RowOps",
            "natural_language": "...",
            "api_references": [
                {
                    "api_name": "RowMuls",
                    "doc_path": "...",
                    "summary": "...",
                    "anti_patterns": [
                        {
                            "id": "ap1",
                            "pattern": "Duplicate+Mul as substitute for RowMuls",
                            "reason": "RAW hazard",
                            "source": "round_9_failure",
                            "discovered_at": 9,
                        }
                    ],
                },
            ],
        }
    ]

    # Try to add a very similar pattern
    new_ap = {
        "pattern": "Duplicate and Mul as substitute for RowMuls",
        "reason": "Different reason but similar pattern",
        "api_name_hint": "RowMuls",
    }

    updated = learn_anti_pattern_from_failure(
        strategy_catalog=strategy_catalog,
        strategy_id="S4",
        api_name="RowMuls",
        new_anti_pattern=new_ap,
        round_index=13,
    )

    s4 = updated[0]
    row_muls_refs = [r for r in s4["api_references"] if r["api_name"] == "RowMuls"]
    # Should NOT have added because similarity > 0.8
    assert len(row_muls_refs[0]["anti_patterns"]) == 1  # still only the original


# --- Test: auto-generate id for new anti-pattern ---

def test_learn_anti_pattern_auto_id():
    """New anti-pattern should get an auto-generated id."""
    strategy_catalog = [
        {
            "id": "S4",
            "name": "Vectorized RowOps",
            "natural_language": "...",
            "api_references": [
                {
                    "api_name": "RowMuls",
                    "doc_path": "...",
                    "summary": "...",
                    "anti_patterns": [
                        {"id": "ap1", "pattern": "...", "reason": "..."},
                        {"id": "ap2", "pattern": "...", "reason": "..."},
                    ],
                },
            ],
        }
    ]

    new_ap = {
        "pattern": "New unique pattern",
        "reason": "New unique reason",
        "api_name_hint": "RowMuls",
    }

    updated = learn_anti_pattern_from_failure(
        strategy_catalog=strategy_catalog,
        strategy_id="S4",
        api_name="RowMuls",
        new_anti_pattern=new_ap,
        round_index=14,
    )

    s4 = updated[0]
    row_muls_refs = [r for r in s4["api_references"] if r["api_name"] == "RowMuls"]
    new_learned = row_muls_refs[0]["anti_patterns"][2]
    assert new_learned["id"] == "ap3"  # auto-generated next id


# --- Test: strategy_id not found returns unchanged catalog ---

def test_learn_anti_pattern_strategy_not_found():
    """If strategy_id is not found, return catalog unchanged."""
    strategy_catalog = [{"id": "S1", "name": "...", "natural_language": "..."}]
    new_ap = {"pattern": "...", "reason": "...", "api_name_hint": "RowMuls"}

    updated = learn_anti_pattern_from_failure(
        strategy_catalog=strategy_catalog,
        strategy_id="S99",
        api_name="RowMuls",
        new_anti_pattern=new_ap,
        round_index=1,
    )

    assert updated == strategy_catalog  # unchanged


# --- Test: api_name not found in strategy's api_references ---

def test_learn_anti_pattern_api_not_found():
    """If api_name is not found in strategy's api_references, return catalog unchanged."""
    strategy_catalog = [
        {
            "id": "S4",
            "name": "...",
            "natural_language": "...",
            "api_references": [
                {"api_name": "GetValue", "doc_path": "...", "summary": "...", "anti_patterns": []},
            ],
        }
    ]
    new_ap = {"pattern": "...", "reason": "...", "api_name_hint": "RowMuls"}

    updated = learn_anti_pattern_from_failure(
        strategy_catalog=strategy_catalog,
        strategy_id="S4",
        api_name="RowMuls",
        new_anti_pattern=new_ap,
        round_index=1,
    )

    assert updated == strategy_catalog  # unchanged
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/test_strategy_api_injection.py -v 2>&1 | head -30
```

预期：FAIL，因为 `learn_anti_pattern_from_failure` 和 `_text_similarity_ratio` 函数尚未实现。

- [ ] **Step 3: Commit**

```bash
cd /mnt/workspace/K-Search
git add tests/kernel_generators/test_strategy_api_injection.py
git commit -m "test: add unit tests for anti-pattern learning from failures"
```

---

### Task 7: 实现 anti-pattern 学习逻辑

**Files:**
- Modify: `K-Search/k_search/kernel_generators/strategy_injection.py`

- [ ] **Step 1: 添加 `_text_similarity_ratio` 函数**

在 `strategy_injection.py` 中，`_render_anti_patterns_section` 函数之后，添加：

```python
def _text_similarity_ratio(text_a: str, text_b: str) -> float:
    """Compute simple word-level overlap ratio between two strings.

    Returns a float in [0, 1] where 1.0 means identical.
    Used for deduplication: patterns with similarity > 0.8 are considered duplicates.
    """
    words_a = set(str(text_a or "").lower().split())
    words_b = set(str(text_b or "").lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
```

- [ ] **Step 2: 添加 `learn_anti_pattern_from_failure` 函数**

```python
def learn_anti_pattern_from_failure(
    strategy_catalog: list[dict[str, Any]],
    strategy_id: str,
    api_name: str,
    new_anti_pattern: dict[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    """Learn a new anti-pattern from an optimization failure and append it to the catalog.

    Args:
        strategy_catalog: The strategy catalog list (will be modified in-place and returned).
        strategy_id: The strategy id to find (e.g., "S4").
        api_name: The API name to match within the strategy's api_references.
        new_anti_pattern: Dict with at least "pattern" and "reason" keys.
            May also contain "api_name_hint" (ignored, api_name param is used instead).
        round_index: The optimization round where this failure was discovered.

    Returns:
        The updated strategy catalog (same list, modified in-place).
        If strategy_id or api_name is not found, returns catalog unchanged.
    """
    # Find the strategy
    strategy = None
    for s in strategy_catalog:
        if str(s.get("id", "") or "").strip() == str(strategy_id).strip():
            strategy = s
            break

    if strategy is None:
        return strategy_catalog

    # Find the api_reference
    api_refs = strategy.get("api_references", [])
    if not isinstance(api_refs, list):
        return strategy_catalog

    target_ref = None
    for ref in api_refs:
        if str(ref.get("api_name", "") or "").strip() == str(api_name).strip():
            target_ref = ref
            break

    if target_ref is None:
        return strategy_catalog

    # Get existing anti_patterns
    existing_aps = target_ref.get("anti_patterns", [])
    if not isinstance(existing_aps, list):
        existing_aps = []
        target_ref["anti_patterns"] = existing_aps

    # Dedup: check if similar pattern already exists
    new_pattern_text = str(new_anti_pattern.get("pattern", "") or "").strip()
    for existing_ap in existing_aps:
        existing_pattern = str(existing_ap.get("pattern", "") or "").strip()
        if _text_similarity_ratio(new_pattern_text, existing_pattern) > 0.8:
            # Similar pattern already exists, skip
            return strategy_catalog

    # Auto-generate id: find the next available id number
    max_id_num = 0
    for existing_ap in existing_aps:
        existing_id = str(existing_ap.get("id", "") or "").strip()
        if existing_id.startswith("ap"):
            try:
                num = int(existing_id[2:])
                if num > max_id_num:
                    max_id_num = num
            except ValueError:
                pass
    new_id = f"ap{max_id_num + 1}"

    # Build the new anti_pattern entry
    learned_ap = {
        "id": new_id,
        "pattern": new_pattern_text,
        "reason": str(new_anti_pattern.get("reason", "") or "").strip(),
        "source": f"round_{round_index}_failure",
        "discovered_at": round_index,
    }

    existing_aps.append(learned_ap)
    return strategy_catalog
```

- [ ] **Step 3: 运行测试验证通过**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/test_strategy_api_injection.py -v
```

预期：所有测试PASS。

- [ ] **Step 4: Commit**

```bash
cd /mnt/workspace/K-Search
git add k_search/kernel_generators/strategy_injection.py
git commit -m "feat: add anti-pattern learning function with dedup and auto-id"
```

---

### Task 8: 在 world_model_manager.py 中集成 anti-pattern 学习触发

**Files:**
- Modify: `K-Search/k_search/kernel_generators/world_model_manager.py`

- [ ] **Step 1: 在 WorldModelManager 类中添加 strategy_catalog_path 属性和 anti-pattern 学习方法**

在 `WorldModelManager.__init__` 方法（第82-99行）中，新增 `strategy_catalog_path` 参数和初始化：

在 `__init__` 的参数列表中添加 `strategy_catalog_path: str | None = None`，在 `self._world_models` 之后添加 `self._strategy_catalog_path = strategy_catalog_path`。

```python
def __init__(
    self,
    *,
    llm_call: LLMCall,
    target_gpu: str,
    language: str,
    config: WorldModelConfig | None = None,
    strategy_catalog_path: str | None = None,
):
    self._llm_call = llm_call
    self._target_gpu = target_gpu
    self._language = language
    self._cfg = config or WorldModelConfig()
    self._world_models: Dict[str, str] = {}
    self._strategy_catalog_path = strategy_catalog_path
    # Debug/reporting: last edit-ops application summary (applied vs skipped).
    self._last_apply_ops_report: dict | None = None
```

- [ ] **Step 2: 添加 `learn_anti_pattern_from_round_failure` 方法**

在 `WorldModelManager` 类中（建议在 `note_action_too_hard` 方法之后，约第1290行附近），添加新方法：

```python
    def learn_anti_pattern_from_round_failure(
        self,
        *,
        definition_name: str,
        round_index: int,
        failure_reason: str | None = None,
        changed_files: list[str] | None = None,
    ) -> bool:
        """Attempt to learn a new anti-pattern from a round failure.

        This is a lightweight, deterministic trigger that checks if:
        1. A strategy_catalog_path is configured
        2. The failure involves an API misuse (based on failure_reason keywords)
        3. The active leaf node's strategy has api_references

        If conditions are met, uses the LLM to analyze the failure and generate
        an anti_pattern, then writes it back to the catalog file.

        Returns True if a new anti-pattern was learned and saved, False otherwise.
        """
        if not self._strategy_catalog_path:
            return False

        from .strategy_injection import load_strategy_catalog, learn_anti_pattern_from_failure

        catalog_path = Path(self._strategy_catalog_path).expanduser().resolve()
        if not catalog_path.exists():
            return False

        # Load current catalog
        try:
            catalog_list = load_strategy_catalog(catalog_path)
        except Exception:
            return False

        # Find which strategy the active leaf belongs to
        name = str(definition_name or "").strip()
        if not name:
            return False

        wm = self.get(name)
        if not wm:
            return False

        obj = load_world_model_obj(wm)
        if not isinstance(obj, dict):
            return False
        dt = obj.get("decision_tree")
        if not isinstance(dt, dict):
            return False
        active_id = str(dt.get("active_leaf_id", "") or "").strip()

        # Match active leaf to a strategy by strategy_combination or node_id pattern
        # Strategy nodes have node_id "s{i+1}" format
        strategy_idx = None
        if active_id.startswith("s"):
            try:
                strategy_idx = int(active_id[1:]) - 1
            except ValueError:
                pass

        if strategy_idx is None or strategy_idx < 0 or strategy_idx >= len(catalog_list):
            return False

        strategy = catalog_list[strategy_idx]
        strategy_id = str(strategy.get("id", "") or "").strip()
        api_refs = strategy.get("api_references", [])
        if not isinstance(api_refs, list) or not api_refs:
            return False

        # Check if failure_reason suggests API misuse
        failure_text = str(failure_reason or "").lower()
        api_keywords = ["api", "compile", "undeclared", "argument", "type", "precision", "accuracy", "mismatch"]
        has_api_failure_hint = any(kw in failure_text for kw in api_keywords)
        if not has_api_failure_hint and not changed_files:
            return False

        # Use LLM to analyze failure and extract anti-pattern
        prompt = (
            f"A kernel optimization round failed. Analyze the failure and extract an anti-pattern.\n\n"
            f"Strategy: {strategy_id} - {strategy.get('name', '')}\n"
            f"Strategy natural_language: {str(strategy.get('natural_language', '') or '')[:500]}\n\n"
            f"APIs involved in this strategy:\n"
        )
        for ref in api_refs:
            prompt += f"  - {ref.get('api_name', '')}: {str(ref.get('summary', '') or '')[:200]}\n"
        prompt += f"\nFailure reason: {str(failure_reason or 'unknown')}\n"
        if changed_files:
            prompt += f"Changed files: {', '.join(changed_files)}\n"
        prompt += (
            f"\nExisting anti-patterns for these APIs:\n"
        )
        for ref in api_refs:
            for ap in ref.get("anti_patterns", []):
                prompt += f"  - [{ap.get('id', '')}] {ap.get('pattern', '')}: {ap.get('reason', '')}\n"
        prompt += (
            "\nBased on this failure, identify which API was misused and how. "
            "Output a JSON object with exactly these fields:\n"
            "  api_name: which API was misused (must match one of the API names listed above)\n"
            "  pattern: short description of the misused pattern (what the code did wrong)\n"
            "  reason: why this pattern is incorrect on AscendC hardware\n\n"
            "Output ONLY the JSON object, no commentary."
        )

        try:
            raw = (self._llm_call(prompt) or "").strip()
        except Exception:
            return False

        # Parse the LLM response as JSON
        import json as json_mod
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)
        try:
            parsed = json_mod.loads(raw)
        except json_mod.JSONDecodeError:
            return False

        if not isinstance(parsed, dict):
            return False

        api_name = str(parsed.get("api_name", "") or "").strip()
        pattern = str(parsed.get("pattern", "") or "").strip()
        reason = str(parsed.get("reason", "") or "").strip()

        if not api_name or not pattern or not reason:
            return False

        # Verify api_name exists in the strategy's api_references
        matching_apis = [r for r in api_refs if str(r.get("api_name", "") or "").strip() == api_name]
        if not matching_apis:
            return False

        # Learn the anti-pattern
        new_ap = {"pattern": pattern, "reason": reason, "api_name_hint": api_name}
        updated_catalog = learn_anti_pattern_from_failure(
            strategy_catalog=catalog_list,
            strategy_id=strategy_id,
            api_name=api_name,
            new_anti_pattern=new_ap,
            round_index=round_index,
        )

        # Write back to catalog file
        try:
            catalog_data = {"strategy_catalog": updated_catalog}
            with open(catalog_path, "w", encoding="utf-8") as f:
                json_mod.dump(catalog_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
```

- [ ] **Step 3: 添加必要的 import**

确保 `world_model_manager.py` 顶部有 `from pathlib import Path`（已在第7行有）和 `from .world_model import ... load_world_model_obj`（已在import中）。

- [ ] **Step 4: 运行现有测试确保不破坏**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/ -v --timeout=60 2>&1 | tail -20
```

预期：所有现有测试仍然PASS。

- [ ] **Step 5: Commit**

```bash
cd /mnt/workspace/K-Search
git add k_search/kernel_generators/world_model_manager.py
git commit -m "feat: add anti-pattern learning trigger in WorldModelManager"
```

---

### Task 9: 集成验证 — 运行所有测试

**Files:** 无新增文件

- [ ] **Step 1: 运行全部相关测试**

```bash
cd /mnt/workspace/K-Search
python3 -m pytest tests/kernel_generators/test_strategy_injection.py tests/kernel_generators/test_strategy_api_injection.py -v
```

预期：全部PASS。

- [ ] **Step 2: 手动验证策略渲染输出**

```bash
cd /mnt/workspace/K-Search
python3 -c "
from k_search.kernel_generators.strategy_injection import load_strategy_catalog, render_strategy_as_action_text
catalog = load_strategy_catalog('strategies/mqa_strategies_catalog.json')
s4 = catalog[3]  # S4 is at index 3
result = render_strategy_as_action_text(s4, 'natural_language')
print(result[:1500])
print('---')
print('Total length:', len(result))
"
```

预期：输出包含 "=== AscendC API Reference ===" section 和 "=== Anti-Pattern Warnings ===" section，带有RowMuls签名、GetValue摘要、3个anti-pattern警告。

- [ ] **Step 3: 验证知识库可读性**

```bash
cd /mnt/workspace/K-Search
head -20 references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md
head -20 references/api_reference_docs/基础API/矢量计算/基础算术/RowDivs.md
head -20 references/api_reference_docs/基础API/矢量计算/基础算术/Muls.md
```

预期：三个文件都能正常读取，RowMuls和RowDivs是占位文件，Muls是完整的API文档。

- [ ] **Step 4: Commit (如有任何遗漏的修改)**

```bash
cd /mnt/workspace/K-Search
git add -A
git status
# Only commit if there are meaningful changes
git commit -m "chore: final integration verification and cleanup"
```

---

## Self-Review

### 1. Spec Coverage

| Spec Section | Plan Task | Status |
|-------------|-----------|--------|
| api_references字段格式 | Task 3 (JSON修改) + Task 4 (测试) + Task 5 (渲染) | ✅ |
| summary + anti_patterns分离 | Task 3 (JSON结构) + Task 4 (测试验证) | ✅ |
| strategy_injection.py渲染逻辑 | Task 5 (实现) | ✅ |
| API Reference section渲染 | Task 5 (`_render_api_references_section`) | ✅ |
| Anti-Pattern section渲染 | Task 5 (`_render_anti_patterns_section`) | ✅ |
| LLM查阅提示 | Task 5 (doc_path + hint) | ✅ |
| 知识库拷贝 | Task 1 | ✅ |
| 缺失API占位文件 | Task 2 (RowMuls + RowDivs) | ✅ |
| Anti-pattern运行时学习 | Task 7 (`learn_anti_pattern_from_failure`) + Task 8 (WM集成) | ✅ |
| 防重复机制 | Task 7 (`_text_similarity_ratio`) | ✅ |
| auto-id生成 | Task 7 (auto id logic) | ✅ |
| 持久化到catalog文件 | Task 8 (WM写入catalog文件) | ✅ |

### 2. Placeholder Scan

No TBD/TODO/placeholders found. All steps contain complete code.

### 3. Type Consistency

- `api_references` is `list[dict[str, Any]]` in JSON, Python, and tests — consistent.
- `anti_patterns` items have `id`, `pattern`, `reason`, `source`, `discovered_at` — consistent across all tasks.
- `doc_path` is relative to `K-Search/references/` — consistent in JSON, rendering, and docs.
- `strategy_id` used as string "S4" in anti-pattern tags `[AP-{strategy_id}-{id}]` — consistent.