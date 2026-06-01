# AscendC API 信息注入策略设计

> 日期：2026-06-01
> 问题来源：[nl_strategy_problem_analysis.md](../../nl_strategy_problem_analysis.md)

## 背景

MQA AscendC 策略优化实验中，自然语言策略的通过率仅 33%（4/12轮），其中 2轮因API签名缺失导致编译失败（R1/R2），5轮因缺少不等价陷阱警告导致精度失败（R6/R7/R9-R11）。核心洞察：**自然语言策略缺少硬件层关键约束信息**——LLM必须猜测AscendC API用法，猜测基于通用编程经验而非AscendC专用知识。

本设计解决"如何为策略提供AscendC API信息"的问题，采用**混合方案**：关键摘要静态注入策略文本 + 完整文档路径供LLM按需查阅。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 注入方式 | 摘要静态注入 + 文档路径补充 | Summary覆盖签名/约束解决编译失败，doc_path提供完整查阅解决精度问题 |
| 索引格式 | 策略JSON新增 `api_references` 字段 | 与现有策略catalog结构一致，每个策略明确声明所需API |
| 知识库位置 | 拷贝到 `K-Search/references/` | 路径稳定，LLM可直接read |
| 文档组织 | 保持原样（一个API一个MD文件） | 864个文件/8MB，LLM可按需read单个API文档 |
| 缺失API处理 | 创建占位文件，后续手动补充 | RowMuls/RowDivs不在知识库中，需占位+手写初始内容 |

## 一、策略JSON `api_references` 字段

### 字段结构

在策略catalog JSON中新增 `api_references` 数组，每项包含：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `api_name` | string | 是 | API名称，如 `"RowMuls"`, `"Brcb"` |
| `doc_path` | string | 是 | 相对于 `K-Search/references/` 的文档路径 |
| `summary` | string | 是 | 关键摘要：函数签名 + 核心参数约束 |
| `anti_patterns` | array | 否 | 已知误用模式列表，动态增长 |

`summary` 与 `anti_patterns` 分开设计：
- **summary**：API签名和核心参数约束，人工编写，相对稳定
- **anti_patterns**：每个误用模式+原因+发现来源，初始人工填充已知陷阱 + 运行时从失败轮次自动学习追加

### anti_patterns 子项结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | anti-pattern编号，如 `"ap1"` |
| `pattern` | string | 误用模式描述 |
| `reason` | string | 为什么不等价/不正确 |
| `source` | string | 发现来源，如 `"round_9_failure"` |
| `discovered_at` | int | 发现的轮次编号 |

### S4策略示例

```json
"api_references": [
  {
    "api_name": "RowMuls",
    "doc_path": "api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md",
    "summary": "RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32). scale参数是LocalTensor而非float标量。dealRows是行数，actualCols是实际列数（含padding处理）。",
    "anti_patterns": [
      {
        "id": "ap1",
        "pattern": "Duplicate(scalar,dimAlign)+Mul(dst,src,buf,dimAlign) 替代 RowMuls",
        "reason": "PIPE_V RAW冒险 + padding区域垃圾值不清零 + 累积精度误差",
        "source": "round_9_failure",
        "discovered_at": 9
      },
      {
        "id": "ap2",
        "pattern": "LocalTensor[] 下标运算符作为 float 标量参数传入 Muls",
        "reason": "LocalTensor[] 返回子张量视图 LocalTensor<T>，不是标量 T。需用 GetValue(tensor, index)",
        "source": "round_2_failure",
        "discovered_at": 2
      }
    ]
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
    "summary": "Brcb(dst, src, repeatTimes, dstStride). 需要4参数，不可只传3个。用于行级广播，需配合RowMuls而非Mulcs(Mulcs在CANN 9.0中不存在)。",
    "anti_patterns": []
  }
]
```

## 二、`strategy_injection.py` 渲染逻辑

### 修改点

在 `render_strategy_as_action_text()` 中，当策略有 `api_references` 且非空时，在策略文本末尾追加两个section：

### 渲染格式

```
Strategy S4: Vectorized RowOps (category=compute, impact=high, difficulty=2)

[原有 natural_language 文本...]

=== AscendC API Reference ===

RowMuls:
  Signature: RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32)
  Key constraints: scale参数是LocalTensor而非float标量
  Full doc: references/api_reference_docs/基础API/矢量计算/基础算术/RowMuls.md

GetValue:
  Signature: GetValue(index: uint32) → PrimType
  Key constraints: 仅在VECIN/VECCALC/VECOUT位置支持
  Full doc: references/api_reference_docs/基础数据结构/LocalTensor/GetValue.md

If you need more details (supported types, examples, alignment constraints), read the doc files at the paths above.

=== Anti-Pattern Warnings (DO NOT do the following) ===

[AP-S4-ap1] Do NOT use Duplicate(scalar,dimAlign)+Mul(dst,src,buf,dimAlign) as a substitute for RowMuls.
  Reason: PIPE_V RAW冒险 + padding区域垃圾值不清零 + 累积精度误差
  (Discovered from round 9 failure)

[AP-S4-ap2] Do NOT use LocalTensor[] subscript as a float scalar argument to Muls.
  Reason: LocalTensor[] returns sub-tensor view LocalTensor<T>, not scalar T. Use GetValue(tensor, index) instead.
  (Discovered from round 2 failure)
```

### 渲染规则

1. **API Reference section**：每个API显示summary中的签名和核心约束 + doc_path查阅路径
2. **Anti-Pattern section**：所有anti_patterns跨API汇总到一个统一section，用 `[AP-{策略ID}-{anti_pattern.id}]` 编号方便引用
3. **查阅提示**：API Reference section末尾添加 "If you need more details..." 提示
4. **渲染条件**：仅当策略有 `api_references` 且非空时追加；无API引用的策略保持原样
5. **Anti-Pattern可选**：如果某API的anti_patterns为空数组，不在Anti-Pattern section中显示该API

## 三、Anti-Pattern 运行时学习机制

### 学习流程

```
优化轮次失败 → WM记录 failure_type + failure_reason
  → 检测失败涉及API误用
  → 调用LLM分析 failure_reason + changed_files，生成结构化anti_pattern
  → 匹配到策略catalog中对应 api_references[].api_name
  → 追加新anti_pattern到 anti_patterns 数组
  → 写入策略catalog JSON文件
  → 下一轮渲染时LLM看到更新后的anti-pattern警告
```

### 实现位置

在 `world_model_manager.py` 的轮次更新逻辑中新增anti-pattern学习函数。

### 提取方式

调用LLM分析失败信息，生成结构化输出：

```json
{
  "pattern": "<误用模式描述>",
  "reason": "<为什么不等价/不正确>",
  "source": "round_{N}_failure",
  "discovered_at": N
}
```

### 匹配逻辑

根据 `failure_reason` 中提及的API名称，匹配到策略catalog中对应 `api_references[].api_name`，追加到其 `anti_patterns` 数组。

### 防重复机制

追加前检查 `anti_patterns` 中是否已有 `pattern` 文本相似度 > 80% 的条目，避免重复学习同一陷阱。文本相似度可通过简单的字符串重叠比例计算。

### 持久化

追加后写入 `strategies/mqa_strategies_catalog.json`，确保跨轮次持久化。

## 四、知识库拷贝与缺失API占位

### 知识库拷贝

将 `/home/developer/.claude/skills/ascendc-dev-knowledge/references/` 拷贝到 `K-Search/references/`，保持原样目录结构（864个MD文件，约8MB）。

策略JSON中的 `doc_path` 使用相对于 `K-Search/references/` 的路径。LLM查阅时使用项目相对路径 `references/{doc_path}`（即 `{K-Search项目根}/references/{doc_path}`），需确保codegen agent的工作目录为K-Search项目根或能访问该路径。

### 缺失API占位文件

为不在现有知识库中的API创建占位MD文件，放在对应分类目录下。

当前已知缺失API：

| API名 | 目录位置 | 优先级 |
|--------|----------|--------|
| RowMuls | 基础API/矢量计算/基础算术/ | 高（S4核心） |
| RowDivs | 基础API/矢量计算/基础算术/ | 高（S4核心） |

占位文件模板：

```markdown
# RowMuls

> ⚠️ 此文档为占位文件，内容待补充完整。

#### 功能说明
逐行向量乘法：将每行的元素与scale张量中对应行的标量因子相乘。

#### 函数原型
```
template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dst, LocalTensor<T> src,
                                LocalTensor<T> scale, int32_t dealRows,
                                int32_t cols, int32_t actualCols)
```

#### 参数说明
| 参数名 | 类型 | 说明 |
|--------|------|------|
| dst | LocalTensor<T> | 目标张量 |
| src | LocalTensor<T> | 源张量 |
| scale | LocalTensor<T> | 缩放因子张量（每行一个因子，位于 offset row*BRCB_NUM） |
| dealRows | int32_t | 处理行数 |
| cols | int32_t | 对齐列数 |
| actualCols | int32_t | 实际有效列数 |

#### 关键约束
- scale参数是LocalTensor<T>，不是float标量
- 每行的缩放因子存储在 offset row*BRCB_NUM 处

#### Anti-Pattern（已知误用模式）
<!-- 待补充 -->
```

## 五、改动范围

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `K-Search/references/` | 新增目录 | 拷贝ascendc-dev-knowledge的references |
| `K-Search/references/api_reference_docs/.../RowMuls.md` | 新增 | RowMuls API文档占位 |
| `K-Search/references/api_reference_docs/.../RowDivs.md` | 新增 | RowDivs API文档占位 |
| `K-Search/strategies/mqa_strategies_catalog.json` | 修改 | 为S4等策略新增 `api_references` 字段 |
| `K-Search/k_search/kernel_generators/strategy_injection.py` | 修改 | 渲染逻辑追加API Reference + Anti-Pattern section |
| `K-Search/k_search/kernel_generators/world_model_manager.py` | 修改 | 新增anti-pattern学习逻辑 |

**不改动的文件**：
- `kernel_generator_prompts.py` — API信息通过action_text注入extra_context，与现有prompt流程兼容
- `world_model.py` / `kernel_generator.py` — WM和codegen流程不变

**设计边界**：
- 本设计只解决"如何为策略提供API信息"的问题
- 策略内容改进（最小变更优先、预期加速置信区间）不在本范围
- WM框架改进（连续失败自动回退、变更文件数限制）不在本范围

## 六、预期效果

根据问题分析文档的估计：

| 问题 | 影响轮次 | 建议改进 | 预期改进 |
|------|----------|----------|----------|
| API签名缺失 | R1, R2 (2轮) | summary注入精确签名 | 编译失败减少 |
| 不等价陷阱缺失 | R6,R7,R9-R11 (5轮) | anti-pattern警告 | 精度失败减少 |
| 综合 | 7/12轮失败 | 本设计全部改动 | 通过率从33%提升到约75% |