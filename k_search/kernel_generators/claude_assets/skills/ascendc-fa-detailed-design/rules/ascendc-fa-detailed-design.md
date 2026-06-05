# AscendC Operator Document Generation

为 AscendC 算子生成**代码梳理文档**、**基线详设文档**和**增量详设文档**。支持两条路径：

- **路径 A（已有算子）**：代码梳理（理解现状）→ 基线详设（描述现有设计决策）→ 增量详设（在基线上标注变更）
- **路径 B（全新算子）**：golden + 需求 → 特性识别 → 设计模式匹配 → 完整详设文档

两条路径产出的 V1.0 格式完全一致（同一套 6 章模板），均可被后续 codegen skill 消费。

## Prerequisites Check（按路径区分）

| 路径 | 条件 | 必须/可选 | 缺失时处理 |
|------|------|----------|-----------|
| **路径 A** | 算子代码目录路径 | 必须 | AskUserQuestion 要求提供 |
| 路径 A Step 2/3 | 代码梳理文档 | 必须 | 若无，提示先完成 Step 1 |
| 路径 A Step 3 | V1.0 详设文档 | 必须 | 若无，提示先完成 Step 2 |
| 路径 A Step 3 | 增量需求描述（口述/文档均可） | 必须 | AskUserQuestion 引导补充 |
| **路径 B** | golden 参考实现路径 | 必须 | AskUserQuestion 要求提供 |
| 路径 B | 需求描述（特性列表、目标 layout/dtype/SoC） | 必须 | AskUserQuestion 引导补充 |
| 路径 B | 目标代码参考（类似算子） | 可选 | 有则参考架构，无则纯从 design-patterns 推导 |

## Platform Detection

**检查代码或设计中是否含 `arch35/351x/Regbase` → A5，否则 → A3。**

Design-patterns 引用路径：
- **A3 平台**：`references/a3-design-patterns/`
- **A5 平台**：`references/a5-design-patterns/`

## Task Selection

使用 AskUserQuestion 确认任务范围：

```
AskUserQuestion:
  question: "请选择需要执行的任务："
  header: "任务范围"
  options:
    - label: "仅代码梳理"
      description: "分析现有算子代码架构，生成代码梳理文档（Step 1）"
    - label: "代码梳理 + 基线详设"
      description: "梳理代码后，生成描述现有实现的基线详设文档（Step 1→2）"
    - label: "仅增量详设"
      description: "基于已有基线详设 + 增量需求，生成增量详设文档（Step 3，需已有基线详设）"
    - label: "完整流程（已有算子）"
      description: "代码梳理 → 基线详设 → 增量详设（Step 1→2→3）"
    - label: "新建算子详设"
      description: "从 golden + 需求描述生成新算子的完整详设文档（无现有代码基线，路径 B）"
    - label: "设计检查清单"
      description: "对已有详设运行 FA 专项检查（F1-F9/R1-R3/E1-E3）"
```

---

## 路径 A: Step 1 — Code Analysis Document Generation

### 1.1 Read Operator Code

**系统性读取算子全部代码，不可遗漏任何文件：**

| 目录 | 读取内容 | 关注点 |
|------|---------|--------|
| `docs/` | 已有文档 | 已有的架构说明、API 文档 |
| `op_host/*_def.cpp` | OpDef | 输入/输出/属性定义、类型约束、SoC 配置 |
| `op_host/*_tiling.h` | Tiling 头文件 | TilingData 结构体、枚举、类声明 |
| `op_host/*_tiling.cpp` | Tiling 实现 | 切分策略主流程、参数校验、FillTiling |
| `op_host/arch*/` | 架构特化 | 不同 SoC 的 Tiling 差异 |
| `op_kernel/*.cpp` | Kernel 入口 | `__global__` 函数、模板参数、数据类型组合 |
| `op_kernel/*_kernel*.h` | 主 Kernel 类 | Init、主循环编排、核间同步 |
| `op_kernel/*_service_cube*.h` | Cube 服务 | AIC 侧矩阵乘法（BMM1/BMM2） |
| `op_kernel/*_service_vector*.h` | Vector 服务 | AIV 侧向量运算（加载/Softmax/输出） |
| `op_kernel/*_common.h` | 公共定义 | CVSharedParams、枚举、对齐工具 |
| `op_kernel/util_regbase.h` | 运行时结构体 | RunInfo、ConstInfo 定义 |
| `op_kernel/vf/` | 微核函数 | Softmax 路径选择、Flash Update |
| `tests/` | 测试框架 | 测试脚本结构、golden 计算方式 |
| `CMakeLists.txt` | 构建配置 | 编译依赖、SoC 配置 |

### 1.2 Generate Code Analysis Document

按照 references/code-analysis-template.md 模板生成文档，核心章节：

1. **算子概述**：功能定位、计算公式、支持平台
2. **目录结构**：带注释的完整文件树
3. **输入/输出/属性规格**：完整的张量和属性参数表
4. **Host 侧逻辑**：OpDef、InferShape、Tiling 策略
5. **Kernel 侧逻辑**：AIC/AIV 架构、主循环流水、Cube/Vector 服务、微核函数
6. **运行时参数结构体**：RunInfo、ConstInfo、CVSharedParams
7. **端到端数据流**：完整的数据通路图
8. **关键设计总结**：设计要点与实现方式对照表

**输出路径**：`docs/{op_name}_代码梳理.md`

---

## 路径 A: Step 2 — Baseline Design Document Generation

基于代码梳理文档，按详设模板格式描述**现有实现的设计决策**。

### 2.2 开发场景识别

根据概要设计文档或用户描述，判断开发场景：

| 场景 | 代号 | 识别特征 |
|------|------|---------|
| 特性注入 | S1 | 在现有 FA 算子上新增特性/mask/dtype/模式 |
| 变体 Fork | S2 | 新算子但核心流程与某基座高度相似 |
| 架构迁移 | S3 | arch32 迁移到 arch35 |
| 全新 FA | S4 | 全新算法，不基于任何现有基座 |

### 2.3 基座匹配（S1/S2/S3）

**自动匹配表**：

| 新算子特征 | 匹配基座 |
|-----------|---------|
| 前向训练 | FAS |
| 反向训练 | FASG |
| 推理（含KV Cache） | FIAS |
| Prefill | PFA |
| Decode/Incremental | IFA |
| Lightning Indexer（非量化） | LI |
| 量化 Lightning Indexer | QuantLI |
| 稀疏 Lightning Indexer 反向 | SparseLI_GradKLLoss |

### 2.4 Read Design Pattern References

根据平台检测结果选择正确的 design-patterns 目录：

| 平台 | Design-patterns 路径 |
|------|---------------------|
| A3 | `references/a3-design-patterns/` |
| A5 | `references/a5-design-patterns/` |

| 详设章节 | 参考文档 | 提取内容 |
|---------|---------|---------|
| 4.1.1 计算流程图（CV流水） | fa_01_xxx | 流水模型、RunInfo环形缓冲、同步拓扑 |
| 4.1.1 计算流程图（反向） | fa_07_xxx | 反向Cube三级流水、梯度计算路径 |
| 4.1.2.2 核间tiling（分核） | fa_03_xxx | Cost函数、三级分配 vs 均分 |
| 4.1.2.2 核间tiling（稀疏） | fa_08_xxx | 两阶段负载均衡 |
| 4.1.3.1 UB空间 | fa_05_xxx | UB Queue/TBuf分配方案 |
| 4.1.3.2 workspace | fa_04_xxx | Workspace布局、双缓冲 |
| 4.1.3.3 L1/L0 | fa_02_xxx | L1缓冲方案选型、Sink常驻 |
| 4.1.4 Kernel（同步） | fa_01 + fa_04 | CrossCore flag、同步协议 |
| 4.1.4 Kernel（稀疏KV） | fa_06_xxx | 稀疏KV处理方案 |
| 4.1.4 Kernel（Gather/Scatter） | fa_08 | Ping-Pong双缓冲、ScatterAdd |

### 2.5 Generate Baseline Design Document

**关键原则**：
- 第 4 章是核心，必须包含伪代码级别的现有实现方案
- 标注当前算子采用的 Variant 及选型理由
- 每个设计决策标注约束条件和 Trade-off
- 不涉及任何变更描述，纯粹描述现状
- **增量场景（S1/S2/S3）：变更处标注需求编号，无变更处明确标注"继承自{基座名}{具体设计点}"**
- **禁止使用模糊表述如"继承V1.0"、"与基线一致"——必须明确写出继承自哪个基座算子的哪个具体文件/模块/设计**

**输出路径**：`docs/{op_name}_详设_V1.0.md`

---

## 路径 A: Step 3 — Incremental Design Document Generation

**核心原则：完整继承 + 增量标注**

V1.x 文档必须是**自包含的完整文档**：
1. 以 V1.0 全文为起点，完整复制所有章节
2. 无变更章节：保留 V1.0 原文不动
3. 有变更章节：用 **【变更】** 标签标注修改/新增部分
4. 新增章节：用 **【新增】** 标签标注

**禁止做法**：
- ❌ 无变更章节只写"无变更"——丢失完整信息
- ❌ 写成只包含变更点的"差异文档"
- ❌ 使用"继承V1.0"等模糊表述——必须明确写出继承自哪个基座的具体设计
- ✅ 无变更章节完整保留原文
- ✅ 有变更章节用【变更】标签标注差异点

**版本号规则**：扫描已有 V1.*.md，取最大版本号 +1。

**输出路径**：`docs/{op_name}_详设_V1.{x}.md`

---

## 路径 B: New Operator Design Document Generation

从 golden + 需求描述生成全新算子的完整详设文档，无需现有代码基线。

### B.1 Collect Inputs

| 输入 | 必须/可选 | 说明 |
|------|----------|------|
| golden 参考实现 | 必须 | PyTorch model.py，定义算子语义和接口 |
| 需求描述 | 必须 | 特性列表、目标 layout、dtype、SoC 约束、设计原则 |
| 目标代码参考 | 可选 | 已有的类似生产算子代码，用于参考架构 |

### B.2 Step N1: 需求分析与特性识别

1. 读取 golden，提取输入/输出签名、计算语义
2. 从需求描述提取特性列表（PA/Mask/Sparse/GQA/Sink/FD）
3. 确定目标约束：SoC、dtype、layout
4. **平台检测**：检查是否含 `arch35/351x/Regbase` → A5，否则 → A3
5. 输出：特性组合清单 + 接口规格 + 平台类型

### B.3 Step N2: 设计模式匹配

根据特性组合和平台类型，从 design-patterns 中选择相关模式：

| 特性组合 | 主骨架模式 | 辅助模式 |
|----------|-----------|----------|
| 基础 FA | fa_01(四级流水) | — |
| FA + GQA | fa_01 + fa_03(分核) | — |
| FA + Sparse | fa_01 + fa_06(稀疏KV合并) | fa_08(负载均衡) |
| FA + PA | fa_01 + fa_04(workspace双缓冲) | — |
| FA + Sparse + PA | fa_01 + fa_06 + fa_04 | fa_08 |
| FA + Sink | fa_01 + fa_02(L1多缓冲sink常驻) | — |
| FA 反向 | fa_07(反向多Cube流水) | fa_01 |

执行步骤：
1. Read 匹配到的所有 design-pattern 参考文档（根据平台选择 a3 或 a5 目录）
2. 如有目标代码参考，Read 其架构作为主骨架
3. 确定主流水编排方案（三级/四级/五级）
4. 确定 AIC/AIV 分工和同步拓扑

### B.4 Step N3: 详设文档生成

按 references/design-doc-template.md 模板生成完整 6 章详设。

**第 4 章生成的核心约束**：
- 每个设计决策标注选型理由和 Trade-off
- 引用 design-pattern 中的 Variant 对比表（根据平台选择正确目录）
- 必须包含伪代码级别的实现方案
- 必须包含完整 Buffer 分配表、CrossCore flag 协议表、边界条件处理策略表

**边界条件处理策略**（必须在 4.1.5 中明确）：

| 边界场景 | 处理策略 |
|----------|----------|
| seq_len 不对齐 block_M | 尾块 mask + 有效长度判定 |
| topk 不对齐 block_I | 无效 index 填充 -1 + gather 后 zero-fill |
| actual_kv_len < 预分配 | causal_limit 动态计算 + block_table 越界保护 |
| head_kv 不对齐 | padding 到 next_power_of_2 + 输出时 mask |
| 多 batch 不等长 | per-batch actual_len + 跳过无效 query |

**输出路径**：`docs/{op_name}_详设_V1.0.md`

### B.5 User Confirmation

确认后建议下一步调用 `ascendc-fa-new-operator-codegen` skill 进行代码生成。

**路径 B 产出的 V1.0 后续可走路径 A 的 Step 3 做增量迭代**。

---

## Step 2.5: 设计检查清单

当用户选择"设计检查清单"任务时执行。

### 检查模块

| 模块 | 检查项 | 适用平台 |
|------|--------|---------|
| FA 通用 | F1-F9（流水/同步/分核/Buffer/精度/Mask/确定性/异常/同步正确性） | 所有 |
| Regbase 专项 | R1-R3（VF函数/Scalar前移/寄存器分配） | A5 |
| 流水效率 | E1-E3（Bubble分析/Double Buffer/带宽瓶颈） | 推荐执行 |

### 检查执行

Read references/checklists/ 目录下的对应检查清单文档。

---

## Output Summary

| 步骤 | 输出文件 | 描述 |
|------|---------|------|
| Step 1 | `docs/{op_name}_代码梳理.md` | 算子代码架构全面梳理 |
| Step 2 / Step N3 | `docs/{op_name}_详设_V1.0.md` | 设计决策文档（路径 A 从代码提取 / 路径 B 从零设计） |
| Step 3 | `docs/{op_name}_详设_V1.{x}.md` | 增量需求的变更设计文档 |

## References

- Code Analysis Template: references/code-analysis-template.md
- Design Document Template: references/design-doc-template.md
- A3 Design Patterns: references/a3-design-patterns/
- A5 Design Patterns: references/a5-design-patterns/
- Baseline Feature Cards: references/baselines/
- Design Checklists: references/checklists/