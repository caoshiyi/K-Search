---
name: ascendc-fa-detailed-design
description: |
  Generate code analysis documents and detailed design documents for AscendC operators.
  Use when:
  (1) User asks "代码梳理", "分析一下这个算子代码", "帮我理解这个算子", "梳理代码结构"
  (2) User asks "生成详设文档", "写详细设计", "生成设计文档", "doc generation"
  (3) User provides operator code path and wants to understand architecture before development
  (4) User has a new feature requirement and needs a formal design document before coding
  (5) User says "我要开始增量开发，先帮我梳理代码" or similar pre-development preparation
  (6) Before invoking ascendc-feature-dev, when code analysis or design docs are missing
  (7) User asks "帮我出一份详设", "写个设计文档", "分析下代码架构"
  (8) User says "新建算子详设", "从零设计", "根据golden生成详设", "新算子设计"
  (9) User has golden + requirements and wants to generate design doc for a new operator (no existing code)
---

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

- 选择前 5 项 → 进入路径 A，检查路径 A 前置条件
- 选择"新建算子详设" → 进入路径 B，检查路径 B 前置条件
- 选择"设计检查清单" → 进入 Step 2.5

---

## Step 1: Code Analysis Document Generation

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

按照 [references/code-analysis-template.md](references/code-analysis-template.md) 模板生成文档，核心章节：

1. **算子概述**：功能定位、计算公式、支持平台
2. **目录结构**：带注释的完整文件树
3. **输入/输出/属性规格**：完整的张量和属性参数表
4. **Host 侧逻辑**：OpDef、InferShape、Tiling 策略（主流程、切分参数、TilingData 结构、Workspace、参数校验）
5. **Kernel 侧逻辑**：
   - AIC/AIV 异构架构图
   - Kernel 入口（模板参数、数据类型组合）
   - 主 Kernel 类（内存层级与 Buffer 布局、初始化流程、主循环三级流水编排）
   - Cube 服务（BMM1/BMM2 核心方法）
   - Vector 服务（三阶段处理：Vec0/Vec1/Vec2）
   - 微核函数（路径选择、Flash Update）
6. **运行时参数结构体**：RunInfo、ConstInfo、CVSharedParams 字段说明
7. **端到端数据流**：完整的数据通路图
8. **关键设计总结**：设计要点与实现方式对照表

**输出路径**：`docs/{op_name}_代码梳理.md`

### 1.3 User Confirmation

将生成的文档提交用户确认：

```
请确认代码梳理文档：
□ 算子概述：功能描述是否准确
□ 目录结构：是否有遗漏文件
□ Host 侧：Tiling 策略描述是否完整
□ Kernel 侧：主循环流水、核间同步描述是否正确
□ 数据流：端到端路径是否清晰
□ 文档路径：docs/{op_name}_代码梳理.md
```

用户确认 → 如需继续则进入 Step 2（基线详设）。用户修改意见 → 更新文档后重新确认。

---

## Step 2: Baseline Design Document Generation

基于代码梳理文档，按详设模板格式描述**现有实现的设计决策**。这是增量详设的基础——reviewer 需要先看到"现状是什么"，才能评估"要改成什么"。

**代码梳理 vs 基线详设的区别**：
- 代码梳理：偏"是什么"——代码结构、函数调用链、数据流
- 基线详设：偏"为什么这样设计"——设计决策、选型理由、Trade-off、约束条件

### 2.1 Read Inputs

| 输入 | 来源 |
|------|------|
| 代码梳理文档 | Step 1 输出 或 用户提供路径 |
| 概要设计文档 | 用户提供（概设模式时） |

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

**匹配后，必须要求用户提供基座算子代码路径**，用于提取具体设计继承。

### 2.4 平台检测

检查代码或设计中是否含 `arch35/351x/Regbase` → **A5**，否则 → **A3**。

### 2.5 Read Design Pattern References

**在生成第 4 章（模板设计）各小节之前，必须先读取对应的设计模式参考文档**，识别当前算子采用了哪个 Variant，标注选型理由。

根据平台检测结果选择正确的 design-patterns 目录：

| 平台 | Design-patterns 路径 |
|------|---------------------|
| A3 | `references/a3-design-patterns/` |
| A5 | `references/a5-design-patterns/` |

| 详设章节 | 参考文档（A3/A5 前缀自适应） | 提取内容 |
|---------|---------|---------|
| 4.1.1 计算流程图（CV流水并行设计） | fa_01_xxx | 流水模型（四级/五级）、RunInfo环形缓冲、同步拓扑、时序图、流水启排空气泡 |
| 4.1.1 计算流程图（反向多Cube流水） | fa_07_xxx | 反向Cube1/2/3三级流水、Cube12/345两阶段流水、dQ/dK/dV梯度计算路径、Softmax梯度中间结果 |
| 4.1.2.2 核间tiling策略（分核设计） | fa_03_xxx | 分核位置（Host/Kernel侧）、Cost函数设计、三级分配 vs 加权均分 vs 简单均分、核数决策、FD归约触发与Workspace |
| 4.1.2.2 核间tiling策略（稀疏负载均衡） | fa_08_xxx | 两阶段负载均衡（粗调+精调）、Kernel侧块分配、确定性模式支持 |
| 4.1.2.3 核内tiling策略 | fa_03 + fa_01 | M/S2轴切分参数、分档逻辑、循环结构、尾块处理、NBuffer二次切分 |
| 4.1.3.1 UB空间 | fa_05_xxx | UB Queue/TBuf分配方案（GQA标准/Sink扩展/Pioneer Vec0）、Softmax状态preLoadNum缩放、UB容量规划、时分复用策略 |
| 4.1.3.2 workspace | fa_04_xxx | Workspace四/六区域布局、双缓冲偏移计算、生产消费配对、FD Workspace条件分配 |
| 4.1.3.3 L1/L0分配 | fa_02_xxx | L1缓冲方案选型（Q2V2KP3 / QP2+KV2 / QP2+Sink2+KV3）、Event同步、Sink常驻策略、缓冲块数与流水深度权衡 |
| 4.1.4 Kernel设计（核间同步） | fa_01 + fa_04 | CrossCore flag定义、syncC1V1/syncV1C2/syncC2V2 同步协议、预加载策略、双向同步（Pioneer Vec0场景） |
| 4.1.4 Kernel设计（稀疏KV数据通道） | fa_06_xxx | 稀疏KV处理方案（直读/Vec0合并/训练版本）、GM→UB→Workspace→L1数据流、Vec0-Cube跨核同步 |
| 4.1.4 Kernel设计（Gather/Scatter） | fa_08_xxx | Gather/Scatter Ping-Pong双缓冲、三阶段Vector流水（V0/V1/V2）、ScatterAdd原子操作、12路CrossCore同步 |

**使用方式**：
1. 写每个章节前，Read 对应参考文档（根据平台选择 a3 或 a5 目录）
2. 识别当前算子最接近哪个 Variant（A/B/C），说明选型理由
3. 如果当前设计与已有 Variant 不同，明确标注差异点和 Trade-off
4. 在设计决策处引用参考文档中的对比表，增强 review 可信度

### 2.6 Generate Baseline Design Document

按照 [references/design-doc-template.md](references/design-doc-template.md) 模板生成，内容描述**现有实现**而非变更。

**必须严格按以下子章节清单逐节生成，不可遗漏任何章节**（即使内容为"不涉及"也要保留章节标题）：

```
# 1 算子概述
  ## 1.1 功能定义          — 功能定位、计算公式、设计原则
  ## 1.2 支持场景          — Layout × 数据类型 × 场景 表格

# 2 接口实现设计
  ## 2.1 PTA接口实现       — PTA函数原型
  ## 2.2 aclnn接口实现     — aclnn函数原型 + 完整参数表（参数名/输入输出/描述/数据类型/维度）
  ## 2.3 算子信息库（OpDef）— 输入/输出/属性数量、关键约束
  ## 2.4 图模式设计        — GE/AclGraph适配说明，或"不涉及"

# 3 总体设计
  ## 3.1 交付方式          — 交付内容 + 代码承载路径
  ## 3.2 交付件汇总        — 13项交付件表格（pta/aclnn/OpDef/tiling/kernel/...）
  ## 3.3 TilingKey设计     — 模板参数组合表
  ## 3.4 模板列表          — 规格 × 场景 × 模板 × 核间/核内切分
  ## 3.5 代码结构设计      — 目录/内容/变化点/需求编号/需求标题 表格

# 4 模板设计
  ## 4.1 {OpName} 计算模板
    ### 4.1.1 计算流程图   — CV流水时序图、同步拓扑、任务缓存、NBuffer二次切分
    ### 4.1.2 Tiling实现设计
      #### 4.1.2.1 TilingData设计  — 完整字段表
      #### 4.1.2.2 核间tiling策略  — Variant选型 + 伪代码
      #### 4.1.2.3 核内tiling策略  — 切分参数 + 主循环结构
    ### 4.1.3 Buffer设计
      #### 4.1.3.1 UB空间的分配   — UB Buffer表（块大小/BUF_NUM/数据类型/总大小/用途）
      #### 4.1.3.2 workspace的分配 — Workspace布局图 + Variant选型
      #### 4.1.3.3 L1/L0的分配    — L1 Buffer表
    ### 4.1.4 Kernel设计   — 核间同步协议表 + 关键设计说明
    ### 4.1.5 异常场景设计  — 异常场景/处理方式/处理层级 表格
    ### 4.1.6 支持确定性计算设计 — 说明，或"不涉及"
    ### 4.1.7 精度分析及设计     — 精度标准 + 关键精度路径分析
    ### 4.1.8 性能分析及设计     — 性能特征 + 性能瓶颈
    ### 4.1.9 训推一致性设计     — 说明，或"不涉及"

# 5 维测设计
  ## 5.1 可测试性
    ### 5.1.1 功能测试     — 测试场景/验证点/优先级 表格
    ### 5.1.2 精度测试     — CPU参考实现(Golden)说明 + 精度标准
    ### 5.1.3 边界条件测试  — 边界场景/预期行为 表格
  ## 5.2 可观察性          — ST/UT/cannsim/DumpTensor 说明
  ## 5.3 可维护性
    ### 5.3.1 关键风险与应对 — 风险/描述/应对措施 表格
    ### 5.3.2 后续演进      — 后续扩展方向

# 6 资料设计              — 资料类型/是否涉及变更/变更内容 表格
```

**生成时逐节对照上述清单，确保每个章节标题和层级与模板完全一致。**

**关键原则**：
- 第 4 章是核心，必须包含伪代码级别的现有实现方案
- 第 4 章各小节须参照 Step 2.5 的设计模式参考，标注当前算子采用的 Variant 及选型理由
- 每个设计决策标注**约束条件**和**Trade-off**，为后续增量设计提供决策基础
- 不涉及任何变更描述，纯粹描述现状
- 即使某章节内容为"不涉及"，也必须保留章节标题（如 2.4 图模式设计、4.1.6 确定性计算、4.1.9 训推一致性）
- **增量场景（S1/S2/S3）：变更处标注需求编号，无变更处明确标注"继承自{基座名}{具体设计点}"**
- **禁止使用模糊表述如"继承V1.0"、"与基线一致"——必须明确写出继承自哪个基座算子的哪个具体文件/模块/设计**

**输出路径**：`docs/{op_name}_详设_V1.0.md`

### 2.7 User Confirmation

将生成的基线详设提交用户确认：

```
请确认基线详设文档：
□ 算子概述：功能定义和设计原则是否准确
□ 接口设计：当前 OpDef/aclnn 描述是否完整
□ 模板设计（核心）：
  - 分核策略：Variant 选型和理由是否正确
  - Tiling 策略：切分参数和分档逻辑是否完整
  - CV 流水：流水模型和同步拓扑是否准确
  - Buffer 布局：L1/L0/UB/Workspace 描述是否完整
□ 核间同步：CrossCore flag 协议是否正确
□ 文档路径：docs/{op_name}_详设_V1.0.md
```

用户确认 → 如需继续则进入 Step 3（增量详设）。用户修改意见 → 更新文档后重新确认。

---

## Step 3: Incremental Design Document Generation

基于基线详设 + 增量需求，生成增量详设文档。

**核心原则：完整继承 + 增量标注**

V1.x 文档必须是**自包含的完整文档**，读者无需翻阅 V1.0 即可理解全部设计。具体做法：
1. **以 V1.0 全文为起点**，完整复制所有章节内容
2. **无变更章节**：保留 V1.0 原文不动（不要写"无变更"替代原文）
3. **有变更章节**：在 V1.0 原文基础上，用 **【变更】** 标签标注修改/新增的部分
4. **新增章节**：用 **【新增】** 标签标注整个章节

**禁止做法**：
- ❌ 无变更章节只写一句"无变更"或"【不变】"——这丢失了 V1.0 的完整信息
- ❌ 把 V1.1 写成只包含变更点的"差异文档"——reviewer 需要完整上下文
- ❌ 使用"继承V1.0"等模糊表述——必须明确写出继承自哪个基座的具体设计
- ✅ 无变更章节完整保留 V1.0 原文，让文档自包含
- ✅ 有变更章节在原文中用【变更】标签标注差异点，reviewer 可快速定位

**标注格式示例**：

```markdown
## 3.2 TilingKey设计

当前TilingKey由4个模板参数组合生成：
... (V1.0 原文保留) ...

**【变更】** 激活FLASH_DECODE=1的模板参数组合：
... (新增内容) ...
```

### 3.1 Read Inputs

| 输入 | 来源 |
|------|------|
| V1.0 详设文档 | Step 2 输出 或 用户提供路径 |
| 增量需求描述 | 用户口述 / 需求文档（.md/.docx/PDF） |

如增量需求描述不充分，使用 AskUserQuestion 引导补充：

```
AskUserQuestion:
  question: "请补充以下需求信息："
  header: "需求细节"
  options:
    - label: "我来补充"
      description: |
        需要明确：
        1. 功能定义（新特性做什么）
        2. 新增输入/输出/属性（如有）
        3. 约束条件（支持哪些场景/布局/数据类型）
        4. 设计原则（是否需要零框架侵入、向后兼容等）
    - label: "按我已提供的信息先写"
      description: "基于当前信息生成初版，后续迭代补充"
```

### 3.2 Generate Incremental Design Document

以基线详设（V1.0）全文为起点，完整继承后在变更处用标签标注增量修改。

**版本号规则**：
- 扫描 `docs/` 目录下已有的 `{op_name}_详设_V1.*.md` 文件
- 取最大版本号 +1 作为本次版本号（如已有 V1.0 → 本次 V1.1；已有 V1.2 → 本次 V1.3）
- 若无已有版本文件，提示需先完成 Step 2 生成 V1.0

**生成步骤**：
1. 读取 V1.0 全文
2. 复制 V1.0 全部内容作为 V1.x 的起点
3. 更新修订记录表（新增一行）
4. 在第 1 章前插入"关联需求"章节（增量需求背景）
5. 逐章对比增量需求，在需要变更的位置用【变更】/【新增】标签标注修改
6. 无变更的章节保持 V1.0 原文不动

**必须确保 V1.x 包含以下全部子章节（继承自 V1.0 + 新增关联需求）**：

```
# 【新增】0 关联需求     — 增量需求背景、功能定义、设计原则
# 1 算子概述
  ## 1.1 功能定义
  ## 1.2 支持场景
# 2 接口实现设计
  ## 2.1 PTA接口实现
  ## 2.2 aclnn接口实现
  ## 2.3 算子信息库（OpDef）
  ## 2.4 图模式设计
# 3 总体设计
  ## 3.1 交付方式
  ## 3.2 交付件汇总
  ## 3.3 TilingKey设计
  ## 3.4 模板列表
  ## 3.5 代码结构设计
# 4 模板设计
  ## 4.1 {OpName} 计算模板
    ### 4.1.1 计算流程图
    ### 4.1.2 Tiling实现设计
      #### 4.1.2.1 TilingData设计
      #### 4.1.2.2 核间tiling策略
      #### 4.1.2.3 核内tiling策略
    ### 4.1.3 Buffer设计
      #### 4.1.3.1 UB空间的分配
      #### 4.1.3.2 workspace的分配
      #### 4.1.3.3 L1/L0的分配
    ### 4.1.4 Kernel设计
    ### 4.1.5 异常场景设计
    ### 4.1.6 支持确定性计算设计
    ### 4.1.7 精度分析及设计
    ### 4.1.8 性能分析及设计
    ### 4.1.9 训推一致性设计
# 5 维测设计
  ## 5.1 可测试性
    ### 5.1.1 功能测试
    ### 5.1.2 精度测试
    ### 5.1.3 边界条件测试
  ## 5.2 可观察性
  ## 5.3 可维护性
    ### 5.3.1 关键风险与应对
    ### 5.3.2 后续演进
# 6 资料设计
```

**如果 V1.0 缺少上述任何章节，说明 V1.0 本身不完整，应先补齐 V1.0 再生成 V1.x。**

| 章节 | 处理方式 | 说明 |
|------|---------|------|
| 修订记录 | 新增一行 | 新版本号 + 变更描述 |
| 1 关联需求 | **【新增】整章** | 增量需求背景、功能定义、设计原则（V1.0 的第1章"算子概述"保留，关联需求作为新增前置章节） |
| 2 接口实现设计 | 保留原文 + 在变更处标注 | 无变更则完整保留 V1.0 原文 |
| 3 总体设计 | 保留原文 + 在变更处标注 | TilingKey/代码结构等有变更的小节在原文后追加【变更】内容 |
| 4 模板设计 | 保留原文 + 在变更处标注 | 核心章节，Tiling/Buffer/Kernel 的变更在对应小节原文后追加 |
| 5 维测设计 | 保留原文 + 追加新增用例 | 基线测试保留，追加【新增】FD 测试用例 |
| 6 资料设计 | 保留原文 + 追加新增文档 | 追加新增的文档条目 |

**关键原则**：
- **自包含**：V1.x 是完整文档，读者无需参考 V1.0
- **变更可追踪**：所有变更点用【变更】/【新增】标签标注，reviewer 可快速定位
- 第 4 章是核心，变更部分必须包含伪代码级别的增量实现方案
- 第 4 章各小节引用基线中的 Variant 选型，说明增量变更是否改变了 Variant 选择
- 向后兼容：明确功能关闭时的退化行为（与基线等价）
- 回归保障：第 5 章必须包含基线功能的回归测试用例

**输出路径**：`docs/{op_name}_详设_V1.{x}.md`（x 为自动递增版本号）

### 3.3 User Confirmation

逐章提交用户确认：

```
增量详设文档已生成，请 review 关键设计决策：
□ 基线引用：各章节的【基线】描述是否与基线详设一致
□ 接口层：【变更】方式（新增输入 / 激活预留参数 / 新增属性）
□ 数据通路：【变更】新功能的数据搬运路径
□ 循环结构：【变更】主循环如何适配新功能
□ 核间同步：【变更】AIV/AIC 角色分工变化
□ 向后兼容：功能关闭时是否退化到基线行为
□ 回归测试：基线功能的回归用例是否充分
□ 风险项：已识别风险及应对措施
□ 文档路径：docs/{op_name}_详设_V1.{x}.md
```

用户确认 → 文档生成完成。建议用户下一步调用 代码生成skill 进行代码实现。
用户有修改意见 → 更新对应章节后重新确认。

---

## Step B: New Operator Design Document Generation (路径 B)

从 golden + 需求描述生成全新算子的完整详设文档，无需现有代码基线。

### B.1 Collect Inputs

如用户未提供完整输入，使用 AskUserQuestion 引导：

```
AskUserQuestion:
  question: "请提供以下信息用于新建算子详设："
  header: "新建算子输入"
  options:
    - label: "我来提供"
      description: |
        需要：
        1. golden 参考实现路径（PyTorch model.py）
        2. 需求描述（特性列表、目标 layout/dtype/SoC 约束）
        3. 目标代码参考路径（可选，类似算子代码供参考架构）
    - label: "我先准备下"
      description: "稍后提供完整信息"
```

| 输入 | 必须/可选 | 说明 |
|------|----------|------|
| golden 参考实现 | 必须 | PyTorch model.py，定义算子语义和接口 |
| 需求描述 | 必须 | 特性列表、目标 layout、dtype、SoC 约束、设计原则 |
| 目标代码参考 | 可选 | 已有的类似生产算子代码，用于参考架构和代码风格 |

### B.2 Step N1: 需求分析与特性识别

1. **读取 golden**：提取输入/输出签名、计算语义、数据流
2. **从需求描述提取特性列表**：PA/Mask/Sparse/GQA/Sink/FD 等
3. **确定目标约束**：SoC（Ascend910B/C）、dtype（fp16/bf16）、layout（BSH/BNSD）
4. **平台检测**：检查需求描述或目标代码中是否含 `arch35/351x/Regbase` → A5，否则 → A3
5. **输出**：特性组合清单 + 接口规格 + 平台类型

### B.3 Step N2: 设计模式匹配

**根据特性组合和平台类型，从 design-patterns 中选择相关模式**：

| 特性组合 | 主骨架模式 | 辅助模式 |
|----------|-----------|----------|
| 基础 FA (无特性) | fa_01(四级流水) | — |
| FA + GQA | fa_01 + fa_03(分核) | — |
| FA + Sparse | fa_01 + fa_06(稀疏KV合并) | fa_08(负载均衡) |
| FA + PA | fa_01 + fa_04(workspace双缓冲) | — |
| FA + Sparse + PA | fa_01 + fa_06 + fa_04 | fa_08 |
| FA + Sparse + PA + Mask | fa_01 + fa_06 + fa_04 | fa_08 |
| FA + Sink (长序列) | fa_01 + fa_02(L1多缓冲sink常驻) | — |
| FA 反向 | fa_07(反向多Cube流水) | fa_01 |

**执行步骤**：
1. Read 匹配到的所有 design-pattern 参考文档（根据平台选择 `references/a3-design-patterns/` 或 `references/a5-design-patterns/`）
2. 如有目标代码参考，Read 其架构作为主骨架
3. 确定主流水编排方案（三级/四级/五级）
4. 确定 AIC/AIV 分工和同步拓扑
5. **输出**：设计模式选择理由 + 主骨架方案

### B.4 Step N3: 详设文档生成

按照 [references/design-doc-template.md](references/design-doc-template.md) 模板生成完整 6 章详设。

**与路径 A Step 2 的区别**：
- 路径 A：从现有代码中提取设计决策（描述现状）
- 路径 B：从 golden + design-patterns 推导设计决策（从零设计）

**生成顺序**：

```
第 1 章：算子概述
  ├─ 从 golden 提取功能定义和计算公式
  ├─ 从需求描述提取支持场景（Layout × dtype × 场景）
  └─ 明确设计原则（零框架侵入、向后兼容等）

第 2 章：接口实现设计
  ├─ 从 golden 签名推导 OpDef（输入/输出/属性）
  ├─ 推导 aclnn 两段式接口
  └─ 确定数据类型约束

第 3 章：总体设计
  ├─ TilingKey 设计（从特性组合推导模板参数）
  ├─ 模板列表（规格 × 场景 × 模板）
  └─ 代码结构设计（按 directory-template 规范）

第 4 章：模板设计（核心）
  ├─ 4.1.1 计算流程图
  │   ├─ Read fa_01（流水模型）——根据平台选择 a3 或 a5 目录
  │   ├─ 确定流水级数和同步拓扑
  │   └─ 绘制完整数据通路图和时序图
  ├─ 4.1.2 Tiling 实现设计
  │   ├─ Read fa_03（分核策略）
  │   ├─ 设计 TilingData 字段表
  │   ├─ 设计核间 tiling 策略（选择 Variant + 伪代码）
  │   └─ 设计核内 tiling 策略（切分参数 + 主循环结构）
  ├─ 4.1.3 Buffer 设计
  │   ├─ Read fa_05（UB 分配）
  │   ├─ Read fa_04（workspace 布局）
  │   ├─ Read fa_02（L1 多缓冲）
  │   └─ 设计完整 Buffer 分配表
  ├─ 4.1.4 Kernel 设计
  │   ├─ Read fa_01 + fa_04（同步协议）
  │   ├─ 如有 Sparse：Read fa_06（稀疏KV合并）
  │   ├─ 如有 Gather/Scatter：Read fa_08
  │   ├─ 设计 CrossCore flag 协议表
  │   └─ 设计主循环伪代码（Cube 服务 + Vector 服务）
  ├─ 4.1.5 异常场景设计（边界条件处理表）
  ├─ 4.1.7 精度分析（累积误差风险点）
  └─ 4.1.8 性能分析（流水利用率目标、带宽瓶颈）

第 5 章：维测设计
  ├─ 功能测试用例（覆盖所有特性组合）
  ├─ 精度测试（golden 对比标准）
  └─ 边界条件测试

第 6 章：资料设计
```

**第 4 章生成的核心约束**：
- 每个设计决策必须标注**选型理由**和**Trade-off**
- 引用 design-pattern 中的 Variant 对比表（根据平台选择正确目录）
- 必须包含伪代码级别的实现方案
- 必须包含完整的 Buffer 分配表（L1/L0/UB/workspace 每个 buffer 的大小、数量、用途）
- 必须包含完整的 CrossCore flag 协议表（flag 名、设置点、等待点、含义）
- 必须包含边界条件处理策略表

**边界条件处理策略**（必须在 4.1.5 中明确）：

| 边界场景 | 处理策略 |
|----------|----------|
| seq_len 不对齐 block_M | 尾块 mask + 有效长度判定 |
| topk 不对齐 block_I | 无效 index 填充 -1 + gather 后 zero-fill |
| actual_kv_len < 预分配 | causal_limit 动态计算 + block_table 越界保护 |
| head_kv 不对齐 | padding 到 next_power_of_2 + 输出时 mask |
| 多 batch 不等长 | per-batch actual_len + 跳过无效 query |

**输出路径**：`docs/{op_name}_详设_V1.0.md`

### B.5 Step N4: User Confirmation

```
请确认新建算子详设文档：
□ 算子概述：功能定义和计算公式是否与 golden 一致
□ 接口设计：OpDef/aclnn 参数是否完整覆盖 golden 签名
□ 特性识别：特性组合清单是否正确
□ 设计模式选择：Variant 选型和理由是否合理
□ 模板设计（核心）：
  - 流水编排：流水级数和时序图是否合理
  - 分核策略：核间分配方案是否正确
  - Tiling 策略：切分参数和主循环结构是否完整
  - Buffer 布局：L1/L0/UB/Workspace 分配是否合理
  - 同步协议：CrossCore flag 配对是否完整
  - 边界条件：所有边界场景是否覆盖
□ 性能目标：是否明确了量化的性能目标
□ 文档路径：docs/{op_name}_详设_V1.0.md
```

用户确认 → 详设生成完成。建议下一步调用 `ascendc-fa-new-operator-codegen` skill 进行代码生成。
用户有修改意见 → 更新对应章节后重新确认。

**路径 B 产出的 V1.0 后续可走路径 A 的 Step 3 做增量迭代**（如需添加新特性）。

---

## Step 2.5: 设计检查清单

当用户选择"设计检查清单"任务时执行。

### 检查模块

| 模块 | 检查项 | 适用平台 |
|------|--------|---------|
| FA 通用 | F1-F9（流水/同步/分核/Buffer/精度/Mask/确定性/异常/同步正确性） | 所有 |
| Regbase 专项 | R1-R3（VF函数/Scalar前移/寄存器分配） | A5 |
| 流水效率 | E1-E3（Bubble分析/Double Buffer/带宽瓶颈） | 推荐执行 |

### 交互模式

使用 AskUserQuestion 选择：

```
AskUserQuestion:
  question: "请选择检查模式："
  header: "检查模式"
  options:
    - label: "逐项引导模式"
      description: "每次提出一个设计问题，根据回答决定下一步"
    - label: "批量填写模式"
      description: "一次性列出所有待确认的设计决策点"
    - label: "仅 FA 通用检查"
      description: "执行 F1-F9 检查项"
    - label: "全量检查"
      description: "执行 F1-F9 + R1-R3 + E1-E3"
```

### 检查执行

Read [references/checklists/](references/checklists/) 目录下的对应检查清单文档：

- `checklist_fa_common.md` — F1-F9 检查项
- `checklist_regbase.md` — R1-R3 检查项（A5 平台）
- `checklist_efficiency.md` — E1-E3 检查项

---

## Output Summary

| 步骤 | 输出文件 | 描述 |
|------|---------|------|
| Step 1 | `docs/{op_name}_代码梳理.md` | 算子代码架构全面梳理（是什么） |
| Step 2 / Step N3 | `docs/{op_name}_详设_V1.0.md` | 设计决策文档（路径 A 从代码提取 / 路径 B 从零设计） |
| Step 3 | `docs/{op_name}_详设_V1.{x}.md` | 增量需求的变更设计文档（V1.0→变更→变更后），x 自动递增 |

## References

- [Code Analysis Template](references/code-analysis-template.md) — 代码梳理文档标准模板
- [Design Document Template](references/design-doc-template.md) — 6 章详细设计文档标准模板
- [A3 Design Patterns](references/a3-design-patterns/) — A3 平台设计决策依据
- [A5 Design Patterns](references/a5-design-patterns/) — A5 平台设计决策依据
- [Baseline Feature Cards](references/baselines/) — 8 个基座算子特征卡
- [Design Checklists](references/checklists/) — FA 设计检查清单