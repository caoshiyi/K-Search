# FA 详细设计文档结构与生成规范

输出文档**严格对齐**官方《算子详细设计说明书 v1.0》模板结构。FA 类算子在第 4 章计算模板内容上有专属扩展（以流水阶段为核心），但章节编号和格式必须与模板一致。

**图表绘制规范**：
- **计算流程图**（4.1.1） **必须** 使用 Mermaid `flowchart` / `graph` 语法（按阶段串接 vec0/mm1/vec1/mm2/vec2 等节点的数据流向 + 控制流），并附伪代码
- **禁止** 在 4.1.1 节使用 `sequenceDiagram`（流水时序图），本节只要计算流程图，不要时序图
- 同步事件图 **应尽量** 使用 Mermaid 序列图（如需要单独绘制）
- Buffer 布局图可使用表格或 Mermaid

**全局规范**：
- **需求变更追溯**：§2 各子节、§3.5 每处变更必须附变更表（需求编号/需求标题/变化点）
- **S1 增量场景**：变更表中标注"新增/修改/无"；无变更的章节仍保留框架并注明继承来源
- **【强制】继承来源明确性**：禁止使用模糊表述如"继承V1.0"、"与基线一致"、"继承基座设计"。必须明确写出继承自哪个基座算子的哪个具体设计（如"继承自FAS的五级流水设计"、"继承自FIAS的KV Cache数据通道"）。每个继承点需标注基座算子名+具体模块名。

---

## 文档结构

```
# [算子名] 详细设计说明书

<center>**修订记录**</center>             ← HTML 居中格式
<div align="center">
| 日期 | 修订版本 | 修改描述 | 作者 |
</div>

# 1 算子概述
# 2 接口实现设计
  ## 2.1 PTA接口实现                        ← 函数原型 + 逐参数说明 + 变更表
  ## 2.2 aclnn接口实现                      ← 函数原型 + 8列参数表 + 变更表
  ## 2.3 算子信息库                         ← 完整 OpDef C++ 代码 + 变更表
  ## 2.4 图模式设计                         ← 按需 + 变更表
# 3 总体设计
  ## 3.1 交付方式                           ← 3行表格（交付内容/代码承载/期望时间）
  ## 3.2 交付件汇总                         ← 13项标准 checklist
  ## 3.3 TilingKey设计
  ## 3.4 模板列表                           ← 规格/应用场景/模板/核间切分/核内切分/备注
  ## 3.5 代码结构设计                       ← 表格含 变化点/需求编号/需求标题 列
# 4 模板设计                               ← ★★ 核心
  ## 4.1 [模板名]计算模板                   ← 嵌套在模板下
    ### 4.1.1 计算流程图                    ← FA: 流水阶段表 + 同步矩阵 + Mermaid flowchart + 伪代码（不要时序图）
    ### 4.1.2 Tiling实现设计
      #### 4.1.2.1 TilingData设计           ← 完整代码 + 字段表(含变更描述/需求编号)
      #### 4.1.2.2 核间tiling策略和数据地址偏移计算
      #### 4.1.2.3 核内tiling策略和数据地址偏移计算
    ### 4.1.3 Buffer设计
      #### 4.1.3.1 UB空间的分配             ← 列: UB/Position/块大小/BUF_NUM/数据类型/总大小/备注
      #### 4.1.3.2 workspace的分配          ← 列: WorkSpace/块大小/BUF_NUM/数据类型/大小/备注
      #### 4.1.3.3 L1\L0的分配              ← 列: Buffer/块大小/BUF_NUM/数据类型/大小/备注
    ### 4.1.4 Kernel设计                    ← FA: 顶层调度 + 各流水阶段伪代码 + VF函数设计 + 【新增】同步正确性检查
    ### 4.1.5 异常场景设计
    ### 4.1.6 支持确定性计算设计
    ### 4.1.7 精度分析及设计                ← 强制
    ### 4.1.8 性能分析及设计                ← 强制
    ### 4.1.9 训推一致性设计
  ## 4.2 [模板名]计算模板                   ← 多模板时重复，可引用 4.1 内容
# 5 维测设计                               ← DFX 三维度
# 6 资料设计

附录：代码生成任务拆分表                    ← AI 编码入口索引
```

---

## 各章节细节规范

### 修订记录

必须使用 HTML 居中格式：

```html
<center>**修订记录**</center>
<div align="center">

| 日期 | 修订版本 | 修改描述 | 作者 |
| :---: | :---: | :---: | :---: |
| yyyy-mm-dd | 1.0 | 详细设计初稿 | 姓名/工号 |
| yyyy-mm-dd | 1.1 | 新增xx特性（增量修改处需与特性关联） | 姓名/工号 |

</div>
```

### 第 1 章 算子概述

包括功能定位、计算公式、支持场景（Layout × 数据类型 × 场景矩阵）。

### 第 2 章 接口实现设计

#### 2.1 PTA 接口实现

三部分组成：

**1. 函数原型**（代码块）：
```
npu_xxx_op(input1, input2, *, optional_param1=default1, ...) -> (Tensor, ...)
```

**2. 参数说明**（逐参数详细描述）：
FA 接口参数通常较多（15-20+），每个参数必须按以下格式逐一说明：
- **参数名**（`类型`）：必选/可选参数，描述。数据类型支持`xxx`，数据格式要求为`ND`。是否支持非连续Tensor。

**3. 变更表**（S1 增量必须包含）：

```markdown
**PTA接口实现变更，需体现变化部分：**

| **需求编号** | 需求标题 | 变化点 |
| --- | --- | --- |
| US202601290xxx | xxx特性 | 新增XX参数 |
```

**【强制】无论是否变更，函数原型和参数说明都必须完整列出**：
- 即使 PTA 接口与基座完全一致、无任何变更，也必须给出完整的 PTA 函数原型（"1. 函数原型"段）和逐参数详细说明（"2. 参数说明"段）
- 接口完全继承时，从基座算子的 PTA 接口实现中复制完整签名和参数说明，写入本节
- 仅在第 3 段"变更表"中标注 "无变化"
- ❌ 禁止以 "与基座一致，省略接口签名" / "PTA 接口继承基座，详见基座文档" 等表述代替具体接口签名
- ✅ 正确做法：列出完整接口 → 在变更表标注"无变化"

#### 2.2 aclnn 接口实现

三部分组成：

**1. 函数原型**（代码块）：
```cpp
aclnnStatus aclnnXxxOpGetWorkspaceSize(
    const aclTensor *input1,
    ...
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

**2. 参数表**（8 列标准格式）：

```markdown
| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| input1 | 输入 | 描述 | -不支持空Tensor: [B,S,N,D] | FLOAT16, BF16 | ND | 4 | √ |
```

**3. 变更表**（同 2.1 格式）。

**【强制】无论是否变更，函数原型和参数表都必须完整列出**：
- 即使 aclnn 接口与基座完全一致、无任何变更，也必须给出完整的 `aclnnXxxOpGetWorkspaceSize` 和 `aclnnXxxOp` 函数原型，以及完整的 8 列参数表
- 接口完全继承时，从基座算子的 aclnn 接口实现中复制完整签名和参数表，写入本节
- 仅在第 3 段"变更表"中标注 "无变化"
- ❌ 禁止以 "与基座一致，省略接口签名" / "aclnn 接口继承基座，详见基座文档" 等表述代替具体接口签名
- ✅ 正确做法：列出完整接口 → 在变更表标注"无变化"

#### 2.3 算子信息库

两部分：

**1. 完整 OpDef C++ 代码**（含 ascend910_95 配置）：
```cpp
namespace ops {
class XxxOp : public OpDef {
public:
    explicit XxxOp(const char *name) : OpDef(name) {
        this->Input("xxx")...;
        this->Output("xxx")...;
        this->Attr("xxx")...;
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        // A5 配置
        OpAICoreConfig config_95;
        config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "{op_name}_apt");
        this->AICore().AddConfig("ascend910_95", config_95);
    }
};
```

S1 无修改时，仍列出完整代码并标注"无变更"。

**2. 变更表**（同 2.1 格式）。

#### 2.4 图模式设计

按需填写 + 变更表。

### 第 3 章 总体设计

#### 3.1 交付方式

```markdown
| 类型 | 描述 | 备注 |
| --- | --- | --- |
| 交付内容 | | CANN包/自定义算子包/torch_npu等 |
| 代码承载 | | XX仓XX目录 |
| 期望交付时间 | | |
```

#### 3.2 交付件汇总

**必须使用 13 项标准 checklist**，且表头必须严格为 `序号 | 交付件 | 是否需要 | 涉及变动 | 备注` 五列：

```markdown
| **序号** | **交付件** | **是否需要** | **涉及变动** | **备注** |
| :---: | --- | :---: | :---: | --- |
| 01 | pta接口适配 | 是/否 | 是/否 | 必选 |
| 02 | pta接口文档 | 是/否 | 是/否 | 必选 |
| 03 | aclnn接口适配 | 是/否 | 是/否 | 必选 |
| 04 | aclnn接口文档 | 是/否 | 是/否 | 必选 |
| 05 | GE图模式适配 | 是/否 | 是/否 | 可选，GE入图必选 |
| 06 | AclGraph图模式适配 | 是/否 | 是/否 | 可选，AclGraph入图必选 |
| 07 | 算子原型 | 是/否 | 是/否 | 可选，GE入图必选 |
| 08 | OpDef定义（信息库） | 是/否 | 是/否 | 必选 |
| 09 | 算子tiling函数 | 是/否 | 是/否 | 必选 |
| 10 | 算子kernel实现 | 是/否 | 是/否 | 必选 |
| 11 | 算子二进制配置 | 是/否 | 是/否 | 必选 |
| 12 | 算子inferShape/inferDataType | 是/否 | 是/否 | 可选，GE入图必选 |
| 13 | 图融合pass | 是/否 | 是/否 | 可选 |
```

**【强制】"是否需要"栏填写规则**（对每一行 13 项交付件都必须填写）：

| 填写值 | 含义 | 适用场景 |
| :---: | --- | --- |
| **是** | 该项需要交付 | 概设文档中明确要求；或基座算子已有该交付件且本次特性继承使用；或备注列标注"必选" |
| **否** | 该项无需交付 | 概设文档中明确不涉及该项；或备注列为"可选"且本次特性不涉及（如不入图则 GE/AclGraph/inferShape 等可填"否"） |

**信息源优先级**：
1. **概设文档明确说明** → 优先采纳概设文档结论
2. **基座算子的现有交付状态** → 增量场景下，基座已有的交付件默认填"是"
3. **备注列规则** → "必选"项默认"是"；"可选"项需根据特性场景判断
4. **缺省规则** → 概设和基座均未提供信息时，默认填"是"（即需要提供），并在备注列追加注释"待确认"

**"涉及变动"栏填写规则**：
- **是** → 本次特性涉及该交付件的修改/新增
- **否** → 该交付件本次特性不涉及变动（完全继承基座）

❌ 禁止留空、填"-"、填"待补充"等模糊值；"是否需要"栏不接受任何非"是/否"的值。

#### 3.4 模板列表

```markdown
| 规格 | 应用场景 | 模板 | 核间切分 | 核内切分 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 大BS场景 | 训练场景 | | | | 是否新特性引入，具体特性 |
```

#### 3.5 代码结构设计

**必须使用含变更追溯列的表格**：

```markdown
| 目录 | 内容 | 变化点 | **需求编号** | 需求标题 |
| --- | --- | --- | --- | --- |
| attention/XXOp/op_host/xx_def.cpp | 算子定义 | 无 | | |
| attention/XXOp/op_kernel/xx.h | 算子kernel实现 | 新增XX计算流程 | US... | xxx特性 |
```

### 第 4 章 模板设计（FA 专属内容）

**关键**：所有子节嵌套在 `4.1 [模板名]计算模板` 下（如 `4.1 SFAG计算模板`），编号为 4.1.1 ~ 4.1.9。多模板时重复为 4.2。

#### 4.1.1 计算流程图（FA 扩展为流水总览）

FA 算子此节必须包含三部分：

**1. 流水阶段表**（标注每个阶段的归属：vec0/mm1/vec1/mm2/vec2 等，对应 AIV/AIC）：

| 阶段 | 名称 | 执行核 | 计算内容 | 输入来源 | 输出去向 |
|------|------|--------|---------|---------|---------|
| mm1 | BMM1 | AIC | Q×K^T→S | L1(Q), L1(K) → L0A/L0B | L0C → workspace/UB |
| vec1 | Softmax | AIV | scale+mask+exp+norm | UB(S) | UB(P) |
| mm2 | BMM2 | AIC | P×V→O | L1(P), L1(V) → L0A/L0B | L0C → workspace/UB |
| vec2 | FlashUpdate | AIV | rescale+add+normalize | UB(O_tile) | UB → GM(attentionOut) |

**2. 同步事件矩阵**（事件 ID、发送方、接收方、语义、AIV1 偏移规则）：

| 事件ID | 名称 | 发送方 | 接收方 | 语义 | AIV1偏移 |
|--------|------|--------|--------|------|---------|
| 0 | SYNC_C1_V1 | AIC | AIV0 | BMM1完成→Softmax开始 | +16 |

**3. 计算流程图**（必须使用 **Mermaid flowchart 语法**）：

> ⚠️ **本节绘图规范**：
> - **必须使用 Mermaid `flowchart` / `graph` 语法**绘制**计算流程图**（按阶段串接 vec0/mm1/vec1/mm2/vec2 等节点的数据流向 + 控制流）
> - **禁止**使用 `sequenceDiagram`（流水时序图），本节不要求展示时序流水
> - 图节点必须明确标识阶段名（vec1/vec2/mm1/mm2/...）+ 执行核（AIV/AIC）
> - 节点之间的箭头标注：传递的数据（如 `S 矩阵`、`P 矩阵`、`O_tile`）或同步事件（如 `SYNC_C1_V1`）
> - 关键分支（如 FD 模式 / 非 FD 模式、首轮 / 非首轮）使用 `子图(subgraph)` 或菱形判定节点表示
> - 图后**必须附伪代码描述**整体流程（覆盖核间调度顺序、循环结构、关键分支条件）

**Mermaid 计算流程图标准模板**：

```mermaid
flowchart TD
    Start([Start: Init / Tiling 解析])
    Start --> BN_Loop{BN 循环<br/>bnIdx ∈ [bN2Start, bN2End)}
    BN_Loop --> S1_Loop{S1G 循环<br/>gS1Index ∈ [gs1Start, gs1End)}
    S1_Loop --> S2_Loop{S2 循环<br/>s2LoopCount = 0..s2LoopLimit}

    subgraph MainPipeline[主流水阶段]
        direction LR
        Vec0[vec0: MergeKv<br/>AIV] -- kvMergeGm --> MM1[mm1: BMM1 Q×K^T<br/>AIC]
        MM1 -- mm1ResGm<br/>SYNC_C1_V1 --> Vec1[vec1: Softmax<br/>AIV]
        Vec1 -- vec1ResGm<br/>SYNC_V1_C2 --> MM2[mm2: BMM2 P×V<br/>AIC]
        MM2 -- mm2ResGm<br/>SYNC_C2_V2 --> Vec2[vec2: FlashUpdate<br/>AIV]
    end

    S2_Loop --> MainPipeline
    Vec2 --> S2_End{S2 循环结束?}
    S2_End -- No --> S2_Loop
    S2_End -- Yes --> S1_End{S1 循环结束?}
    S1_End -- No --> S1_Loop
    S1_End -- Yes --> BN_End{BN 循环结束?}
    BN_End -- No --> BN_Loop
    BN_End -- Yes --> EndCheck{是否 FD 模式?}
    EndCheck -- Yes --> FDCombine[FD Combine 规约<br/>AIV]
    EndCheck -- No --> Output([Output: attentionOutGm])
    FDCombine --> Output
```

**伪代码描述（必填）**：

```
Process() {
    Init();
    for (bnIdx in [bN2Start, bN2End)) {                       // BN 循环
        for (gS1Index in [gs1Start, gs1End)) {                // S1G 循环
            for (s2LoopCount = 0..s2LoopLimit) {              // S2 循环
                vec0_MergeKv();                  // AIV: Gather + Dequant
                mm1_BMM1();                      // AIC: Q × K^T  → mm1Res
                vec1_Softmax();                  // AIV: Scale + Mask + Softmax + Cast
                mm2_BMM2();                      // AIC: P × V    → mm2Res
                vec2_FlashUpdate();              // AIV: rescale + add + normalize
            }
        }
    }
    if (FLASH_DECODE) {
        SyncAll();
        FDCombine();                              // AIV: 多核局部 O 规约
    }
}
```

> 备注：
> - 上述模板适用于含 vec0 的 5 阶段流水（如 SFA / QSFA）；不含 vec0 的 4 阶段流水（如 FAS / FIA）将首节点替换为 mm1 直接读 K 即可
> - 各 stage 内部的 SetFlag/WaitFlag 在 4.1.4 Kernel 设计中展开
> - 不要再用 `sequenceDiagram` 画时序流水图，本节只要"流程图"（节点 + 数据/控制流向）

#### 4.1.2 Tiling 实现设计

##### 4.1.2.1 TilingData 设计

两部分：

**1. 完整 TilingData 代码**：
```cpp
BEGIN_TILING_DATA_DEF(XxxTilingData)
TILING_DATA_FIELD_DEF(uint32_t, fieldName);
...
END_TILING_DATA_DEF;
```

**2. 字段说明表**（含变更描述和需求编号）：

| 字段 | 类型 | 功能描述 | 变更描述 | 需求**编号** |
| --- | --- | --- | --- | --- |
| fieldName | uint32_t | 功能说明 | 新增 | US... |

##### 4.1.2.2 核间tiling策略和数据地址偏移计算

具体实现逻辑说明（公式、伪代码）。

##### 4.1.2.3 核内tiling策略和数据地址偏移计算

具体实现逻辑说明。

#### 4.1.3 Buffer 设计

分三个子节，表格列必须与模板对齐：

##### 4.1.3.1 UB空间的分配

| UB | Position | 块大小 | BUF_NUM | 数据类型 | 总大小 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| xxxQueue | VecIn/VecOut | 计算公式=结果值 | DB(×2)/1 | FP32 | 块大小×BUF_NUM | |
| xxxBuffer | Tbuf | 计算公式=结果值 | 1 | FP32 | 同块大小 | |

FA 特殊列说明：
- Position 列：FA 使用 `CROSS_CORE_SYNC_BOTH` 标注 CV 共享 buffer
- 备注列：标注 CrossCore 读写方（如"AIC写/AIV读"）
- **块大小列：必须给出 `dim1 × dim2 × sizeof(dtype) = xxx KB` 格式的计算公式和结果**
- **总大小列：必须给出实际字节数（= 块大小 × BUF_NUM）**

##### 4.1.3.2 workspace的分配

| WorkSpace | 块大小 | BUF_NUM | 数据类型 | 大小 | 备注 |
| --- | --- | --- | --- | --- | --- |
| xxxRes | 计算公式 | | | 实际字节数 | |

##### 4.1.3.3 L1\L0的分配

| Buffer | 块大小 | BUF_NUM | 数据类型 | 大小 | 备注 |
| --- | --- | --- | --- | --- | --- |
| L1 xxxBuf | 计算公式=结果值 | 3(triple) | FP16 | 块大小×BUF_NUM | |
| L0A | 计算公式=结果值 | DB(×2) | | 实际字节数 | Matmul库管理 |
| L0B | 计算公式=结果值 | DB(×2) | | 实际字节数 | Matmul库管理 |
| L0C | 计算公式=结果值 | DB(×2) | | 实际字节数 | Matmul库管理 |

##### 【必填】硬件容量约束校验汇总

**此表不可省略**，必须在 §4.1.3 末尾输出：

| 存储层级 | 硬件容量 | 代码初始化 | 实际使用 | 使用率 | 特性影响 | 校验结果 |
|---------|---------|-----------|---------|-------|---------|---------|
| **UB** | 248 KB | 各Queue分配 | xxx KB | xx% | 无新增 / 新增 xxx KB | ✅通过 / ❌超限 |
| **L1** | 1 MB | xxx KB | xxx KB | xx% | 复用 / 新增 xxx KB | ✅通过 / ❌超限 |
| **L0A** | 64 KB | 64 KB | xxx KB | xx% | 无变化 | ✅通过 / ❌超限 |
| **L0B** | 64 KB | 64 KB | xxx KB | xx% | 无变化 | ✅通过 / ❌超限 |
| **L0C** | 256 KB(A5)/128 KB(A3) | xxx KB | xxx KB | xx% | 无变化 | ✅通过 / ❌超限 |
| **Workspace** | GM | 动态计算 | xxx MB | — | 无变化 / 新增 xxx | ✅通过 / ❌超限 |

规则：
- 使用率 >90%：添加 ⚠️ 风险提示
- 校验不通过（超限）：必须附解决方案
- S1 增量场景：特性影响列说明新特性是复用还是新增 buffer

#### 4.1.4 Kernel 设计

FA 算子此节扩展为：

**顶层调度伪代码**（Process/ExecuteTask 结构）：
```
Process() {
    for (...) {
        ProcessPreload();
        ProcessNotFirst();
    }
}
```

**各流水阶段伪代码**（对每个 Cx/Vx 阶段）：
```
// 阶段 C1: BMM1 (Q × K^T)
void StageBMM1() {
    WaitFlag(SYNC_V1_C2);
    LoadToL0A(Q_tile);
    MatMul(S_tile, Q_tile, K_tile);
    SetFlag(SYNC_C1_V1);
}
```

**A5 VF 函数设计**（若涉及 regbase）：
- 4.1.4 内子节，列出每个 `__simd_vf__` 函数的签名、MicroAPI 序列、寄存器需求

**A5 否定性论证**（若不涉及 regbase）：
- 4.1.4 内子节，说明为什么高层 API 足够，列出不修改的 regbase 文件范围

**【新增】同步正确性检查子节**：
- 基于 F9 检查结果，输出同步正确性判定表
- 包含：配对完整性、循环依赖、CrossCore 访问序、事件 ID 冲突、Ping-Pong 隔离检查
- 必须全部通过方可进入编码阶段

#### 4.1.5 异常场景设计

空 tensor、异常 shape、不支持的 dtype 等处理方式。

#### 4.1.6 支持确定性计算设计

是否支持确定性计算；若涉及，完整设计（方案、同步、保证论证）。

#### 4.1.7 精度分析及设计（强制，不可省略）

必须包含：
- **完整精度链条表**（每个计算步骤的输入→计算→输出 dtype）
- **Online Softmax 精度说明**：m(max)/l(sum) 统计量精度、rescale 精度
- **升降精度节点标注**：哪些 Cast 是显式的、哪些是 Cube 自动的
- **精度验收标准**：atol/rtol

#### 4.1.8 性能分析及设计（强制，不可省略）

必须包含：
- **瓶颈类型判断**（Compute Bound / Memory Bound）
- **Bubble 分析结果**（各阶段耗时、Bubble 比例）
- **Preload/Double Buffer 方案**
- **A5 附加**：VF 函数性能分析、寄存器利用率

#### 4.1.9 训推一致性设计

按需填写。

### 第 5 章 维测设计

按需填写，可包含：
- 算子可测试性：接口、功能、性能、精度
- 算子可观察性：正常/异常场景测试目标输出
- 算子可维护性：后续演进与维护

### 第 6 章 资料设计

按需填写。

---

## 增量开发场景的文档特殊处理

### S1（特性注入）文档规范
- 所有章节框架保留（即使无变更也保留节标题）
- 变更表在每个涉及变更的章节出现，标注需求编号
- 第 4 章仅详细设计 **新增/修改** 的部分
- **【强制】对未修改的模块，必须明确写出继承来源**：
  - ✅ 正确：`继承自FAS的五级流水设计（Vec0→BMM1→Vec1→BMM2→Vec2）`
  - ✅ 正确：`继承自FIAS的KV Cache数据通道设计`
  - ❌ 错误：`与基座一致`、`继承V1.0`、`与基线相同`
- 必须包含兼容性说明：新特性不劣化现有功能

### S2（变体 Fork）文档规范
- 第 4 章完整设计，每个章节标注与基座的差异点
- 差异点使用 `[变更]` 标签突出显示
- **【强制】每个子章节必须写出继承自哪个基座的哪个设计**，禁止使用模糊表述

### S3（架构迁移）文档规范
- 重点在 4.1.4（VF 函数）和 4.1.3（Buffer 重新分配）
- **【强制】对计算逻辑未变的部分**，必须写出具体继承来源：
  - ✅ 正确：`继承自arch32版本的计算流程，逻辑不变，仅regbase适配`
  - ❌ 错误：`继承V1.0`、`与旧版本一致`

---

## 代码生成任务拆分表（附录）

| 任务编号 | 文件 | 生成内容 | 依赖任务 | 对应文档节 |
|---------|------|---------|---------|-----------|
| T1 | `op_host/{op}_tiling.h` | TilingData 结构体 + CompileInfo | 无 | 4.1.2.1 |
| T2 | `op_host/{op}_tiling.cpp` | 核间/核内 Tiling 逻辑 | T1 | 4.1.2.2/2.3 |
| T3 | `op_host/{op}_def.cpp` | OpDef（含 A5 的 ascend910_95） | T1 | 2.3 |
| T4 | `op_kernel/{op}.h` | Kernel 主类 + Process/ExecuteTask | T2 | 4.1.4 |
| T5 | `op_kernel/{op}_vector_api.h` | Vector 阶段实现 | T4 | 4.1.4 |
| T5-A5 | `op_kernel/arch35/{op}_vf_*.h` | A5 VF 函数（regbase 路径） | T5 | 4.1.4 |
| T6 | `op_kernel/{op}.cpp` | Kernel 入口函数 | T4 | 4.1.4 |
| T7 | `op_kernel/{op}_common.h` | 公共定义、常量、枚举 | T1 | 4.1.1 |

> 注：T5-A5 仅在目标平台含 A5 且使用 regbase 路径时生成。
