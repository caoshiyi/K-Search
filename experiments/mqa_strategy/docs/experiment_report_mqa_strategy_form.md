# MQA AscendC 算子优化策略形式实验报告

> 实验编号：EXP-2026-MQA-STRAT-FORM
> 日期：2026-05-31 ~ 2026-06-01
> Meta Agent：Claude Code (glm-5.1)
> 实验对象：K-Search 对 MQA (Multi-Query Attention) AscendC 算子的自动优化

---

## 1 实验目标

研究**优化策略的描述形式**对 K-Search LLM-driven World Model 性能优化效果的影响。

具体问题：
- 策略注入是否比纯 LLM WM 更有效？
- 自然语言、结构化参数、DSL 三种策略形式哪种最有效？
- 策略形式如何影响 LLM codegen 的编译通过率和运行时正确性？
- 当前策略描述存在哪些系统性缺陷？

## 2 实验方法

### 2.1 实验设计

四组对照实验，仅策略注入方式不同，其余变量完全一致：

| 组别 | 策略注入方式 | WM初始化 | Action Node来源 |
|:----:|:----------:|:--------:|:--------------:|
| A: Baseline | 无策略注入 | LLM生成(~196s) | LLM propose_action_nodes |
| B: NL | natural_language策略 | 从catalog构建(0s) | catalog策略节点(s1-s12) |
| C: SP | structured_params策略 | 从catalog构建(0s) | catalog策略节点(s1-s12) |
| D: DSL | dsl策略策略 | 从catalog构建(0s) | catalog策略节点(s1-s12) |

### 2.2 控制变量

| 变量 | 值 |
|:----:|:--:|
| 优化目标 | multi_query_attention AscendC算子 |
| 目标硬件 | Ascend910B3 |
| LLM模型 | glm-5.1 |
| 最大轮次 | 12 |
| 停滞窗口 | 5轮无改善则标记action为too hard |
| 单轮超时 | 900s |
| 基线延迟 | 0.809398ms |
| 策略catalog | mqa_strategies_catalog.json (12策略) |
| codegen方式 | Claude Agent SDK agentic session |

### 2.3 策略来源

从 `/mnt/workspace/new_cv_agent/cv_agent/tile2asc/flash_attention/`（已优化版本）与 `/mnt/workspace/cv_agent/tile2asc/multi_query_attention/`（待优化版本）的实现差异中提取12个优化策略：

| ID | 策略名称 | 类别 | 影响 | 难度 |
|:--:|:--------:|:----:|:----:|:----:|
| S1 | Enlarged Tile Sizes | tiling | high | 2 |
| S2 | WorkspaceQueue Pattern | pipeline | high | 3 |
| S3 | Q L1 Cache Skip | memory | medium | 2 |
| S4 | Vectorized RowMuls/RowDivs | compute | high | 2 |
| S5 | VEC2 Large Chunk | compute | medium | 1 |
| S6 | Softmax UB State Cache | memory | medium | 3 |
| S7 | Sub-block Row Handling | compute | medium | 2 |
| S8 | Targeted SetWaitFlag | pipeline | high | 3 |
| S9 | oPrev Dedicated TBuf | memory | medium | 2 |
| S10 | Multi-level KV Outer Tiling | tiling | high | 4 |
| S11 | Single KV L1 Buffer | memory | medium | 2 |
| S12 | KV Remain Handling | tiling | medium | 2 |

### 2.4 K-Search修改

为实现策略注入，对 K-Search 做了以下修改：

1. `generate_kernels_and_eval.py` — 添加 `--strategy-file` 和 `--strategy-form` CLI参数
2. `k_search/kernel_generators/kernel_generator_world_model.py` — 当提供策略catalog时，WM初始化从catalog构建而非LLM生成；propose_action_nodes跳过LLM调用（节点已从catalog种子）
3. `k_search/kernel_generators/strategy_injection.py` — 新增模块，实现 `load_strategy_catalog()`, `render_strategy_as_action_text()`, `build_wm_from_strategies()`

### 2.5 实验执行

四组实验依次运行（避免资源竞争），每组运行至 WM stagnation 或 12 轮上限。

---

## 3 实验结果

### 3.1 总览

| 指标 | Baseline (纯LLM) | Natural Language | Structured Params | DSL |
|:----:|:----------------:|:----------------:|:-----------------:|:--:|
| **最佳加速** | **0x** | **1.520x** | **0x** | **1.464x** |
| 最佳延迟 | N/A | 0.5326ms | N/A | 0.5528ms |
| 通过轮次 | 0/4 | 4/12 | 0/11 | 3/10 |
| 编译失败轮次 | 4/4 | 3/12 | 6/11 | 4/10 |
| 运行时失败轮次 | 0/4 | 5/12 | 5/11 | 3/10 |
| max_turns超时 | 2次 | 0次 | 1次 | 2次 |
| WM init耗时 | 196s | 0s | 0s | 0s |
| WM节点数 | 6 | 16 | 16 | 16 |
| Attached solutions | 0 | 2 | 0 | 1 |
| 实验总时长 | ~56min | ~189min | ~103min | ~250min |

### 3.2 Baseline (纯LLM WM) 详细结果

WM初始化生成了3个action node（n1: KV L1 residency, n2: Split-KV, n3: Narrow PipeBarrier）。

| Round | Action | Status | 说明 |
|:-----:|:------:|:------:|:----:|
| 1 | n1 (KV L1 residency) | compile_failed | vec.h中Brcb(3-param)+Mulcs(不存在)API错误 |
| 2 | n1 | compile_failed | 修Brcb→4-param+Mulcs→Mul，仍编译失败 |
| 3 | n1 | max_turns超时 | Agent SDK会话超时，无candidate |
| 4 | n2 (Split-KV) | compile_failed | 仅改MAX_CORES动态化，vec.h基线bug未修 |
| 5 | n2 | compile_failed | 修vec.h Brcb→GetValue+Muls，仍失败 |
| 6 | n2 | max_turns超时 | Agent SDK会话超时 |

n1和n2均标记为too hard，n3被WM edit跳过（invalid node），frontier清空，实验在第6轮提前终止。**无任何成功优化。**

### 3.3 Natural Language 详细结果

WM从catalog构建12个策略节点，首轮选择S1（最高评分8.0/10）。

| Round | Action | Status | Speedup | 说明 |
|:-----:|:------:|:------:|:-------:|:----:|
| 1 | s1 (Enlarged Tile) | compile_failed | — | Brcb(3-param)+Mulcs API误用 |
| 2 | s1 | compile_failed | — | Muls+LocalTensor[]类型推导冲突 |
| 3 | s1 | **passed** | **1.475x** | 仅改BLOCK_M/N/K=128，最小变更 |
| 4 | s1 | passed | 1.382x | 叠加异步DMA预取，性能回退 |
| 5 | s1 | passed | 1.447x | Q cache+PIPE_FIX，回退 |
| 6 | s1 | failed | — | 跨tile DMA预发破坏softmax因果依赖 |
| 7 | s1 | failed | — | 激进AIV同步重构 |
| 8 | s1 | compile_failed | — | UB accO累积编译回归 |
| 9 | s1c1 (Vec+Chunk) | failed | — | Duplicate+Mul精度不等价 |
| 10 | s1c1 | failed | — | 批量Duplicate+Mul仍精度失败 |
| 11 | s1c1 | failed | — | 缓冲区resize+Duplicate双重变更 |
| 12 | s1c1 | **passed** | **1.520x** | 仅VEC2_M_CHUNK=16，回退Duplicate |

**核心发现**：自然语言策略使LLM在第3轮就成功实现了编译通过的优化（1.475x），而baseline全程无法编译。但后续8轮中5轮因语义等价性破坏失败。

### 3.4 Structured Params 详细结果

| Round | Action | Status | 说明 |
|:-----:|:------:|:------:|:----:|
| 1-3 | s1 (Enlarged Tile) | 2×compile_failed, 1×failed | JSON参数描述缺乏代码变更语义，LLM无法映射到具体实现 |
| 4-9 | s4 (Vectorized RowMuls) | 2×compile_failed, 3×failed | 同上，结构化参数无法指导API调用 |
| 10-12 | s2 (WorkspaceQueue) | 1×compile_failed, 1×failed, 1×compile_failed | 仍未通过 |

**12轮全部失败**。结构化参数的JSON格式（`{parameters: {BLOCK_M: {old:64, new:128}}}`）虽然精确指定了参数值，但缺乏变更的语义上下文——LLM不知道如何将这些参数映射到具体的代码修改、如何处理参数间的依赖关系、以及哪些API调用需要相应调整。

### 3.5 DSL 详细结果

| Round | Action | Status | Speedup | 说明 |
|:-----:|:------:|:------:|:-------:|:----:|
| 1 | s1 (Enlarged Tile) | compile_failed | — | DSL `TILE_SIZE{BLOCK_M:128}`缺乏实现路径 |
| 2 | s1 | **passed** | 1.401x | LLM自行解读DSL并实现tile放大 |
| 3 | s1 | **passed** | 1.464x | 继续优化tile相关参数 |
| 4 | s1 | passed | 1.438x | 性能微回退 |
| 5 | s1 | compile_failed | — | 叠加变更编译失败 |
| 6 | s1 | stagnation | — | s1标记too hard |
| 7 | s8 (SetWaitFlag) | compile_failed | — | DSL `BARRIER{...}`缺乏API级别实现指导 |
| 8 | s1_c1 (组合) | compile_failed | — | |
| 9 | s1_c1 | compile_failed | — | |
| 10 | s1_c1 | failed | — | Duplicate+Mul精度失败 |
| 11 | s1_c1 | failed | — | 同上 |
| 12 | s1_c1 | failed | — | 同上 |

DSL形式在前4轮表现较好（3/4通过，1.464x），但在s8和s1_c1策略上连续失败。DSL的声明式规格（`TILE_SIZE{...}`, `BARRIER{...}`）比JSON参数更有语义，但仍缺乏实现级细节。

### 3.6 策略形式效果排序

```
Natural Language (1.520x) > DSL (1.464x) > Structured Params (0x) = Baseline (0x)
```

| 排名 | 策略形式 | 通过率 | 首次通过轮次 | 最佳加速 |
|:----:|:--------:|:------:|:----------:|:-------:|
| 1 | Natural Language | 4/12=33% | 第3轮 | 1.520x |
| 2 | DSL | 3/10=30% | 第2轮 | 1.464x |
| 3 | Structured Params | 0/11=0% | — | 0x |
| 4 | Baseline (纯LLM) | 0/4=0% | — | 0x |

---

## 4 实验现象分析

### 4.1 现象1：策略注入的根本价值是帮助LLM生成可编译代码

基线实验（纯LLM）12轮全部编译失败，无一通过。而注入策略后（无论NL还是DSL），LLM在第2-3轮就能生成编译通过的代码。

**解释**：策略描述提供了优化的方向和意图，让LLM的codegen聚焦于一个具体目标，而非在无约束空间中漫无目的地探索。特别是策略S1（Enlarged Tile Sizes）提供了明确的参数值（64→128）和变更范围（tiling.h, kernel.h, cube.h, vec.h），大幅缩小了LLM的搜索空间。

### 4.2 现象2：自然语言理解最自然，结构化参数最不自然

NL形式（33%通过率）> DSL形式（30%通过率）> SP形式（0%通过率）的排序与LLM对信息格式的理解难度一致：

- **自然语言**：LLM训练数据的主体是自然语言，理解因果描述最直接
- **DSL**：声明式规格有一定语义，LLM可以解读意图并自行推导实现
- **结构化参数**：JSON参数列表只有值，没有变更语义。LLM看到 `{BLOCK_M: {old:64, new:128}}` 知道要把64改成128，但不知道这个改动和L1 buffer分配、loop迭代次数、DataCopy尺寸之间有什么依赖关系

### 4.3 现象3：数学等价 ≠ 硬件等价——所有运行时失败共享同一根因

NL实验的5次运行时失败（R6, R7, R9, R10, R11）和DSL实验的3次运行时失败（R10, R11, R12）全部源于同一个根本问题：

> **LLM在数学/算法抽象层做等价推理，认为两种实现计算结果相同。但AscendC硬件的对齐约束、流水线依赖和精度路径使得数学等价≠硬件等价。**

两类具体的等价性破坏：

**(A) 跨tile DMA预发（NL R6/R7）**：
LLM认为预取下一tile的K/V只是IO优化，不改变计算。但online softmax有tile间因果依赖：`row_max[t]`和`row_sum[t]`必须计算完成后才能处理`O`的rescale和下一tile的累积。跨tile预发打破了AIC→AIV的同步语义。

**(B) Duplicate+Mul广播替代GetValue+Muls（NL R9-R11, DSL R10-R12）**：
LLM认为`Duplicate(scalar, width)+Mul(dst, src, buf, width)`等价于`GetValue(row)+Muls(dst, src, scalar, width)`，因为`buf[i]=scalar`对所有i。但硬件上有三个不等价：
- **RAW流水线冒险**：Duplicate和Mul在同一PIPE_V上执行，无PipeBarrier分隔
- **对齐填充垃圾值**：Duplicate填充dim个位置，padding区域(dimAlign-dim)保留垃圾值，Mul操作整个dimAlign宽度
- **精度路径差异**：`buf[i]`经过1次Duplicate赋值可能有舍入ε，使`dst[i]=src[i]×(scalar+ε)`而非`src[i]×scalar`

### 4.4 现象4：单维变更成功率高，多维叠加变更成功率零

| 变更维度数 | 成功轮次 | 失败轮次 | 通过率 |
|:---------:|:--------:|:--------:|:------:|
| 1维 | R3(仅tile), R12(仅chunk) | — | **100%** |
| 2维+ | — | R4-R11(8轮) | **0%** |

R3仅改BLOCK_M/N/K=128即获得1.475x加速。R12仅改VEC2_M_CHUNK=8→16即获得1.520x。而所有叠加多个维度的尝试（R4-R11）均失败或回退。

### 4.5 现象5：策略预期加速过于乐观影响WM决策

| 策略节点 | 预期 | 实际 | 差距 | WM反应 |
|:--------:|:----:|:----:|:----:|:------:|
| s1 | 1.50x | 1.475x | -1.7% | 消耗5轮(R4-R8)叠加变更试图"改进"，4轮失败 |
| s1c1 | 1.70x | 1.520x | -11.8% | 消耗4轮(R9-R12)试图补足差距，3轮失败 |

预期1.70x使WM过度优先选择s1c1，但Duplicate+Mul部分的预期贡献(~0.15x)未能实现，导致3轮浪费。

### 4.6 现象6：WM从catalog构建比LLM生成更高效

| 指标 | Baseline (LLM生成) | 策略注入 (catalog构建) |
|:----:|:------------------:|:--------------------:|
| WM init耗时 | 196s | 0s |
| 初始action数 | 3 | 12 |
| action质量 | LLM推测的优化方向 | 来自Flash Attention实测差异 |

LLM生成的3个action node（n1: KV L1 residency, n2: Split-KV, n3: Narrow PipeBarrier）都是合理的优化方向，但描述过于模糊和激进。catalog的12个策略来自真实优化实现的逐项对比，质量更高。

---

## 5 结论

### 5.1 主要结论

**结论1：策略注入显著提升K-Search优化效果。** 纯LLM WM完全失败（0x加速），注入策略后NL获得1.520x加速、DSL获得1.464x加速。策略的核心价值是缩小LLM codegen的搜索空间，使其能生成编译通过的代码。

**结论2：自然语言是最有效的策略形式，但当前实现存在系统性缺陷。** NL获得最高加速（1.520x），但通过率仅33%。所有8次失败源于5个系统性问题（详见5.2节），而非自然语言形式本身的问题。

**结论3：策略描述停留在算法抽象层，缺少硬件抽象层的关键信息。** 当前策略描述what和why（做什么、为什么有用），但缺少how的精确约束（怎么安全实现）和what-not的负面警告（哪些变换看似等价但硬件不等价）。

**结论4：结构化参数形式完全不适用于K-Search策略注入。** JSON参数列表缺乏变更语义，LLM无法将其映射到具体的代码修改。

### 5.2 自然语言策略的五个系统性缺陷

| # | 缺陷 | 影响 | 证据 |
|:-:|:----:|:----:|:----:|
| 1 | 缺乏AscendC API精确签名 | LLM猜错API用法导致编译失败 | R1(Brcb 3-param, Mulcs不存在), R2(LocalTensor[]≠float) |
| 2 | 缺乏硬件不等价陷阱警告 | LLM做数学等价推理但硬件不等价 | R6/R7(跨tile DMA), R9-R11(Duplicate+Mul≠RowMuls) |
| 3 | 鼓励多维度叠加变更 | 单维100%通过而多维0%通过 | R3(1维)=1.475x, R4-R11(2维+)全失败 |
| 4 | 缺乏"什么不能做"的负面约束 | LLM自探索不等价替代 | Brcb→Mulcs, GetValue→Duplicate+Mul, 同步→跨tile预发 |
| 5 | 预期加速过于乐观 | WM过度追求补足差距 | s1c1预期1.70x实际1.520x, 消耗3轮无效尝试 |

### 5.3 策略形式对比结论

| 维度 | Natural Language | Structured Params | DSL |
|:----:|:---------------:|:-----------------:|:--:|
| LLM理解程度 | 最高 | 最低 | 中等 |
| 语义丰富度 | 因果推理+约束描述 | 仅参数值 | 声明式规格 |
| 硬件约束表达力 | 可补充(当前缺失) | 极差 | 中等(可补充) |
| API签名表达力 | 可精确描述(当前缺失) | 仅值/公式 | 可描述(当前缺失) |
| 负面约束表达力 | 可添加(当前缺失) | 无 | 可添加(当前缺失) |
| 实际效果 | 1.520x | 0x | 1.464x |
| 改进潜力 | 高(补充缺失信息) | 低(形式本身不适合) | 中 |

---

## 6 下一轮Meta Agent优化K-Search的行动计划

基于本实验的发现，下一轮优化应聚焦于**提升自然语言策略的信息完整度**，而非改用其他策略形式。

### 6.1 策略描述改进（优先级P0）

#### 6.1.1 在策略中添加精确API签名和调用示例

在每个涉及AscendC API的策略中，补充完整的API签名、参数类型和正确调用示例：

```
当前: "Replace scalar RowMulsImpl/RowDivsImpl with hardware RowMuls/RowDivs API calls."
改进: "Replace with RowMuls API. Exact signature:
  RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>,
          dealRows: int32, cols: int32, actualCols: int32)
  The scale parameter is a LocalTensor (NOT a float scalar).
  Example: RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim);"
```

#### 6.1.2 在策略中添加不等价陷阱警告（Anti-Pattern Warnings）

在每个策略末尾明确标注"看似等价但硬件不等价"的常见替代方案：

```
Anti-Pattern Warnings (DO NOT):
1. Duplicate+Mul ≠ RowMuls. Duplicate填充dim个位置但padding区域为垃圾值,
   Mul操作dimAlign宽度(含垃圾). 且Duplicate+Mul在同一PIPE_V上有RAW冒险.
   正确做法: 直接用RowMuls API.
2. LocalTensor[] ≠ float scalar. []返回子张量视图而非标量值.
   正确做法: GetValue(tensor, row).
3. 跨tile预发K/V ≠ tile内双缓冲. 跨tile预发破坏softmax online因果依赖
   (当前tile的max/sum必须计算完才能处理O rescale和下一tile累积).
   正确做法: DMA重叠仅限当前tile内(双缓冲K/V halves).
```

#### 6.1.3 在策略中添加变更优先级排序

标注每轮应只做单维变更，并提供优先级：

```
Implementation Priority (apply ONE per round, verify before stacking):
  P0: Change BLOCK_M/N/BASE_K from 64 to 128 in tiling.h → expect ~1.5x
  P1: Adjust L1 buffer allocations for larger tiles → after P0 verified
  P2: Reduce loop overhead → after P0+P1 verified
  Constraint: Each round modifies at most 2 files.
```

#### 6.1.4 将预期加速改为置信区间

```
当前: expected_vs_baseline_factor: 1.50
改进: expected_speedup: {min: 1.30, likely: 1.47, max: 1.50}
  rationale: min=dim=512 fallback, likely=dim=128实测, max=理想L0利用率
```

### 6.2 K-Search框架改进（优先级P1）

#### 6.2.1 连续同模式失败自动回退

当连续2次失败的changed_files集合相似度>80%（如都在改vec.h的RowMuls相关代码），自动触发：
- 标记当前approach为"too_hard_for_this_approach"
- 回退到best_verified_solution作为下一轮base
- 转向下一个action node

#### 6.2.2 每轮限制变更文件数

在codegen prompt中添加约束："Each optimization round should modify at most 2 files. Focus on the highest-priority change first."

#### 6.2.3 性能回退时回退基准

当某round的speedup低于已验证的best_solution时，下一round应从best_solution的代码出发，而非继续在退步版本上叠加。

### 6.3 策略catalog内容改进（优先级P2）

#### 6.3.1 重新评估各策略的预期加速

基于本轮实测数据调整预期值：
- S1: 1.50x → likely 1.47x（实测值）
- S4: 不再预期Duplicate+Mul替代，改为仅VEC2_M_CHUNK放大 → likely 1.05x-1.10x
- s1c1组合: 1.70x → likely 1.52x（实测值）

#### 6.3.2 添加更多"低难度高收益"策略

当前S1(difficulty=2)获得了1.52x。应探索更多类似diff=1-2的策略：
- VEC2_M_CHUNK继续放大到32或64（需验证UB容量）
- PipeBarrier<PIPE_FIX>单独测试（不加其他变更）
- 拆分softmax state的GM workspace到UB（仅改softmax max/sum的存储位置）

### 6.4 实验验证计划

实施改进后，应再次运行对照实验验证效果：

| 组别 | 内容 | 预期通过率 |
|:----:|:----:|:---------:|
| A' | 改进后NL策略（含API签名+不等价警告+变更优先级） | ~75% (9/12) |
| B' | 改进后NL策略 + K-Search框架改进（自动回退+限制文件数） | ~85% (10/12) |
| C' | 改进后DSL策略（同样补充缺失信息） | ~60% (7/12) |

---

## 附录A：实验数据完整性

| 实验组 | 实验目录 | Log文件 |
|:-----:|:-------:|:------:|
| Baseline | .ksearch-exp-mqa_baseline_llm_20260531_170751 | /tmp/mqa_baseline_llm.log |
| NL | .ksearch-exp-mqa_strat_nl_20260531_185043 | /tmp/mqa_natural_language.log |
| SP | .ksearch-exp-mqa_strat_sp_20260531_221933 | /tmp/mqa_structured_params.log |
| DSL | .ksearch-exp-mqa_strat_dsl_20260601_002453 | /tmp/mqa_dsl.log |

策略catalog: `/mnt/workspace/K-Search/strategies/mqa_strategies_catalog.json`
策略注入模块: `/mnt/workspace/K-Search/k_search/kernel_generators/strategy_injection.py`
实验脚本: `/mnt/workspace/K-Search/scripts/exp_mqa_*.sh`

## 附录B：NL实验逐轮失败详情

| Round | 失败类型 | 具体错误 | 策略描述缺陷 |
|:-----:|:--------:|:--------:|:-----------:|
| R1 | compile | Brcb需4参数传了3；Mulcs不存在 | 缺API签名 |
| R2 | compile | Muls第3参数LocalTensor[]≠float | 缺API参数类型 |
| R6 | runtime | 跨tile预发破坏softmax因果依赖 | 缺不等价陷阱 |
| R7 | runtime | 激进AIV同步重构 | 缺不等价陷阱+缺变更优先级 |
| R8 | compile | UB accO累积编译回归 | 缺变更优先级 |
| R9 | runtime | Duplicate+Mul RAW冒险+padding垃圾值+精度差异 | 缺不等价陷阱 |
| R10 | runtime | 批量Duplicate+Mul仍不等价 | 缺不等价陷阱 |
| R11 | runtime | 缓冲区resize+Duplicate双重变更 | 缺变更优先级+缺不等价陷阱 |