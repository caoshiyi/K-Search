# MQA AscendC 策略优化实验：自然语言策略的问题分析与优化建议

> 实验日期：2026-05-31 ~ 2026-06-01
> Baseline: 0.809398ms on Ascend910B3
> 控制变量：model=glm-5.1, max_rounds=12, stagnation=5, timeout=900s

---

## 一、实验概况

自然语言策略注入实验在12轮优化中，4轮通过、3轮编译失败、5轮运行时失败（精度校验不通过），通过率 **33.3%**。最终获得 1.520x 加速（latency 0.5326ms），优于基线实验（0x）和其他策略形式（SP 0x，DSL 1.464x），但通过率仍然偏低。

### 逐轮状态一览

| Round | Action Node | Status | Latency | Speedup | 失败类别 |
|------:|:-----------:|:------:|:-------:|:-------:|:--------:|
| 1 | s1 (Enlarged Tile) | compile_failed | — | — | API签名错误 |
| 2 | s1 | compile_failed | — | — | 类型推导冲突 |
| 3 | s1 | passed | 0.5486ms | 1.475x | — |
| 4 | s1 | passed | 0.5858ms | 1.382x | 性能回退 |
| 5 | s1 | passed | 0.5595ms | 1.447x | 性能回退 |
| 6 | s1 | failed | — | — | 精度校验失败 |
| 7 | s1 | failed | — | — | 精度校验失败 |
| 8 | s1 | compile_failed | — | — | 编译回归 |
| 9 | s1c1 (VEC2 Vec+Chunk) | failed | — | — | 精度校验失败 |
| 10 | s1c1 | failed | — | — | 精度校验失败 |
| 11 | s1c1 | failed | — | — | 精度校验失败 |
| 12 | s1c1 | passed | 0.5326ms | 1.520x | — |

---

## 二、问题1：策略描述缺乏精确API约束，导致LLM误用AscendC API

### 问题描述

自然语言策略描述优化意图，但不提供AscendC API的精确签名、参数类型和调用约束。LLM在实现时必须"猜"API用法，猜测基于通用编程经验而非AscendC专用知识，导致编译失败。

### 证据

**Round 1 编译失败**：策略S4描述"硬件RowMuls API"，LLM将其解读为`Brcb`+`Mulcs`组合：

```cpp
// LLM生成的错误代码 (vec.h:264)
Brcb(brcbMat[row * cols], scale[row], cols);   // Brcb需要4参数，只传了3
Mulcs(dst, src, brcbMat, cols);                 // Mulcs在CANN 9.0中不存在
```

编译错误：
```
error: 'Brcb' requires 4 arguments, 3 were provided
error: 'Mulcs' undeclared (did you mean 'Muls'?)
```

**Round 2 编译失败**：LLM修正为`Muls`后，类型推导冲突：

```cpp
// LLM尝试修复 (vec.h)
Muls(dst[row*cols], src[row*cols], scale[row], cols);
// scale[row] 返回 LocalTensor<float> 子视图，不是 float 标量
```

编译错误：
```
error: deduced conflicting types for parameter 'T' ('float' vs. 'AscendC::LocalTensor<float>')
```

LLM不知道`LocalTensor[]`下标运算符返回的是子张量视图而非标量值，需要用`GetValue(scale, row)`提取float。

### 根因

策略S4的自然语言描述为：

> "Replace scalar RowMulsImpl/RowDivsImpl (which use GetValue per-row loops) with hardware RowMuls/RowDivs API calls. The hardware RowMuls API takes a scale source tensor and broadcasts it across each row..."

描述了优化意图（用硬件API替代标量循环）和API名称（RowMuls/RowDivs），但**没有提供精确的API签名**：

```
// 缺失的关键信息
RowMuls(LocalTensor<T> dst, LocalTensor<T> src, LocalTensor<T> scale, int32_t dealRows, int32_t cols, int32_t actualCols)
// 第3个参数是 LocalTensor<T> 而非 float scalar
// 第4个参数是行数(dealRows)，不是列数(cols)
// 第6个参数是实际列数(actualCols)，用于padding区域的正确处理
```

LLM从"broadcasts it across each row"推断API语义，但无法确定参数数量、类型和顺序，只能基于通用编程直觉猜测，结果猜错。

---

## 三、问题2：策略描述缺少硬件不等价陷阱警告，导致语义等价性破坏

### 问题描述

LLM在数学/算法层面做等价推理，认为两种实现方式计算结果相同，因此可以安全替换。但AscendC硬件的**对齐约束、流水线依赖和精度路径**使得数学等价 ≠ 硬件等价。策略描述没有标注这些不等价陷阱，LLM无从知晓。

### 证据A：跨tile DMA预发破坏softmax因果依赖（R6, R7）

LLM推理链：

```
当前tile计算MM1时，AIC可以用MTE2预加载下一tile的K/V到L1
→ 这只是数据搬运的时序优化，不改变任何计算逻辑
→ 和双缓冲是同一原理，只是把预取范围扩大到下一个tile
→ 应该能减少GM→L1等待延迟
```

数学推理完全正确——数据预取不改变计算结果。但Flash Attention online softmax有tile间因果依赖：

```
for each kv_tile t:
    S = Q @ K[t]                     // MM1: 当前tile的S矩阵
    row_max_new = max(S, row_max_old) // 需要上一tile的row_max
    row_sum_new = sum + Σexp(S-row_max_new) // 需要新max做rescale
    O *= exp(row_max_old - row_max_new)      // 用新旧max差rescale O
    O += (S-row_max_new) * V[t]              // MM2: 需要新max
```

跨tile预发K[t+1]后，AIC可能在当前tile的`row_max_new`还没算出来时就开始计算S[t+1]，而AIV还没拿到`row_max[t]`来做O的rescale。原本`SetFlag/WaitFlag`保证的"AIC MM1完成 → AIV Vec1开始"因果顺序被打破。

**Round 6和7均因精度校验失败**：softmax的max/sum/exp状态依赖被破坏，O累积的rescale使用了过期的max值。

### 证据B：Duplicate+Mul广播 ≠ GetValue+Muls（R9, R10, R11）

LLM推理链：

```
GetValue(row) → 取1个float标量
Muls(dst, src, scalar, width) → dst[i] = src[i] * scalar, 逐行操作

数学上等价于：
Duplicate(scalar, width) → 把scalar广播到width个位置
Mul(dst, src, broadcast_buf, width) → dst[i] = src[i] * broadcast_buf[i]

因为 broadcast_buf[i] = scalar 对所有i，所以 dst[i] = src[i] * scalar
→ 完全等价，且Duplicate+Mul用向量宽度(32 fp32/cycle)而非标量(1/cycle)
→ 应该快32倍
```

数学推理完全正确。但硬件层面有三个不等价：

**不等价1：PIPE_V上的RAW流水线冒险**

```cpp
Duplicate(brcbUb_, alpha, cols);  // Duplicate写入brcbUb_
Mul(dst, src, brcbUb_, cols);     // Mul立即读取brcbUb_，同一PIPE_V上无PipeBarrier
```

Duplicate和Mul在同一个PIPE_V上执行。Duplicate写brcbUb_后，Mul紧跟着读brcbUb_，没有`PipeBarrier<PIPE_V>`分隔。这是Read-After-Write冒险：Mul可能读到Duplicate尚未完成写入的数据（过期值或部分写入值）。

**不等价2：对齐填充区域的垃圾值**

AscendC向量操作必须操作对齐宽度 `dimAlign = AlignUp(dim, 32/sizeof(float))`：

```
GetValue+Muls(dst, src, scalar, dim):
  Muls内部只处理dim个有效元素，padding区域不参与运算

Duplicate(scalar, dimAlign):
  Duplicate只填充dim个位置为scalar值
  padding区域(dimAlign - dim)保留为随机垃圾值（未清零）

Mul(dst, src, buf, dimAlign):
  Mul操作整个dimAlign宽度，包括padding区域的垃圾值
  dst的padding区域被写入 src[padding] * garbage
```

当后续代码沿aligned width读取dst时，padding区域被污染。

**不等价3：精度路径差异**

```
Muls路径: dst[i] = src[i] × scalar       // 硬件RowMuls指令，1次乘法
Duplicate+Mul路径:
  step1: buf[j] = scalar                  // Duplicate写入，1次浮点赋值
  step2: dst[i] = src[i] × buf[i]        // Mul向量乘法，1次乘法
```

单看有效区域，数学等价。但`buf[i]`经过1次Duplicate赋值可能有舍入ε，使得实际计算为 `dst[i] = src[i] × (scalar + ε)` 而非 `dst[i] = src[i] × scalar`。对softmax的exp/rescale/sum这类精度敏感路径，ε累积可超过atol=1e-2容差。

**R9-R11三次尝试均因torch.allclose精度校验失败**，直到R12回退Duplicate+Mul改用仅VEC2_M_CHUNK放大才通过。

### 根因

策略S4的描述为：

> "Replace scalar RowMulsImpl/RowDivsImpl with hardware RowMuls/RowDivs API calls..."

这句话暗示了"标量 → 硬件向量"的等价替换意图，但没有明确标注：
1. `RowMuls`是AscendC专有硬件指令，不是`Duplicate+Mul`组合的通用等价
2. `Duplicate+Mul`在PIPE_V上存在RAW冒险，必须加`PipeBarrier<PIPE_V>`
3. `Duplicate`的对齐padding区域不会被自动清零
4. softmax的exp/rescale路径对精度敏感，1次额外舍入可能导致累积误差超标

---

## 四、问题3：策略描述鼓励多维度叠加变更，但单维变更成功率远高于多维变更

### 问题描述

自然语言策略描述通常覆盖一个优化方向的多个侧面（如S1描述了tile放大+L1 buffer调整+loop开销减少），给LLM一种"可以同时改多个维度"的暗示。但实验数据表明：**最小变更成功率高，激进多维度变更成功率低**。

### 证据

| 变更类型 | Round | 维度数 | 结果 |
|:--------:|:-----:|:------:|:----:|
| 仅改BLOCK_M/N/K=128 | R3 | 1 | **1.475x** |
| 仅改VEC2_M_CHUNK=16 | R12 | 1 | **1.520x** |
| tile放大 + 异步DMA预取 | R4 | 2 | 1.382x（回退） |
| tile放大 + Q cache + PIPE_FIX | R5 | 3 | 1.447x（回退） |
| 跨tile DMA + 同步重构 + UB accO | R6 | 3 | 精度失败 |
| 跨tile DMA + 事件驱动同步 | R7 | 2 | 精度失败 |
| Duplicate+Mul广播 + VEC2_CHUNK | R9 | 2 | 精度失败 |
| Duplicate+Mul + 缓冲区resize | R11 | 3 | 精度失败 |

**单维变更通过率 = 2/2 (100%)，多维变更通过率 = 0/6 (0%)。**

### 根因

自然语言策略的描述风格倾向于全面阐述：

> S1描述: "Increase BLOCK_M, BLOCK_N, and BASE_K from 64 to 128. Larger tile sizes amortize loop overhead, improve L0A/L0B utilization from 25% to 50%..."

这种描述同时说明了tile放大、L0利用率提升、loop开销减少三个维度。LLM读到后倾向于同时实现这三个维度的变更，而非先实现最核心的tile放大、验证通过后再叠加其他维度。

---

## 五、问题4：策略描述缺少"什么不能做"的负面约束

### 问题描述

当前策略只描述"应该做什么优化"，没有描述"哪些看似合理的替代方案实际不可行"。LLM在没有负面约束的情况下，会自行探索"等价但更高效"的实现方式，而这些方式在AscendC硬件上可能不等价。

### 证据

| 策略描述 | LLM自探索的"等价"替代 | 实际结果 |
|:--------:|:--------------------:|:--------:|
| S4: "硬件RowMuls API" | Brcb(3-param)+Mulcs | 编译失败（API不存在） |
| S4: "硬件RowMuls API" | Muls+LocalTensor[]下标 | 编译失败（类型不匹配） |
| S4: "硬件RowMuls API" | Duplicate+Mul广播 | 精度失败（不等价） |
| S1: "更大的tile减少loop开销" | 跨tile DMA预取减少GM等待 | 精度失败（因果依赖破坏） |
| S1: "L0利用率提升" | UB accO累积替代GM workspace | 编译/运行失败 |

每个"等价替代"都是LLM基于通用编程知识做出的合理推断，但在AscendC特定硬件约束下均不等价。

### 根因

策略描述采用纯正面表述（"做什么、为什么有用"），缺少反面约束（"不能做什么、为什么看似等价实际不等价"）。这就像C标准只定义了行为规范而没有列举undefined behavior——LLM无法预知哪些操作在特定硬件上是非法的。

---

## 六、问题5：策略预期加速与实际差距过大，降低WM决策质量

### 问题描述

策略S1预期1.50x，实际1.475x（差距-1.7%），达成率尚可。但策略S1c1预期1.70x，实际1.520x（差距-11.8%），达成率低。预期过高会导致WM过度优先选择该策略，并在实际加速不及预期时浪费后续round尝试"补足差距"。

### 证据

| 策略节点 | 预期 | 最佳实际 | 差距 | WM对差距的反应 |
|:--------:|:----:|:--------:|:----:|:-------------:|
| s1 | 1.50x | 1.475x | -1.7% | 继续叠加变更尝试"改进"→ R4-R8全部回退或失败 |
| s1c1 | 1.70x | 1.520x | -11.8% | 4轮尝试Duplicate+Mul补足差距→ R9-R11全部失败 |

s1实际仅差预期1.7%，但WM仍消耗5轮(R4-R8)尝试"改进"，其中4轮失败。s1c1差预期11.8%，主要因为Duplicate+Mul广播部分预期贡献~0.15x未能实现。

### 根因

策略预期加速是基于Flash Attention参考实现的理想值，但Flash Attention与MQA在硬件适配细节（GQA vs MQA的KV共享模式、dim=128 vs dim=512的L1空间约束）上有差异。预期值未考虑这些差异，导致过于乐观。

---

## 七、优化建议

### 建议1：在策略描述中添加精确API签名和调用示例

**现状**：策略S4描述"硬件RowMuls API"但不提供签名。

**改进**：在每个涉及AscendC API的策略中，添加精确的API签名和正确的调用示例：

```diff
  当前策略S4 natural_language:
  "Replace scalar RowMulsImpl/RowDivsImpl with hardware RowMuls/RowDivs API calls.
   The hardware RowMuls API takes a scale source tensor and broadcasts it..."

+ 改进后策略S4 natural_language:
+ "Replace scalar RowMulsImpl/RowDivsImpl with hardware RowMuls/RowDivs API calls.
+  Correct API signatures:
+    RowMuls(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32)
+    RowDivs(dst: LocalTensor<T>, src: LocalTensor<T>, scale: LocalTensor<T>, dealRows: int32, cols: int32, actualCols: int32)
+  The scale parameter is a LocalTensor (NOT a float scalar). Each row's scale factor
+  must be pre-loaded at offset row*BRCB_NUM in the scale buffer.
+  Example call:
+    LocalTensor<float> expStateUb = expStateBuf_.Get<float>();
+    RowMuls(oPrevUb, oPrevUb, expStateUb, dealRows, dim, actualDim);
+  Do NOT use Duplicate+Mul as a substitute — see Anti-Pattern Warning below."
```

**预期效果**：消除R1/R2类型的API误用编译失败，提升通过率约16%（2/12轮）。

### 建议2：在策略描述中添加"不等价陷阱"警告（Anti-Pattern Warnings）

**现状**：策略只描述正面优化方向，无反面约束。

**改进**：在每个策略末尾添加显式的不等价陷阱警告：

```diff
  当前策略S4 natural_language (结尾):
  "...The state tensors must be properly sized: each row's scale factor
   is at offset row*BRCB_NUM in the state buffer."

+ 改进后策略S4 natural_language (追加):
+ "
+ Anti-Pattern Warnings (DO NOT do the following):
+ 1. Do NOT replace RowMuls with Duplicate(scalar, dimAlign)+Mul(dst, src, buf, dimAlign).
+    Reason: Duplicate fills only dim elements with scalar; padding area (dimAlign-dim)
+    retains garbage values. Mul processes full dimAlign width including garbage.
+    Result: precision test fails due to padding pollution + fp32 rounding difference.
+ 2. Do NOT use LocalTensor[] subscript as a float scalar argument to Muls.
+    Reason: LocalTensor[] returns a sub-tensor view (LocalTensor<T>), not a scalar T.
+    Use GetValue(tensor, row) to extract a scalar float.
+ 3. Do NOT use Brcb(dst, src, count) with 3 arguments.
+    Reason: AscendC Brcb requires 4 arguments: Brcb(dst, src, repeatTimes, dstStride).
+    The correct alternative for row-level broadcast multiply is RowMuls API."
```

类似地，对策略S1（跨tile DMA预发）添加警告：

```
Anti-Pattern Warnings:
1. Do NOT prefetch K/V data for tile t+1 before tile t's softmax state is computed.
   Reason: Online softmax has tile-level causal dependency — O accumulation for tile t+1
   requires row_max_new and row_sum_new from tile t. Prefetching breaks the
   AIC→AIV sync semantics that enforce this dependency.
   Safe scope for DMA overlap: only within the current tile (double-buffer K/V halves).
```

**预期效果**：消除R6/R7/R9/R10/R11类型的语义等价性失败，提升通过率约42%（5/12轮）。

### 建议3：策略描述应明确标注"最小变更优先"原则

**现状**：策略描述全面阐述多个维度，暗示可同时变更。

**改进**：在每个策略描述开头添加变更优先级排序：

```diff
  当前策略S1 natural_language (开头):
  "Increase BLOCK_M, BLOCK_N, and BASE_K from 64 to 128..."

+ 改进后策略S1 natural_language (开头):
+ "Implementation Priority (apply ONE change at a time, verify before stacking):
+  P0: Change BLOCK_M/N/BASE_K constants from 64 to 128 in tiling.h — this alone
+      should yield ~1.5x speedup. Verify this works before attempting other changes.
+  P1: Adjust L1 buffer allocations for larger tile sizes (after P0 is verified).
+  P2: Reduce loop iteration count and sync overhead (after P0+P1 are verified).
+  Do NOT combine P0+P1+P2 in one round. Each round should change only ONE priority level."
```

**预期效果**：减少多维叠加变更导致的失败，可能将R4-R8从5轮浪费缩减为2-3轮有序递进。

### 建议4：策略预期加速应标注置信区间而非单一乐观值

**现状**：策略S1预期1.50x，S1c1预期1.70x。

**改进**：将预期加速改为置信区间，并注明参考来源和差距风险：

```diff
  当前策略S1:
  expected_vs_baseline_factor: 1.50

+ 改进后策略S1:
+ expected_speedup: {min: 1.30, likely: 1.47, max: 1.50}
+ speedup_rationale: "min=1.30 accounts for dim=512 L1 pressure fallback;
+  likely=1.47 based on dim=128 basic case实测; max=1.50 is ideal L0 utilization."
+ reference: "Flash Attention V2 on Ascend910B achieves 1.52x with same tile config,
+  but MQA has lower KV reuse factor (1:1 vs H:1) which may reduce achievable gain."
```

**预期效果**：WM在遇到likely值而非max值时不会过度追求"补足差距"，减少后续轮的无效尝试。

### 建议5：K-Search框架层改进——连续同模式失败后自动回退

**现状**：Duplicate+Mul连续失败3次（R9-R11）才在R12手动回退。

**改进**：在WM的cycle管理逻辑中添加自动回退触发：

```
if consecutive_failures_same_pattern >= 2:
    mark current action node as "too_hard_for_this_approach"
    revert to best_verified_solution as base for next attempt
    try next action node instead of continuing on same node
```

判定"同模式失败"可以基于：连续2次的changed_files集合相似度 > 80%（都在改vec.h的RowMuls相关代码）。

**预期效果**：将R9-R11的3轮浪费缩减为2轮，节省约25分钟实验时间。

### 建议6：K-Search框架层改进——每轮限制变更文件数

**现状**：LLM在单轮中同时修改4-5个文件（cube.h + vec.h + kernel.h + tiling.h + pybind11.cpp）。

**改进**：在codegen prompt中添加约束：

```
Constraint: Each optimization round should modify at most 2 files.
If you need to change more files, split the work across multiple rounds.
Focus on the highest-priority change first.
```

**预期效果**：强制LLM做最小变更，降低多维叠加的风险。

---

## 八、改进预期效果汇总

| 问题 | 影响轮次 | 当前通过率 | 建议改进 | 预期改进后通过率 |
|:----:|:--------:|:----------:|:--------:|:---------------:|
| API签名缺失 | R1, R2 (2轮) | 4/12=33% | 建议1：添加精确API签名 | 6/12=50% |
| 不等价陷阱缺失 | R6, R7, R9-R11 (5轮) | 4/12=33% | 建议2：添加Anti-Pattern | 9/12=75% |
| 多维叠加变更 | R4-R8 (5轮浪费) | 4/12=33% | 建议3+6：最小变更优先 | 减少浪费轮次 |
| 预期加速过于乐观 | s1c1差距-11.8% | — | 建议4：置信区间 | 改善WM决策质量 |
| 连续失败不回退 | R9-R11 (3轮浪费) | 4/12=33% | 建议5：自动回退 | 减少浪费轮次 |

**综合预期**：如果同时实施建议1-6，通过率可从33%提升到**约75%**（9/12轮），浪费轮次从8轮缩减到约3轮。

---

## 九、结论

自然语言策略形式在当前实验中是最有效的策略形式（1.520x加速），但其33%的通过率说明策略描述存在五个系统性问题：

1. **缺乏精确API签名** → LLM猜错AscendC API用法
2. **缺乏不等价陷阱警告** → LLM在数学层做等价推理但硬件层不等价
3. **鼓励多维度叠加** → 单维变更100%通过而多维变更0%通过
4. **缺乏负面约束** → LLM自探索"等价替代"但实际不等价
5. **预期加速过于乐观** → WM过度追求补足差距

这些问题不是自然语言形式本身的缺陷，而是**策略描述的信息完整度不足**——缺少硬件层的关键约束信息。解决方案不是改用另一种策略形式（结构化参数完全失败，DSL中等效果），而是在自然语言描述中**补充缺失的信息维度**：精确API签名、不等价陷阱警告、变更优先级排序、预期加速置信区间。

核心洞察：**自然语言策略的优势在于LLM理解最自然，但当前描述停留在算法抽象层。补充硬件抽象层的关键约束信息后，自然语言策略有望成为最有效的策略形式。**