# FA-03: 成本感知分核与FlashDecode归约 (Cost-Aware Core Splitting & FD Reduction)

## Overview

FA类算子面临不规则的计算负载分布：不同batch的actualSeqLen不同，不同S1行的有效S2长度也不同（受mask/sparse影响）。简单均分会导致慢核瓶颈。Sink和GQA采用Host侧成本感知分核，通过加权Cost函数和多级分配策略实现负载均衡；Pioneer采用更简单的Kernel侧按token数均分。当S2维度被跨核切分时，需要FlashDecode归约来合并部分Softmax结果。

## When to Use

- 不同batch的序列长度差异大（actualSeqLen变化范围广）
- 存在稀疏mask导致不同行的有效计算量差异大
- S2维度需要跨核切分以提高并行度（触发FD归约）

## Trade-off

- 成本感知分核：Host侧计算开销增加，但Kernel侧负载更均衡
- 简单均分：Host侧零开销，但可能存在负载不均
- FD归约：额外的workspace和Vector核计算开销

**Source operators**: fused_infer_attention_sink, sparse_flash_attention_gqa, sparse_flash_attention_pioneer

---

## Variant A: 成本感知三级分配（Sink）

Source: fused_infer_attention_sink

Sink算子采用最精细的分核策略：基于加权Cost函数的三级分配（Batch→Row→Block），并支持Sink token的特殊Cost累加。

**Cost函数**（`split_core.cpp:86-93`）：
```cpp
// M轴系数6，S2轴系数10，反映S2方向对性能的更大影响
Cost(M, S2) = 6 × Align(M, 16) / 16 + 10 × Align(S2, 64) / 64

// 按 NormalBlock/TailBlock 组合生成4种开销表
CalcCostTable():
  costTable[0] = Cost(mBaseSize, s2BaseSize)        // Normal×Normal
  costTable[1] = Cost(mBaseSize, s2TailSize)         // Normal×Tail
  costTable[2] = Cost(mTailSize, s2BaseSize)         // Tail×Normal
  costTable[3] = Cost(mTailSize, s2TailSize)         // Tail×Tail
```

**三级分配策略**（`split_core.cpp:431-550`）：
```cpp
// 第一级：按整Batch分配
AssignByBatch():
  while (加入整个batch后仍在容忍度内):
    将整个batch分配给当前核

// 第二级：按GS1行分配
AssignByRow():
  while (加入整行后仍在容忍度内):
    将整行分配给当前核

// 第三级：按S2方向单块分配
AssignByBlock():
  逐块分配直到达到costLimit

// 兜底：强制分配
ForceAssign():
  若某核分配为0块，强制分配1块
```

**容忍度机制**：
```cpp
FA_TOLERANCE_RATIO = 2  // 允许超出costLimit最多一半的lastBlockCost
```

**Sink特殊处理**（`split_core.cpp:211-238`）：
```cpp
// sinkNumber > 0 且 sparseMode == 4 时
// S2方向的sink块需要额外累加进Cost
CalcS1GCacheWithSinkNumber():
  sinkCost = CalcCost(mSize, sinkBlocks)
  totalCost += sinkCost

// 按块分配时跳过sink和preToken之间被掩掉的块
UpdateCurS2IdxWithSinkNumber()
```

**核数决策**（`split_core.cpp:753-768`）：
```cpp
maxCore = min(aicNum, totalBlockNum)
minCore = ceil(sqrt(totalBlockNum))
// 遍历 [minCore, maxCore]，选择 maxCost 最小的方案
for (coreNum = minCore; coreNum <= maxCore; coreNum++):
  plan = CalcSplitPlan(coreNum)
  if plan.maxCost < bestMaxCost:
    bestPlan = plan
```

Benefit: 慢核开销最小化；支持Sink token的精确Cost建模
Trade-off: Host侧需要遍历核数区间，计算开销较大

---

## Variant B: 成本感知加权均分（GQA）

Source: sparse_flash_attention_gqa

GQA采用类似的成本感知策略，但分配算法更简洁：按 `avgBaseNum = totalBaseNum / coreNum` 阈值切分。

**Cost函数**（`split_core.h:256-280`）：
```cpp
// 与Sink相同的Cost模型
s1SparseBlockNum[b][s1] = 6 × alignBasicM + 10 × s2SizeCost
其中: s2SizeCost = (s2Caculatelen + 63) >> 6
```

**分配算法**：
```cpp
// 计算平均负载
avgBaseNum = totalBaseNum / coreNum

// 遍历 (bN2, gS1) 空间，累积成本达到阈值时切换核
for each (bN2, gS1):
  accumCost += blockCost
  if accumCost >= avgBaseNum:
    记录当前核的End点: bN2End[core], gS1End[core], s2End[core]
    切换到下一核
```

**与Sink的区别**：
- 无三级分配（Batch→Row→Block），直接按成本阈值切分
- 无容忍度机制
- 无Sink特殊处理

Benefit: 分配逻辑简洁，Host侧开销小
Trade-off: 负载均衡精度略低于Sink的三级分配

---

## Variant C: Kernel侧简单均分（Pioneer）

Source: sparse_flash_attention_pioneer

Pioneer采用完全不同的策略：在Kernel侧按token数简单均分，不使用成本感知。

**分配算法**（`kernel_mla.h:522-604`）：
```cpp
void InitCalcParamsEach() {
    // 计算总token数
    uint32_t totalBaseNum = 0;
    for (bIdx = 0; bIdx < batchSize; bIdx++) {
        totalBaseNum += GetBalanceActualSeqLengths(bIdx) * actBatchS2;
    }

    // 简单均分
    uint32_t avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    uint32_t targetBaseNum = (currCoreIdx + 1) * avgBaseNum;

    // 遍历 (bN2, gS1) 空间，累积到目标数时记录End点
    for (bN2Idx ...) {
        for (s1GIdx ...) {
            accumBaseNum += 1;
            if (accumBaseNum >= targetBaseNum) {
                constInfo.bN2End = bN2Idx;
                constInfo.gS1End = s1GIdx;
                return;
            }
        }
    }
}
```

**与Host侧分核的区别**：
- 在Kernel侧执行，每个核独立计算自己的范围
- 按token数均分，不考虑M/S2方向的计算量差异
- 无Cost函数，无容忍度机制
- 适用于MLA场景（kvHeadNum=1，负载相对均匀）

Benefit: 零Host侧开销；Kernel侧计算简单；适合负载均匀的MLA场景
Trade-off: 负载不均匀时（如不同batch的seqLen差异大）可能产生慢核

---

## Variant D: FlashDecode归约（Sink/GQA共有）

Source: fused_infer_attention_sink, sparse_flash_attention_gqa

当S2维度被跨核切分时，同一GS1行的部分Softmax结果分布在多个核上，需要FD归约合并。

**触发条件**：某一GS1行在S2方向被切分到多个核上

**FD记录**（`split_core.cpp:597-621`）：
```cpp
// 在核切换点检测跨核行
IsNeedRecordFDInfo():
  if (当前行的S2被切分到不同核):
    记录归约任务: {bN2Idx, gS1Idx, s2SplitNum}
    对归约任务沿M轴进一步切分: gS1BaseSizeOfFd = 8
```

**FD负载均衡**（`split_core.cpp:693-733`）：
```cpp
SplitFD():
  totalVecNum = usedCoreNum × vecCubeRatio  // 可用Vector核数
  // 每个归约任务的负载 = s2SplitNum × gS1SplitNum
  // 按总负载/可用Vector数计算阈值，依次分配
```

**FD Workspace布局**：
```cpp
// Sink: fia_tiling_nonquant_sink.cpp:565-572
fdWorkspace = (fdParamNums × headDimAlign + 2 × fdParamNums × 8) × 4B
// fdParamNums = coreNum × 2 × mBaseSize  (头归约+尾归约)

// 包含：
// accumOut: fdParamNums × headDimAlign × 4B  — 归约中间结果(fp32)
// logSumExp: 2 × fdParamNums × 8 × 4B       — Max和Sum各一份(fp32)
```

Benefit: 支持S2跨核切分，提高大S2场景的并行度
Trade-off: 额外的workspace和Vector核归约计算开销

---

## 三种分核方案对比

| 特性 | Sink (三级分配) | GQA (加权均分) | Pioneer (简单均分) |
|------|----------------|---------------|-------------------|
| 分核位置 | Host侧 | Host侧 | Kernel侧 |
| Cost函数 | 6M+10S2 | 6M+10S2 | 无（按token数） |
| 分配粒度 | Batch→Row→Block | 阈值切分 | token级 |
| 容忍度 | FA_TOLERANCE_RATIO=2 | 无 | 无 |
| Sink支持 | 特殊Cost累加 | 无 | 无 |
| 核数优化 | 遍历[minCore,maxCore] | 固定全部AIC | 固定全部AIC |
| FD归约 | 支持 | 支持 | 支持(简化版) |
| 适用场景 | 负载高度不均匀 | 中等不均匀 | 负载相对均匀(MLA) |
