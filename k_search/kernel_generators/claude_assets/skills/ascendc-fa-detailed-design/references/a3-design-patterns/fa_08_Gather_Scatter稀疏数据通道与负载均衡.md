# FA-08: Gather/Scatter稀疏数据通道与两阶段负载均衡 (Sparse Gather/Scatter & Two-Phase Load Balancing)

## Overview

稀疏注意力算子需要根据TopK索引从GM中非连续地读取KV数据（Gather）或将梯度累加回GM（ScatterAdd）。与Pioneer的Vec0 MergeKv方案（参见FA-06）不同，反向算子和Indexer类算子采用AIV核直接执行Gather/Scatter操作，通过Ping-Pong双缓冲与Cube计算重叠。同时，这些算子普遍采用两阶段负载均衡（粗调+精调）来处理稀疏场景下的不均匀计算量。

## When to Use

- 反向算子需要根据稀疏索引读取KV并将梯度累加回原位置
- 不同S1行的有效S2长度差异大（稀疏模式/下三角mask）
- 需要精确的负载均衡以最小化核间最大负载差异

## Trade-off

- Gather/Scatter增加GM带宽消耗（离散访存效率低于连续访存）
- 两阶段负载均衡增加Host侧计算开销
- ScatterAdd需要原子操作或确定性同步保证正确性

**Source operators**: sparse_flash_attention_grad_enhance, sparse_lightning_indexer_grad_kl_loss_enhance, flash_attention_score_enhance, lightning_indexer_enhance

---

## Variant A: Gather/Scatter Ping-Pong（SFAGE）

Source: sparse_flash_attention_grad_enhance

SFAGE的AIV核通过Gather从GM按TopK索引读取稀疏Key/Value到UB，Cube计算完成后再通过ScatterAdd将dK/dV梯度累加回GM原位置。Gather和Scatter通过Ping-Pong双缓冲与Cube计算重叠。

**Gather操作**（`vec_op.h:46-120`）：
```cpp
// AIV: 根据topk_indices从GM读取稀疏Key到UB
for (idx = 0; idx < selectedBlockCount; idx++) {
    uint32_t sparseIdx = topkIndices[idx];
    DataCopy(gatherTensorPing[idx * blockStride],
             keyGm[sparseIdx * keyStride],
             dataCopyParams);
}
// Gather完成后写入selectedK workspace
DataCopy(selectedKWorkspace[pingOffset], gatherTensorPing, compactParams);
CrossCoreSetFlag(VEC_WAIT_CUBE_PING);  // 通知Cube可以开始
```

**Scatter操作**（`vec_op.h:130-192`）：
```cpp
// AIV: 将dK梯度按原索引累加回GM
CrossCoreWaitFlag(CUBE_WAIT_VEC_PING);  // 等待Cube完成
for (idx = 0; idx < selectedBlockCount; idx++) {
    uint32_t sparseIdx = topkIndices[idx];
    // 从workspace读取dK结果
    DataCopy(scatterAddTensorK, dkWorkspace[idx * blockStride], params);
    // 累加到GM原位置
    AtomicAdd(dkGm[sparseIdx * keyStride], scatterAddTensorK, params);
}
```

**Ping-Pong同步**：
```
时间 →   T0              T1              T2
AIV:  Gather[Ping]    Gather[Pong]    Scatter[Ping]
AIC:                  Cube12[Ping]    Cube345[Ping] + Cube12[Pong]
      ├─VEC→CUBE─┤    ├─VEC→CUBE─┤
                      ├─CUBE→VEC─┤
```

Benefit: Gather/Scatter与Cube通过Ping-Pong完全重叠；支持任意稀疏模式
Trade-off: 离散GM访存效率低；ScatterAdd需要原子操作或确定性同步

---

## Variant B: Gather→Compute→ScatterAdd三阶段（SLIGKLE）

Source: sparse_lightning_indexer_grad_kl_loss_enhance

SLIGKLE将Vector处理分为三个独立阶段：Vector0（Gather Key/KeyRope/KeyIndex）、Vector1（计算P/SY/DW/Loss）、Vector2（ScatterAdd到输出）。每个阶段通过独立的CrossCore flag与Cube核同步。

**三阶段流水**（`sparse_lightning_indexer_grad_kl_loss_enhance_vector.h:31-82`）：
```cpp
// Vector0: Gather稀疏数据
void ProcessVector0(RunInfo &runInfo) {
    // 从GM按索引读取Key、KeyRope、KeyIndex到workspace
    CrossCoreSetFlag(SYNC_V0_TO_C1_P_FLAG[pingpong]);   // 通知Cube: P数据就绪
    CrossCoreSetFlag(SYNC_V0_TO_C1_SY_FLAG[pingpong]);  // 通知Cube: SY数据就绪
}

// Vector1: 向量计算
void ProcessVector1(RunInfo &runInfo) {
    CrossCoreWaitFlag(SYNC_C1_TO_V1_P_FLAG[pingpong]);  // 等待Cube完成BMM1
    CrossCoreWaitFlag(SYNC_C1_TO_V1_SY_FLAG[pingpong]); // 等待Cube完成BMM2
    // 计算梯度、Loss
    CrossCoreSetFlag(SYNC_V1_TO_C2_DW_FLAG[pingpong]);  // 通知Cube: DW就绪
}

// Vector2: ScatterAdd
void ProcessVector2(RunInfo &runInfo) {
    CrossCoreWaitFlag(SYNC_C2_TO_V2_SA_FLAG[pingpong]);  // 等待Cube完成BMM3
    // 将梯度ScatterAdd到输出GM
}
```

**12个CrossCore同步标志**（`sparse_lightning_indexer_grad_kl_loss_enhance_common.h:28-44`）：
```cpp
// 6对同步标志（每对Ping/Pong各1个）
SYNC_V0_TO_C1_P_FLAG[2]  = {0, 1};   // Vec0→Cube1: P数据
SYNC_V0_TO_C1_SY_FLAG[2] = {2, 3};   // Vec0→Cube1: SY数据
SYNC_C1_TO_V1_P_FLAG[2]  = {4, 5};   // Cube1→Vec1: BMM1结果
SYNC_C1_TO_V1_SY_FLAG[2] = {6, 7};   // Cube1→Vec1: BMM2结果
SYNC_V1_TO_C2_DW_FLAG[2] = {8, 9};   // Vec1→Cube2: DW梯度
SYNC_C2_TO_V2_SA_FLAG[2] = {10, 11}; // Cube2→Vec2: BMM3结果
```

**与Variant A的关键区别**：
1. 三个独立Vector阶段（V0/V1/V2），而非Gather和Scatter两阶段
2. 12个CrossCore flag（6对×Ping/Pong），同步粒度更细
3. Vector1包含向量计算（梯度、Loss），不仅仅是数据搬运
4. 支持KL Loss计算，输出包含loss标量

Benefit: 三阶段分离使每个阶段职责清晰；细粒度同步支持更深的流水重叠
Trade-off: 12个同步flag增加管理复杂度；三阶段串行约束限制了单阶段的并行度

---

## Variant C: 两阶段负载均衡——粗调+精调（FAE/SLIGKLE）

Source: flash_attention_score_enhance, sparse_lightning_indexer_grad_kl_loss_enhance

FAE和SLIGKLE都采用两阶段负载均衡策略：第一阶段粗调按平均值分配，第二阶段精调通过迭代优化最小化核间最大负载差异。这种策略特别适合稀疏场景下不同S1行有效S2长度差异大的情况。

**粗调阶段**（`flash_attention_score_enhance_tiling_general.cpp:4452-4476`）：
```cpp
// 计算平均负载
int64_t avgVal = CeilDivision(sparseArraySum, validAiCoreNum);

// 从前往后遍历，每核累加S2直到接近avgVal
for (int64_t idx = 0; idx < validAiCoreNum; idx++) {
    int64_t singleLoadValue = 0;
    while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
        singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
        tmpsparseStartIdx[idx] += 1;
    }
}
```

**精调阶段**（`flash_attention_score_enhance_tiling_general.cpp:4480-4483`）：
```cpp
// 迭代优化：前向+后向边界调整
while (BalanceLoad(sparseValidArray, multiCoreParams, tmpLocalValue, tmpsparseStartIdx)) {
    // BalanceLoad内部：
    // 1. 前向扫描：如果前一核+当前S2 < maxVal，移动边界
    // 2. 后向扫描：如果后一核+前一个S2 < maxVal，移动边界
    // 3. 直到无法进一步优化
}
```

**SLIGKLE的精调实现**（`sparse_lightning_indexer_grad_kl_loss_enhance_tiling_general.cpp:698-745`）：
```cpp
bool BalanceLoad(...) {
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    // 前向：尝试将后核的首块移给前核
    for (idx = 1; idx < validAicNum; idx++) {
        if ((localValue[idx-1] + sparseValidArray[start]) < maxVal) {
            localValue[idx-1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        }
    }
    // 后向：尝试将前核的末块移给后核
    for (idx = validAicNum-1; idx > 0; idx--) {
        if ((localValue[idx] + sparseValidArray[start-1]) < tmpMaxVal) {
            localValue[idx-1] -= sparseValidArray[start-1];
            localValue[idx] += sparseValidArray[start-1];
            sparseStartIdx[idx] -= 1;
        }
    }
    return changed;
}
```

**与FA-03中成本感知分核的区别**：
1. FA-03的Sink/GQA使用Cost函数（6M+10S2），这里直接用实际S2长度作为负载
2. FA-03的Sink采用三级分配（Batch→Row→Block），这里采用粗调+精调两阶段
3. 精调阶段通过迭代优化全局最优，而非贪心分配

Benefit: 两阶段策略在稀疏场景下负载均衡效果优于单次分配；迭代精调收敛快
Trade-off: Host侧计算开销略高于简单均分；需要预计算每个S1的有效S2长度

---

## Variant D: Kernel侧块数均分（LIE）

Source: lightning_indexer_enhance

LIE在Kernel侧执行分核，按总基本块数简单均分，前几个核各多处理1块。

**分核逻辑**（`lightning_indexer_enhance_kernel.h:276-354`）：
```cpp
void SplitCore() {
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    // 前deal1MoreBlockCoreNum个核各处理 minBlockPerCore+1 块
    // 其余核各处理 minBlockPerCore 块
    // 记录每核的 (bN2Start, bN2End, gS1Start, gS1End, s2Start, s2End)
}
```

Benefit: 零Host侧开销；Kernel侧计算简单
Trade-off: 不考虑稀疏模式下不同块的计算量差异，可能产生负载不均

---

## 稀疏数据通道与负载均衡对比总结

| 特性 | SFAGE (Gather/Scatter PP) | SLIGKLE (三阶段Vec) | FAE/SLIGKLE (粗调+精调) | LIE (块数均分) |
|------|--------------------------|---------------------|------------------------|---------------|
| 数据通道 | Gather+ScatterAdd | Gather→Compute→ScatterAdd | N/A（负载均衡） | N/A（负载均衡） |
| 缓冲策略 | Ping-Pong双缓冲 | Ping-Pong双缓冲 | — | — |
| CrossCore flag数 | 4个 | 12个（6对×PP） | — | — |
| 负载均衡 | 简单不均匀分配 | 两阶段粗调+精调 | 两阶段粗调+精调 | Kernel侧块数均分 |
| 分核位置 | Host侧 | Host侧 | Host侧 | Kernel侧 |
| 稀疏感知 | 是（TopK索引） | 是（下三角mask） | 是（实际S2长度） | 否（按块数） |
| Host开销 | 低 | 中 | 中 | 零 |
| 适用场景 | 稀疏FA反向 | 稀疏Indexer反向+KL Loss | 变长序列/稀疏前向 | 负载均匀的Indexer |
