# FA-06: 稀疏KV合并与Vec0数据通道 (Sparse KV Merge & Vec0 Data Channel)

## Overview

FA类算子在BMM1阶段需要将KV数据从GM搬运到L1参与矩阵乘计算。标准算子（GQA/Sink）的KV在GM上连续存储，AIC通过MTE2直接搬运。Pioneer引入稀疏注意力机制，KV块在GM上不连续，需要AIV在Vec0阶段先将TopK选中的稀疏KV块合并到连续的Workspace区域（kvMergeGm），再由AIC从Workspace搬运到L1。这一设计将稀疏KV的gather操作从Cube侧卸载到Vector侧，并通过Vec0与BMM1的流水重叠隐藏合并开销。

## When to Use

- KV数据在GM上不连续（稀疏注意力、TopK选择、块级稀疏）
- 需要在BMM1之前将稀疏KV块gather为连续数据
- Vector核有空闲时间片可用于KV合并（如四级流水扩展为五级）

## Trade-off

- 新增Vec0阶段引入额外的CrossCore同步开销（syncV0C1）
- kvMergeGm workspace占用大量GM（约 4×512×576×2B×coreNum ≈ 2.8MB/核）
- 合并路径增加一次GM写+GM读（UB→GM→L1），相比直读多一次GM访问
- 流水启动阶段气泡增加（Vec0需要先完成才能启动BMM1）

**Source operators**: sparse_flash_attention_gqa, fused_infer_attention_sink, sparse_flash_attention_pioneer, sparse_flash_attention_enhance

---

## Variant A: 无合并直读（GQA/Sink）

Source: sparse_flash_attention_gqa, fused_infer_attention_sink

GQA和Sink算子的KV数据在GM上连续存储，AIC通过MTE2直接将KV从GM搬运到L1，无需Vector侧参与。

**KV数据流**：
```
GM (连续KV) ──MTE2──→ L1 (kvL1Tensor) ──MTE1──→ L0B ──Cube──→ BMM1结果
```

**Cube侧KV搬运**（`cube_nonquant_gqa.h`）：
```cpp
// AIC直接从GM搬运连续KV到L1
for (kL1 = 0; kL1 < kL1Loops; kL1++) {
    kvL1BufIter++;
    uint32_t kb = kvL1BufIter % 3;
    WaitFlag<MTE1_MTE2>(mte21KVIds[kb]);

    // 直接从GM连续地址搬运
    DataCopy(l1KVTensor[kb * L1_BLOCK_OFFSET],
             keyGm[s2Cur * headDimAlign],    // GM上连续偏移
             dataCopyParams);

    SetFlag<MTE1_MTE2>(mte21KVIds[kb]);
}
```

**流水结构（四级）**：
```
时间 →    T0          T1          T2          T3
AIC:   BMM1[0]     BMM1[1]     BMM1[2]+BMM2[0]  BMM1[3]+BMM2[1]
AIV:                            Vec1[0]          Vec1[1]+Vec2[0]

无Vec0阶段，BMM1直接开始，无需等待KV合并
```

**Tiling影响**：
- S2循环次数 = CeilDiv(kvSeqLen, s2BaseSize)，与实际KV长度成正比
- 每轮搬运的KV块大小固定 = s2BaseSize × headDimAlign

Benefit: 数据路径最短（GM→L1→L0），无额外GM写入；无Vec0同步开销；流水启动快
Trade-off: 不支持稀疏注意力；全量KV参与计算，大序列长度时计算量大

---

## Variant B: Vec0稀疏KV合并（Pioneer）

Source: sparse_flash_attention_pioneer

Pioneer通过TopK机制选择最相关的KV块，AIV在Vec0阶段将选中的稀疏KV块从GM gather到UB，compact后写入Workspace的kvMergeGm区域，AIC再从kvMergeGm搬运到L1。

**端到端数据流**：
```
GM (稀疏KV)                    Workspace (kvMergeGm)
  ├─ block[idx0] ──MTE2──→ UB ──┐
  ├─ block[idx1] ──MTE2──→ UB ──┼──compact──→ kvMergeGm ──MTE2──→ L1 ──MTE1──→ L0B
  ├─ block[idx2] ──MTE2──→ UB ──┘                                          │
  └─ ...                                                              Cube BMM1
                                                                          │
                                 kvValidSizeGm ◄── 记录有效搬运长度 ──────┘
```

**Vec0合并函数**（`service_vector_mla.h`）：
```cpp
void MergeKv(const RunInfo &info) {
    // 1. 读取TopK索引，确定要合并的稀疏块
    // 2. 逐块从GM搬运到UB
    for (uint32_t i = 0; i < info.validBlockNum; i++) {
        uint32_t sparseIdx = topkIndices[i];
        // 从GM稀疏地址搬运到UB
        DataCopy(ubKvMergeTensor[i * blockSize],
                 kvGm[sparseIdx * blockStride],
                 dataCopyParams);
    }
    // 3. compact后写入Workspace kvMergeGm
    DataCopy(kvMergeGm[info.mergeOffset],
             ubKvMergeTensor,
             compactParams);
    // 4. 记录有效搬运长度到kvValidSizeGm
    // AIC根据此值控制MTE2搬运长度，避免搬运无效数据
    kvValidSizeGm[info.sizeOffset] = info.validBlockNum * blockSize;
}
```

**流水集成（五级）**（`kernel_mla.h`）：
```cpp
// Vec0与BMM1的同步
if ASCEND_IS_AIV {
    CrossCoreWaitFlag(3);          // 等待AIC释放workspace（BMM2完成）
    vectorService.MergeKv(info);   // Vec0: 稀疏KV合并
    CrossCoreSetFlag(syncV0C1);    // 通知AIC: KV合并完成
}
if ASCEND_IS_AIC {
    CrossCoreWaitFlag(syncV0C1);   // 等待Vec0完成
    ComputeMm1(info);              // BMM1: 从kvMergeGm读取合并后KV
}
```

**流水时序**：
```
时间 →    T0          T1              T2                  T3
AIV:   Vec0[0]     Vec0[1]         Vec0[2]+Vec1[0]     Vec0[3]+Vec1[1]+Vec2[0]
AIC:               BMM1[0]         BMM1[1]+BMM2[0]     BMM1[2]+BMM2[1]
                   ↑                ↑
                   等待syncV0C1     等待syncV0C1
```

**Cube侧消费合并后KV**（`service_cube_mla.h`）：
```cpp
// AIC从kvMergeGm读取合并后的连续KV
uint32_t validSize = kvValidSizeGm[info.sizeOffset];  // 读取有效长度
DataCopy(l1KVTensor[kb * L1_BLOCK_OFFSET],
         kvMergeGm[info.mergeOffset],    // 从Workspace连续地址搬运
         {.blockLen = validSize});        // 按有效长度搬运，跳过无效数据
```

**Workspace区域**（参见 FA-04 Variant B）：
```
kvMergeGm:     4块 × 512(S2) × 576(D+Rope) × 2B × coreNum ≈ 2.8MB/核
kvValidSizeGm: 4 × 128份 × 4B × 2(aiv核数) × coreNum
```

**与Variant A的关键区别**：
1. KV数据路径多一跳：GM→UB→GM(workspace)→L1，而非GM→L1
2. 新增Vec0 stage和syncV0C1同步点
3. 需要双向同步：Vec0→AIC(syncV0C1) + AIC→Vec0(flag 3释放workspace)
4. S2循环次数由有效稀疏块数决定，而非全量kvSeqLen
5. 需要kvValidSizeGm控制AIC搬运长度，避免搬运无效数据

Benefit: 稀疏场景下BMM1计算量从O(kvSeqLen)降为O(topK×blockSize)；Vec0与上一轮BMM2并行执行隐藏合并开销；kvValidSize精确控制搬运量
Trade-off: 额外GM写+读增加带宽消耗；kvMergeGm workspace占用大；syncV0C1增加流水启动延迟

---

## Variant C: 训练版Vec0稀疏KV合并（SFAE）

Source: sparse_flash_attention_enhance

SFAE（训练版稀疏FA）复用Pioneer（Variant B）的Vec0 MergeKv架构，采用相同的五级流水和CrossCore同步拓扑。主要区别在于训练场景需要额外保存Softmax中间状态（Max/Sum），以及支持FlashDecode时S2轴自适应切分。

**与Pioneer的共同点**：
```cpp
// sparse_flash_attention_enhance_kernel_mla.h:84-96
static constexpr uint32_t PRELOAD_NUM = 2;
static constexpr uint32_t SFA_PRELOAD_TASK_CACHE_SIZE = 3;
static constexpr uint32_t SYNC_V0_C1_FLAG = 6;  // Vec0→Cube1（同Pioneer）
```

**训练场景扩展**（`sparse_flash_attention_enhance_kernel_mla.h:136-165`）：
```cpp
// 训练版额外输出：Softmax Max/Sum（供反向传播使用）
GlobalTensor<T> softmaxMaxGm;   // 每行Softmax最大值
GlobalTensor<T> softmaxSumGm;   // 每行Softmax求和值

// FlashDecode支持
GlobalTensor<T> accumOutGm;     // FD累积输出
GlobalTensor<T> lseSumFdGm;     // FD LogSumExp Sum
GlobalTensor<T> lseMaxFdGm;     // FD LogSumExp Max
```

**FlashDecode自适应S2切分**（`sparse_flash_attention_enhance_tiling.cpp:276-310`）：
```cpp
void SFAMlaTiling::CalcInnerSize(uint32_t s2Size) {
    sInnerSize_ = 512;  // 默认
    if (splitKVFlag_ && sfaInfo_->qLayout != SFALayout::TND) {
        if (s2Size == 256) sInnerSize_ = 128;
        else if (s2Size > 256 && s2Size <= sInnerSize_)
            sInnerSize_ = (sInnerSize_ + 1) / 2;  // 减半以保证FD并行度
    }
}
```

**与Pioneer（Variant B）的关键区别**：
1. 额外输出softmaxMax/softmaxSum供反向传播使用
2. 支持FlashDecode（splitKVFlag），S2轴可自适应减半
3. Workspace增加FD区域（accumOut + logSumExp）

Benefit: 复用Pioneer成熟的Vec0架构；训练场景保存中间状态；FlashDecode提高大S2并行度
Trade-off: 额外的softmaxMax/Sum输出增加GM写入；FD workspace增加内存占用

---

## 稀疏KV合并设计对比总结

| 特性 | GQA/Sink (无合并直读) | Pioneer (Vec0稀疏合并) | SFAE (训练版Vec0合并) |
|------|----------------------|----------------------|----------------------|
| KV数据布局 | GM上连续 | GM上稀疏（TopK选择） | GM上稀疏（TopK选择） |
| 数据路径 | GM→L1→L0B（1跳） | GM→UB→GM(ws)→L1→L0B（2跳） | GM→UB→GM(ws)→L1→L0B（2跳） |
| Vector参与 | 无 | Vec0阶段执行合并 | Vec0阶段执行合并 |
| 流水级数 | 四级（BMM1→Vec1→BMM2→Vec2） | 五级（Vec0→BMM1→Vec1→BMM2→Vec2） | 五级（同Pioneer） |
| 额外同步 | 无 | syncV0C1 + flag 3反向通知 | syncV0C1 + flag 3反向通知 |
| Workspace额外占用 | 无 | kvMergeGm + kvValidSizeGm（≈3MB/核） | kvMergeGm + kvValidSizeGm + FD区域 |
| S2循环次数 | CeilDiv(kvSeqLen, s2Base) | CeilDiv(topK×blockSize, s2Base) | CeilDiv(topK×blockSize, s2Base) |
| GM带宽开销 | 1×读 | 1×读(稀疏) + 1×写(ws) + 1×读(ws) | 同Pioneer + softmaxMax/Sum写 |
| 训练支持 | 无 | 无（推理） | 是（保存Softmax状态） |
| FlashDecode | 无 | 无 | 支持（S2自适应减半） |
| 适用场景 | 标准全量注意力 | 稀疏推理注意力 | 稀疏训练注意力 |
