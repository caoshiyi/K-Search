# FA-02: L1多缓冲轮转与Sink常驻 (L1 Multi-Buffering & Sink Resident)

## Overview

FA类算子在L1层级采用多缓冲轮转策略，使MTE2(GM→L1)搬运与MTE1(L1→L0)加载重叠执行。不同算子根据数据复用特性选择不同的缓冲方案：GQA采用Q2V2KP3方案，Sink采用QP2+KV2方案，Pioneer创新性地将QP从4块缩减为2块，省出的2块用于Sink数据常驻L1，避免重复搬运。

## When to Use

- Cube核执行BMM1/BMM2，需要从GM搬运Q/K/V到L1
- S2方向有多次循环，K/V数据需要反复搬运
- 存在Sink token时，Sink数据在所有S2循环中都需要参与计算

## Trade-off

- 多缓冲增加L1占用，减少可用于数据的空间
- 缓冲管理逻辑复杂，需要精确的Event同步
- Sink常驻需要牺牲其他缓冲的份数

**Source operators**: fused_infer_attention_sink, sparse_flash_attention_gqa, sparse_flash_attention_pioneer

---

## Variant A: Q2V2KP3方案（GQA）

Source: sparse_flash_attention_gqa

GQA采用Q/V各2块、K/P共享3块的L1缓冲方案。K和Projection(Softmax结果)共享同一组L1缓冲，因为BMM1使用K、BMM2使用P，两者不会同时需要。

**L1缓冲分配**（`cube_nonquant_gqa.h:295-305, 643-651`）：
```cpp
// L1 Buffer分配
L1_Q:  64KB × 2份 = 128KB   // Query常驻，双缓冲
L1_V:  64KB × 2-4份          // Value，根据S2_BASICSIZE动态选择
L1_KP: 64KB × 3份 = 192KB   // Key/Projection共享，三缓冲轮转

// 三缓冲轮转逻辑
for (kL1 = 0; kL1 < kL1Loops; kL1++) {
    kvL1BufIter++;
    uint32_t kb = kvL1BufIter % 3;  // 0→1→2→0→1→2...
    WaitFlag<MTE1_MTE2>(mte21KVIds[kb]);  // 等待该块可用
    bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
    // ... 使用该块 ...
    SetFlag<MTE1_MTE2>(mte21KVIds[kb]);   // 释放该块
}
```

**Event同步**：
```cpp
// 3个Event ID对应3块KP缓冲
KP_EVENT0/+1/+2  // MTE2→MTE1 K/P数据同步
Q_EVENT0/+1      // MTE2→MTE1 Q数据同步（2块）
V_EVENT0/+1/+2/+3 // MTE2→MTE1 V数据同步
```

Benefit: K/P复用减少L1总占用；三缓冲支持加载→计算→写回完全流水
Trade-off: K/P共享需要确保BMM1和BMM2不同时访问同一块

---

## Variant B: QP2+KV2方案（Sink）

Source: fused_infer_attention_sink

Sink算子采用QP共享2块、KV共享2块的方案。Q和Softmax结果(P)共享同一组L1缓冲，K和V共享另一组。

**L1缓冲分配**（`fia_block_cube_nonquant_sink.h`）：
```cpp
// L1 Buffer分配
qpBufL1: TPosition::A1, 128KB × 2份 = 256KB  // Q/P共享，双缓冲
kvBufL1: TPosition::A1, 128KB × 2份 = 256KB  // K/V共享，双缓冲

// 双缓冲Ping-Pong
pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2);  // 256KB
pipe->InitBuffer(kvBufL1, L1_KV_SIZE * 2);  // 256KB

// 使用时通过bufId切换
(qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T))  // Ping/Pong切换
```

**Event同步**：
```cpp
QP_EVENT0/1 (EVENT_ID2/3)  // Q/P数据双缓冲
KV_EVENT0/1 (EVENT_ID4/5)  // K/V数据双缓冲
```

**复用关系**：
- BMM1阶段：qpBufL1存Q，kvBufL1存K
- BMM2阶段：qpBufL1存P(Softmax结果)，kvBufL1存V

Benefit: QP/KV各自复用，减少L1总占用；双缓冲实现搬运与计算重叠
Trade-off: 双缓冲深度有限，大S2场景下搬运延迟可能暴露

---

## Variant C: QP2+Sink2+KV3方案（Pioneer）

Source: sparse_flash_attention_pioneer

Pioneer的创新设计：将QP从4块缩减为2块，省出的2块L1空间用于Sink数据常驻。Sink Key只在第一轮S2循环搬入L1，后续所有循环直接复用，避免重复搬运。

**L1缓冲分配**（`service_cube_mla.h:307-312`）：
```cpp
// L1 Buffer分配 - Sink常驻设计
static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T)); // 72KB

pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 2);   // 72KB × 2 = 144KB  QP双缓冲
pipe->InitBuffer(bufSinkL1, L1_BLOCK_SIZE * 2);  // 72KB × 2 = 144KB  Sink常驻！
pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);    // 72KB × 3 = 216KB  KV三缓冲
```

**Sink常驻逻辑**（`service_cube_mla.h:604-619`）：
```cpp
// 只在第一轮S2循环搬运Sink数据到L1
if (info.isFirstGlobalLoop) {
    // 搬运Sink Key到L1常驻缓冲
    DataCopy(l1SinkTensor[kL1 * L1_BLOCK_OFFSET],
             keySinkGm[kL1 * (constInfo.combineHeadDim >> 1)], nd2nzPara);
}
// 后续循环直接使用l1SinkTensor，无需重新搬运
LocalTensor<KV_T> l1SinkTensorActual = l1SinkTensor[kL1 * L1_BLOCK_OFFSET];
```

**KV三缓冲轮转**：
```cpp
// 3个Event ID对应3块KV缓冲
static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

kvL1BufIter++;
uint32_t kb = kvL1BufIter % 3;  // 三块轮转
WaitFlag<MTE1_MTE2>(mte21KVIds[kb]);
bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
```

**设计决策链**：
```
原始方案: QP 4块 + KV 3块 = 7块 × 72KB = 504KB
优化方案: QP 2块 + Sink 2块 + KV 3块 = 7块 × 72KB = 504KB
                    ↑
                    Sink常驻，避免每轮S2循环重复搬运
```

Benefit: Sink数据搬运次数从 O(s2LoopTimes) 降为 O(1)；L1总占用不变
Trade-off: QP缓冲从4块降为2块，QP方向的流水深度降低

---

## 三种方案对比

| 特性 | GQA (Q2V2KP3) | Sink (QP2+KV2) | Pioneer (QP2+Sink2+KV3) |
|------|---------------|-----------------|--------------------------|
| L1总块数 | 7-9块 | 4块 | 7块 |
| 单块大小 | 64KB | 128KB | 72KB |
| Q/P策略 | Q独立2块 | QP共享2块 | QP共享2块 |
| K/V策略 | KP共享3块+V独立2-4块 | KV共享2块 | KV共享3块 |
| Sink处理 | 无 | 无独立缓冲 | 独立2块常驻 |
| 最大流水深度 | 3(KP) | 2(QP/KV) | 3(KV) |
| Event ID数 | 9+ | 4 | 5 |
