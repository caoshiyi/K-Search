# FA-A5-04: Workspace与CrossCore共享UB数据流 (Workspace & CrossCore Shared UB Data Flow)

## Overview

A5平台FA算子的CV数据交换引入CrossCore共享UB模式（CROSS_CORE_SYNC_BOTH），AIC和AIV通过同一UB地址直接交换BMM中间结果，减少对GM Workspace的依赖。对齐要求从A3的512B降低到32B。FlashDecode场景仍需GM Workspace存储partial O/LSE。整体数据流从A3的"全部经GM"演进为"小数据走共享UB，大数据走Workspace"的混合模式。

## When to Use

- A5平台FA算子的AIC/AIV数据交换设计
- BMM1/BMM2结果需要从Cube核传递到Vector核
- FlashDecode场景需要跨核存储partial结果
- 需要评估共享UB vs GM Workspace的选择

## Trade-off

- 共享UB减少GM带宽消耗，但UB空间被AIC/AIV共享，可用容量减半
- 32B对齐（vs A3的512B）降低了padding开销
- 共享UB需要精确的CrossCore同步flag，错误会导致数据竞争
- FlashDecode的GM Workspace仍是必需的（partial结果跨核）

**Source operators**: flash_attention_score(arch35), fused_infer_attention_score(arch35), flash_attention_score_grad(arch35)

---

## Variant A: CrossCore共享UB双缓冲（FAS arch35）

Source: flash_attention_score(arch35)

FAS在A5上的BMM1/BMM2结果通过CrossCore共享UB传递：AIC通过Fixpipe将L0C结果写入UB，AIV通过LoadAlign从同一UB地址读取。双缓冲（CROSS_CORE_SYNC_BOTH + DB）实现写入与读取的流水重叠。

**共享UB buffer策略**（`common/arch35/flash_attention_score_block_cube.h:61-90`）：
```cpp
// BMM1结果：CrossCore共享双缓冲
struct Bmm1ResBuffSel {
    using Type = BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    // AIC写(Fixpipe L0C→UB) + AIV读(LoadAlign UB→Reg)
};

// BMM2结果：条件选择
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type = std::conditional_t<(useDn && isFp8),
        BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};
```

**数据流**：
```
AIC:  L0C ──Fixpipe──→ UB[共享区域, loop%2]
                         ↓ CrossCoreSetFlag(SYNC_C1_V1)
AIV:                   UB[共享区域, (loop-1)%2] ──LoadAlign──→ Register → Softmax
                         ↓ CrossCoreSetFlag(SYNC_V1_C2)
AIC:                   等待SYNC_V1_C2 → 开始BMM2
```

**对齐要求**：
```
A5: 32B对齐（BLOCK_BYTE = 32）
A3: 512B对齐（GM_ALIGN = 512）

→ A5的padding开销显著降低
→ 例: D=65时, A3需padding到512B=256个fp16, A5只需padding到32B=16个fp16
```

**与A3的关键区别**：
1. A3的BMM结果通过GM Workspace传递（Fixpipe→GM→CopyIn→UB）
2. A5直接通过共享UB传递（Fixpipe→UB，AIV直接读）
3. A5减少了一次GM读写，节省带宽

Benefit: 消除GM中转，降低带宽消耗；32B对齐减少padding
Trade-off: 共享UB占用AIC/AIV的UB空间；需要精确的CrossCore同步

---

## Variant B: FASG CrossCore双缓冲mm结果（反向算子）

Source: flash_attention_score_grad(arch35)

FASG在A5上的mm1ResBuf和mm2ResBuf采用CrossCore共享双缓冲，5个MatMul的结果通过共享UB传递给6个Vector阶段。

**共享UB分配**（`flash_attention_score_grad_kernel_base.h`）：
```cpp
// mm1ResBuf[2]: dp结果(dy×V^T)，AIC写AIV读
// mm2ResBuf[2]: QK^T结果，AIC写AIV读
// 双缓冲: taskId & 1 选择Ping/Pong

// UB预算
PRE_BUFFER  = 112KB   // mm1ResBuf[2] + mm2ResBuf[2]
CAST_BUFFER = 60KB    // V3/V4的Cast中间结果
OUTPUT      = 30KB    // dq/dk/dv输出
RESERVE     = 8KB
// 总计: 210KB / 248KB
```

**同步flag映射**：
```cpp
// Cube → Vector (AIC → AIV)
SYNC_C1_TO_V2_FLAG[taskId & 1]  // mm1(dp)就绪 → V1/V2可读
SYNC_C2_TO_V2_FLAG[taskId & 1]  // mm2(QK^T)就绪 → V2可读
SYNC_C3_TO_V5_FLAG              // mm3(dq)就绪 → V5可读
SYNC_C4_TO_V6_FLAG              // mm4(dk)就绪 → V6可读
SYNC_C5_TO_V4_FLAG              // mm5(dv)就绪 → V4可读

// Vector → Cube (AIV → AIC)
SYNC_V3_TO_C3_FLAG              // ds写L1完成 → C3可读
SYNC_V4_TO_C5_FLAG              // P写L1完成 → C5可读
```

**与A3的关键区别**：
1. A3的FASG mm结果通过GM Workspace传递
2. A5通过共享UB直接传递，减少5次GM读写
3. A5的248KB UB为FASG的多buffer提供了更多空间

Benefit: 5个MatMul结果全部走共享UB，大幅减少GM带宽消耗
Trade-off: 210KB/248KB的UB利用率很高，扩展空间有限

---

## Variant C: FlashDecode专用Workspace（FIAS/IFA arch35）

Source: fused_infer_attention_score(arch35), incre_flash_attention(arch35)

FlashDecode场景下，各核的partial O/LSE必须通过GM Workspace存储，因为归约阶段需要跨核读取所有partial结果。

**FIAS Workspace布局**（基于FIAS baseline card）：
```cpp
// Workspace区域
gS:       mm1ResSize × 4B          // BMM1中间结果(Score矩阵)
gP:       smOnlineOutSize × 2B     // Softmax输出(P矩阵)
gOTmp:    mm2ResSize × 4B          // BMM2中间结果(部分O)
gOUpdate: rescale中间结果           // Online update
gLseFD:   accumOutSize             // FlashDecode partial LSE
gOFD:     logSumExpSize            // FlashDecode partial O
```

**IFA FlashDecode Workspace**（基于IFA baseline card）：
```cpp
// partial结果存储
accumOut:   G × D × 4B × splitKVNum     // 各核的partial O
logSumExp:  G × 4B × splitKVNum         // 各核的partial LSE
prefix:     prefixAttenOutOffset         // 共享前缀

// 归约同步
coreSidxEnd barrier  // AIV核间: 所有partial核完成后触发combine
```

**共享UB vs Workspace选择**：
```
共享UB适用:
  - 同一核的AIC→AIV数据传递（BMM结果）
  - 数据量 ≤ UB可用空间
  - 不需要跨核访问

GM Workspace适用:
  - 跨核数据交换（FlashDecode partial结果）
  - 数据量超过UB容量
  - 需要持久化存储（归约阶段读取）
```

Benefit: FlashDecode Workspace支持任意核数的partial结果存储
Trade-off: GM带宽成为瓶颈；workspace大小随核数线性增长

---

## Workspace与数据流对比总结

| 特性 | 共享UB(Variant A) | FASG共享UB(Variant B) | FD Workspace(Variant C) |
|------|-------------------|---------------------|------------------------|
| 数据通道 | UB(AIC↔AIV) | UB(AIC↔AIV) | GM(跨核) |
| 对齐要求 | 32B | 32B | 32B |
| 缓冲策略 | 双缓冲(DB) | 双缓冲(taskId&1) | 每核独立区域 |
| 同步机制 | CrossCore flag | CrossCore flag | 核间barrier |
| 带宽消耗 | 低（UB内部） | 低（UB内部） | 高（GM读写） |
| 适用场景 | 同核AIC→AIV | 同核AIC→AIV(5个MM) | 跨核FlashDecode |
| 与A3差异 | 新增共享UB模式 | 新增共享UB模式 | 66核vs50核 |
| 对齐改进 | 32B vs A3的512B | 32B vs A3的512B | 32B vs A3的512B |
