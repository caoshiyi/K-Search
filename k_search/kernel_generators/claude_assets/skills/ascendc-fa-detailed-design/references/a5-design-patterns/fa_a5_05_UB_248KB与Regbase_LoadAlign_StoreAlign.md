# FA-A5-05: UB 248KB与Regbase LoadAlign/StoreAlign (UB 248KB & Regbase LoadAlign/StoreAlign)

## Overview

A5平台UB容量从A3的192KB(184KB可用)扩展到248KB，为FA算子的Vector侧buffer提供更大的空间预算。但Regbase架构下，所有Vector计算必须通过LoadAlign/StoreAlign在UB与寄存器之间搬运数据，引入了显式的对齐约束（32B）和Bank冲突风险（8 group × 2 bank × 16KB）。UB buffer设计需要同时考虑容量规划和Bank冲突规避。

## When to Use

- A5平台FA算子的UB buffer分配和容量规划
- Vector阶段需要通过LoadAlign/StoreAlign访问UB数据
- 需要评估Bank冲突风险（同Bank group的并发读写）
- preLoadNum或Softmax状态buffer需要按UB 248KB重新规划

## Trade-off

- UB 248KB比A3多64KB，可支持更大的tile或更多buffer层数
- LoadAlign/StoreAlign要求32B对齐（vs A3的无显式对齐要求），非对齐数据需要padding
- Bank结构变化（8×2×16KB vs A3的16×3×4KB），冲突模式不同
- 256个VecReg × 256B = 64KB寄存器空间，部分中间结果可驻留寄存器而非UB

**Source operators**: flash_attention_score(arch35), fused_infer_attention_score(arch35), flash_attention_score_grad(arch35), prompt_flash_attention(arch35)

---

## Variant A: 训练前向UB规划 + CrossCore共享（FAS arch35）

Source: flash_attention_score(arch35)

FAS在A5上的UB buffer采用CrossCore共享模式（CROSS_CORE_SYNC_BOTH），AIC写入BMM结果到UB，AIV通过LoadAlign读取。Softmax状态buffer（max/sum/exp）按preLoadNum缩放，利用248KB的额外空间支持更大的s1BaseSize。

**Buffer策略选择**（`common/arch35/flash_attention_score_block_cube.h:61-125`）：
```cpp
// BMM1结果buffer：CrossCore共享，AIC写AIV读
template <bool useDn, bool isFp8>
struct Bmm1ResBuffSel {
    using Type = BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    // 双缓冲 + 跨核同步
};

// BMM2结果buffer：条件选择单缓冲或双缓冲
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type = std::conditional_t<(useDn && isFp8),
        BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};
```

**UB容量规划**（基于248KB）：
```
BMM1结果(CrossCore DB):  2 × s1BaseSize × s2BaseSize × 4B     ≈ 2 × 128 × 128 × 4 = 128KB
BMM2结果(CrossCore DB):  2 × s1BaseSize × dBaseSize × 4B      ≈ 2 × 128 × 128 × 4 = 128KB
Softmax状态(×preLoadNum): 3 × s1BaseSize × preLoadNum × 4B    ≈ 3 × 128 × 2 × 4 = 3KB
临时buffer:              tmpBuf × 2                            ≈ 16KB
总计:                    约 200-240KB（248KB预算内）
```

**与A3的关键区别**：
1. A3 UB 184KB可用，大tile场景下BMM结果buffer紧张；A5 248KB缓解此问题
2. A5的CrossCore共享通过CROSS_CORE_SYNC_BOTH实现，AIC/AIV通过同一UB地址交换数据
3. A5的32B对齐要求使得buffer起始地址必须是32的倍数

Benefit: 248KB支持更大的s1BaseSize×s2BaseSize组合；CrossCore共享减少GM搬运
Trade-off: CrossCore共享需要精确的同步flag管理；大buffer占用后剩余空间有限

---

## Variant B: 推理UB规划 + TSCM协同（PFA/IFA arch35）

Source: prompt_flash_attention(arch35), incre_flash_attention(arch35)

PFA/IFA在A5上的UB buffer与TSCM协同工作。BMM1结果通过TSCM(SCM)传递给AIV，AIV在UB中完成Softmax后将P矩阵写回TSCM供BMM2使用。UB主要用于Softmax计算和最终输出。

**IFA UB buffer分配**（基于IFA baseline card）：
```
mmResUbSize:     G × S_inner × 4B    ≈ 16 × 256 × 4 = 16KB   (BMM1输出)
bmm2ResUbSize:   G × D × 4B          ≈ 16 × 128 × 4 = 8KB    (BMM2输出)
softmax状态:     G × 4B × 3          ≈ 16 × 4 × 3 = 192B     (max/sum/exp)
mask/pse临时:    按需分配             ≈ 8-16KB
总计:            约 40-60KB（248KB预算充裕）
```

**PFA UB buffer分配**（基于PFA baseline card）：
```
mmResUbSize:     S_inner × S_outer × 4B   ≈ 512 × 128 × 4 = 256KB  ⚠️ 超预算
→ 实际: 需要分块处理，每次处理部分S_inner
bmm2ResUbSize:   S_outer × D × 4B         ≈ 128 × 128 × 4 = 64KB
softmax状态:     S_outer × 4B × 4         ≈ 128 × 4 × 4 = 2KB
总计:            需要精确tiling控制
```

**与A3的关键区别**：
1. IFA场景（Sq=1）UB压力小，248KB绰绰有余
2. PFA场景（Sq>1）BMM1结果可能超UB，需要分块或走TSCM
3. A5的TSCM分担了部分数据中转，减轻UB压力

Benefit: IFA场景UB充裕，可增大S_inner提升计算效率；TSCM协同减少UB占用
Trade-off: PFA大序列场景仍需精确tiling；TSCM与UB的数据流需要额外同步

---

## Variant C: 反向算子UB规划 + 双缓冲mm结果（FASG arch35）

Source: flash_attention_score_grad(arch35)

FASG在A5上需要在UB中同时维护mm1ResBuf和mm2ResBuf的双缓冲（各2份），加上Softmax状态和梯度中间结果。248KB的UB为反向算子提供了关键的额外空间。

**UB预算常量**（`flash_attention_score_grad_kernel_base.h:101-108`）：
```cpp
// UB预算分配
PRE_BUFFER  = 112KB   // mm1ResBuf[2] + mm2ResBuf[2]（CrossCore双缓冲）
CAST_BUFFER = 60KB    // Cast/ND2NZ中间结果
OUTPUT      = 30KB    // dq/dk/dv输出缓冲
RESERVE     = 8KB     // 预留
// 总计: 210KB（248KB预算内，剩余38KB用于Softmax状态等）
```

**mm结果双缓冲**（`flash_attention_score_grad_block_cube.h`）：
```cpp
// mm1ResBuf[2]: dp结果，AIC写AIV读，CrossCore同步
// mm2ResBuf[2]: QK^T结果，AIC写AIV读，CrossCore同步
// 双缓冲实现Ping-Pong：taskId & 1选择当前buffer

// 条件输出路由
IS_DQ_WRITE_UB  // dQ写UB（BN2模板，非多核累加）
IS_DK_WRITE_UB  // dK写UB（BN2/BN2S2模板）
IS_DV_WRITE_UB  // dV写UB（BN2S2模板）
```

**与A3的关键区别**：
1. A3的184KB UB下，PRE_BUFFER+CAST+OUTPUT=202KB已接近极限
2. A5的248KB提供38KB额外空间，可支持更大的CAST_BUFFER或更多Softmax状态
3. A5的L0C 256KB（vs A3 128KB）允许更多MatMul结果驻留L0C，减轻UB压力

Benefit: 248KB缓解反向算子的UB紧张问题；L0C翻倍进一步减轻UB负担
Trade-off: 5个MatMul的结果buffer仍需精确规划；确定性模式下额外的dSTransL1Buf增加压力

---

## Variant D: Bank冲突规避与LoadAlign/StoreAlign对齐（A5通用）

Source: common/arch35/vf/, common/arch35/util_regbase.h

A5的UB Bank结构为8 group × 2 bank × 16KB，与A3的16 group × 3 bank × 4KB完全不同。LoadAlign/StoreAlign的32B对齐要求和Bank冲突模式需要在buffer布局时统一考虑。

**Bank结构对比**：
```
A3 (910B):  16 group × 3 bank × 4KB = 192KB
            → 每4KB一个bank，3路bank交错
            → 冲突粒度: 4KB

A5 (950):   8 group × 2 bank × 16KB = 256KB (248KB可用)
            → 每16KB一个bank，2路bank交错
            → 冲突粒度: 16KB
```

**LoadAlign/StoreAlign约束**：
```cpp
// 所有UB地址必须32B对齐
MicroAPI::LoadAlign(vreg, ubPtr);       // ubPtr必须32B对齐
MicroAPI::StoreAlign(ubPtr, vreg, mask); // ubPtr必须32B对齐

// 非对齐数据处理
// 方案1: padding到32B对齐
uint32_t alignedSize = (rawSize + 31) / 32 * 32;
// 方案2: 使用DataCopyPad搬运非对齐数据到对齐位置
```

**Bank冲突规避策略**：
```
策略1: 相邻buffer分配到不同bank group
  → buffer_A起始地址: 0x0000 (group 0)
  → buffer_B起始地址: 0x4000 (group 1, 偏移16KB)

策略2: 同一VF内的src和dst使用不同bank
  → LoadAlign(src)和StoreAlign(dst)的地址间隔 ≥ 16KB

策略3: 利用寄存器驻留减少UB访问
  → 中间结果保留在RegTensor中，不写回UB
```

Benefit: 16KB粒度的bank结构减少了小buffer场景的冲突概率；寄存器驻留策略有效
Trade-off: 大buffer（>16KB）更容易跨bank group，需要仔细布局；2路bank比A3的3路更容易冲突

---

## UB Buffer设计对比总结

| 特性 | FAS训练(Variant A) | PFA/IFA推理(Variant B) | FASG反向(Variant C) | Bank规避(Variant D) |
|------|-------------------|----------------------|--------------------|--------------------|
| UB总预算 | 248KB | 248KB | 248KB | 248KB |
| 主要buffer | BMM1/2结果(CrossCore DB) | mmRes + softmax | mm1/2ResBuf[2] + Cast + Output | N/A |
| CrossCore共享 | CROSS_CORE_SYNC_BOTH | 通过TSCM | CROSS_CORE_SYNC | N/A |
| Softmax状态 | 3个 × preLoadNum | 3-4个 × S_outer | Softmax max/sum(来自FAS) | N/A |
| UB利用率 | ~200-240KB (80-97%) | IFA: ~60KB (24%) / PFA: 需分块 | ~210KB (85%) | N/A |
| Bank冲突风险 | 中（大buffer跨group） | 低（buffer小） | 中高（多个大buffer） | 需主动规避 |
| 与A3差异 | +64KB缓解大tile压力 | TSCM分担数据中转 | +64KB支持更大Cast buffer | Bank粒度16KB vs 4KB |
| 对齐要求 | 32B（LoadAlign/StoreAlign） | 32B | 32B | 32B |
