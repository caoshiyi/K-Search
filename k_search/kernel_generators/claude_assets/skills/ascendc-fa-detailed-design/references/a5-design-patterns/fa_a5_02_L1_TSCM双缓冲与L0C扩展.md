# FA-A5-02: L1/TSCM双缓冲与L0C扩展 (L1/TSCM Double-Buffering & L0C Extension)

## Overview

A5平台的L1管理从A3的显式InitBuffer模式切换到TSCM(Tensor Shared Compute Memory)自管理模式。L1容量仍为512KB，但通过GlobalTscmArray/PFAGlobalTscmArray统一管理6组L1 buffer，支持SCM双缓冲。L0C从A3的128KB翻倍到256KB，使得BMM1和BMM2的结果可以采用4-buffer策略（vs A3的双缓冲），减少L0C→UB的搬运等待。

## When to Use

- A5平台FA算子的L1 buffer分配和TSCM管理
- Cube核需要从GM预加载Q/K/V到L1，并通过L0A/L0B送入Cube计算单元
- L0C结果需要多缓冲策略以支持BMM1/BMM2的流水重叠
- 需要评估L1 512KB预算下的buffer分配方案

## Trade-off

- TSCM自管理简化了buffer生命周期，但调试困难（无法直接观察L1状态）
- L0C 256KB支持4-buffer，但仅在s1×s2×4≤L0C/4且s1×dV×4≤L0C/4时生效
- SCM双缓冲（bmm2Scm[2]×64KB=128KB）占用L1预算的25%
- Bank结构变化（8×2×16KB）影响L1的并发访问模式

**Source operators**: flash_attention_score(arch35), prompt_flash_attention(arch35), incre_flash_attention(arch35), flash_attention_score_grad(arch35)

---

## Variant A: PFA TSCM 6组管理 + SCM双缓冲

Source: prompt_flash_attention(arch35)

PFA在A5上通过PFAGlobalTscmArray管理6组L1 buffer（localScm[0..5]），加上2个SCM双缓冲（bmm2Scm[2]）用于BMM2的Softmax P矩阵传递。

**TSCM初始化**（`prompt_flash_attention_entry_regbase.h:59-75`）：
```cpp
AscendC::Impl::Detail::PFAGlobalTscmArray tscmArray;
AscendC::Impl::Detail::tscmGlobalPFA = &tscmArray;

// SCM双缓冲：BMM2的A tensor（Softmax P矩阵）
TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2];
tPipe.InitBuffer(bmm2Scm[0], 1, 65536);   // 64KB Ping
tPipe.InitBuffer(bmm2Scm[1], 1, 65536);   // 64KB Pong

// 6组L1 buffer：Q/K/V的预加载和流水
tPipe.InitBuffer(tscmGlobalPFA->localScm[0], 1, L1BUFSIZE);
tPipe.InitBuffer(tscmGlobalPFA->localScm[1], 1, L1BUFSIZE);
// ... localScm[2] ~ localScm[5]
```

**L1预算分配**：
```
SCM双缓冲:    2 × 64KB = 128KB    (bmm2Scm Ping/Pong)
6组localScm:  6 × L1BUFSIZE       (Q/K/V预加载)
总计:         128KB + 6×L1BUFSIZE ≤ 512KB
→ L1BUFSIZE ≈ 64KB时: 128 + 384 = 512KB（满载）
```

**与A3的关键区别**：
1. A3使用显式L1 TBuf（pipe->InitBuffer(l1Buf, size)），A5使用TSCM全局数组
2. A3的L1 buffer通过Event ID同步，A5的TSCM通过SCM队列自动管理
3. A5的SCM双缓冲（64KB×2）是A5特有的，A3无此概念

Benefit: TSCM自管理减少手动Event同步；SCM双缓冲与Cube流水天然匹配
Trade-off: TSCM总量约192KB（6×32KB），大D场景下可能不足；SCM占用128KB L1预算

---

## Variant B: IFA TSCM + L1自管理（L1_SELF_CONTROL）

Source: incre_flash_attention(arch35)

IFA在A5上采用L1自管理模式（L1_SELF_CONTROL），通过GlobalTscmArray管理6组QKV队列，并使用HardEvent M_MTE1(0~7)进行L1预置同步。

**TSCM初始化**（`incre_flash_attention_entry_regbase.h:120-131`）：
```cpp
AscendC::Impl::Detail::GlobalTscmArray tscmArray;
AscendC::Impl::Detail::tscmGlobal = &tscmArray;

// Vec1 SCM Ping/Pong
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPing;
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPong;
tPipe.InitBuffer(vec1ScmPing, 1, vec1ResultSize);
tPipe.InitBuffer(vec1ScmPong, 1, vec1ResultSize);

// 6组QKV队列
tPipe.InitBuffer(tscmGlobal->localQue[0], 1, qkvSize);
// ... localQue[1] ~ localQue[5]
```

**L1自管理参数**（基于IFA baseline card）：
```
MM1 A(Q):  TPosition::TSCM, depthA1=1
MM1 B(K):  TPosition::TSCM, depthB1=2(自管理2块), dbL0A=2, dbL0B=2
PRE_LOAD_NUM=2, TSCM_DOUBLE_BUFFER=2
MM2 A(P):  TPosition::TSCM, 来自Vec1输出SCM queue
MM2 B(V):  TPosition::GM, ND格式

SHARED_CO1_BUFFER_SIZE_KB = 64KB  // L1自管理总量上限
```

**与PFA的区别**：
1. IFA的Q tile很小（G×D，典型16×128×2=4KB），L1压力小
2. IFA使用HardEvent M_MTE1(0~7)预置L1搬入，PFA使用localScm队列
3. IFA的MM2 B(V)直接从GM读取（ND格式），不经过L1

Benefit: L1自管理适合IFA的小Q场景；HardEvent预置减少搬运延迟
Trade-off: SHARED_CO1_BUFFER_SIZE_KB=64KB限制了L1自管理总量

---

## Variant C: L0C 256KB 4-buffer策略（FAS/FASG arch35）

Source: flash_attention_score(arch35), flash_attention_score_grad(arch35)

A5的L0C从128KB翻倍到256KB，使得BMM1和BMM2的结果可以同时驻留L0C。当tile尺寸满足条件时，采用4-buffer策略（BMM1 Ping/Pong + BMM2 Ping/Pong），消除L0C→UB的搬运等待。

**L0C buffer策略选择**（`common/arch35/flash_attention_score_block_cube.h:100-115`）：
```cpp
// L0C 4-buffer条件：两个BMM的结果都能放入L0C/4
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<
        (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
         s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
        BuffersPolicy4buff<BufferType::L0C>,    // 4-buffer: 256KB / 4 = 64KB per buffer
        BuffersPolicyDB<BufferType::L0C>>;      // 双缓冲: 256KB / 2 = 128KB per buffer
};

// A5: L0C_SIZE = 256KB
// 4-buffer条件: s1×s2×4 ≤ 64KB 且 s1×dV×4 ≤ 64KB
// → s1=128, s2=128: 128×128×4 = 64KB ✓
// → s1=128, s2=256: 128×256×4 = 128KB ✗ → 降级为双缓冲
```

**FASG L0C利用**（基于FASG baseline card）：
```
L0C 256KB(arch35):
  DKV Resident条件: max(mm1,mm2,mm3) + mm4 + mm5 ≤ L0C_MAX_SIZE
  → 允许更多MatMul结果同时驻留L0C
  → 减少L0C→UB的搬运次数

A3 L0C 128KB:
  → 同样条件下更容易超限
  → 更频繁的L0C→UB搬运
```

**与A3的关键区别**：
1. A3 L0C 128KB，4-buffer条件为s1×s2×4≤32KB（s1=128,s2仅支持64）
2. A5 L0C 256KB，4-buffer条件为s1×s2×4≤64KB（s1=128,s2支持128）
3. A5的FASG可以让更多MatMul结果驻留L0C，减少搬运开销

Benefit: L0C翻倍支持更大tile的4-buffer策略；FASG的DKV Resident更容易满足
Trade-off: 4-buffer仅在小tile时生效；大D场景仍需降级为双缓冲

---

## L1/L0C设计对比总结

| 特性 | PFA TSCM(Variant A) | IFA L1自管理(Variant B) | L0C 4-buffer(Variant C) |
|------|---------------------|------------------------|------------------------|
| L1管理模式 | PFAGlobalTscmArray | GlobalTscmArray + HardEvent | N/A (L0C) |
| L1 buffer组数 | 6组localScm + 2×SCM | 6组localQue + 2×SCM | N/A |
| SCM双缓冲 | 2×64KB = 128KB | 2×vec1ResultSize | N/A |
| L1总预算 | ~512KB（满载） | ~64KB（自管理上限） | N/A |
| L0C容量 | 256KB | 256KB | 256KB |
| L0C策略 | 双缓冲 | 双缓冲 | 4-buffer（条件满足时） |
| 4-buffer条件 | N/A | N/A | s1×s2×4≤64KB 且 s1×dV×4≤64KB |
| 与A3差异 | TSCM替代显式L1 TBuf | L1自管理+TSCM | L0C 256KB vs 128KB |
