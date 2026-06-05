# FA-A5-03: BALANCE分核与FlashDecode细粒度KV分块 (BALANCE Core Splitting & FlashDecode Fine-Grained KV Partitioning)

## Overview

A5平台FA算子的分核策略在A3成本感知分核基础上引入BALANCE模式：通过startBlkArray/endBlkArray数组为每核指定精确的token block范围，支持Causal对称切分和变长序列的细粒度负载均衡。FlashDecode场景下，S2维度的细粒度分块（SplitS2）允许更多核参与KV序列的并行处理。A5最多支持66核（vs A3的50核），进一步提升并行度。

## When to Use

- A5平台FA算子需要多核负载均衡（变长序列、Causal mask、稀疏模式）
- 推理场景需要FlashDecode（S2跨核切分）以提高大KV序列的并行度
- 需要利用A5的66核（vs A3的50核）提升吞吐

## Trade-off

- BALANCE模式需要Host侧计算每核的block范围，增加tiling开销
- startBlkArray/endBlkArray数组大小有限（48-66），限制最大核数
- FlashDecode的SplitS2引入额外的workspace和归约开销
- Causal对称切分在非Causal场景下无收益

**Source operators**: flash_attention_score(arch35), fused_infer_attention_score(arch35), prompt_flash_attention(arch35), incre_flash_attention(arch35)

---

## Variant A: BALANCE模式 + startBlkArray/endBlkArray（FIAS/PFA arch35）

Source: fused_infer_attention_score(arch35), prompt_flash_attention(arch35)

FIAS/PFA在A5上使用BALANCE模式，通过startBlkArray[50]/endBlkArray[50]为每核指定token block的起止范围。Host侧根据actualSeqLen计算每核的工作量，确保负载均衡。

**BALANCE模式tiling**（基于PFA baseline card）：
```cpp
// SplitCoreMode枚举
BALANCE_VECTOR = 4   // 负载均衡，Vec核主导
BALANCE_CUBE   = 5   // 负载均衡，Cube核主导

// 每核的block范围
startBlkArray[50]  // 核i的起始block索引
endBlkArray[50]    // 核i的结束block索引

// Host侧计算
splitFactorSize = CeilDivision(totalSize, actualUsedCoreNum);
splitFactorTailSize = CalcTailSize(totalSize, splitFactorSize);
```

**MultiCoreParamsRegbase结构**（`flash_attention_score_tiling_regbase.h:292-332`）：
```cpp
class MultiCoreParamsRegbase {
public:
    int32_t coreNum;
    int64_t totalSize;
    int64_t s1OuterSize;
    int64_t splitFactorSize;        // 每核分配大小
    int64_t splitFactorTailSize;    // 尾核大小
    uint32_t bnStartIdx[48];        // 批次-头起始索引
    int64_t sparseStartIdx[48];     // 稀疏起始索引
    uint8_t splitCoreMode;          // 分割模式
};
```

**与A3的关键区别**：
1. A3使用固定的SQ_SINGLE_CORE_FIRST/SQ_MULTI_CORE_FIRST模式
2. A5新增BALANCE_VECTOR/BALANCE_CUBE模式，支持更灵活的负载均衡
3. A5的数组大小支持最多66核（coreSidxEndRegbase[66]），A3最多50核

Benefit: BALANCE模式精确控制每核工作量；支持变长序列的细粒度均衡
Trade-off: Host侧计算开销增加；数组大小限制最大核数

---

## Variant B: Causal对称切分（FAS arch35）

Source: flash_attention_score(arch35)

FAS在A5上支持Causal对称切分：利用Causal mask的三角形特性，将S1方向的工作量按对称方式分配给各核，使得每核的有效计算量接近均等。

**对称切分原理**：
```
Causal mask下的有效计算量分布：
  S1行0:  有效S2 = 1
  S1行1:  有效S2 = 2
  ...
  S1行N:  有效S2 = N+1

对称配对：
  核0: 行0(cost=1) + 行N(cost=N+1)     → 总cost = N+2
  核1: 行1(cost=2) + 行N-1(cost=N)     → 总cost = N+2
  ...
  每核cost相等！
```

**FAS分核模式**（基于FAS baseline card）：
```cpp
// 分核模式选择
SQ_SINGLE_CORE_FIRST  // 静态顺序分核
SQ_MULTI_CORE_FIRST   // 动态分核（arch35优化）

// 三种分核策略
// 1. 顺序分核：按S1顺序分配
// 2. 对称分核：Causal负载均衡
// 3. 正倒序循环分核：TND L2复用
```

**正倒序循环分核**（`flash_attention_score_kernel_train.h:160-200`）：
```cpp
// TND布局下的正倒序循环分核
if (layout == LayOutTypeEnum::LAYOUT_TND) {
    realCoreInnerIdx = CalcRealCoreIdxVarlen(curCalcLoops, curCalcLoopsRemain,
                                              varlenCycleCoreNums);
} else {
    // 标准分核：部分计算(正/倒序) + 完整计算区域
    realCoreInnerIdx = CalcRealCoreIdx(...);
}
```

Benefit: Causal对称切分实现理论最优负载均衡；正倒序循环提升L2 cache复用
Trade-off: 仅适用于Causal mask场景；非Causal场景退化为顺序分核

---

## Variant C: FlashDecode S2细粒度分块（FIAS/IFA arch35）

Source: fused_infer_attention_score(arch35), incre_flash_attention(arch35)

FlashDecode将S2(KV序列)维度切分到多核并行处理，每核计算partial O/m/l，最终单核归约合并。A5的66核支持更细粒度的S2分块。

**FlashDecode触发与S2分割**（`fused_infer_attention_score_tiling_impl.cpp:681-700`）：
```cpp
bool CheckFlashDecode(const FiaTilingInfo &fiaInfo) {
    if (fiaInfo.s1Size == 1 &&
        fiaInfo.fullQuantMode == FiaFullQuantMode::PER_BLOCK_FULL_QUANT) {
        return false;  // S1=1且PER_BLOCK量化不支持
    }
    // ... 其他检查
}

// 启用FlashDecode
flashDecodeFlag_ = true;
SplitS2(fiaInfo);  // S2维度分割
```

**IFA FlashDecode参数**（基于IFA baseline card）：
```cpp
IncreFlashAttentionSplitKVParams:
  s2SplitNum        // S2分割数
  s2SplitStart      // 每段起始位置
  sInnerLoopSize    // 每段内循环大小
  accumOutSize      // partial O大小
  logSumExpSize     // partial LSE大小

// A5最多66核参与FlashDecode
coreSidxEndRegbase[66]  // vs A3的coreSidxEnd[50]
```

**FD归约流程**：
```
各核独立计算:
  核i → partial_O[i], partial_max[i], partial_sum[i]

单核归约:
  global_max = max(partial_max[0..N])
  rescale_factor[i] = exp(partial_max[i] - global_max)
  final_O = Σ(partial_O[i] × rescale_factor[i]) / global_sum
```

**与A3的关键区别**：
1. A5支持66核FlashDecode（vs A3的50核），更细粒度的S2分块
2. A5的FIAS统一调度器集成了FlashDecode（CheckFlashDecode→SplitS2）
3. A5的IFA使用coreSidxEndRegbase[66]替代A3的coreSidxEnd[50]

Benefit: 更多核参与S2并行，降低大KV序列的延迟；统一调度器简化FlashDecode触发
Trade-off: 归约开销随核数增加；workspace需要为每核预留partial O/LSE空间

---

## Variant D: 三维展开分核（FIAS arch35通用）

Source: fused_infer_attention_score(arch35)

FIAS的通用分核策略：将(B, N2, gS1)三维展开为线性索引，round-robin分配到各核。

**三维展开**（基于FIAS baseline card）：
```cpp
// 外切参数
FusedInferAttentionOuterSplitParams:
  bN2End[core_i]    // 核i的B×N2结束索引
  gS1End[core_i]    // 核i的gS1结束索引
  s2End[core_i]     // 核i的S2结束索引

// 线性化: (B, N1, S1_block) → linear_idx
// round-robin: linear_idx % coreNum → 分配到对应核

FIA_MAX_AIC_CORE_NUM = 26  // 最大AIC核数
```

**内切参数**：
```cpp
mBaseSize = qSBlockTile = 128      // Q方向基本块
s2BaseSize = kSBlockTile = 512     // KV方向基本块（MAX_KV_STACK_LEN）
```

Benefit: 三维展开覆盖所有FA推理场景；round-robin简单高效
Trade-off: round-robin不考虑计算量差异，变长序列下可能不均衡

---

## 分核策略对比总结

| 特性 | BALANCE模式(Variant A) | Causal对称(Variant B) | FlashDecode(Variant C) | 三维展开(Variant D) |
|------|----------------------|---------------------|----------------------|-------------------|
| 适用算子 | FIAS/PFA | FAS | FIAS/IFA | FIAS |
| 切分维度 | B×N2×S1 | S1 | S2 | B×N2×gS1 |
| 均衡策略 | startBlk/endBlk数组 | 对称配对 | S2细粒度分块 | round-robin |
| 最大核数 | 66(A5) / 50(A3) | coreNum | 66(A5) / 50(A3) | 26(AIC) |
| Host侧开销 | 中（计算block范围） | 低 | 中（SplitS2） | 低 |
| 归约需求 | 无 | 无 | 需要FD归约 | 无 |
| 与A3差异 | 新增BALANCE模式 | 新增对称切分 | 66核vs50核 | 相同 |
