# FA-A5-06: Regbase VF中mask计算与稀疏KV处理 (Regbase VF Mask Computation & Sparse KV Processing)

## Overview

A5平台FA算子的Softmax和mask处理通过VF函数实现，将A3的多步高阶API调用（Muls→Select→SoftmaxFlash→Cast）融合为单个或少数VF函数。VF内部通过MicroAPI的LoadAlign/StoreAlign和RegTensor完成mask应用、Softmax计算、Cast和ND→NZ格式转换。mask边界的Scalar计算前移到VF外部，更大的s2BaseSize（最大1024）支持更大的block_size处理。

## When to Use

- A5平台FA算子的Softmax+mask集成VF设计
- 需要在VF内部融合scale→PSE→attenMask→Softmax→Cast→ND2NZ多步操作
- 稀疏模式下需要处理不同的mask压缩模式（Causal/Band/Prefix等）
- 需要评估VF内部的mask计算与Scalar前移策略

## Trade-off

- VF融合减少UB中间结果写回，但增加单个VF的寄存器压力
- mask边界Scalar前移消除Vector停顿，但增加VF参数数量
- 更大的s2BaseSize（512/1024）提升计算效率，但增加UB buffer需求
- 多种mask压缩模式需要编译期分发，增加代码变体数量

**Source operators**: flash_attention_score(arch35), fused_infer_attention_score(arch35), common/arch35/vf/, common/arch35/attenmask.h

---

## Variant A: Softmax+mask融合VF（FAS/FIAS arch35）

Source: flash_attention_score(arch35), fused_infer_attention_score(arch35)

FAS/FIAS在A5上将Softmax的完整流程（scale→PSE→attenMask→ReduceMax→SubExp→ReduceSum→MulRcp→Cast→ND2NZ）融合到VF函数中。通过编译期模板参数控制PSE模式、mask模式、dropout等特性。

**融合VF函数签名**（`common/arch35/vf/vf_basic_block_aligned128_update.h:28-45`）：
```cpp
template <typename T, typename T2, typename pseShiftType,
    uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128,
    bool hasAtten = 0, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasDrop = 0, bool isMlaSgd = false, bool isMlaFullQuant = false,
    bool hasSink = false>
__simd_vf__ void ProcessVec1UpdateImpl128VF(
    __ubuf__ T2 * expUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb, ...,
    float divValue, const uint32_t blockStride, const float dScale,
    const uint16_t m, const T scale, const float dScaleQK, const T minValue, ...)
{
    // 寄存器声明
    RegTensor<float> vreg_min, vreg_sel, vreg_input_x, vreg_max, vreg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    // 主循环：逐行处理
    for (uint16_t i = 0; i < m; ++i) {
        // 1. LoadAlign BMM1结果
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);

        // 2. Scale
        Muls(vreg_input_x, vreg_input_x, dScale, preg_all);

        // 3. PSE (条件编译)
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            LoadAlign(vreg_pse, pseUb + i * pseStride);
            Add(vreg_input_x, vreg_input_x, vreg_pse, preg_all);
        }

        // 4. AttenMask (条件编译)
        if constexpr (hasAtten) {
            Select(vreg_input_x, preg_compare, vreg_input_x, vreg_min, preg_all);
        }

        // 5. Online Softmax (ReduceMax → Sub → Exp → ReduceSum)
        // 6. Cast + ND2NZ (输出)
        StoreAlign(expUb + i * s2BaseSize, vreg_exp, preg_all);
    }
}
```

**与A3的关键区别**：
1. A3使用多个独立API调用（Muls→Add→Select→Exp→ReduceMax→...），中间结果写回UB
2. A5在VF内部完成全部操作，中间结果驻留寄存器，不写回UB
3. A5的PSE/mask/dropout通过编译期模板参数控制，A3通过运行时分支

Benefit: 寄存器驻留消除UB中间写回；编译期分发消除运行时分支开销
Trade-off: 单个VF的寄存器压力大（5-8个RegTensor）；模板参数组合爆炸

---

## Variant B: N范围编译期分发（通用VF调度器）

Source: common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h

A5的VF函数按s2BaseSize（N范围）编译期分发到不同的优化实现。每个N范围有独立的VF实现，针对该范围的对齐特性和循环次数进行优化。

**N范围枚举与分发**（`vf_mul_sel_softmaxflashv2_cast_nz.h:44-80`）：
```cpp
enum OriginNRange {
    GT_128_AND_LTE_256 = 0,   // 128 < N ≤ 256
    GT_64_AND_LTE_128,        // 64 < N ≤ 128
    EQ_128,                   // N == 128（特殊优化）
    GT_0_AND_LTE_64,          // 0 < N ≤ 64（尾块）
    GT_256_AND_LTE_512,       // 256 < N ≤ 512
    GT_512_AND_LTE_1024,      // 512 < N ≤ 1024
    GT_0_AND_LTE_256,         // 0 < N ≤ 256（通用）
};

// 编译期分发
template <..., OriginNRange oriNRange = GT_64_AND_LTE_128, ...>
__aicore__ inline void ProcessVec1NoUpdate(...) {
    if constexpr (oriNRange == EQ_128) {
        ProcessVec1NoUpdateImpl128<...>(...);       // 128对齐优化
    } else if constexpr (oriNRange == GT_256_AND_LTE_512) {
        ProcessVec1NoUpdateGeneralImpl512<...>(...); // 512范围
    } else if constexpr (oriNRange == GT_512_AND_LTE_1024) {
        ProcessVec1NoUpdateGeneralImpl1024<...>(...); // 1024范围
    }
    // ...
}
```

**VF文件组织**（`common/arch35/vf/`目录，25个文件）：
```
按对齐变体:
  vf_basic_block_aligned128_update.h      // 128对齐+状态更新
  vf_basic_block_unaligned64_*.h          // 64非对齐
  vf_basic_block_unaligned128_*.h         // 128非对齐
  vf_basic_block_unaligned256_*.h         // 256非对齐
  vf_basic_block_unaligned512_*.h         // 512非对齐
  vf_basic_block_unaligned1024_*.h        // 1024非对齐

按功能:
  vf_mul_sel_softmaxflashv2_cast_nz.h     // Softmax+Cast+NZ
  vf_mul_sel_softmaxflashv2_cast_nz_dn.h  // DN模式
  vf_flash_decode.h                        // FlashDecode专用
  vf_antiquant_w4.h / vf_antiquant_w8.h   // 反量化
  vf_post_quant.h                          // 后量化
```

Benefit: 每个N范围的VF针对性优化（循环展开、寄存器分配）；编译期消除分支
Trade-off: 25+个VF文件的维护成本高；新增N范围需要新增VF变体

---

## Variant C: AttenMask压缩模式处理（通用）

Source: common/arch35/attenmask.h

A5的注意力mask支持7种压缩模式，在VF外部计算mask边界（Scalar前移），VF内部通过MaskReg应用mask。GQA场景下需要循环拷贝mask以适配合轴后的s1Size。

**mask压缩模式**（`attenmask.h:25-33`）：
```cpp
enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,           // 无压缩，完整mask矩阵
    LEFT_UP_CAUSAL_MODE = 1,        // 左上三角Causal
    RIGHT_DOWN_CAUSAL_MODE = 2,     // 右下三角Causal
    BAND_MODE = 3,                  // 带状mask
    PREFIX_MODE = 4,                // 前缀mask
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,// 右下Causal+带状
    BAND_LEFT_UP_CAUSAL_MODE = 6    // 带状+左上Causal
};
```

**mask计算模式**（`attenmask.h:35-43`）：
```cpp
enum class AttenMaskComputeMode {
    NORMAL_MODE = 0,                // 标准模式
    CAUSAL_OR_NEXT_ONLY_MODE,       // Causal或仅next
    PRE_ONLY_MODE,                  // 仅pre
    PRE_AND_NEXT_MODE,              // pre+next
    NO_NEED_COMPUTE_MODE,           // 无需计算
    PREFIX_COMPUTE_MODE,            // 前缀计算
    PREFIX_N_COMPUTE_MODE           // 前缀N计算
};
```

**GQA mask循环拷贝**（`attenmask.h:75-82`）：
```cpp
// GQA合轴后 s1Size = gSize × s1(1)，但mask实际只有 s1(1) 行
if (constInfo.isGqa) {
    // 需要将 s1(1) 行的mask循环拷贝 gSize 次
    for (uint32_t g = 0; g < gSize; g++) {
        DataCopy(maskUb + g * s1PerGroup * s2BaseSize,
                 maskGm, {s1PerGroup * s2BaseSize});
    }
}
```

**Scalar前移示例**（mask边界计算在VF外部完成）：
```cpp
// VF外部：计算mask边界
int64_t maskStartCol = max(0, s2StartIdx - preTokens);
int64_t maskEndCol = min(s2Size, s2StartIdx + nextTokens + s1Idx + 1);
uint32_t validCols = maskEndCol - maskStartCol;

// 传入VF作为参数
SIMD_VF(ProcessVec1UpdateImpl128VF, ..., validCols, maskStartCol, ...);
```

Benefit: 7种压缩模式覆盖所有FA稀疏场景；Scalar前移消除VF内部停顿
Trade-off: 压缩模式组合增加代码复杂度；GQA循环拷贝增加UB带宽消耗

---

## mask与VF设计对比总结

| 特性 | 融合VF(Variant A) | N范围分发(Variant B) | mask压缩(Variant C) |
|------|-------------------|---------------------|---------------------|
| 核心机制 | Scale+PSE+Mask+Softmax融合 | 按s2BaseSize编译期分发 | 7种压缩模式+Scalar前移 |
| 中间结果 | 寄存器驻留（不写UB） | 寄存器驻留 | mask边界在VF外计算 |
| 模板参数 | hasAtten/pseMode/hasDrop等 | OriginNRange | AttenMaskCompressMode |
| 代码变体 | 按特性组合 | 按N范围（7种） | 按压缩模式（7种） |
| 寄存器压力 | 高（5-8个RegTensor） | 中（按N范围优化） | 低（mask用MaskReg） |
| 与A3差异 | VF融合 vs 多API调用 | 编译期分发 vs 运行时 | 相同压缩模式，不同实现 |
