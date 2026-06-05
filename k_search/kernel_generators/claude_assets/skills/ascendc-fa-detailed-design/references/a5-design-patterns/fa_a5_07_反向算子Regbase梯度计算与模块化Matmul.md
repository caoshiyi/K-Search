# FA-A5-07: 反向算子Regbase梯度计算与模块化Matmul (Backward Regbase Gradient & Modular Matmul)

## Overview

A5平台FA反向算子（FASG）采用5个MatMul + 6个Vector阶段的流水结构，通过模块化matmul设计（matmul_modules/目录）将各MatMul操作封装为独立模块。AIV侧的Softmax梯度计算通过VF函数实现（vector_api/目录，20个VF文件），支持按对齐宽度（256/512/768）和数据类型（FP16/FP32）分发。确定性模式（DETER_OLD/DETER_NEW）通过独立的workspace区域和单核归约保证bit-exact结果。

## When to Use

- A5平台FA反向算子的流水设计和梯度计算
- 需要5个MatMul阶段的编排和跨核同步
- 需要VF函数实现Softmax梯度（softmax_grad_front_cast等）
- 需要确定性计算模式保证训练可复现性

## Trade-off

- 5个MatMul + 6个Vector阶段的同步复杂度高（8+个CrossCore flag）
- 模块化matmul增加代码组织清晰度，但增加文件数量
- VF函数按对齐宽度分发（20个文件），维护成本高
- 确定性模式需要额外workspace和归约开销

**Source operators**: flash_attention_score_grad(arch35)

---

## Variant A: 5-MatMul + 6-Vector流水编排（FASG arch35）

Source: flash_attention_score_grad(arch35)

FASG在A5上采用3阶段入口（Pre→Base→Post），Base阶段包含5个MatMul和6个Vector阶段，通过双任务Ping-Pong（FagRunInfo runInfos[2]）实现当前轮Cube与上一轮Vector的重叠。

**3阶段入口**（`flash_attention_score_grad_entry_regbase.h:41-75`）：
```cpp
// Pre阶段：初始化workspace、dequant scales
FlashAttentionScoreGradS1S2BNGS1S2PreRegbase<...> preOp;
preOp.Process();

// Base阶段：主计算（5 MatMul + 6 Vector）
// 根据确定性模式选择不同的kernel类
if constexpr (DETER_SPARSE_TYPE == NO_DETER) {
    FlashAttentionScoreGradKernel<...> baseOp;
    baseOp.Process();
} else {
    FlashAttentionScoreGradKernelDeter<...> baseOp;
    baseOp.Process();
}

// Post阶段：FP8量化输出
FlashAttentionScoreGradS1S2BNGS1S2PostRegbase<...> postOp;
postOp.Process();
```

**5-MatMul流水**（`flash_attention_score_grad_kernel.h`）：
```
时序（双任务Ping-Pong，taskId模2）：

Task 0:  C1(DyV) → C2(QK) → [等V3] → C3(DsK) → C4(DsQ) → C5(PDy)
Task 1:                V1 → V2 → V3 → V4 → [等C3] → V5 → [等C4] → V6

重叠：当前轮C1/C2与上一轮V1-V6并行执行
```

**同步flag完整映射**（`flash_attention_score_grad_kernel_base.h`）：
```cpp
// Cube → Vector (8个)
SYNC_C1_TO_V2_FLAG[2] = {0, 1}   // mm1(dp)就绪 → V1/V2
SYNC_C2_TO_V2_FLAG[2] = {2, 3}   // mm2(QK^T)就绪 → V2
SYNC_C3_TO_V5_FLAG    = 6        // mm3(dq)就绪 → V5
SYNC_C4_TO_V6_FLAG    = 7        // mm4(dk)就绪 → V6
SYNC_C4_TO_V3_FLAG    = 8        // C4写L1完成 → V3 MTE3
SYNC_C5_TO_V4_FLAG    = 10       // mm5(dv)就绪 → V4

// Vector → Cube (2个)
SYNC_V3_TO_C3_FLAG    = 4        // ds写L1完成 → C3可读
SYNC_V4_TO_C5_FLAG    = 5        // P写L1完成 → C5可读

// 确定性模式专用
SYNC_DETER_FIX_FLAG   = 9        // UB完成通知
SYNC_SINK_UB_REUSE    = 11       // Sink场景UB复用
```

**与A3的关键区别**：
1. A3的FASG同样是5-MatMul结构，但Vector阶段使用高阶API
2. A5的Vector阶段通过VF函数实现（vector_api/目录）
3. A5的3阶段入口（Pre/Base/Post）支持FP8量化的前后处理

Benefit: 3阶段入口清晰分离初始化/计算/后处理；双任务Ping-Pong最大化Cube/Vector重叠
Trade-off: 10+个同步flag管理复杂；双任务需要2份RunInfo存储

---

## Variant B: 模块化Matmul设计（matmul_modules/）

Source: flash_attention_score_grad(arch35)/op_kernel/arch35/

FASG在A5上将5个MatMul操作封装为独立模块，每个模块包含条件输出路由（写UB vs 写Workspace vs Fixpipe输出）和确定性变体。

**模块化MatMul结构**（`flash_attention_score_grad_block_cube.h:197-238`）：
```cpp
// 5个MatMul模块，每个有3种变体
// 标准版
IterateMmDyV<CALC_TYPE>()           // MM1: dy×V^T → dp
IterateMmQK<CALC_TYPE>()            // MM2: Q×K^T → S
IterateMmDsK<CALC_TYPE, IS_WRITE_UB>()  // MM3: ds×K → dq
IterateMmDsQ<CALC_TYPE, IS_WRITE_UB>()  // MM4: ds^T×Q → dk
IterateMmPDy<OUTDTYPE, IS_WRITE_UB>()   // MM5: P^T×dy → dv

// 确定性变体
IterateMmDsKOlderDeter()            // MM3确定性版
IterateMmDsQOlderDeter()            // MM4确定性版
IterateMmPDyOlderDeter()            // MM5确定性版

// Fixpipe输出变体
IterateMmDsKFixpout()               // MM3 Fixpipe输出
IterateMmDsQFixpout()               // MM4 Fixpipe输出
IterateMmPDyFixpout()               // MM5 Fixpipe输出
```

**输出路由条件**（`flash_attention_score_grad_kernel_base.h:104-108`）：
```cpp
// 条件编译控制输出目标
IS_DQ_WRITE_UB   // dQ写UB（BN2模板，非多核累加）
IS_DK_WRITE_UB   // dK写UB（BN2/BN2S2模板，非确定性）
IS_DV_WRITE_UB   // dV写UB（BN2S2模板，非确定性）

// 不满足条件时写Workspace（GM），后续归约
```

**L1复用条件**：
```cpp
IS_L1_REUSE:    非确定性 && HEAD_DIM_ALIGN≤256(确定性≤192) && 非FP32/FP8
IS_L1_PRELOAD:  HEAD_DIM_ALIGN≤192 && 非FP32/FP8
```

Benefit: 模块化设计使每个MatMul独立可测试；输出路由条件编译消除运行时分支
Trade-off: 15个MatMul变体（5×3）增加代码量；条件编译组合需要仔细验证

---

## Variant C: VF Softmax梯度实现（vector_api/）

Source: flash_attention_score_grad(arch35)/op_kernel/arch35/vector_api/

FASG的6个Vector阶段通过20个VF文件实现，按功能分为Softmax梯度、反量化、dropout、Cast/转置、归约等类别。每个VF按对齐宽度（256/512/768）和数据类型分发。

**VF文件分类**（`vector_api/`目录）：
```
Softmax梯度（6个变体）:
  vf_softmax_grad_front_cast_aligned256_f16.h
  vf_softmax_grad_front_cast_aligned256_f32.h
  vf_softmax_grad_front_cast_aligned512_f16.h
  vf_softmax_grad_front_cast_aligned512_f32.h
  vf_softmax_grad_front_cast_aligned768_f16.h
  vf_softmax_grad_front_cast_aligned768_f32.h

反量化:
  vf_anti_quant_softmax_grad_front_cast.h

Dropout:
  dropout.h

PSE+Mask+Softmax:
  pse_atten_mask_muls_simple_softmax.h

Cast/转置:
  vf_cast_transdata_deconflict.h

归约:
  vf_ds_abs_reduce_max.h

工具:
  vf_common_utils.h
```

**对齐分发**（`vf_softmax_grad_front_cast.h:37-57`）：
```cpp
template <typename T1, typename T, uint32_t srcN, uint32_t HEAD_DIM_ALIGN>
__aicore__ inline void MySoftmaxGradFrontCast(...) {
    if constexpr (IsSameType<T1, float>::value) {
        if constexpr (srcN <= 256) {
            MySoftmaxGradFrontCastAligned256F32<...>(...);
        } else if constexpr (srcN <= 512) {
            MySoftmaxGradFrontCastAligned512F32<...>(...);
        } else if constexpr (srcN <= 768) {
            MySoftmaxGradFrontCastAligned768F32<...>(...);
        }
    } else { /* FP16 variants */ }
}
```

**与A3的关键区别**：
1. A3使用高阶API（Mul/Sub/Exp等）实现Softmax梯度
2. A5通过VF函数+寄存器实现，中间结果不写回UB
3. A5按对齐宽度分发（256/512/768），A3无此分发

Benefit: VF实现的Softmax梯度避免UB中间写回；对齐分发优化每种宽度的性能
Trade-off: 20个VF文件维护成本高；新增HEAD_DIM需要新增对齐变体

---

## Variant D: 确定性模式（DETER_OLD/DETER_NEW）

Source: flash_attention_score_grad(arch35)

FASG支持确定性计算模式，通过独立的workspace区域和单核归约保证多次运行结果bit-exact。

**确定性模式枚举**：
```cpp
DeterSparseType:
  NO_DETER       // 非确定性（AtomicAdd直接写GM）
  DETER_OLD      // 旧确定性算法（双倍dSL1Buf）
  DETER_DENSE    // 确定性密集
  DETER_CAUSAL   // 确定性Causal
  DETER_BAND     // 确定性Band
```

**确定性buffer分配**（`flash_attention_score_grad_kernel_base.h:296-300`）：
```cpp
if constexpr (DETER_SPARSE_TYPE == DETER_OLD) {
    // 双倍dSL1Buf用于旧确定性算法
    dSL1Buf: 2 × normal_size
} else if constexpr (DETER_SPARSE_TYPE != NO_DETER) {
    // 新确定性算法：dSL1Buf + dSTransL1Buf
    dSL1Buf: normal_size
    dSTransL1Buf: normal_size
}
```

**确定性workspace**：
```
非确定性: dq/dk/dv直接AtomicAdd到GM
确定性:   各核写独立workspace区域 → 单核FP32归约 → 最终输出

workspace额外开销:
  dq_workspace: coreNum × s1Size × dSize × 4B
  dk_workspace: coreNum × s2Size × dSize × 4B
  dv_workspace: coreNum × s2Size × dSize × 4B
```

Benefit: 确定性模式保证训练可复现性；独立workspace避免AtomicAdd的非确定性
Trade-off: workspace随核数线性增长；单核归约成为性能瓶颈

---

## 反向算子设计对比总结

| 特性 | 5-MatMul流水(A) | 模块化Matmul(B) | VF梯度(C) | 确定性模式(D) |
|------|----------------|----------------|----------|-------------|
| 核心机制 | Pre→Base→Post 3阶段 | 5×3=15个MatMul变体 | 20个VF文件 | 独立workspace+归约 |
| 同步flag | 10+个CrossCore | 输出路由条件编译 | N/A | SYNC_DETER_FIX_FLAG |
| 流水深度 | 双任务Ping-Pong | 每MatMul独立 | 按对齐宽度分发 | 额外同步点 |
| UB预算 | 210KB/248KB | IS_DQ/DK/DV_WRITE_UB | VF内寄存器驻留 | 额外dSTransL1Buf |
| 与A3差异 | VF替代高阶API | 模块化封装 | 对齐分发(256/512/768) | 相同确定性逻辑 |
| 代码量 | kernel.h(409行) | block_cube.h(1795行) | 20个VF文件 | kernel_deter.h |
