# FA-A5-08: DataCopy预处理与StoreAlign对齐 (DataCopy Preprocessing & StoreAlign Alignment)

## Overview

A5平台的Regbase架构要求所有Vector计算通过LoadAlign/StoreAlign在UB与寄存器之间搬运数据，对齐粒度为32B。与A3的直接UB操作不同，A5的数据搬运需要显式处理非对齐场景（DataCopyPad）、格式转换（ND→NZ）和模块化copy_cube_in设计。StoreAlign替代了A3的直接UB写入，所有Vector计算结果必须通过StoreAlign写回UB。

## When to Use

- A5平台FA算子的数据搬运设计（GM→L1→UB→Register）
- 需要处理非32B对齐的数据（DataCopyPad）
- 需要在VF内部或外部进行格式转换（ND→NZ、Cast）
- 需要设计模块化的copy_cube_in流程（GM→L1预加载）

## Trade-off

- 32B对齐（vs A3的512B）降低了padding开销，但仍需处理非对齐尾块
- StoreAlign引入显式写回步骤，但保证了寄存器→UB的对齐正确性
- DataCopyPad支持非对齐搬运，但增加了API调用复杂度
- 模块化copy_cube_in提升代码复用，但增加了函数调用层次

**Source operators**: flash_attention_score_grad(arch35), common/arch35/, flash_attention_score(arch35)

---

## Variant A: LoadAlign/StoreAlign基本模式（通用VF）

Source: common/arch35/vf/

A5的所有VF函数遵循统一的LoadAlign→Compute→StoreAlign数据流。LoadAlign从UB加载数据到寄存器（要求32B对齐），StoreAlign将计算结果从寄存器写回UB（带mask控制有效元素）。

**基本数据流**：
```
UB (32B对齐地址)
  ↓ LoadAlign(vreg, ubPtr)
Register (RegTensor<T>)
  ↓ Compute (Muls/Add/Exp/...)
Register (RegTensor<T>)
  ↓ StoreAlign(ubPtr, vreg, mask)
UB (32B对齐地址)
```

**LoadAlign/StoreAlign使用**（`vf_basic_block_aligned128_update.h`）：
```cpp
__simd_vf__ void ProcessVec1UpdateImpl128VF(...) {
    RegTensor<float> vreg_input_x;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        // LoadAlign: UB → Register（32B对齐）
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);

        // Compute: 寄存器内计算
        Muls(vreg_input_x, vreg_input_x, dScale, preg_all);

        // StoreAlign: Register → UB（带mask）
        StoreAlign(dstUb + i * s2BaseSize, vreg_input_x, preg_all);
    }
}
```

**非对齐数据处理**：
```cpp
// 方案1: padding到32B对齐
uint32_t alignedSize = (rawSize + 31) / 32 * 32;
// 分配buffer时使用alignedSize

// 方案2: UnalignRegForStore处理非对齐写回
UnalignRegForStore ureg_max, ureg_exp_sum;
// 用于Softmax状态等小数据的非对齐写回
```

**与A3的关键区别**：
1. A3直接操作UB（如`Add(dstUb, srcUb1, srcUb2, mask, repeatParams)`）
2. A5必须通过LoadAlign/StoreAlign中转寄存器
3. A5的32B对齐 vs A3的无显式对齐要求（API内部处理）

Benefit: 显式对齐保证数据访问正确性；寄存器计算带宽高于UB直接操作
Trade-off: 每次UB访问都需要LoadAlign/StoreAlign，增加指令数

---

## Variant B: DataCopyPad非对齐搬运（通用）

Source: common/arch35/

当源数据或目标数据不满足32B对齐时，使用DataCopyPad进行非对齐搬运。DataCopyPad在搬运过程中自动处理padding，将非对齐数据搬运到对齐的UB位置。

**DataCopyPad使用场景**：
```
场景1: GM→UB搬运，GM地址非32B对齐
  → DataCopyPad(ubDst, gmSrc, padParams)
  → padParams指定padding模式和填充值

场景2: 尾块处理，实际数据量不是32B的倍数
  → DataCopyPad自动padding到32B边界
  → 后续计算通过mask控制有效元素

场景3: 格式转换搬运（ND→NZ）
  → DataCopyPad + 格式转换参数
```

**与DataCopy的区别**：
```cpp
// DataCopy: 要求源和目标都32B对齐
DataCopy(ubDst, gmSrc, {blockLen, blockCount, srcStride, dstStride});
// 如果gmSrc非32B对齐 → 运行时错误

// DataCopyPad: 支持非对齐源
DataCopyPad(ubDst, gmSrc, padParams);
// padParams.leftPadding / rightPadding 控制padding
// 自动将非对齐数据搬运到对齐的UB位置
```

**A5 vs A3对齐要求**：
```
A3: GM地址512B对齐（GM_ALIGN = 512）
    → 大量padding开销
    → 例: D=65, 需padding到512B=256个fp16

A5: GM地址32B对齐（BLOCK_BYTE = 32）
    → padding开销大幅降低
    → 例: D=65, 只需padding到32B=16个fp16
    → DataCopyPad进一步处理剩余非对齐
```

Benefit: DataCopyPad覆盖所有非对齐场景；32B对齐大幅降低padding开销
Trade-off: DataCopyPad比DataCopy慢（额外padding处理）；需要正确设置padParams

---

## Variant C: 模块化copy_cube_in设计（FASG arch35）

Source: flash_attention_score_grad(arch35)/op_kernel/arch35/

FASG在A5上将GM→L1→L0的数据预加载封装为模块化的copy_cube_in函数，每个MatMul的输入数据搬运独立封装，支持L1复用和预加载优化。

**模块化搬运结构**（基于FASG arch35代码组织）：
```
flash_attention_score_grad/op_kernel/arch35/
├── matmul_modules/           # 模块化MatMul基础设施
│   ├── copy_cube_in_*.h      # 各MatMul的输入搬运模块
│   └── matmul_config_*.h     # MatMul配置
├── flash_attention_score_grad_block_cube.h  # Cube核实现
└── flash_attention_score_grad_kernel_base.h # 基类（buffer管理）
```

**L1复用与预加载**：
```cpp
// L1复用条件
IS_L1_REUSE:
  非确定性 && HEAD_DIM_ALIGN ≤ 256 (确定性 ≤ 192) && 非FP32/FP8

// L1预加载条件
IS_L1_PRELOAD:
  HEAD_DIM_ALIGN ≤ 192 && 非FP32/FP8

// L1 buffer分配
dSL1Buf:  ds (softmax grad) — AIV(V3) UB→L1, AIC(C3/C4) L1→L0
pL1Buf:   P (重计算概率)   — AIV(V4) UB→L1, AIC(C5) L1→L0
```

**搬运同步**：
```cpp
// V3完成ds写L1 → C3可以读L1
SYNC_V3_TO_C3_FLAG = 4   // AIV → AIC

// V4完成P写L1 → C5可以读L1
SYNC_V4_TO_C5_FLAG = 5   // AIV → AIC

// ds必须先完整写L1后C3/C4才能读
// → V3和C3之间有严格的数据依赖
```

Benefit: 模块化封装提升代码复用；L1复用减少GM搬运次数
Trade-off: 模块化增加函数调用层次；L1复用条件限制了适用场景

---

## Variant D: ND→NZ格式转换与Cast融合（FASG VF）

Source: flash_attention_score_grad(arch35)/op_kernel/arch35/vector_api/

FASG的V3/V4阶段需要将ds和P从ND格式转换为NZ格式（Cube核要求的数据布局），并同时执行Cast（FP32→FP16/BF16）。A5通过VF函数将Cast和ND→NZ融合为单个操作。

**Cast+ND2NZ融合VF**（`vector_api/vf_cast_transdata_deconflict.h`）：
```cpp
// V3: ds的Cast + ND→NZ
// 输入: ds (FP32, ND格式, 在UB中)
// 输出: ds (FP16/BF16, NZ格式, 写入L1 dSL1Buf)

// V4: P的Cast + ND→NZ
// 输入: P (FP32, ND格式, 在UB中)
// 输出: P (FP16/BF16, NZ格式, 写入L1 pL1Buf)
```

**Bank冲突规避**（`vf_cast_transdata_deconflict.h`）：
```
ND→NZ转换涉及数据重排，可能触发UB Bank冲突
→ deconflict策略：
  1. 源和目标使用不同bank group
  2. 中间结果通过寄存器中转（不经过UB）
  3. 分块处理避免同时访问同一bank
```

**与A3的关键区别**：
1. A3的Cast和ND→NZ是两个独立API调用，中间结果写回UB
2. A5在VF内融合Cast+ND→NZ，中间结果驻留寄存器
3. A5需要显式处理Bank冲突（deconflict），A3的Bank结构不同

Benefit: Cast+ND→NZ融合减少UB中间写回；寄存器中转避免Bank冲突
Trade-off: 融合VF的寄存器压力大；deconflict策略增加代码复杂度

---

## DataCopy与对齐设计对比总结

| 特性 | LoadAlign/StoreAlign(A) | DataCopyPad(B) | 模块化copy_cube_in(C) | Cast+ND2NZ融合(D) |
|------|------------------------|----------------|----------------------|-------------------|
| 核心机制 | UB↔Register显式搬运 | 非对齐GM→UB搬运 | GM→L1→L0模块化封装 | VF内Cast+格式转换 |
| 对齐要求 | 32B | 自动padding | 32B(GM) | 32B(UB) |
| 适用场景 | 所有VF内部 | 非对齐数据搬运 | FASG各MatMul输入 | FASG V3/V4阶段 |
| 性能影响 | 每次UB访问+1指令 | 比DataCopy慢 | L1复用减少GM搬运 | 减少UB中间写回 |
| Bank冲突 | 需注意src/dst bank | N/A | N/A | deconflict策略 |
| 与A3差异 | 新增(A3无显式搬运) | 32B vs 512B对齐 | 模块化封装(A3无) | 融合(A3分步执行) |
