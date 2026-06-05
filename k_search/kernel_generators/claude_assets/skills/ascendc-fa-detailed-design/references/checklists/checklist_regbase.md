# Regbase 专项检查模块（R1-R3）

仅 A5（arch35/351x）平台需执行。A5 的 AIV 核采用 Regbase 架构，数据流为：
`GM → L1 → UB → Register → 计算 → Register → UB → GM`

---

## 【R1】VF 函数结构设计

```
问题：请设计 __simd_vf__ 函数的结构。

FA 算子中，Vector 阶段通常包含多个子步骤（如 Softmax 内部的 max/sub/exp/sum/div），
每个子步骤可以是独立的 VF 函数，也可以融合为一个大 VF 函数。

设计决策：
  1. VF 函数划分策略：
     □ 每个 Vector 子步骤一个 VF 函数（细粒度，易调试）
     □ 每个 Vector 阶段一个 VF 函数（粗粒度，减少函数调用开销）
     □ 混合策略（热点路径融合，非热点路径分开）

  2. 对每个 VF 函数，列出：
     - 函数名和 __simd_vf__ 修饰
     - 参数列表（所有 Scalar 计算结果作为参数传入）
     - 内部的 MicroAPI 调用序列
     - 迭代次数和 mask 计算

  3. VF 函数与高阶 API 的混合使用：
     - 哪些步骤用 VF 函数（Regbase 路径，性能关键）
     - 哪些步骤用高阶 API（Membase 路径，逻辑复杂但非瓶颈）

典型 Softmax VF 分解（参考 FAS arch35）：
  vf_ReduceMax    → 求行最大值
  vf_SubAndExp    → 减 max + exp
  vf_ReduceSum    → 求 exp 之和
  vf_MulWithRcp   → × (1/sum)，完成归一化

⚠️ AscendC Tip（Regbase 关键约束）：
  - SIMD_VF 内部禁止 Scalar 计算（会打断 Vector 流水）
  - 寄存器申请顺序：MaskReg → RegTensor<dst_type> → RegTensor<src_type>
  - 数据流：UB → LoadAlign → Register → 计算 → StoreAlign → UB
  - A5 的 VecReg 宽度 = 256B（支持 2×VL=512 模式）
```

## 【R2】Scalar 前移审查

```
问题：审查所有 Vector 阶段中的 Scalar 计算，确保已移到 SIMD_VF 外部。

  ┌──────────────────────────┬────────────┬──────────────────────────────┐
  │ Scalar 计算内容          │ 当前位置   │ 处理方案                     │
  ├──────────────────────────┼────────────┼──────────────────────────────┤
  │ repeatTimes 计算         │ VF 外      │ ✅ 作为参数传入 VF           │
  │ oneRepSize 计算          │ VF 外      │ ✅ constexpr 编译期常量       │
  │ Scale 系数 (1/√d)        │ VF 外      │ ✅ host 侧计算，tiling 传入  │
  │ Mask 边界计算            │ ?          │ 需评估是否可移出             │
  │ 地址偏移计算             │ ?          │ 需评估是否可移出             │
  │ 循环控制变量             │ ?          │ VF 内部的 for 循环仍需保留   │
  │ [其他]                   │ ...        │ ...                          │
  └──────────────────────────┴────────────┴──────────────────────────────┘

⚠️ 不可移出的 Scalar 计算：
  - VF 内部 for 循环的计数器（i++）—— 必须在 VF 内
  - 依赖 VF 计算结果的标量判断 —— 需评估是否可重构

⚠️ AscendC Tip：
  - Scalar 指令发射会导致 Vector 流水停顿 1-2 个 cycle
  - 即使计算量很小（如 i*oneRepSize），累积效果在高迭代次数时显著
  - 预计算后以参数形式传入是标准做法
```

## 【R3】寄存器分配策略

```
问题：评估各 VF 函数的寄存器压力，确认是否存在溢出风险。

A5 寄存器资源：
  - Vector Register: 256 个（支持 2×VL=512 模式时减半）
  - Mask Register: 专用 mask 寄存器
  - Scalar Register: 通用标量寄存器

对每个 VF 函数，列出：
  1. 需要的 RegTensor 数量（输入 + 输出 + 中间临时）
  2. 是否使用 2×VL 模式（宽寄存器，寄存器数量减半）
  3. 寄存器复用策略（哪些 RegTensor 可以复用）

FA 典型场景的寄存器需求：
  - Softmax VF：src(1) + dst(1) + max(1) + tmp(1-2) = 4-5 个
  - BMM 后处理 VF：src(1) + scale(1) + dst(1) = 3 个
  - Update rescale VF：old_o(1) + new_s(1) + factor(1) + dst(1) = 4 个

若单个 VF 函数需要的寄存器超过可用数量，需要：
  - 拆分为多个 VF 函数
  - 通过 UB 中转部分中间结果
  - 减少同时活跃的 RegTensor 数量
```
