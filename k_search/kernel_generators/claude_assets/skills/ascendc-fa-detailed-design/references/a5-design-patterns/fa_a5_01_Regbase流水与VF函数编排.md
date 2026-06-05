# FA-A5-01: Regbase流水与VF函数编排 (Regbase Pipeline & VF Function Orchestration)

## Overview

A5平台FA类算子的核心流水设计：AIC(Cube)与AIV(Vector)仍采用四级流水（BMM1→Vec1→BMM2→Vec2），但AIV侧从Membase切换到Regbase编程模型。Vector计算通过`__simd_vf__`修饰的VF函数执行，数据流变为 UB→LoadAlign→Register→Compute→StoreAlign→UB。Scalar计算必须前移到VF外部，避免阻塞Vector流水。多AIV核场景下，AIV1的CrossCore事件ID自动偏移+16。

## When to Use

- A5(arch35/351x)平台的FA类算子，AIV核采用Regbase架构
- Vector阶段包含Softmax、Flash Update、CopyOut等多步计算，需要设计VF函数划分策略
- 需要在AIC/AIV之间编排四级或五级流水，并处理跨核同步
- 需要将Scalar计算（repeatTimes、mask边界、地址偏移）前移到VF外部

## Trade-off

- VF函数粒度选择：粗粒度（整个Vec阶段一个VF）减少调用开销但增加寄存器压力；细粒度（每个子步骤一个VF）易调试但增加UB中转
- Scalar前移增加VF参数数量，但消除Vector流水停顿（每次Scalar指令停顿1-2 cycle）
- AIV1事件偏移(+16)增加同步设计复杂度，需要为双AIV核分配独立事件集合
- Regbase的LoadAlign/StoreAlign要求32B对齐，非对齐数据需要额外处理

**Source operators**: flash_attention_score(arch35), fused_infer_attention_score(arch35), prompt_flash_attention(arch35), incre_flash_attention(arch35)

---

## Variant A: 训练前向四级流水 + VF Softmax分解（FAS arch35）

Source: flash_attention_score(arch35)

FAS在A5上保持与A3相同的四级流水结构（BMM1→Vec1→BMM2→Vec2），但Vec1/Vec2阶段改用VF函数实现。Softmax被分解为多个VF函数（ReduceMax→SubExp→ReduceSum→MulRcp），每个VF内部通过LoadAlign/StoreAlign访问UB数据。

**内核入口与CV比例声明**（`flash_attention_score_entry_regbase.h:88-95`）：
```cpp
template<uint8_t implMode, uint8_t layout, uint16_t s1TemplateType, ...>
inline __aicore__ void flash_attention_score_regbase(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 固定1:2 CV比例
    INVOKE_FA_OP_IMPL_BASEAPI(BaseApi::FlashAttentionScoreKernelTrain, ...);
}
```

**四级流水编排**（`flash_attention_score_kernel_train.h:160-220`）：
```cpp
// 环形缓冲 runInfo[taskId & 3] 实现4阶段重叠
RunInfo<isInfer> &runInfo1 = runInfo[taskId & 3];
this->SetRunInfo(runInfo1, runParam, taskId, s2LoopCount, ...);

if ASCEND_IS_AIC {
    this->cubeBlock.IterateBmm1(this->bmm1Buffers.Get(), runInfo1, this->constInfo);
}
if ASCEND_IS_AIV {
    this->vecBlock.IterateBmm2(this->bmm2Buffers.Get(), this->bmm1Buffers.Get(),
                                runInfo1, this->constInfo);
}
```

**VF Softmax分解模式**（`common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h:44-80`）：
```cpp
// 按N范围编译期分发不同VF实现
enum OriginNRange {
    GT_128_AND_LTE_256 = 0,  // 128 < N ≤ 256
    GT_64_AND_LTE_128,       // 64 < N ≤ 128
    EQ_128,                  // N == 128（优化路径）
    GT_0_AND_LTE_64,         // 0 < N ≤ 64（尾块）
    GT_256_AND_LTE_512,      // 256 < N ≤ 512
    GT_512_AND_LTE_1024,     // 512 < N ≤ 1024
};

// 编译期选择对应的VF实现
template <typename T, typename T2, ..., OriginNRange oriNRange = GT_64_AND_LTE_128, ...>
__aicore__ inline void ProcessVec1NoUpdate(...) {
    if constexpr (oriNRange == GT_128_AND_LTE_256) {
        ProcessVec1NoUpdateGeneralImpl256<...>(...);
    } else if constexpr (oriNRange == EQ_128) {
        ProcessVec1NoUpdateImpl128<...>(...);  // 128对齐优化
    }
    // ...
}
```

**与A3的关键区别**：
1. A3的Vec阶段直接操作UB（Membase），A5通过VF函数+寄存器间接操作
2. A3无N范围分发，A5按s2BaseSize编译期选择不同VF实现
3. A5的CV比例固定1:2（A3可选1:1或1:2）

Benefit: 编译期N范围分发消除运行时分支；VF函数内寄存器计算带宽高于UB直接操作
Trade-off: VF函数数量多（20+个文件），维护复杂度高；N范围变化需要新增VF变体

---

## Variant B: 推理统一调度器流水 + TSCM自管理（FIAS/PFA arch35）

Source: fused_infer_attention_score(arch35), prompt_flash_attention(arch35)

FIAS/PFA在A5上采用TSCM(Tensor Shared Compute Memory)自管理模式替代A3的L1显式管理。TSCM通过GlobalTscmArray管理6组L1 buffer，支持SCM双缓冲。VF函数通过TSCM队列与Cube核交换数据。

**TSCM初始化**（`prompt_flash_attention_entry_regbase.h:59-75`）：
```cpp
AscendC::Impl::Detail::PFAGlobalTscmArray tscmArray;
AscendC::Impl::Detail::tscmGlobalPFA = &tscmArray;
TSCM<QuePosition::VECIN, 1, 0x4> bmm2Scm[2];       // SCM双缓冲
tPipe.InitBuffer(bmm2Scm[0], 1, 65536);              // 64KB per buffer
tPipe.InitBuffer(bmm2Scm[1], 1, 65536);
tPipe.InitBuffer(tscmGlobalPFA->localScm[0], 1, L1BUFSIZE);  // 6组L1 buffer
// ... 初始化 localScm[1] ~ localScm[5]
```

**IFA TSCM初始化**（`incre_flash_attention_entry_regbase.h:120-131`）：
```cpp
AscendC::Impl::Detail::GlobalTscmArray tscmArray;
AscendC::Impl::Detail::tscmGlobal = &tscmArray;
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPing;       // Vec1 Ping
TSCM<QuePosition::VECIN, 1, 0x4> vec1ScmPong;       // Vec1 Pong
tPipe.InitBuffer(vec1ScmPing, 1, vec1ResultSize);
tPipe.InitBuffer(vec1ScmPong, 1, vec1ResultSize);
tPipe.InitBuffer(tscmGlobal->localQue[0], 1, qkvSize);  // 6组QKV队列
```

**与A3的关键区别**：
1. A3使用显式L1 buffer分配（InitBuffer + L1 TBuf），A5使用TSCM自管理（GlobalTscmArray）
2. A5的SCM双缓冲（bmm2Scm[2]）替代A3的L1 pingpong
3. A5的TSCM总量约192KB（6×L1BUFSIZE），需要精确规划

Benefit: TSCM自管理简化L1 buffer生命周期管理；SCM双缓冲与Cube流水天然匹配
Trade-off: TSCM总量有限（~192KB），大D场景下buffer紧张；自管理模式调试困难

---

## Variant C: VF函数内部结构与Scalar前移（通用模式）

Source: common/arch35/vf/

所有A5 FA算子的VF函数遵循统一的内部结构：`__simd_vf__`修饰 → MaskReg/RegTensor声明 → for循环（LoadAlign→Compute→StoreAlign）。Scalar计算（repeatTimes、oneRepSize、mask边界）必须在VF外部完成并作为参数传入。

**标准VF函数结构**（`vf_basic_block_aligned128_update.h:28-60`）：
```cpp
template <typename T, typename T2, ..., uint32_t s1BaseSize = 128, uint32_t s2BaseSize = 128, ...>
__simd_vf__ void ProcessVec1UpdateImpl128VF(
    __ubuf__ T2 * expUb, __ubuf__ T * maxUb, __ubuf__ T * srcUb, ...,
    float divValue, const uint32_t blockStride, const uint32_t repeatStride,
    const float dScale, const uint16_t m, ...)
{
    // 1. 寄存器声明（顺序：MaskReg → dst RegTensor → src RegTensor）
    RegTensor<float> vreg_min, vreg_sel, vreg_input_x, vreg_max, vreg_exp_sum;
    UnalignRegForStore ureg_max, ureg_exp_sum;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();

    // 2. 主计算循环
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);       // UB → Register
        Muls(vreg_input_x, vreg_input_x, dScale, preg_all);   // Scale
        // ... Softmax计算 ...
        StoreAlign(expUb + i * s2BaseSize, vreg_exp, preg_all); // Register → UB
    }
}
```

**Scalar前移示例**（VF外部预计算）：
```cpp
// VF外部：所有Scalar计算在此完成
constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);  // 256B / sizeof(T)
uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
float dScale = scaleValue * rsqrt_d;  // host侧传入的scale

// 调用VF
SIMD_VF(ProcessVec1UpdateImpl128VF, expUb, maxUb, srcUb, ...,
        dScale, blockStride, repeatStride, m);
```

**寄存器申请规则**：
- MaskReg 最先声明（`preg_all`, `preg_compare`）
- dst RegTensor 次之（`vreg_exp`, `vreg_max`）
- src RegTensor 最后（`vreg_input_x`）
- 256个VecReg × 256B，2×VL模式下减半

Benefit: 统一的VF结构降低开发门槛；Scalar前移消除Vector流水停顿
Trade-off: 参数列表较长（10+个参数）；寄存器压力需要逐VF评估

---

## Variant D: 多AIV核事件偏移与Ping-Pong同步（A5通用）

Source: flash_attention_score(arch35), fused_infer_attention_score(arch35)

A5的1:2 CV比例意味着1个AIC核对应2个AIV核（AIV0和AIV1）。AIV1的CrossCore事件ID自动偏移+16，需要在同步设计中为两个AIV核分配独立的事件集合，并使用Ping-Pong模式避免事件冲突。

**事件ID分配规则**（`hw_a5_specs.md`）：
```
AIV0 和 AIC 共享一套事件ID空间
AIV1 的事件ID = AIV0 事件ID + 16
同一事件不能被多个消费者同时Wait

Ping-Pong同步：
  iter 0 (ping): 使用 eventId_set_A
  iter 1 (pong): 使用 eventId_set_B
  iter 2 (ping): 使用 eventId_set_A（已被iter 0消费完毕）
```

**FAS同步Flag定义**（`util_regbase.h`）：
```cpp
// AIC → AIV 同步
SYNC_C1_V1_FLAG = {0, 1}    // BMM1完成 → Vec1开始（ping/pong）
SYNC_C2_V2_FLAG = {5, 6}    // BMM2完成 → Vec2开始（ping/pong）

// AIV → AIC 同步
SYNC_V1_C2_FLAG = {2, 3, 4} // Vec1完成(P写L1) → BMM2开始

// AIV1偏移：实际事件ID = 上述ID + 16
```

**与A3的关键区别**：
1. A3的1:1模式下只有1个AIV核，无事件偏移问题
2. A5固定1:2，必须处理AIV1的+16偏移
3. A5需要更多事件ID（每个同步点×2个AIV核×ping/pong）

Benefit: 双AIV核并行处理提升Vector吞吐；硬件自动偏移简化软件逻辑
Trade-off: 事件ID空间有限（32个），复杂流水可能耗尽；调试时需要区分AIV0/AIV1的事件

---

## A3 vs A5 流水设计对比

| 特性 | A3 (910B/Membase) | A5 (950/Regbase) |
|------|-------------------|------------------|
| Vector编程模型 | Membase (直接操作UB) | Regbase (VF函数+寄存器) |
| 数据流 | UB → 计算 → UB | UB → LoadAlign → Reg → 计算 → StoreAlign → UB |
| CV比例 | 1:1 或 1:2 | 1:2（固定） |
| AIV核数 | 1或2 | 2（固定） |
| 事件ID偏移 | 无（1:1）或+16（1:2） | +16（固定） |
| Softmax实现 | 高阶API（Add/Mul/Exp等） | VF函数（LoadAlign+MicroAPI） |
| L1管理 | 显式InitBuffer | TSCM自管理（GlobalTscmArray） |
| Scalar计算 | 可在Vector阶段内执行 | 必须前移到VF外部 |
| 寄存器 | 无显式寄存器 | 256 VecReg × 256B |
| N范围分发 | 无 | 编译期按OriginNRange分发 |
| 流水深度 | 3-4阶段 | 4阶段（训练）/ 4+FD（推理） |
