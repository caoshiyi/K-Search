# FASG (flash_attention_score_grad) 基座特征卡

## 基本信息
- 算子名称：flash_attention_score_grad
- 场景：训练反向，FA 梯度计算
- 架构支持：arch32 (910B) + arch35 (950/A5)
- 输入：8 必选(query/key/value/dy/softmax_max/softmax_sum/softmax_in/attention_in) + 19 可选(pse_shift/drop_mask/atten_mask/prefix/d_scale_*/query_rope/key_rope/sink 等)
- 输出：3 个(dq/dk/dv)
- 关键属性：scale_value, keep_prob, pre_tokens, next_tokens, head_num(必选), input_layout(必选), inner_precise, sparse_mode

## 分核逻辑

### 【A3】分核逻辑
- 切分轴：B × N2 × S1 × S2 多维展开
- 分核模式：BN2 / BN2S2 / BN2GS1S2（泛化）/ 确定性
- CV 比例：AIC:AIV = 1:2（AICV_RATIO_DEFAULT=2）

### 【A5】分核模式
| 模板 | 切分轴 | 进入条件 |
|------|--------|----------|
| BN2 | 核间切 B/N2 | S1≤128 且 S2≤128 且 G=1，非FP32 |
| BN2S2 | 核间切 B/N2/S2 | 非FP32，TND，B×N2×S2 > cubeCoreNum，G=1 |
| 确定性 | 核间切 B/N2/G/S1/S2 | isDeterministic=true |
| BN2GS1S2(泛化) | 核间切 B/N2/G/S1/S2 | 以上都不满足 |

- CV 比例：AIC:AIV = 1:2（AICV_RATIO_DEFAULT=2）
- C 侧基本块 128×128，V 侧基本块 64×128

## 流水设计
- 5 个 MatMul + 多个 Vector 阶段，双任务 Ping-Pong（FagRunInfo runInfos[2]）

### 【A3+A5 通用】流水阶段
| 阶段 | 名称 | 执行核 | 计算内容 |
|------|------|--------|---------|
| C1 | IterateMmDyV | AIC | dy × V^T → dp，写 mm1ResBuf |
| C2 | IterateMmQK | AIC | Q × K^T → S（重计算），写 mm2ResBuf |
| V1 | ProcessVec1 | AIV | softmax_grad: p 重计算 + ds 计算 |
| V2 | ProcessVec2 | AIV | PSE + atten_mask + SimpleSoftmax（重计算 P）|
| V3 | ProcessVec3 | AIV | dropout + Cast + ND→NZ → dSL1Buf 写 L1 |
| V4 | ProcessVec4 | AIV | P Cast + ND→NZ → pL1Buf 写 L1 |
| C3 | IterateMmDsK | AIC | ds × K → dq |
| V5 | MulsAndCast(DQ) | AIV | dq × scale + Cast → 输出 |
| C4 | IterateMmDsQ | AIC | ds^t × Q → dk |
| V6 | MulsAndCast(DK) | AIV | dk × scale + Cast → 输出 |
| C5 | IterateMmPDy | AIC | P^t × dy → dv |

- 时序：当前轮 C 侧与上一轮 V 侧重叠（taskId 模 2）
- IS_DQ_WRITE_UB / IS_DK_WRITE_UB / IS_DV_WRITE_UB：控制梯度是写 UB 还是 GM

## 核间同步

### 【A3+A5 通用】核间同步
| 事件名 | 事件ID | 方向 | 语义 |
|--------|--------|------|------|
| SYNC_C1_TO_V2_FLAG | {0, 1} | AIC→AIV | mm1(dp) 就绪→V2 可读 |
| SYNC_C2_TO_V2_FLAG | {2, 3} | AIC→AIV | mm2(QK^T) 就绪→V2 可读 |
| SYNC_V3_TO_C3_FLAG | 4 | AIV→AIC | ds 写 L1 完成→C3(dq) 可读 |
| SYNC_V4_TO_C5_FLAG | 5 | AIV→AIC | P 写 L1 完成→C5(dv) 可读 |
| SYNC_C3_TO_V5_FLAG | 6 | AIC→AIV | C3(dq) 完成→V5 |
| SYNC_C4_TO_V6_FLAG | 7 | AIC→AIV | C4(dk) 完成→V6 |
| SYNC_C4_TO_V3_FLAG | 8 | AIC→AIV | C4 写 L1 完成→V3 MTE3 |
| SYNC_DETER_FIX_FLAG | 9 | AIV→AIC | 确定性场景 UB 完成通知 |
| SYNC_C5_TO_V4_FLAG | 10 | AIC→AIV | C5(dv) 完成→V4 |
| SYNC_SINK_UB_REUSE | 11 | AIV→AIC | Sink 场景 UB 复用通知 |

## Buffer 设计

### 【A3+A5 通用】L1
| Buffer | 内容 | 策略 | 说明 |
|--------|------|------|------|
| dSL1Buf | ds (softmax grad) | DB | AIV(V3) UB→L1, AIC(C3/C4) L1→L0 |
| pL1Buf | P (重计算概率) | DB | AIV(V4) UB→L1, AIC(C5) L1→L0 |

- L1 复用条件(IS_L1_REUSE): 非确定性，HEAD_DIM_ALIGN≤256(确定性≤192)，非FP32/FP8
- L1 preload 条件(IS_L1_PRELOAD): HEAD_DIM_ALIGN≤192，非FP32/FP8

### 【A3】L0
| Buffer | 大小 | 说明 |
|--------|------|------|
| L0A/B | 各 64KB | - |
| L0C | 128KB | DKV Resident 条件: max(mm1,mm2,mm3)+mm4+mm5 ≤ L0C_MAX_SIZE |

### 【A5】L0
| Buffer | 大小 | 说明 |
|--------|------|------|
| L0A/B | 各 64KB | - |
| L0C | 256KB | DKV Resident 条件: max(mm1,mm2,mm3)+mm4+mm5 ≤ L0C_MAX_SIZE |

### 【A3+A5 通用】UB
| Buffer | 数量 | CrossCore | 说明 |
|--------|------|-----------|------|
| mm1ResBuf | DB[2] | AIC→AIV | dp 结果, fp32 |
| mm2ResBuf | DB[2] | AIC→AIV | QK^T 结果, fp32 |
| softmax_max/sum | DB | AIV 独占 | 来自 FAS 输出 |
| UB 总预算常量 | - | - | PRE_BUFFER=112KB, CAST=60KB, OUTPUT=30KB, RESERVE=8KB |

### 【A3+A5 通用】Workspace
| 场景 | 策略 |
|------|------|
| 固定预留 | WORKSPACE_BUFFER=20MB, RESERVED=64KB |
| dq workspace | 多核切 S1 时 dq 先写 workspace 再归约 |
| dk/dv workspace | BN2S2 切分时先写 workspace 后合并 |
| BN2 模板 | dk/dv 无需 workspace（无 S1 切分） |

## 精度链

### 【A3+A5 通用】精度链
| 步骤 | 精度 |
|------|------|
| mm1: dy×V^T → dp | fp32 |
| mm2: Q×K^T → S | fp32 |
| V1: SoftmaxGrad | fp32 |
| V2: SimpleSoftmax → P | fp32 |
| V3: ds = P×(dp-sfmg) | fp32→Cast fp16/bf16 NZ格式 |
| mm3: ds×K → dq | fp32 |
| mm4: ds^t×Q → dk | fp32 |
| mm5: P^t×dy → dv | fp32 |
| 输出 dq/dk/dv | ×scale 后 Cast 按 out_dtype |

## 确定性计算

### 【A3+A5 通用】确定性计算
- DeterSparseType: NO_DETER / DETER_OLD / DETER_DENSE / DETER_CAUSAL / DETER_BAND
- 确定性模式: 各核写 Workspace 独立区域，单核归约（FP32 bit-exact）
- 非确定性模式: 直接写 GM（AtomicAdd）

## 关键约束
1. D 模板化与 FAS 一致
2. BN2 模板进入条件严格(S1≤128且S2≤128且G=1)，性能最优
3. BN2S2 模板: dk/dv 按 S2 切，无需多核 S1 累加
4. ds 必须先完整写 L1(SYNC_V3_TO_C3_FLAG) 后 C3/C4 才能读
5. ROPE 影响 L1 preload: 确定性+ROPE 仍可走 L1 preload
6. TND Swizzle: MIN_SWIZZLE_S1=16384, BASE_SWIZZLE_BLOCK_NUM=8
7. FP8 quant block: S1=512, S2=512
8. GM 对齐: GM_ALIGN=512; BN2 最大 D 限制 BN2_MAX_D=512
