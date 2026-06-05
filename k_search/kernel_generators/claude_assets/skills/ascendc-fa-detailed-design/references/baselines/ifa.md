# IFA (incre_flash_attention) 基座特征卡

## 基本信息
- 算子名称：incre_flash_attention（Decode 阶段 FA）
- 场景：推理 Decode 阶段，单/少 token 推理（Sq=1 为主）
- 架构支持：arch32 (910B, allvec) + arch35 (950/A5, RegBase)
- 输入：13 个（query/key/value 必选 + pse_shift/atten_mask/actual_seq_lengths*/block_table/kv_padding_size/antiquant_*/softmax_lse 等）
- 输出：attention_out
- 关键属性：num_heads(必选), scale_value, input_layout(BSH/BNSD/BSND), num_key_value_heads, block_size, inner_precise

## 分核逻辑

### 切分维度：B × N2(KV head) × S2(KV seq) — "Bbn2s2"

| 架构 | 每核 S2 结束索引数组 | 最大核数 |
|------|---------------------|---------|
| A3 (910B) | coreSidxEnd[50] | 50 |
| A5 (950) | coreSidxEndRegbase[66] | 66 |

- 负载均衡: formerCoreNum 个核多处理一块
- GQA: groupSplitSize, nNumOfQInOneGroup = qHeadNum/kvHeadNum
- multi-token IFA: s1SplitSize（Q seq 切分）

### 【A3】FlashDecode (splitKV)
- KV 序列切分到多核，各核算 partial O/m/l
- IncreFlashAttentionSplitKVParams: s2/sInnerLoopSize/accumOutSize/logSumExpSize
- 最终单核归约合并
- 最多 50 核

### 【A5】FlashDecode (splitKV, RegBase)
- KV 序列切分到多核，各核算 partial O/m/l
- IncreFlashAttentionSplitKVParams: s2/sInnerLoopSize/accumOutSize/logSumExpSize
- 最终单核归约合并
- 最多 66 核

## IFAProfile（编译期绑定 head_dim）

| Profile | D | S(inner) | M1 | N1 | K1 | M2 | N2 | K2 |
|---------|---|----------|----|----|----|----|----|----|
| D64 | 64 | 512 | 16 | 256 | 64 | 16 | 64 | 256 |
| D128 | 128 | 256 | 16 | 128 | 128 | 16 | 128 | 128 |
| D256 | 256 | 128 | 16 | 128 | 128 | 16 | 128 | 128 |
| D512 | 512 | 64 | 16 | 64 | 256 | 16 | 256 | 64 |

- Profile 驱动 GenConfMM1MDL/GenConfMM2Normal 生成 Matmul 静态 tiling

## 流水设计
CV 分离流水 + L1 自管理（L1_SELF_CONTROL）:

### 【A3+A5 通用】流水阶段
| 阶段 | 名称 | 执行核 | 计算内容 |
|------|------|--------|---------|
| C1 | MM1 | AIC | Mm1Processor::Send(): Q(TSCM) × K^T(GM) → S(UB/VECCALC) |
| V1 | Softmax | AIV | Vec1Processor: +pse_shift+mask, online softmax, P→TSCM |
| C2 | MM2 | AIC | Mm2Processor::Send(): P(TSCM) × V(GM) → O(UB) |
| V2 | Rescale | AIV | Vec2Processor: online rescale, 累积到 accumOutGm 或 attentionOutGm |
| FD | Combine | AIV | (splitKV only) 多段 partial O 按 LSE 加权合并 |

### 【A3+A5 通用】L1 自管理细节
| 阶段 | A Tensor | B Tensor | 说明 |
|------|----------|----------|------|
| MM1 A(Q) | TPosition::TSCM, depthA1=1 | — | — |
| MM1 B(K) | TPosition::TSCM, depthB1=2(自管理2块) | dbL0A=2, dbL0B=2 | — |
| PRE_LOAD_NUM | 2 | — | — |
| TSCM_DOUBLE_BUFFER | 2 | — | — |
| MM2 A(P) | TPosition::TSCM, 来自 Vec1 输出 SCM queue | — | — |
| MM2 B(V) | TPosition::GM, ND 格式 | — | — |

## 核间同步

### 【A3+A5 通用】核间同步
| 事件 | 方向 | 语义 |
|------|------|------|
| MM1完成 flag | AIC→AIV | UB 中 S 就绪，Vec1 可读 |
| Vec1完成(TSCM ready) | AIV→AIC | P 写入 TSCM，MM2 A tensor 可用 |
| MM2完成 flag | AIC→AIV | UB 中 O_tmp 就绪，Vec2 可读 |
| coreSidxEnd barrier | AIV 核间 | splitKV: 所有 partial 核完成后触发 combine |
| HardEvent M_MTE1(0~7) | AIC preset | L1 自管理预置 |
| MTE3_V/MTE3_MTE2/V_MTE2 | AIV preset | Vec 流水各阶段同步 |

## Buffer 设计

### 【A3】L1/TSCM (AIC, 自管理)
- Q tile: G × D（典型 16×128×2 = 4KB）
- K tile: N1 × D × 2(DB)（典型 128×128×2×2 = 64KB）
- 总 TSCM: SHARED_CO1_BUFFER_SIZE_KB = 64KB

### 【A5】L1/TSCM (AIC, 自管理, RegBase)
- 与 A3 相同 TSCM 结构和自管理方式
- 差异：A5 使用 RegBase，具体 register 分配有差异

### 【A3+A5 通用】UB (AIC/AIV 共享)
| Buffer | 内容 | 大小(示例 D128) |
|--------|------|-----------------|
| mmResUbSize | MM1 输出 | G×S_inner×4 ≈ 16×256×4 = 16KB |
| bmm2ResUbSize | MM2 输出 | G×D×4 ≈ 16×128×4 = 8KB |
| softmax tmp | max/sum/exp | G×4 each |

### 【A3+A5 通用】Workspace (GM)
| 用途 | 场景 | 大小 |
|------|------|------|
| accumOut | partial O | G×D×4×splitKVNum |
| logSumExp | partial LSE | G×4×splitKVNum |
| prefix | 共享前缀 | prefixAttenOutOffset/tmpLseOffset |

## 精度链

### 【A3+A5 通用】精度链
| 场景 | Q/KV | MM 中间 | Softmax | 输出 |
|------|------|---------|---------|------|
| 非量化 | FP16/BF16 | FP32 | FP32(online max/sum) | FP16/BF16 |
| 伪量化(per-tensor) | Q:FP16, KV:INT8 | dequant→FP16→FP32 | FP32 | FP16/BF16 |
| 伪量化(per-head) | 同上 | 逐head dequant | FP32 | FP16/BF16 |
| 伪量化(PA per-block) | 同上 | 每block dequant | FP32 | FP16/BF16 |

- antiquantPerTensorFlag / antiquantPerHeadFlag / antiquantParamsInPagedAttentionFlag 三者互斥

## 关键约束
1. 主要适用 Decode（Sq=1 或极短），Prefill 用 PFA
2. Profile 编译期绑定 D=64/128/256/512，不支持运行时切换
3. PagedAttention: block_size 需 BLOCK_SIZE_ALIGN_SIZE_16/128 对齐
4. L1 自管理总量不超 SHARED_CO1_BUFFER_SIZE_KB=64KB
5. 910B 最多 50 核，A5 最多 66 核
6. IFA TilingData 与 PFA 共享 tiling 注册类（直接 include PFA tiling.h）
7. Memory Bound 为主（Sq=1, 计算量小, 访存量大）
