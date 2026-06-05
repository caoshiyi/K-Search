# PFA (prompt_flash_attention) 基座特征卡

## 基本信息
- 算子名称：prompt_flash_attention（Prefill 阶段 FA）
- 场景：推理 Prefill 阶段，多 token 前向计算（Sq>1）
- 架构支持：arch32 (910B, CVSAME/CVDIFF) + arch35 (950/A5, CVDIFF_BASE_API/CVDIFF_MLA)
- 输入：13 个（query/key/value 必选 + pse_shift/atten_mask/actual_seq_lengths*/deq_scale/quant_scale/quant_offset 等）
- 输出：attention_out
- 关键属性：num_heads(必选), scale_value, pre_tokens(default=214748647), next_tokens(default=0), input_layout(BSH/BNSD/BSND/SH/NSD/TND/NTD/BBH等), num_key_value_heads, sparse_mode, inner_precise

## 分核逻辑

### SplitCoreMode 枚举
| Mode | 值 | 切分方式 | 适用场景 |
|------|----|----------|----------|
| SPLIT_NBS_VECTOR | 0 | N×B×S 展开, Vec核均分 | 标准 |
| SPLIT_NBS_CUBE | 1 | 同上, Cube核主导 | Q 全量化 |
| SPLIT_ONEN_VECTOR | 2 | 每核1个head, Vec | 大 head |
| SPLIT_ONEN_CUBE | 3 | 每核1个head, Cube | 大head量化 |
| BALANCE_VECTOR | 4 | 负载均衡, Vec | 变长序列 |
| BALANCE_CUBE | 5 | 负载均衡, Cube | 变长量化 |

- 最多 64 核（CoreHeadNumTail[64]）
- Q tile: 固定 128（Q_TILE_CEIL）, KV tile: kSBlockTile=512

### 【A3】分核逻辑
- arch32 CVSAME/CVDIFF 模式
- 核数配置根据 layout 和 head 数动态选择

### 【A5】分核逻辑
- arch35 BALANCE 模式: startBlkArray[50]/endBlkArray[50] 每核 token block 范围
- arch35 CVDIFF_BASE_API/CVDIFF_MLA 模式

## 流水设计
CV 分离流水 + L1 自管理（TSCM）:

### 【A3+A5 通用】流水阶段
| 阶段 | 名称 | 执行核 | 计算内容 |
|------|------|--------|---------|
| C1 | BMM1 | AIC | Q·K^T, A tensor: TSCM(SCM DB), 输出到 SCM |
| V1 | Softmax | AIV | 读 SCM, +pse_shift+atten_mask, FlashV3 在线 Softmax, 更新 max/sum |
| C2 | BMM2 | AIC | P·V, A tensor: TSCM(from Softmax), 输出到 UB/SCM |
| V2 | Rescale | AIV | online rescale × exp(max_prev-max_cur), 累积输出到 GM |

### 【A3】TSCM 管理
- SCM 双缓冲: bmm2Scm[2], 各 64KB/32KB

### 【A5】TSCM 管理
- PFAGlobalTscmArray 管理 6 组 L1 buffer
- SCM 双缓冲: bmm2Scm[2], 各 64KB/32KB
- Softmax VF 变体: vf_softmaxflashv3.h / vf_softmaxflashv3_dn.h / vf_row_invalid.h

### 【A3】CV 配比
- KERNEL_TYPE_MIX_AIC_1_2

### 【A5】CV 配比
- KERNEL_TYPE_MIX_AIC_1_1

## 核间同步

### 【A3+A5 通用】核间同步
| 事件 | 方向 | 语义 |
|------|------|------|
| BMM1完成 flag | AIC→AIV | SCM 中 S tile 就绪 |
| Softmax完成 flag | AIV→AIC | TSCM 中 P tile 就绪 |

## Buffer 设计

### 【A3】L1/TSCM
- K/V 通过 SCM 双缓冲传递
- L1 buffer 数量和管理待补充

### 【A5】L1/TSCM
- 6 个 localScm, 每个 L1BUFSIZE
- K/V 通过 SCM 双缓冲传递
- 总 TSCM ~192KB

### 【A3+A5 通用】UB (AIV, PromptAttentionSingleCoreTensorSize)
| Buffer | 内容 | 大小公式 |
|--------|------|----------|
| mmResUbSize | BMM1 输出 | S_inner × S_outer × 4 |
| attenMaskUbSize | mask tile | - |
| pseShiftUbSize | pse tile | - |
| softmaxMax/Sum/Exp/Value | online softmax 状态 | - |
| bmm2ResUbSize | BMM2 输出(fp32) | S_outer × D × 4 |
| kvAntiquantUbSize | 伪量化 dequant | 按需 |

### 【A3+A5 通用】Workspace
- 较小，主要用于 FlashDecode 合并场景（PFA 自身无 splitKV）

## 精度链

### 【A3+A5 通用】精度链
| 场景 | Q/K/V | BMM 中间 | Softmax | 输出 |
|------|-------|----------|---------|------|
| 非量化 | FP16/BF16 | FP32 | FP32 | FP16/BF16 |
| W8A8 | INT8 | FP32(deq_scale1) | FP32 | INT8(quant_scale2) |
| 伪量化(KV INT8) | Q:FP16, KV:INT8 | FP32 | FP32 | FP16/BF16 |

## 关键约束
1. Prefill 场景 Sq>1
2. Q_TILE_CEIL=128 固定, kSBlockTile=512
3. arch35 SCM 总量约 192KB 不可超
4. 支持 GQA (headNumRatio = qHeadNum/kvHeadNum)
5. splitS2=1 开启 S2 细分(大序列), splitD=1 开启 D 细分(大 head_dim)
6. isRowInvalid=1 启用全 mask 行跳过优化
7. 支持 10+ 种 inputLayout: SH/BSH/BNSD/NSD/BSND/TND/NTD/NZ/BBH/BNBD
8. PagedAttention: blockTableDim2/blockSize/PABlockNumSum 控制
