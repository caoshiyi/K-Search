# FIAS (fused_infer_attention_score) 基座特征卡

## 基本信息
- 算子名称：fused_infer_attention_score (ai_infra_fused_infer_attention_sink)
- 场景：推理统一调度器，集成 IFA/PFA/MLA/SplitFuse 等多种推理模式
- 架构支持：arch32 (910B) + arch35 (950/A5, FAInferKernel)
- 架构风格：Cutlass-style attn_infra 基础设施
- 输入：25+ 个（query/key/value 必选 + pse_shift/atten_mask/block_table/antiquant_*/quant_*/deq_scale 等可选）
- 输出：attention_out + softmax_lse
- 关键属性：num_heads(必选), scale, input_layout(BSH/BNSD/BSND/TND), num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag

## 架构设计
```
attn_infra/
├── core/           # 核心计算抽象 (bmm/softmax/update)
├── dispatch/       # 模式分发 (ifa/pfa/mla/split_fuse)
└── infra/          # 基础工具 (buffer/sync/tiling)
```

### 调度模式
| 模式 | 触发条件 | 说明 |
|------|----------|------|
| IFA | Sq=1 (Decode) | 单 token 推理 |
| PFA | Sq>1 (Prefill) | 多 token 预填充 |
| MLA | numKeyValueGroupsPerBlock>1 | Multi-Latent Attention |
| SplitFuse | splitFuse=true | Split + Fuse 混合 |
| FlashDecode | splitKV=true | KV 分块并行 decode |

## 分核逻辑

### 外切（B×N2×gS1 三维展开）
- TilingData: FusedInferAttentionOuterSplitParams
  - bN2End[core_i], gS1End[core_i], s2End[core_i]
- 非 FlashDecode: (B, N1, S1_block) 三维线性化，round-robin 分核
- FlashDecode: 额外按 S2 切分，FlashDecodeParams 管理 bN2Idx/gS1Idx/s2SplitNum/s2SplitStart
- FIA_MAX_AIC_CORE_NUM = 26

### 内切（UB 分块）
- mBaseSize = qSBlockTile = 128（Q 方向）
- s2BaseSize = kSBlockTile = MAX_KV_STACK_LEN = 512（KV 方向）

### 【A3】CV 比例
- KERNEL_TYPE_MIX_AIC_1_2（1:2）

### 【A5】CV 比例
- KERNEL_TYPE_MIX_AIC_1_1 或视任务选择

## 流水设计
采用 SplitFuse::FAInferKernel 框架:

### 【A3+A5 通用】流水阶段
| 阶段 | 名称 | 执行核 | 计算内容 |
|------|------|--------|---------|
| C1 | BMM1 | AIC | BlockMmadQK: Q(128×D) × K^T(D×512) → S(workspace gS) |
| V1 | Softmax | AIV | EpilogueOnlineSoftmax: pse_shift + atten_mask + online softmax → P(gP) + LSE(gLse) |
| C2 | BMM2 | AIC | BlockMmadPV: P × V → O_tmp(gOTmp) |
| V2 | Rescale | AIV | EpilogueRescaleO: online rescale + 累积到最终 O |
| FD | Combine | AIV | (FlashDecode only) 多段 partial O 经 LSE 加权合并 |

## 核间同步

### 【A3+A5 通用】核间同步
| 事件ID | 方向 | 语义 |
|--------|------|------|
| QK_READY_ID=1 | AIC→AIV | BMM1 输出写入 workspace 完成 |
| SOFTMAX_READY_ID=2 | AIV→AIC | Softmax P 写入 TSCM 完成，可开始 BMM2 |
| PV_READY_ID=3 | AIC→AIV | BMM2 输出写入 workspace 完成 |
| AIV_INTER_BLOCK_BARRIER=8 | AIV 核间 | FlashDecode 全局 barrier |
| AIC_INTER_BLOCK_BARRIER=9 | AIC 核间 | 多核 AIC barrier |
| AIV_INTER_SUBBLOCK=10 | AIV 子块间 | 子块级同步 |
| M_MTE1(0~7) | AIC 内部 | L1 自管理搬入 preset |

## Buffer 设计

### 【A3】L1 (TSCM, self-control)
- Q tile: Q_TILE_CEIL(128) × D_round × sizeof(T)
- K tile: 双缓冲, nDynNum × D × sizeof(K_T) × 2; nDynNum ≤ L1_MAX_N_NUM=128
- V tile: nDynNum × D × sizeof(V_T) × 2
- L1 总上限: L1_MAX_SIZE = 524288 (512KB)

### 【A5】L1 (TSCM, self-control)
- 与 A3 相同 L1 自管理结构
- 差异：A5 使用 RegBase 管理 L1，具体 buffer 布局可能有细微差异

### 【A3+A5 通用】Workspace (GM)
| Buffer | 内容 | 说明 |
|--------|------|------|
| gS | BMM1 中间结果(mm1ResSize) | Score 矩阵 |
| gP | Softmax 输出(smOnlineOutSize) | P 矩阵 |
| gOTmp | BMM2 中间结果(mm2ResSize) | 部分 O |
| gOUpdate | rescale 中间 | Online update |
| gLseFD | FD accumOutSize | FlashDecode partial LSE |
| gOFD | FD logSumExpSize | FlashDecode partial O |

### 【A3+A5 通用】UB (AIV)
- softmax max/sum/exp 各一份，按 S1_block × N_tile 大小
- mask/pse_shift 临时 buffer

## 精度链

### 【A3+A5 通用】精度链
| 场景 | Q/K/V | 中间计算 | 输出 |
|------|-------|----------|------|
| 非量化 | FP16/BF16 | FP32(softmax/rescale) | FP16/BF16 |
| W8A8 | INT8 | FP32(dequant→softmax→quant) | INT8/FP16 |
| 伪量化 | Q:FP16/BF16, KV:INT8/INT4 | dequant→FP16→FP32 | FP16/BF16 |
| FP8/FP4/HIFLOAT8 KV | Q:FP16/BF16, KV:低精度 | dequant→FP16→FP32 | FP16/BF16 |
| MLA | Q:FP16/BF16(D=192) | FP32 | FP16/BF16 |

## 关键约束
1. B≤65536, N1/N2≤256, D≤512, G≤64
2. S1/S2 最大 20971520（约20M tokens）
3. PagedAttention: block_size 需 16 对齐（非量化max 1024，量化max 512）
4. GM 地址 32 字节对齐
5. FlashDecode: workspace 需为每个 partial 任务预留 LSE + O 空间
6. MLA: headDimRope=64, headDim=192(QK) + 128(V)
7. TilingKey: 12 维编码，编译期+运行期联合决策
