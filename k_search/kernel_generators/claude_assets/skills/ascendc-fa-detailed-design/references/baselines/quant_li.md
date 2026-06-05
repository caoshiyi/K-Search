# QuantLI (quant_lightning_indexer) 基座特征卡

## 基本信息
- 算子名称：quant_lightning_indexer (aclnnQuantLightningIndexer)
- 场景：INT8/FP8 量化版 LightningIndexer
- 架构支持：ascend910b(INT8) + ascend950/A5(FP8_E4M3FN/HiFP8, arch35 特化)
- 输入：5 必选(query/key/weights/query_dequant_scale/key_dequant_scale) + 3 可选(actual_seq_lengths_query/key, block_table)
- 输出：sparse_indices(INT32)（**无 sparse_values**，与 LI 不同）
- 关键属性：query_quant_mode(0=per-token-head), key_quant_mode, layout_query, layout_key, sparse_count(≤2048), sparse_mode(3=causal)

## 与 LI 的关键差异
| 方面 | LI | QuantLI |
|------|-----|---------|
| Q/K dtype | FP16/BF16 | 910b:INT8, 950:FP8 |
| 输出 | indices + values | **仅 indices** |
| A5 支持 | 不支持 | **支持**（arch35 特化） |
| gSize 约束 | Q head ≤ 64 | 910b:≤64, 950:仅16/24/32/64 |
| LD 归约 | 支持 | 910b支持, arch35不支持 |

## 分核逻辑

### 【A3】分核逻辑
- M_BASE_SIZE=256, S2_BASE_SIZE=2048
- 基本块尺寸与 LI 不同（LI 为 M_BASE_SIZE=512, S2_BASE_SIZE=512）

### 【A5】分核逻辑
- S2_BASE_SIZE=128, S1_BASE_SIZE=4(固定)
- 与 A3 差异较大

## 流水设计

### 【A3】流水设计 (QLIPreload, 双buffer缓存任务)
- LI_QUANT_PRELOAD_TASK_CACHE_SIZE=2 的 RunInfo 缓存

| 时序 | AIC | AIV |
|------|-----|-----|
| 每 S1 首块 | 等syncV0C1→ComputeMm1→发syncC1V0 | 等syncC1V0→ProcessVec0(weight×scale)→发syncV0C1 |
| S2 主循环 | ComputeMm1→发syncC1V1 | 等syncC1V1→ProcessVec1(Score×weight+TopK)→发syncV1C1 |

### 【A5】流水设计 (严格同步, 无 LD)
```
AIC: WaitFlag(CROSS_VC_EVENT+loop%2)×2 → ComputeMm1 → SetFlag(CROSS_CV_EVENT+loop%2)×2
AIV: ProcessVec1 → isLastS2: ProcessTopK
```

## 核间同步

### 【A3】核间同步
| 事件 | 固定ID | 方向 | 管道 |
|------|--------|------|------|
| syncC1V1 | 0 | AIC→AIV | PIPE_FIX |
| syncC1V0 | 2 | AIC→AIV | PIPE_FIX(每S1首块) |
| syncV1C1 | 0 | AIV→AIC | PIPE_MTE3 |
| syncV0C1 | 1 | AIV→AIC | PIPE_MTE3 |
| AIV0_AIV1_OFFSET | +16 | - | AIV1 偏移 |

### 【A5】核间同步
| 事件 | 含义 |
|------|------|
| CROSS_VC_EVENT+0/1 (ID 0,1) | AIV→AIC, 通知 Cube 可启动 |
| CROSS_CV_EVENT+0/1 (ID 2,3) | AIC→AIV, 通知 Vec 完成 |
| QLI_SYNC_MODE4=4 | arch35 CrossCore 同步模式 |

## Buffer 设计

### 【A3】Workspace
| Buffer | 大小 |
|--------|------|
| mm1ResGm | 核数×2(DB)×s1BaseSize×s2BaseSize×fp32 |
| vec1ResGm | 核数×s1BaseSize×2×2×BASE_TOPK×fp32 |
| vec1ParamGm | 核数×s1BaseSize×2×16×int64 |
| weightWorkspaceGm | 核数×2(DB)×BLOCK_CUBE×mBaseSize×half |

### 【A5】Workspace
| Buffer | 大小 |
|--------|------|
| scoreGm | 核数×s1BaseSize×align(s2Size,s2BaseSize)×uint16 |

### 【A3+A5 通用】L1 (AIC)
| Buffer | 大小 |
|--------|------|
| bufQL1_ | 2×mBaseSize×128×sizeof(Q_T) |
| bufKeyL1_ | 3×128×128×sizeof(K_T) |
| bufUB_(VECCALC) | 2×CeilDiv(mBaseSize,2)×s2BaseSize×fp32 ≈128KB |

### 【A5】Fixpipe 特殊设计
- L0C→UB: FixpipeParamsC310<CO2Layout::ROW_MAJOR>, dualDstCtl=1 双目标模式
- UB bank 参数: UB_BLOCK=32, UB_BANK_GROUPS=8, UB_BANKS=2

## 精度链

### 【A3】精度链
| 运算 | 910b 精度 |
|------|-----------|
| Q×K^T | FP32(Fixpipe GM, 含反量化) |
| 反量化 | Vec0: weight×qScale×kScale→FP16(weightWS) |
| TopK 排序 | FP32(MrgSort) |
| 输出 | INT32 |

### 【A5】精度链
| 运算 | arch35 精度 |
|------|-------------|
| Q×K^T | FP32(Fixpipe→UB) |
| 反量化 | AIC Fixpipe 内置 scale |
| TopK 排序 | FP16(score以uint16存, ProcessTopK) |
| 输出 | INT32 |

## 关键约束
1. HEAD_DIM 固定 128, K_HEAD_NUM 固定 1
2. arch35 s1BaseSize=4 固定, 不支持 LD 归约（无 ProcessDecode）
3. 量化 mode 仅支持 per-token-head (mode=0)
4. 950 gSize 仅支持 16/24/32/64
5. 无 return_values（仅输出 indices）
6. blockSize(PA) ≤ 1024 且为 16 的倍数
