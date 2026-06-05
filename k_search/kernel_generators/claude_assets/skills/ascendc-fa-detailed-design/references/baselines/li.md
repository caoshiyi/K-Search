# LI (lightning_indexer) 基座特征卡

## 基本信息
- 算子名称：lightning_indexer (aclnnLightningIndexer)
- 场景：Q×K^T 加权 TopK 索引选择，稀疏注意力核心索引算子
- 架构支持：**仅 A3（ascend910b + ascend910_93），不支持 A5（ascend950）**
- 输入：3 必选(query/key/weights) + 3 可选(actual_seq_lengths_query/key, block_table)
- 输出：sparse_indices(INT32) + sparse_values(BF16/FP16, 由 return_values 控制)
- 关键属性：layout_query("BSND"/TND), layout_key("BSND"/TND/PA_BSND), sparse_count(≤2048), sparse_mode(3=causal), return_values(bool)

> **重要**：LI **仅支持 A3 架构**，不支持 A5。以下所有设计章节均为 A3 专属内容。

## 分核逻辑

### 【A3】分核逻辑
- 切分维度：B×N2×gS1×S2 四维展开，均匀分基本块
- M_BASE_SIZE=512(gS1方向), S2_BASE_SIZE=512(KV方向)
- s1BaseSize: sparseCount≤2048 时 =8; >2048 时 =SPARSE_COUNT_8K/sparseCount×2
- SplitCore: 双闭区间 [bN2Start..bN2End, gS1Start..gS1End, s2Start..s2End]
- CV 比例: 1 AIC : 2 AIV (aiCoreIdx = blockIdx/2)

## 流水设计
主流程: **ProcessMain** + **ProcessDecode**(LD 归约)

### ProcessMain (ProcessBaseBlock)
```
AIC: WaitFlag(syncV1C1) → ComputeMm1 → SetFlag(syncC1V1)
AIV: WaitFlag(syncC1V1) → ProcessVec  → SetFlag(syncV1C1)
```

### ProcessVec 内部（每 S1 行）
1. 从 workspace mm1ResGm 搬入 Score (DB ping/pong)
2. DoScale: weights 加权 (groupInner=16 切 G 维)
3. DoReduce: G 维规约
4. SortAll + MergeSort / Sort4+MrgBasicBlock → 更新 globalTopkUb_
5. S2 结束: Extract 取索引 + CopyOut 写 GM

### ProcessDecode (LD 阶段)
- 触发条件: splitCoreInfo.isLD == true
- SyncAll() 全局栅栏后，单独由 isLD AIV 执行 MrgSort4 跨核归约

## 核间同步

### 【A3】核间同步
| 事件 | 方向 | 管道 | 语义 |
|------|------|------|------|
| syncC1V1 | AIC→AIV | PIPE_FIX | Score 计算完成 |
| syncV1C1 | AIV→AIC | PIPE_MTE2 | Vec 处理完成 |
| V_MTE2 PING/PONG/TMPUB(0/1/2) | AIV 内 | HardEvent | Score+weight 搬入三路流水 |
| CrossCoreSetFlag<MODE2> | 核间 | FIA_SYNC_MODE2=2 | 固定模式 |

## Buffer 设计

### 【A3】Workspace (每 AIC 核独占)
| Buffer | 大小 | 说明 |
|--------|------|------|
| mm1ResGm | 核数×2(DB)×mBaseSizeAlign×s2BaseSize×fp32 | Score 矩阵 |
| vec1ResGm | 核数×s1BaseSize×2×2×BASE_TOPK×fp32 | TopK 结果 |
| vec1ParamGm | 核数×s1BaseSize×2×16×int64 | TopK 参数 |

### 【A3】L1 (AIC 侧)
| Buffer | 大小 | 数量 |
|--------|------|------|
| bufQL1_ | 512×128×sizeof(Q_T) | QUERY_BUF_NUM=2(DB) |
| bufKeyL1_ | 512×128×sizeof(K_T) | KEY_BUF_NUM=3(三 buffer) |

### 【A3】L0
| Buffer | 大小 | 数量 |
|--------|------|------|
| bufQL0_ | L0_BUF_NUM(2)×128×128×sizeof(Q_T) | 2 |
| bufKeyL0_ | L0_BUF_NUM(2)×128×128×sizeof(K_T) | 2 |
| bufL0C_ | L0_BUF_NUM(2)×128×128×fp32 | 2 |

### 【A3】UB (AIV 侧, LIVector::InitBuffers)
| Buffer | 大小 | 说明 |
|--------|------|------|
| outQueue_ | max(32KB, reduceCacheSize) | extract 输出 |
| tmpBuf_ | (groupInner×s2BaseSize+s2BaseSize)×2×4 ≈68KB | Score+weight DB |
| sortOutBuf_ | CeilDiv(s1BaseSize,2)×virTopK×2×4 ≈64KB | MrgSort 中间 |
| indexBuf_ | s2BaseSize×4 = 2KB | 索引 |
| reduceOutBuf_ | s2BaseSize×2×4 = 4KB | 归约输出 |
| ldToBeMrgBuf_/ldTmpBuf_ | 各 2×BASE_TOPK×4×4 = 64KB | LD 阶段 |

## 精度链

### 【A3】精度链
| 运算 | 精度 | 说明 |
|------|------|------|
| Q×K^T (Matmul) | FP32 (MM1_OUT_T=float) | L0C→workspace |
| Scale (DoScale) | FP32 | weights 若 FP16/BF16 则 Cast 到 FP32 |
| TopK 排序 | FP32 | Sort/MrgSort 在 FP32 域 |
| 输出 indices | INT32 | Extract 后 ReinterpretCast |
| 输出 values | 同 K_T | Cast from FP32 |

## 关键约束
1. HEAD_DIM 固定 128（编译期常量）
2. K_HEAD_NUM 固定 1（GQA, KV head=1）
3. sparse_count ≤ 2048（SPARSE_LIMIT）; >2048 时进入 isSparseCountOver2K 路径
4. sparseMode==3 时启用 attenMaskFlag, S2 有效长度随 S1 收窄
5. PageAttention: block_table 非空启用, blockSize 须 16 的倍数
6. 超出 usedCoreNum 的核调用 ProcessInvalid（输出 -1）
7. **不支持 A5**（无 ascend950 配置）
