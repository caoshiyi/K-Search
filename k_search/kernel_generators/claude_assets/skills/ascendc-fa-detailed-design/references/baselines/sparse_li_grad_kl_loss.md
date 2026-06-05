# SparseLI_GradKLLoss (sparse_lightning_indexer_grad_kl_loss) 基座特征卡

## 基本信息
- 算子名称：sparse_lightning_indexer_grad_kl_loss (aclnnSparseLightningIndexerGradKLLoss)
- 场景：Sparse LI 反向 + KL Loss 梯度融合算子
- 架构支持：arch32 (910b, MAX_CORE_NUM=25) + arch35 (950/A5, MAX_CORE_NUM_REGBASE=36, RegBase)
- 输入：8 必选(query/key/query_index/key_index/weight/sparse_indices/softmax_max/softmax_sum) + 4 可选(query_rope/key_rope/actual_seq_lengths_query/key)
- 输出：4 个(d_query_index/d_key_index/d_weight/loss)
- 关键属性：scale_value, layout("BSND"/TND), sparse_mode(3=RightDown causal), deterministic(bool)

## 接口特殊点
- query D=512（不带 RoPE）, key D=128
- query_index/key_index: D=dSizeQueryIndex=128（低维投影）
- query_rope/key_rope: D=dSizeRope=64（可选）
- sparse_indices: 来自前向 LI 输出的 TopK 索引
- kSize(TopK数) 当前仅支持 2048

## 分核逻辑
- 切分轴：B × S1 合轴，(bIdx, s1Idx) 线性化均匀分核
- tiling: bS1Index[MAX_CORE_NUM] 每核起始 B×S1 索引
- TND: FindBIndex 遍历 actualSeqLengths 映射
- CV: 每 AIC 对应 2 AIV, subBlockIdx=aivIdx%2, 两 AIV 各处理 N 头前/后半(nVecSize=AlignTo(gSize,2)/2)

### 【A3】分核逻辑
- MAX_CORE_NUM=25
- RegBase: 不适用，使用标准 FA base

### 【A5】分核逻辑
- MAX_CORE_NUM_REGBASE=36
- RegBase: 使用 BufferManager/BuffersPolicyDB 等 FA base 框架组件
- TilingData: SparseLightningIndexerGradKLLossRegBaseTilingData
- gSizeQuery 支持 128 或 64; gSizeQueryIndex 对应 64 或 32

## 流水设计
每个 (bIdx, s1Idx) 任务执行 **ProcessSY** + **ProcessP** 两大阶段:

### ProcessSY (SY = Q_index × K_index^T)
双 buffer 流水（kRunInfos[2]），syKBaseSize = min(1024, 16384/(n2Size×gSizeQueryIndex))

| 时序 | AIC | AIV |
|------|-----|-----|
| s2TaskId | ComputeMmSy(QI×KI^T) | ProcessVector0(VEC_SY_BASESIZE=256, gather KI) |
| +1 | 下一块 Cube | ProcessVector1(读bmm1, weighted softmax部分) |
| 结束后 | — | ProcessVector2(汇聚→reduceSumRes) |

### ProcessP (P = Q × K^T, 含 RoPE)
三 buffer 流水（kRunInfos[3]），pKBaseSize = min(1024, 16384/(n2Size×gSizeQuery))

| 时序 | AIC | AIV |
|------|-----|-----|
| s2TaskId | ComputeMmP(Q×K^T) | ProcessVector3(gather K) |
| +1 | 下一块 Cube | Vec4(softmax差值)→Vec5(ReLU grad)→Vec6(写reluGradResL1Buf) |
| +2 | ComputeMm3(dKI grad)→ComputeMm4(dQI) | ProcessVector7(读bmm3, 计算dKI部分) |

- ProcessVector8(SyncAll后): ScatterAdd 写 d_weight

## 核间同步

### 【A3+A5 通用】核间同步
所有事件: CrossCoreSetFlag<mode=2> / CrossCoreWaitFlag<mode=2>

| 事件名 | ID | 方向 | 触发时机 |
|--------|------|------|---------|
| SYNC_GATHER_TO_MM12_FLAG | {9, 10} | AIV→AIC | Vec0/3(gather)完成→Mm启动 |
| SYNC_MM2_TO_V1_FLAG | {0, 1} | AIC→AIV | Mm1/MmP 写bmm1Buffers完成→Vec1/4可读 |
| SYNC_V6_TO_C3_FLAG | 8 | AIV→AIC | Vec6(reluGrad写L1)完成→ComputeMm3 |
| SYNC_C3_TO_V7_FLAG | {2, 3} | AIC→AIV | Mm3 写bmm3Buffer完成→Vec7可读 |

## Buffer 设计

### 【A3+A5 通用】共享 Buffer (CV 间)
| Buffer | 位置 | 大小 | 用途 |
|--------|------|------|------|
| bmm1Buffers(DB) | UB, CROSS_CORE_SYNC_BOTH | 2×32×256×4=64KB | Mm1/MmP 结果 |
| bmm3Buffer(DB) | UB | 2×64×128×4=64KB | Mm3(dKI grad) |
| sYL1Buf(DB) | L1, CROSS_CORE_SYNC_BOTH | 2×128×576×sizeof(T) | gather SY数据 |
| reluGradResL1Buf | L1, SingleBuffer | 64×256×sizeof(T) | ReLU grad 中间 |

- UB_MAX_SIZE=128KB, L1_MAX_SIZE=512KB

### 【A3+A5 通用】Workspace (每核独立)
```
per_core:
| reduceSumRes   | 2 × kSize × fp32            |
| reluRes        | gSizeQueryIndex × kSize × fp32 |
| gatherSYRes    | kSize × dSizeQueryIndex × INPUT_T |

shared (all cores):
| scatterAddResGm | totalS2 × dSizeQueryIndex × fp32 |  (d_weight 累加)
```

## 精度链

### 【A3+A5 通用】精度链
| 步骤 | 精度 | 说明 |
|------|------|------|
| QI × KI^T (Mm1) | FP32(L0C)→UB/L1 | SY 矩阵 |
| Vec1/2(softmax 重算) | FP32 | 利用 softmax_max/sum |
| Q × K^T (MmP) | FP32(L0C)→UB | P 矩阵 |
| ReLU grad (Vec5) | FP32 | |
| dKI matmul (Mm3) | FP32 | (reluGrad^t)×QI |
| dQI matmul (Mm4) | FP32→FP16/BF16 | Fixpipe 下转 |
| ScatterAdd(d_weight) | FP32(workspace)→FP16/BF16 | Vec8 |
| loss 输出 | FP32 | KL 散度累加 |

## 关键约束
1. kSize(TopK) 当前仅支持 2048
2. dSizeQuery=512, dSizeQueryIndex=128, dSizeRope=64
3. n2Size(KV head) 默认 1
4. sparseMode 仅支持 3(RightDown causal)
5. 每 AIV 仅处理 G 维一半(subBlockIdx)，两 AIV 合并
6. deterministic=false 时 ScatterAdd 非原子（各核独立区域）; =true 路径待完善
7. syKBaseSize/pKBaseSize 由 UB 容量动态决定
