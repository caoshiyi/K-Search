# FAS (flash_attention_score) 基座特征卡

## 基本信息
- 算子名称：flash_attention_score
- 场景：训练前向，FlashAttention 核心算子
- 架构支持：arch32 (910B) + arch35 (950/A5)
- 输入：3 必选(query/key/value) + 16 可选
- 输出：4 个(softmax_max/softmax_sum/softmax_out/attention_out)
- 关键属性：scale_value, keep_prob, pre_tockens, next_tockens, head_num(必选), input_layout(必选), inner_precise, sparse_mode

## 分核逻辑

### 【A3】分核逻辑
- 切分轴：S1（Query 序列长度方向）
- 总工作量：totalSize = B × N2 × G × ceil(S1 / s1BasicBlock)
- 分核模式：SQ_SINGLE_CORE_FIRST（静态）
- CV 比例：AIC:AIV = 1:2（KERNEL_TYPE_MIX_AIC_1_2）

### 【A5】分核逻辑
- 切分轴：S1（Query 序列长度方向）
- 总工作量：totalSize = B × N2 × G × ceil(S1 / s1BasicBlock)
- 分核模式：SQ_MULTI_CORE_FIRST（动态，优化）
  - 顺序分核 / 对称分核（Causal 负载均衡）/ 正倒序循环分核（TND L2 复用）
- CV 比例：AIC:AIV = 1:2（KERNEL_TYPE_MIX_AIC_1_2）

## 流水设计
- 流水深度：4 阶段环形缓冲（runInfo[4]）

### 【A3+A5 通用】流水阶段
| 阶段 | 名称 | 执行核 | 计算内容 |
|------|------|--------|---------|
| C1 | BMM1 | AIC | Q × K^T → S (L0C) |
| V1 | Softmax+Update | AIV | scale + mask + exp + sum + Online rescale |
| C2 | BMM2 | AIC | P × V → O_tile (L0C→UB/workspace) |
| V2 | CopyOut | AIV | attention output 归约 + CopyOut |

- 时序：BMM1(i) ‖ Vec1(i-1) ‖ BMM2(i-2) ‖ Vec2(i-3)，最后 3 轮排空

## 核间同步

### 【A3】核间同步
| 事件名 | 事件ID | 方向 | 语义 |
|--------|--------|------|------|
| SYNC_C1_V1_FLAG | {0, 1} | AIC→AIV | BMM1 完成→Vec1 开始 |
| SYNC_V1_C2_FLAG | {2, 3, 4} | AIV→AIC | Vec1 完成(P写L1)→BMM2 开始 |
| SYNC_C2_V2_FLAG | {5, 6} | AIC→AIV | BMM2 完成→Vec2 开始 |

### 【A5】核间同步
- 与 A3 相同事件 ID 语义
- 差异：A5 使用 RegBase，事件语义与 A3 完全一致

## Buffer 设计

### 【A3】L1 Buffer
| Buffer | 内容 | 策略 |
|--------|------|------|
| l1QBuffers | Query tile | D≤256非fp32: DB; 否则单块 |
| l1KBuffers | Key tile | 默认 DB; fp8特殊4-buffer |
| l1VBuffers | Value tile | 同 l1KBuffers |
| l1PBuffers | Softmax P | 3-buffer, CROSS_CORE_SYNC, AIV写→AIC读 |

### 【A5】L1 Buffer
- L1 TSCM 双缓冲方案：Sink 常驻策略
- L1 buffer 数量与 A3 相同，管理方式不同

### 【A3+A5 通用】L0
| Buffer | 内容 | 策略 |
|--------|------|------|
| L0A/L0B | - | 默认 DB（fp32单块） |
| L0C | s1×s2×4≤L0C/4 且 s1×dV×4≤L0C/4 | 4-buffer; 否则 DB |

### 【A3+A5 通用】UB
| Buffer | 内容 | 策略 |
|--------|------|------|
| bmm1Buffers | BMM1 结果 | CROSS_CORE_SYNC_BOTH DB, AIC写AIV读, fp32 |
| bmm2Buffers | BMM2 结果 | CROSS_CORE_SYNC_BOTH, 默认DB |
| softmaxMax/Sum/ExpBuf | Softmax 状态 | 各3缓冲×256B |
| attenMaskInQue/pseInQue/dropMaskBuf | Mask/Pse | 各缓冲 |

### 【A3+A5 通用】Workspace
| 条件 | 是否开启 | 大小 |
|------|----------|------|
| D>128(非DN) 或 D>192(DN) | 开启 | (bmm2Bytes + vec2Bytes) × 3 × coreNum |
| 其他 | 不开启 | - |

## 精度链

### 【A3+A5 通用】精度链
| 输入dtype | BMM1 | Vec1(Softmax) | BMM2 | 输出 |
|----------|------|---------------|------|------|
| fp16高精度 | fp16→fp32 | fp32 | fp32 | fp16 |
| bf16 | bf16→fp32 | fp32 | fp32 | bf16 |
| fp8 | fp8→fp32 | fp32 | fp32 | fp16 |

- softmax_max/softmax_sum 恒为 fp32

## 关键约束
1. D模板分档：16/32/.../256/320/384/448/768，超768报错
2. s1/s2BasicBlock 默认128×128(arch35)，按 D/dtype 动态调整
3. GM 地址 512B 对齐
4. Sparse 模式 9 种(NO_MASK/.../BAND_LEFT_UP_CAUSAL)
5. PSE 类型 6 种
6. attenMask 压缩模式: S≤2048(PREFIX≤3072)
7. FP8 场景 s2LineStartIdx 按 128 对齐
8. TilingKey: 64位编码(dtype+layout+mode+sparse组合)
