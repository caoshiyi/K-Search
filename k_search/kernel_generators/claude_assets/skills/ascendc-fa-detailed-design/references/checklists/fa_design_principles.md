# FA 类算子设计原则（详设参考）

本文档总结 FA 类算子的核心设计原则，供详细设计阶段参考。

---

## 1. FlashAttention 核心思想

### 算法本质
- **减少 HBM 访问**：通过分块（Tiling）计算 Attention，避免物化完整的 S=QK^T 矩阵到 HBM
- **Online Softmax**：在分块迭代中在线维护 max(m) 和 sum(l)，无需两遍遍历
- **IO 复杂度**：O(N²d/M)，其中 M 是 SRAM（UB/L1）大小，显著优于标准 Attention 的 O(N²)

### 核心公式
```
标准 Attention:
  S = QK^T / √d
  P = Softmax(S)
  O = PV

Online Softmax 伪代码:
  m = -inf, l = 0, O = 0
  for each block j:
    S_j = Q × K_j^T / √d
    m_new = max(m, rowmax(S_j))
    P_j = exp(S_j - m_new)
    l_new = l × exp(m - m_new) + rowsum(P_j)
    O = O × (l × exp(m - m_new) / l_new) + P_j × V_j / l_new
    m = m_new, l = l_new
```

## 2. 昇腾 FA 设计原则

### 2.1 Cube+Vector 流水分工
| 计算类型 | 执行单元 | 典型操作 |
|---------|---------|---------|
| 矩阵乘法 | AIC (Cube) | Q×K^T (BMM1), P×V (BMM2) |
| 逐元素操作 | AIV (Vector) | Scale, Mask, Exp, Sum, Div, Update |

**CV 比例**：A5 固定 1:2（1 个 AIC 配 2 个 AIV），两个 AIV 处理不同的数据块。

### 2.2 分块策略

#### 核间切分（多核并行）
- **训练前向**：S1(Sq) 轴切分，每个 core 组处理不同的 Query 块
- **训练反向**：S1 轴切分，需要跨核归约 dK/dV
- **推理**：B×N 轴切分，每个 core 处理不同的 batch/head

#### 核内切分（单核内分块）
- **外层循环**：遍历 K/V 的 block（Sk 轴）
- **内层循环**：遍历 Q 的 block（Sq 轴，若 Q 块大于 UB 容量）
- **Block 大小选择**：由 L1/L0/UB 容量共同约束

### 2.3 数据预加载（L1 Preload）
- L1 作为数据中转站，在 Cube 计算当前 block 时，预加载下一个 block
- **3-buffer 策略**：buffer[0] 正在被 Cube 使用，buffer[1] 刚加载完毕待用，buffer[2] 正在从 HBM 加载
- 预加载是 FA 性能的关键优化点

### 2.4 CrossCore 共享 UB
- Cube 输出（L0C）→ UB → Vector 读取：**CrossCore 共享 buffer**
- 需要严格的同步保护（SetFlag/WaitFlag）
- Ping-Pong 双缓冲避免读写冲突

### 2.5 Workspace 使用场景
1. **确定性计算**：各核将梯度写入 Workspace 独立区域，最后单核归约
2. **跨核通信**：反向算子中 dK/dV 的多核累加
3. **中间结果暂存**：LSE（log-sum-exp）等需要跨 forward/backward 传递的数据

## 3. 精度设计原则

### 3.1 Online Softmax 精度
- m(max) 和 l(sum) 统计量 **必须 FP32**
- exp 计算前必须减去 max，防止上溢
- rescale 因子 exp(m_old - m_new) 必须 FP32

### 3.2 BMM 精度
- Cube 支持 FP16/BF16 输入，自动升精到 FP32 计算，输出 FP32
- BMM2 输入 P 需从 FP32 降到 FP16/BF16（Cube 输入要求），损失精度
- 最终输出从 FP32 降到目标 dtype

### 3.3 累加精度
- 多块累加（Online Update）的 rescale 必须 FP32
- 反向的 dQ 多核累加必须 FP32

## 4. 同步设计原则

### 4.1 事件设计规则
1. 每个 AIC→AIV 数据传递需要一对 SetFlag/WaitFlag
2. 每个 AIV→AIC 数据传递需要一对 SetFlag/WaitFlag
3. Ping-Pong 使用两套事件，交替使用
4. AIV1 事件 ID = AIV0 事件 ID + 16

### 4.2 死锁预防
1. 单向依赖：不允许循环等待
2. 事件消费完全：每个 SetFlag 必须有且仅有一个对应的 WaitFlag
3. Ping-Pong 隔离：上一轮事件在本轮使用前必须已被消费

## 5. Mask 处理原则

### 5.1 Mask 类型优先级
```
attenMask (custom tensor) > sparseIndices > causalMask > bandMask > paddingMask
```

### 5.2 Mask 应用位置
- 在 BMM1 输出的 S 矩阵上应用 Mask
- 在 Scale (÷√d) 之后、Softmax 之前
- Mask 值：被 mask 的位置设为 -inf（或 -10000）

### 5.3 Sparse Mask 特殊处理
- sparseIndices 决定哪些 K/V block 参与计算
- 未命中的 block 直接跳过（不搬运、不计算）
- 可显著减少计算量和访存量

## 6. 性能优化原则

### 6.1 优化优先级
1. **L1 Preload**：最重要，隐藏 HBM→L1 延迟
2. **Ping-Pong 流水**：Cube 和 Vector 全重叠
3. **Tile Size 优化**：在 L1/L0/UB 约束下最大化 tile
4. **Cast 最小化**：减少不必要的精度转换
5. **Mask 跳过**：稀疏场景下跳过无效 block

### 6.2 Bubble 最小化
- 目标：Cube 和 Vector 执行时间接近，最小化空闲等待
- 若 Cube >> Vector：增大 Sq/Sk 让 Vector 有更多工作
- 若 Vector >> Cube：拆分 Vector 阶段或融合操作
