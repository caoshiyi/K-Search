# 流水效率专项检查模块（E1-E3）

评估 Cube+Vector 流水设计的执行效率，识别潜在的性能瓶颈。

---

## 【E1】Bubble 分析（流水气泡评估）

```
问题：评估流水各阶段的耗时，分析 Bubble 比例。

Step 1: 列出各流水阶段的理论耗时

  ┌──────────┬──────────┬───────────────────────────────────┐
  │ 阶段     │ 执行核   │ 耗时估算公式                       │
  ├──────────┼──────────┼───────────────────────────────────┤
  │ C1(BMM1) │ AIC      │ 2×Sq×Sk×D / Cube_FLOPS           │
  │ V1(Soft) │ AIV      │ Sq×Sk×(N_ops) / Vec_FLOPS         │
  │ C2(BMM2) │ AIC      │ 2×Sq×Sk×D / Cube_FLOPS           │
  │ V2(Upd)  │ AIV      │ Sq×D×(N_ops) / Vec_FLOPS          │
  └──────────┴──────────┴───────────────────────────────────┘

Step 2: 绘制流水时序图（Ping-Pong 模式下至少 2 个迭代）

  iter0:  |--C1--|--wait--|--C2--|--wait--|
          |--wait--|--V1--|--wait--|--V2--|
  iter1:       |--C1--|--wait--|--C2--|--wait--|
                    |--wait--|--V1--|--wait--|--V2--|

Step 3: 计算 Bubble 比例
  Bubble_ratio = idle_time / total_time

  - AIC Bubble: V 阶段执行期间 AIC 空闲的时间
  - AIV Bubble: C 阶段执行期间 AIV 空闲的时间

  理想目标：Bubble_ratio < 20%

Step 4: 优化建议
  若 Bubble 过大：
  - Cube 耗时 >> Vector：增大 Sq 或 Sk 的 tile 大小，让 Vector 有更多工作
  - Vector 耗时 >> Cube：考虑拆分 Vector 阶段，部分计算与 Cube 重叠
  - 考虑 L1 预加载与计算重叠
```

## 【E2】Double Buffer / Preload 策略

```
问题：以下性能优化措施哪些适用？请逐项确认。

### L1 Preload（数据预加载）
  □ 启用 L1 Preload
    - 预加载时机：在 Cube 计算当前 tile 时，预加载下一个 tile 到 L1
    - Buffer 策略：N-buffer（如 3-buffer），保证预加载和计算不冲突
    - 预加载粒度：整个 block / 按 row 预加载
  □ 不启用（L1 容量不足或访存不是瓶颈时）

### L0 Double Buffer
  □ L0C Double Buffer（Cube 输出缓冲）
    - 当前结果写 L0C[0] 的同时，上一结果从 L0C[1] 搬到 UB
  □ L0A/L0B Double Buffer（Cube 输入缓冲）
  □ 不启用

### UB Double Buffer（Vector 阶段）
  □ Vector 输入/输出 Double Buffer
    - CopyIn 当前 tile 的同时，Compute 处理上一个 tile
  □ 不启用（UB 容量紧张时）

### Ping-Pong 流水
  □ 全局 Ping-Pong（奇偶迭代使用不同 buffer 集合，实现 C/V 全重叠）
    - Ping 集合：bmm1Res_ping, softmaxRes_ping, ...
    - Pong 集合：bmm1Res_pong, softmaxRes_pong, ...
  □ 不使用全局 Ping-Pong

⚠️ FA 性能关键点：
  - L1 Preload 是 FA 性能的核心优化，通常必须启用
  - Ping-Pong 是 Cube+Vector 重叠的标准做法
  - 所有 Double Buffer 都会增加内存消耗，需与 F4 的 Buffer 设计联动
```

## 【E3】带宽瓶颈评估

```
问题：评估算子的瓶颈类型（Compute Bound / Memory Bound）。

### 计算强度分析
  Arithmetic Intensity = FLOPs / Bytes_Accessed

  典型 FA 前向：
    FLOPs ≈ 4 × B × N × Sq × Sk × D  （两次 BMM + Softmax）
    Bytes ≈ B × N × (Sq×D + Sk×D + Sk×D + Sq×D) × sizeof(dtype)

  若 AI > 硬件 Compute/Memory 比值 → Compute Bound
  若 AI < 硬件 Compute/Memory 比值 → Memory Bound

### A5 硬件参数（参考值）
  - Cube 峰值算力：[查阅硬件手册]
  - Vector 峰值算力：[查阅硬件手册]
  - HBM 带宽：[查阅硬件手册]
  - L1 带宽：[查阅硬件手册]

### 瓶颈应对策略
  Memory Bound：
    - 增大 tile size 提高数据复用率
    - 启用 L1 preload 隐藏访存延迟
    - 减少不必要的数据搬运（UB 内复用）

  Compute Bound：
    - 优化 Cube 利用率（对齐 tile 到 Cube 最优尺寸）
    - 减少 Vector 阶段计算量（融合操作、减少 Cast）
    - 提高流水重叠度（减少 Bubble）
```
