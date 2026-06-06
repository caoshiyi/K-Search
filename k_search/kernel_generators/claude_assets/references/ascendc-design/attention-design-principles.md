# Attention / FA 设计原则

这份文档不是术语解释，也不是短句提纲。它的作用是：把 attention 详设阶段必须遵守的设计判断方式固定下来。

## 1. Attention 的核心目标

attention 详设的首要目标不是“把计算写出来”，而是**把 HBM 访问和核间等待压到最小**。因此，任何设计都要先回答两个问题：

1. 哪些数据可以在片上重复使用，而不必每轮从 GM 重新搬？
2. 哪些阶段必须并行重叠，不能串行等待？

### 1.1 设计判断的优先级

按以下顺序判断：

| 优先级 | 先问什么 | 为什么先问 |
|---|---|---|
| 1 | 主路径阶段是什么 | 没有阶段拆解，就无法判断 buffer 和同步 |
| 2 | 哪些 stage 属于 AIC / AIV | 决定是否需要 Cube-Vector 协同 |
| 3 | 哪些数据需要常驻 / 轮转 / 双缓冲 | 决定 L1/L0/UB 的分配 |
| 4 | 哪些地方需要同步 | 决定流水是否能重叠 |
| 5 | 哪些特殊分支必须展开 | 决定正文是否要单独写分支章节 |
| 6 | 哪些地方影响精度 | 决定 online softmax、mask 和 update 的写法 |

## 2. Cube / Vector 的职责边界

attention 的设计必须先把 Cube 和 Vector 的职责分开，再把它们接起来。

| 计算类型 | 通常执行单元 | 详设要写的内容 |
|---|---|---|
| 矩阵乘法 | AIC | BMM1 / BMM2 / 辅助 matmul、L1/L0 数据流、Fixpipe 写回 |
| 逐元素操作 | AIV | Scale、Mask、Exp、Reduce、Update、CopyOut |
| 地址 / 索引预处理 | AIV 或 Host | sparse / page / topk / sink 的地址映射 |
| 跨核调度 | Host 或 Kernel | 分核策略、负载均衡、FD 归约 |

### 2.1 设计时的硬规则

- 不要把矩阵乘和逐元素处理混成一句“融合计算”。
- 不要只写“Cube 计算”，而不写 Vector 负责什么。
- 不要只写“Vector 后处理”，而不写它处理的张量来自哪里。
- 如果 AIC / AIV 的职责不清，后面的 buffer 和同步一定会错。

## 3. 分块与 tiling 的原则

attention 的 tiling 不是“把张量切成固定块”，而是**按数据复用和硬件容量切**。

### 3.1 分块要同时满足三件事

1. 片上容量能装下当前 tile。
2. 当前 tile 的数据可以被尽量复用。
3. 当前 tile 的切分不会把同步和尾块复杂度推到无法接受的程度。

### 3.2 写详设时必须给出的分块结论

- 切分轴：例如 S / T / B×N / 稀疏块
- 主块大小：每轮搬运和计算的基础单位
- 尾块大小：最后一块怎么收口
- 轮转关系：哪些块会在多个 loop 中复用
- 约束来源：是 L1、L0、UB 还是 workspace 限制

### 3.3 不允许的写法

- 只写“按块切分”
- 只写“按硬件容量分配”
- 只写“考虑尾块”
- 不写具体公式、不写实际尺寸、不写最终约束

## 4. 数据流与 buffer 的原则

attention 的 buffer 设计要从“数据流”出发，不是从“我有几个 buffer”出发。

### 4.1 先决定数据流，再决定 buffer

顺序必须是：
1. 阶段表
2. 每段输入输出
3. 每段数据流到哪个层级
4. 再决定 buffer 名称、大小和份数

### 4.2 每个 buffer 必须回答的问题

| 问题 | 例子 |
|---|---|
| 它存什么 | Q / K / V / score / prob / update / softmax state |
| 它在哪一层 | L1 / L0A / L0B / L0C / UB / workspace |
| 它是常驻还是轮转 | sink 常驻 / ping-pong / 3-buffer |
| 它是生产者还是消费者 | AIC 写、AIV 读 |
| 它要不要双缓冲 | 是否用于重叠搬运和计算 |
| 它的大小怎么算 | 维度 × dtype 字节数 × 份数 |

### 4.3 常见 buffer 类型的作用

- L1：重用数据的中转站，通常承接预加载和复用数据。
- L0A / L0B：Cube 左右操作数。
- L0C：Cube 结果和 Fixpipe 过渡。
- UB：Vector 中间结果、softmax 状态、update 和 copyout。
- workspace：跨核共享的中间结果或合并结果。

## 5. 同步的原则

attention 的同步不是”哪里写个 flag”，而是**把跨核数据依赖显式化**。

WorkspaceQueue 是封装好的跨核同步工具类；如项目内存在 FA baseline，可参考其中的 `workspace_queue.h` 等价实现。它内部管理 ring buffer + CrossCoreSetFlag/WaitFlag。

**详设中不许写裸 API**，只写 WorkspaceQueue 的使用方式：
- `Init(workspace, slotSize, notifyId)` — 初始化
- `ProducerAcquire()` — 获取可写槽位
- `ProducerReleaseFix()/ProducerReleaseMte3()` — 释放槽位
- `ConsumerAcquire()` — 等待并获取可读槽位

### 详设里必须写清的同步信息

- 谁发信号（哪个 WorkspaceQueue）
- 谁等信号（哪个 WorkspaceQueue）
- 这个信号保护了什么数据
- 这个同步是否会造成死锁风险

### 常见误区

- 只写”使用同步”
- 写裸 CrossCoreSetFlag/WaitFlag 而不是 WorkspaceQueue
- 只写发送方，不写接收方
- 不写数据依赖关系

### 核内同步（SetWaitFlag）的原则

核内同步用于同一 AI Core 内不同硬件管线（MTE2/MTE1/M/FIX/V/MTE3）之间的数据依赖。与跨核同步（WorkspaceQueue）不同，核内同步不涉及 GM 数据交换，只保证管线内指令顺序。

**必须写清的内容**：
- 每个 SetWaitFlag<HardEvent::XXX> 的位置和语义
- 这个同步保护了什么数据依赖（如”workspace_kv 搬入 L1 完成后才能 LoadData 到 L0”）
- 按执行顺序列出，形成同步链

常见 HardEvent 对及语义：

| HardEvent | 语义 | 典型场景 |
|---|---|---|
| MTE2_MTE1 | GM→L1 搬完后才能 L1→L0 | workspace_kv 搬入 L1 后才能 LoadData |
| M_MTE1 | Cube 空闲后才能 L1→L0 | 上一轮 Mmad 结束后才能搬下一轮 L0 |
| MTE1_M | L1→L0 搬完后才能 Mmad | LoadData 完成后才能启动 Cube |
| M_FIX | Mmad 完成后才能 Fixpipe | L0C 写完后才能搬出 |
| FIX_MTE2 | Fixpipe 完成后才能下一轮 MTE2 | workspace 写完后才能读下一轮 |
| MTE2_V | GM→UB 搬完后才能 Vector | workspace 读到 UB 后才能计算 |
| V_MTE3 | Vector 完成后才能 UB→GM | 计算结果写完后才能搬出 |
| MTE3_V | UB→GM 完成后才能继续 Vector | 搬出完成后才能做下一轮计算 |
| MTE3_MTE2 | UB→GM 完成后才能 GM→UB | 上一轮写 workspace 完成后才能读下一轮 |

**常见误区**：
- 只写跨核同步（WorkspaceQueue），不写核内同步（SetWaitFlag）
- 混淆 HardEvent 方向（如 MTE2_V 和 V_MTE3 是不同方向）
- 遗漏 SetWaitFlag 导致数据竞争

### WorkspaceQueue 与多子数据的关系

一个阶段可能涉及多个子数据（如 K 和 K_rope 都需要 gather），它们**共享同一个 WorkspaceQueue**，slotSize 为所有子数据的总和，数据拼接写入同一个 slot，全部完成后统一发一次信号。不要为每个子数据创建独立的 WorkspaceQueue。

示例：Gather 阶段需要 gather K [BLOCK_N, D] 和 K_rope [BLOCK_N, D_rope]
- slotSize = BLOCK_N * (D + D_rope)
- 正确：gather K 写入 slot[0 : BLOCK_N*D]，gather K_rope 写入 slot[BLOCK_N*D : ]，然后 release（一次信号）
- 错误：为 K 和 K_rope 各建一个 WorkspaceQueue（两次信号）

## 6. 在线 softmax 的精度原则

attention 的精度链不是附属内容，而是设计核心。

### 6.1 必须写清的精度路径

| 步骤 | 常见 dtype 路径 | 设计要求 |
|---|---|---|
| BMM1 | fp16/bf16 -> fp32 | 要写清升精度发生在哪里 |
| Scale | fp32 | 不能随意降精度 |
| Mask | fp32 | 要写清 mask 注入位置 |
| Softmax | fp32 | 要写清 max / sum / exp 的更新方式 |
| BMM2 | fp32 -> fp16/bf16 | 要写清何时降精度 |
| Update | fp32 | 要写清 rescale 和累加 |
| Output | fp32 -> 目标 dtype | 要写清输出格式 |

### 6.2 设计时必须说明的风险

- 哪些地方会有溢出风险
- 哪些地方会有精度丢失
- 哪些地方必须保持 fp32
- 哪些地方可以接受降精度

## 7. Mask / sparse / page / sink 的原则

### 7.1 先决定它是”改路径”还是”改值”

- 改路径：例如 sparse / page / sink 影响哪些块要搬、哪些块直接跳过
- 改值：例如 causal / padding / custom mask 只改变 score visibility

### 7.2 详设里必须分开写

- 路径类分支：地址映射、块选择、跳过逻辑
- 值类分支：mask 注入位置、-inf / 大负值策略、softmax 前后顺序

### 7.3 不允许的写法

- 只写”支持 mask”
- 只写”支持 sparse”
- 不写触发条件
- 不写具体 stage

### 7.4 数据加载阶段的实现要求

数据加载阶段（K/V 从 GM 到片上存储）是算子差异化的关键阶段。根据算子特征，可能是以下之一或组合：
- **Gather**：sparse/PA 场景，逐 token 从离散地址搬运
- **Dense page load**：连续 PA 场景，按 tile 批量搬运
- **Sink 复制**：sink 场景，固定前缀 K/V 常驻
- **Dense mask 构建**：显式 mask 场景，从 GM 读取 mask tile

无论哪种，不允许只写”加载 K/V 到 L1/UB”。必须写清：

1. **搬运指令和循环结构**：
   - 用什么指令：`DataCopy`（ND）、`LoadNdGmToNzL1`（ND→NZ）、`LoadData`（2D）
   - 循环粒度：逐行/逐块/批量，每轮处理多少数据
   - 必须明确写出循环体，不能用”类似上述”代替

2. **AIV 侧预处理的任务划分**（MIX 模式下）：
   - 如果 AIV 参与数据加载（gather/sink/mask），必须写清 vid=0 和 vid=1 各处理什么
   - 典型模式：`for bi in range(TILE / 2): process(tile_base + bi + vid * TILE/2)`

3. **无效数据的处理**：
   - zero-fill 方式：`Duplicate(dst, 0.0, size)` 还是条件跳过
   - mask 注入方式：在加载阶段记录 mask，还是在 Softmax 阶段重新判断
   - 推荐：加载阶段 zero-fill + Softmax 阶段重新判断注入 -inf（双重保护）

4. **目标存储层级**：
   - GM→L1（AIC 侧，通过 MTE2）
   - GM→UB（AIV 侧，通过 MTE2）
   - GM→workspace（AIV 侧，通过 MTE3）
   - 必须写清选择哪条路径，以及为什么

### 7.5 Cube 操作参数的写法要求

通用的 Mmad/Fixpipe/LoadData 参数（m/n/k 对应什么、srcStride 怎么算）是标准用法，不需要在详设中逐项列出。只有以下特殊情况才需要写：

- **多段输入的合并策略**：分段 Mmad 累加（cmatrixInitVal 首次 true）还是拼接后一次算（K 维度变大），以及对 L1 容量的影响
- **非标准的 cmatrixInitVal 使用**
- **非标准的 L1→L0 搬运模式**

## 8. 性能分析的原则

attention 的性能不是“尽量快”，而是**找出 bubble、带宽瓶颈和复用机会**。

### 8.1 要分析的三件事

1. Bubble：AIC / AIV 谁在等谁
2. Bandwidth：HBM / workspace / L1/L0/UB 哪一层最吃带宽
3. Reuse：哪些数据能通过常驻、预加载、双缓冲重复使用

### 8.2 设计里必须写的性能结论

- 哪个 stage 最慢
- 哪个 buffer 最紧
- 哪个同步最可能拖慢流水
- 哪个特殊分支最可能影响吞吐
- 哪些 case 用来暴露性能瓶颈

## 9. 写法约束

### 9.0 行数与代码控制

详设正文控制在 **300-500 行**。

设计文档记录的是设计决策，不是实现代码。以下写法严格禁止：
- 把 DataCopy / Mmad / Fixpipe 的完整调用展开写成流水账
- 逐行罗列 AscendC API 调用序列
- 用代码片段代替设计描述
- 每个阶段写超过 5 行的连续代码

正确写法：写"做什么、为什么、关键参数"，用表格和公式代替代码罗列。

## 10. 硬性约束

以下约束必须遵守，违反将导致设计不合格：

### 10.1 跨核同步必须使用 WorkspaceQueue 模式

**约束内容**：
- 所有跨核数据传递必须采用 WorkspaceQueue 模式封装
- WorkspaceQueue 是自定义工具类；如项目内存在 FA baseline，可参考其中的 `workspace_queue.h` 等价实现
- 内部封装：ring buffer 管理 + CrossCoreSetFlag/WaitFlag
- 严禁直接使用 CrossCore 同步传递数据

### 10.2 Softmax 必须使用 SoftmaxFlashV2

**约束内容**：
- 所有 online softmax 必须调用 AscendC 官方 API `SoftmaxFlashV2`
- 严禁手动实现 exp/reduce/rescale 组合

**SoftmaxFlashV2 使用方式**（如项目内存在 FA baseline，可参考其中的 vector 实现）：
- 先计算 tiling：`SoftMaxFlashV2TilingFunc(srcShape, sizeof(inType), sizeof(outType), tmpBufSize, isUpdate, isOutput)`
- 调用：`SoftmaxFlashV2<T, isUpdate, isOutput, ..., CFG>(dst, sum, max, src, exp, inSum, inMax, tmp, smTiling, srcShape)`
- 需传入 prev state（inSum, inMax）用于 online softmax 状态更新

### 10.3 详设中禁止出现 TileLang 代码

**约束内容**：
- 详设文档中不得出现任何 TileLang 代码或 TileLang API 调用
- 所有伪码/代码片段必须使用 AscendC 风格（如 `DataCopy`等）或纯伪码
- 参考 TileLang 设计可以，但不能把 TileLang 代码直接搬进详设
