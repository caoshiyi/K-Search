# Attention 设计检查表

## 输入与约束

- [ ] proto.yaml 已读
- [ ] model.py 已读
- [ ] basic/general case 已读
- [ ] AttentionPatternIndex.md 已读，已判断哪些 pattern 相关/无关
- [ ] 相关 pattern 的 `先读这个` 部分已读
- [ ] 每个命中 pattern 的职责边界已记录（影响什么、不决定什么）
- [ ] 相关 tile-level / TileLang 参考已判定：存在且适用则已读，不存在或不适用则已说明
- [ ] 相关 FA AscendC baseline 已判定：存在且适用则已读，不存在或不适用则已说明
- [ ] 输入布局已判定
- [ ] Q/K/V 连续性已判定
- [ ] Hq/Hkv 已判定
- [ ] mask 语义已判定
- [ ] sink / sparse / topk 已判定
- [ ] MLA / Paged / TND 已判定
- [ ] dtype 已判定
- [ ] stride / offset 已判定
- [ ] workspace / FD / Vec0 需求已判定

## 分析结论

- [ ] 主路径已明确
- [ ] 需要展开的章节已列明
- [ ] 不需要的章节已说明原因
- [ ] 特殊分支触发条件已写清

## 详设正文

### 阶段
- [ ] 每个阶段有 C/V 编号和名称（V0→C1→V1→C2→V2）
- [ ] 每个阶段有核类型
- [ ] 每个阶段有输入来源
- [ ] 每个阶段有输出去向
- [ ] 每个阶段有所属 buffer
- [ ] 每个阶段有依赖关系

### 分核
- [ ] 切分轴已写
- [ ] 每核处理量公式已写
- [ ] 余量分配已写
- [ ] 小张量保护已写
- [ ] FD / 归约需求已写

### 存储
- [ ] 基本块BASE_M/N/K大小已写
- [ ] L1 / L0 / UB / workspace 分配已写
- [ ] 每个 buffer 字节公式已写
- [ ] 每个 buffer 份数理由已写
- [ ] 常驻 / 双缓冲 / 轮转 已区分
- [ ] 容量校验已写

### Cube 操作参数（仅在有特殊设计决策时检查）
- [ ] 多段输入（prefix+tail）的合并策略已写（分段 Mmad 还是拼接）
- [ ] 非标准的 cmatrixInitVal 使用已写（如累加模式）
- [ ] 非标准的 L1→L0 搬运模式已写

### 各阶段数据流（对应 2.4）
- [ ] 阶段命名使用 C/V 编号（V0→C1→V1→C2→V2）
- [ ] 每个阶段的搬运方式和地址公式已写
- [ ] 数据加载阶段写到搬运指令粒度（DataCopy/LoadNdGmToNzL1/LoadData）
- [ ] AIV 任务划分已写（vid=0/1 各处理什么）
- [ ] 无效数据的处理方式已写（zero-fill/条件跳过/mask 注入）
- [ ] 每个阶段的核内同步已写（SetWaitFlag 跟在阶段后面）
- [ ] 每个阶段的跨核同步触发点已写（ProducerRelease/ConsumerAcquire 位置）

### 流水
- [ ] 一轮计算的阶段划分已写（如 C1→V1→C2→V2）
- [ ] 每个阶段的执行单元已写（AIC/AIV）
- [ ] 阶段间数据依赖已写

### 同步（跨核在 2.5，核内跟在 2.4 各阶段后）
- [ ] WorkspaceQueue 表格已写（queue 名/slotSize 公式/notifyId/生产者/消费者/语义）
- [ ] 未使用裸 CrossCoreSetFlag/WaitFlag
- [ ] 信号在流水中的具体位置已写（哪个步骤之后 ProducerRelease）
- [ ] 核内同步跟在对应阶段后面写（未单独拎出）
- [ ] HardEvent 方向正确（源_目标，如 MTE2_V 表示 MTE2→Vector）

### 尾块与特殊分支
- [ ] 尾块处理层级已写
- [ ] padding 位置已写
- [ ] 外部 shape 是否保持原样已写
- [ ] sink / page / sparse / GQA / MLA / mask 已写
- [ ] 不适用项已写

### 契约
- [ ] shape / stride / offset / index / attr 已写
- [ ] tiling 字段已写
- [ ] workspace 大小已写
- [ ] dtype 分发已写
- [ ] 输出 shape 约束已写

### 性能
- [ ] 主性能瓶颈已写
- [ ] 双缓冲 / 常驻 / 预加载收益已写
- [ ] 基础性能风险点已写

## 反例检查
- [ ] 没有把 pattern 当结论替代品
- [ ] 没有省略不适用项
- [ ] 没有用"与基座一致"代替具体结论
- [ ] 没有出现 TileLang 代码
- [ ] 没有堆砌流水账式的 AscendC API 调用序列
- [ ] 没有超过 5 行的连续 AscendC 代码
- [ ] 设计文档正文在 300-500 行以内

## basic_case 验证
- [ ] 用 basic_case 参数走了一遍完整流程
- [ ] 每个阶段的 tensor 形状已推演
- [ ] buffer 占用不超限已验证
- [ ] 同步点数量和顺序已验证
