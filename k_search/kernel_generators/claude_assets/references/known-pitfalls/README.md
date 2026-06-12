# 跨任务已知坑点库（known-pitfalls）

这是 K-Search 的**常驻、跨任务**坑点库，区别于每个候选项目根目录里 per-project 的 `KNOWLEDGE.md`：

- `KNOWLEDGE.md`：单个候选项目内、本轮调试沉淀，任务结束即弃。
- 本目录：经多次任务复核、根因级、值得每次任务前读取的可复用教训，常驻在 agent 资产里。

## 谁读、何时读

- `designer`：写详设前读相关条目，把坑点对应的设计契约写进 `ASCENDC_DESIGN.md`。
- `codegen`：实现前读，避免在等价改写中引入回归。
- `bug-fixer`：分类失败后读，优先匹配已知坑点缩短定位。

## 收录标准（与 knowledge-curator 一致）

1. 通用性：跨多个算子/任务复现，不是某一行的特例。
2. 非显然：报错信息本身不直接指向修复。
3. 可复用：下次遇到同类问题能加速定位。

## 索引

| ID | 标题 | 主要受益 agent |
|----|------|----------------|
| KP-001 | 分段/子块写回偏移必须用单一全局行坐标系 | designer / codegen / bug-fixer |
| KP-002 | 单缓冲 L1 复用必须补齐 MTE1→MTE2 反向同步 | designer / codegen / bug-fixer / reviewer |
| KP-003 | 多 Q outer block 任务的 softmax 状态缓存必须在每块开始时重置 | designer / codegen / reviewer / bug-fixer |
| KP-004 | Vec chunk 循环中 softmax state cache 索引必须使用全局行偏移 | codegen / reviewer / bug-fixer |
