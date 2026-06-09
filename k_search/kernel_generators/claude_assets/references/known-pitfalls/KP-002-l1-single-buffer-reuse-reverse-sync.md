# KP-002：单缓冲 L1 复用必须补齐 MTE1→MTE2 反向同步

- 状态：adopted（已在 K-Search flash_attention round1 调试复现）
- 适用阶段：AscendC / AIC / Cube
- 主要受益 agent：designer / codegen / bug-fixer / reviewer

## 适用场景

使用 `TBuf<A1>` 或等价裸 L1 buffer 作为单缓冲，并在循环中反复把新的
K / P / V / workspace tile 通过 MTE2 覆盖到同一个 L1 地址；随后又通过 MTE1
从该 L1 tile 加载到 L0A / L0B 参与 `Mmad`。

典型结构：

- `LoadNdGmToNzL1(kvL1, ...)` 后用 `LoadData` / `LoadNzL1ToZnL0B` 从同一 `kvL1`
  多次搬到 L0B。
- 下一轮循环继续调用 `LoadNdGmToNzL1(kvL1, ...)` 覆盖同一个 `kvBufL1_`。
- P / V 也用同一个 `pBufL1_` / `vBufL1_` 在 reduce 分段之间复用。

若改用 `TQue<A1, depth>` 管理 L1 ping-pong，则应通过 `AllocTensor` / `EnQue` /
`DeQue` / `FreeTensor` 的队列语义表达同一生命周期；不要再混用裸 buffer 手工
flag，除非设计明确说明。

## 典型现象

- 编译通过、无越界、无死锁，但精度失败。
- KP-001 一类写回行偏移问题修复后，仍有中等比例 mismatch；错误分布像数据被污染，
  不是所有元素整体偏移一个常数。
- basic shape 下可能从约 50% mismatch 降到约 10%~20% mismatch 后仍 fail。
- 对照代码能看到只有 `MTE2_MTE1`（GM->L1 完成后允许 L1->L0），但没有
  `MTE1_MTE2`（上一轮 L1->L0 读取完成后允许下一轮 GM->L1 覆盖）。

## 根本原因

`MTE2_MTE1` 只保护“新 tile 已经搬进 L1，MTE1 可以读”。它不保护反方向的生命周期：
“上一 tile 的 L1->L0 读取已经结束，MTE2 可以覆盖同一块 L1”。

当 K / P / V L1 buffer 是单缓冲时，下一轮 `LoadNdGmToNzL1` 可能提前覆盖 L1，
而上一轮发起的 `LoadData` / `LoadNzL1ToZzL0A` / `LoadNzL1ToZnL0B` 仍在从同一 L1
地址读数据，形成读写竞争。结果通常表现为 BMM1 / BMM2 的矩阵乘输入被污染，
后续 softmax 或 O 累加再把错误扩散。

## 修复 / 预防清单

1. **画出每个 L1 buffer 的生命周期**：GM/workspace -> L1 是 MTE2，L1 -> L0A/L0B
   是 MTE1。凡是同一个 L1 buffer 会被下一轮覆盖，都必须标出“最后一次 MTE1 读”
   和“下一次 MTE2 写”之间的依赖。
2. **裸 `TBuf<A1>` 单缓冲必须补反向 wait**：覆盖同一个 L1 buffer 前，使用
   `SetWaitFlag<HardEvent::MTE1_MTE2>()` 或等价同步，确保上一 tile 的 L1->L0 读已完成。
3. **成对检查正向和反向同步**：
   - `MTE2_MTE1`：本轮 GM/workspace -> L1 完成后，允许 MTE1 读。
   - `MTE1_MTE2`：上一轮 L1 -> L0 读完成后，允许下一轮 MTE2 覆盖。
4. **TQue 多缓冲改造时用队列语义替代手工 wait**：生产端 `AllocTensor` 必须等槽位上次
   MTE1 消费完成，消费端必须在该 tile 所有 L1->L0 读发起后才 `FreeTensor`。不要提前
   `FreeTensor`，否则等价于提前允许覆盖。
5. **定位口诀**：若单缓冲 L1 复用路径有 `MTE2_MTE1` 没有 `MTE1_MTE2`，优先怀疑
   “L1 被下一块提前覆盖”。

## 反例 / 不适用

- L1 buffer 只加载一次且全任务常驻只读，不会被下一轮覆盖，例如某些 Q 常驻 L1 路径。
- 每轮使用不同 L1 地址且容量证明不会复用同一槽位。
- 使用正确配对的 `TQue` ping-pong / multi-buffer，且 `FreeTensor` 发生在该 tile 的所有
  L1->L0 读取之后。

## 证据来源

- K-Search flash_attention round1（2026-06-08）：s1 两级 tiling 已在 worktree 中，但
  `ComputeMM1` 的 `kvBufL1_` 和 `ComputeMM2` 的 `pBufL1_` / `vBufL1_` 是单缓冲复用。
  代码只等待 `MTE2_MTE1`，没有等待 `MTE1_MTE2`，导致下一块 K/P/V 覆盖仍被 MTE1
  读取的 L1 buffer。
- 实测修复链路：
  - 未修：basic mismatch 57.04%。
  - 仅修 KP-001 写回行偏移：mismatch 降到 14.24%，仍 FAIL。
  - 再补 `MTE1_MTE2` 反向同步：basic PASS，mismatch 0.00%，REL max/mean
    9.77e-04 / 1.84e-05。
