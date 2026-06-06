# Round 1 快照索引

本目录是「双 subagent 盲实现闭环」实验**第一轮**的归档快照，防止后续轮次覆盖。

## 对应 commit
- 成果提交：`6fc58de` — *exp(blind): 第1轮盲实现QKV两级分块 1590us/1.49x (精度PASS)*
- 父提交（基线）：`66a35f8` — flash attention example support gqa（单级 128 微块）
- 分支：`exp/blind-design-impl`

## 本轮优化点
**QKV 两级分块 + L1 大块复用**（不引入 double buffer / 流水，单缓冲串行）：
- `mBaseSize=512`、`s2BaseSize=512`，外层大块 + 内层 128 微块两级调度。
- K/V 各只从 GM 搬入 L1 一次，被同一任务内的多个 Q 微块复用。
- 与旧实验「仅放大 Q 单向」不同，本轮是 **Q+KV 双向真两级**（K/V 都省搬运）。

## 性能结果
| 版本 | ascendc mean | 加速比 |
|------|------|--------|
| 基线 66a35f8（单级128） | 2391 us | ~1.0x |
| 第一轮（QKV双向两级+L1复用，无流水） | **1590 us**（主控复核） | **1.49x** |

- 精度：PASS，REL 9.77e-04 / 1.84e-05，mismatch 0.00%。
- subagent 自测 1572us/1.52x，与主控复核 1590us/1.49x 在测量波动内一致。
- 一次编过、一次精度通过，无修复迭代。

## 文件清单
| 文件 | 说明 |
|------|------|
| `design_doc.md` | 设计 subagent 产出的第一轮设计文档（约 400 行，算法/伪代码层面） |
| `blind_impl_round1.md` | 实现 subagent 的盲实现记录 + 主控 double check |
| `kernel_changes.diff` | 第一轮 kernel 代码改动（`git diff 66a35f8 6fc58de -- flash_attention/kernel/`），涉及 cube.h / kernel.h / tiling.cpp / tiling.h / vec.h |

## 相关文件（非本目录，供追溯）
- `current_task/artifacts/two_level_tiling_verify.md` — 更早的旧实验（仅 Q 单向两级，误判两级分块无收益），本轮已纠正其结论。
- `current_task/artifacts/lessons_round1.md` — 第一轮经验教训复盘。
