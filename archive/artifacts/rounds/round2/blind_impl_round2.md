# 第二轮盲实现：多缓冲软流水（搬运-计算重叠）

基线：第一轮 6fc58de = 双向两级分块、单缓冲、无流水，1590us（精度PASS）。
原始基线：2391us。
目标：≈605us。

## 改动摘要（照 design_doc_round2.md §6 最小改动边界）

### flash_attention_cube.h
- InitBuffers：K/P 合并为 `TQue<A1,3> kpQueL1_`（单槽 max(BLOCK_N*dim, BLOCK_M*BLOCK_N)），
  V → `TQue<A1,2> vQueL1_`，L0C `TQue<CO1,1>` → `TQue<CO1,2>`。删 kvBufL1_/vBufL1_/pBufL1_。
- ComputeMM1：K-load 改 kpQueL1_ Alloc/EnQue/DeQue，块尾 FreeTensor；删所有手工
  `MTE1_MTE2/MTE2_MTE1/M_MTE1/MTE1_M/M_FIX/FIX_MTE2`；保留每个 Mmad 前 `PipeBarrier<PIPE_M>`。
  MM1 K-load 仍用裸 LoadData2D 方案B（不引入转置 helper）。
- ComputeMM2：P 用 kpQueL1_、V 用 vQueL1_ 的 Alloc/EnQue/DeQue/Free；删手工跨流水 flag；保留 PIPE_M。

### flash_attention_vec.h
- outputQue1_ 深度 1→2。
- Vec1 入口 `vecDealM==0` 分支：先 ConsumerAcquire(S)（已在入口），补 `pQueue_.ProducerReleaseMte3()` 再 return。

### flash_attention_kernel.h
- Process 内层循环改 PRELAUNCH 错位发射：上界 `s2OuterBlocks+PRELAUNCH`，
  t>=PRELAUNCH 先 MM2(t-PRELAUNCH)/Vec2(o)，t<s2OuterBlocks 再 MM1(t)/Vec1(t)。

### 未改：tiling、kernel_common、workspace_queue、matmul_tile。

## 验证事件

### 事件1 编译 + 基础用例精度（2026-06-06）
- `rm -rf build` 后 `evaluate_ascendc.sh flash_attention basic`，TILE_FWK_DEVICE_ID=1。
- 编译通过，无死锁。
- basic[0] 4,32,4,1024,1024,128,float16：**PASS**  REL 9.77e-04/1.84e-05  ABS 4.88e-04/9.19e-06  mismatch 0.00%。
- REL 与第一轮同量级（流水不改算法）。

### 事件2 性能（2026-06-06）
- `evaluate_performance.sh flash_attention`，profiler 模式，预热5/测试10。
- base mean=2354us；**ascendc mean=990.128us（min 988.8 / max 992.4）精度 PASS，加速比 2.38x**。
- tilelang ERROR（无 tilelang 模块，与本轮无关）。

## 结果对比
| 版本 | mean(us) | 相对原始2391 | 相对第一轮1590 |
|:--|:--|:--|:--|
| 原始基线 | 2391 | 1.0x | — |
| 第一轮(6fc58de) | 1590 | 1.49x | 1.0x |
| **本轮(流水+多缓冲)** | **990** | **~2.4x** | **~1.61x** |
| 目标(main) | 605 | 3.95x | 2.63x |

本轮在第一轮 1.49x 基础上乘性叠加约 1.61x，达原始基线 2.4x。
未达 605us 目标（达成约 61%），但方向正确、精度无损、无死锁。


---

## 主控 double check (round2)
- 精度复核: PASS — REL 9.77e-04/1.84e-05, mismatch 0.00% (与 subagent 一致, 与第一轮同量级)
- 性能复核: ascendc mean=977.269us, base=2357.732us
  - 相对原始基线 2391us: ~2.41x
  - 相对第一轮 1590us: ~1.63x (与 subagent 自测 990us/1.61x 在波动内一致)
  - 距 main 605us 目标: 达成约 62%
- 无死锁、无溢出, 一次编过一次精度过。
- 结论: 第二轮流水+多缓冲在第一轮双向两级(1.49x)基础上乘性叠加 ~1.63x, 验证了
  "搬运-计算重叠"与"两级分块省搬运"正交可叠加, 推翻历史"提速全靠流水"的单因论。
- 仍有差距(977 vs 605)的可能原因: prologue/epilogue 占比(s2OuterBlocks=2 偏小, steady 段短),
  及未引入 main 的 MM1 转置加载路径(本轮为隔离变量刻意保留方案B裸LoadData2D)。
- tilelang ERROR 为环境缺模块, 与本轮无关。
