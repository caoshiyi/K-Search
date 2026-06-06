# 两级分块实验验证记录

## 分支
exp/two-level-tiling (从 66a35f8 切出)

## 改动
在 66a35f8 上引入「两级分块 + L1 大块复用」单一变量：
- mBaseSize=512，Q 方向大块；KV 仍 128 步进
- cube MM1/MM2 新增 mL0 微块循环，K/V 搬入 L1 一次被多微块复用
- vec ComputeVec1/Vec2 新增 mSub 子块循环，状态按 (slot,mSub) 索引
- workspace S/P/O/meta 放大到 mBaseSize 粒度
- 不引入 double buffer / TQue 流水

## 编译 + 精度（basic）
用例: 4,32,4,1024,1024,128,float16 (main shape)
- Build completed
- PASS  [REL] 9.77e-04/1.84e-05  [ABS] 4.88e-04/9.17e-06  mismatch: 0.00%
- 汇总: 1/1 PASS

精度与原 66a35f8 一致，数值正确。

## 性能（profiler, 同 main shape）
| 版本 | ascendc mean | 加速比 |
|------|------|--------|
| 66a35f8 基线 | 2391 us | 0.99x |
| 本实验(两级分块+L1复用) | 2406 us | 1.00x |
| main(deae846) | 605 us | 4.00x |

base mean=2418us, ascendc 精度 PASS (max_rel=0.000977)

## 结论（负面结果，有价值）
单纯引入「两级分块 + L1 大块复用」(不带 double buffer/TQue 流水)，
把 K/V GM 搬运降到 1/4，性能几乎不变（2406 vs 2391us，略慢 0.6%）。

- GM 搬运总量不是本 kernel 瓶颈
- 真正瓶颈：66a35f8 用同核硬同步(SetWaitFlag)，搬运与计算完全串行，
  减少搬运次数省下的时间被计算阶段的等待吃掉
- main 的 4x 提速主要来自 double buffer + TQue 流水的搬运/计算重叠，
  而非 L1 复用减少搬运量——这正是本实验刻意排除的变量
- 即「两级分块 + L1 复用」对性能的独立贡献接近 0

---

# 第二轮：KV 两级分块（整套移植 main）

## 背景
读完 main 代码后确认：main 的 KV 两级分块（s2BaseSize 大块 + vec1ChunkRows_ 行二次切分 +
大列 softmax）与 TQue 流水深度耦合，无法干净剥离。用户决策：整套移植 main（含流水）。

## 操作
git checkout deae846 -- kernel/  （kernel 整套对齐 main）
model_new_ascendc.py / model.py 与 main 一致，无需改动。

## 编译 + 精度（basic, 同 shape 4,32,4,1024,1024,128,fp16）
- PASS  [REL] 1.18e-02/8.66e-04  mismatch: 0.00%

## 性能（profiler）
| 版本 | ascendc mean | 加速比 |
|------|------|--------|
| 66a35f8 基线（单级128） | 2391 us | 0.99x |
| 第一轮 Q单向两级+L1复用（无流水） | 2406 us | 1.00x |
| 第二轮 KV两级+流水（=main） | 606 us | 3.92x |

## 最终结论（实验闭环）
三轮对比形成完整证据链：
1. Q 单向两级+L1复用（无流水）→ 无提速（搬运非瓶颈）
2. KV 两级+流水（整套 main）→ 4x 提速
=> main 的提速主源是 **TQue/double buffer 流水带来的搬运/计算重叠**，
   而非两级分块的 L1 大块复用本身。两级分块的价值在于"为流水提供足够大的
   重叠粒度"，单独存在不产生收益。
