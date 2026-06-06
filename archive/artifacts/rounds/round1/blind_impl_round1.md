# 盲实现 Round1: QKV 两级分块 + L1 大块复用

分支: exp/blind-design-impl (kernel = 66a35f8 基线)
设计依据: current_task/artifacts/design_doc.md
基准 shape: 4,32,4,1024,1024,128,float16 ; 基线性能 2391us

## 实现决策(基于设计文档自行推导补齐处)
- Q 在 L1 用**单段**存放, dstNzC0Stride = qRowsAlign(本 shape=512), 取代设计 3.1 的 256 双段方案
  —— 微块寻址用 startIndex=mStart/C0+i + srcStride=qC0Stride/C0 即可,更简单且等效。
- nL1Size=BLOCK_N=128, KL1_SPLIT=128 —— K/V 的 GM->L1 加载与基线**逐字节一致**, 规避设计 4.1
  的 L0B 转置风险(approach B)。两级复用收益来自 mBaseSize=512 把 4 个 Q 微块并到一个任务,
  使每个 K/V-128块每 head 只搬 2 次(=qOuterBlocks)而非 8 次, 仍达 4x 搬运下降。
- mBaseSize=512, s2BaseSize=512(由 Select 函数动态选), qOuterBlocks=2, s2OuterBlocks=2。
- MM2 在一个 s2 块内沿 kv 微块(128)累加 P·V(cmatrixInitVal 仅首个 kStart)。
- Vec: qRows(512) 按 16 对齐二分到两个 AIV subblock(各 256 行), 再按 vec1ChunkRows_=16 行分块;
  删除手写 -inf mask, 改用 softmax oriSrcK=kvRows。

## 事件记录

### [基础用例验证] evaluate_ascendc.sh flash_attention basic
- 清空 build 后全新编译, 一次编过(无修复迭代)。
- 结果: PASS  REL 9.77e-04/1.84e-05  ABS 4.88e-04/9.19e-06  mismatch 0.00%

### [性能采集] evaluate_performance.sh flash_attention
- [base(torch)] mean=2391.5us
- [ascendc]     mean=1572.1us  min=1566.9  max=1575.9  加速比 1.52x  精度 PASS
- [tilelang]    ERROR: ModuleNotFoundError 'tilelang'(与本轮无关, 环境缺包)

### 关键观察(与设计文档预期偏差)
- 设计文档 §0.2/§11.2 预测「两级分块+L1复用、不带流水」收益接近 0(~1.0x)。
- 实测 1572us, 1.52x, 明显优于文档预期。推测: K/V GM->L1 重复搬运 8x->2x;
  MM2 沿 reduce 维在 L0C 累加(每个 dim n-tile 一次 Fixpipe)减少 Fixpipe/往返;
  mBaseSize=512 把任务数从 256 降到 64, 降低同步/切换开销。串行下亦有实测收益。
- 无内存越界/同步卡死, 一次通过。

---

## 主控 double check (round1)
- 精度复核: PASS — REL 9.77e-04/1.84e-05, ABS 4.88e-04/9.19e-06, mismatch 0.00% (与 subagent 一致)
- 性能复核: ascendc mean=1590.964us (min 1588 / max 1595), base=2377us, 加速比 1.49x (与 subagent 1572us/1.52x 在测量波动内一致)
- 结论: 第一轮盲实现两级分块已产生明确性能收益 (2391us 基线 → 1590us, ~1.49x)。
- 重要修正: Q+KV 双向真两级 (K/V 各只搬一次) 本身带来 ~1.5x; 此前 "两级分块单独无收益" 的结论来自不完整实验 (仅放大 Q、KV 仍 128、V 未省搬运)。
- tilelang ERROR 为环境缺 tilelang 模块, 与本轮 AscendC 改动无关。
