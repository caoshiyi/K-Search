# FlashAttention 性能优化设计文档

> **本轮优化点：QKV 两级分块 + L1 大块复用**
> 本文只描述这一个优化点。流水线 / double buffer / TQue 多缓冲 / 搬运-计算重叠等其他优化**不在本轮范围**，留给后续轮次。实现者应在现有单缓冲（TBuf）结构上落地两级分块，不要引入多 buffer 流水。

---

## 0. 问题背景与优化动机

### 0.1 当前实现（基线）的结构
基线 kernel 采用**单级固定 128 微块**调度：

- 外层循环：把 Q 序列按 `BLOCK_M = 128` 行切成 `seqBlocks = ceil(qSeqLenAlign / 128)` 个 Q 块，每个核领若干 Q 块任务。
- 内层循环：对每个 Q 块，沿 KV 方向按 `BLOCK_N = 128` 步进，循环 `kvLoops = ceil(kvSeqLen / 128)` 次。
- 每次 KV 迭代里：
  - MM1 从 **GM** 加载一个 `128 × dim` 的 K tile 到 L1，算 `S = Q · Kᵀ`（128×128）。
  - MM2 从 **GM** 加载对应的 `128 × dim` 的 V tile 到 L1，算 `O_tile = P · V`。

### 0.2 性能根因
seq=1024、dim=128、shape `4,32,4,1024,1024,128,fp16` 下：
- 每个 head 有 `8` 个 Q 块（1024/128），KV 也有 8 个 tile。
- 因为外层是 Q 块、内层是 KV，**每处理一个 Q 块就把全部 K、V 从 GM 完整重搬一遍** → K、V 各被重复搬运 **8 次**。
- K/V 的 GM→L1 搬运量 = `8 × (kvSeqLen × dim)`，是理论下界（搬 1 次）的 8 倍。

### 0.3 本轮优化思路（一句话）
**把 Q 方向和 KV 方向都升级为「外层大块 + 内层 128 微块」两级分块**，让一个大的 K/V 块（`s2BaseSize` 行）一次性搬进 L1 后，被同一任务内的多个 Q 微块（共 `mBaseSize` 行）反复复用，从而把 K/V 的 GM 重复搬运次数从 `qOuterBlocks` 降到 1（每个 KV 外层块只搬一次）。

> **事后校正（来自 lessons_round1）**：旧实验 `two_level_tiling_verify.md` 的 2406us/1.0x 结论只覆盖「Q 单向放大、KV 仍 128、V 未省搬运」的范围，不能外推到本轮的 **Q+KV 双向真两级**。本轮盲实现精度 PASS，性能从 2391us 到约 1590us（1.49x），证明双向两级分块本身就是有效优化；后续流水是在此基础上继续叠加。

---

## 1. 两级分块的粒度定义与推荐取值

引入两个新的 tiling 粒度参数（区别于固定常量 `BLOCK_M = BLOCK_N = BASE_K = 128`）：

| 参数 | 含义 | 推荐取值（本 shape） | 理由 |
|------|------|------|------|
| `mBaseSize` | **Q 方向外层大块**行数：一个调度任务一次处理多少行 Q | `512`（dim≤192 时） | 一个任务覆盖更多 Q 行，使一次搬入 L1 的 K/V 被更多 Q 微块复用；512=4×128，正好 4 个 M 微块 |
| `s2BaseSize` | **KV 方向外层大块**行数：online-softmax 一次迭代覆盖多少 KV 行 | `512`（qSeqLen>16 时） | 一个 KV 外层块一次搬进 L1，被该任务全部 Q 微块复用；512=4×128 |
| `BLOCK_M` | M 方向**内层微块**（Mmad 的 m） | `128`（固定） | Cube L0A/L0C 的基本计算粒度 |
| `BLOCK_N` | N 方向**内层微块**（MM1 的 n / MM2 的 k 切分） | `128`（固定） | Cube L0B 基本计算粒度 |
| `BASE_K` | K 方向（reduce 维 = dim）**内层微块** | `128`（固定） | dim=128 恰好一个 BASE_K |

派生量（host 侧 tiling 计算，kernel 侧从 tiling 读取）：
- `qOuterBlocks = ceil(qSeqLenAlign / mBaseSize)` —— Q 外层块数。本 shape：1024/512 = 2。
- `s2OuterBlocks = ceil(kvSeqLen / s2BaseSize)` —— KV 外层块数。本 shape：1024/512 = 2。
- 任务总数 `totalGSq = groupNum × qOuterBlocks`，`totalBNkv = batch × nkv`，与基线相同的核间分配逻辑（线性切分到 `usedCoreNum` 个核，`bNkvEnd[]/gSqEnd[]`）。

### 1.1 取值约束与选择函数（host 侧）
推荐用两个 host 辅助函数动态选择，而不是写死：

```
SelectS2BaseSize(qSeqLen, groupNum):
    # s2BaseSize 上限 1024：保证 vec 侧 vec1ChunkRows = UB_QUEUE_BYTES/(s2BaseSize*4) >= 8，
    # 使 softmax state 写入地址 32B 对齐；UB 192KB 下 s2BaseSize>1024 会令 vec input queue 超限
    if qSeqLen <= 16: return 1024      # 短 Q：KV 块尽量大
    else:             return 512       # 长 Q（本 shape=1024）：512

SelectMBaseSize(qSeqLen, s2BaseSize):
    base = 512
    if qSeqLen <= 16:                  # 短 Q 时按 s2BaseSize 反比缩 mBaseSize 控 workspace
        if   s2BaseSize <= 512:  base = 512
        elif s2BaseSize <= 1024: base = 256
        elif s2BaseSize <= 2048: base = 128
        else:                    base = 64
    return max(base, BLOCK_M)          # 最小执行微块仍是 128
```

> 关键约束：`mBaseSize` 必须是 `BLOCK_M=128` 的整数倍，`s2BaseSize` 必须是 `BLOCK_N=128` 的整数倍，且 `s2BaseSize <= 1024`（vec 侧 UB 与对齐限制）。


---

## 2. 调度循环结构（kernel 顶层）

每个核遍历分给它的 `(bNkv, gSq)` 任务区间。一个任务 = 一个 `(batch, kvHead, group, qOuterIdx)`，即「某个 head 的某个 Q 外层大块」。

```
for each task (bz, kvHead, groupId, qOuterIdx) assigned to this core:
    qHead     = kvHead * groupNum + groupId
    qRowStart = qOuterIdx * mBaseSize
    qRows     = min(mBaseSize, qSeqLen - qRowStart)   # 尾块可能 < mBaseSize

    if AIC:  cube.LoadQ(bz, qHead, qRowStart, qRows)   # 整个 Q 大块一次搬进 L1
    if AIV:  vec.SetTaskContext(bz, qHead, qRowStart, qRows)

    # KV 外层大块循环（online-softmax 的迭代维）
    for t2 in 0 .. s2OuterBlocks-1:
        s2Start = t2 * s2BaseSize
        s2Rows  = min(s2BaseSize, kvSeqLen - s2Start)   # 尾块可能 < s2BaseSize

        if AIC:
            cube.ComputeMM1(bz, kvHead, s2Start, s2Rows, qRows)   # S = Q·Kᵀ (mBaseSize × s2BaseSize)
            cube.ComputeMM2(bz, kvHead, s2Start, s2Rows, qRows)   # O += P·V
        if AIV:
            vec.ComputeVec1(slot, isFirst=(t2==0), s2Rows)        # scale+mask+online-softmax
            vec.ComputeVec2(slot, isFirst=(t2==0), isLast=(t2==s2OuterBlocks-1))
```

> **本轮不做流水**：直接用「MM1→Vec1→MM2→Vec2」串行同步（沿用基线 `WorkspaceQueue` 的 `CrossCoreSetFlag/WaitFlag` + 同核 `SetWaitFlag<HardEvent>`）。`PRELAUNCH` 可保持，但不是本轮重点；如实现困难，本轮可退化为 PRELAUNCH=0 的纯串行，正确性优先。

任务内 KV 外层只循环 `s2OuterBlocks` 次（本 shape=2），而每个 KV 外层块内部再按 128 切微块——这就是「两级」。

---

## 3. L1 Buffer 布局与放大（Cube 侧核心）

910B L1 容量约 **512 KB**。本轮要把 Q 大块和 K/V 大块都放进 L1，需要重新规划 L1 分配。所有 L1 张量按 **NZ（zN）格式**存放（`LoadNdGmToNzL1` 产出的布局），dim 方向按 `C0=16`（fp16）对齐。

### 3.1 Q 常驻 L1（一个任务搬一次，被所有 KV 外层块复用）
- Q 大块 = `qRows × dim`（≤ `mBaseSize × dim` = 512×128）。
- **推荐单段连续存放**：不要再按 256 行切成双段。Q 的 L1 行容量取 `qRowsAlign = AlignUp(qRows, C0)`，NZ 写入时 `dstNzC0Stride = qRowsAlign`。
- 128 行 Q 微块寻址直接使用全局任务内行偏移：第 `mStart` 行微块的 L1 起点按 `mStart / C0` 个 C0 行块推进，`srcStride = qRowsAlign / C0`。这样 subblock、尾块和 `qRows=512` 都落在同一套寻址公式里，避免双段切换歧义。
- Q 常驻大小按实际 `qRowsAlign × dimAlign × sizeof(fp16)` 计算。本 shape 最大约 512×128×2 = **128 KB**。
- LoadQ 在**任务开始时调用一次**，整个 KV 外层循环不再重搬 Q。这是「Q 复用」。

### 3.2 K/V 大块进 L1（一个 KV 外层块搬一次，被所有 Q 微块复用）
- **默认保持 K/V L1 微块粒度为 `BLOCK_N=128`**，即 `nL1Size = BLOCK_N`、`KL1_SPLIT = BLOCK_N`。本轮收益来自 `mBaseSize=512` 把多个 Q 微块合进同一任务，让同一 K/V 微块被多个 Q 微块复用，而不是来自把 K/V 的单次 L1 搬运粒度放大到 256。
- 选择 128 的原因是与基线 K/V GM→L1 和 L1→L0 的加载形态逐字节一致，能显著降低 MM1 转置、L0B stride、尾块处理的实现风险。
- K/V 的 L1 buffer 单块 = `BLOCK_N × dimAlign × sizeof(fp16)` = 128×128×2 = **32 KB**。如果后续轮次专门抽取 256 行 K 大块加载，需要重新写 stride 契约并重新评估收益；不要在本轮混入。

### 3.3 P（softmax 概率）回搬 L1（MM2 的左矩阵）
- MM2 需要 `P (mBaseSize × s2BaseSize)` 做左矩阵，P 由 vec 侧写回 workspace（GM），MM2 再从 GM 按 `mRows × kL1Size` 搬进 L1。
- P 的 L1 块 = `BLOCK_N × KP_L1_COLS`（推荐 KP_L1_COLS=256）= 128×256×2 = **64 KB**。

### 3.4 L1 容量校验（关键，必须满足 ≤ 512KB）
单缓冲（本轮不 double buffer）峰值同时存在：
- Q 常驻：最多 128 KB（单段连续，容量按 `qRowsAlign`）。
- MM1 阶段：Q(128KB) + K 当前块(32KB) = **160 KB** ✅
- MM2 阶段：Q 已不需（可复用其空间，但保守按仍占）+ P 块(64KB) + V 块(32KB)。即使 Q 仍占 128KB，峰值 = 128+64+32 = **224 KB** ✅
- 加上 L0A/L0B/L0C（各 ≤ 128×128×2/4，KB 级，不在 L1）不影响 L1。

结论：本轮在单缓冲下 L1 峰值约 **224 KB < 512 KB**，安全。若后续加 double buffer 需重新核算。


---

## 4. MM1 两级循环（S = Q · Kᵀ）

输出 `S` 形状 `qRows × s2Rows`（≤ 512×512），写到 workspace_S 的当前 slot，列跨度（dstStride）= `s2BaseSize`（`outerCols`）。

### 4.1 可用的 matmul helper（已核实基线 `matmul_tile.h` 提供）
- `LoadNdGmToNzL1(dst, srcGm, m, n, ld[, dstNzC0Stride])` —— GM(ND)→L1(NZ)。
- `LoadNzL1ToZzL0A(dst, srcL1, m, k, colC0Stride)` —— L1(NZ)→L0A(Zz)，给 Mmad 左矩阵。
- `LoadNzL1ToZnL0B(dst, srcL1, k, n, colC0Stride)` —— L1(NZ)→L0B(Zn)，给 Mmad 右矩阵（**不转置**）。
- `FixpipeNzL0cToNdGm` / `FixpipeNzL0cToNdGmStride` —— L0C→GM。
- **缺失项（重要）**：基线 `matmul_tile.h` **没有** `LoadNzL1ToZnL0BWithTranspose`。本轮默认选择 **(B) 沿用基线 MM1 的裸 L0B 加载方式**，只把调度从单级改成两级，不新增转置 helper。
- **(A) 自补转置 helper 只作为后续独立优化项**：如果后续轮次要把 K L1 块放大到 256 并做转置感知抽取，必须另写一份完整的 stride/容量/收益设计，不能与本轮混在一起。这样能保持本轮变量隔离，也避免把高风险 stride 逻辑留给盲实现者临场选择。

### 4.2 两级循环伪代码
```
ComputeMM1(bz, kvHead, s2Start, s2Rows, qRows):
    sSlot = sQueue.ProducerAcquire()        # workspace_S 当前 slot
    outerCols = s2BaseSize                   # S 在 workspace 的列跨度
    nL1Size = BLOCK_N                       # 本轮默认 128，保持 K 加载与基线一致
    kTiles  = dim / BASE_K                    # 本 shape dim=128 → kTiles=1

    # —— 外层：K 大块沿 s2Rows 分段搬进 L1（每段被下面所有 Q 微块复用）——
    for nL1Start in 0 .. s2Rows step nL1Size:
        kvRows = min(nL1Size, s2Rows - nL1Start)
        kvRowsAlign = AlignUp(kvRows, C0)
        kOffset = ((bz*nkv + kvHead)*kvSeqLen + (s2Start + nL1Start)) * dim
        kvL1 = load K[kvRows × dim] from GM → L1(NZ), dstNzC0Stride = kvRowsAlign   # 搬一次

        # —— 中层：Q 微块（128 行）——
        for mL0Start in 0 .. qRows step BLOCK_M:          # 128 行 M 微块
            mRows = min(BLOCK_M, qRows - mL0Start)
            mRowsAlign = AlignUp(mRows, C0)

            # —— 内层：N 微块（128 列 KV）——
            for nL0Start in 0 .. kvRows step BLOCK_N:
                nRows = min(BLOCK_N, kvRows - nL0Start)
                nRowsAlign = AlignUp(nRows, C0)
                cL0 = L0C.Alloc()
                for ki in 0 .. kTiles:                 # reduce 维 dim 分 BASE_K
                    aL0 = load Q slice from qL1[mL0Start 行偏移] → L0A(Zz)   # 复用常驻 Q
                    bL0 = load K slice from kvL1 → L0B (默认沿用基线裸加载)
                    Mmad(cL0, aL0, bL0, m=mRowsAlign, n=nRowsAlign, k=BASE_K,
                         cmatrixInitVal=(ki==0))
                # dim 尾块 kRemain = dim % BASE_K（本 shape=0，可省）
                Fixpipe cL0 → sSlot[mL0Start*outerCols + nL1Start + nL0Start]
                        (mSize=mRows, nSize=nRowsAlign, srcStride=mRowsAlign, dstStride=outerCols)
    sQueue.ProducerReleaseFix()
```

要点：
- **K/V 微块在任务内只搬一次**，被该任务内所有 Q 微块复用。对本 shape 而言，每个 head 的同一 K/V-128 微块随 2 个 Q 外层任务搬入，而不是随 8 个 Q 微块搬入，这是约 4× 搬运下降的来源。
- S 直接按 `(mGlobal, nGlobal)` 偏移写进 `mBaseSize × s2BaseSize` 的大 workspace slot，vec 侧据此读。

### 4.3 单缓冲 L1 复用同步契约（必须显式实现）
本轮不引入 TQue 多缓冲，K/V/P L1 buffer 仍是单缓冲复用。因此每次覆盖同一个 L1 buffer 前，必须保证上一块已经完成所有 L1→L0 读取。

- MM1：覆盖 K L1 buffer 前，等待上一 K 微块的全部 L0B 读取完成；本 K 微块的所有 Q 微块和 K-tile 读取都发起之后，才允许下一次 GM→L1 覆盖。
- MM2：覆盖 P/V L1 buffer 前，等待上一 reduce 微块的全部 L0A/L0B 读取完成；本 reduce 微块对应的 Mmad 输入读取都发起之后，才允许下一段 P/V 覆盖。
- 这个同步只保护单缓冲 L1 生命周期，不应改变跨核 `WorkspaceQueue` 的 S/P/O 生产消费计数。实现者应把"覆盖前等待"和"读完后释放"成对检查，漏掉会读脏数据，过度等待只会退化性能。


---

## 5. MM2 两级循环（O = P · V）

输入 `P (qRows × s2Rows)` 来自 vec 写回的 workspace_P slot（列跨度 = `s2BaseSize`）。输出 `O (qRows × dim)` 写 workspace_O slot（列跨度 = `dimAlign`）。reduce 维是 KV 行（`s2Rows`）。

```
ComputeMM2(bz, kvHead, s2Start, s2Rows, qRows):
    pSlot = pQueue.ConsumerAcquire()     # 等 vec1 写好的 P
    oSlot = oQueue.ProducerAcquire()
    outerCols = s2BaseSize
    KL1_SPLIT = BLOCK_N                   # 本轮默认 128，保持 V/P 加载与基线一致

    for mL0Start in 0 .. qRows step BLOCK_M:        # Q 微块 128 行
        mRows = min(BLOCK_M, qRows - mL0Start);  mRowsAlign = AlignUp(mRows, C0)
        cL0 = L0C.Alloc();  isFirstK = true
        for kL1Start in 0 .. s2Rows step KL1_SPLIT:  # reduce 维分段
            kL1Size = min(KL1_SPLIT, s2Rows - kL1Start)
            kL1SizeAlign = AlignUp(kL1Size, C0)
            # P 段：mRows × kL1Size，从 workspace_P 搬进 L1(NZ)
            pL1 = load P[ pSlot[mL0Start*outerCols + kL1Start] ] → L1, dstNzC0Stride=mRowsAlign
            # V 段：kL1Size × dim，从 GM 搬进 L1(NZ) —— 一次搬，被该 reduce 段复用
            vOffset = ((bz*nkv + kvHead)*kvSeqLen + s2Start + kL1Start) * dim
            vL1 = load V[kL1Size × dim] from GM → L1, dstNzC0Stride = kL1SizeAlign
            for kL0Start in 0 .. kL1Size step BLOCK_N:   # reduce 微块 128
                kL0Size = min(BLOCK_N, kL1Size - kL0Start); kL0SizeAlign=AlignUp(kL0Size,C0)
                aL0 = LoadNzL1ToZzL0A(pL1[kL0Start*mRowsAlign], mRowsAlign, kL0SizeAlign, mRowsAlign)
                bL0 = LoadNzL1ToZnL0B(vL1[kL0Start*C0], kL0SizeAlign, BASE_K, kL1SizeAlign)
                Mmad(cL0, aL0, bL0, cL0, m=mRowsAlign, n=BASE_K, k=kL0SizeAlign,
                     cmatrixInitVal=isFirstK); isFirstK=false
        Fixpipe cL0 → oSlot[mL0Start*dimAlign]  (mSize=mRows, nSize=BASE_K,
                       srcStride=mRowsAlign, dstStride=dimAlign)
    oQueue.ProducerReleaseFix()
```

要点：
- MM2 的右矩阵 V 用基线已有的 `LoadNzL1ToZnL0B`（**不需要转置 helper**）。
- `P·V` 沿 reduce 维（KV 行）累加，`cmatrixInitVal` 仅第一段为 true，后续累加（`Mmad(cL0,...,cL0,...)`）。
- V 每个 reduce 段只搬一次 L1，被该任务该 Q 微块复用。

---

## 6. Vec 侧联动适配（必须改，否则结果错）

分块粒度从 128 变成 `mBaseSize × s2BaseSize` 后，vec 侧的迭代结构、workspace 粒度、softmax 状态索引都要联动改。online-softmax 沿 **KV 外层块（s2OuterBlocks）** 迭代，每个外层块一次性处理 `s2Rows` 列（最多 512）。

### 6.1 workspace slot 粒度放大
所有 per-slot workspace 从 `BLOCK_M × BLOCK_N`（128×128）放大到 `mBaseSize × s2BaseSize`：
- `wsS`（float，S 矩阵）：slot = `mBaseSize × s2BaseSize`，列跨度 `s2BaseSize`。
- `wsP`（fp16，概率）：slot = `mBaseSize × s2BaseSize`。
- `wsO`（float，MM2 输出）：slot = `mBaseSize × dimAlign`。
- `wsMeta`/`wsAccO`：见 6.4。
vec 侧 `sQueue_/pQueue_/oQueue_.Init` 的 slotSize 必须与 cube 侧一致地用 `mBaseSize × s2BaseSize`、`mBaseSize × dimAlign`。

### 6.2 行方向二次切分（M 子块）
一个任务的 `qRows`（最多 512）在两个 AIV subblock 间按行劈分（`SetMSplitInfo`：偶 subblock 取前半、奇 subblock 取后半，按 16 行对齐），每个 subblock 再按 `vec1ChunkRows_` 行分块循环处理 softmax，避免 UB 溢出。
- `vec1ChunkRows_ = VEC_QUEUE_BUF_SIZE / (s2BaseSize × sizeof(float))`，再向上对齐到 8 的倍数（保证 softmax state 写入 32B 对齐）。`VEC_QUEUE_BUF_SIZE` 推荐 32 KB。
  - 本 shape：32768/(512×4)=16 行/块 → 对齐后 16。
- **`s2BaseSize ≤ 1024` 的硬约束就来自这里**：若 s2BaseSize=2048，则 vec1ChunkRows=4 < 8，state 写入地址不再 32B 对齐 → 精度错或对齐异常。

### 6.3 ComputeVec1（scale + mask + online-softmax）
```
ComputeVec1(slot, isFirst, kvRows):     # kvRows = 本 KV 外层块实际行数 s2Rows
    sSlot = sQueue.ConsumerAcquire(); pSlot = pQueue.ProducerAcquire()
    outerCols = s2BaseSize
    for chunkRow in 0 .. vecDealM step vec1ChunkRows_:
        dealRows = min(vec1ChunkRows_, vecDealM - chunkRow)
        sUb = copy S[ sSlot[(vecStartM+chunkRow)*outerCols] ]  (dealRows × outerCols)
        Muls(sUb, smScale)
        # SoftMaxShapeInfo: srcM=dealRows, srcK=outerCols, oriSrcK=kvRows
        #   oriSrcK=kvRows(实际KV列) 让 softmax 只在有效列上归约，自动屏蔽 padding 列，
        #   省掉基线里手写的 -inf mask（基线对 128 列尾块用 maskBuf 填 -inf）。
        SoftmaxFlashV2(sUb, sum=sumCache[slot,base], max=maxCache[slot,base], sUb,
                       exp=expCache[slot,base], inSum/inMax = (isFirst? default : prevSlot),
                       smTiling, srcShape)
        pHalf = Cast(sUb → fp16);  copy pHalf → pSlot[(vecStartM+chunkRow)*outerCols]
    pQueue.ProducerReleaseMte3()
```
关键变化 vs 基线：
- 基线 srcK 固定 128 且手工 `-inf` mask 尾块；本轮 srcK=`s2BaseSize`、`oriSrcK=kvRows`，**靠 softmax 的 oriSrcK 自动处理 KV 尾块**，删掉 maskBuf 逻辑。
- softmax 的 max/sum/exp 状态按 `slot` 环形缓存，`isFirst`（首个 KV 外层块）用 -inf/0 默认值，否则读 `prevSlot` 状态做 online 合并。


### 6.4 ComputeVec2（O 的 online 重缩放 + 归一化 + 写回）
```
ComputeVec2(slot, isFirst, isLast):
    oSlot = oQueue.ConsumerAcquire()
    mChunk = (VEC_QUEUE_BUF_SIZE/dimAlign) 向下取 BRCB(8) 倍, clamp ≤255, ≤vecDealM
    for ci in 0 .. loopCount:
        dealRows, rowOffset = ...
        oNewUb = copy O_tile[ oSlot[rowOffset*dimAlign] ]   (dealRows × dimAlign)
        if not isFirst:                       # 与上一 KV 外层块累计 O 做 online 合并
            oPrevUb = copy wsAccO[prevSlot, rowOffset]       # 上轮累计
            expBrcb = Brcb(expCache[slot, rowOffset])         # 上轮→本轮的 rescale 因子
            RowMuls(oPrevUb, oPrevUb, expBrcb)
            Add(oNewUb, oNewUb, oPrevUb)
        if isLast:                            # 最后一个 KV 外层块：除以 sum 归一化后写 outGm
            sumBrcb = Brcb(sumCache[slot, rowOffset]); RowDivs(oNewUb, sumBrcb)
            FinalizeOutputChunk(oNewUb → outGm)
        else:                                 # 中间块：累计存回 wsAccO[slot]
            copy oNewUb → wsAccO[slot, rowOffset]
```
- `wsAccO`、`wsMeta` 的 slot 也按 `mBaseSize` 放大（`RING_SLOTS × mBaseSize × dimAlign`、`RING_SLOTS × mBaseSize × 3`）。
- exp/sum 状态从 6.3 的 cache 读；`accOffset` 用 `prevSlot * mBaseSize + rowOffset` 寻址（基线是 `* BLOCK_M`）。
- **state cache stride 契约**：`maxCache/sumCache/expCache` 的每个 ring slot 使用固定 `stateStride = AlignUp(mBaseSize, 32)`，而不是当前 subblock 的 `vecDealM`。subblock 二分后，cache 行号仍使用任务内全局行 `mGlobal = vecStartM + chunkRow + localRow`。读取上一 slot 时使用同一个 `mGlobal`，确保两个 AIV subblock 的状态不会互相覆盖或错位。
- **`FinalizeOutputChunk` 写回地址必须用任务内全局行 `mGlobal`（= `vecStartM + chunkRow`），不是 subblock 内局部偏移**：输出地址 = `curQRowStart_ + mGlobal`（再 `qSeqLen` clamp），而非基线的 `curBx_ * BLOCK_M`。
  - ⚠️ 高危坑（KP-001，已实测复现）：本节内 state cache 用了全局行 `mGlobal`，但若写回这里误用 subblock 内局部偏移（如循环变量 `startRow`，漏掉 `vecStartM`/`rowStart_`），第二个 AIV subblock（Q 后半行）会写到错误地址，整段错位，mismatch ~50%（实测 57%，改回全局偏移后 PASS）。
  - **单一坐标系硬约束**：本设计全程只用一个任务内全局行变量 `mGlobal` 表达 GM/workspace/state/exp-meta 的行偏移。`FinalizeOutputChunk` 的行入参应命名为 `globalRow`（不要沿用会误导的 `startRow`），且 `globalRowStart` 在 clamp 前只计算一次，不叠加两个等价偏移。

### 6.5 Q Outer Block State Cache Reset (KP-001修正版/AP-002修正版)

> **⚠️ 反模式警告 (AP-002)**：此问题已在实际实现中导致严重错误，必须严格遵循以下规范。

Vec 侧 softmax 状态缓存必须在**每个 Q outer block 任务开始时重置**。

#### 6.5.1 问题根因

Vec class 的 softmax 状态缓存（`maxCacheUb_`, `sumCacheUb_`）在跨 Q outer block 时未重置，导致：

- **触发条件**：`qSeqLen > mBaseSize`（即 `qOuterBlocks > 1`）
- **症状表现**：
  - 50% 数据 mismatch
  - 误差分布呈 bimodal 特征（前半正确、后半错误）
  - 第二个及后续 Q outer block 继承前一个 block 的错误 softmax 状态

#### 6.5.2 实现方法

**方法 A：添加 `isFirstQBlock` 参数（推荐）**

```cpp
ComputeVec1(slot, isFirst, kvRows, isFirstQBlock):
    // isFirstQBlock: true 表示这是任务的第一个 Q outer block
    // isFirst: 语义更新为"本 KV 外层块是该 Q outer block 的第一个块"
    
    if (isFirstQBlock) {
        // 重置 softmax 状态缓存到初始值
        for (int i = 0; i < stateStride; i++) {
            maxCacheUb_.SetValue(i, -INFINITY);
            sumCacheUb_.SetValue(i, 0.0f);
        }
        // 如果有 exp cache，也需要重置
        expCacheUb_.SetValue(0.0f);
    }
    
    sSlot = sQueue.ConsumerAcquire(); 
    pSlot = pQueue.ProducerAcquire()
    outerCols = s2BaseSize
    // ... 后续处理与 6.3 相同
```

**方法 B：添加 `ResetState()` 方法**

```cpp
// 在 Vec kernel class 中添加公共方法
public:
    __aicore__ inline void ResetSoftmaxState() {
        // 重置当前 slot 对应的 softmax 状态
        maxCacheUb_.SetValue(-INFINITY);
        sumCacheUb_.SetValue(0.0f);
        // 如果有 exp cache，也需要重置
        if (expCacheUb_.GetSize() > 0) {
            expCacheUb_.SetValue(0.0f);
        }
    }

// 在 kernel 顶层循环调用（见第 2 节调度循环）
for each task (bz, kvHead, groupId, qOuterIdx):
    // 任务开始时重置状态（关键！）
    if AIV:
        vec.ResetSoftmaxState()
        vec.SetTaskContext(bz, qHead, qRowStart, qRows)
    
    // ... KV 外层循环
```

#### 6.5.3 调用时机

状态重置必须在以下时机触发：

| 时机 | 条件 | 说明 |
|------|------|------|
| 任务开始时 | `isFirstQBlock == true` | 在处理第一个 KV 外层块之前 |
| 参数语义 | `isFirst` 表示 | 更新为"本 KV 外层块是该 Q outer block 的第一个块" |

#### 6.5.4 注意事项

1. **问题触发条件**：只在 `qOuterBlocks > 1` 时触发（即 `qSeqLen > mBaseSize`）
2. **安全边界**：对于 `qOuterBlocks = 1` 的情况（`qSeqLen <= mBaseSize`），重置操作是安全的但非必须
3. **重置范围**：必须覆盖所有 softmax 状态：
   - `maxCacheUb_`：设置为 `-INFINITY`
   - `sumCacheUb_`：设置为 `0.0f`
   - `expCacheUb_`（如果使用）：设置为 `0.0f` 或 `1.0f`（根据实现）

#### 6.5.5 诊断方法

如遇到以下症状，应首先检查 AP-002：

```
症状特征：
- mismatch rate ≈ 50%
- 误差分布集中在后半段
- qSeqLen > mBaseSize 的 shape 出现问题
- qSeqLen <= mBaseSize 的 shape 正常

诊断步骤：
1. 检查 ComputeVec1 是否有 isFirstQBlock 参数或 ResetSoftmaxState 调用
2. 在第一个 KV 外层块开始前打印/验证 maxCache/sumCache 值
3. 对比不同 Q outer block 的 softmax 状态初始值
```

### 6.6 UB 容量校验（192 KB）
单缓冲峰值（推荐取值，本 shape）：
- `inputQue1_` 1×32KB（或 double 时 2×32KB=64KB，本轮单缓冲取 32KB）。
- `outputQue1_` 32KB。
- `tmpBuf_`（softmax work）32KB。
- `maxCache/sumCache/expCache` 各 `RING_SLOTS × 2KB` = 3×2KB×3 = 18KB。
- `softmaxMax/SumDefault` 各 2KB。
- 合计约 32+32+32+18+4 ≈ **118 KB < 192 KB** ✅。
- 若 `s2BaseSize` 偏大导致 input/output queue 元素超 `VEC_QUEUE_BUF_SIZE/4`，必须靠 `vec1ChunkRows_` 行切分把单次处理量压到 ≤ buffer，这是 6.2 行切分的根本目的。

---

## 7. tiling 结构体新增字段（host↔kernel 契约）

在 `FlashAttentionTiling` 中**新增**以下字段（kernel 侧只读）：
- `uint32_t mBaseSize;`
- `uint32_t s2BaseSize;`
- `uint32_t qOuterBlocks;`     // = ceil(qSeqLenAlign / mBaseSize)
- `uint32_t s2OuterBlocks;`    // = ceil(kvSeqLen / s2BaseSize)

保留基线已有的 `seqBlocks/kvLoops/tailValid/...`（可不再用，但不要删，避免破坏结构体布局对齐）。建议把整个结构体字段类型从 `int32_t` 统一为 `uint32_t`，并相应地让 `CopyTiling` 按 `uint32_t` 拷贝（基线是 int32），保持 host/kernel 一致。

### 7.1 host 侧 doOpTiling 计算（新增部分）
```
s2BaseSize    = SelectS2BaseSize(qSeqLen, groupNum)     # 见 1.1
mBaseSize     = SelectMBaseSize(qSeqLen, s2BaseSize)
qOuterBlocks  = ceil(qSeqLenAlign / mBaseSize)
s2OuterBlocks = ceil(kvSeqLen   / s2BaseSize)
totalGSq      = groupNum * qOuterBlocks                  # 替换基线的 groupNum*seqBlocks
totalBNkv     = batch * nkv
usedCoreNum   = min(totalBNkv*totalGSq, MAX_CORES)
FillCoreRangeEnds(totalBNkv, totalGSq, usedCoreNum, tiling)   # 逻辑同基线
```

### 7.2 workspace 放大（host 侧 perCoreBytes）
所有 per-slot 尺寸把 `BLOCK_M/BLOCK_N` 换成 `mBaseSize/s2BaseSize`：
```
wsSSize    = RING_SLOTS * mBaseSize * s2BaseSize * sizeof(float)
wsPSize    = RING_SLOTS * mBaseSize * s2BaseSize * qElementSize
wsOSize    = RING_SLOTS * mBaseSize * dimAlign  * sizeof(float)
wsMetaSize = RING_SLOTS * mBaseSize * 3         * sizeof(float)
wsAccOSize = RING_SLOTS * mBaseSize * dimAlign  * sizeof(float)
perCoreBytes = wsS+wsP+wsO+wsMeta+wsAccO ;  totalWsBytes = perCoreBytes * usedCoreNum
```
本 shape（mBaseSize=512, s2BaseSize=512, dimAlign=128, RING_SLOTS=3, fp16）：
- wsS=3×512×512×4=3MB，wsP=3×512×512×2=1.5MB，wsO=wsAccO=3×512×128×4=768KB，wsMeta≈18KB。
- perCore ≈ 6 MB；×usedCoreNum(≤20) ≈ 120 MB —— GM workspace，远小于 device HBM，OK。


---

## 8. 数据布局速查（NZ / Mmad 视角）

- GM 中 Q/K/V/O 均为 ND（行优先），形状 `[batch, heads, seqLen, dim]`。
- `LoadNdGmToNzL1` 把 `m×n` 的 ND 块转成 NZ：C0=16 的小列块沿行堆叠，`dstNzC0Stride` = 行容量（决定每个 C0 块在 L1 的行跨度）。尾块行数 < 容量时，传 `dstNzC0Stride = AlignUp(actualRows, C0)`。
- Mmad 约定：左矩阵 L0A 为 Zz、右矩阵 L0B 为 Zn、输出 L0C 为 Nz；`m/n/k` 必须按 C0/16 对齐传入（用 `AlignUp(x, C0)`）。
- MM1：A=Q（m=Q行, k=dim），B=Kᵀ（k=dim, n=KV行），C=S（m×n）。
- MM2：A=P（m=Q行, k=KV行），B=V（k=KV行, n=dim），C=O（m×n）。

---

## 9. 正确性检查清单（实现者自检）

1. `mBaseSize % BLOCK_M == 0`、`s2BaseSize % BLOCK_N == 0`、`s2BaseSize ≤ 1024`。
2. cube 与 vec 两侧的 `sQueue_/pQueue_/oQueue_.Init` slotSize 必须用**同一套** `mBaseSize×s2BaseSize` / `mBaseSize×dimAlign`，否则 AIC/AIV 读写错位。
3. Q 尾块（`qRows < mBaseSize`）、KV 尾块（`s2Rows < s2BaseSize`）、dim 尾块（`dim % BASE_K`，本 shape 为 0）都要用 `min/AlignUp` 正确收口。
4. softmax `oriSrcK = kvRows`（实际有效 KV 列）务必正确传入，替代基线手写 `-inf` mask；否则尾块归约会把 padding 算进去。
5. online-softmax 状态 cache 的 `slot`/`prevSlot` 环形索引、`isFirst/isLast` 边界要和 KV 外层循环对齐。
6. 输出写回地址基于任务内全局行 `mGlobal`（`curQRowStart_ + mGlobal`），并用 `qSeqLen` 钳位有效行（避免写越界到 padding 区）。**禁止用 subblock 内局部偏移（循环变量 `startRow`）直接当写回行号**——subblock 二分后会少加 `vecStartM`，第二个 subblock 整段写错地址（KP-001，实测 mismatch 57%）。全程只用一个全局行变量表达所有行偏移，写回入参命名为 `globalRow`。
7. L1 峰值 ≤ 512KB、UB 峰值 ≤ 192KB（见 3.4 / 6.6）。
8. **（AP-002 新增）softmax 状态缓存在每个 Q outer block 开始时必须重置**，否则 `qOuterBlocks > 1` 时产生 50% mismatch。

---

## 10. Known Pitfalls（已知陷阱）

### AP-001: Subblock 写回偏移坐标系错误
- **问题**：`FinalizeOutputChunk` 写回 `outGm` 误用 subblock 内局部偏移而非任务内全局行 `mGlobal`
- **症状**：第二个 AIV subblock 整段写错地址，mismatch ~57%
- **修复**：全程使用单一全局行坐标系，写回入参命名为 `globalRow`
- **详见**：第 6.4 节

### AP-002: Q Outer Block State Cache Pollution（高优先级）
- **问题**：Vec class softmax 状态缓存（`maxCacheUb_`, `sumCacheUb_`）跨 Q outer block 未重置
- **触发条件**：`qSeqLen > mBaseSize`（即 `qOuterBlocks > 1`）
- **症状**：
  - 50% 数据 mismatch
  - bimodal 误差分布（前半正确、后半错误）
  - 第二个及后续 Q outer block 继承错误状态
- **修复**：
  - 方法 A：添加 `isFirstQBlock` 参数到 `ComputeVec1`
  - 方法 B：添加 `ResetSoftmaxState()` 方法在 kernel 中调用
- **详见**：第 6.5 节
- **实测严重性**：已验证，会导致严重精度问题

---

## 11. 验证方式
- 编译 + 精度：`scripts/evaluate_ascendc.sh current_task basic`（先过基础用例）。
- 基准 shape：`4,32,4,1024,1024,128,float16`。
- 精度判据：参考前序实验，REL ~1e-3、mismatch 0% 即与基线一致。

---

## 12. 待验证假设（收益预期与风险）

### 12.1 预期收益与已验证结论
- **直接效果（确定）**：K、V 的 GM→L1 重复搬运次数从 `qOuterBlocks`（基线等效 8 次）降到每个 KV 外层块 1 次。本 shape 下，`mBaseSize=512` 把 4 个 Q 微块合到一个任务，每个 K/V-128 微块每 head 只需随 2 个 Q 外层任务搬入，而不是随 8 个 Q 微块重复搬入。
- **已验证结果（lessons_round1）**：盲实现一次编过、精度 PASS，主控复核性能约 **2391us → 1590us（1.49x）**。因此"Q+KV 双向真两级"不是只有结构价值，而是独立有效的性能优化。
- **历史负面结论边界**：`two_level_tiling_verify.md` 的 2406us/1.0x 来自"仅 Q 单向放大、KV 仍 128、V 未省搬运"的不完整实验，只能作为反例边界，不能用于预测本设计。

### 12.2 根因分析（为何本轮能独立见效）
- 真实收益来自 **Q+KV 双向**：K 和 V 都在任务内被多个 Q 微块复用，GM→L1 重复搬运显著下降；仅放大 Q 而不减少 K/V 重搬不会产生同等收益。
- 本轮仍是单缓冲串行，同步等待会吃掉一部分搬运节省，所以收益约 1.5x 而不是接近理论搬运下降倍数。
- **两级分块仍是后续流水的结构前提**：第二轮可以在本轮 1590us 基线上继续通过多缓冲软流水叠加收益。后续设计必须把基线锚定到本轮结果，而不是回到 2391us 原始基线。

### 12.3 主要风险点
1. **softmax 大列归约精度**：srcK 从 128 增大到 512，`SoftMaxFlashV2TilingFunc` 的临时 buffer 与 `vec1ChunkRows_` 对齐（8 的倍数 / 32B）必须正确，否则尾块/对齐错误导致精度劣化或 alignment 异常。
2. **UB 溢出**：`s2BaseSize` 越大，单行 S 越长，`vec1ChunkRows_` 越小；若实现时忘了按 `vec1ChunkRows_` 行切分而整块进 UB，会超 192KB。务必落实 6.2 的行切分。
3. **MM1 K 转置加载**：基线无 `WithTranspose` helper（见 4.1）。若实现者新写 helper，要核对 C0 分块循环的 src/dst 偏移；优先复用基线 (B) 方案降低风险。
4. **tiling 结构体布局一致性**：新增字段后 host 写入顺序与 kernel `CopyTiling` 读取顺序必须逐字段对齐；建议字段类型统一 uint32 并同步改 `CopyTiling`。
5. **slot 粒度放大后的 workspace 寻址**：`* BLOCK_M` 全部要改成 `* mBaseSize`（accO/meta/O 偏移），漏改会读写错位、结果错乱。
6. **单缓冲 L1 覆盖同步**：复用同一 K/V/P L1 buffer 时，覆盖前必须等待上一块所有 L1→L0 读取完成；这是本轮最容易被遗漏的正确性契约。
7. **subblock 写回偏移坐标系（KP-001/AP-001，最高优先级，已实测复现）**：`FinalizeOutputChunk` 写回 `outGm` 必须用任务内全局行 `mGlobal`，不是 subblock 内局部偏移。详设里 state cache 用全局行、写回却写成 `curQRowStart_ + startRow`（局部偏移），实现者照字面写就会少加 `vecStartM`/`rowStart_`，导致第二个 AIV subblock 整段错位、mismatch 57%。规避：全程单一全局行坐标系，写回入参命名 `globalRow`，详设勿在不同小节混用 `startRow`/`mGlobal` 两套名。
8. **Q outer block softmax 状态重置（AP-002，高优先级，已实测复现）**：`ComputeVec1` 必须在每个 Q outer block 开始时重置 softmax 状态缓存（`maxCacheUb_`, `sumCacheUb_`），否则 `qOuterBlocks > 1` 时产生 50% mismatch。规避：添加 `isFirstQBlock` 参数或 `ResetSoftmaxState()` 方法，详见第 6.5 节。