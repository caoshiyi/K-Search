# 第三轮设计文档：MM1 K 大块转置加载（`LoadNzL1ToZnL0BWithTranspose`）

> **本轮优化点：把 MM1 的 K（B 矩阵）加载从「每块 128 行裸 LoadData2D」升级为「每块 256 行大块加载 + 转置感知的 L0B 子块抽取」。**
> **基线 = 第二轮 977us 的「双向两级分块 + 多缓冲 PRELAUNCH 软流水」版（commit 0dc4b54，分支 exp/blind-design-impl）。**
> 收益与风险一律相对 **977us** 评估（见 lessons_round2 §5.2：基线锚定第二轮，不锚定 1590us/2391us）。
> **事后校正（来自 lessons_round3）**：本轮盲实现精度 PASS、无死锁，但性能 **977us → 978us，完全持平**。因此 MM1 K 256 行大块转置加载应沉淀为“已验证边际很小/持平”的边界结论，不应再作为逼近 605us 的优先性能路径。

---

## 0. 范围声明（先划清「改什么 / 不改什么」）

本轮**只抽取一个增量优化点**：MM1（S = Q@Kᵀ）里 K 矩阵从 L1 加载到 L0B 的路径。

具体地，把第二轮 MM1 的「外层每 128 行（BLOCK_N）K 微块 GM→L1，K→L0B 用裸 `LoadData2D` 整块连续拷贝」，改为「外层每 256 行（`nL1Size`）K 大块 GM→L1，再在内层用 `LoadNzL1ToZnL0BWithTranspose` 从大块里按 128 行 n 子块抽取到 L0B」。

**为什么这两件事是同一个变量（不可拆）：** `LoadNzL1ToZnL0BWithTranspose` 在 `colC0Stride == n` 时**退化为与基线裸 `LoadData2D` 逐字节等价的单条指令**（见 §4 helper 的 if 分支）。它只有在「L1 里的 K 块比 L0B 的 n 子块更宽（colC0Stride=256 ≠ n=128）」时才走 C0 分块循环、才与基线不同。所以「引入转置 helper」与「把 K 块放大到 256 行」是一件事的两面，必须一起做，否则 helper 等于没改。本轮把这二者作为**单一逻辑变量**抽取。

**保持不变（第一/二轮已落地、已验证的结构前提，禁止重构）：**
- 第二轮的 PRELAUNCH=2 软流水顶层调度（`flash_attention_kernel.h` 的 `for t < s2OuterBlocks+PRELAUNCH` 错位发射循环）——**完全不动**。
- 第二轮的全部跨核 CrossCore 同步契约（SIG_S/P/O_READY、WorkspaceQueue 的 ProducerAcquire/ReleaseFix/ConsumerAcquire）——**完全不动**（见 §5）。
- 第二轮的多缓冲 TQue 结构（kpQueL1_ 深度3、vQueL1_ 深度2、queL0A/B 深度2、queL0C_ 深度2）与 EnQue/DeQue/FreeTensor 自动跨流水 flag——**完全不动**。
- **MM1 的 Q（A 矩阵）→ L0A 加载路径**：保持第二轮的「逐 C0 行 `LoadData2D` 循环」**逐字节不变**（见 §3 伪代码 Q 段）。本轮**不**引入 main 的 `LoadNzL1ToZzL0A` 单指令 Q 加载——那是另一个独立增量，留给后续轮次，以保持本轮单变量隔离纯净。
- **MM2（O = P@V）整段**：加载方式、KL1 切分、循环结构**完全保持第二轮**。本轮不碰 MM2。
- LoadQ、Vec1/Vec2、tiling 公式（mBaseSize/s2BaseSize/PRELAUNCH/RING_SLOTS/ws 尺寸）——**完全不动**。

**本轮要改的（且仅此）：**
1. `matmul_tile.h`：**新增**一个自补 helper `LoadNzL1ToZnL0BWithTranspose`（基线 matmul_tile.h 没有，见 §4，须逐字节按本文档实现）。
2. `flash_attention_cube.h` 的 `ComputeMM1`：K 外层循环步长 128→`nL1Size`（自适应 256/128），新增 n 方向 L0B 子块内层循环，K→L0B 改调 `LoadNzL1ToZnL0BWithTranspose`。
3. `flash_attention_cube.h` 的 `InitBuffers`：`kpQueL1_` 单槽尺寸按 MM1 K 大块重算（见 §6）。

---

## 1. 根因与差距的诚实拆解（吸取 lessons_round2 §5.2 教训）

### 1.1 第二轮为何停在 977us（未达 605us）
lessons_round2 曾把 977us→605us 的差距重点指向第二轮排除的 MM1 K 加载路径；lessons_round3 已用盲实现复核修正了这个判断：**MM1 K 256 行大块转置加载不是主要差距来源**。它不减少 GM 总流量，且第二轮软流水已把 MTE2 大量藏到计算背后，实测净收益被新增 MTE1 微指令抵消。

### 1.2 本轮抽取的增量到底改变了什么（量化，basic_case：seq=1024, dim=128, s2BaseSize=512, mBaseSize=512）
单个任务 MM1 内（qRows=512, s2Rows=512, BASE_K=128, dim=128 → kTiles=1, kRemain=0）：

| 项 | 第二轮（128 行 K 块） | 本轮（256 行 K 块） | 变化 |
|:--|:--|:--|:--|
| K 的 GM→L1 DataCopy 次数 | 4（每次 128 行） | **2（每次 256 行）** | **减半**，单次传输更大 |
| K 的 GM→L1 总字节 | 512×128×2=128KB | 128KB | **不变**（每行只搬一次） |
| 128×128 Mmad tile 数 | 4×4=16 | 16 | 不变 |
| K 的 L1→L0B 微指令数 | 16（每 tile 1 条连续 `LoadData2D`） | **128（每 tile 8 条 C0 分块 `LoadData2D`）** | **增加**（转置抽取的代价） |
| Q 的 L1→L0A 加载 | 16 tile × 8 = 128 条 | 128 条 | **不变**（Q 路径本轮不动） |

**关键诚实结论：本轮不减少任何 GM 流量**。收益来自两点：(a) MTE2（GM→L1）传输从 4 次大粒度合并为 2 次更大粒度，摊薄每次 DMA 的固定启动开销；(b) kpQueL1_ 的 ping-pong 周期在 MM1 内从 4 次减为 2 次，减少跨流水 flag 与队列调度开销、加深与计算的重叠。**代价是** K 的 L1→L0B 微指令数增加（16→128 条 MTE1）。因此这是一次「用更便宜的片上 MTE1 微指令换更贵的 MTE2 大块传输次数」的权衡，净收益取决于 MTE2 是否为 MM1 瓶颈。

### 1.3 这是「单增量上界」而非「累计上界」（不重蹈第二轮高估覆辙）
lessons_round2 §5.2 明确告诫：不要把「某单一增量」等同于「main 全部剩余增量」。977us→605us 的差距实际由**多个独立增量**构成，至少包括：
- (i) **MM1 K 大块转置加载**（本轮抽取）；
- (ii) **MM1 Q→L0A 单指令加载**（main 的 `LoadNzL1ToZzL0A`，把 Q 的 128 条 `LoadData2D` 降为 16 条，本轮**未抽取**）；
- (iii) **MM2 P/V 大块加载**（main 的 `KL1_SPLIT=256`，本轮**未抽取**）。

本轮只回收 (i)。因此**本轮目标绝不是 605us**；事后实测证明 (i) 的边际为持平，(ii)(iii) 才是后续更值得优先验证的方向。详见 §8 的边界结论。

### 1.4 事后候选排序规则（lessons_round3 新增）
当多个剩余增量并存时，先按以下顺序做快速排序，再决定下一轮抽取对象：

1. 是否减少当前瓶颈资源的总量：优先减少 GM 流量或片上 MTE1 指令总数的增量。
2. 是否与已落地优化正交：优先选择不会被第二轮软流水提前重叠掉的资源路径。
3. 片上指令方向：减少 MTE1/MTE2 指令数的候选优先于“减少 MTE2 次数但增加大量 MTE1 微指令”的候选。

按这个规则，本轮 (i) 事后应排在 (ii) Q→L0A 单指令加载之后；第四轮应优先抽取 (ii)，而不是继续在 MM1 K 路径上调参。

---

## 2. 增量优化点总览（一图）

```
第二轮(977us)                         本轮
────────────                          ──────────────────────────────
K 外层步长 = BLOCK_N(128)        →     K 外层步长 = nL1Size(dim<=128?256:128)
每 128 行 K: 1 次 GM->L1              每 256 行 K: 1 次 GM->L1（次数减半，块更大）
K->L0B: 裸 LoadData2D 整块连续拷贝 →   K->L0B: LoadNzL1ToZnL0BWithTranspose
                                       （256 行 L1 块里按 128 行 n 子块 C0 分块抽取）
(Q->L0A 逐行 LoadData2D)              (Q->L0A 逐行 LoadData2D  —— 不变)
(MM2 / 流水 / 同步)                   (MM2 / 流水 / 同步        —— 全部不变)
kpQueL1_ 单槽 = BLOCK_N*dim          kpQueL1_ 单槽 = (dim<=128?256:128)*dim（dim=128 时 32KB->64KB）
```

> **生效范围（诚实标注）：** 本优化仅对 `dim ≤ BASE_K(128)` 路径生效（实测用例中即 `dim=128`，含 basic_case）。对 `dim > 128`（用例中 256/512）路径，`nL1Size` 自适应取 `BLOCK_N=128`，`colC0Stride==n`，`LoadNzL1ToZnL0BWithTranspose` 走单指令分支，**与第二轮逐字节等价、零改变、零风险**。故聚合性能提升只体现在 dim=128 的用例上。

---

## 3. ComputeMM1 改造（算法结构，实现者按契约落地）

> 下面给出**改造后**的 `ComputeMM1` 算法结构与关键公式。实现者**只改 MM1**，函数签名保持第二轮 `ComputeMM1(int bz, int kvHead, uint32_t s2Start, uint32_t s2Rows, uint32_t qRows)` 不变（**不**学 main 增加 `qLocalOffset` 形参——第二轮 Q 寻址用 `mStart/C0+i` 逐行，本轮不动）。

算法步骤：

1. 保持第二轮 `ComputeMM1` 的函数签名、跨核 S slot 获取、Q 常驻 L1 读取方式和 Q→L0A 逐 C0 行加载方式不变。
2. 在函数内推导 `nL1Size`：`dim <= BASE_K` 时取 256，否则退回 `BLOCK_N=128`，保证 dim>128 路径与第二轮逐字节等价。
3. K 外层循环按 `nL1Size` 遍历 `s2Rows`，每次把 `kvRows × dim` 的 K 大块搬到 `kpQueL1_` 的一个 L1 slot；`kvRowsAlign = AlignUp(kvRows, C0)`，作为该 K 大块的 L1 NZ 行容量。
4. Q 中层仍按 `BLOCK_M=128` 遍历 `qRows`，每个 Q 微块的 `mRowsAlign = AlignUp(mRows, C0)`。
5. 新增 n 子块层：在当前 K 大块内部按 `BLOCK_N=128` 遍历 `nL0Start`，每个 n 子块的 `nRowsAlign = AlignUp(nRows, C0)`。Mmad 的 `n` 必须使用 `nRowsAlign`，不能使用整个 K 大块的 `kvRowsAlign`。
6. 每个 `ki` K-tile 内，Q→L0A 完全沿用第二轮逐行加载；K→L0B 改为调用 §4 的 helper，从 `kvL1` 中按 n 子块抽取。
7. K→L0B 的源偏移必须由两段组成：K-tile 起点 `ki * kvRowsAlign * BASE_K`，加上 n 子块偏移 `C0 * nL0Start`。尾部 K-tile 若存在，把 `BASE_K` 换成 `kAligned`，n 子块偏移公式不变。
8. L0C→workspace S 的 Fixpipe 目标列偏移必须使用 `nL1Start + nL0Start`，避免 256 行 K 大块中的第二个 128 行 n 子块覆盖第一个子块。
9. `kpQueL1_` 的 `FreeTensor` 仍在当前 K 大块的所有 Q 微块、所有 n 子块、所有 K-tile 的 L0B 读取都发起之后；不得提前释放，否则下一次 MTE2 可能覆盖仍在被 MTE1 读取的 L1 数据。
10. 最后按第二轮契约释放 S slot；不新增、不删除任何 CrossCore flag。

> **循环结构差异总结（三层 vs 三层）：** 第二轮是「K 微块(128) → Q 微块 → ki」三层，K 微块即 L0B 的 n 维。本轮变成「K 大块(256) → Q 微块 → n 子块(128) → ki」四层，多出的 **n 子块层** 把 256 行 L1 大块切回 128 行喂给 Mmad（Mmad 的 n 仍是 128，硬件约束不变）。Fixpipe 的目标列偏移因此要带 `nL1Start + nL0Start` 两段。

---

## 4. 强制章节：转置加载 helper 的契约级伪代码（盲实现最易错处）

> 基线 `matmul_tile.h` **没有** `LoadNzL1ToZnL0BWithTranspose`。如果为了复现实验或做负面边界验证仍要实现本轮，必须按本节契约新增 helper；如果目标是继续逼近 605us，优先转向第四轮 Q→L0A 单指令加载，不要继续优化本 helper。

### 4.1 helper 源码实现（独立通用函数）

helper 的语义是：从 L1 NZ 大块中抽取当前 `k × n` 的 K 子块，写入 L0B Zn 目标。它是独立于 FlashAttention kernel 调度的通用 L1→L0B 加载 helper，因此可以在设计文档中给出源码形式，供实现者直接落地。

```cpp
template <typename T>
__aicore__ inline void LoadNzL1ToZnL0BWithTranspose(
    const AscendC::LocalTensor<T>& dst,
    const AscendC::LocalTensor<T>& src,
    uint32_t k,
    uint32_t n,
    uint32_t colC0Stride) {
  constexpr uint32_t c0 = 32 / sizeof(T);

  AscendC::LoadData2DParams params;
  params.startIndex = 0;
  params.srcStride = 1;
  params.dstGap = 0;
  params.ifTranspose = false;

  if (colC0Stride == n) {
    params.repeatTimes = (k / c0) * (n / c0);
    AscendC::LoadData(dst, src, params);
    return;
  }

  params.repeatTimes = n / c0;
  for (uint32_t i = 0; i < k / c0; ++i) {
    AscendC::LoadData(
        dst[static_cast<uint64_t>(n) * i * c0],
        src[static_cast<uint64_t>(colC0Stride) * i * c0],
        params);
  }
}
```

这个 helper 不是在 `LoadData` 内做 16×16 转置，`params.ifTranspose` 必须保持 `false`；转置语义来自 K 作为 Mmad B 矩阵的读法。名称里的 `WithTranspose` 表示它服务于 `S = Q @ K^T` 的 B 矩阵抽取，而不是要求 `LoadData` 执行内部转置。

参数契约：
- `dst`：L0B 目标，容量为当前子块 `k × n`。
- `src`：L1 源指针，调用方已经把 K-tile 偏移和 n 子块偏移合入起点。
- `k`：当前 K-tile 的 reduce 维，对齐到 C0；常规为 `BASE_K=128`。
- `n`：当前 n 子块的对齐行数；常规为 `BLOCK_N=128`，尾块按 C0 对齐。
- `colC0Stride`：L1 大块的 NZ 行容量，也就是 `kvRowsAlign`；大块分支下通常为 256，退化分支下等于 `n`。

分支契约：
- 当 `colC0Stride == n` 时，helper 必须退化为单块连续加载，与第二轮裸 L0B 加载逐字节等价。
- 当 `colC0Stride > n` 时，helper 按 `k / C0` 个 C0 段抽取；每段只搬当前 n 子块对应的 `n / C0` 个分形。
- 不允许使用 `colC0Stride < n` 的形态；出现该形态说明调用方传参或尾块对齐错误。

### 4.2 LoadData2DParams 逐字段契约（实现者据此自检，不可臆测）

| 字段 | 取值 | 含义 / 为什么 |
|:--|:--|:--|
| `startIndex` | `0` | 起始分形索引。调用方已通过 `src[...]` 把 n 子块/k-tile 偏移算进指针，故 helper 内恒 0。 |
| `srcStride`  | `1` | 相邻 fractal（16×16）在 L1 NZ 中的源步长（以 fractal 为单位）。连续抽取，恒 1。 |
| `dstGap`     | `0` | L0B 目标 fractal 间无间隙，连续写。 |
| `ifTranspose`| `false` | **不在 LoadData 内做 16×16 转置**。Kᵀ 的转置语义由 NZ 排布 + Mmad 的 B 矩阵读法天然完成，这里只是「抽取」不是「转置数据」。命名里的 Transpose 指它服务于 S=Q@Kᵀ 的 B 矩阵抽取，**取值务必 false**。 |
| `repeatTimes`（退化分支）| `(k/c0)*(n/c0)` | 整块一次性搬 `k×n` 个元素 = `(k/16)*(n/16)` 个 fractal。 |
| `repeatTimes`（大块分支）| `n/c0` | 每个 C0 行（16 行 k）只搬 `n/16` 个 fractal。 |
| 循环次数（大块分支）| `k/c0` | 沿 k 方向分 `k/16` 段；k=128 时 8 次（与 §1.2 表「8 条 C0 分块」一致）。 |
| `dst` 偏移 | `n * i * c0` | 第 i 个 C0 段在 L0B 里的元素偏移 = `n(列)×16(行)×i`，连续排布。 |
| `src` 偏移 | `colC0Stride * i * c0` | 第 i 个 C0 段在 L1 里跳 `colC0Stride×16×i` 个元素——**关键**：因为 L1 大块列步长是 256（colC0Stride），不是 128（n），所以不能用 `n*i*c0`，否则读错位（这正是大块分支与退化分支唯一的本质差异）。 |

### 4.3 它如何替换第二轮 MM1 里的裸 L0B 加载

替换只发生在 MM1 的 K→L0B 位置，且仅替换加载形态，不改变 L0B 队列、Mmad、Fixpipe 或跨核同步。

- 第二轮的 L1 K 块宽度等于 L0B n 子块宽度，因此一次连续加载即可。
- 本轮在 `dim≤128` 路径把 L1 K 块宽度扩到 256，但 L0B 仍按 128 列子块喂给 Mmad，因此调用 helper 前必须把源指针定位到当前 128 行 n 子块。
- 源指针偏移由两部分组成：K-tile 的 `ki * kvRowsAlign * BASE_K`，以及 n 子块的 `C0 * nL0Start`。这里 `nL0Start` 以行计，NZ 排布下不能改写成 `nL0Start * dim` 或其他 ND 公式。
- helper 的 `k/n/colC0Stride` 三个维度分别传当前 K-tile、当前 n 子块、L1 大块宽度。三者含义不同，不能互换。

- **当 dim>128（`kvRowsAlign == BLOCK_N == 128`，`nL0Start` 恒 0）：** `colC0Stride == n == 128`，走退化分支，与第二轮逐字节等价。
- **当 dim≤128（`kvRowsAlign == 256`）：** `colC0Stride=256 != n=128`，走大块抽取分支，按 C0 段抽取当前 n 子块。这是本轮真正的新路径。

> **自检要点（实现者务必逐条核对）：**
> 1. helper 第三参数 `k` 传 `BASE_K`（或尾块 `kAligned`），**不是** kvRows；第四参数 `n` 传 `nRowsAlign`（当前 n 子块）；第五参数 `colC0Stride` 传 `kvRowsAlign`（L1 大块宽度，256）。三者顺序极易搞反。
> 2. `src` 指针偏移里的 `C0 * nL0Start`：`C0=16`，n 子块沿列方向，每跨一个 128 行子块需跳 `16×128` 个元素？**否**——NZ 排布下 n 子块偏移是 `C0 * nL0Start = 16 * nL0Start`（nL0Start 以行计，第二子块 nL0Start=128 → 偏移 2048 元素 = 一个 fractal 列条）。务必照 §3 的 `(uint64_t)C0 * nL0Start` 写，不要自行推导成 `nL0Start * something_else`。
> 3. Mmad 的 `mp.n` 改用 `nRowsAlign`（128），**不是** kvRowsAlign（256）。否则 Mmad 维度与 L0B 实际子块不符。

---

## 5. 同步契约：完全保持第二轮，不动（强制声明）

本轮**只改 L1→L0B 的加载指令形态**，不改任何同步语义。以下契约**逐字沿用第二轮 design_doc_round2.md §4，实现者不得改动**：

- **跨核 CrossCore（§4.A）：** SIG_S/P/O_READY 三信号、`ProducerAcquire/ProducerReleaseFix/ProducerReleaseMte3/ConsumerAcquire` 的调用点与配对计数（每信号各 s2OuterBlocks 次，prologue 只 Producer、epilogue 只 Consumer），**完全不变**。MM1 入口仍 `sQueue_.ProducerAcquire()`、出口仍 `sQueue_.ProducerReleaseFix()`，位置不动。
- **Cube L1/L0 TQue 自动同步（§4.B）：** `kpQueL1_` 深度 3、`AllocTensor→LoadNd→EnQue→DeQue→...→FreeTensor` 的 ping-pong 流程**不变**。本轮只是把 DeQue 之后、FreeTensor 之前的「对 kvL1 的若干次 MTE1 读」从「1 次裸 LoadData2D」变成「n 子块 × ki 次 `LoadNzL1ToZnL0BWithTranspose`（内部含 C0 循环）」。**所有这些 MTE1 读仍发生在同一个 DeQue 与 FreeTensor 之间**，TQue 的 MTE2↔MTE1 flag 覆盖范围不变 —— FreeTensor 仍在「该 K 大块的全部 n 子块、全部 ki 的 L0B load 都发起之后」调用（§3 伪代码已把 FreeTensor 放在最外层 K 大块循环末尾，位置正确）。
- **同 L0C 多次 Mmad 累加序：** `PipeBarrier<PIPE_M>()` 在 ki 循环内**保留不动**（同 cL0 累加）。注意本轮 n 子块之间用的是**不同** cL0（每个 n 子块 AllocTensor 一次新 cL0），它们之间靠 queL0C_ 深度 2 的 ping-pong flag 隔离，无需额外 barrier。
- **Vec 侧 UB 队列（§4.C）：** 完全不碰（本轮不动 vec.h）。

> **一句话契约：本轮对同步系统是「透明」的——把一条 MTE1 指令换成一簇 MTE1 指令，不增删任何 SetFlag/WaitFlag/CrossCore/EnQue/DeQue/FreeTensor。** 实现者若发现自己在动 flag，说明改错了。

---

## 6. 容量校验（L0B/L1 占用变化，必须 ≤ L0B 64KB / L1 512KB）

数据类型 half/bf16（sizeof=2，C0=16）。主路径 dim=128, s2BaseSize=512, mBaseSize=512。

### 6.1 L0B 占用：**不变**（关键结论）
- 第二轮 L0B 单槽 = `BASE_K × BLOCK_N = 128×128 half = 32KB`，深度 2 = 64KB（恰满）。
- 本轮 L0B 仍按 **n 子块（128）** 喂 Mmad，`LoadNzL1ToZnL0BWithTranspose` 的 dst 一次写 `k×n=128×128`。**L0B 单槽仍 32KB，深度 2 仍 64KB，零变化。** 大块只发生在 L1，不发生在 L0B。

### 6.2 L1 占用：**kpQueL1_ 单槽翻倍（dim=128 时）**
- 第二轮 kpQueL1_ 单槽 = `max(BLOCK_N×dim, BLOCK_M×BLOCK_N) = max(128×128, 128×128) = 128×128 half = 32KB`，深度 3 = 96KB。
- 本轮 MM1 的 K 大块需 `nL1Size×dim = 256×128 half = 64KB`；MM2 的 P 块仍 `BLOCK_M×BLOCK_N=32KB`（MM2 不改）。故 **kpSlot = max(256×128, 128×128) = 64KB**，深度 3 = **192KB**。
- L1 其余：qBufL1_（mBaseSize×dim=512×128 half=128KB）、vQueL1_（BLOCK_N×dim=128×128×2槽=64KB）。
- **L1 合计 = 192 + 128 + 64 = 384KB ≤ 512KB ✓**（余 128KB，安全）。

> **InitBuffers 改法（实现者照此）：** 把 `uint32_t kElems = BLOCK_N * dim;` 改为 `uint32_t kElems = ((dim <= BASE_K) ? 256U : BLOCK_N) * dim;`（即 `nL1Size * dim`，与 MM1 的 nL1Size 公式严格一致）。`pElems = BLOCK_M * BLOCK_N` 不变。`kpSlot = max(kElems, pElems)`。**严禁**把 nL1Size 写死 256——dim>128 时必须退回 128，否则浪费 L1 且与 MM1 循环步长不一致。
> **dim 边界自检：** 通用用例含 dim=256/512（>128），此时 nL1Size=128，kElems=128×256 或 128×512 = 64KB 或 128KB。需重算：dim=512 时 kpSlot=128×512×2=128KB，深度3=384KB；qBufL1_ 此时 mBaseSize 较小（短 Q 路径 mBaseSize≤256）。实现者必须用**实际 dim** 算 kElems，不要假设 dim=128。dim=512 长 Q 不在 basic_case，但通用用例会触发，InitBuffers 公式必须 dim 自适应（已由 `kElems=nL1Size*dim` 保证）。

---

## 7. 最小改动边界（实现者只动这两个文件）

| 文件 | 改动 | 不改 |
|:--|:--|:--|
| `matmul_tile.h` | **新增** `LoadNzL1ToZnL0BWithTranspose`（§4.1 逐字节） | 其余 helper 全部不动 |
| `flash_attention_cube.h` | (a) `ComputeMM1` 按 §3 改造（外层步长 nL1Size、新增 n 子块层、K→L0B 改调 helper、Fixpipe 列偏移加 nL1Start+nL0Start、Mmad.n 改 nRowsAlign）；(b) `InitBuffers` 的 kElems 按 §6.2 改 | `ComputeMM2`、`LoadQ`、`Init`、所有 buffer 声明（kpQueL1_/vQueL1_/queL0A/B/C 深度与类型）、私有成员 —— **全部不动** |
| `flash_attention_kernel.h` | **不改**（顶层流水循环、MM1 调用签名不变） | 全部 |
| `flash_attention_vec.h` | **不改** | 全部 |
| `flash_attention_tiling.h/.cpp` | **不改**（nL1Size 在 kernel 内由 dim 推导，不进 tiling） | 全部 |
| `workspace_queue.h`、`kernel_common.h` | **不改** | 全部 |

> **签名约束（重要）：** 第二轮 `ComputeMM1` 签名是 5 参 `(bz, kvHead, s2Start, s2Rows, qRows)`，`flash_attention_kernel.h` 按此调用。本轮**保持 5 参不变**，不引入 main 的 `qLocalOffset`。`nL1Size` 在函数体内由 `dim` 局部推导，不入参、不入 tiling。这样 kernel.h 零改动。

---

## 8. 待验证假设：保守收益估计与风险（吸取 lessons_round2 §5.2「不高估上界」教训）

**基线锚定：本节所有收益相对第二轮 977us（commit 0dc4b54），绝不锚定 605us/1590us/2391us。**

### 8.1 已验证收益与边界结论
- **已验证结果：977us → 978us，完全持平（噪声带内）。** 精度 PASS、无死锁，说明本设计的 stride/同步契约可实现，但它不是有效性能增量。
- **持平原因（lessons_round3 已验证）：** (i) 不减少 GM 总流量，只把 MM1 内 K 的 GM→L1 次数从 4 减到 2；第二轮软流水已经把大量 MTE2 等待藏到计算背后，新增的 L1→L0B MTE1 微指令又抵消了剩余收益。
- **策略归类：负面或有界策略。** 该设计可作为“不要优先抽取 MM1 K 大块转置加载”的证据；除非目标是复现实验或验证边界，不建议再围绕本增量继续优化。
- **后续方向：** 按 §1.4 排序，第四轮应优先抽 (ii) Q→L0A 单指令加载，因为它直接减少片上 MTE1 指令数；若仍持平，再转向 (iii) MM2 大块加载或 `s2OuterBlocks`/steady 占比问题。

### 8.2 主要风险点（按概率排序）
1. **stride/寻址错位（最高风险，精度 FAIL 而非死锁）：** `LoadNzL1ToZnL0BWithTranspose` 的 `src` 偏移 `colC0Stride*i*c0` vs `dst` 偏移 `n*i*c0`、调用处 `kL1Off` 的 `C0*nL0Start`、第三/四/五参数顺序（k/n/colC0Stride）——任一错位都会让 K 子块读错，精度 mismatch。§4.2/§4.3 已给逐字段契约 + 自检三条，但这是盲实现历史最易翻车处（lessons_round1 §4）。
2. **Fixpipe 目标列偏移漏加 nL0Start：** 本轮 Fixpipe dst 列偏移从第二轮的 `nStart` 变为 `nL1Start + nL0Start`（两段）。漏掉 nL0Start → 大块内第二个 n 子块写回覆盖第一个 → 精度错且只在 dim≤128 体现。
3. **Mmad.n 用错（256 vs 128）：** §3/§4.3 自检点 3。用 kvRowsAlign(256) 而非 nRowsAlign(128) → Mmad 维度越界或读脏。
4. **InitBuffers kElems 没跟着 nL1Size 改：** 若 InitBuffers 仍按 `BLOCK_N*dim=32KB` 分配，而 MM1 按 256 行写 64KB → L1 越界踩踏（dim=128 时）。§6.2 已强制「kElems=nL1Size*dim」公式一致性。
5. **helper 退化分支边界（colC0Stride==n）误判：** dim>128 时必须走退化分支与第二轮逐字节等价。若实现者把 if 条件写错（如写成 `colC0Stride<=n` 或漏掉退化分支），dim>128 用例会回归。§4.1 的 if 条件须照抄。
6. **kRemain 尾块路径（dim 非 128 整数倍）：** 通用用例 dim=256/512 是 128 整数倍（kRemain=0），但 dim=473 之类不在用例。当前用例 kRemain 恒 0，尾块分支不触发；实现者仍须把尾块分支按 §3 写对（防御性），但不是本轮验证重点。

### 8.3 验证建议（交给主控/debugger）
- **先 basic_case（seq=1024, dim=128）：** 这是唯一稳定触发大块分支（nL1Size=256）的用例，精度必须 PASS（REL ~1e-3，与第二轮同量级，因不改算法）。若 FAIL，优先查风险 1/2/3（stride/Fixpipe/Mmad.n）。
- **再全量：** 确认 dim>128 用例（256/512）与第二轮**逐字节一致**（走退化分支），不应有任何精度或性能回归——若回归，查风险 5（退化分支条件）。
- **性能：** 与 977us 对比。若持平在 970~990us 且精度 PASS，应按 lessons_round3 记录为有效负面结果，不要继续调本增量。若明显低于 940us，复核是否误掺 Q→L0A、MM2 大块加载或其他非本轮改动；若明显退化，优先查 MTE1 指令增多、helper 大块分支和 buffer 尺寸。
- **死锁排查：** 本轮不动同步，理论上不该死锁；若 TIMEOUT，首查是否误改了 §5 的 TQue/CrossCore 调用（说明越界改了同步）。

---

## 9. 与前两轮范式的衔接（自检清单）

| lessons 教训 | 本轮落实 |
|:--|:--|
| R1§5：基线锚定要正确 | §0/§8 全程锚定 977us |
| R1§4：stride/寻址必须契约级给定 | §4.2 逐字段表 + §4.3 替换对照 + §8.2 风险1 |
| R1§4：多方案二选一要给默认 | 本轮只给方案 A（自补 helper），无二义（main 也是 A） |
| R2§5.2：不把单增量当累计上界 | §1.3 拆三子项，§8.1 明确「目标≠605us」，并用实测持平更新边界 |
| R2§5.3：动 L0B 加载要写 LoadData2DParams 契约 | §4.1/§4.2 完整给出 |
| R2§5：同步契约延续不动 | §5 强制声明「对同步透明」 |
| R2§5.4：单变量隔离 | §0 声明「nL1Size+转置 helper 是同一逻辑变量」，不掺 Q/MM2 |
| R3§2：多候选要先排序 | §1.4 按瓶颈资源、正交性、片上指令方向排序，后续优先 Q→L0A |

> **本轮与 main 的唯一对照差异（供主控复核归因）：** main 的 MM1 同时用了 (i) K 转置大块加载 **和** (ii) Q 的 `LoadNzL1ToZzL0A` 单指令加载。本轮**只取 (i)，Q 路径保持第二轮逐行 LoadData2D**。实测持平后，剩余差距应主要转向 (ii)+(iii) 或流水 steady 占比，而不是继续归因到 MM1 K 加载。
