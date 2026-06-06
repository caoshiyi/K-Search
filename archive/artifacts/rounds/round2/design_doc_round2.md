# 第二轮设计文档：多缓冲软流水（搬运-计算重叠）

> **本轮优化点：多缓冲流水（MTE2 搬运与 Mmad/Vector 计算重叠）。**
> **基线 = 第一轮 1590us 的「QKV 双向两级分块 + L1 大块复用」版（commit 6fc58de，单缓冲、无流水、1.49x）。**
> 收益与风险一律相对 **1590us** 评估，绝不再锚定 2391us 原始基线（见 lessons_round1 §2/§5.1 的误判教训）。
> **事后校正（来自 lessons_round2）**：本轮盲实现精度 PASS、无死锁，性能约 **1590us → 977us（1.63x）**。这验证了多缓冲软流水是有效增量，但也证明 **605us 不是本轮单增量目标**；本轮主动排除了 MM1 转置加载等其他 main 增量，不能把“流水子集”写成“main 全部剩余差异”。

---

## 0. 范围声明（先划清「改什么 / 不改什么」）

本轮**只抽取一个增量优化点**：把第一轮的「单缓冲 + 逐 KV 块串行（MM1→等 Vec1→MM2→等 Vec2）」改成「多缓冲 + PRELAUNCH 软流水」，让下一 KV 块的搬运与当前块的计算重叠。

**保持不变（第一轮已落地、已验证 1.49x 的结构前提，禁止重构）：**
- QKV 双向两级分块：Q 大块（mBaseSize 行）常驻 L1，K/V/P 以 128 行微块（BLOCK_N/BLOCK_M）粒度处理。
- 两级循环结构（外层 KV 微块、中层 Q 微块、内层 K-tile）保持第一轮 `flash_attention_cube.h` 的形态。
- MM1 用裸 `LoadData2D`（第一轮选定方案 B，规避缺失的 transpose helper）。
- Vec 侧的在线 softmax 状态缓存（maxCache/sumCache + wsMeta exp + wsAccO 累加器）按 RING_SLOTS 槽寻址的方式**完全保持**——它本来就是为 ring-slot 流水而设计的，本轮不动其算法，只改顶层调度的发射时序。
- 所有 workspace 布局、tiling 公式（mBaseSize/s2BaseSize/RING_SLOTS/PRELAUNCH/wsXxxSize）保持不变。

**本轮要改的三处（且仅此三处）：**
1. **顶层调度循环**：串行 `for t2 { MM1(t2); MM2(t2); }` → **PRELAUNCH 错位软流水**（MM1 领先 MM2 两个 KV 块发射；Vec 同步错位）。
2. **Cube 侧 L1 buffer**：K/P 合并为深度 3 的 A1 队列，V 使用深度 2 的 A1 队列；L0C 从深度 1 改为深度 2。K/P 深度 3 用来覆盖 PRELAUNCH=2 下的在飞块，V 深度 2 用于普通 ping-pong。
3. **删除 Cube 侧手工全序化 barrier**：第一轮在 MM1/MM2 里大量使用 `SetWaitFlag<...>`（= SetFlag 紧跟 WaitFlag，等于硬同步、零重叠），本轮删除这些，改由 TQue 的 EnQue/DeQue/FreeTensor 自动插入带 ping-pong 的跨流水线 flag，从而获得重叠。

---

## 1. 为什么第一轮没有重叠（根因，决定本轮怎么改）

第一轮顶层（`flash_attention_kernel.h` Process）：

```
AIC: for t2 in [0, s2OuterBlocks):
        ComputeMM1(t2)          // 产 S[t2] -> CrossCoreSetFlag(SIG_S_READY)
        ComputeMM2(t2)          // ConsumerAcquire 等 SIG_P_READY 才能开始
AIV: for t2 in [0, s2OuterBlocks):
        ComputeVec1(t2)         // 等 SIG_S_READY，产 P[t2] -> SIG_P_READY
        ComputeVec2(t2)         // 等 SIG_O_READY
```

虽然第一轮已经有 WorkspaceQueue + CrossCore flag + RING_SLOTS=3 槽位，但**发射时序是紧耦合串行链**：
`MM1(t2) → Vec1(t2) → MM2(t2) → Vec2(t2) → MM1(t2+1) → …`
- `MM2(t2)` 的 `pQueue_.ConsumerAcquire()` 会 `CrossCoreWaitFlag(SIG_P_READY)`，**阻塞**直到 AIV 把 Vec1(t2) 做完。
- AIV `Vec2(t2)` 同理阻塞等 AIC 的 `SIG_O_READY`。
- 因为循环里在 `MM2(t2)` 返回前**绝不发射 `MM1(t2+1)`**，所以 t2+1 块的 K 搬运（MTE2）永远无法与 t2 块的计算重叠。RING_SLOTS=3 的槽位被白白浪费。

此外，单块内部 MM1/MM2 里每个 `SetWaitFlag<MTE2_MTE1>` / `<MTE1_M>` / `<M_MTE1>` 都是硬同步，**连块内**下一微块的 K-load 也无法与当前 Mmad 重叠。

→ 结论：要拿到重叠，必须 (a) 顶层让 MM1 领先 MM2 发射（跨块软流水），(b) L1/L0C 多缓冲 + 去掉手工 barrier（块内 + 跨块搬运-计算重叠）。这只是 main(deae846) 相对第一轮的**一个可隔离子集**；MM1 转置加载、Q→L0A 单指令、MM2 大块加载等路径若存在，必须作为后续独立增量另行评估。

---

## 2. 增量优化点总览（一图）

```
第一轮(1590us)         本轮(已验证≈977us)
─────────────         ──────────────────
MM1(t)                MM1(t)  ┐ 领先 PRELAUNCH=2 块发射
[硬等 Vec1(t)]   →     MM2(t-2)┘ 同一顶层迭代内先做 MM2(t-2) 再做 MM1(t)
MM2(t)                ── 搬运(t)与计算(t-2)重叠 ──
单缓冲 K/V/P TBuf      K/P 合并深度3，V深度2（载入下块时算当前块）
L0C 单缓冲             L0C TQue<CO1,2>
满地手工 SetWaitFlag   删除手工 barrier，交给 TQue EnQue/DeQue 管 MTE↔M↔FIX
```

---

## 3. 顶层调度循环改造（PRELAUNCH 软流水）

### 3.1 不变量
- `PRELAUNCH = 2`，`RING_SLOTS = PRELAUNCH + 1 = 3`（tiling.h 已有，**保持不动**）。
- 每个任务（一个 (bz, kvHead, qHead, qOuterIdx) 组合）内，沿 KV 方向有 `s2OuterBlocks` 个大块（KV 大块 = s2BaseSize 行）。软流水在**这一层**展开。
- ring slot 数与 PRELAUNCH 的关系（强约束，决定 WorkspaceQueue 无需 free 信号即安全）：
  `RING_SLOTS (3) > PRELAUNCH (2)`。在任意时刻，最多有 PRELAUNCH+1=3 个 KV 块的 S/P/O slot 同时「在飞」（in-flight），3 个槽恰好够用且不互相覆盖。**实现者不得把 PRELAUNCH 调到 ≥ RING_SLOTS**，否则生产者会覆盖消费者尚未读走的 slot（无 free 信号兜底）。

### 3.2 新顶层循环（替换第一轮 Process 里的 `for t2` 段）

把第一轮「先全做 MM1 再全做 MM2」的串行循环，改成单层错位发射循环，迭代次数 = `s2OuterBlocks + PRELAUNCH`：

```
// AIC 侧
LoadQ(bz, qHead, qRowStart, qRows)          // Q 大块常驻 L1，整任务一次（不变）
for (t = 0; t < s2OuterBlocks + PRELAUNCH; ++t) {
    // 先消费：做落后 PRELAUNCH 块的 MM2（它的 P 此刻应已由 Vec1 产出）
    if (t >= PRELAUNCH) {
        uint32_t o = t - PRELAUNCH;
        ComputeMM2(bz, kvHead, s2Start(o), s2Rows(o), qRows);   // 内部 ConsumerAcquire(SIG_P_READY)
    }
    // 再生产：发射当前块的 MM1（领先 MM2 共 PRELAUNCH 块）
    if (t < s2OuterBlocks) {
        ComputeMM1(bz, kvHead, s2Start(t), s2Rows(t), qRows);   // 内部 ProducerReleaseFix(SIG_S_READY)
    }
}

// AIV 侧（与 AIC 严格对称，同一个 t 循环、同样错位）
SetTaskContext(bz, qHead, qRowStart, qRows)
for (t = 0; t < s2OuterBlocks + PRELAUNCH; ++t) {
    if (t >= PRELAUNCH) {
        uint32_t o = t - PRELAUNCH;
        uint32_t slot = o % RING_SLOTS;
        ComputeVec2(slot, /*isFirst=*/o==0, /*isLast=*/o+1==s2OuterBlocks);  // ConsumerAcquire(SIG_O_READY)
    }
    if (t < s2OuterBlocks) {
        uint32_t slot = t % RING_SLOTS;
        ComputeVec1(slot, /*isFirst=*/t==0, s2Rows(t));   // ConsumerAcquire(SIG_S_READY)+ProducerReleaseMte3(SIG_P_READY)
    }
}
```

> 关键点：**同一顶层迭代 t 里，先发 MM2(t-2) 再发 MM1(t)**。这样 MM1(t) 的 K/V 搬运（MTE2）就和 MM2(t-2) 的 Mmad/Fixpipe 计算落在同一拍，靠多缓冲实现重叠。第一轮做不到，是因为它在 MM1(t) 之后立刻硬等 MM2(t)。

### 3.3 prologue / steady / epilogue 语义（错位发射等价于三段式软流水）
- **prologue**（t=0..PRELAUNCH-1）：只发 MM1/Vec1，填满流水（产出 S[0],S[1] / P[0],P[1]），还没有 MM2/Vec2。
- **steady**（t=PRELAUNCH..s2OuterBlocks-1）：每拍同时「MM2(t-2) + MM1(t)」，搬运与计算满重叠。
- **epilogue**（t=s2OuterBlocks..s2OuterBlocks+PRELAUNCH-1）：只发 MM2/Vec2，排空流水（消费剩余 P[*]，O[*]）。
- 这与 main 的 `s2OuterBlocks + PRELAUNCH` 循环上界完全一致，**实现者直接照此上界写**，不要自己另设 prologue/epilogue 分支。

### 3.4 跨任务边界
- 软流水**在单个任务的 KV 维度内**展开。任务之间（不同 bz/qHead）不跨流水：每个任务结束时 epilogue 已排空所有 in-flight slot，head_/tail_ 计数自然对齐，下一个任务从干净状态重新 prologue。无需额外屏障。

---

## 4. 同步契约（强制章节，最详细 —— 盲实现最大盲区）

本轮同步分三层：**(A) AIC↔AIV 跨核 CrossCore flag**（沿用第一轮，仅发射时序变）、**(B) Cube 侧 L1/L0 的 TQue 自动同步**（替换第一轮手工 barrier）、**(C) Vec 侧 UB 队列同步**（沿用第一轮）。下面逐层给出显式契约。

### 4.A 跨核 CrossCore 信号（生产者-消费者依赖图）

三个信号沿用第一轮 `tiling.h` 定义，**不新增、不改语义**，只改「谁在第几拍发/收」：

```
SIG_S_READY (id=0): AIC.MM1 产 S[t]   --CrossCoreSetFlag<PIPE_FIX>-->  AIV.Vec1 消费
SIG_P_READY (id=1): AIV.Vec1 产 P[t]  --CrossCoreSetFlag<PIPE_MTE3>--> AIC.MM2 消费
SIG_O_READY (id=2): AIC.MM2 产 O[t]   --CrossCoreSetFlag<PIPE_FIX>-->  AIV.Vec2 消费
```

依赖图（一个 KV 块 t 的生命周期，跨 PRELAUNCH=2 拍）：

```
   AIC                         AIV
   MM1(t) --SIG_S_READY(t)-->  Vec1(t) --SIG_P_READY(t)--> MM2(t) --SIG_O_READY(t)--> Vec2(t)
   [拍 t]                      [拍 t]                       [拍 t+2]                    [拍 t+2]
```

WorkspaceQueue 原语映射（**保持第一轮 workspace_queue.h 不变**）：
- `ProducerAcquire()`：返回 `head_ % DEPTH` 槽地址，**不阻塞**（靠 RING_SLOTS>PRELAUNCH 保证安全）。
- `ProducerReleaseFix()`：`CrossCoreSetFlag<0x2, PIPE_FIX>(id)` 后 `head_++`。MM1/MM2 末尾各调一次。
- `ProducerReleaseMte3()`：`CrossCoreSetFlag<0x2, PIPE_MTE3>(id)` 后 `head_++`。Vec1 末尾调一次。
- `ConsumerAcquire()`：`CrossCoreWaitFlag<0x2, PIPE_MTE2>(id)` 后返回 `tail_++ % DEPTH` 槽。MM2/Vec1/Vec2 入口各调一次。

> **强制时序不变量（实现者必须保证）：** 每个 KV 块的 `ProducerRelease*` 与对应消费侧的 `ConsumerAcquire` 必须**配对且数量相等**。错位发射后，AIC 一个 t 迭代里 MM2 先 `ConsumerAcquire(P)` 再 MM1 `ProducerReleaseFix(S)`；AIV 一个 t 迭代里 Vec2 先 `ConsumerAcquire(O)`，Vec1 再 `ConsumerAcquire(S)`+`ProducerReleaseMte3(P)`。**Acquire/Release 的总次数 = s2OuterBlocks（每信号），prologue 只发 Producer、epilogue 只发 Consumer，循环上界 s2OuterBlocks+PRELAUNCH 恰好令两者各自累计 s2OuterBlocks 次。** 实现者不得在 prologue/epilogue 漏发或多发任何一次 flag，否则跨核 flag 计数错位 → 死锁或脏数据。

> **关键：vecDealM_==0 的空闲 subblock 也必须发 flag。** 当某 AIV subblock 分到 0 行（短 Q）时，Vec1 仍须执行 `ConsumerAcquire(S)` + `ProducerReleaseMte3(P)`（即使不算），否则与 AIC 的 flag 计数失配。第一轮 Vec1 在 `vecDealM==0` 时 `return` 前**没有**补发 P_READY —— 这是 bug 温床。**本轮强制：Vec1 入口先 `ConsumerAcquire(S)`，若 vecDealM==0 则补 `ProducerReleaseMte3(P)` 再 return**（main 正是这么做的）。Vec2 的 O_READY 已被 ConsumerAcquire 在入口消费，vecDealM==0 时直接 return 即可（不产新 flag）。

### 4.B Cube 侧 L1 / L0 同步（核心改动：手工 barrier → TQue 自动同步）

**第一轮做法（要删除）：** L1 用 `TBuf<A1>`（无队列语义），靠人肉插 `SetWaitFlag<HardEvent::XXX>()`（= SetFlag 后立刻 WaitFlag，纯硬同步、零重叠）串起 MTE2→MTE1→M→FIX 全链。代表性硬同步点（第一轮 cube.h）：
- `LoadQ` 尾部 `MTE2_MTE1`；
- MM1 里 `MTE1_MTE2`（L1 复用前等上次 L0 读完）、`MTE2_MTE1`（K 载入后）、每个 ki 的 `M_MTE1`/`MTE1_M`、`PipeBarrier<PIPE_M>`、Fixpipe 前后 `M_FIX`/`FIX_MTE2`；
- MM2 里同构的一串。

**本轮做法：** 把 L1 三个 buffer 改成 TQue，并**删除上述所有手工 `SetWaitFlag`**，让 TQue 的 `EnQue/DeQue/FreeTensor` 在 ping-pong 缓冲间自动生成跨流水线 flag，从而让「下块搬运」与「本块计算」真正重叠。

buffer 角色与队列深度（与 main 一致）：

| buffer | 第一轮 | 本轮 | 流水线对 | 作用 |
|:--|:--|:--|:--|:--|
| Q L1 | `TBuf<A1>` 单缓冲 | **`TBuf<A1>`（保持单缓冲）** | — | Q 整任务常驻、只读不重载，无需 ping-pong |
| K / P L1 | `TBuf<A1>` ×2 | **合并为 `TQue<A1, 3>` kpQueL1_** | MTE2↔MTE1 | MM1 装 K、MM2 装 P，深度 3 配合 PRELAUNCH |
| V L1 | `TBuf<A1>` | **`TQue<A1, 2>` vQueL1_** | MTE2↔MTE1 | MM2 装 V，ping-pong |
| L0A | `TQue<A2,2>` | 保持 `TQue<A2,2>` | MTE1↔M | 不变 |
| L0B | `TQue<B2,2>` | 保持 `TQue<B2,2>` | MTE1↔M | 不变 |
| L0C | **`TQue<CO1,1>`** | **改 `TQue<CO1,2>`** | M↔FIX | 双缓冲让 Fixpipe 写回与下个 Mmad 重叠 |

**L1 队列的标准用法（实现者逐字遵守，替代手工 flag）：**
```
// 生产端（MTE2 GM->L1）
LocalTensor<QType> kvL1 = kpQueL1_.AllocTensor<QType>();   // 自动等该 ping-pong 槽的上次 MTE1 读完
LoadNdGmToNzL1(kvL1, kGm_[...], ...);                      // MTE2
kpQueL1_.EnQue(kvL1);                                      // 发 MTE2->MTE1 flag
// 消费端（MTE1 L1->L0）
kvL1 = kpQueL1_.DeQue<QType>();                            // 等 MTE2 完成
LoadNzL1ToZnL0B(bL0, kvL1[...], ...);                      // MTE1（多次 ki/n 循环都从该 kvL1 读）
...                                                         // 所有对该 kvL1 的 MTE1 读发起后
kpQueL1_.FreeTensor(kvL1);                                 // 释放该槽，允许下块 MTE2 复用（带 MTE1->MTE2 flag）
```
要点：
1. **`AllocTensor`/`EnQue`/`DeQue`/`FreeTensor` 成对出现**，深度 N 的队列允许 N 个 in-flight，硬件据此在两个物理槽间插 flag → 下块 K 的 MTE2 可与本块 Mmad 并行。
2. **L0A/L0B 内层循环**保持第一轮形态（`AllocTensor→Load→EnQue→DeQue→Mmad→FreeTensor`），但**删除 `M_MTE1`/`MTE1_M`/`MTE2_MTE1` 等手工 flag**；`PipeBarrier<PIPE_M>` 在多个 ki 累加同一 L0C 时仍需保留（保证 Mmad 顺序累加，见下）。
3. **L0C 双缓冲**：`cL0 = queL0C_.AllocTensor()` → 内层 ki 循环对同一 cL0 累加 Mmad（`cmatrixInitVal=(ki==0)`）→ `EnQue(cL0)`→`DeQue`→`Fixpipe`→`FreeTensor`。深度 2 让上一个微块的 Fixpipe 写回与当前微块的 Mmad 重叠。
4. **MM1/MM2 尾部不再需要 `SetWaitFlag<FIX_MTE2>`**：FreeTensor + 下一块 AllocTensor 的 ping-pong flag 已覆盖该依赖。仅保留各自末尾的 `ProducerReleaseFix()`（跨核 flag）。

> **同一 L0C 多次 Mmad 累加的顺序保证（必须保留）：** MM1 的 K-tile 循环、MM2 的 KV-reduce 循环都对同一个 `cL0` 做多次 `Mmad(cL0,...,cL0,mp)` 累加。这些 Mmad 之间靠 `PipeBarrier<PIPE_M>()` 保证按序，**不能删**。删的是跨流水线（MTE↔M↔FIX）的手工 flag，不是 PIPE_M 内部的累加序。

### 4.C Vec 侧 UB 队列同步（沿用第一轮，深度已够）

Vec 侧第一轮已用 TQue（`inputQue1_ TQue<VECIN,2>`、`outputQue1_ TQue<VECOUT,1>`）+ 少量 `SetWaitFlag<MTE3_MTE2>`/`<MTE2_V>`/`<V_MTE3>`。本轮**Vec 侧算法与 UB 同步基本保持**，仅两点对齐流水：
1. `inputQue1_` 保持深度 2、`outputQue1_` 建议提到深度 2（main 即 2），让 Vec1 下一行块的 DataCopy(S) 与当前块的 softmax 计算重叠；UB 预算（§5.2）允许。
2. softmax 状态缓存 `maxCacheUb_/sumCacheUb_/expCache`（或 wsMeta exp）按 `slot=t%RING_SLOTS` 寻址的逻辑**完全保持第一轮**——它本就是 ring-slot 设计，错位发射后 `slot` 取值序列不变（仍是 0,1,2,0,1,...），无需改寻址。
3. Vec1↔Vec2 之间的跨块依赖通过 `prevSlot=(slot+RING_SLOTS-1)%RING_SLOTS` 读上一块状态，这套逻辑**第一轮已正确**，错位发射不影响（因为 Vec2(o) 在 t=o+PRELAUNCH 拍执行时，Vec1(o) 早在 t=o 拍已产出状态，时序天然满足）。

---

## 5. 多缓冲下的 buffer 尺寸表与容量校验（必须 ≤ L1 512KB / L0A,L0B 64KB / L0C 128KB / UB 192KB）

> 数据类型以 half/bf16 计（sizeof=2，C0=16）。tiling 取值用第一轮实际值：典型 `dim=128`、`s2BaseSize=512`、`mBaseSize=512`（短 Q≤16 时 s2BaseSize=1024、mBaseSize 相应缩小，见 tiling.cpp §SelectMBaseSize；下表先按主路径 dim=128 核算，再给短 Q 边界说明）。

### 5.1 Cube 侧 L1 占用（AIC 独享 512KB）

沿用 main 的 L1 buffer 常量（dim≤192 路径）：`Q_L1_ROWS=256, Q_L1_COLS=192, V_L1_ROWS=256, KP_L1_COLS=256`。

| buffer | 深度 | 单槽元素 | 单槽字节 | 总字节 | 说明 |
|:--|:--|:--|:--|:--|:--|
| Q L1 (qBufL1_) | 2 段* | 256×192 | 96KB | **96KB** | TBuf 单缓冲，但留 2 段容纳 mBaseSize>256；常驻只读 |
| K/P L1 (kpQueL1_) | **3** | 128×256 | 64KB | **192KB** | TQue 深度 3 配合 PRELAUNCH=2 |
| V L1 (vQueL1_) | **2** | 256×128 | 64KB | **128KB** | TQue 深度 2 ping-pong |
| **L1 合计** | | | | **416KB ≤ 512KB ✓** | 余 96KB |

\* Q 的「2 段」是 main 在 `qBufL1_` 里手工切的两个 `Q_L1_BUF_ELEMS` 区（容纳 mBaseSize 行 = 最多 512 行，分 256+256 两段），不是 TQue 深度。它仍是 TBuf 单缓冲、整任务只载一次、只读，不参与 ping-pong。

> **校验结论：主路径（dim=128, s2BaseSize=512）L1 占用 416KB，安全。** 若 dim>192，main 走 `Q_L1_ROWS` 不变但 Q 段尺寸随 dimAlign 变大，实现者需用实际 dimAlign 重算 Q_L1_COLS；本算子 dim=128 不触发该路径。

### 5.2 Cube 侧 L0 占用

| buffer | 深度 | 单槽 | 单槽字节 | 总字节 | 上限 |
|:--|:--|:--|:--|:--|:--|
| L0A (queL0A_) | 2 | 128×128 half | 32KB | 64KB | ≤64KB ✓（恰满，与第一轮相同） |
| L0B (queL0B_) | 2 | 128×128 half | 32KB | 64KB | ≤64KB ✓ |
| L0C (queL0C_) | **2** | 128×128 float | 64KB | **128KB** | ≤128KB ✓（**第一轮深度1=64KB，本轮翻倍到128KB，恰好占满 L0C**） |

> **L0C 翻倍是本轮唯一逼近上限的改动。** 128×128 float = 64KB，深度 2 = 128KB = L0C 全部容量。**强制约束：实现者不得把 MM1 的 n 微块或 MM2 的 m 微块放大到 >128**，否则单槽 >64KB，深度 2 溢出 L0C。保持 BLOCK_M=BLOCK_N=128 即安全。

### 5.3 Vec 侧 UB 占用（每个 AIV 独享 192KB）

| buffer | 深度 | 字节/槽 | 总字节 | 说明 |
|:--|:--|:--|:--|:--|
| inputQue1_ | 2 | 32KB | 64KB | VEC_QUEUE_BUF_SIZE=32KB |
| outputQue1_ | 2 | 32KB | 64KB | 本轮由 1→2（main 即 2） |
| tmpBuf_ (softmax work) | 1 | 32KB | 32KB | |
| max/sum/exp Cache (RING_SLOTS=3) | — | 3×2KB×3 | 18KB | per-slot 状态 |
| max/sum default + brcb + oPrev | — | — | ≈12KB | 默认值/广播/累加临时 |
| **UB 合计** | | | **≈190KB ≤ 192KB** | **逼近上限，余 ≈2KB** |

> **UB 是 Vec 侧最紧的资源。** outputQue1_ 由深度 1→2 增加 32KB，合计逼近 192KB。**强制约束：实现者若发现 UB 溢出（编译期 InitBuffer 报错），优先把 outputQue1_ 退回深度 1**（牺牲 Vec 输出重叠，保 Cube 流水主收益），不要缩小 softmax 工作 buffer 或状态缓存。**短 Q（s2BaseSize=1024）时** `inputQue1_` 单槽仍为 32KB 上限（VEC_QUEUE_BUF_SIZE 固定），vec1ChunkRows_ 自动缩小以适配，无需改 buffer 尺寸。

---

## 6. 实现者落地清单（把 lessons §4 的 5 类盲区在本轮显式封堵）

| # | 盲区类别 | 本轮显式约定 |
|:--|:--|:--|
| 1 | flag 配对计数 | §4.A 强制不变量：循环上界 `s2OuterBlocks+PRELAUNCH`，每信号 Acquire/Release 各累计 s2OuterBlocks 次；prologue 只 Producer、epilogue 只 Consumer。 |
| 2 | vecDealM==0 补发 flag | §4.A：Vec1 入口先 ConsumerAcquire(S)，若 vecDealM==0 则补 ProducerReleaseMte3(P) 再 return。 |
| 3 | L1 复用同步 | §4.B：删手工 `MTE1_MTE2`，改 TQue FreeTensor→AllocTensor 自动 ping-pong flag。 |
| 4 | L0C 累加序 vs 跨流水序 | §4.B：保留 `PipeBarrier<PIPE_M>`（同 L0C 多 Mmad 累加序），删跨流水 `M_MTE1/MTE1_M/M_FIX/FIX_MTE2`。 |
| 5 | buffer 深度溢出 | §5.2/§5.3：L0C 深度 2=128KB 恰满（禁放大微块>128）；UB≈190KB 逼近上限（溢出则 outputQue1_ 退回深度 1）。 |

**最小改动边界（实现者只动这些文件）：**
- `flash_attention_kernel.h`：替换 Process 内层循环为 §3.2 的错位发射循环。
- `flash_attention_cube.h`：L1 buffer 声明 `TBuf→TQue`（K/P 合并 kpQueL1_ 深度3、V 深度2、L0C 深度2）；MM1/MM2/LoadQ 删手工跨流水 SetWaitFlag，改 TQue Alloc/EnQue/DeQue/Free；MM1 多传一个 `qLocalOffset` 形参以对齐 main 签名（值传 0 即可，保持第一轮 Q 寻址）。
- `flash_attention_vec.h`：outputQue1_ 深度 1→2；Vec1 的 vecDealM==0 分支补发 P_READY。
- `flash_attention_tiling.h/.cpp`：**不改**（PRELAUNCH/RING_SLOTS/ws 尺寸已就绪）。
- `kernel_common.h`、`workspace_queue.h`、`matmul_tile.h`：**不改**。

> **MM1 的 K-load 方案保持第一轮方案 B（裸 LoadData2D，不转置）。** main 用了 `LoadNzL1ToZnL0BWithTranspose`（基线 matmul_tile.h 无此 helper）。本轮**不引入** main 的转置路径，沿用第一轮已验证的 §MM1 裸 LoadData2D 写法，仅把其外层 K-block L1 从 TBuf 改 TQue。这是为隔离变量——本轮只验证「流水/多缓冲」单一增量，不掺入 MM1 加载方式的改动。

---

## 7. 生产者-消费者依赖图（汇总，实现者据此自检流水正确性）

```
拍:        0      1      2      3      4    ...   N      N+1    N+2   (N=s2OuterBlocks)
AIC:    MM1(0) MM1(1) MM1(2) MM1(3) ...        MM1(N-1)
                      MM2(0) MM2(1) MM2(2)...  MM2(N-3) MM2(N-2) MM2(N-1)
AIV:    V1(0)  V1(1)  V1(2)  V1(3)  ...        V1(N-1)
                      V2(0)  V2(1)  V2(2) ...  V2(N-3)  V2(N-2)  V2(N-1)
        └prologue┘    └────────── steady ──────────┘   └─ epilogue ─┘

重叠关系（steady 态，同一拍 t）:
  AIC: MM2(t-2) 的 Mmad/Fixpipe  ‖  MM1(t) 的 K MTE2 搬运   ← Cube 内搬运-计算重叠
  AIC.MM1(t) 产 S[t]  →(SIG_S)→  AIV.V1(t)               ← 跨核生产
  AIV.V1(t) 产 P[t]   →(SIG_P)→  AIC.MM2(t)（在 t+2 拍消费）← 跨核 PRELAUNCH 错位
  AIC.MM2(t) 产 O[t]  →(SIG_O)→  AIV.V2(t)（在 t+2 拍消费）
in-flight slot 数 = PRELAUNCH+1 = 3 = RING_SLOTS（恰好不覆盖）
```

---

## 8. 待验证假设（基于第一轮教训，校验变量范围后给出）

**基线锚定（吸取 lessons §2/§5.1 误判教训）：** 本节所有收益相对**第一轮 1590us 双向两级版（6fc58de）**，不是 2391us 原始基线。

### 8.1 预期收益与已验证结论
- **已验证目标：1590us → 约 977us（相对第一轮 1.63x；相对 2391us 原始基线约 2.41x）。** 这是本轮“多缓冲 + PRELAUNCH 软流水 + 删除手工 barrier”单增量的正确性能锚点。
- **不要再承诺 605us 作为本轮目标。** 605us 是更多增量累计后的上界参考，不是本轮范围内可保证回收的收益。本轮范围声明保留了第一轮 MM1 裸 L0B 加载路径，主动排除了转置加载等差异，因此收益天花板被设计范围锁定在流水子集上。
- **收益归因（已校验变量范围）：** 第一轮的 1.49x 来自「双向两级分块省搬运」，本轮的 1.63x 来自「剩余搬运与计算重叠」。两者是正交维度，可乘性叠加到约 2.4x；这正是 lessons_round2 的核心验证结果。
- **边界说明：** `s2OuterBlocks` 很小时 prologue/epilogue 占比高，且本轮未减少 MM1 转置/片上加载指令开销，因此不能用长 steady 流水的理想情况直接推 605us。后续若要逼近 605us，应拆成独立增量逐轮验证。

### 8.2 主要风险点
1. **跨核 flag 计数失配 → 死锁**（最高风险）：prologue/epilogue 边界、vecDealM==0 分支若漏发/多发 flag，CrossCoreWait 永久阻塞。§4.A 已给强制配对规则，实现者须逐块核对 Acquire/Release 次数。
2. **L0C 深度 2 = 128KB 恰好占满**：任何微块放大 >128 即溢出。§5.2 已约束 BLOCK_M=BLOCK_N=128 不变。
3. **UB ≈190KB 逼近 192KB**：outputQue1_ 升深度后可能溢出；§5.3 给了退化方案（退回深度 1）。
4. **L1 复用正确性**：删手工 `MTE1_MTE2` 后，若 TQue 的 EnQue/DeQue/FreeTensor 配对错误（如 Free 早于全部 MTE1 读），下块 MTE2 会覆盖未读完的 L1。须保证「该块所有 L0 load 发起后才 FreeTensor」。
5. **短 Q（mBaseSize/s2BaseSize 非主路径）边界**：s2BaseSize=1024 时 in-flight slot 的 ws 占用更大，但 ws 在 GM、tiling 已按 RING_SLOTS 预留，不溢出；需验证 vec1ChunkRows_ 自适应缩小后 softmax 状态寻址仍正确（沿用第一轮 stateStride_ 逻辑）。
6. **prologue/epilogue 收益稀释**：s2OuterBlocks 很小（如=2~3）时填充/排空占比高，重叠收益被摊薄——属预期内，非正确性问题。

### 8.3 验证建议（交给主控/debugger）
- 先跑 basic_case 确认**精度 PASS**（流水不改算法，REL 应与第一轮同量级 ~1e-3）。
- 精度过后采性能，与 1590us 对比；若 >1000us，重点查 §8.2 风险 1（flag 错位致流水退化为串行）与风险 6（s2OuterBlocks 过小）。
- 若死锁/TIMEOUT，优先二分定位 prologue/epilogue 的 flag 配对（风险 1）。
- 若性能稳定在 970~1000us 且精度 PASS、无死锁，应记录为本轮有效达标；不要为了追 605us 在同一轮混入 MM1 转置加载、Q→L0A 单指令或 MM2 大块加载，否则收益归因会失真。
