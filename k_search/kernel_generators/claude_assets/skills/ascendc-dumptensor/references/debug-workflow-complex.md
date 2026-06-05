# 复杂 Cube+Vector 算子调试流程

适用于复杂 `Cube + Vector` 双核分离算子。下面的推荐点位以 Flash Attention 类算子为例。

核心规则和日志管理见 SKILL.md，此处只列具体步骤和点位。

## 数据流概览

```
[GM: Q/K/V] ──Cube──> [Workspace: S] ──Vector──> [Workspace: P]
                                            │
                                            └─> [Workspace: O_tmp]
                                            │
[Workspace: P] + [GM: V] ──Cube──> [Workspace: O_tmp] ──Vector──> [GM: Output]
```

## 通用调试步骤

1. 缩到最小可复现 case（单 batch、最小 q/kv seq），写入 `debug_case.csv`（仅 1 条用例）
2. 确认失败可复现：`bash scripts/evaluate_ascendc.sh current_task debug`
3. 生成 CPU Golden
4. 从输入开始顺序 dump，只看 `blockidx=0, t=0, slot=0`，`dumpSize=8/16/32`
5. 与 CPU Golden 逐段对比，定位出错段
6. 修复后回归：`debug` → `basic|general|all`

## 推荐调试点位

### Point 1：验证输入数据

如在 Cube 的 LoadQ/LoadK/LoadV 中，LoadNdGmToNzL1 之前：

```cpp
void LoadQ(int bz, int by, int bx) {
    uint64_t qOffset = ...;
    DumpTensor(qGm_[qOffset], 100 + bx, 32);
    // ... LoadNdGmToNzL1
}
```

### Point 2：验证 Cube 输出（Workspace S）

在 Cube 的 ComputeMM1 中，Fixpipe 写入后、CrossCoreSetFlag 前：

```cpp
void ComputeMM1(int bz, int by, int t, int slot) {
    // ... Mmad 和 Fixpipe
    Fixpipe(wsSGm_[wsOffset], cL0, fixParams);
    DumpTensor(wsSGm_[sOffset], 300 + slot, 32);
    CrossCoreSetFlag<0x2, PIPE_FIX>(SIG_S_READY);
}
```

`300 + slot` 异常时排查顺序：Q dump → K dump → MMAD 参数 → 同步信号时机。

### Point 3：验证 Vector 读入（Workspace S）

在 Vector 的 ComputeVec1 中，WaitFlag 后、DataCopy 读入后：

```cpp
void ComputeVec1(int slot, bool isFirst, bool isTailKV) {
    CrossCoreWaitFlag<0x2>(SIG_S_READY);
    DataCopy(sUb, wsSGm_[sOffset], tileSize);
    DumpTensor(sUb, 400 + slot, 32);
    // ... Softmax
}
```

`300+slot` 正确但 `400+slot` 错误 → 查跨核同步、slot/offset、DataCopy 参数。

### Point 4：验证 Softmax 输出（Workspace P）

在 ComputeVec1 中，Softmax 完成、写入 Workspace P 后：

```cpp
DataCopy(wsPGm_[pOffset], pHalf, tileSize);
DumpTensor(wsPGm_[pOffset], 600 + slot, 32);
CrossCoreSetFlag<0x2, PIPE_MTE3>(SIG_P_READY);
```

P 异常时先确认：`400+slot` 的 S 是否正确、tail mask、isFirst 状态、UB→MTE3 同步。

### Point 5：验证 Cube MM2 输出（Workspace O_tmp）

在 Cube 的 ComputeMM2 中，Fixpipe 写入后：

```cpp
void ComputeMM2(int bz, int by, int t, int slot) {
    CrossCoreWaitFlag<0x2>(SIG_P_READY);
    // ... Mmad 和 Fixpipe
    DumpTensor(wsOGm_[wsBase], 310 + slot, 32);
    CrossCoreSetFlag<0x2, PIPE_FIX>(SIG_O_READY);
}
```

### Point 6：验证 Vector 最终输出

在 Vector 的 FinalizeOutputChunk 中，写 GM 前：

```cpp
Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);
DumpTensor(outHalf, 600 + 50 + curBx_, 32);
DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
```

## 编号约定

| 范围 | 位置 | 说明 |
|------|------|------|
| 100-199 | C输入 | 进入 Cube 的输入，如 Q/K/V |
| 200-299 | C中间 | Cube 内中间量，只有需要时才 dump |
| 300-399 | C输出 | Cube 写到 Workspace 的结果，如 S、O_tmp |
| 400-499 | V输入 | Vector 从 Workspace 读入后的数据 |
| 500-599 | V中间 | Softmax、mask、累加状态等 |
| 600-699 | V输出 | Vector 写到 Workspace/GM 前后的结果 |

## 特殊场景调试

### 输出太多时的抽样策略

1. `bx=0, t=0, slot=0`，每个点先 dump 8 或 16 个元素
2. 首核首轮正常时，改看出错 block 对应的 `bx/t/slot`
3. 只怀疑尾块时，直接只保留最后一个 `slot` 和 tail 分支附近的 dump

### 尾块调试

只用一个触发尾块的 case（kvSeqLen 不对齐 BLOCK_N），重点检查：
- `tiling_.tailValid` 值是否正确
- 最后一个 slot 的 S 矩阵
- `isTailKV` 分支的 mask 处理

```cpp
if (isTailKV) {
    DumpTensor(sUb, 400 + slot, 32);
    // mask 处理
    Duplicate(maskUb, SOFTMAX_NEG_INF, BLOCK_N);
    Duplicate(maskUb, 0.0f, tiling_.tailValid);
    // ... Add mask
    DumpTensor(sUb, 500 + slot, 32);
}
```

### 首次 vs 非首次迭代

```cpp
if (isFirst) {
    DumpTensor(softmaxMaxDefaultUb_, 500, 16);
    DumpTensor(softmaxSumDefaultUb_, 510, 16);
} else {
    DumpTensor(inStateUb, 520 + prevSlot, 16);
    DumpTensor(inSumUb, 530 + prevSlot, 16);
}
```

### 跨核同步问题

Cube 侧 SetFlag 前和 Vector 侧 WaitFlag 后分别 dump，对比数据是否一致：

```bash
grep "desc=300\|desc=400" current_task/artifacts/ascendc_dumplog/dump_XX.log
```

同时检查核内流水同步：
- `MTE2 → V`：GM 搬到 UB 后，Vector 使用前是否完成同步
- `V → MTE3`：UB 计算后写回前，MTE3 是否读到了未完成的数据
- `MTE2 → MTE3`：同一 UB 缓冲区搬入搬出交叠时，检查 buffer 生命周期

## 定位完成后

移除所有 DumpTensor，重新验证。
