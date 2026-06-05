# FA-04: Workspace双缓冲与CV数据流 (Workspace Double-Buffering & CV Data Flow)

## Overview

FA类算子的AIC(Cube)和AIV(Vector)核通过GM上的Workspace区域交换数据。每个核在Workspace中拥有独立的 `PRELOAD_NUM=2` 份缓冲，实现双缓冲交替：当前loop写入 `loop%2` 对应的区域，上一loop的数据从另一区域读取。这种设计使得BMM1写入Workspace与Vec1读取Workspace可以并行，消除了Cube→Vector的数据传输瓶颈。

## When to Use

- Cube-Vector双核分离模型，两核通过GM交换中间结果
- 流水线深度≥2，需要当前写入与上一轮读取并行
- 中间结果（BMM1/BMM2输出）无法直接通过L0/UB传递

## Trade-off

- Workspace大小翻倍（PRELOAD_NUM=2）
- GM带宽成为潜在瓶颈（所有中间结果都经过GM）
- 需要精确的偏移计算确保各核区域不重叠

**Source operators**: fused_infer_attention_sink, sparse_flash_attention_gqa, sparse_flash_attention_pioneer

---

## Variant A: 标准四区域Workspace（GQA/Sink）

Source: sparse_flash_attention_gqa, fused_infer_attention_sink

标准FA Workspace包含四个区域，对应四级流水的四个中间结果。

**Workspace布局**：
```
|---mm1ResGm---|---vec1ResGm---|---mm2ResGm---|---vec2ResGm---|---FD区域---|
|  Q×K分数(4B) | Softmax权重(2B)| P×V输出(4B)  | Update值(4B)  |accumOut+LSE|
```

**大小计算**（`fia_tiling_nonquant_sink.cpp:549-563`）：
```cpp
workspaceSize = libapiSize
  + PRE_LOAD_NUM × coreNum × mm1ResSize × 4B     // BMM1结果(fp32)
  + PRE_LOAD_NUM × coreNum × mm1ResSize × 2B     // Vec1结果(fp16/bf16)
  + PRE_LOAD_NUM × coreNum × mm2ResSize × 4B     // BMM2结果(fp32)
  + PRE_LOAD_NUM × coreNum × mm2ResSize × 4B     // Vec2结果(fp32)

// 其中：
mm1ResSize = sInnerSizeAlign × min(gSize*s1Size, mBaseSize)
mm2ResSize = headDimAlign × min(gSize*s1Size, mBaseSize)
```

**双缓冲偏移**（`kernel_nonquant.h:357-394`）：
```cpp
// 每个核的Workspace基地址
mm1ResGm = workspace + offset + aiCoreIdx × dbWorkspaceRatio × mmResUbSize × sizeof(fp32)

// 双缓冲交替
Fixpipe输出偏移 = (info.loop % PRELOAD_NUM) × mmResUbSize
// loop=0 写入区域0，loop=1 写入区域1，loop=2 写入区域0...
```

**生产者→消费者关系**：
```
mm1ResGm:  AIC BMM1 (Fixpipe输出) ──→ AIV Vec1 (Softmax输入)
vec1ResGm: AIV Vec1 (Softmax输出) ──→ AIC BMM2 (CopyToL1输入)
mm2ResGm:  AIC BMM2 (Fixpipe输出) ──→ AIV Vec2 (Update输入)
vec2ResGm: AIV Vec2 (Update输出)  ──→ 输出到GM / FD accumOut
```

Benefit: 双缓冲消除Cube→Vector数据传输等待；每核独立区域避免竞争
Trade-off: Workspace大小 = 2 × coreNum × (mm1+vec1+mm2+vec2)，可能占用大量GM

---

## Variant B: 扩展六区域Workspace（Pioneer）

Source: sparse_flash_attention_pioneer

Pioneer在标准四区域基础上新增两个区域：kvMergeGm（KV合并结果）和kvValidSizeGm（有效MTE2 size），支持Vec0阶段的稀疏KV合并。

**Workspace布局**（`kernel_mla.h:448-478`）：
```
|--mm1Res--|--vec1Res--|--mm2Res--|--vec2Res--|--kvMerge--|--kvValidSize--|--FD--|
| Q×K(4B)  | Softmax(2B)| P×V(4B) | Update(4B)| KV合并(2B)| 有效size(4B) |FD区域|
```

**新增区域大小**（`tiling.cpp:410-412`）：
```cpp
// KV合并缓冲：4块 × 512(S2) × 576(D+Rope) × 2B(half) × coreNum
workspaceSize += 4 * 512 * (512 + 64) * 2 * actCoreNum;

// 有效MTE2 size缓冲：4 × 128份 × 4B(int32) × 2(aiv核数) × coreNum
workspaceSize += 4 * 128 * 4 * (2 * actCoreNum);
```

**数据流扩展**：
```
kvMergeGm:     AIV Vec0 (KV合并) ──→ AIC BMM1 (CopyToL1输入)
kvValidSizeGm: AIV Vec0 (记录有效size) ──→ AIC BMM1 (控制搬运长度)
```

**与标准方案的关键区别**：
1. kvMergeGm是Vec0→AIC的数据通道（标准方案无此通道）
2. 4块缓冲支持Vec0与BMM1的流水重叠
3. kvValidSizeGm记录每个稀疏块的实际有效长度，避免搬运无效数据

Benefit: 支持稀疏KV合并的流水化；有效size控制减少无效数据搬运
Trade-off: 额外的kvMerge workspace占用大量GM（约 4×512×576×2×coreNum ≈ 2.8MB/核）

---

## Variant C: FD Workspace区域（三算子共有）

Source: fused_infer_attention_sink, sparse_flash_attention_gqa, sparse_flash_attention_pioneer

当S2被跨核切分时，FD Workspace存储各核的部分Softmax结果，供归约阶段使用。

**FD Workspace布局**：
```cpp
// Sink/GQA: split_core.cpp
fdParamNums = coreNum × 2 × mBaseSize  // 每核可能有头归约和尾归约

accumOut:   fdParamNums × headDimAlign × 4B   // 归约中间结果(fp32)
logSumExp:  2 × fdParamNums × 8 × 4B          // Max和Sum各一份(fp32)

// Pioneer: tiling.cpp:329-331
accumOutSize = aicNum × 2 × n2Size × mBaseSize × headDimAlign
logSumExpSize = 2 × aicNum × 2 × n2Size × mBaseSize × (32/4)
```

**FD数据流**：
```
FA阶段(各核独立):
  AIV Vec2 ──→ accumOutGm[核i]     // 部分加权和
  AIV Vec1 ──→ lseMaxFdGm[核i]     // 部分Max
  AIV Vec1 ──→ lseSumFdGm[核i]     // 部分Sum

FD归约阶段(Vector核):
  accumOutGm[核0..N] ──→ AIV FD ──→ 最终输出
  lseMaxFdGm[核0..N] ──→ AIV FD ──→ 全局Max
  lseSumFdGm[核0..N] ──→ AIV FD ──→ 全局Sum
```

Benefit: 支持S2跨核切分，提高大序列长度的并行度
Trade-off: FD workspace额外占用 ≈ fdParamNums × (headDimAlign + 16) × 4B

---

## Workspace设计要点总结

| 设计要点 | 说明 |
|---------|------|
| 双缓冲交替 | `loop % PRELOAD_NUM` 控制写入区域，消除读写冲突 |
| 每核独立 | `aiCoreIdx × dbWorkspaceRatio × size` 确保核间不重叠 |
| 类型精度 | BMM结果用fp32(4B)，Vec1结果用fp16/bf16(2B)，节省带宽 |
| 对齐约束 | mm1ResSize/mm2ResSize 按16对齐，确保Fixpipe输出对齐 |
| FD条件分配 | 仅在splitKVFlag=true时分配FD workspace，避免浪费 |
| 生产消费配对 | 每个区域有明确的生产者和消费者，通过CrossCore flag同步 |
