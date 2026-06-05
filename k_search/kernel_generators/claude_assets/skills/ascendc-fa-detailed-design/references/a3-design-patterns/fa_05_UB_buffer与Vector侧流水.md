# FA-05: UB Buffer与Vector侧流水 (UB Buffer & Vector Pipeline)

## Overview

FA类算子的AIV(Vector)核使用UB(Unified Buffer)完成Softmax、Flash Update、CopyOut等计算。不同算子根据流水级数和preLoadNum选择不同的Queue/Buffer方案：GQA采用标准VECIN/VECOUT双缓冲队列，Sink在此基础上增加Softmax状态的preLoadNum缩放，Pioneer因新增Vec0阶段引入额外的C2V/V2C队列和KV合并缓冲。

## When to Use

- Vector核需要处理BMM1/BMM2的中间结果（Softmax、Flash Update）
- 需要跨S2迭代累积Softmax状态（max/sum/exp）
- preLoadNum > 1时，Softmax状态buffer需要按preLoadNum倍数分配
- 流水级数增加时，需要新增Queue类型支持额外的CV数据通道

## Trade-off

- UB总容量有限（通常256KB），Queue和Static Buffer需要精确规划
- preLoadNum增大可提升流水深度，但Softmax状态buffer成倍增长
- 新增Queue类型增加同步复杂度

**Source operators**: fused_infer_attention_sink, sparse_flash_attention_gqa, sparse_flash_attention_pioneer, flash_attention_score_enhance, lightning_indexer_enhance

---

## Variant A: 标准VECIN/VECOUT双缓冲（GQA）

Source: sparse_flash_attention_gqa

GQA采用经典的VECIN/VECOUT队列模型，BMM1/BMM2结果通过VECIN队列输入，Softmax/Update结果通过VECOUT队列输出。Softmax状态buffer按preLoadNum=2分配。

**Pipeline Queue**（`service_vector_gqa.h`）：
```cpp
// Pipeline Queue定义
TQue<QuePosition::VECIN, BUFFER_NUM>  inputQue1;   // BMM1结果输入（Softmax）
TQue<QuePosition::VECIN, BUFFER_NUM>  inputQue2;   // BMM2结果输入（Update）
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue1;  // Softmax结果输出
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue2;  // 最终结果输出

// 初始化 — 大小由mm1ResSize/mm2ResSize决定
pipe->InitBuffer(inputQue1, BUFFER_NUM, mm1ResUbSize * sizeof(float));
pipe->InitBuffer(inputQue2, BUFFER_NUM, mm2ResUbSize * sizeof(float));
pipe->InitBuffer(outputQue1, BUFFER_NUM, mm1ResUbSize * sizeof(half));
pipe->InitBuffer(outputQue2, BUFFER_NUM, mm2ResUbSize * sizeof(float));
```

**Static Buffer — Softmax状态**（`service_vector_gqa.h`）：
```cpp
// Softmax状态 × preLoadNum
TBuf<> softmaxMaxBuf;   // 每行max值，大小 = mBaseSize × preLoadNum × sizeof(float)
TBuf<> softmaxSumBuf;   // 每行sum值，大小 = mBaseSize × preLoadNum × sizeof(float)
TBuf<> softmaxExpBuf;   // 每行exp和，大小 = mBaseSize × preLoadNum × sizeof(float)

// preLoadNum=2时，buffer大小翻倍
pipe->InitBuffer(softmaxMaxBuf, mBaseSize * PRE_LOAD_NUM * sizeof(float));
pipe->InitBuffer(softmaxSumBuf, mBaseSize * PRE_LOAD_NUM * sizeof(float));
```

**Static Buffer — 临时计算**：
```cpp
TBuf<> tmpBuf1;  // 通用临时空间（Cast、Muls等中间结果）
TBuf<> tmpBuf2;  // Softmax计算临时空间（Exp、ReduceMax等）
```

**UB容量规划**：
```
Pipeline Queue:  2 × (mm1ResUbSize×4B + mm2ResUbSize×4B + mm1ResUbSize×2B + mm2ResUbSize×4B)
Softmax状态:     3 × mBaseSize × preLoadNum × 4B
临时Buffer:      2 × tmpBufSize
总计:            约 160-200KB（取决于mBaseSize和sInnerSize）
```

Benefit: 经典Queue模型，逻辑清晰；BUFFER_NUM=2实现CopyIn/Compute/CopyOut三级流水
Trade-off: Queue占用UB大头，大mBaseSize或大sInnerSize时容量紧张

---

## Variant B: preLoadNum缩放Softmax状态（Sink）

Source: fused_infer_attention_sink

Sink算子在GQA基础上，因Sink token的存在需要额外的Softmax状态管理。Sink token的Softmax max/sum需要独立维护，且preLoadNum对状态buffer的缩放更显著。

**Pipeline Queue**（`fia_service_vector_nonquant_sink.h`）：
```cpp
// 与GQA类似的Queue结构
TQue<QuePosition::VECIN, BUFFER_NUM>  mm1ResQue;    // BMM1结果
TQue<QuePosition::VECIN, BUFFER_NUM>  mm2ResQue;    // BMM2结果
TQue<QuePosition::VECOUT, BUFFER_NUM> vec1ResQue;   // Softmax输出
TQue<QuePosition::VECOUT, BUFFER_NUM> vec2ResQue;   // Update输出
```

**Softmax状态 — Sink扩展**（`fia_service_vector_nonquant_sink.h`）：
```cpp
// 标准Softmax状态 × preLoadNum
TBuf<> softmaxMaxBuf;   // mBaseSize × PRE_LOAD_NUM × sizeof(float)
TBuf<> softmaxSumBuf;   // mBaseSize × PRE_LOAD_NUM × sizeof(float)

// Sink token的Softmax状态（独立维护）
// Sink部分的max/sum在所有S2循环中持续累积，不随loop重置
// 因此需要额外的buffer保存Sink部分的中间状态
TBuf<> sinkSoftmaxMaxBuf;  // sinkNumber × mBaseSize × sizeof(float)
TBuf<> sinkSoftmaxSumBuf;  // sinkNumber × mBaseSize × sizeof(float)
```

**与GQA的关键区别**：
1. Sink token的Softmax状态不随S2 loop重置，需要跨loop累积
2. 额外的sinkSoftmax buffer占用UB空间
3. Softmax计算需要区分Sink部分和Normal部分

Benefit: 独立的Sink状态管理确保Sink token的Softmax精度；与Normal部分解耦
Trade-off: 额外的Sink状态buffer增加UB压力；Softmax逻辑分支增多

---

## Variant C: Vec0阶段扩展Queue（Pioneer）

Source: sparse_flash_attention_pioneer

Pioneer新增Vec0阶段（KV合并），需要额外的Queue类型支持Vec0→AIC的数据通道。UB buffer设计需要同时服务Vec0/Vec1/Vec2三个Vector阶段。

**Pipeline Queue — 扩展**（`service_vector_mla.h`）：
```cpp
// 标准Queue（Vec1/Vec2使用）
TQue<QuePosition::VECIN, BUFFER_NUM>  mm1ResQue;    // BMM1结果 → Vec1
TQue<QuePosition::VECIN, BUFFER_NUM>  mm2ResQue;    // BMM2结果 → Vec2
TQue<QuePosition::VECOUT, BUFFER_NUM> vec1ResQue;   // Vec1 Softmax输出
TQue<QuePosition::VECOUT, BUFFER_NUM> vec2ResQue;   // Vec2 Update输出

// Vec0专用Queue（KV合并）
TQue<QuePosition::VECIN, BUFFER_NUM>  kvMergeInQue;   // 稀疏KV数据输入
TQue<QuePosition::VECOUT, BUFFER_NUM> kvMergeOutQue;  // 合并后KV输出到Workspace
```

**Static Buffer — Vec0 KV合并**：
```cpp
// KV合并临时空间
TBuf<> kvMergeTmpBuf;    // 稀疏KV块的拼接/排序临时空间
TBuf<> kvValidMaskBuf;   // 有效KV块的mask标记

// Softmax状态（与GQA/Sink类似）
TBuf<> softmaxMaxBuf;    // mBaseSize × PRE_LOAD_NUM × sizeof(float)
TBuf<> softmaxSumBuf;    // mBaseSize × PRE_LOAD_NUM × sizeof(float)
```

**UB时分复用策略**：
```
Vec0阶段: kvMergeInQue + kvMergeTmpBuf + kvMergeOutQue  （Vec1/Vec2 Queue未使用）
Vec1阶段: mm1ResQue + softmax bufs + vec1ResQue          （Vec0 Queue未使用）
Vec2阶段: mm2ResQue + tmpBuf + vec2ResQue                （Vec0/Vec1 Queue未使用）

关键：Vec0/Vec1/Vec2不同时执行，Queue可时分复用UB空间
```

**与GQA/Sink的关键区别**：
1. 新增kvMergeInQue/kvMergeOutQue支持Vec0阶段
2. Vec0/Vec1/Vec2时分复用UB，总容量不显著增加
3. kvValidMaskBuf用于稀疏KV的有效性判断

Benefit: 时分复用避免UB容量爆炸；Vec0阶段的KV合并在UB内完成，减少GM带宽
Trade-off: 时分复用要求Vec0/Vec1/Vec2严格串行，增加同步约束；Queue初始化逻辑更复杂

---

## UB Buffer设计对比总结

| 特性 | GQA (标准Queue) | Sink (Sink状态扩展) | Pioneer (Vec0扩展) |
|------|-----------------|--------------------|--------------------|
| Queue类型数 | 4 (2×VECIN + 2×VECOUT) | 4 (2×VECIN + 2×VECOUT) | 6 (3×VECIN + 3×VECOUT) |
| BUFFER_NUM | 2 (pingpong) | 2 (pingpong) | 2 (pingpong) |
| Softmax状态buffer | 3个 × preLoadNum | 3个 × preLoadNum + Sink独立状态 | 3个 × preLoadNum |
| 额外Static Buffer | tmpBuf × 2 | tmpBuf × 2 + sinkSoftmax × 2 | tmpBuf × 2 + kvMergeTmp + kvValidMask |
| UB时分复用 | 无（Vec1/Vec2交替） | 无（Vec1/Vec2交替） | 有（Vec0/Vec1/Vec2时分） |
| preLoadNum影响 | Softmax状态 ×N | Softmax状态 ×N | Softmax状态 ×N |
| UB容量压力 | 中等 | 中高（Sink状态额外占用） | 中等（时分复用缓解） |
| 适用场景 | 标准GQA注意力 | 带Sink token的注意力 | 稀疏KV + MLA注意力 |

---

## Variant D: TBuf静态分配 + Softmax双缓冲（FAE）

Source: flash_attention_score_enhance

FAE不使用TQue队列模型，而是采用TBuf静态分配方式管理UB。每个功能模块（Mask、PSE、Softmax、BMM结果）拥有独立的TBuf，Softmax状态使用数组双缓冲`[0]/[1]`实现Ping-Pong。

**TBuf分配**（`flash_attention_score_enhance_s1s2_bn2gs1.h:507-527`）：
```cpp
// Mask缓冲
pipe->InitBuffer(this->maskTBufPing, stage1AttenSize);         // 9KB - Attention Mask
pipe->InitBuffer(this->maskTBufPong, maskTBufPongSize);        // 16KB - Dropout Mask

// PSE缓冲
pipe->InitBuffer(this->pseTBuf, 16384);                        // 16KB - Position Encoding

// BMM结果缓冲（Ping/Pong）
pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // 32KB - BMM1结果Ping
pipe->InitBuffer(this->stage1PongBuf, stage1PongSize);         // 35KB - BMM1结果Pong
pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // 32KB - BMM2中间结果
pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // 32KB - 通用缓冲

// Softmax状态（数组双缓冲）
pipe->InitBuffer(this->softmaxSumBuf[0], s1BaseSize * 4 * softmaxReduceSize);
pipe->InitBuffer(this->softmaxSumBuf[1], s1BaseSize * 4 * softmaxReduceSize);
pipe->InitBuffer(this->softmaxMaxBuf, s1BaseSize * 4 * softmaxReduceSize);
pipe->InitBuffer(this->softmaxExpBuf[0], s1BaseSize * 4 * softmaxReduceSize);
pipe->InitBuffer(this->softmaxExpBuf[1], s1BaseSize * 4 * softmaxReduceSize);
```

**UB容量规划**：
```
Mask:          9KB + 16KB = 25KB
PSE:           16KB
BMM结果:       32KB + 35KB + 32KB + 32KB = 131KB
Softmax状态:   5 × (s1BaseSize × 4 × 8) ≈ 40KB (s1BaseSize=128, reduceSize=8)
总计:          约 212KB
```

**与GQA（Variant A）的关键区别**：
1. 使用TBuf静态分配而非TQue队列，无EnQue/DeQue开销
2. Softmax状态用数组`[0]/[1]`双缓冲，而非preLoadNum缩放
3. 额外的Mask和PSE专用TBuf，支持Attention Mask和Position Encoding
4. 支持多种数据类型（FP16/BF16/FP8），通过stage1PingBuf/PongBuf适配

Benefit: TBuf静态分配避免队列管理开销；独立的Mask/PSE缓冲支持丰富的注意力变体
Trade-off: 静态分配灵活性低于队列模型；多个独立TBuf增加UB碎片化风险

---

## Variant E: TopK专用TBuf（LIE）

Source: lightning_indexer_enhance

LIE的AIV核执行TopK选取和排序，UB缓冲设计围绕排序和索引操作，与FA类算子的Softmax/Update缓冲完全不同。

**UB缓冲分配**（`lightning_indexer_enhance_service_vector.h:68-79`）：
```cpp
TQue<QuePosition::VECOUT, 1> outQueue_;        // 输出队列（单缓冲）
TBuf<TPosition::VECCALC> sortOutBuf_;          // TopK排序结果
TBuf<TPosition::VECCALC> tmpBuf_;              // 临时缓冲
TBuf<TPosition::VECCALC> indexBuf_;            // 索引缓冲
TBuf<TPosition::VECCALC> reduceOutBuf_;        // 归约输出
TBuf<TPosition::VECCALC> paramBuf_;            // 参数缓冲

// sortOutBuf大小与sparseCount相关
pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));
```

**与FA类Variant的区别**：
1. 无Softmax状态缓冲（不做Softmax）
2. 输出队列仅单缓冲（BUFFER_NUM=1），因TopK输出量小
3. sortOutBuf大小与sparseCount（TopK的K值）成正比
4. 所有TBuf使用VECCALC位置，不区分VECIN/VECOUT

Benefit: 缓冲设计精确匹配TopK操作需求；单缓冲输出减少UB占用
Trade-off: 不支持Softmax/Update等FA标准操作

---

## UB Buffer设计对比总结（扩展）

| 特性 | GQA (标准Queue) | Sink (Sink扩展) | Pioneer (Vec0扩展) | FAE (TBuf静态) | LIE (TopK专用) |
|------|-----------------|--------------------|--------------------|----------------|----------------|
| 缓冲模型 | TQue队列 | TQue队列 | TQue队列 | TBuf静态 | TBuf+TQue混合 |
| Queue类型数 | 4 | 4 | 6 | 0（纯TBuf） | 1（VECOUT） |
| Softmax状态 | 3个×preLoadNum | 3个×preLoadNum+Sink | 3个×preLoadNum | 5个TBuf(数组双缓冲) | 无 |
| 特殊缓冲 | tmpBuf×2 | sinkSoftmax×2 | kvMergeTmp+kvValidMask | maskTBuf+pseTBuf | sortOut+index+reduce |
| UB时分复用 | 无 | 无 | 有(Vec0/1/2) | 无 | 无 |
| UB容量压力 | 中等 | 中高 | 中等 | 中高(~212KB) | 低 |
| 适用场景 | 标准GQA | 带Sink token | 稀疏KV+MLA | 多模式训练FA | TopK索引选取 |
