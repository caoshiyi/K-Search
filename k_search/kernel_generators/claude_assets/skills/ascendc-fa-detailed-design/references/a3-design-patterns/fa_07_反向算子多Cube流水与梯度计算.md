# FA-07: 反向算子多Cube流水与梯度计算 (Backward Multi-Cube Pipeline & Gradient Computation)

## Overview

FA类反向算子需要计算dQ、dK、dV三个梯度，涉及多次矩阵乘法和Softmax梯度计算。与前向算子的BMM1→Vec1→BMM2→Vec2四级流水不同，反向算子采用多Cube阶段流水：FASGE使用Cube1/Cube2/Cube3三级流水，SFAGE使用Cube12/Cube345两阶段流水。AIC核负责矩阵乘法，AIV核负责Softmax梯度和稀疏Gather/Scatter操作。

## When to Use

- 反向传播需要计算dQ、dK、dV三个梯度
- 计算流程包含多次矩阵乘法（≥3次Cube操作）
- 需要Softmax梯度作为中间结果参与后续Cube计算

## Trade-off

- 多Cube阶段增加Workspace中间结果存储（mm1/mm2/dQ/dK/dV均需FP32缓冲）
- 反向流水深度受限于Cube阶段间的数据依赖
- 跨核同步复杂度高于前向算子

**Source operators**: flash_attention_score_grad_enhance, sparse_flash_attention_grad_enhance

---

## Variant A: Cube1/Cube2/Cube3三级流水（FASGE）

Source: flash_attention_score_grad_enhance

FASGE采用三级Cube流水设计：Cube1计算Q×K^T和dY×V^T两个矩阵乘，Cube2利用Cube1结果计算dQ，Cube3利用Cube1结果计算dK和dV。通过双缓冲`taskId%2`实现Cube1加载与Cube2/Cube3计算的重叠。

**计算路径**：
```
Cube1: mm1 = dY × V^T,  mm2 = Q × K^T
Cube2: dQ = mm1 × K × softmax_grad / sqrt(d)
Cube3: dK = mm2^T × dY × softmax_grad / sqrt(d),  dV = Attention^T × dY
```

**流水编排**（`flash_attention_score_grad_enhance_s1s2_bn2gs1s2_basic.h:123-164`）：
```cpp
while (running) {
    cubeAddrInfo[taskId % 2].taskId = taskId;  // 双缓冲索引
    cubeAddr.addr_mapping(&cubeAddrInfo[taskId % 2], &cube3AddrInfo[taskId % 2]);

    if (cubeAddrInfo[taskId % 2].blockLength > 0) {
        // Cube1: 当前任务 — 计算mm1(dY×V^T)和mm2(Q×K^T)
        cubeOp.Cube1Process(cubeAddrInfo[taskId % 2], mm1WorkSpaceAddr, mm2WorkSpaceAddr);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
    }

    if (taskId > 0 && cubeAddrInfo[(taskId - 1) % 2].blockLength > 0) {
        // Cube2: 上一任务 — 利用mm1计算dQ
        cubeOp.Cube2Process(cubeAddrInfo[(taskId-1) % 2], mm1WorkSpaceAddr, key, dqWorkSpaceAddr);
        // Cube3: 上一任务 — 利用mm1/mm2计算dK、dV
        cubeOp.Cube3Process(cube3AddrInfo[(taskId-1) % 2], mm1WorkSpaceAddr, query, dkWorkSpaceAddr);
        cubeOp.Cube3Process(cube3AddrInfo[(taskId-1) % 2], mm2WorkSpaceAddr, dy, dvWorkSpaceAddr);
    }
    taskId++;
}
```

**跨核同步**：
```cpp
// AIC → AIV
CUBE2VEC   // Cube1完成后通知Vector处理Softmax梯度
CUBE2POST  // 所有Cube完成后通知Post处理（Cast FP32→FP16）

// AIV → AIC
VEC2CUBE   // Vector完成Softmax梯度后通知Cube2/3开始
```

**AIV核职责**（`flash_attention_score_grad_enhance_sfmg.h:246-258`）：
```cpp
// Softmax梯度计算：sfmg = dy * attention（逐元素乘）
Cast(sfmgClc1, input1Buf, RoundMode::CAST_NONE, calcSize);  // dy Cast to FP32
Cast(sfmgClc2, input2Buf, RoundMode::CAST_NONE, calcSize);  // attention Cast to FP32
// 计算 sfmg 写入 sfmgWorkspace，供Cube2/3使用
```

**流水时序**：
```
Task 0:  [Cube1_0]──────────────────────────────────────
Task 1:              [Cube1_1]  [Cube2_0 + Cube3_0]────
Task 2:                          [Cube1_2]  [Cube2_1 + Cube3_1]
         AIC: ───────────────────────────────────────────→
         AIV: ──[Pre]──[Sfmg]──[Main]──────────[Post]──→
```

Benefit: Cube1与Cube2/3通过双缓冲重叠，隐藏加载延迟；dQ和dK/dV可并行计算
Trade-off: mm1/mm2中间结果需要双缓冲Workspace（`matmulSize × DB_NUM × coreNum × 4B`）

---

## Variant B: Cube12/Cube345两阶段流水 + Gather/Scatter（SFAGE）

Source: sparse_flash_attention_grad_enhance

SFAGE将反向计算分为两个Cube阶段：Cube12计算P=Q@K^T和dP=dO@V^T，Cube345计算dV、dS、dQ、dK。AIV核负责稀疏KV的Gather（从GM按TopK索引搬入）和梯度的Scatter（累加回GM）。

**计算路径**：
```
Cube12: P = Q × K^T,  dP = dO × V^T
Vec:    dS = P * (dP - SoftmaxGrad(dO, O))
Cube345: dQ = dS × K / sqrt(d),  dK = dS^T × Q / sqrt(d),  dV = P^T × dO
```

**流水编排**（`sparse_flash_attention_grad_enhance_bs1_basic.h:220-260`）：
```cpp
for (int32_t i = 0; i < processBS1ByCore; i++) {
    for (n2Index = 0; n2Index < dimN2; n2Index++) {
        for (blkCntOffset = 0; blkCntOffset < actualSelectedBlockCount;
             blkCntOffset += selectedCountOffset) {
            // AIC: Cube12 → Cube345
            // AIV: Gather → Scatter
            CubeCompute(cubeOp);
        }
    }
}
```

**Cube-Vector同步**（`sparse_flash_attention_grad_enhance_bs1_basic.h`）：
```cpp
// 4个同步标志（Ping-Pong双缓冲）
CUBE_WAIT_VEC_PING = 0;  // Cube等待Vector完成Ping缓冲的Gather
CUBE_WAIT_VEC_PONG = 1;  // Cube等待Vector完成Pong缓冲的Gather
VEC_WAIT_CUBE_PING = 2;  // Vector等待Cube完成Ping缓冲的计算
VEC_WAIT_CUBE_PONG = 3;  // Vector等待Cube完成Pong缓冲的计算
```

**确定性模式**：
```cpp
if constexpr (DETERMINISTIC_ENABLE) {
    // 额外同步点确保ScatterAdd结果一致
    CrossCoreSetFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
    CrossCoreWaitFlag<0, PIPE_FIX>(SCATTER_CUBE_SYNC_FLAG);
}
```

**与Variant A的关键区别**：
1. Cube阶段分为Cube12和Cube345两组，而非Cube1/2/3三级
2. AIV核负责稀疏Gather/Scatter，而非仅做Softmax梯度
3. 支持确定性模式（DETERMINISTIC_ENABLE），增加额外同步保证结果一致
4. 使用selectedK/V workspace存储Gather后的稀疏KV数据

Benefit: Gather/Scatter与Cube计算通过Ping-Pong重叠；确定性模式保证多次运行结果一致
Trade-off: Gather/Scatter增加GM带宽消耗；确定性模式的额外同步降低吞吐

---

## 反向流水设计对比总结

| 特性 | FASGE (Cube1/2/3) | SFAGE (Cube12/345) |
|------|-------------------|---------------------|
| Cube阶段数 | 3级（Cube1→Cube2→Cube3） | 2组（Cube12→Cube345） |
| 流水深度 | 双缓冲 taskId%2 | Ping-Pong双缓冲 |
| AIV核职责 | Softmax梯度 + Pre/Post处理 | 稀疏Gather/Scatter |
| 跨核同步flag | 3个（CUBE2VEC/VEC2CUBE/CUBE2POST） | 4个（Ping/Pong × Cube/Vec） |
| 中间结果存储 | mm1/mm2 workspace（FP32） | selectedK/V + mm12 workspace |
| 确定性模式 | 无 | 支持（DETERMINISTIC_ENABLE） |
| 稀疏支持 | 无（密集KV） | Gather/Scatter稀疏KV |
| Workspace开销 | 16MB预留 + dQ/dK/dV + mm1/mm2 | 32MB预留 + selectedK/V + dQ/dK/dV + ScatterAdd |
| 适用场景 | 标准FA反向（密集注意力） | 稀疏FA反向（TopK注意力） |
