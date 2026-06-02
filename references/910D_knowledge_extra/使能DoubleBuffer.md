# 使能DoubleBuffer<a name="ZH-CN_TOPIC_0000002554328991"></a>

【优先级】中

【描述】执行于AI Core上的指令队列主要包括如下几类，Vector指令队列（V）、Cube指令队列（M）、Scalar指令队列（S）和搬运指令队列（MTE1/MTE2/MTE3）。不同指令队列间的相互独立性和可并行执行特性，是DoubleBuffer优化机制的基石。

以纯Vector计算为例，矢量计算前后的CopyIn、CopyOut过程使用搬运指令队列（MTE2/MTE3），Compute过程使用Vector指令队列（V），不同指令队列可并行执行，意味着CopyIn、CopyOut过程和Compute过程是可以并行的。如[图1](#fig994415385460)所示，考虑一个完整的数据搬运和计算过程，CopyIn过程将数据从Global Memory搬运到Local Memory，Vector计算单元完成Compute计算后，经过CopyOut过程将计算结果搬回Global Memory。

**图 1**  数据搬运与Vector计算过程<a name="fig994415385460"></a>  
<!-- img2text -->
```text
CopyIn
───────────────►

┌──────────────────────┐            ┌──────────────┐
│                      │            │              │
│                      │            │              │
│        Tensor        │     ╭──────┤   Compute    │
│                      │     │      │              │
│                      │     │      │              │
└──────────────────────┘     │      └──────────────┘
                             │
Copyout                      │
◄───────────────             ╰──────◄
```

**图 2**  未使能DoubleBuffer的流水图<a name="fig101953515215"></a>  
<!-- img2text -->
```text
Signals                                      Waves
                                             1            2 us            3 us            4 us            5 us            6 us
┌───────────────────────────────────────┐    ─┬────────────┬───────────────┬───────────────┬───────────────┬───────────────┬─
│ Time                                  │     │            │               │               │               │               │
│ mte2_suv0_busy = 1          CopyIn    │     ┌────────────┐      Tensor1 2 us      ┌────────┐    Tensor1     ┌────────┐
│ vec0_busy      = 0          Compute   │     │            │                         │        │                │        │
│ mte3_suv0_busy = 0          CopyOut   │     └────────────┘                         └────────┘                └────────┘
└───────────────────────────────────────┘                  ┌───────────────────────┐           ┌───────────────────────┐
                                                           │        Tensor1        │           │        Tensor1        │
                                                           └───────────────────────┘           └───────────────────────┘
                                                                            ┌───────────────────────┐           ┌────────
                                                                            │        Tensor1        │           │ Tensor1
                                                                            └───────────────────────┘           └────────
```

说明:
- 左侧信号列表包含:
  - `mte2_suv0_busy=1` → `CopyIn`
  - `vec0_busy=0` → `Compute`
  - `mte3_suv0_busy=0` → `CopyOut`
- 右侧是时间轴波形图，刻度标注为 `1`, `2 us`, `3 us`, `4 us`, `5 us`, `6 us`
- 波形中多处标注 `Tensor1`
- 顶部红色波形对应 `CopyIn`
- 中间黄色波形对应 `Compute`
- 底部青色波形对应 `CopyOut`
- 该图标题对应上下文中的“未使能DoubleBuffer的流水图”

在此过程中，数据搬运与Vector计算串行执行，Vector计算单元不可避免存在资源闲置问题，假设CopyIn、Compute、CopyOut三阶段分别耗时相同均为_t_，则Vector的利用率仅为1/3，等待时间过长，Vector利用率严重不足。

为减少Vector等待时间，使能DoubleBuffer机制将待处理的数据一分为二，例如Tensor1、Tensor2。如[图3](#fig189541246194710)所示，当Vector单元对Tensor1中数据进行Compute计算时，Tensor2数据流可以执行CopyIn的过程；而当Vector切换到计算Tensor2时，Tensor1数据流可以执行CopyOut的过程。由此，数据的进出搬运和Vector计算实现并行执行，Vector闲置问题得以有效缓解。

总体来说，DoubleBuffer是基于MTE指令队列与Vector指令队列的独立性和可并行性，通过将数据搬运与Vector计算并行执行以隐藏大部分的数据搬运时间，并降低Vector指令的等待时间，最终提高Vector单元的利用效率。通过为队列申请内存时设置内存块的个数为2，使能DoubleBuffer，实现数据并行，简单代码示例如下：

```
pipe.InitBuffer(inQueueX, 2, 256);
```

**图 3**  DoubleBuffer机制<a name="fig189541246194710"></a>  
<!-- img2text -->
```text
                CopyIn
──────────────────────→

┌───────────────────────┐        ┌──────────────┐
│                       │   ↗↘   │              │
│        Tensor1        │  ↗  ↘  │   Compute    │
│                       │  ↘  ↗  │              │
├───────────────────────┤        └──────────────┘
│                       │
│        Tensor2        │
│                       │
└───────────────────────┘

←──────────────────────
        CopyOut

                CopyIn
──────────────────────→

←──────────────────────
        CopyOut
```

**图 4**  使能DoubleBuffer的流水图<a name="fig166411527185118"></a>  
<!-- img2text -->
```
Signals                                   Waves
┌───────────────────────────────┐         1 us        2 us        3 us        4 us        5 us        6 us
│ Time                          │         ──────────────────────────────────────────────────────────────────▶
│ mte2_suv0_busy = 0     CopyIn │
│ vec0_ub_write = 0      Compute│         Tensor1      Tensor2      Tensor1      Tensor2      Tensor1      Tensor2
│ mte3_suv0_busy = 0     CopyOut│
└───────────────────────────────┘

CopyIn   : ────────────────┐        ┌──────────────────┐        ┌──────────────────┐
                           └────────┘                  └────────┘                  └──
                           Tensor1                     Tensor2                     Tensor1

Compute  :                 ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
                           │  │  │  │  │  │   │        │  │  │  │  │  │   │        │  │  │  │  │  │   │
                           └──────────────────┘        └──────────────────┘        └──────────────────┘
                           Tensor2                     Tensor1                     Tensor2

CopyOut  : ───────────────────────────────┐        ┌──────────────────┐        ┌──────────────────┐
                                          └────────┘                  └────────┘                  └──
                                          Tensor1                     Tensor2                     Tensor1
```

**需要注意：**

多数情况下，采用DoubleBuffer能有效提升Vector的利用率，缩减算子执行时间。然而，DoubleBuffer机制缓解Vector闲置问题，并不代表它总能带来明显的整体性能提升。例如：

-   当数据搬运时间较短，而Vector计算时间较长时，由于数据搬运在整个计算过程中的时间占比较低，DoubleBuffer机制带来的性能收益会偏小。
-   当原始数据较小且Vector可一次性完成所有数据量的计算时，强行使用DoubleBuffer会降低Vector计算资源的利用率，最终效果可能适得其反。

因此，DoubleBuffer的使用需综合考虑Vector算力、数据量大小、搬运与计算时间占比等多种因素。

【反例】

```
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // 不使能DoubleBuffer,占用的物理空间是1 * sizeSrc0 * sizeof(half)
    // 3个InitBuffer执行后总空间为1 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half)) 
    pipe.InitBuffer(inQueueSrc0, 1, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 1, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, sizeDst0 * sizeof(half));
    }
__aicore__ inline void Process()
{
    // 需要round*2次循环才能处理完数据
    for (uint32_t index = 0; index < round * 2; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```

【正例】

```
__aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
    src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
    dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    // InitBuffer中使用2表示使能DoubleBuffer,占用的物理空间是2 * sizeSrc0 * sizeof(half)
    // 3个InitBuffer执行后总空间为2 * (sizeSrc0 * sizeof(half) + sizeSrc1 * sizeof(half) + sizeDst0 * sizeof(half)) 
    pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));
    pipe.InitBuffer(inQueueSrc1, 2, sizeSrc1 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 2, sizeDst0 * sizeof(half));
    }
__aicore__ inline void Process()
{
    // 开启DoubleBuffer的前提是循环次数 >= 2
    for (uint32_t index = 0; index < round; ++index) {
        CopyIn(index);
        Compute();
        CopyOut(index);
    }
}
```

