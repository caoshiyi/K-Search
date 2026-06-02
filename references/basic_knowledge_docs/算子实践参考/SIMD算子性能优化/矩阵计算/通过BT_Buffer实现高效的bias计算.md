# 通过BT Buffer实现高效的bias计算

**页面ID:** atlas_ascendc_best_practices_10_0021  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0021.html

---

【优先级】高

【描述】算子中进行带bias的矩阵乘计算时，可将bias数据搬运至C2(Bias Table Buffer)上，调用一次Mmad接口实现矩阵乘加bias的计算，或者直接调用Matmul高阶API完成功能。相比于先将矩阵乘的结果从CO1(L0C)搬运到GM上，再搬运到UB上进行加bias的过程，减少了数据搬运的次数，可提升内存使用效率。数据流图对比如下：

**图1 **反例数据流图
<!-- img2text -->
```
┌────┐                                                        ┌────┐
│ GM │                                                        │ GM │
└─┬──┘                                                        └─┬──┘
  │                                                              ▲
  │                                                              │
  │   ┌────────────────────────────────────────────────────────┐   │
  ├──→│                          A矩阵                         │   │
  │   │   ┌──────┐      ┌──────┐                               │   │
  │   │   │  A1  │ ───→ │  A2  │ ──┐                           │   │
  │   │   └──────┘      └──────┘   │                           │   │
  │   │                            ├──→ Mmad ─→ ○ ─→ ┌──────┐  │   │
  ├──→│                          B矩阵 │              │ CO1  │ ─┼──→│
  │   │   ┌──────┐      ┌──────┐   │              └──────┘  │   │
  │   │   │  B1  │ ───→ │  B2  │ ──┘                           │   │
  │   │   └──────┘      └──────┘                               │   │
  │   │                A*B                                     │   │
  │   └────────────────────────────────────────────────────────┘   │
  │                                                                │
  │   ┌────────────────────────────────────────────────────────┐   │
  ├──→│ Bias                                                   │   │
  │   │   │                                                    │   │
  │   │   ↓                                                    │   │
  │   │ ┌───────┐                    Add ─→ ○ ─────────→ ┌───────┐│
  │   │ │ VECIN │ ─────────────────────────────────────→ │ VECOUT│├──→
  │   │ └───────┘                                        └───────┘│
  │   └────────────────────────────────────────────────────────┘   │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```**图2 **正例数据流图
<!-- img2text -->
```
图1 反例数据流图

┌──────┐                                                       ┌──────┐
│  GM  │                                                       │  GM  │
└──────┘                                                       └──────┘
    │                                                               ▲
    │                                                               │
    │   ┌─────────────────────────────────────────────────────────┐   │
    ├──→│ A矩阵                                                   │   │
    │   │                                                         │   │
    │   │   ┌──────┐      ┌──────┐                                │   │
    │   └──→│  A1  │ ───→ │  A2  │ ───┐                           │   │
    │       └──────┘      └──────┘    │                           │   │
    │                                  │                           │   │
    ├──→│ B矩阵                         │                           │   │
    │   │                              ▼                           │   │
    │   │   ┌──────┐      ┌──────┐   Mmad      ┌──────┐            │   │
    │   └──→│  B1  │ ───→ │  B2  │ ───○──────→ │ CO1  │ ───────────┼──→│
    │       └──────┘      └──────┘    ▲        └──────┘            │
    │                                  │                           │
    ├──→│ Bias矩阵                      │                           │
    │   │                              │                           │
    │   │   ┌──────┐      ┌────────┐   │                           │
    │   └──→│  C1  │ ───→ │ C2(BT) │ ──┘                           │
    │       └──────┘      └────────┘                               │
    │   └─────────────────────────────────────────────────────────┘
```

【反例】

该算子进行带bias的矩阵乘计算时，过程如下：

- 将矩阵乘的计算结果从CO1(L0C)搬运到workspace(GM)上；
- 从workspace搬运到UB上；
- 在UB上进行加bias的运算；
- 最后将结果搬运到GM。

当循环n次该计算过程，则分别增加了n次CO1->workspace、workspace->UB的搬运。

```
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *bias, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ float *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueBias, 1, n * sizeof(float));
        pipe.InitBuffer(inQueueSrc0, 1, cSize * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, cSize * sizeof(float));

    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
        CopyIn1();
        Compute1();
        CopyOut1();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();

        Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        DataCopy(a1Local, aGM, dataCopyA1Params);

        Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        DataCopy(b1Local, bGM, dataCopyB1Params);
        // 将bias搬运到UB
        DataCopy(biasLocal, biasGM, n);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void SplitA()
    {
        ...
    }
    __aicore__ inline void SplitB()
    {
        ...
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        // 矩阵乘
        Mmad(c1Local, a2Local, b2Local, mmadParams); // m*n
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
        xGm.SetGlobalBuffer((__gm__ float *)(usrWorkspace));
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        // 将矩阵乘的计算结果从CO1搬运到workspace
        Fixpipe(xGm, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyIn1()
    {
        PipeBarrier<PIPE_ALL>();
        // 将矩阵乘的计算结果从workspace搬运到UB
        LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        DataCopy(src0Local, xGm, cSize);
        inQueueSrc0.EnQue(src0Local);
    }
    __aicore__ inline void Compute1()
    {
        LocalTensor<float> src0Local = inQueueSrc0.DeQue<float>();
        LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        BinaryRepeatParams addRepeatParams;
        addRepeatParams.dstRepStride = 8;
        addRepeatParams.src0RepStride = 8;
        addRepeatParams.src1RepStride = 0;
        // 加bias的运算
        Add(dstLocal, src0Local, biasLocal, 32, m, addRepeatParams);
        outQueueDst.EnQue<float>(dstLocal);
        inQueueSrc0.FreeTensor(src0Local);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut1()
    {
        ...
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::VECIN, 1> inQueueBias;
    TQue<TPosition::VECIN, 1> inQueueSrc0;
    TQue<TPosition::VECOUT, 1> outQueueDst;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<float> cGM;
    GlobalTensor<float> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;  
...
```

【正例】

该算子进行带bias的矩阵乘计算时，先将bias搬运到BT上，调用一次Mmad接口实现矩阵乘加bias的计算。

```
...
// 该样例仅做示例说明，非完整代码，省略了部分同步控制代码
public:
    __aicore__ inline KernelSample()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *bias, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ float *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueC1, 1, n * sizeof(float));
        pipe.InitBuffer(outQueueC2, 1, n * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        SplitBias();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        LocalTensor<float> bias1Local = inQueueC1.AllocTensor<float>();

        Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        DataCopy(a1Local, aGM, dataCopyA1Params);

        Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        DataCopy(b1Local, bGM, dataCopyB1Params);
        // 将bias从GM搬运到L1
        DataCopy(bias1Local, biasGM, n);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueC1.EnQue(bias1Local);
    }
    __aicore__ inline void SplitA()
    {
        ...
    }
    __aicore__ inline void SplitB()
    {
        ...
    }
    __aicore__ inline void SplitBias()
    {
        LocalTensor<float> bias1Local = inQueueC1.DeQue<float>();
        LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();
        // 将bias从L1搬运到BT
        DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(n * sizeof(float) / 64), 0, 0 });
        outQueueC2.EnQue<float>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false;
        // 矩阵乘
        Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
        outQueueC2.FreeTensor(bias2Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }
private:
    TPipe pipe;
    TQue<TPosition::A1, 1> inQueueA1;
    TQue<TPosition::A2, 1> inQueueA2;
    TQue<TPosition::B1, 1> inQueueB1;
    TQue<TPosition::B2, 1> inQueueB2;
    TQue<TPosition::CO1, 1> outQueueCO1;
    TQue<TPosition::C1, 1> inQueueC1;
    TQue<TPosition::C2, 1> outQueueC2;

    GlobalTensor<half> aGM;
    GlobalTensor<half> bGM;
    GlobalTensor<float> cGM;
    GlobalTensor<float> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
```
