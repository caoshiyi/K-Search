# 静态Tensor编程<a name="ZH-CN_TOPIC_0000002554351521"></a>

在基于Pipe进行算子开发的方式中，由Pipe（TPipe类）统一管理Device端内存等资源，开发者无需感知内存管理、DoubleBuffer流水、同步等处理，只需要按照计算流编写算子即可，但由此也带来了一些运行时开销（如TPipe创建、InitBuffer等）。

基于以上原因，Ascend C提供了静态Tensor编程方式，相比基于Pipe的编程方式，这种方式避免了TPipe内存管理初始化过程（约数百纳秒），从而减少了运行时开销，更有助于开发者实现极致性能。通过直接构造指定地址和存储位置的LocalTensor，并将其传递给计算、搬运等API进行编程，提供了更高的灵活性。然而，这种编程方式也带来了更高的开发复杂性，需要开发者自行管理DoubleBuffer和同步流水，并且只能使用Ascend C的基础API，而非全部功能。

两种编程方式的对比如下：

<!-- img2text -->
```
┌──────────────────────────────────────────────────────┐   ┌──────────────────────────────────────────────────────┐
│           基于TPipe和TQue自动管理内存与同步          │   │          静态Tensor编程自主管理内存与同步           │
├──────────────────────────────────────────────────────┤   ├──────────────────────────────────────────────────────┤
│ AscendC::TPipe pipe;   // 创建全局的资源管理对象     │   │ // 直接定义LocalTensor                              │
│ AscendC::TQue<AscendC::TPosition::VecIn, 1> queIn;   │   │ AscendC::LocalTensor<half> xLocal(AscendC::TPosition::VECIN, xAddr,   │
│ AscendC::TQue<AscendC::TPosition::VecOut, 1> queOut; │   │ bufferSize);                                         │
│ // 初始化队列的内存                                   │   │ AscendC::LocalTensor<half> yLocal(AscendC::TPosition::VECOUT, yAddr,  │
│ pipe.InitBuffer(queIn, 2, bufferSize);               │   │ bufferSize);                                         │
│ pipe.InitBuffer(queOut, 2, bufferSize);              │   │ AscendC::DataCopy(xLocal, xGm[offset], bufferSize); │
│                                                      │   │ AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID); │
│                                                      │   │ AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);│
│ auto xLocal = queIn.AllocTensor<half>();             │   │ AscendC::Exp(yLocal, xLocal, bufferSize);            │
│ auto yLocal = queOut.AllocTensor<half>();            │   │ // ...                                               │
│ AscendC::DataCopy(xLocal, xGm[offset], bufferSize);  │   │                                                      │
│ queIn.EnQue(xLocal);                                 │   │                                                      │
│ xLocal = queIn.DeQue<half>();                        │   │                                                      │
│ AscendC::Exp(yLocal, xLocal, bufferSize);            │   │                                                      │
│ // ...                                               │   │                                                      │
│                                                      │   │                                                      │
│                                ──►  隐式地开buf       │   │                                                      │
└──────────────────────────────────────────────────────┘   └──────────────────────────────────────────────────────┘
```

> **说明：** 
>-   静态Tensor编程的使用约束和限制请参考[使用约束和限制](#section19853161834615)。
>-   本节涉及的完整样例请参考[静态Tensor编程样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/23_static_tensor_programming_kernellaunch)。

## 编程范式<a name="section1486516584319"></a>

-   AI Core包括多种内存单元，比如用于矢量计算的Unified Buffer和用于矩阵计算的L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer等内存资源。
-   AI Core包括多条流水，比如Vector/Cube/Scalar计算流水，MTE1、MTE2、MTE3搬运流水等，每条流水并行执行。

静态Tensor编程方式下，开发者完全自主的管理AI Core上的所有内存资源，调用Ascend C提供的搬运或者计算类API编写算子，并根据数据依赖关系插入对应的同步事件，以达成最优性能。

下图是一个典型矢量算子的示意图，开发者首先根据业务计算量进行数据分块处理，之后根据核内的数据依赖关系完成同步事件的插入：

<!-- img2text -->
```
根据输入和片上内存大小对数据进行切分

输入数据
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────┬──────┐
│ 分块0 │ 分块1 │ 分块2 │ 分块3 │ 分块4 │ 分块5 │ 分块6 │ 分块7 │ 分块8 │ 分块9 │ …  │ 分块N │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴────┴──────┘

┌──────────────────────────────────────────────┐          ┌──────────────────────────────────────────────┐
│                                              │          │                                              │
│         ┌──────────────────────────┐         │          │         ┌──────────────────────────┐         │
│         │        PIPE_MTE2         │         │          │         │        PIPE_MTE2         │         │
│         │            ⇅             │         │          │         │            ⇅             │         │
│         │ WaitFlag<V_MTE2>(0)      │         │          │         │ WaitFlag<V_MTE2>(0)      │         │
│         │ SetFlag<MTE2_V>(0)       │         │          │         │ SetFlag<MTE2_V>(0)       │         │
│         │ SetFlag<V_MTE2>(0)       │         │          │         │ SetFlag<V_MTE2>(0)       │         │
│         │ WaitFlag<MTE2_V>(0)      │         │          │         │ WaitFlag<MTE2_V>(0)      │         │
│         └──────────────┬───────────┘         │          │         └──────────────┬───────────┘         │
│                        │                     │          │                        │                     │
│                        ↓                     │          │                        ↓                     │
│         ┌──────────────────────────┐         │          │         ┌──────────────────────────┐         │
│         │          PIPE_V          │         │          │         │          PIPE_V          │         │
│         │            ⇅             │         │          │         │            ⇅             │         │
│         │ WaitFlag<MTE3_V>(0)      │         │          │         │ WaitFlag<MTE3_V>(0)      │         │
│         │ SetFlag<V_MTE3>(0)       │         │          │         │ SetFlag<V_MTE3>(0)       │         │
│         │ SetFlag<MTE3_V>(0)       │         │          │         │ SetFlag<MTE3_V>(0)       │         │
│         │ WaitFlag<V_MTE3>(0)      │         │          │         │ WaitFlag<V_MTE3>(0)      │         │
│         └──────────────┬───────────┘         │          │         └──────────────┬───────────┘         │
│                        │                     │          │                        │                     │
│                        ↓                     │          │                        ↓                     │
│         ┌──────────────────────────┐         │          │         ┌──────────────────────────┐         │
│         │        PIPE_MTE3         │         │          │         │        PIPE_MTE3         │         │
│         └──────────────────────────┘         │          │         └──────────────────────────┘         │
│                                              │          │                                              │
│   ┌────────┐                         ┌─────┐ │          │   ┌────────┐                         ┌─────┐ │
│   │ 输入   │ ←────── MTE2 / WRITE ───│     │ │          │   │ 输入   │ ←────── MTE2 / WRITE ───│     │ │
│   │ 数据   │                         │     │ │          │   │ 数据   │                         │     │ │
│   └────────┘ ─────── MTE2 / READ ─→ │     │ │          │   └────────┘ ─────── MTE2 / READ ─→ │     │ │
│                                      │     │ │   ...    │                                      │     │ │
│   ┌────────┐ ←────── V / WRITE ───── │     │ │          │   ┌────────┐ ←────── V / WRITE ───── │     │ │
│   │ 输出   │                         │ UB  │ │          │   │ 输出   │                         │ UB  │ │
│   │ 数据   │ ─────── V / READ ────→  │     │ │          │   │ 数据   │ ─────── V / READ ────→  │     │ │
│   └────────┘                         │     │ │          │   └────────┘                         │     │ │
│                                      └─────┘ │          │                                      └─────┘ │
│                                              │          │                                              │
│                 Vector Core 0                │          │                 Vector Core n                │
└──────────────────────────────────────────────┘          └──────────────────────────────────────────────┘

输出数据
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬────┬──────┐
│ 分块0 │ 分块1 │ 分块2 │ 分块3 │ 分块4 │ 分块5 │ 分块6 │ 分块7 │ 分块8 │ 分块9 │ …  │ 分块N │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴────┴──────┘
```

## 内存管理<a name="section626154143513"></a>

静态Tensor编程方式下，开发者可以使用两种方式创建Tensor：

-   通过[LocalMemAllocator](LocalMemAllocator.md)指定硬件位置进行Tensor分配。

    LocalMemAllocator是一种线性内存分配器，开发者可以调用Alloc方法进行内存分配，地址分配从0开始，根据调用次序依次向后进行线性分配，LocalMemAllocator只是一个简单的线性分配器，并不提供内存释放以及其它内存管理的能力。在不关注Bank冲突场景或者算子初始功能开发时，可以使用LocalMemAllocator简化算子编写，在后续性能优化时切换到使用LocalTensor进行地址分配的方式。

-   通过[LocalTensor构造函数](LocalTensor构造函数.md#li1192912551322)创建Tensor，极致性能场景推荐使用此方式。

    开发者可以使用LocalTensor构造函数直接指定内存地址，实现内存的完全自主管理（本质上无需申请和释放内存）。使用时，需根据需求合理指定地址（不超过物理存储上限），并在保证功能正确的前提下进行内存复用。如果需要通过规避Bank冲突或者复用内存来获得极致性能时，推荐使用该方式。

```
    // 方式1：使用LocalMemAllocator进行内存分配
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<float> xLocalPing = ubAllocator.Alloc<float, TILE_LENGTH>();
    AscendC::LocalTensor<float> yLocalPing = ubAllocator.Alloc<float, TILE_LENGTH>();
    AscendC::LocalTensor<float> zLocalPing = ubAllocator.Alloc<float, TILE_LENGTH>();

    // 方式2：直接使用LocalTensor构造函数构造Tensor
    AscendC::LocalTensor<float> xLocalPing(AscendC::TPosition::VECCALC, xAddrPing, TILE_LENGTH);
    AscendC::LocalTensor<float> yLocalPing(AscendC::TPosition::VECCALC, yAddrPing, TILE_LENGTH);
    AscendC::LocalTensor<float> zLocalPing(AscendC::TPosition::VECCALC, zAddrPing, TILE_LENGTH);
```

## 同步管理<a name="section4338114663610"></a>

根据前文介绍的硬件架构，AI Core内部异步并行计算存在多条流水（包括矢量计算、矩阵计算、数据搬入、数据搬出等），多条流水之间存在数据依赖时，需要插入对应的同步事件。静态Tensor编程方式下，开发者使用[SetFlag/WaitFlag\(ISASI\)](SetFlag-WaitFlag(ISASI).md)和[PipeBarrier\(ISASI\)](PipeBarrier(ISASI).md)手动插入同步，事件的类型和事件ID由开发者自行管理，但需要注意事件ID不能使用6和7（可能与内部使用的事件ID出现冲突，进而出现未定义行为）。另外由于需要使用SetFlag/WaitFlag/PipeBarrier底层同步接口（属于ISASI硬件体系结构相关的接口），无法保证跨硬件版本兼容。

在同步依赖中，根据数据依赖和指令执行关系，存在两种依赖关系，即正向同步（循环内依赖）与反向同步（循环间依赖）：

-   正向同步

    在本次数据搬入和计算之间，插入MTE2\_V（矢量计算流水等待MT2搬运流水）同步事件，确保数据搬入之后再进行计算；在本次数据计算和搬出之间，插入V\_MTE3（MTE3搬运流水等待矢量计算流水）同步事件，确保数据计算完成后再进行搬出。

-   反向同步

    在上一次的数据计算和本次数据搬入之间，插入V\_MTE2（MT2搬运流水等待矢量计算流水）同步事件，确保上一次的数据计算完成后，本次的数据再进行搬入。防止本次的数据会覆盖掉上一次未计算完成的数据；在上一次的数据搬出和本次数据计算之间，插入MTE3\_V（矢量计算流水等待MT3搬运流水）同步事件，确保上一次的数据搬出后，再进行本次数据的计算。防止本次的数据会覆盖掉上一次未搬出的数据。

上述的同步逻辑在使用Pipe编程框架时，框架会使用EnQue/DeQue/AllocTensor/FreeTensor进行封装。您可以通过[编程模型设计原理](编程模型设计原理.md)来了解应该如何在使用静态Tensor编程方式时手动进行同步控制。

```
    AscendC::LocalTensor<float> xLocal(AscendC::TPosition::VECCALC, xAddr, TILE_LENGTH);
    AscendC::LocalTensor<float> yLocal(AscendC::TPosition::VECCALC, yAddr, TILE_LENGTH);
    AscendC::LocalTensor<float> zLocal(AscendC::TPosition::VECCALC, zAddr, TILE_LENGTH);
    for (int i = 0; i < loopCount; i++) {
        // dependency of PIPE_V & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
        if (i != 0) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);
        // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (i != 0) {
            // dependency of PIPE_MTE3 & PIPE_V caused by zLocal between 2 sequential loops
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        if (i != (loopCount - 1)) {
            // dependency of PIPE_V & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        // dependency of PIPE_V & PIPE_MTE3 caused by zLocal in one single loop
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
        if (i != (loopCount - 1)) {
            // dependency of PIPE_MTE3 & PIPE_V caused by zLocal between 2 sequential loops
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
```

## 流水优化<a name="section121239188376"></a>

在基于TPipe的编程范式中，开发者只需要在InitBuffer时指定buffer数量为2，即可自动开启Double Buffer。但是静态Tensor编程方式下，开发者需要手动开启Double Buffer，具体示例如下，完整样例请参考[静态Tensor编程样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/23_static_tensor_programming_kernellaunch)中的Double Buffer示例。

```
    // ping
    AscendC::LocalTensor<float> xLocalPing(AscendC::TPosition::VECCALC, xAddrPing, TILE_LENGTH);
    AscendC::LocalTensor<float> yLocalPing(AscendC::TPosition::VECCALC, yAddrPing, TILE_LENGTH);
    AscendC::LocalTensor<float> zLocalPing(AscendC::TPosition::VECCALC, zAddrPing, TILE_LENGTH);
    // pong
    AscendC::LocalTensor<float> xLocalPong(AscendC::TPosition::VECCALC, xAddrPong, TILE_LENGTH);
    AscendC::LocalTensor<float> yLocalPong(AscendC::TPosition::VECCALC, yAddrPong, TILE_LENGTH);
    AscendC::LocalTensor<float> zLocalPong(AscendC::TPosition::VECCALC, zAddrPong, TILE_LENGTH);

    // double buffer
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int i = 0; i < loopCount; i++) {
        int32_t eventID = (i % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
        AscendC::LocalTensor<float> &xLocal = (i % 2 == 0 ? xLocalPing : xLocalPong);
        AscendC::LocalTensor<float> &yLocal = (i % 2 == 0 ? yLocalPing : yLocalPong);
        AscendC::LocalTensor<float> &zLocal = (i % 2 == 0 ? zLocalPing : zLocalPong);
        // dependency of PIPE_MTE3 & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);

        // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID);
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        // dependency of PIPE_V & PIPE_MTE3 caused by zLocal in one single loop
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
        AscendC::DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
        // dependency of PIPE_MTE3 & PIPE_MTE2 caused by zLocal between 2 sequential loops
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
```

以下为不使能DoubleBuffer和使能DoubleBuffer的流水示意图。多数情况下，采用DoubleBuffer能有效提升Vector的时间利用率，缩减算子执行时间，详细内容可参考[DoubleBuffer](DoubleBuffer.md)。

<!-- img2text -->
```
未开启DoubleBuffer流水图

┌───────┐
│ MTE2  │
└───────┘
    │
    │  CopyIn(loop 0)         CopyIn(loop 1)         CopyIn(loop 2)
    ├───────[──────────]────────────[──────────]────────────[──────────]────────────────────→

┌────────┐
│ Vector │
└────────┘
    │
    │            Compute(loop 0)        Compute(loop 1)        Compute(loop 2)
    ├───────────[──────────]────────────[──────────]───────────[──────────]─────────────────→

┌───────┐
│ MTE3  │
└───────┘
    │
    │                       CopyOut(loop 0)        CopyOut(loop 1)        CopyOut(loop 2)
    └──────────────────────[──────────]────────────[──────────]────────────[──────────]────→
```

<!-- img2text -->
```text
使能DoubleBuffer流水示意图

                CopyIn(loop 0)    CopyIn(loop 1)    CopyIn(loop 2)
MTE2     ─────┌───────────────┬───────────────┬───────────────┐────────────────────────────→
              │               │               │               │
              └───────────────┴───────────────┴───────────────┘

                           Compute(loop 0)   Compute(loop 1)   Compute(loop 2)
Vector   ─────────────────┌───────────────┬───────────────┬───────────────┐────────────────→
                          │               │               │               │
                          └───────────────┴───────────────┴───────────────┘

                                          CopyOut(loop 0)   CopyOut(loop 1)   CopyOut(loop 2)
MTE3     ────────────────────────────────┌───────────────┬───────────────┬───────────────┐──→
                                         │               │               │               │
                                         └───────────────┴───────────────┴───────────────┘
```

## 使用约束和限制<a name="section19853161834615"></a>

静态Tensor编程方式需要遵循以下约束和限制：

-   开发者不能使用TPipe/TQue/TQueBind/TBufPool等框架接口，和上述框架接口混用可能会出现未定义行为。
-   只能使用部分API。具体支持的API列表见[支持的API范围](#section2633193623711)。因为不在列表范围内的API内部依赖TPipe分配事件ID，可能会和开发者定义的事件ID产生冲突。
-   同步事件需要由开发者使用[SetFlag/WaitFlag\(ISASI\)](SetFlag-WaitFlag(ISASI).md)和[PipeBarrier\(ISASI\)](PipeBarrier(ISASI).md)手动插入，事件的类型和事件ID由开发者自行管理，但需要注意事件ID不能使用6和7（可能与内部使用的事件ID出现冲突，进而出现未定义行为）。
-   由于需要使用SetFlag/WaitFlag/PipeBarrier底层同步接口（属于ISASI硬件体系结构相关的接口），无法保证算子跨硬件版本兼容。
-   Kernel入口处需要开发者手动调用[InitSocState](InitSocState.md)接口用来初始化全局状态寄存器。因为全局状态寄存器处于不确定状态，如果不调用该接口，可能导致算子执行过程中出现未定义行为。在TPipe框架编程中，初始化过程由TPipe完成，无需开发者关注。

## 支持的API范围<a name="section2633193623711"></a>

**表 1**  针对Ascend 950PR/Ascend 950DT，支持的API范围

<a name="table1798673512413"></a>
<table><thead align="left"><tr id="row109864353413"><th class="cellrowborder" valign="top" width="29.630000000000003%" id="mcps1.2.4.1.1"><p id="p1098613524117"><a name="p1098613524117"></a><a name="p1098613524117"></a>接口分类</p>
</th>
<th class="cellrowborder" valign="top" width="44.519999999999996%" id="mcps1.2.4.1.2"><p id="p29871535174117"><a name="p29871535174117"></a><a name="p29871535174117"></a>接口名称</p>
</th>
<th class="cellrowborder" valign="top" width="25.85%" id="mcps1.2.4.1.3"><p id="p9987835154115"><a name="p9987835154115"></a><a name="p9987835154115"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row1298763513418"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p15987173524114"><a name="p15987173524114"></a><a name="p15987173524114"></a>基础API &gt; 标量计算</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p9987123512412"><a name="p9987123512412"></a><a name="p9987123512412"></a>ScalarGetCountOfValue、ScalarCountLeadingZero、ScalarCast、CountBitsCntSameAsSignBit、ScalarGetSFFValue</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p298711352411"><a name="p298711352411"></a><a name="p298711352411"></a>-</p>
</td>
</tr>
<tr id="row1098743584116"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p69871035114113"><a name="p69871035114113"></a><a name="p69871035114113"></a>基础API &gt; 矢量计算 &gt; 基础算术</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p1068419974813"><a name="p1068419974813"></a><a name="p1068419974813"></a>BilinearInterpolation、Prelu</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p798711356416"><a name="p798711356416"></a><a name="p798711356416"></a>-</p>
</td>
</tr>
<tr id="row1798773512412"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p189871935184120"><a name="p189871935184120"></a><a name="p189871935184120"></a>基础API &gt; 矢量计算 &gt; 复合计算</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p10987133511418"><a name="p10987133511418"></a><a name="p10987133511418"></a>Axpy、FusedAbsSub、FusedExpSub</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p179879356419"><a name="p179879356419"></a><a name="p179879356419"></a>-</p>
</td>
</tr>
<tr id="row998718356411"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p5987335194120"><a name="p5987335194120"></a><a name="p5987335194120"></a>基础API &gt; 矢量计算 &gt; 类型转换</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p198717354419"><a name="p198717354419"></a><a name="p198717354419"></a>Truncate</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p798719352419"><a name="p798719352419"></a><a name="p798719352419"></a>-</p>
</td>
</tr>
<tr id="row7987193534118"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p898712358414"><a name="p898712358414"></a><a name="p898712358414"></a>基础API &gt; 矢量计算 &gt; 归约计算</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p29871135184118"><a name="p29871135184118"></a><a name="p29871135184118"></a>WholeReduceSum、BlockReduceMax、BlockReduceMin、BlockReduceSum、PairReduceSum、RepeatReduceSum、ReduceMax、ReduceMin</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p1398713564110"><a name="p1398713564110"></a><a name="p1398713564110"></a>-</p>
</td>
</tr>
<tr id="row19871835174112"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p298716354416"><a name="p298716354416"></a><a name="p298716354416"></a>基础API &gt; 矢量计算 &gt; 数据转换</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p79872035154118"><a name="p79872035154118"></a><a name="p79872035154118"></a>Transpose</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p149871335154112"><a name="p149871335154112"></a><a name="p149871335154112"></a>-</p>
</td>
</tr>
<tr id="row7987113516412"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p139871235164110"><a name="p139871235164110"></a><a name="p139871235164110"></a>基础API &gt; 矢量计算 &gt; 排序组合</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p8325152773515"><a name="p8325152773515"></a><a name="p8325152773515"></a>Sort32、MrgSort、GetMrgSortResult</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p159871035134111"><a name="p159871035134111"></a><a name="p159871035134111"></a>-</p>
</td>
</tr>
<tr id="row9987535154118"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p14987163518416"><a name="p14987163518416"></a><a name="p14987163518416"></a>基础API &gt; 矢量计算 &gt; 离散与聚合</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p1987163510416"><a name="p1987163510416"></a><a name="p1987163510416"></a>Gather</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p169872035144115"><a name="p169872035144115"></a><a name="p169872035144115"></a>-</p>
</td>
</tr>
<tr id="row098783516417"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p139874354419"><a name="p139874354419"></a><a name="p139874354419"></a>基础API &gt; 矢量计算 &gt; 掩码操作</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p99871335184114"><a name="p99871335184114"></a><a name="p99871335184114"></a>SetMaskCount、SetMaskNorm、SetVectorMask、ResetMask</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p18988435204114"><a name="p18988435204114"></a><a name="p18988435204114"></a>-</p>
</td>
</tr>
<tr id="row1598863534117"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p1998811353419"><a name="p1998811353419"></a><a name="p1998811353419"></a>基础API &gt; 数据搬运 &gt; DataCopy</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p398811358414"><a name="p398811358414"></a><a name="p398811358414"></a>基础数据搬运</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p11988335184112"><a name="p11988335184112"></a><a name="p11988335184112"></a>不支持VECIN/VECCALC/VECOUT -&gt; TSCM通路的数据搬运。</p>
<p id="p123882056152818"><a name="p123882056152818"></a><a name="p123882056152818"></a>不支持GM -&gt; A1、B1、C1通路的数据搬运。</p>
</td>
</tr>
<tr id="row19881535134113"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p1198813574117"><a name="p1198813574117"></a><a name="p1198813574117"></a>基础API &gt; 同步控制 &gt; 核内同步</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p09886358410"><a name="p09886358410"></a><a name="p09886358410"></a>SetFlag/WaitFlag、PipeBarrier、DataSyncBarrier</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p5988435164113"><a name="p5988435164113"></a><a name="p5988435164113"></a>-</p>
</td>
</tr>
<tr id="row831243535112"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p95097467513"><a name="p95097467513"></a><a name="p95097467513"></a>基础API &gt; 同步控制 &gt; 核间同步</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p73131635175111"><a name="p73131635175111"></a><a name="p73131635175111"></a>CrossCoreSetFlag、CrossCoreWaitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 ">&nbsp;&nbsp;</td>
</tr>
<tr id="row49881535124115"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p79882353419"><a name="p79882353419"></a><a name="p79882353419"></a>基础API &gt; 缓存控制</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p2098815359419"><a name="p2098815359419"></a><a name="p2098815359419"></a>DataCachePreload、DataCacheCleanAndInvalid、ICachePreLoad、GetICachePreloadStatus</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p1988183584116"><a name="p1988183584116"></a><a name="p1988183584116"></a>-</p>
</td>
</tr>
<tr id="row16988143510417"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p7988123510411"><a name="p7988123510411"></a><a name="p7988123510411"></a>基础API &gt; 系统变量访问</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p8988183594114"><a name="p8988183594114"></a><a name="p8988183594114"></a>GetBlockNum、GetBlockIdx、InitSocState</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p14988035194113"><a name="p14988035194113"></a><a name="p14988035194113"></a>-</p>
</td>
</tr>
<tr id="row1298811359417"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p10988193511419"><a name="p10988193511419"></a><a name="p10988193511419"></a>基础API &gt; 原子操作</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p59881035164115"><a name="p59881035164115"></a><a name="p59881035164115"></a>SetAtomicAdd</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p89881358416"><a name="p89881358416"></a><a name="p89881358416"></a>-</p>
</td>
</tr>
<tr id="row11988153510410"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p4989143554117"><a name="p4989143554117"></a><a name="p4989143554117"></a>基础API &gt; 矩阵计算</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p398953516417"><a name="p398953516417"></a><a name="p398953516417"></a>LoadData</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p7989123518419"><a name="p7989123518419"></a><a name="p7989123518419"></a>-</p>
</td>
</tr>
<tr id="row2098911359417"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p6989173517418"><a name="p6989173517418"></a><a name="p6989173517418"></a>Utils API &gt; C++标准库 &gt; 算法</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p79891335164117"><a name="p79891335164117"></a><a name="p79891335164117"></a>max、min、index_sequence</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p5989193564117"><a name="p5989193564117"></a><a name="p5989193564117"></a>-</p>
</td>
</tr>
<tr id="row3989143504110"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p119891335204116"><a name="p119891335204116"></a><a name="p119891335204116"></a>Utils API &gt; C++标准库 &gt; 容器函数</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p29891935104116"><a name="p29891935104116"></a><a name="p29891935104116"></a>tuple、get、make_tuple</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p3989113510418"><a name="p3989113510418"></a><a name="p3989113510418"></a>-</p>
</td>
</tr>
<tr id="row59891735184118"><td class="cellrowborder" valign="top" width="29.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p1298918353418"><a name="p1298918353418"></a><a name="p1298918353418"></a>Utils API &gt; C++标准库 &gt; 类型特性</p>
</td>
<td class="cellrowborder" valign="top" width="44.519999999999996%" headers="mcps1.2.4.1.2 "><p id="p1698943519414"><a name="p1698943519414"></a><a name="p1698943519414"></a>is_convertible、is_same、enable_if、conditional</p>
</td>
<td class="cellrowborder" valign="top" width="25.85%" headers="mcps1.2.4.1.3 "><p id="p9989113544116"><a name="p9989113544116"></a><a name="p9989113544116"></a>-</p>
</td>
</tr>
</tbody>
</table>

