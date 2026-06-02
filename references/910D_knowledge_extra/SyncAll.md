# SyncAll<a name="ZH-CN_TOPIC_0000002523303976"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="23.43%" id="mcps1.1.4.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持（软同步原型）</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="22.93%" id="mcps1.1.4.1.3"><p id="p16825819173218"><a name="p16825819173218"></a><a name="p16825819173218"></a>是否支持（硬同步原型）</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="23.43%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="22.93%" headers="mcps1.1.4.1.3 "><p id="p1441038173417"><a name="p1441038173417"></a><a name="p1441038173417"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。

## 函数原型<a name="section620mcpsimp"></a>

-   软同步

    ```
    template <bool isAIVOnly = true>
    __aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)
    ```

-   硬同步

    ```
    template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
    __aicore__ inline void SyncAll()
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1596943217411"></a>
<table><thead align="left"><tr id="row1596916326417"><th class="cellrowborder" valign="top" width="13.43%" id="mcps1.2.3.1.1"><p id="p17970143211417"><a name="p17970143211417"></a><a name="p17970143211417"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.57000000000001%" id="mcps1.2.3.1.2"><p id="p8970163210416"><a name="p8970163210416"></a><a name="p8970163210416"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12970113214410"><td class="cellrowborder" valign="top" width="13.43%" headers="mcps1.2.3.1.1 "><p id="p9970193217417"><a name="p9970193217417"></a><a name="p9970193217417"></a>isAIVOnly</p>
</td>
<td class="cellrowborder" valign="top" width="86.57000000000001%" headers="mcps1.2.3.1.2 "><p id="p6912194943411"><a name="p6912194943411"></a><a name="p6912194943411"></a>控制SyncAll作用于纯Vector算子或融合（Cube和Vector融合）算子。可选值：</p>
<a name="ul1034511561784"></a><a name="ul1034511561784"></a><ul id="ul1034511561784"><li><strong id="b891244983410"><a name="b891244983410"></a><a name="b891244983410"></a>true</strong>（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。</li><li><strong id="b79121249133420"><a name="b79121249133420"></a><a name="b79121249133420"></a>false</strong>：融合算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。</li></ul>
</td>
</tr>
<tr id="row47986408408"><td class="cellrowborder" valign="top" width="13.43%" headers="mcps1.2.3.1.1 "><p id="p18798194094017"><a name="p18798194094017"></a><a name="p18798194094017"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="86.57000000000001%" headers="mcps1.2.3.1.2 "><p id="p479815400409"><a name="p479815400409"></a><a name="p479815400409"></a>控制SyncAll函数的行为 ，在多个AI Core之间进行流水线同步时，指定哪些管道（pipe）用于触发和等待。</p>
<a name="ul047327174317"></a><a name="ul047327174317"></a><ul id="ul047327174317"><li><strong id="b7762111519458"><a name="b7762111519458"></a><a name="b7762111519458"></a>triggerPipe</strong>：指定哪个管道用于“发送触发信号”。</li><li><strong id="b1849919164510"><a name="b1849919164510"></a><a name="b1849919164510"></a>waitPipe</strong>：指定哪个管道用于“接收等待信号”。</li></ul>
<p id="p16336151565418"><a name="p16336151565418"></a><a name="p16336151565418"></a>默认为SyncAllConfig  DEFAULT_SYNC_ALL_CONFIG= {PIPE_ALL, PIPE_ALL} ，使用全部管道来进行触发和等待行为。</p>
<p id="p141061754195214"><a name="p141061754195214"></a><a name="p141061754195214"></a>该参数仅支持如下型号：</p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p68529181952"><a name="p68529181952"></a><a name="p68529181952"></a>gmWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p208511818357"><a name="p208511818357"></a><a name="p208511818357"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p223862673213"><a name="p223862673213"></a><a name="p223862673213"></a>gmWorkspace为用户定义的全局Global空间，作为所有核共用的缓存，用于保存每个核的状态标记，类型为GlobalTensor，支持的数据类型为int32_t。GlobalTensor数据结构的定义请参考<a href="GlobalTensor.md">GlobalTensor</a>。</p>
<p id="p1585111181855"><a name="p1585111181855"></a><a name="p1585111181855"></a>所需空间大小和使用注意项参见<a href="#section633mcpsimp">约束说明</a>。</p>
<p id="p377231420357"><a name="p377231420357"></a><a name="p377231420357"></a>硬同步接口不支持该参数。</p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p17849918857"><a name="p17849918857"></a><a name="p17849918857"></a>ubWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p1484911181458"><a name="p1484911181458"></a><a name="p1484911181458"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p01242034135414"><a name="p01242034135414"></a><a name="p01242034135414"></a>ubWorkspace为用户定义的局部Local空间，每个核单独自用，用于标记当前核的状态。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT，支持的数据类型为int32_t。</p>
<p id="p684819181755"><a name="p684819181755"></a><a name="p684819181755"></a>所需空间大小参见<a href="#section633mcpsimp">约束说明</a>。</p>
<p id="p1139714285356"><a name="p1139714285356"></a><a name="p1139714285356"></a>硬同步接口不支持该参数。</p>
</td>
</tr>
<tr id="row0101323102711"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p2010172315276"><a name="p2010172315276"></a><a name="p2010172315276"></a>usedCores</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p131012372711"><a name="p131012372711"></a><a name="p131012372711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p15101023102712"><a name="p15101023102712"></a><a name="p15101023102712"></a>指定多少个核之间的同步，传入数值不能超过算子调用时指定的逻辑numBlocks。此参数为默认参数，不传此参数表示全核软同步。</p>
<p id="p13881143914351"><a name="p13881143914351"></a><a name="p13881143914351"></a>仅在软同步接口中支持，硬同步接口不支持该参数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   gmWorkspace缓存申请的空间大小要求大于等于核数\*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：
    -   通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
    -   在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。

-   ubWorkspace申请的空间大小要求大于等于核数\*32Bytes。
-   使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
-   在分离模式下，建议使用硬同步接口而非软同步接口。软同步接口仅适用于纯Vector场景，且性能较低。使用硬同步接口时，需根据场景设置Kernel类型：
    -   在纯Vector/Cube场景下，需设置Kernel类型为KERNEL\_TYPE\_MIX\_AIV\_1\_0或KERNEL\_TYPE\_MIX\_AIC\_1\_0。
    -   对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。

-   使用该接口时，建议开启batchmode模式，使算子独占全部所需核资源，否则可能因满足以下条件导致死锁：

    -   多流并发场景（≥2条执行流）。
    -   ≥2个算子并发执行。
    -   所有并发算子的核数总和超过物理核数。
    -   ≥2个并发算子使用了核间同步功能。

    具体而言，在多流场景下，某条流的核间同步算子虽分配到n个物理核，但可能仅有n-m个核先被调度执行，而其余m个核因被其他流的核间同步算子抢占而尚未启动。先启动的n-m个核执行到核间同步时等待剩余m核完成，而剩余m核因被其他流的核间同步算子占用而无法释放，形成死锁。

    Kernel直调场景下通过[\_\_schedmode\_\_\(mode\)](SIMD-BuiltIn关键字和API.md#li1365012910475)限定符来设置batchmode模式；工程化算子开发场景下，通过TilingContext的SetScheduleMode接口来设置batchmode模式，具体请参考《基础数据结构和接口》[《基础数据结构和接口》](zh-cn_topic_0000002554332575.md)。

## 调用示例<a name="section642mcpsimp"></a>

本示例实现功能为使用8个核进行数据处理，每个核均是处理32个float类型数据，对该数据乘2后再与其他核上进行同样乘2的数据进行相加，中间结果保存到workGm，因此多个核之间需要进行数据同步。此样例中，使用软同步，入口函数传入的syncGm里的值都已经在host侧初始化为0。若以下用例改成使用硬同步，则不需要传入syncGm，并且不需要使用workQueue。如需运行，请参考[sync\_all](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/11_synchronous_control/sync_all)实现完整的代码。

```
#include "kernel_operator.h"

const int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

class KernelSyncAll {
public:
    __aicore__ inline KernelSyncAll() {}
    __aicore__ inline void Init(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm, __gm__ uint8_t* workGm,
        __gm__ uint8_t* syncGm)
    {
        blockNum = AscendC::GetBlockNum(); // 获取核总数
        perBlockSize = srcDataSize / blockNum; // 每个核平分处理相同个数
        blockIdx = AscendC::GetBlockIdx(); // 获取当前工作的核ID
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(srcGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dstGm + blockIdx * perBlockSize * sizeof(float)),
            perBlockSize);
        workGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workGm), srcDataSize);
        syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(syncGm), blockNum * DEFAULT_SYNCALL_NEED_SIZE);
        pipe.InitBuffer(inQueueSrc1, 1, perBlockSize * sizeof(float));
        pipe.InitBuffer(inQueueSrc2, 1, perBlockSize * sizeof(float));
        pipe.InitBuffer(workQueue, 1, blockNum * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));
        pipe.InitBuffer(outQueueDst, 1, perBlockSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        FirstCompute();
        CopyToWorkGlobal(); // 当前工作核计算后的数据先保存到外部工作空间
        // 等待所有核都完成计算
        AscendC::LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();
        AscendC::SyncAll(syncGlobal, workLocal);
        workQueue.FreeTensor(workLocal);
        // 最终累加结果需要等所有核都计算完成
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        AscendC::DataCopy(dstLocal,srcLocal2,perBlockSize); // 当前核计算结果先保存到目的空间
        inQueueSrc2.FreeTensor(srcLocal2);
        for (int i = 0; i < blockNum; i++) {
            if (i != blockIdx) {
                CopyFromOtherCore(i); // 从外部工作空间读取数据
                Accumulate(dstLocal); // 所有数据都累加到目的空间
            }
        }
        outQueueDst.EnQue(dstLocal);
        CopyOut();
    }
private:
    __aicore__ inline void CopyToWorkGlobal()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(workGlobal[blockIdx * perBlockSize], dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyFromOtherCore(int index)
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, workGlobal[index * perBlockSize], perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }
    __aicore__ inline void Accumulate(const AscendC::LocalTensor<float> &dstLocal)
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::Add(dstLocal, dstLocal, srcLocal1, perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
    }
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> srcLocal = inQueueSrc1.AllocTensor<float>();
        AscendC::DataCopy(srcLocal, srcGlobal, perBlockSize);
        inQueueSrc1.EnQue(srcLocal);
    }
    __aicore__ inline void FirstCompute()
    {
        AscendC::LocalTensor<float> srcLocal1 = inQueueSrc1.DeQue<float>();
        AscendC::LocalTensor<float> srcLocal2 = inQueueSrc2.AllocTensor<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
        float scalarValue(2.0);
        AscendC::Muls(dstLocal, srcLocal1, scalarValue, perBlockSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(srcLocal2,dstLocal,perBlockSize);
        inQueueSrc1.FreeTensor(srcLocal1);
        inQueueSrc2.EnQue(srcLocal2);
        outQueueDst.EnQue(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> dstLocal = outQueueDst.DeQue<float>();
        AscendC::DataCopy(dstGlobal, dstLocal, perBlockSize);
        outQueueDst.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> srcGlobal;
    AscendC::GlobalTensor<float> dstGlobal;
    AscendC::GlobalTensor<float> workGlobal;
    AscendC::GlobalTensor<int32_t> syncGlobal;
    int srcDataSize = 256;
    int32_t blockNum = 0;
    int32_t blockIdx = 0;
    uint32_t perBlockSize = 0;
};

extern "C" __global__ __aicore__ void kernel_syncAll_float(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm,
    __gm__ uint8_t* workGm, __gm__ uint8_t* syncGm)
{
    KernelSyncAll op;
    op.Init(srcGm, dstGm, workGm, syncGm);
    op.Process();
}
```

```
输入数据(srcGm):
[1,1,1,1,1,...,1]
输出数据(dstGm):
[16,16,16,16,16,...,16]
```

