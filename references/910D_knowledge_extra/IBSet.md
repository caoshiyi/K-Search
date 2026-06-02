# IBSet<a name="ZH-CN_TOPIC_0000002523344854"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用IBSet设置某一个核的标志位，与IBWait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。

## 函数原型<a name="section620mcpsimp"></a>

```
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx, int32_t eventID)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1596943217411"></a>
<table><thead align="left"><tr id="row1596916326417"><th class="cellrowborder" valign="top" width="18.62%" id="mcps1.2.3.1.1"><p id="p17970143211417"><a name="p17970143211417"></a><a name="p17970143211417"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.38%" id="mcps1.2.3.1.2"><p id="p8970163210416"><a name="p8970163210416"></a><a name="p8970163210416"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12970113214410"><td class="cellrowborder" valign="top" width="18.62%" headers="mcps1.2.3.1.1 "><p id="p9970193217417"><a name="p9970193217417"></a><a name="p9970193217417"></a>isAIVOnly</p>
</td>
<td class="cellrowborder" valign="top" width="81.38%" headers="mcps1.2.3.1.2 "><p id="p29702032748"><a name="p29702032748"></a><a name="p29702032748"></a>控制是否为AIVOnly模式，默认为true。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>gmWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p3844958114318"><a name="p3844958114318"></a><a name="p3844958114318"></a>外部存储核状态的公共缓存，类型为GlobalTensor。GlobalTensor数据结构的定义请参考<a href="GlobalTensor.md">GlobalTensor</a>。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>ubWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p89581532195312"><a name="p89581532195312"></a><a name="p89581532195312"></a>存储当前核状态的公共缓存。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>blockIdx</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p514534616180"><a name="p514534616180"></a><a name="p514534616180"></a>表示等待核的idx号，取值范围：[0, 核数-1]。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p98451586430"><a name="p98451586430"></a><a name="p98451586430"></a>eventID</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p20845205894317"><a name="p20845205894317"></a><a name="p20845205894317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>用来控制当前核的set、wait事件。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   gmWorkspace申请的空间最少要求为：核数 \* 32Bytes \* eventID\_max + blockIdx\_max \* 32Bytes + 32Bytes。（eventID\_max和blockIdx\_max分别指eventID、blockIdx的最大值 ）；
-   注意：如果是AIVOnly模式，核数 = GetBlockNum\(\)；如果是MIX模式，核数 = GetBlockNum\(\) \* 2；
-   ubWorkspace申请的空间最少要求为：32Bytes；
-   gmWorkspace缓存的值需要初始化为0。
-   使用该接口进行多核控制时，算子调用时指定的逻辑numBlocks必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

## 调用示例<a name="section177231425115410"></a>

本示例实现功能为使用2个核进行数据处理，每个核均是处理256个half类型数据。核0实现x+y的操作，并将结果放入z的前半部分，核1将核0的计算结果放入x，之后与y相加，结果存入z的后半部分，因此多个核之间需要进行数据同步。如需运行，请参考[ib\_set](https://gitcode.com/cann/asc-devkit/tree/master/examples/02_features/11_synchronous_control/ib_set)实现完整的代码。

```
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 2 * 256;
constexpr int32_t USE_CORE_NUM = 2;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* sync, __gm__ uint8_t* z)
    {
        blockIdx = AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ half*)x);
        yGm.SetGlobalBuffer((__gm__ half*)y);
        sync_gm.SetGlobalBuffer((__gm__ int32_t *)(sync),256);
        zGm.SetGlobalBuffer((__gm__ half*)z);

        pipe.InitBuffer(inQueueX, 1, BLOCK_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, BLOCK_LENGTH * sizeof(half));
        pipe.InitBuffer(vecIn, 1, 8 * sizeof(int32_t));
        pipe.InitBuffer(outQueueZ, 1, BLOCK_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        if (blockIdx == 1) { // 在核1设置IBWait，将阻塞指令执行直到核0操作完成
            auto sync_buf = vecIn.AllocTensor<int32_t>();
            AscendC::IBWait(sync_gm, sync_buf, 0, 0);
            vecIn.FreeTensor(sync_buf);
        }
        CopyIn();
        Compute();
        CopyOut();
        if (blockIdx == 0) { // 在核0设置IBSet，当核0的操作完成后再执行核1的指令
            auto sync_buf = vecIn.AllocTensor<int32_t>();
            AscendC::IBSet(sync_gm, sync_buf, 0, 0);
            vecIn.FreeTensor(sync_buf);
        }
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        if (blockIdx == 1) {
            AscendC::DataCopy(xLocal, zGm[0 * BLOCK_LENGTH], BLOCK_LENGTH);
            AscendC::DataCopy(yLocal, yGm[1 * BLOCK_LENGTH], BLOCK_LENGTH);
        } else {
            AscendC::DataCopy(xLocal, xGm[0], BLOCK_LENGTH);
            AscendC::DataCopy(yLocal, yGm[0], BLOCK_LENGTH);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, BLOCK_LENGTH);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[blockIdx * BLOCK_LENGTH], zLocal, BLOCK_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueY, vecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    AscendC::GlobalTensor<int32_t> sync_gm;
    int32_t blockIdx = 0;
};

extern "C" __global__ __aicore__ void add_simple_kernel(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* sync,
    __gm__ uint8_t* z)
{
    KernelAdd op;
    op.Init(x, y, sync, z);
    op.Process();
}
```

```
输入数据:
x: [1,1,1,1,1,...,1] // 512个1
y: [1,1,1,1,1,...,1] // 512个1
输出数据(dstGm):
[2,2,2,2,2,...,2,3,3,3,3,3,...,3] //前256个数是2，后256个数是3
```

