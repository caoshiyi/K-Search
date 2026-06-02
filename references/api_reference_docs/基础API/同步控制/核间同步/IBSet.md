# IBSet

**页面ID:** atlasascendc_api_07_0202  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0202.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。调用IBSet设置某一个核的标志位，与IBWait成对出现配合使用，表示核之间的同步等待指令，等待某一个核操作完成。

#### 函数原型

```
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, int32_t blockIdx, int32_t eventID)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| isAIVOnly | 控制是否为AIVOnly模式，默认为true。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gmWorkspace | 输出 | 外部存储核状态的公共缓存，类型为GlobalTensor。GlobalTensor数据结构的定义请参考GlobalTensor。 |
| ubWorkspace | 输入 | 存储当前核状态的公共缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| blockIdx | 输入 | 表示等待核的idx号，取值范围：[0, 核数-1]。 |
| eventID | 输入 | 用来控制当前核的set、wait事件。 |

#### 约束说明

- gmWorkspace申请的空间最少要求为：核数 * 32Bytes * eventID_max + blockIdx_max * 32Bytes + 32Bytes。（eventID_max和blockIdx_max分别指eventID、blockIdx的最大值 ）；
- 注意：如果是AIVOnly模式，核数 = GetBlockNum()；如果是MIX模式，核数 = GetBlockNum() * 2；
- ubWorkspace申请的空间最少要求为：32Bytes；
- gmWorkspace缓存的值需要初始化为0。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockDim必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。

#### 调用示例

本示例实现功能为使用2个核进行数据处理，每个核均是处理256个half类型数据。核0实现x+y的操作，并将结果放入z的前半部分，核1将核0的计算结果放入x，之后与y相加，结果存入z的后半部分，因此多个核之间需要进行数据同步。

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
        if (blockIdx == 1) {
            auto sync_buf = vecIn.AllocTensor<int32_t>();
            AscendC::IBWait(sync_gm, sync_buf, 0, 0);
            vecIn.FreeTensor(sync_buf);
        }
        CopyIn();
        Compute();
        CopyOut();
        if (blockIdx == 0) {
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
