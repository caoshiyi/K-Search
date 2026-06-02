# SyncAll

**页面ID:** atlasascendc_api_07_0204  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0204.html

---

#### 产品支持情况

| 产品 | 是否支持（软同步原型） | 是否支持（硬同步原型） |
| --- | --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ | √ |
| Atlas 200I/500 A2 推理产品 | x | x |
| Atlas 推理系列产品AI Core | √ | x |
| Atlas 推理系列产品Vector Core | x | x |
| Atlas 训练系列产品 | √ | x |

#### 功能说明

当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用该函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬件同步是利用硬件自带的全核同步指令由硬件保证多核同步，软件同步是使用软件算法模拟实现。

#### 函数原型

- 软同步

```
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace, const int32_t usedCores = 0)
```

- 硬同步

```
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll()
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| isAIVOnly | 控制SyncAll作用于纯Vector算子或融合（Cube和Vector融合）算子。可选值： - **true**（默认值）：纯Vector算子的全核同步，仅执行Vector核的全核同步。- **false**：融合算子的全核同步，先分别完成Vector核和Cube核的全核同步，再执行两者之间的同步（软同步接口不支持此功能）。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| gmWorkspace | 输入 | gmWorkspace为用户定义的全局Global空间，作为所有核共用的缓存，用于保存每个核的状态标记，类型为GlobalTensor，支持的数据类型为int32_t。GlobalTensor数据结构的定义请参考GlobalTensor。 所需空间大小和使用注意项参见约束说明。 硬同步接口不支持该参数。 |
| ubWorkspace | 输入 | ubWorkspace为用户定义的局部Local空间，每个核单独自用，用于标记当前核的状态。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT，支持的数据类型为int32_t。 所需空间大小参见约束说明。 硬同步接口不支持该参数。 |
| usedCores | 输入 | 指定多少个核之间的同步，传入数值不能超过算子调用时指定的逻辑blockDim。此参数为默认参数，不传此参数表示全核软同步。 仅在软同步接口中支持，硬同步接口不支持该参数。 |

#### 约束说明

- gmWorkspace缓存申请的空间大小要求大于等于核数*32Bytes，并且缓存的值需要初始化为0。目前常见的有两种初始化方式：

  - 通过在host侧进行初始化操作，确保传入该接口时，gmWorkspace缓存已经初始化为0；
  - 在kernel侧初始化的时候对gmWorkspace缓存初始化，需要注意的是，每个核上都需要初始化全部的gmWorkspace缓存空间。

- ubWorkspace申请的空间大小要求大于等于核数*32Bytes。
- 使用该接口进行多核控制时，算子调用时指定的逻辑blockDim必须保证不大于实际运行该算子的AI处理器核数，否则框架进行多轮调度时会插入异常同步，导致Kernel“卡死”现象。
- 在分离模式下，建议使用硬同步接口而非软同步接口。软同步接口仅适用于纯Vector场景，且性能较低。使用硬同步接口时，需根据场景设置Kernel类型：

  - 在纯Vector/Cube场景下，需设置Kernel类型为KERNEL_TYPE_MIX_AIV_1_0或KERNEL_TYPE_MIX_AIC_1_0。
  - 对于Vector和Cube混合场景，需根据实际情况灵活配置Kernel类型。

#### 调用示例

本示例实现功能为使用8个核进行数据处理，每个核均是处理32个float类型数据，对该数据乘2后再与其他核上进行同样乘2的数据进行相加，中间结果保存到workGm，因此多个核之间需要进行数据同步。此样例中，使用软同步，入口函数传入的syncGm里的值都已经在host侧初始化为0。若以下用例改成使用硬同步，则不需要传入syncGm，并且不需要使用workQueue。

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
