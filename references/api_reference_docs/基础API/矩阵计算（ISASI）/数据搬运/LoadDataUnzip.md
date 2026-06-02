# LoadDataUnzip

**页面ID:** atlasascendc_api_07_0243  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0243.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

将GM上的数据解压并搬运到A1/B1/B2上。执行该API前需要执行LoadUnzipIndex加载压缩索引表。

#### 函数原型

```
template <typename T>
__aicore__ inline void LoadDataUnzip(const LocalTensor<T>& dst, const GlobalTensor<T>& src)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，类型为LocalTensor，支持的TPosition为A1/B1/B2。 LocalTensor的起始地址需要保证：TPosition为A1/B1时，32字节对齐；TPosition为B2时，512B对齐。 支持的数据类型为：int8_t。 |
| src | 输入 | 源操作数，类型为GlobalTensor。数据类型需要与dst保持一致。 |

#### 约束说明

#### 调用示例

该调用示例支持的运行平台为Atlas 推理系列产品AI Core。

```
#include "kernel_operator.h"

class KernelLoadUnzip {
public:
    __aicore__ inline KernelLoadUnzip() {}
    __aicore__ inline void Init(__gm__ int8_t *weGm, __gm__ int8_t *indexGm, __gm__ int8_t *dstGm)
    {
        weGlobal.SetGlobalBuffer((__gm__ int8_t *)weGm);
        indexGlobal.SetGlobalBuffer((__gm__ int8_t *)indexGm);
        dstGlobal.SetGlobalBuffer((__gm__ int8_t *)dstGm);
        pipe.InitBuffer(inQueueB1, 1, dstLen * sizeof(int8_t));
        pipe.InitBuffer(outQueueUB, 1, dstLen * sizeof(int8_t));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        CopyToUB();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int8_t> weightB1 = inQueueB1.AllocTensor<int8_t>();
        AscendC::LoadUnzipIndex(indexGlobal, numOfIndexTabEntry);
        AscendC::LoadDataUnzip(weightB1, weGlobal);
        inQueueB1.EnQue(weightB1);
    }
    __aicore__ inline void CopyToUB()
    {
        AscendC::LocalTensor<int8_t> weightB1 = inQueueB1.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.AllocTensor<int8_t>();
        AscendC::DataCopy(featureMapUB, weightB1, dstLen);
        outQueueUB.EnQue<int8_t>(featureMapUB);
        inQueueB1.FreeTensor(weightB1);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int8_t> featureMapUB = outQueueUB.DeQue<int8_t>();
        event_t eventIdMTE1ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE1_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE3>(eventIdMTE1ToMTE3);
        AscendC::DataCopy(dstGlobal, featureMapUB, dstLen);
        outQueueUB.FreeTensor(featureMapUB);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueUB;
    AscendC::GlobalTensor<int8_t> weGlobal;
    AscendC::GlobalTensor<int8_t> dstGlobal;
    AscendC::GlobalTensor<int8_t> indexGlobal;
    uint32_t srcLen = 896, dstLen = 1024, numOfIndexTabEntry = 1;
};
extern "C" __global__ __aicore__ void cube_load_unzip_simple_kernel(__gm__ int8_t *weightGm,
    __gm__ int8_t *indexGm, __gm__ int8_t *dstGm)
{
    KernelLoadUnzip op;
    op.Init(weightGm, indexGm, dstGm);
    op.Process();
}
```
