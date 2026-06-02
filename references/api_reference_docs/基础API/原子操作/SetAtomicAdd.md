# SetAtomicAdd

**页面ID:** atlasascendc_api_07_0210  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0210.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

调用该接口后，可对后续的从VECOUT/L0C/L1到GM的数据传输开启原子累加，通过模板参数设定不同的累加数据类型。

#### 函数原型

```
template <typename T>
__aicore__ inline void SetAtomicAdd()
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 设定不同的累加数据类型。 Atlas 训练系列产品，支持的数据类型为：float；支持的数据通路为VECOUT->GM。 Atlas 推理系列产品AI Core，支持的数据类型为int16_t/half/float；支持的数据通路为VECOUT->GM。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为int8_t/int16_t/half/bfloat16_t/int32_t/float；支持的数据通路为VECOUT/L0C/L1->GM。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为int8_t/int16_t/half/bfloat16_t/int32_t/float；支持的数据通路为VECOUT/L0C/L1->GM。 Atlas 200I/500 A2 推理产品，支持的数据类型为int16_t/half/int32_t/float；支持的数据通路为VECOUT/L0C/L1->GM |

#### 约束说明

- 累加操作完成后，建议通过SetAtomicNone关闭原子累加，以免影响后续相关指令功能。
- 该指令执行前不会对GM的数据做清零操作，开发者根据实际的算子逻辑判断是否需要清零，如果需要自行进行清零操作。

#### 调用示例

本示例中，使用DataCopy从VECOUT搬出数据到外部dstGlobal时进行原子累加。为保证原子累加的正确性，在核函数调用前，需要对dstGm清零。

调用核函数时，blockDim设置为3，核函数调用示例如下：

```
...
// x为输入，z为输出
set_atomic_add_ops_kernel<<<3, nullptr, stream>>>(x, z);
...
```

核函数示例如下：

```
#include "kernel_operator.h"
class KernelSetAtomicAdd {
public:
    __aicore__ inline KernelSetAtomicAdd() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
        dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
        pipe.InitBuffer(inQueueSrc0, 1, 256 * sizeof(float));
        pipe.InitBuffer(outQueueDst, 1, 256 * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {}
    __aicore__ inline void Compute()
    {}
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> src0Local = inQueueSrc0.AllocTensor<float>();
        AscendC::SetAtomicNone();
        AscendC::DataCopy(src0Local, src0Global, 256);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(dstGlobal, src0Local, 256);
        AscendC::SetAtomicNone();
        inQueueSrc0.FreeTensor(src0Local);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<float> src0Global, dstGlobal;
};
extern "C" __global__ __aicore__ void set_atomic_add_ops_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* dstGm)
{
    KernelSetAtomicAdd op;
    op.Init(src0Gm, dstGm);
    op.Process();
}
```

结果示例如下：

```
每个核的输入数据Src0: [1,1,1,1,1,...,1] // 256个1
最终输出数据dstGm: [3,3,3,3,3,...,3] // 256个3
```
