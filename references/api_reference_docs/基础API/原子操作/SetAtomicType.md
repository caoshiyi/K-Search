# SetAtomicType

**页面ID:** atlasascendc_api_07_0211  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0211.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

通过设置模板参数来设定原子操作不同的数据类型。

#### 函数原型

```
template <typename T>
__aicore__ inline void SetAtomicType()
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 设定不同的数据类型。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持int8_t/int16_t/half/bfloat16_t/int32_t/float Atlas 推理系列产品AI Core，支持int16_t/half/float Atlas 200I/500 A2 推理产品，支持int16_t/half/int32_t/float |

#### 约束说明

需要和SetAtomicAdd、SetAtomicMax、SetAtomicMin配合使用。

使用完成后，建议清空原子操作的状态（详见SetAtomicNone），以免影响后续相关指令功能。

#### 调用示例

```
// 本演示示例使用DataCopy从VECOUT搬出到外部dstGlobal时进行原子最小，并调用SetAtomicType修改原子最小的数据类型。
#include "kernel_operator.h"

static const int data_size = 256;
template <typename T>
class KernelDataCopyAtomicMin {
public:
    __aicore__ inline KernelDataCopyAtomicMin() {}
    __aicore__ inline void Init(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR dst_gm, uint32_t size)
    {
        this->size = size;
        src0Global.SetGlobalBuffer((__gm__ T *)src0_gm);
        src1Global.SetGlobalBuffer((__gm__ T *)src1_gm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dst_gm);
        pipe.InitBuffer(queueSrc0, 1, size * sizeof(T));
        pipe.InitBuffer(queueSrc1, 1, size * sizeof(T));
        pipe.InitBuffer(queueDst0, 1, size * sizeof(T));
        pipe.InitBuffer(queueDst1, 1, size * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> src0local = queueSrc0.AllocTensor<T>();
        AscendC::LocalTensor<T> src1local = queueSrc1.AllocTensor<T>();
        AscendC::DataCopy(src0local, src0Global, size);
        AscendC::DataCopy(src1local, src1Global, size);
        queueSrc0.EnQue(src0local);
        queueSrc1.EnQue(src1local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> src0local = queueSrc0.DeQue<T>();
        AscendC::LocalTensor<T> src1local = queueSrc1.DeQue<T>();
        AscendC::LocalTensor<T> dst0Local = queueDst0.AllocTensor<T>();
        AscendC::LocalTensor<T> dst1Local = queueDst1.AllocTensor<T>();
        AscendC::Abs(dst0Local, src0local, size);
        AscendC::Abs(dst1Local, src1local, size);
        queueDst0.EnQue(dst0Local);
        queueDst1.EnQue(dst1Local);
        queueSrc0.FreeTensor(src0local);
        queueSrc1.FreeTensor(src1local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dst0Local = queueDst0.DeQue<T>();
        AscendC::LocalTensor<T> dst1Local = queueDst1.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dst1Local, size);
        AscendC::PipeBarrier<PIPE_MTE3>();
        AscendC::SetAtomicMin<int8_t>();  // 此处设置的类型可随意，此例中以int8_t为例
        AscendC::SetAtomicType<T>();  // 此处设置真实的数据类型
        AscendC::DataCopy(dstGlobal, dst0Local, size);
        queueDst0.FreeTensor(dst0Local);
        queueDst1.FreeTensor(dst1Local);
        AscendC::SetAtomicNone();
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queueSrc0;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queueSrc1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queueDst0;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queueDst1;
    AscendC::GlobalTensor<T> src0Global, src1Global, dstGlobal;
    uint32_t size;
};
extern "C" __global__ __aicore__ void data_copy_atomic_min_kernel(GM_ADDR src0_gm, GM_ADDR src1_gm, GM_ADDR dst_gm)
{
    KernelDataCopyAtomicMin<half> op;
    op.Init(src0_gm, src1_gm, dst_gm, data_size);
    op.Process();
}
```

```
每个核的输入数据为: 
Src0: [1,1,1,1,1,...,1] // 256个1
Src1: [2,2,2,2,2,...,2] // 256个2
最终输出数据: [1,1,1,1,1,...,1] // 256个1
```
