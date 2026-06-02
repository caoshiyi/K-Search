# Trunc

**页面ID:** atlasascendc_api_07_0536  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0536.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

按元素做浮点数截断，即向零取整操作。计算公式如下：

<!-- img2text -->
```
                 x > 0
      x ─────────────────────→ floor(x)
      │
      │ x = 0
      └──────────────────────→ 0
      │
      │ x < 0
      └──────────────────────→ ceil(x)
```

举例如下：

Trunc(3.9) = 3

Trunc(-3.9) = -3

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Trunc(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Trunc(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Trunc(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Trunc(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
```

由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过GetTruncMaxMinTmpSize中提供的GetTruncMaxMinTmpSize接口获取需要预留空间的大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcTensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于Trunc内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetTruncMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- 针对Atlas 推理系列产品AI Core，输入数据限制在[-2147483647.0, 2147483647.0]范围内。
- 支持源操作数与目的操作数地址重叠。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

```
#include "kernel_operator.h"

template <typename srcType>
class KernelTrunc
{
public:
    __aicore__ inline KernelTrunc(){}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(srcGm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), srcSize);
        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, srcSize * sizeof(srcType));
        bufferSize = srcSize;
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
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal, src_global, bufferSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        AscendC::Trunc<srcType, false>(dstLocal, srcLocal);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dst_global, dstLocal, bufferSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src_global;
    AscendC::GlobalTensor<srcType> dst_global;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};
template <typename dataType>
__aicore__ void kernel_trunc_operator(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelTrunc<dataType> op;
    op.Init(srcGm, dstGm, srcSize);
    op.Process();
}

extern "C" __global__ __aicore__ void trunc_operator(GM_ADDR srcGm, GM_ADDR dstGm, uint32_t srcSize)
{
    kernel_trunc_operator<half>(srcGm, dstGm, srcSize);
}
```

结果示例如下：

```
输入数据(srcLocal): [ 0.5317103  -6.37912032  5.53408647 ...  11.11059642 -11.67860335]
输出数据(dstLocal): [ 0.0        -6.0         5.0        ...  11.0        -11.0       ]
```
