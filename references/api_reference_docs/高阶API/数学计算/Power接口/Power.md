# Power

**页面ID:** atlasascendc_api_07_0520  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0520.html

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

实现按元素做幂运算功能，提供3类接口，处理逻辑如下：

![](images/atlasascendc_api_07_0520_img_001.png)

<!-- img2text -->
```
                           ┌───────────────────────────┐
                           │  Power(dst, src0, src1)   │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                               src0^src1 按元素计算


      ┌───────────────────────────┐
      │ Power(dst, src0, src1)    │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌──────────────────────────────────────────────┐
      │ dst[i] = pow(src0[i], src1[i])              │
      └──────────────────────────────────────────────┘


      ┌───────────────────────────┐
      │ Power(dst, src0, scalar)  │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌──────────────────────────────────────────────┐
      │ dst[i] = pow(src0[i], scalar)               │
      └──────────────────────────────────────────────┘


      ┌───────────────────────────┐
      │ Power(dst, scalar, src1)  │
      └─────────────┬─────────────┘
                    │
                    ▼
      ┌──────────────────────────────────────────────┐
      │ dst[i] = pow(scalar, src1[i])               │
      └──────────────────────────────────────────────┘
```

#### 函数原型

- Power(dstTensor, src0Tensor, src1Tensor)

  - 通过sharedTmpBuffer入参传入临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

  - 接口框架申请临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor)
```

- Power(dstTensor, src0Tensor, src1Scalar)

  - 通过sharedTmpBuffer入参传入临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

  - 接口框架申请临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar)
```

- Power(dstTensor, src0Scalar, src1Tensor)

  - 通过sharedTmpBuffer入参传入临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

  - 接口框架申请临时空间

    - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, uint32_t calCount)
```

    - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过GetPowerMaxMinTmpSize中提供的GetPowerMaxMinTmpSize接口获取需要预留空间的范围大小。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float、int32_t。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float、int32_t。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float、int32_t。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstTensor | 输出 | 目的操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| src0Tensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| src1Tensor | 输入 | 源操作数。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 源操作数的数据类型需要与目的操作数保持一致。 |
| src0Scalar/src1Scalar | 输入 | 源操作数，类型为Scalar。源操作数的数据类型需要与目的操作数保持一致。 |
| sharedTmpBuffer | 输入 | 临时内存空间。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 针对3个power接口，不同输入数据类型情况下，临时空间大小BufferSize的获取方式请参考GetPowerMaxMinTmpSize。 |
| calCount | 输入 | 参与计算的元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 对于Atlas 推理系列产品AI Core，幂运算的指数必须小于231-1。

#### 调用示例

本样例中只展示Compute流程中的部分代码。如果您需要运行样例代码，请将该代码段拷贝并替换样例模板中Compute函数的部分代码即可。

- Power(dstTensor, srcTensor1, srcTensor2)

```
Power(dstLocal, srcLocal1, srcLocal2)
```

结果示例如下：

```
输入数据(srcLocal1): [1.4608411 4.344736 ... 0.46437776]
输入数据(srcLocal2): [-5.4534287 4.5122147 ... -0.9344089]
输出数据(dstLocal): [0.12657544 756.1846 ... 2.0477564]
```

- Power(dstTensor, srcTensor1, scalarValue)

```
Power(dstLocal, srcLocal1, scalarValue)
```

结果示例如下：

```
输入数据(srcLocal1): [2.263972 2.902264 ... 0.40299487]
输入数据(scalarValue): 1.2260373
输出数据(dstLocal): [2.7232351 3.6926038 ... 0.32815763]
```

- Power(dstTensor, scalarValue, srcTensor2)

```
Power(dstLocal, scalarValue, srcLocal2)
```

结果示例如下：

```
输入数据(scalarValue): 4.382112
输入数据(srcLocal2): [5.504859 2.0677629 ... 1.053188]
输出数据(dstLocal): [3407.0386 21.225077 ... 4.7403817]
```

#### 样例模板

```
#include "kernel_operator.h"
template <typename srcType>
class KernelPower
{
public:
    __aicore__ inline KernelPower() {}
    __aicore__ inline void Init(GM_ADDR src1Gm, GM_ADDR src2Gm, GM_ADDR dstGm, uint32_t srcSize)
    {
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), srcSize);
        src2Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src2Gm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), srcSize);
        pipe.InitBuffer(inQueueX1, 1, srcSize * sizeof(srcType));
        pipe.InitBuffer(inQueueX2, 1, srcSize * sizeof(srcType));
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
        AscendC::LocalTensor<srcType> srcLocal1 = inQueueX1.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal1, src1Global, bufferSize);
        inQueueX1.EnQue(srcLocal1);
        AscendC::LocalTensor<srcType> srcLocal2 = inQueueX2.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal2, src2Global, bufferSize);
        inQueueX2.EnQue(srcLocal2);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal1 = inQueueX1.DeQue<srcType>();
        AscendC::LocalTensor<srcType> srcLocal2 = inQueueX2.DeQue<srcType>();
        AscendC::LocalTensor<srcType> tmpLocal;
        srcType scalarValue1 = srcLocal1.GetValue(0);
        srcType scalarValue2 = srcLocal2.GetValue(0);
        AscendC::Power<srcType, false>(dstLocal, scalarValue1, srcLocal2);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX1.FreeTensor(srcLocal1);
        inQueueX2.FreeTensor(srcLocal2);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dstGlobal, dstLocal, bufferSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src1Global;
    AscendC::GlobalTensor<srcType> src2Global;
    AscendC::GlobalTensor<srcType> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};
template <typename dataType>
__aicore__ void kernel_power_operator(GM_ADDR src1Gm, GM_ADDR src2Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelPower<dataType> op;
    op.Init(src1Gm, src2Gm, dstGm, srcSize);
    op.Process();
}

extern "C" __global__ __aicore__ void power_operator_custom(GM_ADDR src1Gm, GM_ADDR src2Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    kernel_power_operator<half>(src1Gm, src2Gm, dstGm, srcSize);
}
```
