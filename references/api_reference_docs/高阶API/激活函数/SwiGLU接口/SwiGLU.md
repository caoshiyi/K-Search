# SwiGLU

**页面ID:** atlasascendc_api_07_0776  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0776.html

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

SwiGLU是采用Swish作为激活函数的GLU变体。具体计算公式如下：

![](images/atlasascendc_api_07_0776_img_001.png)

其中Swish激活函数的计算公式如下（β为常量）：

![](images/atlasascendc_api_07_0776_img_002.png)

#### 函数原型

- 通过sharedTmpBuffer入参传入临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer)
```

- 接口框架申请临时空间

  - 源操作数Tensor全部/部分参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const uint32_t calCount)
```

  - 源操作数Tensor全部参与计算

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor0, LocalTensor<T>& srcTensor1, const float& scalarValue)
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

- 通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
- 接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过GetSwiGLUMaxMinTmpSize中提供的接口获取需要预留空间范围的大小。

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
| srcTensor0/srcTensor1 | 输入 | 源操作数。 源操作数的数据类型需要与目的操作数保持一致。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| scalarValue | 输入 | 激活函数中的β参数。 |
| sharedTmpBuffer | 输入 | 临时缓存。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 用于SwiGLU内部复杂计算时存储中间变量，由开发者提供。 临时空间大小BufferSize的获取方式请参考GetSwiGLUMaxMinTmpSize。 |
| calCount | 输入 | 实际计算数据元素个数。 |

#### 约束说明

- **不支持源操作数与目的操作数地址重叠。**
- 当前仅支持ND格式的输入，不支持其他格式。
- 不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

#### 调用示例

```
#include "kernel_operator.h"

template <typename srcType>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU(){}
    __aicore__ inline void Init(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize, float beta)
    {
        betaValue = beta;
        dataSize = srcSize;
        src0Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src0Gm), dataSize);
        src1Global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src1Gm), dataSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dstGm), dataSize);
        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
        if (sizeof(srcType) != sizeof(float)) {
            pipe.InitBuffer(calcBufs, dataSize * (sizeof(float) / sizeof(uint8_t)) * 3);
        }

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
        AscendC::LocalTensor<srcType> src0Local = inQueueX.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.AllocTensor<srcType>();
        AscendC::DataCopy(src0Local, src0Global, dataSize);
        AscendC::DataCopy(src1Local, src1Global, dataSize);
        inQueueX.EnQue(src0Local);
        inQueueY.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> src0Local = inQueueX.DeQue<srcType>();
        AscendC::LocalTensor<srcType> src1Local = inQueueY.DeQue<srcType>();
        AscendC::LocalTensor<uint8_t> tmpLocal;
        if (sizeof(srcType) != sizeof(float)) {
            tmpLocal = calcBufs.Get<uint8_t>();
            AscendC::SwiGLU<srcType, false>(dstLocal, src0Local, src1Local, betaValue, tmpLocal, dataSize);
        } else {
            AscendC::SwiGLU<srcType, false>(dstLocal, src0Local, src1Local, betaValue, dataSize);
        }

        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(src0Local);
        inQueueY.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dst_global, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> src0Global;
    AscendC::GlobalTensor<srcType> src1Global;
    AscendC::GlobalTensor<srcType> dst_global;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufs;
    uint32_t dataSize = 0;
    float betaValue = 0;
};

template <typename dataType>
__aicore__ void kernel_swiglu_operator(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, uint32_t srcSize)
{
    KernelSwiGLU<dataType> op;
    float scalarValue = 1;
    op.Init(src0Gm, src1Gm, dstGm, srcSize, scalarValue);
    op.Process();
}
```

结果示例如下：

```
输入数据(srcTensor0): 
 [ 0.4065 -0.2167 -0.963  -3.895  -0.7275  3.227  -0.522  -2.299  -1.813
 -1.569   3.764   1.407  -1.633   3.908  -0.9927 -2.234   1.545   2.
 -3.06    1.94    0.765  -1.313   3.27    2.055   2.842   2.979   2.732
  2.533   2.03    1.154  -2.363  -2.451 ]
输入数据(srcTensor1)
 [-2.285  -1.502   2.783  -3.72    0.352  -2.615   0.8604  0.612   3.582
 -3.102  -3.86    2.88   -0.2117 -0.592  -0.5586  1.315   0.4087  3.771
  2.69    0.755  -2.154  -1.03   -3.459  -3.125   3.531  -0.657   3.885
  2.807   0.469  -1.434  -3.455  -1.3   ]
输出数据(dstLocal): 
[-0.0858   0.05927 -2.523    0.3425  -0.1504  -0.575   -0.3157  -0.912
 -6.32     0.2095  -0.2998   3.838    0.1545  -0.8237   0.2018  -2.316
  0.3794   7.375   -7.707    0.9966  -0.1713   0.356   -0.345   -0.2703
  9.75    -0.6685  10.4      6.703    0.5854  -0.3186   0.25     0.6826 ]
```
