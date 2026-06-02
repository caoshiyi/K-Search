# BatchNorm

**页面ID:** atlasascendc_api_07_0806  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0806.html

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

BatchNorm是对于每一层的输入做规范化处理，使得每一层的分布尽可能的相同，从而加速训练过程和提高模型的泛化能力（有效减少梯度消失和梯度爆炸问题）。基本思想是对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。具体来说，对于输入特征x，BatchNorm的计算过程可以表示为：

1. 对输入特征x，在batch维度上计算均值μ和方差σ：

![](images/atlasascendc_api_07_0806_img_001.png)

2. 对于每个特征i，对输入特征x进行归一化：

![](images/atlasascendc_api_07_0806_img_002.png)

3. 对归一化后的特征进行缩放和平移：

![](images/atlasascendc_api_07_0806_img_003.png)

#### 函数原型

- 接口框架申请临时空间

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm, const LocalTensor<T>& beta, const T epsilon, BatchNormTiling& tiling)
```

- 通过sharedTmpBuffer入参传入临时空间

```
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm, const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：half、float。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：half、float。 Atlas 推理系列产品AI Core，支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |
| isBasicBlock | inputX、output的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求如下： - originB是8的倍数；- S*H是64的倍数，但小于2048。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output | 输出 | 目的操作数，shape为[B，S，H]。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputMean | 输出 | 均值，目的操作数，shape为[S，H]。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| outputVariance | 输出 | 方差，目的操作数，shape为[S，H]。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| inputX | 输入 | 源操作数，shape为[B，S，H]。inputX的数据类型需要与目的操作数保持一致，S*H需要32B对齐。支持inputX与output地址重叠。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| gamm | 输入 | 源操作数，shape为[B]。gamm的数据类型需要与目的操作数保持一致，长度需要32B对齐。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| beta | 输入 | 源操作数，shape为[B]。beta的数据类型需要与目的操作数保持一致，长度需要32B对齐。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| sharedTmpBuffer | 输入 | 接口内部复杂计算时用于存储中间变量，由开发者提供。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 临时空间大小BufferSize的获取方式请参考BatchNorm Tiling。 |
| epsilon | 输入 | 防除0的权重系数。数据类型需要与inputX/output保持一致。 |
| tiling | 输入 | 输入数据的切分信息，Tiling信息的获取请参考BatchNorm Tiling。 |

#### 约束说明

- 当前仅支持ND格式的输入，不支持其他格式。
- 输入数据的S*H必须满足32B对齐的要求。

#### 调用示例

```
#include "kernel_operator.h"

template <typename dataType, bool isReuseSource = false, bool isBasicBlock = false>
class KernelBatchnorm {
public:
    __aicore__ inline KernelBatchnorm()
    {}
    __aicore__ inline void Init(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
        GM_ADDR outputMeanGm, GM_ADDR outputVariance_gm, const TilingData &tiling)
    {
        bLength = tiling.bLength;
        sLength = tiling.sLength;
        hLength = tiling.hLength;
        batchNormTilling = tiling.batchNormTilingData;
        originalBLength = tiling.originalBLength;
        bshLength = originalBLength * sLength * hLength;
        shLength = sLength * hLength;
        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(inputXGm), bshLength);
        gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gammGm), bLength);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(betaGm), bLength);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputGm), bshLength);
        outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputMeanGm), shLength);
        outputVarianceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(outputVariance_gm), shLength);
        pipe.InitBuffer(inQueueX, 1, sizeof(dataType) * bshLength);
        pipe.InitBuffer(inQueueGamma, 1, sizeof(dataType) * bLength);
        pipe.InitBuffer(inQueueBeta, 1, sizeof(dataType) * bLength);
        pipe.InitBuffer(outQueue, 1, sizeof(dataType) * bshLength);
        pipe.InitBuffer(outQueueMean, 1, sizeof(dataType) * shLength);
        pipe.InitBuffer(outQueueVariance, 1, sizeof(dataType) * shLength);
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
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.AllocTensor<dataType>();
        AscendC::DataCopy(inputXLocal, inputXGlobal, bshLength);
        AscendC::DataCopy(gammaLocal, gammGlobal, bLength);
        AscendC::DataCopy(betaLocal, betaGlobal, bLength);
        inQueueX.EnQue(inputXLocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<dataType> inputXLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> gammaLocal = inQueueGamma.DeQue<dataType>();
        AscendC::LocalTensor<dataType> betaLocal = inQueueBeta.DeQue<dataType>();
        AscendC::LocalTensor<dataType> outputLocal = outQueue.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> meanLocal = outQueueMean.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> varianceLocal = outQueueVariance.AllocTensor<dataType>();
        AscendC::BatchNorm<dataType, isReuseSource, isBasicBlock>(outputLocal,
            meanLocal,
            varianceLocal,
            inputXLocal,
            gammaLocal,
            betaLocal,
            (dataType)epsilon,
            batchNormTilling);
        outQueue.EnQue<dataType>(outputLocal);
        outQueueMean.EnQue<dataType>(meanLocal);
        outQueueVariance.EnQue<dataType>(varianceLocal);
        inQueueX.FreeTensor(inputXLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dataType> outputLocal = outQueue.DeQue<dataType>();
        AscendC::LocalTensor<dataType> meanLocal = outQueueMean.DeQue<dataType>();
        AscendC::LocalTensor<dataType> varianceLocal = outQueueVariance.DeQue<dataType>();

        AscendC::DataCopy(outputGlobal, outputLocal, bshLength);
        AscendC::DataCopy(outputMeanGlobal, meanLocal, shLength);
        AscendC::DataCopy(outputVarianceGlobal, varianceLocal, shLength);

        outQueue.FreeTensor(outputLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueVariance.FreeTensor(varianceLocal);
    }

private:
    AscendC::GlobalTensor<dataType> inputXGlobal;
    AscendC::GlobalTensor<dataType> gammGlobal;
    AscendC::GlobalTensor<dataType> betaGlobal;
    AscendC::GlobalTensor<dataType> outputGlobal;
    AscendC::GlobalTensor<dataType> outputMeanGlobal;
    AscendC::GlobalTensor<dataType> outputVarianceGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueMean;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueVariance;
    uint32_t bLength;
    uint32_t sLength;
    uint32_t hLength;
    uint32_t originalBLength;
    dataType epsilon = 0.001;
    uint32_t bshLength;
    uint32_t shLength;
    BatchNormTiling batchNormTilling;
};

extern "C" __global__ __aicore__ void kernel_batchnorm_operator(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm,
    GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputVariance_gm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBatchnorm<half, false, false> op;
    op.Init(inputXGm, gammGm, betaGm, outputGm, outputMeanGm, outputVariance_gm, tilingData);
    op.Process();
}
```
